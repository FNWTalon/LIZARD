// A: detrend and transform, the host side (gpu/back/DESIGN.md sections 1 to 3, 10). Runs in node and in the browser:
// sim/ob.mjs loads build/ob.wasm in both, and the format's tables are read from it rather than restated.
//
// build(device, { B, tables, precision }) compiles the four shaders for every built picture size and holds the shared
// tables; lane(...) makes one batch's buffers and bind groups over the front half's grid, frames and sel buffers;
// encode(enc, lane, { ts }) records gate, pass 1, reduce and pass 2 as four passes, and the dispatch*() calls append
// the same work to a pass the caller holds open (back.mjs). Buffer names follow DESIGN section 2.
// ai: Built with `picture` (the front half's rings: where each ring puts a picture and its lattice), pass 1 also comes
// ai: fused: it samples the frame through the front half's texture, maps and residuals (bindPicture), on the grid of
// ai: the frame's (ring, picture) pair, and no grid is read.
// ai: Built with `cancel`, either pass 1 keeps its samples as unorm16 in the lane's KEEP (2 MB a frame at 1024), and
// ai: the cancel variant (pass two's pass 1c, DESIGN 13.7) reads them instead of sampling the frame again.
import { init, Focus } from "../../sim/ob.mjs";
import { SIZES, CPW, ARGS_WORDS, ARGS_SLOT, LATTICE_MAX, PICTURES, argsWords, listsWords, gateRowsAt, gateWords, gateSource, pass1Source, reduceSource, pass2Source } from "../wgsl/back_transform.mjs";
import { RING_COUNT } from "../wgsl/finder.mjs";
import { gridOf } from "../tables.mjs";

// The coefficient at pos p, as (u, v): the C's half-plane block layout (src/focus.c init).
export function uvOf(p, n, bw) {
  const v = Math.floor(p / (n * bw)) * bw + ((p % (n * bw)) % bw), uw = Math.floor((p % (n * bw)) / bw);
  return [uw < n / 2 ? uw : uw - n, v];
}

// S's rows (DESIGN section 3): row v holds u = ustart .. umaxP then -umaxN .. -1, ustart 1 on v = 0 and 0 elsewhere,
// packed at off[v]. rows[v] = (off, count, nonnegative entries, umaxN) for the shader; UV[i] is the S entry of pos[i]
// for the soft-value stage. Every entry of pos maps to exactly one entry of S, and S has npos entries.
// ai: UC[i] is pos[i]'s (u, v) itself, u in the high half (signed) and v in the low, for the soft stage's turn of a
// ai: coefficient by the grid's shift (wgsl/back_soft.mjs align).
export function discLayout(shape, pos) {
  const { n, bw, npos } = shape;
  const byV = new Map();
  for (let i = 0; i < npos; i++) { const [u, v] = uvOf(pos[i], n, bw); if (!byV.has(v)) byV.set(v, []); byV.get(v).push(u); }
  const V = Math.max(...byV.keys()) + 1;
  const rows = new Uint32Array(4 * V), off = new Uint32Array(V), np = new Uint32Array(V), un = new Uint32Array(V);
  let count = 0;
  for (let v = 0; v < V; v++) {
    const us = byV.get(v) ?? [], set = new Set(us), ustart = v === 0 ? 1 : 0;
    const maxP = Math.max(ustart - 1, ...us.filter((u) => u >= 0)), maxN = Math.max(0, ...us.filter((u) => u < 0).map((u) => -u));
    let ok = us.length === maxP + 1 - ustart + maxN;
    for (let u = ustart; u <= maxP && ok; u++) ok = set.has(u);
    for (let u = -maxN; u < 0 && ok; u++) ok = set.has(u);
    if (!ok) throw new Error(`disc row v = ${v} of ${n}/${npos / 320} is not u = ${ustart}..${maxP}, -${maxN}..-1`);
    off[v] = count; np[v] = maxP + 1 - ustart; un[v] = maxN;
    rows.set([count, us.length, np[v], maxN], 4 * v);
    count += us.length;
  }
  if (count !== npos) throw new Error(`disc holds ${count} entries for ${npos} coefficients`);
  const UV = new Uint32Array(npos), UC = new Uint32Array(npos);
  for (let i = 0; i < npos; i++) {
    const [u, v] = uvOf(pos[i], n, bw), ustart = v === 0 ? 1 : 0;
    UV[i] = off[v] + (u >= 0 ? u - ustart : np[v] + un[v] + u);
    UC[i] = (((u & 0xffff) << 16) | (v & 0xffff)) >>> 0;
  }
  return { V, rows, off, UV, UC, count };
}

// ai: rowsBy[c], c = 0 .. blocks: the disc rows (1 + the largest v) that the coefficients of a frame's first c blocks
// ai: take, rowsBy[blocks] = V. Block b reads pos[i] for i in 8 b x 320 .. 8 (b + 1) x 320 (sub-channels 8b to
// ai: 8b + 7, 320 coefficients each, as wgsl/back_soft.mjs gathers them), and every version is a prefix of its
// ai: size's largest, so rowsBy[v] is version v's own V (scripts/gpu/back/check_rows.mjs holds every version of every size to it).
export function rowsByCount({ n, bw, blocks, npos }, pos) {
  const per = npos / blocks, rowsBy = new Uint32Array(blocks + 1);
  let top = -1;
  for (let c = 1; c <= blocks; c++) {
    for (let i = per * (c - 1); i < per * c; i++) top = Math.max(top, uvOf(pos[i], n, bw)[1]);
    rowsBy[c] = top + 1;
  }
  return rowsBy;
}

// ai: The tables for every picture size, from the codec: sizes = [{ n, subch, bitmap }] with a size's index its
// ai: position (gpu/tables.mjs PICTURES) and null where a size is not built; bitmap left out is the format's own.
// ai: A picture's tables do not depend on the ring it is painted in, so the codec is set up in its default ring.
// Besides A's own tables each entry carries what B and C bind to: blocks, the bit map mode and the LDPC code. Read
// once a build; the wasm holds one configuration at a time, so this is not called while another Focus is in use.
export async function transformTables(sizes) {
  await init();
  return sizes.map((z) => {
    if (!z) return null;
    const { n, subch } = z;
    if (SIZES.indexOf(n) < 0) throw new Error(`picture size ${n}`);
    // focus_setup puts the format's own bit map in place, so a size that names none gets that (a replay of a
    // recording names its own, 0 before the bit map existed).
    const fc = new Focus(n, subch, 1), bitmap = z.bitmap ?? fc.bitmap;
    // ai: blocks here are 8-sub-channel chunks, the gate's unit: a frame's version, whose rows and soft values a
    // ai: profile does not change (2026-10-07: the format's rate profile, src/focus.h focus_tiers_for, gives a frame
    // ai: fewer blocks than chunks, gpu/back/tiers.mjs); code is the 3/4 code the one-rate stages are built with
    const shape = { ...fc.shape(), blocks: subch / 8 }, pos = fc.posTable(), tab = fc.tables();
    const disc = discLayout(shape, pos), rowsBy = rowsByCount(shape, pos);
    if (rowsBy[shape.blocks] !== disc.V) throw new Error(`${n}/${subch}: the blocks take ${rowsBy[shape.blocks]} rows of the disc's ${disc.V}`);
    const x12 = new Float32Array(4 * n), tw = new Float32Array(2 * n);
    for (let u = 0; u < n; u++) x12.set([tab[2 * n + u], tab[3 * n + u], tab[4 * n + u], tab[5 * n + u]], 4 * u);
    for (let m = 0; m < n; m++) { const th = -2 * Math.PI * m / n; tw[2 * m] = Math.cos(th); tw[2 * m + 1] = Math.sin(th); }
    const sx2 = tab[6 * n], sq2 = tab[6 * n + 1];
    fc.free();
    const one = new Focus(256, 8, 1), code = one.ldpcCode().code;
    one.free();
    return { n, subch, bitmap, blocks: shape.blocks, code, shape, pos, V: disc.V, rowsBy, rows: disc.rows, off: disc.off, UV: disc.UV, UC: disc.UC, npos: shape.npos, x12, tw, sx2, sq2, m2: sx2 / n, cx: tab.slice(0, n), qx: tab.slice(n, 2 * n) };
  });
}

// The usage flags by value, so the module loads in node (the tables and the layout are used there without a device).
const U = globalThis.GPUBufferUsage ?? { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };

export const PASSES = ["gate", "pass1", "reduce", "pass2"];
export const argsOffset = (s, slot) => 4 * (ARGS_WORDS * s + ARGS_SLOT[slot]);

export class Transform {
  // precision: "f16" stores Y and S and the workgroup exchange as f16 (needs shader-f16 on the device); "f32" otherwise.
  // tw: how the butterflies get their twiddles (wgsl/back_transform.mjs laterStages): "chain", "table" or "sincos".
  // grid: the grid buffer's element, "f32" (the sampler's today) or "f16" (two an u32).
  // zero: the gate also zeroes the chain's counters, verdicts, iteration counts and record count (blkStride entries
  // a frame for V and ITS); the lane then binds the gate after those buffers exist (bindGate).
  // ai: picture: the rings, [{ span, margin, lattice }] a ring (gpu/tables.mjs formatTables rings: span, margin and
  // ai: nodeX, what F9 samples by); given, the fused pass 1 is built beside the grid one, its PIC a size holding each
  // ai: ring's grid for that size (gridOf) and lattice.
  // ai: cancel: { refStride, refMeans } (bytes, dims.mjs derived) builds pass 1's cancel variant (wgsl/
  // ai: back_transform.mjs pass1Source cancel): the same pass over the cancel stage's second lists with REF and
  // ai: CANCEL bound at 10 and 11, dispatched by dispatchPass1(p, ln, picture, true). It reads the samples pass 1
  // ai: kept (KEEP, the lane's own unorm16 buffer, written by either picture variant at binding 10), so the picture
  // ai: is sampled once a batch and the variant is one, whichever picture the batch used.
  static async build(device, { B, tables, precision = "f32", cpw = CPW, tw = "chain", grid = "f32", zero = false, blkStride = 0, picture = null, cancel = null }) {
    if (precision === "f16" && !device.features.has("shader-f16")) throw new Error("precision f16 without shader-f16");
    const t = new Transform();
    Object.assign(t, { device, B, tables, precision, cpw, tw, grid, zero, blkStride, cbytes: precision === "f16" ? 4 : 8, fusable: !!picture, cancel: cancel ? { refSlot: cancel.refStride / 8, refMeans: cancel.refMeans / 8 } : null });
    t.built = tables.map((tb, s) => (tb ? s : -1)).filter((s) => s >= 0);
    if (!t.built.length) throw new Error("no size built");
    // ai: The cancel variant reads REF's three-base record (24 B a quad a row, the means behind 6 nmax^2 bytes): a
    // ai: REF laid out otherwise would fit nothing (a timing once read 0 gained that way), so the build refuses it.
    const nmax = Math.max(...t.built.map((s) => tables[s].n));
    if (cancel && cancel.refMeans !== 6 * nmax * nmax) throw new Error(`REF means at ${cancel.refMeans} B with nmax ${nmax}: the cancel variant reads three bases a reference`);
    const d = device;
    const RO = "read-only-storage", RW = "storage", UN = "uniform";
    const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = (types) => d.createBindGroupLayout({ entries: types.map((type, binding) => entry(binding, type)) });
    t.gateBgl = layout(zero ? [RO, RO, RW, RW, UN, RW, RW, RW, RW] : [RO, RO, RW, RW, UN]);
    // ai: Built with cancel, both picture variants keep their samples at binding 10 (the fused pass's eighth
    // ai: storage buffer).
    const keepTail = cancel ? [entry(10, RW)] : [];
    const gridHead = [RO, RO, RO, RW, RW, RW, UN].map((type, binding) => entry(binding, type));
    t.p1Bgl = d.createBindGroupLayout({ entries: [...gridHead, ...keepTail] });
    // The fused pass 1: the frame texture where the grid was, the same six after it, then maps, resid and PIC.
    const texEntry = { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d-array" } };
    const fusedTail = [RO, RO, RW, RW, RW, UN, RO, RO, UN].map((type, k) => entry(k + 1, type));
    if (picture) t.p1fBgl = d.createBindGroupLayout({ entries: [texEntry, ...fusedTail, ...keepTail] });
    // ai: The cancel variant: KEEP read-only at 0 where the grid was, then REF (read-only) at 10 and CANCEL
    // ai: (uniform) at 11.
    if (cancel) t.p1cBgl = d.createBindGroupLayout({ entries: [...gridHead, entry(10, RO), entry(11, UN)] });
    t.redBgl = layout([RO, RO, RW, UN]);
    t.p2Bgl = layout([RO, RO, RO, RO, RO, RO, UN, RW, RW, UN]);
    const pipe = (bgl, code, label) => d.createComputePipelineAsync({ label, layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: d.createShaderModule({ label, code }), entryPoint: "main" } });
    // ai: The gate's rows table: an entry a block count up to the largest built size's.
    const counts = 1 + Math.max(...t.built.map((s) => tables[s].blocks));
    const jobs = [pipe(t.gateBgl, gateSource(B, { zero, blkStride, counts }), "back gate")];
    t.pass1 = []; t.reduce = []; t.pass2 = []; t.pass1f = []; t.pass1c = [];
    const set = (arr, s) => (p) => { arr[s] = p; };
    for (const s of t.built) {
      const { n } = tables[s], common = { n, prec: precision, B, cpw, tw, keep: !!cancel };
      jobs.push(pipe(t.p1Bgl, pass1Source({ ...common, grid }), `back pass1 ${n}`).then(set(t.pass1, s)));
      jobs.push(pipe(t.redBgl, reduceSource({ n, B }), `back reduce ${n}`).then(set(t.reduce, s)));
      jobs.push(pipe(t.p2Bgl, pass2Source({ n, prec: precision, B, tw }), `back pass2 ${n}`).then(set(t.pass2, s)));
      if (picture) jobs.push(pipe(t.p1fBgl, pass1Source({ ...common, picture: "fused" }), `back pass1 fused ${n}`).then(set(t.pass1f, s)));
      if (cancel) jobs.push(pipe(t.p1cBgl, pass1Source({ ...common, keep: false, cancel: true, ...t.cancel }), `back pass1 cancel ${n}`).then(set(t.pass1c, s)));
    }
    const done = await Promise.all(jobs);
    t.gate = done[0];
    // Shared tables, uploaded once.
    const up = (arr, usage = U.STORAGE) => { const b = d.createBuffer({ size: Math.max(16, arr.byteLength), usage: usage | U.COPY_DST }); d.queue.writeBuffer(b, 0, arr); return b; };
    t.twBufs = []; t.x12Bufs = []; t.rowsBufs = [];
    for (const s of t.built) {
      const tb = tables[s];
      t.twBufs[s] = up(tb.tw);
      t.x12Bufs[s] = up(tb.x12);
      if (tb.V > 512) throw new Error(`V = ${tb.V} rows at n = ${tb.n}: the rows uniform holds 512`);
      const r = new Uint32Array(2048); r.set(tb.rows); t.rowsBufs[s] = up(r, U.UNIFORM);
    }
    // ai: PIC a size (wgsl/back_transform.mjs PICTURE): per ring (g0, step, nodes a side, first coordinate) as f32,
    // ai: then every ring's lattice coordinates, LATTICE_MAX a ring.
    t.picUnis = [];
    if (picture && picture.length !== RING_COUNT) throw new Error(`${picture.length} rings: PIC holds ${RING_COUNT}`);
    if (picture) for (const s of t.built) {
      const ab = new ArrayBuffer(16 * RING_COUNT + 4 * RING_COUNT * LATTICE_MAX), fl = new Float32Array(ab);
      picture.forEach((rg, r) => {
        if (rg.lattice.length > LATTICE_MAX || rg.lattice.length < 2) throw new Error(`${rg.lattice.length} lattice nodes a side in ring ${r}: PIC holds 2 to ${LATTICE_MAX}`);
        const { g0, step } = gridOf(rg, tables[s].n);
        fl.set([g0, step, rg.lattice.length, r * LATTICE_MAX], 4 * r);
        fl.set(rg.lattice, 4 * RING_COUNT + r * LATTICE_MAX);
      });
      t.picUnis[s] = up(new Uint8Array(ab), U.UNIFORM);
    }
    // ai: G (wgsl/back_transform.mjs gateWords): at[s] = (n / cpw, V, blocks, 0), then the rows c blocks take at
    // ai: each slot (rowsBy).
    const gate = new Uint32Array(gateWords(counts));
    for (const s of t.built) {
      const tb = tables[s];
      gate.set([tb.n / cpw, tb.V, tb.blocks, 0], 4 * s);
      for (let c = 0; c < counts; c++) gate[gateRowsAt(c, s)] = tb.rowsBy[Math.min(c, tb.blocks)];
    }
    t.gateUni = up(gate, U.UNIFORM);
    // Per frame strides, the largest built size's (a batch may mix sizes).
    // ai: Y holds V rows a frame where pass 1 writes V - 1: the cancel stage's inverse rows use a frame's slot as
    // ai: scratch for a whole disc (rows 0 to V - 1), and the column means ride in PART's w (no Y0 buffer).
    const over = (f) => Math.max(...t.built.map((s) => f(tables[s])));
    t.nmax = nmax;
    t.yStride = over((tb) => tb.V * tb.n);
    t.sStride = over((tb) => tb.npos);
    t.partStride = t.nmax;
    // ai: KEEP: a frame's samples as unorm16, vec2u a quad of columns a row (nmax^2 / 4 a frame, 2 MB at 1024).
    t.sampStride = (t.nmax * t.nmax) / 4;
    return t;
  }

  // One batch's buffers over the front half's lane buffers: gridBuf (gridStride elements a frame, strip order; null
  // where only the fused pass 1 runs, or until bindGrid), framesBuf (Frame structs) and selBuf (vec4u a frame).
  // Returns the lane with its device bytes counted. With zero the gate's bind group waits for bindGate(ln) once ln.V,
  // ln.ITS and ln.REC exist; the fused pass 1's wait for bindPicture(ln, ...).
  lane({ gridBuf = null, gridStride, framesBuf, selBuf }) {
    const d = this.device, B = this.B, cb = this.cbytes;
    const ln = { gridStride, gridBuf: null, framesBuf, selBuf, bytes: 0 };
    const mk = (size, usage) => { const b = d.createBuffer({ size, usage }); ln.bytes += size; return b; };
    ln.listsBuf = mk(4 * listsWords(B), U.STORAGE | U.COPY_SRC);
    ln.argsBuf = mk(4 * argsWords, U.STORAGE | U.INDIRECT | U.COPY_SRC);
    ln.partBuf = mk(16 * this.partStride * B, U.STORAGE | U.COPY_SRC);
    ln.coefBuf = mk(32 * B, U.STORAGE | U.COPY_SRC);
    ln.yBuf = mk(cb * this.yStride * B, U.STORAGE | U.COPY_SRC);
    ln.sBuf = mk(cb * this.sStride * B, U.STORAGE | U.COPY_SRC);
    ln.countsBuf = mk(4 * 8 * B, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    // ai: The samples pass 1 keeps for the cancel variant (B x 2 MB at 1024), counted here so the Batcher prices it.
    ln.keepBuf = this.cancel ? mk(8 * this.sampStride * B, U.STORAGE | U.COPY_SRC) : null;
    ln.params = [];
    for (const s of this.built) {
      const tb = this.tables[s], b = mk(48, U.UNIFORM | U.COPY_DST);
      const u = new Uint32Array(12), f = new Float32Array(u.buffer);
      u.set([gridStride, this.yStride, this.partStride, s, tb.n, 0]);
      f[6] = tb.m2; f[7] = tb.sx2; f[8] = tb.sq2; u[9] = this.sStride; u[10] = this.sampStride;
      d.queue.writeBuffer(b, 0, u);
      ln.params[s] = b;
    }
    const group = (bgl, bufs) => d.createBindGroup({ layout: bgl, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.gateGroup = this.zero ? null : group(this.gateBgl, [framesBuf, selBuf, ln.listsBuf, ln.argsBuf, this.gateUni]);
    ln.p1Groups = null; ln.p1fGroups = null; ln.redGroups = []; ln.p2Groups = [];
    for (const s of this.built) {
      ln.redGroups[s] = group(this.redBgl, [ln.listsBuf, ln.partBuf, ln.coefBuf, ln.params[s]]);
      ln.p2Groups[s] = group(this.p2Bgl, [ln.listsBuf, this.twBufs[s], ln.partBuf, ln.yBuf, ln.coefBuf, this.x12Bufs[s], this.rowsBufs[s], ln.sBuf, ln.countsBuf, ln.params[s]]);
    }
    if (gridBuf) this.bindGrid(ln, gridBuf);
    return ln;
  }

  // The grid pass 1's bind groups over gridBuf (gridStride elements a frame, as the lane was made with).
  // ai: With cancel, KEEP at 10.
  bindGrid(ln, gridBuf) {
    const d = this.device, keep = this.cancel ? [[10, ln.keepBuf]] : [];
    ln.gridBuf = gridBuf;
    ln.p1Groups = [];
    for (const s of this.built) {
      const at = [gridBuf, ln.listsBuf, this.twBufs[s], ln.partBuf, ln.yBuf, ln.countsBuf, ln.params[s]].map((buffer, binding) => [binding, buffer]);
      ln.p1Groups[s] = d.createBindGroup({ layout: this.p1Bgl, entries: [...at, ...keep].map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
    }
  }

  // The fused pass 1's bind groups over the front half's frame texture (a 2d-array view, a frame a layer), maps (16
  // f32 a frame) and residuals (NODES_MAX vec4f a frame); made again whenever the front half replaces the texture.
  // ai: With cancel, KEEP at 10.
  bindPicture(ln, { texView, mapsBuf, residBuf }) {
    if (!this.fusable) throw new Error("the transform was built without picture tables: no fused pass 1");
    const d = this.device, keep = this.cancel ? [[10, ln.keepBuf]] : [];
    ln.p1fGroups = [];
    for (const s of this.built) {
      const at = [ln.listsBuf, this.twBufs[s], ln.partBuf, ln.yBuf, ln.countsBuf, ln.params[s], mapsBuf, residBuf, this.picUnis[s]].map((buffer, k) => [k + 1, buffer]);
      ln.p1fGroups[s] = d.createBindGroup({ layout: this.p1fBgl, entries: [{ binding: 0, resource: texView }, ...[...at, ...keep].map(([binding, buffer]) => ({ binding, resource: { buffer } }))] });
    }
    ln.picture = { texView, mapsBuf, residBuf };
  }

  // ai: The second bind groups (reduce and pass 2 over the cancel stage's LISTS2, counting into BCOUNTS2), once
  // ai: the lane holds them (cancel.mjs lane), and the cancel variant's pass 1 group over the lane's kept samples,
  // ai: REF and CANCEL: nothing of the batch's picture, so no rebind when a texture or grid is bound later.
  bindSecond(ln) {
    const d = this.device;
    const group = (bgl, bufs) => d.createBindGroup({ layout: bgl, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.redGroups2 = []; ln.p2Groups2 = []; ln.p1cGroups = [];
    for (const s of this.built) {
      ln.redGroups2[s] = group(this.redBgl, [ln.LISTS2, ln.partBuf, ln.coefBuf, ln.params[s]]);
      ln.p2Groups2[s] = group(this.p2Bgl, [ln.LISTS2, this.twBufs[s], ln.partBuf, ln.yBuf, ln.coefBuf, this.x12Bufs[s], this.rowsBufs[s], ln.sBuf, ln.BCOUNTS2, ln.params[s]]);
      const at = [[0, ln.keepBuf], [1, ln.LISTS2], [2, this.twBufs[s]], [3, ln.partBuf], [4, ln.yBuf], [5, ln.BCOUNTS2], [6, ln.params[s]], [10, ln.REF], [11, ln.CANCEL]];
      ln.p1cGroups[s] = d.createBindGroup({ layout: this.p1cBgl, entries: at.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
    }
  }

  // The zeroing gate's bind group, once the lane holds the LDPC's V, ITS and REC.
  bindGate(ln) {
    if (!this.zero) return ln.gateGroup;
    ln.gateGroup = this.device.createBindGroup({ layout: this.gateBgl, entries: [ln.framesBuf, ln.selBuf, ln.listsBuf, ln.argsBuf, this.gateUni, ln.countsBuf, ln.V, ln.ITS, ln.REC].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return ln.gateGroup;
  }

  // The four stages, each appended to an open compute pass.
  dispatchGate(p, ln) { p.setPipeline(this.gate); p.setBindGroup(0, ln.gateGroup); p.dispatchWorkgroups(1); }
  // picture: "fused" samples the frame itself (bindPicture), "grid" reads the grid (bindGrid).
  // ai: cancel: the cancel variant (the subtraction at the sample, wgsl/back_transform.mjs pass1Source cancel) over
  // ai: the second lists and args (LISTS2, ARGS2: the short frames pass two runs), reading the samples the batch's
  // ai: pass 1 kept, whichever picture that was.
  dispatchPass1(p, ln, picture = "grid", cancel = false) {
    if (!PICTURES.includes(picture)) throw new Error(`pass 1 picture ${picture}`);
    if (cancel && !this.cancel) throw new Error("the transform was built without the cancel variant");
    const fused = picture === "fused";
    const pipes = cancel ? this.pass1c : fused ? this.pass1f : this.pass1;
    const groups = cancel ? ln.p1cGroups : fused ? ln.p1fGroups : ln.p1Groups;
    if (!groups) throw new Error(cancel ? "the cancel variant before bindSecond" : fused ? "fused pass 1 before bindPicture" : "grid pass 1 with no grid bound");
    const args = cancel ? ln.ARGS2 : ln.argsBuf;
    for (const s of this.built) { p.setPipeline(pipes[s]); p.setBindGroup(0, groups[s]); p.dispatchWorkgroupsIndirect(args, argsOffset(s, "pass1")); }
  }
  // ai: second: over the cancel stage's lists and args (bindSecond).
  dispatchReduce(p, ln, second = false) {
    const groups = second ? ln.redGroups2 : ln.redGroups, args = second ? ln.ARGS2 : ln.argsBuf;
    for (const s of this.built) { p.setPipeline(this.reduce[s]); p.setBindGroup(0, groups[s]); p.dispatchWorkgroupsIndirect(args, argsOffset(s, "reduce")); }
  }
  dispatchPass2(p, ln, second = false) {
    const groups = second ? ln.p2Groups2 : ln.p2Groups, args = second ? ln.ARGS2 : ln.argsBuf;
    for (const s of this.built) { p.setPipeline(this.pass2[s]); p.setBindGroup(0, groups[s]); p.dispatchWorkgroupsIndirect(args, argsOffset(s, "pass2")); }
  }

  // Records the four passes. ts = { qs, base }: a timestamp pair a pass at qs[base + 2k], k = 0..3.
  encode(enc, ln, { ts = null, fused = false } = {}) {
    const tw = (k) => (ts ? { timestampWrites: { querySet: ts.qs, beginningOfPassWriteIndex: ts.base + 2 * k, endOfPassWriteIndex: ts.base + 2 * k + 1 } } : {});
    if (!this.zero) enc.clearBuffer(ln.countsBuf);
    const steps = [(p) => this.dispatchGate(p, ln), (p) => this.dispatchPass1(p, ln, fused ? "fused" : "grid"), (p) => this.dispatchReduce(p, ln), (p) => this.dispatchPass2(p, ln)];
    steps.forEach((step, k) => { const p = enc.beginComputePass(tw(k)); step(p); p.end(); });
  }
}
