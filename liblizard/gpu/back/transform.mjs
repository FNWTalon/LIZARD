// A: detrend and transform, the host side (gpu/back/DESIGN.md sections 1 to 3). Runs in node and in the browser:
// sim/ob.mjs loads the codec's wasm (liblizard/build/ob.mjs) in both, and the format's tables are read from it rather
// than restated.
//
// build(device, { B, tables, precision, blkStride, picture }) compiles the four shaders for every built picture size
// and holds the shared tables; lane(...) makes one batch's buffers and bind groups over the front half's frames and
// sel buffers; the dispatch*() calls append gate, pass 1, reduce and pass 2 to a pass the caller holds open (back.mjs).
// Buffer names follow DESIGN section 2.
// ai: Pass 1 is fused: it samples the frame through the front half's texture, maps and residuals (bindPicture), on the
// ai: grid of the frame's (ring, picture) pair (`picture`: the front half's rings, where each ring puts a picture and
// ai: its lattice). The grid pass 1, which read F9's sampled grid, went 2026-10-22.
// ai: It reads the front half's anchors records too (binding 10: SPEC 6.13, the mesh the anchors give and their erase).
import { init, Focus } from "../../sim/ob.mjs";
import { SIZES, CPW, ARGS_WORDS, ARGS_SLOT, LATTICE_MAX, argsWords, listsWords, gateRowsAt, gateWords, gateSource, pass1Source, reduceSource, pass2Source } from "../wgsl/back_transform.mjs";
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
// ai: coefficient by the grid's shift (wgsl/back_tiers.mjs talign).
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
// ai: 8b + 7, 320 coefficients each, as wgsl/back_tiers.mjs tsoft gathers them), and every version is a prefix of its
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

// ai: The transform size a picture of n samples is read at by the fused pass 1: the ladder size below n, 192 below
// ai: 256 (2026-10-21). n is the first ladder size of at least 3 R, so the one below it holds 2 R and every coefficient
// ai: the picture carries; sampled on the picture's own square from its first sample, at n / nt of its step, a picture
// ai: band-limited to R gives the same coefficients at nt (each pass divides by its own size).
export const TRANSFORM_DOWN = { 256: 192, 384: 256, 512: 384, 768: 512, 1024: 768, 1536: 1024 };

// ai: The detrend's basis and its transforms at m points, as src/focus.c init defines them (in double, then f32):
// ai: cx, qx, then x12 (x1r, x1i, x2r, x2i a u), sx2 and sq2.
function detrendTables(m) {
  const cx = new Float32Array(m), qx = new Float32Array(m), x12 = new Float32Array(4 * m), er = new Float64Array(m), ei = new Float64Array(m);
  let m2 = 0, sq2 = 0;
  for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m; m2 += c * c; er[i] = Math.cos(-2 * 3.14159265358979323846 * i / m); ei[i] = Math.sin(-2 * 3.14159265358979323846 * i / m); }
  const sx2 = m2;
  m2 /= m;
  for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m, q = c * c - m2; cx[i] = c; qx[i] = q; sq2 += q * q; }
  for (let u = 0; u < m; u++) {
    let ar = 0, ai = 0, br = 0, bi = 0;
    for (let i = 0; i < m; i++) { const c = (2 * i + 1 - m) / m, q = c * c - m2, k = (u * i) % m; ar += c * er[k]; ai += c * ei[k]; br += q * er[k]; bi += q * ei[k]; }
    x12.set([ar, ai, br, bi], 4 * u);
  }
  return { cx, qx, x12, sx2: Math.fround(sx2), sq2: Math.fround(sq2) };
}

// ai: The tables for every picture size, from the codec: sizes = [{ n, subch, nt }] with a size's index its position
// ai: (gpu/tables.mjs PICTURES) and null where a size is not built. A picture's tables do not depend on the ring it
// ai: is painted in, so the codec is set up in its default ring. Besides A's own tables each entry carries what the
// ai: soft values bind to (UV, UC) and its blocks. Read once a build; the wasm holds one configuration at a time, so
// ai: this is not called while another Focus is in use.
export async function transformTables(sizes) {
  await init();
  return sizes.map((z) => {
    if (!z) return null;
    const { n, subch } = z, nt = z.nt ?? n;
    if (SIZES.indexOf(n) < 0) throw new Error(`picture size ${n}`);
    const fc = new Focus(n, subch, 1);
    // ai: blocks here are 8-sub-channel chunks, the gate's unit: a frame's version, whose rows and soft values a
    // ai: profile does not change (2026-10-07: the format's rate profile, src/focus.h focus_tiers_for, gives a frame
    // ai: fewer blocks than chunks, gpu/back/tiers.mjs)
    const shape = { ...fc.shape(), blocks: subch / 8 }, pos = fc.posTable(), tab = fc.tables();
    const disc = discLayout(shape, pos), rowsBy = rowsByCount(shape, pos);
    if (rowsBy[shape.blocks] !== disc.V) throw new Error(`${n}/${subch}: the blocks take ${rowsBy[shape.blocks]} rows of the disc's ${disc.V}`);
    // ai: the transform's tables at nt: the codec's own where nt is n, else the same definitions at nt
    let x12 = new Float32Array(4 * n), sx2 = tab[6 * n], sq2 = tab[6 * n + 1], cx = tab.slice(0, n), qx = tab.slice(n, 2 * n);
    for (let u = 0; u < n; u++) x12.set([tab[2 * n + u], tab[3 * n + u], tab[4 * n + u], tab[5 * n + u]], 4 * u);
    if (nt !== n) ({ x12, sx2, sq2, cx, qx } = detrendTables(nt));
    const tw = new Float32Array(2 * nt);
    for (let m = 0; m < nt; m++) { const th = -2 * Math.PI * m / nt; tw[2 * m] = Math.cos(th); tw[2 * m + 1] = Math.sin(th); }
    fc.free();
    return { n, subch, blocks: shape.blocks, shape, pos, V: disc.V, rowsBy, rows: disc.rows, off: disc.off, UV: disc.UV, UC: disc.UC, npos: shape.npos, x12, tw, sx2, sq2, m2: sx2 / nt, cx, qx, nt };
  });
}

// The usage flags by value, so the module loads in node (the tables and the layout are used there without a device).
const U = globalThis.GPUBufferUsage ?? { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };

export const argsOffset = (s, slot) => 4 * (ARGS_WORDS * s + ARGS_SLOT[slot]);

export class Transform {
  // precision: "f16" stores Y and S and the workgroup exchange as f16 (needs shader-f16 on the device); "f32" otherwise.
  // ai: The gate also zeroes the chain's counters, verdicts, iteration counts and record count (blkStride entries a
  // ai: frame for V and ITS); the lane binds the gate after those buffers exist (bindGate).
  // ai: picture: the rings, [{ span, margin, lattice }] a ring (gpu/tables.mjs formatTables rings: span, margin and
  // ai: nodeX), PIC a size holding each ring's grid for that size (gridOf) and lattice.
  static async build(device, { B, tables, precision = "f32", blkStride, picture }) {
    if (precision === "f16" && !device.features.has("shader-f16")) throw new Error("precision f16 without shader-f16");
    if (!picture) throw new Error("the transform samples the frame itself: picture, the front half's rings, is required");
    if (!(blkStride > 0)) throw new Error(`blkStride ${blkStride}: the blocks a frame the gate zeroes`);
    const t = new Transform();
    Object.assign(t, { device, B, tables, precision, blkStride, cbytes: precision === "f16" ? 4 : 8 });
    t.built = tables.map((tb, s) => (tb ? s : -1)).filter((s) => s >= 0);
    if (!t.built.length) throw new Error("no size built");
    const d = device;
    const RO = "read-only-storage", RW = "storage", UN = "uniform";
    const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = (types) => d.createBindGroupLayout({ entries: types.map((type, binding) => entry(binding, type)) });
    t.gateBgl = layout([RO, RO, RW, RW, UN, RW, RW, RW, RW]);
    // The fused pass 1: the frame texture, lists, twiddles, PART, Y, counters and params, then maps, resid and PIC.
    // ai: and the anchors' records at binding 10 (read-only: eight storage buffers)
    const texEntry = { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d-array" } };
    const fusedTail = [RO, RO, RW, RW, RW, UN, RO, RO, UN, RO].map((type, k) => entry(k + 1, type));
    t.p1fBgl = d.createBindGroupLayout({ entries: [texEntry, ...fusedTail] });
    t.redBgl = layout([RO, RO, RW, UN]);
    t.p2Bgl = layout([RO, RO, RO, RO, RO, RO, UN, RW, RW, UN]);
    const pipe = (bgl, code, label) => d.createComputePipelineAsync({ label, layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: d.createShaderModule({ label, code }), entryPoint: "main" } });
    t.pipe = pipe;
    // ai: The gate's rows table: an entry a block count up to the largest built size's.
    const counts = 1 + Math.max(...t.built.map((s) => tables[s].blocks));
    const jobs = [pipe(t.gateBgl, gateSource(B, { blkStride, counts }), "back gate")];
    t.reduce = []; t.pass2 = []; t.pass1f = [];
    const set = (arr, s) => (p) => { arr[s] = p; };
    for (const s of t.built) {
      // ai: pass 1 at the size's nt with the picture's own n for the anchors' erase
      const { nt: n, n: picture } = tables[s];
      jobs.push(pipe(t.redBgl, reduceSource({ n, B }), `back reduce ${n}`).then(set(t.reduce, s)));
      jobs.push(pipe(t.p2Bgl, pass2Source({ n, prec: precision, B }), `back pass2 ${n}`).then(set(t.pass2, s)));
      jobs.push(pipe(t.p1fBgl, pass1Source({ n, prec: precision, B, picture }), `back pass1 fused ${n}`).then(set(t.pass1f, s)));
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
    if (picture.length !== RING_COUNT) throw new Error(`${picture.length} rings: PIC holds ${RING_COUNT}`);
    for (const s of t.built) {
      const ab = new ArrayBuffer(16 * RING_COUNT + 4 * RING_COUNT * LATTICE_MAX), fl = new Float32Array(ab);
      picture.forEach((rg, r) => {
        if (rg.lattice.length > LATTICE_MAX || rg.lattice.length < 2) throw new Error(`${rg.lattice.length} lattice nodes a side in ring ${r}: PIC holds 2 to ${LATTICE_MAX}`);
        // ai: the picture's own grid, from its first sample at n / nt of its step where it is read at nt
        const { g0, step } = gridOf(rg, tables[s].n);
        fl.set([g0, (step * tables[s].n) / tables[s].nt, rg.lattice.length, r * LATTICE_MAX], 4 * r);
        fl.set(rg.lattice, 4 * RING_COUNT + r * LATTICE_MAX);
      });
      t.picUnis[s] = up(new Uint8Array(ab), U.UNIFORM);
    }
    // ai: G (wgsl/back_transform.mjs gateWords): at[s] = (n / CPW, V, blocks, 0), then the rows c blocks take at
    // ai: each slot (rowsBy).
    const gate = new Uint32Array(gateWords(counts));
    for (const s of t.built) {
      const tb = tables[s];
      gate.set([tb.nt / CPW, tb.V, tb.blocks, 0], 4 * s);
      for (let c = 0; c < counts; c++) gate[gateRowsAt(c, s)] = tb.rowsBy[Math.min(c, tb.blocks)];
    }
    t.gateUni = up(gate, U.UNIFORM);
    // Per frame strides, the largest built size's (a batch may mix sizes).
    // ai: Y holds V rows a frame where pass 1 writes V - 1, and the column means ride in PART's w (no Y0 buffer).
    const over = (f) => Math.max(...t.built.map((s) => f(tables[s])));
    t.yStride = over((tb) => tb.V * tb.nt);
    t.sStride = over((tb) => tb.npos);
    t.partStride = over((tb) => tb.nt);
    return t;
  }

  // One batch's buffers over the front half's lane buffers: framesBuf (Frame structs) and selBuf (vec4u a frame).
  // Returns the lane with its device bytes counted. The gate's bind group waits for bindGate(ln) once ln.V, ln.ITS
  // and ln.REC exist; the fused pass 1's for bindPicture(ln, ...).
  lane({ framesBuf, selBuf }) {
    const d = this.device, B = this.B, cb = this.cbytes;
    const ln = { framesBuf, selBuf, bytes: 0 };
    const mk = (size, usage) => { const b = d.createBuffer({ size, usage }); ln.bytes += size; return b; };
    ln.listsBuf = mk(4 * listsWords(B), U.STORAGE | U.COPY_SRC);
    ln.argsBuf = mk(4 * argsWords, U.STORAGE | U.INDIRECT | U.COPY_SRC);
    ln.partBuf = mk(16 * this.partStride * B, U.STORAGE | U.COPY_SRC);
    ln.coefBuf = mk(32 * B, U.STORAGE | U.COPY_SRC);
    ln.yBuf = mk(cb * this.yStride * B, U.STORAGE | U.COPY_SRC);
    ln.sBuf = mk(cb * this.sStride * B, U.STORAGE | U.COPY_SRC);
    ln.countsBuf = mk(4 * 8 * B, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.params = [];
    for (const s of this.built) {
      const tb = this.tables[s], b = mk(48, U.UNIFORM | U.COPY_DST);
      const u = new Uint32Array(12), f = new Float32Array(u.buffer);
      u.set([0, this.yStride, this.partStride, s, 0, 0]);
      f[6] = tb.m2; f[7] = tb.sx2; f[8] = tb.sq2; u[9] = this.sStride;
      d.queue.writeBuffer(b, 0, u);
      ln.params[s] = b;
    }
    const group = (bgl, bufs) => d.createBindGroup({ layout: bgl, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.gateGroup = null;
    ln.p1fGroups = null; ln.redGroups = []; ln.p2Groups = [];
    for (const s of this.built) {
      ln.redGroups[s] = group(this.redBgl, [ln.listsBuf, ln.partBuf, ln.coefBuf, ln.params[s]]);
      ln.p2Groups[s] = group(this.p2Bgl, [ln.listsBuf, this.twBufs[s], ln.partBuf, ln.yBuf, ln.coefBuf, this.x12Bufs[s], this.rowsBufs[s], ln.sBuf, ln.countsBuf, ln.params[s]]);
    }
    return ln;
  }

  // The fused pass 1's bind groups over the front half's frame texture (a 2d-array view, a frame a layer), maps (16
  // f32 a frame) and residuals (NODES_MAX vec4f a frame); made again whenever the front half replaces the texture.
  // ai: arecBuf: the front half's anchors records (ANCH_REC vec4f a frame), binding 10.
  bindPicture(ln, { texView, mapsBuf, residBuf, arecBuf }) {
    const d = this.device;
    if (!arecBuf) throw new Error("pass 1 bound with no anchors records");
    ln.p1fGroups = [];
    for (const s of this.built) {
      const at = [ln.listsBuf, this.twBufs[s], ln.partBuf, ln.yBuf, ln.countsBuf, ln.params[s], mapsBuf, residBuf, this.picUnis[s], arecBuf].map((buffer, k) => [k + 1, buffer]);
      ln.p1fGroups[s] = d.createBindGroup({ layout: this.p1fBgl, entries: [{ binding: 0, resource: texView }, ...at.map(([binding, buffer]) => ({ binding, resource: { buffer } }))] });
    }
    ln.picture = { texView, mapsBuf, residBuf, arecBuf };
  }

  // The zeroing gate's bind group, once the lane holds the LDPC's V, ITS and REC.
  bindGate(ln) {
    ln.gateGroup = this.device.createBindGroup({ layout: this.gateBgl, entries: [ln.framesBuf, ln.selBuf, ln.listsBuf, ln.argsBuf, this.gateUni, ln.countsBuf, ln.V, ln.ITS, ln.REC].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return ln.gateGroup;
  }

  // The four stages, each appended to an open compute pass.
  dispatchGate(p, ln) { p.setPipeline(this.gate); p.setBindGroup(0, ln.gateGroup); p.dispatchWorkgroups(1); }
  dispatchPass1(p, ln) {
    if (!ln.p1fGroups) throw new Error("pass 1 before bindPicture");
    for (const s of this.built) { p.setPipeline(this.pass1f[s]); p.setBindGroup(0, ln.p1fGroups[s]); p.dispatchWorkgroupsIndirect(ln.argsBuf, argsOffset(s, "pass1")); }
  }
  dispatchReduce(p, ln) {
    for (const s of this.built) { p.setPipeline(this.reduce[s]); p.setBindGroup(0, ln.redGroups[s]); p.dispatchWorkgroupsIndirect(ln.argsBuf, argsOffset(s, "reduce")); }
  }
  dispatchPass2(p, ln) {
    for (const s of this.built) { p.setPipeline(this.pass2[s]); p.setBindGroup(0, ln.p2Groups[s]); p.dispatchWorkgroupsIndirect(ln.argsBuf, argsOffset(s, "pass2")); }
  }
}
