// A: detrend and transform, the host side (gpu/back/DESIGN.md sections 1 to 3). The tables it builds from
// (transformTables, the disc's layout, the rows by version) are disc.mjs's, read from the codec's wasm rather than
// restated.
//
// build(device, { B, tables, precision, blkStride, picture }) compiles the four shaders for every built picture size
// (../glsl/back_transform.mjs: the gate, the fused pass 1, reduce, pass 2) and holds the shared tables; a lane's
// buffers and bind groups and the dispatches are the native host's (core/dec/front.cpp backLane). Buffer names follow
// DESIGN section 2.
// ai: Pass 1 is fused: it samples the frame through the front half's texture, maps and residuals, on the grid of the
// ai: frame's (ring, picture) pair (`picture`: the front half's rings, where each ring puts a picture and its
// ai: lattice). It reads the front half's anchors records too (binding 10: SPEC 6.13, the mesh the anchors give and
// ai: their erase).
import { CPW, ARGS_WORDS, ARGS_SLOT, LATTICE_MAX, argsLdpc, gateRowsAt, gateCodesAt, gateWords } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { CHUNKS } from "../plan/back_tiers.mjs";
import { gridOf } from "../tables.mjs";
import { rowsOfVersion } from "./disc.mjs";
import { gateSource, pass1Source, reduceSource, pass2Source } from "../glsl/back_transform.mjs";
import { shaderModule } from "../pipeline.mjs";

// The usage flags by value, so the module loads in node (the tables and the layout are used there without a device).
const U = globalThis.GPUBufferUsage ?? { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };

// ai: ARGS entry `slot` of picture slot s (pass1, reduce, pass2, soft: ../plan/back_transform.mjs ARGS_SLOT) and code c's
// ai: LDPC dispatch (argsLdpc), in bytes: the rule stage's steps' argsAt (tiers.mjs exportFor), which the native host
// ai: dispatches indirect from (core/dec/front.cpp dispatchRule; its own argsOffset for pass 1, reduce and pass 2).
export const argsOffset = (s, slot) => 4 * (ARGS_WORDS * s + ARGS_SLOT[slot]);
export const ldpcArgsOffset = (c) => 4 * argsLdpc(c);

export class Transform {
  // precision: "f16" stores Y and S and the workgroup exchange as f16 (needs shader-f16 on the device); "f32" otherwise.
  // ai: The gate also zeroes the chain's counters, verdicts, iteration counts and record count (blkStride entries a
  // ai: frame for V and ITS).
  // ai: picture: the rings, [{ span, margin, lattice }] a ring (gpu/tables.mjs formatTables rings: span, margin and
  // ai: nodeX), PIC a size holding each ring's grid for that size (gridOf) and lattice. codes: a word a version, each
  // ai: code's blocks in its rate profile a byte (codes.mjs ruleTables; the gate's codes table, the LDPC's dispatches).
  static async build(device, { B, tables, precision = "f32", blkStride, picture, codes }) {
    if (!codes) throw new Error("the gate sizes the LDPC's dispatches: codes, each version's blocks a code, is required");
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
    // ai: and the anchors' records at binding 10 (read-only: eight storage buffers), then the linear sampler its
    // ai: bilinear reads take (binding 11, ../glsl/common.mjs LUMH; natively the layout's immutable sampler)
    const texEntry = { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d-array" } };
    const fusedTail = [RO, RO, RW, RW, RW, UN, RO, RO, UN, RO].map((type, k) => entry(k + 1, type));
    const smpEntry = { binding: 11, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } };
    t.p1fBgl = d.createBindGroupLayout({ entries: [texEntry, ...fusedTail, smpEntry] });
    t.redBgl = layout([RO, RO, RW, UN]);
    t.p2Bgl = layout([RO, RO, RO, RO, RO, RO, UN, RW, RW, UN]);
    const pipe = (bgl, code, label) => d.createComputePipelineAsync({ label, layout: d.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: shaderModule(d, code, label), entryPoint: "main" } });
    // ai: The gate's rows table: an entry a version up to the largest built size's top.
    const counts = 1 + Math.max(...t.built.map((s) => tables[s].blocks / CHUNKS));
    const jobs = [pipe(t.gateBgl, { glsl: gateSource(B, { blkStride, counts, chunks: CHUNKS }) }, "back gate")];
    t.reduce = []; t.pass2 = []; t.pass1f = [];
    const set = (arr, s) => (p) => { arr[s] = p; };
    for (const s of t.built) {
      // ai: pass 1 at the size's nt with the picture's own n for the anchors' erase
      const { nt: n, n: picture } = tables[s];
      jobs.push(pipe(t.redBgl, { glsl: reduceSource({ n, B }) }, `back reduce ${n}`).then(set(t.reduce, s)));
      jobs.push(pipe(t.p2Bgl, { glsl: pass2Source({ n, prec: precision, B }) }, `back pass2 ${n}`).then(set(t.pass2, s)));
      jobs.push(pipe(t.p1fBgl, { glsl: pass1Source({ n, prec: precision, B, picture }) }, `back pass1 fused ${n}`).then(set(t.pass1f, s)));
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
    // ai: PIC a size (glsl/back_transform.mjs PICTURE): per ring (g0, step, nodes a side, first coordinate) as f32,
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
    // ai: G (plan/back_transform.mjs gateWords): at[s] = (n / CPW, V, the size's top version, 0), then the rows
    // ai: version v takes at each slot (rowsOfVersion).
    const gate = new Uint32Array(gateWords(counts));
    for (const s of t.built) {
      const tb = tables[s];
      gate.set([tb.nt / CPW, tb.V, tb.blocks / CHUNKS, 0], 4 * s);
      for (let v = 0; v < counts; v++) gate[gateRowsAt(v, s)] = rowsOfVersion(tb, v);
    }
    for (let v = 0; v < counts; v++) gate[gateCodesAt(counts, v)] = codes[v] ?? 0;
    t.gateUni = up(gate, U.UNIFORM);
    // Per frame strides, the largest built size's (a batch may mix sizes).
    // ai: Y holds V rows a frame where pass 1 writes V - 1, and the column means ride in PART's w (no Y0 buffer).
    const over = (f) => Math.max(...t.built.map((s) => f(tables[s])));
    t.yStride = over((tb) => tb.V * tb.nt);
    t.sStride = over((tb) => tb.npos);
    t.partStride = over((tb) => tb.nt);
    return t;
  }
}
