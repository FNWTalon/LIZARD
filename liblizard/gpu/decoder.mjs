// The GPU decoder's setup: the device objects every native host builds (../core/dec/: setup.cpp builds them, front.cpp
// runs the batches), made here by FrontHalf.create on the recording device of ../gen/mockgpu.mjs and written out by
// ../gen/gen.mjs. Every stage is GLSL (glsl/), its constants, tables and plans plain JS (plan/).
//
// A batch is up to B frames, each a layer of one texture. Everything a batch does is encoded into one command
// buffer and comes back in one readback; nothing in between waits on the host. Frames in a batch may differ in
// size and in picture size, so every per-frame fact lives in a buffer the shaders index by frame. create() makes
// what a lane and a batch read and never write: the pipelines and their bind group layouts, the format's tables,
// the weights, the back half at every batch size the host may need. What a lane holds, the batch controller, the
// ring and a batch's recording are the native host's alone (core/dec/front.cpp).
// ai: The WebGPU run-time half that ran these on a browser's device (lanes, ensure, plan, the Batcher, the ring,
// ai: run and batch, the int8 twins' measurement, the self-test's probe) and the WGSL twins of every stage are in
// ai: archive/webgpu-decoder/ (2026-10-31): the native host is its port, name for name.
import { PYRAMID } from "./glsl/pyramid.mjs";
import { VOTE, GATHER, COMPLETE, SCORE, PICK, REDO, PEAKS, TRACE } from "./glsl/finder.mjs";
import { SELECT } from "./glsl/bank.mjs";
import { NODES, REFIT, REFIT_POST, nodesSource } from "./glsl/register.mjs";
import { WORD } from "./glsl/word.mjs";
import { ANCH_FIND, ANCH_PREDICT, ANCH_REFIT, ANCH_FIT } from "./glsl/anchors.mjs";
import { classifySource } from "./glsl/classify.mjs";
import { SLOTS } from "./plan/common.mjs";
import { RING_COUNT } from "./plan/finder.mjs";
import { ROUND, NODES_MAX, LIGHT_MAX, WINDOW, SUB, ringOrder } from "./plan/register.mjs";
import { wordUniform, wordPoints, wordLattice } from "./plan/word.mjs";
import { anchUniform } from "./plan/anchors.mjs";
// The classifier's plan and its weight packers come from one module: plan/classify_gemm.mjs lays the weights out for
// its tiles, and the kernel refuses a buffer packed by cnn/weights.mjs.
import { classifyPlan, packWeights, packWeightsF16, packWeightsInt8 } from "./plan/classify_gemm.mjs";
import { INT8_TWINS, CLASSIFIER } from "./plan/classify.mjs";
import { loadNet } from "./cnn/netfile.mjs";
import { BackHalf } from "./back/back.mjs";
import { computePipeline } from "./pipeline.mjs";

export const COUNTERS = 16;     // u32 counters a frame, read back with every batch
// The bank's floor on the normalised response, and the deviation added under it so a flat patch does not
// divide by nothing (luma is 0..1). ai: In constants.mjs since 2026-10-01 (the kernels' plans that need them may not
// ai: import this module, which imports them), re-exported here.
import { LEVELS, TAU, EPS } from "./constants.mjs";
export { LEVELS, TAU, EPS };
// F6: the track score a frame's best hypothesis must reach (src/acquire.c accepts a mark quad at 0.15).
export const TAU_TRACK = 0.15;
// ai: The host's constants the native port (core/dec/front.cpp) reads from the generated setup (../gen/gen.mjs).
export const RESULT = 32;   // floats a frame in the finder's result record (glsl/finder.mjs PICK)
// ai: sel a frame: (ring, 1 + the version it decodes at, picture slot, 0), vec4u (glsl/word.mjs).
export const SEL_WORDS = 4;
export const TS_MAX = 80;   // timestamps a batch's readback has room for: 40 profiled stages
export const TS_PLAIN = 2;   // ai: an unprofiled batch's one compute pass: its pair at 0, 1
// ai: Ring slots past Bmax: a batch's worth can wait to launch while frames keep arriving during its copies. Slots
// ai: are freed at the submit, so four is slack enough at the camera's rate.
export const RING_SLACK = 4;

const U = globalThis.GPUBufferUsage ?? {};

// B "auto": the batch controller's submit cap, and the most frames a batch carries. 400 ms is a fifth of the ~2 s submit
// that lost the iGPU's device (a batch of eight 2160 frames, 2026-09-24): room for thermal throttling to double a
// batch's time, and a batch heavier than the ones before it, before the watch can shrink it.
// ai: 2026-09-25: batches as large as the camera's buffers allow or 16, whichever is lower, then 32 after a run at 32: B_CEIL caps every batch
// ai: (past 16 the ms a frame no longer falls with the symbol in view, 6.2 at 32 against 6.3 at 16 on the S26; the 400 ms cap, the guard against a device reset near 2 s, is
// ai: far from 32 x 6 ms, and it is what holds 4K to about 8, where a frame costs 20 ms). The native host's batcher
// ai: (core/dec/front.cpp) reads both from the setup.
export const CAP_MS = 400, B_CEIL = 32;   // ai: 32 since 2026-09-25 10:44Z (16 before)
// ai: SELECT's cap, the peaks a frame the classifiers read: 256 since 2026-09-30 (1,024 before). On the simulator's
// ai: cells every mark's strongest peak ranks within the first 168 of a frame's 853 to 8,192; the recordings read the
// ai: same blocks at 256 (STATUS "The small stages are their own work, and cnn-v3 reads four times the patches the
// ai: marks need").
export const CAP = 256;
// ai: The receivers' decoder, stated once: the classifier installed (a path under liblizard/: cnn-v3 on every peak,
// ai: since 2026-10-27; the cascade's cnn-v1 on its top 64 before, which read the same blocks to 0.15% on the
// ai: published lenses and today's replay for less of the iGPU's frame) and the options every receiver builds
// ai: FrontHalf with, read by the native setup (../gen/gen.mjs, a tree a variant). net: that file's document
// ai: (cnn/netfile.mjs loadNet). The caller adds precision, subgroups, backN, log and twins.
export const INSTALLED = Object.freeze({ classifier: CLASSIFIER });
export const receiverOptions = (net) => ({ B: "auto", cap: CAP, weights: net });

export const PRECISIONS = ["int8", "f16", "f32"];
// ai: The int8 twins' measurement (natively core/dec/front.cpp measureTwins): frames in its batch, and rounds of each
// ai: net's describe dispatch in each form, the first round untimed (a pipeline's first use costs more than any later
// ai: one); both arms read the same frames.
export const TWIN_FRAMES = 2, TWIN_ROUNDS = 3;
const TWINS = ["measure", "int8", "off"];
// ai: The bindings every classifier kernel (and SCORE, NODES, WORD) reads the pyramid through.
const LV = ["tex", "ro", "ro", "ro", "ro", "ro", "uniform"];

// ai: A classifier document's int8 twin (plan/classify.mjs INT8_TWINS), loaded, or null when the document is not
// ai: an installed float file (a candidate, a q8 file, one given packed or read from bytes). strict (twins "int8"):
// ai: a twin that will not load throws; otherwise it is logged and the float file runs.
async function int8Twin(doc, strict, log) {
  const src = doc?.source ? new URL(doc.source, import.meta.url).href : null;
  const name = src && Object.keys(INT8_TWINS).find((n) => new URL(`../${n}`, import.meta.url).href === src);
  if (!name) return null;
  try {
    const twin = await loadNet(new URL(`../${INT8_TWINS[name]}`, import.meta.url));
    if (!twin.arch?.quant || twin.arch.name !== doc.arch?.name) throw new Error(`${INT8_TWINS[name]} is not ${doc.arch?.name} in int8`);
    return twin;
  } catch (e) {
    if (strict) throw e;
    log(`no int8 twin for ${name} (${e.message}): the float file`);
    return null;
  }
}

export class FrontHalf {
  // cap: peaks a frame F3 keeps for F4.
  // ai: weights: F4's network (glsl/classify.mjs), a weights file's document (cnn/netfile.mjs), packed here for
  // ai: `precision` (or already packed by plan/classify_gemm.mjs packWeights / packWeightsF16). Required: every stage
  // ai: that finds is a trained net since 2026-09-26.
  // precision: "f16" stores the network's weights and activations as f16 where the adapter has shader-f16; without
  // it the classifier runs in f32 and the log says so.
  // ai: "int8" runs the proposer (bank fcn2) on int8 codes (glsl/bank_fcn2.mjs, the contract in cnn/q8.mjs); a q8
  // ai: classifier file (arch.quant, the contract in cnn/q8c.mjs) runs on the classifier's int8 form; everything
  // ai: else, the back half, the proposer and a float classifier file, runs in fh.floatPrecision: f16 where the
  // ai: adapter has shader-f16, else f32 ("f32" asked keeps f32 everywhere). A q8 classifier file at f16 or f32 is
  // ai: refused (no float path runs codes). fh.precision is the precision asked; fh.nets = { bank, small, large } what
  // ai: each net runs in ("int8", "f16", "f32", or null for none; small is always null, the field kept for the
  // ai: readers of the setup), logged once.
  // ai: The proposer (bank_fcn2.mjs, version 2) makes the peaks; a device whose workgroup memory holds none of its forms
  // ai: throws its FitError: no GPU decoder there, the C decodes. The classifier runs on every kept peak.
  // ai: B: frames a lane holds, or "auto" (the native host's batcher): the back half is built for B, B_CEIL under auto.
  // ai: tables: gpu/tables.mjs formatTables() ({ rings, pictures }). backN: the largest picture the back half builds
  // ai: (each picture at its top, tables.pictures); a frame whose word names a larger one is found and not decoded,
  // ai: and the largest built picture sets every stride and the readback.
  // ai: twins: at precision int8, what a classifier given as an installed float file runs (its int8 twin,
  // ai: plan/classify.mjs INT8_TWINS). "measure" (the default under B "auto"): both forms built, the host timing them on
  // ai: its first stream batch and keeping the faster (front.cpp measureTwins); "int8": the twin, unmeasured; "off"
  // ai: (the default at a fixed B): the file as given.
  // ai: The anchors (SPEC 6.13; glsl/anchors.mjs) always: ANCH_FIND, ANCH_PREDICT, the border's second search,
  // ai: ANCH_REFIT and ANCH_FIT after F7, pass 1 reading their records.
  static async create(adapter, tables, { B = "auto", cap = CAP, log = () => {}, weights = null, precision = "f32", backN = 1024, subgroups = true, twins = null } = {}) {
    if (tables?.rings?.length !== RING_COUNT || tables.pictures?.length !== SLOTS) throw new Error(`tables: gpu/tables.mjs formatTables(), ${RING_COUNT} rings and ${SLOTS} pictures`);
    const auto = B === "auto";
    if (!auto && !(B >= 1 && B === Math.floor(B))) throw new Error(`B ${B}: a whole number of frames, or "auto"`);
    if (!PRECISIONS.includes(precision)) throw new Error(`precision ${precision}: ${PRECISIONS.join(", ")}`);
    twins ??= auto ? "measure" : "off";
    if (!TWINS.includes(twins)) throw new Error(`twins ${twins}: ${TWINS.join(", ")}`);
    if (twins === "measure" && !auto) throw new Error("twins measure times them on the first stream batch under B \"auto\" (or twins int8 or off)");
    if (!weights) throw new Error("F4 needs a classifier's weights");
    // ai: A q8 classifier file carries codes and no float weights: it runs only at precision int8.
    if (precision !== "int8" && (weights.arch?.quant || weights.offsets?.int8)) throw new Error(`a q8 classifier file runs only at precision int8, not ${precision}`);
    // subgroups: the classifier reduces a split conv across adjacent lanes by shuffles where the adapter has it.
    // ai: subgroups false leaves the feature off even where the adapter has it (a device whose shuffles are suspect).
    const want = ["timestamp-query", "shader-f16", "subgroups"].filter((f) => adapter.features.has(f) && (subgroups || f !== "subgroups"));
    if (precision === "f16" && !want.includes("shader-f16")) { log("no shader-f16 on this adapter: f32"); precision = "f32"; }
    const floatPrecision = precision !== "f32" && want.includes("shader-f16") ? "f16" : "f32";
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: want,
      requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage },
    });
    const fh = new FrontHalf(device, tables, auto ? 0 : B, want, log, cap);
    fh.precision = precision;
    fh.floatPrecision = floatPrecision;
    // ai: twinForms: { large } the int8 twin built beside the float form for the host to measure, null when none.
    fh.twinForms = null;
    const f16 = floatPrecision === "f16";
    // ai: A net's packed weights and the precision it runs in: int8 for a q8 file, else the float precision.
    const packed = (w) => {
      const int8 = !!(w.offsets?.int8 || w.arch?.quant);
      const p = w.data ? w : int8 ? packWeightsInt8(w) : f16 ? packWeightsF16(w) : packWeights(w);
      if (!int8 && f16 !== (p.data instanceof Uint16Array)) throw new Error(`weights packed for the wrong precision (${floatPrecision})`);
      return { ...p, precision: int8 ? "int8" : floatPrecision };
    };
    fh.weights = packed(weights);
    // ai: The int8 twin of the file given, packed: taken at once (twins "int8") or built beside for the host's measurement.
    if (precision === "int8" && twins !== "off") {
      const twin = await int8Twin(weights, twins === "int8", log);
      if (twin && twins === "int8") fh.weights = packed(twin);
      else if (twin) fh.twinForms = { large: packed(twin) };
    }
    await fh.build();
    // ai: What each net runs in. A net bank's store leads its name ("int8 32x32 ...", "f16", "f32, halves stored";
    // ai: "packed" is f32 arithmetic with halves stored). A net whose twin waits to be measured runs its float form
    // ai: until then, and the line says so.
    const store = /^(int8|f16|f32|packed)\b/.exec(fh.bankStore ?? "")?.[1] ?? null;
    fh.nets = { bank: store === "packed" ? "f32" : store, small: null, large: fh.weights?.precision ?? null };
    log(`nets: bank ${fh.nets.bank}, small ${fh.nets.small}, large ${fh.nets.large}${fh.twinForms ? " (its int8 twin measured on the first batch)" : ""}`);
    // Its tables come from the wasm, which holds no other Lizard here (formatTables frees its own).
    // ai: Every picture at its top, so a frame decodes the version its word (or the held one) names out of it.
    const sizes = tables.pictures.map(({ n, subch }) => (n <= backN ? { n, subch } : null));
    // ai: picture: the rings, where each puts a picture's samples and the lattice its Coons patch runs on, which
    // ai: pass 1 samples by (on the frame's (ring, picture) pair).
    const picture = tables.rings.map((t) => ({ span: t.span, margin: t.margin, lattice: t.nodeX }));
    fh.backOpts = { sizes, precision: floatPrecision, cap: 0, picture, log };
    // ai: Under "auto" for B_CEIL, the lanes' size on every device whose memory allows it (the phone, the iGPU, the
    // ai: 4090); the native host builds it again at the batch size its memory holds under B_CEIL (gen writes every one).
    await fh.buildBack(auto ? B_CEIL : B);
    return fh;
  }

  constructor(device, tables, B, features, log, cap) {
    Object.assign(this, { device, tables, B, features, log, cap });
  }

  // The back half, its shaders compiled for B frames.
  async buildBack(B, quiet = false) {
    this.bh = await BackHalf.build(this.device, { B, ...this.backOpts, ...(quiet ? { log: () => {} } : {}) });
  }

  // Explicit layouts: `layout: "auto"` drops a binding the shader does not read, and then every bind group built
  // against the full list fails validation, which on some paths is silent (STATUS.md).
  layout(entries) {
    return this.device.createBindGroupLayout({ entries: entries.map((type, binding) => {
      const e = { binding, visibility: GPUShaderStage.COMPUTE };
      if (type === "tex") e.texture = { sampleType: "float", viewDimension: "2d-array" };
      else if (type === "sampler") e.sampler = { type: "filtering" };
      else if (type === "uniform") e.buffer = { type: "uniform" };
      else e.buffer = { type: type === "ro" ? "read-only-storage" : "storage" };
      return e;
    }) });
  }

  // ai: A stage's pipeline from its GLSL (pipeline.mjs computePipeline) on a layout of these entries.
  async pipeline(glsl, entries) {
    const bgl = this.layout(entries);
    const p = await computePipeline(this.device, { code: { glsl }, layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }) });
    return { p, bgl };
  }

  // ai: The classifier's kernel and weights buffer, from its packed weights (create), run on every kept peak.
  // ai: { pipe, buf, w, plan }; pipe.P is the patches a workgroup of this kernel takes (its plan's), which a dispatch
  // ai: divides its slots by.
  async netForm(w) {
    const f = { f16: w.precision === "f16", int8: w.precision === "int8", subgroups: this.features.includes("subgroups") };
    const plan = classifyPlan(w.arch, f);
    const pipe = { ...(await this.pipeline(classifySource(w.offsets, w.arch, f), [...LV, "ro", "ro", "rw", "uniform", "ro"])), P: plan.P };
    const buf = this.device.createBuffer({ size: w.data.byteLength, usage: U.STORAGE | U.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, w.data);
    return { pipe, buf, w, plan };
  }

  async build() {
    // ai: Every pipeline compiled at once (2026-09-29): one at a time, the S26's front half took 1.7 to 5.2 s of the GPU
    // ai: worker's 4.5 to 8 s start (STATUS "The ramp at the start").
    const tf = this.twinForms, subgroups = this.features.includes("subgroups") ? " subgroups" : "";
    const P = (code, entries) => this.pipeline(code, entries);
    // ai: The proposer is a module (gpu/bank_fcn2.mjs, version 2). A device whose workgroup memory cannot hold it even with its
    // ai: activations stored as halves (under 18,692 B; WebGPU promises only 16 KB) gets its FitError: no GPU decoder
    // ai: there, and the receiver decodes with the C.
    const bank = (async () => { this.bank = await import("./bank_fcn2.mjs"); await this.bank.build(this); })();
    const [pyramid, large, twinLarge, vote, gather, complete, score, pick, redo, nodesP, refit, refitPost, wordP, select, peaks, trace] = await Promise.all([
      // ai: The pyramid in one kernel (glsl/pyramid.mjs): the layer, the frame records, levels
      // ai: 1 to 4, the levels' dims.
      P(PYRAMID, ["tex", "ro", "rw", "rw", "rw", "rw", "uniform"]),
      this.netForm(this.weights),
      // ai: The twin waiting to be measured, built beside the float form: { pipe, buf, w, plan }.
      tf ? this.netForm(tf.large) : null,
      // ai: F5 (glsl/finder.mjs): VOTE a lane a reading into the accumulator, then GATHER a workgroup a frame, its
      // ai: centre peaks (cand) and their quads.
      P(VOTE, ["ro", "ro", "rw", "uniform"]),
      P(GATHER, ["ro", "ro", "ro", "rw", "rw", "uniform", "ro", "rw"]),
      // ai: F5d: a quad of three corners completed from the ring's sides (the levels, readings, quads, P5)
      P(COMPLETE, [...LV, "ro", "rw", "uniform"]),
      P(SCORE, [...LV, "ro", "ro", "rw", "uniform"]),
      P(PICK, ["ro", "ro", "ro", "rw", "rw", "rw", "uniform"]),
      // ai: The second round's frame records and its frames' resets: frames, sel, frames2, acc, readings, cand, quads.
      P(REDO, ["ro", "ro", "rw", "rw", "rw", "rw", "rw", "uniform", "rw", "uniform"]),
      P(NODES, [...LV, "ro", "ro", "ro", "ro", "rw", "uniform"]),
      P(REFIT, ["ro", "ro", "ro", "ro", "rw", "rw", "uniform", "rw"]),
      // ai: F7's last step after the last round's REFIT (glsl/register.mjs REFIT_POST), on REFIT's bindings
      P(REFIT_POST, ["ro", "ro", "ro", "ro", "rw", "rw", "uniform", "rw"]),
      // ai: F8 (glsl/word.mjs): the levels, then maps, sel (read-write: sel.y = 1 + the version the frame decodes at, sel.z
      // ai: its picture), resid, the word plan's points, the finder's results, the word code, the lattice and the lane's
      // ai: held version.
      P(WORD, [...LV, "ro", "rw", "ro", "ro", "rw", "uniform", "uniform", "uniform"]),
      // ai: F3 (glsl/bank.mjs SELECT)
      P(SELECT, ["ro", "ro", "rw", "rw", "uniform"]),
      // ai: F5b (glsl/finder.mjs PEAKS): a lane a cell, the merged rings' peaks into the frame's list
      P(PEAKS, ["ro", "ro", "rw", "rw", "uniform"]),
      // ai: TRACE (glsl/finder.mjs): each quad's bend, a workgroup a (frame, quad): the levels, then quads
      P(TRACE, [...LV, "rw"]),
      bank,
    ]);
    Object.assign(this, { pyramid, vote, gather, complete, score, pick, redo, nodesP, refit, refitPost, wordP, select, peaks, trace });
    this.classify = large.pipe;
    this.weightsBuf = large.buf;
    this.arch = large.plan.name;
    this.log(`classifier ${large.plan.name} ${this.weights.precision}${subgroups}: ${large.plan.macs} MACs a patch, ${large.plan.P} patches a workgroup, ${large.plan.bytes} bytes of workgroup memory`);
    if (tf) {
      tf.large = twinLarge;
      this.log("int8 twins built beside the float forms");
    }
    const d = this.device, rings = this.tables.rings, pictures = this.tables.pictures;
    // ai: The finder's fixed buffers. The track plan: per ring, the reference points (4 a step) then the track cells
    // with their signs, as (x, y, sign, 0) in module coordinates.
    const pts = [], at = new Uint32Array(4 * RING_COUNT);
    rings.forEach((t, i) => {
      at.set([pts.length / 4, t.track.refSteps, t.track.cells, t.modules], 4 * i);
      const np = t.track.pts.length / 2, nref = 4 * t.track.refSteps;
      for (let k = 0; k < np; k++) pts.push(t.track.pts[2 * k], t.track.pts[2 * k + 1], k < nref ? 0 : t.track.signs[k - nref], 0);
    });
    this.planBuf = d.createBuffer({ size: 4 * pts.length, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.planBuf, 0, new Float32Array(pts));
    this.planUni = d.createBuffer({ size: at.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.planUni, 0, at);
    // ai: Each ring's modules a side, padded to a vec4f.
    this.modules = new Float32Array(4);
    this.modules.set(rings.map((t) => t.modules));
    const p4 = new Uint32Array([this.cap, 0, 0, 0]);
    this.p4Uni = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.p4Uni, 0, p4);
    // ai: PICK's uniform (glsl/finder.mjs PICK's): the accept line and the rings' module counts, the same every batch.
    const pickAb = new ArrayBuffer(32);
    new Float32Array(pickAb)[0] = TAU_TRACK; new Float32Array(pickAb).set(this.modules, 4);
    this.pickUni = d.createBuffer({ size: pickAb.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.pickUni, 0, pickAb);
    // ai: F7's tables. Per ring: the border nodes in ring order, each with its window of the border as painted
    // (glsl/register.mjs NODES).
    const nodes = [], tmpl = [], count = new Uint32Array(4), nbase = new Uint32Array(4);
    rings.forEach((t, si) => {
      const nx = t.nodeX.length, M = t.modules;
      count[si] = 2 * nx + 2 * nx - 4; nbase[si] = nodes.length / 4;
      for (const [i, j] of ringOrder(nx, nx)) {
        const sx = t.nodeX[i], sy = t.nodeX[j], first = tmpl.length / 4;
        for (let my = Math.floor(sy - WINDOW / 2); my < Math.floor(sy - WINDOW / 2) + WINDOW; my++) {
          for (let mx = Math.floor(sx - WINDOW / 2); mx < Math.floor(sx - WINDOW / 2) + WINDOW; mx++) {
            if (mx < 0 || my < 0 || mx >= M || my >= M) continue;
            const k = t.kind[my * M + mx];
            if (k & 8 || (k & 3) === 0) continue;   // a format-word cell (its content is the sender's) or picture
            for (let b = 0; b < SUB; b++) for (let a = 0; a < SUB; a++) tmpl.push(mx + (a + 0.5) / SUB, my + (b + 0.5) / SUB, (k & 3) === 2 ? 0 : 1, 0);
          }
        }
        nodes.push(sx, sy, first, tmpl.length / 4 - first);
      }
    });
    if (Math.max(...count) > NODES_MAX) throw new Error(`${Math.max(...count)} border nodes, room for ${NODES_MAX}`);
    this.nodesMax = Math.max(...count);
    this.nodesBuf = d.createBuffer({ size: 4 * nodes.length, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.nodesBuf, 0, new Float32Array(nodes));
    this.tmplBuf = d.createBuffer({ size: 4 * tmpl.length, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.tmplBuf, 0, new Float32Array(tmpl));
    // ai: F8's tables: the word plan's points (every ring's), the code, plan and picture uniform, the lattice uniform.
    const word = wordPoints(rings), wu = wordUniform(rings, word.plans, pictures), wl = wordLattice(rings);
    this.wordPtsBuf = d.createBuffer({ size: word.pts.byteLength, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.wordPtsBuf, 0, word.pts);
    this.wordUni = d.createBuffer({ size: wu.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.wordUni, 0, wu);
    this.wordLatUni = d.createBuffer({ size: wl.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.wordLatUni, 0, wl);
    // Rounds of the node search: +-3 modules at half a module, one template sample a module; then +-0.6 at a
    // ai: fifth, all four. Each with every node's count of the light samples it takes (glsl/register.mjs Round's light).
    if (nodes.length / 4 > LIGHT_MAX) throw new Error(`${nodes.length / 4} nodes over the rings, room for ${LIGHT_MAX}`);
    // ai: a round's uniform (steps, stride, step), kept as a maker for the anchors' own round (buildAnchors)
    this.roundOf = ([steps, stride, step]) => {
      const ab = new ArrayBuffer(64 + 4 * LIGHT_MAX), u = new Uint32Array(ab), fl = new Float32Array(ab);
      u.set([steps, stride]); fl[4] = step; u.set(count, 8); u.set(nbase, 12);
      for (let q = 0; q < nodes.length / 4; q++) {
        const first = nodes[4 * q + 2], P = Math.ceil(nodes[4 * q + 3] / stride);
        for (let j = 0; j < P; j++) u[16 + q] += tmpl[4 * (first + j * stride) + 2];
      }
      const b = d.createBuffer({ size: ab.byteLength, usage: U.UNIFORM | U.COPY_DST });
      d.queue.writeBuffer(b, 0, ab);
      return b;
    };
    // ai: F7's one round: +-3 modules at half a module from the finder's map, half the template's samples (a fine round
    // ai: about the refit's map, +-0.6 at a fifth, until 2026-10-28: 0.4% more blocks on an anchored replay for 7 to 8%
    // ai: more of the iGPU's frame, the anchors' border search doing its work)
    this.round = this.roundOf(ROUND);
    const fitAb = new ArrayBuffer(48);
    new Uint32Array(fitAb).set(count, 0); new Uint32Array(fitAb).set(nbase, 4); new Float32Array(fitAb).set(this.modules, 8);
    this.fitUni = d.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.fitUni, 0, fitAb);
    await this.buildAnchors();
  }

  // ai: The anchors' stages (glsl/anchors.mjs), once, from build(): ANCH_FIND, ANCH_PREDICT, the border's second search
  // ai: (nodesC: glsl/register.mjs nodesSource centred, on NODES' layout, with its own round, anchRound), ANCH_REFIT (ANCH_PREDICT
  // ai: on its layout, so its group serves both), ANCH_FIT, their uniform (anchUni: each ring's knots, a, span and
  // ai: margin, then each ring's inverse of the squares' fill's normal matrix).
  async buildAnchors() {
    if (this.anchFind) return;
    const P = (code, entries) => this.pipeline(code, entries);
    // ai: 0 to 6 the same in both: the frame texture, frames, maps, sel, resid, the lattice (F8's wordLatUni), anchUni;
    // ai: then ANCH_FIND's anch (rw), ANCH_FIT's anch (ro), arec and result (rw)
    // ai: ANCH_REFIT between them, on REFIT_POST's first seven (frames, nodes, sel, meas, maps, resid, fitUni; meas and
    // ai: maps written too), the anchors' records (rw), the lattice and anchUni
    const [anchFind, anchFit, anchRefit] = await Promise.all([
      P(ANCH_FIND, ["tex", "ro", "ro", "ro", "ro", "uniform", "uniform", "rw"]),
      P(ANCH_FIT, ["tex", "ro", "ro", "ro", "ro", "uniform", "uniform", "ro", "rw", "rw"]),
      P(ANCH_REFIT, ["ro", "ro", "ro", "rw", "rw", "rw", "uniform", "rw", "uniform", "uniform"]),
    ]);
    const on = async (glsl, bgl) => ({ p: await computePipeline(this.device, { code: { glsl }, layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }) }), bgl });
    // ai: the border's second search's own round: +-0.5 module at a quarter, every template sample, so the samples (half
    // ai: a module apart) and the offsets share a lattice, 51 points a side read once each (+-0.4 at a fifth until
    // ai: 2026-10-31, each position read directly; +-0.6, the fine round F7 had until 2026-10-28: 0.07% more blocks over
    // ai: the published lenses for 0.06 to 0.08 ms more of the S26's half; +-0.2 no cheaper there and 0.22% fewer; one
    // ai: sample a module 4.2% fewer); its kernel built for it, a quarter of the template a pass and 256 (offset,
    // ai: group) pairs, 15.0 KB of workgroup memory
    const ANCH_ROUND = [2, 1, 0.25], nodesCOpts = { centred: true, round: ANCH_ROUND, chunk: 144, items: 256 };
    const [anchPredict, nodesC] = await Promise.all([on(ANCH_PREDICT, anchRefit.bgl), on(nodesSource(nodesCOpts), this.nodesP.bgl)]);
    const au = anchUniform(this.tables.rings);
    this.anchUni = this.device.createBuffer({ size: au.byteLength, usage: U.UNIFORM | U.COPY_DST });
    this.device.queue.writeBuffer(this.anchUni, 0, au);
    const anchRound = this.roundOf(ANCH_ROUND);
    Object.assign(this, { anchFind, anchPredict, nodesC, anchRound, anchFit, anchRefit });
  }

}
