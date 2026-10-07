// The GPU decoder's host: the device, the pipelines, and batches of frames through them.
//
// A batch is up to B frames, each a layer of one texture. Everything a batch does is encoded into one command
// buffer and comes back in one readback; nothing in between waits on the host. Frames in a batch may differ in
// size and in picture size, so every per-frame fact lives in a buffer the shaders index by frame.
//
// Lanes: a batch's working buffers, bind groups and readback are one lane, and there are `inflight` of them (2) so
// the next batch is encoded and submitted while the one before it runs and reads back (throughput over latency).
// Pipelines, format tables and weights are compiled and uploaded once and shared. A lane is made over the shared
// object (Object.create), so its own buffers shadow the shared slots and every read of a pipeline or a table falls
// through; the bank modules take a lane as their `fh` unchanged. The shared object never runs a batch, so nothing
// a lane shadows exists on it.
//
// ai: The ring (the GPU queue): a frame goes onto the device the moment it arrives (enqueue: F0's ingest pass for a
// ai: VideoFrame, writeTexture for luma), into one of R layers of a texture at the lanes' shape, and the VideoFrame
// ai: is closed at once. A batch is cut from staged slots, never from open frames (each held a camera buffer until
// ai: its batch submitted, and the phone's camera starves at about five): its encoder begins with a copy a slot into
// ai: the lane's layer, and the slots are freed at the submit. No stage reads the ring.
//
// B "auto" (the default): the batch controller (Batcher, at the end) sizes the lanes and the ring for the largest
// batch the memory budget allows and chooses how many frames a batch carries from what the device is measured to
// tolerate. A batch may carry any number of frames up to its lane's B: every dispatch covers only the frames it
// carries, and a slot left empty exits at once.
import { PYRAMID, PYRAMID_BLOCK, HIST_BINS } from "./wgsl/pyramid.mjs";
import { SAMPLE } from "./wgsl/sample.mjs";
import { SLOTS } from "./wgsl/common.mjs";
import { SELECT } from "./wgsl/bank.mjs";
import { VOTE, GATHER, SCORE, PICK, K, CELL, HYP, RING_COUNT } from "./wgsl/finder.mjs";
import { NODES, REFIT, NODES_MAX, WINDOW, SUB, REFIT_DBG, ringOrder } from "./wgsl/register.mjs";
import { WORD, wordUniform, wordPoints, wordLattice, heldUniform } from "./wgsl/word.mjs";
import { gridOf, slotOf } from "./tables.mjs";
import { VERSION_MAX } from "./wordcode.mjs";
// The classifier kernel and its weight packers come from one module: classify_gemm.mjs lays the weights out for its
// tiles, and its classifySource refuses a buffer packed by cnn/weights.mjs.
import { classifySource, classifyPlan, rankSource, packWeights, packWeightsF16, packWeightsInt8 } from "./wgsl/classify_gemm.mjs";
import { INT8_TWINS } from "./wgsl/classify.mjs";
import { loadNet } from "./cnn/netfile.mjs";
import { BackHalf, parseReadback, PASSES as BACK_PASSES, PASSES2 as CANCEL_PASSES } from "./back/back.mjs";
import { blocksAt } from "./back/tiers.mjs";
import { Ingest } from "./ingest.mjs";
import { computePipeline } from "./pipeline.mjs";
import { Probe } from "./probe.mjs";

export const COUNTERS = 16;     // u32 counters a frame, read back with every batch (see COUNT)
// What each counter slot holds, so a stage that silently did nothing reads as a zero here.
export const COUNT = { samples: 0, rawPeaks: 1, peaks: 2, cascade: 3 };   // cascade: peaks the large net ran on
// F2/F3: each 16 x 16 tile of every level keeps its KEEP strongest peaks in fixed slots; F3 keeps the strongest
// `cap` of a frame's slots.
// The bank's floor on the normalised response, and the deviation added under it so a flat patch does not
// divide by nothing (luma is 0..1). ai: In constants.mjs since 2026-10-01 (the kernels that need them may not import
// ai: this module, which imports them), re-exported here with INT8.
import { LEVELS, TAU, EPS, INT8 } from "./constants.mjs";
export { LEVELS, TAU, EPS, INT8 };
// F6: the track score a frame's best hypothesis must reach (src/acquire.c accepts a mark quad at 0.15).
export const TAU_TRACK = 0.15;
const RESULT = 32;   // floats a frame in the finder's result record (wgsl/finder.mjs PICK)
// ai: sel a frame: (ring, 1 + the version it decodes at, picture slot, 0), vec4u (wgsl/word.mjs).
const SEL_WORDS = 4;
const TS_MAX = 80;   // timestamps a batch's readback has room for: 40 profiled stages (pass two's 14 among them)
const TS_PLAIN = 2;   // ai: an unprofiled batch's one compute pass: its pair at 0, 1
const TS_INGEST = 2;   // ai: a ring slot's F0 pair (its ingest pass ran before any batch, outside the batch's timestamps)
// ai: Ring slots past Bmax: a batch's worth can wait to launch while frames keep arriving during its copies. Slots
// ai: are freed at the submit, so four is slack enough at the camera's rate.
const RING_SLACK = 4;

const U = globalThis.GPUBufferUsage ?? {};   // node imports this file for its constants only
const ceilDiv = (a, b) => Math.ceil(a / b);
const MiB = 1 << 20;

// ai: A frame staged on the ring: layer `id` holds its crop, w x h at the origin, until a batch copies it out. at:
// ai: performance.now() at enqueue (the queue's launch deadline runs from it); f0: a VideoFrame's crop, whose ingest
// ai: pass wrote the slot's timestamp pair; ring: the ring the frame was told, when it was (maps given).
class Slot {
  constructor(id, w, h, f0, ring) {
    Object.assign(this, { id, w, h, at: performance.now(), f0 });
    if (ring !== undefined) this.ring = ring;
  }
}
// B "auto": the device memory the decoder may hold, and the submit time a batch is kept under. 128 MiB is a budget of
// about 100 MB with margin. 400 ms is a fifth of the ~2 s submit that lost the iGPU's device (a batch of eight 2160
// frames, 2026-09-24): room for thermal throttling to double a batch's time, and a batch heavier than the ones before
// it, before the watch can shrink it.
// ai: 2026-09-25: batches as large as the camera's buffers allow or 16, whichever is lower, then 32 after a run at 32: B_CEIL caps every batch
// ai: (past 16 the ms a frame no longer falls with the symbol in view, 6.2 at 32 against 6.3 at 16 on the S26; the 400 ms cap, the guard against a device reset near 2 s, is
// ai: far from 32 x 6 ms, and it is what holds 4K to about 8, where a frame costs 20 ms). The camera's buffers never
// ai: bound it: a frame moves to the device as it arrives and the camera held 60 a second with two batches of 32 ahead.
// ai: The memory budget is the device's own limit (WebGPU's limits), starting at maxBufferSize and degrading
// ai: gracefully: WebGPU states no memory size, so the budget is maxBufferSize itself (2 GiB on the S26 Ultra, 4 on the
// ai: 4090, 256 MiB on SwiftShader), and B_CEIL, not memory, sizes the lanes on any device we have (two lanes of 16 and
// ai: their ring take about 220 MB at a 1080 crop in f16, 330 at 1440, 600 at a 2052 crop). The degrading is in
// ai: makeLanes: an allocation that fails its out-of-memory scope halves the batch and tries again down to 1; a lane's
// ai: buffers are held under maxBufferSize and the layers under maxTextureArrayLayers in the plan; a device lost is
// ai: recovered once at half the batch. BUDGET_MIN is the fallback for an adapter that states no limits. A caller's own
// ai: budget (the harness's BUDGET env) still overrides.
// ai: What a frame's pilots read (SPEC 7.3, src/focus.c focus_pilot), over its first `blocks` blocks of the soft
// ai: stage's per-block r (wgsl/back_soft.mjs PILOT): r the mean over the even blocks (bit 0 of the painted count: +1
// ai: on an even picture, -1 on an odd one) and r2 over the odd (bit 1: it flips every second picture), sd and sd2
// ai: their standard errors; 1 - 2 f on a tear f of whose samples came from pictures of the other bit ((1 - 2 f) /
// ai: sqrt(1 - 2 f + 2 f^2) on a blend of every sample). { r, sd, r2, sd2, blocks }, or null for none.
export function pilotOf(pilots, blocks) {
  // ai: the even blocks (bit 0 of the painted count) and the odd (bit 1), each its own mean and standard error
  const out = { r: 0, sd: 0, r2: 0, sd2: 0, blocks: 0 };
  for (let g = 0; g < 2; g++) {
    let s = 0, s2 = 0, k = 0;
    for (let b = g; b < Math.min(blocks, pilots.length); b += 2) { const v = pilots[b]; if (Number.isFinite(v)) { s += v; s2 += v * v; k++; } }
    if (!k) continue;
    const r = s / k, v = k > 1 ? (s2 - s * r) / (k - 1) : 0;
    if (g) { out.r2 = r; out.sd2 = Math.sqrt(Math.max(v, 0) / k); } else { out.r = r; out.sd = Math.sqrt(Math.max(v, 0) / k); }
    out.blocks += k;
  }
  return out.blocks ? out : null;
}
export const CAP_MS = 400, B_CEIL = 32, BUDGET_MIN = 128 * MiB;   // ai: 32 since 2026-09-25 10:44Z (16 before)
// ai: SELECT's cap, the peaks a frame the classifiers read: 256 since 2026-09-30 (1,024 before). On the simulator's
// ai: cells every mark's strongest peak ranks within the first 168 of a frame's 853 to 8,192 (scripts/exp/gpu_marks.mjs
// ai: CAP=8192); the recordings read the same blocks at 256 (STATUS "The small stages are their own work, and cnn-v3
// ai: reads four times the patches the marks need").
export const CAP = 256;
export const budgetFor = (limits) => limits?.maxBufferSize || BUDGET_MIN;

// ai: INT8, the WGSL language feature the int8 nets need, is constants.mjs's (imported and re-exported above).
export const PRECISIONS = ["int8", "f16", "f32"];
// ai: The precisions this browser and adapter can run, best first: int8 where WGSL has INT8, f16 where the adapter has
// ai: shader-f16, then f32. create() takes one; the caller tries them in turn (since 2026-09-25).
export function precisionChain(adapter) {
  return [
    ...(globalThis.navigator?.gpu?.wgslLanguageFeatures?.has(INT8) ? ["int8"] : []),
    ...(adapter?.features?.has("shader-f16") ? ["f16"] : []),
    "f32",
  ];
}
// ai: The int8 twins' measurement (FrontHalf.measureTwins): frames in its batch, and rounds of each net's describe
// ai: dispatch in each form, the first round untimed (a pipeline's first use costs more than any later one). Two and
// ai: three keep it to about 6 s on SwiftShader, where four and four took 15 (2026-09-26); both arms read the same frames.
const TWIN_FRAMES = 2, TWIN_ROUNDS = 3;
const TWINS = ["measure", "int8", "off"];
// ai: The proposers (gpu/bank_<name>.mjs): v2, the default, and v1. (v3, 2026-10-01, is in archive/proposer-v3/.)
export const BANKS = ["fcn2", "fcn"];
// ai: The bindings every classifier kernel (and SCORE, NODES, WORD) reads the pyramid through.
const LV = ["tex", "ro", "ro", "ro", "ro", "ro", "uniform"];

// ai: A classifier document's int8 twin (gpu/wgsl/classify.mjs INT8_TWINS), loaded, or null when the document is not
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
  // register: run F7 after the finder (off gives the finder's bare homography, for comparison).
  // ai: weights: F4's network (wgsl/classify_gemm.mjs), a weights file's document (cnn/netfile.mjs), packed here for
  // ai: `precision` (or already packed by classify_gemm.mjs packWeights / packWeightsF16). Required: every stage that
  // ai: finds is a trained net since 2026-09-26; the hand-written F4 (DESCRIBE) and the DoG banks (global, tiled,
  // ai: perscale) are deleted.
  // precision: "f16" stores the network's weights and activations as f16 where the adapter has shader-f16; without
  // it the classifier runs in f32 and the log says so.
  // ai: "int8" runs the proposer (bank fcn2) on int8 codes (wgsl/bank_fcn2.mjs, the contract in cnn/q8.mjs) and throws
  // ai: where WGSL lacks INT8 (the caller's precisionChain falls back); a q8 classifier file (arch.quant, the contract
  // ai: in cnn/q8c.mjs) runs on classify_gemm.mjs's int8 form; everything else, the back half, bank fcn and a float
  // ai: classifier file, runs in fh.floatPrecision: f16 where the adapter has shader-f16, else f32 ("f32" asked keeps
  // ai: f32 everywhere). A q8 classifier file at f16 or f32 is refused (no float path runs codes). fh.precision is the
  // ai: precision asked; fh.nets = { bank, small, large } what each net runs in ("int8", "f16", "f32", or null for
  // ai: none), logged once.
  // ai: bank: which proposer module (gpu/bank_<name>.mjs) makes the peaks: fcn2 (proposer v2, the default) or fcn (v1).
  // ai: A device whose workgroup memory holds neither throws the bank's FitError: no GPU decoder there, the C decodes.
  // cascade: { weights, keep, rest } runs that (small) net on every kept peak and `weights` only on the `keep`
  // strongest a frame by its mark logit (wgsl/classify.mjs CASCADE); null runs `weights` on every kept peak.
  // B: frames a lane holds, or "auto" (Batcher). budget, capMs: the controller's memory budget in bytes and submit
  // cap in ms. restart: under "auto", the batch size to start at and never grow past (a decoder made again after its
  // device was lost, at half the batch that was running).
  // inflight: lanes, the batches that can be on the device at once (see run()); "auto" is 2.
  // back: "gpu" (the default) puts the back half (back/back.mjs: transform, soft values, LDPC, CRC) behind the
  // sampler in the batch's encoder, each lane owning a BackHalf lane over its grid, and the readback carries the
  // verified blocks, a verdict a block and the counters; "wasm" reads each sampled grid back for the host's back half.
  // ai: tables: gpu/tables.mjs formatTables() ({ rings, pictures }). backN: the largest picture the back half builds
  // ai: (each picture at its top, tables.pictures); a frame whose word names a larger one is found and not decoded,
  // ai: and the largest built picture sets every stride and the readback.
  // fuse: with the GPU back half, its pass 1 samples each picture itself (wgsl/back_transform.mjs, fused) and F9 and
  // the grid are left out, except in a batch that asks for the grids; false keeps F9 and the grid on every batch.
  // ai: probe: a batch run with probe: true also reads back what every stage left (probe.mjs), for the self-test page
  // ai: (scripts/pages/gpu_selftest.mjs); it gives four kinds of buffer COPY_SRC, so the receiver never sets it.
  // ai: cancel: with the GPU back half, straddle cancellation as pass two of every batch (back/cancel.mjs): the short
  // ai: frames read again with their known neighbours subtracted, in the batch's own command buffer after pass one,
  // ai: its blocks and verdicts in the batch's readback. The carry it keeps between batches (the batch before's last
  // ai: reference) is a shared buffer made with the back half (buildBack).
  // ai: twins: at precision int8, what a classifier given as an installed float file runs (its int8 twin,
  // ai: gpu/wgsl/classify.mjs INT8_TWINS). "measure" (the default under B "auto"): both forms are built and timed on the
  // ai: first stream batch's frames (measureTwins), and each net keeps the faster, by measurement alone; "int8": the
  // ai: twins, unmeasured; "off" (the default at a fixed B): the files as given. fh.twins is the measurement once made;
  // ai: fh.nets says what each net runs in.
  // ai: fh.boot: what create spent, ms: the device, the int8 twins fetched, the front half's pipelines (build), the
  // ai: probe kit and the back half (under B "auto" built here at B_CEIL, 2026-09-29), and the total.
  static async create(adapter, tables, { B = "auto", budget = budgetFor(adapter?.limits), capMs = CAP_MS, restart = 0, cap = CAP, register = true, log = () => {}, weights = null, precision = "f32", bank = "fcn2", cascade = null, inflight = "auto", back = "gpu", backN = 1024, fuse = true, subgroups = true, probe = false, cancel = false, twins = null, tiers = null } = {}) {
    if (back !== "wasm" && back !== "gpu") throw new Error(`back ${back}`);
    if (tables?.rings?.length !== RING_COUNT || tables.pictures?.length !== SLOTS) throw new Error(`tables: gpu/tables.mjs formatTables(), ${RING_COUNT} rings and ${SLOTS} pictures`);
    const whole = (v) => v >= 1 && v === Math.floor(v);
    const auto = B === "auto";
    if (!auto && !whole(B)) throw new Error(`B ${B}: a whole number of frames, or "auto"`);
    if (inflight !== "auto" && !whole(inflight)) throw new Error(`inflight ${inflight}`);
    if (auto && !(budget > 0 && capMs > 0)) throw new Error(`budget ${budget}, capMs ${capMs}`);
    if (!PRECISIONS.includes(precision)) throw new Error(`precision ${precision}: ${PRECISIONS.join(", ")}`);
    if (precision === "int8" && !globalThis.navigator?.gpu?.wgslLanguageFeatures?.has(INT8)) throw new Error(`precision int8: this browser's WGSL has no ${INT8}`);
    twins ??= auto ? "measure" : "off";
    if (!TWINS.includes(twins)) throw new Error(`twins ${twins}: ${TWINS.join(", ")}`);
    if (twins === "measure" && !auto) throw new Error("twins measure times them on the first stream batch under B \"auto\" (or twins int8 or off)");
    if (!weights) throw new Error("F4 needs a classifier's weights");
    // ai: A q8 classifier file carries codes and no float weights: it runs only at precision int8.
    if (precision !== "int8" && [weights, cascade?.weights].some((w) => w?.arch?.quant || w?.offsets?.int8)) throw new Error(`a q8 classifier file runs only at precision int8, not ${precision}`);
    if (cascade && (!cascade.weights || !(cascade.keep > 0) || cascade.keep > cap || ((cascade.rest ?? "small") !== "small" && cascade.rest !== "zero"))) throw new Error(`cascade: a small net's weights, keep in 1..cap (${cap}), rest small or zero`);
    // subgroups: the classifier reduces a split conv across adjacent lanes by shuffles where the adapter has it.
    // ai: subgroups false leaves the feature off even where the adapter has it (a device whose shuffles are suspect).
    const want = ["timestamp-query", "shader-f16", "subgroups"].filter((f) => adapter.features.has(f) && (subgroups || f !== "subgroups"));
    if (precision === "f16" && !want.includes("shader-f16")) { log("no shader-f16 on this adapter: f32"); precision = "f32"; }
    const floatPrecision = precision !== "f32" && want.includes("shader-f16") ? "f16" : "f32";
    const L = adapter.limits, t0 = performance.now(), boot = {}, lapBoot = (k) => { boot[k] = Math.round(performance.now() - t0 - Object.values(boot).reduce((a, v) => a + v, 0)); };
    const device = await adapter.requestDevice({
      requiredFeatures: want,
      requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize,
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage },
    });
    // ai: A build that throws (an int8 bank this device cannot run, a bad weights file) takes its device with it,
    // ai: so a caller trying the next precision (lizard-web/gpuqueue.mjs) does not leave one alive.
    try {
      device.addEventListener("uncapturederror", (e) => log(`uncaptured: ${e.error.message}`));
      lapBoot("device");
      const fh = new FrontHalf(device, tables, auto ? 0 : B, want, log, cap);
      // A caller that finds its batch failed asks here whether the device went with it (the page then makes a new one).
      fh.lost = null;
      device.lost.then((info) => { fh.lost = info; });
      fh.probeOn = probe;
      fh.auto = auto;
      fh.batcher = auto ? new Batcher(fh, { budget, capMs, restart, inflight }) : null;
      fh.fixedInflight = auto ? 0 : inflight === "auto" ? 2 : inflight;
      fh.bankName = bank;
      fh.register = register;
      fh.precision = precision;
      fh.floatPrecision = floatPrecision;
      fh.weights = null;
      fh.cascade = null;
      // ai: twinForms: { large, small } the int8 twins waiting to be measured (packed, then built), null when none;
      // ai: twins: the measurement once made (measureTwins).
      fh.twinForms = null;
      fh.twins = null;
      const f16 = floatPrecision === "f16";
      // ai: A net's packed weights and the precision it runs in: int8 for a q8 file, else the float precision.
      const packed = (w) => {
        const int8 = !!(w.offsets?.int8 || w.arch?.quant);
        const p = w.data ? w : int8 ? packWeightsInt8(w) : f16 ? packWeightsF16(w) : packWeights(w);
        if (!int8 && f16 !== (p.data instanceof Uint16Array)) throw new Error(`weights packed for the wrong precision (${floatPrecision})`);
        return { ...p, precision: int8 ? "int8" : floatPrecision };
      };
      fh.weights = packed(weights);
      if (cascade) fh.cascade = { weights: packed(cascade.weights), keep: cascade.keep, rest: cascade.rest ?? "small" };
      // ai: The int8 twins of the files given, packed: taken at once (twins "int8") or left for measureTwins.
      if (precision === "int8" && twins !== "off") {
        const tw = { large: await int8Twin(weights, twins === "int8", log), small: fh.cascade ? await int8Twin(cascade.weights, twins === "int8", log) : null };
        if (twins === "int8") {
          if (tw.large) fh.weights = packed(tw.large);
          if (tw.small) fh.cascade.weights = packed(tw.small);
        } else if (tw.large || tw.small) fh.twinForms = { large: tw.large && packed(tw.large), small: tw.small && packed(tw.small) };
      }
      lapBoot("twins");
      await fh.build();
      lapBoot("build");
      // ai: What each net runs in. A net bank's store leads its name ("int8 32x32 ...", "f16", "f32, halves stored";
      // ai: "packed" is f32 arithmetic with halves stored). A net whose twin waits to be measured
      // ai: runs its float form until then, and the line says so.
      const store = /^(int8|f16|f32|packed)\b/.exec(fh.bankStore ?? "")?.[1] ?? null;
      fh.nets = { bank: store === "packed" ? "f32" : store, small: fh.cascade?.weights.precision ?? null, large: fh.weights?.precision ?? null };
      const pend = (k) => (fh.twinForms?.[k] ? " (its int8 twin measured on the first batch)" : "");
      log(`nets: bank ${fh.nets.bank}, small ${fh.nets.small}${pend("small")}, large ${fh.nets.large}${pend("large")}`);
      fh.probeKit = probe ? await Probe.create(fh) : null;
      lapBoot("probe");
      fh.back = back;
      fh.bh = null;
      fh.backShared = 0;
      fh.backOwned = null;
      if (back === "gpu") {
        // Its tables come from the wasm, which holds no other Focus here (formatTables frees its own).
        // ai: Every picture at its top, so a frame decodes the version its word (or the held one) names out of it.
        const sizes = tables.pictures.map(({ n, subch }) => (n <= backN ? { n, subch } : null));
        // ai: picture: the rings, where each puts a picture's samples and the lattice its Coons patch runs on, which the
        // ai: fused pass 1 samples by (on the frame's (ring, picture) pair).
        const picture = fuse ? tables.rings.map((t) => ({ span: t.span, margin: t.margin, lattice: t.nodeX })) : null;
        fh.backOpts = { sizes, precision: floatPrecision, cap: 0, variant: "i16", picture, cancel, log };
        // ai: tiers: a rate profile's text (back/tiers.mjs; the lab's rate-by-ring arm, 2026-10-07), its stage added to
        // ai: every back half built and run; its frames decode at its picture's slot, which then reads no other version
        fh.tiersText = tiers;
        // Built before any lane, since a lane's ensure() makes its BackHalf lane. Its shaders are compiled for B
        // frames.
        // ai: Under "auto" for B_CEIL, the lanes' size on every device whose memory allows it (the phone, the iGPU, the
        // ai: 4090), so the first frame's plan compiles nothing (2026-09-29: the plan compiled the back half twice on
        // ai: the first frame, at 1 to price a frame and at 32, and the S26's camera delivered almost nothing for
        // ai: 2.6 s; STATUS "The ramp at the start"). makeLanes compiles it again only for lanes memory holds under
        // ai: B_CEIL.
        await fh.buildBack(auto ? B_CEIL : B);
      }
      lapBoot("back");
      boot.total = Math.round(performance.now() - t0);
      fh.boot = boot;
      log(`built in ${boot.total} ms (device ${boot.device}, twins ${boot.twins}, front half ${boot.build}, probe ${boot.probe}, back half ${boot.back})`);
      fh.fuse = back === "gpu" && fuse;
      fh.cancel = back === "gpu" && cancel;
      fh.lanes = auto ? [] : Array.from({ length: fh.fixedInflight }, () => fh.lane());
      fh.planning = null;
      fh.planned = null;
      // ai: The ring comes with the first ready() (under B "auto" with the lanes' plan; at a fixed B at the frames' shape).
      fh.ring = null;
      return fh;
    } catch (e) {
      device.destroy();
      throw e;
    }
  }

  constructor(device, tables, B, features, log, cap) {
    Object.assign(this, { device, tables, B, features, log, cap });
    this.W = 0; this.H = 0; this.gridStride = 0;
    // Lanes read the decoder through their prototype; what they write for the decoder as a whole goes here.
    this.root = this;
    // When the last batch's readback arrived, for the device time of a batch on a device without timestamps.
    this.lastDone = 0;
    // Device memory this decoder holds, for the batch-size question: every buffer and texture it makes is counted
    // here and taken off when destroyed (once, however often destroy is called). While `sink` is set every one is
    // also listed there, so a lane or the back half can be destroyed whole.
    this.allocated = 0;
    this.sink = null;
    const d = device, mk = d.createBuffer.bind(d), mt = d.createTexture.bind(d), self = this;
    const counted = (o, bytes) => {
      self.allocated += bytes;
      self.sink?.push(o);
      const del = o.destroy.bind(o);
      let live = true;
      o.destroy = () => { if (live) { live = false; self.allocated -= bytes; } del(); };
      return o;
    };
    d.createBuffer = (desc) => counted(mk(desc), desc.size);
    d.createTexture = (desc) => counted(mt(desc), desc.size[0] * desc.size[1] * (desc.size[2] ?? 1));
  }

  // Frames the caller's next batch should carry, and how many batches it should keep on the device.
  get size() { return this.batcher ? this.batcher.size : this.B; }
  get inflight() { return this.batcher ? this.batcher.inflight : this.fixedInflight; }

  // The back half, its shaders compiled for B frames; its buffers (the shared tables) listed so it can go whole.
  // ai: The cancel stage's carry (one reference slot, shared by the lanes) is made here with it, counted in
  // ai: backShared and destroyed with it: a new back half starts with no carry.
  async buildBack(B, quiet = false) {
    this.destroyBack();
    const a0 = this.allocated;
    this.sink = this.backOwned = [];
    try {
      this.bh = await BackHalf.build(this.device, { B, ...this.backOpts, ...(quiet ? { log: () => {} } : {}) });
      if (this.tiersText) this.bh.useTiers((await this.bh.addTiers(this.tiersText)).key);
    } finally { this.sink = null; }
    this.backShared = this.allocated - a0;
  }

  destroyBack() {
    for (const b of this.backOwned ?? []) b.destroy();
    this.backOwned = null; this.bh = null; this.backShared = 0;
  }

  // Explicit layouts: `layout: "auto"` drops a binding the shader does not read, and then every bind group built
  // against the full list fails validation, which on some paths is silent (STATUS.md).
  layout(entries) {
    return this.device.createBindGroupLayout({ entries: entries.map((type, binding) => {
      const e = { binding, visibility: GPUShaderStage.COMPUTE };
      if (type === "tex") e.texture = { sampleType: "float", viewDimension: "2d-array" };
      else if (type === "uniform") e.buffer = { type: "uniform" };
      else e.buffer = { type: type === "ro" ? "read-only-storage" : "storage" };
      return e;
    }) });
  }

  // ai: No error scope (gpu/pipeline.mjs): build() has every one of these in flight at once.
  async pipeline(code, entries) {
    const bgl = this.layout(entries);
    const p = await computePipeline(this.device, { code, layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }) });
    return { p, bgl };
  }

  // ai: A classifier's kernel and weights buffer, from its packed weights (create): the large net (on every kept peak,
  // ai: or on the cascade's list) or, small, the cascade's first net (on every kept peak, writing its mark logits).
  // ai: { pipe, buf, w, plan }; pipe.P is the patches a workgroup of this kernel takes (its plan's), which a dispatch
  // ai: divides its slots by.
  async netForm(w, small) {
    const c = this.cascade, f = { f16: w.precision === "f16", int8: w.precision === "int8", subgroups: this.features.includes("subgroups") };
    const plan = classifyPlan(w.arch, f);
    const src = small ? classifySource(w.offsets, w.arch, { ...f, role: "first", rest: c.rest }) : classifySource(w.offsets, w.arch, { ...f, role: c ? "second" : null, keep: c?.keep });
    const pipe = { ...(await this.pipeline(src, [...LV, "ro", "ro", "rw", "uniform", "ro", ...(small ? ["rw"] : c ? ["ro"] : [])])), P: plan.P };
    const buf = this.device.createBuffer({ size: w.data.byteLength, usage: U.STORAGE | U.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, w.data);
    return { pipe, buf, w, plan };
  }

  // ai: A classifier's bind group on this lane (called on a lane, after ensure()): its kernel pipe over weights buf,
  // ai: small (the cascade's first net: the logits) or the large net (the cascade's list, when there is one).
  netGroup(pipe, buf, small) {
    const extra = small ? [this.logitBuf] : this.cascade ? [this.listBuf] : [];
    return this.device.createBindGroup({ layout: pipe.bgl, entries: [...this.lvEntries, ...[this.peaksBuf, this.countsBuf, this.readingsBuf, this.p4Uni, buf, ...extra].map((buffer, k) => ({ binding: 7 + k, resource: { buffer } }))] });
  }

  // ai: The lane's classifier bind groups over the nets' current forms: made by ensure(), and again once
  // ai: measureTwins has chosen.
  bindNets() {
    this.classifyGroup = this.netGroup(this.classify, this.weightsBuf, false);
    if (this.cascade) this.classify0Group = this.netGroup(this.classify0, this.weightsBuf0, true);
  }

  async build() {
    // ai: Every pipeline compiled at once (2026-09-29): one at a time, the S26's front half took 1.7 to 5.2 s of the GPU
    // ai: worker's 4.5 to 8 s start (STATUS "The ramp at the start"). F0: a frame given as a source (a VideoFrame) is
    // ai: cropped to luma on the device (ingest.mjs).
    const c = this.cascade, tf = this.twinForms, subgroups = this.features.includes("subgroups") ? " subgroups" : "";
    if (!BANKS.includes(this.bankName)) throw new Error(`bank ${this.bankName}: ${BANKS.join(", ")}`);
    const P = (code, entries) => this.pipeline(code, entries);
    // ai: The proposer is a module (gpu/bank_<name>.mjs). A device whose workgroup memory cannot hold it even with its
    // ai: activations stored as halves (under 18,692 B; WebGPU promises only 16 KB) gets its FitError: no GPU decoder
    // ai: there, and the receiver's C decodes (lizard-web/gpuqueue.mjs openGpu throws Unavailable).
    // ai: The default, v2, by a literal, so the installed app carries it under its hashed name (lizard-web/pwa/hash.mjs rewrites
    // ai: literal references only); another bank by its name, from the source tree alone (a harness's choice).
    const bank = (async () => { this.bank = await (this.bankName === "fcn2" ? import("./bank_fcn2.mjs") : import(`./bank_${this.bankName}.mjs`)); await this.bank.build(this); })();
    const [ingest, pyramid, sampler, large, small, rank, twinLarge, twinSmall, vote, gather, score, pick, nodesP, refit, wordP, select] = await Promise.all([
      Ingest.create(this.device),
      // ai: The pyramid and each frame's grey levels in one kernel (wgsl/pyramid.mjs): the layer, the frame records,
      // ai: levels 1 to 4, the histogram (read back with the batch for the receiver's stats row), the levels' dims.
      P(PYRAMID, ["tex", "ro", "rw", "rw", "rw", "rw", "rw", "uniform"]),
      P(SAMPLE, ["tex", "ro", "ro", "ro", "rw", "rw", "uniform", "ro", "ro", "ro", "uniform"]),
      this.netForm(this.weights, false),
      // The cascade: the small net on every kept peak (writing its mark logit), RANK, then the net above on the list.
      c ? this.netForm(c.weights, true) : null,
      c ? P(rankSource({ cap: this.cap, keep: c.keep }), ["ro", "ro", "rw", "rw"]) : null,
      // ai: The twins waiting to be measured, built beside the float forms: { pipe, buf, w, plan } a net.
      tf?.large ? this.netForm(tf.large, false) : null,
      tf?.small ? this.netForm(tf.small, true) : null,
      // ai: F5 (wgsl/finder.mjs): VOTE a lane a reading into the accumulator, then GATHER a workgroup a frame, its
      // ai: centre peaks (cand) and their quads.
      P(VOTE, ["ro", "ro", "rw", "uniform"]),
      P(GATHER, ["ro", "ro", "ro", "rw", "rw", "uniform"]),
      P(SCORE, [...LV, "ro", "ro", "rw", "uniform"]),
      P(PICK, ["ro", "ro", "ro", "rw", "rw", "rw", "uniform"]),
      P(NODES, [...LV, "ro", "ro", "ro", "ro", "rw", "uniform"]),
      P(REFIT, ["ro", "ro", "ro", "ro", "rw", "rw", "uniform", "rw"]),
      // ai: F8 (wgsl/word.mjs): the levels, then maps, sel (read-write: sel.y = 1 + the version the frame decodes at, sel.z
      // ai: its picture), resid, the word plan's points, the finder's results, the word code, the lattice and the lane's
      // ai: held version.
      P(WORD, [...LV, "ro", "rw", "ro", "ro", "rw", "uniform", "uniform", "uniform"]),
      P(SELECT, ["ro", "ro", "rw", "rw", "uniform"]),
      bank,
    ]);
    Object.assign(this, { ingest, pyramid, sampler, vote, gather, score, pick, nodesP, refit, wordP, select });
    this.classify = large.pipe;
    this.weightsBuf = large.buf;
    this.arch = large.plan.name;
    this.log(`classifier ${large.plan.name} ${this.weights.precision}${subgroups}: ${large.plan.macs} MACs a patch, ${large.plan.P} patches a workgroup, ${large.plan.bytes} bytes of workgroup memory`);
    if (c) {
      this.classify0 = small.pipe;
      this.weightsBuf0 = small.buf;
      this.rank = rank;
      this.arch = `${small.plan.name}>${large.plan.name} keep ${c.keep} rest ${c.rest}`;
      this.log(`cascade ${small.plan.name} ${c.weights.precision}${subgroups}: ${small.plan.macs} MACs a patch, ${small.plan.P} patches a workgroup, ${small.plan.bytes} bytes of workgroup memory; ${large.plan.name} on the ${c.keep} strongest a frame, the rest ${c.rest}`);
    }
    if (tf) {
      if (tf.large) tf.large = twinLarge;
      if (tf.small) tf.small = twinSmall;
      this.log("int8 twins built beside the float forms");
    }
    const d = this.device, rings = this.tables.rings, pictures = this.tables.pictures;
    // ai: F9's grids: every (ring, picture) pair at SLOTS ring + slot, (n, where sample 0 sits and how far apart
    // ai: samples are, in modules, 0), gpu/tables.mjs gridOf.
    const sz = new ArrayBuffer(16 * SLOTS * RING_COUNT), su = new Uint32Array(sz), sf = new Float32Array(sz);
    rings.forEach((t, r) => pictures.forEach(({ n }, p) => { const g = gridOf(t, n), k = 4 * (SLOTS * r + p); su[k] = n; sf[k + 1] = g.g0; sf[k + 2] = g.step; }));
    this.sizesBuf = d.createBuffer({ size: sz.byteLength, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.sizesBuf, 0, sz);
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
    // ai: PICK's uniform (wgsl/finder.mjs Pick): the accept line and the rings' module counts, the same every batch.
    const pickAb = new ArrayBuffer(32);
    new Float32Array(pickAb)[0] = TAU_TRACK; new Float32Array(pickAb).set(this.modules, 4);
    this.pickUni = d.createBuffer({ size: pickAb.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.pickUni, 0, pickAb);
    // ai: F7's tables. Per ring: the border nodes in ring order, each with its window of the border as painted
    // (wgsl/register.mjs), and the lattice's coordinates for the Coons patch.
    const nodes = [], tmpl = [], lattice = [], count = new Uint32Array(4), nbase = new Uint32Array(4), lat = new Uint32Array(16);
    rings.forEach((t, si) => {
      const nx = t.nodeX.length, M = t.modules;
      count[si] = 2 * nx + 2 * nx - 4; nbase[si] = nodes.length / 4;
      lat.set([lattice.length, nx, 0, 0], 4 * si);
      lattice.push(...t.nodeX);
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
    this.latticeBuf = d.createBuffer({ size: 4 * lattice.length, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.latticeBuf, 0, new Float32Array(lattice));
    this.latUni = d.createBuffer({ size: lat.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.latUni, 0, lat);
    // ai: F8's tables: the word plan's points (every ring's), the code, plan and picture uniform, the lattice uniform.
    const word = wordPoints(rings), wu = wordUniform(rings, word.plans, pictures), wl = wordLattice(rings);
    this.wordPtsBuf = d.createBuffer({ size: word.pts.byteLength, usage: U.STORAGE | U.COPY_DST });
    d.queue.writeBuffer(this.wordPtsBuf, 0, word.pts);
    this.wordUni = d.createBuffer({ size: wu.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.wordUni, 0, wu);
    this.wordLatUni = d.createBuffer({ size: wl.byteLength, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.wordLatUni, 0, wl);
    // Rounds of the node search: +-3 modules at half a module, one template sample a module; then +-0.6 at a
    // ai: fifth, all four.
    this.rounds = [[6, SUB * SUB, 0.5], [3, 1, 0.2]].map(([steps, stride, step]) => {
      const ab = new ArrayBuffer(64), u = new Uint32Array(ab), fl = new Float32Array(ab);
      u.set([steps, stride]); fl[4] = step; u.set(count, 8); u.set(nbase, 12);
      const b = d.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST });
      d.queue.writeBuffer(b, 0, ab);
      return b;
    });
    const fitAb = new ArrayBuffer(48);
    new Uint32Array(fitAb).set(count, 0); new Uint32Array(fitAb).set(nbase, 4); new Float32Array(fitAb).set(this.modules, 8);
    this.fitUni = d.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
    d.queue.writeBuffer(this.fitUni, 0, fitAb);
  }

  // One lane: the buffers a batch writes, sized for B frames, and the bind groups over them that do not depend on
  // the frame size (ensure() makes the rest). Every uniform a batch or ensure() writes is the lane's own, so lanes
  // at different frame sizes never see each other's values. Everything it makes is listed in `owned` (ensure()'s
  // too), so destroyLane() frees it whole.
  lane(B = this.B) {
    const ln = Object.create(this), d = this.device, cap = this.cap, a0 = this.allocated;
    ln.B = B;
    ln.owned = [];
    this.sink = ln.owned;
    ln.busy = false; ln.pending = null; ln.watched = false;
    ln.W = 0; ln.H = 0; ln.gridStride = 0;
    ln.tex = null; ln.rawBuf = null; ln.accBuf = null; ln.gridBuf = null; ln.gridUni = null; ln.readBuf = null; ln.sampleGroup = null; ln.backLane = null; ln.backPicture = null;
    // The back half's gate reads a frame record in each of the B slots it was compiled for, so the frame and sel
    // records cover those (16 and 8 bytes a slot) even where the lane holds fewer frames.
    ln.slotsG = Math.max(B, this.bh?.B ?? 0);
    ln.peaksBuf = d.createBuffer({ size: 16 * cap * B, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    ln.selUni = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    ln.framesBuf = d.createBuffer({ size: 16 * ln.slotsG, usage: U.STORAGE | U.COPY_DST });
    ln.mapsBuf = d.createBuffer({ size: 64 * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.countsBuf = d.createBuffer({ size: 4 * COUNTERS * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.readingsBuf = d.createBuffer({ size: 48 * cap * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.candBuf = d.createBuffer({ size: 16 * K * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.quadsBuf = d.createBuffer({ size: 16 * 5 * K * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.scoreBuf = d.createBuffer({ size: 4 * HYP * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.selBuf = d.createBuffer({ size: 4 * SEL_WORDS * ln.slotsG, usage: U.STORAGE | U.COPY_DST });
    ln.resultBuf = d.createBuffer({ size: 4 * RESULT * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    // ai: F8's held version for the lane's batch (wgsl/word.mjs Held), written with each batch.
    ln.heldUni = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
    ln.p5Uni = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
    ln.ldUni = d.createBuffer({ size: 80, usage: U.UNIFORM | U.COPY_DST });
    ln.measBuf = d.createBuffer({ size: 16 * NODES_MAX * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.residBuf = d.createBuffer({ size: 16 * NODES_MAX * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    ln.refitDbg = d.createBuffer({ size: 4 * REFIT_DBG * B, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    ln.histBuf = d.createBuffer({ size: 4 * HIST_BINS * B, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    if (this.cascade) {
      const probe = this.probeOn ? U.COPY_SRC : 0;
      ln.logitBuf = d.createBuffer({ size: 4 * cap * B, usage: U.STORAGE | probe });
      ln.listBuf = d.createBuffer({ size: 4 * this.cascade.keep * B, usage: U.STORAGE | probe });
    }
    ln.refitGroup = d.createBindGroup({ layout: this.refit.bgl, entries: [ln.framesBuf, this.nodesBuf, ln.selBuf, ln.measBuf, ln.mapsBuf, ln.residBuf, this.fitUni, ln.refitDbg]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.pickGroup = d.createBindGroup({ layout: this.pick.bgl, entries: [ln.framesBuf, ln.quadsBuf, ln.scoreBuf, ln.mapsBuf, ln.selBuf, ln.resultBuf, ln.pickUni]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.qs = null; ln.qsBuf = null;
    if (this.features.includes("timestamp-query")) {
      ln.qs = d.createQuerySet({ type: "timestamp", count: TS_PLAIN });
      ln.qsBuf = d.createBuffer({ size: 8 * TS_PLAIN, usage: U.QUERY_RESOLVE | U.COPY_SRC });
    }
    this.sink = null;
    // The lane's own device bytes; ensure() adds what it grows by (the BackHalf lane included).
    ln.laneBytes = this.allocated - a0;
    return ln;
  }

  // Called on a lane: frees everything it made. The query set is not device memory the budget counts.
  destroyLane() {
    for (const o of this.owned) o.destroy();
    this.owned = [];
    this.qs?.destroy();
    this.laneBytes = 0;
  }

  // Capacity for frames up to W x H and pictures up to the largest size any frame asks for. Grows, never shrinks.
  // grid: the batch runs F9 into the grid (the wasm back half, grids asked for, or no fused pass 1). Called on a lane.
  ensure(W, H, nmax, grid = true) {
    const d = this.device, B = this.B, a0 = this.allocated, root = this.root;
    root.sink = this.owned;
    if (W > this.W || H > this.H) {
      this.W = Math.max(W, this.W); this.H = Math.max(H, this.H);
      this.tex?.destroy();
      this.tex = d.createTexture({ size: [this.W, this.H, B], format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
      this.texView = this.tex.createView({ dimension: "2d-array" });
      this.level = [];
      for (let l = 1; l < LEVELS; l++) {
        const w = ceilDiv(this.W, 1 << l), h = ceilDiv(this.H, 1 << l);
        const buf = d.createBuffer({ size: 4 * w * h * B, usage: U.STORAGE | (this.probeOn ? U.COPY_SRC : 0) });
        this.level.push({ w, h, buf });
      }
      // The bank runs on level 0 (the texture) and levels 1 to 4 (the buffers). Each level's tiles have KEEP slots
      // apiece in the frame's slot array, level after level.
      const dimsAt = (l) => (l === 0 ? [this.W, this.H] : [this.level[l - 1].w, this.level[l - 1].h]);
      let slots = 0;
      const bases = [];
      for (let l = 0; l < LEVELS; l++) { const [w, h] = dimsAt(l); bases.push(slots); slots += ceilDiv(w, 16) * ceilDiv(h, 16) * this.bank.KEEP; }
      this.slots = slots;
      this.rawBuf?.destroy();
      this.rawBuf = d.createBuffer({ size: 16 * slots * B, usage: U.STORAGE | U.COPY_DST });
      const su0 = new ArrayBuffer(16);
      new Uint32Array(su0).set([slots, this.cap]); new Float32Array(su0)[2] = TAU;
      d.queue.writeBuffer(this.selUni, 0, su0);
    this.selGroup = d.createBindGroup({ layout: this.select.bgl, entries: [
      { binding: 0, resource: { buffer: this.framesBuf } }, { binding: 1, resource: { buffer: this.rawBuf } },
      { binding: 2, resource: { buffer: this.peaksBuf } }, { binding: 3, resource: { buffer: this.countsBuf } }, { binding: 4, resource: { buffer: this.selUni } }] });
      this.bank.ensure(this, dimsAt, bases, slots);
      // ai: The finder reads every level; its accumulator, one a ring, covers the largest frame in CELL-pixel cells.
      const ld = new Uint32Array(20);
      this.level.forEach((lv, i) => ld.set([lv.w, lv.h, 0, 0], 4 * (i + 1)));
      d.queue.writeBuffer(this.ldUni, 0, ld);
      this.gx = ceilDiv(this.W, CELL); this.gy = ceilDiv(this.H, CELL);
      this.accBuf?.destroy();
      this.accBuf = d.createBuffer({ size: 4 * RING_COUNT * this.gx * this.gy * B, usage: U.STORAGE | U.COPY_DST | (this.probeOn ? U.COPY_SRC : 0) });
      const p5 = new ArrayBuffer(32);
      new Uint32Array(p5).set([this.cap, this.gx, this.gy, 0]); new Float32Array(p5).set(this.modules, 4);
      d.queue.writeBuffer(this.p5Uni, 0, p5);
      const lvEntries = this.lvEntries = [{ binding: 0, resource: this.texView }, ...this.level.map((lv, i) => ({ binding: 1 + i, resource: { buffer: lv.buf } })),
        { binding: 5, resource: { buffer: this.framesBuf } }, { binding: 6, resource: { buffer: this.ldUni } }];
      const group = (pl, extra, first = 0) => d.createBindGroup({ layout: pl.bgl, entries: [...(first ? lvEntries : []), ...extra.map((buffer, k) => ({ binding: first + k, resource: { buffer } }))] });
      this.bindNets();
      if (this.cascade) this.rankGroup = group(this.rank, [this.framesBuf, this.logitBuf, this.countsBuf, this.listBuf]);
      this.scoreGroup = group(this.score, [this.quadsBuf, this.planBuf, this.scoreBuf, this.planUni], 7);
      this.voteGroup = group(this.vote, [this.framesBuf, this.readingsBuf, this.accBuf, this.p5Uni]);
      this.gatherGroup = group(this.gather, [this.framesBuf, this.readingsBuf, this.accBuf, this.candBuf, this.quadsBuf, this.p5Uni]);
      this.nodesGroups = this.rounds.map((r) => group(this.nodesP, [this.nodesBuf, this.tmplBuf, this.mapsBuf, this.selBuf, this.measBuf, r], 7));
      this.wordGroup = group(this.wordP, [this.mapsBuf, this.selBuf, this.residBuf, this.wordPtsBuf, this.resultBuf, this.wordUni, this.wordLatUni, this.heldUni], 7);
      this.pyrGroup = d.createBindGroup({ layout: this.pyramid.bgl, entries: [{ binding: 0, resource: this.texView }, { binding: 1, resource: { buffer: this.framesBuf } },
        ...this.level.map((lv, i) => ({ binding: 2 + i, resource: { buffer: lv.buf } })), { binding: 6, resource: { buffer: this.histBuf } }, { binding: 7, resource: { buffer: this.ldUni } }] });
      this.sampleGroup = null;
    }
    // ai: gridStride is the picture capacity a frame, which F9 and the back half's params are sized by (F9 samples no
    // ai: picture past it); the grid buffer (4 gridStride B bytes, 33.5 MB at nmax 1024 and B 8) exists only once a
    // ai: batch runs F9.
    if (nmax * nmax > this.gridStride) {
      this.gridStride = nmax * nmax;
      this.gridBuf?.destroy();
      this.gridBuf = null;
      this.gridUni?.destroy();
      this.gridUni = d.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST });
      d.queue.writeBuffer(this.gridUni, 0, new Uint32Array([this.gridStride, 0, 0, 0]));
      this.sampleGroup = null;
      this.readBuf?.destroy();
      this.readBuf = null;
      if (this.bh) {
        if (this.backLane) this.bh.destroyLane(this.backLane);
        // ai: backAt, backBytes: which of the lane's owned buffers are the BackHalf lane's (it holds bh.B frames
        // ai: whatever this lane's B), and their bytes, for plan() to price a frame.
        const o0 = this.owned.length, b0 = root.allocated;
        this.backLane = this.bh.lane({ gridBuf: null, gridStride: this.gridStride, framesBuf: this.framesBuf, selBuf: this.selBuf });
        this.backAt = [o0, this.owned.length]; this.backBytes = root.allocated - b0;
        this.backPicture = null;
      }
    }
    if (grid && !this.gridBuf) {
      this.gridBuf = d.createBuffer({ size: 4 * this.gridStride * B, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      this.sampleGroup = null;
      if (this.bh) this.bh.bindGrid(this.backLane, this.gridBuf);
    }
    // The fused pass 1's bind groups hold the texture's view, which the W, H branch above replaces.
    if (this.fuse && this.backPicture !== this.texView) {
      this.bh.bindPicture(this.backLane, { texView: this.texView, mapsBuf: this.mapsBuf, residBuf: this.residBuf });
      this.backPicture = this.texView;
    }
    if (this.gridBuf && !this.sampleGroup) {
      this.sampleGroup = d.createBindGroup({ layout: this.sampler.bgl, entries: [
        { binding: 0, resource: this.texView }, { binding: 1, resource: { buffer: this.framesBuf } }, { binding: 2, resource: { buffer: this.mapsBuf } },
        { binding: 3, resource: { buffer: this.sizesBuf } }, { binding: 4, resource: { buffer: this.gridBuf } }, { binding: 5, resource: { buffer: this.countsBuf } },
        { binding: 6, resource: { buffer: this.gridUni } }, { binding: 7, resource: { buffer: this.selBuf } },
        { binding: 8, resource: { buffer: this.residBuf } }, { binding: 9, resource: { buffer: this.latticeBuf } }, { binding: 10, resource: { buffer: this.latUni } }] });
    }
    if (!this.readBuf) {
      // Timestamps last. With the GPU back half the grid and the peaks stay on the device: the counters, the finder's
      // results and the back half's readback (bh.readBytes, a multiple of 8) come back instead.
      // ai: Plus each frame's grey-level histogram (HIST_BINS u32) on either path, and a ring pair a slot at the tail.
      const bytes = this.bh ? 4 * COUNTERS * this.B + 4 * RESULT * this.B + 4 * HIST_BINS * this.B + this.bh.readBytes + 8 * TS_MAX + 8 * TS_INGEST * this.B
        : 4 * this.gridStride * this.B + 4 * COUNTERS * this.B + 16 * this.cap * this.B + 4 * RESULT * this.B + 4 * HIST_BINS * this.B + 8 * TS_MAX + 8 * TS_INGEST * this.B;
      this.readBuf = d.createBuffer({ size: bytes, usage: U.MAP_READ | U.COPY_DST });
    }
    root.sink = null;
    this.laneBytes += this.allocated - a0;
  }

  // The capacity a batch needs: its largest frame, the largest picture it may sample, and whether F9 fills a grid.
  // ai: The picture is the word's to name (F8), found or told, so the grid holds up to nmaxFind whatever the frames.
  shapeOf(frames, maps, nmaxFind = 1024, grids = false) {
    const W = Math.max(1, ...frames.map((f) => (f ? f.w : 0))), H = Math.max(1, ...frames.map((f) => (f ? f.h : 0)));
    const nmax = nmaxFind;
    // With the fused pass 1 the back half samples each picture itself, so F9 and the grid run only for the wasm back
    // half or a batch that reads the grids back.
    const fused = !!this.fuse && !grids;
    return { W, H, nmax, fused, grid: !fused };
  }

  // B "auto": does this batch need lanes planned first (none yet, or a larger frame, picture or a grid)?
  // ai: At a fixed B, whether the ring is missing or too small for it. frames may be slots (only w and h are read).
  needsPlan(frames, maps, { nmaxFind = 1024, grids = false } = {}) {
    const p = this.planned, s = this.shapeOf(frames, maps, nmaxFind, grids), r = this.ring;
    if (!this.auto) return !r || s.W > r.W || s.H > r.H;
    return !p || s.W > p.W || s.H > p.H || s.nmax > p.nmax || (s.grid && !p.grid);
  }

  // B "auto": the lanes planned for every shape seen so far and this batch's, and the batch size set to theirs.
  // Waits for every lane to be idle, and frees them, first.
  // ai: Only the frames' shapes are read ({ w, h }, luma or slots): nothing is decoded or compiled here since 2026-09-29
  // ai: (the back half is built at create, B_CEIL), so a plan on the first frame is buffers alone. The ring is freed
  // ai: with the lanes and made again with them.
  async plan(frames, maps, opts = {}) {
    const t0 = performance.now(), bt = this.batcher, d = this.device, L = d.limits, s = this.shapeOf(frames, maps, opts.nmaxFind ?? 1024, !!opts.grids);
    const p = this.planned ?? { W: 0, H: 0, nmax: 0, grid: false };
    const W = Math.max(p.W, s.W), H = Math.max(p.H, s.H), nmax = Math.max(p.nmax, s.nmax), grid = p.grid || s.grid;
    await Promise.allSettled(this.lanes.map((l) => l.pending));
    for (const l of this.lanes) l.destroyLane();
    this.lanes = [];
    this.destroyRing();
    // A frame's price: a one-frame lane at this shape. Its largest buffer times B must stay under the device's buffer
    // limits, its texture's layers under the layer limit.
    // ai: Its BackHalf lane holds the built back half's bh.B frames, so that part (and the readback buffer's share of the
    // ai: back half's readback, bh.readBytes) is priced a frame as its bytes over bh.B: a lane's few fixed buffers shared
    // ai: out, where a one-frame back half counted them whole. Where the budget then holds fewer frames than bh.B, the
    // ai: back half is built at that size and the frame priced again with it, until the two agree (a device whose memory
    // ai: holds under B_CEIL: SwiftShader, the 128 MiB fallback); a probe the device has no memory for halves it the same
    // ai: way. Plus a ring slot, one layer at this shape (slotBytes).
    const MB = (v) => (v / MiB).toFixed(1);
    for (;;) {
      const backShared = this.backShared, front = this.allocated - backShared, Bb = this.bh?.B ?? 1;
      d.pushErrorScope("out-of-memory");
      const probe = this.lane(1);
      probe.ensure(W, H, nmax, grid);
      // ai: The readback buffer holds the back half's readback for its bh.B frames too: shared out the same.
      const [o0, o1] = probe.backAt ?? [0, 0], readBack = this.bh?.readBytes ?? 0, backBytes = (probe.backBytes ?? 0) + readBack;
      const size = (o, i) => (o === probe.readBuf ? o.size - readBack + readBack / Bb : (o.size ?? 0) / (i >= o0 && i < o1 ? Bb : 1));
      // BACK=both reads the grids back through a buffer made for the batch, which the lane does not hold.
      const perFrame = probe.laneBytes - backBytes + backBytes / Bb + (opts.grids && this.bh ? 4 * nmax * nmax : 0);
      const largest = Math.max(...probe.owned.map(size));
      probe.destroyLane();
      const oom = await d.popErrorScope();
      if (oom) {
        if (!this.bh || Bb === 1) throw new Error(`out of memory for a one-frame lane: ${oom.message}`);
        await this.buildBack(Bb >> 1);
        continue;
      }
      let cap = Infinity, capWhy = null;
      const byBuffer = Math.floor(Math.min(L.maxBufferSize, L.maxStorageBufferBindingSize) / largest);
      if (byBuffer < cap) { cap = byBuffer; capWhy = `the device's buffer size limit (a frame's largest buffer ${MB(largest)} MB)`; }
      if (L.maxTextureArrayLayers < cap) { cap = L.maxTextureArrayLayers; capWhy = "the device's texture layer limit"; }
      bt.sized({ perFrame, shared: front + backShared, room: bt.budget - front - backShared, cap, capWhy, slotBytes: W * H });
      const { Bmax } = bt.limitFor(bt.lanes);
      if (!this.backOpts || Bmax >= Bb) break;
      await this.buildBack(Bmax);
    }
    this.planned = { W, H, nmax, grid };
    await this.makeLanes(bt.lanes);
    bt.start();
    bt.planMs = performance.now() - t0;
    this.log(`batch auto: planned in ${bt.planMs.toFixed(0)} ms`);
  }

  // ai: create's twins "measure": each classifier with an int8 twin timed in both forms and the faster kept (the S26
  // ai: Ultra's Adreno ran the int8 classifiers at 8.6 times f16's describe time where the iGPU ran them at half: a
  // ai: measurement, never the device's name). On the first stream batch a lane reads back with a live frame first
  // ai: (batch(), 2026-09-29; a measuring batch of the planning frame before): its first nb (up to TWIN_FRAMES) frames'
  // ai: levels, peaks, logits and list are still on that lane, which is busy until this returns; then timeArms. The
  // ai: fastest round a form (interference only adds); int8 kept only when faster. Once a decoder: the loser's weights
  // ai: are freed, every lane's groups remade (a batch already submitted keeps the ones it was encoded with; none is
  // ai: encoded across an await), fh.nets and fh.twins set. A measurement that fails on a live device is logged and
  // ai: the float forms kept.
  async measureTwins(lane, nb) {
    const tf = this.twinForms, c = this.cascade, arms = [];
    this.twinForms = null;
    // ai: Without timestamps a dispatch is timed from its submit to its completion, so no other lane's batch may be on
    // ai: the device meanwhile: the other lanes are reserved (launch() leaves a reserved lane busy when its batch
    // ai: settles) and waited for. With timestamps a pass times itself, whatever else the queue holds.
    const held = this.features.includes("timestamp-query") ? [] : this.lanes.filter((l) => l !== lane);
    for (const l of held) { l.reserved = true; l.busy = true; }
    await Promise.allSettled(held.map((l) => l.pending));
    try {
      for (const net of ["small", "large"]) {
        if (!tf[net]) continue;
        const small = net === "small", slots = small || !c ? this.cap : c.keep;
        const now = small ? { pipe: this.classify0, buf: this.weightsBuf0, precision: c.weights.precision } : { pipe: this.classify, buf: this.weightsBuf, precision: this.weights.precision };
        for (const f of [now, { ...tf[net], precision: "int8" }]) arms.push({ net, precision: f.precision, pipe: f.pipe, group: lane.netGroup(f.pipe, f.buf, small), x: ceilDiv(slots, f.pipe.P), ms: [] });
      }
      await this.timeArms(arms, nb);
    } catch (e) {
      if (this.lost) throw e;
      this.log(`int8 twins not measured (${e.message}): the float files`);
      for (const k of ["small", "large"]) tf[k]?.buf.destroy();
      return;
    } finally { for (const l of held) { l.reserved = false; l.busy = false; } }
    const best = (v) => Math.min(...v) / nb;
    const out = { frames: nb, by: this.features.includes("timestamp-query") ? "timestamps" : "submits" };
    for (const net of ["small", "large"]) {
      const [fl, i8] = arms.filter((a) => a.net === net);
      if (!fl) continue;
      const keep = best(i8.ms) < best(fl.ms) ? i8 : fl, t = tf[net];
      out[net] = { [fl.precision]: +best(fl.ms).toFixed(3), int8: +best(i8.ms).toFixed(3), kept: keep.precision };
      if (keep === i8) {
        if (net === "small") { this.weightsBuf0.destroy(); this.classify0 = t.pipe; this.weightsBuf0 = t.buf; c.weights = t.w; }
        else { this.weightsBuf.destroy(); this.classify = t.pipe; this.weightsBuf = t.buf; this.weights = t.w; }
      } else t.buf.destroy();
      this.nets[net] = keep.precision;
    }
    this.twins = out;
    for (const l of this.lanes) l.bindNets();
    const said = (net) => (out[net] ? `${net} ${out[net].kept} (${Object.entries(out[net]).filter(([k]) => k !== "kept").map(([k, v]) => `${k} ${v}`).join(", ")} ms a frame)` : null);
    this.log(`int8 twins measured on ${nb} frames by ${out.by}: ${["small", "large"].map(said).filter(Boolean).join("; ")}`);
    this.log(`nets: bank ${this.nets.bank}, small ${this.nets.small}, large ${this.nets.large}`);
  }

  // ai: measureTwins' timing: TWIN_ROUNDS rounds of each arm's describe dispatch ({ pipe, group, x } over nb frames
  // ai: of the lane), the order reversed every other round, the first round untimed; each arm's ms pushed onto
  // ai: arm.ms. By timestamp pairs, a pass an arm, in one command buffer where the device has them; else each
  // ai: dispatch alone from its submit to its completion (the wait's round trip in every arm alike).
  async timeArms(arms, nb) {
    const d = this.device, order = (r) => (r % 2 ? [...arms].reverse() : arms);
    const qs = this.features.includes("timestamp-query") ? d.createQuerySet({ type: "timestamp", count: 2 * arms.length * TWIN_ROUNDS }) : null;
    const pass = (enc, a, k) => {
      const p = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } } : {});
      p.setPipeline(a.pipe.p); p.setBindGroup(0, a.group); p.dispatchWorkgroups(a.x, nb); p.end();
    };
    d.pushErrorScope("validation");
    let err = null;
    try {
      if (qs) {
        const enc = d.createCommandEncoder(), res = d.createBuffer({ size: 16 * arms.length * TWIN_ROUNDS, usage: U.QUERY_RESOLVE | U.COPY_SRC });
        const read = d.createBuffer({ size: res.size, usage: U.MAP_READ | U.COPY_DST });
        try {
          for (let r = 0; r < TWIN_ROUNDS; r++) for (const a of order(r)) pass(enc, a, r * arms.length + arms.indexOf(a));
          enc.resolveQuerySet(qs, 0, qs.count, res, 0);
          enc.copyBufferToBuffer(res, 0, read, 0, res.size);
          d.queue.submit([enc.finish()]);
          await read.mapAsync(GPUMapMode.READ);
          const t = new BigUint64Array(read.getMappedRange().slice(0));
          read.unmap();
          for (let r = 1; r < TWIN_ROUNDS; r++) arms.forEach((a, i) => { const k = r * arms.length + i; a.ms.push(Number(t[2 * k + 1] - t[2 * k]) / 1e6); });
        } finally { read.destroy(); res.destroy(); qs.destroy(); }
      } else {
        for (let r = 0; r < TWIN_ROUNDS; r++) for (const a of order(r)) {
          const enc = d.createCommandEncoder(), t0 = performance.now();
          pass(enc, a, 0);
          d.queue.submit([enc.finish()]);
          await d.queue.onSubmittedWorkDone();
          if (r) a.ms.push(performance.now() - t0);
        }
      }
    } finally { err = await d.popErrorScope(); }
    if (err) throw new Error(err.message);
  }

  // B "auto": n lanes, each for the largest batch the budget allows over n (plan() priced a frame), and the ring
  // beside them, made once every lane is idle. An allocation that fails is tried again at half.
  async makeLanes(n) {
    const bt = this.batcher, d = this.device, p = this.planned;
    await Promise.allSettled(this.lanes.map((l) => l.pending));
    for (const l of this.lanes) l.destroyLane();
    this.lanes = [];
    let { Bmax, why } = bt.limitFor(n);
    for (;;) {
      d.pushErrorScope("out-of-memory");
      if (this.backOpts && this.bh?.B !== Bmax) await this.buildBack(Bmax);
      this.B = Bmax;
      this.lanes = Array.from({ length: n }, () => this.lane(Bmax));
      for (const l of this.lanes) l.ensure(p.W, p.H, p.nmax, p.grid);
      this.makeRing(Bmax + RING_SLACK, p.W, p.H);
      const err = await d.popErrorScope();
      if (!err) break;
      for (const l of this.lanes) l.destroyLane();
      this.lanes = [];
      this.destroyBack();
      this.destroyRing();
      if (Bmax === 1) throw new Error(`out of memory for lanes of one frame: ${err.message}`);
      why = `an allocation failed at ${Bmax} (${err.message})`;
      Bmax = Math.floor(Bmax / 2);
    }
    bt.laned(n, Bmax, why);
  }

  // ai: The ring: R layers at W x H, a timestamp pair a slot where the device has them. Made with the lanes (a plan,
  // ai: a new device) or, at a fixed B, grown with the batches' shape; a rebuild frees every staged slot, and run()
  // ai: refuses a slot of an older ring. Never inside a lane's sink: it is shared, and outlives any lane.
  makeRing(R, W, H) {
    const d = this.device, ts = this.features.includes("timestamp-query");
    this.destroyRing();
    const tex = d.createTexture({ size: [W, H, R], format: "r8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    this.ring = { R, W, H, tex, views: Array.from({ length: R }, (_, id) => tex.createView({ dimension: "2d", baseArrayLayer: id, arrayLayerCount: 1 })),
      slots: new Array(R).fill(null), head: 0, staged: 0,
      qs: ts ? d.createQuerySet({ type: "timestamp", count: TS_INGEST * R }) : null,
      qsBuf: ts ? d.createBuffer({ size: 8 * TS_INGEST * R, usage: U.QUERY_RESOLVE | U.COPY_SRC }) : null };
  }

  destroyRing() {
    const r = this.ring;
    if (!r) return;
    r.tex.destroy(); r.qs?.destroy(); r.qsBuf?.destroy();
    this.ring = null;
  }

  // B "auto": resolves once the lanes suit this batch (planned for its shape). A caller that sizes
  // its batches by `size` calls it before each.
  // ai: Only the frames' shapes are read: luma ({ w, h, bytes }), slots, or { w, h }. At a fixed B: the ring made,
  // ai: or grown, to the frames' shape, at once.
  ready(frames = null, maps = null, opts = {}) {
    if (!this.auto) {
      if (frames && this.needsPlan(frames, maps, opts)) {
        const s = this.shapeOf(frames, maps, opts.nmaxFind ?? 1024, !!opts.grids), r = this.ring;
        this.makeRing(this.B + RING_SLACK, Math.max(s.W, r?.W ?? 0), Math.max(s.H, r?.H ?? 0));
      }
      return Promise.resolve();
    }
    if (!this.planning && frames && this.needsPlan(frames, maps, opts)) this.planning = this.plan(frames, maps, opts).finally(() => { this.planning = null; });
    return this.planning ?? Promise.resolve();
  }

  // ai: A frame onto the ring: { w, h, bytes } (w * h luma, writeTexture) or { w, h, source, x, y } (F0: the crop
  // ai: (x, y, w, h) of source, a VideoFrame, to luma on the device, ingest.mjs), with `ring` (an index into
  // ai: tables.rings) when told. On the device before this returns (its pass or write is submitted), so the caller closes
  // ai: the VideoFrame at once. Returns the slot, or null when every slot is staged (the caller chooses what to
  // ai: drop). Throws on a crop outside its source, bytes not w x h, a frame the ring is too small for (ready()
  // ai: plans a larger one), or no ring yet (ready() first).
  enqueue(frame) {
    const r = this.ring, d = this.device, { w, h, source, bytes } = frame;
    if (!r) throw new Error("no ring: ready() before enqueue()");
    if (source) Ingest.check(frame);
    else if (!bytes || bytes.length !== w * h) throw new Error(`${bytes?.length ?? "no"} bytes for a ${w} x ${h} frame`);
    if (w > r.W || h > r.H) throw new Error(`${w} x ${h}: the ring holds ${r.W} x ${r.H}`);
    let id = -1;
    for (let k = 0; k < r.R && id < 0; k++) if (!r.slots[(r.head + k) % r.R]) id = (r.head + k) % r.R;
    if (id < 0) return null;
    r.head = (id + 1) % r.R;
    if (source) {
      // ai: One pass, one submit a frame; a validation error is logged, not thrown: the frame reads as empty and
      // ai: the batch goes on (a bad crop was refused above, with no scope open).
      const enc = d.createCommandEncoder();
      d.pushErrorScope("validation");
      try {
        this.ingest.encode(enc, [{ source, x: frame.x ?? 0, y: frame.y ?? 0, w, h, view: r.views[id] }], r.qs ? { querySet: r.qs, index: TS_INGEST * id } : null);
        d.queue.submit([enc.finish()]);
      } finally { d.popErrorScope().then((e) => { if (e) this.log(`enqueue: ${e.message}`); }, () => {}); }
    } else d.queue.writeTexture({ texture: r.tex, origin: [0, 0, id] }, bytes, { bytesPerRow: w, rowsPerImage: h }, [w, h, 1]);
    const slot = new Slot(id, w, h, !!source, frame.ring);
    r.slots[id] = slot; r.staged++;
    return slot;
  }

  // ai: Frees a staged slot undecoded (the queue drops the oldest when the ring is full). A slot already freed, or
  // ai: of a ring since rebuilt, is left alone: batch() frees at its submit, and a failed run() releases once more.
  release(slot) {
    const r = this.ring;
    if (r && r.slots[slot.id] === slot) { r.slots[slot.id] = null; r.staged--; }
  }

  // ai: frames: up to B of { w, h, ring (index into tables.rings, when told), bytes (w * h luma) }; a null slot is
  // ai: empty. Or a slot from enqueue(), or { w, h, ring, source, x, y } enqueued here on the spot (see enqueue: a
  // ai: VideoFrame's source is imported before run() returns, so close it then).
  // maps: per frame 16 floats (wgsl/common.mjs MAP), module coordinates to image pixels, until the finder makes them.
  // ai: With maps given (a frame's 16 floats, or null) the host's maps and told rings are read (F8, then the picture
  // ai: its word or the held version names); with maps null the finder makes both.
  // ai: held (opts): the version of the last word the caller read, 0 for none (the held configuration, wgsl/word.mjs):
  // ai: a frame of this batch whose own word does not read decodes at it, and with none decodes nothing. The caller
  // ai: keeps it from batch to batch (lizard-web/recv-gpu-worker.mjs, scripts/gpu/harness/page.mjs), the one state that does.
  // peaks: per frame a Float32Array of 4 n (x, y, sigma, response) or null. Given, they are F3's output for the
  // batch: the bank and select do not run, and F4 reads these (scripts/gpu/cnn/check.mjs feeds known patches this way).
  // profile: one compute pass per stage group with its own timestamps, so a slow device says which stage.
  //
  // Called without await, the upload, the encoding and the submit are done when it returns (an async function runs
  // to its first await); the promise settles once the batch is read back. Batches take the lanes in turn, so a
  // caller holding up to `inflight` unsettled promises keeps that many batches on the device, and the readback of
  // one never waits on the submit of the next. A batch with no free lane is refused, not queued.
  // Under B "auto" a batch that needs lanes planned (the first, or a larger shape) waits for that before its upload;
  // so does every batch the caller submits meanwhile. A batch carries at most B frames (the lanes' size); `size` is
  // what the controller wants the next to carry.
  // ai: The upload is the frames onto the ring (enqueue) and a copy a slot into the lane. Staged slots cannot wait
  // ai: for a plan (it rebuilds the ring), so such a batch is refused: ready() on the frame's shape, then enqueue again.
  run(frames, maps, opts = {}) {
    const staged = frames.some((f) => f instanceof Slot);
    if (this.planning || this.needsPlan(frames, maps, opts)) {
      if (staged) throw new Error("staged slots cannot wait for a plan (it rebuilds the ring): ready() first, then enqueue again");
      if (this.auto) return this.ready(frames, maps, opts).then(() => this.run(frames, maps, opts));
      this.ready(frames, maps, opts);
    }
    if (frames.length > this.B) throw new Error(`${frames.length} frames in a batch of at most ${this.B}`);
    const lane = this.lanes.find((l) => !l.busy);
    if (!lane) throw new Error(`${this.lanes.length} batches in flight already`);
    const slots = [], mine = [];
    try {
      for (const f of frames) {
        if (!f || !(f.w > 0 && f.h > 0)) { slots.push(null); continue; }
        const s = f instanceof Slot ? f : this.enqueue(f);
        if (!s) throw new Error(`the ring is full: ${this.ring.staged} of ${this.ring.R} slots staged`);
        if (s !== f) mine.push(s);
        slots.push(s);
      }
    } catch (e) { for (const s of mine) this.release(s); throw e; }
    return this.launch(lane, slots, maps, opts);
  }

  launch(lane, slots, maps, opts) {
    lane.busy = true;
    const p = lane.batch(slots, maps, opts).finally(() => {
      lane.busy = !!lane.reserved;   // ai: a lane measureTwins holds stays out of the queue's reach until it lets go
      lane.pending = null;
    });
    // ai: A batch that fails before its submit leaves its slots staged; one that fails after it freed them already.
    p.catch(() => { for (const s of slots) if (s) this.release(s); });
    lane.pending = p.catch(() => {});
    return p;
  }

  // grids: with the GPU back half, read the sampled grids back as well, a diagnostic that holds the two back halves to
  // the same grids (scripts/exp/gpu_captures.mjs BACK=both); never on a timed path.
  // ai: probe: every stage's buffers read back too, as out.probe (probe.mjs; a decoder made with probe: true).
  // A batch carries nb = slots.length slots (at most the lane's B): every dispatch, clear and readback covers those.
  // ai: With the cancel stage the batch's pass two (back/cancel.mjs, back/DESIGN.md 13.3) follows pass one on this
  // ai: lane in this command buffer, its carry in (the batch before's last reference) at the encoder's head; its
  // ai: records, and each frame's cancelled, refs and gained, come back in the batch's own readback.
  async batch(slots, maps, { debug = false, nmaxFind = 1024, peaks: inject = null, profile = false, grids = false, probe = false, held = 0 } = {}) {
    const d = this.device, B = this.B, nb = slots.length, root = this.root, ring = root.ring;
    if (nb > B) throw new Error(`${nb} frames in a batch of ${B}`);
    if (!(held >= 0 && held <= VERSION_MAX && held === Math.floor(held))) throw new Error(`held ${held}: a version 1 to ${VERSION_MAX}, or 0 for none`);
    if (maps) slots.forEach((s, i) => { if (s && maps[i] && !(s.ring >= 0 && s.ring < RING_COUNT)) throw new Error(`a told frame names ring ${s.ring}: 0 to ${RING_COUNT - 1}`); });
    if (!nb) throw new Error("a batch of no slots");
    if (probe && !this.probeKit) throw new Error("probe: the decoder was not made with probe: true");
    for (const s of slots) if (s && ring.slots[s.id] !== s) throw new Error(`slot ${s.id}: freed already, or of a ring since rebuilt`);
    // The host's side of the batch, in ms: upload (the frames into the texture), encode (to the submit), scopes
    // (the error scopes' round trip), map (the readback's wait) and read (copying out of the mapped range).
    // ai: upload is now the slots' copies and the frame records: the frames went onto the ring at enqueue.
    const host = {}, t = [performance.now()];
    const lap = (name) => { t.push(performance.now()); host[name] = t[t.length - 1] - t[t.length - 2]; };
    const live = slots.map((s) => !!s);
    const { W, H, nmax, fused } = this.shapeOf(slots, maps, nmaxFind, grids);
    // Errors are caught over the whole batch (buffers, bind groups, the submit) rather than only logged: a batch
    // that failed validation or ran out of memory would otherwise read back stale zeros and score as "nothing found".
    d.pushErrorScope("out-of-memory");
    d.pushErrorScope("validation");
    this.ensure(W, H, nmax, !fused);
    const find = !maps;
    const G = this.slotsG;
    d.queue.writeBuffer(this.heldUni, 0, heldUniform(held));
    // Every slot's frame record is written, so a slot past nb (or a gate slot past B) reads as no frame.
    // ai: A told frame's sel is (its ring, 1, 0, 0): F8 then names its version and picture as a found frame's.
    const dims = new Uint32Array(4 * G), mp = new Float32Array(16 * B), sel = new Uint32Array(SEL_WORDS * G);
    slots.forEach((s, i) => {
      if (!s) return;
      dims.set([s.w, s.h, 0, 1], 4 * i);
      if (!find && maps[i]) { mp.set(maps[i], 16 * i); sel.set([s.ring, 1, 0, 0], SEL_WORDS * i); }
    });
    d.queue.writeBuffer(this.framesBuf, 0, dims);
    if (!find) { d.queue.writeBuffer(this.mapsBuf, 0, mp); d.queue.writeBuffer(this.selBuf, 0, sel); }
    if (inject) {
      // Whole buffers, in place of the clears below: a queue write lands before the command buffer's clear would.
      const pk = new Float32Array(4 * this.cap * B), cnt = new Uint32Array(COUNTERS * B);
      inject.forEach((p, i) => {
        if (!p || !live[i]) return;
        if (p.length > 4 * this.cap) throw new Error(`${p.length / 4} peaks injected, cap ${this.cap}`);
        pk.set(p, 4 * this.cap * i);
        cnt[COUNTERS * i + COUNT.peaks] = p.length / 4;
      });
      d.queue.writeBuffer(this.peaksBuf, 0, pk);
      d.queue.writeBuffer(this.countsBuf, 0, cnt);
    }
    const enc = d.createCommandEncoder();
    // ai: The cancel stage's carry in (the shared carry into this lane's REF slot R): a copy, so on the encoder
    // ai: before the compute pass; no stage before pass two reads REF.
    if (this.cancel) this.bh.prepareCancel(enc, this.backLane);
    // Stage groups for the profile: each gets its own pass and timestamp pair; otherwise one pass, one pair.
    // F7 is timed a dispatch at a time (nodes and refit, each round), so the split is on record with every profile.
    const STAGES = ["pyramid", "bank", "select", "describe", "vote", "score", ...this.rounds.flatMap((_, r) => [`nodes${r + 1}`, `refit${r + 1}`]), "word", ...(fused ? [] : ["sample"]), ...(this.bh ? [...BACK_PASSES, ...(this.cancel ? CANCEL_PASSES : [])] : [])];
    if (profile && this.qs && this.qs.count < 2 * STAGES.length) { this.qs.destroy(); this.qs = d.createQuerySet({ type: "timestamp", count: 2 * STAGES.length }); this.qsBuf.destroy(); this.qsBuf = d.createBuffer({ size: 8 * 2 * STAGES.length, usage: U.QUERY_RESOLVE | U.COPY_SRC }); }
    const pair = (k) => (this.qs ? { timestampWrites: { querySet: this.qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } } : {});
    // ai: Each slot's crop from its ring layer into the lane's layer i (1.2 MB at 1080, about 0.1 ms a batch): no
    // ai: stage reads the ring. Outside w x h the layer keeps what its last frame left, as writeTexture left it; every
    // ai: stage clamps its reads to the frame record.
    slots.forEach((s, i) => { if (s) enc.copyTextureToTexture({ texture: ring.tex, origin: [0, 0, s.id] }, { texture: this.tex, origin: [0, 0, i] }, [s.w, s.h, 1]); });
    lap("upload");
    // Each of these holds B equal frame strides, and only the batch's nb are dispatched or read, so only they are cleared.
    const clear = (b) => enc.clearBuffer(b, 0, (b.size / B) * nb);
    if (!inject) { clear(this.countsBuf); clear(this.peaksBuf); }
    // The back half reads only the frames its gate lists, each sampled whole this batch, so its grid needs no clear.
    if (!this.bh) clear(this.gridBuf);
    clear(this.rawBuf);
    clear(this.resultBuf);
    clear(this.residBuf);
    clear(this.histBuf);
    if (find) { for (const b of [this.readingsBuf, this.accBuf, this.candBuf, this.quadsBuf, this.scoreBuf, this.mapsBuf, this.measBuf]) clear(b); enc.clearBuffer(this.selBuf); }
    let pass = null;
    const begin = (name) => {
      if (pass && !profile) return;
      if (pass) pass.end();
      pass = enc.beginComputePass(pair(profile ? STAGES.indexOf(name) : 0));
    };
    // ai: The pyramid and the grey levels, a PYRAMID_BLOCK square of a frame a workgroup (wgsl/pyramid.mjs); the histogram
    // ai: is written only to its own buffer, so no decode stage sees it.
    begin("pyramid");
    pass.setPipeline(this.pyramid.p);
    pass.setBindGroup(0, this.pyrGroup);
    pass.dispatchWorkgroups(ceilDiv(W, PYRAMID_BLOCK), ceilDiv(H, PYRAMID_BLOCK), nb);
    // F2 on every level, then F3 once a frame.
    if (!inject) {
      begin("bank");
      this.bank.dispatch(pass, this, nb);
      begin("select");
      pass.setPipeline(this.select.p);
      pass.setBindGroup(0, this.selGroup);
      pass.dispatchWorkgroups(nb);
    }
    if (find) {
      const step = (pl, group, x, y = 1) => { pass.setPipeline(pl.p); pass.setBindGroup(0, group); pass.dispatchWorkgroups(x, y); };
      // F4: a workgroup of the network takes its kernel's P patches from slot P wg.x (and returns at once past the
      // frame's count), so ceil(cap / P) workgroups a frame cover the cap slots. The cascade: the small net on every
      // slot, RANK a frame, the large net on `keep` list slots.
      // ai: cap (or keep) workgroups before 2026-09-30, P - 1 in P of them returning at once: the small net's
      // ai: describe 0.90 to 0.83 ms a frame on the iGPU for launching them (STATUS "Native speed").
      begin("describe");
      const net = (pipe, group, slots) => step(pipe, group, ceilDiv(slots, pipe.P), nb);
      if (this.cascade) { net(this.classify0, this.classify0Group, this.cap); step(this.rank, this.rankGroup, nb); net(this.classify, this.classifyGroup, this.cascade.keep); }
      else net(this.classify, this.classifyGroup, this.cap);
      begin("vote");
      step(this.vote, this.voteGroup, ceilDiv(this.cap, 64), nb);
      step(this.gather, this.gatherGroup, nb);
      begin("score");
      step(this.score, this.scoreGroup, HYP * nb);
      step(this.pick, this.pickGroup, nb);
      if (this.register) {
        // ai: One workgroup a node of the largest ring; a frame's own ring's nodes past its count return at once.
        this.nodesGroups.forEach((g, r) => {
          begin(`nodes${r + 1}`); step(this.nodesP, g, this.nodesMax, nb);
          begin(`refit${r + 1}`); step(this.refit, this.refitGroup, nb);
        });
      }
    }
    // ai: F8 on every frame the finder found or the host told: the word through the final map, a workgroup a frame,
    // ai: and the version and picture the frame decodes at (its word's, or the held one).
    begin("word");
    pass.setPipeline(this.wordP.p);
    pass.setBindGroup(0, this.wordGroup);
    pass.dispatchWorkgroups(nb);
    if (!fused) {
      begin("sample");
      pass.setPipeline(this.sampler.p);
      pass.setBindGroup(0, this.sampleGroup);
      pass.dispatchWorkgroups(ceilDiv(nmax, 16), ceilDiv(nmax, 16), nb);
    }
    // ai: The back half behind the sampler, and with the cancel stage its pass two behind that: in the same pass, or
    // ai: profiled as their passes with a timestamp pair each.
    if (profile) pass.end();
    const target = profile ? enc : pass, at = (first) => (profile && this.qs ? { qs: this.qs, base: 2 * STAGES.indexOf(first) } : null);
    if (this.bh) this.bh.encode(target, this.backLane, { fused, frames: nb, ts: at(BACK_PASSES[0]) });
    if (this.cancel) this.bh.encodeCancel(target, this.backLane, { fused, frames: nb, ts: at(CANCEL_PASSES[0]) });
    if (!profile) pass.end();
    const probed = probe ? this.probeKit.encode(this, enc, nb) : null;
    // The readback's layout. The wasm back half's: grid, counters, peaks, finder results, histograms, timestamps. The
    // GPU's: counters, finder results, histograms, the back half's (REC, V, ITS, BCOUNTS), timestamps; only that much is mapped.
    // ai: With the cancel stage the back half's part holds pass two's counters too (BCOUNTS2, CANCEL).
    const gridBytes = this.bh ? 0 : 4 * this.gridStride * nb, cntBytes = 4 * COUNTERS * nb, pkBytes = this.bh ? 0 : 16 * this.cap * nb, resBytes = 4 * RESULT * nb, histBytes = 4 * HIST_BINS * nb;
    const cntOff = gridBytes, pkOff = cntOff + cntBytes, resOff = pkOff + pkBytes, histOff = resOff + resBytes, backOff = histOff + histBytes;
    const tsOff = backOff + (this.bh ? this.bh.readBytes : 0);
    if (!this.bh) enc.copyBufferToBuffer(this.gridBuf, 0, this.readBuf, 0, gridBytes);
    enc.copyBufferToBuffer(this.countsBuf, 0, this.readBuf, cntOff, cntBytes);
    if (!this.bh) enc.copyBufferToBuffer(this.peaksBuf, 0, this.readBuf, pkOff, pkBytes);
    enc.copyBufferToBuffer(this.resultBuf, 0, this.readBuf, resOff, resBytes);
    enc.copyBufferToBuffer(this.histBuf, 0, this.readBuf, histOff, histBytes);
    if (this.bh) this.bh.readback(enc, this.backLane, this.readBuf, backOff);
    const nq = this.qs ? this.qs.count : 0, tsInOff = tsOff + 8 * nq;
    if (this.qs) { enc.resolveQuerySet(this.qs, 0, nq, this.qsBuf, 0); enc.copyBufferToBuffer(this.qsBuf, 0, this.readBuf, tsOff, 8 * nq); }
    // ai: F0's device time: the ring's pairs (each written by its slot's ingest pass, before this batch on the in-order
    // ai: queue) resolved whole (a resolve lands at a multiple of 256), then each VideoFrame slot's pair copied behind
    // ai: the stages' timestamps, at 8 TS_INGEST i for slot i.
    const inPairs = ring.qs ? slots.map((s, i) => (s?.f0 ? i : -1)).filter((i) => i >= 0) : [];
    if (inPairs.length) {
      enc.resolveQuerySet(ring.qs, 0, TS_INGEST * ring.R, ring.qsBuf, 0);
      for (const i of inPairs) enc.copyBufferToBuffer(ring.qsBuf, 8 * TS_INGEST * slots[i].id, this.readBuf, tsInOff + 8 * TS_INGEST * i, 8 * TS_INGEST);
    }
    const readBytes = tsInOff + (inPairs.length ? 8 * TS_INGEST * nb : 0);
    let gridRead = null;
    if (this.bh && grids) {
      gridRead = d.createBuffer({ size: 4 * this.gridStride * nb, usage: U.MAP_READ | U.COPY_DST });
      enc.copyBufferToBuffer(this.gridBuf, 0, gridRead, 0, gridRead.size);
    }
    // Debug: the finder's intermediate buffers, read back separately (never on a timed path).
    let dbg = null;
    if (debug && find) {
      const parts = [["readings", this.readingsBuf], ["cand", this.candBuf], ["quads", this.quadsBuf], ["score", this.scoreBuf], ["result", this.resultBuf], ["meas", this.measBuf], ["resid", this.residBuf], ["maps", this.mapsBuf], ["refit", this.refitDbg]];
      const size = parts.reduce((a, [, b]) => a + b.size, 0);
      dbg = { buf: d.createBuffer({ size, usage: U.MAP_READ | U.COPY_DST }), parts };
      let o = 0;
      for (const [, b] of parts) { enc.copyBufferToBuffer(b, 0, dbg.buf, o, b.size); o += b.size; }
    }
    d.queue.submit([enc.finish()]);
    // ai: The copies are queued, so the slots may be taken again: a later ingest into one lands behind them.
    for (const s of slots) if (s) root.release(s);
    lap("encode");
    const submitted = t[t.length - 1];
    // The scopes are popped now, before another lane's batch pushes its own over them; the map request goes in now
    // too, so the readback waits on nothing but this batch.
    const scopes = [d.popErrorScope(), d.popErrorScope()];
    const mapped = this.readBuf.mapAsync(GPUMapMode.READ, 0, readBytes);
    probed?.start();
    const errs = (await Promise.all(scopes)).filter(Boolean).map((e) => e.message);
    if (errs.length) { mapped.then(() => this.readBuf.unmap(), () => {}); gridRead?.destroy(); probed?.discard(); throw new Error(`GPU batch failed: ${errs.join(" | ")}`); }
    lap("scopes");
    await mapped;
    lap("map");
    const done = t[t.length - 1];
    let debugOut = null;
    if (dbg) {
      await dbg.buf.mapAsync(GPUMapMode.READ);
      const raw = dbg.buf.getMappedRange().slice(0);
      dbg.buf.unmap(); dbg.buf.destroy();
      debugOut = {};
      let o = 0;
      for (const [name, b] of dbg.parts) { debugOut[name] = new Float32Array(raw, o, b.size / 4); o += b.size; }
    }
    let gridAll = null;
    if (gridRead) { await gridRead.mapAsync(GPUMapMode.READ); gridAll = gridRead.getMappedRange(); }
    // Only what the batch made leaves the mapped range: the counters, the results, the kept peaks and each sampled
    // picture's n x n floats, not the grid's room for the largest picture (4 MB a frame at nmax 1024).
    const all = this.readBuf.getMappedRange(0, readBytes);
    const counts = new Uint32Array(all, cntOff, COUNTERS * nb).slice();
    const ts = this.qs ? new BigUint64Array(all, tsOff, nq).slice() : null;
    const stageMs = profile && ts ? Object.fromEntries(STAGES.map((n, k) => [n, Number(ts[2 * k + 1] - ts[2 * k]) / 1e6])) : null;
    const tsIn = inPairs.length ? new BigUint64Array(all, tsInOff, TS_INGEST * nb).slice() : null;
    const res = new Float32Array(all, resOff, RESULT * nb).slice();
    const hist = new Uint32Array(all, histOff, HIST_BINS * nb).slice();
    const peaks = (i) => (this.bh ? new Float32Array(0) : new Float32Array(all, pkOff + 16 * this.cap * i, 4 * Math.min(this.cap, counts[COUNTERS * i + COUNT.peaks])).slice());
    const gridAt = (i, n) => (this.bh ? (gridAll ? new Float32Array(gridAll, 4 * this.gridStride * i, n * n).slice() : new Float32Array(0)) : new Float32Array(all, 4 * this.gridStride * i, n * n).slice());
    // The back half's frame: its counters (back.mjs COUNT), how many blocks got each verdict (0 not run, 1 verified,
    // 2 declined, 3 stall or cap, 4 CRC), the verified blocks (block, iterations, id, 473 payload bytes, the id
    // first), and with grids every block's verdict. An empty slot must show zeros and no block.
    // ai: cancelled: pass two ran on the frame with a solved fit; refs: the references the gate gave it; gained: the
    // ai: blocks pass two verified, among the records (0, 0 and false without the cancel stage).
    const bk = this.bh ? parseReadback(all.slice(backOff, backOff + this.bh.readBytes), 0, this.bh.dims) : null;
    const backOf = (i) => {
      if (!bk) return null;
      const fr = bk.frames[i], verdicts = [0, 0, 0, 0, 0];
      for (const v of fr.verdict) verdicts[Math.min(v, 4)]++;
      return { counts: fr.counts, verdicts, records: bk.records.filter((x) => x.frame === i), verdict: grids ? Array.from(fr.verdict) : null, cancelled: !!fr.cancel?.cancelled, refs: fr.cancel?.refs ?? 0, gained: fr.cancel?.gained ?? 0 };
    };
    // ai: The batch's device time: the timestamps where the device has them (unprofiled, its one compute pass), else
    // ai: its readback's arrival less when the device could start it (its submit, or the batch before it done).
    const out = {
      debug: debugOut,
      probe: null,
      gpuMs: ts ? (profile ? Object.values(stageMs).reduce((a, v) => a + v, 0) : Number(ts[1] - ts[0]) / 1e6) : null,
      stageMs,
      // ai: F0's device time for the batch's VideoFrame slots, their ingest passes summed (0 with none, or no
      // ai: timestamps): gpuMs and deviceMs leave it out.
      ingestMs: tsIn ? inPairs.reduce((a, i) => a + Number(tsIn[TS_INGEST * i + 1] - tsIn[TS_INGEST * i]) / 1e6, 0) : 0,
      carried: nb,
      submitted,
      read: 0,   // the page's clock once the readback is out of the mapped range (set below)
      readBytes,   // what came back from the device this batch
      backLane: this.backLane?.bytes ?? 0,
      host,
      // An empty slot still reports its counters: every one must be zero, which is the check that no stage
      // wrote into a frame that was not there.
      // ai: ring: the finder's (result[2]) or the told one, -1 for a frame not found. version: the one the frame
      // ai: decodes at (F8: its word's, or the held one; 0 for none), its picture n and slot (size, -1 for none),
      // ai: held: the held version stood in. sampled: there was a picture to decode; its grid only when F9 ran.
      frames: slots.map((s, i) => {
        if (!live[i]) return { empty: true, counts: Array.from(counts.subarray(COUNTERS * i, COUNTERS * (i + 1))), back: backOf(i) };
        const r = Array.from(res.subarray(RESULT * i, RESULT * (i + 1)));
        const ring = find ? (r[0] === 1 ? r[2] : -1) : s.ring, version = r[29], size = version > 0 ? slotOf(version) : -1;
        const n = size >= 0 ? this.tables.pictures[size].n : 0, sampled = version > 0 && (!find || r[0] === 1);
        return {
        ring, version, n, size, held: r[30] === 1, sampled, finder: find ? { found: r[0], score: r[1], ring: r[2], orient: r[3], quad: r[4], H: r.slice(8, 17), corners: r.slice(17, 25), depths: r.slice(25, 29) } : null,
        // ai: F8's reading (wgsl/word.mjs): the version and rate the band states and the best word's soft score, null
        // ai: when no word was taken.
        word: r[5] > 0 ? { version: r[5], fps: r[6], score: r[7] } : null,
        grid: sampled && n * n <= this.gridStride ? gridAt(i, n) : new Float32Array(0),
        counts: Array.from(counts.subarray(COUNTERS * i, COUNTERS * (i + 1))),
        peaks: peaks(i),
        back: backOf(i),
        // ai: the frame's grey levels: HIST_BINS counts over the middle half of its crop, every second pixel (wgsl/pyramid.mjs)
        hist: hist.subarray(HIST_BINS * i, HIST_BINS * (i + 1)),
        // ai: the pilots (SPEC 7.3; pilotOf): { r, sd, r2, sd2, blocks } over the blocks its version carries, null where the soft
        // ai: stage did not run on it this batch (its column 6 of the back half's counters, sub-channels quantised, is 0)
        // ai: (a rate profile's frames over its own blocks, back/tiers.mjs blocksAt)
        pilot: bk && version > 0 && bk.frames[i].counts[6] > 0 ? pilotOf(bk.frames[i].pilots, blocksAt(this.bh?.tiers, size, version)) : null,
      };
      }),
    };
    if (gridRead) { gridRead.unmap(); gridRead.destroy(); }
    this.readBuf.unmap();
    lap("read");
    out.read = t[t.length - 1];
    // ai: After the lane's readback is let go, so a probe that fails to read leaves the lane usable.
    if (probed) out.probe = await probed.read();
    out.deviceMs = out.gpuMs ?? done - Math.max(submitted, root.lastDone);
    // ai: Submitted with no batch of ours still on the device: then its wait past deviceMs is the readback's round
    // ai: trip alone, and not a lane ahead of it.
    out.idle = submitted >= root.lastDone;
    root.lastDone = Math.max(root.lastDone, done);
    // ai: A lane's first batch is watched for the cap alone: every pipeline's first use, and every buffer's (WebGPU
    // ai: zero-fills it then; about 17 ms for a lane at a 1 GB budget on the iGPU), costs more than any later one, so it
    // ai: counts toward neither the 90% run nor the room to grow. warm() spent a frame through each lane for it before
    // ai: 2026-09-29.
    if (root.batcher) root.batcher.observe({ frames: nb, ms: out.deviceMs, submitted, done, first: !this.watched });
    this.watched = true;
    out.batching = root.batcher ? root.batcher.report() : { auto: false, B, inflight: root.fixedInflight };
    // ai: The int8 twins timed on this batch's first live frames, while this lane still holds what they left.
    let lead = 0;
    while (lead < Math.min(nb, TWIN_FRAMES) && live[lead]) lead++;
    // ai: A measurement that fails (a lost device included) leaves the batch's answer standing: its readback is in.
    if (root.twinForms && lead) {
      const t1 = performance.now();
      try { await root.measureTwins(this, lead); } catch (e) { root.log(`int8 twins not measured (${e?.message ?? e}): the float files`); }
      out.twinMs = performance.now() - t1;
    }
    return out;
  }
}

// The batch controller behind B "auto". WebGPU tells a page its buffer and workgroup limits, not the memory free,
// the clocks or how long a submit may run before the device is reset, so those are measured on the decoder's own
// batches. Memory: the lanes are planned for Bmax, the largest batch whose lanes (two) fit the budget together
// (FrontHalf.plan prices a frame with a one-frame lane). Time: the stream starts at Bmax, and every batch but a
// lane's first is watched: one over the cap halves the size at once, three running over 90% of it halve it, and
// sixteen running with room for a double grow it back toward Bmax.
// ai: Started at Bmax since 2026-09-29, shrunk on evidence: before, the first plan doubled a batch of copies of one
// ai: held frame from 1 while it held under the cap, 63 decodes on the S26's first camera frame while the camera's
// ai: frames waited or were dropped, and settled on the largest that held (32 on the phone, the iGPU and the 4090 at
// ai: 1080). The cap, a fifth of the ~2 s submit that lost the iGPU's device, stays the guard, now on stream batches
// ai: alone; a device where Bmax runs over it halves on its first watched batch.
// ai: The ring (Bmax + RING_SLACK layers) is in the budget beside the lanes (plan prices a slot too). In flight: 2
// ai: lanes, or the number named (the harness's INFLIGHT); a third lane never paid (2026-09-25: the iGPU is 96 to
// ai: 99% busy at 2, and on the phone the feed, not the device, starved).
export class Batcher {
  constructor(fh, { budget, capMs, restart = 0, inflight = "auto" }) {
    Object.assign(this, { fh, budget, capMs, restart });
    this.size = 0; this.ceiling = 0; this.Bmax = 0; this.perFrame = 0; this.slotBytes = 0; this.planMs = 0;
    this.bound = null; this.BmaxWhy = null;
    this.events = [];     // what the watch did, in order
    this.over = 0; this.spare = 0;
    this.fixedInflight = inflight === "auto" ? 0 : inflight;
    this.inflight = this.lanes = this.fixedInflight || 2;
    this.inflightWhy = this.fixedInflight ? "named" : "the next batch encoded while the one before it runs";
  }

  log(m) { this.events.push(m); if (this.events.length > 16) this.events.shift(); this.fh.log(`batch auto: ${m}`); }

  // The plan's memory side: a frame's bytes a lane, what is shared, the room the budget leaves the lanes, and the
  // device's own cap on a batch.
  // ai: slotBytes: a ring slot's, since the ring shares the room.
  sized({ perFrame, shared, room, cap, capWhy, slotBytes }) {
    Object.assign(this, { perFrame, shared, room, cap, capWhy, slotBytes });
  }

  // The largest batch n lanes of it fit, and what bounds it.
  // ai: With its ring: n lanes of Bmax frames plus Bmax + RING_SLACK slots.
  limitFor(n) {
    const MB = (v) => (v / MiB).toFixed(1), slot = this.slotBytes;
    let Bmax = Math.floor((this.room - RING_SLACK * slot) / (n * this.perFrame + slot)), why = `memory: ${n} lanes of ${Bmax} and a ring of ${Bmax + RING_SLACK} fill the ${MB(this.budget)} MB budget (a frame ${MB(this.perFrame)} MB a lane and ${MB(slot)} MB on the ring, ${MB(this.shared)} MB shared)`;
    if (this.cap < Bmax) { Bmax = this.cap; why = this.capWhy; }
    if (Bmax > B_CEIL) { Bmax = B_CEIL; why = `the ceiling of ${B_CEIL} a batch (past it there is nothing left to amortise)`; }
    if (Bmax < 1) { Bmax = 1; why = `memory: ${n} lanes of one frame and a ring of ${1 + RING_SLACK} (${MB(n * this.perFrame + (1 + RING_SLACK) * slot)} MB) are over the ${MB(this.budget)} MB budget; batches of 1`; }
    return { Bmax, why };
  }

  // The lanes made: n of Bmax. A batch size past Bmax comes down to it.
  laned(n, Bmax, why) {
    Object.assign(this, { lanes: n, Bmax, BmaxWhy: why });
    if (this.size > Bmax) { this.size = Bmax; this.bound = why; }
    if (this.ceiling > Bmax) this.ceiling = Bmax;
  }

  // The size to start at: Bmax on the first plan; on a later one (a larger frame or picture) the size the watch had
  // come to, if smaller; after a lost device the restart size, never grown past.
  start() {
    const MB = (v) => (v / MiB).toFixed(0), prev = this.size;
    const best = this.restart ? Math.min(this.restart, this.Bmax) : prev > 0 ? Math.min(prev, this.Bmax) : this.Bmax;
    this.bound = this.restart ? `restarted at ${best} after the device was lost` : best < this.Bmax ? `${best}, the size the watch had come to before this plan` : this.BmaxWhy;
    this.size = best;
    this.ceiling = this.restart ? best : this.Bmax;
    this.over = 0; this.spare = 0;
    this.fh.log(`batch auto: ${best} frames a batch (lanes of ${this.Bmax}, ${MB(this.perFrame * this.Bmax)} MB each; ${this.bound})`);
  }

  resize(n, why) {
    if (n === this.size) return;
    this.log(`${this.size} to ${n} frames: ${why}`);
    this.size = n;
    this.over = 0; this.spare = 0;
  }

  // A settled batch of the stream: frames it carried, its device time, and its submit and readback on the page's clock.
  // ai: first: a lane's first batch, watched for the cap alone (batch()).
  observe(b) {
    if (!this.size) return;
    const s = this.size, cap = this.capMs;
    if (b.ms > cap) this.resize(Math.max(1, s >> 1), `a batch of ${b.frames} took ${b.ms.toFixed(0)} ms, over the ${cap} ms cap`);
    else if (b.first) return;
    else if (b.ms > 0.9 * cap && b.frames >= s) { if (++this.over >= 3) this.resize(Math.max(1, s >> 1), `three batches running over 90% of the ${cap} ms cap (${b.ms.toFixed(0)} ms)`); }
    else this.over = 0;
    if (this.size === s && s < this.ceiling && b.frames >= s && b.ms * 2 <= cap) {
      if (++this.spare >= 16) this.resize(Math.min(this.ceiling, 2 * s), `sixteen batches running with room for a double (${b.ms.toFixed(0)} ms)`);
    } else this.spare = 0;
  }

  report() {
    return { auto: true, size: this.size, ceiling: this.ceiling, Bmax: this.Bmax, lanes: this.lanes, inflight: this.inflight, bound: this.bound, inflightWhy: this.inflightWhy, events: this.events.slice(), perFrame: this.perFrame, capMs: this.capMs, budget: this.budget, planMs: this.planMs };
  }
}
