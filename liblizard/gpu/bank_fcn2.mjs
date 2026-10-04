// The filter bank as the trained proposer, version 2 (wgsl/bank_fcn2.mjs, scripts/gpu/cnn/proposer/README.md "Version 2"), to
// the contract in wgsl/bank.mjs: levels 1 to 4 (level 0's slots stay empty), one workgroup a region of 16 x 16
// tiles. The map does not depend on the region, so the region is the largest the device's workgroup memory holds
// (32 x 32 in f16 or halves where it gives 64 KB, 32 x 16 in 32 KB). With the X plane (the default) an X pass writes
// each level's standardised map once, halves where the activations are halves, and the network kernel reads it; the
// fused form recomputes it on every region's halo instead. The shader runs in f16 when the decoder's precision is
// f16, else f32 storage where it fits, else f32 with halves stored. A device too small for all of them gets FitError.
// ai: FitError leaves the device with no GPU decoder: the C decodes (since 2026-09-26, no DoG bank to fall to).
//
// ?fcnstore= (FCN_STORE in the harness), comma separated: packed or f32 forces the storage, region=32x16 the region,
// fused the one-kernel form, xblock=32x32 the X pass's block, walk=<n> the regions a network workgroup walks, unfit
// the fallback, weights=<path under liblizard/> another version 2 weights file.
// ai: The decoder's precision "int8" runs the int8 form (wgsl/bank_fcn2.mjs) on a q8 file (cnn/q8.mjs): the one named
// ai: by weights=, else WEIGHTS_INT8; a float precision runs a float form on a float file. Either refuses the other's
// ai: file, and a precision it does not know.
import { fcn2Source, fcn2Plan, fcn2Forms, fcn2Bytes, KEEP, TILE } from "./wgsl/bank_fcn2.mjs";
import { loadNet } from "./cnn/netfile.mjs";
import { LEVELS, TAU, EPS } from "./constants.mjs";   // ai: not ./decoder.mjs, which imports this module (constants.mjs)

export const name = "fcn2";
export { KEEP };
// ai: The installed proposer since 2026-10-03: the final retrain's arm A seed 1 trained further on blown-out scenes
// ai: (STATUS "Blown-out codes: the finder measured and its classifiers trained further, round 1" and its round 2):
// ai: warm-started at LR 1e-3 on sim210000 and the half-blown sim440000, seed 1
// ai: (research/results/gpu/prop_blown/pb-s1.safetensors; its f64 reference pb-s1.ref there), its floor 0.15 (pb-s1-f15): the
// ai: highest at which its judges' misses stay in the three-seed band, the raw peaks then 0.78 times the one before at
// ai: 0.1 (at 0.1 it wrote 1.83 times as many, 0.5% of the iGPU's frame). scripts/train/decide.py: adopt (a tie). The one
// ai: before it: weights2-before-blown.safetensors (research/results/gpu/final/a-s1.safetensors, reference a-s1.ref there);
// ai: before that weights2-before-final.safetensors (round 1 of the version 2 loop, research/results/gpu/fcn2_loop/reference-r1.bin).
// ai: A reference holds the net's values, not its peaks: the checks take the floor from the file.
export const WEIGHTS = "gpu/cnn/proposer/weights2.safetensors";
export const REFERENCE = "../research/results/gpu/prop_blown/pb-s1.ref";
// ai: The installed int8 proposer (a q8 file) and its PQ81 reference: the post-training quantisation of WEIGHTS
// ai: (research/results/gpu/prop_blown/pb-s1-q8-f15.safetensors, the floor 0.15), installed 2026-10-03; the one before it,
// ai: weights2-q8-before-blown.safetensors (research/results/gpu/final/a-s1-q8.safetensors, reference a-s1-q8.ref there), and
// ai: before that weights2-q8-before-final.safetensors (research/results/gpu/q8/p2q8p-s2.json, reference research/results/gpu/q8/reference-p2q8p-s2.bin).
// ai: weights= (the harness's FCN_STORE=weights=<path>) runs another.
export const WEIGHTS_INT8 = "gpu/cnn/proposer/weights2-q8.safetensors";
export const REFERENCE_INT8 = "../research/results/gpu/prop_blown/pb-s1-q8.ref";
const U = globalThis.GPUBufferUsage ?? {};
const ceilDiv = (a, b) => Math.ceil(a / b);

export class FitError extends Error {}

export function storeOptions(s = globalThis.location ? new URLSearchParams(globalThis.location.search).get("fcnstore") ?? "" : "") {
  const o = {};
  for (const kv of s.split(",").filter(Boolean)) { const i = kv.indexOf("="); if (i < 0) o[kv] = true; else o[kv.slice(0, i)] = kv.slice(i + 1); }
  return o;
}

export async function build(fh) {
  const o = storeOptions();
  // ai: a path under liblizard/, or a candidate under research/ (../research/..., the trainers' and the checks')
  if (o.weights && (o.weights.replace(/^\.\.\/research\//, "").includes("..") || o.weights.startsWith("/"))) throw new Error(`fcn2 weights ${o.weights}: a path under liblizard/ or ../research/`);
  if (!["int8", "f16", "f32"].includes(fh.precision)) throw new Error(`fcn2 bank: precision ${fh.precision}: int8, f16 or f32`);
  const int8 = fh.precision === "int8", file = o.weights ?? (int8 ? WEIGHTS_INT8 : WEIGHTS);
  if (!file) throw new Error("fcn2 bank: no int8 proposer file (none is installed; name one with fcnstore weights=<path under liblizard/>)");
  const json = await loadNet(new URL(`../${file}`, import.meta.url));
  const f16 = !int8 && fh.floatPrecision === "f16";
  if (f16 && !fh.features.includes("shader-f16")) throw new Error("fcn2 bank: f16 asked for without shader-f16");
  const limit = fh.device.limits.maxComputeWorkgroupStorageSize;
  let forms;
  try { forms = fcn2Forms(json, { f16, int8, force: o }); } catch (e) { throw new Error(`fcn2 bank ${fh.precision}, ${file}: ${e.message}`); }
  // ai: A value that leaves no form (region=2 is v1's) is a mistake, not a small device: FitError would hide it as a
  // ai: device too small.
  if (!forms.length) throw new Error(`fcn2 bank: fcnstore ${JSON.stringify(o)} matches no form; the forms are ${fcn2Forms(json, { f16, int8 }).map((t) => t.name).join(", ")}`);
  const form = o.unfit ? null : forms.find((t) => t.bytes <= limit);
  // ai: int8 too big for the device is a failed int8 build, not a small device: the caller's precision chain then
  // ai: tries f16 or f32, whose halves-stored forms are smaller.
  if (!form && int8 && !o.unfit) throw new Error(`fcn2 bank int8: ${Math.min(...forms.map((t) => t.bytes))} bytes of workgroup memory at the least, the device gives ${limit}`);
  if (!form) throw new FitError(`fcn2 bank: ${Math.min(...forms.map((t) => t.bytes))} bytes of workgroup memory at the least, the device gives ${limit}`);
  const plan = fcn2Plan(json);
  const src = fcn2Source(json, { ...form, stage: form.xplane ? "net" : "all" });
  fh.fcn2 = await fh.pipeline(src, ["ro", "ro", "rw", "rw", "uniform"]);
  fh.fcn2X = form.xplane ? await fh.pipeline(fcn2Source(json, { ...form, stage: "x" }), ["ro", "ro", "rw", "uniform"]) : null;
  fh.fcn2Form = form;
  fh.bankStore = form.name;
  fh.log(`fcn2 bank ${json.spec ?? ""} (${file}) ${form.name}: ${plan.macs} MACs a map pixel, box ${plan.K}, floor ${plan.floor}, ${form.bytes} bytes of workgroup memory (network ${fcn2Bytes(src)}; the device gives ${limit}), levels 1 to ${LEVELS - 1}`);
}

// dimsAt(l): the level's (w, h) at the current capacity; bases[l]: where its slots start; slots: a frame's total.
// With the plane, each level has one per lane: B frames of rows x stride rounded up to even, halves in pairs or f32.
export function ensure(fh, dimsAt, bases, slots) {
  const d = fh.device, form = fh.fcn2Form;
  for (const lv of fh.fcn2Levels ?? []) lv.plane?.destroy();
  fh.fcn2Levels = [];
  for (let l = 1; l < LEVELS; l++) {
    const uni = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST }), ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    const [w, h] = dimsAt(l);
    u.set([l, w, h, slots]); f[4] = TAU; f[5] = EPS; u[6] = bases[l]; u[7] = ceilDiv(w, TILE);
    d.queue.writeBuffer(uni, 0, ab);
    // ai: int8: Xq, a byte a pixel at a row stride rounded up to four (a word never straddles two rows).
    const plane = !form.xplane ? null : d.createBuffer({ size: form.int8 ? ((w + 3) & ~3) * h * fh.B : (form.store === "f32" ? 4 : 2) * (w + (w & 1)) * h * fh.B, usage: U.STORAGE });
    const group = d.createBindGroup({ layout: fh.fcn2.bgl, entries: [
      { binding: 0, resource: { buffer: plane ?? fh.level[l - 1].buf } }, { binding: 1, resource: { buffer: fh.framesBuf } }, { binding: 2, resource: { buffer: fh.rawBuf } },
      { binding: 3, resource: { buffer: fh.countsBuf } }, { binding: 4, resource: { buffer: uni } }] });
    const xgroup = plane ? d.createBindGroup({ layout: fh.fcn2X.bgl, entries: [fh.level[l - 1].buf, fh.framesBuf, plane, uni].map((buffer, binding) => ({ binding, resource: { buffer } })) }) : null;
    fh.fcn2Levels.push({ w, h, group, xgroup, plane });
  }
}

export function dispatch(pass, fh, B) {
  const { TX, TY } = fh.fcn2Form;
  if (fh.fcn2X) {
    pass.setPipeline(fh.fcn2X.p);
    for (const lv of fh.fcn2Levels) { pass.setBindGroup(0, lv.xgroup); pass.dispatchWorkgroups(ceilDiv(lv.w, fh.fcn2Form.XB[0]), ceilDiv(lv.h, fh.fcn2Form.XB[1]), B); }
  }
  pass.setPipeline(fh.fcn2.p);
  for (const lv of fh.fcn2Levels) {
    pass.setBindGroup(0, lv.group);
    pass.dispatchWorkgroups(ceilDiv(ceilDiv(lv.w, TILE * TX), fh.fcn2Form.walk), ceilDiv(lv.h, TILE * TY), B);
  }
}
