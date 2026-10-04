// The filter bank as a trained proposer (wgsl/bank_fcn.mjs, scripts/gpu/cnn/proposer/README.md), to the contract in
// wgsl/bank.mjs: one dispatch a pyramid level, levels 1 to 4 (level 0's slots stay empty), one workgroup a
// 16 x 16 tile. The weights come from cnn/proposer/weights.safetensors beside this module, baked into the shader; the
// shader runs in f16 when the decoder's precision is f16 (shader-f16 on the device), else f32, and in f32 with its
// activations stored as halves when f32 storage does not fit the device's workgroup memory (37 KB against the 32 KB
// of many adapters; 18.7 KB halves). A device too small even for that gets FitError.
// ai: FitError leaves the device with no GPU decoder: the C decodes (since 2026-09-26, no DoG bank to fall to).
//
// ?fcnstore= (FCN_STORE in the harness), comma separated variants: packed stores halves where f32 would fit;
// region=2 runs two tiles a workgroup standardised over one window (a different function, wgsl/bank_fcn.mjs);
// flat[=sd] skips a workgroup whose window's deviation is under sd (FLAT without a value; a frame's cost then
// depends on what it shows); unfit takes the fallback on purpose, so the fallback can be run where everything fits.
// ?fcnweights=<path under liblizard/> (FCN_WEIGHTS in the harness) loads another version 1 weights file, so a candidate
// is scored without being installed (scripts/gpu/cnn/TRAINING.md).
import { fcnSource, fcnPlan, fcnBytes, KEEP, TILE } from "./wgsl/bank_fcn.mjs";
import { loadNet } from "./cnn/netfile.mjs";
import { LEVELS, TAU, EPS } from "./constants.mjs";   // ai: not ./decoder.mjs, which imports this module (constants.mjs)

export const name = "fcn";
export { KEEP };
// The flat variant's default floor: on 3,000 assembled scenes and 1,000 recorded frames no mark the bank keeps sits
// in a window under it; at 0.005 a mark under off-angle falloff (window sd 0.0023) was lost (README).
export const FLAT = 0.002;
const U = globalThis.GPUBufferUsage ?? {};
const ceilDiv = (a, b) => Math.ceil(a / b);

export class FitError extends Error {}

function storeOptions() {
  const s = globalThis.location ? new URLSearchParams(globalThis.location.search).get("fcnstore") ?? "" : "";
  const o = {};
  for (const kv of s.split(",").filter(Boolean)) { const [k, v] = kv.split("="); o[k] = v === undefined ? true : +v; }
  return o;
}

export async function build(fh) {
  const file = globalThis.location ? new URLSearchParams(globalThis.location.search).get("fcnweights") : null;
  // ai: a path under liblizard/, or a candidate under research/ (../research/..., the trainers' and the checks')
  if (file && (file.replace(/^\.\.\/research\//, "").includes("..") || file.startsWith("/"))) throw new Error(`fcn weights ${file}: a path under liblizard/ or ../research/`);
  const json = await loadNet(new URL(file ? `../${file}` : "./cnn/proposer/weights.safetensors", import.meta.url));
  // This kernel standardises over the tile's window; a version 2 file (each pixel over its box) needs bank_fcn2.mjs.
  if (json.arch?.norm === "pixel") throw new Error(`fcn weights ${file ?? "weights.safetensors"}: a version 2 file (norm pixel), run it as BANK=fcn2`);
  if (json.arch?.quant || /-q8$/.test(json.spec ?? "")) throw new Error(`fcn weights ${file ?? "weights.safetensors"}: an int8 file (${json.spec}); this bank runs floats, and int8 is BANK=fcn2's`);
  // ai: The arithmetic: f16 or f32 as the decoder's precision names, or, when that is int8 (a form this bank has
  // ai: none of), the decoder's float precision. FrontHalf.create sets fh.precision to int8, f16 or f32 whatever the
  // ai: finder, so anything else is a mistake, not f32.
  const prec = fh.precision === "int8" ? fh.floatPrecision : fh.precision;
  if (prec !== "f16" && prec !== "f32") throw new Error(`fcn bank: precision ${fh.precision}${fh.precision === "int8" ? ` (float ${fh.floatPrecision})` : ""}: f16, f32, or int8 with a float precision`);
  const f16 = prec === "f16";
  if (f16 && !fh.features.includes("shader-f16")) throw new Error("fcn bank: f16 asked for without shader-f16");
  const limit = fh.device.limits.maxComputeWorkgroupStorageSize, o = storeOptions();
  const region = o.region === true ? 2 : o.region ?? 1, flat = o.flat === true ? FLAT : o.flat ?? 0;
  // The first that fits; storage gives way before the region does, so every device computes the same function.
  const tries = f16 ? [{ f16, region }] : [...(o.packed ? [] : [{ region }]), { packed: true, region }];
  const fit = o.unfit ? null : tries.map((t) => ({ t, src: fcnSource(json, { ...t, flat }) })).find(({ src }) => fcnBytes(src) <= limit);
  if (!fit) throw new FitError(`fcn bank: ${fcnBytes(fcnSource(json, { ...tries.at(-1), flat }))} bytes of workgroup memory at the least, the device gives ${limit}`);
  const plan = fcnPlan(json), { t, src } = fit;
  fh.fcn = await fh.pipeline(src, ["ro", "ro", "rw", "rw", "uniform"]);
  fh.fcnRegion = t.region;
  fh.bankStore = f16 ? "f16" : t.packed ? "f32, halves stored" : "f32";
  fh.log(`fcn bank ${json.spec ?? ""}${file ? ` (${file})` : ""} ${fh.bankStore}: ${plan.macs} MACs a map pixel, halo ${plan.halo}, ${t.region} tile${t.region > 1 ? "s" : ""} a workgroup${flat ? `, windows under sd ${flat} skipped` : ""}, ${fcnBytes(src)} bytes of workgroup memory (the device gives ${limit}), levels 1 to ${LEVELS - 1}`);
}

// dimsAt(l): the level's (w, h) at the current capacity; bases[l]: where its slots start; slots: a frame's total.
export function ensure(fh, dimsAt, bases, slots) {
  const d = fh.device;
  fh.fcnLevels = [];
  for (let l = 1; l < LEVELS; l++) {
    const uni = d.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST }), ab = new ArrayBuffer(32), u = new Uint32Array(ab), f = new Float32Array(ab);
    const [w, h] = dimsAt(l);
    u.set([l, w, h, slots]); f[4] = TAU; f[5] = EPS; u[6] = bases[l]; u[7] = ceilDiv(w, TILE);
    d.queue.writeBuffer(uni, 0, ab);
    const group = d.createBindGroup({ layout: fh.fcn.bgl, entries: [
      { binding: 0, resource: { buffer: fh.level[l - 1].buf } }, { binding: 1, resource: { buffer: fh.framesBuf } }, { binding: 2, resource: { buffer: fh.rawBuf } },
      { binding: 3, resource: { buffer: fh.countsBuf } }, { binding: 4, resource: { buffer: uni } }] });
    fh.fcnLevels.push({ w, h, group });
  }
}

export function dispatch(pass, fh, B) {
  pass.setPipeline(fh.fcn.p);
  for (const lv of fh.fcnLevels) {
    pass.setBindGroup(0, lv.group);
    pass.dispatchWorkgroups(ceilDiv(ceilDiv(lv.w, TILE), fh.fcnRegion), ceilDiv(lv.h, TILE), B);
  }
}
