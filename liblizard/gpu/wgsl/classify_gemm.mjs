// F4 as a network, GEMM-shaped: the same stage as classify.mjs (bindings, uniforms, roles, the reading layout and
// RANK are identical, so the decoder switches by changing its import), with the convolutions restructured as small
// matrix multiplies. classify.mjs gives each lane one output channel at four positions and reloads three weights
// and three input vectors for every twelve multiply-adds; on the iGPU that is load-bound. Here a workgroup takes
// several patches at once and each lane owns a tile of TC channels x 4 TX positions, accumulated in registers:
// one row of inputs (three vectors, two of them shared across the taps by register shifting) and TC x 3 weights
// serve 12 TC TX multiply-adds. Weights are packed for it (packWeights here, not cnn/weights.mjs): per (ci, ky, kx)
// all output channels in one run, so a lane's block of TC channels is one or two 16-byte loads.
//
// Layers whose outputs are too few for 256 lanes at TC >= 2 split the input channels across lanes (KS slices) and
// reduce the partial sums by subgroup shuffles (`subgroups`, when the device has the feature) or through
// workgroup memory. The first fully connected layer is a GEMM too: a lane owns four outputs of one patch over a
// slice of the inputs, partials reduced through workgroup memory.
//
// Two more things the measurements asked for (cnn/test_gemm.mjs, MODE=time with stop=): the cut is generated once
// per pyramid level with the level's dimensions and fetch fixed outside the sample loop, and a conv row outside
// the input (the zero padding) is read at a clamped row and multiplied by an exact 0 or 1 rather than branched
// around, so the compiler can issue a whole input channel's loads over the multiply-adds. What limits the kernel
// now is occupancy: the LDS footprint caps the waves a SIMD holds (30 KB is four workgroups a WGP on the RDNA-2
// iGPU), and every wider tile or padded pitch that cost memory measured slower.
//
// Precision as classify.mjs: f16 stores weights and activations as f16 and multiplies in f32 (an f16 x f16 product
// is exact in f32), accumulation is f32 in both. Sums are ordered differently, so values move by f32 rounding.
//
// Workgroup memory: patches x (X + Y) vec4s plus an f32 scratch. The plan picks the most patches a workgroup can
// hold within `budget` (32 KB by default, a phone's usual limit): cnn-v3 f16 four, cnn-v1 f16 two, cnn-v1 f32 one.
// ai: The decoder dispatches ceil(cap / P) (or keep) workgroups a frame, P the plan's patches a workgroup; a workgroup
// ai: past the frame's count returns at once. (It launched cap until 2026-09-30, three in four of cnn-v3's returning
// ai: at once: 0.07 ms a frame on the iGPU, not the few microseconds this comment gave it.)
//
// ai: int8 (2026-09-26): a q8 classifier file (arch.quant, the contract in cnn/q8c.mjs, scheme int8-v1) runs on
// ai: codes, computing q8c.mjs forwardCodesC bit for bit on the kernel's own Xq. The cut and the standardisation are
// ai: the float form's, in f32; the patch is stored as Xq = clamp(round(x * INV), -QMAX, QMAX), four samples along x
// ai: a u32 (pack4xI8), and every conv's output as words of four channels at one position, a vec4<u32> four
// ai: positions along x at [channel group][y][x / 4]. A conv lane owns TC channels (4 or 8) x 4 TX positions in
// ai: vec4<i32> accumulators: per (group, ky, kx) one weight word a channel (a vec4<u32> load for four channels) and
// ai: dot4I8Packed on each position's word; conv1 (one input channel, stride 2) cuts each tap row's word of three
// ai: samples from two patch words by byte shifts, its weight word's fourth byte 0. Sums are exact i32 in any order,
// ai: so a K-split reduces in i32 (shuffles or workgroup memory) with nothing to round; requantisation is
// ai: clamp(round(f32(acc) * m), 0, QMAX), the bias folded into acc. fc1 runs on the last conv's words in the kernel's
// ai: order (the packer permutes torch's (c, y, x) rows to it) and writes relu(f32(acc) * s) in f32; fc2 and the
// ai: readings are the f32 form's. packWeightsInt8 lays the codes, biases and scales out (weightLayoutQ8); dump (a
// ai: check build) writes each patch's Xq, codes, fc1 output, z and the patch's mean and deviation to a thirteenth
// ai: binding (dumpLayoutQ8), and checkDumpQ8 holds one patch's dump to the contract layer by layer (test_gemm.mjs and
// ai: the phone's scripts/pages/gpu_cls_check.mjs). Four patches a workgroup for both nets: cnn-v3 18,688 bytes, cnn-v1 30,976.
import { DIMS } from "./common.mjs";
import { KAPPA, PATCH } from "../cnn/patch.mjs";
import { planArch, toHalf } from "../cnn/weights.mjs";
import { readQuantC, forwardCodesC } from "../cnn/q8c.mjs";
import { packI8x4, quantiseInput } from "../cnn/q8.mjs";
import { INT8 } from "../constants.mjs";   // ai: not ../decoder.mjs, which imports this module (gpu/constants.mjs)
export { rankSource, CASCADE } from "./classify.mjs";

// Copied from classify.mjs (which copied finder.mjs): the patch has to be cut exactly as finder.mjs reads the pyramid.
const LEVELS = /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }   // d[l] = (stride, rows) of level l's buffer
fn pix(l: u32, f: u32, x: i32, y: i32) -> f32 {
  let F = frames[f];
  let w = i32((F.w + (1u << l) - 1u) >> l);
  let h = i32((F.h + (1u << l) - 1u) >> l);
  let xc = clamp(x, 0, w - 1);
  let yc = clamp(y, 0, h - 1);
  if (l == 0u) { return textureLoad(img, vec2i(xc, yc), i32(f), 0).r; }
  let s = LD.d[l];
  let i = (f * s.y + u32(yc)) * s.x + u32(xc);
  switch l {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
fn look(l: u32, f: u32, p: vec2f) -> f32 {
  let q = p / f32(1u << l) - vec2f(0.5);
  let q0 = floor(q);
  let a = q - q0;
  let i = vec2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
fn levelFor(u: f32) -> u32 { return u32(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
`;
const LEVEL_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> lv1: array<f32>;
@group(0) @binding(2) var<storage, read> lv2: array<f32>;
@group(0) @binding(3) var<storage, read> lv3: array<f32>;
@group(0) @binding(4) var<storage, read> lv4: array<f32>;
@group(0) @binding(5) var<storage, read> frames: array<Frame>;
@group(0) @binding(6) var<uniform> LD: LevelDims;
`;

const WG = 256;
export const BUDGET = 32768;   // workgroup memory a plan may use, bytes
const RED_MIN = 1024;          // f32 scratch the conv and fc partials may take
const TCF = 4;                 // outputs a lane owns in the first fc layer

// Rows of vec4s: S / 4, plus one when padded. A stride-2 layer reads every other row, and with a pitch of 4 or 8
// vec4s those rows share LDS banks (every row of the 32 x 32 patch lands on the same banks); an odd pitch spreads
// them. pad 0 pads nothing, 1 the patch only, 2 every buffer with rows of 8 or more samples; the pad costs memory
// and so patches a workgroup, which is why it is a level.
const pitchOf = (S, pad) => S / 4 + (S >= (pad >= 2 ? 8 : pad === 1 ? PATCH : Infinity) ? 1 : 0);

// The weight buffer's layout, in vec4<u32> units (E elements a vec: 8 f16 or 4 f32). Conv: per (ci, ky, kx) the
// output channels in one run padded to whole vecs, then the biases; fc: rows [out][in] padded to whole vecs, then
// the biases. It depends on the arch and the precision only, so the packer and the shader agree by construction.
function weightLayout(plan, f16) {
  const E = f16 ? 8 : 4, layers = {};
  let at = 0;
  for (const L of plan.layers) {
    if (L.type === "conv") {
      const kxVecs = Math.ceil(L.outC / E);
      layers[L.name] = { w: at, kxVecs, b: at + L.inC * 9 * kxVecs };
      at += L.inC * 9 * kxVecs + Math.ceil(L.outC / E);
    } else {
      const rowVecs = Math.ceil(L.inN / E);
      layers[L.name] = { w: at, rowVecs, b: at + L.outN * rowVecs };
      at += L.outN * rowVecs + Math.ceil(L.outN / E);
    }
  }
  return { E, layers, vecs: at, key: `${plan.name}:${f16 ? "f16" : "f32"}` };
}

// A conv layer's tile: TC channels x 4 TX positions a lane, KS slices of the input channels. T is the outputs a
// lane must produce for 256 lanes to cover patches x outC x outS^2; the widest channel block first (fewer loads a
// multiply-add), splitting the input channels when the outputs alone are too few, as long as the partials fit.
// ai: int8: TC 8 or 4 (a lane packs whole words of four channels), and the slices split the input's words (groups of
// ai: four channels; conv1's one channel is not split).
function convTile(L, patches, partialsMax = RED_MIN, subgroups = false, int8 = false) {
  const T = (patches * L.outC * L.outS * L.outS) / WG;
  if (!Number.isInteger(T) || T < 1) return null;
  for (const TC of int8 ? [8, 4] : [8, 4, 2, 1]) {
    if (L.outC % TC) continue;
    for (const KS of [1, 2, 4]) {
      if ((int8 ? (L.inC === 1 ? 1 : L.inC / 4) : L.inC) % KS) continue;
      const TX = (T * KS) / (4 * TC);
      if (!Number.isInteger(TX) || TX < 1 || TX > 2 || (L.outS / 4) % TX) continue;
      // With subgroups the slices of a tile are adjacent lanes and reduce by shuffles, so no partials are stored.
      const partials = subgroups ? 0 : ((KS - 1) / KS) * WG * 4 * TC * TX;
      if (partials > partialsMax) continue;
      return { TC, TX, KS, T, nX: L.outS / (4 * TX), NB: L.outC / TC, partials };
    }
  }
  return null;
}

// The plan: patches a workgroup, buffers, tiles, scratch, bytes. `patches` forces the count (for measurement);
// `pad` the row pitch level (see pitchOf); `budget` bounds the workgroup memory.
// `subgroups` (the adapter's "subgroups" feature, and the caller's choice) reduces a K-split's slices by subgroup
// shuffles; `unroll` is the input-channel loop's unrolling (1 or 2).
// ai: int8: the int8 form (planQ8), for an arch with arch.quant: its rows are unpadded and its partials REDI_MAX, so
// ai: pad and partials do not apply to it; unroll must be 1.
export function classifyPlan(arch, { f16 = false, int8 = false, budget = BUDGET, pad = 1, patches = 0, partials = RED_MIN, subgroups = false, unroll = 1 } = {}) {
  const plan = planArch(arch);
  const convs = plan.layers.filter((l) => l.type === "conv"), fcs = plan.layers.filter((l) => l.type === "fc");
  if (!convs.length) throw new Error(`arch ${plan.name}: no conv layer`);
  for (const l of convs) {
    if (l.inS % 4 || l.outS % 4 || (l.stride === 2 && l.inS !== 2 * l.outS) || (l.stride !== 1 && l.stride !== 2)) throw new Error(`arch ${plan.name}: ${l.name} ${l.inS} -> ${l.outS} at stride ${l.stride}; sizes must be multiples of 4`);
  }
  if (fcs[0].outN % TCF) throw new Error(`arch ${plan.name}: ${fcs[0].name} has ${fcs[0].outN} outputs, a multiple of ${TCF} is needed`);
  for (const l of fcs) if (l.outN > WG) throw new Error(`arch ${plan.name}: ${l.name} has ${l.out} outputs, the workgroup has ${WG}`);
  if (int8) return planQ8(arch, plan, convs, fcs, { f16, budget, patches, subgroups, unroll });
  if (arch.quant) throw new Error(`arch ${plan.name}: an int8 file (arch.quant) runs only in the int8 form`);
  const layout = weightLayout(plan, f16), E = layout.E;
  // Buffers X (the patch, then every second conv's output) and Y (the others), per patch, in vec4s.
  const vecsOf = (l) => l.outC * l.outS * pitchOf(l.outS, pad);
  const XS = Math.max(PATCH * pitchOf(PATCH, pad), ...convs.filter((_, k) => k % 2 === 1).map(vecsOf));
  const YS = Math.max(...convs.filter((_, k) => k % 2 === 0).map(vecsOf));
  const elBytes = f16 ? 8 : 16;
  const tryPatches = (P) => {
    if (WG % P || (WG / P) % 8) return null;
    const tiles = convs.map((l) => convTile(l, P, partials, subgroups));
    if (tiles.some((t) => !t)) return null;
    const fc1 = fcs[0], NB = fc1.outN / TCF, KSf = WG / (P * NB), slice = fc1.inN / KSf;
    if (!Number.isInteger(KSf) || KSf < 1 || !Number.isInteger(slice) || slice % E) return null;
    const fcOut = {};
    let at = WG * TCF;
    for (const l of fcs) { fcOut[l.name] = at; at += P * l.outN; }
    const red = Math.max(576, RED_MIN, ...tiles.map((t) => t.partials), at);
    const bytes = P * (XS + YS) * elBytes + 4 * red;
    return { P, tiles, fc1: { NB, KSf, slice }, fcOut, red, bytes };
  };
  let chosen = null;
  if (patches) { chosen = tryPatches(patches); if (!chosen) throw new Error(`arch ${plan.name}: ${patches} patches a workgroup does not tile`); }
  else for (const P of [8, 4, 2, 1]) { const c = tryPatches(P); if (c && c.bytes <= budget) { chosen = c; break; } }
  if (!chosen) throw new Error(`arch ${plan.name}: no patch count fits ${budget} bytes`);
  return { ...plan, convs, fcs, f16, pad, layout, XS, YS, ...chosen, budget, subgroups, unroll };
}

// Weights packed for this kernel (the same fields cnn/weights.mjs packWeights returns, so the decoder can take
// either); `offsets` carries the layout and its key, which classifySource checks.
function pack(json, f16) {
  if (json.arch?.quant) throw new Error(`weights: ${json.arch.name} is an int8 file (arch.quant); packWeightsInt8 packs it`);
  const arch = json.arch, plan = planArch(arch), layout = weightLayout(plan, f16), E = layout.E;
  const size = (shape) => shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(layout.vecs * E);
  for (const L of plan.layers) {
    const l = json.layers.find((x) => x.name === L.name);
    if (!l) throw new Error(`weights: no layer ${L.name}`);
    if (JSON.stringify(l.shape) !== JSON.stringify(L.shape) || l.w.length !== size(L.shape) || l.b.length !== L.shape[0]) throw new Error(`weights: ${L.name} is ${JSON.stringify(l.shape)} (${l.w.length} + ${l.b.length}), expected ${JSON.stringify(L.shape)}`);
    const o = layout.layers[L.name];
    if (L.type === "conv") {
      for (let co = 0; co < L.outC; co++) for (let ci = 0; ci < L.inC; ci++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        data[(o.w + ((ci * 3 + ky) * 3 + kx) * o.kxVecs) * E + co] = l.w[((co * L.inC + ci) * 3 + ky) * 3 + kx];
      }
      for (let co = 0; co < L.outC; co++) data[o.b * E + co] = l.b[co];
    } else {
      for (let n = 0; n < L.outN; n++) {
        for (let k = 0; k < L.inN; k++) data[(o.w + n * o.rowVecs) * E + k] = l.w[n * L.inN + k];
        data[o.b * E + n] = l.b[n];
      }
    }
  }
  const out = f16 ? Uint16Array.from(data, toHalf) : data;
  return { data: out, offsets: { gemm: true, f16, key: layout.key, layers: layout.layers }, arch, val: json.val ?? null, total: plan.total, macs: plan.macs };
}
export const packWeights = (json) => pack(json, false);
export const packWeightsF16 = (json) => pack(json, true);

// WGSL pieces. A vec4<u32> `v` holds E weights; these give f32 expressions for them.
const elems = (v, f16) => f16
  ? { code: `let ${v}a = bitcast<vec4<f16>>(${v}.xy); let ${v}b = bitcast<vec4<f16>>(${v}.zw);`, at: ["xyzw".split("").map((c) => `f32(${v}a.${c})`), "xyzw".split("").map((c) => `f32(${v}b.${c})`)].flat() }
  : { code: `let ${v}f = bitcast<vec4<f32>>(${v});`, at: "xyzw".split("").map((c) => `${v}f.${c}`) };
// A block of TC consecutive weights starting at element `sub` (a lane constant, a multiple of TC) of vec `v`.
// The selection is done on the u32 words, so TC = E costs nothing and TC = 1 two or three selects.
function block(v, TC, f16, sub) {
  const E = f16 ? 8 : 4;
  if (TC === E) return elems(v, f16);
  if (f16) {
    if (TC === 4) return { code: `let ${v}a = bitcast<vec4<f16>>(select(${v}.xy, ${v}.zw, ${sub} == 4u));`, at: "xyzw".split("").map((c) => `f32(${v}a.${c})`) };
    const word = `select(select(${v}.x, ${v}.y, (${sub} >> 1u) == 1u), select(${v}.z, ${v}.w, (${sub} >> 1u) == 3u), ${sub} >= 4u)`;
    if (TC === 2) return { code: `let ${v}a = bitcast<vec2<f16>>(${word});`, at: [`f32(${v}a.x)`, `f32(${v}a.y)`] };
    return { code: `let ${v}a = bitcast<vec2<f16>>(${word}); let ${v}s = select(${v}a.x, ${v}a.y, (${sub} & 1u) == 1u);`, at: [`f32(${v}s)`] };
  }
  if (TC === 2) return { code: `let ${v}a = bitcast<vec2<f32>>(select(${v}.xy, ${v}.zw, ${sub} == 2u));`, at: [`${v}a.x`, `${v}a.y`] };
  return { code: `let ${v}s = bitcast<f32>(select(select(${v}.x, ${v}.y, ${sub} == 1u), select(${v}.z, ${v}.w, ${sub} == 3u), ${sub} >= 2u));`, at: [`${v}s`] };
}
// Element `idx` (a lane value, 0..E-1) of vec `v`, as f32.
const elem = (v, idx, f16) => f16
  ? `f32(select(bitcast<vec2<f16>>(select(select(${v}.x, ${v}.y, (${idx} >> 1u) == 1u), select(${v}.z, ${v}.w, (${idx} >> 1u) == 3u), ${idx} >= 4u)).x, bitcast<vec2<f16>>(select(select(${v}.x, ${v}.y, (${idx} >> 1u) == 1u), select(${v}.z, ${v}.w, (${idx} >> 1u) == 3u), ${idx} >= 4u)).y, (${idx} & 1u) == 1u))`
  : `bitcast<f32>(select(select(${v}.x, ${v}.y, ${idx} == 1u), select(${v}.z, ${v}.w, ${idx} == 3u), ${idx} >= 2u))`;

// One conv layer. src/dst: buffer names with per-patch strides SS/DS; pin/pout the row pitches; T the element type.
function convGemm(L, tile, P, f16, src, SS, dst, DS, pin, pout, lw, T, subgroups = false, unroll = 1) {
  const { TC, TX, KS, nX, NB } = tile, E = f16 ? 8 : 4, rowG = L.inS / 4, ciPer = L.inC / KS;
  const U = ciPer % unroll === 0 ? unroll : 1;
  const vecsPerBlock = Math.ceil(TC / E);
  const accs = [];
  for (let c = 0; c < TC; c++) for (let t = 0; t < TX; t++) accs.push(`a${c}_${t}`);
  // Bias into the slice-0 accumulators; the other slices start at zero.
  const bias = [];
  for (let vb = 0; vb < vecsPerBlock; vb++) {
    const bl = block(`bv${vb}`, Math.min(TC, E), f16, "wsub");
    bias.push(`let bv${vb} = W[${lw.b}u + wvec + ${vb}u]; ${bl.code}`);
    for (let c = 0; c < Math.min(TC, E); c++) for (let t = 0; t < TX; t++) bias.push(`a${vb * E + c}_${t} = vec4f(${bl.at[c]});`);
  }
  // The input row: TX vec4s of outputs need, at stride 1, TX vec4s of inputs plus one sample each side; at stride
  // 2, 2 TX vec4s plus one sample on the left. The three tap-shifted vectors i0/i1/i2 come from register shifts.
  // A row above or below the input (the zero padding) is read at a clamped row and multiplied by an exact 0 or 1
  // (`okm`, in the activation type, so it is one packed multiply a pair in f16) instead of branched around: a
  // wave always holds a lane on row 0, so a branch never skipped the work, and it kept the compiler from
  // issuing the next row's loads over this row's multiply-adds.
  const loads = [], i0 = [], i1 = [], i2 = [];
  const cv = (e) => (f16 ? `vec4f(${e} * okm)` : `(${e} * okm)`), cs = (e) => (f16 ? `f32(${e} * okm)` : `(${e} * okm)`);
  if (L.stride === 1) {
    for (let t = 0; t < TX; t++) loads.push(`let m${t} = ${cv(`${src}[r + x4 + ${t}u]`)};`);
    loads.push(`let lf = select(0.0, ${cs(`${src}[r + max(x4, 1u) - 1u].w`)}, x4 > 0u);`);
    loads.push(`let rt = select(0.0, ${cs(`${src}[r + min(x4 + ${TX}u, ${rowG - 1}u)].x`)}, x4 + ${TX}u < ${rowG}u);`);
    for (let t = 0; t < TX; t++) {
      i0.push(`vec4f(${t ? `m${t - 1}.w` : "lf"}, m${t}.xyz)`); i1.push(`m${t}`); i2.push(`vec4f(m${t}.yzw, ${t < TX - 1 ? `m${t + 1}.x` : "rt"})`);
    }
  } else {
    for (let t = 0; t < TX; t++) loads.push(`let m${t}a = ${cv(`${src}[r + 2u * (x4 + ${t}u)]`)}; let m${t}b = ${cv(`${src}[r + 2u * (x4 + ${t}u) + 1u]`)};`);
    loads.push(`let lf = select(0.0, ${cs(`${src}[r + 2u * max(x4, 1u) - 1u].w`)}, x4 > 0u);`);
    for (let t = 0; t < TX; t++) {
      i0.push(`vec4f(${t ? `m${t - 1}b.w` : "lf"}, m${t}a.y, m${t}a.w, m${t}b.y)`); i1.push(`vec4f(m${t}a.x, m${t}a.z, m${t}b.x, m${t}b.z)`); i2.push(`vec4f(m${t}a.y, m${t}a.w, m${t}b.y, m${t}b.w)`);
    }
  }
  const ins = [i0, i1, i2].map((iv, kx) => iv.map((e, t) => `let i${kx}_${t} = ${e};`).join(" ")).join("\n          ");
  // The weights of the three taps for this lane's channel block, then the multiply-adds.
  const wcode = [], fmas = [];
  for (let kx = 0; kx < 3; kx++) {
    for (let vb = 0; vb < vecsPerBlock; vb++) {
      const bl = block(`w${kx}_${vb}`, Math.min(TC, E), f16, "wsub");
      wcode.push(`let w${kx}_${vb} = W[wb + ${kx * lw.kxVecs + vb}u]; ${bl.code}`);
      for (let c = 0; c < Math.min(TC, E); c++) for (let t = 0; t < TX; t++) fmas.push(`a${vb * E + c}_${t} = fma(vec4f(${bl.at[c]}), i${kx}_${t}, a${vb * E + c}_${t});`);
    }
  }
  const outAt = (c, t) => `${dst}[dst0 + ((cbE + ${c}u) * ${L.outS}u + y) * ${pout}u + x4 + ${t}u]`;
  const finish = accs.map((a, k) => `${outAt(Math.floor(k / TX), k % TX)} = vec4<${T}>(max(${a}, vec4f(0.0)));`).join("\n      ");
  const per = 4 * TC * TX, tilesN = WG / KS;
  const partAt = (s, k) => `red[((${s} - 1u) * ${tilesN}u + ti) * ${per}u + ${k}u]`;
  // The slices' sum: by shuffles when the subgroup holds them (adjacent lanes), else through the scratch.
  const reduce = KS === 1 ? finish : subgroups ? /* wgsl */ `
    ${[1, 2].filter((m) => m < KS).map((m) => accs.map((a) => `${a} += subgroupShuffleXor(${a}, ${m}u);`).join(" ")).join("\n    ")}
    if (ks == 0u) {
      ${finish}
    }` : /* wgsl */ `
    let ti = li / ${KS}u;
    if (ks > 0u) {
      ${accs.map((a, k) => [0, 1, 2, 3].map((e) => `${partAt("ks", 4 * k + e)} = ${a}[${e}];`).join(" ")).join("\n      ")}
    }
    workgroupBarrier();
    if (ks == 0u) {
      ${Array.from({ length: KS - 1 }, (_, s) => accs.map((a, k) => `${a} += vec4f(${[0, 1, 2, 3].map((e) => partAt(`${s + 1}u`, 4 * k + e)).join(", ")});`).join(" ")).join("\n      ")}
      ${finish}
    }`;
  return /* wgsl */ `
  // ${L.name}: ${L.inC} x ${L.inS}^2 -> ${L.outC} x ${L.outS}^2 stride ${L.stride}; tile ${TC} ch x ${4 * TX} px, ${KS} slice${KS > 1 ? "s" : ""}
  {
    let ks = li % ${KS}u;
    let r0 = li / ${KS}u;
    let xg = r0 % ${nX}u;
    let r1 = r0 / ${nX}u;
    let y = r1 % ${L.outS}u;
    let r2 = r1 / ${L.outS}u;
    let cb = r2 % ${NB}u;
    let p = r2 / ${NB}u;
    let cbE = cb * ${TC}u;
    let wvec = cbE / ${E}u;
    let wsub = cbE % ${E}u;
    let x4 = xg * ${TX}u;
    let src0 = p * ${SS}u;
    let dst0 = p * ${DS}u;
    ${accs.map((a) => `var ${a} = vec4f(0.0);`).join(" ")}
    if (ks == 0u) { ${bias.join(" ")} }
    for (var ci0 = ks * ${ciPer}u; ci0 < (ks + 1u) * ${ciPer}u; ci0 += ${U}u) {
      ${Array.from({ length: U }, (_, uu) => [0, 1, 2].map((ky) => /* wgsl */ `
      {
        let ci = ci0 + ${uu}u;
        let iy = i32(y) * ${L.stride} - 1 + ${ky};
        let okm = select(${T}(0.0), ${T}(1.0), iy >= 0 && iy < ${L.inS});
        let r = src0 + (ci * ${L.inS}u + u32(clamp(iy, 0, ${L.inS - 1}))) * ${pin}u;
        ${loads.join(" ")}
        ${ins}
        let wb = ${lw.w}u + ((ci * 3u + ${ky}u) * 3u) * ${lw.kxVecs}u + wvec;
        ${wcode.join("\n        ")}
        ${fmas.join("\n        ")}
      }`).join("")).join("")}
    }
    ${reduce}
  }
  workgroupBarrier();
`;
}

// The first fc layer as a GEMM over the last conv's buffer: lane (patch, block of TCF outputs, slice of the inputs).
function fc1Gemm(L, plan, src, SS, pitch, lw, S) {
  const { NB, KSf, slice } = plan.fc1, E = plan.layout.E, f16 = plan.f16, steps = slice / E, out = plan.fcOut[L.name];
  // Element k of the flattened (c, y, x) input sits at vec ((c S + y) pitch + x / 4).
  const vecAt = (k) => `src0 + ((${k} / ${S * S}u) * ${S}u + (${k} % ${S * S}u) / ${S}u) * ${pitch}u + (${k} % ${S}u) / 4u`;
  const xs = [];
  for (let q = 0; q < E / 4; q++) xs.push(`let k${q} = k + ${4 * q}u; let x${q} = ${f16 ? "vec4f" : ""}(${src}[${vecAt(`k${q}`)}]);`);
  const wl = [], fm = [];
  for (let c = 0; c < TCF; c++) {
    const bl = elems(`w${c}`, f16);
    wl.push(`let w${c} = W[${lw.w}u + (o0 + ${c}u) * ${lw.rowVecs}u + kv]; ${bl.code}`);
    for (let q = 0; q < E / 4; q++) fm.push(`f${c} += dot(x${q}, vec4f(${bl.at.slice(4 * q, 4 * q + 4).join(", ")}));`);
  }
  return /* wgsl */ `
  // ${L.name}: ${L.inN} -> ${L.outN}; lane = (patch, ${TCF} outputs, slice of ${slice} inputs), ${KSf} slices
  {
    let ks = li % ${KSf}u;
    let r0 = li / ${KSf}u;
    let ob = r0 % ${NB}u;
    let p = r0 / ${NB}u;
    let src0 = p * ${SS}u;
    let o0 = ob * ${TCF}u;
    ${Array.from({ length: TCF }, (_, c) => `var f${c} = 0.0;`).join(" ")}
    for (var v = 0u; v < ${steps}u; v++) {
      let k = ks * ${slice}u + v * ${E}u;
      let kv = k / ${E}u;
      ${xs.join(" ")}
      ${wl.join("\n      ")}
      ${fm.join(" ")}
    }
    ${Array.from({ length: TCF }, (_, c) => `red[li * ${TCF}u + ${c}u] = f${c};`).join(" ")}
  }
  workgroupBarrier();
  if (li < ${plan.P * L.outN}u) {
    let p = li / ${L.outN}u;
    let o = li % ${L.outN}u;
    let bv = W[${lw.b}u + o / ${E}u];
    var acc = ${elem("bv", `(o % ${E}u)`, f16)};
    for (var s = 0u; s < ${KSf}u; s++) { acc += red[((p * ${NB}u + o / ${TCF}u) * ${KSf}u + s) * ${TCF}u + o % ${TCF}u]; }
    red[${out}u + li] = max(acc, 0.0);
  }
  workgroupBarrier();
`;
}

// A later fc layer: one lane an output of a patch, inputs from the previous layer's outputs in the scratch.
function fcPlain(L, plan, inAt, lw, relu) {
  const E = plan.layout.E, f16 = plan.f16, out = plan.fcOut[L.name], steps = Math.ceil(L.inN / E);
  const bl = elems("wv", f16);
  const terms = bl.at.map((w, e) => `if (${e}u < ${L.inN}u - k) { acc += ${w} * red[${inAt}u + p * ${L.inN}u + k + ${e}u]; }`);
  return /* wgsl */ `
  // ${L.name}: ${L.inN} -> ${L.outN}
  if (li < ${plan.P * L.outN}u) {
    let p = li / ${L.outN}u;
    let o = li % ${L.outN}u;
    let bv = W[${lw.b}u + o / ${E}u];
    var acc = ${elem("bv", `(o % ${E}u)`, f16)};
    for (var v = 0u; v < ${steps}u; v++) {
      let k = v * ${E}u;
      let wv = W[${lw.w}u + o * ${lw.rowVecs}u + v]; ${bl.code}
      ${terms.join(" ")}
    }
    red[${out}u + li] = ${relu ? "max(acc, 0.0)" : "acc"};
  }
  workgroupBarrier();
`;
}

// ai: The float stage's parts for classifySource: the directives, the workgroup declarations, the patch store (x in
// ai: the activation type) and the network (convs, then fcs).
function partsFloat(plan, { lw, stop, subgroups, unroll, G, T }) {
  const P = plan.P, f16 = plan.f16, pitch32 = pitchOf(PATCH, plan.pad);
  let convs = "";
  plan.convs.forEach((L, k) => {
    if (stop === "cut" || (stop && stop.startsWith("conv") && +stop.slice(4) < k + 1)) return;
    const [src, SS, dst, DS] = k % 2 === 0 ? ["bufX", plan.XS, "bufY", plan.YS] : ["bufY", plan.YS, "bufX", plan.XS];
    convs += convGemm(L, plan.tiles[k], P, f16, src, SS, dst, DS, pitchOf(L.inS, plan.pad), pitchOf(L.outS, plan.pad), lw(L.name), T, subgroups, unroll);
  });
  const last = plan.convs[plan.convs.length - 1], lastBuf = plan.convs.length % 2 ? ["bufY", plan.YS] : ["bufX", plan.XS];
  let fcs = stop === "cut" || stop?.startsWith("conv") ? "" : fc1Gemm(plan.fcs[0], plan, lastBuf[0], lastBuf[1], pitchOf(last.outS, plan.pad), lw(plan.fcs[0].name), last.outS);
  for (let k = 1; k < plan.fcs.length && !stop; k++) fcs += fcPlain(plan.fcs[k], plan, plan.fcOut[plan.fcs[k - 1].name], lw(plan.fcs[k].name), k < plan.fcs.length - 1);
  const store = Array.from({ length: P }, (_, k) => `{ let q = lq + ${G * k}u; bufX[cp * ${plan.XS}u + (q / 8u) * ${pitch32}u + (q % 8u)] = vec4<${T}>((s${k} - vec4f(mu)) / (sd + 0.02)); }`).join("\n  ");
  const decls = [
    `// ${plan.name}, ${T}: ${P} patches a workgroup, ${plan.bytes} bytes of workgroup memory (X ${plan.XS}, Y ${plan.YS} vec4s a patch, ${plan.red} f32 scratch).`,
    `var<workgroup> bufX: array<vec4<${T}>, ${P * plan.XS}>;`,
    `var<workgroup> bufY: array<vec4<${T}>, ${P * plan.YS}>;`,
    `var<workgroup> red: array<f32, ${plan.red}>;`,
  ].join("\n");
  return { directives: (f16 ? "enable f16;\n" : "") + (subgroups ? "enable subgroups;\n" : ""), decls, store, convs, fcs };
}

// ai: ---- The int8 form (a q8 file; the contract in cnn/q8c.mjs) ----

// ai: The int8 form's rows are unpadded: the patch's row is 2 vec4<u32>s (16 samples a vec4), a conv output's S / 4
// ai: (four positions a vec4). Padding lost on the iGPU (test_gemm MODE=time, 2026-09-26, the cascade's shape on
// ai: 20-04-09: both nets 0.717 ms a frame unpadded, 0.731 with the patch padded, 0.741 with every row), and was
// ai: deleted.
const PITCH_PATCH_Q8 = PATCH / 16;

// ai: The int8 weight buffer, in vec4<u32>s. A conv: per tap a run of outC / 4 vec4s, a word an output channel
// ai: (conv1: per ky, the row's three taps in bytes 0 to 2 and 0 in byte 3; a later conv: per (group, ky, kx), the
// ai: group's four input channels), then bq (i32) and m (f32) a channel. fc1: rows [n][K / 16] in the kernel's
// ai: order (vec kv = (g S + y) S / 4 + x / 4 of the last conv's words, g its channel group, x = 4 (kv % (S / 4)) +
// ai: the component), then bq and s. fc2: the f32 form's (rows [out][in] in whole vec4s, then the biases).
function weightLayoutQ8(plan) {
  const layers = {};
  let at = 0;
  for (const L of plan.layers) {
    if (L.type === "conv") {
      const kxVecs = L.outC / 4, taps = L.inC === 1 ? 3 : (L.inC / 4) * 9;
      layers[L.name] = { w: at, kxVecs, b: at + taps * kxVecs, m: at + (taps + 1) * kxVecs };
      at += (taps + 2) * kxVecs;
    } else if (L.name === "fc1") {
      const rowVecs = L.inN / 16;
      layers[L.name] = { w: at, rowVecs, b: at + L.outN * rowVecs, m: at + L.outN * rowVecs + L.outN / 4 };
      at += L.outN * rowVecs + L.outN / 2;
    } else {
      const rowVecs = Math.ceil(L.inN / 4);
      layers[L.name] = { w: at, rowVecs, b: at + L.outN * rowVecs };
      at += L.outN * rowVecs + Math.ceil(L.outN / 4);
    }
  }
  return { E: 4, layers, vecs: at, key: `${plan.name}:int8` };
}

// ai: classifyPlan's int8 branch: the same tiles over words (convTile int8), buffers of vec4<u32>, and two scratch
// ai: arrays, red (f32: the patch statistics, then fc1's and fc2's outputs) and redi (i32: the K-split partials and
// ai: fc1's slices). Without subgroups a K-split's partials may take REDI_MAX (a four-slice split of a 4 x 4 tile,
// ai: the smallest tile int8 has), so every arch that tiles with shuffles tiles without them. Patches a workgroup:
// ai: the most of 4, 2, 1 that tile and fit; 8 lost on the iGPU (cnn-v3 0.676 against 0.486 ms a frame at 4) and was
// ai: deleted, and cnn-v1 at 2 lost to 4 (0.253 against 0.232).
const REDI_MAX = 3 * WG * TCF;
function planQ8(arch, plan, convs, fcs, { f16, budget, patches, subgroups, unroll }) {
  if (f16) throw new Error(`arch ${plan.name}: int8 or f16, not both`);
  if (unroll !== 1) throw new Error(`arch ${plan.name}: the int8 form does not unroll`);
  if (!arch.quant) throw new Error(`arch ${plan.name}: a float file (no arch.quant); the int8 form runs a q8 file`);
  const c1 = convs[0];
  if (c1.inC !== 1 || c1.stride !== 2) throw new Error(`arch ${plan.name}: the int8 conv1 takes one channel at stride 2`);
  for (const l of convs) if (l.outC % 4 || (l !== c1 && l.inC % 4)) throw new Error(`arch ${plan.name}: ${l.name} ${l.inC} -> ${l.outC} channels; int8 packs them in fours`);
  if (fcs.length !== 2) throw new Error(`arch ${plan.name}: the int8 form runs fc1, then fc2`);
  const layout = weightLayoutQ8(plan);
  const vecsOf = (l) => (l.outC / 4) * l.outS * (l.outS / 4);
  const XS = Math.max(PATCH * PITCH_PATCH_Q8, ...convs.filter((_, k) => k % 2 === 1).map(vecsOf));
  const YS = Math.max(...convs.filter((_, k) => k % 2 === 0).map(vecsOf));
  const tryPatches = (P) => {
    if (WG % P || (WG / P) % 8) return null;
    const tiles = convs.map((l) => convTile(l, P, REDI_MAX, subgroups, true));
    if (tiles.some((t) => !t)) return null;
    const fc1 = fcs[0], NB = fc1.outN / TCF, KSf = WG / (P * NB), slice = fc1.inN / 16 / KSf;
    if (!Number.isInteger(KSf) || KSf < 1 || !Number.isInteger(slice) || slice < 1) return null;
    const fcOut = { [fc1.name]: 0, [fcs[1].name]: P * fc1.outN };
    const red = Math.max(2 * WG + 64, P * (fc1.outN + fcs[1].outN)), redi = Math.max(WG * TCF, ...tiles.map((t) => t.partials));
    return { P, tiles, fc1: { NB, KSf, slice }, fcOut, red, redi, bytes: P * (XS + YS) * 16 + 4 * (red + redi) };
  };
  let chosen = null;
  if (patches) { chosen = tryPatches(patches); if (!chosen) throw new Error(`arch ${plan.name}: ${patches} patches a workgroup does not tile`); }
  else for (const P of [4, 2, 1]) { const c = tryPatches(P); if (c && c.bytes <= budget) { chosen = c; break; } }
  if (!chosen) throw new Error(`arch ${plan.name}: no patch count fits ${budget} bytes`);
  return { ...plan, convs, fcs, f16: false, int8: true, pad: 0, layout, XS, YS, ...chosen, budget, subgroups, unroll: 1 };
}

// ai: A q8 classifier file (its document, cnn/netfile.mjs) packed for the int8 form: readQuantC refuses a float file,
// ai: an unknown scheme or version and a code out of range; offsets carry the layout's key and the input's
// ai: quantisation ({ bits, qmax, inv }), which the shader takes as constants.
export function packWeightsInt8(doc) {
  const q = readQuantC(doc), arch = doc.arch, plan = planArch(arch), layout = weightLayoutQ8(plan);
  q.layers.forEach((Q, i) => {
    const L = plan.layers[i];
    if (Q.name !== L.name || Q.cout !== (L.outC ?? L.outN) || Q.k !== (L.type === "conv" ? L.inC * 9 : L.inN)) throw new Error(`weights: ${Q.name} of the q8 file is not ${L.name} of ${plan.name}`);
  });
  const u = new Uint32Array(layout.vecs * 4), f = new Float32Array(u.buffer), s = new Int32Array(u.buffer);
  for (const Q of q.layers) {
    const o = layout.layers[Q.name], K = Q.k, w = (n, j) => Q.wq[n * K + j];
    for (let n = 0; n < Q.cout; n++) {
      if (Q.type === "conv" && Q.cin === 1) for (let ky = 0; ky < 3; ky++) u[(o.w + ky * o.kxVecs) * 4 + n] = packI8x4(w(n, 3 * ky), w(n, 3 * ky + 1), w(n, 3 * ky + 2), 0);
      else if (Q.type === "conv") {
        for (let g = 0; g < Q.cin / 4; g++) for (let t = 0; t < 9; t++) u[(o.w + (g * 9 + t) * o.kxVecs) * 4 + n] = packI8x4(...[0, 1, 2, 3].map((j) => w(n, (4 * g + j) * 9 + t)));
      } else {
        const S = Q.side, G4 = S / 4;
        for (let kv = 0; kv < o.rowVecs; kv++) {
          const xg = kv % G4, gy = (kv - xg) / G4, y = gy % S, g = (gy - y) / S;
          for (let e = 0; e < 4; e++) u[(o.w + n * o.rowVecs + kv) * 4 + e] = packI8x4(...[0, 1, 2, 3].map((j) => w(n, ((4 * g + j) * S + y) * S + 4 * xg + e)));
        }
      }
      s[o.b * 4 + n] = Q.bq[n];
      f[o.m * 4 + n] = Q.mul[n];
    }
  }
  const fc2 = plan.layers[plan.layers.length - 1], o2 = layout.layers[fc2.name], { cin, w: w2, b: b2 } = q.fc2;
  for (let n = 0; n < 5; n++) {
    for (let k = 0; k < cin; k++) f[(o2.w + n * o2.rowVecs) * 4 + k] = w2[n * cin + k];
    f[o2.b * 4 + n] = b2[n];
  }
  return { data: u, offsets: { gemm: true, int8: true, key: layout.key, layers: layout.layers, quant: { bits: q.bits, qmax: q.qmax, inv: q.inv } }, arch, val: doc.val ?? null, total: plan.total, macs: plan.macs };
}

// ai: What a dump build writes a patch, in u32 words at (f * cap + slot) * words: Xq (four samples a word, 256 words),
// ai: each conv's codes ([channel group][y][x] words of four channels, the first in the low byte), fc1's output (f32
// ai: bits), the five outputs z (f32 bits), then the patch's mean and deviation as the kernel computed them (f32 bits).
export function dumpLayoutQ8(arch) {
  const plan = planArch(arch), fc1 = plan.layers.find((l) => l.name === "fc1");
  let at = (PATCH * PATCH) / 4;
  const codes = plan.layers.filter((l) => l.type === "conv").map((l) => { const c = { at, ch: l.outC, side: l.outS }; at += (l.outC * l.outS * l.outS) / 4; return c; });
  const a = at, z = a + fc1.outN;
  return { xq: 0, codes, a, z, stats: z + 5, words: z + 7 };
}

// ai: One patch's dump (its dumpLayoutQ8(arch).words u32s) against the contract on the kernel's own Xq, a layer at a
// ai: time in the order the kernel makes them; q is readQuantC of the file, px the patch's 1,024 u8 samples. Layers:
// ai:   stats  the kernel's mean and deviation against f64 statistics, in x's units: bad over `stats`
// ai:   xq     against x * inv quantised in f32 on those statistics: a +-1 within `boundary` of a half is a rounding
// ai:          flip (WGSL's f32 division is 2.5 ULP; counted in flips, not bad), any other difference bad
// ai:   convN  every code exactly q8c.mjs forwardCodesC on the dumped Xq
// ai:   fc1    within `tol` of that forward
// ai:   z      within `tol` of it over the sum of its terms' magnitudes (fc2 is an f32 sum; |z| reaches 388)
// ai: Returns { layers: [{ name, n, bad, max, first }], flips, far (a flip's distance from a half, the largest), fw
// ai: (the forward) }; first is the first bad value, { i, got, want } plus c, y and x on a plane.
export function checkDumpQ8(q, arch, words, px, { boundary = 1e-4, stats = 5e-3, tol = 1e-5 } = {}) {
  const D = dumpLayoutQ8(arch), N = PATCH * PATCH, f32 = (at, n) => new Float32Array(Uint32Array.from(words.subarray(at, at + n)).buffer);
  const layer = (name, n) => ({ name, n, bad: 0, max: 0, first: null });
  const miss = (L, i, got, want, where = {}) => { L.bad++; L.first ??= { i, ...where, got, want }; };
  const st = layer("stats", 2), [mu, sd] = f32(D.stats, 2);
  let m0 = 0, v0 = 0;
  for (const v of px) m0 += v / 255;
  m0 /= N;
  for (const v of px) v0 += (v / 255 - m0) ** 2;
  const sd0 = Math.sqrt(v0 / N);
  [[mu, m0], [sd, sd0]].forEach(([got, want], i) => { const e = Math.abs(got - want) / (sd0 + 0.02); st.max = Math.max(st.max, e); if (e > stats) miss(st, i, got, want); });
  // ai: Xq: four samples a word, the first in the low byte, so the words' bytes are the samples in order
  const xq = new Int8Array(Uint32Array.from(words.subarray(D.xq, D.xq + N / 4)).buffer), den = Math.fround(sd + Math.fround(0.02));
  const xk = Float32Array.from(px, (v) => Math.fround(Math.fround(Math.fround(v / 255) - mu) / den)), xqk = quantiseInput(q, xk), lx = layer("xq", N);
  let flips = 0, far = 0;
  for (let s = 0; s < N; s++) {
    if (xq[s] === xqk[s]) continue;
    const t = Math.fround(xk[s] * q.inv), d = Math.abs(t - Math.floor(t) - 0.5);
    flips++; far = Math.max(far, d);
    if (Math.abs(xq[s] - xqk[s]) !== 1 || d >= boundary) miss(lx, s, xq[s], xqk[s], { c: 0, y: s >> 5, x: s & 31 });
  }
  const fw = forwardCodesC(q, xq), convs = planArch(arch).layers.filter((l) => l.type === "conv");
  const planes = fw.codes.map((c, j) => {
    const { at, ch, side: S } = D.codes[j], L = layer(convs[j].name, ch * S * S);
    for (let cg = 0; cg < ch / 4; cg++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const word = words[at + (cg * S + y) * S + x];
      for (let e = 0; e < 4; e++) {
        const got = ((word >>> (8 * e)) << 24) >> 24, i = ((4 * cg + e) * S + y) * S + x;
        if (got !== c[i]) miss(L, i, got, c[i], { c: 4 * cg + e, y, x });
      }
    }
    return L;
  });
  const fa = f32(D.a, fw.a.length), la = layer("fc1", fa.length);
  fa.forEach((v, o) => { const e = Math.abs(v - fw.a[o]); la.max = Math.max(la.max, e); if (e > tol) miss(la, o, v, fw.a[o]); });
  const z = f32(D.z, 5), lz = layer("z", 5);
  z.forEach((v, o) => {
    let S = Math.abs(q.fc2.b[o]);
    for (let c = 0; c < q.fc2.cin; c++) S += Math.abs(q.fc2.w[o * q.fc2.cin + c] * fw.a[c]);
    const e = Math.abs(v - fw.out[o]) / Math.max(1, S);
    lz.max = Math.max(lz.max, e);
    if (e > tol) miss(lz, o, v, fw.out[o]);
  });
  return { layers: [st, lx, ...planes, la, lz], flips, far, fw };
}

// ai: The int8 form's WGSL helpers: a weight word's dot4 with four positions' words (d4l without the first position,
// ai: d4r without the last: see convGemmQ8's edge); a channel's four positions requantised; and four channels at four
// ai: positions packed as four words, word e position e's channels.
const Q8_FNS = /* wgsl */ `
fn d4(v: vec4<u32>, w: u32) -> vec4<i32> { return vec4<i32>(dot4I8Packed(v.x, w), dot4I8Packed(v.y, w), dot4I8Packed(v.z, w), dot4I8Packed(v.w, w)); }
fn d4l(v: vec4<u32>, w: u32) -> vec4<i32> { return vec4<i32>(0, dot4I8Packed(v.y, w), dot4I8Packed(v.z, w), dot4I8Packed(v.w, w)); }
fn d4r(v: vec4<u32>, w: u32) -> vec4<i32> { return vec4<i32>(dot4I8Packed(v.x, w), dot4I8Packed(v.y, w), dot4I8Packed(v.z, w), 0); }
fn rq(a: vec4<i32>, m: f32) -> vec4<i32> { return vec4<i32>(clamp(round(vec4f(a) * m), vec4f(0.0), vec4f(QMAX))); }
fn packq(c0: vec4<i32>, c1: vec4<i32>, c2: vec4<i32>, c3: vec4<i32>) -> vec4<u32> {
  return vec4<u32>(pack4xI8(vec4<i32>(c0.x, c1.x, c2.x, c3.x)), pack4xI8(vec4<i32>(c0.y, c1.y, c2.y, c3.y)), pack4xI8(vec4<i32>(c0.z, c1.z, c2.z, c3.z)), pack4xI8(vec4<i32>(c0.w, c1.w, c2.w, c3.w)));
}
`;

// ai: One int8 conv, as convGemm over words: src/dst buffers with per-patch strides SS/DS (vec4<u32>s), pin/pout the
// ai: row pitches. A row outside the input is read at a clamped row and masked to code 0, as the float form
// ai: multiplies by 0; x outside it is a word 0.
function convGemmQ8(L, tile, src, SS, dst, DS, pin, pout, lw, subgroups) {
  const { TC, TX, KS, nX, NB } = tile, VB = TC / 4, X = "xyzw", first = L.inC === 1, rowG = L.inS / 4;
  const accs = [];
  for (let c = 0; c < TC; c++) for (let t = 0; t < TX; t++) accs.push(`a${c}_${t}`);
  const bias = [];
  for (let vb = 0; vb < VB; vb++) {
    bias.push(`let bv${vb} = bitcast<vec4<i32>>(W[${lw.b}u + wvec + ${vb}u]);`);
    for (let c = 0; c < 4; c++) for (let t = 0; t < TX; t++) bias.push(`a${4 * vb + c}_${t} = vec4<i32>(bv${vb}.${X[c]});`);
  }
  // ai: every channel of the lane's block against each position's tap words; dot(t) names the helper for position t
  const dots = (tap, wv, dot = () => "d4") => {
    const out = [];
    for (let vb = 0; vb < VB; vb++) for (let c = 0; c < 4; c++) for (let t = 0; t < TX; t++) out.push(`a${4 * vb + c}_${t} += ${dot(t)}(${tap(t)}, ${wv(vb)}.${X[c]});`);
    return out.join("\n        ");
  };
  let rows;
  if (first) {
    // ai: conv1: outputs 4 (x4 + t) + e read patch bytes 8 (x4 + t) + 2e - 1 .. + 1, from the words A_t (bytes
    // ai: 8 (x4 + t) .. + 3) and B_t (+ 4 .. + 7) and the top byte of the word before (L0, or B_(t-1)).
    const loads = TX === 1 ? `let v = ${src}[r + x4 / 2u] & vec4<u32>(okm);
        let hi = (x4 & 1u) == 1u;
        let A0 = select(v.x, v.z, hi);
        let B0 = select(v.y, v.w, hi);
        let L0 = select(select(0u, ${src}[r + max(x4 / 2u, 1u) - 1u].w & okm, x4 > 0u), v.y, hi);`
      : `let v = ${src}[r + x4 / 2u] & vec4<u32>(okm);
        let A0 = v.x;
        let B0 = v.y;
        let A1 = v.z;
        let B1 = v.w;
        let L0 = select(0u, ${src}[r + max(x4 / 2u, 1u) - 1u].w & okm, x4 > 0u);`;
    const taps = Array.from({ length: TX }, (_, t) => `let t${t} = vec4<u32>((${t ? `B${t - 1}` : "L0"} >> 24u) | (A${t} << 8u), A${t} >> 8u, (A${t} >> 24u) | (B${t} << 8u), B${t} >> 8u);`).join("\n        ");
    rows = [0, 1, 2].map((ky) => /* wgsl */ `
      {
        let iy = i32(y) * 2 - 1 + ${ky};
        let okm = select(0u, 0xffffffffu, iy >= 0 && iy < ${L.inS});
        let r = src0 + u32(clamp(iy, 0, ${L.inS - 1})) * ${pin}u;
        ${loads}
        ${taps}
        ${Array.from({ length: VB }, (_, vb) => `let w${vb} = W[${lw.w + ky * lw.kxVecs}u + wvec + ${vb}u];`).join(" ")}
        ${dots((t) => `t${t}`, (vb) => `w${vb}`)}
      }`).join("");
  } else {
    // ai: edge: one tile spans the row (nX 1), so x4 is 0 in every lane and the pad word left of x 0 (at stride 1 also
    // ai: the one right of the row's end) is zero for every lane. Its products are left out (d4l, d4r) instead of taken
    // ai: with a word the driver can fold to a constant 0: on the S26 Ultra's Adreno 8xx, dot4I8Packed of such a word
    // ai: returned garbage (2026-09-26: every last conv's x = 0 wrong, 0 blocks).
    const loads = [], ins = [[], [], []], edge = nX === 1;
    if (L.stride === 1) {
      for (let t = 0; t < TX; t++) loads.push(`let m${t} = ${src}[r + x4 + ${t}u] & vec4<u32>(okm);`);
      if (!edge) loads.push(`let lf = select(0u, ${src}[r + max(x4, 1u) - 1u].w & okm, x4 > 0u);`);
      if (!edge) loads.push(`let rt = select(0u, ${src}[r + min(x4 + ${TX}u, ${rowG - 1}u)].x & okm, x4 + ${TX}u < ${rowG}u);`);
      for (let t = 0; t < TX; t++) {
        ins[0].push(`vec4<u32>(${t ? `m${t - 1}.w` : edge ? "0u" : "lf"}, m${t}.xyz)`); ins[1].push(`m${t}`); ins[2].push(`vec4<u32>(m${t}.yzw, ${t < TX - 1 ? `m${t + 1}.x` : edge ? "0u" : "rt"})`);
      }
    } else {
      for (let t = 0; t < TX; t++) loads.push(`let m${t}a = ${src}[r + 2u * (x4 + ${t}u)] & vec4<u32>(okm); let m${t}b = ${src}[r + 2u * (x4 + ${t}u) + 1u] & vec4<u32>(okm);`);
      if (!edge) loads.push(`let lf = select(0u, ${src}[r + 2u * max(x4, 1u) - 1u].w & okm, x4 > 0u);`);
      for (let t = 0; t < TX; t++) {
        ins[0].push(`vec4<u32>(${t ? `m${t - 1}b.w` : edge ? "0u" : "lf"}, m${t}a.y, m${t}a.w, m${t}b.y)`); ins[1].push(`vec4<u32>(m${t}a.x, m${t}a.z, m${t}b.x, m${t}b.z)`); ins[2].push(`vec4<u32>(m${t}a.y, m${t}a.w, m${t}b.y, m${t}b.w)`);
      }
    }
    const dot = (kx) => (t) => (edge && kx === 0 && t === 0 ? "d4l" : edge && L.stride === 1 && kx === 2 && t === TX - 1 ? "d4r" : "d4");
    const taps = ins.map((iv, kx) => iv.map((e, t) => `let i${kx}_${t} = ${e};`).join(" ")).join("\n        ");
    const wts = [0, 1, 2].map((kx) => Array.from({ length: VB }, (_, vb) => `let w${kx}_${vb} = W[wb + ${kx * lw.kxVecs + vb}u];`).join(" ")).join("\n        ");
    const gPer = L.inC / 4 / KS;
    rows = /* wgsl */ `
    for (var g = ks * ${gPer}u; g < (ks + 1u) * ${gPer}u; g++) {${[0, 1, 2].map((ky) => /* wgsl */ `
      {
        let iy = i32(y) * ${L.stride} - 1 + ${ky};
        let okm = select(0u, 0xffffffffu, iy >= 0 && iy < ${L.inS});
        let r = src0 + (g * ${L.inS}u + u32(clamp(iy, 0, ${L.inS - 1}))) * ${pin}u;
        ${loads.join(" ")}
        ${taps}
        let wb = ${lw.w}u + ((g * 3u + ${ky}u) * 3u) * ${lw.kxVecs}u + wvec;
        ${wts}
        ${[0, 1, 2].map((kx) => dots((t) => `i${kx}_${t}`, (vb) => `w${kx}_${vb}`, dot(kx))).join("\n        ")}
      }`).join("")}
    }`;
  }
  const finish = [];
  for (let vb = 0; vb < VB; vb++) {
    finish.push(`let mv${vb} = bitcast<vec4f>(W[${lw.m}u + wvec + ${vb}u]);`);
    for (let t = 0; t < TX; t++) finish.push(`${dst}[dst0 + ((wvec + ${vb}u) * ${L.outS}u + y) * ${pout}u + x4 + ${t}u] = packq(${[0, 1, 2, 3].map((c) => `rq(a${4 * vb + c}_${t}, mv${vb}.${X[c]})`).join(", ")});`);
  }
  const fin = finish.join("\n      ");
  const per = 4 * TC * TX, tilesN = WG / KS;
  const partAt = (s, k) => `redi[((${s} - 1u) * ${tilesN}u + ti) * ${per}u + ${k}u]`;
  const reduce = KS === 1 ? fin : subgroups ? /* wgsl */ `
    ${[1, 2].filter((m) => m < KS).map((m) => accs.map((a) => `${a} += subgroupShuffleXor(${a}, ${m}u);`).join(" ")).join("\n    ")}
    if (ks == 0u) {
      ${fin}
    }` : /* wgsl */ `
    let ti = li / ${KS}u;
    if (ks > 0u) {
      ${accs.map((a, k) => [0, 1, 2, 3].map((e) => `${partAt("ks", 4 * k + e)} = ${a}[${e}];`).join(" ")).join("\n      ")}
    }
    workgroupBarrier();
    if (ks == 0u) {
      ${Array.from({ length: KS - 1 }, (_, s) => accs.map((a, k) => `${a} += vec4<i32>(${[0, 1, 2, 3].map((e) => partAt(`${s + 1}u`, 4 * k + e)).join(", ")});`).join(" ")).join("\n      ")}
      ${fin}
    }`;
  return /* wgsl */ `
  // ai: ${L.name} int8: ${L.inC} x ${L.inS}^2 -> ${L.outC} x ${L.outS}^2 stride ${L.stride}; tile ${TC} ch x ${4 * TX} px, ${KS} slice${KS > 1 ? "s" : ""}
  {
    let ks = li % ${KS}u;
    let r0 = li / ${KS}u;
    let xg = r0 % ${nX}u;
    let r1 = r0 / ${nX}u;
    let y = r1 % ${L.outS}u;
    let r2 = r1 / ${L.outS}u;
    let cb = r2 % ${NB}u;
    let p = r2 / ${NB}u;
    let wvec = cb * ${VB}u;
    let x4 = xg * ${TX}u;
    let src0 = p * ${SS}u;
    let dst0 = p * ${DS}u;
    ${accs.map((a) => `var ${a} = vec4<i32>(0);`).join(" ")}
    if (ks == 0u) { ${bias.join(" ")} }
    ${rows}
    ${reduce}
  }
  workgroupBarrier();
`;
}

// ai: fc1 int8 over the last conv's words: lane (patch, TCF outputs, a slice of the input vecs), the slices summed in
// ai: i32 through redi; relu(f32(acc) * s) into red for fc2.
function fc1GemmQ8(L, plan, src, SS, pitch, lw, S) {
  const { NB, KSf, slice } = plan.fc1, out = plan.fcOut[L.name], G4 = S / 4;
  const at = pitch === G4 ? "kv" : `(kv / ${G4}u) * ${pitch}u + kv % ${G4}u`;
  return /* wgsl */ `
  // ai: ${L.name} int8: ${L.inN} codes -> ${L.outN} f32; lane = (patch, ${TCF} outputs, ${slice} vecs of 16 codes), ${KSf} slices
  {
    let ks = li % ${KSf}u;
    let r0 = li / ${KSf}u;
    let ob = r0 % ${NB}u;
    let p = r0 / ${NB}u;
    let src0 = p * ${SS}u;
    let o0 = ob * ${TCF}u;
    ${Array.from({ length: TCF }, (_, c) => `var f${c} = 0i;`).join(" ")}
    for (var v = 0u; v < ${slice}u; v++) {
      let kv = ks * ${slice}u + v;
      let x = ${src}[src0 + ${at}];
      ${Array.from({ length: TCF }, (_, c) => `let w${c} = W[${lw.w}u + (o0 + ${c}u) * ${lw.rowVecs}u + kv];`).join("\n      ")}
      ${Array.from({ length: TCF }, (_, c) => `f${c} += dot4I8Packed(x.x, w${c}.x) + dot4I8Packed(x.y, w${c}.y) + dot4I8Packed(x.z, w${c}.z) + dot4I8Packed(x.w, w${c}.w);`).join("\n      ")}
    }
    ${Array.from({ length: TCF }, (_, c) => `redi[li * ${TCF}u + ${c}u] = f${c};`).join(" ")}
  }
  workgroupBarrier();
  if (li < ${plan.P * L.outN}u) {
    let p = li / ${L.outN}u;
    let o = li % ${L.outN}u;
    var acc = bitcast<i32>(W[${lw.b}u + o / 4u][o % 4u]);
    for (var s = 0u; s < ${KSf}u; s++) { acc += redi[((p * ${NB}u + o / ${TCF}u) * ${KSf}u + s) * ${TCF}u + o % ${TCF}u]; }
    red[${out}u + li] = max(f32(acc) * bitcast<f32>(W[${lw.m}u + o / 4u][o % 4u]), 0.0);
  }
  workgroupBarrier();
`;
}

// ai: A WGSL f32 literal that gives back exactly v (an f32 the contract multiplies by).
function exactF32(v) {
  let s = (+v).toPrecision(9);
  if (!/[.e]/.test(s)) s += ".0";
  if (Math.fround(v) !== v || Math.fround(+s) !== v) throw new Error(`classify_gemm int8: ${v} is not an f32 the literal ${s} gives back`);
  return s;
}

// ai: The int8 stage's parts for classifySource: the directives, the declarations and helpers, the patch store (Xq
// ai: from the cut's f32 x), and the network (convs, then fcs), with the dump's copies when dump is set.
function partsQ8(plan, offsets, arch, { lw, slot, stop, subgroups, dump, G }) {
  const P = plan.P, q = offsets.quant, aq = arch.quant;
  if (!q || q.bits !== aq.bits || q.inv !== aq.input?.inv) throw new Error(`classify_gemm int8: weights packed with quant ${JSON.stringify(q)}, the arch says bits ${aq.bits} inv ${aq.input?.inv}`);
  const pp = PITCH_PATCH_Q8, D = dumpLayoutQ8(arch);
  // ai: word w of a patch's buffer: its vec4 read whole and the word taken by select, no component index at a runtime
  // ai: value, so the check build does not share the Adreno suspects it is there to find
  const dumpTo = (words, off, vecAt) => /* wgsl */ `
  for (var i = li; i < ${P * words}u; i += ${WG}u) {
    let pd = i / ${words}u;
    let w = i % ${words}u;
    let jd = j0 + pd;
    let dv = ${vecAt};
    let dc = w % 4u;
    if (jd < n) { dbg[(f * P.cap + ${slot("jd")}) * ${D.words}u + ${off}u + w] = select(select(dv.x, dv.y, dc == 1u), select(dv.z, dv.w, dc == 3u), dc >= 2u); }
  }`;
  let convs = dump ? dumpTo(PATCH * 8, D.xq, `bufX[pd * ${plan.XS}u + (w / 8u) * ${pp}u + (w % 8u) / 4u]`) + /* wgsl */ `
  if (lq == 0u && jc < n) {
    dbg[(f * P.cap + ${slot("jc")}) * ${D.words}u + ${D.stats}u] = bitcast<u32>(mu);
    dbg[(f * P.cap + ${slot("jc")}) * ${D.words}u + ${D.stats + 1}u] = bitcast<u32>(sd);
  }` : "";
  plan.convs.forEach((L, k) => {
    if (stop === "cut" || (stop && stop.startsWith("conv") && +stop.slice(4) < k + 1)) return;
    const [src, SS, dst, DS] = k % 2 === 0 ? ["bufX", plan.XS, "bufY", plan.YS] : ["bufY", plan.YS, "bufX", plan.XS];
    const pin = k === 0 ? pp : L.inS / 4, pout = L.outS / 4;
    convs += convGemmQ8(L, plan.tiles[k], src, SS, dst, DS, pin, pout, lw(L.name), subgroups);
    if (dump) convs += dumpTo((L.outC * L.outS * L.outS) / 4, D.codes[k].at, `${dst}[pd * ${DS}u + (w / ${L.outS}u) * ${pout}u + (w % ${L.outS}u) / 4u]`);
  });
  const last = plan.convs[plan.convs.length - 1], lastBuf = plan.convs.length % 2 ? ["bufY", plan.YS] : ["bufX", plan.XS];
  const [fc1, fc2] = plan.fcs, dumpFc = (L, off) => (dump ? /* wgsl */ `
  if (li < ${P * L.outN}u) {
    let jd = j0 + li / ${L.outN}u;
    if (jd < n) { dbg[(f * P.cap + ${slot("jd")}) * ${D.words}u + ${off}u + li % ${L.outN}u] = bitcast<u32>(red[${plan.fcOut[L.name]}u + li]); }
  }` : "");
  let fcs = "";
  if (!(stop === "cut" || stop?.startsWith("conv"))) fcs += fc1GemmQ8(fc1, plan, lastBuf[0], lastBuf[1], last.outS / 4, lw(fc1.name), last.outS) + dumpFc(fc1, D.a);
  if (!stop) fcs += fcPlain(fc2, plan, plan.fcOut[fc1.name], lw(fc2.name), false) + dumpFc(fc2, D.z);
  const store = Array.from({ length: P }, (_, k) => `{ let q = lq + ${G * k}u; bufX[cp * ${plan.XS}u + (q / 8u) * ${pp}u + (q % 8u) / 4u][q % 4u] = pack4xI8(vec4<i32>(clamp(round(((s${k} - vec4f(mu)) / (sd + 0.02)) * INV), vec4f(-QMAX), vec4f(QMAX)))); }`).join("\n  ");
  const decls = [
    `const QMAX = ${q.qmax}.0;`,
    `const INV = ${exactF32(q.inv)};`,
    ...(dump ? ["@group(0) @binding(13) var<storage, read_write> dbg: array<u32>;"] : []),
    `// ai: ${plan.name}, int8 codes of ${q.bits} bits: ${P} patches a workgroup, ${plan.bytes} bytes of workgroup memory (X ${plan.XS}, Y ${plan.YS} vec4<u32>s a patch, ${plan.red} f32 and ${plan.redi} i32 scratch).`,
    `var<workgroup> bufX: array<vec4<u32>, ${P * plan.XS}>;`,
    `var<workgroup> bufY: array<vec4<u32>, ${P * plan.YS}>;`,
    `var<workgroup> red: array<f32, ${plan.red}>;`,
    `var<workgroup> redi: array<i32, ${plan.redi}>;`,
  ].join("\n") + Q8_FNS;
  return { directives: `requires ${INT8};\n` + (subgroups ? "enable subgroups;\n" : ""), decls, store, convs, fcs };
}

// offsets: what packWeights here returned (its `offsets`); the layout key must match the arch and precision.
// role, keep, rest: as classify.mjs. The dispatch is the decoder's (ceil(cap / P) or ceil(keep / P) workgroups a
// frame); a workgroup takes the `P` slots from wg.x * P and returns at once when the first is past the frame's count.
// subgroups: the device was created with the "subgroups" feature (adjacent lanes then reduce a K-split by
// shuffles; measured 11% faster on the iGPU and 17% on the 4090). budget, pad: see classifyPlan.
// patches, partials, unroll, stop: measurement knobs (cnn/test_gemm.mjs). stop ends the network after a phase
// ("cut", "conv1", .., "fc1"): the reading is then garbage, and the time is that phase's share.
// ai: int8: the int8 form, offsets from packWeightsInt8 (the layout key `<arch>:int8`). dump (int8 only, a check
// ai: build): a thirteenth binding, dbg (u32), receives each patch's Xq, codes, fc1 output and z (dumpLayoutQ8).
export const classifySource = (offsets, arch, { f16 = false, int8 = false, role = null, keep = 0, rest = "small", budget = BUDGET, pad = 1, patches = 0, partials = RED_MIN, subgroups = false, unroll = 1, stop = null, dump = false } = {}) => {
  const plan = classifyPlan(arch, { f16, int8, budget, pad, patches, partials, subgroups, unroll }), T = f16 ? "f16" : "f32", P = plan.P, G = WG / P;
  if (!offsets?.gemm || offsets.key !== plan.layout.key) throw new Error(`classify_gemm: weights packed for ${offsets?.key ?? "classify.mjs"}, the shader is ${plan.layout.key}; pack with classify_gemm.mjs packWeights${int8 ? "Int8" : ""}`);
  if (role === "second" && !(keep > 0)) throw new Error(`cascade keep ${keep}`);
  if (dump && !int8) throw new Error("classify_gemm: dump is the int8 form's check");
  const lw = (name) => plan.layout.layers[name];
  const slot = (j) => (role === "second" ? `list[f * ${keep}u + min(${j}, ${keep - 1}u)]` : j);
  const zAt = plan.fcOut[plan.fcs[plan.fcs.length - 1].name];
  const { directives, decls, store, convs, fcs } = int8 ? partsQ8(plan, offsets, arch, { lw, slot, stop, subgroups, dump, G }) : partsFloat(plan, { lw, stop, subgroups, unroll, G, T });
  // The cut, once per pyramid level so the level's dimensions, base and fetch are fixed outside the sample loop
  // (classify.mjs's look() re-derives them and switches on the level inside every one of the four taps, which
  // made the cut a third of cnn-v3's time). The arithmetic is look()'s exactly: the level's scale is a power of
  // two, so the multiply by its inverse is the division it replaces.
  const fetch = (l) => (l === 0 ? (x, y) => `textureLoad(img, vec2i(${x}, ${y}), i32(f), 0).r` : (x, y) => `lv${l}[${y === "iy0" ? "row0" : "row1"} + u32(${x})]`);
  const cutAt = (l) => Array.from({ length: P }, (_, k) => {
    const at = fetch(l);
    return /* wgsl */ `
      {
        let q = lq + ${G * k}u;
        let sy = f32(q / 8u);
        let sx = f32((q % 8u) * 4u);
        let py = (pk.y + u * (sy - 15.5)) * inv - 0.5;
        let fy = floor(py);
        let ay = py - fy;
        let jy = i32(fy);
        let iy0 = clamp(jy, 0, lh - 1);
        let iy1 = clamp(jy + 1, 0, lh - 1);
        ${l ? `let row0 = lbase + u32(iy0) * lsx; let row1 = lbase + u32(iy1) * lsx;` : ""}
        ${[0, 1, 2, 3].map((t) => /* wgsl */ `
        {
          let px = (pk.x + u * (sx + ${t}.0 - 15.5)) * inv - 0.5;
          let fx = floor(px);
          let ax = px - fx;
          let jx = i32(fx);
          let ix0 = clamp(jx, 0, lw - 1);
          let ix1 = clamp(jx + 1, 0, lw - 1);
          s${k}[${t}] = mix(mix(${at("ix0", "iy0")}, ${at("ix1", "iy0")}, ax), mix(${at("ix0", "iy1")}, ${at("ix1", "iy1")}, ax), ay);
        }`).join("")}
        ssum += s${k}.x + s${k}.y + s${k}.z + s${k}.w;
        ssq += dot(s${k}, s${k});
      }`;
  }).join("");
  const cut = /* wgsl */ `
  let F = frames[f];
  let lw = i32((F.w + (1u << l) - 1u) >> l);
  let lh = i32((F.h + (1u << l) - 1u) >> l);
  let ls = LD.d[l];
  let lsx = ls.x;
  let lbase = f * ls.y * lsx;
  let inv = 1.0 / f32(1u << l);
  switch l {
    ${[0, 1, 2, 3].map((l) => `case ${l}u: {${cutAt(l)}
    }`).join("\n    ")}
    default: {${cutAt(4)}
    }
  }`;
  return directives + DIMS + LEVEL_BINDINGS + LEVELS + /* wgsl */ `
struct P4 { cap: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(7) var<storage, read> peaks: array<vec4f>;
@group(0) @binding(8) var<storage, read> counts: array<u32>;
@group(0) @binding(9) var<storage, read_write> readings: array<vec4f>;
@group(0) @binding(10) var<uniform> P: P4;
@group(0) @binding(11) var<storage, read> W: array<vec4<u32>>;
${role === "first" ? "@group(0) @binding(12) var<storage, read_write> score: array<f32>;" : ""}
${role === "second" ? "@group(0) @binding(12) var<storage, read> list: array<u32>;" : ""}
const KAPPA = ${KAPPA};
const PI = 3.14159265;
${decls}
@compute @workgroup_size(${WG}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.y;
  // Uniform over the workgroup (workgroup id and read-only storage), so the whole group returns before any barrier.
  if (frames[f].valid == 0u) { return; }
  let n = min(counts[f * 16u + ${role === "second" ? 3 : 2}u], ${role === "second" ? `${keep}u` : "P.cap"});
  let j0 = wg.x * ${P}u;
  if (j0 >= n) { return; }
  // The cut: ${G} lanes a patch, each ${P} runs of four samples in a row; sample (a, b) at pk + ((a, b) - 15.5) u, luma
  // 0..1 as the texture holds it. Held in registers until the mean and deviation are known.
  let cp = li / ${G}u;
  let lq = li % ${G}u;
  let jc = j0 + cp;
  let ic = select(0u, ${slot("jc")}, jc < n);
  let pk = peaks[f * P.cap + ic];
  let u = pk.z / KAPPA;
  let l = levelFor(u);
  ${Array.from({ length: P }, (_, k) => `var s${k} = vec4f(0.0);`).join(" ")}
  var ssum = 0.0;
  var ssq = 0.0;
  ${cut}
  red[li] = ssum;
  red[${WG}u + li] = ssq;
  workgroupBarrier();
  if (li < 32u) {
    var t1 = 0.0;
    var t2 = 0.0;
    for (var k = 0u; k < 8u; k++) { t1 += red[li * 8u + k]; t2 += red[${WG}u + li * 8u + k]; }
    red[${2 * WG}u + li] = t1;
    red[${2 * WG + 32}u + li] = t2;
  }
  workgroupBarrier();
  var tsum = 0.0;
  var tsq = 0.0;
  for (var k = 0u; k < ${G / 8}u; k++) { tsum += red[${2 * WG}u + cp * ${G / 8}u + k]; tsq += red[${2 * WG + 32}u + cp * ${G / 8}u + k]; }
  let mu = tsum / ${PATCH * PATCH}.0;
  let sd = sqrt(max(tsq / ${PATCH * PATCH}.0 - mu * mu, 0.0));
  ${store}
  workgroupBarrier();
  ${convs}
  ${fcs}
  if (li < ${P}u) {
    let j = j0 + li;
    if (j < n) {
      let i = ${slot("j")};
      let pw = peaks[f * P.cap + i];
      let uw = pw.z / KAPPA;
      // The last fc: (mark logit, cos t, sin t, dlogu, form logit).
      let z0 = red[${zAt}u + li * 5u];
      let z1 = red[${zAt}u + li * 5u + 1u];
      let z2 = red[${zAt}u + li * 5u + 2u];
      let z3 = red[${zAt}u + li * 5u + 3u];
      let z4 = red[${zAt}u + li * 5u + 4u];
      let t = atan2(z2, z1);
      let pr = 1.0 / (1.0 + exp(-z0));
      let o3 = 3u * (f * P.cap + i);
      ${role === "first" ? "score[f * P.cap + i] = z0;" : ""}
      ${role === "first" && rest === "zero" ? "" : /* wgsl */ `readings[o3] = vec4f(pw.xy, uw * exp(z3), t - PI / 4.0);
      readings[o3 + 1u] = vec4f(pr, 0.0, 0.0, 0.0);
      readings[o3 + 2u] = vec4f(select(0.0, 1.0, z4 > 0.0), pw.w, f32(levelFor(uw)), 1.0);`}
    }
  }
}
`;
};
