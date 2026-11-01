// ai: F4's plan and its weight packers (glsl/classify.mjs, the int8, f16 and f32 forms on this plan): patches a
// ai: workgroup, tiles, buffers, scratch, the weights' layouts (the host packs a classifier file by them).
import { PATCH } from "../cnn/patch.mjs";
import { planArch, toHalf } from "../cnn/weights.mjs";
import { readQuantC } from "../cnn/q8c.mjs";
import { packI8x4 } from "../cnn/q8.mjs";

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

// ai: An f32 literal (WGSL's and GLSL's) that gives back exactly v (an f32 the contract multiplies by); the GLSL twin's
// ai: too (../glsl/classify.mjs).
export function exactF32(v) {
  let s = (+v).toPrecision(9);
  if (!/[.e]/.test(s)) s += ".0";
  if (Math.fround(v) !== v || Math.fround(+s) !== v) throw new Error(`classify_gemm int8: ${v} is not an f32 the literal ${s} gives back`);
  return s;
}

// ai: the plan's own pieces the archived WGSL twin also read (archive/webgpu-decoder/)
export { WG, RED_MIN, TCF, pitchOf, PITCH_PATCH_Q8 };
