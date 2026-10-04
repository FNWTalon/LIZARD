// F4 as a network, the reference kernel: the decoder runs classify_gemm.mjs (the same stage, GEMM-shaped), and
// cnn/test_gemm.mjs checks that one against this. RANK and CASCADE live here and are re-exported from there.
// One workgroup a (frame, kept peak) cuts the 32 x 32 patch
// at the peak, standardises it, runs the arch's layers in workgroup memory and writes one reading in the layout
// finder.mjs states at the peak's index. Its best diagonal is index 0, at the network's outward angle, scored by the
// mark probability.
// ai: The layout is the deleted hand-written F4's (DESCRIBE, 2026-09-26), which VOTE and GATHER were built on.
// readings[f][i]: (x, y, u exp(dlogu), t - pi/4), (p, 0, 0, 0), (form, response, level, 1).
//
// The shader is generated from the arch (cnn/weights.mjs planArch): any chain of 3 x 3 convs at stride 1 or 2 from
// 32 x 32 x 1, then fully connected layers. Activations are vec4 along x, so a conv reads a row of taps with three
// loads and does twelve multiply-adds on them; on the iGPU the stage was bound by loads, not arithmetic. With f16
// the weights and the activations are stored as f16 and multiplied in f32 (an f16 x f16 product is exact in f32);
// accumulation is f32 in both.
import { DIMS } from "./common.mjs";
import { KAPPA, PATCH } from "../cnn/patch.mjs";
import { planArch } from "../cnn/weights.mjs";

// finder.mjs keeps these private; the patch has to be cut exactly as finder.mjs's SCORE reads the pyramid, so they are copied.
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

// Workgroup memory for an arch: the patch (bufP), then the convs ping-pong between bufA and bufB. The stats
// reduction and the fc layers need f32 scratch: with f32 activations bufA (free before conv1) and bufP (free after
// it) serve; with f16 a separate f32 array does. Sizes in elements of the activation type.
export function classifyPlan(arch, { f16 = false } = {}) {
  if (arch?.quant) throw new Error(`classify: ${arch.name} is an int8 file (arch.quant); classify_gemm.mjs runs it`);
  const plan = planArch(arch);
  const convs = plan.layers.filter((l) => l.type === "conv"), fcs = plan.layers.filter((l) => l.type === "fc");
  if (!convs.length) throw new Error(`arch ${plan.name}: no conv layer`);
  for (const l of convs) {
    if (l.inS % 4 || l.outS % 4 || (l.stride === 2 && l.inS !== 2 * l.outS) || (l.stride !== 1 && l.stride !== 2)) throw new Error(`arch ${plan.name}: ${l.name} ${l.inS} -> ${l.outS} at stride ${l.stride}; sizes must be multiples of 4`);
  }
  const outOf = (l) => l.outC * l.outS * l.outS;
  const A = Math.max(f16 ? 4 : 512, ...convs.filter((_, k) => k % 2 === 0).map(outOf));
  const B = Math.max(4, ...convs.filter((_, k) => k % 2 === 1).map(outOf));
  const scratch = WG + Math.max(...fcs.map((l) => l.outN));
  if (f16 ? scratch > 512 : scratch > PATCH * PATCH) throw new Error(`arch ${plan.name}: an fc layer with ${scratch - WG} outputs`);
  // fc() gives each output a group of invocations; past WG outputs the rest would never be computed.
  for (const l of plan.layers) if (l.type === "fc" && l.out > WG) throw new Error(`arch ${plan.name}: ${l.name} has ${l.out} outputs, the workgroup has ${WG}`);
  const el = f16 ? 2 : 4, bytes = el * (PATCH * PATCH + A + B) + (f16 ? 4 * 512 : 0);
  return { ...plan, convs, fcs, A, B, f16, bytes };
}

// One conv, padding 1, ReLU, PyTorch's cross-correlation with weight[out][in][ky][kx]. src and dst are [c][y][x] in
// vec4s along x; an invocation makes four outputs in a row, consecutive invocations consecutive x, so a wave shares
// its weight loads. The zero padding is the select at the row's ends.
const conv = (L, src, dst, w, b, T) => {
  const G = (L.outC * L.outS * L.outS) / 4, rowG = L.inS / 4, dRow = L.outS / 4;
  const taps = L.stride === 1
    ? /* wgsl */ `
        let lf = select(0.0, f32(${src}[r + max(x4, 1u) - 1u].w), x4 > 0u);
        let m = vec4f(${src}[r + x4]);
        let rt = select(0.0, f32(${src}[r + min(x4 + 1u, ${rowG - 1}u)].x), x4 + 1u < ${rowG}u);
        acc += w.x * vec4f(lf, m.xyz) + w.y * m + w.z * vec4f(m.yzw, rt);`
    : /* wgsl */ `
        let lf = select(0.0, f32(${src}[r + 2u * max(x4, 1u) - 1u].w), x4 > 0u);
        let m0 = vec4f(${src}[r + 2u * x4]);
        let m1 = vec4f(${src}[r + 2u * x4 + 1u]);
        acc += w.x * vec4f(lf, m0.y, m0.w, m1.y) + w.y * vec4f(m0.x, m0.z, m1.x, m1.z) + w.z * vec4f(m0.y, m0.w, m1.y, m1.w);`;
  return /* wgsl */ `
  for (var g = li; g < ${G}u; g += ${WG}u) {
    let c = g / ${dRow * L.outS}u;
    let y = i32((g / ${dRow}u) % ${L.outS}u);
    let x4 = g % ${dRow}u;
    var acc = vec4f(f32(W[${b}u + c]));
    for (var ci = 0u; ci < ${L.inC}u; ci++) {
      for (var ky = 0; ky < 3; ky++) {
        let iy = ${L.stride} * y - 1 + ky;
        if (iy < 0 || iy >= ${L.inS}) { continue; }
        let r = (ci * ${L.inS}u + u32(iy)) * ${rowG}u;
        let wi = ${w}u + ((c * ${L.inC}u + ci) * 3u + u32(ky)) * 3u;
        let w = vec3f(f32(W[wi]), f32(W[wi + 1u]), f32(W[wi + 2u]));${taps}
      }
    }
    ${dst}[g] = vec4<${T}>(max(acc, vec4f(0.0)));
  }
  workgroupBarrier();
`;
};

// One fc layer: g invocations an output, each a contiguous run of the inputs, partials summed by the first outN.
// Inputs come from the last conv's buffer (vec4s, the PyTorch flattening) or the previous fc's outputs in the f32
// scratch; outputs go to the scratch past the partials.
const fc = (L, src, w, b, relu, at) => {
  let g = 1;
  for (let k = 2; k <= WG / L.outN && L.inN % (src ? 4 * k : k) === 0 && L.inN / k >= 8; k *= 2) g = k;
  const chunk = L.inN / g;
  const dotK = src
    ? /* wgsl */ `let a = vec4f(${src}[k >> 2u]); acc += f32(W[wi + k]) * a.x + f32(W[wi + k + 1u]) * a.y + f32(W[wi + k + 2u]) * a.z + f32(W[wi + k + 3u]) * a.w;`
    : /* wgsl */ `acc += f32(W[wi + k]) * ${at(`${WG}u + k`)};`;
  return /* wgsl */ `
  {
    var acc = 0.0;
    if (li < ${g * L.outN}u) {
      let wi = ${w}u + (li / ${g}u) * ${L.inN}u;
      let k0 = (li % ${g}u) * ${chunk}u;
      for (var k = k0; k < k0 + ${chunk}u; k += ${src ? 4 : 1}u) { ${dotK} }
    }
    ${at("li")} = acc;
  }
  workgroupBarrier();
  if (li < ${L.outN}u) {
    var acc = f32(W[${b}u + li]);
    for (var k = 0u; k < ${g}u; k++) { acc += ${at(`li * ${g}u + k`)}; }
    ${at(`${WG}u + li`)} = ${relu ? "max(acc, 0.0)" : "acc"};
  }
  workgroupBarrier();
`;
};

// The cascade (cnn/README.md, "The cascade"): a small net on every kept peak, RANK keeps the `keep` strongest a
// frame by mark logit as a compacted index list, and the large net runs on that list only, overwriting those
// readings. The harness default; FrontHalf.create's `cascade` option and CASCADE=<file> CASCADE_KEEP=<n> select it.
// rest: what a peak outside the list keeps, "small" (the small net's reading) or "zero" (no reading at all).
// keep 64, not 128 (2026-09-24): the replay's blocks the same to 0.01%, and the small net ranks 99.83% of the noisy
// scenes' marks within its top 64 (scripts/train/cnn_v3_topk.py --k=64; 99.60% at 32, which is why not 32).
// ai: weights: a path under liblizard/ since 2026-10-01 (a name under gpu/cnn/ before): the installed app names every file
// ai: by a hash of its bytes and rewrites only whole literal paths (lizard-web/pwa/hash.mjs), and the harness page takes either.
export const CASCADE = { weights: "gpu/cnn/weights-v3.safetensors", keep: 64, rest: "small" };
// ai: The installed classifiers' int8 twins, paths under liblizard/ (names under gpu/cnn/ until 2026-10-01, which the built
// ai: app's renamed files no longer matched: it would have run the float classifiers without a word): a float file's
// ai: q8 form (cnn/q8c.mjs), run where the decoder runs at precision int8 and its measurement on the planning frames
// ai: says int8 is faster there (gpu/decoder.mjs create's twins, measureTwins). Callers pass the float files; the
// ai: decoder finds the twin by the file the document was loaded from (cnn/netfile.mjs loadNet's source). Each twin's
// ai: CQ81 reference is build/cnn/reference-<X>.bin for weights-<X>.safetensors, as the float files' (cnn/check.mjs).
export const INT8_TWINS = { "gpu/cnn/weights.safetensors": "gpu/cnn/weights-q8.safetensors", "gpu/cnn/weights-v3.safetensors": "gpu/cnn/weights-v3-q8.safetensors" };

// offsets: element offsets of each layer's weights and biases in the W buffer (cnn/weights.mjs, packWeights).
// role: null runs this net on every kept peak. "first" also writes the mark logit to `score` (binding 12) and, with
// rest "zero", writes no reading. "second" takes its peak from `list` (binding 12), `keep` entries a frame, and
// runs on counts[3] of them.
export const classifySource = (offsets, arch, { f16 = false, role = null, keep = 0, rest = "small" } = {}) => {
  // ai: the reference kernel is float only: a q8 file runs in classify_gemm.mjs's int8 form
  if (arch?.quant) throw new Error(`classify: ${arch.name} is an int8 file (arch.quant); classify_gemm.mjs runs it`);
  const P = classifyPlan(arch, { f16 }), T = f16 ? "f16" : "f32";
  if (role === "second" && !(keep > 0)) throw new Error(`cascade keep ${keep}`);
  // Scalar f32 scratch: the stats reduction (512) before conv1, the fc partials and outputs after the convs.
  const statAt = f16 ? (i) => `red[${i}]` : (i) => `bufA[(${i}) >> 2u][(${i}) & 3u]`;
  const fcAt = f16 ? (i) => `red[${i}]` : (i) => `bufP[(${i}) >> 2u][(${i}) & 3u]`;
  const convs = P.convs.map((L, k) => conv(L, k === 0 ? "bufP" : k % 2 ? "bufA" : "bufB", k % 2 ? "bufB" : "bufA", offsets[`${L.name}w`], offsets[`${L.name}b`], T)).join("");
  const lastConv = P.convs.length % 2 ? "bufA" : "bufB";
  const fcs = P.fcs.map((L, k) => fc(L, k === 0 ? lastConv : null, offsets[`${L.name}w`], offsets[`${L.name}b`], k < P.fcs.length - 1, fcAt)).join("");
  return (f16 ? "enable f16;\n" : "") + DIMS + LEVEL_BINDINGS + LEVELS + /* wgsl */ `
struct P4 { cap: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(7) var<storage, read> peaks: array<vec4f>;
@group(0) @binding(8) var<storage, read> counts: array<u32>;
@group(0) @binding(9) var<storage, read_write> readings: array<vec4f>;
@group(0) @binding(10) var<uniform> P: P4;
@group(0) @binding(11) var<storage, read> W: array<${T}>;
${role === "first" ? "@group(0) @binding(12) var<storage, read_write> score: array<f32>;" : ""}
${role === "second" ? "@group(0) @binding(12) var<storage, read> list: array<u32>;" : ""}
const KAPPA = ${KAPPA};
const PI = 3.14159265;
// ${P.name}, ${T}: ${P.bytes} bytes of workgroup memory.
var<workgroup> bufP: array<vec4<${T}>, ${(PATCH * PATCH) / 4}>;
var<workgroup> bufA: array<vec4<${T}>, ${P.A / 4}>;
var<workgroup> bufB: array<vec4<${T}>, ${P.B / 4}>;
${f16 ? "var<workgroup> red: array<f32, 512>;" : ""}
@compute @workgroup_size(${WG}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.y;
  // Uniform over the workgroup (workgroup id and read-only storage), so the whole group returns before any barrier.
  ${role === "second"
    ? /* wgsl */ `if (frames[f].valid == 0u || wg.x >= counts[f * 16u + 3u]) { return; }
  let i = list[f * ${keep}u + wg.x];`
    : /* wgsl */ `let i = wg.x;
  if (frames[f].valid == 0u || i >= counts[f * 16u + 2u]) { return; }`}
  let pk = peaks[f * P.cap + i];
  let u = pk.z / KAPPA;
  let l = levelFor(u);
  // The patch, four samples in a row an invocation: sample (a, b) at p + ((a, b) - 15.5) u, luma 0..1 as the
  // texture holds it (the dataset's u8 / 255). Held in registers until the mean and deviation are known.
  var s4 = vec4f(0.0);
  let sy = f32(li / ${PATCH / 4}u);
  for (var k = 0u; k < 4u; k++) {
    let sx = f32((li % ${PATCH / 4}u) * 4u + k);
    s4[k] = look(l, f, pk.xy + u * (vec2f(sx, sy) - vec2f(${PATCH / 2 - 0.5})));
  }
  ${statAt("li")} = s4.x + s4.y + s4.z + s4.w;
  ${statAt(`${WG}u + li`)} = dot(s4, s4);
  workgroupBarrier();
  for (var st = ${WG / 2}u; st > 0u; st >>= 1u) {
    if (li < st) { ${statAt("li")} += ${statAt("li + st")}; ${statAt(`${WG}u + li`)} += ${statAt(`${WG}u + li + st`)}; }
    workgroupBarrier();
  }
  let mu = ${statAt("0u")} / ${PATCH * PATCH}.0;
  let sd = sqrt(max(${statAt(`${WG}u`)} / ${PATCH * PATCH}.0 - mu * mu, 0.0));
  bufP[li] = vec4<${T}>((s4 - vec4f(mu)) / (sd + 0.02));
  workgroupBarrier();
  ${convs}
  ${fcs}
  if (li == 0u) {
    // The last fc: (mark logit, cos t, sin t, dlogu, form logit).
    let z0 = ${fcAt(`${WG}u`)};
    let z1 = ${fcAt(`${WG + 1}u`)};
    let z2 = ${fcAt(`${WG + 2}u`)};
    let z3 = ${fcAt(`${WG + 3}u`)};
    let z4 = ${fcAt(`${WG + 4}u`)};
    let t = atan2(z2, z1);
    let p = 1.0 / (1.0 + exp(-z0));
    let o3 = 3u * (f * P.cap + i);
    ${role === "first" ? "score[f * P.cap + i] = z0;" : ""}
    ${role === "first" && rest === "zero" ? "" : /* wgsl */ `readings[o3] = vec4f(pk.xy, u * exp(z3), t - PI / 4.0);
    readings[o3 + 1u] = vec4f(p, 0.0, 0.0, 0.0);
    readings[o3 + 2u] = vec4f(select(0.0, 1.0, z4 > 0.0), pk.w, f32(l), 1.0);`}
  }
}
`;
};

// RANK, the cascade's middle: one workgroup a frame keeps the `keep` peaks with the highest score as a compacted
// index list, in score order, and writes how many to counts[3]. Rank by counting: every peak counts the peaks above
// it (ties by index), so ranks are unique and no atomics or host round trip are needed; n^2 compares a frame from
// workgroup memory. A frame with fewer than `keep` peaks lists them all, and every slot past them is written too,
// so the list never carries a slot from an earlier batch.
// The order is on the score's bits, not the float: a float compare ranks a NaN 0, where it collides with the true
// maximum and the losing write leaves a slot holding whatever was there before. The key folds the sign so larger
// scores give larger keys, and a NaN keys 0, below every number.
export const rankSource = ({ cap, keep }) => DIMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> score: array<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> list: array<u32>;
const CAP = ${cap}u;
const KEEP = ${keep}u;
var<workgroup> z: array<u32, ${cap}>;
fn key(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  if ((b & 0x7fffffffu) > 0x7f800000u) { return 0u; }
  return select(~b, b | 0x80000000u, (b >> 31u) == 0u);
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  if (frames[f].valid == 0u) { return; }
  let n = min(counts[f * 16u + 2u], CAP);
  for (var i = li; i < n; i += 256u) { z[i] = key(score[f * CAP + i]); }
  workgroupBarrier();
  for (var i = li; i < n; i += 256u) {
    let zi = z[i];
    var r = 0u;
    for (var j = 0u; j < n; j++) { r += u32(z[j] > zi || (z[j] == zi && j < i)); }
    if (r < KEEP) { list[f * KEEP + r] = i; }
  }
  // Ranks are a permutation, so the slots under min(n, KEEP) are all written above; the rest are cleared here.
  for (var r = li; r < KEEP; r += 256u) { if (r >= n) { list[f * KEEP + r] = 0u; } }
  if (li == 0u) { counts[f * 16u + 3u] = min(n, KEEP); }
}
`;
