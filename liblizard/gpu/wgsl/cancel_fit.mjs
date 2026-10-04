// ai: The fit of a short frame on its references' bases (gpu/back/DESIGN.md section 13.5): z - zbar(y) regressed on
// ai: f = (b_j - bbar_j(y)) (1, v, u) over the six bases of its two reference slots, k = 18 unknowns (a single
// ai: reference leaves 9..17 zero), u = x / (n - 1) - 0.5, v = y / (n - 1) - 0.5, unweighted normal equations
// ai: (the archived CPU arm's "full" model, archive/straddle-cancel/rig/straddle.c focus_sx_cancel). FIT takes ROWS
// ai: sampled rows a workgroup, every ROW_STEP-th row of the picture (y = 4 j), and samples each at x = floor(t n /
// ai: 256) + (j mod st), thread t of 256, st = ceil(n / 256) (4 at 1024, 2 at 384: 256 samples a row at every size,
// ai: so a small picture keeps every thread busy, and x < n at every size the ladder has, 3 x 2^k included), reading
// ai: the frame's picture as fused pass 1 does (the texture through the map, the Coons patch and the bilinear read:
// ai: wgsl/back_transform.mjs pass1Source) or from the sampler's grid; each row's features go to
// ai: workgroup memory and every thread accumulates half the products of two samples in registers over the
// ai: workgroup's rows (the entries of A, the upper triangle row-major a <= c, then the right-hand side), folded
// ai: once at the end into FITP[f][wg][208]. SOLVE sums a frame's partials,
// ai: Gauss-Jordan with partial pivoting on a workgroup array (thread 0 picks the pivot, every thread a barrier a
// ai: step), a pivot under 1e-6 (1 + |A00|) fails the frame (ok 0, coefficients 0), and writes CANCEL[f].
import { LUM0, MAP } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { PICTURES, GRID_FORMATS, PICTURE, pictureHead } from "./back_transform.mjs";
import { PARAMS } from "./cancel_blur.mjs";
import { CANCEL_WORDS, FR_WORDS } from "./cancel_gate.mjs";

export const K = 18, PAIRS = (K * (K + 1)) / 2, FIT_THREADS = PAIRS + K, FITP_WORDS = 208;
export const samplesARow = (n) => Math.min(n, 256);
// ai: The offsets a thread's column takes from row to row, so the rows between them cover every column.
export const sampleStride = (n) => Math.ceil(n / samplesARow(n));
// ai: The fit samples every ROW_STEP-th row (y = ROW_STEP j), ROWS of them a workgroup, so a frame has
// ai: fitWorkgroups(n) partials in FITP.
export const ROW_STEP = 4, ROWS = 8;
export const fitWorkgroups = (n) => n / (ROW_STEP * ROWS);

// ai: The features of one sample from REF: the three bases of column x of the slot's quad record less the row's
// ai: means (the means are f32 behind the same vec2u view).
const FEATURES = /* wgsl */ `
fn basesAt(slot: u32, x: u32, y: u32) -> vec3f {
  let rec: u32 = P.refSlot * slot + ((x >> 2u) * N + y) * 3u;
  let w0: vec2u = REF[rec];
  let w1: vec2u = REF[rec + 1u];
  let w2: vec2u = REF[rec + 2u];
  let hi: bool = (x & 2u) != 0u;
  let odd: bool = (x & 1u) != 0u;
  let p0: vec2f = unpack2x16float(select(w0.x, w0.y, hi));
  let p1: vec2f = unpack2x16float(select(w1.x, w1.y, hi));
  let p2: vec2f = unpack2x16float(select(w2.x, w2.y, hi));
  return vec3f(select(p0.x, p0.y, odd), select(p1.x, p1.y, odd), select(p2.x, p2.y, odd));
}
fn meansAt(slot: u32, y: u32) -> vec3f {
  let m: u32 = P.refSlot * slot + P.refMeans + 2u * y;
  let a: vec2u = REF[m];
  let b: vec2u = REF[m + 1u];
  return vec3f(bitcast<f32>(a.x), bitcast<f32>(a.y), bitcast<f32>(b.x));
}
`;

// ai: FIT for one picture size and picture variant, dispatched (fitWorkgroups(n), 1, B): workgroup (w, 0, z) takes
// ai: sampled rows ROWS w .. ROWS w + ROWS - 1 of the z-th frame of this size's LISTS2 list, and returns past the
// ai: list's count. A row: thread t samples x = floor(t N / 256) + (j mod ST), the 256 samples are folded for zbar,
// ai: and every thread writes its 19-vector (the 18 features and z - zbar) to F. Then the products: thread t holds in
// ai: registers the sums of GROUP products (generated code, constant indices) for the two samples t mod 128 and
// ai: t mod 128 + 128, threads under 128 products 0 .. 94 and the rest 95 .. 188 (a branch uniform over any wave
// ai: of up to 128 lanes), so a sample's products cost 38 workgroup reads and 190 multiply-adds a thread a row, where
// ai: the accumulator-a-thread shape read 2 workgroup words a multiply-add (3.84 ms a frame on the iGPU). After the
// ai: rows, the accumulators are folded through F in chunks of 16 products (a barrier a step) into FITP.
export const GROUP = 95;
export const PRODUCTS = (() => {
  const out = [];
  for (let a = 0; a < K; a++) for (let c = a; c < K; c++) out.push([a, c]);
  for (let a = 0; a < K; a++) out.push([a, K]);
  return out;
})();
export function fitSource({ n, B, picture = "grid", grid = "f32" }) {
  if (!PICTURES.includes(picture)) throw new Error(`fit picture ${picture}`);
  if (!GRID_FORMATS.includes(grid)) throw new Error(`grid format ${grid}`);
  if (n < 256 || n % (ROW_STEP * ROWS)) throw new Error(`fit: 256 samples a row and ${ROW_STEP * ROWS} rows a workgroup, n ${n}`);
  if (PRODUCTS.length !== FIT_THREADS || 2 * GROUP < PRODUCTS.length) throw new Error("fit products");
  const fused = picture === "fused", g16 = grid === "f16";
  const NS = samplesARow(n), ST = sampleStride(n), K1 = K + 1;
  const source = fused ? /* wgsl */ `
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(7) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(8) var<storage, read> resid: array<vec4f>;
` + PICTURE + MAP + LUM0 + COONS + /* wgsl */ `
fn sampleAt(f: u32, M: array<f32, 16>, m: vec2f, u: f32, v: f32, top: vec2f, bot: vec2f, lft: vec2f, rgt: vec2f, cn: Corners, fw: i32, fh: i32) -> f32 {
  return bilin0(f, toImage(M, m) + coonsMix(u, v, top, bot, lft, rgt, cn), fw, fh);
}
` : /* wgsl */ `
@group(0) @binding(0) var<storage, read> grid: array<${g16 ? "u32" : "f32"}>;
`;
  // ai: The fused head: the frame's edge, map, and its ring's grid and lattice (LISTS2 carries the ring as LISTS does).
  const head = fused ? pictureHead("LISTS2", B) : `  let gbase: u32 = f * P.gridStride;\n`;
  // ai: One sample of the frame at picture (x, y): fused pass 1's arithmetic exactly, the column's and the row's
  // ai: curves taken here a sample, since a thread's column changes with the row.
  const sample = fused
    ? `      let mx: f32 = g0 + f32(x) * gstep;\n      let my: f32 = g0 + f32(y) * gstep;\n      let cu: f32 = coonsT(mx, first, nn);\n      let cv: f32 = coonsT(my, first, nn);\n      let tp: vec2f = curve(f, 0u, mx, first, nn);\n      let bt: vec2f = curve(f, 2u, mx, first, nn);\n      let lf: vec2f = curve(f, 3u, my, first, nn);\n      let rg: vec2f = curve(f, 1u, my, first, nn);\n      z = sampleAt(f, M, vec2f(mx, my), cu, cv, tp, bt, lf, rg, cn, fw, fh);\n`
    : g16
      ? `      let e: u32 = gbase + (x >> 6u) * N * 64u + y * 64u + (x & 63u);\n      let gw: vec2f = unpack2x16float(grid[e >> 1u]);\n      z = select(gw.x, gw.y, (e & 1u) == 1u);\n`
      : `      z = grid[gbase + (x >> 6u) * N * 64u + y * 64u + (x & 63u)];\n`;
  // ai: Group g's products over the thread's two samples: the 19-vector read once a sample, then one multiply-add
  // ai: an accumulator, the indices constants.
  const accumulate = (g) => {
    let out = "";
    for (let k = 0; k < 2; k++) {
      out += `      {\n        let sb: u32 = ((t & 127u) + ${k * 128}u) * ${K1}u;\n`;
      for (let a = 0; a <= K; a++) out += `        let f${a}: f32 = F[sb + ${a}u];\n`;
      for (let i = 0; i < GROUP; i++) {
        const p = g * GROUP + i;
        if (p >= PRODUCTS.length) break;
        const [a, c] = PRODUCTS[p];
        out += `        acc${i} += f${a} * f${c};\n`;
      }
      out += `      }\n`;
    }
    return out;
  };
  // ai: The fold: 16 accumulators a chunk through F (16 x 256 words), 16 threads a product summing 16 each, then 32
  // ai: threads write the two groups' sums; the chunk's product i of group g is FITP entry g GROUP + i.
  let fold = "";
  for (let ch = 0; ch * 16 < GROUP; ch++) {
    fold += `  {\n`;
    for (let j = 0; j < 16; j++) { const i = ch * 16 + j; fold += `    F[${j * 256}u + t] = ${i < GROUP ? `acc${i}` : "0.0"};\n`; }
    fold += `    workgroupBarrier();\n`;
    fold += `    {\n      let c: u32 = t >> 4u;\n      let part: u32 = t & 15u;\n      var s: f32 = 0.0;\n      for (var i: u32 = 0u; i < 16u; i += 1u) { s += F[c * 256u + part * 16u + i]; }\n      red[t] = s;\n    }\n`;
    fold += `    workgroupBarrier();\n`;
    fold += `    if (t < 32u) {\n      let c: u32 = t & 15u;\n      let g: u32 = t >> 4u;\n      var s: f32 = 0.0;\n      for (var i: u32 = 0u; i < 8u; i += 1u) { s += red[c * 16u + g * 8u + i]; }\n      let i: u32 = ${ch * 16}u + c;\n      let p: u32 = g * ${GROUP}u + i;\n      if (i < ${GROUP}u && p < ${PRODUCTS.length}u) { FITP[base + p] = s; }\n    }\n`;
    fold += `    workgroupBarrier();\n  }\n`;
  }
  let accs = "";
  for (let i = 0; i < GROUP; i++) accs += `  var acc${i}: f32 = 0.0;\n`;
  return PARAMS + source + /* wgsl */ `
@group(0) @binding(1) var<storage, read> LISTS2: array<u32>;
@group(0) @binding(2) var<storage, read> REF: array<vec2u>;
@group(0) @binding(3) var<storage, read> PLAN: array<u32>;
@group(0) @binding(4) var<storage, read_write> FITP: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;
const N: u32 = ${n}u;
const NS: u32 = ${NS}u;
const ST: u32 = ${ST}u;
const LS: u32 = ${B + 1}u;
var<workgroup> F: array<f32, ${K1 * 256}>;
var<workgroup> red: array<f32, 256>;
` + FEATURES + /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (wg.z >= LISTS2[P.size * LS]) { return; }
  let f: u32 = LISTS2[P.size * LS + 1u + wg.z];
  let fr: u32 = P.frOff + ${FR_WORDS}u * f;
  let slotA: u32 = PLAN[fr + 2u];
  let slotB: u32 = PLAN[fr + 3u];
  let two: bool = PLAN[fr + 4u] >= 2u;
  let base: u32 = f * P.fitpStride + wg.x * ${FITP_WORDS}u;
` + accs + head + /* wgsl */ `
  for (var i: u32 = 0u; i < ${ROWS}u; i += 1u) {
    let j: u32 = wg.x * ${ROWS}u + i;
    let y: u32 = j * ${ROW_STEP}u;
    let x: u32 = (t * N) / NS + j % ST;
    var z: f32 = 0.0;
    {
${sample}    }
    red[t] = z;
    workgroupBarrier();
    if (t < 8u) {
      var s: f32 = 0.0;
      for (var k: u32 = t; k < 256u; k += 8u) { s += red[k]; }
      red[t] = s;
    }
    workgroupBarrier();
    var zbar: f32 = 0.0;
    for (var k: u32 = 0u; k < 8u; k += 1u) { zbar += red[k]; }
    zbar = zbar / f32(NS);
    {
      let v: f32 = f32(y) / f32(N - 1u) - 0.5;
      let u: f32 = f32(x) / f32(N - 1u) - 0.5;
      let dA: vec3f = basesAt(slotA, x, y) - meansAt(slotA, y);
      var dB: vec3f = vec3f(0.0);
      if (two) { dB = basesAt(slotB, x, y) - meansAt(slotB, y); }
      let o: u32 = t * ${K1}u;
      F[o] = dA.x; F[o + 1u] = dA.x * v; F[o + 2u] = dA.x * u;
      F[o + 3u] = dA.y; F[o + 4u] = dA.y * v; F[o + 5u] = dA.y * u;
      F[o + 6u] = dA.z; F[o + 7u] = dA.z * v; F[o + 8u] = dA.z * u;
      F[o + 9u] = dB.x; F[o + 10u] = dB.x * v; F[o + 11u] = dB.x * u;
      F[o + 12u] = dB.y; F[o + 13u] = dB.y * v; F[o + 14u] = dB.y * u;
      F[o + 15u] = dB.z; F[o + 16u] = dB.z * v; F[o + 17u] = dB.z * u;
      F[o + 18u] = z - zbar;
    }
    workgroupBarrier();
    if (t < 128u) {
${accumulate(0)}    } else {
${accumulate(1)}    }
    workgroupBarrier();
  }
` + fold + /* wgsl */ `}
`;
}

// ai: SOLVE, one pipeline for every size (the size's count of partials in its params), dispatched (B, 1, 1) a size:
// ai: workgroup z takes the z-th frame of the size's LISTS2 list. k = 9 nrefs: a single reference solves the 9 x 9
// ai: corner, the other coefficients stay 0. BCOUNTS2[f][3] takes k (ldpc2 adds its blocks tried after).
export function solveSource({ B }) {
  const K1 = K + 1, CELLS = K * K1;
  return PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> LISTS2: array<u32>;
@group(0) @binding(1) var<storage, read> FITP: array<f32>;
@group(0) @binding(2) var<storage, read> PLAN: array<u32>;
@group(0) @binding(3) var<storage, read_write> CANCEL: array<u32>;
@group(0) @binding(4) var<storage, read_write> BCOUNTS2: array<u32>;
@group(0) @binding(5) var<uniform> P: Params;
const LS: u32 = ${B + 1}u;
const K: u32 = ${K}u;
const K1: u32 = ${K1}u;
const PAIRS: u32 = ${PAIRS}u;
const NT: u32 = ${FIT_THREADS}u;
const CELLS: u32 = ${CELLS}u;
const CW: u32 = ${CANCEL_WORDS}u;
var<workgroup> A: array<f32, ${CELLS}>;
var<workgroup> pivRow: u32;
var<workgroup> failed: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  if (wg.x >= LISTS2[P.size * LS]) { return; }
  let f: u32 = LISTS2[P.size * LS + 1u + wg.x];
  let nrefs: u32 = PLAN[P.frOff + ${FR_WORDS}u * f + 4u];
  if (nrefs == 0u) { return; }
  let k: u32 = 9u * min(nrefs, 2u);
  // ai: A from the frame's partials: the triangle mirrored, the right-hand side in column K.
  if (t < NT) {
    var s: f32 = 0.0;
    let at: u32 = f * P.fitpStride + t;
    for (var w: u32 = 0u; w < P.nwg; w += 1u) { s += FITP[at + w * ${FITP_WORDS}u]; }
    if (t < PAIRS) {
      var pa: u32 = 0u;
      var rem: u32 = t;
      loop {
        let len: u32 = K - pa;
        if (rem < len) { break; }
        rem -= len;
        pa += 1u;
      }
      let pc: u32 = pa + rem;
      A[pa * K1 + pc] = s;
      A[pc * K1 + pa] = s;
    } else {
      A[(t - PAIRS) * K1 + K] = s;
    }
  }
  if (t == 0u) { failed = 0u; }
  workgroupBarrier();
  let tiny: f32 = 1e-6 * (1.0 + abs(A[0]));
  for (var col: u32 = 0u; col < k; col += 1u) {
    if (t == 0u) {
      var p: u32 = col;
      var best: f32 = abs(A[col * K1 + col]);
      for (var r: u32 = col + 1u; r < k; r += 1u) {
        let v: f32 = abs(A[r * K1 + col]);
        if (v > best) { best = v; p = r; }
      }
      pivRow = p;
      if (best < tiny) { failed = 1u; }
    }
    workgroupBarrier();
    let p: u32 = workgroupUniformLoad(&pivRow);
    if (p != col && t < K1) {
      let x: f32 = A[col * K1 + t];
      A[col * K1 + t] = A[p * K1 + t];
      A[p * K1 + t] = x;
    }
    workgroupBarrier();
    // ai: Every entry off the pivot row: read the pivot row and the entry's column factor first, then write.
    let piv: f32 = A[col * K1 + col];
    let r0: u32 = t / K1;
    let b0: u32 = t % K1;
    let e1: u32 = t + 256u;
    let r1: u32 = e1 / K1;
    let b1: u32 = e1 % K1;
    let do0: bool = r0 != col && r0 < k;
    let do1: bool = e1 < CELLS && r1 != col && r1 < k;
    var q0: f32 = 0.0;
    var v0: f32 = 0.0;
    var q1: f32 = 0.0;
    var v1: f32 = 0.0;
    if (do0) { q0 = A[r0 * K1 + col] / piv; v0 = A[col * K1 + b0]; }
    if (do1) { q1 = A[r1 * K1 + col] / piv; v1 = A[col * K1 + b1]; }
    workgroupBarrier();
    if (do0) { A[t] -= q0 * v0; }
    if (do1) { A[e1] -= q1 * v1; }
    workgroupBarrier();
  }
  let fail: u32 = workgroupUniformLoad(&failed);
  if (t < K) {
    var cf: f32 = 0.0;
    if (fail == 0u && t < k) { cf = A[t * K1 + K] / A[t * K1 + t]; }
    CANCEL[f * CW + 4u + t] = bitcast<u32>(cf);
  }
  if (t == 0u) {
    CANCEL[f * CW + 3u] = select(1u, 0u, fail != 0u);
    BCOUNTS2[f * 8u + 3u] = k;
  }
}
`;
}
