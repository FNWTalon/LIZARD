// ai: The bases of a reference (gpu/back/DESIGN.md section 13.5): from a painted picture in PIC (f16, row-major, x
// ai: fastest) the three pictures the fit regresses a short frame on, base0 = blur(p, 0.8), base1 = blur(p, 1.6)
// ai: and base2 = blur((p - 0.5)^2, 0.8), separable Gaussians with the edge sample repeated (the archived CPU arm's
// ai: blur, archive/straddle-cancel/rig/straddle.c), into REF's 24 B records (a quad of columns a row, f16) and each
// ai: base's row mean into REF's means (f32, vec4f a row). BLUR takes a 16 x 16 tile with a halo of 5 (the wider
// ai: kernel's radius) through workgroup memory, horizontal taps then vertical, and leaves each tile row's sums of
// ai: the three bases in BPART; MEANS folds a row's n / 16 partials. A slot is processed when PLAN's SLOTS names a
// ai: KEYTAB entry whose size bit is this dispatch's size (the paint's test too), so a batch mixing sizes blurs each
// ai: painted slot once, by its own pipeline.
import { KEYTAB_WORDS } from "./cancel_gate.mjs";

// ai: The kernels: sigma 0.8 (r = 3, 7 taps) and max(0.5, 2 sigma) = 1.6 (r = 5, 11 taps), normalised, as the
// ai: archived arm computed them; literals in the shader, since sigma is fixed.
export const SIGMA = 0.8;
export function gaussian(sigma) {
  const r = Math.min(Math.ceil(3 * sigma), 15), k = [];
  for (let i = -r; i <= r; i++) k.push(Math.exp(-(i * i) / (2 * sigma * sigma)));
  const s = k.reduce((a, b) => a + b, 0);
  return k.map((v) => v / s);
}
export const K7 = gaussian(SIGMA), K11 = gaussian(Math.max(0.5, 2 * SIGMA));
export const HALO = 5, TILE = 16, HW = TILE + 2 * HALO;

// ai: One uniform for the four stages of the bases and the fit, a size a lane (gpu/back/fit.mjs lane): the size's
// ai: index and n, the fit's samples a row (ns) and their stride (st), then the strides in each buffer's own units:
// ai: picSlot (f16 elements a PIC slot, nmax^2), refSlot (vec2u a REF slot, refStride / 8), refMeans (vec2u from a
// ai: slot's start to its means), bpartSlot (f32 a BPART slot), fitpStride (f32 a frame of FITP), frOff and slotsOff
// ai: (PLAN word offsets of FR and SLOTS), gridStride (grid elements a frame), nwg (fit workgroups a frame,
// ai: cancel_fit.mjs fitWorkgroups).
export const PARAMS_WORDS = 16;
export const PARAMS = /* wgsl */ `
struct Params { size: u32, n: u32, ns: u32, st: u32, picSlot: u32, refSlot: u32, refMeans: u32, bpartSlot: u32, fitpStride: u32, frOff: u32, slotsOff: u32, gridStride: u32, nwg: u32, p13: u32, p14: u32, p15: u32 }
`;
const lit = (v) => { const s = String(v); return s.includes(".") || s.includes("e") ? s : `${s}.0`; };

// ai: Is paint slot r a reference of this dispatch's size: SLOTS[r] names a KEYTAB entry (NONE unused) whose tag's
// ai: size bits are this size alone. PLAN read-only, so the test is uniform and may precede a barrier.
const SLOT_LIVE = /* wgsl */ `
const NONE: u32 = 0xffffffffu;
fn slotLive(r: u32) -> bool {
  let e: u32 = PLAN[P.slotsOff + r];
  if (e == NONE) { return false; }
  return (PLAN[e * ${KEYTAB_WORDS}u + 7u] & 0xffu) == (1u << P.size);
}
`;

// ai: BLUR for one picture size, dispatched (n / 16, n / 16, R): workgroup (tx, ty, r) blurs tile (tx, ty) of slot
// ai: r. 256 threads: the 26 x 26 halo tile comes in clamped to the picture's edge (edge sample repeated, as the
// ai: C clamps its index), the horizontal pass makes 26 rows x 16 columns of the three sums, the vertical pass one
// ai: output a thread; then 64 threads pack the quads into REF records and 16 threads sum their row into BPART.
export function blurSource({ n }) {
  if (n % TILE) throw new Error(`blur tile ${TILE} over n ${n}`);
  // ai: The taps unrolled: j runs over the 11 halo offsets, the 7-tap kernels over the middle seven.
  const taps = (at, p, q) => {
    let out = "";
    for (let j = 0; j < 11; j++) {
      out += `    { let s: f32 = ${at(j)}; a1 += ${lit(K11[j])} * s;`;
      if (j >= 2 && j <= 8) out += ` a0 += ${lit(K7[j - 2])} * ${p}; a2 += ${lit(K7[j - 2])} * ${q};`;
      out += ` }\n`;
    }
    return out;
  };
  return PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PIC: array<u32>;
@group(0) @binding(1) var<storage, read> PLAN: array<u32>;
@group(0) @binding(2) var<storage, read_write> REF: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> BPART: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const N: u32 = ${n}u;
const NI: i32 = ${n};
const HW: u32 = ${HW}u;
const TW: u32 = ${TILE}u;
var<workgroup> tp: array<f32, ${HW * HW}>;
var<workgroup> h0: array<f32, ${HW * TILE}>;
var<workgroup> h1: array<f32, ${HW * TILE}>;
var<workgroup> h2: array<f32, ${HW * TILE}>;
var<workgroup> o0: array<f32, ${TILE * TILE}>;
var<workgroup> o1: array<f32, ${TILE * TILE}>;
var<workgroup> o2: array<f32, ${TILE * TILE}>;
` + SLOT_LIVE + /* wgsl */ `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let r: u32 = wg.z;
  if (!slotLive(r)) { return; }
  let x0: i32 = i32(wg.x * TW) - ${HALO};
  let y0: i32 = i32(wg.y * TW) - ${HALO};
  let pic: u32 = P.picSlot * r;
  for (var i: u32 = t; i < HW * HW; i += 256u) {
    let x: u32 = u32(clamp(x0 + i32(i % HW), 0, NI - 1));
    let y: u32 = u32(clamp(y0 + i32(i / HW), 0, NI - 1));
    let e: u32 = pic + y * N + x;
    let w: vec2f = unpack2x16float(PIC[e >> 1u]);
    tp[i] = select(w.x, w.y, (e & 1u) == 1u);
  }
  workgroupBarrier();
  for (var i: u32 = t; i < HW * TW; i += 256u) {
    let c: u32 = (i / TW) * HW + (i % TW);
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
${taps((j) => `tp[c + ${j}u]`, "s", "(s - 0.5) * (s - 0.5)")}    h0[i] = a0; h1[i] = a1; h2[i] = a2;
  }
  workgroupBarrier();
  let lx: u32 = t & 15u;
  let ly: u32 = t >> 4u;
  {
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    // ai: The vertical taps read the three horizontal sums at the same row offset: s is h1's, the others h0's and h2's.
${[...Array(11)].map((_, j) => `    { let s: f32 = h1[(ly + ${j}u) * TW + lx]; a1 += ${lit(K11[j])} * s;${j >= 2 && j <= 8 ? ` a0 += ${lit(K7[j - 2])} * h0[(ly + ${j}u) * TW + lx]; a2 += ${lit(K7[j - 2])} * h2[(ly + ${j}u) * TW + lx];` : ""} }`).join("\n")}
    o0[t] = a0; o1[t] = a1; o2[t] = a2;
  }
  workgroupBarrier();
  if (t < 64u) {
    let qy: u32 = t >> 2u;
    let o: u32 = qy * TW + (t & 3u) * 4u;
    let x4: u32 = wg.x * 4u + (t & 3u);
    let y: u32 = wg.y * TW + qy;
    let rec: u32 = P.refSlot * r + (x4 * N + y) * 3u;
    REF[rec] = vec2u(pack2x16float(vec2f(o0[o], o0[o + 1u])), pack2x16float(vec2f(o0[o + 2u], o0[o + 3u])));
    REF[rec + 1u] = vec2u(pack2x16float(vec2f(o1[o], o1[o + 1u])), pack2x16float(vec2f(o1[o + 2u], o1[o + 3u])));
    REF[rec + 2u] = vec2u(pack2x16float(vec2f(o2[o], o2[o + 1u])), pack2x16float(vec2f(o2[o + 2u], o2[o + 3u])));
  }
  if (t < TW) {
    var s0: f32 = 0.0;
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var k: u32 = 0u; k < TW; k += 1u) { s0 += o0[t * TW + k]; s1 += o1[t * TW + k]; s2 += o2[t * TW + k]; }
    let rows: u32 = N * (N / TW);
    let bp: u32 = P.bpartSlot * r + (wg.y * TW + t) * (N / TW) + wg.x;
    BPART[bp] = s0; BPART[bp + rows] = s1; BPART[bp + 2u * rows] = s2;
  }
}
`;
}

// ai: MEANS for one picture size, dispatched (meansWorkgroups(n), 1, R): a thread a row folds the row's n / 16 tile
// ai: sums of each base and writes the means as REF's vec4f (bbar0, bbar1, bbar2, 0), through the same vec2u view of
// ai: REF as the records (bitcast, so the means are f32); a thread past the last row (n = 384: 512 threads) returns.
// ai: Bound as BLUR is: PIC unused here.
export const MEANS_THREADS = 256;
export const meansWorkgroups = (n) => Math.ceil(n / MEANS_THREADS);
export function meansSource({ n }) {
  return PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> PIC: array<u32>;
@group(0) @binding(1) var<storage, read> PLAN: array<u32>;
@group(0) @binding(2) var<storage, read_write> REF: array<vec2u>;
@group(0) @binding(3) var<storage, read_write> BPART: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;
const N: u32 = ${n}u;
const TW: u32 = ${TILE}u;
` + SLOT_LIVE + /* wgsl */ `
@compute @workgroup_size(${MEANS_THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let r: u32 = wg.z;
  if (!slotLive(r)) { return; }
  let y: u32 = wg.x * ${MEANS_THREADS}u + t;
  if (y >= N) { return; }
  let rows: u32 = N * (N / TW);
  let bp: u32 = P.bpartSlot * r + y * (N / TW);
  var s0: f32 = 0.0;
  var s1: f32 = 0.0;
  var s2: f32 = 0.0;
  for (var k: u32 = 0u; k < N / TW; k += 1u) { s0 += BPART[bp + k]; s1 += BPART[bp + rows + k]; s2 += BPART[bp + 2u * rows + k]; }
  let inv: f32 = 1.0 / f32(N);
  let m: u32 = P.refSlot * r + P.refMeans + 2u * y;
  REF[m] = vec2u(bitcast<u32>(s0 * inv), bitcast<u32>(s1 * inv));
  REF[m + 1u] = vec2u(bitcast<u32>(s2 * inv), 0u);
}
`;
}
