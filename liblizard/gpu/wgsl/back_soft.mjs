// Back half part B: soft values, and the decline gate (DESIGN.md sections 1, 4 and 5; src/focus.c focus_finish_bits).
// One workgroup a block, 256 threads: sub-channel g = t / 32 of the block, lane l = t % 32 holding 10 of
// its 320 coefficients, gathered from S through the UV table (the S index of pos[i]). The per sub-channel
// moments are a 32-lane tree in workgroup memory; the block's estimate is the mean of its 8 and needs no atomic
// because the 8 sub-channels sit in the one workgroup. A lane's 10 coefficients are 20 consecutive int8 slots,
// five u32 words, so the pack is lane local. The bit map and whitening are part C's gather (MAP, bitmap.mjs):
// L stays in the C's slot order so it can be handed to focus_rx_ldpc as it is. Dispatch (blocks_s, 1, B), the
// frame on z as part A's passes take it, sized by the LISTS count in the kernel; or (blocks_s, 1, count_s)
// indirect from ARGS where a slot for it exists.
//
// Arithmetic is the C's in f32 where the C is double: m2, m4, a2, nv, k. round() is half to even, the C's
// nearbyintf. A slot within f32 drift of a rounding boundary may differ by one from the C.
import { SLOTS } from "./common.mjs";

export const LANES = 32, COEF_PER_LANE = 10, THREADS = 256;
// ai: The pilots (SPEC 7.3, 2026-09-30): a block's slots from PILOT_FROM (the codeword's end, n = 5088) to 5120 carry
// ai: the whitening XOR the painted picture's parity: the top 16 coefficients of its last sub-channel, lanes 30 and 31.
// ai: TW (a word a block, bit i the whitening of tail slot i: bitmap.mjs whiten) holds their signs; PILOT[f][b] gets
// ai: the block's r, the tail's axis values against them in units of the sub-channel's rms axis value sqrt(m2 / 2),
// ai: as src/focus.c reads it (the host averages a frame's blocks, gpu/decoder.mjs pilotOf).
export const PILOT_FROM = 5088, PILOT_BLOCKS = 128;
// ai: The grid's shift by the pilots (2026-10-01; the format paints the pilots whatever the phone, so a decoder may
// ai: steer by them; src/focus.c pilot_align, the same fit). A grid (dx, dy) samples off turns coefficient (u, v) by
// ai: 2 pi (u dx + v dy) / n. `align` (alignSource, one workgroup a frame, a thread a block, before the soft stage in
// ai: its pass and on its bind groups) reads every block's 16 pilot coefficients, each against its known symbol with
// ai: the block's sign from its own pilots' sum, fits zi = k zr (u dx + v dy) by least squares twice (the second about
// ai: the first's answer), and keeps the shift only where it stands ALIGN_CHI over the fit's residual variance (one
// ai: chance in a thousand for noise), else 0, 0. It goes to the end of PILOT: PILOT[B blocksMax + 2 f] and the next.
// ai: The soft stage turns every coefficient back by it as it gathers (a turn leaves |y|, so the moments and the
// ai: estimate read as before). UV's second half (from P.ucOff) is each coefficient's (u, v), u in the high 16 bits
// ai: (signed), v in the low.
export const ALIGN_CHI = 13.8;

export const softSource = ({ f16 = false } = {}) => (f16 ? "enable f16;\n" : "") + /* wgsl */ `
// P: which size this dispatch serves and the strides of every frame-indexed buffer.
struct Params { s: u32, subch: u32, blocks: u32, uvOff: u32, sStride: u32, lStride: u32, subchMax: u32, blocksMax: u32, B: u32, bar: f32, ucOff: u32, n: u32 }
@group(0) @binding(0) var<storage, read> S: array<${f16 ? "vec2<f16>" : "vec2<f32>"}>;
@group(0) @binding(1) var<storage, read> UV: array<u32>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(3) var<storage, read_write> L: array<u32>;
@group(0) @binding(4) var<storage, read_write> EST: array<f32>;
@group(0) @binding(5) var<storage, read_write> BLK: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> COUNTS: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> P: Params;
@group(0) @binding(8) var<storage, read_write> PILOT: array<f32>;
struct TailW { w: array<vec4u, ${PILOT_BLOCKS / 4}> }
@group(0) @binding(9) var<uniform> TW: TailW;

var<workgroup> red2: array<f32, ${THREADS}>;
var<workgroup> m2Of: array<f32, 8>;
var<workgroup> red4: array<f32, ${THREADS}>;
var<workgroup> kOf: array<f32, 8>;
var<workgroup> estOf: array<f32, 8>;

// axis_info (src/focus.c): the information an axis at SNR g carries.
fn jinfo(g: f32) -> f32 {
  if (g <= 0.0) { return 0.0; }
  let base = 1.0 - exp2(-0.3073 * pow(2.0 * sqrt(g), 1.7870));
  if (base <= 0.0) { return 0.0; }
  return pow(base, 1.1064);
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b: u32 = wg.x;
  // Dispatched (blocks, 1, B) over the batch: a workgroup past the size's frame count has no frame and leaves
  // before the first barrier. An indirect dispatch (blocks, 1, count) never gets here with z past the count.
  let count: u32 = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  let f: u32 = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  // ai: A block past the frame's own count (the version its word named, back_transform.mjs listsBlocks) is not run.
  if (b >= LISTS[${SLOTS}u * (P.B + 1u) + 2u * P.B + f]) { return; }
  let g: u32 = t >> 5u;
  let l: u32 = t & 31u;
  let j: u32 = 8u * b + g;
  let uvBase: u32 = P.uvOff + 320u * j + ${COEF_PER_LANE}u * l;
  let sBase: u32 = f * P.sStride;
  var y: array<vec2<f32>, ${COEF_PER_LANE}> = array<vec2<f32>, ${COEF_PER_LANE}>();
  var m2: f32 = 0.0;
  var m4: f32 = 0.0;
  // ai: the frame's shift (align, above): each coefficient turned back by it as it is read
  let sh: vec2<f32> = vec2<f32>(PILOT[P.B * P.blocksMax + 2u * f], PILOT[P.B * P.blocksMax + 2u * f + 1u]);
  let turn: bool = sh.x != 0.0 || sh.y != 0.0;
  let ucBase: u32 = P.ucOff + 320u * j + ${COEF_PER_LANE}u * l;
  for (var c: u32 = 0u; c < ${COEF_PER_LANE}u; c++) {
    var v: vec2<f32> = vec2<f32>(S[sBase + UV[uvBase + c]]);
    if (turn) {
      let w: u32 = UV[ucBase + c];
      let th: f32 = ${(2 * Math.PI).toFixed(9)} / f32(P.n) * (f32(bitcast<i32>(w) >> 16u) * sh.x + f32(w & 0xffffu) * sh.y);
      let cs: f32 = cos(th);
      let sn: f32 = sin(th);
      v = vec2<f32>(v.x * cs + v.y * sn, v.y * cs - v.x * sn);
    }
    y[c] = v;
    let e: f32 = dot(v, v);
    m2 += e;
    m4 += e * e;
  }
  red2[t] = m2;
  red4[t] = m4;
  workgroupBarrier();
  for (var h: u32 = 16u; h > 0u; h = h >> 1u) {
    if (l < h) { red2[t] = red2[t] + red2[t + h]; red4[t] = red4[t] + red4[t + h]; }
    workgroupBarrier();
  }
  if (l == 0u) {
    let mm2: f32 = red2[t] / 320.0;
    let mm4: f32 = red4[t] / 320.0;
    let a2: f32 = sqrt(max(2.0 * mm2 * mm2 - mm4, 0.0));
    let nv: f32 = max(mm2 - a2, 0.02 * mm2);
    var k: f32 = 0.0;
    var est: f32 = 0.0;
    // An all-zero sub-channel (an empty frame) has nv = 0: the C makes a NaN there, this makes nothing.
    if (nv > 0.0) { k = 0.7 * 2.0 * sqrt(2.0 * a2) / nv; est = jinfo(a2 / nv); }
    kOf[g] = k;
    estOf[g] = est;
    m2Of[g] = mm2;
    EST[f * P.subchMax + j] = est;
  }
  workgroupBarrier();
  let k: f32 = kOf[g];
  let lBase: u32 = f * P.lStride + 160u * j + 5u * l;
  for (var w: u32 = 0u; w < 5u; w++) {
    var word: u32 = 0u;
    for (var q: u32 = 0u; q < 2u; q++) {
      let v: vec2<f32> = clamp(k * y[2u * w + q], vec2<f32>(-10.0), vec2<f32>(10.0));
      let r: vec2<i32> = vec2<i32>(round(v * 8.0));
      word = word | ((u32(r.x) & 0xffu) << (16u * q)) | ((u32(r.y) & 0xffu) << (16u * q + 8u));
    }
    L[lBase + w] = word;
  }
  // ai: The pilots: this lane's share of the tail against the whitening's signs (red2 is free since the moments'
  // ai: barrier), summed by the sub-channel's lane 0.
  var cs: f32 = 0.0;
  if (g == ${Math.floor(PILOT_FROM / 2 / 320)}u) {
    let tw: u32 = TW.w[b >> 2u][b & 3u];
    for (var c: u32 = 0u; c < ${COEF_PER_LANE}u; c++) {
      let ci: u32 = ${COEF_PER_LANE}u * l + c;
      if (ci >= ${(PILOT_FROM / 2) % 320}u) {
        let q: u32 = ci - ${(PILOT_FROM / 2) % 320}u;
        cs += select(y[c].x, -y[c].x, ((tw >> (2u * q)) & 1u) != 0u) + select(y[c].y, -y[c].y, ((tw >> (2u * q + 1u)) & 1u) != 0u);
      }
    }
  }
  red2[t] = cs;
  workgroupBarrier();
  if (t == ${Math.floor(PILOT_FROM / 2 / 320) * 32}u) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 32u; i++) { sum += red2[t + i]; }
    let m: f32 = m2Of[${Math.floor(PILOT_FROM / 2 / 320)}u];
    PILOT[f * P.blocksMax + b] = select(0.0, sum / ${5120 - PILOT_FROM}.0 / sqrt(m / 2.0), m > 0.0);
  }
  if (t == 0u) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i++) { sum += estOf[i]; }
    let be: f32 = sum / 8.0;
    BLK[f * P.blocksMax + b] = vec2<u32>(bitcast<u32>(be), select(0u, 1u, be < P.bar));
    atomicAdd(&COUNTS[f * 8u + 6u], 8u);
  }
}
`;

// ai: The shift's fit (the comment at ALIGN_CHI). The soft stage's bindings and bind groups; dispatch (1, 1, B) a
// ai: size, the frame on z as the soft stage takes it.
export const alignSource = ({ f16 = false } = {}) => (f16 ? "enable f16;\n" : "") + /* wgsl */ `
struct Params { s: u32, subch: u32, blocks: u32, uvOff: u32, sStride: u32, lStride: u32, subchMax: u32, blocksMax: u32, B: u32, bar: f32, ucOff: u32, n: u32 }
@group(0) @binding(0) var<storage, read> S: array<${f16 ? "vec2<f16>" : "vec2<f32>"}>;
@group(0) @binding(1) var<storage, read> UV: array<u32>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(7) var<uniform> P: Params;
@group(0) @binding(8) var<storage, read_write> PILOT: array<f32>;
struct TailW { w: array<vec4u, ${PILOT_BLOCKS / 4}> }
@group(0) @binding(9) var<uniform> TW: TailW;

var<workgroup> acc: array<f32, ${7 * PILOT_BLOCKS}>;
var<workgroup> cur: vec2<f32>;
var<workgroup> last: array<f32, 7>;

@compute @workgroup_size(${PILOT_BLOCKS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let count: u32 = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  let f: u32 = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  let blocks: u32 = min(LISTS[${SLOTS}u * (P.B + 1u) + 2u * P.B + f], ${PILOT_BLOCKS}u);
  let kk: f32 = ${(2 * Math.PI).toFixed(9)} / f32(P.n);
  if (t == 0u) { cur = vec2<f32>(0.0); }
  workgroupBarrier();
  for (var it: u32 = 0u; it < 2u; it++) {
    var a: f32 = 0.0; var b: f32 = 0.0; var c: f32 = 0.0; var p: f32 = 0.0; var q: f32 = 0.0; var yy: f32 = 0.0; var used: f32 = 0.0;
    if (t < blocks) {
      let d: vec2<f32> = cur;
      let at: u32 = 320u * (8u * t + 7u) + ${(PILOT_FROM / 2) % 320}u;
      let tw: u32 = TW.w[t >> 2u][t & 3u];
      var z: array<vec2<f32>, ${(5120 - PILOT_FROM) / 2}>;
      var uv: array<vec2<f32>, ${(5120 - PILOT_FROM) / 2}>;
      var sum: f32 = 0.0;
      for (var i: u32 = 0u; i < ${(5120 - PILOT_FROM) / 2}u; i++) {
        let y0: vec2<f32> = vec2<f32>(S[f * P.sStride + UV[P.uvOff + at + i]]);
        let w: u32 = UV[P.ucOff + at + i];
        let u: f32 = f32(bitcast<i32>(w) >> 16u);
        let v: f32 = f32(w & 0xffffu);
        let th: f32 = kk * (u * d.x + v * d.y);
        let cs: f32 = cos(th);
        let sn: f32 = sin(th);
        let y: vec2<f32> = vec2<f32>(y0.x * cs + y0.y * sn, y0.y * cs - y0.x * sn);
        let xr: f32 = select(1.0, -1.0, ((tw >> (2u * i)) & 1u) != 0u);
        let xi: f32 = select(1.0, -1.0, ((tw >> (2u * i + 1u)) & 1u) != 0u);
        z[i] = vec2<f32>(y.x * xr + y.y * xi, y.y * xr - y.x * xi);
        uv[i] = vec2<f32>(u, v);
        sum += z[i].x;
      }
      // the block's sign: its pilots' own sum
      let sg: f32 = select(1.0, -1.0, sum < 0.0);
      for (var i: u32 = 0u; i < ${(5120 - PILOT_FROM) / 2}u; i++) {
        let zr: f32 = sg * z[i].x;
        let zi: f32 = sg * z[i].y;
        let cu: f32 = kk * zr * uv[i].x;
        let cv: f32 = kk * zr * uv[i].y;
        a += cu * cu; b += cu * cv; c += cv * cv; p += cu * zi; q += cv * zi; yy += zi * zi;
      }
      used = ${(5120 - PILOT_FROM) / 2}.0;
    }
    acc[7u * t] = a; acc[7u * t + 1u] = b; acc[7u * t + 2u] = c; acc[7u * t + 3u] = p; acc[7u * t + 4u] = q; acc[7u * t + 5u] = yy; acc[7u * t + 6u] = used;
    workgroupBarrier();
    if (t == 0u) {
      var s: array<f32, 7>;
      for (var k: u32 = 0u; k < 7u; k++) { s[k] = 0.0; }
      for (var i: u32 = 0u; i < ${PILOT_BLOCKS}u; i++) { for (var k: u32 = 0u; k < 7u; k++) { s[k] += acc[7u * i + k]; } }
      let det: f32 = s[0] * s[2] - s[1] * s[1];
      if (s[6] >= 8.0 && det > 0.0) { cur = cur + vec2<f32>((s[3] * s[2] - s[4] * s[1]) / det, (s[0] * s[4] - s[1] * s[3]) / det); } else { s[6] = 0.0; }
      for (var k: u32 = 0u; k < 7u; k++) { last[k] = s[k]; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    // the last fit's residual variance (about its own answer), and the whole shift against it
    var out: vec2<f32> = vec2<f32>(0.0);
    let det: f32 = last[0] * last[2] - last[1] * last[1];
    if (last[6] >= 8.0 && det > 0.0) {
      let e: vec2<f32> = vec2<f32>((last[3] * last[2] - last[4] * last[1]) / det, (last[0] * last[4] - last[1] * last[3]) / det);
      let s2: f32 = (last[5] - e.x * last[3] - e.y * last[4]) / (last[6] - 2.0);
      let d: vec2<f32> = cur;
      if (s2 > 0.0 && (d.x * d.x * last[0] + 2.0 * d.x * d.y * last[1] + d.y * d.y * last[2]) > ${ALIGN_CHI} * s2) { out = d; }
    }
    PILOT[P.B * P.blocksMax + 2u * f] = out.x;
    PILOT[P.B * P.blocksMax + 2u * f + 1u] = out.y;
  }
}
`;
