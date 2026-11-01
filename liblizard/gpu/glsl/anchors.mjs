// The anchors in GLSL: the WGSL anchors.mjs (archived) statement for statement (its comments say what each stage does, SPEC 6.13):
// ANCH_FIND (a workgroup of 128 an (anchor, frame): the squares found by normalised correlation from integral images,
// coarse then fine), ANCH_FIT (the knots, the fill, the picture's mean, the record), and anchRefitSource: ANCH_REFIT
// (the lens about the axis over the border and the anchors, robust over the border, the sides' curves) and
// ANCH_PREDICT (the fit over the anchors alone, the border's places for its second search).
// ai: Bindings, workgroup sizes and layouts are the WGSL's. Workgroup memory is not zeroed but where the WGSL reads it
// ai: before writing it: ANCH_REFIT's three counters (wcnt, wnba, wchg), whose atomics start from WGSL's zero, are
// ai: null-initialised (GL_EXT_null_initializer: the same OpConstantNull initializer naga writes for every workgroup
// ai: variable).
import { VERSION, SAMPLERLESS, NULL_INIT, DIMS, CONVERT, MAP, LUM0, lit, zeros, uniformLoad, STORAGE_BARRIER } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { LATTICE_MAX, ANCH_SIDE, ANCH_K, ANCH_REC } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { NODES_MAX, RESID, WMIN, LENS_P } from "../plan/register.mjs";
import { ANCH_TS, ANCH_PEAK, ANCH_MIN, ANCH_ROUNDS } from "../plan/anchors.mjs";
import { lensNode, lensAccumulate, lensModel, lensNormalSolve, SIDE_FITS, SIDE_DECL } from "./register.mjs";

const RR = ANCH_TS, TC = ANCH_TS / 2, RC = RR / 2, WC = TC + 2 * RC, W1C = WC + 1, OC = 2 * RC + 1;
const FD = 3, WF = ANCH_TS + 2 * FD, W1F = WF + 1, OF = 2 * FD + 1;
const pattern = (ts) => {
  const dist = (t) => Math.abs((t + 0.5) / (ts / 2) - 1);
  const span1 = (lim) => { const q = [...Array(ts).keys()].filter((t) => dist(t) < lim); return [q[0], q.at(-1)]; };
  const [i25, j25] = span1(0.25), [i6, j6] = span1(0.6);
  const dark = (j6 - i6 + 1) ** 2 - (j25 - i25 + 1) ** 2, light = ts * ts - dark;
  const tm = (light - dark) / (ts * ts);
  return { ts, i25, j25, i6, j6, tm, tn: Math.fround(Math.sqrt(light * (1 - tm) ** 2 + dark * (1 + tm) ** 2)) };
};

// ai: What both kernels bind first, 0 to 6 (the WGSL anchors.mjs (archived) HEAD), and the final map at a module point.
const HEAD = DIMS + CONVERT + MAP + /* glsl */ `
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 2, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 3, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 4, std430) readonly buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 5, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LT;
layout(set = 0, binding = 6, std140) uniform Anchors { vec4 k[${3 * RING_COUNT}]; } A;
float latAt(uint i) { return LT.lat[i >> 2u][i & 3u]; }
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
` + COONS + LUM0 + /* glsl */ `
vec2 mapAt(uint f, float M[16], Lens lz, uint first, uint nn, vec2 m) {
  vec2 p = toImage(M, m);
  return p + lensAt(lz, p) + coons(f, first, nn, m);
}
`;

const AL = 128;
const box = (I) => (x, y, sz, w1) => `(${I}[(${y} + ${sz}) * ${w1} + ${x} + ${sz}] - ${I}[${y} * ${w1} + ${x} + ${sz}] - ${I}[(${y} + ${sz}) * ${w1} + ${x}] + ${I}[${y} * ${w1} + ${x}])`;
const scoreFn = (name, I1, I2, w1, { ts, i25, j25, i6, j6, tm, tn }) => /* glsl */ `
float ${name}(uint px, uint py) {
  float s1 = ${box(I1)("px", "py", `${ts}u`, `${w1}u`)};
  float s2 = ${box(I2)("px", "py", `${ts}u`, `${w1}u`)};
  float vr = s2 - s1 * s1 / ${lit(ts * ts)};
  float s25 = ${box(I1)(`(px + ${i25}u)`, `(py + ${i25}u)`, `${j25 - i25 + 1}u`, `${w1}u`)};
  float s6 = ${box(I1)(`(px + ${i6}u)`, `(py + ${i6}u)`, `${j6 - i6 + 1}u`, `${w1}u`)};
  float st = 2.0 * (s25 - s6) + s1 * ${lit(1 - tm)};
  float sc = st / (sqrt(vr) * ${lit(tn)});
  return vr > VAR_MIN ? sc : 0.0;
}`;
const integral = (I1, I2, w, w1) => /* glsl */ `
  if (li < ${2 * w}u) {
    uint r = (li % ${w}u + 1u) * ${w1}u;
    float a = 0.0;
    if (li < ${w}u) { for (uint x = 1u; x <= ${w}u; x++) { a += ${I1}[r + x]; ${I1}[r + x] = a; } }
    else { for (uint x = 1u; x <= ${w}u; x++) { a += ${I2}[r + x]; ${I2}[r + x] = a; } }
  }
  barrier();
  if (li < ${2 * w}u) {
    uint c = li % ${w}u + 1u;
    float a = 0.0;
    if (li < ${w}u) { for (uint y = 1u; y <= ${w}u; y++) { a += ${I1}[y * ${w1}u + c]; ${I1}[y * ${w1}u + c] = a; } }
    else { for (uint y = 1u; y <= ${w}u; y++) { a += ${I2}[y * ${w1}u + c]; ${I2}[y * ${w1}u + c] = a; } }
  }
  barrier();`;
const samples = (I1, I2, w, w1) => /* glsl */ `
  for (uint t = li; t < ${w * w}u; t += ${AL}u) {
    uint wx = t % ${w}u;
    uint wy = t / ${w}u;
    vec2 p0 = toImage(M, vec2(wcx[wx], wry[wy]));
    float v = bilin0(f, p0 + lensAt(lz, p0) + coonsMix(wcu[wx], wrv[wy], wct[wx], wcb[wx], wrl[wy], wrr[wy], cn), fw, fh);
    uint o = (wy + 1u) * ${w1}u + wx + 1u;
    ${I1}[o] = v;
    ${I2}[o] = v * v;
  }
  barrier();`;
const coords = (w, lo, scale) => /* glsl */ `
  for (uint t = li; t < ${2 * w}u; t += ${AL}u) {
    if (t < ${w}u) {
      float mx = x0 + (float(${lo.x}) + ${lit(scale)} * float(t) + ${lit(scale / 2)}) * h;
      wcx[t] = mx;
      wcu[t] = coonsT(mx, first, nn);
      wct[t] = curve(f, 0u, mx, first, nn);
      wcb[t] = curve(f, 2u, mx, first, nn);
    } else {
      uint s = t - ${w}u;
      float my = y0 + (float(${lo.y}) + ${lit(scale)} * float(s) + ${lit(scale / 2)}) * h;
      wry[s] = my;
      wrv[s] = coonsT(my, first, nn);
      wrl[s] = curve(f, 3u, my, first, nn);
      wrr[s] = curve(f, 1u, my, first, nn);
    }
  }`;
const pack = (s, o) => `(f2u(clamp(${s} + 1.0, 0.0, 1.999) * 1048575.0) << 11u) | (2047u - ${o})`;
export const ANCH_FIND = VERSION + SAMPLERLESS + HEAD + /* glsl */ `
layout(local_size_x = ${AL}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) buffer Anch { vec4 anch[]; };
const uint TS = ${ANCH_TS}u;
const int RR = ${RR};
const int FD = ${FD};
const float VAR_MIN = 1e-6;
shared float IC1[${W1C * W1C}];
shared float IC2[${W1C * W1C}];
shared float IF1[${W1F * W1F}];
shared float IF2[${W1F * W1F}];
shared float CF[${OF * OF}];
shared float wcx[${WC}];
shared float wcu[${WC}];
shared vec2 wct[${WC}];
shared vec2 wcb[${WC}];
shared float wry[${WC}];
shared float wrv[${WC}];
shared vec2 wrl[${WC}];
shared vec2 wrr[${WC}];
shared uint bestC;
shared uint bestF;
${scoreFn("scoreC", "IC1", "IC2", W1C, pattern(TC))}
${scoreFn("scoreF", "IF1", "IF2", W1F, pattern(ANCH_TS))}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  uint k = wg.x;
  if (k >= ${ANCH_K}u) { return; }
  Frame F = frames[f];
  uvec4 pick = sel[f];
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { anch[${ANCH_K}u * f + k] = vec4(0.0); }
    return;
  }
  uint rgi = pick.x;
  float a = A.k[3u * rgi + 2u].y;
  float h = a / float(TS);
  int rr = RR;
  float x0 = knot(rgi, 1u + k % ${ANCH_SIDE}u) - 0.5 * a;
  float y0 = knot(rgi, 1u + k / ${ANCH_SIDE}u) - 0.5 * a;
  float M[16] = maps[f];
  Lens lz = lensOf(f);
  uvec4 at = LT.at[rgi];
  uint first = at.x;
  uint nn = at.y;
  Corners cn = corners(f, nn);
  int fw = int(F.w);
  int fh = int(F.h);
  ${coords(WC, { x: -RR, y: -RR }, 2)}
  for (uint t = li; t < ${W1C}u; t += ${AL}u) { IC1[t] = 0.0; IC2[t] = 0.0; IC1[t * ${W1C}u] = 0.0; IC2[t * ${W1C}u] = 0.0; }
  for (uint t = li; t < ${W1F}u; t += ${AL}u) { IF1[t] = 0.0; IF2[t] = 0.0; IF1[t * ${W1F}u] = 0.0; IF2[t * ${W1F}u] = 0.0; }
  if (li == 0u) { bestC = 0u; bestF = 0u; }
  barrier();
  ${samples("IC1", "IC2", WC, W1C)}
  ${integral("IC1", "IC2", WC, W1C)}
  for (uint o = li; o < ${OC * OC}u; o += ${AL}u) { atomicMax(bestC, ${pack("scoreC(o % " + OC + "u, o / " + OC + "u)", "o")}); }
  barrier();
  uint oc = 2047u - (bestC & 2047u);
  int cx = clamp(2 * (int(oc % ${OC}u) - ${RC}), FD - rr, rr - FD);
  int cy = clamp(2 * (int(oc / ${OC}u) - ${RC}), FD - rr, rr - FD);
  ${coords(WF, { x: "cx - FD", y: "cy - FD" }, 1)}
  barrier();
  ${samples("IF1", "IF2", WF, W1F)}
  ${integral("IF1", "IF2", WF, W1F)}
  for (uint o = li; o < ${OF * OF}u; o += ${AL}u) {
    float s = scoreF(o % ${OF}u, o / ${OF}u);
    CF[o] = s;
    atomicMax(bestF, ${pack("s", "o")});
  }
  barrier();
  if (li == 0u) {
    uint po = 2047u - (bestF & 2047u);
    float pk = CF[po];
    int px = int(po % ${OF}u);
    int py = int(po / ${OF}u);
    int bx = cx - FD + px;
    int by = cy - FD + py;
    float sx = 0.0;
    float sy = 0.0;
    if (px > 0 && px < 2 * FD) { float cl = CF[po - 1u]; float cr = CF[po + 1u]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sx = 0.5 * (cl - cr) / d; } }
    if (py > 0 && py < 2 * FD) { float cl = CF[po - ${OF}u]; float cr = CF[po + ${OF}u]; float d = cl - 2.0 * pk + cr; if (d < 0.0) { sy = 0.5 * (cl - cr) / d; } }
    bool ok = pk >= ${lit(ANCH_PEAK)} && bx > -rr && bx < rr && by > -rr && by < rr;
    vec2 X = vec2(x0 + 0.5 * a, y0 + 0.5 * a);
    vec2 D = vec2((float(bx) + sx) * h, (float(by) + sy) * h);
    vec2 rk = mapAt(f, M, lz, first, nn, X + D) - mapAt(f, M, lz, first, nn, X);
    anch[${ANCH_K}u * f + k] = vec4(rk, pk, ok ? 1.0 : 0.0);
  }
}
`;

export const ANCH_FIT = VERSION + SAMPLERLESS + HEAD + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
struct Result { float r[32]; };
layout(set = 0, binding = 7, std430) readonly buffer Anch { vec4 anch[]; };
layout(set = 0, binding = 8, std430) buffer Arec { vec4 arec[]; };
layout(set = 0, binding = 9, std430) buffer Results { Result result[]; };
const uint NK = ${ANCH_K}u;
const uint SIDE = ${ANCH_SIDE}u;
shared vec2 kv[${ANCH_K}];
shared uint kn[${ANCH_K}];
shared uint kn2[${ANCH_K}];
shared float red[64];
shared uint wfound;
vec3 around(uint q) {
  uint i = q % SIDE;
  uint j = q / SIDE;
  vec2 s = vec2(0.0);
  float c = 0.0;
  if (i > 0u && kn[q - 1u] == 1u) { s += kv[q - 1u]; c += 1.0; }
  if (i + 1u < SIDE && kn[q + 1u] == 1u) { s += kv[q + 1u]; c += 1.0; }
  if (j > 0u && kn[q - SIDE] == 1u) { s += kv[q - SIDE]; c += 1.0; }
  if (j + 1u < SIDE && kn[q + SIDE] == 1u) { s += kv[q + SIDE]; c += 1.0; }
  return vec3(s, c);
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  Frame F = frames[f];
  uvec4 pick = sel[f];
  uint ab = f * ${ANCH_REC}u;
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { arec[ab] = vec4(0.0); result[f].r[31] = 0.0; }
    return;
  }
  uint rgi = pick.x;
  if (li < NK) {
    vec4 e = anch[NK * f + li];
    bool ok = e.w > 0.5;
    kv[li] = ok ? e.xy : vec2(0.0);
    kn[li] = ok ? 1u : 0u;
  }
  barrier();
  if (li == 0u) {
    uint c = 0u;
    for (uint q = 0u; q < NK; q++) { c += kn[q]; }
    wfound = c;
  }
  ${uniformLoad("uint found", "wfound")}
  if (li == 0u) { result[f].r[31] = float(found); }
  if (found < ${ANCH_MIN}u && li < NK) { kv[li] = vec2(0.0); kn[li] = 1u; }
  for (uint rd = 0u; rd < ${ANCH_ROUNDS}u; rd++) {
    barrier();
    if (li < NK) {
      uint now = kn[li];
      if (now == 0u) {
        vec3 nb = around(li);
        if (nb.z > 0.0) { kv[li] = nb.xy / nb.z; now = 1u; }
      }
      kn2[li] = now;
    }
    barrier();
    if (li < NK) { kn[li] = kn2[li]; }
  }
  vec4 geo = A.k[3u * rgi + 2u];
  float span = geo.z;
  float margin = geo.w;
  float M[16] = maps[f];
  Lens lz = lensOf(f);
  uvec4 at = LT.at[rgi];
  float sum = 0.0;
  for (uint p = li; p < 256u; p += 64u) {
    vec2 m = vec2(margin + span * (float(p % 16u) + 0.5) / 16.0, margin + span * (float(p / 16u) + 0.5) / 16.0);
    sum += bilin0(f, mapAt(f, M, lz, at.x, at.y, m), int(F.w), int(F.h));
  }
  red[li] = sum;
  barrier();
  if (li == 0u) {
    float s = 0.0;
    for (uint q = 0u; q < 64u; q++) { s += red[q]; }
    arec[ab] = vec4(1.0, s / 256.0, 0.0, float(found));
    arec[ab + 1u] = A.k[3u * rgi];
    arec[ab + 2u] = A.k[3u * rgi + 1u];
    arec[ab + 3u] = geo;
  }
  if (li < ${ANCH_REC - 4}u) {
    uint q = 2u * li;
    vec2 kq = kv[min(q + 1u, NK - 1u)];
    arec[ab + 4u + li] = vec4(kv[q], q + 1u < NK ? kq : vec2(0.0));
  }
}
`;

// ai: ANCH_REFIT and ANCH_PREDICT (the WGSL anchors.mjs (archived) anchRefitSource, its comments).
const NPTS = NODES_MAX + ANCH_K;
const TAU_SPREAD = String(Math.fround(3.0 * 1.4826));   // ai: the WGSL's 3.0 * 1.4826, folded in abstract precision then rounded to f32
export const anchRefitSource = ({ predict = false } = {}) => VERSION + NULL_INIT + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Nodes { vec4 nodes[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 3, std430) buffer Meas { vec4 meas[]; };
layout(set = 0, binding = 4, std430) buffer Maps { float maps[][16]; };
layout(set = 0, binding = 5, std430) buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 6, std140) uniform Fit { uvec4 count; uvec4 base; vec4 modules; } P;
layout(set = 0, binding = 7, std430) buffer Anch { vec4 anch[]; };
layout(set = 0, binding = 8, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LAT;
layout(set = 0, binding = 9, std140) uniform Anchors { vec4 k[${3 * RING_COUNT}]; } A;
float latAt(uint i) { return LAT.lat[i >> 2u][i & 3u]; }
float knot(uint r, uint i) { return A.k[3u * r + (i >> 2u)][i & 3u]; }
` + COONS + /* glsl */ `
struct LensFrame { vec2 ctr; float sig; float smid; float ish; uint kmax; uint nw; uint on; };
const uint NK = ${ANCH_K}u;
const uint SIDE = ${ANCH_SIDE}u;
shared vec4 pp[${NPTS}];
shared float pw[${NPTS}];
shared float pr[${NODES_MAX}];
shared uint pd[${NODES_MAX}];
shared uint ak[${ANCH_K}];
shared LensFrame LF;
shared float LS[16][77];
shared float LT[4][${LENS_P}];
shared uint LK[4];
shared vec4 LR[64];
shared vec4 LQ[64];
shared float LP[77];
shared Lens LZ;
shared vec4 rw[${NODES_MAX}];
${SIDE_DECL}shared uint wu;
shared uint wok;
shared float wmed;
shared uint wnb;
shared float wtau;
shared uint wcnt = {};
shared uint wnba = {};
shared uint wchg = {};
shared vec4 LR8[8];
shared vec4 LQ8[8];
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  uvec4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  Frame F = frames[f];
  uint rgi = pick.x;
  uint count = P.count[rgi];
  uint base = P.base[rgi];
  if (li < NK) {
    bool on = anch[NK * f + li].w > 0.5;
    ak[li] = on ? 1u : 0u;
    if (on) { atomicAdd(wcnt, 1u); }
  }
  barrier();
  if (li == 0u) { wu = wcnt; }
  { ${uniformLoad("uint wu0", "wu")} if (wu0 < ${ANCH_MIN}u) { return; } }
  float M[16] = maps[f];
  Lens L0 = lensOf(f);
  uvec4 at = LAT.at[rgi];
  uint first = at.x;
  uint nl = at.y;
${predict ? `  for (uint n = li; n < count; n += 64u) { pp[n] = vec4(toImage(M, nodes[base + n].xy), 0.0, 0.0); }` : `  for (uint n = li; n < count; n += 64u) {
    vec4 e = meas[f * ${NODES_MAX}u + n];
    bool ok = e.w != 0.0 && e.z >= ${WMIN};
    vec2 p = toImage(M, nodes[base + n].xy);
    vec2 pe = e.xy - p;
    pp[n] = vec4(p, ok ? pe : vec2(0.0));
    pw[n] = ok ? e.z : 0.0;
    pd[n] = 0u;
    if (ok) { atomicAdd(wnba, 1u); }
  }`}
  if (li < NK) {
    vec4 e = anch[NK * f + li];
    vec2 X = vec2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    vec2 p = toImage(M, X);
    vec2 h = p;
    bool on = ak[li] == 1u;
    vec2 pe = p + e.xy + lensAt(L0, p) + coons(f, first, nl, X) - h;
    pp[count + li] = vec4(h, on ? pe : vec2(0.0));
    pw[count + li] = on ? 1.0 : 0.0;
  }
  ${STORAGE_BARRIER}
  barrier();
  uint npts = count + NK;
  uint aw = F.size;
  vec2 LA = vec2(0.5 * float(F.w), 0.5 * float(F.h));
  if (aw != 0u) { LA = vec2(0.5 * float(aw & 0xffffu) - 16384.0, 0.5 * float(aw >> 16u) - 16384.0); }
  vec2 fwh = vec2(float(F.w), float(F.h));
  float lR = max(max(length(LA), length(vec2(fwh.x, 0.0) - LA)), max(length(vec2(0.0, fwh.y) - LA), length(fwh - LA)));
  float lir = 1.0 / max(lR, 1.0);
  {
    vec2 lo = vec2(3.0e38);
    vec2 hi = vec2(-3.0e38);
    float smin = 3.0e38;
    float smax = 0.0;
    uint nw = 0u;
    for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 64u) {
      if (!(pw[i] > 0.0)) { continue; }
      vec2 p = pp[i].xy;
      lo = min(lo, p);
      hi = max(hi, p);
      vec2 q = (p - LA) * lir;
      float s = dot(q, q);
      smin = min(smin, s);
      smax = max(smax, s);
      nw += 1u;
    }
    LR[li] = vec4(lo, hi);
    LQ[li] = vec4(smin, smax, float(nw), 0.0);
  }
  barrier();
  if (li < 8u) {
    vec2 lo = LR[8u * li].xy;
    vec2 hi = LR[8u * li].zw;
    float smin = LQ[8u * li].x;
    float smax = LQ[8u * li].y;
    float nw = LQ[8u * li].z;
    for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) {
      lo = min(lo, LR[s].xy);
      hi = max(hi, LR[s].zw);
      smin = min(smin, LQ[s].x);
      smax = max(smax, LQ[s].y);
      nw += LQ[s].z;
    }
    LR8[li] = vec4(lo, hi);
    LQ8[li] = vec4(smin, smax, nw, 0.0);
  }
  barrier();
  if (li == 0u) {
    vec2 lo = LR8[0].xy;
    vec2 hi = LR8[0].zw;
    float smin = LQ8[0].x;
    float smax = LQ8[0].y;
    uint nw = f2u(LQ8[0].z);
    for (uint s = 1u; s < 8u; s++) {
      lo = min(lo, LR8[s].xy);
      hi = max(hi, LR8[s].zw);
      smin = min(smin, LQ8[s].x);
      smax = max(smax, LQ8[s].y);
      nw += f2u(LQ8[s].z);
    }
    LF = LensFrame(0.5 * (lo + hi), 2.0 / max(max(hi.x - lo.x, hi.y - lo.y), 1.0), 0.5 * (smin + smax),
      2.0 / max(smax - smin, 1e-6), 3u, nw, nw > 0u ? 1u : 0u);
    wok = 0u;
${predict ? "  }" : `    wnb = wnba;
    uint nq = (count + 4u) / 4u;
    float px = 0.0;
    float md = 0.0;
    for (uint c = 0u; c < 4u; c++) {
      uint a = (c * (nq - 1u)) % count;
      uint b = (((c + 1u) % 4u) * (nq - 1u)) % count;
      px += distance(pp[a].xy, pp[b].xy);
      md += distance(nodes[base + a].xy, nodes[base + b].xy);
    }
    wtau = 0.5 * px / max(md, 1.0);
  }`}
  barrier();
  ${predict ? `{` : "for (uint rd = 0u; rd < 4u; rd++) {"}
    if (li < 16u) {
      float acc[77] = ${zeros(77)};
      for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 16u) {
        float w = ${predict ? "pw[i]" : `(i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u) ? 0.0 : pw[i]`};
        if (!(w > 0.0)) { continue; }
        vec2 p = pp[i].xy;
        vec2 r = pp[i].zw;
        ${lensNode}
        ${lensAccumulate}
      }
      LS[li] = acc;
    }
    barrier();
    for (uint e = li; e < 77u; e += 64u) {
      float a = 0.0;
      for (uint s = 0u; s < 16u; s++) { a += LS[s][e]; }
      LP[e] = a;
    }
    barrier();
    if (li == 0u) {
      for (uint k = 0u; k < 4u; k++) { LK[k] = 0u; }
${lensNormalSolve}    }
    barrier();
    {
      vec4 rs = vec4(0.0);
      uint nu = 0u;
      for (uint i = ${predict ? "count + li" : "li"}; i < npts; i += 64u) {
        float w = ${predict ? "pw[i]" : `(i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u) ? 0.0 : pw[i]`};
        if (!(w > 0.0)) { continue; }
        nu += 1u;
        vec2 p = pp[i].xy;
        vec2 r = pp[i].zw;
        ${lensNode}
        ${[0, 1, 2, 3].map((K) => `if (LK[${K}] != 0u) { vec2 e = r - ${lensModel(K)}; rs[${K}] += w * dot(e, e); }`).join("\n        ")}
      }
      LR[li] = rs;
      LQ[li] = vec4(float(nu), 0.0, 0.0, 0.0);
    }
    barrier();
    if (li < 8u) {
      vec4 rss = LR[8u * li];
      float nu = LQ[8u * li].x;
      for (uint s = 8u * li + 1u; s < 8u * li + 8u; s++) { rss += LR[s]; nu += LQ[s].x; }
      LR8[li] = rss;
      LQ8[li] = vec4(nu, 0.0, 0.0, 0.0);
    }
    barrier();
    if (li == 0u) {
      vec4 rss = vec4(0.0);
      uint nu = 0u;
      for (uint s = 0u; s < 8u; s++) { rss += LR8[s]; nu += f2u(LQ8[s].x); }
      uint kb = 4u;
      float best = 3.0e38;
      float nobs = 2.0 * float(nu);
      for (uint k = 0u; k < 4u; k++) {
        float pk = 8.0 + float(k);
        if (LK[k] == 0u || nobs - pk - 1.0 <= 0.0) { continue; }
        float score = nobs * log(max(rss[k], 1e-30) / nobs) + 2.0 * pk + 2.0 * pk * (pk + 1.0) / (nobs - pk - 1.0);
        if (score < best) { best = score; kb = k; }
      }
      float tk[${LENS_P}] = ${zeros(LENS_P)};
      ${[0, 1, 2, 3].map((K) => `if (kb == ${K}u) { ${Array.from({ length: LENS_P }, (_, a) => `tk[${a}] = LT[${K}][${a}];`).join(" ")} }`).join("\n      ")}
      if (kb < 4u) {
        LZ = Lens(vec4(LF.ctr, LF.sig, LF.smid), vec4(LA, lir, LF.ish), vec4(tk[0], tk[1], tk[2], tk[3]),
          vec4(tk[4], tk[5], tk[6], tk[7]), vec4(tk[8], tk[9], tk[10], 0.0));
        wok = 1u;
      }
      wu = kb < 4u ? 1u : 0u;
    }
${predict ? `  }
` : `    { ${uniformLoad("uint wu1", "wu")} if (wu1 == 0u || rd == 3u) { break; } }
    Lens lz = LZ;
    for (uint n = li; n < count; n += 64u) { float dn = length(pp[n].zw - lensAt(lz, pp[n].xy)); pr[n] = pw[n] > 0.0 ? dn : 0.0; }
    if (li == 0u) { wmed = 0.0; wchg = 0u; }
    barrier();
    uint nb = wnb;
    for (uint n = li; n < count; n += 64u) {
      if (!(pw[n] > 0.0)) { continue; }
      float v = pr[n];
      uint rank = 0u;
      for (uint j = 0u; j < count; j++) { float u = pr[j]; rank += (pw[j] > 0.0 && (u < v || (u == v && j < n))) ? 1u : 0u; }
      if (rank == nb / 2u) { wmed = v; }
    }
    barrier();
    float tau = max(wtau, ${TAU_SPREAD} * wmed);
    for (uint n = li; n < count; n += 64u) {
      uint d = (pw[n] > 0.0 && pr[n] > tau) ? 1u : 0u;
      if (d != pd[n]) { atomicAdd(wchg, 1u); }
      pd[n] = d;
    }
    barrier();
    if (li == 0u) { wu = wchg; }
    { ${uniformLoad("uint wu2", "wu")} if (wu2 == 0u) { break; } }
  }
`}  { ${uniformLoad("uint wok0", "wok")} if (wok0 == 0u) { return; } }
  Lens lz = LZ;
${predict ? `  for (uint n = li; n < count; n += 64u) { meas[f * ${NODES_MAX}u + n] = vec4(pp[n].xy + lensAt(lz, pp[n].xy), 0.0, 2.0); }
  return;
` : ""}  if (li == 0u) {
    uint b = f * ${RESID}u + ${NODES_MAX}u;
    resid[b] = lz.c;
    resid[b + 1u] = lz.a;
    resid[b + 2u] = lz.t0;
    resid[b + 3u] = lz.t1;
    resid[b + 4u] = lz.t2;
  }
  for (uint n = li; n < count; n += 64u) {
    vec4 vk = vec4(pp[n].zw - lensAt(lz, pp[n].xy), pw[n], 0.0);
    vec4 vd = vec4(0.0, 0.0, pw[n], 0.0);
    vec4 v = pd[n] == 1u ? vd : vk;
    resid[f * ${RESID}u + n] = v;
    rw[n] = v;
  }
  ${STORAGE_BARRIER}
  barrier();
` + SIDE_FITS + /* glsl */ `
  ${STORAGE_BARRIER}
  barrier();
  if (li < NK && ak[li] == 1u) {
    vec2 X = vec2(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    vec4 e = anch[NK * f + li];
    vec4 q = pp[count + li];
    anch[NK * f + li] = vec4(q.zw - lensAt(lz, q.xy) - coons(f, first, nl, X), e.zw);
  }
}
`;
export const ANCH_REFIT = anchRefitSource(), ANCH_PREDICT = anchRefitSource({ predict: true });
