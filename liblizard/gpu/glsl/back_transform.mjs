// The back half's transform in GLSL, the WGSL back_transform.mjs (archived) statement for statement (that file's comments say
// what each pass does, its LISTS and ARGS layout, and gpu/back/DESIGN.md sections 1 to 3 and 10 the arithmetic):
// the gate, the fused pass 1 (the map on a node grid, the anchors' mesh and erase, the column FFT of four columns),
// reduce (the five surface coefficients) and pass 2 (the row FFT and the detrend). Built a batch size and a picture
// size each, as the WGSL is.
// ai: The gate: a batch of B frames listed by picture slot, each listed frame's version,
// ai: rows and ring, the passes' indirect dispatches, and the chain's counters, verdicts, iteration counts and record
// ai: count zeroed. One workgroup of 64.
// ai: Bindings as the WGSL's: 0 frames, 1 sel, 2 LISTS, 3 ARGS, 4 the gate's uniform (at[s] a slot's (workgroups of
// ai: pass 1, rows of pass 2, top version, 0), then the rows table four versions' words a vec4), 5 BCOUNTS, 6 V, 7
// ai: ITS, 8 REC. Built a batch size, as the WGSL is: B, the lists' heads and the zeroed lengths are literals.
import { VERSION, SAMPLERLESS, F16, STORAGE16, DIMS, CONVERT, LUMH, lumhHead, MAP } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { SLOTS } from "../plan/common.mjs";
import { ARGS_WORDS, CPW, LATTICE_MAX, ANCH_SIDE, ANCH_REC, ANCH_ERASE, listsHead, listsRows, listsRing, argsLdpc } from "../plan/back_transform.mjs";
import { NC } from "../plan/back_tiers.mjs";
import { stagesOf } from "../back/disc.mjs";
import { RING_COUNT } from "../plan/finder.mjs";

export function gateSource(B, { blkStride, counts, chunks } = {}) {
  if (!(counts > 1)) throw new Error(`gate: rows table of ${counts} versions`);
  if (!(chunks > 0)) throw new Error(`gate: ${chunks} chunks a version`);
  return VERSION + DIMS + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 2, std430) buffer Lists { uint lists[]; };
layout(set = 0, binding = 3, std430) buffer Args { uint args[]; };
layout(set = 0, binding = 4, std140) uniform Gate { uvec4 at[${SLOTS}]; uvec4 rows[${Math.ceil((counts * SLOTS) / 4)}]; uvec4 codes[${Math.ceil(counts / 4)}]; } G;
layout(set = 0, binding = 5, std430) buffer Counts { uint counts[]; };
layout(set = 0, binding = 6, std430) buffer Verdicts { uint V[]; };
layout(set = 0, binding = 7, std430) buffer Iterations { uint ITS[]; };
layout(set = 0, binding = 8, std430) buffer Records { uint rec[]; };
const uint B = ${B}u;
const uint LS = ${B + 1}u;
const uint SLOTS = ${SLOTS}u;
const uint HEAD = ${listsHead(B)}u;
const uint LDPC = ${argsLdpc(0)}u;
void main() {
  uint t = gl_LocalInvocationIndex;
  for (uint i = t; i < ${8 * B}u; i += 64u) { counts[i] = 0u; }
  for (uint i = t; i < ${B * blkStride}u; i += 64u) { V[i] = 0u; ITS[i] = 0u; }
  if (t == 0u) { rec[0] = 0u; }
  if (t != 0u) { return; }
  // ai: pass 2's workgroups a frame gather here as the frames are listed: the most rows any frame of the size needs.
  // ai: and tsoft's, the chunks of the most version any frame of the size takes.
  for (uint s = 0u; s < SLOTS; s++) { lists[s * LS] = 0u; args[s * ${ARGS_WORDS}u + 6u] = 0u; args[s * ${ARGS_WORDS}u + 9u] = 0u; }
  // ai: and each code's, the most blocks of it a listed frame's version takes, the frames listed counted in m.
  for (uint c = 0u; c < ${NC}u; c++) { args[LDPC + 3u * c] = 0u; }
  uint m = 0u;
  for (uint f = 0u; f < B; f++) {
    lists[HEAD + 2u * f] = frames[f].w;
    lists[HEAD + 2u * f + 1u] = frames[f].h;
    lists[HEAD + 2u * B + f] = 0u;
    lists[HEAD + 3u * B + f] = 0u;
    lists[HEAD + 4u * B + f] = 0u;
    if (frames[f].valid == 0u || sel[f].y < 2u) { continue; }
    uint s = sel[f].z;
    if (s >= SLOTS || G.at[s].x == 0u) { continue; }
    uint c = lists[s * LS];
    lists[s * LS + 1u + c] = f;
    lists[s * LS] = c + 1u;
    uint k = min(sel[f].y - 1u, G.at[s].z);
    uint i = k * SLOTS + s;
    uint r = G.rows[i >> 2u][i & 3u];
    lists[HEAD + 2u * B + f] = k;
    lists[HEAD + 3u * B + f] = r;
    lists[HEAD + 4u * B + f] = sel[f].x;
    args[s * ${ARGS_WORDS}u + 6u] = max(args[s * ${ARGS_WORDS}u + 6u], r);
    args[s * ${ARGS_WORDS}u + 9u] = max(args[s * ${ARGS_WORDS}u + 9u], ${chunks}u * k);
    lists[HEAD + 5u * B + m] = f;
    m = m + 1u;
    uint cc = G.codes[k >> 2u][k & 3u];
    for (uint c = 0u; c < ${NC}u; c++) { args[LDPC + 3u * c] = max(args[LDPC + 3u * c], (cc >> (8u * c)) & 255u); }
  }
  for (uint c = 0u; c < ${NC}u; c++) { args[LDPC + 3u * c + 1u] = m; args[LDPC + 3u * c + 2u] = 1u; }
  for (uint s = 0u; s < SLOTS; s++) {
    uint c = lists[s * LS];
    uint a = s * ${ARGS_WORDS}u;
    args[a] = G.at[s].x; args[a + 1u] = 1u; args[a + 2u] = c;
    args[a + 3u] = 1u; args[a + 4u] = 1u; args[a + 5u] = c;
    args[a + 7u] = 1u; args[a + 8u] = c;
    args[a + 10u] = 1u; args[a + 11u] = c;
  }
}
`;
}

// ai: The complex helpers and the butterflies (the WGSL's BUTTERFLIES): a function-scope array of eight points taken
// ai: inout where the WGSL takes a pointer to it.
const BUTTERFLIES = /* glsl */ `
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 mneg_i(vec2 a) { return vec2(a.y, -a.x); }
const float C8 = 0.70710678118654752;
const float S3 = 0.86602540378443864676;
void dft2(inout vec2 p[8], uint o) {
  vec2 a = p[o]; vec2 b = p[o + 1u];
  p[o] = a + b; p[o + 1u] = a - b;
}
void dft4(inout vec2 p[8], uint o) {
  vec2 p0 = p[o]; vec2 p1 = p[o + 1u]; vec2 p2 = p[o + 2u]; vec2 p3 = p[o + 3u];
  vec2 b0 = p0 + p2; vec2 b2 = p0 - p2; vec2 b1 = p1 + p3; vec2 b3 = mneg_i(p1 - p3);
  p[o] = b0 + b1; p[o + 1u] = b2 + b3; p[o + 2u] = b0 - b1; p[o + 3u] = b2 - b3;
}
void dft8(inout vec2 p[8]) {
  vec2 a0 = p[0] + p[4]; vec2 a4 = p[0] - p[4];
  vec2 a1 = p[1] + p[5]; vec2 d1 = p[1] - p[5];
  vec2 a2 = p[2] + p[6]; vec2 a6 = mneg_i(p[2] - p[6]);
  vec2 a3 = p[3] + p[7]; vec2 d3 = p[3] - p[7];
  vec2 a5 = vec2(C8 * (d1.x + d1.y), C8 * (d1.y - d1.x));
  vec2 a7 = vec2(C8 * (d3.y - d3.x), -C8 * (d3.x + d3.y));
  vec2 b0 = a0 + a2; vec2 b2 = a0 - a2; vec2 b1 = a1 + a3; vec2 b3 = mneg_i(a1 - a3);
  vec2 c0 = a4 + a6; vec2 c2 = a4 - a6; vec2 c1 = a5 + a7; vec2 c3 = mneg_i(a5 - a7);
  p[0] = b0 + b1; p[4] = b0 - b1; p[2] = b2 + b3; p[6] = b2 - b3;
  p[1] = c0 + c1; p[5] = c0 - c1; p[3] = c2 + c3; p[7] = c2 - c3;
}
`;

// ai: The exchange buffer's element type and the conversions in and out of it (the WGSL's store): f16 is f16vec2
// ai: (GL_EXT_shader_explicit_arithmetic_types_float16), computed in f32 as the WGSL's vec2<f16> is.
const store = (prec) => ({
  ty: prec === "f16" ? "f16vec2" : "vec2",
  ld: (name, i) => (prec === "f16" ? `vec2(${name}[${i}])` : `${name}[${i}]`),
  st: (name, i, x) => (prec === "f16" ? `${name}[${i}] = f16vec2(${x});` : `${name}[${i}] = ${x};`),
});

// ai: The stages after the first (the WGSL's laterStages, lanes the same): the decoder's sizes, 192 to 1024 (TRANSFORM_DOWN),
// ai: are 8 x 2^k or 8 x 3 x 2^k, so an odd stage is a 3 and the last (radix3Stage); the sender's 5s and earlier 3s
// ai: (oddStage, oddLastStage) stay the sender's WGSL.
function laterStages(n, prec, lanes = [["v", "buf"]]) {
  const { ld, st } = store(prec);
  const T = n / 8, stages = stagesOf(n);
  let Ns = stages[0], out = "";
  for (let s = 1; s < stages.length; s++) {
    if (stages[s] === 3 || stages[s] === 5) {
      if (stages[s] === 5 || s < stages.length - 1) throw new Error(`the GLSL transform of ${n}: an odd stage other than a last radix 3`);
      out += radix3Stage(n, prec, lanes);
      Ns *= 3; continue;
    }
    const R = stages[s], per = 8 / R, span = n / R, step = n / (Ns * R);
    out += `  // stage ${s + 1}: radix ${R}, Ns ${Ns}\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      for (const [v, buf] of lanes) for (let r = 0; r < R; r++) out += `  ${v}[${q * R + r}] = ${ld(buf, `${j} + ${r * span}u`)};\n`;
    }
    out += `  barrier();\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      out += `  {\n    uint j = ${j};\n    uint k = j & ${Ns - 1}u;\n`;
      out += `    vec2 w1 = tw[k * ${step}u];\n`;
      if (R >= 4) out += `    vec2 w2 = cmul(w1, w1);\n    vec2 w3 = cmul(w2, w1);\n`;
      if (R === 8) out += `    vec2 w4 = cmul(w2, w2);\n    vec2 w5 = cmul(w4, w1);\n    vec2 w6 = cmul(w3, w3);\n    vec2 w7 = cmul(w4, w3);\n`;
      out += `    uint d = (j - k) * ${R}u + k;\n`;
      for (const [v, buf] of lanes) {
        for (let r = 1; r < R; r++) out += `    ${v}[${q * R + r}] = cmul(${v}[${q * R + r}], w${r});\n`;
        out += R === 8 ? `    dft8(${v});\n` : R === 4 ? `    dft4(${v}, ${q * R}u);\n` : `    dft2(${v}, ${q * R}u);\n`;
        for (let r = 0; r < R; r++) out += `    ${st(buf, `d + ${r * Ns}u`, `${v}[${q * R + r}]`)}\n`;
      }
      out += `  }\n`;
    }
    out += `  barrier();\n`;
    Ns *= R;
  }
  return out;
}

// ai: The last stage when n has a factor of 3, in place (the WGSL's radix3Stage).
function radix3Stage(n, prec, lanes) {
  const { ld, st } = store(prec);
  const T = n / 8, Ns = n / 3;
  let out = `  // stage: radix 3, Ns ${Ns}, in place\n  for (uint j = tid; j < ${Ns}u; j += ${T}u) {\n`;
  out += `    vec2 w1 = tw[j];\n    vec2 w2 = tw[2u * j];\n`;
  for (const [, buf] of lanes) {
    out += `    {\n      vec2 a = ${ld(buf, "j")};\n      vec2 b = cmul(${ld(buf, `j + ${Ns}u`)}, w1);\n      vec2 c = cmul(${ld(buf, `j + ${2 * Ns}u`)}, w2);\n`;
    out += `      vec2 t1 = b + c;\n      vec2 t2 = mneg_i(b - c) * S3;\n      vec2 m = a - 0.5 * t1;\n`;
    out += `      ${st(buf, "j", "a + t1")}\n      ${st(buf, `j + ${Ns}u`, "m + t2")}\n      ${st(buf, `j + ${2 * Ns}u`, "m - t2")}\n    }\n`;
  }
  return out + `  }\n  barrier();\n`;
}

// ai: The first stage (Ns = 1, no twiddles), the registers loaded by the caller.
function firstStage(prec, lanes = [["v", "buf"]]) {
  const { st } = store(prec);
  let out = `  // stage 1: radix 8, Ns 1\n`;
  for (const [v, buf] of lanes) {
    out += `  dft8(${v});\n`;
    for (let r = 0; r < 8; r++) out += `  ${st(buf, `tid * 8u + ${r}u`, `${v}[${r}]`)}\n`;
  }
  out += `  barrier();\n`;
  return out;
}

const V8 = "vec2[8](vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0), vec2(0.0))";

// ai: The per size uniform pass 1, reduce and pass 2 share (the WGSL's Params, 48 B, std140 at the same offsets).
const PARAMS = (binding) => /* glsl */ `
layout(set = 0, binding = ${binding}, std140) uniform Params { uint p0; uint yStride; uint partStride; uint size; uint p4; uint p5; float m2; float sx2; float sq2; uint sStride; uint p10; uint p11; } P;
`;

// ai: PIC (the WGSL's PICTURE): per ring (g0, step, nodes a side, first lattice coordinate), then every ring's lattice.
const PICTURE = /* glsl */ `
layout(set = 0, binding = 9, std140) uniform Picture { vec4 ring[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } PIC;
float latAt(uint i) { return PIC.lat[i >> 2u][i & 3u]; }
`;
const pictureHead = (B) => `  int fw = int(lists[${listsHead(B)}u + 2u * f]);\n  int fh = int(lists[${listsHead(B)}u + 2u * f + 1u]);\n  float M[16] = maps[f];\n  vec4 pr = PIC.ring[lists[${listsRing(B)}u + f]];\n  float g0 = pr.x;\n  float gstep = pr.y;\n  uint nn = f2u(pr.z);\n  uint first = f2u(pr.w);\n  Corners cn = corners(f, nn);\n  Lens lz = lensOf(f);\n`;

// ai: Pass 1's anchors (the WGSL's ANCH_P1, its workgroup memory and arithmetic; the functions pass 1 calls).
const ANCH_ROWS = ANCH_SIDE + 4;
const ANCH_P1 = /* glsl */ `
layout(set = 0, binding = 10, std430) readonly buffer ARec { vec4 arec[]; };
shared vec2 anE[${4 * ANCH_ROWS}];
shared float anK[${ANCH_SIDE + 2}];
shared int anLo[${ANCH_SIDE}];
shared int anHi[${ANCH_SIDE}];
shared uint anCol[4];
struct AnchHerm { int b; vec4 w; uint i; };
float aknot(uint ab, uint i) { return arec[ab + 1u + (i >> 2u)][i & 3u]; }
AnchHerm anchHerm(float x) {
  if (!(x >= anK[0] && x <= anK[8])) { return AnchHerm(0, vec4(0.0), 9u); }
  uint i = 0u;
  for (uint j = 1u; j < 8u; j++) { i += (x >= anK[j]) ? 1u : 0u; }
  float ta = anK[i];
  float tb = anK[i + 1u];
  float hc = tb - ta;
  float s = (x - ta) / hc;
  float s2 = s * s;
  float s3 = s2 * s;
  float ha = (s3 - 2.0 * s2 + s) * hc;
  float hb = (s3 - s2) * hc;
  vec4 w = vec4(0.0, 2.0 * s3 - 3.0 * s2 + 1.0, -2.0 * s3 + 3.0 * s2, 0.0);
  if (i > 0u) { float d = tb - anK[i - 1u]; w.x -= ha / d; w.z += ha / d; } else { w.y -= ha / hc; w.z += ha / hc; }
  if (i < 7u) { float d = anK[i + 2u] - ta; w.y -= hb / d; w.w += hb / d; } else { w.y -= hb / hc; w.z += hb / hc; }
  return AnchHerm(int(i) - 1, w, i);
}
bool anchIn(int x) {
  bool hit = false;
  for (uint i = 0u; i < ${ANCH_SIDE}u; i++) { hit = hit || (x >= anLo[i] && x <= anHi[i]); }
  return hit;
}
vec2 aval(uint ab, int c, uint r) {
  if (c < 1 || c > ${ANCH_SIDE}) { return vec2(0.0); }
  uint q = ${ANCH_SIDE}u * (r - 1u) + uint(c - 1);
  vec4 v = arec[ab + 4u + (q >> 1u)];
  return ((q & 1u) == 1u) ? v.zw : v.xy;
}
vec2 anchE(uint ab, float x, uint r) {
  AnchHerm h = anchHerm(x);
  return h.w.x * aval(ab, h.b, r) + h.w.y * aval(ab, h.b + 1, r) + h.w.z * aval(ab, h.b + 2, r) + h.w.w * aval(ab, h.b + 3, r);
}
vec2 anchMesh(AnchHerm h, uint col) {
  uint o = ${ANCH_ROWS}u * col + uint(h.b + 1);
  return h.w.x * anE[o] + h.w.y * anE[o + 1u] + h.w.z * anE[o + 2u] + h.w.w * anE[o + 3u];
}
`;
const anchHead = () => `  uint ab = f * ${ANCH_REC}u;\n  vec4 ahd = arec[ab];\n  bool aon = ahd.x > 0.5;\n  float amean = ahd.y;\n`;
const anchErase = (n, picture) => `      if (tid < ${ANCH_SIDE}u) {
        vec4 ag = arec[ab + 3u];
        int ia = f2i(ag.y);
        int img = f2i(ag.w);
        int im0 = f2i(aknot(ab, tid + 1u)) - ia / 2;
        int ed = ${128 * picture} * ia;
        int el = ${n} * (${2 * picture} * (im0 - img) - ${64 * (2 * ANCH_ERASE + 2)} * ia);
        int eu = ${n} * (${2 * picture} * (im0 - img + ia) + ${128 * ANCH_ERASE} * ia);
        anLo[tid] = max((el + ed - 1) / ed, 0);
        anHi[tid] = min(eu / ed, ${n - 1});
      }
`;

// ai: Pass 1 (the WGSL's pass1Source): n the transform's size, prec, B, picture the picture's own size. Every workgroup
// ai: variable is written before the barrier that precedes its reads (the anchors' only where the record is on, and
// ai: read only there), so none is zeroed.
export function pass1Source({ n, prec, B, picture = n }) {
  const cpw = CPW, { ty, ld } = store(prec);
  const T = n / 8, quads = cpw / 4, f16 = prec === "f16";
  const lanes = [["v0", "buf0"], ["v1", "buf1"]];
  let body = "";
  for (let q = 0; q < quads; q++) {
    body += `  {\n    uint xq = x0 + ${4 * q}u;\n`;
    for (let k = 0; k < 4; k++) body += `    float mx${k} = g0 + float(xq + ${k}u) * gstep;\n    float u${k} = coonsT(mx${k}, first, nn);\n    vec2 tp${k} = curve(f, 0u, mx${k}, first, nn);\n    vec2 bt${k} = curve(f, 2u, mx${k}, first, nn);\n`;
    body += `    bvec4 aix = bvec4(false);\n    if (aon) {\n`;
    body += `      if (tid < ${ANCH_SIDE + 2}u) { anK[tid] = aknot(ab, tid); }\n` + anchErase(n, picture) + `      barrier();\n      if (tid < 4u) { anCol[tid] = anchIn(int(xq + tid)) ? 1u : 0u; }\n`;
    body += `      for (uint ai = tid; ai < ${4 * ANCH_ROWS}u; ai += T) {\n        uint arow = ai % ${ANCH_ROWS}u;\n        vec2 aev = vec2(0.0);\n`;
    body += `        if (arow >= 2u && arow <= ${ANCH_SIDE + 1}u) { aev = anchE(ab, g0 + float(xq + ai / ${ANCH_ROWS}u) * gstep, arow - 1u); }\n        anE[ai] = aev;\n      }\n      barrier();\n`;
    body += `      aix = bvec4(${[0, 1, 2, 3].map((k) => `anCol[${k}] != 0u`).join(", ")});\n    }\n`;
    body += `    for (uint j = tid; j <= T; j += T) {\n      float my = g0 + float(8u * j) * gstep;\n      float v = coonsT(my, first, nn);\n      vec2 lf = curve(f, 3u, my, first, nn);\n      vec2 rg = curve(f, 1u, my, first, nn);\n`;
    body += `      vec2 m0 = vec2(0.0);\n      vec2 m3 = vec2(0.0);\n      if (aon) { AnchHerm ah = anchHerm(my); m0 = anchMesh(ah, 0u); m3 = anchMesh(ah, 3u); }\n`;
    body += `      vec2 p0 = toImage(M, vec2(mx0, my));\n      vec2 p3 = toImage(M, vec2(mx3, my));\n`;
    body += `      nodeP[j] = vec4(p0 + lensAt(lz, p0) + coonsMix(u0, v, tp0, bt0, lf, rg, cn) + m0, p3 + lensAt(lz, p3) + coonsMix(u3, v, tp3, bt3, lf, rg, cn) + m3);\n    }\n    barrier();\n`;
    body += `    vec4 z[8] = vec4[8](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));\n`;
    body += `    vec4 s1 = vec4(0.0);\n    vec4 sc = vec4(0.0);\n    vec4 sq = vec4(0.0);\n`;
    for (let r = 0; r < 8; r++) {
      body += `    {\n      uint y = tid + ${r * T}u;\n`;
      body += `      vec4 nA = nodeP[y / 8u];\n      vec4 nB = nodeP[y / 8u + 1u];\n      float fy = float(y % 8u) * 0.125;\n      vec2 pL = mix(nA.xy, nB.xy, fy);\n      vec2 pR = mix(nA.zw, nB.zw, fy);\n`;
      body += `      vec4 zz = vec4(amean);\n      if (aon) {\n`;
      body += `        zz = vec4(bilinH(f, pL, hhi, hinv), bilinH(f, mix(pL, pR, 1.0 / 3.0), hhi, hinv), bilinH(f, mix(pL, pR, 2.0 / 3.0), hhi, hinv), bilinH(f, pR, hhi, hinv));\n`;
      body += `        bool ay = anchIn(int(y));\n        zz = mix(zz, vec4(amean), bvec4(aix.x && ay, aix.y && ay, aix.z && ay, aix.w && ay));\n      }\n`;
      body += `      z[${r}] = zz;\n`;
      body += `      float c = float(2 * int(y) + 1 - int(N)) / float(N);\n      s1 += zz;\n      sc += zz * c;\n      sq += zz * (c * c - P.m2);\n    }\n`;
    }
    body += `    vec4 S1 = vec4(0.0);\n    vec4 SC = vec4(0.0);\n    vec4 SQ = vec4(0.0);\n`;
    [["s1", "S1"], ["sc", "SC"], ["sq", "SQ"]].forEach(([part, sum], k) => {
      if (k > 0) body += `    barrier();\n`;
      body += `    red[tid] = ${part};\n    barrier();\n`;
      body += `    if (tid < 8u) {\n      vec4 a = vec4(0.0);\n      for (uint i = tid; i < T; i += 8u) { a += red[i]; }\n      red[tid] = a;\n    }\n    barrier();\n`;
      body += `    for (uint i = 0u; i < 8u; i++) { ${sum} += red[i]; }\n`;
    });
    body += `    vec4 mean = S1 / float(N);\n`;
    body += `    if (tid == 0u) {\n`;
    for (let k = 0; k < 4; k++) body += `      part[f * P.partStride + xq + ${k}u] = vec4(S1[${k}], SC[${k}], SQ[${k}], mean[${k}]);\n`;
    body += `    }\n`;
    for (let r = 0; r < 8; r++) body += `    v0[${r}] = z[${r}].xy - mean.xy;\n    v1[${r}] = z[${r}].zw - mean.zw;\n`;
    body += firstStage(prec, lanes) + laterStages(n, prec, lanes);
    body += `    for (uint vv = 1u + tid; vv < rowsF; vv += T) {\n`;
    body += `      vec2 z0 = ${ld("buf0", "vv")};\n      vec2 m0 = ${ld("buf0", "N - vv")};\n      vec2 z1 = ${ld("buf1", "vv")};\n      vec2 m1 = ${ld("buf1", "N - vv")};\n`;
    body += `      vec2 a0 = vec2(z0.x + m0.x, z0.y - m0.y) * (0.5 / float(N));\n      vec2 b0 = vec2(z0.y + m0.y, m0.x - z0.x) * (0.5 / float(N));\n`;
    body += `      vec2 a1 = vec2(z1.x + m1.x, z1.y - m1.y) * (0.5 / float(N));\n      vec2 b1 = vec2(z1.y + m1.y, m1.x - z1.x) * (0.5 / float(N));\n`;
    body += `      uint e = f * P.yStride + (vv - 1u) * N + xq;\n`;
    body += f16
      ? `      yb[e >> 2u] = uvec4(packHalf2x16(a0), packHalf2x16(b0), packHalf2x16(a1), packHalf2x16(b1));\n`
      : `      yb[e >> 1u] = vec4(a0, b0);\n      yb[(e >> 1u) + 1u] = vec4(a1, b1);\n`;
    body += `    }\n    barrier();\n  }\n`;
  }
  return VERSION + SAMPLERLESS + (f16 ? F16 : "") + PARAMS(6) + /* glsl */ `
layout(local_size_x = ${T}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Lists { uint lists[]; };
layout(set = 0, binding = 2, std430) readonly buffer Twiddles { vec2 tw[]; };
layout(set = 0, binding = 3, std430) buffer Part { vec4 part[]; };
layout(set = 0, binding = 4, std430) buffer Y { ${f16 ? "uvec4" : "vec4"} yb[]; };
layout(set = 0, binding = 5, std430) buffer Counts { uint counts[]; };
layout(set = 0, binding = 7, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 8, std430) readonly buffer Resid { vec4 resid[]; };
const uint N = ${n}u;
const uint T = ${T}u;
const uint LS = ${B + 1}u;
const uint ROWS_AT = ${listsRows(B)}u;
shared ${ty} buf0[N];
shared ${ty} buf1[N];
shared vec4 red[T];
shared vec4 nodeP[${T + 1}];
` + PICTURE + CONVERT + MAP + LUMH(11) + COONS + ANCH_P1 + BUTTERFLIES + /* glsl */ `
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint tid = gl_LocalInvocationIndex;
  uint f = lists[P.size * LS + 1u + wg.z];
  uint rowsF = lists[ROWS_AT + f];
  uint x0 = wg.x * ${cpw}u;
  vec2 v0[8] = ${V8};
  vec2 v1[8] = ${V8};
` + pictureHead(B) + lumhHead("fw", "fh") + anchHead() + body + /* glsl */ `
  if (tid == 0u) { atomicAdd(counts[f * 8u + 0u], ${cpw}u); }
}
`;
}

// ai: Reduce (the WGSL's reduceSource): the five surface coefficients from the column partials. red is written
// ai: whole before its first read.
export function reduceSource({ n, B }) {
  return VERSION + PARAMS(3) + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Lists { uint lists[]; };
layout(set = 0, binding = 1, std430) readonly buffer Part { vec4 part[]; };
layout(set = 0, binding = 2, std430) buffer Coef { vec4 coef[]; };
const uint N = ${n}u;
const uint LS = ${B + 1}u;
shared vec4 red[512];
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint tid = gl_LocalInvocationIndex;
  uint f = lists[P.size * LS + 1u + wg.z];
  vec4 s1 = vec4(0.0);
  float s2 = 0.0;
  for (uint x = tid; x < N; x += 256u) {
    vec4 p = part[f * P.partStride + x];
    float c = float(2 * int(x) + 1 - int(N)) / float(N);
    float q = c * c - P.m2;
    s1 += vec4(c * p.x, p.y, c * p.y, q * p.x);
    s2 += p.z;
  }
  red[tid] = s1;
  red[256u + tid] = vec4(s2, 0.0, 0.0, 0.0);
  barrier();
  if (tid < 32u) {
    vec4 a = vec4(0.0);
    vec4 b = vec4(0.0);
    for (uint i = tid; i < 256u; i += 32u) { a += red[i]; b += red[256u + i]; }
    red[tid] = a;
    red[256u + tid] = b;
  }
  barrier();
  if (tid == 0u) {
    vec4 a = vec4(0.0);
    vec4 b = vec4(0.0);
    for (uint i = 0u; i < 32u; i++) { a += red[i]; b += red[256u + i]; }
    float nf = float(N);
    coef[f * 2u] = vec4(a.x / (nf * P.sx2), a.y / (nf * P.sx2), a.z / (P.sx2 * P.sx2), a.w / (nf * P.sq2));
    coef[f * 2u + 1u] = vec4(b.x / (nf * P.sq2), 0.0, 0.0, 0.0);
  }
}
`;
}

// ai: Pass 2 (the WGSL's pass2Source): row v of one frame along x, its compact disc row stored, the detrend taken off.
// ai: f16 reads Y and writes S as f16vec2 (GL_EXT_shader_16bit_storage, storage buffers only).
export function pass2Source({ n, prec, B }) {
  const { ty, ld } = store(prec);
  const T = n / 8, f16 = prec === "f16";
  return VERSION + (f16 ? F16 + STORAGE16 : "") + PARAMS(9) + /* glsl */ `
layout(local_size_x = ${T}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Lists { uint lists[]; };
layout(set = 0, binding = 1, std430) readonly buffer Twiddles { vec2 tw[]; };
layout(set = 0, binding = 2, std430) readonly buffer Part { vec4 part[]; };
layout(set = 0, binding = 3, std430) readonly buffer Y { ${ty} yb[]; };
layout(set = 0, binding = 4, std430) readonly buffer Coef { vec4 coef[]; };
layout(set = 0, binding = 5, std430) readonly buffer X12 { vec4 x12[]; };
layout(set = 0, binding = 6, std140) uniform Rows { uvec4 r[512]; } rows;
layout(set = 0, binding = 7, std430) buffer S { ${ty} sb[]; };
layout(set = 0, binding = 8, std430) buffer Counts { uint counts[]; };
const uint N = ${n}u;
const uint T = ${T}u;
const uint LS = ${B + 1}u;
const uint ROWS_AT = ${listsRows(B)}u;
shared ${ty} buf[N];
` + BUTTERFLIES + /* glsl */ `
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint tid = gl_LocalInvocationIndex;
  uint f = lists[P.size * LS + 1u + wg.z];
  uint vv = wg.x;
  if (vv >= lists[ROWS_AT + f]) { return; }
  vec2 v[8] = ${V8};
  if (vv == 0u) {
    for (uint r = 0u; r < 8u; r++) { v[r] = vec2(part[f * P.partStride + tid + r * T].w, 0.0); }
  } else {
    uint ybase = f * P.yStride + (vv - 1u) * N;
    for (uint r = 0u; r < 8u; r++) { v[r] = ${ld("yb", "ybase + tid + r * T")}; }
  }
` + firstStage(prec) + laterStages(n, prec) + /* glsl */ `
  uvec4 row = rows.r[vv];
  vec4 c0 = coef[f * 2u];
  vec4 c1 = coef[f * 2u + 1u];
  vec4 xv = x12[vv];
  float nf = float(N);
  uint ustart = (vv == 0u) ? 1u : 0u;
  for (uint e = tid; e < row.y; e += T) {
    int u = 0;
    if (e < row.z) { u = int(e + ustart); } else { u = int(e) - int(row.z) - int(row.w); }
    uint m = uint(u + int(N)) % N;
    vec4 xu = x12[m];
    vec2 d = c0.z * cmul(xu.xy, xv.xy);
    if (vv == 0u) { d += nf * (c0.x * xu.xy + c0.w * xu.zw); }
    if (u == 0) { d += nf * (c0.y * xv.xy + c1.x * xv.zw); }
    vec2 s = (${ld("buf", "m")} - d / nf) / nf;
    ${f16 ? "sb[f * P.sStride + row.x + e] = f16vec2(s);" : "sb[f * P.sStride + row.x + e] = s;"}
  }
  if (tid == 0u) {
    atomicAdd(counts[f * 8u + 1u], 1u);
    atomicAdd(counts[f * 8u + 2u], row.y);
  }
}
`;
}
