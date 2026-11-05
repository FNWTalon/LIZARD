// The back half's rate-profile stage in GLSL: the WGSL back_tiers.mjs (archived) talignSource, tsoftSource, tblocksSource and
// tldpcSource statement for statement (that file's header and comments say what each kernel does; TAB's layout, the
// shapes and the forms are its tabLayout, tierShape and LDPC_FORMS). The native tree takes these; the WGSL is their
// twin for a WebGPU device.
// ai: As the WGSL: the same bindings (set 0, binding k), storage arrays std430 and uniform blocks std140 (vec4 arrays and
// ai: scalars, WGSL's offsets), the same arithmetic in the same order. Where GLSL differs from WGSL:
// ai:   WGSL zeroes workgroup and function variables; here every one is written before it is read (the check records
// ai:     Rr zeroed by the kernel, as the WGSL's `var Rr` is);
// ai:   workgroupUniformLoad is a barrier, the load and a barrier, as naga writes it (common.mjs uniformLoad);
// ai:   workgroup atomics at workgroup scope and relaxed (common.mjs wgAdd and the rest), storage buffer atomics at
// ai:     device scope, as naga writes both;
// ai:   round is WGSL's round half to even (roundEven); select(f, t, c) is c ? t : f;
// ai:   rowUpdate, which takes the check records by pointer in WGSL, is written into main at its call (rowBody): GLSL
// ai:     has no pointer parameters, and an inout array is copied in and out;
// ai:   the learned stop's sums are `precise`: no contraction into fused multiply-adds, so the kernel keeps the float32
// ai:     order of stop/rule.mjs scoreF32 and the trainer's (the WGSL leaves contraction to the driver);
// ai:   a check's slot loops are unrolled ([[unroll]], GL_EXT_control_flow_attributes), as the drivers unroll naga's.
import { VERSION, F16, STORAGE16, INT16, ATOMICS, UNROLL, uniformLoad, wgAdd, wgLoad, wgStore, wgXor } from "./common.mjs";
import { TIER_BLOCKS, CHUNKS, KNOWN, NC, TTW, ALIGN_CHI, tierUniformWords } from "../plan/back_tiers.mjs";
import { SLOTS } from "../plan/common.mjs";
import { CRC_CHUNKS } from "../wgsl/back_ldpc.mjs";

const TWO_PI = (2 * Math.PI).toFixed(9);
// ai: A picture slot's uniform (back/soft.mjs build), the WGSL's Params: nine u32, 48 B.
const PARAMS = (binding) => `layout(set = 0, binding = ${binding}, std140) uniform ParamsU { uint s; uint uvOff; uint sStride; uint lStride; uint subchMax; uint blocksMax; uint B; uint ucOff; uint n; } P;`;
// ai: a NaN made at run time, as the WGSL's (P a uniform)
const NAN = "uintBitsToFloat(0x7fc00000u | (P.n & 0u))";

// ai: TAB's constants and readers (the WGSL's tables()); TAB is the buffer named TAB. codes: [{ n, k, slots, tail, bar }]:
// ai: n the sent bits, slots a block's (its sent bits, then its tail of pilots), tail the pilots' coefficients.
function tables(L, codes) {
  const pick = (f) => codes.slice(0, -1).reduceRight((e, x, c) => `(c == ${c}u ? ${f(x)} : ${e})`, f(codes[codes.length - 1]));
  const barBits = (x) => { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; };
  return /* glsl */ `
const uint TT_AT = ${L.TT}u;
const uint WHITE_AT = ${L.WHITE}u;
uint nOf(uint c) { return ${pick((x) => `${x.n}u`)}; }
uint slotsOf(uint c) { return ${pick((x) => `${x.slots}u`)}; }
uint tailOf(uint c) { return ${pick((x) => `${x.tail}u`)}; }
float barOf(uint c) { return uintBitsToFloat(${pick((x) => `${barBits(x.bar)}u`)}); }
const uint NCODES = ${NC}u;
uint blocksOf(uint v) { return TAB[TT_AT + ${TTW}u * v]; }
uvec4 ttOf(uint v, uint c) { uint o = TT_AT + ${TTW}u * v + 4u + 4u * c; return uvec4(TAB[o], TAB[o + 1u], TAB[o + 2u], TAB[o + 3u]); }
uint whiteWord(uint slot) {
  uint w = WHITE_AT + (slot >> 5u);
  uint s = slot & 31u;
  if (s == 0u) { return TAB[w]; }
  return (TAB[w] >> s) | (TAB[w + 1u] << (32u - s));
}
uvec2 blockAt(uint v, uint b) {
  for (uint c = 0u; c < NCODES; c++) { uvec4 t = ttOf(v, c); if (b >= t.x && b < t.x + t.z) { return uvec2(c, b - t.x); } }
  return uvec2(NCODES, 0u);
}
// ai: The pilots' part in sub-channel j (its slots 640 j to 640 j + 640): the coefficients (from, to) within it of the
// ai: one tail that meets it, from == to where none does. A tail is a block's last 2 tail slots; blocks are 4496 slots
// ai: or more, so at most one tail meets a sub-channel, and only the block holding the sub-channel's first slot (or,
// ai: before a code's first block, that block) can own it.
uvec2 tailIn(uint v, uint j) {
  uint s = 640u * j;
  for (uint c = 0u; c < NCODES; c++) {
    uvec4 t = ttOf(v, c);
    uint sl = slotsOf(c);
    uint tl = 2u * tailOf(c);
    if (t.z == 0u || tl == 0u || s + 640u <= t.y || s >= t.y + t.z * sl) { continue; }
    uint q = s > t.y ? (s - t.y) / sl : 0u;
    uint e = t.y + (q + 1u) * sl;
    uint lo = max(e - tl, s);
    uint hi = min(e, s + 640u);
    if (lo < hi) { return uvec2((lo - s) / 2u, (hi - s) / 2u); }
  }
  return uvec2(0u);
}
`;
}
const versionOf = (lists, B, f) => `${lists}[${SLOTS}u * (${B} + 1u) + 2u * ${B} + ${f}]`;

const TAIL_MAX = 16;
export const talignSource = ({ f16 = false, layout, codes }) => {
  const tailMax = Math.max(...codes.map((x) => x.tail));
  if (!(tailMax > 0 && tailMax <= TAIL_MAX)) throw new Error(`a tail of ${tailMax} coefficients: talign takes 1 to ${TAIL_MAX}`);
  const I = [...Array(tailMax).keys()];
  let gather = "";
  for (const i of I) gather += `    uint q${i} = UV[P.uvOff + at + min(${i}u, cnt - 1u)];\n`;
  for (const i of I) gather += `    uint w${i} = UV[P.ucOff + at + min(${i}u, cnt - 1u)];\n`;
  for (const i of I) {
    gather += `    vec2 y${i} = vec2(S[f * P.sStride + q${i}]);\n`;
    gather += `    vec2 x${i} = vec2(((tw >> ${2 * i}u) & 1u) != 0u ? -1.0 : 1.0, ((tw >> ${2 * i + 1}u) & 1u) != 0u ? -1.0 : 1.0);\n`;
    gather += `    z${i} = ${i}u < cnt ? vec2(y${i}.x * x${i}.x + y${i}.y * x${i}.y, y${i}.y * x${i}.x - y${i}.x * x${i}.y) : vec2(0.0);\n`;
    gather += `    u${i} = w${i};\n`;
  }
  const fit = (z) => {
    let o = `    float sum = 0.0;\n`;
    for (const i of I) o += `    vec2 e${i} = ${z(i)};\n`;
    for (const i of I) o += `    sum += e${i}.x;\n`;
    o += `    float sg = sum < 0.0 ? -1.0 : 1.0;\n`;
    for (const i of I) o += `    { float zr = sg * e${i}.x; float zi = sg * e${i}.y; vec2 uv = uvOf(u${i}); float cu = kk * zr * uv.x; float cv = kk * zr * uv.y; s0 += vec4(cu * cu, cu * cv, cv * cv, cu * zi); s1 += vec4(cv * zi, zi * zi, 0.0, 0.0); }\n`;
    return o + `    s1.z = float(cnt);\n`;
  };
  const decls = I.map((i) => `  vec2 z${i} = vec2(0.0);\n  uint u${i} = 0u;\n`).join("");
  const reduce = (then) => `  acc0[t] = s0;
  acc1[t] = s1;
  barrier();
  if (t < 16u) {
    vec4 a0 = vec4(0.0);
    vec4 a1 = vec4(0.0);
    for (uint i = t; i < ${TIER_BLOCKS}u; i += 16u) { a0 += acc0[i]; a1 += acc1[i]; }
    acc0[t] = a0;
    acc1[t] = a1;
  }
  barrier();
  if (t == 0u) {
    vec4 a0 = vec4(0.0);
    vec4 a1 = vec4(0.0);
    for (uint i = 0u; i < 16u; i++) { a0 += acc0[i]; a1 += acc1[i]; }
${then}  }
  barrier();
`;
  return VERSION + (f16 ? STORAGE16 + F16 : "") + /* glsl */ `
layout(local_size_x = ${TIER_BLOCKS}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer SB { ${f16 ? "f16vec2" : "vec2"} S[]; };
layout(set = 0, binding = 1, std430) readonly buffer UVB { uint UV[]; };
layout(set = 0, binding = 2, std430) readonly buffer ListsB { uint LISTS[]; };
layout(set = 0, binding = 3, std430) buffer PilotB { float PILOT[]; };
layout(set = 0, binding = 4, std430) readonly buffer TabB { uint TAB[]; };
${PARAMS(5)}
${tables(layout, codes)}
shared vec4 acc0[${TIER_BLOCKS}];
shared vec4 acc1[${TIER_BLOCKS}];
shared vec2 cur;
shared vec4 last0;
shared vec4 last1;
vec2 solved(vec4 a0, vec4 a1) {
  float det = a0.x * a0.z - a0.y * a0.y;
  if (a1.z >= 8.0 && det > 0.0) { return vec2((a0.w * a0.z - a1.x * a0.y) / det, (a0.x * a1.x - a0.y * a0.w) / det); }
  return vec2(0.0);
}
vec2 uvOf(uint w) { return vec2(float(int(w) >> 16u), float(w & 0xffffu)); }
vec2 turnedBy(vec2 y, float th) {
  float cs = cos(th);
  float sn = sin(th);
  return vec2(y.x * cs + y.y * sn, y.y * cs - y.x * sn);
}

void main() {
  uvec3 wg = gl_WorkGroupID;
  uint t = gl_LocalInvocationIndex;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint v = ${versionOf("LISTS", "P.B", "f")};
  uint blocks = min(blocksOf(v), ${TIER_BLOCKS}u);
  float kk = ${TWO_PI} / float(P.n);
  uint cnt = 0u;
  uint at = 0u;
  uint tw = 0u;
  if (t < blocks) {
    uvec2 bc = blockAt(v, t);
    if (bc.x < NCODES) {
      cnt = tailOf(bc.x);
      // ai: the tail's first slot: the block's first (its tier's first slot and the blocks before it) and its sent bits
      uint ts = ttOf(v, bc.x).y + bc.y * slotsOf(bc.x) + nOf(bc.x);
      at = ts / 2u;
      tw = whiteWord(ts);
    }
  }
${decls}  if (cnt > 0u) {
${gather}  }
  vec4 s0 = vec4(0.0);
  vec4 s1 = vec4(0.0);
  if (cnt > 0u) {
${fit((i) => `z${i}`)}  }
${reduce(`    cur = solved(a0, a1);\n`)}  vec2 d = cur;
  s0 = vec4(0.0);
  s1 = vec4(0.0);
  if (cnt > 0u) {
${fit((i) => `turnedBy(z${i}, kk * dot(uvOf(u${i}), d))`)}  }
${reduce(`    last0 = a0;\n    last1 = a1;\n`)}  if (t == 0u) {
    vec2 outv = vec2(0.0);
    vec4 a0 = last0;
    vec4 a1 = last1;
    float det = a0.x * a0.z - a0.y * a0.y;
    if (a1.z >= 8.0 && det > 0.0) {
      vec2 e = vec2((a0.w * a0.z - a1.x * a0.y) / det, (a0.x * a1.x - a0.y * a0.w) / det);
      float s2 = (a1.y - e.x * a0.w - e.y * a1.x) / (a1.z - 2.0);
      vec2 dd = d + e;
      if (s2 > 0.0 && (dd.x * dd.x * a0.x + 2.0 * dd.x * dd.y * a0.y + dd.y * dd.y * a0.z) > ${ALIGN_CHI} * s2) { outv = dd; }
    }
    PILOT[P.B * P.blocksMax + 2u * f] = outv.x;
    PILOT[P.B * P.blocksMax + 2u * f + 1u] = outv.y;
  }
}
`;
};

const PAIRS = 5;
export const tsoftSource = ({ f16 = false, layout, codes }) => {
  const tailMax = Math.max(...codes.map((x) => x.tail));
  if (!(tailMax >= 0 && tailMax <= 320)) throw new Error(`a tail of ${tailMax} coefficients`);
  // ai: a tail's part may fall anywhere in a sub-channel (the blocks are packed by slots): every pair is tested
  const ks = [...Array(PAIRS).keys()];
  const at = (k) => (k ? ` + ${32 * k}u` : "");
  let body = "";
  for (const k of ks) body += `  uvec2 i${k} = UV[ub${at(k)}];\n`;
  for (const k of ks) body += `  vec2 y${2 * k} = vec2(S[sBase + i${k}.x]);\n  vec2 y${2 * k + 1} = vec2(S[sBase + i${k}.y]);\n`;
  body += `  if (turn) {\n    uint cb = (P.ucOff >> 1u) + 160u * j + l;\n`;
  for (const k of ks) body += `    uvec2 w${k} = UV[cb${at(k)}];\n`;
  for (const k of ks) for (const h of [0, 1]) {
    const c = 2 * k + h, w = `w${k}.${h ? "y" : "x"}`;
    body += `    y${c} = turned(y${c}, kk * (float(int(${w}) >> 16u) * sh.x + float(${w} & 0xffffu) * sh.y));\n`;
  }
  body += `  }\n`;
  body += `  uint wb = WHITE_AT + 20u * j + (l >> 3u);\n  uint ws = 4u * (l & 7u);\n`;
  for (const k of ks) {
    body += `  uint n${k} = TAB[wb${k ? ` + ${4 * k}u` : ""}] >> ws;\n`;
    for (const h of [0, 1]) { const c = 2 * k + h; body += `  y${c} = vec2((n${k} & ${1 << (2 * h)}u) != 0u ? -y${c}.x : y${c}.x, (n${k} & ${2 << (2 * h)}u) != 0u ? -y${c}.y : y${c}.y);\n`; }
  }
  body += `  float m2 = 0.0;\n  float m4 = 0.0;\n`;
  for (const k of ks) for (const h of [0, 1]) { const c = 2 * k + h; body += `  { float e = dot(y${c}, y${c}); m2 += e; m4 += e * e; }\n`; }
  body += `  float cs = 0.0;\n`;
  for (const k of ks) for (const h of [0, 1]) { const c = 2 * k + h; body += `  { uint ci = 2u * l + ${64 * k + h}u; cs += ci >= tp.x && ci < tp.y ? y${c}.x + y${c}.y : 0.0; }\n`; }
  let store = "";
  for (const k of ks) {
    store += `  { ivec2 r0 = ivec2(roundEven(clamp(kq * y${2 * k}, vec2(-10.0), vec2(10.0)) * 8.0));\n    ivec2 r1 = ivec2(roundEven(clamp(kq * y${2 * k + 1}, vec2(-10.0), vec2(10.0)) * 8.0));\n`;
    store += `    L[lBase${at(k)}] = (uint(r0.x) & 0xffu) | ((uint(r0.y) & 0xffu) << 8u) | ((uint(r1.x) & 0xffu) << 16u) | ((uint(r1.y) & 0xffu) << 24u); }\n`;
  }
  return VERSION + (f16 ? STORAGE16 + F16 : "") + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer SB { ${f16 ? "f16vec2" : "vec2"} S[]; };
layout(set = 0, binding = 1, std430) readonly buffer UVB { uvec2 UV[]; };
layout(set = 0, binding = 2, std430) readonly buffer ListsB { uint LISTS[]; };
layout(set = 0, binding = 3, std430) buffer LB { uint L[]; };
layout(set = 0, binding = 4, std430) buffer EPB { vec2 EP[]; };
layout(set = 0, binding = 5, std430) buffer CountsB { uint COUNTS[]; };
layout(set = 0, binding = 6, std430) readonly buffer PilotB { float PILOT[]; };
layout(set = 0, binding = 7, std430) readonly buffer TabB { uint TAB[]; };
${PARAMS(8)}
${tables(layout, codes)}
shared vec4 red[256];
shared float kOf[8];

float jinfo(float g) {
  if (g <= 0.0) { return 0.0; }
  float base = 1.0 - exp2(-0.3073 * pow(2.0 * sqrt(g), 1.7870));
  if (base <= 0.0) { return 0.0; }
  return pow(base, 1.1064);
}
vec2 turned(vec2 y, float th) {
  float cs = cos(th);
  float sn = sin(th);
  return vec2(y.x * cs + y.y * sn, y.y * cs - y.x * sn);
}

void main() {
  uvec3 wg = gl_WorkGroupID;
  uint t = gl_LocalInvocationIndex;
  uint b = wg.x;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint v = ${versionOf("LISTS", "P.B", "f")};
  if (b >= ${CHUNKS}u * v) { return; }
  uint g = t >> 5u;
  uint l = t & 31u;
  uint j = 8u * b + g;
  uint ub = (P.uvOff >> 1u) + 160u * j + l;
  uint sBase = f * P.sStride;
  vec2 sh = vec2(PILOT[P.B * P.blocksMax + 2u * f], PILOT[P.B * P.blocksMax + 2u * f + 1u]);
  bool turn = sh.x != 0.0 || sh.y != 0.0;
  float kk = ${TWO_PI} / float(P.n);
  // ai: the pilots' coefficients in this sub-channel, if a tail meets it (tailIn)
  uvec2 tp = tailIn(v, j);
${body}  red[t] = vec4(m2, m4, cs, 0.0);
  barrier();
  for (uint h = 16u; h > 0u; h = h >> 1u) {
    if (l < h) { red[t] = red[t] + red[t + h]; }
    barrier();
  }
  if (l == 0u) {
    vec4 s3 = red[t];
    float mm2 = s3.x / 320.0;
    float mm4 = s3.y / 320.0;
    float a2 = sqrt(max(2.0 * mm2 * mm2 - mm4, 0.0));
    float nv = max(mm2 - a2, 0.02 * mm2);
    float k = 0.0;
    float est = 0.0;
    if (nv > 0.0) { k = 0.7 * 2.0 * sqrt(2.0 * a2) / nv; est = jinfo(a2 / nv); }
    kOf[g] = k;
    // ai: the tail's part here: its axis values against the whitening's signs over this sub-channel's rms axis value,
    // ai: summed (tblocks adds the one or two parts of a tail and divides by its slots); 0 where none or nothing read
    float r = 0.0;
    if (tp.x < tp.y && mm2 > 0.0) { r = s3.z / sqrt(mm2 / 2.0); }
    EP[f * P.subchMax + j] = vec2(est, r);
  }
  barrier();
  float kq = kOf[g];
  uint lBase = f * P.lStride + 160u * j + l;
${store}  if (t == 0u) { atomicAdd(COUNTS[f * 8u + 6u], 8u); }
}
`;
};

export const tblocksSource = ({ layout, codes }) => VERSION + /* glsl */ `
layout(local_size_x = ${TIER_BLOCKS}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer ListsB { uint LISTS[]; };
layout(set = 0, binding = 1, std430) readonly buffer EPB { vec2 EP[]; };
layout(set = 0, binding = 2, std430) buffer BlkB { uvec2 BLK[]; };
layout(set = 0, binding = 3, std430) buffer PilotB { float PILOT[]; };
layout(set = 0, binding = 4, std430) readonly buffer TabB { uint TAB[]; };
${PARAMS(5)}
${tables(layout, codes)}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint t = gl_LocalInvocationIndex;
  uint count = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  uint f = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  uint v = ${versionOf("LISTS", "P.B", "f")};
  if (t >= min(blocksOf(v), ${TIER_BLOCKS}u)) { return; }
  uvec2 bc = blockAt(v, t);
  if (bc.x >= NCODES) { return; }
  // ai: the block's slots s0 to s0 + sl; its estimate its sub-channels' (EP.x), each by the slots of the block it holds
  uint sl = slotsOf(bc.x);
  uint s0 = ttOf(v, bc.x).y + bc.y * sl;
  float sum = 0.0;
  for (uint j = s0 / 640u; 640u * j < s0 + sl; j++) { sum += EP[f * P.subchMax + j].x * float(min(s0 + sl, 640u * j + 640u) - max(s0, 640u * j)); }
  float be = sum / float(sl);
  BLK[f * P.blocksMax + t] = uvec2(floatBitsToUint(be), be < barOf(bc.x) ? 1u : 0u);
  // ai: its pilots' reading: the tail's parts (EP.y of the one or two sub-channels its slots fall in) over its slots
  float r = ${NAN};
  if (tailOf(bc.x) > 0u) {
    uint ts = s0 + nOf(bc.x);
    uint j0 = ts / 640u;
    uint j1 = (s0 + sl - 1u) / 640u;
    float rs = EP[f * P.subchMax + j0].y;
    if (j1 != j0) { rs += EP[f * P.subchMax + j1].y; }
    r = rs / float(sl - nOf(bc.x));
  }
  PILOT[f * P.blocksMax + t] = r;
}
`;

// ai: A code's uniform T (the WGSL's tierUniformStruct, std140: vec4 arrays at the WGSL's offsets).
const tierUniform = (sh) => `layout(set = 0, binding = 8, std140) uniform TierCode { uvec4 dims; uvec4 rows[${sh.rowv}]; uvec4 pairs[${sh.pairv}]; uvec4 pw[30]; } T;`;
// ai: The split CRC-32C (../wgsl/back_ldpc.mjs CRC_SPLIT's arithmetic) over packw, folded into crcw.
const CRC = /* glsl */ `
const uint CRC_CHUNKS = ${CRC_CHUNKS}u;
const uint CRC_POLY = 0x82f63b78u;
uint crcStep(uint c) { return (c >> 1u) ^ (CRC_POLY & (0u - (c & 1u))); }
uint crcOf(uint i) {
  if (i == 0u) { ${wgStore("crcw", "0u")}; }
  barrier();
  if (i < CRC_CHUNKS) {
    uint w = packw[i];
    uint c = 0u;
    for (uint t = 0u; t < 4u; t = t + 1u) {
      c = c ^ ((w >> (8u * t)) & 0xffu);
      for (uint q = 0u; q < 8u; q = q + 1u) { c = crcStep(c); }
    }
    uint p = T.pw[i >> 2u][i & 3u];
    uint acc = 0u;
    uint tt = c;
    for (uint j = 0u; j < 32u; j = j + 1u) {
      if (((p >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = crcStep(tt);
    }
    if (i == 0u) { acc = acc ^ T.pw[CRC_CHUNKS >> 2u][CRC_CHUNKS & 3u]; }
    ${wgXor("crcw", "acc")};
  }
  barrier();
  return ${wgLoad("crcw")};
}
`;
const packWords = (n) => Math.max(120, Math.ceil(n / 32) + 1);

// ai: The LDPC kernel (the WGSL's tldpcSource): sh from ../plan/back_tiers.mjs tierShape (its form), c the code's index
// ai: in CODES, inv its INV section in TAB, net { hidden, at } its learned stop.
export function tldpcSource(sh, { c, inv, layout, codes, net }) {
  const { n: N, k: K, z: Z, mb: MB, ns: NS, threads: TH, one } = sh;
  if (K < KNOWN || NS > 32 || (one && NS > 12)) throw new Error(`tier code ${JSON.stringify(sh)}`);
  if (!(net?.hidden > 0)) throw new Error(`the code ${c}'s kernel needs its learned stop`);
  const R = one ? "uint" : "uvec2", H = net.hidden;
  // ai: the check's slot loops unrolled ([[unroll]]): RADV left the 7/8 and 3/4 codes' (NS 24 and 17) rolled from
  // ai: glslang's SPIR-V, where it unrolls naga's, and took 1.24 times as long; unrolled, every code's kernel is naga's
  // ai: size there again, and the Adreno's "reg+i32" 0.80 to 0.93 times the WGSL's fastest form (2026-10-31)
  const UNR = "[[unroll]] ";
  // ai: the posteriors' store: "pair" two int16 halves a uint, updated by workgroup atomic XOR (two lanes may write the
  // ai: two halves of one word in a layer); "i32" an int a posterior; "i16" an int16_t a posterior (sh.narrow), plain
  // ai: loads and stores: a 16-bit location is its own memory location, so the two halves need no atomics. The
  // ai: int16_t array is declared in whole 32-bit words (2 ceil(N / 2)): the Adreno 840 (driver E031.50.19.27) read an
  // ai: odd-length Lw[4557] (the 7/8 code at z = 93 until 2026-10-31) wrongly, every codeword of that code failing its
  // ai: CRC, where the 4090 and RADV read it exact; every code's n is even at z = 128, and the rule stays for any odd n
  const lw = sh.wide ? "i32" : sh.narrow ? "i16" : "pair";
  const record = (at) => one ? `Rr[${at}] = uint(m1) | (uint(m2) << 8u) | (arg << 16u) | (signs << 20u);` : `Rr[${at}] = uvec2(uint(m1) | (uint(m2) << 8u) | (arg << 16u), signs);`;
  // ai: rowUpdate(r, i, ri) written in place (the WGSL's, one lane a check): its record at Rr[ri]
  const rowBody = (r, i, ri) => /* glsl */ `{
        ${R} rec = Rr[${ri}];
        int min1 = 32767;
        int min2 = 32767;
        uint arg = 0u;
        uint sgn = 0u;
        int idxs[${NS}];
        int qs[${NS}];
        ${UNR}for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = slotIdx(${r}, ${i}, e);
          idxs[e] = idx;
          int q = 8191;
          if (idx >= 0) { q = ldL(idx) - oldMsg(rec, e); }
          qs[e] = q;
        }
        ${UNR}for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = idxs[e];
          if (idx < 0) { continue; }
          int q = qs[e];
          int a = abs(q);
          sgn = sgn ^ uint(q < 0);
          bool lt = a < min1;
          min2 = min(min2, max(min1, a));
          arg = lt ? e : arg;
          min1 = min(min1, a);
        }
        int m1 = min(127, (min1 * NORM) >> 4u);
        int m2 = min(127, (min2 * NORM) >> 4u);
        uint signs = 0u;
        ${UNR}for (uint e = 0u; e < NS; e = e + 1u) {
          int idx = idxs[e];
          if (idx < 0) { continue; }
          int q = qs[e];
          int mag = e == arg ? m2 : m1;
          uint s = sgn ^ uint(q < 0);
          int rn = s != 0u ? -mag : mag;
          signs = signs | (s << e);
          int nl = clamp(q + rn, -8191, 8191);
          stL(idx, q + oldMsg(rec, e), nl);
        }
        ${record(ri)}
      }`;
  const layers = /* glsl */ `for (uint st = 0u; st < MB; st = st + 1u) {
      uint r = st;
      if ((it & 1u) != 0u) { r = MB - 1u - st; }
      if (i < Z) ${rowBody("r", "i", "r")}
      barrier();
    }`;
  return VERSION + ATOMICS + UNROLL + (lw === "i16" ? INT16 : "") + /* glsl */ `
layout(local_size_x = ${TH}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer LinB { uint Lin[]; };
layout(set = 0, binding = 1, std430) readonly buffer BlkB { uvec2 BLK[]; };
layout(set = 0, binding = 2, std430) readonly buffer ListsB { uint LISTS[]; };
layout(set = 0, binding = 3, std430) readonly buffer TabB { uint TAB[]; };
layout(set = 0, binding = 4, std430) buffer VB { uint V[]; };
layout(set = 0, binding = 5, std430) buffer ItsB { int ITS[]; };
layout(set = 0, binding = 6, std430) buffer RecB { uint count; uint words[]; } REC;
layout(set = 0, binding = 7, std430) buffer CountsB { uint BCOUNTS[]; };
${tierUniform(sh)}
${tables(layout, codes)}
const uint CODE = ${c}u;
const uint INV_AT = ${inv}u;
const uint N = ${N}u;
const uint K = ${K}u;
const uint Z = ${Z}u;
const uint MB = ${MB}u;
const uint NS = ${NS}u;
const uint KNOWN = ${KNOWN}u;
const int NORM = 13;
const uint MAXIT = 30u;
const int MAG_CAP = 1016;
const uint GROW_MAX = 65535u;
const uint DATA_WORDS = ${CRC_CHUNKS + 1}u;
const uint REC_WORDS = ${CRC_CHUNKS + 2}u;

${{ i32: `shared int Lw[${N}];`, i16: `shared int16_t Lw[${2 * Math.ceil(N / 2)}];`, pair: `shared uint Lw[${Math.ceil(N / 2)}];` }[lw]}
shared uint badc;
shared uint magc;
shared uint flag;
shared uint packw[${packWords(N)}];
shared uint crcw;
const uint NET_H = ${H}u;
const uint NET_AT = ${net.at}u;
shared float netX[6];
shared float netH[${H}];
shared float netP[${H}];
float netW(uint q) { return uintBitsToFloat(TAB[NET_AT + q]); }

${lw === "i32" ? `int ldL(int idx) { return Lw[uint(idx)]; }
void stL(int idx, int old, int nl) { Lw[uint(idx)] = nl; }` : lw === "i16" ? `int ldL(int idx) { return int(Lw[uint(idx)]); }
void stL(int idx, int old, int nl) { Lw[uint(idx)] = int16_t(nl); }` : `int ldL(int idx) {
  uint w = ${wgLoad("Lw[uint(idx) >> 1u]")};
  return int(w << (16u * (1u - (uint(idx) & 1u)))) >> 16u;
}
void stL(int idx, int old, int nl) {
  ${wgXor("Lw[uint(idx) >> 1u]", "(uint(old ^ nl) & 0xffffu) << (16u * (uint(idx) & 1u))")};
}`}
uint rowAt(uint r) { return T.rows[r >> 2u][r & 3u]; }
int slotIdx(uint r, uint i, uint e) {
  uint p0 = rowAt(r);
  uint nd = rowAt(r + 1u) - p0;
  if (e < nd) {
    uint q = p0 + e;
    uint lc = T.pairs[q >> 2u][q & 3u];
    uint t = i + Z - (lc >> 16u);
    if (t >= Z) { t = t - Z; }
    return int((lc & 0xffffu) * Z + t);
  }
  if (e == nd) {
    if (r > 0u) { return int(K + (r - 1u) * Z + i); }
    if (i > 0u) { return int(K + (MB - 1u) * Z + i - 1u); }
    return -1;
  }
  if (e == nd + 1u) { return int(K + r * Z + i); }
  return -1;
}

int oldMsg(${R} rec, uint e) {
${one ? `  int m = e == ((rec >> 16u) & 0xfu) ? int((rec >> 8u) & 0xffu) : int(rec & 0xffu);
  return ((rec >> (20u + e)) & 1u) != 0u ? -m : m;` : `  int m = e == ((rec.x >> 16u) & 0x1fu) ? int((rec.x >> 8u) & 0xffu) : int(rec.x & 0xffu);
  return ((rec.y >> e) & 1u) != 0u ? -m : m;`}
}

const uint CH = ${Math.ceil(Z / 32)}u;
const uint KCOL = ${K / Z}u;
uint get32(uint p) {
  uint w = p >> 5u;
  uint s = p & 31u;
  uint lo = packw[w] >> s;
  return s == 0u ? lo : lo | (packw[w + 1u] << (32u - s));
}
uint win(uint c, uint t0) {
  uint a = get32(c * Z + t0);
  if (t0 + 32u <= Z) { return a; }
  uint n1 = Z - t0;
  return (a & ((1u << n1) - 1u)) | (get32(c * Z) << n1);
}
void signsToBitmap(uint i) {
  uint mg = 0u;
  for (uint w = i; w < ${Math.ceil(N / 32)}u; w = w + ${TH}u) {
    uint b = 0u;
${lw !== "pair" ? `    for (uint t = 0u; t < 32u; t = t + 1u) {
      uint idx = 32u * w + t;
      if (idx < N) {
        int v = ${lw === "i16" ? "int(Lw[idx])" : "Lw[idx]"};
        if (v < 0) { b = b | (1u << t); }
        mg = mg + uint(min(abs(v), MAG_CAP));
      }
    }` : `    for (uint h = 0u; h < 16u; h = h + 1u) {
      uint q = 16u * w + h;
      if (q < ${Math.ceil(N / 2)}u) {
        uint v = ${wgLoad("Lw[q]")};
        b = b | (((v >> 15u) & 1u) << (2u * h)) | ((v >> 31u) << (2u * h + 1u));
        mg = mg + uint(min(abs(int(v << 16u) >> 16u), MAG_CAP)) + uint(min(abs(int(v) >> 16u), MAG_CAP));
      }
    }`}
    packw[w] = b;
  }
  if (mg != 0u) { ${wgAdd("magc", "mg")}; }
  if (i == 0u) { packw[${Math.ceil(N / 32)}u] = 0u; }
}
uint violatedChunks(uint i) {
  uint c = 0u;
  for (uint u = i; u < MB * CH; u = u + ${TH}u) {
    uint r = u / CH;
    uint i0 = 32u * (u - r * CH);
    uint acc = win(KCOL + r, i0);
    if (r > 0u) { acc = acc ^ win(KCOL + r - 1u, i0); }
    else { acc = acc ^ (win(KCOL + MB - 1u, (i0 + Z - 1u) % Z) & (i0 == 0u ? 0xfffffffeu : 0xffffffffu)); }
    uint q1 = rowAt(r + 1u);
    for (uint q = rowAt(r); q < q1; q = q + 1u) {
      uint lc = T.pairs[q >> 2u][q & 3u];
      uint t0 = i0 + Z - (lc >> 16u);
      if (t0 >= Z) { t0 = t0 - Z; }
      acc = acc ^ win(lc & 0xffffu, t0);
    }
    uint m = min(32u, Z - i0);
    c = c + uint(bitCount(acc & (m == 32u ? 0xffffffffu : (1u << m) - 1u)));
  }
  return c;
}
${CRC}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint i = gl_LocalInvocationIndex;
  uint B = T.dims.x;
  uint lStride = T.dims.y;
  uint blocksMax = T.dims.z;
  // ai: a workgroup a (block of this code, listed frame): the frame from the gate's list of every listed frame (listsAll)
  uint f = LISTS[${SLOTS}u * (B + 1u) + 5u * B + wg.y];
  uint v = ${versionOf("LISTS", "B", "f")};
  uvec4 tt = ttOf(v, CODE);
  uint x = wg.x;
  if (x >= tt.z) { return; }
  uint b = tt.x + x;
  uint slotV = f * blocksMax + b;
  if (BLK[slotV].y != 0u) {
    if (i == 0u) { V[slotV] = 2u; ITS[slotV] = -2; }
    return;
  }
  // ai: the block's first slot (its tier's first and the blocks before it, a multiple of 16), as L's word: four slots a word
  uint lBase = f * lStride + (tt.y + x * slotsOf(CODE)) / 4u;
  for (uint wd = i; wd < (N + 1u) / 2u; wd = wd + ${TH}u) {
    uint pk = 0u;
    for (uint h = 0u; h < 2u; h = h + 1u) {
      uint d = 2u * wd + h;
      if (d >= N) { continue; }
      uint j = d;
      if (d >= K) { uint t = d - K; j = K + MB * (t % Z) + t / Z; }
      int vv = 127;
      if (j < KNOWN || j >= K) {
        uint slot = (TAB[INV_AT + (j >> 1u)] >> (16u * (j & 1u))) & 0xffffu;
        if (slot == 0xffffu) { vv = 0; }
        else {
          uint w = Lin[lBase + (slot >> 2u)];
          vv = int(w << (8u * (3u - (slot & 3u)))) >> 24u;
        }
      }
      ${{ i32: `if (d < N) { Lw[d] = vv; }`, i16: `if (d < N) { Lw[d] = int16_t(vv); }`, pair: `pk = pk | ((uint(vv) & 0xffffu) << (16u * h));` }[lw]}
    }
    ${lw === "pair" ? `${wgStore("Lw[wd]", "pk")};` : ``}
  }
  ${R} Rr[${MB}];
  for (uint q = 0u; q < ${MB}u; q = q + 1u) { Rr[q] = ${R}(0u); }
  if (i == 0u) { ${wgStore("badc", "0u")}; ${wgStore("magc", "0u")}; flag = 0u; }
  barrier();

  uint first0 = 0u;
  uint low = 0xffffffffu;
  uint prev = 0u;
  uint mag1 = 0u;
  uint it = 0u;
  uint exitCode = 0u;
  for (it = 0u; it < MAXIT; it = it + 1u) {
    ${layers}
    signsToBitmap(i);
    barrier();
    uint c = violatedChunks(i);
    if (c != 0u) { ${wgAdd("badc", "c")}; }
    barrier();
    if (i == 0u) {
      uint bad = ${wgLoad("badc")};
      ${wgStore("badc", "0u")};
      uint mg = ${wgLoad("magc")};
      ${wgStore("magc", "0u")};
      uint d = 0u;
      if (bad == 0u) { d = 1u; }
      else {
        if (it == 0u) { first0 = bad; prev = bad; mag1 = mg; }
        low = min(low, bad);
        uint grow = 256u;
        if (mag1 != 0u) { grow = min((256u * mg) / mag1, GROW_MAX); }
        netX[0] = float(bad); netX[1] = float(low); netX[2] = float(first0); netX[3] = float(prev); netX[4] = float(it + 1u); netX[5] = float(grow);
        prev = bad;
      }
      flag = d;
    }
    ${uniformLoad("uint d", "flag")}
    if (d != 0u) { exitCode = d; break; }
    if (it + 1u < MAXIT) {
      if (i < NET_H) {
        precise float a = netW(6u * NET_H + i);
        for (uint k = 0u; k < 6u; k++) { a = a + netW(k * NET_H + i) * netX[k]; }
        netH[i] = max(a, 0.0);
      }
      barrier();
      if (i < NET_H) {
        precise float a = netW((7u + NET_H) * NET_H + i);
        for (uint k = 0u; k < NET_H; k++) { a = a + netW((7u + k) * NET_H + i) * netH[k]; }
        netP[i] = netW((8u + NET_H) * NET_H + i) * max(a, 0.0);
      }
      barrier();
      if (i == 0u) {
        precise float sum = netP[0];
        for (uint k = 1u; k < NET_H; k++) { sum = sum + netP[k]; }
        flag = sum < netW((9u + NET_H) * NET_H) ? 2u : 0u;
      }
      ${uniformLoad("uint g", "flag")}
      if (g != 0u) { exitCode = g; break; }
    }
  }
  int its = -1;
  uint ran = MAXIT;
  if (exitCode == 1u) { its = int(it) + 1; ran = it + 1u; }
  if (exitCode == 2u) { its = -3; ran = it + 1u; }
  if (i == 0u) {
    ITS[slotV] = its;
    atomicAdd(BCOUNTS[f * 8u + 3u], 1u);
    atomicAdd(BCOUNTS[f * 8u + 7u], ran);
  }
  if (its < 0) {
    if (i == 0u) { V[slotV] = 3u; }
    return;
  }
  if (i < DATA_WORDS) {
    uint w = 0u;
    for (uint t = 0u; t < 32u; t = t + 1u) {
      uint j = 32u * i + t;
      if (j < K && ldL(int(j)) < 0) { w = w | (1u << (8u * (t >> 3u) + 7u - (t & 7u))); }
    }
    packw[i] = w;
  }
  uint crc = crcOf(i);
  if (i == 0u) {
    // ai: the CRC-32C the block carries, big-endian in the word after its payload's (bytes 476 to 479)
    uint wc = packw[CRC_CHUNKS];
    uint expect = ((wc & 0xffu) << 24u) | (((wc >> 8u) & 0xffu) << 16u) | (((wc >> 16u) & 0xffu) << 8u) | (wc >> 24u);
    uint sl = 0u;
    if (crc == expect) {
      V[slotV] = 1u;
      atomicAdd(BCOUNTS[f * 8u + 4u], 1u);
      uint n = atomicAdd(REC.count, 1u);
      if (n < T.dims.w) { sl = n + 1u; atomicAdd(BCOUNTS[f * 8u + 5u], 1u); }
    } else {
      V[slotV] = 4u;
    }
    flag = sl;
  }
  ${uniformLoad("uint sl", "flag")}
  if (sl == 0u || i >= REC_WORDS) { return; }
  uint base = (sl - 1u) * REC_WORDS;
  uint word = 0u;
  if (i == 0u) { word = f + 65536u * b; }
  else if (i == 1u) { word = uint(its); }
  else { word = packw[i - 2u]; }
  REC.words[base + i] = word;
}
`;
}
// ai: the uniform's words, the WGSL's (one layout, one buffer: tierUniform in ../back/tiers.mjs writes it)
export { tierUniformWords };
