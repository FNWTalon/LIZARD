// F8, the format word, in GLSL: the WGSL word.mjs (archived) WORD statement for statement (its comments say what it reads and how
// the word is found: the band through the final map, each word cell against its track pairs' grey, ordered statistics
// decoding at order 3, the side-aware gate, sel and result written). One workgroup of 256 a frame.
// ai: Bindings, the Word, Lattice and Held uniforms (std140, every member an array of uvec4 or vec4) and the dispatch
// ai: are the WGSL's; the uniform's layout and tables come from ../plan/word.mjs. No private array is indexed at run
// ai: time (the S26's Adreno returned zeros from one): the code's rows are read from the uniform, the lattice too.
// ai: WGSL's extractBits clamps its offset and count (naga writes the mins): so does byteOf here.
import { VERSION, SAMPLERLESS, DIMS, CONVERT, MAP, uniformLoad } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { SLOTS } from "../plan/common.mjs";
import { LATTICE_MAX } from "../plan/back_transform.mjs";
import { RING_COUNT } from "../plan/finder.mjs";
import { VERSION_MIN, VERSION_MAX, TAU, SIDE_KEEP } from "../wordcode.mjs";
import { CON_MIN, WMAX, ROWS, MIDS_MAX, VECS, K, NW, NCAND, GATE, LAYOUT } from "../plan/word.mjs";

const TOPS = Math.ceil(SLOTS / 4);
const WORDS = ["x", "y", "z", "w"];

const SEARCH = /* glsl */ `
  uint at0 = rg * ${ROWS}u;
  if (li == 0u) { nbw = nb; }
  ${uniformLoad("uint nbu", "nbw")}
  uint nw = nbu / 32u;
  for (uint bit = li; bit < nbu; bit += 256u) {
    float x = ((keep >> (bit / vr.x)) & 1u) != 0u ? q[bit] : 0.0;
    qs[bit >> 2u][bit & 3u] = x;
    key[bit] = abs(x);
    idx[bit] = bit;
  }
  {
    uint g0 = W.code[${VECS}u * (at0 + 1u + (li >> 4u)) + ((li & 15u) >> 2u)][li & 3u];
    gb[li] = (li & 15u) < nw ? g0 : 0u;
  }
  barrier();
  for (uint k = 2u; k <= nbu; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      if (li < nbu / 2u) {
        uint a = 2u * j * (li / j) + (li % j);
        uint b = a + j;
        float ka = key[a];
        float kb = key[b];
        uint ia = idx[a];
        uint ib = idx[b];
        if ((ka > kb || (ka == kb && ia < ib)) != ((a & k) == 0u)) { key[a] = kb; key[b] = ka; idx[a] = ib; idx[b] = ia; }
      }
      barrier();
    }
  }
  for (uint t = li; t < nbu; t += 256u) {
    uint p = idx[t];
    uint m = 0u;
    for (uint k = 0u; k < ${K}u; k++) { m |= ((gb[k * ${NW}u + (p >> 5u)] >> (p & 31u)) & 1u) << k; }
    col[t] = m;
  }
  barrier();
  uint tcur = 0u;
  uint used = 0u;
  for (uint s = 0u; s < ${K}u; s++) {
    if (li == 0u) {
      uint r = ${K}u;
      uint mask = 0u;
      while (true) {
        if (tcur >= nbu) { break; }
        uint m = col[tcur];
        tcur += 1u;
        uint spare = m & ~used;
        if (spare != 0u) { r = uint(findLSB(spare)); mask = m & ~(1u << r); piv[r] = idx[tcur - 1u]; break; }
      }
      if (r < ${K}u) { used |= 1u << r; }
      stepw = uvec4(r, mask, 0u, 0u);
    }
    ${uniformLoad("uvec4 st", "stepw")}
    if (st.x < ${K}u) {
      if (((st.y >> (li >> 4u)) & 1u) != 0u && (li & 15u) < nw) { gb[li] ^= gb[st.x * ${NW}u + (li & 15u)]; }
      for (uint t = li; t < nbu; t += 256u) { uint m = col[t]; col[t] = m ^ (((m >> st.x) & 1u) != 0u ? st.y : 0u); }
    }
    barrier();
  }
  if (li == 0u) {
    uint uu = 0u;
    for (uint r = 0u; r < ${K}u; r++) {
      if (((used >> r) & 1u) == 0u) { continue; }
      uint p = piv[r];
      uint b0 = (W.code[${VECS}u * at0 + (p >> 7u)][(p >> 5u) & 3u] >> (p & 31u)) & 1u;
      uu |= ((qs[p >> 2u][p & 3u] > 0.0 ? 1u : 0u) ^ b0) << r;
    }
    stepw = uvec4(uu, 0u, 0u, 0u);
  }
  ${uniformLoad("uvec4 st0", "stepw")}
  uint u0 = st0.x;
  if (li < nw) {
    uint cw = W.code[${VECS}u * at0 + (li >> 2u)][li & 3u];
    for (uint r = 0u; r < ${K}u; r++) { if (((u0 >> r) & 1u) != 0u) { cw ^= gb[r * ${NW}u + li]; } }
    c0w[li] = cw;
  }
  barrier();
  float best = -1e9;
  uint bc = 0u;
  for (uint c = li; c < ${NCAND}u; c += 256u) {
    uvec4 sub = unrank(c);
    uint v = byteOf(sub, vr.x);
    if (v < ${VERSION_MIN}u || v > ${VERSION_MAX}u) { continue; }
    float sc = 0.0;
    for (uint w = 0u; w < nw; w++) {
      uint cw = c0w[w] ^ rowWord(sub.y, w) ^ rowWord(sub.z, w) ^ rowWord(sub.w, w);
      for (uint b = 0u; b < 8u; b++) {
        uint m = cw >> (4u * b);
        sc += dot(qs[8u * w + b], mix(vec4(-1.0), vec4(1.0), notEqual(uvec4(m) & uvec4(1u, 2u, 4u, 8u), uvec4(0u))));
      }
    }
    if (sc > best) { best = sc; bc = c; }
  }
  red[li] = vec4(best, float(bc), 0.0, 0.0);`;

export const WORD = VERSION + SAMPLERLESS + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
struct Result { float r[32]; };
layout(set = 0, binding = 0) uniform texture2DArray img;
layout(set = 0, binding = 1, std430) readonly buffer Level1 { float lv1[]; };
layout(set = 0, binding = 2, std430) readonly buffer Level2 { float lv2[]; };
layout(set = 0, binding = 3, std430) readonly buffer Level3 { float lv3[]; };
layout(set = 0, binding = 4, std430) readonly buffer Level4 { float lv4[]; };
layout(set = 0, binding = 5, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 6, std140) uniform LevelDims { uvec4 d[5]; } LD;
layout(set = 0, binding = 7, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 8, std430) buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 9, std430) readonly buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 10, std430) readonly buffer Pts { vec4 pts[]; };
layout(set = 0, binding = 11, std430) buffer Results { Result result[]; };
layout(set = 0, binding = 12, std140) uniform Word { uvec4 code[${LAYOUT.range}]; uvec4 range[${RING_COUNT}]; uvec4 plan[${RING_COUNT}]; uvec4 tops[${TOPS}]; uvec4 gate[${(RING_COUNT * GATE) / 4}]; } W;
layout(set = 0, binding = 13, std140) uniform Lattice { uvec4 at[${RING_COUNT}]; vec4 lat[${(RING_COUNT * LATTICE_MAX) / 4}]; } LT;
layout(set = 0, binding = 14, std140) uniform Held { uint version; uint p0; uint p1; uint p2; } HB;
float latAt(uint i) { return LT.lat[i >> 2u][i & 3u]; }
float pix(uint l, uint f, int x, int y) {
  Frame F = frames[f];
  int w = int((F.w + (1u << l) - 1u) >> l);
  int h = int((F.h + (1u << l) - 1u) >> l);
  int xc = clamp(x, 0, w - 1);
  int yc = clamp(y, 0, h - 1);
  if (l == 0u) { return texelFetch(img, ivec3(xc, yc, int(f)), 0).r; }
  uvec4 s = LD.d[l];
  uint i = (f * s.y + uint(yc)) * s.x + uint(xc);
  switch (l) {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
float look(uint l, uint f, vec2 p) {
  vec2 q = p / float(1u << l) - vec2(0.5);
  vec2 q0 = floor(q);
  vec2 a = q - q0;
  ivec2 i = f2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
` + COONS + /* glsl */ `
shared vec4 red[256];
shared float mids[${MIDS_MAX}];
shared float q[${4 * WMAX}];
shared uint go;
shared vec4 win;
shared vec4 qs[${WMAX}];
shared float key[${4 * WMAX}];
shared uint idx[${4 * WMAX}];
shared uint gb[${K * NW}];
shared uint col[${4 * WMAX}];
shared uint c0w[${NW}];
shared uint piv[${K}];
shared uint nbw;
shared uvec4 stepw;
uint rowWord(uint r, uint w) { uint v = gb[min(r, ${K - 1}u) * ${NW}u + w]; return r < ${K}u ? v : 0u; }
uint byteOf(uvec4 sub, uint at) {
  uint w = at >> 5u;
  uint o = min(at & 31u, 32u);
  uint n = min(8u, 32u - o);
  return bitfieldReverse(bitfieldExtract(c0w[w] ^ rowWord(sub.y, w) ^ rowWord(sub.z, w) ^ rowWord(sub.w, w), int(o), int(n))) >> 24u;
}
uvec4 unrank(uint c0) {
  uint c = c0;
  if (c == 0u) { return uvec4(0u, ${K}u, ${K}u, ${K}u); }
  c -= 1u;
  if (c < ${K}u) { return uvec4(1u, c, ${K}u, ${K}u); }
  c -= ${K}u;
  if (c < ${(K * (K - 1)) / 2}u) {
    for (uint a = 0u; a + 1u < ${K}u; a++) { uint n = ${K - 1}u - a; if (c < n) { return uvec4(2u, a, a + 1u + c, ${K}u); } c -= n; }
  }
  c -= ${(K * (K - 1)) / 2}u;
  for (uint a = 0u; a + 2u < ${K}u; a++) {
    for (uint b = a + 1u; b + 1u < ${K}u; b++) { uint n = ${K - 1}u - b; if (c < n) { return uvec4(3u, a, b, b + 1u + c); } c -= n; }
  }
  return uvec4(4u, ${K}u, ${K}u, ${K}u);
}
uint codeBit(uint at, uint v, uint fps, uint i) {
  uint k = i >> 5u;
  uint w = W.code[${VECS}u * at + (k >> 2u)][k & 3u];
  for (uint b = 0u; b < 8u; b++) {
    if (((v >> b) & 1u) != 0u) { w ^= W.code[${VECS}u * (at + 1u + b) + (k >> 2u)][k & 3u]; }
    if (((fps >> b) & 1u) != 0u) { w ^= W.code[${VECS}u * (at + 9u + b) + (k >> 2u)][k & 3u]; }
  }
  return (w >> (i & 31u)) & 1u;
}
uint leastSide(vec4 s, uint skip) {
  uint b = 0u;
  float v = 3.0e38;
  if ((skip & 1u) == 0u && s.x < v) { v = s.x; b = 1u; }
  if ((skip & 2u) == 0u && s.y < v) { v = s.y; b = 2u; }
  if ((skip & 4u) == 0u && s.z < v) { v = s.z; b = 4u; }
  if ((skip & 8u) == 0u && s.w < v) { v = s.w; b = 8u; }
  return b;
}
uint gateAt(uint r, uint m) { uint k = r * ${GATE}u + (m >> 1u); return (W.gate[k >> 2u][k & 3u] >> (16u * (m & 1u))) & 0xffffu; }
uint slotOf(uint v) {
  return 0u${Array.from({ length: SLOTS - 1 }, (_, k) => ` + (v > W.tops[${k >> 2}].${WORDS[k & 3]} ? 1u : 0u)`).join("")};
}
vec2 imgAt(uint f, float M[16], uvec4 L, vec2 m) { vec2 p = toImage(M, m); return p + lensAt(lensOf(f), p) + coons(f, L.x, L.y, m); }
void reduce(uint li) {
  barrier();
  for (uint st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    barrier();
  }
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  if (li == 0u) { go = (frames[f].valid != 0u && sel[f].y != 0u) ? 1u : 0u; }
  { ${uniformLoad("uint go0", "go")} if (go0 == 0u) { return; } }
  uint rg = sel[f].x;
  uvec4 P = W.plan[rg];
  float M[16] = maps[f];
  uvec4 L = LT.at[rg];
  vec2 c0 = toImage(M, pts[P.x].xy);
  float u = distance(toImage(M, pts[P.x].xy + vec2(1.0, 0.0)), c0);
  uint l = f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0));
  vec2 lg = vec2(0.0);
  for (uint s = li; s < P.y; s += 256u) {
    lg += vec2(look(l, f, imgAt(f, M, L, pts[P.x + 2u * s].xy)), look(l, f, imgAt(f, M, L, pts[P.x + 2u * s + 1u].xy)));
  }
  red[li] = vec4(lg, 0.0, 0.0);
  reduce(li);
  float con = (red[0].y - red[0].x) / float(max(P.y, 1u));
  uint mp = P.x + 2u * P.y;
  for (uint k = li; k < P.z; k += 256u) { mids[k] = 0.5 * (look(l, f, imgAt(f, M, L, pts[mp + 2u * k].xy)) + look(l, f, imgAt(f, M, L, pts[mp + 2u * k + 1u].xy))); }
  barrier();
  uint wp = mp + 2u * P.z;
  uvec4 vr = W.range[rg];
  uint nb = 4u * vr.x;
  for (uint bit = li; bit < nb; bit += 256u) {
    vec4 e = pts[wp + bit];
    uint lo = f2u(e.z) % 4096u;
    uint hi = f2u(e.z) / 4096u;
    float grey = 0.0;
    for (uint j = lo; j <= hi; j++) { grey += mids[j]; }
    q[bit] = clamp((grey / float(hi - lo + 1u) - look(l, f, imgAt(f, M, L, e.xy))) / max(con, 1e-3), -0.5, 0.5);
  }
  barrier();
  vec4 sa = vec4(0.0);
  for (uint bit = li; bit < nb; bit += 256u) { sa += mix(vec4(0.0), vec4(abs(q[bit])), equal(uvec4(0u, 1u, 2u, 3u), uvec4(bit / vr.x))); }
  red[li] = sa;
  reduce(li);
  vec4 sides = red[0];
  uint d1 = leastSide(sides, 0u);
  uint d2 = d1 | leastSide(sides, d1);
  float top = ${SIDE_KEEP} * max(max(sides.x, sides.y), max(sides.z, sides.w));
  uint low = (sides.x < top ? 1u : 0u) | (sides.y < top ? 2u : 0u) | (sides.z < top ? 4u : 0u) | (sides.w < top ? 8u : 0u);
  uint keep = 15u & ~(low & d2);
  barrier();${SEARCH}
  barrier();
  if (li < 16u) {
    vec4 b = red[16u * li];
    for (uint k = 16u * li + 1u; k < 16u * li + 16u; k++) {
      vec4 o = red[k];
      if (o.x > b.x || (o.x == b.x && o.y < b.y)) { b = o; }
    }
    red[16u * li] = b;
  }
  barrier();
  if (li == 0u) {
    vec4 b = red[0];
    for (uint k = 1u; k < 16u; k++) {
      vec4 o = red[16u * k];
      if (o.x > b.x || (o.x == b.x && o.y < b.y)) { b = o; }
    }
    uvec4 sub = unrank(f2u(b.y));
    uint v = byteOf(sub, vr.x);
    uint fr = byteOf(sub, 2u * vr.x);
    bool valid = b.x > -1e8 && v >= ${VERSION_MIN}u && v <= ${VERSION_MAX}u;
    uint cv = (v - 1u) * 256u + fr;
    win = vec4(b.x, float(valid ? cv : 0u), valid ? 1.0 : 0.0, 0.0);
  }
  ${uniformLoad("vec4 wn", "win")}
  uint c = f2u(wn.y);
  uint v = 1u + c / 256u;
  uint fps = c % 256u;
  vec3 g = vec3(0.0);
  for (uint bit = li; bit < nb; bit += 256u) {
    if (abs(q[bit]) < ${TAU}) { continue; }
    uint s = 1u << (bit / vr.x);
    float one = 1024.0 + (((q[bit] > 0.0) != (codeBit(at0, v, fps, bit) != 0u)) ? 1.0 : 0.0);
    g += mix(vec3(0.0), vec3(one), bvec3(true, (s & d1) == 0u, (s & d2) == 0u));
  }
  red[li] = vec4(g, 0.0);
  reduce(li);
  if (li == 0u) {
    vec4 G = red[0];
    uint m0 = f2u(G.x / 1024.0);
    uint m1 = f2u(G.y / 1024.0);
    uint m2 = f2u(G.z / 1024.0);
    uint e0 = gateAt(rg, m0);
    uint e1 = gateAt(rg, m1);
    uint e2 = gateAt(rg, m2);
    bool through = (e0 != 0xffffu && f2u(G.x) - 1024u * m0 <= e0) || (e1 != 0xffffu && f2u(G.y) - 1024u * m1 <= e1) || (e2 != 0xffffu && f2u(G.z) - 1024u * m2 <= e2);
    bool ok = con >= ${CON_MIN} && wn.z > 0.5 && through;
    result[f].r[5] = ok ? float(v) : 0.0;
    result[f].r[6] = ok ? float(fps) : 0.0;
    result[f].r[7] = wn.x;
    uint dv = ok ? v : HB.version;
    result[f].r[29] = float(dv);
    result[f].r[30] = (!ok && HB.version > 0u) ? 1.0 : 0.0;
    if (dv > 0u) { sel[f] = uvec4(rg, 1u + dv, slotOf(dv), 0u); }
    else { sel[f] = uvec4(rg, 0u, 0u, 0u); }
  }
}
`;
