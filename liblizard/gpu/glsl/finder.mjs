// F5 to F6 and the second round's REDO in GLSL, the WGSL finder.mjs (archived) statement for statement (its comments say what
// each kernel does and why, and its constants are the ones used here): VOTE, PEAKS, GATHER, COMPLETE, SCORE, PICK and
// REDO, with the bindings, workgroup sizes and dispatches of the WGSL kernels; TRACE (each quad's bend, before SCORE) is
// GLSL alone.
// ai: WGSL's workgroupUniformLoad is common.mjs uniformLoad; its atomicLoad and atomicStore are common.mjs wgLoad and
// ai: wgStore on a workgroup variable, devLoad and devStore on a storage buffer. Workgroup memory is not zeroed but
// ai: GATHER's keys, which atomicMax builds on from 0 (NULL_INIT); every other workgroup variable is written before the
// ai: barrier that precedes its reads.
import { VERSION, SAMPLERLESS, NULL_INIT, ATOMICS, DIMS, CONVERT, LEVEL_BINDINGS, LEVELS, lit, zeros, uniformLoad, wgLoad, wgStore, devLoad, devStore } from "./common.mjs";
import {
  CENTRE, K, CELL, ORIENT, RING_COUNT, HYP, SMIN, VOTE_READS, RAY_LO, RAY_HI, RAY_SPREAD, NEAR, SUPPRESS, PEAK_LIST, STATIONS, R2_N,
} from "../plan/finder.mjs";

// ai: The constants the WGSL inlines that are not exported there (its PROFILE, NCC_MIN, PASSES, GROUP).
const PROFILE = 13, NCC_MIN = 0.6, PASSES = [[0.12, 3.0], [0.4, 1.5], [1.0, 1.5]], GROUP = 256 / K;

const P5 = (binding) => `layout(set = 0, binding = ${binding}, std140) uniform P5 { uint cap; uint gx; uint gy; uint p0; vec4 modules; } P;\n`;
const CENTRES = `const float CENTRE[2] = float[2](${lit(CENTRE[0])}, ${lit(CENTRE[1])});\n`;
const PI = "const float PI = 3.14159265;\n";

// ai: F5a: votes (the WGSL's VOTE). Dispatched (ceil(cap / VOTE_READS), frames).
export const VOTE = VERSION + ATOMICS + DIMS + CONVERT + PI + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 2, std430) buffer Acc { uint acc[]; };
` + P5(3) + CENTRES + /* glsl */ `
const uint CHUNK = 256u;
shared uint vlist[${VOTE_READS}];
shared uint vcount;
shared uint nvote;
shared float dk[CHUNK];
shared uint nk;
shared float dnext;
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  if (li == 0u) { ${wgStore("vcount", "0u")}; }
  barrier();
  uint i0 = wg.x * ${VOTE_READS}u + li;
  if (li < ${VOTE_READS}u && i0 < P.cap && frames[f].valid == 1u) {
    uint o0 = 3u * (f * P.cap + i0);
    if (!(readings[o0 + 2u].w == 0.0 || readings[o0 + 1u].x < ${SMIN})) { vlist[atomicAdd(vcount, 1u)] = i0; }
  }
  barrier();
  if (li == 0u) { nvote = ${wgLoad("vcount")}; }
  ${uniformLoad("uint nv", "nvote")}
  for (uint e = 0u; e < nv; e++) {
    uint i = vlist[e];
    uint o = 3u * (f * P.cap + i);
    vec4 r0 = readings[o];
    vec4 r2 = readings[o + 2u];
    float p = readings[o + 1u].x;
    float beta = r0.w + PI / 4.0;
    vec2 inward = -vec2(cos(beta), sin(beta));
    vec2 perp = vec2(-inward.y, inward.x);
    float c = CENTRE[f2u(r2.x)];
    uint w = f2u(clamp(p, 0.0, 1.0) * 256.0);
    int gxf = int((frames[f].w + ${CELL - 1}u) / ${CELL}u);
    int gyf = int((frames[f].h + ${CELL - 1}u) / ${CELL}u);
    float far = length(max(r0.xy, vec2(float(gxf), float(gyf)) * ${CELL}.0 - r0.xy)) + ${CELL}.0;
    for (uint m = 0u; m < ${RING_COUNT}u; m++) {
      float reach = (P.modules[m] - 2.0 * c) / sqrt(2.0) * r0.z;
      float d0 = ${RAY_LO} * reach;
      float d1 = min(${RAY_HI} * reach, far);
      int spread = f2i(clamp(ceil(${RAY_SPREAD} * reach / ${CELL}.0), 1.0, 6.0));
      uint S = uint(2 * spread + 1);
      barrier();
      if (li == 0u) { dnext = d0; }
      for (;;) {
        barrier();
        if (li == 0u) {
          float d = dnext;
          uint k = 0u;
          for (; k < CHUNK && d <= d1; k++) { dk[k] = d; d += ${CELL}.0; }
          nk = k;
          dnext = d;
        }
        ${uniformLoad("uint n", "nk")}
        if (n == 0u) { break; }
        for (uint t2 = li; t2 < n * S; t2 += 64u) {
          int q = int(t2 % S) - spread;
          vec2 at = (r0.xy + dk[t2 / S] * inward) / ${CELL}.0;
          vec2 cell = at + float(q) * perp;
          int x = f2i(floor(cell.x));
          int y = f2i(floor(cell.y));
          if (x < 0 || y < 0 || x >= gxf || y >= gyf) { continue; }
          float t = 1.0 - float(abs(q)) / float(spread + 1);
          atomicAdd(acc[((f * ${RING_COUNT}u + m) * P.gy + uint(y)) * P.gx + uint(x)], f2u(float(w) * t));
        }
        barrier();
      }
    }
  }
}
`;

// ai: The scan's functions, PEAKS's (the WGSL's PEAK_FNS), over acc read as plain words.
const PEAK_FNS = /* glsl */ `
uint accAt(uint f, uint m, int x, int y, int gw, int gh) {
  if (x < 0 || y < 0 || x >= gw || y >= gh) { return 0u; }
  return acc[((f * ${RING_COUNT}u + m) * P.gy + uint(y)) * P.gx + uint(x)];
}
uint merged(uint f, int x, int y, int gw, int gh) {
  uint v = 0u;
  for (uint m = 0u; m < ${RING_COUNT}u; m++) { v = max(v, accAt(f, m, x, y, gw, gh)); }
  return v;
}
bool outranks(uint f, int x, int y, uint v, int dx, int dy, int gw, int gh) {
  uint q = merged(f, x + dx, y + dy, gw, gh);
  return q > v || (q == v && (dy < 0 || (dy == 0 && dx < 0)));
}
bool tallest(uint f, int x, int y, uint v, int r, int gw, int gh) {
  for (int d = 1; d <= r; d++) {
    for (int k = -d; k <= d; k++) {
      if (outranks(f, x, y, v, k, -d, gw, gh) || outranks(f, x, y, v, k, d, gw, gh) || outranks(f, x, y, v, -d, k, gw, gh) || outranks(f, x, y, v, d, k, gw, gh)) { return false; }
    }
  }
  return true;
}
`;

// ai: F5b: every peak of a frame's merged rings, a lane a cell (the WGSL's PEAKS). Dispatched (ceil(gx gy / 256), frames).
export const PEAKS = VERSION + DIMS + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Acc { uint acc[]; };
layout(set = 0, binding = 2, std430) buffer VList { vec4 vlist[]; };
layout(set = 0, binding = 3, std430) buffer VCount { uint vcount[]; };
` + P5(4) + PEAK_FNS + /* glsl */ `
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  if (frames[f].valid > 1u) { return; }
  int gw = int((frames[f].w + ${CELL - 1}u) / ${CELL}u);
  int gh = int((frames[f].h + ${CELL - 1}u) / ${CELL}u);
  uint c = wg.x * 256u + li;
  if (c >= uint(gw * gh)) { return; }
  int x = int(c % uint(gw));
  int y = int(c / uint(gw));
  uint v = merged(f, x, y, gw, gh);
  if (v == 0u || !tallest(f, x, y, v, ${SUPPRESS}, gw, gh)) { return; }
  uint ring = 0u;
  for (uint m = ${RING_COUNT}u; m > 0u; m--) { if (accAt(f, m - 1u, x, y, gw, gh) == v) { ring = m - 1u; } }
  uint k = atomicAdd(vcount[f], 1u);
  if (k < ${PEAK_LIST}u) { vlist[f * ${PEAK_LIST}u + k] = vec4((float(x) + 0.5) * ${CELL}.0, (float(y) + 0.5) * ${CELL}.0, float(ring), float(v)); }
}
`;

// ai: F5b's selection and F5c's quads, a workgroup a frame (the WGSL's GATHER).
export const GATHER = VERSION + ATOMICS + NULL_INIT + DIMS + CONVERT + PI + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 2, std430) readonly buffer Acc { uint acc[]; };
layout(set = 0, binding = 3, std430) buffer Cand { vec4 cand[]; };
layout(set = 0, binding = 4, std430) buffer Quads { vec4 quads[]; };
` + P5(5) + /* glsl */ `
layout(set = 0, binding = 6, std430) readonly buffer VList { vec4 vlist[]; };
layout(set = 0, binding = 7, std430) buffer VCount { uint vcount[]; };
` + CENTRES + /* glsl */ `
shared vec4 list[${PEAK_LIST}];
shared uint nlist;
shared vec4 peak[${K}];
shared uint leadK[${4 * K}] = {};
shared uint slotK[${16 * K}] = {};
bool ahead(vec4 a, vec4 b) {
  return a.w > b.w || (a.w == b.w && (a.y < b.y || (a.y == b.y && a.x < b.x)));
}
uvec4 keyOf(uint f, uint i) {
  uint o = 3u * (f * P.cap + i);
  vec4 r0 = readings[o];
  uint w = f2u(floor(clamp(readings[o + 1u].x, 0.0, 1.0) * 1000.0)) * 300u;
  uint rb = floatBitsToUint(readings[o + 2u].y) & 0x7fffffffu;
  uint yb = floatBitsToUint(max(r0.y, 0.0)) & 0x7fffffffu;
  uint xb = floatBitsToUint(max(r0.x, 0.0)) & 0x7fffffffu;
  return uvec4((w << 13u) | (rb >> 19u), ((rb & 0x7ffffu) << 13u) | (yb >> 19u), ((yb & 0x7ffffu) << 13u) | (xb >> 19u),
               ((xb & 0x7ffffu) << 13u) | i);
}
bool leadHas(uvec4 k, uint pk, uint j) {
  bool ok = true;
  for (uint t = 0u; t < j; t++) { ok = ok && k[t] == ${wgLoad("leadK[4u * pk + t]")}; }
  return ok;
}
bool slotHas(uvec4 k, uint pk, uint q, uint j) {
  bool ok = true;
  for (uint t = 0u; t < j; t++) { ok = ok && k[t] == ${wgLoad("slotK[16u * pk + 4u * q + t]")}; }
  return ok;
}
bool support(uint f, uint i, vec4 c) {
  uint o = 3u * (f * P.cap + i);
  vec4 r0 = readings[o];
  vec4 r2 = readings[o + 2u];
  if (r2.w == 0.0 || readings[o + 1u].x < ${SMIN}) { return false; }
  float edge = CENTRE[f2u(r2.x)];
  float near = (P.modules[0] - 2.0 * edge) / sqrt(2.0) * r0.z;
  float far = (P.modules[${RING_COUNT - 1}] - 2.0 * edge) / sqrt(2.0) * r0.z;
  float beta = r0.w + PI / 4.0;
  vec2 inward = -vec2(cos(beta), sin(beta));
  vec2 d = c.xy - r0.xy;
  float along = dot(d, inward);
  float across = abs(d.x * inward.y - d.y * inward.x);
  if (along < ${RAY_LO} * near || along > ${RAY_HI} * far) { return false; }
  return across <= max(2.0 * ${CELL}.0, ${RAY_SPREAD} * along + ${CELL}.0);
}
uint cornerOf(uint f, uint i, vec4 c, float u, float alpha) {
  if (!support(f, i, c)) { return 4u; }
  vec4 r = readings[3u * (f * P.cap + i)];
  if (r.z < 0.2 * u || r.z > 5.0 * u) { return 4u; }
  float a = atan(r.y - c.y, r.x - c.x) - alpha - PI / 4.0;
  return uint(f2i(roundEven(a / (PI / 2.0))) & 3);
}
vec2 slotAt(uint f, uint pk, uint q) { return readings[3u * (f * P.cap + (${wgLoad("slotK[16u * pk + 4u * q + 3u]")} & 8191u))].xy; }
vec2 centreOf(uint f, uint pk, bool p02, bool p13) {
  vec2 a = slotAt(f, pk, 0u);
  vec2 b = slotAt(f, pk, 2u);
  vec2 e = slotAt(f, pk, 1u);
  vec2 g = slotAt(f, pk, 3u);
  vec2 r = b - a;
  vec2 s = g - e;
  float den = r.x * s.y - r.y * s.x;
  if (p02 && p13 && abs(den) > 1e-3) { return a + ((e.x - a.x) * s.y - (e.y - a.y) * s.x) / den * r; }
  vec2 mid = p02 ? 0.5 * (a + b) : 0.5 * (e + g);
  vec2 A = p02 ? a : e;
  vec2 d = p02 ? r : s;
  uint q1 = 4u;
  for (uint q = p02 ? 1u : 0u; q < 4u; q += 2u) { if (${wgLoad("slotK[16u * pk + 4u * q]")} != 0u) { q1 = q; } }
  if (q1 == 4u) { return mid; }
  vec4 rq = readings[3u * (f * P.cap + (${wgLoad("slotK[16u * pk + 4u * q1 + 3u]")} & 8191u))];
  float beta = rq.w + PI / 4.0;
  vec2 inward = -vec2(cos(beta), sin(beta));
  float dn = inward.x * d.y - inward.y * d.x;
  if (abs(dn) < 1e-3 * length(d)) { return mid; }
  float t = ((A.x - rq.x) * d.y - (A.y - rq.y) * d.x) / dn;
  return (t > 0.0) ? rq.xy + t * inward : mid;
}
void gather(uint f, uint pk, vec4 c, uint lane) {
  bool live = frames[f].valid != 0u && c.w > 0.0;
  for (uint j = 0u; j < 4u; j++) {
    for (uint i = lane; i < P.cap; i += ${GROUP}u) {
      if (!live) { break; }
      uvec4 k = keyOf(f, i);
      if (leadHas(k, pk, j) && support(f, i, c)) { atomicMax(leadK[4u * pk + j], k[j]); }
    }
    barrier();
  }
  bool lead = ${wgLoad("leadK[4u * pk]")} != 0u;
  vec4 lr = readings[3u * (f * P.cap + (${wgLoad("leadK[4u * pk + 3u]")} & 8191u))];
  float alpha = lr.w;
  vec4 at = c;
  bool refined = false;
  for (uint rnd = 0u; rnd < 2u; rnd++) {
    if (rnd == 1u) {
      bool p02 = ${wgLoad("slotK[16u * pk]")} != 0u && ${wgLoad("slotK[16u * pk + 8u]")} != 0u;
      bool p13 = ${wgLoad("slotK[16u * pk + 4u]")} != 0u && ${wgLoad("slotK[16u * pk + 12u]")} != 0u;
      refined = lead && (p02 || p13);
      if (refined) { at = vec4(centreOf(f, pk, p02, p13), c.z, c.w); }
    }
    barrier();
    if (rnd == 1u && refined && lane < 16u) { ${wgStore("slotK[16u * pk + lane]", "0u")}; }
    barrier();
    bool go = lead && (rnd == 0u || refined);
    for (uint j = 0u; j < 4u; j++) {
      uvec4 firsts = uvec4(${wgLoad("slotK[16u * pk]")}, ${wgLoad("slotK[16u * pk + 4u]")}, ${wgLoad("slotK[16u * pk + 8u]")}, ${wgLoad("slotK[16u * pk + 12u]")});
      for (uint i = lane; i < P.cap; i += ${GROUP}u) {
        if (!go) { break; }
        uvec4 k = keyOf(f, i);
        if (j > 0u && !any(equal(firsts, uvec4(k.x)))) { continue; }
        uint q = cornerOf(f, i, at, lr.z, alpha);
        if (q < 4u && slotHas(k, pk, q, j)) { atomicMax(slotK[16u * pk + 4u * q + j], k[j]); }
      }
      barrier();
    }
  }
  if (lane == 0u) {
    uint base = 5u * (f * ${K}u + pk);
    uint n = 0u;
    vec4 pts[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));
    for (uint q = 0u; q < 4u; q++) {
      if (!lead || ${wgLoad("slotK[16u * pk + 4u * q]")} == 0u) { pts[q] = vec4(0.0); continue; }
      uint o = 3u * (f * P.cap + (${wgLoad("slotK[16u * pk + 4u * q + 3u]")} & 8191u));
      pts[q] = vec4(readings[o].xy, CENTRE[f2u(readings[o + 2u].x)], max(readings[o].z, 1e-3));   // ai: w: the mark's own module size (> 0: the corner is there)
      n++;
    }
    float u = 0.0;
    if (n == 4u) {
      uint m = f2u(c.z);
      u = 0.25 * (distance(pts[0].xy, pts[1].xy) + distance(pts[1].xy, pts[2].xy) + distance(pts[2].xy, pts[3].xy) + distance(pts[3].xy, pts[0].xy)) / (P.modules[m] - 2.0 * pts[0].z);
    }
    for (uint q = 0u; q < 4u; q++) { quads[base + q] = pts[q]; }
    quads[base + 4u] = vec4(c.z, u, 0.0, float(n));   // ai: z: the quad's bend (TRACE), 0 until traced
  }
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  if (frames[f].valid > 1u) { return; }
  uint np = min(${devLoad("vcount[f]")}, ${PEAK_LIST}u);
  for (uint k = li; k < np; k += 256u) { list[k] = vlist[f * ${PEAK_LIST}u + k]; }
  if (li == 0u) { ${wgStore("nlist", "np")}; }
  barrier();
  if (li == 0u) {
    ${devStore("vcount[f]", "0u")};
    uint n = min(${wgLoad("nlist")}, ${PEAK_LIST}u);
    for (uint k = 0u; k < ${K}u; k++) {
      uint b = k;
      for (uint q = k + 1u; q < n; q++) { if (ahead(list[q], list[b])) { b = q; } }
      if (k < n) { vec4 e = list[b]; list[b] = list[k]; list[k] = e; cand[f * ${K}u + k] = e; peak[k] = e; } else { peak[k] = vec4(0.0); }
    }
  }
  barrier();
  gather(f, li / ${GROUP}u, peak[li / ${GROUP}u], li % ${GROUP}u);
}
`;

// ai: F5d: a quad of three corners completed from the ring (the WGSL's COMPLETE). A workgroup a (frame, quad).
const passAfter = ([frac, reach]) => /* glsl */ `{
    vec3 hit = vec3(0.0);
    if (span) {
      float t = mix(t0, t1, (float(j) + 0.5) / ${STATIONS}.0 * ${lit(frac)});
      float uu = mix(u0, um, t / L);
      vec4 ln = line[side];
      vec2 nn = vec2(-ln.w, ln.z);
      if (dot(nn, n0) < 0.0) { nn = -nn; }
      vec2 B = P0 + t * normalize(Pm0 - P0);
      hit = ringAt(f, ln.xy + dot(B - ln.xy, ln.zw) * ln.zw, nn, uu, ${lit(reach)});
    }
    st[li] = vec4(hit.xy, hit.z, 0.0);
    barrier();
    if (j == 0u) { fitSide(side, 0.5 * (u0 + um), (t1 - t0) * ${lit(frac)}, 6u); }
    barrier();
  }`;
if (STATIONS > 32) throw new Error(`COMPLETE keeps a side's stations in one word: ${STATIONS} past 32`);
export const COMPLETE = VERSION + ATOMICS + SAMPLERLESS + DIMS + CONVERT + LEVEL_BINDINGS + LEVELS + /* glsl */ `
layout(local_size_x = ${2 * STATIONS}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) readonly buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 8, std430) buffer Quads { vec4 quads[]; };
` + P5(9) + /* glsl */ `
shared uint go;
shared uint ri[4];
shared vec4 G[6];
shared vec4 st[${2 * STATIONS}];
shared vec4 line[2];
shared uint kept[2];
float tmpl(uint s) { float r = -3.25 + 0.5 * float(s); return (r >= -1.0 && r < 1.0) ? -1.0 : 1.0; }
// ai: A search's profile read once: its windows (PROFILE samples half a module apart, centred a quarter module a step
// ai: from -(steps + 1) to steps + 1) share one lattice of quarter modules, X + m u n / 4 for m from -steps - 12 to
// ai: steps + 14, so each point is read once into the lane's own slice (PROF_MAX points; off the frame -1, as no
// ai: bilinear read of the levels is) and every window reads its samples there.
const int PROF_MAX = ${2 * Math.round(4 * Math.max(...PASSES.map((p) => p[1]))) + 2 * PROFILE + 1};
shared float prof[${2 * STATIONS} * PROF_MAX];
float ringNcc(uint at) {
  float k = 0.0;
  float sv = 0.0;
  float sv2 = 0.0;
  float sumT = 0.0;
  float sumT2 = 0.0;
  float svt = 0.0;
  bool bad = false;
  for (uint s = 0u; s < ${PROFILE}u; s++) {
    float v = prof[at - 2u * s];
    bool inside = v >= 0.0;
    bad = bad || (!inside && s >= 5u);
    float w = inside ? 1.0 : 0.0;
    float t = tmpl(s);
    k += w; sv += w * v; sv2 += w * v * v; sumT += w * t; sumT2 += w * t * t; svt += w * v * t;
  }
  if (bad || k < 8.0) { return -2.0; }
  float cov = svt - sv * sumT / k;
  float vv = sv2 - sv * sv / k;
  float tv = sumT2 - sumT * sumT / k;
  return cov / sqrt(max(vv * tv, 1e-12));
}
vec3 ringAt(uint f, vec2 X, vec2 n, float u, float reach) {
  uint l = levelFor(u);
  int steps = f2i(roundEven(reach * 4.0));
  uint base = gl_LocalInvocationIndex * uint(PROF_MAX);
  for (int m = -steps - 12; m <= steps + 14; m++) {
    vec2 p = X + (0.25 * float(m)) * u * n;
    prof[base + uint(m + steps + 12)] = onFrame(f, p) ? look(l, f, p) : -1.0;
  }
  // ai: the window centred at X + i u n / 4 has its sample s at m = i + 13 - 2 s: its first at base + i + steps + 25
  uint at0 = base + uint(steps) + 25u;
  float a = ringNcc(uint(int(at0) - steps - 1));
  float b = ringNcc(uint(int(at0) - steps));
  float best = -2.0;
  float at = 0.0;
  for (int i = -steps; i <= steps; i++) {
    float c = ringNcc(uint(int(at0) + i + 1));
    if (b >= ${NCC_MIN} && b >= a && b >= c) {
      float dt = 0.0;
      float e = a + c - 2.0 * b;
      if (a > -2.0 && c > -2.0 && e < 0.0) { dt = clamp(0.5 * (a - c) / e, -0.5, 0.5); }
      best = b;
      at = 0.25 * (float(i) + dt);
    }
    a = b; b = c;
  }
  return vec3(X + at * u * n, (best >= ${NCC_MIN}) ? best : 0.0);
}
// ai: the stations kept, a bit each of one word (STATIONS 32): on the Adreno 16 registers and 4,495 instructions
// ai: against a private array's 24 and 5,648
bool fitSide(uint side, float u, float span, uint need) {
  uint ok = 0u;
  for (uint j = 0u; j < ${STATIONS}u; j++) { ok |= (st[side * ${STATIONS}u + j].z > 0.0) ? (1u << j) : 0u; }
  vec2 o = vec2(0.0);
  vec2 d = vec2(1.0, 0.0);
  for (uint ps = 0u; ps < 2u; ps++) {
    float sw = 0.0;
    vec2 m = vec2(0.0);
    for (uint j = 0u; j < ${STATIONS}u; j++) { vec4 p = st[side * ${STATIONS}u + j]; if ((ok & (1u << j)) != 0u) { sw += p.z; m += p.z * p.xy; } }
    if (sw <= 0.0) { return false; }
    m /= sw;
    float cxx = 0.0; float cxy = 0.0; float cyy = 0.0;
    for (uint j = 0u; j < ${STATIONS}u; j++) {
      vec4 p = st[side * ${STATIONS}u + j];
      if ((ok & (1u << j)) == 0u) { continue; }
      vec2 e = p.xy - m;
      cxx += p.z * e.x * e.x; cxy += p.z * e.x * e.y; cyy += p.z * e.y * e.y;
    }
    float th = 0.5 * atan(2.0 * cxy, cxx - cyy);
    o = m;
    d = vec2(cos(th), sin(th));
    if (ps == 0u) {
      for (uint j = 0u; j < ${STATIONS}u; j++) {
        vec2 e = st[side * ${STATIONS}u + j].xy - o;
        if ((ok & (1u << j)) != 0u && abs(e.x * d.y - e.y * d.x) > u) { ok &= ~(1u << j); }
      }
    }
  }
  uint n = 0u;
  float tmin = 1e30;
  float tmax = -1e30;
  for (uint j = 0u; j < ${STATIONS}u; j++) {
    if ((ok & (1u << j)) == 0u) { continue; }
    n++;
    float t = dot(st[side * ${STATIONS}u + j].xy - o, d);
    tmin = min(tmin, t); tmax = max(tmax, t);
  }
  if (n < max(need, 6u) || tmax - tmin < 0.25 * span) { return false; }
  line[side] = vec4(o, d);
  kept[side] = n;
  return true;
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x / ${K}u;
  uint base = 5u * wg.x;
  if (li == 0u) {
    bool four = false;
    for (uint q = 0u; q < ${K}u; q++) { four = four || quads[5u * (f * ${K}u + q) + 4u].w == 4.0; }
    go = (frames[f].valid == 1u && quads[base + 4u].w == 3.0 && !four) ? 1u : 0u;
    for (uint q = 0u; q < 4u; q++) { ${wgStore("ri[q]", "0u")}; }
  }
  ${uniformLoad("uint go0", "go")}
  if (go0 == 0u) { return; }
  for (uint i = li; i < P.cap; i += ${2 * STATIONS}u) {
    vec4 r = readings[3u * (f * P.cap + i)];
    for (uint q = 0u; q < 4u; q++) {
      vec4 c = quads[base + q];
      if (c.w > 0.0 && r.x == c.x && r.y == c.y && r.z > 0.0) { atomicMax(ri[q], i + 1u); }
    }
  }
  barrier();
  if (li == 0u) {
    uint m = 0u;
    for (uint q = 0u; q < 4u; q++) { if (quads[base + q].w == 0.0) { m = q; } }
    uint a = (m + 1u) % 4u;
    uint b = (m + 3u) % 4u;
    uint o = (m + 2u) % 4u;
    uint ia = ${wgLoad("ri[a]")};
    uint ib = ${wgLoad("ri[b]")};
    uint io = ${wgLoad("ri[o]")};
    if (ia == 0u || ib == 0u || io == 0u) { go = 0u; }
    else {
      vec4 ra = readings[3u * (f * P.cap + ia - 1u)];
      vec4 rb = readings[3u * (f * P.cap + ib - 1u)];
      vec4 ro = readings[3u * (f * P.cap + io - 1u)];
      vec2 Pa = ra.xy;
      vec2 Pb = rb.xy;
      vec2 Po = ro.xy;
      G[0] = vec4(Pa, Pb);
      G[1] = vec4(Po, Pa + Pb - Po);
      G[2] = vec4(ra.z, rb.z, ro.z, clamp(ra.z + rb.z - ro.z, 0.5 * min(ra.z, rb.z), 2.0 * max(ra.z, rb.z)));
      G[3] = vec4(float(m), quads[base + a].z, 0.0, 0.0);
      for (uint sd = 0u; sd < 2u; sd++) {
        vec2 P0 = (sd == 1u) ? Pb : Pa;
        float al = (sd == 1u) ? rb.w : ra.w;
        vec2 k = normalize(Po - P0);
        vec2 e = -dir(al + 0.25 * 3.14159265);
        G[4u + sd] = vec4(2.0 * dot(k, e) * e - k, normalize(Pa + Pb - Po - P0));
      }
      kept[0] = 0u;
      kept[1] = 0u;
    }
  }
  ${uniformLoad("uint go1", "go")}
  if (go1 == 0u) { return; }
  uint side = li / ${STATIONS}u;
  uint j = li % ${STATIONS}u;
  vec2 P0 = (side == 1u) ? G[0].zw : G[0].xy;
  vec2 Pm0 = G[1].zw;
  float u0 = (side == 1u) ? G[2].y : G[2].x;
  float um = G[2].w;
  float c = G[3].y;
  vec2 Q = 0.25 * (G[0].xy + G[0].zw + G[1].xy + Pm0);
  float L = distance(P0, Pm0);
  float t0 = (13.0 - c) * u0;
  float t1 = L - (13.0 - c) * um;
  bool span = t1 > t0 + 8.0 * u0;
  vec2 n0 = vec2(-G[4u + side].y, G[4u + side].x);
  if (dot(n0, P0 - Q) < 0.0) { n0 = -n0; }
  for (uint g = 0u; g < 2u; g++) {
    vec2 dd = (g == 1u) ? G[4u + side].zw : G[4u + side].xy;
    vec2 n = vec2(-dd.y, dd.x);
    if (dot(n, n0) < 0.0) { n = -n; }
    vec3 hit = vec3(0.0);
    if (span) {
      float t = mix(t0, t1, (float(j) + 0.5) / ${STATIONS}.0 * ${lit(PASSES[0][0])});
      float uu = mix(u0, um, t / L);
      hit = ringAt(f, P0 + t * dd + (c - 2.0) * uu * n, n, uu, ${lit(PASSES[0][1])});
    }
    st[li] = vec4(hit.xy, hit.z, 0.0);
    barrier();
    if (j == 0u) { fitSide(side, 0.5 * (u0 + um), (t1 - t0) * ${lit(PASSES[0][0])}, kept[side] + 1u); }
    barrier();
  }
  ${uniformLoad("uint k0", "kept[0]")}
  ${uniformLoad("uint k1", "kept[1]")}
  if (k0 == 0u || k1 == 0u) { return; }
  ${PASSES.slice(1).map(passAfter).join("\n  ")}
  if (li == 0u) {
    vec4 A = line[0];
    vec4 Bl = line[1];
    float den = A.z * Bl.w - A.w * Bl.z;
    if (abs(den) < 0.2) { return; }
    float s = ((Bl.x - A.x) * Bl.w - (Bl.y - A.y) * Bl.z) / den;
    vec2 R = A.xy + s * A.zw;
    vec2 ta = A.zw;
    if (dot(ta, G[0].xy - R) < 0.0) { ta = -ta; }
    vec2 tb = Bl.zw;
    if (dot(tb, G[0].zw - R) < 0.0) { tb = -tb; }
    vec2 Pm = R + (c - 2.0) * um * (ta + tb);
    if (distance(Pm, Pm0) > 0.25 * distance(G[0].xy, G[0].zw)) { return; }
    uint m = f2u(G[3].x);
    quads[base + m] = vec4(Pm, c, um);
    vec4 qm = quads[base + 4u];
    vec2 p0 = quads[base].xy;
    vec2 p1 = quads[base + 1u].xy;
    vec2 p2 = quads[base + 2u].xy;
    vec2 p3 = quads[base + 3u].xy;
    float uq = 0.25 * (distance(p0, p1) + distance(p1, p2) + distance(p2, p3) + distance(p3, p0)) / (P.modules[f2u(qm.x)] - 2.0 * quads[base].z);
    quads[base + 4u] = vec4(qm.x, uq, qm.z, 5.0);
  }
}
`;

// ai: A homography from four point pairs, closed form, no private array indexed at a run-time value (the WGSL's
// ai: HOMOGRAPHY, mul3's nine products written out by JS).
const Z9 = "float[9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)";
const MUL3 = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) =>
  `r[${3 * i + j}] = a[${3 * i}] * b[${j}] + a[${3 * i + 1}] * b[${3 + j}] + a[${3 * i + 2}] * b[${6 + j}];`)).join("\n  ");
const HOMOGRAPHY = /* glsl */ `
float[9] squareToQuad(vec2 q0, vec2 q1, vec2 q2, vec2 q3) {
  vec2 d1 = q1 - q2;
  vec2 d2 = q3 - q2;
  vec2 d3 = q0 - q1 + q2 - q3;
  float det = d1.x * d2.y - d2.x * d1.y;
  if (abs(det) < 1e-12) { return ${Z9}; }
  float g = (d3.x * d2.y - d2.x * d3.y) / det;
  float h = (d1.x * d3.y - d3.x * d1.y) / det;
  return float[9](q1.x - q0.x + g * q1.x, q3.x - q0.x + h * q3.x, q0.x,
                  q1.y - q0.y + g * q1.y, q3.y - q0.y + h * q3.y, q0.y, g, h, 1.0);
}
float[9] adjugate(float m[9]) {
  return float[9](m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                  m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                  m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]);
}
float[9] mul3(float a[9], float b[9]) {
  float r[9] = ${Z9};
  ${MUL3}
  return r;
}
float[9] solveH(vec2 src[4], vec2 dst[4], float M) {
  vec2 c = 0.25 * (dst[0] + dst[1] + dst[2] + dst[3]);
  float sc = max(0.25 * (distance(dst[0], c) + distance(dst[1], c) + distance(dst[2], c) + distance(dst[3], c)), 1e-3);
  float S[9] = squareToQuad(src[0] / M, src[1] / M, src[2] / M, src[3] / M);
  float D[9] = squareToQuad((dst[0] - c) / sc, (dst[1] - c) / sc, (dst[2] - c) / sc, (dst[3] - c) / sc);
  float hs[9] = mul3(D, adjugate(S));
  if (S[8] == 0.0 || D[8] == 0.0 || abs(hs[8]) < 1e-12) { return ${Z9}; }
  float hn[9] = ${Z9};
  for (int j = 0; j < 9; j++) { hn[j] = hs[j] / hs[8]; }
  float H[9] = ${Z9};
  for (int j = 0; j < 3; j++) {
    float s = (j < 2) ? 1.0 / M : 1.0;
    H[j] = (sc * hn[j] + c.x * hn[6 + j]) * s;
    H[3 + j] = (sc * hn[3 + j] + c.y * hn[6 + j]) * s;
    H[6 + j] = hn[6 + j] * s;
  }
  return H;
}
vec2 applyH(float H[9], vec2 m) {
  float w = H[6] * m.x + H[7] * m.y + H[8];
  return vec2(H[0] * m.x + H[1] * m.y + H[2], H[3] * m.x + H[4] * m.y + H[5]) / w;
}
vec2 cornerAt(uint k, float c, float M) {
  float x = (k == 1u || k == 2u) ? M - c : c;
  float y = (k >= 2u) ? M - c : c;
  return vec2(x, y);
}
uint slotOf(uint o, uint k) { return (o >= 4u) ? ((o + 4u - k) & 3u) : ((o + k) & 3u); }
vec4 atSlot(vec4 quad[4], uint s) {
  return (s >= 2u) ? ((s == 3u) ? quad[3] : quad[2]) : ((s == 1u) ? quad[1] : quad[0]);
}
float[9] hypH(vec4 quad[4], uint o, float M) {
  vec4 p0 = atSlot(quad, slotOf(o, 0u));
  vec4 p1 = atSlot(quad, slotOf(o, 1u));
  vec4 p2 = atSlot(quad, slotOf(o, 2u));
  vec4 p3 = atSlot(quad, slotOf(o, 3u));
  return solveH(vec2[4](cornerAt(0u, p0.z, M), cornerAt(1u, p1.z, M), cornerAt(2u, p2.z, M), cornerAt(3u, p3.z, M)),
                vec2[4](p0.xy, p1.xy, p2.xy, p3.xy), M);
}
`;


// ai: A quad's bend about the camera's axis in the map's own radial form (common.mjs
// ai: MAP toImage): an image point p is, undistorted, c + e (1 + k |e|^2 iR2), e = p - c, c the frame's axis (its dims'
// ai: third word, the centre where 0) and R its farthest corner from c; bend() the inverse, toImage's Newton steps.
const BEND = /* glsl */ `
vec3 bendAxis(Frame F) {
  vec2 LA = vec2(0.5 * float(F.w), 0.5 * float(F.h));
  if (F.size != 0u) { LA = vec2(0.5 * float(F.size & 0xffffu) - 16384.0, 0.5 * float(F.size >> 16u) - 16384.0); }
  vec2 fwh = vec2(float(F.w), float(F.h));
  float R = max(max(length(LA), length(vec2(fwh.x, 0.0) - LA)), max(length(vec2(0.0, fwh.y) - LA), length(fwh - LA)));
  return vec3(LA, 1.0 / max(R * R, 1.0));
}
vec2 unbend(vec2 p, vec3 ax, float k) { vec2 e = p - ax.xy; return ax.xy + e * (1.0 + k * dot(e, e) * ax.z); }
vec2 bend(vec2 u, vec3 ax, float k) {
  if (k == 0.0) { return u; }
  vec2 e = u - ax.xy;
  float g = k * dot(e, e) * ax.z;
  float s = 1.0;
  for (int i = 0; i < 3; i++) { float gs = g * s * s; s = s - (s * (1.0 + gs) - 1.0) / (1.0 + 3.0 * gs); }
  return ax.xy + e * s;
}
`;

// ai: TRACE: each quad's bend, a workgroup a (frame, quad) (COMPLETE's grid), 32 lanes a side from one of its ends, in
// ai: the marks' own module sizes (each corner's w, the classifier's, toward the other end's along the chord; not the
// ai: meta's u, which is in GATHER's guessed ring's modules): the ring's inner edge (depth 3) from 16 modules past the
// ai: mark, a station every TR_STEP modules, each
// ai: predicted from those found (quadratic in the station index), located across by the light (gap), dark (ring), light
// ai: (rim and margin) profile within three modules, accepted at a correlation of 0.6 off the window's edge; to the
// ai: chord's middle, or two misses. A station's 49 samples are fetched across its 32 lanes and its 25 offsets
// ai: correlated at once; every lane of a side then takes the same decision from shared memory, so the stations' barriers
// ai: stay in uniform control flow. Then the bend: k scanned -0.3 to 1.0 by 0.02 (a lane each) and refined by 0.002 about
// ai: the best, the traced sides unbent and each scored by its scatter's smaller over larger eigenvalue (the plumb line);
// ai: written to the quad's meta z where two sides traced four points or more, else 0. (A workgroup a frame with a lane
// ai: a side, each station's 49 samples one after another, cost the S26 0.11 to 0.17 ms a frame; this, 0.01.)
const TR_MAX = 24, TR_STEP = 12, TR_L = 32;
export const TRACE = VERSION + SAMPLERLESS + DIMS + CONVERT + LEVEL_BINDINGS + LEVELS + BEND + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) buffer Quads { vec4 quads[]; };
shared vec2 tp[8][${TR_MAX}];
shared uint tn[8];
shared float prof[8][49];
shared float nc[8][25];
shared uint alive[8];
shared vec2 best[66];
float tmpl(uint k) { float a = -2.0 + 0.25 * float(k) + 0.125; return (a >= 0.0 && a < 2.0) ? -4.0 / 3.0 : 2.0 / 3.0; }
float ncc(uint tr, int o) {
  float m = 0.0;
  for (uint k = 0u; k < 24u; k++) { m += prof[tr][uint(int(k) + 12 + o)]; }
  m /= 24.0;
  float sv = 0.0;
  float sx = 0.0;
  for (uint k = 0u; k < 24u; k++) { float d = prof[tr][uint(int(k) + 12 + o)] - m; sv += d * d; sx += d * tmpl(k); }
  return sv > 1e-9 ? sx / (sqrt(sv) * 4.6188022) : 0.0;
}
float plumb(float k, vec3 ax) {
  float e = 0.0;
  uint sides = 0u;
  for (uint sd = 0u; sd < 4u; sd++) {
    uint t0 = 2u * sd;
    uint n0 = tn[t0];
    uint n1 = tn[t0 + 1u];
    if (n0 + n1 < 4u) { continue; }
    vec2 mu = vec2(0.0);
    for (uint i = 0u; i < n0; i++) { mu += unbend(tp[t0][i], ax, k); }
    for (uint i = 0u; i < n1; i++) { mu += unbend(tp[t0 + 1u][i], ax, k); }
    mu /= float(n0 + n1);
    vec3 S = vec3(0.0);
    for (uint i = 0u; i < n0; i++) { vec2 d = unbend(tp[t0][i], ax, k) - mu; S += vec3(d.x * d.x, d.y * d.y, d.x * d.y); }
    for (uint i = 0u; i < n1; i++) { vec2 d = unbend(tp[t0 + 1u][i], ax, k) - mu; S += vec3(d.x * d.x, d.y * d.y, d.x * d.y); }
    float tr = S.x + S.y;
    float rt = sqrt(max(0.25 * tr * tr - (S.x * S.y - S.z * S.z), 0.0));
    float l1 = 0.5 * tr + rt;
    if (l1 > 0.0) { e += (0.5 * tr - rt) / l1; }
    sides++;
  }
  return sides >= 2u ? e : 3.0e38;
}
void main() {
  uint wq = gl_WorkGroupID.x;
  uint f = wq / ${K}u;
  uint base = 5u * wq;
  uint li = gl_LocalInvocationIndex;
  uint tr = li / ${TR_L}u;
  uint j = li % ${TR_L}u;
  Frame F = frames[f];
  vec4 qm = quads[base + 4u];
  if (F.valid > 1u || F.valid == 0u || !(qm.w == 4.0 || qm.w == 5.0)) { return; }
  vec3 ax = bendAxis(F);
  uint side = tr >> 1u;
  vec4 a4 = quads[base + side];
  vec4 b4 = quads[base + ((side + 1u) & 3u)];
  if ((tr & 1u) == 1u) { vec4 t4 = a4; a4 = b4; b4 = t4; }
  vec2 A = a4.xy;
  vec2 B = b4.xy;
  vec2 g = 0.25 * (quads[base].xy + quads[base + 1u].xy + quads[base + 2u].xy + quads[base + 3u].xy);
  float dc = a4.z;
  float ua = a4.w;
  float ub = b4.w;
  float len = distance(A, B);
  vec2 U = (B - A) / max(len, 1e-6);
  vec2 n = vec2(-U.y, U.x);
  if (dot(n, 0.5 * (A + B) - g) < 0.0) { n = -n; }
  vec2 p0 = A + U * (16.0 - dc) * ua + n * (dc - 3.0) * ua;
  uint np = 0u;
  uint miss = 0u;
  bool live = ua > 0.0;
  for (uint st = 0u; st < ${TR_MAX}u; st++) {
    vec2 P = vec2(0.0);
    vec2 tv = vec2(0.0);
    vec2 v = vec2(0.0);
    if (live) {
      float t = 16.0 + ${TR_STEP}.0 * float(st);
      if (np < 2u) {
        P = A + U * (t - dc) * ua + n * (dc - 3.0) * ua;
        tv = U;
        if (np == 1u) { P += tp[tr][0] - p0; }
      } else if (np == 2u) {
        P = 2.0 * tp[tr][1] - tp[tr][0];
        tv = tp[tr][1] - tp[tr][0];
      } else {
        P = 3.0 * tp[tr][np - 1u] - 3.0 * tp[tr][np - 2u] + tp[tr][np - 3u];
        tv = P - tp[tr][np - 1u];
      }
      float tl = length(tv);
      if (!(tl > 0.0)) { live = false; } else {
        tv /= tl;
        v = vec2(-tv.y, tv.x);
        if (dot(v, n) < 0.0) { v = -v; }
        if (dot(P - A, U) > 0.5 * len + 3.0 * ua) { live = false; }
        if (P.x < 2.0 || P.y < 2.0 || P.x > float(F.w) - 3.0 || P.y > float(F.h) - 3.0) { live = false; }
      }
    }
    if (live) {
      float u = ua + (ub - ua) * clamp(dot(P - A, U) / len, 0.0, 1.0);
      for (uint k = j; k < 49u; k += ${TR_L}u) {
        float a = -5.0 + 0.25 * float(k);
        prof[tr][k] = look(0u, f, P + (v * a - tv) * u) + look(0u, f, P + v * a * u) + look(0u, f, P + (v * a + tv) * u);
      }
    }
    barrier();
    if (live && j < 25u) { nc[tr][j] = ncc(tr, int(j) - 12); }
    barrier();
    if (live) {
      float bc = -2.0;
      int bo = 0;
      for (int o = -12; o <= 12; o++) { float c = nc[tr][o + 12]; if (c > bc) { bc = c; bo = o; } }
      if (bc < 0.6 || bo == -12 || bo == 12) { miss++; if (miss > 2u) { live = false; } } else {
        miss = 0u;
        float l = nc[tr][bo + 11];
        float r = nc[tr][bo + 13];
        float dd = l - 2.0 * bc + r;
        float off = float(bo) + ((dd < 0.0) ? 0.5 * (l - r) / dd : 0.0);
        float u = ua + (ub - ua) * clamp(dot(P - A, U) / len, 0.0, 1.0);
        if (j == 0u) { tp[tr][np] = P + v * (0.25 * off) * u; }
        np++;
      }
    }
    if (j == 0u) { alive[tr] = live ? 1u : 0u; }
    barrier();
    uint alive0 = 0u;
    for (uint t = 0u; t < 8u; t++) { alive0 |= alive[t]; }
    if (alive0 == 0u) { break; }
  }
  if (j == 0u) { tn[tr] = np; }
  barrier();
  if (li < 66u) { float k = -0.3 + 0.02 * float(li); best[li] = vec2(plumb(k, ax), k); }
  barrier();
  if (li == 0u) {
    vec2 b = best[0];
    for (uint i = 1u; i < 66u; i++) { if (best[i].x < b.x) { b = best[i]; } }
    best[0] = vec2(b.x, b.x < 3.0e38 ? b.y : 0.0);
  }
  barrier();
  float k0 = best[0].y;
  barrier();
  if (li < 21u) { float k = k0 - 0.02 + 0.002 * float(li); best[li] = vec2(plumb(k, ax), k); }
  barrier();
  if (li == 0u) {
    vec2 b = vec2(3.0e38, k0);
    for (uint i = 0u; i < 21u; i++) { if (best[i].x < b.x) { b = best[i]; } }
    quads[base + 4u] = vec4(qm.x, qm.y, b.x < 3.0e38 ? b.y : 0.0, qm.w);
  }
}
`;

// ai: F6a: every hypothesis through the track read, a workgroup each (the WGSL's SCORE).
export const SCORE = VERSION + SAMPLERLESS + DIMS + CONVERT + LEVEL_BINDINGS + LEVELS + HOMOGRAPHY + BEND + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) readonly buffer Quads { vec4 quads[]; };
layout(set = 0, binding = 8, std430) readonly buffer PlanPoints { vec4 plan[]; };
layout(set = 0, binding = 9, std430) buffer Score { float score[]; };
layout(set = 0, binding = 10, std140) uniform Plan { uvec4 at[${RING_COUNT}]; } PL;
shared vec4 sums[64];
shared float H[9];
bool near(vec2 m, float M) {
  float along = (m.x < 7.0 || m.x > M - 7.0) ? m.y : m.x;
  return min(along, M - along) <= ${NEAR}.0;
}
shared uint ok;
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x / ${HYP}u;
  if (frames[f].valid > 1u) { return; }
  uint hyp = wg.x % ${HYP}u;
  uint h0 = hyp % ${K * ORIENT * RING_COUNT}u;
  uint bendBit = hyp / ${K * ORIENT * RING_COUNT}u;
  uint qk = h0 / ${ORIENT * RING_COUNT}u;
  uint o = (h0 / ${RING_COUNT}u) % ${ORIENT}u;
  uint rg = h0 % ${RING_COUNT}u;
  uint base = 5u * (f * ${K}u + qk);
  vec4 qm = quads[base + 4u];
  uvec4 at = PL.at[rg];
  float M = float(at.w);
  // ai: the bend bit: 0 the marks' homography alone, 1 the quad's bend (TRACE), the marks unbent for H and every read
  // ai: bent back
  float kb = bendBit == 0u ? 0.0 : qm.z;
  vec3 ax = bendAxis(frames[f]);
  if (li == 0u) {
    ok = (frames[f].valid != 0u && (qm.w == 4.0 || qm.w == 5.0)) ? 1u : 0u;
    if (ok == 1u) {
      vec4 c4[4] = vec4[4](quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u]);
      if (kb != 0.0) { for (int k = 0; k < 4; k++) { c4[k].xy = unbend(c4[k].xy, ax, kb); } }
      float h[9] = hypH(c4, o, M);
      for (int j = 0; j < 9; j++) { H[j] = h[j]; }
      if (h[8] == 0.0) { ok = 0u; }
    }
  }
  ${uniformLoad("uint live", "ok")}
  if (live == 0u) {
    if (li == 0u) { score[f * ${HYP}u + hyp] = -1.0; }
    return;
  }
  float Hl[9] = ${Z9};
  for (int j = 0; j < 9; j++) { Hl[j] = H[j]; }
  float u = distance(applyH(Hl, vec2(0.5 * M, 0.5 * M)), applyH(Hl, vec2(0.5 * M + 1.0, 0.5 * M)));
  uint l = levelFor(u);
  vec4 acc = vec4(0.0);
  for (uint s = li; s < at.y; s += 64u) {
    uint b = at.x + 4u * s;
    if (!near(plan[b].xy, M)) { continue; }
    vec2 q0 = bend(applyH(Hl, plan[b].xy), ax, kb);
    vec2 q1 = bend(applyH(Hl, plan[b + 1u].xy), ax, kb);
    vec2 q2 = bend(applyH(Hl, plan[b + 2u].xy), ax, kb);
    vec2 q3 = bend(applyH(Hl, plan[b + 3u].xy), ax, kb);
    if (!(onFrame(f, q0) && onFrame(f, q1) && onFrame(f, q2) && onFrame(f, q3))) { continue; }
    acc += vec4(look(l, f, q0), look(l, f, q1), look(l, f, q2) + look(l, f, q3), 1.0);
  }
  sums[li] = acc;
  barrier();
  for (uint st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    barrier();
  }
  vec4 lev = sums[0] / max(sums[0].w, 1.0);
  float thr = 0.5 * lev.z;
  float con = lev.y - lev.x;
  barrier();
  float q = 0.0;
  float nq = 0.0;
  for (uint j = li; j < at.z; j += 64u) {
    vec4 pt = plan[at.x + 4u * at.y + j];
    if (!near(pt.xy, M)) { continue; }
    vec2 qp = bend(applyH(Hl, pt.xy), ax, kb);
    if (!onFrame(f, qp)) { continue; }
    q += pt.z * clamp((thr - look(l, f, qp)) / max(con, 1e-3), -0.5, 0.5);
    nq += 1.0;
  }
  sums[li] = vec4(q, nq, 0.0, 0.0);
  barrier();
  for (uint st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    barrier();
  }
  if (li == 0u) { score[f * ${HYP}u + hyp] = (con >= 0.08) ? sums[0].x / max(sums[0].y, 1.0) : -1.0; }
}
`;

// ai: F6b: the frame's answer, one workgroup a frame (the WGSL's PICK).
export const PICK = VERSION + DIMS + HOMOGRAPHY + BEND + /* glsl */ `
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Quads { vec4 quads[]; };
layout(set = 0, binding = 2, std430) readonly buffer Score { float score[]; };
layout(set = 0, binding = 3, std430) buffer Maps { float maps[][16]; };
layout(set = 0, binding = 4, std430) buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 5, std430) buffer Result { float result[][32]; };
layout(set = 0, binding = 6, std140) uniform Pick { float tau; uint p0; uint p1; uint p2; vec4 modules; } P;
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint f = wg.x;
  if (frames[f].valid > 1u) { return; }
  float best = -2.0;
  uint bh = 0u;
  for (uint h = 0u; h < ${HYP}u; h++) {
    float s = score[f * ${HYP}u + h];
    if (s > best) { best = s; bh = h; }
  }
  bool found = frames[f].valid != 0u && best >= P.tau;
  uint bh0 = bh % ${K * ORIENT * RING_COUNT}u;
  uint bendBit = bh / ${K * ORIENT * RING_COUNT}u;
  uint qk = bh0 / ${ORIENT * RING_COUNT}u;
  uint o = (bh0 / ${RING_COUNT}u) % ${ORIENT}u;
  uint rg = bh0 % ${RING_COUNT}u;
  maps[f] = ${zeros(16)};
  result[f] = float[32](${Array(32).fill("0.0").join(", ")});
  result[f][1] = best;
  if (found) {
    uint base = 5u * (f * ${K}u + qk);
    vec4 quad[4] = vec4[4](quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u]);
    // ai: a bent winner: H of the unbent marks, the bend in the map's radial slot (toImage reads it)
    float kb = bendBit == 0u ? 0.0 : quads[base + 4u].z;
    vec3 ax = bendAxis(frames[f]);
    vec4 qu[4] = quad;
    if (kb != 0.0) { for (int k = 0; k < 4; k++) { qu[k].xy = unbend(qu[k].xy, ax, kb); } }
    float H[9] = hypH(qu, o, P.modules[rg]);
    if (kb != 0.0) { maps[f][9] = kb; maps[f][10] = ax.x; maps[f][11] = ax.y; maps[f][12] = ax.z; }
    ${Array.from({ length: 9 }, (_, j) => `maps[f][${j}] = H[${j}]; result[f][${8 + j}] = H[${j}];`).join("\n    ")}
    for (uint k = 0u; k < 4u; k++) {
      vec4 p = atSlot(quad, slotOf(o, k));
      result[f][17u + 2u * k] = p.x;
      result[f][18u + 2u * k] = p.y;
      result[f][25u + k] = p.z;
    }
    result[f][0] = 1.0;
    result[f][2] = float(rg);
    result[f][3] = float(o);
    result[f][4] = float(qk);
  }
  sel[f] = uvec4(rg, found ? 1u : 0u, 0u, 0u);
}
`;

// ai: The finder's second round's frame records and its frames' resets, and its grids (the WGSL's REDO).
export const REDO = VERSION + ATOMICS + DIMS + /* glsl */ `
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 2, std430) buffer Frames2 { Frame frames2[]; };
layout(set = 0, binding = 3, std430) buffer Acc { uint acc[]; };
layout(set = 0, binding = 4, std430) buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 5, std430) buffer Cand { vec4 cand[]; };
layout(set = 0, binding = 6, std430) buffer Quads { vec4 quads[]; };
` + P5(7) + /* glsl */ `
layout(set = 0, binding = 8, std430) buffer Args2 { uint args2[]; };
layout(set = 0, binding = 9, std140) uniform R2T { uvec4 d[${R2_N}]; } R2;
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  Frame F = frames[f];
  bool again = F.valid == 1u && sel[f].y == 0u;
  if (li == 0u) { frames2[f] = Frame(F.w, F.h, F.size, again ? 1u : (F.valid != 0u ? 2u : 0u)); }
  if (!again) { return; }
  if (li < ${R2_N}u) {
    uvec4 t = R2.d[li];
    for (uint c = 0u; c < 3u; c++) {
      if (t.w > 2u) { break; }
      if (c == t.w) { atomicMax(args2[3u * li + c], t[c] * (f + 1u)); } else { ${devStore("args2[3u * li + c]", "t[c]")}; }
    }
  }
  uint na = ${RING_COUNT}u * P.gx * P.gy;
  for (uint i = li; i < na; i += 256u) { acc[f * na + i] = 0u; }
  uint nr = 3u * P.cap;
  for (uint i = li; i < nr; i += 256u) { readings[f * nr + i] = vec4(0.0); }
  if (li < ${K}u) { cand[f * ${K}u + li] = vec4(0.0); }
  if (li < ${5 * K}u) { quads[f * ${5 * K}u + li] = vec4(0.0); }
}
`;
