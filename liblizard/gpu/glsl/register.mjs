// F7, registration, in GLSL: the WGSL register.mjs (archived) statement for statement (its comments say what each stage does and
// why): NODES (nodesSource, the border nodes measured by normalised cross-correlation), REFIT (the homography by
// Gauss-Newton), REFIT_POST (the nodes' residuals, each side's curve), and the pieces
// the anchors' stages share (LENS, the lens fit's rows, SIDE_FITS, SIDE_DECL, RING; LENS and RING sample.mjs's COONS too).
// ai: Bindings, workgroup sizes, dispatches and every buffer's layout are the WGSL's (the host's bind group layouts are
// ai: unchanged): storage arrays std430, uniforms std140. Every solve is straight-line with literal indices, generated
// ai: in JS as the WGSL's is (the S26's Adreno returned zeros from a private array indexed at run time). A WGSL select
// ai: of arms that are not plain values is written with its arms in temporaries, so both are evaluated as WGSL's are
// ai: and the choice stays a select. Workgroup memory is not zeroed: every workgroup variable here is written before
// ai: it is read. WGSL's storageBarrier() is common.mjs STORAGE_BARRIER.
import { VERSION, SAMPLERLESS, DIMS, CONVERT, MAP, LEVEL_BINDINGS, zeros, uniformLoad, STORAGE_BARRIER } from "./common.mjs";
import {
  WINDOW, SUB, NODES_MAX, LIGHT_MAX, LENS_SLOTS, RESID, ROUND, latticeSide, WMIN, DELTA, ITERS, SIDE_MAX, SIDE_NODES, LENS_P, lensIdx,
} from "../plan/register.mjs";

const TMPL_MAX = WINDOW * WINDOW * SUB * SUB;
const WG = 256;
const ITEMS = 512;
const unroll = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join("\n");

// ai: The pyramid's levels and the frames, bindings 0 to 6, and registration's own level reader (the WGSL register.mjs (archived)
// ai: LEVELS: a level's four taps in one switch).
const LEVELS = LEVEL_BINDINGS + /* glsl */ `struct Lvl { uint l; int w; int h; uint base; uint stride; float scale; };
Lvl lvlOf(uint l, uint f) {
  Frame F = frames[f];
  Lvl L = Lvl(l, int((F.w + (1u << l) - 1u) >> l), int((F.h + (1u << l) - 1u) >> l), 0u, 0u, 1.0 / float(1u << l));
  if (l > 0u) { uvec4 s = LD.d[l]; L.base = f * s.y * s.x; L.stride = s.x; }
  return L;
}
vec4 tap4(Lvl L, uint f, int x0, int x1, int y0, int y1) {
  switch (L.l) {
    case 0u: {
      return vec4(texelFetch(img, ivec3(x0, y0, int(f)), 0).r, texelFetch(img, ivec3(x1, y0, int(f)), 0).r,
        texelFetch(img, ivec3(x0, y1, int(f)), 0).r, texelFetch(img, ivec3(x1, y1, int(f)), 0).r);
    }
    case 1u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return vec4(lv1[r0 + uint(x0)], lv1[r0 + uint(x1)], lv1[r1 + uint(x0)], lv1[r1 + uint(x1)]); }
    case 2u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return vec4(lv2[r0 + uint(x0)], lv2[r0 + uint(x1)], lv2[r1 + uint(x0)], lv2[r1 + uint(x1)]); }
    case 3u: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return vec4(lv3[r0 + uint(x0)], lv3[r0 + uint(x1)], lv3[r1 + uint(x0)], lv3[r1 + uint(x1)]); }
    default: { uint r0 = L.base + uint(y0) * L.stride; uint r1 = L.base + uint(y1) * L.stride; return vec4(lv4[r0 + uint(x0)], lv4[r0 + uint(x1)], lv4[r1 + uint(x0)], lv4[r1 + uint(x1)]); }
  }
}
float bilinear(Lvl L, uint f, vec2 p) {
  vec2 q = p * L.scale - vec2(0.5);
  vec2 q0 = floor(q);
  vec2 a = q - q0;
  ivec2 i = f2i(q0);
  int x0 = clamp(i.x, 0, L.w - 1);
  int x1 = clamp(i.x + 1, 0, L.w - 1);
  int y0 = clamp(i.y, 0, L.h - 1);
  int y1 = clamp(i.y + 1, 0, L.h - 1);
  vec4 v = tap4(L, f, x0, x1, y0, y1);
  return mix(mix(v.x, v.y, a.x), mix(v.z, v.w, a.x), a.y);
}
uint levelFor(float u) { return f2u(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
`;

// ai: NODES and its centred form (the WGSL register.mjs (archived) nodesSource: the same options, the same kernel for a round).
export const nodesSource = ({ centred = false, round = ROUND, chunk = TMPL_MAX / 2, items = ITEMS } = {}) => {
  const grid = latticeSide(round), lattice = grid > 0;
  return VERSION + SAMPLERLESS + DIMS + CONVERT + LEVELS + MAP + /* glsl */ `
layout(local_size_x = ${WG}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) readonly buffer Nodes { vec4 nodes[]; };
layout(set = 0, binding = 8, std430) readonly buffer Tmpl { vec4 tmpl[]; };
layout(set = 0, binding = 9, std430) readonly buffer Maps { float maps[][16]; };
layout(set = 0, binding = 10, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 11, std430) buffer Meas { vec4 meas[]; };
layout(set = 0, binding = 12, std140) uniform Round { uint steps; uint stride; uint p0; uint p1; float step; float p2; float p3; float p4; uvec4 count; uvec4 base; uvec4 light[${LIGHT_MAX / 4}]; } R;
const uint CHUNK = ${chunk}u;
const uint ITEMS = ${items}u;
const uint WG = ${WG}u;
${lattice ? `const uint GRID = ${grid}u;
shared uint sp[CHUNK];
shared float gv[${grid * grid}];` : `shared vec4 sp[CHUNK];
shared vec4 sj[CHUNK];`}
shared float psv[ITEMS];
shared float psv2[ITEMS];
shared float psev[ITEMS];
shared float ncc[225];
shared uint best;
${centred ? "shared vec4 wm0;\n" : ""}void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  uint n = wg.x;
  uvec4 pick = sel[f];
  uint count = R.count[pick.x];
  if (frames[f].valid == 0u || pick.y == 0u || n >= count) { return; }
  vec4 node = nodes[R.base[pick.x] + n];
  float M[16] = maps[f];
${centred ? `  if (li == 0u) { wm0 = meas[f * ${NODES_MAX}u + n]; }
  ${uniformLoad("vec4 m0", "wm0")}
  if (m0.w != 2.0) { return; }
  vec2 cm = toImage(M, node.xy);
  vec2 sh = m0.xy - cm;
  vec2 c = cm + sh;
  float u = 0.5 * (distance(toImage(M, node.xy + vec2(1.0, 0.0)), cm) + distance(toImage(M, node.xy + vec2(0.0, 1.0)), cm));` : `  vec2 c = toImage(M, node.xy);
  float u = 0.5 * (distance(toImage(M, node.xy + vec2(1.0, 0.0)), c) + distance(toImage(M, node.xy + vec2(0.0, 1.0)), c));`}
  Frame F0 = frames[f];
  if (c.x < 2.0 * u || c.y < 2.0 * u || c.x > float(F0.w) - 2.0 * u || c.y > float(F0.h) - 2.0 * u) {
    if (li == 0u) { meas[f * ${NODES_MAX}u + n] = vec4(c, 0.0, 0.0); }
    return;
  }
  Lvl L = lvlOf(levelFor(u), f);
  uint first = f2u(node.z);
  uint pts = f2u(node.w);
  uint P = (pts + R.stride - 1u) / R.stride;
  int side = int(R.steps);
  uint gridW = uint(2 * side + 1);
  uint O = gridW * gridW;
  uint G = max(1u, ITEMS / O);
  uint items = O * G;
${lattice ? `  float spacing = ${round[1] === SUB * SUB ? "1.0" : `1.0 / ${SUB}.0`};
  uint ratioN = ${Math.round((round[1] === SUB * SUB ? 1 : 1 / SUB) / round[2])}u;
  vec2 origin = floor(node.xy - ${WINDOW / 2}.0) + vec2(0.5 / ${SUB}.0);` : ""}
  if (li == 0u) { best = 0u; }
  uint nq = R.base[pick.x] + n;
  float tse = float(R.light[nq >> 2u][nq & 3u]);
  float tse2 = tse;
${lattice ? `  for (uint q = li; q < GRID * GRID; q += WG) {
    vec2 g = vec2(float(q % GRID), float(q / GRID));
    gv[q] = bilinear(L, f, toImage(M, origin + (g - float(side)) * R.step)${centred ? " + sh" : ""});
  }` : ""}
  for (uint item = li; item < items; item += WG) { psv[item] = 0.0; psv2[item] = 0.0; psev[item] = 0.0; }
  for (uint c0 = 0u; c0 < P; c0 += CHUNK) {
    uint c1 = min(P, c0 + CHUNK);
    for (uint j = c0 + li; j < c1; j += WG) {
      vec4 t = tmpl[first + j * R.stride];
${lattice ? `      uvec2 g = f2u(roundEven((t.xy - origin) / spacing)) * ratioN;
      sp[j - c0] = g.x | (g.y << 8u) | (f2u(t.z) << 16u);` : `      {
        vec2 p = toImage(M, t.xy)${centred ? " + sh" : ""};
        float w = M[6] * t.x + M[7] * t.y + M[8];
        vec2 uu = vec2(M[0] * t.x + M[1] * t.y + M[2], M[3] * t.x + M[4] * t.y + M[5]) / w;
        vec2 jx = vec2(M[0] - uu.x * M[6], M[3] - uu.y * M[6]) / w;
        vec2 jy = vec2(M[1] - uu.x * M[7], M[4] - uu.y * M[7]) / w;
        sp[j - c0] = vec4(p, t.z, 0.0);
        sj[j - c0] = vec4(jx.x, jy.x, jx.y, jy.y);
      }`}
    }
    barrier();
    for (uint item = li; item < items; item += WG) {
      uint o = item % O;
      uint g = item / O;
      uint jl = g + G * ((c0 - g + G - 1u) / G);
      uint j = c0 > g ? jl : g;
      float sv = 0.0;
      float sv2 = 0.0;
      float sev = 0.0;
${lattice ? `
        uint d = (o % gridW) + (o / gridW) * GRID;
        for (; j < c1; j += G) {
          uint s = sp[j - c0];
          float v = gv[(s & 255u) + ((s >> 8u) & 255u) * GRID + d];
          sv += v; sv2 += v * v; sev += ((s >> 16u) & 1u) != 0u ? v : 0.0;
        }
` : `
        vec2 off = R.step * vec2(float(int(o % gridW) - side), float(int(o / gridW) - side));
        for (; j + G < c1; j += 2u * G) {
          vec4 s0 = sp[j - c0];
          vec4 s1 = sp[j - c0 + G];
          vec4 J0 = sj[j - c0];
          vec4 J1 = sj[j - c0 + G];
          float v0 = bilinear(L, f, s0.xy + vec2(J0.x * off.x + J0.y * off.y, J0.z * off.x + J0.w * off.y));
          float v1 = bilinear(L, f, s1.xy + vec2(J1.x * off.x + J1.y * off.y, J1.z * off.x + J1.w * off.y));
          sv += v0 + v1; sv2 += v0 * v0 + v1 * v1; sev += s0.z * v0 + s1.z * v1;
        }
        if (j < c1) {
          vec4 s = sp[j - c0];
          vec4 J = sj[j - c0];
          float v = bilinear(L, f, s.xy + vec2(J.x * off.x + J.y * off.y, J.z * off.x + J.w * off.y));
          sv += v; sv2 += v * v; sev += s.z * v;
        }
`}
      psv[item] += sv;
      psv2[item] += sv2;
      psev[item] += sev;
    }
    barrier();
  }
  for (uint o = li; o < O; o += WG) {
    float sv = 0.0;
    float sv2 = 0.0;
    float sev = 0.0;
    for (uint g = 0u; g < G; g++) { uint i = g * O + o; sv += psv[i]; sv2 += psv2[i]; sev += psev[i]; }
    float k = float(P);
    float cov = sev - tse * sv / k;
    float den = sqrt(max((tse2 - tse * tse / k) * (sv2 - sv * sv / k), 1e-12));
    ncc[o] = cov / den;
    atomicMax(best, (f2u(clamp(ncc[o] + 1.0, 0.0, 1.999) * 4000000.0) << 8u) | o);
  }
  barrier();
  if (li == 0u) {
    uint b = best & 255u;
    int bx = int(b % gridW);
    int by = int(b / gridW);
    float v = ncc[b];
    float dx = 0.0;
    float dy = 0.0;
    if (bx > 0 && bx < 2 * side) {
      float a = ncc[b - 1u];
      float z = ncc[b + 1u];
      float e = a + z - 2.0 * v;
      if (e < 0.0) { dx = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    if (by > 0 && by < 2 * side) {
      float a = ncc[b - gridW];
      float z = ncc[b + gridW];
      float e = a + z - 2.0 * v;
      if (e < 0.0) { dy = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    vec2 off = R.step * vec2(float(bx - side) + dx, float(by - side) + dy);
    meas[f * ${NODES_MAX}u + n] = vec4(toImage(M, node.xy + off)${centred ? " + sh, v, 3.0" : ", v, 1.0"});
  }
}
`;
};
export const NODES = nodesSource();

// ai: Gauss-Jordan with partial pivoting as straight-line GLSL (the WGSL register.mjs (archived) gaussJordan): the pivot chosen and
// ai: swapped in by selects, every index a literal.
function gaussJordan(N, rows, cols, eps, ok) {
  const s = [];
  for (let col = 0; col < rows; col++) {
    s.push(`{ int piv = ${col}; float pv = abs(${N}[${col}][${col}]);`);
    for (let r = col + 1; r < rows; r++) s.push(`{ float a = abs(${N}[${r}][${col}]); piv = a > pv ? ${r} : piv; pv = max(pv, a); }`);
    for (let r = col + 1; r < rows; r++) {
      s.push(`{ bool sw = piv == ${r};`);
      for (let j = col; j < cols; j++) s.push(`{ float a = ${N}[${col}][${j}]; float b = ${N}[${r}][${j}]; ${N}[${col}][${j}] = sw ? b : a; ${N}[${r}][${j}] = sw ? a : b; }`);
      s.push("}");
    }
    s.push(`${ok} = ${ok} && abs(${N}[${col}][${col}]) >= ${eps};`);
    for (let r = 0; r < rows; r++) {
      if (r === col) continue;
      s.push(`{ float fc = ${N}[${r}][${col}] / ${N}[${col}][${col}];`);
      for (let j = col; j < cols; j++) s.push(`${N}[${r}][${j}] -= fc * ${N}[${col}][${j}];`);
      s.push("}");
    }
    s.push("}");
  }
  return s.join("\n");
}
const jacobian = unroll(8, (k) => {
  const arr = (sign) => `float[8](${Array.from({ length: 8 }, (_, j) => (j === k ? `t[${j}] ${sign} hs${k}` : `t[${j}]`)).join(", ")})`;
  return `float hs${k} = 1e-3;\nvec2 J${k} = (fwd(${arr("+")}, m, c, S) - fwd(${arr("-")}, m, c, S)) / (2.0 * hs${k});`;
});
const accumulate = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 8; a++) {
    s.push(`jr[${a}] += w * dot(J${a}, r);`);
    for (let b = a; b < 8; b++) s.push(`jj[${q++}] += w * dot(J${a}, J${b});`);
  }
  return s.join("\n");
})();
const normal = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 8; a++) for (let b = a; b < 8; b++) { s.push(`N[${a}][${b}] = A[${q}]; N[${b}][${a}] = A[${q}];`); q++; }
  return s.join("\n");
})();
const SP_N = 2 * SIDE_MAX - 1, SIDE_SUMS = SP_N + 2 * SIDE_MAX;
// ai: SIDE_FITS's workgroup memory (the WGSL register.mjs (archived) SIDE_DECL).
export const SIDE_DECL = /* glsl */ `
shared float sfP[${SIDE_NODES}][${SP_N}];
shared vec4 sfW[${SIDE_NODES}];
shared float sfS[${SIDE_SUMS}];
shared float sfC[${SIDE_MAX + 1}][${2 * SIDE_MAX}];
shared float sfR[${SIDE_NODES}][${SIDE_MAX}];
shared float sfA[${SIDE_MAX}];
shared uint sfN;
`;
// ai: the WGSL register.mjs (archived) gaussJordanIn: the masked elimination, a lane's count D, in gaussJordan's order.
function gaussJordanIn(N, D, eps, ok) {
  const R = SIDE_MAX, C = SIDE_MAX + 2, s = [];
  for (let col = 0; col < R; col++) {
    s.push(`{ bool ac = ${col}u < ${D}; int piv = ${col}; float pv = abs(${N}[${col}][${col}]);`);
    for (let r = col + 1; r < R; r++) s.push(`{ bool m = ${r}u < ${D}; float a = abs(${N}[${r}][${col}]); piv = (m && a > pv) ? ${r} : piv; float pm = max(pv, a); pv = m ? pm : pv; }`);
    for (let r = col + 1; r < R; r++) {
      s.push(`{ bool sw = ac && piv == ${r};`);
      for (let j = col; j < C; j++) s.push(`{ float a = ${N}[${col}][${j}]; float b = ${N}[${r}][${j}]; ${N}[${col}][${j}] = sw ? b : a; ${N}[${r}][${j}] = sw ? a : b; }`);
      s.push("}");
    }
    s.push(`{ bool okn = ${ok} && abs(${N}[${col}][${col}]) >= ${eps}; ${ok} = ac ? okn : ${ok}; }`);
    for (let r = 0; r < R; r++) {
      if (r === col) continue;
      s.push(`{ float fq = ${N}[${r}][${col}] / ${N}[${col}][${col}]; float fc = (ac && ${r}u < ${D}) ? fq : 0.0;`);
      for (let j = col; j < C; j++) s.push(`${N}[${r}][${j}] -= fc * ${N}[${col}][${j}];`);
      s.push("}");
    }
    s.push("}");
  }
  return s.join("\n");
}

// ai: The lens about the camera's axis (the WGSL register.mjs (archived) LENS): a stage including it binds `resid` first.
export const LENS = /* glsl */ `
struct Lens { vec4 c; vec4 a; vec4 t0; vec4 t1; vec4 t2; };
Lens lensOf(uint f) {
  uint b = f * ${RESID}u + ${NODES_MAX}u;
  return Lens(resid[b], resid[b + 1u], resid[b + 2u], resid[b + 3u], resid[b + 4u]);
}
vec2 lensAt(Lens L, vec2 p) {
  vec2 d = (p - L.c.xy) * L.c.z;
  vec2 q = (p - L.a.xy) * L.a.z;
  float t = (dot(q, q) - L.c.w) * L.a.w;
  return vec2(L.t0.x * d.x + L.t0.y * d.y + L.t1.x, L.t0.z * d.x + L.t0.w * d.y + L.t1.y)
    - (L.t1.z * d.x + L.t1.w * d.y) * d + (t * (L.t2.x + t * (L.t2.y + t * L.t2.z))) * q;
}
`;
const lensRows = () => {
  const rx = ["d.x", "d.y", null, null, "1.0", null, "(-d.x * d.x)", "(-d.y * d.x)", "(t * q.x)", "(t2 * q.x)", "(t3 * q.x)"];
  const ry = [null, null, "d.x", "d.y", null, "1.0", "(-d.x * d.y)", "(-d.y * d.y)", "(t * q.y)", "(t2 * q.y)", "(t3 * q.y)"];
  return { rx, ry };
};
export const lensNode = /* glsl */ `vec2 d = (p - LF.ctr) * LF.sig;
        vec2 q = (p - LA) * lir;
        float t = (dot(q, q) - LF.smid) * LF.ish;
        float t2 = t * t;
        float t3 = t2 * t;`;
export const lensAccumulate = (() => {
  const { rx, ry } = lensRows(), s = [];
  for (let a = 0; a < LENS_P; a++) for (let b = a; b < LENS_P; b++) {
    const t = [];
    if (rx[a] && rx[b]) t.push(`${rx[a]} * ${rx[b]}`);
    if (ry[a] && ry[b]) t.push(`${ry[a]} * ${ry[b]}`);
    if (t.length) s.push(`acc[${lensIdx(a, b)}] += w * (${t.join(" + ")});`);
  }
  for (let a = 0; a < LENS_P; a++) {
    const t = [];
    if (rx[a]) t.push(`${rx[a]} * r.x`);
    if (ry[a]) t.push(`${ry[a]} * r.y`);
    s.push(`acc[${66 + a}] += w * (${t.join(" + ")});`);
  }
  return s.join("\n        ");
})();
export const lensModel = (K) => {
  const { rx, ry } = lensRows(), x = [], y = [];
  for (let a = 0; a < 8 + K; a++) {
    if (rx[a]) x.push(`LT[${K}][${a}] * ${rx[a]}`);
    if (ry[a]) y.push(`LT[${K}][${a}] * ${ry[a]}`);
  }
  return `vec2(${x.join(" + ")}, ${y.join(" + ")})`;
};
const lensSolveTo = (kTop) => {
  const s = [];
  for (let j = 0; j < LENS_P; j++) {
    let sum = `Ns[${j}][${j}]`;
    for (let k = 0; k < j; k++) sum += ` - Ln[${j}][${k}] * Ln[${j}][${k}]`;
    s.push(`{ float v = ${sum}; uint om = min(okD, ${j}u); okD = !(v > 1e-7) ? om : okD; Ln[${j}][${j}] = sqrt(max(v, 1e-7)); }`);
    for (let i = j + 1; i < LENS_P; i++) {
      let e = `Ns[${i}][${j}]`;
      for (let k = 0; k < j; k++) e += ` - Ln[${i}][${k}] * Ln[${j}][${k}]`;
      s.push(`Ln[${i}][${j}] = (${e}) / Ln[${j}][${j}];`);
    }
  }
  for (let i = 0; i < LENS_P; i++) {
    let e = `bs[${i}]`;
    for (let k = 0; k < i; k++) e += ` - Ln[${i}][${k}] * z[${k}]`;
    s.push(`z[${i}] = (${e}) / Ln[${i}][${i}];`);
  }
  for (let K = 0; K <= kTop; K++) {
    const D = 8 + K;
    s.push(`if (${D}u <= okD && ${K}u <= LF.kmax) {`, `  float y[${D}] = ${zeros(D)};`);
    for (let i = D - 1; i >= 0; i--) {
      let e = `z[${i}]`;
      for (let k = i + 1; k < D; k++) e += ` - Ln[${k}][${i}] * y[${k}]`;
      s.push(`  y[${i}] = (${e}) / Ln[${i}][${i}];`);
    }
    for (let a = 0; a < LENS_P; a++) s.push(`  LT[${K}][${a}] = ${a < D ? `Dg[${a}] * y[${a}]` : "0.0"};`);
    s.push(`  LK[${K}] = 1u;`, "}");
  }
  return s.join("\n      ");
};
const lensNormalSolveTo = (kTop) => /* glsl */ `      float Ns[${LENS_P}][${LENS_P}] = ${zeros(LENS_P, LENS_P)};
      float bs[${LENS_P}] = ${zeros(LENS_P)};
      float Dg[${LENS_P}] = ${zeros(LENS_P)};
      float Ln[${LENS_P}][${LENS_P}] = ${zeros(LENS_P, LENS_P)};
      float z[${LENS_P}] = ${zeros(LENS_P)};
      float Ap[77] = ${zeros(77)};
      ${Array.from({ length: 77 }, (_, i) => `Ap[${i}] = LP[${i}];`).join(" ")}
      ${(() => { const s = []; for (let a = 0; a < LENS_P; a++) for (let b = a; b < LENS_P; b++) s.push(`Ns[${a}][${b}] = Ap[${lensIdx(a, b)}]; Ns[${b}][${a}] = Ap[${lensIdx(a, b)}];`); return s.join("\n      "); })()}
      ${Array.from({ length: LENS_P }, (_, a) => `Dg[${a}] = 1.0 / sqrt(max(Ns[${a}][${a}], 1e-30));`).join("\n      ")}
      ${Array.from({ length: LENS_P }, (_, a) => `${Array.from({ length: LENS_P }, (_, b) => `Ns[${a}][${b}] *= Dg[${a}] * Dg[${b}];`).join(" ")}\n      Ns[${a}][${a}] = Ns[${a}][${a}] * 1.0001 + 1e-9; bs[${a}] = Ap[${66 + a}] * Dg[${a}];`).join("\n      ")}
      uint okD = ${LENS_P}u;
      ${lensSolveTo(kTop)}
`;
export const lensNormalSolve = lensNormalSolveTo(3);
// ai: REFIT (the WGSL register.mjs (archived) REFIT): a workgroup of 64 a frame.
export const REFIT = VERSION + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Nodes { vec4 nodes[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 3, std430) readonly buffer Meas { vec4 meas[]; };
layout(set = 0, binding = 4, std430) buffer Maps { float maps[][16]; };
layout(set = 0, binding = 5, std430) buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 6, std140) uniform Fit { uvec4 count; uvec4 base; vec4 modules; } P;
` + LENS + /* glsl */ `
shared float JJ[64][36];
shared float Jr[64][8];
shared float Ag[44];
shared float th[8];
const int ITERS = ${ITERS};
vec2 fwd(float t[8], vec2 m, vec2 c, float S) {
  float w = t[6] * m.x + t[7] * m.y + 1.0;
  return c + S * vec2(t[0] * m.x + t[1] * m.y + t[2], t[3] * m.x + t[4] * m.y + t[5]) / w;
}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  uvec4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  Frame F = frames[f];
  uint count = P.count[pick.x];
  uint base = P.base[pick.x];
  float Mod = P.modules[pick.x];
  vec2 c = vec2(0.5 * float(F.w), 0.5 * float(F.h));
  float S = length(c);
  float inv = 1.0 / (S * S);
  float m0[16] = maps[f];
  if (li == 0u) {
    float h[9] = ${zeros(9)};
    ${unroll(3, (j) => `{ float s = ${j < 2 ? "Mod" : "1.0"}; h[${j}] = (m0[${j}] - c.x * m0[${6 + j}]) / S * s; h[${3 + j}] = (m0[${3 + j}] - c.y * m0[${6 + j}]) / S * s; h[${6 + j}] = m0[${6 + j}] * s; }`)}
    ${unroll(8, (j) => `th[${j}] = h[${j}] / h[8];`)}
  }
  barrier();
  for (int it = 0; it < ITERS; it++) {
    float t[8] = float[8](th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7]);
    float jj[36] = ${zeros(36)};
    float jr[8] = ${zeros(8)};
    for (uint n = li; n < count; n += 64u) {
      vec4 e = meas[f * ${NODES_MAX}u + n];
      if (e.w == 0.0 || e.z < ${WMIN}) { continue; }
      vec2 m = nodes[base + n].xy / Mod;
      vec2 p = fwd(t, m, c, S);
      vec2 r = e.xy - p;
      float du = distance(fwd(t, m + vec2(1.0 / Mod, 0.0), c, S), p);
      float rr = length(r);
      float hw = ${DELTA} * du / rr;
      float w = e.z * (rr > ${DELTA} * du ? hw : 1.0);
      ${jacobian}
      ${accumulate}
    }
    JJ[li] = jj;
    Jr[li] = jr;
    barrier();
    if (li < 44u) {
      float acc = 0.0;
      if (li < 36u) { for (uint s = 0u; s < 64u; s++) { acc += JJ[s][li]; } }
      else { for (uint s = 0u; s < 64u; s++) { acc += Jr[s][li - 36u]; } }
      Ag[li] = acc;
    }
    barrier();
    if (li == 0u) {
      float A[36] = ${zeros(36)};
      float g[8] = ${zeros(8)};
      ${unroll(36, (q) => `A[${q}] = Ag[${q}];`)}
      ${unroll(8, (a) => `g[${a}] = Ag[${36 + a}];`)}
      float N[8][9] = ${zeros(8, 9)};
      float D[8] = ${zeros(8)};
      ${normal}
      ${unroll(8, (a) => `D[${a}] = 1.0 / sqrt(max(N[${a}][${a}], 1e-30));`)}
      ${unroll(8, (a) => `${unroll(8, (b) => `N[${a}][${b}] *= D[${a}] * D[${b}];`)}\nN[${a}][${a}] = N[${a}][${a}] * 1.001 + 1e-9; N[${a}][8] = g[${a}] * D[${a}];`)}
      bool okSolve = true;
      ${gaussJordan("N", 8, 9, "1e-20", "okSolve")}
      if (okSolve) { ${unroll(8, (a) => `th[${a}] += D[${a}] * N[${a}][8] / N[${a}][${a}];`)} }
    }
    barrier();
  }
  if (li == 0u) {
    float mo[16] = ${zeros(16)};
    float t[9] = float[9](th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], 1.0);
    ${unroll(3, (j) => `{ float s = ${j < 2 ? "1.0 / Mod" : "1.0"}; mo[${j}] = (S * t[${j}] + c.x * t[${6 + j}]) * s; mo[${3 + j}] = (S * t[${3 + j}] + c.y * t[${6 + j}]) * s; mo[${6 + j}] = t[${6 + j}] * s; }`)}
    mo[9] = 0.0;
    mo[10] = c.x;
    mo[11] = c.y;
    mo[12] = inv;
    maps[f] = mo;
  }
}
`;

// ai: Each side's residual curve (the WGSL register.mjs (archived) SIDE_FITS): lane 0 of the sides' fits, every lane in each.
export const SIDE_FITS = /* glsl */ `
  {
    uint nn = (count + 4u) / 4u;
    for (uint sd = 0u; sd < 4u; sd++) {
      if (li < nn) {
        vec4 r = rw[(sd * (nn - 1u) + li) % count];
        bool done = (li == 0u && sd > 0u) || (li == nn - 1u && sd == 3u);
        float w = done ? 1.0 : r.z;
        float x = 2.0 * float(li) / float(nn - 1u) - 1.0;
        float xp = 1.0;
        ${unroll(SP_N, (p) => `sfP[li][${p}] = xp; xp *= x;`)}
        sfW[li] = vec4(w, w * r.x, w * r.y, w <= 0.0 ? 0.0 : 1.0);
      }
      barrier();
      if (li < ${SIDE_SUMS}u) {
        uint pa = li - ${SP_N + SIDE_MAX}u;
        uint pb = li - ${SP_N}u;
        uint pc = li < ${SP_N + SIDE_MAX}u ? pb : pa;
        uint p = li < ${SP_N}u ? li : pc;
        uint ca = li < ${SP_N + SIDE_MAX}u ? 1u : 2u;
        uint c = li < ${SP_N}u ? 0u : ca;
        float a = 0.0;
        for (uint k = 0u; k < nn; k++) { vec4 q = sfW[k]; if (q.w != 0.0) { a += q[c] * sfP[k][p]; } }
        sfS[li] = a;
      }
      if (li == ${SIDE_SUMS}u) {
        uint c = 0u;
        for (uint k = 0u; k < nn; k++) { c += sfW[k].w != 0.0 ? 1u : 0u; }
        sfN = c;
      }
      ${uniformLoad("uint nw", "sfN")}
      if (li < ${SIDE_MAX}u) {
        uint D = li + 1u;
        float Sp[${SP_N}] = ${zeros(SP_N)};
        ${unroll(SP_N, (p) => `Sp[${p}] = sfS[${p}];`)}
        float Ns[${SIDE_MAX}][${SIDE_MAX + 2}] = ${zeros(SIDE_MAX, SIDE_MAX + 2)};
        ${unroll(SIDE_MAX, (i) => `${unroll(SIDE_MAX, (j) => `Ns[${i}][${j}] = Sp[${i + j}];`)}\nNs[${i}][${i}] += 1e-6 * Sp[0]; Ns[${i}][${SIDE_MAX}] = sfS[${SP_N + i}]; Ns[${i}][${SIDE_MAX + 1}] = sfS[${SP_N + SIDE_MAX + i}];`)}
        bool ok = Sp[0] > 0.0;
        ${gaussJordanIn("Ns", "D", "1e-12", "ok")}
        bool fit = nw >= D + 2u;
        ${unroll(SIDE_MAX, (i) => [0, 1].map((h) => { const rhs = SIDE_MAX + h, sum = SP_N + h * SIDE_MAX + i; return `{ float q0 = sfS[${sum}] / Sp[0]; float q1 = (${i}u == 0u && Sp[0] > 0.0) ? q0 : 0.0; float q2 = Ns[${i}][${rhs}] / Ns[${i}][${i}]; float q3 = (ok && ${i}u < D) ? q2 : q1; sfC[li][${h * SIDE_MAX + i}] = fit ? q3 : 0.0; }`; }).join("\n"))}
      }
      barrier();
      if (li < nn) {
        vec4 r = rw[(sd * (nn - 1u) + li) % count];
        float x = 2.0 * float(li) / float(nn - 1u) - 1.0;
        ${unroll(SIDE_MAX, (d) => `{
          float fx = 0.0;
          float fy = 0.0;
          float xq = 1.0;
          ${unroll(d + 1, (i) => `fx += sfC[${d}][${i}] * xq; fy += sfC[${d}][${SIDE_MAX + i}] * xq; xq *= x;`)}
          sfR[li][${d}] = (r.x - fx) * (r.x - fx) + (r.y - fy) * (r.y - fy);
        }`)}
      }
      barrier();
      if (li < ${SIDE_MAX}u) {
        float rss = 0.0;
        for (uint k = 0u; k < nn; k++) { vec4 q = sfW[k]; if (q.w != 0.0) { rss += q.x * sfR[k][li]; } }
        sfA[li] = rss;
      }
      barrier();
      if (li == 0u) {
        float best = 3.0e38;
        uint kb = ${SIDE_MAX}u;
        float nobs = 2.0 * float(nw);
        ${unroll(SIDE_MAX, (d) => { const D = d + 1; return `if (nw >= ${D + 2}u) { float score = nobs * log(max(sfA[${d}], 1e-30) / nobs) + ${4 * D}.0 + ${4 * D * (2 * D + 1)}.0 / max(nobs - ${2 * D + 1}.0, 1e-9); if (score < best) { best = score; kb = ${d}u; } }`; })}
        float s0 = sfS[0];
        ${unroll(2 * SIDE_MAX, (i) => `sfC[${SIDE_MAX}][${i}] = 0.0;`)}
        if (kb < ${SIDE_MAX}u) { for (uint i = 0u; i < ${2 * SIDE_MAX}u; i++) { sfC[${SIDE_MAX}][i] = sfC[kb][i]; } }
        else if (s0 > 0.0) { sfC[${SIDE_MAX}][0] = sfS[${SP_N}] / s0; sfC[${SIDE_MAX}][${SIDE_MAX}] = sfS[${SP_N + SIDE_MAX}] / s0; }
      }
      barrier();
      if (li < nn) {
        uint n = (sd * (nn - 1u) + li) % count;
        float x = 2.0 * float(li) / float(nn - 1u) - 1.0;
        vec4 prev = rw[n];
        vec2 fitted = vec2(0.0);
        float xq = 1.0;
        ${unroll(SIDE_MAX, (i) => `fitted += vec2(sfC[${SIDE_MAX}][${i}], sfC[${SIDE_MAX}][${SIDE_MAX + i}]) * xq; xq *= x;`)}
        bool twoSided = li == 0u && sd > 0u;
        bool done = twoSided || (li == nn - 1u && sd == 3u);
        vec2 fm = 0.5 * (prev.xy + fitted);
        vec4 ov = vec4(twoSided ? fm : fitted, done ? prev.zw : prev.xy);
        resid[f * ${RESID}u + n] = ov;
        rw[n] = ov;
      }
      barrier();
    }
  }
`;

// ai: REFIT_POST (the WGSL register.mjs (archived) REFIT_POST, less its lens fit at K 0): a workgroup of 64 a frame,
// ai: REFIT's bindings. The residuals of REFIT's homography go straight to the sides' curves, the lens record none
// ai: (ANCH_REFIT's where the anchors are found). The fit at K 0, a first-order change of the homography fitted to the
// ai: same residuals without REFIT's weights, went 2026-11-06: 4090 blocks +0.008% over the eleven replays and the
// ai: lens set's places, the refit pass 1.0% less on the iGPU.
export const REFIT_POST = VERSION + DIMS + CONVERT + MAP + /* glsl */ `
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 1, std430) readonly buffer Nodes { vec4 nodes[]; };
layout(set = 0, binding = 2, std430) readonly buffer Sel { uvec4 sel[]; };
layout(set = 0, binding = 3, std430) readonly buffer Meas { vec4 meas[]; };
layout(set = 0, binding = 4, std430) buffer Maps { float maps[][16]; };
layout(set = 0, binding = 5, std430) buffer Resid { vec4 resid[]; };
layout(set = 0, binding = 6, std140) uniform Fit { uvec4 count; uvec4 base; vec4 modules; } P;
shared vec4 rw[${NODES_MAX}];
${SIDE_DECL}void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.x;
  uvec4 pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  uint count = P.count[pick.x];
  uint base = P.base[pick.x];
  float mf[16] = maps[f];
  for (uint n = li; n < count; n += 64u) {
    vec4 e = meas[f * ${NODES_MAX}u + n];
    bool ok = e.w != 0.0 && e.z >= ${WMIN};
    vec2 p = toImage(mf, nodes[base + n].xy);
    vec2 rd = e.xy - p;
    vec2 r = ok ? rd : vec2(0.0);
    vec4 v = vec4(r, ok ? e.z : 0.0, 0.0);
    resid[f * ${RESID}u + n] = v;
    rw[n] = v;
  }
  // ai: the lens record (LENS_SLOTS past the nodes) none: the map is REFIT's homography and the sides' curves
  if (li < ${LENS_SLOTS}u) { resid[f * ${RESID}u + ${NODES_MAX}u + li] = vec4(0.0); }
  ${STORAGE_BARRIER}
  barrier();
` + SIDE_FITS + /* glsl */ `}
`;

// ai: the WGSL register.mjs (archived) RING: where lattice node (i, j) of an nx x ny border ring sits in ring order.
export const RING = /* glsl */ `
uint ring(uint i, uint j, uint nx, uint ny) {
  if (j == 0u) { return i; }
  if (i == nx - 1u) { return nx - 1u + j; }
  if (j == ny - 1u) { return (nx - 1u) + (ny - 1u) + (nx - 1u - i); }
  return 2u * (nx - 1u) + (ny - 1u) + (ny - 1u - j);
}
`;
