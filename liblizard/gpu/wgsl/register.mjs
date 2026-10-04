// F7: registration. The finder's map is four points' homography; the picture needs a tenth of a module everywhere,
// under lens distortion and rolling-shutter wobble. The border is all a receiver has to measure that with (nothing
// is painted into the picture), so:
//   NODES   every border node of the lattice (src/layout.c: depth 3.5, every 16 modules, the corners included) is
//           measured at once, one workgroup each: its window of the border as painted (the kind map, 12 x 12
//           modules, word cells and picture left out) against the image under a grid of offsets, by normalised
//           cross-correlation, and a parabola through the best. Sampled at the level where a module is 1.5 to 3 px.
//   REFIT   one workgroup a frame fits a homography and radial distortion to every node by Gauss-Newton, each node
//           weighted by its correlation and a Huber weight, so a node under glare or off the frame is discounted,
//           not followed. What the fit leaves at each node is kept.
// Run twice: a wide coarse search from the finder's map, a refit, a narrow fine search from the refit's, a refit.
// F9 then adds the nodes' residuals over the picture as a Coons patch (wgsl/sample.mjs).
import { DIMS, MAP } from "./common.mjs";

export const WINDOW = 12;   // modules a side of a node's template
export const SUB = 2;       // template samples a module, each way
// ai: Room for 2 nx + 2 ny - 4 nodes; the rings take 20, 36, 52 and 68 (6, 10, 14 and 18 a side at 94, 158, 222 and
// ai: 286 modules), and the buffers keep the 136 of the 542-module border before the rings (2026-09-27).
export const NODES_MAX = 136;
const TMPL_MAX = WINDOW * WINDOW * SUB * SUB;   // every point of the window, the bound on a node's template
// The coarse round's sample-offset positions all lie on one lattice (samples a module apart, offsets half a module),
// (WINDOW - 1) * 2 + 2 * 6 + 1 points a side; a round whose lattice is larger measures each position directly.
const GRID_N = 35;
const WG = 256;      // NODES lanes: offsets x sample groups
const ITEMS = 512;   // the most (offset, group) pairs a round splits into

const LEVELS = /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> lv1: array<f32>;
@group(0) @binding(2) var<storage, read> lv2: array<f32>;
@group(0) @binding(3) var<storage, read> lv3: array<f32>;
@group(0) @binding(4) var<storage, read> lv4: array<f32>;
@group(0) @binding(5) var<storage, read> frames: array<Frame>;
@group(0) @binding(6) var<uniform> LD: LevelDims;
// One level of one frame, resolved once a workgroup: the taps then cost their address and nothing else.
struct Lvl { l: u32, w: i32, h: i32, base: u32, stride: u32, scale: f32 }
fn lvlOf(l: u32, f: u32) -> Lvl {
  let F = frames[f];
  var L = Lvl(l, i32((F.w + (1u << l) - 1u) >> l), i32((F.h + (1u << l) - 1u) >> l), 0u, 0u, 1.0 / f32(1u << l));
  if (l > 0u) { let s = LD.d[l]; L.base = f * s.y * s.x; L.stride = s.x; }
  return L;
}
fn tap4(L: Lvl, f: u32, x0: i32, x1: i32, y0: i32, y1: i32) -> vec4f {
  switch L.l {
    case 0u: {
      return vec4f(textureLoad(img, vec2i(x0, y0), i32(f), 0).r, textureLoad(img, vec2i(x1, y0), i32(f), 0).r,
        textureLoad(img, vec2i(x0, y1), i32(f), 0).r, textureLoad(img, vec2i(x1, y1), i32(f), 0).r);
    }
    case 1u: { let r0 = L.base + u32(y0) * L.stride; let r1 = L.base + u32(y1) * L.stride; return vec4f(lv1[r0 + u32(x0)], lv1[r0 + u32(x1)], lv1[r1 + u32(x0)], lv1[r1 + u32(x1)]); }
    case 2u: { let r0 = L.base + u32(y0) * L.stride; let r1 = L.base + u32(y1) * L.stride; return vec4f(lv2[r0 + u32(x0)], lv2[r0 + u32(x1)], lv2[r1 + u32(x0)], lv2[r1 + u32(x1)]); }
    case 3u: { let r0 = L.base + u32(y0) * L.stride; let r1 = L.base + u32(y1) * L.stride; return vec4f(lv3[r0 + u32(x0)], lv3[r0 + u32(x1)], lv3[r1 + u32(x0)], lv3[r1 + u32(x1)]); }
    default: { let r0 = L.base + u32(y0) * L.stride; let r1 = L.base + u32(y1) * L.stride; return vec4f(lv4[r0 + u32(x0)], lv4[r0 + u32(x1)], lv4[r1 + u32(x0)], lv4[r1 + u32(x1)]); }
  }
}
// Bilinear at level-0 pixel p, pixel centres at i + 0.5, clamped to the frame's own edge.
fn bilinear(L: Lvl, f: u32, p: vec2f) -> f32 {
  let q = p * L.scale - vec2f(0.5);
  let q0 = floor(q);
  let a = q - q0;
  let i = vec2i(q0);
  let x0 = clamp(i.x, 0, L.w - 1);
  let x1 = clamp(i.x + 1, 0, L.w - 1);
  let y0 = clamp(i.y, 0, L.h - 1);
  let y1 = clamp(i.y + 1, 0, L.h - 1);
  let v = tap4(L, f, x0, x1, y0, y1);
  return mix(mix(v.x, v.y, a.x), mix(v.z, v.w, a.x), a.y);
}
fn levelFor(u: f32) -> u32 { return u32(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
`;

// ai: Per ring: nodes[ring][n] = (x, y, first template point, points) in module coordinates, in ring order round the
// border (top left to right, right side down, bottom right to left, left side up; the corners once each); template
// points are (x, y, expected (0 dark, 1 light), 0), SUB^2 to a module, modules in order.
// meas[f][n] = (image x, image y, correlation, 1 where measured).
//
// A round is (offsets x samples) evaluations of the image; the 256 lanes split them as (offset, sample group)
// items, so every lane works whatever the offset count, and the groups' partial sums meet in workgroup memory.
// Where the round's samples and offsets share one lattice (the coarse round: samples a module apart, offsets half
// a module) the image is read once a lattice point and the correlations sum those reads; otherwise each sample's
// image position and the map's Jacobian there are found once, exactly, and an offset moves the sample by
// J * offset (the offsets are within 0.6 module, where the map's curvature is under a thousandth of a pixel).
export const NODES = DIMS + LEVELS + MAP + /* wgsl */ `
struct Round { steps: u32, stride: u32, p0: u32, p1: u32, step: f32, p2: f32, p3: f32, p4: f32, count: vec4u, base: vec4u }
@group(0) @binding(7) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(8) var<storage, read> tmpl: array<vec4f>;
@group(0) @binding(9) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(10) var<storage, read> sel: array<vec4u>;
@group(0) @binding(11) var<storage, read_write> meas: array<vec4f>;
@group(0) @binding(12) var<uniform> R: Round;
const CHUNK = ${TMPL_MAX / 2}u;
const GRID = ${GRID_N}u;
const ITEMS = ${ITEMS}u;
const WG = ${WG}u;
// Per sample of the current half of the template: (image x, image y, expected value, 0) and the map's Jacobian
// (px a module) there; on the lattice path sp.x holds the packed lattice column, row and expected value instead.
// Half the template at a time keeps the workgroup's memory near 21 KB, so more nodes are in flight on a core.
var<workgroup> sp: array<vec4f, CHUNK>;
var<workgroup> sj: array<vec4f, CHUNK>;
var<workgroup> gv: array<f32, ${GRID_N * GRID_N}>;
var<workgroup> psv: array<f32, ITEMS>;
var<workgroup> psv2: array<f32, ITEMS>;
var<workgroup> psev: array<f32, ITEMS>;
var<workgroup> ncc: array<f32, 225>;
var<workgroup> best: atomic<u32>;
@compute @workgroup_size(${WG}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.y;
  let n = wg.x;
  let pick = sel[f];
  let count = R.count[pick.x];
  if (frames[f].valid == 0u || pick.y == 0u || n >= count) { return; }
  let node = nodes[R.base[pick.x] + n];
  let M = maps[f];
  // The module size here, from the map, picks the level.
  let c = toImage(M, node.xy);
  let u = 0.5 * (distance(toImage(M, node.xy + vec2f(1.0, 0.0)), c) + distance(toImage(M, node.xy + vec2f(0.0, 1.0)), c));
  let L = lvlOf(levelFor(u), f);
  let first = u32(node.z);
  let pts = u32(node.w);
  let P = (pts + R.stride - 1u) / R.stride;
  let side = i32(R.steps);
  let gridW = u32(2 * side + 1);
  let O = gridW * gridW;
  let G = max(1u, ITEMS / O);
  let items = O * G;
  // The samples' spacing in modules: every SUB^2-th point is one a module, every point is 1 / SUB.
  let spacing = select(select(0.0, 1.0 / ${SUB}.0, R.stride == 1u), 1.0, R.stride == ${SUB * SUB}u);
  let ratio = spacing / R.step;
  let ratioN = u32(round(ratio));
  let gridN = (${WINDOW}u - 1u) * ratioN + u32(2 * side) + 1u;
  let lattice = spacing > 0.0 && abs(ratio - round(ratio)) < 1e-6 && gridN <= GRID;
  // The lattice's origin is the window's first sample; a lattice point (gx, gy) is origin + (g - steps) * step,
  // exact in single precision as the direct sum sample + offset is, so the two paths read the same positions.
  let origin = floor(node.xy - ${WINDOW / 2}.0) + vec2f(0.5 / ${SUB}.0);
  if (li == 0u) { atomicStore(&best, 0u); }
  // The template's own sums, reduced through the partials' arrays before those are used.
  var se = 0.0;
  var se2 = 0.0;
  for (var j = li; j < P; j += WG) {
    let t = tmpl[first + j * R.stride];
    se += t.z;
    se2 += t.z * t.z;
  }
  psv[li] = se;
  psv2[li] = se2;
  workgroupBarrier();
  if (li < 16u) {
    var s1 = 0.0;
    var s2 = 0.0;
    for (var k = 0u; k < 16u; k++) { s1 += psv[16u * li + k]; s2 += psv2[16u * li + k]; }
    psev[li] = s1;
    psev[16u + li] = s2;
  }
  workgroupBarrier();
  var tse = 0.0;
  var tse2 = 0.0;
  for (var k = 0u; k < 16u; k++) { tse += psev[k]; tse2 += psev[16u + k]; }
  workgroupBarrier();
  if (lattice) {
    for (var q = li; q < gridN * gridN; q += WG) {
      let g = vec2f(f32(q % gridN), f32(q / gridN));
      gv[q] = bilinear(L, f, toImage(M, origin + (g - f32(side)) * R.step));
    }
  }
  for (var item = li; item < items; item += WG) { psv[item] = 0.0; psv2[item] = 0.0; psev[item] = 0.0; }
  for (var c0 = 0u; c0 < P; c0 += CHUNK) {
    let c1 = min(P, c0 + CHUNK);
    for (var j = c0 + li; j < c1; j += WG) {
      let t = tmpl[first + j * R.stride];
      if (lattice) {
        // Column, row and expected value packed in one word, tagged with bit 30 so it is a normal float and no
        // denormal flush can touch it on the way through sp.x.
        let g = vec2u(round((t.xy - origin) / spacing)) * ratioN;
        sp[j - c0] = vec4f(bitcast<f32>(0x40000000u | g.x | (g.y << 8u) | (u32(t.z) << 16u)), 0.0, 0.0, 0.0);
      } else {
        let p = toImage(M, t.xy);
        // The map's Jacobian at t: the homography's, then through the lens inverse (u - c = d (1 + k1 |d|^2 / R^2)
        // with d = p - c, whose 2 x 2 derivative s I + g d d^T inverts by Sherman-Morrison).
        let w = M[6] * t.x + M[7] * t.y + M[8];
        let uu = vec2f(M[0] * t.x + M[1] * t.y + M[2], M[3] * t.x + M[4] * t.y + M[5]) / w;
        var jx = vec2f(M[0] - uu.x * M[6], M[3] - uu.y * M[6]) / w;
        var jy = vec2f(M[1] - uu.x * M[7], M[4] - uu.y * M[7]) / w;
        if (M[9] != 0.0) {
          let d = p - vec2f(M[10], M[11]);
          let r2 = dot(d, d);
          let s = 1.0 + M[9] * r2 * M[12];
          let g = 2.0 * M[9] * M[12];
          let q = g / (s + g * r2);
          jx = (jx - d * (q * dot(d, jx))) / s;
          jy = (jy - d * (q * dot(d, jy))) / s;
        }
        sp[j - c0] = vec4f(p, t.z, 0.0);
        sj[j - c0] = vec4f(jx.x, jy.x, jx.y, jy.y);
      }
    }
    workgroupBarrier();
    for (var item = li; item < items; item += WG) {
      let o = item % O;
      let g = item / O;
      // The group's first sample in this half: the first j = g (mod G) at or past c0.
      var j = select(g, g + G * ((c0 - g + G - 1u) / G), c0 > g);
      var sv = 0.0;
      var sv2 = 0.0;
      var sev = 0.0;
      if (lattice) {
        let d = (o % gridW) + (o / gridW) * gridN;
        for (; j < c1; j += G) {
          let s = bitcast<u32>(sp[j - c0].x);
          let v = gv[(s & 255u) + ((s >> 8u) & 255u) * gridN + d];
          sv += v; sv2 += v * v; sev += select(0.0, v, ((s >> 16u) & 1u) != 0u);
        }
      } else {
        let off = R.step * vec2f(f32(i32(o % gridW) - side), f32(i32(o / gridW) - side));
        // Two samples an iteration, so eight taps are in flight before their mixes: the taps' latency, not the
        // arithmetic, is what this loop waits on.
        for (; j + G < c1; j += 2u * G) {
          let s0 = sp[j - c0];
          let s1 = sp[j - c0 + G];
          let J0 = sj[j - c0];
          let J1 = sj[j - c0 + G];
          let v0 = bilinear(L, f, s0.xy + vec2f(J0.x * off.x + J0.y * off.y, J0.z * off.x + J0.w * off.y));
          let v1 = bilinear(L, f, s1.xy + vec2f(J1.x * off.x + J1.y * off.y, J1.z * off.x + J1.w * off.y));
          sv += v0 + v1; sv2 += v0 * v0 + v1 * v1; sev += s0.z * v0 + s1.z * v1;
        }
        if (j < c1) {
          let s = sp[j - c0];
          let J = sj[j - c0];
          let v = bilinear(L, f, s.xy + vec2f(J.x * off.x + J.y * off.y, J.z * off.x + J.w * off.y));
          sv += v; sv2 += v * v; sev += s.z * v;
        }
      }
      psv[item] += sv;
      psv2[item] += sv2;
      psev[item] += sev;
    }
    workgroupBarrier();
  }
  for (var o = li; o < O; o += WG) {
    var sv = 0.0;
    var sv2 = 0.0;
    var sev = 0.0;
    for (var g = 0u; g < G; g++) { let i = g * O + o; sv += psv[i]; sv2 += psv2[i]; sev += psev[i]; }
    let k = f32(P);
    let cov = sev - tse * sv / k;
    let den = sqrt(max((tse2 - tse * tse / k) * (sv2 - sv * sv / k), 1e-12));
    ncc[o] = cov / den;
    // The correlation in the high bits and the offset's index in the low 8, so one atomicMax picks the best.
    atomicMax(&best, (u32(clamp(ncc[o] + 1.0, 0.0, 1.999) * 4000000.0) << 8u) | o);
  }
  workgroupBarrier();
  if (li == 0u) {
    let b = atomicLoad(&best) & 255u;
    let bx = i32(b % gridW);
    let by = i32(b / gridW);
    let v = ncc[b];
    // Sub-step by a parabola through each axis's neighbours, where they exist.
    var dx = 0.0;
    var dy = 0.0;
    if (bx > 0 && bx < 2 * side) {
      let a = ncc[b - 1u];
      let z = ncc[b + 1u];
      let e = a + z - 2.0 * v;
      if (e < 0.0) { dx = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    if (by > 0 && by < 2 * side) {
      let a = ncc[b - gridW];
      let z = ncc[b + gridW];
      let e = a + z - 2.0 * v;
      if (e < 0.0) { dy = clamp(0.5 * (a - z) / e, -0.5, 0.5); }
    }
    let off = R.step * vec2f(f32(bx - side) + dx, f32(by - side) + dy);
    meas[f * ${NODES_MAX}u + n] = vec4f(toImage(M, node.xy + off), v, 1.0);
  }
}
`;

// REFIT: the map (homography and k1 about the image centre, the form the camera model has) fitted to the measured
// nodes by Gauss-Newton, in coordinates normalised so single precision holds: modules / M, pixels about the image
// centre / its half-diagonal. Five damped steps, each node weighted by its correlation (none under WMIN) and a
// Huber weight on its residual (DELTA modules). Writes the map, and each node's residual (measured minus fitted,
// image pixels) for the Coons patch; resid[f][n] = (dx, dy, weight, 0).
//
// The solves are written out with literal indices (generated below): an array indexed at run time lives in scratch
// memory on a GPU, and the pivoted 9 x 9 elimination that way took over 2 ms a call on the RDNA-2 iGPU.
export const WMIN = 0.3, DELTA = 0.4, ITERS = 5;
// ai: REFIT's debug words a frame (dbg, read by gpu/probe.mjs): the first refit's first iteration, th (0-8), the
// ai: normal matrix A (9-53) and the gradient g (54-62).
export const REFIT_DBG = 63;

const unroll = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join("\n");
// Gauss-Jordan elimination with partial pivoting on N (rows x cols, the right-hand sides after the square part),
// as straight-line WGSL: the pivot is chosen and swapped in by selects, so every index is a literal. `ok` is
// cleared on a vanishing pivot, and what follows is then garbage nobody reads.
function gaussJordan(N, rows, cols, eps, ok) {
  const s = [];
  for (let col = 0; col < rows; col++) {
    s.push(`{ var piv = ${col}; var pv = abs(${N}[${col}][${col}]);`);
    for (let r = col + 1; r < rows; r++) s.push(`{ let a = abs(${N}[${r}][${col}]); piv = select(piv, ${r}, a > pv); pv = max(pv, a); }`);
    for (let r = col + 1; r < rows; r++) {
      s.push(`{ let sw = piv == ${r};`);
      for (let j = col; j < cols; j++) s.push(`{ let a = ${N}[${col}][${j}]; let b = ${N}[${r}][${j}]; ${N}[${col}][${j}] = select(a, b, sw); ${N}[${r}][${j}] = select(b, a, sw); }`);
      s.push("}");
    }
    s.push(`${ok} = ${ok} && abs(${N}[${col}][${col}]) >= ${eps};`);
    for (let r = 0; r < rows; r++) {
      if (r === col) continue;
      s.push(`{ let fc = ${N}[${r}][${col}] / ${N}[${col}][${col}];`);
      for (let j = col; j < cols; j++) s.push(`${N}[${r}][${j}] -= fc * ${N}[${col}][${j}];`);
      s.push("}");
    }
    s.push("}");
  }
  return s.join("\n");
}
const jacobian = unroll(9, (k) => {
  const arr = (sign) => `array<f32, 9>(${Array.from({ length: 9 }, (_, j) => (j === k ? `t[${j}] ${sign} hs${k}` : `t[${j}]`)).join(", ")})`;
  return `let hs${k} = ${k === 8 ? "1e-4" : "1e-3"};\nlet J${k} = (fwd(${arr("+")}, m, c, S, inv) - fwd(${arr("-")}, m, c, S, inv)) / (2.0 * hs${k});`;
});
const accumulate = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 9; a++) {
    s.push(`jr[${a}] += w * dot(J${a}, r);`);
    for (let b = a; b < 9; b++) s.push(`jj[${q++}] += w * dot(J${a}, J${b});`);
  }
  return s.join("\n");
})();
const normal = (() => {
  const s = [];
  let q = 0;
  for (let a = 0; a < 9; a++) for (let b = a; b < 9; b++) { s.push(`N[${a}][${b}] = A[${q}]; N[${b}][${a}] = A[${q}];`); q++; }
  return s.join("\n");
})();
// ai: A side's residual curve takes 1 to SIDE_MAX coefficients (a constant to a quintic in the position along it),
// ai: the count chosen per side and frame by AICc on the side's own nodes (2026-09-29: each frame decided from its own
// ai: evidence, with no constant set on a phone). SIDE_MAX is the power basis's conditioning on
// ai: [-1, 1] at the rings' node counts, not a fitted value.
export const SIDE_MAX = 6;
// ai: One coefficient count D: the weighted fit's normal equations in Ns (D x D, the two right-hand sides in columns D
// ai: and D + 1), reduced; its coefficients into cx, cy; its weighted residual sum over the side's nodes (both axes) by
// ai: a second pass; AICc on N = 2 nodes and K = 2 D; the best kept in cxB, cyB. Literal indices only (Invariants).
const sideCount = (D) => `
      if (nw >= ${D + 2}u) {
        var Ns = array<array<f32, ${D + 2}>, ${D}>();
        ${unroll(D, (i) => `${unroll(D, (j) => `Ns[${i}][${j}] = Sp[${i + j}];`)}\nNs[${i}][${i}] += 1e-6 * Sp[0]; Ns[${i}][${D}] = bx[${i}]; Ns[${i}][${D + 1}] = by[${i}];`)}
        var ok = Sp[0] > 0.0;
        ${gaussJordan("Ns", D, D + 2, "1e-12", "ok")}
        var cx = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
        var cy = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
        if (ok) { ${unroll(D, (i) => `cx[${i}] = Ns[${i}][${D}] / Ns[${i}][${i}]; cy[${i}] = Ns[${i}][${D + 1}] / Ns[${i}][${i}];`)} }
        else if (Sp[0] > 0.0) { cx[0] = bx[0] / Sp[0]; cy[0] = by[0] / Sp[0]; }
        var rss = 0.0;
        for (var k = 0u; k < nn; k++) {
          let n = (sd * (nn - 1u) + k) % count;
          let r = resid[f * ${NODES_MAX}u + n];
          let done = (k == 0u && sd > 0u) || (k == nn - 1u && sd == 3u);
          let w = select(r.z, 1.0, done);
          if (w <= 0.0) { continue; }
          let x = 2.0 * f32(k) / f32(nn - 1u) - 1.0;
          var fx = 0.0;
          var fy = 0.0;
          var xq = 1.0;
          ${unroll(D, (i) => `fx += cx[${i}] * xq; fy += cy[${i}] * xq; xq *= x;`)}
          rss += w * ((r.x - fx) * (r.x - fx) + (r.y - fy) * (r.y - fy));
        }
        let nobs = 2.0 * f32(nw);
        let score = nobs * log(max(rss, 1e-30) / nobs) + ${4 * D}.0 + ${4 * D * (2 * D + 1)}.0 / max(nobs - ${2 * D + 1}.0, 1e-9);
        if (score < best) { best = score; ${unroll(SIDE_MAX, (i) => `cxB[${i}] = cx[${i}]; cyB[${i}] = cy[${i}];`)} }
      }`;

export const REFIT = DIMS + MAP + /* wgsl */ `
struct Fit { count: vec4u, base: vec4u, modules: vec4f }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(2) var<storage, read> sel: array<vec4u>;
@group(0) @binding(3) var<storage, read> meas: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> maps: array<array<f32, 16>>;
@group(0) @binding(5) var<storage, read_write> resid: array<vec4f>;
@group(0) @binding(6) var<uniform> P: Fit;
// ai: Debug, REFIT_DBG words a frame: the first iteration of the first refit's th (0-8), A (9-53) and g (54-62).
@group(0) @binding(7) var<storage, read_write> dbg: array<f32>;
var<workgroup> JJ: array<array<f32, 45>, 64>;
var<workgroup> Jr: array<array<f32, 9>, 64>;
var<workgroup> Ag: array<f32, 54>;
var<workgroup> th: array<f32, 9>;
var<workgroup> lastStep: f32;
const ITERS = ${ITERS};
// Normalised map: m in [0, 1], homography th[0..7] (th[8] = 1 implied) to u about the centre / S, then the lens.
fn fwd(t: array<f32, 9>, m: vec2f, c: vec2f, S: f32, inv: f32) -> vec2f {
  let w = t[6] * m.x + t[7] * m.y + 1.0;
  let u = c + S * vec2f(t[0] * m.x + t[1] * m.y + t[2], t[3] * m.x + t[4] * m.y + t[5]) / w;
  var p = u;
  for (var i = 0; i < 6; i++) { let d = p - c; p = c + (u - c) / (1.0 + t[8] * dot(d, d) * inv); }
  return p;
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  let F = frames[f];
  let count = P.count[pick.x];
  let base = P.base[pick.x];
  let Mod = P.modules[pick.x];
  let c = vec2f(0.5 * f32(F.w), 0.5 * f32(F.h));
  let S = length(c);
  let inv = 1.0 / (S * S);
  let m0 = maps[f];
  if (li == 0u) {
    // The current map in normalised form: H' = diag(1/S) (H - c e3^T) diag(M, M, 1), scaled so th[8] = 1.
    var h = array<f32, 9>();
    ${unroll(3, (j) => `{ let s = ${j < 2 ? "Mod" : "1.0"}; h[${j}] = (m0[${j}] - c.x * m0[${6 + j}]) / S * s; h[${3 + j}] = (m0[${3 + j}] - c.y * m0[${6 + j}]) / S * s; h[${6 + j}] = m0[${6 + j}] * s; }`)}
    ${unroll(8, (j) => `th[${j}] = h[${j}] / h[8];`)}
    th[8] = m0[9];
  }
  workgroupBarrier();
  for (var it = 0; it < ITERS; it++) {
    let t = array<f32, 9>(th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], th[8]);
    // Initialised every iteration: a loop-body var without an initializer kept its old value on Chrome/Vulkan.
    var jj = array<f32, 45>();
    var jr = array<f32, 9>();
    for (var n = li; n < count; n += 64u) {
      let e = meas[f * ${NODES_MAX}u + n];
      if (e.w == 0.0 || e.z < ${WMIN}) { continue; }
      let m = nodes[base + n].xy / Mod;
      let p = fwd(t, m, c, S, inv);
      let r = e.xy - p;
      // The module size here, in pixels, sets the Huber scale.
      let du = distance(fwd(t, m + vec2f(1.0 / Mod, 0.0), c, S, inv), p);
      let rr = length(r);
      let w = e.z * select(1.0, ${DELTA} * du / rr, rr > ${DELTA} * du);
      // The Jacobian by central differences in the normalised parameters.
      ${jacobian}
      ${accumulate}
    }
    JJ[li] = jj;
    Jr[li] = jr;
    workgroupBarrier();
    // The normal equations summed over the lanes, one lane an entry.
    if (li < 54u) {
      var acc = 0.0;
      if (li < 45u) { for (var s = 0u; s < 64u; s++) { acc += JJ[s][li]; } }
      else { for (var s = 0u; s < 64u; s++) { acc += Jr[s][li - 45u]; } }
      Ag[li] = acc;
    }
    workgroupBarrier();
    if (li == 0u) {
      var A = array<f32, 45>();
      var g = array<f32, 9>();
      ${unroll(45, (q) => `A[${q}] = Ag[${q}];`)}
      ${unroll(9, (a) => `g[${a}] = Ag[${45 + a}];`)}
      // Dense normal matrix with a little damping, scaled to a unit diagonal before elimination: the lens term is
      // nearly collinear with the perspective terms, and unscaled that is past what single precision can solve
      // (a float64 copy of this fit reached 0.3 samples where this reached 3.8 before the scaling).
      var N = array<array<f32, 10>, 9>();
      var D = array<f32, 9>();
      ${normal}
      ${unroll(9, (a) => `D[${a}] = 1.0 / sqrt(max(N[${a}][${a}], 1e-30));`)}
      ${unroll(9, (a) => `${unroll(9, (b) => `N[${a}][${b}] *= D[${a}] * D[${b}];`)}\nN[${a}][${a}] = N[${a}][${a}] * 1.001 + 1e-9; N[${a}][9] = g[${a}] * D[${a}];`)}
      var okSolve = true;
      ${gaussJordan("N", 9, 10, "1e-20", "okSolve")}
      let first = it == 0 && m0[13] == 0.0;
      if (first) {
        ${unroll(9, (a) => `dbg[f * ${REFIT_DBG}u + ${a}u] = th[${a}]; dbg[f * ${REFIT_DBG}u + ${54 + a}u] = g[${a}];`)}
        ${unroll(45, (q) => `dbg[f * ${REFIT_DBG}u + ${9 + q}u] = A[${q}];`)}
      }
      if (okSolve) {
        var st = 0.0;
        ${unroll(9, (a) => `{ let d = D[${a}] * N[${a}][9] / N[${a}][${a}]; th[${a}] += d; st += d * d; }`)}
        lastStep = sqrt(st);
      } else { lastStep = -1.0; }
    }
    workgroupBarrier();
  }
  // Back to pixels: H = [[S, 0, cx], [0, S, cy], [0, 0, 1]] H' diag(1/M, 1/M, 1).
  if (li == 0u) {
    var mo = array<f32, 16>();
    let t = array<f32, 9>(th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], 1.0);
    ${unroll(3, (j) => `{ let s = ${j < 2 ? "1.0 / Mod" : "1.0"}; mo[${j}] = (S * t[${j}] + c.x * t[${6 + j}]) * s; mo[${3 + j}] = (S * t[${3 + j}] + c.y * t[${6 + j}]) * s; mo[${6 + j}] = t[${6 + j}] * s; }`)}
    mo[9] = th[8];
    mo[10] = c.x;
    mo[11] = c.y;
    mo[12] = inv;
    // Diagnostics after the model's 13 floats: how many refits this frame has had, and the last step's size.
    mo[13] = m0[13] + 1.0;
    mo[14] = lastStep;
    maps[f] = mo;
  }
  workgroupBarrier();
  let mf = maps[f];
  for (var n = li; n < count; n += 64u) {
    let e = meas[f * ${NODES_MAX}u + n];
    let ok = e.w != 0.0 && e.z >= ${WMIN};
    let r = select(vec2f(0.0), e.xy - toImage(mf, nodes[base + n].xy), ok);
    resid[f * ${NODES_MAX}u + n] = vec4f(r, select(0.0, e.z, ok), 0.0);
  }
  workgroupBarrier();
  // What the fit leaves at the nodes is rolling-shutter wobble and lens the model missed, smooth along a side,
  // plus the measurement's own noise, which is not: on the simulator the map alone was 0.06 samples from the
  // truth and the raw residuals put it at 0.27. So each side's residual is replaced by a weighted polynomial fit
  // in the position along it, which keeps the wobble and drops the noise. Sides run in ring order: top 0..nn-1,
  // right nn-1..2nn-2, bottom 2nn-2..3nn-3, left 3nn-3..4nn-4 and 0.
  // ai: How much wobble to believe is the side's own evidence (2026-09-29): each count of coefficients from 1 to
  // ai: SIDE_MAX (as its nodes allow) is fitted and AICc keeps one. The fixed quadratic before it was picked on a
  // ai: recording (a cubic cost blocks on the recorded LIZARD-512 run), and on the v0.3 runs it fitted noise: the
  // ai: rule takes a constant on nearly every side there (STATUS "A side's degree by AICc").
  // The sides run in order on one lane, as the fit was measured: a side's first node is the corner the side
  // before has just replaced by its own fit (weight 1), and the mean of the two fits is what the corner keeps.
  if (li == 0u) {
    let nn = (count + 4u) / 4u;
    for (var sd = 0u; sd < 4u; sd++) {
      // ai: The power sums Sp[p] = sum w x^p to x^(2 SIDE_MAX - 2) and the right sides to x^(SIDE_MAX - 1), once.
      var Sp = array<f32, ${2 * SIDE_MAX - 1}>(${Array(2 * SIDE_MAX - 1).fill("0.0").join(", ")});
      var bx = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
      var by = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
      var nw = 0u;
      for (var k = 0u; k < nn; k++) {
        let n = (sd * (nn - 1u) + k) % count;
        let r = resid[f * ${NODES_MAX}u + n];
        // ai: a corner the side before has already replaced counts with weight 1 (its zw hold the measured residual)
        let done = (k == 0u && sd > 0u) || (k == nn - 1u && sd == 3u);
        let w = select(r.z, 1.0, done);
        if (w <= 0.0) { continue; }
        nw += 1u;
        let x = 2.0 * f32(k) / f32(nn - 1u) - 1.0;
        var xp = 1.0;
        ${unroll(2 * SIDE_MAX - 1, (p) => `Sp[${p}] += w * xp; xp *= x;`)}
        var xq = 1.0;
        ${unroll(SIDE_MAX, (i) => `bx[${i}] += w * r.x * xq; by[${i}] += w * r.y * xq; xq *= x;`)}
      }
      var best = 3.0e38;
      var cxB = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
      var cyB = array<f32, ${SIDE_MAX}>(${Array(SIDE_MAX).fill("0.0").join(", ")});
      ${unroll(SIDE_MAX, (d) => sideCount(d + 1))}
      // ai: under 3 nodes with weight no count leaves a degree of freedom: their weighted mean
      if (best == 3.0e38 && Sp[0] > 0.0) { cxB[0] = bx[0] / Sp[0]; cyB[0] = by[0] / Sp[0]; }
      for (var k = 0u; k < nn; k++) {
        let n = (sd * (nn - 1u) + k) % count;
        let x = 2.0 * f32(k) / f32(nn - 1u) - 1.0;
        // A corner belongs to two sides; it takes the mean of the two fits.
        let prev = resid[f * ${NODES_MAX}u + n];
        var fitted = vec2f(0.0);
        var xq = 1.0;
        ${unroll(SIDE_MAX, (i) => `fitted += vec2f(cxB[${i}], cyB[${i}]) * xq; xq *= x;`)}
        let twoSided = k == 0u && sd > 0u;
        let done = twoSided || (k == nn - 1u && sd == 3u);
        // ai: zw keeps the node's measured residual (already moved there by the side before, at a corner): nothing
        // ai: downstream reads it (sample.mjs takes xy), the probe does.
        resid[f * ${NODES_MAX}u + n] = vec4f(select(fitted, 0.5 * (prev.xy + fitted), twoSided), select(prev.xy, prev.zw, done));
      }
    }
    // The last corner (left side's end) is the first node: it ends up with the left side's fit alone.
  }
}
`;
// Where lattice node (i, j) of an nx x ny border ring sits in ring order.
export const RING = /* wgsl */ `
fn ring(i: u32, j: u32, nx: u32, ny: u32) -> u32 {
  if (j == 0u) { return i; }
  if (i == nx - 1u) { return nx - 1u + j; }
  if (j == ny - 1u) { return (nx - 1u) + (ny - 1u) + (nx - 1u - i); }
  return 2u * (nx - 1u) + (ny - 1u) + (ny - 1u - j);
}
`;
// Host side of the same order: the ring's lattice coordinates.
export function ringOrder(nx, ny) {
  const out = [];
  for (let i = 0; i < nx; i++) out.push([i, 0]);
  for (let j = 1; j < ny; j++) out.push([nx - 1, j]);
  for (let i = nx - 2; i >= 0; i--) out.push([i, ny - 1]);
  for (let j = ny - 2; j >= 1; j--) out.push([0, j]);
  return out;
}
