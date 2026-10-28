// ai: The anchors on the GPU (2026-10-26; part of every symbol since 2026-10-27, SPEC 6.13), the GPU's own algorithm
// ai: for what the C reads (src/acquire.c anchor_find, anch_herm4, border_fill_anchors; src/focus.c focus_finish_ext's
// ai: erase): a 7 x 7 grid of squares of a = span / 64 modules painted inside the picture (src/focus.c focus_encode),
// ai: whose middles say what the border's map misses inside. Its stages after F7's last REFIT_POST, in its refit's
// ai: profile pass, before F8, on the frames the finder found (sel = (ring, found, 0, 0) then); a told batch (no F7)
// ai: runs ANCH_FIND and ANCH_FIT alone, through the map it was told:
// ai:   ANCH_FIND, a workgroup a (anchor, frame): a window of the square and one side of it each way, sampled on a
// ai:     grid of a / ANCH_TS modules through the frame's final map (F9's composition: the homography, the lens about the
// ai:     axis, the Coons patch of the border's residuals), the pattern's normalised correlation at every whole step from
// ai:     box sums of the window's integral images (the pattern is two levels in concentric squares), the best (a tie to
// ai:     the lower offset), a parabola each way. anch[ANCH_K f + k] = (the residual the map needs at the middle in
// ai:     image px, the peak, 1 where it reached ANCH_PEAK off the window's edge).
// ai:   ANCH_PREDICT, a workgroup a frame: the border nodes' places from the anchors alone, into meas (below).
// ai:   The border searched again about them: F7's node search in its centred form (wgsl/register.mjs nodesSource), a
// ai:     workgroup a node, on F7's layout, its own round (+-0.4 module at a fifth, every template sample: the host's
// ai:     anchRound).
// ai:   ANCH_REFIT, a workgroup a frame: the lens about the axis fitted again over the border and the anchors (below).
// ai:   ANCH_FIT, a workgroup a frame: the anchors found (result[f][31]); the mesh's knots (with ANCH_MIN or more: the
// ai:     border's 0, an anchor's reading whole, a missing one the mean of its found 4-neighbours, in rounds; else all 0)
// ai:     and the picture's mean, into the frame's record (wgsl/back_transform.mjs ANCH_REC), which pass 1 reads: every
// ai:     sample through the mesh, the squares erased to the mean before the detrend.
// ai: F9 (wgsl/sample.mjs, the wasm back half's grids) reads no anchors.
import { DIMS, LUM0, MAP } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { LATTICE_MAX, ANCH_SIDE, ANCH_K, ANCH_REC } from "./back_transform.mjs";
import { RING_COUNT } from "./finder.mjs";
import { NODES_MAX, RESID, WMIN, LENS_P, lensNode, lensAccumulate, lensModel, lensNormalSolve, SIDE_FITS } from "./register.mjs";

export { ANCH_SIDE, ANCH_K, ANCH_REC };
// ai: The C's constants (src/acquire.c): template points across a square (OB_ANCH_TS), the peak an anchor is found at
// ai: (ANCH_PEAK), the anchors the geometry takes (src/internal.h OB_ANCH_MIN, the lens model's terms: 8 until
// ai: 2026-10-27), the fill's rounds.
export const ANCH_TS = 16, ANCH_PEAK = 0.6, ANCH_MIN = 12, ANCH_ROUNDS = 6;
// ai: The search's reach, template points each way about where the final map puts a square: one side of the square, 2,
// ai: 4 and 8 modules in the 64, 128 and 256 rings (the strongest lenses put squares more than 2 modules off the border's
// ai: map in the larger rings, and a peak on the window's edge is refused); the window's side, its integral images' (one
// ai: row and column of zeros more) and the offsets scored a side.
const RR = ANCH_TS, WN = ANCH_TS + 2 * RR, W1N = WN + 1, ON = 2 * RR + 1;
// ai: The pattern's two squares on the template's points: point t's Chebyshev distance over the half-side is
// ai: |(t + 0.5) / (TS / 2) - 1| in every ring, light under 0.25 or from 0.6, dark between; [I25, J25] and [I6, J6]
// ai: the points under 0.25 and 0.6 along an axis. TM the template's mean, TN the norm of T - TM (the C's tm, tn).
const dist = (t) => Math.abs((t + 0.5) / (ANCH_TS / 2) - 1);
const span1 = (lim) => { const s = [...Array(ANCH_TS).keys()].filter((t) => dist(t) < lim); return [s[0], s.at(-1)]; };
const [I25, J25] = span1(0.25), [I6, J6] = span1(0.6);
const DARK = (J6 - I6 + 1) ** 2 - (J25 - I25 + 1) ** 2, LIGHT = ANCH_TS * ANCH_TS - DARK;
const TM = (LIGHT - DARK) / (ANCH_TS * ANCH_TS), TN = Math.fround(Math.sqrt(LIGHT * (1 - TM) ** 2 + DARK * (1 + TM) ** 2));
const lit = (x) => (Number.isInteger(x) ? `${x}.0` : String(x));

// ai: A, the anchors' uniform (ANCH_FIND's and ANCH_FIT's), from gpu/tables.mjs formatTables() rings: three vec4f a
// ai: ring, (k0, k1, k2, k3), (k4, k5, k6, k7), (k8, a, span, margin): the 9 knots of an axis in modules, the lattice's
// ai: ends (nodeX[0] and nodeX[nn - 1], where F7's Coons patch holds the map) and the 7 anchors' centres between, the
// ai: C's m0(i) + a / 2 (src/acquire.c ob_anchor_m0: floor(margin + span (i + 0.5) / 7 - a / 2 + 0.5)); a = span / 64.
export function anchUniform(rings) {
  if (rings.length !== RING_COUNT) throw new Error(`${rings.length} rings: the anchors' uniform holds ${RING_COUNT}`);
  const u = new Float32Array(12 * RING_COUNT);
  rings.forEach(({ cells, span, margin, nodeX }, r) => {
    const a = Math.floor(span / 64);
    const c = Array.from({ length: ANCH_SIDE }, (_, i) => Math.floor(margin + (span * (i + 0.5)) / ANCH_SIDE - a / 2 + 0.5) + a / 2);
    const k = [nodeX[0], ...c, nodeX[nodeX.length - 1]];
    if (k.some((x, i) => i && !(x > k[i - 1]))) throw new Error(`ring ${cells}: knots ${k.join(", ")} not rising`);
    u.set([...k, a, span, margin], 12 * r);
  });
  return u;
}

// ai: What both kernels bind first (0 to 6, the same in each): the frame texture, the frame records, the maps, sel,
// ai: resid (the border's residuals and the lens, register.mjs), the lattice (F8's LT, gpu/wgsl/word.mjs wordLattice:
// ai: at[ring] = (first coordinate, nodes a side)) and A; the final map at a module point (mapAt).
const HEAD = DIMS + MAP + /* wgsl */ `
struct Lattice { at: array<vec4u, ${RING_COUNT}>, lat: array<vec4f, ${(RING_COUNT * LATTICE_MAX) / 4}> }
struct Anchors { k: array<vec4f, ${3 * RING_COUNT}> }
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(3) var<storage, read> sel: array<vec4u>;
@group(0) @binding(4) var<storage, read> resid: array<vec4f>;
@group(0) @binding(5) var<uniform> LT: Lattice;
@group(0) @binding(6) var<uniform> A: Anchors;
fn latAt(i: u32) -> f32 { return LT.lat[i >> 2u][i & 3u]; }
// ai: knot i (0 to 8) of ring r's axis
fn knot(r: u32, i: u32) -> f32 { return A.k[3u * r + (i >> 2u)][i & 3u]; }
` + COONS + LUM0 + /* wgsl */ `
fn mapAt(f: u32, M: array<f32, 16>, lz: Lens, first: u32, nn: u32, m: vec2f) -> vec2f {
  let p = toImage(M, m);
  return p + lensAt(lz, p) + coons(f, first, nn, m);
}
`;

// ai: ANCH_FIND: dispatch (ANCH_K, frames), 256 lanes; wg.x the anchor k = 7 j + i, wg.y the frame. The window's
// ai: columns and rows first (their module coordinate and their pieces of the Coons patch, which depend on one
// ai: coordinate each, as the fused pass 1 takes them), then its W x W samples into the integral images I1 (values) and
// ai: I2 (squares), row 0 and column 0 zero, summed along the rows and then down the columns. At an offset (ox, oy)
// ai: the template's top left sits at window point (rr + ox, rr + oy): S, S2 its box sums, S25 and S6 the squares',
// ai: var = S2 - S^2 / TS^2, the sum against T - TM is 2 (S25 - S6) + S (1 - TM), its correlation that over
// ai: sqrt(var) TN (0 at a var under VAR_MIN). Every offset's in C; the best a lane (strictly greater, so its lowest),
// ai: then the workgroup's by a tree (a tie to the lower offset). A peak on the window's edge, or under ANCH_PEAK, is
// ai: not found; its record is written all the same.
export const ANCH_FIND = HEAD + /* wgsl */ `
@group(0) @binding(7) var<storage, read_write> anch: array<vec4f>;
const TS: u32 = ${ANCH_TS}u;
const RR: i32 = ${RR};
const TM: f32 = ${lit(TM)};
const TN: f32 = ${lit(TN)};
const VAR_MIN: f32 = 1e-6;
var<workgroup> I1: array<f32, ${W1N * W1N}>;
var<workgroup> I2: array<f32, ${W1N * W1N}>;
var<workgroup> C: array<f32, ${ON * ON}>;
var<workgroup> wcx: array<f32, ${WN}>;
var<workgroup> wcu: array<f32, ${WN}>;
var<workgroup> wct: array<vec2f, ${WN}>;
var<workgroup> wcb: array<vec2f, ${WN}>;
var<workgroup> wry: array<f32, ${WN}>;
var<workgroup> wrv: array<f32, ${WN}>;
var<workgroup> wrl: array<vec2f, ${WN}>;
var<workgroup> wrr: array<vec2f, ${WN}>;
var<workgroup> rs: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
// ai: the box of side s at window point (x, y) of I1 or I2, rows w1 apart (src/acquire.c ABOX)
fn box1(x: u32, y: u32, s: u32, w1: u32) -> f32 { return I1[(y + s) * w1 + x + s] - I1[y * w1 + x + s] - I1[(y + s) * w1 + x] + I1[y * w1 + x]; }
fn box2(x: u32, y: u32, s: u32, w1: u32) -> f32 { return I2[(y + s) * w1 + x + s] - I2[y * w1 + x + s] - I2[(y + s) * w1 + x] + I2[y * w1 + x]; }
fn scoreAt(px: u32, py: u32, w1: u32) -> f32 {
  let s1 = box1(px, py, TS, w1);
  let s2 = box2(px, py, TS, w1);
  let vr = s2 - s1 * s1 / f32(TS * TS);
  let s25 = box1(px + ${I25}u, py + ${I25}u, ${J25 - I25 + 1}u, w1);
  let s6 = box1(px + ${I6}u, py + ${I6}u, ${J6 - I6 + 1}u, w1);
  let st = 2.0 * (s25 - s6) + s1 * (1.0 - TM);
  return select(0.0, st / (sqrt(vr) * TN), vr > VAR_MIN);
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.y;
  let k = wg.x;
  if (k >= ${ANCH_K}u) { return; }
  let F = frames[f];
  let pick = sel[f];
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { anch[${ANCH_K}u * f + k] = vec4f(0.0); }
    return;
  }
  let rgi = pick.x;
  let a = A.k[3u * rgi + 2u].y;
  let h = a / f32(TS);
  let rr = RR;
  let rf = f32(rr);
  let W = TS + 2u * u32(rr);
  let W1 = W + 1u;
  let O = 2u * u32(rr) + 1u;
  let x0 = knot(rgi, 1u + k % ${ANCH_SIDE}u) - 0.5 * a;
  let y0 = knot(rgi, 1u + k / ${ANCH_SIDE}u) - 0.5 * a;
  let M = maps[f];
  let lz = lensOf(f);
  let at = LT.at[rgi];
  let first = at.x;
  let nn = at.y;
  let cn = corners(f, nn);
  let fw = i32(F.w);
  let fh = i32(F.h);
  for (var t = li; t < 2u * W; t += 256u) {
    if (t < W) {
      let mx = x0 + (f32(t) - rf + 0.5) * h;
      wcx[t] = mx;
      wcu[t] = coonsT(mx, first, nn);
      wct[t] = curve(f, 0u, mx, first, nn);
      wcb[t] = curve(f, 2u, mx, first, nn);
    } else {
      let s = t - W;
      let my = y0 + (f32(s) - rf + 0.5) * h;
      wry[s] = my;
      wrv[s] = coonsT(my, first, nn);
      wrl[s] = curve(f, 3u, my, first, nn);
      wrr[s] = curve(f, 1u, my, first, nn);
    }
  }
  for (var t = li; t < W1; t += 256u) { I1[t] = 0.0; I2[t] = 0.0; I1[t * W1] = 0.0; I2[t * W1] = 0.0; }
  workgroupBarrier();
  for (var t = li; t < W * W; t += 256u) {
    let wx = t % W;
    let wy = t / W;
    let p0 = toImage(M, vec2f(wcx[wx], wry[wy]));
    let v = bilin0(f, p0 + lensAt(lz, p0) + coonsMix(wcu[wx], wrv[wy], wct[wx], wcb[wx], wrl[wy], wrr[wy], cn), fw, fh);
    let o = (wy + 1u) * W1 + wx + 1u;
    I1[o] = v;
    I2[o] = v * v;
  }
  workgroupBarrier();
  if (li < 2u * W) {
    let r = li % W + 1u;
    if (li < W) { for (var x = 2u; x <= W; x++) { I1[r * W1 + x] += I1[r * W1 + x - 1u]; } }
    else { for (var x = 2u; x <= W; x++) { I2[r * W1 + x] += I2[r * W1 + x - 1u]; } }
  }
  workgroupBarrier();
  if (li < 2u * W) {
    let c = li % W + 1u;
    if (li < W) { for (var y = 2u; y <= W; y++) { I1[y * W1 + c] += I1[(y - 1u) * W1 + c]; } }
    else { for (var y = 2u; y <= W; y++) { I2[y * W1 + c] += I2[(y - 1u) * W1 + c]; } }
  }
  workgroupBarrier();
  var best = -3.0;
  var bo = 0xffffffffu;
  for (var o = li; o < O * O; o += 256u) {
    let s = scoreAt(o % O, o / O, W1);
    C[o] = s;
    if (s > best) { best = s; bo = o; }
  }
  rs[li] = best;
  ri[li] = bo;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) {
      let s2 = rs[li + st];
      let o2 = ri[li + st];
      if (s2 > rs[li] || (s2 == rs[li] && o2 < ri[li])) { rs[li] = s2; ri[li] = o2; }
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let pk = rs[0];
    let po = ri[0];
    let bx = i32(po % O) - rr;
    let by = i32(po / O) - rr;
    var sx = 0.0;
    var sy = 0.0;
    if (bx > -rr && bx < rr) { let cl = C[po - 1u]; let cr = C[po + 1u]; let d = cl - 2.0 * pk + cr; if (d < 0.0) { sx = 0.5 * (cl - cr) / d; } }
    if (by > -rr && by < rr) { let cl = C[po - O]; let cr = C[po + O]; let d = cl - 2.0 * pk + cr; if (d < 0.0) { sy = 0.5 * (cl - cr) / d; } }
    let ok = pk >= ${lit(ANCH_PEAK)} && bx > -rr && bx < rr && by > -rr && by < rr;
    let X = vec2f(x0 + 0.5 * a, y0 + 0.5 * a);
    let D = vec2f((f32(bx) + sx) * h, (f32(by) + sy) * h);
    let rk = mapAt(f, M, lz, first, nn, X + D) - mapAt(f, M, lz, first, nn, X);
    anch[${ANCH_K}u * f + k] = vec4f(rk, pk, select(0.0, 1.0, ok));
  }
}
`;

// ai: ANCH_FIT: dispatch (frames), 64 lanes. result[f][31] = the anchors found (0 for a frame not found; PICK zeroed
// ai: the record and nothing else writes word 31). Every frame the finder found or the host told gets its record on,
// ai: so pass 1 erases its anchors whatever was found (SPEC 6.13); under ANCH_MIN found its knots are all 0 (no mesh).
// ai: Otherwise every found anchor's reading is its knot whole (the C shrinks its readings by their share of signal over
// ai: their own noise, after its lens refit, where little but noise is left; here they carry the whole of what F7's
// ai: border fit misses inside, which a shrink cuts). Then the missing ones, round by round (ANCH_ROUNDS at most), the
// ai: mean of the 4-neighbours known before the round (none: 0). The picture's mean for the erase: 16 x 16 points evenly
// ai: over the picture through the final map. The record (wgsl/back_transform.mjs ANCH_REC): (1, mean, 0, found), A's
// ai: knots and a, the 49 inner knots.
export const ANCH_FIT = HEAD + /* wgsl */ `
@group(0) @binding(7) var<storage, read> anch: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> arec: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> result: array<array<f32, 32>>;
const NK: u32 = ${ANCH_K}u;
const SIDE: u32 = ${ANCH_SIDE}u;
var<workgroup> kv: array<vec2f, ${ANCH_K}>;
var<workgroup> kn: array<u32, ${ANCH_K}>;
var<workgroup> kn2: array<u32, ${ANCH_K}>;
var<workgroup> red: array<f32, 64>;
var<workgroup> wfound: u32;
// ai: the sum of knot q's 4-neighbours known (kn) and their count, the order the C takes them: left, right, up, down
fn around(q: u32) -> vec3f {
  let i = q % SIDE;
  let j = q / SIDE;
  var s = vec2f(0.0);
  var c = 0.0;
  if (i > 0u && kn[q - 1u] == 1u) { s += kv[q - 1u]; c += 1.0; }
  if (i + 1u < SIDE && kn[q + 1u] == 1u) { s += kv[q + 1u]; c += 1.0; }
  if (j > 0u && kn[q - SIDE] == 1u) { s += kv[q - SIDE]; c += 1.0; }
  if (j + 1u < SIDE && kn[q + SIDE] == 1u) { s += kv[q + SIDE]; c += 1.0; }
  return vec3f(s, c);
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let F = frames[f];
  let pick = sel[f];
  let ab = f * ${ANCH_REC}u;
  if (F.valid == 0u || pick.y == 0u) {
    if (li == 0u) { arec[ab] = vec4f(0.0); result[f][31] = 0.0; }
    return;
  }
  let rgi = pick.x;
  if (li < NK) {
    let e = anch[NK * f + li];
    let ok = e.w > 0.5;
    kv[li] = select(vec2f(0.0), e.xy, ok);
    kn[li] = select(0u, 1u, ok);
  }
  workgroupBarrier();
  if (li == 0u) {
    var c = 0u;
    for (var q = 0u; q < NK; q++) { c += kn[q]; }
    wfound = c;
  }
  let found = workgroupUniformLoad(&wfound);
  if (li == 0u) { result[f][31] = f32(found); }
  if (found < ${ANCH_MIN}u && li < NK) { kv[li] = vec2f(0.0); kn[li] = 1u; }
  for (var rd = 0u; rd < ${ANCH_ROUNDS}u; rd++) {
    workgroupBarrier();
    if (li < NK) {
      var now = kn[li];
      if (now == 0u) {
        let nb = around(li);
        if (nb.z > 0.0) { kv[li] = nb.xy / nb.z; now = 1u; }
      }
      kn2[li] = now;
    }
    workgroupBarrier();
    if (li < NK) { kn[li] = kn2[li]; }
  }
  let geo = A.k[3u * rgi + 2u];
  let span = geo.z;
  let margin = geo.w;
  let M = maps[f];
  let lz = lensOf(f);
  let at = LT.at[rgi];
  var sum = 0.0;
  for (var p = li; p < 256u; p += 64u) {
    let m = vec2f(margin + span * (f32(p % 16u) + 0.5) / 16.0, margin + span * (f32(p / 16u) + 0.5) / 16.0);
    sum += bilin0(f, mapAt(f, M, lz, at.x, at.y, m), i32(F.w), i32(F.h));
  }
  red[li] = sum;
  workgroupBarrier();
  if (li == 0u) {
    var s = 0.0;
    for (var q = 0u; q < 64u; q++) { s += red[q]; }
    arec[ab] = vec4f(1.0, s / 256.0, 0.0, f32(found));
    arec[ab + 1u] = A.k[3u * rgi];
    arec[ab + 2u] = A.k[3u * rgi + 1u];
    arec[ab + 3u] = geo;
  }
  if (li < ${ANCH_REC - 4}u) {
    let q = 2u * li;
    arec[ab + 4u + li] = vec4f(kv[q], select(vec2f(0.0), kv[min(q + 1u, NK - 1u)], q + 1u < NK));
  }
}
`;

// ai: ANCH_REFIT: dispatch (frames), 64 lanes, after the border's second search (ANCH_PREDICT's, below) and before ANCH_FIT,
// ai: on a frame with ANCH_MIN anchors found or more (the others are left as F7 wrote them). The lens about the camera's
// ai: axis fitted again over the border's nodes as measured and the anchors found, the C's border_fill_anchors in the
// ai: GPU's form (src/acquire.c lensm_fit),
// ai: so the map the interior is read through follows what the anchors see and the mesh carries only what the model
// ai: cannot (F7's lens, fitted to the border alone, misses the interior by up to 2 or 3 modules on the strongest
// ai: lenses in a 2:1 half, more than the mesh's cubic from the border's 0 to the outer anchors follows):
// ai: - The points, about H, F7's homography (the map's k1 slot is 0: REFIT fits the homography alone, F7's lens being
// ai:   the one about the axis, which an axis inside a 2:1 half's edge leaves a field a k1 about the image's centre
// ai:   could not take back from: about H the anchors alone put the border of the strongest lenses within 0.82 module
// ai:   of the C's map, about F7's whole map with that k1 3.6 to 4.2 modules off): each border node's displacement from
// ai:   H as measured, weighted by its correlation; each found anchor's, its record (the residual the final map needed
// ai:   at its centre) plus what that map adds over H there (the lens and the Coons patch REFIT_POST wrote), weighted 1
// ai:   (the C's weight; by its peak, 38 fewer blocks over the published lenses). Every later sampler reads H, the
// ai:   lens and the patch.
// ai: - The model (wgsl/register.mjs LENS: a homography's first-order change and K = 0 to 3 radial terms about the axis,
// ai:   K by AICc), with no cap inside the code's quad: the anchors sample the interior's radii, which a 1:1 crop's
// ai:   border does not.
// ai: - Robust over the border: a node further from the model than the larger of half a module (px, the corners
// ai:   through H) and three robust spreads (1.4826 times the median over the border's nodes) is dropped and the model
// ai:   fitted again, four fits at most; the anchors are never dropped. A dropped node reads the model there (its
// ai:   residual 0), so the patch leaves nothing of it.
// ai: Then the lens record, each side's curve of what the model leaves on the border (SIDE_FITS, REFIT_POST's), and
// ai: each found anchor's record against the new final map, which ANCH_FIT reads.
// ai: ANCH_PREDICT (predict): the same fit over the anchors alone (no border node a point), right after ANCH_FIND, and
// ai: in place of everything it writes, each border node's place through H and that model into meas (w 2): the
// ai: centres of the border's second search (wgsl/register.mjs nodesSource centred, +-0.4 module), the C's
// ai: order (src/acquire.c: the border's search starts from the anchors' model). F7's border search reaches 3 modules
// ai: about the corners' homography and then 0.6 about a homography and k1, and the strongest lenses' sides in a 2:1
// ai: half bow 1.7 to 3 modules (median) and up to 10 from the one, 1 to 1.7 and up to 3.5 from the other; the
// ai: anchors' places come within 0.82 module of the C's map (median 0.1), which a search of +-0.4 reads as F7's
// ai: +-0.6 fine round does (and F7's coarse round before it, 0.1% more over the published lenses for 1 to 2% more of
// ai: the S26's frame). Writes nothing where the fit does not hold (the second search then leaves F7's nodes).
const NPTS = NODES_MAX + ANCH_K;
export const anchRefitSource = ({ predict = false } = {}) => DIMS + MAP + /* wgsl */ `
struct Fit { count: vec4u, base: vec4u, modules: vec4f }
struct Lattice { at: array<vec4u, ${RING_COUNT}>, lat: array<vec4f, ${(RING_COUNT * LATTICE_MAX) / 4}> }
struct Anchors { k: array<vec4f, ${3 * RING_COUNT}> }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(2) var<storage, read> sel: array<vec4u>;
// ai: meas and maps bound read-write (ANCH_PREDICT writes meas; maps is only read, its layout kept): one layout for both forms
@group(0) @binding(3) var<storage, read_write> meas: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> maps: array<array<f32, 16>>;
@group(0) @binding(5) var<storage, read_write> resid: array<vec4f>;
@group(0) @binding(6) var<uniform> P: Fit;
@group(0) @binding(7) var<storage, read_write> anch: array<vec4f>;
@group(0) @binding(8) var<uniform> LAT: Lattice;
@group(0) @binding(9) var<uniform> A: Anchors;
fn latAt(i: u32) -> f32 { return LAT.lat[i >> 2u][i & 3u]; }
fn knot(r: u32, i: u32) -> f32 { return A.k[3u * r + (i >> 2u)][i & 3u]; }
` + COONS + /* wgsl */ `
struct LensFrame { ctr: vec2f, sig: f32, smid: f32, ish: f32, kmax: u32, nw: u32, on: u32 }
const NK: u32 = ${ANCH_K}u;
const SIDE: u32 = ${ANCH_SIDE}u;
// ai: the points (the border's nodes 0 to count - 1, then the anchors): (place through H, displacement from H) in px,
// ai: their weights (0: not a point), a border node's distance from the model, its drop now and before
var<workgroup> pp: array<vec4f, ${NPTS}>;
var<workgroup> pw: array<f32, ${NPTS}>;
var<workgroup> pr: array<f32, ${NODES_MAX}>;
var<workgroup> pd: array<u32, ${NODES_MAX}>;
var<workgroup> pd2: array<u32, ${NODES_MAX}>;
var<workgroup> ak: array<u32, ${ANCH_K}>;
// ai: the lens fit's working, as REFIT_POST's (wgsl/register.mjs lensFit)
var<workgroup> LF: LensFrame;
var<workgroup> LS: array<array<f32, 77>, 16>;
var<workgroup> LT: array<array<f32, ${LENS_P}>, 4>;
var<workgroup> LK: array<u32, 4>;
var<workgroup> LR: array<vec4f, 64>;
var<workgroup> LQ: array<vec4f, 64>;
var<workgroup> LP: array<f32, 77>;
var<workgroup> LZ: Lens;
var<workgroup> rw: array<vec4f, ${NODES_MAX}>;
// ai: what lane 0 says to every lane (workgroupUniformLoad), the model fitted at least once, the border's median
// ai: distance, its weighted nodes, half a module in px
var<workgroup> wu: u32;
var<workgroup> wok: u32;
var<workgroup> wmed: f32;
var<workgroup> wnb: u32;
var<workgroup> wtau: f32;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let pick = sel[f];
  if (frames[f].valid == 0u || pick.y == 0u) { return; }
  let F = frames[f];
  let rgi = pick.x;
  let count = P.count[rgi];
  let base = P.base[rgi];
  if (li < NK) { ak[li] = select(0u, 1u, anch[NK * f + li].w > 0.5); }
  workgroupBarrier();
  if (li == 0u) {
    var c = 0u;
    for (var q = 0u; q < NK; q++) { c += ak[q]; }
    wu = c;
  }
  if (workgroupUniformLoad(&wu) < ${ANCH_MIN}u) { return; }
  let M = maps[f];
  let L0 = lensOf(f);
  let at = LAT.at[rgi];
  let first = at.x;
  let nl = at.y;
  for (var n = li; n < count; n += 64u) {
    let e = meas[f * ${NODES_MAX}u + n];
    let ok = e.w != 0.0 && e.z >= ${WMIN};
    let p = toImage(M, nodes[base + n].xy);
    pp[n] = vec4f(p, select(vec2f(0.0), e.xy - p, ok));
    pw[n] = ${predict ? "0.0" : "select(0.0, e.z, ok)"};
    pd[n] = 0u;
  }
  if (li < NK) {
    let e = anch[NK * f + li];
    let X = vec2f(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    let p = toImage(M, X);
    let h = p;
    let on = ak[li] == 1u;
    pp[count + li] = vec4f(h, select(vec2f(0.0), p + e.xy + lensAt(L0, p) + coons(f, first, nl, X) - h, on));
    pw[count + li] = select(0.0, 1.0, on);
  }
  // ai: every read of the old map done before anything is written over it
  storageBarrier();
  workgroupBarrier();
  let npts = count + NK;
  let aw = F.size;
  var LA = vec2f(0.5 * f32(F.w), 0.5 * f32(F.h));
  if (aw != 0u) { LA = vec2f(0.5 * f32(aw & 0xffffu) - 16384.0, 0.5 * f32(aw >> 16u) - 16384.0); }
  let fwh = vec2f(f32(F.w), f32(F.h));
  let lR = max(max(length(LA), length(vec2f(fwh.x, 0.0) - LA)), max(length(vec2f(0.0, fwh.y) - LA), length(fwh - LA)));
  let lir = 1.0 / max(lR, 1.0);
  // ai: the fit's frame over every point (lensFit's), and lane 0's: the border's weighted nodes and half a module (the
  // ai: lattice's four corners through H: their sides' px over their modules)
  {
    var lo = vec2f(3.0e38);
    var hi = vec2f(-3.0e38);
    var smin = 3.0e38;
    var smax = 0.0;
    var nw = 0u;
    for (var i = li; i < npts; i += 64u) {
      if (!(pw[i] > 0.0)) { continue; }
      let p = pp[i].xy;
      lo = min(lo, p);
      hi = max(hi, p);
      let q = (p - LA) * lir;
      let s = dot(q, q);
      smin = min(smin, s);
      smax = max(smax, s);
      nw += 1u;
    }
    LR[li] = vec4f(lo, hi);
    LQ[li] = vec4f(smin, smax, f32(nw), 0.0);
  }
  workgroupBarrier();
  if (li == 0u) {
    var lo = LR[0].xy;
    var hi = LR[0].zw;
    var smin = LQ[0].x;
    var smax = LQ[0].y;
    var nw = u32(LQ[0].z);
    for (var s = 1u; s < 64u; s++) {
      lo = min(lo, LR[s].xy);
      hi = max(hi, LR[s].zw);
      smin = min(smin, LQ[s].x);
      smax = max(smax, LQ[s].y);
      nw += u32(LQ[s].z);
    }
    LF = LensFrame(0.5 * (lo + hi), 2.0 / max(max(hi.x - lo.x, hi.y - lo.y), 1.0), 0.5 * (smin + smax),
      2.0 / max(smax - smin, 1e-6), 3u, nw, select(0u, 1u, nw > 0u));
    var nb = 0u;
    for (var n = 0u; n < count; n++) { nb += select(0u, 1u, pw[n] > 0.0); }
    wnb = nb;
    let nq = (count + 4u) / 4u;
    var px = 0.0;
    var md = 0.0;
    for (var c = 0u; c < 4u; c++) {
      let a = (c * (nq - 1u)) % count;
      let b = (((c + 1u) % 4u) * (nq - 1u)) % count;
      px += distance(pp[a].xy, pp[b].xy);
      md += distance(nodes[base + a].xy, nodes[base + b].xy);
    }
    wtau = 0.5 * px / max(md, 1.0);
    wok = 0u;
  }
  workgroupBarrier();
  for (var rd = 0u; rd < 4u; rd++) {
    // ai: the normal equations over the points not dropped, 16 lanes a share
    if (li < 16u) {
      var acc = array<f32, 77>();
      for (var i = li; i < npts; i += 16u) {
        let w = select(pw[i], 0.0, i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u);
        if (!(w > 0.0)) { continue; }
        let p = pp[i].xy;
        let r = pp[i].zw;
        ${lensNode}
        ${lensAccumulate}
      }
      LS[li] = acc;
    }
    workgroupBarrier();
    for (var e = li; e < 77u; e += 64u) {
      var a = 0.0;
      for (var s = 0u; s < 16u; s++) { a += LS[s][e]; }
      LP[e] = a;
    }
    workgroupBarrier();
    if (li == 0u) {
      for (var k = 0u; k < 4u; k++) { LK[k] = 0u; }
${lensNormalSolve}    }
    workgroupBarrier();
    {
      var rs = vec4f(0.0);
      var nu = 0u;
      for (var i = li; i < npts; i += 64u) {
        let w = select(pw[i], 0.0, i < count && pd[min(i, ${NODES_MAX - 1}u)] == 1u);
        if (!(w > 0.0)) { continue; }
        nu += 1u;
        let p = pp[i].xy;
        let r = pp[i].zw;
        ${lensNode}
        ${[0, 1, 2, 3].map((K) => `if (LK[${K}] != 0u) { let e = r - ${lensModel(K)}; rs[${K}] += w * dot(e, e); }`).join("\n        ")}
      }
      LR[li] = rs;
      LQ[li] = vec4f(f32(nu), 0.0, 0.0, 0.0);
    }
    workgroupBarrier();
    if (li == 0u) {
      var rss = vec4f(0.0);
      var nu = 0u;
      for (var s = 0u; s < 64u; s++) { rss += LR[s]; nu += u32(LQ[s].x); }
      var kb = 4u;
      var best = 3.0e38;
      let nobs = 2.0 * f32(nu);
      for (var k = 0u; k < 4u; k++) {
        let pk = 8.0 + f32(k);
        if (LK[k] == 0u || nobs - pk - 1.0 <= 0.0) { continue; }
        let score = nobs * log(max(rss[k], 1e-30) / nobs) + 2.0 * pk + 2.0 * pk * (pk + 1.0) / (nobs - pk - 1.0);
        if (score < best) { best = score; kb = k; }
      }
      var tk = array<f32, ${LENS_P}>();
      ${[0, 1, 2, 3].map((K) => `if (kb == ${K}u) { ${Array.from({ length: LENS_P }, (_, a) => `tk[${a}] = LT[${K}][${a}];`).join(" ")} }`).join("\n      ")}
      if (kb < 4u) {
        LZ = Lens(vec4f(LF.ctr, LF.sig, LF.smid), vec4f(LA, lir, LF.ish), vec4f(tk[0], tk[1], tk[2], tk[3]),
          vec4f(tk[4], tk[5], tk[6], tk[7]), vec4f(tk[8], tk[9], tk[10], 0.0));
        wok = 1u;
      }
      wu = select(0u, 1u, kb < 4u);
    }
    if (workgroupUniformLoad(&wu) == 0u || rd == 3u) { break; }
    // ai: the border's nodes against the model, their median (the node of rank nb / 2), the drops
    let lz = LZ;
    for (var n = li; n < count; n += 64u) { pr[n] = select(0.0, length(pp[n].zw - lensAt(lz, pp[n].xy)), pw[n] > 0.0); }
    if (li == 0u) { wmed = 0.0; }
    workgroupBarrier();
    let nb = wnb;
    for (var n = li; n < count; n += 64u) {
      if (!(pw[n] > 0.0)) { continue; }
      let v = pr[n];
      var rank = 0u;
      for (var j = 0u; j < count; j++) { let u = pr[j]; rank += select(0u, 1u, pw[j] > 0.0 && (u < v || (u == v && j < n))); }
      if (rank == nb / 2u) { wmed = v; }
    }
    workgroupBarrier();
    let tau = max(wtau, 3.0 * 1.4826 * wmed);
    for (var n = li; n < count; n += 64u) { pd2[n] = select(0u, 1u, pw[n] > 0.0 && pr[n] > tau); }
    workgroupBarrier();
    if (li == 0u) {
      var c = 0u;
      for (var n = 0u; n < count; n++) { c += select(0u, 1u, pd2[n] != pd[n]); pd[n] = pd2[n]; }
      wu = c;
    }
    if (workgroupUniformLoad(&wu) == 0u) { break; }
  }
  if (workgroupUniformLoad(&wok) == 0u) { return; }
  let lz = LZ;
${predict ? `  for (var n = li; n < count; n += 64u) { meas[f * ${NODES_MAX}u + n] = vec4f(pp[n].xy + lensAt(lz, pp[n].xy), 0.0, 2.0); }
  return;
` : ""}  if (li == 0u) {
    let b = f * ${RESID}u + ${NODES_MAX}u;
    resid[b] = lz.c;
    resid[b + 1u] = lz.a;
    resid[b + 2u] = lz.t0;
    resid[b + 3u] = lz.t1;
    resid[b + 4u] = lz.t2;
  }
  for (var n = li; n < count; n += 64u) {
    let v = select(vec4f(pp[n].zw - lensAt(lz, pp[n].xy), pw[n], 0.0), vec4f(0.0, 0.0, pw[n], 0.0), pd[n] == 1u);
    resid[f * ${RESID}u + n] = v;
    rw[n] = v;
  }
  storageBarrier();
  workgroupBarrier();
` + SIDE_FITS + /* wgsl */ `
  storageBarrier();
  workgroupBarrier();
  if (li < NK && ak[li] == 1u) {
    let X = vec2f(knot(rgi, 1u + li % SIDE), knot(rgi, 1u + li / SIDE));
    let e = anch[NK * f + li];
    let q = pp[count + li];
    anch[NK * f + li] = vec4f(q.zw - lensAt(lz, q.xy) - coons(f, first, nl, X), e.zw);
  }
}
`;
export const ANCH_REFIT = anchRefitSource(), ANCH_PREDICT = anchRefitSource({ predict: true });
