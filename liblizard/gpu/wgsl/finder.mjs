// F5 to F6: from F4's readings of the bank's peaks to a map. No list is walked on the host and no stage stops early:
// ai: every reading votes, the strongest vote peaks each make a quad, and every (quad, orientation, ring) is scored
// ai: against the track. The frame's answer is an argmax, written straight into the map and the ring F7 registers with.
// ai: The ring is all the finder decides: which picture the ring carries is the format word's to say (F8).
//
// ai: Geometry (src/layout.c, gpu/tables.mjs): a corner mark is the square [0, 12)^2 of the symbol, the border's dark
// ai: line continued at depths 1-2, a light gap at 3-4, a 6 x 6 dark core at [5, 11) and a light edge 1 module wide
// ai: inside it, centre 8 modules in from each edge. Blurred until the gap
// ai: fuses it is a solid 10 x 10 square, centre 6 in. Adjacent centres are M - 16 (or M - 12) modules apart on a
// ai: symbol M modules a side: M = 94, 158, 286 or 542 (the rings of 32, 64, 128 and 256 band cells; 222, the 96's,
// ai: until 2026-10-10).
import { DIMS } from "./common.mjs";
import { RINGS as RING_CELLS } from "../../sim/lizard_pick.mjs";

// ai: A mark's centre depth in modules by reading form: the gapped core, the merged square.
export const CENTRE = [8, 6];
export const K = 4;          // vote peaks a frame, so quads a frame
export const CELL = 16;      // accumulator cell, level-0 pixels
export const ORIENT = 8;     // 4 rotations x mirror
// ai: The rings a hypothesis may be (sim/lizard_pick.mjs RINGS): an accumulator and a module count each. Four at most:
// ai: the stages' uniforms hold the rings' module counts, node counts and lattices in one vec4 (P5, Pick, register.mjs
// ai: Fit and Round, sample.mjs Lattice, gpu/decoder.mjs).
export const RING_COUNT = RING_CELLS.length;
if (RING_COUNT > 4) throw new Error(`${RING_COUNT} rings: the finder's uniforms hold four`);
// ai: GATHER's reach runs from the first ring's to the last's, so the rings go smallest first.
if (RING_CELLS.some((c, i) => i && c <= RING_CELLS[i - 1])) throw new Error(`rings ${RING_CELLS.join(", ")}: smallest first`);
export const HYP = K * ORIENT * RING_COUNT;   // ai: hypotheses a frame
// ai: A reading votes, and supports a quad, when the network's mark probability p reaches SMIN (the trainer reports
// ai: at 0.5).
export const SMIN = 0.35;
// A vote ray runs from RAY_LO to RAY_HI times the reach the corner's own module size implies (perspective moves
// the centre within that), in a band RAY_SPREAD of the reach wide (an angle error of about 5 degrees).
export const RAY_LO = 0.45, RAY_HI = 2.2, RAY_SPREAD = 0.09;
// F6 reads the track only this many modules from a corner: the four corners' homography is exact there, while lens
// distortion bows the middle of a side by modules (k1 = 0.06). Orientation and module count need no more.
export const NEAR = 40;

// Every level of the pyramid, read by level-0 coordinates.
const LEVELS = /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }   // d[l] = (stride, rows) of level l's buffer
fn pix(l: u32, f: u32, x: i32, y: i32) -> f32 {
  let F = frames[f];
  let w = i32((F.w + (1u << l) - 1u) >> l);
  let h = i32((F.h + (1u << l) - 1u) >> l);
  let xc = clamp(x, 0, w - 1);
  let yc = clamp(y, 0, h - 1);
  if (l == 0u) { return textureLoad(img, vec2i(xc, yc), i32(f), 0).r; }
  let s = LD.d[l];
  let i = (f * s.y + u32(yc)) * s.x + u32(xc);
  switch l {
    case 1u: { return lv1[i]; }
    case 2u: { return lv2[i]; }
    case 3u: { return lv3[i]; }
    default: { return lv4[i]; }
  }
}
// p in level-0 pixels (pixel centres at i + 0.5 on every level), bilinear on level l.
fn look(l: u32, f: u32, p: vec2f) -> f32 {
  let q = p / f32(1u << l) - vec2f(0.5);
  let q0 = floor(q);
  let a = q - q0;
  let i = vec2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
// The level where a module of u level-0 pixels is 1.5 to 3 pixels: the blur a module-sized feature wants.
fn levelFor(u: f32) -> u32 { return u32(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0)); }
fn dir(a: f32) -> vec2f { return vec2f(cos(a), sin(a)); }
`;
const LEVEL_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> lv1: array<f32>;
@group(0) @binding(2) var<storage, read> lv2: array<f32>;
@group(0) @binding(3) var<storage, read> lv3: array<f32>;
@group(0) @binding(4) var<storage, read> lv4: array<f32>;
@group(0) @binding(5) var<storage, read> frames: array<Frame>;
@group(0) @binding(6) var<uniform> LD: LevelDims;
`;
const PI = /* wgsl */ `const PI = 3.14159265;`;

// ai: readings[f][i], i < cap (the kept peaks' slots), written by F4 (wgsl/classify_gemm.mjs, the network): (x, y, u,
// ai: alpha), (p, 0, 0, 0), (form, response, level, valid). alpha + 45 degrees points out of the symbol along the
// ai: mark's diagonal; p is the mark probability; form picks CENTRE. The three zeros are the slots where the deleted
// ai: hand-written F4 (DESCRIBE, 2026-09-26) scored the other diagonals.

export const SUPPRESS = 10;
// ai: The peaks a frame's list holds: any two lie SUPPRESS + 1 cells apart, so a 4K frame's 240 x 135 cells hold at
// ai: most 286.
export const PEAK_LIST = 512;
// ai: F5 in two dispatches since 2026-09-30 (the small stages merged for organization; three before):
// ai: VOTE, a lane a reading, then GATHER, a workgroup a frame: its K centre peaks (F5b, PEAKS before) and each one's
// ai: quad (F5c), the peaks passed to the quads through workgroup memory as well as cand (read back by the probe), the
// ai: K quads gathered side by side, a group of 256 / K lanes each with its own keys. VOTE stays its own dispatch: in
// ai: one kernel with the rest (a workgroup a frame, 256 lanes, PEAKS' 8 KB list) the S26's stage took 0.06 ms a frame
// ai: against the three dispatches' 0.03, the votes' workgroups fewer and fewer of them at once. The same accumulator,
// ai: candidates and quads word for word (scripts/pages/gpu_selftest_stages.mjs hashes them).
// F5a: votes. A reading at a corner of a symbol M modules a side has that symbol's centre along the inward
// diagonal, (M - 2c) / sqrt(2) modules away (c its centre's depth: 8 gapped, 6 merged) at the module size the
// symbol has THERE; seen at an angle the module size changes across the symbol by up to 3x, so the distance is
// not known from the corner alone. What is known is the direction, and the image of the centre is where the two
// diagonals cross whatever the perspective. So every reading votes along a RAY, from 0.45 to 2.2 times the reach
// ai: its own module size implies, into a coarse accumulator per (frame, ring), as a band SPREAD wide in cells (an
// angle error of a few degrees over hundreds of pixels). Four rays cross at the centre and each is one vote
// ai: elsewhere. One lane a reading; it votes when its p reaches SMIN.
export const VOTE = DIMS + PI + /* wgsl */ `
struct P5 { cap: u32, gx: u32, gy: u32, p0: u32, modules: vec4f }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> readings: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> P: P5;
const CENTRE = array<f32, 2>(${CENTRE[0]}, ${CENTRE[1]});
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let f = g.y;
  let i = g.x;
  if (i >= P.cap || frames[f].valid == 0u) { return; }
  let o = 3u * (f * P.cap + i);
  let r0 = readings[o];
  let r2 = readings[o + 2u];
  let p = readings[o + 1u].x;
  if (r2.w == 0.0 || p < ${SMIN}) { return; }
  let beta = r0.w + PI / 4.0;
  let inward = -vec2f(cos(beta), sin(beta));
  let perp = vec2f(-inward.y, inward.x);
  let c = CENTRE[u32(r2.x)];
  let w = u32(clamp(p, 0.0, 1.5) * 256.0);
  // The frame's own cells, not the lane's: a lane sized for a larger frame earlier would otherwise keep the votes
  // that run past this frame's edge, and a frame's answer would depend on which lane it ran in.
  let gxf = i32((frames[f].w + ${CELL - 1}u) / ${CELL}u);
  let gyf = i32((frames[f].h + ${CELL - 1}u) / ${CELL}u);
  // ai: A ray stops at the grid's farthest corner from the reading (and a cell of slack): a cell a ray reaches at d
  // ai: lies at least d from the reading, so past that no cell is in the grid and the votes are the same. The ray's
  // ai: length is the reading's module size, the network's uw exp(dlogu), which nothing bounds: a reading whose
  // ai: size was absurd walked this loop until the device was lost (2026-09-30, found with a kernel that wrote
  // ai: such readings; STATUS "Native speed"). A reach that is not a number walks nothing.
  let far = length(max(r0.xy, vec2f(f32(gxf), f32(gyf)) * ${CELL}.0 - r0.xy)) + ${CELL}.0;
  for (var m = 0u; m < ${RING_COUNT}u; m++) {
    let reach = (P.modules[m] - 2.0 * c) / sqrt(2.0) * r0.z;
    let d0 = ${RAY_LO} * reach;
    let d1 = min(${RAY_HI} * reach, far);
    let spread = i32(clamp(ceil(${RAY_SPREAD} * reach / ${CELL}.0), 1.0, 6.0));
    // One step a cell along the ray; at each, a triangle across the band. A cell is hit by more than one step
    // of the same ray when the ray runs near an axis, which is the same small bias for every ray.
    for (var d = d0; d <= d1; d += ${CELL}.0) {
      let at = (r0.xy + d * inward) / ${CELL}.0;
      for (var q = -spread; q <= spread; q++) {
        let cell = at + f32(q) * perp;
        let x = i32(floor(cell.x));
        let y = i32(floor(cell.y));
        if (x < 0 || y < 0 || x >= gxf || y >= gyf) { continue; }
        let t = 1.0 - f32(abs(q)) / f32(spread + 1);
        atomicAdd(&acc[((f * ${RING_COUNT}u + m) * P.gy + u32(y)) * P.gx + u32(x)], u32(f32(w) * t));
      }
    }
  }
}
`;

// ai: F5b: the frame's K strongest centre peaks, one workgroup a frame. cand[f][k] = (x, y, ring, height) in level-0
// ai: pixels. One peak a symbol centre (2026-09-27, two symbols a frame): the rings are merged into one grid, each cell
// ai: its tallest accumulator (and the ring it came from, the first on a tie), and a cell is a peak when it is the
// ai: tallest within SUPPRESS cells each way. Every reading votes into every ring, so a symbol's centre peaks in each
// ai: ring's accumulator, and the long rings' wide bands give several maxima round it: the four tallest maxima a ring
// ai: went to one symbol and left a second symbol none in 446 of 768 pair frames (STATUS "Two codes in one frame"); a
// ai: JS replay of this rule on those frames put both centres first and second in 766 and in the top four in all 768.
// ai: A tie in height goes to the earlier cell (row, then column; within a window, the earlier in raster order), never
// ai: to the order the lanes appended in, so two runs of one frame pick the same peaks in the same order.
// F5c: the readings that voted for a peak make its quad, 256 / K lanes a (frame, peak). A reading supports the peak
// ai: if its ray passes it (VOTE's band, at a distance its module size allows for some ring: PEAKS merges the rings, so
// ai: a peak is a centre, not a ring, and which ring the symbol has is SCORE's to say). The strongest
// supporter sets the symbol's rotation and module size; every supporter of a like module size is put in the corner
// slot its direction from the centre says (corner q at alpha + 45 + 90 q degrees), and each slot keeps its
// strongest.
// ai: Strongest is by key (keyOf): the weight, then the proposer's response at the reading's peak, then its image
// ai: position (y, then x), then its index. The network's p and the proposer's sigmoid both saturate at 1 in f32 on a
// ai: sharp mark, so weight and response can tie bit for bit; a tie left to the index followed the peak list's order,
// ai: which is the order SELECT's appends landed in (wgsl/bank.mjs), and one frame kept a mark's gapped reading in
// ai: some runs and its merged one in others (the self-test's frame 1, 2026-09-27). The index now decides only between
// ai: two readings at the same position, bit for bit.
// ai: The weight is counted in thousandths (2026-09-27): two readings of one mark whose p differ further down (0.9999975
// ai: against 0.9999955) tie on it and the response decides, so which of them leads its slot no longer turns on
// ai: whether the other's peak was kept (SELECT's tiles keep a lanes'-order share of an overfull list), which moved
// ai: with the batch: the self-test's frame 1, once PEAKS gave one peak a centre.
// ai: quads[f][k]: 4 corners (x, y, depth c, present), then (the peak's ring, module px as that ring's, mass, corners
// ai: present).
// ai: The lanes that gather one peak's quad: the K peaks gather side by side, each group with its own keys.
const GROUP = 256 / K;
if (!Number.isInteger(GROUP) || GROUP < 16) throw new Error(`K ${K}: 256 lanes do not split into K groups of 16 or more`);
export const GATHER = DIMS + PI + /* wgsl */ `
struct P5 { cap: u32, gx: u32, gy: u32, p0: u32, modules: vec4f }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> readings: array<vec4f>;
@group(0) @binding(2) var<storage, read> acc: array<u32>;
@group(0) @binding(3) var<storage, read_write> cand: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> quads: array<vec4f>;
@group(0) @binding(5) var<uniform> P: P5;
const CENTRE = array<f32, 2>(${CENTRE[0]}, ${CENTRE[1]});
var<workgroup> list: array<vec4f, ${PEAK_LIST}>;
var<workgroup> nlist: atomic<u32>;
var<workgroup> peak: array<vec4f, ${K}>;
// ai: The best key so far of peak pk's lead (leadK[4 pk + j], word j) and of each of its corner slots (slotK[16 pk + 4 q
// ai: + j]): word j is the highest among the candidates whose words before j are the ones already kept.
var<workgroup> leadK: array<atomic<u32>, ${4 * K}>;
var<workgroup> slotK: array<atomic<u32>, ${16 * K}>;
// ai: Candidate a ranks before b: taller, or as tall and at an earlier cell (row, column). Cells are exact in f32.
fn ahead(a: vec4f, b: vec4f) -> bool {
  return a.w > b.w || (a.w == b.w && (a.y < b.y || (a.y == b.y && a.x < b.x)));
}
// The frame's own cells (gw x gh), addressed in the lane's (P.gx x P.gy): which cells an invocation scans, and so
// which maxima share one, is then the frame's alone, not the largest frame's the lane has held.
fn accAt(f: u32, m: u32, x: i32, y: i32, gw: i32, gh: i32) -> u32 {
  if (x < 0 || y < 0 || x >= gw || y >= gh) { return 0u; }
  return acc[((f * ${RING_COUNT}u + m) * P.gy + u32(y)) * P.gx + u32(x)];
}
// ai: The rings merged: the cell's tallest accumulator (0 off the frame's cells).
fn merged(f: u32, x: i32, y: i32, gw: i32, gh: i32) -> u32 {
  var v = 0u;
  for (var m = 0u; m < ${RING_COUNT}u; m++) { v = max(v, accAt(f, m, x, y, gw, gh)); }
  return v;
}
// ai: Whether the cell (dx, dy) from one holding v outranks it: taller, or as tall and earlier in raster order.
fn outranks(f: u32, x: i32, y: i32, v: u32, dx: i32, dy: i32, gw: i32, gh: i32) -> bool {
  let q = merged(f, x + dx, y + dy, gw, gh);
  return q > v || (q == v && (dy < 0 || (dy == 0 && dx < 0)));
}
// ai: Whether cell (x, y), holding v, is the tallest of the merged grid within r cells each way. The window is read a
// ai: ring at a time from the middle out, so a cell on a ridge meets its taller neighbour in the first ring or two and
// ai: only a peak reads all of it (the rule is a cell's against each other's, so the order moves no answer).
fn tallest(f: u32, x: i32, y: i32, v: u32, r: i32, gw: i32, gh: i32) -> bool {
  for (var d = 1; d <= r; d++) {
    for (var k = -d; k <= d; k++) {
      if (outranks(f, x, y, v, k, -d, gw, gh) || outranks(f, x, y, v, k, d, gw, gh) || outranks(f, x, y, v, -d, k, gw, gh) || outranks(f, x, y, v, d, k, gw, gh)) { return false; }
    }
  }
  return true;
}
// ai: Reading i's key, four words compared in turn, the fields packed across them so each compares exactly: the
// ai: weight (its p in thousandths, floor(clamp(p, 0, 1.5) x 1000) x 300 in 19 bits), the response,
// ai: y, x (f32 bits, which order a float that is not negative as a u32; the sign bit cleared so -0 is 0), the index
// ai: (13 bits, cap <= 8192). Every supporter's first word is at least 349 x 300 << 13, so 0 there means none.
fn keyOf(f: u32, i: u32) -> vec4u {
  let o = 3u * (f * P.cap + i);
  let r0 = readings[o];
  let w = u32(floor(clamp(readings[o + 1u].x, 0.0, 1.5) * 1000.0)) * 300u;
  let rb = bitcast<u32>(readings[o + 2u].y) & 0x7fffffffu;
  let yb = bitcast<u32>(max(r0.y, 0.0)) & 0x7fffffffu;
  let xb = bitcast<u32>(max(r0.x, 0.0)) & 0x7fffffffu;
  return vec4u((w << 13u) | (rb >> 19u), ((rb & 0x7ffffu) << 13u) | (yb >> 19u), ((yb & 0x7ffffu) << 13u) | (xb >> 19u),
               ((xb & 0x7ffffu) << 13u) | i);
}
// ai: Whether key k's words before j are the lead's (a slot's, at word 4 q) kept so far.
fn leadHas(k: vec4u, pk: u32, j: u32) -> bool {
  var ok = true;
  for (var t = 0u; t < j; t++) { ok = ok && k[t] == atomicLoad(&leadK[4u * pk + t]); }
  return ok;
}
fn slotHas(k: vec4u, pk: u32, q: u32, j: u32) -> bool {
  var ok = true;
  for (var t = 0u; t < j; t++) { ok = ok && k[t] == atomicLoad(&slotK[16u * pk + 4u * q + t]); }
  return ok;
}
fn support(f: u32, i: u32, c: vec4f) -> bool {
  // ai: Whether reading i votes for this peak: its ray passes within VOTE's band of the peak, at a distance along it
  // ai: that its module size allows.
  let o = 3u * (f * P.cap + i);
  let r0 = readings[o];
  let r2 = readings[o + 2u];
  if (r2.w == 0.0 || readings[o + 1u].x < ${SMIN}) { return false; }
  // ai: From RAY_LO times the smallest ring's reach to RAY_HI times the largest's: every ring's range, one piece (a
  // ai: ring's spans 4.9 times its reach, the next ring's reach is under 2 times it: 1.95 from the 128 ring to the 256). With the peak's ring's range
  // ai: alone, a 64-ring symbol's peak, labelled by the 128 ring's taller accumulator, lost its near corner at 45
  // ai: degrees (gpu_front's 45 deg cell, 2026-09-27).
  let edge = CENTRE[u32(r2.x)];
  let near = (P.modules[0] - 2.0 * edge) / sqrt(2.0) * r0.z;
  let far = (P.modules[${RING_COUNT - 1}] - 2.0 * edge) / sqrt(2.0) * r0.z;
  let beta = r0.w + PI / 4.0;
  let inward = -vec2f(cos(beta), sin(beta));
  let d = c.xy - r0.xy;
  let along = dot(d, inward);
  let across = abs(d.x * inward.y - d.y * inward.x);
  if (along < ${RAY_LO} * near || along > ${RAY_HI} * far) { return false; }
  return across <= max(2.0 * ${CELL}.0, ${RAY_SPREAD} * along + ${CELL}.0);
}
// ai: The corner slot supporter i fills (4: none) about centre c: a slot by its direction from the centre, for a
// ai: supporter whose module size is within the spread of the lead's (u) that one symbol's corners have (4x at 45
// ai: degrees of yaw, measured; a symbol of another size next to this one is further off).
fn cornerOf(f: u32, i: u32, c: vec4f, u: f32, alpha: f32) -> u32 {
  if (!support(f, i, c)) { return 4u; }
  let r = readings[3u * (f * P.cap + i)];
  if (r.z < 0.2 * u || r.z > 5.0 * u) { return 4u; }
  let a = atan2(r.y - c.y, r.x - c.x) - alpha - PI / 4.0;
  return u32(i32(round(a / (PI / 2.0))) & 3);
}
// ai: The image position of the reading slot q's key holds.
fn slotAt(f: u32, pk: u32, q: u32) -> vec2f { return readings[3u * (f * P.cap + (atomicLoad(&slotK[16u * pk + 4u * q + 3u]) & 8191u))].xy; }
// ai: The centre the slots' corners put the symbol at: with all four the crossing of the diagonals (a projective
// ai: square's centre), else the middle of the one filled diagonal pair (p02: slots 0 and 2, p13: 1 and 3).
fn centreOf(f: u32, pk: u32, p02: bool, p13: bool) -> vec2f {
  let a = slotAt(f, pk, 0u);
  let b = slotAt(f, pk, 2u);
  let e = slotAt(f, pk, 1u);
  let g = slotAt(f, pk, 3u);
  let r = b - a;
  let s = g - e;
  let den = r.x * s.y - r.y * s.x;
  if (p02 && p13 && abs(den) > 1e-3) { return a + ((e.x - a.x) * s.y - (e.y - a.y) * s.x) / den * r; }
  return select(0.5 * (e + g), 0.5 * (a + b), p02);
}
// ai: F5c: the quad of peak pk (c).
fn gather(f: u32, pk: u32, c: vec4f, lane: u32) {
  let live = frames[f].valid != 0u && c.w > 0.0;
  // ai: The lead: the highest key over the supporters, a word a pass. After the first pass a reading whose earlier
  // ai: words are not the lead's is passed over before its ray is tested.
  for (var j = 0u; j < 4u; j++) {
    for (var i = lane; i < P.cap; i += ${GROUP}u) {
      if (!live) { break; }
      let k = keyOf(f, i);
      if (leadHas(k, pk, j) && support(f, i, c)) { atomicMax(&leadK[4u * pk + j], k[j]); }
    }
    workgroupBarrier();
  }
  let lead = atomicLoad(&leadK[4u * pk]) != 0u;
  let lr = readings[3u * (f * P.cap + (atomicLoad(&leadK[4u * pk + 3u]) & 8191u))];
  let alpha = lr.w;
  // ai: Two rounds, each keying its slots as the lead's (a word a pass over the supporters each slot takes): the band
  // ai: about the peak; then, where that filled a diagonal pair, every slot again about the centre those corners put
  // ai: the symbol at (centreOf). The peak is a cell of the merged rings (PEAKS), and which of two near-tied cells it
  // ai: is moves with the batch, while beside a second symbol it sat 3 to 4 cells off the rays' crossing, pulled there
  // ai: by that symbol's long-ring rays: about the peak alone a corner missed the band or a mark's other reading took
  // ai: its slot (2026-09-27, the pair scenes and the self-test's frame 1); about the corners' own centre the same
  // ai: readings win whichever cell the peak was.
  var at = c;
  var refined = false;
  for (var round = 0u; round < 2u; round++) {
    if (round == 1u) {
      let p02 = atomicLoad(&slotK[16u * pk]) != 0u && atomicLoad(&slotK[16u * pk + 8u]) != 0u;
      let p13 = atomicLoad(&slotK[16u * pk + 4u]) != 0u && atomicLoad(&slotK[16u * pk + 12u]) != 0u;
      refined = lead && (p02 || p13);
      if (refined) { at = vec4f(centreOf(f, pk, p02, p13), c.z, c.w); }
    }
    workgroupBarrier();
    // ai: The second round starts over: every slot cleared once each invocation has read them.
    if (round == 1u && refined && lane < 16u) { atomicStore(&slotK[16u * pk + lane], 0u); }
    workgroupBarrier();
    let go = lead && (round == 0u || refined);
    for (var j = 0u; j < 4u; j++) {
      // ai: The slots' first words, so a reading that shares none of them is passed over before its ray is tested.
      let firsts = vec4u(atomicLoad(&slotK[16u * pk]), atomicLoad(&slotK[16u * pk + 4u]), atomicLoad(&slotK[16u * pk + 8u]), atomicLoad(&slotK[16u * pk + 12u]));
      for (var i = lane; i < P.cap; i += ${GROUP}u) {
        if (!go) { break; }
        let k = keyOf(f, i);
        if (j > 0u && !any(firsts == vec4u(k.x))) { continue; }
        let q = cornerOf(f, i, at, lr.z, alpha);
        if (q < 4u && slotHas(k, pk, q, j)) { atomicMax(&slotK[16u * pk + 4u * q + j], k[j]); }
      }
      workgroupBarrier();
    }
  }
  if (lane == 0u) {
    let base = 5u * (f * ${K}u + pk);
    var n = 0u;
    var pts = array<vec4f, 4>();
    for (var q = 0u; q < 4u; q++) {
      if (!lead || atomicLoad(&slotK[16u * pk + 4u * q]) == 0u) { pts[q] = vec4f(0.0); continue; }
      let o = 3u * (f * P.cap + (atomicLoad(&slotK[16u * pk + 4u * q + 3u]) & 8191u));
      pts[q] = vec4f(readings[o].xy, CENTRE[u32(readings[o + 2u].x)], 1.0);
      n++;
    }
    // ai: A quad is its four corners: with fewer SCORE reads none of its hypotheses.
    var u = 0.0;
    if (n == 4u) {
      let m = u32(c.z);
      u = 0.25 * (distance(pts[0].xy, pts[1].xy) + distance(pts[1].xy, pts[2].xy) + distance(pts[2].xy, pts[3].xy) + distance(pts[3].xy, pts[0].xy)) / (P.modules[m] - 2.0 * pts[0].z);
    }
    for (var q = 0u; q < 4u; q++) { quads[base + q] = pts[q]; }
    quads[base + 4u] = vec4f(c.z, u, c.w, f32(n));
  }
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let gw = i32((frames[f].w + ${CELL - 1}u) / ${CELL}u);
  let gh = i32((frames[f].h + ${CELL - 1}u) / ${CELL}u);
  let cells = u32(gw * gh);
  // ai: Every peak into the list.
  for (var c = li; c < cells; c += 256u) {
    let x = i32(c % u32(gw));
    let y = i32(c / u32(gw));
    let v = merged(f, x, y, gw, gh);
    if (v == 0u || !tallest(f, x, y, v, ${SUPPRESS}, gw, gh)) { continue; }
    var ring = 0u;
    for (var m = ${RING_COUNT}u; m > 0u; m--) { if (accAt(f, m - 1u, x, y, gw, gh) == v) { ring = m - 1u; } }
    let k = atomicAdd(&nlist, 1u);
    if (k < ${PEAK_LIST}u) { list[k] = vec4f((f32(x) + 0.5) * ${CELL}.0, (f32(y) + 0.5) * ${CELL}.0, f32(ring), f32(v)); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let n = min(atomicLoad(&nlist), ${PEAK_LIST}u);
    for (var k = 0u; k < ${K}u; k++) {
      var b = k;
      for (var q = k + 1u; q < n; q++) { if (ahead(list[q], list[b])) { b = q; } }
      if (k < n) { let e = list[b]; list[b] = list[k]; list[k] = e; cand[f * ${K}u + k] = e; peak[k] = e; } else { peak[k] = vec4f(0.0); }
    }
  }
  workgroupBarrier();
  // ai: the K peaks' quads side by side, a group of GROUP lanes each with its own keys
  gather(f, li / ${GROUP}u, peak[li / ${GROUP}u], li % ${GROUP}u);
}
`;

// A homography from four point pairs, solved by one invocation: module coordinates to image pixels. Both sides
// are normalised first (modules by M, pixels about their centroid) so single precision holds.
// ai: Closed form, with no private array indexed at a runtime value: the S26 Ultra's Adreno 8xx returned the zero
// ai: matrix from the pivoted elimination this replaced (a private array<array<f32, 9>, 8> with data-dependent row
// ai: swaps) on every hypothesis of every frame, with the same inputs the 4090 solves (STATUS.md "S26 SCORE").
// ai: mul3's nine products are written out by JS for the same reason.
const MUL3 = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) =>
  `r[${3 * i + j}] = a[${3 * i}] * b[${j}] + a[${3 * i + 1}] * b[${3 + j}] + a[${3 * i + 2}] * b[${6 + j}];`)).join("\n  ");
const HOMOGRAPHY = /* wgsl */ `
// ai: The unit square's corners (0, 0), (1, 0), (1, 1), (0, 1) to q0..q3 (Heckbert 1989, 2.2.1): the matrix taking
// ai: (u, v) to (a u + b v + c, d u + e v + f) / (g u + h v + 1). Zero when three corners lie on a line.
fn squareToQuad(q0: vec2f, q1: vec2f, q2: vec2f, q3: vec2f) -> array<f32, 9> {
  let d1 = q1 - q2;
  let d2 = q3 - q2;
  let d3 = q0 - q1 + q2 - q3;
  let det = d1.x * d2.y - d2.x * d1.y;
  if (abs(det) < 1e-12) { return array<f32, 9>(); }
  let g = (d3.x * d2.y - d2.x * d3.y) / det;
  let h = (d1.x * d3.y - d3.x * d1.y) / det;
  return array<f32, 9>(q1.x - q0.x + g * q1.x, q3.x - q0.x + h * q3.x, q0.x,
                       q1.y - q0.y + g * q1.y, q3.y - q0.y + h * q3.y, q0.y, g, h, 1.0);
}
// ai: m adj(m) = det(m) I: the inverse up to a scale a homography does not carry.
fn adjugate(m: array<f32, 9>) -> array<f32, 9> {
  return array<f32, 9>(m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                       m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                       m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]);
}
fn mul3(a: array<f32, 9>, b: array<f32, 9>) -> array<f32, 9> {
  var r = array<f32, 9>();
  ${MUL3}
  return r;
}
// ai: src (modules) to dst (pixels): square-to-dst composed with the inverse of square-to-src, scaled to H[8] = 1.
fn solveH(src: array<vec2f, 4>, dst: array<vec2f, 4>, M: f32) -> array<f32, 9> {
  let c = 0.25 * (dst[0] + dst[1] + dst[2] + dst[3]);
  let sc = max(0.25 * (distance(dst[0], c) + distance(dst[1], c) + distance(dst[2], c) + distance(dst[3], c)), 1e-3);
  let S = squareToQuad(src[0] / M, src[1] / M, src[2] / M, src[3] / M);
  let D = squareToQuad((dst[0] - c) / sc, (dst[1] - c) / sc, (dst[2] - c) / sc, (dst[3] - c) / sc);
  let hs = mul3(D, adjugate(S));
  if (S[8] == 0.0 || D[8] == 0.0 || abs(hs[8]) < 1e-12) { return array<f32, 9>(); }
  var hn = array<f32, 9>();
  for (var j = 0; j < 9; j++) { hn[j] = hs[j] / hs[8]; }
  // H = T_dst^-1 Hn T_src, T_src = diag(1/M, 1/M, 1), T_dst^-1 = [[sc, 0, cx], [0, sc, cy], [0, 0, 1]].
  var H = array<f32, 9>();
  for (var j = 0; j < 3; j++) {
    let s = select(1.0, 1.0 / M, j < 2);
    H[j] = (sc * hn[j] + c.x * hn[6 + j]) * s;
    H[3 + j] = (sc * hn[3 + j] + c.y * hn[6 + j]) * s;
    H[6 + j] = hn[6 + j] * s;
  }
  return H;
}
fn applyH(H: array<f32, 9>, m: vec2f) -> vec2f {
  let w = H[6] * m.x + H[7] * m.y + H[8];
  return vec2f(H[0] * m.x + H[1] * m.y + H[2], H[3] * m.x + H[4] * m.y + H[5]) / w;
}
// Corner k of the symbol (TL, TR, BR, BL) at depth c, for a symbol M modules a side.
fn cornerAt(k: u32, c: f32, M: f32) -> vec2f {
  let x = select(c, M - c, k == 1u || k == 2u);
  let y = select(c, M - c, k >= 2u);
  return vec2f(x, y);
}
// Hypothesis o: symbol corner k is quad slot (o + k) mod 4, or (o - k) mod 4 mirrored.
fn slotOf(o: u32, k: u32) -> u32 { return select((o + k) & 3u, (o + 4u - k) & 3u, o >= 4u); }
// ai: quad[s] without indexing a private array at a runtime value (see solveH).
fn atSlot(quad: array<vec4f, 4>, s: u32) -> vec4f {
  return select(select(quad[0], quad[1], s == 1u), select(quad[2], quad[3], s == 3u), s >= 2u);
}
fn hypH(quad: array<vec4f, 4>, o: u32, M: f32) -> array<f32, 9> {
  let p0 = atSlot(quad, slotOf(o, 0u));
  let p1 = atSlot(quad, slotOf(o, 1u));
  let p2 = atSlot(quad, slotOf(o, 2u));
  let p3 = atSlot(quad, slotOf(o, 3u));
  return solveH(array<vec2f, 4>(cornerAt(0u, p0.z, M), cornerAt(1u, p1.z, M), cornerAt(2u, p2.z, M), cornerAt(3u, p3.z, M)),
                array<vec2f, 4>(p0.xy, p1.xy, p2.xy, p3.xy), M);
}
`;

// ai: F6a: every hypothesis (quad, orientation, ring) through the track read, one workgroup each. The read is
// the format's: the solid line and the gap every 16 modules give the contrast, the Manchester pair nearest each
// step gives the threshold (a pair is one dark and one light cell), and each track cell adds its sign's agreement,
// clamped to +-0.5 (gpu/tables.mjs track: the reference codec's own plan). Only points within NEAR modules of a
// corner. Sampled at the level where a module is 1.5 to 3 pixels. score[f][h] in -0.5..0.5, -1 where the quad is
// incomplete or the contrast is too low to read.
export const SCORE = DIMS + LEVEL_BINDINGS + LEVELS + HOMOGRAPHY + /* wgsl */ `
struct Plan { at: array<vec4u, ${RING_COUNT}> }   // ai: per ring: (point offset, reference steps, track cells, modules)
@group(0) @binding(7) var<storage, read> quads: array<vec4f>;
@group(0) @binding(8) var<storage, read> plan: array<vec4f>;   // ai: per ring: points (x, y, sign, 0), refs first
@group(0) @binding(9) var<storage, read_write> score: array<f32>;
@group(0) @binding(10) var<uniform> PL: Plan;
var<workgroup> sums: array<vec4f, 64>;
var<workgroup> H: array<f32, 9>;
// A border point near enough a corner for the corners' homography to hold (its distance along its side).
fn near(m: vec2f, M: f32) -> bool {
  let along = select(m.x, m.y, m.x < 7.0 || m.x > M - 7.0);
  return min(along, M - along) <= ${NEAR}.0;
}
var<workgroup> ok: u32;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x / ${HYP}u;
  let hyp = wg.x % ${HYP}u;
  let qk = hyp / ${ORIENT * RING_COUNT}u;
  let o = (hyp / ${RING_COUNT}u) % ${ORIENT}u;
  let rg = hyp % ${RING_COUNT}u;
  let base = 5u * (f * ${K}u + qk);
  let qm = quads[base + 4u];
  let at = PL.at[rg];
  let M = f32(at.w);
  if (li == 0u) {
    ok = select(0u, 1u, frames[f].valid != 0u && qm.w == 4.0);
    if (ok == 1u) {
      let h = hypH(array<vec4f, 4>(quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u]), o, M);
      for (var j = 0; j < 9; j++) { H[j] = h[j]; }
      if (h[8] == 0.0) { ok = 0u; }
    }
  }
  let live = workgroupUniformLoad(&ok);
  if (live == 0u) {
    if (li == 0u) { score[f * ${HYP}u + hyp] = -1.0; }
    return;
  }
  var Hl: array<f32, 9>;
  for (var j = 0; j < 9; j++) { Hl[j] = H[j]; }
  // The module size this hypothesis implies, from its own map, picks the level.
  let u = distance(applyH(Hl, vec2f(0.5 * M, 0.5 * M)), applyH(Hl, vec2f(0.5 * M + 1.0, 0.5 * M)));
  let l = levelFor(u);
  // Levels: line (dark), gap (light) and the two cells of the nearest pair, per step.
  var acc = vec4f(0.0);
  for (var s = li; s < at.y; s += 64u) {
    let b = at.x + 4u * s;
    if (!near(plan[b].xy, M)) { continue; }
    acc += vec4f(look(l, f, applyH(Hl, plan[b].xy)), look(l, f, applyH(Hl, plan[b + 1u].xy)),
                 look(l, f, applyH(Hl, plan[b + 2u].xy)) + look(l, f, applyH(Hl, plan[b + 3u].xy)), 1.0);
  }
  sums[li] = acc;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    workgroupBarrier();
  }
  let lev = sums[0] / max(sums[0].w, 1.0);
  let thr = 0.5 * lev.z;
  let con = lev.y - lev.x;
  workgroupBarrier();
  var q = 0.0;
  var nq = 0.0;
  for (var j = li; j < at.z; j += 64u) {
    let pt = plan[at.x + 4u * at.y + j];
    if (!near(pt.xy, M)) { continue; }
    q += pt.z * clamp((thr - look(l, f, applyH(Hl, pt.xy))) / max(con, 1e-3), -0.5, 0.5);
    nq += 1.0;
  }
  sums[li] = vec4f(q, nq, 0.0, 0.0);
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sums[li] += sums[li + st]; }
    workgroupBarrier();
  }
  if (li == 0u) { score[f * ${HYP}u + hyp] = select(-1.0, sums[0].x / max(sums[0].y, 1.0), con >= 0.08); }
}
`;

// F6b: the frame's answer, one workgroup a frame: the best hypothesis if it clears TAU_TRACK. It writes the map
// ai: and the ring F7 registers with (a frame with no answer goes no further), and the result record read back:
// ai: (found, score, ring, orientation), (quad, then F8's word: version, fps, score), H (9 floats), corners (8
// ai: floats), the corners' depths (4), then F8's picture: the version the frame decodes at and whether the held
// ai: configuration stood in (29, 30), padded to 32 floats.
// ai: sel[f] = (ring, found, picture slot, 0): F8 sets y to 1 + the version and z to its picture (wgsl/word.mjs).
export const PICK = DIMS + HOMOGRAPHY + /* wgsl */ `
struct Pick { tau: f32, p0: u32, p1: u32, p2: u32, modules: vec4f }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> quads: array<vec4f>;
@group(0) @binding(2) var<storage, read> score: array<f32>;
@group(0) @binding(3) var<storage, read_write> maps: array<array<f32, 16>>;
@group(0) @binding(4) var<storage, read_write> sel: array<vec4u>;
@group(0) @binding(5) var<storage, read_write> result: array<array<f32, 32>>;
@group(0) @binding(6) var<uniform> P: Pick;
@compute @workgroup_size(1, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u) {
  let f = wg.x;
  var best = -2.0;
  var bh = 0u;
  for (var h = 0u; h < ${HYP}u; h++) {
    let s = score[f * ${HYP}u + h];
    if (s > best) { best = s; bh = h; }
  }
  let found = frames[f].valid != 0u && best >= P.tau;
  let qk = bh / ${ORIENT * RING_COUNT}u;
  let o = (bh / ${RING_COUNT}u) % ${ORIENT}u;
  let rg = bh % ${RING_COUNT}u;
  // ai: Written whole, straight to storage (no private array): a frame not found reads 0 everywhere, and a found one's
  // ai: map has k1 = 0 past H.
  maps[f] = array<f32, 16>();
  result[f] = array<f32, 32>();
  result[f][1] = best;
  if (found) {
    let base = 5u * (f * ${K}u + qk);
    let quad = array<vec4f, 4>(quads[base], quads[base + 1u], quads[base + 2u], quads[base + 3u]);
    let H = hypH(quad, o, P.modules[rg]);
    ${Array.from({ length: 9 }, (_, j) => `maps[f][${j}] = H[${j}]; result[f][${8 + j}] = H[${j}];`).join("\n    ")}
    for (var k = 0u; k < 4u; k++) {
      let p = atSlot(quad, slotOf(o, k));
      result[f][17u + 2u * k] = p.x;
      result[f][18u + 2u * k] = p.y;
      result[f][25u + k] = p.z;   // the corner's depth: 8 read as the gapped core, 6 as the merged square
    }
    result[f][0] = 1.0;
    result[f][2] = f32(rg);
    result[f][3] = f32(o);
    result[f][4] = f32(qk);
  }
  sel[f] = vec4u(rg, select(0u, 1u, found), 0u, 0u);
}
`;
