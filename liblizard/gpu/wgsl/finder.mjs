// F5 to F6: from F4's readings of the bank's peaks to a map. No list is walked on the host and no stage stops early:
// ai: every reading votes, the strongest vote peaks each make a quad, and every (quad, orientation, ring) is scored
// ai: against the track. The frame's answer is an argmax, written straight into the map and the ring F7 registers with.
// ai: The ring is all the finder decides: which picture the ring carries is the format word's to say (F8).
//
// ai: Geometry (src/layout.c, gpu/tables.mjs): a corner mark is the square [0, 12)^2 of the symbol, the border's dark
// ai: line continued at depths 1-2, a light gap at 3-4, a 6 x 6 dark core at [5, 11) and a light edge 1 module wide
// ai: inside it, centre 8 modules in from each edge. Blurred until the gap
// ai: fuses it is a solid 10 x 10 square, centre 6 in. Adjacent centres are M - 16 (or M - 12) modules apart on a
// ai: symbol M modules a side: M = 158, 286 or 542 (the rings of 64, 128 and 256 band cells).
import { DIMS } from "./common.mjs";
import { RINGS as RING_CELLS } from "../../sim/lizard_pick.mjs";

// ai: A mark's centre depth in modules by reading form: the gapped core, the merged square.
export const CENTRE = [8, 6];
export const K = 4;          // vote peaks a frame, so quads a frame
export const CELL = 16;      // accumulator cell, level-0 pixels
export const ORIENT = 8;     // 4 rotations x mirror
// ai: The rings a hypothesis may be (sim/lizard_pick.mjs RINGS, three): an accumulator and a module count each. Four at most:
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
// ai: VOTE's readings a workgroup (2026-10-21; 64 before): a workgroup's voting readings walk their rays one after another
// ai: on one lane, so at a launch of a frame or two the stage took the longest such chain over the frame's 4 workgroups.
export const VOTE_READS = 16;
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
// ai: Whether level-0 point p lies on frame f (a symbol's corner may not: COMPLETE). Reads outside it are the edge's
// ai: clamped pixels, which say nothing of the symbol.
fn onFrame(f: u32, p: vec2f) -> bool { let F = frames[f]; return p.x >= 0.0 && p.y >= 0.0 && p.x < f32(F.w) && p.y < f32(F.h); }
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
// ai: The workgroup's readings that vote, then each one's rays a chunk of steps at a time: lane 0 walks a ray's
// ai: distances by the same additions a lane walked them by (into dk), and the 64 lanes cast that chunk's votes
// ai: together (2026-10-21; a lane walked its own reading's rays, every vote one after another, before). The votes
// ai: are the same: the same distances, cells and weights, summed by atomics that commute.
const CHUNK = 256u;
var<workgroup> vlist: array<u32, ${VOTE_READS}>;
var<workgroup> vcount: atomic<u32>;
var<workgroup> nvote: u32;
var<workgroup> dk: array<f32, CHUNK>;
var<workgroup> nk: u32;
var<workgroup> dnext: f32;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.y;
  if (li == 0u) { atomicStore(&vcount, 0u); }
  workgroupBarrier();
  let i0 = wg.x * ${VOTE_READS}u + li;
  if (li < ${VOTE_READS}u && i0 < P.cap && frames[f].valid == 1u) {
    let o0 = 3u * (f * P.cap + i0);
    // ai: as the lane a reading did: none where r2.w is 0 or p under SMIN (a NaN in either votes)
    if (!(readings[o0 + 2u].w == 0.0 || readings[o0 + 1u].x < ${SMIN})) { vlist[atomicAdd(&vcount, 1u)] = i0; }
  }
  workgroupBarrier();
  if (li == 0u) { nvote = atomicLoad(&vcount); }
  let nv = workgroupUniformLoad(&nvote);
  for (var e = 0u; e < nv; e++) {
    let i = vlist[e];
    let o = 3u * (f * P.cap + i);
    let r0 = readings[o];
    let r2 = readings[o + 2u];
    let p = readings[o + 1u].x;
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
      let S = u32(2 * spread + 1);
      // One step a cell along the ray; at each, a triangle across the band. A cell is hit by more than one step
      // of the same ray when the ray runs near an axis, which is the same small bias for every ray.
      // ai: every lane past its last read of dk, nk and dnext before lane 0 writes them again (workgroupUniformLoad's
      // ai: barrier is before its load, not after: a lane still loading nk would read the next chunk's)
      workgroupBarrier();
      if (li == 0u) { dnext = d0; }
      loop {
        workgroupBarrier();
        if (li == 0u) {
          var d = dnext;
          var k = 0u;
          for (; k < CHUNK && d <= d1; k++) { dk[k] = d; d += ${CELL}.0; }
          nk = k;
          dnext = d;
        }
        let n = workgroupUniformLoad(&nk);
        if (n == 0u) { break; }
        for (var t2 = li; t2 < n * S; t2 += 64u) {
          let q = i32(t2 % S) - spread;
          let at = (r0.xy + dk[t2 / S] * inward) / ${CELL}.0;
          let cell = at + f32(q) * perp;
          let x = i32(floor(cell.x));
          let y = i32(floor(cell.y));
          if (x < 0 || y < 0 || x >= gxf || y >= gyf) { continue; }
          let t = 1.0 - f32(abs(q)) / f32(spread + 1);
          atomicAdd(&acc[((f * ${RING_COUNT}u + m) * P.gy + u32(y)) * P.gx + u32(x)], u32(f32(w) * t));
        }
        workgroupBarrier();
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
// ai: The centre the slots' corners put the symbol at (p02: slots 0 and 2 filled, p13: 1 and 3): with all four the
// ai: crossing of the diagonals; with one diagonal pair and one corner of the other, where the pair's line crosses that
// ai: corner's own diagonal (its reading's ray); with the pair alone its middle. Both crossings are the square's
// ai: centre under any perspective. The pair's middle is so only seen square on: at 45 degrees on the simulator it
// ai: lay 80 to 122 px off the centre, so the third corner, whose ray passes through the centre, fell outside the band
// ai: about it and the quad kept two corners.
fn centreOf(f: u32, pk: u32, p02: bool, p13: bool) -> vec2f {
  let a = slotAt(f, pk, 0u);
  let b = slotAt(f, pk, 2u);
  let e = slotAt(f, pk, 1u);
  let g = slotAt(f, pk, 3u);
  let r = b - a;
  let s = g - e;
  let den = r.x * s.y - r.y * s.x;
  if (p02 && p13 && abs(den) > 1e-3) { return a + ((e.x - a.x) * s.y - (e.y - a.y) * s.x) / den * r; }
  let mid = select(0.5 * (e + g), 0.5 * (a + b), p02);
  // ai: the filled pair's line (A, d) and the other diagonal's one corner (slot q1), where there is one
  let A = select(e, a, p02);
  let d = select(s, r, p02);
  var q1 = 4u;
  for (var q = select(0u, 1u, p02); q < 4u; q += 2u) { if (atomicLoad(&slotK[16u * pk + 4u * q]) != 0u) { q1 = q; } }
  if (q1 == 4u) { return mid; }
  let rq = readings[3u * (f * P.cap + (atomicLoad(&slotK[16u * pk + 4u * q1 + 3u]) & 8191u))];
  let beta = rq.w + PI / 4.0;
  let inward = -vec2f(cos(beta), sin(beta));
  let dn = inward.x * d.y - inward.y * d.x;
  if (abs(dn) < 1e-3 * length(d)) { return mid; }
  let t = ((A.x - rq.x) * d.y - (A.y - rq.y) * d.x) / dn;
  return select(mid, rq.xy + t * inward, t > 0.0);
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
  // ai: a frame the first round found keeps its quads (common.mjs DIMS: valid 2); uniform, before any barrier
  if (frames[f].valid > 1u) { return; }
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

// ai: F5d (2026-10-17): a quad of three corners completed from the ring. A corner mark the crop cuts is no mark the
// ai: network reads, so GATHER keeps three corners and SCORE reads no quad of fewer than four: a symbol whole on the
// ai: frame but for that corner was never registered (on a 2:1 replay of 2026-10-17, 57 of the 150 halves whose code
// ai: crossed the crop's edge were found). The reference finds such a symbol from its straight sides
// ai: (liblizard/src/acquire.c fit_lines). Here, from each of the two corners next to the missing one, the ring (depths
// ai: 1 and 2 dark, 3 modules of light outside it and 2 inside: SPEC 4.3) is read across at STATIONS stations along
// ai: the side by correlation with that profile, on what the frame holds of it, a line fitted through them; the two
// ai: lines meet at the ring's corner, and the missing mark's centre lies (c - 2, c - 2) modules in from it.
// ai: - The side's first direction is one of two, the one whose first pass holds more stations: the three corners'
// ai:   parallelogram, or the known side at the same corner reflected across the mark's diagonal (alpha). Under strong
// ai:   tilt (the simulator's) the parallelogram's far corner was 40 modules off and the reflection a degree to a few;
// ai:   at 2.2x zoom (the replay) the parallelogram was within a degree and the reflection 10 off where the known
// ai:   mark is half cut too.
// ai: - Three passes a side, each about the line the one before fitted: the first 12% of the side (3 modules either
// ai:   side of the guess), 40%, all of it (1.5 modules); a pass that cannot fit keeps the line before it (a side running
// ai:   off the frame toward the missing corner leaves its far stations unread).
// ai: - A station takes the outermost offset whose correlation is a local maximum at NCC_MIN or more: the timing band
// ai:   3.5 modules inside the ring has the same light, dark, light profile where its cell is dark.
// ai: A workgroup a (frame, quad), after the finder's second round's GATHER (REDO), on the frames neither round found:
// ai: a frame holding a quad of four is left as it is, so is a quad of fewer than three corners, and one whose sides
// ai: do not read or whose corner lands far from the other three's frame. What it makes is a quad of four that SCORE
// ai: judges as any other.
export const STATIONS = 32;
const PROFILE = 13;          // ai: samples across the ring, half a module apart, from 3.25 modules outside its centre
const NCC_MIN = 0.6;         // ai: a station holds where the profile's correlation reaches it
const PASSES = [[0.12, 3.0], [0.4, 1.5], [1.0, 1.5]];   // ai: a pass's share of the side and its reach, modules
export const COMPLETE = DIMS + LEVEL_BINDINGS + LEVELS + /* wgsl */ `
struct P5 { cap: u32, gx: u32, gy: u32, p0: u32, modules: vec4f }
@group(0) @binding(7) var<storage, read> readings: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> quads: array<vec4f>;
@group(0) @binding(9) var<uniform> P: P5;
var<workgroup> go: u32;
var<workgroup> ri: array<atomic<u32>, 4>;    // ai: a known corner's reading, its index + 1 (GATHER copies its position)
var<workgroup> G: array<vec4f, 6>;           // ai: (Pa, Pb), (Po, Pm0), (ua, ub, uo, um), (m, c, 0, 0), the first directions
var<workgroup> st: array<vec4f, ${2 * STATIONS}>;   // ai: a station's ring point and weight
var<workgroup> line: array<vec4f, 2>;        // ai: a side's fitted ring line: a point on it, its direction
var<workgroup> kept: array<u32, 2>;           // ai: the stations the side's line holds
// ai: The template across the ring at sample s, half a module apart from 3.25 modules outside its centre: light, dark,
// ai: light (the margin and rim, the ring, the gap).
fn tmpl(s: u32) -> f32 { let r = -3.25 + 0.5 * f32(s); return select(1.0, -1.0, r >= -1.0 && r < 1.0); }
// ai: The profile's correlation with centre C on the samples frame f holds, or -2 where it holds under 8 of them or not
// ai: the ring's inner half and the gap (samples 5 to 12): a ring along the frame's edge can lie half off it, its inner
// ai: edge on it, 1 module from its centre as anywhere.
fn ringNcc(f: u32, l: u32, C: vec2f, n: vec2f, u: f32) -> f32 {
  // ai: one pass of running sums, with no private arrays and no return inside the loop: the Adreno's compiler
  // ai: faulted on the arrays-and-early-return form (2026-10-17)
  var k = 0.0;
  var sv = 0.0;
  var sv2 = 0.0;
  var sumT = 0.0;
  var sumT2 = 0.0;
  var svt = 0.0;
  var bad = false;
  for (var s = 0u; s < ${PROFILE}u; s++) {
    let p = C - (-3.25 + 0.5 * f32(s)) * u * n;
    let inside = onFrame(f, p);
    bad = bad || (!inside && s >= 5u);
    let w = select(0.0, 1.0, inside);
    let v = look(l, f, p);
    let t = tmpl(s);
    k += w; sv += w * v; sv2 += w * v * v; sumT += w * t; sumT2 += w * t * t; svt += w * v * t;
  }
  if (bad || k < 8.0) { return -2.0; }
  let cov = svt - sv * sumT / k;
  let vv = sv2 - sv * sv / k;
  let tv = sumT2 - sumT * sumT / k;
  return cov / sqrt(max(vv * tv, 1e-12));
}
// ai: The ring about X (its expected centre) along the outward normal n, offsets +-reach modules a quarter module
// ai: apart: the outermost local maximum of the correlation at NCC_MIN or more, and its correlation (0: none).
fn ringAt(f: u32, X: vec2f, n: vec2f, u: f32, reach: f32) -> vec3f {
  let l = levelFor(u);
  let steps = i32(round(reach * 4.0));
  var a = ringNcc(f, l, X - (0.25 * f32(steps + 1)) * u * n, n, u);
  var b = ringNcc(f, l, X - (0.25 * f32(steps)) * u * n, n, u);
  var best = -2.0;
  var at = 0.0;
  for (var i = -steps; i <= steps; i++) {
    let c = ringNcc(f, l, X + (0.25 * f32(i + 1)) * u * n, n, u);
    // ai: b is offset i; a the one inside it, c the one outside; the last that holds is the outermost
    if (b >= ${NCC_MIN} && b >= a && b >= c) {
      var dt = 0.0;
      let e = a + c - 2.0 * b;
      if (a > -2.0 && c > -2.0 && e < 0.0) { dt = clamp(0.5 * (a - c) / e, -0.5, 0.5); }
      best = b;
      at = 0.25 * (f32(i) + dt);
    }
    a = b; b = c;
  }
  return vec3f(X + at * u * n, select(0.0, best, best >= ${NCC_MIN}));
}
// ai: A weighted line through a side's stations by total least squares, then again without those over a module off
// ai: it; written to line[side] (and its stations to kept[side]) where it holds need stations or more and spans a
// ai: quarter of the pass's length, else line[side] kept as it was.
fn fitSide(side: u32, u: f32, span: f32, need: u32) -> bool {
  var ok = array<u32, ${STATIONS}>();
  for (var j = 0u; j < ${STATIONS}u; j++) { ok[j] = select(0u, 1u, st[side * ${STATIONS}u + j].z > 0.0); }
  var o = vec2f(0.0);
  var d = vec2f(1.0, 0.0);
  for (var ps = 0u; ps < 2u; ps++) {
    var sw = 0.0;
    var m = vec2f(0.0);
    for (var j = 0u; j < ${STATIONS}u; j++) { let p = st[side * ${STATIONS}u + j]; if (ok[j] != 0u) { sw += p.z; m += p.z * p.xy; } }
    if (sw <= 0.0) { return false; }
    m /= sw;
    var cxx = 0.0; var cxy = 0.0; var cyy = 0.0;
    for (var j = 0u; j < ${STATIONS}u; j++) {
      let p = st[side * ${STATIONS}u + j];
      if (ok[j] == 0u) { continue; }
      let e = p.xy - m;
      cxx += p.z * e.x * e.x; cxy += p.z * e.x * e.y; cyy += p.z * e.y * e.y;
    }
    let th = 0.5 * atan2(2.0 * cxy, cxx - cyy);
    o = m;
    d = vec2f(cos(th), sin(th));
    if (ps == 0u) {
      for (var j = 0u; j < ${STATIONS}u; j++) {
        let e = st[side * ${STATIONS}u + j].xy - o;
        if (ok[j] != 0u && abs(e.x * d.y - e.y * d.x) > u) { ok[j] = 0u; }
      }
    }
  }
  var n = 0u;
  var tmin = 1e30;
  var tmax = -1e30;
  for (var j = 0u; j < ${STATIONS}u; j++) {
    if (ok[j] == 0u) { continue; }
    n++;
    let t = dot(st[side * ${STATIONS}u + j].xy - o, d);
    tmin = min(tmin, t); tmax = max(tmax, t);
  }
  if (n < max(need, 6u) || tmax - tmin < 0.25 * span) { return false; }
  line[side] = vec4f(o, d);
  kept[side] = n;
  return true;
}
@compute @workgroup_size(${2 * STATIONS}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x / ${K}u;
  let base = 5u * wg.x;
  if (li == 0u) {
    // ai: a frame the first round missed (valid 1) and of which neither round holds a quad of four (GATHER's 4; a quad
    // ai: completed here is written as 5, which SCORE reads as four and a sibling workgroup's test here does not)
    var four = false;
    for (var q = 0u; q < ${K}u; q++) { four = four || quads[5u * (f * ${K}u + q) + 4u].w == 4.0; }
    go = select(0u, 1u, frames[f].valid == 1u && quads[base + 4u].w == 3.0 && !four);
    for (var q = 0u; q < 4u; q++) { atomicStore(&ri[q], 0u); }
  }
  if (workgroupUniformLoad(&go) == 0u) { return; }
  // ai: each known corner's reading, by its position, for its module size and the mark's diagonal
  for (var i = li; i < P.cap; i += ${2 * STATIONS}u) {
    let r = readings[3u * (f * P.cap + i)];
    for (var q = 0u; q < 4u; q++) {
      let c = quads[base + q];
      if (c.w > 0.0 && r.x == c.x && r.y == c.y && r.z > 0.0) { atomicMax(&ri[q], i + 1u); }
    }
  }
  workgroupBarrier();
  if (li == 0u) {
    var m = 0u;
    for (var q = 0u; q < 4u; q++) { if (quads[base + q].w == 0.0) { m = q; } }
    let a = (m + 1u) % 4u;
    let b = (m + 3u) % 4u;
    let o = (m + 2u) % 4u;
    let ia = atomicLoad(&ri[a]);
    let ib = atomicLoad(&ri[b]);
    let io = atomicLoad(&ri[o]);
    if (ia == 0u || ib == 0u || io == 0u) { go = 0u; }
    else {
      let ra = readings[3u * (f * P.cap + ia - 1u)];
      let rb = readings[3u * (f * P.cap + ib - 1u)];
      let ro = readings[3u * (f * P.cap + io - 1u)];
      let Pa = ra.xy;
      let Pb = rb.xy;
      let Po = ro.xy;
      G[0] = vec4f(Pa, Pb);
      G[1] = vec4f(Po, Pa + Pb - Po);
      G[2] = vec4f(ra.z, rb.z, ro.z, clamp(ra.z + rb.z - ro.z, 0.5 * min(ra.z, rb.z), 2.0 * max(ra.z, rb.z)));
      G[3] = vec4f(f32(m), quads[base + a].z, 0.0, 0.0);
      // ai: each side's two first directions: the known side at its corner (toward o) reflected across the mark's
      // ai: diagonal, and toward the parallelogram's corner
      for (var sd = 0u; sd < 2u; sd++) {
        let P0 = select(Pa, Pb, sd == 1u);
        let al = select(ra.w, rb.w, sd == 1u);
        let k = normalize(Po - P0);
        let e = -dir(al + 0.25 * 3.14159265);
        G[4u + sd] = vec4f(2.0 * dot(k, e) * e - k, normalize(Pa + Pb - Po - P0));
      }
      kept[0] = 0u;
      kept[1] = 0u;
    }
  }
  if (workgroupUniformLoad(&go) == 0u) { return; }
  let side = li / ${STATIONS}u;
  let j = li % ${STATIONS}u;
  let P0 = select(G[0].xy, G[0].zw, side == 1u);
  let Pm0 = G[1].zw;
  let u0 = select(G[2].x, G[2].y, side == 1u);
  let um = G[2].w;
  let c = G[3].y;
  let Q = 0.25 * (G[0].xy + G[0].zw + G[1].xy + Pm0);
  let L = distance(P0, Pm0);
  // ai: the ring runs from 12 modules past a corner (the break beside the mark at 11) to 12 before the next
  let t0 = (13.0 - c) * u0;
  let t1 = L - (13.0 - c) * um;
  let span = t1 > t0 + 8.0 * u0;
  // ai: the outward normal's side, from the first guess (both guesses lie within degrees of each other)
  var n0 = vec2f(-G[4u + side].y, G[4u + side].x);
  if (dot(n0, P0 - Q) < 0.0) { n0 = -n0; }
  // ai: the first pass from each guess; the side keeps the fit that holds more stations
  for (var g = 0u; g < 2u; g++) {
    let dd = select(G[4u + side].xy, G[4u + side].zw, g == 1u);
    var n = vec2f(-dd.y, dd.x);
    if (dot(n, n0) < 0.0) { n = -n; }
    var hit = vec3f(0.0);
    if (span) {
      let t = mix(t0, t1, (f32(j) + 0.5) / ${STATIONS}.0 * ${PASSES[0][0]});
      let uu = mix(u0, um, t / L);
      hit = ringAt(f, P0 + t * dd + (c - 2.0) * uu * n, n, uu, ${PASSES[0][1]});
    }
    st[li] = vec4f(hit.xy, hit.z, 0.0);
    workgroupBarrier();
    if (j == 0u) { fitSide(side, 0.5 * (u0 + um), (t1 - t0) * ${PASSES[0][0]}, kept[side] + 1u); }
    workgroupBarrier();
  }
  if (workgroupUniformLoad(&kept[0]) == 0u || workgroupUniformLoad(&kept[1]) == 0u) { return; }
  // ai: the passes after, each about the line the one before kept: the station's foot on it, the normal its own
  ${PASSES.slice(1).map(([frac, reach]) => `{
    var hit = vec3f(0.0);
    if (span) {
      let t = mix(t0, t1, (f32(j) + 0.5) / ${STATIONS}.0 * ${frac});
      let uu = mix(u0, um, t / L);
      let ln = line[side];
      var nn = vec2f(-ln.w, ln.z);
      if (dot(nn, n0) < 0.0) { nn = -nn; }
      let B = P0 + t * normalize(Pm0 - P0);
      hit = ringAt(f, ln.xy + dot(B - ln.xy, ln.zw) * ln.zw, nn, uu, ${reach});
    }
    st[li] = vec4f(hit.xy, hit.z, 0.0);
    workgroupBarrier();
    if (j == 0u) { fitSide(side, 0.5 * (u0 + um), (t1 - t0) * ${frac}, 6u); }
    workgroupBarrier();
  }`).join("\n  ")}
  if (li == 0u) {
    // ai: the two ring lines' crossing, the ring's corner (module (2, 2) from the missing corner)
    let A = line[0];
    let Bl = line[1];
    let den = A.z * Bl.w - A.w * Bl.z;
    if (abs(den) < 0.2) { return; }
    let s = ((Bl.x - A.x) * Bl.w - (Bl.y - A.y) * Bl.z) / den;
    let R = A.xy + s * A.zw;
    var ta = A.zw;
    if (dot(ta, G[0].xy - R) < 0.0) { ta = -ta; }
    var tb = Bl.zw;
    if (dot(tb, G[0].zw - R) < 0.0) { tb = -tb; }
    let Pm = R + (c - 2.0) * um * (ta + tb);
    // ai: a crossing far from where the other three corners put it is not this symbol's
    if (distance(Pm, Pm0) > 0.25 * distance(G[0].xy, G[0].zw)) { return; }
    let m = u32(G[3].x);
    quads[base + m] = vec4f(Pm, c, 1.0);
    let qm = quads[base + 4u];
    let p0 = quads[base].xy;
    let p1 = quads[base + 1u].xy;
    let p2 = quads[base + 2u].xy;
    let p3 = quads[base + 3u].xy;
    let uq = 0.25 * (distance(p0, p1) + distance(p1, p2) + distance(p2, p3) + distance(p3, p0)) / (P.modules[u32(qm.x)] - 2.0 * quads[base].z);
    quads[base + 4u] = vec4f(qm.x, uq, qm.z, 5.0);
  }
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
  // ai: a frame the first round found keeps its scores (valid 2); uniform, before any barrier
  if (frames[f].valid > 1u) { return; }
  let hyp = wg.x % ${HYP}u;
  let qk = hyp / ${ORIENT * RING_COUNT}u;
  let o = (hyp / ${RING_COUNT}u) % ${ORIENT}u;
  let rg = hyp % ${RING_COUNT}u;
  let base = 5u * (f * ${K}u + qk);
  let qm = quads[base + 4u];
  let at = PL.at[rg];
  let M = f32(at.w);
  if (li == 0u) {
    ok = select(0u, 1u, frames[f].valid != 0u && (qm.w == 4.0 || qm.w == 5.0));
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
    // ai: a step whose points the frame does not hold (a corner past the crop, COMPLETE's) reads nothing
    let q0 = applyH(Hl, plan[b].xy);
    let q1 = applyH(Hl, plan[b + 1u].xy);
    let q2 = applyH(Hl, plan[b + 2u].xy);
    let q3 = applyH(Hl, plan[b + 3u].xy);
    if (!(onFrame(f, q0) && onFrame(f, q1) && onFrame(f, q2) && onFrame(f, q3))) { continue; }
    acc += vec4f(look(l, f, q0), look(l, f, q1), look(l, f, q2) + look(l, f, q3), 1.0);
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
    let qp = applyH(Hl, pt.xy);
    if (!onFrame(f, qp)) { continue; }
    q += pt.z * clamp((thr - look(l, f, qp)) / max(con, 1e-3), -0.5, 0.5);
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
  if (frames[f].valid > 1u) { return; }   // ai: a frame the first round found keeps what it picked (valid 2)
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

// ai: The finder's second round (2026-10-21): the first runs the proposer on levels 2 to 4 (bank_fcn2.mjs COARSE_FROM);
// ai: a frame it did not find is found again with level 1's peaks beside those, by the same stages on a second
// ai: frame record (frames2), which says which frames: valid 1 for a frame of the batch PICK did not find, 2 for one
// ai: it found (left as it is: common.mjs DIMS), 0 for an empty slot. A frame to run again gets back what the batch's
// ai: clears gave its first round: its accumulator (VOTE adds into it), readings (a list slot the cascade's large net
// ai: wrote in the first round may hold none in the second), centre peaks and quads. A workgroup a frame.
export const REDO = DIMS + /* wgsl */ `
struct P5 { cap: u32, gx: u32, gy: u32, p0: u32, modules: vec4f }
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
@group(0) @binding(1) var<storage, read> sel: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> frames2: array<Frame>;
@group(0) @binding(3) var<storage, read_write> acc: array<u32>;
@group(0) @binding(4) var<storage, read_write> readings: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> cand: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> quads: array<vec4f>;
@group(0) @binding(7) var<uniform> P: P5;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  let F = frames[f];
  let again = F.valid == 1u && sel[f].y == 0u;
  if (li == 0u) { frames2[f] = Frame(F.w, F.h, F.size, select(select(0u, 2u, F.valid != 0u), 1u, again)); }
  if (!again) { return; }
  let na = ${RING_COUNT}u * P.gx * P.gy;
  for (var i = li; i < na; i += 256u) { acc[f * na + i] = 0u; }
  let nr = 3u * P.cap;
  for (var i = li; i < nr; i += 256u) { readings[f * nr + i] = vec4f(0.0); }
  if (li < ${K}u) { cand[f * ${K}u + li] = vec4f(0.0); }
  if (li < ${5 * K}u) { quads[f * ${5 * K}u + li] = vec4f(0.0); }
}
`;
