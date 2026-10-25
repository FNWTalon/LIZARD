// ai: F8: the format word, read through the final map (homography, lens and the border's Coons patch), one workgroup
// ai: a frame, after F7 and before the back half (gpu/decoder.mjs). The reads are the format's (gpu/tables.mjs word
// ai: plan, the reference codec's own): the solid line and the gap every 16 modules give the contrast; each word cell
// ai: is read at its planned centre against the mean of the Manchester pairs of the track near it (a pair is one dark
// ai: and one light cell, so its mean is the local grey). One code a ring over every word cell, W = 16, 32, 64, 128 a
// ai: side (src/fmt.h; 48 for the 96 ring until 2026-10-10), so 4 W soft bits, one cell each. Then the word is found
// ai: by ordered statistics decoding (2026-10-25, research STATUS "The ring's word"): the cells sorted by |q|, the
// ai: code's K rows (gpu/wordcode.mjs) reduced over them to an information set, the word agreeing with the hard bits
// ai: there, and every word up to ORDER of those bits away scored against the soft bits, the best kept. Until then it
// ai: was maximum likelihood over all VERSION_MAX x 256 words (a Walsh-Hadamard transform over the rate from
// ai: 2026-10-21), which OSD matched on every lab frame and replay it was held to; the search's work doubled with each
// ai: information bit of the word, OSD's grows as K^ORDER, which is what a longer word needs. Where the reference
// ai: decodes hard bytes with erasures (src/fmt.c), this decodes soft. A word is taken if the contrast reads (CON_MIN, as
// ai: src/acquire.c) and it passes the gate (2026-10-25, gpu/wordcode.mjs gateTable and gateChoices): its
// ai: disagreements with the hard bits on the cells at |q| >= TAU of the sides counted, within E[m] for the m that
// ai: count, the sides counted three ways (all, all but the least sum of |q|, all but the two least) and any passing.
// ai: The search leaves out (keep) a side under SIDE_KEEP of the best side's sum, at most the two least. Until then the
// ai: gate was DMAX[W] over every bit, which a glared side (its signs a coin at |q| about 0.33) spent half of, so two
// ai: glared sides were never read (research STATUS "The ring's word").
// ai: The frame then decodes at the word's version, or, with no word taken, at the batch's held version (the last word
// ai: the host read, 0 for none, since 2026-09-27: the ring bootstraps the word, and a frame whose word does not decode
// ai: leaves the held one as it is); with neither it decodes nothing. So sel[f].y = 1 + that version (the back
// ai: half's gate reads it as the frame's blocks) and sel[f].z its picture slot (gpu/tables.mjs PICTURES), or sel[f].y
// ai: = 0 and the gate drops the frame. result[f]: [5] the word's version (0 for none), [6] fps, [7] the best word's
// ai: soft score; [29] the version the frame decodes at (0 for none), [30] 1 where the held version stood in.
// ai: No private array: the gate reads each cell's bit of the best word off the code's rows in the uniform, a lane a
// ai: cell (codeBit), and the lattice is read from a uniform (the S26's Adreno returned zeros from a private array
// ai: indexed at run time).
import { DIMS, MAP, SLOTS } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { LATTICE_MAX } from "./back_transform.mjs";
import { RING_COUNT } from "./finder.mjs";
import { wordCode, VERSION_MAX, gateTable, TAU, SIDE_KEEP } from "../wordcode.mjs";

export const CON_MIN = 0.08;
// ai: The most hard bits the best word may disagree with, by word cells a side (so by ring): 3/16 of the bits, or
// ai: fewer where the union bound on a random cell set taking a word (every one of the ring's 32,768 words within DMAX
// ai: of a random hard word) would pass 1e-6 a frame. 64 bits: 6 (1.5e-7; 7 gives 1.3e-6); 128 bits: 24, 3/16
// ai: (7.5e-9); 192 bits: 36, 3/16 (8.6e-15); 256 bits: 48, 3/16 (1.1e-20); 512 bits (the 256 ring, 2026-10-10): 96, 3/16. Until the evening of 2026-09-27 the rings
// ai: were 32, 48, 64 and 96 bits took 16 (3.4e-7); until that morning each picture size scored only its own versions
// ai: (1,024 to 14,592 words): 2/16 at n = 256 (2.9e-7), 3/16 elsewhere. scripts/gpu/word_check.mjs prints the bounds and a
// ai: garbage trial. F8 no longer reads it (the gate above, 2026-10-25); it stays for the check's comparison.
export const DMAX = { 16: 6, 32: 24, 48: 36, 64: 48, 128: 96 };
// ai: WMAX: word cells a side at most (the 256 ring; the 128's 64 until 2026-10-10), so a word's 4 WMAX bits fill VECS
// ai: vec4u (a row of the code); ROWS: BASE and the 16 rows a ring; MIDS_MAX: track pairs a frame (the 256 ring's 512), each
// ai: invocation taking every 256th.
export const WMAX = 128, ROWS = 17, MIDS_MAX = 512;
export const VECS = (4 * WMAX) / 128;
// ai: vec4u the pictures' last versions take, four a vec4u.
const TOPS = Math.ceil(SLOTS / 4);
// ai: OSD's shape: K rows of information (the code's 16 under BASE), ORDER the most of them a candidate flips (unrank
// ai: is written for 3), NCAND the candidates (1 + 16 + 120 + 560), an invocation taking every 256th; NW the u32 of a
// ai: word at most. K NW is the workgroup, an invocation a (row, u32) of the reduction.
export const K = ROWS - 1, ORDER = 3, NW = (4 * WMAX) / 32;
const choose = (n, k) => { let c = 1; for (let i = 0; i < k; i++) c = (c * (n - i)) / (i + 1); return c; };
export const NCAND = Array.from({ length: ORDER + 1 }, (_, j) => choose(K, j)).reduce((a, b) => a + b, 0);
if (K * NW !== 256 || ORDER !== 3) throw new Error(`F8's OSD is laid out for 16 rows of 16 u32 at order 3, not ${K} of ${NW} at ${ORDER}`);
// ai: The gate's table a ring: E[m] for m = 0 to 4 W as u16, two a u32 (m even low), 0xffff where no count passes;
// ai: GATE u32 a ring holds 4 WMAX + 1 of them, padded to a vec4u.
export const GATE = 4 * Math.ceil((4 * WMAX + 2) / 8);
// ai: The Word uniform's layout in vec4u: code (every ring's ROWS rows, a row VECS vec4u: ring r's row k at (r ROWS +
// ai: k) VECS), then range (W, DMAX, 0, 0) a ring, then plan a ring, then tops (the last version of pictures 0 to
// ai: SLOTS - 1), then gate (GATE u32 a ring).
export const LAYOUT = (() => {
  const range = RING_COUNT * ROWS * VECS, plan = range + RING_COUNT, tops = plan + RING_COUNT, gate = tops + TOPS;
  return { code: 0, range, plan, tops, gate, size: gate + (RING_COUNT * GATE) / 4 };
})();

// ai: The Word uniform for the rings, from gpu/tables.mjs formatTables() rings (word.cells), the plan wordPoints lays
// ai: out, and the pictures (their last versions, which pick a version's slot), as u32 in LAYOUT's order.
export function wordUniform(rings, plans, pictures) {
  if (rings.length !== RING_COUNT) throw new Error(`${rings.length} rings: F8 is built for ${RING_COUNT}`);
  if (pictures.length !== SLOTS || pictures.at(-1).hi !== VERSION_MAX) throw new Error(`F8 names a picture of ${SLOTS} by its last version, the last VERSION_MAX`);
  const u = new Uint32Array(4 * LAYOUT.size);
  rings.forEach(({ word }, r) => {
    if (!(word.cells in DMAX)) throw new Error(`ring ${r}: ${word.cells} word cells a side, F8 has a DMAX for ${Object.keys(DMAX).join(", ")}`);
    if (word.cells > WMAX) throw new Error(`ring ${r}: ${word.cells} word cells a side, F8 holds ${WMAX}`);
    const { base, rows } = wordCode(word.cells);
    [base, ...rows].forEach((v, k) => u.set(v, 4 * VECS * (r * ROWS + k)));
    gateTable(word.cells).forEach((e, m) => { u[4 * LAYOUT.gate + GATE * r + (m >> 1)] |= (e < 0 ? 0xffff : e) << (16 * (m & 1)); });
    u.set([word.cells, DMAX[word.cells], 0, 0], 4 * (LAYOUT.range + r));
    u.set(plans[r], 4 * (LAYOUT.plan + r));
  });
  u.set(pictures.map((p) => p.hi), 4 * LAYOUT.tops);
  return u;
}

// ai: The plan's points as the kernel reads them, every ring after the one before: per ring the references (2 a
// ai: step, the line then the gap), the track pairs (2 a pair), then the word cells in bit order, as vec4f (x, y,
// ai: lo + 4096 hi, 0) in module coordinates, lo and hi the pair range a cell's grey is the mean of. plans[r] =
// ai: (first point, reference steps, pairs, word cells). Throws where the plan leaves the kernel's shape.
export function wordPoints(rings) {
  const pts = [], plans = [];
  for (const { cells: ring, word: w } of rings) {
    if (w.reps !== 1 || w.words !== 4 * w.cells || w.cells > WMAX) throw new Error(`ring ${ring}: ${w.words} word points of ${w.cells} cells a side, reps ${w.reps}: F8 reads one point a cell, W <= ${WMAX}`);
    if (w.mids > MIDS_MAX) throw new Error(`ring ${ring}: ${w.mids} track pairs, F8 holds ${MIDS_MAX}`);
    plans.push([pts.length / 4, w.refSteps, w.mids, w.words]);
    const np = 2 * w.refSteps + 2 * w.mids;
    for (let k = 0; k < np; k++) pts.push(w.pts[2 * k], w.pts[2 * k + 1], 0, 0);
    for (let k = 0; k < w.words; k++) {
      const [lo, hi, bit] = w.widx.subarray(3 * k, 3 * k + 3);
      if (bit !== k || lo < 0 || hi < lo || hi >= w.mids) throw new Error(`ring ${ring}: word point ${k} is bit ${bit}, pairs ${lo} to ${hi}`);
      pts.push(w.pts[2 * (np + k)], w.pts[2 * (np + k) + 1], lo + 4096 * hi, 0);
    }
  }
  return { pts: new Float32Array(pts), plans };
}

// ai: The LT uniform: at[r] = (first coordinate, nodes a side, 0, 0), then every ring's lattice coordinates,
// ai: LATTICE_MAX a ring, four a vec4f (the fused pass 1's PIC holds them the same way).
export function wordLattice(rings) {
  const ab = new ArrayBuffer(16 * RING_COUNT + 4 * RING_COUNT * LATTICE_MAX), u = new Uint32Array(ab), f = new Float32Array(ab);
  rings.forEach(({ cells, nodeX }, r) => {
    if (nodeX.length > LATTICE_MAX || nodeX.length < 2) throw new Error(`ring ${cells}: ${nodeX.length} lattice nodes a side, LT holds 2 to ${LATTICE_MAX}`);
    u.set([LATTICE_MAX * r, nodeX.length, 0, 0], 4 * r);
    f.set(nodeX, 4 * RING_COUNT + LATTICE_MAX * r);
  });
  return new Uint8Array(ab);
}

// ai: The held uniform, a lane's own, written with each batch: (the held version, 0 for none, 0, 0).
export const heldUniform = (held) => new Uint32Array([held, 0, 0, 0]);

const WORDS = ["x", "y", "z", "w"];

// ai: The search, after q and keep: OSD over the ring's K rows, each invocation's best (score, candidate) into
// ai: red[li], candidates numbered as unrank numbers them, ties to the lower.
const SEARCH = /* wgsl */ `
  let at0 = rg * ${ROWS}u;
  if (li == 0u) { nbw = nb; }
  let nbu = workgroupUniformLoad(&nbw);
  let nw = nbu / 32u;
  // ai: the search's soft bits (a side keep leaves out at 0), and every cell's |qs| and index for the sort
  for (var bit = li; bit < nbu; bit += 256u) {
    let x = select(0.0, q[bit], ((keep >> (bit / vr.x)) & 1u) != 0u);
    qs[bit] = x;
    key[bit] = abs(x);
    idx[bit] = bit;
  }
  // ai: the K rows as bitsets, an invocation a (row, u32): rows at0 + 1 .. (the version's bits, then the rate's)
  gb[li] = select(0u, W.code[${VECS}u * (at0 + 1u + (li >> 4u)) + ((li & 15u) >> 2u)][li & 3u], (li & 15u) < nw);
  workgroupBarrier();
  // ai: The cells in order of confidence, |qs| falling and ties to the lower cell: a bitonic sort (nbu is a power of
  // ai: two, 64 to 512), an invocation a pair.
  for (var k = 2u; k <= nbu; k <<= 1u) {
    for (var j = k >> 1u; j > 0u; j >>= 1u) {
      if (li < nbu / 2u) {
        let a = 2u * j * (li / j) + (li % j);
        let b = a + j;
        let ka = key[a];
        let kb = key[b];
        let ia = idx[a];
        let ib = idx[b];
        if ((ka > kb || (ka == kb && ia < ib)) != ((a & k) == 0u)) { key[a] = kb; key[b] = ka; idx[a] = ib; idx[b] = ia; }
      }
      workgroupBarrier();
    }
  }
  // ai: Gauss-Jordan over the cells in that order, a pivot a step: invocation 0 finds the next cell some row not yet a
  // ai: pivot has set (the lowest such row its pivot), then every invocation clears that cell from the other rows.
  var tcur = 0u;
  var used = 0u;
  for (var s = 0u; s < ${K}u; s++) {
    if (li == 0u) {
      var r = ${K}u;
      var mask = 0u;
      var p = 0u;
      loop {
        if (tcur >= nbu) { break; }
        p = idx[tcur];
        tcur += 1u;
        var m = 0u;
        for (var k = 0u; k < ${K}u; k++) { m |= ((gb[k * ${NW}u + (p >> 5u)] >> (p & 31u)) & 1u) << k; }
        let spare = m & ~used;
        if (spare != 0u) { r = firstTrailingBit(spare); mask = m & ~(1u << r); break; }
      }
      if (r < ${K}u) { used |= 1u << r; piv[r] = p; }
      stepw = vec4u(r, mask, 0u, 0u);
    }
    let st = workgroupUniformLoad(&stepw);
    if (st.x < ${K}u && ((st.y >> (li >> 4u)) & 1u) != 0u && (li & 15u) < nw) { gb[li] ^= gb[st.x * ${NW}u + (li & 15u)]; }
    workgroupBarrier();
  }
  // ai: Order 0: BASE with the pivot rows whose cell's hard bit differs from BASE's, so it agrees with q's signs on
  // ai: every pivot.
  if (li == 0u) {
    var uu = 0u;
    for (var r = 0u; r < ${K}u; r++) {
      if (((used >> r) & 1u) == 0u) { continue; }
      let p = piv[r];
      let b0 = (W.code[${VECS}u * at0 + (p >> 7u)][(p >> 5u) & 3u] >> (p & 31u)) & 1u;
      uu |= (select(0u, 1u, qs[p] > 0.0) ^ b0) << r;
    }
    stepw = vec4u(uu, 0u, 0u, 0u);
  }
  let u0 = workgroupUniformLoad(&stepw).x;
  if (li < nw) {
    var cw = W.code[${VECS}u * at0 + (li >> 2u)][li & 3u];
    for (var r = 0u; r < ${K}u; r++) { if (((u0 >> r) & 1u) != 0u) { cw ^= gb[r * ${NW}u + li]; } }
    c0w[li] = cw;
  }
  workgroupBarrier();
  // ai: Every candidate, an invocation every 256th: order 0 with up to ORDER pivot rows flipped, its version byte in
  // ai: 1 .. VERSION_MAX, its score sum qs_i (2 c_i - 1).
  var best = -1e9;
  var bc = 0u;
  for (var c = li; c < ${NCAND}u; c += 256u) {
    let sub = unrank(c);
    let v = byteOf(sub, vr.x);
    if (v < 1u || v > ${VERSION_MAX}u) { continue; }
    var sc = 0.0;
    for (var w = 0u; w < nw; w++) {
      let cw = c0w[w] ^ rowWord(sub.y, w) ^ rowWord(sub.z, w) ^ rowWord(sub.w, w);
      for (var b = 0u; b < 32u; b++) { let x = qs[32u * w + b]; sc += select(-x, x, ((cw >> b) & 1u) != 0u); }
    }
    if (sc > best) { best = sc; bc = c; }
  }
  red[li] = vec4f(best, f32(bc), 0.0, 0.0);`;

export const WORD = DIMS + MAP + /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }
// ai: code: per ring, BASE then the 16 rows (version bits, then fps bits), VECS vec4u each; range[ring] = (word cells
// ai: a side W, DMAX, 0, 0); plan[ring] = (first point, reference steps, track pairs, word points); tops: the last
// ai: version of pictures 0 to SLOTS - 1, four a vec4u; gate: the gate's E[m], GATE u32 a ring.
struct Word { code: array<vec4u, ${LAYOUT.range}>, range: array<vec4u, ${RING_COUNT}>, plan: array<vec4u, ${RING_COUNT}>, tops: array<vec4u, ${TOPS}>, gate: array<vec4u, ${(RING_COUNT * GATE) / 4}> }
struct Lattice { at: array<vec4u, ${RING_COUNT}>, lat: array<vec4f, ${(RING_COUNT * LATTICE_MAX) / 4}> }
struct Held { version: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var img: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> lv1: array<f32>;
@group(0) @binding(2) var<storage, read> lv2: array<f32>;
@group(0) @binding(3) var<storage, read> lv3: array<f32>;
@group(0) @binding(4) var<storage, read> lv4: array<f32>;
@group(0) @binding(5) var<storage, read> frames: array<Frame>;
@group(0) @binding(6) var<uniform> LD: LevelDims;
@group(0) @binding(7) var<storage, read> maps: array<array<f32, 16>>;
@group(0) @binding(8) var<storage, read_write> sel: array<vec4u>;
@group(0) @binding(9) var<storage, read> resid: array<vec4f>;
@group(0) @binding(10) var<storage, read> pts: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> result: array<array<f32, 32>>;
@group(0) @binding(12) var<uniform> W: Word;
@group(0) @binding(13) var<uniform> LT: Lattice;
@group(0) @binding(14) var<uniform> HB: Held;
fn latAt(i: u32) -> f32 { return LT.lat[i >> 2u][i & 3u]; }
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
fn look(l: u32, f: u32, p: vec2f) -> f32 {
  let q = p / f32(1u << l) - vec2f(0.5);
  let q0 = floor(q);
  let a = q - q0;
  let i = vec2i(q0);
  return mix(mix(pix(l, f, i.x, i.y), pix(l, f, i.x + 1, i.y), a.x), mix(pix(l, f, i.x, i.y + 1), pix(l, f, i.x + 1, i.y + 1), a.x), a.y);
}
` + COONS + /* wgsl */ `
var<workgroup> red: array<vec4f, 256>;
var<workgroup> mids: array<f32, ${MIDS_MAX}>;
var<workgroup> q: array<f32, ${4 * WMAX}>;
var<workgroup> go: u32;
// ai: the best word (score, (v - 1) 256 + fps, 1 where a candidate had a version in range), lane 0's to every lane
var<workgroup> win: vec4f;
// ai: OSD's: the search's soft bits, the sort's keys and cells, the K rows as the reduction leaves them (NW u32 a
// ai: row), the order-0 word, each row's pivot cell, the bit count made uniform, and invocation 0's word to the rest
var<workgroup> qs: array<f32, ${4 * WMAX}>;
var<workgroup> key: array<f32, ${4 * WMAX}>;
var<workgroup> idx: array<u32, ${4 * WMAX}>;
var<workgroup> gb: array<u32, ${K * NW}>;
var<workgroup> c0w: array<u32, ${NW}>;
var<workgroup> piv: array<u32, ${K}>;
var<workgroup> nbw: u32;
var<workgroup> stepw: vec4u;
// ai: Row r's u32 w as the reduction left it, 0 for r = K (no row).
fn rowWord(r: u32, w: u32) -> u32 { return select(0u, gb[min(r, ${K - 1}u) * ${NW}u + w], r < ${K}u); }
// ai: A candidate's byte at cells at .. at + 7, most significant first (one u32: at is a multiple of 8, W of 16).
fn byteOf(sub: vec4u, at: u32) -> u32 {
  let w = at >> 5u;
  return reverseBits(extractBits(c0w[w] ^ rowWord(sub.y, w) ^ rowWord(sub.z, w) ^ rowWord(sub.w, w), at & 31u, 8u)) >> 24u;
}
// ai: Candidate c's rows as (how many, then the rows flipped, K where none): 0 none, then the K singles, the pairs and
// ai: the triples in lexicographic order.
fn unrank(c0: u32) -> vec4u {
  var c = c0;
  if (c == 0u) { return vec4u(0u, ${K}u, ${K}u, ${K}u); }
  c -= 1u;
  if (c < ${K}u) { return vec4u(1u, c, ${K}u, ${K}u); }
  c -= ${K}u;
  if (c < ${(K * (K - 1)) / 2}u) {
    for (var a = 0u; a + 1u < ${K}u; a++) { let n = ${K - 1}u - a; if (c < n) { return vec4u(2u, a, a + 1u + c, ${K}u); } c -= n; }
  }
  c -= ${(K * (K - 1)) / 2}u;
  for (var a = 0u; a + 2u < ${K}u; a++) {
    for (var b = a + 1u; b + 1u < ${K}u; b++) { let n = ${K - 1}u - b; if (c < n) { return vec4u(3u, a, b, b + 1u + c); } c -= n; }
  }
  return vec4u(4u, ${K}u, ${K}u, ${K}u);
}
// ai: Bit i of the word (v, fps) in the ring's code at "at": BASE's, xor version v's rows (at + 1 ..) and the rate's
// ai: (at + 9 ..), each row VECS vec4u in the uniform, bit i in u32 i / 32 (a uniform indexed at run time, which the
// ai: Adreno reads right; it is a private array it did not).
fn codeBit(at: u32, v: u32, fps: u32, i: u32) -> u32 {
  let k = i >> 5u;
  var w = W.code[${VECS}u * at + (k >> 2u)][k & 3u];
  for (var b = 0u; b < 8u; b++) {
    if (((v >> b) & 1u) != 0u) { w ^= W.code[${VECS}u * (at + 1u + b) + (k >> 2u)][k & 3u]; }
    if (((fps >> b) & 1u) != 0u) { w ^= W.code[${VECS}u * (at + 9u + b) + (k >> 2u)][k & 3u]; }
  }
  return (w >> (i & 31u)) & 1u;
}
// ai: The bit of the side with the least sum of |q| among those not in skip (ties to the lower side, as wordcode.mjs
// ai: leastSide).
fn leastSide(s: vec4f, skip: u32) -> u32 {
  var b = 0u;
  var v = 3.0e38;
  if ((skip & 1u) == 0u && s.x < v) { v = s.x; b = 1u; }
  if ((skip & 2u) == 0u && s.y < v) { v = s.y; b = 2u; }
  if ((skip & 4u) == 0u && s.z < v) { v = s.z; b = 4u; }
  if ((skip & 8u) == 0u && s.w < v) { v = s.w; b = 8u; }
  return b;
}
// ai: The gate's E[m] for ring r (0xffff: no count of m passes).
fn gateAt(r: u32, m: u32) -> u32 { let k = r * ${GATE}u + (m >> 1u); return (W.gate[k >> 2u][k & 3u] >> (16u * (m & 1u))) & 0xffffu; }
// ai: The picture slot a version decodes at: how many pictures end below it (every top but the last, static
// ai: components).
fn slotOf(v: u32) -> u32 {
  return 0u${Array.from({ length: SLOTS - 1 }, (_, k) => ` + select(0u, 1u, v > W.tops[${k >> 2}].${WORDS[k & 3]})`).join("")};
}
// ai: The image point of a plan point, through the map and the Coons patch of the ring's lattice.
fn imgAt(f: u32, M: array<f32, 16>, L: vec4u, m: vec2f) -> vec2f { let p = toImage(M, m); return p + lensAt(lensOf(f), p) + coons(f, L.x, L.y, m); }
// ai: The workgroup's red[] summed into red[0] (every invocation calls it; red[li] written before).
fn reduce(li: u32) {
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) { red[li] += red[li + st]; }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.x;
  // ai: sel is read-write here, so its load is not uniform: the return goes through the workgroup.
  if (li == 0u) { go = select(0u, 1u, frames[f].valid != 0u && sel[f].y != 0u); }
  if (workgroupUniformLoad(&go) == 0u) { return; }
  let rg = sel[f].x;
  let P = W.plan[rg];
  let M = maps[f];
  let L = LT.at[rg];
  // The module size near the border picks the level, as every read of the border does.
  let c0 = toImage(M, pts[P.x].xy);
  let u = distance(toImage(M, pts[P.x].xy + vec2f(1.0, 0.0)), c0);
  let l = u32(clamp(floor(log2(max(u, 1.0) / 1.5)), 0.0, 4.0));
  // Contrast: the line against the gap, every 16 modules.
  var lg = vec2f(0.0);
  for (var s = li; s < P.y; s += 256u) {
    lg += vec2f(look(l, f, imgAt(f, M, L, pts[P.x + 2u * s].xy)), look(l, f, imgAt(f, M, L, pts[P.x + 2u * s + 1u].xy)));
  }
  red[li] = vec4f(lg, 0.0, 0.0);
  reduce(li);
  let con = (red[0].y - red[0].x) / f32(max(P.y, 1u));
  // The track pairs' means: the local grey each word cell is read against.
  let mp = P.x + 2u * P.y;
  for (var k = li; k < P.z; k += 256u) { mids[k] = 0.5 * (look(l, f, imgAt(f, M, L, pts[mp + 2u * k].xy)) + look(l, f, imgAt(f, M, L, pts[mp + 2u * k + 1u].xy))); }
  workgroupBarrier();
  // ai: One cell a bit, up to 4 WMAX bits: its local grey less the cell, positive dark, over the contrast and clamped
  // ai: to half of it either way.
  let wp = mp + 2u * P.z;
  let vr = W.range[rg];
  let nb = 4u * vr.x;
  for (var bit = li; bit < nb; bit += 256u) {
    let e = pts[wp + bit];
    let lo = u32(e.z) % 4096u;
    let hi = u32(e.z) / 4096u;
    var grey = 0.0;
    for (var j = lo; j <= hi; j++) { grey += mids[j]; }
    q[bit] = clamp((grey / f32(hi - lo + 1u) - look(l, f, imgAt(f, M, L, e.xy))) / max(con, 1e-3), -0.5, 0.5);
  }
  workgroupBarrier();
  // ai: Each side's sum of |q|, in red while the scoring has not yet written it. d1 and d2: the side with the least sum,
  // ai: and it with the next. The search leaves out a side under SIDE_KEEP of the best side's sum, at most those two
  // ai: (keep: the soft form of an erasure, so a glared side's coin signs cannot pull the best word off the true one);
  // ai: the gate tries all four, all but d1, all but d2 (gpu/wordcode.mjs gateChoices).
  var sa = vec4f(0.0);
  for (var bit = li; bit < nb; bit += 256u) { sa += select(vec4f(0.0), vec4f(abs(q[bit])), vec4u(0u, 1u, 2u, 3u) == vec4u(bit / vr.x)); }
  red[li] = sa;
  reduce(li);
  let sides = red[0];
  let d1 = leastSide(sides, 0u);
  let d2 = d1 | leastSide(sides, d1);
  let top = ${SIDE_KEEP} * max(max(sides.x, sides.y), max(sides.z, sides.w));
  let low = select(0u, 1u, sides.x < top) | select(0u, 2u, sides.y < top) | select(0u, 4u, sides.z < top) | select(0u, 8u, sides.w < top);
  let keep = 15u & ~(low & d2);
  workgroupBarrier();${SEARCH}
  workgroupBarrier();
  if (li == 0u) {
    var b = red[0];
    for (var k = 1u; k < 256u; k++) {
      let o = red[k];
      if (o.x > b.x || (o.x == b.x && o.y < b.y)) { b = o; }
    }
    // ai: the winner's version and rate off its own bits (the code is systematic: bytes 1 and 2, sides 1 and 2)
    let sub = unrank(u32(b.y));
    let v = byteOf(sub, vr.x);
    let fr = byteOf(sub, 2u * vr.x);
    let valid = b.x > -1e8 && v >= 1u && v <= ${VERSION_MAX}u;
    win = vec4f(b.x, f32(select(0u, (v - 1u) * 256u + fr, valid)), select(0.0, 1.0, valid), 0.0);
  }
  let wn = workgroupUniformLoad(&win);
  let c = u32(wn.y);
  let v = 1u + c / 256u;
  let fps = c % 256u;
  // ai: The gate, a lane a cell: the cells at |q| >= TAU of the sides each choice counts (all, all but d1, all but d2),
  // ai: each as 1024 plus 1 where the winner disagrees, so one sum carries m (the 1024s) and the disagreements (the
  // ai: rest), exact in f32 to 4 WMAX cells. It was lane 0's alone until it measured 0.03 ms a frame in the 256 ring.
  var g = vec3f(0.0);
  for (var bit = li; bit < nb; bit += 256u) {
    if (abs(q[bit]) < ${TAU}) { continue; }
    let s = 1u << (bit / vr.x);
    let one = 1024.0 + select(0.0, 1.0, (q[bit] > 0.0) != (codeBit(at0, v, fps, bit) != 0u));
    g += select(vec3f(0.0), vec3f(one), vec3<bool>(true, (s & d1) == 0u, (s & d2) == 0u));
  }
  red[li] = vec4f(g, 0.0);
  reduce(li);
  if (li == 0u) {
    let G = red[0];
    let m0 = u32(G.x / 1024.0);
    let m1 = u32(G.y / 1024.0);
    let m2 = u32(G.z / 1024.0);
    let e0 = gateAt(rg, m0);
    let e1 = gateAt(rg, m1);
    let e2 = gateAt(rg, m2);
    let through = (e0 != 0xffffu && u32(G.x) - 1024u * m0 <= e0) || (e1 != 0xffffu && u32(G.y) - 1024u * m1 <= e1) || (e2 != 0xffffu && u32(G.z) - 1024u * m2 <= e2);
    let ok = con >= ${CON_MIN} && wn.z > 0.5 && through;
    result[f][5] = select(0.0, f32(v), ok);
    result[f][6] = select(0.0, f32(fps), ok);
    result[f][7] = wn.x;
    // ai: The version the frame decodes at: its own word's, else the held one, else none.
    let dv = select(HB.version, v, ok);
    result[f][29] = f32(dv);
    result[f][30] = select(0.0, 1.0, !ok && HB.version > 0u);
    if (dv > 0u) { sel[f] = vec4u(rg, 1u + dv, slotOf(dv), 0u); }
    else { sel[f] = vec4u(rg, 0u, 0u, 0u); }
  }
}
`;
