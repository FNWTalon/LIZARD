// ai: F8: the format word, read through the final map (homography, lens and the border's Coons patch), one workgroup
// ai: a frame, after F7 and before the back half (gpu/decoder.mjs). The reads are the format's (gpu/tables.mjs word
// ai: plan, the reference codec's own): the solid line and the gap every 16 modules give the contrast; each word cell
// ai: is read at its planned centre against the mean of the Manchester pairs of the track near it (a pair is one dark
// ai: and one light cell, so its mean is the local grey). One code a ring over every word cell, W = 16, 32, 48, 64 a
// ai: side (src/fmt.h), so 4 W soft bits, one cell each. Then every valid word is scored against them and the best
// ai: kept (gpu/wordcode.mjs): maximum likelihood over VERSION_MAX x 256 words a ring (any ring carries any version
// ai: since 2026-09-27), where the reference decodes hard bytes. A word is taken if the contrast reads (CON_MIN, as
// ai: src/acquire.c) and the best word disagrees with at most DMAX[W] of the hard bits.
// ai: The frame then decodes at the word's version, or, with no word taken, at the batch's held version (the last word
// ai: the host read, 0 for none, since 2026-09-27: the ring bootstraps the word, and a frame whose word does not decode
// ai: leaves the held one as it is); with neither it decodes nothing. So sel[f].y = 1 + that version (the back
// ai: half's gate reads it as the frame's blocks) and sel[f].z its picture slot (gpu/tables.mjs PICTURES), or sel[f].y
// ai: = 0 and the gate drops the frame. result[f]: [5] the word's version (0 for none), [6] fps, [7] the best word's
// ai: soft score; [29] the version the frame decodes at (0 for none), [30] 1 where the held version stood in.
// ai: No private array: a codeword is a struct of VECS vec4u (Cw) built from the uniform with static members and
// ai: components, and the lattice is read from a uniform (the S26's Adreno returned zeros from a private array indexed
// ai: at run time).
import { DIMS, MAP, SLOTS } from "./common.mjs";
import { COONS } from "./sample.mjs";
import { LATTICE_MAX } from "./back_transform.mjs";
import { RING_COUNT } from "./finder.mjs";
import { wordCode, VERSION_MAX } from "../wordcode.mjs";

export const CON_MIN = 0.08;
// ai: The most hard bits the best word may disagree with, by word cells a side (so by ring): 3/16 of the bits, or
// ai: fewer where the union bound on a random cell set taking a word (every one of the ring's 32,768 words within DMAX
// ai: of a random hard word) would pass 1e-6 a frame. 64 bits: 6 (1.5e-7; 7 gives 1.3e-6); 128 bits: 24, 3/16
// ai: (7.5e-9); 192 bits: 36, 3/16 (8.6e-15); 256 bits: 48, 3/16 (1.1e-20). Until the evening of 2026-09-27 the rings
// ai: were 32, 48, 64 and 96 bits took 16 (3.4e-7); until that morning each picture size scored only its own versions
// ai: (1,024 to 14,592 words): 2/16 at n = 256 (2.9e-7), 3/16 elsewhere. scripts/gpu/word_check.mjs prints the bounds and a
// ai: garbage trial.
export const DMAX = { 16: 6, 32: 24, 48: 36, 64: 48 };
// ai: WMAX: word cells a side at most (the 128 ring), so a word's 4 WMAX bits fill VECS vec4u (a Cw); ROWS: BASE and
// ai: the 16 rows a ring; MIDS_MAX: track pairs a frame, one an invocation.
export const WMAX = 64, ROWS = 17, MIDS_MAX = 256;
export const VECS = (4 * WMAX) / 128;
// ai: vec4u the pictures' last versions take, four a vec4u.
const TOPS = Math.ceil(SLOTS / 4);
// ai: The Word uniform's layout in vec4u: code (every ring's ROWS rows, a row VECS vec4u: ring r's row k at (r ROWS +
// ai: k) VECS), then range (W, DMAX, 0, 0) a ring, then plan a ring, then tops (the last version of pictures 0 to
// ai: SLOTS - 1).
export const LAYOUT = (() => {
  const range = RING_COUNT * ROWS * VECS, plan = range + RING_COUNT, tops = plan + RING_COUNT;
  return { code: 0, range, plan, tops, size: tops + TOPS };
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
// ai: Cw's members v0 .. v(VECS - 1), and every u32 of a codeword by its static name, bits 32 k to 32 k + 31 in the k-th.
const MEMBERS = Array.from({ length: VECS }, (_, j) => `v${j}`);
const U32S = MEMBERS.flatMap((m) => WORDS.map((c) => `${m}.${c}`));

export const WORD = DIMS + MAP + /* wgsl */ `
struct LevelDims { d: array<vec4u, 5> }
// ai: code: per ring, BASE then the 16 rows (version bits, then fps bits), VECS vec4u each; range[ring] = (word cells
// ai: a side W, DMAX, 0, 0); plan[ring] = (first point, reference steps, track pairs, word points); tops: the last
// ai: version of pictures 0 to SLOTS - 1, four a vec4u (LAYOUT).
struct Word { code: array<vec4u, ${LAYOUT.range}>, range: array<vec4u, ${RING_COUNT}>, plan: array<vec4u, ${RING_COUNT}>, tops: array<vec4u, ${TOPS}> }
// ai: A codeword or a row of the code, 4 WMAX bits: bit i in u32 i / 32 of v0, v1, ... in turn (4 W of them used).
struct Cw { ${MEMBERS.map((m) => `${m}: vec4u`).join(", ")} }
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
// ai: Row i of the code (every ring's rows in turn).
fn rowAt(i: u32) -> Cw { return Cw(${MEMBERS.map((_, j) => `W.code[${VECS}u * i + ${j}u]`).join(", ")}); }
// ai: rows first .. first + 7 of the ring's code at "at", those whose bit of x is set, xored into cw.
fn addRows(cw: Cw, at: u32, first: u32, x: u32) -> Cw {
  var o: Cw = cw;
  for (var b = 0u; b < 8u; b++) {
    let m = select(vec4u(0u), vec4u(0xffffffffu), ((x >> b) & 1u) != 0u);
    let r = rowAt(at + first + b);
    ${MEMBERS.map((v) => `o.${v} ^= r.${v} & m;`).join(" ")}
  }
  return o;
}
// ai: One u32 of a codeword against q[base ..]: a set bit (dark) adds its soft value, a clear one takes it away.
fn wordScore(w: u32, base: u32, nb: u32) -> f32 {
  var s = 0.0;
  for (var i = base; i < min(nb, base + 32u); i++) { s += select(-q[i], q[i], ((w >> (i - base)) & 1u) != 0u); }
  return s;
}
fn score(cw: Cw, nb: u32) -> f32 {
  return ${U32S.map((c, k) => `wordScore(cw.${c}, ${32 * k}u, nb)`).join(" + ")};
}
// ai: The hard bits a codeword disagrees with.
fn wordDist(w: u32, base: u32, nb: u32) -> u32 {
  var d = 0u;
  for (var i = base; i < min(nb, base + 32u); i++) { d += select(0u, 1u, (q[i] > 0.0) != (((w >> (i - base)) & 1u) != 0u)); }
  return d;
}
fn dist(cw: Cw, nb: u32) -> u32 {
  return ${U32S.map((c, k) => `wordDist(cw.${c}, ${32 * k}u, nb)`).join(" + ")};
}
// ai: The picture slot a version decodes at: how many pictures end below it (every top but the last, static
// ai: components).
fn slotOf(v: u32) -> u32 {
  return 0u${Array.from({ length: SLOTS - 1 }, (_, k) => ` + select(0u, 1u, v > W.tops[${k >> 2}].${WORDS[k & 3]})`).join("")};
}
// ai: The image point of a plan point, through the map and the Coons patch of the ring's lattice.
fn imgAt(f: u32, M: array<f32, 16>, L: vec4u, m: vec2f) -> vec2f { return toImage(M, m) + coons(f, L.x, L.y, m); }
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
  if (li < P.z) { mids[li] = 0.5 * (look(l, f, imgAt(f, M, L, pts[mp + 2u * li].xy)) + look(l, f, imgAt(f, M, L, pts[mp + 2u * li + 1u].xy))); }
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
  // ai: Every word, against the soft bits: invocation li scores rate li at every version, so the rate's rows are added
  // ai: once. Candidate c = (v - 1) 256 + fps; ties go to the lower c.
  let at0 = rg * ${ROWS}u;
  let fp = addRows(rowAt(at0), at0, 9u, li);
  var best = -1e9;
  var bv = 1u;
  for (var v = 1u; v <= ${VERSION_MAX}u; v++) {
    let sc = score(addRows(fp, at0, 1u, v), nb);
    if (sc > best) { best = sc; bv = v; }
  }
  red[li] = vec4f(best, f32((bv - 1u) * 256u + li), 0.0, 0.0);
  workgroupBarrier();
  if (li == 0u) {
    var b = red[0];
    for (var k = 1u; k < 256u; k++) {
      let o = red[k];
      if (o.x > b.x || (o.x == b.x && o.y < b.y)) { b = o; }
    }
    let c = u32(b.y);
    let v = 1u + c / 256u;
    let fps = c % 256u;
    let d = dist(addRows(addRows(rowAt(at0), at0, 9u, fps), at0, 1u, v), nb);
    let ok = con >= ${CON_MIN} && d <= vr.y;
    result[f][5] = select(0.0, f32(v), ok);
    result[f][6] = select(0.0, f32(fps), ok);
    result[f][7] = b.x;
    // ai: The version the frame decodes at: its own word's, else the held one, else none.
    let dv = select(HB.version, v, ok);
    result[f][29] = f32(dv);
    result[f][30] = select(0.0, 1.0, !ok && HB.version > 0u);
    if (dv > 0u) { sel[f] = vec4u(rg, 1u + dv, slotOf(dv), 0u); }
    else { sel[f] = vec4u(rg, 0u, 0u, 0u); }
  }
}
`;
