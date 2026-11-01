// ai: F8's shape and its tables (glsl/word.mjs WORD): the OSD's constants, the Word uniform's layout and the packers of
// ai: its code, plan points and lattice (the host's build; scripts/gpu/word_check.mjs's JS mirror).
import { SLOTS } from "./common.mjs";
import { LATTICE_MAX } from "./back_transform.mjs";
import { RING_COUNT } from "./finder.mjs";
import { wordCode, VERSION_MAX, gateTable } from "../wordcode.mjs";

export const CON_MIN = 0.08;
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
// ai: k) VECS), then range (W, 0, 0, 0) a ring, then plan a ring, then tops (the last version of pictures 0 to
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
    if (word.cells > WMAX) throw new Error(`ring ${r}: ${word.cells} word cells a side, F8 holds ${WMAX}`);
    const { base, rows } = wordCode(word.cells);
    [base, ...rows].forEach((v, k) => u.set(v, 4 * VECS * (r * ROWS + k)));
    gateTable(word.cells).forEach((e, m) => { u[4 * LAYOUT.gate + GATE * r + (m >> 1)] |= (e < 0 ? 0xffff : e) << (16 * (m & 1)); });
    u.set([word.cells, 0, 0, 0], 4 * (LAYOUT.range + r));
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
