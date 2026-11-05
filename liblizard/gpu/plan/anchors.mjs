// ai: The anchors' constants and uniform (glsl/anchors.mjs ANCH_FIND, ANCH_PREDICT, ANCH_REFIT, ANCH_FIT; SPEC 6.13).
import { RING_COUNT } from "./finder.mjs";
import { ANCH_SIDE, ANCH_K, ANCH_REC, ANCH_SURF } from "./back_transform.mjs";

export { ANCH_SIDE, ANCH_K, ANCH_REC, ANCH_SURF };
// ai: The C's constants (src/acquire.c): template points across a square (OB_ANCH_TS), the peak an anchor is found at
// ai: (ANCH_PEAK), the anchors the geometry takes (src/internal.h OB_ANCH_MIN, the lens model's terms: 8 until
// ai: 2026-10-27), the fill's rounds.
export const ANCH_TS = 16, ANCH_PEAK = 0.6, ANCH_MIN = 12, ANCH_ROUNDS = 6;
// ai: The squares' fill (SPEC 6.13; the C's hole_value): the least-squares surface in 1, u, u^2, w, w^2, u w (the
// ai: detrend's functions, 6.10, in u = 2 (m - margin) / span - 1 of a module coordinate m, w the same of y) over
// ai: ANCH_FILL x ANCH_FILL points of the picture, m = margin + span (2 p + 1) / (2 ANCH_FILL), those in no square.
// ai: A point is in a square where both of its coordinates are within ANCH_BAND a of an anchor's centre: 1.5 a, half
// ai: the side and as much again, past the 2.5 samples the erase reaches beyond a square in every picture of 160
// ai: samples up (a = span / 64 modules). Decided in integers, 2 ANCH_FILL times the module coordinate, in the
// ai: kernel and here alike. 16 a side, the 256 points the picture's mean was taken over before; 32, 64 and 128 read
// ai: within 0.02% of it on recorded captures.
export const ANCH_FILL = 16, ANCH_BAND2 = 3;   // ai: ANCH_BAND2 = 2 ANCH_BAND

// ai: A, the anchors' uniform (ANCH_FIND's and ANCH_FIT's), from gpu/tables.mjs formatTables() rings: three vec4f a
// ai: ring, (k0, k1, k2, k3), (k4, k5, k6, k7), (k8, a, span, margin): the 9 knots of an axis in modules, the lattice's
// ai: ends (nodeX[0] and nodeX[nn - 1], where F7's Coons patch holds the map) and the 7 anchors' centres between, the
// ai: C's m0(i) + a / 2 (src/acquire.c ob_anchor_m0: floor(margin + span (i + 0.5) / 7 - a / 2 + 0.5)); a = span / 64.
// ai: Then, after every ring's three, nine vec4f a ring: the inverse of the fill's normal matrix (fillPoints), row
// ai: by row, 36 floats, fixed by the ring as the C's hole_inv is by the format.
export function anchUniform(rings) {
  if (rings.length !== RING_COUNT) throw new Error(`${rings.length} rings: the anchors' uniform holds ${RING_COUNT}`);
  const u = new Float32Array(12 * RING_COUNT + 36 * RING_COUNT);
  rings.forEach(({ cells, span, margin, nodeX }, r) => {
    const a = Math.floor(span / 64);
    const c = Array.from({ length: ANCH_SIDE }, (_, i) => Math.floor(margin + (span * (i + 0.5)) / ANCH_SIDE - a / 2 + 0.5) + a / 2);
    const k = [nodeX[0], ...c, nodeX[nodeX.length - 1]];
    if (k.some((x, i) => i && !(x > k[i - 1]))) throw new Error(`ring ${cells}: knots ${k.join(", ")} not rising`);
    if (!Number.isInteger(a / 2) || c.some((x) => !Number.isInteger(x)) || !Number.isInteger(span) || !Number.isInteger(margin)) {
      throw new Error(`ring ${cells}: the fill's bands are decided in integers (a ${a}, centres ${c.join(", ")}, span ${span}, margin ${margin})`);
    }
    u.set([...k, a, span, margin], 12 * r);
    u.set(fillInverse(fillPoints({ a, span, margin, c })), 12 * RING_COUNT + 36 * r);
  });
  return u;
}

// ai: The fill's basis at a point and the points it is fitted over (ANCH_FILL's comment), a ring's.
export const fillBasis = (ux, uy) => [1, ux, ux * ux, uy, uy * uy, ux * uy];
export function fillPoints({ a, span, margin, c }) {
  const band = (p) => c.some((ci) => Math.abs(2 * ANCH_FILL * (ci - margin) - span * (2 * p + 1)) <= ANCH_BAND2 * ANCH_FILL * a);
  const pts = [];
  for (let py = 0; py < ANCH_FILL; py++) for (let px = 0; px < ANCH_FILL; px++) {
    if (band(px) && band(py)) continue;
    pts.push([(2 * px + 1 - ANCH_FILL) / ANCH_FILL, (2 * py + 1 - ANCH_FILL) / ANCH_FILL]);
  }
  return pts;
}
// ai: (sum over the points of b b^T)^-1 by Gauss-Jordan with partial pivots in float64, row by row
export function fillInverse(pts) {
  const n = 6, g = Array.from({ length: n }, (_, i) => [...Array(n).fill(0), ...Array.from({ length: n }, (_, j) => +(i === j))]);
  for (const [ux, uy] of pts) {
    const b = fillBasis(ux, uy);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) g[i][j] += b[i] * b[j];
  }
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let i = col + 1; i < n; i++) if (Math.abs(g[i][col]) > Math.abs(g[piv][col])) piv = i;
    [g[col], g[piv]] = [g[piv], g[col]];
    const d = g[col][col];
    if (!(Math.abs(d) > 1e-9)) throw new Error(`the fill's normal matrix is singular (${pts.length} points)`);
    for (let j = 0; j < 2 * n; j++) g[col][j] /= d;
    for (let i = 0; i < n; i++) if (i !== col) { const m = g[i][col]; for (let j = 0; j < 2 * n; j++) g[i][j] -= m * g[col][j]; }
  }
  return g.flatMap((row) => row.slice(n));
}
