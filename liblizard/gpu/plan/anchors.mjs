// ai: The anchors' constants and uniform (glsl/anchors.mjs ANCH_FIND, ANCH_PREDICT, ANCH_REFIT, ANCH_FIT; SPEC 6.13).
import { RING_COUNT } from "./finder.mjs";
import { ANCH_SIDE, ANCH_K, ANCH_REC } from "./back_transform.mjs";

export { ANCH_SIDE, ANCH_K, ANCH_REC };
// ai: The C's constants (src/acquire.c): template points across a square (OB_ANCH_TS), the peak an anchor is found at
// ai: (ANCH_PEAK), the anchors the geometry takes (src/internal.h OB_ANCH_MIN, the lens model's terms: 8 until
// ai: 2026-10-27), the fill's rounds.
export const ANCH_TS = 16, ANCH_PEAK = 0.6, ANCH_MIN = 12, ANCH_ROUNDS = 6;

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
