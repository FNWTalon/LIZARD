// The format as the GPU decoder needs it, taken from the reference codec rather than restated, so the symbol is
// defined in one place (src/). Runs in node and in the browser: sim/ob.mjs loads build/ob.wasm in both.
//
// ai: Two kinds of table since the rings (2026-09-27). A RING is the border: its module count, node lattice, kind map,
// ai: corner marks, track plan and word plan, the same whatever picture it carries. The finder and F7 register against
// ai: every ring and F8 reads the word under the ring's plan. A PICTURE is its sample count n (PICTURE_SIZES, 256 to
// ai: 1536), which only the word names (N_FOR of the version's sub-channels); the back half's tables are a picture's.
// ai: Any ring may carry any picture, and a frame's sample grid is the pair's (gridOf).
// ai: Everything below is in MODULE coordinates: module k spans [k, k + 1), so a cell's middle is k + 0.5.
import { init, Focus } from "../sim/ob.mjs";
import { RINGS, RING_DEFAULT, PICTURE_SIZES } from "../sim/lizard_pick.mjs";
import { versionRange } from "./wordcode.mjs";

export { RINGS };
// ai: The ring a told spec is painted in: its span's (sim/lizard_pick.mjs SPAN), or the default ring.
export const ringOfSpec = (spec) => (spec.span ? RINGS.indexOf(spec.span / 2) : RING_DEFAULT);
// ai: The picture sizes, each with the versions whose picture it is (lo to hi) and its largest sub-channel count:
// ai: every version is a prefix of its picture's top (gpu/wgsl/back_transform.mjs listsBlocks), so one set of back
// ai: half tables a picture serves every version. A picture's slot is its index here (sel.z on the device).
export const PICTURES = PICTURE_SIZES.map((n) => { const [lo, hi] = versionRange(n); return { n, subch: 8 * hi, lo, hi }; });
// ai: The picture slot a version decodes at, -1 outside the word's range (1 to 128).
export const slotOf = (version) => PICTURES.findIndex((p) => version >= p.lo && version <= p.hi);
// ai: Where picture sample 0 sits and how far apart samples are, in modules, for picture n in ring table t: the
// ai: codec's margin + 0.5 / scale and 1 / scale, scale = n / span (src/wasm.c focus_grid_out).
export const gridOf = (t, n) => ({ g0: t.margin + (0.5 * t.span) / n, step: t.span / n });

// ai: { rings, pictures }: rings[r] for ring r (RINGS[r] band cells a side), read from the codec at n = 256 in that
// ai: ring (the border does not depend on the picture); pictures is PICTURES.
export async function formatTables() {
  const M = await init();
  const f32 = (p, n) => M.HEAPF32.slice(p >> 2, (p >> 2) + n), i32 = (p, n) => M.HEAP32.slice(p >> 2, (p >> 2) + n);
  const rings = RINGS.map((cells, ring) => {
    const fc = new Focus(256, PICTURES[0].subch, 1, { span: 2 * cells });
    const scratch = M._malloc(4 * 64);
    // The lattice, the kind map (CELL_* per module: 1 light, 2 dark, flag 8 a format-word cell) and which nodes
    // carry a mark.
    M._ob_test_mesh_tables_out(scratch, 0, 0, 0, 0);
    const [nx, ny, w, h] = i32(scratch, 4);
    const pX = M._malloc(4 * nx), pY = M._malloc(4 * ny), pK = M._malloc(w * h), pM = M._malloc(4 * nx * ny);
    M._ob_test_mesh_tables_out(scratch, pX, pY, pK, pM);
    const nodeX = f32(pX, nx), nodeY = f32(pY, ny), kind = M.HEAPU8.slice(pK, pK + w * h), nodeMark = i32(pM, nx * ny);
    // Mark geometry: side, core, merged square, gapped centre, merged centre (in modules from the corner).
    M._ob_test_marks(scratch);
    const [, markSide, core, merged, gappedMid, mergedMid] = f32(scratch, 8);
    // The track read's sample plan: per 16-module step, the line, the gap and the Manchester pair nearest it (the
    // levels), then one point per track cell with the sign its side's sequence gives it.
    const planCap = 16 * (4 * w + 4 * h);
    const pPts = M._malloc(8 * planCap), pMeta = M._malloc(4 * 32), pSig = M._malloc(4 * planCap);
    const np = M._ob_test_thin_plan(pPts, pMeta, pSig);
    const tmeta = i32(pMeta, 9);
    const track = { refSteps: tmeta[0], cells: tmeta[1], perSide: Array.from(tmeta.subarray(4, 8)), pts: f32(pPts, 2 * np), signs: i32(pSig, tmeta[1]) };
    // The format word's plan: the levels (line and gap every 16 modules), the track pairs a word cell is read
    // against, and every word cell with its pair range and the soft value it is (one code over every word cell).
    const wcells = M._ob_test_fmt_cells();
    const pW = M._malloc(4 * 3 * planCap);
    const nw = M._ob_test_fmt_plan(wcells, pPts, pMeta, pW);
    const wmeta = i32(pMeta, 17);
    const word = { cells: wcells, refSteps: wmeta[0], reps: wmeta[2], mids: wmeta[3], words: wmeta[4], pairs: Array.from(wmeta.subarray(5, 9)),
      pts: f32(pPts, 2 * nw), widx: i32(pW, 3 * wmeta[4]) };
    // ai: The ring's span (modules across the picture) and margin (border modules before it) from the codec's grid
    // ai: at n = 256: g0 = margin + 0.5 step, step = span / 256.
    M._focus_grid_out(scratch);
    const [g0, step] = f32(scratch, 2);
    const span = Math.round(256 * step), margin = g0 - 0.5 * step;
    [scratch, pX, pY, pK, pM, pPts, pMeta, pSig, pW].forEach((p) => M._free(p));
    fc.free();
    if (span !== 2 * cells || w !== span + 2 * margin) throw new Error(`ring ${cells}: the codec's span ${span}, margin ${margin} and side ${w} disagree with sim/lizard_pick.mjs`);
    return { ring, cells, span, margin, modules: w, nodeX, nodeY, nodeMark, kind, mark: { side: markSide, core, merged, gappedMid, mergedMid }, track, word };
  });
  return { rings, pictures: PICTURES };
}
