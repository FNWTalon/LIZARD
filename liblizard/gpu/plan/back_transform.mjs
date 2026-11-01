// ai: The back half's transform's layouts (glsl/back_transform.mjs: the gate, the fused pass 1, reduce, pass 2): LISTS,
// ai: ARGS, the gate's uniform, PIC's lattice, the anchors' record and pass 1's workgroup memory, which ../back/transform.mjs,
// ai: ../back/back.mjs and ../../gen/gen.mjs read.
import { SIZES, SLOTS } from "./common.mjs";
import { NC } from "./back_tiers.mjs";

// ai: The picture sizes and their count (common.mjs), a frame's picture slot its index in SIZES (sel.z, from F8's
// ai: version).
export { SIZES, SLOTS };
export const CPW = 4;   // picture columns a pass 1 workgroup: a quad is one 16 B load a row and one 16 B store of Y

// ai: ARGS is u32 [SLOTS][12]: pass 1's, reduce's, pass 2's and the soft values' indirect dispatches (the rule stage's
// ai: talign and tblocks take reduce's, a workgroup a listed frame; tsoft its own, the chunks of the size's most
// ai: version a listed frame takes, ../back/tiers.mjs exportFor), then each code's LDPC dispatch (argsLdpc): the most
// ai: blocks of the code any listed frame's version takes, the listed frames, 1.
export const ARGS_WORDS = 12;
export const argsLdpc = (c) => SLOTS * ARGS_WORDS + 3 * c;
// ai: ARGS whole, u32 words: ARGS_WORDS a picture slot, then 3 a code.
export const argsWords = SLOTS * ARGS_WORDS + 3 * NC;
// ai: LISTS: a size's count and frames (SLOTS (1 + B)), then from listsHead each frame's (w, h), version, rows and
// ai: ring, then from listsAll every listed frame of every size in the batch's order (the LDPC's workgroups' frames:
// ai: a workgroup a (block of a code, listed frame)).
export const listsHead = (B) => SLOTS * (1 + B);
export const listsWords = (B) => listsHead(B) + 6 * B;
export const listsBlocks = (B) => listsHead(B) + 2 * B;
export const listsRows = (B) => listsHead(B) + 3 * B;
export const listsRing = (B) => listsHead(B) + 4 * B;
export const listsAll = (B) => listsHead(B) + 5 * B;
export const ARGS_SLOT = { pass1: 0, reduce: 3, pass2: 6, soft: 9 };
// ai: The gate's uniform in u32 words: at[s] = (n / CPW, V, the size's top version, 0) a slot, then the rows table
// ai: flat, entry v SLOTS + s the rows version v takes at slot s (transform.mjs fills it), padded to whole vec4u.
// ai: counts: versions 0 to the largest built size's top. Then the codes table, a word a version: each code's blocks in
// ai: the version's rate profile, a byte a code in CODES' order (../back/codes.mjs ruleTables).
export const gateRowsAt = (c, s) => 4 * SLOTS + c * SLOTS + s;
export const gateCodesAt = (counts, v) => 4 * SLOTS + 4 * Math.ceil((counts * SLOTS) / 4) + v;
export const gateWords = (counts) => gateCodesAt(counts, 0) + 4 * Math.ceil(counts / 4);

export const LATTICE_MAX = 36;   // node coordinates a side PIC holds (register.mjs NODES_MAX / 4 is 34)

// ai: The anchors (SPEC 6.13; glsl/anchors.mjs; the C's src/acquire.c anchor_find and border_fill_anchors, src/focus.c
// ai: focus_finish_ext's erase): ANCH_SIDE x ANCH_SIDE anchors, ANCH_K a frame. ANCH_FIT writes a frame's record,
// ai: ANCH_REC vec4f, which pass 1 reads: [0] (on, the picture's mean, 0, the anchors found);
// ai: [1] to [3] the 9 knots of an axis in modules (the lattice's ends, the anchors' centres between) and the anchor's
// ai: side a, the ring's span and margin, as (k0, k1, k2, k3), (k4, k5, k6, k7), (k8, a, span, margin); [4] on, the
// ai: inner knots' residuals in image px,
// ai: knot q = 7 j + i in .xy of [4 + q / 2] for an even q, .zw for an odd. Here, not in anchors.mjs, because pass 1
// ai: reads it and anchors.mjs imports this module.
export const ANCH_SIDE = 7, ANCH_K = ANCH_SIDE * ANCH_SIDE, ANCH_REC = 4 + Math.ceil(ANCH_K / 2);
// ai: The samples the erase takes round a square on the picture's grid (src/internal.h OB_ANCH_MARGIN).
export const ANCH_ERASE = 2;

// ai: Pass 1's workgroup memory for the anchors: what a quad's four columns blend of each knot row, -1 to 9 (anE: rows
// ai: 1 to 7 the inner knots', the rest 0, so a sample's 4 x 4 blend reads it with no test), an axis's 9 knots (anK),
// ai: each anchor's erased samples on this grid, first and last (anLo, anHi), and whether the erase takes each of the
// ai: quad's four columns (anCol: four lanes answer it once, where every lane asked it of all seven anchors before).
const ANCH_ROWS = ANCH_SIDE + 4, ANCH_P1_BYTES = 8 * 4 * ANCH_ROWS + 4 * (ANCH_SIDE + 2) + 2 * 4 * ANCH_SIDE + 4 * 4;
// Workgroup memory pass 1 declares for a size: the two exchange buffers and 16 B a thread of partials.
// ai: and the anchors' ANCH_P1_BYTES.
export const pass1WorkgroupBytes = (n, prec) => 2 * n * (prec === "f16" ? 4 : 8) + 16 * (n / 8) + 16 * (n / 8 + 1) + ANCH_P1_BYTES;
