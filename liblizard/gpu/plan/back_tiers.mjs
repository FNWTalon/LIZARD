// ai: The rate profile's stage's constants, TAB's layout and each code's shape (glsl/back_tiers.mjs talign, tsoft, tblocks,
// ai: tldpc; ../back/tiers.mjs builds it), which the sender's paint (../wgsl/send.mjs, ../encoder.mjs) shares: CODES, KNOWN, NC.
import { VERSION_MAX } from "../wordcode.mjs";
import { STEP } from "../../sim/lizard_pick.mjs";

// ai: The grid's shift by the pilots (2026-10-01; the format paints the pilots whatever the phone, so a decoder may
// ai: steer by them; src/focus.c pilot_align, the same fit). A grid (dx, dy) samples off turns coefficient (u, v) by
// ai: 2 pi (u dx + v dy) / n. `talign` (one workgroup a frame, a thread a block, before the soft values in their pass
// ai: and on their bind groups) reads every 7/8 and 3/4 block's 8 or 16 pilot coefficients (its tail, its last slots),
// ai: each against its known symbol with the block's sign from its own pilots' sum, fits zi = k zr (u dx + v dy) by
// ai: least squares twice (the second about the first's answer), and keeps the shift only where it stands ALIGN_CHI over the fit's residual variance (one
// ai: chance in a thousand for noise), else 0, 0. It goes to the end of PILOT: PILOT[B blocksMax + 2 f] and the next.
// ai: The soft stage turns every coefficient back by it as it gathers (a turn leaves |y|, so the moments and the
// ai: estimate read as before). UV's second half (from P.ucOff) is each coefficient's (u, v), u in the high 16 bits
// ai: (signed), v in the low.
export const ALIGN_CHI = 13.8;

export const TIER_BLOCKS = 128;   // ai: blocks a frame at most (talign's and tblocks' lanes; the profile's top is 116)
// ai: TT holds rows 0 to VERSIONS, the versions a word names (../wordcode.mjs VERSION_MAX); row 0 is zero (no format).
export const VERSIONS = VERSION_MAX;
// ai: the 8-sub-channel chunks a version takes (16 sub-channels, sim/lizard_pick.mjs STEP): the soft stage's workgroups
// ai: are a chunk each, and the gate's rows table (gpu/back/transform.mjs rowsOfVersion) counts chunks.
export const CHUNKS = STEP / 8;
export const KNOWN = 3840;        // ai: (476 + 4) x 8: the bits a block's payload and CRC fill, the k of the 3/4, 2/3 and 1/2 codes; past them
                                  // ai: the 7/8 code's data bits are zeros
export const WHITE_SLOTS = 1024 * 640;
// ai: The LDPC kernel's forms (2026-10-08, the decoder's speed; research STATUS "The LDPC's speed"), all counting the
// ai: violated checks after each iteration from a bitmap of the posteriors' signs, 32 checks of a row at a time
// ai: (violatedChunks), and keeping each check's record in its lane's own array (lane i is check i of every row, so no
// ai: other lane reads it): "reg" two posteriors a word in workgroup memory (atomic stores); "reg+i32" a posterior a
// ai: word with plain loads and stores where the code's workgroup memory still fits 32 KB; "reg+i16" (2026-10-31) a
// ai: posterior an int16_t with plain loads and stores, "reg"'s bytes without its atomics, its module needing the
// ai: device's shaderInt16 (its pipelines' `needs`: the host leaves it out where that is missing). All are exact (the
// ai: same decisions and iterations) and differ by workgroup memory alone, so the one a device runs faster is the
// ai: faster on any batch: a generated tree carries every form and the native host times them on its first batch and
// ai: keeps the faster ("reg+i16" on the iGPU and the S26, at one and two frames a launch and batched: research STATUS
// ai: "The LDPC's posteriors as int16_t"; "reg" and "reg+i32" before); a browser (the lab's harness in Chrome) builds
// ai: "reg" alone. The pair form (a check on two lanes, the 1/2 code's rows side by side) went 2026-10-31: in GLSL with
// ai: the slot loops unrolled "reg+i32" beat it on the S26's iterating captures at every launch, and its lower cost on
// ai: clean batches at one and two frames a launch turned the first-batch pick wrong (research STATUS "GLSL as the GPU
// ai: decoder's native source"). The lab's other forms (records in workgroup memory, a parity bit array; LIZ_TLDPC)
// ai: went 2026-10-22; the check's own values in 16-bit registers ("reg+i16+r16") was slower on both devices
// ai: (2026-10-31).
export const LDPC_FORMS = typeof process !== "undefined" ? ["reg", "reg+i32", "reg+i16"] : ["reg"];
// ai: the codes TT names, in order: src/focus.c focus_tiers_for's four rates (ldpc.h index, slots a block: the code's
// ai: sent bits and its pilots, src/focus.c focus_block_slots). Every block's slots are a multiple of 16, so every
// ai: block's first slot is too (the blocks are packed from the frame's first slot, the remainder padded).
export const CODES = [{ rate: 6, slots: 4496, name: "7/8" }, { rate: 4, slots: 5152, name: "3/4" }, { rate: 3, slots: 5760, name: "2/3" }, { rate: 2, slots: 7680, name: "1/2" }];
// ai: TT's words a version: (blocks, used slots, 0, 0), then (first block, first slot, blocks, 0) a code in CODES' order.
// ai: 2/3 among the codes since 2026-10-07, in the format's profile since 2026-10-08 (four rates, 7/8, 3/4, 2/3, 1/2).
export const NC = CODES.length;
export const TTW = 4 + 4 * NC;

// ai: TAB's sections' word offsets, from the codes' n and the nets' words (all codes' together).
export function tabLayout(codes, netWords) {
  let o = TTW * (VERSIONS + 1);
  const inv = codes.map((c) => { const at = o; o += Math.ceil(c.n / 2); return at; });
  const WHITE = o;
  o += WHITE_SLOTS / 32 + 1;   // ai: one word past the last slot: whiteWord reads the word after a slot's
  const NET = o;
  o += netWords;
  return { TT: 0, inv, WHITE, NET, words: o };
}

// ai: A code's module constants: its shape (n, k, z, block rows mb, the most slots a row, staircase included), the
// ai: workgroup's lanes (z rounded up to 32, at least 128), whether a check's record fits one word, and the kernel's
// ai: form (LDPC_FORMS; wide: i32, only where it fits 32 KB; narrow: i16).
export function tierShape(code, form = LDPC_FORMS[0]) {
  const { n, k, z, mb, lay } = code;
  let nd = 0;
  for (let r = 0; r < mb; r++) nd = Math.max(nd, lay[r + 1] - lay[r]);
  const ns = nd + 2, slots = lay[mb];
  const parts = form.split("+");
  if (!parts.includes("reg") || parts.some((p) => !["reg", "i32", "i16"].includes(p)) || (parts.includes("i32") && parts.includes("i16")))
    throw new Error(`LDPC form ${form}: one of ${LDPC_FORMS.join(", ")}`);
  return { n, k, z, mb, ns, slots, threads: Math.max(128, 32 * Math.ceil(z / 32)), one: ns <= 12, rowv: Math.ceil((mb + 1) / 4), pairv: Math.ceil(slots / 4), form,
    wide: parts.includes("i32") && 4 * n + 4 * packWords(n) + 12 + 4 * (6 + 2 * 64) <= 32768, narrow: parts.includes("i16") };
}
// ai: packw's words: the 120 of a block's data for the CRC, and between iterations the posteriors' signs (N bits and a
// ai: word past them, which get32 may read).
const packWords = (n) => Math.max(120, Math.ceil(n / 32) + 1);
// ai: Workgroup bytes a code's kernel declares (32 KB is the floor every phone we know offers).
// ai: ("reg+i16"'s int16_t array, 2 ceil(n / 2) of them, is "reg"'s bytes)
export const tierWgBytes = (sh, net) => (sh.wide ? 4 * sh.n : 4 * Math.ceil(sh.n / 2)) + 4 * packWords(sh.n) + 16 + 4 * (6 + 2 * net.hidden);

export const tierUniformWords = (sh) => 4 + 4 * sh.rowv + 4 * sh.pairv + 120;
