// The LDPC's stop rules and the CRC-32 split's powers, shared by the rate profile's stage (tiers.mjs,
// wgsl/back_tiers.mjs) and the sender's paint (../encoder.mjs, wgsl/send.mjs).
// ai: The JS reference decoder that held the one-rate kernel to the C (decodeStall, the tracer, finishBlocks) and the
// ai: split's self-check went 2026-10-22 with that kernel and its tests.
import { loadStop, ruleUrlFor } from "./stop/rule.mjs";

export const PAYLOAD = 473;                   // bytes a block, the id first
export const STALL_RATIO = Math.fround(0.95);
export const STALL_IT = 9;

// ai: The stop rules (DESIGN section 6): a rule is { name, stall, funnel, net }, what wgsl/back_tiers.mjs builds a
// ai: code's kernel from.
// ai: - "learned", the first read's since 2026-09-28: after iterations 1 to 29, the MLP of stop/ (net, from
// ai:   stop/stop.safetensors) on (bad, low, first, prev, t, est) gives up when its score is under its threshold. It
// ai:   replaced the funnel (the C's stall test and seven checkpoints of the running minimum, 2026-09-24 to 28), which
// ai:   gave up 4 verified blocks of the three dumps where the rule gives up none (scripts/gpu/back/stop/README.md).
// ai: - "c", the C's rule: the stall test (after iteration 9, the count above 0.95 of the count after iteration 1) and
// ai:   the cap of 30. The check that the GPU's min-sum arithmetic is the C's: under it the GPU gives the C's verdict and
// ai:   iterations on every block (a page's ?ldpcstop=c).
export const FIRST_READ = ["learned", "c"];
export const DEFAULT_STOP = "learned";
export const C_RULE = Object.freeze({ name: "c", stall: true, funnel: [], net: null });
// ai: The rule a name gives for a code with m checks (a checkpoint's count is ceil(fraction m)); "learned" loads that
// ai: code's weights file (stop/rule.mjs ruleUrlFor, loadStop, once a file; the 7/8 and 1/2 codes' since 2026-10-07).
export async function stopRule(name, m) {
  if (name === "c") return C_RULE;
  if (name === "learned") {
    const net = await loadStop(ruleUrlFor(m));
    if (net.m !== m) throw new Error(`stop rule: the file for ${m} checks was trained on a code of ${net.m}`);
    return { name, stall: false, funnel: [], net };
  }
  throw new Error(`ldpc stop rule ${name}: one of ${FIRST_READ.join(", ")}`);
}
// ---- the CRC-32 split (DESIGN section 7, as built) ----
//
// The reflected bitwise step is multiplication by x in GF(2)[x] / P with x^0 at bit 31. Both halves of the
// algorithm are linear in the message, so the CRC of 473 bytes is the XOR over 4-byte chunks of (chunk from a
// zero state) times x^(8 * bytes after the chunk), plus the initial state pushed through the whole message, then
// the final inversion. The shader gives each lane one chunk, one table entry and a 32-step multiply.
const P = 0xedb88320;
const mulx = (c) => ((c >>> 1) ^ (P & -(c & 1))) >>> 0;
function xpow(e) { let c = 0x80000000; for (let i = 0; i < e; i++) c = mulx(c); return c >>> 0; }
// c times the polynomial p (both in the reflected representation).
function mulPoly(c, p) { let r = 0, t = c >>> 0; for (let j = 0; j < 32; j++) { if ((p >>> (31 - j)) & 1) r ^= t; t = mulx(t); } return r >>> 0; }
// pw[w] for chunk w (bytes 4 w .. 4 w + 3, the last one byte 472 alone); pw[119] = the init term with the final
// inversion folded in. What the shader's uniform carries.
export function crcPowers(len = PAYLOAD) {
  const words = Math.ceil(len / 4), pw = new Uint32Array(words + 1);
  for (let w = 0; w < words; w++) { const end = Math.min(4 * w + 4, len); pw[w] = xpow(8 * (len - end)); }
  pw[words] = (mulPoly(0xffffffff, xpow(8 * len)) ^ 0xffffffff) >>> 0;
  return pw;
}
