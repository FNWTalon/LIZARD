// The LDPC's learned stop for a code and the CRC-32 split's powers, shared by the rate profile's stage (tiers.mjs,
// wgsl/back_tiers.mjs) and the sender's paint (../encoder.mjs, wgsl/send.mjs).
import { loadStop, ruleUrlFor } from "./stop/rule.mjs";

export const PAYLOAD = 473;                   // bytes a block, the id first

// ai: A code's learned stop (DESIGN section 6), what wgsl/back_tiers.mjs builds the code's kernel from: after
// ai: iterations 1 to 29 the MLP of stop/ on (bad, low, first, prev, t, est) gives up when its score is under its
// ai: threshold, each code its own file by its m checks (stop/rule.mjs ruleUrlFor, loadStop, once a file). It replaced
// ai: the funnel and the C's stall test on 2026-09-28 (scripts/gpu/back/stop/README.md).
export async function stopFor(m) {
  const net = await loadStop(ruleUrlFor(m));
  if (net.m !== m) throw new Error(`stop rule: the file for ${m} checks was trained on a code of ${net.m}`);
  return net;
}
// ---- the CRC-32 split (DESIGN section 7) ----
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
