// The format word as a linear code, for the GPU reader (wgsl/word.mjs). The word is [0x4C, version, fps] and
// ai: bytes - 3 Reed-Solomon parity bytes, one code a ring over every word cell: RS(8,3), (16,3), (24,3), (32,3) at W =
// ai: 16, 32, 48, 64 word cells a side (src/fmt.c, src/rs.c: GF(256), polynomial 0x11D, roots alpha^0 upward, data first).
// Reed-Solomon is linear over GF(2), so every valid word is BASE xor the rows of its version's and its fps's set bits,
// ai: and there are VERSION_MAX x 256 of them a ring (any ring may carry any version since 2026-09-27): the reader
// ai: scores them all against the soft bits and keeps the best, where the reference decodes hard bytes with erasures.
//
// Bit order is the reader's: q index side * W + cell, cell 0..W - 1 along the side; byte k lives on side k % 4 at
// cells 8 (k / 4) .. + 7, most significant bit first (src/fmt.c ob_fmt_bit). Cells past the codeword are light.
import { N_FOR } from "../sim/lizard_pick.mjs";

const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
{ let x = 1; for (let i = 0; i < 255; i++) { EXP[i] = x; LOG[x] = i; x <<= 1; if (x & 256) x ^= 0x11d; } for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255]; }
const mul = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);

export const BYTES_MAX = 36;
// ai: The largest version a word states (src/fmt.h OB_FMT_VERSION_MAX: LIZARD-1024's 128 blocks).
export const VERSION_MAX = 128;
// The codeword's length for W word cells a side (src/fmt.c ob_fmt_bytes).
export const bytesFor = (W) => (W < 16 ? 0 : Math.min(4 * Math.floor(W / 8), BYTES_MAX));

// src/rs.c rs_encode, for the word's 3 data bytes and bytes - 3 roots.
export function wordBytes(magic, version, fps, bytes = 8) {
  const k = 3, nroots = bytes - k, cw = new Uint8Array(bytes), g = new Uint8Array(nroots + 1);
  cw[0] = magic; cw[1] = version; cw[2] = fps;
  g[0] = 1;
  for (let i = 0; i < nroots; i++) { g[i + 1] = 0; for (let j = i + 1; j > 0; j--) g[j] ^= mul(g[j - 1], EXP[i]); }
  const par = new Uint8Array(nroots);
  for (let i = 0; i < k; i++) {
    const fb = cw[i] ^ par[0];
    par.copyWithin(0, 1); par[nroots - 1] = 0;
    if (fb) for (let j = 0; j < nroots; j++) par[j] ^= mul(fb, g[j + 1]);
  }
  cw.set(par, k);
  return cw;
}

// The word's 4W bits in the reader's order, as u32 (q index 0..31 in the first, and so on).
export function wordBits(cw, W) {
  const out = new Uint32Array(Math.ceil((4 * W) / 32));
  for (let side = 0; side < 4; side++) for (let c = 0; c < W; c++) {
    const i = side * W + c, k = 4 * (c >> 3) + side;
    if (k < cw.length && (cw[k] >> (7 - (c & 7))) & 1) out[i >> 5] |= 1 << (i & 31);
  }
  return out;
}

// BASE and the 16 rows (version bits 0..7, then fps bits 0..7) for W word cells a side.
export function wordCode(W) {
  const bytes = bytesFor(W), bits = (v, f, m = 0) => wordBits(wordBytes(m, v, f, bytes), W);
  const base = bits(0, 0, 0x4c), rows = [];
  for (let b = 0; b < 8; b++) rows.push(bits(1 << b, 0));
  for (let b = 0; b < 8; b++) rows.push(bits(0, 1 << b));
  return { W, bytes, base, rows };
}

// ai: The versions whose picture is n (sim/lizard_pick.mjs N_FOR), first and last: the word names the version, and the
// ai: version the picture (gpu/tables.mjs PICTURES).
export function versionRange(n) {
  let lo = 0, hi = -1;
  for (let v = 1; v <= VERSION_MAX; v++) if (N_FOR(8 * v) === n) { if (!lo) lo = v; hi = v; }
  return [lo, hi];
}
