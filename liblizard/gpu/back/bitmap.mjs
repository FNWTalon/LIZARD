// The bit map (perm, inverse) and the whitening (PRBS-23), regenerated in JS from src/focus.c and src/ldpc.c (nothing
// here is exported by the wasm); and a block's systematic codeword and data bits (the C's encoder), which the profile
// stage's checks paint with.

// x^23 + x^18 + 1 from the all-ones state, one bit a slot of the frame in pos order (focus.c whiten_fill).
export function whiten(count) {
  const w = new Uint8Array(count);
  let s = 0x7fffff;
  for (let i = 0; i < count; i++) { const b = ((s >>> 22) ^ (s >>> 17)) & 1; s = ((s << 1) | b) & 0x7fffff; w[i] = b; }
  return w;
}

const gcd = (a, b) => { while (b) { const t = a % b; a = b; b = t; } return a; };

// Slot i of a block carries sent bit perm[i], codeword bit np + perm[i] (focus.c perm_fill; np the bits a code never
// sends, 93 at 7/8 since 2026-10-08, 0 elsewhere): n is the sent length nt. The format's one bit map, LINEAR (src/focus.h
// FOCUS_BITMAP): slot i takes bit i P mod n, P the integer nearest n over the golden ratio squared, stepped up until
// prime to n.
export function perm(n) {
  const p = new Int32Array(n);
  let P = Math.floor(0.3819660113 * n + 0.5);
  while (gcd(P, n) !== 1) P++;
  for (let i = 0; i < n; i++) p[i] = (i * P) % n;
  return p;
}

export function inverse(p) {
  const inv = new Int32Array(p.length);
  for (let i = 0; i < p.length; i++) inv[p[i]] = i;
  return inv;
}

// CRC-32, reflected 0xEDB88320, init and final xor 0xffffffff (src/layout.c ob_crc32).
export function crc32(bytes, len = bytes.length) {
  let c = 0xffffffff;
  for (let i = 0; i < len; i++) { c ^= bytes[i]; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); }
  return (c ^ 0xffffffff) >>> 0;
}

// The systematic codeword of one block from the C's own tables (ldpc.c ldpc_encode through the block-row layout
// ldpc_tables gives): check chk = 12 i + r reads data slot e of row r as bit col_e z + (i - shift_e + z) mod z,
// its own parity bit k + chk, and parity bit k + chk - 1 (absent at chk = 0). lay: [mb + 1] row pointers, then
// (col, shift) pairs.
export function encodeCodeword(code, data) {
  const { n, k, m, z, mb, lay } = code, cw = new Uint8Array(n);
  cw.set(data.subarray(0, k));
  let acc = 0;
  for (let chk = 0; chk < m; chk++) {
    const r = chk % mb, i = Math.floor(chk / mb);
    for (let e = lay[r]; e < lay[r + 1]; e++) {
      const col = lay[mb + 1 + 2 * e], s = lay[mb + 2 + 2 * e];
      acc ^= cw[col * z + (i - s + z) % z];
    }
    cw[k + chk] = acc;
  }
  return cw;
}

// A block's 473 payload bytes (id first) to its k data bits: bytes MSB first, then the CRC-32 MSB first.
export function dataBits(payload, k) {
  const bits = new Uint8Array(k), B = payload.length;
  for (let i = 0; i < B * 8; i++) bits[i] = (payload[i >> 3] >> (7 - (i & 7))) & 1;
  const c = crc32(payload);
  for (let i = 0; i < 32; i++) bits[B * 8 + i] = (c >>> (31 - i)) & 1;
  return bits;
}
