// The bit map (perm, inverse) and the whitening (PRBS-23), regenerated in JS from src/focus.c and src/ldpc.c (nothing
// here is exported by the wasm); the disc layout of S and the UV gather table the soft values read; and a block's
// systematic codeword and data bits (the C's encoder), which the profile stage's checks paint with.

// x^23 + x^18 + 1 from the all-ones state, one bit a slot of the frame in pos order (focus.c whiten_fill).
export function whiten(count) {
  const w = new Uint8Array(count);
  let s = 0x7fffff;
  for (let i = 0; i < count; i++) { const b = ((s >>> 22) ^ (s >>> 17)) & 1; s = ((s << 1) | b) & 0x7fffff; w[i] = b; }
  return w;
}

const gcd = (a, b) => { while (b) { const t = a % b; a = b; b = t; } return a; };

// Slot i of a block carries sent bit perm[i], codeword bit np + perm[i] (focus.c perm_fill; np the bits a code never
// sends, 93 at 7/8 since 2026-10-08, 0 elsewhere): n is the sent length nt, k the sent data bits. Modes are
// src/focus.h FOCUS_BITMAP_*: 0 none and 1 white leave the order alone; 2 spread, 3 linear (the format's), 4 inner.
export function perm(n, k, mode) {
  const p = new Int32Array(n), m = n - k;
  if (mode === 2) {
    let dc = 0, pc = 0;
    for (let i = 0; i < n; i++) p[i] = Math.floor((i + 1) * m / n) > Math.floor(i * m / n) ? k + pc++ : dc++;
  } else if (mode === 3) {
    let P = Math.floor(0.3819660113 * n + 0.5);
    while (gcd(P, n) !== 1) P++;
    for (let i = 0; i < n; i++) p[i] = (i * P) % n;
  } else if (mode === 4) {
    for (let i = 0; i < n; i++) p[i] = i < m ? k + i : i - m;
  } else for (let i = 0; i < n; i++) p[i] = i;
  return p;
}

export function inverse(p) {
  const inv = new Int32Array(p.length);
  for (let i = 0; i < p.length; i++) inv[p[i]] = i;
  return inv;
}

// pos entry p to (u, v) (focus.c init: pos = (v / bw) n bw + ((u + n) % n) bw + v % bw).
export function decodePos(pos, n, bw) {
  const u = new Int32Array(pos.length), v = new Int32Array(pos.length);
  for (let i = 0; i < pos.length; i++) {
    const p = pos[i], vb = Math.floor(p / (n * bw)), rem = p % (n * bw), uw = Math.floor(rem / bw);
    v[i] = vb * bw + rem % bw;
    u[i] = uw < n / 2 ? uw : uw - n;
  }
  return { u, v };
}

// S's layout (DESIGN.md section 3): row v at off[v]; v = 0 holds u = 1..umax[0], v > 0 holds u = 0..umax[v] then
// u = -umax[v]..-1. Built from pos itself, so it is whatever the C's disc is; every pos entry maps to exactly one
// entry of S and the count is npos (checked here, every format: the disc has no holes and is symmetric in u).
// uv[i] is the S index of pos[i]: what part B gathers through.
export function discLayout(shape, pos) {
  const { n, bw, npos } = shape, { u, v } = decodePos(pos, n, bw);
  let vmax = 0;
  for (let i = 0; i < npos; i++) if (v[i] > vmax) vmax = v[i];
  const V = vmax + 1, umax = new Int32Array(V).fill(-1), uneg = new Int32Array(V);
  for (let i = 0; i < npos; i++) {
    if (v[i] < 0 || (v[i] === 0 && u[i] <= 0)) throw new Error(`pos ${i}: (u ${u[i]}, v ${v[i]}) outside the half disc`);
    if (u[i] >= 0) umax[v[i]] = Math.max(umax[v[i]], u[i]); else uneg[v[i]] = Math.max(uneg[v[i]], -u[i]);
  }
  const off = new Uint32Array(V + 1);
  for (let r = 0; r < V; r++) {
    if (r > 0 && umax[r] !== uneg[r]) throw new Error(`row v ${r}: u runs to ${umax[r]} and -${uneg[r]}`);
    off[r + 1] = off[r] + (r === 0 ? umax[0] : umax[r] + 1 + uneg[r]);
  }
  if (off[V] !== npos) throw new Error(`disc holds ${off[V]} entries, pos ${npos}`);
  const uv = new Uint32Array(npos), seen = new Uint8Array(npos);
  for (let i = 0; i < npos; i++) {
    const r = v[i], e = off[r] + (u[i] >= 0 ? (r === 0 ? u[i] - 1 : u[i]) : umax[r] + 1 + u[i] + uneg[r]);
    if (seen[e]) throw new Error(`S entry ${e} claimed twice`);
    seen[e] = 1; uv[i] = e;
  }
  return { V, off, umax, uv, count: npos, u, v };
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
