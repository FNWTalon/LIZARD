#include "rs.h"
#include <string.h>

// ai: each thread's own, built on its first call: no thread reads a table another is still filling
static _Thread_local uint8_t EXP[512], LOG[256];
static _Thread_local int ready;

static void init(void) {
  int x = 1;
  for (int i = 0; i < 255; i++) { EXP[i] = (uint8_t)x; LOG[x] = (uint8_t)i; x <<= 1; if (x & 256) x ^= 0x11d; }
  for (int i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
  ready = 1;
}
static inline uint8_t mul(uint8_t a, uint8_t b) { return a && b ? EXP[LOG[a] + LOG[b]] : 0; }
static inline uint8_t inv(uint8_t a) { return EXP[255 - LOG[a]]; }

void rs_encode(uint8_t *cw, int k, int nroots) {
  if (!ready) init();
  uint8_t g[256] = { 1 };
  for (int i = 0; i < nroots; i++) {
    // g(x) *= (x - alpha^i), coefficients highest degree first.
    g[i + 1] = 0;
    for (int j = i + 1; j > 0; j--) g[j] = g[j] ^ mul(g[j - 1], EXP[i]);
  }
  uint8_t *par = cw + k;
  memset(par, 0, (size_t)nroots);
  for (int i = 0; i < k; i++) {
    uint8_t fb = cw[i] ^ par[0];
    memmove(par, par + 1, (size_t)nroots - 1);
    par[nroots - 1] = 0;
    if (fb) for (int j = 0; j < nroots; j++) par[j] ^= mul(fb, g[j + 1]);
  }
}

int rs_decode(uint8_t *cw, int n, int nroots) {
  if (!ready) init();
  uint8_t S[256], any = 0;
  for (int i = 0; i < nroots; i++) {
    uint8_t s = 0;
    for (int j = 0; j < n; j++) s = mul(s, EXP[i]) ^ cw[j];
    S[i] = s; any |= s;
  }
  if (!any) return 0;
  // Berlekamp-Massey: the error locator, lowest degree first.
  uint8_t C[256] = { 1 }, B[256] = { 1 }, T[256];
  int L = 0, m = 1;
  uint8_t b = 1;
  for (int r = 0; r < nroots; r++) {
    uint8_t d = S[r];
    for (int i = 1; i <= L; i++) d ^= mul(C[i], S[r - i]);
    if (!d) { m++; continue; }
    memcpy(T, C, sizeof T);
    uint8_t f = mul(d, inv(b));
    for (int i = 0; i + m < 256; i++) C[i + m] ^= mul(f, B[i]);
    if (2 * L <= r) { L = r + 1 - L; memcpy(B, T, sizeof B); b = d; m = 1; } else m++;
  }
  if (L > nroots / 2) return -1;
  // Chien search. Position j (from the front) has locator alpha^(n - 1 - j).
  int pos[128], found = 0;
  for (int j = 0; j < n; j++) {
    int e = (255 - (n - 1 - j)) % 255;   // log of X^-1
    uint8_t v = 0;
    for (int i = L; i >= 0; i--) v = mul(v, EXP[e]) ^ C[i];
    if (!v) { if (found == 128) return -1; pos[found++] = j; }
  }
  if (found != L) return -1;
  // Forney: omega = S * C mod x^nroots; e = X * omega(X^-1) / C'(X^-1) for roots from alpha^0.
  uint8_t W[256] = { 0 };
  for (int i = 0; i < nroots; i++) for (int j = 0; j <= i && j <= L; j++) W[i] ^= mul(S[i - j], C[j]);
  for (int q = 0; q < found; q++) {
    int j = pos[q], e = (255 - (n - 1 - j)) % 255;
    uint8_t xi = EXP[e], num = 0, den = 0, p = 1;
    for (int i = 0; i < nroots; i++) { num ^= mul(W[i], p); p = mul(p, xi); }
    p = 1;
    for (int i = 1; i <= L; i += 2) { den ^= mul(C[i], p); p = mul(p, mul(xi, xi)); }
    if (!den) return -1;
    cw[j] ^= mul(mul(num, inv(den)), EXP[(n - 1 - j) % 255]);
  }
  return found;
}

// ---------------------------------------------------------------- errors and erasures, interleaved blocks

// The work after the syndromes, on a codeword whose byte at position j is cw[j * stride]. Berlekamp-Massey
// started from the erasure locator (so what it finds is the locator of errors and erasures together),
// Chien search, Forney. Generator roots start at alpha^0, which is where the factor X in the value comes from.
static int solve(uint8_t *cw, int stride, int n, int nroots, const uint8_t *S, const int *eras, int neras) {
  if (neras > nroots) return -1;
  uint8_t lam[257] = { 1 }, b[257], t[257];
  for (int k = 0; k < neras; k++) {
    uint8_t X = EXP[(n - 1 - eras[k]) % 255];
    for (int i = k + 1; i > 0; i--) lam[i] ^= mul(lam[i - 1], X);
  }
  memcpy(b, lam, sizeof b);
  int el = neras;
  for (int r = neras + 1; r <= nroots; r++) {
    uint8_t d = 0;
    for (int i = 0; i < r && i <= el; i++) d ^= mul(lam[i], S[r - i - 1]);
    if (!d) { memmove(b + 1, b, 256); b[0] = 0; continue; }
    t[0] = lam[0];
    for (int i = 0; i < 256; i++) t[i + 1] = lam[i + 1] ^ mul(d, b[i]);
    if (2 * el <= r + neras - 1) { el = r + neras - el; uint8_t di = inv(d); for (int i = 0; i < 257; i++) b[i] = mul(lam[i], di); }
    else { memmove(b + 1, b, 256); b[0] = 0; }
    memcpy(lam, t, sizeof lam);
  }
  int deg = 0;
  for (int i = 0; i <= nroots; i++) if (lam[i]) deg = i;
  if (deg > nroots || 2 * (deg - neras) + neras > nroots) return -1;
  int pos[256], found = 0;
  for (int j = 0; j < n; j++) {
    int e = (255 - (n - 1 - j)) % 255;
    uint8_t v = 0;
    for (int i = deg; i >= 0; i--) v = mul(v, EXP[e]) ^ lam[i];
    if (!v) pos[found++] = j;
  }
  if (found != deg) return -1;
  uint8_t W[256] = { 0 };
  for (int i = 0; i < nroots; i++) for (int j = 0; j <= i && j <= deg; j++) W[i] ^= mul(S[i - j], lam[j]);
  for (int q = 0; q < found; q++) {
    int j = pos[q], e = (255 - (n - 1 - j)) % 255;
    uint8_t xi = EXP[e], num = 0, den = 0, p = 1;
    for (int i = 0; i < nroots; i++) { num ^= mul(W[i], p); p = mul(p, xi); }
    p = 1;
    for (int i = 1; i <= deg; i += 2) { den ^= mul(lam[i], p); p = mul(p, mul(xi, xi)); }
    if (!den) return -1;
    cw[j * stride] ^= mul(mul(num, inv(den)), EXP[(n - 1 - j) % 255]);
  }
  return found;
}

int rs_decode_er(uint8_t *cw, int n, int nroots, const int *eras, int neras) {
  if (!ready) init();
  uint8_t S[256], any = 0;
  for (int i = 0; i < nroots; i++) {
    uint8_t s = 0;
    for (int j = 0; j < n; j++) s = mul(s, EXP[i]) ^ cw[j];
    S[i] = s; any |= s;
  }
  if (!any && !neras) return 0;
  return solve(cw, 1, n, nroots, S, eras, neras);
}

#include "simd.h"

int rs_syndromes_il(const uint8_t *buf, int m, int n, int nroots, uint8_t *S) {
  if (!ready) init();
  uint8_t any = 0;
  int c0 = 0;
#ifdef OB_SIMD
  // Sixteen codewords a pass. The last group may be short: its spare lanes read the next codewords' bytes
  // (or, near the end of the buffer, a copy padded with zeros) and are thrown away.
  const v128_t nib = wasm_i8x16_splat(15);
  for (; c0 < m; c0 += 16) {
    int lanes = m - c0 < 16 ? m - c0 : 16;
    for (int i = 0; i < nroots; i++) {
      uint8_t lo[16], hi[16];
      for (int x = 0; x < 16; x++) { lo[x] = mul((uint8_t)x, EXP[i]); hi[x] = mul((uint8_t)(x << 4), EXP[i]); }
      const v128_t tlo = wasm_v128_load(lo), thi = wasm_v128_load(hi);
      v128_t s = wasm_i8x16_splat(0);
      for (int j = 0; j < n; j++) {
        v128_t r;
        if (j * m + c0 + 16 <= m * n) r = wasm_v128_load(buf + j * m + c0);   // sixteen bytes from here are all inside the buffer
        else { uint8_t tail[16] = { 0 }; memcpy(tail, buf + j * m + c0, (size_t)lanes); r = wasm_v128_load(tail); }
        s = wasm_v128_xor(wasm_v128_xor(wasm_i8x16_swizzle(tlo, wasm_v128_and(s, nib)), wasm_i8x16_swizzle(thi, wasm_u8x16_shr(s, 4))), r);
      }
      uint8_t out[16];
      wasm_v128_store(out, s);
      for (int c = 0; c < lanes; c++) { S[(c0 + c) * nroots + i] = out[c]; any |= out[c]; }
    }
  }
#else
  for (; c0 < m; c0++) for (int i = 0; i < nroots; i++) {
    uint8_t s = 0;
    for (int j = 0; j < n; j++) s = mul(s, EXP[i]) ^ buf[j * m + c0];
    S[c0 * nroots + i] = s; any |= s;
  }
#endif
  return any != 0;
}

int rs_decode_il(uint8_t *buf, int m, int n, int nroots, int c, const uint8_t *S, const int *eras, int neras) {
  if (!ready) init();
  const uint8_t *s = S + c * nroots;
  uint8_t any = 0;
  for (int i = 0; i < nroots; i++) any |= s[i];
  if (!any && !neras) return 0;
  return solve(buf + c, m, n, nroots, s, eras, neras);
}
