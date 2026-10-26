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

// ---------------------------------------------------------------- errors and erasures

// The work after the syndromes: Berlekamp-Massey started from the erasure locator (so what it finds is the locator of
// errors and erasures together), Chien search, Forney. Generator roots start at alpha^0, which is where the factor X in
// the value comes from.
// ai: A word is corrected only where the locator's degree is the register's length el (the errors and erasures it
// ai: claims) and the Chien search finds that many roots: a locator whose degree fell below el (BM's last step cancelled
// ai: its top terms) claims a correction it does not make, and the word it leaves is no codeword. Until 2026-10-25 the
// ai: test was against the degree alone, so such a word came back as decoded with its syndromes nonzero: rs_decode's
// ai: own test (found against L) is the same rule.
static int solve(uint8_t *cw, int n, int nroots, const uint8_t *S, const int *eras, int neras) {
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
  if (deg != el || 2 * (el - neras) + neras > nroots) return -1;
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
    cw[j] ^= mul(mul(num, inv(den)), EXP[(n - 1 - j) % 255]);
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
  return solve(cw, n, nroots, S, eras, neras);
}
