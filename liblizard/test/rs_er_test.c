// Errors-and-erasures decoding and the interleaved (SIMD) syndromes against the plain decoder.
#include "../src/rs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint32_t S = 12345; static uint32_t rnd(void) { S = S * 1664525u + 1013904223u; return S >> 8; }
int main(void) {
  int fails = 0, trials = 0;
  for (int t = 0; t < 4000; t++, trials++) {
    int n = 40 + rnd() % 216, nroots = 2 + 2 * (rnd() % 32); if (nroots >= n - 2) nroots = (n - 4) & ~1;
    uint8_t cw[255], good[255]; for (int i = 0; i < n - nroots; i++) cw[i] = (uint8_t)rnd();
    rs_encode(cw, n - nroots, nroots); memcpy(good, cw, (size_t)n);
    int s = rnd() % (nroots + 1), e = (nroots - s) / 2; if (rnd() & 1) e = rnd() % (e + 1);
    int eras[255], used[255] = { 0 };
    for (int k = 0; k < s; k++) { int p; do p = rnd() % n; while (used[p]); used[p] = 1; eras[k] = p; if (rnd() & 1) cw[p] ^= (uint8_t)(1 + rnd() % 255); }
    for (int k = 0; k < e; k++) { int p; do p = rnd() % n; while (used[p]); used[p] = 1; cw[p] ^= (uint8_t)(1 + rnd() % 255); }
    int r = rs_decode_er(cw, n, nroots, eras, s);
    if (r < 0 || memcmp(cw, good, (size_t)n)) { if (++fails < 5) printf("FAIL n=%d nroots=%d s=%d e=%d r=%d\n", n, nroots, s, e, r); }
  }
  // Interleaved: m codewords, syndromes all at once, decode each.
  for (int t = 0; t < 600; t++, trials++) {
    int m = 1 + rnd() % 40, n = 30 + rnd() % 220, nroots = 2 + 2 * (rnd() % 20); if (nroots >= n - 2) nroots = (n - 4) & ~1;
    uint8_t *buf = malloc((size_t)m * n), *good = malloc((size_t)m * n), *syn = malloc((size_t)m * nroots), cw[255];
    for (int c = 0; c < m; c++) { for (int i = 0; i < n - nroots; i++) cw[i] = (uint8_t)rnd(); rs_encode(cw, n - nroots, nroots); for (int j = 0; j < n; j++) buf[j * m + c] = cw[j]; }
    memcpy(good, buf, (size_t)m * n);
    for (int c = 0; c < m; c++) { int e = rnd() % (nroots / 2 + 1); for (int k = 0; k < e; k++) buf[(rnd() % n) * m + c] ^= (uint8_t)(1 + rnd() % 255); }
    rs_syndromes_il(buf, m, n, nroots, syn);
    int bad = 0;
    for (int c = 0; c < m; c++) if (rs_decode_il(buf, m, n, nroots, c, syn, NULL, 0) < 0) bad = 1;
    if (bad || memcmp(buf, good, (size_t)m * n)) { if (++fails < 5) printf("FAIL interleaved m=%d n=%d nroots=%d\n", m, n, nroots); }
    free(buf); free(good); free(syn);
  }
  printf("%d trials, %d failures\n", trials, fails);
  return fails != 0;
}
