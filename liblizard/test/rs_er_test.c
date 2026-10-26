// Errors-and-erasures decoding: every word within the bound decoded back, and no word (random bytes, any erasures)
// decoded to anything but a codeword (2026-10-25: a locator whose degree fell below its length did).
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
  // ai: Garbage: whatever rs_decode_er accepts must be a codeword (its data bytes re-encode to its parity).
  int accepted = 0;
  for (int t = 0; t < 4000; t++, trials++) {
    int n = 8 + rnd() % 120, nroots = 2 + rnd() % 30; if (nroots >= n - 2) nroots = n - 3;
    uint8_t cw[255], re[255]; int eras[255];
    for (int i = 0; i < n; i++) { cw[i] = (uint8_t)rnd(); eras[i] = i; }
    for (int i = n - 1; i > 0; i--) { int k = rnd() % (i + 1), x = eras[i]; eras[i] = eras[k]; eras[k] = x; }
    const int s = rnd() % (nroots + 1);
    if (rs_decode_er(cw, n, nroots, eras, s) < 0) continue;
    accepted++;
    memcpy(re, cw, (size_t)(n - nroots)); rs_encode(re, n - nroots, nroots);
    if (memcmp(re, cw, (size_t)n)) { if (++fails < 5) printf("FAIL garbage n=%d nroots=%d erasures %d: decoded to a word that is no codeword\n", n, nroots, s); }
  }
  printf("garbage: %d of 4000 random words decoded (each held to be a codeword)\n", accepted);
  printf("%d trials, %d failures\n", trials, fails);
  return fails != 0;
}
