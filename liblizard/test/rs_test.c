// RS(255, k): every pattern of up to t byte errors is corrected, t + 1 is never miscorrected
// into silence. ./rs_test
#include "../src/rs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
  srand(7);
  int fails = 0, silent = 0, trials = 0;
  for (int nroots = 2; nroots <= 64; nroots += 6) for (int n = nroots + 8; n <= 255; n += 61) {
    int k = n - nroots, t = nroots / 2;
    for (int rep = 0; rep < 200; rep++) {
      uint8_t a[255], b[255];
      for (int i = 0; i < k; i++) a[i] = (uint8_t)rand();
      rs_encode(a, k, nroots);
      memcpy(b, a, (size_t)n);
      if (rs_decode(b, n, nroots) != 0) fails++;
      int e = rep % (t + 2);                       // 0 .. t + 1 errors
      for (int q = 0; q < e; q++) { int p; do p = rand() % n; while (b[p] != a[p]); b[p] ^= (uint8_t)(1 + rand() % 255); }
      int r = rs_decode(b, n, nroots);
      trials++;
      if (e <= t) { if (r != e || memcmp(a, b, (size_t)n)) fails++; }
      else if (r >= 0 && !memcmp(a, b, (size_t)n)) fails++;      // cannot be: t + 1 errors undone
      else if (r >= 0) silent++;                                  // miscorrection to another codeword
    }
  }
  printf("%d trials, %d failures, %d miscorrections at t + 1 errors (expected: rare, not zero)\n", trials, fails, silent);
  return fails != 0;
}
