// Block-row schedule against the serial one on identical noise: frame errors, undetected errors,
// iterations, time. The schedules differ in how fast the staircase carries information, so this
// is the test that says whether the vectorizable one costs anything.
//   ldpc_sched [n] [frames]
#include "../src/ldpc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t S = 88172645463325252ull;
static double uni(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (double)(S >> 11) / 9007199254740992.0; }
static double gauss(void) { double u = 1 - uni(), v = uni(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 10000, frames = argc > 2 ? atoi(argv[2]) : 300;
  // Per rate: noise levels from comfortable to the edge of the waterfall.
  static const double SIG[LDPC_RATES][3] = { { 1.30, 1.42, 1.48 }, { 1.10, 1.20, 1.25 }, { 0.85, 0.91, 0.94 }, { 0.68, 0.73, 0.75 }, { 0.60, 0.645, 0.665 }, { 0.52, 0.56, 0.575 }, { 0.48, 0.515, 0.53 } };
  printf("n<=%d, %d frames per point, 30 iterations max\nrate sigma |  serial: FER  undet  iters  us/frame | block-row: FER  undet  iters  us/frame\n", n, frames);
  for (int rate = 2; rate < LDPC_RATES; rate++) {
    ldpc_t c;
    if (ldpc_init(&c, n, rate, 1)) return 1;
    uint8_t *d = malloc(c.k), *cw = malloc(c.n), *out = malloc(c.n);
    int8_t *llr = malloc((size_t)c.n * frames);
    uint8_t *cws = malloc((size_t)c.n * frames);
    for (int p = 0; p < 3; p++) {
      double sigma = SIG[rate][p];
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < c.k; i++) d[i] = uni() < 0.5;
        ldpc_encode(&c, d, cw);
        memcpy(cws + (size_t)f * c.n, cw, c.n);
        for (int i = 0; i < c.n; i++) { double y = (cw[i] ? -1 : 1) + sigma * gauss(), L = 2 * y / (sigma * sigma) * 8; llr[(size_t)f * c.n + i] = (int8_t)(L > 127 ? 127 : L < -127 ? -127 : lrint(L)); }
      }
      printf("%-4s %.3f |", ldpc_rate_name[rate], sigma);
      for (int sched = 0; sched < 2; sched++) {
        int fail = 0, undet = 0; double its = 0; clock_t t0 = clock();
        for (int f = 0; f < frames; f++) {
          int it = (sched ? ldpc_decode : ldpc_decode_serial)(&c, llr + (size_t)f * c.n, out, 30), wrong = memcmp(out, cws + (size_t)f * c.n, c.k) != 0;
          if (it < 0 || wrong) fail++;
          if (it >= 0 && wrong) undet++;
          its += it < 0 ? 30 : it;
        }
        printf("  %10.3f %6d %6.1f %9.0f %s", (double)fail / frames, undet, its / frames, 1e6 * (clock() - t0) / CLOCKS_PER_SEC / frames, sched ? "\n" : "|");
      }
    }
    free(d); free(cw); free(out); free(llr); free(cws); ldpc_free(&c);
  }
  return 0;
}
