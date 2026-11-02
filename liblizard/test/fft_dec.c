// ai: arm64's decoder transform (src/fft.c fft_cols_dec: radix 4, fused multiply-adds) against the format's (fft_cols): at
// ai: n = 384 to 1536 and strips of 32 and 64 columns (pitch 64) on random input, each against a double-precision DFT
// ai: of the same input over 4 columns (the worst error over the output's rms), then both timed, interleaved, the best
// ai: of REPS. Fails where the decoder's error passes twice the format's at any size.
// ai: arm64 (test/neon_kernels.sh); elsewhere fft_cols_dec is fft_cols: fft_dec [reps]
#include "../src/fft.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
static double err_of(const float *re, const float *im, const float *r0, const float *i0, int n, int pitch) {
  double worst = 0, pw = 0;
  for (int x = 0; x < 4; x++) for (int k = 0; k < n; k++) {
    double sr = 0, si = 0;
    for (int t = 0; t < n; t++) { const double a = -2 * 3.14159265358979323846 * (double)((long)k * t % n) / n, c = cos(a), s = sin(a); sr += r0[(size_t)t * pitch + x] * c - i0[(size_t)t * pitch + x] * s; si += r0[(size_t)t * pitch + x] * s + i0[(size_t)t * pitch + x] * c; }
    const double er = re[(size_t)k * pitch + x] - sr, ei = im[(size_t)k * pitch + x] - si, e = sqrt(er * er + ei * ei);
    if (e > worst) worst = e;
    pw += sr * sr + si * si;
  }
  return worst / sqrt(pw / (4.0 * n));
}
int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 30;
  static const int NS[6] = { 384, 768, 1024, 1152, 1280, 1536 }, CW[2] = { 32, 64 };
  uint32_t z = 11;
  int bad = 0;
  for (int ni = 0; ni < 6; ni++) for (int ci = 0; ci < 2; ci++) {
    const int n = NS[ni], cols = CW[ci], pitch = 64; const size_t sz = (size_t)n * pitch * 4;
    float *r0 = malloc(sz), *i0 = malloc(sz), *ra = malloc(sz), *ia = malloc(sz), *rb = malloc(sz), *ib = malloc(sz);
    for (int i = 0; i < n * pitch; i++) { z = z * 1664525u + 1013904223u; r0[i] = (float)(z >> 8) / 16777216.0f - 0.5f; z = z * 1664525u + 1013904223u; i0[i] = (float)(z >> 8) / 16777216.0f - 0.5f; }
    double ta = 1e30, tb = 1e30;
    for (int rep = 0; rep < reps; rep++) {
      memcpy(ra, r0, sz); memcpy(ia, i0, sz); double t0 = now_us(); fft_cols(ra, ia, n, cols, pitch, 0); double t = now_us() - t0; if (t < ta) ta = t;
      memcpy(rb, r0, sz); memcpy(ib, i0, sz); t0 = now_us(); fft_cols_dec(rb, ib, n, cols, pitch); t = now_us() - t0; if (t < tb) tb = t;
    }
    const double ea = err_of(ra, ia, r0, i0, n, pitch), eb = err_of(rb, ib, r0, i0, n, pitch);
    printf("n %4d cols %2d  format %6.1f us  decoder %6.1f us  %.2fx   worst error / rms: format %.2e, decoder %.2e%s\n", n, cols, ta, tb, ta / tb, ea, eb, eb > 2 * ea ? "  TOO LARGE" : "");
    bad += eb > 2 * ea;
    free(r0); free(i0); free(ra); free(ia); free(rb); free(ib);
  }
  return bad != 0;
}
