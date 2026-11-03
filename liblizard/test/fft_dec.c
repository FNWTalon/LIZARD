// ai: arm64's decoder transforms (src/fft.c fft_dec_dif and fft_dec_dit: their own kernels, fused multiply-adds; on x86-64
// ai: the same kernels in SSE4.1, -msse4.1) against
// ai: the format's (fft_cols): at every picture size from 384 to 1536 and strips of 30, 32 and 64 columns (pitch 64) on
// ai: random input, each against a double-precision DFT of the same input over 4 columns (two first, two last: the
// ai: vector loops and the tail), the worst error over the output's rms. Rows are read and written through
// ai: fft_dec_place: dif's output and dit's input are in its order; dit is also checked with keep n / 3 on the rows it
// ai: then writes, [0, keep] and [n - keep, n). Then each timed against fft_cols, interleaved, the best of REPS. Fails
// ai: where an error passes twice the format's.
// ai: arm64 (test/neon_kernels.sh) and x86-64 with SSE4.1; elsewhere (and -DOB_SCALAR) both are fft_cols and place NULL:
// ai: fft_dec [reps]
#include "../src/fft.c"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
// ai: the DFT of the input (sample t in row pin[t]) against the output (frequency k in row pout[k]), k within keep
static double err_of(const float *re, const float *im, const float *r0, const float *i0, int n, int cols, int pitch, const int *pin, const int *pout, int keep) {
  double worst = 0, pw = 0;
  int cnt = 0;
  const int xs[4] = { 0, 1, cols - 2, cols - 1 };
  for (int q = 0; q < 4; q++) for (int k = 0; k < n; k++) {
    if (!(k <= keep || k >= n - keep)) continue;
    const int x = xs[q];
    double sr = 0, si = 0;
    for (int t = 0; t < n; t++) {
      const size_t at = (size_t)(pin ? pin[t] : t) * pitch + x;
      const double a = -2 * 3.14159265358979323846 * (double)((long)k * t % n) / n, c = cos(a), s = sin(a);
      sr += r0[at] * c - i0[at] * s; si += r0[at] * s + i0[at] * c;
    }
    const size_t o = (size_t)(pout ? pout[k] : k) * pitch + x;
    const double er = re[o] - sr, ei = im[o] - si, e = sqrt(er * er + ei * ei);
    if (e > worst) worst = e;
    pw += sr * sr + si * si; cnt++;
  }
  return worst / sqrt(pw / cnt);
}
int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 30;
  static const int NS[7] = { 384, 512, 768, 1024, 1152, 1280, 1536 }, CW[3] = { 30, 32, 64 };
  uint32_t z = 11;
  int bad = 0;
  for (int ni = 0; ni < 7; ni++) for (int ci = 0; ci < 3; ci++) {
    const int n = NS[ni], cols = CW[ci], pitch = 64, keep = n / 3; const size_t sz = (size_t)n * pitch * 4;
    float *r0 = malloc(sz), *i0 = malloc(sz), *ra = malloc(sz), *ia = malloc(sz), *rb = malloc(sz), *ib = malloc(sz), *rc = malloc(sz), *ic = malloc(sz);
    for (int i = 0; i < n * pitch; i++) { z = z * 1664525u + 1013904223u; r0[i] = (float)(z >> 8) / 16777216.0f - 0.5f; z = z * 1664525u + 1013904223u; i0[i] = (float)(z >> 8) / 16777216.0f - 0.5f; }
    double ta = 1e30, tb = 1e30, tc = 1e30;
    for (int rep = 0; rep < reps; rep++) {
      memcpy(ra, r0, sz); memcpy(ia, i0, sz); double t0 = now_us(); fft_cols(ra, ia, n, cols, pitch, 0); double t = now_us() - t0; if (t < ta) ta = t;
      memcpy(rb, r0, sz); memcpy(ib, i0, sz); t0 = now_us(); fft_dec_dif(rb, ib, n, cols, pitch); t = now_us() - t0; if (t < tb) tb = t;
      memcpy(rc, r0, sz); memcpy(ic, i0, sz); t0 = now_us(); fft_dec_dit(rc, ic, n, cols, pitch, n); t = now_us() - t0; if (t < tc) tc = t;
    }
    // ai: place after the decoder's last transform: fft_cols' plan of the same size may have taken the decoder's slot
    // ai: (seven sizes, two plans each, eight slots), freeing the place taken before
    const int *place = fft_dec_place(n);
    const double ea = err_of(ra, ia, r0, i0, n, cols, pitch, 0, 0, n), eb = err_of(rb, ib, r0, i0, n, cols, pitch, 0, place, n);
    const double ec = err_of(rc, ic, r0, i0, n, cols, pitch, place, 0, n);
    // ai: keep: the rows past it hold what the stage before left; those within it are checked
    memcpy(rc, r0, sz); memcpy(ic, i0, sz);
    fft_dec_dit(rc, ic, n, cols, pitch, keep);
    const double ek = err_of(rc, ic, r0, i0, n, cols, pitch, place, 0, keep);
    const int fail = eb > 2 * ea || ec > 2 * ea || ek > 2 * ea;
    printf("n %4d cols %2d  format %6.1f us  dif %6.1f (%.2fx)  dit %6.1f (%.2fx)   worst error / rms: format %.2e, dif %.2e, dit %.2e, dit keep %d %.2e%s\n",
      n, cols, ta, tb, ta / tb, tc, ta / tc, ea, eb, ec, keep, ek, fail ? "  FAIL" : "");
    bad += fail;
    free(r0); free(i0); free(ra); free(ia); free(rb); free(ib); free(rc); free(ic);
  }
  return bad != 0;
}
