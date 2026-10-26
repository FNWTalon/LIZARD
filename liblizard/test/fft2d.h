// ai: n x n complex samples in place, rows then columns, over src/fft.c fft_cols: inverse = 1 divides by n * n, so
// ai: forward then inverse is the identity. The tests' (fft_test.c, neon_check.c): the codec only ever needs half the
// ai: plane and uses fft_cols directly (src/focus.c). In src/fft.c until 2026-10-25.
#ifndef OB_TEST_FFT2D_H
#define OB_TEST_FFT2D_H
#include "../src/fft.h"

static void fft2d(float *re, float *im, int n, int inverse) {
  // Along y, turn the plane over, along y again (which was x), turn it back. Sixty-four columns at a time.
  for (int pass = 0; pass < 2; pass++) {
    for (int x0 = 0; x0 < n; x0 += 64) fft_cols(re + x0, im + x0, n, n - x0 < 64 ? n - x0 : 64, n, inverse);
    for (int y = 0; y < n; y++) for (int x = y + 1; x < n; x++) { float t = re[y * n + x]; re[y * n + x] = re[x * n + y]; re[x * n + y] = t; t = im[y * n + x]; im[y * n + x] = im[x * n + y]; im[x * n + y] = t; }
  }
  if (inverse) { float s = 1.0f / ((float)n * n); for (int i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; } }
}
#endif
