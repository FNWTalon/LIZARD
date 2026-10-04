// The transform against a naive DFT (small sizes, which also covers a block that is a slice of a wider array),
// against the textbook radix-2 FFT it replaced (512 x 512), a round trip, and the Hermitian symmetry of a real input.
#include "../src/fft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void old_fft1d(float *re, float *im, int n, int stride, int inverse) {
  for (int i = 1, j = 0; i < n; i++) { int bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { float t = re[i * stride]; re[i * stride] = re[j * stride]; re[j * stride] = t; t = im[i * stride]; im[i * stride] = im[j * stride]; im[j * stride] = t; } }
  for (int len = 2; len <= n; len <<= 1) {
    double ang = (inverse ? 2 : -2) * 3.14159265358979323846 / len; float wr = (float)cos(ang), wi = (float)sin(ang);
    for (int i = 0; i < n; i += len) { float cr = 1, ci = 0; for (int k = 0; k < len / 2; k++) { int a = (i + k) * stride, b = (i + k + len / 2) * stride; float xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; float t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
static void old_fft2d(float *re, float *im, int n, int inverse) {
  for (int y = 0; y < n; y++) old_fft1d(re + y * n, im + y * n, n, 1, inverse);
  for (int x = 0; x < n; x++) old_fft1d(re + x, im + x, n, n, inverse);
  if (inverse) { float s = 1.0f / ((float)n * n); for (int i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; } }
}

int main(void) {
  int fails = 0;
  srand(3);
  // fft_cols against the definition, on a slice: rows x cols inside an array whose rows are `pitch` apart, for every
  // radix fft_set_radix takes (archive/stash/fft-gen2 has four more). The error allowed grows with log n, which is how a
  // float transform's error grows.
  const int radices[] = { 2, 4 };
  for (int ri = 0; ri < 2; ri++) {
  fft_set_radix(radices[ri]);
  if (fft_get_radix() != radices[ri]) { fails++; printf("FAIL radix %d not taken\n", radices[ri]); continue; }
  double worstAll = 0;
  // ai: The powers of two to 1024, then 3 x 2^k to 1536, which take a radix-3 stage.
  for (int three = 0; three < 2; three++)
  for (int rows = three ? 3 : 2; rows <= (three ? 1536 : 1024); rows <<= 1) for (int cols = 1; cols <= 9; cols += 4) for (int inverse = 0; inverse < 2; inverse++) {
    int pitch = cols + 3; float *re = malloc(sizeof(float) * rows * pitch), *im = malloc(sizeof(float) * rows * pitch), *r0 = malloc(sizeof(float) * rows * pitch), *i0 = malloc(sizeof(float) * rows * pitch);
    for (int i = 0; i < rows * pitch; i++) { r0[i] = re[i] = (float)rand() / RAND_MAX - 0.5f; i0[i] = im[i] = (float)rand() / RAND_MAX - 0.5f; }
    fft_cols(re, im, rows, cols, pitch, inverse);
    double worst = 0;
    for (int x = 0; x < pitch; x++) for (int k = 0; k < rows; k++) {
      double sr = 0, si = 0;
      if (x < cols) for (int t = 0; t < rows; t++) { double a = (inverse ? 2 : -2) * 3.14159265358979323846 * k * t / rows; sr += r0[t * pitch + x] * cos(a) - i0[t * pitch + x] * sin(a); si += r0[t * pitch + x] * sin(a) + i0[t * pitch + x] * cos(a); }
      else { sr = r0[k * pitch + x]; si = i0[k * pitch + x]; }   // outside the block: untouched
      double e = fabs(sr - re[k * pitch + x]) + fabs(si - im[k * pitch + x]); if (e > worst) worst = e;
    }
    double bits = 0; for (int t = rows; t > 1; t >>= 1) bits++;
    if (worst > 2e-5 * bits * sqrt((double)rows)) { fails++; printf("FAIL fft_cols radix %d rows %d cols %d inverse %d: %.2e\n", radices[ri], rows, cols, inverse, worst); }
    if (worst / (bits * sqrt((double)rows)) > worstAll) worstAll = worst / (bits * sqrt((double)rows));
    free(re); free(im); free(r0); free(i0);
  }
  printf("radix %2d: sizes 2 to 1024 and 3 to 1536, forward and inverse, against the definition: worst error %.2e x log2(n) sqrt(n)\n", radices[ri], worstAll);
  }
  fft_set_radix(2);
  // 512 x 512 against the transform it replaced, then a round trip and the symmetry of a real input.
  int n = 512; size_t nn = (size_t)n * n;
  float *re = malloc(sizeof(float) * nn), *im = calloc(nn, sizeof(float)), *ore = malloc(sizeof(float) * nn), *oim = calloc(nn, sizeof(float)), *ref = malloc(sizeof(float) * nn);
  for (size_t i = 0; i < nn; i++) ref[i] = ore[i] = re[i] = (float)rand() / RAND_MAX - 0.5f;
  fft2d(re, im, n, 0); old_fft2d(ore, oim, n, 0);
  float diff = 0, herm = 0, big = 0;
  for (size_t i = 0; i < nn; i++) { diff = fmaxf(diff, fabsf(re[i] - ore[i]) + fabsf(im[i] - oim[i])); big = fmaxf(big, fabsf(ore[i])); }
  for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) { int j = ((n - y) % n) * n + (n - x) % n, i = y * n + x; herm = fmaxf(herm, fabsf(re[i] - re[j]) + fabsf(im[i] + im[j])); }
  fft2d(re, im, n, 1);
  float err = 0;
  for (size_t i = 0; i < nn; i++) err = fmaxf(err, fabsf(re[i] - ref[i]) + fabsf(im[i]));
  printf("512 x 512: against the old transform %.2e (largest coefficient %.0f), round trip %.2e, hermitian residue %.2e\n", diff, big, err, herm);
  if (!(diff < 0.05f && err < 1e-5f && herm < 0.05f)) fails++;
  free(re); free(im); free(ore); free(oim); free(ref);
  // ai: A 768 x 768 round trip and the symmetry of a real input: fft2d's 64-column slices on a size with a factor of 3.
  n = 768; nn = (size_t)n * n;
  re = malloc(sizeof(float) * nn); im = calloc(nn, sizeof(float)); ref = malloc(sizeof(float) * nn);
  for (size_t i = 0; i < nn; i++) ref[i] = re[i] = (float)rand() / RAND_MAX - 0.5f;
  fft2d(re, im, n, 0);
  herm = 0;
  for (int y = 0; y < n; y++) for (int x = 0; x < n; x++) { int j = ((n - y) % n) * n + (n - x) % n, i = y * n + x; herm = fmaxf(herm, fabsf(re[i] - re[j]) + fabsf(im[i] + im[j])); }
  fft2d(re, im, n, 1);
  err = 0;
  for (size_t i = 0; i < nn; i++) err = fmaxf(err, fabsf(re[i] - ref[i]) + fabsf(im[i]));
  printf("768 x 768: round trip %.2e, hermitian residue %.2e\n", err, herm);
  if (!(err < 1e-5f && herm < 0.05f)) fails++;
  free(re); free(im); free(ref);
  printf("%d failures\n", fails);
  return fails != 0;
}
