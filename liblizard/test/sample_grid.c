// ai: The decode's picture sampler (src/acquire.c ob_sample_grid_pic) against the exact walk (ob_sample_grid) on
// ai: synthetic registrations: a random frame, a code's homography with a little perspective and a mesh of random
// ai: node displacements (up to 2 px), pictures of 256 to 1536 samples a side taken a strip of LIZARD_STRIP columns at a
// ai: time, as lizard_phy.c takes them, and the mean's grid of every eighth sample in one call. On arm64 and x86-64 at
// ai: gamma 1 ob_sample_grid_pic is sample_grid_lin, its read in integers (the x fraction a rounded 256th of a pixel)
// ai: and its lines stepped along a row (up to 48 adds, a few thousandths of a pixel at the mean grid's far columns),
// ai: held within 1/200 of the grey range of the walk at every point a pixel or more inside the frame (x86-64 measured
// ai: 0.0027 in the strips, 0.0038 in the mean's grid, on frames of random bytes; points past the frame's edges blend
// ai: the last two pixels by their own fraction: counted, not bounded); elsewhere, and at gamma 2.2, it is the walk
// ai: itself, 0 apart. On arm64 and x86-64 the integer read (lin_pts, lin_get) is also held
// ai: bit for bit to a scalar statement of arm64's instructions (FCVTZU to 16.16, the clamps on the integer parts, the
// ai: fraction's top byte and top 15 bits, 256 a + (b - a) fx modulo 2^16, UHSUB, SQRDMULH, USRA, 257 / 2^23) at random
// ai: points in and past the frame, NaN and infinities among them, under 2^15 pixels. Exit 1 on a bound or a bit
// ai: missed. It includes acquire.c and links the rest of the codec, from liblizard/:
// ai:   cc -O2 -ffp-contract=off -msse4.1 -Isrc test/sample_grid.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any,shake}.c -lm
#include "acquire.c"
#include "lizard_phy.h"
#include <stdio.h>
uint32_t ob_debug_hash;
static uint32_t rs = 1;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }

#if defined(OB_NEON) || defined(OB_X86)
static int64_t fdiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }   // ai: floor(a / b), b > 0
// ai: FCVTZU with 16 fraction bits: NaN and below 0 to 0, past 2^32 saturated
static uint32_t cvtzu16(float x) {
  if (!(x > 0)) return 0;
  const double d = (double)x * 65536.0;
  return d >= 4294967296.0 ? 0xffffffffu : (uint32_t)d;
}
// ai: arm64's integer read at one point (lin_pts, lin_grey, lin_get), x and y as lin_pts takes them
static float lin_model(const uint8_t *px, int w, int h, float x, float y) {
  const uint32_t xq = cvtzu16(x), yq = cvtzu16(y);
  uint32_t xi = xq >> 16, yi = yq >> 16;
  if (xi > (uint32_t)(w - 2)) xi = (uint32_t)(w - 2);
  if (yi > (uint32_t)(h - 2)) yi = (uint32_t)(h - 2);
  const int fx = (int)((xq >> 8) & 255), fy = (int)((yq & 0xffff) >> 1);
  const uint8_t *p = px + (size_t)yi * w + xi;
  const int top = (256 * p[0] + (p[1] - p[0]) * fx) & 0xffff, bot = (256 * p[w] + (p[w + 1] - p[w]) * fx) & 0xffff;
  const int hd = (int)fdiv(bot - top, 2);
  const int m = (int)fdiv(2 * (int64_t)hd * fy + (1 << 15), 1 << 16);
  const int16_t r = (int16_t)(uint16_t)((m + (top >> 1)) & 0xffff);
  return (float)(r * 257) / 8388608.0f;
}
#endif

int main(void) {
  const int iw = 1280, ih = 960, sides[3] = { 158, 286, 542 }, sizes[8] = { 256, 384, 512, 768, 1024, 1152, 1280, 1536 };
  uint8_t *px = malloc((size_t)iw * ih);
  float *a = malloc((size_t)1536 * LIZARD_STRIP * sizeof(float)), *b = malloc((size_t)1536 * LIZARD_STRIP * sizeof(float));
  double dmax[2] = { 0, 0 }, dsum = 0;   // ai: the strips', the mean grid's
  long npts = 0, nout = 0;
#if defined(OB_NEON) || defined(OB_X86)
  long nbits = 0, nmodel = 0;
#endif
#ifdef OB_V3_ON
  // ai: x86-64-v3's sampler (sample_grid_lin_v3) against the v2 form, bit for bit, where the processor has the level
  float *c = malloc((size_t)1536 * LIZARD_STRIP * sizeof(float));
  long v3diff = 0, v3pts = 0;
  const int v3 = ob_v3();
#endif
  for (int t = 0; t < 24; t++) {
    // ai: random bytes over a slow ramp (the steepest gradients there are), or blocks of 4 x 4
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
      const int v = t & 1 ? (int)(rnd() & 255) : (int)(((x / 4 * 7 + y / 4 * 13) * 37 + t * 11) & 255);
      px[(size_t)y * iw + x] = (uint8_t)v;
    }
    image_t im;
    ob_image_init(&im, px, iw, ih, t % 6 == 5 ? 2.2f : 1.0f);
    ob_layout_t L;
    ob_layout_init(&L, sides[t % 3]);
    // ai: the code inside the frame, or (every fourth) moved 400 px left, its left part past the frame's edge
    const float cx = iw * 0.5f - (t % 4 == 3 ? 400 : 0), cy = ih * 0.5f, s = 280 + 120 * frand();
    const float quad[8] = { cx - s + 30 * frand(), cy - s * 0.9f + 30 * frand(), cx + s + 30 * frand(), cy - s * 0.9f + 30 * frand(),
                            cx + s + 30 * frand(), cy + s * 0.9f + 30 * frand(), cx - s + 30 * frand(), cy + s * 0.9f + 30 * frand() };
    float src[8];
    corner_coords(&L, src);
    ob_reg_t reg;
    homography(src, quad, &reg.H);
    reg.nodes = calloc((size_t)L.nx * L.ny, sizeof(node_t));
    for (int k = 0; k < L.nx * L.ny; k++) { reg.nodes[k].dx = (frand() - 0.5f) * 4; reg.nodes[k].dy = (frand() - 0.5f) * 4; reg.nodes[k].done = 1; }
    const int n = sizes[t % 8];
    const float step = (float)(L.w - 2 * OB_MARGIN) / n, g0 = OB_MARGIN + 0.5f * step;
    // ai: the strips, then the mean's grid (n / 8 a side, every eighth sample) in one call
    for (int x0 = 0; x0 <= n; x0 += LIZARD_STRIP) {
      const int mean = x0 == n, sw = mean ? n / 8 : LIZARD_STRIP, rows = mean ? n / 8 : n, xs = mean ? 0 : x0;
      const float st = mean ? 8 * step : step;
      ob_sample_grid_pic(&L, &im, &reg, g0, g0, st, xs, sw, 0, rows, a, sw);
      ob_sample_grid(&L, &im, &reg, g0, g0, st, xs, sw, 0, rows, b, sw);
#ifdef OB_V3_ON
      if (v3 && im.lin) {
        for (int q = 0; q < 2; q++) {   // ai: q 0 the v2 form into c, q 1 the v3 form into a again
          float *dst = q ? a : c;
          for (int cc = 0; cc < sw; cc += GRID_LIN) {
            const int m = sw - cc < GRID_LIN ? sw - cc : GRID_LIN;
            if (q) sample_grid_lin_v3(&L, &im, &reg, g0, g0, st, xs + cc, m, 0, rows, dst + cc, sw);
            else sample_grid_lin(&L, &im, &reg, g0, g0, st, xs + cc, m, 0, rows, dst + cc, sw);
          }
        }
        for (int i = 0; i < rows * sw; i++) { v3diff += memcmp(&a[i], &c[i], sizeof(float)) != 0; v3pts++; }
      }
#endif
      for (int y = 0; y < rows; y++) for (int x = 0; x < sw; x++) {
        float u, v;
        ob_map_point(&L, &reg, g0 + (float)(xs + x) * st, g0 + (float)y * st, &u, &v);
        const double d = fabs((double)a[y * sw + x] - b[y * sw + x]);
        if (!(u >= 1.5f && u <= iw - 1.5f && v >= 1.5f && v <= ih - 1.5f)) { nout++; continue; }
        dmax[mean] = fmax(dmax[mean], d); dsum += d; npts++;
      }
    }
#if defined(OB_NEON) || defined(OB_X86)
    // ai: the integer read against its statement, eight points a call
    const lin_k_t K = lin_k(iw, ih);
    for (int k = 0; k < 4096; k += 8) {
      float X[8], Y[8], o8[8];
      for (int l = 0; l < 8; l++) {
        X[l] = frand() * (iw + 80) - 40; Y[l] = frand() * (ih + 80) - 40;
        if (rnd() % 31 == 0) X[l] = floorf(X[l]);
        if (rnd() % 97 == 0) X[l] = NAN;
        if (rnd() % 97 == 0) Y[l] = rnd() & 1 ? -INFINITY : 30000.0f;
      }
#if defined(OB_NEON)
      lin_read(px, px + iw - 2, &K, lin_pts(&K, vld1q_f32(X), vld1q_f32(Y), vld1q_f32(X + 4), vld1q_f32(Y + 4)), o8);
#else
      lin_read(px, px + iw - 2, &K, lin_pts(&K, _mm_loadu_ps(X), _mm_loadu_ps(Y), _mm_loadu_ps(X + 4), _mm_loadu_ps(Y + 4)), o8);
#endif
      for (int l = 0; l < 8; l++) { const float m = lin_model(px, iw, ih, X[l], Y[l]); nbits += memcmp(&m, &o8[l], sizeof m) != 0; nmodel++; }
    }
#endif
    free(reg.nodes);
    ob_layout_free(&L);
  }
  free(px); free(a); free(b);
  printf("picture sampler against ob_sample_grid: %.5f of the grey range at most in the strips, %.5f in the mean's grid, %.2e on average, over %ld points (%ld past the frame's edges, not bounded)\n",
         dmax[0], dmax[1], dsum / (npts ? npts : 1), npts, nout);
  int fail = dmax[0] > 1.0 / 200 || dmax[1] > 1.0 / 200;
#if defined(OB_NEON) || defined(OB_X86)
  printf("integer read against its statement: %ld of %ld points differ\n", nbits, nmodel);
  fail |= nbits != 0;
#endif
#ifdef OB_V3_ON
  if (v3) printf("x86-64-v3 sampler against the v2 form: %ld of %ld points differ\n", v3diff, v3pts);
  else printf("x86-64-v3 sampler: this processor has no v3, not checked\n");
  fail |= v3diff != 0;
  free(c);
#endif
  if (fail) printf("a bound or a bit missed\n");
  return fail;
}
