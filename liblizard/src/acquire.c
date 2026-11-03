// Registration: find the border (its corner marks, or its line), fix orientation by the timing track, hang a mesh on
// the border's nodes. The binarizer serves the border's search alone; data cells are never thresholded.
#include "internal.h"
#include "fmt.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ai: Off wasm the clock is the wall's: clock() is the whole process's CPU time, which a receiver on several threads
// ai: cannot time a stage by.
#ifdef __EMSCRIPTEN__
double ob_now_ms(void) { return 1000.0 * clock() / CLOCKS_PER_SEC; }
#else
double ob_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec; }
#endif
OB_TLS double ob_prof_ms[PROF_N];
// The quad the finder settled on, BEFORE the corner nodes refine it. res->quad is the refined one, and handing
// that back to ob_acquire_quad would refine it a second time; this is what a finder running elsewhere has to
// reproduce and what it has to be compared against.
// What the thin frame's finder settled on, for a finder elsewhere to be held to: the final
// score and orient, then which form and which mark quad won, the source of the final quad (0 marks, 1 lines,
// -1 none), whether the marks alone settled it, the mark stage's best, and form 0's quad count.
OB_TLS float ob_finder_info[8];
// ai: The lens's axis in the next image this thread decodes (lizard_set_axis), image px; NaN for the image's centre.
static OB_TLS float ob_axis[2] = { NAN, NAN };
void lizard_set_axis(float x, float y) { ob_axis[0] = x; ob_axis[1] = y; }
// ai: The anchors (2026-10-26; part of every symbol since 2026-10-27, as the pilots are; SPEC 6.13,
// ai: research/build/anchors/PLAN.md): a 7 x 7 grid of squares inside the picture, B / 32 modules a side (2, 4, 8 in
// ai: the 64, 128, 256 rings), square (i, j) on the modules from round(OB_MARGIN + span (i + 0.5) / 7 - a / 2) on, each
// ai: a light middle (Chebyshev distance under a quarter of the half-side), a dark ring (to 0.6 of it) and a light rim.
// ai: lizard_encode paints them over the picture. acquire_core finds them through the corners' homography first, three
// ai: modules wide, in rounds through the lens model the ones found so far give (anchor_grow), starts every border
// ai: node's search from that model plus its measured neighbours' residuals from it, and fills the interior from the
// ai: border and the anchors (border_fill_anchors) where it found OB_ANCH_MIN (12) or more; lizard_finish erases them always.
//
// ai: the samples erased round an anchor's square (lizard_phy.c): two, measured on the published lenses' captures (one left
// ai: the blurred rim in where the map sits a few tenths of a sample off: 26 of 192 frames lost blocks; three read as two)
// ai: the anchors the last acquisition counted (anchor_register's, read out by the wasm's lizard_anchors_found for the
// ai: lab's checks), -1 where no mesh was made
OB_TLS int ob_anch_found = -1;
int ob_anchor_side(const ob_layout_t *L) { return (L->w - 2 * OB_MARGIN) / 64; }
int ob_anchor_m0(const ob_layout_t *L, int i) {
  const int span = L->w - 2 * OB_MARGIN, a = span / 64;
  return (int)floor(OB_MARGIN + (double)span * (i + 0.5) / OB_ANCH_K - a / 2.0 + 0.5);
}
// ai: the lens model the anchors give (border_fill_anchors' fit), evaluated at an image point: defined further on
struct lensm;
static void lensm_at(const struct lensm *m, double u, double v, float *dx, float *dy);

void ob_image_init(image_t *im, const uint8_t *px, int w, int h, float gamma) {
  im->px = px; im->w = w; im->h = h;
  for (int i = 0; i < 256; i++) im->lut[i] = gamma == 1.0f ? i / 255.0f : powf(i / 255.0f, gamma);
  im->lin = gamma == 1.0f;
}

// ---------------------------------------------------------------- finder search

// Dark where a pixel sits below the mean of the 5 x 5 blocks around it. Only the finder search
// reads this, and a finder's rings are at least three camera px wide wherever the data is
// readable, so a plain local mean is enough.
enum { BLK = 8 };

#include "simd.h"

// RGBA to luma, (77 r + 150 g + 29 b + 128) >> 8. The weights sum to 256, so the sum fits 16 bits
// and sixteen pixels go through as two lanes of eight.
void ob_luma(const uint8_t *rgba, int n, uint8_t *out) {
  int i = 0;
#ifdef OB_SIMD
  const v128_t cr = wasm_i16x8_splat(77), cg = wasm_i16x8_splat(150), cb = wasm_i16x8_splat(29), half = wasm_i16x8_splat(128);
  for (; i + 16 <= n; i += 16) {
    const v128_t p0 = wasm_v128_load(rgba + 4 * i), p1 = wasm_v128_load(rgba + 4 * i + 16), p2 = wasm_v128_load(rgba + 4 * i + 32), p3 = wasm_v128_load(rgba + 4 * i + 48);
    // Two rounds of byte shuffles pull each channel of sixteen pixels into one vector.
    const v128_t r01 = wasm_i8x16_shuffle(p0, p1, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29), b01 = wasm_i8x16_shuffle(p0, p1, 2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    const v128_t r23 = wasm_i8x16_shuffle(p2, p3, 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29), b23 = wasm_i8x16_shuffle(p2, p3, 2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0);
    const v128_t r = wasm_i8x16_shuffle(r01, r23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23), g = wasm_i8x16_shuffle(r01, r23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
    const v128_t b = wasm_i8x16_shuffle(b01, b23, 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
    #define LUM8(ext) wasm_u16x8_shr(wasm_i16x8_add(wasm_i16x8_add(wasm_i16x8_mul(ext(r), cr), wasm_i16x8_mul(ext(g), cg)), wasm_i16x8_add(wasm_i16x8_mul(ext(b), cb), half)), 8)
    wasm_v128_store(out + i, wasm_u8x16_narrow_i16x8(LUM8(wasm_u16x8_extend_low_u8x16), LUM8(wasm_u16x8_extend_high_u8x16)));
    #undef LUM8
  }
#endif
  for (; i < n; i++) out[i] = (uint8_t)((77 * rgba[4 * i] + 150 * rgba[4 * i + 1] + 29 * rgba[4 * i + 2] + 128) >> 8);
}

// A block's threshold: the mean of the block means within two blocks of it, as many as the image has, less 3.
static inline int bin_threshold(const int *mean, int bw, int bh, int bx, int by) {
  int s = 0, n = 0;
  for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
    int yy = by + dy, xx = bx + dx;
    if (yy < 0 || xx < 0 || yy >= bh || xx >= bw) continue;
    s += mean[yy * bw + xx]; n++;
  }
  const int t = s / n - 3;
  return t < 0 ? 0 : t;   // a pixel is never below 0, so a negative threshold and 0 mark the same (nothing)
}

// ai: an acquisition's buffer slot of n bytes (internal.h ob_fbufs_t): on arm64 the workspace's, grown where it is short
// ai: (its old bytes not kept), and frame_buf_done leaves it there; with no workspace, or elsewhere, a fresh allocation
// ai: frame_buf_done frees
#ifdef OB_NEON
static void *frame_buf(ob_fbufs_t *fb, int slot, size_t n) {
  if (!fb) return malloc(n);
  if (fb->cap[slot] < n) { free(fb->p[slot]); fb->p[slot] = malloc(n); fb->cap[slot] = fb->p[slot] ? n : 0; }
  return fb->p[slot];
}
static void frame_buf_done(ob_fbufs_t *fb, void *p) { if (!fb) free(p); }
#else
#define frame_buf(fb, slot, n) ((void)(fb), malloc(n))
#define frame_buf_done(fb, p) ((void)(fb), free(p))
#endif
void ob_fbufs_free(ob_fbufs_t *fb) { for (int k = 0; k < OB_FB_SLOTS; k++) { free(fb->p[k]); fb->p[k] = NULL; fb->cap[k] = 0; } }
#ifdef OB_NEON
// ai: arm64: the binarized image's columns as bits (the corner marks' column walks, cb_run), 32 rows a word: word
// ai: (y >> 5) w + x holds column x's rows y & ~31 on, bit y & 31 its pixel (rows past h 0). binarize makes them as it
// ai: compares: a block row's eight rows' bytes (0 or 1) into one by shift-inserts (SLI), a band's four of those
// ai: interleaved into words by ST4; this fills band b's words from column x0 on from the bytes, where that cannot run
// ai: (the columns past the last sixteen, the last band where h is not a multiple of 32)
static void col_bits_band(const uint8_t *bin, int w, int h, int b, int x0, uint32_t *cb) {
  const int y0 = 32 * b, rows = h - y0 < 32 ? h - y0 : 32;
  for (int x = x0; x < w; x++) {
    uint32_t m = 0;
    for (int r = 0; r < rows; r++) m |= (uint32_t)(bin[(size_t)(y0 + r) * w + x] != 0) << r;
    cb[(size_t)b * w + x] = m;
  }
}
#endif
// Three passes: 8 x 8 block means, a 5 x 5 mean of those less 3 as each block's threshold, then the
// compare. The vector paths cover whole blocks; a ragged right or bottom edge takes the scalar one.
// ai: cb: on arm64, the column bits made as well where not NULL ((h + 31) / 32 w words); NULL elsewhere. fb: frame_buf's
static void binarize(const uint8_t *img, int w, int h, uint8_t *bin, uint32_t *cb, ob_fbufs_t *fb) {
  int bw = (w + BLK - 1) / BLK, bh = (h + BLK - 1) / BLK;
  int *mean = frame_buf(fb, OB_FB_MEAN, (size_t)bw * bh * sizeof(int));
  uint8_t *trow = malloc((size_t)bw * BLK + 16);
  uint16_t *col = malloc(((size_t)bw * BLK + 16) * sizeof(uint16_t));
#ifdef OB_NEON
  int *vsum = malloc(((size_t)bw + 4) * sizeof(int));   // a block row's sums down five block rows
  const int bands = h / 32;                             // the whole bands, whose words the compare makes
  uint8_t *planes = cb ? malloc((size_t)4 * w + 16) : NULL;   // a band's four block rows' bytes of bits
#else
  (void)cb;
#endif
  for (int by = 0; by < bh; by++) {
    int bx = 0;
#ifdef OB_NEON
    // ai: arm64: four whole blocks a step: each image row's pixel pairs accumulated into 16-bit lanes (UADALP, 8 x 510
    // ai: at most), the pairs added to fours (ADDP, 8,160 at most) and the fours to eights in 32-bit lanes (UADDLP):
    // ai: the integer sums of the loop below, so the same means
    if (by < h / BLK) {
      const uint8_t *p = img + (size_t)by * BLK * w;
      for (; (bx + 4) * BLK <= w; bx += 4) {
        uint16x8_t a = vdupq_n_u16(0), b = a;
        for (int r = 0; r < BLK; r++) { a = vpadalq_u8(a, vld1q_u8(p + (size_t)r * w + bx * BLK)); b = vpadalq_u8(b, vld1q_u8(p + (size_t)r * w + bx * BLK + 16)); }
        vst1q_s32(mean + by * bw + bx, vreinterpretq_s32_u32(vshrq_n_u32(vpaddlq_u16(vpaddq_u16(a, b)), 6)));
      }
    }
#elif defined(OB_SIMD)
    const int fw = w / BLK, fh = h / BLK;   // whole blocks
    if (by < fh) {
      // Column sums of the block row's eight image rows (2040 at most, so 16 bits), then eight columns to a block.
      int x = 0;
      for (; x + 16 <= fw * BLK; x += 16) {
        v128_t lo = wasm_i16x8_splat(0), hi = lo;
        for (int r = 0; r < BLK; r++) { v128_t p = wasm_v128_load(img + (by * BLK + r) * w + x); lo = wasm_i16x8_add(lo, wasm_u16x8_extend_low_u8x16(p)); hi = wasm_i16x8_add(hi, wasm_u16x8_extend_high_u8x16(p)); }
        wasm_v128_store(col + x, lo); wasm_v128_store(col + x + 8, hi);
      }
      for (; bx < x / BLK; bx++) {
        v128_t q = wasm_u32x4_extadd_pairwise_u16x8(wasm_v128_load(col + bx * BLK));
        int s = wasm_i32x4_extract_lane(q, 0) + wasm_i32x4_extract_lane(q, 1) + wasm_i32x4_extract_lane(q, 2) + wasm_i32x4_extract_lane(q, 3);
        mean[by * bw + bx] = s >> 6;
      }
    }
#endif
    for (; bx < bw; bx++) {
      int s = 0, n = 0;
      for (int y = by * BLK; y < (by + 1) * BLK && y < h; y++) for (int x = bx * BLK; x < (bx + 1) * BLK && x < w; x++) { s += img[y * w + x]; n++; }
      mean[by * bw + bx] = s / n;
    }
  }
  for (int by = 0; by < bh; by++) {
    int bx = 0;
#ifdef OB_NEON
    // ai: arm64: a block row with two rows of blocks above and below, its blocks from the third to the third from the
    // ai: end four a step: the five rows' sums, then five of those, all 25 blocks there (n = 25), s / 25 as
    // ai: (s * 5243) >> 17 (exact for every s to 25 x 255 = 6,375), the threshold's byte spread over its block's 8
    if (by >= 2 && by + 2 < bh && vsum) {
      const int *m0 = mean + (by - 2) * bw;
      int x = 0;
      for (; x + 4 <= bw; x += 4) {
        int32x4_t s = vld1q_s32(m0 + x);
        for (int d = 1; d < 5; d++) s = vaddq_s32(s, vld1q_s32(m0 + d * bw + x));
        vst1q_s32(vsum + x, s);
      }
      for (; x < bw; x++) { int s = 0; for (int d = 0; d < 5; d++) s += m0[d * bw + x]; vsum[x] = s; }
      static const uint8_t SPREAD[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };
      const uint8x16_t lo = vld1q_u8(SPREAD), hi = vld1q_u8(SPREAD + 16);
      for (bx = 2; bx + 4 <= bw - 2; bx += 4) {
        const int32x4_t s = vaddq_s32(vaddq_s32(vaddq_s32(vld1q_s32(vsum + bx - 2), vld1q_s32(vsum + bx - 1)), vaddq_s32(vld1q_s32(vsum + bx), vld1q_s32(vsum + bx + 1))), vld1q_s32(vsum + bx + 2));
        const uint32x4_t q = vshrq_n_u32(vmulq_u32(vreinterpretq_u32_s32(s), vdupq_n_u32(5243)), 17);
        const uint16x4_t t = vmovn_u32(vreinterpretq_u32_s32(vmaxq_s32(vsubq_s32(vreinterpretq_s32_u32(q), vdupq_n_s32(3)), vdupq_n_s32(0))));
        const uint8x16_t t8 = vcombine_u8(vmovn_u16(vcombine_u16(t, t)), vdup_n_u8(0));
        vst1q_u8(trow + bx * BLK, vqtbl1q_u8(t8, lo)); vst1q_u8(trow + bx * BLK + 16, vqtbl1q_u8(t8, hi));
      }
      for (int e = 0; e < 2; e++) memset(trow + e * BLK, bin_threshold(mean, bw, bh, e, by), BLK);
    }
#endif
    for (; bx < bw; bx++) memset(trow + bx * BLK, bin_threshold(mean, bw, bh, bx, by), BLK);
#ifdef OB_NEON
    // ai: a block row of a whole band: its eight rows compared sixteen columns at a time (the same bytes as below),
    // ai: their bytes shifted into one plane of bits; a band's fourth block row stores the four planes as words
    if (planes && by < 4 * bands) {
      const uint8_t *s0 = img + (size_t)by * BLK * w;
      uint8_t *d0 = bin + (size_t)by * BLK * w, *pl = planes + (size_t)(by & 3) * w;
      const uint8x16_t one = vdupq_n_u8(1);
      int x = 0;
      for (; x + 16 <= w; x += 16) {
        const uint8x16_t t = vld1q_u8(trow + x);
        uint8x16_t b, v;
#define BIN_ROW(r) b = vandq_u8(vcltq_u8(vld1q_u8(s0 + (size_t)(r) * w + x), t), one); vst1q_u8(d0 + (size_t)(r) * w + x, b);
        BIN_ROW(0) v = b;
        BIN_ROW(1) v = vsliq_n_u8(v, b, 1); BIN_ROW(2) v = vsliq_n_u8(v, b, 2); BIN_ROW(3) v = vsliq_n_u8(v, b, 3);
        BIN_ROW(4) v = vsliq_n_u8(v, b, 4); BIN_ROW(5) v = vsliq_n_u8(v, b, 5); BIN_ROW(6) v = vsliq_n_u8(v, b, 6);
        BIN_ROW(7) v = vsliq_n_u8(v, b, 7);
#undef BIN_ROW
        vst1q_u8(pl + x, v);
      }
      for (int r = 0; r < BLK; r++) for (int c = x; c < w; c++) d0[(size_t)r * w + c] = s0[(size_t)r * w + c] < trow[c];
      if ((by & 3) == 3) {
        uint32_t *o = cb + (size_t)(by >> 2) * w;
        int c = 0;
        for (; c + 16 <= w; c += 16) {
          const uint8x16x4_t q = { { vld1q_u8(planes + c), vld1q_u8(planes + w + c), vld1q_u8(planes + 2 * (size_t)w + c), vld1q_u8(planes + 3 * (size_t)w + c) } };
          vst4q_u8((uint8_t *)(o + c), q);
        }
        col_bits_band(bin, w, h, by >> 2, c, cb);
      }
      continue;
    }
#endif
    for (int y = by * BLK; y < (by + 1) * BLK && y < h; y++) {
      const uint8_t *src = img + y * w;
      uint8_t *dst = bin + y * w;
      int x = 0;
#ifdef OB_SIMD
      const v128_t one = wasm_i8x16_splat(1);
      for (; x + 16 <= w; x += 16) wasm_v128_store(dst + x, wasm_v128_and(wasm_u8x16_lt(wasm_v128_load(src + x), wasm_v128_load(trow + x)), one));
#endif
      for (; x < w; x++) dst[x] = src[x] < trow[x];
    }
  }
#ifdef OB_NEON
  // ai: the last band where h is not a multiple of 32, or every band where the planes could not be had
  if (cb) for (int b = planes ? bands : 0; b < (h + 31) / 32; b++) col_bits_band(bin, w, h, b, 0, cb);
  free(vsum); free(planes);
#endif
  frame_buf_done(fb, mean); free(trow); free(col);
}

void ob_test_binarize(const uint8_t *img, int w, int h, uint8_t *bin) { binarize(img, w, h, bin, NULL, NULL); }

// ai: The node templates read a blurred image (box_blur) whenever the box has a radius, r = floor(mpx / 2) pixels,
// ai: so from a module of 2 px (2026-09-29: the threshold of 2.0 px this replaced was exactly r >= 1, the same
// ai: bytes). The fifteen recorded phone runs (scripts/exp/capture_check.mjs) had tried later starts: every threshold from 2
// ai: to 3 took 31005 blocks to 44200 where 2 took them to 45477, one run the difference (7677 to 8953); on the
// ai: simulator 2 and 3 the same under 2 px and within a few percent either way between them (LIZARD-128 at 45
// ai: degrees 2843 to 3166 B).
// Why the node templates read a blurred image when a module is several pixels across (border_blur). A template
// reads one point a module, at its centre, and that point only moves when an edge reaches it: at 8 px a module it
// sits 4 px from every edge, the correlation is flat over most of a module and the node cannot be placed to better
// than a large part of one. Measured on LIZARD-16 at 94 modules, 1080 fill 0.7: corners 0.76 px RMS off the truth
// against 0.16 on the 286-module grid, and half the blocks gone at 30 degrees. Under a box as wide as a module the
// centre point IS the module's mean, which moves linearly with the offset, so the peak is sharp again at any size.
//
// One rectangle [x0, x1) x [y0, y1) of the blurred image written into out, reading real pixels past its edges,
// so a set of rectangles blurs exactly as the whole image would inside them. Rows then columns by running sums,
// the image's own edges clamped, rounded back to bytes. tmp holds (x1 - x0) * (y1 - y0 + 2r) and acc (x1 - x0).
static void box_blur_rect(const uint8_t *src, uint8_t *out, int w, int h, int r, int x0, int y0, int x1, int y1, uint16_t *tmp, int *acc) {
  const int rw = x1 - x0, rows = y1 - y0 + 2 * r, kk = (2 * r + 1) * (2 * r + 1);
#ifdef OB_SIMD
  // ai: the same integer sums a vector at a time, so the same bytes. A row's sums directly, 2r + 1 widened byte loads per
  // ai: eight columns where no read is clamped (the clamped columns by the scalar sum); the column sums in 16-bit lanes
  // ai: (at most 255 (2r + 1)^2, 57,375 at r 7, and a sum less its leaving row before the entering one is added); the
  // ai: rounded division by kk a multiply and a shift, exact for sums under 2^32 / kk^2 (85,000 and more at r 7)
  if (r <= 7) {
    // the columns whose vector reads x0 + x - r .. x0 + x + r + 15 all lie in the row: x from xa while x <= xz
    const int xa = r - x0 > 0 ? r - x0 : 0, xz = w - 16 - r - x0;
    for (int j = 0; j < rows; j++) {
      const int sy = y0 - r + j;
      const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w;
      uint16_t *t = tmp + (size_t)j * rw;
      int x = 0;
      for (; x < rw && x < xa; x++) { int a = 0; for (int i = x0 + x - r; i <= x0 + x + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i]; t[x] = (uint16_t)a; }
      for (; x + 8 <= rw && x <= xz; x += 8) {
        v128_t a = wasm_i16x8_splat(0);
        for (int sft = -r; sft <= r; sft++) a = wasm_i16x8_add(a, wasm_u16x8_extend_low_u8x16(wasm_v128_load(srow + x0 + x + sft)));
        wasm_v128_store(t + x, a);
      }
      for (; x < rw; x++) { int a = 0; for (int i = x0 + x - r; i <= x0 + x + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i]; t[x] = (uint16_t)a; }
    }
    uint16_t *col = (uint16_t *)acc;   // rw ints hold rw 16-bit sums twice over
    int x = 0;
    for (; x + 8 <= rw; x += 8) {
      v128_t c = wasm_i16x8_splat(0);
      for (int j = 0; j < 2 * r + 1; j++) c = wasm_i16x8_add(c, wasm_v128_load(tmp + (size_t)j * rw + x));
      wasm_v128_store(col + x, c);
    }
    for (; x < rw; x++) { int c = 0; for (int j = 0; j < 2 * r + 1; j++) c += tmp[(size_t)j * rw + x]; col[x] = (uint16_t)c; }
    const uint64_t M = ((uint64_t)1 << 32) / (uint64_t)kk + 1;
    const unsigned half = (unsigned)kk / 2;
    for (int y = y0; y < y1; y++) {
      uint8_t *o = out + (size_t)y * w + x0;
      for (int xx = 0; xx < rw; xx++) o[xx] = (uint8_t)(((uint64_t)(col[xx] + half) * M) >> 32);
      if (y + 1 == y1) break;
      const uint16_t *ta = tmp + (size_t)(y - y0 + 2 * r + 1) * rw, *ts = tmp + (size_t)(y - y0) * rw;
      int xx = 0;
      for (; xx + 8 <= rw; xx += 8) wasm_v128_store(col + xx, wasm_i16x8_add(wasm_i16x8_sub(wasm_v128_load(col + xx), wasm_v128_load(ts + xx)), wasm_v128_load(ta + xx)));
      for (; xx < rw; xx++) col[xx] = (uint16_t)(col[xx] - ts[xx] + ta[xx]);
    }
    return;
  }
#endif
  for (int j = 0; j < rows; j++) {
    const int sy = y0 - r + j;
    const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w;
    uint16_t *t = tmp + (size_t)j * rw;
    int a = 0;
    for (int i = x0 - r; i <= x0 + r; i++) a += srow[i < 0 ? 0 : i >= w ? w - 1 : i];
    for (int x = 0; x < rw; x++) {
      t[x] = (uint16_t)a;
      const int add = x0 + x + r + 1, sub = x0 + x - r;
      a += srow[add >= w ? w - 1 : add] - srow[sub < 0 ? 0 : sub];
    }
  }
  for (int x = 0; x < rw; x++) acc[x] = 0;
  for (int j = 0; j < 2 * r + 1; j++) { const uint16_t *t = tmp + (size_t)j * rw; for (int x = 0; x < rw; x++) acc[x] += t[x]; }
  for (int y = y0; y < y1; y++) {
    uint8_t *o = out + (size_t)y * w + x0;
    for (int x = 0; x < rw; x++) o[x] = (uint8_t)((acc[x] + kk / 2) / kk);
    if (y + 1 == y1) break;
    const uint16_t *ta = tmp + (size_t)(y - y0 + 2 * r + 1) * rw, *ts = tmp + (size_t)(y - y0) * rw;
    for (int x = 0; x < rw; x++) acc[x] += ta[x] - ts[x];
  }
}

#ifdef OB_NEON
// ai: arm64's box_blur_rect for r <= 7 and rectangles 16 or more wide: the same integer sums, so the same bytes, with no
// ai: rectangle's worth of row sums: each row's sums are made as the column sums take them in, into a ring of 2r + 2
// ai: rows (ring: (2r + 2) rw), and a row's output and its column sums' step are one pass, from one copy of the column
// ai: sums into the other (col: 2 rw, so the last step's columns, back over ones already done, give the same values).
// ai: A row's sums sixteen columns a step from its rw + 2r pixels, through buf (rw + 2r bytes) only where a read is
// ai: clamped; the last step back over columns already summed.
static inline __attribute__((always_inline)) void msh_row_sums(const uint8_t *src, int w, int h, int r, int x0, int rw, int sy, uint16_t *t, uint8_t *buf) {
  const int lo = x0 - r, n = rw + 2 * r;
  const uint8_t *srow = src + (size_t)(sy < 0 ? 0 : sy >= h ? h - 1 : sy) * w, *p = srow + lo;
  if (lo < 0 || lo + n > w) {
    const int a0 = lo < 0 ? -lo : 0, a1 = (lo + n <= w ? lo + n : w) - lo;
    memset(buf, srow[0], (size_t)a0); memcpy(buf + a0, srow + lo + a0, (size_t)(a1 - a0)); memset(buf + a1, srow[w - 1], (size_t)(n - a1));
    p = buf;
  }
  for (int x = 0;; x += 16) {
    if (x + 16 > rw) x = rw - 16;
    uint16x8_t s0 = vdupq_n_u16(0), s1 = s0;
    for (int sft = 0; sft <= 2 * r; sft++) { const uint8x16_t b = vld1q_u8(p + x + sft); s0 = vaddw_u8(s0, vget_low_u8(b)); s1 = vaddw_high_u8(s1, b); }
    vst1q_u16(t + x, s0); vst1q_u16(t + x + 8, s1);
    if (x + 16 >= rw) break;
  }
}
static void msh_blur_rect(const uint8_t *src, uint8_t *out, int w, int h, int r, int x0, int y0, int x1, int y1, uint16_t *ring, uint16_t *col, uint8_t *buf) {
  const int rw = x1 - x0, R = 2 * r + 2, kk = (2 * r + 1) * (2 * r + 1);
  for (int j = 0; j <= 2 * r; j++) msh_row_sums(src, w, h, r, x0, rw, y0 - r + j, ring + (size_t)j * rw, buf);
  uint16_t *ca = col, *cb = col + rw;
  for (int x = 0;; x += 8) {
    if (x + 8 > rw) x = rw - 8;
    uint16x8_t c = vld1q_u16(ring + x);
    for (int j = 1; j <= 2 * r; j++) c = vaddq_u16(c, vld1q_u16(ring + (size_t)j * rw + x));
    vst1q_u16(ca + x, c);
    if (x + 8 >= rw) break;
  }
  // ai: the rounded division by kk: (c + kk / 2) M >> 32 in 64-bit lanes (M under 2^32), its high half (a 16-bit
  // ai: multiply's high half and a shift, exact at every radius, measured no faster)
  const uint64_t M = ((uint64_t)1 << 32) / (uint64_t)kk + 1;
  const uint32x4_t vh = vdupq_n_u32((uint32_t)kk / 2), vm = vdupq_n_u32((uint32_t)M);
  for (int y = y0; y < y1; y++) {
    uint8_t *o = out + (size_t)y * w + x0;
    const int last = y + 1 == y1, ja = y - y0 + 2 * r + 1;
    uint16_t *ta = ring + (size_t)(ja % R) * rw;
    const uint16_t *ts = ring + (size_t)((y - y0) % R) * rw;
    if (!last) msh_row_sums(src, w, h, r, x0, rw, y0 - r + ja, ta, buf);
    for (int x = 0;; x += 8) {
      if (x + 8 > rw) x = rw - 8;
      const uint16x8_t c = vld1q_u16(ca + x);
      const uint32x4_t lo = vaddw_u16(vh, vget_low_u16(c)), hi = vaddw_high_u16(vh, c);
      const uint32x4_t qlo = vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(lo), vget_low_u32(vm)), 32), vshrn_n_u64(vmull_high_u32(lo, vm), 32));
      const uint32x4_t qhi = vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(hi), vget_low_u32(vm)), 32), vshrn_n_u64(vmull_high_u32(hi, vm), 32));
      vst1_u8(o + x, vmovn_u16(vcombine_u16(vmovn_u32(qlo), vmovn_u32(qhi))));
      if (!last) vst1q_u16(cb + x, vaddq_u16(vsubq_u16(c, vld1q_u16(ts + x)), vld1q_u16(ta + x)));
      if (x + 8 >= rw) break;
    }
    uint16_t *sw = ca; ca = cb; cb = sw;
  }
}
#endif

typedef struct { float x, y, unit; int hits; } cand_t;
enum { MAX_CAND = 1024 };
#ifdef OB_NEON
// ai: arm64: a candidate list's x and y side by side past its MAX_CAND entries (x at mark_xy(c)[k], y at
// ai: mark_xy(c)[MAX_CAND + k]), which mark_note's scan reads as plain vectors (LD4 over the entries read twice the
// ai: bytes, in more operations: the marks 4 to 18% faster on the S26); kept in step with the entries by mark_note and
// ai: mark_candidates' prune, and read nowhere else. A list is allocated MAX_CAND entries and CAND_XY bytes.
enum { CAND_XY = 2 * MAX_CAND * sizeof(float) };
static inline float *mark_xy(cand_t *c) { return (float *)(c + MAX_CAND); }
#else
enum { CAND_XY = 0 };
#endif

static float cross2(const cand_t *a, const cand_t *b, const cand_t *c) { return (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x); }

#ifdef OB_NEON
// ai: arm64: a four-pixel group's boundaries, as the bits of its lanes (bit i: lane i differs from the pixel before
// ai: it), packed to the front, and how many there are
static const int32_t RUN_PACK[16][4] = {
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 2, 0, 0, 0 }, { 0, 2, 0, 0 }, { 1, 2, 0, 0 }, { 0, 1, 2, 0 },
  { 3, 0, 0, 0 }, { 0, 3, 0, 0 }, { 1, 3, 0, 0 }, { 0, 1, 3, 0 }, { 2, 3, 0, 0 }, { 0, 2, 3, 0 }, { 1, 2, 3, 0 }, { 0, 1, 2, 3 } };
static const uint8_t RUN_COUNT[16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
#endif
// Run boundaries of one row, sixteen pixels to a compare: pos[2k] is where dark run k starts and
// pos[2k + 1] where it ends. The pixel left of the row counts as light, and a dark run that reaches
// the right edge ends there. Returns the number of boundaries, always even.
static int run_edges(const uint8_t *row, int w, int *pos) {
  int np = 0, x = 1;
  if (row[0]) pos[np++] = 0;
#ifdef OB_NEON
  // ai: arm64: a chunk's sixteen compares weighted 1, 2, 4, 8 within each four and added pairwise twice (ADDP), which
  // ai: leaves the four groups' bits in four bytes; each group's positions are one table row plus x and one store of
  // ai: four, np moved on by the group's count: four stores where the loop below makes sixteen. A store writes past
  // ai: np by at most three, never past x + 15.
  const uint8x16_t wt = { 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8 };
  for (; x + 16 <= w; x += 16) {
    const uint8x16_t bits = vbicq_u8(wt, vceqq_u8(vld1q_u8(row + x), vld1q_u8(row + x - 1))), b2 = vpaddq_u8(bits, bits);
    const uint32_t g = vgetq_lane_u32(vreinterpretq_u32_u8(vpaddq_u8(b2, b2)), 0);
    if (!g) continue;
    for (int q = 0; q < 4; q++) {
      const unsigned k = (g >> 8 * q) & 15;
      vst1q_s32(pos + np, vaddq_s32(vld1q_s32(RUN_PACK[k]), vdupq_n_s32(x + 4 * q)));
      np += RUN_COUNT[k];
    }
  }
#elif defined(OB_SIMD)
  for (; x + 16 <= w; x += 16) {
    const unsigned m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(row + x), wasm_v128_load(row + x - 1)));
    if (!m) continue;
    // ai: every position written and the count moved on by its bit, with no branch a boundary: a binarized picture
    // ai: has one about every other pixel, and a loop over the set bits mispredicted its exit once a chunk (2026-11-01,
    // ai: a third of the finder's time on the S26). np never passes x + k + 1, so pos's w + 2 entries hold every write.
    for (int k = 0; k < 16; k++) { pos[np] = x + k; np += (int)(m >> k) & 1; }
  }
#endif
  for (; x < w; x++) if (row[x] != row[x - 1]) pos[np++] = x;
  if (np & 1) pos[np++] = w;
  return np;
}

// ---------------------------------------------------------------- geometry

static int solve8(double A[8][9]) {
  for (int c = 0; c < 8; c++) {
    int p = c;
    for (int r = c + 1; r < 8; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (fabs(A[p][c]) < 1e-12) return 0;
    for (int k = 0; k < 9; k++) { double t = A[c][k]; A[c][k] = A[p][k]; A[p][k] = t; }
    for (int r = 0; r < 8; r++) {
      if (r == c) continue;
      double f = A[r][c] / A[c][c];
#ifdef OB_NEON
      // ai: arm64: two columns a step in double pairs, each the multiply then the subtraction as below (no fused
      // ai: multiply-add, so the same doubles); the finder solves eight of these a quad it tries (1.4 M instructions a
      // ai: frame fewer on the blurred cell sim-lp-15)
      int k = c;
      if ((9 - c) & 1) { A[r][k] -= f * A[c][k]; k++; }
      const float64x2_t vf = vdupq_n_f64(f);
      for (; k < 9; k += 2) vst1q_f64(&A[r][k], vsubq_f64(vld1q_f64(&A[r][k]), vmulq_f64(vf, vld1q_f64(&A[c][k]))));
#else
      for (int k = c; k < 9; k++) A[r][k] -= f * A[c][k];
#endif
    }
  }
  return 1;
}

// src (module coordinates) to dst (image px), four point pairs.
static int homography(const float *src, const float *dst, homo_t *H) {
  double A[8][9];
  for (int i = 0; i < 4; i++) {
    double x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
    double r0[9] = { x, y, 1, 0, 0, 0, -u * x, -u * y, u }, r1[9] = { 0, 0, 0, x, y, 1, -v * x, -v * y, v };
    memcpy(A[2 * i], r0, sizeof r0); memcpy(A[2 * i + 1], r1, sizeof r1);
  }
  if (!solve8(A)) return 0;
  for (int i = 0; i < 8; i++) H->h[i] = (float)(A[i][8] / A[i][i]);
  H->h[8] = 1;
  return 1;
}

static void corner_coords(const ob_layout_t *L, float *src) {
  float x0 = L->node_x[0], x1 = L->node_x[L->nx - 1], y0 = L->node_y[0], y1 = L->node_y[L->ny - 1];
  float s[8] = { x0, y0, x1, y0, x1, y1, x0, y1 };
  memcpy(src, s, sizeof s);
}

#ifdef OB_NEON
// ai: arm64's bilinear read in integers at gamma 1 (image_t lin), eight points a call (lin_pts, then lin_get): the
// ai: decode's picture sampler's (sample_grid_lin, which describes it) and the registration mesh's (msh_read8). A point
// ai: comes as sample()'s coordinates less the half pixel, x a 512th of a pixel on (its fraction's top byte is then the
// ai: rounded one); the word reads stay inside the frame only through lin_pts' integer clamps to w - 2 and h - 2.

// ai: a point's word: row y's at r0 + off and row y + 1's at r1 + off, row y's low half into row y + 1's (a b c d), the
// ai: offset the low 32 bits of off (uxtw: no AND) or all of it. In asm: in C the compiler narrows the loads to halves
// ai: and adds an address for each, four instructions more a point. The frame is not written while it is read, so the
// ai: asm takes no memory operand.
static inline __attribute__((always_inline)) uint64_t lin_word_lo(const uint8_t *r0, const uint8_t *r1, uint64_t off) {
  uint64_t t, m;
  __asm__("ldr %w0, [%2, %w4, uxtw]\n\tldr %w1, [%3, %w4, uxtw]\n\tbfxil %w1, %w0, #0, #16" : "=&r"(t), "=&r"(m) : "r"(r0), "r"(r1), "r"(off));
  return m;
}
static inline __attribute__((always_inline)) uint64_t lin_word(const uint8_t *r0, const uint8_t *r1, uint64_t off) {
  uint64_t t, m;
  __asm__("ldr %w0, [%2, %4]\n\tldr %w1, [%3, %4]\n\tbfxil %w1, %w0, #0, #16" : "=&r"(t), "=&r"(m) : "r"(r0), "r"(r1), "r"(off));
  return m;
}
// ai: eight points' offsets (two vectors), fx twice (bytes), fy (Q15)
typedef struct { uint32x4_t i0, i1; uint8x16_t fx; int16x8_t fy; } lin_pts_t;
typedef struct { uint16x8_t wm, hm; uint8x16_t kab, kcd, kf; int16x8_t kc; } lin_k_t;   // kc: w, 257 (lanes 0, 1)
// ai: a frame's constants: the clamps, the TBL indices (a | b, c | d, the fraction bytes), w and 257
static inline lin_k_t lin_k(int w, int h) {
  return (lin_k_t){ vdupq_n_u16((uint16_t)(w - 2)), vdupq_n_u16((uint16_t)(h - 2)),
                    { 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29 }, { 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31 },
                    { 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29 }, { (int16_t)w, 257, 0, 0, 0, 0, 0, 0 } };
}
static inline __attribute__((always_inline)) lin_pts_t lin_pts(const lin_k_t *k, float32x4_t x0, float32x4_t y0, float32x4_t x1, float32x4_t y1) {
  const uint32x4_t xq0 = vcvtq_n_u32_f32(x0, 16), xq1 = vcvtq_n_u32_f32(x1, 16), yq0 = vcvtq_n_u32_f32(y0, 16), yq1 = vcvtq_n_u32_f32(y1, 16);
  const uint16x8_t xi = vminq_u16(vuzp2q_u16(vreinterpretq_u16_u32(xq0), vreinterpretq_u16_u32(xq1)), k->wm);
  const uint16x8_t yi = vminq_u16(vuzp2q_u16(vreinterpretq_u16_u32(yq0), vreinterpretq_u16_u32(yq1)), k->hm);
  const uint16x8_t kw = vreinterpretq_u16_s16(k->kc);
  const uint8x16x2_t xq = { { vreinterpretq_u8_u32(xq0), vreinterpretq_u8_u32(xq1) } };
  return (lin_pts_t){ vmlal_laneq_u16(vmovl_u16(vget_low_u16(xi)), vget_low_u16(yi), kw, 0), vmlal_high_laneq_u16(vmovl_high_u16(xi), yi, kw, 0),
                      vqtbl2q_u8(xq, k->kf), vreinterpretq_s16_u16(vshrq_n_u16(vuzp1q_u16(vreinterpretq_u16_u32(yq0), vreinterpretq_u16_u32(yq1)), 1)) };
}
static inline __attribute__((always_inline)) float32x4x2_t lin_get(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p) {
  const uint64_t p01 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i0), 0), p23 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i0), 1);
  const uint64_t p45 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i1), 0), p67 = vgetq_lane_u64(vreinterpretq_u64_u32(p.i1), 1);
  const uint8x16x2_t m = { { vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lin_word_lo(r0, r1, p01) | lin_word(r0, r1, p01 >> 32) << 32), vcreate_u64(lin_word_lo(r0, r1, p23) | lin_word(r0, r1, p23 >> 32) << 32))),
                             vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lin_word_lo(r0, r1, p45) | lin_word(r0, r1, p45 >> 32) << 32), vcreate_u64(lin_word_lo(r0, r1, p67) | lin_word(r0, r1, p67 >> 32) << 32))) } };
  const uint8x16_t ab = vqtbl2q_u8(m, k->kab), cd = vqtbl2q_u8(m, k->kcd);   // a | b, c | d: eight points each
  const uint8x8_t fx = vget_low_u8(p.fx);
  const uint16x8_t top = vmlsl_u8(vmlal_high_u8(vshll_n_u8(vget_low_u8(ab), 8), ab, p.fx), vget_low_u8(ab), fx);   // modulo 2^16 on the way
  const uint16x8_t bot = vmlsl_u8(vmlal_high_u8(vshll_n_u8(vget_low_u8(cd), 8), cd, p.fx), vget_low_u8(cd), fx);
  // ai: top / 2 + (bot - top) / 2 fy: 0 to 32640, or -1 where top is 1 and bot 0 (the 16-bit lane read as signed)
  const int16x8_t r = vreinterpretq_s16_u16(vsraq_n_u16(vreinterpretq_u16_s16(vqrdmulhq_s16(vreinterpretq_s16_u16(vhsubq_u16(bot, top)), p.fy)), top, 1));
  return (float32x4x2_t){ { vcvtq_n_f32_s32(vmull_laneq_s16(vget_low_s16(r), k->kc, 1), 23), vcvtq_n_f32_s32(vmull_high_laneq_s16(r, k->kc, 1), 23) } };
}
static inline __attribute__((always_inline)) void lin_read(const uint8_t *r0, const uint8_t *r1, const lin_k_t *k, lin_pts_t p, float *o) {
  const float32x4x2_t g = lin_get(r0, r1, k, p);
  vst1q_f32(o, g.val[0]); vst1q_f32(o + 4, g.val[1]);
}
#endif

// ---------------------------------------------------------------- registration mesh

// A node's search judges a position only where the capture holds most of its template. A cell whose sample point (u, v),
// in pixels as sample() takes it, falls outside the capture (u under 0.5 or over w - 0.5, likewise v, or NaN) reads the
// edge's pixels repeated (sample()'s clamp): a little way past the edge they carry on the code's border and its quiet
// light, but they are no measurement, and a template mostly outside correlates with the repeated edge alone. So a
// position with fewer than half its template's cells inside is no candidate (NODE_NONE, under any correlation, and no
// point of the parabola either), and a node none of whose positions is judged keeps the place the neighbours and the
// model give it, as one judged under 0.35 does. Half: the correlation is then mostly of the capture's own pixels.
// ai: The case (run-2026-11-02T00-18-12-969Z frame 287, 2026-11-02): a corner searched again from the anchors' model
// ai: 55 px left of a 1920 x 1920 capture, every cell reading column 0 repeated; its correlations along x differed by
// ai: 1e-5, and a sampler rounding its own way took a spurious 0.55 three modules off where the exact one's path ended
// ai: under 0.35 (78 blocks to 2). Measured on the 11 S26 replays and 5 cells against 977469a: this rule +129 blocks
// ai: (+128 on that replay, +1 on run-2026-11-01T22-40-56-684Z), every other run the same; the cells outside masked out
// ai: of the sums as well lost 16 with half the template judged and 384 with 65 cells (a side node's whole template)
// ai: judged: the masked rim and quiet zone just past the edge moved corners of codes filling the frame by 0.05 px.
static const float NODE_NONE = -2.0f;
#ifndef OB_NEON   // ai: arm64's is msh_in4r, four points at a time on its read's coordinates
static inline int in_capture(const image_t *im, float u, float v) { return u >= 0.5f && u <= im->w - 0.5f && v >= 0.5f && v <= im->h - 0.5f; }
#endif
// Whether every point a stage of a node's search reads is inside the capture with a pixel to spare: the template's cell
// middles (rect r about (cx0, cy0)) moved by offsets in [ax, bx] x [ay, by] modules (the coarse grid's +-1.5, a climb's
// move one step about its centre), through H, then by (pdx, pdy). w positive at the region's four corners holds it on
// the region (w is linear), so the region's image is the convex quad of the corners' images. Inside, every position is
// judged and the stage counts no cells: the same floats as with no rule.
static int node_inside(const image_t *im, const homo_t *H, int cx0, int cy0, const int *r, float pdx, float pdy, float ax, float bx, float ay, float by) {
  const float x0 = cx0 + r[0] + 0.5f + ax, x1 = cx0 + r[1] - 0.5f + bx, y0 = cy0 + r[2] + 0.5f + ay, y1 = cy0 + r[3] - 0.5f + by;
  const float *h = H->h;
  for (int c = 0; c < 4; c++) {
    const float x = c & 1 ? x1 : x0, y = c & 2 ? y1 : y0;
    float u, v;
    if (!(h[6] * x + h[7] * y + 1 > 0)) return 0;
    project(H, x, y, &u, &v);
    if (!(u + pdx >= 1.5f && u + pdx <= im->w - 1.5f && v + pdy >= 1.5f && v + pdy <= im->h - 1.5f)) return 0;
  }
  return 1;
}

#ifdef OB_NEON
// ai: arm64's registration mesh (refine_node, anchor_find) in NEON's own instructions and its own values: each form
// ai: says what it computes and where its floats part from the scalar twins' (node_ncc, the plain loops), which are the
// ai: statement of what each stage computes; verified blocks are its test. msh_read8 is sample() at eight points whose
// ai: coordinates come less (MSH_OX, MSH_OY), the half pixel and the x fraction's rounding (lin_pts), which every caller
// ai: folds into the offset it adds to its points anyway: at gamma 1 the integer read (lin_get: weights on bytes, the
// ai: grey 128 times over on 16-bit lanes, one conversion), otherwise sample4 through the table at the points put back.
// ai: It is sample()'s blend with x's fraction a rounded 256th of a pixel and the value within 2e-5 of the grey range
// ai: (at random points in and past the frame, within 0.0018 of the range of sample()'s, test/mesh_neon.c), past the
// ai: frame's edges too: a point beyond the last column or row is first held a fraction short of it (xm, ym: FMIN), as
// ai: sample() clamps it, where the integer clamps alone (lin_pts') would blend the last two pixels by the point's own
// ai: fraction (it moved corners of a code filling the frame up to 2.4 px and lost three frames of 79 blocks on
// ai: run-2026-11-02T00-15-17-920Z); below 0 FCVTZU holds pixel 0 as sample() does. msh_in4r is in_capture on the
// ai: read's coordinates: 1/512 <= x <= w - 1 + 1/512, 0 <= y <= h - 1 (xl, xh, yh; a NaN fails each compare).
typedef struct { lin_k_t k; float32x4_t xm, ym, xl, xh, yh; } msh_k_t;
static const float MSH_OX = 0.5f - 1.0f / 512, MSH_OY = 0.5f;
static inline msh_k_t msh_k(const image_t *im) {
  return (msh_k_t){ lin_k(im->w, im->h), vdupq_n_f32((float)(im->w - 2) + 0.998f), vdupq_n_f32((float)(im->h - 2) + 0.9999f),
                    vdupq_n_f32(0.5f - MSH_OX), vdupq_n_f32(im->w - 0.5f - MSH_OX), vdupq_n_f32(im->h - 0.5f - MSH_OY) };
}
static inline __attribute__((always_inline)) float32x4x2_t msh_read8(const image_t *im, const msh_k_t *K, float32x4_t x0, float32x4_t y0, float32x4_t x1, float32x4_t y1) {
  if (!im->lin) {
    const float32x4_t ox = vdupq_n_f32(MSH_OX), oy = vdupq_n_f32(MSH_OY);
    return (float32x4x2_t){ { vreinterpretq_f32_s32(sample4(im, vreinterpretq_s32_f32(vaddq_f32(x0, ox)), vreinterpretq_s32_f32(vaddq_f32(y0, oy)))),
                              vreinterpretq_f32_s32(sample4(im, vreinterpretq_s32_f32(vaddq_f32(x1, ox)), vreinterpretq_s32_f32(vaddq_f32(y1, oy)))) } };
  }
  return lin_get(im->px, im->px + im->w - 2, &K->k, lin_pts(&K->k, vminq_f32(x0, K->xm), vminq_f32(y0, K->ym), vminq_f32(x1, K->xm), vminq_f32(y1, K->ym)));
}
static inline __attribute__((always_inline)) uint32x4_t msh_in4r(const msh_k_t *K, float32x4_t x, float32x4_t y) {
  return vandq_u32(vandq_u32(vcleq_f32(K->xl, x), vcleq_f32(x, K->xh)), vandq_u32(vcgeq_f32(y, vdupq_n_f32(0)), vcleq_f32(y, K->yh)));
}
// ai: The homography at four points from its lines' values there (nu, nv, nw: h0 x + h1 y + h2, h3 x + h4 y + h5,
// ai: h6 x + h7 y + 1), which each caller makes from its points' lattice (a row's terms stepped along it, a cell's
// ai: tabled once a node): one reciprocal of nw for both coordinates where project() divides each, the offset (qx, qy)
// ai: added in the same fused multiply-add. (The reciprocal by FRECPE and two Newton steps measured 4 to 9% slower.)
static inline __attribute__((always_inline)) void msh_proj(float32x4_t nu, float32x4_t nv, float32x4_t nw, float32x4_t qx, float32x4_t qy, float32x4_t *x, float32x4_t *y) {
  const float32x4_t r = vdivq_f32(vdupq_n_f32(1), nw);
  *x = vfmaq_f32(qx, nu, r); *y = vfmaq_f32(qy, nv, r);
}
// ai: a lane's sums, node_ncc's; near the capture's edge its count of cells inside the capture
static inline __attribute__((always_inline)) void msh_sums(float32x4_t sv, float32x4_t tk, float32x4_t *s, float32x4_t *q, float32x4_t *t) {
  *s = vaddq_f32(*s, sv); *q = vaddq_f32(*q, vmulq_f32(sv, sv)); *t = vaddq_f32(*t, vmulq_f32(tk, sv));
}
static inline __attribute__((always_inline)) void msh_count(const msh_k_t *K, float32x4_t x, float32x4_t y, float32x4_t *n) {
  *n = vaddq_f32(*n, vbslq_f32(msh_in4r(K, x, y), vdupq_n_f32(1), vdupq_n_f32(0)));
}
// ai: node_ncc at G (4 or 8) offsets (ox[l], oy[l]) at once, a lane each: a lane places, samples and sums the node's
// ai: kept cells with node_ncc's operations; st and vt are the template's sums over those cells (refine_node's, the
// ai: same). Only the cells node_ncc keeps are read. A kept cell k comes as its sign tv[k] and its middle's three lines
// ai: (cu, cv, cw: refine_node's table), so a point's lines are the cell's plus the offset's (du, dv, dw, a lane each);
// ai: (qx, qy): the neighbours' offset (pdx, pdy) less (MSH_OX, MSH_OY). Eight points a read: at G 2 a cell's eight offsets, at G 1
// ai: two cells' four, the second cell's sums kept apart and added at the end (an odd last cell read twice, once summed).
// ai: edge (a constant of each call): the move's positions may read past the capture (node_inside), so each lane also
// ai: counts its cells inside it and is no candidate under nmin of them.
static inline __attribute__((always_inline)) void msh_ncc_g(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, const int G, const int edge, int nmin) {
  const msh_k_t K = msh_k(im);
  const float *h = H->h;
  const float32x4_t px = vdupq_n_f32(qx), py = vdupq_n_f32(qy), z = vdupq_n_f32(0);
  float32x4_t du[2], dv[2], dw[2];
  for (int g = 0; g < 2; g++) {
    const float32x4_t a = vld1q_f32(ox + 4 * g), b = vld1q_f32(oy + 4 * g);
    du[g] = vfmaq_n_f32(vmulq_n_f32(a, h[0]), b, h[1]); dv[g] = vfmaq_n_f32(vmulq_n_f32(a, h[3]), b, h[4]); dw[g] = vfmaq_n_f32(vmulq_n_f32(a, h[6]), b, h[7]);
  }
#define MSH_CELL(k_, g_, X, Y) msh_proj(vaddq_f32(vdupq_n_f32(cu[k_]), du[g_]), vaddq_f32(vdupq_n_f32(cv[k_]), dv[g_]), vaddq_f32(vdupq_n_f32(cw[k_]), dw[g_]), px, py, &X, &Y)
  float32x4_t s[2] = { z, z }, q[2] = { z, z }, t[2] = { z, z }, nn[2] = { z, z }, xa, ya, xb, yb;
  int k = 0;
  if (G == 2) {
    for (; k < cells; k++) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k, 1, xb, yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const float32x4_t tk = vdupq_n_f32(tv[k]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], tk, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tk, &s[1], &q[1], &t[1]);
    }
  } else {
    for (; k + 1 < cells; k += 2) {
      MSH_CELL(k, 0, xa, ya); MSH_CELL(k + 1, 0, xb, yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      const float32x4_t ta = vdupq_n_f32(tv[k]), tb = vdupq_n_f32(tv[k + 1]);
      if (edge) { msh_count(&K, xa, ya, &nn[0]); msh_count(&K, xb, yb, &nn[1]); }
      msh_sums(sv.val[0], ta, &s[0], &q[0], &t[0]); msh_sums(sv.val[1], tb, &s[1], &q[1], &t[1]);
    }
    if (k < cells) {
      MSH_CELL(k, 0, xa, ya);
      const float32x4_t ta = vdupq_n_f32(tv[k]);
      if (edge) msh_count(&K, xa, ya, &nn[0]);
      msh_sums(msh_read8(im, &K, xa, ya, xa, ya).val[0], ta, &s[0], &q[0], &t[0]);
    }
    s[0] = vaddq_f32(s[0], s[1]); q[0] = vaddq_f32(q[0], q[1]); t[0] = vaddq_f32(t[0], t[1]); nn[0] = vaddq_f32(nn[0], nn[1]);
  }
#undef MSH_CELL
  const float32x4_t n = vdupq_n_f32((float)cells), vst = vdupq_n_f32(st), vvt = vdupq_n_f32(vt);
  for (int g = 0; g < G; g++) {
    const float32x4_t vs = vsubq_f32(q[g], vdivq_f32(vmulq_f32(s[g], s[g]), n));
    const float32x4_t c = vdivq_f32(vsubq_f32(t[g], vdivq_f32(vmulq_f32(vst, s[g]), n)), vsqrtq_f32(vmulq_f32(vvt, vs)));
    float32x4_t r = vt <= 0 ? z : vbslq_f32(vcleq_f32(vs, vdupq_n_f32(1e-9f)), z, c);
    if (edge) r = vbslq_f32(vcltq_f32(nn[g], vdupq_n_f32((float)nmin)), vdupq_n_f32(NODE_NONE), r);
    vst1q_f32(out + 4 * g, r);
  }
}
static void msh_ncc(const image_t *im, const homo_t *H, int cells, const float *cu, const float *cv, const float *cw, const float *tv, float st, float vt, float qx, float qy, const float *ox, const float *oy, float *out, int lanes, int edge, int nmin) {
  if (edge) {
    if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 1, nmin);
    else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 1, nmin);
  } else if (lanes > 4) msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 2, 0, nmin);
  else msh_ncc_g(im, H, cells, cu, cv, cw, tv, st, vt, qx, qy, ox, oy, out, 1, 0, nmin);
}
#endif

#ifndef OB_NEON   // ai: arm64's is msh_ncc's lanes
// Normalized correlation of a node's known cells against the image, with the node shifted by
// (ox, oy) modules from where the current estimate puts it.
// r: the patch as cell offsets from (cx0, cy0), x from r[0] to r[1] and y from r[2] to r[3], ends excluded.
// Every cell is placed through the homography itself, then moved by what the neighbours say (pdx, pdy). A local
// affine map at the node was used until 2026-09-23 and is right only while the template is small in pixels: a
// corner template is 12 modules, 31 px on a 286-module symbol at 1080 fill 0.7 and 96 px on a 94-module one, and
// at 30 degrees this camera's perspective changes scale by about 6% across 96 px, which misplaces the template's
// far cells by over a pixel. Measured: the refined corners of LIZARD-16 at 94 modules came out 1.7 px off the
// truth where the unrefined mark quad was 0.6.
// edge: the position may read past the capture (node_inside), so each cell's point is placed again (project, the same
// floats as the read's) and those inside counted; nmin: the cells inside a judged position needs.
static float node_ncc(const ob_layout_t *L, const image_t *im, int cx0, int cy0, const int *r, const homo_t *H, float pdx, float pdy, float ox, float oy, int edge, int nmin) {
  float st = 0, ss = 0, stt = 0, sss = 0, sts = 0;
  int n = 0, nin = 0, lo = r[0], hi = r[1], ylo = r[2], yhi = r[3], side = hi - lo;
  // Image reads first, four to a vector along each row of the patch; the sums then run over them
  // in the original order, so the correlation comes out the same float.
  float S[OB_FB * OB_FB];
  for (int dy = ylo; dy < yhi; dy++) {
    const float my = cy0 + dy + 0.5f + oy;
    int dx = lo;
#ifdef OB_SIMD
    const v128_t vy = wasm_f32x4_splat(my);
    for (; dx + 4 <= hi; dx += 4) {
      int cx = cx0 + dx;
      const v128_t mx = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_make((float)cx, (float)(cx + 1), (float)(cx + 2), (float)(cx + 3)), wasm_f32x4_splat(0.5f)), wasm_f32x4_splat(ox));
      v128_t u, v;
      project4(H, mx, vy, &u, &v);
      wasm_v128_store(S + (dy - ylo) * side + dx - lo, sample4(im, wasm_f32x4_add(u, wasm_f32x4_splat(pdx)), wasm_f32x4_add(v, wasm_f32x4_splat(pdy))));
    }
#endif
    for (; dx < hi; dx++) { const float mx = cx0 + dx + 0.5f + ox; float u, v; project(H, mx, my, &u, &v); S[(dy - ylo) * side + dx - lo] = sample(im, u + pdx, v + pdy); }
  }
  for (int dy = ylo; dy < yhi; dy++) for (int dx = lo; dx < hi; dx++) {
    int kind = L->kind[(cy0 + dy) * L->w + cx0 + dx];
    if (kind == CELL_DATA) continue;   // the picture inside the border: not the border's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    if (edge) { float u, v; project(H, cx0 + dx + 0.5f + ox, cy0 + dy + 0.5f + oy, &u, &v); nin += in_capture(im, u + pdx, v + pdy); }
    float t = (kind & 3) == CELL_DARK ? -1.0f : 1.0f, s = S[(dy - ylo) * side + dx - lo];
    st += t; ss += s; stt += t * t; sss += s * s; sts += t * s; n++;
  }
  if (edge && nin < nmin) return NODE_NONE;
  float vt = stt - st * st / n, vs = sss - ss * ss / n;
  if (vt <= 0 || vs <= 1e-9f) return 0;
  return (sts - st * ss / n) / sqrtf(vt * vs);
}
#endif


static void refine_node(const ob_layout_t *L, const image_t *im, const homo_t *H, int i, int j, node_t *nodes, const struct lensm *lm) {
  const int cx0 = (int)L->node_x[i], cy0 = (int)L->node_y[j];
  const float mcx = cx0 + 0.5f, mcy = cy0 + 0.5f;
  // A thin frame's template is as long as the border lets it be: thirteen modules along, the border's
  // five across, and an L of both at a corner. Small symbols need the area; 5 x 5 is 8 px there.
  int rect[4];
  {
    int ex = (j == 0 || j == L->ny - 1) ? 6 : 2, ey = (i == 0 || i == L->nx - 1) ? 6 : 2;
    // A corner that carries a mark is a different thing to match. Six modules either way stays INSIDE a 12-module
    // mark, so the template is a dark block with one module of light rim, and once blur has closed the mark's gap
    // there is nothing in it to lock to: measured at 2 px of defocus, refining the corners took a marked symbol
    // from 6 blocks of 96 to 2 where it takes an unmarked one from 19 to 35. Reaching a module past the mark's
    // inner edge puts the square's own edges in the template, which is the strongest feature the symbol has.
    if (ex == 6 && ey == 6) { ex = L->corner - 1 < OB_FB - 3 ? L->corner - 1 : OB_FB - 3; ey = ex; }
    rect[0] = -ex < -cx0 ? -cx0 : -ex; rect[1] = ex + 1 > L->w - cx0 ? L->w - cx0 : ex + 1;
    rect[2] = -ey < -cy0 ? -cy0 : -ey; rect[3] = ey + 1 > L->h - cy0 ? L->h - cy0 : ey + 1;
  }
  // Start from what the already-measured neighbours say the local error is.
  float pdx = 0, pdy = 0;
  int pn = 0;
  for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
    int ii = i + di, jj = j + dj;
    if (ii < 0 || jj < 0 || ii >= L->nx || jj >= L->ny || !nodes[jj * L->nx + ii].done) continue;
    float rx = nodes[jj * L->nx + ii].dx, ry = nodes[jj * L->nx + ii].dy;
    // ai: with the anchors' model, the neighbour's residual from it (the model carries the lens, the neighbours the rest)
    if (lm) { float u, v, mx, my; project(H, L->node_x[ii], L->node_y[jj], &u, &v); lensm_at(lm, u, v, &mx, &my); rx -= mx; ry -= my; }
    pdx += rx; pdy += ry; pn++;
  }
  if (pn) { pdx /= pn; pdy /= pn; }
  if (lm) { float u, v, mx, my; project(H, L->node_x[i], L->node_y[j], &u, &v); lensm_at(lm, u, v, &mx, &my); pdx += mx; pdy += my; }
  // ai: near the capture's edge a position judged where most of its template is inside it (the rule above); GM marks
  // ai: the coarse grid's points inside
  const int edge = !node_inside(im, H, cx0, cy0, rect, pdx, pdy, -1.5f, 1.5f, -1.5f, 1.5f);
  float px, py, ax, ay, bx, by;
  project(H, mcx, mcy, &px, &py); project(H, mcx + 1, mcy, &ax, &ay); project(H, mcx, mcy + 1, &bx, &by);
  // The node's local Jacobian, only to turn the offset found in modules into pixels at the end.
  float jxx = ax - px, jxy = ay - py, jyx = bx - px, jyy = by - py;
  // The coarse search, 7 x 7 offsets of half a module. A shift of a whole module puts every cell where its
  // neighbour was, so the 49 correlations read the same image points many times over. They are read once, on a
  // half-module grid, and each correlation picks its cells out of it: the same floats summed in the same order
  // as node_ncc would, for about a seventh of the reads.
  enum { GMAX = 2 * OB_FB + 5 };
  float G[GMAX * GMAX + 8], GM[GMAX * GMAX + 8], tv[OB_FB * OB_FB];   // + 8: arm64's last row written eight floats at a time; the eighth lane of its sums reads one float past the grid
  int at[OB_FB * OB_FB], cells = 0;
  const int lo = rect[0], ylo = rect[2], side = rect[1] - rect[0], tall = rect[3] - rect[2], GW = 2 * side + 5, GH = 2 * tall + 5;
  // Placed through the homography, as node_ncc does and for the same reason.
#ifdef OB_NEON
  // ai: a row eight points a read: its last read runs on into the next row's first floats, which that row writes
  // ai: again (the last row's into G's spare floats, zeroed below); near the capture's edge GM marks the points inside
  const msh_k_t K = msh_k(im);
  const float32x4_t qx = vdupq_n_f32(pdx - MSH_OX), qy = vdupq_n_f32(pdy - MSH_OY), z = vdupq_n_f32(0), one = vdupq_n_f32(1);
  const float *hh = H->h;
  for (int gy = 0; gy < GH; gy++) {
    const float my = (float)(2 * (cy0 + ylo) + gy - 2) * 0.5f;
    const float32x4_t ru = vdupq_n_f32(hh[1] * my + hh[2]), rv = vdupq_n_f32(hh[4] * my + hh[5]), rw = vdupq_n_f32(hh[7] * my + 1), lane = { 0, 1, 2, 3 };
    for (int gx = 0; gx < GW; gx += 8) {
      const float32x4_t ma = vmulq_f32(vaddq_f32(vdupq_n_f32((float)(2 * (cx0 + lo) + gx - 2)), lane), vdupq_n_f32(0.5f)), mb = vaddq_f32(ma, vdupq_n_f32(2));
      float32x4_t xa, ya, xb, yb;
      msh_proj(vfmaq_n_f32(ru, ma, hh[0]), vfmaq_n_f32(rv, ma, hh[3]), vfmaq_n_f32(rw, ma, hh[6]), qx, qy, &xa, &ya);
      msh_proj(vfmaq_n_f32(ru, mb, hh[0]), vfmaq_n_f32(rv, mb, hh[3]), vfmaq_n_f32(rw, mb, hh[6]), qx, qy, &xb, &yb);
      const float32x4x2_t sv = msh_read8(im, &K, xa, ya, xb, yb);
      if (edge) { vst1q_f32(GM + gy * GW + gx, vbslq_f32(msh_in4r(&K, xa, ya), one, z)); vst1q_f32(GM + gy * GW + gx + 4, vbslq_f32(msh_in4r(&K, xb, yb), one, z)); }
      vst1q_f32(G + gy * GW + gx, sv.val[0]); vst1q_f32(G + gy * GW + gx + 4, sv.val[1]);
    }
  }
#else
  for (int gy = 0; gy < GH; gy++) {
    const float my = (float)(2 * (cy0 + ylo) + gy - 2) * 0.5f;
    int gx = 0;
#ifdef OB_SIMD
    const v128_t vy = wasm_f32x4_splat(my);
    for (; gx + 4 <= GW; gx += 4) {
      const int k = 2 * (cx0 + lo) + gx - 2;
      const v128_t mx = wasm_f32x4_mul(wasm_f32x4_make((float)k, (float)(k + 1), (float)(k + 2), (float)(k + 3)), wasm_f32x4_splat(0.5f));
      v128_t u, v;
      project4(H, mx, vy, &u, &v);
      const v128_t x = wasm_f32x4_add(u, wasm_f32x4_splat(pdx)), y = wasm_f32x4_add(v, wasm_f32x4_splat(pdy)), lo5 = wasm_f32x4_splat(0.5f);
      wasm_v128_store(G + gy * GW + gx, sample4(im, x, y));
      if (edge) {
        const v128_t in = wasm_v128_and(wasm_v128_and(wasm_f32x4_le(lo5, x), wasm_f32x4_le(x, wasm_f32x4_splat(im->w - 0.5f))), wasm_v128_and(wasm_f32x4_le(lo5, y), wasm_f32x4_le(y, wasm_f32x4_splat(im->h - 0.5f))));
        wasm_v128_store(GM + gy * GW + gx, wasm_v128_and(wasm_f32x4_splat(1), in));
      }
    }
#endif
    for (; gx < GW; gx++) {
      const float mx = (float)(2 * (cx0 + lo) + gx - 2) * 0.5f;
      float u, v;
      project(H, mx, my, &u, &v);
      G[gy * GW + gx] = sample(im, u + pdx, v + pdy); GM[gy * GW + gx] = !edge || in_capture(im, u + pdx, v + pdy) ? 1.0f : 0;
    }
  }
#endif
  float st = 0, stt = 0;
#ifdef OB_NEON
  float cu[OB_FB * OB_FB], cv[OB_FB * OB_FB], cw[OB_FB * OB_FB];   // ai: a kept cell's middle (node_ncc's cx0 + dx + 0.5f) through H's three lines
#endif
  for (int dy = 0; dy < tall; dy++) for (int dx = 0; dx < side; dx++) {
    int kind = L->kind[(cy0 + ylo + dy) * L->w + cx0 + lo + dx];
    if (kind == CELL_DATA) continue;   // the picture inside the border: not the border's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    tv[cells] = (kind & 3) == CELL_DARK ? -1.0f : 1.0f; at[cells] = (2 * dy + 3) * GW + 2 * dx + 3;
#ifdef OB_NEON
    const float mx = cx0 + lo + dx + 0.5f, my = cy0 + ylo + dy + 0.5f;
    cu[cells] = H->h[0] * mx + H->h[1] * my + H->h[2]; cv[cells] = H->h[3] * mx + H->h[4] * my + H->h[5]; cw[cells] = H->h[6] * mx + H->h[7] * my + 1;
#endif
    st += tv[cells]; stt += tv[cells] * tv[cells]; cells++;
  }
  const float vt = stt - st * st / cells;
  const int nmin = (cells + 1) / 2;
  float best = NODE_NONE, box = 0, boy = 0;
  for (int k = 0; k < 4; k++) G[GW * GH + k] = GM[GW * GH + k] = 0;
  for (int b = -3; b <= 3; b++) {
    // The seven offsets a of one b sit side by side in the grid, so their sums run as lanes: each lane adds the
    // same cells in the same order as a loop of its own would. Near the capture's edge a lane's cells inside the
    // capture are counted too (SN, from GM), in a pass of their own.
    float SS[8], SQ[8], ST[8], SN[8];
#ifdef OB_SIMD
    const float *g0 = G + b * GW - 3, *m0 = GM + b * GW - 3;
    v128_t s0 = wasm_f32x4_splat(0), s1 = s0, q0 = s0, q1 = s0, t0 = s0, t1 = s0, n0 = s0, n1 = s0;
    for (int k = 0; k < cells; k++) {
      const v128_t v0 = wasm_v128_load(g0 + at[k]), v1 = wasm_v128_load(g0 + at[k] + 4), t = wasm_f32x4_splat(tv[k]);
      s0 = wasm_f32x4_add(s0, v0); s1 = wasm_f32x4_add(s1, v1); q0 = wasm_f32x4_add(q0, wasm_f32x4_mul(v0, v0)); q1 = wasm_f32x4_add(q1, wasm_f32x4_mul(v1, v1));
      t0 = wasm_f32x4_add(t0, wasm_f32x4_mul(t, v0)); t1 = wasm_f32x4_add(t1, wasm_f32x4_mul(t, v1));
    }
    if (edge) for (int k = 0; k < cells; k++) { n0 = wasm_f32x4_add(n0, wasm_v128_load(m0 + at[k])); n1 = wasm_f32x4_add(n1, wasm_v128_load(m0 + at[k] + 4)); }
    wasm_v128_store(SS, s0); wasm_v128_store(SS + 4, s1); wasm_v128_store(SQ, q0); wasm_v128_store(SQ + 4, q1); wasm_v128_store(ST, t0); wasm_v128_store(ST + 4, t1);
    wasm_v128_store(SN, n0); wasm_v128_store(SN + 4, n1);
#else
    for (int a = -3; a <= 3; a++) {
      const float *g0 = G + b * GW + a, *m0 = GM + b * GW + a;
      float ss = 0, sss = 0, sts = 0, sn = 0;
      for (int k = 0; k < cells; k++) { const float v = g0[at[k]]; ss += v; sss += v * v; sts += tv[k] * v; }
      if (edge) for (int k = 0; k < cells; k++) sn += m0[at[k]];
      SS[a + 3] = ss; SQ[a + 3] = sss; ST[a + 3] = sts; SN[a + 3] = sn;
    }
#endif
    for (int a = -3; a <= 3; a++) {
      const float ss = SS[a + 3], vs = SQ[a + 3] - ss * ss / cells;
      const float sc = edge && SN[a + 3] < nmin ? NODE_NONE : vt <= 0 || vs <= 1e-9f ? 0 : (ST[a + 3] - st * ss / cells) / sqrtf(vt * vs);
      if (sc > best) { best = sc; box = a * 0.5f; boy = b * 0.5f; }
    }
  }
  // Hill-climb at a quarter module, then an eighth, then a parabola through the peak. A correlation already
  // known is not taken again: the centre always is, and after a move so are the cells the old grid shares.
  float g[3][3], step = 0.25f;
  int have[3][3] = { { 0 } }, ma = 1, mb = 1;   // (ma, mb): the cell the last step moved to, not yet made the centre
  g[1][1] = best; have[1][1] = 1;
  for (int pass = 0; pass < 2; pass++, step *= 0.5f) {
    for (int move = 0; move < 8; move++) {
      if (ma != 1 || mb != 1) {
        float ng[3][3]; int nh[3][3];
        for (int b2 = 0; b2 < 3; b2++) for (int a2 = 0; a2 < 3; a2++) { int sb = b2 + mb - 1, sa = a2 + ma - 1, in = sb >= 0 && sb < 3 && sa >= 0 && sa < 3; nh[b2][a2] = in && have[sb][sa]; ng[b2][a2] = in ? g[sb][sa] : 0; }
        memcpy(g, ng, sizeof g); memcpy(have, nh, sizeof have); ma = mb = 1;
      }
      // ai: whether this move's positions may read past the capture (the climb's edge)
      const int ce = !node_inside(im, H, cx0, cy0, rect, pdx, pdy, box - step, box + step, boy - step, boy + step);
#ifdef OB_NEON
      {
        // ai: the correlations this move lacks, a lane each (lanes past them repeat the first and are dropped)
        float ox[8], oy[8], got[8];
        int cell[8], m = 0;
        for (int b2 = -1; b2 <= 1; b2++) for (int a2 = -1; a2 <= 1; a2++) if (!have[b2 + 1][a2 + 1]) { ox[m] = box + a2 * step; oy[m] = boy + b2 * step; cell[m++] = (b2 + 1) * 3 + a2 + 1; }
        if (m) {
          for (int l = m; l < 8; l++) { ox[l] = ox[0]; oy[l] = oy[0]; }
          msh_ncc(im, H, cells, cu, cv, cw, tv, st, vt, pdx - MSH_OX, pdy - MSH_OY, ox, oy, got, m, ce, nmin);
          for (int l = 0; l < m; l++) { g[cell[l] / 3][cell[l] % 3] = got[l]; have[cell[l] / 3][cell[l] % 3] = 1; }
        }
      }
#else
      for (int b2 = -1; b2 <= 1; b2++) for (int a2 = -1; a2 <= 1; a2++) if (!have[b2 + 1][a2 + 1]) { g[b2 + 1][a2 + 1] = node_ncc(L, im, cx0, cy0, rect, H, pdx, pdy, box + a2 * step, boy + b2 * step, ce, nmin); have[b2 + 1][a2 + 1] = 1; }
#endif
      int ba = 1, bb = 1;
      for (int b2 = 0; b2 < 3; b2++) for (int a2 = 0; a2 < 3; a2++) if (g[b2][a2] > g[bb][ba]) { ba = a2; bb = b2; }
      if (ba == 1 && bb == 1) break;
      box += (ba - 1) * step; boy += (bb - 1) * step; ma = ba; mb = bb;
    }
    if (pass == 0) {
      // The finer pass starts from the centre alone. If the moves ran out, that is the cell last moved to.
      float c = g[mb][ma];
      memset(have, 0, sizeof have); g[1][1] = c; have[1][1] = 1; ma = mb = 1;
    }
  }
  best = g[1][1];
  float dxx = g[1][0] - 2 * g[1][1] + g[1][2], dyy = g[0][1] - 2 * g[1][1] + g[2][1];
  if (dxx < -1e-6f && g[1][0] > NODE_NONE && g[1][2] > NODE_NONE) box += 0.125f * 0.5f * (g[1][0] - g[1][2]) / dxx;
  if (dyy < -1e-6f && g[0][1] > NODE_NONE && g[2][1] > NODE_NONE) boy += 0.125f * 0.5f * (g[0][1] - g[2][1]) / dyy;
  node_t *nd = &nodes[j * L->nx + i];
  nd->done = 1; nd->score = best;
  if (best < 0.35f) { nd->dx = pdx; nd->dy = pdy; return; }
  nd->dx = pdx + box * jxx + boy * jyx;
  nd->dy = pdy + box * jxy + boy * jyy;
}

// The mesh (any mesh but 0): the interior of a code that carries marks on its border only, computed from the border. Averaging
// neighbours inward is wrong for the two errors that matter: a lens's radial distortion, once the corners are pinned,
// is largest along the edges and zero in the middle, which no average of edge values reproduces, and rolling-shutter
// jello is a sideways shift that depends on the image row alone. So the border's residuals are fitted with a model of
// the lens and of what the homography missed (below), per frame, nothing calibrated, and what the fit leaves is carried
// inward by a Coons patch, the blend of the four border curves, which is exact for any field that is a function of the
// row plus one of the column.
// ai: The lens about its axis (2026-10-17; STATUS "The lens about its axis"). What the border nodes say H
// ai: missed, fitted as the radial terms k1 to kK about the camera's optical axis, k |q|^(2k) q (the radial part of
// ai: Android's lens form, linearised about H), beside the first-order change of the homography itself (an affine map,
// ai: a shift and the two perspective terms -(g . q) q, which H's four corners, distorted points, leave), q a node's
// ai: place about the axis over the farthest image corner's distance from it. The Coons patch below carries inward
// ai: exactly every term but the higher radial ones' (any cubic in the lattice's coordinates is affine in one of them
// ai: for each value of the other, so k1's |q|^2 q and the homography's terms are its own), so those are what the fit
// ai: is for: a lens about an axis off the code's middle (a 2:1 half's is the middle of its inner edge) bent the main
// ai: camera's picture 2.3 samples off in the middle of the code, where one term about the image's centre left it.
// ai: Two rules decide K (0 to 3), each frame on its own evidence:
// ai: - The border vouches only for the radii it spans. With the axis outside the code's quad (a 2:1 half's), every
// ai:   interior point's distance from it lies between the border's least and greatest (a distance's extremes over a
// ai:   convex region not holding the point are on its boundary), so the radial terms interpolate. With the axis inside
// ai:   (a 1:1 crop's), the interior's radii run down to 0, below the band from a side's middle to a corner that the
// ai:   border sees, and higher terms fitted there extrapolate: on the three 1:1 recordings of 2026-09, K chosen by the
// ai:   fit alone cost 20 to 43% of the blocks. So K is at most 1 there, a term the patch carries anyway.
// ai: - Within that, the K the evidence pays for, by AICc (below).
// ai: The axis is the image's centre unless the caller says where the camera's axis falls in this crop
// ai: (lizard_set_axis). What the fit leaves is carried inward by the same Coons patch.
enum { LENS_TERMS = 3, LENS_P = 8 + LENS_TERMS };
_Static_assert((int)OB_ANCH_MIN >= (int)LENS_P, "the anchors the geometry takes are at least the lens model's terms (internal.h)");
static int solve_lens(double A[LENS_P][LENS_P + 1], int n) {
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (fabs(A[p][c]) < 1e-14) return 0;
    for (int k = 0; k <= LENS_P; k++) { double t = A[c][k]; A[c][k] = A[p][k]; A[p][k] = t; }
    for (int r = 0; r < n; r++) {
      if (r == c) continue;
      const double f = A[r][c] / A[c][c];
      for (int k = c; k <= LENS_P; k++) A[r][k] -= f * A[c][k];
    }
  }
  return 1;
}
// ai: a node's rows of the model at q = (x, y): x, y, the shift, the two perspective terms, then the K radial terms
static void lens_rows(double x, double y, int K, double *rx, double *ry) {
  const double ex[8] = { x, y, 0, 0, 1, 0, -x * x, -x * y }, ey[8] = { 0, 0, x, y, 0, 1, -x * y, -y * y };
  for (int a = 0; a < 8; a++) { rx[a] = ex[a]; ry[a] = ey[a]; }
  const double s = x * x + y * y;
  double sk = s;
  for (int k = 0; k < K; k++, sk *= s) { rx[8 + k] = sk * x; ry[8 + k] = sk * y; }
}
// ai: the model with K radial terms fitted by least squares on the used nodes; its coefficients in t
static int lens_fit(int K, int nn, const int *use, const float *qx, const float *qy, const node_t *nodes, double *t) {
  const int p = 8 + K;
  double A[LENS_P][LENS_P + 1], rx[LENS_P], ry[LENS_P];
  memset(A, 0, sizeof A);
  int n = 0;
  for (int m = 0; m < nn; m++) {
    if (!use[m]) continue;
    n++;
    lens_rows(qx[m], qy[m], K, rx, ry);
    for (int a = 0; a < p; a++) { for (int b = 0; b < p; b++) A[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A[a][LENS_P] += rx[a] * nodes[m].dx + ry[a] * nodes[m].dy; }
  }
  for (int a = 0; a < p; a++) A[a][a] += 1e-12;
  // ai: the right-hand side sits in column LENS_P whatever p is; the solve reads only the first p rows and columns
  if (2 * n <= p || !solve_lens(A, p)) return 0;
  for (int a = 0; a < p; a++) t[a] = A[a][LENS_P] / A[a][a];
  return 1;
}
static void border_fill_axis(const ob_layout_t *L, const homo_t *H, int iw, int ih, node_t *nodes) {
  const int nx = L->nx, ny = L->ny, nn = nx * ny;
  float ax = ob_axis[0], ay = ob_axis[1];
  if (!(ax == ax) || !(ay == ay)) { ax = iw * 0.5f; ay = ih * 0.5f; }
  double R = 0;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - (double)ax, (c & 2 ? ih : 0) - (double)ay); if (d > R) R = d; }
  if (!(R > 0)) R = 1;
  float *qx = malloc((size_t)nn * sizeof(float)), *qy = malloc((size_t)nn * sizeof(float));
  int *use = malloc((size_t)nn * sizeof(int));
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    const int k = j * nx + i;
    qx[k] = (float)((u - ax) / R); qy[k] = (float)((v - ay) / R);
    const node_t *nd = &nodes[k];
    const int corner = (i == 0 || i == nx - 1) && (j == 0 || j == ny - 1);
    use[k] = nd->done && (corner || (L->node_mark[k] && nd->score >= 0.35f));
  }
  // ai: is the axis (q = 0) inside the code's quad? Its corners in ring order; the axis is inside where it sits on the
  // ai: same side of all four edges.
  const int cn[4] = { 0, nx - 1, nn - 1, nn - nx };
  int pos = 0, neg = 0;
  for (int c = 0; c < 4; c++) {
    const int a = cn[c], b = cn[(c + 1) & 3];
    const double cr = (double)qx[a] * qy[b] - (double)qy[a] * qx[b];   // ai: (corner a - 0) x (corner b - 0)
    pos += cr > 0; neg += cr < 0;
  }
  const int kmax = (pos == 4 || neg == 4) ? 1 : LENS_TERMS;
  // ai: The model by AICc over the used nodes' 2n residuals (a node's dx and dy each): K = 0 to kmax radial terms;
  // ai: H's fit kept as it is where none fits. Beside it a fit on three sides scored on the fourth chose as often on
  // ai: the simulator and less well on every 2:1 replay (up to 2.5 blocks a half; STATUS "The lens about its axis").
  double rx[LENS_P], ry[LENS_P];
  int K = -1;
  double best = INFINITY;
  for (int k = 0; k <= kmax; k++) {
    const int p = 8 + k;
    double t[LENS_P] = { 0 }, rss = 0;
    int n = 0;
    if (!lens_fit(k, nn, use, qx, qy, nodes, t)) continue;
    for (int m = 0; m < nn; m++) {
      if (!use[m]) continue;
      n += 2;
      lens_rows(qx[m], qy[m], k, rx, ry);
      double fx = 0, fy = 0;
      for (int a = 0; a < p; a++) { fx += t[a] * rx[a]; fy += t[a] * ry[a]; }
      rss += (nodes[m].dx - fx) * (nodes[m].dx - fx) + (nodes[m].dy - fy) * (nodes[m].dy - fy);
    }
    if (n - p - 1 <= 0) continue;
    const double aicc = n * log(rss / n + 1e-30) + 2.0 * p + 2.0 * p * (p + 1) / (n - p - 1);
    if (aicc < best) { best = aicc; K = k; }
  }
  double th[LENS_P] = { 0 };
  if (K >= 0 && !lens_fit(K, nn, use, qx, qy, nodes, th)) K = -1;
  float *mx = malloc((size_t)nn * sizeof(float)), *my = malloc((size_t)nn * sizeof(float));
  for (int k = 0; k < nn; k++) {
    double X = 0, Y = 0;
    if (K >= 0) { lens_rows(qx[k], qy[k], K, rx, ry); for (int a = 0; a < 8 + K; a++) { X += th[a] * rx[a]; Y += th[a] * ry[a]; } }
    mx[k] = (float)X; my[k] = (float)Y;
  }
  #define EX(i, j) (nodes[(j) * nx + (i)].dx - mx[(j) * nx + (i)])
  #define EY(i, j) (nodes[(j) * nx + (i)].dy - my[(j) * nx + (i)])
  for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
    node_t *nd = &nodes[j * nx + i];
    if (L->node_mark[j * nx + i]) continue;
    float u = (L->node_x[i] - L->node_x[0]) / (L->node_x[nx - 1] - L->node_x[0]), v = (L->node_y[j] - L->node_y[0]) / (L->node_y[ny - 1] - L->node_y[0]);
    float fx = (1 - u) * EX(0, j) + u * EX(nx - 1, j) + (1 - v) * EX(i, 0) + v * EX(i, ny - 1)
             - ((1 - u) * (1 - v) * EX(0, 0) + u * (1 - v) * EX(nx - 1, 0) + (1 - u) * v * EX(0, ny - 1) + u * v * EX(nx - 1, ny - 1));
    float fy = (1 - u) * EY(0, j) + u * EY(nx - 1, j) + (1 - v) * EY(i, 0) + v * EY(i, ny - 1)
             - ((1 - u) * (1 - v) * EY(0, 0) + u * (1 - v) * EY(nx - 1, 0) + (1 - u) * v * EY(0, ny - 1) + u * v * EY(nx - 1, ny - 1));
    nd->dx = mx[j * nx + i] + fx; nd->dy = my[j * nx + i] + fy; nd->done = 1; nd->score = 1;
  }
  #undef EX
  #undef EY
  free(qx); free(qy); free(use); free(mx); free(my);
}

// ---------------------------------------------------------------- the anchors

// ai: The lens model over points: q a point's place about the axis over R, d its displacement from H in px; K = 0 to
// ai: LENS_TERMS radial terms by AICc as border_fill_axis chooses them, with no cap inside a 1:1 crop (the anchors sample
// ai: the interior's radii). Robust (isb given): a border point further from the model than the larger of tau_floor
// ai: and three robust spreads (1.4826 MAD of the border's residuals) is dropped and the model refitted, three rounds.
struct lensm { int K; double t[LENS_P], ax, ay, R; };
typedef struct lensm lensm_t;
static void lensm_at(const struct lensm *m, double u, double v, float *dx, float *dy) {
  double rx[LENS_P], ry[LENS_P], fx = 0, fy = 0;
  lens_rows((u - m->ax) / m->R, (v - m->ay) / m->R, m->K, rx, ry);
  for (int a = 0; a < 8 + m->K; a++) { fx += m->t[a] * rx[a]; fy += m->t[a] * ry[a]; }
  *dx = (float)fx; *dy = (float)fy;
}
// ai: the axis and R as border_fill_axis takes them
static void lensm_axis(int iw, int ih, lensm_t *m) {
  float x = ob_axis[0], y = ob_axis[1];
  if (!(x == x) || !(y == y)) { x = iw * 0.5f; y = ih * 0.5f; }
  m->ax = x; m->ay = y; m->R = 0;
  for (int c = 0; c < 4; c++) { const double d = hypot((c & 1 ? iw : 0) - (double)x, (c & 2 ? ih : 0) - (double)y); if (d > m->R) m->R = d; }
  if (!(m->R > 0)) m->R = 1;
}
static int cmp_double(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int lensm_fit(int np, const double *qx, const double *qy, const double *dx, const double *dy, const unsigned char *isb, unsigned char *drop, double tau_floor, lensm_t *m) {
  double *res = isb ? malloc((size_t)np * sizeof(double)) : 0;
  int ok = 0;
  for (int round = 0; round < 4; round++) {
    double best = INFINITY, tb[LENS_P] = { 0 };
    int K = -1;
    // ai: The normal equations summed once, at every term: a model of fewer terms takes their leading block (lens_rows'
    // ai: first 8 + k entries are the same whatever follows, and an entry sums its products in the points' order), and
    // ai: an entry below the diagonal is its transpose's sum (the same products), so the upper triangle is summed and
    // ai: copied down. Each k's equations are the floats a sum of its own made; the block past p is never read (solve_lens
    // ai: works rows below p and each column on its own).
    double A3[LENS_P][LENS_P + 1], rx[LENS_P], ry[LENS_P];
    memset(A3, 0, sizeof A3);
    int n = 0;
    for (int i = 0; i < np; i++) {
      if (drop[i]) continue;
      n++;
      lens_rows(qx[i], qy[i], LENS_TERMS, rx, ry);
      for (int a = 0; a < LENS_P; a++) { for (int b = a; b < LENS_P; b++) A3[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A3[a][LENS_P] += rx[a] * dx[i] + ry[a] * dy[i]; }
    }
    for (int a = 1; a < LENS_P; a++) for (int b = 0; b < a; b++) A3[a][b] = A3[b][a];
    for (int k = 0; k <= LENS_TERMS; k++) {
      const int p = 8 + k;
      double A[LENS_P][LENS_P + 1];
      memcpy(A, A3, sizeof A);
      for (int a = 0; a < p; a++) A[a][a] += 1e-12;
      if (2 * n - p - 1 <= 0 || !solve_lens(A, p)) continue;
      double t[LENS_P] = { 0 }, rss = 0;
      for (int a = 0; a < p; a++) t[a] = A[a][LENS_P] / A[a][a];
      for (int i = 0; i < np; i++) {
        if (drop[i]) continue;
        lens_rows(qx[i], qy[i], k, rx, ry);
        double fx = 0, fy = 0;
        for (int a = 0; a < p; a++) { fx += t[a] * rx[a]; fy += t[a] * ry[a]; }
        rss += (dx[i] - fx) * (dx[i] - fx) + (dy[i] - fy) * (dy[i] - fy);
      }
      const double nn = 2.0 * n, aicc = nn * log(rss / nn + 1e-30) + 2.0 * p + 2.0 * p * (p + 1) / (nn - p - 1);
      if (aicc < best) { best = aicc; K = k; memcpy(tb, t, sizeof tb); }
    }
    if (K < 0) break;
    m->K = K; memcpy(m->t, tb, sizeof tb); ok = 1;
    if (!isb || round == 3) break;
    int nb = 0, changed = 0;
    for (int i = 0; i < np; i++) {
      if (!isb[i]) continue;
      float fx, fy;
      lensm_at(m, qx[i] * m->R + m->ax, qy[i] * m->R + m->ay, &fx, &fy);
      res[nb++] = hypot(dx[i] - fx, dy[i] - fy);
    }
    if (!nb) break;
    double *srt = malloc((size_t)nb * sizeof(double));
    memcpy(srt, res, (size_t)nb * sizeof(double));
    qsort(srt, (size_t)nb, sizeof(double), cmp_double);
    const double tau = fmax(tau_floor, 3 * 1.4826 * srt[nb / 2]);
    free(srt);
    for (int i = 0, b = 0; i < np; i++) {
      if (!isb[i]) continue;
      const unsigned char bad = res[b++] > tau;
      if (bad != drop[i]) { drop[i] = bad; changed++; }
    }
    if (!changed) break;
  }
  free(res);
  return ok;
}

// ai: where module point (X, Y) is in the image: through H and the node mesh when given (ob_map_point), else H and the
// ai: lens model when given, else H alone
typedef struct { float X, Y, dx, dy, peak, u, v; int ok; } anchor_t;
static void anchor_map(const homo_t *H, const lensm_t *lm, float X, float Y, float *u, float *v) {
  project(H, X, Y, u, v);
  if (lm) { float dx, dy; lensm_at(lm, *u, *v, &dx, &dy); *u += dx; *v += dy; }
}

// ai: Every anchor not yet ok found as anchor_map places it: a window resampled on a grid of a / OB_ANCH_TS modules
// ai: about its place, the pattern's normalised correlation at every whole step within ANCH_REACH modules, a parabola each
// ai: way. An anchor's dx, dy is where its middle really is less where H puts it (px), peak its correlation, and it is
// ai: ok where the peak reaches ANCH_PEAK inside the window (a peak on the window's edge may stand short of one beyond
// ai: it). ANCH_PEAK from 771 captures of the published lenses searched three modules wide through H alone
// ai: (research/build/anchors/PLAN.md): every right find (35,293) peaked at 0.669 or more, 0.774 at its 0.1%; the
// ai: wrong ones (1,498, more than a module off) at 0.404 in the median, 0.584 at their 99%; 586 of 792 edge peaks
// ai: wrong. The count of anchors ok. (The pattern inverted, a dark middle and rim, took 20 wrong finds over 0.6 and
// ai: read the anchors four times less precisely in a 2:1 half with no lens, 0.060 modules against 0.014.)
// ai: ANCH_CORNER: how far, in modules, the finder's corner may sit from where the anchors' map puts it before it is
// ai: searched again (the corners the finder lost on the published lenses sat 4 to 63 modules off, the ones it held
// ai: within 1.2; the anchors' map, extrapolated past the last anchor to a corner, is a module or two off on a strong lens)
enum { ANCH_ROUNDS = 6, ANCH_CORNER = 3, ANCH_REACH = 3 };
static const float ANCH_PEAK = 0.6f;
#ifdef OB_NEON
// ai: lensm_at's displacement added to four points (u, v), two lanes of doubles at a time: lens_rows' entries and the
// ai: sum over them in its order, a constant entry's product (t[a] times 0 or 1) taken once as the scalar takes it
static inline void msh_lens4(const struct lensm *m, float32x4_t *u, float32x4_t *v) {
  const double *t = m->t;
  const float64x2_t ax = vdupq_n_f64(m->ax), ay = vdupq_n_f64(m->ay), R = vdupq_n_f64(m->R), zero = vdupq_n_f64(0);
  const float64x2_t t0 = vdupq_n_f64(t[0]), t1 = vdupq_n_f64(t[1]), t2 = vdupq_n_f64(t[2]), t3 = vdupq_n_f64(t[3]), t6 = vdupq_n_f64(t[6]), t7 = vdupq_n_f64(t[7]);
  const float64x2_t x2 = vdupq_n_f64(t[2] * 0.0), x3 = vdupq_n_f64(t[3] * 0.0), x4 = vdupq_n_f64(t[4] * 1.0), x5 = vdupq_n_f64(t[5] * 0.0);
  const float64x2_t y0 = vdupq_n_f64(t[0] * 0.0), y1 = vdupq_n_f64(t[1] * 0.0), y4 = vdupq_n_f64(t[4] * 0.0), y5 = vdupq_n_f64(t[5] * 1.0);
  float32x2_t dx[2], dy[2];
  for (int hf = 0; hf < 2; hf++) {
    const float64x2_t uu = hf ? vcvt_high_f64_f32(*u) : vcvt_f64_f32(vget_low_f32(*u)), vv = hf ? vcvt_high_f64_f32(*v) : vcvt_f64_f32(vget_low_f32(*v));
    const float64x2_t x = vdivq_f64(vsubq_f64(uu, ax), R), y = vdivq_f64(vsubq_f64(vv, ay), R), nx = vnegq_f64(x), ny = vnegq_f64(y);
    float64x2_t fx = vaddq_f64(zero, vmulq_f64(t0, x)), fy = vaddq_f64(zero, y0);
    fx = vaddq_f64(fx, vmulq_f64(t1, y)); fy = vaddq_f64(fy, y1);
    fx = vaddq_f64(fx, x2); fy = vaddq_f64(fy, vmulq_f64(t2, x));
    fx = vaddq_f64(fx, x3); fy = vaddq_f64(fy, vmulq_f64(t3, y));
    fx = vaddq_f64(fx, x4); fy = vaddq_f64(fy, y4);
    fx = vaddq_f64(fx, x5); fy = vaddq_f64(fy, y5);
    fx = vaddq_f64(fx, vmulq_f64(t6, vmulq_f64(nx, x))); fy = vaddq_f64(fy, vmulq_f64(t6, vmulq_f64(nx, y)));
    fx = vaddq_f64(fx, vmulq_f64(t7, vmulq_f64(nx, y))); fy = vaddq_f64(fy, vmulq_f64(t7, vmulq_f64(ny, y)));
    const float64x2_t s = vaddq_f64(vmulq_f64(x, x), vmulq_f64(y, y));
    float64x2_t sk = s;
    for (int k = 0; k < m->K; k++, sk = vmulq_f64(sk, s)) {
      const float64x2_t tk = vdupq_n_f64(t[8 + k]);
      fx = vaddq_f64(fx, vmulq_f64(tk, vmulq_f64(sk, x))); fy = vaddq_f64(fy, vmulq_f64(tk, vmulq_f64(sk, y)));
    }
    dx[hf] = vcvt_f32_f64(fx); dy[hf] = vcvt_f32_f64(fy);
  }
  *u = vaddq_f32(*u, vcombine_f32(dx[0], dx[1])); *v = vaddq_f32(*v, vcombine_f32(dy[0], dy[1]));
}
// ai: anchor_find's window, eight points a read (W is 28, 40 or 64: whole reads, then a half), each placed as
// ai: anchor_map places it and read by msh_read8
static void msh_anchor_window(const image_t *im, const homo_t *H, const struct lensm *lm, float x0, float y0, float h, int r, int W, float *win) {
  const msh_k_t K = msh_k(im);
  const float *hh = H->h;
  const float32x4_t lane = { 0, 1, 2, 3 }, half = vdupq_n_f32(0.5f), vh = vdupq_n_f32(h), vx0 = vdupq_n_f32(x0), ox = vdupq_n_f32(MSH_OX), oy = vdupq_n_f32(MSH_OY), z = vdupq_n_f32(0);
#define MSH_WPT(wx_, u_, v_) do { const float32x4_t X_ = vaddq_f32(vx0, vmulq_f32(vaddq_f32(vaddq_f32(vdupq_n_f32((float)((wx_) - r)), lane), half), vh)); \
                                  msh_proj(vfmaq_n_f32(ru, X_, hh[0]), vfmaq_n_f32(rv, X_, hh[3]), vfmaq_n_f32(rw, X_, hh[6]), z, z, &u_, &v_); \
                                  if (lm) msh_lens4(lm, &u_, &v_); u_ = vsubq_f32(u_, ox); v_ = vsubq_f32(v_, oy); } while (0)
  for (int wy = 0; wy < W; wy++) {
    const float Y = y0 + (wy - r + 0.5f) * h;
    const float32x4_t ru = vdupq_n_f32(hh[1] * Y + hh[2]), rv = vdupq_n_f32(hh[4] * Y + hh[5]), rw = vdupq_n_f32(hh[7] * Y + 1);
    int wx = 0;
    for (; wx + 8 <= W; wx += 8) {
      float32x4_t u0, v0, u1, v1;
      MSH_WPT(wx, u0, v0); MSH_WPT(wx + 4, u1, v1);
      const float32x4x2_t sv = msh_read8(im, &K, u0, v0, u1, v1);
      vst1q_f32(win + wy * W + wx, sv.val[0]); vst1q_f32(win + wy * W + wx + 4, sv.val[1]);
    }
    if (wx + 4 <= W) {
      float32x4_t u0, v0;
      MSH_WPT(wx, u0, v0);
      vst1q_f32(win + wy * W + wx, msh_read8(im, &K, u0, v0, u0, v0).val[0]);
      wx += 4;
    }
    for (; wx < W; wx++) {
      float u, v;
      anchor_map(H, lm, x0 + (wx - r + 0.5f) * h, y0 + (wy - r + 0.5f) * h, &u, &v);
      win[wy * W + wx] = sample(im, u, v);
    }
  }
#undef MSH_WPT
}
// ai: the window's integral images (anchor_find's, its sums in its order): four rows at a time, a row's running sums
// ai: in a lane of doubles (a 4 x 4 block of the window turned so a vector holds a column), each row's entries its
// ai: running sum plus the row above's, as the scalar loop adds them
static void msh_integral(const float *win, int W, double *I1, double *I2) {
  const int W1 = W + 1;
  for (int x = 0; x < W1; x++) I1[x] = I2[x] = 0;
  int wy = 0;
  if (!(W & 3) && W <= 64) {
    double P[4 * 64], Q[4 * 64];   // ai: a row's running sums, row wy + k at [4 wx + k]
    for (; wy + 4 <= W; wy += 4) {
      float64x2_t s01 = vdupq_n_f64(0), s23 = s01, q01 = s01, q23 = s01;
      for (int wx = 0; wx < W; wx += 4) {
        const float32x4_t r0 = vld1q_f32(win + wy * W + wx), r1 = vld1q_f32(win + (wy + 1) * W + wx), r2 = vld1q_f32(win + (wy + 2) * W + wx), r3 = vld1q_f32(win + (wy + 3) * W + wx);
        const float32x4x2_t a = vtrnq_f32(r0, r1), b = vtrnq_f32(r2, r3);
        float32x4_t col[4];   // ai: col[c] = the four rows at wx + c
        col[0] = vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0])); col[1] = vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1]));
        col[2] = vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])); col[3] = vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1]));
        for (int c = 0; c < 4; c++) {
          const float64x2_t v01 = vcvt_f64_f32(vget_low_f32(col[c])), v23 = vcvt_high_f64_f32(col[c]);
          s01 = vaddq_f64(s01, v01); s23 = vaddq_f64(s23, v23); q01 = vaddq_f64(q01, vmulq_f64(v01, v01)); q23 = vaddq_f64(q23, vmulq_f64(v23, v23));
          vst1q_f64(P + 4 * (wx + c), s01); vst1q_f64(P + 4 * (wx + c) + 2, s23); vst1q_f64(Q + 4 * (wx + c), q01); vst1q_f64(Q + 4 * (wx + c) + 2, q23);
        }
      }
      for (int k = 0; k < 4; k++) {
        double *i1 = I1 + (wy + k + 1) * W1, *i2 = I2 + (wy + k + 1) * W1;
        const double *u1 = i1 - W1, *u2 = i2 - W1;
        i1[0] = i2[0] = 0;
        for (int wx = 0; wx < W; wx += 2) {
          vst1q_f64(i1 + wx + 1, vaddq_f64(vld1q_f64(u1 + wx + 1), (float64x2_t){ P[4 * wx + k], P[4 * wx + 4 + k] }));
          vst1q_f64(i2 + wx + 1, vaddq_f64(vld1q_f64(u2 + wx + 1), (float64x2_t){ Q[4 * wx + k], Q[4 * wx + 4 + k] }));
        }
      }
    }
  }
  for (; wy < W; wy++) {
    double s = 0, q = 0;
    I1[(wy + 1) * W1] = I2[(wy + 1) * W1] = 0;
    for (int wx = 0; wx < W; wx++) {
      const double v = win[wy * W + wx];
      s += v; q += v * v;
      I1[(wy + 1) * W1 + wx + 1] = I1[wy * W1 + wx + 1] + s; I2[(wy + 1) * W1 + wx + 1] = I2[wy * W1 + wx + 1] + q;
    }
  }
}
#endif
static int anchor_find(const ob_layout_t *L, const image_t *im, const homo_t *H, const lensm_t *lm, anchor_t *A) {
  const int a = ob_anchor_side(L), TS = OB_ANCH_TS;
  if (a < 1) return 0;
  const float h = (float)a / TS;
  const int r = (int)ceilf((float)ANCH_REACH / h), W = TS + 2 * r, O = 2 * r + 1;
  float *win = malloc((size_t)W * W * sizeof(float)), *C = malloc((size_t)O * O * sizeof(float)), T[OB_ANCH_TS * OB_ANCH_TS];
  double *I1 = malloc((size_t)(W + 1) * (W + 1) * sizeof(double)), *I2 = malloc((size_t)(W + 1) * (W + 1) * sizeof(double));
  float tm = 0, tn = 0;
  for (int y = 0; y < TS; y++) for (int x = 0; x < TS; x++) {
    const float u = (x + 0.5f) * h - a * 0.5f, v = (y + 0.5f) * h - a * 0.5f, dc = fmaxf(fabsf(u), fabsf(v)) / (a * 0.5f);
    T[y * TS + x] = dc < 0.25f || dc >= 0.6f ? 1.0f : -1.0f;
    tm += T[y * TS + x];
  }
  tm /= (float)(TS * TS);
  for (int k = 0; k < TS * TS; k++) { T[k] -= tm; tn += T[k] * T[k]; }
  tn = sqrtf(tn);
  // ai: The pattern is two levels in concentric squares (dc is the larger of the two axes' distances, so dc under 0.25
  // ai: or 0.6 is a square, [i25, j25] or [i6, j6] along each axis), and the window's sum against it at an offset is
  // ai: 2 (S25 - S6) + S (1 - tm), S the sum over the template's box and S25, S6 over the squares': box sums of the
  // ai: window's integral images (in double, its squares' too), a few reads an offset where the whole template was
  // ai: summed (16 x 16 a step, 169 steps an anchor in the 256 ring at three modules, 625 in the 128, 2,401 in the 64)
  int i25 = TS, j25 = -1, i6 = TS, j6 = -1;
  for (int x = 0; x < TS; x++) {
    const float d = fabsf((x + 0.5f) * h - a * 0.5f) / (a * 0.5f);
    if (d < 0.25f) { if (x < i25) i25 = x; j25 = x; }
    if (d < 0.6f) { if (x < i6) i6 = x; j6 = x; }
  }
  const int W1 = W + 1;
#define ABOX(I, x, y, s) ((I)[((y) + (s)) * W1 + (x) + (s)] - (I)[(y) * W1 + (x) + (s)] - (I)[((y) + (s)) * W1 + (x)] + (I)[(y) * W1 + (x)])
  int found = 0;
  for (int j = 0; j < OB_ANCH_K; j++) for (int i = 0; i < OB_ANCH_K; i++) {
    anchor_t *an = &A[j * OB_ANCH_K + i];
    if (an->ok) { found++; continue; }
    const float x0 = (float)ob_anchor_m0(L, i), y0 = (float)ob_anchor_m0(L, j);
    an->X = x0 + a * 0.5f; an->Y = y0 + a * 0.5f; an->peak = -1; an->dx = an->dy = 0;
#ifdef OB_NEON
    msh_anchor_window(im, H, lm, x0, y0, h, r, W, win);
    msh_integral(win, W, I1, I2);
    float best = -2;
    int bx = 0, by = 0;
    {
      // ai: the correlations two offsets a vector of doubles, each lane the scalar's operations; the peak then taken
      // ai: in the scalar's order (the division by 256 as a multiply by 2^-8, the same doubles, measured no faster)
      const float64x2_t k256 = vdupq_n_f64(TS * TS), ktm = vdupq_n_f64(1 - (double)tm), two = vdupq_n_f64(2), tnd = vdupq_n_f64(tn), vmin = vdupq_n_f64(1e-12), zero = vdupq_n_f64(0);
      const int l25 = j25 - i25 + 1, l6 = j6 - i6 + 1;
#define MBOX(I, x, y, s) vaddq_f64(vsubq_f64(vsubq_f64(vld1q_f64((I) + ((y) + (s)) * W1 + (x) + (s)), vld1q_f64((I) + (y) * W1 + (x) + (s))), vld1q_f64((I) + ((y) + (s)) * W1 + (x))), vld1q_f64((I) + (y) * W1 + (x)))
      for (int oy = -r; oy <= r; oy++) {
        const int py = r + oy;
        int ox = -r;
        for (; ox + 1 <= r; ox += 2) {
          const int px = r + ox;
          const float64x2_t s1 = MBOX(I1, px, py, TS), s2 = MBOX(I2, px, py, TS), var = vsubq_f64(s2, vdivq_f64(vmulq_f64(s1, s1), k256));
          const float64x2_t s25 = j25 >= i25 ? MBOX(I1, px + i25, py + i25, l25) : zero, s6 = j6 >= i6 ? MBOX(I1, px + i6, py + i6, l6) : zero;
          const float64x2_t st = vaddq_f64(vmulq_f64(two, vsubq_f64(s25, s6)), vmulq_f64(s1, ktm));
          const float32x2_t c = vcvt_f32_f64(vdivq_f64(st, vmulq_f64(vsqrtq_f64(var), tnd)));
          vst1_f32(C + (oy + r) * O + ox + r, vbsl_f32(vmovn_u64(vcgtq_f64(var, vmin)), c, vdup_n_f32(0)));
        }
        for (; ox <= r; ox++) {
          const int px = r + ox;
          const double s1 = ABOX(I1, px, py, TS), s2 = ABOX(I2, px, py, TS), var = s2 - s1 * s1 / (TS * TS);
          const double s25 = j25 >= i25 ? ABOX(I1, px + i25, py + i25, j25 - i25 + 1) : 0, s6 = j6 >= i6 ? ABOX(I1, px + i6, py + i6, j6 - i6 + 1) : 0;
          const double st = 2 * (s25 - s6) + s1 * (1 - (double)tm);
          C[(oy + r) * O + ox + r] = var > 1e-12 ? (float)(st / (sqrt(var) * tn)) : 0;
        }
      }
#undef MBOX
    }
    for (int oy = -r; oy <= r; oy++) for (int ox = -r; ox <= r; ox++) {
      const float c = C[(oy + r) * O + ox + r];
      if (c > best) { best = c; bx = ox; by = oy; }
    }
#else
    for (int wy = 0; wy < W; wy++) for (int wx = 0; wx < W; wx++) {
      float u, v;
      anchor_map(H, lm, x0 + (wx - r + 0.5f) * h, y0 + (wy - r + 0.5f) * h, &u, &v);
      win[wy * W + wx] = sample(im, u, v);
    }
    for (int x = 0; x < W1; x++) I1[x] = I2[x] = 0;
    for (int wy = 0; wy < W; wy++) {
      double s = 0, q = 0;
      I1[(wy + 1) * W1] = I2[(wy + 1) * W1] = 0;
      for (int wx = 0; wx < W; wx++) {
        const double v = win[wy * W + wx];
        s += v; q += v * v;
        I1[(wy + 1) * W1 + wx + 1] = I1[wy * W1 + wx + 1] + s; I2[(wy + 1) * W1 + wx + 1] = I2[wy * W1 + wx + 1] + q;
      }
    }
    float best = -2;
    int bx = 0, by = 0;
    for (int oy = -r; oy <= r; oy++) for (int ox = -r; ox <= r; ox++) {
      const int px = r + ox, py = r + oy;
      const double s1 = ABOX(I1, px, py, TS), s2 = ABOX(I2, px, py, TS), var = s2 - s1 * s1 / (TS * TS);
      const double s25 = j25 >= i25 ? ABOX(I1, px + i25, py + i25, j25 - i25 + 1) : 0, s6 = j6 >= i6 ? ABOX(I1, px + i6, py + i6, j6 - i6 + 1) : 0;
      const double st = 2 * (s25 - s6) + s1 * (1 - (double)tm);
      const float c = var > 1e-12 ? (float)(st / (sqrt(var) * tn)) : 0;
      C[(oy + r) * O + ox + r] = c;
      if (c > best) { best = c; bx = ox; by = oy; }
    }
#endif
    float sx = 0, sy = 0;
    if (bx > -r && bx < r) { const float l = C[(by + r) * O + bx + r - 1], c = best, rr = C[(by + r) * O + bx + r + 1], d = l - 2 * c + rr; if (d < 0) sx = 0.5f * (l - rr) / d; }
    if (by > -r && by < r) { const float l = C[(by + r - 1) * O + bx + r], c = best, rr = C[(by + r + 1) * O + bx + r], d = l - 2 * c + rr; if (d < 0) sy = 0.5f * (l - rr) / d; }
    an->peak = best;
    float u, v, hu, hv;
    anchor_map(H, lm, an->X + (bx + sx) * h, an->Y + (by + sy) * h, &u, &v);
    project(H, an->X, an->Y, &hu, &hv);
    an->dx = u - hu; an->dy = v - hv; an->u = u; an->v = v;
    an->ok = best >= ANCH_PEAK && bx > -r && bx < r && by > -r && by < r;
    found += an->ok;
  }
  free(win); free(C); free(I1); free(I2);
#undef ABOX
  return found;
}

// ai: the lens model the anchors and the four corners say (the corners' displacement 0: H passes through them), for
// ai: the border's search to start from; 0 where it does not fit
static int anchor_model(const ob_layout_t *L, const homo_t *H, int iw, int ih, const anchor_t *A, lensm_t *m) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  double qx[NA + 4], qy[NA + 4], dx[NA + 4], dy[NA + 4];
  unsigned char drop[NA + 4] = { 0 };
  int np = 0;
  lensm_axis(iw, ih, m);
  for (int c = 0; c < 4; c++) {
    float u, v;
    project(H, L->node_x[(c == 1 || c == 2) ? L->nx - 1 : 0], L->node_y[c >= 2 ? L->ny - 1 : 0], &u, &v);
    qx[np] = (u - m->ax) / m->R; qy[np] = (v - m->ay) / m->R; dx[np] = 0; dy[np] = 0; np++;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float u, v;
    project(H, A[k].X, A[k].Y, &u, &v);
    qx[np] = (u - m->ax) / m->R; qy[np] = (v - m->ay) / m->R; dx[np] = A[k].dx; dy[np] = A[k].dy; np++;
  }
  return lensm_fit(np, qx, qy, dx, dy, 0, drop, 0, m);
}

// ai: A homography through the anchors found (module place to image place), least squares with h33 = 1 on
// ai: coordinates centred and scaled; 0 where they do not hold it (fewer than eight, or within three rows or columns
// ai: of the grid)
static int anchor_homo(const anchor_t *A, homo_t *H) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  int n = 0, rows = 0, cols = 0, rm = 0, cm = 0;
  double mx = 0, my = 0, mu = 0, mv = 0;
  for (int k = 0; k < NA; k++) if (A[k].ok) { n++; mx += A[k].X; my += A[k].Y; mu += A[k].u; mv += A[k].v; rm |= 1 << (k / OB_ANCH_K); cm |= 1 << (k % OB_ANCH_K); }
  for (int b = 0; b < OB_ANCH_K; b++) { rows += (rm >> b) & 1; cols += (cm >> b) & 1; }
  if (n < 8 || rows < 3 || cols < 3) return 0;
  mx /= n; my /= n; mu /= n; mv /= n;
  double sx = 0, su = 0;
  for (int k = 0; k < NA; k++) if (A[k].ok) { sx += hypot(A[k].X - mx, A[k].Y - my); su += hypot(A[k].u - mu, A[k].v - mv); }
  sx = sx / n / 1.4142135623730951; su = su / n / 1.4142135623730951;
  if (!(sx > 0) || !(su > 0)) return 0;
  double M[LENS_P][LENS_P + 1];
  memset(M, 0, sizeof M);
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    const double x = (A[k].X - mx) / sx, y = (A[k].Y - my) / sx, u = (A[k].u - mu) / su, v = (A[k].v - mv) / su;
    const double r0[8] = { x, y, 1, 0, 0, 0, -u * x, -u * y }, r1[8] = { 0, 0, 0, x, y, 1, -v * x, -v * y };
    for (int a = 0; a < 8; a++) { for (int b = 0; b < 8; b++) M[a][b] += r0[a] * r0[b] + r1[a] * r1[b]; M[a][LENS_P] += r0[a] * u + r1[a] * v; }
  }
  if (!solve_lens(M, 8)) return 0;
  double g[9];
  for (int a = 0; a < 8; a++) g[a] = M[a][LENS_P] / M[a][a];
  g[8] = 1;
  // ai: undo the scalings: H = Tu^-1 G Tx, Tx = [1/sx 0 -mx/sx; 0 1/sx -my/sx; 0 0 1], Tu^-1 = [su 0 mu; 0 su mv; 0 0 1]
  const double tx[9] = { 1 / sx, 0, -mx / sx, 0, 1 / sx, -my / sx, 0, 0, 1 }, ti[9] = { su, 0, mu, 0, su, mv, 0, 0, 1 };
  double gt[9], h[9];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { gt[3 * r + c] = 0; for (int k = 0; k < 3; k++) gt[3 * r + c] += g[3 * r + k] * tx[3 * k + c]; }
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) { h[3 * r + c] = 0; for (int k = 0; k < 3; k++) h[3 * r + c] += ti[3 * r + k] * gt[3 * k + c]; }
  if (!(fabs(h[8]) > 1e-12)) return 0;
  for (int a = 0; a < 9; a++) H->h[a] = (float)(h[a] / h[8]);
  return 1;
}

// ai: the lens model the anchors alone say, about H: their displacements from it (the corners left out, so a corner
// ai: the finder misplaced does not bend it), held to the border nodes' robust rule (lensm_fit: a reading further from
// ai: the model than half a module, mpx px, or three robust spreads is dropped and the model fitted again), the dropped
// ai: anchors no longer found; 0 where it does not fit
static int anchor_model_free(const homo_t *H, int iw, int ih, anchor_t *A, lensm_t *m, float mpx) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  double qx[NA], qy[NA], dx[NA], dy[NA];
  unsigned char drop[NA] = { 0 }, isb[NA];
  int np = 0, at[NA];
  memset(isb, 1, sizeof isb);
  lensm_axis(iw, ih, m);
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float hu, hv;
    project(H, A[k].X, A[k].Y, &hu, &hv);
    A[k].dx = A[k].u - hu; A[k].dy = A[k].v - hv;
    qx[np] = (hu - m->ax) / m->R; qy[np] = (hv - m->ay) / m->R; dx[np] = A[k].dx; dy[np] = A[k].dy; at[np] = k; np++;
  }
  if (np < 4 || !lensm_fit(np, qx, qy, dx, dy, isb, drop, 0.5 * mpx, m)) return 0;
  for (int i = 0; i < np; i++) if (drop[i]) A[at[i]].ok = 0;
  return 1;
}

// ai: The code registered from the inside out. The anchors searched three modules wide through the corners'
// ai: homography, then, round after round, those still missing through the homography the anchors found say and the
// ai: lens model they say about it (each round's readings held to the robust rule, anchor_model_free) (an anchor further out than the reach comes within it once the model has the anchors
// ai: nearer the middle). Then each corner judged by that map: one further than ANCH_CORNER modules from where it puts
// ai: the corner (its mark cut by the crop, or its sides bent past the line fit) searched again from where the model
// ai: over the anchors and the kept corners puts it, kept where its template reads 0.35 or more, else the model's place,
// ai: and H made again from the four. (Judging the corners by that joint model instead, dropping outliers round by
// ai: round, kept 82.4% of the exact map's blocks on the hard cells against this rule's 85.6%: a misplaced corner bends
// ai: it.) Then the anchors' model about H (anchor_model) for the border's search. 0 where fewer than OB_ANCH_MIN
// ai: anchors were found (the rounds grow from four).
static int anchor_register(const ob_layout_t *L, const image_t *im, const image_t *rim, homo_t *H, int iw, int ih, float mpx, float *quad, node_t *nodes, anchor_t *A, lensm_t *m) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  homo_t Ha = *H;
  lensm_t mf;
  int have = 0, nf = 0;
  for (int k = 0; k < NA; k++) A[k].ok = 0;
  for (int round = 0; round < ANCH_ROUNDS; round++) {
    const int now = anchor_find(L, im, &Ha, have ? &mf : 0, A);
    if (now == nf || now < 4) break;
    nf = now;
    homo_t Hn;
    if (anchor_homo(A, &Hn)) Ha = Hn;
    if (!(have = anchor_model_free(&Ha, iw, ih, A, &mf, mpx))) break;
    nf = 0;
    for (int k = 0; k < NA; k++) nf += A[k].ok;
  }
  if (!have || nf < OB_ANCH_MIN) return 0;
  const int nx = L->nx, ny = L->ny;
  // ai: the anchors and the corners about the corners' H, the corners' outliers left out round by round
  double qx[NA + 4], qy[NA + 4], dx[NA + 4], dy[NA + 4];
  unsigned char drop[NA + 4] = { 0 };
  int np = 4;
  lensm_t mc;
  lensm_axis(iw, ih, &mc);
  for (int c = 0; c < 4; c++) {
    float u, v;
    project(H, L->node_x[(c == 1 || c == 2) ? nx - 1 : 0], L->node_y[c >= 2 ? ny - 1 : 0], &u, &v);
    qx[c] = (u - mc.ax) / mc.R; qy[c] = (v - mc.ay) / mc.R; dx[c] = dy[c] = 0;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float hu, hv;
    project(H, A[k].X, A[k].Y, &hu, &hv);
    qx[np] = (hu - mc.ax) / mc.R; qy[np] = (hv - mc.ay) / mc.R; dx[np] = A[k].u - hu; dy[np] = A[k].v - hv; np++;
  }
  // ai: each corner against the anchors' own map (Ha and mf: no corner holds it, so a misplaced one cannot bend it);
  // ai: the model about H fitted over the anchors and the corners it keeps, for the search of those it does not
  for (int c = 0; c < 4; c++) {
    float pu, pv, mu, mv;
    project(&Ha, L->node_x[(c == 1 || c == 2) ? nx - 1 : 0], L->node_y[c >= 2 ? ny - 1 : 0], &pu, &pv);
    lensm_at(&mf, pu, pv, &mu, &mv);
    drop[c] = hypotf(pu + mu - quad[2 * c], pv + mv - quad[2 * c + 1]) > ANCH_CORNER * mpx;
  }
  const int fit = lensm_fit(np, qx, qy, dx, dy, 0, drop, 0, &mc);
  float src[8];
  corner_coords(L, src);
  int moved = 0;
  for (int c = 0; fit && c < 4; c++) {
    if (!drop[c]) continue;
    const int i = (c == 1 || c == 2) ? nx - 1 : 0, j = c >= 2 ? ny - 1 : 0;
    float hu, hv, mu, mv;
    project(H, L->node_x[i], L->node_y[j], &hu, &hv);
    lensm_at(&mc, hu, hv, &mu, &mv);
    node_t *nd = &nodes[j * nx + i];
    nd->done = 0;
    refine_node(L, rim, H, i, j, nodes, &mc);
    if (nd->score >= 0.35f) { quad[2 * c] = hu + nd->dx; quad[2 * c + 1] = hv + nd->dy; }
    else { quad[2 * c] = hu + mu; quad[2 * c + 1] = hv + mv; }
    moved++;
  }
  homo_t Hc;
  if (moved && homography(src, quad, &Hc)) {
    *H = Hc;
    for (int c = 0; c < 4; c++) { node_t *nd = &nodes[((c >= 2) ? ny - 1 : 0) * nx + ((c == 1 || c == 2) ? nx - 1 : 0)]; nd->dx = nd->dy = 0; nd->done = 1; }
  }
  for (int k = 0; k < NA; k++) if (A[k].ok) { float hu, hv; project(H, A[k].X, A[k].Y, &hu, &hv); A[k].dx = A[k].u - hu; A[k].dy = A[k].v - hv; }
  return anchor_model(L, H, iw, ih, A, m);
}

// ai: A cubic Hermite through knots t[0..n-1] at x, its slopes from the neighbouring knots (one-sided at the ends), as
// ai: weights: the value is sum w[a] v[b + a], a = 0..3, b = the cell's first knot less one (out-of-range weights 0)
static void anch_herm4(const double *t, int n, double x, int *b, double w[4]) {
  int i = 0;
  while (i < n - 2 && x >= t[i + 1]) i++;
  const double hc = t[i + 1] - t[i], s = (x - t[i]) / hc, s2 = s * s, s3 = s2 * s;
  const double h00 = 2 * s3 - 3 * s2 + 1, h01 = -2 * s3 + 3 * s2, A = (s3 - 2 * s2 + s) * hc, Bc = (s3 - s2) * hc;
  w[0] = w[3] = 0; w[1] = h00; w[2] = h01;
  // ai: the slope at knot i: (v[i+1] - v[i-1]) / (t[i+1] - t[i-1]), or (v[i+1] - v[i]) / hc at the first knot
  if (i > 0) { const double d = t[i + 1] - t[i - 1]; w[0] -= A / d; w[2] += A / d; } else { w[1] -= A / hc; w[2] += A / hc; }
  // ai: the slope at knot i + 1: (v[i+2] - v[i]) / (t[i+2] - t[i]), or (v[i+1] - v[i]) / hc at the last knot
  if (i + 2 < n) { const double d = t[i + 2] - t[i]; w[1] -= Bc / d; w[3] += Bc / d; } else { w[1] -= Bc / hc; w[2] += Bc / hc; }
  *b = i - 1;
}
// ai: The interior from the border and the anchors: the lens model refitted over the border's trusted nodes and the
// ai: anchors found (robust: border nodes the anchors outvote, further than half a module or three robust spreads, are
// ai: dropped and take the model's displacement), the model at every node and the Coons patch of what it leaves on the
// ai: border, then a cubic mesh of what the anchors still see (0 at the lattice's ends, the anchors' shrunk readings at
// ai: their centres). Without four anchors, border_fill_axis as before.
static void border_fill_anchors(const ob_layout_t *L, const homo_t *H, int iw, int ih, node_t *nodes, const anchor_t *A, float mpx) {
  enum { NA = OB_ANCH_K * OB_ANCH_K };
  const int nx = L->nx, ny = L->ny, nn = nx * ny, cap = 2 * (nx + ny) + NA;
  int na = 0;
  for (int k = 0; k < NA; k++) na += A[k].ok;
  if (na < 4) { border_fill_axis(L, H, iw, ih, nodes); return; }
  double *qx = malloc((size_t)cap * sizeof(double)), *qy = malloc((size_t)cap * sizeof(double)), *dx = malloc((size_t)cap * sizeof(double)), *dy = malloc((size_t)cap * sizeof(double));
  unsigned char *isb = calloc((size_t)cap, 1), *drop = calloc((size_t)cap, 1);
  int *at = malloc((size_t)cap * sizeof(int)), np = 0;
  lensm_t lm;
  lensm_axis(iw, ih, &lm);
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    if (i && j && i < nx - 1 && j < ny - 1) continue;
    const int k = j * nx + i, corner = (i == 0 || i == nx - 1) && (j == 0 || j == ny - 1);
    if (!(nodes[k].done && (corner || (L->node_mark[k] && nodes[k].score >= 0.35f)))) continue;
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    qx[np] = (u - lm.ax) / lm.R; qy[np] = (v - lm.ay) / lm.R; dx[np] = nodes[k].dx; dy[np] = nodes[k].dy; isb[np] = 1; at[np] = k; np++;
  }
  for (int k = 0; k < NA; k++) {
    if (!A[k].ok) continue;
    float u, v;
    project(H, A[k].X, A[k].Y, &u, &v);
    qx[np] = (u - lm.ax) / lm.R; qy[np] = (v - lm.ay) / lm.R; dx[np] = A[k].dx; dy[np] = A[k].dy; at[np] = -1 - k; np++;
  }
  if (!lensm_fit(np, qx, qy, dx, dy, isb, drop, 0.5 * mpx, &lm)) {
    free(qx); free(qy); free(dx); free(dy); free(isb); free(drop); free(at);
    border_fill_axis(L, H, iw, ih, nodes); return;
  }
  float *mx = malloc((size_t)nn * sizeof(float)), *my = malloc((size_t)nn * sizeof(float));
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    lensm_at(&lm, u, v, &mx[j * nx + i], &my[j * nx + i]);
  }
  // ai: a border node the anchors outvote reads the model's displacement (so the patch leaves nothing there)
  for (int p = 0; p < np; p++) if (isb[p] && drop[p]) { nodes[at[p]].dx = mx[at[p]]; nodes[at[p]].dy = my[at[p]]; }
  #define EX(i, j) (nodes[(j) * nx + (i)].dx - mx[(j) * nx + (i)])
  #define EY(i, j) (nodes[(j) * nx + (i)].dy - my[(j) * nx + (i)])
  for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
    node_t *nd = &nodes[j * nx + i];
    if (L->node_mark[j * nx + i]) continue;
    float u = (L->node_x[i] - L->node_x[0]) / (L->node_x[nx - 1] - L->node_x[0]), v = (L->node_y[j] - L->node_y[0]) / (L->node_y[ny - 1] - L->node_y[0]);
    float fx = (1 - u) * EX(0, j) + u * EX(nx - 1, j) + (1 - v) * EX(i, 0) + v * EX(i, ny - 1)
             - ((1 - u) * (1 - v) * EX(0, 0) + u * (1 - v) * EX(nx - 1, 0) + (1 - u) * v * EX(0, ny - 1) + u * v * EX(nx - 1, ny - 1));
    float fy = (1 - u) * EY(0, j) + u * EY(nx - 1, j) + (1 - v) * EY(i, 0) + v * EY(i, ny - 1)
             - ((1 - u) * (1 - v) * EY(0, 0) + u * (1 - v) * EY(nx - 1, 0) + (1 - u) * v * EY(0, ny - 1) + u * v * EY(nx - 1, ny - 1));
    nd->dx = mx[j * nx + i] + fx; nd->dy = my[j * nx + i] + fy; nd->done = 1; nd->score = 1;
  }
  #undef EX
  #undef EY
  // ai: what the anchors still see through that mesh, as the sampler blends it, carried inward by the knots
  const double X0 = L->node_x[0], Y0 = L->node_y[0], SX = L->node_x[nx - 1] - X0, SY = L->node_y[ny - 1] - Y0;
  double (*tv)[2] = malloc((size_t)cap * sizeof *tv);
  int *gi = malloc((size_t)cap * sizeof(int)), m = 0;
  for (int p = 0; p < np; p++) {
    if (isb[p]) continue;
    const anchor_t *an = &A[-1 - at[p]];
    const ob_reg_t r = { *H, nodes };
    float u, v, hu, hv;
    ob_map_point(L, &r, an->X, an->Y, &u, &v);
    project(H, an->X, an->Y, &hu, &hv);
    tv[m][0] = an->dx - (u - hu); tv[m][1] = an->dy - (v - hv); gi[m] = -1 - at[p]; m++;
  }
  // ai: An anchor's reading is noisy (0.15 of a sample rms, measured against the truth on the published lenses, with no
  // ai: common offset), and where the mesh already holds the field the knots would carry that noise in. So what the
  // ai: anchors see is shrunk by the share of it that is signal (empirical Bayes): the noise's variance from each
  // ai: reading against the mean of its grid neighbours' (var(d) = s2 (1 + 1 / neighbours) for a field smooth at the
  // ai: grid's spacing; robust, 1.4826 MAD over both components), the readings' power P per component, every residual
  // ai: scaled by max(0, 1 - s2 / P). A strong lens leaves a residual far over the noise and passes whole.
  {
    double g[NA][2], dd[2 * NA], pw = 0;
    unsigned char has[NA] = { 0 };
    int na2 = 0, nd = 0;
    for (int q = 0; q < m; q++) if (gi[q] >= 0) { has[gi[q]] = 1; g[gi[q]][0] = tv[q][0]; g[gi[q]][1] = tv[q][1]; pw += tv[q][0] * tv[q][0] + tv[q][1] * tv[q][1]; na2++; }
    for (int k = 0; k < NA; k++) {
      if (!has[k]) continue;
      const int i = k % OB_ANCH_K, j = k / OB_ANCH_K, nb[4] = { i > 0 ? k - 1 : -1, i < OB_ANCH_K - 1 ? k + 1 : -1, j > 0 ? k - OB_ANCH_K : -1, j < OB_ANCH_K - 1 ? k + OB_ANCH_K : -1 };
      double sx = 0, sy = 0;
      int c = 0;
      for (int e = 0; e < 4; e++) if (nb[e] >= 0 && has[nb[e]]) { sx += g[nb[e]][0]; sy += g[nb[e]][1]; c++; }
      if (c < 2) continue;
      const double f = 1 / sqrt(1 + 1.0 / c);
      dd[nd++] = fabs(g[k][0] - sx / c) * f; dd[nd++] = fabs(g[k][1] - sy / c) * f;
    }
    if (nd >= 16 && na2 > 0) {
      qsort(dd, (size_t)nd, sizeof(double), cmp_double);
      const double sg = 1.4826 * dd[nd / 2], s2 = sg * sg, P = pw / (2.0 * na2), kf = P > 0 ? fmax(0, 1 - s2 / P) : 0;
      for (int q = 0; q < m; q++) if (gi[q] >= 0) { tv[q][0] *= kf; tv[q][1] *= kf; }
    }
  }
  {
    // ai: The mesh (2026-10-26; a thin-plate spline before it, the same blocks over the published lenses for 2.6 to 8.2%
    // ai: less of a frame): knots on both axes at the lattice's ends, where the border holds the map (0), and the 7
    // ai: anchors' centres between; an inner knot is its anchor's shrunk reading, a missing one the mean of its found
    // ai: 4-neighbours (rounds until none is left, at most 6; none near: 0).
    enum { KN = OB_ANCH_K + 2 };
    double kx[KN], ky[KN], kv[KN][KN][2];
    unsigned char kn[KN][KN];
    memset(kv, 0, sizeof kv); memset(kn, 0, sizeof kn);
    const double ha = 0.5 * ob_anchor_side(L);
    kx[0] = ky[0] = 0; kx[KN - 1] = ky[KN - 1] = 1;
    for (int i = 0; i < OB_ANCH_K; i++) { kx[i + 1] = (ob_anchor_m0(L, i) + ha - X0) / SX; ky[i + 1] = (ob_anchor_m0(L, i) + ha - Y0) / SY; }
    for (int q = 0; q < m; q++) if (gi[q] >= 0) {
      const int i = gi[q] % OB_ANCH_K + 1, j = gi[q] / OB_ANCH_K + 1;
      kv[j][i][0] = tv[q][0]; kv[j][i][1] = tv[q][1]; kn[j][i] = 1;
    }
    for (int round = 0; round < 6; round++) {
      unsigned char now[KN][KN];
      memcpy(now, kn, sizeof now);
      int left = 0;
      for (int j = 1; j < KN - 1; j++) for (int i = 1; i < KN - 1; i++) {
        if (kn[j][i]) continue;
        const int nb[4][2] = { { i - 1, j }, { i + 1, j }, { i, j - 1 }, { i, j + 1 } };
        double s0 = 0, s1 = 0;
        int c = 0;
        for (int e = 0; e < 4; e++) {
          const int a = nb[e][0], b = nb[e][1];
          if (a < 1 || a > KN - 2 || b < 1 || b > KN - 2 || !kn[b][a]) continue;
          s0 += kv[b][a][0]; s1 += kv[b][a][1]; c++;
        }
        if (c) { kv[j][i][0] = s0 / c; kv[j][i][1] = s1 / c; now[j][i] = 1; } else left++;
      }
      memcpy(kn, now, sizeof kn);
      if (!left) break;
    }
    // ai: each axis a cubic Hermite through the knots, the slopes from the neighbouring knots (one-sided at the ends):
    // ai: four knots' weights an axis (anch_herm4), so a node is a 4 x 4 blend of knot values
    for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
      if (L->node_mark[j * nx + i]) continue;
      int bx, by;
      double wx[4], wy[4];
      anch_herm4(kx, KN, (L->node_x[i] - X0) / SX, &bx, wx);
      anch_herm4(ky, KN, (L->node_y[j] - Y0) / SY, &by, wy);
      double d0 = 0, d1 = 0;
      for (int b = 0; b < 4; b++) for (int a = 0; a < 4; a++) {
        const int qx = bx + a, qy = by + b;
        if (qx < 0 || qx >= KN || qy < 0 || qy >= KN || (wx[a] == 0 && wy[b] == 0)) continue;
        d0 += wy[b] * wx[a] * kv[qy][qx][0]; d1 += wy[b] * wx[a] * kv[qy][qx][1];
      }
      nodes[j * nx + i].dx += (float)d0; nodes[j * nx + i].dy += (float)d1;
    }
  }
  free(tv); free(gi);
  free(mx); free(my); free(qx); free(qy); free(dx); free(dy); free(isb); free(drop); free(at);
}

// ---------------------------------------------------------------- thin frame: four lines
//
// The thin frame has no finder. What a receiver looks for is its solid line: a dark run a few
// pixels wide with light on both sides, at about the same place in row after row. Every scanned
// row gives the first few such runs from the left and from the right, every scanned column the
// same from the top and the bottom, and a robust line fit per side picks the border out of
// whatever else qualified. The corners are where the lines meet. A long line collects far more
// signal than a finder does, and it needs no margin.

#ifdef OB_VERIFY_POINTS
#include <stdio.h>
#endif
typedef struct { float along, across; int rank; } fpt_t;   // rank: which candidate of its scan line this is, counted from the outside
typedef struct { float a, b; int n; float c, mid; } fline_t;   // across = c (along - mid)^2 + a * along + b
// SIDE_K was 4 until a page of clutter round the symbol was measured (2026-09-21). A scan line
// stops at SIDE_K, so every dark run the clutter puts between the edge of the frame and the border
// costs one slot, and fit_lines draws its triples WITHIN one rank: the border's rank then varies
// along the side and no rank holds the twelve points a fit needs. Frames registered of 6 at 64 and
// 160 shapes: 1 and 1 at four, 4 and 5 at eight, 3 and 4 at sixteen. Sixteen is worse than eight
// because the slots past the border fill with clutter and the 120 RANSAC iterations go to it, so
// this is a plateau and not a knob to keep turning. It costs 0.5 to 0.7 ms of a 1.7 to 2.5 ms
// detect, and it is worth that outside clutter too: the phone cell gains 7% of its payload.
enum { SIDE_K = 8, SIDE_LINES = 3, EDGE_REACH = 6 };

// A candidate is the INNER edge of a dark run: where, going into the picture, dark turns to light
// with real contrast. Not the run's middle. A symbol shown full screen sits against the monitor's
// bezel, and from a distance the module of light between them blurs away, so bezel and line become
// one thick run whose middle is nowhere useful; its inner edge is still the line's inner edge. For
// the same reason a run's thickness plays no part: it is the line's, or the line's plus a bezel's.
// The contrast test is what keeps out the specks a flat light surround binarizes into (anything a
// little under the local mean is dark): the darkest pixel of the run's last few against the
// brightest of the light run's first few.
// g is the grey row, read at g[x * sx]. inner: +1 when going into the picture means rising x.
static int edge_contrast(const uint8_t *g, int sx, int edge, int inner, int dark_len, int light_len) {
  int dark = 255, light = 0;
  if (dark_len > EDGE_REACH) dark_len = EDGE_REACH;
  if (light_len > EDGE_REACH) light_len = EDGE_REACH;
  for (int k = 1; k <= dark_len; k++) { int v = g[(inner > 0 ? edge - k : edge + k - 1) * sx]; if (v < dark) dark = v; }
  for (int k = 0; k < light_len; k++) { int v = g[(inner > 0 ? edge + k : edge - k - 1) * sx]; if (v > light) light = v; }
  return light - dark;
}

// Darkest and brightest pixel of every BLK x BLK block, at mn[by * bs + bx]. What lets the scans below pass over
// a flat surround: it binarizes into specks by the ten thousand, every one of them a dark run to be refused.
static void block_range(const uint8_t *img, int w, int h, uint8_t *mn, uint8_t *mx, int bs) {
  int bw = (w + BLK - 1) / BLK, bh = (h + BLK - 1) / BLK;
  for (int by = 0; by < bh; by++) {
    int y0 = by * BLK, y1 = y0 + BLK > h ? h : y0 + BLK, bx = 0;
#ifdef OB_NEON
    // ai: arm64: eight blocks a step, a row's 64 pixels in four vectors, the blocks' extremes then by pairwise
    // ai: minima and maxima (UMINP, UMAXP) three times over: the same bytes
    for (; (bx + 8) * BLK <= w; bx += 8) {
      uint8x16_t l0 = vdupq_n_u8(255), l1 = l0, l2 = l0, l3 = l0, h0 = vdupq_n_u8(0), h1 = h0, h2 = h0, h3 = h0;
      for (int y = y0; y < y1; y++) {
        const uint8_t *q = img + (size_t)y * w + bx * BLK;
        const uint8x16_t a = vld1q_u8(q), b = vld1q_u8(q + 16), c = vld1q_u8(q + 32), d = vld1q_u8(q + 48);
        l0 = vminq_u8(l0, a); l1 = vminq_u8(l1, b); l2 = vminq_u8(l2, c); l3 = vminq_u8(l3, d);
        h0 = vmaxq_u8(h0, a); h1 = vmaxq_u8(h1, b); h2 = vmaxq_u8(h2, c); h3 = vmaxq_u8(h3, d);
      }
      const uint8x16_t lq = vpminq_u8(vpminq_u8(l0, l1), vpminq_u8(l2, l3)), hq = vpmaxq_u8(vpmaxq_u8(h0, h1), vpmaxq_u8(h2, h3));
      vst1_u8(mn + by * bs + bx, vpmin_u8(vget_low_u8(lq), vget_high_u8(lq)));
      vst1_u8(mx + by * bs + bx, vpmax_u8(vget_low_u8(hq), vget_high_u8(hq)));
    }
#endif
#ifdef OB_SIMD
    for (; (bx + 2) * BLK <= w; bx += 2) {
      v128_t lo = wasm_u8x16_splat(255), hi = wasm_u8x16_splat(0);
      for (int y = y0; y < y1; y++) { const v128_t v = wasm_v128_load(img + (size_t)y * w + bx * BLK); lo = wasm_u8x16_min(lo, v); hi = wasm_u8x16_max(hi, v); }
      // Each half of sixteen down to one value: neighbours, then pairs, then fours.
      lo = wasm_u8x16_min(lo, wasm_i8x16_shuffle(lo, lo, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)); hi = wasm_u8x16_max(hi, wasm_i8x16_shuffle(hi, hi, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
      lo = wasm_u8x16_min(lo, wasm_i8x16_shuffle(lo, lo, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13)); hi = wasm_u8x16_max(hi, wasm_i8x16_shuffle(hi, hi, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13));
      lo = wasm_u8x16_min(lo, wasm_i8x16_shuffle(lo, lo, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11)); hi = wasm_u8x16_max(hi, wasm_i8x16_shuffle(hi, hi, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11));
      mn[by * bs + bx] = (uint8_t)wasm_u8x16_extract_lane(lo, 0); mn[by * bs + bx + 1] = (uint8_t)wasm_u8x16_extract_lane(lo, 8);
      mx[by * bs + bx] = (uint8_t)wasm_u8x16_extract_lane(hi, 0); mx[by * bs + bx + 1] = (uint8_t)wasm_u8x16_extract_lane(hi, 8);
    }
#endif
    for (; bx < bw; bx++) {
      int x0 = bx * BLK, x1 = x0 + BLK > w ? w : x0 + BLK, lo = 255, hi = 0;
      for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) { int v = img[(size_t)y * w + x]; if (v < lo) lo = v; if (v > hi) hi = v; }
      mn[by * bs + bx] = (uint8_t)lo; mx[by * bs + bx] = (uint8_t)hi;
    }
  }
}

// What the candidate scans know about the image. mn, mx: the block map, bs to a row, with a margin of blocks that
// hold nothing (mn 255, mx 0) all round, so a window over it needs no edge cases.
// rskip[by * chunks + j]: no edge in pixels 16 j .. 16 j + 15 of a row of block row by can pass, nor one in the block
// either side. cskip[by * bw + bx]: the same for a column's pixels in block (bx, by) and the blocks above and below.
typedef struct { const uint8_t *grey, *mn, *mx, *rskip, *cskip; int w, h, bs, bw, chunks, min_con; uint8_t *buf; } edge_ctx_t;
_Static_assert(BLK == 8, "the scans take a block for half of a sixteen-pixel compare");
enum { MAP_PAD_X = 4, MAP_PAD_Y = 2 };

// ai: fb: frame_buf's (the map's buffer, c->buf, given back by frame_buf_done)
static int edge_ctx_init(edge_ctx_t *c, const uint8_t *grey, int w, int h, int min_con, ob_fbufs_t *fb) {
  int bw = (w + BLK - 1) / BLK, bh = (h + BLK - 1) / BLK, bs = bw + 24, chunks = (w + 15) / 16;
  size_t map = (size_t)bs * (bh + 2 * MAP_PAD_Y);
  uint8_t *buf = frame_buf(fb, OB_FB_EDGE, 2 * map + (size_t)bh * chunks + (size_t)bh * bw + 2 * (size_t)bs);
  if (!buf) return -1;
  memset(buf, 255, map); memset(buf + map, 0, map);
  uint8_t *mn = buf + MAP_PAD_Y * bs + MAP_PAD_X, *mx = mn + map, *rskip = buf + 2 * map, *cskip = rskip + (size_t)bh * chunks, *tmp = cskip + (size_t)bh * bw;
  block_range(grey, w, h, mn, mx, bs);
  for (int by = 0; by < bh; by++) {
    // Blocks b - 2 .. b + 3 along the row for the row scans (a sixteen-pixel unit is blocks b and b + 1), and
    // blocks by - 2 .. by + 2 down the column for the column scans: flat means their extremes are under min_con apart.
    const uint8_t *n0 = mn + by * bs, *x0 = mx + by * bs;
    uint8_t *flat = tmp, *cs = cskip + (size_t)by * bw;
    int b = 0;
#ifdef OB_SIMD
    const v128_t need = wasm_u8x16_splat((uint8_t)min_con);
    for (; b < bw; b += 16) {
      v128_t lo = wasm_v128_load(n0 + b - 2), hi = wasm_v128_load(x0 + b - 2), vlo = wasm_v128_load(n0 + b - 2 * bs), vhi = wasm_v128_load(x0 + b - 2 * bs);
      for (int o = -1; o <= 3; o++) { lo = wasm_u8x16_min(lo, wasm_v128_load(n0 + b + o)); hi = wasm_u8x16_max(hi, wasm_v128_load(x0 + b + o)); }
      for (int o = -1; o <= 2; o++) { vlo = wasm_u8x16_min(vlo, wasm_v128_load(n0 + b + o * bs)); vhi = wasm_u8x16_max(vhi, wasm_v128_load(x0 + b + o * bs)); }
      wasm_v128_store(flat + b, wasm_u8x16_lt(wasm_u8x16_sub_sat(hi, lo), need));
      uint8_t t[16];
      wasm_v128_store(t, wasm_u8x16_lt(wasm_u8x16_sub_sat(vhi, vlo), need));
      for (int q = 0; q < 16 && b + q < bw; q++) cs[b + q] = t[q] & 1;
    }
#else
    for (; b < bw; b++) {
      int lo = 255, hi = 0, vlo = 255, vhi = 0;
      for (int o = -2; o <= 3; o++) { if (n0[b + o] < lo) lo = n0[b + o]; if (x0[b + o] > hi) hi = x0[b + o]; }
      for (int o = -2; o <= 2; o++) { if (n0[b + o * bs] < vlo) vlo = n0[b + o * bs]; if (x0[b + o * bs] > vhi) vhi = x0[b + o * bs]; }
      flat[b] = (hi > lo ? hi - lo : 0) < min_con; cs[b] = (vhi > vlo ? vhi - vlo : 0) < min_con;
    }
#endif
    for (int j = 0; j < chunks; j++) rskip[(size_t)by * chunks + j] = 16 * j + 16 <= w && flat[2 * j] != 0;   // whole units only
  }
  *c = (edge_ctx_t){ grey, mn, mx, rskip, cskip, w, h, bs, bw, chunks, min_con, buf };
  return 0;
}

#ifdef OB_NEON
// ai: arm64: edge_contrast over the pixels edge - EDGE_REACH .. edge + EDGE_REACH - 1 of the line held in one vector
// ai: (a row's in one load of sixteen, a column's twelve by lane loads), the dark run's and the light run's lanes
// ai: picked by masks on the lane number: the same darkest and brightest pixel, without a branch on either length.
// ai: The caller keeps the pixels read inside the line (a row's load of sixteen reads four past the twelve).
static inline int edge_contrast_n(const uint8_t *g, int sx, int edge, int inner, int dark_len, int light_len) {
  static const uint8_t LANE[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  dark_len = dark_len < 0 ? 0 : dark_len > EDGE_REACH ? EDGE_REACH : dark_len;
  light_len = light_len < 0 ? 0 : light_len > EDGE_REACH ? EDGE_REACH : light_len;
  const uint8_t *b = g + (ptrdiff_t)(edge - EDGE_REACH) * sx;
  uint8x16_t v;
  if (sx == 1) v = vld1q_u8(b);
  else {
    v = vdupq_n_u8(0);
    v = vld1q_lane_u8(b, v, 0); v = vld1q_lane_u8(b + sx, v, 1); v = vld1q_lane_u8(b + 2 * sx, v, 2); v = vld1q_lane_u8(b + 3 * sx, v, 3);
    v = vld1q_lane_u8(b + 4 * sx, v, 4); v = vld1q_lane_u8(b + 5 * sx, v, 5); v = vld1q_lane_u8(b + 6 * sx, v, 6); v = vld1q_lane_u8(b + 7 * sx, v, 7);
    v = vld1q_lane_u8(b + 8 * sx, v, 8); v = vld1q_lane_u8(b + 9 * sx, v, 9); v = vld1q_lane_u8(b + 10 * sx, v, 10); v = vld1q_lane_u8(b + 11 * sx, v, 11);
  }
  // ai: going into the picture with rising position the dark run ends below edge and the light run starts at it;
  // ai: the other way the light run is below and the dark run from edge on
  const int below = inner > 0 ? dark_len : light_len, above = inner > 0 ? light_len : dark_len;
  const uint8x16_t lane = vld1q_u8(LANE);
  const uint8x16_t lo = vandq_u8(vcgeq_u8(lane, vdupq_n_u8((uint8_t)(EDGE_REACH - below))), vcltq_u8(lane, vdupq_n_u8(EDGE_REACH)));
  const uint8x16_t hi = vandq_u8(vcgeq_u8(lane, vdupq_n_u8(EDGE_REACH)), vcltq_u8(lane, vdupq_n_u8((uint8_t)(EDGE_REACH + above))));
  const uint8x16_t dm = inner > 0 ? lo : hi, lm = inner > 0 ? hi : lo;
  return (int)vmaxvq_u8(vandq_u8(v, lm)) - (int)vminvq_u8(vornq_u8(v, dm));
}
#endif

// edge_contrast(...) >= min_con for an edge of scan line `line` (a row, or with axis 1 a column). The test reads
// pixels edge - EDGE_REACH .. edge + EDGE_REACH - 1 of the line at most, so where the blocks under those hold no
// such contrast between them the answer is no, and no pixel is read.
static inline int edge_passes(const edge_ctx_t *c, int axis, int line, int edge, int inner, int dark_len, int light_len) {
#ifdef OB_NEON
  // ai: arm64: where the window lies inside the line its pixels are read at once; the blocks' test below is only a
  // ai: shortcut to the same no (light - dark is never more than the blocks' range), so the answer is the same
  if (edge >= EDGE_REACH && edge + (axis ? EDGE_REACH : 16 - EDGE_REACH) <= (axis ? c->h : c->w))
    return edge_contrast_n(axis ? c->grey + line : c->grey + (size_t)line * c->w, axis ? c->w : 1, edge, inner, dark_len, light_len) >= c->min_con;
#endif
  int len = axis ? c->h : c->w, lo = edge - EDGE_REACH < 0 ? 0 : edge - EDGE_REACH, hi = edge + EDGE_REACH - 1 >= len ? len - 1 : edge + EDGE_REACH - 1, mn = 255, mx = 0;
  for (int b = lo / BLK; b <= hi / BLK; b++) { int k = axis ? b * c->bs + line / BLK : (line / BLK) * c->bs + b; if (c->mn[k] < mn) mn = c->mn[k]; if (c->mx[k] > mx) mx = c->mx[k]; }
  if (mx - mn < c->min_con) return 0;
  return edge_contrast(axis ? c->grey + line : c->grey + (size_t)line * c->w, axis ? c->w : 1, edge, inner, dark_len, light_len) >= c->min_con;
}

// A scan of one line for candidates, from one end. A dark run becomes a candidate once the light run past it is
// known, so it waits for the next boundary (or the line's end). Going forwards `open` is where the dark run the
// scan is inside began, going backwards where it ends; -1 in light. pa, pb: the finished run in waiting (pb -1: none).
typedef struct { int open, pa, pb, got; } scan_t;

// A boundary at p (pixel p differs from pixel p - 1), met going forwards. dark: pixel p is dark.
static inline void near_edge(scan_t *s, int p, int dark, const edge_ctx_t *c, int axis, int line, fpt_t *out) {
  if (!dark) { s->pa = s->open; s->pb = p; s->open = -1; return; }
  if (s->pb >= 0) { if (edge_passes(c, axis, line, s->pb, 1, s->pb - s->pa, p - s->pb)) { out[s->got] = (fpt_t){ (float)line, (float)s->pb, s->got }; s->got++; } s->pb = -1; }
  s->open = p;
}
// The same boundary met going backwards: a light pixel p is the end of a dark run being entered, a dark one the start of the run being left.
static inline void far_edge(scan_t *s, int p, int dark, const edge_ctx_t *c, int axis, int line, fpt_t *out) {
  if (dark) { s->pa = p; s->pb = s->open; s->open = -1; return; }
  if (s->pb >= 0) { if (edge_passes(c, axis, line, s->pa, -1, s->pb - s->pa, s->pa - p)) { out[s->got] = (fpt_t){ (float)line, (float)s->pa, s->got }; s->got++; } s->pb = -1; }
  s->open = p;
}
// The line's end: a run in waiting has the rest of the line as its light run. One still open touches the edge and has none.
static inline void near_end(scan_t *s, int len, const edge_ctx_t *c, int axis, int line, fpt_t *out) {
  if (s->pb >= 0 && edge_passes(c, axis, line, s->pb, 1, s->pb - s->pa, len - s->pb)) { out[s->got] = (fpt_t){ (float)line, (float)s->pb, s->got }; s->got++; }
}
static inline void far_end(scan_t *s, const edge_ctx_t *c, int axis, int line, fpt_t *out) {
  if (s->pb >= 0 && s->pa >= 1 && edge_passes(c, axis, line, s->pa, -1, s->pb - s->pa, s->pa)) { out[s->got] = (fpt_t){ (float)line, (float)s->pa, s->got }; s->got++; }
}

// A scan can pass over a stretch where rskip / cskip says nothing can pass, without following its boundaries, and
// come out in the state it would have had, as far as any later test can tell:
//   a run in waiting is settled on the way in. If its edge is EDGE_REACH or more back, its light run is already
//     known to be as long as the test reads. If it is nearer, it lies in the block before the stretch, where
//     nothing can pass either;
//   on the way out only whether the scan is inside a dark run matters, not where the run began: the next edge that
//     could pass is a block or more further on, and the test reads no more than EDGE_REACH of a run.
#define ACCEPT(s, out, line, at) { (out)[(s)->got] = (fpt_t){ (float)(line), (float)(at), (s)->got }; (s)->got++; }

#ifdef OB_NEON
// ai: arm64: which of the sixteen pixels at p differ from those at q, as bit 4 i for pixel i (the equal bytes' mask
// ai: narrowed to a nibble a byte by SHRN, its low bits cleared where equal)
static inline uint64_t bounds16(const uint8_t *p, const uint8_t *q) {
  const uint8x16_t eq = vceqq_u8(vld1q_u8(p), vld1q_u8(q));
  return ~vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0) & 0x1111111111111111ull;
}
#endif

// Candidates of the left border (near) and the right (far) from every second row: the first SIDE_K from each end.
// Inside the picture nearly every dark run qualifies, so a scan is over a few runs past the border, and the
// middle of the row, where most of the boundaries are, is never looked at.
static void row_points(const uint8_t *bin, int w, int h, const edge_ctx_t *c, fpt_t *near, int *nn, fpt_t *far, int *nf) {
  const int chunks = c->chunks;
  *nn = *nf = 0;
  for (int y = 1; y < h; y += 2) {
    const uint8_t *row = bin + (size_t)y * w, *skip = c->rskip + (size_t)(y / BLK) * chunks;
    scan_t s = { row[0] ? 0 : -1, 0, -1, 0 };
    fpt_t *out = near + *nn;
    for (int j = 0; j < chunks && s.got < SIDE_K; j++) {
      const int u0 = 16 * j, u1 = u0 + 16 > w ? w : u0 + 16;
      if (skip[j]) {
        if (s.pb >= 0) { if (u0 - s.pb >= EDGE_REACH && edge_passes(c, 0, y, s.pb, 1, s.pb - s.pa, u0 - s.pb)) ACCEPT(&s, out, y, s.pb); s.pb = -1; }
        s.open = row[u1 - 1] ? u0 : -1;
        continue;
      }
#ifdef OB_NEON
      // ai: arm64: the sixteen boundaries as a nibble each (SHRN), bit 4 i for pixel u0 + i
      if (u0 && u1 == u0 + 16) {
        for (uint64_t m = bounds16(row + u0, row + u0 - 1); m && s.got < SIDE_K; m &= m - 1) { int p = u0 + (__builtin_ctzll(m) >> 2); near_edge(&s, p, row[p], c, 0, y, out); }
        continue;
      }
#elif defined(OB_SIMD)
      if (u0 && u1 == u0 + 16) {
        for (unsigned m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(row + u0), wasm_v128_load(row + u0 - 1))); m && s.got < SIDE_K; m &= m - 1) { int p = u0 + __builtin_ctz(m); near_edge(&s, p, row[p], c, 0, y, out); }
        continue;
      }
#endif
      for (int x = u0 ? u0 : 1; x < u1 && s.got < SIDE_K; x++) if (row[x] != row[x - 1]) near_edge(&s, x, row[x], c, 0, y, out);
    }
    if (s.got < SIDE_K) near_end(&s, w, c, 0, y, out);
    *nn += s.got;
    s = (scan_t){ row[w - 1] ? w : -1, 0, -1, 0 };
    out = far + *nf;
    for (int j = chunks - 1; j >= 0 && s.got < SIDE_K; j--) {
      const int u0 = 16 * j, u1 = u0 + 16 > w ? w : u0 + 16;
      if (skip[j]) {
        if (s.pb >= 0) { if (s.pa - (u1 - 1) >= EDGE_REACH && edge_passes(c, 0, y, s.pa, -1, s.pb - s.pa, s.pa - (u1 - 1))) ACCEPT(&s, out, y, s.pa); s.pb = -1; }
        s.open = u0 && row[u0 - 1] ? u1 : -1;
        continue;
      }
#ifdef OB_NEON
      if (u0 && u1 == u0 + 16) {
        for (uint64_t m = bounds16(row + u0, row + u0 - 1); m && s.got < SIDE_K; ) { int b = 63 - __builtin_clzll(m), p = u0 + (b >> 2); m ^= 1ull << b; far_edge(&s, p, row[p], c, 0, y, out); }
        continue;
      }
#elif defined(OB_SIMD)
      if (u0 && u1 == u0 + 16) {
        for (unsigned m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(row + u0), wasm_v128_load(row + u0 - 1))); m && s.got < SIDE_K; ) { int i = 31 - __builtin_clz(m), p = u0 + i; m &= ~(1u << i); far_edge(&s, p, row[p], c, 0, y, out); }
        continue;
      }
#endif
      for (int x = u1 - 1; x >= (u0 ? u0 : 1) && s.got < SIDE_K; x--) if (row[x] != row[x - 1]) far_edge(&s, x, row[x], c, 0, y, out);
    }
    if (s.got < SIDE_K) far_end(&s, c, 0, y, out);
    *nf += s.got;
  }
}

#ifdef OB_NEON
// ai: arm64: col_points' boundaries m of row y in chunk c (bit i: column 16 c + i), in column order
static inline void col_bounds(scan_t *st, fpt_t *slot, const edge_ctx_t *ctx, int dir, int y, const uint8_t *r1, int c, unsigned m, uint16_t *active, int *live) {
  for (; m; m &= m - 1) {
    int i = __builtin_ctz(m), x = 16 * c + i;
    scan_t *sc = &st[x >> 1];
    if (dir) far_edge(sc, y, r1[x], ctx, 1, x, slot + (size_t)(x >> 1) * SIDE_K); else near_edge(sc, y, r1[x], ctx, 1, x, slot + (size_t)(x >> 1) * SIDE_K);
    if (sc->got == SIDE_K) { active[c] &= (uint16_t)~(1u << i); (*live)--; }
  }
}
#endif

// The same for the top border (near) and the bottom (far), from every second column, without turning the image
// over: all the columns are scanned at once, a row at a time. Comparing a row with the one before it gives the
// columns that have a boundary there, sixteen to a compare, and a column drops out of the compare once it has its
// SIDE_K candidates, or for as long as it runs through blocks where nothing can pass. Sixteen columns with none
// left in the compare cost nothing, which is most of them: the surround sleeps, and a few rows into the picture
// the columns that cross it are done.
static void col_points(const uint8_t *bin, int w, int h, const edge_ctx_t *ctx, fpt_t *near, int *nn, fpt_t *far, int *nf) {
  int nc = w / 2, chunks = ctx->chunks;
  scan_t *st = malloc((size_t)nc * sizeof(scan_t));
  fpt_t *slot = malloc((size_t)nc * SIDE_K * sizeof(fpt_t));
#ifdef OB_NEON
  // ai: arm64: the masks padded to whole groups of eight chunks, the padding never active, for the group test below
  const int cpad = (chunks + 7) & ~7;
  uint16_t *active = calloc((size_t)cpad * 2, sizeof(uint16_t)), *asleep = active + cpad;
#else
  uint16_t *active = malloc((size_t)chunks * 2 * sizeof(uint16_t)), *asleep = active + chunks;
#endif
  *nn = *nf = 0;
  for (int dir = 0; dir < 2 && st && slot && active; dir++) {
    int live = nc, cur = -1;
    for (int i = 0; i < nc; i++) st[i] = (scan_t){ bin[(size_t)(dir ? h - 1 : 0) * w + 2 * i + 1] ? (dir ? h : 0) : -1, 0, -1, 0 };
    for (int c = 0; c < chunks; c++) { unsigned m = 0; for (int i = 1; i < 16 && 16 * c + i < w; i += 2) m |= 1u << i; active[c] = (uint16_t)m; asleep[c] = 0; }
    for (int k = 1; k < h && live; k++) {
      int y = dir ? h - k : k, by = y / BLK;
      if (by != cur) {
        // A new row of blocks: which columns fall asleep in it, and which wake. first, last: its rows. at: the row a
        // waker stands on, having followed no boundary since it fell asleep: its pixel says if it wakes inside a dark run.
        const int first = by * BLK, last = first + BLK > h ? h - 1 : first + BLK - 1, at = dir ? last : first - 1;
        cur = by;
        for (int c = 0; c < chunks; c++) {
          if (!active[c]) continue;
          unsigned sm = (ctx->cskip[by * ctx->bw + 2 * c] ? 0x00ffu : 0) | (2 * c + 1 < ctx->bw && ctx->cskip[by * ctx->bw + 2 * c + 1] ? 0xff00u : 0);
          for (unsigned a = sm & ~asleep[c] & active[c]; a; a &= a - 1) {
            int i = __builtin_ctz(a), x = 16 * c + i;
            scan_t *sc = &st[x >> 1];
            fpt_t *out = slot + (size_t)(x >> 1) * SIDE_K;
            if (sc->pb < 0) continue;
            if (dir) { if (sc->pa - last >= EDGE_REACH && edge_passes(ctx, 1, x, sc->pa, -1, sc->pb - sc->pa, sc->pa - last)) ACCEPT(sc, out, x, sc->pa); }
            else if (first - sc->pb >= EDGE_REACH && edge_passes(ctx, 1, x, sc->pb, 1, sc->pb - sc->pa, first - sc->pb)) ACCEPT(sc, out, x, sc->pb);
            sc->pb = -1;
            if (sc->got == SIDE_K) { active[c] &= (uint16_t)~(1u << i); live--; }
          }
          for (unsigned a = asleep[c] & ~sm & active[c]; a; a &= a - 1) {
            int x = 16 * c + __builtin_ctz(a);
            st[x >> 1].open = bin[(size_t)at * w + x] ? at + dir : -1; st[x >> 1].pb = -1;
          }
          asleep[c] = (uint16_t)sm;
        }
      }
      const uint8_t *r1 = bin + (size_t)y * w, *r0 = r1 - w;
#ifdef OB_NEON
      // ai: arm64: eight chunks at a time: a group with no column awake passed over whole; a whole group's sixteen-bit
      // ai: boundary masks made together (each byte's bit of its place in its half, then pairwise adds three times over,
      // ai: ADDP: chunk k's two bytes are lane k), masked by the columns awake; then each chunk in order as the plain
      // ai: loop takes it (a chunk's own boundaries change only its own mask), the group's last chunks one by one
      for (int c0 = 0; c0 < chunks; c0 += 8) {
        const uint16x8_t aw = vbicq_u16(vld1q_u16(active + c0), vld1q_u16(asleep + c0));
        if (!vmaxvq_u16(aw)) continue;
        if (16 * c0 + 128 <= w) {
          static const uint8_t BIT[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
          const uint8x16_t bit = vld1q_u8(BIT);
          uint8x16_t b[8];
          for (int k = 0; k < 8; k++) b[k] = vbicq_u8(bit, vceqq_u8(vld1q_u8(r1 + 16 * (c0 + k)), vld1q_u8(r0 + 16 * (c0 + k))));
          const uint8x16_t q = vpaddq_u8(vpaddq_u8(vpaddq_u8(b[0], b[1]), vpaddq_u8(b[2], b[3])), vpaddq_u8(vpaddq_u8(b[4], b[5]), vpaddq_u8(b[6], b[7])));
          const uint16x8_t m8 = vandq_u16(vreinterpretq_u16_u8(q), aw);
          if (!vmaxvq_u16(m8)) continue;
          uint16_t ms[8];
          vst1q_u16(ms, m8);
          for (int k = 0; k < 8; k++) if (ms[k]) col_bounds(st, slot, ctx, dir, y, r1, c0 + k, ms[k], active, &live);
          continue;
        }
        for (int c = c0; c < chunks; c++) {
          const unsigned awake = active[c] & ~(unsigned)asleep[c];
          if (!awake) continue;
          unsigned m = 0;
          if (16 * c + 16 <= w) m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(r1 + 16 * c), wasm_v128_load(r0 + 16 * c))) & awake;
          else for (unsigned a = awake; a; a &= a - 1) { int i = __builtin_ctz(a); m |= (unsigned)(r1[16 * c + i] != r0[16 * c + i]) << i; }
          col_bounds(st, slot, ctx, dir, y, r1, c, m, active, &live);
        }
      }
#else
      for (int c = 0; c < chunks; c++) {
        const unsigned awake = active[c] & ~(unsigned)asleep[c];
        if (!awake) continue;
        unsigned m = 0;
#ifdef OB_SIMD
        if (16 * c + 16 <= w) m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(r1 + 16 * c), wasm_v128_load(r0 + 16 * c))) & awake;
        else
#endif
        for (unsigned a = awake; a; a &= a - 1) { int i = __builtin_ctz(a); m |= (unsigned)(r1[16 * c + i] != r0[16 * c + i]) << i; }
        for (; m; m &= m - 1) {
          int i = __builtin_ctz(m), x = 16 * c + i;
          scan_t *sc = &st[x >> 1];
          if (dir) far_edge(sc, y, r1[x], ctx, 1, x, slot + (size_t)(x >> 1) * SIDE_K); else near_edge(sc, y, r1[x], ctx, 1, x, slot + (size_t)(x >> 1) * SIDE_K);
          if (sc->got == SIDE_K) { active[c] &= (uint16_t)~(1u << i); live--; }
        }
      }
#endif
    }
    fpt_t *out = dir ? far : near;
    int n = 0;
    for (int i = 0; i < nc; i++) {
      int x = 2 * i + 1;
      if (st[i].got < SIDE_K) { if (dir) far_end(&st[i], ctx, 1, x, slot + (size_t)i * SIDE_K); else near_end(&st[i], h, ctx, 1, x, slot + (size_t)i * SIDE_K); }
      for (int k = 0; k < st[i].got; k++) out[n++] = slot[(size_t)i * SIDE_K + k];
    }
    *(dir ? nf : nn) = n;
  }
  free(st); free(slot); free(active);
}
#undef ACCEPT

#ifdef OB_VERIFY_POINTS
// The scan these two replaced, kept for a build that runs both and compares: every run boundary of every scanned
// line, and for the columns a transposed copy of the whole image.
// grey is read at grey[row * sy + x * sx], so the same code serves the transposed pass.
static void side_points(const uint8_t *bin, int w, int h, const uint8_t *grey, int sx, int sy, int min_con, int *pos, fpt_t *near, int *nn, fpt_t *far, int *nf) {
  *nn = *nf = 0;
  for (int y = 1; y < h; y += 2) {
    const uint8_t *g = grey + (size_t)y * sy;
    int np = run_edges(bin + y * w, w, pos), runs = np / 2, got = 0;
    for (int k = 0; k < runs && got < SIDE_K; k++) {
      int a = pos[2 * k], b = pos[2 * k + 1], gap = (k + 1 < runs ? pos[2 * k + 2] : w) - b;
      if (gap < 1 || edge_contrast(g, sx, b, 1, b - a, gap) < min_con) continue;
      near[(*nn)++] = (fpt_t){ (float)y, (float)b, got }; got++;
    }
    got = 0;
    for (int k = runs - 1; k >= 0 && got < SIDE_K; k--) {
      int a = pos[2 * k], b = pos[2 * k + 1], gap = a - (k ? pos[2 * k - 1] : 0);
      if (gap < 1 || edge_contrast(g, sx, a, -1, b - a, gap) < min_con) continue;
      far[(*nf)++] = (fpt_t){ (float)y, (float)a, got }; got++;
    }
  }
}

#endif

static inline float line_at(const fline_t *l, float along) { float d = along - l->mid; return l->c * d * d + l->a * along + l->b; }

// The fits test the same points against curve after curve, so the points are kept as two plain arrays and tested
// four at a time. in4: which of four points lie within tol of the curve, as a bit each.
#if defined(OB_SIMD) && !defined(OB_NEON)
static inline v128_t curve_in4(const fline_t *l, v128_t al, v128_t ac, v128_t tol) {
  const v128_t d = wasm_f32x4_sub(al, wasm_f32x4_splat(l->mid));
  const v128_t at = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(l->c), d), d), wasm_f32x4_mul(wasm_f32x4_splat(l->a), al)), wasm_f32x4_splat(l->b));
  return wasm_f32x4_le(wasm_f32x4_abs(wasm_f32x4_sub(ac, at)), tol);
}
#endif
#ifdef OB_NEON
// ai: arm64: curve_in4's test in NEON's own instructions, the same float operations in the same order (FABD is the
// ai: rounded difference made positive, as fabsf of it): a lane all ones where the point is within tol
typedef struct { float32x4_t mid, c, a, b, tol; } curve4_t;
static inline curve4_t curve4_of(const fline_t *l, float tol) {
  return (curve4_t){ vdupq_n_f32(l->mid), vdupq_n_f32(l->c), vdupq_n_f32(l->a), vdupq_n_f32(l->b), vdupq_n_f32(tol) };
}
static inline uint32x4_t curve_in4n(const curve4_t *k, float32x4_t al, float32x4_t ac) {
  const float32x4_t d = vsubq_f32(al, k->mid);
  const float32x4_t at = vaddq_f32(vaddq_f32(vmulq_f32(vmulq_f32(k->c, d), d), vmulq_f32(k->a, al)), k->b);
  return vcleq_f32(vabdq_f32(ac, at), k->tol);
}
#endif
// How many of m points lie within tol of the curve, and the stretch of `along` they cover (lo > hi when none do).
static int curve_support(const fline_t *l, const float *al, const float *ac, int m, float tol, float *plo, float *phi) {
  float lo = 1e9f, hi = -1e9f;
  int n = 0, k = 0;
#ifdef OB_NEON
  // ai: arm64: eight points a step; the count in lanes (a passing lane's all-ones subtracted), the stretch by
  // ai: FMIN / FMAX over the passing lanes' along (a line number, whole and positive, so no NaN or signed zero for
  // ai: them to treat otherwise than the compare-and-select), each order-free, so the same n, lo and hi
  {
    const curve4_t K = curve4_of(l, tol);
    const float32x4_t big = vdupq_n_f32(lo), small = vdupq_n_f32(hi);
    uint32x4_t c0 = vdupq_n_u32(0), c1 = c0;
    float32x4_t lo0 = big, lo1 = big, hi0 = small, hi1 = small;
    for (; k + 8 <= m; k += 8) {
      const float32x4_t a0 = vld1q_f32(al + k), a1 = vld1q_f32(al + k + 4);
      const uint32x4_t in0 = curve_in4n(&K, a0, vld1q_f32(ac + k)), in1 = curve_in4n(&K, a1, vld1q_f32(ac + k + 4));
      c0 = vsubq_u32(c0, in0); c1 = vsubq_u32(c1, in1);
      lo0 = vminq_f32(lo0, vbslq_f32(in0, a0, big)); lo1 = vminq_f32(lo1, vbslq_f32(in1, a1, big));
      hi0 = vmaxq_f32(hi0, vbslq_f32(in0, a0, small)); hi1 = vmaxq_f32(hi1, vbslq_f32(in1, a1, small));
    }
    for (; k + 4 <= m; k += 4) {
      const float32x4_t a0 = vld1q_f32(al + k);
      const uint32x4_t in0 = curve_in4n(&K, a0, vld1q_f32(ac + k));
      c0 = vsubq_u32(c0, in0);
      lo0 = vminq_f32(lo0, vbslq_f32(in0, a0, big)); hi0 = vmaxq_f32(hi0, vbslq_f32(in0, a0, small));
    }
    n = (int)vaddvq_u32(vaddq_u32(c0, c1)); lo = vminvq_f32(vminq_f32(lo0, lo1)); hi = vmaxvq_f32(vmaxq_f32(hi0, hi1));
  }
#elif defined(OB_SIMD)
  v128_t vlo = wasm_f32x4_splat(lo), vhi = wasm_f32x4_splat(hi);
  const v128_t vtol = wasm_f32x4_splat(tol);
  for (; k + 4 <= m; k += 4) {
    const v128_t a = wasm_v128_load(al + k), in = curve_in4(l, a, wasm_v128_load(ac + k), vtol);
    n += __builtin_popcount((unsigned)wasm_i32x4_bitmask(in));
    vlo = wasm_f32x4_pmin(vlo, wasm_v128_bitselect(a, vlo, in)); vhi = wasm_f32x4_pmax(vhi, wasm_v128_bitselect(a, vhi, in));
  }
  float t[4];
  wasm_v128_store(t, vlo); for (int q = 0; q < 4; q++) if (t[q] < lo) lo = t[q];
  wasm_v128_store(t, vhi); for (int q = 0; q < 4; q++) if (t[q] > hi) hi = t[q];
#endif
  for (; k < m; k++) if (fabsf(ac[k] - line_at(l, al[k])) <= tol) { n++; if (al[k] < lo) lo = al[k]; if (al[k] > hi) hi = al[k]; }
  *plo = lo; *phi = hi;
  return n;
}

// Least-squares polish of a sampled parabola, over the points it holds. Same reach as the sampling
// stage and two passes only: anything wider or longer lets the curve walk onto the track where the
// border's own points run out near a corner.
static void refit_curve(const float *al, const float *ac, int n, float tol, fline_t *l) {
  float lo, hi;
  curve_support(l, al, ac, n, tol, &lo, &hi);
  if (hi <= lo) return;
  // Re-centre without moving the curve: c d^2 + a x + b with d = x - mid is c e^2 + (a + 2 c (m - mid)) x + const with e = x - m.
  { float m = 0.5f * (lo + hi), at = line_at(l, m); l->a += 2 * l->c * (m - l->mid); l->mid = m; l->b = at - l->a * m; }
  float half = 0.5f * (hi - lo);
  for (int pass = 0; pass < 2; pass++) {
    // across = c u^2 + a' u + b' in u = (along - mid) / half, which keeps the normal equations well scaled.
    double S[3][4] = { { 0 } }; int m = 0, i = 0;
    #define TAKE(i) { double u = (al[i] - l->mid) / half, f[3] = { u * u, u, 1 }; \
      for (int r = 0; r < 3; r++) { for (int q = 0; q < 3; q++) S[r][q] += f[r] * f[q]; S[r][3] += f[r] * ac[i]; } m++; }
#ifdef OB_NEON
    // ai: arm64: the test eight points a step, a step with none passing left at one compare; u four a step (FDIV,
    // ai: the same float division); the sums in their order but only the distinct ones, in pairs of lanes: with
    // ai: f = (u u, u, 1), S[0][0] sums f0 f0 and S[0][1] = S[1][0] f0 u; S[0][2], S[2][0] and S[1][1] all sum f0
    // ai: (f0 1, 1 f0 and u u are each f0 itself); S[1][2] = S[2][1] sums u, S[2][2] the count; S[r][3] sums f[r] ac,
    // ai: S[2][3] ac. The same addends in the same order, so the same sums
    {
      const curve4_t K = curve4_of(l, tol);
      const float32x4_t vmid = vdupq_n_f32(l->mid), vhalf = vdupq_n_f32(half);
      float64x2_t s01 = vdupq_n_f64(0), s02 = s01, s03 = s01;
      double s23 = 0;
      float u4[8];
      #define TAKE2(u, acv) { const double u_ = (u), ac_ = (acv); const float64x2_t f_ = { u_ * u_, u_ }; \
        s01 = vaddq_f64(s01, vmulq_f64(f_, vdupq_n_f64(f_[0]))); s02 = vaddq_f64(s02, f_); s03 = vaddq_f64(s03, vmulq_f64(f_, vdupq_n_f64(ac_))); s23 += ac_; m++; }
      for (; i + 8 <= n; i += 8) {
        const float32x4_t a0 = vld1q_f32(al + i), a1 = vld1q_f32(al + i + 4);
        const uint32x4_t in0 = curve_in4n(&K, a0, vld1q_f32(ac + i)), in1 = curve_in4n(&K, a1, vld1q_f32(ac + i + 4));
        uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcombine_u16(vmovn_u32(in0), vmovn_u32(in1)))), 0);
        if (!bits) continue;
        vst1q_f32(u4, vdivq_f32(vsubq_f32(a0, vmid), vhalf)); vst1q_f32(u4 + 4, vdivq_f32(vsubq_f32(a1, vmid), vhalf));
        for (bits &= 0x0101010101010101ull; bits; bits &= bits - 1) { const int j = __builtin_ctzll(bits) >> 3; TAKE2(u4[j], ac[i + j]); }
      }
      for (; i < n; i++) if (fabsf(ac[i] - line_at(l, al[i])) <= tol) TAKE2((al[i] - l->mid) / half, ac[i]);
      #undef TAKE2
      const double f0 = vgetq_lane_f64(s02, 0), f1 = vgetq_lane_f64(s02, 1);
      const double T[3][4] = { { vgetq_lane_f64(s01, 0), vgetq_lane_f64(s01, 1), f0, vgetq_lane_f64(s03, 0) },
                               { vgetq_lane_f64(s01, 1), f0, f1, vgetq_lane_f64(s03, 1) }, { f0, f1, (double)m, s23 } };
      memcpy(S, T, sizeof S);
    }
#elif defined(OB_SIMD)
    // The test four at a time, the sums over those that pass one at a time and in order, so they round as ever.
    const v128_t vtol = wasm_f32x4_splat(tol);
    for (; i + 4 <= n; i += 4) for (unsigned in = (unsigned)wasm_i32x4_bitmask(curve_in4(l, wasm_v128_load(al + i), wasm_v128_load(ac + i), vtol)); in; in &= in - 1) TAKE(i + __builtin_ctz(in));
#endif
    for (; i < n; i++) if (fabsf(ac[i] - line_at(l, al[i])) <= tol) TAKE(i);
    #undef TAKE
    if (m < 12) return;
    for (int col = 0; col < 3; col++) {
      int piv = col;
      for (int r = col + 1; r < 3; r++) if (fabs(S[r][col]) > fabs(S[piv][col])) piv = r;
      if (fabs(S[piv][col]) < 1e-9) return;
      for (int q = 0; q < 4; q++) { double tmp = S[col][q]; S[col][q] = S[piv][q]; S[piv][q] = tmp; }
      for (int r = 0; r < 3; r++) if (r != col) { double g = S[r][col] / S[col][col]; for (int q = col; q < 4; q++) S[r][q] -= g * S[col][q]; }
    }
    double c = S[0][3] / S[0][0], a1 = S[1][3] / S[1][1], b1 = S[2][3] / S[2][2];
    l->c = (float)(c / ((double)half * half)); l->a = (float)(a1 / half); l->b = (float)(b1 - a1 * l->mid / half); l->n = m;
  }
}

// ---------------------------------------------------------------- thin frame: the corner marks
//
// The marks are found DIRECTLY IN THE IMAGE here, not out of the line scan's candidate lists, and that is the
// whole point of this pass. Measured on a noisy surround (2026-09-21): it costs the thin frame everything, a quiet zone
// does not buy it back at any width, and cropping the capture to the symbol restores all of it. Everything that
// works from the line scan's lists fails for one reason: a scan STOPS after SIDE_K candidates, so when noise
// supplies that many before the border, the border was never recorded and nothing downstream can recover it.
// Raising the budget records it and still does not return the payload, at several times the detect time.
//
// So this is a LOCAL finder, in the sense QR's is: one place in the image, judged on its own evidence, with
// nothing to crowd out. A gapped mark's cross-section is the same whichever way the scan crosses it, because
// the mark is symmetric about its diagonal: the border line's OB_THIN_LINE modules, the gap's OB_RING_DEEP, then
// the core, each bounded by light. That is a run-length ratio, so it is scale-free and the module size falls out
// of it rather than having to be known. Filled marks have no such cross-section (the line and the core are one
// run) and are not detectable, which is no loss: they cost half the payload at 720.
enum { MARK_MIN_CORE = 3 };

// The core's modules, and where its centre sits in layout coordinates. thin_corner paints the core from the end
// of the gap band to the mark's light rim, so it spans [OB_THIN_TRACK, S - 1) and its middle is the two averaged. A
// corner mark borrows the border's line and gap for two of its sides.
static inline int mark_core(const ob_layout_t *L) { return L->corner - 1 - OB_THIN_TRACK; }
static inline float mark_mid(const ob_layout_t *L) { return (OB_THIN_TRACK + L->corner - 1) / 2.0f; }

// line, gap and core in pixels against the ratio OB_THIN_LINE : OB_RING_DEEP : core. `unit` comes back as the
// module size the three of them agree on.
// ai: arm64 takes it for 64 pairs at once, in its own operations (mark_pairs64)
#ifndef OB_NEON
static int mark_ratio(int line, int gap, int core_px, int core, float *unit) {
  const int total = line + gap + core_px;
  if (core_px < MARK_MIN_CORE || total < OB_THIN_LINE + OB_RING_DEEP + core) return 0;
  const float u = (float)total / (OB_THIN_LINE + OB_RING_DEEP + core), tol = 0.6f * u + 0.5f;
  if (fabsf(line - OB_THIN_LINE * u) > tol || fabsf(gap - OB_RING_DEEP * u) > tol) return 0;
  if (fabsf(core_px - core * u) > 1.5f * tol * core / 2) return 0;
  *unit = u;
  return 1;
}
#endif

#ifdef OB_NEON
// ai: how many pixels of column x from row y on, down (dir 1) or up (-1), are want (0 light, 1 dark) before one is not
// ai: or the image ends, at most cap: the count the byte loops below take (0 where y is outside the image)
static inline int cb_run(const uint32_t *cb, int w, int h, int x, int y, int dir, int want, int cap) {
  if (cap <= 0) return 0;
  const uint32_t flip = want ? 0 : ~0u;
  int n = 0;
  while (n < cap && y >= 0 && y < h) {
    const uint32_t word = cb[(size_t)(y >> 5) * w + x] ^ flip;   // ones where the pixel is want
    int t, avail;
    if (dir > 0) {
      const uint32_t s = ~(word >> (y & 31));
      t = s ? __builtin_ctz(s) : 32; avail = 32 - (y & 31); if (avail > h - y) avail = h - y;
    } else {
      const uint32_t s = ~(word << (31 - (y & 31)));
      t = s ? __builtin_clz(s) : 32; avail = (y & 31) + 1;
    }
    if (t > avail) t = avail;
    n += t; y += dir * t;
    if (t < avail) break;
  }
  return n < cap ? n : cap;
}
#endif
// A light run then a dark run, walked from `at` along `dir`: the gap and the line on one side of a core. This is
// what tells a mark from any other dark square in the picture, and it has to hold on the corner side only.
#ifdef OB_NEON
// ai: mark_flank on the column bits: the same walks, the same test
static int mark_flank_cb(const uint32_t *cb, int w, int h, int x, int at, int dir, float u) {
  const int cap = (int)((2 + OB_THIN_LINE + OB_RING_DEEP) * u) + 4;
  const int gap = cb_run(cb, w, h, x, at, dir, 0, cap), line = cb_run(cb, w, h, x, at + dir * gap, dir, 1, cap), i = at + dir * (gap + line);
  if (i < 0 || i >= h || (cb[(size_t)(i >> 5) * w + x] >> (i & 31) & 1)) return 0;   // the line must end in light, not run off or run on
  const float tol = 0.6f * u + 0.5f;
  return fabsf(gap - OB_RING_DEEP * u) <= tol && fabsf(line - OB_THIN_LINE * u) <= tol;
}
#else
static int mark_flank(const uint8_t *bin, int len, int stride, int at, int dir, float u) {
  int gap = 0, line = 0, i = at;
  const int cap = (int)((2 + OB_THIN_LINE + OB_RING_DEEP) * u) + 4;
  while (i >= 0 && i < len && !bin[i * stride] && gap < cap) { gap++; i += dir; }
  while (i >= 0 && i < len && bin[i * stride] && line < cap) { line++; i += dir; }
  if (i < 0 || i >= len || bin[i * stride]) return 0;   // the line must end in light, not run off or run on
  const float tol = 0.6f * u + 0.5f;
  return fabsf(gap - OB_RING_DEEP * u) <= tol && fabsf(line - OB_THIN_LINE * u) <= tol;
}
#endif

// A candidate core found on a row, confirmed down the column through it: the same extent, within what
// perspective does to a square, and the gap and line on one side of it vertically too.
// ai: cb: the column bits (binarize's) on arm64, which its walks read; NULL elsewhere
static int mark_cross(const uint8_t *bin, const uint32_t *cb, int w, int h, int cx, int cy, int core, float uh, float *centre, float *uv) {
  const int cap = (int)(3 * core * uh) + 8;
  int up = 0, dn = 0;
#ifdef OB_NEON
  up = cb_run(cb, w, h, cx, cy - 1, -1, 1, cap); dn = cb_run(cb, w, h, cx, cy + 1, 1, 1, cap);
#else
  (void)cb;
  while (cy - up - 1 >= 0 && bin[(cy - up - 1) * w + cx] && up < cap) up++;
  while (cy + dn + 1 < h && bin[(cy + dn + 1) * w + cx] && dn < cap) dn++;
#endif
  const int ext = up + dn + 1;
  const float u = (float)ext / core;
  // A screen seen from the side is squeezed along one axis, so the two crossings of one mark can differ a lot.
  // 2.5 is what the finder path allows, about 65 degrees.
  if (u > 2.5f * uh || uh > 2.5f * u) return 0;
#ifdef OB_NEON
  if (!mark_flank_cb(cb, w, h, cx, cy - up - 1, -1, u) && !mark_flank_cb(cb, w, h, cx, cy + dn + 1, 1, u)) return 0;
#else
  if (!mark_flank(bin + cx, h, w, cy - up - 1, -1, u) && !mark_flank(bin + cx, h, w, cy + dn + 1, 1, u)) return 0;   // + cx: the walk is down THIS column
#endif
  // Solid, not ragged: a picture blob can cross one column correctly and still be full of holes.
  const float my = cy - up + (ext - 1) / 2.0f, r = 0.3f * ext;
  for (int k = 0; k < 4; k++) {
    const int sx = cx + (int)((k & 1 ? r : -r) * uh / u), sy = (int)(my + 0.5f + (k & 2 ? r : -r));   // my indexes pixels; the point is my + 0.5
    if (sx < 0 || sx >= w || sy < 0 || sy >= h || !bin[sy * w + sx]) return 0;
  }
  *centre = my + 0.5f;   // my indexes pixels and sample() puts pixel i's middle at i + 0.5
  *uv = u;
  return 1;
}

// The same mark with its gap CLOSED, which is how blur shows it. binarize calls a pixel dark when it is under its
// 40 px neighbourhood's mean, so the one-module light gap between the line and the core, blurred to mid grey
// with white all round it, goes dark, and line, gap and core become one solid square of S - 2 modules. Traced
// at 1.5 px of defocus: the three-run cross-section above matches NOTHING (0 supported candidates against 4 with
// 7 to 12 hits on a sharp capture), and the line path dies of the same closure at the same moment, so without
// this form a blurred symbol has no anchor at all. A solid square has no ratio inside it to be scale-free by;
// what keeps it honest is mark_quad, where four of them have to agree on a module size and sit the known number
// of modules apart.
static int mark_cross_merged(const uint8_t *bin, const uint32_t *cb, int w, int h, int cx, int cy, int sq, float uh, float *centre, float *uv) {
  const int cap = (int)(3 * sq * uh) + 8;
  int up = 0, dn = 0;
#ifdef OB_NEON
  up = cb_run(cb, w, h, cx, cy - 1, -1, 1, cap); dn = cb_run(cb, w, h, cx, cy + 1, 1, 1, cap);
#else
  (void)cb;
  while (cy - up - 1 >= 0 && bin[(cy - up - 1) * w + cx] && up < cap) up++;
  while (cy + dn + 1 < h && bin[(cy + dn + 1) * w + cx] && dn < cap) dn++;
#endif
  if (up >= cap || dn >= cap || cy - up - 1 < 0 || cy + dn + 1 >= h) return 0;   // not bounded by light: a bar, or the frame's edge
  const int ext = up + dn + 1;
  const float u = (float)ext / sq;
  if (u > 2.5f * uh || uh > 2.5f * u) return 0;
  // Probes 1.5 modules out from the middle of a square of S - 2, capped so they stay half a module inside the
  // core: the square's middle is module S / 2, the core starts at OB_THIN_TRACK, so the cap is sq / 2 minus that.
  // Further out lands on the gap's own corner, where its two arms meet and the LAST pixel closes: traced at 1.5 px
  // of defocus, one light pixel there refused a plain mark on every row that crossed it. A deeper gap pushes the
  // core in and the cap down (0.5 modules at S = 12, OB_RING_DEEP = 2), which leaves this test weak and
  // mark_quads, where four of them must agree on a module size and a spacing, carrying the weight.
  const float d = fminf(1.5f, sq / 2.0f - (OB_THIN_TRACK - 0.5f)), my = cy - up + (ext - 1) / 2.0f, r = (d > 0 ? d : 0) * u;
  for (int k = 0; k < 4; k++) {
    const int sx = cx + (int)((k & 1 ? r : -r) * uh / u), sy = (int)(my + 0.5f + (k & 2 ? r : -r));   // my indexes pixels; the point is my + 0.5
    if (sx < 0 || sx >= w || sy < 0 || sy >= h || !bin[sy * w + sx]) return 0;
  }
  *centre = my + 0.5f;   // my indexes pixels and sample() puts pixel i's middle at i + 0.5
  *uv = u;
  return 1;
}

// Every corner mark in the image, by its own cross-section. Every second row, run boundaries sixteen pixels to a
// compare, then each pair of dark runs sharing a light gap is tried both ways round, because the core lies on
// the far side of the line from the corner and the corner can be to either side.
//
// One form a call. The merged form is any solid dark square bounded by light, which is also exactly what a grain
// of background noise is: scanned alongside the gapped form and set against it on spacing, it took amp 1 grain 2
// from 83% to 17% and grain 4 from 100% to 0%, at three times the detect time. So it is scanned only when the
// gapped form settled nothing (find_frame), and a sharp capture never pays for it.
static void mark_note(cand_t *out, int *n, float fx, float fy, float u) {
  int c = 0;
#ifdef OB_NEON
  // ai: arm64: eight candidates a step from the list's x and y side by side (mark_xy), the same compares (FABD is
  // ai: fabsf of the difference; both under the limit is their larger under it, and a NaN fails either way); the step
  // ai: holding the first that is near stops it, and the loop below finds that one
  const float32x4_t vx = vdupq_n_f32(fx), vy = vdupq_n_f32(fy), lim = vdupq_n_f32(u * 2);
  const float *const sx = mark_xy(out), *const sy = sx + MAX_CAND;
  for (; c + 8 <= *n; c += 8) {
    const float32x4_t dp = vmaxq_f32(vabdq_f32(vld1q_f32(sx + c), vx), vabdq_f32(vld1q_f32(sy + c), vy));
    const float32x4_t dq = vmaxq_f32(vabdq_f32(vld1q_f32(sx + c + 4), vx), vabdq_f32(vld1q_f32(sy + c + 4), vy));
    if (vmaxvq_u32(vorrq_u32(vcltq_f32(dp, lim), vcltq_f32(dq, lim)))) break;
  }
#endif
  for (; c < *n; c++) if (fabsf(out[c].x - fx) < u * 2 && fabsf(out[c].y - fy) < u * 2) break;
  if (c < *n) {
    out[c].x = (out[c].x * out[c].hits + fx) / (out[c].hits + 1);
    out[c].y = (out[c].y * out[c].hits + fy) / (out[c].hits + 1);
    out[c].unit = (out[c].unit * out[c].hits + u) / (out[c].hits + 1);
    out[c].hits++;
#ifdef OB_NEON
    mark_xy(out)[c] = out[c].x; mark_xy(out)[MAX_CAND + c] = out[c].y;
#endif
  } else if (*n < MAX_CAND) {
#ifdef OB_NEON
    mark_xy(out)[*n] = fx; mark_xy(out)[MAX_CAND + *n] = fy;
#endif
    out[(*n)++] = (cand_t){ fx, fy, u, 1 };
  }
}
// The gapped form at dark runs k - 1 and k of row y (pos as run_edges leaves it): each way round, the line then
// the core. noted: the core this row has counted, which met second in one pair is met first in the next.
#ifdef OB_NEON
// ai: arm64: mark_pair with mark_ratio's answers for the pair already taken (mark_pairs64): pass bit 0 side 0's, bit
// ai: 1 side 1's, and u the module size both sides share (line, gap and core sum to the same total either way round)
static inline void mark_pair_v(const uint8_t *bin, const uint32_t *cb, int w, int h, int y, int core, const int *pos, int k, unsigned pass, float u, int *noted, cand_t *out, int *n) {
  const int a0 = pos[2 * k - 2], a1 = pos[2 * k - 1], b0 = pos[2 * k], b1 = pos[2 * k + 1];
  if (a0 == 0 || b1 >= w) return;            // a run touching either end of the row has no light outside it
  for (int side = 0; side < 2; side++) {
    const int cx0 = side ? a0 : b0, cx1 = side ? a1 : b1;
    float uv, fy;
    if (!(pass >> side & 1) || cx0 == *noted) continue;
    const int cx = (cx0 + cx1) >> 1;
    if (!mark_cross(bin, cb, w, h, cx, y, core, u, &fy, &uv)) continue;
    mark_note(out, n, (cx0 + cx1) / 2.0f, fy, 0.5f * (u + uv));   // the run is [cx0, cx1) in sample()'s coordinates
    *noted = cx0;
    break;
  }
}
#else
static inline void mark_pair(const uint8_t *bin, const uint32_t *cb, int w, int h, int y, int core, const int *pos, int k, int *noted, cand_t *out, int *n) {
  const int a0 = pos[2 * k - 2], a1 = pos[2 * k - 1], b0 = pos[2 * k], b1 = pos[2 * k + 1];
  if (a0 == 0 || b1 >= w) return;            // a run touching either end of the row has no light outside it
  const int first = a1 - a0, gap = b0 - a1, second = b1 - b0;
  for (int side = 0; side < 2; side++) {
    // side 0: the corner is to the left, so the line is met first and the core second.
    const int line = side ? second : first, cpx = side ? first : second;
    const int cx0 = side ? a0 : b0, cx1 = side ? a1 : b1;
    float uh, uv, fy;
    if (cx0 == *noted) continue;
    if (!mark_ratio(line, gap, cpx, core, &uh)) continue;
    const int cx = (cx0 + cx1) >> 1;
    if (!mark_cross(bin, cb, w, h, cx, y, core, uh, &fy, &uv)) continue;
    mark_note(out, n, (cx0 + cx1) / 2.0f, fy, 0.5f * (uh + uv));   // the run is [cx0, cx1) in sample()'s coordinates
    *noted = cx0;
    break;
  }
}
#endif
// The merged form at dark run k of row y: one dark run with light at both ends that is a square of S - 2 modules
// down its column.
static inline void mark_square(const uint8_t *bin, const uint32_t *cb, int w, int h, int y, int sq, const int *pos, int k, cand_t *out, int *n) {
  const int a0 = pos[2 * k], a1 = pos[2 * k + 1], run = a1 - a0;
  if (a0 == 0 || a1 >= w || run < 8 || run > w / 3) return;
  const float uh = (float)run / sq;
  float uv, fy;
  if (!mark_cross_merged(bin, cb, w, h, (a0 + a1) >> 1, y, sq, uh, &fy, &uv)) return;
  mark_note(out, n, (a0 + a1) / 2.0f, fy, 0.5f * (uh + uv));
}
#ifdef OB_NEON
// ai: arm64: of the pairs of runs k0 - 1, k0 to k0 + 62, k0 + 63 in pos (runs to nr), the bits of those mark_ratio
// ai: passes on side 0 (the return) and on side 1 (*one), and u[g] each pair's module size (g = k - k0), four pairs a
// ai: step on mark_ratio's own float operations in its order (FABD is fabsf of the difference, the halving a multiply
// ai: by 0.5, the same value), so its answers. Reads pos to 2 nr + 5.
static uint64_t mark_pairs64(const int *pos, int k0, int nr, int core, uint64_t *one, float *uk) {
  const float32x4_t den = vdupq_n_f32((float)(OB_THIN_LINE + OB_RING_DEEP + core)), fl = vdupq_n_f32((float)OB_THIN_LINE), fg = vdupq_n_f32((float)OB_RING_DEEP), fc = vdupq_n_f32((float)core);
  const int32x4_t tmin = vdupq_n_s32(OB_THIN_LINE + OB_RING_DEEP + core), cmin = vdupq_n_s32(MARK_MIN_CORE);
  const uint32x4_t bit = { 1, 2, 4, 8 };
  uint64_t m = 0, m1 = 0;
  for (int g = 0; g < 64 && k0 + g < nr; g += 4) {
    const int32x4x2_t a = vld2q_s32(pos + 2 * (k0 + g) - 2), b = vld2q_s32(pos + 2 * (k0 + g));   // starts, ends of runs k - 1 and k
    const int32x4_t first = vsubq_s32(a.val[1], a.val[0]), gap = vsubq_s32(b.val[0], a.val[1]), second = vsubq_s32(b.val[1], b.val[0]);
    const int32x4_t total = vaddq_s32(vaddq_s32(first, gap), second);
    const float32x4_t ff = vcvtq_f32_s32(first), gf = vcvtq_f32_s32(gap), sf = vcvtq_f32_s32(second);
    const float32x4_t u = vdivq_f32(vcvtq_f32_s32(total), den), tol = vaddq_f32(vmulq_f32(vdupq_n_f32(0.6f), u), vdupq_n_f32(0.5f));
    const float32x4_t lu = vmulq_f32(fl, u), cu = vmulq_f32(fc, u), ctol = vmulq_f32(vmulq_f32(vmulq_f32(vdupq_n_f32(1.5f), tol), fc), vdupq_n_f32(0.5f));
    const uint32x4_t both = vandq_u32(vcgeq_s32(total, tmin), vcleq_f32(vabdq_f32(gf, vmulq_f32(fg, u)), tol));
    const uint32x4_t s0 = vandq_u32(vandq_u32(vcgeq_s32(second, cmin), vcleq_f32(vabdq_f32(ff, lu), tol)), vcleq_f32(vabdq_f32(sf, cu), ctol));
    const uint32x4_t s1 = vandq_u32(vandq_u32(vcgeq_s32(first, cmin), vcleq_f32(vabdq_f32(sf, lu), tol)), vcleq_f32(vabdq_f32(ff, cu), ctol));
    m |= (uint64_t)vaddvq_u32(vandq_u32(vandq_u32(both, s0), bit)) << g;
    m1 |= (uint64_t)vaddvq_u32(vandq_u32(vandq_u32(both, s1), bit)) << g;
    vst1q_f32(uk + g, u);
  }
  const uint64_t keep = nr - k0 < 64 ? (1ull << (nr - k0)) - 1 : ~0ull;
  *one = m1 & keep;
  return m & keep;
}
// ai: arm64: of runs k0 to k0 + 63 (to nr), the bits of those 8 to w / 3 pixels long, which mark_square tests
static uint64_t mark_runs64(const int *pos, int k0, int nr, int w) {
  const int32x4_t lo = vdupq_n_s32(8), hi = vdupq_n_s32(w / 3);
  const uint32x4_t bit = { 1, 2, 4, 8 };
  uint64_t m = 0;
  for (int g = 0; g < 64 && k0 + g < nr; g += 4) {
    const int32x4x2_t r = vld2q_s32(pos + 2 * (k0 + g));
    const int32x4_t run = vsubq_s32(r.val[1], r.val[0]);
    m |= (uint64_t)vaddvq_u32(vandq_u32(vandq_u32(vcgeq_s32(run, lo), vcleq_s32(run, hi)), bit)) << g;
  }
  return nr - k0 < 64 ? m & ((1ull << (nr - k0)) - 1) : m;
}
#endif
static int mark_candidates(const ob_layout_t *L, const uint8_t *bin, const uint32_t *cb, int w, int h, int merged, cand_t *out) {
  const int core = mark_core(L), sq = L->corner - 2;
  // ai: pos: run_edges' w + 2 boundaries and the vector filters' reads past them (calloc: those reads are of written memory)
  int n = 0, *pos = calloc((size_t)w + 8, sizeof(int)), minc = merged ? sq : core;
  if (!pos) return 0;
  for (int y = 1; y < h; y += 2) {
    const int np = run_edges(bin + y * w, w, pos);
    int noted = -1;
#ifdef OB_NEON
    // ai: arm64: the pairs (or runs) the vector filters keep, in the order the loops below meet them
    for (int k0 = 1; !merged && k0 < np / 2; k0 += 64) {
      uint64_t m1;
      float uk[64];
      const uint64_t m0 = mark_pairs64(pos, k0, np / 2, core, &m1, uk);
      for (uint64_t m = m0 | m1; m; m &= m - 1) {
        const int g = __builtin_ctzll(m);
        mark_pair_v(bin, cb, w, h, y, core, pos, k0 + g, (unsigned)(m0 >> g & 1) | (unsigned)(m1 >> g & 1) << 1, uk[g], &noted, out, &n);
      }
    }
    for (int k0 = 0; merged && k0 < np / 2; k0 += 64)
      for (uint64_t m = mark_runs64(pos, k0, np / 2, w); m; m &= m - 1) mark_square(bin, cb, w, h, y, sq, pos, k0 + __builtin_ctzll(m), out, &n);
#else
    for (int k = 1; !merged && 2 * k + 1 < np; k++) mark_pair(bin, cb, w, h, y, core, pos, k, &noted, out, &n);
    for (int k = 0; merged && 2 * k + 1 < np; k++) mark_square(bin, cb, w, h, y, sq, pos, k, out, &n);
#endif
    // No budget. A blurred picture offers about a thousand one-row blobs, and a list that fills before the scan
    // reaches the bottom marks has not recorded them, which is the failure SIDE_K has and the reason this finder
    // exists. A mark is crossed by every scan row through it, so a candidate the scan has left behind with fewer
    // hits than its size guarantees was never one, and dropping those keeps the list at the few dozen that matter.
    if (n > MAX_CAND / 4) {
      int m = 0;
      for (int c = 0; c < n; c++) {
        // Rows that must cross the smallest mark this call accepts, at this candidate's size: three, or fewer for a small core.
        int need = (int)(minc * out[c].unit) / 2;
        need = need > 3 ? 3 : need < 1 ? 1 : need;
        if (out[c].hits >= need || out[c].y + L->corner * out[c].unit > y) {
#ifdef OB_NEON
          mark_xy(out)[m] = mark_xy(out)[c]; mark_xy(out)[MAX_CAND + m] = mark_xy(out)[MAX_CAND + c];
#endif
          out[m++] = out[c];
        }
      }
      n = m;
    }
  }
  free(pos);
  // Every merged candidate once more, as a whole square at its accumulated middle: bounded along the row, the
  // column, both diagonals and the four directions between, within a factor of 3.2. A square turned any amount has its
  // four extents between its side and its side times root two; a straight dark band does not, since along its own
  // direction it runs on. The row and column tests alone cannot tell them apart once the band is turned: the
  // symbol's own border, blurred to one dark band at 1.9 px a module and turned 30 degrees, passed as a square on
  // every scan row along it, and those crowded the four real marks out of mark_quads' pool (ranked about 140th
  // where the pool keeps 24; 4th to 13th unturned). Here and not per row: under heavy blur a merged mark is joined
  // to the border line on two sides, a row-by-row test turns away the rows on those sides, and the middle the rest
  // average to moved half a pixel inwards, enough at 2.5 px of defocus to fail the track's contrast gate.
  //
  // The module size comes from here too, the four extents' geometric mean: the row and column overstate a turned
  // square by 1 / cos of the turn, up to root two at 45 degrees, which put the real quad's spacing that far off;
  // the mean of four directions 45 degrees apart is 1.08 to 1.19 of the side at any turn, within 5% over 1.13. Not
  // the shortest extent, which picks the foreshortened axis of a screen seen from the side (at 30 degrees of yaw
  // the near and far marks came out more than two to one, which mark_quads refuses).
  if (merged) {
    int m = 0;
    // Eight directions through the middle, a band's own direction never more than 13 degrees from one of them:
    // 0, 45, 90, 135 and the four between on steps of (2, 1). The first four give the module size.
    static const int DIR[8][2] = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { 1, -1 }, { 2, 1 }, { 1, 2 }, { -1, 2 }, { -2, 1 } };
    static const float LEN[8] = { 1, 1, 1.41421356f, 1.41421356f, 2.23606798f, 2.23606798f, 2.23606798f, 2.23606798f };
    for (int k = 0; k < n; k++) {
      const int x = (int)out[k].x, y = (int)out[k].y, cap = (int)(3 * sq * out[k].unit) + 8;
      float ext[8];
      int ok = x >= 0 && x < w && y >= 0 && y < h && bin[y * w + x];
      for (int d = 0; ok && d < 8; d++) {
        int run = 1;
        for (int sgn = -1; ok && sgn <= 1; sgn += 2) {
          int t = 0, px = x, py = y;
          for (;;) {
            const int nx = px + sgn * DIR[d][0], ny = py + sgn * DIR[d][1];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) { ok = 0; break; }   // ran off the frame, not into light
            if (!bin[ny * w + nx]) break;
            if (++t * LEN[d] >= cap) { ok = 0; break; }                      // runs on: a band, not a square
            px = nx; py = ny;
          }
          run += t;
        }
        ext[d] = run * LEN[d];
      }
      if (!ok) continue;
      float lo = ext[0], hi = ext[0];
      for (int d = 1; d < 8; d++) { lo = fminf(lo, ext[d]); hi = fmaxf(hi, ext[d]); }
      // A band's extents are 4.2 or more apart at any turn; a square's are root two apart, and a mark blurred into
      // the prefix beside it into a lopsided blob came to about three (2.8 on four directions, traced at 2.5 px).
      if (hi > 3.2f * lo) continue;
      out[k].unit = sqrtf(sqrtf(ext[0] * ext[1] * ext[2] * ext[3])) / (1.13f * sq);
      out[m++] = out[k];
    }
    n = m;
  }
  return n;
}

// Four marks chosen the way zxing chooses finder patterns: not by where they are but by whether they AGREE with
// each other (zxing's QR detector rejects a triple of finder patterns on size ratio, leg ratio, implied module count
// and angle). The widest convex quad picks picture texture here: the picture offers
// dozens of small things with a mark's cross-section, and four of them always span more of the image than the marks do.
//
// LIZARD can press the module-count test much harder than QR can, because the count is not being estimated: two
// adjacent marks are exactly L->w - 2 * mark_mid modules apart and a mark's own cross-section gives the module
// size, so a quad whose side over its module size is not that number is not this symbol.
// One form at a time. The two forms have different centres (the gapped core's middle is (OB_THIN_TRACK + S - 1) / 2 in
// from the corner, 8 at S = 12; the merged square's is S / 2), and blur is the same all over a capture, so four marks
// of one symbol are one form; mixing them would only let a picture blob stand in for a mark. `score` comes back as the quad's
// disagreement with the known spacing, smaller is better, so the caller can set the forms against each other.
//
// Which quad is RIGHT is not decided here. Spacing ranks them; the track read in find_frame is what believes one,
// the way zxing walks its pattern sets until one decodes. So the best few come back, not the best one.
// form: 0 gapped, 1 merged, 2 the merged scan's squares read as a sharp mark's core (find_frame). A core's middle
// is the gapped middle, since the core is what the gapped form's cross-section is centred on.
static float mark_form_mid(const ob_layout_t *L, int form) { return form == 1 ? L->corner / 2.0f : mark_mid(L); }

// The mark geometry a finder outside the codec needs to be scale-free the way mark_ratio and mark_quads are
// (gpu/tables.mjs, scripts/exp/fmt_word.mjs). Nothing in the decoder calls this. A blob detector measures a mark in PIXELS and has
// no module size of its own, so it has to get one the way this file does: divide the blob by the mark's known
// module count, then check that four of them sit the known number of modules apart. Both forms are given
// because a threshold that keeps the light gap open finds the CORE and one that closes it finds the whole
// square, and the two have different centres, so they imply different quads from the same four blobs.
// The last two are what an external quad is MEASURED AGAINST: acquire_core reads one through corner_coords,
// which spans the mesh's outermost nodes and not modules 0 to w. Seven modules of difference on a 286 module
// symbol, which is 2.5% of the quad and far more than the registration tolerates.
//   0 w  1 corner  2 core modules (gapped)  3 square modules (merged)  4 mid gapped  5 mid merged
//   6 node span x  7 node span y
void ob_mark_geom(const ob_layout_t *L, float *out) {
  out[0] = (float)L->w;
  out[1] = (float)L->corner;
  out[2] = (float)mark_core(L);
  out[3] = (float)(L->corner - 2);
  out[4] = mark_form_mid(L, 0);
  out[5] = mark_form_mid(L, 1);
  out[6] = L->nx > 1 ? L->node_x[L->nx - 1] - L->node_x[0] : (float)L->w;
  out[7] = L->ny > 1 ? L->node_y[L->ny - 1] - L->node_y[0] : (float)L->h;
}
enum { MARK_POOL = 24, MARK_QUADS = 4 };
typedef struct { cand_t q[4]; float score; } mquad_t;
// Four marks at the corners, as a quad. The pool is ranked by support TIMES size, not support alone: a mark is
// the biggest solid thing of its kind in the symbol, and a blurred picture is full of smaller blobs that a scan
// crosses nearly as often. Ranked by hits alone, a true mark seen on four rows (traced: the top right one at
// 1.5 px of defocus) lost its place in the pool to blobs two thirds its size seen on five.
//
// A TRIPLE was tried on 2026-09-21, QR's three corners with the fourth inferred as a parallelogram's. It reads
// its own rotation, which is elegant, and it costs the perspective: three points determine an affine map, and
// the term that makes a square photograph as a trapezium is exactly what the fourth measured point supplies.
// The edge marks were meant to pay that back, and they are found in the captures where the map was already
// right and missed in the ones where it is not. fill 0.3 fell from 2990 B on 16 of 16 frames to 352 B on 7.
// ai: The pool, and every pair's distance in module sizes, once a form: they hold for every layout a receiver looks
// ai: for (only the spacing the layout asks for, want, differs), so mark_quads takes them for each of the rings.
typedef struct { cand_t c[MARK_POOL]; int n; float mods[MARK_POOL][MARK_POOL]; } mpool_t;
static void mark_pool(const cand_t *all, int nall, mpool_t *P) {
  cand_t *c = P->c;
  int n = 0;
  #define RANK(k) ((k).hits * (k).unit)
  for (int i = 0; i < nall; i++) {
    int j;
    if (n < MARK_POOL) j = n++;
    else { if (RANK(c[MARK_POOL - 1]) >= RANK(all[i])) continue; j = MARK_POOL - 1; }
    while (j > 0 && RANK(c[j - 1]) < RANK(all[i])) { c[j] = c[j - 1]; j--; }
    c[j] = all[i];
  }
  #undef RANK
  P->n = n;
  for (int a = 0; a < n; a++) for (int b = a; b < n; b++) {
    const float dx = c[b].x - c[a].x, dy = c[b].y - c[a].y;
    P->mods[a][b] = sqrtf(dx * dx + dy * dy) / (0.5f * (c[a].unit + c[b].unit));
  }
}
static int mark_quads(const ob_layout_t *L, const mpool_t *P, int merged, mquad_t *out) {
  const float want = L->w - 2 * mark_form_mid(L, merged);
  const cand_t *c = P->c;
  const int n = P->n;
  int nq = 0;
  // Whether two candidates can be ADJACENT marks, and how far off the known spacing they are if so. Most pairs
  // cannot, so walking four-cycles of this table visits a few hundred quads where every four of the pool is
  // 10626, each of which used to be sorted round its centroid by atan2 before it could be refused.
  // rat: a side's length over the module size its two ends report, as a share of the known count, or -1 where
  // the two cannot be adjacent marks at all.
  float rat[MARK_POOL][MARK_POOL];
  // ai: adj[a]: bit b where rat[a][b] >= 0, so the walk below takes each next corner from a mask, in the same order
  uint32_t adj[MARK_POOL] = { 0 };
  _Static_assert(MARK_POOL <= 32, "a corner's neighbours are the bits of one word");
  for (int a = 0; a < n; a++) for (int b = a; b < n; b++) {
    const float mods = P->mods[a][b];
    const int can = a != b && mods >= 0.6f * want && mods <= 1.5f * want;   // unit shrinks with the same tilt, so a squeezed side reads 2cos / (1 + cos): 0.6 is about 65 degrees
    rat[a][b] = rat[b][a] = can ? mods / want : -1;
    if (can) { adj[a] |= 1u << b; adj[b] |= 1u << a; }
  }
  // a is the cycle's lowest index and b < e, so each cycle is met once and not again backwards. No corner is its own
  // neighbour, so d != b and e != d hold of the masks.
  for (int a = 0; a < n; a++) for (uint32_t mb = adj[a] & (~1u << a); mb; mb &= mb - 1) {
    const int b = __builtin_ctz(mb);
    for (uint32_t md = adj[b] & (~1u << a); md; md &= md - 1) {
      const int d = __builtin_ctz(md);
      for (uint32_t me = adj[d] & adj[a] & (~1u << b); me; me &= me - 1) {
        const int e = __builtin_ctz(me);
        // The gapped form reads its module size from a ratio of line, gap and core, which a threshold that shaves
        // dark runs leaves alone, so its sides are scored on their distance from the count. The two square forms
        // read it from a dark square alone, and the binarizer shaves every one of them alike, by 10 to 25% on a
        // 6-module core of 11 to 20 px: four real marks then agree with each other and are off the count
        // TOGETHER (1.12 on every side, traced at 40 degrees of turn), where four picture blobs do not agree. So
        // for those the score is the sides' spread about their own mean plus half the mean's distance from the
        // count, which still keeps a wrong count out. Not for the gapped form: under perspective its four real
        // sides legitimately disagree, and blobs that happen to agree then outrank it (30 degrees of yaw).
        // ai: The score first: it reads the cycle's four sides, the same whichever way round the cycle is taken, so a
        // ai: quad that could not enter the best MARK_QUADS is refused before the module-size and convexity tests (the
        // ai: same quads kept; 0.17 to 0.44 M instructions a frame fewer on the S26 runs).
        const float r4[4] = { rat[a][b], rat[b][d], rat[d][e], rat[e][a] }, k = 0.25f * (r4[0] + r4[1] + r4[2] + r4[3]);
        const float score = merged ? fabsf(r4[0] - k) + fabsf(r4[1] - k) + fabsf(r4[2] - k) + fabsf(r4[3] - k) + 0.5f * fabsf(k - 1)
                                   : fabsf(r4[0] - 1) + fabsf(r4[1] - 1) + fabsf(r4[2] - 1) + fabsf(r4[3] - 1);
        if (nq == MARK_QUADS && out[MARK_QUADS - 1].score <= score) continue;
        int id[4] = { a, b, d, e };
        float umin = 1e9f, umax = 0;
        for (int k = 0; k < 4; k++) { umin = fminf(umin, c[id[k]].unit); umax = fmaxf(umax, c[id[k]].unit); }
        if (umax > 2 * umin) continue;   // one symbol, one module size, within what perspective does to it
        int pos = 0, neg = 0;
        for (int k = 0; k < 4; k++) { const float x = cross2(&c[id[k]], &c[id[(k + 1) & 3]], &c[id[(k + 2) & 3]]); pos += x > 0; neg += x < 0; }
        if (pos != 4 && neg != 4) continue;   // convex
        if (neg) { id[1] = e; id[3] = b; }
        int j;
        if (nq < MARK_QUADS) j = nq++;
        else { if (out[MARK_QUADS - 1].score <= score) continue; j = MARK_QUADS - 1; }
        while (j > 0 && out[j - 1].score > score) { out[j] = out[j - 1]; j--; }
        for (int k = 0; k < 4; k++) out[j].q[k] = c[id[k]];
        out[j].score = score;
      }
    }
  }
  // Each starts at the corner nearest the image's top left as seen from its own middle, which is what the
  // hypothesis numbering downstream was built on.
  for (int i = 0; i < nq; i++) {
    cand_t *q = out[i].q, r[4];
    float gx = 0, gy = 0, lo = 1e9f;
    int first = 0;
    for (int k = 0; k < 4; k++) { gx += q[k].x / 4; gy += q[k].y / 4; }
    for (int k = 0; k < 4; k++) { const float an = atan2f(q[k].y - gy, q[k].x - gx); if (an < lo) { lo = an; first = k; } }
    for (int k = 0; k < 4; k++) r[k] = q[(first + k) & 3];
    memcpy(q, r, sizeof r);
  }
  return nq;
}

// What a mesh built elsewhere has to be built against: the layout's node lattice, the per-module kind map and which
// nodes carry a mark. Read once a configuration. dims: nx, ny, w, h, corner.
int ob_test_mesh_tables(const ob_layout_t *L, int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark) {
  dims[0] = L->nx; dims[1] = L->ny; dims[2] = L->w; dims[3] = L->h; dims[4] = L->corner;
  if (!nodeX) return 1;
  for (int i = 0; i < L->nx; i++) nodeX[i] = L->node_x[i];
  for (int j = 0; j < L->ny; j++) nodeY[j] = L->node_y[j];
  memcpy(kind, L->kind, (size_t)L->w * L->h);
  for (int k = 0; k < L->nx * L->ny; k++) mark[k] = L->node_mark[k];
  return 1;
}
// Every mark of one form in a binarized image at 1 / scale of the capture, in the capture's own coordinates. The
// candidates depend on the mark's design alone, never on how many modules the symbol has, so one scan serves every
// layout a receiver is looking for.
// ai: cb: the image's column bits on arm64 (binarize's), NULL elsewhere
static int mark_scan(const ob_layout_t *L, const uint8_t *bin, const uint32_t *cb, int bw, int bh, int scale, int merged, cand_t *c) {
  const int n = mark_candidates(L, bin, cb, bw, bh, merged, c);
  for (int k = 0; k < n; k++) { c[k].x *= scale; c[k].y *= scale; c[k].unit *= scale; }
  return n;
}

// The mark centres of one form as quads, best spacing first, and the rectangle they make in layout modules.
static int mark_rect(const ob_layout_t *L, const mpool_t *P, int n, int iw, int ih, int merged, float *src, mquad_t *quads) {
  if (n < 4) return 0;
  const float m = mark_form_mid(L, merged), s4[8] = { m, m, L->w - m, m, L->w - m, L->h - m, m, L->h - m };
  memcpy(src, s4, sizeof s4);
  int nq = mark_quads(L, P, merged, quads), keep = 0;
  for (int i = 0; i < nq; i++) {
    const cand_t *q = quads[i].q;
    float x0 = q[0].x, x1 = x0, y0 = q[0].y, y1 = y0;
    for (int k = 1; k < 4; k++) { x0 = fminf(x0, q[k].x); x1 = fmaxf(x1, q[k].x); y0 = fminf(y0, q[k].y); y1 = fmaxf(y1, q[k].y); }
    if (x1 - x0 > 0.08f * iw && y1 - y0 > 0.08f * ih) quads[keep++] = quads[i];   // by extent: which corner a quad starts at turns with the symbol
  }
  return keep;
}

// Up to SIDE_LINES curves, best supported first. A bezel's inner edge is a long straight edge too
// and is met first, and the frame's own track runs two modules inside the line; the caller tells
// them apart by reading the track.
//
// The model is a shallow parabola from the start, through three points. A straight line cannot do
// it: a lens bows the border by more than any tolerance tight enough to keep out a line that merely
// crosses it, and then the border scores no better than the crossing. The bow is bounded (1.5% of
// the span and a pixel), which throws out most random triples before they are scored.
//
// Triples are drawn within one rank. Border points are under a fifth of a side's candidates, so a
// triple drawn from all of them is all border less than once in a hundred; but along a side the
// border is the same candidate from the outside nearly everywhere (first, or second behind a
// bezel), so within its rank nearly every triple is good. Support is counted over every point.
//
// A noisy surround destroys that invariant, and a pass over every point at once, ignoring rank, with
// five times the draws, was tried and does NOT get it back (the noisy-surround cells, unchanged in every
// cell). The reason is upstream of the fit: a scan STOPS after SIDE_K candidates, so when noise
// supplies eight before the border the border's points were never recorded, and no draw over what
// was recorded can find them. Raising SIDE_K does record them, and then the frame registers and the
// payload still does not come back, at three to eight times the detect time and at the cost of clean
// captures (SIDE_K 64: 1080 falls to 67%, detect 2.6 ms to 20). Neither is the answer.
static int fit_lines(const fpt_t *p, int n, float tol, float min_span, fline_t *out) {
  fline_t cand[SIDE_K];
  int nc = 0, *idx = malloc((size_t)n * sizeof(int)), third = n / 3 + 1, m3[3] = { 0, 0, 0 };
  // The points as plain arrays, whole and as the three every-third subsets the hypotheses are ranked on.
  float *soa = malloc(((size_t)2 * n + 6 * third) * sizeof(float)), *al = soa, *ac = soa + n, *al3[3], *ac3[3];
  for (int k = 0; k < 3; k++) { al3[k] = soa + 2 * n + 2 * k * third; ac3[k] = al3[k] + third; }
  for (int k = 0; k < n; k++) { al[k] = p[k].along; ac[k] = p[k].across; al3[k % 3][m3[k % 3]] = p[k].along; ac3[k % 3][m3[k % 3]++] = p[k].across; }
  uint32_t s = 12345u;
  #define RND(m) (s = s * 1664525u + 1013904223u, (int)((s >> 8) % (uint32_t)(m)))
#ifdef OB_NEON
  // ai: arm64: every rank's points in one pass, each rank's indices in their order (a counting sort on rank)
  int rank_at[SIDE_K + 1] = { 0 };
  for (int k = 0; k < n; k++) rank_at[p[k].rank + 1]++;
  for (int r = 0; r < SIDE_K; r++) rank_at[r + 1] += rank_at[r];
  { int fill[SIDE_K]; memcpy(fill, rank_at, sizeof fill); for (int k = 0; k < n; k++) idx[fill[p[k].rank]++] = k; }
#endif
  for (int rank = 0; rank < SIDE_K; rank++) {
#ifdef OB_NEON
    const int *ix = idx + rank_at[rank], m = rank_at[rank + 1] - rank_at[rank];
#else
    int m = 0;
    const int *ix = idx;
    for (int k = 0; k < n; k++) if (p[k].rank == rank) idx[m++] = k;
#endif
    if (m < 12) continue;
    fline_t best = { 0, 0, 0, 0, 0 };
    for (int it = 0; it < 120; it++) {
      int i0 = ix[RND(m)], i1 = ix[RND(m)], i2 = ix[RND(m)];
      if (p[i0].along > p[i1].along) { int q = i0; i0 = i1; i1 = q; }
      if (p[i1].along > p[i2].along) { int q = i1; i1 = i2; i2 = q; }
      if (p[i0].along > p[i1].along) { int q = i0; i0 = i1; i1 = q; }
      float d0 = p[i0].along - p[i1].along, d2 = p[i2].along - p[i1].along, span = d2 - d0;
      if (span < min_span || -d0 < 0.2f * span || d2 < 0.2f * span) continue;
      float c = ((p[i2].across - p[i1].across) / d2 - (p[i0].across - p[i1].across) / d0) / span, a = (p[i2].across - p[i1].across) / d2 - c * d2;
      if (fabsf(a) > 1 || fabsf(c) * 0.25f * span * span > 0.015f * span + 1) continue;
      fline_t h = { a, p[i1].across - a * p[i1].along, 0, c, p[i1].along };
      float lo, hi;
      // Every third point: enough to rank hypotheses, and not a multiple of SIDE_K, which would show each one a single rank.
      h.n = curve_support(&h, al3[it % 3], ac3[it % 3], m3[it % 3], tol, &lo, &hi);
      if (h.n > best.n && hi - lo >= min_span) best = h;
    }
    if (best.n < 4) continue;
    refit_curve(al, ac, n, tol, &best);
    if (best.n < 12) continue;
    // Two ranks often find the same curve (the border is second behind a bezel on one stretch and first on another).
    int dup = 0;
    for (int k = 0; k < nc; k++) if (fabsf(line_at(&cand[k], best.mid) - line_at(&best, best.mid)) < 2 * tol) { dup = 1; if (best.n > cand[k].n) cand[k] = best; }
    if (!dup) cand[nc++] = best;
  }
  #undef RND
  free(idx); free(soa);
  for (int i = 0; i < nc; i++) for (int j = i + 1; j < nc; j++) if (cand[j].n > cand[i].n) { fline_t q = cand[i]; cand[i] = cand[j]; cand[j] = q; }
  if (nc > SIDE_LINES) nc = SIDE_LINES;
  for (int i = 0; i < nc; i++) out[i] = cand[i];
  return nc;
}

// Where a near-vertical side (x from y) meets a near-horizontal one (y from x). Both are nearly
// straight, so stepping from one to the other settles in a few rounds.
static void meet(const fline_t *v, const fline_t *hz, float *x, float *y) {
  *x = (v->a * hz->b + v->b) / (1 - v->a * hz->a);
  *y = hz->a * *x + hz->b;
  for (int it = 0; it < 4; it++) { *x = line_at(v, *y); *y = line_at(hz, *x); }
}

// The depth the fitted curves run at. They follow the inner edge of the FIRST DARK RUN a scan meets, and which run
// that is depends on the blur: the solid line alone (depth 1 + OB_THIN_LINE) while the one-module light gap inside
// it survives, or line, track and word merged into one (depth OB_THIN_FMT + 1) once it does not. binarize
// marks a pixel dark when it is under its 40 px neighbourhood's mean, so a gap blurred to mid grey closes as soon
// as that neighbourhood is bright, which a border wider than six modules makes it (traced at 1.5 px of defocus:
// every fitted side 2.5 to 3.1 modules inside the truth, track score 0.01). find_frame tries both; every reader
// below has to agree with it about which one a curve is. One decode at a time, so a file static serves.
static OB_TLS float thin_edge_depth = 1 + OB_THIN_LINE;

// A point of the border pattern in the image. H places it as if the sides were straight; a lens bows
// them, by more than the solid line is wide when the distortion is strong. The fitted curve of that
// side says where the solid line really runs, so the point is moved across the side by the distance
// between the curve and where H puts the solid line at the same place along the side.
// cv[0..3]: the image's top, right, bottom and left curves. m: which of them this layout side lies on.
#ifndef OB_NEON   // ai: arm64's track readers have their own (trk_side, trk_bend)
static void thin_point(const homo_t *H, const fline_t *const *cv, int m, float x, float y, float xs, float ys, float *u, float *v) {
  float lu, lv;
  project(H, x, y, u, v);
  if (!cv) return;   // a quad from the corner marks has no fitted sides, so there is no bow to take out: H alone
  project(H, xs, ys, &lu, &lv);
  if (m & 1) *u += line_at(cv[m], lv) - lu; else *v += line_at(cv[m], lu) - lv;
}
#endif

// Layout coordinates on side 0..3 (top, right, bottom, left) at module t along it and depth d from the edge.
static void side_xy(const ob_layout_t *L, int side, float t, float d, float *x, float *y) {
  *x = side == 0 || side == 2 ? t : side == 3 ? d : L->w - d;
  *y = side == 0 ? d : side == 2 ? L->h - d : t;
}

#ifdef OB_SIMD
// The modules at the middle of track cells j .. j + 3, taken through the macro one cell at a time rather than
// stepped from the first. It was stepped until 2026-09-21, when the band's runs made the spacing non-uniform and
// the vector build read the format word's cells as the track's: it registered off the scalar build's quad by up
// to 0.8 px and lost frames the scalar kept. The runs are gone, but a reader that asks the layout where a cell
// is cannot be wrong about it again.
#define TRACK_AT4(L, j) wasm_f32x4_make((float)((L)->reserve + OB_TRACK_AT(j)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 1)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 2)) + OB_CELL_MID, \
                                        (float)((L)->reserve + OB_TRACK_AT((j) + 3)) + OB_CELL_MID)
#ifdef OB_NEON
// ai: arm64's track reads (thin_levels, thin_score, thin_read_both, each its own form), their own floats: a
// ai: side's map at one depth folded to u = (hu t + ku) / w, v = (hv t + kv) / w, w = hw t + kw in the module t along
// ai: it, fused, one division for both; the bow's curve fused; the read sample4r (internal.h) at gamma 1, sample4
// ai: otherwise. Each side's maps, curve and frame are taken into locals before its loop (through the pointers, a store
// ai: to the reading reloaded them every step), and the bow and the side's parity are constants of the loop. Every
// ai: layout's track cells a side, B / 2, are a multiple of 4: none of these loops has a tail.
typedef struct { float hu, ku, hv, kv, hw, kw; } trk_map_t;
static inline trk_map_t trk_map(const ob_layout_t *L, const homo_t *H, int side, float d) {
  const float *h = H->h, f = side == 0 || side == 3 ? d : side == 2 ? L->h - d : L->w - d;
  const int a = side & 1;   // t is x on sides 0 and 2, y on 1 and 3
  return (trk_map_t){ h[a], h[1 - a] * f + h[2], h[3 + a], h[4 - a] * f + h[5], h[6 + a], h[7 - a] * f + 1 };
}
static inline __attribute__((always_inline)) float32x4_t trk_w(const trk_map_t *p, float32x4_t t) { return vfmaq_n_f32(vdupq_n_f32(p->kw), t, p->hw); }
static inline __attribute__((always_inline)) void trk_uv(const trk_map_t *p, float32x4_t t, float32x4_t r, float32x4_t *u, float32x4_t *v) {
  *u = vmulq_f32(vfmaq_n_f32(vdupq_n_f32(p->ku), t, p->hu), r);
  *v = vmulq_f32(vfmaq_n_f32(vdupq_n_f32(p->kv), t, p->hv), r);
}
// a side as one depth reads it: its point's map, and the bow (thin_point): the map at the curves' depth, the curve
// of image side m, odd where m is (the curve gives u from v)
typedef struct { trk_map_t pt, bw; float c, a, b, mid; int odd, bow; } trk_side_t;
static inline trk_side_t trk_side(const ob_layout_t *L, const homo_t *H, const fline_t *cv, int m, int side, float d) {
  trk_side_t s = { trk_map(L, H, side, d), trk_map(L, H, side, thin_edge_depth), 0, 0, 0, 0, m & 1, cv != NULL };
  if (cv) { s.c = cv->c; s.a = cv->a; s.b = cv->b; s.mid = cv->mid; }
  return s;
}
// the frame as sample() reads it
typedef struct { const image_t *im; const uint8_t *px; int w, lin; float wm, hm; } trk_img_t;
static inline trk_img_t trk_img(const image_t *im) { return (trk_img_t){ im, im->px, im->w, im->lin, im->w - 1.001f, im->h - 1.001f }; }
static inline __attribute__((always_inline)) float32x4_t trk_px(const trk_img_t *g, float32x4_t u, float32x4_t v) {
  if (!g->lin) return vreinterpretq_f32_s32(sample4(g->im, vreinterpretq_s32_f32(u), vreinterpretq_s32_f32(v)));
  const float32x4_t half = vdupq_n_f32(0.5f), zero = vdupq_n_f32(0);
  const float32x4_t x = vminq_f32(vmaxq_f32(vsubq_f32(u, half), zero), vdupq_n_f32(g->wm));
  const float32x4_t y = vminq_f32(vmaxq_f32(vsubq_f32(v, half), zero), vdupq_n_f32(g->hm));
  return sample4r(g->px, g->w, x, y);
}
// what the bow adds across the side (to u where odd, else to v) at t, given the curves' map's reciprocal r
static inline __attribute__((always_inline)) float32x4_t trk_bend(const trk_side_t *s, float32x4_t t, float32x4_t r, const int odd) {
  float32x4_t lu, lv;
  trk_uv(&s->bw, t, r, &lu, &lv);
  const float32x4_t at = odd ? lv : lu, dd = vsubq_f32(at, vdupq_n_f32(s->mid));
  return vsubq_f32(vfmaq_f32(vfmaq_n_f32(vdupq_n_f32(s->b), at, s->a), vmulq_n_f32(dd, s->c), dd), odd ? lu : lv);
}
// the side's grey at four points t; bow and odd s's own, as constants
static inline __attribute__((always_inline)) float32x4_t trk_get(const trk_side_t *s, const trk_img_t *g, float32x4_t t, const int bow, const int odd) {
  const float32x4_t one = vdupq_n_f32(1);
  float32x4_t u, v;
  trk_uv(&s->pt, t, vdivq_f32(one, trk_w(&s->pt, t)), &u, &v);
  if (bow) { const float32x4_t e = trk_bend(s, t, vdivq_f32(one, trk_w(&s->bw, t)), odd); if (odd) u = vaddq_f32(u, e); else v = vaddq_f32(v, e); }
  return trk_px(g, u, v);
}
static inline __attribute__((always_inline)) float32x4_t trk_get_any(const trk_side_t *s, const trk_img_t *g, float32x4_t t) {
  return !s->bow ? trk_get(s, g, t, 0, 0) : s->odd ? trk_get(s, g, t, 1, 1) : trk_get(s, g, t, 1, 0);
}
// a row of thin_read_both's reading: the track cells at w0 + sg pos[i] (pos: each cell's module along the side)
static inline __attribute__((always_inline)) void trk_row(const trk_side_t *s, const trk_img_t *g, const float *pos, int npts, float32x4_t w0, float32x4_t sg, float *out, const int bow, const int odd) {
  for (int i = 0; i < npts; i += 4) vst1q_f32(out + i, trk_get(s, g, vfmaq_f32(w0, vld1q_f32(pos + i), sg), bow, odd));
}
// ai: the track's signs (ob_thin_track) as the sign bit a cell's -q takes, 0 where the cell is dark, side by side,
// ai: made once a thread; TRK_SIGNS cells a side reach the 512 ring (the 256, the largest, has 128)
enum { TRK_SIGNS = 256 };
static OB_TLS uint32_t trk_sign_tab[4][TRK_SIGNS];
static OB_TLS int trk_sign_made;
static const uint32_t *trk_signs(int side) {
  if (!trk_sign_made) {
    for (int sd = 0; sd < 4; sd++) for (int j = 0; j < TRK_SIGNS; j++) trk_sign_tab[sd][j] = ob_thin_track(sd, j) ? 0 : 0x80000000u;
    trk_sign_made = 1;
  }
  return trk_sign_tab[side];
}
#else
// thin_point and the read for four points of one side, at module positions `along` and depth d: the same
// operations in the same order as the scalar pair, so the same floats.
static inline v128_t thin_sample4(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *cv, int m, int side, v128_t along, float d) {
  const float *h = H->h, ds = thin_edge_depth;
  const v128_t one = wasm_f32x4_splat(1);
  v128_t x = along, y = along, xs = along, ys = along;
  if (side == 0 || side == 2) { y = wasm_f32x4_splat(side == 0 ? d : L->h - d); ys = wasm_f32x4_splat(side == 0 ? ds : L->h - ds); }
  else { x = wasm_f32x4_splat(side == 3 ? d : L->w - d); xs = wasm_f32x4_splat(side == 3 ? ds : L->w - ds); }
  #define PROJ4(X, Y, U, V) { const v128_t w_ = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[6]), X), wasm_f32x4_mul(wasm_f32x4_splat(h[7]), Y)), one); \
    U = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[0]), X), wasm_f32x4_mul(wasm_f32x4_splat(h[1]), Y)), wasm_f32x4_splat(h[2])), w_); \
    V = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[3]), X), wasm_f32x4_mul(wasm_f32x4_splat(h[4]), Y)), wasm_f32x4_splat(h[5])), w_); }
  v128_t u, v, lu, lv;
  PROJ4(x, y, u, v);
  if (!cv) return sample4(im, u, v);   // no fitted side to take a bow out of: H alone
  PROJ4(xs, ys, lu, lv);
  #undef PROJ4
  const v128_t at = m & 1 ? lv : lu, dd = wasm_f32x4_sub(at, wasm_f32x4_splat(cv->mid));
  const v128_t on = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(cv->c), dd), dd), wasm_f32x4_mul(wasm_f32x4_splat(cv->a), at)), wasm_f32x4_splat(cv->b));
  if (m & 1) u = wasm_f32x4_add(u, wasm_f32x4_sub(on, lu)); else v = wasm_f32x4_add(v, wasm_f32x4_sub(on, lv));
  return sample4(im, u, v);
}
#endif
#endif

// Grey levels of the solid line and of the light line inside it, and the threshold from the track pairs where
// `map` puts them. The line and the gap are the same for every rotation and mirror image of a quad, so the contrast
// alone rejects a quad (a bezel is dark where the light line should be).
#ifdef OB_NEON
// ai: arm64's form: each side's three depths taken into locals once (trk_side), four steps a vector, a last vector's
// ai: lanes past the side's end read and dropped; the sums in the steps' order
static int thin_levels(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, const int *map, float *ref, float *con) {
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  float dark = 0, light = 0, trk = 0;
  int nref = 0;
  for (int side = 0; side < 4; side++) {
    const int end = (side & 1 ? L->h : L->w) - L->reserve;
    const fline_t *c = cv ? cv[map[side]] : NULL;
    const trk_side_t sl = trk_side(L, H, c, map[side], side, dl), sp = trk_side(L, H, c, map[side], side, OB_GAP_MID), st = trk_side(L, H, c, map[side], side, OB_TRACK_MID);
    for (int t = L->reserve; t < end; t += 64) {
      const float32x4_t at = { t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f };
      // The pair near each of the four points, taken through the macro for the reason TRACK_AT4 gives.
      #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
      const float32x4_t ca = { PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0) }, cb = { PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1) };
      #undef PAIR_AT
      float D[4], G[4], A[4], B[4];
      vst1q_f32(D, trk_get_any(&sl, &g, at)); vst1q_f32(G, trk_get_any(&sp, &g, at));
      vst1q_f32(A, trk_get_any(&st, &g, ca)); vst1q_f32(B, trk_get_any(&st, &g, cb));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nref++) { dark += D[l]; light += G[l]; trk += A[l]; trk += B[l]; }
    }
  }
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}
#else
static int thin_levels(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, const int *map, float *ref, float *con) {
  float dark = 0, light = 0, trk = 0, u, v, x, y, xs, ys, dl = 1 + OB_THIN_LINE / 2.0f;
  int nref = 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
#ifdef OB_SIMD
    if (t + 48 < (side & 1 ? L->h : L->w) - L->reserve) {
      const v128_t along = wasm_f32x4_make(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f);
      // The pair near each of the four points, taken through the macro for the reason TRACK_AT4 gives.
      #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
      const v128_t ca = wasm_f32x4_make(PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0)),
                   cb = wasm_f32x4_make(PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1));
      #undef PAIR_AT
      float D[4], G[4], A[4], B[4];
      wasm_v128_store(D, thin_sample4(L, im, H, cv ? cv[map[side]] : NULL, map[side], side, along, dl));
      wasm_v128_store(G, thin_sample4(L, im, H, cv ? cv[map[side]] : NULL, map[side], side, along, OB_GAP_MID));
      wasm_v128_store(A, thin_sample4(L, im, H, cv ? cv[map[side]] : NULL, map[side], side, ca, OB_TRACK_MID));
      wasm_v128_store(B, thin_sample4(L, im, H, cv ? cv[map[side]] : NULL, map[side], side, cb, OB_TRACK_MID));
      for (int l = 0; l < 4; l++) { dark += D[l]; light += G[l]; trk += A[l]; trk += B[l]; }
      t += 48; nref += 3;
      continue;
    }
#endif
    side_xy(L, side, t + 0.5f, thin_edge_depth, &xs, &ys);   // on the curve: the first dark run's inner edge
    side_xy(L, side, t + 0.5f, dl, &x, &y);
    thin_point(H, cv, map[side], x, y, xs, ys, &u, &v); dark += sample(im, u, v);
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
    thin_point(H, cv, map[side], x, y, xs, ys, &u, &v); light += sample(im, u, v);
    for (int k = 0; k < 2; k++) {
      const float ct = L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k) + OB_CELL_MID;
      side_xy(L, side, ct, thin_edge_depth, &xs, &ys);
      side_xy(L, side, ct, OB_TRACK_MID, &x, &y);
      thin_point(H, cv, map[side], x, y, xs, ys, &u, &v); trk += sample(im, u, v);
    }
  }
  // The threshold the TRACK is read against is the track's own. A pair of track cells is one dark and one light
  // by construction (Manchester, or the clock), so the mean of a pair is the right threshold wherever it is
  // measured and whatever the blur. The old one, halfway between the solid line and the light ring, is right for
  // features as thick as those two; a track cell is one module deep, and under blur a thin dark cell with white
  // on both sides floats above it. With the picture at depth 6 its grey happened to hold the level down; with a
  // wider border there is white there, and the track stopped reading at 1.5 px of defocus (3% against 70%).
  *ref = 0.5f * trk / nref; *con = (light - dark) / nref;
  return *con >= 0.08f;
}
#endif

// How well the coded track reads as the layout says, in -0.5 .. 0.5. It validates a quad and picks
// its rotation in one measurement.
// beat: the caller keeps a quad only if it scores above this, and most of the up to 81 line combinations are
// nowhere near. A point adds at most 0.5, so once the points left cannot lift the sum past beat the rest are not
// read and beat itself is returned. A quad that is read to the end gets the same sum in the same order as ever.
#ifdef OB_NEON
// ai: arm64's form: four cells a vector, the clamped levels (by the contrast's reciprocal) summed a lane each and the
// ai: stop tested on the lanes' total every eight cells: the sum its own floats, the stop as sound (the same margin).
static float thin_score(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, const int *map, float ref, float con, float beat) {
  int n = 0, total = 0;
  for (int side = 0; side < 4; side++) total += ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
  const float need = (beat - 0.001f) * total;
  const float32x4_t rf = vdupq_n_f32(ref), rc = vdupq_n_f32(1 / con), hi = vdupq_n_f32(0.5f), lo = vdupq_n_f32(-0.5f);
  const trk_img_t g = trk_img(im);
  float32x4_t acc = vdupq_n_f32(0);
  for (int side = 0; side < 4; side++) {
    const int nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
    const trk_side_t st = trk_side(L, H, cv ? cv[map[side]] : NULL, map[side], side, OB_TRACK_MID);
    const uint32_t *fl = trk_signs(side);
    for (int j = 0; j < nj; j += 4) {
      const float32x4_t q = vmulq_f32(vsubq_f32(rf, trk_get_any(&st, &g, vreinterpretq_f32_s32(TRACK_AT4(L, j)))), rc);
      acc = vaddq_f32(acc, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vmaxq_f32(vminq_f32(q, hi), lo)), vld1q_u32(fl + j))));
      if (!((n += 4) & 7) && vaddvq_f32(acc) + 0.5f * (total - n) < need) return beat;
    }
  }
  return vaddvq_f32(acc) / n;
}
#else
static float thin_score(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, const int *map, float ref, float con, float beat) {
  float u, v, x, y, xs, ys, score = 0;
  int n = 0, total = 0;
  for (int side = 0; side < 4; side++) total += ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
  const float need = (beat - 0.001f) * total;   // the margin is a thousand times the sum's rounding
  for (int side = 0; side < 4; side++) for (int j = 0, nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve); j < nj; j++) {
    const int t = L->reserve + OB_TRACK_AT(j);
#ifdef OB_SIMD
    if (j + 3 < nj) {
      float Q[4];
      // The lane positions come from the macro, one per cell, never from a stride off the first (TRACK_AT4).
      const v128_t smp = thin_sample4(L, im, H, cv ? cv[map[side]] : NULL, map[side], side, TRACK_AT4(L, j), OB_TRACK_MID);
      const v128_t q4 = wasm_f32x4_div(wasm_f32x4_sub(wasm_f32x4_splat(ref), smp), wasm_f32x4_splat(con));
      wasm_v128_store(Q, wasm_f32x4_pmax(wasm_f32x4_pmin(q4, wasm_f32x4_splat(0.5f)), wasm_f32x4_splat(-0.5f)));
      for (int l = 0; l < 4; l++) {
        score += ob_thin_track(side, j + l) ? Q[l] : -Q[l];
        if (!(++n & 7) && score + 0.5f * (total - n) < need) return beat;
      }
      j += 3;
      continue;
    }
#endif
    side_xy(L, side, t + OB_CELL_MID, OB_TRACK_MID, &x, &y); side_xy(L, side, t + OB_CELL_MID, thin_edge_depth, &xs, &ys);
    thin_point(H, cv, map[side], x, y, xs, ys, &u, &v);
    float q = (ref - sample(im, u, v)) / con;
    if (q > 0.5f) q = 0.5f; else if (q < -0.5f) q = -0.5f;
    score += ob_thin_track(side, j) ? q : -q;
    if (!(++n & 7) && score + 0.5f * (total - n) < need) return beat;
  }
  return score / n;
}
#endif

// thin_levels and the track for all eight ways round a SQUARE quad from one read, taken the way up the quad is
// (layout side k on image side k) and in BOTH directions along each side. A turn or a flip of the layout square puts
// each of its sides on one of the quad's, running with that side's modules or against them. Against is not the same
// cells backwards: since OB_BAND_RUN 4 a side starts with a track run and ends with a word run, so a side turned end
// for end has its track where the word was. Read at W - (reserve + OB_TRACK_AT(i) + OB_CELL_MID), a cell is where the
// turned layout puts it.
//   s     s[(dir * 4 + side) * npts + i]: grey of track cell i of a layout side on image side `side`, dir 1 against.
//   pair  pair[dir * 4 + side]: the Manchester pairs thin_levels takes its threshold from, summed along the side and
//         read the same two ways, so a hypothesis's threshold is the one its own map would give it.
// The line and the gap are the same all along a side, so the contrast is read once and serves every hypothesis.
// Returns it, or 0 under thin_levels' floor; *nref: the steps the pairs were summed over.
#ifdef OB_NEON
// ai: arm64's form. A step's Manchester pair, track cells OB_PAIR_J(16 m) and the next of a side (m the step), are
// ai: cells the track's own read takes in any case, at the same points, so they are not read on their own: once the
// ai: contrast has passed, each side's pairs are summed from s in the steps' order. The line and the gap four steps a
// ai: vector, a last vector's lanes past the side's end read and dropped. The pairs were 256 of a quad's 896 reads at
// ai: the 128 ring.
static float thin_read_both(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, float *s, float *pair, int npts, int *nref) {
  const int W = L->w, end = W - L->reserve;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  const trk_img_t g = trk_img(im);
  const float32x4_t one = vdupq_n_f32(1);
  float dark = 0, light = 0;
  int nr = 0;
  for (int side = 0; side < 4; side++) {
    // ai: the line's point and the gap's share the bow, made once
    const trk_side_t sl = trk_side(L, H, cv ? cv[side] : NULL, side, side, dl);
    const trk_map_t gp = trk_map(L, H, side, OB_GAP_MID);
    for (int t = L->reserve; t < end; t += 64) {
      const float32x4_t at = { t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f };
      const float32x4_t e = !sl.bow ? vdupq_n_f32(0) : trk_bend(&sl, at, vdivq_f32(one, trk_w(&sl.bw, at)), sl.odd);
      float32x4_t u, v, x, y;
      trk_uv(&sl.pt, at, vdivq_f32(one, trk_w(&sl.pt, at)), &u, &v);
      trk_uv(&gp, at, vdivq_f32(one, trk_w(&gp, at)), &x, &y);
      if (sl.odd) { u = vaddq_f32(u, e); x = vaddq_f32(x, e); } else { v = vaddq_f32(v, e); y = vaddq_f32(y, e); }
      float D[4], G[4];
      vst1q_f32(D, trk_px(&g, u, v));
      vst1q_f32(G, trk_px(&g, x, y));
      for (int l = 0; l < 4 && t + 16 * l < end; l++, nr++) { dark += D[l]; light += G[l]; }
    }
  }
  *nref = nr;
  const float con = (light - dark) / nr;
  if (con < 0.08f) return 0;
  const float *pos = s + 8 * npts;
  for (int side = 0; side < 4; side++) {
    const trk_side_t st = trk_side(L, H, cv ? cv[side] : NULL, side, side, OB_TRACK_MID);
    for (int dir = 0; dir < 2; dir++) {
      // ai: against the side, the cell at W - ct: W + ct * -1, rounded once as the subtraction is
      float *out = s + (dir * 4 + side) * npts;
      const float32x4_t w0 = vdupq_n_f32(dir ? (float)W : 0), sg = vdupq_n_f32(dir ? -1.0f : 1.0f);
      if (!st.bow) trk_row(&st, &g, pos, npts, w0, sg, out, 0, 0);
      else if (st.odd) trk_row(&st, &g, pos, npts, w0, sg, out, 1, 1);
      else trk_row(&st, &g, pos, npts, w0, sg, out, 1, 0);
      float sum = 0;
      for (int t = L->reserve; t < end; t += 16) { const int j = OB_PAIR_J(t - L->reserve); sum += out[j]; sum += out[j + 1]; }
      pair[dir * 4 + side] = sum;
    }
  }
  return con;
}
#else
static float thin_read_both(const ob_layout_t *L, const image_t *im, const homo_t *H, const fline_t *const *cv, float *s, float *pair, int npts, int *nref) {
  const int W = L->w;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  float dark = 0, light = 0, u, v, x, y, xs, ys;
  int nr = 0;
  for (int side = 0; side < 4; side++) {
    float *fw = pair + side, *bw = pair + 4 + side;
    *fw = *bw = 0;
    for (int t = L->reserve; t < W - L->reserve; t += 16, nr++) {
#ifdef OB_SIMD
      if (t + 48 < W - L->reserve) {
        const v128_t along = wasm_f32x4_make(t + 0.5f, t + 16 + 0.5f, t + 32 + 0.5f, t + 48 + 0.5f), w4 = wasm_f32x4_splat((float)W);
        #define PAIR_AT(k, w) ((float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t + 16 * (k) - L->reserve) + (w))) + OB_CELL_MID)
        const v128_t ca = wasm_f32x4_make(PAIR_AT(0, 0), PAIR_AT(1, 0), PAIR_AT(2, 0), PAIR_AT(3, 0)),
                     cb = wasm_f32x4_make(PAIR_AT(0, 1), PAIR_AT(1, 1), PAIR_AT(2, 1), PAIR_AT(3, 1));
        #undef PAIR_AT
        const fline_t *c = cv ? cv[side] : NULL;
        float D[4], G[4], A[4], B[4], RA[4], RB[4];
        wasm_v128_store(D, thin_sample4(L, im, H, c, side, side, along, dl));
        wasm_v128_store(G, thin_sample4(L, im, H, c, side, side, along, OB_GAP_MID));
        wasm_v128_store(A, thin_sample4(L, im, H, c, side, side, ca, OB_TRACK_MID));
        wasm_v128_store(B, thin_sample4(L, im, H, c, side, side, cb, OB_TRACK_MID));
        wasm_v128_store(RA, thin_sample4(L, im, H, c, side, side, wasm_f32x4_sub(w4, ca), OB_TRACK_MID));
        wasm_v128_store(RB, thin_sample4(L, im, H, c, side, side, wasm_f32x4_sub(w4, cb), OB_TRACK_MID));
        for (int l = 0; l < 4; l++) { dark += D[l]; light += G[l]; *fw += A[l]; *fw += B[l]; *bw += RA[l]; *bw += RB[l]; }
        t += 48; nr += 3;
        continue;
      }
#endif
      side_xy(L, side, t + 0.5f, thin_edge_depth, &xs, &ys);
      side_xy(L, side, t + 0.5f, dl, &x, &y);
      thin_point(H, cv, side, x, y, xs, ys, &u, &v); dark += sample(im, u, v);
      side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
      thin_point(H, cv, side, x, y, xs, ys, &u, &v); light += sample(im, u, v);
      for (int k = 0; k < 2; k++) {
        const float ct = (float)(L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k)) + OB_CELL_MID;
        for (int dir = 0; dir < 2; dir++) {
          const float at = dir ? W - ct : ct;
          side_xy(L, side, at, thin_edge_depth, &xs, &ys);
          side_xy(L, side, at, OB_TRACK_MID, &x, &y);
          thin_point(H, cv, side, x, y, xs, ys, &u, &v);
          *(dir ? bw : fw) += sample(im, u, v);
        }
      }
    }
  }
  *nref = nr;
  const float con = (light - dark) / nr;
  if (con < 0.08f) return 0;
  for (int side = 0; side < 4; side++) for (int dir = 0; dir < 2; dir++) {
    float *out = s + (dir * 4 + side) * npts;
    for (int i = 0; i < npts; i++) {
#ifdef OB_SIMD
      if (i + 3 < npts) {
        const v128_t at = dir ? wasm_f32x4_sub(wasm_f32x4_splat((float)W), TRACK_AT4(L, i)) : TRACK_AT4(L, i);
        wasm_v128_store(out + i, thin_sample4(L, im, H, cv ? cv[side] : NULL, side, side, at, OB_TRACK_MID));
        i += 3;
        continue;
      }
#endif
      const float ct = (float)(L->reserve + OB_TRACK_AT(i)) + OB_CELL_MID, at = dir ? W - ct : ct;
      side_xy(L, side, at, OB_TRACK_MID, &x, &y); side_xy(L, side, at, thin_edge_depth, &xs, &ys);
      thin_point(H, cv, side, x, y, xs, ys, &u, &v);
      out[i] = sample(im, u, v);
    }
  }
  return con;
}
#endif

#ifdef OB_NEON
// ai: arm64: find_frame's hypothesis scores from thin_read_both's reading, four or eight at once (G vectors of four),
// ai: each a vector of four sums, cell i's in lane i mod 4, added up at the end; the level by the contrast's
// ai: reciprocal, the clamp as min then max, the sign as the sign bit. row[h][sd]: hypothesis h's row of s for layout
// ai: side sd; ref[h] its threshold.
static inline __attribute__((always_inline)) void trk_scores_g(const float *s, int npts, int (*row)[4], const float *ref, float con, float *sc, const int G) {
  const float32x4_t rc = vdupq_n_f32(1 / con), hi = vdupq_n_f32(0.5f), lo = vdupq_n_f32(-0.5f);
  float32x4_t acc[8], rf[8];
  for (int h = 0; h < 4 * G; h++) { acc[h] = vdupq_n_f32(0); rf[h] = vdupq_n_f32(ref[h]); }
  for (int sd = 0; sd < 4; sd++) {
    const float *p[8];
    for (int h = 0; h < 4 * G; h++) p[h] = s + row[h][sd] * npts;
    const uint32_t *fl = trk_signs(sd);
    for (int i = 0; i < npts; i += 4) {
      const uint32x4_t f = vld1q_u32(fl + i);
      for (int h = 0; h < 4 * G; h++) {
        const float32x4_t q = vmulq_f32(vsubq_f32(rf[h], vld1q_f32(p[h] + i)), rc);
        acc[h] = vaddq_f32(acc[h], vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(vmaxq_f32(vminq_f32(q, hi), lo)), f)));
      }
    }
  }
  for (int h = 0; h < 4 * G; h++) sc[h] = vaddvq_f32(acc[h]) / (float)(4 * npts);
}
// ai: hypotheses h0 to h0 + 4 G - 1 into sc[h0 ...], each threshold as the scalar loop makes it; returns h0 + 4 G
static int trk_scores(const float *s, int npts, int (*row)[4], const float *pair, int nref, float con, int h0, int G, float *sc) {
  float ref[8];
  for (int h = 0; h < 4 * G; h++) {
    float trk = 0;
    for (int sd = 0; sd < 4; sd++) trk += pair[row[h0 + h][sd]];
    ref[h] = 0.5f * trk / nref;
  }
  if (G == 2) trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 2);
  else trk_scores_g(s, npts, row + h0, ref, con, sc + h0, 1);
  return h0 + 4 * G;
}
#endif

// The format word (fmt.h), read after the frame has registered and before the picture is sampled.
// It goes through the mesh rather than the bare homography, so a bowed or tilted border reads as
// well as the picture does, and it is the last thing the decoder can learn without already knowing
// the picture size.
//
// Only the contrast comes from the two references thin_levels uses, the solid line and the light ring
// inside it, on a coarse pass. The threshold starts at the local track level and two means over the word's
// own cells settle the rest.
// The format read's sample PLAN, emitted by the reader itself so the two cannot drift: every position it would
// sample, in the order it samples them, plus the indices its arithmetic needs. Positions are module coordinates
// and depend only on the layout, so this is read once a configuration and not once a frame; what changes with
// the frame is the map they go through, which is the GPU's business (gpu/tables.mjs carries this plan to it).
//
//   pts   2 floats a point. refs first (2 a reference, dark then light), then each side's track pairs
//         (2 a pair), then each side's word cells.
//   meta  nref, cells, 1, nmid, nword, then npair[4], midBase[4], wordBase[4]. Slot 2 counted the copies of the word
//         while there were copies; it stays so the layout of meta does not move.
//   widx  3 ints a word cell: lo, hi (absolute into the mid array) and which soft value it is
// Returns the total number of points, or 0 where there is no word to read.
int ob_thin_fmt_plan(const ob_layout_t *L, int cells, float *pts, int32_t *meta, int32_t *widx) {
  float x, y;
  int nref = 0, np = 0;
  if (cells < 1) return 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, 1 + OB_THIN_LINE / 2.0f, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
  }
  const int REACH = 12 / (2 * OB_CELL), PAIRMOD = 4 * OB_CELL;
  if (!nref) return 0;
  int npair[4], midBase[4], wordBase[4], nmid = 0;
  for (int side = 0; side < 4; side++) {
    const int len = side & 1 ? L->h : L->w;
    npair[side] = ob_track_cells(len, L->reserve) / 2;
    if (npair[side] < 1) return 0;
    midBase[side] = nmid;
    for (int pr = 0; pr < npair[side]; pr++) for (int k = 0; k < 2; k++) {
      side_xy(L, side, L->reserve + OB_TRACK_AT(2 * pr + k) + OB_CELL_MID, OB_TRACK_MID, &x, &y);
      pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    }
    nmid += npair[side];
  }
  int nword = 0;
  for (int side = 0; side < 4; side++) {
    wordBase[side] = nword;
    for (int i = 0; i < cells; i++) {
      const int off = OB_WORD_AT(i);
      side_xy(L, side, L->reserve + off + OB_CELL_MID, OB_FMT_MID, &x, &y);
      pts[2 * np] = x; pts[2 * np + 1] = y; np++;
      const int at = off / PAIRMOD, lo = at - REACH < 0 ? 0 : at - REACH, hi = at + REACH >= npair[side] ? npair[side] - 1 : at + REACH;
      widx[3 * nword] = midBase[side] + lo; widx[3 * nword + 1] = midBase[side] + hi;
      widx[3 * nword + 2] = side * cells + i;
      nword++;
    }
  }
  meta[0] = nref; meta[1] = cells; meta[2] = 1; meta[3] = nmid; meta[4] = nword;
  for (int k = 0; k < 4; k++) { meta[5 + k] = npair[k]; meta[9 + k] = midBase[k]; meta[13 + k] = wordBase[k]; }
  return np;
}

// The track read's own sample plan, in the reader's order, for a reader elsewhere. Layout
// coordinates: the caller projects them through its hypothesis and samples. thin_levels takes four points a
// step (the solid line, the light ring, the Manchester pair of track cells nearest the step); thin_score one
// point a track cell with a sign from the per-side hash. Neither takes the fitted-line bow: a mark quad has no
// fitted sides, and thin_point with cv == NULL is H alone. meta: steps, track cells, w, h, cells a side x4,
// reserve. Returns the point count.
int ob_thin_score_plan(const ob_layout_t *L, float *pts, int32_t *meta, int32_t *signs) {
  float x, y;
  int np = 0, nref = 0, total = 0;
  const float dl = 1 + OB_THIN_LINE / 2.0f;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, dl, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    for (int k = 0; k < 2; k++) {
      const float ct = L->reserve + OB_TRACK_AT(OB_PAIR_J(t - L->reserve) + k) + OB_CELL_MID;
      side_xy(L, side, ct, OB_TRACK_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y; np++;
    }
  }
  for (int side = 0; side < 4; side++) {
    const int nj = ob_track_cells(side & 1 ? L->h : L->w, L->reserve);
    meta[4 + side] = nj;
    for (int j = 0; j < nj; j++) {
      const int t = L->reserve + OB_TRACK_AT(j);
      side_xy(L, side, t + OB_CELL_MID, OB_TRACK_MID, &x, &y); pts[2 * np] = x; pts[2 * np + 1] = y;
      signs[total] = ob_thin_track(side, j) ? 1 : -1;
      np++; total++;
    }
  }
  meta[0] = nref; meta[1] = total; meta[2] = L->w; meta[3] = L->h; meta[8] = L->reserve;
  return np;
}

// One (mark quad, hypothesis) through the track read, exactly as find_frame does it, to hold a device's score
// to. centres: the four mark centres as mark_quads orders them; hyp: rotation and mirror as
// find_frame numbers them; merged: which form's mark middle the map is solved from. out: levels ok, ref, con,
// the score read to the end (no early-out), then H (9) and the registration quad corner_coords projects to (8).
int ob_test_thin_score(const ob_layout_t *L, const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out) {
  image_t im;
  ob_image_init(&im, img, w, h, 1.0f);
  const float m = mark_form_mid(L, merged), msrc[8] = { m, m, L->w - m, m, L->w - m, L->h - m, m, L->h - m };
  const int rot = hyp & 3, mir = hyp >> 2, map[4] = { 0, 1, 2, 3 };
  float d2[8], ref = 0, con = 0, nc[8];
  homo_t H;
  for (int k = 0; k < 21; k++) out[k] = 0;
  out[0] = -1;
  for (int k = 0; k < 4; k++) { const int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = centres[2 * from]; d2[2 * k + 1] = centres[2 * from + 1]; }
  if (!homography(msrc, d2, &H)) return 0;
  for (int k = 0; k < 9; k++) out[4 + k] = H.h[k];
  corner_coords(L, nc);
  for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &out[13 + 2 * k], &out[14 + 2 * k]);
  const int lv = thin_levels(L, &im, &H, NULL, map, &ref, &con);
  out[0] = (float)lv; out[1] = ref; out[2] = con;
  if (!lv) return 1;
  out[3] = thin_score(L, &im, &H, NULL, map, ref, con, -1e9f);
  return 1;
}

int ob_thin_read_fmt(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float *q, int cells) {
  float dark = 0, light = 0, x, y, u, v;
  int nref = 0;
  if (cells < 1) return 0;
  for (int side = 0; side < 4; side++) for (int t = L->reserve; t < (side & 1 ? L->h : L->w) - L->reserve; t += 16, nref++) {
    side_xy(L, side, t + 0.5f, 1 + OB_THIN_LINE / 2.0f, &x, &y);
    ob_map_point(L, reg, x, y, &u, &v); dark += sample(im, u, v);
    side_xy(L, side, t + 0.5f, OB_GAP_MID, &x, &y);
    ob_map_point(L, reg, x, y, &u, &v); light += sample(im, u, v);
  }
  if (!nref) return 0;
  const float con = (light - dark) / nref;
  if (con < 0.08f) return 0;   // the same floor thin_levels rejects a quad on
  // Each cell is read against the track beside it, not against one level for the whole band. The band's cells are
  // the track's size, and a Manchester pair of track cells averages to the right threshold at any blur (the reason
  // thin_levels gives), but only for the light it was read under: traced on a phone capture with glare over one
  // corner, the word's levels there sat above the global threshold's dark AND light, and RS lost the word on a
  // frame that had registered to within a pixel. Three pairs either side is 28 modules, far inside any glare.
  // Reach in MODULES, not in pairs: the window has to cover the same stretch of border whatever the cell size,
  // or a smaller cell averages over less light and the threshold gets noisier exactly where the cell is weakest.
  const int REACH = 12 / (2 * OB_CELL);
  // A Manchester pair is two TRACK cells, which are cells 2p and 2p + 1 of a run, so it spans four band cells.
  const int PAIRMOD = 4 * OB_CELL;
  float *mid = malloc(((size_t)(L->w > L->h ? L->w : L->h) / PAIRMOD + 2) * sizeof(float));
  if (!mid) return 0;
  for (int side = 0; side < 4; side++) {
    const int len = side & 1 ? L->h : L->w, npair = ob_track_cells(len, L->reserve) / 2;
    if (npair < 1) { free(mid); return 0; }
    for (int pr = 0; pr < npair; pr++) {
      float s = 0;
      for (int k = 0; k < 2; k++) { side_xy(L, side, L->reserve + OB_TRACK_AT(2 * pr + k) + OB_CELL_MID, OB_TRACK_MID, &x, &y); ob_map_point(L, reg, x, y, &u, &v); s += sample(im, u, v); }
      mid[pr] = 0.5f * s;
    }
    for (int i = 0; i < cells; i++) {
      const int off = OB_WORD_AT(i);
      side_xy(L, side, L->reserve + off + OB_CELL_MID, OB_FMT_MID, &x, &y);
      ob_map_point(L, reg, x, y, &u, &v);
      const int at = off / PAIRMOD, lo = at - REACH < 0 ? 0 : at - REACH, hi = at + REACH >= npair ? npair - 1 : at + REACH;
      float m = 0;
      for (int k = lo; k <= hi; k++) m += mid[k];
      q[side * cells + i] = sample(im, u, v) - m / (hi - lo + 1);
    }
  }
  free(mid);
  // What is left is the cell against its local track level, and the word's own two populations settle the rest.
  // Under blur its cells sit a little lighter than track cells do and the track's midpoint calls the faintest
  // dark ones light. A codeword is near enough half dark for two means to find its own levels; a blank band has
  // one population, keeps the track's level and reads as the nothing it is.
  float ref = 0;
  for (int it = 0; it < 4; it++) {
    float lo = 0, hi = 0;
    int nlo = 0;
    for (int i = 0; i < 4 * cells; i++) if (q[i] < ref) { lo += q[i]; nlo++; } else hi += q[i];
    if (nlo < cells / 2 || 4 * cells - nlo < cells / 2) break;
    ref = 0.5f * (lo / nlo + hi / (4 * cells - nlo));
  }
  for (int i = 0; i < 4 * cells; i++) { const float s = (ref - q[i]) / con; q[i] = s > 0.5f ? 0.5f : s < -0.5f ? -0.5f : s; }
  return 1;
}

// Finds the frame and its rotation. quad: the four lattice corners in layout order, as image points.
// The image halved, each pixel the mean of four.
// ai: fb, slot: frame_buf's (OB_FB_HALF or OB_FB_QUARTER)
static uint8_t *shrink2(const uint8_t *src, int w, int h, ob_fbufs_t *fb, int slot) {
  const int sw = w / 2, sh = h / 2;
  uint8_t *out = frame_buf(fb, slot, (size_t)sw * sh);
  (void)slot;
  if (!out) return NULL;
  for (int y = 0; y < sh; y++) {
    const uint8_t *a = src + (size_t)2 * y * w, *b = a + w;
    uint8_t *o = out + (size_t)y * sw;
    int x = 0;
#ifdef OB_NEON
    // ai: arm64: sixteen a step, each 2 x 2's sum by pairwise adds (UADDLP, UADALP) and (s + 2) >> 2 by a rounding
    // ai: narrowing shift (RSHRN): the same integers
    for (; x + 16 <= sw; x += 16) {
      const uint16x8_t s0 = vpadalq_u8(vpaddlq_u8(vld1q_u8(a + 2 * x)), vld1q_u8(b + 2 * x));
      const uint16x8_t s1 = vpadalq_u8(vpaddlq_u8(vld1q_u8(a + 2 * x + 16)), vld1q_u8(b + 2 * x + 16));
      vst1q_u8(o + x, vcombine_u8(vrshrn_n_u16(s0, 2), vrshrn_n_u16(s1, 2)));
    }
#endif
    for (; x < sw; x++) o[x] = (uint8_t)((a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) >> 2);
  }
  return out;
}

// A screen is seen mirrored only through a mirror, so the mirror images (hypotheses 4 to 7) are a last resort, not
// tried once a rotation holds the best. A mirror image holding it does not stop them: that is a capture through a
// mirror, and its other quads are read the same way. 0.15 is 3.4 sd of a texture read (a sign test, sd 0.5 /
// sqrt(cells)) over the 64 ring's 128 track cells, the fewest any ring has: the same significance or more on every
// ring (2026-10-04: the 32 ring's 64 cells held at 0.212 until it left the format on 2026-10-25).
static inline int rotation_holds(float best, int orient) { return best >= 0.15f && orient < 4; }

// Ls: the layouts the symbol may be, which differ only in how many modules a side (lizard_phy.h LIZARD_RING); *which
// comes back as the one that registered. The mark path scores every one before a hold settles it; the line path tries
// the first (the 64 ring) first and the others only when it does not register.
// ai: cb: the capture's column bits on arm64, made with bin (binarize); NULL elsewhere. fb: frame_buf's
static int find_frame(const ob_layout_t *const *Ls, int nL, const image_t *im, const uint8_t *bin, uint32_t *cb, ob_fbufs_t *fb, float *quad, int *orient, float *score_out, int *which) {
  int w = im->w, h = im->h, big = w > h ? w : h, n[4], nl[4];
  fpt_t *pts[4] = { 0 };            // allocated only if the scan runs, which a settled mark quad skips
  fline_t line[4][SIDE_LINES];
  float best = -1, src[8], dst[8];
  int ok = 1;
  double tp = ob_now_ms();
  // The corner a layout side runs between, in the direction its modules count. Shared with the combination loop.
  static const int ENDS[4][2] = { { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 } };
  // The corner marks' quads, scored on the same scale as the line quads and tried FIRST. Measured with the line
  // fit switched off altogether: the mark quad alone equals the full pipeline in every cell and beats it at 480 px
  // (3957 B against 3635). So once one has settled there is nothing for the 81 line combinations to add, only a
  // chance for a wrong line quad (a bar, a merged run's ragged inner edge) to outscore a right one, and most of
  // the finder's time to spend doing it.
  //
  // The gapped form, then the merged one only if that settled nothing, and within a form every quad, the TRACK
  // choosing between them. Spacing cannot choose: a noisy background holds four grains at any spacing asked for, and
  // only the symbol has a track between them. The best reading of the set wins, not the first to reach 0.15: taking
  // the first lost 2 of 32 clean LIZARD-16 simulator frames (scripts/exp/fmt_word.mjs DAMAGE) to a mirror image of a wrong
  // quad at 0.16, ahead of the true quad at 0.49 the right way round. A set is MARK_QUADS quads, so this is at most
  // 32 hypotheses, what a set that settled nothing always cost, and thin_score stops reading one that cannot win.
  //
  // Then the same again on the image halved and quartered. binarize calls a pixel dark against the mean of its
  // 40 px neighbourhood, so a mark whose core outgrows that window is light in the middle and fails its solidity
  // probes: a 94-module symbol at 1080 fill 0.7 is 8 px a module and its cores are 48 px, and at 30 degrees the
  // near marks were not found at all. A full-scale scan alone had sufficed for the 286-module side, where a core
  // only reaches the window at very close range.
  float msrc[8];
  cand_t *mcand = malloc((size_t)MAX_CAND * sizeof(cand_t) + CAND_XY);
  mquad_t mq[MARK_QUADS];
  uint8_t *grey2 = NULL, *lbin = NULL;
  const uint8_t *grey = im->px;
  const uint32_t *scb = cb;   // the column bits of the scale's image (arm64)
  *which = 0;
  for (int k = 0; k < 8; k++) ob_finder_info[k] = k == 5 || k == 7 ? 0 : -1;
  for (int scale = 1; mcand && scale <= 4 && !rotation_holds(best, *orient); scale *= 2) {
    const uint8_t *sbin = bin;
    const int sw = w / scale, sh = h / scale;
    if (scale > 1) {
      if (sw < 64 || sh < 64) break;
      uint8_t *next = shrink2(grey, w / (scale / 2), h / (scale / 2), fb, scale == 2 ? OB_FB_HALF : OB_FB_QUARTER);
      if (!next) break;
      frame_buf_done(fb, grey2); grey2 = next; grey = next;
      frame_buf_done(fb, lbin);
      if (!(lbin = frame_buf(fb, OB_FB_LBIN, (size_t)sw * sh))) break;
#ifdef OB_NEON
      uint32_t *lcb = frame_buf(fb, OB_FB_LCB, (size_t)((sh + 31) / 32) * sw * sizeof(uint32_t) + 16);
      if (!lcb) break;
      scb = lcb;
      PROF(PROF_BINARIZE, binarize(grey, sw, sh, lbin, lcb, fb));
#else
      PROF(PROF_BINARIZE, binarize(grey, sw, sh, lbin, NULL, fb));
#endif
      sbin = lbin;
    }
    // A third form reads the merged scan's squares again, as the CORE of a sharp mark. The merged scan finds any
    // solid dark square bounded by light, and on a sharp capture the one it finds at a mark is its 6-module core,
    // which the gap still separates from the line: read as a closed 10-module mark, its module size comes out
    // 0.6 of the truth and its middle two modules off, and mark_quads refuses the real quad on spacing. The gapped
    // form cannot stand in: its row cross-section, line then gap then core, and its solidity probes assume the
    // mark nearly square to the rows, and past about 25 degrees of in-plane turn it stops finding marks at all
    // (traced on LIZARD-16: 8 to 12 rows a mark at 20 degrees, 1 or 2 at 30, a mark missing outright at 30 and
    // 35), where the merged scan still finds all four, 15 to 17 rows each, within half a pixel. The square test is
    // a run and a column and four probes near the middle, none of which cares how the square is turned. No new
    // scan: the candidates are the merged form's, their module size rescaled.
    int nmark = 0;
    for (int form = 0; form < 3 && !rotation_holds(best, *orient); form++) {
    if (form < 2) PROF(PROF_FF_MARKS, nmark = mark_scan(Ls[0], sbin, scb, sw, sh, scale, form, mcand));
    else {
      const int core = mark_core(Ls[0]), sq = Ls[0]->corner - 2;
      for (int k = 0; k < nmark; k++) mcand[k].unit *= (float)sq / core;
    }
    mpool_t pool;
    mark_pool(mcand, nmark, &pool);
    for (int li = 0; li < nL; li++) {
    const ob_layout_t *L = Ls[li];
    const int nq = mark_rect(L, &pool, nmark, w, h, form, msrc, mq);
    if (form == 0 && scale == 1 && li == 0) ob_finder_info[7] = (float)nq;
    for (int h0 = 0; h0 < 8 && !(h0 == 4 && rotation_holds(best, *orient)); h0 += 4)
    for (int qi = 0; qi < nq; qi++) for (int hyp = h0; hyp < h0 + 4; hyp++) {
      int rot = hyp & 3, mir = hyp >> 2, map[4];
      float d2[8], ref = 0, con = 0, nc[8];
      homo_t H;
      for (int k = 0; k < 4; k++) { int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = mq[qi].q[from].x; d2[2 * k + 1] = mq[qi].q[from].y; }
      if (!homography(msrc, d2, &H)) continue;
      for (int sd = 0; sd < 4; sd++) { int ia = (rot + (mir ? 4 - ENDS[sd][0] : ENDS[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS[sd][1] : ENDS[sd][1])) & 3; map[sd] = ((ia + 1) & 3) == ib ? ia : ib; }
      int lv;
      PROF(PROF_FF_TRACK, lv = thin_levels(L, im, &H, NULL, map, &ref, &con));
      if (!lv) continue;
      float sc;
      PROF(PROF_FF_TRACK, sc = thin_score(L, im, &H, NULL, map, ref, con, best));
      if (sc <= best) continue;
      best = sc; *orient = hyp; *which = li;
      ob_finder_info[0] = best; ob_finder_info[1] = (float)hyp; ob_finder_info[2] = (float)form; ob_finder_info[3] = (float)qi; ob_finder_info[4] = 0;
      corner_coords(L, nc);
      for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &quad[2 * k], &quad[2 * k + 1]);
    }
    }
    }
  }
  free(mcand); frame_buf_done(fb, grey2); frame_buf_done(fb, lbin);
  ob_finder_info[5] = rotation_holds(best, *orient); ob_finder_info[6] = best;
  if (rotation_holds(best, *orient)) ok = 0;
  ob_prof_ms[PROF_FF_SCORE] += ob_now_ms() - tp; tp = ob_now_ms();
  // THE SCAN AND THE LINE FIT, only if the marks settled nothing. A settled mark quad already cleared `ok`,
  // which skipped the 81 combinations below; what it could not skip was the scan and the fit that feed them,
  // and those were being paid for on every capture. Hoisting the marks above them changes no output (nothing
  // between the two writes thin_edge_depth, and mark_rect reads `bin` and its own candidate array, never
  // `pts`), so this is the same finder with the wasted half of it skipped when it is wasted.
  if (ok) {
  // Sides in layout order top, right, bottom, left. Rows give left (near) and right (far); the transposed image gives top and bottom.
  // Contrast floor: a quarter of the picture's own range (5th to 95th percentile of a coarse grid).
  int hist[256] = { 0 }, cnt = 0, p5 = 0, p95 = 255, acc = 0;
  for (int y = 4; y < h; y += 8) for (int x = 4; x < w; x += 8) { hist[im->px[y * w + x]]++; cnt++; }
  for (int v = 0; v < 256; v++) { acc += hist[v]; if (acc * 20 < cnt) p5 = v; if (acc * 20 < cnt * 19) p95 = v; }
  int min_con = (p95 - p5) / 4 < 12 ? 12 : (p95 - p5) / 4;
  for (int s = 0; s < 4; s++) pts[s] = malloc((size_t)(big / 2 + 1) * SIDE_K * sizeof(fpt_t));
  edge_ctx_t ctx;
  if (edge_ctx_init(&ctx, im->px, w, h, min_con, fb)) { for (int s = 0; s < 4; s++) free(pts[s]); return 0; }
  row_points(bin, w, h, &ctx, pts[3], &n[3], pts[1], &n[1]);
  ob_prof_ms[PROF_FF_POINTS] += ob_now_ms() - tp; tp = ob_now_ms();
  col_points(bin, w, h, &ctx, pts[0], &n[0], pts[2], &n[2]);
  ob_prof_ms[PROF_FF_TURN] += ob_now_ms() - tp; tp = ob_now_ms();
  frame_buf_done(fb, ctx.buf);
#ifdef OB_VERIFY_POINTS
  {
    int *pos = malloc(((size_t)big + 2) * sizeof(int)), m[4];
    uint8_t *tr = malloc((size_t)w * h);
    fpt_t *ref[4];
    for (int s = 0; s < 4; s++) ref[s] = malloc((size_t)(big / 2 + 1) * SIDE_K * sizeof(fpt_t));
    side_points(bin, w, h, im->px, 1, w, min_con, pos, ref[3], &m[3], ref[1], &m[1]);
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) tr[x * h + y] = bin[y * w + x];
    side_points(tr, h, w, im->px, w, 1, min_con, pos, ref[0], &m[0], ref[2], &m[2]);
    for (int s = 0; s < 4; s++) {
      int bad = m[s] != n[s];
      for (int k = 0; !bad && k < n[s]; k++) bad = ref[s][k].along != pts[s][k].along || ref[s][k].across != pts[s][k].across || ref[s][k].rank != pts[s][k].rank;
      if (bad) { printf("OB_VERIFY_POINTS: side %d differs (%d candidates against %d), image %d x %d\n", s, n[s], m[s], w, h); abort(); }
      free(ref[s]);
    }
    free(pos); free(tr);
  }
#endif
  // Strict: the border's points scatter by half a pixel, and with a loose tolerance a line that only crosses
  // the border gathers nearly as many as the border does. A lens's bow is the curve fit's business, which
  // grows outwards from whatever stretch the straight line holds.
  float tol = 1.5f;
  for (int s = 0; s < 4; s++) { nl[s] = fit_lines(pts[s], n[s], tol, 0.12f * (s & 1 ? h : w), line[s]); ok &= nl[s] > 0; }
  ob_prof_ms[PROF_FF_FIT] += ob_now_ms() - tp; tp = ob_now_ms();
  }
  int combos = SIDE_LINES * SIDE_LINES * SIDE_LINES * SIDE_LINES;
  // The lines are the image's and serve every layout; the track that judges them is each layout's own.
  for (int li = 0; ok && li < nL && !rotation_holds(best, *orient); li++) {
  const ob_layout_t *L = Ls[li];
  // One reading serves every way round on a square, where a turn or a flip lands each side on another's.
  const int npts = ob_track_cells(L->w, L->reserve);
#ifdef OB_NEON
  // ai: arm64: past the reading's eight rows, the track cells' modules along a side (TRACK_AT4's) for thin_read_both;
  // ai: the signs are trk_signs'; each hypothesis's rows of track_s, 4 rev + map a layout side, as the loop finds them
  float *track_s = malloc((size_t)9 * npts * sizeof(float));
  if (!track_s) break;
  for (int i = 0; i < npts; i++) track_s[8 * npts + i] = (float)(L->reserve + OB_TRACK_AT(i)) + OB_CELL_MID;
  int hrow[8][4];
  for (int hyp = 0; hyp < 8; hyp++) for (int sd = 0; sd < 4; sd++) {
    const int rot = hyp & 3, mir = hyp >> 2;
    const int ia = (rot + (mir ? 4 - ENDS[sd][0] : ENDS[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS[sd][1] : ENDS[sd][1])) & 3;
    const int m = ((ia + 1) & 3) == ib ? ia : ib;
    hrow[hyp][sd] = 4 * (ia != ENDS[m][0]) + m;
  }
#else
  float *track_s = malloc((size_t)8 * npts * sizeof(float));
  int8_t *track_sign = malloc((size_t)4 * npts);
  if (!track_s || !track_sign) { free(track_s); free(track_sign); break; }
  for (int sd = 0; sd < 4; sd++) for (int i = 0; i < npts; i++) track_sign[sd * npts + i] = (int8_t)ob_thin_track(sd, i);
#endif
  // Two passes, one per depth the curves may run at (thin_edge_depth). Blur is the same all round the symbol, so
  // the four sides share one state and this is two hypotheses, not sixteen. The second runs only when the first
  // settled nothing, so a capture sharp enough for the gap to survive pays for one.
  //
  // Every combination is scored under every rotation, and the best of them all wins. Until 2026-09-24 the first
  // combination to reach 0.15 fixed the rotation and the rest were scored under it alone, so a wrong combination
  // read passably one way round shut the true one out. One read serves all eight ways round, so this costs reading
  // each side both ways, not eight reads: on the line path a decode went from 2.7 to 2.9 ms at n = 256 and from
  // 15.5 to 15.8 at 1024 (an ideal image, mark cores painted out).
  for (int pass = 0; pass < 2 && !(pass && rotation_holds(best, *orient)); pass++) {
  thin_edge_depth = pass ? OB_THIN : 1 + OB_THIN_LINE;   // OB_THIN: the picture's edge, which is what line, track and ring merge to
  { const float e = thin_edge_depth, se[8] = { e, e, L->w - e, e, L->w - e, L->h - e, e, L->h - e }; memcpy(src, se, sizeof src); }
  for (int combo = 0; ok && combo < combos; combo++) {
    int pick[4] = { combo % SIDE_LINES, (combo / SIDE_LINES) % SIDE_LINES, (combo / (SIDE_LINES * SIDE_LINES)) % SIDE_LINES, combo / (SIDE_LINES * SIDE_LINES * SIDE_LINES) }, skip = 0;
    for (int s = 0; s < 4; s++) skip |= pick[s] >= nl[s];
    if (skip) continue;
    const fline_t *T = &line[0][pick[0]], *Rr = &line[1][pick[1]], *B = &line[2][pick[2]], *Lf = &line[3][pick[3]];
    float q[8];
    meet(Lf, T, &q[0], &q[1]); meet(Rr, T, &q[2], &q[3]); meet(Rr, B, &q[4], &q[5]); meet(Lf, B, &q[6], &q[7]);
    if (q[2] - q[0] < 0.1f * w || q[7] - q[1] < 0.1f * h) continue;
    float wide = 0.5f * ((q[2] - q[0]) + (q[4] - q[6])), tall = 0.5f * ((q[7] - q[1]) + (q[5] - q[3]));
    if (wide > 2 * tall || tall > 2 * wide) continue;
    memcpy(dst, q, sizeof q);
    const fline_t *cv[4] = { T, Rr, B, Lf };
    float pair[8], con = 0;
    int nref = 0;
    homo_t H0;
    // The track is read once, the way up the quad is and both ways along each side (thin_read_both), and each
    // hypothesis takes its sides from that reading with its own threshold and signs: the score its own map gives,
    // without eight reads. The one that wins gets its own homography for the corners it hands back.
    if (!homography(src, dst, &H0) || !(con = thin_read_both(L, im, &H0, cv, track_s, pair, npts, &nref))) continue;
#ifdef OB_NEON
    float hsc[8];
    int scored = 0;
#endif
    for (int hyp = 0; hyp < 8 && !(hyp == 4 && rotation_holds(best, *orient)); hyp++) {
#ifdef OB_NEON
      const int rot = hyp & 3, mir = hyp >> 2;
      float d2[8];
      homo_t H;
      // ai: all eight at once where the hold cannot stop the loop at 4 (it held before this quad: it holds there)
      if (hyp >= scored) scored = trk_scores(track_s, npts, hrow, pair, nref, con, hyp, hyp == 0 && !rotation_holds(best, *orient) ? 2 : 1, hsc);
      const float sc = hsc[hyp];
      if (sc <= best) continue;
#else
      int rot = hyp & 3, mir = hyp >> 2, map[4];
      float d2[8], sc = 0;
      homo_t H;
      // Which image side each layout side lies on under this hypothesis: the one joining the two image corners its
      // ends map to; rev where it runs against that side's own direction.
      int rev[4];
      for (int sd = 0; sd < 4; sd++) {
        const int ia = (rot + (mir ? 4 - ENDS[sd][0] : ENDS[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS[sd][1] : ENDS[sd][1])) & 3;
        map[sd] = ((ia + 1) & 3) == ib ? ia : ib;
        rev[sd] = ia != ENDS[map[sd]][0];
      }
      {
        float trk = 0;
        for (int sd = 0; sd < 4; sd++) trk += pair[4 * rev[sd] + map[sd]];
        const float ref = 0.5f * trk / nref;
        sc = 0;
        for (int sd = 0; sd < 4; sd++) {
          const float *sm = track_s + (4 * rev[sd] + map[sd]) * npts;
          const int8_t *sg = track_sign + sd * npts;
          for (int i = 0; i < npts; i++) {
            float qq = (ref - sm[i]) / con;
            if (qq > 0.5f) qq = 0.5f; else if (qq < -0.5f) qq = -0.5f;
            sc += sg[i] ? qq : -qq;
          }
        }
        sc /= 4 * npts;
        if (sc <= best) continue;
      }
#endif
      for (int k = 0; k < 4; k++) { int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = dst[2 * from]; d2[2 * k + 1] = dst[2 * from + 1]; }
      if (!homography(src, d2, &H)) continue;
      best = sc; *orient = hyp; *which = li;
      ob_finder_info[0] = best; ob_finder_info[1] = (float)hyp; ob_finder_info[4] = 1;
      // Hand back the lattice's own corners, which sit half a module further in than the line's middle.
      float nc[8];
      corner_coords(L, nc);
      for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &quad[2 * k], &quad[2 * k + 1]);
    }
  }
  }
#ifdef OB_NEON
  free(track_s);
#else
  free(track_s); free(track_sign);
#endif
  }
  thin_edge_depth = 1 + OB_THIN_LINE;
  ob_prof_ms[PROF_FF_SCORE] += ob_now_ms() - tp;
  for (int s = 0; s < 4; s++) free(pts[s]);
  *score_out = best;
  return best >= 0.06f;
}

// ---------------------------------------------------------------- top level

// The image the node templates read when the box has a radius (a module of 2 px or more): a copy with the thin frame's border
// band under a (2r + 1)-pixel box (box_blur_rect). The band is where every node a thin frame refines lives, the
// line of nodes at depth 1.5 modules and the corners' templates, which reach 12 modules in; taken 16 modules deep
// and 3 outside, the coarse search's module and a half included, through H and a few pixels wider. Blurring the
// whole frame instead cost 1.5 to 1.9 ms of a 3 to 4 ms acquisition at 1080; outside the band nothing reads it,
// and what is there is the image itself.
// ai: spare: a buffer of iw x ih bytes to write the copy into (arm64 hands it the binarized picture's, whose pages are
// ai: already the process's: a fresh one's first touches were most of the copy's time on an S26), or NULL for a new one
static uint8_t *border_blur(const ob_layout_t *L, const homo_t *H, const uint8_t *img, int iw, int ih, int r, uint8_t *spare) {
  uint8_t *out = spare ? spare : malloc((size_t)iw * ih);
  if (!out) return NULL;
  memcpy(out, img, (size_t)iw * ih);
  const float D = 16, Mo = 3, W = (float)L->w, Hh = (float)L->h;
  const float band[4][4] = { { -Mo, -Mo, W + Mo, D }, { -Mo, Hh - D, W + Mo, Hh + Mo }, { -Mo, -Mo, D, Hh + Mo }, { W - D, -Mo, W + Mo, Hh + Mo } };
#ifdef OB_NEON
  // ai: msh_blur_rect's ring, column sums and row copy, for rectangles up to the image's width
  uint16_t *ring = r <= 7 ? malloc(((size_t)(2 * r + 4) * iw) * sizeof(uint16_t) + (size_t)iw + 2 * r) : NULL;
#endif
  for (int k = 0; k < 4; k++) {
    float lo[2] = { 1e9f, 1e9f }, hi[2] = { -1e9f, -1e9f };
    for (int c = 0; c < 4; c++) {
      float u, v;
      project(H, band[k][c & 1 ? 2 : 0], band[k][c & 2 ? 3 : 1], &u, &v);
      lo[0] = fminf(lo[0], u); hi[0] = fmaxf(hi[0], u); lo[1] = fminf(lo[1], v); hi[1] = fmaxf(hi[1], v);
    }
    const int pad = r + 4;
    int x0 = (int)floorf(lo[0]) - pad, y0 = (int)floorf(lo[1]) - pad, x1 = (int)ceilf(hi[0]) + pad, y1 = (int)ceilf(hi[1]) + pad;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > iw) x1 = iw;
    if (y1 > ih) y1 = ih;
    if (x1 <= x0 || y1 <= y0) continue;
#ifdef OB_NEON
    if (ring && x1 - x0 >= 16) {
      msh_blur_rect(img, out, iw, ih, r, x0, y0, x1, y1, ring, ring + (size_t)(2 * r + 2) * iw, (uint8_t *)(ring + (size_t)(2 * r + 4) * iw));
      continue;
    }
#endif
    uint16_t *tmp = malloc((size_t)(x1 - x0) * (y1 - y0 + 2 * r) * sizeof(uint16_t));
    int *acc = malloc((size_t)(x1 - x0) * sizeof(int));
    if (tmp && acc) box_blur_rect(img, out, iw, ih, r, x0, y0, x1, y1, tmp, acc);
    free(tmp); free(acc);
  }
#ifdef OB_NEON
  free(ring);
#endif
  return out;
}

void ob_reg_free(ob_reg_t *reg) { free(reg->nodes); reg->nodes = NULL; }

// in_quad: a quad found somewhere else, in image pixels, with its orientation and track score. The scan, the
// line fit and the track scoring are then all skipped and everything from the homography onwards is shared, so
// a quad found on a GPU becomes a registration on exactly the terms one found here does.
// Ls: the layouts the symbol may be (find_frame); the one that registered comes back in *which and is the one the
// registration is built on. An external quad is the first layout's.
static int acquire_core(const ob_layout_t *const *Ls, int nL, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res,
                        const float *in_quad, int in_orient, float in_score, int *which, ob_fbufs_t *fb) {
  const ob_layout_t *L = Ls[0];
  *which = 0;
  const image_t im = *pim;
  const uint8_t *img = im.px;
  int iw = im.w, ih = im.h;
  reg->nodes = NULL;
  uint8_t *bin = in_quad ? NULL : frame_buf(fb, OB_FB_BIN, (size_t)iw * ih);
#ifdef OB_NEON
  uint32_t *cb = in_quad ? NULL : frame_buf(fb, OB_FB_CB, (size_t)((ih + 31) / 32) * iw * sizeof(uint32_t) + 16);
  if (!in_quad && !cb) { frame_buf_done(fb, bin); return 0; }
#else
  uint32_t *cb = NULL;
#endif
  int got = 0, thin_orient = -1;
  float thin_quad[8], thin_sc = 0;
  if (in_quad) { memcpy(thin_quad, in_quad, sizeof thin_quad); thin_orient = in_orient; thin_sc = in_score; got = 1; }
  else {
    PROF(PROF_BINARIZE, binarize(img, iw, ih, bin, cb, fb));
    PROF(PROF_FINDERS, got = find_frame(Ls, nL, &im, bin, cb, fb, thin_quad, &thin_orient, &thin_sc, which));
    L = Ls[*which];
  }
#ifdef OB_NEON
  frame_buf_done(fb, cb);
#else
  frame_buf_done(fb, bin);
  bin = NULL;
#endif
  if (!got) { frame_buf_done(fb, bin); return 0; }

  double tm = ob_now_ms();
  float src[8];
  corner_coords(L, src);
  homo_t H;
  // The thin frame's finder has already settled rotation: it scores the coded track under every hypothesis.
  if (!homography(src, thin_quad, &H)) { frame_buf_done(fb, bin); return 0; }
  memcpy(res->quad, thin_quad, sizeof thin_quad);
  if (thin_orient < 0 || 2.0f + thin_sc < 2.0f) { frame_buf_done(fb, bin); return 0; }
  res->orient = thin_orient;

  // The node templates read the image under a box a module wide once a module is several pixels (border_blur). The
  // module's size is the quad's sides over the lattice's, in modules.
  image_t rim = im;
  uint8_t *blurred = NULL;
  float mpx = 0;
  for (int k = 0; k < 4; k++) {
    const int k1 = (k + 1) & 3;
    mpx += hypotf(res->quad[2 * k1] - res->quad[2 * k], res->quad[2 * k1 + 1] - res->quad[2 * k + 1]) / hypotf(src[2 * k1] - src[2 * k], src[2 * k1 + 1] - src[2 * k + 1]) / 4;
  }
  // ai: a box radius of at least one pixel, tested before the conversion (a NaN mpx blurs nothing, as before)
  // ai: on arm64 the copy goes into the binarized picture's buffer (bin, the workspace's), which the finder is done with
  if (mpx / 2 >= 1 && (blurred = border_blur(L, &H, img, iw, ih, (int)(mpx / 2), bin))) rim.px = blurred;
  // Finders first, so the homography the marks are measured against is as good as four
  // points can make it.
  int nn = L->nx * L->ny;
  node_t *nodes = calloc((size_t)nn, sizeof(node_t));
  for (int c = 0; c < 4; c++) {
    int i = (c == 1 || c == 2) ? L->nx - 1 : 0, j = c >= 2 ? L->ny - 1 : 0;
    refine_node(L, &rim, &H, i, j, nodes, 0);
    node_t *nd = &nodes[j * L->nx + i];
    res->quad[2 * c] += nd->dx; res->quad[2 * c + 1] += nd->dy;
    nd->done = 0;
  }
  homography(src, res->quad, &H);
  for (int c = 0; c < 4; c++) { node_t *nd = &nodes[((c >= 2) ? L->ny - 1 : 0) * L->nx + ((c == 1 || c == 2) ? L->nx - 1 : 0)]; nd->dx = nd->dy = 0; nd->done = 1; }
  // ai: the anchors: the code registered from the anchors outward (anchor_register: a corner they disagree with
  // ai: searched again from their map), and the lens model they and the corners say, which every border node's search
  // ai: below starts from
  anchor_t anchors[OB_ANCH_K * OB_ANCH_K];
  lensm_t alm;
  int have_alm = 0;
  ob_anch_found = -1;
  if (mesh) {
    have_alm = anchor_register(L, &im, &rim, &H, iw, ih, mpx, res->quad, nodes, anchors, &alm);
    ob_anch_found = 0;
    for (int k = 0; k < OB_ANCH_K * OB_ANCH_K; k++) ob_anch_found += anchors[k].ok;
  }
  // Rings outward from the corners, so each mark starts from measured neighbours.
  // ai: the marked nodes (the border's) in that order, ring by ring and in rows within one, by a counting sort on
  // ai: the ring: a pass over the lattice, not one a ring
  if (mesh) {
    const int nx = L->nx, ny = L->ny;
    int start[OB_MAX_NODES + 2] = { 0 }, order[4 * OB_MAX_NODES], no = 0;
#define RING(i, j) (((i) < nx - 1 - (i) ? (i) : nx - 1 - (i)) + ((j) < ny - 1 - (j) ? (j) : ny - 1 - (j)))
    for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) if (L->node_mark[j * nx + i]) start[RING(i, j) + 1]++;
    for (int k = 1; k < OB_MAX_NODES + 2; k++) start[k] += start[k - 1];
    for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) if (L->node_mark[j * nx + i]) { order[start[RING(i, j)]++] = j * nx + i; no++; }
#undef RING
    for (int k = 0; k < no; k++) {
      if (nodes[order[k]].done) continue;
      refine_node(L, &rim, &H, order[k] % L->nx, order[k] / L->nx, nodes, have_alm ? &alm : 0);
    }
  }
  // ai: a blur made in bin is bin's, given back with it; one made fresh (no spare) freed here
  if (blurred != bin) free(blurred);
  frame_buf_done(fb, bin);
  // the border is all there is to hang the mesh on
  if (mesh) {
    if (have_alm) border_fill_anchors(L, &H, iw, ih, nodes, anchors, mpx);
    else border_fill_axis(L, &H, iw, ih, nodes);
  }
  float msum = 0;
  int mcount = 0;
  for (int k = 0; k < nn; k++) if (nodes[k].done) { msum += nodes[k].score; mcount++; }
  res->mark_score = mesh ? msum / (mcount ? mcount : 1) : 0;
  res->found = 1;
  reg->H = H; reg->nodes = nodes;
  ob_prof_ms[PROF_MESH] += ob_now_ms() - tm;
  return 1;
}

int ob_acquire(const ob_layout_t *L, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res, ob_fbufs_t *fb) {
  int which;
  return acquire_core(&L, 1, pim, mesh, reg, res, NULL, 0, 0, &which, fb);
}
int ob_acquire_any(const ob_layout_t *const *Ls, int nL, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res, int *which, ob_fbufs_t *fb) {
  return acquire_core(Ls, nL, pim, mesh, reg, res, NULL, 0, 0, which, fb);
}
int ob_acquire_quad(const ob_layout_t *L, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res,
                    const float *quad, int orient, float score, ob_fbufs_t *fb) {
  int which;
  return acquire_core(&L, 1, pim, mesh, reg, res, quad, orient, score, &which, fb);
}

// ai: the mesh cell m falls in along one axis (node ascending, nn of them): the last k of 0 to nn - 2 with k = 0 or
// ai: m >= node[k], by halving (the walk from 0 it replaces gave the same k, NaN included: 0)
static inline int node_cell(const float *node, int nn, float m) {
  int lo = 0, hi = nn - 2;
  while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (m >= node[mid]) lo = mid; else hi = mid - 1; }
  return lo;
}
void ob_map_point(const ob_layout_t *L, const ob_reg_t *reg, float mx, float my, float *pu, float *pv) {
  const int j = node_cell(L->node_y, L->ny, my), i = node_cell(L->node_x, L->nx, mx);
  float v = (my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]), u = (mx - L->node_x[i]) / (L->node_x[i + 1] - L->node_x[i]);
  const node_t *n00 = &reg->nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
  float dx = (n00->dx * (1 - u) + n10->dx * u) * (1 - v) + (n01->dx * (1 - u) + n11->dx * u) * v;
  float dy = (n00->dy * (1 - u) + n10->dy * u) * (1 - v) + (n01->dy * (1 - u) + n11->dy * u) * v;
  project(&reg->H, mx, my, pu, pv);
  *pu += dx; *pv += dy;
}

// Any regular grid of points in module coordinates, through the homography and the mesh: point (xx, yy) of the
// block is at (mx0 + (x0 + xx) * step, my0 + (y0 + yy) * step) and lands in out[yy * pitch + xx]. Points up to the
// next mesh column share their four nodes and differ in mx alone, so a row is walked in such runs, four points
// to a vector. The picture is such a grid, n samples across its span, taken a strip of columns at a time.
// Which mesh column a point falls in depends on its column alone, so the runs are found once, not once a row,
// and point by point with the comparison the scalar walk makes, so no point can land in a different column.
static inline int grid_runs(const ob_layout_t *L, float mx0, float step, int x0, int nx, int *run_i, int *run_end) {
  int runs = 0;
  for (int xx = 0, i = 0; xx < nx; runs++) {
    while (i < L->nx - 2 && mx0 + (float)(x0 + xx) * step >= L->node_x[i + 1]) i++;
    int e = xx + 1;
    while (e < nx && !(i < L->nx - 2 && mx0 + (float)(x0 + e) * step >= L->node_x[i + 1])) e++;
    run_i[runs] = i; run_end[runs] = e; xx = e;
  }
  return runs;
}
void ob_sample_grid(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const image_t im = *pim;
  const homo_t H = reg->H;
  const node_t *nodes = reg->nodes;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: the rows go down the grid, so the mesh row a row falls in only moves on: the same comparisons as a search from 0
  int j = 0;
  for (int yy = 0; yy < ny; yy++) {
    float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    float v = (my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]);
    // ai: Along a row, mx is the one thing that changes, and within a run the mesh's displacement is linear in it, so
    // ai: everything but the homography's division is a + b mx: dx = cx + ex mx (the cell's bilinear blend at this v,
    // ai: its slope over the column's width folded in), the homography's numerators and denominator likewise, and one
    // ai: reciprocal of the denominator for both. (2026-10-26: about 35 operations and three divisions a vector before,
    // ai: the division in u on the critical path; the same map to rounding.)
    const float kw = H.h[7] * my + 1, ku = H.h[1] * my + H.h[2], kv = H.h[4] * my + H.h[5];
#ifdef OB_SIMD
    const v128_t one = wasm_f32x4_splat(1), four = wasm_f32x4_splat(4);
    const v128_t h0 = wasm_f32x4_splat(H.h[0]), h3 = wasm_f32x4_splat(H.h[3]), h6 = wasm_f32x4_splat(H.h[6]);
    const v128_t vku = wasm_f32x4_splat(ku), vkv = wasm_f32x4_splat(kv), vkw = wasm_f32x4_splat(kw);
    const v128_t vmx0 = wasm_f32x4_splat(mx0), vstep = wasm_f32x4_splat(step);
    // ai: Each run's terms first, then the row four points at a time from its first point: a vector across a run's end
    // ai: takes each lane's own run's terms, so every point is worked out as the scalar walk works it out. (2026-11-01:
    // ai: each run took its own vectors from its own first point, and its last one, reaching past its end, was worked
    // ai: out again by the next run: about one vector a run, in a strip of 16, and a loop exit mispredicted a run.)
    float tex[OB_MAX_NODES + 1], tcx[OB_MAX_NODES + 1], tey[OB_MAX_NODES + 1], tcy[OB_MAX_NODES + 1];
    for (int r = 0; r < runs; r++) {
      const int i = run_i[r];
      const node_t *n00 = &nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
      const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
      tex[r] = ((n10->dx - n00->dx) * (1 - v) + (n11->dx - n01->dx) * v) * inv; tcx[r] = n00->dx * (1 - v) + n01->dx * v - tex[r] * L->node_x[i];
      tey[r] = ((n10->dy - n00->dy) * (1 - v) + (n11->dy - n01->dy) * v) * inv; tcy[r] = n00->dy * (1 - v) + n01->dy * v - tey[r] * L->node_x[i];
    }
    v128_t vex = wasm_f32x4_splat(tex[0]), vcx = wasm_f32x4_splat(tcx[0]), vey = wasm_f32x4_splat(tey[0]), vcy = wasm_f32x4_splat(tcy[0]);
    // The column index as floats, stepped by four: whole numbers, so exact.
    v128_t vgx = wasm_f32x4_make((float)x0, (float)(x0 + 1), (float)(x0 + 2), (float)(x0 + 3));
    for (int xx = 0, r = 0; xx < nx; xx += 4, vgx = wasm_f32x4_add(vgx, four)) {
      if (xx >= run_end[r]) {
        while (xx >= run_end[r]) r++;
        vex = wasm_f32x4_splat(tex[r]); vcx = wasm_f32x4_splat(tcx[r]); vey = wasm_f32x4_splat(tey[r]); vcy = wasm_f32x4_splat(tcy[r]);
      }
      v128_t ex4 = vex, cx4 = vcx, ey4 = vey, cy4 = vcy;
      for (int q = r; q + 1 < runs && run_end[q] < xx + 4; q++) {   // the lanes from run q's end on are run q + 1's
        const v128_t past = wasm_f32x4_le(wasm_f32x4_splat((float)(x0 + run_end[q])), vgx);
        ex4 = wasm_v128_bitselect(wasm_f32x4_splat(tex[q + 1]), ex4, past); cx4 = wasm_v128_bitselect(wasm_f32x4_splat(tcx[q + 1]), cx4, past);
        ey4 = wasm_v128_bitselect(wasm_f32x4_splat(tey[q + 1]), ey4, past); cy4 = wasm_v128_bitselect(wasm_f32x4_splat(tcy[q + 1]), cy4, past);
      }
      const v128_t vmx = wasm_f32x4_add(vmx0, wasm_f32x4_mul(vgx, vstep));
      const v128_t rw = wasm_f32x4_div(one, wasm_f32x4_add(wasm_f32x4_mul(h6, vmx), vkw));
      const v128_t pu = wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(h0, vmx), vku), rw), pv = wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(h3, vmx), vkv), rw);
      const v128_t dx = wasm_f32x4_add(cx4, wasm_f32x4_mul(ex4, vmx)), dy = wasm_f32x4_add(cy4, wasm_f32x4_mul(ey4, vmx));
      const v128_t got = sample4(&im, wasm_f32x4_add(pu, dx), wasm_f32x4_add(pv, dy));
      if (xx + 4 <= nx) { wasm_v128_store(out + yy * pitch + xx, got); continue; }
      float t[4];   // the row's last vector, past its end: those lanes are dropped
      wasm_v128_store(t, got);
      for (int l = 0; xx + l < nx; l++) out[yy * pitch + xx + l] = t[l];
    }
#else
    for (int r = 0, xx = 0; r < runs; r++) {
      const int i = run_i[r], end = run_end[r];
      const node_t *n00 = &nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
      const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
      const float ex = ((n10->dx - n00->dx) * (1 - v) + (n11->dx - n01->dx) * v) * inv, cx = n00->dx * (1 - v) + n01->dx * v - ex * L->node_x[i];
      const float ey = ((n10->dy - n00->dy) * (1 - v) + (n11->dy - n01->dy) * v) * inv, cy = n00->dy * (1 - v) + n01->dy * v - ey * L->node_x[i];
      for (; xx < end; xx++) {
        const float mx = mx0 + (float)(x0 + xx) * step, rw = 1 / (H.h[6] * mx + kw);
        const float pu = (H.h[0] * mx + ku) * rw, pv = (H.h[3] * mx + kv) * rw;
        const float dx = cx + ex * mx, dy = cy + ey * mx;
        out[yy * pitch + xx] = sample(&im, pu + dx, pv + dy);
      }
    }
#endif
  }
}
#ifdef OB_NEON
// ai: arm64, the decode's picture at gamma 1 (ob_sample_grid_pic), the same map with its read in integers (its own
// ai: values; the blocks a decode verifies are its test). Every term of a row is a line in the column: the homography's
// ai: numerators and denominator stepped by four columns a vector, the mesh's displacement D + v E tabled a column at a
// ai: time once a mesh row (D on mesh row j less the half pixel, E the step to row j + 1; D x, E x, D y, E y for four
// ai: columns side by side, two LDP a vector), one division a vector. The point's x and y to 16.16 by FCVTZU (below 0
// ai: is 0), their integer parts by UZP2, clamped on 16-bit lanes to w - 2 and h - 2 (so the reads stay in the frame),
// ai: fx the fraction's top byte (rounded: x is a 512th of a pixel on, in D), fy its top 15 bits. A point's two pixel
// ai: pairs are one word in a general register: row y's from the pixel and row y + 1's from two bytes before, merged
// ai: by BFXIL (lin_word_lo, lin_word); TBL sorts eight points' words into a b and c d; each row's pair blended on
// ai: bytes, 256 a + (b - a) fx (USHLL, UMLAL2, UMLSL: weights summing to 256, as the fraction), the two rows by
// ai: SQRDMULH on their halved difference times fy, the result (128 times the grey) to floats by one widening multiply
// ai: by 257 and a conversion by 2^-23 (1 / 32640 within 2e-5). A row is taken 16 points a step, the next step's
// ai: offsets and weights worked out before this step's reads (lin_pts, lin_read), so the reads do not wait on the
// ai: division. At most GRID_LIN columns a pass. (2026-11-02, an S26 prime core, pictures of 768 to 1536: 2.41
// ai: cycles a point in the strips, 0.51 ns at 4.74 GHz, against 3.26 for the float form before it, sample4r four
// ai: vectors at a time: its blend, clamps and fractions in floats, its words two transfers a point; 13.9 instructions
// ai: a point against 19.8. Its values within 3e-3 of the grey range of the float form's on a frame of random bytes,
// ai: 3e-4 on average; verified blocks over the eleven S26 replays and five cells +2 in 173 810, -11 to +10 a run,
// ai: where a shift of the float form's points by 4e-6 pixels moves them -2 to +8 a run. Weights summing to 255, the
// ai: fraction cut: -19.)
enum { GRID_LIN = 256 };
static void sample_grid_lin(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const int w = pim->w, nn = L->nx;
  const float *h = reg->H.h;
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1];
  const int runs = grid_runs(L, mx0, step, x0, nx, run_i, run_end);
  // ai: a column's place in its mesh column (0 to 1), and the table, 24 columns past the end (zeros) for the step that
  // ai: works out points past the row
  enum { TW = GRID_LIN + 24 };
  float tu[TW], T[4 * TW];
  for (int r = 0, k = 0; r < runs; r++) {
    const int i = run_i[r];
    const float inv = 1.0f / (L->node_x[i + 1] - L->node_x[i]);
    for (; k < run_end[r]; k++) tu[k] = (mx0 + (float)(x0 + k) * step - L->node_x[i]) * inv;
  }
  for (int k = nx; k < nx + 4; k++) tu[k] = 0;
  for (int i = 4 * ((nx + 3) & ~3); i < 4 * (nx + 24); i++) T[i] = 0;
  const float m0 = mx0 + (float)x0 * step;
  const float32x4_t lane = { 0, 1, 2, 3 }, one = vdupq_n_f32(1);
  const float32x4_t BwL = vmulq_n_f32(lane, h[6] * step), BuL = vmulq_n_f32(lane, h[0] * step), BvL = vmulq_n_f32(lane, h[3] * step);
  // ai: the three lines' steps over four columns, and v (lane 3, a row's)
  float32x4_t C = { 4 * (h[6] * step), 4 * (h[0] * step), 4 * (h[3] * step), 0 };
  const lin_k_t K = lin_k(w, pim->h);
  const uint8_t *r0 = pim->px, *r1 = pim->px + w - 2;
  int j = 0, built = -1;
  for (int yy = 0; yy < ny; yy++) {
    const float my = my0 + (float)(y0 + yy) * step;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    if (j != built) {
      const node_t *n0 = reg->nodes + j * nn, *n1 = n0 + nn;
      for (int r = 0, s = 0; r < runs; s = run_end[r++]) {
        const node_t *a = n0 + run_i[r], *b = n1 + run_i[r];
        const float32x4_t c0x = vdupq_n_f32(a[0].dx - (0.5f - 1.0f / 512)), c1x = vdupq_n_f32(a[1].dx - a[0].dx), e0x = vdupq_n_f32(b[0].dx - a[0].dx), e1x = vdupq_n_f32((b[1].dx - b[0].dx) - (a[1].dx - a[0].dx));
        const float32x4_t c0y = vdupq_n_f32(a[0].dy - 0.5f), c1y = vdupq_n_f32(a[1].dy - a[0].dy), e0y = vdupq_n_f32(b[0].dy - a[0].dy), e1y = vdupq_n_f32((b[1].dy - b[0].dy) - (a[1].dy - a[0].dy));
        // ai: four columns a vector from the one holding the run's first; its lanes before the run are the last run's
        // ai: (kept), and a vector past the run's end is rewritten by the next run
        for (int k = s & ~3; k < run_end[r]; k += 4) {
          const float32x4_t U = vld1q_f32(tu + k);
          float32x4_t dx = vfmaq_f32(c0x, c1x, U), ex = vfmaq_f32(e0x, e1x, U), dy = vfmaq_f32(c0y, c1y, U), ey = vfmaq_f32(e0y, e1y, U);
          if (k < s) {
            const uint32x4_t keep = vcltq_f32(vaddq_f32(vdupq_n_f32((float)k), lane), vdupq_n_f32((float)s));
            dx = vbslq_f32(keep, vld1q_f32(T + 4 * k), dx); ex = vbslq_f32(keep, vld1q_f32(T + 4 * k + 4), ex);
            dy = vbslq_f32(keep, vld1q_f32(T + 4 * k + 8), dy); ey = vbslq_f32(keep, vld1q_f32(T + 4 * k + 12), ey);
          }
          vst1q_f32(T + 4 * k, dx); vst1q_f32(T + 4 * k + 4, ex); vst1q_f32(T + 4 * k + 8, dy); vst1q_f32(T + 4 * k + 12, ey);
        }
      }
      built = j;
    }
    C = vsetq_lane_f32((my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]), C, 3);
    float32x4_t nw = vaddq_f32(BwL, vdupq_n_f32(h[6] * m0 + (h[7] * my + 1))), nu = vaddq_f32(BuL, vdupq_n_f32(h[0] * m0 + (h[1] * my + h[2]))), nv = vaddq_f32(BvL, vdupq_n_f32(h[3] * m0 + (h[4] * my + h[5])));
    // ai: the point's coordinates less the half pixel (in D); the steps by FMLA by element (exact as an add)
#define LIN_XY(K_, X, Y) do { \
      const float *tk = T + 4 * (K_); \
      const float32x4_t rw_ = vdivq_f32(one, nw); \
      X = vfmaq_f32(vfmaq_laneq_f32(vld1q_f32(tk), vld1q_f32(tk + 4), C, 3), nu, rw_); \
      Y = vfmaq_f32(vfmaq_laneq_f32(vld1q_f32(tk + 8), vld1q_f32(tk + 12), C, 3), nv, rw_); \
      nw = vfmaq_laneq_f32(nw, one, C, 0); nu = vfmaq_laneq_f32(nu, one, C, 1); nv = vfmaq_laneq_f32(nv, one, C, 2); \
    } while (0)
    float *o = out + (size_t)yy * pitch;
    float32x4_t xa, ya, xb, yb, xc, yc, xd, yd;
    LIN_XY(0, xa, ya); LIN_XY(4, xb, yb); LIN_XY(8, xc, yc); LIN_XY(12, xd, yd);
    lin_pts_t pa = lin_pts(&K, xa, ya, xb, yb), pb = lin_pts(&K, xc, yc, xd, yd);
    int k = 0;
    for (; k + 16 < nx; k += 16) {
      LIN_XY(k + 16, xa, ya); LIN_XY(k + 20, xb, yb); LIN_XY(k + 24, xc, yc); LIN_XY(k + 28, xd, yd);
      const lin_pts_t na = lin_pts(&K, xa, ya, xb, yb), nb = lin_pts(&K, xc, yc, xd, yd);
      lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8);
      pa = na; pb = nb;
    }
#undef LIN_XY
    if (k + 16 <= nx) { lin_read(r0, r1, &K, pa, o + k); lin_read(r0, r1, &K, pb, o + k + 8); continue; }
    float t[16];   // the row's last points, past its end: those are dropped
    lin_read(r0, r1, &K, pa, t); lin_read(r0, r1, &K, pb, t + 8);
    for (int l = 0; k + l < nx; l++) o[k + l] = t[l];
  }
}
#endif
void ob_sample_grid_pic(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
#ifdef OB_NEON
  if (pim->lin) {
    for (int c = 0; c < nx; c += GRID_LIN) sample_grid_lin(L, pim, reg, mx0, my0, step, x0 + c, nx - c < GRID_LIN ? nx - c : GRID_LIN, y0, ny, out + c, pitch);
    return;
  }
#endif
  ob_sample_grid(L, pim, reg, mx0, my0, step, x0, nx, y0, ny, out, pitch);
}

