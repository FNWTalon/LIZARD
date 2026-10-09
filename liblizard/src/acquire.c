// Registration: find the four finders, fix orientation, hang a mesh on the alignment marks.
// The binarizer here only serves the finder search; data cells are never thresholded.
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
OB_TLS float ob_raw_quad[8];
// What the thin frame's finder settled on, for a finder elsewhere to be held to: the final
// score and orient, then which form and which mark quad won, the source of the final quad (0 marks, 1 lines,
// 2 mark_crop, -1 none), whether the marks alone settled it, the mark stage's best, and form 0's quad count.
OB_TLS float ob_finder_info[8];

void ob_image_init(image_t *im, const uint8_t *px, int w, int h, float gamma) {
  im->px = px; im->w = w; im->h = h;
  for (int i = 0; i < 256; i++) im->lut[i] = gamma == 1.0f ? i / 255.0f : powf(i / 255.0f, gamma);
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

// Three passes: 8 x 8 block means, a 5 x 5 mean of those less 3 as each block's threshold, then the
// compare. The vector paths cover whole blocks; a ragged right or bottom edge takes the scalar one.
static void binarize(const uint8_t *img, int w, int h, uint8_t *bin) {
  int bw = (w + BLK - 1) / BLK, bh = (h + BLK - 1) / BLK;
  int *mean = malloc((size_t)bw * bh * sizeof(int));
  uint8_t *trow = malloc((size_t)bw * BLK + 16);
  uint16_t *col = malloc(((size_t)bw * BLK + 16) * sizeof(uint16_t));
  for (int by = 0; by < bh; by++) {
    int bx = 0;
#ifdef OB_SIMD
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
    for (int bx = 0; bx < bw; bx++) {
      int s = 0, n = 0;
      for (int dy = -2; dy <= 2; dy++) for (int dx = -2; dx <= 2; dx++) {
        int yy = by + dy, xx = bx + dx;
        if (yy < 0 || xx < 0 || yy >= bh || xx >= bw) continue;
        s += mean[yy * bw + xx]; n++;
      }
      int t = s / n - 3;
      memset(trow + bx * BLK, t < 0 ? 0 : t, BLK);   // a pixel is never below 0, so a negative threshold and 0 mark the same (nothing)
    }
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
  free(mean); free(trow); free(col);
}

void ob_test_binarize(const uint8_t *img, int w, int h, uint8_t *bin) { binarize(img, w, h, bin); }

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

typedef struct { float x, y, unit; int hits; } cand_t;
enum { MAX_CAND = 1024 };

static int ratio_ok(const int *r, float *unit_out) {
  int total = r[0] + r[1] + r[2] + r[3] + r[4];
  if (total < 12) return 0;
  float unit = total / 7.0f, tol = unit * 0.7f + 0.5f;
  if (fabsf(r[0] - unit) > tol || fabsf(r[1] - unit) > tol || fabsf(r[3] - unit) > tol || fabsf(r[4] - unit) > tol) return 0;
  if (fabsf(r[2] - 3 * unit) > 1.5f * tol) return 0;
  *unit_out = unit;
  return 1;
}

// Walk outwards from a point inside the centre run along (dx, dy); fills the five runs and
// the centre coordinate along that axis.
static int cross_check(const uint8_t *bin, int w, int h, int x, int y, int dx, int dy, int maxrun, float *centre, float *unit) {
  int r[5] = { 0 }, i;
  #define AT(k) bin[(y + (k) * dy) * w + x + (k) * dx]
  #define IN(k) (x + (k) * dx >= 0 && x + (k) * dx < w && y + (k) * dy >= 0 && y + (k) * dy < h)
  if (!bin[y * w + x]) return 0;
  for (i = 0; IN(-i) && AT(-i); i++) if (++r[2] > maxrun) return 0;
  for (; IN(-i) && !AT(-i); i++) if (++r[1] > maxrun) return 0;
  for (; IN(-i) && AT(-i); i++) if (++r[0] > maxrun) return 0;
  int j;
  for (j = 1; IN(j) && AT(j); j++) if (++r[2] > maxrun) return 0;
  for (; IN(j) && !AT(j); j++) if (++r[3] > maxrun) return 0;
  for (; IN(j) && AT(j); j++) if (++r[4] > maxrun) return 0;
  #undef AT
  #undef IN
  if (!ratio_ok(r, unit)) return 0;
  *centre = (dx ? x : y) + j - r[4] - r[3] - r[2] / 2.0f;
  return 1;
}

// Run boundaries of one row, sixteen pixels to a compare: pos[2k] is where dark run k starts and
// pos[2k + 1] where it ends. The pixel left of the row counts as light, and a dark run that reaches
// the right edge ends there. Returns the number of boundaries, always even.
static int run_edges(const uint8_t *row, int w, int *pos) {
  int np = 0, x = 1;
  if (row[0]) pos[np++] = 0;
#ifdef OB_SIMD
  for (; x + 16 <= w; x += 16) {
    unsigned m = (unsigned)wasm_i8x16_bitmask(wasm_i8x16_ne(wasm_v128_load(row + x), wasm_v128_load(row + x - 1)));
    for (; m; m &= m - 1) pos[np++] = x + __builtin_ctz(m);
  }
#endif
  for (; x < w; x++) if (row[x] != row[x - 1]) pos[np++] = x;
  if (np & 1) pos[np++] = w;
  return np;
}

// Every dark run that has two dark runs before it on its row closes a window of five runs; a window
// in the finder's 1:1:3:1:1 is cross-checked down and across. Rows are first turned into run
// boundaries, sixteen pixels to a compare, so the work follows the number of edges, not of pixels.
static int find_candidates(const uint8_t *bin, int w, int h, cand_t *out) {
  int n = 0;
  int *pos = malloc(((size_t)w + 2) * sizeof(int));
  if (!pos) return 0;
  for (int y = 1; y < h; y += 2) {
    int np = run_edges(bin + y * w, w, pos), x;
    for (int k = 2; 2 * k + 1 < np; k++) {
      const int *q = pos + 2 * k - 4;
      int r[5] = { q[1] - q[0], q[2] - q[1], q[3] - q[2], q[4] - q[3], q[5] - q[4] };
      x = q[5];
      float unit;
      if (!ratio_ok(r, &unit)) continue;
      int cx = x - r[4] - r[3] - r[2] / 2;
      float fy, fx, uv, uh;
      if (!cross_check(bin, w, h, cx, y, 0, 1, r[2] * 3, &fy, &uv) || !cross_check(bin, w, h, cx, (int)fy, 1, 0, r[2] * 2, &fx, &uh)) continue;
      float u = 0.5f * (uv + uh);
      // A screen seen from the side is squeezed along one axis (by cos of the angle), so the
      // two crossings of a real finder can differ a lot. 2.5 admits about 65 degrees.
      if (uv > 2.5f * uh || uh > 2.5f * uv) continue;
      int c;
      for (c = 0; c < n; c++) if (fabsf(out[c].x - fx) < u * 2 && fabsf(out[c].y - fy) < u * 2) break;
      if (c < n) {
        out[c].x = (out[c].x * out[c].hits + fx) / (out[c].hits + 1);
        out[c].y = (out[c].y * out[c].hits + fy) / (out[c].hits + 1);
        out[c].unit = (out[c].unit * out[c].hits + u) / (out[c].hits + 1);
        out[c].hits++;
      } else if (n < MAX_CAND) out[n++] = (cand_t){ fx, fy, u, 1 };
    }
  }
  free(pos);
  return n;
}

static float cross2(const cand_t *a, const cand_t *b, const cand_t *c) { return (b->x - a->x) * (c->y - a->y) - (b->y - a->y) * (c->x - a->x); }

// The four true finders are the largest-unit candidates (data can mimic the ratio only at
// half the ring width) and span the widest convex quad. Output is clockwise in the image.
static int pick_quad(cand_t *c, int n, cand_t *q) {
  int m = 0;
  float umax = 0;
  for (int i = 0; i < n; i++) if (c[i].hits >= 2 && c[i].unit > umax) umax = c[i].unit;
  // Perspective and a steep angle together put a factor of two between the near and far
  // finder, so the size gate is loose; the widest convex quad does the real selection.
  for (int i = 0; i < n; i++) if (c[i].hits >= 2 && c[i].unit >= 0.4f * umax) c[m++] = c[i];
  if (m < 4) return 0;
  // A real finder is crossed by many scan rows; a data cell that mimics the ratio by two or
  // three. Keep the best-supported dozen.
  for (int i = 1; i < m; i++) { cand_t t = c[i]; int j = i; while (j > 0 && c[j - 1].hits * c[j - 1].unit < t.hits * t.unit) { c[j] = c[j - 1]; j--; } c[j] = t; }
  if (m > 12) m = 12;
  float best = 0;
  int bi[4] = { 0 };
  for (int a = 0; a < m; a++) for (int b = a + 1; b < m; b++) for (int d = b + 1; d < m; d++) for (int e = d + 1; e < m; e++) {
    int id[4] = { a, b, d, e };
    float gx = 0, gy = 0;
    for (int k = 0; k < 4; k++) { gx += c[id[k]].x / 4; gy += c[id[k]].y / 4; }
    for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++)
      if (atan2f(c[id[j]].y - gy, c[id[j]].x - gx) < atan2f(c[id[i]].y - gy, c[id[i]].x - gx)) { int t = id[i]; id[i] = id[j]; id[j] = t; }
    float area = 0;
    int convex = 1;
    for (int k = 0; k < 4; k++) {
      float z = cross2(&c[id[k]], &c[id[(k + 1) & 3]], &c[id[(k + 2) & 3]]);
      if (z <= 0) convex = 0;
      area += z;
    }
    if (convex && area > best) { best = area; memcpy(bi, id, sizeof bi); }
  }
  if (best <= 0) return 0;
  for (int k = 0; k < 4; k++) q[k] = c[bi[k]];
  return 1;
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
      for (int k = c; k < 9; k++) A[r][k] -= f * A[c][k];
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

static float orient_score(const ob_layout_t *L, const image_t *im, const homo_t *H) {
  float score = 0;
  for (int c = 0; c < 4; c++) {
    float fx = (c == 1 || c == 2) ? L->w - OB_FB / 2.0f : OB_FB / 2.0f, fy = c >= 2 ? L->h - OB_FB / 2.0f : OB_FB / 2.0f;
    float u, v, dark, light = 0;
    project(H, fx, fy, &u, &v); dark = sample(im, u, v);
    for (int k = 0; k < 4; k++) { project(H, fx + (k == 0 ? 4 : k == 1 ? -4 : 0), fy + (k == 2 ? 4 : k == 3 ? -4 : 0), &u, &v); light += sample(im, u, v) / 4; }
    float ref = 0.5f * (dark + light), con = light - dark;
    if (con < 1e-4f) return -1e9f;
    int sx, sy;
    ob_orient_rect(L, c, &sx, &sy);
    for (int b = 0; b < 3; b++) {
      float s = 0;
      for (int q = 0; q < 4; q++) { project(H, sx + 2 * b + (q & 1) + 0.5f, sy + (q >> 1) + 0.5f, &u, &v); s += sample(im, u, v) / 4; }
      float d = (ref - s) / con;
      if (d > 0.5f) d = 0.5f; else if (d < -0.5f) d = -0.5f;
      score += ((OB_ORIENT[c] >> (2 - b)) & 1) ? d : -d;
    }
  }
  return score;
}

// ---------------------------------------------------------------- registration mesh

// Normalized correlation of a node's known cells against the image, with the node shifted by
// (ox, oy) modules from where the current estimate puts it.
// r: the patch as cell offsets from (cx0, cy0), x from r[0] to r[1] and y from r[2] to r[3], ends excluded.
// Every cell is placed through the homography itself, then moved by what the neighbours say (pdx, pdy). A local
// affine map at the node was used until 2026-09-23 and is right only while the template is small in pixels: a
// corner template is 12 modules, 31 px on a 286-module symbol at 1080 fill 0.7 and 96 px on a 94-module one, and
// at 30 degrees this camera's perspective changes scale by about 6% across 96 px, which misplaces the template's
// far cells by over a pixel. Measured: the refined corners of LIZARD-16 at 94 modules came out 1.7 px off the
// truth where the unrefined mark quad was 0.6.
static float node_ncc(const ob_layout_t *L, const image_t *im, int cx0, int cy0, const int *r, const homo_t *H, float pdx, float pdy, float ox, float oy) {
  float st = 0, ss = 0, stt = 0, sss = 0, sts = 0;
  int n = 0, lo = r[0], hi = r[1], ylo = r[2], yhi = r[3], side = hi - lo;
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
    if (L->thin && kind == CELL_DATA) continue;   // the picture inside a thin frame: not the frame's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    float t = (kind & 3) == CELL_DARK ? -1.0f : 1.0f, s = S[(dy - ylo) * side + dx - lo];
    st += t; ss += s; stt += t * t; sss += s * s; sts += t * s; n++;
  }
  float vt = stt - st * st / n, vs = sss - ss * ss / n;
  if (vt <= 0 || vs <= 1e-9f) return 0;
  return (sts - st * ss / n) / sqrtf(vt * vs);
}


static void refine_node(const ob_layout_t *L, const image_t *im, const homo_t *H, int i, int j, node_t *nodes) {
  int finder = !L->thin && (i == 0 || i == L->nx - 1) && (j == 0 || j == L->ny - 1);
  float mcx, mcy;
  int cx0, cy0, half;
  if (finder) {
    mcx = i ? L->w - OB_FB / 2.0f : OB_FB / 2.0f; mcy = j ? L->h - OB_FB / 2.0f : OB_FB / 2.0f;
    cx0 = (int)mcx; cy0 = (int)mcy; half = OB_FB / 2;
  } else {
    cx0 = L->thin ? (int)L->node_x[i] : i == 0 ? OB_FB / 2 : i == L->nx - 1 ? L->w - OB_FB / 2 - 1 : (int)L->node_x[i];
    cy0 = L->thin ? (int)L->node_y[j] : j == 0 ? OB_FB / 2 : j == L->ny - 1 ? L->h - OB_FB / 2 - 1 : (int)L->node_y[j];
    mcx = cx0 + 0.5f; mcy = cy0 + 0.5f; half = 2;
  }
  // A thin frame's template is as long as the border lets it be: thirteen modules along, the border's
  // five across, and an L of both at a corner. Small symbols need the area; 5 x 5 is 8 px there.
  int rect[4] = { -half, half + (half == 2), -half, half + (half == 2) };
  if (L->thin) {
    int ex = (j == 0 || j == L->ny - 1) ? 6 : 2, ey = (i == 0 || i == L->nx - 1) ? 6 : 2;
    // A corner that carries a mark is a different thing to match. Six modules either way stays INSIDE a 12-module
    // mark, so the template is a dark block with one module of light rim, and once blur has closed the mark's gap
    // there is nothing in it to lock to: measured at 2 px of defocus, refining the corners took a marked symbol
    // from 6 blocks of 96 to 2 where it takes an unmarked one from 19 to 35. Reaching a module past the mark's
    // inner edge puts the square's own edges in the template, which is the strongest feature the symbol has.
    if (L->corner >= 5 && ex == 6 && ey == 6) { ex = L->corner - 1 < OB_FB - 3 ? L->corner - 1 : OB_FB - 3; ey = ex; }
    rect[0] = -ex < -cx0 ? -cx0 : -ex; rect[1] = ex + 1 > L->w - cx0 ? L->w - cx0 : ex + 1;
    rect[2] = -ey < -cy0 ? -cy0 : -ey; rect[3] = ey + 1 > L->h - cy0 ? L->h - cy0 : ey + 1;
  }
  // Start from what the already-measured neighbours say the local error is.
  float pdx = 0, pdy = 0;
  int pn = 0;
  for (int dj = -1; dj <= 1; dj++) for (int di = -1; di <= 1; di++) {
    int ii = i + di, jj = j + dj;
    if (ii < 0 || jj < 0 || ii >= L->nx || jj >= L->ny || !nodes[jj * L->nx + ii].done) continue;
    pdx += nodes[jj * L->nx + ii].dx; pdy += nodes[jj * L->nx + ii].dy; pn++;
  }
  if (pn) { pdx /= pn; pdy /= pn; }
  float px, py, ax, ay, bx, by;
  project(H, mcx, mcy, &px, &py); project(H, mcx + 1, mcy, &ax, &ay); project(H, mcx, mcy + 1, &bx, &by);
  // The node's local Jacobian, only to turn the offset found in modules into pixels at the end.
  float jxx = ax - px, jxy = ay - py, jyx = bx - px, jyy = by - py;
  // The coarse search, 7 x 7 offsets of half a module. A shift of a whole module puts every cell where its
  // neighbour was, so the 49 correlations read the same image points many times over. They are read once, on a
  // half-module grid, and each correlation picks its cells out of it: the same floats summed in the same order
  // as node_ncc would, for about a seventh of the reads.
  enum { GMAX = 2 * OB_FB + 5 };
  float G[GMAX * GMAX + 4], tv[OB_FB * OB_FB];   // + 4: the eighth lane of the last row's sums reads one float past the grid
  int at[OB_FB * OB_FB], cells = 0;
  const int lo = rect[0], ylo = rect[2], side = rect[1] - rect[0], tall = rect[3] - rect[2], GW = 2 * side + 5, GH = 2 * tall + 5;
  // Placed through the homography, as node_ncc does and for the same reason.
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
      wasm_v128_store(G + gy * GW + gx, sample4(im, wasm_f32x4_add(u, wasm_f32x4_splat(pdx)), wasm_f32x4_add(v, wasm_f32x4_splat(pdy))));
    }
#endif
    for (; gx < GW; gx++) { const float mx = (float)(2 * (cx0 + lo) + gx - 2) * 0.5f; float u, v; project(H, mx, my, &u, &v); G[gy * GW + gx] = sample(im, u + pdx, v + pdy); }
  }
  float st = 0, stt = 0;
  for (int dy = 0; dy < tall; dy++) for (int dx = 0; dx < side; dx++) {
    int kind = L->kind[(cy0 + ylo + dy) * L->w + cx0 + lo + dx];
    if (L->thin && kind == CELL_DATA) continue;   // the picture inside a thin frame: not the frame's to know
    if (kind & CELL_WORD) continue;               // nor the format word, whose bits this end did not choose (layout.h)
    tv[cells] = (kind & 3) == CELL_DARK ? -1.0f : 1.0f; at[cells] = (2 * dy + 3) * GW + 2 * dx + 3;
    st += tv[cells]; stt += tv[cells] * tv[cells]; cells++;
  }
  const float vt = stt - st * st / cells;
  float best = -2, box = 0, boy = 0;
  for (int k = 0; k < 4; k++) G[GW * GH + k] = 0;
  for (int b = -3; b <= 3; b++) {
    // The seven offsets a of one b sit side by side in the grid, so their sums run as lanes: each lane adds the
    // same cells in the same order as a loop of its own would.
    float SS[8], SQ[8], ST[8];
#ifdef OB_SIMD
    const float *g0 = G + b * GW - 3;
    v128_t s0 = wasm_f32x4_splat(0), s1 = s0, q0 = s0, q1 = s0, t0 = s0, t1 = s0;
    for (int k = 0; k < cells; k++) {
      const v128_t v0 = wasm_v128_load(g0 + at[k]), v1 = wasm_v128_load(g0 + at[k] + 4), t = wasm_f32x4_splat(tv[k]);
      s0 = wasm_f32x4_add(s0, v0); s1 = wasm_f32x4_add(s1, v1); q0 = wasm_f32x4_add(q0, wasm_f32x4_mul(v0, v0)); q1 = wasm_f32x4_add(q1, wasm_f32x4_mul(v1, v1));
      t0 = wasm_f32x4_add(t0, wasm_f32x4_mul(t, v0)); t1 = wasm_f32x4_add(t1, wasm_f32x4_mul(t, v1));
    }
    wasm_v128_store(SS, s0); wasm_v128_store(SS + 4, s1); wasm_v128_store(SQ, q0); wasm_v128_store(SQ + 4, q1); wasm_v128_store(ST, t0); wasm_v128_store(ST + 4, t1);
#else
    for (int a = -3; a <= 3; a++) {
      const float *g0 = G + b * GW + a;
      float ss = 0, sss = 0, sts = 0;
      for (int k = 0; k < cells; k++) { const float v = g0[at[k]]; ss += v; sss += v * v; sts += tv[k] * v; }
      SS[a + 3] = ss; SQ[a + 3] = sss; ST[a + 3] = sts;
    }
#endif
    for (int a = -3; a <= 3; a++) {
      const float ss = SS[a + 3], vs = SQ[a + 3] - ss * ss / cells, sc = vt <= 0 || vs <= 1e-9f ? 0 : (ST[a + 3] - st * ss / cells) / sqrtf(vt * vs);
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
      for (int b2 = -1; b2 <= 1; b2++) for (int a2 = -1; a2 <= 1; a2++) if (!have[b2 + 1][a2 + 1]) { g[b2 + 1][a2 + 1] = node_ncc(L, im, cx0, cy0, rect, H, pdx, pdy, box + a2 * step, boy + b2 * step); have[b2 + 1][a2 + 1] = 1; }
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
  if (dxx < -1e-6f) box += 0.125f * 0.5f * (g[1][0] - g[1][2]) / dxx;
  if (dyy < -1e-6f) boy += 0.125f * 0.5f * (g[0][1] - g[2][1]) / dyy;
  node_t *nd = &nodes[j * L->nx + i];
  nd->done = 1; nd->score = best;
  if (best < 0.35f) { nd->dx = pdx; nd->dy = pdy; return; }
  nd->dx = pdx + box * jxx + boy * jyx;
  nd->dy = pdy + box * jxy + boy * jyy;
}

// ai: Each side's nodes moved ALONG their side onto a cubic fitted to their along-side displacement; the across-side
// ai: part and the corners are left as measured. A display that scales the symbol by a fraction (part of the channel:
// ai: a page's stretch, an OS scale, a monitor's scaler) moves the border's sharp cells along each side by a sub-pixel
// ai: amount that repeats every 1 / frac(device px a module) modules, where the band-limited picture keeps its
// ai: geometry. The nodes, 16 modules apart, alias that ripple, and border_fill's Coons blend would carry it across
// ai: the picture as stripes (STATUS "The grid misalignment's source": on 07-56-38 a +-0.3 to 0.4-sample ripple of 23
// ai: modules). The cubic is in each node's projected position along the corner chord, not its module position: a
// ai: lens bend is a cubic in the image, which under perspective is not a cubic in modules (at 30 degrees and k1
// ai: 0.03 a module-coordinate cubic missed a 96-ring node by 0.47 px, this one by 0.02). Rolling-shutter jello moves
// ai: the left and right sides across themselves and the top and bottom as a whole, which the fit leaves alone.
// ai: Only trusted nodes enter (border_fill's rule), and a side is left as measured unless it has six, two of them
// ai: marks in each half (a cubic over one half would swing free over the other); a mark more than a fifth of a module
// ai: and three times the median off the first fit (a wrong lock) is dropped and the side fitted once more.
enum { ALONG_MIN = 6, ALONG_HALF = 2 };
static int along_fit(const double *t, const double *a, const int *use, int cnt, double c[4]) {
  double A[8][9];
  memset(A, 0, sizeof A);
  int used = 0;
  for (int k = 0; k < cnt; k++) {
    if (!use[k]) continue;
    const double b[4] = { 1, t[k], t[k] * t[k], t[k] * t[k] * t[k] };
    for (int p = 0; p < 4; p++) { for (int q = 0; q < 4; q++) A[p][q] += b[p] * b[q]; A[p][8] += b[p] * a[k]; }
    used++;
  }
  int lo = 0, hi = 0;
  for (int k = 1; k < cnt - 1; k++) if (use[k]) { if (t[k] < 0) lo++; else hi++; }
  if (used < ALONG_MIN || lo < ALONG_HALF || hi < ALONG_HALF) return 0;
  for (int p = 4; p < 8; p++) A[p][p] = 1;   // ai: four unknowns in the eight-row solver
  if (!solve8(A)) return 0;
  for (int p = 0; p < 4; p++) c[p] = A[p][8] / A[p][p];
  return 1;
}
static void smooth_along(const ob_layout_t *L, const homo_t *H, node_t *nodes) {
  const int nx = L->nx, ny = L->ny;
  for (int side = 0; side < 4; side++) {
    const int horiz = side == 0 || side == 2, cnt = horiz ? nx : ny;
    double ux[OB_MAX_NODES], uy[OB_MAX_NODES], mod[OB_MAX_NODES], t[OB_MAX_NODES], a[OB_MAX_NODES];
    float px[OB_MAX_NODES], py[OB_MAX_NODES];
    int use[OB_MAX_NODES], idx[OB_MAX_NODES];
    for (int k = 0; k < cnt; k++) {
      const int i = horiz ? k : (side == 1 ? nx - 1 : 0), j = horiz ? (side == 0 ? 0 : ny - 1) : k;
      idx[k] = j * nx + i;
      project(H, L->node_x[i], L->node_y[j], &px[k], &py[k]);
      // ai: the unit vector along the side at this node, in image px, and a module's length there
      float u0, v0, u1, v1;
      project(H, L->node_x[i] - (horiz ? 0.5f : 0), L->node_y[j] - (horiz ? 0 : 0.5f), &u0, &v0);
      project(H, L->node_x[i] + (horiz ? 0.5f : 0), L->node_y[j] + (horiz ? 0 : 0.5f), &u1, &v1);
      mod[k] = hypot((double)u1 - u0, (double)v1 - v0);
      ux[k] = mod[k] > 0 ? (u1 - u0) / mod[k] : 0; uy[k] = mod[k] > 0 ? (v1 - v0) / mod[k] : 0;
      const node_t *nd = &nodes[idx[k]];
      a[k] = nd->dx * ux[k] + nd->dy * uy[k];
      use[k] = nd->done && (k == 0 || k == cnt - 1 || (L->node_mark[idx[k]] && nd->score >= 0.35f));
    }
    // ai: t: the node's projected position along the chord between the side's two corners, -1 to 1
    const double cx = (double)px[cnt - 1] - px[0], cy = (double)py[cnt - 1] - py[0], cc = cx * cx + cy * cy;
    if (!(cc > 0)) continue;
    for (int k = 0; k < cnt; k++) t[k] = 2 * (((double)px[k] - px[0]) * cx + ((double)py[k] - py[0]) * cy) / cc - 1;
    double c[4];
    if (!along_fit(t, a, use, cnt, c)) continue;
    // ai: one refit without the marks the first fit calls wrong locks
    double off[OB_MAX_NODES], sorted[OB_MAX_NODES];
    int m = 0;
    for (int k = 1; k < cnt - 1; k++) if (use[k]) { off[k] = fabs(a[k] - (c[0] + t[k] * (c[1] + t[k] * (c[2] + t[k] * c[3])))); sorted[m++] = off[k]; }
    for (int p = 1; p < m; p++) { const double v = sorted[p]; int q = p - 1; while (q >= 0 && sorted[q] > v) { sorted[q + 1] = sorted[q]; q--; } sorted[q + 1] = v; }
    const double med = m ? sorted[m / 2] : 0;
    int dropped = 0;
    for (int k = 1; k < cnt - 1; k++) if (use[k] && off[k] > 0.2 * mod[k] && off[k] > 3 * med) { use[k] = 0; dropped++; }
    if (dropped && !along_fit(t, a, use, cnt, c)) continue;
    for (int k = 1; k < cnt - 1; k++) {
      node_t *nd = &nodes[idx[k]];
      if (!L->node_mark[idx[k]]) continue;
      const double d = c[0] + t[k] * (c[1] + t[k] * (c[2] + t[k] * c[3])) - a[k];
      nd->dx += (float)(d * ux[k]); nd->dy += (float)(d * uy[k]);
    }
  }
}

// mesh = 2: the interior of a code that carries marks on its border only, computed from the border.
// Averaging neighbours inward is wrong for the two errors that matter. A lens's radial distortion,
// once the corners are pinned, is largest along the edges and zero in the middle, which no average
// of edge values reproduces; and rolling-shutter jello is a sideways shift that depends on the
// image row alone. So: (1) fit the border residuals with k |q|^2 q about the image centre plus an
// affine term (the homography has already swallowed part of the distortion), per frame, nothing
// calibrated; (2) carry what the fit leaves inward with a Coons patch, the blend of the four
// border curves, which is exact for any field that is a function of the row plus one of the column.
static void border_fill(const ob_layout_t *L, const homo_t *H, int iw, int ih, node_t *nodes) {
  int nx = L->nx, ny = L->ny;
  float cx = iw * 0.5f, cy = ih * 0.5f, R = 0.5f * sqrtf((float)iw * iw + (float)ih * ih);
  double A[8][9];
  memset(A, 0, sizeof A);
  float *qx = malloc((size_t)nx * ny * sizeof(float)), *qy = malloc((size_t)nx * ny * sizeof(float));
  for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
    float u, v;
    project(H, L->node_x[i], L->node_y[j], &u, &v);
    qx[j * nx + i] = (u - cx) / R; qy[j * nx + i] = (v - cy) / R;
    const node_t *nd = &nodes[j * nx + i];
    int corner = (i == 0 || i == nx - 1) && (j == 0 || j == ny - 1);
    if (!nd->done || !(corner || (L->node_mark[j * nx + i] && nd->score >= 0.35f))) continue;
    double x = qx[j * nx + i], y = qy[j * nx + i], s = x * x + y * y;
    double rx[8] = { s * x, x, y, 0, 0, 1, 0, 0 }, ry[8] = { s * y, 0, 0, x, y, 0, 1, 0 };
    for (int a = 0; a < 8; a++) { for (int b = 0; b < 8; b++) A[a][b] += rx[a] * rx[b] + ry[a] * ry[b]; A[a][8] += rx[a] * nd->dx + ry[a] * nd->dy; }
  }
  for (int a = 0; a < 8; a++) A[a][a] += 1e-9;
  A[7][7] = 1;   // seven unknowns in an eight-row solver
  double th[8] = { 0 };
  if (solve8(A)) for (int a = 0; a < 7; a++) th[a] = A[a][8] / A[a][a];
  #define MODEL_X(k) (float)(th[0] * (qx[k] * qx[k] + qy[k] * qy[k]) * qx[k] + th[1] * qx[k] + th[2] * qy[k] + th[5])
  #define MODEL_Y(k) (float)(th[0] * (qx[k] * qx[k] + qy[k] * qy[k]) * qy[k] + th[3] * qx[k] + th[4] * qy[k] + th[6])
  #define EX(i, j) (nodes[(j) * nx + (i)].dx - MODEL_X((j) * nx + (i)))
  #define EY(i, j) (nodes[(j) * nx + (i)].dy - MODEL_Y((j) * nx + (i)))
  for (int j = 1; j < ny - 1; j++) for (int i = 1; i < nx - 1; i++) {
    node_t *nd = &nodes[j * nx + i];
    if (L->node_mark[j * nx + i]) continue;
    float u = (L->node_x[i] - L->node_x[0]) / (L->node_x[nx - 1] - L->node_x[0]), v = (L->node_y[j] - L->node_y[0]) / (L->node_y[ny - 1] - L->node_y[0]);
    float fx = (1 - u) * EX(0, j) + u * EX(nx - 1, j) + (1 - v) * EX(i, 0) + v * EX(i, ny - 1)
             - ((1 - u) * (1 - v) * EX(0, 0) + u * (1 - v) * EX(nx - 1, 0) + (1 - u) * v * EX(0, ny - 1) + u * v * EX(nx - 1, ny - 1));
    float fy = (1 - u) * EY(0, j) + u * EY(nx - 1, j) + (1 - v) * EY(i, 0) + v * EY(i, ny - 1)
             - ((1 - u) * (1 - v) * EY(0, 0) + u * (1 - v) * EY(nx - 1, 0) + (1 - u) * v * EY(0, ny - 1) + u * v * EY(nx - 1, ny - 1));
    nd->dx = MODEL_X(j * nx + i) + fx; nd->dy = MODEL_Y(j * nx + i) + fy; nd->done = 1; nd->score = 1;
  }
  #undef MODEL_X
  #undef MODEL_Y
  #undef EX
  #undef EY
  free(qx); free(qy);
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
int ob_verify_points_ok;
#endif
typedef struct { float along, across; int rank; } fpt_t;   // rank: which candidate of its scan line this is, counted from the outside
typedef struct { float a, b; int n; float c, mid; } fline_t;   // across = c (along - mid)^2 + a * along + b
// SIDE_K was 4 until a page of clutter round the symbol was measured (scripts/exp/clutter.mjs). A scan line
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

static int edge_ctx_init(edge_ctx_t *c, const uint8_t *grey, int w, int h, int min_con) {
  int bw = (w + BLK - 1) / BLK, bh = (h + BLK - 1) / BLK, bs = bw + 24, chunks = (w + 15) / 16;
  size_t map = (size_t)bs * (bh + 2 * MAP_PAD_Y);
  uint8_t *buf = malloc(2 * map + (size_t)bh * chunks + (size_t)bh * bw + 2 * (size_t)bs);
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

// edge_contrast(...) >= min_con for an edge of scan line `line` (a row, or with axis 1 a column). The test reads
// pixels edge - EDGE_REACH .. edge + EDGE_REACH - 1 of the line at most, so where the blocks under those hold no
// such contrast between them the answer is no, and no pixel is read.
static inline int edge_passes(const edge_ctx_t *c, int axis, int line, int edge, int inner, int dark_len, int light_len) {
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
#ifdef OB_SIMD
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
#ifdef OB_SIMD
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
  uint16_t *active = malloc((size_t)chunks * 2 * sizeof(uint16_t)), *asleep = active + chunks;
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
#ifdef OB_SIMD
static inline v128_t curve_in4(const fline_t *l, v128_t al, v128_t ac, v128_t tol) {
  const v128_t d = wasm_f32x4_sub(al, wasm_f32x4_splat(l->mid));
  const v128_t at = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_splat(l->c), d), d), wasm_f32x4_mul(wasm_f32x4_splat(l->a), al)), wasm_f32x4_splat(l->b));
  return wasm_f32x4_le(wasm_f32x4_abs(wasm_f32x4_sub(ac, at)), tol);
}
#endif
// How many of m points lie within tol of the curve, and the stretch of `along` they cover (lo > hi when none do).
static int curve_support(const fline_t *l, const float *al, const float *ac, int m, float tol, float *plo, float *phi) {
  float lo = 1e9f, hi = -1e9f;
  int n = 0, k = 0;
#ifdef OB_SIMD
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
#ifdef OB_SIMD
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
// whole point of this pass. scripts/exp/noise_bg.mjs: a noisy surround costs the thin frame everything, a quiet zone
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
// of the gap band to the mark's light rim, so it spans [OB_THIN_TRACK, S - 1) and its middle is the two averaged.
// A mark is the same shape wherever it sits, so one size serves both designs.
static inline int mark_side(const ob_layout_t *L) { return L->corner >= 5 ? L->corner : (L->centre >= 5 ? L->centre : L->edge); }
// Every distinct core the layout paints, because a design may carry more than one shape at once: a corner mark
// borrows the border's line and gap for two of its sides, so it loses them once; a mark anywhere else carries its
// own all round and loses them twice. Returns how many.
static int mark_cores(const ob_layout_t *L, int *out) {
  int n = 0;
  const int ring = 2 * OB_THIN_TRACK;
  if (L->corner >= 5) out[n++] = L->corner - 1 - OB_THIN_TRACK;
  if (L->centre >= 5) out[n++] = L->centre - ring;
  if (L->edge >= 5 && (!L->centre || L->edge != L->centre)) out[n++] = L->edge - ring;
  return n;
}
static inline int mark_core(const ob_layout_t *L) {
  return L->corner >= 5 ? L->corner - 1 - OB_THIN_TRACK : (L->centre >= 5 ? L->centre : L->edge) - 2 * OB_THIN_TRACK;
}
static inline float mark_mid(const ob_layout_t *L) { return (OB_THIN_TRACK + L->corner - 1) / 2.0f; }

// line, gap and core in pixels against the ratio OB_THIN_LINE : OB_RING_DEEP : core. `unit` comes back as the
// module size the three of them agree on.
static int mark_ratio(int line, int gap, int core_px, int core, float *unit) {
  const int total = line + gap + core_px;
  if (core_px < MARK_MIN_CORE || total < OB_THIN_LINE + OB_RING_DEEP + core) return 0;
  const float u = (float)total / (OB_THIN_LINE + OB_RING_DEEP + core), tol = 0.6f * u + 0.5f;
  if (fabsf(line - OB_THIN_LINE * u) > tol || fabsf(gap - OB_RING_DEEP * u) > tol) return 0;
  if (fabsf(core_px - core * u) > 1.5f * tol * core / 2) return 0;
  *unit = u;
  return 1;
}

// A light run then a dark run, walked from `at` along `dir`: the gap and the line on one side of a core. This is
// what tells a mark from any other dark square in the picture, and it has to hold on the corner side only.
static int mark_flank(const uint8_t *bin, int len, int stride, int at, int dir, float u) {
  int gap = 0, line = 0, i = at;
  const int cap = (int)((2 + OB_THIN_LINE + OB_RING_DEEP) * u) + 4;
  while (i >= 0 && i < len && !bin[i * stride] && gap < cap) { gap++; i += dir; }
  while (i >= 0 && i < len && bin[i * stride] && line < cap) { line++; i += dir; }
  if (i < 0 || i >= len || bin[i * stride]) return 0;   // the line must end in light, not run off or run on
  const float tol = 0.6f * u + 0.5f;
  return fabsf(gap - OB_RING_DEEP * u) <= tol && fabsf(line - OB_THIN_LINE * u) <= tol;
}

// A candidate core found on a row, confirmed down the column through it: the same extent, within what
// perspective does to a square, and the gap and line on one side of it vertically too.
static int mark_cross(const uint8_t *bin, int w, int h, int cx, int cy, int core, float uh, float *centre, float *uv) {
  const int cap = (int)(3 * core * uh) + 8;
  int up = 0, dn = 0;
  while (cy - up - 1 >= 0 && bin[(cy - up - 1) * w + cx] && up < cap) up++;
  while (cy + dn + 1 < h && bin[(cy + dn + 1) * w + cx] && dn < cap) dn++;
  const int ext = up + dn + 1;
  const float u = (float)ext / core;
  // A screen seen from the side is squeezed along one axis, so the two crossings of one mark can differ a lot.
  // 2.5 is what the finder path allows, about 65 degrees.
  if (u > 2.5f * uh || uh > 2.5f * u) return 0;
  if (!mark_flank(bin + cx, h, w, cy - up - 1, -1, u) && !mark_flank(bin + cx, h, w, cy + dn + 1, 1, u)) return 0;   // + cx: the walk is down THIS column
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
static int mark_cross_merged(const uint8_t *bin, int w, int h, int cx, int cy, int sq, float uh, float *centre, float *uv) {
  const int cap = (int)(3 * sq * uh) + 8;
  int up = 0, dn = 0;
  while (cy - up - 1 >= 0 && bin[(cy - up - 1) * w + cx] && up < cap) up++;
  while (cy + dn + 1 < h && bin[(cy + dn + 1) * w + cx] && dn < cap) dn++;
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
  for (; c < *n; c++) if (fabsf(out[c].x - fx) < u * 2 && fabsf(out[c].y - fy) < u * 2) break;
  if (c < *n) {
    out[c].x = (out[c].x * out[c].hits + fx) / (out[c].hits + 1);
    out[c].y = (out[c].y * out[c].hits + fy) / (out[c].hits + 1);
    out[c].unit = (out[c].unit * out[c].hits + u) / (out[c].hits + 1);
    out[c].hits++;
  } else if (*n < MAX_CAND) out[(*n)++] = (cand_t){ fx, fy, u, 1 };
}
static int mark_candidates(const ob_layout_t *L, const uint8_t *bin, int w, int h, int merged, cand_t *out) {
  int cores[3], ncore = mark_cores(L, cores);
  if (mark_side(L) < 5 || L->corner_filled || !ncore) return 0;
  // Corner marks only: a mark with its ring all round merges into a ring and a core, not into a square.
  const int sq = L->corner >= 5 ? L->corner - 2 : 0;
  if (merged && !sq) return 0;
  int n = 0, *pos = malloc(((size_t)w + 2) * sizeof(int)), minc = merged ? sq : 1 << 20;
  if (!pos) return 0;
  for (int i = 0; !merged && i < ncore; i++) if (cores[i] >= MARK_MIN_CORE && cores[i] < minc) minc = cores[i];
  for (int y = 1; y < h; y += 2) {
    const int np = run_edges(bin + y * w, w, pos);
    int noted = -1;   // the core this row has counted: a ring mark's core lies between two lines and both pairs match it
    for (int k = 1; !merged && 2 * k + 1 < np; k++) {
      const int a0 = pos[2 * k - 2], a1 = pos[2 * k - 1], b0 = pos[2 * k], b1 = pos[2 * k + 1];
      if (a0 == 0 || b1 >= w) continue;            // a run touching either end of the row has no light outside it
      const int first = a1 - a0, gap = b0 - a1, second = b1 - b0;
      for (int sc = 0; sc < 2 * ncore; sc++) {
        const int side = sc & 1, core = cores[sc >> 1];
        if (core < MARK_MIN_CORE) continue;
        // side 0: the corner is to the left, so the line is met first and the core second.
        const int line = side ? second : first, cpx = side ? first : second;
        const int cx0 = side ? a0 : b0, cx1 = side ? a1 : b1;
        float uh, uv, fy;
        if (cx0 == noted) continue;
        if (!mark_ratio(line, gap, cpx, core, &uh)) continue;
        const int cx = (cx0 + cx1) >> 1;
        if (!mark_cross(bin, w, h, cx, y, core, uh, &fy, &uv)) continue;
        mark_note(out, &n, (cx0 + cx1) / 2.0f, fy, 0.5f * (uh + uv));   // the run is [cx0, cx1) in sample()'s coordinates
        noted = cx0;
        break;
      }
    }
    // The merged form: any one dark run with light at both ends that is a square of S - 2 modules down its column.
    for (int k = 0; merged && 2 * k + 1 < np; k++) {
      const int a0 = pos[2 * k], a1 = pos[2 * k + 1], run = a1 - a0;
      if (a0 == 0 || a1 >= w || run < 8 || run > w / 3) continue;
      const float uh = (float)run / sq;
      float uv, fy;
      if (!mark_cross_merged(bin, w, h, (a0 + a1) >> 1, y, sq, uh, &fy, &uv)) continue;
      mark_note(out, &n, (a0 + a1) / 2.0f, fy, 0.5f * (uh + uv));
    }
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
        if (out[c].hits >= need || out[c].y + mark_side(L) * out[c].unit > y) out[m++] = out[c];
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

// The mark scan's own output, for a second implementation of it to be held to. Four floats
// a candidate: x, y, unit, hits. Returns how many, or -1 if the layout paints no marks. Nothing in the decoder
// calls this; mark_rect is the path.
int ob_test_mark_scan(const ob_layout_t *L, const uint8_t *bin, int w, int h, int merged, float *out, int cap) {
  cand_t *c = malloc((size_t)MAX_CAND * sizeof(cand_t));
  if (!c) return -1;
  const int n = mark_candidates(L, bin, w, h, merged, c);
  const int m = n < cap ? n : cap;
  for (int i = 0; i < m; i++) { out[4 * i] = c[i].x; out[4 * i + 1] = c[i].y; out[4 * i + 2] = c[i].unit; out[4 * i + 3] = (float)c[i].hits; }
  free(c);
  return n;
}

// The mark geometry the scan runs on, so a second implementation can build the same tests: the module counts
// mark_ratio matches against, and the band depths it matches them in. Returns how many cores.
int ob_test_mark_cfg(const ob_layout_t *L, int32_t *out) {
  int cores[3];
  const int nc = mark_cores(L, cores);
  out[0] = OB_THIN_LINE; out[1] = OB_RING_DEEP; out[2] = OB_THIN_TRACK; out[3] = MARK_MIN_CORE;
  out[4] = L->corner >= 5 ? L->corner - 2 : 0;   // the merged square's side in modules
  out[5] = mark_side(L);
  out[6] = nc;
  for (int i = 0; i < 3; i++) out[7 + i] = i < nc ? cores[i] : 0;
  return nc;
}

// Four marks chosen the way zxing chooses finder patterns: not by where they are but by whether they AGREE with
// each other (archive/qr-aztec/vendor/zxing-cpp, QRDetector.cpp GenerateFinderPatternSets rejects a triple on size ratio, leg
// ratio, implied module count and angle). The widest convex quad, which is what the finder frame used, picks
// picture texture here: the picture offers dozens of small things with a mark's cross-section, and four of them
// always span more of the image than the marks do.
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

// The mark geometry a harness outside the codec needs to be scale-free the way mark_ratio and mark_quads are
// (archive/opencv-finder/opencv_finder.mjs). Nothing in the decoder calls this. A blob detector measures a mark in PIXELS and has
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
  out[2] = (float)(L->corner >= 5 ? L->corner - 1 - OB_THIN_TRACK : 0);
  out[3] = (float)(L->corner >= 5 ? L->corner - 2 : 0);
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
static int mark_quads(const ob_layout_t *L, const cand_t *all, int nall, int merged, mquad_t *out) {
  const float want = L->w - 2 * mark_form_mid(L, merged);
  cand_t c[MARK_POOL];
  int n = 0, nq = 0;
  #define RANK(k) ((k).hits * (k).unit)
  for (int i = 0; i < nall; i++) {
    int j;
    if (n < MARK_POOL) j = n++;
    else { if (RANK(c[MARK_POOL - 1]) >= RANK(all[i])) continue; j = MARK_POOL - 1; }
    while (j > 0 && RANK(c[j - 1]) < RANK(all[i])) { c[j] = c[j - 1]; j--; }
    c[j] = all[i];
  }
  #undef RANK
  // Whether two candidates can be ADJACENT marks, and how far off the known spacing they are if so. Most pairs
  // cannot, so walking four-cycles of this table visits a few hundred quads where every four of the pool is
  // 10626, each of which used to be sorted round its centroid by atan2 before it could be refused.
  // rat: a side's length over the module size its two ends report, as a share of the known count, or -1 where
  // the two cannot be adjacent marks at all.
  float rat[MARK_POOL][MARK_POOL];
  for (int a = 0; a < n; a++) for (int b = a; b < n; b++) {
    const float dx = c[b].x - c[a].x, dy = c[b].y - c[a].y, mods = sqrtf(dx * dx + dy * dy) / (0.5f * (c[a].unit + c[b].unit));
    const int can = a != b && mods >= 0.6f * want && mods <= 1.5f * want;   // unit shrinks with the same tilt, so a squeezed side reads 2cos / (1 + cos): 0.6 is about 65 degrees
    rat[a][b] = rat[b][a] = can ? mods / want : -1;
  }
  // a is the cycle's lowest index and b < e, so each cycle is met once and not again backwards.
  for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (rat[a][b] >= 0)
    for (int d = a + 1; d < n; d++) if (d != b && rat[b][d] >= 0)
      for (int e = b + 1; e < n; e++) if (e != d && rat[d][e] >= 0 && rat[e][a] >= 0) {
        int id[4] = { a, b, d, e };
        float umin = 1e9f, umax = 0;
        for (int k = 0; k < 4; k++) { umin = fminf(umin, c[id[k]].unit); umax = fmaxf(umax, c[id[k]].unit); }
        if (umax > 2 * umin) continue;   // one symbol, one module size, within what perspective does to it
        int pos = 0, neg = 0;
        for (int k = 0; k < 4; k++) { const float x = cross2(&c[id[k]], &c[id[(k + 1) & 3]], &c[id[(k + 2) & 3]]); pos += x > 0; neg += x < 0; }
        if (pos != 4 && neg != 4) continue;   // convex
        if (neg) { id[1] = e; id[3] = b; }
        // The gapped form reads its module size from a ratio of line, gap and core, which a threshold that shaves
        // dark runs leaves alone, so its sides are scored on their distance from the count. The two square forms
        // read it from a dark square alone, and the binarizer shaves every one of them alike, by 10 to 25% on a
        // 6-module core of 11 to 20 px: four real marks then agree with each other and are off the count
        // TOGETHER (1.12 on every side, traced at 40 degrees of turn), where four picture blobs do not agree. So
        // for those the score is the sides' spread about their own mean plus half the mean's distance from the
        // count, which still keeps a wrong count out. Not for the gapped form: under perspective its four real
        // sides legitimately disagree, and blobs that happen to agree then outrank it (30 degrees of yaw).
        const float r4[4] = { rat[a][b], rat[b][d], rat[d][e], rat[e][a] }, k = 0.25f * (r4[0] + r4[1] + r4[2] + r4[3]);
        const float score = merged ? fabsf(r4[0] - k) + fabsf(r4[1] - k) + fabsf(r4[2] - k) + fabsf(r4[3] - k) + 0.5f * fabsf(k - 1)
                                   : fabsf(r4[0] - 1) + fabsf(r4[1] - 1) + fabsf(r4[2] - 1) + fabsf(r4[3] - 1);
        int j;
        if (nq < MARK_QUADS) j = nq++;
        else { if (out[MARK_QUADS - 1].score <= score) continue; j = MARK_QUADS - 1; }
        while (j > 0 && out[j - 1].score > score) { out[j] = out[j - 1]; j--; }
        for (int k = 0; k < 4; k++) out[j].q[k] = c[id[k]];
        out[j].score = score;
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

// What mark_quads picks from a candidate list somebody else produced. Four floats a
// candidate in, MARK_QUADS quads out, each four candidates then its score. Returns how many quads.
//
// This is the test that matters for a ported scan. Whether two candidate lists are identical is a proxy; whether
// they lead to the same QUAD is the thing, because a candidate that never reaches the pool or never joins a
// four-cycle at the known spacing cannot change any decode however wrong it is.
int ob_test_mark_quads(const ob_layout_t *L, const float *cands, int n, int merged, float *out) {
  cand_t *c = malloc((size_t)(n > 0 ? n : 1) * sizeof(cand_t));
  mquad_t mq[MARK_QUADS];
  if (!c) return -1;
  for (int i = 0; i < n; i++) c[i] = (cand_t){ cands[4 * i], cands[4 * i + 1], cands[4 * i + 2], (int)cands[4 * i + 3] };
  const int nq = mark_quads(L, c, n, merged, mq);
  for (int i = 0; i < nq; i++) {
    for (int k = 0; k < 4; k++) {
      out[i * 17 + 4 * k] = mq[i].q[k].x; out[i * 17 + 4 * k + 1] = mq[i].q[k].y;
      out[i * 17 + 4 * k + 2] = mq[i].q[k].unit; out[i * 17 + 4 * k + 3] = (float)mq[i].q[k].hits;
    }
    out[i * 17 + 16] = mq[i].score;
  }
  free(c);
  return nq;
}

// What a mesh built elsewhere has to be built against: the layout's node lattice, the per-module kind map and which
// nodes carry a mark. Read once a configuration. dims: nx, ny, w, h, thin, corner.
int ob_test_mesh_tables(const ob_layout_t *L, int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark) {
  dims[0] = L->nx; dims[1] = L->ny; dims[2] = L->w; dims[3] = L->h; dims[4] = L->thin; dims[5] = L->corner;
  if (!nodeX) return 1;
  for (int i = 0; i < L->nx; i++) nodeX[i] = L->node_x[i];
  for (int j = 0; j < L->ny; j++) nodeY[j] = L->node_y[j];
  memcpy(kind, L->kind, (size_t)L->w * L->h);
  for (int k = 0; k < L->nx * L->ny; k++) mark[k] = L->node_mark[k];
  return 1;
}
// The same two solves the mesh has to agree with, so a caller driving it from outside does not reimplement them
// and drift: corner_coords gives the lattice corners the quad is measured against, homography the map itself.
void ob_test_corner_coords(const ob_layout_t *L, float *src) { corner_coords(L, src); }
int ob_test_homography(const float *src, const float *dst, float *out) {
  homo_t H;
  if (!homography(src, dst, &H)) return 0;
  memcpy(out, H.h, sizeof H.h);
  return 1;
}

// Four points determine a homography exactly, so every error in them is an error in it. Eight over-determine it,
// which is the difference between trusting one measurement and averaging several: QR v40 carries 46 alignment
// patterns for the same reason LIZARD carries more than its corners.
//
// The correspondences cannot be known before the map is, so this runs after it: the corner quad gives a first H,
// H says where the edge marks must be, and anything found within half a mark of a prediction joins the fit.
// `src` and `dst` receive the layout and image coordinates of everything matched, appended from `have`; the new count comes back.
static int mark_pairs(const ob_layout_t *L, const cand_t *c, int n, const homo_t *H, float *src, float *dst, int have) {
  if (L->edge < 5) return have;
  // The painter's boxes (layout.c thin_corner): [cw, cw + S) along the side and [0, S) in from it, core in the middle.
  const int S = L->edge, cw = (L->w - S + 1) / 2, ch = (L->h - S + 1) / 2;
  const float m = S / 2.0f;
  const float lay8[8] = { cw + m, m, L->w - m, ch + m, cw + m, L->h - m, m, ch + m };
  for (int k = 0; k < 4 && have < 8; k++) {
    float px, py;
    const float lay[2] = { lay8[2 * k], lay8[2 * k + 1] };
    project(H, lay[0], lay[1], &px, &py);
    int best = -1;
    float bd = 1e9f;
    for (int i = 0; i < n; i++) {
      const float dx = c[i].x - px, dy = c[i].y - py, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = i; }
    }
    // Within half a mark of where it was predicted, or it is not that mark.
    if (best < 0 || bd > 0.25f * L->edge * L->edge * c[best].unit * c[best].unit) continue;
    src[2 * have] = lay[0]; src[2 * have + 1] = lay[1];
    dst[2 * have] = c[best].x; dst[2 * have + 1] = c[best].y;
    have++;
  }
  return have;
}

// Least squares over `n` correspondences, through the same normal equations solve8 already does for four.
static int homography_n(const float *src, const float *dst, int n, homo_t *H) {
  double N[8][9] = { { 0 } };
  for (int i = 0; i < n; i++) {
    const double x = src[2 * i], y = src[2 * i + 1], u = dst[2 * i], v = dst[2 * i + 1];
    const double r0[9] = { x, y, 1, 0, 0, 0, -u * x, -u * y, u }, r1[9] = { 0, 0, 0, x, y, 1, -v * x, -v * y, v };
    for (int a = 0; a < 8; a++) for (int b = 0; b < 9; b++) N[a][b] += r0[a] * r0[b] + r1[a] * r1[b];
  }
  if (!solve8(N)) return 0;
  for (int i = 0; i < 8; i++) H->h[i] = (float)(N[i][8] / N[i][i]);
  H->h[8] = 1;
  return 1;
}

// Every mark of one form in a binarized image at 1 / scale of the capture, in the capture's own coordinates. The
// candidates depend on the mark's design alone, never on how many modules the symbol has, so one scan serves every
// layout a receiver is looking for.
static int mark_scan(const ob_layout_t *L, const uint8_t *bin, int bw, int bh, int scale, int merged, float *best_mark, cand_t *c) {
  const int n = mark_candidates(L, bin, bw, bh, merged, c);
  for (int k = 0; k < n; k++) { c[k].x *= scale; c[k].y *= scale; c[k].unit *= scale; }
  // The best-supported mark, whatever the design: four at the corners give a quad, one in the middle does not.
  // It is a centre and a module size, which says where the symbol is and how big, not how it is turned (layout.h).
  // A gapped mark is believed over any merged one: nothing else in a capture has its cross-section. A scan of a
  // shrunk image counts fewer rows, so it only speaks when the full one found nothing.
  if (scale == 1 ? !merged || best_mark[2] <= 0 : best_mark[2] <= 0) { int bh2 = 0; for (int k = 0; k < n; k++) if (c[k].hits > bh2) { bh2 = c[k].hits; best_mark[0] = c[k].x; best_mark[1] = c[k].y; best_mark[2] = c[k].unit; } }
  return n;
}

// The mark centres of one form as quads, best spacing first, and the rectangle they make in layout modules.
static int mark_rect(const ob_layout_t *L, const cand_t *c, int n, int iw, int ih, int merged, float *src, mquad_t *quads) {
  if (L->corner < 5 || n < 4) return 0;
  const float m = mark_form_mid(L, merged), s4[8] = { m, m, L->w - m, m, L->w - m, L->h - m, m, L->h - m };
  memcpy(src, s4, sizeof s4);
  int nq = mark_quads(L, c, n, merged, quads), keep = 0;
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
// five times the draws, was tried and does NOT get it back (scripts/exp/noise_bg.mjs, unchanged in every
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
  for (int rank = 0; rank < SIDE_K; rank++) {
    int m = 0;
    for (int k = 0; k < n; k++) if (p[k].rank == rank) idx[m++] = k;
    if (m < 12) continue;
    fline_t best = { 0, 0, 0, 0, 0 };
    for (int it = 0; it < 120; it++) {
      int i0 = idx[RND(m)], i1 = idx[RND(m)], i2 = idx[RND(m)];
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
static void thin_point(const homo_t *H, const fline_t *const *cv, int m, float x, float y, float xs, float ys, float *u, float *v) {
  float lu, lv;
  project(H, x, y, u, v);
  if (!cv) return;   // a quad from the corner marks has no fitted sides, so there is no bow to take out: H alone
  project(H, xs, ys, &lu, &lv);
  if (m & 1) *u += line_at(cv[m], lv) - lu; else *v += line_at(cv[m], lu) - lv;
}

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

// Grey levels of the solid line and of the light line inside it, and the threshold from the track pairs where
// `map` puts them. The line and the gap are the same for every rotation and mirror image of a quad, so the contrast
// alone rejects a quad (a bezel is dark where the light line should be).
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

// How well the coded track reads as the layout says, in -0.5 .. 0.5. It validates a quad and picks
// its rotation in one measurement.
// beat: the caller keeps a quad only if it scores above this, and most of the up to 81 line combinations are
// nowhere near. A point adds at most 0.5, so once the points left cannot lift the sum past beat the rest are not
// read and beat itself is returned. A quad that is read to the end gets the same sum in the same order as ever.
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
        score += ob_thin_track(side, j + l, L->track_alt) ? Q[l] : -Q[l];
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
    score += ob_thin_track(side, j, L->track_alt) ? q : -q;
    if (!(++n & 7) && score + 0.5f * (total - n) < need) return beat;
  }
  return score / n;
}

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
      signs[total] = ob_thin_track(side, j, L->track_alt) ? 1 : -1;
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
static uint8_t *shrink2(const uint8_t *src, int w, int h) {
  const int sw = w / 2, sh = h / 2;
  uint8_t *out = malloc((size_t)sw * sh);
  if (!out) return NULL;
  for (int y = 0; y < sh; y++) {
    const uint8_t *a = src + (size_t)2 * y * w, *b = a + w;
    uint8_t *o = out + (size_t)y * sw;
    for (int x = 0; x < sw; x++) o[x] = (uint8_t)((a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) >> 2);
  }
  return out;
}

// A screen is seen mirrored only through a mirror, so the mirror images (hypotheses 4 to 7) are a last resort, not
// tried once a rotation holds the best (holds_in, below). A mirror image holding it does not stop them: that is a
// capture through a mirror, and its other quads are read the same way. rotation_holds is the flat 0.15 hold that
// mark_crop keeps.
static inline int rotation_holds(float best, int orient) { return best >= 0.15f && orient < 4; }
// The hold at equal significance on every ring (2026-10-04): 0.15 is 3.4 sd of a
// texture read (a sign test, sd 0.5 / sqrt(cells)) over the 64 ring's 128 track cells, so a ring with fewer cells holds
// at the same sd: 0.212 on the 32 ring's 64, 0.15 on the others. At a flat 0.15, mark-like squares in a picture
// settled a false 32-ring quad before the true ring was scored (8 of 2,000 clean frames at LIZARD-208).
static inline int holds_in(float best, int orient, const ob_layout_t *L) {
  const int cells = 4 * ob_track_cells(L->w, L->reserve);
  return orient < 4 && best >= 0.15f * fmaxf(1.0f, sqrtf(128.0f / (float)cells));
}

// Ls: the layouts the symbol may be, which differ only in how many modules a side (focus.h FOCUS_RING); *which
// comes back as the one that registered. The mark path scores every one before a hold settles it (holds_in); the
// line path tries the first first and the others only when it does not register.
static int find_frame(const ob_layout_t *const *Ls, int nL, const image_t *im, const uint8_t *bin, float *quad, int *orient, float *score_out, float *best_mark, int *which) {
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
  // near marks were not found at all. The finder frame has always been searched this way (acquire_core); the
  // marks were not, because at 286 modules a core only reaches the window at very close range.
  float msrc[8];
  cand_t *mcand = malloc((size_t)MAX_CAND * sizeof(cand_t));
  mquad_t mq[MARK_QUADS];
  uint8_t *grey2 = NULL, *lbin = NULL;
  const uint8_t *grey = im->px;
  *which = 0;
  for (int k = 0; k < 8; k++) ob_finder_info[k] = k == 5 || k == 7 ? 0 : -1;
  for (int scale = 1; mcand && scale <= 4 && !holds_in(best, *orient, Ls[*which]); scale *= 2) {
    const uint8_t *sbin = bin;
    const int sw = w / scale, sh = h / scale;
    if (scale > 1) {
      if (sw < 64 || sh < 64) break;
      uint8_t *next = shrink2(grey, w / (scale / 2), h / (scale / 2));
      if (!next) break;
      free(grey2); grey2 = next; grey = next;
      free(lbin);
      if (!(lbin = malloc((size_t)sw * sh))) break;
      PROF(PROF_BINARIZE, binarize(grey, sw, sh, lbin));
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
    for (int form = 0; form < 3 && !holds_in(best, *orient, Ls[*which]); form++) {
    if (form < 2) PROF(PROF_FF_MARKS, nmark = mark_scan(Ls[0], sbin, sw, sh, scale, form, best_mark, mcand));
    else {
      const int core = Ls[0]->corner - 1 - OB_THIN_TRACK, sq = Ls[0]->corner - 2;
      if (core < MARK_MIN_CORE) break;
      for (int k = 0; k < nmark; k++) mcand[k].unit *= (float)sq / core;
    }
    for (int li = 0; li < nL; li++) {
    const ob_layout_t *L = Ls[li];
    const int nq = mark_rect(L, mcand, nmark, w, h, form, msrc, mq);
    if (form == 0 && scale == 1 && li == 0) ob_finder_info[7] = (float)nq;
    for (int h0 = 0; h0 < 8 && !(h0 == 4 && holds_in(best, *orient, Ls[*which])); h0 += 4)
    for (int qi = 0; qi < nq; qi++) for (int hyp = h0; hyp < h0 + 4; hyp++) {
      int rot = hyp & 3, mir = hyp >> 2, map[4];
      float d2[8], ref = 0, con = 0, nc[8];
      homo_t H;
      for (int k = 0; k < 4; k++) { int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = mq[qi].q[from].x; d2[2 * k + 1] = mq[qi].q[from].y; }
      if (!homography(msrc, d2, &H)) continue;
      // The corner quad places the edge marks; whatever is found where one was predicted joins the fit, and the
      // map stops being four points exactly solved and becomes as many as were seen, least squares.
      float s8[16], d8[16];
      memcpy(s8, msrc, sizeof msrc);
      memcpy(d8, d2, sizeof d2);
      const int np = form ? 4 : mark_pairs(L, mcand, nmark, &H, s8, d8, 4);
      if (np > 4) homography_n(s8, d8, np, &H);
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
  free(mcand); free(grey2); free(lbin);
  ob_finder_info[5] = holds_in(best, *orient, Ls[*which]); ob_finder_info[6] = best;
  if (holds_in(best, *orient, Ls[*which])) ok = 0;
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
  if (edge_ctx_init(&ctx, im->px, w, h, min_con)) { for (int s = 0; s < 4; s++) free(pts[s]); return 0; }
  row_points(bin, w, h, &ctx, pts[3], &n[3], pts[1], &n[1]);
  ob_prof_ms[PROF_FF_POINTS] += ob_now_ms() - tp; tp = ob_now_ms();
  col_points(bin, w, h, &ctx, pts[0], &n[0], pts[2], &n[2]);
  ob_prof_ms[PROF_FF_TURN] += ob_now_ms() - tp; tp = ob_now_ms();
  free(ctx.buf);
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
    ob_verify_points_ok++;
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
  for (int li = 0; ok && li < nL && !holds_in(best, *orient, Ls[*which]); li++) {
  const ob_layout_t *L = Ls[li];
  // One reading serves every way round on a square, where a turn or a flip lands each side on another's.
  const int share = L->w == L->h && !(L->w & 1) && L->w > 16, npts = share ? ob_track_cells(L->w, L->reserve) : 0;
  float *track_s = share ? malloc((size_t)8 * npts * sizeof(float)) : NULL;
  int8_t *track_sign = share ? malloc((size_t)4 * npts) : NULL;
  for (int sd = 0; sd < 4 && share; sd++) for (int i = 0; i < npts; i++) track_sign[sd * npts + i] = (int8_t)ob_thin_track(sd, i, L->track_alt);
  // Two passes, one per depth the curves may run at (thin_edge_depth). Blur is the same all round the symbol, so
  // the four sides share one state and this is two hypotheses, not sixteen. The second runs only when the first
  // settled nothing, so a capture sharp enough for the gap to survive pays for one.
  //
  // Every combination is scored under every rotation, and the best of them all wins. Until 2026-09-24 the first
  // combination to reach 0.15 fixed the rotation and the rest were scored under it alone, so a wrong combination
  // read passably one way round shut the true one out. One read serves all eight ways round, so this costs reading
  // each side both ways, not eight reads: on the line path a decode went from 2.7 to 2.9 ms at n = 256 and from
  // 15.5 to 15.8 at 1024 (an ideal image, mark cores painted out).
  for (int pass = 0; pass < 2 && !(pass && holds_in(best, *orient, Ls[*which])); pass++) {
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
    if (share && (!homography(src, dst, &H0) || !(con = thin_read_both(L, im, &H0, cv, track_s, pair, npts, &nref)))) continue;
    for (int hyp = 0; hyp < 8 && !(hyp == 4 && holds_in(best, *orient, Ls[*which])); hyp++) {
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
      if (share) {
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
      for (int k = 0; k < 4; k++) { int from = (rot + (mir ? 4 - k : k)) & 3; d2[2 * k] = dst[2 * from]; d2[2 * k + 1] = dst[2 * from + 1]; }
      if (!homography(src, d2, &H)) continue;
      if (!share) {
        float ref;
        if (!thin_levels(L, im, &H, cv, map, &ref, &con)) continue;
        sc = thin_score(L, im, &H, cv, map, ref, con, best);
        if (sc <= best) continue;
      }
      best = sc; *orient = hyp; *which = li;
      ob_finder_info[0] = best; ob_finder_info[1] = (float)hyp; ob_finder_info[4] = 1;
      // Hand back the lattice's own corners, which sit half a module further in than the line's middle.
      float nc[8];
      corner_coords(L, nc);
      for (int k = 0; k < 4; k++) project(&H, nc[2 * k], nc[2 * k + 1], &quad[2 * k], &quad[2 * k + 1]);
    }
  }
  }
  free(track_s); free(track_sign);
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
static uint8_t *border_blur(const ob_layout_t *L, const homo_t *H, const uint8_t *img, int iw, int ih, int r) {
  uint8_t *out = malloc((size_t)iw * ih);
  if (!out) return NULL;
  memcpy(out, img, (size_t)iw * ih);
  const float D = 16, Mo = 3, W = (float)L->w, Hh = (float)L->h;
  const float band[4][4] = { { -Mo, -Mo, W + Mo, D }, { -Mo, Hh - D, W + Mo, Hh + Mo }, { -Mo, -Mo, D, Hh + Mo }, { W - D, -Mo, W + Mo, Hh + Mo } };
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
    uint16_t *tmp = malloc((size_t)(x1 - x0) * (y1 - y0 + 2 * r) * sizeof(uint16_t));
    int *acc = malloc((size_t)(x1 - x0) * sizeof(int));
    if (tmp && acc) box_blur_rect(img, out, iw, ih, r, x0, y0, x1, y1, tmp, acc);
    free(tmp); free(acc);
  }
  return out;
}

void ob_reg_free(ob_reg_t *reg) { free(reg->nodes); reg->nodes = NULL; }

// Look again inside what one mark says the symbol is, for the case the first look found nothing.
//
// A scan starts at the frame's edge and stops after SIDE_K candidates, so whatever lies between the edge and the
// symbol spends them: scripts/exp/noise_bg.mjs measured a noisy surround taking the whole payload, and
// archive/build-scratch-2026-09/crop_ceiling.mjs measured a crop to within a few percent of the symbol giving all of it back. One mark
// gives a centre and a module size, which is that crop to about a tenth of the symbol, and a tenth is enough
// (archive/build-scratch-2026-09/mark_race.mjs: a 2% margin buys nothing, a 10% margin buys everything).
//
// A CORNER mark does not say which corner it is, so all four are tried. That is up to four more passes, and they
// only ever run where the answer was otherwise going to be nothing. The grey is copied and binarized again
// rather than the first binarization being cropped, because the threshold is local and the noise that is now
// outside the crop was part of what set it.
// With several layouts each is tried in turn, since the box a mark implies is its layout's size.
// The best crop wins, and only a rotation at a flat 0.15 ends the search (rotation_holds). find_frame's hold by ring
// (holds_in) was tried here on 2026-10-04 and read no frame better: on 1,600 made hard captures it acted on 3, changed
// none of them (a 32-ring crop at 0.16 to 0.17, no word either way) and made their registration 2.5 to 3 times as
// slow. The first crop to reach 0.06 won until 2026-09-24, and once the line path read all eight ways round a wrong
// box's 0.07 came before the right box's 0.25 (LIZARD-16, far 0.8 px a module, scripts/exp/fmt_word.mjs DAMAGE frame
// 1). Still at most 4 crops a layout.
static int mark_crop(const ob_layout_t *const *Ls, int nL, const image_t *im, float *quad, int *orient, float *sc, const float *mark, int *which) {
  uint8_t *cg = NULL, *cb = NULL;
  int ok = 0;
  float best = -1, info[8];
  for (int li = 0; li < nL && !rotation_holds(best, *orient); li++) {
  const ob_layout_t *L = Ls[li];
  const float u = mark[2], side = L->w * u, in = side / 2 - mark_mid(L) * u, half = side * 0.55f;
  static const int SX[4] = { -1, 1, 1, -1 }, SY[4] = { -1, -1, 1, 1 };
  if (!(u > 0) || side < 64 || side > 4.0f * (im->w > im->h ? im->w : im->h)) continue;
  for (int k = 0; k < 4 && !rotation_holds(best, *orient); k++) {
    // A centre mark is its own answer; a corner mark is one of four, so step to where each would put the middle.
    const float cx = mark[0] + (L->corner >= 5 ? SX[k] * in : 0), cy = mark[1] + (L->corner >= 5 ? SY[k] * in : 0);
    int x0 = (int)(cx - half), y0 = (int)(cy - half), x1 = (int)(cx + half), y1 = (int)(cy + half);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > im->w) x1 = im->w;
    if (y1 > im->h) y1 = im->h;
    const int cw = x1 - x0, ch = y1 - y0;
    if (cw < 64 || ch < 64 || (cw >= im->w && ch >= im->h)) continue;   // nothing was cropped: the same look again
    if (!cg && (!(cg = malloc((size_t)im->w * im->h)) || !(cb = malloc((size_t)im->w * im->h)))) break;
    for (int y = 0; y < ch; y++) memcpy(cg + (size_t)y * cw, im->px + (size_t)(y0 + y) * im->w + x0, (size_t)cw);
    binarize(cg, cw, ch, cb);
    image_t sub = *im;
    sub.px = cg; sub.w = cw; sub.h = ch;
    float ignored[3] = { 0, 0, 0 }, q[8], s;
    int w1, o;
    if (find_frame(&Ls[li], 1, &sub, cb, q, &o, &s, ignored, &w1) && s > best) {
      for (int i = 0; i < 4; i++) { quad[2 * i] = q[2 * i] + x0; quad[2 * i + 1] = q[2 * i + 1] + y0; }
      best = *sc = s; *orient = o; ok = 1; *which = li;
      memcpy(info, ob_finder_info, sizeof info);
    }
    if (L->corner < 5) break;   // a centre mark has only the one box to try
  }
  }
  if (ok) memcpy(ob_finder_info, info, sizeof info);   // the record of the crop that won, not of the last one tried
  free(cg); free(cb);
  return ok;
}

// in_quad: a quad found somewhere else, in image pixels, with its orientation and track score. The scan, the
// line fit and the track scoring are then all skipped and everything from the homography onwards is shared, so
// a quad found on a GPU becomes a registration on exactly the terms one found here does.
// Ls: the layouts the symbol may be (find_frame); the one that registered comes back in *which and is the one the
// registration is built on. An external quad is the first layout's.
static int acquire_core(const ob_layout_t *const *Ls, int nL, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res,
                        const float *in_quad, int in_orient, float in_score, int *which) {
  const ob_layout_t *L = Ls[0];
  *which = 0;
  const image_t im = *pim;
  const uint8_t *img = im.px;
  int iw = im.w, ih = im.h;
  reg->nodes = NULL;
  // Full size first, then halved twice: the local-mean window is 40 px, so a finder whose dark
  // centre outgrows it (a close or small symbol) only resolves once the image is shrunk.
  static cand_t cand[MAX_CAND];
  cand_t q[4];
  uint8_t *bin = in_quad ? NULL : malloc((size_t)iw * ih), *small = NULL;
  int got = 0, thin_orient = -1;
  float thin_quad[8], thin_sc = 0;
  if (in_quad) { memcpy(thin_quad, in_quad, sizeof thin_quad); thin_orient = in_orient; thin_sc = in_score; got = 1; }
  else if (L->thin) {
    PROF(PROF_BINARIZE, binarize(img, iw, ih, bin));
    PROF(PROF_FINDERS, got = find_frame(Ls, nL, &im, bin, thin_quad, &thin_orient, &thin_sc, res->mark, which));
    if (!got && res->mark[2] > 0) { PROF(PROF_FINDERS, got = mark_crop(Ls, nL, &im, thin_quad, &thin_orient, &thin_sc, res->mark, which)); if (got) ob_finder_info[4] = 2; }
    L = Ls[*which];
  }
  for (int scale = 1; !in_quad && !L->thin && scale <= 4 && !got; scale *= 2) {
    int sw = iw / scale, sh = ih / scale;
    const uint8_t *src_img = img;
    double td = ob_now_ms();
    if (scale > 1) {
      if (!small) small = malloc((size_t)(iw / 2) * (ih / 2));
      for (int y = 0; y < sh; y++) for (int x = 0; x < sw; x++) {
        int a = 0;
        for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) a += img[(y * scale + dy) * iw + x * scale + dx];
        small[y * sw + x] = (uint8_t)(a / (scale * scale));
      }
      src_img = small;
    }
    ob_prof_ms[PROF_BINARIZE] += ob_now_ms() - td;
    PROF(PROF_BINARIZE, binarize(src_img, sw, sh, bin));
    int nc;
    PROF(PROF_FINDERS, nc = find_candidates(bin, sw, sh, cand); got = pick_quad(cand, nc, q));
    res->finders = nc;
    for (int k = 0; got && k < 4; k++) { q[k].x *= scale; q[k].y *= scale; q[k].unit *= scale; }
  }
  free(bin); free(small);
  if (!got) return 0;

  double tm = ob_now_ms();
  float src[8], dst[8];
  corner_coords(L, src);
  homo_t H, best_h;
  float best = -1e9f;
  int best_o = -1;
  // The thin frame's finder has already settled rotation: it scores the coded track under every hypothesis.
  if (L->thin && homography(src, thin_quad, &best_h)) { best = 2.0f + thin_sc; best_o = thin_orient; memcpy(res->quad, thin_quad, sizeof thin_quad); }
  for (int hyp = 0; !L->thin && hyp < 8; hyp++) {
    int rot = hyp & 3, mir = hyp >> 2;
    for (int k = 0; k < 4; k++) { const cand_t *c = &q[(rot + (mir ? 4 - k : k)) & 3]; dst[2 * k] = c->x; dst[2 * k + 1] = c->y; }
    // A rectangular symbol only fits the rotations that keep its long side on the long side.
    if (!homography(src, dst, &H)) continue;
    float s = orient_score(L, &im, &H);
    if (s > best) { best = s; best_o = hyp; best_h = H; memcpy(res->quad, dst, sizeof dst); }
  }
  if (best_o < 0 || best < 2.0f) return 0;
  res->orient = best_o;
  memcpy(ob_raw_quad, res->quad, sizeof ob_raw_quad);
  H = best_h;

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
  if (L->thin && mpx / 2 >= 1 && (blurred = border_blur(L, &best_h, img, iw, ih, (int)(mpx / 2)))) rim.px = blurred;
  // Finders first, so the homography the marks are measured against is as good as four
  // points can make it.
  int nn = L->nx * L->ny;
  node_t *nodes = calloc((size_t)nn, sizeof(node_t));
  for (int c = 0; c < 4; c++) {
    int i = (c == 1 || c == 2) ? L->nx - 1 : 0, j = c >= 2 ? L->ny - 1 : 0;
    refine_node(L, &rim, &H, i, j, nodes);
    node_t *nd = &nodes[j * L->nx + i];
    res->quad[2 * c] += nd->dx; res->quad[2 * c + 1] += nd->dy;
    nd->done = 0;
  }
  homography(src, res->quad, &H);
  for (int c = 0; c < 4; c++) { node_t *nd = &nodes[((c >= 2) ? L->ny - 1 : 0) * L->nx + ((c == 1 || c == 2) ? L->nx - 1 : 0)]; nd->dx = nd->dy = 0; nd->done = 1; }
  // Rings outward from the corners, so each mark starts from measured neighbours.
  for (int ring = 1; mesh && ring < L->nx + L->ny; ring++) for (int j = 0; j < L->ny; j++) for (int i = 0; i < L->nx; i++) {
    int di = i < L->nx - 1 - i ? i : L->nx - 1 - i, dj = j < L->ny - 1 - j ? j : L->ny - 1 - j;
    if (di + dj != ring || nodes[j * L->nx + i].done) continue;
    if (!L->node_mark[j * L->nx + i]) continue;
    refine_node(L, &rim, &H, i, j, nodes);
  }
  free(blurred);
  if (mesh == 2 || (mesh && L->thin)) { smooth_along(L, &H, nodes); border_fill(L, &H, iw, ih, nodes); }   // a thin frame has nothing but its border
  float msum = 0;
  int mcount = 0;
  for (int k = 0; k < nn; k++) if (nodes[k].done) { msum += nodes[k].score; mcount++; }
  res->mark_score = mesh ? msum / (mcount ? mcount : 1) : 0;
  res->found = 1;
  reg->H = H; reg->nodes = nodes;
  ob_prof_ms[PROF_MESH] += ob_now_ms() - tm;
  return 1;
}

int ob_acquire(const ob_layout_t *L, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res) {
  int which;
  return acquire_core(&L, 1, pim, mesh, reg, res, NULL, 0, 0, &which);
}
int ob_acquire_any(const ob_layout_t *const *Ls, int nL, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res, int *which) {
  return acquire_core(Ls, nL, pim, mesh, reg, res, NULL, 0, 0, which);
}
int ob_acquire_quad(const ob_layout_t *L, const image_t *pim, int mesh, ob_reg_t *reg, ob_result_t *res,
                    const float *quad, int orient, float score) {
  int which;
  return acquire_core(&L, 1, pim, mesh, reg, res, quad, orient, score, &which);
}

void ob_map_point(const ob_layout_t *L, const ob_reg_t *reg, float mx, float my, float *pu, float *pv) {
  int i = 0, j = 0;
  while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
  while (i < L->nx - 2 && mx >= L->node_x[i + 1]) i++;
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
// to a vector. The binary code's cells are the grid with origin 0.5 and step 1; FOCUS's picture is a finer one,
// taken a strip of columns at a time.
void ob_sample_grid(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch) {
  const image_t im = *pim;
  const homo_t H = reg->H;
  const node_t *nodes = reg->nodes;
  // Which mesh column a point falls in depends on its column alone, so the runs are found once, not once a row,
  // and point by point with the comparison the scalar walk makes, so no point can land in a different column.
  int run_i[OB_MAX_NODES + 1], run_end[OB_MAX_NODES + 1], runs = 0;
  for (int xx = 0, i = 0; xx < nx; runs++) {
    while (i < L->nx - 2 && mx0 + (float)(x0 + xx) * step >= L->node_x[i + 1]) i++;
    int e = xx + 1;
    while (e < nx && !(i < L->nx - 2 && mx0 + (float)(x0 + e) * step >= L->node_x[i + 1])) e++;
    run_i[runs] = i; run_end[runs] = e; xx = e;
  }
  for (int yy = 0; yy < ny; yy++) {
    float my = my0 + (float)(y0 + yy) * step;
    int j = 0;
    while (j < L->ny - 2 && my >= L->node_y[j + 1]) j++;
    float v = (my - L->node_y[j]) / (L->node_y[j + 1] - L->node_y[j]);
#ifdef OB_SIMD
    const v128_t one = wasm_f32x4_splat(1), four = wasm_f32x4_splat(4), vv = wasm_f32x4_splat(v), vw = wasm_f32x4_splat(1 - v);
    const v128_t h0 = wasm_f32x4_splat(H.h[0]), h1y = wasm_f32x4_splat(H.h[1] * my), h2 = wasm_f32x4_splat(H.h[2]), h3 = wasm_f32x4_splat(H.h[3]), h4y = wasm_f32x4_splat(H.h[4] * my), h5 = wasm_f32x4_splat(H.h[5]), h6 = wasm_f32x4_splat(H.h[6]), h7y = wasm_f32x4_splat(H.h[7] * my);
    const v128_t vmx0 = wasm_f32x4_splat(mx0), vstep = wasm_f32x4_splat(step);
#endif
    for (int r = 0, xx = 0; r < runs; r++) {
      const int i = run_i[r], end = run_end[r];
      const node_t *n00 = &nodes[j * L->nx + i], *n10 = n00 + 1, *n01 = n00 + L->nx, *n11 = n01 + 1;
#ifdef OB_SIMD
      const v128_t nx0 = wasm_f32x4_splat(L->node_x[i]), nxs = wasm_f32x4_splat(L->node_x[i + 1] - L->node_x[i]);
      const v128_t d00 = wasm_f32x4_splat(n00->dx), d10 = wasm_f32x4_splat(n10->dx), d01 = wasm_f32x4_splat(n01->dx), d11 = wasm_f32x4_splat(n11->dx);
      const v128_t e00 = wasm_f32x4_splat(n00->dy), e10 = wasm_f32x4_splat(n10->dy), e01 = wasm_f32x4_splat(n01->dy), e11 = wasm_f32x4_splat(n11->dy);
      // The column index as floats, stepped by four: whole numbers, so exact.
      v128_t vgx = wasm_f32x4_make((float)(x0 + xx), (float)(x0 + xx + 1), (float)(x0 + xx + 2), (float)(x0 + xx + 3));
      for (; xx < end; xx += 4, vgx = wasm_f32x4_add(vgx, four)) {
        const v128_t vmx = wasm_f32x4_add(vmx0, wasm_f32x4_mul(vgx, vstep));
        const v128_t u = wasm_f32x4_div(wasm_f32x4_sub(vmx, nx0), nxs), iu = wasm_f32x4_sub(one, u);
        const v128_t dx = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(d00, iu), wasm_f32x4_mul(d10, u)), vw), wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(d01, iu), wasm_f32x4_mul(d11, u)), vv));
        const v128_t dy = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(e00, iu), wasm_f32x4_mul(e10, u)), vw), wasm_f32x4_mul(wasm_f32x4_add(wasm_f32x4_mul(e01, iu), wasm_f32x4_mul(e11, u)), vv));
        const v128_t pw = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(h6, vmx), h7y), one);
        const v128_t pu = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(h0, vmx), h1y), h2), pw), pv = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(h3, vmx), h4y), h5), pw);
        const v128_t got = sample4(&im, wasm_f32x4_add(pu, dx), wasm_f32x4_add(pv, dy));
        if (xx + 4 <= end) { wasm_v128_store(out + yy * pitch + xx, got); continue; }
        // The run's last vector reaches past its end. Those lanes were worked out with this run's nodes, which are
        // not theirs, so they are dropped: a whole vector and a copy is still cheaper than scalar points.
        float t[4];
        wasm_v128_store(t, got);
        for (int l = 0; xx + l < end; l++) out[yy * pitch + xx + l] = t[l];
      }
      xx = end;
#else
      for (; xx < end; xx++) {
        float mx = mx0 + (float)(x0 + xx) * step, u = (mx - L->node_x[i]) / (L->node_x[i + 1] - L->node_x[i]);
        float dx = (n00->dx * (1 - u) + n10->dx * u) * (1 - v) + (n01->dx * (1 - u) + n11->dx * u) * v;
        float dy = (n00->dy * (1 - u) + n10->dy * u) * (1 - v) + (n01->dy * (1 - u) + n11->dy * u) * v;
        float pu, pv;
        project(&H, mx, my, &pu, &pv);
        out[yy * pitch + xx] = sample(&im, pu + dx, pv + dy);
      }
#endif
    }
  }
}

void ob_sample_cells(const ob_layout_t *L, const image_t *pim, const ob_reg_t *reg, float *s) { ob_sample_grid(L, pim, reg, 0.5f, 0.5f, 1.0f, 0, L->w, 0, L->h, s, L->w); }

// The line anchor's scan on its own, for the first WebGPU port (since deleted) to be held to candidate for
// candidate. No layout: the scan is geometry-free, it only needs the grey image and
// its binarization. Sides in find_frame's order (0 top, 1 right, 2 bottom, 3 left), three floats a candidate:
// along, across, rank. n takes the four counts and then min_con, which find_frame works out here and the port
// has to work out for itself.
int ob_test_line_points(const uint8_t *grey, const uint8_t *bin, int w, int h, float *out, int32_t *n, int cap) {
  int hist[256] = { 0 }, cnt = 0, p5 = 0, p95 = 255, acc = 0;
  for (int y = 4; y < h; y += 8) for (int x = 4; x < w; x += 8) { hist[grey[(size_t)y * w + x]]++; cnt++; }
  for (int v = 0; v < 256; v++) { acc += hist[v]; if (acc * 20 < cnt) p5 = v; if (acc * 20 < cnt * 19) p95 = v; }
  const int min_con = (p95 - p5) / 4 < 12 ? 12 : (p95 - p5) / 4;
  const int big = w > h ? w : h;
  fpt_t *pts[4] = { 0 };
  int m[4] = { 0 }, at = 0, ok = 1;
  edge_ctx_t ctx;
  for (int s = 0; s < 4; s++) if (!(pts[s] = malloc((size_t)(big / 2 + 1) * SIDE_K * sizeof(fpt_t)))) ok = 0;
  if (!ok || edge_ctx_init(&ctx, grey, w, h, min_con)) { for (int s = 0; s < 4; s++) free(pts[s]); return -1; }
  row_points(bin, w, h, &ctx, pts[3], &m[3], pts[1], &m[1]);
  col_points(bin, w, h, &ctx, pts[0], &m[0], pts[2], &m[2]);
  free(ctx.buf);
  for (int s = 0; s < 4; s++) {
    n[s] = m[s];
    for (int k = 0; k < m[s] && at < cap; k++, at++) {
      out[3 * at] = pts[s][k].along; out[3 * at + 1] = pts[s][k].across; out[3 * at + 2] = (float)pts[s][k].rank;
    }
    free(pts[s]);
  }
  n[4] = min_con;
  return at;
}
