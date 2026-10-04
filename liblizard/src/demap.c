// Grey samples to per-cell LLRs: local normalization, a small linear equalizer in module space,
// and a blind LLR scale. Blur is a channel to be inverted, not noise to be outvoted.
#include "internal.h"
#include <stdlib.h>
#include <string.h>

static void block_stats(const float *s, int w, int h, int win, int bw, int bh, float *m, float *d, int absolute) {
  for (int by = 0; by < bh; by++) for (int bx = 0; bx < bw; bx++) {
    int x1 = (bx + 1) * win < w ? (bx + 1) * win : w, y1 = (by + 1) * win < h ? (by + 1) * win : h, n = 0;
    double a = 0, q = 0;
    for (int y = by * win; y < y1; y++) for (int x = bx * win; x < x1; x++) { float v = s[y * w + x]; a += absolute ? fabsf(v) : v; q += v * v; n++; }
    a /= n;
    m[by * bw + bx] = (float)a;
    if (absolute) d[by * bw + bx] = (float)(q / n - a * a);
    else {
      double dev = 0;
      for (int y = by * win; y < y1; y++) for (int x = bx * win; x < x1; x++) dev += fabs(s[y * w + x] - a);
      d[by * bw + bx] = (float)(dev / n);
    }
  }
}

#include "simd.h"

// Bilinear read of a per-block grid at every cell of a row. The column part of the lookup (corner
// blocks x0, x1 and weight fx) depends on x alone, so it is tabulated once, and a row is walked in
// runs of cells that share their corner blocks: inside a run the four corners are constants and
// only fx varies, which is four cells to a vector. Same expression, same order, as one cell at a time.
typedef struct { int *x0, *x1, *run; float *fx; int runs; } xtab_t;

static int xtab_init(xtab_t *t, int w, int bw, int win) {
  t->x0 = malloc((size_t)w * sizeof(int)); t->x1 = malloc((size_t)w * sizeof(int)); t->run = malloc((size_t)(w + 1) * sizeof(int)); t->fx = malloc((size_t)w * sizeof(float));
  if (!t->x0 || !t->x1 || !t->run || !t->fx) return -1;
  t->runs = 0;
  for (int x = 0; x < w; x++) {
    float fx = (x + 0.5f) / win - 0.5f;
    if (fx < 0) fx = 0; else if (fx > bw - 1) fx = bw - 1;
    int x0 = (int)fx;
    t->x0[x] = x0; t->x1[x] = x0 + 1 < bw ? x0 + 1 : x0; t->fx[x] = fx - x0;
    if (!x || t->x0[x - 1] != x0) t->run[t->runs++] = x;
  }
  t->run[t->runs] = w;
  return 0;
}
static void xtab_free(xtab_t *t) { free(t->x0); free(t->x1); free(t->run); free(t->fx); }

static void interp_row(const float *g, int bw, int bh, int win, int y, const xtab_t *t, float *out) {
  float fy = (y + 0.5f) / win - 0.5f;
  if (fy < 0) fy = 0; else if (fy > bh - 1) fy = bh - 1;
  int y0 = (int)fy, y1 = y0 + 1 < bh ? y0 + 1 : y0;
  fy -= y0;
  const float *g0 = g + y0 * bw, *g1 = g + y1 * bw;
  for (int k = 0; k < t->runs; k++) {
    int x = t->run[k], end = t->run[k + 1];
    float a = g0[t->x0[x]], b = g0[t->x1[x]], c = g1[t->x0[x]], d = g1[t->x1[x]];
#ifdef OB_SIMD
    const v128_t va = wasm_f32x4_splat(a), vb = wasm_f32x4_splat(b), vc = wasm_f32x4_splat(c), vd = wasm_f32x4_splat(d), one = wasm_f32x4_splat(1), vfy = wasm_f32x4_splat(fy), vgy = wasm_f32x4_splat(1 - fy);
    for (; x + 4 <= end; x += 4) {
      v128_t fx = wasm_v128_load(t->fx + x), gx = wasm_f32x4_sub(one, fx);
      v128_t top = wasm_f32x4_add(wasm_f32x4_mul(va, gx), wasm_f32x4_mul(vb, fx)), bot = wasm_f32x4_add(wasm_f32x4_mul(vc, gx), wasm_f32x4_mul(vd, fx));
      wasm_v128_store(out + x, wasm_f32x4_add(wasm_f32x4_mul(top, vgy), wasm_f32x4_mul(bot, vfy)));
    }
#endif
    for (; x < end; x++) { float fx = t->fx[x]; out[x] = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy; }
  }
}


enum { EQ_MAX = 26 };   // 5 x 5 taps and a bias

// Least squares for a K x K equalizer over the cells of one region. Targets come from `mode`:
//   blind  fixed cells, and data cells whose current value ref is at least conf from zero
//   pilot  scattered pilot cells only (their neighbours are data, so the fit sees data statistics)
//   genie  the transmitted cells, an upper bound for any linear equalizer of this size
// Blind and genie fits take every third cell: there are at most 26 unknowns.
static int fit_equalizer(const ob_layout_t *L, const float *x, const float *ref, float conf, int mode, int K, const uint8_t *truth,
                         int rx0, int ry0, int rx1, int ry1, float *wout) {
  int w = L->w, h = L->h, R = K / 2, N = K * K + 1;
  double A[EQ_MAX][EQ_MAX + 1];
  memset(A, 0, sizeof A);
  if (rx0 < R) rx0 = R;
  if (ry0 < R) ry0 = R;
  if (rx1 > w - R) rx1 = w - R;
  if (ry1 > h - R) ry1 = h - R;
  int used = 0;
  // A row of cells is gathered first (taps, bias, target, zero padding to a multiple of four), and
  // each entry of the normal equations is then one pass over the row with its running sum in a
  // register. Sums run in single precision along a row (a hundred terms at most, so nothing is
  // lost that matters), cells taken four ways by position so four chains run side by side, and each row's
  // sums are added in double. The target rides as column N.
  enum { FW = (EQ_MAX + 4) & ~3 };
  int cols = (N + 4) & ~3;
  float *V = malloc((size_t)(w + 2) * FW * sizeof(float));
  if (!V) return 0;
  for (int y = ry0; y < ry1; y++) {
    int hits = 0;
    // Whether a cell is confident enough to train on is a coin toss to the branch predictor, so the
    // row is written for every candidate and the write position only advances for the ones kept.
    for (int xx = mode == OB_EQ_PILOT ? rx0 : rx0 + (y % 3); xx < rx1; xx += mode == OB_EQ_PILOT ? 1 : 3) {
      int i = y * w + xx, kind = L->kind[i], keep;
      float t, fixed_t = (kind & 3) == CELL_DARK ? -1.0f : 1.0f;
      if (mode == OB_EQ_PILOT) { keep = (kind & CELL_PILOT) != 0; t = fixed_t; }
      else if (mode == OB_EQ_GENIE) { keep = kind == CELL_DATA; t = truth[i] ? -1.0f : 1.0f; }
      else { int data = kind == CELL_DATA; keep = !data | (fabsf(ref[i]) >= conf); t = data ? (ref[i] > 0 ? 1.0f : -1.0f) : fixed_t; }
      float *v = V + hits * cols;
#ifdef OB_SIMD
      if (K == 3) {   // three taps a row, read four wide; each store's fourth float is overwritten by what follows
        wasm_v128_store(v, wasm_v128_load(x + i - w - 1)); wasm_v128_store(v + 3, wasm_v128_load(x + i - 1)); wasm_v128_store(v + 6, wasm_v128_load(x + i + w - 1));
      } else
#endif
      { int k = 0; for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) v[k++] = x[i + dy * w + dx]; }
      v[N - 1] = 1; v[N] = t;
      for (int k = N + 1; k < cols; k++) v[k] = 0;
      hits += keep;
    }
    // From the vector boundary at or below the diagonal: the few entries left of it are never read.
    for (int a = 0; a < N; a++) for (int b = a & ~3; b < cols; b += 4) {
      float sum[4];
      int c = 0;
#ifdef OB_SIMD
      #define MAC(s, p) s = wasm_f32x4_add(s, wasm_f32x4_mul(wasm_f32x4_splat((p)[a]), wasm_v128_load((p) + b)))
      v128_t s0 = wasm_f32x4_splat(0), s1 = s0, s2 = s0, s3 = s0;
      for (; c + 4 <= hits; c += 4) { const float *p = V + c * cols; MAC(s0, p); MAC(s1, p + cols); MAC(s2, p + 2 * cols); MAC(s3, p + 3 * cols); }
      for (; c < hits; c++) { const float *p = V + c * cols; switch (c & 3) { case 0: MAC(s0, p); break; case 1: MAC(s1, p); break; case 2: MAC(s2, p); break; default: MAC(s3, p); } }
      #undef MAC
      wasm_v128_store(sum, wasm_f32x4_add(wasm_f32x4_add(s0, s1), wasm_f32x4_add(s2, s3)));
#else
      float s[4][4] = { { 0 } };
      for (; c < hits; c++) { const float *p = V + c * cols; for (int k = 0; k < 4; k++) s[c & 3][k] += p[a] * p[b + k]; }
      for (int k = 0; k < 4; k++) sum[k] = (s[0][k] + s[1][k]) + (s[2][k] + s[3][k]);
#endif
      for (int k = 0; k < 4; k++) if (b + k >= a && b + k <= N) A[a][b + k] += sum[k];
    }
    used += hits;
  }
  free(V);
  if (used < 4 * N) return 0;
  for (int a = 0; a < N; a++) { for (int b = 0; b < a; b++) A[a][b] = A[b][a]; A[a][a] += 1e-3; }
  for (int c = 0; c < N; c++) {
    int p = c;
    for (int r = c + 1; r < N; r++) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (fabs(A[p][c]) < 1e-12) return 0;
    for (int k = 0; k <= N; k++) { double t = A[c][k]; A[c][k] = A[p][k]; A[p][k] = t; }
    for (int r = 0; r < N; r++) { if (r == c) continue; double f = A[r][c] / A[c][c]; for (int k = c; k <= N; k++) A[r][k] -= f * A[c][k]; }
  }
  for (int a = 0; a < N; a++) wout[a] = (float)(A[a][N] / A[a][a]);
  return 1;
}

// coef holds regions x regions filters. With more than one, each cell blends the four nearest
// by distance to their region centres, so the filter follows a blur that varies over the frame.
static void apply_equalizer(const float *x, int w, int h, int K, int regions, const float *coef, float *e) {
  int R = K / 2, N = K * K + 1;
  float c[EQ_MAX];
  memcpy(e, x, (size_t)w * h * sizeof(float));
  for (int y = R; y < h - R; y++) for (int xx = R; xx < w - R; xx++) {
#ifdef OB_SIMD
    // One filter for the frame: the taps are constants, so the sum runs over four cells at once, in the same tap order.
    if (regions <= 1) {
      for (; xx + 4 <= w - R; xx += 4) {
        v128_t acc = wasm_f32x4_splat(coef[N - 1]);
        int k = 0;
        for (int dy = -R; dy <= R; dy++) { const float *p = x + (y + dy) * w + xx - R; for (int dx = 0; dx < K; dx++) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(coef[k++]), wasm_v128_load(p + dx))); }
        wasm_v128_store(e + y * w + xx, acc);
      }
      if (xx >= w - R) continue;
    }
#endif
    const float *cc = coef;
    if (regions > 1) {
      float fx = (xx + 0.5f) * regions / w - 0.5f, fy = (y + 0.5f) * regions / h - 0.5f;
      if (fx < 0) fx = 0; else if (fx > regions - 1) fx = (float)(regions - 1);
      if (fy < 0) fy = 0; else if (fy > regions - 1) fy = (float)(regions - 1);
      int x0 = (int)fx, y0 = (int)fy, x1 = x0 + 1 < regions ? x0 + 1 : x0, y1 = y0 + 1 < regions ? y0 + 1 : y0;
      fx -= x0; fy -= y0;
      for (int k = 0; k < N; k++)
        c[k] = (coef[(y0 * regions + x0) * EQ_MAX + k] * (1 - fx) + coef[(y0 * regions + x1) * EQ_MAX + k] * fx) * (1 - fy)
             + (coef[(y1 * regions + x0) * EQ_MAX + k] * (1 - fx) + coef[(y1 * regions + x1) * EQ_MAX + k] * fx) * fy;
      cc = c;
    }
    float acc = cc[N - 1];
    int k = 0;
    for (int dy = -R; dy <= R; dy++) { const float *p = x + (y + dy) * w + xx - R; for (int dx = 0; dx < K; dx++) acc += cc[k++] * p[dx]; }
    e[y * w + xx] = acc;
  }
}

static int fit_all(const ob_layout_t *L, const float *x, const float *ref, float conf, const ob_opts_t *o, const uint8_t *truth, float *coef) {
  int G = o->eq_regions > 1 ? o->eq_regions : 1, K = o->eq_taps == 5 ? 5 : 3;
  float whole[EQ_MAX];
  if (!fit_equalizer(L, x, ref, conf, o->eq_mode, K, truth, 0, 0, L->w, L->h, whole)) return 0;
  for (int gy = 0; gy < G; gy++) for (int gx = 0; gx < G; gx++) {
    float *dst = coef + (gy * G + gx) * EQ_MAX;
    // A region short of targets keeps the whole-frame filter.
    if (G == 1 || !fit_equalizer(L, x, ref, conf, o->eq_mode, K, truth, gx * L->w / G, gy * L->h / G, (gx + 1) * L->w / G, (gy + 1) * L->h / G, dst))
      memcpy(dst, whole, sizeof whole);
  }
  return 1;
}

void ob_demap(const ob_layout_t *L, const float *s, const ob_opts_t *o, const uint8_t *truth, int8_t *llr_cell, ob_result_t *res) {
  int w = L->w, h = L->h, cells = w * h;
  float *x = malloc(((size_t)cells + 4) * sizeof(float)), *e = malloc((size_t)cells * sizeof(float));   // + 4: the fit reads taps four wide
  enum { WIN = 16 };
  int bw = (w + WIN - 1) / WIN, bh = (h + WIN - 1) / WIN;
  float *gm = malloc((size_t)bw * bh * sizeof(float)), *gd = malloc((size_t)bw * bh * sizeof(float));
  PROF(PROF_D_STATS, block_stats(s, w, h, WIN, bw, bh, gm, gd, 0));
  double tn = ob_now_ms();
  xtab_t tab;
  float *ra = malloc((size_t)w * sizeof(float)), *rb = malloc((size_t)w * sizeof(float));
  xtab_init(&tab, w, bw, WIN);
  for (int y = 0; y < h; y++) {
    interp_row(gd, bw, bh, WIN, y, &tab, ra);
    interp_row(gm, bw, bh, WIN, y, &tab, rb);
    for (int xx = 0; xx < w; xx++) { float d = ra[xx]; x[y * w + xx] = (s[y * w + xx] - rb[xx]) / (d > 1e-4f ? d : 1e-4f); }
  }
  ob_prof_ms[PROF_D_NORM] += ob_now_ms() - tn;
  int mode = o->eq_mode;
  if (mode == OB_EQ_GENIE && !truth) mode = OB_EQ_NONE;
  memcpy(e, x, (size_t)cells * sizeof(float));
  if (mode != OB_EQ_NONE) {
    int G = o->eq_regions > 1 ? o->eq_regions : 1, K = o->eq_taps == 5 ? 5 : 3;
    float *coef = malloc((size_t)G * G * EQ_MAX * sizeof(float));
    ob_opts_t oo = *o;
    oo.eq_mode = mode;
    // Blind fitting feeds on its own decisions, so it goes twice; the others know their targets.
    for (int pass = 0; pass < (mode == OB_EQ_BLIND ? 2 : 1); pass++) {
      int fitted;
      PROF(PROF_D_FIT, fitted = fit_all(L, x, e, pass ? 0.3f : 0.6f, &oo, truth, coef));
      if (fitted) PROF(PROF_D_APPLY, apply_equalizer(x, w, h, K, G, coef, e));
    }
    free(coef);
  }

  // Blind LLR scale per block: |e| clusters at mu with spread var, so LLR = 2 mu e / var.
  PROF(PROF_D_STATS, block_stats(e, w, h, WIN, bw, bh, gm, gd, 1));
  double tl = ob_now_ms();
  const float g2 = o->llr_gain * 2, clip = o->llr_clip;
  for (int y = 0; y < h; y++) {
    interp_row(gm, bw, bh, WIN, y, &tab, ra);
    interp_row(gd, bw, bh, WIN, y, &tab, rb);
    const float *er = e + y * w;
    int8_t *lr = llr_cell + y * w;
    int xx = 0;
#ifdef OB_SIMD
    const v128_t k02 = wasm_f32x4_splat(0.02f), vg2 = wasm_f32x4_splat(g2), tiny = wasm_f32x4_splat(1e-9f), vclip = wasm_f32x4_splat(clip), nclip = wasm_f32x4_splat(-clip), eight = wasm_f32x4_splat(8);
    for (; xx + 4 <= w; xx += 4) {
      v128_t mu = wasm_v128_load(ra + xx), var = wasm_v128_load(rb + xx), floor_ = wasm_f32x4_mul(wasm_f32x4_mul(k02, mu), mu);
      var = wasm_v128_bitselect(floor_, var, wasm_f32x4_lt(var, floor_));
      v128_t l = wasm_f32x4_div(wasm_f32x4_mul(wasm_f32x4_mul(vg2, mu), wasm_v128_load(er + xx)), var);
      l = wasm_v128_and(l, wasm_f32x4_gt(var, tiny));
      l = wasm_v128_bitselect(vclip, l, wasm_f32x4_gt(l, vclip));
      l = wasm_v128_bitselect(nclip, l, wasm_f32x4_lt(l, nclip));
      // nearest rounds ties to even, which is what lrintf does in the default mode.
      v128_t q = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(l, eight)));
      v128_t b = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(q, q), q);
      int32_t four = wasm_i32x4_extract_lane(b, 0);
      memcpy(lr + xx, &four, 4);
    }
#endif
    for (; xx < w; xx++) {
      float mu = ra[xx], var = rb[xx];
      if (var < 0.02f * mu * mu) var = 0.02f * mu * mu;
      float l = var > 1e-9f ? g2 * mu * er[xx] / var : 0;
      if (l > clip) l = clip; else if (l < -clip) l = -clip;
      lr[xx] = (int8_t)nearbyintf(l * 8);
    }
  }
  xtab_free(&tab); free(ra); free(rb);
  ob_prof_ms[PROF_D_LLR] += ob_now_ms() - tl;
  free(gm); free(gd);

  if (truth) {
    double errs = 0, acc = 0;
    int n = 0;
    for (int i = 0; i < cells; i++) {
      if (L->kind[i] != CELL_DATA) continue;
      float l = llr_cell[i] / 8.0f, z = truth[i] ? l : -l;
      errs += (e[i] > 0) == (truth[i] != 0);
      acc += z > 30 ? z / 0.6931472f : log2f(1 + expf(z));
      n++;
    }
    res->ber = (float)(errs / n); res->gmi = (float)(1 - acc / n);
  }
  free(x); free(e);
}
