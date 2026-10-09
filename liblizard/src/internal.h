// Shared by the receivers: image access, the projective map, and the registration result.
#ifndef OB_INTERNAL_H
#define OB_INTERNAL_H
#include "ob.h"
#include <math.h>

typedef struct { const uint8_t *px; int w, h; float lut[256]; } image_t;

static inline float sample(const image_t *im, float x, float y) {
  x -= 0.5f; y -= 0.5f;
  if (x < 0) x = 0; else if (x > im->w - 1.001f) x = im->w - 1.001f;
  if (y < 0) y = 0; else if (y > im->h - 1.001f) y = im->h - 1.001f;
  // ai: a NaN coordinate (it passes both clamps) indexes pixel 0, as arm64's fcvtzs and wasm's trunc_sat make it; x86's
  // ai: cvttss2si gave INT_MIN, an out-of-bounds read (2026-10-03, for the library's x86 builds). fx stays NaN, so the
  // ai: result is what those platforms already compute.
  int xi = x >= 0 ? (int)x : 0, yi = y >= 0 ? (int)y : 0;
  float fx = x - xi, fy = y - yi;
  const uint8_t *p = im->px + yi * im->w + xi;
  float a = im->lut[p[0]], b = im->lut[p[1]], c = im->lut[p[im->w]], d = im->lut[p[im->w + 1]];
  return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
}

#include "simd.h"
#ifdef OB_SIMD
// sample() for four points at once: the same operations in the same order, so the same floats.
// The sixteen pixel reads stay scalar (wasm has no gather); the arithmetic around them is what is shared.
static inline v128_t sample4(const image_t *im, v128_t x, v128_t y) {
  const v128_t half = wasm_f32x4_splat(0.5f), zero = wasm_f32x4_splat(0), one = wasm_f32x4_splat(1);
  x = wasm_f32x4_pmin(wasm_f32x4_pmax(wasm_f32x4_sub(x, half), zero), wasm_f32x4_splat(im->w - 1.001f));
  y = wasm_f32x4_pmin(wasm_f32x4_pmax(wasm_f32x4_sub(y, half), zero), wasm_f32x4_splat(im->h - 1.001f));
  const v128_t xi = wasm_i32x4_trunc_sat_f32x4(x), yi = wasm_i32x4_trunc_sat_f32x4(y);
  const v128_t fx = wasm_f32x4_sub(x, wasm_f32x4_convert_i32x4(xi)), fy = wasm_f32x4_sub(y, wasm_f32x4_convert_i32x4(yi));
  int32_t idx[4];
  wasm_v128_store(idx, wasm_i32x4_add(wasm_i32x4_mul(yi, wasm_i32x4_splat(im->w)), xi));
  const uint8_t *p0 = im->px + idx[0], *p1 = im->px + idx[1], *p2 = im->px + idx[2], *p3 = im->px + idx[3];
  const float *lut = im->lut;
  const int w = im->w;
  const v128_t a = wasm_f32x4_make(lut[p0[0]], lut[p1[0]], lut[p2[0]], lut[p3[0]]), b = wasm_f32x4_make(lut[p0[1]], lut[p1[1]], lut[p2[1]], lut[p3[1]]);
  const v128_t c = wasm_f32x4_make(lut[p0[w]], lut[p1[w]], lut[p2[w]], lut[p3[w]]), d = wasm_f32x4_make(lut[p0[w + 1]], lut[p1[w + 1]], lut[p2[w + 1]], lut[p3[w + 1]]);
  const v128_t top = wasm_f32x4_add(a, wasm_f32x4_mul(wasm_f32x4_sub(b, a), fx)), bot = wasm_f32x4_add(c, wasm_f32x4_mul(wasm_f32x4_sub(d, c), fx));
  return wasm_f32x4_add(wasm_f32x4_mul(top, wasm_f32x4_sub(one, fy)), wasm_f32x4_mul(bot, fy));
}
#endif

typedef struct { float h[9]; } homo_t;

static inline void project(const homo_t *H, float x, float y, float *u, float *v) {
  const float *h = H->h;
  float w = h[6] * x + h[7] * y + 1;
  *u = (h[0] * x + h[1] * y + h[2]) / w;
  *v = (h[3] * x + h[4] * y + h[5]) / w;
}
#ifdef OB_SIMD
// project() for four points at once, the same operations in the same order.
static inline void project4(const homo_t *H, v128_t x, v128_t y, v128_t *u, v128_t *v) {
  const float *h = H->h;
  const v128_t w = wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[6]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[7]), y)), wasm_f32x4_splat(1));
  *u = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[0]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[1]), y)), wasm_f32x4_splat(h[2])), w);
  *v = wasm_f32x4_div(wasm_f32x4_add(wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_splat(h[3]), x), wasm_f32x4_mul(wasm_f32x4_splat(h[4]), y)), wasm_f32x4_splat(h[5])), w);
}
#endif

typedef struct { float dx, dy, score; int done; } node_t;
typedef struct { homo_t H; node_t *nodes; } ob_reg_t;

// ai: State a decode leaves behind or keeps between its stages is each thread's own: a native receiver runs a decoder
// ai: a thread (any.h). The wasm build has one thread, where this is a plain global.
#define OB_TLS _Thread_local
double ob_now_ms(void);
// Stage timers for the benches (scripts/exp/focus_bench.mjs). A handful of clock reads per frame, left in. The slots
// before PROF_F_SAMPLE were the binary grid code's (deleted 2026-10-09), kept so the JS readers' indices hold.
enum { PROF_LUMA, PROF_BINARIZE, PROF_FINDERS, PROF_MESH, PROF_SAMPLE, PROF_DEMAP, PROF_LDPC, PROF_PACK, PROF_D_STATS, PROF_D_NORM, PROF_D_FIT, PROF_D_APPLY, PROF_D_LLR,
       PROF_F_SAMPLE, PROF_F_DETREND, PROF_F_FFT, PROF_F_LLR, PROF_F_LDPC, PROF_F_ENC_FFT, PROF_F_ENC_REST,   // F: the FOCUS decoder and encoder (scripts/exp/focus_bench.mjs)
       PROF_FF_POINTS, PROF_FF_TURN, PROF_FF_FIT, PROF_FF_SCORE,   // FF: inside the thin frame's finder (slots 20..23)
       PROF_FF_MARKS, PROF_FF_TRACK, PROF_N };   // 24, 25: the mark scan, and the track read that judges a quad
extern OB_TLS double ob_prof_ms[PROF_N];
#define PROF(i, stmt) do { double t_ = ob_now_ms(); stmt; ob_prof_ms[i] += ob_now_ms() - t_; } while (0)
void ob_luma(const uint8_t *rgba, int n, uint8_t *out);
void ob_image_init(image_t *im, const uint8_t *px, int w, int h, float gamma);

// Finders, orientation, homography, then the mark mesh. mesh = 0 leaves every mark at zero
// displacement, which is what a single global homography gives. Returns 1 when registered.
int ob_acquire(const ob_layout_t *L, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res);
// The same when the symbol may be any of several layouts, which differ only in modules a side: the marks are
// scanned once and each layout's spacing and track tried on them, the first first. *which is the one that registered.
int ob_acquire_any(const ob_layout_t *const *Ls, int nL, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res, int *which);
// The same with the quad already found: the scan, the fit and the track scoring are skipped and the rest is
// shared. quad is eight floats, image pixels, in the order find_frame reports.
int ob_acquire_quad(const ob_layout_t *L, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res,
                    const float *quad, int orient, float score);
extern OB_TLS float ob_raw_quad[8];   // the finder's quad before the corner nodes refine it
int ob_test_mesh_tables(const ob_layout_t *L, int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark);
void ob_test_corner_coords(const ob_layout_t *L, float *src);
int ob_test_homography(const float *src, const float *dst, float *out);
int ob_thin_fmt_plan(const ob_layout_t *L, int cells, float *pts, int32_t *meta, int32_t *widx);
int ob_thin_score_plan(const ob_layout_t *L, float *pts, int32_t *meta, int32_t *signs);
int ob_test_thin_score(const ob_layout_t *L, const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out);
extern OB_TLS float ob_finder_info[8];
// Mark geometry in modules, for a finder measured outside this file. See acquire.c.
void ob_mark_geom(const ob_layout_t *L, float *out);
int ob_test_mark_scan(const ob_layout_t *L, const uint8_t *bin, int w, int h, int merged, float *out, int cap);
int ob_test_mark_cfg(const ob_layout_t *L, int32_t *out);
int ob_test_mark_quads(const ob_layout_t *L, const float *cands, int n, int merged, float *out);
// The line anchor's candidate scan alone, for a port measured outside this file. See acquire.c.
int ob_test_line_points(const uint8_t *grey, const uint8_t *bin, int w, int h, float *out, int32_t *n, int cap);
void ob_reg_free(ob_reg_t *reg);
// Module coordinates to image px through the homography and the mesh.
void ob_map_point(const ob_layout_t *L, const ob_reg_t *reg, float mx, float my, float *u, float *v);
// The thin frame's format word as soft values, q[side * cells + i] in -0.5 .. 0.5, positive for
// dark. Returns 0 where the border has no contrast. See fmt.h.
int ob_thin_read_fmt(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float *q, int cells);
// A block of a regular grid in module coordinates: see acquire.c.
void ob_sample_grid(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch);
// What focus_acquire left for focus_finish (focus.h), for a caller that has to sample somewhere else. Valid
// until the matching focus_finish takes it up.
struct focus_s;
const image_t *focus_held_image(const struct focus_s *f);
const ob_reg_t *focus_held_reg(const struct focus_s *f);

// The block tail of focus_finish_bits, for a second implementation to be measured against (src/focus.c).
int ob_test_crc_pack(const uint8_t *bits, int B, uint8_t *dst, uint32_t *out);
uint32_t ob_test_crc_hash(const uint8_t *p, int n);
int ob_test_crc_gate(const struct focus_s *f, const float *est, float *blk_est, float *bar, uint8_t *declined);

#endif
