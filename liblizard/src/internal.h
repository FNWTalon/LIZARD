// Shared by the receivers: image access, the projective map, and the registration result.
#ifndef OB_INTERNAL_H
#define OB_INTERNAL_H
#include "ob.h"
#include <math.h>
#include <stddef.h>

// ai: lin: the table is i / 255 (gamma 1, as both C decoders read), which the picture sampler on arm64 makes in registers
typedef struct { const uint8_t *px; int w, h; float lut[256]; int lin; } image_t;

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
#ifdef OB_NEON
#include <string.h>
// ai: arm64, the finder's track reads at gamma 1 (acquire.c trk_px; the decode's picture has its own read in integers,
// ai: acquire.c sample_grid_lin; registration keeps sample4): four points' bilinear reads, x and y already less the
// ai: half pixel and clamped to the frame as sample() clamps them, on a frame of w pixels a row. A point's pixel pairs
// ai: as two 32-bit words, row y's from its pixel and row y + 1's from two bytes before (both inside the frame at its
// ai: last pixel), two words to a general register; one TBL over a word register and a register of 0x4B (the exponent
// ai: of 2^23) makes each byte the float 2^23 + byte, so the differences are exact and a and c come back by one
// ai: subtraction, with no conversion; blended on 0 to 255 with fused multiply-adds and scaled by 1/255 once. Its own
// ai: floats: the blocks a decode verifies are its test.
static inline float32x4_t sample4r(const uint8_t *px, int w, float32x4_t x, float32x4_t y) {
  const float32x4_t fx = vsubq_f32(x, vrndq_f32(x)), fy = vsubq_f32(y, vrndq_f32(y));
  const int32x4_t idx = vmlaq_s32(vcvtq_s32_f32(x), vcvtq_s32_f32(y), vdupq_n_s32(w));
  const uint64_t i01 = vgetq_lane_u64(vreinterpretq_u64_s32(idx), 0), i23 = vgetq_lane_u64(vreinterpretq_u64_s32(idx), 1);
  const size_t o0 = (uint32_t)i01, o1 = i01 >> 32, o2 = (uint32_t)i23, o3 = i23 >> 32;
  const uint8_t *r0 = px, *r1 = px + w - 2;
  uint32_t t0, t1, t2, t3, u0, u1, u2, u3;
  memcpy(&t0, r0 + o0, 4); memcpy(&t1, r0 + o1, 4); memcpy(&t2, r0 + o2, 4); memcpy(&t3, r0 + o3, 4);
  memcpy(&u0, r1 + o0, 4); memcpy(&u1, r1 + o1, 4); memcpy(&u2, r1 + o2, 4); memcpy(&u3, r1 + o3, 4);
  const uint8x16_t k23 = vdupq_n_u8(0x4b);
  const uint8x16x2_t T = { { vreinterpretq_u8_u64(vcombine_u64(vcreate_u64((uint64_t)t0 | (uint64_t)t1 << 32), vcreate_u64((uint64_t)t2 | (uint64_t)t3 << 32))), k23 } };
  const uint8x16x2_t U = { { vreinterpretq_u8_u64(vcombine_u64(vcreate_u64((uint64_t)u0 | (uint64_t)u1 << 32), vcreate_u64((uint64_t)u2 | (uint64_t)u3 << 32))), k23 } };
  // ai: a word's bytes: row y's a b (and two past them), row y + 1's (two before) c d; index 16 is the 0x4B, 255 a zero
  const uint8x16_t ka = { 0, 255, 255, 16, 4, 255, 255, 16, 8, 255, 255, 16, 12, 255, 255, 16 };
  const uint8x16_t kb = { 1, 255, 255, 16, 5, 255, 255, 16, 9, 255, 255, 16, 13, 255, 255, 16 };
  const uint8x16_t kc = { 2, 255, 255, 16, 6, 255, 255, 16, 10, 255, 255, 16, 14, 255, 255, 16 };
  const uint8x16_t kd = { 3, 255, 255, 16, 7, 255, 255, 16, 11, 255, 255, 16, 15, 255, 255, 16 };
  const float32x4_t am = vreinterpretq_f32_u8(vqtbl2q_u8(T, ka)), bm = vreinterpretq_f32_u8(vqtbl2q_u8(T, kb));
  const float32x4_t cm = vreinterpretq_f32_u8(vqtbl2q_u8(U, kc)), dm = vreinterpretq_f32_u8(vqtbl2q_u8(U, kd));
  const float32x4_t a = vsubq_f32(am, vdupq_n_f32(8388608.0f)), c = vsubq_f32(cm, vdupq_n_f32(8388608.0f));
  const float32x4_t top = vfmaq_f32(a, vsubq_f32(bm, am), fx), bot = vfmaq_f32(c, vsubq_f32(dm, cm), fx);
  return vmulq_f32(vfmaq_f32(top, vsubq_f32(bot, top), fy), vdupq_n_f32(1.0f / 255.0f));
}
#elif defined(OB_X86)
#include <string.h>
// ai: x86-64: arm64's sample4r (above) in SSE4.1, for the same track reads: the same word reads, each byte made the
// ai: float 2^23 + byte by PSHUFB over the words with 0x4B ORed in as the top byte, the blends a multiply and an add
// ai: each (no fused multiply-add): its own floats.
static inline __m128 sample4r(const uint8_t *px, int w, __m128 x, __m128 y) {
  const __m128 fx = _mm_sub_ps(x, _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)), fy = _mm_sub_ps(y, _mm_round_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  const __m128i idx = _mm_add_epi32(_mm_cvttps_epi32(x), _mm_mullo_epi32(_mm_cvttps_epi32(y), _mm_set1_epi32(w)));
  const size_t o0 = (uint32_t)_mm_cvtsi128_si32(idx), o1 = (uint32_t)_mm_extract_epi32(idx, 1), o2 = (uint32_t)_mm_extract_epi32(idx, 2), o3 = (uint32_t)_mm_extract_epi32(idx, 3);
  const uint8_t *r0 = px, *r1 = px + w - 2;
  uint32_t t0, t1, t2, t3, u0, u1, u2, u3;
  memcpy(&t0, r0 + o0, 4); memcpy(&t1, r0 + o1, 4); memcpy(&t2, r0 + o2, 4); memcpy(&t3, r0 + o3, 4);
  memcpy(&u0, r1 + o0, 4); memcpy(&u1, r1 + o1, 4); memcpy(&u2, r1 + o2, 4); memcpy(&u3, r1 + o3, 4);
  const __m128i T = _mm_setr_epi32((int)t0, (int)t1, (int)t2, (int)t3), U = _mm_setr_epi32((int)u0, (int)u1, (int)u2, (int)u3);
  const __m128i k23 = _mm_set1_epi32(0x4b000000);
  // ai: a word's bytes: row y's a b (and two past them), row y + 1's (two before) c d; -1 a zero
  const __m128i ka = _mm_setr_epi8(0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1), kb = _mm_setr_epi8(1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1);
  const __m128i kc = _mm_setr_epi8(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1), kd = _mm_setr_epi8(3, -1, -1, -1, 7, -1, -1, -1, 11, -1, -1, -1, 15, -1, -1, -1);
  const __m128 am = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(T, ka), k23)), bm = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(T, kb), k23));
  const __m128 cm = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(U, kc), k23)), dm = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(U, kd), k23));
  const __m128 a = _mm_sub_ps(am, _mm_set1_ps(8388608.0f)), c = _mm_sub_ps(cm, _mm_set1_ps(8388608.0f));
  const __m128 top = _mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(bm, am), fx)), bot = _mm_add_ps(c, _mm_mul_ps(_mm_sub_ps(dm, cm), fx));
  return _mm_mul_ps(_mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bot, top), fy)), _mm_set1_ps(1.0f / 255.0f));
}
#endif
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
// Stage timers for the benches (scripts/exp/lizard_bench.mjs). A handful of clock reads per frame, left in. The slots by
// index, as the JS and the native tools read them: 0 luma, 1 binarize, 2 finders, 3 mesh; 4 to 10 the picture's decoder
// and encoder (sample, detrend, transform, LLR, LDPC, the encoder's transform and the rest of it); 11 to 16 inside the
// thin frame's finder (points, turn, fit, score, the mark scan, the track read that judges a quad). Since 2026-10-25:
// slots 4 to 12 were the binary grid code's until then, and every later slot sat 9 higher.
enum { PROF_LUMA, PROF_BINARIZE, PROF_FINDERS, PROF_MESH,
       PROF_F_SAMPLE, PROF_F_DETREND, PROF_F_FFT, PROF_F_LLR, PROF_F_LDPC, PROF_F_ENC_FFT, PROF_F_ENC_REST,   // F: the picture's decoder and encoder
       PROF_FF_POINTS, PROF_FF_TURN, PROF_FF_FIT, PROF_FF_SCORE,   // FF: inside the thin frame's finder
       PROF_FF_MARKS, PROF_FF_TRACK, PROF_N };   // the mark scan, and the track read that judges a quad
extern OB_TLS double ob_prof_ms[PROF_N];
#define PROF(i, stmt) do { double t_ = ob_now_ms(); stmt; ob_prof_ms[i] += ob_now_ms() - t_; } while (0)
void ob_luma(const uint8_t *rgba, int n, uint8_t *out);
void ob_image_init(image_t *im, const uint8_t *px, int w, int h, float gamma);

// ai: An acquisition's buffers of 64 KB and more at 1280 x 960 (the size from which Scudo maps a fresh allocation and
// ai: faults its pages in on first touch, about 0.2 ms a megabyte on an S26 core): BIN the binarized capture, whose
// ai: buffer border_blur's copy takes over once the finder is done with it, CB its column bits, HALF and QUARTER the
// ai: halved and quartered pictures, LBIN and LCB theirs binarized, MEAN binarize's block means, EDGE the line scan's
// ai: block map (edge_ctx_init). Held by the caller's workspace (lizard_ws), grown as a frame needs and freed with it
// ai: (ob_fbufs_free). On arm64 and x86-64 the acquisition takes them from here; elsewhere it allocates them a frame as
// ai: before.
// ai: Every byte is written before it is read within a frame, so nothing carries from one frame to the next; one
// ai: acquisition at a time, as the workspace belongs to one thread at a time.
enum { OB_FB_BIN, OB_FB_CB, OB_FB_HALF, OB_FB_QUARTER, OB_FB_LBIN, OB_FB_LCB, OB_FB_MEAN, OB_FB_EDGE, OB_FB_SLOTS };
typedef struct { void *p[OB_FB_SLOTS]; size_t cap[OB_FB_SLOTS]; } ob_fbufs_t;
void ob_fbufs_free(ob_fbufs_t *fb);
// Finders, orientation, homography, then the mark mesh. mesh = 0 leaves every mark at zero
// displacement, which is what a single global homography gives. Returns 1 when registered.
// ai: fb: the workspace's buffers (above), or NULL to allocate them for this call alone
int ob_acquire(const ob_layout_t *L, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res, ob_fbufs_t *fb);
// The same when the symbol may be any of several layouts, which differ only in modules a side: the marks are
// scanned once and each layout's spacing and track tried on them, the first first. *which is the one that registered.
int ob_acquire_any(const ob_layout_t *const *Ls, int nL, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res, int *which, ob_fbufs_t *fb);
// The same with the quad already found: the scan, the fit and the track scoring are skipped and the rest is
// shared. quad is eight floats, image pixels, in the order find_frame reports.
int ob_acquire_quad(const ob_layout_t *L, const image_t *im, int mesh, ob_reg_t *reg, ob_result_t *res,
                    const float *quad, int orient, float score, ob_fbufs_t *fb);
int ob_test_mesh_tables(const ob_layout_t *L, int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark);
int ob_thin_fmt_plan(const ob_layout_t *L, int cells, float *pts, int32_t *meta, int32_t *widx);
int ob_thin_score_plan(const ob_layout_t *L, float *pts, int32_t *meta, int32_t *signs);
int ob_test_thin_score(const ob_layout_t *L, const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out);
extern OB_TLS float ob_finder_info[8];
// Mark geometry in modules, for a finder measured outside this file. See acquire.c.
void ob_mark_geom(const ob_layout_t *L, float *out);
void ob_reg_free(ob_reg_t *reg);
// Module coordinates to image px through the homography and the mesh.
void ob_map_point(const ob_layout_t *L, const ob_reg_t *reg, float mx, float my, float *u, float *v);
// ai: The anchors (acquire.c, SPEC 6.13): a 7 x 7 grid's square side in modules (B / 32), and anchor i's first module
// ai: along either axis; OB_ANCH_TS template points across one. OB_ANCH_PX_MIN: the drive pixels a square needs a side
// ai: for its light middle (a quarter of the half-side) to land on a pixel, which sets a floor under pxm (lizard_phy.c init):
// ai: at 4 (the 256 picture in the 64 and 128 rings at ceil(n / span)) a square paints as a dark dot.
// ai: OB_ANCH_MIN: the anchors found that the geometry takes, at least the lens model's most terms (acquire.c LENS_P,
// ai: 11): their 2n readings then leave as many as the model takes, so its robust rule can judge one. Fewer, a false reading (a
// ai: blurred square's neighbourhood at 0.6) is fitted exactly and bends the map (the bench's 64 ring at 2.7 camera px
// ai: a module, 8 found, one 2.3 modules off: 0 blocks of 23, 16 without the anchors).
enum { OB_ANCH_K = 7, OB_ANCH_TS = 16, OB_ANCH_PX_MIN = 6, OB_ANCH_MIN = 12 };
enum { OB_ANCH_MARGIN = 2 };   // ai: the samples erased round an anchor's square
extern OB_TLS int ob_anch_found;   // ai: the anchors the last acquisition counted, -1 where no mesh was made (acquire.c)
int ob_anchor_side(const ob_layout_t *L);
int ob_anchor_m0(const ob_layout_t *L, int i);
// The thin frame's format word as soft values, q[side * cells + i] in -0.5 .. 0.5, positive for
// dark. Returns 0 where the border has no contrast. See fmt.h.
int ob_thin_read_fmt(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float *q, int cells);
// A block of a regular grid in module coordinates: see acquire.c.
void ob_sample_grid(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch);
// ai: the decode's picture: on arm64 and x86-64 at gamma 1 a walk of its own and its own floats (acquire.c), else ob_sample_grid's
void ob_sample_grid_pic(const ob_layout_t *L, const image_t *im, const ob_reg_t *reg, float mx0, float my0, float step, int x0, int nx, int y0, int ny, float *out, int pitch);
// What lizard_acquire left for lizard_finish (lizard_phy.h), for a caller that has to sample somewhere else. Valid
// until the matching lizard_finish takes it up.
struct lizard_s;
const image_t *lizard_held_image(const struct lizard_s *f);
const ob_reg_t *lizard_held_reg(const struct lizard_s *f);

// The block CRC (test/vectors.mjs) and the decline gate (the GPU back half's is held to it), from src/lizard_phy.c.
uint32_t ob_test_crc_hash(const uint8_t *p, int n);
int ob_test_crc_gate(const struct lizard_s *f, const float *est, float *blk_est, float *bar, uint8_t *declined);

#endif
