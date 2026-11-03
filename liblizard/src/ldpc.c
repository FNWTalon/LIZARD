#include "ldpc.h"
#include "ldpc_base.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>

// ai: arm64's decode keeps a pointer a slot (ldpc_decode_stall): the codes with at most this many take the four-times
// ai: columns (the format's most, the 1/2 code's, 210)
enum { LDPC_VIEWS = 256 };

// ai: x86-64's wider layer kernels (below) exist where the 128-bit vector path does and the compiler takes a function's
// ai: own instruction set (not clang-cl, whose MSVC-style link may lack the runtime the processor check calls)
#if defined(OB_SIMD) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define OB_LDPC_WIDE 1
static int wide_for(int zp);
#endif

// ai: The rate indices are the format's four, 2, 3, 4 and 6 (1/2, 2/3, 3/4, 7/8); 0, 1 and 5 (1/4, 1/3, 5/6) were lab
// ai: rates (archive/ldpc-lab-rates/) and have no code. norm is the min-sum scale in sixteenths, 13 at every rate.
typedef struct { int norm; const char *name; double r; } profile_t;
static const profile_t PROFILE[LDPC_RATES] = {
  [2] = { 13, "1/2", 0.5 },
  [3] = { 13, "2/3", 2.0 / 3 },
  [4] = { 13, "3/4", 0.75 },
  [6] = { 13, "7/8", 0.875 },
};
const char *const ldpc_rate_name[LDPC_RATES] = { [2] = "1/2", [3] = "2/3", [4] = "3/4", [6] = "7/8" };
double ldpc_rate_value(int rate) { return rate >= 0 && rate < LDPC_RATES ? PROFILE[rate].r : 0; }

static int mod(int a, int z) { a %= z; return a < 0 ? a + z : a; }
// ai: the format's table at a rate index (ldpc_base.h), or none
static const ldpc_base_t *base_of(int rate) {
  for (int t = 0; t < LDPC_BASES_N; t++) if (LDPC_BASES[t].rate == rate) return &LDPC_BASES[t];
  return 0;
}
int ldpc_sent_bits(int rate) { const ldpc_base_t *b = base_of(rate); return b ? (b->kb - b->np + b->mb) * b->z : 0; }

// shift[r][j] is the circulant offset, -1 where the base graph has no edge: data bit t of
// block column j sits in position i = (t + shift) % z of block row r, which is check
// i * mb + r. Interleaving the block rows along the staircase is what gives the code a
// minimum distance: with checks numbered r * z + i instead, data bits t and t + 1 of a
// degree-3 column land on adjacent staircase checks in every block row, and flipping both
// plus the three parity bits between them is a weight-5 codeword.
int ldpc_init(ldpc_t *c, int rate) {
  memset(c, 0, sizeof *c);
  const ldpc_base_t *base = base_of(rate);
  if (!base) return -1;
  const int z = base->z, mb = base->mb, kb = base->kb;
  const short *shift = base->shift;   // ai: [mb][kb]
  c->z = z; c->mb = mb; c->kb = kb; c->rate = rate; c->norm = PROFILE[rate].norm;
  c->n = (kb + mb) * z; c->k = kb * z; c->m = mb * z; c->np = base->np * z; c->nt = c->n - c->np;

  int edges = 0;
  for (int r = 0; r < mb; r++) { int d = 0; for (int j = 0; j < kb; j++) d += shift[r * kb + j] >= 0; edges += (d + 2) * z; }
  c->row_ptr = malloc((size_t)(c->m + 1) * sizeof(int));
  c->col_idx = malloc((size_t)edges * sizeof(int));
  if (!c->row_ptr || !c->col_idx) { ldpc_free(c); return -1; }
  int e = 0;
  for (int chk = 0; chk < c->m; chk++) {
    int i = chk / mb, r = chk % mb;
    c->row_ptr[chk] = e;
    for (int j = 0; j < kb; j++) if (shift[r * kb + j] >= 0) c->col_idx[e++] = j * z + mod(i - shift[r * kb + j], z);
    if (chk > 0) c->col_idx[e++] = c->k + chk - 1;
    c->col_idx[e++] = c->k + chk;
  }
  c->row_ptr[c->m] = e;
  c->edges = e;

  c->zp = (z + 7) & ~7;
#ifdef OB_LDPC_WIDE
  c->wide = wide_for(c->zp);
#endif
  c->lay_ptr = malloc((size_t)(mb + 1) * sizeof(int));
  c->lay_col = malloc((size_t)mb * kb * sizeof(int));
  c->lay_shift = malloc((size_t)mb * kb * sizeof(int));
  if (!c->lay_ptr || !c->lay_col || !c->lay_shift) { ldpc_free(c); return -1; }
  int s2 = 0;
  for (int r = 0; r < mb; r++) {
    c->lay_ptr[r] = s2;
    for (int j = 0; j < kb; j++) if (shift[r * kb + j] >= 0) { c->lay_col[s2] = j; c->lay_shift[s2++] = shift[r * kb + j]; }
    int ns = s2 - c->lay_ptr[r] + 2;
    if (ns > c->slots_max) c->slots_max = ns;
    c->slots_total += ns;
  }
  c->lay_ptr[mb] = s2;
  c->wl = malloc((size_t)c->n * sizeof(int16_t));
  c->wq = calloc((size_t)c->slots_max * c->zp, sizeof(int16_t));
  c->wr = malloc((size_t)c->slots_total * c->zp);
  if (!c->wl || !c->wq || !c->wr) { ldpc_free(c); return -1; }
#ifdef OB_NEON
  // ai: arm64's kernel leaves out the posteriors' clamp at 8191, which never acts while a posterior, its LLR (within
  // ai: +-128) plus its column's messages (within +-127 each), stays inside it: the largest column degree at most 63
  int dmax = 0;
  for (int j = 0; j < kb; j++) { int d = 0; for (int r = 0; r < mb; r++) d += shift[r * kb + j] >= 0; if (d > dmax) dmax = d; }
  if (c->z % 32 == 0 && c->zp == c->z && c->slots_max <= 64 && c->slots_total <= LDPC_VIEWS && 128 + 127 * dmax <= 8191 && c->norm >= 8) {
    c->w4 = malloc((size_t)c->kb * 4 * c->z * sizeof(int16_t));
    if (!c->w4) { ldpc_free(c); return -1; }
  }
#endif
  return 0;
}

void ldpc_free(ldpc_t *c) { free(c->row_ptr); free(c->col_idx); free(c->lay_ptr); free(c->lay_col); free(c->lay_shift); free(c->wl); free(c->wq); free(c->wr); free(c->w4); memset(c, 0, sizeof *c); }

void ldpc_encode(const ldpc_t *c, const uint8_t *data, uint8_t *cw) {
  memcpy(cw, data, (size_t)c->k);
  uint8_t acc = 0;
  for (int chk = 0; chk < c->m; chk++) {
    // The last entry of a row is its own parity bit, the one before it the previous one.
    int end = c->row_ptr[chk + 1] - (chk > 0 ? 2 : 1);
    for (int e = c->row_ptr[chk]; e < end; e++) acc ^= data[c->col_idx[e]];
    cw[c->k + chk] = acc;
  }
}

// ---------------------------------------------------------------- block-row schedule
//
// The z checks of one block row share no bit: a circulant puts each data bit in one of them, and
// check i * mb + r owns staircase bits i * mb + r - 1 and i * mb + r, which no other check of
// row r touches. So a layer is a whole block row, and its z checks run side by side, eight to a
// vector. Each circulant's bits are first rotated into check order (two memcpys), which makes
// every slot a plain array indexed by check position.

#include "simd.h"

// Q: ns rows of zp posteriors in check order. R: the matching messages. Layered normalized min-sum, check by check
// (one check a layer, the sweep's direction alternating, measured against it on 2026-09-19: research/05); the scalar
// loop below is the reference for the vector one.
static void layer_kernel(int16_t *Q, int8_t *R, int ns, int zp, int norm) {
#ifdef OB_SIMD
  const v128_t big = wasm_i16x8_splat(32767), cap = wasm_i16x8_splat(2047), top = wasm_i16x8_splat(127), vn = wasm_i16x8_splat((int16_t)norm);
  const v128_t lo = wasm_i16x8_splat(-8191), hi = wasm_i16x8_splat(8191);
  for (int i = 0; i < zp; i += 8) {
    v128_t min1 = big, min2 = big, arg = wasm_i16x8_splat(-1), sgn = wasm_i16x8_splat(0);
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      v128_t v = wasm_i16x8_sub(wasm_v128_load(q), wasm_i16x8_load8x8(R + e * zp + i));
      wasm_v128_store(q, v);
      v128_t a = wasm_i16x8_abs(v), lt = wasm_i16x8_lt(a, min1);
      sgn = wasm_v128_xor(sgn, v);
      min2 = wasm_i16x8_min(min2, wasm_i16x8_max(min1, a));
      arg = wasm_v128_bitselect(wasm_i16x8_splat((int16_t)e), arg, lt);
      min1 = wasm_i16x8_min(min1, a);
    }
    // (x * norm) >> 4 capped at 127. Capping x at 2047 first keeps the product inside 16 bits and changes nothing.
    v128_t m1 = wasm_i16x8_min(wasm_i16x8_shr(wasm_i16x8_mul(wasm_i16x8_min(min1, cap), vn), 4), top);
    v128_t m2 = wasm_i16x8_min(wasm_i16x8_shr(wasm_i16x8_mul(wasm_i16x8_min(min2, cap), vn), 4), top);
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      v128_t v = wasm_v128_load(q);
      v128_t mag = wasm_v128_bitselect(m2, m1, wasm_i16x8_eq(arg, wasm_i16x8_splat((int16_t)e)));
      v128_t neg = wasm_i16x8_shr(wasm_v128_xor(sgn, v), 15);
      v128_t r = wasm_i16x8_sub(wasm_v128_xor(mag, neg), neg);
      int64_t r8 = wasm_i64x2_extract_lane(wasm_i8x16_narrow_i16x8(r, r), 0);
      memcpy(R + e * zp + i, &r8, 8);
      wasm_v128_store(q, wasm_i16x8_max(wasm_i16x8_min(wasm_i16x8_add(v, r), hi), lo));
    }
  }
#else
  for (int i = 0; i < zp; i++) {
    int min1 = 32767, min2 = 32767, arg = -1, sign = 0;
    for (int e = 0; e < ns; e++) {
      int q = Q[e * zp + i] - R[e * zp + i];
      Q[e * zp + i] = (int16_t)q;
      int a = q < 0 ? -q : q;
      sign ^= q < 0;
      if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) min2 = a;
    }
    min1 = (min1 * norm) >> 4; min2 = (min2 * norm) >> 4;
    if (min1 > 127) min1 = 127;
    if (min2 > 127) min2 = 127;
    for (int e = 0; e < ns; e++) {
      int q = Q[e * zp + i], mag = e == arg ? min2 : min1, r = (sign ^ (q < 0)) ? -mag : mag;
      R[e * zp + i] = (int8_t)r;
      q += r;
      Q[e * zp + i] = (int16_t)(q > 8191 ? 8191 : q < -8191 ? -8191 : q);
    }
  }
#endif
}

#ifdef OB_NEON
// ai: arm64: the layer in NEON's own instructions, 32 checks a step. The two passes run over the slots once a step
// ai: together: pass 2 of step i (messages and posteriors out, the stores) beside pass 1 of step i + 32 (extrinsics,
// ai: their minima and signs, the arithmetic), so the stores of one and the operations of the other share the core;
// ai: one pass alone was bound by either. Pass 1: a message widened and taken off in one instruction (SSUBW), the
// ai: extrinsics kept for pass 2 (vb) with their magnitudes as bytes saturated at 255 (ab), the minima on those bytes
// ai: (UMIN, UMAX), the signs as the XOR of the high bytes. Pass 2: the scaled minima carry the check's sign; a message
// ai: is min1's or min2's (by the byte against min1) under the extrinsic's sign, as bytes, stored as they are and added
// ai: to the extrinsic (SADDW). No index of the least magnitude: an edge whose magnitude is min1 takes min2, which
// ai: decides as the index does (a second edge at min1 leaves min2 equal to min1), so every form decides alike; on the
// ai: saturated bytes alike, as min1 and min2 then scale to 127 both. No clamp at +-8191: a decode's posteriors stay
// ai: inside it (ldpc_init).
// ai: view: where each of the ns slots is read from and written to: slots 0 to nd - 1 in the decoder's four-times
// ai: columns (w4, run at 2z - sh[e]: a step's new posteriors written at the run and a column up below 2z, a column
// ai: down from 2z, the step across 2z a column down too once the layer is done), the rest once where they lie (the
// ai: staircase's rows of wl). A view of Q's own rows with nd 0 is the layer as layer_kernel takes it
// ai: (test/ldpc_neon.c holds the two alike, on posteriors where no clamp acts).
typedef struct { uint8x16_t a1, b1, a2, b2; int8x16_t sa, sb; } p1_acc;
typedef struct { uint8x16_t a1, b1; int8x16_t an, am, bn, bm; } p2_msg;

static inline __attribute__((always_inline)) void p1_slot(p1_acc *x, const int8_t *Rp, const int16_t *src, int16_t *vbp, uint8_t *abp) {
  const int8x16x2_t r = vld1q_s8_x2(Rp);
  int16x8x4_t v = vld1q_s16_x4(src);
  v.val[0] = vsubw_s8(v.val[0], vget_low_s8(r.val[0])); v.val[1] = vsubw_high_s8(v.val[1], r.val[0]);
  v.val[2] = vsubw_s8(v.val[2], vget_low_s8(r.val[1])); v.val[3] = vsubw_high_s8(v.val[3], r.val[1]);
  vst1q_s16_x4(vbp, v);
  const uint8x16x2_t m = { { vqmovn_high_u16(vqmovn_u16(vreinterpretq_u16_s16(vabsq_s16(v.val[0]))), vreinterpretq_u16_s16(vabsq_s16(v.val[1]))),
                             vqmovn_high_u16(vqmovn_u16(vreinterpretq_u16_s16(vabsq_s16(v.val[2]))), vreinterpretq_u16_s16(vabsq_s16(v.val[3]))) } };
  vst1q_u8_x2(abp, m);
  x->sa = veorq_s8(x->sa, vuzp2q_s8(vreinterpretq_s8_s16(v.val[0]), vreinterpretq_s8_s16(v.val[1])));
  x->sb = veorq_s8(x->sb, vuzp2q_s8(vreinterpretq_s8_s16(v.val[2]), vreinterpretq_s8_s16(v.val[3])));
  x->a2 = vminq_u8(x->a2, vmaxq_u8(x->a1, m.val[0])); x->b2 = vminq_u8(x->b2, vmaxq_u8(x->b1, m.val[1]));
  x->a1 = vminq_u8(x->a1, m.val[0]); x->b1 = vminq_u8(x->b1, m.val[1]);
}

// ai: a slot's new messages from its extrinsics, stored; the new posteriors returned
static inline __attribute__((always_inline)) int16x8x4_t p2_slot(const p2_msg *g, int8_t *Rp, const int16_t *vbp, const uint8_t *abp) {
  int16x8x4_t v = vld1q_s16_x4(vbp);
  const uint8x16x2_t m = vld1q_u8_x2(abp);
  const int8x16_t ta = vbslq_s8(vceqq_u8(m.val[0], g->a1), g->am, g->an), tb = vbslq_s8(vceqq_u8(m.val[1], g->b1), g->bm, g->bn);
  const int8x16_t ga = vcltzq_s8(vuzp2q_s8(vreinterpretq_s8_s16(v.val[0]), vreinterpretq_s8_s16(v.val[1])));
  const int8x16_t gb = vcltzq_s8(vuzp2q_s8(vreinterpretq_s8_s16(v.val[2]), vreinterpretq_s8_s16(v.val[3])));
  const int8x16x2_t ro = { { vsubq_s8(veorq_s8(ta, ga), ga), vsubq_s8(veorq_s8(tb, gb), gb) } };
  vst1q_s8_x2(Rp, ro);
  v.val[0] = vaddw_s8(v.val[0], vget_low_s8(ro.val[0])); v.val[1] = vaddw_high_s8(v.val[1], ro.val[0]);
  v.val[2] = vaddw_s8(v.val[2], vget_low_s8(ro.val[1])); v.val[3] = vaddw_high_s8(v.val[3], ro.val[1]);
  return v;
}

static inline __attribute__((always_inline)) void p1_reset(p1_acc *x) {
  x->a1 = x->b1 = x->a2 = x->b2 = vdupq_n_u8(255);
  x->sa = x->sb = vdupq_n_s8(0);
}

// ai: (x * norm) >> 4 capped at 127, on the saturated bytes (alike from x = 157 up at norm 13), the check's sign folded in
static inline __attribute__((always_inline)) p2_msg p2_from(const p1_acc *x, int norm) {
  const uint8x16_t n = vdupq_n_u8((uint8_t)norm), top = vdupq_n_u8(127);
#define OB_SCALE(v) vminq_u8(vshrn_high_n_u16(vshrn_n_u16(vmull_u8(vget_low_u8(v), vget_low_u8(n)), 4), vmull_high_u8((v), n), 4), top)
#define OB_SIGNED(v, s) vsubq_s8(veorq_s8(vreinterpretq_s8_u8(OB_SCALE(v)), (s)), (s))
  const int8x16_t sa = vcltzq_s8(x->sa), sb = vcltzq_s8(x->sb);
  p2_msg g = { x->a1, x->b1, OB_SIGNED(x->a1, sa), OB_SIGNED(x->a2, sa), OB_SIGNED(x->b1, sb), OB_SIGNED(x->b2, sb) };
#undef OB_SIGNED
#undef OB_SCALE
  // ai: opaque to the compiler, which otherwise moves the fold after the selection, into every slot
  __asm__("" : "+w"(g.an), "+w"(g.am), "+w"(g.bn), "+w"(g.bm));
  return g;
}

static void layer_kernel_neon(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh) {
  int16_t vb[2][64 * 32];   // ai: a step's extrinsics, slot by slot, this step's and the next's
  uint8_t ab[2][64 * 32];   // ai: their magnitudes as bytes, saturated at 255
  p1_acc x;
  p1_reset(&x);
  for (int e = 0; e < ns; e++) p1_slot(&x, R + e * zp, view[e], vb[0] + e * 32, ab[0] + e * 32);
  for (int i = 0, b = 0; i < zp; i += 32, b ^= 1) {
    const p2_msg g = p2_from(&x, norm);
    const int16_t *vc = vb[b];
    const uint8_t *ac = ab[b];
    int8_t *Rp = R + i;
    if (i + 32 < zp) {
      int16_t *vn = vb[b ^ 1];
      uint8_t *an = ab[b ^ 1];
      p1_reset(&x);
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        const int16x8x4_t v = p2_slot(&g, Rp, vc, ac);
        p1_slot(&x, Rp + 32, d + 32, vn, an);
        vst1q_s16_x4(d, v); vst1q_s16_x4(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        vst1q_s16_x4(d, p2_slot(&g, Rp, vc, ac));
        p1_slot(&x, Rp + 32, d + 32, vn, an);
      }
    } else {
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32) {
        int16_t *d = view[e] + i;
        const int16x8x4_t v = p2_slot(&g, Rp, vc, ac);
        vst1q_s16_x4(d, v); vst1q_s16_x4(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32) vst1q_s16_x4(view[e] + i, p2_slot(&g, Rp, vc, ac));
    }
  }
  // ai: the run that crosses 2z (s not a multiple of 32) was stored a column up; its part past 2z also a column down
  for (int e = 0; e < nd; e++) if (sh[e] & 31) { int16_t *d = view[e] + (sh[e] & ~31); vst1q_s16_x4(d - z, vld1q_s16_x4(d)); }
}
#endif

// x86-64: the same layer, 16 and 32 checks a vector (AVX2, AVX-512 with its byte and word lanes), each compiled for
// its own instruction set and taken by ldpc_init where the processor has it; lane for lane the 128-bit kernel's
// operations, so every form decides alike (test/ldpc_wide.c holds them to the scalar twin).
#ifdef OB_LDPC_WIDE
#include <immintrin.h>
#ifndef OB_LDPC_WIDE_MAX
#define OB_LDPC_WIDE_MAX 2   // ai: a lab build caps it (1: AVX2 at most, 0: the 128-bit kernel), for an A/B
#endif
__attribute__((target("avx2"))) static void layer_kernel_avx2(int16_t *Q, int8_t *R, int ns, int zp, int norm) {
  const __m256i big = _mm256_set1_epi16(32767), cap = _mm256_set1_epi16(2047), top = _mm256_set1_epi16(127), vn = _mm256_set1_epi16((int16_t)norm);
  const __m256i lo = _mm256_set1_epi16(-8191), hi = _mm256_set1_epi16(8191);
  for (int i = 0; i < zp; i += 16) {
    __m256i min1 = big, min2 = big, arg = _mm256_set1_epi16(-1), sgn = _mm256_setzero_si256();
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      __m256i v = _mm256_sub_epi16(_mm256_loadu_si256((const __m256i *)q), _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(R + e * zp + i))));
      _mm256_storeu_si256((__m256i *)q, v);
      __m256i a = _mm256_abs_epi16(v), lt = _mm256_cmpgt_epi16(min1, a);
      sgn = _mm256_xor_si256(sgn, v);
      min2 = _mm256_min_epi16(min2, _mm256_max_epi16(min1, a));
      arg = _mm256_blendv_epi8(arg, _mm256_set1_epi16((int16_t)e), lt);
      min1 = _mm256_min_epi16(min1, a);
    }
    __m256i m1 = _mm256_min_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_min_epi16(min1, cap), vn), 4), top);
    __m256i m2 = _mm256_min_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_min_epi16(min2, cap), vn), 4), top);
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      __m256i v = _mm256_loadu_si256((const __m256i *)q);
      __m256i mag = _mm256_blendv_epi8(m1, m2, _mm256_cmpeq_epi16(arg, _mm256_set1_epi16((int16_t)e)));
      __m256i neg = _mm256_srai_epi16(_mm256_xor_si256(sgn, v), 15);
      __m256i r = _mm256_sub_epi16(_mm256_xor_si256(mag, neg), neg);
      // ai: the messages to bytes (within +-127, so the saturation never acts); PACKSSWB packs within each 128-bit half,
      // ai: so the halves' first eight bytes are brought together
      __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi16(r, r), 0xD8);
      _mm_storeu_si128((__m128i *)(R + e * zp + i), _mm256_castsi256_si128(p));
      _mm256_storeu_si256((__m256i *)q, _mm256_max_epi16(_mm256_min_epi16(_mm256_add_epi16(v, r), hi), lo));
    }
  }
}
__attribute__((target("avx512f,avx512bw"))) static void layer_kernel_avx512(int16_t *Q, int8_t *R, int ns, int zp, int norm) {
  const __m512i big = _mm512_set1_epi16(32767), cap = _mm512_set1_epi16(2047), top = _mm512_set1_epi16(127), vn = _mm512_set1_epi16((int16_t)norm);
  const __m512i lo = _mm512_set1_epi16(-8191), hi = _mm512_set1_epi16(8191);
  for (int i = 0; i < zp; i += 32) {
    __m512i min1 = big, min2 = big, arg = _mm512_set1_epi16(-1), sgn = _mm512_setzero_si512();
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      __m512i v = _mm512_sub_epi16(_mm512_loadu_si512((const void *)q), _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(R + e * zp + i))));
      _mm512_storeu_si512((void *)q, v);
      __m512i a = _mm512_abs_epi16(v);
      __mmask32 lt = _mm512_cmplt_epi16_mask(a, min1);
      sgn = _mm512_xor_si512(sgn, v);
      min2 = _mm512_min_epi16(min2, _mm512_max_epi16(min1, a));
      arg = _mm512_mask_mov_epi16(arg, lt, _mm512_set1_epi16((int16_t)e));
      min1 = _mm512_min_epi16(min1, a);
    }
    __m512i m1 = _mm512_min_epi16(_mm512_srai_epi16(_mm512_mullo_epi16(_mm512_min_epi16(min1, cap), vn), 4), top);
    __m512i m2 = _mm512_min_epi16(_mm512_srai_epi16(_mm512_mullo_epi16(_mm512_min_epi16(min2, cap), vn), 4), top);
    for (int e = 0; e < ns; e++) {
      int16_t *q = Q + e * zp + i;
      __m512i v = _mm512_loadu_si512((const void *)q);
      __m512i mag = _mm512_mask_mov_epi16(m1, _mm512_cmpeq_epi16_mask(arg, _mm512_set1_epi16((int16_t)e)), m2);
      __m512i neg = _mm512_srai_epi16(_mm512_xor_si512(sgn, v), 15);
      __m512i r = _mm512_sub_epi16(_mm512_xor_si512(mag, neg), neg);
      _mm256_storeu_si256((__m256i *)(R + e * zp + i), _mm512_cvtsepi16_epi8(r));
      _mm512_storeu_si512((void *)q, _mm512_max_epi16(_mm512_min_epi16(_mm512_add_epi16(v, r), hi), lo));
    }
  }
}
// ai: The widest kernel the processor has (and the build allows), its z a whole number of the kernel's vectors.
static int wide_for(int zp) {
  __builtin_cpu_init();
  if (OB_LDPC_WIDE_MAX >= 2 && zp % 32 == 0 && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512f")) return 2;
  if (OB_LDPC_WIDE_MAX >= 1 && zp % 16 == 0 && __builtin_cpu_supports("avx2")) return 1;
  return 0;
}
#endif

static void layer_kernel_for(const ldpc_t *c, int16_t *Q, int8_t *R, int ns) {
#ifdef OB_LDPC_WIDE
  if (c->wide == 2) { layer_kernel_avx512(Q, R, ns, c->zp, c->norm); return; }
  if (c->wide == 1) { layer_kernel_avx2(Q, R, ns, c->zp, c->norm); return; }
#endif
  layer_kernel(Q, R, ns, c->zp, c->norm);
}

// Row r's slots, rotated into check order: data circulants, then the staircase bit shared with the
// check before, then the check's own. Check 0 has no check before it; its slot holds a bit that is
// certainly zero, which is what an absent edge means to min-sum.
static void layer_gather(const ldpc_t *c, int r, int16_t *Q) {
  int z = c->z, zp = c->zp, nd = c->lay_ptr[r + 1] - c->lay_ptr[r];
  const int16_t *Ls = c->wl, *Lp = c->wl + c->k;
  for (int e = 0; e < nd; e++) {
    int s = c->lay_shift[c->lay_ptr[r] + e];
    const int16_t *col = Ls + c->lay_col[c->lay_ptr[r] + e] * z;
    memcpy(Q + e * zp, col + z - s, (size_t)s * 2);
    memcpy(Q + e * zp + s, col, (size_t)(z - s) * 2);
  }
  int16_t *prev = Q + nd * zp;
  if (r > 0) memcpy(prev, Lp + (r - 1) * z, (size_t)z * 2);
  else { prev[0] = 8191; memcpy(prev + 1, Lp + (c->mb - 1) * z, (size_t)(z - 1) * 2); }
  memcpy(Q + (nd + 1) * zp, Lp + r * z, (size_t)z * 2);
}

static void layer_scatter(const ldpc_t *c, int r, const int16_t *Q) {
  int z = c->z, zp = c->zp, nd = c->lay_ptr[r + 1] - c->lay_ptr[r];
  int16_t *Ls = c->wl, *Lp = c->wl + c->k;
  for (int e = 0; e < nd; e++) {
    int s = c->lay_shift[c->lay_ptr[r] + e];
    int16_t *col = Ls + c->lay_col[c->lay_ptr[r] + e] * z;
    memcpy(col + z - s, Q + e * zp, (size_t)s * 2);
    memcpy(col, Q + e * zp + s, (size_t)(z - s) * 2);
  }
  const int16_t *prev = Q + nd * zp;
  if (r > 0) memcpy(Lp + (r - 1) * z, prev, (size_t)z * 2);
  else memcpy(Lp + (c->mb - 1) * z, prev + 1, (size_t)(z - 1) * 2);
  memcpy(Lp + r * z, Q + (nd + 1) * zp, (size_t)z * 2);
}

#ifdef OB_NEON
// ai: arm64: the layer read from and written to the columns where they lie (w4), so no slot is copied into check order
// ai: and back: slot e's rotation by its shift s is the run of z at 2z - s; the two staircase slots are wl's rows in
// ai: place, but check 0's bit before it (the last row moved up a lane under a certain zero, as layer_gather lays it),
// ai: which goes through Q. (The copies cost about as much as the kernel, and a row stored split at s and read back in
// ai: 64-byte pieces across the split waited on both stores.)
// ai: Where every slot of every layer lies, laid out as the messages are (layer r's ns slots from lay_ptr[r] + 2r):
// ai: fixed by the code and the workspace, so made once a decode rather than once a layer.
static void views_neon(const ldpc_t *c, int16_t **vw) {
  const int z = c->z;
  int16_t *Lp = c->wl + c->k;
  for (int r = 0; r < c->mb; r++) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t **v = vw + p0 + 2 * r;
    for (int e = 0; e < nd; e++) v[e] = c->w4 + (size_t)c->lay_col[p0 + e] * 4 * z + 2 * z - c->lay_shift[p0 + e];
    v[nd] = r > 0 ? Lp + (r - 1) * z : c->wq + nd * c->zp;
    v[nd + 1] = Lp + r * z;
  }
}

static void layer_direct(const ldpc_t *c, int r, int8_t *R, int ns, int16_t *const *view) {
  const int z = c->z, nd = ns - 2;
  int16_t *Lp = c->wl + c->k;
  if (r == 0) { view[nd][0] = 8191; memcpy(view[nd] + 1, Lp + (c->mb - 1) * z, (size_t)(z - 1) * 2); }
  layer_kernel_neon(R, ns, c->zp, c->norm, view, nd, z, c->lay_shift + c->lay_ptr[r]);
  if (r == 0) memcpy(Lp + (c->mb - 1) * z, view[nd] + 1, (size_t)(z - 1) * 2);
}
#endif

// The sign of an XOR is the XOR of the signs, so a row's parities are one XOR per slot.
// How many of the m checks are violated: layers_clear without the early way out.
static int layers_unsat(const ldpc_t *c, int16_t *Q) {
  int zp = c->zp, z = c->z, bad = 0;
  for (int r = 0; r < c->mb; r++) {
    int ns = c->lay_ptr[r + 1] - c->lay_ptr[r] + 2;
    layer_gather(c, r, Q);
    if (r == 0) Q[(ns - 2) * zp] = 0;
    for (int e = 1; e < ns; e++) for (int i = 0; i < z; i++) Q[i] ^= Q[e * zp + i];
    for (int i = 0; i < z; i++) bad += Q[i] < 0;
  }
  return bad;
}

static int layers_clear(const ldpc_t *c, int16_t *Q) {
  int zp = c->zp, z = c->z;
  for (int r = 0; r < c->mb; r++) {
    int ns = c->lay_ptr[r + 1] - c->lay_ptr[r] + 2;
    layer_gather(c, r, Q);
    if (r == 0) Q[(ns - 2) * zp] = 0;
    for (int e = 1; e < ns; e++) for (int i = 0; i < z; i++) Q[i] ^= Q[e * zp + i];
    int bad = 0;
    for (int i = 0; i < z; i++) bad |= Q[i];
    if (bad < 0) return 0;
  }
  return 1;
}

#ifdef OB_NEON
// ai: arm64: the checks' parities read where the posteriors lie, nothing gathered: a data slot's rotation is its run in
// ai: the four-times columns (w4), the staircase slots wl's rows (check 0's bit before it the last row moved up a lane,
// ai: a zero in front, as layer_gather lays it); one XOR a slot over 32 checks at a time, only the signs counted.
// ai: count 0: 1 when every check holds, else 0 at the first layer holding a violated one (layers_clear);
// ai: count 1: how many of the m checks are violated (layers_unsat).
static int layers_parity_neon(const ldpc_t *c, int16_t *const *vw, int count) {
  const int z = c->z, mb = c->mb;
  const int16_t *Lp = c->wl + c->k;
  int bad = 0;
  for (int r = 0; r < mb; r++) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t *const *view = vw + p0 + 2 * r;
    const int16_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    uint16x8_t acc = vdupq_n_u16(0);
    int16x8_t carry = vdupq_n_s16(0);
    for (int i = 0; i < z; i += 32) {
      int16x8x4_t x = vld1q_s16_x4(own + i);
      const int16x8x4_t p = vld1q_s16_x4(prev + i);
      if (r > 0) {
        x.val[0] = veorq_s16(x.val[0], p.val[0]); x.val[1] = veorq_s16(x.val[1], p.val[1]);
        x.val[2] = veorq_s16(x.val[2], p.val[2]); x.val[3] = veorq_s16(x.val[3], p.val[3]);
      } else {
        x.val[0] = veorq_s16(x.val[0], vextq_s16(carry, p.val[0], 7)); x.val[1] = veorq_s16(x.val[1], vextq_s16(p.val[0], p.val[1], 7));
        x.val[2] = veorq_s16(x.val[2], vextq_s16(p.val[1], p.val[2], 7)); x.val[3] = veorq_s16(x.val[3], vextq_s16(p.val[2], p.val[3], 7));
        carry = p.val[3];
      }
      for (int e = 0; e < nd; e++) {
        const int16x8x4_t v = vld1q_s16_x4(view[e] + i);
        x.val[0] = veorq_s16(x.val[0], v.val[0]); x.val[1] = veorq_s16(x.val[1], v.val[1]);
        x.val[2] = veorq_s16(x.val[2], v.val[2]); x.val[3] = veorq_s16(x.val[3], v.val[3]);
      }
      const uint16x8_t s0 = vshrq_n_u16(vreinterpretq_u16_s16(x.val[0]), 15), s1 = vshrq_n_u16(vreinterpretq_u16_s16(x.val[1]), 15);
      const uint16x8_t s2 = vshrq_n_u16(vreinterpretq_u16_s16(x.val[2]), 15), s3 = vshrq_n_u16(vreinterpretq_u16_s16(x.val[3]), 15);
      acc = vaddq_u16(acc, vaddq_u16(vaddq_u16(s0, s1), vaddq_u16(s2, s3)));
    }
    const int b = (int)vaddvq_u16(acc);
    if (!count && b) return 0;
    bad += b;
  }
  return count ? bad : 1;
}

// ai: 16 rows of 16 bytes transposed in place: TRN1 and TRN2 on rows t and t + 2^s, at 8, 16, 32 and 64 bits
static inline void transpose16(uint8x16_t x[16]) {
  for (int t = 0; t < 16; t += 2) { const uint8x16_t a = x[t], b = x[t + 1]; x[t] = vtrn1q_u8(a, b); x[t + 1] = vtrn2q_u8(a, b); }
  for (int t = 0; t < 16; t++) if (!(t & 2)) {
    const uint16x8_t a = vreinterpretq_u16_u8(x[t]), b = vreinterpretq_u16_u8(x[t + 2]);
    x[t] = vreinterpretq_u8_u16(vtrn1q_u16(a, b)); x[t + 2] = vreinterpretq_u8_u16(vtrn2q_u16(a, b));
  }
  for (int t = 0; t < 16; t++) if (!(t & 4)) {
    const uint32x4_t a = vreinterpretq_u32_u8(x[t]), b = vreinterpretq_u32_u8(x[t + 4]);
    x[t] = vreinterpretq_u8_u32(vtrn1q_u32(a, b)); x[t + 4] = vreinterpretq_u8_u32(vtrn2q_u32(a, b));
  }
  for (int t = 0; t < 8; t++) {
    const uint64x2_t a = vreinterpretq_u64_u8(x[t]), b = vreinterpretq_u64_u8(x[t + 8]);
    x[t] = vreinterpretq_u8_u64(vtrn1q_u64(a, b)); x[t + 8] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
  }
}

// ai: The parity bits between code order (check i * mb + r, a row of mb a position) and wl's [block row][position],
// ai: 16 positions by 16 block rows at a time (a code of 16 rows or more: the last 16 of them where fewer remain, two
// ai: rows written alike). With fewer than 16 rows a position's 16-byte row runs into the next positions', which the
// ai: next row's store puts right; the last 16 positions go through a buffer, so nothing is read or written past m.
static void parity_in(const int8_t *src, int16_t *dst, int z, int mb) {
  uint8_t tail[16 * 16 + 16] = { 0 };
  for (int i0 = 0; i0 < z; i0 += 16) {
    const uint8_t *s = (const uint8_t *)src + i0 * mb;
    if (mb < 16 && i0 + 16 == z) { memcpy(tail, s, (size_t)16 * mb); s = tail; }
    for (int r0 = 0; r0 < mb; r0 += 16) {
      const int c0 = r0 + 16 <= mb ? r0 : mb >= 16 ? mb - 16 : 0, nc = mb - c0 < 16 ? mb - c0 : 16;
      uint8x16_t x[16];
      for (int t = 0; t < 16; t++) x[t] = vld1q_u8(s + t * mb + c0);
      transpose16(x);
      for (int r = 0; r < nc; r++) {
        const int8x16_t b = vreinterpretq_s8_u8(x[r]);
        const int16x8x2_t v = { { vmovl_s8(vget_low_s8(b)), vmovl_high_s8(b) } };
        vst1q_s16_x2(dst + (c0 + r) * z + i0, v);
      }
    }
  }
}

static void parity_out(const int16_t *src, uint8_t *dst, int z, int mb) {
  uint8_t tail[16 * 16 + 16];
  for (int i0 = 0; i0 < z; i0 += 16) {
    const int last = mb < 16 && i0 + 16 == z;
    uint8_t *d = last ? tail : dst + i0 * mb;
    for (int r0 = 0; r0 < mb; r0 += 16) {
      const int c0 = r0 + 16 <= mb ? r0 : mb >= 16 ? mb - 16 : 0, nc = mb - c0 < 16 ? mb - c0 : 16;
      uint8x16_t x[16];
      for (int r = 0; r < 16; r++) x[r] = r < nc ? vshrq_n_u8(vld2q_u8((const uint8_t *)(src + (c0 + r) * z + i0)).val[1], 7) : vdupq_n_u8(0);
      transpose16(x);
      for (int t = 0; t < 16; t++) vst1q_u8(d + t * mb + c0, x[t]);
    }
    if (last) memcpy(dst + i0 * mb, tail, (size_t)16 * mb);
  }
}
#endif

int ldpc_decode(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter) { return ldpc_decode_stall(c, llr, out, max_iter, 0, 0); }

// stall_it > 0: a block that will never decode is let go early. Such a block makes no headway at all: its count of
// violated checks stays where the first iteration left it, while one that is merely slow has brought it down. So
// the count after iteration stall_it is held against the count after the first, and the block is given up (-3) if
// it is still above stall_ratio of it. The two counts replace those iterations' cleared test, so they cost nothing.
int ldpc_decode_stall(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter, int stall_it, float stall_ratio) {
  int z = c->z, zp = c->zp, mb = c->mb, used = -1, first = 0;
  int16_t *L = c->wl, *Q = c->wq;
#ifdef OB_NEON
  const int four = c->w4 && c->wide != -1;
  int16_t *vw[LDPC_VIEWS];
  if (four) {
    views_neon(c, vw);
    // ai: each data column's z LLRs widened once and written into the middle two of its four copies, the only ones a
    // ai: run reads ([z, 3z): the outer two only take the stores either side of a run); wl's data part goes unused.
    for (int j = 0; j < c->kb; j++) {
      int16_t *d = c->w4 + (size_t)j * 4 * z + z;
      for (int t = 0; t < z; t += 16) {
        const int8x16_t b = vld1q_s8(llr + j * z + t);
        const int16x8x2_t v = { { vmovl_s8(vget_low_s8(b)), vmovl_high_s8(b) } };
        vst1q_s16_x2(d + t, v); vst1q_s16_x2(d + z + t, v);
      }
    }
    parity_in(llr + c->k, L + c->k, z, mb);
  } else
#endif
  {
    for (int v = 0; v < c->k; v++) L[v] = llr[v];
    for (int chk = 0; chk < c->m; chk++) L[c->k + (chk % mb) * z + chk / mb] = llr[c->k + chk];
  }
  memset(c->wr, 0, (size_t)c->slots_total * zp);
  for (int it = 0; it < max_iter; it++) {
    for (int step = 0; step < mb; step++) {
      int r = (it & 1) ? mb - 1 - step : step, ns = c->lay_ptr[r + 1] - c->lay_ptr[r] + 2;
      int8_t *R = c->wr + (size_t)(c->lay_ptr[r] + 2 * r) * zp;
      if (r == 0) R[(ns - 2) * zp] = 0;
#ifdef OB_NEON
      if (four) { layer_direct(c, r, R, ns, vw + c->lay_ptr[r] + 2 * r); continue; }
#endif
      layer_gather(c, r, Q);
      layer_kernel_for(c, Q, R, ns);
      layer_scatter(c, r, Q);
    }
#ifdef OB_NEON
    if (four) {
      if (stall_it > 0 && (it == 0 || it == stall_it - 1)) {
        const int bad = layers_parity_neon(c, vw, 1);
        if (!bad) { used = it + 1; break; }
        if (it == 0) first = bad;
        else if ((float)bad > stall_ratio * (float)first) { used = -3; break; }
      } else if (layers_parity_neon(c, vw, 0)) { used = it + 1; break; }
      continue;
    }
#endif
    if (stall_it > 0 && (it == 0 || it == stall_it - 1)) {
      int bad = layers_unsat(c, Q);
      if (!bad) { used = it + 1; break; }
      if (it == 0) first = bad;
      else if ((float)bad > stall_ratio * (float)first) { used = -3; break; }
    } else if (layers_clear(c, Q)) { used = it + 1; break; }
  }
#ifdef OB_NEON
  if (four) {
    // ai: the data from w4's second copy, a sign bit to a byte
    for (int j = 0; j < c->kb; j++) {
      const int16_t *col = c->w4 + (size_t)j * 4 * z + z;
      for (int t = 0; t < z; t += 16) {
        const uint8x16x2_t b = vld2q_u8((const uint8_t *)(col + t));
        vst1q_u8(out + j * z + t, vshrq_n_u8(b.val[1], 7));
      }
    }
    parity_out(L + c->k, out + c->k, z, mb);
    return used;
  }
#endif
  for (int v = 0; v < c->k; v++) out[v] = L[v] < 0;
  for (int chk = 0; chk < c->m; chk++) out[c->k + chk] = L[c->k + (chk % mb) * z + chk / mb] < 0;
  return used;
}

// The code's shape and its block-row tables, for a decoder running somewhere else (gpu/back/ldpc.mjs).
// dims: n, k, m, z, zp, mb, kb, norm, slots_max, slots_total, edges. lay: lay_ptr[mb + 1], then lay_col and
// lay_shift interleaved, one pair per slot of lay_ptr[mb]. Pass lay = 0 to size the buffers first.
void ldpc_tables(const ldpc_t *c, int32_t *dims, int32_t *lay) {
  dims[0] = c->n; dims[1] = c->k; dims[2] = c->m; dims[3] = c->z; dims[4] = c->zp;
  dims[5] = c->mb; dims[6] = c->kb; dims[7] = c->norm; dims[8] = c->slots_max; dims[9] = c->slots_total;
  dims[10] = c->lay_ptr[c->mb]; dims[11] = c->np;
  if (!lay) return;
  for (int r = 0; r <= c->mb; r++) lay[r] = c->lay_ptr[r];
  for (int e = 0; e < c->lay_ptr[c->mb]; e++) { lay[c->mb + 1 + 2 * e] = c->lay_col[e]; lay[c->mb + 2 + 2 * e] = c->lay_shift[e]; }
}
