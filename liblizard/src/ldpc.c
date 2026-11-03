#include "ldpc.h"
#include "ldpc_base.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>

// ai: arm64's decode keeps the first iteration's hard decisions in this many bytes (2 kb z + m, ldpc_decode_stall; the
// ai: 1/2 code's 11,520)
enum { LDPC_SIGNS = 16384 };
// ai: OB_LDPC_W4: the decode on the data columns held four times over (ldpc_t w4): arm64's, in NEON's own instructions,
// ai: and x86-64's port of it (SSE4.1, and AVX2 where the processor has it)
#if defined(OB_NEON) || defined(OB_X86)
#define OB_LDPC_W4 1
static int views_neon(ldpc_t *c);
#endif

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
#ifdef OB_LDPC_W4
  // ai: arm64's kernel (and x86-64's port) leaves out the posteriors' clamp at 8191, which never acts while a posterior,
  // ai: its LLR (within +-128) plus its column's messages (within +-127 each), stays inside it: the largest column
  // ai: degree at most 63
  int dmax = 0;
  for (int j = 0; j < kb; j++) { int d = 0; for (int r = 0; r < mb; r++) d += shift[r * kb + j] >= 0; if (d > dmax) dmax = d; }
  if (c->z % 32 == 0 && c->zp == c->z && c->slots_max <= 64 && 2 * c->k + c->m <= LDPC_SIGNS && 128 + 127 * dmax <= 8191 && c->norm >= 8) {
    // ai: the columns, then the tables views_neon makes of them
    c->w4 = malloc((size_t)c->kb * 4 * c->z * sizeof(int16_t) + ((size_t)c->slots_total + c->lay_ptr[mb]) * sizeof(int16_t *) + (size_t)mb * sizeof(int));
    if (!c->w4 || views_neon(c)) { ldpc_free(c); return -1; }
#ifdef OB_LDPC_WIDE
    // ai: x86-64: wide is then the four-times kernel's form: 0 SSE4.1, 1 AVX2 (where wide_for took AVX2 or AVX-512;
    // ai: the kernel has no wider form)
    if (c->wide) c->wide = __builtin_cpu_supports("avx2") ? 1 : 0;
#endif
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
// ai: extrinsics kept for pass 2 (vb) and narrowed to signed bytes saturated at -128 and 127 (SQXTN; ab), whose XOR is
// ai: the check's sign and whose magnitudes (SQABS, at most 127) take the minima (UMIN, UMAX). Pass 2: the scaled minima
// ai: carry the check's sign; a message is min1's or min2's (by the byte's magnitude against min1) under the byte's
// ai: sign, as bytes, stored as they are and added to the extrinsic (SADDW). No index of the least magnitude: an edge
// ai: whose magnitude is min1 takes min2, which decides as the index does (a second edge at min1 leaves min2 equal to
// ai: min1). The magnitudes held at 127 (the 128-bit kernel's and the scalar loop's reach 2047) change a message only
// ai: where a check's two least reach 128: 103 at most ((127 x 13) >> 4) where theirs reach 127 (from 157 at norm 13).
// ai: Such a check is settled, its signs and its edges' signs alike either way; every decision measured is the 128-bit
// ai: kernel's (test/ldpc_neon.c's 4,000 decodes; the 16 phone runs' digests). No clamp at +-8191: a decode's
// ai: posteriors stay inside it (ldpc_init).
// ai: view: where each of the ns slots is read from and written to: slots 0 to nd - 1 in the decoder's four-times
// ai: columns (w4, run at 2z - sh[e]: a step's new posteriors written at the run and a column up below 2z, a column
// ai: down from 2z, the step across 2z a column down too once the layer is done where the nc pointers of cp list it),
// ai: the rest once where they lie (the staircase's rows of wl). A view of Q's own rows with nd 0 is the layer as
// ai: layer_kernel takes it (test/ldpc_neon.c holds the two alike, on posteriors where no clamp acts).
// ai: first: the decode's first iteration, whose messages are all zero (none sent yet), so an extrinsic is its posterior
// ai: as it lies: pass 1 neither reads the messages nor keeps the extrinsics aside, and pass 2 reads them from the slot
// ai: itself, which nothing in the layer writes before pass 2 does (a layer's slots are distinct columns and rows, and
// ai: a step's second store lands a column away, outside the run); the messages are written whole by pass 2, so the
// ai: decode clears none. The same values as with zero messages.
typedef struct { uint8x16_t a1, b1, a2, b2; int8x16_t sa, sb; } p1_acc;
typedef struct { uint8x16_t a1, b1; int8x16_t an, am, bn, bm; } p2_msg;

static inline __attribute__((always_inline)) void p1_slot(p1_acc *x, const int8_t *Rp, const int16_t *src, int16_t *vbp, uint8_t *abp, const int first) {
  int16x8x4_t v = vld1q_s16_x4(src);
  if (!first) {
    const int8x16x2_t r = vld1q_s8_x2(Rp);
    v.val[0] = vsubw_s8(v.val[0], vget_low_s8(r.val[0])); v.val[1] = vsubw_high_s8(v.val[1], r.val[0]);
    v.val[2] = vsubw_s8(v.val[2], vget_low_s8(r.val[1])); v.val[3] = vsubw_high_s8(v.val[3], r.val[1]);
    vst1q_s16_x4(vbp, v);
  }
  const int8x16x2_t n = { { vqmovn_high_s16(vqmovn_s16(v.val[0]), v.val[1]), vqmovn_high_s16(vqmovn_s16(v.val[2]), v.val[3]) } };
  vst1q_s8_x2((int8_t *)abp, n);
  const uint8x16x2_t m = { { vreinterpretq_u8_s8(vqabsq_s8(n.val[0])), vreinterpretq_u8_s8(vqabsq_s8(n.val[1])) } };
  x->sa = veorq_s8(x->sa, n.val[0]);
  x->sb = veorq_s8(x->sb, n.val[1]);
  x->a2 = vminq_u8(x->a2, vmaxq_u8(x->a1, m.val[0])); x->b2 = vminq_u8(x->b2, vmaxq_u8(x->b1, m.val[1]));
  x->a1 = vminq_u8(x->a1, m.val[0]); x->b1 = vminq_u8(x->b1, m.val[1]);
}

// ai: a slot's new messages from its extrinsics, stored; the new posteriors returned
static inline __attribute__((always_inline)) int16x8x4_t p2_slot(const p2_msg *g, int8_t *Rp, const int16_t *vbp, const uint8_t *abp) {
  int16x8x4_t v = vld1q_s16_x4(vbp);
  const int8x16x2_t n = vld1q_s8_x2((const int8_t *)abp);
  const uint8x16_t ma = vreinterpretq_u8_s8(vqabsq_s8(n.val[0])), mb = vreinterpretq_u8_s8(vqabsq_s8(n.val[1]));
  const int8x16_t ta = vbslq_s8(vceqq_u8(ma, g->a1), g->am, g->an), tb = vbslq_s8(vceqq_u8(mb, g->b1), g->bm, g->bn);
  const int8x16_t ga = vcltzq_s8(n.val[0]), gb = vcltzq_s8(n.val[1]);
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

// ai: (x * norm) >> 4 capped at 127 (x at most 127 here: 103 at most at norm 13), the check's sign folded in
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

static inline __attribute__((always_inline)) void layer_neon(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, const int first) {
  int16_t vb[2][64 * 32];   // ai: a step's extrinsics, slot by slot, this step's and the next's
  uint8_t ab[2][64 * 32];   // ai: them as signed bytes, saturated at -128 and 127
  p1_acc x;
  p1_reset(&x);
  for (int e = 0; e < ns; e++) p1_slot(&x, R + e * zp, view[e], vb[0] + e * 32, ab[0] + e * 32, first);
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
        const int16x8x4_t v = p2_slot(&g, Rp, first ? d : vc, ac);
        p1_slot(&x, Rp + 32, d + 32, vn, an, first);
        vst1q_s16_x4(d, v); vst1q_s16_x4(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        vst1q_s16_x4(d, p2_slot(&g, Rp, first ? d : vc, ac));
        p1_slot(&x, Rp + 32, d + 32, vn, an, first);
      }
    } else {
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32) {
        int16_t *d = view[e] + i;
        const int16x8x4_t v = p2_slot(&g, Rp, first ? d : vc, ac);
        vst1q_s16_x4(d, v); vst1q_s16_x4(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32) { int16_t *d = view[e] + i; vst1q_s16_x4(d, p2_slot(&g, Rp, first ? d : vc, ac)); }
    }
  }
  // ai: the runs that cross 2z (s not a multiple of 32) were stored a column up; the part past 2z also a column down
  // ai: where it is read from there (cp: the steps across 2z the layer copies, views_neon)
  for (int k = 0; k < nc; k++) vst1q_s16_x4(cp[k] - z, vld1q_s16_x4(cp[k]));
}

static void layer_kernel_neon(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, int first) {
  if (first) layer_neon(R, ns, zp, norm, view, nd, z, sh, cp, nc, 1);
  else layer_neon(R, ns, zp, norm, view, nd, z, sh, cp, nc, 0);
}
#elif defined(OB_X86)
// ai: x86-64: arm64's layer kernel above, ported: steps of 32 checks, pass 2 of step i fused with pass 1 of step i + 32,
// ai: the extrinsics double-buffered and narrowed once to signed bytes, minima by unsigned byte min and max, messages
// ai: as signed bytes, two column stores a step, no clamp at +-8191, the same views and first-iteration form; every
// ai: value NEON's (test/ldpc_neon.c). Two forms: SSE4.1, the baseline (32 checks four registers of eight, their bytes
// ai: two of sixteen), and AVX2 (two of sixteen, the bytes one register; compiled for AVX2 alone, ldpc_t wide 1).
// ai: Where the instructions differ, each gives NEON's values: PACKSSWB saturates as SQXTN, and the bytes are then held
// ai: at -127 (PMAXSB), so PABSB gives SQABS's magnitudes (at most 127; the sign and magnitude, all pass 2 reads of a
// ai: byte, unchanged); a message takes its byte's sign by PSIGNB against the byte with its low bit set (never zero,
// ai: the same sign), as NEON's XOR and subtract of the sign mask; the minima are scaled on 16-bit lanes and packed back
// ai: with unsigned saturation, NEON's narrowing while (127 norm) >> 4 fits a byte (norm at most 32; 13 at every rate).
typedef struct { __m128i a1, b1, a2, b2, sa, sb; } p1_acc_sse41;
typedef struct { __m128i a1, b1, an, am, bn, bm; } p2_msg_sse41;
typedef struct { __m128i v[4]; } q32_sse41;   // ai: 32 posteriors

static inline __attribute__((always_inline)) q32_sse41 ld32_sse41(const int16_t *p) {
  const q32_sse41 v = { { _mm_loadu_si128((const __m128i *)p), _mm_loadu_si128((const __m128i *)(p + 8)), _mm_loadu_si128((const __m128i *)(p + 16)), _mm_loadu_si128((const __m128i *)(p + 24)) } };
  return v;
}
static inline __attribute__((always_inline)) void st32_sse41(int16_t *p, const q32_sse41 v) {
  _mm_storeu_si128((__m128i *)p, v.v[0]); _mm_storeu_si128((__m128i *)(p + 8), v.v[1]);
  _mm_storeu_si128((__m128i *)(p + 16), v.v[2]); _mm_storeu_si128((__m128i *)(p + 24), v.v[3]);
}

static inline __attribute__((always_inline)) void p1_slot_sse41(p1_acc_sse41 *x, const int8_t *Rp, const int16_t *src, int16_t *vbp, uint8_t *abp, const int first) {
  q32_sse41 v = ld32_sse41(src);
  if (!first) {
    for (int u = 0; u < 4; u++) v.v[u] = _mm_sub_epi16(v.v[u], _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(Rp + 8 * u))));
    st32_sse41(vbp, v);
  }
  const __m128i lo = _mm_set1_epi8(-127);
  const __m128i na = _mm_max_epi8(_mm_packs_epi16(v.v[0], v.v[1]), lo), nb = _mm_max_epi8(_mm_packs_epi16(v.v[2], v.v[3]), lo);
  _mm_storeu_si128((__m128i *)abp, na); _mm_storeu_si128((__m128i *)(abp + 16), nb);
  const __m128i ma = _mm_abs_epi8(na), mb = _mm_abs_epi8(nb);
  x->sa = _mm_xor_si128(x->sa, na);
  x->sb = _mm_xor_si128(x->sb, nb);
  x->a2 = _mm_min_epu8(x->a2, _mm_max_epu8(x->a1, ma)); x->b2 = _mm_min_epu8(x->b2, _mm_max_epu8(x->b1, mb));
  x->a1 = _mm_min_epu8(x->a1, ma); x->b1 = _mm_min_epu8(x->b1, mb);
}

// ai: a slot's new messages from its extrinsics, stored; the new posteriors returned
static inline __attribute__((always_inline)) q32_sse41 p2_slot_sse41(const p2_msg_sse41 *g, int8_t *Rp, const int16_t *vbp, const uint8_t *abp) {
  q32_sse41 v = ld32_sse41(vbp);
  const __m128i na = _mm_loadu_si128((const __m128i *)abp), nb = _mm_loadu_si128((const __m128i *)(abp + 16)), one = _mm_set1_epi8(1);
  const __m128i ta = _mm_blendv_epi8(g->an, g->am, _mm_cmpeq_epi8(_mm_abs_epi8(na), g->a1));
  const __m128i tb = _mm_blendv_epi8(g->bn, g->bm, _mm_cmpeq_epi8(_mm_abs_epi8(nb), g->b1));
  const __m128i ra = _mm_sign_epi8(ta, _mm_or_si128(na, one)), rb = _mm_sign_epi8(tb, _mm_or_si128(nb, one));
  _mm_storeu_si128((__m128i *)Rp, ra); _mm_storeu_si128((__m128i *)(Rp + 16), rb);
  v.v[0] = _mm_add_epi16(v.v[0], _mm_cvtepi8_epi16(ra)); v.v[1] = _mm_add_epi16(v.v[1], _mm_cvtepi8_epi16(_mm_srli_si128(ra, 8)));
  v.v[2] = _mm_add_epi16(v.v[2], _mm_cvtepi8_epi16(rb)); v.v[3] = _mm_add_epi16(v.v[3], _mm_cvtepi8_epi16(_mm_srli_si128(rb, 8)));
  return v;
}

static inline __attribute__((always_inline)) void p1_reset_sse41(p1_acc_sse41 *x) {
  x->a1 = x->b1 = x->a2 = x->b2 = _mm_set1_epi8(-1);
  x->sa = x->sb = _mm_setzero_si128();
}

// ai: (x * norm) >> 4 capped at 127, the check's sign folded in
static inline __attribute__((always_inline)) p2_msg_sse41 p2_from_sse41(const p1_acc_sse41 *x, int norm) {
  const __m128i zero = _mm_setzero_si128(), n = _mm_set1_epi16((int16_t)norm), top = _mm_set1_epi8(127), one = _mm_set1_epi8(1);
#define OB_SCALE(v) _mm_min_epu8(_mm_packus_epi16(_mm_srli_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8((v), zero), n), 4), _mm_srli_epi16(_mm_mullo_epi16(_mm_unpackhi_epi8((v), zero), n), 4)), top)
  const __m128i sa = _mm_or_si128(x->sa, one), sb = _mm_or_si128(x->sb, one);
  p2_msg_sse41 g = { x->a1, x->b1, _mm_sign_epi8(OB_SCALE(x->a1), sa), _mm_sign_epi8(OB_SCALE(x->a2), sa), _mm_sign_epi8(OB_SCALE(x->b1), sb), _mm_sign_epi8(OB_SCALE(x->b2), sb) };
#undef OB_SCALE
  // ai: opaque to the compiler, as on arm64, so the fold stays out of the slots
  __asm__("" : "+x"(g.an), "+x"(g.am), "+x"(g.bn), "+x"(g.bm));
  return g;
}

static inline __attribute__((always_inline)) void layer_sse41(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, const int first) {
  int16_t vb[2][64 * 32];   // ai: a step's extrinsics, slot by slot, this step's and the next's
  uint8_t ab[2][64 * 32];   // ai: them as signed bytes, saturated at -128 and 127, then held at -127
  p1_acc_sse41 x;
  p1_reset_sse41(&x);
  for (int e = 0; e < ns; e++) p1_slot_sse41(&x, R + e * zp, view[e], vb[0] + e * 32, ab[0] + e * 32, first);
  for (int i = 0, b = 0; i < zp; i += 32, b ^= 1) {
    const p2_msg_sse41 g = p2_from_sse41(&x, norm);
    const int16_t *vc = vb[b];
    const uint8_t *ac = ab[b];
    int8_t *Rp = R + i;
    if (i + 32 < zp) {
      int16_t *vn = vb[b ^ 1];
      uint8_t *an = ab[b ^ 1];
      p1_reset_sse41(&x);
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        const q32_sse41 v = p2_slot_sse41(&g, Rp, first ? d : vc, ac);
        p1_slot_sse41(&x, Rp + 32, d + 32, vn, an, first);
        st32_sse41(d, v); st32_sse41(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        st32_sse41(d, p2_slot_sse41(&g, Rp, first ? d : vc, ac));
        p1_slot_sse41(&x, Rp + 32, d + 32, vn, an, first);
      }
    } else {
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32) {
        int16_t *d = view[e] + i;
        const q32_sse41 v = p2_slot_sse41(&g, Rp, first ? d : vc, ac);
        st32_sse41(d, v); st32_sse41(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32) { int16_t *d = view[e] + i; st32_sse41(d, p2_slot_sse41(&g, Rp, first ? d : vc, ac)); }
    }
  }
  // ai: the runs past 2z copied a column down where read from there, as on arm64
  for (int k = 0; k < nc; k++) st32_sse41(cp[k] - z, ld32_sse41(cp[k]));
}

static void layer_kernel_four_sse41(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, int first) {
  if (first) layer_sse41(R, ns, zp, norm, view, nd, z, sh, cp, nc, 1);
  else layer_sse41(R, ns, zp, norm, view, nd, z, sh, cp, nc, 0);
}

#ifdef OB_LDPC_WIDE
#include <immintrin.h>
// ai: The AVX2 form: a step's 32 bytes one register. PACKSSWB packs within each 128-bit half, so the bytes come in the
// ai: order of checks 0-7, 16-23, 8-15, 24-31, which the accumulators, the extrinsics' bytes and the scaled minima
// ai: keep (byte for byte, lane for lane); the messages alone are put back in order (VPERMQ), for their store and to
// ai: be widened onto the posteriors.
#define OB_AVX2 static inline __attribute__((always_inline, target("avx2")))
typedef struct { __m256i a1, a2, s; } p1_acc_avx2;
typedef struct { __m256i a1, n, m; } p2_msg_avx2;
typedef struct { __m256i v[2]; } q32_avx2;

OB_AVX2 q32_avx2 ld32_avx2(const int16_t *p) {
  const q32_avx2 v = { { _mm256_loadu_si256((const __m256i *)p), _mm256_loadu_si256((const __m256i *)(p + 16)) } };
  return v;
}
OB_AVX2 void st32_avx2(int16_t *p, const q32_avx2 v) { _mm256_storeu_si256((__m256i *)p, v.v[0]); _mm256_storeu_si256((__m256i *)(p + 16), v.v[1]); }

OB_AVX2 void p1_slot_avx2(p1_acc_avx2 *x, const int8_t *Rp, const int16_t *src, int16_t *vbp, uint8_t *abp, const int first) {
  q32_avx2 v = ld32_avx2(src);
  if (!first) {
    v.v[0] = _mm256_sub_epi16(v.v[0], _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)Rp)));
    v.v[1] = _mm256_sub_epi16(v.v[1], _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(Rp + 16))));
    st32_avx2(vbp, v);
  }
  const __m256i n = _mm256_max_epi8(_mm256_packs_epi16(v.v[0], v.v[1]), _mm256_set1_epi8(-127));
  _mm256_storeu_si256((__m256i *)abp, n);
  const __m256i m = _mm256_abs_epi8(n);
  x->s = _mm256_xor_si256(x->s, n);
  x->a2 = _mm256_min_epu8(x->a2, _mm256_max_epu8(x->a1, m));
  x->a1 = _mm256_min_epu8(x->a1, m);
}

OB_AVX2 q32_avx2 p2_slot_avx2(const p2_msg_avx2 *g, int8_t *Rp, const int16_t *vbp, const uint8_t *abp) {
  q32_avx2 v = ld32_avx2(vbp);
  const __m256i n = _mm256_loadu_si256((const __m256i *)abp);
  const __m256i t = _mm256_blendv_epi8(g->n, g->m, _mm256_cmpeq_epi8(_mm256_abs_epi8(n), g->a1));
  const __m256i r = _mm256_permute4x64_epi64(_mm256_sign_epi8(t, _mm256_or_si256(n, _mm256_set1_epi8(1))), 0xD8);
  _mm256_storeu_si256((__m256i *)Rp, r);
  v.v[0] = _mm256_add_epi16(v.v[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r)));
  v.v[1] = _mm256_add_epi16(v.v[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r, 1)));
  return v;
}

OB_AVX2 void p1_reset_avx2(p1_acc_avx2 *x) {
  x->a1 = x->a2 = _mm256_set1_epi8(-1);
  x->s = _mm256_setzero_si256();
}

// ai: the scaling's unpacks and pack work within each half, so the bytes keep their order
OB_AVX2 p2_msg_avx2 p2_from_avx2(const p1_acc_avx2 *x, int norm) {
  const __m256i zero = _mm256_setzero_si256(), n = _mm256_set1_epi16((int16_t)norm), top = _mm256_set1_epi8(127);
#define OB_SCALE(v) _mm256_min_epu8(_mm256_packus_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(_mm256_unpacklo_epi8((v), zero), n), 4), _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_unpackhi_epi8((v), zero), n), 4)), top)
  const __m256i s = _mm256_or_si256(x->s, _mm256_set1_epi8(1));
  p2_msg_avx2 g = { x->a1, _mm256_sign_epi8(OB_SCALE(x->a1), s), _mm256_sign_epi8(OB_SCALE(x->a2), s) };
#undef OB_SCALE
  __asm__("" : "+x"(g.n), "+x"(g.m));
  return g;
}

OB_AVX2 void layer_avx2(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, const int first) {
  int16_t vb[2][64 * 32];
  uint8_t ab[2][64 * 32];
  p1_acc_avx2 x;
  p1_reset_avx2(&x);
  for (int e = 0; e < ns; e++) p1_slot_avx2(&x, R + e * zp, view[e], vb[0] + e * 32, ab[0] + e * 32, first);
  for (int i = 0, b = 0; i < zp; i += 32, b ^= 1) {
    const p2_msg_avx2 g = p2_from_avx2(&x, norm);
    const int16_t *vc = vb[b];
    const uint8_t *ac = ab[b];
    int8_t *Rp = R + i;
    if (i + 32 < zp) {
      int16_t *vn = vb[b ^ 1];
      uint8_t *an = ab[b ^ 1];
      p1_reset_avx2(&x);
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        const q32_avx2 v = p2_slot_avx2(&g, Rp, first ? d : vc, ac);
        p1_slot_avx2(&x, Rp + 32, d + 32, vn, an, first);
        st32_avx2(d, v); st32_avx2(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32, vn += 32, an += 32) {
        int16_t *d = view[e] + i;
        st32_avx2(d, p2_slot_avx2(&g, Rp, first ? d : vc, ac));
        p1_slot_avx2(&x, Rp + 32, d + 32, vn, an, first);
      }
    } else {
      for (int e = 0; e < nd; e++, Rp += zp, vc += 32, ac += 32) {
        int16_t *d = view[e] + i;
        const q32_avx2 v = p2_slot_avx2(&g, Rp, first ? d : vc, ac);
        st32_avx2(d, v); st32_avx2(d + (i < sh[e] ? z : -z), v);
      }
      for (int e = nd; e < ns; e++, Rp += zp, vc += 32, ac += 32) { int16_t *d = view[e] + i; st32_avx2(d, p2_slot_avx2(&g, Rp, first ? d : vc, ac)); }
    }
  }
  for (int k = 0; k < nc; k++) st32_avx2(cp[k] - z, ld32_avx2(cp[k]));
}

__attribute__((target("avx2"))) static void layer_kernel_four_avx2(int8_t *R, int ns, int zp, int norm, int16_t *const *view, int nd, int z, const int *sh, int16_t *const *cp, int nc, int first) {
  if (first) layer_avx2(R, ns, zp, norm, view, nd, z, sh, cp, nc, 1);
  else layer_avx2(R, ns, zp, norm, view, nd, z, sh, cp, nc, 0);
}
#endif
#endif

// x86-64: the same layer, 16 and 32 checks a vector (AVX2, AVX-512 with its byte and word lanes), each compiled for
// its own instruction set and taken by ldpc_init where the processor has it; lane for lane the 128-bit kernel's
// operations, so every form decides alike (test/ldpc_wide.c holds them to the scalar twin).
// ai: A code ldpc_init gives w4 runs on the four-times columns (x86-64's port above); these run a code it gives none.
#ifdef OB_LDPC_WIDE
#include <immintrin.h>
#ifndef OB_LDPC_WIDE_MAX
#define OB_LDPC_WIDE_MAX 2   // ai: a lab build caps it (1: AVX2 at most, 0: the 128-bit kernel or the four-times SSE4.1 form), for an A/B
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

#ifdef OB_LDPC_W4
// ai: arm64 (and x86-64's port, on the same views): the layer read from and written to the columns where they lie
// ai: (w4), so no slot is copied into check order and back: slot e's rotation by its shift s is the run of z at 2z - s;
// ai: the two staircase slots are wl's rows in place, but check 0's bit before it (the last row moved up a lane under
// ai: a certain zero, as layer_gather lays it), which goes through Q. (The copies cost about as much as the kernel, and
// ai: a row stored split at s and read back in 64-byte pieces across the split waited on both stores.)
// ai: Where every slot of every layer lies (w4_views), laid out as the messages are (layer r's ns slots from
// ai: lay_ptr[r] + 2r): fixed by the code and its workspace, so made by ldpc_init, in w4's allocation past the columns.
// ai: w4_copies, w4_ncopies: the steps across 2z each layer copies a column down (layer r's w4_ncopies[r] from
// ai: w4_copies + lay_ptr[r]). A slot of shift s = 32 q + a (a > 0) writes elements 0 to 31 - a of its column past 2z
// ai: in its step across, and the copy below 2z is stale until copied; a run of shift s' reads element t below 2z where
// ai: t >= z - s', so only a slot of shift z - 31 + a or more in the same column (never the slot itself: s <= z - 32 +
// ai: a) reads the stale part, and the copy is needed only where the column has one. The copy past 2z is never stale
// ai: (every write of an element lands there, as the run or its copy), which the decoder's output reads.
static inline int16_t **w4_views(const ldpc_t *c) { return (int16_t **)(c->w4 + (size_t)c->kb * 4 * c->z); }
static inline int16_t **w4_copies(const ldpc_t *c) { return w4_views(c) + c->slots_total; }
static inline int *w4_ncopies(const ldpc_t *c) { return (int *)(w4_copies(c) + c->lay_ptr[c->mb]); }

static int views_neon(ldpc_t *c) {
  const int z = c->z, ne = c->lay_ptr[c->mb];
  int16_t *Lp = c->wl + c->k, **vw = w4_views(c), **cp = w4_copies(c);
  int *ncp = w4_ncopies(c), *smax = malloc((size_t)c->kb * sizeof(int));   // ai: a column's largest shift
  if (!smax) return -1;
  for (int j = 0; j < c->kb; j++) smax[j] = -1;
  for (int e = 0; e < ne; e++) if (c->lay_shift[e] > smax[c->lay_col[e]]) smax[c->lay_col[e]] = c->lay_shift[e];
  for (int r = 0; r < c->mb; r++) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t **v = vw + p0 + 2 * r;
    int n = 0;
    for (int e = 0; e < nd; e++) {
      const int s = c->lay_shift[p0 + e], a = s & 31;
      v[e] = c->w4 + (size_t)c->lay_col[p0 + e] * 4 * z + 2 * z - s;
      if (a && smax[c->lay_col[p0 + e]] >= z - 31 + a) cp[p0 + n++] = v[e] + (s & ~31);
    }
    ncp[r] = n;
    v[nd] = r > 0 ? Lp + (r - 1) * z : c->wq + nd * c->zp;
    v[nd + 1] = Lp + r * z;
  }
  free(smax);
  return 0;
}

static void layer_direct(const ldpc_t *c, int r, int8_t *R, int ns, int16_t *const *view, int16_t *const *cp, int nc, int first) {
  const int z = c->z, nd = ns - 2;
  int16_t *Lp = c->wl + c->k;
  if (r == 0) { view[nd][0] = 8191; memcpy(view[nd] + 1, Lp + (c->mb - 1) * z, (size_t)(z - 1) * 2); }
#ifdef OB_NEON
  layer_kernel_neon(R, ns, c->zp, c->norm, view, nd, z, c->lay_shift + c->lay_ptr[r], cp, nc, first);
#else
  // ai: x86-64: the form ldpc_init chose (wide 1 AVX2, else SSE4.1)
#ifdef OB_LDPC_WIDE
  if (c->wide >= 1) layer_kernel_four_avx2(R, ns, c->zp, c->norm, view, nd, z, c->lay_shift + c->lay_ptr[r], cp, nc, first);
  else
#endif
    layer_kernel_four_sse41(R, ns, c->zp, c->norm, view, nd, z, c->lay_shift + c->lay_ptr[r], cp, nc, first);
#endif
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
// ai: a zero in front, as layer_gather lays it); one XOR a slot, only the signs counted.
// ai: parity_groups: g groups of 32 checks from i (4, or 1 for a row's last groups short of 128), the sign bits of the
// ai: XOR of every slot as a count a lane (USRA); the groups' XORs held in registers while each slot's pointer is read
// ai: once for all g.
static inline __attribute__((always_inline)) uint16x8_t parity_groups(int16_t *const *view, int nd, const int16_t *own, const int16_t *prev, int i, int row0, int16x8_t *carry, const int g) {
  int16x8_t x[16];
  for (int q = 0; q < g; q++) {
    const int16x8x4_t o = vld1q_s16_x4(own + i + 32 * q), p = vld1q_s16_x4(prev + i + 32 * q);
    if (!row0) {
      x[4 * q] = veorq_s16(o.val[0], p.val[0]); x[4 * q + 1] = veorq_s16(o.val[1], p.val[1]);
      x[4 * q + 2] = veorq_s16(o.val[2], p.val[2]); x[4 * q + 3] = veorq_s16(o.val[3], p.val[3]);
    } else {
      x[4 * q] = veorq_s16(o.val[0], vextq_s16(*carry, p.val[0], 7)); x[4 * q + 1] = veorq_s16(o.val[1], vextq_s16(p.val[0], p.val[1], 7));
      x[4 * q + 2] = veorq_s16(o.val[2], vextq_s16(p.val[1], p.val[2], 7)); x[4 * q + 3] = veorq_s16(o.val[3], vextq_s16(p.val[2], p.val[3], 7));
      *carry = p.val[3];
    }
  }
  for (int e = 0; e < nd; e++) {
    const int16_t *v = view[e] + i;
    for (int q = 0; q < g; q++) {
      const int16x8x4_t w = vld1q_s16_x4(v + 32 * q);
      x[4 * q] = veorq_s16(x[4 * q], w.val[0]); x[4 * q + 1] = veorq_s16(x[4 * q + 1], w.val[1]);
      x[4 * q + 2] = veorq_s16(x[4 * q + 2], w.val[2]); x[4 * q + 3] = veorq_s16(x[4 * q + 3], w.val[3]);
    }
  }
  uint16x8_t acc = vdupq_n_u16(0);
  for (int u = 0; u < 4 * g; u++) acc = vsraq_n_u16(acc, vreinterpretq_u16_s16(x[u]), 15);
  return acc;
}

// ai: count 0: 1 when every check holds, else 0 at the first layer holding a violated one (layers_clear);
// ai: count 1: how many of the m checks are violated (layers_unsat). rev: the layers from the last down. The answer is
// ai: the same in either order; the decoder tests them in the order its iteration updated them, so the layer longest
// ai: under the later layers' changes, the likeliest to hold a violated check, comes first and a failing test ends
// ai: sooner.
static int layers_parity_neon(const ldpc_t *c, int16_t *const *vw, int count, int rev) {
  const int z = c->z, mb = c->mb;
  const int16_t *Lp = c->wl + c->k;
  int bad = 0;
  for (int k = 0, r = rev ? mb - 1 : 0, dr = rev ? -1 : 1; k < mb; k++, r += dr) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t *const *view = vw + p0 + 2 * r;
    const int16_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    uint16x8_t acc = vdupq_n_u16(0);
    int16x8_t carry = vdupq_n_s16(0);
    int i = 0;
    for (; i + 128 <= z; i += 128) acc = vaddq_u16(acc, parity_groups(view, nd, own, prev, i, r == 0, &carry, 4));
    for (; i < z; i += 32) acc = vaddq_u16(acc, parity_groups(view, nd, own, prev, i, r == 0, &carry, 1));
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

// ai: The stall rule's first count (the checks violated after the first iteration) is wanted only by a block still
// ai: unsolved at stall_it, so the decode keeps the first iteration's hard decisions and counts them there. Kept as
// ai: sign bytes (a posterior's high byte): each data column twice over (2z, so the rotation by s is the run at z - s,
// ai: as w4's at 2z - s), then the staircase rows; the count is layers_parity_neon's on the bytes, the same number.
static void signs_keep(const ldpc_t *c, uint8_t *sv) {
  const int z = c->z;
  for (int j = 0; j < c->kb; j++, sv += 2 * z) {
    const int16_t *col = c->w4 + (size_t)j * 4 * z + 2 * z;
    for (int t = 0; t < z; t += 32) {
      const int16x8x4_t v = vld1q_s16_x4(col + t);
      const uint8x16x2_t b = { { vuzp2q_u8(vreinterpretq_u8_s16(v.val[0]), vreinterpretq_u8_s16(v.val[1])), vuzp2q_u8(vreinterpretq_u8_s16(v.val[2]), vreinterpretq_u8_s16(v.val[3])) } };
      vst1q_u8_x2(sv + t, b); vst1q_u8_x2(sv + z + t, b);
    }
  }
  const int16_t *Lp = c->wl + c->k;
  for (int t = 0; t < c->mb * z; t += 32, sv += 32) {
    const int16x8x4_t v = vld1q_s16_x4(Lp + t);
    const uint8x16x2_t b = { { vuzp2q_u8(vreinterpretq_u8_s16(v.val[0]), vreinterpretq_u8_s16(v.val[1])), vuzp2q_u8(vreinterpretq_u8_s16(v.val[2]), vreinterpretq_u8_s16(v.val[3])) } };
    vst1q_u8_x2(sv, b);
  }
}

static int signs_unsat(const ldpc_t *c, const uint8_t *sv) {
  const int z = c->z, mb = c->mb;
  const uint8_t *Lp = sv + (size_t)c->kb * 2 * z;
  int bad = 0;
  for (int r = 0; r < mb; r++) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    const uint8_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    uint8x16_t carry = vdupq_n_u8(0);
    for (int i = 0; i < z; i += 64) {
      uint8x16x4_t x = vld1q_u8_x4(own + i);
      const uint8x16x4_t p = vld1q_u8_x4(prev + i);
      if (r > 0) {
        x.val[0] = veorq_u8(x.val[0], p.val[0]); x.val[1] = veorq_u8(x.val[1], p.val[1]);
        x.val[2] = veorq_u8(x.val[2], p.val[2]); x.val[3] = veorq_u8(x.val[3], p.val[3]);
      } else {
        x.val[0] = veorq_u8(x.val[0], vextq_u8(carry, p.val[0], 15)); x.val[1] = veorq_u8(x.val[1], vextq_u8(p.val[0], p.val[1], 15));
        x.val[2] = veorq_u8(x.val[2], vextq_u8(p.val[1], p.val[2], 15)); x.val[3] = veorq_u8(x.val[3], vextq_u8(p.val[2], p.val[3], 15));
        carry = p.val[3];
      }
      for (int e = 0; e < nd; e++) {
        const uint8x16x4_t v = vld1q_u8_x4(sv + (size_t)c->lay_col[p0 + e] * 2 * z + z - c->lay_shift[p0 + e] + i);
        x.val[0] = veorq_u8(x.val[0], v.val[0]); x.val[1] = veorq_u8(x.val[1], v.val[1]);
        x.val[2] = veorq_u8(x.val[2], v.val[2]); x.val[3] = veorq_u8(x.val[3], v.val[3]);
      }
      const uint8x16_t s = vaddq_u8(vaddq_u8(vshrq_n_u8(x.val[0], 7), vshrq_n_u8(x.val[1], 7)), vaddq_u8(vshrq_n_u8(x.val[2], 7), vshrq_n_u8(x.val[3], 7)));
      bad += (int)vaddlvq_u8(s);
    }
  }
  return bad;
}
#elif defined(OB_X86)
// ai: x86-64: arm64's parity test, transposes and first-iteration signs above, ported; all exact (integers and sign
// ai: bits). The parity test in both forms (SSE4.1; AVX2 for ldpc_t wide 1), the rest once a decode, SSE4.1 alone.
// ai: parity_groups_sse41: g groups of 32 checks from i (2, or 1 for a row's last group short of 64: eight registers
// ai: of x86-64's sixteen, as arm64 holds 128 checks in sixteen of its 32), the row before moved up a lane for block
// ai: row 0 (PALIGNR, NEON's EXT), the sign bits of the XOR of every slot as a count a lane (PSRLW and PADDW, NEON's
// ai: USRA).
static inline __attribute__((always_inline)) __m128i parity_groups_sse41(int16_t *const *view, int nd, const int16_t *own, const int16_t *prev, int i, int row0, __m128i *carry, const int g) {
  __m128i x[8];
  for (int u = 0; u < 4 * g; u++) {
    const __m128i o = _mm_loadu_si128((const __m128i *)(own + i + 8 * u)), p = _mm_loadu_si128((const __m128i *)(prev + i + 8 * u));
    if (!row0) x[u] = _mm_xor_si128(o, p);
    else { x[u] = _mm_xor_si128(o, _mm_alignr_epi8(p, *carry, 14)); *carry = p; }
  }
  for (int e = 0; e < nd; e++) {
    const int16_t *v = view[e] + i;
    for (int u = 0; u < 4 * g; u++) x[u] = _mm_xor_si128(x[u], _mm_loadu_si128((const __m128i *)(v + 8 * u)));
  }
  __m128i acc = _mm_setzero_si128();
  for (int u = 0; u < 4 * g; u++) acc = _mm_add_epi16(acc, _mm_srli_epi16(x[u], 15));
  return acc;
}

// ai: the eight 16-bit counts of a vector summed
static inline int sum16_sse41(__m128i a) {
  a = _mm_add_epi16(a, _mm_srli_si128(a, 8)); a = _mm_add_epi16(a, _mm_srli_si128(a, 4)); a = _mm_add_epi16(a, _mm_srli_si128(a, 2));
  return _mm_extract_epi16(a, 0);
}

// ai: count and rev as layers_parity_neon's
static int layers_parity_sse41(const ldpc_t *c, int16_t *const *vw, int count, int rev) {
  const int z = c->z, mb = c->mb;
  const int16_t *Lp = c->wl + c->k;
  int bad = 0;
  for (int k = 0, r = rev ? mb - 1 : 0, dr = rev ? -1 : 1; k < mb; k++, r += dr) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t *const *view = vw + p0 + 2 * r;
    const int16_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    __m128i acc = _mm_setzero_si128(), carry = _mm_setzero_si128();
    int i = 0;
    for (; i + 64 <= z; i += 64) acc = _mm_add_epi16(acc, parity_groups_sse41(view, nd, own, prev, i, r == 0, &carry, 2));
    for (; i < z; i += 32) acc = _mm_add_epi16(acc, parity_groups_sse41(view, nd, own, prev, i, r == 0, &carry, 1));
    const int b = sum16_sse41(acc);
    if (!count && b) return 0;
    bad += b;
  }
  return count ? bad : 1;
}

#ifdef OB_LDPC_WIDE
// ai: The AVX2 form: arm64's 128 checks in eight registers; the row before moved up a lane across the halves by
// ai: VPERM2I128 and VPALIGNR.
OB_AVX2 __m256i parity_groups_avx2(int16_t *const *view, int nd, const int16_t *own, const int16_t *prev, int i, int row0, __m256i *carry, const int g) {
  __m256i x[8];
  for (int u = 0; u < 2 * g; u++) {
    const __m256i o = _mm256_loadu_si256((const __m256i *)(own + i + 16 * u)), p = _mm256_loadu_si256((const __m256i *)(prev + i + 16 * u));
    if (!row0) x[u] = _mm256_xor_si256(o, p);
    else { x[u] = _mm256_xor_si256(o, _mm256_alignr_epi8(p, _mm256_permute2x128_si256(*carry, p, 0x21), 14)); *carry = p; }
  }
  for (int e = 0; e < nd; e++) {
    const int16_t *v = view[e] + i;
    for (int u = 0; u < 2 * g; u++) x[u] = _mm256_xor_si256(x[u], _mm256_loadu_si256((const __m256i *)(v + 16 * u)));
  }
  __m256i acc = _mm256_setzero_si256();
  for (int u = 0; u < 2 * g; u++) acc = _mm256_add_epi16(acc, _mm256_srli_epi16(x[u], 15));
  return acc;
}

__attribute__((target("avx2"))) static int layers_parity_avx2(const ldpc_t *c, int16_t *const *vw, int count, int rev) {
  const int z = c->z, mb = c->mb;
  const int16_t *Lp = c->wl + c->k;
  int bad = 0;
  for (int k = 0, r = rev ? mb - 1 : 0, dr = rev ? -1 : 1; k < mb; k++, r += dr) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    int16_t *const *view = vw + p0 + 2 * r;
    const int16_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    __m256i acc = _mm256_setzero_si256(), carry = _mm256_setzero_si256();
    int i = 0;
    for (; i + 128 <= z; i += 128) acc = _mm256_add_epi16(acc, parity_groups_avx2(view, nd, own, prev, i, r == 0, &carry, 4));
    for (; i < z; i += 32) acc = _mm256_add_epi16(acc, parity_groups_avx2(view, nd, own, prev, i, r == 0, &carry, 1));
    const int b = sum16_sse41(_mm_add_epi16(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1)));
    if (!count && b) return 0;
    bad += b;
  }
  return count ? bad : 1;
}
#endif

// ai: 16 rows of 16 bytes transposed: four rounds of PUNPCKLBW and PUNPCKHBW on rows t and t + 8, each taking the byte
// ai: at (row r, column c) to row (2r + c / 8) mod 16, column (2c + r / 8) mod 16, so the four take it to (c, r)
static inline void transpose16(__m128i x[16]) {
  for (int k = 0; k < 4; k++) {
    __m128i y[16];
    for (int t = 0; t < 8; t++) { y[2 * t] = _mm_unpacklo_epi8(x[t], x[t + 8]); y[2 * t + 1] = _mm_unpackhi_epi8(x[t], x[t + 8]); }
    memcpy(x, y, sizeof y);
  }
}

// ai: parity_in, parity_out: as arm64's, the same blocks, reads and writes
static void parity_in(const int8_t *src, int16_t *dst, int z, int mb) {
  uint8_t tail[16 * 16 + 16] = { 0 };
  for (int i0 = 0; i0 < z; i0 += 16) {
    const uint8_t *s = (const uint8_t *)src + i0 * mb;
    if (mb < 16 && i0 + 16 == z) { memcpy(tail, s, (size_t)16 * mb); s = tail; }
    for (int r0 = 0; r0 < mb; r0 += 16) {
      const int c0 = r0 + 16 <= mb ? r0 : mb >= 16 ? mb - 16 : 0, nc = mb - c0 < 16 ? mb - c0 : 16;
      __m128i x[16];
      for (int t = 0; t < 16; t++) x[t] = _mm_loadu_si128((const __m128i *)(s + t * mb + c0));
      transpose16(x);
      for (int r = 0; r < nc; r++) {
        int16_t *d = dst + (c0 + r) * z + i0;
        _mm_storeu_si128((__m128i *)d, _mm_cvtepi8_epi16(x[r]));
        _mm_storeu_si128((__m128i *)(d + 8), _mm_cvtepi8_epi16(_mm_srli_si128(x[r], 8)));
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
      __m128i x[16];
      for (int r = 0; r < 16; r++) {
        if (r >= nc) { x[r] = _mm_setzero_si128(); continue; }
        const int16_t *q = src + (c0 + r) * z + i0;
        x[r] = _mm_packs_epi16(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)q), 15), _mm_srli_epi16(_mm_loadu_si128((const __m128i *)(q + 8)), 15));
      }
      transpose16(x);
      for (int t = 0; t < 16; t++) _mm_storeu_si128((__m128i *)(d + t * mb + c0), x[t]);
    }
    if (last) memcpy(dst + i0 * mb, tail, (size_t)16 * mb);
  }
}

// ai: signs_keep, signs_unsat: as arm64's, a byte a posterior; PACKSSWB's byte where NEON keeps the high one, the same
// ai: sign bit, the only one counted (PCMPGTB against zero, the -1s taken off a count a byte, summed by PSADBW)
static void signs_keep(const ldpc_t *c, uint8_t *sv) {
  const int z = c->z;
  for (int j = 0; j < c->kb; j++, sv += 2 * z) {
    const int16_t *col = c->w4 + (size_t)j * 4 * z + 2 * z;
    for (int t = 0; t < z; t += 32) {
      const q32_sse41 v = ld32_sse41(col + t);
      const __m128i a = _mm_packs_epi16(v.v[0], v.v[1]), b = _mm_packs_epi16(v.v[2], v.v[3]);
      _mm_storeu_si128((__m128i *)(sv + t), a); _mm_storeu_si128((__m128i *)(sv + t + 16), b);
      _mm_storeu_si128((__m128i *)(sv + z + t), a); _mm_storeu_si128((__m128i *)(sv + z + t + 16), b);
    }
  }
  const int16_t *Lp = c->wl + c->k;
  for (int t = 0; t < c->mb * z; t += 32, sv += 32) {
    const q32_sse41 v = ld32_sse41(Lp + t);
    _mm_storeu_si128((__m128i *)sv, _mm_packs_epi16(v.v[0], v.v[1])); _mm_storeu_si128((__m128i *)(sv + 16), _mm_packs_epi16(v.v[2], v.v[3]));
  }
}

static int signs_unsat(const ldpc_t *c, const uint8_t *sv) {
  const int z = c->z, mb = c->mb;
  const uint8_t *Lp = sv + (size_t)c->kb * 2 * z;
  const __m128i zero = _mm_setzero_si128();
  int bad = 0;
  for (int r = 0; r < mb; r++) {
    const int p0 = c->lay_ptr[r], nd = c->lay_ptr[r + 1] - p0;
    const uint8_t *own = Lp + r * z, *prev = Lp + (r > 0 ? r - 1 : mb - 1) * z;
    __m128i carry = zero;
    for (int i = 0; i < z; i += 64) {
      __m128i x[4];
      for (int u = 0; u < 4; u++) {
        const __m128i p = _mm_loadu_si128((const __m128i *)(prev + i + 16 * u));
        x[u] = _mm_loadu_si128((const __m128i *)(own + i + 16 * u));
        if (r > 0) x[u] = _mm_xor_si128(x[u], p);
        else { x[u] = _mm_xor_si128(x[u], _mm_alignr_epi8(p, carry, 15)); carry = p; }
      }
      for (int e = 0; e < nd; e++) {
        const uint8_t *v = sv + (size_t)c->lay_col[p0 + e] * 2 * z + z - c->lay_shift[p0 + e] + i;
        for (int u = 0; u < 4; u++) x[u] = _mm_xor_si128(x[u], _mm_loadu_si128((const __m128i *)(v + 16 * u)));
      }
      __m128i s = zero;
      for (int u = 0; u < 4; u++) s = _mm_sub_epi8(s, _mm_cmpgt_epi8(zero, x[u]));
      s = _mm_sad_epu8(s, zero);
      bad += _mm_cvtsi128_si32(s) + _mm_extract_epi16(s, 4);
    }
  }
  return bad;
}
#endif

// ai: the parity test on the four-times columns: arm64's, or x86-64's form for the kernel ldpc_init chose
#ifdef OB_NEON
#define layers_parity_w4 layers_parity_neon
#elif defined(OB_X86)
static int layers_parity_w4(const ldpc_t *c, int16_t *const *vw, int count, int rev) {
#ifdef OB_LDPC_WIDE
  if (c->wide >= 1) return layers_parity_avx2(c, vw, count, rev);
#endif
  return layers_parity_sse41(c, vw, count, rev);
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
#ifdef OB_LDPC_W4
  const int four = c->w4 && c->wide != -1;
  int16_t *const *vw = four ? w4_views(c) : 0, *const *cp = four ? w4_copies(c) : 0;
  const int *ncp = four ? w4_ncopies(c) : 0;
  uint8_t sv[LDPC_SIGNS];
  if (four) {
    // ai: each data column's z LLRs widened once and written into the middle two of its four copies, the only ones a
    // ai: run reads ([z, 3z): the outer two only take the stores either side of a run); wl's data part goes unused.
    for (int j = 0; j < c->kb; j++) {
      int16_t *d = c->w4 + (size_t)j * 4 * z + z;
      for (int t = 0; t < z; t += 16) {
#ifdef OB_NEON
        const int8x16_t b = vld1q_s8(llr + j * z + t);
        const int16x8x2_t v = { { vmovl_s8(vget_low_s8(b)), vmovl_high_s8(b) } };
        vst1q_s16_x2(d + t, v); vst1q_s16_x2(d + z + t, v);
#else
        const __m128i b = _mm_loadu_si128((const __m128i *)(llr + j * z + t)), lo = _mm_cvtepi8_epi16(b), hi = _mm_cvtepi8_epi16(_mm_srli_si128(b, 8));
        _mm_storeu_si128((__m128i *)(d + t), lo); _mm_storeu_si128((__m128i *)(d + t + 8), hi);
        _mm_storeu_si128((__m128i *)(d + z + t), lo); _mm_storeu_si128((__m128i *)(d + z + t + 8), hi);
#endif
      }
    }
    parity_in(llr + c->k, L + c->k, z, mb);
  } else
#endif
  {
    for (int v = 0; v < c->k; v++) L[v] = llr[v];
    for (int chk = 0; chk < c->m; chk++) L[c->k + (chk % mb) * z + chk / mb] = llr[c->k + chk];
  }
#ifdef OB_LDPC_W4
  if (!four)
#endif
  memset(c->wr, 0, (size_t)c->slots_total * zp);
  for (int it = 0; it < max_iter; it++) {
    for (int step = 0; step < mb; step++) {
      int r = (it & 1) ? mb - 1 - step : step, ns = c->lay_ptr[r + 1] - c->lay_ptr[r] + 2;
      int8_t *R = c->wr + (size_t)(c->lay_ptr[r] + 2 * r) * zp;
      if (r == 0) R[(ns - 2) * zp] = 0;
#ifdef OB_LDPC_W4
      if (four) { layer_direct(c, r, R, ns, vw + c->lay_ptr[r] + 2 * r, cp + c->lay_ptr[r], ncp[r], it == 0); continue; }
#endif
      layer_gather(c, r, Q);
      layer_kernel_for(c, Q, R, ns);
      layer_scatter(c, r, Q);
    }
#ifdef OB_LDPC_W4
    if (four) {
      // ai: The stop, as the scalar path's below: the checks tested after every iteration (the layers in the order the
      // ai: iteration updated them), a word whose every check holds stops there; at stall_it the violated checks are
      // ai: counted, and the block given up while still above stall_ratio of the count after the first iteration, which
      // ai: is taken then from that iteration's hard decisions, kept as they stood (signs_keep, signs_unsat).
      if (stall_it > 1 && it == stall_it - 1) {
        const int bad = layers_parity_w4(c, vw, 1, 0);
        if (!bad) { used = it + 1; break; }
        if ((float)bad > stall_ratio * (float)signs_unsat(c, sv)) { used = -3; break; }
      } else if (layers_parity_w4(c, vw, 0, it & 1)) { used = it + 1; break; }
      else if (it == 0 && stall_it > 1) signs_keep(c, sv);
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
#ifdef OB_LDPC_W4
  if (four) {
    // ai: the data from w4's third copy (past 2z, never stale), a sign bit to a byte
    for (int j = 0; j < c->kb; j++) {
      const int16_t *col = c->w4 + (size_t)j * 4 * z + 2 * z;
      for (int t = 0; t < z; t += 16) {
#ifdef OB_NEON
        const uint8x16x2_t b = vld2q_u8((const uint8_t *)(col + t));
        vst1q_u8(out + j * z + t, vshrq_n_u8(b.val[1], 7));
#else
        const __m128i lo = _mm_srli_epi16(_mm_loadu_si128((const __m128i *)(col + t)), 15), hi = _mm_srli_epi16(_mm_loadu_si128((const __m128i *)(col + t + 8)), 15);
        _mm_storeu_si128((__m128i *)(out + j * z + t), _mm_packs_epi16(lo, hi));
#endif
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
