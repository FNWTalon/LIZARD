#include "ldpc.h"
#include "ldpc_base.h"
#include <stdlib.h>
#include <string.h>

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
  return 0;
}

void ldpc_free(ldpc_t *c) { free(c->row_ptr); free(c->col_idx); free(c->lay_ptr); free(c->lay_col); free(c->lay_shift); free(c->wl); free(c->wq); free(c->wr); memset(c, 0, sizeof *c); }

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

int ldpc_decode(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter) { return ldpc_decode_stall(c, llr, out, max_iter, 0, 0); }

// stall_it > 0: a block that will never decode is let go early. Such a block makes no headway at all: its count of
// violated checks stays where the first iteration left it, while one that is merely slow has brought it down. So
// the count after iteration stall_it is held against the count after the first, and the block is given up (-3) if
// it is still above stall_ratio of it. The two counts replace those iterations' cleared test, so they cost nothing.
int ldpc_decode_stall(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter, int stall_it, float stall_ratio) {
  int z = c->z, zp = c->zp, mb = c->mb, used = -1, first = 0;
  int16_t *L = c->wl, *Q = c->wq;
  for (int v = 0; v < c->k; v++) L[v] = llr[v];
  for (int chk = 0; chk < c->m; chk++) L[c->k + (chk % mb) * z + chk / mb] = llr[c->k + chk];
  memset(c->wr, 0, (size_t)c->slots_total * zp);
  for (int it = 0; it < max_iter; it++) {
    for (int step = 0; step < mb; step++) {
      int r = (it & 1) ? mb - 1 - step : step, ns = c->lay_ptr[r + 1] - c->lay_ptr[r] + 2;
      int8_t *R = c->wr + (size_t)(c->lay_ptr[r] + 2 * r) * zp;
      layer_gather(c, r, Q);
      if (r == 0) R[(ns - 2) * zp] = 0;
      layer_kernel(Q, R, ns, zp, c->norm);
      layer_scatter(c, r, Q);
    }
    if (stall_it > 0 && (it == 0 || it == stall_it - 1)) {
      int bad = layers_unsat(c, Q);
      if (!bad) { used = it + 1; break; }
      if (it == 0) first = bad;
      else if ((float)bad > stall_ratio * (float)first) { used = -3; break; }
    } else if (layers_clear(c, Q)) { used = it + 1; break; }
  }
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
