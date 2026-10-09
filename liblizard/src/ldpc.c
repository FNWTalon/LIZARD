#include "ldpc.h"
#include "ldpc_base.h"
#include <stdlib.h>
#include <string.h>

// Per rate: parity block rows out of 48, and the data-column degree profile as
// (count of heavy columns, heavy degree); the remaining data columns have degree 3.
// Heavy/light mixes follow the published IRA profiles (a minority of high-degree columns
// carries convergence, degree 3 carries the rate), with the heavy degree capped at mb.
// norm is the min-sum scale in sixteenths. All of it is measured, test/ldpc_ablate.c: against
// all-degree-3 columns the heavy minority is worth 8% of the channel at rate 1/2 and 1 to 3%
// above 2/3, and low rates want a gentler normalization.
typedef struct { int mb, heavy, dh, norm; const char *name; double r; } profile_t;
static const profile_t PROFILE[LDPC_RATES] = {
  { 36, 10, 8, 14, "1/4", 0.25 },
  { 32, 10, 8, 14, "1/3", 1.0 / 3 },
  { 24, 10, 8, 13, "1/2", 0.5 },
  { 16, 8, 12, 13, "2/3", 2.0 / 3 },
  { 12, 8, 12, 13, "3/4", 0.75 },
  {  8, 8,  8, 13, "5/6", 5.0 / 6 },
  {  6, 6,  4, 13, "7/8", 0.875 },
};
const char *const ldpc_rate_name[LDPC_RATES] = { "1/4", "1/3", "1/2", "2/3", "3/4", "5/6", "7/8" };
double ldpc_rate_value(int rate) { return PROFILE[rate].r; }

static uint32_t rnd(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static int mod(int a, int z) { a %= z; return a < 0 ? a + z : a; }

// shift[r][j] is the circulant offset, -1 where the base graph has no edge: data bit t of
// block column j sits in position i = (t + shift) % z of block row r, which is check
// i * mb + r. Interleaving the block rows along the staircase is what gives the code a
// minimum distance: with checks numbered r * z + i instead, data bits t and t + 1 of a
// degree-3 column land on adjacent staircase checks in every block row, and flipping both
// plus the three parity bits between them is a weight-5 codeword.
static int count_cycles(int mb, int z, int shift[][LDPC_NB], int r, int j, int s, int *four) {
  int six = 0;
  *four = 0;
  for (int r2 = 0; r2 < mb; r2++) {
    if (r2 == r || shift[r2][j] < 0) continue;
    int d = s - shift[r2][j];
    // The staircase joins check c to c + 1: the same position in the next block row, or the
    // next position of row 0 after the last row. A data bit in both closes a 4-cycle.
    if ((r2 == r + 1 || r2 == r - 1) && shift[r2][j] == s) (*four)++;
    if (r == mb - 1 && r2 == 0 && mod(shift[r2][j] - s, z) == 1) (*four)++;
    if (r == 0 && r2 == mb - 1 && mod(s - shift[r2][j], z) == 1) (*four)++;
    for (int j2 = 0; j2 < LDPC_NB; j2++) {
      if (j2 == j || shift[r2][j2] < 0) continue;
      if (shift[r][j2] >= 0 && mod(d - (shift[r][j2] - shift[r2][j2]), z) == 0) (*four)++;
      for (int r3 = 0; r3 < mb; r3++) {
        if (r3 == r || r3 == r2 || shift[r3][j2] < 0) continue;
        int d2 = d + shift[r2][j2] - shift[r3][j2];
        for (int j3 = 0; j3 < LDPC_NB; j3++) {
          if (j3 == j || j3 == j2 || shift[r3][j3] < 0 || shift[r][j3] < 0) continue;
          if (mod(d2 + shift[r3][j3] - shift[r][j3], z) == 0) six++;
        }
      }
    }
  }
  return six;
}

// Test hook: a degree profile to use instead of the table's, for the ablations in test/.
int ldpc_override_heavy = -1, ldpc_override_dh = 0, ldpc_norm = 0;
int ldpc_codes_set = -1;
static int codes_set(void) {
  if (ldpc_codes_set < 0) { const char *e = getenv("LIZ_TABLES"); ldpc_codes_set = e && *e ? atoi(e) : 0; }
  return ldpc_codes_set;
}

static profile_t profile(int rate) {
  profile_t prof = PROFILE[rate];
  if (ldpc_override_heavy >= 0) { prof.heavy = ldpc_override_heavy; prof.dh = ldpc_override_dh; }
  return prof;
}

int ldpc_generate(int z, int rate, uint32_t seed, int shift[][LDPC_NB]) {
  if (rate < 0 || rate >= LDPC_RATES || z < 8) return -1;
  const profile_t prof = profile(rate), *p = &prof;
  int mb = p->mb, kb = LDPC_NB - mb;
  int rowdeg[LDPC_NB] = { 0 };
  for (int r = 0; r < mb; r++) for (int j = 0; j < LDPC_NB; j++) shift[r][j] = -1;
  uint32_t s = seed * 2654435761u + (uint32_t)(rate * 97 + 1);
  if (!s) s = 1;
  for (int j = 0; j < kb; j++) {
    int deg = j < p->heavy ? p->dh : 3;
    if (deg > mb) deg = mb;
    for (int e = 0; e < deg; e++) {
      // Lightest row first keeps check degrees concentrated; ties go to the PRNG.
      int best = -1, bd = 1 << 30;
      uint32_t tie = 0;
      for (int r = 0; r < mb; r++) {
        if (shift[r][j] >= 0) continue;
        uint32_t t = rnd(&s);
        if (rowdeg[r] < bd || (rowdeg[r] == bd && t > tie)) { best = r; bd = rowdeg[r]; tie = t; }
      }
      int pick = 0, pick6 = 1 << 30, pick4 = 1 << 30;
      for (int t = 0; t < 12; t++) {
        int cand = (int)(rnd(&s) % (uint32_t)z), four, six = count_cycles(mb, z, shift, best, j, cand, &four);
        if (four < pick4 || (four == pick4 && six < pick6)) { pick = cand; pick4 = four; pick6 = six; }
        if (!four && !six) break;
      }
      shift[best][j] = pick;
      rowdeg[best]++;
    }
  }
  return mb;
}

int ldpc_init(ldpc_t *c, int n_max, int rate, uint32_t seed) {
  memset(c, 0, sizeof *c);
  if (rate < 0 || rate >= LDPC_RATES) return -1;
  const profile_t prof = profile(rate), *p = &prof;
  int z = n_max / LDPC_NB;
  if (z < 8) return -1;
  // The format's codes are the tables (ldpc_base.h, one a rate of the rate profile, each with its own block rows, data
  // columns and columns never sent); every other code, and the format's own under a test's profile override, is
  // generated over 48 columns.
  static _Thread_local int shift[LDPC_NB][LDPC_NB];   // ai: scratch, a thread's own
  const ldpc_base_t *base = 0, *set = codes_set() == 1 ? LDPC_BASES_V1 : codes_set() == 2 ? LDPC_BASES_V2 : LDPC_BASES;
  for (int t = 0; t < LDPC_BASES_N; t++)
    if (z == set[t].z && rate == set[t].rate && seed == LDPC_BASE_SEED && ldpc_override_heavy < 0) base = &set[t];
  // ai: the lab set 2's tables carry their own z (128): matched by rate where the codeword fits the block's n_max
  if (!base && codes_set() == 2)
    for (int t = 0; t < LDPC_BASES_N; t++)
      if (rate == set[t].rate && seed == LDPC_BASE_SEED && ldpc_override_heavy < 0 && (set[t].kb + set[t].mb) * set[t].z <= n_max) { base = &set[t]; z = set[t].z; }
  int mb = base ? base->mb : p->mb, kb = base ? base->kb : LDPC_NB - mb, np = base ? base->np * z : 0;
  if (mb > LDPC_NB || kb > LDPC_NB) return -1;
  c->z = z; c->mb = mb; c->kb = kb; c->rate = rate; c->norm = ldpc_norm ? ldpc_norm : p->norm;
  c->n = (kb + mb) * z; c->k = kb * z; c->m = mb * z; c->np = np; c->nt = c->n - np;
  if (base) {
    for (int r = 0; r < mb; r++) for (int j = 0; j < LDPC_NB; j++) shift[r][j] = j < base->kb ? base->shift[r * base->kb + j] : -1;
  } else if (ldpc_generate(z, rate, seed, shift) < 0) return -1;
  int rowdeg[LDPC_NB] = { 0 };
  for (int r = 0; r < mb; r++) for (int j = 0; j < kb; j++) rowdeg[r] += shift[r][j] >= 0;

  int edges = 0;
  for (int r = 0; r < mb; r++) edges += (rowdeg[r] + 2) * z;
  c->row_ptr = malloc((size_t)(c->m + 1) * sizeof(int));
  c->col_idx = malloc((size_t)edges * sizeof(int));
  if (!c->row_ptr || !c->col_idx) { ldpc_free(c); return -1; }
  int e = 0;
  for (int chk = 0; chk < c->m; chk++) {
    int i = chk / mb, r = chk % mb;
    c->row_ptr[chk] = e;
    for (int j = 0; j < kb; j++) if (shift[r][j] >= 0) c->col_idx[e++] = j * z + mod(i - shift[r][j], z);
    if (chk > 0) c->col_idx[e++] = c->k + chk - 1;
    c->col_idx[e++] = c->k + chk;
  }
  c->row_ptr[c->m] = e;
  c->edges = e;

  c->zp = (z + 7) & ~7;
  c->lay_ptr = malloc((size_t)(mb + 1) * sizeof(int));
  c->lay_col = malloc((size_t)mb * LDPC_NB * sizeof(int));
  c->lay_shift = malloc((size_t)mb * LDPC_NB * sizeof(int));
  if (!c->lay_ptr || !c->lay_col || !c->lay_shift) { ldpc_free(c); return -1; }
  int s2 = 0;
  for (int r = 0; r < mb; r++) {
    c->lay_ptr[r] = s2;
    for (int j = 0; j < kb; j++) if (shift[r][j] >= 0) { c->lay_col[s2] = j; c->lay_shift[s2++] = shift[r][j]; }
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

static int syndrome_clear(const ldpc_t *c, const int16_t *L) {
  for (int chk = 0; chk < c->m; chk++) {
    int par = 0;
    for (int e = c->row_ptr[chk]; e < c->row_ptr[chk + 1]; e++) par ^= L[c->col_idx[e]] < 0;
    if (par) return 0;
  }
  return 1;
}

// Layered normalized min-sum, one check per layer. The sweep direction alternates because the
// staircase only carries parity information one step against the sweep per pass.
int ldpc_decode_serial(const ldpc_t *c, const int8_t *llr, uint8_t *out, int max_iter) {
  int16_t *L = malloc((size_t)c->n * sizeof(int16_t));
  int8_t *R = calloc((size_t)c->edges, 1);
  int used = -1;
  if (!L || !R) { free(L); free(R); return -1; }
  for (int i = 0; i < c->n; i++) L[i] = llr[i];
  for (int it = 0; it < max_iter; it++) {
    for (int step = 0; step < c->m; step++) {
      int chk = (it & 1) ? c->m - 1 - step : step;
      int e0 = c->row_ptr[chk], e1 = c->row_ptr[chk + 1];
      int min1 = 32767, min2 = 32767, arg = -1, sign = 0;
      for (int e = e0; e < e1; e++) {
        int q = L[c->col_idx[e]] - R[e];
        L[c->col_idx[e]] = (int16_t)q;
        int a = q < 0 ? -q : q;
        sign ^= q < 0;
        if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) min2 = a;
      }
      min1 = (min1 * c->norm) >> 4; min2 = (min2 * c->norm) >> 4;
      if (min1 > 127) min1 = 127;
      if (min2 > 127) min2 = 127;
      for (int e = e0; e < e1; e++) {
        int v = c->col_idx[e], q = L[v];
        int mag = e == arg ? min2 : min1;
        int r = (sign ^ (q < 0)) ? -mag : mag;
        R[e] = (int8_t)r;
        q += r;
        L[v] = (int16_t)(q > 8191 ? 8191 : q < -8191 ? -8191 : q);
      }
    }
    if (syndrome_clear(c, L)) { used = it + 1; break; }
  }
  for (int i = 0; i < c->n; i++) out[i] = L[i] < 0;
  free(L); free(R);
  return used;
}

// ---------------------------------------------------------------- block-row schedule
//
// The z checks of one block row share no bit: a circulant puts each data bit in one of them, and
// check i * mb + r owns staircase bits i * mb + r - 1 and i * mb + r, which no other check of
// row r touches. So a layer is a whole block row, and its z checks run side by side, eight to a
// vector. Each circulant's bits are first rotated into check order (two memcpys), which makes
// every slot a plain array indexed by check position.

#include "simd.h"

// Q: ns rows of zp posteriors in check order. R: the matching messages. Same arithmetic as the
// serial decoder, check by check; the scalar loop below is the reference for the vector one.
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
