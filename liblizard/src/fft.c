#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include "simd.h"

// W[j] = exp(-2 pi i j / rows), j < rows / 2. One table serves every stage: a stage of length len steps through it
// by rows / len. Computed in double, which the old recurrence (one multiply per butterfly) was not.
// st: the radix of each stage, the SHORTEST len first: 2 throughout, and a size with a factor of 3 (384, 768, 1536)
// takes one radix-3 stage, the shortest, ahead of them.
// ai: A size 2^a 3^b 5^c takes its b radix-3 stages, then its c radix-5 stages (rows5), ahead of the 2s: 1152 two
// ai: radix-3 stages, 1280 one radix-5 stage; 3 x 2^k keeps its one radix-3 stage first, so its bytes do not move.
// swp: the input permutation as a list of row swaps. The digit reversal stops being its own inverse as soon as the
// radices differ, so it cannot be applied by walking i < rev[i] the way bit reversal can; the plan works the swaps
// out once, by putting the right row in each place in turn, and the transform just performs them.
// ai: half: the twiddles the table holds, rows / 2 for an even size (past it W^j is -W^(j - rows / 2), tw_of), the whole
// ai: circle for an odd one (3^b 5^c, which only fft_test asks for)
// ai: place (kind 1 only): where the decoder's transforms keep each index, the digit reversal's inverse: fft_dec_dif
// ai: leaves frequency k in row place[k], fft_dec_dit takes sample t from row place[t] (fft.h)
typedef struct { int rows, nst, nswp, half, kind; float *wr, *wi; int *swp, *place; int st[24], len[24]; } plan_t;
// ai: each thread's own (a native receiver transforms on several): a plan is built on first use and a ninth size
// ai: replaces the eighth, neither of which two threads could share
static _Thread_local plan_t plans[8];

// ai: 2^a 3^b 5^c: the sizes a stage list can be made of (at most 24 stages: rows below 2^24).
static int size_ok(int rows) {
  if (rows < 1 || rows >= 1 << 24) return 0;
  while (rows % 3 == 0) rows /= 3;
  while (rows % 5 == 0) rows /= 5;
  return !(rows & (rows - 1));
}

// ai: kind 1, the decoder's transforms on arm64 (fft_dec_dit, and fft_dec_dif the same stages the other way round): a
// ai: first stage without twiddles of radix 12, 6, 5, 4, 3 or 2, then a 6 where a 3 and an odd 2 are left, the 3s, the
// ai: 5s, the one 2, the 4s (the last a 4 wherever the size has two 2s past the first stage). The first stage is the one
// ai: whose plan costs least by each stage's cycles a point on an S26 prime core (the stage benchmark, 1152 rows by 32
// ai: and 64 columns: a 12 without twiddles 3.0, a 6 with 2.9, a 4 2.05; per stage, a pass over every row): 768 and
// ai: 1152 lead with a 12, 384 and 1536 a 6, 1280 a 5, 512 a 2, 1024 a 4.
static int stage_list_dec(int rows, int *st) {
  int bits = 0, t3 = 0, t5 = 0;
  for (int t = rows; t % 3 == 0; t /= 3) t3++;
  for (int t = rows; t % 5 == 0; t /= 5) t5++;
  for (int t = rows; t > 1 && !(t & 1); t >>= 1) bits++;
  static const int R[6] = { 12, 6, 5, 4, 3, 2 }, B[6] = { 2, 1, 0, 2, 0, 1 }, T3[6] = { 1, 1, 0, 0, 1, 0 }, T5[6] = { 0, 0, 1, 0, 0, 0 };
  static const float C0[6] = { 3.0f, 2.6f, 2.7f, 2.4f, 2.2f, 1.6f };   // ai: the first stage's, no twiddles
  const float c6 = 2.9f, c3 = 2.4f, c5 = 3.2f, c4 = 2.05f, c2 = 1.8f;
  int best = -1;
  float bc = 0;
  for (int f = 0; f < 6; f++) {
    if (bits < B[f] || t3 < T3[f] || t5 < T5[f]) continue;
    const int b = bits - B[f], u3 = t3 - T3[f], u5 = t5 - T5[f], six = (b & 1) && u3;
    const float c = C0[f] + six * c6 + (u3 - six) * c3 + u5 * c5 + ((b - six) & 1) * c2 + ((b - six) / 2) * c4;
    if (best < 0 || c < bc) { best = f; bc = c; }
  }
  if (best < 0) return 0;
  int k = 0, b = bits - B[best], u3 = t3 - T3[best], u5 = t5 - T5[best];
  st[k++] = R[best];
  const int six = (b & 1) && u3;
  if (six) { st[k++] = 6; b--; u3--; }
  while (u3--) st[k++] = 3;
  while (u5--) st[k++] = 5;
  if (b & 1) st[k++] = 2;
  for (int i = 0; i < b / 2; i++) st[k++] = 4;
  return k;
}
// ai: kind 0, the format's schedule: the 3s, the 5s, then 2s
static int stage_list(int rows, int *st, int kind) {
  if (kind == 1) return stage_list_dec(rows, st);
  int k = 0;
  while (rows % 3 == 0) { st[k++] = 3; rows /= 3; }
  while (rows % 5 == 0) { st[k++] = 5; rows /= 5; }
  for (int t = rows; t > 1; t >>= 1) st[k++] = 2;
  return k;
}

static const plan_t *plan_kind(int rows, int kind) {
  if (!size_ok(rows)) return NULL;
  int slot = 0;
  for (; slot < 8 && plans[slot].rows; slot++) if (plans[slot].rows == rows && plans[slot].kind == kind) return &plans[slot];
  if (slot == 8) slot = 7;   // more sizes than slots: the last one is rebuilt. LIZARD uses one size.
  plan_t *p = &plans[slot];
  free(p->wr); free(p->wi); free(p->swp); free(p->place); p->place = NULL;
  p->rows = rows; p->kind = kind; p->half = rows & 1 ? rows : rows / 2;
  p->wr = malloc((size_t)(p->half + 1) * sizeof(float)); p->wi = malloc((size_t)(p->half + 1) * sizeof(float));
  p->swp = malloc((size_t)2 * rows * sizeof(int));
  int *rev = malloc((size_t)rows * sizeof(int)), *at = malloc((size_t)rows * sizeof(int)), *cur = malloc((size_t)rows * sizeof(int));
  if (!p->wr || !p->wi || !p->swp || !rev || !at || !cur) { free(rev); free(at); free(cur); p->rows = 0; return NULL; }
  for (int j = 0; j < p->half; j++) { double a = -2 * 3.14159265358979323846 * j / rows; p->wr[j] = (float)cos(a); p->wi[j] = (float)sin(a); }
  p->nst = stage_list(rows, p->st, kind);
  for (int s = 0, L = 1; s < p->nst; s++) { L *= p->st[s]; p->len[s] = L; }
  // The input goes to its digit-reversed place: the digits of i in the stages' own radices, read back to front.
  for (int i = 0; i < rows; i++) {
    int j = 0, x = i;
    for (int s = 0; s < p->nst; s++) { j = j * p->st[s] + x % p->st[s]; x /= p->st[s]; }
    rev[i] = j;
  }
  p->nswp = 0;
  if (kind == 1) {
    // ai: no swaps: row i of the decoder's transforms holds index rev[i] (place its inverse), which unpack_turn reads
    // ai: and writes through (lizard_phy.c)
    p->place = rev;
    for (int i = 0; i < rows; i++) at[rev[i]] = i;
    for (int i = 0; i < rows; i++) rev[i] = at[i];
    free(at); free(cur);
    return p;
  }
  for (int i = 0; i < rows; i++) { at[i] = i; cur[i] = i; }
  for (int i = 0; i < rows; i++) {
    const int from = at[rev[i]];
    if (from == i) continue;
    p->swp[2 * p->nswp] = i; p->swp[2 * p->nswp + 1] = from; p->nswp++;
    const int a = cur[i], b = cur[from];
    cur[i] = b; cur[from] = a; at[b] = i; at[a] = from;
  }
  free(rev); free(at); free(cur);
  return p;
}
static const plan_t *plan_for(int rows) { return plan_kind(rows, 0); }

// W_rows^j for any j < rows, from a table that holds only the first half circle: past it the sign flips.
// ai: (an odd size's table holds the whole circle, half = rows, so j never passes it)
static inline void tw_of(const plan_t *p, int rows, int j, int inverse, float *tr, float *ti) {
  float s = 1.0f;
  (void)rows;
  if (j >= p->half) { j -= p->half; s = -1.0f; }
  *tr = s * p->wr[j];
  *ti = inverse ? -s * p->wi[j] : s * p->wi[j];
}

// One butterfly between two rows, every column: (a, b) -> (a + w b, a - w b). unit: w is 1, a sum and a difference.
static inline void rows2(float *ar, float *ai, float *br, float *bi, int cols, float wr, float wi, int unit) {
  int x = 0;
  if (unit) {
#ifdef OB_SIMD
    for (; x + 4 <= cols; x += 4) {
      v128_t pr = wasm_v128_load(ar + x), pi = wasm_v128_load(ai + x), qr = wasm_v128_load(br + x), qi = wasm_v128_load(bi + x);
      wasm_v128_store(br + x, wasm_f32x4_sub(pr, qr)); wasm_v128_store(bi + x, wasm_f32x4_sub(pi, qi));
      wasm_v128_store(ar + x, wasm_f32x4_add(pr, qr)); wasm_v128_store(ai + x, wasm_f32x4_add(pi, qi));
    }
#endif
    for (; x < cols; x++) { float qr = br[x], qi = bi[x]; br[x] = ar[x] - qr; bi[x] = ai[x] - qi; ar[x] = ar[x] + qr; ai[x] = ai[x] + qi; }
    return;
  }
#ifdef OB_SIMD
  const v128_t vwr = wasm_f32x4_splat(wr), vwi = wasm_f32x4_splat(wi);
  for (; x + 4 <= cols; x += 4) {
    v128_t pr = wasm_v128_load(ar + x), pi = wasm_v128_load(ai + x), qr = wasm_v128_load(br + x), qi = wasm_v128_load(bi + x);
    v128_t xr = wasm_f32x4_sub(wasm_f32x4_mul(qr, vwr), wasm_f32x4_mul(qi, vwi)), xi = wasm_f32x4_add(wasm_f32x4_mul(qr, vwi), wasm_f32x4_mul(qi, vwr));
    wasm_v128_store(br + x, wasm_f32x4_sub(pr, xr)); wasm_v128_store(bi + x, wasm_f32x4_sub(pi, xi));
    wasm_v128_store(ar + x, wasm_f32x4_add(pr, xr)); wasm_v128_store(ai + x, wasm_f32x4_add(pi, xi));
  }
#endif
  for (; x < cols; x++) {
    float xr = br[x] * wr - bi[x] * wi, xi = br[x] * wi + bi[x] * wr;
    br[x] = ar[x] - xr; bi[x] = ai[x] - xi; ar[x] = ar[x] + xr; ai[x] = ai[x] + xi;
  }
}

// Two stages in one pass over four rows: (r0, r1) and (r2, r3) with twiddle w, then (r0, r2) with u and (r1, r3)
// with v. The same operations on every element as two calls of rows2 a stage apart, so the same floats; what
// changes is that the rows are read and written once, not twice.
static inline void rows4(float *r[4], float *m[4], int cols, float wr, float wi, float ur, float ui, float vr, float vi) {
  int x = 0;
#ifdef OB_SIMD
  const v128_t vwr = wasm_f32x4_splat(wr), vwi = wasm_f32x4_splat(wi), vur = wasm_f32x4_splat(ur), vui = wasm_f32x4_splat(ui), vvr = wasm_f32x4_splat(vr), vvi = wasm_f32x4_splat(vi);
  for (; x + 4 <= cols; x += 4) {
    v128_t ar = wasm_v128_load(r[0] + x), ai = wasm_v128_load(m[0] + x), br = wasm_v128_load(r[1] + x), bi = wasm_v128_load(m[1] + x);
    v128_t cr = wasm_v128_load(r[2] + x), ci = wasm_v128_load(m[2] + x), dr = wasm_v128_load(r[3] + x), di = wasm_v128_load(m[3] + x);
    v128_t xr = wasm_f32x4_sub(wasm_f32x4_mul(br, vwr), wasm_f32x4_mul(bi, vwi)), xi = wasm_f32x4_add(wasm_f32x4_mul(br, vwi), wasm_f32x4_mul(bi, vwr));
    br = wasm_f32x4_sub(ar, xr); bi = wasm_f32x4_sub(ai, xi); ar = wasm_f32x4_add(ar, xr); ai = wasm_f32x4_add(ai, xi);
    xr = wasm_f32x4_sub(wasm_f32x4_mul(dr, vwr), wasm_f32x4_mul(di, vwi)); xi = wasm_f32x4_add(wasm_f32x4_mul(dr, vwi), wasm_f32x4_mul(di, vwr));
    dr = wasm_f32x4_sub(cr, xr); di = wasm_f32x4_sub(ci, xi); cr = wasm_f32x4_add(cr, xr); ci = wasm_f32x4_add(ci, xi);
    xr = wasm_f32x4_sub(wasm_f32x4_mul(cr, vur), wasm_f32x4_mul(ci, vui)); xi = wasm_f32x4_add(wasm_f32x4_mul(cr, vui), wasm_f32x4_mul(ci, vur));
    wasm_v128_store(r[2] + x, wasm_f32x4_sub(ar, xr)); wasm_v128_store(m[2] + x, wasm_f32x4_sub(ai, xi)); wasm_v128_store(r[0] + x, wasm_f32x4_add(ar, xr)); wasm_v128_store(m[0] + x, wasm_f32x4_add(ai, xi));
    xr = wasm_f32x4_sub(wasm_f32x4_mul(dr, vvr), wasm_f32x4_mul(di, vvi)); xi = wasm_f32x4_add(wasm_f32x4_mul(dr, vvi), wasm_f32x4_mul(di, vvr));
    wasm_v128_store(r[3] + x, wasm_f32x4_sub(br, xr)); wasm_v128_store(m[3] + x, wasm_f32x4_sub(bi, xi)); wasm_v128_store(r[1] + x, wasm_f32x4_add(br, xr)); wasm_v128_store(m[1] + x, wasm_f32x4_add(bi, xi));
  }
#endif
  for (; x < cols; x++) {
    float ar = r[0][x], ai = m[0][x], br = r[1][x], bi = m[1][x], cr = r[2][x], ci = m[2][x], dr = r[3][x], di = m[3][x];
    float xr = br * wr - bi * wi, xi = br * wi + bi * wr;
    br = ar - xr; bi = ai - xi; ar = ar + xr; ai = ai + xi;
    xr = dr * wr - di * wi; xi = dr * wi + di * wr;
    dr = cr - xr; di = ci - xi; cr = cr + xr; ci = ci + xi;
    xr = cr * ur - ci * ui; xi = cr * ui + ci * ur;
    r[2][x] = ar - xr; m[2][x] = ai - xi; r[0][x] = ar + xr; m[0][x] = ai + xi;
    xr = dr * vr - di * vi; xi = dr * vi + di * vr;
    r[3][x] = br - xr; m[3][x] = bi - xi; r[1][x] = br + xr; m[1][x] = bi + xi;
  }
}

// ai: One radix-3 butterfly across three rows, every column, for the sizes with a factor of 3. With b and c twiddled,
// ai: t1 = b + c and t2 = b - c: out0 = a + t1, out1 = a - t1 / 2 - s i (sqrt 3 / 2) t2, out2 the same with + s i.
// ai: s is the sign of the rotation, +1 forward and -1 inverse, so no branch reaches the loop. The vector and scalar
// ai: loops do the same float ops in the same order.
static void rows3(float *const r[3], float *const m[3], int cols, const float *w, float s, int unit) {
  const float w1r = w[0], w1i = w[1], w2r = w[2], w2i = w[3], k = s * 0.86602540378443864676f, nk = -k;
  int x = 0;
#ifdef OB_SIMD
  const v128_t v1r = wasm_f32x4_splat(w1r), v1i = wasm_f32x4_splat(w1i), v2r = wasm_f32x4_splat(w2r), v2i = wasm_f32x4_splat(w2i);
  const v128_t vk = wasm_f32x4_splat(k), vnk = wasm_f32x4_splat(nk), vh = wasm_f32x4_splat(0.5f);
  for (; x + 4 <= cols; x += 4) {
    v128_t br = wasm_v128_load(r[1] + x), bi = wasm_v128_load(m[1] + x), cr = wasm_v128_load(r[2] + x), ci = wasm_v128_load(m[2] + x);
    if (!unit) {
      const v128_t b2r = wasm_f32x4_sub(wasm_f32x4_mul(br, v1r), wasm_f32x4_mul(bi, v1i)), b2i = wasm_f32x4_add(wasm_f32x4_mul(br, v1i), wasm_f32x4_mul(bi, v1r));
      const v128_t c2r = wasm_f32x4_sub(wasm_f32x4_mul(cr, v2r), wasm_f32x4_mul(ci, v2i)), c2i = wasm_f32x4_add(wasm_f32x4_mul(cr, v2i), wasm_f32x4_mul(ci, v2r));
      br = b2r; bi = b2i; cr = c2r; ci = c2i;
    }
    const v128_t ar = wasm_v128_load(r[0] + x), ai = wasm_v128_load(m[0] + x);
    const v128_t t1r = wasm_f32x4_add(br, cr), t1i = wasm_f32x4_add(bi, ci), t2r = wasm_f32x4_sub(br, cr), t2i = wasm_f32x4_sub(bi, ci);
    const v128_t hr = wasm_f32x4_sub(ar, wasm_f32x4_mul(vh, t1r)), hi = wasm_f32x4_sub(ai, wasm_f32x4_mul(vh, t1i));
    const v128_t rr = wasm_f32x4_mul(vk, t2i), ri = wasm_f32x4_mul(vnk, t2r);
    wasm_v128_store(r[0] + x, wasm_f32x4_add(ar, t1r)); wasm_v128_store(m[0] + x, wasm_f32x4_add(ai, t1i));
    wasm_v128_store(r[1] + x, wasm_f32x4_add(hr, rr));  wasm_v128_store(m[1] + x, wasm_f32x4_add(hi, ri));
    wasm_v128_store(r[2] + x, wasm_f32x4_sub(hr, rr));  wasm_v128_store(m[2] + x, wasm_f32x4_sub(hi, ri));
  }
#endif
  for (; x < cols; x++) {
    const float ar = r[0][x], ai = m[0][x], br = r[1][x], bi = m[1][x], cr = r[2][x], ci = m[2][x];
    const float b2r = unit ? br : br * w1r - bi * w1i, b2i = unit ? bi : br * w1i + bi * w1r;
    const float c2r = unit ? cr : cr * w2r - ci * w2i, c2i = unit ? ci : cr * w2i + ci * w2r;
    const float t1r = b2r + c2r, t1i = b2i + c2i, t2r = b2r - c2r, t2i = b2i - c2i;
    const float hr = ar - 0.5f * t1r, hi = ai - 0.5f * t1i, rr = k * t2i, ri = nk * t2r;
    r[0][x] = ar + t1r; m[0][x] = ai + t1i;
    r[1][x] = hr + rr;  m[1][x] = hi + ri;
    r[2][x] = hr - rr;  m[2][x] = hi - ri;
  }
}

// ai: One radix-5 butterfly across five rows, every column, for the sizes with a factor of 5 (1280).
// ai: With b to e twiddled: t1 = b + e, t2 = c + d, t3 = b - e, t4 = c - d; out0 = a + t1 + t2; m1 = a + c1 t1 + c2 t2
// ai: and m2 = a + c2 t1 + c1 t2 (c1 = cos 2 pi / 5, c2 = cos 4 pi / 5); n1 = s1 t3 + s2 t4 and n2 = s2 t3 - s1 t4
// ai: (s1 = s sin 2 pi / 5, s2 = s sin 4 pi / 5); out1 = m1 - i n1, out4 = m1 + i n1, out2 = m2 - i n2, out3 = m2 + i n2.
// ai: s is the sign of the rotation, +1 forward and -1 inverse, as rows3's. The vector and scalar loops do the same
// ai: float ops in the same order.
static void rows5(float *const r[5], float *const m[5], int cols, const float *w, float s, int unit) {
  const float c1 = 0.30901699437494742410f, c2 = -0.80901699437494742410f, s1 = s * 0.95105651629515357212f, s2 = s * 0.58778525229247312917f;
  int x = 0;
#ifdef OB_SIMD
  const v128_t vc1 = wasm_f32x4_splat(c1), vc2 = wasm_f32x4_splat(c2), vs1 = wasm_f32x4_splat(s1), vs2 = wasm_f32x4_splat(s2);
  v128_t vwr[4], vwi[4];
  for (int j = 0; j < 4; j++) { vwr[j] = wasm_f32x4_splat(w[2 * j]); vwi[j] = wasm_f32x4_splat(w[2 * j + 1]); }
  for (; x + 4 <= cols; x += 4) {
    v128_t qr[4], qi[4];
    for (int j = 0; j < 4; j++) {
      const v128_t pr = wasm_v128_load(r[j + 1] + x), pi = wasm_v128_load(m[j + 1] + x);
      if (unit) { qr[j] = pr; qi[j] = pi; continue; }
      qr[j] = wasm_f32x4_sub(wasm_f32x4_mul(pr, vwr[j]), wasm_f32x4_mul(pi, vwi[j]));
      qi[j] = wasm_f32x4_add(wasm_f32x4_mul(pr, vwi[j]), wasm_f32x4_mul(pi, vwr[j]));
    }
    const v128_t ar = wasm_v128_load(r[0] + x), ai = wasm_v128_load(m[0] + x);
    const v128_t t1r = wasm_f32x4_add(qr[0], qr[3]), t1i = wasm_f32x4_add(qi[0], qi[3]), t2r = wasm_f32x4_add(qr[1], qr[2]), t2i = wasm_f32x4_add(qi[1], qi[2]);
    const v128_t t3r = wasm_f32x4_sub(qr[0], qr[3]), t3i = wasm_f32x4_sub(qi[0], qi[3]), t4r = wasm_f32x4_sub(qr[1], qr[2]), t4i = wasm_f32x4_sub(qi[1], qi[2]);
    const v128_t m1r = wasm_f32x4_add(wasm_f32x4_add(ar, wasm_f32x4_mul(vc1, t1r)), wasm_f32x4_mul(vc2, t2r));
    const v128_t m1i = wasm_f32x4_add(wasm_f32x4_add(ai, wasm_f32x4_mul(vc1, t1i)), wasm_f32x4_mul(vc2, t2i));
    const v128_t m2r = wasm_f32x4_add(wasm_f32x4_add(ar, wasm_f32x4_mul(vc2, t1r)), wasm_f32x4_mul(vc1, t2r));
    const v128_t m2i = wasm_f32x4_add(wasm_f32x4_add(ai, wasm_f32x4_mul(vc2, t1i)), wasm_f32x4_mul(vc1, t2i));
    const v128_t n1r = wasm_f32x4_add(wasm_f32x4_mul(vs1, t3r), wasm_f32x4_mul(vs2, t4r)), n1i = wasm_f32x4_add(wasm_f32x4_mul(vs1, t3i), wasm_f32x4_mul(vs2, t4i));
    const v128_t n2r = wasm_f32x4_sub(wasm_f32x4_mul(vs2, t3r), wasm_f32x4_mul(vs1, t4r)), n2i = wasm_f32x4_sub(wasm_f32x4_mul(vs2, t3i), wasm_f32x4_mul(vs1, t4i));
    wasm_v128_store(r[0] + x, wasm_f32x4_add(wasm_f32x4_add(ar, t1r), t2r)); wasm_v128_store(m[0] + x, wasm_f32x4_add(wasm_f32x4_add(ai, t1i), t2i));
    wasm_v128_store(r[1] + x, wasm_f32x4_add(m1r, n1i)); wasm_v128_store(m[1] + x, wasm_f32x4_sub(m1i, n1r));
    wasm_v128_store(r[4] + x, wasm_f32x4_sub(m1r, n1i)); wasm_v128_store(m[4] + x, wasm_f32x4_add(m1i, n1r));
    wasm_v128_store(r[2] + x, wasm_f32x4_add(m2r, n2i)); wasm_v128_store(m[2] + x, wasm_f32x4_sub(m2i, n2r));
    wasm_v128_store(r[3] + x, wasm_f32x4_sub(m2r, n2i)); wasm_v128_store(m[3] + x, wasm_f32x4_add(m2i, n2r));
  }
#endif
  for (; x < cols; x++) {
    float qr[4], qi[4];
    for (int j = 0; j < 4; j++) {
      const float pr = r[j + 1][x], pi = m[j + 1][x];
      qr[j] = unit ? pr : pr * w[2 * j] - pi * w[2 * j + 1];
      qi[j] = unit ? pi : pr * w[2 * j + 1] + pi * w[2 * j];
    }
    const float ar = r[0][x], ai = m[0][x];
    const float t1r = qr[0] + qr[3], t1i = qi[0] + qi[3], t2r = qr[1] + qr[2], t2i = qi[1] + qi[2];
    const float t3r = qr[0] - qr[3], t3i = qi[0] - qi[3], t4r = qr[1] - qr[2], t4i = qi[1] - qi[2];
    const float m1r = ar + c1 * t1r + c2 * t2r, m1i = ai + c1 * t1i + c2 * t2i;
    const float m2r = ar + c2 * t1r + c1 * t2r, m2i = ai + c2 * t1i + c1 * t2i;
    const float n1r = s1 * t3r + s2 * t4r, n1i = s1 * t3i + s2 * t4i;
    const float n2r = s2 * t3r - s1 * t4r, n2i = s2 * t3i - s1 * t4i;
    r[0][x] = ar + t1r + t2r; m[0][x] = ai + t1i + t2i;
    r[1][x] = m1r + n1i; m[1][x] = m1i - n1r;
    r[4][x] = m1r - n1i; m[4][x] = m1i + n1r;
    r[2][x] = m2r + n2i; m[2][x] = m2i - n2r;
    r[3][x] = m2r - n2i; m[3][x] = m2i + n2r;
  }
}

// One stage of the mixed-radix schedule (2, 3 or 5), over the rows [from, from + span). The short stages are run group by group
// while a group sits in the first-level cache, exactly as the radix-2 schedule does.
static void do_stage(const plan_t *p, float *re, float *im, int rows, int cols, int pitch, int inverse, float sgn, int s, int from, int span) {
  const int len = p->len[s], r = p->st[s], h = len / r, step = rows / len;
  for (int i0 = from; i0 < from + span; i0 += len) for (int k = 0; k < h; k++) {
    float w[8];
    if (r == 2) {
      tw_of(p, rows, k * step, inverse, &w[0], &w[1]);
      rows2(re + (size_t)(i0 + k) * pitch, im + (size_t)(i0 + k) * pitch, re + (size_t)(i0 + k + h) * pitch, im + (size_t)(i0 + k + h) * pitch, cols, w[0], w[1], k == 0);
      continue;
    }
    // ai: r rows h apart, the butterfly's r - 1 twiddles W^(j k step), then rows3 or rows5
    float *rr[5], *mm[5];
    for (int j = 0; j < r; j++) { rr[j] = re + (size_t)(i0 + k + j * h) * pitch; mm[j] = im + (size_t)(i0 + k + j * h) * pitch; }
    for (int j = 1; j < r; j++) tw_of(p, rows, j * k * step, inverse, &w[2 * j - 2], &w[2 * j - 1]);
    if (r == 3) rows3(rr, mm, cols, w, sgn, k == 0); else rows5(rr, mm, cols, w, sgn, k == 0);
  }
}

// ai: The sizes with a factor of 3 or 5: their radix-3 and radix-5 stages have no place in the power-of-two schedule
// ai: below. Same arithmetic as that schedule's butterflies, a stage a pass.
static void fft_cols_mixed(const plan_t *p, float *re, float *im, int rows, int cols, int pitch, int inverse) {
  for (int s = 0; s < p->nswp; s++) {
    float *a = re + (size_t)p->swp[2 * s] * pitch, *b = re + (size_t)p->swp[2 * s + 1] * pitch;
    float *c = im + (size_t)p->swp[2 * s] * pitch, *d = im + (size_t)p->swp[2 * s + 1] * pitch;
    for (int x = 0; x < cols; x++) { float t = a[x]; a[x] = b[x]; b[x] = t; t = c[x]; c[x] = d[x]; d[x] = t; }
  }
  const float sgn = inverse ? -1.0f : 1.0f;
  int group = 2;
  while (group * 2 <= rows && (size_t)group * 2 * cols * 2 * sizeof(float) <= 16384) group *= 2;
  int ngrp = 0;
  while (ngrp < p->nst && p->len[ngrp] <= group) ngrp++;
  const int gsz = ngrp ? p->len[ngrp - 1] : 1;
  for (int g0 = 0; g0 < rows; g0 += gsz) for (int s = 0; s < ngrp; s++) do_stage(p, re, im, rows, cols, pitch, inverse, sgn, s, g0, gsz);
  // ai: The long stages: two radix-2 stages in a row go two to a pass over four rows (rows4), as the power-of-two
  // ai: schedule runs them, with this plan's twiddles; the same butterflies on every element, so the same floats.
  for (int s = ngrp; s < p->nst; ) {
    if (p->st[s] != 2 || s + 1 >= p->nst || p->st[s + 1] != 2) { do_stage(p, re, im, rows, cols, pitch, inverse, sgn, s, 0, rows); s++; continue; }
    const int len = p->len[s], h = len / 2, step = rows / len, step2 = step / 2;
    for (int i0 = 0; i0 < rows; i0 += 2 * len) for (int k = 0; k < h; k++) {
      const int q[4] = { i0 + k, i0 + k + h, i0 + k + len, i0 + k + len + h };
      float *r[4], *m[4], w[6];
      for (int j = 0; j < 4; j++) { r[j] = re + (size_t)q[j] * pitch; m[j] = im + (size_t)q[j] * pitch; }
      tw_of(p, rows, k * step, inverse, &w[0], &w[1]);
      tw_of(p, rows, k * step2, inverse, &w[2], &w[3]);
      tw_of(p, rows, (k + h) * step2, inverse, &w[4], &w[5]);
      if (k == 0) {   // unit twiddles keep their multiply-free form, as do_stage gives them
        rows2(r[0], m[0], r[1], m[1], cols, w[0], w[1], 1); rows2(r[2], m[2], r[3], m[3], cols, w[0], w[1], 1);
        rows2(r[0], m[0], r[2], m[2], cols, w[2], w[3], 1); rows2(r[1], m[1], r[3], m[3], cols, w[4], w[5], 0);
        continue;
      }
      rows4(r, m, cols, w[0], w[1], w[2], w[3], w[4], w[5]);
    }
    s += 2;
  }
}

void fft_cols(float *re, float *im, int rows, int cols, int pitch, int inverse) {
  const plan_t *p = plan_for(rows);
  if (!p) return;
  if (rows & (rows - 1)) { fft_cols_mixed(p, re, im, rows, cols, pitch, inverse); return; }
  for (int s = 0; s < p->nswp; s++) {
    float *a = re + (size_t)p->swp[2 * s] * pitch, *b = re + (size_t)p->swp[2 * s + 1] * pitch;
    float *c = im + (size_t)p->swp[2 * s] * pitch, *d = im + (size_t)p->swp[2 * s + 1] * pitch;
    for (int x = 0; x < cols; x++) { float t = a[x]; a[x] = b[x]; b[x] = t; t = c[x]; c[x] = d[x]; d[x] = t; }
  }
  #define ROW(a, i) ((a) + (size_t)(i) * pitch)
  #define WI(k) (inverse ? -p->wi[k] : p->wi[k])
  // A stage is a pass over every row, and nine of them over a 256 KB strip is nine trips to the second-level cache
  // and back: that, not the arithmetic, is what the transform costs. Butterflies of different groups do not touch,
  // so the order between them is free. The short stages are run group by group, all of them on one group of rows
  // while it sits in the first-level cache (16 KB), and the long ones two to a pass.
  int group = 2;
  while (group * 2 <= rows && (size_t)group * 2 * cols * 2 * sizeof(float) <= 16384) group *= 2;
  for (int g0 = 0; g0 < rows; g0 += group) for (int len = 2; len <= group; len <<= 1) {
    int h = len >> 1, step = rows / len;
    for (int i0 = g0; i0 < g0 + group; i0 += len) for (int k = 0; k < h; k++) rows2(ROW(re, i0 + k), ROW(im, i0 + k), ROW(re, i0 + k + h), ROW(im, i0 + k + h), cols, p->wr[k * step], WI(k * step), k == 0);
  }
  for (int len = group * 2; len <= rows; ) {
    int h = len >> 1, step = rows / len;
    if (len * 2 > rows) {   // one stage left over
      for (int i0 = 0; i0 < rows; i0 += len) for (int k = 0; k < h; k++) rows2(ROW(re, i0 + k), ROW(im, i0 + k), ROW(re, i0 + k + h), ROW(im, i0 + k + h), cols, p->wr[k * step], WI(k * step), k == 0);
      break;
    }
    int step2 = step >> 1;   // the stage of length 2 len steps half as far through the table
    for (int i0 = 0; i0 < rows; i0 += 2 * len) for (int k = 0; k < h; k++) {
      int q[4] = { i0 + k, i0 + k + h, i0 + k + len, i0 + k + len + h };
      float *r[4] = { ROW(re, q[0]), ROW(re, q[1]), ROW(re, q[2]), ROW(re, q[3]) }, *m[4] = { ROW(im, q[0]), ROW(im, q[1]), ROW(im, q[2]), ROW(im, q[3]) };
      if (k == 0) {   // unit twiddles keep their multiply-free form, so this group of four goes the long way round
        rows2(r[0], m[0], r[1], m[1], cols, 1, 0, 1); rows2(r[2], m[2], r[3], m[3], cols, 1, 0, 1);
        rows2(r[0], m[0], r[2], m[2], cols, 1, 0, 1); rows2(r[1], m[1], r[3], m[3], cols, p->wr[h * step2], WI(h * step2), 0);
        continue;
      }
      rows4(r, m, cols, p->wr[k * step], WI(k * step), p->wr[k * step2], WI(k * step2), p->wr[(k + h) * step2], WI((k + h) * step2));
    }
    len <<= 2;
  }
  #undef ROW
  #undef WI
}

// ai: The decoder's forward transforms (fft.h fft_dec_dif, fft_dec_dit). On arm64 their own kernels and floats (fused
// ai: multiply-adds), the plan of kind 1 (stage_list_dec), no row moved for the digit reversal: the strips' transform
// ai: (pass 1) is decimation in frequency, natural rows in and frequency k left in row place[k]; the blocks' (pass 2)
// ai: decimation in time, sample t taken from row place[t] and frequencies out in order; unpack_turn reads the one
// ai: and writes the other through place (lizard_phy.c). The short stages run group by group while a group's rows sit
// ai: in the first-level cache (32 KB of rows, re and im), the long ones a stage a pass. Elsewhere both are fft_cols.
#ifdef OB_NEON
// ai: a twiddle: (qr, qi) times lanes l, l + 1 of V (re, im), or times the splats (wr, wi)
#define TWL(qr, qi, V, l) { const float32x4_t pr_ = qr, pi_ = qi; qr = vfmsq_laneq_f32(vmulq_laneq_f32(pr_, V, l), pi_, V, l + 1); qi = vfmaq_laneq_f32(vmulq_laneq_f32(pr_, V, l + 1), pi_, V, l); }
#define TWS(qr, qi, wr, wi) { const float32x4_t pr_ = qr, pi_ = qi; qr = vfmsq_f32(vmulq_f32(pr_, wr), pi_, wi); qi = vfmaq_f32(vmulq_f32(pr_, wi), pi_, wr); }
// ai: the 3-point DFT in place: a <- a + t1, b <- a - t1 / 2 - i (sqrt 3 / 2) t2, c <- ... + i (sqrt 3 / 2) t2 (t1 = b + c,
// ai: t2 = b - c); K holds sqrt 3 / 2 in lane 0 and 1 / 2 in lane 1
#define DFT3F(ar, ai, br, bi, cr, ci, K) { \
  const float32x4_t t1r = vaddq_f32(br, cr), t1i = vaddq_f32(bi, ci), t2r = vsubq_f32(br, cr), t2i = vsubq_f32(bi, ci); \
  const float32x4_t hr = vfmsq_laneq_f32(ar, t1r, K, 1), hi = vfmsq_laneq_f32(ai, t1i, K, 1); \
  ar = vaddq_f32(ar, t1r); ai = vaddq_f32(ai, t1i); \
  br = vfmaq_laneq_f32(hr, t2i, K, 0); bi = vfmsq_laneq_f32(hi, t2r, K, 0); \
  cr = vfmsq_laneq_f32(hr, t2i, K, 0); ci = vfmaq_laneq_f32(hi, t2r, K, 0); }
// ai: the 4-point DFT in place (a, b, c, d) -> (X0, X1, X2, X3), its -i a swap: t0 = a + c, t1 = a - c, t2 = b + d,
// ai: t3 = b - d; X0 = t0 + t2, X2 = t0 - t2, X1 = t1 - i t3, X3 = t1 + i t3
#define DFT4F(ar, ai, br, bi, cr, ci, dr, di) { \
  const float32x4_t t0r = vaddq_f32(ar, cr), t0i = vaddq_f32(ai, ci), t1r = vsubq_f32(ar, cr), t1i = vsubq_f32(ai, ci); \
  const float32x4_t t2r = vaddq_f32(br, dr), t2i = vaddq_f32(bi, di), t3r = vsubq_f32(br, dr), t3i = vsubq_f32(bi, di); \
  ar = vaddq_f32(t0r, t2r); ai = vaddq_f32(t0i, t2i); cr = vsubq_f32(t0r, t2r); ci = vsubq_f32(t0i, t2i); \
  br = vaddq_f32(t1r, t3i); bi = vsubq_f32(t1i, t3r); dr = vsubq_f32(t1r, t3i); di = vaddq_f32(t1i, t3r); }

// ai: The columns past the last whole vector (cols % 4; never the decoder's, whose strips and blocks are multiples of
// ai: 4): one radix-rad butterfly a column, twiddled before (dif 0) or after (dif 1) unless unit, the DFT in double.
static __attribute__((noinline)) void tail_dec(float *re, float *im, size_t st, int x0, int cols, int rad, const float *w, int unit, int dif) {
  for (int x = x0; x < cols; x++) {
    double xr[12], xi[12], yr[12], yi[12];
    for (int j = 0; j < rad; j++) { xr[j] = re[j * st + x]; xi[j] = im[j * st + x]; }
    if (!unit && !dif) for (int j = 1; j < rad; j++) { const double a = xr[j], b = xi[j]; xr[j] = a * w[2 * j - 2] - b * w[2 * j - 1]; xi[j] = a * w[2 * j - 1] + b * w[2 * j - 2]; }
    for (int k = 0; k < rad; k++) {
      yr[k] = yi[k] = 0;
      for (int j = 0; j < rad; j++) { const double a = -2 * 3.14159265358979323846 * (j * k % rad) / rad, c = cos(a), s = sin(a); yr[k] += xr[j] * c - xi[j] * s; yi[k] += xr[j] * s + xi[j] * c; }
    }
    if (!unit && dif) for (int j = 1; j < rad; j++) { const double a = yr[j], b = yi[j]; yr[j] = a * w[2 * j - 2] - b * w[2 * j - 1]; yi[j] = a * w[2 * j - 1] + b * w[2 * j - 2]; }
    for (int j = 0; j < rad; j++) { re[j * st + x] = (float)yr[j]; im[j * st + x] = (float)yi[j]; }
  }
}

// ai: The butterflies, every column, over rows j st apart from re and im: decimation in time (dif 0) twiddles rows
// ai: 1 .. r - 1 by W^(j k) (w: their (re, im) pairs) and then takes the r-point DFT; in frequency (dif 1) the DFT
// ai: first, then output j twiddled by W^(j k). unit: k = 0, no twiddles.
static inline __attribute__((always_inline)) void rows2f(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const float32x4_t wr = vdupq_n_f32(w[0]), wi = vdupq_n_f32(w[1]);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t ar = vld1q_f32(re + x), ai = vld1q_f32(im + x), br = vld1q_f32(re + st + x), bi = vld1q_f32(im + st + x);
    if (!unit && !dif) TWS(br, bi, wr, wi)
    float32x4_t yr = vsubq_f32(ar, br), yi = vsubq_f32(ai, bi);
    if (!unit && dif) TWS(yr, yi, wr, wi)
    vst1q_f32(re + x, vaddq_f32(ar, br)); vst1q_f32(im + x, vaddq_f32(ai, bi)); vst1q_f32(re + st + x, yr); vst1q_f32(im + st + x, yi);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 2, w, unit, dif);
}
static inline __attribute__((always_inline)) void rows3f(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const float32x4_t K = { 0.86602540378443864676f, 0.5f, 0, 0 }, W = vld1q_f32(w);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t ar = vld1q_f32(re + x), ai = vld1q_f32(im + x), br = vld1q_f32(re + st + x), bi = vld1q_f32(im + st + x), cr = vld1q_f32(re + 2 * st + x), ci = vld1q_f32(im + 2 * st + x);
    if (!unit && !dif) { TWL(br, bi, W, 0) TWL(cr, ci, W, 2) }
    DFT3F(ar, ai, br, bi, cr, ci, K)
    if (!unit && dif) { TWL(br, bi, W, 0) TWL(cr, ci, W, 2) }
    vst1q_f32(re + x, ar); vst1q_f32(im + x, ai); vst1q_f32(re + st + x, br); vst1q_f32(im + st + x, bi); vst1q_f32(re + 2 * st + x, cr); vst1q_f32(im + 2 * st + x, ci);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 3, w, unit, dif);
}
static inline __attribute__((always_inline)) void rows4f(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const float32x4_t w1r = vdupq_n_f32(w[0]), w1i = vdupq_n_f32(w[1]), w2r = vdupq_n_f32(w[2]), w2i = vdupq_n_f32(w[3]), w3r = vdupq_n_f32(w[4]), w3i = vdupq_n_f32(w[5]);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t ar = vld1q_f32(re + x), ai = vld1q_f32(im + x), br = vld1q_f32(re + st + x), bi = vld1q_f32(im + st + x);
    float32x4_t cr = vld1q_f32(re + 2 * st + x), ci = vld1q_f32(im + 2 * st + x), dr = vld1q_f32(re + 3 * st + x), di = vld1q_f32(im + 3 * st + x);
    if (!unit && !dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i) }
    DFT4F(ar, ai, br, bi, cr, ci, dr, di)
    if (!unit && dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i) }
    vst1q_f32(re + x, ar); vst1q_f32(im + x, ai); vst1q_f32(re + 2 * st + x, cr); vst1q_f32(im + 2 * st + x, ci);
    vst1q_f32(re + st + x, br); vst1q_f32(im + st + x, bi); vst1q_f32(re + 3 * st + x, dr); vst1q_f32(im + 3 * st + x, di);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 4, w, unit, dif);
}
// ai: the radix-4 butterfly in time writing only the outputs in need (bit j: output j), the last stage of a block's
// ai: transform where the format reads fewer rows than the transform makes (fft_dec_dit keep); 15 is rows4f's
static inline __attribute__((always_inline)) void rows4k(float *re, float *im, size_t st, int cols, const float *w, int need) {
  const float32x4_t w1r = vdupq_n_f32(w[0]), w1i = vdupq_n_f32(w[1]), w2r = vdupq_n_f32(w[2]), w2i = vdupq_n_f32(w[3]), w3r = vdupq_n_f32(w[4]), w3i = vdupq_n_f32(w[5]);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t ar = vld1q_f32(re + x), ai = vld1q_f32(im + x), br = vld1q_f32(re + st + x), bi = vld1q_f32(im + st + x);
    float32x4_t cr = vld1q_f32(re + 2 * st + x), ci = vld1q_f32(im + 2 * st + x), dr = vld1q_f32(re + 3 * st + x), di = vld1q_f32(im + 3 * st + x);
    TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i)
    if (need & 5) {
      const float32x4_t t0r = vaddq_f32(ar, cr), t0i = vaddq_f32(ai, ci), t2r = vaddq_f32(br, dr), t2i = vaddq_f32(bi, di);
      if (need & 1) { vst1q_f32(re + x, vaddq_f32(t0r, t2r)); vst1q_f32(im + x, vaddq_f32(t0i, t2i)); }
      if (need & 4) { vst1q_f32(re + 2 * st + x, vsubq_f32(t0r, t2r)); vst1q_f32(im + 2 * st + x, vsubq_f32(t0i, t2i)); }
    }
    if (need & 10) {
      const float32x4_t t1r = vsubq_f32(ar, cr), t1i = vsubq_f32(ai, ci), t3r = vsubq_f32(br, dr), t3i = vsubq_f32(bi, di);
      if (need & 2) { vst1q_f32(re + st + x, vaddq_f32(t1r, t3i)); vst1q_f32(im + st + x, vsubq_f32(t1i, t3r)); }
      if (need & 8) { vst1q_f32(re + 3 * st + x, vsubq_f32(t1r, t3i)); vst1q_f32(im + 3 * st + x, vaddq_f32(t1i, t3r)); }
    }
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 4, w, 0, 0);
}
// ai: radix 5: t1 = b + e, t2 = c + d, t3 = b - e, t4 = c - d; m1 = a + c1 t1 + c2 t2, m2 = a + c2 t1 + c1 t2 (c1 = cos 2 pi
// ai: / 5, c2 = cos 4 pi / 5); n1 = s1 t3 + s2 t4, n2 = s2 t3 - s1 t4 (s1 = sin 2 pi / 5, s2 = sin 4 pi / 5); X1 = m1 - i n1,
// ai: X4 = m1 + i n1, X2 = m2 - i n2, X3 = m2 + i n2
static inline __attribute__((always_inline)) void rows5f(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const float32x4_t K = { 0.30901699437494742410f, -0.80901699437494742410f, 0.95105651629515357212f, 0.58778525229247312917f };
  const float32x4_t W0 = vld1q_f32(w), W1 = vld1q_f32(w + 4);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t qr[4], qi[4];
    for (int j = 0; j < 4; j++) { qr[j] = vld1q_f32(re + (j + 1) * st + x); qi[j] = vld1q_f32(im + (j + 1) * st + x); }
    if (!unit && !dif) { TWL(qr[0], qi[0], W0, 0) TWL(qr[1], qi[1], W0, 2) TWL(qr[2], qi[2], W1, 0) TWL(qr[3], qi[3], W1, 2) }
    const float32x4_t ar = vld1q_f32(re + x), ai = vld1q_f32(im + x);
    const float32x4_t t1r = vaddq_f32(qr[0], qr[3]), t1i = vaddq_f32(qi[0], qi[3]), t2r = vaddq_f32(qr[1], qr[2]), t2i = vaddq_f32(qi[1], qi[2]);
    const float32x4_t t3r = vsubq_f32(qr[0], qr[3]), t3i = vsubq_f32(qi[0], qi[3]), t4r = vsubq_f32(qr[1], qr[2]), t4i = vsubq_f32(qi[1], qi[2]);
    const float32x4_t m1r = vfmaq_laneq_f32(vfmaq_laneq_f32(ar, t1r, K, 0), t2r, K, 1), m1i = vfmaq_laneq_f32(vfmaq_laneq_f32(ai, t1i, K, 0), t2i, K, 1);
    const float32x4_t m2r = vfmaq_laneq_f32(vfmaq_laneq_f32(ar, t1r, K, 1), t2r, K, 0), m2i = vfmaq_laneq_f32(vfmaq_laneq_f32(ai, t1i, K, 1), t2i, K, 0);
    const float32x4_t n1r = vfmaq_laneq_f32(vmulq_laneq_f32(t3r, K, 2), t4r, K, 3), n1i = vfmaq_laneq_f32(vmulq_laneq_f32(t3i, K, 2), t4i, K, 3);
    const float32x4_t n2r = vfmsq_laneq_f32(vmulq_laneq_f32(t3r, K, 3), t4r, K, 2), n2i = vfmsq_laneq_f32(vmulq_laneq_f32(t3i, K, 3), t4i, K, 2);
    float32x4_t y1r = vaddq_f32(m1r, n1i), y1i = vsubq_f32(m1i, n1r), y4r = vsubq_f32(m1r, n1i), y4i = vaddq_f32(m1i, n1r);
    float32x4_t y2r = vaddq_f32(m2r, n2i), y2i = vsubq_f32(m2i, n2r), y3r = vsubq_f32(m2r, n2i), y3i = vaddq_f32(m2i, n2r);
    if (!unit && dif) { TWL(y1r, y1i, W0, 0) TWL(y2r, y2i, W0, 2) TWL(y3r, y3i, W1, 0) TWL(y4r, y4i, W1, 2) }
    vst1q_f32(re + x, vaddq_f32(vaddq_f32(ar, t1r), t2r)); vst1q_f32(im + x, vaddq_f32(vaddq_f32(ai, t1i), t2i));
    vst1q_f32(re + st + x, y1r); vst1q_f32(im + st + x, y1i); vst1q_f32(re + 4 * st + x, y4r); vst1q_f32(im + 4 * st + x, y4i);
    vst1q_f32(re + 2 * st + x, y2r); vst1q_f32(im + 2 * st + x, y2i); vst1q_f32(re + 3 * st + x, y3r); vst1q_f32(im + 3 * st + x, y3i);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 5, w, unit, dif);
}
// ai: radix 6 by Good-Thomas over 2 x 3, no inner twiddles: input n = 3 n1 + 2 n2 (mod 6), so 2-point DFTs of (x0, x3),
// ai: (x2, x5), (x4, x1), then the 3-point DFT of their sums gives X0, X4, X2 and of their differences X3, X1, X5 (output
// ai: k = 3 k1 + 4 k2 mod 6)
static inline __attribute__((always_inline)) void rows6f(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const float32x4_t K = { 0.86602540378443864676f, 0.5f, 0, 0 }, W0 = vld1q_f32(w), W1 = vld1q_f32(w + 4), W2 = vcombine_f32(vld1_f32(w + 8), vdup_n_f32(0));
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t x0r = vld1q_f32(re + x), x0i = vld1q_f32(im + x), x1r = vld1q_f32(re + st + x), x1i = vld1q_f32(im + st + x);
    float32x4_t x2r = vld1q_f32(re + 2 * st + x), x2i = vld1q_f32(im + 2 * st + x), x3r = vld1q_f32(re + 3 * st + x), x3i = vld1q_f32(im + 3 * st + x);
    float32x4_t x4r = vld1q_f32(re + 4 * st + x), x4i = vld1q_f32(im + 4 * st + x), x5r = vld1q_f32(re + 5 * st + x), x5i = vld1q_f32(im + 5 * st + x);
    if (!unit && !dif) { TWL(x1r, x1i, W0, 0) TWL(x2r, x2i, W0, 2) TWL(x3r, x3i, W1, 0) TWL(x4r, x4i, W1, 2) TWL(x5r, x5i, W2, 0) }
    float32x4_t u0r = vaddq_f32(x0r, x3r), u0i = vaddq_f32(x0i, x3i), v0r = vsubq_f32(x0r, x3r), v0i = vsubq_f32(x0i, x3i);
    float32x4_t u1r = vaddq_f32(x2r, x5r), u1i = vaddq_f32(x2i, x5i), v1r = vsubq_f32(x2r, x5r), v1i = vsubq_f32(x2i, x5i);
    float32x4_t u2r = vaddq_f32(x4r, x1r), u2i = vaddq_f32(x4i, x1i), v2r = vsubq_f32(x4r, x1r), v2i = vsubq_f32(x4i, x1i);
    DFT3F(u0r, u0i, u1r, u1i, u2r, u2i, K)   // ai: X0, X4, X2
    DFT3F(v0r, v0i, v1r, v1i, v2r, v2i, K)   // ai: X3, X1, X5
    if (!unit && dif) { TWL(v1r, v1i, W0, 0) TWL(u2r, u2i, W0, 2) TWL(v0r, v0i, W1, 0) TWL(u1r, u1i, W1, 2) TWL(v2r, v2i, W2, 0) }
    vst1q_f32(re + x, u0r); vst1q_f32(im + x, u0i); vst1q_f32(re + 4 * st + x, u1r); vst1q_f32(im + 4 * st + x, u1i); vst1q_f32(re + 2 * st + x, u2r); vst1q_f32(im + 2 * st + x, u2i);
    vst1q_f32(re + 3 * st + x, v0r); vst1q_f32(im + 3 * st + x, v0i); vst1q_f32(re + st + x, v1r); vst1q_f32(im + st + x, v1i); vst1q_f32(re + 5 * st + x, v2r); vst1q_f32(im + 5 * st + x, v2i);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 6, w, unit, dif);
}
// ai: radix 12 without twiddles (a plan's first stage only) by Good-Thomas over 4 x 3: input n = 3 n1 + 4 n2 (mod 12), the
// ai: 4-point DFTs over n1, then 3-point DFTs over n2; output k = 9 k1 + 4 k2 (mod 12)
static inline __attribute__((always_inline)) void rows12u(float *re, float *im, size_t st, int cols) {
  const float32x4_t K = { 0.86602540378443864676f, 0.5f, 0, 0 };
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t a0r = vld1q_f32(re + x), a0i = vld1q_f32(im + x), a1r = vld1q_f32(re + 3 * st + x), a1i = vld1q_f32(im + 3 * st + x);
    float32x4_t a2r = vld1q_f32(re + 6 * st + x), a2i = vld1q_f32(im + 6 * st + x), a3r = vld1q_f32(re + 9 * st + x), a3i = vld1q_f32(im + 9 * st + x);
    DFT4F(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i)
    float32x4_t b0r = vld1q_f32(re + 4 * st + x), b0i = vld1q_f32(im + 4 * st + x), b1r = vld1q_f32(re + 7 * st + x), b1i = vld1q_f32(im + 7 * st + x);
    float32x4_t b2r = vld1q_f32(re + 10 * st + x), b2i = vld1q_f32(im + 10 * st + x), b3r = vld1q_f32(re + st + x), b3i = vld1q_f32(im + st + x);
    DFT4F(b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i)
    float32x4_t c0r = vld1q_f32(re + 8 * st + x), c0i = vld1q_f32(im + 8 * st + x), c1r = vld1q_f32(re + 11 * st + x), c1i = vld1q_f32(im + 11 * st + x);
    float32x4_t c2r = vld1q_f32(re + 2 * st + x), c2i = vld1q_f32(im + 2 * st + x), c3r = vld1q_f32(re + 5 * st + x), c3i = vld1q_f32(im + 5 * st + x);
    DFT4F(c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i)
    DFT3F(a0r, a0i, b0r, b0i, c0r, c0i, K)   // ai: X0, X4, X8
    vst1q_f32(re + x, a0r); vst1q_f32(im + x, a0i); vst1q_f32(re + 4 * st + x, b0r); vst1q_f32(im + 4 * st + x, b0i); vst1q_f32(re + 8 * st + x, c0r); vst1q_f32(im + 8 * st + x, c0i);
    DFT3F(a1r, a1i, b1r, b1i, c1r, c1i, K)   // ai: X9, X1, X5
    vst1q_f32(re + 9 * st + x, a1r); vst1q_f32(im + 9 * st + x, a1i); vst1q_f32(re + st + x, b1r); vst1q_f32(im + st + x, b1i); vst1q_f32(re + 5 * st + x, c1r); vst1q_f32(im + 5 * st + x, c1i);
    DFT3F(a2r, a2i, b2r, b2i, c2r, c2i, K)   // ai: X6, X10, X2
    vst1q_f32(re + 6 * st + x, a2r); vst1q_f32(im + 6 * st + x, a2i); vst1q_f32(re + 10 * st + x, b2r); vst1q_f32(im + 10 * st + x, b2i); vst1q_f32(re + 2 * st + x, c2r); vst1q_f32(im + 2 * st + x, c2i);
    DFT3F(a3r, a3i, b3r, b3i, c3r, c3i, K)   // ai: X3, X7, X11
    vst1q_f32(re + 3 * st + x, a3r); vst1q_f32(im + 3 * st + x, a3i); vst1q_f32(re + 7 * st + x, b3r); vst1q_f32(im + 7 * st + x, b3i); vst1q_f32(re + 11 * st + x, c3r); vst1q_f32(im + 11 * st + x, c3i);
  }
  if (x < cols) tail_dec(re, im, st, x, cols, 12, 0, 1, 0);
}

// ai: each stage's twiddles in the order its butterflies take them (k, then j = 1 .. r - 1, re and im), made once a
// ai: size and thread, so a butterfly reads its floats in a row
typedef struct { int rows; float *tw[24]; } dec_tw_t;
static _Thread_local dec_tw_t dec_tws[8];
static const dec_tw_t *dec_tw_for(const plan_t *p) {
  int slot = 0;
  for (; slot < 8 && dec_tws[slot].rows; slot++) if (dec_tws[slot].rows == p->rows) return &dec_tws[slot];
  if (slot == 8) slot = 7;
  dec_tw_t *t = &dec_tws[slot];
  for (int s = 0; s < 24; s++) { free(t->tw[s]); t->tw[s] = 0; }
  t->rows = p->rows;
  for (int s = 0; s < p->nst; s++) {
    const int r = p->st[s], len = p->len[s], h = len / r, step = p->rows / len;
    float *w = t->tw[s] = malloc((size_t)h * (r - 1) * 2 * sizeof(float) + 8 * sizeof(float));   // ai: 8 past the end: a vector load of the last pairs
    if (!w) { t->rows = 0; return 0; }
    for (int k = 0; k < h; k++) for (int j = 1; j < r; j++) tw_of(p, p->rows, j * k * step, 0, &w[(k * (r - 1) + j - 1) * 2], &w[(k * (r - 1) + j - 1) * 2 + 1]);
    for (int i = 0; i < 8; i++) w[(size_t)h * (r - 1) * 2 + i] = 0;
  }
  return t;
}
// ai: one stage over the rows [from, from + span), in time (dif 0) or frequency (dif 1): a group's r row pointers
// ai: stepped down a row a butterfly
static inline __attribute__((always_inline)) void stage_run(const plan_t *p, const float *tw, float *re, float *im, int cols, int pitch, int s, int from, int span, int dif) {
  const int r = p->st[s], len = p->len[s], h = len / r;
  const size_t st = (size_t)h * pitch;
  for (int i0 = from; i0 < from + span; i0 += len) {
    float *br = re + (size_t)i0 * pitch, *bi = im + (size_t)i0 * pitch;
    const float *w = tw;
    switch (r) {
      case 4: for (int k = 0; k < h; k++, w += 6, br += pitch, bi += pitch) { if (k) rows4f(br, bi, st, cols, w, 0, dif); else rows4f(br, bi, st, cols, w, 1, dif); } break;
      case 6: for (int k = 0; k < h; k++, w += 10, br += pitch, bi += pitch) { if (k) rows6f(br, bi, st, cols, w, 0, dif); else rows6f(br, bi, st, cols, w, 1, dif); } break;
      case 3: for (int k = 0; k < h; k++, w += 4, br += pitch, bi += pitch) { if (k) rows3f(br, bi, st, cols, w, 0, dif); else rows3f(br, bi, st, cols, w, 1, dif); } break;
      case 5: for (int k = 0; k < h; k++, w += 8, br += pitch, bi += pitch) { if (k) rows5f(br, bi, st, cols, w, 0, dif); else rows5f(br, bi, st, cols, w, 1, dif); } break;
      case 2: for (int k = 0; k < h; k++, w += 2, br += pitch, bi += pitch) { if (k) rows2f(br, bi, st, cols, w, 0, dif); else rows2f(br, bi, st, cols, w, 1, dif); } break;
      default: rows12u(br, bi, st, cols); break;   // ai: 12: a first stage, h = 1
    }
  }
}
static void stage_dit(const plan_t *p, const float *tw, float *re, float *im, int cols, int pitch, int s, int from, int span) { stage_run(p, tw, re, im, cols, pitch, s, from, span, 0); }
static void stage_dif(const plan_t *p, const float *tw, float *re, float *im, int cols, int pitch, int s, int from, int span) { stage_run(p, tw, re, im, cols, pitch, s, from, span, 1); }
// ai: the last stage in time where it is a 4 and the caller reads rows [0, keep] and [rows - keep, rows) alone:
// ai: butterfly k writes output j (row k + j h) only where that row is read, and none where none is
static void stage_dit_keep(const plan_t *p, const float *tw, float *re, float *im, int cols, int pitch, int s, int keep) {
  const int rows = p->rows, h = rows / 4;
  const size_t st = (size_t)h * pitch;
  float *br = re, *bi = im;
  const float *w = tw;
  for (int k = 0; k < h; k++, w += 6, br += pitch, bi += pitch) {
    int need = 0;
    for (int j = 0; j < 4; j++) { const int i = k + j * h; need |= (i <= keep || i >= rows - keep) << j; }
    if (need == 15) { if (k) rows4f(br, bi, st, cols, w, 0, 0); else rows4f(br, bi, st, cols, w, 1, 0); }
    else if (need) rows4k(br, bi, st, cols, w, need);
  }
  (void)s;
}
// ai: the stages a group of rows sitting in the first-level cache takes (those no longer than the group), and the group
static int dec_groups(const plan_t *p, int cols, int *gsz) {
  int group = 2;
  while (group * 2 <= p->rows && (size_t)group * 2 * cols * 2 * sizeof(float) <= 32768) group *= 2;
  int ngrp = 0;
  while (ngrp < p->nst && p->len[ngrp] <= group) ngrp++;
  *gsz = ngrp ? p->len[ngrp - 1] : 1;
  return ngrp;
}
#endif
void fft_dec_dif(float *re, float *im, int rows, int cols, int pitch) {
#ifdef OB_NEON
  const plan_t *p = plan_kind(rows, 1);
  const dec_tw_t *t = p ? dec_tw_for(p) : 0;
  if (!t) return;
  int gsz;
  const int ngrp = dec_groups(p, cols, &gsz);
  for (int s = p->nst - 1; s >= ngrp; s--) stage_dif(p, t->tw[s], re, im, cols, pitch, s, 0, rows);
  if (ngrp) for (int g0 = 0; g0 < rows; g0 += gsz) for (int s = ngrp - 1; s >= 0; s--) stage_dif(p, t->tw[s], re, im, cols, pitch, s, g0, gsz);
#else
  fft_cols(re, im, rows, cols, pitch, 0);
#endif
}
void fft_dec_dit(float *re, float *im, int rows, int cols, int pitch, int keep) {
#ifdef OB_NEON
  const plan_t *p = plan_kind(rows, 1);
  const dec_tw_t *t = p ? dec_tw_for(p) : 0;
  if (!t) return;
  int gsz;
  const int ngrp = dec_groups(p, cols, &gsz);
  if (ngrp) for (int g0 = 0; g0 < rows; g0 += gsz) for (int s = 0; s < ngrp; s++) stage_dit(p, t->tw[s], re, im, cols, pitch, s, g0, gsz);
  for (int s = ngrp; s < p->nst; s++) {
    if (s == p->nst - 1 && p->st[s] == 4 && 2 * keep < rows) stage_dit_keep(p, t->tw[s], re, im, cols, pitch, s, keep);
    else stage_dit(p, t->tw[s], re, im, cols, pitch, s, 0, rows);
  }
#else
  (void)keep;
  fft_cols(re, im, rows, cols, pitch, 0);
#endif
}
const int *fft_dec_place(int rows) {
#ifdef OB_NEON
  const plan_t *p = plan_kind(rows, 1);
  return p ? p->place : 0;
#else
  (void)rows;
  return 0;
#endif
}
