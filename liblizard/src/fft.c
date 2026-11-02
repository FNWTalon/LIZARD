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
typedef struct { int rows, nst, nswp, half, kind; float *wr, *wi; int *swp; int st[24], len[24]; } plan_t;
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

// ai: kind 0 the format's schedule (2s); kind 1 the decoder's transform on arm64 (fft_cols_dec), a power of two's stages
// ai: as 4s, one 2 first where the exponent is odd
static int stage_list(int rows, int *st, int kind) {
  int k = 0;
  while (rows % 3 == 0) { st[k++] = 3; rows /= 3; }
  while (rows % 5 == 0) { st[k++] = 5; rows /= 5; }
  if (kind == 1) {
    int bits = 0;
    for (int t = rows; t > 1; t >>= 1) bits++;
    if (bits & 1) st[k++] = 2;
    for (int b = bits & 1; b < bits; b += 2) st[k++] = 4;
    return k;
  }
  for (int t = rows; t > 1; t >>= 1) st[k++] = 2;
  return k;
}

static const plan_t *plan_kind(int rows, int kind) {
  if (!size_ok(rows)) return NULL;
  int slot = 0;
  for (; slot < 8 && plans[slot].rows; slot++) if (plans[slot].rows == rows && plans[slot].kind == kind) return &plans[slot];
  if (slot == 8) slot = 7;   // more sizes than slots: the last one is rebuilt. LIZARD uses one size.
  plan_t *p = &plans[slot];
  free(p->wr); free(p->wi); free(p->swp);
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
  for (int i = 0; i < rows; i++) { at[i] = i; cur[i] = i; }
  p->nswp = 0;
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

#ifdef OB_NEON
// ai: arm64, the decoder's transform alone (fft_cols_dec): one radix-4 butterfly across four rows (decimation in time,
// ai: forward), every column: rows 1 to 3 twiddled by W^k, W^2k, W^3k (w: three (re, im) pairs) with fused
// ai: multiply-adds, then the 4-point DFT, its -i a swap: t0 = a + c, t1 = a - c, t2 = b + d, t3 = b - d;
// ai: out0 = t0 + t2, out2 = t0 - t2, out1 = t1 - i t3, out3 = t1 + i t3. unit: k = 0, no twiddles. It rounds otherwise
// ai: than the format's radix-2 pairs: its check is the definition (test/fft_dec.c) and the blocks a decode verifies.
static inline __attribute__((always_inline)) void rows4f(float *const r[4], float *const m[4], int cols, const float *w, int unit) {
  const float32x4_t w1r = vdupq_n_f32(w[0]), w1i = vdupq_n_f32(w[1]), w2r = vdupq_n_f32(w[2]), w2i = vdupq_n_f32(w[3]), w3r = vdupq_n_f32(w[4]), w3i = vdupq_n_f32(w[5]);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t br = vld1q_f32(r[1] + x), bi = vld1q_f32(m[1] + x), cr = vld1q_f32(r[2] + x), ci = vld1q_f32(m[2] + x), dr = vld1q_f32(r[3] + x), di = vld1q_f32(m[3] + x);
    if (!unit) {
      const float32x4_t b2r = vfmsq_f32(vmulq_f32(br, w1r), bi, w1i), b2i = vfmaq_f32(vmulq_f32(br, w1i), bi, w1r);
      const float32x4_t c2r = vfmsq_f32(vmulq_f32(cr, w2r), ci, w2i), c2i = vfmaq_f32(vmulq_f32(cr, w2i), ci, w2r);
      const float32x4_t d2r = vfmsq_f32(vmulq_f32(dr, w3r), di, w3i), d2i = vfmaq_f32(vmulq_f32(dr, w3i), di, w3r);
      br = b2r; bi = b2i; cr = c2r; ci = c2i; dr = d2r; di = d2i;
    }
    const float32x4_t ar = vld1q_f32(r[0] + x), ai = vld1q_f32(m[0] + x);
    const float32x4_t t0r = vaddq_f32(ar, cr), t0i = vaddq_f32(ai, ci), t1r = vsubq_f32(ar, cr), t1i = vsubq_f32(ai, ci);
    const float32x4_t t2r = vaddq_f32(br, dr), t2i = vaddq_f32(bi, di), t3r = vsubq_f32(br, dr), t3i = vsubq_f32(bi, di);
    vst1q_f32(r[0] + x, vaddq_f32(t0r, t2r)); vst1q_f32(m[0] + x, vaddq_f32(t0i, t2i));
    vst1q_f32(r[2] + x, vsubq_f32(t0r, t2r)); vst1q_f32(m[2] + x, vsubq_f32(t0i, t2i));
    vst1q_f32(r[1] + x, vaddq_f32(t1r, t3i)); vst1q_f32(m[1] + x, vsubq_f32(t1i, t3r));
    vst1q_f32(r[3] + x, vsubq_f32(t1r, t3i)); vst1q_f32(m[3] + x, vaddq_f32(t1i, t3r));
  }
  for (; x < cols; x++) {
    float br = r[1][x], bi = m[1][x], cr = r[2][x], ci = m[2][x], dr = r[3][x], di = m[3][x];
    if (!unit) {
      const float b2r = fmaf(-bi, w[1], br * w[0]), b2i = fmaf(bi, w[0], br * w[1]);
      const float c2r = fmaf(-ci, w[3], cr * w[2]), c2i = fmaf(ci, w[2], cr * w[3]);
      const float d2r = fmaf(-di, w[5], dr * w[4]), d2i = fmaf(di, w[4], dr * w[5]);
      br = b2r; bi = b2i; cr = c2r; ci = c2i; dr = d2r; di = d2i;
    }
    const float ar = r[0][x], ai = m[0][x], t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci, t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
    r[0][x] = t0r + t2r; m[0][x] = t0i + t2i; r[2][x] = t0r - t2r; m[2][x] = t0i - t2i;
    r[1][x] = t1r + t3i; m[1][x] = t1i - t3r; r[3][x] = t1r - t3i; m[3][x] = t1i + t3r;
  }
}
#endif

#ifdef OB_NEON
// ai: the decoder's radix-3 and radix-2 butterflies on arm64, with fused multiply-adds (forward; rows3 and rows2's
// ai: arithmetic otherwise): out0 = a + t1, out1 = a - t1 / 2 - i (sqrt 3 / 2) t2, out2 the same with + i, t1 = b + c,
// ai: t2 = b - c; and (a, b) -> (a + w b, a - w b).
static inline __attribute__((always_inline)) void rows3f(float *const r[3], float *const m[3], int cols, const float *w, int unit) {
  const float32x4_t w1r = vdupq_n_f32(w[0]), w1i = vdupq_n_f32(w[1]), w2r = vdupq_n_f32(w[2]), w2i = vdupq_n_f32(w[3]);
  const float32x4_t vk = vdupq_n_f32(0.86602540378443864676f), vh = vdupq_n_f32(0.5f);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    float32x4_t br = vld1q_f32(r[1] + x), bi = vld1q_f32(m[1] + x), cr = vld1q_f32(r[2] + x), ci = vld1q_f32(m[2] + x);
    if (!unit) {
      const float32x4_t b2r = vfmsq_f32(vmulq_f32(br, w1r), bi, w1i), b2i = vfmaq_f32(vmulq_f32(br, w1i), bi, w1r);
      const float32x4_t c2r = vfmsq_f32(vmulq_f32(cr, w2r), ci, w2i), c2i = vfmaq_f32(vmulq_f32(cr, w2i), ci, w2r);
      br = b2r; bi = b2i; cr = c2r; ci = c2i;
    }
    const float32x4_t ar = vld1q_f32(r[0] + x), ai = vld1q_f32(m[0] + x);
    const float32x4_t t1r = vaddq_f32(br, cr), t1i = vaddq_f32(bi, ci), t2r = vsubq_f32(br, cr), t2i = vsubq_f32(bi, ci);
    const float32x4_t hr = vfmsq_f32(ar, vh, t1r), hi = vfmsq_f32(ai, vh, t1i);
    vst1q_f32(r[0] + x, vaddq_f32(ar, t1r)); vst1q_f32(m[0] + x, vaddq_f32(ai, t1i));
    vst1q_f32(r[1] + x, vfmaq_f32(hr, vk, t2i)); vst1q_f32(m[1] + x, vfmsq_f32(hi, vk, t2r));
    vst1q_f32(r[2] + x, vfmsq_f32(hr, vk, t2i)); vst1q_f32(m[2] + x, vfmaq_f32(hi, vk, t2r));
  }
  for (; x < cols; x++) {
    float br = r[1][x], bi = m[1][x], cr = r[2][x], ci = m[2][x];
    if (!unit) {
      const float b2r = fmaf(-bi, w[1], br * w[0]), b2i = fmaf(bi, w[0], br * w[1]), c2r = fmaf(-ci, w[3], cr * w[2]), c2i = fmaf(ci, w[2], cr * w[3]);
      br = b2r; bi = b2i; cr = c2r; ci = c2i;
    }
    const float ar = r[0][x], ai = m[0][x], t1r = br + cr, t1i = bi + ci, t2r = br - cr, t2i = bi - ci, k = 0.86602540378443864676f;
    const float hr = fmaf(-0.5f, t1r, ar), hi = fmaf(-0.5f, t1i, ai);
    r[0][x] = ar + t1r; m[0][x] = ai + t1i;
    r[1][x] = fmaf(k, t2i, hr); m[1][x] = fmaf(-k, t2r, hi);
    r[2][x] = fmaf(-k, t2i, hr); m[2][x] = fmaf(k, t2r, hi);
  }
}
static inline __attribute__((always_inline)) void rows2f(float *ar, float *ai, float *br, float *bi, int cols, float wr, float wi, int unit) {
  if (unit) { rows2(ar, ai, br, bi, cols, 1, 0, 1); return; }
  const float32x4_t vwr = vdupq_n_f32(wr), vwi = vdupq_n_f32(wi);
  int x = 0;
  for (; x + 4 <= cols; x += 4) {
    const float32x4_t pr = vld1q_f32(ar + x), pi = vld1q_f32(ai + x), qr = vld1q_f32(br + x), qi = vld1q_f32(bi + x);
    const float32x4_t xr = vfmsq_f32(vmulq_f32(qr, vwr), qi, vwi), xi = vfmaq_f32(vmulq_f32(qr, vwi), qi, vwr);
    vst1q_f32(br + x, vsubq_f32(pr, xr)); vst1q_f32(bi + x, vsubq_f32(pi, xi)); vst1q_f32(ar + x, vaddq_f32(pr, xr)); vst1q_f32(ai + x, vaddq_f32(pi, xi));
  }
  for (; x < cols; x++) {
    const float xr = fmaf(-bi[x], wi, br[x] * wr), xi = fmaf(bi[x], wr, br[x] * wi);
    br[x] = ar[x] - xr; bi[x] = ai[x] - xi; ar[x] = ar[x] + xr; ai[x] = ai[x] + xi;
  }
}
#endif

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

// ai: The decoder's forward transform: on arm64 the radix-4 schedule (its own floats, fused multiply-adds), its short stages group by group in the
// ai: first-level cache as fft_cols_mixed runs them, the long ones a stage a pass; elsewhere, and inverse, fft_cols.
#ifdef OB_NEON
// ai: each stage's twiddles in the order its butterflies take them (k, then j = 1 .. r - 1, re and im), made once a
// ai: size and thread, so a butterfly reads its three or four floats in a row
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
    float *w = t->tw[s] = malloc((size_t)h * (r - 1) * 2 * sizeof(float));
    if (!w) { t->rows = 0; return 0; }
    for (int k = 0; k < h; k++) for (int j = 1; j < r; j++) tw_of(p, p->rows, j * k * step, 0, &w[(k * (r - 1) + j - 1) * 2], &w[(k * (r - 1) + j - 1) * 2 + 1]);
  }
  return t;
}
// ai: one stage over the rows [from, from + span): a group's r row pointers stepped down a row a butterfly
static void stage_dec(const plan_t *p, const float *tw, float *re, float *im, int cols, int pitch, int s, int from, int span) {
  const int r = p->st[s], len = p->len[s], h = len / r;
  for (int i0 = from; i0 < from + span; i0 += len) {
    float *rr[5], *mm[5];
    for (int j = 0; j < r; j++) { rr[j] = re + (size_t)(i0 + j * h) * pitch; mm[j] = im + (size_t)(i0 + j * h) * pitch; }
    const float *w = tw;
    switch (r) {
      case 4: for (int k = 0; k < h; k++, w += 6) { rows4f(rr, mm, cols, w, k == 0); for (int j = 0; j < 4; j++) { rr[j] += pitch; mm[j] += pitch; } } break;
      case 3: for (int k = 0; k < h; k++, w += 4) { rows3f(rr, mm, cols, w, k == 0); for (int j = 0; j < 3; j++) { rr[j] += pitch; mm[j] += pitch; } } break;
      case 2: for (int k = 0; k < h; k++, w += 2) { rows2f(rr[0], mm[0], rr[1], mm[1], cols, w[0], w[1], k == 0); rr[0] += pitch; mm[0] += pitch; rr[1] += pitch; mm[1] += pitch; } break;
      default: for (int k = 0; k < h; k++, w += 8) { rows5(rr, mm, cols, w, 1.0f, k == 0); for (int j = 0; j < 5; j++) { rr[j] += pitch; mm[j] += pitch; } } break;
    }
  }
}
#endif
void fft_cols_dec(float *re, float *im, int rows, int cols, int pitch) {
#ifdef OB_NEON
  const plan_t *p = plan_kind(rows, 1);
  const dec_tw_t *t = p ? dec_tw_for(p) : 0;
  if (!t) return;
  for (int s = 0; s < p->nswp; s++) {
    float *a = re + (size_t)p->swp[2 * s] * pitch, *b = re + (size_t)p->swp[2 * s + 1] * pitch;
    float *c = im + (size_t)p->swp[2 * s] * pitch, *d = im + (size_t)p->swp[2 * s + 1] * pitch;
    for (int x = 0; x < cols; x++) { float u = a[x]; a[x] = b[x]; b[x] = u; u = c[x]; c[x] = d[x]; d[x] = u; }
  }
  int group = 2;
  while (group * 2 <= rows && (size_t)group * 2 * cols * 2 * sizeof(float) <= 16384) group *= 2;
  int ngrp = 0;
  while (ngrp < p->nst && p->len[ngrp] <= group) ngrp++;
  const int gsz = ngrp ? p->len[ngrp - 1] : 1;
  for (int g0 = 0; g0 < rows; g0 += gsz) for (int s = 0; s < ngrp; s++) stage_dec(p, t->tw[s], re, im, cols, pitch, s, g0, gsz);
  for (int s = ngrp; s < p->nst; s++) stage_dec(p, t->tw[s], re, im, cols, pitch, s, 0, rows);
#else
  fft_cols(re, im, rows, cols, pitch, 0);
#endif
}
