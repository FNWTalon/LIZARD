#include "fft.h"
#include <math.h>
#include <stdlib.h>
#include "simd.h"

// W[j] = exp(-2 pi i j / rows), j < rows / 2. One table serves every stage: a stage of length len steps through it
// by rows / len. Computed in double, which the old recurrence (one multiply per butterfly) was not.
// st: the radix of each stage, the SHORTEST len first. Any factor of 2 or 4 left over by the radix goes first, so
// every stage after it is the full radix. 512 is 8^3, so radix 8 divides it exactly; radix 4 takes it as 2 x 4^4.
// ai: A size with a factor of 3 (384, 768, 1536) takes one radix-3 stage, the shortest, ahead of the powers of two,
// ai: at either radix; every power of two keeps the schedule above.
// swp: the input permutation as a list of row swaps. The digit reversal stops being its own inverse as soon as the
// radices differ, so it cannot be applied by walking i < rev[i] the way bit reversal can; the plan works the swaps
// out once, by putting the right row in each place in turn, and the transform just performs them.
typedef struct { int rows, radix, nst, nswp; float *wr, *wi; int *swp; int st[16], len[16]; } plan_t;
// ai: each thread's own (a native receiver transforms on several): a plan is built on first use and a ninth size
// ai: replaces the eighth, neither of which two threads could share
static _Thread_local plan_t plans[8];
static int radix_want = 2;

void fft_set_radix(int r) { radix_want = r == 4 ? r : 2; }   // 8 once rows8x exists; do_stage's arrays are already sized for it
int fft_get_radix(void) { return radix_want; }

// ai: 2^k or 3 x 2^k: the sizes a stage list can be made of.
static int size_ok(int rows) {
  if (rows % 3 == 0) rows /= 3;
  return rows >= 1 && !(rows & (rows - 1));
}

static int stage_list(int rows, int radix, int *st) {
  int k = 0, bits = 0;
  if (rows % 3 == 0) { st[k++] = 3; rows /= 3; }
  for (int t = rows; t > 1; t >>= 1) bits++;
  const int per = radix == 8 ? 3 : radix == 4 ? 2 : 1, lead = bits % per;
  if (lead) st[k++] = 1 << lead;
  for (int b = lead; b < bits; b += per) st[k++] = radix;
  return k;
}

static const plan_t *plan_for(int rows, int radix) {
  if (!size_ok(rows)) return NULL;
  int slot = 0;
  for (; slot < 8 && plans[slot].rows; slot++) if (plans[slot].rows == rows && plans[slot].radix == radix) return &plans[slot];
  if (slot == 8) slot = 7;   // more sizes than slots: the last one is rebuilt. FOCUS uses one size.
  plan_t *p = &plans[slot];
  free(p->wr); free(p->wi); free(p->swp);
  p->rows = rows; p->radix = radix;
  p->wr = malloc((size_t)(rows / 2 + 1) * sizeof(float)); p->wi = malloc((size_t)(rows / 2 + 1) * sizeof(float));
  p->swp = malloc((size_t)2 * rows * sizeof(int));
  int *rev = malloc((size_t)rows * sizeof(int)), *at = malloc((size_t)rows * sizeof(int)), *cur = malloc((size_t)rows * sizeof(int));
  if (!p->wr || !p->wi || !p->swp || !rev || !at || !cur) { free(rev); free(at); free(cur); p->rows = 0; return NULL; }
  for (int j = 0; j < rows / 2; j++) { double a = -2 * 3.14159265358979323846 * j / rows; p->wr[j] = (float)cos(a); p->wi[j] = (float)sin(a); }
  p->nst = stage_list(rows, radix, p->st);
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

// W_rows^j for any j < rows, from a table that holds only the first half circle: past it the sign flips.
static inline void tw_of(const plan_t *p, int rows, int j, int inverse, float *tr, float *ti) {
  float s = 1.0f;
  if (j >= rows / 2) { j -= rows / 2; s = -1.0f; }
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

// One radix-4 butterfly across four rows, every column: three complex multiplies where two fused radix-2 stages
// spend four, and the rotation by -i inside the four-point core is a swap and a sign rather than a multiply. The
// memory traffic is the same as rows4's. It is about 15% fewer real operations, not the 25% the multiply count
// alone suggests: the adds outnumber the multiplies and do not shrink.
// s is the sign of that internal rotation, +1 forward and -1 inverse, so no branch reaches the loop.
static void rows4x(float *const r[4], float *const m[4], int cols, const float *w, float s, int unit) {
  const float w1r = w[0], w1i = w[1], w2r = w[2], w2i = w[3], w3r = w[4], w3i = w[5];
  int x = 0;
#ifdef OB_SIMD
  const v128_t vs = wasm_f32x4_splat(s), vns = wasm_f32x4_splat(-s);
  // k = 0: every twiddle is 1 and the butterfly is the four-point core alone, no multiply at all. Worth its own
  // path because it is one butterfly in h, and h is 2 in the shortest stage: half of them.
  if (unit) {
    for (; x + 4 <= cols; x += 4) {
      const v128_t ar = wasm_v128_load(r[0] + x), ai = wasm_v128_load(m[0] + x);
      const v128_t br = wasm_v128_load(r[1] + x), bi = wasm_v128_load(m[1] + x);
      const v128_t cr = wasm_v128_load(r[2] + x), ci = wasm_v128_load(m[2] + x);
      const v128_t dr = wasm_v128_load(r[3] + x), di = wasm_v128_load(m[3] + x);
      const v128_t t0r = wasm_f32x4_add(ar, cr), t0i = wasm_f32x4_add(ai, ci), t1r = wasm_f32x4_sub(ar, cr), t1i = wasm_f32x4_sub(ai, ci);
      const v128_t t2r = wasm_f32x4_add(br, dr), t2i = wasm_f32x4_add(bi, di), t3r = wasm_f32x4_sub(br, dr), t3i = wasm_f32x4_sub(bi, di);
      const v128_t rr = wasm_f32x4_mul(vs, t3i), ri = wasm_f32x4_mul(vns, t3r);
      wasm_v128_store(r[0] + x, wasm_f32x4_add(t0r, t2r)); wasm_v128_store(m[0] + x, wasm_f32x4_add(t0i, t2i));
      wasm_v128_store(r[1] + x, wasm_f32x4_add(t1r, rr));  wasm_v128_store(m[1] + x, wasm_f32x4_add(t1i, ri));
      wasm_v128_store(r[2] + x, wasm_f32x4_sub(t0r, t2r)); wasm_v128_store(m[2] + x, wasm_f32x4_sub(t0i, t2i));
      wasm_v128_store(r[3] + x, wasm_f32x4_sub(t1r, rr));  wasm_v128_store(m[3] + x, wasm_f32x4_sub(t1i, ri));
    }
  } else
  // Twiddled rows first and in pairs, so that b, d and their twiddles are done with before c and a are loaded.
  // Holding all eight rows and all six twiddle splats at once is fourteen live vectors, which spills where there
  // are sixteen registers, and that gave back more than the multiply saving.
  for (; x + 4 <= cols; x += 4) {
    v128_t t2r, t2i, t3r, t3i;
    {
      const v128_t br = wasm_v128_load(r[1] + x), bi = wasm_v128_load(m[1] + x);
      const v128_t v1r = wasm_f32x4_splat(w1r), v1i = wasm_f32x4_splat(w1i);
      const v128_t b2r = wasm_f32x4_sub(wasm_f32x4_mul(br, v1r), wasm_f32x4_mul(bi, v1i)), b2i = wasm_f32x4_add(wasm_f32x4_mul(br, v1i), wasm_f32x4_mul(bi, v1r));
      const v128_t dr = wasm_v128_load(r[3] + x), di = wasm_v128_load(m[3] + x);
      const v128_t v3r = wasm_f32x4_splat(w3r), v3i = wasm_f32x4_splat(w3i);
      const v128_t d2r = wasm_f32x4_sub(wasm_f32x4_mul(dr, v3r), wasm_f32x4_mul(di, v3i)), d2i = wasm_f32x4_add(wasm_f32x4_mul(dr, v3i), wasm_f32x4_mul(di, v3r));
      t2r = wasm_f32x4_add(b2r, d2r); t2i = wasm_f32x4_add(b2i, d2i); t3r = wasm_f32x4_sub(b2r, d2r); t3i = wasm_f32x4_sub(b2i, d2i);
    }
    const v128_t cr = wasm_v128_load(r[2] + x), ci = wasm_v128_load(m[2] + x);
    const v128_t v2r = wasm_f32x4_splat(w2r), v2i = wasm_f32x4_splat(w2i);
    const v128_t c2r = wasm_f32x4_sub(wasm_f32x4_mul(cr, v2r), wasm_f32x4_mul(ci, v2i)), c2i = wasm_f32x4_add(wasm_f32x4_mul(cr, v2i), wasm_f32x4_mul(ci, v2r));
    const v128_t ar = wasm_v128_load(r[0] + x), ai = wasm_v128_load(m[0] + x);
    const v128_t t0r = wasm_f32x4_add(ar, c2r), t0i = wasm_f32x4_add(ai, c2i), t1r = wasm_f32x4_sub(ar, c2r), t1i = wasm_f32x4_sub(ai, c2i);
    const v128_t rr = wasm_f32x4_mul(vs, t3i), ri = wasm_f32x4_mul(vns, t3r);
    wasm_v128_store(r[0] + x, wasm_f32x4_add(t0r, t2r)); wasm_v128_store(m[0] + x, wasm_f32x4_add(t0i, t2i));
    wasm_v128_store(r[1] + x, wasm_f32x4_add(t1r, rr));  wasm_v128_store(m[1] + x, wasm_f32x4_add(t1i, ri));
    wasm_v128_store(r[2] + x, wasm_f32x4_sub(t0r, t2r)); wasm_v128_store(m[2] + x, wasm_f32x4_sub(t0i, t2i));
    wasm_v128_store(r[3] + x, wasm_f32x4_sub(t1r, rr));  wasm_v128_store(m[3] + x, wasm_f32x4_sub(t1i, ri));
  }
#endif
  for (; x < cols; x++) {
    const float ar = r[0][x], ai = m[0][x], br = r[1][x], bi = m[1][x], cr = r[2][x], ci = m[2][x], dr = r[3][x], di = m[3][x];
    const float b2r = unit ? br : br * w1r - bi * w1i, b2i = unit ? bi : br * w1i + bi * w1r;
    const float c2r = unit ? cr : cr * w2r - ci * w2i, c2i = unit ? ci : cr * w2i + ci * w2r;
    const float d2r = unit ? dr : dr * w3r - di * w3i, d2i = unit ? di : dr * w3i + di * w3r;
    const float t0r = ar + c2r, t0i = ai + c2i, t1r = ar - c2r, t1i = ai - c2i;
    const float t2r = b2r + d2r, t2i = b2i + d2i, t3r = b2r - d2r, t3i = b2i - d2i;
    const float rr = s * t3i, ri = -s * t3r;
    r[0][x] = t0r + t2r; m[0][x] = t0i + t2i;
    r[1][x] = t1r + rr;  m[1][x] = t1i + ri;
    r[2][x] = t0r - t2r; m[2][x] = t0i - t2i;
    r[3][x] = t1r - rr;  m[3][x] = t1i - ri;
  }
}

// ai: One radix-3 butterfly across three rows, every column, for the sizes with a factor of 3. With b and c twiddled,
// ai: t1 = b + c and t2 = b - c: out0 = a + t1, out1 = a - t1 / 2 - s i (sqrt 3 / 2) t2, out2 the same with + s i.
// ai: s as in rows4x, +1 forward and -1 inverse. The vector and scalar loops do the same float ops in the same order.
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

// One stage of the mixed-radix schedule, over the rows [from, from + span). The short stages are run group by group
// while a group sits in the first-level cache, exactly as the radix-2 schedule does.
static void do_stage(const plan_t *p, float *re, float *im, int rows, int cols, int pitch, int inverse, float sgn, int s, int from, int span) {
  const int len = p->len[s], r = p->st[s], h = len / r, step = rows / len;
  for (int i0 = from; i0 < from + span; i0 += len) for (int k = 0; k < h; k++) {
    float w[6];
    if (r == 2) {
      tw_of(p, rows, k * step, inverse, &w[0], &w[1]);
      rows2(re + (size_t)(i0 + k) * pitch, im + (size_t)(i0 + k) * pitch, re + (size_t)(i0 + k + h) * pitch, im + (size_t)(i0 + k + h) * pitch, cols, w[0], w[1], k == 0);
      continue;
    }
    float *r4[8], *m4[8];
    for (int j = 0; j < r; j++) { r4[j] = re + (size_t)(i0 + k + j * h) * pitch; m4[j] = im + (size_t)(i0 + k + j * h) * pitch; }
    for (int j = 1; j < r; j++) tw_of(p, rows, j * k * step, inverse, &w[2 * j - 2], &w[2 * j - 1]);
    if (r == 3) rows3(r4, m4, cols, w, sgn, k == 0);
    else rows4x(r4, m4, cols, w, sgn, k == 0);
  }
}

// radix 4 (and the leading radix-2 stage a size like 512 needs): fft_set_radix picks it, scripts/exp/fft_bench.mjs and
// scripts/pages/fft.html time it. Same output as the radix-2 path to a few last bits, not to the byte.
// ai: Every size with a factor of 3 runs here too, at either radix: its radix-3 stage has no place in the schedule below.
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
  for (int s = ngrp; s < p->nst; s++) do_stage(p, re, im, rows, cols, pitch, inverse, sgn, s, 0, rows);
}

void fft_cols(float *re, float *im, int rows, int cols, int pitch, int inverse) {
  const plan_t *p = plan_for(rows, radix_want);
  if (!p) return;
  if (p->radix != 2 || (rows & (rows - 1))) { fft_cols_mixed(p, re, im, rows, cols, pitch, inverse); return; }
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

void fft2d(float *re, float *im, int n, int inverse) {
  // Along y, turn the plane over, along y again (which was x), turn it back. Sixty-four columns at a time.
  for (int pass = 0; pass < 2; pass++) {
    for (int x0 = 0; x0 < n; x0 += 64) fft_cols(re + x0, im + x0, n, n - x0 < 64 ? n - x0 : 64, n, inverse);
    for (int y = 0; y < n; y++) for (int x = y + 1; x < n; x++) { float t = re[y * n + x]; re[y * n + x] = re[x * n + y]; re[x * n + y] = t; t = im[y * n + x]; im[y * n + x] = im[x * n + y]; im[x * n + y] = t; }
  }
  if (inverse) { float s = 1.0f / ((float)n * n); for (int i = 0; i < n * n; i++) { re[i] *= s; im[i] *= s; } }
}
