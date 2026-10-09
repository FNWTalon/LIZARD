// ai: Floating-point reference decoders for the format's LDPC codes (ldpc_ref.h): layered sum-product and layered
// ai: min-sum with a scale and an offset, one block row a layer as ldpc_decode runs it. Test-only; see
// ai: test/ldpc_awgn.c and test/ldpc_replay.c for what they measure.
#include "ldpc_ref.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ai: The largest message magnitude BP sends. tanh(15) is 1 - 1.9e-13, still apart from 1 in double, so a product
// ai: of them stays below 1 and atanh finite; an LLR of 30 is an error rate of 1e-13, certainty for any block here.
#define REF_MSG_MAX 30.0

int ref_init(ref_dec_t *d, const ldpc_t *c) {
  memset(d, 0, sizeof *d);
  d->c = c;
  for (int chk = 0; chk < c->m; chk++) {
    int deg = c->row_ptr[chk + 1] - c->row_ptr[chk];
    if (deg > d->deg_max) d->deg_max = deg;
  }
  d->L = malloc((size_t)c->n * sizeof(double));
  d->R = malloc((size_t)c->edges * sizeof(double));
  d->q = malloc((size_t)d->deg_max * sizeof(double));
  d->t = malloc((size_t)d->deg_max * sizeof(double));
  d->bw = malloc((size_t)(d->deg_max + 1) * sizeof(double));
  if (!d->L || !d->R || !d->q || !d->t || !d->bw) { ref_free(d); return -1; }
  return 0;
}

void ref_free(ref_dec_t *d) { free(d->L); free(d->R); free(d->q); free(d->t); free(d->bw); memset(d, 0, sizeof *d); }

static double clamp(double x, double lim) { return x > lim ? lim : x < -lim ? -lim : x; }

// ai: Sum-product at one check in the tanh form: each outgoing message is 2 atanh of the product of the other
// ai: inputs' tanh(q / 2), that product taken as prefix times suffix so no division by a tanh near zero is needed.
static void check_bp(const double *q, double *r, int deg, double *t, double *bw) {
  for (int j = 0; j < deg; j++) t[j] = tanh(0.5 * clamp(q[j], REF_MSG_MAX));
  bw[deg] = 1;
  for (int j = deg - 1; j >= 0; j--) bw[j] = bw[j + 1] * t[j];
  double fw = 1;
  for (int j = 0; j < deg; j++) {
    r[j] = clamp(2 * atanh(fw * bw[j + 1]), REF_MSG_MAX);
    fw *= t[j];
  }
}

// ai: Min-sum at one check: the smallest other magnitude, times alpha, less beta, floored at zero, with the parity
// ai: of the other signs. A zero input counts as positive, as in src/ldpc.c.
static void check_minsum(const double *q, double *r, int deg, double alpha, double beta) {
  double min1 = INFINITY, min2 = INFINITY;
  int arg = -1, sign = 0;
  for (int j = 0; j < deg; j++) {
    double a = fabs(q[j]);
    sign ^= q[j] < 0;
    if (a < min1) { min2 = min1; min1 = a; arg = j; } else if (a < min2) min2 = a;
  }
  double m1 = alpha * min1 - beta, m2 = alpha * min2 - beta;
  if (m1 < 0) m1 = 0;
  if (m2 < 0) m2 = 0;
  for (int j = 0; j < deg; j++) {
    double mag = j == arg ? m2 : m1;
    r[j] = (sign ^ (q[j] < 0)) ? -mag : mag;
  }
}

// ai: One check of a layer: take its old messages out of the posteriors, compute new ones, put them back.
static void update_check(ref_dec_t *d, int chk, const ref_opts_t *o) {
  const ldpc_t *c = d->c;
  int e0 = c->row_ptr[chk], deg = c->row_ptr[chk + 1] - e0;
  const int *col = c->col_idx + e0;
  double *R = d->R + e0, *q = d->q;
  for (int j = 0; j < deg; j++) q[j] = d->L[col[j]] - R[j];
  if (o->rule == REF_BP) check_bp(q, R, deg, d->t, d->bw);
  else check_minsum(q, R, deg, o->alpha, o->beta);
  for (int j = 0; j < deg; j++) d->L[col[j]] = q[j] + R[j];
}

static int syndrome_clear(const ldpc_t *c, const double *L) {
  for (int chk = 0; chk < c->m; chk++) {
    int par = 0;
    for (int e = c->row_ptr[chk]; e < c->row_ptr[chk + 1]; e++) par ^= L[c->col_idx[e]] < 0;
    if (par) return 0;
  }
  return 1;
}

// ai: Check chk sits at position chk / mb of block row chk % mb (ldpc_init), and the z checks of a block row share
// ai: no bit, so running them one after another is the same as ldpc_decode running them side by side.
int ref_decode(ref_dec_t *d, const float *llr, uint8_t *out, const ref_opts_t *o) {
  const ldpc_t *c = d->c;
  int mb = c->mb, z = c->z, used = -1;
  for (int v = 0; v < c->n; v++) d->L[v] = llr[v];
  memset(d->R, 0, (size_t)c->edges * sizeof(double));
  for (int it = 0; it < o->max_iter; it++) {
    for (int step = 0; step < mb; step++) {
      int r = (it & 1) ? mb - 1 - step : step;
      for (int i = 0; i < z; i++) update_check(d, i * mb + r, o);
    }
    if (o->stop && syndrome_clear(c, d->L)) { used = it + 1; break; }
  }
  if (!o->stop && syndrome_clear(c, d->L)) used = o->max_iter;
  for (int v = 0; v < c->n; v++) out[v] = d->L[v] < 0;
  return used;
}

int8_t ref_quant8(float llr) {
  float x = llr > 10 ? 10 : llr < -10 ? -10 : llr;
  return (int8_t)nearbyintf(x * 8);
}

int ref_arm_decode(const ref_arm_t *a, ldpc_t *code, ref_dec_t *ref, const int8_t *q8, const float *fl, float *buf,
                   uint8_t *out, int *spent) {
  int cap = a->ref.max_iter, used;
  if (a->today) {
    code->norm = a->norm;
    used = a->stall ? ldpc_decode_stall(code, q8, out, cap, REF_STALL_IT, REF_STALL_RATIO) : ldpc_decode(code, q8, out, cap);
  } else {
    const float *in = fl;
    if (a->quant) {
      for (int i = 0; i < code->n; i++) buf[i] = q8[i] / 8.0f;
      in = buf;
    }
    used = ref_decode(ref, in, out, &a->ref);
  }
  *spent = used > 0 ? used : used == -3 ? REF_STALL_IT : cap;
  return used;
}

ref_rng_t ref_rng_at(uint64_t stream, uint64_t index) {
  uint64_t z = stream * 0x9E3779B97F4A7C15ull + index * 0xD1B54A32D192ED03ull + 0x94D049BB133111EBull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return (ref_rng_t){ z ? z : 1 };
}

double ref_uni(ref_rng_t *r) {
  r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17;
  return (double)(r->s >> 11) / 9007199254740992.0;
}

double ref_gauss(ref_rng_t *r) { double u = 1 - ref_uni(r), v = ref_uni(r); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

// ai: 1 - E[log2(1 + e^-L)], L the LLR of a sent 0, by Simpson's rule over the Gaussian (error far under 1e-6; at
// ai: Eb/N0 1.627 dB, the capacity limit of rate 3/4, it gives 0.75005).
double ref_mi_awgn(double sigma) {
  const int N = 4000;
  const double lim = 12, h = 2 * lim / N;
  double acc = 0;
  for (int i = 0; i <= N; i++) {
    double u = -lim + i * h, L = 2 * (1 + sigma * u) / (sigma * sigma);
    double loss = L > 0 ? log1p(exp(-L)) : -L + log1p(exp(L));
    double w = (i == 0 || i == N) ? 1 : (i & 1) ? 4 : 2;
    acc += w * loss * exp(-0.5 * u * u);
  }
  return 1 - acc * h / 3 / sqrt(2 * M_PI) / M_LN2;
}

double ref_sigma_for(double mi) {
  double lo = 0.05, hi = 5;
  for (int i = 0; i < 60; i++) { double s = 0.5 * (lo + hi); if (ref_mi_awgn(s) > mi) lo = s; else hi = s; }
  return 0.5 * (lo + hi);
}
