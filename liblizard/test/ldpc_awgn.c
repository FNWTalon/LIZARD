// BI-AWGN waterfall for the LDPC family: for each rate and length, the channel mutual
// information at which the frame error rate crosses 10%, which is the inner code's operating
// point under a fountain. The gap between that and the rate is what the code wastes.
// ai: When n gives the format's code (5088 to 5135), a second table follows: the same crossing on the format's code
// ai: for decoder arms (ldpc_ref.h), each on the same frames: today's decoder as focus.c runs it (the int8 soft values,
// ai: 30 iterations, the stall rule), without the rule and at 100 iterations, at other scales, float min-sum, offset
// ai: min-sum over a sweep of offsets, and sum-product (BP) on the float LLRs and on the int8 ones. Each arm's crossing
// ai: comes from a grid in channel information, 0.004 bits a bit apart, walked with a quarter of the frames to the
// ai: two points that bracket 10% and then run at the full count there, log FER interpolated between them.
// ai:   gcc -O2 -fopenmp -Wall -Wextra -o build/ldpc_awgn test/ldpc_awgn.c test/ldpc_ref.c src/ldpc.c -lm
// ai:   build/ldpc_awgn [n] [frames] [arm_frames] 2>>build/ldpc_progress.log
// ai: n 4800, frames 200, arm_frames 2000 (0 skips a table; progress on stderr). Without -fopenmp it runs on one core.
// ai: Since 2026-09-26 the rate sweep draws each frame from its own seed and integrates the channel information
// ai: exactly (it was a 200,000-sample Monte Carlo), so its numbers moved within their noise.
// ai: Runtime (2026-09-26, OMP_NUM_THREADS=16): `5088 200 8000` 97.5 s wall, output in research/results/ldpc/awgn.txt.
#include "ldpc_ref.h"
#include "../src/ldpc_base.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

// ai: One thread's code, decoders and frame buffers.
typedef struct {
  ldpc_t code;
  ref_dec_t ref;
  uint8_t *data, *cw, *out;
  int8_t *q8;
  float *fl, *buf;
} ctx_t;

static ctx_t *ctx_make(int count, int n, int rate) {
  ctx_t *ctx = calloc((size_t)count, sizeof *ctx);
  for (int t = 0; t < count; t++) {
    ctx_t *x = &ctx[t];
    if (ldpc_init(&x->code, n, rate, 1) || ref_init(&x->ref, &x->code)) { fprintf(stderr, "init failed\n"); exit(1); }
    int cn = x->code.n;
    x->data = malloc((size_t)x->code.k); x->cw = malloc((size_t)cn); x->out = malloc((size_t)cn);
    x->q8 = malloc((size_t)cn); x->fl = malloc((size_t)cn * sizeof(float)); x->buf = malloc((size_t)cn * sizeof(float));
  }
  return ctx;
}

static void ctx_free(ctx_t *ctx, int count) {
  for (int t = 0; t < count; t++) {
    ctx_t *x = &ctx[t];
    ref_free(&x->ref); ldpc_free(&x->code);
    free(x->data); free(x->cw); free(x->out); free(x->q8); free(x->fl); free(x->buf);
  }
  free(ctx);
}

// ai: How a frame's LLRs become int8 soft values: round(scale x LLR), clipped to +-clip.
typedef struct { double scale; int clip; } quant_t;
static const quant_t SOFT_STAGE = { 8, 80 }, SWEEP = { 8, 127 }, HALF = { 4, 80 };

// ai: Frame f of a stream through the channel at noise sigma: its codeword, the LLRs and their int8 soft values.
static void make_frame(ctx_t *x, uint64_t stream, int f, double sigma, quant_t q) {
  ref_rng_t r = ref_rng_at(stream, (uint64_t)f);
  for (int i = 0; i < x->code.k; i++) x->data[i] = ref_uni(&r) < 0.5;
  ldpc_encode(&x->code, x->data, x->cw);
  for (int i = 0; i < x->code.n; i++) {
    double y = (x->cw[i] ? -1 : 1) + sigma * ref_gauss(&r);
    x->fl[i] = (float)(2 * y / (sigma * sigma));
    double v = nearbyint(q.scale * (double)x->fl[i]);
    x->q8[i] = (int8_t)(v > q.clip ? q.clip : v < -q.clip ? -q.clip : v);
  }
}

// ai: An arm's count at one noise level: frames run, frames not decoded to the truth, clear syndromes that were not
// ai: the truth, and iterations run.
typedef struct { int frames, fails, wrong; double its; } tally_t;

static double fer(const tally_t *t) { return t->frames ? (double)t->fails / t->frames : 1; }

// ai: Runs arm a on frames [t->frames, upto) of the stream at noise sigma, soft values by q, and adds them to t.
static void tally_to(ctx_t *ctx, const ref_arm_t *a, uint64_t stream, double sigma, quant_t q, tally_t *t, int upto) {
  int fails = 0, wrong = 0;
  double its = 0;
#pragma omp parallel for schedule(dynamic, 2) reduction(+ : fails, wrong, its)
  for (int f = t->frames; f < upto; f++) {
    ctx_t *x = &ctx[omp_get_thread_num()];
    make_frame(x, stream, f, sigma, q);
    int spent, used = ref_arm_decode(a, &x->code, &x->ref, x->q8, x->fl, x->buf, x->out, &spent);
    int right = used > 0 && !memcmp(x->out, x->cw, (size_t)x->code.n);
    fails += !right;
    wrong += used > 0 && !right;
    its += spent;
  }
  if (upto > t->frames) { t->fails += fails; t->wrong += wrong; t->its += its; t->frames = upto; }
}

// ai: The table this tool always printed: every rate at length n, today's decoder at 40 iterations, bisected in
// ai: noise for 10% frame error with the given frames a step.
static void rate_sweep(int n, int frames, int threads) {
  printf("n=%d, %d frames per point, 40 iterations max\n", n, frames);
  printf("rate     n     k  edges  init_ms | MI@FER10%%   gap  gap%%  | iters@op  us/frame@op\n");
  for (int rate = 0; rate < LDPC_RATES; rate++) {
    clock_t t0 = clock();
    ctx_t *ctx = ctx_make(threads, n, rate);
    double init_ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC / threads;
    const ldpc_t *c = &ctx[0].code;
    ref_arm_t a = { .name = "sweep", .today = 1, .norm = c->norm, .ref = { .max_iter = 40 } };
    double R = ldpc_rate_value(rate), lo = 0.3, hi = 2.0, op_it = 0, op_us = 0;
    for (int step = 0; step < 9; step++) {
      double sigma = 0.5 * (lo + hi);
      tally_t t = { 0 };
      clock_t t1 = clock();
      tally_to(ctx, &a, (uint64_t)rate, sigma, SWEEP, &t, frames);
      op_it = t.its / frames; op_us = 1e6 * (clock() - t1) / CLOCKS_PER_SEC / frames;
      if (fer(&t) > 0.1) hi = sigma; else lo = sigma;
    }
    double mi = ref_mi_awgn(0.5 * (lo + hi));
    printf("%-4s %5d %5d %6d %7.1f | %9.3f %6.3f %5.1f | %8.1f %11.0f\n", ldpc_rate_name[rate], c->n, c->k, c->edges, init_ms, mi, mi - R, 100 * (mi - R) / R, op_it, op_us);
    fflush(stdout);
    ctx_free(ctx, threads);
  }
}

// ai: The grid in channel information that every arm is measured on.
#define GRID_MI0 0.70
#define GRID_STEP 0.004
enum { GRID = 60, ARM_STREAM = 1000 };

typedef struct { double mi, se, its, fer_lo, fer_hi; int j, frames, wrong; } crossing_t;

// ai: An arm on this channel: the decoder and the quantiser in front of it. A ref arm with quant set reads the int8
// ai: values over 8, so it takes the soft stage's scale; the other quantisers are for today's decoder.
typedef struct { ref_arm_t arm; quant_t q; } awgn_arm_t;

// ai: Walks the grid from j to the two neighbouring points whose FER brackets 10% (a quarter of the frames per point,
// ai: then the full count there, walking on if the bracket moves), and interpolates log FER between them. se treats
// ai: the two points' counts as one binomial sample of the crossing, since they share frames.
static crossing_t crossing(ctx_t *ctx, const awgn_arm_t *aa, const double *sigma, int frames, int j) {
  const ref_arm_t *a = &aa->arm;
  tally_t t[GRID];
  memset(t, 0, sizeof t);
  int n = frames / 4 < 50 ? 50 : frames / 4;
  crossing_t x = { .mi = NAN };
  for (;;) {
    if (j < 0 || j + 1 >= GRID) return x;
    for (int p = j; p <= j + 1; p++) {
      if (t[p].frames >= n) continue;
      tally_to(ctx, a, ARM_STREAM, sigma[p], aa->q, &t[p], n);
      fprintf(stderr, "awgn %-34s mi %.3f frames %5d fer %.4f its %6.2f wrong %d\n", a->name, GRID_MI0 + p * GRID_STEP,
              t[p].frames, fer(&t[p]), t[p].its / t[p].frames, t[p].wrong);
    }
    double lo = fer(&t[j]), hi = fer(&t[j + 1]);
    if (lo > 0.1 && hi <= 0.1) {
      if (n == frames) break;
      n = frames;
      continue;
    }
    j += lo <= 0.1 ? -1 : 1;
  }
  double lo = fer(&t[j]), hi = fer(&t[j + 1]), w;
  if (hi > 0) w = (log(lo) - log(0.1)) / (log(lo) - log(hi));
  else w = (lo - 0.1) / lo;
  double its_lo = t[j].its / t[j].frames, its_hi = t[j + 1].its / t[j + 1].frames;
  x.mi = GRID_MI0 + (j + w) * GRID_STEP;
  x.se = hi > 0 ? sqrt(0.9 / (0.1 * frames)) * GRID_STEP / (log(lo) - log(hi)) : NAN;
  x.its = its_lo + w * (its_hi - its_lo);
  x.fer_lo = lo; x.fer_hi = hi; x.j = j; x.frames = frames;
  for (int p = 0; p < GRID; p++) x.wrong += t[p].wrong;
  return x;
}

static awgn_arm_t arm_today(const char *name, int norm, int stall, int iters, quant_t q) {
  return (awgn_arm_t){ { .name = name, .today = 1, .norm = norm, .stall = stall, .ref = { .max_iter = iters } }, q };
}

static awgn_arm_t arm_ref(const char *name, int rule, double alpha, double beta, int quant, int iters) {
  return (awgn_arm_t){ { .name = name, .quant = quant, .ref = { .rule = rule, .max_iter = iters, .stop = 1, .alpha = alpha, .beta = beta } },
                       SOFT_STAGE };
}

static int j_near(double mi) { return (int)floor((mi - GRID_MI0) / GRID_STEP); }

static crossing_t report(ctx_t *ctx, const awgn_arm_t *a, const double *sigma, int frames, double guess) {
  clock_t t0 = clock();
  crossing_t x = crossing(ctx, a, sigma, frames, j_near(guess));
  double cpu = (double)(clock() - t0) / CLOCKS_PER_SEC;
  if (isnan(x.mi)) printf("%-38s | off the grid\n", a->arm.name);
  else printf("%-38s | %7.4f %6.4f %7.4f | %6.2f | %.3f %.3f %5d | %5d | %6.0f\n", a->arm.name, x.mi, x.se, x.mi - 0.75, x.its,
              x.fer_lo, x.fer_hi, x.frames, x.wrong, cpu);
  fflush(stdout);
  return x;
}

static void report_all(ctx_t *ctx, const awgn_arm_t *a, size_t count, const double *sigma, int frames) {
  for (size_t i = 0; i < count; i++) report(ctx, &a[i], sigma, frames, 0.79);
}

// ai: The format's code, every arm. Offset min-sum's offsets are swept at 30 iterations and the best is run at 100.
// ai: Two arms change only the quantiser in front of today's decoder: the rate sweep's (clipped at +-127 instead of
// ai: +-80, so LLRs up to 15.9) and half the scale (round(4 x LLR), clipped at +-80: LLRs up to 20, and every message
// ai: cap doubled in LLR terms).
static void arms(int n, int frames, int threads) {
  static const double BETAS[] = { 0.25, 0.375, 0.5, 0.625, 0.75, 1.0 };
  static char names[8][48];
  ctx_t *ctx = ctx_make(threads, n, LDPC_BASE_RATE);
  double sigma[GRID];
  for (int j = 0; j < GRID; j++) sigma[j] = ref_sigma_for(GRID_MI0 + j * GRID_STEP);
  printf("\nthe format's code (n %d, k %d, z %d), BI-AWGN, %d frames at each bracketing point, %d threads\n",
         ctx[0].code.n, ctx[0].code.k, ctx[0].code.z, frames, threads);
  printf("int8 = the soft stage's round(8 x LLR), LLR clipped to +-10; float = the channel's LLR; stall = focus.c's rule;\n");
  printf("its = mean iterations run at the crossing, failures at what they spent; wrong = clear syndromes not the truth\n");
  printf("arm                                    |  MI@10%%     se     gap |   its | FER at bracket  n | wrong | cpu s\n");
  const awgn_arm_t today[] = {
    arm_today("today: int8, norm 13/16, 30, stall", 13, 1, 30, SOFT_STAGE),
    arm_today("today, no stall, 30", 13, 0, 30, SOFT_STAGE),
    arm_today("today, no stall, 100", 13, 0, 100, SOFT_STAGE),
    arm_today("today at norm 12/16, 30, stall", 12, 1, 30, SOFT_STAGE),
    arm_today("today at norm 14/16, 30, stall", 14, 1, 30, SOFT_STAGE),
    arm_today("today, int8 clipped at 127, 30, stall", 13, 1, 30, SWEEP),
    arm_today("today, int8 at 4 x LLR, 30, stall", 13, 1, 30, HALF),
    arm_ref("float min-sum 13/16, int8, 30", REF_MINSUM, 13.0 / 16, 0, 1, 30),
  };
  report_all(ctx, today, sizeof today / sizeof *today, sigma, frames);
  double best = 0, best_mi = 1;
  for (size_t i = 0; i < sizeof BETAS / sizeof *BETAS; i++) {
    snprintf(names[i], sizeof names[i], "offset min-sum %.3f, int8, 30", BETAS[i]);
    awgn_arm_t a = arm_ref(names[i], REF_MINSUM, 1, BETAS[i], 1, 30);
    crossing_t x = report(ctx, &a, sigma, frames, 0.79);
    if (x.mi < best_mi) { best_mi = x.mi; best = BETAS[i]; }
  }
  snprintf(names[7], sizeof names[7], "offset min-sum %.3f, int8, 100", best);
  const awgn_arm_t rest[] = {
    arm_ref(names[7], REF_MINSUM, 1, best, 1, 100),
    arm_ref("BP, float, 30", REF_BP, 0, 0, 0, 30),
    arm_ref("BP, float, 100", REF_BP, 0, 0, 0, 100),
    arm_ref("BP, int8, 30", REF_BP, 0, 0, 1, 30),
    arm_ref("BP, int8, 100", REF_BP, 0, 0, 1, 100),
  };
  report_all(ctx, rest, sizeof rest / sizeof *rest, sigma, frames);
  ctx_free(ctx, threads);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 4800, frames = argc > 2 ? atoi(argv[2]) : 200, arm_frames = argc > 3 ? atoi(argv[3]) : 2000;
  int threads = omp_get_max_threads();
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (frames > 0) rate_sweep(n, frames, threads);
  if (arm_frames > 0 && n / LDPC_NB == LDPC_BASE_Z) arms(n, arm_frames, threads);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("wall %.1f s\n", (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec));
  return 0;
}
