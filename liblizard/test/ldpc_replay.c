// ai: How many of a recording's blocks would a better LDPC decoder have read? Replays dumps of the soft values the
// ai: GPU back half handed its LDPC (one record a block, below) through decoder arms (ldpc_ref.h) and counts, for each
// ai: arm, the blocks decoded to the true codeword split by the GPU's verdict, the clear syndromes that were not the
// ai: truth ("wrong"; those from a record whose int8 values are all zero, which the soft gate zeroes and any decoder
// ai: clears at the all-zero word, are counted apart as "zero in"), and the mean iterations over every record.
// ai: Arms: today's decoder as focus.c runs it (int8, 30 iterations, the stall rule); the same with no stop rule at
// ai: 100; at norm 12/16; offset min-sum at the given offset, 30 and 100; BP on the int8 values over 8, 30 and 100; BP
// ai: at 100 on the genie LLRs; today's decoder on the genie LLRs quantised as the soft stage would. Every arm runs on
// ai: every record, including those the soft gate declined.
// ai: A record, little-endian (this reads it on a little-endian host), 15,920 bytes:
// ai:   i32 frame, i32 block, u8 verdict (1 verified, 2 declined by the soft gate, 3 given up by the stop rule,
// ai:   4 CRC failed), u8 pad x3, f32 blind estimate, f32 genie information a coded bit,
// ai:   i8 llr[5088] (codeword order, ldpc_decode's convention), i16 genie_llr[5088] (256 x LLR), u8 truth[636]
// ai:   (the codeword, bit i at byte i / 8, bit i % 8).
// ai:   gcc -O2 -fopenmp -Wall -Wextra -o build/ldpc_replay test/ldpc_replay.c test/ldpc_ref.c src/ldpc.c -lm
// ai:   build/ldpc_replay <offset> <run.llr>... 2>>build/ldpc_progress.log
// ai:   build/ldpc_replay synth <out.llr> <blocks> <mi>   (a dump from BI-AWGN, to test the reader; see synth below)
// ai: Beside each dump it writes <run>.replay.tsv: a row a record (frame, block, verdict, estimate, genie information,
// ai: then each arm's return, iterations to a clear syndrome or -1 or -3, and 1 if that was the truth).
// ai: Runtime (2026-09-26, OMP_NUM_THREADS=16): the three dumps, 9,616 records, 19.5 s wall; research/results/ldpc/replay.txt.
#include "ldpc_ref.h"
#include "../src/ldpc_base.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void) { return 0; }
static int omp_get_max_threads(void) { return 1; }
#endif

enum { N = 5088, HEAD = 20, OFF_LLR = HEAD, OFF_GENIE = OFF_LLR + N, OFF_TRUTH = OFF_GENIE + 2 * N, REC = OFF_TRUTH + N / 8 };
enum { VERDICTS = 5 };   // ai: columns: 0 any verdict outside 1 to 4, then 1 to 4
static const char *const VERDICT_NAME[VERDICTS] = { "other", "verified", "declined", "given up", "CRC failed" };

// ai: An arm and where its input comes from: the dumped int8 values, or the genie LLRs.
typedef struct { ref_arm_t arm; int genie; } replay_arm_t;
enum { ARMS = 9 };

typedef struct { long ok[VERDICTS], wrong, wrong_zero; double its; } tally_t;

// ai: One thread's code, decoders, buffers and counts.
typedef struct {
  ldpc_t code;
  ref_dec_t ref;
  uint8_t truth[N], out[N];
  int8_t q8[N];
  float fl[N], buf[N];
  tally_t tally[ARMS];
} ctx_t;

static ctx_t *ctx_make(int count) {
  ctx_t *ctx = calloc((size_t)count, sizeof *ctx);
  for (int t = 0; t < count; t++) {
    if (ldpc_init(&ctx[t].code, N, LDPC_BASE_RATE, LDPC_BASE_SEED) || ref_init(&ctx[t].ref, &ctx[t].code) ||
        ctx[t].code.n != N || ctx[t].code.z != LDPC_BASE_Z) { fprintf(stderr, "not the format's code\n"); exit(1); }
  }
  return ctx;
}

static void ctx_free(ctx_t *ctx, int count) {
  for (int t = 0; t < count; t++) { ref_free(&ctx[t].ref); ldpc_free(&ctx[t].code); }
  free(ctx);
}

static int verdict_col(int v) { return v >= 1 && v <= 4 ? v : 0; }

// ai: One arm's outcome on one record: the decoder's return and whether it was the truth.
typedef struct { int16_t used; uint8_t right; } outcome_t;

// ai: Runs every arm on the record at p, counts it in the thread's tallies and puts each arm's outcome in res.
static void replay_record(ctx_t *x, const replay_arm_t *arms, const uint8_t *p, outcome_t *res) {
  int col = verdict_col(p[8]), zero = 1;
  const int8_t *llr = (const int8_t *)(p + OFF_LLR);
  for (int i = 0; i < N && zero; i++) zero = !llr[i];
  for (int i = 0; i < N; i++) x->truth[i] = (uint8_t)(p[OFF_TRUTH + i / 8] >> (i % 8) & 1);
  int have_genie = 0;
  for (int a = 0; a < ARMS; a++) {
    const replay_arm_t *ra = &arms[a];
    const int8_t *q8 = llr;
    if (ra->genie) {
      if (!have_genie) {
        for (int i = 0; i < N; i++) {
          int16_t g;
          memcpy(&g, p + OFF_GENIE + 2 * i, 2);
          x->fl[i] = g / 256.0f;
          x->q8[i] = ref_quant8(x->fl[i]);
        }
        have_genie = 1;
      }
      q8 = x->q8;
    }
    int spent, used = ref_arm_decode(&ra->arm, &x->code, &x->ref, q8, x->fl, x->buf, x->out, &spent);
    int right = used > 0 && !memcmp(x->out, x->truth, N);
    x->tally[a].ok[col] += right;
    if (used > 0 && !right) {
      if (zero && !ra->genie) x->tally[a].wrong_zero++;
      else x->tally[a].wrong++;
    }
    x->tally[a].its += spent;
    res[a] = (outcome_t){ (int16_t)used, (uint8_t)right };
  }
}

static void write_tsv(const char *path, const uint8_t *m, long count, const replay_arm_t *arms, const outcome_t *res) {
  char out[1024];
  size_t len = strlen(path);
  snprintf(out, sizeof out, "%.*s.replay.tsv", (int)(len > 4 && !strcmp(path + len - 4, ".llr") ? len - 4 : len), path);
  FILE *f = fopen(out, "w");
  if (!f) { perror(out); exit(1); }
  fprintf(f, "frame\tblock\tverdict\test\tinfo");
  for (int a = 0; a < ARMS; a++) fprintf(f, "\t%s\t%s ok", arms[a].arm.name, arms[a].arm.name);
  fprintf(f, "\n");
  for (long r = 0; r < count; r++) {
    const uint8_t *p = m + r * REC;
    int32_t frame, block;
    float est, info;
    memcpy(&frame, p, 4); memcpy(&block, p + 4, 4); memcpy(&est, p + 12, 4); memcpy(&info, p + 16, 4);
    fprintf(f, "%d\t%d\t%d\t%.4f\t%.4f", frame, block, p[8], est, info);
    for (int a = 0; a < ARMS; a++) fprintf(f, "\t%d\t%d", res[r * ARMS + a].used, res[r * ARMS + a].right);
    fprintf(f, "\n");
  }
  fclose(f);
}

static const uint8_t *map_file(const char *path, size_t *size) {
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
  *size = (size_t)st.st_size;
  if (*size % REC) { fprintf(stderr, "%s: %zu bytes, not a whole number of %d-byte records\n", path, *size, REC); exit(1); }
  void *m = *size ? mmap(0, *size, PROT_READ, MAP_PRIVATE, fd, 0) : 0;
  close(fd);
  if (m == MAP_FAILED) { perror(path); exit(1); }
  return m;
}

static void replay_file(const char *path, ctx_t *ctx, int threads, const replay_arm_t *arms) {
  size_t size;
  const uint8_t *m = map_file(path, &size);
  long count = (long)(size / REC), by[VERDICTS] = { 0 }, frames = 0, last = -1, zero = 0;
  for (long r = 0; r < count; r++) {
    int32_t frame;
    memcpy(&frame, m + r * REC, 4);
    by[verdict_col(m[r * REC + 8])]++;
    int z = 1;
    for (int i = 0; i < N && z; i++) z = !m[r * REC + OFF_LLR + i];
    zero += z;
    if (frame != last) { frames++; last = frame; }
  }
  for (int t = 0; t < threads; t++) memset(ctx[t].tally, 0, sizeof ctx[t].tally);
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  long done = 0;
  outcome_t *res = malloc((size_t)(count ? count : 1) * ARMS * sizeof *res);
#pragma omp parallel for schedule(dynamic, 4)
  for (long r = 0; r < count; r++) {
    replay_record(&ctx[omp_get_thread_num()], arms, m + r * REC, res + r * ARMS);
    long d;
#pragma omp atomic capture
    d = ++done;
    if (d % 2000 == 0) fprintf(stderr, "replay %s %ld of %ld\n", path, d, count);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double wall = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
  printf("\n%s: %ld records, %ld frame changes; by GPU verdict:", path, count, frames);
  for (int v = 1; v <= VERDICTS; v++) printf(" %s %ld,", VERDICT_NAME[v % VERDICTS], by[v % VERDICTS]);
  printf(" %ld with all-zero int8 values; %.1f s on %d threads\n", zero, wall, threads);
  printf("arm                                 | to truth: verified declined given-up CRC-fail other |  total | wrong | zero in |   its\n");
  for (int a = 0; a < ARMS; a++) {
    tally_t s = { 0 };
    for (int t = 0; t < threads; t++) {
      for (int v = 0; v < VERDICTS; v++) s.ok[v] += ctx[t].tally[a].ok[v];
      s.wrong += ctx[t].tally[a].wrong;
      s.wrong_zero += ctx[t].tally[a].wrong_zero;
      s.its += ctx[t].tally[a].its;
    }
    long total = 0;
    for (int v = 0; v < VERDICTS; v++) total += s.ok[v];
    printf("%-36s |           %8ld %8ld %8ld %8ld %5ld | %6ld | %5ld | %7ld | %5.2f\n", arms[a].arm.name, s.ok[1], s.ok[2],
           s.ok[3], s.ok[4], s.ok[0], total, s.wrong, s.wrong_zero, count ? s.its / count : 0);
  }
  fflush(stdout);
  write_tsv(path, m, count, arms, res);
  free(res);
  munmap((void *)m, size);
}

static replay_arm_t arm_today(const char *name, int norm, int stall, int iters, int genie) {
  return (replay_arm_t){ { .name = name, .today = 1, .norm = norm, .stall = stall, .ref = { .max_iter = iters } }, genie };
}

static replay_arm_t arm_ref(const char *name, int rule, double beta, int quant, int iters, int genie) {
  return (replay_arm_t){ { .name = name, .quant = quant, .ref = { .rule = rule, .max_iter = iters, .stop = 1, .alpha = 1, .beta = beta } }, genie };
}

// ai: A dump from BI-AWGN in the record format, to test the reader. Block i's channel is drawn from 21 levels within
// ai: 0.04 bits a bit of mi; the genie LLRs are the channel's, the dumped int8 values the soft stage's quantisation
// ai: of 0.8 times them (a blind estimate off by a fifth). The verdict: declined (2) below mi - 0.03, else today's
// ai: decoder on the dumped values: verified (1) at the truth, given up (3) by the stall rule, CRC failed (4) at the
// ai: cap or a wrong codeword. So the today arm must read every verified record and nothing else of 1, 3 and 4.
static void synth(const char *path, long blocks, double mi) {
  enum { LEVELS = 21 };
  double sig[LEVELS];
  for (int l = 0; l < LEVELS; l++) sig[l] = ref_sigma_for(mi - 0.04 + 0.004 * l);
  ctx_t *ctx = ctx_make(1);
  ldpc_t *c = &ctx->code;
  replay_arm_t today = arm_today("today", 13, 1, 30, 0);
  uint8_t data[N], cw[N], rec[REC];
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); exit(1); }
  for (long b = 0; b < blocks; b++) {
    ref_rng_t r = ref_rng_at(77, (uint64_t)b);
    int level = (int)(ref_uni(&r) * LEVELS);
    double s = sig[level], bmi = mi - 0.04 + 0.004 * level;
    for (int i = 0; i < c->k; i++) data[i] = ref_uni(&r) < 0.5;
    ldpc_encode(c, data, cw);
    memset(rec, 0, sizeof rec);
    int8_t *q8 = (int8_t *)(rec + OFF_LLR);
    for (int i = 0; i < N; i++) {
      float L = (float)(2 * ((cw[i] ? -1 : 1) + s * ref_gauss(&r)) / (s * s));
      double g = nearbyint(256.0 * L);
      int16_t gi = (int16_t)(g > 32767 ? 32767 : g < -32767 ? -32767 : g);
      memcpy(rec + OFF_GENIE + 2 * i, &gi, 2);
      q8[i] = ref_quant8(0.8f * L);
      rec[OFF_TRUTH + i / 8] |= (uint8_t)(cw[i] << (i % 8));
    }
    int verdict = 2;
    if (bmi >= mi - 0.03) {
      int spent, used = ref_arm_decode(&today.arm, c, &ctx->ref, q8, 0, ctx->buf, ctx->out, &spent);
      verdict = used > 0 && !memcmp(ctx->out, cw, N) ? 1 : used == -3 ? 3 : 4;
    }
    int32_t frame = (int32_t)(b / 32), block = (int32_t)(b % 32);
    float est = (float)bmi, info = (float)bmi;
    memcpy(rec, &frame, 4); memcpy(rec + 4, &block, 4); rec[8] = (uint8_t)verdict;
    memcpy(rec + 12, &est, 4); memcpy(rec + 16, &info, 4);
    fwrite(rec, REC, 1, f);
  }
  fclose(f);
  ctx_free(ctx, 1);
}

int main(int argc, char **argv) {
  if (argc >= 5 && !strcmp(argv[1], "synth")) { synth(argv[2], atol(argv[3]), atof(argv[4])); return 0; }
  if (argc < 3) { fprintf(stderr, "ldpc_replay <offset> <run.llr>... | ldpc_replay synth <out.llr> <blocks> <mi>\n"); return 1; }
  double beta = atof(argv[1]);
  static char oms30[48], oms100[48];
  snprintf(oms30, sizeof oms30, "offset min-sum %.3f, int8, 30", beta);
  snprintf(oms100, sizeof oms100, "offset min-sum %.3f, int8, 100", beta);
  const replay_arm_t arms[ARMS] = {
    arm_today("today: int8, 30, stall", 13, 1, 30, 0),
    arm_today("today, no stop, 100", 13, 0, 100, 0),
    arm_today("today at norm 12/16, 30, stall", 12, 1, 30, 0),
    arm_ref(oms30, REF_MINSUM, beta, 1, 30, 0),
    arm_ref(oms100, REF_MINSUM, beta, 1, 100, 0),
    arm_ref("BP, int8, 30", REF_BP, 0, 1, 30, 0),
    arm_ref("BP, int8, 100", REF_BP, 0, 1, 100, 0),
    arm_ref("BP, genie, 100", REF_BP, 0, 0, 100, 1),
    arm_today("today on genie int8, 30, stall", 13, 1, 30, 1),
  };
  int threads = omp_get_max_threads();
  ctx_t *ctx = ctx_make(threads);
  printf("record %d bytes; offset %.3f; its = mean iterations run over every record\n", REC, beta);
  for (int i = 2; i < argc; i++) replay_file(argv[i], ctx, threads, arms);
  ctx_free(ctx, threads);
  return 0;
}
