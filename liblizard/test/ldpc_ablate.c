// Which LDPC suits this channel: degree profile and min-sum normalization, by the channel
// information needed for 10% frame loss at n = 4800.  ldpc_ablate [n] [frames]
#include "../src/ldpc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int ldpc_override_heavy, ldpc_override_dh, ldpc_norm;
static uint64_t S = 88172645463325252ull;
static double uni(void) { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (double)(S >> 11) / 9007199254740992.0; }
static double gauss(void) { double u = 1 - uni(), v = uni(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
static double mi_awgn(double sigma) { double a = 0; int N = 200000; for (int i = 0; i < N; i++) { double L = 2 * (1 + sigma * gauss()) / (sigma * sigma); a += log2(1 + exp(-L)); } return 1 - a / N; }
static double op_point(int n, int rate, int frames, double *its) {
  ldpc_t c; if (ldpc_init(&c, n, rate, 1)) return -1;
  uint8_t *d = malloc(c.k), *cw = malloc(c.n), *out = malloc(c.n); int8_t *llr = malloc(c.n);
  double lo = 0.3, hi = 2.0;
  for (int step = 0; step < 8; step++) {
    double sigma = 0.5 * (lo + hi), it_acc = 0; int fail = 0;
    for (int f = 0; f < frames; f++) {
      for (int i = 0; i < c.k; i++) d[i] = uni() < 0.5;
      ldpc_encode(&c, d, cw);
      for (int i = 0; i < c.n; i++) { double y = (cw[i] ? -1 : 1) + sigma * gauss(), L = 2 * y / (sigma * sigma) * 8; llr[i] = (int8_t)(L > 127 ? 127 : L < -127 ? -127 : lrint(L)); }
      int it = ldpc_decode(&c, llr, out, 40);
      if (it < 0 || memcmp(out, cw, c.k)) fail++;
      it_acc += it < 0 ? 40 : it;
    }
    *its = it_acc / frames;
    if ((double)fail / frames > 0.1) hi = sigma; else lo = sigma;
  }
  free(d); free(cw); free(out); free(llr); ldpc_free(&c);
  return mi_awgn(0.5 * (lo + hi));
}
int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 4800, frames = argc > 2 ? atoi(argv[2]) : 150;
  static const int prof[][2] = { { 0, 0 }, { 4, 6 }, { 8, 6 }, { 4, 10 }, { 10, 8 }, { 16, 8 }, { 8, 12 } };
  printf("channel MI needed for FER 10%% (iterations), n=%d. profile = heavy columns x degree, rest degree 3\n", n);
  printf("%-14s", "profile");
  for (int rate = 1; rate <= 5; rate++) printf("   rate %-8s", ldpc_rate_name[rate]);
  printf("\n");
  for (unsigned p = 0; p < sizeof prof / sizeof prof[0]; p++) {
    ldpc_override_heavy = prof[p][0]; ldpc_override_dh = prof[p][1];
    char name[32]; snprintf(name, sizeof name, p ? "%dx%d" : "all degree 3", prof[p][0], prof[p][1]);
    printf("%-14s", name);
    for (int rate = 1; rate <= 5; rate++) { double its, mi = op_point(n, rate, frames, &its); printf("   %.3f (%4.1f)", mi, its); }
    printf("\n");
  }
  ldpc_override_heavy = -1;
  printf("min-sum normalization /16, table profiles\n");
  for (int a = 10; a <= 16; a += 1) {
    ldpc_norm = a;
    printf("%-14d", a);
    for (int rate = 1; rate <= 5; rate++) { double its, mi = op_point(n, rate, frames, &its); printf("   %.3f (%4.1f)", mi, its); }
    printf("\n");
  }
  return 0;
}
