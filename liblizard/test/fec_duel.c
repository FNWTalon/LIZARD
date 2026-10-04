// Reed-Solomon against LDPC as the code inside one tile (about 10 000 bits), at the same rates.
//   LDPC soft      the decoder as the pipeline uses it: int8 LLRs, block-row min-sum
//   LDPC hard      the same decoder given signs only, to show what soft values are worth to it
//   RS hard        5 interleaved RS(249, k) codewords over GF(256), bounded-distance, errors only
//   RS erasures    the same, retried with its least reliable bytes erased (0, 1/4, 1/2, 3/4 of the parity),
//                  which is the cheap way for RS to use reliability
// Channels: binary-input AWGN, and the same with 3% of the block wiped out in one run (a glare blob,
// a finger, a torn band). LDPC bits are scattered over the block as the layout scatters them; RS bytes
// are contiguous and the codewords interleaved, which is the right arrangement for each.
// For each: the channel (as mutual information per bit) at which 10% of blocks fail, which is the
// operating point under a fountain, and the decode time there, on a clean channel and on a hopeless one.
//   fec_duel [frames]
#include "../src/ldpc.h"
#include "../src/rs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t SEED = 88172645463325252ull;
static double uni(void) { SEED ^= SEED << 13; SEED ^= SEED >> 7; SEED ^= SEED << 17; return (double)(SEED >> 11) / 9007199254740992.0; }
static double gauss(void) { double u = 1 - uni(), v = uni(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
static double mi_awgn(double sigma) { double acc = 0; int N = 100000; for (int i = 0; i < N; i++) { double L = 2 * (1 + sigma * gauss()) / (sigma * sigma); acc += log2(1 + exp(-L)); } return 1 - acc / N; }
static double now_us(void) { return 1e6 * clock() / CLOCKS_PER_SEC; }

enum { NB = 9984, M = 5, NCW = 249, BURST = 300 };   // block bits; RS codewords and their length in bytes; burst length in bits (3%)
enum { LDPC_SOFT, LDPC_HARD, RS_HARD, RS_ERAS, RS_ERAS_SCALAR, SCHEMES };
static const char *NAME[SCHEMES] = { "LDPC soft", "LDPC hard", "RS hard", "RS erasures", "RS erasures, scalar syndromes" };

static ldpc_t code;
static int nroots, perm[NB];
static uint8_t bits[NB], cwb[NB], data[NB], rsbuf[M * NCW], rsgood[M * NCW], out[NB];
static int8_t llr[NB];
static double iters;

// One block through the channel and the decoder. Returns 1 on failure, 2 on an undetected wrong decode.
static int trial(int scheme, double sigma, int burst, double *us) {
  int n = scheme < RS_HARD ? code.n : M * NCW * 8, b0 = burst ? (int)(uni() * (n - BURST)) : -1;
  if (scheme < RS_HARD) { for (int i = 0; i < code.k; i++) data[i] = uni() < 0.5; ldpc_encode(&code, data, cwb); for (int i = 0; i < n; i++) bits[perm[i]] = cwb[i]; }
  else {
    uint8_t cw[NCW];
    for (int c = 0; c < M; c++) { for (int i = 0; i < NCW - nroots; i++) cw[i] = (uint8_t)(uni() * 256); rs_encode(cw, NCW - nroots, nroots); for (int j = 0; j < NCW; j++) rsbuf[j * M + c] = cw[j]; }
    memcpy(rsgood, rsbuf, sizeof rsbuf);
    for (int i = 0; i < n; i++) bits[i] = (rsbuf[i >> 3] >> (7 - (i & 7))) & 1;
  }
  // Channel order is position in the block. In the burst there is no signal, and a soft receiver knows it: its LLRs there are next to nothing.
  static int8_t ch[NB];
  for (int i = 0; i < n; i++) {
    if (i >= b0 && i < b0 + BURST && b0 >= 0) { ch[i] = (int8_t)((uni() < 0.5 ? -1 : 1) * (1 + (int)(uni() * 3))); continue; }
    double y = (bits[i] ? -1 : 1) + sigma * gauss(), L = 2 * y / (sigma * sigma) * 8;
    ch[i] = (int8_t)(L > 127 ? 127 : L < -127 ? -127 : L == 0 ? 1 : lrint(L));
  }
  double t0 = now_us();
  int fail;
  if (scheme < RS_HARD) {
    for (int i = 0; i < n; i++) { int8_t v = ch[perm[i]]; llr[i] = scheme == LDPC_HARD ? (v < 0 ? -32 : 32) : v; }
    int it = ldpc_decode(&code, llr, out, 50);
    iters += it < 0 ? 50 : it;
    fail = it < 0 ? 1 : memcmp(out, cwb, (size_t)code.k) ? 2 : 0;
  } else {
    int rel[M * NCW];
    for (int j = 0; j < M * NCW; j++) { uint8_t v = 0; int r = 127; for (int k = 0; k < 8; k++) { int8_t l = ch[8 * j + k]; v = (uint8_t)(v << 1 | (l < 0)); int a = l < 0 ? -l : l; if (a < r) r = a; } rsbuf[j] = v; rel[j] = r; }
    static uint8_t syn[M * 256];
    if (scheme != RS_ERAS_SCALAR) rs_syndromes_il(rsbuf, M, NCW, nroots, syn);
    fail = 0;
    for (int c = 0; c < M; c++) {
      uint8_t cw[NCW];
      int order[NCW], ok = -1;
      if (scheme != RS_HARD) {   // least reliable first: a counting sort on the reliability, 0 .. 127
        int cnt[129] = { 0 };
        for (int j = 0; j < NCW; j++) cnt[rel[j * M + c] + 1]++;
        for (int v = 0; v < 128; v++) cnt[v + 1] += cnt[v];
        for (int j = 0; j < NCW; j++) order[cnt[rel[j * M + c]]++] = j;
      }
      // Erasing more leaves less parity to catch a wrong answer: a decode correcting e errors is wrong about once in e!,
      // so stop while six errors can still be corrected.
      for (int step = 0; step <= (scheme == RS_HARD ? 0 : 3) && ok < 0; step++) {
        int s = step * nroots / 4;
        if (step && nroots - s < 12) break;
        if (scheme == RS_ERAS_SCALAR) { for (int j = 0; j < NCW; j++) cw[j] = rsbuf[j * M + c]; ok = rs_decode_er(cw, NCW, nroots, order, s); if (ok >= 0) for (int j = 0; j < NCW; j++) rsbuf[j * M + c] = cw[j]; }
        else ok = rs_decode_il(rsbuf, M, NCW, nroots, c, syn, order, s);
      }
      if (ok < 0) fail = 1;
    }
    if (!fail && memcmp(rsbuf, rsgood, sizeof rsbuf)) fail = 2;
  }
  *us += now_us() - t0;
  return fail;
}

static double run(int scheme, double sigma, int burst, int frames, double *us, double *it, int *undet) {
  int f = 0; *us = 0; iters = 0;
  for (int k = 0; k < frames; k++) { int r = trial(scheme, sigma, burst, us); f += r != 0; *undet += r == 2; }
  *us /= frames; *it = iters / frames;
  return (double)f / frames;
}

int main(int argc, char **argv) {
  int frames = argc > 1 ? atoi(argv[1]) : 120;
#ifdef __wasm_simd128__
  const char *build = "wasm with SIMD";
#elif defined(__wasm__)
  const char *build = "wasm, no SIMD";
#else
  const char *build = "native";
#endif
  printf("%s. Block %d bits (RS: %d x RS(%d, k), %d bits). %d blocks a point, failure target 10%%.\n", build, NB, M, NCW, M * NCW * 8, frames);
  for (int burst = 0; burst < 2; burst++) {
    printf("\n%s\n", burst ? "AWGN with 3% of the block wiped out in one run" : "AWGN");
    printf("%-30s %5s %7s | %8s %7s %6s | %9s %9s %9s | %6s %s\n", "code", "rate", "payload", "MI @10%", "wasted", "effic.", "us clean", "us at op", "us fail", "iters", "undetected");
    for (int rate = 2; rate < LDPC_RATES; rate++) {
      if (ldpc_init(&code, NB, rate, 1)) return 1;
      int g = (int)(code.n * 0.6180339887) | 1, a, b;
      for (;;) { for (a = g, b = code.n; b; ) { int t = a % b; a = b; b = t; } if (a == 1) break; g += 2; }
      for (int i = 0; i < code.n; i++) perm[i] = (int)((int64_t)i * g % code.n);
      nroots = (int)lrint((1 - ldpc_rate_value(rate)) * NCW); nroots += nroots & 1;
      for (int scheme = 0; scheme < SCHEMES; scheme++) {
        double lo = 0.2, hi = 2.0, us, it; int undet = 0;
        if (scheme == RS_ERAS_SCALAR) { lo = hi = 0; }   // same code as RS erasures: only its time differs, taken at that operating point
        static double op_sigma;
        if (scheme != RS_ERAS_SCALAR) { for (int step = 0; step < 8; step++) { double s = 0.5 * (lo + hi); if (run(scheme, s, burst, frames, &us, &it, &undet) > 0.1) hi = s; else lo = s; } op_sigma = 0.5 * (lo + hi); }
        double us_op, us_clean, us_fail, it_op, d;
        run(scheme, op_sigma, burst, frames, &us_op, &it_op, &undet);
        run(scheme, 0.6 * op_sigma, burst, frames / 2, &us_clean, &d, &undet);
        run(scheme, 1.5 * op_sigma, burst, frames / 2, &us_fail, &d, &undet);
        double R = scheme < RS_HARD ? (double)code.k / code.n : (double)(NCW - nroots) / NCW, mi = mi_awgn(op_sigma) * (burst ? 1 - (double)BURST / NB : 1);
        int payload = scheme < RS_HARD ? code.k / 8 : M * (NCW - nroots);
        printf("%-30s %5.3f %6dB | %8.3f %7.3f %5.0f%% | %9.0f %9.0f %9.0f | %6.1f %d\n", NAME[scheme], R, payload, mi, mi - R, 100 * R / mi, us_clean, us_op, us_fail, scheme < RS_HARD ? it_op : 0.0, undet);
      }
      ldpc_free(&code);
      printf("\n");
    }
  }
  return 0;
}
