// ai: Does arm64's LDPC (src/ldpc.c: the NEON layer kernel, the decoder's four-times columns) decide as the scalar twin
// ai: and the 128-bit kernel do? 1. the NEON kernel on plain rows and the 128-bit one against the scalar loop on random
// ai: layers (3 to 32 slots, posteriors over the whole range and its ends, messages over +-127), both outputs byte for
// ai: byte; 2. whole decodes of the four codes through BI-AWGN at three levels, arm64's decode against the 128-bit
// ai: kernel on today's layout (wide -1): iterations and every output bit; 3. each code's decodes at its middle level
// ai: timed both ways, interleaved, the best of REPS.
// ai: arm64 only (test/neon_kernels.sh builds it with the NDK and runs it on a phone): ldpc_neon [layers] [codewords] [reps]
#include "../src/ldpc.c"
#include <math.h>
#include <stdio.h>
#include <time.h>

static uint64_t rng = 88172645463325252ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double unif(void) { return (next() >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) { double u = unif() + 1e-300, v = unif(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }

// ai: the scalar twin of src/ldpc.c layer_kernel, as written there under #else
static void kernel_scalar(int16_t *Q, int8_t *R, int ns, int zp, int norm) {
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
}

static int16_t pick16(void) {
  switch (next() % 8) {
    case 0: return 8191;
    case 1: return -8191;
    case 2: return 0;
    case 3: return (int16_t)((int)(next() % 7) - 3);
    default: return (int16_t)((int)(next() % 16383) - 8191);
  }
}


static void make_word(ldpc_t *c, double db, uint8_t *data, uint8_t *cw, int8_t *llr) {
  for (int v = 0; v < c->k; v++) data[v] = v < 3840 ? (uint8_t)(next() & 1) : 0;
  ldpc_encode(c, data, cw);
  const double n0 = pow(10, -db / 10), sc = 0.7 * 2 * sqrt(2) / n0;
  for (int v = 0; v < c->n; v++) {
    double y = (1 - 2 * cw[v]) / sqrt(2) + gauss() * sqrt(n0 / 2), l = sc * y;
    if (l > 10) l = 10;
    if (l < -10) l = -10;
    llr[v] = (int8_t)lrint(8 * l);
  }
  for (int v = 3840; v < c->k; v++) llr[v] = 127;
}

int main(int argc, char **argv) {
  const int layers = argc > 1 ? atoi(argv[1]) : 20000, words = argc > 2 ? atoi(argv[2]) : 500, reps = argc > 3 ? atoi(argv[3]) : 5, zp = 128;
  enum { NS_MAX = 32 };
  static int16_t q0[NS_MAX * 128], qa[NS_MAX * 128], qb[NS_MAX * 128];
  static int8_t r0[NS_MAX * 128], ra[NS_MAX * 128], rb[NS_MAX * 128];
  int16_t *rows[NS_MAX];   // ai: the NEON kernel's view: every slot Q's own row
  for (int e = 0; e < NS_MAX; e++) rows[e] = qb + e * zp;
  int fails = 0;
  for (int t = 0; t < layers; t++) {
    const int ns = 3 + (int)(next() % (NS_MAX - 2)), norm = t % 5 == 0 ? 16 : 13;
    for (int k = 0; k < ns * zp; k++) { q0[k] = pick16(); r0[k] = (int8_t)((int)(next() % 255) - 127); }
    memcpy(qa, q0, sizeof q0); memcpy(ra, r0, sizeof r0);
    kernel_scalar(qa, ra, ns, zp, norm);
    for (int w = 0; w < 2; w++) {
      memcpy(qb, q0, sizeof q0); memcpy(rb, r0, sizeof r0);
      if (w) layer_kernel_neon(qb, rb, ns, zp, norm, rows, 0, 0); else layer_kernel(qb, rb, ns, zp, norm);
      if (memcmp(qa, qb, (size_t)ns * zp * 2) || memcmp(ra, rb, (size_t)ns * zp)) {
        if (fails < 5) printf("layer %d (ns %d, norm %d): %s apart from the scalar loop\n", t, ns, norm, w ? "NEON" : "128-bit");
        fails++;
      }
    }
  }
  printf("layers: %d a kernel (128-bit, NEON) against the scalar loop, %d apart\n", layers, fails);

  static const int RATES[4] = { 6, 4, 3, 2 };
  static const double DB[4][3] = { { 5.4, 5.9, 7.0 }, { 3.8, 4.3, 5.5 }, { 2.7, 3.2, 4.5 }, { 0.8, 1.3, 2.5 } };
  int dfails = 0;
  long decodes = 0;
  for (int ci = 0; ci < 4; ci++) {
    ldpc_t c;
    if (ldpc_init(&c, RATES[ci])) { printf("ldpc_init %d failed\n", RATES[ci]); return 1; }
    uint8_t *data = calloc((size_t)c.k, 1), *cw = calloc((size_t)c.n, 1), *out = calloc((size_t)c.n, 1), *ref = calloc((size_t)c.n, 1);
    int8_t *llr = calloc((size_t)c.n, 1);
    for (int wi = 0; wi < words; wi++) {
      make_word(&c, DB[ci][wi % 3], data, cw, llr);
      c.wide = -1; const int its0 = ldpc_decode_stall(&c, llr, ref, 30, 9, 0.95f);
      c.wide = 0; const int its1 = ldpc_decode_stall(&c, llr, out, 30, 9, 0.95f);
      decodes += 2;
      if (its0 != its1 || memcmp(out, ref, (size_t)c.n)) {
        if (dfails < 5) printf("rate %d word %d: NEON apart (iterations %d against %d)\n", RATES[ci], wi, its1, its0);
        dfails++;
      }
    }
    // time: the middle level's words, made once, decoded by each kernel in turn, the best sweep of each
    enum { TW = 64 };
    int8_t *L = malloc((size_t)TW * c.n);
    for (int wi = 0; wi < TW; wi++) make_word(&c, DB[ci][1], data, cw, L + (size_t)wi * c.n);
    double best[2] = { 1e30, 1e30 };
    long its[2] = { 0, 0 };
    for (int rep = 0; rep < reps; rep++) for (int w = 0; w < 2; w++) {
      c.wide = w ? 0 : -1;
      long it = 0;
      const double t0 = now_ms();
      for (int wi = 0; wi < TW; wi++) { const int u = ldpc_decode_stall(&c, L + (size_t)wi * c.n, out, 30, 9, 0.95f); it += u > 0 ? u : u == -3 ? 9 : 30; }
      const double t = now_ms() - t0;
      if (t < best[w]) best[w] = t;
      its[w] = it;
    }
    printf("rate %d (%s) at %.1f dB: a decode %.1f us (128-bit) against %.1f us (NEON), %.2fx; %.1f iterations a decode; a layer %.2f against %.2f us\n",
           RATES[ci], ldpc_rate_name[RATES[ci]], DB[ci][1], 1e3 * best[0] / TW, 1e3 * best[1] / TW, best[0] / best[1], (double)its[1] / TW,
           1e3 * best[0] / its[0] / c.mb, 1e3 * best[1] / its[1] / c.mb);
    free(L); free(data); free(cw); free(out); free(ref); free(llr);
    ldpc_free(&c);
  }
  printf("decodes: %ld, %d apart\n", decodes, dfails);
  return fails || dfails ? 1 : 0;
}
