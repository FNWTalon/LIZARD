// ai: Do x86-64's wider LDPC layer kernels (src/ldpc.c: AVX2, AVX-512) decide as the scalar twin does? Two checks:
// ai:   1. every layer kernel the build and the processor have (the 128-bit one, AVX2, AVX-512) against the scalar
// ai:      loop on random layers: 3 to 32 slots, posteriors over the decoder's whole range and its ends, messages over
// ai:      +-127, both outputs (the posteriors and the messages) byte for byte;
// ai:   2. whole decodes of every code of the profile, random codewords through BI-AWGN at the decoder's int8 scale
// ai:      from below its threshold to clean, decoded with each kernel forced: the iterations and every output bit
// ai:      alike.
// ai: From liblizard/: gcc -O3 -ffp-contract=off -msse4.1 -Isrc -o build/ldpc_wide test/ldpc_wide.c src/fmt.c -lm
// ai: && build/ldpc_wide [layers] [codewords]. Exit 1 on any difference; it prints which kernels the processor ran.
#include "../src/ldpc.c"
#include <math.h>
#include <stdio.h>

static uint64_t rng = 88172645463325252ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double unif(void) { return (next() >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(void) { double u = unif() + 1e-300, v = unif(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

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

int main(int argc, char **argv) {
  const int layers = argc > 1 ? atoi(argv[1]) : 20000, words = argc > 2 ? atoi(argv[2]) : 2000, zp = 128;
  int fails = 0;
  int have[3] = { 1, 0, 0 };
#ifdef OB_LDPC_WIDE
  __builtin_cpu_init();
  have[1] = __builtin_cpu_supports("avx2");
  have[2] = __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512f");
#endif
  printf("kernels: 128-bit%s%s%s\n",
#ifdef OB_SIMD
         " (vector)",
#else
         " (the scalar twin)",
#endif
         have[1] ? ", AVX2" : "", have[2] ? ", AVX-512" : "");
  enum { NS_MAX = 32 };
  static int16_t q0[NS_MAX * 128], qa[NS_MAX * 128], qb[NS_MAX * 128];
  static int8_t r0[NS_MAX * 128], ra[NS_MAX * 128], rb[NS_MAX * 128];
  long checked[3] = { 0, 0, 0 };
  for (int t = 0; t < layers; t++) {
    const int ns = 3 + (int)(next() % (NS_MAX - 2)), norm = t % 5 == 0 ? 16 : 13;
    for (int k = 0; k < ns * zp; k++) { q0[k] = pick16(); r0[k] = (int8_t)((int)(next() % 255) - 127); }
    memcpy(qa, q0, sizeof q0); memcpy(ra, r0, sizeof r0);
    kernel_scalar(qa, ra, ns, zp, norm);
    for (int w = 0; w < 3; w++) {
      if (!have[w]) continue;
      memcpy(qb, q0, sizeof q0); memcpy(rb, r0, sizeof r0);
#ifdef OB_LDPC_WIDE
      if (w == 2) layer_kernel_avx512(qb, rb, ns, zp, norm);
      else if (w == 1) layer_kernel_avx2(qb, rb, ns, zp, norm);
      else
#endif
        layer_kernel(qb, rb, ns, zp, norm);
      checked[w]++;
      if (memcmp(qa, qb, (size_t)ns * zp * 2) || memcmp(ra, rb, (size_t)ns * zp)) {
        if (fails < 5) printf("layer %d (ns %d, norm %d): kernel %d apart from the scalar loop\n", t, ns, norm, w);
        fails++;
      }
    }
  }
  printf("layers: %ld, %ld, %ld against the scalar loop (128-bit, AVX2, AVX-512), %d apart\n", checked[0], checked[1], checked[2], fails);

  static const int RATES[4] = { 6, 4, 3, 2 };
  static const double DB[4][3] = { { 5.4, 5.9, 7.0 }, { 3.8, 4.3, 5.5 }, { 2.7, 3.2, 4.5 }, { 0.8, 1.3, 2.5 } };
  int dfails = 0;
  long decodes = 0;
  for (int ci = 0; ci < 4; ci++) {
    ldpc_t c;
    if (ldpc_init(&c, RATES[ci])) { printf("ldpc_init %d failed\n", RATES[ci]); return 1; }
    uint8_t *data = calloc((size_t)c.k, 1), *cw = calloc((size_t)c.n, 1), *out = calloc((size_t)c.n, 1), *ref = calloc((size_t)c.n, 1);
    int8_t *llr = calloc((size_t)c.n, 1);
    const int chosen = c.wide;
    for (int wi = 0; wi < words; wi++) {
      for (int v = 0; v < c.k; v++) data[v] = v < 3840 ? (uint8_t)(next() & 1) : 0;
      ldpc_encode(&c, data, cw);
      const double db = DB[ci][wi % 3], n0 = pow(10, -db / 10), sc = 0.7 * 2 * sqrt(2) / n0;
      for (int v = 0; v < c.n; v++) {
        double y = (1 - 2 * cw[v]) / sqrt(2) + gauss() * sqrt(n0 / 2), l = sc * y;
        if (l > 10) l = 10;
        if (l < -10) l = -10;
        llr[v] = (int8_t)lrint(8 * l);
      }
      for (int v = 3840; v < c.k; v++) llr[v] = 127;
      int its0 = 0;
      for (int w = 0; w < 3; w++) {
        if (!have[w]) continue;
        c.wide = w;
        const int its = ldpc_decode_stall(&c, llr, w ? out : ref, 30, 9, 0.95f);
        decodes++;
        if (w == 0) its0 = its;
        else if (its != its0 || memcmp(out, ref, (size_t)c.n)) {
          if (dfails < 5) printf("rate %d word %d at %.1f dB: kernel %d apart (iterations %d against %d)\n", RATES[ci], wi, db, w, its, its0);
          dfails++;
        }
      }
    }
    printf("rate %d (%s): %d codewords decoded by each kernel the processor has, init chose kernel %d\n", RATES[ci], ldpc_rate_name[RATES[ci]], words, chosen);
    free(data); free(cw); free(out); free(ref); free(llr);
    ldpc_free(&c);
  }
  printf("decodes: %ld, %d apart\n", decodes, dfails);
  return fails || dfails;
}
