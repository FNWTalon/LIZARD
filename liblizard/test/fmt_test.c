// The format word (src/fmt.c) at each ring's code: RS(8,3), RS(16,3), RS(24,3), RS(32,3) over every word cell of the
// 32, 64, 96 and 128 ring. Lengths from the layout, the SPEC test vectors, round trips through the soft values, whole sides lost, errors
// with and without the least confident bytes erased, the acceptance checks, and how often garbage is accepted.
//   gcc -O2 -Wall -Wextra -Isrc -o build/fmt_test test/fmt_test.c src/fmt.c src/rs.c src/layout.c src/ldpc.c -lm && build/fmt_test
//   build/fmt_test N      N garbage reads a length through ob_fmt_decode (default 20000, about 10 s; 2000000 for
//                         counts worth reading, about 10 minutes)
#include "../src/fmt.h"
#include "../src/layout.h"
#include "../src/rs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t st = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (uint32_t)(st >> 32); }
static float uni(void) { return (float)(rnd() >> 8) / 16777216.0f; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// The four rings: B band cells a side, modules a side S = 2B + 30 (src/focus.h), band reserve 15
// (ob_thin_reserve(OB_THIN_CORNER)); W word cells a side, RS(BS, 3). Any ring carries any version (SPEC 5.2).
static const int RB[4] = { 32, 64, 128, 256 }, WS[4] = { 16, 32, 64, 128 }, BS[4] = { 8, 16, 32, 64 };   // ai: the 96 ring (48, 24) until 2026-10-10

// SPEC.md 5.2's seven words, a the ring's index (test/vectors.mjs holds the wasm build to the same words). A change
// here is a change to the format.
static const struct { int a, version, fps; const char *hex; } VEC[] = {
  { 0, 2, 0, "4c0200723f148394" },
  { 1, 64, 24, "4c401843c236ee95dc072a1c61cb63fc" },
  { 1, 128, 240, "4c80f01a03b06a272bf125d409be9dd9" },
  { 1, 72, 30, "4c481e8a5889b2ffdc13581b53702083" },
  { 1, 2, 0, "4c0200bf8a272653d9ff5f11927947ed" },
  { 2, 8, 60, "4c083c787c7c416d2b6b3fc18075004ad5cd92b1945ab6165f18ed021c053ff6" },
  { 3, 32, 120, "4c2078c9d4c6572e79e041b1f59cf0bf61b4d51a391c456426ff193c5616df3bfa872faddf956cda9ffb1371b5bfca1460e86e4f278c5a2c1e79b982387ef7f7" },
};

// Soft values as the reader lays them out, q[side * cells + i]: +mag for a dark cell, -mag for a light one.
static void soften(const uint8_t *cw, int bytes, int cells, float mag, float *q) {
  for (int s = 0; s < 4; s++) for (int i = 0; i < cells; i++) q[s * cells + i] = ob_fmt_bit(cw, bytes, s, i) ? mag : -mag;
}
static float *cell_of(float *q, int cells, int k, int b) { return q + (k & 3) * cells + 8 * (k >> 2) + b; }
// Byte k read wrong with confidence mag (the word's own bits flipped), or faded to its local level, which is what a
// side under glare reads as (src/acquire.c ob_thin_read_fmt).
static void flip_byte(float *q, int cells, int k, float mag) { for (int b = 0; b < 8; b++) { float *c = cell_of(q, cells, k, b); *c = *c > 0 ? -mag : mag; } }
static void fade_side(float *q, int cells, int s) { for (int i = 0; i < cells; i++) q[s * cells + i] = (uni() - 0.5f) * 0.04f; }

// 1 decoded right, 0 nothing, -1 a wrong word accepted.
static int read(const float *q, int cells, int bytes, int vlo, int vhi, int version, int fps) {
  ob_fmt_t f = { 0, 0 };
  if (ob_fmt_decode(q, cells, bytes, vlo, vhi, &f)) return 0;
  return f.version == version && f.fps == fps ? 1 : -1;
}

static double binom(int n, int k) { double r = 1; for (int i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; }
// A received word uniform at random passes a bounded-distance decode of RS(n, 3) with f erasures when its other n - f
// bytes lie within t = (nroots - f) / 2 of a codeword of the punctured code: V(n - f, t) / 256^(nroots - f).
static double p_pass(int n, int f) {
  const int nroots = n - 3, t = (nroots - f) / 2;
  double v = 0;
  for (int i = 0; i <= t; i++) v += binom(n - f, i) * pow(255, i);
  return v / pow(256, nroots - f);
}

int main(int argc, char **argv) {
  const long NG = argc > 1 ? atol(argv[1]) : 20000;
  float q[4 * 160];
  uint8_t cw[OB_FMT_BYTES_MAX + 1];

  // Lengths from the layout, as a receiver takes them from the module count it registered.
  const int r = ob_thin_reserve(OB_THIN_CORNER);
  for (int a = 0; a < 4; a++) {
    const int S = 2 * RB[a] + 30;
    CHECK(ob_fmt_side_cells(S, r) == WS[a] && ob_fmt_bytes(S, r) == BS[a], "ring %d: W %d, %d bytes", RB[a], ob_fmt_side_cells(S, r), ob_fmt_bytes(S, r));
  }
  CHECK(ob_fmt_bytes(90, r) == 0 && ob_fmt_side_cells(90, r) == 0, "a side too short for the shortest word");
  CHECK(ob_fmt_bytes(700, r) == OB_FMT_BYTES_MAX, "a band past the longest word");

  // The vectors, and the encoder's range.
  for (size_t v = 0; v < sizeof VEC / sizeof VEC[0]; v++) {
    const int a = VEC[v].a;
    const ob_fmt_t f = { VEC[v].version, VEC[v].fps };
    char hex[2 * OB_FMT_BYTES_MAX + 1];
    CHECK(ob_fmt_encode(&f, cw, BS[a]) == 0, "encode version %d", f.version);
    for (int k = 0; k < BS[a]; k++) sprintf(hex + 2 * k, "%02x", cw[k]);
    CHECK(!strcmp(hex, VEC[v].hex), "vector ring %d version %d fps %d: %s", RB[a], f.version, f.fps, hex);
  }
  { const ob_fmt_t f = { 2, 0 }, v0 = { 0, 0 }, v129 = { 129, 0 }, fps = { 2, 256 };
    CHECK(ob_fmt_encode(&f, cw, 16) == 0 && ob_fmt_encode(&f, cw, 7) < 0 && ob_fmt_encode(&f, cw, OB_FMT_BYTES_MAX + 1) < 0, "encode lengths");
    CHECK(ob_fmt_encode(&v0, cw, 8) < 0 && ob_fmt_encode(&v129, cw, 8) < 0 && ob_fmt_encode(&fps, cw, 8) < 0, "encode fields"); }

  for (int a = 0; a < 4; a++) {
    const int W = WS[a], B = BS[a], nroots = B - 3, t = nroots / 2;
    // ai: The erasure steps' last (src/fmt.c): nroots - 1 at RS(8, 3), the 32 ring, else nroots - 3. Weak bytes past it are
    // ai: errors over the roots it leaves, so reach is the most weak bytes a word reads with.
    const int last = B == OB_FMT_BYTES ? nroots - 1 : nroots - 3, reach = last + (nroots - last) / 2;
    int ok = 0, wrong = 0, tries = 0;
    printf("RS(%d,3), ring %d, %d word cells a side, %d bytes a side, %d roots\n", B, RB[a], W, B / 4, nroots);
    // Round trips, with cell noise.
    for (int it = 0; it < 500; it++) {
      const ob_fmt_t f = { 1 + (int)(rnd() % 128), (int)(rnd() % 256) };
      ob_fmt_encode(&f, cw, B);
      soften(cw, B, W, 0.3f, q);
      for (int i = 0; i < 4 * W; i++) q[i] += (uni() - 0.5f) * 0.2f;
      const int r = read(q, W, B, 1, 128, f.version, f.fps);
      ok += r == 1; wrong += r < 0; tries++;
    }
    CHECK(ok == tries && !wrong, "round trips %d of %d, %d wrong", ok, tries, wrong);
    // Whole sides lost: 1, 2 (adjacent and opposite) and 3. A side is B / 4 bytes, erasures once faded.
    for (int lost = 1; lost <= 3; lost++) {
      int got = 0, bad = 0, n = 0;
      for (int m = 0; m < 16; m++) {
        if (__builtin_popcount(m) != lost) continue;
        for (int it = 0; it < 50; it++) {
          const ob_fmt_t f = { 1 + (int)(rnd() % 128), (int)(rnd() % 256) };
          ob_fmt_encode(&f, cw, B);
          soften(cw, B, W, 0.4f, q);
          for (int s = 0; s < 4; s++) if (m >> s & 1) fade_side(q, W, s);
          const int r = read(q, W, B, 1, 128, f.version, f.fps);
          got += r == 1; bad += r < 0; n++;
        }
      }
      // ai: lost * B / 4 faded bytes, read while they are within reach.
      const int need = lost * B / 4, in = need <= reach;
      printf("  %d side%s lost, %2d bytes: read %3d of %3d, wrong %d%s\n", lost, lost > 1 ? "s" : "", need, got, n, bad, in ? "" : "  (past the steps)");
      CHECK(!bad, "%d sides lost: %d wrong", lost, bad);
      CHECK(in ? got == n : got < n / 50 + 1, "%d sides lost: read %d of %d, reach %d", lost, got, n, reach);
    }
    // ai: Errors. The error bytes are made the most confident (0.45 against 0.4), so no erasure step lands on one: t
    // ai: read, t + 1 does not. Then errors in the least confident bytes (0.05), which the steps erase: reach of them
    // ai: read, and nroots - 3 with one confident error; reach + 1 do not.
    struct { int conf, weak, expect; } E[] = { { t, 0, 1 }, { t + 1, 0, 0 }, { 0, reach, 1 }, { 1, nroots - 3, 1 }, { 0, reach + 1, 0 } };
    for (size_t e = 0; e < sizeof E / sizeof E[0]; e++) {
      int got = 0, bad = 0;
      for (int it = 0; it < 200; it++) {
        const ob_fmt_t f = { 1 + (int)(rnd() % 128), (int)(rnd() % 256) };
        int pos[OB_FMT_BYTES_MAX], used[OB_FMT_BYTES_MAX] = { 0 };
        ob_fmt_encode(&f, cw, B);
        soften(cw, B, W, 0.4f, q);
        for (int i = 0; i < E[e].conf + E[e].weak; i++) { int p; do p = (int)(rnd() % B); while (used[p]); used[p] = 1; pos[i] = p; }
        for (int i = 0; i < E[e].conf + E[e].weak; i++) flip_byte(q, W, pos[i], i < E[e].conf ? 0.45f : 0.05f);
        const int r = read(q, W, B, 1, 128, f.version, f.fps);
        got += r == 1; bad += r < 0;
      }
      printf("  %2d confident errors, %2d weak: read %3d of 200, wrong %d\n", E[e].conf, E[e].weak, got, bad);
      CHECK(E[e].expect ? got == 200 && !bad : !got, "%d confident, %d weak: read %d, wrong %d", E[e].conf, E[e].weak, got, bad);
    }
    // Acceptance: a codeword RS takes that is not a word.
    { uint8_t c[OB_FMT_BYTES_MAX] = { 0x4d, 8, 60 };
      rs_encode(c, 3, nroots); soften(c, B, W, 0.4f, q);
      CHECK(read(q, W, B, 1, 128, 8, 60) == 0, "magic 0x4d");
      for (int v = 0; v <= 129; v += 129) { c[0] = 0x4c; c[1] = (uint8_t)v; rs_encode(c, 3, nroots); soften(c, B, W, 0.4f, q); CHECK(read(q, W, B, 1, 128, v, 60) == 0, "version %d", v); }
      c[1] = 128; rs_encode(c, 3, nroots); soften(c, B, W, 0.4f, q); CHECK(read(q, W, B, 1, 128, 128, 60) == 1, "version 128");
      c[1] = 3; c[2] = 255; rs_encode(c, 3, nroots); soften(c, B, W, 0.4f, q); CHECK(read(q, W, B, 1, 128, 3, 255) == 1, "version 3, fps 255");
    }

    // ai: Garbage: soft values uniform in -0.5 .. 0.5, what a wrong grid reads. Exact rates first: RS passing a step,
    // ai: the steps together, then with the magic and a version of 1 .. 128 (a false codeword's data bytes are
    // ai: uniform). Then measured through ob_fmt_decode, with the other last step's rate beside it (nroots - 3 and
    // ai: nroots - 1).
    const int other = last == nroots - 1 ? nroots - 3 : nroots - 1;
    double steps = p_pass(B, 0), alt = p_pass(B, 0);
    for (int f = 2; f <= last; f += 2) steps += p_pass(B, f);
    for (int f = 2; f <= other; f += 2) alt += p_pass(B, f);
    const double mv = steps / 512;
    long acc = 0;
    for (long it = 0; it < NG; it++) {
      for (int i = 0; i < 4 * W; i++) q[i] = uni() - 0.5f;
      ob_fmt_t f;
      acc += ob_fmt_decode(q, W, B, 1, 128, &f) == 0;
    }
    printf("  garbage, exact: errors only %.1e, steps to %d %.1e, magic and version %.1e (steps to %d: %.1e)\n",
           p_pass(B, 0), last, steps, mv, other, alt / 512);
    printf("  garbage, measured over %ld reads: %ld accepted (expected %.2f)\n", NG, acc, mv * NG);
    CHECK(acc <= 10 + 4 * mv * NG, "garbage accepted %ld of %ld", acc, NG);
  }
  printf(fails ? "%d FAILED\n" : "all passed\n", fails);
  return fails != 0;
}
