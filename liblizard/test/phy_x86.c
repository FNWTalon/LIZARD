// ai: x86-64's own forms of the back half's exact kernels (src/lizard_phy.c, OB_X86), each held to its plain loop on
// ai: inputs the decodes never reach, at every picture size (the formats of test/neon_check.c, 384 to 1536) in the 128
// ai: ring: map_in on every block, its slots over every byte value (-128 among them) with the frame's whitening, against
// ai: the loop's (int8_t)-x; soft_q on NaN, infinities, values past int32's range, ties at a half and the clamp's edges, at
// ai: scales NaN and infinite among them, against the loop's clamp at +-10 and nearbyintf (a NaN taken as 0, as the
// ai: loop's conversion gives it on x86-64); turn4 against turned() on every coefficient the frame carries, at shifts up
// ai: to +-4 and at +-300. Every byte and float compared exactly. The bit packing and the pilots' shifts are held by the
// ai: decodes (a wrong byte fails its block's CRC). Exit 1 on any difference. It includes lizard_phy.c:
// ai:   gcc -O3 -ffp-contract=off -msse4.1 -Isrc -o build/phy_x86 test/phy_x86.c src/{layout,fmt,rs,ldpc,fft,acquire,any,shake}.c -lm
#include "lizard_phy.c"
#include <stdio.h>
uint32_t ob_debug_hash;
#ifdef OB_X86
static uint32_t rs = 7;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffffff) / 16777216.0f; }
// ai: the plain loop's soft value: k y clamped at +-10, times 8, rounded; NaN 0
static int8_t soft_ref(float k, float y) {
  float lr = k * y;
  lr = lr > 10 ? 10 : lr < -10 ? -10 : lr;
  return lr != lr ? 0 : (int8_t)nearbyintf(lr * 8);
}
#endif
int main(void) {
#ifndef OB_X86
  printf("phy_x86: not an x86-64 SSE4.1 build (OB_X86), nothing to check\n");
  return 0;
#else
  static const int SUBCH[] = { 64, 80, 96, 320, 336, 656, 800, 1024 };
  long bad = 0, nmap = 0, nturn = 0, nsoft = 0;
  for (size_t si = 0; si < sizeof SUBCH / sizeof *SUBCH; si++) {
    const int subch = SUBCH[si], n = lizard_n_for(subch);
    lizard_t f;
    if (lizard_init(&f, n, subch, 2 * LIZARD_RING[LIZARD_RING_DEFAULT])) { printf("lizard_init %d failed\n", subch); return 1; }
    struct lizard_ws *w = f.ws;
    // ai: map_in, every block
    const int mn = max_n(&f), mb = max_bits(&f);
    int8_t *slot = malloc((size_t)mb), *ca = malloc((size_t)mn + 64), *cb = malloc((size_t)mn + 64);
    for (int b = 0; b < f.blocks; b++) {
      const int t = f.block_tier[b], first = f.block_slot[b] / 2;
      const ldpc_t *c = &f.code[t];
      for (int i = 0; i < mb; i++) slot[i] = (int8_t)(i < 256 ? i - 128 : (int)(rnd() & 255) - 128);
      memset(ca, 0x55, (size_t)mn + 64); memset(cb, 0x55, (size_t)mn + 64);
      map_in(&f, w, t, first, slot, ca);
      const uint8_t *wh = w->white + 2 * (size_t)first;
      for (int i = c->k - c->np; i < c->k; i++) cb[i] = 0;
      for (int i = 0; i < c->nt; i++) cb[w->perm[t][i]] = wh[i] ? (int8_t)-slot[i] : slot[i];
      if (memcmp(ca, cb, (size_t)mn + 64)) { printf("map_in apart: LIZARD-%d block %d\n", subch, b); bad++; }
      nmap++;
    }
    free(slot); free(ca); free(cb);
    // ai: turn4 against turned(), every coefficient the frame carries, in fours as soft4 takes them
    const int all = f.subch * LIZARD_SUB;
    const size_t tn = (size_t)f.vblocks * n * f.bw;
    for (size_t i = 0; i < tn; i++) { w->tre[i] = frand() * 2 - 1; w->tim[i] = frand() * 2 - 1; }
    for (int rep = 0; rep < 6; rep++) {
      const float big = rep >= 4 ? 300.0f : 4.0f, d[2] = { (frand() * 2 - 1) * big, (frand() * 2 - 1) * big };
      const turn4_t tt = turn4_at(&f, d);
      for (int i = 0; i + 4 <= all; i += 4) {
        __m128 yr, yi;
        turn4(&tt, w->tre, w->tim, w->rot, _mm_loadu_si128((const __m128i *)(const void *)(f.pos + i)), &yr, &yi);
        float ar[4], ai[4];
        _mm_storeu_ps(ar, yr); _mm_storeu_ps(ai, yi);
        for (int l = 0; l < 4; l++) {
          float er, ei;
          turned(&f, w, f.pos[i + l], d[0], d[1], &er, &ei);
          if (memcmp(&er, &ar[l], 4) || memcmp(&ei, &ai[l], 4)) { if (bad < 10) printf("turn4 apart: LIZARD-%d coefficient %d (%g %g against %g %g)\n", subch, i + l, ar[l], ai[l], er, ei); bad++; }
          nturn++;
        }
      }
    }
    lizard_free(&f);
  }
  // ai: soft_q: scales and values at the edges, then at random
  static const float KS[] = { 0.125f, 1.0f, 1.25f, 0.0f, 1e-30f, 1e30f, 1.0f / 0.0f, 0.0f / 0.0f, -1.0f };
  float ys[512];
  int ny = 0;
  const float edge[] = { 0.0f, -0.0f, 0.0f / 0.0f, 1.0f / 0.0f, -1.0f / 0.0f, 1e30f, -1e30f, 2147483648.0f, -2147483648.0f, 4294967296.0f, -4294967296.0f,
                         79.5f, -79.5f, 80.0f, -80.0f, 80.5f, -80.5f, 10.0f, -10.0f, 1e-40f, -1e-40f };
  for (size_t i = 0; i < sizeof edge / sizeof *edge; i++) ys[ny++] = edge[i];
  for (int m = -100; m <= 100; m++) ys[ny++] = (float)m + 0.5f;
  while (ny < 512) ys[ny++] = (frand() * 2 - 1) * 20.0f;
  for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++) {
    const float k = KS[ki];
    for (int i = 0; i + 8 <= ny; i += 8) {
      int8_t got[16], want[8];
      _mm_storeu_si128((__m128i *)(void *)got, soft_q(_mm_set1_ps(8 * k), _mm_loadu_ps(ys + i), _mm_loadu_ps(ys + i + 4)));
      for (int l = 0; l < 8; l++) want[l] = soft_ref(k, ys[i + l]);
      if (memcmp(got, want, 8)) { printf("soft_q apart: k %g at values %d to %d\n", k, i, i + 7); bad++; }
      nsoft += 8;
    }
  }
  printf("phy_x86: map_in %ld blocks, turn4 %ld coefficients, soft_q %ld values: %ld apart\n", nmap, nturn, nsoft, bad);
  return bad != 0;
#endif
}
