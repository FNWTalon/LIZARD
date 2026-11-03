// ai: The native encoder's paint (src/lizard_phy.c lizard_encode on arm64 and x86-64: the decoder's transforms run as
// ai: the inverses) against the format's (fft_cols: a -DOB_SCALAR build's, which the wasm paints alike), on every
// ai: version, LIZARD-64 to -1024, in each ring: a frame of ring_pairs' payload each, painted grey (lizard_paint_grey).
// ai: The reference build streams its pictures out ("emit"); the native build ("check") reads them and holds its own
// ai: to them: the border (every pixel outside the modules the picture and its guard interval are resampled over, the
// ai: quiet margin included) and the anchors' squares byte for byte, the rest within one grey level; the largest
// ai: difference and the pixels a level apart printed, with a hash of the native pictures (an exact change to the
// ai: native encoder keeps it). Exit 1 on a miss. From liblizard/ (src/{layout,fmt,rs,ldpc,fft,lizard_phy,acquire,any,shake}.c):
// ai:   cc -O3 -ffp-contract=off -DOB_SCALAR -Isrc -o build/paint_ref_scalar test/paint_ref.c <the codec> -lm
// ai:   cc -O3 -ffp-contract=off -msse4.1 -Isrc -o build/paint_ref test/paint_ref.c <the codec> -lm   (arm64: no -msse4.1)
// ai:   build/paint_ref_scalar emit | build/paint_ref check
#include "lizard_phy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
uint32_t ob_debug_hash;
int main(int argc, char **argv) {
  const int emit = argc > 1 && !strcmp(argv[1], "emit");
  if (!emit && !(argc > 1 && !strcmp(argv[1], "check"))) { fprintf(stderr, "usage: paint_ref emit | paint_ref check < emitted\n"); return 2; }
  int bad = 0, pics = 0;
  long ones = 0, pixels = 0;
  uint64_t hs = 1469598103934665603ull;
  for (int r = 0; r < LIZARD_RINGS; r++) for (int subch = 64; subch <= 1024; subch += LIZARD_STEP) {
    lizard_t f;
    if (lizard_init(&f, lizard_n_for(subch), subch, 2, 2 * LIZARD_RING[r], 0)) { fprintf(stderr, "lizard_init LIZARD-%d in ring %d failed\n", subch, LIZARD_RING[r]); return 1; }
    const int side = f.px, C = f.pxm, q = LIZARD_QUIET * C, W = side + 2 * q;
    uint8_t *blocks = malloc((size_t)f.blocks * f.block_bytes), *mine = malloc((size_t)W * W), *ref = malloc((size_t)W * W);
    float *drive = malloc((size_t)side * side * sizeof(float));
    for (int i = 0; i < f.blocks * f.block_bytes; i++) blocks[i] = (uint8_t)((uint32_t)((uint64_t)i * 2654435761ULL + (uint64_t)(r * 97 + subch)) >> 13);
    lizard_encode(&f, blocks, drive);
    lizard_paint_grey(&f, drive, mine);
    int hdr[3] = { subch, r, W };
    if (emit) {
      if (fwrite(hdr, sizeof hdr, 1, stdout) != 1 || fwrite(mine, (size_t)W * W, 1, stdout) != 1) { fprintf(stderr, "write failed\n"); return 1; }
    } else {
      int got[3];
      if (fread(got, sizeof got, 1, stdin) != 1 || memcmp(got, hdr, sizeof got) || fread(ref, (size_t)W * W, 1, stdin) != 1) { fprintf(stderr, "LIZARD-%d ring %d: the reference stream does not match\n", subch, LIZARD_RING[r]); return 1; }
      // ai: exact: outside the resampled modules [m0, m1) (drive pixels), and the anchors' squares
      const int m0 = f.margin - LIZARD_CP, m1 = f.frame.w - m0;
      int ag[8];
      lizard_anchor_geom(&f, ag);
      uint8_t *exact = calloc((size_t)W, 1), *ain = calloc((size_t)W, 1);   // ai: a drive line's exact columns and anchor columns, by symmetry rows too
      for (int x = 0; x < W; x++) { const int d = x - q; exact[x] = d < m0 * C || d >= m1 * C; }
      for (int i = 0; i < 7; i++) for (int x = ag[1 + i] * C; x < (ag[1 + i] + ag[0]) * C; x++) ain[x + q] = 1;
      int most = 0, apart = 0;
      for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
        const int d = abs((int)mine[(size_t)y * W + x] - (int)ref[(size_t)y * W + x]);
        const int must = exact[y] || exact[x] || (ain[y] && ain[x]);
        if (d > most) most = d;
        if (d && must) apart++;
        if (d == 1) ones++;
        if (d > 1) apart++;
      }
      pixels += (long)W * W; pics++;
      if (apart) { printf("LIZARD-%d ring %d: %d pixels apart past the bound (largest %d levels)\n", subch, LIZARD_RING[r], apart, most); bad++; }
      free(exact); free(ain);
    }
    for (size_t i = 0; i < (size_t)W * W; i++) { hs ^= mine[i]; hs *= 1099511628211ull; }
    lizard_free(&f); free(blocks); free(mine); free(ref); free(drive);
  }
  if (!emit) printf("paint_ref: %d pictures, %ld pixels: %ld a level apart (%.4f%%), %d past the bound; native paint hash %016llx\n",
                    pics, pixels, ones, 100.0 * ones / (pixels ? pixels : 1), bad, (unsigned long long)hs);
  else fprintf(stderr, "paint_ref emit: paint hash %016llx\n", (unsigned long long)hs);
  return bad != 0;
}
