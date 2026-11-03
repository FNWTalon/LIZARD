// ai: The finder's front (src/acquire.c) on synthetic captures, every output hashed: binarize, shrink2 and the corner
// ai: marks' scan (both forms, scales 1, 1/2 and 1/4), on codes of the three rings painted through a homography with
// ai: a little perspective, sharp and blurred, over noise, at sizes whose rows and columns end ragged. arm64's forms
// ai: (OB_NEON) and x86-64's (OB_X86, -msse4.1) are exact, so a build with -DOB_SCALAR prints the same hash. On arm64
// ai: and x86-64 the column bits' walk (cb_run) is also held to the byte loop at every row, both ways, both colours and
// ai: every cap to 40, and the edge test's window read (edge_contrast_n) to edge_contrast on random rows and columns,
// ai: every length to past EDGE_REACH. The line fit (the block map, both scans' candidates and fit_lines' curves, at full scale) has
// ai: a hash of its own, exact on every target and pinned (LINE_HASH): the test fails where it differs. test/neon_kernels.sh
// ai: builds it both ways on a phone and compares. It includes acquire.c and links the rest of the codec:
// ai:   cc -O2 -ffp-contract=off -Isrc test/finder_neon.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any,shake}.c -lm
#include "acquire.c"
#include <stdio.h>
uint32_t ob_debug_hash;
static uint64_t hs = 1469598103934665603ull, hl = 1469598103934665603ull;
static void mix(const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { hs ^= b[i]; hs *= 1099511628211ull; } }
static void lmix(const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { hl ^= b[i]; hl *= 1099511628211ull; } }
// ai: the line fit's hash as the scalar twins give it (07807d8)
#define LINE_HASH 0x84ca16fdf20490d8ull
static uint32_t rs = 7;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }
#if defined(OB_NEON) || defined(OB_X86)
// ai: the walk check's own stream, so the captures are the same in every build
static uint32_t ws = 11;
static uint32_t wrnd(void) { ws = ws * 1664525u + 1013904223u; return ws >> 8; }
// ai: the walk the column bits replace: from row y of column x along dir, pixels equal to want, at most cap
static int byte_run(const uint8_t *bin, int w, int h, int x, int y, int dir, int want, int cap) {
  int n = 0;
  while (y >= 0 && y < h && (bin[(size_t)y * w + x] != 0) == want && n < cap) { n++; y += dir; }
  return n;
}
#endif
int main(void) {
  static const int SIZE[6][2] = { { 1280, 960 }, { 1277, 957 }, { 640, 480 }, { 1920, 1080 }, { 803, 1203 }, { 1500, 1001 } };
  static const int sides[3] = { 158, 286, 542 };
  long ncand = 0, nwalk = 0, bad = 0, nlines = 0;
  double tbin = 0, tmark = 0, tline = 0, tfit = 0;
  for (int t = 0; t < 24; t++) {
    const int iw = SIZE[t % 6][0], ih = SIZE[t % 6][1];
    uint8_t *px = malloc((size_t)iw * ih), *tmp = malloc((size_t)iw * ih), *bin = malloc((size_t)iw * ih);
    ob_layout_t L;
    ob_layout_init(&L, sides[t % 3]);
    // ai: the code over a fill of 0.55 to 0.9, turned up to 20 degrees, with its corners moved
    const float cx = iw * (0.45f + 0.1f * frand()), cy = ih * (0.45f + 0.1f * frand());
    const float s = 0.5f * (iw < ih ? iw : ih) * (0.55f + 0.35f * frand()), an = (frand() - 0.5f) * 0.7f, ca = cosf(an), sa = sinf(an);
    float quad[8];
    static const float CX[4] = { -1, 1, 1, -1 }, CY[4] = { -1, -1, 1, 1 };
    for (int k = 0; k < 4; k++) {
      const float x = CX[k] * s + (frand() - 0.5f) * 0.1f * s, y = CY[k] * s + (frand() - 0.5f) * 0.1f * s;
      quad[2 * k] = cx + ca * x - sa * y; quad[2 * k + 1] = cy + sa * x + ca * y;
    }
    float src[8];
    corner_coords(&L, src);
    homo_t Hi;
    homography(quad, src, &Hi);
    const int grain = 1 + (int)(rnd() % 6);
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
      float mx, my;
      project(&Hi, x + 0.5f, y + 0.5f, &mx, &my);
      const int ix = (int)floorf(mx), iy = (int)floorf(my);
      int v = (int)((((x / grain) * 2654435761u) ^ ((y / grain) * 2246822519u)) >> 24);   // a background of blocks
      if (ix >= 0 && iy >= 0 && ix < L.w && iy < L.h) {
        const int k = L.kind[iy * L.w + ix];
        v = k != CELL_DATA ? ((k & 3) == CELL_DARK ? 40 : 215) + (int)(rnd() % 21) - 10 : (int)(rnd() & 255);
      }
      px[y * iw + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    // ai: every third capture blurred (a 3 x 3 box, twice), which closes the marks' gaps for the merged form
    for (int pass = 0; t % 3 == 2 && pass < 2; pass++) {
      memcpy(tmp, px, (size_t)iw * ih);
      for (int y = 1; y < ih - 1; y++) for (int x = 1; x < iw - 1; x++) {
        int a = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) a += tmp[(y + dy) * iw + x + dx];
        px[y * iw + x] = (uint8_t)((a + 4) / 9);
      }
    }
    const uint8_t *grey = px;
    uint8_t *g2 = NULL;
    for (int scale = 1; scale <= 4; scale *= 2) {
      const int sw = iw / scale, sh = ih / scale;
      if (scale > 1) {
        uint8_t *next = shrink2(grey, iw / (scale / 2), ih / (scale / 2), NULL, scale == 2 ? OB_FB_HALF : OB_FB_QUARTER);
        mix(next, (size_t)sw * sh);
        frame_buf_done(NULL, g2); g2 = next; grey = next;
      }
#if defined(OB_NEON) || defined(OB_X86)
      // ai: the column bits as binarize makes them, and once more as it makes them without its planes (every band
      // ai: from the bytes): the same words
      const size_t nw = (size_t)((sh + 31) / 32) * sw;
      uint32_t *cb = malloc(nw * sizeof(uint32_t)), *cb2 = malloc(nw * sizeof(uint32_t));
      double t0 = ob_now_ms();
      binarize(grey, sw, sh, bin, cb, NULL);
      tbin += ob_now_ms() - t0;
      for (int b = 0; b < (sh + 31) / 32; b++) col_bits_band(bin, sw, sh, b, 0, cb2);
      if (memcmp(cb, cb2, nw * sizeof(uint32_t))) { if (bad++ < 5) printf("column bits differ at %d x %d\n", sw, sh); }
      free(cb2);
#else
      uint32_t *cb = NULL;
      double t0 = ob_now_ms();
      binarize(grey, sw, sh, bin, NULL, NULL);
      tbin += ob_now_ms() - t0;
#endif
      mix(bin, (size_t)sw * sh);
      if (scale == 1) {
        // ai: the line fit as find_frame runs it: its contrast floor, the block map, the four sides' candidates, the curves
        int hist[256] = { 0 }, cnt = 0, p5 = 0, p95 = 255, acc = 0, n[4], nl;
        for (int y = 4; y < sh; y += 8) for (int x = 4; x < sw; x += 8) { hist[grey[y * sw + x]]++; cnt++; }
        for (int v = 0; v < 256; v++) { acc += hist[v]; if (acc * 20 < cnt) p5 = v; if (acc * 20 < cnt * 19) p95 = v; }
        const int min_con = (p95 - p5) / 4 < 12 ? 12 : (p95 - p5) / 4, big = sw > sh ? sw : sh;
        fpt_t *pts[4];
        fline_t line[SIDE_LINES];
        edge_ctx_t ctx;
        for (int k = 0; k < 4; k++) pts[k] = malloc((size_t)(big / 2 + 1) * SIDE_K * sizeof(fpt_t));
        edge_ctx_init(&ctx, grey, sw, sh, min_con, NULL);
        for (int by = 0; by < (sh + BLK - 1) / BLK; by++) { lmix(ctx.mn + by * ctx.bs, (size_t)ctx.bw); lmix(ctx.mx + by * ctx.bs, (size_t)ctx.bw); }
        double t1 = ob_now_ms();
        row_points(bin, sw, sh, &ctx, pts[3], &n[3], pts[1], &n[1]);
        col_points(bin, sw, sh, &ctx, pts[0], &n[0], pts[2], &n[2]);
        tline += ob_now_ms() - t1;
        frame_buf_done(NULL, ctx.buf);
        for (int k = 0; k < 4; k++) {
          lmix(&n[k], sizeof n[k]); lmix(pts[k], (size_t)n[k] * sizeof(fpt_t));
          t1 = ob_now_ms();
          nl = fit_lines(pts[k], n[k], 1.5f, 0.12f * (k & 1 ? sh : sw), line);
          tfit += ob_now_ms() - t1;
          nlines += nl;
          lmix(&nl, sizeof nl); lmix(line, (size_t)nl * sizeof(fline_t));
          free(pts[k]);
        }
      }
      cand_t *c = malloc((size_t)MAX_CAND * sizeof(cand_t) + CAND_XY);
      for (int form = 0; form < 2; form++) {
        t0 = ob_now_ms();
        const int n = mark_scan(&L, bin, cb, sw, sh, scale, form, c);
        tmark += ob_now_ms() - t0;
        ncand += n;
        mix(&n, sizeof n); mix(c, (size_t)n * sizeof(cand_t));
      }
#if defined(OB_NEON) || defined(OB_X86)
      // ai: every row of 48 columns (the last among them), both ways, both colours, caps 0 to 40
      for (int k = 0; k < 48; k++) {
        const int x = k == 47 ? sw - 1 : (int)(wrnd() % (unsigned)sw);
        for (int y = -1; y <= sh; y++) for (int d = -1; d <= 1; d += 2) for (int want = 0; want < 2; want++) {
          const int cap = (int)(wrnd() % 41);
          nwalk++;
          if (cb_run(cb, sw, sh, x, y, d, want, cap) != byte_run(bin, sw, sh, x, y, d, want, cap)) {
            if (bad++ < 5) printf("cb_run differs: %d x %d, x %d y %d dir %d want %d cap %d: %d against %d\n", sw, sh, x, y, d, want, cap,
                                  cb_run(cb, sw, sh, x, y, d, want, cap), byte_run(bin, sw, sh, x, y, d, want, cap));
          }
        }
      }
#endif
      free(cb);
      free(c);
    }
    frame_buf_done(NULL, g2); free(px); free(tmp); free(bin);
    ob_layout_free(&L);
  }
  long ncon = 0;
#if defined(OB_NEON) || defined(OB_X86)
  {
    // ai: a line of 40 pixels as a row and as a column of a 7-wide picture, each edge with its window inside it
    uint8_t g[40 * 7];
    for (int t = 0; t < 20000; t++) {
      for (int i = 0; i < 40 * 7; i++) g[i] = (uint8_t)(wrnd() & 255);
      for (int e = EDGE_REACH; e + EDGE_REACH + 4 <= 40; e++) {
        const int inner = wrnd() & 1 ? 1 : -1, dl = (int)(wrnd() % 10) - 1, ll = (int)(wrnd() % 10) - 1;
        ncon += 2;
        if (edge_contrast_n(g, 1, e, inner, dl, ll) != edge_contrast(g, 1, e, inner, dl, ll) || edge_contrast_n(g + 3, 7, e, inner, dl, ll) != edge_contrast(g + 3, 7, e, inner, dl, ll)) {
          if (bad++ < 5) printf("edge_contrast_n differs: edge %d inner %d lengths %d %d\n", e, inner, dl, ll);
        }
      }
    }
  }
#endif
  printf("finder hash %016llx (candidates %ld)\n", (unsigned long long)hs, ncand);
  printf("line fit hash %016llx (curves %ld)%s\n", (unsigned long long)hl, nlines, hl == LINE_HASH ? "" : ", not the pinned one");
  if (nwalk) printf("column walks %ld and edge windows %ld, %ld apart\n", nwalk, ncon, bad);
  printf("ms: binarize %.2f, marks %.2f, points %.2f, fit %.2f\n", tbin, tmark, tline, tfit);
  return bad != 0 || hl != LINE_HASH;
}
