// ai: The finder's track reads (src/acquire.c thin_levels, thin_score, thin_read_both and the eight hypotheses' scores
// ai: from its reading) on synthetic captures: codes of the three rings painted through a homography with a little
// ai: perspective, sharp and blurred, over noise, read through the true map with and without fitted curves (both
// ai: parities of the bow), at gamma 1 (arm64's and x86-64's sample4r) and 2.2 (sample4), under two side maps.
// ai: arm64's and x86-64's forms have their own floats, so they are held to the scalar twins within a tolerance: the
// ai: scalar build (-DOB_SCALAR) prints its readings, a value a line, and a build given that list compares its own with
// ai: it: the flags and counts equal, a grey or a mean within 1e-3, a sum of k greys within k 1e-3, a score within 1e-3;
// ai: exit 1 past any. From liblizard/:
// ai:   cc -O2 -ffp-contract=off -DOB_SCALAR -Isrc -o build/track_scalar test/track_check.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any,shake}.c -lm
// ai:   cc -O2 -ffp-contract=off -msse4.1 -Isrc -o build/track_x86 test/track_check.c src/{...}.c -lm   (arm64: no -msse4.1)
// ai:   build/track_scalar > build/track_ref.txt && build/track_x86 build/track_ref.txt
#include "acquire.c"
#include <stdio.h>
uint32_t ob_debug_hash;
static uint32_t rs = 5;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }

// ai: the readings in order, each with its kind: 0 a flag or count (equal), 1 a grey or a mean, 2 a sum (k terms), 3 a score
enum { NMAX = 1 << 20 };
static double val[NMAX];
static int kind[NMAX], terms[NMAX], nv;
static void put(double v, int k, int t) { if (nv < NMAX) { val[nv] = v; kind[nv] = k; terms[nv] = t; nv++; } }

// ai: the eight hypotheses' scores from a reading, as find_frame takes them
static const int ENDS_T[4][2] = { { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 } };
static void hyp_scores(const ob_layout_t *L, const float *s, int npts, const float *pair, int nref, float con, float *sc) {
  int hrow[8][4];
  for (int hyp = 0; hyp < 8; hyp++) for (int sd = 0; sd < 4; sd++) {
    const int rot = hyp & 3, mir = hyp >> 2;
    const int ia = (rot + (mir ? 4 - ENDS_T[sd][0] : ENDS_T[sd][0])) & 3, ib = (rot + (mir ? 4 - ENDS_T[sd][1] : ENDS_T[sd][1])) & 3;
    const int m = ((ia + 1) & 3) == ib ? ia : ib;
    hrow[hyp][sd] = 4 * (ia != ENDS_T[m][0]) + m;
  }
  (void)L;
#if defined(OB_NEON) || defined(OB_X86)
  trk_scores(s, npts, hrow, pair, nref, con, 0, 2, sc);
#else
  for (int hyp = 0; hyp < 8; hyp++) {
    float trk = 0, q = 0;
    for (int sd = 0; sd < 4; sd++) trk += pair[hrow[hyp][sd]];
    const float ref = 0.5f * trk / nref;
    for (int sd = 0; sd < 4; sd++) for (int i = 0; i < npts; i++) {
      float qq = (ref - s[hrow[hyp][sd] * npts + i]) / con;
      if (qq > 0.5f) qq = 0.5f; else if (qq < -0.5f) qq = -0.5f;
      q += ob_thin_track(sd, i) ? qq : -qq;
    }
    sc[hyp] = q / (4 * npts);
  }
#endif
}

int main(int argc, char **argv) {
  static const int SIZE[4][2] = { { 1280, 960 }, { 1277, 957 }, { 1920, 1080 }, { 803, 1203 } };
  static const int sides[3] = { 158, 286, 542 };
  for (int t = 0; t < 24; t++) {
    const int iw = SIZE[t % 4][0], ih = SIZE[t % 4][1];
    uint8_t *px = malloc((size_t)iw * ih), *tmp = malloc((size_t)iw * ih);
    ob_layout_t L;
    ob_layout_init(&L, sides[t % 3]);
    // ai: the code over a fill of 0.55 to 0.9, turned up to 20 degrees, its corners moved (finder_neon.c's captures)
    const float cx = iw * (0.45f + 0.1f * frand()), cy = ih * (0.45f + 0.1f * frand());
    const float sz = 0.5f * (iw < ih ? iw : ih) * (0.55f + 0.35f * frand()), an = (frand() - 0.5f) * 0.7f, ca = cosf(an), sa = sinf(an);
    float quad[8], src[8];
    static const float CX[4] = { -1, 1, 1, -1 }, CY[4] = { -1, -1, 1, 1 };
    for (int k = 0; k < 4; k++) {
      const float x = CX[k] * sz + (frand() - 0.5f) * 0.1f * sz, y = CY[k] * sz + (frand() - 0.5f) * 0.1f * sz;
      quad[2 * k] = cx + ca * x - sa * y; quad[2 * k + 1] = cy + sa * x + ca * y;
    }
    corner_coords(&L, src);
    homo_t Hi, H;
    homography(quad, src, &Hi);
    homography(src, quad, &H);
    const int grain = 1 + (int)(rnd() % 6);
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
      float mx, my;
      project(&Hi, x + 0.5f, y + 0.5f, &mx, &my);
      const int ix = (int)floorf(mx), iy = (int)floorf(my);
      int v = (int)((((x / grain) * 2654435761u) ^ ((y / grain) * 2246822519u)) >> 24);
      if (ix >= 0 && iy >= 0 && ix < L.w && iy < L.h) {
        const int k = L.kind[iy * L.w + ix];
        v = k != CELL_DATA ? ((k & 3) == CELL_DARK ? 40 : 215) + (int)(rnd() % 21) - 10 : (int)(rnd() & 255);
      }
      px[y * iw + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    for (int pass = 0; t % 3 == 2 && pass < 2; pass++) {
      memcpy(tmp, px, (size_t)iw * ih);
      for (int y = 1; y < ih - 1; y++) for (int x = 1; x < iw - 1; x++) {
        int a = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) a += tmp[(y + dy) * iw + x + dx];
        px[y * iw + x] = (uint8_t)((a + 4) / 9);
      }
    }
    image_t im;
    ob_image_init(&im, px, iw, ih, t % 4 == 3 ? 2.2f : 1.0f);
    // ai: fitted curves near the true sides at the curves' depth: through the side's two ends there, a small bow
    fline_t curve[4];
    const fline_t *cvp[4];
    for (int m = 0; m < 4; m++) {
      float x0, y0, x1, y1, u0, v0, u1, v1;
      const float e = thin_edge_depth, len = m & 1 ? (float)L.h : (float)L.w;
      side_xy(&L, m, 0, e, &x0, &y0); side_xy(&L, m, len, e, &x1, &y1);
      project(&H, x0, y0, &u0, &v0); project(&H, x1, y1, &u1, &v1);
      const float a0 = m & 1 ? v0 : u0, a1 = m & 1 ? v1 : u1, c0 = m & 1 ? u0 : v0, c1 = m & 1 ? u1 : v1;
      const float a = (c1 - c0) / (a1 - a0), bow = (frand() - 0.5f) * 8e-5f;
      curve[m] = (fline_t){ a, c0 - a * a0, 0, bow, 0.5f * (a0 + a1) };
      cvp[m] = &curve[m];
    }
    const int npts = ob_track_cells(L.w, L.reserve);
    float *s = malloc((size_t)9 * npts * sizeof(float));
    for (int i = 0; i < npts; i++) s[8 * npts + i] = (float)(L.reserve + OB_TRACK_AT(i)) + OB_CELL_MID;
    for (int bend = 0; bend < 2; bend++) {
      const fline_t *const *cv = bend ? cvp : NULL;
      for (int mi = 0; mi < 2; mi++) {
        const int map[4] = { mi, (mi + 1) & 3, (mi + 2) & 3, (mi + 3) & 3 };
        float ref = 0, con = 0;
        const int lv = thin_levels(&L, &im, &H, cv, map, &ref, &con);
        put(lv, 0, 0); put(ref, 1, 1); put(con, 1, 1);
        if (lv) put(thin_score(&L, &im, &H, cv, map, ref, con, -1e9f), 3, 1);
      }
      float pair[8], sc[8];
      int nref = 0;
      const float con = thin_read_both(&L, &im, &H, cv, s, pair, npts, &nref);
      put(con > 0, 0, 0); put(nref, 0, 0); put(con, 1, 1);
      if (con > 0) {
        for (int k = 0; k < 8 * npts; k++) put(s[k], 1, 1);
        for (int k = 0; k < 8; k++) put(pair[k], 2, 2 * nref / 4);
        hyp_scores(&L, s, npts, pair, nref, con, sc);
        for (int k = 0; k < 8; k++) put(sc[k], 3, 1);
      }
    }
    free(s); free(px); free(tmp);
    ob_layout_free(&L);
  }
  if (argc < 2) { for (int i = 0; i < nv; i++) printf("%.9g\n", val[i]); return 0; }
  FILE *f = fopen(argv[1], "r");
  if (!f) { printf("no list at %s\n", argv[1]); return 1; }
  static const char *NAME[4] = { "flags and counts", "greys and means", "sums", "scores" };
  double worst[4] = { 0 };
  long bad = 0, n = 0;
  for (double r; n < nv && fscanf(f, "%lf", &r) == 1; n++) {
    const double d = fabs(val[n] - r), lim = kind[n] == 0 ? 0 : kind[n] == 2 ? 1e-3 * terms[n] : 1e-3;
    if (kind[n] == 2 ? d / terms[n] > worst[2] : d > worst[kind[n]]) worst[kind[n]] = kind[n] == 2 ? d / terms[n] : d;
    if (d > lim && bad++ < 5) printf("reading %ld (%s): %.9g against %.9g\n", n, NAME[kind[n]], val[n], r);
  }
  fclose(f);
  if (n != nv) { printf("%ld readings in the list, %d here\n", n, nv); return 1; }
  printf("track readings %d: %ld past the tolerance; largest differences: greys and means %.3g, sums %.3g a term, scores %.3g, flags %.3g\n",
         nv, bad, worst[1], worst[2], worst[3], worst[0]);
  return bad != 0;
}
