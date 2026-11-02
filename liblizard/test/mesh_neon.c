// ai: The registration mesh (src/acquire.c) on synthetic frames, every output hashed: refine_node with and without a lens
// ai: model, anchor_find likewise (and again with a third of its anchors to find), border_blur at radii 1 to 7, the
// ai: mesh's sampler at random points (NaN and points off the image among them) at gamma 1 and 2.2, lensm_fit robust and
// ai: plain. arm64's own forms (OB_NEON) are the scalar twins' floats, so a build with -DOB_SCALAR prints the same hash,
// ai: as does x86-64 with or without -msse4.1; then each part's time. test/neon_kernels.sh builds it both ways on a
// ai: phone and compares. It includes acquire.c and links the rest of the codec:
// ai:   cc -O2 -ffp-contract=off -Isrc test/mesh_neon.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any,shake}.c -lm
#include "acquire.c"
#include <stdio.h>
uint32_t ob_debug_hash;
static uint64_t hs = 1469598103934665603ull;
static void mix(const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { hs ^= b[i]; hs *= 1099511628211ull; } }
static uint32_t rs = 1;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }
int main(void) {
  const int iw = 1280, ih = 960, sides[3] = { 158, 286, 542 };
  uint8_t *px = malloc((size_t)iw * ih);
  int nfound = 0, ngood = 0, nnodes = 0;
  double tref = 0, tanc = 0, tblur = 0, tfit = 0;
  for (int t = 0; t < 36; t++) {
    ob_layout_t L;
    ob_layout_init(&L, sides[t % 3]);
    if (t & 4) { uint8_t cw[64]; for (int k = 0; k < 64; k++) cw[k] = (uint8_t)rnd(); ob_layout_set_fmt(&L, cw); }
    // ai: the code painted through a homography with a little perspective: its border's cells, its anchors, noise
    const float cx = iw * 0.5f, cy = ih * 0.5f, s = 300 + 150 * frand();
    float quad[8] = { cx - s + 40 * frand(), cy - s * 0.8f + 40 * frand(), cx + s + 40 * frand(), cy - s * 0.8f + 40 * frand(),
                      cx + s + 40 * frand(), cy + s * 0.8f + 40 * frand(), cx - s + 40 * frand(), cy + s * 0.8f + 40 * frand() };
    float src[8];
    corner_coords(&L, src);
    homo_t Hi;
    homography(quad, src, &Hi);
    const int a = ob_anchor_side(&L);
    for (int y = 0; y < ih; y++) for (int x = 0; x < iw; x++) {
      float mx, my;
      project(&Hi, x + 0.5f, y + 0.5f, &mx, &my);
      const int ix = (int)floorf(mx), iy = (int)floorf(my);
      int v = 128 + (int)(rnd() % 31) - 15;
      if (ix >= 0 && iy >= 0 && ix < L.w && iy < L.h) {
        const int k = L.kind[iy * L.w + ix];
        v = k != CELL_DATA ? ((k & 3) == CELL_DARK ? 50 : 200) + (int)(rnd() % 21) - 10 : (int)(rnd() & 255);
        for (int q = 0; q < OB_ANCH_K * OB_ANCH_K && a > 0; q++) {
          const float x0 = (float)ob_anchor_m0(&L, q % OB_ANCH_K), y0 = (float)ob_anchor_m0(&L, q / OB_ANCH_K);
          if (mx < x0 || mx >= x0 + a || my < y0 || my >= y0 + a) continue;
          const float dc = fmaxf(fabsf(mx - x0 - a * 0.5f), fabsf(my - y0 - a * 0.5f)) / (a * 0.5f);
          v = dc < 0.25f || dc >= 0.6f ? 210 : 40;
        }
      }
      px[y * iw + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    image_t im;
    ob_image_init(&im, px, iw, ih, t % 6 == 5 ? 2.2f : 1.0f);
    // ai: the homography the search starts from: the corners moved 3 px, or 40 on every fourth frame
    float q2[8];
    for (int k = 0; k < 8; k++) q2[k] = quad[k] + (frand() - 0.5f) * (t % 4 == 3 ? 40 : 3);
    homo_t H0;
    homography(src, q2, &H0);
    lensm_t lm;
    lensm_axis(iw, ih, &lm);
    lm.K = (int)(rnd() % 4);
    for (int k = 0; k < LENS_P; k++) lm.t[k] = (frand() - 0.5f) * (k < 8 ? 2.0 : 8.0);
    const lensm_t *lmp = t & 1 ? &lm : 0;
    node_t *nodes = calloc((size_t)L.nx * L.ny, sizeof(node_t));
    double t0 = ob_now_ms();
    for (int ring = 0; ring < L.nx + L.ny; ring++) for (int j = 0; j < L.ny; j++) for (int i = 0; i < L.nx; i++) {
      const int di = i < L.nx - 1 - i ? i : L.nx - 1 - i, dj = j < L.ny - 1 - j ? j : L.ny - 1 - j;
      const int corner = (i == 0 || i == L.nx - 1) && (j == 0 || j == L.ny - 1);
      if (di + dj != ring || nodes[j * L.nx + i].done || !(corner || L.node_mark[j * L.nx + i])) continue;
      refine_node(&L, &im, &H0, i, j, nodes, lmp);
    }
    tref += ob_now_ms() - t0;
    mix(nodes, (size_t)L.nx * L.ny * sizeof(node_t));
    for (int k = 0; k < L.nx * L.ny; k++) if (nodes[k].done) { nnodes++; ngood += nodes[k].score >= 0.35f; }
    anchor_t A[OB_ANCH_K * OB_ANCH_K];
    memset(A, 0, sizeof A);
    t0 = ob_now_ms();
    const int f1 = anchor_find(&L, &im, &H0, lmp, A);
    tanc += ob_now_ms() - t0;
    nfound += f1; mix(&f1, sizeof f1); mix(A, sizeof A);
    for (int k = 0; k < OB_ANCH_K * OB_ANCH_K; k += 3) A[k].ok = 0;
    const int f2 = anchor_find(&L, &im, &H0, &lm, A);
    mix(&f2, sizeof f2); mix(A, sizeof A);
    t0 = ob_now_ms();
    uint8_t *bl = border_blur(&L, &H0, px, iw, ih, 1 + t % 7);
    tblur += ob_now_ms() - t0;
    mix(bl, (size_t)iw * ih);
    free(bl);
    for (int k = 0; k < 4096; k += 4) {
      float X[4], Y[4], o[4];
      for (int l = 0; l < 4; l++) {
        X[l] = frand() * (iw + 40) - 20; Y[l] = frand() * (ih + 40) - 20;
        if (rnd() % 97 == 0) X[l] = NAN;
        if (rnd() % 97 == 0) Y[l] = -INFINITY;
      }
#if defined(OB_NEON)
      vst1q_f32(o, msh_sample4(&im, vld1q_f32(X), vld1q_f32(Y)));
#elif defined(OB_SIMD)
      wasm_v128_store(o, sample4(&im, wasm_v128_load(X), wasm_v128_load(Y)));
#else
      for (int l = 0; l < 4; l++) o[l] = sample(&im, X[l], Y[l]);
#endif
      mix(o, sizeof o);
    }
    for (int fi = 0; fi < 40; fi++) {
      enum { NPM = 160 };
      double qx[NPM], qy[NPM], dx[NPM], dy[NPM];
      unsigned char isb[NPM], drop[NPM];
      const int np = 12 + (int)(rnd() % (NPM - 12));
      lensm_t m;
      lensm_axis(iw, ih, &m);
      const double k1 = (frand() - 0.5) * 4, k2 = (frand() - 0.5) * 4;
      for (int i = 0; i < np; i++) {
        qx[i] = (frand() - 0.5) * 1.6; qy[i] = (frand() - 0.5) * 1.2;
        const double r2 = qx[i] * qx[i] + qy[i] * qy[i];
        dx[i] = k1 * r2 * qx[i] + k2 * r2 * r2 * qx[i] + 0.3 * qx[i] + (frand() - 0.5) * 0.2;
        dy[i] = k1 * r2 * qy[i] + k2 * r2 * r2 * qy[i] - 0.2 * qy[i] + (frand() - 0.5) * 0.2;
        if (rnd() % 13 == 0) dx[i] += 5;
        isb[i] = (unsigned char)(rnd() & 1); drop[i] = 0;
      }
      t0 = ob_now_ms();
      const int ok = lensm_fit(np, qx, qy, dx, dy, fi & 1 ? isb : 0, drop, 0.3, &m);
      tfit += ob_now_ms() - t0;
      mix(&ok, sizeof ok); mix(&m.K, sizeof m.K); mix(m.t, sizeof m.t); mix(drop, (size_t)np);
    }
    free(nodes);
    ob_layout_free(&L);
  }
  free(px);
  printf("mesh hash %016llx (nodes %d, trusted %d, anchors found %d of %d)\n", (unsigned long long)hs, nnodes, ngood, nfound, 36 * OB_ANCH_K * OB_ANCH_K);
  printf("ms: nodes %.2f, anchors %.2f, blur %.2f, lens fits %.2f\n", tref, tanc, tblur, tfit);
  return 0;
}
