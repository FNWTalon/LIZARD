// ai: The registration mesh (src/acquire.c) on synthetic frames, every output hashed: refine_node with and without a lens
// ai: model, anchor_find likewise (and again with a third of its anchors to find), border_blur at radii 1 to 7, the
// ai: mesh's sampler at random points (NaN and points off the image among them) at gamma 1 and 2.2, lensm_fit robust and
// ai: plain; then each part's time. The scalar twins, x86-64 without -msse4.1 and a build with -DOB_SCALAR print one
// ai: hash (SCALAR_HASH); arm64's own forms (OB_NEON) and x86-64's ports of them (OB_X86, -msse4.1) read the picture in
// ai: integers and are their own values, so theirs are NEON_HASH and X86_HASH, and those builds also hold their reads to
// ai: sample() at every point not NaN: within 1/250 of the grey range (the x fraction is a 256th of a pixel; at the
// ai: frame's far edges sample() holds 0.999 of the last pixel, the read 255/256), the anchors' 12-bit greys within
// ai: MSH_Q / 250; and msh_blur_rect to box_blur_rect's bytes and msh_integral to a plain loop's sums (both exact) on
// ai: random rectangles and windows. Exit 1 on a hash, a bound or a byte missed. test/neon_kernels.sh builds it both
// ai: ways on a phone. It includes acquire.c and links the rest of the codec:
// ai:   cc -O2 -ffp-contract=off [-msse4.1] -Isrc test/mesh_neon.c src/{layout,fmt,rs,ldpc,fft,lizard_phy,any,shake}.c -lm
#include "acquire.c"
#include <stdio.h>
uint32_t ob_debug_hash;
#define SCALAR_HASH 0xf42c8b6f71af790eull
#define NEON_HASH 0xba9af2ba4a56adc7ull
#define X86_HASH 0x79e76f9d730624feull
static uint64_t hs = 1469598103934665603ull;
static void mix(const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) { hs ^= b[i]; hs *= 1099511628211ull; } }
static uint32_t rs = 1;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
#if defined(OB_NEON) || defined(OB_X86)
// ai: the exact checks' own stream, so the hashed stream is the same with them or without
static uint32_t rs2 = 7;
static uint32_t rnd2(void) { rs2 = rs2 * 1664525u + 1013904223u; return rs2 >> 8; }
#endif
static float frand(void) { return (float)(rnd() & 0xffff) / 65536.0f; }
int main(void) {
  const int iw = 1280, ih = 960, sides[3] = { 158, 286, 542 };
  uint8_t *px = malloc((size_t)iw * ih), *spare = malloc((size_t)iw * ih);   // ai: spare: border_blur's copy, as acquire_core hands it one
  int nfound = 0, ngood = 0, nnodes = 0;
  double tref = 0, tanc = 0, tblur = 0, tfit = 0;
#if defined(OB_NEON) || defined(OB_X86)
  double dread = 0, dq = 0;   // ai: the reads' largest distance from sample()
#endif
#if defined(OB_NEON) || defined(OB_X86)
  uint8_t *b1 = malloc((size_t)iw * ih), *b2 = malloc((size_t)iw * ih);   // ai: msh_blur_rect's and box_blur_rect's rectangles
  long nblur = 0, nint = 0;   // ai: rows where the two blurs differ, windows where the integral images do
#endif
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
#if defined(OB_NEON) || defined(OB_X86)
    // ai: msh_integral held to a plain loop's 32-bit sums (exact, the squares' image modulo 2^32) on random windows of
    // ai: 12-bit greys, at the three widths anchor_find takes
    {
      enum { WM = 64 };
      static uint16_t win[WM * WM];
      static uint32_t I1[(WM + 1) * (WM + 1) + 4], I2[(WM + 1) * (WM + 1) + 4], R1[(WM + 1) * (WM + 1)], R2[(WM + 1) * (WM + 1)];
      const int W = (const int[]){ 28, 40, 64 }[t % 3], W1 = W + 1;
      for (int k = 0; k < W * W; k++) win[k] = (uint16_t)(rnd2() % 8 ? rnd2() % (MSH_Q + 1) : (rnd2() & 1) * MSH_Q);
      msh_integral(win, W, I1, I2);
      for (int x = 0; x < W1; x++) R1[x] = R2[x] = 0;
      for (int y = 0; y < W; y++) {
        uint32_t s1 = 0, s2 = 0;
        R1[(y + 1) * W1] = R2[(y + 1) * W1] = 0;
        for (int x = 0; x < W; x++) {
          s1 += win[y * W + x]; s2 += (uint32_t)win[y * W + x] * win[y * W + x];
          R1[(y + 1) * W1 + x + 1] = R1[y * W1 + x + 1] + s1; R2[(y + 1) * W1 + x + 1] = R2[y * W1 + x + 1] + s2;
        }
      }
      nint += memcmp(I1, R1, (size_t)W1 * W1 * sizeof(uint32_t)) != 0 || memcmp(I2, R2, (size_t)W1 * W1 * sizeof(uint32_t)) != 0;
    }
#endif
    for (int k = 0; k < OB_ANCH_K * OB_ANCH_K; k += 3) A[k].ok = 0;
    const int f2 = anchor_find(&L, &im, &H0, &lm, A);
    mix(&f2, sizeof f2); mix(A, sizeof A);
    t0 = ob_now_ms();
    uint8_t *bl = border_blur(&L, &H0, px, iw, ih, 1 + t % 7, spare);
    tblur += ob_now_ms() - t0;
    mix(bl, (size_t)iw * ih);
#if defined(OB_NEON) || defined(OB_X86)
    // ai: msh_blur_rect held to box_blur_rect's bytes on random rectangles 16 to 315 wide, some against the frame's edges
    {
      const int r = 1 + t % 7;
      uint16_t *ring = malloc(((size_t)(2 * r + 4) * iw) * sizeof(uint16_t) + (size_t)iw + 2 * r), *tmp = malloc((size_t)iw * (ih + 2 * r) * sizeof(uint16_t));
      int *acc = malloc((size_t)iw * sizeof(int));
      for (int q = 0; q < 16; q++) {
        const int rw = 16 + (int)(rnd2() % 300), rh = 1 + (int)(rnd2() % 80);
        int x0 = q & 1 ? 0 : (int)(rnd2() % (unsigned)(iw - rw + 1)), y0 = q & 2 ? ih - rh : (int)(rnd2() % (unsigned)(ih - rh + 1));
        if (q & 4) x0 = iw - rw;
        msh_blur_rect(px, b1, iw, ih, r, x0, y0, x0 + rw, y0 + rh, ring, ring + (size_t)(2 * r + 2) * iw, (uint8_t *)(ring + (size_t)(2 * r + 4) * iw));
        box_blur_rect(px, b2, iw, ih, r, x0, y0, x0 + rw, y0 + rh, tmp, acc);
        for (int y = y0; y < y0 + rh; y++) nblur += memcmp(b1 + (size_t)y * iw + x0, b2 + (size_t)y * iw + x0, (size_t)rw) != 0;
      }
      free(ring); free(tmp); free(acc);
    }
#endif
    for (int k = 0; k < 4096; k += 4) {
      float X[4], Y[4], o[4];
      for (int l = 0; l < 4; l++) {
        X[l] = frand() * (iw + 40) - 20; Y[l] = frand() * (ih + 40) - 20;
        if (rnd() % 97 == 0) X[l] = NAN;
        if (rnd() % 97 == 0) Y[l] = -INFINITY;
      }
#if defined(OB_NEON)
      const msh_k_t K = msh_k(&im);
      const float32x4_t x = vsubq_f32(vld1q_f32(X), vdupq_n_f32(MSH_OX)), y = vsubq_f32(vld1q_f32(Y), vdupq_n_f32(MSH_OY));
      vst1q_f32(o, msh_read8(&im, &K, x, y, x, y).val[0]);
      uint16_t q[8];
      vst1q_u16(q, msh_read8q(&im, &K, x, y, x, y));
      for (int l = 0; l < 4; l++) {
        const float ref = sample(&im, X[l], Y[l]);
        if (ref != ref) continue;
        dread = fmax(dread, fabs(o[l] - ref)); dq = fmax(dq, fabs(q[l] - ref * MSH_Q));
      }
#elif defined(OB_X86)
      const msh_k_t K = msh_k(&im);
      const __m128 x = _mm_sub_ps(_mm_loadu_ps(X), _mm_set1_ps(MSH_OX)), y = _mm_sub_ps(_mm_loadu_ps(Y), _mm_set1_ps(MSH_OY));
      _mm_storeu_ps(o, msh_read8(&im, &K, x, y, x, y).val[0]);
      uint16_t q[8];
      _mm_storeu_si128((__m128i *)q, msh_read8q(&im, &K, x, y, x, y));
      for (int l = 0; l < 4; l++) {
        const float ref = sample(&im, X[l], Y[l]);
        if (ref != ref) continue;
        dread = fmax(dread, fabs(o[l] - ref)); dq = fmax(dq, fabs(q[l] - ref * MSH_Q));
      }
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
  free(px); free(spare);
  printf("mesh hash %016llx (nodes %d, trusted %d, anchors found %d of %d)\n", (unsigned long long)hs, nnodes, ngood, nfound, 36 * OB_ANCH_K * OB_ANCH_K);
  printf("ms: nodes %.2f, anchors %.2f, blur %.2f, lens fits %.2f\n", tref, tanc, tblur, tfit);
#if defined(OB_NEON) || defined(OB_X86)
  free(b1); free(b2);
  printf("msh_blur_rect against box_blur_rect: %ld rows differ; msh_integral against the plain sums: %ld windows differ\n", nblur, nint);
  if (nblur || nint) return 1;
#endif
#if defined(OB_NEON) || defined(OB_X86)
#if defined(OB_NEON)
  const uint64_t want = NEON_HASH;
#else
  const uint64_t want = X86_HASH;
#endif
  printf("reads against sample(): %.5f of the grey range, 12-bit %.2f of %d\n", dread, dq, (int)MSH_Q);
  if (dread > 1.0 / 250 || dq > MSH_Q / 250.0) { printf("a read past its bound\n"); return 1; }
#else
  const uint64_t want = SCALAR_HASH;
#endif
  if (hs != want) { printf("mesh hash differs from %016llx\n", (unsigned long long)want); return 1; }
  return 0;
}
