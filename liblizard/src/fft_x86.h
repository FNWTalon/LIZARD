// ai: The decoder's transform kernels on x86-64 (fft.c, OB_X86), one source at two widths: fft.c includes this twice,
// ai: at four lanes (SSE4.1, x86-64-v2: the names as they stand) and at eight (AVX, x86-64-v3: FX_N's _v3 names, each
// ai: compiled for the level), so the butterflies are written once. The columns past the last whole vector go to the
// ai: v2 kernel at eight lanes (FX_TAIL), and to the scalar tail at four: every column takes the same operations at
// ai: either width, so the v3 transform is the v2 one's bit for bit.
// ai: Before each include: FX_V (the vector type), FX_W (its lanes), FX_ADD, FX_SUB, FX_MUL, FX_SET1, FX_LD, FX_ST,
// ai: FX_FN (a kernel's attributes), FX_N(name) (its name), FX_TAIL(v2's tail, v3's tail); fft.c undefines them after.
// ai: arm64's kernels (fft.c's OB_NEON section), the same butterflies, data layout and plans, in SSE's and AVX's instructions: a multiply
// ai: then an add where NEON fuses the two (no FMA in the baseline), a twiddle or constant as a splat where NEON takes a
// ai: lane, so their own floats again (test/fft_dec.c holds them to the definition)
// ai: a twiddle: (qr, qi) times the splats (wr, wi)
#define TWS(qr, qi, wr, wi) { const FX_V pr_ = qr, pi_ = qi; qr = FX_SUB(FX_MUL(pr_, wr), FX_MUL(pi_, wi)); qi = FX_ADD(FX_MUL(pr_, wi), FX_MUL(pi_, wr)); }
// ai: the 3-point DFT in place, as arm64's: K3 the splat of sqrt 3 / 2, KH of 1 / 2
#define DFT3F(ar, ai, br, bi, cr, ci, K3, KH) { \
  const FX_V t1r = FX_ADD(br, cr), t1i = FX_ADD(bi, ci), t2r = FX_SUB(br, cr), t2i = FX_SUB(bi, ci); \
  const FX_V hr = FX_SUB(ar, FX_MUL(t1r, KH)), hi = FX_SUB(ai, FX_MUL(t1i, KH)), sr = FX_MUL(t2i, K3), si = FX_MUL(t2r, K3); \
  ar = FX_ADD(ar, t1r); ai = FX_ADD(ai, t1i); \
  br = FX_ADD(hr, sr); bi = FX_SUB(hi, si); cr = FX_SUB(hr, sr); ci = FX_ADD(hi, si); }
// ai: the 4-point DFT in place, as arm64's
#define DFT4F(ar, ai, br, bi, cr, ci, dr, di) { \
  const FX_V t0r = FX_ADD(ar, cr), t0i = FX_ADD(ai, ci), t1r = FX_SUB(ar, cr), t1i = FX_SUB(ai, ci); \
  const FX_V t2r = FX_ADD(br, dr), t2i = FX_ADD(bi, di), t3r = FX_SUB(br, dr), t3i = FX_SUB(bi, di); \
  ar = FX_ADD(t0r, t2r); ai = FX_ADD(t0i, t2i); cr = FX_SUB(t0r, t2r); ci = FX_SUB(t0i, t2i); \
  br = FX_ADD(t1r, t3i); bi = FX_SUB(t1i, t3r); dr = FX_SUB(t1r, t3i); di = FX_ADD(t1i, t3r); }
FX_FN void FX_N(rows2f)(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const FX_V wr = FX_SET1(w[0]), wi = FX_SET1(w[1]);
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V ar = FX_LD(re + x), ai = FX_LD(im + x), br = FX_LD(re + st + x), bi = FX_LD(im + st + x);
    if (!unit && !dif) TWS(br, bi, wr, wi)
    FX_V yr = FX_SUB(ar, br), yi = FX_SUB(ai, bi);
    if (!unit && dif) TWS(yr, yi, wr, wi)
    FX_ST(re + x, FX_ADD(ar, br)); FX_ST(im + x, FX_ADD(ai, bi)); FX_ST(re + st + x, yr); FX_ST(im + st + x, yi);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 2, w, unit, dif), rows2f(re + x, im + x, st, cols - x, w, unit, dif));
}
FX_FN void FX_N(rows3f)(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const FX_V K3 = FX_SET1(0.86602540378443864676f), KH = FX_SET1(0.5f);
  const FX_V w1r = FX_SET1(w[0]), w1i = FX_SET1(w[1]), w2r = FX_SET1(w[2]), w2i = FX_SET1(w[3]);
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V ar = FX_LD(re + x), ai = FX_LD(im + x), br = FX_LD(re + st + x), bi = FX_LD(im + st + x), cr = FX_LD(re + 2 * st + x), ci = FX_LD(im + 2 * st + x);
    if (!unit && !dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) }
    DFT3F(ar, ai, br, bi, cr, ci, K3, KH)
    if (!unit && dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) }
    FX_ST(re + x, ar); FX_ST(im + x, ai); FX_ST(re + st + x, br); FX_ST(im + st + x, bi); FX_ST(re + 2 * st + x, cr); FX_ST(im + 2 * st + x, ci);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 3, w, unit, dif), rows3f(re + x, im + x, st, cols - x, w, unit, dif));
}
FX_FN void FX_N(rows4f)(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const FX_V w1r = FX_SET1(w[0]), w1i = FX_SET1(w[1]), w2r = FX_SET1(w[2]), w2i = FX_SET1(w[3]), w3r = FX_SET1(w[4]), w3i = FX_SET1(w[5]);
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V ar = FX_LD(re + x), ai = FX_LD(im + x), br = FX_LD(re + st + x), bi = FX_LD(im + st + x);
    FX_V cr = FX_LD(re + 2 * st + x), ci = FX_LD(im + 2 * st + x), dr = FX_LD(re + 3 * st + x), di = FX_LD(im + 3 * st + x);
    if (!unit && !dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i) }
    DFT4F(ar, ai, br, bi, cr, ci, dr, di)
    if (!unit && dif) { TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i) }
    FX_ST(re + x, ar); FX_ST(im + x, ai); FX_ST(re + 2 * st + x, cr); FX_ST(im + 2 * st + x, ci);
    FX_ST(re + st + x, br); FX_ST(im + st + x, bi); FX_ST(re + 3 * st + x, dr); FX_ST(im + 3 * st + x, di);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 4, w, unit, dif), rows4f(re + x, im + x, st, cols - x, w, unit, dif));
}
// ai: rows4f in time writing only the outputs in need (bit j: output j), as arm64's rows4k
FX_FN void FX_N(rows4k)(float *re, float *im, size_t st, int cols, const float *w, int need) {
  const FX_V w1r = FX_SET1(w[0]), w1i = FX_SET1(w[1]), w2r = FX_SET1(w[2]), w2i = FX_SET1(w[3]), w3r = FX_SET1(w[4]), w3i = FX_SET1(w[5]);
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V ar = FX_LD(re + x), ai = FX_LD(im + x), br = FX_LD(re + st + x), bi = FX_LD(im + st + x);
    FX_V cr = FX_LD(re + 2 * st + x), ci = FX_LD(im + 2 * st + x), dr = FX_LD(re + 3 * st + x), di = FX_LD(im + 3 * st + x);
    TWS(br, bi, w1r, w1i) TWS(cr, ci, w2r, w2i) TWS(dr, di, w3r, w3i)
    if (need & 5) {
      const FX_V t0r = FX_ADD(ar, cr), t0i = FX_ADD(ai, ci), t2r = FX_ADD(br, dr), t2i = FX_ADD(bi, di);
      if (need & 1) { FX_ST(re + x, FX_ADD(t0r, t2r)); FX_ST(im + x, FX_ADD(t0i, t2i)); }
      if (need & 4) { FX_ST(re + 2 * st + x, FX_SUB(t0r, t2r)); FX_ST(im + 2 * st + x, FX_SUB(t0i, t2i)); }
    }
    if (need & 10) {
      const FX_V t1r = FX_SUB(ar, cr), t1i = FX_SUB(ai, ci), t3r = FX_SUB(br, dr), t3i = FX_SUB(bi, di);
      if (need & 2) { FX_ST(re + st + x, FX_ADD(t1r, t3i)); FX_ST(im + st + x, FX_SUB(t1i, t3r)); }
      if (need & 8) { FX_ST(re + 3 * st + x, FX_SUB(t1r, t3i)); FX_ST(im + 3 * st + x, FX_ADD(t1i, t3r)); }
    }
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 4, w, 0, 0), rows4k(re + x, im + x, st, cols - x, w, need));
}
// ai: radix 5, arm64's rows5f: m1 = (a + c1 t1) + c2 t2, m2 = (a + c2 t1) + c1 t2, n1 = s1 t3 + s2 t4, n2 = s2 t3 - s1 t4
FX_FN void FX_N(rows5f)(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const FX_V c1 = FX_SET1(0.30901699437494742410f), c2 = FX_SET1(-0.80901699437494742410f), s1 = FX_SET1(0.95105651629515357212f), s2 = FX_SET1(0.58778525229247312917f);
  FX_V wr[4], wi[4];
  for (int j = 0; j < 4; j++) { wr[j] = FX_SET1(w[2 * j]); wi[j] = FX_SET1(w[2 * j + 1]); }
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V qr[4], qi[4];
    for (int j = 0; j < 4; j++) { qr[j] = FX_LD(re + (j + 1) * st + x); qi[j] = FX_LD(im + (j + 1) * st + x); }
    if (!unit && !dif) for (int j = 0; j < 4; j++) TWS(qr[j], qi[j], wr[j], wi[j])
    const FX_V ar = FX_LD(re + x), ai = FX_LD(im + x);
    const FX_V t1r = FX_ADD(qr[0], qr[3]), t1i = FX_ADD(qi[0], qi[3]), t2r = FX_ADD(qr[1], qr[2]), t2i = FX_ADD(qi[1], qi[2]);
    const FX_V t3r = FX_SUB(qr[0], qr[3]), t3i = FX_SUB(qi[0], qi[3]), t4r = FX_SUB(qr[1], qr[2]), t4i = FX_SUB(qi[1], qi[2]);
    const FX_V m1r = FX_ADD(FX_ADD(ar, FX_MUL(t1r, c1)), FX_MUL(t2r, c2)), m1i = FX_ADD(FX_ADD(ai, FX_MUL(t1i, c1)), FX_MUL(t2i, c2));
    const FX_V m2r = FX_ADD(FX_ADD(ar, FX_MUL(t1r, c2)), FX_MUL(t2r, c1)), m2i = FX_ADD(FX_ADD(ai, FX_MUL(t1i, c2)), FX_MUL(t2i, c1));
    const FX_V n1r = FX_ADD(FX_MUL(t3r, s1), FX_MUL(t4r, s2)), n1i = FX_ADD(FX_MUL(t3i, s1), FX_MUL(t4i, s2));
    const FX_V n2r = FX_SUB(FX_MUL(t3r, s2), FX_MUL(t4r, s1)), n2i = FX_SUB(FX_MUL(t3i, s2), FX_MUL(t4i, s1));
    FX_V y1r = FX_ADD(m1r, n1i), y1i = FX_SUB(m1i, n1r), y4r = FX_SUB(m1r, n1i), y4i = FX_ADD(m1i, n1r);
    FX_V y2r = FX_ADD(m2r, n2i), y2i = FX_SUB(m2i, n2r), y3r = FX_SUB(m2r, n2i), y3i = FX_ADD(m2i, n2r);
    if (!unit && dif) { TWS(y1r, y1i, wr[0], wi[0]) TWS(y2r, y2i, wr[1], wi[1]) TWS(y3r, y3i, wr[2], wi[2]) TWS(y4r, y4i, wr[3], wi[3]) }
    FX_ST(re + x, FX_ADD(FX_ADD(ar, t1r), t2r)); FX_ST(im + x, FX_ADD(FX_ADD(ai, t1i), t2i));
    FX_ST(re + st + x, y1r); FX_ST(im + st + x, y1i); FX_ST(re + 4 * st + x, y4r); FX_ST(im + 4 * st + x, y4i);
    FX_ST(re + 2 * st + x, y2r); FX_ST(im + 2 * st + x, y2i); FX_ST(re + 3 * st + x, y3r); FX_ST(im + 3 * st + x, y3i);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 5, w, unit, dif), rows5f(re + x, im + x, st, cols - x, w, unit, dif));
}
// ai: radix 6 by Good-Thomas over 2 x 3, arm64's rows6f
FX_FN void FX_N(rows6f)(float *re, float *im, size_t st, int cols, const float *w, int unit, int dif) {
  const FX_V K3 = FX_SET1(0.86602540378443864676f), KH = FX_SET1(0.5f);
  FX_V wr[5], wi[5];
  for (int j = 0; j < 5; j++) { wr[j] = FX_SET1(w[2 * j]); wi[j] = FX_SET1(w[2 * j + 1]); }
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V x0r = FX_LD(re + x), x0i = FX_LD(im + x), x1r = FX_LD(re + st + x), x1i = FX_LD(im + st + x);
    FX_V x2r = FX_LD(re + 2 * st + x), x2i = FX_LD(im + 2 * st + x), x3r = FX_LD(re + 3 * st + x), x3i = FX_LD(im + 3 * st + x);
    FX_V x4r = FX_LD(re + 4 * st + x), x4i = FX_LD(im + 4 * st + x), x5r = FX_LD(re + 5 * st + x), x5i = FX_LD(im + 5 * st + x);
    if (!unit && !dif) { TWS(x1r, x1i, wr[0], wi[0]) TWS(x2r, x2i, wr[1], wi[1]) TWS(x3r, x3i, wr[2], wi[2]) TWS(x4r, x4i, wr[3], wi[3]) TWS(x5r, x5i, wr[4], wi[4]) }
    FX_V u0r = FX_ADD(x0r, x3r), u0i = FX_ADD(x0i, x3i), v0r = FX_SUB(x0r, x3r), v0i = FX_SUB(x0i, x3i);
    FX_V u1r = FX_ADD(x2r, x5r), u1i = FX_ADD(x2i, x5i), v1r = FX_SUB(x2r, x5r), v1i = FX_SUB(x2i, x5i);
    FX_V u2r = FX_ADD(x4r, x1r), u2i = FX_ADD(x4i, x1i), v2r = FX_SUB(x4r, x1r), v2i = FX_SUB(x4i, x1i);
    DFT3F(u0r, u0i, u1r, u1i, u2r, u2i, K3, KH)   // ai: X0, X4, X2
    DFT3F(v0r, v0i, v1r, v1i, v2r, v2i, K3, KH)   // ai: X3, X1, X5
    if (!unit && dif) { TWS(v1r, v1i, wr[0], wi[0]) TWS(u2r, u2i, wr[1], wi[1]) TWS(v0r, v0i, wr[2], wi[2]) TWS(u1r, u1i, wr[3], wi[3]) TWS(v2r, v2i, wr[4], wi[4]) }
    FX_ST(re + x, u0r); FX_ST(im + x, u0i); FX_ST(re + 4 * st + x, u1r); FX_ST(im + 4 * st + x, u1i); FX_ST(re + 2 * st + x, u2r); FX_ST(im + 2 * st + x, u2i);
    FX_ST(re + 3 * st + x, v0r); FX_ST(im + 3 * st + x, v0i); FX_ST(re + st + x, v1r); FX_ST(im + st + x, v1i); FX_ST(re + 5 * st + x, v2r); FX_ST(im + 5 * st + x, v2i);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 6, w, unit, dif), rows6f(re + x, im + x, st, cols - x, w, unit, dif));
}
// ai: radix 12 without twiddles by Good-Thomas over 4 x 3, arm64's rows12u
FX_FN void FX_N(rows12u)(float *re, float *im, size_t st, int cols) {
  const FX_V K3 = FX_SET1(0.86602540378443864676f), KH = FX_SET1(0.5f);
  int x = 0;
  for (; x + FX_W <= cols; x += FX_W) {
    FX_V a0r = FX_LD(re + x), a0i = FX_LD(im + x), a1r = FX_LD(re + 3 * st + x), a1i = FX_LD(im + 3 * st + x);
    FX_V a2r = FX_LD(re + 6 * st + x), a2i = FX_LD(im + 6 * st + x), a3r = FX_LD(re + 9 * st + x), a3i = FX_LD(im + 9 * st + x);
    DFT4F(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i)
    FX_V b0r = FX_LD(re + 4 * st + x), b0i = FX_LD(im + 4 * st + x), b1r = FX_LD(re + 7 * st + x), b1i = FX_LD(im + 7 * st + x);
    FX_V b2r = FX_LD(re + 10 * st + x), b2i = FX_LD(im + 10 * st + x), b3r = FX_LD(re + st + x), b3i = FX_LD(im + st + x);
    DFT4F(b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i)
    FX_V c0r = FX_LD(re + 8 * st + x), c0i = FX_LD(im + 8 * st + x), c1r = FX_LD(re + 11 * st + x), c1i = FX_LD(im + 11 * st + x);
    FX_V c2r = FX_LD(re + 2 * st + x), c2i = FX_LD(im + 2 * st + x), c3r = FX_LD(re + 5 * st + x), c3i = FX_LD(im + 5 * st + x);
    DFT4F(c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i)
    DFT3F(a0r, a0i, b0r, b0i, c0r, c0i, K3, KH)   // ai: X0, X4, X8
    FX_ST(re + x, a0r); FX_ST(im + x, a0i); FX_ST(re + 4 * st + x, b0r); FX_ST(im + 4 * st + x, b0i); FX_ST(re + 8 * st + x, c0r); FX_ST(im + 8 * st + x, c0i);
    DFT3F(a1r, a1i, b1r, b1i, c1r, c1i, K3, KH)   // ai: X9, X1, X5
    FX_ST(re + 9 * st + x, a1r); FX_ST(im + 9 * st + x, a1i); FX_ST(re + st + x, b1r); FX_ST(im + st + x, b1i); FX_ST(re + 5 * st + x, c1r); FX_ST(im + 5 * st + x, c1i);
    DFT3F(a2r, a2i, b2r, b2i, c2r, c2i, K3, KH)   // ai: X6, X10, X2
    FX_ST(re + 6 * st + x, a2r); FX_ST(im + 6 * st + x, a2i); FX_ST(re + 10 * st + x, b2r); FX_ST(im + 10 * st + x, b2i); FX_ST(re + 2 * st + x, c2r); FX_ST(im + 2 * st + x, c2i);
    DFT3F(a3r, a3i, b3r, b3i, c3r, c3i, K3, KH)   // ai: X3, X7, X11
    FX_ST(re + 3 * st + x, a3r); FX_ST(im + 3 * st + x, a3i); FX_ST(re + 7 * st + x, b3r); FX_ST(im + 7 * st + x, b3i); FX_ST(re + 11 * st + x, c3r); FX_ST(im + 11 * st + x, c3i);
  }
  if (x < cols) FX_TAIL(tail_dec(re, im, st, x, cols, 12, 0, 1, 0), rows12u(re + x, im + x, st, cols - x));
}

#undef TWS
#undef DFT3F
#undef DFT4F
