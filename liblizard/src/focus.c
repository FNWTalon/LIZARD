#include "focus.h"
#include "internal.h"
#include "fft.h"
#include "rs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Everything a call writes, allocated once: a frame costs no malloc. Behind a pointer so a const focus_t can use it.
struct focus_ws {
  float *tre, *tim;                 // the spectrum's upper half-plane: vblocks blocks of n rows (u) by bw columns (v). pos indexes it.
  float *wre, *wim;                 // one strip of the picture, n rows by FOCUS_STRIP columns
  float *cx, *qx;                   // the detrend's basis vectors: c = (2 i + 1 - n) / n, and c^2 less its mean
  float *x1r, *x1i, *x2r, *x2i;     // their transforms
  double sx2, sq2;                  // their squared norms
  float *amp;                       // per sub-channel, the magnitude its coefficients are sent at (focus.h, tilt)
  double power;                     // the sum of their squares over every coefficient
  int8_t *llr; uint8_t *bits, *data;
  // The bit map (focus.h FOCUS_BITMAP): the whitening bit of every slot of the frame, one a byte in pos order, two
  // slots a coefficient; each tier's slot to codeword bit permutation; and the slot-order and codeword-order
  // buffers the map is applied between.
  uint8_t *white, *slot; int *perm[FOCUS_TIERS]; int8_t *llr2;
  // Measurement hooks (focus_debug): the bit pair sent on every coefficient by the last encode, and what the last
  // decode found there, so an experiment can put a number on each ring (scripts/exp/focus_capacity.mjs).
  int8_t *dbg_sym; float *dbg_coef;
  // The encoder's picture before it is resampled onto the ring's grid, the rows half done, and the resampler's taps:
  // for each of the rs_q pixels across the picture and its guard interval, the first of six picture samples and their
  // weights. Made on the first encode, so a receiver never holds them.
  float *rs_pic, *rs_tmp, *rs_w, *rs_hw; int *rs_i0, *rs_hb, rs_q, rs_pad, rs_row;
  // Per block of the last decode (FOCUS_LDPC): iterations used (-1 never converged, -2 declined, 0 not tried) and
  // the information its sub-channels were estimated to hold before decoding, bits per coded bit.
  int8_t *blk_its; float *blk_est;
  // ai: The pilots the last decode read (focus_pilot): each block's r, NaN where it read none.
  float *blk_pilot;
  // ai: The grid's shift the last decode read off the pilots and turned back (pilot_align): samples along u and v,
  // ai: 0 where it read none it could tell from nothing; and cos, sin round the circle, ALIGN_TURNS steps.
  float align[2];
  float *rot;
  // The format word of the last decode (fmt.h): its soft values, what they said, and whether they said anything.
  float *fmt_q; int fmt_cells, fmt_bytes, fmt_rx_ok; ob_fmt_t fmt_rx;
  // focus_acquire's answer, waiting for focus_finish. The image is held by pointer, so the caller's pixels must
  // outlive the pair; the rig's do, since a worker owns the frame for the whole decode.
  image_t hold_im; ob_reg_t hold_reg; int hold_valid;
};

extern uint32_t ob_debug_hash;   // the host's (wasm.c, core/cpu/codec.c): armed by a test, which then reads back a hash of the floats and soft values behind the decision
static uint32_t fnv(uint32_t h, const void *p, size_t n) { const uint8_t *b = p; for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u; return h; }

// Information a soft QPSK axis carries at signal to noise g (amplitude squared over noise variance, per axis), bits:
// ten Brink's J at sigma = 2 sqrt g. What a sub-channel is worth to the code before a single iteration is spent.
static double axis_info(double g) { return g > 0 ? pow(1 - pow(2, -0.3073 * pow(2 * sqrt(g), 1.7870)), 1.1064) : 0; }

// Two picture columns share one transform (turn_pack, unpack_turn), so a strip is transformed half as wide.
#define STRIP_HALF (FOCUS_STRIP / 2)

#define FOCUS_DECLINE 0.2f
#define FOCUS_STALL_IT 9
#define FOCUS_STALL_RATIO 0.95f

static uint16_t crc16(const uint8_t *p, int n) {
  uint16_t c = 0xffff;
  for (int i = 0; i < n; i++) { c ^= (uint16_t)(p[i] << 8); for (int k = 0; k < 8; k++) c = (uint16_t)(c & 0x8000 ? (c << 1) ^ 0x1021 : c << 1); }
  return c;
}

typedef struct { int r2, u, v; } coef_t;

// Bits block b is sent on (two a coefficient), and the most any block of the frame is.
static int block_bits(const focus_t *f, int b) { return f->mode == FOCUS_LDPC ? f->tier[f->block_tier[b]].subs * FOCUS_SUB * 2 : FOCUS_SUB * 2; }
static int max_bits(const focus_t *f) { int m = FOCUS_SUB * 2; for (int t = 0; t < f->tiers; t++) if (f->tier[t].subs * FOCUS_SUB * 2 > m) m = f->tier[t].subs * FOCUS_SUB * 2; return m; }
static int max_k(const focus_t *f) { int m = 0; for (int t = 0; t < f->tiers; t++) if (f->code[t].k > m) m = f->code[t].k; return m; }
// ai: the longest codeword among the tiers: more than its block's slots where a code has bits never sent (ldpc.h np)
static int max_n(const focus_t *f) { int m = 0; for (int t = 0; t < f->tiers; t++) if (f->code[t].n > m) m = f->code[t].n; return m; }

// ---------------------------------------------------------------- the bit map (focus.h FOCUS_BITMAP)

// The whitening sequence: PRBS-23, x^23 + x^18 + 1, from the all-ones state, one bit a slot of the frame in pos
// order. It depends on nothing but the slot, so a format's frame is a prefix of a bigger one's at the same n and a
// receiver standing at the top format still reads every lower one.
static void whiten_fill(uint8_t *out, int n) {
  uint32_t s = 0x7fffff;
  for (int i = 0; i < n; i++) { const uint32_t b = ((s >> 22) ^ (s >> 17)) & 1; s = ((s << 1) | b) & 0x7fffff; out[i] = (uint8_t)b; }
}
static int gcd(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }
// Slot i of a block carries sent bit perm[i], which is codeword bit np + perm[i] (np the bits the code never sends,
// 93 at 7/8, 0 elsewhere): n here is the sent length, k the sent data bits; the codeword is systematic, data then parity.
static void perm_fill(int *perm, int n, int k, int mode) {
  const int m = n - k;
  if (mode == FOCUS_BITMAP_SPREAD) {
    // A parity bit every n / m slots, evenly (Bresenham), data in order in between: at 3/4 every fourth slot.
    for (int i = 0, dc = 0, pc = 0; i < n; i++) perm[i] = (long)(i + 1) * m / n > (long)i * m / n ? k + pc++ : dc++;
  } else if (mode == FOCUS_BITMAP_LINEAR) {
    // Slot i takes bit i P mod n, P near n over the golden ratio and prime to n: every run of the codeword, a
    // circulant's z bits (106 at 3/4) or the parity chain, lands spread over the whole ring.
    int P = (int)(0.3819660113 * n + 0.5);
    while (gcd(P, n) != 1) P++;
    for (int i = 0; i < n; i++) perm[i] = (int)((long)i * P % n);
  } else if (mode == FOCUS_BITMAP_INNER) {
    for (int i = 0; i < n; i++) perm[i] = i < m ? k + i : i - m;   // parity on the innermost slots, data outside it
  } else for (int i = 0; i < n; i++) perm[i] = i;
}
int focus_bitmap(focus_t *f, int mode) {
  struct focus_ws *w = f->ws;
  if (!w || mode < FOCUS_BITMAP_NONE || mode > FOCUS_BITMAP_INNER) return -1;
  for (int t = 0; t < f->tiers; t++) {
    if (!w->perm[t] && !(w->perm[t] = malloc((size_t)f->code[t].n * sizeof(int)))) return -1;
    perm_fill(w->perm[t], f->code[t].nt, f->code[t].k - f->code[t].np, mode);
  }
  f->bitmap = mode;
  return 0;
}
// A block's codeword onto its slots: slot i = codeword bit perm[i], whitened; the slots past the codeword carry a
// ai: bit of the painted picture's count (the pilots, focus_parity: bit 0 on an even block b, bit 1 on an odd one;
// ai: the count 0 before 2026-09-30, one bit for every block until 2026-10-01), whitened too. first is the block's
// first coefficient.
static void map_out(const focus_t *f, const struct focus_ws *w, int t, int b, int first, int nb, const uint8_t *cw, uint8_t *slot) {
  const int nt = f->code[t].nt, np = f->code[t].np, *perm = w->perm[t];
  const uint8_t q = (uint8_t)((f->parity >> (b & 1)) & 1);
  const uint8_t *wh = w->white + 2 * (size_t)first;
  for (int i = 0; i < nb; i++) slot[i] = (uint8_t)((i < nt ? cw[np + perm[i]] : q) ^ wh[i]);
}
// The soft values the other way: slot order in, codeword order out, the whitening taken back off by sign; the bits
// never sent come in as 0, nothing known.
static void map_in(const focus_t *f, const struct focus_ws *w, int t, int first, const int8_t *slot, int8_t *cw) {
  const int nt = f->code[t].nt, np = f->code[t].np, *perm = w->perm[t];
  const uint8_t *wh = w->white + 2 * (size_t)first;
  for (int i = 0; i < np; i++) cw[i] = 0;
  for (int i = 0; i < nt; i++) cw[np + perm[i]] = wh[i] ? (int8_t)-slot[i] : slot[i];
}

// ai: The grid's shift from the pilots (2026-10-01; the format paints them whatever the phone, so a decoder may steer
// ai: by them). A grid that sits (dx, dy) samples off turns coefficient (u, v) by
// ai: 2 pi (u dx + v dy) / n, which costs the outer sub-channels their soft values (0.2 of a sample is 0.3 rad at
// ai: a radius of n / 4). Every block's tail holds 16 coefficients whose symbols are known up to one sign, the
// ai: block's bit of the painted count (SPEC 7.3), at the top of its last sub-channel, so over a frame's blocks
// ai: they sit at every radius the data does. For each, z is what was read times the conjugate of what was sent,
// ai: with the block's sign taken from its own pilots' sum: its imaginary part over its real one is the turn. The
// ai: shift is the least squares fit of zi = k zr (u dx + v dy), k = 2 pi / n (each coefficient weighted by its own
// ai: zr, so one that noise turned far counts little), fitted twice, the second time about the first's answer.
// ai: It is taken only where the pilots tell it from no shift at all: d' M d over the fit's residual variance (M
// ai: the fit's matrix) above ALIGN_CHI, a chance of one in a thousand for two numbers that are noise; a frame of
// ai: few blocks, whose pilots say little, is left as it was read. Returns the pilots it used; d the shift or 0, 0.
#define ALIGN_TURNS 4096
#define ALIGN_CHI 13.8
static inline void turned(const focus_t *f, const struct focus_ws *w, int p, float dx, float dy, float *yr, float *yi) {
  const int n = f->n, bw = f->bw, half = n / 2, nbw = n * bw;
  // (u, v) back out of the packed index init() built: pos = (v / bw) n bw + ((u + n) % n) bw + v % bw.
  const int rem = p % nbw, vv = (p / nbw) * bw + rem % bw, uw = rem / bw, uu = uw < half ? uw : uw - n;
  const unsigned k = (unsigned)(int)nearbyintf(((float)uu * dx + (float)vv * dy) * ((float)ALIGN_TURNS / (float)n)) & (ALIGN_TURNS - 1);
  const float c = w->rot[2 * k], sn = w->rot[2 * k + 1], r = w->tre[p], i = w->tim[p];
  *yr = r * c + i * sn; *yi = i * c - r * sn;
}
static int pilot_align(const focus_t *f, struct focus_ws *w, float d[2]) {
  const int n = f->n, bw = f->bw, half = n / 2, nbw = n * bw;
  const double k = 2.0 * 3.14159265358979323846 / n;
  double dx = 0, dy = 0, a = 0, b = 0, c = 0, p = 0, q = 0, yy = 0;
  int used = 0;
  for (int it = 0; it < 2; it++) {
    a = b = c = p = q = yy = 0; used = 0;
    for (int blk = 0; blk < f->blocks; blk++) {
      const ldpc_t *code = &f->code[f->block_tier[blk]];
      const int nb = block_bits(f, blk), first = f->block_sub[blk] * FOCUS_SUB;
      const uint8_t *wh = w->white + 2 * (size_t)first;
      if (code->nt >= nb) continue;
      // ai: the block's sign: its pilots' own sum, read at the shift so far
      double sum = 0;
      for (int s = code->nt / 2; s < nb / 2; s++) {
        float yr, yi;
        turned(f, w, f->pos[first + s], (float)dx, (float)dy, &yr, &yi);
        sum += (wh[2 * s] ? -1.0 : 1.0) * yr + (wh[2 * s + 1] ? -1.0 : 1.0) * yi;
      }
      const double sg = sum < 0 ? -1.0 : 1.0;
      for (int s = code->nt / 2; s < nb / 2; s++) {
        const int pp = f->pos[first + s];
        const int rem = pp % nbw, vv = (pp / nbw) * bw + rem % bw, uw = rem / bw, uu = uw < half ? uw : uw - n;
        float yr, yi;
        turned(f, w, pp, (float)dx, (float)dy, &yr, &yi);
        const double xr = wh[2 * s] ? -sg : sg, xi = wh[2 * s + 1] ? -sg : sg;
        const double zr = yr * xr + yi * xi, zi = yi * xr - yr * xi;
        const double cu = k * zr * uu, cv = k * zr * vv;
        a += cu * cu; b += cu * cv; c += cv * cv; p += cu * zi; q += cv * zi; yy += zi * zi;
        used++;
      }
    }
    const double det = a * c - b * b;
    if (used < 8 || !(det > 0)) { d[0] = d[1] = 0; return used; }
    dx += (p * c - q * b) / det; dy += (a * q - b * p) / det;
  }
  // ai: the last fit's residual variance (about its own answer), and the whole shift against it
  const double det = a * c - b * b, ex = (p * c - q * b) / det, ey = (a * q - b * p) / det;
  const double s2 = (yy - ex * p - ey * q) / (used - 2);
  const double chi = s2 > 0 ? (dx * dx * a + 2 * dx * dy * b + dy * dy * c) / s2 : 0;
  if (!(chi > ALIGN_CHI) || !(dx == dx) || !(dy == dy)) { d[0] = d[1] = 0; return used; }
  d[0] = (float)dx; d[1] = (float)dy;
  return used;
}

static int ws_init(focus_t *f) {
  struct focus_ws *w = calloc(1, sizeof *w);
  if (!w) return -1;
  f->ws = w;
  int n = f->n;
  size_t tn = (size_t)f->vblocks * n * f->bw, sn = (size_t)n * FOCUS_STRIP, nb = (size_t)max_bits(f);
  w->tre = malloc(tn * sizeof(float)); w->tim = malloc(tn * sizeof(float));
  w->wre = malloc(sn * sizeof(float)); w->wim = malloc(sn * sizeof(float));
  float *tab = malloc((size_t)6 * n * sizeof(float));
  const size_t nc = (size_t)max_n(f) > nb ? (size_t)max_n(f) : nb;   // ai: a codeword's bits, or a block's slots, whichever is more
  w->llr = malloc(nc); w->bits = malloc(nc); w->data = malloc((size_t)max_k(f) + 8);
  w->slot = malloc(nc); w->llr2 = malloc(nc); w->white = malloc((size_t)f->subch * FOCUS_SUB * 2);
  w->amp = malloc((size_t)f->subch * sizeof(float));
  w->blk_its = calloc((size_t)f->blocks, 1); w->blk_est = calloc((size_t)f->blocks, sizeof(float)); w->blk_pilot = calloc((size_t)f->blocks, sizeof(float));
  w->rot = malloc((size_t)2 * ALIGN_TURNS * sizeof(float));
  if (w->rot) for (int i = 0; i < ALIGN_TURNS; i++) { const double th = 2.0 * 3.14159265358979323846 * i / ALIGN_TURNS; w->rot[2 * i] = (float)cos(th); w->rot[2 * i + 1] = (float)sin(th); }
  const int side_mod = f->frame.w < f->frame.h ? f->frame.w : f->frame.h;
  w->fmt_cells = ob_fmt_side_cells(side_mod, f->frame.reserve);   // 0 where the frame is too small to hold a word at all
  w->fmt_bytes = ob_fmt_bytes(side_mod, f->frame.reserve);
  w->fmt_q = w->fmt_cells ? malloc((size_t)4 * w->fmt_cells * sizeof(float)) : 0;
  double *e = malloc((size_t)2 * n * sizeof(double));
  if (!w->tre || !w->tim || !w->wre || !w->wim || !tab || !w->llr || !w->bits || !w->data || !e || !w->amp || !w->blk_its || !w->blk_est || !w->blk_pilot || (w->fmt_cells && !w->fmt_q) || !w->slot || !w->llr2 || !w->white) { free(tab); free(e); return -1; }
  whiten_fill(w->white, f->subch * FOCUS_SUB * 2);
  for (int j = 0; j < f->subch; j++) { w->amp[j] = f->tilt > 0 && f->subch > 1 ? (float)pow(10.0, -f->tilt * j / (20.0 * (f->subch - 1))) : 1.0f; w->power += (double)FOCUS_SUB * w->amp[j] * w->amp[j]; }
  w->cx = tab; w->qx = tab + n; w->x1r = tab + 2 * n; w->x1i = tab + 3 * n; w->x2r = tab + 4 * n; w->x2i = tab + 5 * n;
  double m2 = 0;
  for (int i = 0; i < n; i++) { double c = (2.0 * i + 1 - n) / n; m2 += c * c; e[i] = cos(-2 * 3.14159265358979323846 * i / n); e[n + i] = sin(-2 * 3.14159265358979323846 * i / n); }
  w->sx2 = m2; m2 /= n;
  for (int i = 0; i < n; i++) { double c = (2.0 * i + 1 - n) / n, q = c * c - m2; w->cx[i] = (float)c; w->qx[i] = (float)q; w->sq2 += q * q; }
  // Once, in double, straight from the definition: n^2 terms, no trigonometry in the loop.
  for (int u = 0; u < n; u++) {
    double ar = 0, ai = 0, br = 0, bi = 0;
    for (int i = 0; i < n; i++) { double c = (2.0 * i + 1 - n) / n, q = c * c - m2; int k = (u * i) % n; ar += c * e[k]; ai += c * e[n + k]; br += q * e[k]; bi += q * e[n + k]; }
    w->x1r[u] = (float)ar; w->x1i[u] = (float)ai; w->x2r[u] = (float)br; w->x2i[u] = (float)bi;
  }
  free(e);
  return 0;
}


// The word restates this focus_t rather than being handed to us, so it cannot come to disagree with what is
// painted. The display rate is the exception, being a property of the sequence and not of this frame:
// focus_fmt_fps sets it and it survives a repaint. Everything else follows from the format, and the fountain's
// header is payload. A configuration the word cannot express (a sub-channel count that is not a multiple of 8, which
// only FOCUS_RS makes) blanks the band instead: a receiver that finds nothing there knows it was told nothing,
// where one that finds a lie does not.
static void fmt_paint(focus_t *f) {
  f->fmt.version = f->subch / FOCUS_GROUP;
  const int fb = ob_fmt_bytes(f->frame.w < f->frame.h ? f->frame.w : f->frame.h, f->frame.reserve);
  if (f->subch % FOCUS_GROUP || !fb || ob_fmt_encode(&f->fmt, f->fmt_cw, fb)) memset(f->fmt_cw, 0, sizeof f->fmt_cw);
  ob_layout_set_fmt(&f->frame, f->fmt_cw);
}

const ob_fmt_t *focus_fmt_rx(const focus_t *f) { return f->ws && f->ws->fmt_rx_ok ? &f->ws->fmt_rx : 0; }

// The display rate the word states, in whole frames a second (fmt.h): the only field a caller supplies, since it
// is the only one that is not a property of this focus_t. Safe at any time, including mid-transfer: repainting
// the word moves the word's cells and nothing else (layout.c ob_layout_set_fmt), so registration is untouched,
// and it takes effect on the next encode, which paints the border from the layout as it then stands.
int focus_parity(focus_t *f, int c) {
  if (c < 0 || c > 3) return -1;
  f->parity = c;
  return 0;
}
int focus_pilot(const focus_t *f, int blocks, float r[2], float sd[2]) {
  const struct focus_ws *w = f->ws;
  if (r) r[0] = r[1] = 0;
  if (sd) sd[0] = sd[1] = 0;
  if (!w) return 0;
  if (blocks <= 0 || blocks > f->blocks) blocks = f->blocks;
  int all = 0;
  // ai: the even blocks (bit 0 of the count) and the odd ones (bit 1), each its own mean and standard error
  for (int g = 0; g < 2; g++) {
    double s = 0, s2 = 0;
    int k = 0;
    for (int b = g; b < blocks; b += 2) if (w->blk_pilot[b] == w->blk_pilot[b]) { s += w->blk_pilot[b]; s2 += (double)w->blk_pilot[b] * w->blk_pilot[b]; k++; }
    // ai: a parity with no blocks (version 1: one block, an even one) reads none: NaN, not a zero a caller would take
    // ai: for a reading of an even mix (the row and the lock test for it)
    if (!k) { if (r) r[g] = NAN; if (sd) sd[g] = NAN; continue; }
    const double mean = s / k, var = k > 1 ? (s2 - s * mean) / (k - 1) : 0;
    if (r) r[g] = (float)mean;
    if (sd) sd[g] = (float)sqrt((var > 0 ? var : 0) / k);
    all += k;
  }
  return all;
}
void focus_align(const focus_t *f, float d[2]) { const struct focus_ws *w = f->ws; d[0] = w ? w->align[0] : 0; d[1] = w ? w->align[1] : 0; }
int focus_fmt_fps(focus_t *f, int fps) {
  if (fps < 0 || fps > 255) return -1;
  f->fmt.fps = fps;
  fmt_paint(f);
  return 0;
}

static int init(focus_t *f, int n, int subch, int mode, const focus_tier_t *tier, int tiers, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border) {
  memset(f, 0, sizeof *f);
  // ai: n is 2^k or 3 x 2^k (the transform's sizes, fft.h). A 3 x 2^k also needs n / 2 to be whole 64-column strips
  // ai: (turn_pack writes rows v and n - v for v below vblocks * bw), so 384 and up: 192 would lose rows 65 to 95.
  const int two = n % 3 ? n : n / 3;
  if (n < 64 || (two & (two - 1)) || (n % 3 == 0 && n % 128) || subch < 1) return -1;
  // Past the ladder's top the version is past what the word may state (fmt.h OB_FMT_VERSION_MAX), so fmt_paint would
  // blank the band and the symbol could not describe itself. Refused. A version off the ladder below it is not this:
  // the word names it, and the sender simply does not offer it (sim/lizard_pick.mjs VERSIONS).
  if (subch > FOCUS_GROUP * OB_FMT_VERSION_MAX) return -1;
  int need = subch * FOCUS_SUB, h = n / 2, cap = 0;
  // Upper half-plane without DC and without the Nyquist row and column, which are their own
  // reflections and cannot carry a free phase.
  // ai: The coefficients in frequency order, radius squared, then v, then u, rising: the first `need` of them, by a
  // ai: counting sort on the radius over the half-plane taken row by row with u rising, which is that order within a
  // ai: radius, and no pair is equal, so the order is the one sort of it. The radii past the one that fills `need`
  // ai: are never placed. (2026-09-29: a qsort of all n h / 2 entries took most of a 1024 picture's 185 ms setup in
  // ai: wasm, musl's smoothsort; the blind receiver builds it on each worker's first word, STATUS "The ramp at the
  // ai: start". pos is the same byte for byte: the ladder's order hash in test/vectors.mjs.)
  const int rmax = 2 * (h - 1) * (h - 1);
  int *at = calloc((size_t)rmax + 1, sizeof(int));
  if (!at) return -1;
  for (int v = 0; v < h; v++) for (int u = -h + 1; u < h; u++) { if (v == 0 && u <= 0) continue; at[u * u + v * v]++; cap++; }
  if (need > cap) { free(at); return -1; }
  int cut = 0, placed = 0;   // cut: the smallest radius squared the first `need` reach; at[r2]: its first index
  for (;; cut++) { const int k = at[cut]; at[cut] = placed; placed += k; if (placed >= need) break; }
  coef_t *c = malloc((size_t)placed * sizeof(coef_t));
  if (!c) { free(at); return -1; }
  for (int v = 0; v < h; v++) for (int u = -h + 1; u < h; u++) {
    if (v == 0 && u <= 0) continue;
    const int r2 = u * u + v * v;
    if (r2 <= cut) c[at[r2]++] = (coef_t){ r2, u, v };
  }
  free(at);
  f->pos = malloc((size_t)need * sizeof(int));
  if (!f->pos) { free(c); return -1; }
  // Only the blocks of columns v that hold a used coefficient exist: nothing else is transformed or stored.
  int bw = h < FOCUS_STRIP ? h : FOCUS_STRIP, vmax = 0;
  for (int i = 0; i < need; i++) { f->pos[i] = (c[i].v / bw) * n * bw + ((c[i].u + n) % n) * bw + c[i].v % bw; if (c[i].v > vmax) vmax = c[i].v; }
  free(c);
  f->n = n; f->subch = subch; f->mode = mode; f->clip = clip > 0 ? clip : 2.0f; f->bw = bw; f->vblocks = vmax / bw + 1;
  f->tilt = tilt > 0 ? tilt : 0;   // also turns a NaN (an argument a JS caller left out) into none
  // span: frame modules across the image, which names the ring (focus.h FOCUS_RING: span = 2 B). 0 is the sender's
  // ai: default ring (FOCUS_RING_DEFAULT, the same for every picture); another ring's span puts this picture in that
  // ai: ring; any other value is an experiment's, and one out of range falls back to the 128 ring's span, or n.
  if (!span) span = 2 * FOCUS_RING[FOCUS_RING_DEFAULT];
  if (span < 32 || span > n) span = n > 256 ? 256 : n;
  f->scale = (float)n / span;
  f->pxm = (int)ceilf(f->scale - 1e-4f);   // whole pixels a module, never fewer than the picture's samples
  // corner: 0 asks for the default mark, negative asks for none, positive is a size of the caller's own.
  const int mark = corner < 0 ? 0 : corner ? corner : OB_THIN_CORNER;

  // THE RULE: the border grows OUTWARDS to hold its marks and nothing ever reaches into the picture. The picture is
  // an inverse FFT, so a hole in it is not a local cost: measured at about eight times its area in payload, and
  // half the blocks at n = 1024. A wider border costs only camera pixels a sample, which is nothing until
  // resolution is short. So the margin is whatever was asked for, or the mark's own size if that is bigger.
  //
  // A mark and the guard interval cannot share depths. The guard is the picture wrapped outward, so the picture's
  // rectangle really runs from margin - FOCUS_CP, and a mark reaching past that line is a mark inside the picture:
  // at margin 12 with a 12-module mark and a 3-module guard, each corner mark took a 3 x 3 bite out of the guard's
  // corner. So a mark pushes the border out by its own size PLUS the guard.
  int margin = border > OB_THIN ? border : OB_THIN;
  if (mark && mark + FOCUS_CP > margin) margin = mark + FOCUS_CP;
  if (edge && edge + FOCUS_CP > margin) margin = edge + FOCUS_CP;   // an edge mark lives in the border like a corner one
  f->margin = margin;
  int side = span + 2 * f->margin;
  f->px = side * f->pxm;
  ob_cfg_t cfg = { .w = side, .h = side, .thin = 1, .border = f->margin, .corner = mark, .corner_filled = corner_filled,
                   .edge = edge > 0 ? edge : 0, .track_alt = track_alt, .centre = centre };
  if (ob_layout_init(&f->frame, &cfg)) { focus_free(f); return -1; }
  fmt_paint(f);
  if (mode == FOCUS_LDPC) {
    // One payload for every block: the smallest k among the tiers, whole bytes of it. A code with more room than
    // that carries zeros there, which its decoder is told.
    int k = 1 << 30;
    f->tiers = tiers;
    for (int t = 0; t < tiers; t++) {
      f->tier[t] = tier[t];
      if (ldpc_init(&f->code[t], tier[t].subs * FOCUS_SUB * 2, tier[t].rate, 1)) { focus_free(f); return -1; }
      f->blocks += tier[t].blocks;
      if (f->code[t].k < k) k = f->code[t].k;
    }
    f->block_bytes = k / 8 - 4;
    f->block_tier = malloc((size_t)f->blocks * 3 * sizeof(int));
    if (!f->block_tier) { focus_free(f); return -1; }
    f->block_sub = f->block_tier + f->blocks; f->block_bit = f->block_sub + f->blocks;
    for (int t = 0, b = 0, sub = 0; t < tiers; t++) for (int q = 0; q < tier[t].blocks; q++, b++, sub += tier[t].subs) { f->block_tier[b] = t; f->block_sub[b] = sub; f->block_bit[b] = sub * FOCUS_SUB * 2; }
  } else { f->blocks = subch; f->block_bytes = FOCUS_FRAG - 2; }
  if (ws_init(f)) { focus_free(f); return -1; }
  if (mode == FOCUS_LDPC && focus_bitmap(f, FOCUS_BITMAP)) { focus_free(f); return -1; }
  return 0;
}

const int FOCUS_RING[FOCUS_RINGS] = { 32, 64, 96, 128 };

// ai: The rate profile under test, if any (focus_profile_set, else LIZ_PROFILE): its rates, sub-channels a block and
// ai: target percents, tier by tier inner first; prof_tiers 0 for the format's own.
static char prof_spec[48];
static int prof_parse(const char *spec, int rate[FOCUS_TIERS], int subs[FOCUS_TIERS], int pct[FOCUS_TIERS]) {
  if (!spec || !*spec) return 0;
  int n = 0;
  const char *p = spec;
  for (; *p && *p != ':'; p++) {
    if (n == FOCUS_TIERS) return -1;
    const int r = *p - '0', s = r == 6 ? 7 : r == 4 ? 8 : r == 3 ? 9 : r == 2 ? 12 : 0;
    if (!s) return -1;
    rate[n] = r; subs[n] = s; pct[n] = 0; n++;
  }
  if (n < 2 || *p != ':') return -1;
  p++;
  for (int t = 0; t < n; t++) {
    if (t == 1) continue;
    char *end;
    const long v = strtol(p, &end, 10);
    if (end == p || v < 0 || v > 100) return -1;
    pct[t] = (int)v;
    p = end;
    if (*p == '/') p++;
  }
  return *p ? -1 : n;
}
int focus_profile_set(const char *spec) {
  int rate[FOCUS_TIERS], subs[FOCUS_TIERS], pct[FOCUS_TIERS];
  const int n = prof_parse(spec, rate, subs, pct);
  if (n < 0 || (spec && strlen(spec) >= sizeof prof_spec)) return -1;
  snprintf(prof_spec, sizeof prof_spec, "%s", spec ? spec : "");
  return n;
}
// ai: the free tiers' blocks (every tier but the second), in tier order, the first of equals kept
static void prof_search(int k, int n, const int *subs, const int *pct, int subch, int used, long long cost, int *cnt, long long *best, int *keep) {
  if (k == n) {
    const int rest = subch - used;
    if (rest < subs[1] || rest % subs[1]) return;
    if (*best < 0 || cost < *best) { *best = cost; for (int t = 0; t < n; t++) keep[t] = cnt[t]; keep[1] = rest / subs[1]; }
    return;
  }
  if (k == 1) { prof_search(2, n, subs, pct, subch, used, cost, cnt, best, keep); return; }
  for (int a = 0; used + subs[k] * a <= subch - subs[1]; a++) {
    const long long d = 100LL * subs[k] * a - (long long)pct[k] * subch;
    cnt[k] = a;
    prof_search(k + 1, n, subs, pct, subch, used + subs[k] * a, cost + d * d, cnt, best, keep);
  }
}

int focus_tiers_for(int subch, focus_tier_t tier[FOCUS_TIERS]) {
  if (subch < FOCUS_GROUP || subch % FOCUS_GROUP || subch > FOCUS_GROUP * OB_FMT_VERSION_MAX) return 0;
  // ai: the format's rule is the search a profile under test takes, at the format's rates and targets: the tiers in
  // ai: order 7/8, 3/4 (filling), 2/3, 1/2, their counts least squares from the targets, the first of equals
  static const int FMT_RATE[FOCUS_TIERS] = { 6, FOCUS_RATE, 3, 2 }, FMT_SUBS[FOCUS_TIERS] = { 7, FOCUS_GROUP, 9, 12 },
                   FMT_PCT[FOCUS_TIERS] = { FOCUS_TIER_IN, 0, FOCUS_TIER_23, FOCUS_TIER_OUT };
  int rate[FOCUS_TIERS], subs[FOCUS_TIERS], pct[FOCUS_TIERS], cnt[FOCUS_TIERS] = { 0 }, keep[FOCUS_TIERS] = { 0 };
  int n = prof_parse(prof_spec[0] ? prof_spec : getenv("LIZ_PROFILE"), rate, subs, pct);
  long long best = -1;
  if (n > 0) prof_search(0, n, subs, pct, subch, 0, 0, cnt, &best, keep);
  // ai: the format's where no profile is under test, or where the one under test cannot fill the count
  if (best < 0) {
    n = FOCUS_TIERS;
    for (int q = 0; q < n; q++) { rate[q] = FMT_RATE[q]; subs[q] = FMT_SUBS[q]; pct[q] = FMT_PCT[q]; cnt[q] = keep[q] = 0; }
    prof_search(0, n, subs, pct, subch, 0, 0, cnt, &best, keep);
  }
  int t = 0;
  for (int q = 0; q < n; q++) if (keep[q]) tier[t++] = (focus_tier_t){ rate[q], keep[q], subs[q] };
  return t;
}
int focus_blocks_for(int subch) {
  focus_tier_t tier[FOCUS_TIERS];
  const int n = focus_tiers_for(subch, tier);
  int b = 0;
  for (int t = 0; t < n; t++) b += tier[t].blocks;
  return b;
}

int focus_init(focus_t *f, int n, int subch, int mode, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border) {
  if (mode == FOCUS_LDPC) {
    focus_tier_t tier[FOCUS_TIERS];
    const int tiers = focus_tiers_for(subch, tier);
    if (!tiers) { memset(f, 0, sizeof *f); return -1; }
    const int r = init(f, n, subch, mode, tier, tiers, clip, span, tilt, corner, corner_filled, centre, edge, track_alt, border);
    // ai: a block is the transfer's 473 B (xfer.h XFER_BLOCK) at every rate: a profile with no 3/4 tier (a rate family
    // ai: under test, focus_profile_set) would take its smallest code's room (476 B beside 2/3 and 1/2), which nothing
    // ai: else in the chain reads; the roomier codes carry the zeros past it, known to the decoder
    if (!r && f->block_bytes > 473) f->block_bytes = 473;
    return r;
  }
  const focus_tier_t one = { FOCUS_RATE, subch / FOCUS_GROUP, FOCUS_GROUP };
  return init(f, n, subch, mode, &one, 0, clip, span, tilt, corner, corner_filled, centre, edge, track_alt, border);
}

int focus_init_tiers(focus_t *f, int n, const focus_tier_t *tier, int tiers, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border) {
  int subch = 0;
  if (tiers < 1 || tiers > FOCUS_TIERS) { memset(f, 0, sizeof *f); return -1; }
  for (int t = 0; t < tiers; t++) { if (tier[t].blocks < 1 || tier[t].subs < 1) { memset(f, 0, sizeof *f); return -1; } subch += tier[t].blocks * tier[t].subs; }
  const int r = init(f, n, subch, FOCUS_LDPC, tier, tiers, clip, span, tilt, corner, corner_filled, centre, edge, track_alt, border);
  // ai: 473 B whatever the tiers, as focus_init (an experiment's frame is the transfer's block too); a tier list of
  // ai: roomier codes alone (7/8 only, 484 B) is no longer one size larger
  if (!r && f->block_bytes > 473) f->block_bytes = 473;
  return r;
}

void focus_block_stats(const focus_t *f, const int8_t **its, const float **est) { *its = f->ws ? f->ws->blk_its : 0; *est = f->ws ? f->ws->blk_est : 0; }

int focus_debug(focus_t *f, int on, const int8_t **sym, const float **coef) {
  struct focus_ws *w = f->ws;
  if (!w) return -1;
  size_t need = (size_t)f->subch * FOCUS_SUB;
  if (on && !w->dbg_sym) { w->dbg_sym = calloc(2 * need, 1); w->dbg_coef = calloc(2 * need, sizeof(float)); }
  if (!on) { free(w->dbg_sym); free(w->dbg_coef); w->dbg_sym = 0; w->dbg_coef = 0; }
  if (sym) *sym = w->dbg_sym;
  if (coef) *coef = w->dbg_coef;
  return on && (!w->dbg_sym || !w->dbg_coef) ? -1 : 0;
}

// The encoder's drive as the sender's page paints it: each sample a grey level, (int)(drive * 255 + 0.5) clamped to
// 0..255, as RGBA with alpha 255 (little-endian, so the bytes are R G B A), inside the margin (focus.h). lizard-web/send.mjs
// did this in a JS loop, 30 ms a frame at LIZARD-1024; here it is SIMD, and the vector and the scalar code do the
// same float operations, so the two builds agree to the byte.
// ai: The optional guard ring (a closed black ring round more white) was painted here from 2026-09-23 and deleted on
// ai: 2026-09-26; STATUS "The guard ring deleted".
void focus_paint_rgba(const focus_t *f, const float *drive, uint8_t *rgba) {
  const int side = f->px, m = FOCUS_QUIET * f->pxm, W = side + 2 * m;
  uint32_t *px = (uint32_t *)(void *)rgba;
  for (int y = 0; y < W; y++) {
    uint32_t *row = px + (size_t)y * W;
    if (y < m || y >= W - m) { for (int x = 0; x < W; x++) row[x] = 0xFFFFFFFFu; continue; }
    for (int x = 0; x < m; x++) { row[x] = 0xFFFFFFFFu; row[W - 1 - x] = 0xFFFFFFFFu; }
    const float *d = drive + (size_t)(y - m) * side;
    uint32_t *o = row + m;
    int x = 0;
#ifdef OB_SIMD
    const v128_t k255 = wasm_f32x4_splat(255.0f), half = wasm_f32x4_splat(0.5f), lo = wasm_i32x4_splat(0), hi = wasm_i32x4_splat(255);
    const v128_t grey = wasm_i32x4_splat(0x010101), alpha = wasm_i32x4_splat((int32_t)0xFF000000u);
    for (; x + 4 <= side; x += 4) {
      v128_t v = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(wasm_f32x4_mul(wasm_v128_load(d + x), k255), half));
      v = wasm_i32x4_min(wasm_i32x4_max(v, lo), hi);
      wasm_v128_store(o + x, wasm_v128_or(wasm_i32x4_mul(v, grey), alpha));
    }
#endif
    for (; x < side; x++) {
      const float t = d[x] * 255.0f + 0.5f;
      int v = t >= 2147483647.0f ? 255 : t <= -2147483648.0f ? 0 : (int)t;
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      o[x] = (uint32_t)v * 0x010101u | 0xFF000000u;
    }
  }
}

void focus_paint_grey(const focus_t *f, const float *drive, uint8_t *grey) {
  const int side = f->px, m = FOCUS_QUIET * f->pxm, W = side + 2 * m;
  for (int y = 0; y < W; y++) {
    uint8_t *row = grey + (size_t)y * W;
    if (y < m || y >= W - m) { memset(row, 0xFF, (size_t)W); continue; }
    memset(row, 0xFF, (size_t)m);
    memset(row + W - m, 0xFF, (size_t)m);
    const float *d = drive + (size_t)(y - m) * side;
    uint8_t *o = row + m;
    for (int x = 0; x < side; x++) {
      const float t = d[x] * 255.0f + 0.5f;
      int v = t >= 2147483647.0f ? 255 : t <= -2147483648.0f ? 0 : (int)t;
      v = v < 0 ? 0 : v > 255 ? 255 : v;
      o[x] = (uint8_t)v;
    }
  }
}

void focus_free(focus_t *f) {
  struct focus_ws *w = f->ws;
  if (w) { free(w->tre); free(w->tim); free(w->wre); free(w->wim); free(w->cx); free(w->llr); free(w->bits); free(w->data); free(w->amp); free(w->fmt_q); free(w->blk_its); free(w->blk_est); free(w->blk_pilot); free(w->rot); free(w->dbg_sym); free(w->dbg_coef); free(w->white); free(w->slot); free(w->llr2); free(w->rs_i0); free(w->rs_w); free(w->rs_hb); free(w->rs_hw); free(w->rs_pic); free(w->rs_tmp); for (int t = 0; t < FOCUS_TIERS; t++) free(w->perm[t]); free(w); }
  for (int t = 0; t < FOCUS_TIERS; t++) ldpc_free(&f->code[t]);
  free(f->pos); free(f->block_tier); ob_layout_free(&f->frame); memset(f, 0, sizeof *f);
}

// The strip and the half-plane array are each other turned over: the strip's rows are v and its columns x,
// a block's rows are x (u after its transform) and its columns v. This is the encoder's direction, block -> strip;
// the decoder's direction is unpack_turn below, which separates two packed columns on the way.
// block -> packed strip for the encoder, the mirror of unpack_turn below. The picture is real, so the spectrum it
// comes from is the conjugate mirror of the half-plane that was filled, and two columns can share one inverse:
// Z = H_left + i H_right inverse-transforms to left + i right, both real, with nothing to untangle at the far end.
//   H(0) = 2 Re S(0), H(v) = S(v), H(n - v) = conj(S(v)), zero where nothing was sent.
// Row v and row n - v are written from the one source row, so this replaces both the copy and the memset that used
// to zero everything above the half-plane, and writes HALF what they did: the packed strip is half as wide.
static void turn_pack(const focus_t *f, int x0) {
  struct focus_ws *w = f->ws;
  const int n = f->n, bw = f->bw, sw = FOCUS_STRIP, hw = STRIP_HALF, rows = f->vblocks * bw;
  const float *t0 = w->tre + (size_t)x0 * bw;
  for (int j = 0; j < hw; j++) {   // v = 0 is its own mirror and real: the imaginary halves of both columns are dropped
    w->wre[j] = 2.0f * t0[(size_t)j * bw]; w->wim[j] = 2.0f * t0[(size_t)(hw + j) * bw];
  }
  // The block array is contiguous in v and strided in x, so x (that is, j) goes OUTSIDE and v inside. The reads are
  // then four contiguous runs and the writes strided, which is the way round the plain copy had. Taking v outside
  // instead reads one cache line per value, 32768 lines for 128 KB, and costs more than the transform it saves.
  for (int b = 0; b < f->vblocks; b++) for (int j = 0; j < hw; j++) {
    const size_t o = ((size_t)b * n + x0 + j) * bw, p = ((size_t)b * n + x0 + hw + j) * bw;
    const float *ar = w->tre + o, *ai = w->tim + o, *br = w->tre + p, *bi = w->tim + p;
    float *zr = w->wre + j, *zi = w->wim + j;
    for (int c = (b ? 0 : 1); c < bw; c++) {   // v = 0 is done above
      const int v = b * bw + c;
      zr[(size_t)v * sw] = ar[c] - bi[c]; zi[(size_t)v * sw] = ai[c] + br[c];               // Z(v)     = S_a(v) + i S_b(v)
      zr[(size_t)(n - v) * sw] = ar[c] + bi[c]; zi[(size_t)(n - v) * sw] = br[c] - ai[c];   // Z(n - v) = conj(S_a) + i conj(S_b)
    }
  }
  // What no coefficient reached, between the half-plane and its mirror. At n = 512 one row at 192 sub-channels, 129 at 96.
  for (int v = rows; v <= n - rows; v++) {
    memset(w->wre + (size_t)v * sw, 0, (size_t)hw * sizeof(float));
    memset(w->wim + (size_t)v * sw, 0, (size_t)hw * sizeof(float));
  }
}

// One row of the picture out of the inverse: clip at the ratio the power was set for, and map to 0..1.
static void clip_row(const float *src, float *dst, int cols, float lim, float sc) {
  int x = 0;
#ifdef OB_SIMD
  const v128_t vl = wasm_f32x4_splat(lim), vn = wasm_f32x4_splat(-lim), vs = wasm_f32x4_splat(sc), vh = wasm_f32x4_splat(0.5f);
  for (; x + 4 <= cols; x += 4) wasm_v128_store(dst + x, wasm_f32x4_add(vh, wasm_f32x4_mul(wasm_f32x4_pmax(wasm_f32x4_pmin(wasm_v128_load(src + x), vl), vn), vs)));
#endif
  for (; x < cols; x++) { float v = src[x]; v = v > lim ? lim : v; v = v < -lim ? -lim : v; dst[x] = 0.5f + v * sc; }
}

// The strip's columns are REAL: the sampler writes luminance and nothing else, and the imaginary half used to be
// memset to zero before every transform. So two picture columns share one transform, as a + i b, and are separated
// afterwards by the conjugate symmetry a real input forces on the result. That halves the y pass, which is 8 of
// the 11 to 12 transforms a decode.
//
// The pairing costs nothing at all, because the kernel takes a pitch: the strip's LEFT half is handed to it as the
// real input and its RIGHT half as the imaginary one, both at the strip's own pitch, so picture column x0 + j
// travels with column x0 + hw + j and not one value is moved. Deinterleaving adjacent columns instead needed a
// pass of its own and gave most of the saving straight back. It also leaves wim untouched by a decode, so the
// live set is the 128 KB strip rather than 256 KB, which is what matters on a phone whose L2 the pair does not fit.
// strip -> block for that transform. Row v holds Z = A + i B, with A the spectrum of the left column and B of the
// right one. A real input makes each symmetric about zero frequency, A(n - v) = conj(A(v)), and that separates them:
//   conj(Z(n - v)) = A(v) - i B(v), so A = (Z(v) + conj(Z(n - v))) / 2 and B = -i (Z(v) - conj(Z(n - v))) / 2.
// v = 0 (and the Nyquist row, which no configuration reaches) is its own partner, and the formula hands back a real
// A and B there by itself, which is what a real column's zero frequency is.
// Only v < vblocks * bw is ever wanted, but the partner row n - v lies in the half the old code threw away, so this
// runs before anything is discarded.
static void unpack_turn(const focus_t *f, int x0, const float *z0) {
  struct focus_ws *w = f->ws;
  const int n = f->n, bw = f->bw, sw = FOCUS_STRIP, hw = STRIP_HALF;
  for (int v = 0; v < f->vblocks * bw; v++) {
    const int b = v / bw, c = v % bw, vm = v ? n - v : 0;
    const float *zr = z0 + (size_t)v * sw, *zi = zr + hw;
    const float *mr = z0 + (size_t)vm * sw, *mi = mr + hw;
    float *lo = w->tre + ((size_t)b * n + x0) * bw + c, *loi = w->tim + ((size_t)b * n + x0) * bw + c;
    float *hi = lo + (size_t)hw * bw, *hii = loi + (size_t)hw * bw;
    for (int j = 0; j < hw; j++) {
      lo[(size_t)j * bw] = 0.5f * (zr[j] + mr[j]); loi[(size_t)j * bw] = 0.5f * (zi[j] - mi[j]);
      hi[(size_t)j * bw] = 0.5f * (zi[j] + mi[j]); hii[(size_t)j * bw] = -0.5f * (zr[j] - mr[j]);
    }
  }
}

// The picture onto the ring's grid (focus.h FOCUS_RING). The ring is painted at pxm whole pixels a module; the
// picture's n samples fill its span modules at scale samples a module, so pixel X of the region the picture and its
// guard interval cover sits at picture sample u = (X + 0.5) scale / pxm - cp scale - 0.5, and is read by a Lanczos-3
// kernel over the picture taken as periodic. Periodic is the point: it makes the guard interval the picture wrapped
// round its own edge, which is what the decoder's circular model of blur needs. Upscaling only (pxm >= scale), so the
// kernel never has to band-limit; the picture's top ring is at least 3 samples a cycle (N_FOR), well inside what
// Lanczos-3 passes. A pixel that lands on a sample exactly takes it and nothing else, so a span that divides n paints
// the exact copy it always did, to the bit.
static double lanczos3(double x) {
  x = fabs(x);
  if (x >= 3) return 0;
  const double px = 3.14159265358979323846 * x;
  return 3 * sin(px) * sin(px / 3) / (px * px);
}
// Taps for both passes, and the horizontal pass's groups: four neighbouring output pixels read inside one window of
// nine picture samples (upscaling moves the window by at most one sample a pixel, so four pixels move it by at most
// three), so each group is nine columns of four weights, zero outside a pixel's own six taps, and the pass is nine
// multiply-adds a group with no gather. Rows are padded with the picture's own wrap, so no index is masked in a loop.
static int resample_init(const focus_t *f, struct focus_ws *w, int cp) {
  const int n = f->n, K = f->pxm, span = f->frame.w - 2 * f->margin, q = (span + 2 * cp) * K, ng = (q + 3) / 4;
  w->rs_i0 = malloc((size_t)q * sizeof(int)); w->rs_w = malloc((size_t)6 * q * sizeof(float));
  w->rs_hb = malloc((size_t)ng * sizeof(int)); w->rs_hw = calloc((size_t)36 * ng, sizeof(float));
  w->rs_pic = malloc((size_t)n * n * sizeof(float));
  if (!w->rs_i0 || !w->rs_w || !w->rs_hb || !w->rs_hw || !w->rs_pic) goto fail;
  w->rs_q = q;
  const double sc = (double)n / span;
  for (int x = 0; x < q; x++) {
    const double u = (x + 0.5) * sc / K - cp * sc - 0.5, r = floor(u + 0.5);
    float *wt = w->rs_w + 6 * x;
    if (fabs(u - r) < 1e-6) {   // on a sample: that sample alone
      w->rs_i0[x] = (int)r - 2;
      for (int t = 0; t < 6; t++) wt[t] = t == 2 ? 1.0f : 0.0f;
      continue;
    }
    const int i0 = (int)floor(u) - 2;
    double tw[6], sum = 0;
    for (int t = 0; t < 6; t++) { tw[t] = lanczos3(u - (i0 + t)); sum += tw[t]; }
    w->rs_i0[x] = i0;
    for (int t = 0; t < 6; t++) wt[t] = (float)(tw[t] / sum);
  }
  int lo = 0, hi = n - 1;
  for (int g = 0; g < ng; g++) {
    const int base = w->rs_i0[4 * g];
    w->rs_hb[g] = base;
    if (base < lo) lo = base;
    if (base + 8 > hi) hi = base + 8;
    for (int l = 0; l < 4 && 4 * g + l < q; l++) {
      const int d = w->rs_i0[4 * g + l] - base;
      if (d < 0 || d > 3) goto fail;   // only a downscale moves faster than a sample a pixel, and nothing downscales
      for (int t = 0; t < 6; t++) w->rs_hw[36 * g + 4 * (d + t) + l] = w->rs_w[6 * (4 * g + l) + t];
    }
  }
  w->rs_pad = -lo; w->rs_row = n - lo + (hi - (n - 1));
  w->rs_tmp = malloc((size_t)q * w->rs_row * sizeof(float));
  if (!w->rs_tmp) goto fail;
  return 0;
fail:
  free(w->rs_i0); free(w->rs_w); free(w->rs_hb); free(w->rs_hw); free(w->rs_pic); free(w->rs_tmp);
  w->rs_i0 = w->rs_hb = NULL; w->rs_w = w->rs_hw = w->rs_pic = w->rs_tmp = NULL;
  return -1;
}
// ai: i mod n in [0, n), i negative included: n need not be a power of two.
static inline int wrap(int i, int n) { const int r = i % n; return r < 0 ? r + n : r; }
// Down the picture into padded rows, then across each into the drive, clamped to what a pixel can show. Both paths do
// the same float operations in the same order, so the vector and scalar builds paint the same bytes.
static void resample(const focus_t *f, struct focus_ws *w, float *drive, int cp) {
  const int n = f->n, q = w->rs_q, W = f->px, o = (f->margin - cp) * f->pxm, pad = w->rs_pad, rl = w->rs_row;
  for (int y = 0; y < q; y++) {
    const float *a = w->rs_w + 6 * y, *r[6];
    for (int t = 0; t < 6; t++) r[t] = w->rs_pic + (size_t)wrap(w->rs_i0[y] + t, n) * n;
    float *row = w->rs_tmp + (size_t)y * rl, *out = row + pad;
    int k = 0;
#ifdef OB_SIMD
    const v128_t a0 = wasm_f32x4_splat(a[0]), a1 = wasm_f32x4_splat(a[1]), a2 = wasm_f32x4_splat(a[2]), a3 = wasm_f32x4_splat(a[3]), a4 = wasm_f32x4_splat(a[4]), a5 = wasm_f32x4_splat(a[5]);
    for (; k + 4 <= n; k += 4) {
      v128_t v = wasm_f32x4_mul(a0, wasm_v128_load(r[0] + k));
      v = wasm_f32x4_add(v, wasm_f32x4_mul(a1, wasm_v128_load(r[1] + k)));
      v = wasm_f32x4_add(v, wasm_f32x4_mul(a2, wasm_v128_load(r[2] + k)));
      v = wasm_f32x4_add(v, wasm_f32x4_mul(a3, wasm_v128_load(r[3] + k)));
      v = wasm_f32x4_add(v, wasm_f32x4_mul(a4, wasm_v128_load(r[4] + k)));
      v = wasm_f32x4_add(v, wasm_f32x4_mul(a5, wasm_v128_load(r[5] + k)));
      wasm_v128_store(out + k, v);
    }
#endif
    for (; k < n; k++) out[k] = a[0] * r[0][k] + a[1] * r[1][k] + a[2] * r[2][k] + a[3] * r[3][k] + a[4] * r[4][k] + a[5] * r[5][k];
    for (int j = 0; j < pad; j++) row[j] = out[wrap(j - pad, n)];                 // the wrap either side
    for (int j = pad + n; j < rl; j++) row[j] = out[wrap(j - pad, n)];
  }
  const int ng = (q + 3) / 4;
  for (int y = 0; y < q; y++) {
    const float *row = w->rs_tmp + (size_t)y * rl + pad;
    float *dst = drive + (size_t)(o + y) * W + o;
    int g = 0;
#ifdef OB_SIMD
    const v128_t zero = wasm_f32x4_splat(0), one = wasm_f32x4_splat(1);
    for (; g < ng && 4 * g + 4 <= q; g++) {
      const float *b = row + w->rs_hb[g], *hw = w->rs_hw + 36 * g;
      v128_t v = wasm_f32x4_mul(wasm_f32x4_splat(b[0]), wasm_v128_load(hw));
      for (int j = 1; j < 9; j++) v = wasm_f32x4_add(v, wasm_f32x4_mul(wasm_f32x4_splat(b[j]), wasm_v128_load(hw + 4 * j)));
      wasm_v128_store(dst + 4 * g, wasm_f32x4_pmin(wasm_f32x4_pmax(v, zero), one));
    }
#endif
    for (; g < ng; g++) {
      const float *b = row + w->rs_hb[g], *hw = w->rs_hw + 36 * g;
      for (int l = 0; l < 4 && 4 * g + l < q; l++) {
        float v = b[0] * hw[l];
        for (int j = 1; j < 9; j++) v = v + b[j] * hw[4 * j + l];
        dst[4 * g + l] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
  }
}

// ai: The guard interval's depth in modules: FOCUS_CP, or what the border has spare below it (OB_THIN upward).
static int guard_of(const focus_t *f) { const int room = f->margin - OB_THIN; return FOCUS_CP < room ? FOCUS_CP : room; }

// ai: The resampler's tables for an encoder elsewhere (2026-09-29, the sender's GPU encoder, gpu/encoder.mjs), built as
// ai: focus_encode builds them on its first call.
int focus_resample_geom(const focus_t *f, int *out, const int **i0, const float **w) {
  struct focus_ws *ws = f->ws;
  const int cp = guard_of(f);
  if (!ws || (!ws->rs_pic && resample_init(f, ws, cp))) return -1;
  out[0] = ws->rs_q; out[1] = (f->margin - cp) * f->pxm; out[2] = f->n; out[3] = f->px;
  *i0 = ws->rs_i0; *w = ws->rs_w;
  return 0;
}

void focus_encode(const focus_t *f, const uint8_t *blocks, float *drive) {
  double tall = ob_now_ms();
  struct focus_ws *w = f->ws;
  const int n = f->n, bw = f->bw, sw = FOCUS_STRIP, C = f->pxm, W = f->px;
  const int cp = guard_of(f);
  if (!w->rs_pic && resample_init(f, w, cp)) return;
  const size_t tn = (size_t)f->vblocks * n * bw;
  uint8_t *bits = w->bits, *data = w->data, buf[FOCUS_FRAG + FOCUS_PAR];
  memset(w->tre, 0, tn * sizeof(float)); memset(w->tim, 0, tn * sizeof(float));
  for (int b = 0; b < f->blocks; b++) {
    const uint8_t *src = blocks + (size_t)b * f->block_bytes;
    // nb: the bits the block is sent on. first: its first coefficient, in pos order (RS blocks are a sub-channel each).
    const int nb = block_bits(f, b), first = f->mode == FOCUS_LDPC ? f->block_sub[b] * FOCUS_SUB : b * FOCUS_SUB;
    memset(bits, 0, (size_t)nb);
    if (f->mode == FOCUS_LDPC) {
      const ldpc_t *code = &f->code[f->block_tier[b]];
      int B = f->block_bytes;
      uint32_t crc = ob_crc32(src, B);
      memset(data, 0, (size_t)code->k);         // past the payload and its CRC a roomier code carries zeros
      for (int i = 0; i < B * 8; i++) data[i] = (src[i >> 3] >> (7 - (i & 7))) & 1;
      for (int i = 0; i < 32; i++) data[B * 8 + i] = (crc >> (31 - i)) & 1;
      ldpc_encode(code, data, bits);            // code.nt <= nb; the tail stays zero
      if (f->bitmap) { map_out(f, w, f->block_tier[b], b, first, nb, bits, w->slot); memcpy(bits, w->slot, (size_t)nb); }
      else if (code->np) { memmove(bits, bits + code->np, (size_t)code->nt); memset(bits + code->nt, 0, (size_t)(nb - code->nt)); }   // the bits never sent dropped
    } else {
      memcpy(buf, src, FOCUS_FRAG - 2);
      uint16_t crc = crc16(buf, FOCUS_FRAG - 2);
      buf[FOCUS_FRAG - 2] = (uint8_t)(crc >> 8); buf[FOCUS_FRAG - 1] = (uint8_t)crc;
      rs_encode(buf, FOCUS_FRAG, FOCUS_PAR);
      for (int i = 0; i < nb; i++) bits[i] = (buf[i >> 3] >> (7 - (i & 7))) & 1;
    }
    // Gray QPSK, unit magnitude: bit pair (i, q) picks the sign of each axis. Only the upper half-plane is filled.
    // The lower is its conjugate reflection, so the picture is twice the real part of this half's transform,
    // and the factor of two goes the way of every other scale, into the clipping level.
    for (int s = 0; s < nb / 2; s++) {
      int p = f->pos[first + s];
      const float a = 0.70710678f * w->amp[(first + s) / FOCUS_SUB];
      w->tre[p] = bits[2 * s] ? -a : a; w->tim[p] = bits[2 * s + 1] ? -a : a;
    }
    if (w->dbg_sym) for (int i = 0; i < nb; i++) w->dbg_sym[2 * (size_t)first + i] = bits[i] ? -1 : 1;
  }
  // The frame, a module's run of pixels at a time, outside the modules the picture and its guard interval are
  // resampled over below: the ring is 15% of a LIZARD-1024 drive, and painting the whole of it was 7 of 25 ms.
  {
    const int fw = f->frame.w, m0 = f->margin - cp, m1 = fw - m0;
    for (int y = 0; y < W; y++) {
      const uint8_t *kr = f->frame.kind + (y / C) * fw;
      const int inside = y / C >= m0 && y / C < m1;
      float *row = drive + (size_t)y * W;
      for (int mx = 0; mx < fw; mx++) {
        if (inside && mx == m0) { mx = m1 - 1; continue; }
        const float v = (kr[mx] & 3) == CELL_DARK ? 0.0f : 1.0f;
        for (int x = mx * C; x < mx * C + C; x++) row[x] = v;
      }
    }
  }
  double te = ob_now_ms();
  for (int b = 0; b < f->vblocks; b++) fft_cols(w->tre + (size_t)b * n * bw, w->tim + (size_t)b * n * bw, n, bw, bw, 1);
  ob_prof_ms[PROF_F_ENC_FFT] += ob_now_ms() - te;
  // Parseval gives the picture's rms without a pass over it: every used coefficient has magnitude 1, the transform is
  // unscaled, and the real part holds half the power. So a strip can be clipped and written as soon as it exists.
  // A Hermitian inverse hands back the whole real picture, where taking the real part of the half-plane's inverse
  // gave half of it, so the clipping level is twice what it was and the scale half. Everything here is relative to
  // the picture's own rms, so the factor of two is bookkeeping and not a change to what is displayed.
  const float lim = 2.0f * f->clip * (f->tilt > 0 ? (float)sqrt(0.5 * w->power) : sqrtf(0.5f * (float)(f->subch * FOCUS_SUB))), sc = 0.5f / lim;
  for (int x0 = 0; x0 < n; x0 += sw) {
    te = ob_now_ms();
    turn_pack(f, x0);
    fft_cols(w->wre, w->wim, n, STRIP_HALF, sw, 1);
    ob_prof_ms[PROF_F_ENC_FFT] += ob_now_ms() - te;
    // The real part is the left half of the strip's columns and the imaginary part the right half.
    for (int y = 0; y < n; y++) {
      float *dst = w->rs_pic + (size_t)y * n + x0;
      clip_row(w->wre + (size_t)y * sw, dst, STRIP_HALF, lim, sc);
      clip_row(w->wim + (size_t)y * sw, dst + STRIP_HALF, STRIP_HALF, lim, sc);
    }
  }
  // The guard interval, in the border's spare depths next to the picture: the picture wrapped round its own edge.
  //
  // The decoder's whole model is that the camera's blur is a per-coefficient gain, which is true of a CIRCULAR
  // convolution. A finite picture with anything else beside it gives a linear one, and the difference is an error
  // shaped like the picture's frame, so it lands on the low frequencies blur has not already taken. White is as
  // far from the picture's own far side as a surround gets. This is the guard interval OFDM has and FOCUS never
  // specified. Measured against white: 480 px 2550 to 3899 B, fill 0.4 5833 to 8706, defocus 1.5 4368 to 7885,
  // defocus 2.0 from 11 blocks of 96 to 40, defocus 2.5 from 0 to 24; it costs 10% at 45 degrees and 8 to 17%
  // under heavy clutter. It saturates at three to four modules, about three sigma of blur.
  //
  // Only depths the border has spare (OB_THIN upward), so a border with none silently gets none: writing there
  // would paint over the format ring and the timing track, which takes an unmarked symbol to nothing at all.
  // FOCUS_CP_GREY paints the picture's mean instead, which is most of the gain for none of the wrapping.
  //
  // The wrap is the resampler's: it reads the picture as periodic, so over the guard interval it hands back the
  // picture's far side, corners included, and the wrap is a torus and not four strips. Blur at a corner of the
  // picture mixes two sides at once, and a strip only answers one of them.
  resample(f, w, drive, cp);
#ifdef FOCUS_CP_GREY
  {
    const int o = (f->margin - cp) * C, g = cp * C, span = f->frame.w - 2 * f->margin, in1 = g + span * C, q = w->rs_q;
    for (int y = 0; y < q; y++) for (int x = 0; x < q; x++) if (y < g || y >= in1 || x < g || x >= in1) drive[(size_t)(o + y) * W + o + x] = 0.5f;
  }
#endif
  // A mark wider than the border reaches into the picture, and the strips above have just written the picture
  // over it, so it is stamped back. Only the mark boxes are walked: everything else inside the margin is the
  // picture's. One list, so a design that carries corners and edges and a middle costs one pass over each.
  {
    const int fw = f->frame.w, fh = f->frame.h, S = f->frame.corner, M = f->frame.centre, E = f->frame.edge;
    int box[9][3], nb = 0;
    if (S > f->margin) for (int c = 0; c < 4; c++) { box[nb][0] = (c & 1) ? fw - S : 0; box[nb][1] = (c & 2) ? fh - S : 0; box[nb++][2] = S; }
    if (M >= 5) { box[nb][0] = (fw - M + 1) / 2; box[nb][1] = (fh - M + 1) / 2; box[nb++][2] = M; }
    if (E >= 5) {
      const int cw = (fw - E + 1) / 2, ch = (fh - E + 1) / 2;
      const int px[4] = { cw, cw, 0, fw - E }, py[4] = { 0, fh - E, ch, ch };
      for (int c = 0; c < 4; c++) { box[nb][0] = px[c]; box[nb][1] = py[c]; box[nb++][2] = E; }
    }
    for (int b = 0; b < nb; b++) for (int cy = box[b][1]; cy < box[b][1] + box[b][2]; cy++) for (int cx = box[b][0]; cx < box[b][0] + box[b][2]; cx++) {
      if (cx < 0 || cy < 0 || cx >= fw || cy >= fh) continue;
      const int k = f->frame.kind[cy * fw + cx] & 3;
      if (k == CELL_DATA) continue;
      const float v = k == CELL_DARK ? 0.0f : 1.0f;
      for (int y = 0; y < C; y++) for (int x = 0; x < C; x++) drive[(size_t)(cy * C + y) * W + cx * C + x] = v;   // C: pixels a module
    }
  }
  ob_prof_ms[PROF_F_ENC_REST] += ob_now_ms() - tall;
}

// Sums over one row of a strip for the detrend's projections: z, z c, z q. Four chains in float, as the vector path
// keeps them, so both paths round alike; the caller carries on in double.
static void row_sums(const float *z, const float *cx, const float *qx, int cols, double *rs, double *rx, double *rq) {
#ifdef OB_SIMD
  v128_t a = wasm_f32x4_splat(0), b = a, c = a;
  for (int x = 0; x < cols; x += 4) {
    const v128_t v = wasm_v128_load(z + x);
    a = wasm_f32x4_add(a, v); b = wasm_f32x4_add(b, wasm_f32x4_mul(v, wasm_v128_load(cx + x))); c = wasm_f32x4_add(c, wasm_f32x4_mul(v, wasm_v128_load(qx + x)));
  }
  float A[4], B[4], Q[4];
  wasm_v128_store(A, a); wasm_v128_store(B, b); wasm_v128_store(Q, c);
#else
  float A[4] = { 0, 0, 0, 0 }, B[4] = { 0, 0, 0, 0 }, Q[4] = { 0, 0, 0, 0 };
  for (int x = 0; x < cols; x += 4) for (int l = 0; l < 4; l++) { const float v = z[x + l]; A[l] += v; B[l] += v * cx[x + l]; Q[l] += v * qx[x + l]; }
#endif
  *rs = (double)A[0] + A[1] + A[2] + A[3]; *rx = (double)B[0] + B[1] + B[2] + B[3]; *rq = (double)Q[0] + Q[1] + Q[2] + Q[3];
}

// t -= (ar + i ai) * x over cols values.
static void sub_scaled(float *tr, float *ti, const float *xr, const float *xi, float ar, float ai, int cols) {
  int c = 0;
#ifdef OB_SIMD
  const v128_t var = wasm_f32x4_splat(ar), vai = wasm_f32x4_splat(ai);
  for (; c + 4 <= cols; c += 4) {
    const v128_t r = wasm_v128_load(xr + c), i = wasm_v128_load(xi + c);
    wasm_v128_store(tr + c, wasm_f32x4_sub(wasm_v128_load(tr + c), wasm_f32x4_sub(wasm_f32x4_mul(var, r), wasm_f32x4_mul(vai, i))));
    wasm_v128_store(ti + c, wasm_f32x4_sub(wasm_v128_load(ti + c), wasm_f32x4_add(wasm_f32x4_mul(var, i), wasm_f32x4_mul(vai, r))));
  }
#endif
  for (; c < cols; c++) { float r = xr[c], i = xi[c]; tr[c] -= ar * r - ai * i; ti[c] -= ar * i + ai * r; }
}

// The decode is split at the sampling boundary so the sampler can live somewhere else: on a GPU (the first WebGPU
// port's sampler, since deleted) a fence costs more than the stage unless the caller can have several frames out
// at once, which it cannot while the decode is one blocking call. focus_decode is the two halves called in a row,
// so the single-call path cannot drift away from the split one and every digest gate covers both.
//
// What focus_acquire leaves behind is the registration and the image POINTER. The caller's pixels have to
// outlive the matching focus_finish; in the rig a worker owns the frame for the whole decode, so they do.
int focus_acquire(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res) {
  return focus_acquire_quad(f, img, iw, ih, gamma, mesh, res, 0, 0, 0);
}

static int acquire_hold(const focus_t *f, const image_t *pim, ob_reg_t *reg, int got);

// The same with the quad found elsewhere. quad is eight floats in image pixels, or 0 to
// run the finder here.
int focus_acquire_quad(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res,
                       const float *quad, int orient, float score) {
  memset(res, 0, sizeof *res);
  double t0 = ob_now_ms();
  image_t im;
  ob_image_init(&im, img, iw, ih, gamma);
  ob_reg_t reg;
  int got = quad ? ob_acquire_quad(&f->frame, &im, mesh, &reg, res, quad, orient, score)
                 : ob_acquire(&f->frame, &im, mesh, &reg, res);
  res->ms_detect = (float)(ob_now_ms() - t0);
  return acquire_hold(f, &im, &reg, got);
}

// ai: The picture size a format takes, sim/lizard_pick.mjs N_FOR: the first of FOCUS_PICTURES that oversamples its top
// ai: ring by half again (n >= 3R), the last where none does. The sizes are 2^k and 3 x 2^k (since 2026-09-27), so no
// ai: step is more than 1.5 times the last and n / 2R stays between 1.5 and 2.24; the powers of two alone put
// ai: LIZARD-144 at 1024 (2.99). A receiver reads n off the word's version through this.
const int FOCUS_PICTURES[FOCUS_NPICTURES] = { 256, 384, 512, 768, 1024, 1536 };
int focus_n_for(int subch) {
  const double r = sqrt(2.0 * FOCUS_SUB * subch / 3.14159265358979323846);
  for (int i = 0; i < FOCUS_NPICTURES; i++) if (FOCUS_PICTURES[i] >= 3 * r) return FOCUS_PICTURES[i];
  return FOCUS_PICTURES[FOCUS_NPICTURES - 1];
}

// A registration held and not finished, let go of.
static void drop_hold(const focus_t *f) {
  struct focus_ws *w = f->ws;
  if (w->hold_valid) { ob_reg_free(&w->hold_reg); w->hold_valid = 0; }
}

// ai: The ring that registered is the finder's call over the side counts the receiver holds (acquire.c find_frame
// ai: scores each ring's track). Its word says what is inside, and nothing checks it against the ring: any ring may
// ai: carry any picture (since 2026-09-27). Until then a codec held each picture size
// ai: at its own side count, a registration the word did not confirm was tried again at the other counts, and one
// ai: with no word stood at the count's picture; a receiver now holds the last word itself (wasm.c focus_any_rx).
int focus_acquire_ring(const focus_t *const *rings, int k, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res, int *which) {
  const ob_layout_t *Ls[FOCUS_RINGS];
  *which = -1;
  memset(res, 0, sizeof *res);
  if (k < 1 || k > FOCUS_RINGS) return 0;
  for (int i = 0; i < k; i++) { drop_hold(rings[i]); Ls[i] = &rings[i]->frame; }
  const double t0 = ob_now_ms();
  image_t im;
  ob_image_init(&im, img, iw, ih, gamma);
  ob_reg_t reg;
  int wi = 0;
  const int got = ob_acquire_any(Ls, k, &im, mesh, &reg, res, &wi);
  res->ms_detect = (float)(ob_now_ms() - t0);
  if (!got) { ob_reg_free(&reg); return 0; }
  acquire_hold(rings[wi], &im, &reg, 1);
  *which = wi;
  return 1;
}

int focus_hand_over(const focus_t *from, const focus_t *to) {
  struct focus_ws *w = from->ws;
  if (!w->hold_valid || from->frame.w != to->frame.w || from->frame.h != to->frame.h) return 0;
  drop_hold(to);
  const image_t im = w->hold_im;
  ob_reg_t reg = w->hold_reg;
  w->hold_valid = 0;   // ai: the registration's nodes now belong to `to`
  return acquire_hold(to, &im, &reg, 1);
}

void focus_release(const focus_t *f) { drop_hold(f); }

static int acquire_hold(const focus_t *f, const image_t *pim, ob_reg_t *reg_in, int got) {
  struct focus_ws *w = f->ws;
  image_t im = *pim;
  ob_reg_t reg = *reg_in;
  // ai: A capture that does not register leaves nothing held.
  if (!got) { ob_reg_free(&reg); w->hold_valid = 0; return 0; }
  // The format word, before anything is sampled: the picture size decides the sampling grid below,
  // so this is the last moment it can be learned. Reading it costs about 570 bilinear samples
  // against the picture's n^2, and it is what lets a receiver that was told nothing read the symbol.
  w->fmt_rx_ok = 0;
  if (w->fmt_cells && ob_thin_read_fmt(&f->frame, &im, &reg, w->fmt_q, w->fmt_cells))
    w->fmt_rx_ok = ob_fmt_decode(w->fmt_q, w->fmt_cells, w->fmt_bytes, 1, OB_FMT_VERSION_MAX, &w->fmt_rx) == 0;
  w->hold_im = im; w->hold_reg = reg; w->hold_valid = 1;
  return 1;
}

// The rest of the decode. grid: n * n samples the caller took, in strip order (strip s at s * n * FOCUS_STRIP,
// n rows of FOCUS_STRIP, which is the shape the transform wants), or NULL to sample here the way it always did.
// A supplied grid is transformed IN PLACE, so it is scratch the caller does not read again.
int focus_finish_ext(const focus_t *f, float *grid, const int8_t *ext_llr, const float *ext_est, uint8_t *blocks, uint8_t *ok, ob_result_t *res) {
  return focus_finish_bits(f, grid, ext_llr, ext_est, 0, 0, blocks, ok, res);
}

// The same again with the LDPC itself run elsewhere: ext_bits is one hard decision a byte, blocks * n of them
// at the widest code's n, and ext_its the iteration count each block used (negative the way ldpc_decode_stall
// ai: reports it). Everything after that, the decline gate and the CRC, is the shared path, so a block
// decoded on a GPU is accepted on exactly the terms one decoded here is.
int focus_finish_bits(const focus_t *f, float *grid, const int8_t *ext_llr, const float *ext_est, const uint8_t *ext_bits, const int8_t *ext_its, uint8_t *blocks, uint8_t *ok, ob_result_t *res) {
  struct focus_ws *w = f->ws;
  memset(ok, 0, (size_t)f->blocks);
  if (!w->hold_valid) return 0;
  image_t im = w->hold_im;
  ob_reg_t reg = w->hold_reg;
  w->hold_valid = 0;
  double t0 = ob_now_ms();
  const int n = f->n, bw = f->bw, sw = FOCUS_STRIP;
  const int armed = ob_debug_hash != 0;
  uint32_t hsh = 2166136261u;
  // A screen is never evenly lit, and a brightness ramp across the picture is not periodic: its spectrum falls off
  // as 1 / k along both axes and buries the coefficients nearest zero frequency, which are the ones meant to be the
  // most robust. So the least-squares surface 1, x, y, xy, x^2, y^2 comes out. On a symmetric grid, with the mean of
  // x^2 taken out of the squares, the six are orthogonal, so each coefficient is one projection, summed here as the
  // strips are sampled. The surface is separable, so its spectrum is cheap, and it is subtracted there: x and x^2
  // live in the column v = 0, y and y^2 in the row u = 0, xy is an outer product, and the constant is DC, which
  // carries nothing.
  double px = 0, py = 0, pxy = 0, pxx = 0, pyy = 0;
  const float g0 = f->margin + 0.5f / f->scale, step = 1.0f / f->scale;
  // A caller that brings its own soft values brings the spectrum's whole product with them, so none of this
  // runs. The registration still has to be let go, which is why the free below is outside the guard.
  if (ext_llr) { ob_reg_free(&reg); goto have_llr; }
  for (int x0 = 0; x0 < n; x0 += sw) {
    double ta = ob_now_ms();
    // The strip either comes from the caller or is taken here. Everything downstream works on `z`, so the two
    // paths differ in this one pointer and nothing else.
    float *z = grid ? grid + (size_t)(x0 / sw) * n * sw : w->wre;
    if (!grid) ob_sample_grid(&f->frame, &im, &reg, g0, g0, step, x0, sw, 0, n, w->wre, sw);
    double tb = ob_now_ms(); ob_prof_ms[PROF_F_SAMPLE] += tb - ta;
    for (int y = 0; y < n; y++) {
      double rs, rx, rq;
      row_sums(z + (size_t)y * sw, w->cx + x0, w->qx + x0, sw, &rs, &rx, &rq);
      px += rx; py += rs * w->cx[y]; pxy += rx * w->cx[y]; pxx += rq; pyy += rs * w->qx[y];
    }
    double tc = ob_now_ms(); ob_prof_ms[PROF_F_DETREND] += tc - tb;
    // The strip's two halves ARE the real and imaginary inputs, in place, at the strip's pitch (unpack_turn).
    fft_cols(z, z + STRIP_HALF, n, STRIP_HALF, sw, 0);
    unpack_turn(f, x0, z);
    ob_prof_ms[PROF_F_FFT] += ob_now_ms() - tc;
  }
  ob_reg_free(&reg);
  double ts = ob_now_ms();
  for (int b = 0; b < f->vblocks; b++) fft_cols(w->tre + (size_t)b * n * bw, w->tim + (size_t)b * n * bw, n, bw, bw, 0);
  double td = ob_now_ms(); ob_prof_ms[PROF_F_FFT] += td - ts;
  px /= n * w->sx2; py /= n * w->sx2; pxy /= w->sx2 * w->sx2; pxx /= n * w->sq2; pyy /= n * w->sq2;
  for (int b = 0; b < f->vblocks; b++) for (int u = 0; u < n; u++)
    sub_scaled(w->tre + ((size_t)b * n + u) * bw, w->tim + ((size_t)b * n + u) * bw, w->x1r + b * bw, w->x1i + b * bw, (float)(pxy * w->x1r[u]), (float)(pxy * w->x1i[u]), bw);
  for (int u = 0; u < n; u++) {
    w->tre[(size_t)u * bw] -= (float)(n * (px * w->x1r[u] + pxx * w->x2r[u])); w->tim[(size_t)u * bw] -= (float)(n * (px * w->x1i[u] + pxx * w->x2i[u]));
  }
  for (int v = 0; v < f->vblocks * bw; v++) {
    size_t at = (size_t)(v / bw) * n * bw + v % bw;
    w->tre[at] -= (float)(n * (py * w->x1r[v] + pyy * w->x2r[v])); w->tim[at] -= (float)(n * (py * w->x1i[v] + pyy * w->x2i[v]));
  }
  ob_prof_ms[PROF_F_DETREND] += ob_now_ms() - td;
have_llr:
  if (w->dbg_coef) for (int i = 0; i < f->subch * FOCUS_SUB; i++) { w->dbg_coef[2 * i] = w->tre[f->pos[i]]; w->dbg_coef[2 * i + 1] = w->tim[f->pos[i]]; }
  if (armed) { hsh = fnv(hsh, w->tre, (size_t)f->vblocks * n * bw * sizeof(float)); hsh = fnv(hsh, w->tim, (size_t)f->vblocks * n * bw * sizeof(float)); }
  res->ms_sample = (float)(ob_now_ms() - t0);
  t0 = ob_now_ms();

  const float *re = w->tre, *imag = w->tim;
  uint8_t buf[FOCUS_FRAG + FOCUS_PAR];
  int8_t *llr = w->llr;
  uint8_t *bits = w->bits;
  int good = 0;
  // ai: each block's pilots (focus_pilot), NaN until they are read
  for (int b = 0; b < f->blocks; b++) w->blk_pilot[b] = NAN;
  // ai: the grid's shift by the pilots, turned back on every coefficient the frame carries before any is read
  // ai: (a turn leaves each |y| as it was, so the moments, the estimate and the decline verdict read the same)
  w->align[0] = w->align[1] = 0;
  if (f->mode == FOCUS_LDPC && f->bitmap && !ext_llr && w->rot && pilot_align(f, w, w->align) && (w->align[0] != 0 || w->align[1] != 0)) {
    const int all = f->subch * FOCUS_SUB;
    for (int i = 0; i < all; i++) { const int p = f->pos[i]; turned(f, w, p, w->align[0], w->align[1], &w->tre[p], &w->tim[p]); }
  }
  for (int b = 0; b < f->blocks; b++) {
    const int nb = block_bits(f, b), first = f->mode == FOCUS_LDPC ? f->block_sub[b] * FOCUS_SUB : b * FOCUS_SUB;
    const int *pos = f->pos + first;
    double tb = ob_now_ms();
    if (f->mode == FOCUS_RS) {
      memset(buf, 0, sizeof buf);
      for (int s = 0; s < nb / 2; s++) {
        if (re[pos[s]] < 0) buf[(2 * s) >> 3] |= (uint8_t)(1 << (7 - ((2 * s) & 7)));
        if (imag[pos[s]] < 0) buf[(2 * s + 1) >> 3] |= (uint8_t)(1 << (7 - ((2 * s + 1) & 7)));
      }
      if (rs_decode(buf, FOCUS_FRAG + FOCUS_PAR, FOCUS_PAR) < 0) continue;
      uint16_t crc = crc16(buf, FOCUS_FRAG - 2);
      if (buf[FOCUS_FRAG - 2] != (uint8_t)(crc >> 8) || buf[FOCUS_FRAG - 1] != (uint8_t)crc) continue;
      memcpy(blocks + (size_t)b * f->block_bytes, buf, (size_t)f->block_bytes);
    } else {
      // Blur scales each ring differently, so signal and noise are estimated per sub-channel
      // from the second and fourth moments of |y| (QPSK in complex Gaussian noise).
      const ldpc_t *code = &f->code[f->block_tier[b]];
      double info = 0;
      // Quantised elsewhere: the soft values arrive per GLOBAL sub-channel, 2 * FOCUS_SUB
      // to each, so this block's run is contiguous from its first one. Everything after this point is the same
      // either way, which is what keeps the two paths from drifting.
      if (ext_llr) {
        const int subs = f->tier[f->block_tier[b]].subs, s0 = f->block_sub[b];
        memcpy(llr, ext_llr + (size_t)s0 * 2 * FOCUS_SUB, (size_t)(nb < subs * 2 * FOCUS_SUB ? nb : subs * 2 * FOCUS_SUB));
        for (int g = 0; g < subs; g++) info += ext_est[s0 + g];
        w->blk_est[b] = (float)(info / subs); w->blk_its[b] = 0;
        if (armed) hsh = fnv(hsh, llr, (size_t)nb);
        ob_prof_ms[PROF_F_LLR] += ob_now_ms() - tb; tb = ob_now_ms();
      } else {
      for (int g = 0; g < f->tier[f->block_tier[b]].subs; g++) {
        const int *pp = pos + g * FOCUS_SUB;
        float gr[FOCUS_SUB], gi[FOCUS_SUB];
        double m2 = 0, m4 = 0;
        for (int s = 0; s < FOCUS_SUB; s++) { gr[s] = re[pp[s]]; gi[s] = imag[pp[s]]; }
        for (int s = 0; s < FOCUS_SUB; s++) { double e = (double)gr[s] * gr[s] + (double)gi[s] * gi[s]; m2 += e; m4 += e * e; }
        m2 /= FOCUS_SUB; m4 /= FOCUS_SUB;
        // ai: The pilots (SPEC 7.3): this block's slots past the codeword, where they fall in this sub-channel (the top
        // ai: 16 coefficients of its last), carry the whitening XOR a bit of the painted picture's count. Their axis
        // ai: values against the whitening's signs, in units of the sub-channel's rms axis value sqrt(m2 / 2), which a
        // ai: straddle leaves as it is: +1 where the block's bit is 0, -1 where it is 1, 1 - 2 f on a mix (f from a
        // ai: picture whose bit is the other).
        if (f->bitmap && code->nt < nb && m2 > 0 && code->nt / 2 >= g * FOCUS_SUB && code->nt / 2 < (g + 1) * FOCUS_SUB) {
          const uint8_t *wh = w->white + 2 * (size_t)first;
          double cs = 0;
          int t = 0;
          for (int i = code->nt; i < nb && i / 2 < (g + 1) * FOCUS_SUB; i++, t++) cs += (wh[i] ? -1.0 : 1.0) * ((i & 1) ? gi[i / 2 - g * FOCUS_SUB] : gr[i / 2 - g * FOCUS_SUB]);
          w->blk_pilot[b] = (float)(cs / t / sqrt(m2 / 2));
        }
        double a2 = 2 * m2 * m2 - m4;
        a2 = a2 > 0 ? sqrt(a2) : 0;
        double nv = m2 - a2;
        if (nv < 0.02 * m2) nv = 0.02 * m2;
        info += axis_info(a2 / nv);
        // Per axis: amplitude sqrt(a2 / 2), noise variance nv / 2, so LLR = 2 sqrt(2 a2) y / nv.
        const float k = (float)(0.7 * 2 * sqrt(2 * a2) / nv);
        int8_t *out = llr + 2 * g * FOCUS_SUB;
        for (int s = 0; s < FOCUS_SUB; s++) {
          float lr = k * gr[s], li = k * gi[s];
          lr = lr > 10 ? 10 : lr < -10 ? -10 : lr; li = li > 10 ? 10 : li < -10 ? -10 : li;
          out[2 * s] = (int8_t)nearbyintf(lr * 8); out[2 * s + 1] = (int8_t)nearbyintf(li * 8);
        }
      }
      w->blk_est[b] = (float)(info / f->tier[f->block_tier[b]].subs); w->blk_its[b] = 0;
      if (armed) hsh = fnv(hsh, llr, (size_t)nb);
      ob_prof_ms[PROF_F_LLR] += ob_now_ms() - tb; tb = ob_now_ms();
      }
      // A failed block used to burn all 30 iterations, up to 44% of a hard frame's decode. Two ways out, both set on
      // 78 000 blocks over 17 conditions and 5 configurations, so that not one block that
      // decodes is lost: a block whose sub-channels hold next to nothing is not tried (the estimate runs low on
      // outer rings at low resolution, hence a bar far under the code rate: the lowest decoded block was at 0.33 of
      // its rate), and one still at its first count of violated checks after 9 iterations is let go (slow blocks
      // that decoded were at 0.91 of it or under by then). Together 45 to 60% of the iterations.
      if (w->blk_est[b] < FOCUS_DECLINE * (float)code->k / (float)code->nt) { w->blk_its[b] = -2; ob_prof_ms[PROF_F_LDPC] += ob_now_ms() - tb; continue; }
      // Slot order to codeword order through the bit map, whitening off, and only for a block that is tried. From
      // here on the soft values are the code's (the bits never sent at 0).
      int8_t *cl = llr;
      if (f->bitmap) { map_in(f, w, f->block_tier[b], first, llr, w->llr2); cl = w->llr2; }
      else if (code->np) { memmove(llr + code->np, llr, (size_t)code->nt); memset(llr, 0, (size_t)code->np); }
      for (int i = (f->block_bytes + 4) * 8; i < code->k; i++) cl[i] = 127;   // the zeros a roomier code carries: known, and said so
      int its;
      if (ext_bits) {
        its = ext_its ? ext_its[b] : 1;
        // The caller's decisions, at this code's length. w->bits is what everything below reads,
        // so they land there and nothing downstream knows the difference.
        for (int i = 0; i < code->n; i++) bits[i] = ext_bits[(size_t)b * f->code[0].n + i];
      } else its = ldpc_decode_stall(code, cl, bits, 30, FOCUS_STALL_IT, FOCUS_STALL_RATIO);
      w->blk_its[b] = (int8_t)its;
      ob_prof_ms[PROF_F_LDPC] += ob_now_ms() - tb;
      if (its < 0) continue;
      int B = f->block_bytes;
      uint8_t *dst = blocks + (size_t)b * B;
      memset(dst, 0, (size_t)B);
      for (int i = 0; i < B * 8; i++) dst[i >> 3] |= (uint8_t)(bits[i] << (7 - (i & 7)));
      uint32_t crc = 0;
      for (int i = 0; i < 32; i++) crc = (crc << 1) | bits[B * 8 + i];
      if (crc != ob_crc32(dst, B)) continue;
    }
    ok[b] = 1; good++;
  }
  if (armed) ob_debug_hash = hsh;
  res->tiles_ok = good;
  res->ms_decode = (float)(ob_now_ms() - t0);
  return good;
}

// The block tail with everything before it done elsewhere: a verdict a block in
// focus_finish_bits's own order (1 passed, 2 declined, 3 never converged, 4 CRC), the payload bytes of the ones
// that passed at `stride` a block, and the numbers the stats read. No registration is needed or consumed; the
// caller says what it registered so ob_last reports it.
int focus_assemble(const focus_t *f, const uint8_t *bytes, int stride, const uint8_t *verdicts, const int8_t *its, const float *est,
                   uint8_t *blocks, uint8_t *ok, ob_result_t *res, const float *quad, int orient, float score) {
  struct focus_ws *w = f->ws;
  memset(ok, 0, (size_t)f->blocks);
  int good = 0;
  for (int b = 0; b < f->blocks; b++) {
    w->blk_est[b] = est ? est[b] : 0;
    w->blk_its[b] = verdicts[b] == 2 ? -2 : its ? its[b] : 1;
    if (verdicts[b] != 1) continue;
    memcpy(blocks + (size_t)b * f->block_bytes, bytes + (size_t)b * stride, (size_t)f->block_bytes);
    ok[b] = 1; good++;
  }
  w->hold_valid = 0;
  res->found = 1; res->orient = orient; res->mark_score = score; res->tiles_ok = good;
  if (quad) { memcpy(res->quad, quad, sizeof res->quad); memcpy(ob_raw_quad, quad, sizeof ob_raw_quad); }
  return good;
}

// The format word from soft values read elsewhere (sim/ob.mjs fmtCheck), [ok, q0..]: what focus_acquire does with
// ai: the soft values it reads itself, without an acquisition. The word feeds only the status band (focus_fmt_rx),
// so one a batch is enough.
int focus_fmt_check(const focus_t *f, const float *q) {
  struct focus_ws *w = f->ws;
  w->fmt_rx_ok = 0;
  if (!w->fmt_cells || !q || q[0] == 0) return 0;
  for (int i = 0; i < 4 * w->fmt_cells; i++) w->fmt_q[i] = q[1 + i];
  w->fmt_rx_ok = ob_fmt_decode(w->fmt_q, w->fmt_cells, w->fmt_bytes, 1, OB_FMT_VERSION_MAX, &w->fmt_rx) == 0;
  return w->fmt_rx_ok;
}

// What focus_acquire is holding, for a caller that has to sample elsewhere. Valid until the matching focus_finish.
const image_t *focus_held_image(const struct focus_s *f) { return &f->ws->hold_im; }
const ob_reg_t *focus_held_reg(const struct focus_s *f) { return &f->ws->hold_reg; }

int focus_finish(const focus_t *f, float *grid, uint8_t *blocks, uint8_t *ok, ob_result_t *res) {
  return focus_finish_ext(f, grid, 0, 0, blocks, ok, res);
}

int focus_decode(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, ob_result_t *res) {
  if (!focus_acquire(f, img, iw, ih, gamma, mesh, res)) { memset(ok, 0, (size_t)f->blocks); return 0; }
  return focus_finish(f, 0, blocks, ok, res);
}

// The detrend's basis and its transforms, for a caller running the transform elsewhere (gpu/back/transform.mjs).
// Layout: cx[n], qx[n], x1r[n], x1i[n], x2r[n], x2i[n], then sx2 and sq2. All f32; the two norms are doubles
// here and are narrowed on the way out, which costs the projections a last bit and nothing else.
void focus_tables(const focus_t *f, float *out) {
  struct focus_ws *w = f->ws;
  const int n = f->n;
  for (int i = 0; i < n; i++) { out[i] = w->cx[i]; out[n + i] = w->qx[i]; out[2 * n + i] = w->x1r[i]; out[3 * n + i] = w->x1i[i]; out[4 * n + i] = w->x2r[i]; out[5 * n + i] = w->x2i[i]; }
  out[6 * n] = (float)w->sx2; out[6 * n + 1] = (float)w->sq2;
}
// The spectrum the last decode produced, vblocks * n * bw of each. What a second implementation of the
// transform has to reproduce.
void focus_spectrum(const focus_t *f, float *re, float *im) {
  const size_t tn = (size_t)f->vblocks * f->n * f->bw;
  for (size_t i = 0; i < tn; i++) { re[i] = f->ws->tre[i]; im[i] = f->ws->tim[i]; }
}

// ---- the block tail, for the first WebGPU port to be held against (since deleted) ----
//
// focus_finish_bits runs the packing, the embedded CRC and ob_crc32 inline, so a second implementation has
// nothing to compare against block by block. These expose exactly those lines and nothing else: the reference
// is the C that ships, not a copy of it written for the test.

// One block's hard decisions to its bytes and the two CRCs. bits is one decision a byte: B payload bytes'
// worth, then the 32 the codeword carries. Returns 1 when they agree, which is what focus_finish_bits keeps.
int ob_test_crc_pack(const uint8_t *bits, int B, uint8_t *dst, uint32_t *out) {
  memset(dst, 0, (size_t)B);
  for (int i = 0; i < B * 8; i++) dst[i >> 3] |= (uint8_t)(bits[i] << (7 - (i & 7)));
  uint32_t emb = 0;
  for (int i = 0; i < 32; i++) emb = (emb << 1) | bits[B * 8 + i];
  uint32_t got = ob_crc32(dst, B);
  out[0] = got; out[1] = emb;
  return got == emb;
}

uint32_t ob_test_crc_hash(const uint8_t *p, int n) { return ob_crc32(p, n); }

// The decline gate's arithmetic on the external path: est is one estimate a sub-channel (what the GPU's a2 and
// nv give), and a block's is the mean over the sub-channels it owns, held to FOCUS_DECLINE of its code rate.
// The sum is a double here because it is a double there, and the bar is near enough to the values to care.
int ob_test_crc_gate(const focus_t *f, const float *est, float *blk_est, float *bar, uint8_t *declined) {
  if (f->mode != FOCUS_LDPC) return 0;
  for (int b = 0; b < f->blocks; b++) {
    const ldpc_t *code = &f->code[f->block_tier[b]];
    const int subs = f->tier[f->block_tier[b]].subs, s0 = f->block_sub[b];
    double info = 0;
    for (int g = 0; g < subs; g++) info += est[s0 + g];
    blk_est[b] = (float)(info / subs);
    bar[b] = FOCUS_DECLINE * (float)code->k / (float)code->nt;
    declined[b] = blk_est[b] < bar[b];
  }
  return f->blocks;
}
