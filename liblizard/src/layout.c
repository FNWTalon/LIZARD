#include "layout.h"
#include "fmt.h"
#include "simd.h"
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
// ai: arm64: the same CRC-32 (reflected 0xedb88320, the register and the result inverted) by ARMv8's CRC32 instructions,
// ai: eight bytes an instruction, 80x the loop below on a phone core; the same value for every input
// ai: (test/crc_check.c). Optional in ARMv8.0, in every ARMv8.1 core: asked of the kernel (HWCAP_CRC32) on Linux and
// ai: Android, always there on Apple's.
#include <arm_acle.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif
#define OB_HAVE_CRC32() ((getauxval(AT_HWCAP) & HWCAP_CRC32) != 0)
#elif defined(__APPLE__)
#define OB_HAVE_CRC32() 1
#endif
#ifdef OB_HAVE_CRC32
__attribute__((target("crc"))) static uint32_t crc32_arm(const uint8_t *p, int n) {
  uint32_t c = 0xffffffffu;
  int i = 0;
  for (; i + 8 <= n; i += 8) { uint64_t v; memcpy(&v, p + i, 8); c = __crc32d(c, v); }
  for (; i < n; i++) c = __crc32b(c, p[i]);
  return ~c;
}
#endif
#elif defined(OB_X86) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
// ai: x86-64: the same CRC-32 by carry-less multiplication (PCLMULQDQ; x86's crc32 instruction is CRC-32C, another
// ai: polynomial), 16 bytes a fold. In the reflected domain a 128-bit remainder D bits before the block it lands on is
// ai: folded onto it as its low half times x^(D + 32) mod P and its high half times x^(D - 32) mod P, each constant
// ai: bit-reflected and shifted up one: four remainders 64 bytes apart while 64 bytes remain, then folded into one, then
// ai: one block at a time; the last 128 bits to 64 by x^64 mod P and to 32 by Barrett's reduction (P and x^64 / P,
// ai: reflected). n a multiple of 16, 64 or more; c the register as it stands, returned as it ends (neither inverted).
// ai: The same value for every input (test/crc_check.c); taken where cpuid has PCLMULQDQ, the bytes past the last whole
// ai: 16 by the loop below.
#include <wmmintrin.h>
#define OB_HAVE_CLMUL() (__builtin_cpu_supports("pclmul") && __builtin_cpu_supports("sse4.1"))
__attribute__((target("pclmul,sse4.1"))) static uint32_t crc32_clmul(const uint8_t *p, int n, uint32_t c) {
  const __m128i k512 = _mm_set_epi64x(0x1c6e41596, 0x154442bd4), k128 = _mm_set_epi64x(0x0ccaa009e, 0x1751997d0);
  const __m128i k64 = _mm_set_epi64x(0, 0x163cd6124), bar = _mm_set_epi64x(0x1f7011641, 0x1db710641), lo32 = _mm_setr_epi32(-1, 0, -1, 0);
#define OB_FOLD(a, k, b) _mm_xor_si128(_mm_xor_si128(_mm_clmulepi64_si128((a), (k), 0x00), _mm_clmulepi64_si128((a), (k), 0x11)), (b))
  __m128i x[4];
  for (int u = 0; u < 4; u++) x[u] = _mm_loadu_si128((const __m128i *)(p + 16 * u));
  x[0] = _mm_xor_si128(x[0], _mm_cvtsi32_si128((int)c));
  int i = 64;
  for (; i + 64 <= n; i += 64)
    for (int u = 0; u < 4; u++) x[u] = OB_FOLD(x[u], k512, _mm_loadu_si128((const __m128i *)(p + i + 16 * u)));
  __m128i a = x[0];
  for (int u = 1; u < 4; u++) a = OB_FOLD(a, k128, x[u]);
  for (; i < n; i += 16) a = OB_FOLD(a, k128, _mm_loadu_si128((const __m128i *)(p + i)));
#undef OB_FOLD
  a = _mm_xor_si128(_mm_srli_si128(a, 8), _mm_clmulepi64_si128(a, k128, 0x10));
  a = _mm_xor_si128(_mm_srli_si128(a, 4), _mm_clmulepi64_si128(_mm_and_si128(a, lo32), k64, 0x00));
  __m128i t = _mm_clmulepi64_si128(_mm_and_si128(a, lo32), bar, 0x10);
  t = _mm_clmulepi64_si128(_mm_and_si128(t, lo32), bar, 0x00);
  return (uint32_t)_mm_extract_epi32(_mm_xor_si128(a, t), 1);
}
#endif
uint32_t ob_crc32(const uint8_t *p, int n) {
#ifdef OB_HAVE_CRC32
  if (OB_HAVE_CRC32()) return crc32_arm(p, n);
#endif
  uint32_t c = 0xffffffffu;
#ifdef OB_HAVE_CLMUL
  if (n >= 64 && OB_HAVE_CLMUL()) { const int m = n & ~15; c = crc32_clmul(p, m, c); p += m; n -= m; }
#endif
  for (int i = 0; i < n; i++) {
    c ^= p[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1)));
  }
  return ~c;
}

// j is the TRACK cell's index, not a module: the track takes every second run of OB_BAND_RUN cells of the band, the word the rest.
int ob_thin_track(int side, int j) {
  uint32_t v = (uint32_t)(side * 4099 + (j >> 1)) * 2654435761u + 40503u;
  v ^= v >> 15; v *= 2246822519u; v ^= v >> 13;
  return (int)((v ^ (uint32_t)j) & 1);   // bit for the pair of track cells, flipped in its second
}

// Cell i of the coded band spans OB_CELL modules from reserve + i * OB_CELL, and the band is read
// from reserve up to `reserve` short of the far corner, so the corners, where two sides meet, are never
// read. That dead stretch is what a corner mark is painted into (layout.h OB_THIN_RESERVE).
int ob_thin_cells(int side, int reserve) { return side > 2 * reserve ? (side - 2 * reserve) / OB_CELL : 0; }
int ob_track_cells(int side, int reserve) { return (ob_thin_cells(side, reserve) + 1) / 2; }

// The format word, over the light cells thin_init left. Only these cells move, and they are flagged CELL_WORD so
// the registration mesh skips them (layout.h), so a sender can change the word without disturbing anything
// registration depends on, at either end. Every word cell is flagged, including any past the codeword (a band whose W
// is not a multiple of 8, or past OB_FMT_BYTES_MAX), which are painted light: no template treats a word position as
// fixed.
void ob_layout_set_fmt(ob_layout_t *L, const uint8_t *cw) {
  const int w = L->w, h = L->h, side_mod = w < h ? w : h;
  const int cells = ob_fmt_side_cells(side_mod, L->reserve), bytes = ob_fmt_bytes(side_mod, L->reserve);
  for (int side = 0; side < 4; side++) for (int i = 0; i < cells; i++) {
    const uint8_t v = (ob_fmt_bit(cw, bytes, side, i) ? CELL_DARK : CELL_LIGHT) | CELL_WORD;
    for (int j = 0; j < OB_CELL; j++) for (int e = 0; e < OB_RING_DEEP; e++) {
      const int t = L->reserve + OB_WORD_AT(i) + j, d = OB_THIN_FMT + e;
      const int x = side == 0 || side == 2 ? t : side == 3 ? d : w - 1 - d;
      const int y = side == 0 ? d : side == 2 ? h - 1 - d : t;
      L->kind[y * w + x] = v;
    }
  }
}


// Thin frame: a lattice every OB_THIN_PITCH modules whose border nodes sit on the border's centre
// line, each a 5 x 5 patch of the border pattern. No mark is painted: the border is the mark.
// 1 where (x, y) falls in a corner mark: a solid block filling the square [0, S) at each corner, with
// its outermost ring light so a symbol against a dark bezel is still one dark run the scan can find
// the inner edge of, and its innermost ring light as the separator between the block and the coded rings inside it.
static int thin_corner(const ob_layout_t *L, int x, int y, int *dark) {
  const int S = L->corner;
  // FOUR corners. Three, QR's arrangement, was tried on 2026-09-21 and reverted: the fourth corner is a MEASURED
  // point, and inferring it as a parallelogram's corner leaves the map affine until an edge mark is found to put
  // the perspective back. The edge marks are found in the easy captures, where the map was already right, and
  // missed in the hard ones, where it is not: fill 0.3 fell from 2990 B on 16 of 16 frames to 352 B on 7.
  const int cx = x < L->w - 1 - x ? x : L->w - 1 - x, cy = y < L->h - 1 - y ? y : L->h - 1 - y;
  if (cx >= S || cy >= S) return 0;
  const int in = cx >= 1 && cx <= S - 2 && cy >= 1 && cy <= S - 2;
  // The gap band stays light, so the line's inner edge, which fit_lines fits, is at the same depth under a mark as
  // anywhere else and the mark cannot pull the curve.
  const int d = cx < cy ? cx : cy;
  *dark = in && !(d >= OB_THIN_GAP && d < OB_THIN_TRACK);
  return 1;
}

static int thin_init(ob_layout_t *L, int w, int h) {
  for (int axis = 0; axis < 2; axis++) {
    int len = axis ? h : w, n = 0;
    float *pos = axis ? L->node_y : L->node_x;
    // OB_NODE_LINE (layout.h) says where this runs and why it is not the border's middle.
    for (float c = OB_NODE_LINE; c < len - OB_NODE_LINE - OB_THIN_PITCH / 2 && n < OB_MAX_NODES - 1; c += OB_THIN_PITCH) pos[n++] = c;
    pos[n++] = len - OB_NODE_LINE;
    if (axis) L->ny = n; else L->nx = n;
  }
  for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
    int dx = x < w - 1 - x ? x : w - 1 - x, dy = y < h - 1 - y ? y : h - 1 - y, d = dx < dy ? dx : dy, dark = 0;
    if (thin_corner(L, x, y, &dark)) { L->kind[y * w + x] = dark ? CELL_DARK : CELL_LIGHT; continue; }
    if (d >= 1 && d <= OB_THIN_LINE) dark = 1;
    else if (d >= OB_THIN_TRACK && d < OB_THIN) {
      // The one coded band. Even runs are the track, odd runs are left light for the word (ob_layout_set_fmt),
      // so a configuration with no expressible word shows gaps rather than someone else's bits.
      const int along = dy <= dx ? x : y, len = dy <= dx ? w : h;
      const int side = dy <= dx ? (y < h / 2 ? 0 : 2) : (x < w / 2 ? 3 : 1);
      const int cell = (along - L->reserve) / OB_CELL, run = cell / OB_BAND_RUN;
      if (along >= L->reserve && along < len - L->reserve && !(run & 1))
        dark = ob_thin_track(side, run / 2 * OB_BAND_RUN + cell % OB_BAND_RUN);
    }
    // The guard interval is the picture's, not the frame's: DATA, so no template reads it and no sampler expects
    // it to be anything, as lizard_phy.c paints it.
    L->kind[y * w + x] = d >= L->margin - LIZARD_CP ? CELL_DATA : dark ? CELL_DARK : CELL_LIGHT;   // DATA: the picture, which the frame's templates skip
  }
  for (int j = 0; j < L->ny; j++) for (int i = 0; i < L->nx; i++) {
    int edge_x = i == 0 || i == L->nx - 1, edge_y = j == 0 || j == L->ny - 1;
    if ((edge_x || edge_y) && !(edge_x && edge_y)) L->node_mark[j * L->nx + i] = 1;
  }
  return 0;
}

int ob_layout_init(ob_layout_t *L, int side) {
  memset(L, 0, sizeof *L);
  if (side < 64) return -1;
  L->w = L->h = side;
  L->margin = OB_MARGIN; L->corner = OB_THIN_CORNER; L->reserve = OB_THIN_RESERVE;
  L->kind = calloc((size_t)side * side, 1);
  return L->kind ? thin_init(L, side, side) : -1;
}

void ob_layout_free(ob_layout_t *L) {
  free(L->kind);
  memset(L, 0, sizeof *L);
}
