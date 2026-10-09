#include "layout.h"
#include "fmt.h"
#include <stdlib.h>
#include <string.h>

uint32_t ob_crc32(const uint8_t *p, int n) {
  uint32_t c = 0xffffffffu;
  for (int i = 0; i < n; i++) {
    c ^= p[i];
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1)));
  }
  return ~c;
}

// j is the TRACK cell's index, not a module: the track takes every second run of OB_BAND_RUN cells of the band, the word the rest.
int ob_thin_track(int side, int j, int alt) {
  if (alt) return j & 1;   // Data Matrix's clock: the same on every side, so it cannot fix rotation
  uint32_t v = (uint32_t)(side * 4099 + (j >> 1)) * 2654435761u + 40503u;
  v ^= v >> 15; v *= 2246822519u; v ^= v >> 13;
  return (int)((v ^ (uint32_t)j) & 1);   // bit for the pair of track cells, flipped in its second
}

// Cell i of the coded band spans OB_CELL modules from reserve + i * OB_CELL, and the band is read
// from reserve up to `reserve` short of the far corner, so the corners, where two sides meet, are never
// read. That dead stretch is what a corner mark is painted into, which is why a mark of 8 modules or
// less is free: `reserve` does not move and no cell that anything reads is touched.
// ai: The mark plus 3 modules of light (2 at least, or blur drags thin_levels' light reference dark): 15 for the
// ai: 12-module mark, the border's depth, so the band's B cells run exactly beside the picture's 2 B modules (focus.h
// ai: FOCUS_RING). 14 until 2026-09-27, written (corner + 3) & ~1, when the side was n / 8 + 60.
int ob_thin_reserve(int corner) { return corner > 8 ? corner + 3 : 8; }
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
// the inner edge of, and its innermost ring light as the separator between the block and whatever
// lies inside it (the coded rings, or the picture once S passes OB_THIN).
// A mark with its rings on all four sides, at (mx, my). Away from the border there is no line to serve as two of
// its sides, so it carries its own, and its core is three modules smaller than a corner mark's for the same S.
static int mark_ring(const ob_layout_t *L, int x, int y, int mx, int my, int S, int *dark) {
  const int dx = x - mx, dy = y - my;
  if (dx < 0 || dx >= S || dy < 0 || dy >= S) return 0;
  const int cx = dx < S - 1 - dx ? dx : S - 1 - dx, cy = dy < S - 1 - dy ? dy : S - 1 - dy, d = cx < cy ? cx : cy;
  *dark = cx >= 1 && cy >= 1 && !(!L->corner_filled && d >= OB_THIN_GAP && d < OB_THIN_TRACK);
  return 1;
}

static int thin_corner(const ob_layout_t *L, int x, int y, int *dark) {
  // The centre and edge marks are the same shape as a corner mark measured from their own middle, so one
  // detector serves all of them and a race between the designs is a layout field rather than another code path.
  if (L->centre >= 5 && mark_ring(L, x, y, (L->w - L->centre + 1) / 2, (L->h - L->centre + 1) / 2, L->centre, dark)) return 1;
  if (L->edge >= 5) {
    // At depth 0, where a corner mark sits, because the border grows outwards to hold its marks and none of them
    // reaches into the picture (focus.c, THE RULE). A mark covers the timing track where it sits, which is why
    // there is at most one a side and why they are off by default (fmt.h).
    const int S = L->edge, cw = (L->w - S + 1) / 2, ch = (L->h - S + 1) / 2;
    if (mark_ring(L, x, y, cw, 0, S, dark)) return 1;                  // top
    if (mark_ring(L, x, y, cw, L->h - S, S, dark)) return 1;           // bottom
    if (mark_ring(L, x, y, 0, ch, S, dark)) return 1;                  // left
    if (mark_ring(L, x, y, L->w - S, ch, S, dark)) return 1;           // right
  }
  const int S = L->corner;
  if (S < 5) return 0;
  // FOUR corners. Three, QR's arrangement, was tried on 2026-09-21 and reverted: the fourth corner is a MEASURED
  // point, and inferring it as a parallelogram's corner leaves the map affine until an edge mark is found to put
  // the perspective back. The edge marks are found in the easy captures, where the map was already right, and
  // missed in the hard ones, where it is not: fill 0.3 fell from 2990 B on 16 of 16 frames to 352 B on 7.
  const int cx = x < L->w - 1 - x ? x : L->w - 1 - x, cy = y < L->h - 1 - y ? y : L->h - 1 - y;
  if (cx >= S || cy >= S) return 0;
  const int in = cx >= 1 && cx <= S - 2 && cy >= 1 && cy <= S - 2;
  // The gap variant leaves the gap band light, so the line's inner edge, which fit_lines fits, is at
  // the same depth under a mark as anywhere else and the mark cannot pull the curve.
  const int d = cx < cy ? cx : cy;
  *dark = in && !(!L->corner_filled && d >= OB_THIN_GAP && d < OB_THIN_TRACK);
  return 1;
}

static int thin_init(ob_layout_t *L, int w, int h) {
  L->thin = 1;
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
        dark = ob_thin_track(side, run / 2 * OB_BAND_RUN + cell % OB_BAND_RUN, L->track_alt);
    }
    // The guard interval is the picture's, not the frame's: DATA, so no template reads it and no sampler expects
    // it to be anything. Only the border's spare depths, as focus.c paints it.
    const int cp = L->margin - OB_THIN >= FOCUS_CP ? FOCUS_CP : L->margin - OB_THIN;
    L->kind[y * w + x] = d >= L->margin - cp ? CELL_DATA : dark ? CELL_DARK : CELL_LIGHT;   // DATA: the picture, which the frame's templates skip
  }
  for (int j = 0; j < L->ny; j++) for (int i = 0; i < L->nx; i++) {
    int edge_x = i == 0 || i == L->nx - 1, edge_y = j == 0 || j == L->ny - 1;
    if ((edge_x || edge_y) && !(edge_x && edge_y)) L->node_mark[j * L->nx + i] = 1;
  }
  return 0;
}

// ai: LIZARD's border is the only frame there is (thin); a configuration without it is refused.
int ob_layout_init(ob_layout_t *L, const ob_cfg_t *cfg) {
  int w = cfg->w, h = cfg->h;
  memset(L, 0, sizeof *L);
  if (cfg->thin) {
    if (w < 64 || h < 64) return -1;
    L->w = w; L->h = h;
    // A mark wider than a quarter of the side would leave the coded band nothing, and one under 5
    // has no light ring on both faces of a dark core, so neither is a mark.
    L->margin = cfg->border > OB_THIN ? cfg->border : OB_THIN;
    L->corner = cfg->corner >= 5 && 4 * cfg->corner <= w && 4 * cfg->corner <= h ? cfg->corner : 0;
    // The frame's size is the caller's, so the border cannot grow here: a mark wider than the border it was
    // given is cut down to it. A corner mark never reaches into the picture (focus.c, THE RULE).
    if (L->corner > L->margin) L->corner = L->margin;
    L->centre = cfg->centre >= 5 && 4 * cfg->centre <= w ? cfg->centre : 0;
    L->edge = cfg->edge >= 5 && 6 * cfg->edge <= w ? cfg->edge : 0;
    if (L->edge > L->margin) L->edge = L->margin;   // as for a corner mark: the frame's size is the caller's and cannot grow here
    // A centre mark is IN the picture wherever it is put, so the rule rules it out. It is kept only because
    // archive/build-scratch-2026-09/mark_race.mjs raced it and lost; nothing builds one by default.
    L->track_alt = cfg->track_alt ? 1 : 0;
    L->corner_filled = cfg->corner_filled;
    L->reserve = ob_thin_reserve(L->corner);
    L->kind = calloc((size_t)w * h, 1);
    return L->kind ? thin_init(L, w, h) : -1;
  }
  return -1;
}

void ob_layout_free(ob_layout_t *L) {
  free(L->kind);
  memset(L, 0, sizeof *L);
}
