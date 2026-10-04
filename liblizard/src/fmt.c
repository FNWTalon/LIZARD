#include "fmt.h"
#include "layout.h"
#include "rs.h"
#include <string.h>

// Whole word runs inside the band. Half the band's cells is the word's share only where the band ends on a run
// boundary; on a lab module count where it does not, that count ran past the band's end, and ob_layout_set_fmt
// painted the cells past it over the reserve and the corner mark. A partial run at the end stays light, unflagged.
int ob_fmt_side_cells(int side, int reserve) {
  const int w = ob_thin_cells(side, reserve) / (2 * OB_BAND_RUN) * OB_BAND_RUN;
  return w >= OB_FMT_SIDE_CELLS ? w : 0;
}

// From the cells, not from a table of n: a lab layout at another module count (a replay at 286 modules) gets the code
// its band holds, and its templates flag the same cells as a format symbol's would.
int ob_fmt_bytes(int side, int reserve) {
  const int w = ob_fmt_side_cells(side, reserve), b = 4 * (w / 8);
  return b > OB_FMT_BYTES_MAX ? OB_FMT_BYTES_MAX : b;
}

// Byte k goes to frame side k % 4, eight consecutive cells at 8 * (k / 4). So a side lost to glare costs every
// fourth byte, a quarter of the word, and a smear along one side takes consecutive bytes of one quarter rather than
// of the whole word.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell) {
  const int k = 4 * (cell >> 3) + edge;
  if (edge < 0 || edge > 3 || cell < 0 || k >= bytes) return 0;
  return (cw[k] >> (7 - (cell & 7))) & 1;
}

int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes) {
  if (bytes < OB_FMT_BYTES || bytes > OB_FMT_BYTES_MAX || f->version < 1 || f->version > OB_FMT_VERSION_MAX || f->fps < 0 || f->fps > 255) return -1;
  memset(cw, 0, (size_t)bytes);
  cw[0] = OB_FMT_MAGIC;
  cw[1] = (uint8_t)f->version;
  cw[2] = (uint8_t)f->fps;
  rs_encode(cw, OB_FMT_DATA, bytes - OB_FMT_DATA);
  return 0;
}

// RS can miscorrect past its distance, and a wrong word is worse than none: it would send the receiver to another
// picture size and cost every block of every frame until it changed again. The magic and the version's range are the
// check. The rate has no structure to check and nothing the decoder does depends on it, so it is taken as it comes.
static int accept(const uint8_t *cw, int vlo, int vhi, ob_fmt_t *f) {
  if (cw[0] != OB_FMT_MAGIC || cw[1] < vlo || cw[1] > vhi) return 0;
  f->version = cw[1];
  f->fps = cw[2];
  return 1;
}

int ob_fmt_decode(const float *q, int cells, int bytes, int vlo, int vhi, ob_fmt_t *f) {
  const int nroots = bytes - OB_FMT_DATA;
  if (bytes < OB_FMT_BYTES || bytes > OB_FMT_BYTES_MAX || cells < 8 * ((bytes + 3) / 4)) return -1;
  if (vlo < 1) vlo = 1;
  if (vhi > OB_FMT_VERSION_MAX) vhi = OB_FMT_VERSION_MAX;
  uint8_t cw[OB_FMT_BYTES_MAX], got[OB_FMT_BYTES_MAX];
  float worst[OB_FMT_BYTES_MAX];
  for (int k = 0; k < bytes; k++) {
    const float *qs = q + (k & 3) * cells + 8 * (k >> 2);
    uint8_t v = 0;
    float lo = 1;
    for (int b = 0; b < 8; b++) {
      const float s = qs[b], a = s < 0 ? -s : s;
      v = (uint8_t)((v << 1) | (s > 0));
      if (a < lo) lo = a;
    }
    cw[k] = v; worst[k] = lo;
  }
  memcpy(got, cw, (size_t)bytes);
  if (rs_decode(got, bytes, nroots) >= 0 && accept(got, vlo, vhi, f)) return 0;
  // ai: Then the soft values: the least confident bytes go in as erasures, 2 more a step. rs_decode_er refuses any
  // ai: result past 2e + f <= nroots (rs.c solve), which the garbage figures below assume. The last step leaves three
  // ai: roots over its erasures (nroots - 3; every nroots here is odd: 5, 13, 21, 29 in the four rings), except at
  // ai: RS(8, 3), the 32 ring, which goes on to nroots - 1 = 4 and leaves one (2026-09-24): stopping at 2 there
  // ai: lost the reads of two lost sides. Garbage (soft values uniform in -0.5 to 0.5) passes the magic and the
  // ai: version at 1.8e-7 a read with three roots left and 7.8e-6 with one (test/fmt_test.c, exact). A faded side
  // ai: reads near 0 confidence and the steps erase its bytes: two faded sides read in every ring, three in the 96 and
  // ai: 128. Two glared sides are erased the same way (SPEC.md 5.3).
  int order[OB_FMT_BYTES_MAX];
  uint8_t used[OB_FMT_BYTES_MAX] = { 0 };
  for (int ne = 0; ne < bytes; ne++) {
    int pick = -1;
    for (int k = 0; k < bytes; k++) if (!used[k] && (pick < 0 || worst[k] < worst[pick])) pick = k;   // the lower index wins a tie
    used[pick] = 1; order[ne] = pick;
  }
  const int last = bytes == OB_FMT_BYTES ? nroots - 1 : nroots - 3;
  for (int ne = 2; ne <= last; ne += 2) {
    memcpy(got, cw, (size_t)bytes);
    if (rs_decode_er(got, bytes, nroots, order, ne) >= 0 && accept(got, vlo, vhi, f)) return 0;
  }
  return -1;
}
