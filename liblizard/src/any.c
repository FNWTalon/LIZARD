// ai: The blind receiver (any.h).
#include "any.h"
#include <stddef.h>

// ai: The largest sub-channel count whose picture is n: the top of n (the ring codecs' format, which registers any).
static int top_subch(int n) {
  for (int s = LIZARD_STEP * OB_FMT_VERSION_MAX; s >= LIZARD_STEP * OB_FMT_VERSION_MIN; s -= LIZARD_STEP) if (lizard_n_for(s) == n) return s;
  return 0;
}
// ai: The ring codecs' picture: the smallest a format has (the ladder's foot's, 384 since LIZARD-32 left it).
static int ring_n(void) { return lizard_n_for(LIZARD_STEP * OB_FMT_VERSION_MIN); }
void lizard_any_free(lizard_any_t *a) {
  for (int r = 0; r < LIZARD_RINGS; r++) lizard_free(&a->ring[r]);
  for (int i = 0; i < LIZARD_ANY_PICS; i++) { if (a->pic_n[i]) lizard_free(&a->pic[i]); a->pic_n[i] = 0; }
  a->ready = 0; a->which = -1; a->last = 0; a->builds = 0;
}
int lizard_any_init(lizard_any_t *a, int nmax) {
  lizard_any_free(a);
  if (nmax < ring_n() || nmax > 2048) return -1;
  for (int r = 0; r < LIZARD_RINGS; r++) {
    if (lizard_init(&a->ring[r], ring_n(), top_subch(ring_n()), 2 * LIZARD_RING[r])) { for (int q = 0; q < r; q++) lizard_free(&a->ring[q]); return -1; }
  }
  a->nmax = nmax; a->ready = 1;
  return a->ring[0].block_bytes;
}
// ai: The codec for subch sub-channels (picture n) in ring r: the ring's own where it is that format (n = ring_n() at its
// ai: top), else the cache, else built now. Each count is a format of its own (its rate profile), not a prefix.
static lizard_t *pic_for(lizard_any_t *a, int r, int n, int subch) {
  if (n == ring_n() && subch == top_subch(n)) return &a->ring[r];
  int slot = -1, oldest = 0;
  for (int i = 0; i < LIZARD_ANY_PICS; i++) {
    if (a->pic_n[i] == n && a->pic_subch[i] == subch && a->pic_ring[i] == r) { a->pic_used[i] = ++a->pic_clock; return &a->pic[i]; }
    if (a->pic_used[i] < a->pic_used[oldest]) oldest = i;
    if (!a->pic_n[i] && slot < 0) slot = i;
  }
  if (slot < 0) { slot = oldest; lizard_free(&a->pic[slot]); a->pic_n[slot] = 0; }
  if (lizard_init(&a->pic[slot], n, subch, 2 * LIZARD_RING[r])) return 0;
  a->pic_n[slot] = n; a->pic_subch[slot] = subch; a->pic_ring[slot] = r; a->pic_used[slot] = ++a->pic_clock; a->builds++;
  return &a->pic[slot];
}
int lizard_any_frame(lizard_any_t *a, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, int held, ob_result_t *res) {
  const lizard_t *rs[LIZARD_RINGS];
  for (int r = 0; r < LIZARD_RINGS; r++) rs[r] = &a->ring[r];
  int wi;
  a->which = -1; a->n = 0; a->held = 0; a->last = 0;
  if (!a->ready || !lizard_acquire_ring(rs, LIZARD_RINGS, img, iw, ih, gamma, mesh, res, &wi)) return 0;
  a->which = wi;
  const ob_fmt_t *fm = lizard_fmt_rx(&a->ring[wi]);
  const int version = fm ? fm->version : held, known = version >= OB_FMT_VERSION_MIN && version <= OB_FMT_VERSION_MAX;
  a->held = !fm && known;
  const int n = known ? lizard_n_for(LIZARD_STEP * version) : 0;
  lizard_t *pic = n && n <= a->nmax ? pic_for(a, wi, n, LIZARD_STEP * version) : 0;
  if (!pic || (pic != &a->ring[wi] && !lizard_hand_over(&a->ring[wi], pic))) { lizard_release(&a->ring[wi]); return 0; }
  a->last = pic; a->n = n;
  return lizard_finish(pic, NULL, blocks, ok, res);
}
int lizard_any_top(const lizard_any_t *a) {
  if (!a->ready) return 0;
  int top = 0;
  for (int s = LIZARD_STEP * OB_FMT_VERSION_MIN; s <= LIZARD_STEP * OB_FMT_VERSION_MAX; s += LIZARD_STEP) if (lizard_n_for(s) <= a->nmax && lizard_blocks_for(s) > top) top = lizard_blocks_for(s);
  return top;
}
const ob_fmt_t *lizard_any_word(const lizard_any_t *a) { return a->which >= 0 ? lizard_fmt_rx(&a->ring[a->which]) : 0; }
int lizard_any_pilot(const lizard_any_t *a, float r[2], float sd[2]) {
  if (!a->last) { if (r) r[0] = r[1] = 0; if (sd) sd[0] = sd[1] = 0; return 0; }
  return lizard_pilot(a->last, 0, r, sd);   // ai: the version's own codec: every block it holds
}
