// ai: The blind receiver (any.h).
#include "any.h"
#include <stddef.h>

// ai: The largest sub-channel count whose picture is n: the top of n (the ring codecs' format, which registers any).
static int top_subch(int n) {
  for (int s = FOCUS_GROUP * OB_FMT_VERSION_MAX; s >= FOCUS_GROUP; s -= FOCUS_GROUP) if (focus_n_for(s) == n) return s;
  return 0;
}
void focus_any_free(focus_any_t *a) {
  for (int r = 0; r < FOCUS_RINGS; r++) focus_free(&a->ring[r]);
  for (int i = 0; i < FOCUS_ANY_PICS; i++) { if (a->pic_n[i]) focus_free(&a->pic[i]); a->pic_n[i] = 0; }
  a->ready = 0; a->which = -1; a->last = 0; a->builds = 0;
}
int focus_any_init(focus_any_t *a, int nmax) {
  focus_any_free(a);
  if (nmax < 256 || nmax > 2048) return -1;
  for (int r = 0; r < FOCUS_RINGS; r++) {
    if (focus_init(&a->ring[r], 256, top_subch(256), FOCUS_LDPC, 2, 2 * FOCUS_RING[r], 0, 0, 0, 0, 0, 0, 0)) { for (int q = 0; q < r; q++) focus_free(&a->ring[q]); return -1; }
  }
  a->nmax = nmax; a->ready = 1;
  return a->ring[0].block_bytes;
}
// ai: The codec for subch sub-channels (picture n) in ring r: the ring's own where it is that format (n = 256 at its
// ai: top), else the cache, else built now. Each count is a format of its own (its rate profile), not a prefix.
static focus_t *pic_for(focus_any_t *a, int r, int n, int subch) {
  if (n == 256 && subch == top_subch(256)) return &a->ring[r];
  int slot = -1, oldest = 0;
  for (int i = 0; i < FOCUS_ANY_PICS; i++) {
    if (a->pic_n[i] == n && a->pic_subch[i] == subch && a->pic_ring[i] == r) { a->pic_used[i] = ++a->pic_clock; return &a->pic[i]; }
    if (a->pic_used[i] < a->pic_used[oldest]) oldest = i;
    if (!a->pic_n[i] && slot < 0) slot = i;
  }
  if (slot < 0) { slot = oldest; focus_free(&a->pic[slot]); a->pic_n[slot] = 0; }
  if (focus_init(&a->pic[slot], n, subch, FOCUS_LDPC, 2, 2 * FOCUS_RING[r], 0, 0, 0, 0, 0, 0, 0)) return 0;
  a->pic_n[slot] = n; a->pic_subch[slot] = subch; a->pic_ring[slot] = r; a->pic_used[slot] = ++a->pic_clock; a->builds++;
  return &a->pic[slot];
}
int focus_any_frame(focus_any_t *a, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, int held, ob_result_t *res) {
  const focus_t *rs[FOCUS_RINGS];
  for (int r = 0; r < FOCUS_RINGS; r++) rs[r] = &a->ring[r];
  int wi;
  a->which = -1; a->n = 0; a->held = 0; a->last = 0; a->version = 0;
  if (!a->ready || !focus_acquire_ring(rs, FOCUS_RINGS, img, iw, ih, gamma, mesh, res, &wi)) return 0;
  a->which = wi;
  const ob_fmt_t *fm = focus_fmt_rx(&a->ring[wi]);
  const int version = fm ? fm->version : held;
  a->held = !fm && held > 0;
  const int n = version >= 1 && version <= OB_FMT_VERSION_MAX ? focus_n_for(FOCUS_GROUP * version) : 0;
  focus_t *pic = n && n <= a->nmax ? pic_for(a, wi, n, FOCUS_GROUP * version) : 0;
  if (!pic || (pic != &a->ring[wi] && !focus_hand_over(&a->ring[wi], pic))) { focus_release(&a->ring[wi]); return 0; }
  a->last = pic; a->n = n; a->version = version;
  return focus_finish(pic, NULL, blocks, ok, res);
}
int focus_any_top(const focus_any_t *a) {
  if (!a->ready) return 0;
  int top = 0;
  for (int s = FOCUS_GROUP; s <= FOCUS_GROUP * OB_FMT_VERSION_MAX; s += FOCUS_GROUP) if (focus_n_for(s) <= a->nmax && focus_blocks_for(s) > top) top = focus_blocks_for(s);
  return top;
}
const ob_fmt_t *focus_any_word(const focus_any_t *a) { return a->which >= 0 ? focus_fmt_rx(&a->ring[a->which]) : 0; }
int focus_any_pilot(const focus_any_t *a, float r[2], float sd[2]) {
  if (!a->last) { if (r) r[0] = r[1] = 0; if (sd) sd[0] = sd[1] = 0; return 0; }
  return focus_pilot(a->last, 0, r, sd);   // ai: the version's own codec: every block it holds
}
