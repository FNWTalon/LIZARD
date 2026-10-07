// ai: The blind receiver (any.h).
#include "any.h"
#include <stddef.h>

// ai: The largest sub-channel count whose picture is n: the top of n, which every smaller count at n is a prefix of.
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
  a->tiers = 0; a->tier_subch = 0; a->tier_blocks = 0;
  return a->ring[0].block_bytes;
}
int focus_any_tiers(focus_any_t *a, const focus_tier_t *tier, int tiers) {
  int subch = 0, blocks = 0;
  if (tiers < 0 || tiers > FOCUS_TIERS) return -1;
  for (int t = 0; t < tiers; t++) {
    if (tier[t].rate < 0 || tier[t].rate >= LDPC_RATES || tier[t].blocks < 1 || tier[t].subs < 1) return -1;
    subch += tier[t].blocks * tier[t].subs; blocks += tier[t].blocks;
  }
  // ai: the word names the frame's sub-channels as a version, so the profile must sum to a whole number of 8
  if (tiers && (subch % FOCUS_GROUP || subch > FOCUS_GROUP * OB_FMT_VERSION_MAX)) return -1;
  for (int i = 0; i < FOCUS_ANY_PICS; i++) { if (a->pic_n[i]) focus_free(&a->pic[i]); a->pic_n[i] = 0; a->pic_tiered[i] = 0; }
  for (int t = 0; t < tiers; t++) a->tier[t] = tier[t];
  a->tiers = tiers; a->tier_subch = tiers ? subch : 0; a->tier_blocks = tiers ? blocks : 0;
  return 0;
}
// ai: The fewest sub-channels a block whose code at this rate holds 473 B (the transfer's block, xfer.h), and that
// ai: code's k: the codes built and dropped. 0 where none up to 32 does.
static int subs_for_rate(int rate, int *k_out) {
  for (int subs = 1; subs <= 32; subs++) {
    ldpc_t c;
    if (ldpc_init(&c, subs * FOCUS_SUB * 2, rate, 1)) continue;
    const int k = c.k;
    ldpc_free(&c);
    if (k >= 8 * (473 + 4)) { *k_out = k; return subs; }
  }
  return 0;
}
static void say(char *label, int label_len, const char *text) {
  if (!label || label_len <= 0) return;
  int i = 0;
  for (; text[i] && i + 1 < label_len; i++) label[i] = text[i];
  label[i] = 0;
}
int focus_tiers_parse(const char *s, focus_tier_t *tier, int *subch, char *label, int label_len) {
  int tiers = 0, total = 0, len = 0, kmin = 1 << 30;
  if (label && label_len > 0) label[0] = 0;
  if (!s) { say(label, label_len, "no profile"); return 0; }
  for (const char *p = s; *p;) {
    while (*p == ' ' || *p == ',' || *p == ';') p++;
    if (!*p) break;
    if (tiers == FOCUS_TIERS) { say(label, label_len, "more than three tiers"); return 0; }
    // ai: the rate: a name of ldpc_rate_name, else an index
    int rate = -1;
    for (int r = 0; r < LDPC_RATES && rate < 0; r++) {
      const char *nm = ldpc_rate_name[r];
      int i = 0;
      while (nm[i] && p[i] == nm[i]) i++;
      if (!nm[i] && p[i] == ':') { rate = r; p += i; }
    }
    if (rate < 0 && *p >= '0' && *p <= '9' && p[1] == ':') { rate = *p - '0'; p++; }
    if (rate < 0 || rate >= LDPC_RATES || *p != ':') { say(label, label_len, "a tier is <rate>:<blocks>, the rate 1/4, 1/3, 1/2, 2/3, 3/4, 5/6 or 7/8"); return 0; }
    p++;
    int blocks = 0, digits = 0;
    while (*p >= '0' && *p <= '9') { blocks = blocks * 10 + (*p - '0'); p++; digits++; if (blocks > 1024) break; }
    if (!digits || blocks < 1 || blocks > 1024 || (*p && *p != ' ' && *p != ',' && *p != ';')) { say(label, label_len, "a tier's blocks are 1 to 1024"); return 0; }
    int k = 0;
    const int subs = subs_for_rate(rate, &k);
    if (!subs) { say(label, label_len, "no block of 473 B at that rate"); return 0; }
    if (k < kmin) kmin = k;
    tier[tiers].rate = rate; tier[tiers].blocks = blocks; tier[tiers].subs = subs;
    total += blocks * subs;
    if (label && label_len > 0) {
      const char *nm = ldpc_rate_name[rate];
      char num[16];
      int k = 0, b = blocks;
      do { num[k++] = (char)('0' + b % 10); b /= 10; } while (b);
      if (tiers) { if (len + 2 < label_len) { label[len++] = ','; label[len++] = ' '; } }
      for (int i = 0; nm[i] && len + 1 < label_len; i++) label[len++] = nm[i];
      if (len + 3 < label_len) { label[len++] = ' '; label[len++] = 'x'; label[len++] = ' '; }
      while (k && len + 1 < label_len) label[len++] = num[--k];
      label[len] = 0;
    }
    tiers++;
  }
  if (!tiers) { say(label, label_len, "no tier named"); return 0; }
  if (total % FOCUS_GROUP) { say(label, label_len, "the tiers' sub-channels must sum to a multiple of 8 (the word's version)"); return 0; }
  if (total > FOCUS_GROUP * OB_FMT_VERSION_MAX) { say(label, label_len, "more sub-channels than LIZARD-1024"); return 0; }
  // ai: the block is the smallest tier's k, whole bytes less the id (focus.c init), and a receiver's buffers stride
  // ai: at the transfer's 473 B: a 3/4 tier (k 3816) holds it there, every other rate's k is 3840 or more
  if (kmin / 8 - 4 != 473) { say(label, label_len, "the profile's block is not 473 B: a 3/4 tier holds it there"); return 0; }
  if (subch) *subch = total;
  return tiers;
}
// ai: The codec for picture n in ring r: the ring's own at n = 256 (one rate), else the cache, else built now; tiered,
// ai: the profile's codec (focus_any_tiers), cached apart from the one rate's at the same n.
static focus_t *pic_for(focus_any_t *a, int r, int n, int tiered) {
  if (n == 256 && !tiered) return &a->ring[r];
  int slot = -1, oldest = 0;
  for (int i = 0; i < FOCUS_ANY_PICS; i++) {
    if (a->pic_n[i] == n && a->pic_ring[i] == r && a->pic_tiered[i] == tiered) { a->pic_used[i] = ++a->pic_clock; return &a->pic[i]; }
    if (a->pic_used[i] < a->pic_used[oldest]) oldest = i;
    if (!a->pic_n[i] && slot < 0) slot = i;
  }
  if (slot < 0) { slot = oldest; focus_free(&a->pic[slot]); a->pic_n[slot] = 0; }
  const int bad = tiered ? focus_init_tiers(&a->pic[slot], n, a->tier, a->tiers, 2, 2 * FOCUS_RING[r], 0, 0, 0, 0, 0, 0, 0)
                         : focus_init(&a->pic[slot], n, top_subch(n), FOCUS_LDPC, 2, 2 * FOCUS_RING[r], 0, 0, 0, 0, 0, 0, 0);
  if (bad) return 0;
  a->pic_n[slot] = n; a->pic_ring[slot] = r; a->pic_tiered[slot] = tiered; a->pic_used[slot] = ++a->pic_clock; a->builds++;
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
  // ai: the profile's frames are the ones whose word names its sub-channels (focus_any_tiers)
  const int tiered = a->tiers > 0 && FOCUS_GROUP * version == a->tier_subch;
  focus_t *pic = n && n <= a->nmax ? pic_for(a, wi, n, tiered) : 0;
  if (!pic || (pic != &a->ring[wi] && !focus_hand_over(&a->ring[wi], pic))) { focus_release(&a->ring[wi]); return 0; }
  a->last = pic; a->n = n; a->version = version;
  return focus_finish(pic, NULL, blocks, ok, res);
}
int focus_any_top(const focus_any_t *a) {
  if (!a->ready) return 0;
  int top = 0;
  for (int s = FOCUS_GROUP * OB_FMT_VERSION_MAX; s >= FOCUS_GROUP; s -= FOCUS_GROUP) if (focus_n_for(s) <= a->nmax) { top = s / FOCUS_GROUP; break; }
  // ai: a profile of blocks smaller than 8 sub-channels holds more blocks than its sub-channels / 8
  if (a->tiers > 0 && a->tier_blocks > top && focus_n_for(a->tier_subch) <= a->nmax) top = a->tier_blocks;
  return top;
}
const ob_fmt_t *focus_any_word(const focus_any_t *a) { return a->which >= 0 ? focus_fmt_rx(&a->ring[a->which]) : 0; }
int focus_any_pilot(const focus_any_t *a, float r[2], float sd[2]) {
  if (!a->last) { if (r) r[0] = r[1] = 0; if (sd) sd[0] = sd[1] = 0; return 0; }
  return focus_pilot(a->last, a->version, r, sd);
}
