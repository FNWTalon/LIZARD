// ai: A receiver told nothing, as an object (focus.h focus_acquire_ring; since 2026-09-27 the ring bootstraps what is
// ai: inside). A codec a ring, each at n = 256, the smallest picture every ring holds: the finder
// ai: registers against their layouts and the one that registered reads the word. The word names the sub-channels, so
// ai: the picture n; the frame is finished by the codec for that picture in that ring, built the first time the pair
// ai: is seen and kept (FOCUS_ANY_PICS of them, the least recently used replaced), each at the top of its n so every
// ai: smaller sub-channel count is a prefix of it. A frame whose word does not read is finished at the caller's held
// ai: configuration (the last word it read); with
// ai: none held, it is not finished.
// ai: One focus_any_t a thread: a decode keeps its registration inside the codecs it holds, so two threads never
// ai: share one, and the codec's per-thread state (internal.h OB_TLS) is then each thread's own. The wasm holds one
// ai: (wasm.c), behind its focus_any_* exports.
#ifndef OB_ANY_H
#define OB_ANY_H
#include "focus.h"

enum { FOCUS_ANY_PICS = 2 };
typedef struct {
  focus_t ring[FOCUS_RINGS], pic[FOCUS_ANY_PICS];
  int ready, nmax, pic_ring[FOCUS_ANY_PICS], pic_n[FOCUS_ANY_PICS], pic_used[FOCUS_ANY_PICS], pic_clock;
  int which;               // the ring the last frame registered in, -1 none
  int n;                   // the picture it was finished at, 0 none
  int held;                // whether the held configuration stood in for its word
  int builds;              // pictures built since the init (a caller timing its first frames)
  const focus_t *last;     // the codec that finished it
  int version;             // ai: the version it was finished at (its word's, or the held one), 0 none
  // ai: A rate profile (focus_any_tiers; the lab's rate-by-ring arm, 2026-10-07): a frame whose word names tier_subch
  // ai: sub-channels is finished on a codec of these tiers (focus_init_tiers) in place of the one rate; every other
  // ai: frame reads as before. tiers 0: none. tier_blocks: the profile's blocks a frame; pic_tiered: which cached
  // ai: pictures were built with it.
  focus_tier_t tier[FOCUS_TIERS];
  int tiers, tier_subch, tier_blocks, pic_tiered[FOCUS_ANY_PICS];
} focus_any_t;

// ai: a: zeroed, or a receiver to set up again. nmax: the largest picture it decodes (1024 by default, 1536 for an
// ai: industrial rig). Returns the block bytes (473), or -1.
int focus_any_init(focus_any_t *a, int nmax);
void focus_any_free(focus_any_t *a);
// ai: One frame. blocks and ok hold focus_any_top(a) blocks. held: the sub-channel count / 8 of the last word the
// ai: caller read (its version field), 0 for none. Returns the blocks decoded; a->which, a->n and a->held say what
// ai: happened, res what the registration found.
int focus_any_frame(focus_any_t *a, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, int held, ob_result_t *res);
// ai: The most blocks a frame this receiver finishes can hold: the largest sub-channel count whose picture fits nmax,
// ai: which need not be a ladder size.
int focus_any_top(const focus_any_t *a);
// ai: The word the last frame's ring read, or 0: the held configuration is the caller's, never reported as read.
const ob_fmt_t *focus_any_word(const focus_any_t *a);
// ai: A rate profile for the frames whose word names its sub-channel count (the lab's rate-by-ring arm, 2026-10-07;
// ai: a sender painting the same profile, core/tx/sender.h TxFormat.tiers), after focus_any_init; tiers 0 clears it.
// ai: The pictures built so far are dropped. 0, or -1 (refused: the receiver unchanged). The GPU decoder has none.
int focus_any_tiers(focus_any_t *a, const focus_tier_t *tier, int tiers);
// ai: A profile from its text, "<rate>:<blocks>[,<rate>:<blocks>...]", the inner tiers first, a rate by name
// ai: ("7/8", ldpc.h ldpc_rate_name) or index; a tier's sub-channels a block are the fewest whose code at that rate
// ai: holds a block of 473 B (7 at 7/8, 8 at 3/4 and 5/6, 9 at 2/3, 12 at 1/2, 18 at 1/3, 24 at 1/4), and a 3/4 tier is
// ai: among them, since the block (the smallest tier's k less the id) must be the transfer's 473 B. Returns the
// ai: tiers (1 to FOCUS_TIERS) into tier[], the frame's sub-channels into subch (a multiple of 8, so the word names
// ai: it) and the profile's text ("7/8 x 20, 3/4 x 20, 1/2 x 11") into label (label_len bytes); or 0, why it was
// ai: refused in label.
int focus_tiers_parse(const char *s, focus_tier_t *tier, int *subch, char *label, int label_len);
// ai: The pilots the last frame's finish read (focus.h focus_pilot) over the blocks its version carries: r of the
// ai: even blocks and of the odd, their standard errors, the blocks read; 0 when the frame was not finished.
int focus_any_pilot(const focus_any_t *a, float r[2], float sd[2]);
#endif
