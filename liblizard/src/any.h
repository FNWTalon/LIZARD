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
  // ai: a picture codec a (ring, sub-channel count): each count's rate profile is its own (focus.h focus_tiers_for), so
  // ai: a frame decodes on its version's codec, not as a prefix of its picture's top
  int ready, nmax, pic_ring[FOCUS_ANY_PICS], pic_n[FOCUS_ANY_PICS], pic_subch[FOCUS_ANY_PICS], pic_used[FOCUS_ANY_PICS], pic_clock;
  int which;               // the ring the last frame registered in, -1 none
  int n;                   // the picture it was finished at, 0 none
  int held;                // whether the held configuration stood in for its word
  int builds;              // pictures built since the init (a caller timing its first frames)
  const focus_t *last;     // the codec that finished it
  int version;             // ai: the version it was finished at (its word's, or the held one), 0 none
} focus_any_t;

// ai: a: zeroed, or a receiver to set up again. nmax: the largest picture it decodes (1024 by default, 1536 for an
// ai: industrial rig). Returns the block bytes (473), or -1.
int focus_any_init(focus_any_t *a, int nmax);
void focus_any_free(focus_any_t *a);
// ai: One frame. blocks and ok hold focus_any_top(a) blocks. held: the sub-channel count / 8 of the last word the
// ai: caller read (its version field), 0 for none. Returns the blocks decoded; a->which, a->n and a->held say what
// ai: happened, res what the registration found.
int focus_any_frame(focus_any_t *a, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, int held, ob_result_t *res);
// ai: The most blocks a frame this receiver finishes can hold: the most any sub-channel count whose picture fits nmax
// ai: carries under the format's rate profile (focus.h focus_blocks_for).
int focus_any_top(const focus_any_t *a);
// ai: The word the last frame's ring read, or 0: the held configuration is the caller's, never reported as read.
const ob_fmt_t *focus_any_word(const focus_any_t *a);
// ai: The pilots the last frame's finish read (focus.h focus_pilot) over the blocks its version carries: r of the
// ai: even blocks and of the odd, their standard errors, the blocks read; 0 when the frame was not finished.
int focus_any_pilot(const focus_any_t *a, float r[2], float sd[2]);
#endif
