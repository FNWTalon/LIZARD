// ai: A receiver told nothing, as an object (lizard_phy.h lizard_acquire_ring; since 2026-09-27 the ring bootstraps what is
// ai: inside). A codec a ring, each at the smallest picture a format has (384, the top format there): the finder
// ai: registers against their layouts and the one that registered reads the word. The word names the sub-channels, so
// ai: the picture n; the frame is finished by the codec for that count in that ring, built the first time the pair
// ai: is seen and kept (LIZARD_ANY_PICS of them, the least recently used replaced): each count is a format of its own
// ai: (its rate profile), not a prefix of a larger one. A frame whose word does not read is finished at the caller's
// ai: held configuration (the last word it read); with none held, it is not finished.
// ai: One lizard_any_t a thread: a decode keeps its registration inside the codecs it holds, so two threads never
// ai: share one, and the codec's per-thread state (internal.h OB_TLS) is then each thread's own. The wasm holds one
// ai: (wasm.c), behind its lizard_any_* exports.
#ifndef OB_ANY_H
#define OB_ANY_H
#include "lizard_phy.h"

enum { LIZARD_ANY_PICS = 2 };
typedef struct {
  lizard_t ring[LIZARD_RINGS], pic[LIZARD_ANY_PICS];
  // ai: a picture codec a (ring, sub-channel count): each count's rate profile is its own (lizard_phy.h lizard_tiers_for), so
  // ai: a frame decodes on its version's codec, not as a prefix of its picture's top
  int ready, nmax, pic_ring[LIZARD_ANY_PICS], pic_n[LIZARD_ANY_PICS], pic_subch[LIZARD_ANY_PICS], pic_used[LIZARD_ANY_PICS], pic_clock;
  int which;               // the ring the last frame registered in, -1 none
  int n;                   // the picture it was finished at, 0 none
  int held;                // whether the held configuration stood in for its word
  int builds;              // pictures built since the init (a caller timing its first frames)
  const lizard_t *last;     // the codec that finished it
} lizard_any_t;

// ai: a: zeroed, or a receiver to set up again. nmax: the largest picture it decodes (1536, the ladder's top, in every
// ai: receiver since 2026-09-29; 1024 before). Returns the block bytes (476), or -1.
int lizard_any_init(lizard_any_t *a, int nmax);
void lizard_any_free(lizard_any_t *a);
// ai: One frame. blocks and ok hold lizard_any_top(a) blocks. held: the sub-channel count / 8 of the last word the
// ai: caller read (its version field), 0 for none; a held version no format has (outside OB_FMT_VERSION_MIN ..
// ai: OB_FMT_VERSION_MAX) decodes nothing and is not reported as held. Returns the blocks decoded; a->which, a->n and
// ai: a->held say what happened, res what the registration found.
int lizard_any_frame(lizard_any_t *a, const uint8_t *img, int iw, int ih, float gamma, uint8_t *blocks, uint8_t *ok, int held, ob_result_t *res);
// ai: The most blocks a frame this receiver finishes can hold: the most any sub-channel count whose picture fits nmax
// ai: carries under the format's rate profile (lizard_phy.h lizard_blocks_for).
int lizard_any_top(const lizard_any_t *a);
// ai: The word the last frame's ring read, or 0: the held configuration is the caller's, never reported as read.
const ob_fmt_t *lizard_any_word(const lizard_any_t *a);
// ai: The pilots the last frame's finish read (lizard_phy.h lizard_pilot) over the blocks its version carries: r of the
// ai: even blocks and of the odd, their standard errors, the blocks read; 0 when the frame was not finished.
int lizard_any_pilot(const lizard_any_t *a, float r[2], float sd[2]);
#endif
