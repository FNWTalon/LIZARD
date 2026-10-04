#ifndef OB_H
#define OB_H
#include "layout.h"

// blocks: tiles * block_bytes, tile order. modules: w * h, 1 = dark.
void ob_encode(const ob_layout_t *L, const uint8_t *blocks, uint8_t *modules);

enum { OB_EQ_NONE = 0, OB_EQ_BLIND = 1, OB_EQ_PILOT = 2, OB_EQ_GENIE = 3 };

typedef struct {
  float gamma;        // camera transfer curve assumed when linearizing, 1 = none
  int eq_mode;        // OB_EQ_*: where the equalizer's training targets come from
  int eq_taps;        // 3 or 5
  int eq_regions;     // 1 = one filter for the frame, n = n x n filters blended across it
  int mesh;           // 0 = global homography only
  int max_iter;
  float llr_gain;     // scale applied to the blind LLR estimate
  float llr_clip;     // natural-log units
} ob_opts_t;
void ob_default_opts(ob_opts_t *o);

typedef struct {
  int found;                  // 0 no symbol, 1 registered
  int finders;                // finder candidates seen
  float quad[8];              // finder centres in the image, logical TL TR BR BL
  int orient;                 // rotation 0..3, +4 when mirrored
  float mark_score;           // mean correlation peak over the mesh nodes
  int tiles_ok;
  uint8_t ok[OB_MAX_TILES];
  int8_t iters[OB_MAX_TILES];
  float ms_detect, ms_sample, ms_decode;
  // Filled when the caller passes the transmitted modules: raw BER and information per cell.
  float ber, gmi;
  // The best-supported corner or centre mark the frame finder saw: x, y in image pixels and the module size it
  // implies, or all zero. One mark cannot register a symbol, but it says where one is and how big, which is what
  // a crop needs (archive/build-scratch-2026-09/crop_ceiling.mjs). Appended last so no reader's offsets move.
  float mark[3];
} ob_result_t;

// img: w * h 8-bit luminance. blocks receives tiles * block_bytes; only tiles flagged ok are
// valid. truth (optional, w * h modules) turns on the channel diagnostics in the result.
int ob_decode(const ob_layout_t *L, const uint8_t *img, int iw, int ih, const ob_opts_t *o,
              uint8_t *blocks, ob_result_t *res, const uint8_t *truth);

#endif
