#ifndef OB_H
#define OB_H
#include "layout.h"

// What a decode found, which the JS reads at fixed offsets (sim/ob.mjs, the wasm's ob_last): the fields before
// ms_detect keep their sizes so the ones after keep their places. ok, iters, ber and gmi were the binary grid code's
// (deleted 2026-10-09) and are written by nothing now; tiles_ok is LIZARD's count of verified blocks.
typedef struct {
  int found;                  // 0 no symbol, 1 registered
  int finders;                // finder candidates seen
  float quad[8];              // the border's corners in the image, logical TL TR BR BL
  int orient;                 // rotation 0..3, +4 when mirrored
  float mark_score;           // mean correlation peak over the mesh nodes
  int tiles_ok;
  uint8_t ok[64];
  int8_t iters[64];
  float ms_detect, ms_sample, ms_decode;
  float ber, gmi;
  // The best-supported corner or centre mark the frame finder saw: x, y in image pixels and the module size it
  // implies, or all zero. One mark cannot register a symbol, but it says where one is and how big, which is what
  // a crop needs (archive/build-scratch-2026-09/crop_ceiling.mjs). Appended last so no reader's offsets move.
  float mark[3];
} ob_result_t;

#endif
