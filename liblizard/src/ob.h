#ifndef OB_H
#define OB_H
#include "layout.h"

// What a decode found, which the JS reads at fixed offsets (sim/ob.mjs, the wasm's ob_last): found 0, quad 8,
// ms_detect 52, ms_sample 56, ms_decode 60, mark 64 (since 2026-10-25, when the binary grid code's ok, iters, ber and
// gmi left it; ms_detect was at 180). tiles_ok is the count of verified blocks.
typedef struct {
  int found;                  // 0 no symbol, 1 registered
  int finders;                // finder candidates seen
  float quad[8];              // the border's corners in the image, logical TL TR BR BL
  int orient;                 // rotation 0..3, +4 when mirrored
  float mark_score;           // mean correlation peak over the mesh nodes
  int tiles_ok;
  float ms_detect, ms_sample, ms_decode;
  // The best-supported corner mark the frame finder saw: x, y in image pixels and the module size it implies, or all
  // zero. One mark cannot register a symbol, but it says where one is and how big, which is what a crop needs.
  float mark[3];
} ob_result_t;

#endif
