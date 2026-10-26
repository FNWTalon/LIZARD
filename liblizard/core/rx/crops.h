// ai: The crops of a camera frame, the one rule for the receiver (receiver.cpp, ReceiverConfig.layout) and the C API
// ai: (liz_layout_rects): 1:1 the centre square; 2:1 the centre region twice as long as it is high along the frame's
// ai: long side, its two halves (each a square, each read as a frame).
#pragma once
#include <algorithm>
#include <cstdint>

namespace lizard {

struct Crops { int n; uint32_t side, x[2], y[2]; };

inline Crops cropsOf(uint32_t w, uint32_t h, bool two) {
  Crops c{};
  const bool wide = w >= h;
  const uint32_t lo = std::min(w, h), hi = std::max(w, h);
  c.side = two ? std::min(lo, hi / 2) : lo;
  c.n = two ? 2 : 1;
  for (int k = 0; k < c.n; k++) {
    const uint32_t along = (hi - c.n * c.side) / 2 + k * c.side, across = (lo - c.side) / 2;
    c.x[k] = wide ? along : across;
    c.y[k] = wide ? across : along;
  }
  return c;
}

}  // namespace lizard
