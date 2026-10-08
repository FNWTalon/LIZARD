// ai: Layer 1, the format's arithmetic (lizard.h): a format's geometry as focus_init paints it, and the room a symbol
// ai: needs and the largest that fits one, ported from the web sender's (sim/lizard_pick.mjs ROOM_FOR, pickVersion;
// ai: two codes' gap as lizard-web/send.mjs gapFrac and desktop Pick.kt). The library is where this arithmetic lives now;
// ai: tests/pick_ref.mjs holds it to the JS.
#include <algorithm>
#include <cmath>

#include "api.hpp"

extern "C" {
#include "focus.h"
}

namespace liz {

static constexpr int OB_MARGIN = 15;   // ai: the border's depth in modules (the 12-module corner mark and FOCUS_CP's 3)
static constexpr double T_DISPLAY_CYCLE = 2.7, PI = 3.14159265358979323846;

int ringIndex(int ring) {
  if (ring == LIZ_RING_DEFAULT) return FOCUS_RING_DEFAULT;
  need(ring >= 0 && ring < FOCUS_RINGS, "ring is 0 to 3 or LIZ_RING_DEFAULT, not " + std::to_string(ring));
  return ring;
}
static int spanOf(int r) { return 2 * FOCUS_RING[r]; }
static int modulesOf(int r) { return spanOf(r) + 2 * OB_MARGIN; }
static void needBlocks(int blocks) { need(blocks >= 1 && blocks <= LIZ_MAX_BLOCKS, "blocks is 1 to 128, not " + std::to_string(blocks)); }

liz_geometry geometryOf(const liz_format& f) {
  needBlocks(f.blocks);
  need(f.fps >= 1 && f.fps <= 255, "fps is 1 to 255, not " + std::to_string(f.fps));
  need(f.codes == 1 || f.codes == 2, "codes is 1 or 2, not " + std::to_string(f.codes));
  const int r = ringIndex(f.ring);
  liz_geometry g{};
  g.n = focus_n_for(8 * f.blocks);
  g.span = spanOf(r);
  g.pxm = int(std::ceil(double(g.n) / g.span - 1e-4));
  g.side = (modulesOf(r) + 2 * FOCUS_QUIET) * g.pxm;
  g.gap = f.codes > 1 ? LIZ_GAP_MODULES * g.pxm : 0;
  g.width = f.codes * g.side + (f.codes - 1) * g.gap;
  g.height = g.side;
  g.frame_blocks = f.codes * focus_blocks_for(8 * f.blocks);
  return g;
}

// ai: ROOM_FOR: the symbol's width over the picture's, times 2.7 display pixels a cycle of the top ring's radius
static double roomFor(int blocks, int r) {
  const double radius = std::sqrt(2.0 * FOCUS_SUB * 8 * blocks / PI);
  return double(modulesOf(r) + 2 * FOCUS_QUIET) / spanOf(r) * T_DISPLAY_CYCLE * radius;
}

}  // namespace liz

using namespace liz;

extern "C" {

LIZ_API int liz_ring_cells(int ring) { return guard([&] { return FOCUS_RING[ringIndex(ring)]; }); }

LIZ_API int liz_geometry_of(const liz_format* format, liz_geometry* out) {
  return guard([&] {
    need(format && out, "a format and a place for its geometry");
    *out = geometryOf(*format);
    return LIZ_OK;
  });
}

LIZ_API double liz_room_for(int blocks, int ring) {
  double v = 0;
  const int rc = guard([&] { needBlocks(blocks); v = roomFor(blocks, ringIndex(ring)); return LIZ_OK; });
  return rc < 0 ? rc : v;
}

LIZ_API int liz_blocks_for(int blocks) { return blocks >= 1 && blocks <= LIZ_MAX_BLOCKS ? focus_blocks_for(8 * blocks) : 0; }

LIZ_API int liz_pick(double w, double h, int codes, int ring, int top) {
  return guard([&] {
    need(codes == 1 || codes == 2, "codes is 1 or 2");
    needBlocks(top);
    const int r = ringIndex(ring);
    const double gapFrac = codes > 1 ? (codes - 1) * double(LIZ_GAP_MODULES) / (modulesOf(r) + 2 * FOCUS_QUIET) : 0;
    const double room = std::max(64.0, std::min(w / (codes + gapFrac), h));
    int best = 1;
    for (int b = 1; b <= top; b++) if (roomFor(b, r) <= room) best = b;
    return best;
  });
}

}  // extern "C"
