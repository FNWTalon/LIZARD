#include "develop.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace lizard {

bool rawFormatKnown(int format) { return format == RAW_FMT_P010 || format == RAW_FMT_RAW16 || format == RAW_FMT_RAW10 || format == RAW_FMT_RAW12; }

size_t rawRowBytes(int format, uint32_t w, uint32_t pixelStride) {
  if (format == RAW_FMT_RAW10) return static_cast<size_t>(w + 3) / 4 * 5;
  if (format == RAW_FMT_RAW12) return static_cast<size_t>(w + 1) / 2 * 3;
  return static_cast<size_t>(w) * std::max<uint32_t>(pixelStride, 2);
}

const uint8_t* gammaLut() {
  static uint8_t t[4097];
  static const bool made = [] { for (int i = 0; i <= 4096; i++) t[i] = static_cast<uint8_t>(lround(255.0 * pow(i / 4096.0, 1 / 2.2))); return true; }();
  (void)made;
  return t;
}

void rawScales(const RawFrame& r, float k[4]) {
  const float* L = r.levels;
  for (int p = 0; p < 4; p++) k[p] = L[5 + p] * 4096.f / std::max(1.f, L[4] - L[p]);
}

namespace {
// ai: sample x of a row in each layout (rawRowBytes' note)
template <int F> inline int rawAt(const uint8_t* row, uint32_t x, uint32_t ps) {
  if (F == RAW_FMT_RAW16) { const uint8_t* p = row + static_cast<size_t>(x) * ps; return p[0] | (p[1] << 8); }
  if (F == RAW_FMT_RAW10) { const uint8_t* g = row + static_cast<size_t>(x >> 2) * 5; const int k = x & 3; return (g[k] << 2) | ((g[4] >> (2 * k)) & 3); }
  const uint8_t* g = row + static_cast<size_t>(x >> 1) * 3;
  return (x & 1) ? ((g[1] << 4) | (g[2] >> 4)) : ((g[0] << 4) | (g[2] & 15));
}

template <int F> void developRaw(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint8_t* out, size_t outStride) {
  const uint8_t* lut = gammaLut();
  const float* L = r.levels;
  float k[4];
  rawScales(r, k);
  const uint32_t ps = r.pixelStride;
  auto lin = [&](int v, int p) { const float f = (v - L[p]) * k[p]; return f <= 0 ? 0.f : f >= 4096.f ? 4096.f : f; };
  for (uint32_t y = 0; y < h; y++) {
    uint8_t* o = out + y * outStride;
    if (r.bin == 2) {
      const uint32_t ry = r.y0 + 2 * (oy + y);
      const uint8_t* r0 = r.plane + static_cast<size_t>(ry) * r.rowStride;
      const uint8_t* r1 = r0 + r.rowStride;
      const int p0 = (ry & 1) << 1, p1 = ((ry + 1) & 1) << 1;
      for (uint32_t x = 0; x < w; x++) {
        const uint32_t rx = r.x0 + 2 * (ox + x);
        const int q0 = rx & 1, q1 = (rx + 1) & 1;
        const float v = lin(rawAt<F>(r0, rx, ps), p0 | q0) + lin(rawAt<F>(r0, rx + 1, ps), p0 | q1) +
                        lin(rawAt<F>(r1, rx, ps), p1 | q0) + lin(rawAt<F>(r1, rx + 1, ps), p1 | q1);
        o[x] = lut[static_cast<int>(v * 0.25f)];
      }
    } else {
      const uint32_t ry = r.y0 + oy + y;
      const uint8_t* row = r.plane + static_cast<size_t>(ry) * r.rowStride;
      const int pr = (ry & 1) << 1;
      for (uint32_t x = 0; x < w; x++) {
        const uint32_t rx = r.x0 + ox + x;
        o[x] = lut[static_cast<int>(lin(rawAt<F>(row, rx, ps), pr | (rx & 1)))];
      }
    }
  }
}

// ai: The crop's staged region: its first sample column of the plane (even; a RAW10 group's first), the crop's first
// ai: sample in it, samples and rows, where its bytes start in a plane's row, and a staged row's bytes and pitch.
struct Region { uint32_t xa, off, n, rows, sy0; size_t from, bytes, pitch; };
Region regionOf(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h) {
  const uint32_t e = r.format == RAW_FMT_P010 ? 1 : r.bin;
  Region g;
  const uint32_t sx0 = r.x0 + ox * e;
  g.sy0 = r.y0 + oy * e;
  g.xa = sx0 / (r.format == RAW_FMT_RAW10 ? 4 : 2) * (r.format == RAW_FMT_RAW10 ? 4 : 2);
  g.off = sx0 - g.xa;
  g.n = w * e;
  g.rows = h * e;
  g.from = r.format == RAW_FMT_RAW10 ? static_cast<size_t>(g.xa) / 4 * 5 : r.format == RAW_FMT_RAW12 ? static_cast<size_t>(g.xa) / 2 * 3 : static_cast<size_t>(g.xa) * r.pixelStride;
  g.bytes = rawRowBytes(r.format, g.off + g.n, r.pixelStride);
  g.pitch = (g.bytes + 3) & ~static_cast<size_t>(3);
  return g;
}
}  // namespace

void developLuma(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint8_t* out, size_t outStride) {
  switch (r.format) {
    case RAW_FMT_P010:
      for (uint32_t y = 0; y < h; y++) {
        const uint8_t* row = r.plane + static_cast<size_t>(r.y0 + oy + y) * r.rowStride + static_cast<size_t>(r.x0 + ox) * r.pixelStride + 1;
        uint8_t* o = out + y * outStride;
        for (uint32_t x = 0; x < w; x++) o[x] = row[static_cast<size_t>(x) * r.pixelStride];
      }
      break;
    case RAW_FMT_RAW16: developRaw<RAW_FMT_RAW16>(r, ox, oy, w, h, out, outStride); break;
    case RAW_FMT_RAW10: developRaw<RAW_FMT_RAW10>(r, ox, oy, w, h, out, outStride); break;
    case RAW_FMT_RAW12: developRaw<RAW_FMT_RAW12>(r, ox, oy, w, h, out, outStride); break;
    default: break;
  }
}

size_t rawStageBytes(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h) {
  const Region g = regionOf(r, ox, oy, w, h);
  return RAW_STAGE_HEADER * 4 + g.pitch * g.rows;
}

size_t stageRaw(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint8_t* dst) {
  const Region g = regionOf(r, ox, oy, w, h);
  uint32_t hd[RAW_STAGE_HEADER] = {};
  hd[0] = r.format == RAW_FMT_P010 ? 0 : r.format == RAW_FMT_RAW16 ? 1 : r.format == RAW_FMT_RAW10 ? 2 : 3;
  hd[1] = w;
  hd[2] = h;
  hd[3] = r.format == RAW_FMT_P010 ? 1 : r.bin;
  hd[4] = static_cast<uint32_t>(g.pitch);
  hd[5] = r.pixelStride;
  hd[6] = g.off;
  hd[7] = g.sy0 & 1;
  float k[4];
  rawScales(r, k);
  memcpy(hd + 8, r.levels, 16);
  memcpy(hd + 12, k, 16);
  memcpy(dst, hd, sizeof hd);
  uint8_t* o = dst + sizeof hd;
  for (uint32_t y = 0; y < g.rows; y++, o += g.pitch) {
    memcpy(o, r.plane + static_cast<size_t>(g.sy0 + y) * r.rowStride + g.from, g.bytes);
    if (g.pitch > g.bytes) memset(o + g.bytes, 0, g.pitch - g.bytes);
  }
  return sizeof hd + g.pitch * g.rows;
}

}  // namespace lizard
