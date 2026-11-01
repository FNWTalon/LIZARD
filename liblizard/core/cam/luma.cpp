// ai: luma.h's conversions (2026-10-25): one function a layout, each a pass over the rows through row pointers with
// ai: the layout's arithmetic in its inner loop and no branch a pixel (the 16-bit clamp a select, a Bayer frame's last
// ai: column done once a row, after the loop), so -O3 vectorises the loops.
#include "luma.h"

#include <cstring>

namespace lizard::cam {

namespace {

// ai: Grey8 and the planar YUV layouts: the first plane's rows; one copy where the rows are contiguous
void copyRows(const uint8_t* d, uint32_t w, uint32_t h, uint32_t stride, uint8_t* o) {
  if (stride == w) { std::memcpy(o, d, size_t(w) * h); return; }
  for (uint32_t y = 0; y < h; y++) std::memcpy(o + size_t(y) * w, d + size_t(y) * stride, w);
}

// ai: the packed 4:2:2 layouts: byte `First` of each pixel's two
template <int First>
void everyOther(const uint8_t* d, uint32_t w, uint32_t h, uint32_t stride, uint8_t* o) {
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* r = d + size_t(y) * stride + First;
    uint8_t* q = o + size_t(y) * w;
    for (uint32_t x = 0; x < w; x++) q[x] = r[2 * size_t(x)];
  }
}

// ai: a little-endian sample's top 8 of its significant bits, clamped where bits above them are set
inline uint8_t top8(const uint8_t* s, unsigned shift) {
  const unsigned v = (unsigned(s[0]) | unsigned(s[1]) << 8) >> shift;
  return uint8_t(v < 255u ? v : 255u);
}

void grey16(const uint8_t* d, uint32_t w, uint32_t h, uint32_t stride, unsigned shift, uint8_t* o) {
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* r = d + size_t(y) * stride;
    uint8_t* q = o + size_t(y) * w;
    for (uint32_t x = 0; x < w; x++) q[x] = top8(r + 2 * size_t(x), shift);
  }
}

// ai: Bayer8: the rounded mean of each 2 x 2 window; the last row pairs with itself, the last column too
void bayer8(const uint8_t* d, uint32_t w, uint32_t h, uint32_t stride, uint8_t* o) {
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* r0 = d + size_t(y) * stride;
    const uint8_t* r1 = d + size_t(y + 1 < h ? y + 1 : y) * stride;
    uint8_t* q = o + size_t(y) * w;
    for (uint32_t x = 0; x + 1 < w; x++)
      q[x] = uint8_t((unsigned(r0[x]) + r0[x + 1] + r1[x] + r1[x + 1] + 2) >> 2);
    q[w - 1] = uint8_t((2 * (unsigned(r0[w - 1]) + r1[w - 1]) + 2) >> 2);
  }
}

// ai: the colour layouts: N bytes a pixel, R, G and B at their offsets
template <int N, int R, int G, int B>
void colour(const uint8_t* d, uint32_t w, uint32_t h, uint32_t stride, uint8_t* o) {
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* r = d + size_t(y) * stride;
    uint8_t* q = o + size_t(y) * w;
    for (uint32_t x = 0; x < w; x++) {
      const uint8_t* p = r + N * size_t(x);
      q[x] = uint8_t((77u * p[R] + 150u * p[G] + 29u * p[B] + 128u) >> 8);
    }
  }
}

}  // namespace

bool lumaInPlace(Pix pix) {
  switch (pix) {
    case Pix::Grey8: case Pix::NV12: case Pix::NV21: case Pix::I420: case Pix::YV12: return true;
    default: return false;
  }
}

size_t lumaRowBytes(Pix pix, uint32_t w) {
  const size_t n = w;
  switch (pix) {
    case Pix::Grey8: case Pix::NV12: case Pix::NV21: case Pix::I420: case Pix::YV12: case Pix::Bayer8: return n;
    case Pix::Grey16: case Pix::YUYV: case Pix::UYVY: case Pix::YVYU: case Pix::VYUY: return 2 * n;
    case Pix::RGB24: case Pix::BGR24: return 3 * n;
    case Pix::BGRA32: return 4 * n;
    case Pix::MJPEG: case Pix::Unknown: return 0;
  }
  return 0;
}

size_t frameBytes(Pix pix, uint32_t w, uint32_t h, uint32_t stride) {
  const size_t row = lumaRowBytes(pix, w);
  if (row == 0 || h == 0) return 0;
  return size_t(h - 1) * stride + row;
}

bool toLuma(Pix pix, uint8_t bits, const uint8_t* data, size_t bytes, uint32_t w, uint32_t h, uint32_t stride,
            std::vector<uint8_t>& out, std::string* why) {
  auto fail = [&](std::string s) { if (why) *why = std::move(s); return false; };
  const char* name = pixName(pix);
  if (pix == Pix::MJPEG) return fail("MJPEG is decoded by jpeg.h, not converted");
  const size_t row = lumaRowBytes(pix, w);
  if (row == 0 && pix == Pix::Unknown) return fail("no conversion for an unknown pixel layout");
  if (w == 0 || h == 0)
    return fail(std::string("an empty ") + name + " frame (" + std::to_string(w) + " x " + std::to_string(h) + ")");
  if (stride < row)
    return fail("rows " + std::to_string(stride) + " bytes apart, under the " + std::to_string(row) + " bytes a " +
                std::to_string(w) + "-pixel " + std::string(name) + " row holds");
  const size_t need = frameBytes(pix, w, h, stride);
  if (!data || bytes < need)
    return fail("a short frame: " + std::to_string(data ? bytes : 0) + " bytes of the " + std::to_string(need) +
                " a " + std::to_string(w) + " x " + std::to_string(h) + " " + name + " frame at stride " +
                std::to_string(stride) + " needs");
  unsigned shift = 0;
  if (pix == Pix::Grey16) {
    if (bits < 8 || bits > 16)
      return fail(std::string(name) + " with " + std::to_string(bits) + " significant bits (8 to 16 read)");
    shift = bits - 8u;
  }
  out.resize(size_t(w) * h);
  uint8_t* o = out.data();
  switch (pix) {
    case Pix::Grey8: case Pix::NV12: case Pix::NV21: case Pix::I420: case Pix::YV12: copyRows(data, w, h, stride, o); break;
    case Pix::Grey16: grey16(data, w, h, stride, shift, o); break;
    case Pix::YUYV: case Pix::YVYU: everyOther<0>(data, w, h, stride, o); break;
    case Pix::UYVY: case Pix::VYUY: everyOther<1>(data, w, h, stride, o); break;
    case Pix::Bayer8: bayer8(data, w, h, stride, o); break;
    case Pix::RGB24: colour<3, 0, 1, 2>(data, w, h, stride, o); break;
    case Pix::BGR24: colour<3, 2, 1, 0>(data, w, h, stride, o); break;
    case Pix::BGRA32: colour<4, 2, 1, 0>(data, w, h, stride, o); break;
    case Pix::MJPEG: case Pix::Unknown: break;
  }
  return true;
}

}  // namespace lizard::cam
