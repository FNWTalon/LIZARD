// ai: Layer 2, the per-frame codec (lizard.h): the C reference's encoder (src/focus.h: focus_init, focus_parity,
// ai: focus_encode, focus_paint_rgba) and its blind receiver (core/cpu/codec.h over src/any.h), behind the API's
// ai: formats. Two codes are painted and composited as the senders do (core/tx/sender.cpp paint, lizard-web/send-worker.mjs);
// ai: the 2:1 regions are core/rx/receiver.cpp cropsOf's.
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "api.hpp"

extern "C" {
#include "codec.h"
#include "focus.h"
#include "shake.h"
// ai: src/acquire.c: RGBA to luma, the web's weights (77, 150, 29, rounded)
void ob_luma(const uint8_t* rgba, int n, uint8_t* out);
}

struct liz_encoder {
  liz_format fmt;
  liz_geometry g;
  focus_t f{};
  bool made = false;
  std::vector<float> drive;
  std::vector<uint8_t> rgba;
  ~liz_encoder() { if (made) focus_free(&f); }
};

struct liz_decoder {
  cpu_dec_t* d = nullptr;
  int top = 0, bb = 0;
  std::vector<uint8_t> luma, blocks, ok;
  ~liz_decoder() { cpu_dec_free(d); }
};

using namespace liz;

static int bytesPer(liz_pixfmt fmt) {
  need(fmt == LIZ_GREY8 || fmt == LIZ_RGBX8 || fmt == LIZ_RGBA8 || fmt == LIZ_BGRA8, "a pixel format of lizard.h's");
  return fmt == LIZ_GREY8 ? 1 : 4;
}

extern "C" {

LIZ_API liz_encoder* liz_encoder_new(const liz_format* format) {
  return guardNew<liz_encoder>([&] {
    need(format, "a format");
    auto e = std::make_unique<liz_encoder>();
    e->fmt = *format;
    e->g = geometryOf(*format);
    if (focus_init(&e->f, e->g.n, 8 * format->blocks, FOCUS_LDPC, 2.0f, e->g.span, 0.f, 0, 0, 0, 0, 0, 0))
      throw Error(LIZ_E_FORMAT, "the codec refused LIZARD-" + std::to_string(8 * format->blocks));
    e->made = true;
    if (focus_fmt_fps(&e->f, format->fps)) throw Error(LIZ_E_FORMAT, "the codec refused " + std::to_string(format->fps) + " a second");
    // ai: the arithmetic of format.cpp held to the codec's own
    if (e->f.pxm != e->g.pxm || e->f.px + 2 * FOCUS_QUIET * e->f.pxm != e->g.side || e->f.block_bytes != LIZ_BLOCK || e->f.blocks != format->blocks)
      throw Error(LIZ_E_INTERNAL, "the geometry differs from the codec's");
    e->drive.resize(size_t(e->f.px) * e->f.px);
    e->rgba.resize(size_t(e->g.side) * e->g.side * 4);
    return e.release();
  });
}

LIZ_API void liz_encoder_free(liz_encoder* e) { delete e; }

LIZ_API int liz_encoder_geometry(const liz_encoder* e, liz_geometry* out) {
  return guard([&] { need(e && out, "an encoder and a place for its geometry"); *out = e->g; return LIZ_OK; });
}

LIZ_API int liz_encoder_paint(liz_encoder* e, const uint8_t* blocks, uint32_t picture, uint8_t* out, int stride, liz_pixfmt fmt) {
  return guard([&] {
    need(e && blocks && out, "an encoder, blocks and a frame to paint");
    const int bpp = bytesPer(fmt);
    const liz_geometry& g = e->g;
    need(stride >= g.width * bpp, "a stride of at least the frame's width in bytes");
    focus_parity(&e->f, int(picture & 3));
    const size_t per = size_t(e->fmt.blocks) * LIZ_BLOCK;
    for (int k = 0; k < e->fmt.codes; k++) {
      focus_encode(&e->f, blocks + k * per, e->drive.data());
      focus_paint_rgba(&e->f, e->drive.data(), e->rgba.data());
      // ai: each symbol into its place, the gap after it its row's last pixel (the margin's colour)
      const int x0 = k * (g.side + g.gap), fill = k + 1 < e->fmt.codes ? g.gap : 0;
      for (int y = 0; y < g.side; y++) {
        const uint8_t* src = e->rgba.data() + size_t(y) * g.side * 4;
        uint8_t* o = out + size_t(y) * stride + size_t(x0) * bpp;
        if (bpp == 4) {
          std::memcpy(o, src, size_t(g.side) * 4);
          for (int x = 0; x < fill; x++) std::memcpy(o + size_t(g.side + x) * 4, src + size_t(g.side - 1) * 4, 4);
        } else {
          for (int x = 0; x < g.side; x++) o[x] = src[4 * x];
          std::memset(o + g.side, src[4 * (g.side - 1)], size_t(fill));
        }
      }
    }
    return LIZ_OK;
  });
}

LIZ_API liz_decoder* liz_decoder_new(int nmax) {
  return guardNew<liz_decoder>([&] {
    if (!nmax) nmax = 1536;
    need(nmax >= 256 && nmax <= 1536, "nmax is 256 to 1536 (or 0)");
    auto d = std::make_unique<liz_decoder>();
    d->d = cpu_dec_new(nmax);
    if (!d->d) throw Error(LIZ_E_NOMEM, "the decoder could not be made");
    d->top = cpu_dec_top(d->d);
    d->bb = cpu_dec_block_bytes(d->d);
    if (d->bb != LIZ_BLOCK) throw Error(LIZ_E_INTERNAL, "the codec's block is not lizard.h's");
    d->blocks.resize(size_t(d->top) * d->bb);
    d->ok.resize(size_t(d->top));
    return d.release();
  });
}

LIZ_API void liz_decoder_free(liz_decoder* d) { delete d; }

LIZ_API int liz_decoder_max_blocks(const liz_decoder* d) { return guard([&] { need(d, "a decoder"); return d->top; }); }

LIZ_API int liz_decode(liz_decoder* d, const uint8_t* px, int w, int h, int stride, liz_pixfmt fmt, int* held,
                       uint8_t* verified, liz_decoded* out) {
  return guard([&] {
    need(d && px && held && verified, "a decoder, an image, the held word and a place for the blocks");
    need(w > 0 && h > 0, "an image of at least a pixel");
    const int bpp = bytesPer(fmt);
    need(stride >= w * bpp, "a stride of at least the image's width in bytes");
    // ai: the codec reads tight luma rows: a grey image with rows w apart as it is, anything else copied
    const uint8_t* img = px;
    if (fmt != LIZ_GREY8 || stride != w) {
      d->luma.resize(size_t(w) * h);
      for (int y = 0; y < h; y++) {
        const uint8_t* row = px + size_t(y) * stride;
        uint8_t* o = d->luma.data() + size_t(y) * w;
        if (fmt == LIZ_GREY8) std::memcpy(o, row, size_t(w));
        else if (fmt == LIZ_BGRA8) for (int x = 0; x < w; x++) o[x] = uint8_t((77 * row[4 * x + 2] + 150 * row[4 * x + 1] + 29 * row[4 * x] + 128) >> 8);
        else ob_luma(row, w, o);
      }
      img = d->luma.data();
    }
    cpu_frame_t fr{};
    cpu_dec_frame(d->d, img, w, h, *held, d->blocks.data(), d->ok.data(), &fr);
    if (fr.word) *held = fr.version;
    int k = 0;
    const int total = std::min(fr.total, d->top);
    for (int b = 0; b < total; b++)
      if (d->ok[b]) std::memcpy(verified + size_t(k++) * LIZ_BLOCK, d->blocks.data() + size_t(b) * d->bb, LIZ_BLOCK);
    if (out) {
      liz_decoded o{};
      o.found = fr.found; o.ring = fr.ring; o.n = fr.n; o.word = fr.word; o.blocks = fr.version; o.fps = fr.fps;
      o.held_used = fr.held; o.total = fr.total; o.verified = k;
      std::memcpy(o.quad, fr.quad, sizeof o.quad);
      o.pilot_blocks = fr.pilot_blocks;
      std::memcpy(o.pilot_r, fr.pilot_r, sizeof o.pilot_r);
      std::memcpy(o.pilot_sd, fr.pilot_sd, sizeof o.pilot_sd);
      *out = o;
    }
    return k;
  });
}

LIZ_API int liz_layout_rects(int w, int h, int layout, int rect[2][4]) {
  return guard([&] {
    need(rect && w > 0 && h > 0, "a frame and a place for its regions");
    need(layout == 1 || layout == 2, "layout is 1 (the centre square) or 2 (2:1)");
    const bool two = layout == 2, wide = w >= h;
    const int lo = std::min(w, h), hi = std::max(w, h), side = two ? std::min(lo, hi / 2) : lo, n = two ? 2 : 1;
    for (int k = 0; k < n; k++) {
      const int along = (hi - n * side) / 2 + k * side, across = (lo - side) / 2;
      rect[k][0] = wide ? along : across;
      rect[k][1] = wide ? across : along;
      rect[k][2] = side;
      rect[k][3] = side;
    }
    return n;
  });
}

LIZ_API void liz_stream_fill(uint32_t id, uint8_t payload[LIZ_PAYLOAD]) {
  if (payload) stream_fill(id, payload, LIZ_PAYLOAD);
}

}  // extern "C"
