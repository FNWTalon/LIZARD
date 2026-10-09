// ai: Layer 4, the engines (lizard.h): core/rx/receiver.h's Receiver (the GPU decoder in batches or the C on a pool,
// ai: the transfer on its own thread) and core/tx/sender.h's Sender (painters ahead on their threads, the GPU's or the
// ai: C's), what the Android app and the desktop sender run, behind the C API.
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "api.hpp"
#include "receiver.h"
#include "sender.h"

struct liz_receiver {
  std::unique_ptr<lizard::Receiver> r;
  void (*release)(void*, uint64_t) = nullptr;
  void (*log)(void*, const char*) = nullptr;
  void* user = nullptr;
  std::string stats, file;
};

struct liz_sender {
  std::vector<uint8_t> owned;
  std::unique_ptr<lizard::Sender> s;
  liz_format fmt{};
  liz_geometry g{};
  std::vector<uint8_t> rgba;
  std::string stats;
};

using namespace liz;

extern "C" {

LIZ_API liz_receiver* liz_receiver_new(const liz_receiver_config* c, void (*release)(void*, uint64_t)) {
  return guardNew<liz_receiver>([&] {
    need(c && c->store_dir && *c->store_dir, "a config with a store directory");
    need(c->decoder >= LIZ_DECODER_AUTO && c->decoder <= LIZ_DECODER_CPU, "decoder is a LIZ_DECODER_* value");
    need(c->layout == 0 || c->layout == 1 || c->layout == 2, "layout is 1 or 2");
    auto r = std::make_unique<liz_receiver>();
    r->release = release; r->log = c->log; r->user = c->user;
    lizard::ReceiverConfig cfg;
    cfg.assets = c->assets ? c->assets : "";
    cfg.cacheDir = c->cache_dir ? c->cache_dir : "";
    cfg.storeDir = c->store_dir;
    cfg.device = c->device ? c->device : "";
    cfg.precision = c->precision ? c->precision : "auto";
    // ai: no assets, no GPU decoder: auto is the C's
    cfg.decoder = c->decoder == LIZ_DECODER_GPU ? "gpu" : c->decoder == LIZ_DECODER_CPU || cfg.assets.empty() ? "cpu" : "auto";
    if (c->decoder == LIZ_DECODER_GPU && cfg.assets.empty()) throw Error(LIZ_E_ARG, "the GPU decoder needs its assets");
    cfg.cpuThreads = c->cpu_threads;
    cfg.layout = c->layout == 2 ? "2:1" : "1:1";
    liz_receiver* raw = r.get();
    if (c->log) cfg.log = [raw](const std::string& s) { raw->log(raw->user, s.c_str()); };
    try {
      r->r = lizard::Receiver::create(cfg, [raw](uint64_t tag) { if (raw->release) raw->release(raw->user, tag); });
    } catch (const std::runtime_error& e) {
      throw Error(LIZ_E_UNSUPPORTED, e.what());   // ai: the GPU decoder asked for and not made (no device, no variant)
    }
    if (!r->r) throw Error(LIZ_E_UNSUPPORTED, "no receiver: the GPU decoder asked for runs on no device here");
    return r.release();
  });
}

LIZ_API void liz_receiver_free(liz_receiver* r) { delete r; }

LIZ_API int liz_receiver_push(liz_receiver* r, const uint8_t* luma, int w, int h, int stride, int64_t ts, uint64_t tag) {
  return guard([&] {
    need(r && luma && w > 0 && h > 0 && stride >= w, "a receiver and a luma plane");
    lizard::CameraFrame f;
    f.luma = luma; f.width = uint32_t(w); f.height = uint32_t(h); f.stride = uint32_t(stride); f.timestampNs = ts; f.tag = tag;
    r->r->push(f);
    return LIZ_OK;
  });
}

LIZ_API int liz_receiver_push_hardware_buffer(liz_receiver* r, void* hb, int w, int h, int64_t ts, uint64_t tag) {
  return guard([&] {
#ifdef __ANDROID__
    need(r && hb && w > 0 && h > 0, "a receiver and a hardware buffer");
    lizard::CameraFrame f;
    f.hb = static_cast<AHardwareBuffer*>(hb); f.width = uint32_t(w); f.height = uint32_t(h); f.timestampNs = ts; f.tag = tag;
    r->r->push(f);
    return LIZ_OK;
#else
    (void)r; (void)hb; (void)w; (void)h; (void)ts; (void)tag;
    return fail(LIZ_E_UNSUPPORTED, "hardware buffers are Android's");
#endif
  });
}

LIZ_API int liz_receiver_wants_luma(liz_receiver* r) { return guard([&] { need(r, "a receiver"); return r->r->wantsLuma() ? 1 : 0; }); }

LIZ_API const char* liz_receiver_stats(liz_receiver* r) {
  const char* s = "";
  guard([&] { need(r, "a receiver"); r->stats = r->r->stats(); s = r->stats.c_str(); return LIZ_OK; });
  return s;
}

LIZ_API int liz_receiver_series(liz_receiver* r, double since, double* out, int cap) {
  return guard([&] {
    need(r && (out || cap <= 0), "a receiver and a place for the series");
    const std::vector<double> v = r->r->series(since);
    const int n = std::min<int>(int(v.size()), cap);
    if (n > 0) std::memcpy(out, v.data(), size_t(n) * sizeof(double));
    return n;
  });
}

LIZ_API void liz_receiver_batch_cap(liz_receiver* r, int n) { guard([&] { need(r, "a receiver"); r->r->batchCap(n); return LIZ_OK; }); }

LIZ_API const char* liz_receiver_file(liz_receiver* r) {
  const char* s = "";
  guard([&] { need(r, "a receiver"); r->file = r->r->file(); s = r->file.c_str(); return LIZ_OK; });
  return s;
}

LIZ_API void liz_receiver_clear(liz_receiver* r) { guard([&] { need(r, "a receiver"); r->r->clear(); return LIZ_OK; }); }
LIZ_API void liz_receiver_camera_closed(liz_receiver* r) { guard([&] { need(r, "a receiver"); r->r->cameraClosed(); return LIZ_OK; }); }

LIZ_API liz_sender* liz_sender_new(const uint8_t* data, size_t length, const char* name, const char* type, int flags) {
  return guardNew<liz_sender>([&] {
    auto s = std::make_unique<liz_sender>();
    static const uint8_t empty = 0;
    if (data && (flags & LIZ_TX_COPY)) { s->owned.assign(data, data + length); data = s->owned.data(); }
    if (data && !length) data = &empty;
    try {
      s->s = std::make_unique<lizard::Sender>(data, data ? length : 0, name ? name : "", type ? type : "");
    } catch (const std::runtime_error& e) {
      throw Error(LIZ_E_TOOBIG, e.what());
    }
    return s.release();
  });
}

LIZ_API void liz_sender_free(liz_sender* s) { delete s; }

LIZ_API int liz_sender_configure(liz_sender* s, const liz_format* f, int painter, int threads, const char* assets, const char* device) {
  return guard([&] {
    need(s && f, "a sender and a format");
    need(painter >= LIZ_PAINTER_CPU && painter <= LIZ_PAINTER_AUTO, "painter is a LIZ_PAINTER_* value");
    const liz_geometry g = geometryOf(*f);
    lizard::TxFormat t;
    t.n = g.n; t.subch = 8 * f->blocks; t.span = g.span; t.fps = f->fps; t.codes = f->codes;
    t.threads = threads > 0 ? threads : 3;
    t.assets = assets ? assets : "";
    t.device = device ? device : "";
    // ai: no assets, no GPU painter: auto is the C's
    t.painter = painter == LIZ_PAINTER_AUTO && t.assets.empty() ? 0 : painter;
    if (painter == LIZ_PAINTER_GPU && t.assets.empty()) throw Error(LIZ_E_ARG, "the GPU painter needs its assets");
    const std::string why = s->s->configure(t);
    if (!why.empty()) throw Error(painter == LIZ_PAINTER_GPU ? LIZ_E_UNSUPPORTED : LIZ_E_FORMAT, why);
    if (s->s->width() != g.width || s->s->side() != g.height) throw Error(LIZ_E_INTERNAL, "the sender's frame is not the geometry's");
    s->fmt = *f;
    s->g = g;
    return LIZ_OK;
  });
}

LIZ_API int liz_sender_geometry(liz_sender* s, liz_geometry* out) {
  return guard([&] {
    need(s && out, "a sender and a place for its geometry");
    if (!s->g.width) throw Error(LIZ_E_STATE, "not configured");
    *out = s->g;
    return LIZ_OK;
  });
}

LIZ_API int liz_sender_ready(liz_sender* s) { return guard([&] { need(s, "a sender"); return s->s->ready() ? 1 : 0; }); }

LIZ_API int liz_sender_take(liz_sender* s, uint8_t* dst, int stride, liz_pixfmt fmt) {
  return guard([&] {
    need(s && dst, "a sender and a frame to fill");
    if (!s->g.width) throw Error(LIZ_E_STATE, "not configured");
    need(fmt == LIZ_GREY8 || fmt == LIZ_RGBX8 || fmt == LIZ_RGBA8 || fmt == LIZ_BGRA8, "a pixel format of lizard.h's");
    const int bpp = fmt == LIZ_GREY8 ? 1 : 4, W = s->g.width, H = s->g.height;
    need(stride >= W * bpp, "a stride of at least the frame's width in bytes");
    // ai: grey pixels read the same as RGBA, RGBX and BGRA; a grey frame is taken through the RGBA one
    if (bpp == 4) return s->s->take(dst, stride) ? 1 : 0;
    s->rgba.resize(size_t(W) * H * 4);
    if (!s->s->take(s->rgba.data(), W * 4)) return 0;
    for (int y = 0; y < H; y++) {
      const uint8_t* src = s->rgba.data() + size_t(y) * W * 4;
      uint8_t* o = dst + size_t(y) * stride;
      for (int x = 0; x < W; x++) o[x] = src[4 * x];
    }
    return 1;
  });
}

LIZ_API const char* liz_sender_stats(liz_sender* s) {
  const char* r = "";
  guard([&] { need(s, "a sender"); s->stats = s->s->stats(); r = s->stats.c_str(); return LIZ_OK; });
  return r;
}

}  // extern "C"
