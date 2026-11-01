// ai: Layer 5, the receive session (lizard.h, 2026-10-31): core/rx/session.h's Session behind the C API, what the apps'
// ai: receive screens run (the Kotlin binding's dev.lizard.Session over it). The commands are the session's own,
// ai: queued on its thread: their outcome is the phase (the stats) or the log, never waited for here, but the preview
// ai: surface's hand-over (liz_session_preview_window), which waits for the camera to take it or let it go.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "api.hpp"
#include "session.h"

struct liz_session {
  std::unique_ptr<lizard::Session> s;
};

using namespace liz;

namespace {

// ai: s into json (at most cap bytes, NUL-ended, cut before a character, never inside one); s's whole length
size_t put(const std::string& s, char* json, size_t cap) {
  if (json && cap) {
    size_t n = std::min(s.size(), cap - 1);
    while (n < s.size() && n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) n--;
    std::memcpy(json, s.data(), n);
    json[n] = '\0';
  }
  return s.size();
}

std::string str(const char* s) { return s ? s : ""; }

// ai: a hook's outcome: "" done; else off Android unsupported, on Android the state it names
int hooked(const std::string& why) {
  if (why.empty()) return LIZ_OK;
#if defined(__ANDROID__)
  throw Error(LIZ_E_STATE, why);
#else
  throw Error(LIZ_E_UNSUPPORTED, why);
#endif
}

}  // namespace

extern "C" {

LIZ_API liz_session* liz_session_new(const liz_session_config* c) {
  return guardNew<liz_session>([&] {
    need(c && c->data_dir && *c->data_dir, "a config with a data folder");
    lizard::Session::Config k;
    k.assets = str(c->assets);
    k.cacheDir = str(c->cache_dir);
    k.dataDir = c->data_dir;
    k.decoder = c->decoder && *c->decoder ? c->decoder : "auto";
    k.layout = c->layout && *c->layout ? c->layout : "1:1";
    k.forgetSettings = c->forget_settings != 0;
    if (c->log) {
      void (*log)(void*, const char*) = c->log;
      void* user = c->user;
      k.log = [log, user](const std::string& m) { log(user, m.c_str()); };
    }
    std::string why;
    auto s = lizard::Session::create(k, why);
    if (!s) throw Error(LIZ_E_ARG, why);
    auto out = std::make_unique<liz_session>();
    out->s = std::move(s);
    return out.release();
  });
}

LIZ_API void liz_session_free(liz_session* s) { delete s; }

LIZ_API size_t liz_cameras(char* json, size_t cap) { return put(lizard::Session::cameras(), json, cap); }
LIZ_API size_t liz_cameras_stamp(char* out, size_t cap) { return put(lizard::Session::camerasStamp(), out, cap); }

LIZ_API void liz_session_open(liz_session* s, const char* id) { guard([&] { need(s, "a session"); s->s->open(str(id)); return LIZ_OK; }); }
LIZ_API void liz_session_close(liz_session* s) { guard([&] { need(s, "a session"); s->s->close(); return LIZ_OK; }); }
LIZ_API void liz_session_set(liz_session* s, const char* control, double value, int automatic) {
  guard([&] { need(s && control, "a session and a control"); s->s->set(control, value, automatic != 0); return LIZ_OK; });
}
LIZ_API void liz_session_start(liz_session* s, const char* mode) { guard([&] { need(s, "a session"); s->s->start(str(mode)); return LIZ_OK; }); }
LIZ_API void liz_session_stop(liz_session* s) { guard([&] { need(s, "a session"); s->s->stop(); return LIZ_OK; }); }
LIZ_API void liz_session_reconfigure(liz_session* s, const char* decoder, const char* layout) {
  guard([&] {
    need(s, "a session");
    s->s->reconfigure(decoder && *decoder ? decoder : "auto", layout && *layout ? layout : "1:1");
    return LIZ_OK;
  });
}
LIZ_API void liz_session_batch_cap(liz_session* s, int frames) { guard([&] { need(s, "a session"); s->s->batchCap(frames); return LIZ_OK; }); }

LIZ_API size_t liz_session_stats(liz_session* s, char* json, size_t cap) {
  size_t n = 0;
  guard([&] { need(s, "a session"); n = put(s->s->stats(), json, cap); return LIZ_OK; });
  return n;
}
LIZ_API size_t liz_session_camera(liz_session* s, char* json, size_t cap) {
  size_t n = 0;
  guard([&] { need(s, "a session"); n = put(s->s->camera(), json, cap); return LIZ_OK; });
  return n;
}

LIZ_API int liz_session_preview(liz_session* s, uint64_t since, int side, uint8_t* out, size_t cap, uint32_t* w, uint32_t* h, uint64_t* seq) {
  return guard([&] {
    need(s && out && w && h && seq, "a session and places for the picture");
    // ai: a side whose square cap does not hold shrunk to one it does
    int sd = side;
    if (sd > 0 && uint64_t(sd) * uint64_t(sd) > cap) sd = int(std::sqrt(double(cap)));
    std::vector<uint8_t> px;
    uint32_t pw = 0, ph = 0;
    uint64_t q = 0;
    if (sd <= 0 || !s->s->preview(since, sd, px, pw, ph, q) || size_t(pw) * ph > cap || px.size() < size_t(pw) * ph) return 0;
    std::memcpy(out, px.data(), size_t(pw) * ph);
    *w = pw;
    *h = ph;
    *seq = q;
    return 1;
  });
}

LIZ_API void liz_session_forget(liz_session* s, const char* root) {
  guard([&] { need(s && root, "a session and a root"); s->s->forget(root); return LIZ_OK; });
}
LIZ_API void liz_session_clear(liz_session* s) { guard([&] { need(s, "a session"); s->s->clear(); return LIZ_OK; }); }
LIZ_API void liz_session_replays(liz_session* s, int on, const char* meta) {
  guard([&] { need(s, "a session"); s->s->replays(on != 0, str(meta)); return LIZ_OK; });
}
LIZ_API void liz_session_extra(liz_session* s, const char* json) { guard([&] { need(s, "a session"); s->s->extra(str(json)); return LIZ_OK; }); }

LIZ_API int liz_session_preview_window(liz_session* s, void* native_window) {
  return guard([&] {
    need(s, "a session");
    return hooked(s->s->previewWindow(native_window).get());
  });
}
LIZ_API int liz_link(int up, const char* layout) {
  return guard([&] {
    std::string why;
    lizard::Session::link(up != 0, layout && *layout ? layout : "1:1", why);
    return hooked(why);
  });
}
LIZ_API int liz_link_pack(int on) {
  return guard([&] {
    std::string why;
    lizard::Session::linkPack(on != 0, why);
    return hooked(why);
  });
}
LIZ_API size_t liz_link_state(char* json, size_t cap) { return put(lizard::Session::linkState(), json, cap); }
LIZ_API int liz_usb_accessory(int fd) {
  return guard([&] {
    need(fd >= 0, "an fd");
    std::string why;
    lizard::Session::usbAccessory(fd, why);
    return hooked(why);
  });
}

LIZ_API size_t liz_kept(const char* data_dir, char* json, size_t cap) {
  size_t n = 0;
  guard([&] { need(data_dir && *data_dir, "a data folder"); n = put(lizard::Session::kept(data_dir), json, cap); return LIZ_OK; });
  return n;
}
LIZ_API int liz_kept_placed(const char* data_dir, const char* id) {
  return guard([&] {
    need(data_dir && *data_dir && id, "a data folder and a take");
    std::string why;
    if (!lizard::Session::placed(data_dir, id, why)) throw Error(LIZ_E_IO, why);
    return LIZ_OK;
  });
}
LIZ_API size_t liz_partials(const char* data_dir, char* json, size_t cap) {
  size_t n = 0;
  guard([&] { need(data_dir && *data_dir, "a data folder"); n = put(lizard::Session::partials(data_dir), json, cap); return LIZ_OK; });
  return n;
}
LIZ_API int liz_partial_delete(const char* data_dir, const char* id) {
  return guard([&] {
    need(data_dir && *data_dir && id, "a data folder and a transfer");
    std::string why;
    // ai: the lock's refusal (session.cpp BUSY) is a state, anything else the file system's
    if (!lizard::Session::deletePartial(data_dir, id, why)) throw Error(why == "another LIZARD window is receiving" ? LIZ_E_STATE : LIZ_E_IO, why);
    return LIZ_OK;
  });
}

}  // extern "C"
