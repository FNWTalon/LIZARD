// ai: The session (rx_session.h, 2026-10-25): one camera into one Receiver. The camera's capture thread is the sink's:
// ai: each frame pushed (the receiver copies its crop and releases it inside push, so the release callback has
// ai: nothing to do) and its crop copied for the preview while one is asked. Locks, so the quick never wait on the
// ai: slow: camMu around every call into the camera (open, start, stop, set: a UVC device's open or start can take
// ai: seconds); rxMu around the receiver's pointer (the
// ai: sink and every reader copy it and call outside the lock; reconfigure swaps it; batchCap and record set the
// ai: receiver's under it, a store each); ptrMu around the camera's pointer and what was read off it (stats, camera()
// ai: copy it, so the camera outlives their call however a close goes); and leaf locks for the preview, the measured
// ai: rate and the camera's counts. The order: camMu before any other; no other is held while
// ai: one is taken but rxMu inside camMu, and a receiver's own inside rxMu.
// ai: The user's settings are kept here and set again after each start (rx_session.h set): every backend puts back at
// ai: stop what set changed, so the app's zoom, exposure and gain would otherwise last one stream (2026-10-25).
#include "rx_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <thread>
#include <utility>

#include "camera.h"
#include "crops.h"
#include "json.hpp"
#include "receiver.h"
#include "replay.h"

namespace lizard {

using json = nlohmann::json;

namespace {

// ai: how long a preview asked keeps the capture thread copying crops
constexpr int64_t PREVIEW_HOLD_NS = 1'000'000'000;
// ai: the stats' figures of the receiver's last window (receiver.cpp stats): a stream's, so none while none runs
const char* const WINDOW_FIGURES[] = {"capturedFps", "processedFps", "dropped", "foundShare", "side", "heldMs", "heldMaxMs", "blocks",
                                      "windowSecs", "goodputKBs", "gpuMs", "cpuMs", "repeatShare"};

int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ai: names from the camera and the light may hold bytes that are not UTF-8: replaced, never a throw
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

std::string num(double v, int digits) {
  char b[48];
  std::snprintf(b, sizeof b, "%.*f", digits, v);
  return b;
}

// ai: a setting's value as the log says it: its own digits, at most six significant
std::string gnum(double v) {
  char b[48];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}

bool oneOf(const std::string& v, std::initializer_list<const char*> of, const char* what, std::string& why) {
  std::string list;
  for (const char* o : of) {
    if (v == o) return true;
    list += std::string(list.empty() ? "" : ", ") + o;
  }
  why = std::string("the ") + what + " is one of " + list + " (not \"" + v + "\")";
  return false;
}

// ai: An area average's taps along one axis, `in` samples to `out` (out <= in): output i the mean of the input span
// ai: [i in / out, (i + 1) in / out), the span's end pixels weighed by the share of them it covers.
struct Tap { uint32_t first = 0; std::vector<float> w; };
std::vector<Tap> tapsFor(uint32_t in, uint32_t out) {
  std::vector<Tap> t(out);
  const double r = double(in) / out;
  for (uint32_t i = 0; i < out; i++) {
    const double a = i * r, b = std::min<double>(in, (i + 1) * r);
    const uint32_t f = std::min(in - 1, uint32_t(a)), l = std::max(f + 1, std::min(in, uint32_t(std::ceil(b - 1e-9))));
    t[i].first = f;
    for (uint32_t k = f; k < l; k++) t[i].w.push_back(float(std::max(0.0, std::min<double>(k + 1, b) - std::max<double>(k, a)) / r));
  }
  return t;
}

// ai: src (sw x sh, rows of sw) area-averaged to dst (dw x dh, rows of dw): rows first, then columns, in floats; mid
// ai: and acc the caller's buffers, grown as needed
void shrink(const uint8_t* src, uint32_t sw, uint32_t sh, uint8_t* dst, uint32_t dw, uint32_t dh, std::vector<float>& mid,
            std::vector<float>& acc) {
  if (dw == sw && dh == sh) {
    std::memcpy(dst, src, size_t(sw) * sh);
    return;
  }
  const std::vector<Tap> tx = tapsFor(sw, dw), ty = tapsFor(sh, dh);
  mid.resize(size_t(dw) * sh);
  acc.resize(dw);
  for (uint32_t y = 0; y < sh; y++) {
    const uint8_t* s = src + size_t(y) * sw;
    float* o = mid.data() + size_t(y) * dw;
    for (uint32_t i = 0; i < dw; i++) {
      const Tap& t = tx[i];
      float a = 0;
      for (size_t k = 0; k < t.w.size(); k++) a += t.w[k] * s[t.first + k];
      o[i] = a;
    }
  }
  for (uint32_t j = 0; j < dh; j++) {
    std::fill(acc.begin(), acc.end(), 0.f);
    const Tap& t = ty[j];
    for (size_t k = 0; k < t.w.size(); k++) {
      const float* m = mid.data() + size_t(t.first + k) * dw;
      const float wk = t.w[k];
      for (uint32_t i = 0; i < dw; i++) acc[i] += wk * m[i];
    }
    uint8_t* o = dst + size_t(j) * dw;
    for (uint32_t i = 0; i < dw; i++) o[i] = uint8_t(std::lround(std::clamp(acc[i], 0.f, 255.f)));
  }
}

json infoJson(const cam::Info& i) { return {{"id", i.id}, {"name", i.name}, {"backend", i.backend}, {"detail", i.detail}}; }

json describe(cam::Camera& c, const std::vector<cam::Mode>& modes, const std::vector<cam::Control>& controls) {
  json ms = json::array();
  for (const auto& m : modes) {
    const char* pix = cam::pixName(m.pix);
    ms.push_back({{"w", m.width}, {"h", m.height}, {"fps", m.fps}, {"fpsMin", m.fpsMin}, {"format", m.format},
                  {"pix", pix ? pix : ""}, {"compressed", m.compressed}});
  }
  json ks = json::array();
  for (const auto& k : controls)
    ks.push_back({{"id", k.id}, {"name", k.name}, {"unit", k.unit}, {"min", k.min}, {"max", k.max}, {"step", k.step},
                  {"def", k.def}, {"value", k.value}, {"canAuto", k.canAuto}, {"isAuto", k.isAuto}});
  return {{"info", infoJson(c.info())}, {"modes", ms}, {"pick", cam::pick(modes, 1920, 1080, 60)}, {"controls", ks}};
}

// ai: An object others may still hold for the length of one call (a stats, a series) let go here, on this
// ai: thread, once they have: its destructor (a receiver's joins and device wait, a camera's teardown) never lands on
// ai: the UI's thread. Past 5 s the last holder ends it.
template <class T>
void letGo(std::shared_ptr<T>& p) {
  for (int i = 0; p && p.use_count() > 1 && i < 5000; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  p.reset();
}

}  // namespace

struct RxSession::Impl {
  Config cfg;
  void say(const std::string& m) const {
    if (cfg.log) cfg.log(m);
  }

  // ai: the receiver with the decoder and layout it was made with
  struct Rx {
    std::unique_ptr<Receiver> r;
    std::string decoder, layout;
  };
  std::mutex rxMu;
  std::shared_ptr<Rx> rx;          // ai: rxMu
  int cap = 0;                     // ai: rxMu: batchCap's n, handed to a receiver made later (0 none asked)
  std::shared_ptr<Replay> rec;     // ai: rxMu: record's replay, handed to a receiver made later
  std::shared_ptr<Rx> current() {
    std::lock_guard<std::mutex> l(rxMu);
    return rx;
  }

  std::mutex camMu;
  std::vector<cam::Mode> modes;    // ai: camMu: the open camera's
  std::vector<cam::Control> ctls;  // ai: camMu: its controls as last read (describeNow)
  int mode = -1;                   // ai: camMu: the index started last
  bool streamed = false;           // ai: camMu: a stream ran since the receiver was last told the camera closed
  std::atomic<bool> streaming{false};
  std::atomic<uint64_t> starts{0};   // ai: the camera's starts since it opened (the stats' "starts")
  // ai: The user's settings (set) since the camera opened, each control's newest in the order first asked, set again
  // ai: after each start: every backend puts back at stop what set changed (a UVC device keeps its controls for the
  // ai: next app), and a start applies the camera rules over the device's own values. camMu.
  struct Want {
    std::string id;
    double value = 0;
    bool automatic = false;
  };
  std::vector<Want> wants;

  std::mutex ptrMu;
  std::shared_ptr<cam::Camera> camera;   // ai: read under ptrMu (or camMu); written under both
  json info, modeNow, described;         // ai: ptrMu: its Info, the mode started last (null none), camera()'s answer

  std::mutex csMu;
  cam::Stats cs;                   // ai: csMu: Camera::stats as last read

  std::mutex endMu;
  std::string ended;               // ai: endMu: the stream's own end's why ("" none)

  std::atomic<uint64_t> tags{0};   // ai: frames pushed: each frame's tag, never reset (the receiver keys stamps by it)
  std::atomic<uint64_t> pushErrors{0};

  // ai: the measured rate: a window from a frame, closed at the first frame a second or more later
  std::mutex fpsMu;
  int64_t fpsStart = 0;            // ai: fpsMu: the window's first frame's arrival (0 none yet)
  int64_t fpsLast = 0;             // ai: fpsMu: the newest frame's arrival
  int fpsN = 0;                    // ai: fpsMu: frames since it
  double fps = 0;                  // ai: fpsMu: the last window's rate (0 none closed since the start)

  struct Crop {
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    uint64_t seq = 0;
    int64_t at = 0;   // ai: its frame's arrival (steady ns)
  };
  // ai: A preview copies `newest` under prevMu and lets its copy go under it too (Held), so the capture thread's test
  // ai: that no preview holds the spare (its use count, under the same lock) comes after every read of it.
  std::mutex prevMu;
  std::shared_ptr<Crop> newest;    // ai: prevMu
  std::shared_ptr<Crop> spare;     // ai: prevMu: the crop before the newest, filled next once no preview holds it
  std::atomic<int64_t> asked{0};   // ai: when a preview was last asked (steady ns, 0 never)
  struct Held {
    std::mutex& mu;
    std::shared_ptr<Crop> c;
    ~Held() {
      std::lock_guard<std::mutex> l(mu);
      c.reset();
    }
  };
  // ai: The preview's scaling buffers (shrink), the session's under a leaf lock of their own, so previews asked from
  // ai: two threads at once take turns (2026-10-25: they were thread_locals, and under MinGW-w64 GCC's emulated TLS
  // ai: threads that end holding a thread_local with a destructor corrupt the heap).
  std::mutex scaleMu;
  std::vector<float> scaleMid, scaleAcc;   // ai: scaleMu

  // ai: from: the receiver whose transfer the new one goes on with (receiver.h create's from), null for one of its own
  std::shared_ptr<Rx> makeRx(const std::string& decoder, const std::string& layout, Receiver* from, std::string& why) {
    ReceiverConfig rc;
    rc.assets = cfg.assets;
    rc.cacheDir = cfg.cacheDir;
    rc.storeDir = cfg.storeDir;
    rc.decoder = decoder;
    rc.layout = layout;
    rc.log = cfg.log;
    try {
      auto r = from ? Receiver::create(rc, [](uint64_t) {}, *from) : Receiver::create(rc, [](uint64_t) {});
      if (!r) {
        why = "the receiver could not be made";
        return nullptr;
      }
      auto x = std::make_shared<Rx>();
      x->r = std::move(r);
      x->decoder = decoder;
      x->layout = layout;
      return x;
    } catch (const std::exception& e) {
      why = std::string(decoder == "gpu" ? "the GPU decoder could not start: " : "the receiver could not start: ") + e.what();
      return nullptr;
    }
  }

  // ai: the capture thread's: a frame into the receiver, its crop for the preview
  void onFrame(const cam::Frame& f) {
    if (!f.luma || !f.width || !f.height || f.stride < f.width) {
      if (pushErrors++ == 0) say("camera: a frame with no luma, or rows shorter than its width, was let go");
      return;
    }
    const int64_t now = nowNs();
    const uint64_t tag = ++tags;
    const std::shared_ptr<Rx> r = current();
    CameraFrame c;
    c.luma = f.luma;
    c.width = f.width;
    c.height = f.height;
    c.stride = f.stride;
    c.timestampNs = f.ns;
    c.tag = tag;
    try {
      if (r) r->r->push(c);
    } catch (const std::exception& e) {
      if (pushErrors++ == 0) say(std::string("receiver: a frame was refused: ") + e.what());
    }
    const int64_t a = asked.load(std::memory_order_relaxed);
    if (a && now - a < PREVIEW_HOLD_NS) keepCrop(f, r && r->layout == "2:1", tag, now);
    std::lock_guard<std::mutex> l(fpsMu);
    fpsLast = now;
    if (!fpsStart) {
      fpsStart = now;
      fpsN = 0;
      return;
    }
    fpsN++;
    if (now - fpsStart >= 1'000'000'000) {
      fps = fpsN * 1e9 / double(now - fpsStart);
      fpsStart = now;
      fpsN = 0;
    }
  }

  // ai: the decoder's crop of the frame copied, into the spare where no preview holds it (else a new one), outside the
  // ai: lock; then the newest, the newest before it the spare
  void keepCrop(const cam::Frame& f, bool two, uint64_t seq, int64_t now) {
    // ai: the region the decoder reads (core/rx/crops.h): the centre square, or the 2:1 halves whole
    const Crops c = cropsOf(f.width, f.height, two);
    const uint32_t gx = c.x[0], gy = c.y[0], gw = c.x[c.n - 1] - c.x[0] + c.side, gh = c.y[c.n - 1] - c.y[0] + c.side;
    if (!c.side) return;
    std::shared_ptr<Crop> k;
    {
      std::lock_guard<std::mutex> l(prevMu);
      if (spare && spare.use_count() == 1) k = std::move(spare);
    }
    if (!k) k = std::make_shared<Crop>();
    k->px.resize(size_t(gw) * gh);
    for (uint32_t y = 0; y < gh; y++) std::memcpy(k->px.data() + size_t(y) * gw, f.luma + size_t(gy + y) * f.stride + gx, gw);
    k->w = gw;
    k->h = gh;
    k->seq = seq;
    k->at = now;
    std::lock_guard<std::mutex> l(prevMu);
    spare = std::move(newest);
    newest = std::move(k);
  }

  void dropCrop() {
    std::lock_guard<std::mutex> l(prevMu);
    newest.reset();
    spare.reset();
  }

  // ai: the capture thread's, once, where the stream ends by itself (camera.h Ended): no camera call from here, since
  // ai: a stop under camMu may be joining this very thread
  void onEnded(const std::string& why) {
    const std::string w = why.empty() ? "the stream ended" : why;
    {
      std::lock_guard<std::mutex> l(endMu);
      ended = w;
    }
    streaming = false;
    say("camera: the stream ended: " + w);
  }

  // ai: camMu held: the camera's counts read into the cache
  void readStats() {
    cam::Stats s;
    if (camera) s = camera->stats();
    std::lock_guard<std::mutex> l(csMu);
    cs = s;
  }

  // ai: camMu held: camera()'s answer read off the camera again
  void describeNow() {
    json j = json::object();
    if (camera) {
      ctls = camera->controls();
      j = describe(*camera, modes, ctls);
      j["mode"] = mode;
      j["streaming"] = streaming.load();
    }
    std::lock_guard<std::mutex> l(ptrMu);
    described = std::move(j);
  }

  // ai: camMu held: a setting the user asked, kept (the control's earlier one replaced)
  void keep(const std::string& id, double value, bool automatic) {
    for (Want& w : wants)
      if (w.id == id) {
        w.value = value;
        w.automatic = automatic;
        return;
      }
    wants.push_back({id, value, automatic});
  }

  // ai: camMu held, a stream just begun: the settings kept set again, in the order first asked; a refusal said in the
  // ai: log (the start stands, the setting stays kept for the next)
  void setAgain() {
    if (wants.empty() || !camera) return;
    std::string said;
    for (const Want& w : wants) {
      std::string why;
      bool ok = false;
      try {
        ok = camera->set(w.id, w.value, w.automatic, why);
      } catch (const std::exception& e) {
        why = e.what();
      }
      said += (said.empty() ? "" : ", ") + w.id + " " + (w.automatic ? std::string("auto") : gnum(w.value)) +
              (ok ? std::string() : " refused (" + (why.empty() ? std::string("no reason given") : why) + ")");
    }
    say("camera: set again after the start: " + said);
  }

  // ai: camMu held. The stream stopped (its thread joined: no sink call after this), then the receiver told.
  void stopLocked() {
    if (!camera) return;
    try {
      camera->stop();
    } catch (const std::exception& e) {
      say(std::string("camera: the stop failed: ") + e.what());
    }
    // ai: every frame of the stream handed back (push copies a frame's crop, and the stop joined the capture thread):
    // ai: the receiver lets go what it kept of the stream, an ingest error with it, and ends its stats window, so a
    // ai: start reads afresh and its first window is that stream's own (receiver.h cameraClosed, as the Android app's
    // ai: at each reader it closes); once a stream, a stop after its own end too
    if (streamed) {
      streamed = false;
      if (const std::shared_ptr<Rx> r = current()) {
        try {
          r->r->cameraClosed();
        } catch (const std::exception& e) {
          say(std::string("receiver: the camera's close: ") + e.what());
        }
      }
    }
    const bool was = streaming.exchange(false);
    try {
      readStats();
    } catch (const std::exception&) {
    }
    if (was) say("camera: stopped");
  }

  // ai: camMu held: modes[index] streamed into the receiver
  bool startLocked(int index, std::string& why) {
    const cam::Mode m = modes[size_t(index)];
    {
      std::lock_guard<std::mutex> l(fpsMu);
      fpsStart = 0;
      fpsN = 0;
      fps = 0;
    }
    {
      std::lock_guard<std::mutex> l(endMu);
      ended.clear();
    }
    // ai: set before the start, so a stream that ends at once (its Ended on the capture thread before start returns)
    // ai: leaves it false
    streaming = true;
    std::string w;
    bool ok = false;
    try {
      ok = camera->start(m, [this](const cam::Frame& f) { onFrame(f); }, [this](const std::string& e) { onEnded(e); }, w);
    } catch (const std::exception& e) {
      w = e.what();
    }
    if (!ok) {
      streaming = false;
      why = w.empty() ? "the camera did not start" : w;
    } else {
      mode = index;
      streamed = true;
      starts++;
      {
        std::lock_guard<std::mutex> l(ptrMu);
        modeNow = {{"w", m.width}, {"h", m.height}, {"format", m.format}, {"fps", m.fps}};
      }
      say("camera: started " + std::to_string(m.width) + " x " + std::to_string(m.height) + " " + m.format + " at " + num(m.fps, 2) + " fps");
      setAgain();
    }
    try {
      readStats();
      describeNow();
    } catch (const std::exception&) {
    }
    return ok;
  }

  // ai: camMu held: the camera stopped and let go (here, once a stats or a delay holding it has let go too)
  void closeLocked() {
    if (!camera) return;
    stopLocked();
    std::shared_ptr<cam::Camera> c;
    {
      std::lock_guard<std::mutex> l(ptrMu);
      c = std::move(camera);
      camera = nullptr;
      info = nullptr;
      modeNow = nullptr;
      described = json::object();
    }
    modes.clear();
    ctls.clear();
    wants.clear();
    mode = -1;
    {
      std::lock_guard<std::mutex> l(csMu);
      cs = cam::Stats{};
    }
    dropCrop();
    const std::string name = c->info().name;
    letGo(c);
    say("camera: closed " + name);
  }

  double fpsNow() {
    std::lock_guard<std::mutex> l(fpsMu);
    // ai: no window closed for 1.5 s (a stall): the frames since the window's start over the time since
    const int64_t now = nowNs();
    if (fpsStart && now - fpsStart > 1'500'000'000) return fpsN * 1e9 / double(now - fpsStart);
    // ai: none closed yet in a stream's first second: its frames so far over their span, from the fourth frame on (0
    // ai: before: not measured yet)
    if (!fps && fpsN >= 3 && fpsLast > fpsStart) return fpsN * 1e9 / double(fpsLast - fpsStart);
    return fps;
  }
};

std::string describeCamera(cam::Camera& c) { return dump(describe(c, c.modes(), c.controls())); }

RxSession::RxSession(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

std::unique_ptr<RxSession> RxSession::create(const Config& c, std::string& why) {
  if (!oneOf(c.decoder, {"auto", "gpu", "cpu"}, "decoder", why) || !oneOf(c.layout, {"1:1", "2:1"}, "layout", why))
    return nullptr;
  if (c.storeDir.empty()) {
    why = "the receiver needs a folder for its store";
    return nullptr;
  }
  auto p = std::make_unique<Impl>();
  p->cfg = c;
  if (!c.cacheDir.empty()) {
    std::error_code e;
    std::filesystem::create_directories(std::filesystem::u8path(c.cacheDir), e);
  }
  p->rx = p->makeRx(c.decoder, c.layout, nullptr, why);
  if (!p->rx) return nullptr;
  return std::unique_ptr<RxSession>(new RxSession(std::move(p)));
}

std::string RxSession::cameras() {
  std::vector<std::string> notes;
  json cs = json::array();
  try {
    for (const auto& i : cam::list(&notes)) cs.push_back(infoJson(i));
  } catch (const std::exception& e) {
    notes.push_back(std::string("the cameras could not be listed: ") + e.what());
  }
  return dump({{"cameras", cs}, {"notes", notes}});
}

// ai: The camera first (no frame after its stop), then the receiver.
RxSession::~RxSession() {
  Impl& p = *p_;
  {
    std::lock_guard<std::mutex> l(p.camMu);
    p.closeLocked();
  }
  std::shared_ptr<Impl::Rx> r;
  {
    std::lock_guard<std::mutex> l(p.rxMu);
    r = std::move(p.rx);
    p.rec.reset();
  }
  letGo(r);
}

bool RxSession::reconfigure(const std::string& decoder, const std::string& layout, std::string& why) {
  Impl& p = *p_;
  if (!oneOf(decoder, {"auto", "gpu", "cpu"}, "decoder", why) || !oneOf(layout, {"1:1", "2:1"}, "layout", why)) return false;
  std::lock_guard<std::mutex> l(p.camMu);
  std::shared_ptr<Impl::Rx> fresh;
  {
    // ai: this copy let go before letGo(old) below, which otherwise waited its whole 5 s on it (every CPU
    // ai: reconfigure took 5.3 s, 2026-10-25)
    const auto cur = p.current();
    if (cur && cur->decoder == decoder && cur->layout == layout) return true;
    // ai: Made while the old one still reads (a GPU receiver's device and pipelines take up to a second cold), on the
    // ai: old one's transfer (receiver.h create's from): no second store on the folder, so a file the old receiver
    // ai: finishes before the swap, or after it with frames it still holds, is in the new one's file() and stats, and
    // ai: nothing of the transfer is taken again (2026-10-25: the new receiver's own store let the finished folder go,
    // ai: a file the old one finished meanwhile with it, and the file was received again from the start)
    fresh = p.makeRx(decoder, layout, cur ? cur->r.get() : nullptr, why);
  }
  if (!fresh) return false;
  const bool was = p.streaming.load();
  const int m = p.mode;
  if (p.camera) p.stopLocked();
  std::shared_ptr<Impl::Rx> old;
  {
    // ai: the batch cap and the replay handed on in the section that swaps (batchCap and record set them under rxMu
    // ai: too): a call from another thread lands on the old receiver before it or on the new one after it, never
    // ai: between the hand-on and the swap (2026-10-25: handed on before the camera's stop, a batchCap meanwhile
    // ai: reached the old receiver alone)
    std::lock_guard<std::mutex> k(p.rxMu);
    if (p.cap) fresh->r->batchCap(p.cap);
    if (p.rec) fresh->r->record(p.rec);
    old = std::move(p.rx);
    p.rx = fresh;
  }
  // ai: the old receiver's threads end here (a frame its decoder delivers meanwhile goes into the transfer both hold)
  letGo(old);
  p.dropCrop();   // ai: the old layout's shape
  p.say("receiver: now " + decoder + ", " + layout);
  if (!was || !p.camera || m < 0) return true;
  if (!p.startLocked(m, why)) {
    why = "the camera did not start again: " + why;
    return false;
  }
  return true;
}

bool RxSession::open(const std::string& id, std::string& why) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> l(p.camMu);
  p.closeLocked();
  std::string w;
  try {
    std::unique_ptr<cam::Camera> c = cam::open(id, w);
    if (!c) {
      why = w.empty() ? "the camera " + id + " could not be opened" : w;
      return false;
    }
    std::shared_ptr<cam::Camera> sp(std::move(c));
    std::vector<cam::Mode> ms = sp->modes();
    {
      std::lock_guard<std::mutex> k(p.ptrMu);
      p.camera = sp;
      p.info = infoJson(sp->info());
      p.modeNow = nullptr;
    }
    p.modes = std::move(ms);
    p.mode = -1;
    p.starts = 0;
    {
      std::lock_guard<std::mutex> k(p.endMu);
      p.ended.clear();
    }
    p.readStats();
    p.describeNow();
    p.say("camera: opened " + sp->info().name + " (" + sp->info().id + "), " + std::to_string(p.modes.size()) + " modes");
    return true;
  } catch (const std::exception& e) {
    why = std::string("the camera could not be opened: ") + e.what();
    p.closeLocked();
    return false;
  }
}

void RxSession::close() {
  std::lock_guard<std::mutex> l(p_->camMu);
  p_->closeLocked();
}

std::string RxSession::camera() {
  json j;
  {
    std::lock_guard<std::mutex> l(p_->ptrMu);
    j = p_->described.is_object() ? p_->described : json::object();
  }
  return dump(j);
}

bool RxSession::set(const std::string& control, double value, bool automatic, std::string& why) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> l(p.camMu);
  if (!p.camera) {
    why = "no camera is open";
    return false;
  }
  bool ok = false;
  if (p.streaming) {
    try {
      ok = p.camera->set(control, value, automatic, why);
    } catch (const std::exception& e) {
      why = e.what();
    }
    if (!ok && why.empty()) why = "the camera refused " + control;
  } else {
    // ai: no stream: kept for the next start, the camera not touched (its controls as its backend left them at the
    // ai: stop: a value set now would go back at the next stop anyway, and Media Foundation would wake the device)
    const auto k = std::find_if(p.ctls.begin(), p.ctls.end(), [&](const cam::Control& c) { return c.id == control; });
    if (k == p.ctls.end()) why = "this camera has no " + control + " to set";
    else if (automatic && !k->canAuto) why = "this camera has no automatic " + control;
    else ok = true;
  }
  if (ok) p.keep(control, value, automatic);
  if (p.streaming) {
    try {
      p.describeNow();
    } catch (const std::exception&) {
    }
  }
  return ok;
}

bool RxSession::start(int mode, std::string& why) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> l(p.camMu);
  if (!p.camera) {
    why = "no camera is open";
    return false;
  }
  if (mode < 0 || size_t(mode) >= p.modes.size()) {
    why = "the camera has no mode " + std::to_string(mode) + " (" + std::to_string(p.modes.size()) + " listed)";
    return false;
  }
  p.stopLocked();
  return p.startLocked(mode, why);
}

void RxSession::stop() {
  Impl& p = *p_;
  std::lock_guard<std::mutex> l(p.camMu);
  p.stopLocked();
  try {
    p.describeNow();
  } catch (const std::exception&) {
  }
}

bool RxSession::streaming() const { return p_->streaming.load(); }

std::string RxSession::stats() {
  Impl& p = *p_;
  json j;
  try {
    j = json::parse(p.current()->r->stats());
  } catch (const std::exception& e) {
    j = {{"state", "error"}, {"error", std::string("the receiver's stats: ") + e.what()}};
  }
  if (!j.is_object()) j = json::object();
  std::shared_ptr<cam::Camera> c;
  json o, m;
  {
    std::lock_guard<std::mutex> l(p.ptrMu);
    c = p.camera;
    o = p.info;
    m = p.modeNow;
  }
  // ai: the receiver's window is a stream's: nothing of it while none runs (after a stream's own end the receiver keeps
  // ai: its last window until the next stop tells it the camera closed); from a start the receiver's own figures,
  // ai: 0 until the stream's first window closes (each stop ends the receiver's window: receiver.h cameraClosed); the
  // ai: transfer, the word and the totals stay
  const bool live = c && p.streaming.load();
  if (!live) {
    for (const char* k : WINDOW_FIGURES)
      if (j.contains(k)) j[k] = 0;
    if (j.contains("test")) j["test"] = false;
  }
  j["live"] = live;
  if (!c) {
    j["camera"] = nullptr;
    return dump(j);
  }
  {
    // ai: the camera's own counts read where no camera call runs (they are one), else as last read
    std::unique_lock<std::mutex> k(p.camMu, std::try_to_lock);
    if (k.owns_lock() && p.camera) {
      try {
        p.readStats();
      } catch (const std::exception&) {
      }
    }
  }
  cam::Stats s;
  {
    std::lock_guard<std::mutex> l(p.csMu);
    s = p.cs;
  }
  const bool on = p.streaming.load();
  if (!o.is_object()) o = json::object();
  o["streaming"] = on;
  o["starts"] = p.starts.load();
  o["mode"] = m;
  o["fps"] = on ? p.fpsNow() : 0.0;
  o["agreedFps"] = s.fps;
  o["frames"] = s.frames;
  o["dropped"] = s.dropped;
  o["corrupt"] = s.corrupt;
  o["convertMs"] = s.convertMs;
  {
    std::lock_guard<std::mutex> l(p.endMu);
    o["ended"] = p.ended;
  }
  j["camera"] = std::move(o);
  return dump(j);
}

std::vector<double> RxSession::series(double sinceMs) { return p_->current()->r->series(sinceMs); }
std::string RxSession::file() { return p_->current()->r->file(); }
void RxSession::clear() { p_->current()->r->clear(); }
void RxSession::forget(const std::string& root) { p_->current()->r->forget(root); }

// ai: kept and set on the receiver in hand under one rxMu section (each receiver's call a store under its own leaf
// ai: lock), as reconfigure's swap hands them on: the last call is the receiver's whatever thread made it
void RxSession::batchCap(int n) {
  std::lock_guard<std::mutex> l(p_->rxMu);
  p_->cap = n;
  p_->rx->r->batchCap(n);
}

void RxSession::record(std::shared_ptr<Replay> rp) {
  std::lock_guard<std::mutex> l(p_->rxMu);
  p_->rec = rp;
  p_->rx->r->record(std::move(rp));
}

bool RxSession::preview(uint64_t since, int side, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, uint64_t& seq) {
  Impl& p = *p_;
  const int64_t now = nowNs();
  p.asked.store(now, std::memory_order_relaxed);
  Impl::Held held{p.prevMu, nullptr};
  {
    std::lock_guard<std::mutex> l(p.prevMu);
    held.c = p.newest;
  }
  const Impl::Crop* c = held.c.get();
  // ai: a crop kept from before a pause in the asking (no copies were made meanwhile) is not the newest frame's
  if (!c || c->seq <= since || now - c->at > PREVIEW_HOLD_NS || side <= 0 || !c->w || !c->h) return false;
  const uint32_t lon = std::max(c->w, c->h), s = uint32_t(side);
  if (lon <= s) {
    w = c->w;
    h = c->h;
  } else {
    w = std::clamp<uint32_t>(uint32_t(std::lround(double(c->w) * s / lon)), 1, s);
    h = std::clamp<uint32_t>(uint32_t(std::lround(double(c->h) * s / lon)), 1, s);
  }
  out.resize(size_t(w) * h);
  {
    std::lock_guard<std::mutex> l(p.scaleMu);
    shrink(c->px.data(), c->w, c->h, out.data(), w, h, p.scaleMid, p.scaleAcc);
  }
  seq = c->seq;
  return true;
}

}  // namespace lizard
