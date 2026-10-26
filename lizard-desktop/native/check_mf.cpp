// ai: cam_check mf (2026-10-25, checks.h): Media Foundation's backend (camera/mf.cpp) against sim/fake_mf.cpp's source,
// ai: through the backend's own path (camera/mf_source.h): Media Foundation's source reader, its MJPEG decoder, the
// ai: capture thread. Cases: load (MF's DLLs and MFStartup); list (the system's cameras: none is no failure, the count
// ai: printed); registry (a test source listed and opened by id through camera.h's registry); modes; controls (the
// ai: fake has none: none listed, a set refused); stream-<type> for each of the fake's six types (frames at about the
// ai: rate, ns rising a frame apart, each frame the picture its time says, the luma the gradient's: exact for the
// ai: uncompressed, bottom-up RGB included, within JPEG's loss through the decoder); restart (a second stream, on a
// ai: new source); rules (a fake with a webcam's controls: listed, exposure in us, the camera rules at start, sets,
// ai: everything put back at stop and at close, a look at the stopped camera answered with no source made); rate
// ai: (MJPG asked at 15 inside its 5 to 30 range: the source runs it at 15, the backend agrees to 15); dead (a
// ai: stopped camera's source shut down behind it: the start that finds it fails, the next makes a new source and
// ai: streams); stall (stop on a stream gone quiet); lost (the device's error ends the stream once, with why); corrupt
// ai: (every tenth MJPG frame cut short: the stream goes on); paint (MJPG encoded a frame at a time from a paint, the
// ai: gradient by the capture's time); session (the simulated screen through MJPG into the session's CPU decoder,
// ai: blocks verified, its stop making no source). Every stream's agreed rate (Stats::fps) its mode's. Arguments:
// ai: secs=<s> (each stream, 3), only=<case>[,<case>...]. Windows only; elsewhere checks.h's kNotHere.
#include "checks.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mferror.h>
#include <strmif.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "camera/backends.h"
#include "camera/camera.h"
#include "camera/mf_source.h"
#include "camera/clock.h"
#include "json.hpp"
#include "rx_session.h"
#include "sim/fake_mf.h"
#include "sim/scene.h"
#endif

namespace lizard::check {

#if !defined(_WIN32)

int mf(const Args&) { return kNotHere; }

#else

namespace {

using sim::fakeMfGradient;
using sim::kFakeMfPictures;

struct Report {
  int failed = 0;
  void ok(const std::string& c, const std::string& s) { line("ok", c, s); }
  void fail(const std::string& c, const std::string& s) {
    line("FAIL", c, s);
    failed++;
  }
  static void line(const char* k, const std::string& c, const std::string& s) {
    std::printf("%s %s: %s\n", k, c.c_str(), s.c_str());
    std::fflush(stdout);
  }
};

std::string fmt(const char* f, ...) {
  char b[1024];
  va_list v;
  va_start(v, f);
  std::vsnprintf(b, sizeof b, f, v);
  va_end(v);
  return b;
}

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// ai: the fake sources a camera is made over: a new one each stream, the newest kept to steer and to read
struct Maker {
  sim::FakeMfConfig cfg;
  std::mutex mu;
  sim::FakeMf* last = nullptr;
  int made = 0;
  Maker() = default;
  Maker(const Maker&) = delete;
  Maker& operator=(const Maker&) = delete;
  ~Maker() {
    if (last) last->Release();
  }
  cam::MakeSource fn() {
    return [this](std::string& why) -> IMFMediaSource* {
      sim::FakeMf* f = sim::FakeMf::create(cfg, why);
      if (!f) return nullptr;
      std::lock_guard<std::mutex> lk(mu);
      if (last) last->Release();
      last = f;
      f->AddRef();
      made++;
      return f;
    };
  }
  template <class F> void with(F f) {
    std::lock_guard<std::mutex> lk(mu);
    if (last) f(*last);
  }
  int count() {
    std::lock_guard<std::mutex> lk(mu);
    return made;
  }
  std::string stats() {
    std::string s = "{}";
    with([&](sim::FakeMf& f) { s = f.stats(); });
    return s;
  }
};

// ai: the gradient's picture a frame shows (the best of the cycle over a 12 x 12 grid) and its mean error there
int pictureOf(const uint8_t* l, uint32_t w, uint32_t h, uint32_t stride, double* err) {
  int best = -1;
  double bestErr = 1e9;
  for (uint32_t p = 0; p < kFakeMfPictures; p++) {
    double e = 0;
    for (uint32_t gy = 0; gy < 12; gy++)
      for (uint32_t gx = 0; gx < 12; gx++) {
        const uint32_t x = (2 * gx + 1) * w / 24, y = (2 * gy + 1) * h / 24;
        e += std::abs(int(l[size_t(y) * stride + x]) - int(fakeMfGradient(x, y, p)));
      }
    e /= 144;
    if (e < bestErr) {
      bestErr = e;
      best = int(p);
    }
  }
  if (err) *err = bestErr;
  return best;
}

// ai: what the sink saw of one stream
struct Rec {
  std::mutex mu;
  std::vector<int64_t> ns, at;
  std::vector<uint64_t> seq;
  std::vector<int> pic;
  std::vector<double> perr;
  std::vector<std::vector<uint8_t>> kept;
  std::vector<int> keptPic;
  uint32_t w = 0, h = 0;
  int ended = 0;
  std::string why;
};

cam::Camera::Sink sinkOf(Rec& r) {
  return [&r](const cam::Frame& f) {
    const int64_t now = cam::steadyNs();
    double e = 0;
    const int p = pictureOf(f.luma, f.width, f.height, f.stride, &e);
    std::lock_guard<std::mutex> lk(r.mu);
    const size_t i = r.ns.size();
    r.w = f.width;
    r.h = f.height;
    r.ns.push_back(f.ns);
    r.at.push_back(now);
    r.seq.push_back(f.seq);
    r.pic.push_back(p);
    r.perr.push_back(e);
    if (i == 3 || i == 12 || i == 40) {
      std::vector<uint8_t> c(size_t(f.width) * f.height);
      for (uint32_t y = 0; y < f.height; y++)
        std::memcpy(c.data() + size_t(f.width) * y, f.luma + size_t(f.stride) * y, f.width);
      r.kept.push_back(std::move(c));
      r.keptPic.push_back(p);
    }
  };
}

cam::Camera::Ended endedOf(Rec& r) {
  return [&r](const std::string& why) {
    std::lock_guard<std::mutex> lk(r.mu);
    r.ended++;
    r.why = why;
  };
}

// ai: a kept frame against picture p of the gradient: the worst difference, the PSNR, and the line a, b of
// ai: luma = a truth + b (a decoder's range change shows there)
struct Cmp { int maxDiff = 0; double psnr = 0, a = 1, b = 0; };
Cmp compare(const std::vector<uint8_t>& l, uint32_t w, uint32_t h, int p) {
  Cmp c;
  double se = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = double(w) * h;
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      const int t = fakeMfGradient(x, y, uint32_t(p)), v = l[size_t(w) * y + x], d = v - t;
      c.maxDiff = std::max(c.maxDiff, std::abs(d));
      se += double(d) * d;
      sx += t;
      sy += v;
      sxx += double(t) * t;
      sxy += double(t) * v;
    }
  const double mse = se / n;
  c.psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
  const double den = n * sxx - sx * sx;
  c.a = den ? (n * sxy - sx * sy) / den : 1;
  c.b = (sy - c.a * sx) / n;
  return c;
}

std::string modeName(const cam::Mode& m) { return fmt("%s %ux%u@%.4g", m.format.c_str(), m.width, m.height, m.fps); }

// ai: One stream of m for secs, checked: the frames' count against the rate, their ns against the schedule (rising,
// ai: a frame apart, the gaps the camera's dropped count), their pictures against their times, the kept frames'
// ai: luma against the gradient (exact, or within JPEG's loss where lossy), the camera's stats.
void stream(Report& out, const std::string& name, cam::Camera& c, const cam::Mode& m, double secs, double fps, bool lossy,
            Maker& mk) {
  Rec r;
  std::string why;
  if (!c.start(m, sinkOf(r), endedOf(r), why)) {
    out.fail(name, "start " + modeName(m) + ": " + why);
    return;
  }
  sleepMs(int(secs * 1000));
  c.stop();
  const cam::Stats st = c.stats();
  std::lock_guard<std::mutex> lk(r.mu);
  std::vector<std::string> bad;
  const size_t n = r.ns.size();
  const double iv = 1e9 / fps;
  if (r.ended) bad.push_back("ended: " + r.why);
  if (double(n) < fps * (secs - 1.0)) bad.push_back(fmt("%zu frames in %.1f s", n, secs));
  if (r.w != m.width || r.h != m.height) bad.push_back(fmt("frames %ux%u", r.w, r.h));
  double rate = 0, median = 0, minGap = 1e18;
  uint64_t gaps = 0, seqBad = 0, picBad = 0, backwards = 0;
  if (n >= 2) {
    rate = double(n - 1) * 1e9 / double(r.at.back() - r.at.front());
    std::vector<double> d;
    for (size_t i = 1; i < n; i++) {
      const double g = double(r.ns[i] - r.ns[i - 1]);
      if (g <= 0) backwards++;
      d.push_back(g);
      minGap = std::min(minGap, g);
      const int64_t k = std::llround(g / iv);
      if (k > 1) gaps += uint64_t(k - 1);
      if (int64_t(r.seq[i] - r.seq[i - 1]) != k) seqBad++;
      if (r.pic[i] >= 0 && r.pic[i - 1] >= 0 &&
          (r.pic[i] - r.pic[i - 1] + int(kFakeMfPictures) * 4) % int(kFakeMfPictures) != int(k % kFakeMfPictures))
        picBad++;
    }
    std::nth_element(d.begin(), d.begin() + ptrdiff_t(d.size() / 2), d.end());
    median = d[d.size() / 2];
    if (std::fabs(rate / fps - 1) > 0.1) bad.push_back(fmt("%.2f frames a second arrived", rate));
    if (std::fabs(median / iv - 1) > 0.01) bad.push_back(fmt("ns a median %.3f ms apart", median / 1e6));
    if (backwards) bad.push_back(fmt("%llu frames' ns not rising", (unsigned long long)backwards));
    if (minGap < 0.5 * iv) bad.push_back(fmt("two frames %.3f ms apart", minGap / 1e6));
    if (seqBad) bad.push_back(fmt("%llu seq steps unlike the ns", (unsigned long long)seqBad));
    if (picBad) bad.push_back(fmt("%llu frames not the picture their time says", (unsigned long long)picBad));
  }
  // ai: the device's mode the one asked (the fake's current type at its Start)
  const std::string fake = mk.stats();
  if (fake.find("\"type\":\"" + m.format + fmt(" %ux%u@", m.width, m.height)) == std::string::npos)
    bad.push_back("the source ran another type");
  if (st.frames != n) bad.push_back(fmt("stats say %llu frames", (unsigned long long)st.frames));
  if (std::fabs(st.fps - m.fps) > 1e-6 * m.fps) bad.push_back(fmt("stats say the camera agreed to %.4g fps", st.fps));
  if (st.dropped != gaps) bad.push_back(fmt("stats say %llu dropped, the gaps %llu", (unsigned long long)st.dropped,
                                            (unsigned long long)gaps));
  if (st.corrupt) bad.push_back(fmt("%llu corrupt", (unsigned long long)st.corrupt));
  std::string luma;
  if (r.kept.empty()) bad.push_back("no frame kept");
  for (size_t i = 0; i < r.kept.size(); i++) {
    const Cmp k = compare(r.kept[i], r.w, r.h, r.keptPic[i]);
    if (lossy) {
      if (k.psnr < 30) bad.push_back(fmt("kept frame %zu %.1f dB from its picture", i, k.psnr));
      if (i == 0) luma = fmt("luma %.1f dB from the gradient (luma = %.3f truth %+.2f)", k.psnr, k.a, k.b);
    } else {
      if (k.maxDiff) bad.push_back(fmt("kept frame %zu off its picture by up to %d", i, k.maxDiff));
      if (i == 0) luma = "luma exact";
    }
  }
  const std::string what =
      fmt("%s: %zu frames in %.1f s, %.2f a second, ns a median %.3f ms apart (%.3f asked), dropped %llu, ",
          modeName(m).c_str(), n, secs, rate, median / 1e6, iv / 1e6, (unsigned long long)st.dropped) +
      luma + fmt(", to luma %.3f ms; fake %s", st.convertMs, fake.c_str());
  if (bad.empty()) out.ok(name, what);
  else {
    std::string s;
    for (auto& b : bad) s += (s.empty() ? "" : "; ") + b;
    out.fail(name, s + " (" + what + ")");
  }
}

const cam::Mode* find(const std::vector<cam::Mode>& ms, const char* format) {
  for (auto& m : ms)
    if (m.format == format) return &m;
  return nullptr;
}

cam::Info fakeInfo() {
  cam::Info i;
  i.id = "mf:fake";
  i.name = "Fake webcam";
  i.backend = "uvc";
  i.detail = "test source";
  return i;
}

// ai: The simulated screen (sim/scene.h: the test stream, the sender at 15 pictures a second for a 30 fps camera)
// ai: through the fake's MJPG at 1920 x 1080, WIC-encoded a frame at a time and decoded by Media Foundation, into
// ai: the session (rx_session.h) on the CPU decoder by its id (mf:test:scene, camera.h's registry): blocks verified.
void session(Report& out, double secs) {
  const std::string name = "session";
  sim::Sender s;
  sim::Display d;
  sim::Optics o;
  std::string why;
  if (!sim::parse("blocks=24 fps=15 w=1920 h=1080", s, d, o, why)) {
    out.fail(name, "scene spec: " + why);
    return;
  }
  auto scene = sim::Scene::create(s, d, o, 2, why);
  if (!scene) {
    out.fail(name, "scene: " + why);
    return;
  }
  Maker mk;
  // ai: the scene renders the optics' frame (1920 x 1080, the MJPG type's); a type of another size gets the gradient
  mk.cfg.paint = [&scene](int64_t t0, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride) {
    if (w == scene->optics().width && h == scene->optics().height) scene->render(t0, luma, stride);
    else sim::fakeMfPaint(0, w, h, luma, stride);
  };
  cam::mfTestSource("scene", "Simulated screen", mk.fn());
  const std::filesystem::path store = std::filesystem::temp_directory_path() / "lizard-check-mf";
  std::error_code ec;
  std::filesystem::remove_all(store, ec);
  std::filesystem::create_directories(store, ec);
  RxSession::Config c;
  c.assets = LIZ_ASSETS;   // ai: the build's GPU tree; the CPU decoder reads none
  c.storeDir = store.u8string();
  c.decoder = "cpu";
  c.layout = "1:1";
  auto rx = RxSession::create(c, why);
  if (!rx) {
    out.fail(name, "session: " + why);
    cam::mfTestSource("scene", "", nullptr);
    return;
  }
  if (!rx->open("mf:test:scene", why)) {
    out.fail(name, "open mf:test:scene: " + why);
  } else {
    int idx = -1;
    const auto cj = nlohmann::json::parse(rx->camera(), nullptr, false);
    if (cj.is_object() && cj.contains("modes"))
      for (size_t i = 0; i < cj["modes"].size(); i++)
        if (cj["modes"][i].value("format", "") == "MJPG") idx = int(i);
    if (idx < 0) out.fail(name, "no MJPG mode in " + rx->camera());
    else if (!rx->start(idx, why)) out.fail(name, "start: " + why);
    else {
      int64_t blocks = 0, frames = 0;
      double camFps = 0;
      std::string state, decoder;
      const auto t0 = std::chrono::steady_clock::now();
      const auto until = t0 + std::chrono::milliseconds(int(secs * 1000) + 12000);
      while (std::chrono::steady_clock::now() < until && blocks < 200) {
        sleepMs(500);
        const auto j = nlohmann::json::parse(rx->stats(), nullptr, false);
        if (!j.is_object()) continue;
        if (j.contains("totals") && j["totals"].is_object()) blocks = j["totals"].value("blocks", int64_t(0));
        state = j.value("state", "");
        decoder = j.value("decoder", "");
        if (j.contains("camera") && j["camera"].is_object()) {
          frames = j["camera"].value("frames", int64_t(0));
          camFps = j["camera"].value("fps", 0.0);
        }
      }
      const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const std::string fake = mk.stats();   // ai: the streaming source's
      const int madeAt = mk.count();
      rx->stop();   // ai: the session describes the camera after it (its controls): no source may be made for that
      const int woke = mk.count() - madeAt;
      // ai: the frames the fake's MJPG ones (its current type at Start), so Media Foundation's decoder made the luma
      const bool mjpg = fake.find("\"type\":\"MJPG 1920x1080@") != std::string::npos;
      if (!mjpg) out.fail(name, "the source ran another type; fake " + fake);
      else if (woke) out.fail(name, fmt("the stop made %d source(s): the stopped camera's device woken", woke));
      else if (blocks > 0)
        out.ok(name, fmt("%lld blocks verified in %.1f s on the %s decoder, the camera %lld frames at %.1f a second "
                         "(simulated: LIZARD at 24 blocks, 15 pictures a second, MJPG 1920x1080 at 30 through Media "
                         "Foundation's decoder), state %s, no source made at the stop; fake %s",
                         (long long)blocks, took, decoder.c_str(), (long long)frames, camFps, state.c_str(),
                         fake.c_str()));
      else out.fail(name, "no block verified, state " + state + "; " + rx->stats().substr(0, 400));
    }
  }
  rx.reset();
  cam::mfTestSource("scene", "", nullptr);
  std::filesystem::remove_all(store, ec);
}

}  // namespace

int mf(const Args& a) {
  double secs = 3;
  std::set<std::string> only;
  for (const std::string& s : a) {
    if (s.rfind("secs=", 0) == 0) {
      secs = std::atof(s.c_str() + 5);
      if (!(secs >= 1.5 && secs <= 60)) {
        std::fprintf(stderr, "mf: secs from 1.5 to 60\n");
        return kUsage;
      }
    } else if (s.rfind("only=", 0) == 0) {
      size_t i = 5;
      while (i <= s.size()) {
        const size_t j = std::min(s.find(',', i), s.size());
        if (j > i) only.insert(s.substr(i, j - i));
        i = j + 1;
      }
    } else {
      std::fprintf(stderr, "mf: secs=<s> only=<case>[,<case>...]\n");
      return kUsage;
    }
  }
  auto want = [&](const std::string& c) { return only.empty() || only.count(c) > 0; };
  // ai: this thread's COM, multithreaded (the fake's WIC encoder may run on it, inside the reader's Start)
  const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  Report out;
  std::string why;
  if (!cam::mfReady(&why)) {
    out.fail("load", why);
    if (SUCCEEDED(com)) CoUninitialize();
    return kFailed;
  }
  out.ok("load", "mfplat, mf and mfreadwrite from System32, MFStartup");

  if (want("list")) {
    std::vector<std::string> notes;
    const auto t0 = std::chrono::steady_clock::now();
    const auto l = cam::mfList(&notes);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string s = fmt("%zu camera(s) on this system in %.1f ms", l.size(), ms);
    for (auto& i : l) s += "; " + i.name + " [" + i.detail + "]";
    for (auto& n : notes) s += "; note: " + n;
    out.ok("list", s);
  }

  if (want("registry")) {
    Maker mk;
    cam::mfTestSource("fake", "Fake webcam", mk.fn());
    std::vector<std::string> notes;
    const auto l = cam::list(&notes);
    const bool listed = std::any_of(l.begin(), l.end(), [](const cam::Info& i) {
      return i.id == "mf:test:fake" && i.name == "Fake webcam" && i.backend == "uvc";
    });
    auto c = cam::open("mf:test:fake", why);
    const size_t nm = c ? c->modes().size() : 0;
    const std::string id = c ? c->info().id : "";
    c.reset();
    cam::mfTestSource("fake", "", nullptr);
    const auto l2 = cam::mfList(nullptr);
    const bool gone = std::none_of(l2.begin(), l2.end(), [](const cam::Info& i) { return i.id == "mf:test:fake"; });
    if (listed && nm == 6 && id == "mf:test:fake" && gone)
      out.ok("registry", fmt("mf:test:fake listed by camera.h's list, opened by its id (%zu modes), taken off", nm));
    else
      out.fail("registry", fmt("listed %d, opened %s (%s), %zu modes, taken off %d", listed, id.c_str(), why.c_str(), nm,
                               gone));
  }

  Maker mk;
  auto cam = cam::mfOpenSource(mk.fn(), fakeInfo(), why);
  if (!cam) {
    out.fail("open", "the fake through mfOpenSource: " + why);
    if (SUCCEEDED(com)) CoUninitialize();
    return kFailed;
  }
  const std::vector<cam::Mode> modes = cam->modes();

  if (want("modes")) {
    struct Want { const char* f; uint32_t w, h; double fps, lo; cam::Pix pix; bool comp; };
    const Want ws[] = {{"MJPG", 1920, 1080, 30, 5, cam::Pix::MJPEG, true},
                       {"YUY2", 1280, 720, 30, 30, cam::Pix::YUYV, false},
                       {"NV12", 640, 480, 60, 60, cam::Pix::NV12, false},
                       {"L8", 640, 480, 30, 30, cam::Pix::Grey8, false},
                       {"RGB24", 320, 240, 30, 30, cam::Pix::BGR24, false},
                       {"RGB32", 320, 240, 15, 15, cam::Pix::BGRA32, false}};
    constexpr size_t kWant = sizeof ws / sizeof ws[0];
    std::string s, badS;
    for (size_t i = 0; i < modes.size(); i++) {
      const auto& m = modes[i];
      s += (i ? ", " : "") + fmt("%s %ux%u %.4g", m.format.c_str(), m.width, m.height, m.fps) +
           (m.fpsMin < m.fps ? fmt(" (from %.4g)", m.fpsMin) : std::string()) + fmt(" key %llu", (unsigned long long)m.key);
    }
    if (modes.size() != kWant) badS = fmt("%zu modes", modes.size());
    for (size_t i = 0; i < kWant && i < modes.size(); i++) {
      const auto& m = modes[i];
      const Want& w = ws[i];
      if (m.format != w.f || m.width != w.w || m.height != w.h || std::fabs(m.fps - w.fps) > 1e-6 ||
          std::fabs(m.fpsMin - w.lo) > 1e-6 || m.pix != w.pix || m.compressed != w.comp || m.key != i)
        badS += fmt(" mode %zu wrong", i);
    }
    if (badS.empty()) out.ok("modes", s);
    else out.fail("modes", badS + ": " + s);
  }

  if (want("controls")) {
    const auto cs = cam->controls();
    std::string w2;
    const bool refused = !cam->set("zoom", 2, false, w2) && !w2.empty();
    std::string w3;
    const bool refused2 = !cam->set("nonsense", 1, false, w3) && !w3.empty();
    if (cs.empty() && refused && refused2)
      out.ok("controls", "none listed; zoom refused (" + w2 + "); an unknown id refused (" + w3 + ")");
    else out.fail("controls", fmt("%zu listed, zoom refused %d (%s)", cs.size(), refused, w2.c_str()));
  }

  const struct { const char* c; const char* f; bool lossy; } streams[] = {
      {"stream-mjpg", "MJPG", true}, {"stream-yuy2", "YUY2", false}, {"stream-nv12", "NV12", false},
      {"stream-l8", "L8", false},    {"stream-rgb24", "RGB24", false}, {"stream-rgb32", "RGB32", false}};
  for (auto& s : streams) {
    if (!want(s.c)) continue;
    const cam::Mode* m = find(modes, s.f);
    if (!m) {
      out.fail(s.c, std::string("no ") + s.f + " mode");
      continue;
    }
    stream(out, s.c, *cam, *m, secs, m->fps, s.lossy, mk);
  }

  if (want("restart")) {
    // ai: a stream, stopped, then another checked whole: the second on a source made anew
    const cam::Mode* m = find(modes, "NV12");
    Rec r;
    std::string w;
    if (!m || !cam->start(*m, sinkOf(r), endedOf(r), w)) {
      out.fail("restart", "first start: " + w);
    } else {
      sleepMs(500);
      cam->stop();
      const int before = mk.made;
      stream(out, "restart", *cam, *m, std::max(2.0, secs - 1), m->fps, false, mk);
      if (mk.made != before + 1) out.fail("restart", fmt("%d sources made for the second stream", mk.made - before));
    }
  }

  if (want("rate")) {
    // ai: The fake's handler (Media Foundation's own, for the fake's stream descriptor) takes a type at a changed rate,
    // ai: so a backend that asks for 15 gets it: the source runs the type at 15, the frames come at 15 and the backend
    // ai: agreed to 15. 2026-10-25: frames at the type's own 30 passed too ("the handler refused"), which here only a
    // ai: backend that never asked, or asked wrongly, could make.
    const cam::Mode* m = find(modes, "MJPG");
    if (!m) {
      out.fail("rate", "no MJPG mode");
    } else {
      cam::Mode at15 = *m;
      at15.fps = 15;
      Rec r;
      std::string w;
      if (!cam->start(at15, sinkOf(r), endedOf(r), w)) {
        out.fail("rate", "start MJPG at 15: " + w);
      } else {
        sleepMs(int(secs * 1000));
        const std::string f = mk.stats();   // ai: the streaming source's
        cam->stop();
        const cam::Stats st = cam->stats();
        std::lock_guard<std::mutex> lk(r.mu);
        double med = 0;
        if (r.ns.size() > 2) {
          std::vector<double> d;
          for (size_t i = 1; i < r.ns.size(); i++) d.push_back(double(r.ns[i] - r.ns[i - 1]));
          std::nth_element(d.begin(), d.begin() + ptrdiff_t(d.size() / 2), d.end());
          med = d[d.size() / 2];
        }
        const double got = med > 0 ? 1e9 / med : 0;
        const bool ran15 = f.find("\"type\":\"MJPG 1920x1080@15\"") != std::string::npos;
        const std::string what = fmt("MJPG asked at 15 inside 5 to 30: frames at %.2f (%zu), the camera agreed to %.4g; fake %s",
                                     got, r.ns.size(), st.fps, f.c_str());
        if (std::fabs(got - 15) < 0.2 && ran15 && std::fabs(st.fps - 15) < 1e-6) out.ok("rate", what);
        else out.fail("rate", what);
      }
    }
  }

  if (want("dead")) {
    // ai: a stopped camera's source shut down behind it (a device reset or gone while stopped): the start that finds
    // ai: it fails saying why and lets it go, so the next start activates a new source and streams
    Maker md;
    auto c5 = cam::mfOpenSource(md.fn(), fakeInfo(), why);
    const std::vector<cam::Mode> ms5 = c5 ? c5->modes() : std::vector<cam::Mode>{};
    const cam::Mode* m = find(ms5, "NV12");
    if (!c5 || !m) {
      out.fail("dead", "open: " + why);
    } else {
      md.with([](sim::FakeMf& f) { f.Shutdown(); });
      Rec r1, r2;
      std::string w1, w2;
      const bool first = c5->start(*m, sinkOf(r1), endedOf(r1), w1);
      if (first) c5->stop();
      const int before = md.count();
      const bool second = c5->start(*m, sinkOf(r2), endedOf(r2), w2);
      if (second) {
        sleepMs(600);
        c5->stop();
      }
      const int made = md.count() - before;
      size_t frames;
      {
        std::lock_guard<std::mutex> lk(r2.mu);
        frames = r2.ns.size();
      }
      const std::string what = fmt("the first start %s (\"%s\"), the second %s on %d new source(s), %zu frames in 0.6 s",
                                   first ? "STREAMED" : "failed", w1.c_str(), second ? "streamed" : "FAILED", made, frames);
      if (!first && !w1.empty() && second && made == 1 && frames > 10) out.ok("dead", what + (w2.empty() ? "" : " (" + w2 + ")"));
      else out.fail("dead", what + (w2.empty() ? "" : "; the second's why: " + w2));
    }
    c5.reset();
  }

  if (want("stall")) {
    const cam::Mode* m = find(modes, "NV12");
    Rec r;
    std::string w;
    if (m && cam->start(*m, sinkOf(r), endedOf(r), w)) {
      sleepMs(600);
      mk.with([](sim::FakeMf& f) { f.stall(true); });
      sleepMs(400);
      size_t before;
      {
        std::lock_guard<std::mutex> lk(r.mu);
        before = r.ns.size();
      }
      const auto t0 = std::chrono::steady_clock::now();
      cam->stop();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      const std::string f = mk.stats();
      std::lock_guard<std::mutex> lk(r.mu);
      if (ms < 3000 && !r.ended && before > 10)
        out.ok("stall", fmt("stop on a quiet stream in %.0f ms (%zu frames before it went quiet); fake %s", ms, before, f.c_str()));
      else out.fail("stall", fmt("stop took %.0f ms, ended %d (%s), %zu frames; fake %s", ms, r.ended, r.why.c_str(), before, f.c_str()));
    } else {
      out.fail("stall", "start: " + w);
    }
  }

  if (want("lost")) {
    const cam::Mode* m = find(modes, "NV12");
    Rec r;
    std::string w;
    if (m && cam->start(*m, sinkOf(r), endedOf(r), w)) {
      sleepMs(500);
      mk.with([](sim::FakeMf& f) { f.fail(MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED); });
      const auto t0 = std::chrono::steady_clock::now();
      double ms = 0;
      for (;;) {
        {
          std::lock_guard<std::mutex> lk(r.mu);
          if (r.ended) break;
        }
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > 3000) break;
        sleepMs(5);
      }
      const auto t1 = std::chrono::steady_clock::now();
      cam->stop();
      const double stopMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
      sleepMs(200);
      std::lock_guard<std::mutex> lk(r.mu);
      if (r.ended == 1 && r.why == "the camera was disconnected" && stopMs < 1000)
        out.ok("lost", fmt("ended once in %.0f ms: \"%s\"; stop then %.0f ms", ms, r.why.c_str(), stopMs));
      else out.fail("lost", fmt("ended %d (\"%s\") after %.0f ms, stop %.0f ms", r.ended, r.why.c_str(), ms, stopMs));
    } else {
      out.fail("lost", "start: " + w);
    }
  }

  if (want("corrupt")) {
    Maker mc;
    mc.cfg.corruptEvery = 10;
    auto c2 = cam::mfOpenSource(mc.fn(), fakeInfo(), why);
    const std::vector<cam::Mode> ms2 = c2 ? c2->modes() : std::vector<cam::Mode>{};
    const cam::Mode* m = find(ms2, "MJPG");
    Rec r;
    std::string w;
    if (m && c2->start(*m, sinkOf(r), endedOf(r), w)) {
      sleepMs(int(secs * 1000));
      c2->stop();
      const cam::Stats st = c2->stats();
      std::lock_guard<std::mutex> lk(r.mu);
      size_t off = 0;
      for (double e : r.perr)
        if (e > 6) off++;
      const std::string s = fmt("%zu frames in %.1f s with every tenth cut to half: %zu of them off their picture "
                                "(mean error over 6), corrupt %llu, dropped %llu; fake %s",
                                r.ns.size(), secs, off, (unsigned long long)st.corrupt, (unsigned long long)st.dropped,
                                mc.stats().c_str());
      if (!r.ended && double(r.ns.size()) >= 0.6 * m->fps * (secs - 1)) out.ok("corrupt", s);
      else out.fail("corrupt", (r.ended ? "ended: " + r.why + "; " : std::string()) + s);
    } else {
      out.fail("corrupt", "start: " + (c2 ? w : why));
    }
    c2.reset();
  }

  if (want("rules")) {
    // ai: a fake device with a webcam's controls (kept across its sources, as a device keeps them): the list
    // ai: (exposure in us), the camera rules at start, set() streaming and stopped, everything put back at stop and
    // ai: when the camera goes
    auto ctl = sim::FakeMfControls::webcam();
    Maker mr;
    mr.cfg.controls = ctl;
    std::vector<std::string> bad;
    auto prop = [&](bool camCtl, long id) {
      std::lock_guard<std::mutex> lk(ctl->mu);
      return (camCtl ? ctl->cam : ctl->amp)[id];
    };
    auto aep = [&] {
      std::lock_guard<std::mutex> lk(ctl->mu);
      return ctl->aep;
    };
    auto expect = [&](const char* what, bool camCtl, long id, long value, long flags) {
      const auto q = prop(camCtl, id);
      if (q.value != value || q.flags != flags)
        bad.push_back(fmt("%s: %ld (flags %ld), %ld (%ld) wanted", what, q.value, q.flags, value, flags));
    };
    auto c4 = cam::mfOpenSource(mr.fn(), fakeInfo(), why);
    const std::vector<cam::Mode> ms4 = c4 ? c4->modes() : std::vector<cam::Mode>{};
    const cam::Mode* m = find(ms4, "NV12");
    std::string listed;
    if (!c4 || !m) {
      bad.push_back("open: " + why);
    } else {
      const auto cs = c4->controls();
      for (auto& k : cs) listed += (listed.empty() ? "" : ", ") + k.id;
      if (listed != "zoom, focus, exposure, gain, sharpness, backlight, brightness, contrast")
        bad.push_back("listed " + listed);
      for (auto& k : cs) {
        if (k.id == "exposure" && (std::fabs(k.min - 488.28125) > 1e-6 || std::fabs(k.max - 250000) > 1e-6 ||
                                   std::fabs(k.value - 31250) > 1e-6 || !k.canAuto || !k.isAuto || k.unit != "us"))
          bad.push_back(fmt("exposure listed %.5g to %.5g at %.5g us, auto %d %d", k.min, k.max, k.value, k.canAuto,
                            k.isAuto));
        if (k.id == "zoom" && (k.min != 100 || k.max != 500 || k.step != 1 || k.value != 100 || k.canAuto))
          bad.push_back(fmt("zoom listed %.5g to %.5g by %.5g at %.5g", k.min, k.max, k.step, k.value));
      }
      Rec r;
      std::string w;
      if (!c4->start(*m, sinkOf(r), endedOf(r), w)) {
        bad.push_back("start: " + w);
      } else {
        sleepMs(300);
        // ai: the rules: sharpness at its minimum, backlight off, the priority 0, all manual
        expect("sharpness streaming", false, VideoProcAmp_Sharpness, 0, 2);
        expect("backlight streaming", false, VideoProcAmp_BacklightCompensation, 0, 2);
        if (aep() != 0) bad.push_back(fmt("auto-exposure priority %ld streaming", aep()));
        std::string w1, w2, w3, w4;
        if (!c4->set("zoom", 251.6, false, w1)) bad.push_back("set zoom: " + w1);
        if (!c4->set("exposure", 10000, false, w2)) bad.push_back("set exposure: " + w2);
        if (!c4->set("focus", 0, true, w3)) bad.push_back("set focus auto: " + w3);
        const bool noAuto = !c4->set("zoom", 0, true, w4) && !w4.empty();
        if (!noAuto) bad.push_back("automatic zoom taken");
        expect("zoom set", true, CameraControl_Zoom, 252, 2);
        expect("exposure set", true, CameraControl_Exposure, -7, 2);
        for (auto& k : c4->controls())
          if (k.id == "exposure" && (std::fabs(k.value - 7812.5) > 1e-6 || k.isAuto))
            bad.push_back(fmt("exposure read back %.5g us, auto %d", k.value, k.isAuto));
        c4->stop();
        // ai: put back: the rules' and the sets'
        expect("sharpness stopped", false, VideoProcAmp_Sharpness, 128, 2);
        expect("backlight stopped", false, VideoProcAmp_BacklightCompensation, 1, 2);
        expect("zoom stopped", true, CameraControl_Zoom, 100, 2);
        expect("exposure stopped", true, CameraControl_Exposure, -5, 1);
        expect("focus stopped", true, CameraControl_Focus, 0, 1);
        if (aep() != 1) bad.push_back(fmt("auto-exposure priority %ld stopped", aep()));
        // ai: a look at the stopped camera wakes no device: no source made, the controls as put back, the modes
        const int madeAt = mr.count();
        const auto cs = c4->controls();
        const auto msNow = c4->modes();
        if (mr.count() != madeAt) bad.push_back(fmt("a look at the stopped camera made %d source(s)", mr.count() - madeAt));
        const auto z = std::find_if(cs.begin(), cs.end(), [](const cam::Control& k) { return k.id == "zoom"; });
        if (cs.size() != 8 || z == cs.end() || z->value != 100) bad.push_back(fmt("the stopped camera's controls: %zu, zoom %s", cs.size(), z == cs.end() ? "none" : fmt("%.5g", z->value).c_str()));
        if (msNow.size() != ms4.size()) bad.push_back(fmt("the stopped camera's modes: %zu", msNow.size()));
      }
      // ai: a set while stopped, clamped to the range, put back when the camera goes: it activates the device once
      const int madeSet = mr.count();
      std::string w5;
      if (!c4->set("gain", 300, false, w5)) bad.push_back("set gain stopped: " + w5);
      expect("gain set stopped", false, VideoProcAmp_Gain, 255, 2);
      if (mr.count() != madeSet + 1) bad.push_back(fmt("a set on the stopped camera made %d source(s)", mr.count() - madeSet));
      c4.reset();
      expect("gain after close", false, VideoProcAmp_Gain, 0, 2);
    }
    int sets;
    {
      std::lock_guard<std::mutex> lk(ctl->mu);
      sets = ctl->sets;
    }
    if (bad.empty())
      out.ok("rules", fmt("listed %s; at start sharpness 0, backlight 0, auto-exposure priority 0; zoom 251.6 set as 252, "
                          "exposure 10000 us as 2^-7 s (7812.5 us); all put back at stop, the stopped camera's controls "
                          "and modes read with no source made, a gain set while stopped (one source made) put back at "
                          "close; %d sets on the device", listed.c_str(), sets));
    else {
      std::string t;
      for (auto& b : bad) t += (t.empty() ? "" : "; ") + b;
      out.fail("rules", t);
    }
  }

  if (want("paint")) {
    // ai: MJPG encoded a frame at a time from a paint, the session case's path: the gradient's picture by the
    // ai: capture's time, a frame's interval a picture
    Maker mp;
    mp.cfg.paint = [](int64_t t0, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride) {
      sim::fakeMfPaint(uint32_t((t0 / 33'333'333) % kFakeMfPictures), w, h, luma, stride);
    };
    auto c3 = cam::mfOpenSource(mp.fn(), fakeInfo(), why);
    const std::vector<cam::Mode> ms3 = c3 ? c3->modes() : std::vector<cam::Mode>{};
    if (const cam::Mode* m = find(ms3, "MJPG")) stream(out, "paint", *c3, *m, secs, m->fps, true, mp);
    else out.fail("paint", "no MJPG mode: " + why);
    c3.reset();
  }

  if (want("session")) session(out, secs);

  cam.reset();
  if (SUCCEEDED(com)) CoUninitialize();
  return out.failed ? kFailed : kHeld;
}

#endif

}  // namespace lizard::check
