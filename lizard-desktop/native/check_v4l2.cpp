// ai: cam_check v4l2 (2026-10-25): the V4L2 backend (camera/v4l2.cpp) against sim/fake_v4l2's devices, the fake's
// ai: table installed before any backend call, so no device node is probed, listed or opened: the kernel's table is
// ai: only asked with LIZ_V4L2_NODES set (env), which never reads /dev, and the probe's default (/dev/video*) is
// ai: checked through probedNodes with one of the check's own. Cases: env, list, open, modes, pick, controls, set,
// ai: start-stop, refuse, stream, stepwise, formats, faults, eio, set-streaming; then the ones that need the
// ai: conversions, TurboJPEG, the simulated screen and the session: mjpeg (the luma within JPEG's loss of the
// ai: gradient; frames cut short handed over, decoded to their cut), session (the scene's test stream through RxSession
// ai: on the CPU decoder: 60 blocks at 30 painted over MJPEG 1920x1080 and YUYV 640x480, 16 blocks over YUYV 640x480),
// ai: file (a file received over MJPEG, byte for byte). The folders the session cases make are removed after them.
// ai: Arguments: case names (prefixes) to run, all where none; a name that is no case's, or that runs nothing, is
// ai: checks.h's kUsage. Every figure is the fake's, never a camera's.
#include "checks.h"

#if defined(__linux__)
#include <errno.h>
#include <linux/videodev2.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "camera/backends.h"
#include "camera/camera.h"
#include "camera/jpeg.h"
#include "camera/luma.h"
#include "camera/v4l2_sys.h"
#include "json.hpp"
#include "rx_session.h"
#include "sim/fake_v4l2.h"
#include "sim/scene.h"
#endif

namespace lizard::check {

#if !defined(__linux__)

int v4l2(const Args&) { return kNotHere; }

#else

namespace {

using nlohmann::json;
using sim::FakeDevice;
using sim::FakeV4l2;

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[2048];
  va_list a;
  va_start(a, f);
  std::vsnprintf(b, sizeof b, f, a);
  va_end(a);
  return b;
}

std::string join(const std::vector<std::string>& v, const char* sep = "; ") {
  std::string s;
  for (size_t i = 0; i < v.size(); i++) s += (i ? sep : "") + v[i];
  return s;
}

void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

double nowS() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Run {
  std::vector<std::string> only;
  int failed = 0;
  int lines = 0;   // ai: case lines reported
  bool wants(const std::string& c) const {
    if (only.empty()) return true;
    for (auto& o : only)
      if (!c.compare(0, o.size(), o)) return true;
    return false;
  }
  void line(const char* tag, const std::string& c, const std::string& m) {
    lines++;
    std::printf("%s %s: %s\n", tag, c.c_str(), m.c_str());
    std::fflush(stdout);
  }
  void ok(const std::string& c, const std::string& m) { line("ok", c, m); }
  void fail(const std::string& c, const std::string& m) {
    failed++;
    line("FAIL", c, m);
  }
  void skip(const std::string& c, const std::string& m) { line("skip", c, m); }
  // ai: ok with what held, or FAIL with what did not
  void verdict(const std::string& c, const std::vector<std::string>& bad, const std::string& m) {
    if (bad.empty()) ok(c, m);
    else fail(c, join(bad) + (m.empty() ? "" : " (" + m + ")"));
  }
};

// ai: The fake's table in use for a scope, the one before back after
struct Use {
  const cam::V4l2Sys* before;
  explicit Use(const FakeV4l2& f) : before(cam::v4l2UseSys(f.sys())) {}
  ~Use() { cam::v4l2UseSys(before); }
  Use(const Use&) = delete;
  Use& operator=(const Use&) = delete;
};

std::vector<FakeDevice> devices(std::initializer_list<const char*> profiles) {
  std::vector<FakeDevice> v;
  for (const char* p : profiles) {
    FakeDevice d;
    d.profile = p;
    v.push_back(d);
  }
  return v;
}

// ai: the four cameras and a metadata node, as /dev/video90 to 94
const char* const kStandard[] = {"c930e", "uvc60", "mono", "bayer", "meta"};
std::vector<FakeDevice> standard() { return devices({"c930e", "uvc60", "mono", "bayer", "meta"}); }

std::string idOf(const FakeV4l2& f, size_t i) { return "v4l2:" + f.byId(i); }

std::string fpsText(double f) { return std::fabs(f - std::round(f)) < 1e-9 ? fmt("%.0f", f) : fmt("%.1f", f); }
std::string modeText(const cam::Mode& m) {
  return fmt("%s %ux%u@%s", m.format.c_str(), m.width, m.height, fpsText(m.fps).c_str());
}

int findMode(const std::vector<cam::Mode>& ms, const std::string& format, uint32_t w, uint32_t h, double fps) {
  for (size_t i = 0; i < ms.size(); i++)
    if (ms[i].format == format && ms[i].width == w && ms[i].height == h && std::fabs(ms[i].fps - fps) < 0.01)
      return (int)i;
  return -1;
}

std::unique_ptr<cam::Camera> openOrNote(const std::string& id, std::vector<std::string>& bad) {
  std::string why;
  auto c = cam::v4l2Open(id, why);
  if (!c) bad.push_back("open " + id + ": " + why);
  return c;
}

int64_t ctl(const FakeV4l2& f, size_t i, uint32_t id) {
  int64_t v = INT64_MIN;
  f.control(i, id, v);
  return v;
}

const cam::Control* control(const std::vector<cam::Control>& cs, const std::string& id) {
  for (auto& c : cs)
    if (c.id == id) return &c;
  return nullptr;
}

// ai: What a stream's sink saw: each frame's time and sequence, and its luma against the fake's gradient (exact, or
// ai: within `tol` on the interior where the conversion filters, Bayer's 2 x 2 window; a JPEG's error as a sum of
// ai: squares for its PSNR)
struct Probe {
  std::mutex mu;
  std::vector<std::pair<int64_t, uint64_t>> stamps;
  uint64_t frames = 0, mismatched = 0;
  uint64_t low = 0;   // ai: frames under 30 dB from the gradient (a JPEG cut short, decoded to its cut)
  int maxErr = 0, tol = 0;
  bool interior = false, compare = true;
  double sq = 0, px = 0;
  std::string firstBad;
  std::atomic<int> ended{0};
  std::string why;
  std::thread::id endedOn;

  cam::Camera::Sink sink() {
    return [this](const cam::Frame& f) {
      std::lock_guard<std::mutex> l(mu);
      frames++;
      stamps.emplace_back(f.ns, f.seq);
      if (!compare) return;
      const uint32_t m = interior ? 1 : 0;
      int worst = 0;
      double s = 0;
      for (uint32_t y = m; y + m < f.height; y++) {
        const uint8_t* row = f.luma + (size_t)y * f.stride;
        for (uint32_t x = m; x + m < f.width; x++) {
          const int e = std::abs((int)row[x] - (int)FakeV4l2::gradient(f.seq, x, y));
          worst = std::max(worst, e);
          s += (double)e * e;
        }
      }
      const double n = (double)(f.width - 2 * m) * (f.height - 2 * m);
      sq += s;
      px += n;
      if (n > 0 && s / n > 255.0 * 255.0 / 1000.0) low++;
      maxErr = std::max(maxErr, worst);
      if (worst > tol) {
        if (!mismatched) firstBad = fmt("frame %llu off by %d", (unsigned long long)f.seq, worst);
        mismatched++;
      }
    };
  }
  cam::Camera::Ended onEnded() {
    return [this](const std::string& w) {
      std::lock_guard<std::mutex> l(mu);
      why = w;
      endedOn = std::this_thread::get_id();
      ended++;
    };
  }
  double psnr() {
    std::lock_guard<std::mutex> l(mu);
    const double mse = px > 0 ? sq / px : 0;
    return mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99;
  }
};

// ai: A stream's timing against its interval: every step forward, and spaced by the interval times the sequence's step
// ai: (exactly where the device stamps the frames, within `tolUs`; else the median within 3%)
void timing(Probe& p, double ivNs, bool exact, std::vector<std::string>& bad, std::string& note) {
  std::lock_guard<std::mutex> l(p.mu);
  std::vector<double> per;
  double worstUs = 0;
  for (size_t i = 1; i < p.stamps.size(); i++) {
    const int64_t dt = p.stamps[i].first - p.stamps[i - 1].first;
    const int64_t ds = (int64_t)(p.stamps[i].second - p.stamps[i - 1].second);
    if (dt <= 0 || ds <= 0) {
      bad.push_back(fmt("frame %zu not after the one before (%lld ns, %lld in sequence)", i, (long long)dt,
                        (long long)ds));
      return;
    }
    per.push_back((double)dt / (double)ds);
    worstUs = std::max(worstUs, std::fabs((double)dt - ds * ivNs) / 1e3);
  }
  if (per.empty()) {
    bad.push_back("fewer than 2 frames");
    return;
  }
  std::sort(per.begin(), per.end());
  const double med = per[per.size() / 2];
  if (exact) {
    if (worstUs > 50) bad.push_back(fmt("a step %.1f us off the interval", worstUs));
    note = fmt("steps the interval within %.1f us", worstUs);
  } else {
    if (std::fabs(med / ivNs - 1) > 0.03) bad.push_back(fmt("median step %.3f ms for %.3f ms", med / 1e6, ivNs / 1e6));
    note = fmt("median step %.3f ms (dequeue times)", med / 1e6);
  }
}

// ai: Whether luma.cpp converts (a packed layout needs it): a 4 x 2 YUYV frame through toLuma
bool convertsPacked(std::string& why) {
  const uint8_t yuyv[16] = {10, 128, 20, 128, 30, 128, 40, 128, 50, 128, 60, 128, 70, 128, 80, 128};
  std::vector<uint8_t> out;
  if (!cam::toLuma(cam::Pix::YUYV, 8, yuyv, sizeof yuyv, 4, 2, 8, out, &why)) return false;
  if (out.size() < 8 || out[0] != 10 || out[7] != 80) {
    why = "toLuma gave the wrong luma for YUYV";
    return false;
  }
  return true;
}

// ai: --- the cases ---

// ai: The probe's nodes and the kernel's table under LIZ_V4L2_NODES, nothing reading /dev (2026-10-25: with probing
// ai: on by default the case had skipped whole, so set and empty meaning none, which every test that lists cameras
// ai: relies on, was checked nowhere). The parse (probedNodes) with a default of the check's own, each way: set, the
// ai: nodes named and the default never asked (empty and colons alone none); unset, the default's nodes. Then the
// ai: kernel's table with the variable set: empty, no node, nothing listed or noted; two missing nodes named, both
// ai: noted and none listed. The kernel's list is asked only after its nodes() came back as named, so a parse gone
// ai: wrong fails here and never opens a real camera; unset, the kernel's table is never asked (it would list
// ai: /dev/video*).
void caseEnv(Run& r) {
  std::vector<std::string> bad;
  const std::vector<std::string> mine = {"/dev/video7", "/dev/video9"};
  int asked = 0;
  const std::function<std::vector<std::string>()> byDefault = [&] {
    asked++;
    return mine;
  };
  const std::string a = "/nonexistent/lizard-a", b = "/nonexistent/lizard-b";
  const struct { std::string env; std::vector<std::string> nodes; } sets[] = {
      {"", {}}, {":", {}}, {"::", {}}, {a + "::" + b, {a, b}}, {a + ":", {a}}, {":" + b, {b}}, {a, {a}}};
  for (const auto& w : sets) {
    const auto got = cam::probedNodes(w.env.c_str(), byDefault);
    if (got != w.nodes) bad.push_back("set to \"" + w.env + "\": " + (got.empty() ? std::string("none") : join(got, ", ")));
  }
  if (asked) bad.push_back(fmt("set, the default asked %d times", asked));
  const auto unset = cam::probedNodes(nullptr, byDefault);
  if (unset != mine || asked != 1) bad.push_back(fmt("unset: %s, the default asked %d times", unset.empty() ? "none" : join(unset, ", ").c_str(), asked));
  const char* had = std::getenv("LIZ_V4L2_NODES");
  const std::string kept = had ? had : "";
  const cam::V4l2Sys* before = cam::v4l2UseSys(nullptr);
  const cam::V4l2Sys& k = cam::v4l2Sys();
  std::string note = "none";
  setenv("LIZ_V4L2_NODES", "", 1);
  const auto none = k.nodes();
  if (!none.empty()) {
    bad.push_back("the kernel's table, LIZ_V4L2_NODES empty: nodes " + join(none, ", ") + " (not listed)");
  } else {
    std::vector<std::string> notes;
    if (!cam::v4l2List(&notes).empty() || !notes.empty()) bad.push_back("the kernel's table, LIZ_V4L2_NODES empty: a list or a note");
  }
  setenv("LIZ_V4L2_NODES", (a + "::" + b).c_str(), 1);
  const auto two = k.nodes();
  if (two != std::vector<std::string>{a, b}) {
    bad.push_back("the kernel's table, two named: nodes " + (two.empty() ? std::string("none") : join(two, ", ")) + " (not listed)");
  } else {
    std::vector<std::string> notes;
    const auto l = cam::v4l2List(&notes);
    if (!l.empty() || notes.size() != 2) bad.push_back(fmt("the kernel's table, two missing nodes: %zu listed, %zu noted", l.size(), notes.size()));
    else note = notes[0];
  }
  if (had) setenv("LIZ_V4L2_NODES", kept.c_str(), 1);
  else unsetenv("LIZ_V4L2_NODES");
  cam::v4l2UseSys(before);
  r.verdict("env", bad,
            "LIZ_V4L2_NODES set: the nodes named, set and empty none, the default never asked; unset: the default's "
            "nodes; the kernel's table with it empty listing and noting nothing, with two missing nodes noting both (\"" +
                note + "\")");
}

void caseList(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  std::vector<std::string> bad, notes;
  const auto l = cam::v4l2List(&notes);
  const char* cards[] = {"Fake C930e", "Fake UVC 60", "Fake Mono", "Fake Bayer"};
  if (l.size() != 4) bad.push_back(fmt("%zu cameras listed, 4 expected", l.size()));
  std::string got;
  for (size_t i = 0; i < l.size() && i < 4; i++) {
    if (l[i].id != idOf(f, i)) bad.push_back("camera " + std::to_string(i) + " is " + l[i].id + ", not " + idOf(f, i));
    if (l[i].name != cards[i]) bad.push_back("camera " + std::to_string(i) + " named " + l[i].name);
    if (l[i].backend != "uvc") bad.push_back(l[i].id + "'s backend " + l[i].backend);
    if (l[i].detail.find(", ") == std::string::npos) bad.push_back(l[i].id + "'s detail " + l[i].detail);
    got += (i ? "; " : "") + l[i].id + " \"" + l[i].name + "\" (" + l[i].detail + ")";
  }
  for (auto& i : l)
    if (i.id == idOf(f, 4) || i.id == "v4l2:" + f.node(4)) bad.push_back("the metadata node listed");
  for (size_t i = 0; i < f.size(); i++)
    if (f.served(i).opens != 1)
      bad.push_back(f.node(i) + fmt(" opened %llu times", (unsigned long long)f.served(i).opens));
  if (!notes.empty()) bad.push_back("notes: " + join(notes));
  r.verdict("list", bad, got + "; the metadata node " + f.node(4) + " opened and skipped");
}

void caseOpen(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  std::vector<std::string> bad;
  std::string why;
  auto a = cam::v4l2Open(idOf(f, 0), why);
  if (!a) bad.push_back("by its by-id link: " + why);
  auto b = cam::v4l2Open("v4l2:" + f.node(1), why);
  if (!b) bad.push_back("by its node: " + why);
  else if (b->info().id != idOf(f, 1)) bad.push_back("opened by node, its id " + b->info().id);
  if (a && a->info().id != idOf(f, 0)) bad.push_back("opened by link, its id " + a->info().id);
  std::string metaWhy, missWhy, otherWhy;
  if (cam::v4l2Open(idOf(f, 4), metaWhy)) bad.push_back("the metadata node opened");
  if (cam::v4l2Open("v4l2:/dev/video99", missWhy)) bad.push_back("a node not probed opened");
  if (cam::v4l2Open("nosuch:x", otherWhy)) bad.push_back("another backend's id opened");
  r.verdict("open", bad,
            "by its link and by its node, two cameras at once, each named by its link (" +
                (b ? b->info().id : std::string("?")) + " opened as v4l2:" + f.node(1) +
                "); refused: the metadata node (\"" + metaWhy + "\"), an unprobed node (\"" + missWhy +
                "\"), \"aravis:x\" (\"" + otherWhy + "\")");
}

void caseModes(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  struct Want { const char* format; uint32_t w, h; double fps; cam::Pix pix; uint8_t bits; bool compressed; };
  struct Expect { size_t dev; size_t count; std::vector<Want> named; };
  const Expect ex[] = {
      {0, 32, {{"MJPG", 1920, 1080, 30, cam::Pix::MJPEG, 8, true}, {"MJPG", 1280, 720, 7.5, cam::Pix::MJPEG, 8, true},
               {"YUYV", 1920, 1080, 5, cam::Pix::YUYV, 8, false}, {"YUYV", 640, 480, 30, cam::Pix::YUYV, 8, false}}},
      {1, 6, {{"MJPG", 1280, 720, 60, cam::Pix::MJPEG, 8, true}, {"YUYV", 640, 480, 60, cam::Pix::YUYV, 8, false},
              {"NV12", 1280, 720, 30, cam::Pix::NV12, 8, false}}},
      {2, 3, {{"GREY", 1280, 1024, 60, cam::Pix::Grey8, 8, false},
              {"Y16", 1280, 1024, 30, cam::Pix::Grey16, 16, false}}},
      {3, 1, {{"RGGB", 1920, 1200, 60, cam::Pix::Bayer8, 8, false}}},
  };
  for (const Expect& e : ex) {
    std::vector<std::string> bad;
    auto c = openOrNote(idOf(f, e.dev), bad);
    std::string got;
    if (c) {
      const auto ms = c->modes();
      if (ms.size() != e.count) bad.push_back(fmt("%zu modes, %zu expected", ms.size(), e.count));
      for (const Want& w : e.named) {
        const int i = findMode(ms, w.format, w.w, w.h, w.fps);
        if (i < 0) {
          bad.push_back(fmt("no %s %ux%u@%s", w.format, w.w, w.h, fpsText(w.fps).c_str()));
          continue;
        }
        const cam::Mode& m = ms[i];
        if (m.pix != w.pix || m.bits != w.bits || m.compressed != w.compressed || m.fpsMin != m.fps)
          bad.push_back(modeText(m) + " mapped wrong");
        got += (got.empty() ? "" : ", ") + modeText(m);
      }
      std::set<std::string> formats;
      for (auto& m : ms) formats.insert(m.format);
      got = fmt("%zu modes (", ms.size()) + join(std::vector<std::string>(formats.begin(), formats.end()), ", ") +
            "), " + got;
    }
    r.verdict(std::string("modes ") + kStandard[e.dev], bad, got);
  }
}

// ai: camera.h pick's rule, read independently: the modes at the size wanted (else the nearest size by pixels), the
// ai: fastest of them up to the rate wanted, then the largest, then uncompressed before MJPEG
int pickRule(const std::vector<cam::Mode>& ms, uint32_t W, uint32_t H, double F, std::string& why) {
  if (ms.empty()) return -1;
  std::vector<int> at;
  for (size_t i = 0; i < ms.size(); i++)
    if (ms[i].width == W && ms[i].height == H) at.push_back((int)i);
  if (at.empty()) {
    const double want = (double)W * H;
    double best = 1e300;
    for (auto& m : ms) best = std::min(best, std::fabs((double)m.width * m.height - want));
    for (size_t i = 0; i < ms.size(); i++)
      if (std::fabs((double)ms[i].width * ms[i].height - want) == best) at.push_back((int)i);
    why = fmt("no %ux%u, the nearest by pixels %ux%u; ", W, H, ms[at[0]].width, ms[at[0]].height);
  } else {
    why = fmt("%ux%u offered; ", W, H);
  }
  int b = -1;
  for (int i : at) {
    const cam::Mode& m = ms[i];
    if (F > 0 && m.fps > F + 1e-9) continue;
    if (b < 0) { b = i; continue; }
    const cam::Mode& o = ms[b];
    const double pm = (double)m.width * m.height, po = (double)o.width * o.height;
    const bool same = std::fabs(m.fps - o.fps) < 1e-9;
    if (m.fps > o.fps + 1e-9 || (same && (pm > po || (pm == po && !m.compressed && o.compressed)))) b = i;
  }
  std::string others;
  for (int i : at)
    if (i != b) others += (others.empty() ? "" : ", ") + modeText(ms[i]);
  why += "the fastest there up to " + fpsText(F) + " is " + (b >= 0 ? modeText(ms[b]) : std::string("none")) +
         (others.empty() ? " (the one mode there)" : " (beside " + others + ")");
  return b;
}

void casePick(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  const char* expect[] = {"MJPG 1920x1080@30", "MJPG 1920x1080@30", "GREY 1280x1024@60", "RGGB 1920x1200@60"};
  for (size_t d = 0; d < 4; d++) {
    std::vector<std::string> bad;
    auto c = openOrNote(idOf(f, d), bad);
    std::string why;
    if (c) {
      const auto ms = c->modes();
      const int got = cam::pick(ms, 1920, 1080, 60);
      const int rule = pickRule(ms, 1920, 1080, 60, why);
      if (got < 0 || got >= (int)ms.size()) bad.push_back(fmt("pick gave %d", got));
      else if (got != rule) bad.push_back("pick gave " + modeText(ms[got]) + ", the rule " + modeText(ms[rule]));
      else if (modeText(ms[got]) != expect[d]) bad.push_back("pick gave " + modeText(ms[got]) + " for " + expect[d]);
      if (got >= 0 && got < (int)ms.size()) why = modeText(ms[got]) + ": " + why;
    }
    r.verdict(std::string("pick ") + kStandard[d], bad, why);
  }
}

void caseControls(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  std::vector<std::string> bad;
  std::string got;
  auto list = [&](size_t d, std::vector<std::string> ids) -> std::vector<cam::Control> {
    auto c = openOrNote(idOf(f, d), bad);
    if (!c) return {};
    auto cs = c->controls();
    std::vector<std::string> have;
    for (auto& k : cs) have.push_back(k.id);
    std::sort(have.begin(), have.end());
    std::sort(ids.begin(), ids.end());
    if (have != ids) bad.push_back(f.node(d) + " lists " + join(have, ",") + " for " + join(ids, ","));
    return cs;
  };
  const auto a = list(0, {"zoom", "focus", "exposure", "gain", "sharpness", "backlight", "brightness", "contrast"});
  if (const cam::Control* e = control(a, "exposure")) {
    if (e->unit != "us" || e->min != 300 || e->max != 204700 || e->step != 100 || e->def != 25000 ||
        e->value != 25000 || !e->canAuto || !e->isAuto)
      bad.push_back(fmt("c930e exposure %s %g..%g step %g def %g value %g auto %d/%d", e->unit.c_str(), e->min, e->max,
                        e->step, e->def, e->value, e->canAuto, e->isAuto));
    got += fmt("c930e: 8 (exposure %g..%g us step %g, at %g, automatic: aperture priority", e->min, e->max, e->step,
               e->value);
  }
  if (const cam::Control* z = control(a, "zoom"))
    if (z->min != 100 || z->max != 500 || z->value != 100 || z->canAuto) bad.push_back("c930e zoom mapped wrong");
  if (const cam::Control* fo = control(a, "focus")) {
    if (fo->step != 5 || !fo->canAuto || !fo->isAuto) bad.push_back("c930e focus mapped wrong");
    got += "; focus automatic; power line frequency not taken)";
  }
  const auto b = list(1, {"exposure", "gain", "sharpness", "backlight", "brightness", "contrast"});
  if (const cam::Control* e = control(b, "exposure")) {
    if (!e->canAuto || !e->isAuto || e->min != 100 || e->max != 500000) bad.push_back("uvc60 exposure mapped wrong");
    got += "; uvc60: 6 (exposure automatic by its auto mode)";
  }
  const auto c = list(2, {"exposure", "gain"});
  if (const cam::Control* e = control(c, "exposure"))
    if (!e->canAuto || !e->isAuto) bad.push_back("mono exposure mapped wrong");
  const auto d = list(3, {"exposure", "gain", "brightness"});
  if (const cam::Control* e = control(d, "exposure")) {
    if (e->canAuto || e->value != 16600) bad.push_back(fmt("bayer exposure canAuto %d value %g", e->canAuto, e->value));
    got += "; mono: 2; bayer (no extended controls): 3 by QUERYCTRL, exposure manual alone";
  }
  r.verdict("controls", bad, got);
}

void caseSet(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  std::vector<std::string> bad;
  auto expect = [&](size_t d, uint32_t id, int64_t want, const char* what) {
    const int64_t v = ctl(f, d, id);
    if (v != want) bad.push_back(fmt("%s: %lld, %lld expected", what, (long long)v, (long long)want));
  };
  auto set = [&](cam::Camera& c, const char* id, double v, bool automatic, bool ok) {
    std::string why;
    if (c.set(id, v, automatic, why) != ok)
      bad.push_back(fmt("set %s %g%s %s%s", id, v, automatic ? " auto" : "", ok ? "refused: " : "taken",
                        ok ? why.c_str() : ""));
    return why;
  };
  std::string refusals;
  {
    auto a = openOrNote(idOf(f, 0), bad);
    auto b = openOrNote(idOf(f, 1), bad);
    auto c = openOrNote(idOf(f, 3), bad);
    if (a && b && c) {
      set(*a, "exposure", 10000, false, true);
      expect(0, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL, "c930e exposure_auto after a manual exposure");
      expect(0, V4L2_CID_EXPOSURE_ABSOLUTE, 100, "c930e exposure_absolute for 10000 us");
      const cam::Control* e = nullptr;
      const auto cs = a->controls();
      if ((e = control(cs, "exposure")) && (e->value != 10000 || e->isAuto))
        bad.push_back("c930e exposure not read back");
      set(*a, "exposure", 12345, false, true);
      expect(0, V4L2_CID_EXPOSURE_ABSOLUTE, 123, "c930e exposure_absolute for 12345 us");
      set(*a, "focus", 52, false, true);
      expect(0, V4L2_CID_FOCUS_AUTO, 0, "c930e focus_auto after a manual focus");
      expect(0, V4L2_CID_FOCUS_ABSOLUTE, 50, "c930e focus_absolute for 52 (step 5)");
      set(*a, "zoom", 250, false, true);
      expect(0, V4L2_CID_ZOOM_ABSOLUTE, 250, "c930e zoom");
      set(*a, "zoom", 9999, false, true);
      expect(0, V4L2_CID_ZOOM_ABSOLUTE, 500, "c930e zoom past its top");
      set(*a, "exposure", 0, true, true);
      expect(0, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_APERTURE_PRIORITY, "c930e exposure automatic");
      set(*a, "focus", 0, true, true);
      expect(0, V4L2_CID_FOCUS_AUTO, 1, "c930e focus automatic");
      refusals = set(*a, "gain", 0, true, false) + "; " + set(*a, "nonsense", 1, false, false);
      set(*b, "exposure", 0, true, true);
      expect(1, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_AUTO, "uvc60 exposure automatic (no aperture priority)");
      set(*b, "exposure", 500, false, true);
      expect(1, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL, "uvc60 exposure_auto after a manual exposure");
      expect(1, V4L2_CID_EXPOSURE_ABSOLUTE, 5, "uvc60 exposure_absolute for 500 us");
      set(*c, "exposure", 5000, false, true);
      expect(3, V4L2_CID_EXPOSURE_ABSOLUTE, 50, "bayer exposure_absolute for 5000 us (S_CTRL)");
      set(*c, "exposure", 0, true, false);
      a->stop();
      c->stop();
      expect(0, V4L2_CID_EXPOSURE_AUTO, 3, "c930e exposure_auto put back");
      expect(0, V4L2_CID_EXPOSURE_ABSOLUTE, 250, "c930e exposure_absolute put back");
      expect(0, V4L2_CID_FOCUS_AUTO, 1, "c930e focus_auto put back");
      expect(0, V4L2_CID_FOCUS_ABSOLUTE, 0, "c930e focus_absolute put back");
      expect(0, V4L2_CID_ZOOM_ABSOLUTE, 100, "c930e zoom put back");
      expect(3, V4L2_CID_EXPOSURE_ABSOLUTE, 166, "bayer exposure put back");
    }
  }
  // ai: b (uvc60) closed without a stop: its destructor puts them back too
  expect(1, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_AUTO, "uvc60 exposure_auto put back at close");
  expect(1, V4L2_CID_EXPOSURE_ABSOLUTE, 156, "uvc60 exposure_absolute put back at close");
  r.verdict("set", bad,
            "exposure in us to exposure_absolute's 100 us with its master manual in the same call, snapped (12345 us "
            "to 123), focus to its step, zoom clamped, automatic: aperture priority on the c930e and auto on the "
            "uvc60, S_CTRL on the old driver; refused (" + refusals + "); every control back at stop and at close");
}

void caseStartStop(Run& r) {
  FakeV4l2 f(devices({"c930e"}));
  Use u(f);
  std::vector<std::string> bad;
  f.setControl(0, V4L2_CID_BACKLIGHT_COMPENSATION, 1);
  auto c = openOrNote(idOf(f, 0), bad);
  std::string note;
  if (c) {
    const auto ms = c->modes();
    const int m = findMode(ms, "YUYV", 640, 480, 30);
    Probe p;
    p.compare = false;
    std::string why;
    if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
      bad.push_back("start: " + why);
    } else {
      sleepMs(600);
      const auto s = f.served(0);
      if (ctl(f, 0, V4L2_CID_SHARPNESS) != 0) bad.push_back("sharpness not at its minimum while streaming");
      if (ctl(f, 0, V4L2_CID_BACKLIGHT_COMPENSATION) != 0) bad.push_back("backlight not at its minimum");
      if (ctl(f, 0, V4L2_CID_EXPOSURE_AUTO_PRIORITY) != 0) bad.push_back("exposure_auto_priority not 0");
      if (!s.streaming || s.fourcc != V4L2_PIX_FMT_YUYV || s.width != 640 || s.height != 480 || s.num != 1 ||
          s.den != 30)
        bad.push_back("the device is not streaming YUYV 640x480 at 1/30");
      if (s.buffers != 4 || s.mapped != 4) bad.push_back(fmt("%u buffers, %u mapped", s.buffers, s.mapped));
      const auto st = c->stats();
      if (std::fabs(st.fps - 30) > 1e-9) bad.push_back(fmt("stats fps %g", st.fps));
      if (s.dequeued < 10) bad.push_back(fmt("%llu frames dequeued in 0.6 s", (unsigned long long)s.dequeued));
      c->stop();
      const uint64_t after = p.frames;
      sleepMs(150);
      if (p.frames != after) bad.push_back("frames after stop");
      const auto t = f.served(0);
      if (t.streaming || t.buffers || t.mapped) bad.push_back(fmt("after stop: streaming %d, %u buffers, %u mapped",
                                                               t.streaming, t.buffers, t.mapped));
      if (ctl(f, 0, V4L2_CID_SHARPNESS) != 128 || ctl(f, 0, V4L2_CID_BACKLIGHT_COMPENSATION) != 1 ||
          ctl(f, 0, V4L2_CID_EXPOSURE_AUTO_PRIORITY) != 1)
        bad.push_back("the rules' controls not put back");
      note = fmt("YUYV 640x480@30 asked and answered (stats fps %g), 4 buffers mapped; streaming: sharpness 0 (was "
                 "128), backlight 0 (was 1), exposure_auto_priority 0 (was 1), %llu frames dequeued in 0.6 s; after "
                 "stop: all three back, no buffers, nothing mapped, no sink call",
                 st.fps, (unsigned long long)s.dequeued);
    }
  }
  r.verdict("start-stop", bad, note);
}

void caseRefuse(Run& r) {
  FakeV4l2 f(devices({"c930e"}));
  Use u(f);
  std::vector<std::string> bad;
  auto c = openOrNote(idOf(f, 0), bad);
  std::string a, b, k;
  if (c) {
    Probe p;
    cam::Mode m;
    m.width = 800, m.height = 600, m.fps = 30, m.format = "YUYV", m.pix = cam::Pix::YUYV, m.key = V4L2_PIX_FMT_YUYV;
    if (c->start(m, p.sink(), p.onEnded(), a)) bad.push_back("YUYV 800x600 started");
    if (a.find("YUYV 640x480") == std::string::npos || a.find("YUYV 800x600") == std::string::npos)
      bad.push_back("the refusal does not name both: " + a);
    m.width = 640, m.height = 480, m.format = "GREY", m.pix = cam::Pix::Grey8, m.key = V4L2_PIX_FMT_GREY;
    if (c->start(m, p.sink(), p.onEnded(), b)) bad.push_back("GREY started");
    if (b.find("GREY 640x480") == std::string::npos || b.find("YUYV 640x480") == std::string::npos)
      bad.push_back("the refusal does not name both: " + b);
    m.key = V4L2_PIX_FMT_H264, m.format = "H264";
    if (c->start(m, p.sink(), p.onEnded(), k)) bad.push_back("H264 started");
    const auto s = f.served(0);
    if (s.streaming || s.buffers) bad.push_back("buffers left after a refusal");
    if (ctl(f, 0, V4L2_CID_SHARPNESS) != 128 || ctl(f, 0, V4L2_CID_EXPOSURE_AUTO_PRIORITY) != 1)
      bad.push_back("a refusal changed the controls");
  }
  r.verdict("refuse", bad, "\"" + a + "\"; \"" + b + "\"; \"" + k + "\"; nothing left changed");
}

struct StreamCase {
  std::string name;
  size_t dev;
  std::string format;
  uint32_t w, h;
  double fps;
  bool packed;         // ai: converted by luma.cpp's toLuma (not the Y plane in place)
  int tol;             // ai: the luma's tolerance (Bayer's window), on the interior
  bool exact = true;   // ai: the device stamps its frames (the old driver does not)
  double secs = 2.0;
  double modeFps = -1; // ai: the mode's rate as listed, where it is not the device's (a device that states none: 0)
};

// ai: Streams the cases at once (each on its own device), then judges each: the frames against fps x secs, the
// ai: timing, the luma against the gradient, and the backend's counts against the fake's
void runStreams(Run& r, FakeV4l2& f, const std::vector<StreamCase>& cases) {
  std::string convWhy;
  const bool converts = convertsPacked(convWhy);
  struct Live { const StreamCase* c; std::unique_ptr<cam::Camera> cam; std::unique_ptr<Probe> p; std::string why; };
  std::vector<Live> live;
  for (const StreamCase& c : cases) {
    if (!r.wants(c.name)) continue;
    if (c.packed && !converts) {
      r.fail(c.name, "luma.cpp does not convert: " + convWhy);
      continue;
    }
    Live l{&c, nullptr, std::make_unique<Probe>(), ""};
    l.p->tol = c.tol;
    l.p->interior = c.tol > 0;
    std::vector<std::string> bad;
    l.cam = openOrNote(idOf(f, c.dev), bad);
    if (l.cam) {
      const auto ms = l.cam->modes();
      const int m = findMode(ms, c.format, c.w, c.h, c.modeFps >= 0 ? c.modeFps : c.fps);
      if (m < 0) l.why = "no such mode";
      else if (!l.cam->start(ms[m], l.p->sink(), l.p->onEnded(), l.why)) l.why = "start: " + l.why;
    } else {
      l.why = join(bad);
    }
    live.push_back(std::move(l));
  }
  if (live.empty()) return;
  double secs = 0;
  for (auto& l : live) secs = std::max(secs, l.c->secs);
  sleepMs((int)(secs * 1000));
  for (auto& l : live)
    if (l.cam) l.cam->stop();
  for (auto& l : live) {
    const StreamCase& c = *l.c;
    std::vector<std::string> bad;
    if (!l.why.empty()) {
      r.fail(c.name, l.why);
      continue;
    }
    const auto st = l.cam->stats();
    const auto sv = f.served(c.dev);
    Probe& p = *l.p;
    const double want = c.fps * c.secs;
    if (p.frames + 2 < 0.85 * want || p.frames > want + 2)
      bad.push_back(fmt("%llu frames in %.1f s at %s", (unsigned long long)p.frames, c.secs, fpsText(c.fps).c_str()));
    std::string tnote;
    timing(p, 1e9 / c.fps, c.exact, bad, tnote);
    if (p.mismatched) bad.push_back(fmt("%llu frames' luma off the gradient (%s)", (unsigned long long)p.mismatched,
                                        p.firstBad.c_str()));
    if (st.frames != p.frames) bad.push_back(fmt("stats frames %llu, the sink %llu", (unsigned long long)st.frames,
                                                 (unsigned long long)p.frames));
    if (st.corrupt) bad.push_back(fmt("%llu corrupt", (unsigned long long)st.corrupt));
    const uint64_t gaps = sv.firstSeq >= 0 ? (uint64_t)(sv.lastSeq - sv.firstSeq + 1) - sv.streamDequeued : 0;
    if (st.dropped != gaps) bad.push_back(fmt("dropped %llu, the fake's gaps %llu", (unsigned long long)st.dropped,
                                              (unsigned long long)gaps));
    if (st.dropped * 50 > p.frames) bad.push_back(fmt("%llu dropped", (unsigned long long)st.dropped));
    if (p.ended) bad.push_back("ended: " + p.why);
    r.verdict(c.name, bad,
              fmt("%llu frames in %.1f s (%s asked, %g agreed), %s; luma %s the gradient (worst %d) over every frame; "
                  "dropped %llu, corrupt 0, %.3f ms to luma",
                  (unsigned long long)p.frames, c.secs, fpsText(c.modeFps >= 0 ? c.modeFps : c.fps).c_str(), st.fps,
                  tnote.c_str(),
                  c.tol ? fmt("within %d of", c.tol).c_str() : "equal to", p.maxErr, (unsigned long long)st.dropped,
                  st.convertMs));
  }
}

void caseStream(Run& r) {
  FakeV4l2 f(standard());
  Use u(f);
  // ai: two rounds, a device a stream in each
  runStreams(r, f, {{"stream YUYV", 0, "YUYV", 640, 480, 30, true, 0},
                    {"stream NV12", 1, "NV12", 1280, 720, 30, false, 0},
                    {"stream GREY", 2, "GREY", 1280, 1024, 60, false, 0},
                    {"stream RGGB", 3, "RGGB", 1920, 1200, 60, true, 2, false}});
  runStreams(r, f, {{"stream YUYV60", 1, "YUYV", 640, 480, 60, true, 0},
                    {"stream Y16", 2, "Y16", 1280, 1024, 30, true, 0}});
}

void caseStepwise(Run& r) {
  FakeV4l2 f(devices({"stepwise"}));
  Use u(f);
  std::vector<std::string> bad;
  auto c = openOrNote(idOf(f, 0), bad);
  std::string got;
  if (c) {
    const auto ms = c->modes();
    const char* sizes[] = {"GREY 2048x1536@120", "GREY 1920x1080@120", "GREY 1280x720@120", "GREY 640x480@120"};
    if (ms.size() != 4) bad.push_back(fmt("%zu modes", ms.size()));
    for (size_t i = 0; i < ms.size() && i < 4; i++) {
      if (modeText(ms[i]) != sizes[i] || ms[i].fpsMin != 1) bad.push_back("mode " + modeText(ms[i]));
      got += (i ? ", " : "") + modeText(ms[i]);
    }
    if (!c->controls().empty()) bad.push_back("controls listed");
    got = fmt("%zu modes from a 160..2048/16 by 120..1536/8 range at 1/120..1/1: ", ms.size()) + got +
          " (each slowest 1); no controls";
  }
  r.verdict("stepwise modes", bad, got);
  c.reset();
  runStreams(r, f, {{"stepwise stream", 0, "GREY", 640, 480, 120, false, 0, true, 1.0}});
}

void caseFormats(Run& r) {
  FakeV4l2 f(devices({"formats"}));
  Use u(f);
  std::vector<std::string> bad;
  struct Want { const char* format; cam::Pix pix; uint8_t bits; bool packed; };
  const Want ws[] = {{"UYVY", cam::Pix::UYVY, 8, true},  {"YVYU", cam::Pix::YVYU, 8, true},
                     {"VYUY", cam::Pix::VYUY, 8, true},  {"NV21", cam::Pix::NV21, 8, false},
                     {"YU12", cam::Pix::I420, 8, false}, {"YV12", cam::Pix::YV12, 8, false},
                     {"Y10", cam::Pix::Grey16, 10, true}, {"Y12", cam::Pix::Grey16, 12, true},
                     {"RGB3", cam::Pix::RGB24, 8, true}, {"BGR3", cam::Pix::BGR24, 8, true}};
  {
    auto c = openOrNote(idOf(f, 0), bad);
    if (c) {
      const auto ms = c->modes();
      if (ms.size() != 10) bad.push_back(fmt("%zu modes (H264 not skipped?)", ms.size()));
      for (const Want& w : ws) {
        const int i = findMode(ms, w.format, 320, 240, 30);
        if (i < 0) bad.push_back(std::string("no ") + w.format);
        else if (ms[i].pix != w.pix || ms[i].bits != w.bits) bad.push_back(std::string(w.format) + " mapped wrong");
      }
    }
  }
  r.verdict("formats modes", bad,
            "10 formats mapped to their Pix (Y10 and Y12 as Grey16 of 10 and 12 bits), H264 skipped");
  for (const Want& w : ws)
    runStreams(r, f, {{std::string("formats ") + w.format, 0, w.format, 320, 240, 30, w.packed, 0, true, 0.6}});
}

// ai: A capture card that lists no size, no interval and no frame rate: its one size from TRY_FMT, its rate unstated
// ai: (a mode at 0, agreed 0), its frames at the card's own 60
void caseCard(Run& r) {
  FakeV4l2 f(devices({"card"}));
  Use u(f);
  std::vector<std::string> bad;
  std::string got;
  {
    auto c = openOrNote(idOf(f, 0), bad);
    if (c) {
      const auto ms = c->modes();
      if (ms.size() != 1 || modeText(ms[0]) != "YUYV 1280x720@0") bad.push_back(fmt("%zu modes", ms.size()));
      for (auto& m : ms) got += (got.empty() ? "" : ", ") + modeText(m);
    }
  }
  r.verdict("card modes", bad, "TRY_FMT for 1920x1080, 1280x720 and 640x480 answered " + got + " (no rate stated)");
  runStreams(r, f, {{"card stream", 0, "YUYV", 1280, 720, 60, true, 0, false, 1.0, 0}});
}

void caseFaults(Run& r) {
  std::vector<FakeDevice> ds = devices({"mono"});
  ds[0].faults.errors = 0.1;
  ds[0].faults.skipped = 0.1;
  ds[0].faults.unplugAfter = 150;
  ds[0].faults.seed = 7;
  FakeV4l2 f(ds);
  Use u(f);
  std::vector<std::string> bad;
  std::string note;
  auto c = openOrNote(idOf(f, 0), bad);
  if (c) {
    const auto ms = c->modes();
    Probe p;
    p.compare = false;
    std::string why;
    const int m = findMode(ms, "GREY", 1280, 1024, 60);
    if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
      bad.push_back("start: " + why);
    } else {
      const double t0 = nowS();
      while (!p.ended && nowS() - t0 < 8) sleepMs(20);
      sleepMs(300);
      const auto st = c->stats();
      const auto sv = f.served(0);
      if (p.ended != 1) bad.push_back(fmt("ended %d times", p.ended.load()));
      if (p.why != "the camera was unplugged") bad.push_back("ended with \"" + p.why + "\"");
      if (p.endedOn == std::this_thread::get_id()) bad.push_back("ended on the caller's thread");
      const uint64_t gaps = (uint64_t)(sv.lastSeq - sv.firstSeq + 1) - sv.streamDequeued;
      if (st.corrupt != sv.dequeuedErrors)
        bad.push_back(fmt("corrupt %llu, flagged %llu", (unsigned long long)st.corrupt,
                          (unsigned long long)sv.dequeuedErrors));
      if (st.frames != sv.dequeued - sv.dequeuedErrors) bad.push_back("frames not the clean ones dequeued");
      if (st.dropped != gaps) bad.push_back(fmt("dropped %llu, gaps %llu", (unsigned long long)st.dropped,
                                                (unsigned long long)gaps));
      if (!st.corrupt || !st.dropped) bad.push_back("no fault seen");
      const double s0 = nowS();
      c->stop();
      const double stopMs = (nowS() - s0) * 1e3;
      if (stopMs > 1000) bad.push_back(fmt("stop took %.0f ms", stopMs));
      if (f.served(0).mapped) bad.push_back("buffers left mapped after the unplug's stop");
      std::string again;
      if (c->start(ms[m], p.sink(), p.onEnded(), again)) bad.push_back("started on an unplugged device");
      std::vector<std::string> notes;
      if (!cam::v4l2List(&notes).empty()) bad.push_back("the unplugged camera still listed");
      note = fmt("%llu frames delivered: %llu flagged (corrupt %llu), %llu skipped and %llu starved (dropped %llu), "
                 "%llu clean (frames %llu); unplugged: ended once (\"%s\") on the capture thread; stop %.1f ms; a "
                 "restart refused (\"%s\"); gone from the list",
                 (unsigned long long)sv.frames, (unsigned long long)sv.errors, (unsigned long long)st.corrupt,
                 (unsigned long long)sv.skipped, (unsigned long long)sv.starved, (unsigned long long)st.dropped,
                 (unsigned long long)(sv.dequeued - sv.dequeuedErrors), (unsigned long long)st.frames, p.why.c_str(),
                 stopMs, again.c_str());
    }
  }
  r.verdict("faults", bad, note);
}

void caseEio(Run& r) {
  std::vector<FakeDevice> ds = devices({"uvc60"});
  ds[0].faults.eioAfter = 40;
  FakeV4l2 f(ds);
  Use u(f);
  std::vector<std::string> bad;
  std::string note;
  auto c = openOrNote(idOf(f, 0), bad);
  if (c) {
    const auto ms = c->modes();
    const int m = findMode(ms, "NV12", 1280, 720, 30);
    Probe p;
    std::string why;
    if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
      bad.push_back("start: " + why);
    } else {
      const double t0 = nowS();
      while (!p.ended && nowS() - t0 < 5) sleepMs(20);
      if (p.ended != 1 || p.why != std::strerror(EIO)) bad.push_back(fmt("ended %d times, \"%s\"", p.ended.load(),
                                                                         p.why.c_str()));
      if (ctl(f, 0, V4L2_CID_SHARPNESS) != 0) bad.push_back("sharpness not at its minimum while streaming");
      c->stop();
      if (ctl(f, 0, V4L2_CID_SHARPNESS) != 3 || ctl(f, 0, V4L2_CID_BACKLIGHT_COMPENSATION) != 1)
        bad.push_back("controls not put back after the stream's error");
      const auto sv = f.served(0);
      if (sv.streaming || sv.buffers || sv.mapped) bad.push_back("the stream not let go");
      // ai: the device is still there: a new stream starts and runs until its own error
      Probe q;
      if (!c->start(ms[m], q.sink(), q.onEnded(), why)) bad.push_back("restart: " + why);
      sleepMs(500);
      if (q.frames < 10) bad.push_back(fmt("%llu frames after a restart", (unsigned long long)q.frames));
      c->stop();
      note = fmt("%llu frames, then EIO: ended once (\"%s\"); stop put sharpness 3 and backlight 1 back; a restart "
                 "streamed %llu frames in 0.5 s",
                 (unsigned long long)p.frames, p.why.c_str(), (unsigned long long)q.frames);
    }
  }
  r.verdict("eio", bad, note);
}

void caseSetStreaming(Run& r) {
  FakeV4l2 f(devices({"c930e"}));
  Use u(f);
  std::vector<std::string> bad;
  std::string note;
  auto c = openOrNote(idOf(f, 0), bad);
  if (c) {
    const auto ms = c->modes();
    const int m = findMode(ms, "YUYV", 640, 480, 30);
    Probe p;
    p.compare = false;
    std::string why;
    if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
      bad.push_back("start: " + why);
    } else {
      sleepMs(400);
      const uint64_t n0 = f.served(0).dequeued;
      if (!c->set("zoom", 300, false, why)) bad.push_back("zoom: " + why);
      if (!c->set("exposure", 20000, false, why)) bad.push_back("exposure: " + why);
      if (ctl(f, 0, V4L2_CID_ZOOM_ABSOLUTE) != 300) bad.push_back("zoom not taken while streaming");
      if (ctl(f, 0, V4L2_CID_EXPOSURE_ABSOLUTE) != 200 || ctl(f, 0, V4L2_CID_EXPOSURE_AUTO) != 1)
        bad.push_back("exposure not taken while streaming");
      const cam::Control* z = nullptr;
      const auto cs = c->controls();
      if (!(z = control(cs, "zoom")) || z->value != 300) bad.push_back("zoom not read back");
      sleepMs(500);
      const uint64_t n1 = f.served(0).dequeued;
      if (n1 - n0 < 10) bad.push_back(fmt("%llu frames after the set", (unsigned long long)(n1 - n0)));
      c->stop();
      if (ctl(f, 0, V4L2_CID_ZOOM_ABSOLUTE) != 100 || ctl(f, 0, V4L2_CID_EXPOSURE_AUTO) != 3 ||
          ctl(f, 0, V4L2_CID_EXPOSURE_ABSOLUTE) != 250)
        bad.push_back("not put back at stop");
      note = fmt("zoom 300 and exposure 20000 us taken while streaming from another thread, %llu frames in the 0.5 s "
                 "after; zoom 100, exposure automatic at 250 again after stop",
                 (unsigned long long)(n1 - n0));
    }
  }
  r.verdict("set-streaming", bad, note);
}

// ai: --- the cases on the other parts (the conversions, TurboJPEG, the scene, the session) ---

void caseMjpeg(Run& r) {
  std::string why;
  if (!cam::Jpeg::load(why)) {
    r.fail("mjpeg", "TurboJPEG: " + why);
    return;
  }
  {
    FakeV4l2 f(devices({"c930e"}));
    Use u(f);
    for (auto sz : {std::pair<uint32_t, uint32_t>{1280, 720}, {1920, 1080}}) {
      const std::string name = fmt("mjpeg %ux%u", sz.first, sz.second);
      if (!r.wants(name)) continue;
      std::vector<std::string> bad;
      auto c = openOrNote(idOf(f, 0), bad);
      std::string note;
      if (c) {
        const auto ms = c->modes();
        const int m = findMode(ms, "MJPG", sz.first, sz.second, 30);
        Probe p;
        p.tol = 255;
        if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
          bad.push_back("start: " + why);
        } else {
          sleepMs(2000);
          c->stop();
          const auto st = c->stats();
          const double db = p.psnr();
          if (p.frames < 50 || p.frames > 62)
            bad.push_back(fmt("%llu frames in 2 s at 30", (unsigned long long)p.frames));
          if (db < 35) bad.push_back(fmt("PSNR %.1f dB against the gradient", db));
          if (p.low) bad.push_back(fmt("%llu frames under 30 dB", (unsigned long long)p.low));
          if (st.corrupt) bad.push_back(fmt("%llu corrupt", (unsigned long long)st.corrupt));
          note = fmt("%llu frames in 2 s, the luma %.1f dB from the gradient (worst pixel %d; JPEG q90 4:2:2, its "
                     "tables cut), %.2f ms to luma",
                     (unsigned long long)p.frames, db, p.maxErr, st.convertMs);
        }
      }
      r.verdict(name, bad, note);
    }
  }
  if (!r.wants("mjpeg truncated")) return;
  std::vector<FakeDevice> ds = devices({"c930e"});
  ds[0].faults.truncated = 0.25;
  ds[0].faults.seed = 3;
  FakeV4l2 f(ds);
  Use u(f);
  std::vector<std::string> bad;
  std::string note;
  auto c = openOrNote(idOf(f, 0), bad);
  if (c) {
    const auto ms = c->modes();
    const int m = findMode(ms, "MJPG", 640, 480, 30);
    Probe p;
    p.tol = 255;
    if (m < 0 || !c->start(ms[m], p.sink(), p.onEnded(), why)) {
      bad.push_back("start: " + why);
    } else {
      sleepMs(2000);
      c->stop();
      const auto st = c->stats();
      const auto sv = f.served(0);
      // ai: jpeg.h takes a frame cut in its scan (decoded to the cut, the rest filled): handed over, not corrupt, and
      // ai: far from the gradient; the whole ones within JPEG's loss
      if (!sv.dequeuedTruncated) bad.push_back("no frame cut");
      if (st.corrupt) bad.push_back(fmt("%llu corrupt", (unsigned long long)st.corrupt));
      if (st.frames != sv.dequeued) bad.push_back(fmt("frames %llu of %llu dequeued", (unsigned long long)st.frames,
                                                      (unsigned long long)sv.dequeued));
      if (p.low != sv.dequeuedTruncated) bad.push_back(fmt("%llu frames under 30 dB for %llu cut",
                                                           (unsigned long long)p.low,
                                                           (unsigned long long)sv.dequeuedTruncated));
      note = fmt("%llu of %llu frames cut to 10 to 60%% of their bytes: all %llu handed over (corrupt %llu), the %llu "
                 "under 30 dB from the gradient exactly the cut ones",
                 (unsigned long long)sv.dequeuedTruncated, (unsigned long long)sv.dequeued,
                 (unsigned long long)st.frames, (unsigned long long)st.corrupt, (unsigned long long)p.low);
    }
  }
  r.verdict("mjpeg truncated", bad, note);
}

// ai: A folder of the check's own under TMPDIR (else /tmp), removed with what it holds when it goes (2026-10-25: each
// ai: run had left five behind)
struct TempDir {
  std::string path;   // ai: "" where none could be made
  TempDir() {
    const char* t = std::getenv("TMPDIR");
    const std::string d = std::string(t && *t ? t : "/tmp") + "/lizard-v4l2-XXXXXX";
    std::vector<char> b(d.begin(), d.end());
    b.push_back(0);
    if (mkdtemp(b.data())) path = b.data();
  }
  ~TempDir() {
    std::error_code ec;
    if (!path.empty()) std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

// ai: The scene on the fake's c930e through a session on the CPU decoder, its store a folder of its own: the mode
// ai: named started, stats read until done says so or secs pass; a file received read out of the store before the
// ai: session goes. error says why where a step failed.
struct Session {
  std::string error;
  json last;
  double secs = 0;
  bool received = false;
  std::vector<uint8_t> file;   // ai: the received file's bytes
};

Session viaSession(const std::string& scene, const std::string& format, uint32_t w, uint32_t h, double fps,
                   double secs, const std::function<bool(const json&)>& done) {
  Session out;
  std::vector<FakeDevice> ds = devices({"c930e"});
  ds[0].scene = scene;
  FakeV4l2 f(ds);
  Use u(f);
  const TempDir dir;   // ai: after the session below has gone
  if (dir.path.empty()) {
    out.error = std::string("no folder for the session's store: ") + std::strerror(errno);
    return out;
  }
  RxSession::Config cfg;
  cfg.assets = LIZ_ASSETS;
  cfg.storeDir = dir.path;
  cfg.decoder = "cpu";
  cfg.layout = "1:1";
  std::string why;
  auto s = RxSession::create(cfg, why);
  if (!s) {
    out.error = "session: " + why;
    return out;
  }
  if (!s->open(idOf(f, 0), why)) {
    out.error = "open: " + why;
    return out;
  }
  int mode = -1;
  try {
    const json cam = json::parse(s->camera());
    const json& ms = cam.at("modes");
    for (size_t i = 0; i < ms.size(); i++)
      if (ms[i].value("format", "") == format && ms[i].value("w", 0u) == w && ms[i].value("h", 0u) == h &&
          std::fabs(ms[i].value("fps", 0.0) - fps) < 0.01)
        mode = (int)i;
  } catch (const std::exception& e) {
    out.error = std::string("camera(): ") + e.what();
    return out;
  }
  if (mode < 0) {
    out.error = "the session lists no " + format + " mode at that size";
    return out;
  }
  if (!s->start(mode, why)) {
    const std::string fe = f.served(0).error;
    out.error = "start: " + why + (fe.empty() ? "" : " (" + fe + ")");
    return out;
  }
  const double t0 = nowS();
  while (nowS() - t0 < secs) {
    sleepMs(250);
    try {
      out.last = json::parse(s->stats());
    } catch (...) {
      continue;
    }
    if (done && done(out.last)) break;
  }
  out.secs = nowS() - t0;
  const std::string got = s->file();
  if (!got.empty()) {
    // ai: out of the store before the session goes (the store empties its finished folders at the next receiver)
    std::ifstream in(got, std::ios::binary);
    out.file.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    out.received = true;
  }
  s->stop();
  s.reset();
  return out;
}

// ai: The scene's test stream through the session on the CPU decoder. 60 blocks at 30 painted over MJPEG 1920x1080
// ai: verifies blocks; in YUYV 640x480 the frame holds that code in about 345 lines, too few for its picture (the
// ai: simulated grey camera at 640x480 finds it on every frame and verifies no block either, 2026-10-25), so that run
// ai: is held to finding the code, and a 16-block code, which 480 lines resolve, to verifying blocks
void caseSession(Run& r) {
  struct C { const char* name; const char* format; uint32_t w, h; int blocks; bool needBlocks; };
  const C cs[] = {{"session mjpeg", "MJPG", 1920, 1080, 60, true},
                  {"session yuyv", "YUYV", 640, 480, 60, false},
                  {"session yuyv16", "YUYV", 640, 480, 16, true}};
  for (const C& c : cs) {
    if (!r.wants(c.name)) continue;
    std::string why;
    if (!std::strcmp(c.format, "MJPG") && !cam::Jpeg::load(why)) {
      r.fail(c.name, "TurboJPEG: " + why);
      continue;
    }
    const Session s = viaSession(fmt("blocks=%d fps=30 seed=3", c.blocks), c.format, c.w, c.h, 30, 8, nullptr);
    if (!s.error.empty()) {
      r.fail(c.name, s.error);
      continue;
    }
    const json& j = s.last;
    const long frames = j.contains("totals") ? j["totals"].value("frames", 0L) : 0;
    const long blocks = j.contains("totals") ? j["totals"].value("blocks", 0L) : 0;
    const double found = j.value("foundShare", 0.0);
    const std::string state = j.value("state", "?");
    std::vector<std::string> bad;
    if (c.needBlocks && blocks <= 0) bad.push_back("no verified block");
    if (c.needBlocks && state != "test") bad.push_back("state " + state);
    if (!c.needBlocks && found < 0.9) bad.push_back(fmt("the code found on %.0f%% of frames", 100 * found));
    const json cam = j.contains("camera") ? j["camera"] : json();
    r.verdict(c.name, bad,
              fmt("simulated: the test stream (%d blocks, 30 painted) on the fake's %s %ux%u@30, CPU decoder, %.1f s: "
                  "%ld frames read, the code found on %.0f%% of the last second's, %ld verified blocks (%.1f a frame), "
                  "state %s; the camera's line: %s frames, %s dropped, %s corrupt, %.2f ms to luma, agreed %s fps",
                  c.blocks, c.format, c.w, c.h, s.secs, frames, 100 * found, blocks,
                  frames ? (double)blocks / frames : 0.0, state.c_str(), cam.value("frames", json()).dump().c_str(),
                  cam.value("dropped", json()).dump().c_str(), cam.value("corrupt", json()).dump().c_str(),
                  cam.value("convertMs", 0.0), cam.value("agreedFps", json()).dump().c_str()));
  }
}

void caseFile(Run& r) {
  if (!r.wants("file")) return;
  std::string why;
  if (!cam::Jpeg::load(why)) {
    r.fail("file", "TurboJPEG: " + why);
    return;
  }
  const TempDir dir;   // ai: the payload's, after the session's own
  if (dir.path.empty()) {
    r.fail("file", std::string("no folder for the payload: ") + std::strerror(errno));
    return;
  }
  const std::string path = dir.path + "/payload.bin";
  std::vector<uint8_t> bytes(120000);
  std::mt19937 g(11);
  for (auto& b : bytes) b = (uint8_t)g();
  std::ofstream(path, std::ios::binary).write((const char*)bytes.data(), (std::streamsize)bytes.size());
  const Session s = viaSession("blocks=60 fps=30 seed=5 file=" + path, "MJPG", 1920, 1080, 30, 40, [](const json& j) {
    return j.contains("file") && j["file"].is_object() && j["file"].value("verified", false);
  });
  if (!s.error.empty()) {
    r.fail("file", s.error);
    return;
  }
  std::vector<std::string> bad;
  if (!s.received) bad.push_back("no file received in 40 s");
  else if (s.file != bytes) bad.push_back(fmt("%zu bytes received, not the %zu sent", s.file.size(), bytes.size()));
  r.verdict("file", bad,
            fmt("simulated: 120,000 random bytes painted at 60 blocks, 30 a second, read over the fake's MJPG "
                "1920x1080@30 by the CPU decoder: received in %.1f s, byte for byte",
                s.secs));
}

}  // namespace

int v4l2(const Args& a) {
  Run r;
  r.only = a;
  // ai: an empty fake in use for the whole check, so nothing reaches the kernel's table between the cases' own
  FakeV4l2 none({});
  Use guard(none);
  struct Case { const char* name; void (*run)(Run&); };
  const Case cases[] = {
      {"env", caseEnv},         {"list", caseList},       {"open", caseOpen},
      {"modes", caseModes},     {"pick", casePick},       {"controls", caseControls},
      {"set", caseSet},         {"start-stop", caseStartStop}, {"refuse", caseRefuse},
      {"stream", caseStream},   {"stepwise", caseStepwise}, {"formats", caseFormats}, {"card", caseCard},
      {"faults", caseFaults},   {"eio", caseEio},         {"set-streaming", caseSetStreaming},
      {"mjpeg", caseMjpeg},     {"session", caseSession}, {"file", caseFile},
  };
  // ai: a name asked is a case's (or the start of one's) or one of its lines' (the case's name, then more)
  for (const std::string& o : r.only) {
    bool known = false;
    for (const Case& c : cases)
      known = known || !std::string(c.name).compare(0, o.size(), o) || !o.compare(0, std::strlen(c.name), c.name);
    if (!known) {
      std::fprintf(stderr, "cam_check v4l2: no case %s\n", o.c_str());
      return kUsage;
    }
  }
  for (const Case& c : cases) {
    // ai: a case runs where its name or one of its lines' names is asked
    bool want = r.wants(c.name);
    for (auto& o : r.only)
      if (!std::string(o).compare(0, std::strlen(c.name), c.name)) want = true;
    if (want) c.run(r);
  }
  if (!r.lines) {
    std::fprintf(stderr, "cam_check v4l2: no line of a case is named");
    for (const std::string& o : r.only) std::fprintf(stderr, " %s", o.c_str());
    std::fprintf(stderr, "\n");
    return kUsage;
  }
  return r.failed ? kFailed : kHeld;
}

#endif

}  // namespace lizard::check
