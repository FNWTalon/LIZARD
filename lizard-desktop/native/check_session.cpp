// ai: cam_check session (2026-10-25, checks.h): the session over the core's receiver (rx_session.h) on the simulated
// ai: camera (camera/sim.cpp), LIZ_CAMERA_SIM set here before each case opens sim:0 by its id (nothing lists the
// ai: cameras, so no other backend is asked), the CPU decoder with LIZ_VK_DEVICE=llvmpipe so the GPUs stay idle. The
// ai: scenes are sized for the machines the checks run on (Cam, 2026-10-25: CI's runners have 3 and 4 vCPUs): every
// ai: case but halves and 1080 films a 960 x 540 camera free at 60 a second and a code of size 16 in the 128 ring (the
// ai: default), painted 30 a second on a 60 Hz display. A case that judges the camera's rate first measures what this
// ai: machine renders of its scene (Room): with every render in half the camera's period it expects the camera's own
// ai: rate, else the rate the camera gives (its drops its own: this machine's), and in both every frame the camera
// ai: gives reaching the receiver at that rate. Cases, each a line "ok <case>: ..." or "FAIL <case>: ..." (a failure
// ai: followed by the receiver's last log lines):
// ai:   test         the test stream: within 10 s the state is test and blocks are verified, blocks a second printed;
// ai:                the stats' camera line (null before a camera opens, then the camera's); over the 2.5 s after the
// ai:                first read the session's measured rate and the receiver's captured one rates some second of the
// ai:                camera's frames had by its own count; about 60 where the machine has the room
// ai:   preview      on test's session: a picture of the crop's shape (the centre square), at most the side asked,
// ai:                never enlarged, seq rising
// ai:   1080         test on the full-HD camera (1920 x 1080 at 60, the sender page's code of size 60), the camera's
// ai:                rate what this machine renders of it (simulated on a Ryzen 9950X: 60 a second on its 32
// ai:                threads, 30 to 37 on 3 and 4 of them)
// ai:   file         a 300 KB random file painted at 30: received within 60 s, byte-equal at file()'s path
// ai:   halves       2:1 with the screen's two codes (size 8 each, which a 480 x 480 half resolves): captures whose
// ai:                two halves both read, and the preview 2:1
// ai:   gpu          1080 on the GPU decoder: run only with LIZ_CHECK_GPU=1, under scripts/tools/gpulock.sh share
// ai:   reconfigure  a 1 MB file part in on 1:1, the receiver made again for 2:1 and back: the transfer goes on (one
// ai:                header read, no resume) and finishes, byte-equal; with the file in, 2:1 and back keep it (file(),
// ai:                the stats) and nothing is taken again
// ai:   kept         the user's settings set again after each start (rx_session.h set): the simulated camera's
// ai:                exposure set while streaming, another while stopped (kept), each in force after a start, a
// ai:                restart in another mode and a reconfigure, a refusal kept out, forgotten at a close; then on Linux
// ai:                sim/fake_v4l2's c930e, whose backend puts back at stop what set changed: its zoom and exposure
// ai:                back at their own after each stop and the user's again after each start, a zoom asked while
// ai:                stopped leaving the device alone until the start
// ai:   live         the stats' window figures (rx_session.h stats "live", receiver.h cameraClosed): this stream's
// ai:                while it runs, 0 once stopped (the transfer's line and the totals kept); after a start, and after
// ai:                a restart while streaming, live at once, 0 until the new stream's first window closes, then that
// ai:                window a second of this stream alone at the rate the camera gives; "live gpu" the same on the GPU
// ai:                decoder, run only with LIZ_CHECK_GPU=1, under scripts/tools/gpulock.sh share
// ai:   soft         decoder auto where the one Vulkan device is a software one (LIZ_VK_DEVICE=llvmpipe): the C, its
// ai:                gpuWhy naming the device; with LIZ_CHECK_GPU=1 alone (a Vulkan device is made), under
// ai:                scripts/tools/gpulock.sh share
// ai: Arguments: the cases to run, all by default. LIZ_CHECK_VERBOSE=1 echoes the receiver's log on stderr. Every
// ai: figure is the simulated camera's (sim/scene.h: a model) or the fake device's, never a real camera's. About 40 s
// ai: here (a Ryzen 9950X), the gpu, live gpu and soft cases 10 s more.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "camera/clock.h"
#include "checks.h"
#include "json.hpp"
#include "rx_session.h"
#include "sim/scene.h"

#if defined(__linux__)
#include <linux/videodev2.h>

#include "camera/v4l2_sys.h"
#include "sim/fake_v4l2.h"
#endif

#ifndef LIZ_ASSETS
#define LIZ_ASSETS "liblizard/out"
#endif

namespace lizard::check {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// ai: A scene a case films (2026-10-25): its sim::parse keys and the simulated camera's free rate (camfps); spec its
// ai: LIZ_CAMERA_SIM value, with a case's own keys after (file=, codes=)
struct Cam {
  std::string scene;
  int fps = 60;
  std::string spec(const std::string& more = "") const { return scene + " camfps=" + std::to_string(fps) + (more.empty() ? "" : " " + more); }
};
// ai: The scenes, sized for the machines the checks run on (2026-10-25: CI's runners have 4 vCPUs, macOS's 3, where a
// ai: 1920 x 1080 camera at 60 cannot be rendered while the CPU decoder runs; simulated on a Ryzen 9950X: 37 a second
// ai: on 4 of its threads, 30 on 3). SMALL, every case's but halves' and 1080's: a 960 x 540 camera free at 60
// ai: filming a code of size 16 in the 128 ring (the default), painted 30 a second on a 60 Hz display; the frame
// ai: resolves it (10 of its 16 blocks a capture, the clean captures' share at 30 painted) and 3 threads of that
// ai: machine render and decode it at the camera's rate (a render in 6 ms, 1.2 CPUs in all). PAIR: two codes of size
// ai: 8 side by side, which each 480 x 480 half of the 2:1 crop resolves (16 a code read 6 of 16 a half). HD: the
// ai: full-HD camera and the sender page's code of size 60, whose rate there is what the machine renders (Room).
const Cam SMALL{"blocks=16 fps=30 hz=60 w=960 h=540"};
const Cam PAIR{"blocks=8 codes=2 fps=30 hz=60 w=960 h=540"};
const Cam HD{"blocks=60 fps=30 hz=60 w=1920 h=1080"};
const std::string SCENE = SMALL.spec();   // ai: the LIZ_CAMERA_SIM value of the cases that film SMALL as it is
const char* const CASES[] = {"test", "preview", "1080", "file", "halves", "gpu", "reconfigure", "kept", "live", "soft"};

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

void setEnv(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}
void unsetEnv(const char* k) {
#if defined(_WIN32)
  _putenv_s(k, "");
#else
  unsetenv(k);
#endif
}

std::string fmt(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  return b;
}

// ai: What this machine renders of a scene, measured before a case films it (2026-10-25): the simulated camera's own
// ai: work a capture (sim/scene.h render, its 3 painters painting meanwhile, as camera/sim.cpp makes the scene), for
// ai: 0.6 s a capture on the camera's grid of periods, past those a slow render overran, the mean and the worst wall
// ai: ms. The camera hands a capture over once it is rendered and drops those whose time a render overran
// ai: (camera/sim.cpp), so it keeps its own rate while the render and the session's hand-over fit in its period. roomy:
// ai: every render within half the period, the other half left to the decoder's share of the machine, and a case
// ai: expects the camera's own rate; else the camera gives what this machine renders under the decoder's load (60 /
// ai: ceil(render / period) a second at most), and a case judges the frames at the rate it gives them.
struct Room {
  double meanMs = 0, worstMs = 0, periodMs = 0;
  int renders = 0;
  std::string why;   // ai: why it could not be measured
  bool roomy() const { return renders > 0 && worstMs <= periodMs / 2; }
  std::string said() const {
    return "the scene rendered in " + fmt("%.1f", meanMs) + " ms a capture here (" + std::to_string(renders) + " renders, the worst " + fmt("%.1f", worstMs) +
           ") of the camera's " + fmt("%.1f", periodMs) + " ms period: " + (roomy() ? "its own rate expected" : "past half of it, the rate the camera gives this machine's");
  }
};

Room measure(const Cam& c) {
  Room m;
  m.periodMs = 1000.0 / c.fps;
  sim::Sender s;
  sim::Display d;
  sim::Optics o;
  if (!sim::parse(c.scene, s, d, o, m.why)) return m;
  const auto sc = sim::Scene::create(s, d, o, 3, m.why);
  if (!sc) return m;
  std::vector<uint8_t> px(size_t(o.width) * o.height);
  const int64_t period = int64_t(1e9 / c.fps), window = int64_t((o.exposureMs + (o.rolling ? o.readoutMs : 0)) * 1e6);
  int64_t t = cam::steadyNs();
  sc->render(t - window, px.data(), o.width);   // ai: a first render outside the figures (its pool's first touch)
  double sum = 0;
  for (const int64_t end = t + 600'000'000; t < end;) {
    t += period;
    const int64_t now = cam::steadyNs();
    if (t < now) t += ((now - t) / period + 1) * period;
    cam::sleepUntil(t);
    const int64_t a = cam::steadyNs();
    sc->render(t - window, px.data(), o.width);
    const double ms = double(cam::steadyNs() - a) * 1e-6;
    sum += ms;
    m.worstMs = std::max(m.worstMs, ms);
    m.renders++;
  }
  m.meanMs = sum / m.renders;
  return m;
}

// ai: the receiver's log, kept for a case's failure
struct Log {
  std::mutex mu;
  std::vector<std::string> lines;
  const bool echo = std::getenv("LIZ_CHECK_VERBOSE") != nullptr;
  void add(const std::string& m) {
    std::lock_guard<std::mutex> l(mu);
    lines.push_back(m);
    if (echo) std::fprintf(stderr, "  log: %s\n", m.c_str());
  }
  std::string find(const std::string& s) {
    std::lock_guard<std::mutex> l(mu);
    for (const auto& x : lines)
      if (x.find(s) != std::string::npos) return x;
    return "";
  }
  size_t count(const std::string& s) {
    std::lock_guard<std::mutex> l(mu);
    return size_t(std::count_if(lines.begin(), lines.end(), [&](const std::string& x) { return x.find(s) != std::string::npos; }));
  }
  std::string last(const std::string& s) {
    std::lock_guard<std::mutex> l(mu);
    for (auto it = lines.rbegin(); it != lines.rend(); ++it)
      if (it->find(s) != std::string::npos) return *it;
    return "";
  }
  void tail(size_t n) {
    std::lock_guard<std::mutex> l(mu);
    for (size_t i = lines.size() > n ? lines.size() - n : 0; i < lines.size(); i++) std::printf("  log: %s\n", lines[i].c_str());
  }
};

struct Run {
  fs::path dir;
  int failed = 0, n = 0;
  bool ok(const char* c, const std::string& m) {
    std::printf("ok %s: %s\n", c, m.c_str());
    std::fflush(stdout);
    return true;
  }
  bool fail(const char* c, const std::string& m, Log* log = nullptr) {
    std::printf("FAIL %s: %s\n", c, m.c_str());
    if (log) log->tail(15);
    std::fflush(stdout);
    failed++;
    return false;
  }
  // ai: a fresh folder for a case's store
  std::string store(const char* c) {
    const fs::path p = dir / (std::string(c) + "-" + std::to_string(++n));
    fs::create_directories(p);
    return p.u8string();
  }
};

double num(const json& j, const char* k) {
  if (!j.is_object()) return 0;
  const auto it = j.find(k);
  return it != j.end() && it->is_number() ? it->get<double>() : 0;
}
std::string text(const json& j, const char* k) {
  if (!j.is_object()) return "";
  const auto it = j.find(k);
  return it != j.end() && it->is_string() ? it->get<std::string>() : "";
}
json sub(const json& j, const char* k) { return j.is_object() && j.contains(k) ? j[k] : json(); }
json statsOf(RxSession& s) {
  try {
    return json::parse(s.stats());
  } catch (const std::exception&) {
    return json::object();
  }
}

std::unique_ptr<RxSession> made(const std::string& store, const std::string& decoder, const std::string& layout, Log& log, std::string& why) {
  RxSession::Config c;
  c.assets = LIZ_ASSETS;
  c.storeDir = store;
  c.decoder = decoder;
  c.layout = layout;
  c.log = [&log](const std::string& m) { log.add(m); };
  return RxSession::create(c, why);
}

// ai: sim:0 showing `spec`, opened and started in its pick; the frame's size into w, h
bool begin(RxSession& s, const std::string& spec, uint32_t& w, uint32_t& h, std::string& why) {
  setEnv("LIZ_CAMERA_SIM", spec);
  if (!s.open("sim:0", why)) return false;
  json cam;
  try {
    cam = json::parse(s.camera());
  } catch (const std::exception& e) {
    why = std::string("camera() is not JSON: ") + e.what();
    return false;
  }
  const json modes = sub(cam, "modes");
  if (!modes.is_array() || modes.empty()) {
    why = "the simulated camera lists no mode";
    return false;
  }
  int mode = int(num(cam, "pick"));
  if (mode < 0 || size_t(mode) >= modes.size()) mode = 0;
  w = uint32_t(num(modes[size_t(mode)], "w"));
  h = uint32_t(num(modes[size_t(mode)], "h"));
  return s.start(mode, why);
}

std::unique_ptr<RxSession> simulated(Run& r, const char* c, const std::string& spec, const std::string& decoder, const std::string& layout, Log& log,
                                     uint32_t& w, uint32_t& h, std::string& why) {
  auto s = made(r.store(c), decoder, layout, log, why);
  if (s && !begin(*s, spec, w, h, why)) s.reset();
  return s;
}

std::vector<char> slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

fs::path randomFile(const fs::path& dir, const char* name, size_t n, uint32_t seed) {
  std::mt19937 g(seed);
  std::vector<char> b(n);
  for (auto& x : b) x = char(g() >> 24);
  const fs::path p = dir / name;
  std::ofstream(p, std::ios::binary).write(b.data(), std::streamsize(b.size()));
  return p;
}

// ai: the decoder's crop of a w x h frame (receiver.cpp cropsOf): 1:1 the short side's square, 2:1 twice as long
void cropOf(uint32_t w, uint32_t h, bool two, uint32_t& cw, uint32_t& ch) {
  const uint32_t lo = std::min(w, h), hi = std::max(w, h), side = two ? std::min(lo, hi / 2) : lo;
  cw = w >= h ? (two ? 2 * side : side) : side;
  ch = w >= h ? side : (two ? 2 * side : side);
}

// ai: the preview of a running session: the crop's shape at most `side`, never enlarged, a picture (not flat), seq
// ai: rising; "" or why
std::string previewHolds(RxSession& s, uint32_t fw, uint32_t fh, bool two, std::string& said) {
  uint32_t cw = 0, ch = 0;
  cropOf(fw, fh, two, cw, ch);
  std::vector<uint8_t> px;
  uint32_t w = 0, h = 0;
  uint64_t seq = 0;
  const auto ask = [&](uint64_t after, int side) {
    for (const auto t = Clock::now(); since(t) < 2;) {
      if (s.preview(after, side, px, w, h, seq)) return true;
      sleepMs(20);
    }
    return false;
  };
  if (!ask(0, 256)) return "no preview within 2 s";
  const uint32_t lon = std::max(cw, ch);
  const uint32_t ew = lon <= 256 ? cw : uint32_t(std::lround(double(cw) * 256 / lon)), eh = lon <= 256 ? ch : uint32_t(std::lround(double(ch) * 256 / lon));
  if (w != ew || h != eh) return "a " + std::to_string(w) + " x " + std::to_string(h) + " preview of the " + std::to_string(cw) + " x " + std::to_string(ch) + " crop, not " + std::to_string(ew) + " x " + std::to_string(eh);
  if (px.size() != size_t(w) * h) return "the preview's bytes are not w x h";
  // ai: the values, not iterators: the asks below resize px
  const auto mm = std::minmax_element(px.begin(), px.end());
  const int lo = *mm.first, hi = *mm.second;
  if (hi - lo < 32) return "a flat preview (" + std::to_string(lo) + " to " + std::to_string(hi) + ")";
  const uint64_t first = seq;
  if (!ask(first, 256)) return "no newer preview than seq " + std::to_string(first);
  if (seq <= first) return "seq " + std::to_string(seq) + " after " + std::to_string(first);
  const uint64_t second = seq;
  if (!ask(0, 100000)) return "no preview at side 100000";
  if (w != cw || h != ch) return "at side 100000 a " + std::to_string(w) + " x " + std::to_string(h) + " preview of the " + std::to_string(cw) + " x " + std::to_string(ch) + " crop (enlarged or cut)";
  said = std::to_string(ew) + " x " + std::to_string(eh) + " of the " + std::to_string(cw) + " x " + std::to_string(ch) + " crop (grey " + std::to_string(lo) + " to " +
         std::to_string(hi) + "), seq " + std::to_string(first) + " then " + std::to_string(second) + ", the crop whole at side 100000";
  return "";
}

// ai: The rates a second the camera's frames came at, from its own count read at each poll (time, frames; oldest
// ai: first, from index `from` on): each stretch from a poll to the first poll a second or more after it, its frames
// ai: over its length; the least and the most (0 and 0 where no stretch is a second long).
void secondRates(const std::vector<std::pair<double, double>>& polls, size_t from, double& lo, double& hi) {
  lo = 1e300;
  hi = 0;
  size_t k = from;
  for (size_t j = from; j < polls.size(); j++) {
    k = std::max(k, j + 1);
    while (k < polls.size() && polls[k].first - polls[j].first < 1) k++;
    if (k >= polls.size()) break;
    const double rate = (polls[k].second - polls[j].second) / (polls[k].first - polls[j].first);
    lo = std::min(lo, rate);
    hi = std::max(hi, rate);
  }
  if (hi == 0) lo = 0;
}

// ai: The test stream read on `decoder` from `scene`; the preview checked on the same session where asked. The frames
// ai: along the session (2026-10-25), from the first read to the end (2.5 s), whatever rate the machine lets the camera
// ai: give: the session's measured rate and the receiver's captured rate each a rate some second of the camera's own
// ai: count of frames handed over had (its stats read at each poll, secondRates over the span, within two frames and
// ai: 3%: a poll reads the count a frame early or late at either end of a second); and the rate about the camera's 60
// ai: where the machine renders the scene with room (Room).
bool testStream(Run& r, const char* c, const std::string& decoder, const Cam& scene, bool preview) {
  const Room room = measure(scene);
  if (!room.why.empty()) return r.fail(c, "the scene's render was not measured: " + room.why);
  Log log;
  std::string why;
  auto s = made(r.store(c), decoder, "1:1", log, why);
  if (!s) return r.fail(c, why, &log);
  const json before = statsOf(*s);
  if (!before.contains("camera") || !before["camera"].is_null()) return r.fail(c, "the stats before a camera opens have no null camera: " + before.dump());
  if (s->camera() != "{}") return r.fail(c, "camera() before a camera opens is " + s->camera());
  uint32_t fw = 0, fh = 0;
  const std::string spec = scene.spec();
  if (!begin(*s, spec, fw, fh, why)) return r.fail(c, why, &log);
  const auto t0 = Clock::now();
  double read = -1, next = 1, readFrames = 0, at = 0;
  // ai: each poll's time and the camera's own count of frames handed over then
  std::vector<std::pair<double, double>> polls;
  size_t readPoll = 0;
  json j;
  while (since(t0) < 10) {
    sleepMs(100);
    j = statsOf(*s);
    at = since(t0);
    polls.emplace_back(at, num(sub(j, "camera"), "frames"));
    const double total = num(sub(j, "totals"), "blocks");
    if (at >= next) {
      next += 1;
      std::printf("  %s %.0f s: %s, %s blocks a second, camera %s fps, processed %.0f a second\n", c, at, text(j, "state").c_str(),
                  fmt("%.0f", num(j, "blocks") / std::max(0.001, num(j, "windowSecs"))).c_str(), fmt("%.1f", num(sub(j, "camera"), "fps")).c_str(),
                  num(j, "processedFps"));
      std::fflush(stdout);
    }
    if (read < 0 && text(j, "state") == "test" && total > 0) {
      read = at;
      readFrames = num(sub(j, "camera"), "frames");
      readPoll = polls.size() - 1;
    }
    if (read >= 0 && at - read >= 2.5) break;
  }
  const json cam = sub(j, "camera");
  if (read < 0)
    return r.fail(c, "state " + text(j, "state") + " after 10 s, " + fmt("%.0f", num(sub(j, "totals"), "blocks")) + " blocks verified", &log);
  if (text(cam, "id") != "sim:0" || !cam.value("streaming", false))
    return r.fail(c, "the stats' camera line is not the streaming sim:0: " + cam.dump(), &log);
  const double fps = num(cam, "fps"), captured = num(j, "capturedFps"), span = at - read;
  const double own = num(cam, "frames") - readFrames, gave = own / std::max(0.001, span);
  double lo = 0, hi = 0;
  secondRates(polls, readPoll, lo, hi);
  const double least = lo * 0.97 - 2, most = hi * 1.03 + 2;
  const std::string seconds = "the " + fmt("%.1f", lo) + " to " + fmt("%.1f", hi) + " a second its frames came at over the " + fmt("%.1f", span) + " s";
  std::vector<std::string> bad;
  if (own <= 0) bad.push_back("the camera gave no frame in the " + fmt("%.1f", span) + " s after the first read, by its own count");
  if (hi <= 0) bad.push_back("no second of frames in the " + fmt("%.1f", span) + " s after the first read");
  else {
    if (fps < least || fps > most) bad.push_back("the session measured the camera at " + fmt("%.1f", fps) + " fps, outside " + seconds);
    if (captured < least || captured > most) bad.push_back("the receiver captured " + fmt("%.1f", captured) + " a second, outside " + seconds);
  }
  if (room.roomy() && (fps < 50 || fps > 70)) bad.push_back("the camera measured at " + fmt("%.1f", fps) + " fps, not about its 60");
  if (!bad.empty()) {
    std::string all;
    for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
    return r.fail(c, all + " (" + room.said() + ")", &log);
  }
  const double perSec = num(j, "blocks") / std::max(0.001, num(j, "windowSecs"));
  r.ok(c, "the test stream read " + fmt("%.1f", read) + " s after the start, " + fmt("%.0f", perSec) + " blocks a second on " + text(j, "decoder") + "; the camera " +
              fmt("%.1f", gave) + " a second by its own count (" + fmt("%.0f", num(cam, "frames")) + " frames, " + fmt("%.0f", num(cam, "dropped")) + " dropped), " +
              fmt("%.0f", own) + " over the last " + fmt("%.1f", span) + " s, its seconds " + fmt("%.1f", lo) +
              " to " + fmt("%.1f", hi) + "; " + fmt("%.1f", fps) + " fps by the session, " + fmt("%.1f", captured) + " captured and " + fmt("%.0f", num(j, "processedFps")) +
              " processed a second by the receiver (simulated: " + spec + ", a " + std::to_string(fw) + " x " + std::to_string(fh) + " camera; " + room.said() + ")");
  if (preview) {
    std::string said;
    const std::string w = previewHolds(*s, fw, fh, false, said);
    if (w.empty()) r.ok("preview", said);
    else r.fail("preview", w, &log);
  }
  return true;
}

bool fileCase(Run& r) {
  const char* c = "file";
  const fs::path src = randomFile(r.dir, "payload-300k.bin", 300'000, 1);
  Log log;
  std::string why;
  uint32_t fw = 0, fh = 0;
  auto s = simulated(r, c, SCENE + " file=" + src.u8string(), "cpu", "1:1", log, fw, fh, why);
  if (!s) return r.fail(c, why, &log);
  const auto t0 = Clock::now();
  std::string got;
  while (since(t0) < 60 && (got = s->file()).empty()) sleepMs(100);
  const json j = statsOf(*s);
  if (got.empty()) return r.fail(c, "no file in 60 s: " + sub(j, "file").dump(), &log);
  const double secs = since(t0);
  if (slurp(fs::u8path(got)) != slurp(src)) return r.fail(c, got + " differs from the file sent", &log);
  return r.ok(c, "300,000 B received " + fmt("%.1f", secs) + " s after the start (the transfer's own clock " + fmt("%.1f", num(sub(j, "file"), "secs")) +
                     " s), byte-equal at " + got + " (simulated: " + SCENE + ")");
}

bool halvesCase(Run& r) {
  const char* c = "halves";
  Log log;
  std::string why;
  uint32_t fw = 0, fh = 0;
  const std::string spec = PAIR.spec();
  auto s = simulated(r, c, spec, "cpu", "2:1", log, fw, fh, why);
  if (!s) return r.fail(c, why, &log);
  // ai: the series' rows by capture (both halves of one share its time), each capture's latest view kept: a half's
  // ai: row can come a poll before its twin's
  std::map<double, std::vector<double>> rows;
  const auto t0 = Clock::now();
  while (since(t0) < 10) {
    sleepMs(200);
    const std::vector<double> v = s->series(0);
    std::map<double, std::vector<double>> snap;
    for (size_t i = 1; i + 8 <= v.size(); i += 8)
      if (v[i] > 0) snap[v[i]].push_back(v[i + 1]);
    for (auto& [ms, x] : snap)
      if (x.size() >= rows[ms].size()) rows[ms] = x;
  }
  size_t both = 0, any = 0, pairs = 0;
  for (const auto& [ms, x] : rows) {
    (void)ms;
    const size_t reading = size_t(std::count_if(x.begin(), x.end(), [](double b) { return b > 0; }));
    pairs += x.size() >= 2;
    any += reading > 0;
    both += x.size() >= 2 && reading >= 2;
  }
  const json j = statsOf(*s);
  if (text(j, "layout") != "2:1") return r.fail(c, "the receiver's layout is " + text(j, "layout"), &log);
  if (both < 10)
    return r.fail(c, std::to_string(both) + " captures read on both halves of " + std::to_string(rows.size()) + " (" + std::to_string(pairs) +
                         " with two rows, " + std::to_string(any) + " reading any)", &log);
  r.ok(c, std::to_string(both) + " of " + std::to_string(any) + " captures that read had both halves read (" + std::to_string(rows.size()) + " captures in 10 s, " +
              fmt("%.0f", num(j, "blocks") / std::max(0.001, num(j, "windowSecs"))) + " blocks a second; simulated: " + spec + ")");
  std::string said;
  const std::string w = previewHolds(*s, fw, fh, true, said);
  if (w.empty()) r.ok("preview", "2:1: " + said);
  else r.fail("preview", "2:1: " + w, &log);
  return true;
}

// ai: The transfer across reconfigures (rx_session.h reconfigure, receiver.h create's from): a 1 MB file part in on
// ai: 1:1, the receiver made again for 2:1 (where the one centred code is in neither half: nothing read) and back: the
// ai: new receivers go on with the transfer in hand, its part kept through 2:1 in the stats (one header read, no
// ai: resume), and the file comes whole once, byte-equal. Then, the file in, made again for 2:1 and back: file() the
// ai: same path and byte-equal after each, the stats' file verified, and 2 s on 1:1 with the code in view nothing taken
// ai: again (still one header and one "verified" line; 2026-10-25: a new receiver's store let the finished folder go,
// ai: and the file was received again from the start).
bool reconfigureCase(Run& r) {
  const char* c = "reconfigure";
  const fs::path src = randomFile(r.dir, "payload-1m.bin", 1'000'000, 2);
  Log log;
  std::string why;
  uint32_t fw = 0, fh = 0;
  const std::string spec = SCENE + " file=" + src.u8string();
  auto s = simulated(r, c, spec, "cpu", "1:1", log, fw, fh, why);
  if (!s) return r.fail(c, why, &log);
  double part = 0;
  for (const auto t = Clock::now(); since(t) < 30;) {
    sleepMs(50);
    if (!s->file().empty()) return r.fail(c, "the file came whole before a part of it could be seen (too fast to test)", &log);
    part = num(sub(statsOf(*s), "file"), "fraction");
    if (part >= 0.15) break;
  }
  if (part < 0.15) return r.fail(c, "the file only " + fmt("%.0f", part * 100) + "% in after 30 s on 1:1", &log);
  if (!s->reconfigure("cpu", "2:1", why)) return r.fail(c, "to 2:1: " + why, &log);
  const json mid = statsOf(*s);
  if (text(mid, "layout") != "2:1" || !s->streaming()) return r.fail(c, "after the reconfigure to 2:1: layout " + text(mid, "layout") + (s->streaming() ? ", streaming" : ", not streaming"), &log);
  const double kept = num(sub(mid, "file"), "fraction");
  if (kept < part) return r.fail(c, "the transfer's part " + fmt("%.3f", kept) + " in the stats after the reconfigure to 2:1, " + fmt("%.3f", part) + " before it", &log);
  sleepMs(1500);
  if (!s->reconfigure("cpu", "1:1", why)) return r.fail(c, "back to 1:1: " + why, &log);
  const auto t1 = Clock::now();
  std::string got;
  while (since(t1) < 60 && (got = s->file()).empty()) sleepMs(100);
  if (got.empty()) return r.fail(c, "no file 60 s after the way back: " + sub(statsOf(*s), "file").dump(), &log);
  const double secs = since(t1);
  if (slurp(fs::u8path(got)) != slurp(src)) return r.fail(c, got + " differs from the file sent", &log);
  const char* HEADER = "xfer: header payload-1m.bin";
  const char* VERIFIED = "xfer: payload-1m.bin verified";
  if (log.count(HEADER) != 1 || !log.find("xfer: resumed").empty())
    return r.fail(c, "the file came, but its transfer began again: " + std::to_string(log.count(HEADER)) + " header lines, " +
                         (log.find("xfer: resumed").empty() ? std::string("no resume") : "\"" + log.find("xfer: resumed") + "\""), &log);
  // ai: the file in: made again for 2:1 and back, the file kept and not taken again
  for (const char* lay : {"2:1", "1:1"}) {
    if (!s->reconfigure("cpu", lay, why)) return r.fail(c, std::string("the file in, to ") + lay + ": " + why, &log);
    const std::string now = s->file();
    const json f = sub(statsOf(*s), "file");
    if (now != got || !f.value("verified", false))
      return r.fail(c, std::string("the file in, after the reconfigure to ") + lay + ": file() \"" + now + "\" (\"" + got + "\" before), the stats' file " + f.dump(), &log);
    if (slurp(fs::u8path(now)) != slurp(src)) return r.fail(c, std::string("the file in, after the reconfigure to ") + lay + ": " + now + " differs from the file sent", &log);
  }
  sleepMs(2000);
  if (s->file() != got || log.count(HEADER) != 1 || log.count(VERIFIED) != 1)
    return r.fail(c, "2 s on 1:1 after the reconfigures with the file in: file() \"" + s->file() + "\", " + std::to_string(log.count(HEADER)) + " header lines, " +
                         std::to_string(log.count(VERIFIED)) + " \"verified\" lines (one each: the file taken again)", &log);
  return r.ok(c, fmt("%.0f", part * 100) + "% in on 1:1, then 2:1 for 1.5 s (the part kept, " + fmt("%.0f", kept * 100) + "%) and back: the transfer went on (one header, no resume), the file whole " +
                     fmt("%.1f", secs) + " s later, byte-equal; with the file in, 2:1 and back: file() the same path, byte-equal, verified, and 2 s on 1:1 not taken again "
                     "(simulated: " + spec.substr(0, spec.find(" file=")) + ", a 1,000,000 B file)");
}

// ai: a control's value as camera() last read it (NaN where the camera lists none)
double controlOf(RxSession& s, const char* id) {
  try {
    const json cam = json::parse(s.camera());
    for (const auto& k : sub(cam, "controls"))
      if (text(k, "id") == id) return num(k, "value");
  } catch (const std::exception&) {
  }
  return NAN;
}

// ai: the index of a mode camera() lists (format "" any), -1 none
int modeIndex(RxSession& s, const std::string& format, uint32_t w, uint32_t h, double fps) {
  try {
    const json ms = sub(json::parse(s.camera()), "modes");
    for (size_t i = 0; i < ms.size(); i++)
      if ((format.empty() || text(ms[i], "format") == format) && num(ms[i], "w") == w && num(ms[i], "h") == h &&
          (fps <= 0 || std::fabs(num(ms[i], "fps") - fps) < 0.01))
        return int(i);
  } catch (const std::exception&) {
  }
  return -1;
}

// ai: The simulated camera's exposure (its model's own: nothing put back at stop, so the log's "set again" lines say
// ai: what the session did, and a value asked while stopped shows the camera untouched until the start).
bool keptSim(Run& r) {
  const char* c = "kept";
  Log log;
  std::string why;
  uint32_t fw = 0, fh = 0;
  auto s = simulated(r, c, SCENE, "cpu", "1:1", log, fw, fh, why);
  if (!s) return r.fail(c, why, &log);
  const char* AGAIN = "camera: set again after the start: ";
  std::vector<std::string> bad;
  const auto expect = [&](const char* when, double us, size_t agains) {
    const double v = controlOf(*s, "exposure");
    if (v != us) bad.push_back(std::string(when) + ": exposure " + fmt("%.0f", v) + " us, not " + fmt("%.0f", us));
    if (log.count(AGAIN) != agains) bad.push_back(std::string(when) + ": " + std::to_string(log.count(AGAIN)) + " \"set again\" lines, not " + std::to_string(agains));
  };
  const double own = controlOf(*s, "exposure");
  if (!s->set("exposure", 4000, false, why)) return r.fail(c, "exposure 4000 us while streaming: " + why, &log);
  expect("set while streaming", 4000, 0);
  s->stop();
  if (!s->set("exposure", 6000, false, why)) return r.fail(c, "exposure 6000 us while stopped: " + why, &log);
  expect("6000 asked while stopped (kept for the start)", 4000, 0);
  if (s->set("focus", 1, false, why)) bad.push_back("a focus the camera does not list was taken");
  const std::string noFocus = why;
  if (s->set("exposure", 0, true, why)) bad.push_back("an automatic exposure the camera has not was taken");
  const int pick = modeIndex(*s, "", fw, fh, 0), other = modeIndex(*s, "", fw / 2, fh / 2, 0);
  if (pick < 0 || other < 0) return r.fail(c, "the simulated camera lists no " + std::to_string(fw) + " x " + std::to_string(fh) + " and half mode", &log);
  if (!s->start(pick, why)) return r.fail(c, "start: " + why, &log);
  expect("after the start", 6000, 1);
  const std::string said = log.last(AGAIN);
  if (!s->start(other, why)) return r.fail(c, "a restart in the half mode: " + why, &log);
  expect("after a restart in another mode", 6000, 2);
  if (s->set("exposure", 99999, false, why)) bad.push_back("an exposure past the camera's range was taken");
  const std::string tooLong = why;
  if (!s->reconfigure("cpu", "2:1", why)) return r.fail(c, "reconfigure to 2:1: " + why, &log);
  expect("after a reconfigure (the refused value not kept)", 6000, 3);
  s->close();
  uint32_t w2 = 0, h2 = 0;
  if (!begin(*s, SCENE, w2, h2, why)) return r.fail(c, "open and start again: " + why, &log);
  expect("opened again (forgotten at the close)", own, 3);
  if (!bad.empty()) {
    std::string all;
    for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
    return r.fail(c, all, &log);
  }
  return r.ok(c, "the simulated camera's exposure: 4000 us set streaming, 6000 asked stopped (kept), 6000 after the start, a restart in the "
                 "half mode and a reconfigure to 2:1, each with its log line (\"" + said +
                 "\"); refused and not kept: focus (\"" + noFocus + "\"), 99999 us (\"" + tooLong + "\"); after a close and open the "
                 "camera's own " + fmt("%.0f", own) + " us, nothing set again (simulated: " + SCENE + ")");
}

#if defined(__linux__)
// ai: sim/fake_v4l2's c930e (the gradient, no screen), opened through the fake's table: its backend puts back at
// ai: stop what set changed, as a UVC device's would be, so what the device holds after each stop and start is the
// ai: backend's put-back and the session's setting again
bool keptV4l2(Run& r) {
  const char* c = "kept v4l2";
  std::vector<sim::FakeDevice> ds(1);
  ds[0].profile = "c930e";
  sim::FakeV4l2 f(ds);
  struct Use {
    const cam::V4l2Sys* before;
    ~Use() { cam::v4l2UseSys(before); }
  } use{cam::v4l2UseSys(f.sys())};
  Log log;
  std::string why;
  auto s = made(r.store("kept-v4l2"), "cpu", "1:1", log, why);
  if (!s) return r.fail(c, why, &log);
  const std::string id = "v4l2:" + f.byId(0);
  if (!s->open(id, why)) return r.fail(c, "open " + id + ": " + why, &log);
  const auto ctl = [&](uint32_t cid) {
    int64_t v = INT64_MIN;
    f.control(0, cid, v);
    return v;
  };
  const int64_t zoom0 = ctl(V4L2_CID_ZOOM_ABSOLUTE), auto0 = ctl(V4L2_CID_EXPOSURE_AUTO), abs0 = ctl(V4L2_CID_EXPOSURE_ABSOLUTE);
  const int a = modeIndex(*s, "YUYV", 640, 480, 30), b = modeIndex(*s, "YUYV", 1280, 720, 10);
  if (a < 0 || b < 0) return r.fail(c, "the fake c930e lists no YUYV 640x480@30 and 1280x720@10", &log);
  std::vector<std::string> bad;
  const auto expect = [&](const std::string& when, int64_t zoom, int64_t ea, int64_t eabs) {
    const int64_t z = ctl(V4L2_CID_ZOOM_ABSOLUTE), e = ctl(V4L2_CID_EXPOSURE_AUTO), x = ctl(V4L2_CID_EXPOSURE_ABSOLUTE);
    if (z != zoom || e != ea || x != eabs)
      bad.push_back(when + ": zoom " + std::to_string(z) + ", exposure_auto " + std::to_string(e) + ", exposure_absolute " + std::to_string(x) +
                    " (not " + std::to_string(zoom) + ", " + std::to_string(ea) + ", " + std::to_string(eabs) + ")");
  };
  if (!s->start(a, why)) return r.fail(c, "start: " + why, &log);
  if (!s->set("zoom", 300, false, why)) return r.fail(c, "zoom 300: " + why, &log);
  if (!s->set("exposure", 20000, false, why)) return r.fail(c, "exposure 20000 us: " + why, &log);
  // ai: 20000 us is 200 of exposure_absolute's 100 us, under its manual master (V4L2_EXPOSURE_MANUAL)
  expect("set while streaming", 300, V4L2_EXPOSURE_MANUAL, 200);
  s->stop();
  expect("after the stop", zoom0, auto0, abs0);
  if (!s->start(a, why)) return r.fail(c, "start again: " + why, &log);
  expect("after the start", 300, V4L2_EXPOSURE_MANUAL, 200);
  s->stop();
  if (!s->set("zoom", 400, false, why)) return r.fail(c, "zoom 400 while stopped: " + why, &log);
  expect("zoom 400 asked while stopped", zoom0, auto0, abs0);
  if (!s->start(a, why)) return r.fail(c, "start after the stopped set: " + why, &log);
  expect("after the next start", 400, V4L2_EXPOSURE_MANUAL, 200);
  if (!s->start(b, why)) return r.fail(c, "a restart in YUYV 1280x720@10: " + why, &log);
  expect("after a restart in another mode", 400, V4L2_EXPOSURE_MANUAL, 200);
  if (!s->reconfigure("cpu", "2:1", why)) return r.fail(c, "reconfigure to 2:1: " + why, &log);
  expect("after a reconfigure", 400, V4L2_EXPOSURE_MANUAL, 200);
  s->close();
  expect("after the close", zoom0, auto0, abs0);
  if (!bad.empty()) {
    std::string all;
    for (const auto& x : bad) all += (all.empty() ? "" : "; ") + x;
    return r.fail(c, all, &log);
  }
  return r.ok(c, "the fake c930e (its backend puts set's changes back at stop): zoom 300 and exposure 20000 us set streaming, the device at its own "
                 "zoom " + std::to_string(zoom0) + " and exposure_auto " + std::to_string(auto0) + " (" + std::to_string(abs0) +
                 ") after each stop and at the user's after each start (\"" + log.last("camera: set again after the start: ") +
                 "\"); zoom 400 asked while stopped left the device alone until the start; held through a restart in YUYV 1280x720@10 "
                 "and a reconfigure to 2:1; the device's own again after the close");
}
#endif

// ai: the stats' window figures, which read 0 with no stream, and from a start until the stream's first window closes
bool windowZero(const json& j) {
  for (const char* k : {"capturedFps", "processedFps", "dropped", "foundShare", "blocks", "goodputKBs", "cpuMs", "gpuMs"})
    if (num(j, k) != 0) return false;
  return !j.value("test", false);
}

// ai: From a start at t0 to the stream's first window: live at every poll (no mask), the window's figures 0 until
// ai: that window closes (nothing of the stream before the stop, nor of the time off: receiver.h cameraClosed), then
// ai: that window a second of this stream alone, 0.95 to 2.5 s after the start: the rate the camera gives (the
// ai: session's count, 60 a second where the machine renders the scene in time; a window over the time off reads
// ai: half that or less), no more frames decoded than captured, and blocks read; what the decoder kept up with is the
// ai: machine's (an instrumented build decodes 26 a second). How it went into said, or why not into bad.
void firstWindow(RxSession& s, Clock::time_point t0, const std::string& when, std::vector<std::string>& bad, std::string& said) {
  json first;
  double at = -1;
  int zero = 0;
  while (since(t0) < 6) {
    const json o = statsOf(s);
    if (!o.value("live", false)) {
      bad.push_back(when + ": not live " + fmt("%.2f", since(t0)) + " s after the start");
      return;
    }
    if (num(o, "capturedFps") > 0) {
      first = o;
      at = since(t0);
      break;
    }
    if (!windowZero(o)) {
      bad.push_back(when + ": " + fmt("%.2f", since(t0)) + " s after the start, before the stream's first window, " + fmt("%.0f", num(o, "blocks")) + " blocks, " +
                    fmt("%.0f", num(o, "processedFps")) + " processed a second, " + fmt("%.0f", num(o, "goodputKBs")) + " KB/s");
      return;
    }
    zero++;
    sleepMs(50);
  }
  if (at < 0) {
    bad.push_back(when + ": no window closed 6 s after the start");
    return;
  }
  const double secs = num(first, "windowSecs"), captured = num(first, "capturedFps");
  // ai: the rate the camera gives this machine (the session's own count, the stats' camera fps), and two of its
  // ai: periods: a window closes at the first frame or batch a second on
  const double camFps = num(sub(first, "camera"), "fps"), late = camFps > 0 ? 2 / camFps : 0;
  const size_t had = bad.size();
  // ai: a window of this stream closes a second after its first frame at the soonest: one seen before is a window kept
  // ai: from the stream before
  if (at < 0.95) bad.push_back(when + ": a window " + fmt("%.2f", at) + " s after the start, before a second of this stream (the stream before's)");
  if (at > 2.5 + late) bad.push_back(when + ": the first window closed " + fmt("%.2f", at) + " s after the start");
  if (secs < 0.95 || secs >= std::max(1.5, 1.3 + late)) bad.push_back(when + ": the first window " + fmt("%.2f", secs) + " s long, not a second of this stream");
  if (captured < 0.75 * camFps)
    bad.push_back(when + ": the first window " + fmt("%.0f", captured) + " captured a second, the camera " + fmt("%.1f", camFps) + " (a window over more than this stream)");
  // ai: a stream's first window decodes only frames it was handed (1:1: a frame each), so never more than it captured;
  // ai: more is frames of the stream before, still in the decoder at the close, counted
  if (num(first, "processedFps") > captured)
    bad.push_back(when + ": the first window " + fmt("%.0f", num(first, "processedFps")) + " processed a second of " + fmt("%.0f", captured) + " captured");
  if (num(first, "blocks") <= 0) bad.push_back(when + ": the first window read no block");
  if (bad.size() > had) return;
  said = when + " live at once, the figures 0 for " + std::to_string(zero) + " polls, then the first window " + fmt("%.2f", at) + " s in: " + fmt("%.2f", secs) +
         " s, " + fmt("%.0f", captured) + " captured (the camera " + fmt("%.1f", camFps) + " fps) and " + fmt("%.0f", num(first, "processedFps")) +
         " processed a second, " + fmt("%.0f", num(first, "blocks")) + " blocks";
}

// ai: decoder cpu ("live") or gpu ("live gpu")
bool liveCase(Run& r, const std::string& decoder) {
  const std::string name = decoder == "cpu" ? "live" : "live " + decoder;
  const char* c = name.c_str();
  Log log;
  std::string why;
  uint32_t fw = 0, fh = 0;
  auto s = simulated(r, "live", SCENE, decoder, "1:1", log, fw, fh, why);
  if (!s) return r.fail(c, why, &log);
  if (!statsOf(*s).value("live", false)) return r.fail(c, "a fresh receiver's first stream not live at its start", &log);
  json j;
  for (const auto t = Clock::now(); since(t) < 10;) {
    sleepMs(100);
    j = statsOf(*s);
    if (j.value("live", false) && text(j, "state") == "test" && num(j, "blocks") > 0 && num(j, "goodputKBs") > 0) break;
  }
  if (num(j, "blocks") <= 0) return r.fail(c, "the test stream read no block in 10 s: " + text(j, "state"), &log);
  const double total = num(sub(j, "totals"), "blocks");
  s->stop();
  std::vector<std::string> bad;
  for (int k = 0; k < 2; k++) {
    if (k) sleepMs(1200);
    const json o = statsOf(*s);
    const char* when = k ? "1.2 s after the stop" : "at the stop";
    if (o.value("live", true)) bad.push_back(std::string(when) + ": live");
    if (!windowZero(o)) bad.push_back(std::string(when) + ": the window's figures " + fmt("%.0f", num(o, "blocks")) + " blocks, " + fmt("%.0f", num(o, "goodputKBs")) + " KB/s");
    if (num(sub(o, "camera"), "fps") != 0 || sub(o, "camera").value("streaming", true)) bad.push_back(std::string(when) + ": the camera's line still streaming");
    if (num(sub(o, "totals"), "blocks") < total) bad.push_back(std::string(when) + ": the totals went back");
  }
  const int pick = std::max(0, modeIndex(*s, "", fw, fh, 0));
  std::string again, restart;
  auto t0 = Clock::now();
  if (!s->start(pick, why)) return r.fail(c, "start again: " + why, &log);
  firstWindow(*s, t0, "after the start", bad, again);
  // ai: a restart while streaming (rx_session.h start: the stream stopped and started again at once), with frames
  // ai: of the stream before still in the decoder at its close
  sleepMs(1000);
  t0 = Clock::now();
  if (!s->start(pick, why)) return r.fail(c, "a restart while streaming: " + why, &log);
  firstWindow(*s, t0, "after a restart while streaming", bad, restart);
  if (!bad.empty()) {
    std::string all;
    for (const auto& x : bad) all += (all.empty() ? "" : "; ") + x;
    return r.fail(c, all, &log);
  }
  return r.ok(c, "decoder " + text(statsOf(*s), "decoder") + ": live at the start (a fresh receiver); after the stop 0 blocks, 0 KB/s, live false, the totals kept (" +
                 fmt("%.0f", total) + " blocks); " + again + "; " + restart + " (simulated: " + SCENE + ")");
}

// ai: auto on a machine whose one Vulkan device is a software one (LIZ_VK_DEVICE=llvmpipe names it): the C
bool softCase(Run& r) {
  const char* c = "soft";
  Log log;
  std::string why;
  auto s = made(r.store(c), "auto", "1:1", log, why);
  if (!s) return r.fail(c, why, &log);
  const json j = statsOf(*s);
  const std::string dec = text(j, "decoder"), gw = text(j, "gpuWhy");
  if (gw.find("no Vulkan") != std::string::npos) {
    std::printf("skip soft: no llvmpipe here (%s)\n", gw.c_str());
    return true;
  }
  if (dec != "cpu" || gw.rfind("a software Vulkan device: ", 0) != 0) return r.fail(c, "decoder " + dec + ", gpuWhy \"" + gw + "\"", &log);
  return r.ok(c, "decoder auto where the device is llvmpipe: the C, gpuWhy \"" + gw + "\"");
}

}  // namespace

int session(const Args& a) {
  std::set<std::string> want(a.begin(), a.end());
  for (const auto& w : want) {
    bool known = false;
    for (const char* k : CASES) known |= w == k;
    if (!known) {
      std::fprintf(stderr, "cam_check session: no case %s (test, preview, 1080, file, halves, gpu, reconfigure, kept, live, soft)\n", w.c_str());
      return 2;
    }
  }
  const auto on = [&](const char* k) { return want.empty() || want.count(k); };
  Run r;
  std::random_device rd;
  char name[48];
  std::snprintf(name, sizeof name, "lizard-check-session-%08x", unsigned(rd()));
  std::error_code e;
  r.dir = fs::temp_directory_path(e) / name;
  fs::create_directories(r.dir, e);
  if (e) {
    std::printf("FAIL session: no folder for the stores: %s\n", e.message().c_str());
    return 1;
  }
  const char* vk = std::getenv("LIZ_VK_DEVICE");
  const std::string userVk = vk ? vk : "";
  setEnv("LIZ_VK_DEVICE", "llvmpipe");
  std::printf("  session: the simulated camera (figures are the model's), stores in %s\n", r.dir.u8string().c_str());
  std::fflush(stdout);
  if (on("test") || on("preview")) testStream(r, on("test") ? "test" : "test (for preview)", "cpu", SMALL, on("preview"));
  if (on("1080")) testStream(r, "1080", "cpu", HD, false);
  if (on("file")) fileCase(r);
  if (on("halves")) halvesCase(r);
  if (on("gpu")) {
    const char* g = std::getenv("LIZ_CHECK_GPU");
    if (!g || std::string(g) != "1") std::printf("skip gpu: LIZ_CHECK_GPU=1 runs it, under scripts/tools/gpulock.sh share\n");
    else {
      std::printf("  gpu: the GPU decoder runs: this check must run under scripts/tools/gpulock.sh share\n");
      if (userVk.empty()) unsetEnv("LIZ_VK_DEVICE");
      else setEnv("LIZ_VK_DEVICE", userVk);
      testStream(r, "gpu", "gpu", HD, false);
      setEnv("LIZ_VK_DEVICE", "llvmpipe");
    }
  }
  if (on("reconfigure")) reconfigureCase(r);
  if (on("kept")) {
    keptSim(r);
#if defined(__linux__)
    keptV4l2(r);
#else
    std::printf("skip kept v4l2: Linux only (sim/fake_v4l2)\n");
#endif
  }
  if (on("live")) {
    liveCase(r, "cpu");
    const char* g = std::getenv("LIZ_CHECK_GPU");
    if (!g || std::string(g) != "1") std::printf("skip live gpu: LIZ_CHECK_GPU=1 runs it, under scripts/tools/gpulock.sh share\n");
    else {
      if (userVk.empty()) unsetEnv("LIZ_VK_DEVICE");
      else setEnv("LIZ_VK_DEVICE", userVk);
      liveCase(r, "gpu");
      setEnv("LIZ_VK_DEVICE", "llvmpipe");
    }
  }
  if (on("soft")) {
    const char* g = std::getenv("LIZ_CHECK_GPU");
    if (!g || std::string(g) != "1") std::printf("skip soft: LIZ_CHECK_GPU=1 runs it (a Vulkan device is made), under scripts/tools/gpulock.sh share\n");
    else softCase(r);
  }
  unsetEnv("LIZ_CAMERA_SIM");
  if (userVk.empty()) unsetEnv("LIZ_VK_DEVICE");
  else setEnv("LIZ_VK_DEVICE", userVk);
  if (!r.failed) fs::remove_all(r.dir, e);
  else std::printf("  session: the stores kept in %s\n", r.dir.u8string().c_str());
  std::fflush(stdout);
  return r.failed ? 1 : 0;
}

}  // namespace lizard::check
