// ai: cam_check scene (2026-10-25): the simulated screen and camera (sim/scene.h) held to what they model, through the
// ai: C decoder (lizard.h liz_decode, the reference): a clean capture reads every block, each the test stream's, and
// ai: the pilots read the picture's count; a capture across a change mixes the two pictures as the pilots' readings
// ai: say; the default optics read most blocks at a good phase across a sweep of phases over a picture; a 2:1 pair
// ai: reads both symbols; a render's wall time at 1920 x 1080; the display's clock and its pictures exact; a file's
// ai: pictures received whole; the simulated camera (camera/sim.cpp) in each of its formats and through its trigger
// ai: (MJPEG skipped saying why where no TurboJPEG loads and the receiver does without it: checks.h jpegOptional).
// ai: Every figure is the simulator's. `cam_check scene [case ...]` runs the cases named (parse, clean, straddle,
// ai: sweep, pair, speed, clock, file, camera), all by default. The camera case sets LIZ_CAMERA_SIM itself and opens sim:0
// ai: alone (never the registry's list, which would look at the system's cameras).
#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "backends.h"
#include "checks.h"
#include "lizard.h"
#include "sim/scene.h"
#include "trigger.h"

namespace lizard::check {
namespace {

int failed = 0;
void ok(const char* c, const std::string& s) { std::printf("ok %s: %s\n", c, s.c_str()); std::fflush(stdout); }
void bad(const char* c, const std::string& s) { std::printf("FAIL %s: %s\n", c, s.c_str()); std::fflush(stdout); failed++; }
void say(const std::string& s) { std::printf("  %s\n", s.c_str()); std::fflush(stdout); }
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  va_list a, b;
  va_start(a, f);
  va_copy(b, a);
  const int n = std::vsnprintf(nullptr, 0, f, a);
  va_end(a);
  std::string s(size_t(std::max(n, 0)) + 1, '\0');
  std::vsnprintf(&s[0], s.size(), f, b);
  va_end(b);
  s.resize(size_t(std::max(n, 0)));
  return s;
}
// ai: the environment variable the camera case sets (MinGW's C library has no setenv)
void setEnv(const char* k, const char* v) {
#if defined(_WIN32)
  _putenv_s(k, v ? v : "");
#else
  if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
int64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void sleepTo(int64_t t) {
  std::this_thread::sleep_until(std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::nanoseconds(t))));
}

// ai: one capture's reading: its regions decoded (layout 1 the centre square, 2 the 2:1 pair), the blocks that are
// ai: the test stream's payload for their id counted
struct Read {
  int blocks = 0, stream = 0, total = 0, found = 0;
  liz_decoded d[2]{};
  int regions = 0;
};
Read readFrame(liz_decoder* dec, const std::vector<uint8_t>& img, int w, int h, int layout, int& held, std::vector<uint8_t>* keep = nullptr) {
  Read r;
  int rect[2][4];
  r.regions = liz_layout_rects(w, h, layout, rect);
  std::vector<uint8_t> v(size_t(liz_decoder_max_blocks(dec)) * LIZ_BLOCK);
  uint8_t pay[LIZ_PAYLOAD];
  if (keep) keep->clear();
  for (int k = 0; k < r.regions; k++) {
    const int n = liz_decode(dec, img.data() + size_t(rect[k][1]) * w + rect[k][0], rect[k][2], rect[k][3], w, LIZ_GREY8, &held, v.data(), &r.d[k]);
    r.total += r.d[k].total;
    r.found += r.d[k].found;
    if (n <= 0) continue;
    r.blocks += n;
    for (int b = 0; b < n; b++) {
      const uint8_t* p = v.data() + size_t(b) * LIZ_BLOCK;
      const uint32_t id = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
      liz_stream_fill(id, pay);
      r.stream += !std::memcmp(pay, p + LIZ_ID_BYTES, LIZ_PAYLOAD);
    }
    if (keep) keep->insert(keep->end(), v.begin(), v.begin() + ptrdiff_t(n) * LIZ_BLOCK);
  }
  return r;
}
// ai: the picture count mod 4 the pilots' signs say (focus.h focus_parity: bit 0 on the even blocks, bit 1 on the odd)
int countOf(const liz_decoded& d) { return (d.pilot_r[0] < 0 ? 1 : 0) | (d.pilot_r[1] < 0 ? 2 : 0); }
std::string pilots(const liz_decoded& d) {
  return fmt("pilots %+.3f (sd %.3f) even, %+.3f (sd %.3f) odd", d.pilot_r[0], d.pilot_sd[0], d.pilot_r[1], d.pilot_sd[1]);
}

std::unique_ptr<sim::Scene> make(const char* c, const sim::Sender& s, const sim::Display& d, const sim::Optics& o, int painters = 4) {
  std::string why;
  auto sc = sim::Scene::create(s, d, o, painters, why);
  if (!sc) bad(c, "the scene: " + why);
  return sc;
}
// ai: the first picture shown at least `leadMs` from now, so its capture renders from pictures already painted
int64_t nextPicture(const sim::Scene& sc, double leadMs) { return sc.pictureAt(now() + int64_t(leadMs * 1e6)) + 1; }

sim::Optics clean() {
  sim::Optics o;
  o.rolling = false;
  o.exposureMs = 2;
  o.blurPx = 0;
  o.readNoise = 0;
  o.shotNoise = 0;
  return o;
}

void caseClean(liz_decoder* dec) {
  const char* c = "clean";
  sim::Sender s;
  sim::Display d;
  d.scan = 0;
  d.responseMs = 0;
  const sim::Optics o = clean();
  auto sc = make(c, s, d, o);
  if (!sc) return;
  std::vector<uint8_t> img(size_t(o.width) * o.height);
  const double pic = 1e9 / s.fps;
  int held = 0, all = 0, wrong = 0, counts = 0;
  std::string line;
  for (int i = 0; i < 3; i++) {
    const int64_t k = nextPicture(*sc, 60);
    const int64_t t0 = sc->shownAt(k) + int64_t(pic / 2 - o.exposureMs * 1e6 / 2);
    sc->render(t0, img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 1, held);
    const bool whole = r.found && r.blocks == r.total && r.total == liz_blocks_for(s.blocks) && r.stream == r.blocks;
    all += whole;
    wrong += r.blocks - r.stream;
    counts += r.found && countOf(r.d[0]) == int(k & 3);
    line += fmt("%s picture %lld (count %lld): %d of %d blocks, %d the stream's, the pilots' count %d, %s", i ? ";" : "",
                (long long)k, (long long)(k & 3), r.blocks, r.total, r.stream, countOf(r.d[0]), pilots(r.d[0]).c_str());
    if (sc->pictureAt(t0) != k || sc->pictureAt(t0 + int64_t(o.exposureMs * 1e6)) != k) bad(c, "the capture's window is not inside its picture");
  }
  const std::string what = fmt("size %d in the 128 ring, global shutter, %.0f ms mid-picture, no noise or blur (simulated):", s.blocks, o.exposureMs) + line;
  if (all == 3 && !wrong && counts == 3) ok(c, what); else bad(c, what);
}

void caseStraddle(liz_decoder* dec) {
  const char* c = "straddle";
  sim::Sender s;
  sim::Display d;
  d.scan = 0;
  d.responseMs = 0;
  const sim::Optics o = clean();
  auto sc = make(c, s, d, o);
  if (!sc) return;
  std::vector<uint8_t> img(size_t(o.width) * o.height);
  int held = 0;
  // ai: a clean capture first (the held word, and the channel's own reading c0: the resampled picture's distortion
  // ai: is in m2 too), then windows over the change from picture k - 1 to k, f of the window before it. The pilots'
  // ai: theory (focus.h focus_pilot), every sample the same blend: a tail both pictures paint alike reads
  // ai: 1 / sqrt(m2), one they paint opposite (1 - 2f) / sqrt(m2), m2 = 1 - 2f + 2f^2 + 1 / c0^2 - 1 (the data's rms
  // ai: lowered by the blend, the channel's distortion as it was); the even blocks' bit both neighbours flip, the odd
  // ai: blocks' bit flips from k - 1 to k where k is even; the sign k's bit
  double c0 = 0;
  int n0 = 0;
  for (int i = 0; i < 3; i++) {
    const int64_t k = nextPicture(*sc, 60);
    sc->render(sc->shownAt(k) + int64_t(5e6), img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 1, held);
    if (r.found) { c0 += std::abs(r.d[0].pilot_r[0]) + std::abs(r.d[0].pilot_r[1]); n0 += 2; }
  }
  c0 = n0 ? c0 / n0 : 1;
  bool good = n0 > 0;
  std::string line = fmt(" three clean captures' pilots %.3f;", c0);
  for (const double f : {0.5, 0.25}) {
    const int64_t k = nextPicture(*sc, 60);
    const int64_t t0 = sc->shownAt(k) - int64_t(f * o.exposureMs * 1e6);
    sc->render(t0, img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 1, held);
    const double q = std::sqrt(1 - 2 * f + 2 * f * f + 1 / (c0 * c0) - 1), sign0 = (k & 1) ? -1 : 1, sign1 = (k & 2) ? -1 : 1;
    const double want0 = sign0 * (1 - 2 * f) / q, want1 = (k & 1) ? sign1 / q : sign1 * (1 - 2 * f) / q;
    const bool near = r.found && std::abs(r.d[0].pilot_r[0] - want0) < 0.08 && std::abs(r.d[0].pilot_r[1] - want1) < 0.08;
    if (f == 0.5) good = good && r.blocks < r.total / 2 && near;
    else good = good && near;
    line += fmt("%s %.0f%% of picture %lld before the change to %lld: found %d, word %d, held %d, %d of %d blocks, %s (theory %+.3f, %+.3f)",
                f == 0.5 ? "" : ";", 100 * f, (long long)(k - 1), (long long)k, r.d[0].found, r.d[0].word, r.d[0].held_used, r.blocks, r.total,
                pilots(r.d[0]).c_str(), want0, want1);
  }
  const std::string what = fmt("global shutter, %.0f ms over a change (simulated):", o.exposureMs) + line;
  if (good) ok(c, what); else bad(c, what);
}

// ai: the phase that read best, for the cases after the sweep (ns past a picture's first refresh), -1 none yet
double bestPhase = -1;

void caseSweep(liz_decoder* dec) {
  const char* c = "sweep";
  const sim::Sender s;
  const sim::Display d;
  const sim::Optics o;
  auto sc = make(c, s, d, o);
  if (!sc) return;
  std::vector<uint8_t> img(size_t(o.width) * o.height);
  const double pic = 1e9 / s.fps;
  int held = 0, best = -1, wrong = 0;
  const int want = liz_blocks_for(s.blocks);
  say(fmt("the default optics: rolling %.0f ms readout, %.0f ms exposure, blur %.1f px, noise %.3f + %.3f, yaw %.0f tilt %.0f roll %.0f deg, "
          "fill %.2f, share %.2f, scan %.2f, response %.0f ms; size %d in the 128 ring at %d a second on %.0f Hz (simulated)",
          o.readoutMs, o.exposureMs, o.blurPx, o.readNoise, o.shotNoise, o.yawDeg, o.tiltDeg, o.rollDeg, o.fill, d.share, d.scan,
          d.responseMs, s.blocks, s.fps, d.hz));
  say("phase ms  blocks  pilots (even, odd)");
  int got[17];
  for (int i = 0; i <= 16; i++) {
    const double ph = pic * i / 16;
    const int64_t k = nextPicture(*sc, 60);
    sc->render(sc->shownAt(k) + int64_t(ph), img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 1, held);
    wrong += r.blocks - r.stream;
    say(fmt("%8.2f  %3d/%-3d %+.3f %+.3f", ph * 1e-6, r.blocks, want, r.d[0].pilot_r[0], r.d[0].pilot_r[1]));
    got[i] = r.blocks;
    best = std::max(best, r.blocks);
  }
  // ai: the phase the later cases take: the middle of the longest run of phases that read the best
  int runAt = 0, runLen = 0;
  for (int i = 0, n = 0; i <= 16; i++) {
    n = got[i] == best ? n + 1 : 0;
    if (n > runLen) { runLen = n; runAt = i - n + 1; }
  }
  bestPhase = pic * (runAt + (runLen - 1) / 2.0) / 16;
  int reading = 0;
  for (int i = 0; i <= 16; i++) reading += got[i] >= want * 9 / 10;
  const std::string what = fmt("%d of 17 phases read 90%% of the blocks or more, %d of %d at best (from %.2f to %.2f ms past a picture's first refresh), "
                               "%d not the stream's; %s", reading, best, want, pic * runAt / 16 * 1e-6, pic * (runAt + runLen - 1) / 16 * 1e-6,
                               wrong, sc->stats().c_str());
  if (best >= want * 9 / 10 && !wrong) ok(c, what); else bad(c, what);
}

void casePair(liz_decoder* dec) {
  const char* c = "pair";
  sim::Sender s;
  s.codes = 2;
  s.blocks = 30;
  const sim::Display d;
  const sim::Optics o;
  auto sc = make(c, s, d, o);
  if (!sc) return;
  std::vector<uint8_t> img(size_t(o.width) * o.height);
  int held = 0, both = 0, wrong = 0;
  const int per = liz_blocks_for(s.blocks);
  // ai: the sweep's best phase where it ran, else a third of the picture
  const double ph = bestPhase >= 0 ? bestPhase : 1e9 / s.fps / 3;
  std::string line;
  for (int i = 0; i < 2; i++) {
    const int64_t k = nextPicture(*sc, 60);
    sc->render(sc->shownAt(k) + int64_t(ph), img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 2, held);
    wrong += r.blocks - r.stream;
    const int a = r.d[0].found ? r.d[0].verified : 0, b = r.d[1].found ? r.d[1].verified : 0;
    both += r.regions == 2 && a >= per * 9 / 10 && b >= per * 9 / 10;
    line += fmt("%s picture %lld: the left %d of %d blocks, the right %d of %d", i ? ";" : "", (long long)k, a, per, b, per);
  }
  const std::string what = fmt("two codes of size %d side by side, the default optics at %.2f ms, the 2:1 squares (simulated):", s.blocks, ph * 1e-6) + line;
  if (both == 2 && !wrong) ok(c, what); else bad(c, what);
}

void caseSpeed() {
  const char* c = "speed";
  const sim::Sender s;
  const sim::Display d;
  const sim::Optics o;
  auto sc = make(c, s, d, o);
  if (!sc) return;
  std::vector<uint8_t> img(size_t(o.width) * o.height);
  const int64_t k = nextPicture(*sc, 100);
  sc->render(sc->shownAt(k), img.data(), o.width);   // ai: a first render outside the mean (its threads' first touch)
  double sum = 0;
  for (int i = 0; i < 20; i++) {
    const int64_t a = now();
    sc->render(sc->shownAt(k) + int64_t(i) * 500'000, img.data(), o.width);
    sum += double(now() - a) * 1e-6;
  }
  // ai: a figure to report (the render's pool is half the hardware threads, at most 16, so a small machine takes
  // ai: longer); a tenth of a second is a render gone wrong anywhere
  const double ms = sum / 20;
  const std::string what = fmt("a %u x %u capture of the default scene in %.2f ms of wall time (mean of 20; %u hardware threads), %s",
                               o.width, o.height, ms, std::thread::hardware_concurrency(), sc->stats().c_str());
  if (ms < 100) ok(c, what); else bad(c, what);
}

void caseClock() {
  const char* c = "clock";
  struct Case { const char* name; int fps; double hz, ppm, repeat; long num, den; };   // ai: the picture of refresh j is floor(j num / den)
  const Case cases[] = {
      {"60 painted on 60 Hz", 60, 60, 0, 0, 1, 1},
      {"30 on 60 Hz", 30, 60, 0, 0, 1, 2},
      {"60 on 59.94 Hz, the display 50 ppm fast", 60, 59.94, 50, 0, 1000, 999},
      {"60 on 60 Hz, a tenth of the pictures held a refresh longer", 60, 60, 0, 0.1, 1, 1},
  };
  for (const Case& t : cases) {
    sim::Sender s;
    s.blocks = 1;
    s.fps = t.fps;
    sim::Display d;
    d.hz = t.hz;
    d.ppm = t.ppm;
    d.phaseMs = 3.25;
    d.repeat = t.repeat;
    sim::Optics o;
    o.width = o.height = 64;
    auto sc = make(c, s, d, o, 1);
    if (!sc) return;
    // ai: the refreshes as the display's clock puts them, worked here apart from the scene: start j = t0 + phase + j P,
    // ai: each tested a whole ns clear of it (the scene's doubles and these long doubles part by 1e-3 ns at most here)
    const long double P = 1e9L / ((long double)t.hz * (1 + (long double)t.ppm * 1e-6L));
    const long double base = (long double)sc->startNs() + 3.25e6L;
    const long J = 200000;
    long misses = 0, edges = 0, twice = 0, skipped = 0;
    int64_t prev = -1, run = 0;
    for (long j = 0; j < J; j++) {
      const int64_t start = int64_t(std::ceil(base + (long double)j * P));
      const int64_t got = sc->pictureAt(start + int64_t(P / 2)), got0 = sc->pictureAt(start + 1), before = sc->pictureAt(start - 2);
      if (got != got0) misses++;
      if (!t.repeat) {
        if (got != j * t.num / t.den) misses++;
        if (j && before != (j - 1) * t.num / t.den) edges++;
      }
      if (got == prev) run++;
      else {
        if (prev >= 0 && t.repeat) { if (got != prev + 1) misses++; if (run + 1 == 2) twice++; else if (run + 1 != 1) misses++; }
        if (prev >= 0 && got > prev + 1) skipped += got - prev - 1;
        run = 0;
      }
      prev = got;
      // ai: the inverse: a picture's first refresh is where it begins
      if (j < 2000 && got && sc->pictureAt(sc->shownAt(got)) != got) edges++;
    }
    const long pictures = prev + 1;
    std::string what = fmt("%s: %ld refreshes of %.6f ms, %ld pictures", t.name, J, double(P * 1e-6L), pictures);
    bool good = !misses && !edges;
    if (t.num == 1000) { what += fmt(", %ld skipped (one in 999 refreshes: %ld)", skipped, J / 999); good = good && skipped == (J - 1) * 1000 / 999 - (J - 1); }
    if (t.repeat) {
      const double share = double(twice) / double(pictures);
      what += fmt(", %.2f%% shown twice", 100 * share);
      good = good && std::abs(share - t.repeat) < 0.01;
    }
    what += fmt("; %ld wrong, %ld at the edges", misses, edges);
    if (good) ok(c, what); else bad(c, what);
  }
}

void caseFile(liz_decoder* dec) {
  const char* c = "file";
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path path = fs::temp_directory_path(ec) / fmt("lizard-scene-%lld.bin", (long long)now());
  std::vector<uint8_t> data(200 * 1000);
  std::mt19937_64 rng(25102026);
  for (auto& b : data) b = uint8_t(rng());
  {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    if (!f) { bad(c, "cannot write " + path.u8string()); return; }
  }
  sim::Sender s;
  s.file = path.u8string();   // ai: UTF-8, as liz_tx_new_path takes it (string() is the ANSI code page's on Windows)
  const sim::Display d;
  const sim::Optics o;
  auto sc = make(c, s, d, o);
  if (!sc) { fs::remove(path, ec); return; }
  liz_store store{nullptr, 0};
  liz_rx* rx = liz_rx_new(&store);
  std::vector<uint8_t> img(size_t(o.width) * o.height), blocks;
  const double ph = bestPhase >= 0 ? bestPhase : 1e9 / s.fps / 3;
  int held = 0, captures = 0, read = 0, bad0 = 0;
  liz_progress pr{};
  const int64_t began = now(), until = began + 30'000'000'000LL;
  while (!pr.done && now() < until) {
    // ai: a capture as the camera makes it: the next picture at the phase, rendered once its window is over
    const int64_t k = nextPicture(*sc, 5);
    const int64_t t0 = sc->shownAt(k) + int64_t(ph);
    sleepTo(t0 + int64_t((o.exposureMs + o.readoutMs) * 1e6));
    sc->render(t0, img.data(), o.width);
    const Read r = readFrame(dec, img, int(o.width), int(o.height), 1, held, &blocks);
    captures++;
    read += r.blocks;
    liz_verdict v{};
    liz_rx_frame(rx, blocks.data(), r.blocks, &v);
    bad0 += v.bad + v.test;
    liz_rx_progress(rx, &pr);
  }
  const uint8_t* got = nullptr;
  size_t n = 0;
  const bool same = pr.done && liz_rx_data(rx, &got, &n) == LIZ_OK && n == data.size() && !std::memcmp(got, data.data(), n);
  const std::string name = liz_rx_meta(rx, LIZ_META_NAME);
  const std::string what = fmt("a %llu B file at size %d, the default optics at %.2f ms: %s after %d captures in %.1f s (%d blocks, %.1f a capture), "
                               "name %s, %s; %s",
                               (unsigned long long)data.size(), s.blocks, ph * 1e-6, pr.done ? "whole" : "NOT whole", captures, double(now() - began) * 1e-9, read,
                               captures ? double(read) / captures : 0.0, name.c_str(), same ? "its bytes the file's" : "its bytes NOT the file's",
                               sc->stats().c_str());
  if (same && name == path.filename().u8string() && !bad0) ok(c, what); else bad(c, what);
  liz_rx_free(rx);
  fs::remove(path, ec);
}

void caseParse() {
  const char* c = "parse";
  struct Try { const char* spec; bool good; };
  const Try tries[] = {
      {"blocks=128 ring=3 fps=255 codes=2 id=4294967295 hz=59.94 ppm=-50 phase=0 scan=1 response=0 sw=2560 sh=1440 share=0.5 "
       "surround=0 repeat=0.1 w=1280 h=720 fill=1.2 yaw=-30 tilt=20 roll=90 blur=0 exposure=0.1 shutter=global readout=0 "
       "gain=2 black=0 noise=0 shot=0 gamma=1 seed=9 file=/x", true},
      {"", true}, {"blocks=0", false}, {"blocks=129", false}, {"blocks=1.5", false}, {"ring=4", false}, {"fps=0", false},
      {"codes=3", false}, {"hz=0", false}, {"share=0", false}, {"shutter=sideways", false}, {"blur=-1", false},
      {"exposure=0", false}, {"exposure=1,5", false}, {"file=", false}, {"yaw=95", false}, {"gain", false}, {"wobble=1", false},
  };
  int wrong = 0;
  std::string line;
  for (const Try& t : tries) {
    sim::Sender s;
    sim::Display d;
    sim::Optics o;
    std::string why;
    const bool got = sim::parse(t.spec, s, d, o, why);
    if (got != t.good) { wrong++; line += fmt(" \"%s\" %s;", t.spec, got ? "taken" : ("refused: " + why).c_str()); }
  }
  // ai: the numbers read whatever the process's locale (the JVM sets LC_NUMERIC to the user's)
  const char* was = std::setlocale(LC_NUMERIC, nullptr);
  const std::string keep = was ? was : "C";
  const bool comma = std::setlocale(LC_NUMERIC, "de_DE.UTF-8") || std::setlocale(LC_NUMERIC, "en_DK.UTF-8") ||
                     std::setlocale(LC_NUMERIC, "fr_FR.UTF-8");
  sim::Sender s;
  sim::Display d;
  sim::Optics o;
  std::string why;
  const bool dotted = sim::parse("share=0.75 exposure=2.5", s, d, o, why) && d.share == 0.75 && o.exposureMs == 2.5;
  std::setlocale(LC_NUMERIC, keep.c_str());
  const std::string what = fmt("%d specs, %d judged wrong;%s 0.75 read as 0.75 under %s", int(sizeof tries / sizeof tries[0]), wrong,
                               line.c_str(), comma ? "a decimal-comma locale" : "the C locale (no decimal-comma locale installed)");
  if (!wrong && dotted) ok(c, what); else bad(c, what);
}

// ai: ---- the simulated camera ----

// ai: what a sink saw: each frame's time, delayed mark and arrival, and the first frames' luma
struct Seen {
  std::mutex mu;
  std::vector<int64_t> ns, at;
  std::vector<char> delayed, triggered;
  std::vector<std::vector<uint8_t>> luma;
  std::vector<int64_t> lumaNs;
  uint32_t w = 0, h = 0;
  size_t keep = 0;
  cam::Camera::Sink sink() {
    return [this](const cam::Frame& f) {
      const int64_t t = now();
      std::lock_guard<std::mutex> l(mu);
      ns.push_back(f.ns);
      at.push_back(t);
      delayed.push_back(f.delayed);
      triggered.push_back(f.triggered);
      w = f.width;
      h = f.height;
      if (luma.size() < keep) {
        std::vector<uint8_t> v(size_t(f.width) * f.height);
        for (uint32_t y = 0; y < f.height; y++) std::memcpy(v.data() + size_t(y) * f.width, f.luma + size_t(y) * f.stride, f.width);
        luma.push_back(std::move(v));
        lumaNs.push_back(f.ns);
      }
    };
  }
};

// ai: The worst wall ms this machine takes to render one capture of a scene (the simulated camera's own work a capture:
// ai: sim/scene.h render with 3 painters, as camera/sim.cpp makes the scene), over 0.6 s of captures on a grid of
// ai: period ns, as check_session's Room measures it; -1 with why where the scene cannot be made. The camera drops a
// ai: triggered capture whose hand-over fell more than a period behind, so where renders take over half the period
// ai: (this machine slow or busy) captures go missing and their triggers count as lost.
double worstRender(const std::string& scene, int64_t period, std::string& why) {
  sim::Sender s;
  sim::Display d;
  sim::Optics o;
  if (!sim::parse(scene, s, d, o, why)) return -1;
  const auto sc = sim::Scene::create(s, d, o, 3, why);
  if (!sc) return -1;
  std::vector<uint8_t> px(size_t(o.width) * o.height);
  const int64_t window = int64_t((o.exposureMs + (o.rolling ? o.readoutMs : 0)) * 1e6);
  int64_t t = now();
  sc->render(t - window, px.data(), o.width);   // ai: its pool's first touch, outside the figure
  double worstMs = 0;
  for (const int64_t end = t + 600'000'000; t < end;) {
    t += period;
    const int64_t n = now();
    if (t < n) t += ((n - t) / period + 1) * period;
    cam::sleepUntil(t);
    const int64_t a = now();
    sc->render(t - window, px.data(), o.width);
    worstMs = std::max(worstMs, double(now() - a) * 1e-6);
  }
  return worstMs;
}

std::unique_ptr<cam::Camera> openSim(const char* c, const std::string& spec) {
  setEnv("LIZ_CAMERA_SIM", spec.c_str());
  const auto listed = cam::simList(nullptr);
  if (listed.size() != 1 || listed[0].id != "sim:0" || listed[0].detail != spec) { bad(c, "simList under LIZ_CAMERA_SIM: not the one camera"); return nullptr; }
  std::string why;
  auto cam = cam::simOpen("sim:0", why);
  if (!cam) bad(c, "simOpen: " + why);
  return cam;
}

void caseCamera(liz_decoder* dec) {
  const char* c = "camera";
  // ai: a key neither the scene nor the camera knows is refused; unset, the camera is not listed
  {
    setEnv("LIZ_CAMERA_SIM", "camfps=30 frobnicate=2");
    std::string why;
    const bool refused = !cam::simOpen("sim:0", why) && why.find("frobnicate") != std::string::npos;
    setEnv("LIZ_CAMERA_SIM", nullptr);
    const bool unlisted = cam::simList(nullptr).empty();
    if (refused && unlisted) ok(c, "an unknown key refused (" + why + "); unset, no camera listed");
    else bad(c, fmt("an unknown key %s, unset %s", refused ? "refused" : "NOT refused", unlisted ? "unlisted" : "STILL listed"));
  }
  // ai: each format for 2 s at the half size: frames counted, their times a whole number of the camera's periods
  // ai: apart, their luma against the same capture rendered by a scene of the same spec: one picture a second, no
  // ai: noise, a global shutter, so a capture away from a picture's edges is the same in both (each scene numbers its
  // ai: pictures from its own start, a few ms apart). MJPEG's frames go through TurboJPEG both ways (camera/sim.cpp):
  // ai: where none loads on a system whose receiver does without it, that format skips saying why (checks.h)
  const std::string still = "blocks=8 fps=1 hz=60 phase=0 scan=0 response=0 shutter=global exposure=2 noise=0 shot=0 id=777";
  for (const char* form : {"grey", "yuyv", "nv12", "mjpeg"}) {
    if (!std::strcmp(form, "mjpeg")) {
      const std::string optional = jpegOptional();
      if (!optional.empty()) {
        std::printf("skip camera: mjpeg not run: %s\n", optional.c_str());
        std::fflush(stdout);
        continue;
      }
    }
    const std::string spec = still + " camfps=60 format=" + form;
    auto camera = openSim(c, spec);
    if (!camera) continue;
    const auto modes = camera->modes();
    if (modes.size() != 2 || modes[1].width * 2 != modes[0].width || modes[0].fps != 60) { bad(c, std::string(form) + ": modes"); continue; }
    const cam::Mode mode = modes[1];
    Seen seen;
    seen.keep = 12;
    std::string why;
    sim::Sender s;
    sim::Display d;
    sim::Optics o;
    sim::parse(still, s, d, o, why);
    o.width = mode.width;
    o.height = mode.height;
    if (!camera->start(mode, seen.sink(), [](const std::string&) {}, why)) { bad(c, std::string(form) + ": start: " + why); continue; }
    auto ref = sim::Scene::create(s, d, o, 1, why);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    camera->stop();
    const cam::Stats st = camera->stats();
    if (!ref) { bad(c, "the reference scene: " + why); continue; }
    std::lock_guard<std::mutex> l(seen.mu);
    const double period = 1e9 / 60;
    long offGrid = 0, gaps = 0;
    for (size_t i = 1; i < seen.ns.size(); i++) {
      const double q = double(seen.ns[i] - seen.ns[i - 1]) / period;
      if (std::abs(q - std::round(q)) > 1e-6 * q || q < 0.5) offGrid++;
      gaps += long(std::lround(q)) - 1;
    }
    // ai: the luma against the reference: exact where the format is lossless, MJPEG by its PSNR
    std::vector<uint8_t> want(size_t(o.width) * o.height);
    double worstDb = 1e9;
    long differ = 0, compared = 0;
    int blocksRead = -1;
    for (size_t i = 0; i < seen.luma.size(); i++) {
      const int64_t t = seen.lumaNs[i];
      if (ref->pictureAt(t - 100'000'000) != ref->pictureAt(t + 100'000'000)) continue;   // ai: near a picture's edge
      ref->render(t, want.data(), o.width, 2.0);
      const auto& got = seen.luma[i];
      double se = 0;
      for (size_t p = 0; p < want.size(); p++) {
        const int e = int(got[p]) - int(want[p]);
        se += double(e) * e;
        differ += e != 0;
      }
      const double mse = se / double(want.size());
      worstDb = std::min(worstDb, mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99.0);
      compared++;
      if (blocksRead < 0) {
        int held = 0;
        blocksRead = readFrame(dec, got, int(o.width), int(o.height), 1, held).blocks;
      }
    }
    const bool lossy = !std::strcmp(form, "mjpeg");
    const bool good = seen.ns.size() >= 100 && st.frames == seen.ns.size() && !st.corrupt && !offGrid && compared >= 6 &&
                      (lossy ? worstDb >= 30 : differ == 0) && blocksRead == liz_blocks_for(s.blocks);
    const std::string what = fmt("%s at %u x %u: %d frames in 2 s (%llu dropped, %llu corrupt, %ld periods skipped), %ld times off the 60 a "
                                 "second grid, %ld captures against the reference %s, %d of %d blocks read from one, convert %.3f ms",
                                 form, mode.width, mode.height, int(seen.ns.size()), (unsigned long long)st.dropped, (unsigned long long)st.corrupt,
                                 gaps, offGrid, compared,
                                 lossy ? fmt("PSNR %.1f dB at worst", worstDb).c_str() : fmt("%ld pixels differ", differ).c_str(),
                                 blocksRead, liz_blocks_for(s.blocks), st.convertMs);
    if (good) ok(c, what); else bad(c, what);
  }
  // ai: the trigger: free run, then a schedule of 16.65 ms (a camera slightly faster than 60 Hz, the lock's), a delay of
  // ai: 3 ms within it, then end() back to free run. Triggered frames carry their triggers' deadlines: spaced by a
  // ai: whole number of periods exactly, whatever the trigger thread's lateness and the exposures' jitter (1 ms here,
  // ai: so a frame stamped by its exposure, or by the moment its trigger went out, would show it); the delayed
  // ai: trigger's frame alone marked, its interval the periods and the delay. Judged against this machine (2026-10-25, as
  // ai: check_session's trigger case): where it renders the scene with room (every render within half the period,
  // ai: worstRender) as before, no trigger lost and every period past one in an interval a deadline the clock skipped
  // ai: (its stats' missed: its thread held off the CPU past a period), so every interval one period where none was;
  // ai: else the captures the camera dropped show as intervals of whole periods too (two busy loops on each of the
  // ai: check's two CPUs lost 1 to 3 triggers of 84). The intervals judged are between two triggered frames
  // ai: (Frame::triggered), so a capture matched after end(), handed over free-running with its exposure's time, is
  // ai: none of them.
  {
    const int64_t P = 16'650'000;
    const std::string scene = "blocks=8 w=960 h=540";
    std::string rwhy;
    const double renderMs = worstRender(scene, P, rwhy);
    if (renderMs < 0) { bad(c, "trigger: the scene's render not measured: " + rwhy); return; }
    const bool roomy = renderMs <= double(P) * 1e-6 / 2;
    auto camera = openSim(c, scene + " trigger=1 latency=0.05 jitter=1 camfps=60");
    if (!camera) return;
    cam::Trigger* trig = camera->trigger();
    if (!trig) { bad(c, "trigger=1 but no trigger"); return; }
    Seen seen;
    std::string why;
    const auto modes = camera->modes();
    if (!camera->start(modes[0], seen.sink(), [](const std::string&) {}, why)) { bad(c, "trigger: start: " + why); return; }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (!trig->begin(P, why)) { bad(c, "trigger: begin: " + why); camera->stop(); return; }
    const int64_t begun = now();
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    trig->delay(3'000'000);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    const std::string ts = trig->stats();
    trig->end();
    const int64_t ended = now();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    camera->stop();
    std::lock_guard<std::mutex> l(seen.mu);
    // ai: the intervals between two triggered frames, from 100 ms after begin; and the free run's after end()
    const int64_t D = 3'000'000;
    long n = 0, offPeriod = 0, marked = 0, after = 0, longer = 0, extra = 0;
    double markedGap = 0, worst = 0;
    for (size_t i = 1; i < seen.ns.size(); i++) {
      if (!seen.triggered[i - 1] || !seen.triggered[i] || seen.at[i - 1] < begun + 100'000'000) {
        if (seen.at[i - 1] > ended + 100'000'000) after++;
        continue;
      }
      const int64_t gap = seen.ns[i] - seen.ns[i - 1], plain = gap - (seen.delayed[i] ? D : 0);
      const int64_t k = std::max<int64_t>(1, std::llround(double(plain) / double(P)));   // ai: its periods
      const double off = std::abs(double(plain - k * P)) * 1e-3;
      n++;
      if (k > 1) longer++;
      extra += long(k - 1);
      if (seen.delayed[i]) { marked++; markedGap = double(gap) * 1e-6; }
      worst = std::max(worst, off);
      if (plain != k * P) offPeriod++;
    }
    long markedAll = 0;
    for (char dl : seen.delayed) markedAll += dl;
    const bool lost0 = ts.find("\"lost\":0,") != std::string::npos;
    const size_t mi = ts.find("\"missed\":");
    const long missed = mi == std::string::npos ? -1 : std::atol(ts.c_str() + mi + 9);
    // ai: enough to judge: with room, the 78 the schedule gives in its 1.3 s and the 18 of free run in 0.3 s nearly
    // ai: all; without, half of what this machine renders in those times
    const double each = std::max(double(P) * 1e-6, renderMs);
    const long least = roomy ? 61 : std::max(5L, long(1300 / each / 2)), leastAfter = roomy ? 11 : std::max(1L, long(300 / each / 2));
    const bool good = n >= least && !offPeriod && marked == 1 && markedAll == 1 && after >= leastAfter &&
                      missed >= 0 && (!roomy || (extra <= missed && lost0));
    const std::string what = fmt("trigger at 16.65 ms, each exposure's latency spread 1 ms (sd): %ld triggered intervals, "
                                 "%ld off a whole number of periods (the worst by %.3f us), %ld of more than one (%ld "
                                 "periods past one, %ld deadlines the clock skipped), %ld frame%s delayed (its interval "
                                 "%.3f ms), %ld free-run frames after end(); the scene rendered in up to %.1f ms here, %s; "
                                 "the trigger's %s",
                                 n, offPeriod, worst, longer, extra, missed, marked, marked == 1 ? "" : "s", markedGap, after, renderMs,
                                 roomy ? "within half the period: none lost, every period past one a skipped deadline"
                                       : "past half the period: the captures the camera dropped allowed",
                                 ts.c_str());
    if (good) ok(c, what); else bad(c, what);
  }
  setEnv("LIZ_CAMERA_SIM", nullptr);
}

}  // namespace

int scene(const Args& a) {
  const auto want = [&](const char* c) { return a.empty() || std::find(a.begin(), a.end(), c) != a.end(); };
  for (const auto& x : a)
    if (x != "parse" && x != "clean" && x != "straddle" && x != "sweep" && x != "pair" && x != "speed" && x != "clock" && x != "file" &&
        x != "camera") {
      std::fprintf(stderr, "cam_check scene [parse|clean|straddle|sweep|pair|speed|clock|file|camera ...]\n");
      return 2;
    }
  failed = 0;
  liz_decoder* dec = liz_decoder_new(0);
  if (!dec) { bad("decoder", liz_last_error()); return 1; }
  if (want("parse")) caseParse();
  if (want("clean")) caseClean(dec);
  if (want("straddle")) caseStraddle(dec);
  if (want("sweep")) caseSweep(dec);
  if (want("pair")) casePair(dec);
  if (want("speed")) caseSpeed();
  if (want("clock")) caseClock();
  if (want("file")) caseFile(dec);
  if (want("camera")) caseCamera(dec);
  liz_decoder_free(dec);
  return failed ? 1 : 0;
}

}  // namespace lizard::check
