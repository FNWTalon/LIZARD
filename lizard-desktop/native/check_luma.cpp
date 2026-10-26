// ai: cam_check luma (2026-10-25), checks.h's form: every Pix but MJPEG against a reference written out here from
// ai: each layout's definition (the 2 x 2 window with its edges clamped for Bayer, never luma.cpp's helpers), at odd
// ai: widths where the layout has them and at strides with padding; short frames, short strides and unknown layouts
// ai: refused with why; camera.h pick's rule on mode lists as cameras give them; a 1920 x 1080 frame timed a layout,
// ai: judged against that layout's definition timed beside it (lumaTiming). MJPEG through TurboJPEG: a grey picture
// ai: (random 8 x 8 blocks of grey and two gradients) at quality 85 with and without its Huffman tables, decoded back
// ai: (PSNR over 35 dB, the cut frame decoding as the whole one); a frame cut short; cuts all through the header (each
// ai: refused, each after a whole frame decoded on the same handle, and the frames after them still decoding) and in
// ai: the scan (each taken with a warning); bytes that are no JPEG, an empty frame, and the library named by
// ai: LIZ_TURBOJPEG absent. Where no TurboJPEG loads, the cases after that one fail on Linux and skip with why on
// ai: Windows and macOS, whose receivers do without it (checks.h jpegOptional). The clock (camera/clock.h): sleepUntil
// ai: over 200 targets 0.5 to 20 ms ahead, never early and no later than the system's own precise sleep measured beside
// ai: it, which is judged against what this machine measured in the same run, never a fixed figure. No arguments.
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <ctime>
#if defined(__linux__)
#include <cerrno>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#include "camera/camera.h"
#include "camera/jpeg.h"
#include "camera/luma.h"
#include "camera/clock.h"
#include "checks.h"

namespace lizard::check {

namespace {

using namespace lizard::cam;

struct Report {
  int failed = 0;
  void ok(const std::string& c, const std::string& s) { std::printf("ok %s: %s\n", c.c_str(), s.c_str()); std::fflush(stdout); }
  void skip(const std::string& c, const std::string& s) { std::printf("skip %s: %s\n", c.c_str(), s.c_str()); std::fflush(stdout); }
  void fail(const std::string& c, const std::string& s) {
    std::printf("FAIL %s: %s\n", c.c_str(), s.c_str());
    std::fflush(stdout);
    failed++;
  }
  void say(bool good, const std::string& c, const std::string& s) { good ? ok(c, s) : fail(c, s); }
  int code() const { return failed ? kFailed : kHeld; }
};

#if defined(__MINGW32__)
std::string fmt(const char* f, ...) __attribute__((format(gnu_printf, 1, 2)));
#else
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
  char b[1024];
  va_list a;
  va_start(a, f);
  std::vsnprintf(b, sizeof b, f, a);
  va_end(a);
  return b;
}

// ai: --- luma ---

const Pix kLayouts[] = {Pix::Grey8, Pix::Grey16, Pix::YUYV, Pix::UYVY, Pix::YVYU, Pix::VYUY, Pix::NV12, Pix::NV21,
                        Pix::I420, Pix::YV12, Pix::Bayer8, Pix::RGB24, Pix::BGR24, Pix::BGRA32};

// ai: the bytes of a row of the layout's first plane, from its definition
size_t refRow(Pix p, uint32_t w) {
  switch (p) {
    case Pix::Grey8: case Pix::NV12: case Pix::NV21: case Pix::I420: case Pix::YV12: case Pix::Bayer8: return w;
    case Pix::Grey16: return 2 * size_t(w);
    case Pix::YUYV: case Pix::UYVY: case Pix::YVYU: case Pix::VYUY: return 2 * size_t(w);   // ai: 4 bytes a pair
    case Pix::RGB24: case Pix::BGR24: return 3 * size_t(w);
    case Pix::BGRA32: return 4 * size_t(w);
    default: return 0;
  }
}

bool evenOnly(Pix p) { return p == Pix::YUYV || p == Pix::UYVY || p == Pix::YVYU || p == Pix::VYUY; }

// ai: each layout's luma, a pixel at a time, from its definition
std::vector<uint8_t> reference(Pix p, unsigned bits, const std::vector<uint8_t>& f, uint32_t w, uint32_t h,
                               uint32_t stride) {
  std::vector<uint8_t> o(size_t(w) * h);
  auto byte = [&](uint32_t y, size_t i) -> unsigned { return f[size_t(y) * stride + i]; };
  auto sample16 = [&](uint32_t x, uint32_t y) -> unsigned {
    const unsigned v = byte(y, 2 * size_t(x)) + 256u * byte(y, 2 * size_t(x) + 1);
    const unsigned t = v / (1u << (bits - 8));
    return t > 255 ? 255 : t;
  };
  auto mean4 = [](unsigned a, unsigned b, unsigned c, unsigned d) { return (a + b + c + d + 2) / 4; };
  auto lum = [](unsigned r, unsigned g, unsigned b) { return (77 * r + 150 * g + 29 * b + 128) / 256; };
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      const uint32_t x1 = std::min(x + 1, w - 1), y1 = std::min(y + 1, h - 1);
      unsigned v = 0;
      switch (p) {
        case Pix::Grey8: case Pix::NV12: case Pix::NV21: case Pix::I420: case Pix::YV12: v = byte(y, x); break;
        case Pix::Grey16: v = sample16(x, y); break;
        case Pix::YUYV: case Pix::YVYU: v = byte(y, 2 * size_t(x)); break;
        case Pix::UYVY: case Pix::VYUY: v = byte(y, 2 * size_t(x) + 1); break;
        case Pix::Bayer8: v = mean4(byte(y, x), byte(y, x1), byte(y1, x), byte(y1, x1)); break;
        case Pix::RGB24: v = lum(byte(y, 3 * size_t(x)), byte(y, 3 * size_t(x) + 1), byte(y, 3 * size_t(x) + 2)); break;
        case Pix::BGR24: v = lum(byte(y, 3 * size_t(x) + 2), byte(y, 3 * size_t(x) + 1), byte(y, 3 * size_t(x))); break;
        case Pix::BGRA32: v = lum(byte(y, 4 * size_t(x) + 2), byte(y, 4 * size_t(x) + 1), byte(y, 4 * size_t(x))); break;
        default: break;
      }
      o[size_t(y) * w + x] = uint8_t(v);
    }
  return o;
}

bool is16(Pix p) { return p == Pix::Grey16; }

void lumaLayouts(Report& r) {
  std::mt19937 rng(20261025);
  const std::pair<uint32_t, uint32_t> sizes[] = {{1, 1}, {2, 1}, {3, 2}, {5, 3}, {7, 5}, {8, 4}, {13, 7}, {64, 3},
                                                 {333, 9}, {1920, 2}};
  for (Pix p : kLayouts) {
    const char* name = pixName(p);
    int cases = 0, wrong = 0;
    std::string first;
    for (auto [w, h] : sizes) {
      if (evenOnly(p) && (w & 1)) w++;   // ai: a 4:2:2 pair is two pixels: no odd width in the layout
      for (unsigned bits : is16(p) ? std::vector<unsigned>{8, 10, 12, 14, 16} : std::vector<unsigned>{8}) {
        for (int pad : {0, 1 + int(rng() % 37)}) {
          const size_t row = refRow(p, w);
          const uint32_t stride = uint32_t(row + size_t(pad));
          const size_t bytes = size_t(h - 1) * stride + row;
          std::vector<uint8_t> f(bytes);
          for (auto& b : f) b = uint8_t(rng());
          std::vector<uint8_t> out;
          std::string why;
          cases++;
          if (frameBytes(p, w, h, stride) != bytes) {
            wrong++;
            if (first.empty()) first = fmt("frameBytes %zu where %zu at %u x %u stride %u", frameBytes(p, w, h, stride), bytes, w, h, stride);
            continue;
          }
          if (!toLuma(p, uint8_t(bits), f.data(), f.size(), w, h, stride, out, &why)) {
            wrong++;
            if (first.empty()) first = fmt("%u x %u stride %u refused: %s", w, h, stride, why.c_str());
            continue;
          }
          const auto ref = reference(p, bits, f, w, h, stride);
          if (out != ref) {
            size_t i = 0;
            while (i < ref.size() && i < out.size() && out[i] == ref[i]) i++;
            wrong++;
            if (first.empty())
              first = fmt("%u x %u stride %u bits %u: pixel (%zu, %zu) %d where %d", w, h, stride, bits, i % w, i / w,
                          i < out.size() ? out[i] : -1, i < ref.size() ? ref[i] : -1);
          }
        }
      }
    }
    r.say(wrong == 0, std::string("layout ") + name,
          wrong ? fmt("%d of %d cases wrong, first %s", wrong, cases, first.c_str())
                : fmt("%d cases (widths 1 to 1920%s, padded and tight strides%s) equal to the reference", cases,
                      evenOnly(p) ? ", even" : "", is16(p) ? ", bits 8 to 16 over full 16-bit samples" : ""));
  }
}

void lumaRefusals(Report& r) {
  std::vector<uint8_t> out;
  std::string why;
  int bad = 0, cases = 0;
  std::string first, example;
  auto expect = [&](bool refused, const std::string& what) {
    cases++;
    if (!refused || why.empty()) { bad++; if (first.empty()) first = what + (refused ? ": no why" : ": taken"); }
    else if (example.empty() && what.rfind("short", 0) == 0) example = why;
    why.clear();
  };
  for (Pix p : kLayouts) {
    const uint32_t w = evenOnly(p) ? 6 : 5, h = 3, stride = uint32_t(refRow(p, w)) + 2;
    const size_t need = size_t(h - 1) * stride + refRow(p, w);
    std::vector<uint8_t> f(need, 128);
    const uint8_t bits = is16(p) ? 12 : 8;
    expect(!toLuma(p, bits, f.data(), need - 1, w, h, stride, out, &why), std::string("short ") + pixName(p));
    if (refRow(p, w) > 1) expect(!toLuma(p, bits, f.data(), need, w, h, uint32_t(refRow(p, w) - 1), out, &why),
                                 std::string("stride under a row, ") + pixName(p));
    expect(!toLuma(p, bits, nullptr, need, w, h, stride, out, &why), std::string("no data, ") + pixName(p));
    expect(!toLuma(p, bits, f.data(), need, 0, h, stride, out, &why), std::string("no width, ") + pixName(p));
  }
  std::vector<uint8_t> f(64, 0);
  expect(!toLuma(Pix::Unknown, 8, f.data(), f.size(), 4, 4, 4, out, &why), "Unknown");
  expect(!toLuma(Pix::MJPEG, 8, f.data(), f.size(), 4, 4, 4, out, &why), "MJPEG");
  expect(!toLuma(Pix::Grey16, 7, f.data(), f.size(), 4, 4, 8, out, &why), "Grey16 at 7 bits");
  expect(!toLuma(Pix::Grey16, 17, f.data(), f.size(), 4, 4, 8, out, &why), "Grey16 at 17 bits");
  r.say(bad == 0, "refusals", bad ? fmt("%d of %d not refused with why, first %s", bad, cases, first.c_str())
                                  : fmt("%d frames refused with why, as \"%s\"", cases, example.c_str()));
  const bool zero = frameBytes(Pix::MJPEG, 640, 480, 640) == 0 && frameBytes(Pix::Unknown, 640, 480, 640) == 0 &&
                    frameBytes(Pix::Grey8, 640, 0, 640) == 0;
  r.say(zero, "frameBytes", zero ? "0 for MJPEG, Unknown and an empty frame; (h - 1) * stride + a row elsewhere (the layouts' cases)"
                                 : "not 0 for MJPEG, Unknown or an empty frame");
  int wrongPlace = 0;
  for (Pix p : kLayouts) {
    const bool want = p == Pix::Grey8 || p == Pix::NV12 || p == Pix::NV21 || p == Pix::I420 || p == Pix::YV12;
    if (lumaInPlace(p) != want) wrongPlace++;
  }
  if (lumaInPlace(Pix::MJPEG) || lumaInPlace(Pix::Unknown)) wrongPlace++;
  r.say(wrongPlace == 0, "in place", wrongPlace ? fmt("%d layouts wrong", wrongPlace)
                                                : "Grey8, NV12, NV21, I420 and YV12 only");
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

// ai: This thread's CPU time in the system's own units (ns where it counts so, CPU cycles on Windows: GetThreadTimes
// ai: moves in 15.6 ms ticks): a share of the CPU the machine gives others is not in it.
uint64_t cpuNow() {
#if defined(_WIN32)
  ULONG64 c = 0;
  QueryThreadCycleTime(GetCurrentThread(), &c);
  return c;
#else
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return uint64_t(ts.tv_sec) * 1'000'000'000u + uint64_t(ts.tv_nsec);
#endif
}

// ai: A 1920 x 1080 frame timed a layout and judged against this machine (2026-10-25: a fixed 2 ms bound, RGB24's
// ai: scalar loop 0.97 ms on a 5.7 GHz core, stood near it on CI's slower ones): each layout's conversion against the
// ai: same frame through reference() (the layout's definition, a pixel at a time), nine pairs taken in turn, each by
// ai: this thread's CPU time (wall time moved with the machine's other work: two busy loops on each of the check's two
// ai: CPUs once put a conversion at 4 times its definition's wall time). A conversion slower than twice its plain
// ai: definition has gone wrong whatever the machine, as has one whose quickest wall time is past a 60 fps camera's
// ai: frame (16.7 ms: no machine it runs on keeps that camera's rate). The times themselves are figures.
void lumaTiming(Report& r) {
  const uint32_t w = 1920, h = 1080;
  std::mt19937 rng(3);
  std::vector<uint8_t> f(size_t(w) * h * 4 + 64);
  for (auto& b : f) b = uint8_t(rng());
  std::vector<uint8_t> out(size_t(w) * h);
  std::string line, over;
  double worst = 0, worstRatio = 0;
  const char* worstName = "";
  const char* ratioName = "";
  volatile unsigned sink = 0;   // ai: the definition's output read, so its work is not optimised away
  for (Pix p : kLayouts) {
    const uint32_t stride = uint32_t(refRow(p, w));
    std::vector<double> wall, cpu, defCpu;
    std::string why;
    bool refused = false;
    for (int i = 0; i < 11 && !refused; i++) {
      const int64_t t0 = steadyNs();
      const uint64_t c0 = cpuNow();
      refused = !toLuma(p, 12, f.data(), f.size(), w, h, stride, out, &why);
      const uint64_t c1 = cpuNow();
      const int64_t t1 = steadyNs();
      const std::vector<uint8_t> def = reference(p, 12, f, w, h, stride);
      const uint64_t c2 = cpuNow();
      sink = sink + def[size_t(i) * 7919 % def.size()];
      if (i >= 2) {
        wall.push_back(double(t1 - t0) / 1e6);
        cpu.push_back(double(c1 - c0));
        defCpu.push_back(double(c2 - c1));
      }
    }
    if (refused) { r.fail(std::string("time ") + pixName(p), "refused: " + why); continue; }
    const double m = *std::min_element(wall.begin(), wall.end()), d = median(defCpu);
    const double ratio = d > 0 ? median(cpu) / d : 0;
    if (m > worst) { worst = m; worstName = pixName(p); }
    if (ratio > worstRatio) { worstRatio = ratio; ratioName = pixName(p); }
    if (ratio > 2 || m > 1e3 / 60) over += fmt("%s%s %.3f ms, %.2f of its definition", over.empty() ? "" : ", ", pixName(p), m, ratio);
    line += fmt("%s%s %.3f (%.2f)", line.empty() ? "" : ", ", pixName(p), m, ratio);
  }
  r.say(over.empty(), "time",
        fmt("a 1920 x 1080 frame's quickest wall ms of nine, and (its CPU time's share of the layout's definition's "
            "beside it): %s; the slowest %s %.3f ms, the nearest its definition %s at %.2f (twice it, or a 60 fps "
            "frame's 16.7 ms, fails)%s%s",
            line.c_str(), worstName, worst, ratioName, worstRatio, over.empty() ? "" : "; OVER: ", over.c_str()));
}

// ai: camera.h pick's rule on mode lists as cameras give them
void lumaPick(Report& r) {
  auto m = [](uint32_t w, uint32_t h, double fps, bool jpeg, double lo = 0) {
    Mode x;
    x.width = w;
    x.height = h;
    x.fps = fps;
    x.fpsMin = lo > 0 ? lo : fps;
    x.compressed = jpeg;
    x.pix = jpeg ? Pix::MJPEG : Pix::YUYV;
    return x;
  };
  struct Case { const char* what; std::vector<Mode> modes; uint32_t w, h; double fps; int want; };
  const Case cases[] = {
      {"a webcam's 1080p: MJPEG at 30 over YUYV at 5",
       {m(1920, 1080, 30, true), m(1920, 1080, 5, false), m(640, 480, 30, false), m(1280, 720, 30, true)}, 1920, 1080, 60, 0},
      {"the same rate: uncompressed first", {m(1920, 1080, 60, true), m(1920, 1080, 60, false)}, 1920, 1080, 60, 1},
      {"the fastest up to the want", {m(1920, 1080, 120, false), m(1920, 1080, 30, false)}, 1920, 1080, 60, 1},
      {"all over the want: the slowest", {m(1920, 1080, 120, false), m(1920, 1080, 90, false)}, 1920, 1080, 60, 1},
      {"a range holding the want", {m(1920, 1080, 30, false), m(1920, 1080, 163, false, 1)}, 1920, 1080, 60, 1},
      {"no 1080p: the nearest by pixels", {m(1600, 1200, 60, false), m(2048, 1080, 60, false), m(640, 480, 60, false)},
       1920, 1080, 60, 1},
      {"no size wanted: the largest of the fastest", {m(640, 480, 60, false), m(1920, 1080, 30, false), m(1280, 720, 60, false)},
       0, 0, 60, 2},
      {"rates within 0.1% equal", {m(1920, 1080, 30, true), m(1920, 1080, 29.98, false)}, 1920, 1080, 60, 1},
      {"60.05 counted at 60", {m(1920, 1080, 50, false), m(1920, 1080, 60.05, false)}, 1920, 1080, 60, 1},
      {"no modes", {}, 1920, 1080, 60, -1},
  };
  int bad = 0;
  std::string first;
  for (const auto& c : cases) {
    const int got = pick(c.modes, c.w, c.h, c.fps);
    if (got != c.want) { bad++; if (first.empty()) first = fmt("%s: %d where %d", c.what, got, c.want); }
  }
  r.say(bad == 0, "pick", bad ? fmt("%d of %zu wrong, first %s", bad, std::size(cases), first.c_str())
                              : fmt("%zu mode lists picked by the rule (a webcam's 1080p MJPEG at 30, a range at the "
                                    "want, the nearest size, ties to uncompressed)", std::size(cases)));
}

// ai: --- MJPEG ---

double psnr(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 0;
  double se = 0;
  for (size_t i = 0; i < a.size(); i++) { const double e = double(a[i]) - b[i]; se += e * e; }
  return se == 0 ? 99 : 10 * std::log10(255.0 * 255.0 / (se / double(a.size())));
}

// ai: where the scan begins: past the SOS segment; 0 where there is none
size_t scanStart(const std::vector<uint8_t>& j) {
  size_t i = 2;
  while (i + 4 <= j.size() && j[i] == 0xFF) {
    if (j[i + 1] == 0xFF) { i++; continue; }
    const size_t len = size_t(j[i + 2]) << 8 | j[i + 3];
    if (j[i + 1] == 0xDA) return i + 2 + len;
    i += 2 + len;
  }
  return 0;
}

// ai: the DHT segments before SOS, -1 where there is no SOS
int tablesBeforeScan(const std::vector<uint8_t>& j) {
  int dht = 0;
  size_t i = 2;
  while (i + 4 <= j.size() && j[i] == 0xFF) {
    const uint8_t mk = j[i + 1];
    if (mk == 0xFF) { i++; continue; }
    if (mk == 0xDA) return dht;
    if (mk == 0xC4) dht++;
    i += 2 + (size_t(j[i + 2]) << 8 | j[i + 3]);
  }
  return -1;
}

std::vector<uint8_t> picture(uint32_t w, uint32_t h, uint32_t seed) {
  std::mt19937 rng(seed);
  const uint32_t bw = (w + 7) / 8, bh = (h + 7) / 8;
  std::vector<uint8_t> level(size_t(bw) * bh);
  for (auto& l : level) l = uint8_t(rng());
  std::vector<uint8_t> g(size_t(w) * h);
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      const double v = 0.6 * level[size_t(y / 8) * bw + x / 8] + 0.25 * (255.0 * x / w) + 0.15 * (255.0 * y / h);
      g[size_t(y) * w + x] = uint8_t(std::lround(std::min(255.0, v)));
    }
  return g;
}

void setLib(const char* path) {
#if defined(_WIN32)
  std::wstring s = L"LIZ_TURBOJPEG=";
  if (path) for (const char* c = path; *c; c++) s += wchar_t(*c);
  _wputenv(s.c_str());
#else
  if (path) setenv("LIZ_TURBOJPEG", path, 1);
  else unsetenv("LIZ_TURBOJPEG");
#endif
}

void mjpeg(Report& r) {
  std::string why;
  {
    // ai: the library named by LIZ_TURBOJPEG absent (the one tried where set); the value before put back
    const char* before = std::getenv("LIZ_TURBOJPEG");
    const std::string kept = before ? before : "";
#if defined(_WIN32)
    const char* none = "C:\\nonexistent\\turbojpeg.dll";
#else
    const char* none = "/nonexistent";
#endif
    setLib(none);
    auto j = Jpeg::load(why);
    const bool said = !j && why.find("nonexistent") != std::string::npos;
    r.say(said, "mjpeg no library", j ? "loaded where LIZ_TURBOJPEG names no file" : "load says: " + why);
    setLib(before ? kept.c_str() : nullptr);
    why.clear();
  }
  auto j = Jpeg::load(why);
  if (!j) {
    const std::string optional = jpegOptional();
    if (!optional.empty()) r.skip("mjpeg", "load, 1280x720, cut short, header cut, not a jpeg, empty, 643x481 and time MJPEG not run: " + optional);
    else r.fail("mjpeg load", why);
    return;
  }
  r.ok("mjpeg load", "TurboJPEG loaded");
  for (auto [w, h] : {std::pair<uint32_t, uint32_t>{1280, 720}, {643, 481}}) {
    const std::string c = fmt("mjpeg %ux%u", w, h);
    const auto src = picture(w, h, w * 31 + h);
    std::vector<uint8_t> whole, cut, back, backWhole;
    if (!j->encode(src.data(), w, h, w, 85, false, whole, &why) || !j->encode(src.data(), w, h, w, 85, true, cut, &why)) {
      r.fail(c, "encode: " + why);
      continue;
    }
    const int dhtWhole = tablesBeforeScan(whole), dhtCut = tablesBeforeScan(cut);
    const bool tail = cut.size() >= 2 && cut[cut.size() - 2] == 0xFF && cut[cut.size() - 1] == 0xD9;
    uint32_t bw = 0, bh = 0, ww = 0, wh = 0;
    const bool dec = j->grey(cut.data(), cut.size(), back, bw, bh, &why);
    const std::string warnCut = j->warning();
    const bool decWhole = j->grey(whole.data(), whole.size(), backWhole, ww, wh, &why);
    const double db = dec && bw == w && bh == h ? psnr(src, back) : 0;
    const bool good = dhtWhole > 0 && dhtCut == 0 && tail && cut.size() < whole.size() && dec && decWhole &&
                      bw == w && bh == h && ww == w && wh == h && back == backWhole && db > 35;
    r.say(good, c,
          fmt("%zu bytes with its %d DHT segments, %zu without (scan and EOI whole); decoded %ux%u%s, %s the whole "
              "frame's decode, PSNR %.1f dB",
              whole.size(), dhtWhole, cut.size(), bw, bh, warnCut.empty() ? "" : (" with a warning: " + warnCut).c_str(),
              back == backWhole ? "byte for byte" : "NOT", db));
    if (w != 1280) continue;
    // ai: a frame cut short (the transfer's tail lost), a header cut short, no JPEG, nothing
    {
      std::vector<uint8_t> o;
      uint32_t ow = 0, oh = 0;
      const size_t n = cut.size() * 6 / 10;
      const bool took = j->grey(cut.data(), n, o, ow, oh, &why);
      const std::string warn = j->warning();
      r.say(!took || !warn.empty(), "mjpeg cut short",
            took ? fmt("60%% of the frame (%zu bytes): accepted with a warning (\"%s\"), %ux%u, PSNR %.1f dB over the "
                       "whole picture, %.1f dB over its top half",
                       n, warn.c_str(), ow, oh, psnr(src, o),
                       psnr(std::vector<uint8_t>(src.begin(), src.begin() + src.size() / 2),
                            std::vector<uint8_t>(o.begin(), o.begin() + std::min(o.size(), src.size() / 2))))
                 : fmt("60%% of the frame (%zu bytes): refused (%s)", n, why.c_str()));
      // ai: every cut inside the header (SOI to the SOS segment's end) refused; every cut in the scan, its EOI
      // ai: included, taken with a warning at the frame's own size. Each cut follows a whole frame decoded on the same
      // ai: handle, as a camera's frames do (2026-10-25: TurboJPEG 3 reports the last header's size for a frame cut
      // ai: before its SOF, so a fresh handle's cut, refused there, hid a stale picture taken on a used one)
      const size_t scan = scanStart(cut);
      int headTaken = 0, headCases = 0, scanBad = 0, scanCases = 0, wholeBad = 0;
      std::string headFirst, headLast, takenFirst, scanFirst;
      std::vector<size_t> heads;
      for (size_t k = 1; k + 1 < scan; k += k < 64 ? 3 : 11) heads.push_back(k);
      heads.push_back(scan - 1);   // ai: all but the SOS segment's last byte
      const uint32_t fw = w, fh = h;   // ai: a lambda takes no structured binding before C++20
      const auto whole = [&] {
        std::vector<uint8_t> b;
        uint32_t bw = 0, bh = 0;
        std::string w2;
        if (!j->grey(cut.data(), cut.size(), b, bw, bh, &w2) || bw != fw || bh != fh) wholeBad++;
      };
      for (size_t k : heads) {
        headCases++;
        whole();
        if (j->grey(cut.data(), k, o, ow, oh, &why)) {
          if (!headTaken++) takenFirst = fmt("%zu bytes taken as %ux%u, warning \"%s\"", k, ow, oh, j->warning().c_str());
        } else {
          (headFirst.empty() ? headFirst : headLast) = fmt("%zu bytes \"%s\"", k, why.c_str());
        }
      }
      for (size_t k : {scan, scan + 1, scan + (cut.size() - scan) / 4, cut.size() - 2, cut.size() - 1}) {
        scanCases++;
        whole();
        const bool t = j->grey(cut.data(), k, o, ow, oh, &why);
        if (!t || j->warning().empty() || ow != w || oh != h) {
          scanBad++;
          if (scanFirst.empty()) scanFirst = fmt("%zu bytes: %s", k, t ? "taken with no warning" : why.c_str());
        }
      }
      r.say(scan > 0 && headTaken == 0 && scanBad == 0 && wholeBad == 0, "mjpeg header cut",
            fmt("%d cuts before the scan (at %zu bytes), each after the whole frame: %d taken%s%s, refused from %s to "
                "%s; %d cuts in the scan (its first bytes to its EOI): %d not taken with a warning%s%s; the whole "
                "frame between them failed %d times",
                headCases, scan, headTaken, takenFirst.empty() ? "" : ", first ", takenFirst.c_str(), headFirst.c_str(),
                headLast.c_str(), scanCases, scanBad, scanFirst.empty() ? "" : ", first ", scanFirst.c_str(), wholeBad));
      std::vector<uint8_t> junk(4096);
      std::mt19937 rng(11);
      for (auto& b : junk) b = uint8_t(rng());
      junk[0] = 0x12;
      const bool any = j->grey(junk.data(), junk.size(), o, ow, oh, &why);
      r.say(!any, "mjpeg not a jpeg", any ? "random bytes taken" : "4096 random bytes: refused (" + why + ")");
      const bool none = j->grey(nullptr, 0, o, ow, oh, &why);
      r.say(!none, "mjpeg empty", none ? "nothing taken" : "refused (" + why + ")");
    }
  }
  // ai: a 1920 x 1080 frame decoded to grey, the time a frame (CPU ms)
  const auto src = picture(1920, 1080, 5);
  std::vector<uint8_t> cut, o;
  if (!j->encode(src.data(), 1920, 1080, 1920, 85, true, cut, &why)) { r.fail("time MJPEG", why); return; }
  std::vector<double> ms;
  uint32_t ow = 0, oh = 0;
  for (int i = 0; i < 18; i++) {
    const int64_t t0 = steadyNs();
    if (!j->grey(cut.data(), cut.size(), o, ow, oh, &why)) break;
    if (i >= 3) ms.push_back(double(steadyNs() - t0) / 1e6);
  }
  r.say(ms.size() == 15, "time MJPEG",
        ms.size() == 15 ? fmt("CPU ms to decode a 1920 x 1080 4:2:2 frame of %zu bytes to grey (median of 15): %.3f", cut.size(), median(ms))
                        : "decode failed: " + why);
}

// ai: --- the clock ---

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))];
}

#if defined(_WIN32) && !defined(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION)
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// ai: The system's own precise sleep to t (steadyNs's clock) with no spin after it: sleepUntil's reference, written
// ai: here apart from clock.cpp, so that a sleepUntil gone coarser than its system's sleep shows against it
// ai: (clock_nanosleep on CLOCK_MONOTONIC's absolute time, a high-resolution waitable timer, mach_wait_until); false
// ai: where the system has none of these (Windows before 10 1803).
bool systemSleep(int64_t t) {
#if defined(__linux__)
  timespec ts{};
  ts.tv_sec = time_t(t / 1'000'000'000);
  ts.tv_nsec = long(t % 1'000'000'000);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
  return true;
#elif defined(_WIN32)
  HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (!h) return false;
  LARGE_INTEGER due;
  due.QuadPart = -std::max<int64_t>((t - steadyNs()) / 100, 1);
  if (SetWaitableTimer(h, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(h, INFINITE);
  CloseHandle(h);
  return true;
#elif defined(__APPLE__)
  static const mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t i{}; mach_timebase_info(&i); return i; }();
  const int64_t left = t - steadyNs();
  if (left > 0) mach_wait_until(mach_absolute_time() + uint64_t(double(left) * tb.denom / tb.numer));
  return true;
#else
  (void)t;
  return false;
#endif
}

// ai: sleepUntil and the system's sleep, 200 targets each 0.5 to 20 ms ahead, taken in turn so both meet the same
// ai: machine: sleepUntil never early, and at the median no later than the system's own sleep and 20 us (the clocks'
// ai: reading): it sleeps to 0.3 ms before t and spins the rest, so it is later than its system's sleep only where it
// ai: went wrong, while a machine's coarse timers make both late alike.
void clockSleep(Report& r) {
  std::mt19937 rng(17);
  std::uniform_int_distribution<int64_t> ahead(500'000, 20'000'000);
  std::vector<double> late, sys;
  int early = 0;
  bool hasSys = true;
  for (int i = 0; i < 200; i++) {
    const int64_t a = ahead(rng);
    const int64_t t = steadyNs() + a;
    sleepUntil(t);
    const int64_t d = steadyNs() - t;
    if (d < 0) early++;
    late.push_back(double(d) / 1e3);
    if (!hasSys) continue;
    const int64_t u = steadyNs() + a;
    if (systemSleep(u)) sys.push_back(double(steadyNs() - u) / 1e3);
    else hasSys = false;
  }
  const double p50 = pct(late, 0.5), p99 = pct(late, 0.99), mx = pct(late, 1.0);
  const double s50 = hasSys ? pct(sys, 0.5) : 0, s99 = hasSys ? pct(sys, 0.99) : 0;
  r.say(early == 0 && (!hasSys || p50 <= s50 + 20), "clock sleepUntil",
        fmt("200 targets 0.5 to 20 ms ahead: late p50 %.2f us, p99 %.2f us, max %.2f us%s; the system's own sleep to the "
            "same targets %s",
            p50, p99, mx, early ? fmt(", %d EARLY", early).c_str() : ", none early",
            hasSys ? fmt("late p50 %.2f us, p99 %.2f us (the median it is held to, and 20 us)", s50, s99).c_str()
                   : "not here (no high-resolution timer): held to never early alone"));
}

}  // namespace

std::string jpegOptional() {
#if defined(_WIN32) || defined(__APPLE__)
  const char* named = std::getenv("LIZ_TURBOJPEG");
  if (named && *named) return "";
  std::string why;
  if (Jpeg::load(why)) return "";
#if defined(_WIN32)
  const char* own = "Media Foundation";
#else
  const char* own = "AVFoundation";
#endif
  return why + " (" + own + " decodes a camera's MJPEG here, so the receiver does without TurboJPEG)";
#else
  return "";
#endif
}

int luma(const Args& a) {
  if (!a.empty()) {
    std::fprintf(stderr, "cam_check luma takes no arguments\n");
    return kUsage;
  }
  Report r;
  lumaLayouts(r);
  lumaRefusals(r);
  lumaPick(r);
  lumaTiming(r);
  mjpeg(r);
  clockSleep(r);
  return r.code();
}

}  // namespace lizard::check
