// ai: cam_check avf (2026-10-25, checks.h): AVFoundation's backend on macOS, with no camera opened. Its cases, all of
// ai: them or those named (`cam_check avf [list|420f|420v|2vuy|bgra|clock|pick|names]`):
// ai:   list: the cameras the backend lists, printed (a CI runner has none, which is no failure); one whose id is not
// ai:     avf: or whose backend is not uvc fails;
// ai:   420f, 420v, 2vuy: the backend's frame path (avf_test.h: the code its capture delegate runs) over a pixel buffer
// ai:     made here, 640 x 480 and 646 x 480 (rows likelier padded past the width), its luma a pattern and the bytes
// ai:     past each row's samples 0xEE: one Frame sunk, the buffer's size, the time and count handed in, every sample
// ai:     as painted; the bi-planar layouts in place (plane 0's own address and pitch);
// ai:   bgra: a layout the backend does not take is refused with why, nothing sunk;
// ai:   clock: captures 50 ms and 2 s before now on the host clock (CLOCK_UPTIME_RAW, mach_absolute_time's) are given
// ai:     a Frame::ns within 1 ms of steadyNs's now less as much, at timescales of 1e9 and 90 kHz, so a frame stamped at
// ai:     its arrival shows 50 ms and 2 s off (a capture of now, as the case had, passed it); a time with no timescale
// ai:     is steadyNs's now. Where the host clock and steadyNs's agree (a Mac that never slept since boot) the offset
// ai:     between them is 0 and its loss cannot show;
// ai:   pick, names: the output format asked of each offer; the FourCharCode names and which subtypes are compressed.
// ai: CoreVideo's buffers are made through its C calls (the file is plain C++), with the default allocator (null) and
// ai: released by CVPixelBufferRelease: nothing of CoreFoundation's is called. Elsewhere the check returns kNotHere.
#include "checks.h"

#if defined(__APPLE__)
#include <CoreVideo/CVPixelBuffer.h>
#include <time.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "avf_test.h"
#include "backends.h"
#include "trigger.h"
#endif

namespace lizard::check {

#if !defined(__APPLE__)

// ai: AVFoundation is macOS's alone
int avf(const Args&) { return kNotHere; }

#else

namespace {

struct Out {
  int bad = 0;
  void ok(const char* c, const std::string& s) { std::printf("ok %s: %s\n", c, s.c_str()); }
  void fail(const char* c, const std::string& s) {
    bad++;
    std::printf("FAIL %s: %s\n", c, s.c_str());
  }
};

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
  char b[512];
  va_list a;
  va_start(a, f);
  std::vsnprintf(b, sizeof b, f, a);
  va_end(a);
  return b;
}

// ai: a FourCharCode from its four characters
constexpr uint32_t fcc(const char* s) {
  return uint32_t(uint8_t(s[0])) << 24 | uint32_t(uint8_t(s[1])) << 16 | uint32_t(uint8_t(s[2])) << 8 | uint8_t(s[3]);
}

// ai: the painted luma: every sample's own value, so a wrong pitch, plane or byte order shows at once
uint8_t lumaAt(uint32_t x, uint32_t y) { return uint8_t((x * 7u + y * 13u + ((x * y) >> 3)) & 255u); }
constexpr uint8_t kPad = 0xEE;

// ai: a w x h buffer of the layout: the Y samples the pattern, the chroma 128, every byte past a row's samples kPad;
// ai: a layout of another kind is left as CoreVideo made it. Null with why where CoreVideo refuses.
CVPixelBufferRef paint(OSType type, uint32_t w, uint32_t h, std::string& why) {
  CVPixelBufferRef pb = nullptr;
  const CVReturn r = CVPixelBufferCreate(nullptr, w, h, type, nullptr, &pb);
  if (r != kCVReturnSuccess || !pb) {
    why = fmt("CVPixelBufferCreate returned %d", int(r));
    return nullptr;
  }
  if (CVPixelBufferLockBaseAddress(pb, 0) != kCVReturnSuccess) {
    why = "the new buffer did not lock";
    CVPixelBufferRelease(pb);
    return nullptr;
  }
  const bool biPlanar =
      type == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || type == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  if ((type == kCVPixelFormatType_422YpCbCr8 && !CVPixelBufferGetBaseAddress(pb)) ||
      (biPlanar && (!CVPixelBufferGetBaseAddressOfPlane(pb, 0) || !CVPixelBufferGetBaseAddressOfPlane(pb, 1)))) {
    why = "the new buffer has no pixels";
    CVPixelBufferUnlockBaseAddress(pb, 0);
    CVPixelBufferRelease(pb);
    return nullptr;
  }
  if (type == kCVPixelFormatType_422YpCbCr8) {
    auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const size_t pitch = CVPixelBufferGetBytesPerRow(pb);
    for (uint32_t y = 0; y < h; y++) {
      uint8_t* row = base + size_t(y) * pitch;
      std::fill(row, row + pitch, kPad);
      for (uint32_t x = 0; x < w; x++) {
        row[2 * size_t(x)] = 128;   // ai: Cb for an even x, Cr for an odd one
        row[2 * size_t(x) + 1] = lumaAt(x, y);
      }
    }
  } else if (biPlanar) {
    auto* y0 = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
    const size_t pitch = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
    for (uint32_t y = 0; y < h; y++) {
      uint8_t* row = y0 + size_t(y) * pitch;
      std::fill(row, row + pitch, kPad);
      for (uint32_t x = 0; x < w; x++) row[x] = lumaAt(x, y);
    }
    auto* c = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 1));
    std::fill(c, c + CVPixelBufferGetBytesPerRowOfPlane(pb, 1) * CVPixelBufferGetHeightOfPlane(pb, 1), uint8_t(128));
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return pb;
}

// ai: one buffer of the layout at w x h through the backend's frame path
void frameCase(Out& o, const char* name, OSType type, uint32_t w, uint32_t h) {
  std::string why;
  CVPixelBufferRef pb = paint(type, w, h, why);
  if (!pb) {
    o.fail(name, fmt("%u x %u: no buffer: %s", w, h, why.c_str()));
    return;
  }
  const bool inPlace = type != kCVPixelFormatType_422YpCbCr8;
  const int64_t ns = 1234567890123;
  const uint64_t seq = 42;
  int calls = 0;
  uint32_t pitch = 0;
  std::string bad;
  std::vector<uint8_t> scratch;
  double ms = -1;
  const bool taken = lizard::cam::avf::frame(pb, ns, seq, scratch, [&](const lizard::cam::Frame& f) {
    calls++;
    pitch = f.stride;
    if (f.width != w || f.height != h) {
      bad = fmt("the frame is %u x %u", f.width, f.height);
      return;
    }
    if (!f.luma || f.stride < f.width) {
      bad = fmt("no luma, or rows %u bytes apart for %u samples", f.stride, f.width);
      return;
    }
    if (f.ns != ns || f.seq != seq || f.delayed) {
      bad = "the time, the count or delayed changed on the way";
      return;
    }
    // ai: the backend holds the buffer locked for this call, so its plane's address is readable here
    const void* plane0 = inPlace ? CVPixelBufferGetBaseAddressOfPlane(pb, 0) : nullptr;
    if (inPlace && (f.luma != plane0 || f.stride != CVPixelBufferGetBytesPerRowOfPlane(pb, 0))) {
      bad = "the luma is not plane 0 in place, at its own pitch";
      return;
    }
    for (uint32_t y = 0; y < h && bad.empty(); y++)
      for (uint32_t x = 0; x < w; x++) {
        const uint8_t v = f.luma[size_t(y) * f.stride + x];
        if (v != lumaAt(x, y)) {
          bad = fmt("sample (%u, %u) is %u, painted %u", x, y, unsigned(v), unsigned(lumaAt(x, y)));
          break;
        }
      }
  }, ms, why);
  CVPixelBufferRelease(pb);
  if (!taken) o.fail(name, fmt("%u x %u refused: %s", w, h, why.c_str()));
  else if (calls != 1) o.fail(name, fmt("%u x %u sunk %d times", w, h, calls));
  else if (!bad.empty()) o.fail(name, fmt("%u x %u: %s", w, h, bad.c_str()));
  else
    o.ok(name, fmt("%u x %u, rows %u bytes apart (%s), every sample as painted, %.3f ms to luma", w, h, pitch,
                   inPlace ? "plane 0 in place" : "converted", ms));
}

void refuseCase(Out& o) {
  std::string why;
  CVPixelBufferRef pb = paint(kCVPixelFormatType_32BGRA, 64, 48, why);
  if (!pb) {
    o.fail("bgra", "no buffer: " + why);
    return;
  }
  int calls = 0;
  std::vector<uint8_t> scratch;
  double ms = 0;
  const bool taken = lizard::cam::avf::frame(pb, 1, 1, scratch, [&](const lizard::cam::Frame&) { calls++; }, ms, why);
  CVPixelBufferRelease(pb);
  if (taken || calls) o.fail("bgra", fmt("a BGRA buffer was taken (%d sink calls)", calls));
  else if (why.find("BGRA") == std::string::npos) o.fail("bgra", "refused, but why does not name it: " + why);
  else o.ok("bgra", "refused: " + why);
}

void clockCase(Out& o) {
  using lizard::cam::avf::frameNs;
  using lizard::cam::steadyNs;
  const auto inMs = [](int64_t d) { return d > -1'000'000 && d < 1'000'000; };
  bool good = true;
  std::string line;
  double offset = 0;
  for (const int64_t ago : {int64_t(50'000'000), int64_t(2'000'000'000)}) {
    const int64_t host = int64_t(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
    if (host <= ago) continue;   // ai: a host clock younger than the capture (not on a running Mac)
    const int64_t t = host - ago;
    const int64_t a = frameNs(t, 1000000000);
    const int64_t b = frameNs(t / 100000 * 9 + t % 100000 * 9 / 100000, 90000);   // ai: t * 90000 / 1e9
    const int64_t now = steadyNs();
    // ai: how far each frame's time is from where its capture was: 0 where it carries the capture's time
    const int64_t da = now - ago - a, db = now - ago - b;
    good = good && inMs(da) && inMs(db);
    offset = double(a - t) / 1e9;
    line += fmt("%s%.3f s ago: %lld ns (1e9), %lld ns (90 kHz)", line.empty() ? "" : "; ", double(ago) / 1e9, (long long)da,
                (long long)db);
  }
  const int64_t c = frameNs(5, 0);
  const int64_t dc = steadyNs() - c;
  good = good && !line.empty() && inMs(dc);
  const std::string what =
      fmt("each capture's Frame::ns against steadyNs's now less its age: %s; no timescale %lld ns before steadyNs's "
          "now; steadyNs less the host clock %.6f s (the sleep since boot, where steadyNs counts it)",
          line.c_str(), (long long)dc, offset);
  if (good) o.ok("clock", what);
  else o.fail("clock", what);
}

void pickCase(Out& o) {
  using lizard::cam::avf::pickFormat;
  const uint32_t f420 = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                 v420 = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, uyvy = kCVPixelFormatType_422YpCbCr8,
                 bgra = kCVPixelFormatType_32BGRA, yuvs = fcc("yuvs");
  struct Want {
    std::vector<uint32_t> offer;
    uint32_t pick;
  };
  const Want wants[] = {
      {{v420, uyvy, f420, bgra}, f420}, {{bgra, uyvy, v420}, v420}, {{yuvs, bgra, uyvy}, uyvy},
      {{bgra, yuvs}, 0},                {{}, 0},
  };
  int n = 0;
  for (const Want& w : wants) {
    const uint32_t got = pickFormat(w.offer);
    if (got != w.pick) {
      o.fail("pick", fmt("offer %d: asked for %08x, %08x wanted", n, unsigned(got), unsigned(w.pick)));
      return;
    }
    n++;
  }
  o.ok("pick", fmt("%d offers: '420f' first, then '420v', then '2vuy', else none", n));
}

void namesCase(Out& o) {
  using lizard::cam::avf::compressed;
  using lizard::cam::avf::fourcc;
  const struct {
    uint32_t code;
    const char* name;
  } names[] = {{fcc("420v"), "420v"}, {fcc("2vuy"), "2vuy"},          {fcc("dmb1"), "dmb1"},
               {32, "0x00000020"},    {fcc("y4\x01" "a"), "0x79340161"}};
  for (auto& n : names)
    if (fourcc(n.code) != n.name) {
      o.fail("names", fmt("%08x named %s, %s wanted", unsigned(n.code), fourcc(n.code).c_str(), n.name));
      return;
    }
  static const char* const kPacked[] = {"dmb1", "jpeg", "avc1", "hvc1"};
  static const char* const kRaw[] = {"420v", "420f", "2vuy", "yuvs"};
  for (const char* c : kPacked)
    if (!compressed(fcc(c))) {
      o.fail("names", std::string(c) + " not taken as compressed");
      return;
    }
  for (const char* c : kRaw)
    if (compressed(fcc(c))) {
      o.fail("names", std::string(c) + " taken as compressed");
      return;
    }
  o.ok("names", "FourCharCodes named as Mode::format gives them; JPEG, MJPEG, H.264 and HEVC compressed, YCbCr not");
}

void listCase(Out& o) {
  std::vector<std::string> notes;
  const std::vector<lizard::cam::Info> cams = lizard::cam::avfList(&notes);
  std::string bad;
  for (const auto& c : cams) {
    std::printf("   %s | %s | %s | %s\n", c.id.c_str(), c.name.c_str(), c.backend.c_str(), c.detail.c_str());
    if (c.id.compare(0, 4, "avf:") != 0 || c.backend != "uvc") bad = c.id + " (" + c.backend + ")";
  }
  for (const auto& n : notes) std::printf("   note: %s\n", n.c_str());
  if (!bad.empty()) o.fail("list", "a camera listed as " + bad);
  else o.ok("list", fmt("%zu camera(s) listed (none is not a failure)", cams.size()));
}

}  // namespace

int avf(const Args& a) {
  static const char* const kCases[] = {"list", "420f", "420v", "2vuy", "bgra", "clock", "pick", "names"};
  for (const auto& n : a)
    if (std::none_of(std::begin(kCases), std::end(kCases), [&](const char* c) { return n == c; })) {
      std::fprintf(stderr, "usage: cam_check avf [list|420f|420v|2vuy|bgra|clock|pick|names]...\n");
      return kUsage;
    }
  const auto want = [&](const char* c) { return a.empty() || std::find(a.begin(), a.end(), c) != a.end(); };
  Out o;
  if (want("list")) listCase(o);
  struct Layout {
    const char* name;
    OSType type;
  };
  static const Layout kLayouts[] = {{"420f", kCVPixelFormatType_420YpCbCr8BiPlanarFullRange},
                                    {"420v", kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange},
                                    {"2vuy", kCVPixelFormatType_422YpCbCr8}};
  for (const Layout& l : kLayouts)
    if (want(l.name)) {
      frameCase(o, l.name, l.type, 640, 480);
      frameCase(o, l.name, l.type, 646, 480);
    }
  if (want("bgra")) refuseCase(o);
  if (want("clock")) clockCase(o);
  if (want("pick")) pickCase(o);
  if (want("names")) namesCase(o);
  return o.bad ? kFailed : kHeld;
}

#endif

}  // namespace lizard::check
