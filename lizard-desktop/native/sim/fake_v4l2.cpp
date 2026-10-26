// ai: sim/fake_v4l2.h's devices (2026-10-25): uvcvideo's and videobuf2's semantics over memory, one mutex a device,
// ai: a producer thread a stream. Nothing here reaches the kernel's video devices: the table's calls are this file's.
#include "sim/fake_v4l2.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>

#include "camera/jpeg.h"
#include "sim/scene.h"

namespace lizard::sim {

namespace {

// ai: steadyNs's clock (camera/clock.h: std::chrono::steady_clock, CLOCK_MONOTONIC here)
int64_t nowNs() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
std::chrono::steady_clock::time_point timeAt(int64_t ns) {
  return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

int fail(int e) {
  errno = e;
  return -1;
}

template <typename T, size_t N>
void copy(T (&to)[N], const std::string& s) {
  std::memset(to, 0, N);
  std::memcpy(to, s.data(), std::min(s.size(), N - 1));
}

struct Iv { uint32_t num = 1, den = 30; };
struct Size { uint32_t w, h; std::vector<Iv> iv; };
struct Format {
  uint32_t fourcc;
  const char* desc;
  bool compressed;
  std::vector<Size> sizes;
  bool stepwise = false;          // ai: sizes a range (range), intervals continuous from fastest to slowest
  v4l2_frmsize_stepwise range{};
  Iv fastest{}, slowest{};
};
struct CtlDef {
  uint32_t id, type;
  const char* name;
  int64_t min, max, step, def;
  std::vector<std::pair<int64_t, const char*>> menu;
  uint32_t master = 0;            // ai: the control that makes this one automatic: writable while it is `manual`
  int64_t manual = 0;
};
struct Profile {
  std::string name, card, driver;
  bool meta = false;              // ai: a metadata node: no video formats, no controls
  bool old = false;               // ai: no extended controls, no device caps, buffers stamped with no clock
  bool bare = false;              // ai: no ENUM_FRAMESIZES, ENUM_FRAMEINTERVALS, G_PARM or S_PARM (ENOTTY)
  std::vector<Format> formats;
  std::vector<CtlDef> ctls;       // ai: by id
};

const std::vector<Iv> kWebcam = {{1, 30}, {1, 24}, {1, 20}, {1, 15}, {1, 10}, {2, 15}, {1, 5}};

CtlDef ctl(uint32_t id, uint32_t type, const char* name, int64_t min, int64_t max, int64_t step, int64_t def) {
  return CtlDef{id, type, name, min, max, step, def, {}, 0, 0};
}
CtlDef menu(uint32_t id, const char* name, int64_t min, int64_t max, int64_t def,
            std::vector<std::pair<int64_t, const char*>> items) {
  return CtlDef{id, V4L2_CTRL_TYPE_MENU, name, min, max, 1, def, std::move(items), 0, 0};
}
CtlDef slave(CtlDef c, uint32_t master, int64_t manual) {
  c.master = master;
  c.manual = manual;
  return c;
}
CtlDef classOf(uint32_t id, const char* name) {
  return CtlDef{id, V4L2_CTRL_TYPE_CTRL_CLASS, name, 0, 0, 0, 0, {}, 0, 0};
}

Profile profileOf(const std::string& name) {
  Profile p;
  p.name = name;
  p.driver = "uvcvideo";
  const uint32_t I = V4L2_CTRL_TYPE_INTEGER, B = V4L2_CTRL_TYPE_BOOLEAN;
  const CtlDef user = classOf(V4L2_CID_USER_CLASS, "User Controls");
  const CtlDef camera = classOf(V4L2_CID_CAMERA_CLASS, "Camera Controls");
  if (name == "c930e") {
    p.card = "Fake C930e";
    p.formats = {
        {V4L2_PIX_FMT_YUYV, "YUYV 4:2:2", false,
         {{1920, 1080, {{1, 5}}}, {1280, 720, {{1, 10}, {2, 15}, {1, 5}}}, {640, 480, kWebcam}}},
        {V4L2_PIX_FMT_MJPEG, "Motion-JPEG", true, {{1920, 1080, kWebcam}, {1280, 720, kWebcam}, {640, 480, kWebcam}}},
    };
    p.ctls = {
        user,
        ctl(V4L2_CID_BRIGHTNESS, I, "Brightness", 0, 255, 1, 128),
        ctl(V4L2_CID_CONTRAST, I, "Contrast", 0, 255, 1, 128),
        ctl(V4L2_CID_GAIN, I, "Gain", 0, 255, 1, 0),
        menu(V4L2_CID_POWER_LINE_FREQUENCY, "Power Line Frequency", 0, 2, 2,
             {{0, "Disabled"}, {1, "50 Hz"}, {2, "60 Hz"}}),
        ctl(V4L2_CID_SHARPNESS, I, "Sharpness", 0, 255, 1, 128),
        ctl(V4L2_CID_BACKLIGHT_COMPENSATION, I, "Backlight Compensation", 0, 1, 1, 1),
        camera,
        menu(V4L2_CID_EXPOSURE_AUTO, "Auto Exposure", 0, 3, 3, {{1, "Manual Mode"}, {3, "Aperture Priority Mode"}}),
        slave(ctl(V4L2_CID_EXPOSURE_ABSOLUTE, I, "Exposure Time, Absolute", 3, 2047, 1, 250), V4L2_CID_EXPOSURE_AUTO,
              V4L2_EXPOSURE_MANUAL),
        ctl(V4L2_CID_EXPOSURE_AUTO_PRIORITY, B, "Exposure, Dynamic Framerate", 0, 1, 1, 1),
        slave(ctl(V4L2_CID_FOCUS_ABSOLUTE, I, "Focus, Absolute", 0, 255, 5, 0), V4L2_CID_FOCUS_AUTO, 0),
        ctl(V4L2_CID_FOCUS_AUTO, B, "Focus, Automatic Continuous", 0, 1, 1, 1),
        ctl(V4L2_CID_ZOOM_ABSOLUTE, I, "Zoom, Absolute", 100, 500, 1, 100),
    };
  } else if (name == "uvc60") {
    p.card = "Fake UVC 60";
    p.formats = {
        {V4L2_PIX_FMT_YUYV, "YUYV 4:2:2", false, {{640, 480, {{1, 60}, {1, 30}}}}},
        {V4L2_PIX_FMT_MJPEG, "Motion-JPEG", true, {{1280, 720, {{1, 60}, {1, 30}}}, {1920, 1080, {{1, 30}}}}},
        {V4L2_PIX_FMT_NV12, "Y/UV 4:2:0", false, {{1280, 720, {{1, 30}}}}},
    };
    p.ctls = {
        user,
        ctl(V4L2_CID_BRIGHTNESS, I, "Brightness", -64, 64, 1, 0),
        ctl(V4L2_CID_CONTRAST, I, "Contrast", 0, 64, 1, 32),
        ctl(V4L2_CID_GAIN, I, "Gain", 0, 100, 1, 10),
        ctl(V4L2_CID_SHARPNESS, I, "Sharpness", 0, 7, 1, 3),
        ctl(V4L2_CID_BACKLIGHT_COMPENSATION, I, "Backlight Compensation", 0, 2, 1, 1),
        camera,
        menu(V4L2_CID_EXPOSURE_AUTO, "Auto Exposure", 0, 1, 0, {{0, "Auto Mode"}, {1, "Manual Mode"}}),
        slave(ctl(V4L2_CID_EXPOSURE_ABSOLUTE, I, "Exposure Time, Absolute", 1, 5000, 1, 156), V4L2_CID_EXPOSURE_AUTO,
              V4L2_EXPOSURE_MANUAL),
    };
  } else if (name == "mono") {
    p.card = "Fake Mono";
    p.formats = {
        {V4L2_PIX_FMT_GREY, "8-bit Greyscale", false, {{1280, 1024, {{1, 60}, {1, 30}}}}},
        {V4L2_PIX_FMT_Y16, "16-bit Greyscale", false, {{1280, 1024, {{1, 30}}}}},
    };
    p.ctls = {
        user,
        ctl(V4L2_CID_GAIN, I, "Gain", 0, 48, 1, 0),
        camera,
        menu(V4L2_CID_EXPOSURE_AUTO, "Auto Exposure", 0, 3, 3, {{1, "Manual Mode"}, {3, "Aperture Priority Mode"}}),
        slave(ctl(V4L2_CID_EXPOSURE_ABSOLUTE, I, "Exposure Time, Absolute", 1, 10000, 1, 100), V4L2_CID_EXPOSURE_AUTO,
              V4L2_EXPOSURE_MANUAL),
        ctl(V4L2_CID_EXPOSURE_AUTO_PRIORITY, B, "Exposure, Dynamic Framerate", 0, 1, 1, 0),
    };
  } else if (name == "bayer") {
    p.card = "Fake Bayer";
    p.driver = "fakeold";
    p.old = true;
    p.formats = {{V4L2_PIX_FMT_SRGGB8, "8-bit Bayer RGRG/GBGB", false, {{1920, 1200, {{1, 60}}}}}};
    p.ctls = {
        ctl(V4L2_CID_BRIGHTNESS, I, "Brightness", 0, 255, 1, 128),
        ctl(V4L2_CID_GAIN, I, "Gain", 0, 255, 1, 32),
        ctl(V4L2_CID_EXPOSURE_ABSOLUTE, I, "Exposure Time, Absolute", 1, 2000, 1, 166),
    };
  } else if (name == "stepwise") {
    p.card = "Fake Stepwise";
    Format f{V4L2_PIX_FMT_GREY, "8-bit Greyscale", false, {}};
    f.stepwise = true;
    f.range = v4l2_frmsize_stepwise{160, 2048, 16, 120, 1536, 8};
    f.fastest = {1, 120};
    f.slowest = {1, 1};
    p.formats = {f};
  } else if (name == "formats") {
    p.card = "Fake Formats";
    const std::vector<Size> qvga = {{320, 240, {{1, 30}}}};
    p.formats = {
        {V4L2_PIX_FMT_UYVY, "UYVY 4:2:2", false, qvga},       {V4L2_PIX_FMT_YVYU, "YVYU 4:2:2", false, qvga},
        {V4L2_PIX_FMT_VYUY, "VYUY 4:2:2", false, qvga},       {V4L2_PIX_FMT_NV21, "Y/VU 4:2:0", false, qvga},
        {V4L2_PIX_FMT_YUV420, "Planar YUV 4:2:0", false, qvga}, {V4L2_PIX_FMT_YVU420, "Planar YVU 4:2:0", false, qvga},
        {V4L2_PIX_FMT_Y10, "10-bit Greyscale", false, qvga},  {V4L2_PIX_FMT_Y12, "12-bit Greyscale", false, qvga},
        {V4L2_PIX_FMT_RGB24, "24-bit RGB 8-8-8", false, qvga}, {V4L2_PIX_FMT_BGR24, "24-bit BGR 8-8-8", false, qvga},
        {V4L2_PIX_FMT_H264, "H.264", true, {{640, 480, {{1, 30}}}}},
    };
  } else if (name == "card") {
    p.card = "Fake Capture Card";
    p.driver = "fakecard";
    p.old = true;
    p.bare = true;
    p.formats = {{V4L2_PIX_FMT_YUYV, "YUYV 4:2:2", false, {{1280, 720, {{1, 60}}}}}};
  } else if (name == "meta") {
    p.card = "Fake C930e";
    p.meta = true;
  } else {
    throw std::invalid_argument("fake_v4l2: no profile " + name);
  }
  std::sort(p.ctls.begin(), p.ctls.end(), [](const CtlDef& a, const CtlDef& b) { return a.id < b.id; });
  return p;
}

const Format* formatOf(const Profile& p, uint32_t fourcc) {
  for (const Format& f : p.formats)
    if (f.fourcc == fourcc) return &f;
  return nullptr;
}

const Size* sizeOf(const Format& f, uint32_t w, uint32_t h) {
  for (const Size& s : f.sizes)
    if (s.w == w && s.h == h) return &s;
  return nullptr;
}

bool inRange(const v4l2_frmsize_stepwise& r, uint32_t w, uint32_t h) {
  return w >= r.min_width && w <= r.max_width && h >= r.min_height && h <= r.max_height &&
         (w - r.min_width) % r.step_width == 0 && (h - r.min_height) % r.step_height == 0;
}

uint32_t snapStep(uint32_t v, uint32_t lo, uint32_t hi, uint32_t step) {
  v = std::min(std::max(v, lo), hi);
  const uint32_t k = (v - lo + step / 2) / step;
  return std::min(lo + k * step, hi - (hi - lo) % step);
}

// ai: The size S_FMT takes for w x h: a range's nearest step; else uvcvideo's nearest, the one whose and the asked
// ai: size's areas overlap the most against both of theirs (uvc_v4l2_try_format)
void nearestSize(const Format& f, uint32_t w, uint32_t h, uint32_t& ow, uint32_t& oh) {
  if (f.stepwise) {
    ow = snapStep(w, f.range.min_width, f.range.max_width, f.range.step_width);
    oh = snapStep(h, f.range.min_height, f.range.max_height, f.range.step_height);
    return;
  }
  uint64_t best = UINT64_MAX;
  for (const Size& s : f.sizes) {
    const uint64_t overlap = (uint64_t)std::min(s.w, w) * std::min(s.h, h);
    const uint64_t d = (uint64_t)s.w * s.h + (uint64_t)w * h - 2 * overlap;
    if (d < best) best = d, ow = s.w, oh = s.h;
  }
}

double secs(const Iv& v) { return v.den ? (double)v.num / v.den : 0; }

// ai: The interval S_PARM takes for `want` at w x h: the nearest the size lists; a continuous range's, want within it
Iv nearestIv(const Format& f, uint32_t w, uint32_t h, Iv want) {
  if (f.stepwise) {
    if (!want.num || !want.den) return f.fastest;
    if (secs(want) < secs(f.fastest)) return f.fastest;
    if (secs(want) > secs(f.slowest)) return f.slowest;
    return want;
  }
  const Size* s = sizeOf(f, w, h);
  if (!s || s->iv.empty()) return Iv{};
  if (!want.num || !want.den) return s->iv.front();
  Iv best = s->iv.front();
  for (const Iv& v : s->iv)
    if (std::fabs(secs(v) - secs(want)) < std::fabs(secs(best) - secs(want))) best = v;
  return best;
}

uint32_t bplOf(uint32_t fourcc, uint32_t w) {
  switch (fourcc) {
    case V4L2_PIX_FMT_Y16: case V4L2_PIX_FMT_Y10: case V4L2_PIX_FMT_Y12:
    case V4L2_PIX_FMT_YUYV: case V4L2_PIX_FMT_UYVY: case V4L2_PIX_FMT_YVYU: case V4L2_PIX_FMT_VYUY:
      return 2 * w;
    case V4L2_PIX_FMT_RGB24: case V4L2_PIX_FMT_BGR24:
      return 3 * w;
    case V4L2_PIX_FMT_MJPEG: case V4L2_PIX_FMT_H264:
      return 0;
    default:
      return w;   // ai: GREY, the Bayer layouts, and the planar 4:2:0 ones (their Y plane's, as uvcvideo says)
  }
}

uint32_t imageOf(uint32_t fourcc, uint32_t w, uint32_t h) {
  switch (fourcc) {
    case V4L2_PIX_FMT_NV12: case V4L2_PIX_FMT_NV21: case V4L2_PIX_FMT_YUV420: case V4L2_PIX_FMT_YVU420:
      return w * h * 3 / 2;
    case V4L2_PIX_FMT_MJPEG: case V4L2_PIX_FMT_H264:
      return w * h * 2;   // ai: a compressed frame's most (uvcvideo's dwMaxVideoFrameBufferSize)
    default:
      return bplOf(fourcc, w) * h;
  }
}

void fillPix(uint32_t fourcc, uint32_t w, uint32_t h, v4l2_pix_format& p) {
  std::memset(&p, 0, sizeof p);
  p.width = w;
  p.height = h;
  p.pixelformat = fourcc;
  p.field = V4L2_FIELD_NONE;
  p.bytesperline = bplOf(fourcc, w);
  p.sizeimage = imageOf(fourcc, w, h);
  p.colorspace = V4L2_COLORSPACE_SRGB;
}

void gradientInto(uint64_t seq, uint32_t w, uint32_t h, uint8_t* out) {
  for (uint32_t y = 0; y < h; y++) {
    uint32_t v = (uint32_t)((y + 7 * seq) % 510);
    uint8_t* row = out + (size_t)y * w;
    for (uint32_t x = 0; x < w; x++) {
      row[x] = (uint8_t)(v < 255 ? v : 509 - v);
      if (++v == 510) v = 0;
    }
  }
}

// ai: A grey picture (rows of w) into a frame of the format, chroma 128; the bytes the frame holds (0 where it cannot)
size_t pack(uint32_t fourcc, const uint8_t* g, uint32_t w, uint32_t h, uint8_t* out, size_t cap, cam::Jpeg* jpeg,
            int quality, std::vector<uint8_t>& jpg) {
  const size_t n = (size_t)w * h;
  if (fourcc == V4L2_PIX_FMT_MJPEG) {
    if (!jpeg || !jpeg->encode(g, w, h, w, quality, true, jpg, nullptr)) return 0;
    const size_t k = std::min(jpg.size(), cap);
    std::memcpy(out, jpg.data(), k);
    return k;
  }
  if (imageOf(fourcc, w, h) > cap) return 0;
  switch (fourcc) {
    case V4L2_PIX_FMT_GREY: case V4L2_PIX_FMT_SBGGR8: case V4L2_PIX_FMT_SGBRG8: case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
      std::memcpy(out, g, n);
      return n;
    case V4L2_PIX_FMT_Y16: case V4L2_PIX_FMT_Y10: case V4L2_PIX_FMT_Y12: {
      const int shift = fourcc == V4L2_PIX_FMT_Y16 ? 8 : fourcc == V4L2_PIX_FMT_Y12 ? 4 : 2;
      for (size_t i = 0; i < n; i++) {
        const uint16_t v = (uint16_t)(g[i] << shift);
        out[2 * i] = (uint8_t)v;
        out[2 * i + 1] = (uint8_t)(v >> 8);
      }
      return 2 * n;
    }
    case V4L2_PIX_FMT_YUYV: case V4L2_PIX_FMT_YVYU:
      for (size_t i = 0; i < n; i++) out[2 * i] = g[i], out[2 * i + 1] = 128;
      return 2 * n;
    case V4L2_PIX_FMT_UYVY: case V4L2_PIX_FMT_VYUY:
      for (size_t i = 0; i < n; i++) out[2 * i] = 128, out[2 * i + 1] = g[i];
      return 2 * n;
    case V4L2_PIX_FMT_NV12: case V4L2_PIX_FMT_NV21: case V4L2_PIX_FMT_YUV420: case V4L2_PIX_FMT_YVU420:
      std::memcpy(out, g, n);
      std::memset(out + n, 128, n / 2);
      return n * 3 / 2;
    case V4L2_PIX_FMT_RGB24: case V4L2_PIX_FMT_BGR24:
      for (size_t i = 0; i < n; i++) out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = g[i];
      return 3 * n;
    default:
      return 0;
  }
}

struct Buf {
  std::shared_ptr<std::vector<uint8_t>> mem;
  uint32_t length = 0, offset = 0;
  int state = 0;                  // ai: 0 the program's, 1 queued, 2 done
  uint32_t bytesused = 0, flags = 0, sequence = 0;
  int64_t ns = 0;
  int maps = 0;
  bool cut = false;
};

struct Dev {
  size_t index = 0;
  FakeDevice cfg;
  Profile prof;
  std::string node, link, bus;
  std::mutex mu;
  std::condition_variable cv;
  std::map<uint32_t, int64_t> values;
  uint32_t fourcc = 0, w = 0, h = 0;
  Iv iv;
  std::vector<Buf> bufs;
  int holder = -1;                // ai: the fd whose REQBUFS holds the queue
  std::deque<uint32_t> queued, done;
  bool streaming = false, stop = false, unplugged = false, eio = false;
  std::thread producer;
  uint32_t seq = 0;
  std::mt19937 rng;
  std::unique_ptr<Scene> scene;   // ai: set at STREAMON, gone at STREAMOFF (the producer's alone between)
  std::unique_ptr<cam::Jpeg> jpeg;
  FakeServed served;
};

const CtlDef* ctlOf(const Dev& d, uint32_t id) {
  for (const CtlDef& c : d.prof.ctls)
    if (c.id == id) return &c;
  return nullptr;
}

// ai: QUERYCTRL's walk: the next control past id where NEXT_CTRL is set, else id itself
const CtlDef* walk(const Dev& d, uint32_t id) {
  if (!(id & V4L2_CTRL_FLAG_NEXT_CTRL)) return ctlOf(d, id);
  const uint32_t base = id & ~(V4L2_CTRL_FLAG_NEXT_CTRL | V4L2_CTRL_FLAG_NEXT_COMPOUND);
  for (const CtlDef& c : d.prof.ctls)
    if (c.id > base) return &c;
  return nullptr;
}

uint32_t flagsOf(const Dev& d, const CtlDef& c) {
  if (c.type == V4L2_CTRL_TYPE_CTRL_CLASS) return V4L2_CTRL_FLAG_READ_ONLY | V4L2_CTRL_FLAG_WRITE_ONLY;
  if (c.master) {
    auto it = d.values.find(c.master);
    if (it != d.values.end() && it->second != c.manual) return V4L2_CTRL_FLAG_INACTIVE;
  }
  return 0;
}

// ai: A value as the device takes it: a menu's index offered (else EINVAL), a number clamped and rounded to its step
int accept(const CtlDef& c, int64_t v, int64_t& out) {
  if (c.type == V4L2_CTRL_TYPE_CTRL_CLASS) return EACCES;
  if (c.type == V4L2_CTRL_TYPE_MENU) {
    for (auto& m : c.menu)
      if (m.first == v) {
        out = v;
        return 0;
      }
    return EINVAL;
  }
  const int64_t step = c.step > 0 ? c.step : 1;
  v = std::min(std::max(v, c.min), c.max);
  out = std::min(c.min + (v - c.min + step / 2) / step * step, c.max);
  return 0;
}

void fillBuf(const Dev& d, uint32_t i, v4l2_buffer& b) {
  const Buf& x = d.bufs[i];
  std::memset(&b, 0, sizeof b);
  b.index = i;
  b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  b.bytesused = x.bytesused;
  b.flags = x.flags | (d.prof.old ? 0 : V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_SOE);
  if (x.maps) b.flags |= V4L2_BUF_FLAG_MAPPED;
  if (x.state == 1) b.flags |= V4L2_BUF_FLAG_QUEUED;
  if (x.state == 2) b.flags |= V4L2_BUF_FLAG_DONE;
  b.field = V4L2_FIELD_NONE;
  if (!d.prof.old) {
    b.timestamp.tv_sec = x.ns / 1000000000;
    b.timestamp.tv_usec = (x.ns % 1000000000) / 1000;
  }
  b.sequence = x.sequence;
  b.memory = V4L2_MEMORY_MMAP;
  b.m.offset = x.offset;
  b.length = x.length;
}

// ai: The stream off: the producer stopped and joined (d.mu let go meanwhile), every buffer the program's again
void streamOff(Dev& d, std::unique_lock<std::mutex>& l) {
  d.stop = true;
  d.cv.notify_all();
  std::thread t = std::move(d.producer);
  if (t.joinable()) {
    l.unlock();
    t.join();
    l.lock();
  }
  d.streaming = false;
  d.stop = false;
  d.eio = false;
  for (Buf& b : d.bufs) b.state = 0;
  d.queued.clear();
  d.done.clear();
  d.scene.reset();
  d.jpeg.reset();
  d.cv.notify_all();
}

// ai: The frames: one a period of the device's clock, each filled outside the lock (the buffer is the producer's
// ai: between QBUF's queue and the done queue) and delivered an interval after its start
void produce(Dev* d) {
  std::vector<uint8_t> grey, jpg;
  std::unique_lock<std::mutex> l(d->mu);
  const FakeFaults& f = d->cfg.faults;
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const int64_t t0 = nowNs() + (int64_t)std::llround(d->cfg.offsetMs * 1e6);
  const double iv = 1e9 * d->iv.num / d->iv.den / (1 + d->cfg.ppm * 1e-6);
  uint64_t k = 0;
  while (!d->stop) {
    const int64_t start = t0 + (int64_t)std::llround((double)k * iv);
    const int64_t due = start + (int64_t)std::llround(iv);
    d->cv.wait_until(l, timeAt(due), [&] { return d->stop; });
    if (d->stop) break;
    const int64_t now = nowNs();
    if (now > due + (int64_t)(2 * iv)) {
      const uint64_t lost = (uint64_t)((double)(now - due) / iv);
      d->served.late += lost;
      d->seq += (uint32_t)lost;
      k += lost;
      continue;
    }
    k++;
    const uint32_t seq = d->seq++;
    if (f.unplugAfter >= 0 && (int64_t)d->served.streamFrames >= f.unplugAfter) {
      d->unplugged = true;
      d->cv.notify_all();
      break;
    }
    if (f.eioAfter >= 0 && (int64_t)d->served.streamFrames >= f.eioAfter) {
      d->eio = true;
      d->cv.notify_all();
      break;
    }
    if (f.skipped > 0 && U(d->rng) < f.skipped) {
      d->served.skipped++;
      continue;
    }
    if (d->queued.empty()) {
      d->served.starved++;
      continue;
    }
    const uint32_t bi = d->queued.front();
    d->queued.pop_front();
    const bool mjpeg = d->fourcc == V4L2_PIX_FMT_MJPEG;
    const bool err = f.errors > 0 && U(d->rng) < f.errors;
    const bool cut = mjpeg && f.truncated > 0 && U(d->rng) < f.truncated;
    const double keep = 0.1 + 0.5 * U(d->rng);
    std::shared_ptr<std::vector<uint8_t>> mem = d->bufs[bi].mem;
    const uint32_t fourcc = d->fourcc, w = d->w, h = d->h;
    l.unlock();
    grey.resize((size_t)w * h);
    if (d->scene) d->scene->render(start, grey.data(), w);
    else gradientInto(seq, w, h, grey.data());
    size_t used = pack(fourcc, grey.data(), w, h, mem->data(), mem->size(), d->jpeg.get(), d->cfg.quality, jpg);
    if (cut) used = std::max<size_t>(1, (size_t)((double)used * keep));
    l.lock();
    if (d->stop) break;
    Buf& b = d->bufs[bi];
    b.state = 2;
    b.bytesused = (uint32_t)used;
    b.flags = err ? V4L2_BUF_FLAG_ERROR : 0;
    b.sequence = seq;
    b.ns = start;
    b.cut = cut;
    d->done.push_back(bi);
    d->served.frames++;
    d->served.streamFrames++;
    if (err) d->served.errors++;
    if (cut) d->served.truncated++;
    d->cv.notify_all();
  }
}

}  // namespace

struct FakeV4l2::Impl {
  std::vector<std::unique_ptr<Dev>> devs;
  std::mutex fdMu;
  struct Open { Dev* dev; bool nonblock; };
  std::map<int, Open> fds;
  int nextFd = 1 << 20;           // ai: far from any real descriptor, so one handed to the kernel fails EBADF
  std::mutex mapMu;
  struct Mapping { std::shared_ptr<std::vector<uint8_t>> mem; Dev* dev; int count; };
  std::map<const void*, Mapping> maps;
  cam::V4l2Sys sys;

  Dev* devOf(int fd, bool* nonblock) {
    std::lock_guard<std::mutex> l(fdMu);
    auto it = fds.find(fd);
    if (it == fds.end()) return nullptr;
    if (nonblock) *nonblock = it->second.nonblock;
    return it->second.dev;
  }

  std::vector<std::string> nodes() {
    std::vector<std::string> out;
    for (auto& d : devs) {
      std::lock_guard<std::mutex> l(d->mu);
      if (!d->unplugged) out.push_back(d->node);
    }
    return out;
  }

  std::string byId(const std::string& node) {
    for (auto& d : devs)
      if (d->node == node) return d->link;
    return "";
  }

  int open(const char* path, int flags) {
    for (auto& d : devs) {
      if (path != d->node && path != d->link) continue;
      {
        std::lock_guard<std::mutex> l(d->mu);
        if (d->unplugged) return fail(ENOENT);
        d->served.opens++;
      }
      std::lock_guard<std::mutex> l(fdMu);
      const int fd = nextFd++;
      fds[fd] = Open{d.get(), (flags & O_NONBLOCK) != 0};
      return fd;
    }
    return fail(ENOENT);
  }

  int close(int fd) {
    Dev* d = nullptr;
    {
      std::lock_guard<std::mutex> l(fdMu);
      auto it = fds.find(fd);
      if (it == fds.end()) return fail(EBADF);
      d = it->second.dev;
      fds.erase(it);
    }
    std::unique_lock<std::mutex> l(d->mu);
    if (d->holder == fd) {
      streamOff(*d, l);
      d->bufs.clear();
      d->holder = -1;
    }
    return 0;
  }

  void* mmap(void*, size_t len, int, int flags, int fd, off_t off) {
    Dev* d = devOf(fd, nullptr);
    if (!d) return errno = EBADF, MAP_FAILED;
    std::lock_guard<std::mutex> l(d->mu);
    if (d->unplugged) return errno = ENODEV, MAP_FAILED;
    if (!(flags & MAP_SHARED)) return errno = EINVAL, MAP_FAILED;
    for (Buf& b : d->bufs) {
      if (b.offset != (uint32_t)off) continue;
      if (len > b.length) return errno = EINVAL, MAP_FAILED;
      b.maps++;
      std::lock_guard<std::mutex> m(mapMu);
      Mapping& x = maps[b.mem->data()];
      if (!x.count) x.mem = b.mem, x.dev = d;
      x.count++;
      return b.mem->data();
    }
    return errno = EINVAL, MAP_FAILED;
  }

  int munmap(void* addr, size_t) {
    std::shared_ptr<std::vector<uint8_t>> mem;
    Dev* d = nullptr;
    {
      std::lock_guard<std::mutex> m(mapMu);
      auto it = maps.find(addr);
      if (it == maps.end()) return fail(EINVAL);
      mem = it->second.mem;
      d = it->second.dev;
      if (--it->second.count == 0) maps.erase(it);
    }
    std::lock_guard<std::mutex> l(d->mu);
    for (Buf& b : d->bufs)
      if (b.mem == mem && b.maps > 0) {
        b.maps--;
        break;
      }
    return 0;
  }

  static short mask(const Dev& d, short events) {
    short r = 0;
    if (d.unplugged) r = POLLERR | POLLHUP | POLLPRI;
    else if (!d.streaming || d.eio) r = POLLERR;
    else if (!d.done.empty()) r = POLLIN | POLLRDNORM;
    return (short)(r & (events | POLLERR | POLLHUP | POLLNVAL));
  }

  int poll(pollfd* fds, nfds_t n, int timeoutMs) {
    using clock = std::chrono::steady_clock;
    const auto end = clock::now() + std::chrono::milliseconds(timeoutMs < 0 ? 86400000 : timeoutMs);
    for (;;) {
      int ready = 0;
      Dev* one = nullptr;
      for (nfds_t i = 0; i < n; i++) {
        fds[i].revents = 0;
        Dev* d = devOf(fds[i].fd, nullptr);
        if (!d) {
          fds[i].revents = POLLNVAL;
          ready++;
          continue;
        }
        std::lock_guard<std::mutex> l(d->mu);
        fds[i].revents = mask(*d, fds[i].events);
        if (fds[i].revents) ready++;
        one = d;
      }
      if (ready || timeoutMs == 0 || clock::now() >= end) return ready;
      if (n == 1 && one) {
        std::unique_lock<std::mutex> l(one->mu);
        one->cv.wait_until(l, end, [&] { return mask(*one, fds[0].events) != 0; });
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  int ctlIoctl(Dev& d, unsigned long req, void* arg);
  int ioctl(int fd, unsigned long req, void* arg);
};

// ai: The control ioctls (d.mu held)
int FakeV4l2::Impl::ctlIoctl(Dev& d, unsigned long req, void* arg) {
  if (d.prof.meta) return fail(ENOTTY);
  const bool ext = req == VIDIOC_QUERY_EXT_CTRL || req == VIDIOC_G_EXT_CTRLS || req == VIDIOC_S_EXT_CTRLS ||
                   req == VIDIOC_TRY_EXT_CTRLS;
  if (ext && d.prof.old) return fail(ENOTTY);
  switch (req) {
    case VIDIOC_QUERYCTRL: {
      auto* q = (v4l2_queryctrl*)arg;
      const CtlDef* c = walk(d, q->id);
      if (!c) return fail(EINVAL);
      std::memset(q, 0, sizeof *q);
      q->id = c->id;
      q->type = c->type;
      copy(q->name, c->name);
      q->minimum = (int32_t)c->min;
      q->maximum = (int32_t)c->max;
      q->step = (int32_t)c->step;
      q->default_value = (int32_t)c->def;
      q->flags = flagsOf(d, *c);
      return 0;
    }
    case VIDIOC_QUERY_EXT_CTRL: {
      auto* q = (v4l2_query_ext_ctrl*)arg;
      const CtlDef* c = walk(d, q->id);
      if (!c) return fail(EINVAL);
      std::memset(q, 0, sizeof *q);
      q->id = c->id;
      q->type = c->type;
      copy(q->name, c->name);
      q->minimum = c->min;
      q->maximum = c->max;
      q->step = (uint64_t)c->step;
      q->default_value = c->def;
      q->flags = flagsOf(d, *c);
      q->elem_size = 4;
      q->elems = 1;
      return 0;
    }
    case VIDIOC_QUERYMENU: {
      auto* m = (v4l2_querymenu*)arg;
      const CtlDef* c = ctlOf(d, m->id);
      if (!c || c->type != V4L2_CTRL_TYPE_MENU) return fail(EINVAL);
      for (auto& item : c->menu)
        if (item.first == (int64_t)m->index) {
          __u8 name[sizeof m->name];   // ai: v4l2_querymenu is packed: no reference to its field
          copy(name, item.second);
          std::memcpy(m->name, name, sizeof name);
          m->reserved = 0;
          return 0;
        }
      return fail(EINVAL);
    }
    case VIDIOC_G_CTRL: {
      auto* c = (v4l2_control*)arg;
      const CtlDef* x = ctlOf(d, c->id);
      if (!x) return fail(EINVAL);
      if (x->type == V4L2_CTRL_TYPE_CTRL_CLASS) return fail(EACCES);
      c->value = (int32_t)d.values[x->id];
      return 0;
    }
    case VIDIOC_S_CTRL: {
      auto* c = (v4l2_control*)arg;
      const CtlDef* x = ctlOf(d, c->id);
      if (!x) return fail(EINVAL);
      int64_t v;
      if (const int e = accept(*x, c->value, v)) return fail(e);
      if (x->master && d.values[x->master] != x->manual) return fail(EACCES);
      d.values[x->id] = v;
      c->value = (int32_t)v;
      return 0;
    }
    case VIDIOC_G_EXT_CTRLS: case VIDIOC_S_EXT_CTRLS: case VIDIOC_TRY_EXT_CTRLS: {
      auto* cs = (v4l2_ext_controls*)arg;
      if (cs->count && !cs->controls) return fail(EFAULT);
      const bool defaults = cs->which == V4L2_CTRL_WHICH_DEF_VAL;
      if (defaults && req != VIDIOC_G_EXT_CTRLS) return fail(EINVAL);
      // ai: validated whole before any is written; an absolute's master as this call leaves it (its last write here)
      std::vector<int64_t> vals(cs->count);
      for (uint32_t i = 0; i < cs->count; i++) {
        v4l2_ext_control& e = cs->controls[i];
        const CtlDef* x = ctlOf(d, e.id);
        cs->error_idx = i;
        if (!x) return fail(EINVAL);
        if (cs->which && !defaults && V4L2_CTRL_ID2WHICH(e.id) != cs->which) return fail(EINVAL);
        if (x->type == V4L2_CTRL_TYPE_CTRL_CLASS) return fail(EACCES);
        if (req == VIDIOC_G_EXT_CTRLS) continue;
        if (const int er = accept(*x, e.value, vals[i])) return fail(er);
        if (x->master) {
          int64_t mv = d.values[x->master];
          for (uint32_t j = cs->count; j-- > 0;)
            if (cs->controls[j].id == x->master) {
              mv = cs->controls[j].value;
              break;
            }
          if (mv != x->manual) return fail(EACCES);
        }
      }
      cs->error_idx = cs->count;
      for (uint32_t i = 0; i < cs->count; i++) {
        v4l2_ext_control& e = cs->controls[i];
        if (req == VIDIOC_G_EXT_CTRLS) {
          e.value = (int32_t)(defaults ? ctlOf(d, e.id)->def : d.values[e.id]);
          continue;
        }
        e.value = (int32_t)vals[i];
        if (req == VIDIOC_S_EXT_CTRLS) d.values[e.id] = vals[i];
      }
      return 0;
    }
    default:
      return fail(ENOTTY);
  }
}

int FakeV4l2::Impl::ioctl(int fd, unsigned long req, void* arg) {
  bool nb = false;
  Dev* dp = devOf(fd, &nb);
  if (!dp) return fail(EBADF);
  Dev& d = *dp;
  std::unique_lock<std::mutex> l(d.mu);
  if (d.unplugged) return fail(ENODEV);
  auto holds = [&]() { return d.holder == fd ? 0 : d.holder < 0 ? EINVAL : EBUSY; };
  if (d.prof.bare && (req == VIDIOC_ENUM_FRAMESIZES || req == VIDIOC_ENUM_FRAMEINTERVALS || req == VIDIOC_G_PARM ||
                      req == VIDIOC_S_PARM))
    return fail(ENOTTY);
  switch (req) {
    case VIDIOC_QUERYCAP: {
      auto* c = (v4l2_capability*)arg;
      std::memset(c, 0, sizeof *c);
      copy(c->driver, d.prof.driver);
      copy(c->card, d.prof.card);
      copy(c->bus_info, d.bus);
      c->version = 7u << 16;
      const uint32_t all = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_META_CAPTURE | V4L2_CAP_STREAMING |
                           V4L2_CAP_EXT_PIX_FORMAT | V4L2_CAP_DEVICE_CAPS;
      if (d.prof.old) {
        c->capabilities = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
      } else {
        c->capabilities = all;
        c->device_caps = (d.prof.meta ? V4L2_CAP_META_CAPTURE : V4L2_CAP_VIDEO_CAPTURE) | V4L2_CAP_STREAMING |
                         V4L2_CAP_EXT_PIX_FORMAT;
      }
      return 0;
    }
    case VIDIOC_ENUM_FMT: {
      auto* f = (v4l2_fmtdesc*)arg;
      if (f->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || f->index >= d.prof.formats.size()) return fail(EINVAL);
      const Format& x = d.prof.formats[f->index];
      const uint32_t index = f->index;
      std::memset(f, 0, sizeof *f);
      f->index = index;
      f->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      f->flags = x.compressed ? V4L2_FMT_FLAG_COMPRESSED : 0;
      copy(f->description, x.desc);
      f->pixelformat = x.fourcc;
      return 0;
    }
    case VIDIOC_ENUM_FRAMESIZES: {
      auto* s = (v4l2_frmsizeenum*)arg;
      const Format* x = formatOf(d.prof, s->pixel_format);
      if (!x) return fail(EINVAL);
      if (x->stepwise) {
        if (s->index) return fail(EINVAL);
        s->type = V4L2_FRMSIZE_TYPE_STEPWISE;
        s->stepwise = x->range;
        return 0;
      }
      if (s->index >= x->sizes.size()) return fail(EINVAL);
      s->type = V4L2_FRMSIZE_TYPE_DISCRETE;
      s->discrete.width = x->sizes[s->index].w;
      s->discrete.height = x->sizes[s->index].h;
      return 0;
    }
    case VIDIOC_ENUM_FRAMEINTERVALS: {
      auto* v = (v4l2_frmivalenum*)arg;
      const Format* x = formatOf(d.prof, v->pixel_format);
      if (!x) return fail(EINVAL);
      if (x->stepwise) {
        if (v->index || !inRange(x->range, v->width, v->height)) return fail(EINVAL);
        v->type = V4L2_FRMIVAL_TYPE_CONTINUOUS;
        v->stepwise.min = v4l2_fract{x->fastest.num, x->fastest.den};
        v->stepwise.max = v4l2_fract{x->slowest.num, x->slowest.den};
        v->stepwise.step = v4l2_fract{1, 1};
        return 0;
      }
      const Size* z = sizeOf(*x, v->width, v->height);
      if (!z || v->index >= z->iv.size()) return fail(EINVAL);
      v->type = V4L2_FRMIVAL_TYPE_DISCRETE;
      v->discrete = v4l2_fract{z->iv[v->index].num, z->iv[v->index].den};
      return 0;
    }
    case VIDIOC_G_FMT: {
      auto* f = (v4l2_format*)arg;
      if (f->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || d.prof.formats.empty()) return fail(EINVAL);
      fillPix(d.fourcc, d.w, d.h, f->fmt.pix);
      return 0;
    }
    case VIDIOC_S_FMT: case VIDIOC_TRY_FMT: {
      auto* f = (v4l2_format*)arg;
      if (f->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || d.prof.formats.empty()) return fail(EINVAL);
      const Format* x = formatOf(d.prof, f->fmt.pix.pixelformat);
      if (!x) x = &d.prof.formats.front();
      uint32_t w = 0, h = 0;
      nearestSize(*x, f->fmt.pix.width, f->fmt.pix.height, w, h);
      if (req == VIDIOC_S_FMT) {
        if (d.holder >= 0 && d.holder != fd) return fail(EBUSY);
        if (d.streaming || !d.bufs.empty()) return fail(EBUSY);
        d.fourcc = x->fourcc;
        d.w = w;
        d.h = h;
        d.iv = nearestIv(*x, w, h, d.iv);
      }
      fillPix(x->fourcc, w, h, f->fmt.pix);
      return 0;
    }
    case VIDIOC_G_PARM: case VIDIOC_S_PARM: {
      auto* p = (v4l2_streamparm*)arg;
      if (p->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || d.prof.formats.empty()) return fail(EINVAL);
      if (req == VIDIOC_S_PARM) {
        if (d.streaming) return fail(EBUSY);
        const Format* x = formatOf(d.prof, d.fourcc);
        if (!x) return fail(EINVAL);
        const v4l2_fract& t = p->parm.capture.timeperframe;
        d.iv = nearestIv(*x, d.w, d.h, Iv{t.numerator, t.denominator});
      }
      std::memset(&p->parm, 0, sizeof p->parm);
      p->parm.capture.capability = V4L2_CAP_TIMEPERFRAME;
      p->parm.capture.timeperframe = v4l2_fract{d.iv.num, d.iv.den};
      return 0;
    }
    case VIDIOC_QUERYCTRL: case VIDIOC_QUERY_EXT_CTRL: case VIDIOC_QUERYMENU: case VIDIOC_G_CTRL: case VIDIOC_S_CTRL:
    case VIDIOC_G_EXT_CTRLS: case VIDIOC_S_EXT_CTRLS: case VIDIOC_TRY_EXT_CTRLS:
      return ctlIoctl(d, req, arg);
    case VIDIOC_REQBUFS: {
      auto* r = (v4l2_requestbuffers*)arg;
      if (r->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || r->memory != V4L2_MEMORY_MMAP || d.prof.formats.empty())
        return fail(EINVAL);
      if (d.holder >= 0 && d.holder != fd) return fail(EBUSY);
      if (d.streaming) return fail(EBUSY);
      d.bufs.clear();
      d.queued.clear();
      d.done.clear();
      if (!r->count) {
        d.holder = -1;
        return 0;
      }
      const uint32_t n = std::min(std::max(r->count, 2u), 8u);
      const uint32_t len = (imageOf(d.fourcc, d.w, d.h) + 4095) / 4096 * 4096;
      d.bufs.resize(n);
      for (uint32_t i = 0; i < n; i++) {
        d.bufs[i].mem = std::make_shared<std::vector<uint8_t>>(len);
        d.bufs[i].length = len;
        d.bufs[i].offset = i << 25;
      }
      d.holder = fd;
      r->count = n;
      r->capabilities = V4L2_BUF_CAP_SUPPORTS_MMAP;
      return 0;
    }
    case VIDIOC_QUERYBUF: {
      auto* b = (v4l2_buffer*)arg;
      if (b->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || b->index >= d.bufs.size()) return fail(EINVAL);
      fillBuf(d, b->index, *b);
      return 0;
    }
    case VIDIOC_QBUF: {
      auto* b = (v4l2_buffer*)arg;
      if (b->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || b->memory != V4L2_MEMORY_MMAP) return fail(EINVAL);
      if (const int e = holds()) return fail(e);
      if (d.eio) return fail(EIO);
      if (b->index >= d.bufs.size() || d.bufs[b->index].state != 0) return fail(EINVAL);
      Buf& x = d.bufs[b->index];
      x.state = 1;
      x.bytesused = 0;
      x.flags = 0;
      x.cut = false;
      d.queued.push_back(b->index);
      fillBuf(d, b->index, *b);
      return 0;
    }
    case VIDIOC_DQBUF: {
      auto* b = (v4l2_buffer*)arg;
      if (b->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || b->memory != V4L2_MEMORY_MMAP) return fail(EINVAL);
      if (const int e = holds()) return fail(e);
      for (;;) {
        if (d.unplugged) return fail(ENODEV);
        if (d.eio) return fail(EIO);
        if (!d.streaming) return fail(EINVAL);
        if (!d.done.empty()) break;
        if (nb) return fail(EAGAIN);
        d.cv.wait(l);
      }
      const uint32_t i = d.done.front();
      d.done.pop_front();
      fillBuf(d, i, *b);
      Buf& x = d.bufs[i];
      x.state = 0;
      d.served.dequeued++;
      d.served.streamDequeued++;
      if (x.flags & V4L2_BUF_FLAG_ERROR) d.served.dequeuedErrors++;
      if (x.cut) d.served.dequeuedTruncated++;
      if (d.served.firstSeq < 0) d.served.firstSeq = x.sequence;
      d.served.lastSeq = x.sequence;
      return 0;
    }
    case VIDIOC_STREAMON: {
      if (*(int*)arg != V4L2_BUF_TYPE_VIDEO_CAPTURE) return fail(EINVAL);
      if (const int e = holds()) return fail(e);
      if (d.streaming) return 0;
      if (d.bufs.empty()) return fail(EINVAL);
      if (d.producer.joinable()) streamOff(d, l);
      d.served.error.clear();
      if (!d.cfg.scene.empty()) {
        Sender s;
        Display dp;
        Optics o;
        std::string why;
        if (!parse(d.cfg.scene, s, dp, o, why)) {
          d.served.error = "scene: " + why;
          return fail(EIO);
        }
        o.width = d.w;
        o.height = d.h;
        d.scene = Scene::create(s, dp, o, 2, why);
        if (!d.scene) {
          d.served.error = "scene: " + why;
          return fail(EIO);
        }
      }
      if (d.fourcc == V4L2_PIX_FMT_MJPEG) {
        std::string why;
        d.jpeg = cam::Jpeg::load(why);
        if (!d.jpeg) {
          d.served.error = "MJPEG: " + why;
          d.scene.reset();
          return fail(EIO);
        }
      }
      d.streaming = true;
      d.stop = false;
      d.eio = false;
      d.seq = 0;
      d.served.streams++;
      d.served.firstSeq = d.served.lastSeq = -1;
      d.served.streamFrames = d.served.streamDequeued = 0;
      d.producer = std::thread(produce, &d);
      return 0;
    }
    case VIDIOC_STREAMOFF: {
      if (*(int*)arg != V4L2_BUF_TYPE_VIDEO_CAPTURE) return fail(EINVAL);
      if (const int e = holds()) return fail(e);
      streamOff(d, l);
      return 0;
    }
    default:
      return fail(ENOTTY);
  }
}

FakeV4l2::FakeV4l2(const std::vector<FakeDevice>& devices) : p_(std::make_unique<Impl>()) {
  std::map<std::string, int> seen;
  for (size_t i = 0; i < devices.size(); i++) {
    auto d = std::make_unique<Dev>();
    d->index = i;
    d->cfg = devices[i];
    d->prof = profileOf(devices[i].profile);
    d->node = "/dev/video" + std::to_string(90 + i);
    const int again = seen[devices[i].profile]++;
    d->link = "/dev/v4l/by-id/usb-LIZARD_Fake_" + devices[i].profile + (again ? "_" + std::to_string(again) : "") +
              (d->prof.meta ? "-video-index1" : "-video-index0");
    d->bus = "usb-0000:00:14.0-9." + std::to_string(i + 1);
    d->rng.seed(devices[i].faults.seed * 7919u + (uint32_t)i);
    for (const CtlDef& c : d->prof.ctls)
      if (c.type != V4L2_CTRL_TYPE_CTRL_CLASS) d->values[c.id] = c.def;
    if (!d->prof.formats.empty()) {
      const Format& f = d->prof.formats.front();
      d->fourcc = f.fourcc;
      if (f.stepwise) {
        d->w = f.range.max_width;
        d->h = f.range.max_height;
        d->iv = f.fastest;
      } else {
        d->w = f.sizes.front().w;
        d->h = f.sizes.front().h;
        d->iv = f.sizes.front().iv.front();
      }
    }
    p_->devs.push_back(std::move(d));
  }
  Impl* im = p_.get();
  p_->sys.nodes = [im]() { return im->nodes(); };
  p_->sys.open = [im](const char* path, int flags) { return im->open(path, flags); };
  p_->sys.close = [im](int fd) { return im->close(fd); };
  p_->sys.ioctl = [im](int fd, unsigned long req, void* arg) { return im->ioctl(fd, req, arg); };
  p_->sys.mmap = [im](void* a, size_t n, int prot, int flags, int fd, off_t off) {
    return im->mmap(a, n, prot, flags, fd, off);
  };
  p_->sys.munmap = [im](void* a, size_t n) { return im->munmap(a, n); };
  p_->sys.poll = [im](pollfd* f, nfds_t n, int t) { return im->poll(f, n, t); };
  p_->sys.byId = [im](const std::string& node) { return im->byId(node); };
}

FakeV4l2::~FakeV4l2() {
  for (auto& d : p_->devs) {
    std::unique_lock<std::mutex> l(d->mu);
    streamOff(*d, l);
  }
}

const cam::V4l2Sys* FakeV4l2::sys() const { return &p_->sys; }
size_t FakeV4l2::size() const { return p_->devs.size(); }
std::string FakeV4l2::node(size_t i) const { return p_->devs.at(i)->node; }
std::string FakeV4l2::byId(size_t i) const { return p_->devs.at(i)->link; }

bool FakeV4l2::control(size_t i, uint32_t id, int64_t& value) const {
  Dev& d = *p_->devs.at(i);
  std::lock_guard<std::mutex> l(d.mu);
  auto it = d.values.find(id);
  if (it == d.values.end()) return false;
  value = it->second;
  return true;
}

bool FakeV4l2::setControl(size_t i, uint32_t id, int64_t value) {
  Dev& d = *p_->devs.at(i);
  std::lock_guard<std::mutex> l(d.mu);
  auto it = d.values.find(id);
  if (it == d.values.end()) return false;
  it->second = value;
  return true;
}

FakeServed FakeV4l2::served(size_t i) const {
  Dev& d = *p_->devs.at(i);
  std::lock_guard<std::mutex> l(d.mu);
  FakeServed s = d.served;
  s.streaming = d.streaming;
  s.unplugged = d.unplugged;
  s.eio = d.eio;
  s.fourcc = d.fourcc;
  s.width = d.w;
  s.height = d.h;
  s.num = d.iv.num;
  s.den = d.iv.den;
  s.buffers = (uint32_t)d.bufs.size();
  std::lock_guard<std::mutex> m(p_->mapMu);
  s.mapped = 0;
  for (auto& x : p_->maps)
    if (x.second.dev == &d) s.mapped += (uint32_t)x.second.count;
  return s;
}

uint8_t FakeV4l2::gradient(uint64_t seq, uint32_t x, uint32_t y) {
  const uint64_t v = ((uint64_t)x + y + 7 * seq) % 510;
  return (uint8_t)(v < 255 ? v : 509 - v);
}

}  // namespace lizard::sim
