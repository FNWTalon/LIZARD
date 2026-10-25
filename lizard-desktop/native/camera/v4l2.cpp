// ai: V4L2 capture devices (2026-10-25): UVC webcams and every other node that captures video and streams, through the
// ai: calls v4l2_sys.h tables (the kernel's, or a test's fake). Listing opens each node the table names, keeps the ones
// ai: whose device caps capture video and stream (a UVC camera's metadata node is skipped) and closes it again. A
// ai: device's modes are its formats' sizes and intervals as the driver enumerates them: a stepwise or continuous size
// ai: range as its largest size and the common sizes the steps hold, a stepwise or continuous interval as one mode at
// ai: its fastest rate. Its controls are the ones camera.h names, exposure in us. start sets the format and the frame
// ai: interval and reads both answers back (a size or format the driver changed is refused, naming both), applies the
// ai: camera rules (sharpness and backlight compensation at their minimum, exposure_auto_priority 0 so the frame rate
// ai: holds while the light drops; each only where the camera lists it), maps 4 buffers (at least 2) and streams on a
// ai: thread of its own: poll 200 ms, dequeue, the frame's luma (the Y plane in place where luma.h takes it so, else
// ai: converted; MJPEG decoded by a TurboJPEG of the thread's own, a frame cut in its scan taken as jpeg.h decodes
// ai: it) to the sink, queue it again. A frame the driver flags as an error, or shorter than its layout's luma (an
// ai: MJPEG frame: under 16 bytes; one that does not decode), is counted corrupt, a gap in the sequence dropped;
// ai: a dequeue or queue the device refuses (ENODEV: unplugged; EIO: its error; anything else but EAGAIN) ends the
// ai: stream, ended told once on the capture thread. stop takes the stream off (which wakes the poll), joins the
// ai: thread, unmaps and frees the buffers, then puts back every control the camera changed (the rules, set), each
// ai: remembered with its value before: a UVC device keeps its controls for the next program.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/ioctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>

#include "backends.h"
#include "jpeg.h"
#include "luma.h"
#include "trigger.h"
#include "v4l2_sys.h"

namespace lizard::cam {

namespace {

// ai: The kernel's table. devNodes: every /dev/video* node by number, the probe's default (probedNodes, v4l2_sys.h).
std::vector<std::string> devNodes() {
  std::vector<std::string> out;
  std::vector<std::pair<long, std::string>> found;
  if (DIR* d = opendir("/dev")) {
    while (dirent* e = readdir(d)) {
      const char* n = e->d_name;
      if (std::strncmp(n, "video", 5) || !n[5]) continue;
      char* end = nullptr;
      const long k = std::strtol(n + 5, &end, 10);
      if (*end) continue;
      found.emplace_back(k, std::string("/dev/") + n);
    }
    closedir(d);
  }
  std::sort(found.begin(), found.end());
  for (auto& f : found) out.push_back(f.second);
  return out;
}

std::vector<std::string> kernelNodes() { return probedNodes(std::getenv("LIZ_V4L2_NODES"), devNodes); }

// ai: The /dev/v4l/by-id link that resolves to node (udev's: usb-<vendor>_<product>_<serial>-video-index<k>), the
// ai: first by name where several do; "" none.
std::string kernelById(const std::string& node) {
  char real[PATH_MAX];
  if (!realpath(node.c_str(), real)) return "";
  std::vector<std::string> hits;
  if (DIR* d = opendir("/dev/v4l/by-id")) {
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      const std::string p = std::string("/dev/v4l/by-id/") + e->d_name;
      char r[PATH_MAX];
      if (realpath(p.c_str(), r) && !std::strcmp(r, real)) hits.push_back(p);
    }
    closedir(d);
  }
  std::sort(hits.begin(), hits.end());
  return hits.empty() ? "" : hits.front();
}

const V4l2Sys& kernelSys() {
  static const V4l2Sys s = {
      kernelNodes,
      [](const char* path, int flags) { return ::open(path, flags); },
      [](int fd) { return ::close(fd); },
      [](int fd, unsigned long request, void* arg) { return ::ioctl(fd, request, arg); },
      [](void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
        return ::mmap(addr, length, prot, flags, fd, offset);
      },
      [](void* addr, size_t length) { return ::munmap(addr, length); },
      [](pollfd* fds, nfds_t n, int timeoutMs) { return ::poll(fds, n, timeoutMs); },
      kernelById,
  };
  return s;
}

std::atomic<const V4l2Sys*> gSys{nullptr};

int xioctl(const V4l2Sys& s, int fd, unsigned long request, void* arg) {
  int r;
  do r = s.ioctl(fd, request, arg);
  while (r < 0 && errno == EINTR);
  return r;
}

std::string err(int e) { return std::strerror(e); }

// ai: A fourcc as text, its trailing spaces cut ("Y16 " is Y16); a big-endian variant's flag (bit 31) as "-BE"
std::string fourccText(uint32_t f) {
  std::string s;
  for (int i = 0; i < 4; i++) {
    const char c = (char)((f >> (8 * i)) & (i == 3 ? 0x7f : 0xff));
    s += (c >= 32 && c < 127) ? c : '?';
  }
  while (!s.empty() && s.back() == ' ') s.pop_back();
  if (f & (1u << 31)) s += "-BE";
  return s;
}

std::string sizeText(uint32_t w, uint32_t h) { return std::to_string(w) + "x" + std::to_string(h); }

// ai: The formats taken, as camera.h's Pix: any other a device lists is skipped (H.264, the 10-bit Bayer layouts...)
struct PixOf { uint32_t fourcc; Pix pix; uint8_t bits; uint8_t bpp; };   // ai: bpp: the first plane's bytes a pixel
const PixOf kPix[] = {
    {V4L2_PIX_FMT_GREY, Pix::Grey8, 8, 1},     {V4L2_PIX_FMT_Y16, Pix::Grey16, 16, 2},
    {V4L2_PIX_FMT_Y10, Pix::Grey16, 10, 2},    {V4L2_PIX_FMT_Y12, Pix::Grey16, 12, 2},
    {V4L2_PIX_FMT_YUYV, Pix::YUYV, 8, 2},      {V4L2_PIX_FMT_UYVY, Pix::UYVY, 8, 2},
    {V4L2_PIX_FMT_YVYU, Pix::YVYU, 8, 2},      {V4L2_PIX_FMT_VYUY, Pix::VYUY, 8, 2},
    {V4L2_PIX_FMT_NV12, Pix::NV12, 8, 1},      {V4L2_PIX_FMT_NV21, Pix::NV21, 8, 1},
    {V4L2_PIX_FMT_YUV420, Pix::I420, 8, 1},    {V4L2_PIX_FMT_YVU420, Pix::YV12, 8, 1},
    {V4L2_PIX_FMT_MJPEG, Pix::MJPEG, 8, 0},    {V4L2_PIX_FMT_JPEG, Pix::MJPEG, 8, 0},
    {V4L2_PIX_FMT_SBGGR8, Pix::Bayer8, 8, 1},  {V4L2_PIX_FMT_SGBRG8, Pix::Bayer8, 8, 1},
    {V4L2_PIX_FMT_SGRBG8, Pix::Bayer8, 8, 1},  {V4L2_PIX_FMT_SRGGB8, Pix::Bayer8, 8, 1},
    {V4L2_PIX_FMT_RGB24, Pix::RGB24, 8, 3},    {V4L2_PIX_FMT_BGR24, Pix::BGR24, 8, 3},
};

const PixOf* pixOf(uint32_t fourcc) {
  for (const PixOf& p : kPix)
    if (p.fourcc == fourcc) return &p;
  return nullptr;
}

// ai: The controls camera.h names, by their V4L2 ids: exposure_absolute counts 100 us, the rest are the camera's units.
// ai: autoId: the control that makes it automatic (exposure_auto's menu, focus_automatic_continuous).
struct CtlOf { const char* id; uint32_t cid; uint32_t autoId; double scale; const char* unit; };
const CtlOf kCtl[] = {
    {"zoom", V4L2_CID_ZOOM_ABSOLUTE, 0, 1, ""},
    {"focus", V4L2_CID_FOCUS_ABSOLUTE, V4L2_CID_FOCUS_AUTO, 1, ""},
    {"exposure", V4L2_CID_EXPOSURE_ABSOLUTE, V4L2_CID_EXPOSURE_AUTO, 100, "us"},
    {"gain", V4L2_CID_GAIN, 0, 1, ""},
    {"sharpness", V4L2_CID_SHARPNESS, 0, 1, ""},
    {"backlight", V4L2_CID_BACKLIGHT_COMPENSATION, 0, 1, ""},
    {"brightness", V4L2_CID_BRIGHTNESS, 0, 1, ""},
    {"contrast", V4L2_CID_CONTRAST, 0, 1, ""},
};

// ai: A value's manual side for a control that makes another automatic: exposure_auto's MANUAL, focus_auto off
int64_t manualOf(uint32_t autoId) { return autoId == V4L2_CID_EXPOSURE_AUTO ? V4L2_EXPOSURE_MANUAL : 0; }
// ai: The control that makes this one automatic, 0 none
uint32_t masterOf(uint32_t id) {
  if (id == V4L2_CID_EXPOSURE_ABSOLUTE) return V4L2_CID_EXPOSURE_AUTO;
  return id == V4L2_CID_FOCUS_ABSOLUTE ? V4L2_CID_FOCUS_AUTO : 0;
}
bool isMaster(uint32_t id) { return id == V4L2_CID_EXPOSURE_AUTO || id == V4L2_CID_FOCUS_AUTO; }

// ai: A control as the driver lists it; menu: the indexes a menu offers (a UVC menu is sparse: exposure_auto on a
// ai: C930e offers manual and aperture priority alone)
struct Ctl {
  uint32_t id = 0, type = 0, flags = 0;
  int64_t min = 0, max = 0, step = 1, def = 0;
  std::string name;
  std::vector<int64_t> menu;
};

// ai: The size of a stepwise or continuous range that its steps hold
bool fits(const v4l2_frmsize_stepwise& s, uint32_t w, uint32_t h) {
  if (w < s.min_width || w > s.max_width || h < s.min_height || h > s.max_height) return false;
  if (s.step_width && (w - s.min_width) % s.step_width) return false;
  if (s.step_height && (h - s.min_height) % s.step_height) return false;
  return true;
}

double fpsOf(const v4l2_fract& f) { return f.numerator && f.denominator ? (double)f.denominator / f.numerator : 0; }

// ai: A frame duration for fps, as V4L2 takes it (the driver answers the nearest interval it has)
v4l2_fract fractOf(double fps) {
  uint64_t n = 1000000, d = (uint64_t)std::llround(fps * 1e6);
  const uint64_t g = std::gcd(n, d);
  if (g) n /= g, d /= g;
  v4l2_fract f{};
  f.numerator = (uint32_t)n;
  f.denominator = (uint32_t)d;
  return f;
}

int64_t timevalNs(const timeval& t) { return (int64_t)t.tv_sec * 1000000000 + (int64_t)t.tv_usec * 1000; }

bool queryCap(const V4l2Sys& s, int fd, v4l2_capability& c) {
  std::memset(&c, 0, sizeof c);
  return xioctl(s, fd, VIDIOC_QUERYCAP, &c) == 0;
}

// ai: The node's own caps (device_caps where the driver sets them; the whole device's otherwise, an old driver's)
uint32_t capsOf(const v4l2_capability& c) {
  return (c.capabilities & V4L2_CAP_DEVICE_CAPS) ? c.device_caps : c.capabilities;
}
bool captures(const v4l2_capability& c) {
  const uint32_t k = capsOf(c);
  return (k & V4L2_CAP_VIDEO_CAPTURE) && (k & V4L2_CAP_STREAMING);
}

std::string text(const __u8* s, size_t n) { return std::string((const char*)s, strnlen((const char*)s, n)); }

Info infoOf(const V4l2Sys& s, const std::string& node, const v4l2_capability& c) {
  Info i;
  const std::string link = s.byId(node);
  i.id = "v4l2:" + (link.empty() ? node : link);
  i.name = text(c.card, sizeof c.card);
  i.backend = "uvc";
  i.detail = text(c.bus_info, sizeof c.bus_info) + ", " + text(c.driver, sizeof c.driver);
  return i;
}

// ai: An MJPEG frame of fewer bytes holds no picture (a JPEG's markers alone take more)
constexpr size_t kJpegMin = 16;
constexpr unsigned kBuffers = 4;

class V4l2Camera final : public Camera {
 public:
  V4l2Camera(const V4l2Sys& sys, int fd, Info info) : sys_(sys), fd_(fd), info_(std::move(info)) {}
  ~V4l2Camera() override {
    stop();
    if (th_.joinable()) {
      if (onCapture()) th_.detach();
      else th_.join();
    }
    sys_.close(fd_);
  }

  const Info& info() const override { return info_; }
  std::vector<Mode> modes() override;
  std::vector<Control> controls() override;
  bool set(const std::string& id, double value, bool automatic, std::string& why) override;
  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override;
  void stop() override;
  Stats stats() override {
    std::lock_guard<std::mutex> l(statMu_);
    return st_;
  }

 private:
  struct Buf { void* p = nullptr; size_t len = 0; };

  int io(unsigned long request, void* arg) { return xioctl(sys_, fd_, request, arg); }
  bool onCapture() const { return capId_.load() == std::this_thread::get_id(); }
  void run(Sink sink, Ended ended, std::unique_ptr<Jpeg> jpeg);
  void teardown();
  void unmap();
  // ai: ctlMu_ held
  void loadCtls();
  const Ctl* find(uint32_t id) const;
  bool get(const Ctl& c, int64_t& v);
  bool put(const std::vector<std::pair<const Ctl*, int64_t>>& w, std::string& why);
  void remember(const Ctl& c);
  bool remembered(uint32_t id) const;
  bool autoOn(const Ctl& a, int64_t& on) const;
  // ai: exposure counts as automatic in its full automatic and in aperture priority (its time the camera's), not in
  // ai: shutter priority (its time manual)
  bool isAuto(const Ctl& a, int64_t v) const {
    if (a.id != V4L2_CID_EXPOSURE_AUTO) return v != 0;
    return v == V4L2_EXPOSURE_AUTO || v == V4L2_EXPOSURE_APERTURE_PRIORITY;
  }
  bool offers(const Ctl& c, int64_t v) const { return std::find(c.menu.begin(), c.menu.end(), v) != c.menu.end(); }
  static int64_t snap(const Ctl& c, double v);
  void rules();
  void restore();

  const V4l2Sys& sys_;
  const int fd_;
  const Info info_;
  std::mutex modeMu_;
  std::vector<Mode> modes_;
  bool haveModes_ = false;
  std::mutex ctlMu_;
  std::map<uint32_t, Ctl> ctls_;               // ai: ctlMu_
  bool haveCtls_ = false;                      // ai: ctlMu_
  std::vector<std::pair<uint32_t, int64_t>> changed_;   // ai: ctlMu_: each control changed, its value before, in order
  std::mutex runMu_;                           // ai: start and stop
  std::thread th_;
  std::atomic<std::thread::id> capId_{};       // ai: the capture thread's while it is not joined (runMu_ writes it)
  std::atomic<bool> running_{false};           // ai: the capture thread is in its loop (false once a stream ends)
  std::atomic<bool> stopping_{false};
  bool streaming_ = false;                     // ai: runMu_: buffers mapped and the stream on (or ended by itself)
  std::vector<Buf> bufs_;
  Pix pix_ = Pix::Unknown;
  uint8_t bits_ = 8;
  uint32_t w_ = 0, h_ = 0, stride_ = 0;
  std::mutex statMu_;
  Stats st_;                                   // ai: statMu_
};

std::vector<Mode> V4l2Camera::modes() {
  std::lock_guard<std::mutex> l(modeMu_);
  if (haveModes_) return modes_;
  std::vector<Mode> out;
  for (uint32_t i = 0;; i++) {
    v4l2_fmtdesc d{};
    d.index = i;
    d.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (io(VIDIOC_ENUM_FMT, &d) < 0) break;
    const PixOf* p = pixOf(d.pixelformat);
    if (!p) continue;
    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    for (uint32_t j = 0;; j++) {
      v4l2_frmsizeenum s{};
      s.index = j;
      s.pixel_format = d.pixelformat;
      if (io(VIDIOC_ENUM_FRAMESIZES, &s) < 0) break;
      if (s.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
        sizes.emplace_back(s.discrete.width, s.discrete.height);
        continue;
      }
      // ai: a range (stepwise, continuous: index 0 alone): its largest, then the common sizes its steps hold
      const v4l2_frmsize_stepwise& r = s.stepwise;
      sizes.emplace_back(r.max_width, r.max_height);
      const std::pair<uint32_t, uint32_t> common[] = {{1920, 1080}, {1280, 720}, {640, 480}};
      for (auto& c : common)
        if (fits(r, c.first, c.second) && c != sizes.front()) sizes.push_back(c);
      break;
    }
    // ai: a driver that enumerates no size (an old capture card's): the sizes TRY_FMT answers for the common ones
    if (sizes.empty()) {
      const std::pair<uint32_t, uint32_t> common[] = {{1920, 1080}, {1280, 720}, {640, 480}};
      for (auto& c : common) {
        v4l2_format t{};
        t.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        t.fmt.pix.width = c.first;
        t.fmt.pix.height = c.second;
        t.fmt.pix.pixelformat = d.pixelformat;
        t.fmt.pix.field = V4L2_FIELD_NONE;
        if (io(VIDIOC_TRY_FMT, &t) < 0 || t.fmt.pix.pixelformat != d.pixelformat || !t.fmt.pix.width) continue;
        const std::pair<uint32_t, uint32_t> got{t.fmt.pix.width, t.fmt.pix.height};
        if (std::find(sizes.begin(), sizes.end(), got) == sizes.end()) sizes.push_back(got);
      }
    }
    for (auto& sz : sizes) {
      Mode m;
      m.width = sz.first;
      m.height = sz.second;
      m.format = fourccText(d.pixelformat);
      m.pix = p->pix;
      m.bits = p->bits;
      m.compressed = p->pix == Pix::MJPEG;
      m.key = d.pixelformat;
      bool any = false;
      for (uint32_t k = 0;; k++) {
        v4l2_frmivalenum v{};
        v.index = k;
        v.pixel_format = d.pixelformat;
        v.width = sz.first;
        v.height = sz.second;
        if (io(VIDIOC_ENUM_FRAMEINTERVALS, &v) < 0) break;
        if (v.type == V4L2_FRMIVAL_TYPE_DISCRETE) {
          m.fps = m.fpsMin = fpsOf(v.discrete);
          if (m.fps > 0) out.push_back(m), any = true;
          continue;
        }
        // ai: a range: one mode, at its fastest (its shortest interval), its slowest beside it
        m.fps = fpsOf(v.stepwise.min);
        m.fpsMin = fpsOf(v.stepwise.max);
        out.push_back(m);
        any = true;
        break;
      }
      // ai: a driver that lists no interval: the size at a rate it does not say
      if (!any) {
        m.fps = m.fpsMin = 0;
        out.push_back(m);
      }
    }
  }
  modes_ = out;
  haveModes_ = true;
  return out;
}

void V4l2Camera::loadCtls() {
  if (haveCtls_) return;
  haveCtls_ = true;
  auto add = [&](uint32_t id, uint32_t type, uint32_t flags, int64_t min, int64_t max, int64_t step, int64_t def,
                 const char* name, size_t nameLen) {
    if (flags & V4L2_CTRL_FLAG_DISABLED) return;
    if (type != V4L2_CTRL_TYPE_INTEGER && type != V4L2_CTRL_TYPE_BOOLEAN && type != V4L2_CTRL_TYPE_MENU &&
        type != V4L2_CTRL_TYPE_INTEGER_MENU && type != V4L2_CTRL_TYPE_INTEGER64)
      return;
    Ctl c;
    c.id = id;
    c.type = type;
    c.flags = flags;
    c.min = min;
    c.max = max;
    c.step = step > 0 ? step : 1;
    c.def = def;
    c.name = std::string(name, strnlen(name, nameLen));
    if (id == V4L2_CID_EXPOSURE_AUTO && (type == V4L2_CTRL_TYPE_MENU || type == V4L2_CTRL_TYPE_INTEGER_MENU))
      for (int64_t k = min; k <= max && k - min < 64; k++) {
        v4l2_querymenu q{};
        q.id = id;
        q.index = (uint32_t)k;
        if (io(VIDIOC_QUERYMENU, &q) == 0) c.menu.push_back(k);
      }
    ctls_[id] = c;
  };
  // ai: QUERY_EXT_CTRL walks every control; an old driver without it, QUERYCTRL's walk; one that walks none, the ids
  // ai: this backend uses asked one by one
  bool walked = false;
  v4l2_query_ext_ctrl q{};
  q.id = V4L2_CTRL_FLAG_NEXT_CTRL;
  while (io(VIDIOC_QUERY_EXT_CTRL, &q) == 0) {
    walked = true;
    add(q.id, q.type, q.flags, q.minimum, q.maximum, (int64_t)q.step, q.default_value, q.name, sizeof q.name);
    q.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
  }
  if (!walked) {
    v4l2_queryctrl c{};
    c.id = V4L2_CTRL_FLAG_NEXT_CTRL;
    while (io(VIDIOC_QUERYCTRL, &c) == 0) {
      walked = true;
      add(c.id, c.type, c.flags, c.minimum, c.maximum, c.step, c.default_value, (const char*)c.name, sizeof c.name);
      c.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }
  }
  if (!walked) {
    std::vector<uint32_t> ids = {V4L2_CID_EXPOSURE_AUTO, V4L2_CID_EXPOSURE_AUTO_PRIORITY, V4L2_CID_FOCUS_AUTO};
    for (const CtlOf& m : kCtl) ids.push_back(m.cid);
    for (uint32_t id : ids) {
      v4l2_queryctrl c{};
      c.id = id;
      if (io(VIDIOC_QUERYCTRL, &c) == 0)
        add(c.id, c.type, c.flags, c.minimum, c.maximum, c.step, c.default_value, (const char*)c.name, sizeof c.name);
    }
  }
}

const Ctl* V4l2Camera::find(uint32_t id) const {
  if (!id) return nullptr;
  auto it = ctls_.find(id);
  return it == ctls_.end() ? nullptr : &it->second;
}

bool V4l2Camera::get(const Ctl& c, int64_t& v) {
  v4l2_ext_control e{};
  e.id = c.id;
  v4l2_ext_controls es{};
  es.which = V4L2_CTRL_WHICH_CUR_VAL;
  es.count = 1;
  es.controls = &e;
  if (io(VIDIOC_G_EXT_CTRLS, &es) == 0) {
    v = c.type == V4L2_CTRL_TYPE_INTEGER64 ? e.value64 : e.value;
    return true;
  }
  if (c.type == V4L2_CTRL_TYPE_INTEGER64) return false;
  v4l2_control k{};
  k.id = c.id;
  if (io(VIDIOC_G_CTRL, &k) < 0) return false;
  v = k.value;
  return true;
}

// ai: The writes as one S_EXT_CTRLS (an automatic's manual side first, then its value: uvcvideo takes a slave's value
// ai: only where its master is manual, the same call's master counting), else S_CTRL one by one in that order
bool V4l2Camera::put(const std::vector<std::pair<const Ctl*, int64_t>>& w, std::string& why) {
  if (w.empty()) return true;
  std::vector<v4l2_ext_control> e(w.size());
  for (size_t i = 0; i < w.size(); i++) {
    e[i].id = w[i].first->id;
    if (w[i].first->type == V4L2_CTRL_TYPE_INTEGER64) e[i].value64 = w[i].second;
    else e[i].value = (int32_t)w[i].second;
  }
  v4l2_ext_controls es{};
  es.which = V4L2_CTRL_WHICH_CUR_VAL;
  es.count = (uint32_t)e.size();
  es.controls = e.data();
  if (io(VIDIOC_S_EXT_CTRLS, &es) == 0) return true;
  const int extErr = errno;
  for (auto& x : w) {
    if (x.first->type == V4L2_CTRL_TYPE_INTEGER64) {
      why = x.first->name + ": " + err(extErr);
      return false;
    }
    v4l2_control k{};
    k.id = x.first->id;
    k.value = (int32_t)x.second;
    if (io(VIDIOC_S_CTRL, &k) < 0) {
      why = x.first->name + ": " + err(errno);
      return false;
    }
  }
  return true;
}

void V4l2Camera::remember(const Ctl& c) {
  if (remembered(c.id)) return;
  int64_t v;
  if (get(c, v)) changed_.emplace_back(c.id, v);
}

bool V4l2Camera::remembered(uint32_t id) const {
  for (auto& c : changed_)
    if (c.first == id) return true;
  return false;
}

// ai: The value that makes a control's partner automatic: exposure's aperture priority where the menu offers it (the
// ai: exposure time automatic, the iris left alone), else its full automatic; focus's on
bool V4l2Camera::autoOn(const Ctl& a, int64_t& on) const {
  if (a.id == V4L2_CID_FOCUS_AUTO) on = 1;
  else if (a.id != V4L2_CID_EXPOSURE_AUTO) return false;
  else if (offers(a, V4L2_EXPOSURE_APERTURE_PRIORITY)) on = V4L2_EXPOSURE_APERTURE_PRIORITY;
  else if (offers(a, V4L2_EXPOSURE_AUTO)) on = V4L2_EXPOSURE_AUTO;
  else return false;
  return true;
}

int64_t V4l2Camera::snap(const Ctl& c, double v) {
  if (!std::isfinite(v)) return c.def;
  double x = std::min(std::max(v, (double)c.min), (double)c.max);
  if (c.step > 1) x = c.min + std::round((x - c.min) / c.step) * c.step;
  int64_t k = std::llround(x);
  if (k > c.max) k -= c.step;
  return std::max(k, c.min);
}

std::vector<Control> V4l2Camera::controls() {
  std::lock_guard<std::mutex> l(ctlMu_);
  loadCtls();
  std::vector<Control> out;
  for (const CtlOf& m : kCtl) {
    const Ctl* c = find(m.cid);
    if (!c || (c->flags & V4L2_CTRL_FLAG_READ_ONLY)) continue;
    Control k;
    k.id = m.id;
    k.name = c->name;
    k.unit = m.unit;
    k.min = c->min * m.scale;
    k.max = c->max * m.scale;
    k.step = c->step * m.scale;
    k.def = c->def * m.scale;
    int64_t v;
    k.value = get(*c, v) ? v * m.scale : k.def;
    if (const Ctl* a = find(m.autoId); a && !(a->flags & V4L2_CTRL_FLAG_READ_ONLY)) {
      int64_t on, now;
      k.canAuto = autoOn(*a, on);
      k.isAuto = k.canAuto && get(*a, now) && isAuto(*a, now);
    }
    out.push_back(k);
  }
  return out;
}

bool V4l2Camera::set(const std::string& id, double value, bool automatic, std::string& why) {
  std::lock_guard<std::mutex> l(ctlMu_);
  loadCtls();
  const CtlOf* m = nullptr;
  for (const CtlOf& k : kCtl)
    if (id == k.id) m = &k;
  const Ctl* c = m ? find(m->cid) : nullptr;
  if (!c || (c->flags & V4L2_CTRL_FLAG_READ_ONLY)) {
    why = "this camera has no " + id + " to set";
    return false;
  }
  const Ctl* a = m->autoId ? find(m->autoId) : nullptr;
  if (a && (a->flags & V4L2_CTRL_FLAG_READ_ONLY)) a = nullptr;
  std::vector<std::pair<const Ctl*, int64_t>> w;
  if (automatic) {
    int64_t on;
    if (!a || !autoOn(*a, on)) {
      why = "this camera has no automatic " + id;
      return false;
    }
    remember(*a);
    w.emplace_back(a, on);
  } else {
    if (a) {
      if (a->id == V4L2_CID_EXPOSURE_AUTO && !offers(*a, V4L2_EXPOSURE_MANUAL)) {
        why = "this camera's exposure has no manual setting";
        return false;
      }
      remember(*a);
      w.emplace_back(a, manualOf(a->id));
    }
    remember(*c);
    w.emplace_back(c, snap(*c, value / m->scale));
  }
  return put(w, why);
}

// ai: The camera rules (camera.cpp's header) as V4L2 has them (2026-10-25): the camera's own sharpening and backlight
// ai: compensation at their minimum, and exposure_auto_priority 0 (a UVC camera otherwise stretches its frames past
// ai: the interval to expose longer as the light drops); each only where the camera lists it and is not there already
void V4l2Camera::rules() {
  std::lock_guard<std::mutex> l(ctlMu_);
  loadCtls();
  struct Rule { uint32_t id; bool atMin; };
  const Rule rs[] = {{V4L2_CID_SHARPNESS, true}, {V4L2_CID_BACKLIGHT_COMPENSATION, true},
                     {V4L2_CID_EXPOSURE_AUTO_PRIORITY, false}};
  for (const Rule& r : rs) {
    const Ctl* c = find(r.id);
    if (!c || (c->flags & V4L2_CTRL_FLAG_READ_ONLY)) continue;
    const int64_t want = r.atMin ? c->min : snap(*c, 0);
    int64_t now;
    if (get(*c, now) && now == want) continue;
    remember(*c);
    std::string why;
    put({{c, want}}, why);
  }
}

// ai: Every control remembered back to its value before, newest change first: the absolute values (each under a
// ai: manual master where its master is automatic now, since uvcvideo refuses it otherwise; a master this camera did
// ai: not change goes back as it was), then the masters themselves. Best effort: a gone device refuses them all.
void V4l2Camera::restore() {
  std::lock_guard<std::mutex> l(ctlMu_);
  std::string why;
  for (auto it = changed_.rbegin(); it != changed_.rend(); ++it) {
    if (isMaster(it->first)) continue;
    const Ctl* c = find(it->first);
    if (!c) continue;
    const Ctl* m = find(masterOf(it->first));
    int64_t mv = 0;
    if (m && get(*m, mv) && mv != manualOf(m->id) &&
        (m->id != V4L2_CID_EXPOSURE_AUTO || offers(*m, V4L2_EXPOSURE_MANUAL))) {
      put({{m, manualOf(m->id)}, {c, it->second}}, why);
      if (!remembered(m->id)) put({{m, mv}}, why);
    } else {
      put({{c, it->second}}, why);
    }
  }
  for (auto it = changed_.rbegin(); it != changed_.rend(); ++it)
    if (isMaster(it->first))
      if (const Ctl* c = find(it->first)) put({{c, it->second}}, why);
  changed_.clear();
}

bool V4l2Camera::start(const Mode& mode, Sink sink, Ended ended, std::string& why) {
  std::lock_guard<std::mutex> l(runMu_);
  if (running_.load()) {
    why = "the camera is already streaming";
    return false;
  }
  // ai: a stream that ended by itself: its thread joined and its buffers let go (started again from that stream's own
  // ai: ended, the thread is let go instead: it returns once ended does)
  teardown();
  if (th_.joinable()) th_.detach();
  const uint32_t fourcc = (uint32_t)mode.key;
  const PixOf* p = pixOf(fourcc);
  if (!p || !mode.width || !mode.height) {
    why = "not a mode of this camera: " + mode.format + " " + sizeText(mode.width, mode.height);
    return false;
  }
  std::unique_ptr<Jpeg> jpeg;
  if (p->pix == Pix::MJPEG) {
    std::string jw;
    jpeg = Jpeg::load(jw);
    if (!jpeg) {
      why = "MJPEG frames need TurboJPEG: " + jw;
      return false;
    }
  }
  const std::string asked = fourccText(fourcc) + " " + sizeText(mode.width, mode.height);
  v4l2_format f{};
  f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  f.fmt.pix.width = mode.width;
  f.fmt.pix.height = mode.height;
  f.fmt.pix.pixelformat = fourcc;
  f.fmt.pix.field = V4L2_FIELD_NONE;
  if (io(VIDIOC_S_FMT, &f) < 0) {
    const int e = errno;
    why = e == EBUSY    ? "the camera is in use by another program"
          : e == ENODEV ? "the camera was unplugged"
                        : "the camera refused " + asked + ": " + err(e);
    return false;
  }
  if (f.fmt.pix.pixelformat != fourcc || f.fmt.pix.width != mode.width || f.fmt.pix.height != mode.height) {
    why = "the camera gave " + fourccText(f.fmt.pix.pixelformat) + " " + sizeText(f.fmt.pix.width, f.fmt.pix.height) +
          " for " + asked;
    return false;
  }
  uint32_t stride = f.fmt.pix.bytesperline;
  if (!stride && p->bpp) stride = mode.width * p->bpp;
  // ai: the frame interval, the driver's answer kept (it takes the nearest it has)
  double fps = 0;
  if (mode.fps > 0) {
    v4l2_streamparm sp{};
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    sp.parm.capture.timeperframe = fractOf(mode.fps);
    if (io(VIDIOC_S_PARM, &sp) == 0) fps = fpsOf(sp.parm.capture.timeperframe);
  }
  if (fps <= 0) {
    v4l2_streamparm sp{};
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (io(VIDIOC_G_PARM, &sp) == 0 && (sp.parm.capture.capability & V4L2_CAP_TIMEPERFRAME))
      fps = fpsOf(sp.parm.capture.timeperframe);
  }
  rules();
  auto fail = [&](const std::string& w) {
    why = w;
    unmap();
    restore();
    return false;
  };
  v4l2_requestbuffers rb{};
  rb.count = kBuffers;
  rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  rb.memory = V4L2_MEMORY_MMAP;
  if (io(VIDIOC_REQBUFS, &rb) < 0) return fail("the camera gave no buffers: " + err(errno));
  if (rb.count < 2) return fail("the camera gave " + std::to_string(rb.count) + " buffer, 2 needed");
  bufs_.assign(rb.count, Buf{});
  for (uint32_t i = 0; i < rb.count; i++) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    if (io(VIDIOC_QUERYBUF, &b) < 0) return fail("the camera's buffer " + std::to_string(i) + ": " + err(errno));
    void* m = sys_.mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
    if (m == MAP_FAILED) return fail("the camera's buffer " + std::to_string(i) + " not mapped: " + err(errno));
    bufs_[i] = Buf{m, b.length};
  }
  for (uint32_t i = 0; i < rb.count; i++) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    if (io(VIDIOC_QBUF, &b) < 0) return fail("the camera's buffer " + std::to_string(i) + " not queued: " + err(errno));
  }
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (io(VIDIOC_STREAMON, &type) < 0) {
    const int e = errno;
    return fail(e == ENOSPC ? "the camera's stream does not fit the USB bus (another camera on it?)"
                            : "the camera did not stream: " + err(e));
  }
  pix_ = p->pix;
  bits_ = p->bits;
  w_ = mode.width;
  h_ = mode.height;
  stride_ = stride;
  {
    std::lock_guard<std::mutex> s(statMu_);
    st_ = Stats{};
    st_.fps = fps;
  }
  stopping_ = false;
  streaming_ = true;
  running_ = true;
  th_ = std::thread([this, sink = std::move(sink), ended = std::move(ended), j = std::move(jpeg)]() mutable {
    run(std::move(sink), std::move(ended), std::move(j));
  });
  capId_ = th_.get_id();
  return true;
}

void V4l2Camera::run(Sink sink, Ended ended, std::unique_ptr<Jpeg> jpeg) {
  std::vector<uint8_t> conv;
  std::string why;
  bool haveSeq = false;
  uint32_t lastSeq = 0;
  while (!stopping_.load()) {
    pollfd pf{};
    pf.fd = fd_;
    pf.events = POLLIN;
    const int r = sys_.poll(&pf, 1, 200);
    if (stopping_.load()) break;
    if (r < 0) {
      if (errno == EINTR) continue;
      why = errno == ENODEV ? "the camera was unplugged" : err(errno);
      break;
    }
    if (r == 0) continue;
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    if (io(VIDIOC_DQBUF, &b) < 0) {
      const int e = errno;
      if (e == EAGAIN) {
        // ai: an error state poll reports with nothing to dequeue: no busy loop while it lasts
        if (pf.revents & (POLLERR | POLLHUP | POLLNVAL)) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      if (stopping_.load()) break;
      why = e == ENODEV ? "the camera was unplugged" : err(e);
      break;
    }
    const int64_t at = steadyNs();
    if (b.index >= bufs_.size()) {
      why = "the camera returned buffer " + std::to_string(b.index) + " of " + std::to_string(bufs_.size());
      break;
    }
    // ai: the sequence counts every frame the device made, a buffer to fill or not: a gap is frames lost
    if (haveSeq) {
      const uint32_t gap = b.sequence - lastSeq - 1;
      if (gap && gap < 0x80000000u) {
        std::lock_guard<std::mutex> s(statMu_);
        st_.dropped += gap;
      }
    }
    lastSeq = b.sequence;
    haveSeq = true;
    const uint8_t* data = (const uint8_t*)bufs_[b.index].p;
    const size_t n = std::min<size_t>(b.bytesused, bufs_[b.index].len);
    Frame f;
    bool good = !(b.flags & V4L2_BUF_FLAG_ERROR);
    if (good) {
      if (pix_ == Pix::MJPEG) {
        uint32_t jw = 0, jh = 0;
        good = n >= kJpegMin && jpeg->grey(data, n, conv, jw, jh, nullptr) && jw == w_ && jh == h_;
        f.luma = conv.data();
        f.stride = w_;
      } else if (n < frameBytes(pix_, w_, h_, stride_)) {
        good = false;
      } else if (lumaInPlace(pix_)) {
        f.luma = data;
        f.stride = stride_;
      } else {
        good = toLuma(pix_, bits_, data, n, w_, h_, stride_, conv, nullptr);
        f.luma = conv.data();
        f.stride = w_;
      }
    }
    if (good) {
      f.width = w_;
      f.height = h_;
      const int64_t t = timevalNs(b.timestamp);
      f.ns = (b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC && t > 0 ? t : at;
      f.seq = b.sequence;
      const double ms = (steadyNs() - at) / 1e6;
      {
        std::lock_guard<std::mutex> s(statMu_);
        st_.frames++;
        st_.convertMs += (ms - st_.convertMs) / (double)st_.frames;
      }
      if (sink) sink(f);
    } else {
      std::lock_guard<std::mutex> s(statMu_);
      st_.corrupt++;
    }
    v4l2_buffer q{};
    q.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    q.memory = V4L2_MEMORY_MMAP;
    q.index = b.index;
    if (io(VIDIOC_QBUF, &q) < 0) {
      const int e = errno;
      if (stopping_.load()) break;
      why = e == ENODEV ? "the camera was unplugged" : err(e);
      break;
    }
  }
  // ai: the stream over before ended is told, so a start from ended itself is taken
  running_ = false;
  if (!why.empty() && !stopping_.load() && ended) ended(why);
}

void V4l2Camera::unmap() {
  for (Buf& b : bufs_)
    if (b.p) sys_.munmap(b.p, b.len);
  bufs_.clear();
  v4l2_requestbuffers rb{};
  rb.count = 0;
  rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  rb.memory = V4L2_MEMORY_MMAP;
  io(VIDIOC_REQBUFS, &rb);
}

// ai: runMu_ held. The stream off (which wakes the thread's poll), the thread joined (not from itself: a sink or
// ai: ended that stops the camera has it joined by the next stop, start or the destructor), the buffers let go.
void V4l2Camera::teardown() {
  const bool self = onCapture();
  if (streaming_) {
    stopping_ = true;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    io(VIDIOC_STREAMOFF, &type);
  }
  if (th_.joinable() && !self) {
    th_.join();
    capId_ = std::thread::id();
  }
  if (streaming_) {
    unmap();
    streaming_ = false;
  }
}

void V4l2Camera::stop() {
  std::unique_lock<std::mutex> l(runMu_, std::defer_lock);
  if (onCapture()) {
    // ai: from the capture thread (a sink or ended stopping the camera): another thread already stopping it holds the
    // ai: lock and joins this thread, so it is not waited for
    if (!l.try_lock()) {
      stopping_ = true;
      return;
    }
  } else {
    l.lock();
  }
  teardown();
  restore();
}

}  // namespace

std::vector<std::string> probedNodes(const char* env, const std::function<std::vector<std::string>()>& byDefault) {
  if (!env) return byDefault ? byDefault() : std::vector<std::string>{};
  std::vector<std::string> out;
  const std::string s(env);
  size_t at = 0;
  while (at <= s.size()) {
    const size_t end = std::min(s.find(':', at), s.size());
    if (end > at) out.push_back(s.substr(at, end - at));
    at = end + 1;
  }
  return out;
}

const V4l2Sys& v4l2Sys() {
  const V4l2Sys* s = gSys.load();
  return s ? *s : kernelSys();
}

const V4l2Sys* v4l2UseSys(const V4l2Sys* table) {
  const V4l2Sys* before = gSys.exchange(table);
  return before ? before : &kernelSys();
}

std::vector<Info> v4l2List(std::vector<std::string>* notes) {
  const V4l2Sys& s = v4l2Sys();
  std::vector<Info> out;
  for (const std::string& node : s.nodes()) {
    const int fd = s.open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
      if (notes) notes->push_back("v4l2: " + node + ": " + err(errno));
      continue;
    }
    v4l2_capability c;
    if (queryCap(s, fd, c) && captures(c)) out.push_back(infoOf(s, node, c));
    s.close(fd);
  }
  return out;
}

std::unique_ptr<Camera> v4l2Open(const std::string& id, std::string& why) {
  const std::string pre = "v4l2:";
  if (id.compare(0, pre.size(), pre)) {
    why = "not a V4L2 camera: " + id;
    return nullptr;
  }
  const std::string path = id.substr(pre.size());
  const V4l2Sys& s = v4l2Sys();
  // ai: the id among the nodes probed (by its node or its by-id link), so nothing unlisted is opened
  std::string node;
  for (const std::string& n : s.nodes())
    if (n == path || s.byId(n) == path) {
      node = n;
      break;
    }
  if (node.empty()) {
    why = "no such camera (" + path + ")";
    return nullptr;
  }
  const int fd = s.open(node.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    why = node + ": " + err(errno);
    return nullptr;
  }
  v4l2_capability c;
  if (!queryCap(s, fd, c)) {
    why = node + ": " + err(errno);
    s.close(fd);
    return nullptr;
  }
  if (!captures(c)) {
    why = node + " does not capture video";
    s.close(fd);
    return nullptr;
  }
  return std::make_unique<V4l2Camera>(s, fd, infoOf(s, node, c));
}

}  // namespace lizard::cam
