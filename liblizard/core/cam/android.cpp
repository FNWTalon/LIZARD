// ai: Android's cameras through Camera2's NDK (2026-10-31; the app's Engine.kt in Kotlin before, its request key for key):
// ai: the phone's back cameras, android:<the camera's id>, the one that focuses closest listed first (a receiver is
// ai: held near a screen: the S26's 2.2 mm camera at 5 cm, where its first listed, the main lens, read a tenth of it).
// ai: Modes: each of the app's resolutions (1280 x 720, 1920 x 1080, 2560 x 1440, 3840 x 2160) the lens offers as
// ai: PRIVATE or YUV_420_888, in each format the app reads (Mode::format): "auto" (PRIVATE for a receiver that samples
// ai: the camera's buffer, the GPU decoder's zero copy, else YUV_420_888: receiverWants), "private", "yuv", "y8" at the
// ai: sizes that format lists, and "raw16", "raw10", "raw12" at each resolution (the lens's whole array, cropped to the
// ai: zoom's centre at the resolution's shape, a 2 x 2 bin where half of it still holds the resolution, developed to
// ai: luma by the receiver: ingest/develop.h). P010 is not offered here: the NDK's image reader takes no YCBCR_P010 and
// ai: its outputs no dynamic range profile. The rates (since 2026-11-01, a mode each): every fixed range the lens
// ai: lists for a regular capture session (CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES; the constrained high-speed
// ai: session's own list is never read) up to 60, the fastest first, so the session's pick at 60 is the default; with
// ai: none fixed, the highest top. Each mode carries its preview size: the largest PRIVATE size of the capture's shape
// ai: up to 1280 wide (it is only for aiming), else the capture's.
// ai: The request (TEMPLATE_RECORD), each enhancement off where the lens lists its off mode (the camera is a machine
// ai: vision camera: STATUS "The ISP's enhancements off"): noise reduction, edge enhancement, hot pixel correction,
// ai: chromatic aberration correction, face detection, scene modes (CONTROL_MODE AUTO with SCENE_MODE DISABLED), the
// ai: statistics maps (lens shading, hot pixel, OIS data); video stabilisation and OIS off; the fixed rate's range;
// ai: autofocus CONTINUOUS_VIDEO where listed, its region the code's box once the receiver finds it (focusOn);
// ai: auto-exposure always (the 2026-10-04 rule: no exposure or gain control is listed); the zoom as CONTROL_ZOOM_RATIO
// ai: from API 30 (the lens's whole range), else SCALER_CROP_REGION's centred crop of the active array (1 to the
// ai: lens's largest digital zoom). The tone curve, lens shading and distortion correction stay the camera's.
// ai: test: `adb shell setprop debug.lizard.camx <name,...>` (read at each start) leaves that enhancement as the
// ai: template has it (nr, edge, hot, ca, face, scene, stats; all); `debug.lizard.camdump 1` logs every key of the
// ai: request as built (`request: <tag> = <values>`); `debug.lizard.nudge <us>[.<anything>]` (read every 30 captures;
// ai: a change asks again) makes one frame that much longer and logs the 30 capture intervals after it (`delay:`).
// ai: Frames: an image reader of 32 images (6 for a format read on the CPU: Y8, RAW), its listener on the reader's
// ai: own thread at the display's priority; PRIVATE and YUV for the GPU's zero copy go as the image's AHardwareBuffer,
// ai: held until the receiver lets it go (release), the rest as plane 0 (Y, Y8) or a RawFrame inside the sink's call.
// ai: The phase lock's delay (delayOnce): one capture with auto-exposure off at the newest undelayed result's exposure
// ai: and sensitivity and SENSOR_FRAME_DURATION that result's frame plus the delay; the repeating request goes on after
// ai: it; its fate (captured or not) reported back. Barred (delayBar) on a lens without manual sensor control or whose
// ai: controls are not applied per frame (SYNC_MAX_LATENCY).
// ai: Threads: the session's thread opens, starts, stops, sets and hands the preview; Camera2's callback thread reports
// ai: each result (captures, the lock's delay asked from there); the reader's thread hands the frames over.
// ai: The log (tools/phone read these lines): `open camera`, every 300 captures `camera:` (captures and images a
// ai: second, images held) and `capture:` (what the camera did), `focus:`, `delay:`, `camx:`.
#include "backends.h"
#include "camera.h"
#include "clock.h"
#include "develop.h"

#include <android/api-level.h>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <android/native_window.h>
#include <camera/NdkCameraCaptureSession.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCaptureRequest.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <sys/resource.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lizard::cam {

namespace {

void say(const char* fmt, ...) {
  va_list a;
  va_start(a, fmt);
  __android_log_vprint(ANDROID_LOG_INFO, "lizard", fmt, a);
  va_end(a);
}

std::string prop(const char* name) {
  char v[PROP_VALUE_MAX] = "";
  __system_property_get(name, v);
  return v;
}

int api() {
  static const int l = android_get_device_api_level();
  return l;
}

// ai: one manager a process (the camera service's connection)
ACameraManager* manager() {
  static ACameraManager* m = ACameraManager_create();
  return m;
}

const uint32_t RESOLUTIONS[][2] = {{1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
// ai: the formats the app reads, the Format menu's keys (auto comes first, at each resolution)
struct Fmt {
  const char* key;
  int32_t img;
};
const Fmt FORMATS[] = {{"private", AIMAGE_FORMAT_PRIVATE}, {"yuv", AIMAGE_FORMAT_YUV_420_888}, {"y8", AIMAGE_FORMAT_Y8},
                       {"raw16", AIMAGE_FORMAT_RAW16},     {"raw10", AIMAGE_FORMAT_RAW10},      {"raw12", AIMAGE_FORMAT_RAW12}};
bool isRaw(int32_t f) { return f == AIMAGE_FORMAT_RAW16 || f == AIMAGE_FORMAT_RAW10 || f == AIMAGE_FORMAT_RAW12; }
// ai: read on the CPU and closed inside the sink's call: a few images do (RAW16 at the S26's 4080 x 3060 is 25 MB each)
bool onCpu(int32_t f) { return isRaw(f) || f == AIMAGE_FORMAT_Y8; }
const char* nameOf(int32_t f) {
  switch (f) {
    case AIMAGE_FORMAT_PRIVATE: return "PRIVATE";
    case AIMAGE_FORMAT_YUV_420_888: return "YUV_420_888";
    case AIMAGE_FORMAT_Y8: return "Y8";
    case AIMAGE_FORMAT_RAW16: return "RAW16";
    case AIMAGE_FORMAT_RAW10: return "RAW10";
    case AIMAGE_FORMAT_RAW12: return "RAW12";
  }
  return "format";
}
// ai: the reader's pool: a camera buffer is held until the GPU has read it, which is after the batch ahead of it; with
// ai: 16 a throttled GPU left the camera no buffer (S26, 2026-09-30)
constexpr int MAX_IMAGES = 32;
constexpr int PLANE_IMAGES = 6;

// ai: A camera's characteristics, read with plain lookups (an entry missing reads as none)
struct Chars {
  ACameraMetadata* m = nullptr;
  ~Chars() {
    if (m) ACameraMetadata_free(m);
  }
  bool get(uint32_t tag, ACameraMetadata_const_entry& e) const {
    return m && ACameraMetadata_getConstEntry(m, tag, &e) == ACAMERA_OK && e.count > 0;
  }
  std::vector<uint8_t> u8s(uint32_t tag) const {
    ACameraMetadata_const_entry e{};
    return get(tag, e) ? std::vector<uint8_t>(e.data.u8, e.data.u8 + e.count) : std::vector<uint8_t>{};
  }
  std::vector<int32_t> i32s(uint32_t tag) const {
    ACameraMetadata_const_entry e{};
    return get(tag, e) ? std::vector<int32_t>(e.data.i32, e.data.i32 + e.count) : std::vector<int32_t>{};
  }
  std::vector<float> floats(uint32_t tag) const {
    ACameraMetadata_const_entry e{};
    return get(tag, e) ? std::vector<float>(e.data.f, e.data.f + e.count) : std::vector<float>{};
  }
  bool lists(uint32_t tag, int v) const {
    for (uint8_t x : u8s(tag))
      if (x == v) return true;
    return false;
  }
};

std::unique_ptr<Chars> charsOf(const std::string& id) {
  auto c = std::make_unique<Chars>();
  if (ACameraManager_getCameraCharacteristics(manager(), id.c_str(), &c->m) != ACAMERA_OK) return nullptr;
  return c;
}

// ai: a result's lookups, as Chars's
template <class T>
bool one(const ACameraMetadata* r, uint32_t tag, T& out) {
  ACameraMetadata_const_entry e{};
  if (!r || ACameraMetadata_getConstEntry(r, tag, &e) != ACAMERA_OK || e.count == 0) return false;
  if constexpr (std::is_same_v<T, int64_t>) out = e.data.i64[0];
  else if constexpr (std::is_same_v<T, int32_t>) out = e.data.i32[0];
  else if constexpr (std::is_same_v<T, float>) out = e.data.f[0];
  else if constexpr (std::is_same_v<T, uint8_t>) out = e.data.u8[0];
  return true;
}
int byteOr(const ACameraMetadata* r, uint32_t tag, int def = -1) {
  uint8_t v = 0;
  return one(r, tag, v) ? int(v) : def;
}
std::string textOf(const ACameraMetadata* r, uint32_t tag) {
  ACameraMetadata_const_entry e{};
  if (!r || ACameraMetadata_getConstEntry(r, tag, &e) != ACAMERA_OK || e.count == 0) return "";
  std::string s(reinterpret_cast<const char*>(e.data.u8), e.count);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

struct Size {
  int32_t w = 0, h = 0;
  bool operator==(const Size& o) const { return w == o.w && h == o.h; }
};

// ai: the sizes a format lists as an output
std::vector<Size> sizesOf(const Chars& c, int32_t fmt) {
  std::vector<Size> out;
  const std::vector<int32_t> v = c.i32s(ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS);
  for (size_t i = 0; i + 3 < v.size(); i += 4)
    if (v[i] == fmt && v[i + 3] == ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS_OUTPUT) out.push_back({v[i + 1], v[i + 2]});
  return out;
}
bool has(const std::vector<Size>& v, Size s) { return std::find(v.begin(), v.end(), s) != v.end(); }
Size largest(const std::vector<Size>& v) {
  Size b;
  for (const Size& s : v)
    if (int64_t(s.w) * s.h > int64_t(b.w) * b.h) b = s;
  return b;
}

// ai: a phone's camera runs at 60 a second at most (the root's Settled: nothing past 60 captured but a machine vision
// ai: camera's)
constexpr int32_t MAX_FPS = 60;
// ai: the rates a mode can take: each fixed range up to MAX_FPS (a fixed range holds the exposure time to the frame),
// ai: the fastest first; with none, the highest top up to MAX_FPS; {{0, 0}} where none is listed
std::vector<std::pair<int32_t, int32_t>> ratesOf(const Chars& c) {
  const std::vector<int32_t> v = c.i32s(ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
  std::vector<std::pair<int32_t, int32_t>> fixed;
  std::pair<int32_t, int32_t> top{0, 0};
  for (size_t i = 0; i + 1 < v.size(); i += 2) {
    if (v[i + 1] > MAX_FPS) continue;
    if (v[i] == v[i + 1] && std::find(fixed.begin(), fixed.end(), std::make_pair(v[i], v[i + 1])) == fixed.end())
      fixed.push_back({v[i], v[i + 1]});
    if (v[i + 1] > top.second) top = {v[i], v[i + 1]};
  }
  std::sort(fixed.begin(), fixed.end(), [](auto& a, auto& b) { return a.second > b.second; });
  return fixed.empty() ? std::vector<std::pair<int32_t, int32_t>>{top} : fixed;
}

// ai: the preview: the largest PRIVATE size of the capture's shape up to 1280 wide, else the capture's own
Size previewFor(const Chars& c, Size cap) {
  Size b;
  for (const Size& s : sizesOf(c, AIMAGE_FORMAT_PRIVATE))
    if (int64_t(s.w) * cap.h == int64_t(s.h) * cap.w && s.w <= 1280 && s.w > b.w) b = s;
  return b.w ? b : cap;
}

// ai: Engine.lockBarOf: "" where the lens takes a delay frame (manual sensor control, its controls applied per frame)
std::string lockBarOf(const Chars& c) {
  if (!c.lists(ACAMERA_REQUEST_AVAILABLE_CAPABILITIES, ACAMERA_REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR))
    return "the lens has no manual sensor control";
  const std::vector<int32_t> l = c.i32s(ACAMERA_SYNC_MAX_LATENCY);
  const int32_t lag = l.empty() ? ACAMERA_SYNC_MAX_LATENCY_UNKNOWN : l[0];
  if (lag == ACAMERA_SYNC_MAX_LATENCY_PER_FRAME_CONTROL) return "";
  if (lag == ACAMERA_SYNC_MAX_LATENCY_UNKNOWN) return "the lens does not say when its controls apply";
  return "the lens applies its controls up to " + std::to_string(lag) + " frames late";
}

bool facingBack(const Chars& c) {
  const std::vector<uint8_t> f = c.u8s(ACAMERA_LENS_FACING);
  return !f.empty() && f[0] == ACAMERA_LENS_FACING_BACK;
}

// ai: the menu's label (Engine.lenses): the id, the focal length, the horizontal field of view, "multi" for a logical
// ai: camera that switches between lenses
std::string labelOf(const std::string& id, const Chars& c) {
  const std::vector<float> f = c.floats(ACAMERA_LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
  const std::vector<float> ps = c.floats(ACAMERA_SENSOR_INFO_PHYSICAL_SIZE);
  std::string l = id;
  char b[64];
  if (!f.empty()) {
    std::snprintf(b, sizeof b, " %.1f mm", f[0]);
    l += b;
    if (!ps.empty() && f[0] > 0) {
      std::snprintf(b, sizeof b, " %d\xc2\xb0", int(std::lround(2 * std::atan(ps[0] / (2.0 * f[0])) * 180 / M_PI)));
      l += b;
    }
  }
  if (c.lists(ACAMERA_REQUEST_AVAILABLE_CAPABILITIES, ACAMERA_REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)) l += " multi";
  return l;
}

// ai: the back cameras, the closest-focusing first (the largest LENS_INFO_MINIMUM_FOCUS_DISTANCE, in dioptres; the
// ai: first listed among equals), then the rest as the system lists them
std::vector<std::pair<std::string, std::unique_ptr<Chars>>> backCameras(std::string* why) {
  std::vector<std::pair<std::string, std::unique_ptr<Chars>>> out;
  ACameraIdList* ids = nullptr;
  if (ACameraManager_getCameraIdList(manager(), &ids) != ACAMERA_OK || !ids) {
    if (why) *why = "the camera service lists no cameras";
    return out;
  }
  for (int i = 0; i < ids->numCameras; i++) {
    const std::string id = ids->cameraIds[i];
    auto c = charsOf(id);
    if (c && facingBack(*c)) out.emplace_back(id, std::move(c));
  }
  ACameraManager_deleteCameraIdList(ids);
  auto focus = [](const Chars& c) {
    const std::vector<float> f = c.floats(ACAMERA_LENS_INFO_MINIMUM_FOCUS_DISTANCE);
    return f.empty() ? 0.f : f[0];
  };
  size_t best = 0;
  for (size_t i = 1; i < out.size(); i++)
    if (focus(*out[i].second) > focus(*out[best].second)) best = i;
  if (best) std::rotate(out.begin(), out.begin() + long(best), out.begin() + long(best) + 1);
  return out;
}

// ai: the modes (the file's header): auto at each resolution offered, then each format the app reads
std::vector<Mode> modesOf(const Chars& c) {
  const std::vector<Size> priv = sizesOf(c, AIMAGE_FORMAT_PRIVATE), yuv = sizesOf(c, AIMAGE_FORMAT_YUV_420_888);
  const auto rates = ratesOf(c);
  std::vector<Size> res;
  for (const auto& r : RESOLUTIONS)
    if (has(priv, {int32_t(r[0]), int32_t(r[1])}) || has(yuv, {int32_t(r[0]), int32_t(r[1])})) res.push_back({int32_t(r[0]), int32_t(r[1])});
  // ai: none of the app's resolutions: the offered size nearest 1920 x 1080
  if (res.empty()) {
    std::vector<Size> all = priv;
    all.insert(all.end(), yuv.begin(), yuv.end());
    Size near;
    for (const Size& s : all)
      if (!near.w || std::llabs(int64_t(s.w) * s.h - 1920LL * 1080) < std::llabs(int64_t(near.w) * near.h - 1920LL * 1080)) near = s;
    if (near.w) res.push_back(near);
  }
  std::vector<Mode> out;
  // ai: a mode each rate
  auto add = [&](Size s, const char* key, int32_t img, Size cap) {
    const Size p = previewFor(c, cap);
    for (const auto& r : rates) {
      Mode m;
      m.width = uint32_t(s.w);
      m.height = uint32_t(s.h);
      m.fps = r.second;
      m.fpsMin = r.first;
      m.format = key;
      m.pix = img && isRaw(img) ? Pix::Grey16 : Pix::Grey8;
      m.key = uint64_t(uint32_t(img));
      m.previewW = uint32_t(p.w);
      m.previewH = uint32_t(p.h);
      out.push_back(m);
    }
  };
  for (const Size& s : res) add(s, "auto", 0, s);
  for (const Fmt& f : FORMATS) {
    const std::vector<Size> z = sizesOf(c, f.img);
    if (z.empty()) continue;
    // ai: RAW: the lens's whole array (its largest), cropped at each resolution's shape
    if (isRaw(f.img)) {
      for (const Size& s : res) add(s, f.key, f.img, largest(z));
      continue;
    }
    for (const Size& s : res)
      if (has(z, s)) add(s, f.key, f.img, s);
  }
  return out;
}

class AndroidCamera;

// ai: a request's marks (ACaptureRequest_setUserContext): the lock's delay, the nudge test's
char LOCK_MARK, NUDGE_MARK;

class AndroidCamera : public Camera {
 public:
  AndroidCamera(std::string camId, std::unique_ptr<Chars> ch) : id_(std::move(camId)), ch_(std::move(ch)) {
    info_.id = "android:" + id_;
    info_.name = labelOf(id_, *ch_);
    info_.backend = "android";
    const bool logical = ch_->lists(ACAMERA_REQUEST_AVAILABLE_CAPABILITIES, ACAMERA_REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA);
    info_.detail = std::string("back") + (logical ? ", logical" : "");
    modes_ = modesOf(*ch_);
    bar_ = lockBarOf(*ch_);
    fps_ = ratesOf(*ch_).front();
    // ai: the zoom's range: CONTROL_ZOOM_RATIO_RANGE from API 30, else 1 to the largest digital zoom (a crop region)
    const std::vector<float> zr = ch_->floats(ACAMERA_CONTROL_ZOOM_RATIO_RANGE);
    if (api() >= 30 && zr.size() >= 2) {
      zoomLo_ = zr[0];
      zoomHi_ = zr[1];
      ratio_ = true;
    } else {
      const std::vector<float> mz = ch_->floats(ACAMERA_SCALER_AVAILABLE_MAX_DIGITAL_ZOOM);
      zoomLo_ = 1;
      zoomHi_ = mz.empty() ? 1 : std::max(1.f, mz[0]);
    }
    // ai: 1.5 unless set (the app's default since 2026-10-02: the code in the middle of the lens's field)
    zoom_ = std::clamp(1.5f, zoomLo_, zoomHi_);
    const std::vector<int32_t> regions = ch_->i32s(ACAMERA_CONTROL_MAX_REGIONS);
    afMax_ = regions.size() >= 3 ? regions[2] : 0;
    active_ = ch_->i32s(ACAMERA_SENSOR_INFO_ACTIVE_ARRAY_SIZE);
    say("camera: %s, %zu modes, lock %s", info_.name.c_str(), modes_.size(), bar_.empty() ? "possible" : bar_.c_str());
  }
  ~AndroidCamera() override {
    stop();
    std::lock_guard<std::mutex> l(winMu_);
    if (window_) ANativeWindow_release(window_);
  }

  const Info& info() const override { return info_; }
  std::vector<Mode> modes() override { return modes_; }
  std::vector<Control> controls() override {
    Control z;
    z.id = "zoom";
    z.name = "Zoom";
    z.unit = "x";
    z.min = zoomLo_;
    z.max = zoomHi_;
    z.step = 0.1;
    z.def = std::clamp(1.5f, zoomLo_, zoomHi_);
    z.value = zoom_.load();
    return {z};
  }
  bool set(const std::string& id, double value, bool automatic, std::string& why) override {
    if (id != "zoom") {
      why = "this camera has no " + id + " to set";
      return false;
    }
    if (automatic) {
      why = "this camera has no automatic zoom";
      return false;
    }
    zoom_ = std::clamp(float(value), zoomLo_, zoomHi_);
    std::lock_guard<std::mutex> l(reqMu_);
    if (req_ && cs_) {
      applyZoom(req_);
      repeat();
    }
    return true;
  }

  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override;
  void stop() override;
  Stats stats() override {
    Stats s;
    s.frames = frames_.load();
    s.dropped = dropped_.load();
    s.corrupt = corrupt_.load();
    s.fps = fps_.second;
    return s;
  }

  void receiverWants(bool luma) override { luma_ = luma; }
  void release(uint64_t hold) override {
    AImage* img = nullptr;
    {
      std::lock_guard<std::mutex> l(holdMu_);
      const auto it = held_.find(hold);
      if (it == held_.end()) return;
      img = it->second;
      held_.erase(it);
    }
    AImage_delete(img);
    holdCv_.notify_all();
  }
  void previewTo(void* w) override;
  void focusOn(const double box[4]) override;
  std::string report() override;
  std::string delayBar() override { return bar_; }
  void captures(Captures each, Fate fate) override {
    std::lock_guard<std::mutex> l(cbMu_);
    each_ = std::move(each);
    fate_ = std::move(fate);
  }
  bool delayOnce(int64_t us) override { return delay(us, true, true); }

  // ai: Camera2's callbacks (the static trampolines below), on its own thread
  void completed(ACaptureRequest* q, const ACameraMetadata* r);
  void failed(ACaptureRequest* q, ACameraCaptureFailure* f);
  void aborted(int seq);
  void deviceGone(const char* why);
  // ai: the reader's listener, on its own thread
  void images();

 private:
  bool open(std::string& why);
  bool configure(std::string& why);
  void takeReader(AImageReader* r);
  void dropReader(bool holds);
  void teardownSession();
  void applyZoom(ACaptureRequest* q);
  void repeat();
  bool delay(int64_t us, bool lock, bool quiet);
  void levels(const ACameraMetadata* r);

  Info info_;
  const std::string id_;
  std::unique_ptr<Chars> ch_;
  std::vector<Mode> modes_;
  std::string bar_;
  std::pair<int32_t, int32_t> fps_{0, 0};
  float zoomLo_ = 1, zoomHi_ = 1;
  bool ratio_ = false;           // ai: CONTROL_ZOOM_RATIO (API 30), else the crop region
  std::atomic<float> zoom_{1.5f};
  int32_t afMax_ = 0;
  std::vector<int32_t> active_;  // ai: the active array: left, top, width, height
  std::atomic<bool> luma_{false};

  // ai: the stream, the session's thread's (start, stop, previewTo, set's re-issue under reqMu_); fmt_, size_,
  // ai: preview_ and note_ written under repMu_ too, for report()'s readers
  Mode mode_;
  std::mutex repMu_;
  int32_t fmt_ = 0;              // ai: the reader's format
  Size size_, preview_;
  std::string note_;
  ACameraDevice* dev_ = nullptr;
  AImageReader* reader_ = nullptr;
  ANativeWindow* readerWin_ = nullptr;   // ai: the reader's own (AImageReader_getWindow: not acquired)
  ACaptureSessionOutputContainer* outs_ = nullptr;
  ACaptureSessionOutput* outReader_ = nullptr;
  ACaptureSessionOutput* outPreview_ = nullptr;
  ACameraOutputTarget* tgtReader_ = nullptr;
  ACameraOutputTarget* tgtPreview_ = nullptr;
  ANativeWindow* inSession_ = nullptr;   // ai: the preview window the capture session writes (acquired), null none
  std::mutex reqMu_;
  ACameraCaptureSession* cs_ = nullptr;  // ai: reqMu_ for the callbacks' delay
  ACaptureRequest* req_ = nullptr;       // ai: reqMu_: the repeating request as built
  ACameraDevice_StateCallbacks devCb_{};
  ACameraCaptureSession_stateCallbacks csCb_{};
  ACameraCaptureSession_captureCallbacks capCb_{};

  std::mutex winMu_;
  ANativeWindow* window_ = nullptr;      // ai: winMu_: the platform's preview surface (acquired)

  // ai: callbacks: cbMu_ held through each callback's body, so stop's flag under it means none runs after
  std::mutex cbMu_;
  bool live_ = false;                    // ai: cbMu_
  Sink sink_;
  Ended ended_;
  Captures each_;
  Fate fate_;
  // ai: The reader's listener and the reader's end (dropReader): imgMu_ held through the listener's body, and reading_
  // ai: under it says the reader is there to read. An image-available message the reader's looper already holds runs
  // ai: after its listener is cleared (the NDK posts it with the callback in it), and AImageReader_delete stops that
  // ai: looper holding the reader's own lock, which acquireNextImage takes: a listener that met a delete under way
  // ai: would wait on it for ever while the delete waits on the listener. So reading_ is cleared under imgMu_ before any
  // ai: delete (a listener running is waited for), and the listener returns at once on a reader not reading, touching
  // ai: none of it (Engine.closeReader's drained frames handler before 2026-10-31).
  std::mutex imgMu_;
  bool reading_ = false;                 // ai: imgMu_; reader_ is written only while it is false

  std::mutex holdMu_;
  std::condition_variable holdCv_;
  std::unordered_map<uint64_t, AImage*> held_;   // ai: holdMu_
  std::atomic<uint64_t> nextHold_{0};

  // ai: the callbacks' own (Camera2's thread)
  std::mutex lastMu_;
  int64_t lastExp_ = 0, lastDur_ = 0, lastSkew_ = 0;   // ai: lastMu_: the newest undelayed result's
  int32_t lastIso_ = 0;
  float lastFocus_ = 0, lastZoom_ = 1;
  std::string lastLens_;
  bool fresh_ = true;            // ai: the next result is a capture session's first
  int delaySeq_ = -1;            // ai: the lock's delay's sequence, -1 none out
  uint64_t n_ = 0;
  int levelsAt_ = 0;
  std::string nudge_;
  int watch_ = 0;
  int64_t prevTs_ = 0;
  std::string gaps_;
  int64_t lineAt_ = 0;
  uint64_t imagesAt_ = 0;
  std::mutex levelMu_;
  float levels_[9] = {0, 0, 0, 0, 1023, 1, 1, 1, 1};   // ai: levelMu_: a RAW frame's (Engine.levelsOf)
  double focusBox_[4] = {0, 0, 0, 0};
  bool focused_ = false;

  std::atomic<uint64_t> frames_{0}, dropped_{0}, corrupt_{0}, images_{0}, seq_{0};
  bool prioritySet_ = false;     // ai: the reader's thread's
};

void onDeviceDisconnected(void* ctx, ACameraDevice*) { static_cast<AndroidCamera*>(ctx)->deviceGone("The camera was taken by another app."); }
void onDeviceError(void* ctx, ACameraDevice*, int error) {
  static_cast<AndroidCamera*>(ctx)->deviceGone(error == ERROR_CAMERA_IN_USE || error == ERROR_MAX_CAMERAS_IN_USE
                                                   ? "The camera is in use by another app."
                                                   : ("The camera stopped (error " + std::to_string(error) + ").").c_str());
}
void onSessionClosed(void*, ACameraCaptureSession*) {}
void onSessionReady(void*, ACameraCaptureSession*) {}
void onSessionActive(void*, ACameraCaptureSession*) {}
void onCompleted(void* ctx, ACameraCaptureSession*, ACaptureRequest* q, const ACameraMetadata* r) { static_cast<AndroidCamera*>(ctx)->completed(q, r); }
void onFailed(void* ctx, ACameraCaptureSession*, ACaptureRequest* q, ACameraCaptureFailure* f) { static_cast<AndroidCamera*>(ctx)->failed(q, f); }
void onAborted(void* ctx, ACameraCaptureSession*, int seq) { static_cast<AndroidCamera*>(ctx)->aborted(seq); }
void onImages(void* ctx, AImageReader*) { static_cast<AndroidCamera*>(ctx)->images(); }

bool AndroidCamera::start(const Mode& mode, Sink sink, Ended ended, std::string& why) {
  stop();
  std::unique_lock<std::mutex> rl(repMu_);
  mode_ = mode;
  note_.clear();
  // ai: the mode's rate (a fixed range; the session keeps the one chosen)
  if (mode.fps > 0) fps_ = {int32_t(mode.fpsMin), int32_t(mode.fps)};
  // ai: the format: auto PRIVATE for the GPU decoder's zero copy where the size lists it, else YUV_420_888; Private
  // ai: only for a receiver that takes the camera's buffer; one the lens does not offer at the size falls to auto
  const std::vector<Size> priv = sizesOf(*ch_, AIMAGE_FORMAT_PRIVATE);
  const Size want{int32_t(mode.width), int32_t(mode.height)};
  int32_t fmt = int32_t(mode.key);
  if (fmt == AIMAGE_FORMAT_PRIVATE && luma_) {
    fmt = 0;
    note_ = "Private needs the GPU decoder, auto instead";
  }
  if (fmt && !isRaw(fmt) && !has(sizesOf(*ch_, fmt), want)) {
    note_ = std::string("no ") + mode.format + " at this size, auto instead";
    fmt = 0;
  }
  if (!fmt) fmt = has(priv, want) && !luma_ ? AIMAGE_FORMAT_PRIVATE : AIMAGE_FORMAT_YUV_420_888;
  fmt_ = fmt;
  size_ = isRaw(fmt) ? largest(sizesOf(*ch_, fmt)) : want;
  preview_ = previewFor(*ch_, size_);
  rl.unlock();
  {
    std::lock_guard<std::mutex> l(levelMu_);
    std::fill(levels_, levels_ + 4, 0.f);
    levels_[4] = 1023;
    std::fill(levels_ + 5, levels_ + 9, 1.f);
  }
  {
    std::lock_guard<std::mutex> l(cbMu_);
    sink_ = std::move(sink);
    ended_ = std::move(ended);
    live_ = true;
  }
  frames_ = 0;
  dropped_ = 0;
  corrupt_ = 0;
  images_ = 0;
  n_ = 0;
  if (!open(why) || !configure(why)) {
    stop();
    return false;
  }
  say("open camera %s: %dx%d %s, preview %dx%d, fps [%d,%d]%s%s", id_.c_str(), size_.w, size_.h, nameOf(fmt_), preview_.w, preview_.h, fps_.first,
      fps_.second, note_.empty() ? "" : "; ", note_.c_str());
  return true;
}

bool AndroidCamera::open(std::string& why) {
  devCb_ = {};
  devCb_.context = this;
  devCb_.onDisconnected = onDeviceDisconnected;
  devCb_.onError = onDeviceError;
  const camera_status_t st = ACameraManager_openCamera(manager(), id_.c_str(), &devCb_, &dev_);
  if (st != ACAMERA_OK || !dev_) {
    dev_ = nullptr;
    why = st == ACAMERA_ERROR_PERMISSION_DENIED ? "the camera's permission is not given"
          : st == ACAMERA_ERROR_CAMERA_IN_USE || st == ACAMERA_ERROR_MAX_CAMERA_IN_USE ? "the camera is in use by another app"
                                                                                       : "the camera " + id_ + " did not open (" + std::to_string(st) + ")";
    return false;
  }
  // ai: the reader: a format read on the CPU (Y8, RAW) a few images, CPU reads; the C's YUV CPU reads; the GPU's
  // ai: sampled in place (PRIVATE; YUV with CPU reads too, so either decoder can take it)
  const bool cpu = onCpu(fmt_) || luma_;
  for (int k = 0; k < 2; k++) {
    const uint64_t usage = cpu ? AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
                               : AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | (fmt_ == AIMAGE_FORMAT_PRIVATE ? 0 : AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN);
    AImageReader* r = nullptr;
    const media_status_t ms = AImageReader_newWithUsage(size_.w, size_.h, fmt_, usage, onCpu(fmt_) ? PLANE_IMAGES : MAX_IMAGES, &r);
    if (ms == AMEDIA_OK && r) {
      takeReader(r);
      break;
    }
    if (fmt_ != AIMAGE_FORMAT_PRIVATE) {
      why = std::string("the camera's frames cannot be read (") + nameOf(fmt_) + ", " + std::to_string(ms) + ")";
      return false;
    }
    say("camera: PRIVATE readers refused, YUV_420_888 instead");
    std::lock_guard<std::mutex> rl(repMu_);
    fmt_ = AIMAGE_FORMAT_YUV_420_888;
  }
  return true;
}

// ai: a reader made in its place (the session's thread, none held): reading from now on, its listener and window set
void AndroidCamera::takeReader(AImageReader* r) {
  {
    std::lock_guard<std::mutex> l(imgMu_);
    reader_ = r;
    reading_ = true;
  }
  AImageReader_ImageListener li{this, onImages};
  AImageReader_setImageListener(reader_, &li);
  AImageReader_getWindow(reader_, &readerWin_);
}

// ai: The reader let go (the session's thread): its listener cleared, reading_ cleared under imgMu_ (a listener running
// ai: waited for; one queued returns at once, imgMu_'s comment), then, where holds is set, a second for the receiver to
// ai: hand back what it holds (its images go with the reader), then deleted: its looper stopped, so nothing of it runs
// ai: after.
void AndroidCamera::dropReader(bool holds) {
  if (!reader_) return;
  AImageReader_setImageListener(reader_, nullptr);
  AImageReader* r = nullptr;
  {
    std::lock_guard<std::mutex> l(imgMu_);
    reading_ = false;
    r = reader_;
    reader_ = nullptr;
  }
  readerWin_ = nullptr;
  if (holds) {
    std::unique_lock<std::mutex> l(holdMu_);
    holdCv_.wait_for(l, std::chrono::seconds(1), [&] { return held_.empty(); });
    if (!held_.empty()) say("camera: %zu frames still held as the reader closes", held_.size());
    for (auto& h : held_) AImage_delete(h.second);
    held_.clear();
  }
  AImageReader_delete(r);
}

void AndroidCamera::teardownSession() {
  std::lock_guard<std::mutex> l(reqMu_);
  if (cs_) {
    ACameraCaptureSession_stopRepeating(cs_);
    ACameraCaptureSession_close(cs_);
    cs_ = nullptr;
  }
  if (req_) {
    ACaptureRequest_free(req_);
    req_ = nullptr;
  }
  if (tgtReader_) ACameraOutputTarget_free(tgtReader_);
  if (tgtPreview_) ACameraOutputTarget_free(tgtPreview_);
  if (outs_) {
    if (outReader_) ACaptureSessionOutputContainer_remove(outs_, outReader_);
    if (outPreview_) ACaptureSessionOutputContainer_remove(outs_, outPreview_);
    ACaptureSessionOutputContainer_free(outs_);
  }
  if (outReader_) ACaptureSessionOutput_free(outReader_);
  if (outPreview_) ACaptureSessionOutput_free(outPreview_);
  tgtReader_ = tgtPreview_ = nullptr;
  outs_ = nullptr;
  outReader_ = outPreview_ = nullptr;
  if (inSession_) ANativeWindow_release(inSession_);
  inSession_ = nullptr;
}

// ai: The capture session (the reader and, where one is given, the preview) and the repeating request (the file's
// ai: header). PRIVATE refused by the session: YUV_420_888 in its place, the reader made again.
bool AndroidCamera::configure(std::string& why) {
  teardownSession();
  ANativeWindow* pw = nullptr;
  {
    std::lock_guard<std::mutex> l(winMu_);
    if (window_) {
      pw = window_;
      ANativeWindow_acquire(pw);
    }
  }
  std::lock_guard<std::mutex> l(reqMu_);
  inSession_ = pw;
  ACaptureSessionOutputContainer_create(&outs_);
  ACaptureSessionOutput_create(readerWin_, &outReader_);
  ACaptureSessionOutputContainer_add(outs_, outReader_);
  if (pw) {
    ACaptureSessionOutput_create(pw, &outPreview_);
    ACaptureSessionOutputContainer_add(outs_, outPreview_);
  }
  csCb_ = {};
  csCb_.context = this;
  csCb_.onClosed = onSessionClosed;
  csCb_.onReady = onSessionReady;
  csCb_.onActive = onSessionActive;
  // ai: the call can fail with the camera service's side gone (2026-10-06, the S26: "Error configuring streams: Broken
  // ai: pipe"): a failure the phase says, and Start tries again
  camera_status_t st = ACameraDevice_createCaptureSession(dev_, outs_, &csCb_, &cs_);
  if (st != ACAMERA_OK && fmt_ == AIMAGE_FORMAT_PRIVATE) {
    say("camera: PRIVATE %dx%d refused, YUV_420_888 instead", size_.w, size_.h);
    cs_ = nullptr;
    ACaptureSessionOutputContainer_remove(outs_, outReader_);
    ACaptureSessionOutput_free(outReader_);
    outReader_ = nullptr;
    // ai: the PRIVATE reader's images held by the receiver stay its own (each AImage keeps its reader's object; the
    // ai: ingest its buffer), let go at their release
    dropReader(false);
    {
      std::lock_guard<std::mutex> rl(repMu_);
      fmt_ = AIMAGE_FORMAT_YUV_420_888;
    }
    const uint64_t usage = luma_ ? AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN : AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
    AImageReader* r = nullptr;
    if (AImageReader_newWithUsage(size_.w, size_.h, fmt_, usage, MAX_IMAGES, &r) != AMEDIA_OK || !r) {
      why = "the camera's frames cannot be read";
      return false;
    }
    takeReader(r);
    ACaptureSessionOutput_create(readerWin_, &outReader_);
    ACaptureSessionOutputContainer_add(outs_, outReader_);
    st = ACameraDevice_createCaptureSession(dev_, outs_, &csCb_, &cs_);
  }
  if (st != ACAMERA_OK || !cs_) {
    cs_ = nullptr;
    why = "the camera refused " + std::to_string(size_.w) + "x" + std::to_string(size_.h) + " (" + std::to_string(st) + ")";
    return false;
  }
  if (ACameraDevice_createCaptureRequest(dev_, TEMPLATE_RECORD, &req_) != ACAMERA_OK || !req_) {
    req_ = nullptr;
    why = "the camera made no request";
    return false;
  }
  ACameraOutputTarget_create(readerWin_, &tgtReader_);
  ACaptureRequest_addTarget(req_, tgtReader_);
  if (pw) {
    ACameraOutputTarget_create(pw, &tgtPreview_);
    ACaptureRequest_addTarget(req_, tgtPreview_);
  }
  // ai: the enhancements off where the lens lists the off mode (the header); debug.lizard.camx keeps some as the
  // ai: template has them
  std::set<std::string> keep;
  {
    const std::string k = prop("debug.lizard.camx");
    size_t a = 0;
    while (a <= k.size()) {
      size_t b = k.find(',', a);
      if (b == std::string::npos) b = k.size();
      std::string t = k.substr(a, b - a);
      t.erase(0, t.find_first_not_of(' '));
      t.erase(t.find_last_not_of(' ') + 1);
      if (!t.empty()) keep.insert(t);
      a = b + 1;
    }
  }
  const auto off = [&](const char* name) { return !(keep.count("all") || keep.count("0") || keep.count(name)); };
  const auto u8 = [&](uint32_t tag, uint8_t v) { ACaptureRequest_setEntry_u8(req_, tag, 1, &v); };
  if (off("nr") && ch_->lists(ACAMERA_NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES, ACAMERA_NOISE_REDUCTION_MODE_OFF))
    u8(ACAMERA_NOISE_REDUCTION_MODE, ACAMERA_NOISE_REDUCTION_MODE_OFF);
  if (off("edge") && ch_->lists(ACAMERA_EDGE_AVAILABLE_EDGE_MODES, ACAMERA_EDGE_MODE_OFF)) u8(ACAMERA_EDGE_MODE, ACAMERA_EDGE_MODE_OFF);
  if (off("hot") && ch_->lists(ACAMERA_HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES, ACAMERA_HOT_PIXEL_MODE_OFF)) u8(ACAMERA_HOT_PIXEL_MODE, ACAMERA_HOT_PIXEL_MODE_OFF);
  if (off("ca") && ch_->lists(ACAMERA_COLOR_CORRECTION_AVAILABLE_ABERRATION_MODES, ACAMERA_COLOR_CORRECTION_ABERRATION_MODE_OFF))
    u8(ACAMERA_COLOR_CORRECTION_ABERRATION_MODE, ACAMERA_COLOR_CORRECTION_ABERRATION_MODE_OFF);
  if (off("face") && ch_->lists(ACAMERA_STATISTICS_INFO_AVAILABLE_FACE_DETECT_MODES, ACAMERA_STATISTICS_FACE_DETECT_MODE_OFF))
    u8(ACAMERA_STATISTICS_FACE_DETECT_MODE, ACAMERA_STATISTICS_FACE_DETECT_MODE_OFF);
  if (off("scene") && ch_->lists(ACAMERA_CONTROL_AVAILABLE_SCENE_MODES, ACAMERA_CONTROL_SCENE_MODE_DISABLED)) {
    u8(ACAMERA_CONTROL_MODE, ACAMERA_CONTROL_MODE_AUTO);
    u8(ACAMERA_CONTROL_SCENE_MODE, ACAMERA_CONTROL_SCENE_MODE_DISABLED);
  }
  if (off("stats")) {
    if (ch_->lists(ACAMERA_STATISTICS_INFO_AVAILABLE_LENS_SHADING_MAP_MODES, ACAMERA_STATISTICS_LENS_SHADING_MAP_MODE_OFF))
      u8(ACAMERA_STATISTICS_LENS_SHADING_MAP_MODE, ACAMERA_STATISTICS_LENS_SHADING_MAP_MODE_OFF);
    if (ch_->lists(ACAMERA_STATISTICS_INFO_AVAILABLE_HOT_PIXEL_MAP_MODES, ACAMERA_STATISTICS_HOT_PIXEL_MAP_MODE_OFF))
      u8(ACAMERA_STATISTICS_HOT_PIXEL_MAP_MODE, ACAMERA_STATISTICS_HOT_PIXEL_MAP_MODE_OFF);
    if (ch_->lists(ACAMERA_STATISTICS_INFO_AVAILABLE_OIS_DATA_MODES, ACAMERA_STATISTICS_OIS_DATA_MODE_OFF))
      u8(ACAMERA_STATISTICS_OIS_DATA_MODE, ACAMERA_STATISTICS_OIS_DATA_MODE_OFF);
  }
  if (!keep.empty()) {
    std::string k;
    for (const auto& x : keep) k += (k.empty() ? "" : ",") + x;
    say("camx: kept as the template has them: %s", k.c_str());
  }
  applyZoom(req_);
  if (fps_.second > 0) {
    const int32_t r[2] = {fps_.first, fps_.second};
    ACaptureRequest_setEntry_i32(req_, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE, 2, r);
  }
  if (ch_->lists(ACAMERA_CONTROL_AF_AVAILABLE_MODES, ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO))
    u8(ACAMERA_CONTROL_AF_MODE, ACAMERA_CONTROL_AF_MODE_CONTINUOUS_VIDEO);
  // ai: stabilisation off: it warps and crops each frame differently, which the border would have to chase
  if (ch_->lists(ACAMERA_CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE_OFF))
    u8(ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE_OFF);
  if (ch_->lists(ACAMERA_LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF))
    u8(ACAMERA_LENS_OPTICAL_STABILIZATION_MODE, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE_OFF);
  // ai: the focus region on the code kept across a stream made again (the next focusOn moves it)
  focused_ = false;
  if (prop("debug.lizard.camdump") == "1") {
    int32_t n = 0;
    const uint32_t* tags = nullptr;
    ACaptureRequest_getAllTags(req_, &n, &tags);
    for (int32_t i = 0; i < n; i++) {
      ACameraMetadata_const_entry e{};
      if (ACaptureRequest_getConstEntry(req_, tags[i], &e) != ACAMERA_OK) continue;
      std::string v;
      for (uint32_t k = 0; k < e.count && k < 16; k++) {
        char b[48];
        switch (e.type) {
          case ACAMERA_TYPE_BYTE: std::snprintf(b, sizeof b, "%d", e.data.u8[k]); break;
          case ACAMERA_TYPE_INT32: std::snprintf(b, sizeof b, "%d", e.data.i32[k]); break;
          case ACAMERA_TYPE_INT64: std::snprintf(b, sizeof b, "%lld", static_cast<long long>(e.data.i64[k])); break;
          case ACAMERA_TYPE_FLOAT: std::snprintf(b, sizeof b, "%g", e.data.f[k]); break;
          case ACAMERA_TYPE_DOUBLE: std::snprintf(b, sizeof b, "%g", e.data.d[k]); break;
          default: std::snprintf(b, sizeof b, "?"); break;
        }
        v += (k ? "," : "") + std::string(b);
      }
      say("request: 0x%08x = [%s]", tags[i], v.c_str());
    }
  }
  capCb_ = {};
  capCb_.context = this;
  capCb_.onCaptureCompleted = onCompleted;
  capCb_.onCaptureFailed = onFailed;
  capCb_.onCaptureSequenceAborted = onAborted;
  {
    std::lock_guard<std::mutex> k(lastMu_);
    fresh_ = true;
    delaySeq_ = -1;
    lastExp_ = lastDur_ = 0;
  }
  if (ACameraCaptureSession_setRepeatingRequest(cs_, &capCb_, 1, &req_, nullptr) != ACAMERA_OK) {
    why = "the camera stopped: the repeating request was refused";
    return false;
  }
  if (afMax_ < 1 || active_.size() < 4) say("focus: no autofocus region on this lens (max regions %d)", afMax_);
  return true;
}

// ai: the zoom into a request: CONTROL_ZOOM_RATIO (API 30), else the active array's centred crop at 1 / zoom
void AndroidCamera::applyZoom(ACaptureRequest* q) {
  const float z = std::clamp(zoom_.load(), zoomLo_, zoomHi_);
  if (ratio_) {
    ACaptureRequest_setEntry_float(q, ACAMERA_CONTROL_ZOOM_RATIO, 1, &z);
  } else if (active_.size() >= 4 && zoomHi_ > 1) {
    const int32_t w = int32_t(active_[2] / z), h = int32_t(active_[3] / z);
    const int32_t r[4] = {(active_[2] - w) / 2, (active_[3] - h) / 2, w, h};
    ACaptureRequest_setEntry_i32(q, ACAMERA_SCALER_CROP_REGION, 4, r);
  }
}

// ai: the repeating request issued again as it stands (reqMu_ held)
void AndroidCamera::repeat() {
  if (cs_ && req_ && ACameraCaptureSession_setRepeatingRequest(cs_, &capCb_, 1, &req_, nullptr) != ACAMERA_OK) say("camera: the request was refused");
}

void AndroidCamera::stop() {
  {
    std::lock_guard<std::mutex> l(cbMu_);
    live_ = false;
  }
  if (cs_ || req_ || outs_) teardownSession();
  if (dev_) {
    ACameraDevice_close(dev_);
    dev_ = nullptr;
    say("close camera");
  }
  {
    std::lock_guard<std::mutex> l(cbMu_);
    sink_ = nullptr;
    ended_ = nullptr;
  }
  dropReader(true);
}

void AndroidCamera::deviceGone(const char* why) {
  Ended e;
  {
    std::lock_guard<std::mutex> l(cbMu_);
    if (!live_) return;
    live_ = false;
    e = ended_;
  }
  say("camera: %s", why);
  if (e) e(why);
}

void AndroidCamera::previewTo(void* w) {
  ANativeWindow* nw = static_cast<ANativeWindow*>(w);
  {
    std::lock_guard<std::mutex> l(winMu_);
    if (nw == window_) return;
    if (nw) ANativeWindow_acquire(nw);
    if (window_) ANativeWindow_release(window_);
    window_ = nw;
  }
  // ai: a stream running: its capture session made again with the preview (or without the one gone), the reader kept
  if (dev_ && reader_) {
    std::string why;
    if (!configure(why)) {
      say("camera: the preview's change: %s", why.c_str());
      Ended e;
      {
        std::lock_guard<std::mutex> l(cbMu_);
        e = live_ ? ended_ : nullptr;
        live_ = false;
      }
      if (e) e("The camera stopped: " + why);
    }
  }
}

// ai: The autofocus region on the code: the receiver's box of the codes it found in the last second (camera pixels from
// ai: the frame's centre) as the region, so the camera's own contrast autofocus judges the code and not the room. 3A
// ai: regions are in the post-zoom field's coordinates (CONTROL_ZOOM_RATIO's rule; the crop region's field below API
// ai: 30), scaled to the active array; the stream is that field's largest centred part of its shape. A dead band of a
// ai: tenth of the box's size, so a hand's jitter does not issue the repeating request again each second. At an
// ai: angle the found share went 67 to 97% on the S26.
void AndroidCamera::focusOn(const double box[4]) {
  if (afMax_ < 1 || active_.size() < 4) return;
  const double span = std::max(box[2] - box[0], box[3] - box[1]);
  if (span <= 0) return;
  if (focused_) {
    bool near = true;
    for (int i = 0; i < 4; i++) near = near && std::fabs(box[i] - focusBox_[i]) <= 0.1 * span;
    if (near) return;
  }
  std::copy(box, box + 4, focusBox_);
  focused_ = true;
  std::lock_guard<std::mutex> l(reqMu_);
  if (!req_ || !cs_ || size_.w <= 0) return;
  const double wa = active_[2], ha = active_[3];
  double k = std::min(wa / size_.w, ha / size_.h);
  if (!ratio_) k /= std::max(1.f, zoom_.load());
  const auto clampi = [](long v, long lo, long hi) { return int32_t(std::max(lo, std::min(hi, v))); };
  const int32_t x0 = clampi(std::lround(wa / 2 + box[0] * k), 0, long(wa) - 1), y0 = clampi(std::lround(ha / 2 + box[1] * k), 0, long(ha) - 1);
  const int32_t x1 = clampi(std::lround(wa / 2 + box[2] * k), x0 + 1, long(wa)), y1 = clampi(std::lround(ha / 2 + box[3] * k), y0 + 1, long(ha));
  // ai: (xmin, ymin, xmax, ymax, weight) on the active array, xmax and ymax exclusive; the weight the most
  // ai: (METERING_WEIGHT_MAX)
  const int32_t reg[5] = {x0, y0, x1, y1, 1000};
  ACaptureRequest_setEntry_i32(req_, ACAMERA_CONTROL_AF_REGIONS, 5, reg);
  repeat();
  say("focus: on the code, %d,%d to %d,%d of the %dx%d array", x0, y0, x1, y1, int(wa), int(ha));
}

// ai: One frame `us` longer, so every frame after it starts that much later: Camera2 takes a frame's duration from
// ai: the request only with auto-exposure off, so the one frame carries the sensor's last exposure and sensitivity
// ai: itself; the repeating request, with auto-exposure, goes on after it. lock: the phase lock's (its fate reported)
bool AndroidCamera::delay(int64_t us, bool lock, bool quiet) {
  int64_t exp = 0, dur = 0;
  int32_t iso = 0;
  {
    std::lock_guard<std::mutex> l(lastMu_);
    exp = lastExp_;
    dur = lastDur_;
    iso = lastIso_;
  }
  if (!exp || !dur || !iso || us <= 0) return false;
  std::lock_guard<std::mutex> l(reqMu_);
  if (!cs_ || !req_) return false;
  ACaptureRequest* one = ACaptureRequest_copy(req_);
  if (!one) return false;
  const uint8_t aeOff = ACAMERA_CONTROL_AE_MODE_OFF;
  const int64_t frame = dur + us * 1000;
  ACaptureRequest_setEntry_u8(one, ACAMERA_CONTROL_AE_MODE, 1, &aeOff);
  ACaptureRequest_setEntry_i64(one, ACAMERA_SENSOR_EXPOSURE_TIME, 1, &exp);
  ACaptureRequest_setEntry_i32(one, ACAMERA_SENSOR_SENSITIVITY, 1, &iso);
  ACaptureRequest_setEntry_i64(one, ACAMERA_SENSOR_FRAME_DURATION, 1, &frame);
  ACaptureRequest_setUserContext(one, lock ? &LOCK_MARK : &NUDGE_MARK);
  int seq = -1;
  const camera_status_t st = ACameraCaptureSession_capture(cs_, &capCb_, 1, &one, &seq);
  ACaptureRequest_free(one);
  if (st != ACAMERA_OK) {
    say("delay: refused (%d)", st);
    return false;
  }
  if (lock) {
    std::lock_guard<std::mutex> k(lastMu_);
    delaySeq_ = seq;
  }
  if (!quiet)
    say("delay: one frame of %.3f ms asked (%.3f and %.3f more), exposure %.2f ms, iso %d", frame / 1e6, dur / 1e6, us / 1e3, exp / 1e6, iso);
  return true;
}

// ai: a RAW frame's levels a 2 x 2 position (row-major: (0,0), (1,0), (0,1), (1,1)): black[4], the white level, the white
// ai: balance's gain[4] (COLOR_CORRECTION_GAINS by the CFA; 1 where none or the sensor has no Bayer CFA): the result's
// ai: dynamic black and white levels where it gives them, else the characteristics'
void AndroidCamera::levels(const ACameraMetadata* r) {
  float o[9];
  ACameraMetadata_const_entry e{};
  const bool dyn = r && ACameraMetadata_getConstEntry(r, ACAMERA_SENSOR_DYNAMIC_BLACK_LEVEL, &e) == ACAMERA_OK && e.count >= 4;
  const std::vector<int32_t> pat = ch_->i32s(ACAMERA_SENSOR_BLACK_LEVEL_PATTERN);
  for (int p = 0; p < 4; p++) o[p] = dyn ? e.data.f[p] : pat.size() >= 4 ? float(pat[size_t(p)]) : 0.f;
  int32_t white = 0;
  if (!one(r, ACAMERA_SENSOR_DYNAMIC_WHITE_LEVEL, white)) {
    const std::vector<int32_t> w = ch_->i32s(ACAMERA_SENSOR_INFO_WHITE_LEVEL);
    white = w.empty() ? 1023 : w[0];
  }
  o[4] = float(white);
  ACameraMetadata_const_entry g{};
  const bool gains = r && ACameraMetadata_getConstEntry(r, ACAMERA_COLOR_CORRECTION_GAINS, &g) == ACAMERA_OK && g.count >= 4;
  // ai: each position's colour as the gains' index (R, G even, G odd, B): RGGB, GRBG, GBRG, BGGR
  static const int COL[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
  const std::vector<uint8_t> cfa = ch_->u8s(ACAMERA_SENSOR_INFO_COLOR_FILTER_ARRANGEMENT);
  const int c = cfa.empty() ? -1 : cfa[0];
  for (int p = 0; p < 4; p++) o[5 + p] = gains && c >= 0 && c < 4 ? g.data.f[COL[c][p]] : 1.f;
  std::lock_guard<std::mutex> l(levelMu_);
  std::copy(o, o + 9, levels_);
}

void AndroidCamera::completed(ACaptureRequest* q, const ACameraMetadata* r) {
  std::lock_guard<std::mutex> cb(cbMu_);
  if (!live_) return;
  int64_t ts = 0, exp = 0, dur = 0, skew = 0;
  int32_t iso = 0;
  one(r, ACAMERA_SENSOR_TIMESTAMP, ts);
  one(r, ACAMERA_SENSOR_EXPOSURE_TIME, exp);
  one(r, ACAMERA_SENSOR_SENSITIVITY, iso);
  one(r, ACAMERA_SENSOR_FRAME_DURATION, dur);
  one(r, ACAMERA_SENSOR_ROLLING_SHUTTER_SKEW, skew);
  void* mark = nullptr;
  ACaptureRequest_getUserContext(q, &mark);
  const bool delayed = mark == &LOCK_MARK || mark == &NUDGE_MARK;
  bool fresh = false;
  {
    std::lock_guard<std::mutex> l(lastMu_);
    if (mark == &LOCK_MARK) delaySeq_ = -1;
    // ai: a delay is asked from the camera's own frame: a delayed frame's duration is not it
    if (!delayed) {
      lastExp_ = exp;
      lastDur_ = dur;
      lastIso_ = iso;
      lastSkew_ = skew;
      one(r, ACAMERA_LENS_FOCUS_DISTANCE, lastFocus_);
      float z = 1;
      if (one(r, ACAMERA_CONTROL_ZOOM_RATIO, z)) lastZoom_ = z;
      lastLens_ = textOf(r, ACAMERA_LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
    }
    fresh = fresh_;
    fresh_ = false;
  }
  // ai: a RAW format's levels from the newest result each half second (the white balance moves them)
  if (isRaw(fmt_) && levelsAt_++ % 30 == 0) levels(r);
  if (watch_ > 0 && prevTs_) {
    char b[48];
    std::snprintf(b, sizeof b, " %.3f", (ts - prevTs_) / 1e6);
    gaps_ += b;
    if (delayed) {
      std::snprintf(b, sizeof b, "*(%.3f)", dur / 1e6);
      gaps_ += b;
    }
    if (--watch_ == 0) {
      say("delay: intervals ms%s", gaps_.c_str());
      gaps_.clear();
    }
  }
  prevTs_ = ts;
  if (n_ % 30 == 15) {
    const std::string p = prop("debug.lizard.nudge");
    if (p != nudge_) {
      nudge_ = p;
      const long long us = std::atoll(p.substr(0, p.find('.')).c_str());
      if (us > 0) {
        watch_ = 30;
        delay(us, false, false);
      }
    }
  }
  if (each_) {
    Capture c;
    c.ns = ts;
    c.delayed = delayed;
    c.frameNs = dur;
    c.exposureNs = exp;
    c.skewNs = skew;
    c.iso = iso;
    c.fresh = fresh;
    each_(c);
  }
  if (n_++ % 300 == 0) {
    const int64_t now = steadyNs();
    const uint64_t im = images_.load();
    size_t held = 0;
    {
      std::lock_guard<std::mutex> l(holdMu_);
      held = held_.size();
    }
    if (lineAt_) say("camera: %.1f captures a second, %.1f images, %zu held", 300e9 / double(now - lineAt_), double(im - imagesAt_) * 1e9 / double(now - lineAt_), held);
    lineAt_ = now;
    imagesAt_ = im;
    // ai: readout: the sensor's first row to its last (the rolling shutter), which with the exposure is how long a
    // ai: capture looks at the display
    float zr = 1;
    if (!one(r, ACAMERA_CONTROL_ZOOM_RATIO, zr)) zr = 1;
    float fd = 0;
    one(r, ACAMERA_LENS_FOCUS_DISTANCE, fd);
    const std::string lens = textOf(r, ACAMERA_LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
    say("capture: exposure %.2f ms, iso %d, frame %.2f ms, readout %.2f ms, focus %.2f D (af state %d), eis %d, ois %d, nr %d, edge %d, face %d, "
        "scene %d, hot %d, tone %d, shade %d, dist %d, zoom %.2f, lens %s",
        exp / 1e6, iso, dur / 1e6, skew / 1e6, fd, byteOr(r, ACAMERA_CONTROL_AF_STATE), byteOr(r, ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE),
        byteOr(r, ACAMERA_LENS_OPTICAL_STABILIZATION_MODE), byteOr(r, ACAMERA_NOISE_REDUCTION_MODE), byteOr(r, ACAMERA_EDGE_MODE),
        byteOr(r, ACAMERA_STATISTICS_FACE_DETECT_MODE), byteOr(r, ACAMERA_CONTROL_SCENE_MODE), byteOr(r, ACAMERA_HOT_PIXEL_MODE),
        byteOr(r, ACAMERA_TONEMAP_MODE), byteOr(r, ACAMERA_SHADING_MODE), byteOr(r, ACAMERA_DISTORTION_CORRECTION_MODE), zr,
        lens.empty() ? "-" : lens.c_str());
  }
}

// ai: the lock is told what became of its delay: its frame failed (captured or not), or its sequence aborted
void AndroidCamera::failed(ACaptureRequest* q, ACameraCaptureFailure* f) {
  std::lock_guard<std::mutex> cb(cbMu_);
  void* mark = nullptr;
  ACaptureRequest_getUserContext(q, &mark);
  if (!live_ || mark != &LOCK_MARK) return;
  {
    std::lock_guard<std::mutex> l(lastMu_);
    delaySeq_ = -1;
  }
  if (fate_) fate_(f && f->wasImageCaptured);
}

void AndroidCamera::aborted(int seq) {
  std::lock_guard<std::mutex> cb(cbMu_);
  if (!live_) return;
  {
    std::lock_guard<std::mutex> l(lastMu_);
    if (delaySeq_ < 0 || seq != delaySeq_) return;
    delaySeq_ = -1;
  }
  if (fate_) fate_(false);
}

// ai: Every image the reader holds, in order (the reader's thread, at the display's priority: the frames thread's
// ai: THREAD_PRIORITY_DISPLAY before). A format developed (RAW): plane 0 handed over as a RawFrame cropped to the zoom's
// ai: centre at the mode's shape, the image let go after the sink's call; PRIVATE and YUV for a receiver that samples
// ai: the buffer: the image's AHardwareBuffer, held until release; else plane 0 (Y) as luma.
void AndroidCamera::images() {
  if (!prioritySet_) {
    prioritySet_ = true;
    setpriority(PRIO_PROCESS, 0, -4);
  }
  std::lock_guard<std::mutex> l(imgMu_);
  for (;;) {
    AImage* img = nullptr;
    // ai: a reader being let go is not touched (imgMu_'s comment): its images go with it
    if (!reading_ || !reader_ || AImageReader_acquireNextImage(reader_, &img) != AMEDIA_OK || !img) return;
    images_++;
    Sink sink;
    {
      std::lock_guard<std::mutex> cb(cbMu_);
      if (live_ && reading_) sink = sink_;
    }
    int32_t w = 0, h = 0, fmt = 0;
    int64_t ts = 0;
    AImage_getWidth(img, &w);
    AImage_getHeight(img, &h);
    AImage_getFormat(img, &fmt);
    AImage_getTimestamp(img, &ts);
    if (!sink || w <= 0 || h <= 0) {
      AImage_delete(img);
      dropped_++;
      continue;
    }
    Frame f;
    f.ns = ts;
    f.seq = ++seq_;
    if (isRaw(fmt)) {
      uint8_t* data = nullptr;
      int len = 0, rowStride = 0, pixelStride = 0;
      if (AImage_getPlaneData(img, 0, &data, &len) != AMEDIA_OK || !data || AImage_getPlaneRowStride(img, 0, &rowStride) != AMEDIA_OK) {
        AImage_delete(img);
        corrupt_++;
        continue;
      }
      AImage_getPlanePixelStride(img, 0, &pixelStride);
      // ai: RAW is the lens's whole array (the ISP's zoom crops the processed streams alone): the zoom's centred crop
      // ai: at the mode's shape, a 2 x 2 bin where that still holds the mode's size
      const int64_t rw = mode_.width, rh = mode_.height;
      const float z = std::max(1.f, zoom_.load());
      int64_t cw = int64_t(w / z), chh = int64_t(h / z);
      if (cw * rh > chh * rw) cw = chh * rw / rh;
      else chh = cw * rh / rw;
      cw &= ~int64_t(7);
      chh &= ~int64_t(7);
      const int64_t x0 = ((w - cw) / 2) & ~int64_t(1), y0 = ((h - chh) / 2) & ~int64_t(1);
      const uint32_t bin = cw / 2 >= rw && chh / 2 >= rh ? 2 : 1;
      RawFrame rf;
      rf.plane = data;
      rf.rowStride = uint32_t(rowStride);
      rf.pixelStride = uint32_t(pixelStride > 0 ? pixelStride : 2);
      rf.format = fmt;
      rf.x0 = uint32_t(x0);
      rf.y0 = uint32_t(y0);
      rf.bin = bin;
      {
        std::lock_guard<std::mutex> k(levelMu_);
        std::copy(levels_, levels_ + 9, rf.levels);
      }
      // ai: the plane holds the crop's rows (the receiver reads them in its push)
      if (cw < bin || chh < bin || int64_t(y0 + chh - 1) * rowStride + int64_t(rawRowBytes(fmt, uint32_t(x0 + cw), rf.pixelStride)) > len) {
        AImage_delete(img);
        corrupt_++;
        continue;
      }
      f.raw = &rf;
      f.width = uint32_t(cw / bin);
      f.height = uint32_t(chh / bin);
      sink(f);
      AImage_delete(img);
      frames_++;
      continue;
    }
    if (!luma_ && fmt != AIMAGE_FORMAT_Y8) {
      AHardwareBuffer* hb = nullptr;
      if (AImage_getHardwareBuffer(img, &hb) != AMEDIA_OK || !hb) {
        AImage_delete(img);
        corrupt_++;
        continue;
      }
      const uint64_t hold = ++nextHold_;
      {
        std::lock_guard<std::mutex> k(holdMu_);
        held_[hold] = img;
      }
      f.hb = hb;
      f.hold = hold;
      f.width = uint32_t(w);
      f.height = uint32_t(h);
      // ai: the receiver lets it go (release), at its ingest's end or at once where it drops the frame
      sink(f);
      frames_++;
      continue;
    }
    uint8_t* data = nullptr;
    int len = 0, rowStride = 0;
    if (AImage_getPlaneData(img, 0, &data, &len) != AMEDIA_OK || !data || AImage_getPlaneRowStride(img, 0, &rowStride) != AMEDIA_OK ||
        rowStride < w || int64_t(rowStride) * (h - 1) + w > len) {
      AImage_delete(img);
      corrupt_++;
      continue;
    }
    f.luma = data;
    f.width = uint32_t(w);
    f.height = uint32_t(h);
    f.stride = uint32_t(rowStride);
    sink(f);
    AImage_delete(img);
    frames_++;
  }
}

// ai: what was asked and given (Engine's CamInfo) and the newest capture (Engine.cameraNow), as the stats'
// ai: camera.report and a replay's meta
std::string AndroidCamera::report() {
  int32_t fmt = 0;
  Size size, preview;
  std::string note;
  {
    std::lock_guard<std::mutex> l(repMu_);
    fmt = fmt_;
    size = size_;
    preview = preview_;
    note = note_;
  }
  int64_t exp = 0, dur = 0, skew = 0;
  int32_t iso = 0;
  std::string lens;
  float zr = 1;
  {
    std::lock_guard<std::mutex> l(lastMu_);
    exp = lastExp_;
    dur = lastDur_;
    skew = lastSkew_;
    iso = lastIso_;
    lens = lastLens_;
    zr = lastZoom_;
  }
  int64_t minNs = 0;
  {
    ACameraMetadata_const_entry e{};
    if (ch_->get(ACAMERA_SCALER_AVAILABLE_MIN_FRAME_DURATIONS, e))
      for (uint32_t i = 0; i + 3 < e.count; i += 4)
        if (e.data.i64[i] == fmt && e.data.i64[i + 1] == size.w && e.data.i64[i + 2] == size.h) minNs = e.data.i64[i + 3];
  }
  const std::vector<int32_t> so = ch_->i32s(ACAMERA_SENSOR_ORIENTATION);
  char b[1024];
  std::string q;
  for (char c : note) {
    if (c == '"' || c == '\\') q += '\\';
    q += c;
  }
  std::snprintf(b, sizeof b,
                "{\"format\":\"%s\",\"size\":\"%dx%d\",\"preview\":\"%dx%d\",\"fps\":\"[%d,%d]\",\"minFrameMs\":%.3f,\"sensorOrientation\":%d,"
                "\"note\":\"%s\",\"exposureMs\":%.3f,\"iso\":%d,\"readoutMs\":%.3f,\"frameMs\":%.3f,\"zoomRatio\":%.2f,\"lens\":%s%s%s,\"lockBar\":\"%s\"}",
                fmt ? nameOf(fmt) : "", size.w, size.h, preview.w, preview.h, fps_.first, fps_.second, minNs / 1e6, so.empty() ? 0 : so[0],
                q.c_str(), exp / 1e6, iso, skew / 1e6, dur / 1e6, zr, lens.empty() ? "null" : "\"", lens.c_str(), lens.empty() ? "" : "\"",
                bar_.c_str());
  return b;
}

}  // namespace

std::vector<Info> androidList(std::vector<std::string>* notes) {
  std::vector<Info> out;
  std::string why;
  for (auto& [id, c] : backCameras(&why)) {
    Info i;
    i.id = "android:" + id;
    i.name = labelOf(id, *c);
    i.backend = "android";
    i.detail = c->lists(ACAMERA_REQUEST_AVAILABLE_CAPABILITIES, ACAMERA_REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA) ? "back, logical" : "back";
    out.push_back(std::move(i));
  }
  if (out.empty() && notes) notes->push_back(why.empty() ? "this phone has no back camera" : why);
  return out;
}

std::string androidStamp() {
  std::string s;
  ACameraIdList* ids = nullptr;
  if (ACameraManager_getCameraIdList(manager(), &ids) == ACAMERA_OK && ids) {
    for (int i = 0; i < ids->numCameras; i++) s += std::string(ids->cameraIds[i]) + ";";
    ACameraManager_deleteCameraIdList(ids);
  }
  return s;
}

std::unique_ptr<Camera> androidOpen(const std::string& id, std::string& why) {
  const std::string cam = id.substr(std::strlen("android:"));
  auto c = charsOf(cam);
  if (!c) {
    why = "no camera " + id;
    return nullptr;
  }
  if (!facingBack(*c)) {
    why = id + " is not a back camera";
    return nullptr;
  }
  return std::make_unique<AndroidCamera>(cam, std::move(c));
}

}  // namespace lizard::cam
