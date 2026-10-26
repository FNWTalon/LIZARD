// ai: Media Foundation's video capture devices (2026-10-25): the UVC backend on Windows (backends.h mfList, mfOpen; the
// ai: test hooks in mf_source.h). Its DLLs (mfplat, mf, mfreadwrite) are loaded from System32 at run time and the six
// ai: functions it calls looked up there, so the app's library loads where Media Foundation is absent (Windows N
// ai: without the Media Feature Pack, a Server without the feature) and the sender with it; everything else is COM's
// ai: vtables, mfapi.h's inline helpers and libmfuuid's GUIDs. Each list and each open camera holds an MFStartup; COM
// ai: is initialised multithreaded on every thread that calls in, and the MTA is kept for the process
// ai: (CoIncrementMTAUsage) since the objects outlive the calls.
// ai: A camera is its device's media source (the MFEnumDeviceSources activate whose symbolic link is the id). modes()
// ai: are the first video stream's native types; start sets the chosen one current on that stream's type handler (the
// ai: device's mode; a rate inside the type's range asked as its MF_MT_FRAME_RATE), then reads through a source
// ai: reader: the native type where luma.h reads it, else (MJPG) a decoded one, NV12 first, Media Foundation's own
// ai: MJPEG decoder put in by the reader. The reader runs asynchronously, a capture thread of the camera's own asking
// ai: for each sample and waiting for its answer (Box, Answers); a sample's buffer is read in place through
// ai: IMF2DBuffer's Lock2D where it is one (its pitch; a negative one, a bottom-up RGB, read upward), else through
// ai: Lock at the type's default stride; the luma goes to the sink in place for NV12, I420, YV12 and L8,
// ai: else converted (luma.h). A frame's ns is its sample time (100 ns units on QPC's clock, as steadyNs is) moved
// ai: onto steadyNs by the offset at the stream's first sample; a gap of more than half a frame in the sample times
// ai: counts the frames it lost as dropped; Stats::fps the rate of the type the device's handler took. Each stream
// ai: has a source of its own: its reader shuts the source down when it goes (stop), and the next stream, or a set on
// ai: a stopped camera, activates the device again; a look at the modes or controls answers from what was last read
// ai: (the modes at open, the controls as they stood when the last source went, put back), so a stopped camera's
// ai: device stays asleep (2026-10-25: the session's look at the controls after a stop had activated it again and
// ai: held it until the camera closed). A start that fails on the source lets that source go, so the next activates
// ai: anew.
// ai: Controls through the source's IAMCameraControl (zoom, focus, exposure: its log2 seconds shown in us) and
// ai: IAMVideoProcAmp (gain, sharpness, backlight, brightness, contrast); the camera rules (camera.cpp's header) at
// ai: start, auto-exposure priority off through IKsControl (KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY, 19, which
// ai: MinGW's ksmedia.h stops short of); every control the backend changed, the rules' and set()'s, put back at stop.
#include "backends.h"
#include "mf_source.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <strmif.h>
#include <ks.h>
#include <ksmedia.h>
#include <ksproxy.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <type_traits>

#include "luma.h"
#include "clock.h"

namespace lizard::cam {

namespace {

using Microsoft::WRL::ComPtr;

const char* const kBlocked = "The camera is blocked: allow desktop apps to use the camera in Windows' privacy settings.";

// ai: KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY: the Windows SDK's next value after FOCAL_LENGTH, UVC's
// ai: auto-exposure priority (0: the frame rate held; 1: the camera may lower it to expose longer)
constexpr ULONG kAutoExposurePriority = 19;
static_assert(KSPROPERTY_CAMERACONTROL_FOCAL_LENGTH == 18, "ksmedia.h's camera-control ids moved");
// ai: PROPSETID_VIDCAP_CAMERACONTROL, the same GUID as IID_IAMCameraControl (libksguid's copy left unlinked)
const GUID kCameraControlSet = {0xc6e13370, 0x30ac, 0x11d0, {0xa1, 0x8c, 0x00, 0xa0, 0xc9, 0x11, 0x89, 0x56}};

std::string hex(HRESULT hr) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%08lX", static_cast<unsigned long>(hr));
  return b;
}

// ai: an HRESULT as the state line words it (after "The camera did not start: " or "The camera stopped: ")
std::string hrWhy(HRESULT hr) {
  switch (hr) {
    case E_ACCESSDENIED: return kBlocked;
    case MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED: return "the camera was disconnected";
    case MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED: return "another app took the camera";
    case MF_E_HW_MFT_FAILED_START_STREAMING: return "the camera would not start streaming (" + hex(hr) + ")";
    case MF_E_SHUTDOWN: return "the camera's source was shut down";
    default: return "Media Foundation error " + hex(hr);
  }
}

std::string utf8(const wchar_t* w) {
  if (!w) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s(n > 0 ? n - 1 : 0, '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
  return s;
}

std::string attrString(IMFAttributes* a, REFGUID key) {
  WCHAR* s = nullptr;
  UINT32 n = 0;
  if (FAILED(a->GetAllocatedString(key, &s, &n)) || !s) return {};
  std::string r = utf8(s);
  CoTaskMemFree(s);
  return r;
}

// ai: The six functions, looked up once a process; the DLLs stay loaded.
struct Api {
  HRESULT(WINAPI* startup)(ULONG, DWORD) = nullptr;
  HRESULT(WINAPI* shutdown)() = nullptr;
  HRESULT(WINAPI* createAttributes)(IMFAttributes**, UINT32) = nullptr;
  HRESULT(WINAPI* createMediaType)(IMFMediaType**) = nullptr;
  HRESULT(WINAPI* enumDeviceSources)(IMFAttributes*, IMFActivate***, UINT32*) = nullptr;
  HRESULT(WINAPI* createReader)(IMFMediaSource*, IMFAttributes*, IMFSourceReader**) = nullptr;
  bool ok = false;
  std::string why;
};

const Api& api() {
  static const Api a = [] {
    Api x;
    HMODULE plat = LoadLibraryExW(L"mfplat.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    HMODULE mf = LoadLibraryExW(L"mf.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    HMODULE rw = LoadLibraryExW(L"mfreadwrite.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!plat || !mf || !rw) {
      x.why = "Media Foundation is not installed (a Windows N edition needs the Media Feature Pack)";
      return x;
    }
    bool all = true;
    auto get = [&](auto& fn, HMODULE m, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(reinterpret_cast<void*>(GetProcAddress(m, name)));
      all = all && fn;
    };
    get(x.startup, plat, "MFStartup");
    get(x.shutdown, plat, "MFShutdown");
    get(x.createAttributes, plat, "MFCreateAttributes");
    get(x.createMediaType, plat, "MFCreateMediaType");
    get(x.enumDeviceSources, mf, "MFEnumDeviceSources");
    get(x.createReader, rw, "MFCreateSourceReaderFromMediaSource");
    if (!all) {
      x.why = "Media Foundation's DLLs lack a function the receiver calls";
      return x;
    }
    // ai: the MTA kept alive with no thread in it, for the sources and readers held between calls
    CO_MTA_USAGE_COOKIE cookie = nullptr;
    CoIncrementMTAUsage(&cookie);
    x.ok = true;
    return x;
  }();
  return a;
}

// ai: COM on the calling thread, multithreaded; a thread already in a single-threaded apartment (RPC_E_CHANGED_MODE)
// ai: is used as it is (Media Foundation's objects are free-threaded)
struct Com {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  Com() = default;
  Com(const Com&) = delete;
  Com& operator=(const Com&) = delete;
  ~Com() {
    if (SUCCEEDED(hr)) CoUninitialize();
  }
};

// ai: one MFStartup, its MFShutdown when it goes (Media Foundation counts them)
struct Started {
  bool ok = false;
  explicit Started(std::string* why) {
    const Api& a = api();
    if (!a.ok) {
      if (why) *why = a.why;
      return;
    }
    const HRESULT hr = a.startup(MF_VERSION, MFSTARTUP_LITE);
    ok = SUCCEEDED(hr);
    if (!ok && why) *why = "Media Foundation did not start (" + hex(hr) + ")";
  }
  Started(const Started&) = delete;
  Started& operator=(const Started&) = delete;
  ~Started() {
    if (ok) api().shutdown();
  }
};

// ai: the subtypes a mode may have, each the camera's name for it, the layout luma.h reads and its sample bits; MF's
// ai: RGB24 and RGB32 are B, G, R (and X) in memory
struct Sub { const GUID* g; const char* name; Pix pix; uint8_t bits; };
const Sub kSubs[] = {
    {&MFVideoFormat_NV12, "NV12", Pix::NV12, 8},   {&MFVideoFormat_YUY2, "YUY2", Pix::YUYV, 8},
    {&MFVideoFormat_UYVY, "UYVY", Pix::UYVY, 8},   {&MFVideoFormat_L8, "L8", Pix::Grey8, 8},
    {&MFVideoFormat_L16, "L16", Pix::Grey16, 16},  {&MFVideoFormat_I420, "I420", Pix::I420, 8},
    {&MFVideoFormat_IYUV, "IYUV", Pix::I420, 8},   {&MFVideoFormat_YV12, "YV12", Pix::YV12, 8},
    {&MFVideoFormat_RGB24, "RGB24", Pix::BGR24, 8}, {&MFVideoFormat_RGB32, "RGB32", Pix::BGRA32, 8},
    {&MFVideoFormat_MJPG, "MJPG", Pix::MJPEG, 8},
};

const Sub* subOf(const GUID& g) {
  for (const Sub& s : kSubs)
    if (*s.g == g) return &s;
  return nullptr;
}

// ai: a subtype's FOURCC where it is one (MFVideoFormat_Base's tail), else its first word in hex
std::string subName(const GUID& g) {
  if (const Sub* s = subOf(g)) return s->name;
  std::string n;
  for (int i = 0; i < 4; i++) {
    const char c = static_cast<char>((g.Data1 >> (8 * i)) & 0xff);
    if (c < 0x20 || c > 0x7e) {
      char b[16];
      std::snprintf(b, sizeof b, "0x%08lX", static_cast<unsigned long>(g.Data1));
      return b;
    }
    n += c;
  }
  return n;
}

double ratio(IMFAttributes* a, REFGUID key) {
  UINT32 n = 0, d = 0;
  if (FAILED(MFGetAttributeRatio(a, key, &n, &d)) || !d) return 0;
  return double(n) / d;
}

// ai: a native type as a camera.h Mode (key its index); false for a subtype luma.h and the decoder do not take
bool modeOf(IMFMediaType* t, DWORD index, Mode& m) {
  GUID major{}, sub{};
  if (FAILED(t->GetGUID(MF_MT_MAJOR_TYPE, &major)) || major != MFMediaType_Video) return false;
  if (FAILED(t->GetGUID(MF_MT_SUBTYPE, &sub))) return false;
  const Sub* s = subOf(sub);
  if (!s) return false;
  UINT32 w = 0, h = 0;
  if (FAILED(MFGetAttributeSize(t, MF_MT_FRAME_SIZE, &w, &h)) || !w || !h) return false;
  const double fps = ratio(t, MF_MT_FRAME_RATE), lo = ratio(t, MF_MT_FRAME_RATE_RANGE_MIN),
               hi = ratio(t, MF_MT_FRAME_RATE_RANGE_MAX);
  m = Mode{};
  m.width = w;
  m.height = h;
  m.fps = hi > 0 ? std::max(hi, fps) : fps;
  m.fpsMin = lo > 0 && lo < m.fps ? lo : m.fps;
  m.format = s->name;
  m.pix = s->pix;
  m.bits = s->bits;
  m.compressed = s->pix == Pix::MJPEG;
  m.key = index;
  return true;
}

bool sameMode(const Mode& a, const Mode& b) {
  return a.width == b.width && a.height == b.height && a.format == b.format && std::fabs(a.fps - b.fps) < 1e-3 &&
         std::fabs(a.fpsMin - b.fpsMin) < 1e-3;
}

std::string desc(const Mode& m) {
  char b[96];
  std::snprintf(b, sizeof b, "%ux%u %s at %.4g fps", m.width, m.height, m.format.c_str(), m.fps);
  return b;
}

// ai: the stride Lock's bytes have where the type names none: a DIB's rows for RGB (4-byte aligned, top-down as MF
// ai: takes a positive stride), the first plane's for the rest
LONG defaultStride(Pix pix, uint32_t w) {
  switch (pix) {
    case Pix::YUYV: case Pix::UYVY: case Pix::Grey16: return LONG(w * 2);
    case Pix::BGR24: return LONG((w * 3 + 3) & ~3u);
    case Pix::BGRA32: return LONG(w * 4);
    default: return LONG(w);
  }
}

// ai: What the reader hands over: the layout, the buffer's frame, Lock's stride, the valid region (the minimum display
// ai: aperture where the type has one: a decoder's frame can be taller than the picture), the rate.
struct Out {
  Pix pix = Pix::Unknown;
  uint8_t bits = 8;
  std::string name;
  uint32_t w = 0, h = 0;
  LONG stride = 0;
  uint32_t ax = 0, ay = 0, aw = 0, ah = 0;
  double fps = 0;
};

bool outOf(IMFMediaType* t, Out& o, std::string& why) {
  GUID sub{};
  t->GetGUID(MF_MT_SUBTYPE, &sub);
  const Sub* s = subOf(sub);
  if (!s || s->pix == Pix::MJPEG) {
    why = "the reader gave " + subName(sub) + ", which the receiver does not read";
    return false;
  }
  UINT32 w = 0, h = 0;
  if (FAILED(MFGetAttributeSize(t, MF_MT_FRAME_SIZE, &w, &h)) || !w || !h) {
    why = "the reader's format has no frame size";
    return false;
  }
  o = Out{};
  o.pix = s->pix;
  o.bits = s->bits;
  o.name = s->name;
  o.w = w;
  o.h = h;
  UINT32 st = 0;
  o.stride = SUCCEEDED(t->GetUINT32(MF_MT_DEFAULT_STRIDE, &st)) ? LONG(int32_t(st)) : defaultStride(o.pix, w);
  o.fps = ratio(t, MF_MT_FRAME_RATE);
  o.aw = w;
  o.ah = h;
  MFVideoArea a{};
  UINT32 got = 0;
  if (SUCCEEDED(t->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE, reinterpret_cast<UINT8*>(&a), sizeof a, &got)) &&
      got == sizeof a && a.OffsetX.value >= 0 && a.OffsetY.value >= 0 && a.Area.cx > 0 && a.Area.cy > 0 &&
      uint32_t(a.OffsetX.value) + uint32_t(a.Area.cx) <= w && uint32_t(a.OffsetY.value) + uint32_t(a.Area.cy) <= h) {
    o.ax = uint32_t(a.OffsetX.value);
    o.ay = uint32_t(a.OffsetY.value);
    o.aw = uint32_t(a.Area.cx);
    o.ah = uint32_t(a.Area.cy);
  }
  return true;
}

// ai: A buffer locked for reading: IMF2DBuffer2's Lock2DSize (the bytes from the first row on known), IMF2DBuffer's
// ai: Lock2D (a plane's rows at the pitch assumed whole), else Lock at the type's stride (a negative one: the rows
// ai: bottom-up from the buffer's start). scan0 is the top row; avail the bytes readable from it for a positive pitch.
struct Locked {
  ComPtr<IMF2DBuffer2> b2;
  ComPtr<IMF2DBuffer> b1;
  ComPtr<IMFMediaBuffer> b0;
  BYTE* scan0 = nullptr;
  LONG pitch = 0;
  size_t avail = 0;
  bool ok = false;
  Locked(IMFMediaBuffer* b, LONG stride, uint32_t h) {
    if (SUCCEEDED(b->QueryInterface(IID_PPV_ARGS(&b2)))) {
      BYTE* start = nullptr;
      DWORD len = 0;
      if (SUCCEEDED(b2->Lock2DSize(MF2DBuffer_LockFlags_Read, &scan0, &pitch, &start, &len))) {
        ok = true;
        avail = scan0 >= start && scan0 <= start + len ? size_t(start + len - scan0) : 0;
        if (pitch < 0) avail = size_t(-pitch) * h;
        return;
      }
      b2.Reset();
    }
    if (SUCCEEDED(b->QueryInterface(IID_PPV_ARGS(&b1)))) {
      if (SUCCEEDED(b1->Lock2D(&scan0, &pitch))) {
        ok = true;
        avail = size_t(pitch < 0 ? -pitch : pitch) * h;
        return;
      }
      b1.Reset();
    }
    BYTE* p = nullptr;
    DWORD max = 0, cur = 0;
    if (SUCCEEDED(b->Lock(&p, &max, &cur))) {
      b0 = b;
      ok = true;
      pitch = stride;
      avail = cur;
      if (pitch < 0) {
        const size_t row = size_t(-pitch);
        if (h && row * h <= cur) scan0 = p + row * (h - 1);
        else ok = false;   // ai: shorter than its rows: unlocked here, counted corrupt
        if (!ok) b0->Unlock();
      } else {
        scan0 = p;
      }
    }
  }
  Locked(const Locked&) = delete;
  Locked& operator=(const Locked&) = delete;
  ~Locked() {
    if (!ok) return;
    if (b2) b2->Unlock2D();
    else if (b1) b1->Unlock2D();
    else if (b0) b0->Unlock();
  }
};

// ai: The controls, ids as camera.h's; IAMCameraControl's or IAMVideoProcAmp's property. Their flags: Auto 1, Manual 2.
struct Ctl { const char* id; const char* name; const char* unit; bool cam; long prop; };
const Ctl kCtls[] = {
    {"zoom", "Zoom", "", true, CameraControl_Zoom},
    {"focus", "Focus", "", true, CameraControl_Focus},
    {"exposure", "Exposure", "us", true, CameraControl_Exposure},
    {"gain", "Gain", "", false, VideoProcAmp_Gain},
    {"sharpness", "Sharpness", "", false, VideoProcAmp_Sharpness},
    {"backlight", "Backlight compensation", "", false, VideoProcAmp_BacklightCompensation},
    {"brightness", "Brightness", "", false, VideoProcAmp_Brightness},
    {"contrast", "Contrast", "", false, VideoProcAmp_Contrast},
};
constexpr long kAuto = CameraControl_Flags_Auto, kManual = CameraControl_Flags_Manual;
static_assert(long(CameraControl_Flags_Auto) == long(VideoProcAmp_Flags_Auto) &&
              long(CameraControl_Flags_Manual) == long(VideoProcAmp_Flags_Manual), "the two interfaces' flags differ");

const Ctl* ctlOf(const std::string& id) {
  for (const Ctl& c : kCtls)
    if (id == c.id) return &c;
  return nullptr;
}

bool isExposure(const Ctl& c) { return c.cam && c.prop == CameraControl_Exposure; }
// ai: CameraControl_Exposure's log2 seconds as us
double expUs(long v) { return std::ldexp(1e6, int(v)); }

struct Raw { long min = 0, max = 0, step = 0, def = 0, caps = 0, value = 0, flags = 0; };

// ai: how long start waits for the stream's first answer (a sample, or why none comes) before it returns anyway
constexpr int kFirstMs = 5000;
// ai: how long a stop waits for the reader's flush (a camera gone quiet may never answer it)
constexpr int kFlushMs = 1000;

// ai: The reader's answers in its asynchronous mode, from Media Foundation's work queue to the capture thread: one
// ai: at a time (one ReadSample out at a time), the flush's end, and a stop's wake. Asynchronous because a
// ai: synchronous ReadSample that a camera gone quiet never answers holds its thread for good: neither the reader's
// ai: Flush (it waits for that read) nor the source's Shutdown ends it (the fake source stalled, 2026-10-25), so stop
// ai: could not return. The callback holds the box, never the camera: an answer after the stream went is dropped.
struct Box {
  std::mutex mu;
  std::condition_variable cv;
  bool has = false, flushed = false, wake = false, closed = false;
  HRESULT hr = S_OK;
  DWORD flags = 0;
  LONGLONG t = 0;
  int64_t at = 0;   // ai: steadyNs when the answer came
  ComPtr<IMFSample> sample;
};

class Answers final : public IMFSourceReaderCallback {
 public:
  explicit Answers(std::shared_ptr<Box> box) : box_(std::move(box)) {}
  STDMETHODIMP QueryInterface(REFIID iid, void** out) override {
    if (!out) return E_POINTER;
    if (iid == IID_IUnknown || iid == IID_IMFSourceReaderCallback) {
      *out = static_cast<IMFSourceReaderCallback*>(this);
      AddRef();
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
  STDMETHODIMP_(ULONG) Release() override {
    const ULONG n = --ref_;
    if (!n) delete this;
    return n;
  }
  STDMETHODIMP OnReadSample(HRESULT hr, DWORD, DWORD flags, LONGLONG t, IMFSample* s) override {
    const int64_t at = steadyNs();
    {
      std::lock_guard<std::mutex> lk(box_->mu);
      if (box_->closed) return S_OK;
      box_->has = true;
      box_->hr = hr;
      box_->flags = flags;
      box_->t = t;
      box_->at = at;
      box_->sample = s;
    }
    box_->cv.notify_all();
    return S_OK;
  }
  STDMETHODIMP OnFlush(DWORD) override {
    {
      std::lock_guard<std::mutex> lk(box_->mu);
      box_->flushed = true;
    }
    box_->cv.notify_all();
    return S_OK;
  }
  STDMETHODIMP OnEvent(DWORD, IMFMediaEvent*) override { return S_OK; }

 private:
  std::atomic<ULONG> ref_{1};
  std::shared_ptr<Box> box_;
};

class MfCamera final : public Camera {
 public:
  // ai: make: a new source for the device (the first one given), drop: what lets the device go at the end
  using Make = std::function<HRESULT(ComPtr<IMFMediaSource>&, std::string&)>;

  // ai: the camera over its first source; null with why (the source shut down, the device let go) where Media
  // ai: Foundation did not start or the source has no video stream
  static std::unique_ptr<Camera> create(const Info& info, Make make, std::function<void()> drop,
                                        ComPtr<IMFMediaSource> first, std::string& why) {
    std::unique_ptr<MfCamera> c(new MfCamera(info, std::move(make), std::move(drop)));
    c->adopt(first);
    if (!c->mf_.ok) {
      why = "Media Foundation did not start";
      return nullptr;
    }
    ComPtr<IMFMediaTypeHandler> h;
    if (!c->handler(h, why)) return nullptr;
    c->modes_ = readModes(h.Get());
    c->ctls_ = c->readControls();
    return c;
  }

  ~MfCamera() override {
    Com com;
    std::lock_guard<std::mutex> lk(mu_);
    halt();
    dropSource();
    if (drop_) drop_();
  }

  const Info& info() const override { return info_; }

  // ai: read at open: a device's types do not change, and a look must not wake a stopped device
  std::vector<Mode> modes() override {
    std::lock_guard<std::mutex> lk(mu_);
    return modes_;
  }

  // ai: read off the source where one is alive (opened, streaming, or activated by a set); a stopped camera's as they
  // ai: stood when its last source went
  std::vector<Control> controls() override {
    Com com;
    std::lock_guard<std::mutex> lk(mu_);
    if (!src_) return ctls_;
    std::vector<Control> out = readControls();
    if (!out.empty()) ctls_ = out;
    return out;
  }

  bool set(const std::string& id, double value, bool automatic, std::string& why) override {
    Com com;
    std::lock_guard<std::mutex> lk(mu_);
    why.clear();
    const Ctl* c = ctlOf(id);
    if (!c) {
      why = "no control " + id;
      return false;
    }
    if (!ensureSource(why)) return false;
    Raw r;
    if (!rawGet(*c, r)) {
      why = std::string("the camera has no ") + c->name + " control";
      return false;
    }
    if (automatic && !(r.caps & kAuto)) {
      why = std::string("the camera's ") + c->name + " has no automatic setting";
      return false;
    }
    long v = r.value;
    if (!automatic) {
      if (isExposure(*c)) {
        v = value > 0 ? long(std::lround(std::log2(value / 1e6))) : r.min;
      } else {
        v = long(std::lround(value));
        if (r.step > 1) v = r.min + long(std::lround(double(v - r.min) / r.step)) * r.step;
      }
      v = std::clamp(v, r.min, r.max);
    }
    save(*c, r);
    const HRESULT hr = rawSet(*c, v, automatic ? kAuto : kManual);
    if (FAILED(hr)) {
      why = std::string("the camera refused its ") + c->name + " (" + hex(hr) + ")";
      return false;
    }
    return true;
  }

  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override {
    Com com;
    std::lock_guard<std::mutex> lk(mu_);
    why.clear();
    if (th_.joinable()) {
      if (!done_.load()) {
        why = "the camera is already streaming";
        return false;
      }
      halt();   // ai: a stream that ended by itself, put away
    }
    ComPtr<IMFMediaTypeHandler> h;
    if (!ensureSource(why)) return false;
    if (!handler(h, why)) {
      letGo();
      return false;
    }
    ComPtr<IMFMediaType> listed;
    Mode m;
    if (!typeFor(h.Get(), mode, listed, m, why)) return false;
    // ai: a rate inside the type's range but not its own asked as its MF_MT_FRAME_RATE (Media Foundation's way to
    // ai: set a capture rate); refused, the type as listed at its own rate
    ComPtr<IMFMediaType> nt = listed;
    const double own = ratio(listed.Get(), MF_MT_FRAME_RATE);
    m.fps = own > 0 ? own : m.fps;
    HRESULT hr = E_FAIL;
    if (mode.fps > 0 && m.fpsMin < mode.fps && std::fabs(own - mode.fps) > 1e-3 * mode.fps) {
      ComPtr<IMFMediaType> c;
      if (SUCCEEDED(api().createMediaType(&c)) && SUCCEEDED(listed->CopyAllItems(c.Get()))) {
        const UINT32 num = UINT32(std::lround(mode.fps * 1000)), g = UINT32(std::gcd(num, 1000u));
        MFSetAttributeRatio(c.Get(), MF_MT_FRAME_RATE, num / g, 1000 / g);
        hr = h->SetCurrentMediaType(c.Get());
        if (SUCCEEDED(hr)) {
          nt = c;
          m.fps = mode.fps;
        }
      }
    }
    if (FAILED(hr)) hr = h->SetCurrentMediaType(nt.Get());
    if (FAILED(hr)) {
      why = hr == E_ACCESSDENIED ? kBlocked : "the camera refused " + desc(m) + " (" + hrWhy(hr) + ")";
      letGo();
      return false;
    }
    rules();
    if (!openReader(nt.Get(), m, why)) {
      halt();
      letGo();   // ai: where no reader was made to shut it down
      return false;
    }
    {
      std::lock_guard<std::mutex> s(statsMu_);
      st_ = Stats{};
      st_.fps = m.fps;   // ai: the rate of the type the device's handler took (asked or its own)
      convSum_ = 0;
      convN_ = 0;
    }
    {
      std::lock_guard<std::mutex> a(amu_);
      answered_ = false;
      gaveUp_ = false;
      firstHr_ = S_OK;
    }
    stop_ = false;
    done_ = false;
    sink_ = std::move(sink);
    ended_ = std::move(ended);
    th_ = std::thread([this] { loop(); });
    std::unique_lock<std::mutex> a(amu_);
    acv_.wait_for(a, std::chrono::milliseconds(kFirstMs), [&] { return answered_; });
    gaveUp_ = true;
    if (!answered_ || SUCCEEDED(firstHr_)) return true;   // ai: frames, or an error after this, come to the sink or ended
    const HRESULT first = firstHr_;
    a.unlock();
    halt();
    why = first == MF_E_END_OF_STREAM ? "the camera's stream ended at once" : hrWhy(first);
    return false;
  }

  void stop() override {
    // ai: from the sink or ended, on the capture thread: the thread ends once the call returns; the next stop, start
    // ai: or the destructor puts the stream away
    if (std::this_thread::get_id() == capture_.load()) {
      stop_ = true;
      return;
    }
    Com com;
    std::lock_guard<std::mutex> lk(mu_);
    halt();
  }

  Stats stats() override {
    std::lock_guard<std::mutex> s(statsMu_);
    Stats r = st_;
    r.convertMs = convN_ ? convSum_ / double(convN_) : 0;
    return r;
  }

 private:
  MfCamera(const Info& info, Make make, std::function<void()> drop)
      : mf_(nullptr), info_(info), make_(std::move(make)), drop_(std::move(drop)) {}

  void adopt(ComPtr<IMFMediaSource> s) {
    src_ = s;
    cc_.Reset();
    pa_.Reset();
    ks_.Reset();
    if (!src_) return;
    src_.As(&cc_);
    src_.As(&pa_);
    src_->QueryInterface(IID_IKsControl, reinterpret_cast<void**>(ks_.ReleaseAndGetAddressOf()));
  }

  // ai: the controls kCtls names that the source answers, ranges and values (mu_ held, the source alive)
  std::vector<Control> readControls() {
    std::vector<Control> out;
    for (const Ctl& c : kCtls) {
      Raw r;
      if (!rawGet(c, r)) continue;
      Control k;
      k.id = c.id;
      k.name = c.name;
      k.unit = c.unit;
      if (isExposure(c)) {
        k.min = expUs(r.min);
        k.max = expUs(r.max);
        k.def = expUs(r.def);
        k.value = expUs(r.value);
        k.step = 0;   // ai: powers of two; set() takes the nearest
      } else {
        k.min = r.min;
        k.max = r.max;
        k.step = r.step;
        k.def = r.def;
        k.value = r.value;
      }
      k.canAuto = (r.caps & kAuto) != 0;
      k.isAuto = (r.flags & kAuto) != 0;
      out.push_back(k);
    }
    return out;
  }

  // ai: the first video stream's native types as modes, each once
  static std::vector<Mode> readModes(IMFMediaTypeHandler* h) {
    std::vector<Mode> out;
    DWORD n = 0;
    h->GetMediaTypeCount(&n);
    for (DWORD i = 0; i < n; i++) {
      ComPtr<IMFMediaType> t;
      Mode m;
      if (FAILED(h->GetMediaTypeByIndex(i, &t)) || !modeOf(t.Get(), i, m)) continue;
      if (std::none_of(out.begin(), out.end(), [&](const Mode& o) { return sameMode(o, m); })) out.push_back(m);
    }
    return out;
  }

  // ai: the device's source, activated again where the last stream's reader shut it down
  bool ensureSource(std::string& why) {
    if (src_) return true;
    ComPtr<IMFMediaSource> s;
    std::string w;
    const HRESULT hr = make_ ? make_(s, w) : E_FAIL;
    if (FAILED(hr) || !s) {
      why = !w.empty() ? w : hr == E_ACCESSDENIED ? std::string(kBlocked) : "the camera would not open (" + hrWhy(hr) + ")";
      return false;
    }
    adopt(s);
    return true;
  }

  // ai: the source let go after a failure on it, its controls read first where it still answers them (the cache
  // ai: kept where it answers none: a device gone), so the next stream or set activates the device anew
  void letGo() {
    if (src_) {
      std::vector<Control> c = readControls();
      if (!c.empty()) ctls_ = std::move(c);
    }
    dropSource();
  }

  // ai: the source shut down (again, where its reader did) and let go
  void dropSource() {
    if (src_) src_->Shutdown();
    src_.Reset();
    cc_.Reset();
    pa_.Reset();
    ks_.Reset();
  }

  // ai: the first video stream's type handler, from the source's presentation descriptor
  bool handler(ComPtr<IMFMediaTypeHandler>& h, std::string& why) {
    ComPtr<IMFPresentationDescriptor> pd;
    const HRESULT hr = src_ ? src_->CreatePresentationDescriptor(&pd) : E_POINTER;
    if (FAILED(hr)) {
      why = "the camera describes no stream (" + hrWhy(hr) + ")";
      return false;
    }
    DWORD n = 0;
    pd->GetStreamDescriptorCount(&n);
    for (DWORD i = 0; i < n; i++) {
      BOOL selected = FALSE;
      ComPtr<IMFStreamDescriptor> sd;
      ComPtr<IMFMediaTypeHandler> th;
      GUID major{};
      if (FAILED(pd->GetStreamDescriptorByIndex(i, &selected, &sd)) || FAILED(sd->GetMediaTypeHandler(&th)) ||
          FAILED(th->GetMajorType(&major)) || major != MFMediaType_Video)
        continue;
      h = th;
      return true;
    }
    why = "the camera has no video stream";
    return false;
  }

  // ai: the native type for mode: its key's, else the first of its format and size whose rates hold mode.fps
  bool typeFor(IMFMediaTypeHandler* h, const Mode& mode, ComPtr<IMFMediaType>& t, Mode& m, std::string& why) {
    auto fits = [&](const Mode& c) {
      return c.width == mode.width && c.height == mode.height && c.format == mode.format &&
             (mode.fps <= 0 || (mode.fps <= c.fps * 1.001 && mode.fps >= c.fpsMin * 0.999));
    };
    if (SUCCEEDED(h->GetMediaTypeByIndex(DWORD(mode.key), &t)) && modeOf(t.Get(), DWORD(mode.key), m) && fits(m))
      return true;
    t.Reset();
    DWORD n = 0;
    h->GetMediaTypeCount(&n);
    for (DWORD i = 0; i < n; i++) {
      ComPtr<IMFMediaType> c;
      if (SUCCEEDED(h->GetMediaTypeByIndex(i, &c)) && modeOf(c.Get(), i, m) && fits(m)) {
        t = c;
        return true;
      }
    }
    why = "the camera has no mode " + desc(mode);
    return false;
  }

  // ai: The reader over the source: converters allowed (the MJPEG decoder), no video processing (the frame as the
  // ai: camera or the decoder gives it); its output the native type, or a decoded MJPG's NV12, YUY2 or I420 at the
  // ai: native type's size and rate. The reader meets an output type with a native type of its own where one holds
  // ai: every attribute asked, switching the device to it: asked by subtype alone, it took a camera's NV12 640x480
  // ai: over decoding its MJPG 1920x1080 (the fake source, 2026-10-25), so the size and the rate are asked too, which
  // ai: only a decoder, or an uncompressed type of the same size and rate (the same picture), meets.
  bool openReader(IMFMediaType* native, const Mode& m, std::string& why) {
    const Api& a = api();
    box_ = std::make_shared<Box>();
    ComPtr<IMFSourceReaderCallback> cb;
    cb.Attach(new Answers(box_));
    ComPtr<IMFAttributes> at;
    HRESULT hr = a.createAttributes(&at, 1);
    if (SUCCEEDED(hr)) hr = at->SetUnknown(MF_SOURCE_READER_ASYNC_CALLBACK, cb.Get());
    if (SUCCEEDED(hr)) hr = a.createReader(src_.Get(), at.Get(), &reader_);
    if (FAILED(hr)) {
      why = "Media Foundation could not read the camera (" + hrWhy(hr) + ")";
      return false;
    }
    reader_->SetStreamSelection(DWORD(MF_SOURCE_READER_ALL_STREAMS), FALSE);
    hr = reader_->SetStreamSelection(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), TRUE);
    if (SUCCEEDED(hr)) {
      if (!m.compressed) {
        hr = reader_->SetCurrentMediaType(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), nullptr, native);
      } else {
        UINT32 w = 0, h = 0, num = 0, den = 0;
        MFGetAttributeSize(native, MF_MT_FRAME_SIZE, &w, &h);
        const bool rated = SUCCEEDED(MFGetAttributeRatio(native, MF_MT_FRAME_RATE, &num, &den)) && num && den;
        for (const GUID* g : {&MFVideoFormat_NV12, &MFVideoFormat_YUY2, &MFVideoFormat_I420}) {
          ComPtr<IMFMediaType> t;
          hr = a.createMediaType(&t);
          if (FAILED(hr)) break;
          t->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
          t->SetGUID(MF_MT_SUBTYPE, *g);
          MFSetAttributeSize(t.Get(), MF_MT_FRAME_SIZE, w, h);
          if (rated) MFSetAttributeRatio(t.Get(), MF_MT_FRAME_RATE, num, den);
          hr = reader_->SetCurrentMediaType(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), nullptr, t.Get());
          if (SUCCEEDED(hr)) break;
        }
      }
    }
    if (FAILED(hr)) {
      why = m.compressed ? "no Media Foundation decoder takes the camera's MJPEG (" + hrWhy(hr) + ")"
                         : "the reader refused " + desc(m) + " (" + hrWhy(hr) + ")";
      return false;
    }
    ComPtr<IMFMediaType> cur;
    hr = reader_->GetCurrentMediaType(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), &cur);
    if (FAILED(hr)) {
      why = "the reader has no format (" + hrWhy(hr) + ")";
      return false;
    }
    if (!outOf(cur.Get(), out_, why)) return false;
    if (out_.fps <= 0) out_.fps = m.fps;
    return true;
  }

  bool rawGet(const Ctl& c, Raw& r) {
    if (c.cam) {
      return cc_ && SUCCEEDED(cc_->GetRange(c.prop, &r.min, &r.max, &r.step, &r.def, &r.caps)) &&
             SUCCEEDED(cc_->Get(c.prop, &r.value, &r.flags));
    }
    return pa_ && SUCCEEDED(pa_->GetRange(c.prop, &r.min, &r.max, &r.step, &r.def, &r.caps)) &&
           SUCCEEDED(pa_->Get(c.prop, &r.value, &r.flags));
  }

  HRESULT rawSet(const Ctl& c, long v, long flags) {
    if (c.cam) return cc_ ? cc_->Set(c.prop, v, flags) : E_NOINTERFACE;
    return pa_ ? pa_->Set(c.prop, v, flags) : E_NOINTERFACE;
  }

  // ai: a control's value and flags before the backend's first change, to put back
  void save(const Ctl& c, const Raw& r) {
    if (!saved_.count(c.id)) saved_[c.id] = {r.value, r.flags};
  }

  // ai: auto-exposure priority through IKsControl: read (set false) or written
  bool aep(LONG& value, ULONG& flags, bool write) {
    if (!ks_) return false;
    KSPROPERTY_CAMERACONTROL_S s{};
    s.Property.Set = kCameraControlSet;
    s.Property.Id = kAutoExposurePriority;
    s.Property.Flags = write ? KSPROPERTY_TYPE_SET : KSPROPERTY_TYPE_GET;
    s.Value = value;
    s.Flags = flags;
    ULONG got = 0;
    if (FAILED(ks_->KsProperty(&s.Property, sizeof s, &s, sizeof s, &got))) return false;
    if (!write) {
      value = s.Value;
      flags = s.Flags;
    }
    return true;
  }

  // ai: The camera rules (camera.cpp): sharpness at its minimum and backlight compensation off, the frame rate held
  // ai: (auto-exposure priority 0); exposure, gain and focus left as they are.
  void rules() {
    Raw r;
    if (const Ctl* c = ctlOf("sharpness"); c && rawGet(*c, r) && (r.value != r.min || !(r.flags & kManual))) {
      save(*c, r);
      rawSet(*c, r.min, kManual);
    }
    if (const Ctl* c = ctlOf("backlight"); c && rawGet(*c, r)) {
      const long off = r.min <= 0 && 0 <= r.max ? 0 : r.min;
      if (r.value != off || !(r.flags & kManual)) {
        save(*c, r);
        rawSet(*c, off, kManual);
      }
    }
    LONG v = 0;
    ULONG f = 0;
    if (aep(v, f, false) && v != 0) {
      if (!aepSaved_) {
        aepSaved_ = true;
        aepValue_ = v;
        aepFlags_ = f;
      }
      LONG zero = 0;
      ULONG manual = KSPROPERTY_CAMERACONTROL_FLAGS_MANUAL;
      aep(zero, manual, true);
    }
  }

  // ai: every control the backend changed back to what it was (the source alive; one activated where it is not)
  void restore() {
    if (saved_.empty() && !aepSaved_) return;
    std::string why;
    if (!ensureSource(why)) return;
    for (const auto& [id, v] : saved_)
      if (const Ctl* c = ctlOf(id)) rawSet(*c, v.first, v.second);
    saved_.clear();
    if (aepSaved_) {
      LONG v = aepValue_;
      ULONG f = aepFlags_;
      aep(v, f, true);
      aepSaved_ = false;
    }
  }

  // ai: The stream ended and put away (mu_ held): the capture thread woken (it waits on the box, never inside Media
  // ai: Foundation) and joined; the controls put back; the read still out cancelled by a flush (a second at most:
  // ai: a camera gone quiet may answer neither), the box closed to late answers, the reader released, which shuts
  // ai: its source down.
  void halt() {
    if (th_.joinable()) {
      stop_ = true;
      if (box_) {
        {
          std::lock_guard<std::mutex> b(box_->mu);
          box_->wake = true;
        }
        box_->cv.notify_all();
      }
      th_.join();
      capture_ = std::thread::id();
    }
    restore();
    // ai: the controls as they stand put back, read before the reader's release shuts the source down: what a look at
    // ai: the stopped camera answers
    if (src_) {
      std::vector<Control> c = readControls();
      if (!c.empty()) ctls_ = std::move(c);
    }
    if (reader_) {
      if (box_) {
        {
          std::lock_guard<std::mutex> b(box_->mu);
          box_->flushed = false;
        }
        if (SUCCEEDED(reader_->Flush(DWORD(MF_SOURCE_READER_ALL_STREAMS)))) {
          std::unique_lock<std::mutex> b(box_->mu);
          box_->cv.wait_for(b, std::chrono::milliseconds(kFlushMs), [&] { return box_->flushed; });
        }
        std::lock_guard<std::mutex> b(box_->mu);
        box_->closed = true;
        box_->has = false;
        box_->sample.Reset();
      }
      reader_.Reset();
      dropSource();
    }
    box_.reset();
    sink_ = nullptr;
    ended_ = nullptr;
  }

  // ai: the first ReadSample's result for start: true where start takes a failure (it still waits)
  bool answer(HRESULT hr) {
    bool mine;
    {
      std::lock_guard<std::mutex> a(amu_);
      answered_ = true;
      firstHr_ = hr;
      mine = !gaveUp_;
    }
    acv_.notify_all();
    return FAILED(hr) && mine;
  }

  // ai: The capture thread: a read asked of the reader, its answer waited for on the box; the next read asked before
  // ai: this answer's frame is read (the reader fetches it meanwhile); the frame's time, its gaps, its luma, the sink.
  void loop() {
    capture_ = std::this_thread::get_id();
    Com com;
    const std::shared_ptr<Box> box = box_;
    Out o = out_;
    std::vector<uint8_t> conv, flip;
    int64_t off = 0, prev = 0;
    uint64_t seq = 0;
    bool based = false, first = true;
    HRESULT asked = reader_->ReadSample(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), 0, nullptr, nullptr, nullptr, nullptr);
    for (;;) {
      HRESULT hr = asked;
      DWORD fl = 0;
      LONGLONG t = 0;
      int64_t arrived = 0;
      ComPtr<IMFSample> s;
      if (SUCCEEDED(asked)) {
        std::unique_lock<std::mutex> b(box->mu);
        box->cv.wait(b, [&] { return box->has || box->wake; });
        if (!box->has) break;   // ai: a stop
        box->has = false;
        hr = box->hr;
        fl = box->flags;
        t = box->t;
        arrived = box->at;
        s = std::move(box->sample);
      }
      const HRESULT bad = FAILED(hr) ? hr : (fl & MF_SOURCE_READERF_ERROR) ? E_FAIL
                          : (fl & MF_SOURCE_READERF_ENDOFSTREAM) ? MF_E_END_OF_STREAM : S_OK;
      if (first) {
        first = false;
        if (answer(bad)) break;
      }
      if (stop_.load()) break;
      if (FAILED(bad)) {
        const std::string why = bad == MF_E_END_OF_STREAM ? "the camera's stream ended"
                                : bad == E_FAIL ? "the camera's stream failed" : hrWhy(bad);
        if (ended_) ended_(why);
        break;
      }
      asked = reader_->ReadSample(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), 0, nullptr, nullptr, nullptr, nullptr);
      if (fl & (MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED | MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)) {
        ComPtr<IMFMediaType> cur;
        std::string why;
        if (FAILED(reader_->GetCurrentMediaType(DWORD(MF_SOURCE_READER_FIRST_VIDEO_STREAM), &cur)) ||
            !outOf(cur.Get(), o, why)) {
          if (ended_) ended_(why.empty() ? "the camera's format changed" : why);
          break;
        }
        if (o.fps <= 0) o.fps = out_.fps;
      }
      if (!s) continue;   // ai: a stream tick (a gap): the next sample's time counts what it lost
      // ai: its time on steadyNs's clock, monotonic: a sample clock that went back is based again on its arrival
      const int64_t tns = int64_t(t) * 100;
      if (!based) {
        off = arrived - tns;
        based = true;
      }
      int64_t ns = tns + off;
      uint64_t lost = 0;
      if (seq) {
        if (ns <= prev) {
          off = arrived - tns;
          ns = std::max(tns + off, prev + 1);
        } else if (o.fps > 0) {
          const double gap = double(ns - prev) * o.fps / 1e9;
          if (gap >= 1.5) lost = uint64_t(std::llround(gap)) - 1;
        }
      }
      seq += lost;
      prev = ns;
      const uint64_t mine = seq++;
      UINT32 broken = 0;
      bool corrupt = SUCCEEDED(s->GetUINT32(MFSampleExtension_FrameCorruption, &broken)) && broken;
      const auto c0 = std::chrono::steady_clock::now();
      Frame f;
      ComPtr<IMFMediaBuffer> b;
      DWORD nb = 0;
      s->GetBufferCount(&nb);
      if (!corrupt && FAILED(nb == 1 ? s->GetBufferByIndex(0, &b) : s->ConvertToContiguousBuffer(&b))) corrupt = true;
      std::optional<Locked> lock;
      if (!corrupt) {
        lock.emplace(b.Get(), o.stride, o.h);
        corrupt = !lock->ok || !lock->scan0;
      }
      if (!corrupt) {
        const uint8_t* src = lock->scan0;
        LONG pitch = lock->pitch;
        size_t avail = lock->avail;
        if (pitch < 0) {
          // ai: bottom-up: the rows top-down into a buffer of their own, read upward from the top row
          const size_t row = size_t(-pitch);
          flip.resize(row * o.h);
          for (uint32_t y = 0; y < o.h; y++) std::memcpy(flip.data() + row * y, src - ptrdiff_t(row) * y, row);
          src = flip.data();
          pitch = LONG(row);
          avail = flip.size();
        }
        const uint32_t stride = uint32_t(pitch);
        if (lumaInPlace(o.pix)) {
          // ai: the luma plane alone is read (frameBytes: its rows, not the chroma after them)
          corrupt = stride < o.w || avail < frameBytes(o.pix, o.w, o.h, stride);
          f.luma = src;
          f.stride = stride;
        } else if (toLuma(o.pix, o.bits, src, avail, o.w, o.h, stride, conv, nullptr)) {
          f.luma = conv.data();
          f.stride = o.w;
        } else {
          corrupt = true;
        }
      }
      if (corrupt) {
        std::lock_guard<std::mutex> sl(statsMu_);
        st_.corrupt++;
        st_.dropped += lost;
        continue;
      }
      f.luma += size_t(o.ay) * f.stride + o.ax;
      f.width = o.aw;
      f.height = o.ah;
      f.ns = ns;
      f.seq = mine;
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
      if (sink_) sink_(f);
      std::lock_guard<std::mutex> sl(statsMu_);
      st_.frames++;
      st_.dropped += lost;
      convSum_ += ms;
      convN_++;
    }
    done_ = true;
  }

  Started mf_;
  Info info_;
  Make make_;
  std::function<void()> drop_;
  std::mutex mu_;   // ai: the driving thread's calls, one at a time
  ComPtr<IMFMediaSource> src_;
  ComPtr<IAMCameraControl> cc_;
  ComPtr<IAMVideoProcAmp> pa_;
  ComPtr<IKsControl> ks_;
  ComPtr<IMFSourceReader> reader_;
  std::shared_ptr<Box> box_;   // ai: the stream's, its reader's callback's too
  Out out_;
  std::vector<Mode> modes_;      // ai: the first source's types, read at open
  std::vector<Control> ctls_;    // ai: the controls as last read off a source (mu_)
  std::map<std::string, std::pair<long, long>> saved_;
  bool aepSaved_ = false;
  LONG aepValue_ = 0;
  ULONG aepFlags_ = 0;
  // ai: the stream: its thread, stop asked, its end; the first answer start waits for
  std::thread th_;
  std::atomic<std::thread::id> capture_{};
  std::atomic<bool> stop_{false}, done_{false};
  std::mutex amu_;
  std::condition_variable acv_;
  bool answered_ = false, gaveUp_ = false;
  HRESULT firstHr_ = S_OK;
  Sink sink_;
  Ended ended_;
  std::mutex statsMu_;
  Stats st_;
  double convSum_ = 0;
  uint64_t convN_ = 0;
};

// ai: Media Foundation's video capture devices, each its activate, its symbolic link and its Info
struct Device {
  ComPtr<IMFActivate> act;
  std::string link;
  Info info;
};

// ai: "USB 046d:0843" from a symbolic link's vid_ and pid_, else ""
std::string usbOf(const std::string& link) {
  std::string l(link);
  std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  const size_t v = l.find("vid_"), p = l.find("pid_");
  if (v == std::string::npos || p == std::string::npos || v + 8 > l.size() || p + 8 > l.size()) return {};
  return "USB " + l.substr(v + 4, 4) + ":" + l.substr(p + 4, 4);
}

bool sameLink(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++)
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
  return true;
}

std::vector<Device> devices(std::string& why) {
  std::vector<Device> out;
  const Api& a = api();
  ComPtr<IMFAttributes> at;
  HRESULT hr = a.createAttributes(&at, 1);
  if (SUCCEEDED(hr)) hr = at->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
  IMFActivate** acts = nullptr;
  UINT32 n = 0;
  if (SUCCEEDED(hr)) hr = a.enumDeviceSources(at.Get(), &acts, &n);
  if (FAILED(hr)) {
    why = "Media Foundation could not list the cameras (" + hrWhy(hr) + ")";
    return out;
  }
  for (UINT32 i = 0; i < n; i++) {
    Device d;
    d.act.Attach(acts[i]);
    d.link = attrString(d.act.Get(), MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
    if (d.link.empty()) continue;
    d.info.id = "mf:" + d.link;
    d.info.name = attrString(d.act.Get(), MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
    if (d.info.name.empty()) d.info.name = "Camera";
    d.info.backend = "uvc";
    d.info.detail = usbOf(d.link);
    out.push_back(std::move(d));
  }
  CoTaskMemFree(acts);
  return out;
}

struct Test { std::string label; MakeSource make; };
std::mutex gTestMu;
std::map<std::string, Test>& tests() {
  static std::map<std::string, Test> m;
  return m;
}

Info testInfo(const std::string& name, const Test& t) {
  Info i;
  i.id = "mf:test:" + name;
  i.name = t.label;
  i.backend = "uvc";
  i.detail = "test source";
  return i;
}

std::unique_ptr<Camera> openMade(const Info& info, MakeSource make, std::string& why) {
  Com com;
  std::string w;
  IMFMediaSource* raw = make ? make(w) : nullptr;
  if (!raw) {
    why = w.empty() ? "the source was not made" : w;
    return nullptr;
  }
  ComPtr<IMFMediaSource> first;
  first.Attach(raw);
  auto again = [make](ComPtr<IMFMediaSource>& s, std::string& w2) -> HRESULT {
    IMFMediaSource* p = make(w2);
    if (!p) return E_FAIL;
    s.Attach(p);
    return S_OK;
  };
  return MfCamera::create(info, again, nullptr, first, why);
}

}  // namespace

bool mfReady(std::string* why) {
  Com com;
  Started s(why);
  return s.ok;
}

std::vector<Info> mfList(std::vector<std::string>* notes) {
  std::vector<Info> out;
  Com com;
  std::string why;
  Started mf(&why);
  if (!mf.ok) {
    if (notes) notes->push_back("mf: " + why);
    return out;
  }
  for (auto& d : devices(why)) out.push_back(d.info);
  if (!why.empty() && notes) notes->push_back("mf: " + why);
  std::lock_guard<std::mutex> lk(gTestMu);
  for (const auto& [name, t] : tests()) out.push_back(testInfo(name, t));
  return out;
}

std::unique_ptr<Camera> mfOpen(const std::string& id, std::string& why) {
  why.clear();
  if (id.compare(0, 3, "mf:") != 0) {
    why = "no camera " + id;
    return nullptr;
  }
  const std::string link = id.substr(3);
  if (link.compare(0, 5, "test:") == 0) {
    Test t;
    {
      std::lock_guard<std::mutex> lk(gTestMu);
      auto it = tests().find(link.substr(5));
      if (it == tests().end()) {
        why = "no camera " + id;
        return nullptr;
      }
      t = it->second;
    }
    return openMade(testInfo(link.substr(5), t), t.make, why);
  }
  Com com;
  Started mf(&why);
  if (!mf.ok) return nullptr;
  Device dev;
  for (auto& d : devices(why))
    if (sameLink(d.link, link)) {
      dev = std::move(d);
      break;
    }
  if (!dev.act) {
    if (why.empty()) why = "the camera is not connected";
    return nullptr;
  }
  ComPtr<IMFMediaSource> src;
  const HRESULT hr = dev.act->ActivateObject(IID_PPV_ARGS(&src));
  if (FAILED(hr)) {
    why = hr == E_ACCESSDENIED ? std::string(kBlocked) : "the camera would not open (" + hrWhy(hr) + ")";
    dev.act->ShutdownObject();
    return nullptr;
  }
  ComPtr<IMFActivate> act = dev.act;
  auto again = [act](ComPtr<IMFMediaSource>& s, std::string&) -> HRESULT {
    act->ShutdownObject();
    return act->ActivateObject(IID_PPV_ARGS(&s));
  };
  auto drop = [act] { act->ShutdownObject(); };
  return MfCamera::create(dev.info, again, drop, src, why);
}

std::unique_ptr<Camera> mfOpenSource(MakeSource make, const Info& info, std::string& why) {
  why.clear();
  return openMade(info, std::move(make), why);
}

void mfTestSource(const std::string& name, const std::string& label, MakeSource make) {
  std::lock_guard<std::mutex> lk(gTestMu);
  if (!make) tests().erase(name);
  else tests()[name] = Test{label, std::move(make)};
}

}  // namespace lizard::cam
