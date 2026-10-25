// ai: The fake webcam behind IMFMediaSource (2026-10-25; fake_mf.h says what it offers). A source and its one stream
// ai: share a count (the stream a part of the source); events through Media Foundation's own event queues; its MF
// ai: functions looked up in mfplat at run time as the backend's are, and an MFStartup held for its life. Start sends
// ai: MENewStream (MEUpdatedStream for a stream started before), MEStreamStarted and MESourceStarted; Stop
// ai: MEStreamStopped and MESourceStopped and drops the requests; Shutdown ends its thread and its queues. A thread of
// ai: its own waits for each frame's time, makes the sample (the picture painted, packed or WIC-encoded) and queues
// ai: MEMediaSample with the request's token.
#include "fake_mf.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <strmif.h>
#include <ks.h>
#include <ksmedia.h>
#include <ksproxy.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "trigger.h"

namespace lizard::sim {

namespace {

using Microsoft::WRL::ComPtr;
using cam::steadyNs;

struct Api {
  HRESULT(WINAPI* startup)(ULONG, DWORD) = nullptr;
  HRESULT(WINAPI* shutdown)() = nullptr;
  HRESULT(WINAPI* eventQueue)(IMFMediaEventQueue**) = nullptr;
  HRESULT(WINAPI* mediaType)(IMFMediaType**) = nullptr;
  HRESULT(WINAPI* streamDescriptor)(DWORD, DWORD, IMFMediaType**, IMFStreamDescriptor**) = nullptr;
  HRESULT(WINAPI* presentation)(DWORD, IMFStreamDescriptor**, IMFPresentationDescriptor**) = nullptr;
  HRESULT(WINAPI* sample)(IMFSample**) = nullptr;
  HRESULT(WINAPI* memoryBuffer)(DWORD, IMFMediaBuffer**) = nullptr;
  HRESULT(WINAPI* buffer2D)(DWORD, DWORD, DWORD, BOOL, IMFMediaBuffer**) = nullptr;
  bool ok = false;
};

const Api& api() {
  static const Api a = [] {
    Api x;
    HMODULE m = LoadLibraryExW(L"mfplat.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!m) return x;
    bool all = true;
    auto get = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(reinterpret_cast<void*>(GetProcAddress(m, name)));
      all = all && fn;
    };
    get(x.startup, "MFStartup");
    get(x.shutdown, "MFShutdown");
    get(x.eventQueue, "MFCreateEventQueue");
    get(x.mediaType, "MFCreateMediaType");
    get(x.streamDescriptor, "MFCreateStreamDescriptor");
    get(x.presentation, "MFCreatePresentationDescriptor");
    get(x.sample, "MFCreateSample");
    get(x.memoryBuffer, "MFCreateMemoryBuffer");
    get(x.buffer2D, "MFCreate2DMediaBuffer");
    x.ok = all;
    return x;
  }();
  return a;
}

// ai: the native types, in the order the stream lists them; lo: the range's bottom (0 none); RGB24 and RGB32 are
// ai: bottom-up (a negative default stride), RGB24 in a plain buffer, RGB32 in a 2D one
struct Kind { const GUID* sub; const char* name; uint32_t w, h, fps, lo; };
const Kind kKinds[] = {
    {&MFVideoFormat_MJPG, "MJPG", 1920, 1080, 30, 5},
    {&MFVideoFormat_YUY2, "YUY2", 1280, 720, 30, 0},
    {&MFVideoFormat_NV12, "NV12", 640, 480, 60, 0},
    {&MFVideoFormat_L8, "L8", 640, 480, 30, 0},
    {&MFVideoFormat_RGB24, "RGB24", 320, 240, 30, 0},
    {&MFVideoFormat_RGB32, "RGB32", 320, 240, 15, 0},
};

// ai: a kind's bytes a pixel in its first plane (MJPG 0)
uint32_t bytesOf(const Kind& k) {
  if (*k.sub == MFVideoFormat_YUY2) return 2;
  if (*k.sub == MFVideoFormat_RGB24) return 3;
  if (*k.sub == MFVideoFormat_RGB32) return 4;
  return *k.sub == MFVideoFormat_MJPG ? 0 : 1;
}

// ai: the gradient's picture p into luma (w x h, rows stride apart), a row's values from a table of the wave
void gradient(uint32_t p, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride) {
  static const std::vector<uint8_t> wave = [] {
    std::vector<uint8_t> t(4096);
    for (uint32_t i = 0; i < t.size(); i++) t[i] = fakeMfGradient(i, 0, 0);
    return t;
  }();
  for (uint32_t y = 0; y < h; y++) {
    const uint32_t base = (2 * y + 30 * (p % kFakeMfPictures)) % 510;
    uint8_t* row = luma + size_t(stride) * y;
    for (uint32_t x = 0; x < w; x++) row[x] = wave[(base + x) % 510];
  }
}

// ai: JPEG's Annex K.3 Huffman tables, the counts a code length of each (DC, AC; luminance, chrominance)
const uint8_t kStdCounts[2][2][16] = {
    {{0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0}},
    {{0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d}, {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77}},
};

// ai: the frame's DHT segments cut where every table in them has the standard counts (a decoder then takes the
// ai: standard tables, as it does for a UVC camera's MJPEG); false (j as it was) where one does not
bool cutTables(std::vector<uint8_t>& j) {
  if (j.size() < 4 || j[0] != 0xff || j[1] != 0xd8) return false;
  std::vector<uint8_t> out(j.begin(), j.begin() + 2);
  size_t i = 2;
  bool cut = false;
  while (i + 4 <= j.size()) {
    if (j[i] != 0xff) return false;
    const uint8_t m = j[i + 1];
    const size_t len = (size_t(j[i + 2]) << 8) | j[i + 3];
    if (m == 0xda) {   // ai: SOS: the rest is the scan
      out.insert(out.end(), j.begin() + ptrdiff_t(i), j.end());
      if (cut) j.swap(out);
      return cut;
    }
    if (len < 2 || i + 2 + len > j.size()) return false;
    if (m == 0xc4) {
      size_t k = i + 4;
      const size_t end = i + 2 + len;
      while (k < end) {
        const uint8_t tc = j[k] >> 4, th = j[k] & 15;
        if (tc > 1 || th > 1 || k + 17 > end || std::memcmp(&j[k + 1], kStdCounts[tc][th], 16)) return false;
        size_t n = 0;
        for (int c = 0; c < 16; c++) n += j[k + 1 + size_t(c)];
        k += 17 + n;
      }
      cut = true;
    } else {
      out.insert(out.end(), j.begin() + ptrdiff_t(i), j.begin() + ptrdiff_t(i + 2 + len));
    }
    i += 2 + len;
  }
  return false;
}

// ai: a grey picture as a JPEG through WIC: B = G = R = Y in, YCbCr 4:2:2 at quality 0.9 out (4:2:0 where the
// ai: encoder takes no subsampling option)
bool wicJpeg(const uint8_t* grey, uint32_t w, uint32_t h, uint32_t stride, std::vector<uint8_t>& out, std::string& why) {
  ComPtr<IWICImagingFactory> f;
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f));
  if (FAILED(hr)) {
    why = "no WIC imaging factory";
    return false;
  }
  std::vector<uint8_t> bgr(size_t(w) * 3 * h);
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* s = grey + size_t(stride) * y;
    uint8_t* d = bgr.data() + size_t(w) * 3 * y;
    for (uint32_t x = 0; x < w; x++) d[3 * x] = d[3 * x + 1] = d[3 * x + 2] = s[x];
  }
  std::vector<uint8_t> room(bgr.size() + 65536);
  ComPtr<IWICStream> st;
  ComPtr<IWICBitmapEncoder> e;
  ComPtr<IWICBitmapFrameEncode> fr;
  ComPtr<IPropertyBag2> bag;
  if (FAILED(hr = f->CreateStream(&st)) || FAILED(hr = st->InitializeFromMemory(room.data(), DWORD(room.size()))) ||
      FAILED(hr = f->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &e)) ||
      FAILED(hr = e->Initialize(st.Get(), WICBitmapEncoderNoCache)) || FAILED(hr = e->CreateNewFrame(&fr, &bag))) {
    why = "WIC's JPEG encoder would not start";
    return false;
  }
  auto option = [&](const wchar_t* name, VARTYPE vt, auto value) {
    PROPBAG2 p{};
    VARIANT v;
    std::memset(&v, 0, sizeof v);
    p.pstrName = const_cast<LPOLESTR>(name);
    v.vt = vt;
    if (vt == VT_R4) v.fltVal = float(value);
    else v.bVal = BYTE(value);
    bag->Write(1, &p, &v);
  };
  option(L"ImageQuality", VT_R4, 0.9);
  option(L"JpegYCrCbSubsampling", VT_UI1, int(WICJpegYCrCbSubsampling422));
  WICPixelFormatGUID pf = GUID_WICPixelFormat24bppBGR;
  if (FAILED(hr = fr->Initialize(bag.Get())) || FAILED(hr = fr->SetSize(w, h)) || FAILED(hr = fr->SetPixelFormat(&pf)) ||
      pf != GUID_WICPixelFormat24bppBGR ||
      FAILED(hr = fr->WritePixels(h, w * 3, UINT(bgr.size()), bgr.data())) || FAILED(hr = fr->Commit()) ||
      FAILED(hr = e->Commit())) {
    why = "WIC's JPEG encoder refused the picture";
    return false;
  }
  ULARGE_INTEGER pos{};
  LARGE_INTEGER zero{};
  if (FAILED(st->Seek(zero, STREAM_SEEK_CUR, &pos)) || !pos.QuadPart || pos.QuadPart > room.size()) {
    why = "WIC's JPEG came out empty";
    return false;
  }
  out.assign(room.begin(), room.begin() + ptrdiff_t(pos.QuadPart));
  return true;
}

// ai: the gradient's cycle as JPEG frames, made once a process for each size and table choice; tables: whether they
// ai: were cut
struct Cycle { std::vector<std::vector<uint8_t>> jpg; bool cut = false; std::string why; };
const Cycle& cycle(uint32_t w, uint32_t h, bool strip) {
  static std::mutex mu;
  static std::vector<std::pair<uint64_t, Cycle>> made;
  const uint64_t key = (uint64_t(w) << 33) | (uint64_t(h) << 1) | (strip ? 1 : 0);
  std::lock_guard<std::mutex> lk(mu);
  for (auto& [k, c] : made)
    if (k == key) return c;
  Cycle c;
  std::vector<uint8_t> grey(size_t(w) * h);
  bool cutAll = strip;
  for (uint32_t p = 0; p < kFakeMfPictures; p++) {
    gradient(p, w, h, grey.data(), w);
    std::vector<uint8_t> j;
    if (!wicJpeg(grey.data(), w, h, w, j, c.why)) {
      c.jpg.clear();
      break;
    }
    c.jpg.push_back(std::move(j));
  }
  if (strip && !c.jpg.empty()) {
    // ai: cut from every frame or from none, so a cycle is one kind of MJPEG
    std::vector<std::vector<uint8_t>> cutJpg = c.jpg;
    for (auto& j : cutJpg) cutAll = cutAll && cutTables(j);
    if (cutAll) c.jpg.swap(cutJpg);
  }
  c.cut = strip && cutAll && !c.jpg.empty();
  made.emplace_back(key, std::move(c));
  return made.back().second;
}

// ai: PROPSETID_VIDCAP_CAMERACONTROL (as camera/mf.cpp declares it) and the priority's id there
const GUID kCameraControlSet = {0xc6e13370, 0x30ac, 0x11d0, {0xa1, 0x8c, 0x00, 0xa0, 0xc9, 0x11, 0x89, 0x56}};
constexpr ULONG kAutoExposurePriority = 19;

class Source;

// ai: IAMCameraControl or IAMVideoProcAmp (alike but for their ids) over the device's controls, a part of its source
template <class I, bool Cam>
class Props final : public I {
 public:
  explicit Props(Source* s) : s_(s) {}
  STDMETHODIMP QueryInterface(REFIID iid, void** out) override;
  STDMETHODIMP_(ULONG) AddRef() override;
  STDMETHODIMP_(ULONG) Release() override;
  STDMETHODIMP GetRange(LONG p, LONG* min, LONG* max, LONG* step, LONG* def, LONG* caps) override;
  STDMETHODIMP Set(LONG p, LONG v, LONG flags) override;
  STDMETHODIMP Get(LONG p, LONG* v, LONG* flags) override;

 private:
  Source* s_;
};

// ai: IKsControl over the auto-exposure priority alone, a part of its source
class Ks final : public IKsControl {
 public:
  explicit Ks(Source* s) : s_(s) {}
  STDMETHODIMP QueryInterface(REFIID iid, void** out) override;
  STDMETHODIMP_(ULONG) AddRef() override;
  STDMETHODIMP_(ULONG) Release() override;
  STDMETHODIMP KsProperty(PKSPROPERTY p, ULONG plen, LPVOID data, ULONG dlen, ULONG* got) override;
  STDMETHODIMP KsMethod(PKSMETHOD, ULONG, LPVOID, ULONG, ULONG*) override { return E_NOTIMPL; }
  STDMETHODIMP KsEvent(PKSEVENT, ULONG, LPVOID, ULONG, ULONG*) override { return E_NOTIMPL; }

 private:
  Source* s_;
};

// ai: the stream, a part of its source: its count is the source's
class Stream final : public IMFMediaStream {
 public:
  explicit Stream(Source* s) : s_(s) {}
  STDMETHODIMP QueryInterface(REFIID iid, void** out) override;
  STDMETHODIMP_(ULONG) AddRef() override;
  STDMETHODIMP_(ULONG) Release() override;
  STDMETHODIMP GetEvent(DWORD flags, IMFMediaEvent** e) override;
  STDMETHODIMP BeginGetEvent(IMFAsyncCallback* cb, IUnknown* state) override;
  STDMETHODIMP EndGetEvent(IMFAsyncResult* r, IMFMediaEvent** e) override;
  STDMETHODIMP QueueEvent(MediaEventType t, REFGUID ext, HRESULT hr, const PROPVARIANT* v) override;
  STDMETHODIMP GetMediaSource(IMFMediaSource** out) override;
  STDMETHODIMP GetStreamDescriptor(IMFStreamDescriptor** out) override;
  STDMETHODIMP RequestSample(IUnknown* token) override;

 private:
  Source* s_;
};

class Source final : public FakeMf {
 public:
  explicit Source(const FakeMfConfig& c) : cfg_(c), stream_(this), cam_(this), amp_(this), ks_(this) {}

  bool init(std::string& why) {
    const Api& a = api();
    if (!a.ok) {
      why = "Media Foundation's mfplat is missing a function the fake source calls";
      return false;
    }
    started_mf_ = SUCCEEDED(a.startup(MF_VERSION, MFSTARTUP_LITE));
    if (!started_mf_) {
      why = "Media Foundation did not start";
      return false;
    }
    std::vector<ComPtr<IMFMediaType>> types;
    for (const Kind& k : kKinds) {
      ComPtr<IMFMediaType> t;
      if (FAILED(a.mediaType(&t))) {
        why = "no media type";
        return false;
      }
      t->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
      t->SetGUID(MF_MT_SUBTYPE, *k.sub);
      MFSetAttributeSize(t.Get(), MF_MT_FRAME_SIZE, k.w, k.h);
      MFSetAttributeRatio(t.Get(), MF_MT_FRAME_RATE, k.fps, 1);
      MFSetAttributeRatio(t.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
      t->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
      t->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE);
      if (k.lo) {
        MFSetAttributeRatio(t.Get(), MF_MT_FRAME_RATE_RANGE_MIN, k.lo, 1);
        MFSetAttributeRatio(t.Get(), MF_MT_FRAME_RATE_RANGE_MAX, k.fps, 1);
      }
      const uint32_t px = bytesOf(k);
      const bool nv12 = *k.sub == MFVideoFormat_NV12, rgb = px >= 3, l8 = *k.sub == MFVideoFormat_L8;
      if (px) {
        t->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, TRUE);
        t->SetUINT32(MF_MT_SAMPLE_SIZE, nv12 ? k.w * k.h * 3 / 2 : k.w * px * k.h);
        // ai: L8 names no stride (the backend's own); RGB's negative: bottom-up
        if (!l8) t->SetUINT32(MF_MT_DEFAULT_STRIDE, rgb ? UINT32(-int32_t(k.w * px)) : k.w * px);
      }
      types.push_back(t);
    }
    std::vector<IMFMediaType*> raw;
    for (auto& t : types) raw.push_back(t.Get());
    ComPtr<IMFMediaTypeHandler> h;
    IMFStreamDescriptor* sd = nullptr;
    if (FAILED(a.streamDescriptor(0, DWORD(raw.size()), raw.data(), &sd))) {
      why = "no stream descriptor";
      return false;
    }
    sd_.Attach(sd);
    // ai: the current type at first a device's default, its first listed
    if (FAILED(sd_->GetMediaTypeHandler(&h)) || FAILED(h->SetCurrentMediaType(raw[0])) ||
        FAILED(a.presentation(1, &sd, &pd_)) || FAILED(pd_->SelectStream(0)) || FAILED(a.eventQueue(&q_)) ||
        FAILED(a.eventQueue(&sq_))) {
      why = "the fake source's descriptors or queues were not made";
      return false;
    }
    th_ = std::thread([this] { run(); });
    return true;
  }

  ~Source() {
    if (!dead_) Shutdown();
    // ai: the frames' thread holds no reference, so the last Release never runs on it
    if (th_.joinable()) th_.join();
    for (IUnknown* t : requests_)
      if (t) t->Release();
    q_.Reset();
    sq_.Reset();
    pd_.Reset();
    sd_.Reset();
    if (started_mf_) api().shutdown();
  }

  // ai: IUnknown
  STDMETHODIMP QueryInterface(REFIID iid, void** out) override {
    if (!out) return E_POINTER;
    if (iid == IID_IUnknown || iid == IID_IMFMediaEventGenerator || iid == IID_IMFMediaSource) {
      *out = static_cast<IMFMediaSource*>(this);
    } else if (cfg_.controls && iid == IID_IAMCameraControl) {
      *out = static_cast<IAMCameraControl*>(&cam_);
    } else if (cfg_.controls && iid == IID_IAMVideoProcAmp) {
      *out = static_cast<IAMVideoProcAmp*>(&amp_);
    } else if (cfg_.controls && iid == IID_IKsControl) {
      *out = static_cast<IKsControl*>(&ks_);
    } else {
      *out = nullptr;
      return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
  }
  STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
  STDMETHODIMP_(ULONG) Release() override {
    const ULONG n = --ref_;
    if (!n) delete this;
    return n;
  }

  // ai: IMFMediaEventGenerator, the source's queue
  STDMETHODIMP GetEvent(DWORD flags, IMFMediaEvent** e) override {
    ComPtr<IMFMediaEventQueue> q;
    if (!queue(q_, q)) return MF_E_SHUTDOWN;
    return q->GetEvent(flags, e);
  }
  STDMETHODIMP BeginGetEvent(IMFAsyncCallback* cb, IUnknown* state) override {
    ComPtr<IMFMediaEventQueue> q;
    if (!queue(q_, q)) return MF_E_SHUTDOWN;
    return q->BeginGetEvent(cb, state);
  }
  STDMETHODIMP EndGetEvent(IMFAsyncResult* r, IMFMediaEvent** e) override {
    ComPtr<IMFMediaEventQueue> q;
    if (!queue(q_, q)) return MF_E_SHUTDOWN;
    return q->EndGetEvent(r, e);
  }
  STDMETHODIMP QueueEvent(MediaEventType t, REFGUID ext, HRESULT hr, const PROPVARIANT* v) override {
    ComPtr<IMFMediaEventQueue> q;
    if (!queue(q_, q)) return MF_E_SHUTDOWN;
    return q->QueueEventParamVar(t, ext, hr, v);
  }

  // ai: IMFMediaSource
  STDMETHODIMP GetCharacteristics(DWORD* c) override {
    if (!c) return E_POINTER;
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return MF_E_SHUTDOWN;
    *c = MFMEDIASOURCE_IS_LIVE;
    return S_OK;
  }
  STDMETHODIMP CreatePresentationDescriptor(IMFPresentationDescriptor** pd) override {
    if (!pd) return E_POINTER;
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return MF_E_SHUTDOWN;
    return pd_->Clone(pd);
  }
  STDMETHODIMP Start(IMFPresentationDescriptor* pd, const GUID* format, const PROPVARIANT* pos) override {
    (void)pos;   // ai: a live source starts now, wherever it is asked to
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return MF_E_SHUTDOWN;
    if (!pd) return E_INVALIDARG;
    if (format && *format != GUID_NULL) return MF_E_UNSUPPORTED_TIME_FORMAT;
    BOOL selected = FALSE;
    ComPtr<IMFStreamDescriptor> sd;
    HRESULT hr = pd->GetStreamDescriptorByIndex(0, &selected, &sd);
    if (FAILED(hr)) return hr;
    starts_++;
    PROPVARIANT at;
    PropVariantInit(&at);
    at.vt = VT_I8;
    at.hVal.QuadPart = 0;
    if (selected) {
      if (FAILED(hr = current())) return hr;
      hr = q_->QueueEventParamUnk(active_ ? MEUpdatedStream : MENewStream, GUID_NULL, S_OK,
                                  static_cast<IMFMediaStream*>(&stream_));
      if (FAILED(hr)) return hr;
      active_ = true;
      sq_->QueueEventParamVar(MEStreamStarted, GUID_NULL, S_OK, &at);
      started_ = true;
      gen_++;
      startNs_ = steadyNs();
      next_ = startNs_ + interval_;
      frame_ = 0;
      sentFirst_ = false;
      cv_.notify_all();
    }
    return q_->QueueEventParamVar(MESourceStarted, GUID_NULL, S_OK, &at);
  }
  STDMETHODIMP Stop() override {
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return MF_E_SHUTDOWN;
    stops_++;
    if (started_) {
      started_ = false;
      gen_++;
      drop();
      sq_->QueueEventParamVar(MEStreamStopped, GUID_NULL, S_OK, nullptr);
    }
    return q_->QueueEventParamVar(MESourceStopped, GUID_NULL, S_OK, nullptr);
  }
  STDMETHODIMP Pause() override {
    std::lock_guard<std::mutex> lk(mu_);
    return dead_ ? MF_E_SHUTDOWN : MF_E_INVALID_STATE_TRANSITION;   // ai: a live source does not pause
  }
  STDMETHODIMP Shutdown() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (dead_) return MF_E_SHUTDOWN;
      dead_ = true;
      started_ = false;
      drop();
    }
    cv_.notify_all();
    if (th_.joinable() && th_.get_id() != std::this_thread::get_id()) th_.join();
    if (q_) q_->Shutdown();
    if (sq_) sq_->Shutdown();
    return S_OK;
  }

  // ai: FakeMf
  void stall(bool on) override {
    std::lock_guard<std::mutex> lk(mu_);
    stall_ = on;
  }
  void fail(HRESULT hr) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (!dead_) q_->QueueEventParamVar(MEError, GUID_NULL, hr, nullptr);
  }
  std::string stats() override {
    std::lock_guard<std::mutex> lk(mu_);
    char b[512];
    std::snprintf(b, sizeof b,
                  "{\"starts\":%u,\"stops\":%u,\"shutdown\":%s,\"requests\":%llu,\"samples\":%llu,\"lost\":%llu,"
                  "\"stalled\":%llu,\"type\":\"%s %ux%u@%.4g\",\"tables\":\"%s\"}",
                  starts_, stops_, dead_ ? "true" : "false", (unsigned long long)nRequests_,
                  (unsigned long long)nSamples_, (unsigned long long)nLost_, (unsigned long long)nStalled_,
                  kind_ ? kind_->name : "", kind_ ? kind_->w : 0, kind_ ? kind_->h : 0, fps_,
                  painted_ == 1 ? "cut" : painted_ == 2 ? "kept" : tables_);
    return b;
  }

  // ai: the stream's, under its source's lock
  bool queue(const ComPtr<IMFMediaEventQueue>& from, ComPtr<IMFMediaEventQueue>& q) {
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return false;
    q = from;
    return true;
  }
  HRESULT request(IUnknown* token) {
    std::lock_guard<std::mutex> lk(mu_);
    if (dead_) return MF_E_SHUTDOWN;
    if (!started_) return MF_E_MEDIA_SOURCE_WRONGSTATE;
    if (token) token->AddRef();
    requests_.push_back(token);
    nRequests_++;
    return S_OK;
  }
  ComPtr<IMFMediaEventQueue> sq() { return sq_; }
  ComPtr<IMFStreamDescriptor> sd() { return sd_; }
  FakeMfControls* controls() { return cfg_.controls.get(); }
  bool dead() {
    std::lock_guard<std::mutex> lk(mu_);
    return dead_;
  }

 private:
  // ai: the type current on the stream's handler (mu_ held): what the frames are and when they fall due
  HRESULT current() {
    ComPtr<IMFMediaTypeHandler> h;
    ComPtr<IMFMediaType> t;
    HRESULT hr = sd_->GetMediaTypeHandler(&h);
    if (SUCCEEDED(hr) && FAILED(h->GetCurrentMediaType(&t))) hr = h->GetMediaTypeByIndex(0, &t);
    if (FAILED(hr)) return hr;
    GUID sub{};
    t->GetGUID(MF_MT_SUBTYPE, &sub);
    kind_ = nullptr;
    for (const Kind& k : kKinds)
      if (*k.sub == sub) kind_ = &k;
    if (!kind_) return MF_E_INVALIDMEDIATYPE;
    UINT32 n = 0, d = 0;
    if (FAILED(MFGetAttributeRatio(t.Get(), MF_MT_FRAME_RATE, &n, &d)) || !n || !d) {
      n = kind_->fps;
      d = 1;
    }
    fps_ = double(n) / d;
    interval_ = int64_t(1e9 * d / n);
    tables_ = "none";
    if (*kind_->sub == MFVideoFormat_MJPG && !cfg_.paint) {
      const Cycle& c = cycle(kind_->w, kind_->h, cfg_.stripTables);
      if (c.jpg.empty()) return E_FAIL;
      jpg_ = &c;
      tables_ = c.cut ? "cut" : "kept";
    }
    return S_OK;
  }

  void drop() {
    for (IUnknown* t : requests_)
      if (t) t->Release();
    requests_.clear();
  }

  // ai: one frame's sample: the picture painted, then packed into the type's layout or encoded
  HRESULT make(const Kind& k, uint64_t n, int64_t t0, LONGLONG t, bool first, ComPtr<IMFSample>& s) {
    const Api& a = api();
    HRESULT hr = a.sample(&s);
    if (FAILED(hr)) return hr;
    luma_.resize(size_t(k.w) * k.h);
    const bool mjpg = *k.sub == MFVideoFormat_MJPG;
    if (cfg_.paint) cfg_.paint(t0, k.w, k.h, luma_.data(), k.w);
    else if (!mjpg) gradient(uint32_t(n % kFakeMfPictures), k.w, k.h, luma_.data(), k.w);
    ComPtr<IMFMediaBuffer> b;
    BYTE* p = nullptr;
    DWORD max = 0, cur = 0;
    if (mjpg) {
      std::vector<uint8_t> own;
      const std::vector<uint8_t>* j = nullptr;
      if (cfg_.paint) {
        std::string why;
        if (!wicJpeg(luma_.data(), k.w, k.h, k.w, own, why)) return E_FAIL;
        painted_ = cfg_.stripTables && cutTables(own) ? 1 : 2;
        j = &own;
      } else {
        j = &jpg_->jpg[n % jpg_->jpg.size()];
      }
      size_t len = j->size();
      if (cfg_.corruptEvery > 0 && n % uint64_t(cfg_.corruptEvery) == uint64_t(cfg_.corruptEvery - 1)) len /= 2;
      if (FAILED(hr = a.memoryBuffer(DWORD(len), &b)) || FAILED(hr = b->Lock(&p, &max, &cur))) return hr;
      std::memcpy(p, j->data(), len);
      b->Unlock();
      b->SetCurrentLength(DWORD(len));
    } else if (*k.sub == MFVideoFormat_NV12) {
      ComPtr<IMF2DBuffer> b2;
      BYTE* row = nullptr;
      LONG pitch = 0;
      if (FAILED(hr = a.buffer2D(k.w, k.h, MFVideoFormat_NV12.Data1, FALSE, &b)) || FAILED(hr = b.As(&b2)) ||
          FAILED(hr = b2->Lock2D(&row, &pitch)))
        return hr;
      for (uint32_t y = 0; y < k.h; y++) std::memcpy(row + ptrdiff_t(pitch) * y, luma_.data() + size_t(k.w) * y, k.w);
      for (uint32_t y = 0; y < k.h / 2; y++) std::memset(row + ptrdiff_t(pitch) * (k.h + y), 128, k.w);
      b2->Unlock2D();
      DWORD len = 0;
      b2->GetContiguousLength(&len);
      b->SetCurrentLength(len);
    } else if (*k.sub == MFVideoFormat_RGB32) {
      // ai: a bottom-up 2D buffer: Lock2D's pitch negative, its first row the picture's top
      ComPtr<IMF2DBuffer> b2;
      BYTE* top = nullptr;
      LONG pitch = 0;
      if (FAILED(hr = a.buffer2D(k.w, k.h, MFVideoFormat_RGB32.Data1, TRUE, &b)) || FAILED(hr = b.As(&b2)) ||
          FAILED(hr = b2->Lock2D(&top, &pitch)))
        return hr;
      for (uint32_t y = 0; y < k.h; y++) {
        BYTE* row = top + ptrdiff_t(pitch) * y;
        const uint8_t* l = luma_.data() + size_t(k.w) * y;
        for (uint32_t x = 0; x < k.w; x++) {
          row[4 * x] = row[4 * x + 1] = row[4 * x + 2] = l[x];
          row[4 * x + 3] = 255;
        }
      }
      b2->Unlock2D();
      DWORD len = 0;
      b2->GetContiguousLength(&len);
      b->SetCurrentLength(len);
    } else if (*k.sub == MFVideoFormat_RGB24) {
      // ai: bottom-up in a plain buffer: the buffer's first row the picture's last
      const DWORD row = DWORD(k.w) * 3, len = row * k.h;
      if (FAILED(hr = a.memoryBuffer(len, &b)) || FAILED(hr = b->Lock(&p, &max, &cur))) return hr;
      for (uint32_t y = 0; y < k.h; y++) {
        BYTE* d = p + size_t(row) * (k.h - 1 - y);
        const uint8_t* l = luma_.data() + size_t(k.w) * y;
        for (uint32_t x = 0; x < k.w; x++) d[3 * x] = d[3 * x + 1] = d[3 * x + 2] = l[x];
      }
      b->Unlock();
      b->SetCurrentLength(len);
    } else {
      const bool yuy2 = *k.sub == MFVideoFormat_YUY2;
      const DWORD len = DWORD(k.w) * k.h * (yuy2 ? 2 : 1);
      if (FAILED(hr = a.memoryBuffer(len, &b)) || FAILED(hr = b->Lock(&p, &max, &cur))) return hr;
      if (yuy2) {
        for (size_t i = 0; i < luma_.size(); i++) {
          p[2 * i] = luma_[i];
          p[2 * i + 1] = 128;
        }
      } else {
        std::memcpy(p, luma_.data(), luma_.size());
      }
      b->Unlock();
      b->SetCurrentLength(len);
    }
    if (FAILED(hr = s->AddBuffer(b.Get()))) return hr;
    s->SetSampleTime(t);
    s->SetSampleDuration(LONGLONG(interval_ / 100));
    s->SetUINT32(MFSampleExtension_CleanPoint, TRUE);
    if (first) s->SetUINT32(MFSampleExtension_Discontinuity, TRUE);
    return S_OK;
  }

  // ai: the frames' thread: each frame's time slept to while started, the sample made outside the lock, then queued
  // ai: where the stream is still the one it was made for
  void run() {
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      cv_.wait(lk, [&] { return dead_ || started_; });
      if (dead_) break;
      const int64_t at = next_;
      const uint64_t gen = gen_;
      lk.unlock();
      cam::sleepUntil(at, 200'000);
      lk.lock();
      if (dead_) break;
      if (!started_ || gen != gen_) continue;
      const uint64_t n = frame_++;
      next_ += interval_;
      const int64_t now = steadyNs();
      if (now > next_ + interval_) {
        // ai: behind by frames (a slow paint): those frames lost, as a camera's
        const int64_t behind = (now - next_) / interval_;
        frame_ += uint64_t(behind);
        next_ += behind * interval_;
        nLost_ += uint64_t(behind);
      }
      if (stall_) {
        nStalled_++;
        continue;
      }
      if (requests_.empty()) {
        nLost_++;
        continue;
      }
      IUnknown* token = requests_.front();
      requests_.pop_front();
      const Kind k = *kind_;
      const LONGLONG t = LONGLONG((at - startNs_) / 100);
      const bool first = !sentFirst_;
      sentFirst_ = true;
      lk.unlock();
      ComPtr<IMFSample> s;
      HRESULT hr = make(k, n, at, t, first, s);
      if (token) {
        if (s) s->SetUnknown(MFSampleExtension_Token, token);
        token->Release();
      }
      lk.lock();
      if (dead_) break;
      if (!started_ || gen != gen_) continue;
      if (FAILED(hr)) sq_->QueueEventParamVar(MEError, GUID_NULL, hr, nullptr);
      else if (SUCCEEDED(sq_->QueueEventParamUnk(MEMediaSample, GUID_NULL, S_OK, s.Get()))) nSamples_++;
    }
    lk.unlock();
    if (SUCCEEDED(com)) CoUninitialize();
  }

  FakeMfConfig cfg_;
  Stream stream_;
  Props<IAMCameraControl, true> cam_;
  Props<IAMVideoProcAmp, false> amp_;
  Ks ks_;
  std::atomic<ULONG> ref_{1};
  bool started_mf_ = false;
  std::mutex mu_;
  std::condition_variable cv_;
  ComPtr<IMFMediaEventQueue> q_, sq_;
  ComPtr<IMFPresentationDescriptor> pd_;
  ComPtr<IMFStreamDescriptor> sd_;
  std::thread th_;
  // ai: mu_: the state, the current type, the frames' schedule, the requests waiting, the counts
  bool dead_ = false, started_ = false, active_ = false, stall_ = false, sentFirst_ = false;
  uint64_t gen_ = 0;
  const Kind* kind_ = nullptr;
  const Cycle* jpg_ = nullptr;
  const char* tables_ = "none";
  double fps_ = 0;
  int64_t interval_ = 33'333'333, startNs_ = 0, next_ = 0;
  uint64_t frame_ = 0;
  std::deque<IUnknown*> requests_;
  unsigned starts_ = 0, stops_ = 0;
  uint64_t nRequests_ = 0, nSamples_ = 0, nLost_ = 0, nStalled_ = 0;
  std::vector<uint8_t> luma_;   // ai: the frames' thread's
  std::atomic<int> painted_{0};  // ai: a painted MJPG frame's tables: 0 none yet, 1 cut, 2 kept
};

STDMETHODIMP Stream::QueryInterface(REFIID iid, void** out) {
  if (!out) return E_POINTER;
  if (iid == IID_IUnknown || iid == IID_IMFMediaEventGenerator || iid == IID_IMFMediaStream) {
    *out = static_cast<IMFMediaStream*>(this);
    AddRef();
    return S_OK;
  }
  *out = nullptr;
  return E_NOINTERFACE;
}
STDMETHODIMP_(ULONG) Stream::AddRef() { return s_->AddRef(); }
STDMETHODIMP_(ULONG) Stream::Release() { return s_->Release(); }
STDMETHODIMP Stream::GetEvent(DWORD flags, IMFMediaEvent** e) {
  if (s_->dead()) return MF_E_SHUTDOWN;
  return s_->sq()->GetEvent(flags, e);
}
STDMETHODIMP Stream::BeginGetEvent(IMFAsyncCallback* cb, IUnknown* state) {
  if (s_->dead()) return MF_E_SHUTDOWN;
  return s_->sq()->BeginGetEvent(cb, state);
}
STDMETHODIMP Stream::EndGetEvent(IMFAsyncResult* r, IMFMediaEvent** e) {
  if (s_->dead()) return MF_E_SHUTDOWN;
  return s_->sq()->EndGetEvent(r, e);
}
STDMETHODIMP Stream::QueueEvent(MediaEventType t, REFGUID ext, HRESULT hr, const PROPVARIANT* v) {
  if (s_->dead()) return MF_E_SHUTDOWN;
  return s_->sq()->QueueEventParamVar(t, ext, hr, v);
}
STDMETHODIMP Stream::GetMediaSource(IMFMediaSource** out) {
  if (!out) return E_POINTER;
  if (s_->dead()) return MF_E_SHUTDOWN;
  *out = s_;
  s_->AddRef();
  return S_OK;
}
STDMETHODIMP Stream::GetStreamDescriptor(IMFStreamDescriptor** out) {
  if (!out) return E_POINTER;
  if (s_->dead()) return MF_E_SHUTDOWN;
  return s_->sd().CopyTo(out);
}
STDMETHODIMP Stream::RequestSample(IUnknown* token) { return s_->request(token); }

template <class I, bool Cam>
STDMETHODIMP Props<I, Cam>::QueryInterface(REFIID iid, void** out) { return s_->QueryInterface(iid, out); }
template <class I, bool Cam>
STDMETHODIMP_(ULONG) Props<I, Cam>::AddRef() { return s_->AddRef(); }
template <class I, bool Cam>
STDMETHODIMP_(ULONG) Props<I, Cam>::Release() { return s_->Release(); }
template <class I, bool Cam>
STDMETHODIMP Props<I, Cam>::GetRange(LONG p, LONG* min, LONG* max, LONG* step, LONG* def, LONG* caps) {
  if (!min || !max || !step || !def || !caps) return E_POINTER;
  FakeMfControls* c = s_->controls();
  std::lock_guard<std::mutex> lk(c->mu);
  auto& m = Cam ? c->cam : c->amp;
  auto it = m.find(p);
  if (it == m.end()) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
  *min = it->second.min;
  *max = it->second.max;
  *step = it->second.step;
  *def = it->second.def;
  *caps = it->second.caps;
  return S_OK;
}
template <class I, bool Cam>
STDMETHODIMP Props<I, Cam>::Set(LONG p, LONG v, LONG flags) {
  FakeMfControls* c = s_->controls();
  std::lock_guard<std::mutex> lk(c->mu);
  auto& m = Cam ? c->cam : c->amp;
  auto it = m.find(p);
  if (it == m.end()) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
  FakeMfControls::Prop& q = it->second;
  if (v < q.min || v > q.max || (q.step > 1 && (v - q.min) % q.step) || ((flags & 1) && !(q.caps & 1)) || !(flags & 3))
    return E_INVALIDARG;
  q.value = v;
  q.flags = flags;
  c->sets++;
  return S_OK;
}
template <class I, bool Cam>
STDMETHODIMP Props<I, Cam>::Get(LONG p, LONG* v, LONG* flags) {
  if (!v || !flags) return E_POINTER;
  FakeMfControls* c = s_->controls();
  std::lock_guard<std::mutex> lk(c->mu);
  auto& m = Cam ? c->cam : c->amp;
  auto it = m.find(p);
  if (it == m.end()) return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
  *v = it->second.value;
  *flags = it->second.flags;
  return S_OK;
}

STDMETHODIMP Ks::QueryInterface(REFIID iid, void** out) { return s_->QueryInterface(iid, out); }
STDMETHODIMP_(ULONG) Ks::AddRef() { return s_->AddRef(); }
STDMETHODIMP_(ULONG) Ks::Release() { return s_->Release(); }
STDMETHODIMP Ks::KsProperty(PKSPROPERTY p, ULONG plen, LPVOID data, ULONG dlen, ULONG* got) {
  if (!p || plen < sizeof(KSPROPERTY)) return E_INVALIDARG;
  if (p->Set != kCameraControlSet || p->Id != kAutoExposurePriority) return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
  if (!data || dlen < sizeof(KSPROPERTY_CAMERACONTROL_S)) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
  auto* s = static_cast<KSPROPERTY_CAMERACONTROL_S*>(data);
  FakeMfControls* c = s_->controls();
  std::lock_guard<std::mutex> lk(c->mu);
  if (p->Flags & KSPROPERTY_TYPE_GET) {
    s->Value = c->aep;
    s->Flags = ULONG(c->aepFlags);
    s->Capabilities = KSPROPERTY_CAMERACONTROL_FLAGS_MANUAL;
    if (got) *got = sizeof *s;
    return S_OK;
  }
  if (p->Flags & KSPROPERTY_TYPE_SET) {
    if (s->Value < 0 || s->Value > 1) return E_INVALIDARG;
    c->aep = s->Value;
    c->aepFlags = long(s->Flags);
    c->sets++;
    if (got) *got = 0;
    return S_OK;
  }
  return E_INVALIDARG;
}

}  // namespace

void fakeMfPaint(uint32_t p, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride) { gradient(p, w, h, luma, stride); }

std::shared_ptr<FakeMfControls> FakeMfControls::webcam() {
  auto c = std::make_shared<FakeMfControls>();
  auto prop = [](long min, long max, long step, long def, bool canAuto) {
    Prop p;
    p.min = min;
    p.max = max;
    p.step = step;
    p.def = def;
    p.caps = canAuto ? 3 : 2;
    p.value = def;
    p.flags = canAuto ? 1 : 2;
    return p;
  };
  c->cam[CameraControl_Zoom] = prop(100, 500, 1, 100, false);
  c->cam[CameraControl_Focus] = prop(0, 250, 5, 0, true);
  c->cam[CameraControl_Exposure] = prop(-11, -2, 1, -5, true);
  c->amp[VideoProcAmp_Gain] = prop(0, 255, 1, 0, false);
  c->amp[VideoProcAmp_Sharpness] = prop(0, 255, 1, 128, false);
  c->amp[VideoProcAmp_BacklightCompensation] = prop(0, 1, 1, 1, false);
  c->amp[VideoProcAmp_Brightness] = prop(0, 255, 1, 128, false);
  c->amp[VideoProcAmp_Contrast] = prop(0, 255, 1, 128, false);
  return c;
}

FakeMf* FakeMf::create(const FakeMfConfig& c, std::string& why) {
  Source* s = new Source(c);
  if (!s->init(why)) {
    s->Release();
    return nullptr;
  }
  return s;
}

}  // namespace lizard::sim

#endif
