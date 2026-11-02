// ai: GenICam cameras through Aravis 0.8 (2026-10-25; on the engine's camera.h since 2026-11-01, free run): USB3
// ai: Vision and GigE Vision on every system but Android, the library loaded at run time (aravis_api.h: where it is
// ai: looked for; absent or another version, no cameras and a note saying why).
// ai: The listing: Aravis's discovery (arv_update_device_list) waits a second for GigE answers, so it runs on a thread
// ai: of its own and a list returns the last discovery's cameras at once, starting another where none runs and the
// ai: last ended more than 2 s ago; only a process's first list waits for its first discovery (3 s at most). The stamp
// ai: (camera.h listStamp) is the last discovery's ids and starts one by the same rule, so a caller that watches it
// ai: (the app, each second on Receive) sees a camera come or go within a discovery or two. Discovery and
// ai: arv_camera_new share one lock (Aravis's interfaces are not thread-safe, and a discovery empties the GigE table
// ai: while it runs); every call on an ArvCamera takes the camera's own (the stream's pop and push are thread-safe).
// ai: Modes: each pixel format luma.h reads (Mono8 first: its bytes are the luma, handed over in place) at the
// ai: camera's largest region in that format, a mode a rate: 30, 60 and 120 where the frame-rate bounds there hold
// ai: them, and the bounds' top where it is above those by more than 0.5% (a machine vision camera's 4K at 120 and
// ai: past it), fastest first; a camera with no frame-rate feature one mode a format, at rate 0. Each read with that
// ai: format and region set, the camera's own put back after. The top depends on the exposure on some cameras: it is
// ai: the one the camera had when its modes were read.
// ai: The camera rules (camera.cpp): exposure and gain as the device has them (the session sets the user's after each
// ai: start); GenICam's standard features name no ISP enhancement; the rate held by the frame-rate limit, Aravis's
// ai: arv_camera_set_frame_rate (AcquisitionFrameRate and its enable by each vendor's names), which also turns every
// ai: trigger off: free run. start, acquisition stopped: the format, the region, every trigger off, Continuous
// ai: acquisition, the mode's rate (Stats::fps the rate read back), GigE's packet size found (a transport setting every
// ai: client finds for itself, not put back) and the stream's socket buffer sized by Aravis, 8 buffers of the payload.
// ai: Each feature start or set changes is remembered with its value before its first change and put back at stop,
// ai: newest change first (a set on a stopped camera at the next stop, or at close).
// ai: A capture thread (its priority raised, sys/priority.h) pops with a 100 ms timeout: a buffer missing packets
// ai: counts as dropped, any other failure as corrupt, either going back to the stream; a gap in the frame ids no failed
// ai: buffer accounts for (frames lost before a buffer held them, the stream's underruns) counts as dropped. A free-run
// ai: frame's time is the device's timestamp moved onto steadyNs by the offset at the stream's first frame (its arrival
// ai: less its stamp; mf.cpp's rule), based again on the arrival where the device's clock goes back; a device that
// ai: stamps none, the arrival. A stream ends by itself (Ended) where the device says its control was lost (GigE's
// ai: heartbeat, USB's unplug) or a camera silent for 3 s does not answer a register read.
// ai: The phase lock's lever (2026-11-01; camera.h delayBar, captures, delayOnce; the session's lock, rx/phase_host.h):
// ai: the camera triggered by software on the host's schedule (trigger.h), one frame a trigger, and a delay moving
// ai: every trigger after it. delayBar, known at open: "" where the camera has TriggerSoftware, a FrameStart trigger
// ai: (where it has a TriggerSelector), TriggerMode, and a TriggerSource offering Software that takes the write (tried
// ai: and put back at once), else why not; a stream at a mode with no rate is not triggered. A stream starts in free
// ai: run (the camera rules above); once 5 frames are in (the camera streaming), the capture thread arms FrameStart's
// ai: trigger (TriggerSource Software, TriggerMode On; a camera refusing that while it acquires stopped around the
// ai: change) and turns the rate limit's switch off (a limit at the mode's rate would refuse triggers a little faster
// ai: than it); waits for the free run's frames still on their way (none for 50 ms; still coming a second on, the
// ai: camera runs free whatever it says: free run, the bar why); fires one trigger and waits for its frame (300 ms,
// ai: three tries; none back: the trigger put off again, free run, the bar saying why from the next start); then begins
// ai: the schedule at a period of 1 / the mode's rate less 0.1% (the desktop's Lock.kt's: the camera a little fast
// ai: against a display at that rate, so the lock's moves are delays). Every change is kept and put back at stop. Each
// ai: frame matched to its trigger (TriggerClock::match) has its trigger's deadline as its time, and one Capture on the
// ai: capture thread after the frame's sink: that time, delayed where the trigger was the one a delay moved (or a lost
// ai: one's mark passed on), fresh on the first after the schedule began, the frame the period (a delayed frame its
// ai: interval from the frame before), the exposure as last read or set. A frame no trigger fits is handed over at its
// ai: device time, never before the frame before it, with no Capture. delayOnce(us) moves the schedule (false where
// ai: none runs). The schedule always moves where a delay was asked, its mark passing to the next trigger where the
// ai: delayed one's frame is lost or its fire refused, so the lock hears the landing from the timestamps and no fate is
// ai: told. report: the schedule's figures (TriggerClock::stats), whether it runs, the bar.
#include "aravis_api.h"
#include "backends.h"
#include "clock.h"
#include "luma.h"
#include "priority.h"
#include "trigger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace lizard::cam {

namespace {

using Fn = void (*)();   // ai: a looked-up call before its cast: void (*)() is the type every function pointer casts from

struct Loaded {
  arv::Api api;
  std::string why;
  bool ok = false;
};

#if defined(_WIN32)
std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n > 0 ? n : 0), '\0');
  if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}
std::string lastError() {
  char* m = nullptr;
  const DWORD code = GetLastError();
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                 code, 0, reinterpret_cast<char*>(&m), 0, nullptr);
  std::string s = m ? m : "error " + std::to_string(code);
  if (m) LocalFree(m);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
  return s;
}
// ai: the folder of a module, with its separator; "" where unknown
std::wstring folderOf(HMODULE h) {
  std::wstring p(32768, L'\0');
  const DWORD n = GetModuleFileNameW(h, p.data(), DWORD(p.size()));
  if (!n || n >= p.size()) return {};
  p.resize(n);
  const size_t cut = p.find_last_of(L"\\/");
  return cut == std::wstring::npos ? std::wstring() : p.substr(0, cut + 1);
}
// ai: the folder of the module holding this code (the app's DLL, cam_check.exe)
std::wstring ownFolder() {
  HMODULE m = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&ownFolder), &m))
    return {};
  return folderOf(m);
}
#else
// ai: the folder of the module holding this code (the app's library, cam_check), with its separator; "" where unknown
std::string ownFolder() {
  Dl_info d{};
  if (!dladdr(reinterpret_cast<void*>(&ownFolder), &d) || !d.dli_fname) return "";
  const std::string s = d.dli_fname;
  const size_t k = s.rfind('/');
  return k == std::string::npos ? "" : s.substr(0, k + 1);
}
#endif

Loaded load() {
  Loaded x;
  std::string tried;   // ai: every file tried and the system's answer, for the why
  auto note = [&](const std::string& where, const std::string& said) { tried += (tried.empty() ? "" : "; ") + where + ": " + said; };
#if defined(_WIN32)
  const std::wstring lib = L"libaravis-0.8-0.dll";
  std::vector<std::wstring> files;
  if (const wchar_t* env = _wgetenv(L"LIZ_ARAVIS")) {
    if (!*env) return x;   // ai: off: no why
    std::wstring p = env;
    const DWORD at = GetFileAttributesW(p.c_str());
    if (at != INVALID_FILE_ATTRIBUTES && (at & FILE_ATTRIBUTE_DIRECTORY)) {
      if (p.back() != L'\\' && p.back() != L'/') p += L'\\';
      p += lib;
    }
    files.push_back(p);
  } else {
    if (const std::wstring own = ownFolder(); !own.empty()) files.push_back(own + lib);
    files.push_back(L"C:\\msys64\\ucrt64\\bin\\" + lib);
    files.push_back(L"C:\\msys64\\clang64\\bin\\" + lib);
    files.push_back(lib);
  }
  for (const auto& f : files) {
    // ai: a path loads the DLLs it needs from its own folder first; the bare name searches as Windows does
    const bool path = f.find_first_of(L"\\/") != std::wstring::npos;
    HMODULE h = path ? LoadLibraryExW(f.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
                     : LoadLibraryW(f.c_str());
    if (!h) { note(narrow(f), lastError()); continue; }
    const std::wstring dir = folderOf(h);
    // ai: GLib's two as Aravis loaded them, else beside it
    auto glib = [&](const wchar_t* name) {
      HMODULE m = GetModuleHandleW(name);
      if (!m && !dir.empty()) m = LoadLibraryExW((dir + name).c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      return m;
    };
    const HMODULE mods[] = {h, glib(L"libglib-2.0-0.dll"), glib(L"libgobject-2.0-0.dll")};
    const std::string from = dir.empty() ? narrow(f) : narrow(dir) + narrow(lib);
    auto sym = [&](const char* name) -> Fn {
      for (HMODULE m : mods)
        if (m)
          if (FARPROC p = GetProcAddress(m, name)) return reinterpret_cast<Fn>(p);
      return nullptr;
    };
    auto close = [&] { FreeLibrary(h); };
#else
#if defined(__APPLE__)
  const std::string lib = "libaravis-0.8.0.dylib";
#else
  const std::string lib = "libaravis-0.8.so.0";
#endif
  std::vector<std::string> files;
  if (const char* env = std::getenv("LIZ_ARAVIS")) {
    if (!*env) return x;   // ai: off: no why
    std::string p = env;
    struct stat st {};
    if (stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) p += (p.back() == '/' ? "" : "/") + lib;
    files.push_back(p);
  } else {
    if (const std::string own = ownFolder(); !own.empty()) files.push_back(own + lib);
#if defined(__APPLE__)
    // ai: Homebrew's (arm64, then Intel), by absolute path: dlopen does not search /opt/homebrew/lib by itself
    for (const char* d : {"/opt/homebrew/lib/", "/opt/homebrew/opt/aravis/lib/", "/usr/local/lib/", "/usr/local/opt/aravis/lib/"})
      files.push_back(std::string(d) + lib);
#endif
    files.push_back(lib);
  }
  for (const auto& f : files) {
    // ai: dlsym through Aravis's handle searches the libraries it needs too (glibc and dyld alike): GLib's calls
    void* h = dlopen(f.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      const char* e = dlerror();   // ai: its words name the file
      tried += (tried.empty() ? "" : "; ") + (e ? std::string(e) : f + ": not loaded");
      continue;
    }
    const std::string& from = f;
    auto sym = [&](const char* name) -> Fn { return reinterpret_cast<Fn>(dlsym(h, name)); };
    auto close = [&] { dlclose(h); };
#endif
    arv::Api api;
    std::string missing;
    int lacks = 0;
    auto get = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym(name));
      if (!fn && lacks++ == 0) missing = name;
    };
#define LIZ_ARV_GET(ret, name, args) get(api.name, #name);
    LIZ_ARV_CALLS(LIZ_ARV_GET)
#undef LIZ_ARV_GET
    if (lacks) {
      note(from, "lacks " + missing + (lacks > 1 ? " and " + std::to_string(lacks - 1) + " more of Aravis 0.8's calls" : std::string()));
      close();
      continue;
    }
    api.vmajor = api.arv_get_major_version();
    api.vminor = api.arv_get_minor_version();
    api.vmicro = api.arv_get_micro_version();
    if (api.vmajor != 0 || api.vminor != 8) {
      note(from, "Aravis " + std::to_string(api.vmajor) + "." + std::to_string(api.vminor) + "." + std::to_string(api.vmicro) +
                     ", where the receiver reads 0.8's API");
      close();
      continue;
    }
    api.from = from;
    x.api = api;
    x.ok = true;
    return x;
  }
  x.why = "Aravis 0.8 is not loaded (" + tried + ")";
  return x;
}

// ai: discovery and arv_camera_new; never destroyed, since a discovery's thread may hold it as the process exits
std::mutex& systemMu() {
  static std::mutex* m = new std::mutex;
  return *m;
}

std::string str(const char* s) { return s ? s : ""; }

std::string hex(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "0x%08llx", static_cast<unsigned long long>(v));
  return b;
}

// ai: One call's GError: p() hands the slot over, emptied first (GLib refuses a call whose error is already set),
// ai: take() its message with the error freed.
class Err {
 public:
  explicit Err(const arv::Api& a) : a_(a) {}
  ~Err() { clear(); }
  Err(const Err&) = delete;
  Err& operator=(const Err&) = delete;
  arv::GError** p() { clear(); return &e_; }
  explicit operator bool() const { return e_ != nullptr; }
  std::string take() {
    std::string m = !e_ ? "" : e_->message ? e_->message : "an error with no message";
    clear();
    return m;
  }
  void clear() {
    if (e_) a_.g_error_free(e_);
    e_ = nullptr;
  }

 private:
  const arv::Api& a_;
  arv::GError* e_ = nullptr;
};

// ai: The pixel formats luma.h reads (PFNC values; Mono10p and Mono12p have no Aravis constant), in the order modes()
// ai: lists them, the cheapest to luma first. GigE's Mono10Packed packs as its Mono12Packed does (each sample's high 8
// ai: bits a byte of their own).
struct Format {
  arv::PixelFormat pf;
  const char* name;
  Pix pix;
  uint8_t bits;
  uint8_t rank;
};
constexpr Format kFormats[] = {
    {0x01080001u, "Mono8", Pix::Grey8, 8, 0},
    {0x01100003u, "Mono10", Pix::Grey16, 10, 1},
    {0x01100005u, "Mono12", Pix::Grey16, 12, 1},
    {0x01100025u, "Mono14", Pix::Grey16, 14, 1},
    {0x01100007u, "Mono16", Pix::Grey16, 16, 1},
    {0x010A0046u, "Mono10p", Pix::Mono10p, 10, 2},
    {0x010C0047u, "Mono12p", Pix::Mono12p, 12, 2},
    {0x010C0004u, "Mono10Packed", Pix::Mono12Packed, 10, 3},
    {0x010C0006u, "Mono12Packed", Pix::Mono12Packed, 12, 3},
    {0x01080008u, "BayerGR8", Pix::Bayer8, 8, 4},
    {0x01080009u, "BayerRG8", Pix::Bayer8, 8, 4},
    {0x0108000Au, "BayerGB8", Pix::Bayer8, 8, 4},
    {0x0108000Bu, "BayerBG8", Pix::Bayer8, 8, 4},
    {0x0110000Cu, "BayerGR10", Pix::Bayer16, 10, 5},
    {0x0110000Du, "BayerRG10", Pix::Bayer16, 10, 5},
    {0x0110000Eu, "BayerGB10", Pix::Bayer16, 10, 5},
    {0x0110000Fu, "BayerBG10", Pix::Bayer16, 10, 5},
    {0x01100010u, "BayerGR12", Pix::Bayer16, 12, 5},
    {0x01100011u, "BayerRG12", Pix::Bayer16, 12, 5},
    {0x01100012u, "BayerGB12", Pix::Bayer16, 12, 5},
    {0x01100013u, "BayerBG12", Pix::Bayer16, 12, 5},
    {0x0110002Eu, "BayerGR16", Pix::Bayer16, 16, 5},
    {0x0110002Fu, "BayerRG16", Pix::Bayer16, 16, 5},
    {0x01100030u, "BayerGB16", Pix::Bayer16, 16, 5},
    {0x01100031u, "BayerBG16", Pix::Bayer16, 16, 5},
    {0x02180014u, "RGB8", Pix::RGB24, 8, 6},
    {0x02180015u, "BGR8", Pix::BGR24, 8, 6},
    {0x02200016u, "RGBa8", Pix::RGBA32, 8, 6},
    {0x02200017u, "BGRa8", Pix::BGRA32, 8, 6},
    {0x02100032u, "YUV422_8", Pix::YUYV, 8, 7},
    {0x0210001Fu, "YUV422_8_UYVY", Pix::UYVY, 8, 7},
};

const Format* formatOf(arv::PixelFormat pf) {
  for (const auto& f : kFormats)
    if (f.pf == pf) return &f;
  return nullptr;
}

constexpr double kRates[] = {120, 60, 30};   // ai: the rates a mode is listed at where the bounds hold them, fastest first
constexpr int kBuffers = 8;
constexpr arv::guint64 kPopUs = 100'000;
constexpr int64_t kSecondNs = 1'000'000'000;
constexpr int64_t kSilentNs = 3 * kSecondNs;   // ai: no buffer this long: the camera asked whether it still answers
// ai: the switch to the trigger (the header): free-run frames before it arms; the quiet that ends the free run's
// ai: frames on their way; a probe trigger's wait for its frame, and its tries; the schedule's period against the
// ai: mode's (the camera that much fast)
constexpr uint64_t kWarm = 5;
constexpr int64_t kQuietNs = 50'000'000;
constexpr int64_t kProbeNs = 300'000'000;
constexpr int kProbes = 3;
constexpr double kFast = 0.001;

class AravisCam final : public Camera {
 public:
  AravisCam(const arv::Api& a, arv::Camera* cam, Info info);
  ~AravisCam() override;
  const Info& info() const override { return info_; }
  std::vector<Mode> modes() override;
  std::vector<Control> controls() override;
  bool set(const std::string& id, double value, bool automatic, std::string& why) override;
  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override;
  void stop() override;
  Stats stats() override;
  std::string report() override;
  std::string delayBar() override;
  void captures(Captures each, Fate fate) override;
  bool delayOnce(int64_t us) override;

 private:
  // ai: the switch to the trigger, the capture thread's (report reads it): not this stream's; free run before the
  // ai: arming; waiting for the free run's last frames; a probe trigger out; the schedule running
  enum class Trig { Off, Free, Drain, Probe, On };
  std::string triggerBar();
  bool fire(std::string& why);
  bool arm(std::string& why);
  void disarm();
  void step(int64_t at);
  void giveUp(const std::string& why);
  std::vector<Mode> readModes();
  void keep(const std::string& key, std::function<void()> back);
  void keepString(const std::string& feature);
  void keepTriggers();
  void putBack();
  void reap();
  void loop();
  void take(arv::Buffer* b, int64_t at);
  bool probe(std::string& why);
  void finish(const std::string& why);
  uint64_t unwrap(uint64_t id);
  static void onLost(void* device, void* self);

  const arv::Api& a_;
  arv::Camera* cam_;
  arv::Device* dev_ = nullptr;
  Info info_;
  std::mutex mu_;   // ai: every call on cam_, and what is under it below
  bool gv_ = false, rate_ = false, expo_ = false, expoAuto_ = false, gain_ = false, gainAuto_ = false;
  std::string enable_;     // ai: AcquisitionFrameRateEnable (FLIR's AcquisitionFrameRateEnabled) where the camera has it
  std::string gainUnit_;   // ai: dB for SFNC's Gain, none for GainRaw's counts
  double expoDef_ = 0, gainDef_ = 0;
  std::vector<Mode> modes_;                   // ai: mu_: the last read, what modes() answers while streaming
  std::vector<std::function<void()>> undo_;   // ai: mu_: what puts each changed feature back, oldest first
  std::set<std::string> kept_;                // ai: mu_: the features undo_ holds

  arv::Stream* stream_ = nullptr;   // ai: set and freed under mu_ with the capture thread not running
  std::thread th_;
  std::atomic<bool> run_{false}, lost_{false};
  Sink sink_;
  Ended ended_;
  // ai: the capture thread's: a converted frame's luma, the ids' count, the times' offset
  std::vector<uint8_t> conv_;
  uint64_t seqBase_ = 0, seqLast_ = 0, lastId_ = 0, failedSince_ = 0;
  bool haveId_ = false, based_ = false;
  int64_t off_ = 0, lastNs_ = 0;
  std::atomic<uint64_t> frames_{0}, dropped_{0}, corrupt_{0}, convNs_{0};
  std::atomic<double> agreedFps_{0};
  arv::gulong lostSignal_ = 0;
  bool selector_ = false;   // ai: the camera has a TriggerSelector (FrameStart's trigger chosen through it)
  // ai: the lock's: why the camera cannot take a trigger ("" it can), the session's capture feed (set before a start)
  std::mutex barMu_;
  std::string bar_;
  Captures each_;
  // ai: the capture thread's: the feed this stream, the switch, the schedule's period, its counts and moments
  Captures feed_;
  std::atomic<Trig> trig_{Trig::Off};
  int64_t periodNs_ = 0, startAt_ = 0, armedAt_ = 0, lastAt_ = 0, probeAt_ = 0, prevDue_ = 0;
  uint64_t freeSeen_ = 0;
  int probes_ = 0;
  bool probeGot_ = false, fresh_ = false;
  std::atomic<int64_t> expoNs_{0}, beganNs_{0};   // ai: the exposure as last read or set; start to the schedule's begin
  std::atomic<uint64_t> matched_{0}, unmatched_{0};
  TriggerClock clock_{[this](std::string& why) { return fire(why); }};   // ai: last: its thread ends before the rest go
};

AravisCam::AravisCam(const arv::Api& a, arv::Camera* cam, Info info) : a_(a), cam_(cam), info_(std::move(info)) {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  auto has = [&](const char* feature) {
    const bool r = a_.arv_camera_is_feature_available(cam_, feature, e.p()) != 0;
    e.clear();
    return r;
  };
  auto yes = [&](arv::gboolean v) { e.clear(); return v != 0; };
  gv_ = a_.arv_camera_is_gv_device(cam_) != 0;
  rate_ = yes(a_.arv_camera_is_frame_rate_available(cam_, e.p()));
  if (has("AcquisitionFrameRateEnable")) enable_ = "AcquisitionFrameRateEnable";
  else if (has("AcquisitionFrameRateEnabled")) enable_ = "AcquisitionFrameRateEnabled";
  expo_ = yes(a_.arv_camera_is_exposure_time_available(cam_, e.p()));
  expoAuto_ = yes(a_.arv_camera_is_exposure_auto_available(cam_, e.p()));
  gain_ = yes(a_.arv_camera_is_gain_available(cam_, e.p()));
  gainAuto_ = yes(a_.arv_camera_is_gain_auto_available(cam_, e.p()));
  gainUnit_ = has("Gain") ? "dB" : "";
  if (expo_) { expoDef_ = a_.arv_camera_get_exposure_time(cam_, e.p()); e.clear(); }
  if (gain_) { gainDef_ = a_.arv_camera_get_gain(cam_, e.p()); e.clear(); }
  selector_ = has("TriggerSelector");
  bar_ = triggerBar();
  dev_ = a_.arv_camera_get_device(cam_);
  // ai: emitted on Aravis's heartbeat (GigE) or libusb's thread (USB3): a flag the capture thread reads
  if (dev_)
    lostSignal_ = a_.g_signal_connect_data(dev_, "control-lost", reinterpret_cast<arv::GCallback>(&AravisCam::onLost), this,
                                           nullptr, 0);
}

// ai: mu_ held, at open: why this camera cannot take the lock's software trigger, "" where it can (the header's
// ai: test, TriggerSource's write tried and put back with the selector)
std::string AravisCam::triggerBar() {
  Err e(a_);
  auto has = [&](const char* feature) {
    const bool r = a_.arv_camera_is_feature_available(cam_, feature, e.p()) != 0;
    e.clear();
    return r;
  };
  auto lists = [&](const char** v, arv::guint n, const char* want) {
    bool in = false;
    for (arv::guint i = 0; v && i < n; i++) in = in || (v[i] && std::string(v[i]) == want);
    a_.g_free(static_cast<void*>(v));
    return in;
  };
  if (!has("TriggerSoftware")) return "the camera has no software trigger (TriggerSoftware)";
  if (!has("TriggerMode")) return "the camera has no TriggerMode";
  if (!has("TriggerSource")) return "the camera has no TriggerSource";
  std::string sel0;
  if (selector_) {
    arv::guint n = 0;
    const char** ts = a_.arv_camera_dup_available_triggers(cam_, &n, e.p());   // ai: before n is read
    e.clear();
    if (!lists(ts, n, "FrameStart")) return "the camera has no FrameStart trigger";
    sel0 = str(a_.arv_camera_get_string(cam_, "TriggerSelector", e.p()));
    e.clear();
    a_.arv_camera_set_string(cam_, "TriggerSelector", "FrameStart", e.p());
    if (e) return "TriggerSelector FrameStart: " + e.take();
  }
  std::string why;
  arv::guint n = 0;
  const char** srcs = a_.arv_camera_dup_available_trigger_sources(cam_, &n, e.p());
  e.clear();
  if (!lists(srcs, n, "Software")) {
    why = "FrameStart's TriggerSource offers no Software";
  } else {
    const std::string src0 = str(a_.arv_camera_get_string(cam_, "TriggerSource", e.p()));
    e.clear();
    a_.arv_camera_set_string(cam_, "TriggerSource", "Software", e.p());
    if (e) why = "TriggerSource Software: " + e.take();
    else if (!src0.empty() && src0 != "Software") a_.arv_camera_set_string(cam_, "TriggerSource", src0.c_str(), e.p());
    e.clear();
  }
  if (selector_ && !sel0.empty() && sel0 != "FrameStart") { a_.arv_camera_set_string(cam_, "TriggerSelector", sel0.c_str(), e.p()); e.clear(); }
  return why;
}

std::string AravisCam::delayBar() {
  std::lock_guard<std::mutex> l(barMu_);
  return bar_;
}

void AravisCam::captures(Captures each, Fate fate) {
  (void)fate;   // ai: never told: the schedule moves wherever a delay was asked (the header)
  std::lock_guard<std::mutex> l(barMu_);
  each_ = std::move(each);
}

bool AravisCam::delayOnce(int64_t us) {
  if (us <= 0 || trig_.load() != Trig::On || !clock_.running()) return false;
  clock_.delay(us * 1000);
  return true;
}

// ai: the clock's thread: one trigger, under the camera's lock
bool AravisCam::fire(std::string& why) {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  a_.arv_camera_software_trigger(cam_, e.p());
  if (e) { why = e.take(); return false; }
  return true;
}

// ai: The capture thread's: FrameStart's trigger on Software (what it was kept, put back at stop), the rate limit's
// ai: switch off; false with why where the camera refused (what it took stays kept)
bool AravisCam::arm(std::string& why) {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  const bool sel = selector_;
  auto choose = [this, sel](Err& x) { if (sel) a_.arv_camera_set_string(cam_, "TriggerSelector", "FrameStart", x.p()); };
  choose(e);
  if (e) { why = "TriggerSelector FrameStart: " + e.take(); return false; }
  for (const char* f : {"TriggerSource", "TriggerMode"}) {
    const std::string key = std::string(f) + "@FrameStart";
    if (kept_.count(key)) continue;
    const std::string was = str(a_.arv_camera_get_string(cam_, f, e.p()));
    if (!e && !was.empty())
      keep(key, [this, f, was, choose] { Err x(a_); choose(x); x.clear(); a_.arv_camera_set_string(cam_, f, was.c_str(), x.p()); });
    e.clear();
  }
  a_.arv_camera_set_string(cam_, "TriggerSource", "Software", e.p());
  if (e) { why = "TriggerSource Software: " + e.take(); return false; }
  a_.arv_camera_set_string(cam_, "TriggerMode", "On", e.p());
  if (e) {
    // ai: a camera that refuses TriggerMode while it acquires: stopped around the change, the stream kept
    e.clear();
    Err s(a_);
    a_.arv_camera_stop_acquisition(cam_, s.p());
    a_.arv_camera_set_string(cam_, "TriggerMode", "On", e.p());
    s.clear();
    a_.arv_camera_start_acquisition(cam_, s.p());
    if (e) { why = "TriggerMode On: " + e.take(); return false; }
    if (s) { why = "AcquisitionStart after TriggerMode On: " + s.take(); return false; }
  }
  if (!enable_.empty()) {
    if (!kept_.count(enable_)) {
      const bool on = a_.arv_camera_get_boolean(cam_, enable_.c_str(), e.p()) != 0;
      if (!e) keep(enable_, [this, on] { Err x(a_); a_.arv_camera_set_boolean(cam_, enable_.c_str(), on, x.p()); });
      e.clear();
    }
    a_.arv_camera_set_boolean(cam_, enable_.c_str(), 0, e.p());
    e.clear();
  }
  return true;
}

// ai: the capture thread's: back to free run at the mode's rate (the trigger off, the rate limit's switch on)
void AravisCam::disarm() {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  if (selector_) { a_.arv_camera_set_string(cam_, "TriggerSelector", "FrameStart", e.p()); e.clear(); }
  a_.arv_camera_set_string(cam_, "TriggerMode", "Off", e.p());
  e.clear();
  if (!enable_.empty()) { a_.arv_camera_set_boolean(cam_, enable_.c_str(), 1, e.p()); e.clear(); }
}

// ai: the capture thread's: no trigger this stream, and why from the next start on (delayBar)
void AravisCam::giveUp(const std::string& why) {
  trig_ = Trig::Off;
  std::lock_guard<std::mutex> l(barMu_);
  bar_ = why;
}

// ai: The capture thread's, after each pop (the header's switch): at at, steadyNs
void AravisCam::step(int64_t at) {
  std::string why;
  switch (trig_.load()) {
    case Trig::Free:
      if (freeSeen_ < kWarm) break;
      if (!arm(why)) { giveUp("the software trigger: " + why); break; }
      armedAt_ = at;
      trig_ = Trig::Drain;
      break;
    case Trig::Drain:
      if (at - armedAt_ >= kSecondNs) {
        // ai: frames still coming a second on: the camera runs free whatever its TriggerMode says
        disarm();
        giveUp("the camera ran on free with its trigger on");
        break;
      }
      if (at - std::max(armedAt_, lastAt_) < kQuietNs) break;
      probes_ = 0;
      probeGot_ = false;
      [[fallthrough]];
    case Trig::Probe:
      if (probeGot_) {
        clock_.begin(periodNs_);
        fresh_ = true;
        prevDue_ = 0;
        agreedFps_ = 1e9 / double(periodNs_);
        beganNs_ = at - startAt_;
        trig_ = Trig::On;
        break;
      }
      if (trig_.load() == Trig::Probe && at - probeAt_ < kProbeNs) break;
      if (probes_ >= kProbes) {
        disarm();
        giveUp("no frame came back for a software trigger (" + std::to_string(kProbes) + " tries)");
        break;
      }
      probes_++;
      probeAt_ = at;
      trig_ = Trig::Probe;
      if (!fire(why)) {
        disarm();
        giveUp("the software trigger: " + why);
      }
      break;
    case Trig::Off:
    case Trig::On:
      break;
  }
}

std::string AravisCam::report() {
  const Trig t = trig_.load();
  std::string bar;
  {
    std::lock_guard<std::mutex> l(barMu_);
    bar = bar_;
  }
  std::string q;   // ai: the bar as a JSON string
  for (char c : bar) {
    if (c == '"' || c == '\\') q += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) q += c;
  }
  const char* state = t == Trig::On ? "triggered" : t == Trig::Off ? "free" : "arming";
  char b[160];
  std::snprintf(b, sizeof b, ",\"matched\":%llu,\"unmatched\":%llu,\"beganMs\":%lld}", (unsigned long long)matched_.load(),
                (unsigned long long)unmatched_.load(), (long long)(beganNs_.load() / 1'000'000));
  return std::string("{\"trigger\":\"") + state + "\",\"bar\":\"" + q + "\",\"schedule\":" +
         (t == Trig::On ? clock_.stats() : std::string("null")) + b;
}

AravisCam::~AravisCam() {
  stop();
  if (dev_ && lostSignal_) a_.g_signal_handler_disconnect(dev_, lostSignal_);
  a_.g_object_unref(cam_);   // ai: the device closed, its heartbeat thread joined
}

void AravisCam::onLost(void*, void* self) { static_cast<AravisCam*>(self)->lost_ = true; }

std::vector<Mode> AravisCam::modes() {
  std::lock_guard<std::mutex> l(mu_);
  if (stream_) return modes_;   // ai: the format and region are the stream's while it runs
  modes_ = readModes();
  return modes_;
}

// ai: mu_ held, the camera not streaming
std::vector<Mode> AravisCam::readModes() {
  Err e(a_);
  arv::guint n = 0, ns = 0;
  arv::gint64* values = a_.arv_camera_dup_available_pixel_formats(cam_, &n, e.p());
  if (e || !values) { e.clear(); a_.g_free(values); return {}; }
  const char** names = a_.arv_camera_dup_available_pixel_formats_as_strings(cam_, &ns, e.p());
  e.clear();
  // ai: the camera's own format and region, put back after
  const arv::PixelFormat pf0 = a_.arv_camera_get_pixel_format(cam_, e.p());
  const bool havePf = !e;
  e.clear();
  int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
  a_.arv_camera_get_region(cam_, &x0, &y0, &w0, &h0, e.p());
  const bool haveRegion = !e;
  e.clear();
  struct Got { uint8_t rank; Mode m; };
  std::vector<Got> got;
  for (arv::guint i = 0; i < n; i++) {
    const auto pf = arv::PixelFormat(values[i]);
    const Format* f = formatOf(pf);
    if (!f) continue;
    a_.arv_camera_set_pixel_format(cam_, pf, e.p());
    if (e) { e.clear(); continue; }
    // ai: the offsets at 0 first (a width's top is the sensor's less the offset on most cameras), then the largest
    a_.arv_camera_set_region(cam_, 0, 0, 0, 0, e.p());
    e.clear();
    int wlo = 0, whi = 0, hlo = 0, hhi = 0;
    a_.arv_camera_get_width_bounds(cam_, &wlo, &whi, e.p());
    if (e) { e.clear(); continue; }
    a_.arv_camera_get_height_bounds(cam_, &hlo, &hhi, e.p());
    if (e || whi <= 0 || hhi <= 0) { e.clear(); continue; }
    a_.arv_camera_set_region(cam_, 0, 0, whi, hhi, e.p());
    if (e) { e.clear(); continue; }
    double lo = 0, hi = 0;
    if (rate_) {
      a_.arv_camera_get_frame_rate_bounds(cam_, &lo, &hi, e.p());
      if (e || !(hi > 0)) { lo = hi = 0; e.clear(); }
    }
    Mode m;
    m.width = uint32_t(whi);
    m.height = uint32_t(hhi);
    m.format = names && ns == n && names[i] ? names[i] : f->name;
    m.pix = f->pix;
    m.bits = f->bits;
    m.key = pf;
    std::vector<double> rates;
    if (hi > 0) {
      for (double r : kRates)
        if (r >= lo && r <= hi) rates.push_back(r);
      if (rates.empty() || hi > rates.front() * 1.005) rates.insert(rates.begin(), hi);
    } else {
      rates.push_back(0);
    }
    for (double r : rates) {
      m.fps = m.fpsMin = r;
      got.push_back({f->rank, m});
    }
  }
  a_.g_free(values);
  a_.g_free(static_cast<void*>(names));
  if (havePf) { a_.arv_camera_set_pixel_format(cam_, pf0, e.p()); e.clear(); }
  if (haveRegion) { a_.arv_camera_set_region(cam_, x0, y0, w0, h0, e.p()); e.clear(); }
  std::stable_sort(got.begin(), got.end(), [](const Got& p, const Got& q) { return p.rank < q.rank; });
  std::vector<Mode> out;
  for (auto& g : got) out.push_back(std::move(g.m));
  return out;
}

std::vector<Control> AravisCam::controls() {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  std::vector<Control> out;
  auto isOn = [&](const char* feature) {
    const std::string v = str(a_.arv_camera_get_string(cam_, feature, e.p()));
    const bool bad = bool(e);
    e.clear();
    return !bad && !v.empty() && v != "Off";
  };
  if (expo_) {
    Control c;
    c.id = "exposure";
    c.name = "Exposure";
    c.unit = "us";
    a_.arv_camera_get_exposure_time_bounds(cam_, &c.min, &c.max, e.p());
    e.clear();
    c.value = a_.arv_camera_get_exposure_time(cam_, e.p());
    e.clear();
    c.def = expoDef_;
    c.canAuto = expoAuto_;
    c.isAuto = expoAuto_ && isOn("ExposureAuto");
    out.push_back(c);
  }
  if (gain_) {
    Control c;
    c.id = "gain";
    c.name = "Gain";
    c.unit = gainUnit_;
    a_.arv_camera_get_gain_bounds(cam_, &c.min, &c.max, e.p());
    e.clear();
    c.step = gainUnit_.empty() ? 1 : 0;   // ai: GainRaw counts in whole steps
    c.value = a_.arv_camera_get_gain(cam_, e.p());
    e.clear();
    c.def = gainDef_;
    c.canAuto = gainAuto_;
    c.isAuto = gainAuto_ && isOn("GainAuto");
    out.push_back(c);
  }
  return out;
}

// ai: mu_ held. back puts `key` back as it is now; kept once, at its first change, so a stop puts back the value from
// ai: before the first
void AravisCam::keep(const std::string& key, std::function<void()> back) {
  if (!kept_.insert(key).second) return;
  undo_.push_back(std::move(back));
}

// ai: mu_ held: an enumeration or string feature as it is now (unread: nothing to put back)
void AravisCam::keepString(const std::string& feature) {
  if (kept_.count(feature)) return;
  Err e(a_);
  const char* v = a_.arv_camera_get_string(cam_, feature.c_str(), e.p());
  if (e || !v) return;
  const std::string was = v;
  keep(feature, [this, feature, was] { Err x(a_); a_.arv_camera_set_string(cam_, feature.c_str(), was.c_str(), x.p()); });
}

// ai: mu_ held: every trigger's TriggerMode where it is not Off (each under its TriggerSelector), and the selector
// ai: itself; put back after the rate (whose set turns them all off), as start keeps them first
void AravisCam::keepTriggers() {
  if (kept_.count("triggers")) return;
  Err e(a_);
  const bool selector = a_.arv_camera_is_feature_available(cam_, "TriggerSelector", e.p()) != 0;
  e.clear();
  if (!selector) { keepString("TriggerMode"); kept_.insert("triggers"); return; }
  keepString("TriggerSelector");
  arv::guint n = 0;
  const char** ts = a_.arv_camera_dup_available_triggers(cam_, &n, e.p());
  e.clear();
  std::vector<std::string> on;
  for (arv::guint i = 0; ts && i < n; i++) {
    if (!ts[i]) continue;
    a_.arv_camera_set_string(cam_, "TriggerSelector", ts[i], e.p());
    if (e) { e.clear(); continue; }
    const std::string mode = str(a_.arv_camera_get_string(cam_, "TriggerMode", e.p()));
    if (!e && !mode.empty() && mode != "Off") on.push_back(ts[i]);
    e.clear();
  }
  a_.g_free(static_cast<void*>(ts));
  keep("triggers", [this, on] {
    Err x(a_);
    for (const auto& t : on) {
      a_.arv_camera_set_string(cam_, "TriggerSelector", t.c_str(), x.p());
      if (!x) a_.arv_camera_set_string(cam_, "TriggerMode", "On", x.p());
    }
  });
}

// ai: mu_ held: undo_ run, newest first
void AravisCam::putBack() {
  for (auto it = undo_.rbegin(); it != undo_.rend(); ++it) (*it)();
  undo_.clear();
  kept_.clear();
}

bool AravisCam::set(const std::string& id, double value, bool automatic, std::string& why) {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  if (id == "exposure" || id == "gain") {
    const bool ex = id == "exposure";
    if (!(ex ? expo_ : gain_)) { why = "this camera has no " + id + " control"; return false; }
    const bool can = ex ? expoAuto_ : gainAuto_;
    if (automatic && !can) { why = "this camera has no automatic " + id; return false; }
    auto setAuto = ex ? a_.arv_camera_set_exposure_time_auto : a_.arv_camera_set_gain_auto;
    if (can) keepString(ex ? "ExposureAuto" : "GainAuto");
    if (automatic) {
      setAuto(cam_, arv::AUTO_CONTINUOUS, e.p());
    } else {
      if (can) setAuto(cam_, arv::AUTO_OFF, e.p());
      if (!e) {
        if (!kept_.count(id)) {
          Err r(a_);
          const double was = (ex ? a_.arv_camera_get_exposure_time : a_.arv_camera_get_gain)(cam_, r.p());
          if (!r) keep(id, [this, ex, was] { Err x(a_); (ex ? a_.arv_camera_set_exposure_time : a_.arv_camera_set_gain)(cam_, was, x.p()); });
        }
        (ex ? a_.arv_camera_set_exposure_time : a_.arv_camera_set_gain)(cam_, value, e.p());
        if (ex && !e) expoNs_ = int64_t(value * 1000);
      }
    }
    if (e) { why = id + ": " + e.take(); return false; }
    return true;
  }
  why = "this camera has no control " + id;
  return false;
}

// ai: a stream that ended by itself (its thread done, its stream still held) let go, nothing put back: that is stop's
void AravisCam::reap() {
  clock_.end();
  if (th_.joinable()) th_.join();
  std::lock_guard<std::mutex> l(mu_);
  if (stream_) { a_.g_object_unref(stream_); stream_ = nullptr; }
}

bool AravisCam::start(const Mode& mode, Sink sink, Ended ended, std::string& why) {
  if (run_.load()) { why = "the camera is already streaming"; return false; }
  reap();
  const Format* f = formatOf(arv::PixelFormat(mode.key));
  if (!f) { why = mode.format + " (" + hex(mode.key) + ") is not a pixel format the receiver reads"; return false; }
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  a_.arv_camera_stop_acquisition(cam_, e.p());
  e.clear();
  if (!kept_.count("PixelFormat")) {
    const arv::PixelFormat was = a_.arv_camera_get_pixel_format(cam_, e.p());
    if (!e) keep("PixelFormat", [this, was] { Err x(a_); a_.arv_camera_set_pixel_format(cam_, was, x.p()); });
    e.clear();
  }
  a_.arv_camera_set_pixel_format(cam_, f->pf, e.p());
  if (e) { why = "the camera refused " + mode.format + ": " + e.take(); return false; }
  if (!kept_.count("region")) {
    int x = 0, y = 0, w = 0, h = 0;
    a_.arv_camera_get_region(cam_, &x, &y, &w, &h, e.p());
    if (!e) keep("region", [this, x, y, w, h] { Err r(a_); a_.arv_camera_set_region(cam_, x, y, w, h, r.p()); });
    e.clear();
  }
  a_.arv_camera_set_region(cam_, 0, 0, int(mode.width), int(mode.height), e.p());
  if (e) { why = "the camera refused " + std::to_string(mode.width) + " x " + std::to_string(mode.height) + ": " + e.take(); return false; }
  // ai: free run: every trigger off, Continuous acquisition, the mode's rate
  keepTriggers();
  a_.arv_camera_clear_triggers(cam_, e.p());
  e.clear();
  keepString("AcquisitionMode");
  a_.arv_camera_set_string(cam_, "AcquisitionMode", "Continuous", e.p());
  e.clear();
  agreedFps_ = 0;
  if (rate_ && mode.fps > 0) {
    if (!enable_.empty() && !kept_.count(enable_)) {
      const bool on = a_.arv_camera_get_boolean(cam_, enable_.c_str(), e.p()) != 0;
      if (!e) keep(enable_, [this, on] { Err x(a_); a_.arv_camera_set_boolean(cam_, enable_.c_str(), on, x.p()); });
      e.clear();
    }
    if (!kept_.count("rate")) {
      const double was = a_.arv_camera_get_frame_rate(cam_, e.p());
      if (!e && was > 0) keep("rate", [this, was] { Err x(a_); a_.arv_camera_set_frame_rate(cam_, was, x.p()); });
      e.clear();
    }
    a_.arv_camera_set_frame_rate(cam_, mode.fps, e.p());
    if (e) {
      char r[32];
      std::snprintf(r, sizeof r, "%.2f", mode.fps);
      why = std::string("the camera refused ") + r + " fps: " + e.take();
      return false;
    }
    const double v = a_.arv_camera_get_frame_rate(cam_, e.p());
    agreedFps_ = e ? 0.0 : v;
    e.clear();
  }
  if (gv_) { a_.arv_camera_gv_auto_packet_size(cam_, e.p()); e.clear(); }
  const arv::guint payload = a_.arv_camera_get_payload(cam_, e.p());
  if (e || payload == 0) { why = "PayloadSize: " + (e ? e.take() : std::string("0")); return false; }
  stream_ = a_.arv_camera_create_stream(cam_, nullptr, nullptr, e.p());
  if (!stream_) { why = e ? e.take() : std::string("the camera gave no stream"); return false; }
  e.clear();
  if (gv_)
    a_.g_object_set(stream_, "socket-buffer", int(arv::SOCKET_BUFFER_AUTO), "socket-buffer-size", 0,
                    static_cast<const char*>(nullptr));
  for (int i = 0; i < kBuffers; i++) a_.arv_stream_push_buffer(stream_, a_.arv_buffer_new_allocate(payload));
  a_.arv_camera_start_acquisition(cam_, e.p());
  if (e) {
    why = "AcquisitionStart: " + e.take();
    a_.g_object_unref(stream_);
    stream_ = nullptr;
    return false;
  }
  if (expo_) {
    const double us = a_.arv_camera_get_exposure_time(cam_, e.p());
    expoNs_ = e ? 0 : int64_t(us * 1000);
    e.clear();
  }
  frames_ = 0;
  dropped_ = 0;
  corrupt_ = 0;
  convNs_ = 0;
  seqBase_ = seqLast_ = lastId_ = failedSince_ = 0;
  haveId_ = based_ = false;
  off_ = lastNs_ = 0;
  // ai: the lock's trigger this stream: where the camera takes one, the mode names a rate and the session feeds a lock
  bool can = false;
  {
    std::lock_guard<std::mutex> b(barMu_);
    feed_ = each_;
    can = bar_.empty();
  }
  periodNs_ = mode.fps > 0 ? int64_t(1e9 / mode.fps * (1 - kFast)) : 0;
  trig_ = can && periodNs_ > 0 && feed_ ? Trig::Free : Trig::Off;
  freeSeen_ = 0;
  startAt_ = steadyNs();
  armedAt_ = lastAt_ = probeAt_ = prevDue_ = 0;
  probes_ = 0;
  probeGot_ = fresh_ = false;
  beganNs_ = 0;
  matched_ = unmatched_ = 0;
  sink_ = std::move(sink);
  ended_ = std::move(ended);
  lost_ = false;
  run_ = true;
  th_ = std::thread([this] { sys::raiseThread(); loop(); });
  return true;
}

void AravisCam::stop() {
  clock_.end();   // ai: no trigger after this (a fire takes mu_, which nothing here holds yet)
  if (th_.joinable() && th_.get_id() == std::this_thread::get_id()) {
    // ai: called from the sink or ended: the thread ends once that returns; the next stop, or the close, frees
    run_ = false;
    return;
  }
  {
    std::lock_guard<std::mutex> l(mu_);
    if (stream_) { Err e(a_); a_.arv_camera_stop_acquisition(cam_, e.p()); }
  }
  run_ = false;
  if (th_.joinable()) th_.join();
  std::lock_guard<std::mutex> l(mu_);
  if (stream_) {
    a_.g_object_unref(stream_);
    stream_ = nullptr;
  }
  putBack();
  sink_ = nullptr;
  ended_ = nullptr;
  feed_ = nullptr;
  trig_ = Trig::Off;
}

Stats AravisCam::stats() {
  Stats s;
  s.frames = frames_;
  s.dropped = dropped_;
  s.corrupt = corrupt_;
  s.convertMs = s.frames ? double(convNs_) / double(s.frames) / 1e6 : 0;
  s.fps = agreedFps_;
  return s;
}

bool AravisCam::probe(std::string& why) {
  std::lock_guard<std::mutex> l(mu_);
  Err e(a_);
  a_.arv_camera_get_payload(cam_, e.p());
  if (e) { why = e.take(); return false; }
  return true;
}

void AravisCam::finish(const std::string& why) {
  run_ = false;
  if (ended_) ended_(why);
}

// ai: GigE Vision's 16-bit block ids run 1 to 65535, then wrap to 1: counted on as one count
uint64_t AravisCam::unwrap(uint64_t id) {
  if (id <= 0xFFFF && seqLast_ <= 0xFFFF && id + 0x8000 < seqLast_) seqBase_ += 0xFFFF;
  seqLast_ = id;
  return seqBase_ + id;
}

void AravisCam::loop() {
  int64_t heard = steadyNs(), probed = heard;
  while (run_.load()) {
    if (lost_.load()) { finish("the camera's control was lost (unplugged, or another host took it)"); return; }
    // ai: the switch's waits (a quiet, a probe's frame) looked at every 10 ms
    const Trig t = trig_.load();
    const bool waiting = t == Trig::Drain || t == Trig::Probe;
    arv::Buffer* b = a_.arv_stream_timeout_pop_buffer(stream_, waiting ? kPopUs / 10 : kPopUs);
    const int64_t at = steadyNs();
    if (b) {
      heard = at;
      take(b, at);
      a_.arv_stream_push_buffer(stream_, b);
    } else if (!waiting && at - heard >= kSilentNs && at - probed >= kSilentNs) {
      probed = at;
      std::string why;
      if (!probe(why)) { finish("the camera stopped answering: " + why); return; }
    }
    step(steadyNs());
  }
}

void AravisCam::take(arv::Buffer* b, int64_t at) {
  const int status = a_.arv_buffer_get_status(b);
  if (status != arv::BUFFER_SUCCESS) {
    // ai: the buffer's part fields read only on success (Aravis's accessors refuse the others)
    (status == arv::BUFFER_MISSING_PACKETS ? dropped_ : corrupt_)++;
    failedSince_++;
    return;
  }
  // ai: the frames between this one and the last handed over that no failed buffer held, lost
  const uint64_t id = unwrap(a_.arv_buffer_get_frame_id(b));
  if (haveId_ && id > lastId_ + 1 && id - lastId_ - 1 > failedSince_) dropped_ += id - lastId_ - 1 - failedSince_;
  haveId_ = true;
  lastId_ = id;
  failedSince_ = 0;
  size_t size = 0;
  const auto* data = static_cast<const uint8_t*>(a_.arv_buffer_get_image_data(b, &size));
  const int w = a_.arv_buffer_get_image_width(b), h = a_.arv_buffer_get_image_height(b);
  const arv::PixelFormat pf = a_.arv_buffer_get_image_pixel_format(b);
  int xpad = 0, ypad = 0;
  a_.arv_buffer_get_image_padding(b, &xpad, &ypad);
  const Format* f = formatOf(pf);
  if (!data || w <= 0 || h <= 0 || !f) { corrupt_++; return; }
  Frame fr;
  fr.width = uint32_t(w);
  fr.height = uint32_t(h);
  // ai: a row: the width's bits, whole bytes, then the transport's padding
  const uint32_t stride = uint32_t((uint64_t(fr.width) * arv::bitsPerPixel(pf) + 7) / 8) + uint32_t(std::max(xpad, 0));
  if (lumaInPlace(f->pix)) {
    if (size < frameBytes(f->pix, fr.width, fr.height, stride)) { corrupt_++; return; }
    fr.luma = data;
    fr.stride = stride;
  } else {
    const int64_t t0 = steadyNs();
    if (!toLuma(f->pix, f->bits, data, size, fr.width, fr.height, stride, conv_, nullptr)) { corrupt_++; return; }
    convNs_ += uint64_t(steadyNs() - t0);
    fr.luma = conv_.data();
    fr.stride = fr.width;
  }
  // ai: a triggered frame's time its trigger's deadline; else the device's time on steadyNs's clock: the offset at the
  // ai: first frame, based again on the arrival where the device's clock went back; never before the frame before
  const Trig trig = trig_.load();
  lastAt_ = at;
  if (trig == Trig::Free) freeSeen_++;
  if (trig == Trig::Probe) probeGot_ = true;
  int64_t due = 0;
  bool delayed = false;
  const bool matched = trig == Trig::On && clock_.match(at, due, delayed);
  const uint64_t ts = a_.arv_buffer_get_timestamp(b);
  if (matched) {
    fr.ns = std::max(due, lastNs_ + 1);
  } else if (!ts) {
    fr.ns = std::max(at, lastNs_ + 1);
  } else {
    const int64_t t = int64_t(ts);
    if (!based_) { off_ = at - t; based_ = true; }
    fr.ns = t + off_;
    if (lastNs_ && fr.ns <= lastNs_) {
      off_ = at - t;
      fr.ns = std::max(t + off_, lastNs_ + 1);
    }
  }
  lastNs_ = fr.ns;
  fr.seq = id;
  sink_(fr);
  frames_++;
  if (trig != Trig::On) return;
  if (!matched) { unmatched_++; return; }
  matched_++;
  Capture c;
  c.ns = fr.ns;
  c.delayed = delayed;
  c.fresh = fresh_;
  c.frameNs = delayed && prevDue_ ? fr.ns - prevDue_ : periodNs_;
  c.exposureNs = expoNs_.load();
  fresh_ = false;
  prevDue_ = fr.ns;
  if (feed_) feed_(c);
}

// ai: the listing's words for device i, copied at once (the next discovery frees them); systemMu held
Info infoAt(const arv::Api& a, unsigned i) {
  Info info;
  const std::string id = str(a.arv_get_device_id(i));
  const std::string vendor = str(a.arv_get_device_vendor(i)), model = str(a.arv_get_device_model(i));
  const std::string serial = str(a.arv_get_device_serial_nbr(i)), protocol = str(a.arv_get_device_protocol(i));
  const std::string address = str(a.arv_get_device_address(i));
  info.id = "aravis:" + id;
  info.name = vendor.empty() ? model : model.empty() ? vendor : vendor + " " + model;
  if (info.name.empty()) info.name = id;
  info.backend = "aravis";
  const std::string where = protocol == "GigEVision" ? address : serial;
  info.detail = protocol.empty() ? where : where.empty() ? protocol : protocol + " " + where;
  return info;
}

// ai: The discoveries (the header's rule): the last one's cameras, begun on a thread of their own. Never destroyed,
// ai: since a discovery's thread may run as the process exits.
struct Listing {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<Info> cams;
  int64_t doneAt = 0;   // ai: steadyNs when the last discovery ended, 0 none yet
  bool running = false;
  static constexpr int64_t kFreshNs = 2 * kSecondNs;

  // ai: mu held: a discovery begun where none runs and the last ended more than kFreshNs ago
  void ask(const arv::Api& a) {
    if (running || (doneAt && steadyNs() - doneAt < kFreshNs)) return;
    running = true;
    std::thread([this, &a] {
      std::vector<Info> got;
      {
        std::lock_guard<std::mutex> s(systemMu());
        a.arv_update_device_list();
        for (unsigned i = 0, n = a.arv_get_n_devices(); i < n; i++)
          if (a.arv_get_device_id(i)) got.push_back(infoAt(a, i));
      }
      std::lock_guard<std::mutex> l(mu);
      cams = std::move(got);
      doneAt = steadyNs();
      running = false;
      cv.notify_all();
    }).detach();
  }
};

Listing& listing() {
  static Listing* l = new Listing;
  return *l;
}

}  // namespace

const arv::Api* aravisApi(std::string* why) {
  static const Loaded l = load();
  if (!l.ok) {
    if (why) *why = l.why;
    return nullptr;
  }
  return &l.api;
}

std::vector<Info> aravisList(std::vector<std::string>* notes) {
  std::string why;
  const arv::Api* a = aravisApi(&why);
  if (!a) {
    if (notes && !why.empty()) notes->push_back("aravis: " + why);
    return {};
  }
  Listing& l = listing();
  std::unique_lock<std::mutex> k(l.mu);
  l.ask(*a);
  if (!l.doneAt && !l.cv.wait_for(k, std::chrono::seconds(3), [&] { return l.doneAt != 0; }) && notes)
    notes->push_back("aravis: the first discovery took over 3 s");
  return l.cams;
}

std::string aravisStamp() {
  const arv::Api* a = aravisApi(nullptr);
  if (!a) return "";
  Listing& l = listing();
  std::lock_guard<std::mutex> k(l.mu);
  l.ask(*a);
  std::string s;
  for (const auto& i : l.cams) s += i.id + ";";
  return s;
}

std::unique_ptr<Camera> aravisOpen(const std::string& id, std::string& why) {
  const arv::Api* a = aravisApi(&why);
  if (!a) {
    if (why.empty()) why = "Aravis is off (LIZ_ARAVIS is set and empty)";
    return nullptr;
  }
  const std::string dev = id.rfind("aravis:", 0) == 0 ? id.substr(7) : id;
  arv::Camera* cam = nullptr;
  Info info;
  {
    std::lock_guard<std::mutex> l(systemMu());
    Err e(*a);
    cam = a->arv_camera_new(dev.c_str(), e.p());
    if (!cam) { why = e ? e.take() : "no camera " + id; return nullptr; }
    for (unsigned i = 0, n = a->arv_get_n_devices(); i < n; i++)
      if (dev == str(a->arv_get_device_id(i))) { info = infoAt(*a, i); break; }
  }
  if (info.id.empty()) {
    // ai: an id the last discovery did not hold (Aravis found it by its own search): the camera's own words
    Err e(*a);
    const std::string vendor = str(a->arv_camera_get_vendor_name(cam, e.p()));
    const std::string model = str(a->arv_camera_get_model_name(cam, e.p()));
    const std::string serial = str(a->arv_camera_get_device_serial_number(cam, e.p()));
    e.clear();
    info.id = "aravis:" + dev;
    info.name = vendor.empty() ? model : model.empty() ? vendor : vendor + " " + model;
    if (info.name.empty()) info.name = dev;
    info.backend = "aravis";
    const std::string protocol = a->arv_camera_is_gv_device(cam) ? "GigEVision" : a->arv_camera_is_uv_device(cam) ? "USB3Vision" : "";
    info.detail = protocol.empty() ? serial : serial.empty() ? protocol : protocol + " " + serial;
  }
  return std::make_unique<AravisCam>(*a, cam, std::move(info));
}

}  // namespace lizard::cam
