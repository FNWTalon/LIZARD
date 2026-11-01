// ai: The receive session (session.h, 2026-10-31; the desktop's RxSession of 2026-10-25 grown). One worker thread runs
// ai: the commands in order and the rules between them, so the camera is driven by that thread alone (open, start,
// ai: stop, set, close: a UVC device's open or start can take seconds) and no caller waits on it. The capture thread
// ai: is the sink's: each frame pushed (the receiver copies its crop and releases it inside push, so the release callback
// ai: has nothing to do) and its crop copied for the preview while one is asked. Locks, leaves all: rxMu around the
// ai: receiver's pointer (the sink and every reader copy it and call outside the lock; reconfigure swaps it; batchCap
// ai: and the replay set the receiver's under it); ptrMu around the camera's pointer and what was read off it (stats and
// ai: camera() copy it); phMu the phase; qMu the queue; and leaf locks for the preview, the measured rate, the camera's
// ai: counts, the kept counters, the replays' list, the platform's fields. Only rxMu is held while a receiver's own is.
// ai: The user's settings are kept here and set again after each start: every backend puts back at stop what set
// ai: changed, so a zoom, exposure or gain would otherwise last one stream (2026-10-25).
// ai: A camera's own buffer (Android's zero copy, cam::Frame::hold) is let go when the receiver releases its frame: the
// ai: receiver's tag to the camera and its hold under holdMu, looked up at the release (on the receiver's thread). The
// ai: phase lock (phase_host.h over phase_lock.h) runs on the camera's thread, from its capture results (onCapture).
#include "session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include "camera.h"
#include "crops.h"
#include "json.hpp"
#include "lizfile.h"
#include "phase_host.h"
#include "receiver.h"
#include "replay.h"
#include "stream.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#include <android/native_window.h>
#include <sys/system_properties.h>

#include "android_env.h"
#endif

namespace lizard {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// ai: the lock file's own (end of this file, where the system's headers are): 1 taken (handle set), 0 held by another
// ai: process, -1 the folder takes no lock (a file system without locks: received as before, unlocked)
int osLock(const std::string& path, intptr_t& handle);
void osUnlock(intptr_t handle);

constexpr int64_t PREVIEW_HOLD_NS = 1'000'000'000;   // ai: how long a preview asked keeps the capture thread copying crops
constexpr int TICK_MS = 100;                         // ai: the rules' period between commands
constexpr int REPLAY_FRAMES = 300;                   // ai: a run's frames (the web's and the apps' 300)
constexpr size_t REPLAY_ROWS = 60;                   // ai: a run's stats rows, one a second
const char* const BUSY = "another LIZARD window is receiving";
// ai: the stats' figures of the receiver's last window (receiver.cpp stats): a stream's, so none while none runs
const char* const WINDOW_FIGURES[] = {"capturedFps", "processedFps", "dropped", "foundShare", "side", "heldMs", "heldMaxMs", "blocks",
                                      "windowSecs", "goodputKBs", "gpuMs", "cpuMs", "repeatShare"};

int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ai: names from the camera and the light may hold bytes that are not UTF-8: replaced, never a throw
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

json parsed(const std::string& s) {
  try {
    json j = json::parse(s);
    return j.is_object() ? j : json::object();
  } catch (const std::exception&) {
    return json::object();
  }
}

std::string text(const json& j, const char* k) {
  if (!j.is_object()) return "";
  const auto it = j.find(k);
  return it != j.end() && it->is_string() ? it->get<std::string>() : "";
}

std::string num(double v, int digits) {
  char b[48];
  std::snprintf(b, sizeof b, "%.*f", digits, v);
  return b;
}

// ai: a setting's value as the log says it: its own digits, at most six significant
std::string gnum(double v) {
  char b[48];
  std::snprintf(b, sizeof b, "%.6g", v);
  return b;
}

// ai: the decoders a session makes: the PC stream's ("pc", stream.h) on Android alone, the phone end of the PC link
#if defined(__ANDROID__)
#define LIZ_DECODERS {"auto", "gpu", "cpu", "pc"}
#else
#define LIZ_DECODERS {"auto", "gpu", "cpu"}
#endif

bool oneOf(const std::string& v, std::initializer_list<const char*> of, const char* what, std::string& why) {
  std::string list;
  for (const char* o : of) {
    if (v == o) return true;
    list += std::string(list.empty() ? "" : ", ") + o;
  }
  why = std::string("the ") + what + " is one of " + list + " (not \"" + v + "\")";
  return false;
}

// ai: a file's time as ms since 1970 (C++17 has no clock_cast: through both clocks' now)
int64_t msOf(fs::file_time_type t) {
  using namespace std::chrono;
  const auto d = duration_cast<system_clock::duration>(t - fs::file_time_type::clock::now());
  return duration_cast<milliseconds>((system_clock::now() + d).time_since_epoch()).count();
}

// ai: a folder name the caller gives (a take's id, a transfer's): one plain name, nothing that leaves the folder
bool plainName(const std::string& s) {
  return !s.empty() && s != "." && s != ".." && s.find('/') == std::string::npos && s.find('\\') == std::string::npos && s.find(':') == std::string::npos;
}

// ai: written whole beside it and renamed over it, so a crash mid-write leaves the file as it was
bool writeWhole(const fs::path& p, const std::string& s, std::string& why) {
  const fs::path tmp = p.parent_path() / (p.filename().u8string() + ".tmp");
  {
    std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
    if (o) o.write(s.data(), std::streamsize(s.size()));
    if (o) o.close();
    if (!o) {
      why = "could not write " + tmp.u8string();
      std::error_code e;
      fs::remove(tmp, e);
      return false;
    }
  }
  std::error_code e;
  fs::rename(tmp, p, e);
  if (e) {
    why = "could not rename " + tmp.u8string() + ": " + e.message();
    fs::remove(tmp, e);
    return false;
  }
  return true;
}

std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// ai: a file's bytes made durable (a copy's, before its source goes); false where they may not have reached the disk
bool syncPath(const fs::path& p) {
#if defined(_WIN32)
  FILE* f = _wfopen(p.c_str(), L"r+b");
  if (!f) return false;
  const bool ok = _commit(_fileno(f)) == 0;
  std::fclose(f);
  return ok;
#else
  const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = ::fsync(fd) == 0 || errno == EINVAL || errno == EROFS;
  ::close(fd);
  return ok;
#endif
}

// ai: An area average's taps along one axis, `in` samples to `out` (out <= in): output i the mean of the input span
// ai: [i in / out, (i + 1) in / out), the span's end pixels weighed by the share of them it covers.
struct Tap { uint32_t first = 0; std::vector<float> w; };
std::vector<Tap> tapsFor(uint32_t in, uint32_t out) {
  std::vector<Tap> t(out);
  const double r = double(in) / out;
  for (uint32_t i = 0; i < out; i++) {
    const double a = i * r, b = std::min<double>(in, (i + 1) * r);
    const uint32_t f = std::min(in - 1, uint32_t(a)), l = std::max(f + 1, std::min(in, uint32_t(std::ceil(b - 1e-9))));
    t[i].first = f;
    for (uint32_t k = f; k < l; k++) t[i].w.push_back(float(std::max(0.0, std::min<double>(k + 1, b) - std::max<double>(k, a)) / r));
  }
  return t;
}

// ai: src (sw x sh, rows of sw) area-averaged to dst (dw x dh, rows of dw): rows first, then columns, in floats; mid
// ai: and acc the caller's buffers, grown as needed
void shrink(const uint8_t* src, uint32_t sw, uint32_t sh, uint8_t* dst, uint32_t dw, uint32_t dh, std::vector<float>& mid,
            std::vector<float>& acc) {
  if (dw == sw && dh == sh) {
    std::memcpy(dst, src, size_t(sw) * sh);
    return;
  }
  const std::vector<Tap> tx = tapsFor(sw, dw), ty = tapsFor(sh, dh);
  mid.resize(size_t(dw) * sh);
  acc.resize(dw);
  for (uint32_t y = 0; y < sh; y++) {
    const uint8_t* s = src + size_t(y) * sw;
    float* o = mid.data() + size_t(y) * dw;
    for (uint32_t i = 0; i < dw; i++) {
      const Tap& t = tx[i];
      float a = 0;
      for (size_t k = 0; k < t.w.size(); k++) a += t.w[k] * s[t.first + k];
      o[i] = a;
    }
  }
  for (uint32_t j = 0; j < dh; j++) {
    std::fill(acc.begin(), acc.end(), 0.f);
    const Tap& t = ty[j];
    for (size_t k = 0; k < t.w.size(); k++) {
      const float* m = mid.data() + size_t(t.first + k) * dw;
      const float wk = t.w[k];
      for (uint32_t i = 0; i < dw; i++) acc[i] += wk * m[i];
    }
    uint8_t* o = dst + size_t(j) * dw;
    for (uint32_t i = 0; i < dw; i++) o[i] = uint8_t(std::lround(std::clamp(acc[i], 0.f, 255.f)));
  }
}

json infoJson(const cam::Info& i) { return {{"id", i.id}, {"name", i.name}, {"backend", i.backend}, {"detail", i.detail}}; }

json describe(cam::Camera& c, const std::vector<cam::Mode>& modes, const std::vector<cam::Control>& controls) {
  json ms = json::array();
  for (const auto& m : modes) {
    const char* pix = cam::pixName(m.pix);
    json x = {{"w", m.width}, {"h", m.height}, {"fps", m.fps}, {"fpsMin", m.fpsMin}, {"format", m.format}, {"pix", pix ? pix : ""}, {"compressed", m.compressed}};
    // ai: the size of a camera's second stream for the platform's preview (camera.h Mode::previewW), where it writes one
    if (m.previewW && m.previewH) x["preview"] = {m.previewW, m.previewH};
    ms.push_back(std::move(x));
  }
  json ks = json::array();
  for (const auto& k : controls)
    ks.push_back({{"id", k.id}, {"name", k.name}, {"unit", k.unit}, {"min", k.min}, {"max", k.max}, {"step", k.step},
                  {"def", k.def}, {"value", k.value}, {"canAuto", k.canAuto}, {"isAuto", k.isAuto}});
  return {{"info", infoJson(c.info())}, {"modes", ms}, {"pick", cam::pick(modes, 1920, 1080, 60)}, {"controls", ks}};
}

// ai: An object others may still hold for the length of one call (a stats, a series) let go here, on this thread, once
// ai: they have: its destructor (a receiver's joins and device wait, a camera's teardown) lands on the worker. Past 5 s
// ai: the last holder ends it.
template <class T>
void letGo(std::shared_ptr<T>& p) {
  for (int i = 0; p && p.use_count() > 1 && i < 5000; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  p.reset();
}

// ai: The one-session lock (2026-10-27, the desktop's ReceiveLock before 2026-10-31): <data>/receive.lock under the
// ai: system's file lock (fcntl's on POSIX, LockFileEx on Windows: the locks the JVM's FileChannel.tryLock took there,
// ai: so a window of an older build is held off too), one receive session a data folder: the store's open lets go what
// ai: holds no header (a file not yet taken, another session's transfer in hand), so a second session on the folder
// ai: would clear it under the first. Within one process a set of the folders held, and of those whose session is
// ai: closing (its destructor begun): a session made while one of its own on the folder is closing waits for it, and
// ai: so does an unfinished transfer's delete with no session (deletePartial), which a live one refuses at once.
std::mutex gHeldMu;
std::condition_variable gHeldCv;
std::set<std::string> gHeld, gClosing;

std::string folderKey(const std::string& dir) {
  std::error_code e;
  const fs::path p = fs::weakly_canonical(fs::u8path(dir), e);
  return (e ? fs::u8path(dir) : p).u8string();
}

struct DataLock {
  std::string key;
  intptr_t handle = -1;
  ~DataLock() {
    if (handle != -1) osUnlock(handle);
    {
      std::lock_guard<std::mutex> l(gHeldMu);
      gHeld.erase(key);
      gClosing.erase(key);
    }
    gHeldCv.notify_all();
  }
};

// ai: the folder's lock: a holder of this process waited for up to waitMs, and up to closingMs where it is closing
std::unique_ptr<DataLock> takeLock(const std::string& dir, int waitMs, std::string& why, int closingMs = 0) {
  const std::string key = folderKey(dir);
  std::unique_lock<std::mutex> l(gHeldMu);
  const auto t0 = std::chrono::steady_clock::now();
  while (gHeld.count(key)) {
    const auto end = t0 + std::chrono::milliseconds(gClosing.count(key) ? std::max(waitMs, closingMs) : waitMs);
    if (std::chrono::steady_clock::now() >= end) {
      why = BUSY;
      return nullptr;
    }
    gHeldCv.wait_until(l, end);
  }
  std::error_code e;
  fs::create_directories(fs::u8path(dir), e);
  intptr_t h = -1;
  const int r = osLock((fs::u8path(dir) / "receive.lock").u8string(), h);
  if (r == 0) {
    why = BUSY;
    return nullptr;
  }
  auto k = std::make_unique<DataLock>();
  k->key = key;
  k->handle = r > 0 ? h : -1;
  gHeld.insert(key);
  return k;
}

// ai: The per-camera settings (session.h's header: cameras.json), the session's alone (held in memory where asked).
class Settings {
 public:
  Settings(fs::path path, bool persist, std::function<void(const std::string&)> say) : path_(std::move(path)), persist_(persist), say_(std::move(say)) {
    all_ = {{"version", 1}, {"cameras", json::object()}};
    if (!persist_) return;
    std::error_code e;
    if (!fs::exists(path_, e)) return;
    const json j = parsed(slurp(path_));
    if (j.contains("cameras") && j["cameras"].is_object()) all_["cameras"] = j["cameras"];
    else say_("camera settings: " + path_.u8string() + " holds no cameras object; begun afresh");
  }
  json of(const std::string& id) {
    std::lock_guard<std::mutex> l(mu_);
    const json& cs = all_["cameras"];
    return cs.contains(id) && cs[id].is_object() ? cs[id] : json::object();
  }
  void put(const std::string& id, const json& entry) {
    std::lock_guard<std::mutex> l(mu_);
    all_["cameras"][id] = entry;
    if (!persist_) return;
    std::string why;
    if (!writeWhole(path_, all_.dump(1), why)) say_("camera settings: " + why);
  }

 private:
  std::mutex mu_;
  const fs::path path_;
  const bool persist_;
  std::function<void(const std::string&)> say_;
  json all_;
};

// ai: a number or null where the value is one, null otherwise; an object member, {} where it is none (cameras.json is a
// ai: file anyone may edit: nothing in it throws)
json numOrNull(const json& j, const char* k) { return j.is_object() && j.contains(k) && j[k].is_number() ? j[k] : json(); }
json objectAt(const json& j, const char* k) { return j.is_object() && j.contains(k) && j[k].is_object() ? j[k] : json::object(); }

}  // namespace

struct Session::Impl {
  Config cfg;
  fs::path data, store, keptDir, replaysDir;
  std::unique_ptr<Settings> settings;
  void say(const std::string& m) const {
    if (cfg.log) cfg.log(m);
  }

  // ai: the queue
  std::mutex qMu;
  std::condition_variable qCv;
  std::deque<std::function<void()>> q;   // ai: qMu
  bool quit = false;                     // ai: qMu: no more commands; the worker ends once the queue is empty
  struct PendingSet {
    double value = 0;
    bool automatic = false;
    std::shared_ptr<std::promise<std::string>> done;
    Done fut;
  };
  std::map<std::string, PendingSet> pendingSets;   // ai: qMu: a set queued and not yet run, by control
  std::thread worker;

  Done post(std::function<std::string()> f) {
    auto pr = std::make_shared<std::promise<std::string>>();
    Done fut = pr->get_future().share();
    {
      std::lock_guard<std::mutex> l(qMu);
      if (quit) {
        pr->set_value("the session is closing");
        return fut;
      }
      q.push_back([this, pr, f = std::move(f)] { pr->set_value(run(f)); });
    }
    qCv.notify_one();
    return fut;
  }
  // ai: a command's body, a throw its refusal
  std::string run(const std::function<std::string()>& f) {
    try {
      return f();
    } catch (const std::exception& e) {
      say(std::string("session: ") + e.what());
      return e.what();
    } catch (...) {
      return "the session failed";
    }
  }

  // ai: the phase
  enum class Phase { Idle, Starting, Loading, On, Error };
  std::mutex phMu;
  Phase phase = Phase::Idle;   // ai: phMu
  std::string why;             // ai: phMu: the error's sentence
  uint64_t gen = 0;            // ai: phMu: a start or a stop asked: what was begun before it changes no phase
  uint64_t ask(Phase p) {
    std::lock_guard<std::mutex> l(phMu);
    phase = p;
    why.clear();
    return ++gen;
  }
  uint64_t genNow() {
    std::lock_guard<std::mutex> l(phMu);
    return gen;
  }
  Phase phaseNow() {
    std::lock_guard<std::mutex> l(phMu);
    return phase;
  }
  // ai: a command's outcome, where no start or stop came after it was asked
  void told(uint64_t g, Phase p, const std::string& w = "") {
    std::lock_guard<std::mutex> l(phMu);
    if (g != gen) return;
    phase = p;
    why = p == Phase::Error ? w : std::string();
  }
  bool running() {
    const Phase p = phaseNow();
    return p == Phase::Starting || p == Phase::Loading || p == Phase::On;
  }

  // ai: the lock
  std::unique_ptr<DataLock> lock;   // ai: the worker's
  std::string lockWhy;              // ai: the worker's: why it is not held
  std::atomic<bool> holding{false}; // ai: the lock taken (the destructor's look: it marks the folder closing)

  bool haveLock(int waitMs, std::string& w) {
    if (lock) return true;
    lock = takeLock(data.u8string(), waitMs, lockWhy);
    if (!lock) {
      w = lockWhy;
      return false;
    }
    holding = true;
    return true;
  }

  // ai: the receiver
  struct Rx {
    std::unique_ptr<Receiver> r;
    std::string decoder, layout;
  };
  std::mutex rxMu;
  std::shared_ptr<Rx> rx;          // ai: rxMu
  int cap = 0;                     // ai: rxMu: batchCap's n, handed to a receiver made later (0 none asked)
  std::shared_ptr<Replay> rec;     // ai: rxMu: the recording replay, handed to a receiver made later
  std::shared_ptr<Rx> current() {
    std::lock_guard<std::mutex> l(rxMu);
    return rx;
  }
  // ai: the worker's: what the next receiver is made with (create's, then each reconfigure's)
  std::string wantDecoder, wantLayout;
  std::mutex askedMu;
  json askedJson;   // ai: askedMu: the same for the stats

  void noteAsked() {
    std::lock_guard<std::mutex> l(askedMu);
    askedJson = {{"decoder", wantDecoder}, {"layout", wantLayout}};
  }

  // ai: from: the receiver whose transfer the new one goes on with (receiver.h create's from), null for one of its own
  std::shared_ptr<Rx> makeRx(const std::string& decoder, const std::string& layout, Receiver* from, std::string& w) {
    ReceiverConfig rc;
    rc.assets = cfg.assets;
    rc.cacheDir = cfg.cacheDir;
    rc.storeDir = store.u8string();
    rc.decoder = decoder;
    rc.layout = layout;
    rc.log = cfg.log;
#if defined(__ANDROID__)
    applyDebugEnv();
#endif
    // ai: a frame's release: a camera's own buffer let go (holds); a luma frame's has nothing to let go
    const auto rel = [this](uint64_t tag) { onRelease(tag); };
    try {
      auto r = from ? Receiver::create(rc, rel, *from) : Receiver::create(rc, rel);
      if (!r) {
        w = "the receiver could not be made";
        return nullptr;
      }
      auto x = std::make_shared<Rx>();
      x->r = std::move(r);
      x->decoder = decoder;
      x->layout = layout;
      return x;
    } catch (const std::exception& e) {
      w = std::string(decoder == "gpu" ? "the GPU decoder could not start: " : "the receiver could not start: ") + e.what();
      return nullptr;
    }
  }

  // ai: A receiver call that threw (2026-10-31: a Vulkan wait past its bound throws wg::Lost, the device then not to be
  // ai: trusted): the receiver is held broken, the camera stopped and the phase the error; the next start makes a new
  // ai: receiver on its transfer (create's from: the transfer is the CPU's) and lets the broken one go. A frame's push
  // ai: that threw is the capture thread's, told here at the next tick.
  bool rxBroken = false;               // ai: the worker's
  std::mutex pushMu;
  std::string pushFailed;              // ai: pushMu: a push's throw, for the next tick
  void rxFailed(const std::string& what) {
    say("receiver: " + what);
    const bool was = running();
    rxBroken = true;
    try {
      if (camera) stopLocked();
      describeNow();
    } catch (const std::exception&) {
    }
    if (was) told(genNow(), Phase::Error, "The receiver stopped: " + what);
  }

  // ai: the worker's: a receiver where none is (or the one in hand broke), with what was asked last; its own crop for a
  // ai: camera that names one
  bool haveRx(std::string& w) {
    const auto cur = current();
    if (cur && !rxBroken) return true;
    const std::string lay = ownLayout().empty() ? wantLayout : ownLayout();
    auto made = makeRx(wantDecoder, lay, cur ? cur->r.get() : nullptr, w);
    if (!made) return false;
    std::shared_ptr<Rx> old;
    try {
      std::lock_guard<std::mutex> l(rxMu);
      if (cap) made->r->batchCap(cap);
      if (rec) made->r->record(rec);
      old = std::move(rx);
      rx = std::move(made);
    } catch (const std::exception& e) {
      w = std::string("the receiver could not start: ") + e.what();
      return false;
    }
    rxBroken = false;
    letGo(old);
    return true;
  }

  // ai: the camera
  std::vector<cam::Mode> modes;    // ai: the worker's: the open camera's
  std::vector<cam::Control> ctls;  // ai: the worker's: its controls as last read (describeNow)
  int mode = -1;                   // ai: the worker's: the index started last
  bool streamed = false;           // ai: the worker's: a stream ran since the receiver was last told the camera closed
  std::atomic<bool> streaming{false};
  std::atomic<uint64_t> starts{0};   // ai: the camera's starts since it opened (the stats' "starts")
  // ai: The user's settings since the camera opened (its kept ones taken up at the open), each control's newest in the
  // ai: order first asked, set again after each start. The worker's.
  struct Want {
    std::string id;
    double value = 0;
    bool automatic = false;
  };
  std::vector<Want> wants;
  int64_t asksSeen = -1;           // ai: the worker's: the phone's own Start and Stop as last counted, -1 before a first read

  std::mutex ptrMu;
  std::shared_ptr<cam::Camera> camera;   // ai: read under ptrMu by others; written by the worker under it
  json info, modeNow, described;         // ai: ptrMu: its Info, the mode started last (null none), camera()'s answer
  std::string backend, detail, camId;    // ai: the worker's: the open camera's

  std::mutex csMu;
  cam::Stats cs;                   // ai: csMu: Camera::stats as last read

  std::mutex endMu;
  std::string ended;               // ai: endMu: the stream's own end's why ("" none)

  std::atomic<uint64_t> tags{0};   // ai: frames pushed: each frame's tag, never reset (the receiver keys stamps by it)
  std::atomic<uint64_t> pushErrors{0};

  // ai: the measured rate: a window from a frame, closed at the first frame a second or more later
  std::mutex fpsMu;
  int64_t fpsStart = 0;            // ai: fpsMu: the window's first frame's arrival (0 none yet)
  int64_t fpsLast = 0;             // ai: fpsMu: the newest frame's arrival
  int fpsN = 0;                    // ai: fpsMu: frames since it
  double fps = 0;                  // ai: fpsMu: the last window's rate (0 none closed since the start)

  struct Crop {
    std::vector<uint8_t> px;
    uint32_t w = 0, h = 0;
    uint64_t seq = 0;
    int64_t at = 0;   // ai: its frame's arrival (steady ns)
  };
  // ai: A preview copies `newest` under prevMu and lets its copy go under it too (Held), so the capture thread's test
  // ai: that no preview holds the spare (its use count, under the same lock) comes after every read of it.
  std::mutex prevMu;
  std::shared_ptr<Crop> newest;    // ai: prevMu
  std::shared_ptr<Crop> spare;     // ai: prevMu: the crop before the newest, filled next once no preview holds it
  std::atomic<int64_t> asked{0};   // ai: when a preview was last asked (steady ns, 0 never)
  struct Held {
    std::mutex& mu;
    std::shared_ptr<Crop> c;
    ~Held() {
      std::lock_guard<std::mutex> l(mu);
      c.reset();
    }
  };
  // ai: The preview's scaling buffers (shrink), the session's under a leaf lock of their own, so previews asked from two
  // ai: threads at once take turns (2026-10-25: they were thread_locals, and under MinGW-w64 GCC's emulated TLS threads
  // ai: that end holding a thread_local with a destructor corrupt the heap).
  std::mutex scaleMu;
  std::vector<float> scaleMid, scaleAcc;   // ai: scaleMu

  // ai: the crop a phone or an upload names in its detail ("2:1, ..."; cam/phone.cpp, upload.cpp), "" for any other
  std::string ownLayout() const {
    if (backend != "phone" && backend != "upload") return "";
    const std::string l = detail.substr(0, detail.find(','));
    return l == "1:1" || l == "2:1" ? l : "";
  }

  // ai: Frames whose buffer their camera holds for the receiver (cam::Frame::hold, Android's zero copy): the frame's
  // ai: tag to its camera and its hold, until the receiver releases the tag (onRelease, on the receiver's threads or
  // ai: inside push for a frame it drops). A camera gone by then let its buffers go at its stop.
  struct Hold {
    std::weak_ptr<cam::Camera> camera;
    uint64_t hold = 0;
  };
  std::mutex holdMu;
  std::unordered_map<uint64_t, Hold> holds;   // ai: holdMu
  void onRelease(uint64_t tag) {
    Hold h;
    {
      std::lock_guard<std::mutex> l(holdMu);
      const auto it = holds.find(tag);
      if (it == holds.end()) return;
      h = std::move(it->second);
      holds.erase(it);
    }
    if (const auto c = h.camera.lock()) c->release(h.hold);
  }

  // ai: the capture thread's: a frame into the receiver (luma, the camera's own buffer, or a raw frame to develop), its
  // ai: crop for the preview (luma's)
  void onFrame(const cam::Frame& f, const std::weak_ptr<cam::Camera>& from) {
    const bool luma = f.luma && f.stride >= f.width;
    if ((!luma && !f.hb && !f.raw) || !f.width || !f.height) {
      if (f.hold)
        if (const auto c = from.lock()) c->release(f.hold);
      if (pushErrors++ == 0) say("camera: a frame with no luma, buffer or raw plane, or rows shorter than its width, was let go");
      return;
    }
    const int64_t now = nowNs();
    const uint64_t tag = ++tags;
    const std::shared_ptr<Rx> r = current();
    CameraFrame c;
    c.luma = luma ? f.luma : nullptr;
    c.hb = f.hb;
    c.raw = f.raw;
    c.width = f.width;
    c.height = f.height;
    c.stride = luma ? f.stride : 0;
    c.timestampNs = f.ns;
    c.tag = tag;
    if (f.hold) {
      std::lock_guard<std::mutex> l(holdMu);
      holds[tag] = Hold{from, f.hold};
    }
    try {
      if (r) r->r->push(c);
      else onRelease(tag);
    } catch (const std::exception& e) {
      onRelease(tag);
      if (pushErrors++ == 0) {
        std::lock_guard<std::mutex> l(pushMu);
        pushFailed = std::string("a frame was refused: ") + e.what();
      }
    }
    const int64_t a = asked.load(std::memory_order_relaxed);
    if (luma && a && now - a < PREVIEW_HOLD_NS) keepCrop(f, r && r->layout == "2:1", tag, now);
    std::lock_guard<std::mutex> l(fpsMu);
    fpsLast = now;
    if (!fpsStart) {
      fpsStart = now;
      fpsN = 0;
      return;
    }
    fpsN++;
    if (now - fpsStart >= 1'000'000'000) {
      fps = fpsN * 1e9 / double(now - fpsStart);
      fpsStart = now;
      fpsN = 0;
    }
  }

  // ai: the decoder's crop of the frame copied, into the spare where no preview holds it (else a new one), outside the
  // ai: lock; then the newest, the newest before it the spare
  void keepCrop(const cam::Frame& f, bool two, uint64_t seq, int64_t now) {
    // ai: the region the decoder reads (crops.h): the centre square, or the 2:1 halves whole
    const Crops c = cropsOf(f.width, f.height, two);
    const uint32_t gx = c.x[0], gy = c.y[0], gw = c.x[c.n - 1] - c.x[0] + c.side, gh = c.y[c.n - 1] - c.y[0] + c.side;
    if (!c.side) return;
    std::shared_ptr<Crop> k;
    {
      std::lock_guard<std::mutex> l(prevMu);
      if (spare && spare.use_count() == 1) k = std::move(spare);
    }
    if (!k) k = std::make_shared<Crop>();
    k->px.resize(size_t(gw) * gh);
    for (uint32_t y = 0; y < gh; y++) std::memcpy(k->px.data() + size_t(y) * gw, f.luma + size_t(gy + y) * f.stride + gx, gw);
    k->w = gw;
    k->h = gh;
    k->seq = seq;
    k->at = now;
    std::lock_guard<std::mutex> l(prevMu);
    spare = std::move(newest);
    newest = std::move(k);
  }

  void dropCrop() {
    std::lock_guard<std::mutex> l(prevMu);
    newest.reset();
    spare.reset();
  }

  // ai: the capture thread's, once, where the stream ends by itself (camera.h Ended): no camera call from here, since
  // ai: a stop on the worker may be joining this very thread; the rules close the camera at the next tick
  void onEnded(const std::string& w0) {
    const std::string w = w0.empty() ? "the stream ended" : w0;
    {
      std::lock_guard<std::mutex> l(endMu);
      ended = w;
    }
    streaming = false;
    say("camera: the stream ended: " + w);
  }

  // ai: the camera's counts read into the cache
  void readStats() {
    cam::Stats s;
    if (camera) s = camera->stats();
    std::lock_guard<std::mutex> l(csMu);
    cs = s;
  }

  // ai: camera()'s answer read off the camera again, with the kept settings
  void describeNow() {
    json j = json::object();
    if (camera) {
      ctls = camera->controls();
      j = describe(*camera, modes, ctls);
      j["mode"] = mode;
      j["streaming"] = streaming.load();
      j["kept"] = keptJson();
    }
    std::lock_guard<std::mutex> l(ptrMu);
    described = std::move(j);
  }

  const cam::Control* control(const std::string& id) const {
    const auto k = std::find_if(ctls.begin(), ctls.end(), [&](const cam::Control& c) { return c.id == id; });
    return k == ctls.end() ? nullptr : &*k;
  }
  // ai: a control a user can move: listed with a range (the app's Settings bars)
  bool movable(const std::string& id) const {
    const cam::Control* c = control(id);
    return c && c->max > c->min;
  }

  // ai: the receiver's crop now (the one in hand's, else the one asked)
  std::string layoutNow() {
    const auto r = current();
    return r ? r->layout : wantLayout;
  }

  // ai: the mode start("") takes: the kept one (its size and rate, its format where it is offered, uncompressed
  // ai: first), else camera.h pick's for 1920 x 1080 at 60, else the first; -1 with no modes
  int keptMode() {
    if (modes.empty()) return -1;
    const json m = objectAt(settings->of(camId), "mode");
    if (numOrNull(m, "w").is_number() && numOrNull(m, "h").is_number() && numOrNull(m, "fps").is_number()) {
      const double w = m["w"].get<double>(), h = m["h"].get<double>(), f = m["fps"].get<double>();
      const std::string fmt = text(m, "format");
      int best = -1;
      for (size_t i = 0; i < modes.size(); i++) {
        const cam::Mode& x = modes[i];
        if (x.width != w || x.height != h || std::fabs(x.fps - f) >= 0.01) continue;
        if (x.format == fmt) return int(i);
        if (best < 0 || (modes[size_t(best)].compressed && !x.compressed)) best = int(i);
      }
      if (best >= 0) return best;
    }
    const int p = cam::pick(modes, 1920, 1080, 60);
    return p >= 0 ? p : 0;
  }

  json keptJson() {
    const json k = settings->of(camId);
    const json z = objectAt(k, "zoom");
    return {{"mode", keptMode()}, {"zoom", numOrNull(z, layoutNow().c_str())}, {"exposure", numOrNull(k, "exposure")}, {"gain", numOrNull(k, "gain")}};
  }

  // ai: a setting kept for the camera where it is one cameras.json holds (zoom its crop's, exposure, gain, mode)
  void save(const std::string& id, double value, bool automatic) {
    if (camId.empty()) return;
    json k = settings->of(camId);
    if (id == "zoom") {
      if (!k.contains("zoom") || !k["zoom"].is_object()) k["zoom"] = json::object();
      k["zoom"][layoutNow()] = value;
    } else if (id == "exposure" || id == "gain") {
      k[id] = automatic ? json() : json(value);
    } else if (id == "mode") {
      const int i = int(value);
      if (i < 0 || size_t(i) >= modes.size()) return;
      k["mode"] = {{"w", modes[size_t(i)].width}, {"h", modes[size_t(i)].height}, {"fps", modes[size_t(i)].fps}, {"format", modes[size_t(i)].format}};
    } else {
      return;
    }
    settings->put(camId, k);
  }

  // ai: a setting the user asked, kept for this open (the control's earlier one replaced)
  void keep(const std::string& id, double value, bool automatic) {
    for (Want& w : wants)
      if (w.id == id) {
        w.value = value;
        w.automatic = automatic;
        return;
      }
    wants.push_back({id, value, automatic});
  }

  // ai: a stream just begun: the settings kept set again, in the order first asked; a refusal said in the log (the
  // ai: start stands, the setting stays kept for the next)
  void setAgain() {
    if (wants.empty() || !camera) return;
    std::string said;
    for (const Want& w : wants) {
      std::string wy;
      bool ok = false;
      try {
        ok = camera->set(w.id, w.value, w.automatic, wy);
      } catch (const std::exception& e) {
        wy = e.what();
      }
      said += (said.empty() ? "" : ", ") + w.id + " " + (w.automatic ? std::string("auto") : gnum(w.value)) +
              (ok ? std::string() : " refused (" + (wy.empty() ? std::string("no reason given") : wy) + ")");
    }
    say("camera: set again after the start: " + said);
  }

  // ai: The stream stopped (its thread joined: no sink call after this), then the receiver told.
  void stopLocked() {
    if (!camera) return;
    try {
      camera->stop();
    } catch (const std::exception& e) {
      say(std::string("camera: the stop failed: ") + e.what());
    }
    // ai: every frame of the stream handed back (push copies a frame's crop, and the stop joined the capture thread):
    // ai: the receiver lets go what it kept of the stream, an ingest error with it, and ends its stats window, so a
    // ai: start reads afresh and its first window is that stream's own (receiver.h cameraClosed); once a stream, a
    // ai: stop after its own end too
    if (streamed) {
      streamed = false;
      if (const std::shared_ptr<Rx> r = current()) {
        try {
          r->r->cameraClosed();
        } catch (const std::exception& e) {
          say(std::string("receiver: the camera's close: ") + e.what());
        }
      }
    }
    const bool was = streaming.exchange(false);
    try {
      readStats();
    } catch (const std::exception&) {
    }
    if (was) say("camera: stopped");
  }

  // ai: modes[index] streamed into the receiver
  bool startLocked(int index, std::string& w) {
    const cam::Mode m = modes[size_t(index)];
    {
      std::lock_guard<std::mutex> l(fpsMu);
      fpsStart = 0;
      fpsN = 0;
      fps = 0;
    }
    {
      std::lock_guard<std::mutex> l(endMu);
      ended.clear();
    }
    // ai: what the receiver takes (camera.h receiverWants): its luma (the C, the PC stream) or the camera's own buffer
    // ai: (the GPU decoder's zero copy)
    if (const auto r = current()) {
      try {
        camera->receiverWants(r->r->wantsLuma());
      } catch (const std::exception&) {
      }
    }
    armLock();
    // ai: set before the start, so a stream that ends at once (its Ended on the capture thread before start returns)
    // ai: leaves it false
    streaming = true;
    std::string wy;
    bool ok = false;
    const std::weak_ptr<cam::Camera> from = camera;
    try {
      ok = camera->start(m, [this, from](const cam::Frame& f) { onFrame(f, from); }, [this](const std::string& e) { onEnded(e); }, wy);
    } catch (const std::exception& e) {
      wy = e.what();
    }
    if (!ok) {
      streaming = false;
      w = wy.empty() ? "the camera did not start" : wy;
    } else {
      mode = index;
      streamed = true;
      starts++;
      {
        std::lock_guard<std::mutex> l(ptrMu);
        modeNow = {{"w", m.width}, {"h", m.height}, {"format", m.format}, {"fps", m.fps}};
      }
      say("camera: started " + std::to_string(m.width) + " x " + std::to_string(m.height) + " " + m.format + " at " + num(m.fps, 2) + " fps");
      // ai: a phone's stream reads back what the receiver reads of it (camera.h readBack): the receiver in hand's
      // ai: on the camera's own thread: a receiver's throw there is an empty answer, the worker told by its own calls
      camera->readBack(
          [this](double since) {
            try {
              const auto r = current();
              return r ? r->r->series(since) : std::vector<double>{0};
            } catch (const std::exception&) {
              return std::vector<double>{0};
            }
          },
          [this] {
            try {
              const auto r = current();
              return r ? r->r->stats() : std::string("{}");
            } catch (const std::exception&) {
              return std::string("{}");
            }
          });
      setAgain();
    }
    try {
      readStats();
      describeNow();
    } catch (const std::exception&) {
    }
    return ok;
  }

  // ai: the camera stopped and let go (here, once a stats holding it has let go too)
  void closeLocked() {
    if (!camera) return;
    stopLocked();
    lockCam = nullptr;   // ai: its results stopped with its stream: the lock's lever with it
    std::shared_ptr<cam::Camera> c;
    {
      std::lock_guard<std::mutex> l(ptrMu);
      c = std::move(camera);
      camera = nullptr;
      info = nullptr;
      modeNow = nullptr;
      described = json::object();
    }
    modes.clear();
    ctls.clear();
    wants.clear();
    mode = -1;
    backend.clear();
    detail.clear();
    camId.clear();
    asksSeen = -1;
    {
      std::lock_guard<std::mutex> l(csMu);
      cs = cam::Stats{};
    }
    dropCrop();
    const std::string name = c->info().name;
    letGo(c);
    say("camera: closed " + name);
  }

  double fpsNow() {
    std::lock_guard<std::mutex> l(fpsMu);
    // ai: no window closed for 1.5 s (a stall): the frames since the window's start over the time since
    const int64_t now = nowNs();
    if (fpsStart && now - fpsStart > 1'500'000'000) return fpsN * 1e9 / double(now - fpsStart);
    // ai: none closed yet in a stream's first second: its frames so far over their span, from the fourth frame on (0
    // ai: before: not measured yet)
    if (!fps && fpsN >= 3 && fpsLast > fpsStart) return fpsN * 1e9 / double(fpsLast - fpsStart);
    return fps;
  }

  // ai: the phase lock (phase_host.h)
  // ai: The lock's host, made with the session: Engine.kt's order around PhaseLock.kt (phase_lock.h) for a camera that
  // ai: can delay a capture, driven from that camera's capture results on its own thread (onCapture); the receiver's
  // ai: series its feed, the camera started last its lever. A camera with no capture results (UVC, a phone, an
  // ai: upload, the simulated one) never reaches it: its stats' lock says why (camera.h delayBar).
  std::unique_ptr<PhaseLockHost> phaseLock;
  std::atomic<cam::Camera*> lockCam{nullptr};   // ai: the camera whose results feed the lock (the worker sets it at each start)
  std::mutex lockMu;
  std::string lockBar;                   // ai: lockMu: why the camera started last cannot delay a capture ("" it can)
  uint64_t lockCaps = 0;                 // ai: the camera's thread's: its results since its capture session began

  // ai: the platform's switch: Android's debug.lizard.phase (track, or unset, tracks; anything else off), the tools'
  // ai: starts from random phases; elsewhere none
  static PhaseLock::Mode platformMode() {
#if defined(__ANDROID__)
    char v[PROP_VALUE_MAX] = "";
    __system_property_get("debug.lizard.phase", v);
    return !*v || std::string(v) == "track" ? PhaseLock::Mode::Track : PhaseLock::Mode::Off;
#else
    return PhaseLock::Mode::Track;
#endif
  }

  // ai: The worker's, before each start: the camera's capture results to the lock (a camera without them ignores it),
  // ai: and why it cannot delay a capture where it cannot (the lock Off for its sessions)
  void armLock() {
    const std::string bar = camera->delayBar();
    {
      std::lock_guard<std::mutex> l(lockMu);
      lockBar = bar;
    }
    cam::Camera* cp = camera.get();
    lockCam = cp;
    camera->captures([this](const cam::Capture& c) { onCapture(c); }, [this](bool taken) { phaseLock->fate(taken); });
  }

  // ai: The camera's thread, at each capture result (the app's Engine before 2026-10-31): a capture session's first is
  // ai: the lock's opened (its bar: Off on a lens that cannot delay one); the platform's switch read every 30 results;
  // ai: then the host's captured (the mode, the feed, the delay asked of the camera, the lock's line)
  void onCapture(const cam::Capture& c) {
    if (c.fresh) {
      std::string bar;
      {
        std::lock_guard<std::mutex> l(lockMu);
        bar = lockBar;
      }
      phaseLock->opened(bar);
      lockCaps = 0;
    }
    if (lockCaps++ % 30 == 15) phaseLock->want(platformMode());
    phaseLock->captured(c.ns, c.delayed);
  }

  // ai: The lock's state for the stats (PhaseLock::State; the Android app's names, NaN as null), or {"off": why} where
  // ai: the camera started last cannot delay a capture
  json lockJson() {
    std::string bar;
    {
      std::lock_guard<std::mutex> l(lockMu);
      bar = lockBar;
    }
    if (!bar.empty()) return {{"off", bar}};
    const std::optional<PhaseLock::State> st = phaseLock->state();
    if (!st) return json::object();
    const auto nz = [](double x) { return std::isfinite(x) ? json(x) : json(); };
    return {{"arm", st->arm}, {"state", st->what}, {"a", nz(st->a)}, {"b", nz(st->b)}, {"k", nz(st->k)}, {"se", nz(st->se)}, {"n", st->n},
            {"gain", nz(st->gain)}, {"pace", nz(st->pace)}, {"stood", st->stood}, {"edge", st->edge}, {"flatMs", nz(st->flatMs)},
            {"delays", st->delays}, {"refreshMs", nz(st->refreshMs)}, {"multiple", st->multiple}, {"frameMs", nz(st->frameMs)},
            {"landMs", nz(st->landMs)}, {"row", nz(st->row)}};
  }

  // ai: the platform's preview surface (Android's)
  void* previewWin = nullptr;   // ai: the worker's: an ANativeWindow*, acquired (previewWindow)

  // ai: the commands (the worker's)

  // ai: The camera by id (session.h open). A phone typed by its address or an upload opens unlisted; another is looked
  // ai: for in the list, the first listed in its place where it is not there (a camera unplugged since it was chosen).
  std::string openFailed;   // ai: the worker's: the last open's refusal as the phase says it ("" it opened)
  std::string doOpen(uint64_t g, const std::string& asked0) {
    std::string id = asked0;
    if (camera && !id.empty() && camId == id) return "";
    const auto refuse = [&](const std::string& said, const std::string& w) {
      closeLocked();
      openFailed = said;
      told(g, Phase::Error, said);
      return w;
    };
    std::string w;
    if (!haveLock(0, w)) return refuse("The camera did not start: " + w, w);
    closeLocked();
    // ai: the cameras listed only where needed (a listing probes devices, and listens for phones a second): for "", and
    // ai: for an id that would not open, which a camera listed in its place stands for where it is not listed itself
    std::unique_ptr<cam::Camera> first;
    if (!id.empty()) {
      try {
        first = cam::open(id, w);
      } catch (const std::exception& e) {
        w = e.what();
      }
    }
    const bool typed = (id.rfind("phone:", 0) == 0 && id.rfind("phone:usb:", 0) != 0) || id.rfind("upload:", 0) == 0;
    if (!first && !typed) {
      std::vector<std::string> notes;
      std::vector<cam::Info> listed;
      try {
        listed = cam::list(&notes);
      } catch (const std::exception& e) {
        notes.push_back(e.what());
      }
      const bool there = std::any_of(listed.begin(), listed.end(), [&](const cam::Info& i) { return i.id == id; });
      if (!there) {
        if (listed.empty()) {
          std::string n;
          for (const auto& x : notes) n += (n.empty() ? "" : "; ") + x;
          if (!n.empty()) say("camera: none listed: " + n);
          return refuse("No camera was found.", "no camera was found");
        }
        if (!id.empty()) say("camera: " + id + " is not listed: " + listed[0].id + " in its place");
        id = listed[0].id;
        w.clear();
      }
    }
    try {
      std::unique_ptr<cam::Camera> c = first ? std::move(first) : (w.empty() ? cam::open(id, w) : nullptr);
      if (!c) {
        w = w.empty() ? "the camera " + id + " could not be opened" : w;
        return refuse("The camera did not start: " + w, w);
      }
      std::shared_ptr<cam::Camera> sp(std::move(c));
      std::vector<cam::Mode> ms = sp->modes();
      {
        std::lock_guard<std::mutex> k(ptrMu);
        camera = sp;
        info = infoJson(sp->info());
        modeNow = nullptr;
      }
      modes = std::move(ms);
      mode = -1;
      starts = 0;
      backend = sp->info().backend;
      detail = sp->info().detail;
      camId = sp->info().id;
      asksSeen = -1;
      {
        std::lock_guard<std::mutex> k(endMu);
        ended.clear();
      }
      readStats();
      ctls = sp->controls();
      // ai: the kept settings taken up (the app's openCam before 2026-10-31): kept for the start, the camera untouched
      const json k = settings->of(camId);
      if (movable("zoom")) {
        const json z = numOrNull(objectAt(k, "zoom"), layoutNow().c_str());
        if (z.is_number()) keep("zoom", z.get<double>(), false);
      }
      for (const char* x : {"exposure", "gain"})
        if (movable(x) && numOrNull(k, x).is_number()) keep(x, k[x].get<double>(), false);
      if (previewWin) sp->previewTo(previewWin);
      describeNow();
      openFailed.clear();
      say("camera: opened " + sp->info().name + " (" + sp->info().id + "), " + std::to_string(modes.size()) + " modes");
      // ai: a camera chosen and open: an error before it said is no longer the line's
      {
        std::lock_guard<std::mutex> l(phMu);
        if (g == gen && phase == Phase::Error) {
          phase = Phase::Idle;
          why.clear();
        }
      }
      return "";
    } catch (const std::exception& e) {
      w = std::string("the camera could not be opened: ") + e.what();
      return refuse("The camera did not start: " + w, w);
    }
  }

  // ai: the mode a start names (session.h start), -1 with why
  int modeFor(const std::string& spec, bool& named, std::string& w) {
    named = !spec.empty();
    if (spec.empty()) {
      const int k = keptMode();
      if (k < 0) w = "the camera offers no mode";
      return k;
    }
    if (spec.find_first_not_of("0123456789") == std::string::npos) {
      const size_t i = spec.size() > 6 ? modes.size() : size_t(std::stoul(spec));
      if (i >= modes.size()) {
        w = "the camera has no mode " + spec + " (" + std::to_string(modes.size()) + " listed)";
        return -1;
      }
      return int(i);
    }
    unsigned mw = 0, mh = 0;
    double f = 0;
    char x = 0;
    if (std::sscanf(spec.c_str(), "%u%c%u@%lf", &mw, &x, &mh, &f) == 4 && (x == 'x' || x == 'X')) {
      int best = -1;
      for (size_t i = 0; i < modes.size(); i++)
        if (modes[i].width == mw && modes[i].height == mh && std::fabs(modes[i].fps - f) < 0.01 &&
            (best < 0 || (modes[size_t(best)].compressed && !modes[i].compressed)))
          best = int(i);
      if (best >= 0) return best;
    }
    w = "the camera offers no mode " + spec;
    return -1;
  }

  std::string doStart(uint64_t g, const std::string& spec) {
    std::string w;
    const auto refuse = [&](const std::string& r) {
      closeLocked();
      told(g, Phase::Error, "The camera did not start: " + r);
      return r;
    };
    if (!haveLock(0, w)) return refuse(w);
    retryTakes();
    if (!camera) {
      // ai: the open asked before this start refused: its sentence is the start's
      told(g, Phase::Error, openFailed.empty() ? "The camera did not start: no camera is open" : openFailed);
      return openFailed.empty() ? "no camera is open" : openFailed;
    }
    // ai: a camera that names its own crop (a phone's, an upload's): the receiver's crop its own
    const std::string own = ownLayout();
    if (!own.empty() && own != layoutNow()) {
      if (current()) {
        if (!swap(wantDecoder, own, w)) return refuse(w);
      }
    }
    if (!haveRx(w)) return refuse(w);
    bool named = false;
    const int m = modeFor(spec, named, w);
    if (m < 0) return refuse(w);
    if (named) save("mode", m, false);
    stopLocked();
    if (!startLocked(m, w)) return refuse(w);
    told(g, Phase::Loading);
    return "";
  }

  std::string doStop() {
    stopLocked();
    try {
      describeNow();
    } catch (const std::exception&) {
    }
    return "";
  }

  // ai: The receiver made again on the one in hand's transfer (session.h reconfigure), the camera stopped around the
  // ai: swap and started again in its mode where it ran. False with why: the old receiver kept where the new one could
  // ai: not be made; where the camera would not start again, the new one kept and the camera stopped.
  bool swap(const std::string& decoder, const std::string& layout, std::string& w) {
    std::shared_ptr<Rx> fresh;
    {
      // ai: this copy let go before letGo(old) below, which otherwise waited its whole 5 s on it (2026-10-25)
      const auto cur = current();
      if (cur && cur->decoder == decoder && cur->layout == layout) return true;
      // ai: Made while the old one still reads (a GPU receiver's device and pipelines take up to a second cold), on
      // ai: the old one's transfer: no second store on the folder, so a file the old receiver finishes before the swap,
      // ai: or after it with frames it still holds, is in the new one's file() and stats, and nothing of the transfer is
      // ai: taken again (2026-10-25)
      fresh = makeRx(decoder, layout, cur ? cur->r.get() : nullptr, w);
    }
    if (!fresh) return false;
    const bool was = streaming.load();
    const int m = mode;
    const std::string before = layoutNow();
    if (camera) stopLocked();
    std::shared_ptr<Rx> old;
    {
      // ai: the batch cap and the replay handed on in the section that swaps: a call from another thread lands on the
      // ai: old receiver before it or on the new one after it, never between
      std::lock_guard<std::mutex> k(rxMu);
      if (cap) fresh->r->batchCap(cap);
      if (rec) fresh->r->record(rec);
      old = std::move(rx);
      rx = fresh;
    }
    // ai: the old receiver's threads end here (a frame its decoder delivers meanwhile goes into the transfer both hold)
    letGo(old);
    dropCrop();   // ai: the old layout's shape
    say("receiver: now " + decoder + ", " + layout);
    // ai: the crop's own zoom (a crop keeps one of its own): the kept one for the new crop, where there is one
    if (camera && layout != before && movable("zoom")) {
      const json z = numOrNull(objectAt(settings->of(camId), "zoom"), layout.c_str());
      if (z.is_number()) keep("zoom", z.get<double>(), false);
    }
    if (!was || !camera || m < 0) {
      if (camera) describeNow();
      return true;
    }
    if (!startLocked(m, w)) {
      w = "the camera did not start again: " + w;
      return false;
    }
    return true;
  }

  std::string doReconfigure(uint64_t g, const std::string& decoder, const std::string& layout) {
    wantDecoder = decoder;
    wantLayout = layout;
    noteAsked();
    if (!current()) return "";
    const std::string own = ownLayout();
    std::string w;
    if (!swap(decoder, own.empty() ? layout : own, w)) {
      if (g == genNow() && running()) {
        closeLocked();
        told(g, Phase::Error, "The camera did not start: " + w);
      }
      return w;
    }
    return "";
  }

  std::string doSet(const std::string& control, double value, bool automatic) {
    std::string w;
    if (!camera) return "no camera is open";
    if (control == "mode") {
      const int i = int(value);
      if (i < 0 || size_t(i) >= modes.size()) return "the camera has no mode " + gnum(value);
      save("mode", i, false);
      if (streaming) {
        stopLocked();
        if (!startLocked(i, w)) {
          const uint64_t g = genNow();
          closeLocked();
          told(g, Phase::Error, "The camera did not start: " + w);
          return w;
        }
        told(genNow(), Phase::Loading);
      } else {
        describeNow();
      }
      return "";
    }
    bool ok = false;
    if (streaming) {
      try {
        ok = camera->set(control, value, automatic, w);
      } catch (const std::exception& e) {
        w = e.what();
      }
      if (!ok && w.empty()) w = "the camera refused " + control;
    } else {
      // ai: no stream: kept for the next start, the camera not touched (its controls as its backend left them at the
      // ai: stop: a value set now would go back at the next stop anyway, and Media Foundation would wake the device)
      const cam::Control* k = control.empty() ? nullptr : this->control(control);
      if (!k) w = "this camera has no " + control + " to set";
      else if (automatic && !k->canAuto) w = "this camera has no automatic " + control;
      else ok = true;
    }
    if (ok) {
      keep(control, value, automatic);
      save(control, value, automatic);
    } else {
      say("camera: " + control + ": " + w);
    }
    try {
      describeNow();
    } catch (const std::exception&) {
    }
    return ok ? "" : w;
  }

  std::string doForget(const std::string& root) {
    if (const auto r = current()) {
      r->r->forget(root);
      return "";
    }
    const std::string id = root.substr(0, 16);
    if (!plainName(id)) return "no transfer " + root;
    std::error_code e;
    fs::remove_all(store / "lizard-xfer" / fs::u8path(id), e);
    return e ? e.message() : "";
  }

  // ai: the rules (the worker's, between commands)
  std::string lastState;   // ai: the receiver's state at the last tick
  uint64_t focusTicks = 0; // ai: ticks while a stream runs, for the focus's second
  std::string heldRoot;    // ai: the transfer the receiver last held (its root, "" none)
  bool stoppedFor = false; // ai: the camera stopped once for heldRoot's file
  bool tookFor = false;    // ai: the take run once for heldRoot's file (one that failed is tried again at a retry)
  std::set<std::string> takeFailed;   // ai: the worker's: store folders whose take failed, not tried until the next retry
  std::mutex keptMu;
  uint64_t taken = 0;      // ai: keptMu
  std::string takenRoot, takeError;   // ai: keptMu

  // ai: A finished file out of the store's folder `from` (store.h fileStore: done/<name> beside meta.json, no header)
  // ai: into kept/<root16>-<n>/ (session.h's header): filing.json first, then the file renamed in (else copied, made
  // ai: durable, and the store's removed: a copy whose source will not go is taken all the same), then meta.json
  // ai: removed from `from`, which leaves it nothing a store's open keeps. A throw leaves the file in the store and no
  // ai: folder. Its root.
  std::string take(const fs::path& from) {
    std::error_code e;
    fs::path src;
    for (const auto& f : fs::directory_iterator(from / "done", e))
      if (f.is_regular_file(e)) {
        src = f.path();
        break;
      }
    if (src.empty()) throw std::runtime_error(from.u8string() + " holds no finished file");
    const json m = parsed(slurp(from / "meta.json"));
    const std::string root = text(m, "root"), type = text(m, "type");
    fs::create_directories(keptDir, e);
    fs::path dir;
    const std::string id = root.size() >= 16 && plainName(root.substr(0, 16)) ? root.substr(0, 16) : from.filename().u8string();
    for (int n = 1; n < 100000; n++) {
      const fs::path d = keptDir / fs::u8path(id + "-" + std::to_string(n));
      if (fs::create_directory(d, e)) {
        dir = d;
        break;
      }
      if (e) throw std::runtime_error(keptDir.u8string() + ": " + e.message());
    }
    if (dir.empty()) throw std::runtime_error("no free folder in " + keptDir.u8string());
    const std::string name = src.filename().u8string();
    try {
      std::string w;
      if (!writeWhole(dir / "filing.json", dump({{"name", name}, {"type", type}, {"root", root}}), w)) throw std::runtime_error(w);
      fs::create_directories(dir / "file");
      const fs::path to = dir / "file" / src.filename();
      fs::rename(src, to, e);
      if (e) {
        fs::copy_file(src, to);
        if (!syncPath(to)) throw std::runtime_error(to.u8string() + " not written to the disk");
        fs::remove(from / "meta.json", e);
        fs::remove(src, e);
        if (e) say("kept: " + name + " copied; the store's copy stays (" + e.message() + "), let go at a later open");
      }
      fs::remove(from / "meta.json", e);
      say("kept: " + name + " in " + dir.u8string());
    } catch (...) {
      fs::remove_all(dir, e);
      throw;
    }
    return root;
  }

  // ai: Every finished file in the store not yet taken out (store.h fileStore) taken into kept/, but those whose take
  // ai: failed since the last retry: the transfer the receiver holds, and one an earlier session or a cut-short close
  // ai: left. Each counted in the stats' kept (its root, or why its take failed). The lock's holder's alone.
  void takeFinished() {
    const fs::path dir = store / "lizard-xfer";
    std::error_code e;
    if (!lock || !fs::is_directory(dir, e)) return;
    std::vector<fs::path> ready;
    for (const auto& d : fs::directory_iterator(dir, e)) {
      std::error_code k;
      if (!d.is_directory(k) || fs::exists(d.path() / "header", k) || !fs::is_regular_file(d.path() / "meta.json", k)) continue;
      if (takeFailed.count(d.path().filename().u8string())) continue;
      bool file = false;
      for (const auto& f : fs::directory_iterator(d.path() / "done", k)) file = file || f.is_regular_file(k);
      if (file) ready.push_back(d.path());
    }
    for (const fs::path& d : ready) {
      try {
        const std::string root = take(d);
        std::lock_guard<std::mutex> l(keptMu);
        taken++;
        takenRoot = root;
        takeError.clear();
      } catch (const std::exception& x) {
        takeFailed.insert(d.filename().u8string());
        say(std::string("kept: the received file not taken out of the store: ") + x.what());
        const std::string root = text(parsed(slurp(d / "meta.json")), "root");
        std::lock_guard<std::mutex> l(keptMu);
        takenRoot = root;
        takeError = x.what();
      }
    }
  }

  // ai: The takes tried again (a start, a camera closed, the session's first moments and its end: where a take that
  // ai: failed is tried again, the old app's Start, leave and close): every finished file in the store, and kept/'s
  // ai: folders that hold no whole take let go (one whose file an app moved out and was cut short before it let the
  // ai: folder go: Session::kept lists none of them). The lock's holder's alone, on the worker: no take is under way.
  void retryTakes() {
    if (!lock) return;
    takeFailed.clear();
    takeFinished();
    std::error_code e;
    if (!fs::is_directory(keptDir, e)) return;
    std::vector<fs::path> gone;
    for (const auto& d : fs::directory_iterator(keptDir, e)) {
      std::error_code k;
      if (!d.is_directory(k)) continue;
      const std::string name = text(parsed(slurp(d.path() / "filing.json")), "name");
      if (name.empty() || !fs::is_regular_file(d.path() / "file" / fs::u8path(name), k)) gone.push_back(d.path());
    }
    for (const fs::path& d : gone) {
      fs::remove_all(d, e);
      say("kept: " + d.filename().u8string() + " let go (no file in it)");
    }
  }

  // ai: The poll's rules (the apps' before 2026-10-31), every TICK_MS between commands
  void tick() {
    if (camera) {
      try {
        readStats();
      } catch (const std::exception&) {
      }
    }
    // ai: a stream that ended by itself (a camera unplugged, a device's error, an upload played): the camera closed as
    // ai: well as stopped, so the next start opens it again by its id (one unplugged and plugged in again is a new
    // ai: device behind the same id); an upload's end is its file's, no error
    std::string end;
    {
      std::lock_guard<std::mutex> l(endMu);
      end = ended;
    }
    if (camera && !streaming && !end.empty()) {
      const bool upload = backend == "upload";
      const uint64_t g = genNow();
      const bool was = running();
      closeLocked();
      if (was) told(g, upload ? Phase::Idle : Phase::Error, upload ? "" : "The camera stopped: " + end);
    }
    {
      std::string pf;
      {
        std::lock_guard<std::mutex> l(pushMu);
        pf.swap(pushFailed);
      }
      if (!pf.empty() && !rxBroken) rxFailed(pf);
    }
    const auto r = rxBroken ? nullptr : current();
    json j = json::object();
    try {
      if (r) j = parsed(r->r->stats());
    } catch (const std::exception& e) {
      rxFailed(e.what());
      return;
    }
    const std::string st = text(j, "state");
    // ai: Loading to On once the receiver is past its first plan
    if (phaseNow() == Phase::Loading && streaming && r && !st.empty() && st != "starting" && st != "loading" && st != "error") told(genNow(), Phase::On);
    // ai: The receiver gone into its error while the camera reads (a plan that threw, no GPU variant left: it decodes
    // ai: nothing more of this stream): the camera stopped, the phase the error in the readout's words; its own in the
    // ai: log and its stats. A start goes on from there (an error kept from before it is no new one).
    if (st == "error" && lastState != "error" && streaming && running()) {
      say("receiver: " + text(j, "error") + "; the camera stopped");
      const uint64_t g = genNow();
      stopLocked();
      describeNow();
      told(g, Phase::Error, "Reading stopped. Start the camera again to go on.");
    }
    lastState = st;
    // ai: the autofocus region on the code once a second (the GPU's codeBox: camera pixels from the frame's centre), for
    // ai: a camera that focuses on a region (camera.h focusOn)
    if (camera && streaming && ++focusTicks % 10 == 0 && j.contains("codeBox") && j["codeBox"].is_array() && j["codeBox"].size() == 4) {
      double b[4];
      bool ok = true;
      for (size_t i = 0; i < 4; i++) {
        ok = ok && j["codeBox"][i].is_number();
        b[i] = ok ? j["codeBox"][i].get<double>() : 0;
      }
      if (ok) camera->focusOn(b);
    }
    const json f = j.contains("file") ? j["file"] : json();
    const std::string root = f.is_object() ? text(f, "root") : "";
    if (root != heldRoot) {
      heldRoot = root;
      stoppedFor = false;
      tookFor = false;
    }
    const auto flag = [](const json& o, const char* k) { return o.is_object() && o.contains(k) && o[k].is_boolean() && o[k].get<bool>(); };
    const bool verified = flag(f, "verified") && !root.empty() && !flag(j, "test");
    if (verified && r && !tookFor) {
      tookFor = true;
      takeFinished();
    }
    // ai: the camera off once the file is in, once a transfer: the receiver holds the verified file after the stop, so
    // ai: a start after it reads on
    if (verified && !stoppedFor) {
      stoppedFor = true;
      if (running() || streaming) {
        ask(Phase::Idle);
        stopLocked();
        describeNow();
        say("camera: stopped, the file is in");
      }
    }
    // ai: Start or Stop pressed on the phone (2026-10-27): followed here, the link open through both
    if (camera && backend == "phone") {
      cam::Stats s;
      {
        std::lock_guard<std::mutex> l(csMu);
        s = cs;
      }
      if (asksSeen >= 0 && int64_t(s.asks) > asksSeen) {
        if (s.askOn && !running()) doStart(ask(Phase::Starting), "");
        else if (!s.askOn && running()) {
          ask(Phase::Idle);
          doStop();
        }
      }
      asksSeen = int64_t(s.asks);
    }
    replayTick();
  }

  // ai: Save replays (the worker's)
  struct Run {
    std::string run;
    fs::path file;
    std::string state;   // ai: recording, saving, ready, failed
    int64_t frames = 0, bytes = 0;
    std::string error;
  };
  std::mutex runMu;
  std::vector<Run> runs;           // ai: runMu
  bool replaysOn = false;          // ai: the worker's: the switch
  bool replayRefused = false;      // ai: the worker's: a run not begun: none asked again until the switch moves
  std::string replayMeta;          // ai: the worker's: the platform's fields for the meta
  std::shared_ptr<Replay> recording;   // ai: the worker's
  std::string recordingRun;
  std::deque<std::string> rows;    // ai: the worker's
  int64_t lastRow = 0;
  std::thread ending;              // ai: a run's end (finish waits for the GPU's held frames); one at a time
  std::atomic<bool> replaysOnNow{false};   // ai: the switch, for the stats

  std::mutex extraMu;
  json extraJson = json::object();   // ai: extraMu

  void putRun(const Run& x) {
    std::lock_guard<std::mutex> l(runMu);
    for (Run& y : runs)
      if (y.run == x.run) {
        y = x;
        return;
      }
    runs.push_back(x);
  }

  // ai: the runs on disk at the session's start: the newest whole one (a .lizard, written whole at its run's end)
  // ai: listed, its frames from the file; the rest deleted, scratch folders and .part files with them
  void adopt() {
    std::error_code e;
    if (!fs::is_directory(replaysDir, e)) return;
    std::vector<fs::path> all;
    fs::path whole;
    for (const auto& d : fs::directory_iterator(replaysDir, e)) {
      all.push_back(d.path());
      const std::string n = d.path().filename().u8string();
      if (d.is_regular_file(e) && n.size() > 7 && n.compare(n.size() - 7, 7, ".lizard") == 0 && (whole.empty() || n > whole.filename().u8string()))
        whole = d.path();
    }
    if (!whole.empty()) {
      Run x;
      x.file = whole;
      x.run = whole.stem().u8string();
      x.state = "ready";
      x.bytes = int64_t(fs::file_size(whole, e));
      std::string w;
      if (auto rd = lizfile::Reader::open(whole.u8string(), w)) {
        x.frames = int64_t(rd->frames().size());
        x.error = text(parsed(rd->meta()), "error");
      }
      putRun(x);
    }
    for (const auto& p : all)
      if (p != whole) fs::remove_all(p, e);
  }

  // ai: what the session adds to a run's meta.json: the camera, its settings, the receiver's; then the platform's
  json sessionMeta() {
    json cam = nullptr;
    {
      std::lock_guard<std::mutex> l(ptrMu);
      if (info.is_object()) {
        cam = info;
        if (modeNow.is_object()) {
          cam["format"] = modeNow.value("format", "");
          cam["size"] = std::to_string(modeNow.value("w", 0)) + "x" + std::to_string(modeNow.value("h", 0));
          cam["fps"] = modeNow.value("fps", 0.0);
        }
      }
    }
    const auto valueOf = [&](const char* id) -> json {
      for (const Want& w : wants)
        if (w.id == id) return w.automatic ? json() : json(w.value);
      if (std::string(id) == "zoom")
        if (const cam::Control* c = control("zoom")) return c->value;
      return nullptr;
    };
    const auto r = current();
    int capNow = 0;
    {
      std::lock_guard<std::mutex> l(rxMu);
      capNow = cap;
    }
    if (cam.is_object() && camera) {
      try {
        cam["report"] = parsed(camera->report());
      } catch (const std::exception&) {
      }
    }
    json m = {{"camera", cam}, {"zoom", valueOf("zoom")}, {"exposure", valueOf("exposure")}, {"gain", valueOf("gain")},
              {"layout", r ? r->layout : wantLayout}, {"decoder", wantDecoder}, {"batch", capNow}, {"phase", lockJson()}};
    {
      // ai: the platform's fields as they are at the run's end (the phone's heat)
      std::lock_guard<std::mutex> l(extraMu);
      for (auto it = extraJson.begin(); it != extraJson.end(); ++it) m[it.key()] = it.value();
    }
    const json p = parsed(replayMeta);
    for (auto it = p.begin(); it != p.end(); ++it) m[it.key()] = it.value();
    return m;
  }

  void beginReplay() {
    const auto r = current();
    if (!r) return;
    using namespace std::chrono;
    int64_t t = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::string run;
    std::error_code e;
    for (;;) {
      const std::time_t s = std::time_t(t / 1000);
      std::tm tm = *std::gmtime(&s);
      char b[64];
      std::snprintf(b, sizeof b, "run-%04d-%02d-%02dT%02d-%02d-%02d-%03dZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, int(t % 1000));
      run = b;
      bool taken0 = fs::exists(replaysDir / fs::u8path(run + ".lizard"), e);
      {
        std::lock_guard<std::mutex> l(runMu);
        taken0 = taken0 || std::any_of(runs.begin(), runs.end(), [&](const Run& x) { return x.run == run; });
      }
      if (!taken0) break;
      t++;
    }
    fs::create_directories(replaysDir, e);
    const fs::path file = replaysDir / fs::u8path(run + ".lizard");
    std::shared_ptr<Replay> rp;
    std::string w;
    try {
      rp = std::make_shared<Replay>(file.u8string(), run, REPLAY_FRAMES);
      if (!rp->live()) w = rp->error().empty() ? "not begun" : rp->error();
    } catch (const std::exception& x) {
      w = x.what();
    }
    Run x;
    x.run = run;
    x.file = file;
    if (!w.empty()) {
      replayRefused = true;
      x.state = "failed";
      x.error = "not begun: " + w;
      putRun(x);
      say("replay: " + run + " not begun: " + w);
      return;
    }
    {
      std::lock_guard<std::mutex> l(rxMu);
      rec = rp;
      if (rx) rx->r->record(rp);
    }
    recording = rp;
    recordingRun = run;
    rows.clear();
    lastRow = nowNs();
    x.state = "recording";
    putRun(x);
    say("replay: " + run);
  }

  // ai: The run's end: no more frames to it, then on a thread of its own the file written (Replay::finish waits up to
  // ai: a few seconds for frames the GPU decoder still holds for it); a run that ends with frames lets the finished runs
  // ai: before it go, a failed one goes itself. wait: here, at the session's end.
  void endReplay(bool wait) {
    if (!recording) return;
    {
      std::lock_guard<std::mutex> l(rxMu);
      rec.reset();
      if (rx) rx->r->record(nullptr);
    }
    std::shared_ptr<Replay> rp = std::move(recording);
    const std::string run = recordingRun;
    recording.reset();
    Run x;
    {
      std::lock_guard<std::mutex> l(runMu);
      for (const Run& y : runs)
        if (y.run == run) x = y;
    }
    x.state = "saving";
    putRun(x);
    const auto r = current();
    const std::string last = r ? r->r->stats() : std::string("{}");
    std::string rowsNow;
    for (const auto& row : rows) rowsNow += (rowsNow.empty() ? "" : "\n") + row;
    rows.clear();
    const std::string meta = dump(sessionMeta());
    if (ending.joinable()) ending.join();
    ending = std::thread([this, rp, x, last, rowsNow, meta]() mutable {
      int n = -1;
      try {
        n = rp->finish(last, rowsNow, meta);
      } catch (const std::exception& e) {
        x.error = std::string("the replay could not be finished: ") + e.what();
      }
      if (n > 0) {
        x.state = "ready";
        x.frames = n;
        x.bytes = rp->bytes();
        x.error = rp->error();
        say("replay: " + x.run + " " + std::to_string(n) + " frames, " + std::to_string(x.bytes) + " B");
      } else {
        x.state = "failed";
        if (x.error.empty()) x.error = rp->error().empty() ? "no frames" : rp->error();
        say("replay: " + x.run + " failed: " + x.error);
      }
      std::vector<Run> gone;
      {
        std::lock_guard<std::mutex> l(runMu);
        for (Run& y : runs)
          if (y.run == x.run) y = x;
        for (const Run& y : runs)
          if (x.state == "ready" ? (y.run != x.run && (y.state == "ready" || y.state == "failed")) : y.run == x.run) gone.push_back(y);
        runs.erase(std::remove_if(runs.begin(), runs.end(), [&](const Run& y) {
          return std::any_of(gone.begin(), gone.end(), [&](const Run& g) { return g.run == y.run; });
        }), runs.end());
      }
      std::error_code e;
      for (const Run& g : gone)
        for (const std::string& p : {g.file.u8string(), g.file.u8string() + ".frames", g.file.u8string() + ".part"}) fs::remove_all(fs::u8path(p), e);
    });
    if (wait) ending.join();
  }

  void replayTick() {
    const bool want = replaysOn && phaseNow() == Phase::On && backend != "upload" && current();
    if (want && !recording && !replayRefused) beginReplay();
    else if (!want && recording) endReplay(false);
    if (recording && nowNs() - lastRow >= 1'000'000'000) {
      lastRow = nowNs();
      // ai: a row is the stats with the lock's state under "phase" (the Android app's rows' key; the session's own
      // ai: phase is "on" while a run records)
      json row = parsed(stats());
      row["phase"] = lockJson();
      rows.push_back(dump(row));
      while (rows.size() > REPLAY_ROWS) rows.pop_front();
    }
  }

  // ai: the stats
  std::string stats() {
    const auto r = current();
    json j = json::object();
    if (r) {
      try {
        j = json::parse(r->r->stats());
      } catch (const std::exception& e) {
        j = {{"state", "error"}, {"error", std::string("the receiver's stats: ") + e.what()}};
      }
      if (!j.is_object()) j = json::object();
    }
    std::shared_ptr<cam::Camera> c;
    json o, m;
    {
      std::lock_guard<std::mutex> l(ptrMu);
      c = camera;
      o = info;
      m = modeNow;
    }
    // ai: the receiver's window is a stream's: nothing of it while none runs (after a stream's own end the receiver
    // ai: keeps its last window until the camera is closed); from a start the receiver's own figures, 0 until the
    // ai: stream's first window closes (each stop ends the receiver's window: receiver.h cameraClosed); the transfer,
    // ai: the word and the totals stay
    const bool live = c && streaming.load();
    if (!live) {
      for (const char* k : WINDOW_FIGURES)
        if (j.contains(k)) j[k] = 0;
      if (j.contains("test")) j["test"] = false;
    }
    j["live"] = live;
    {
      std::lock_guard<std::mutex> l(phMu);
      static const char* const NAMES[] = {"idle", "starting", "loading", "on", "error"};
      j["phase"] = NAMES[int(phase)];
      j["why"] = why;
    }
    {
      std::lock_guard<std::mutex> l(askedMu);
      j["asked"] = askedJson;
    }
    const std::string dec = text(j, "decoder");
    j["autoRan"] = r && r->decoder == "auto" && !dec.empty() ? (dec.rfind("gpu", 0) == 0 ? "GPU" : "CPU") : "";
    {
      std::lock_guard<std::mutex> l(keptMu);
      j["kept"] = {{"taken", taken}, {"root", takenRoot}, {"error", takeError}};
    }
    {
      json rs = json::array();
      std::lock_guard<std::mutex> l(runMu);
      for (const Run& x : runs)
        rs.push_back({{"run", x.run}, {"path", x.file.u8string()}, {"state", x.state}, {"frames", x.frames}, {"bytes", x.bytes}, {"error", x.error}});
      j["replays"] = std::move(rs);
      j["replaysOn"] = replaysOnNow.load();
    }
    j["lock"] = lockJson();
    {
      std::lock_guard<std::mutex> l(extraMu);
      for (auto it = extraJson.begin(); it != extraJson.end(); ++it) j[it.key()] = it.value();
    }
    if (!c) {
      j["camera"] = nullptr;
      return dump(j);
    }
    cam::Stats s;
    {
      std::lock_guard<std::mutex> l(csMu);
      s = cs;
    }
    const bool on = streaming.load();
    if (!o.is_object()) o = json::object();
    o["streaming"] = on;
    o["starts"] = starts.load();
    o["mode"] = m;
    o["fps"] = on ? fpsNow() : 0.0;
    o["agreedFps"] = s.fps;
    o["frames"] = s.frames;
    o["dropped"] = s.dropped;
    o["corrupt"] = s.corrupt;
    o["convertMs"] = s.convertMs;
    o["asks"] = s.asks;
    o["askOn"] = s.askOn;
    {
      std::lock_guard<std::mutex> l(endMu);
      o["ended"] = ended;
    }
    try {
      o["report"] = parsed(c->report());
    } catch (const std::exception&) {
      o["report"] = json::object();
    }
    j["camera"] = std::move(o);
    return dump(j);
  }

  // ai: the worker
  void loop() {
    // ai: the lock first (a session of this process closing on the folder waited for), then the replays adopted and
    // ai: what an earlier session left untaken taken out
    std::string w;
    if (haveLock(10'000, w)) {
      adopt();
      retryTakes();
    } else {
      say("session: " + w);
    }
    auto last = std::chrono::steady_clock::now();
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> l(qMu);
        qCv.wait_until(l, last + std::chrono::milliseconds(TICK_MS), [&] { return !q.empty() || quit; });
        if (!q.empty()) {
          f = std::move(q.front());
          q.pop_front();
        } else if (quit) {
          break;
        }
      }
      if (f) f();
      if (std::chrono::steady_clock::now() - last >= std::chrono::milliseconds(TICK_MS)) {
        last = std::chrono::steady_clock::now();
        try {
          tick();
        } catch (const std::exception& e) {
          say(std::string("session: the rules: ") + e.what());
        }
      }
    }
  }

  // ai: the session's end, on the worker after every command: the rules once more (a file verified since the last tick
  // ai: taken into kept/; no replay begun) and the takes tried again, the camera (no frame after its stop), a recording
  // ai: ended and written, then the receiver, the lock last
  void teardown() {
    replaysOn = false;
    try {
      tick();
      retryTakes();
    } catch (const std::exception& e) {
      say(std::string("session: the rules: ") + e.what());
    }
    try {
      closeLocked();
    } catch (const std::exception& e) {
      say(std::string("camera: the close: ") + e.what());
    }
    endReplay(true);
    if (ending.joinable()) ending.join();
    std::shared_ptr<Rx> r;
    {
      std::lock_guard<std::mutex> l(rxMu);
      r = std::move(rx);
      rec.reset();
    }
    letGo(r);
#if defined(__ANDROID__)
    if (previewWin) ANativeWindow_release(static_cast<ANativeWindow*>(previewWin));
#endif
    previewWin = nullptr;
    lock.reset();
  }
};

std::string describeCamera(cam::Camera& c) { return dump(describe(c, c.modes(), c.controls())); }

Session::Session(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

std::unique_ptr<Session> Session::create(const Config& c, std::string& why) {
  if (!oneOf(c.decoder, LIZ_DECODERS, "decoder", why) || !oneOf(c.layout, {"1:1", "2:1"}, "layout", why)) return nullptr;
  if (c.dataDir.empty()) {
    why = "the session needs a data folder";
    return nullptr;
  }
  auto p = std::make_unique<Impl>();
  p->cfg = c;
  p->data = fs::u8path(c.dataDir);
  p->store = p->data / "store";
  p->keptDir = p->data / "kept";
  p->replaysDir = p->data / "replays";
  std::error_code e;
  fs::create_directories(p->store, e);
  if (e) {
    why = "the data folder " + c.dataDir + ": " + e.message();
    return nullptr;
  }
  if (!c.cacheDir.empty()) fs::create_directories(fs::u8path(c.cacheDir), e);
  Impl* raw = p.get();
  p->settings = std::make_unique<Settings>(p->data / "cameras.json", !c.forgetSettings, [raw](const std::string& m) { raw->say(m); });
  p->wantDecoder = c.decoder;
  p->wantLayout = c.layout;
  p->noteAsked();
  p->phaseLock = std::make_unique<PhaseLockHost>(
      [raw](const std::string& m) { raw->say(m); },
      [raw](double since) {
        try {
          const auto r = raw->current();
          return r ? r->r->series(since) : std::vector<double>{0};
        } catch (const std::exception&) {
          return std::vector<double>{0};
        }
      },
      [raw](int64_t us) {
        cam::Camera* c = raw->lockCam.load();
        return c && c->delayOnce(us);
      });
  p->worker = std::thread([raw] { raw->loop(); });
  return std::unique_ptr<Session>(new Session(std::move(p)));
}

Session::~Session() {
  Impl& p = *p_;
  // ai: the folder marked closing: a delete with no session waits for this one's end rather than being refused
  if (p.holding) {
    {
      std::lock_guard<std::mutex> l(gHeldMu);
      gClosing.insert(folderKey(p.data.u8string()));
    }
    gHeldCv.notify_all();
  }
  p.post([&p] {
    p.teardown();
    return std::string();
  });
  {
    std::lock_guard<std::mutex> l(p.qMu);
    p.quit = true;
  }
  p.qCv.notify_one();
  if (p.worker.joinable()) p.worker.join();
}

Session::Done Session::prepare() {
  Impl& p = *p_;
  return p.post([&p] {
    std::string w;
    if (!p.haveLock(0, w)) return w;
    p.haveRx(w);
    return w;
  });
}

Session::Done Session::open(const std::string& id) {
  Impl& p = *p_;
  const uint64_t g = p.genNow();
  return p.post([&p, g, id] { return p.doOpen(g, id); });
}

Session::Done Session::close() {
  Impl& p = *p_;
  return p.post([&p] {
    p.closeLocked();
    p.retryTakes();
    return std::string();
  });
}

Session::Done Session::set(const std::string& control, double value, bool automatic) {
  Impl& p = *p_;
  std::unique_lock<std::mutex> l(p.qMu);
  auto it = p.pendingSets.find(control);
  if (it != p.pendingSets.end()) {
    it->second.value = value;
    it->second.automatic = automatic;
    return it->second.fut;
  }
  Impl::PendingSet ps;
  ps.value = value;
  ps.automatic = automatic;
  ps.done = std::make_shared<std::promise<std::string>>();
  ps.fut = ps.done->get_future().share();
  Done fut = ps.fut;
  if (p.quit) {
    ps.done->set_value("the session is closing");
    return fut;
  }
  p.pendingSets[control] = ps;
  p.q.push_back([&p, control] {
    Impl::PendingSet x;
    {
      std::lock_guard<std::mutex> k(p.qMu);
      x = p.pendingSets[control];
      p.pendingSets.erase(control);
    }
    x.done->set_value(p.run([&] { return p.doSet(control, x.value, x.automatic); }));
  });
  l.unlock();
  p.qCv.notify_one();
  return fut;
}

Session::Done Session::start(const std::string& mode) {
  Impl& p = *p_;
  const uint64_t g = p.ask(Impl::Phase::Starting);
  return p.post([&p, g, mode] { return p.doStart(g, mode); });
}

Session::Done Session::stop() {
  Impl& p = *p_;
  p.ask(Impl::Phase::Idle);
  return p.post([&p] { return p.doStop(); });
}

Session::Done Session::reconfigure(const std::string& decoder, const std::string& layout) {
  Impl& p = *p_;
  std::string why;
  if (!oneOf(decoder, LIZ_DECODERS, "decoder", why) || !oneOf(layout, {"1:1", "2:1"}, "layout", why)) {
    std::promise<std::string> pr;
    pr.set_value(why);
    return pr.get_future().share();
  }
  uint64_t g;
  {
    // ai: a camera asked to run reads Loading until the new receiver is past its first plan (a start under way stays
    // ai: Starting)
    std::lock_guard<std::mutex> l(p.phMu);
    if (p.phase == Impl::Phase::On) p.phase = Impl::Phase::Loading;
    g = p.gen;
  }
  return p.post([&p, g, decoder, layout] { return p.doReconfigure(g, decoder, layout); });
}

Session::Done Session::batchCap(int n) {
  Impl& p = *p_;
  return p.post([&p, n] {
    try {
      std::lock_guard<std::mutex> l(p.rxMu);
      p.cap = n;
      if (p.rx) p.rx->r->batchCap(n);
    } catch (const std::exception& e) {
      p.rxFailed(e.what());
      return std::string(e.what());
    }
    return std::string();
  });
}

Session::Done Session::forget(const std::string& root) {
  Impl& p = *p_;
  return p.post([&p, root] { return p.doForget(root); });
}

Session::Done Session::clear() {
  Impl& p = *p_;
  return p.post([&p] {
    if (const auto r = p.current()) r->r->clear();
    p.heldRoot.clear();
    p.stoppedFor = false;
    p.tookFor = false;
    return std::string();
  });
}

Session::Done Session::replays(bool on, const std::string& meta) {
  Impl& p = *p_;
  p.replaysOnNow = on;
  return p.post([&p, on, meta] {
    p.replaysOn = on;
    p.replayMeta = meta;
    p.replayRefused = false;
    return std::string();
  });
}

void Session::extra(const std::string& j) {
  const json x = parsed(j);
  std::lock_guard<std::mutex> l(p_->extraMu);
  p_->extraJson = x;
}

// ai: the hooks' refusal off Android
[[maybe_unused]] const char* const NOT_HERE = "the phone's link and preview surface are Android's";

Session::Done Session::previewWindow(void* w) {
  Impl& p = *p_;
#if defined(__ANDROID__)
  // ai: the session's own reference taken on the worker, where the hand-over runs: the caller holds its own until the
  // ai: outcome (lizard.h), and a session closing runs nothing more, so none is taken that is not let go
  return p.post([&p, w] {
    if (w) ANativeWindow_acquire(static_cast<ANativeWindow*>(w));
    if (p.previewWin) ANativeWindow_release(static_cast<ANativeWindow*>(p.previewWin));
    p.previewWin = w;
    if (p.camera) p.camera->previewTo(w);
    return std::string();
  });
#else
  (void)p;
  (void)w;
  std::promise<std::string> pr;
  pr.set_value(NOT_HERE);
  return pr.get_future().share();
#endif
}

#if defined(__ANDROID__)
// ai: the PC link's log (no session's): the system log, the sessions' tag
static void linkLog(const std::string& m) { __android_log_write(ANDROID_LOG_INFO, "lizard", m.c_str()); }
#endif

bool Session::link(bool up, const std::string& layout, std::string& why) {
#if defined(__ANDROID__)
  (void)why;
  if (up) linkUp(layout, linkLog);
  else linkDown();
  return true;
#else
  (void)up;
  (void)layout;
  why = NOT_HERE;
  return false;
#endif
}

bool Session::linkPack(bool on, std::string& why) {
#if defined(__ANDROID__)
  (void)why;
  lizard::linkPack(on);
  return true;
#else
  (void)on;
  why = NOT_HERE;
  return false;
#endif
}

std::string Session::linkState() {
#if defined(__ANDROID__)
  json j = parsed(lizard::linkState());
  j["accessory"] = accessoryOpen();
  return dump(j);
#else
  return "{}";
#endif
}

bool Session::usbAccessory(int fd, std::string& why) {
#if defined(__ANDROID__)
  if (streamAccessory(fd, linkLog)) return true;
  why = "a USB accessory's bridge is up already";
  return false;
#else
  (void)fd;
  why = NOT_HERE;
  return false;
#endif
}

std::string Session::stats() { return p_->stats(); }

std::string Session::camera() {
  json j;
  {
    std::lock_guard<std::mutex> l(p_->ptrMu);
    j = p_->described.is_object() ? p_->described : json::object();
  }
  return dump(j);
}

std::vector<double> Session::series(double sinceMs) {
  try {
    const auto r = p_->current();
    return r ? r->r->series(sinceMs) : std::vector<double>{0};
  } catch (const std::exception&) {
    return {0};
  }
}

bool Session::streaming() const { return p_->streaming.load(); }

bool Session::preview(uint64_t since, int side, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, uint64_t& seq) {
  Impl& p = *p_;
  const int64_t now = nowNs();
  p.asked.store(now, std::memory_order_relaxed);
  Impl::Held held{p.prevMu, nullptr};
  {
    std::lock_guard<std::mutex> l(p.prevMu);
    held.c = p.newest;
  }
  const Impl::Crop* c = held.c.get();
  // ai: a crop kept from before a pause in the asking (no copies were made meanwhile) is not the newest frame's
  if (!c || c->seq <= since || now - c->at > PREVIEW_HOLD_NS || side <= 0 || !c->w || !c->h) return false;
  const uint32_t lon = std::max(c->w, c->h), s = uint32_t(side);
  if (lon <= s) {
    w = c->w;
    h = c->h;
  } else {
    w = std::clamp<uint32_t>(uint32_t(std::lround(double(c->w) * s / lon)), 1, s);
    h = std::clamp<uint32_t>(uint32_t(std::lround(double(c->h) * s / lon)), 1, s);
  }
  out.resize(size_t(w) * h);
  {
    std::lock_guard<std::mutex> l(p.scaleMu);
    shrink(c->px.data(), c->w, c->h, out.data(), w, h, p.scaleMid, p.scaleAcc);
  }
  seq = c->seq;
  return true;
}

std::string Session::cameras() {
  std::vector<std::string> notes;
  json cs = json::array();
  try {
    for (const auto& i : cam::list(&notes)) cs.push_back(infoJson(i));
  } catch (const std::exception& e) {
    notes.push_back(std::string("the cameras could not be listed: ") + e.what());
  }
  return dump({{"cameras", cs}, {"notes", notes}});
}

std::string Session::camerasStamp() {
  try {
    return cam::listStamp();
  } catch (const std::exception&) {
    return "";
  }
}

std::string Session::kept(const std::string& dataDir) {
  json out = json::array();
  const fs::path dir = fs::u8path(dataDir) / "kept";
  std::error_code e;
  std::vector<std::pair<int64_t, json>> takes;
  if (fs::is_directory(dir, e))
    for (const auto& d : fs::directory_iterator(dir, e)) {
      if (!d.is_directory(e)) continue;
      const json m = parsed(slurp(d.path() / "filing.json"));
      const std::string name = text(m, "name");
      if (name.empty()) continue;
      const fs::path f = d.path() / "file" / fs::u8path(name);
      if (!fs::is_regular_file(f, e)) continue;   // ai: placed already, or a take not yet whole
      const int64_t at = msOf(fs::last_write_time(d.path(), e));
      takes.emplace_back(at, json{{"id", d.path().filename().u8string()}, {"name", name}, {"type", text(m, "type")}, {"root", text(m, "root")},
                                  {"path", f.u8string()}, {"at", at}});
    }
  std::stable_sort(takes.begin(), takes.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  for (auto& t : takes) out.push_back(std::move(t.second));
  return dump(out);
}

bool Session::placed(const std::string& dataDir, const std::string& id, std::string& why) {
  if (!plainName(id)) {
    why = "no take " + id;
    return false;
  }
  std::error_code e;
  fs::remove_all(fs::u8path(dataDir) / "kept" / fs::u8path(id), e);
  if (e) why = e.message();
  return !e;
}

std::string Session::partials(const std::string& dataDir) {
  const fs::path dir = fs::u8path(dataDir) / "store" / "lizard-xfer";
  std::error_code e;
  std::vector<std::pair<int64_t, json>> ps;
  if (fs::is_directory(dir, e))
    for (const auto& d : fs::directory_iterator(dir, e)) {
      if (!d.is_directory(e) || !fs::is_regular_file(d.path() / "header", e)) continue;
      const fs::path mf = d.path() / "meta.json";
      if (!fs::is_regular_file(mf, e)) continue;
      json m;
      try {
        m = json::parse(slurp(mf));
      } catch (const std::exception&) {
        continue;
      }
      if (!m.is_object()) continue;
      const int64_t at = msOf(fs::last_write_time(mf, e));
      const auto number = [&](const char* k) { return m.contains(k) && m[k].is_number() ? m[k].get<double>() : 0.0; };
      ps.emplace_back(at, json{{"id", d.path().filename().u8string()}, {"name", text(m, "name")}, {"length", number("length")},
                               {"fraction", number("fraction")}, {"root", text(m, "root")}, {"at", at}});
    }
  std::stable_sort(ps.begin(), ps.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  json out = json::array();
  for (auto& p : ps) out.push_back(std::move(p.second));
  return dump(out);
}

bool Session::deletePartial(const std::string& dataDir, const std::string& id, std::string& why) {
  if (!plainName(id)) {
    why = "no transfer " + id;
    return false;
  }
  // ai: a session of this process on the folder that is closing waited for (one an app let go a moment ago closes on
  // ai: its own thread: its camera, a replay's end, the receiver); one still receiving, or another process's, refuses
  // ai: at once
  const auto held = takeLock(dataDir, 0, why, 10'000);
  if (!held) return false;
  std::error_code e;
  fs::remove_all(fs::u8path(dataDir) / "store" / "lizard-xfer" / fs::u8path(id), e);
  if (e) why = e.message();
  return !e;
}

}  // namespace lizard

// ai: The system's file lock (DataLock), here after everything else so windows.h's macros reach nothing above.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace lizard {
namespace {
int osLock(const std::string& path, intptr_t& handle) {
  const std::wstring w = std::filesystem::u8path(path).wstring();
  HANDLE f = CreateFileW(w.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov{};
  if (LockFileEx(f, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &ov)) {
    handle = reinterpret_cast<intptr_t>(f);
    return 1;
  }
  const DWORD e = GetLastError();
  CloseHandle(f);
  return e == ERROR_LOCK_VIOLATION || e == ERROR_IO_PENDING ? 0 : -1;
}
void osUnlock(intptr_t handle) {
  HANDLE f = reinterpret_cast<HANDLE>(handle);
  OVERLAPPED ov{};
  UnlockFileEx(f, 0, MAXDWORD, MAXDWORD, &ov);
  CloseHandle(f);
}
}  // namespace
}  // namespace lizard
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace lizard {
namespace {
int osLock(const std::string& path, intptr_t& handle) {
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return -1;
  struct flock fl {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;
  if (fcntl(fd, F_SETLK, &fl) == 0) {
    handle = fd;
    return 1;
  }
  const int e = errno;
  ::close(fd);
  return e == EACCES || e == EAGAIN ? 0 : -1;
}
void osUnlock(intptr_t handle) { ::close(int(handle)); }
}  // namespace
}  // namespace lizard
#endif
