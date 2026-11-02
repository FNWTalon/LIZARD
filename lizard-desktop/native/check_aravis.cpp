// ai: cam_check aravis (2026-11-01, checks.h): the GenICam backend (cam/aravis.cpp) against lizard_fakegv
// ai: (sim/fake_gv.cpp: Aravis's GigE fake on loopback, a process of its own beside cam_check), through Aravis loaded at
// ai: run time (cam/aravis_api.h). Aravis's USB3Vision interface is turned off before anything lists, so no USB camera
// ai: is enumerated or opened, and only the fake's id is opened (a GigE camera elsewhere on the network may be listed,
// ai: never opened). Cases:
// ai:   load       Aravis loaded, its version 0.8;
// ai:   list       the fake (the gradient) listed by the backend within 6 s (a process's first list waits for its
// ai:              first discovery), every later list answering under 100 ms (the last discovery's), the stamp holding
// ai:              its id;
// ai:   open       by its id through the registry (camera.h open); its modes the fake's seven formats at its sensor's
// ai:              size, Mono8 first, each at the bounds' top (1000), 120, 60 and 30; pick's for 1920 x 1080 at 60
// ai:              Mono8 at 60;
// ai:   controls   exposure (us) and gain listed; a set on the stopped camera put back at the next stop; a start
// ai:              changing neither; sets while streaming in force (the fake's stamp carries the exposure and gain it
// ai:              filled with), put back at the stop;
// ai:   free 60, free 120  Mono8 free run for 3 s: Stats::fps the rate read back; every fill the fake made (its stamp's
// ai:              count) handed over, none dropped or corrupt; ns rising, each frame's ns its stamp's time plus one
// ai:              offset (the device's time moved onto steadyNs), the intervals a period at the median; every fourth
// ai:              frame's luma the gradient at its stamp's time, exactly; the rate within 3% where the fake made it,
// ai:              else a skip line with the rate the fake made (its fill and send share one thread);
// ai:   formats    the fake's other formats (Mono16, the four Bayer8, RGB8) for a second each: frames, every fourth
// ai:              one's luma the gradient's exactly (Mono16 and RGB8 at the stamp's time; Bayer8 the gradient's 2 x 2
// ai:              mean at the phase that fits, the stamp blurred by the mean);
// ai:   trigger    the phase lock's lever on the same camera (camera.h delayBar, captures, delayOnce): the bar "";
// ai:              the schedule begun on its own once the stream runs; each Capture at a frame's own time, fresh on
// ai:              the first; the deadlines a period (1 / 60 less 0.1%) apart but one: a 5 ms delay asked at the
// ai:              120th capture, its interval the period and 5 ms, marked delayed, the fake's answers (its stamps)
// ai:              as far after their deadlines before it as after (the trigger moved, not only the times); no fate
// ai:              told; the schedule's figures sane (none lost or refused, under 2 ms late, a latency); then a free
// ai:              run at the rate (the trigger put back at the stop);
// ai:   session    a fake filming the simulated screen (check_session's SMALL: 960 x 540, LIZARD-128 in the 128 ring,
// ai:              30 a second on a 60 Hz display), opened by an id the last discovery did not hold (Aravis's own
// ai:              search), into the engine's session on the CPU decoder (LIZ_VK_DEVICE=llvmpipe): the test stream
// ai:              read, blocks verified;
// ai:   lock       the session's phase lock on the trigger from a bad start (the fake's --start-phase 0.1: a capture a
// ai:              tenth into each picture, which reads near nothing; the screen's clock as fast as the schedule,
// ai:              ppm 1000, so a schedule left alone stays there), 14 s with the lock and 14 s with LIZ_PHASE=off,
// ai:              judged over each run's last 8 s: the hold stood, delays asked and landing, captures clean (both
// ai:              pilots' readings at least 0.8, the receiver's series) in 30 points more of the frames than with the
// ai:              lock off, and blocks a second at least twice the run with it off and 400 (2026-11-01, this machine:
// ai:              45 to 100% clean and 660 to 896 blocks a second with the lock over eight runs, 0 and 0 off; the lock's
// ai:              delays 21 to 146 in 14 s, its pace wandering about the schedule's 0);
// ai:   lock 120   the camera at 120 against the 60 Hz screen, lock on, 10 s: the schedule at 8.325 ms, under 1% of
// ai:              its triggers lost, and the stream reading; what the lock makes of it said, not judged (it takes
// ai:              the camera's frame for the refresh; every picture captured twice).
// ai: Where Aravis does not load and LIZ_ARAVIS is not set (the receiver does without it), or LIZ_ARAVIS is set and
// ai: empty (off), every case is a skip line with why and the check holds (kHeld: Aravis is every PC system's, so not
// ai: kNotHere); where LIZ_ARAVIS names a library that does not load, load fails. A build without lizard_fakegv (no
// ai: Aravis headers at configure: LIZ_ARAVIS_ROOT) skips the cases after load, and fails them where LIZ_ARAVIS is set.
// ai: No arguments. About 75 s.
#include "checks.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "cam/aravis_api.h"
#include "cam/backends.h"
#include "cam/camera.h"
#include "cam/clock.h"
#include "json.hpp"
#include "session.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
extern char** environ;
#endif

#ifndef LIZ_ASSETS
#define LIZ_ASSETS "liblizard/out"
#endif

namespace lizard::check {

namespace {

namespace fs = std::filesystem;
using namespace lizard::cam;

#if defined(__MINGW32__)
std::string fmt(const char* f, ...) __attribute__((format(gnu_printf, 1, 2)));
#else
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
  char b[2048];
  va_list a;
  va_start(a, f);
  std::vsnprintf(b, sizeof b, f, a);
  va_end(a);
  return b;
}

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

const char* const kCases[] = {"load", "list", "open", "controls", "free 60", "free 120", "formats", "trigger", "session", "lock", "lock 120"};

void setEnv(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}
void unsetEnv(const char* k) {
#if defined(_WIN32)
  _putenv_s(k, "");
#else
  unsetenv(k);
#endif
}
void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// ai: lizard_fakegv beside this executable
fs::path fakePath() {
#if defined(_WIN32)
  std::wstring p(32768, L'\0');
  const DWORD n = GetModuleFileNameW(nullptr, p.data(), DWORD(p.size()));
  p.resize(n);
  return fs::path(p).parent_path() / "lizard_fakegv.exe";
#elif defined(__APPLE__)
  char b[4096];
  uint32_t n = sizeof b;
  if (_NSGetExecutablePath(b, &n) != 0) return {};
  return fs::path(b).parent_path() / "lizard_fakegv";
#else
  std::error_code e;
  const fs::path self = fs::read_symlink("/proc/self/exe", e);
  return e ? fs::path() : self.parent_path() / "lizard_fakegv";
#endif
}

// ai: A lizard_fakegv process: started, its "ready <id>" line waited for (10 s), ended by SIGTERM (TerminateProcess on
// ai: Windows), killed where it has not gone in 3 s
class Fake {
 public:
  ~Fake() { end(); }
  bool start(const fs::path& exe, const std::vector<std::string>& args, std::string& why) {
    std::string line;
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, &sa, 0)) { why = "no pipe"; return false; }
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    std::wstring cmd = L"\"" + exe.wstring() + L"\"";
    for (const auto& a : args) cmd += L" \"" + fs::u8path(a).wstring() + L"\"";
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = w;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    const BOOL made = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(w);
    if (!made) { CloseHandle(r); why = "CreateProcess failed (" + std::to_string(GetLastError()) + ")"; return false; }
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    out_ = r;
    for (const auto t0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10);) {
      DWORD avail = 0;
      if (!PeekNamedPipe(out_, nullptr, 0, nullptr, &avail, nullptr)) break;
      if (!avail) { sleepMs(20); continue; }
      char c = 0;
      DWORD got = 0;
      if (!ReadFile(out_, &c, 1, &got, nullptr) || !got) break;
      if (c == '\r') continue;
      if (c != '\n') { line += c; continue; }
      if (line.rfind("ready ", 0) == 0) { id = line.substr(6); return true; }
      line.clear();
    }
#else
    int fd[2];
    if (pipe(fd) != 0) { why = "no pipe"; return false; }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
    posix_spawn_file_actions_addclose(&fa, fd[0]);
    posix_spawn_file_actions_addclose(&fa, fd[1]);
    std::vector<std::string> all{exe.string()};
    all.insert(all.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& s : all) argv.push_back(s.data());
    argv.push_back(nullptr);
    const int rc = posix_spawn(&pid_, exe.c_str(), &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fd[1]);
    if (rc != 0) { close(fd[0]); pid_ = -1; why = std::string("posix_spawn: ") + std::strerror(rc); return false; }
    out_ = fd[0];
    for (const auto t0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10);) {
      pollfd p{out_, POLLIN, 0};
      if (poll(&p, 1, 100) <= 0) continue;
      char c = 0;
      if (read(out_, &c, 1) != 1) break;
      if (c != '\n') { line += c; continue; }
      if (line.rfind("ready ", 0) == 0) { id = line.substr(6); return true; }
      line.clear();
    }
#endif
    const int code = end();
    why = "lizard_fakegv gave no ready line in 10 s (" + (code >= 0 ? "it exited " + std::to_string(code) : std::string("killed")) + ")";
    return false;
  }
  // ai: the exit code, -1 where it was killed or none ran
  int end() {
    int code = -1;
#if defined(_WIN32)
    if (proc_) {
      TerminateProcess(proc_, 0);
      WaitForSingleObject(proc_, 3000);
      DWORD c = 0;
      if (GetExitCodeProcess(proc_, &c)) code = int(c);
      CloseHandle(proc_);
      proc_ = nullptr;
    }
    if (out_) { CloseHandle(out_); out_ = nullptr; }
#else
    if (pid_ > 0) {
      kill(pid_, SIGTERM);
      int st = 0;
      bool gone = false;
      for (int i = 0; i < 300 && !gone; i++) {
        const pid_t w = waitpid(pid_, &st, WNOHANG);
        gone = w == pid_;
        if (!gone) sleepMs(10);
      }
      if (!gone) { kill(pid_, SIGKILL); waitpid(pid_, &st, 0); }
      else if (WIFEXITED(st)) code = WEXITSTATUS(st);
      pid_ = -1;
    }
    if (out_ >= 0) { close(out_); out_ = -1; }
#endif
    return code;
  }
  std::string id;   // ai: the device id its ready line names

 private:
#if defined(_WIN32)
  HANDLE proc_ = nullptr, out_ = nullptr;
#else
  pid_t pid_ = -1;
  int out_ = -1;
#endif
};

uint32_t tri(uint32_t k) { return k < 256 ? k : 511 - k; }
uint64_t le(const uint8_t* p, int n) {
  uint64_t v = 0;
  for (int i = n - 1; i >= 0; i--) v = v << 8 | p[i];
  return v;
}

// ai: the gradient's luma at (x, y) for the phase (the fake's formula)
inline uint32_t grad(uint32_t x, uint32_t y, uint32_t phase) { return tri((x + y + phase) & 511); }

// ai: pixels of a frame off the fake's gradient at `phase` (rows 1 on; Bayer: its 2 x 2 mean with the edges clamped,
// ai: luma.h's), the first bad one said
size_t offGradient(const Frame& f, uint32_t phase, bool bayer, std::string* first) {
  size_t bad = 0;
  const uint32_t w = f.width, h = f.height;
  for (uint32_t y = 1; y < h; y++) {
    const uint8_t* r = f.luma + size_t(y) * f.stride;
    const uint32_t y1 = std::min(y + 1, h - 1);
    for (uint32_t x = 0; x < w; x++) {
      uint32_t want = grad(x, y, phase);
      if (bayer) {
        const uint32_t x1 = std::min(x + 1, w - 1);
        want = (grad(x, y, phase) + grad(x1, y, phase) + grad(x, y1, phase) + grad(x1, y1, phase) + 2) >> 2;
      }
      if (r[x] != want) {
        if (!bad && first) *first = fmt("(%u, %u) %u where %u", x, y, unsigned(r[x]), want);
        bad++;
      }
    }
  }
  return bad;
}

// ai: a Bayer frame's gradient phase: the one whose 2 x 2 mean row 1 holds, -1 none
int bayerPhase(const Frame& f) {
  if (f.height < 3) return -1;
  const uint8_t* r = f.luma + f.stride;
  for (uint32_t p = 0; p < 512; p++) {
    bool all = true;
    for (uint32_t x = 0; x + 1 < f.width && all; x++)
      all = r[x] == ((grad(x, 1, p) + grad(x + 1, 1, p) + grad(x, 2, p) + grad(x + 1, 2, p) + 2) >> 2);
    if (all) return int(p);
  }
  return -1;
}

// ai: one frame as the sink saw it
struct Seen {
  int64_t ns = 0, t = 0;   // ai: the frame's time; its stamp's (0 none)
  uint64_t seq = 0, n = 0;
  uint32_t w = 0, h = 0, expo = 0, gain = 0;
  int64_t bad = -1;        // ai: pixels off the gradient, -1 not looked at
  std::string first;
};

struct Rec {
  std::mutex mu;
  std::vector<Seen> seen;
  bool bayer = false;
  uint64_t count = 0;   // ai: the capture thread's
  // ai: the sink: the stamp read, every fourth frame's luma held to the gradient
  Camera::Sink sink() {
    return [this](const Frame& f) {
      Seen s;
      s.ns = f.ns;
      s.seq = f.seq;
      s.w = f.width;
      s.h = f.height;
      const bool look = count++ % 4 == 0;
      if (!bayer && f.width >= 28 && std::memcmp(f.luma, "LZGV", 4) == 0) {
        s.t = int64_t(le(f.luma + 4, 8));
        s.expo = uint32_t(le(f.luma + 12, 4));
        s.gain = uint32_t(le(f.luma + 16, 4));
        s.n = le(f.luma + 20, 8);
        if (look) s.bad = int64_t(offGradient(f, uint32_t((s.t / 1'000'000) % 512), false, &s.first));
      } else if (bayer && look) {
        const int p = bayerPhase(f);
        if (p < 0) { s.bad = int64_t(f.width) * f.height; s.first = "no phase of the gradient fits row 1"; }
        else s.bad = int64_t(offGradient(f, uint32_t(p), true, &s.first));
      }
      std::lock_guard<std::mutex> l(mu);
      seen.push_back(std::move(s));
    };
  }
  std::vector<Seen> take() {
    std::lock_guard<std::mutex> l(mu);
    return seen;
  }
};

int modeOf(const std::vector<Mode>& ms, const std::string& format, double fps) {
  for (size_t i = 0; i < ms.size(); i++)
    if (ms[i].format == format && std::fabs(ms[i].fps - fps) < 0.01) return int(i);
  return -1;
}

double controlOf(Camera& c, const char* id) {
  for (const auto& k : c.controls())
    if (k.id == id) return k.value;
  return NAN;
}

// ai: frames of mode for `secs` into rec; "" or why it did not start
std::string stream(Camera& c, const Mode& m, double secs, Rec& rec, Stats& st) {
  std::string why;
  rec.bayer = m.pix == Pix::Bayer8;
  if (!c.start(m, rec.sink(), [](const std::string&) {}, why)) return why.empty() ? "the start failed" : why;
  sleepMs(int(secs * 1000));
  st = c.stats();
  c.stop();
  return "";
}

// ai: --- the cases ---

void listCase(Report& r, const Fake& fake) {
  const std::string want = "aravis:" + fake.id;
  std::vector<std::string> notes;
  double firstMs = 0, worstLater = 0;
  int calls = 0;
  bool found = false;
  const auto t0 = std::chrono::steady_clock::now();
  std::string detail;
  while (!found && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(6)) {
    const auto a = std::chrono::steady_clock::now();
    const auto l = aravisList(&notes);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    if (calls++ == 0) firstMs = ms;
    else worstLater = std::max(worstLater, ms);
    for (const auto& i : l)
      if (i.id == want) { found = true; detail = i.name + ", " + i.detail + ", backend " + i.backend; }
    if (!found) sleepMs(200);
  }
  const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // ai: a list after it: at once
  const auto a = std::chrono::steady_clock::now();
  aravisList(&notes);
  worstLater = std::max(worstLater, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
  calls++;
  const std::string stamp = aravisStamp();
  const bool stamped = stamp.find(want + ";") != std::string::npos;
  std::string n;
  for (const auto& x : notes) n += (n.empty() ? "" : "; ") + x;
  r.say(found && worstLater < 100 && stamped, "list",
        found ? fmt("%s listed (%s) %.1f s after the first list, which took %.0f ms; %d lists, the slowest after the first "
                    "%.1f ms; the stamp %s it%s%s",
                    want.c_str(), detail.c_str(), took, firstMs, calls, worstLater, stamped ? "holds" : "does NOT hold",
                    n.empty() ? "" : "; notes: ", n.c_str())
              : fmt("%s not listed in 6 s (%d lists)%s%s", want.c_str(), calls, n.empty() ? "" : "; notes: ", n.c_str()));
}

std::unique_ptr<Camera> openCase(Report& r, const Fake& fake) {
  std::string why;
  const std::string id = "aravis:" + fake.id;
  auto c = cam::open(id, why);
  if (!c) { r.fail("open", id + ": " + why); return nullptr; }
  const auto ms = c->modes();
  std::vector<std::string> formats;
  for (const auto& m : ms)
    if (std::find(formats.begin(), formats.end(), m.format) == formats.end()) formats.push_back(m.format);
  std::vector<std::string> bad;
  if (c->info().id != id || c->info().backend != "aravis") bad.push_back("its info " + c->info().id + " (" + c->info().backend + ")");
  if (ms.empty()) bad.push_back("no modes");
  else {
    if (ms[0].format != "Mono8" || ms[0].pix != Pix::Grey8) bad.push_back("the first mode " + ms[0].format);
    if (formats.size() != 7) bad.push_back(fmt("%zu formats, not the fake's 7", formats.size()));
    for (const auto& f : formats)
      for (double fps : {1000.0, 120.0, 60.0, 30.0})
        if (modeOf(ms, f, fps) < 0) bad.push_back(fmt("no %s at %.0f", f.c_str(), fps));
    for (const auto& m : ms)
      if (m.width != 1920 || m.height != 1080 || m.fpsMin != m.fps) { bad.push_back(fmt("a mode %ux%u %s at %.2f (from %.2f)", m.width, m.height, m.format.c_str(), m.fps, m.fpsMin)); break; }
    const int p = pick(ms, 1920, 1080, 60);
    if (p != modeOf(ms, "Mono8", 60)) bad.push_back(fmt("pick's mode %d", p));
  }
  std::string fl;
  for (const auto& f : formats) fl += (fl.empty() ? "" : ", ") + f;
  std::string all;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  r.say(bad.empty(), "open",
        bad.empty() ? fmt("%s opened through the registry, %s; %zu modes: %s at 1920 x 1080, each at 1000, 120, 60 and 30, "
                          "pick's Mono8 at 60", id.c_str(), c->info().name.c_str(), ms.size(), fl.c_str())
                    : all);
  return c;
}

void controlsCase(Report& r, Camera& c) {
  std::vector<std::string> bad;
  const auto ks = c.controls();
  const Control* ex = nullptr;
  const Control* gn = nullptr;
  for (const auto& k : ks) {
    if (k.id == "exposure") ex = &k;
    if (k.id == "gain") gn = &k;
  }
  if (!ex || !gn) { r.fail("controls", fmt("%zu controls, exposure %s, gain %s", ks.size(), ex ? "listed" : "missing", gn ? "listed" : "missing")); return; }
  const double e0 = ex->value, g0 = gn->value;
  std::string why;
  // ai: a set on the stopped camera, put back at the next stop
  if (!c.set("exposure", 5000, false, why)) bad.push_back("a stopped set refused: " + why);
  const double eSet = controlOf(c, "exposure");
  c.stop();
  const double eBack = controlOf(c, "exposure");
  if (eSet != 5000) bad.push_back(fmt("exposure %.0f after a set of 5000", eSet));
  if (eBack != e0) bad.push_back(fmt("exposure %.0f after the stop, not its %.0f", eBack, e0));
  // ai: a start changes neither; sets while streaming in force, then put back
  const auto ms = c.modes();
  const int mi = modeOf(ms, "Mono8", 60);
  Rec rec;
  if (mi < 0 || !c.start(ms[size_t(mi)], rec.sink(), [](const std::string&) {}, why)) {
    r.fail("controls", "Mono8 at 60 did not start: " + why);
    return;
  }
  const double eStart = controlOf(c, "exposure"), gStart = controlOf(c, "gain");
  if (eStart != e0 || gStart != g0) bad.push_back(fmt("the start moved exposure %.0f to %.0f, gain %.0f to %.0f", e0, eStart, g0, gStart));
  sleepMs(300);
  if (!c.set("exposure", 4000, false, why)) bad.push_back("exposure 4000 refused while streaming: " + why);
  if (!c.set("gain", 3, false, why)) bad.push_back("gain 3 refused while streaming: " + why);
  sleepMs(500);
  c.stop();
  const double eEnd = controlOf(c, "exposure"), gEnd = controlOf(c, "gain");
  const auto seen = rec.take();
  size_t at4000 = 0, gained = 0, before = 0;
  for (const auto& s : seen) {
    if (s.expo == 4000 && s.gain == 3) gained++;
    if (s.expo == 4000) at4000++;
    if (s.expo == uint32_t(e0) && s.gain == uint32_t(g0)) before++;
  }
  if (!before) bad.push_back("no frame at the device's own exposure and gain before the sets");
  if (!gained) bad.push_back(fmt("no frame filled at exposure 4000 and gain 3 (%zu frames, %zu at 4000)", seen.size(), at4000));
  if (eEnd != e0 || gEnd != g0) bad.push_back(fmt("after the stop exposure %.0f, gain %.0f, not their %.0f and %.0f", eEnd, gEnd, e0, g0));
  std::string all;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  r.say(bad.empty(), "controls",
        bad.empty() ? fmt("exposure %.0f us (%.0f to %.0f), gain %.0f%s (%.0f to %.0f); a stopped set of 5000 put back at "
                          "the next stop; the start changed neither; set while streaming, %zu of %zu frames filled at "
                          "exposure 4000 and gain 3 (the fake's stamp), both back at the stop",
                          e0, ex->min, ex->max, g0, gn->unit.empty() ? " counts" : (" " + gn->unit).c_str(), gn->min, gn->max,
                          gained, seen.size())
                    : all);
}

void freeCase(Report& r, Camera& c, double fps) {
  const std::string name = fmt("free %.0f", fps);
  const auto ms = c.modes();
  const int mi = modeOf(ms, "Mono8", fps);
  if (mi < 0) { r.fail(name, "no Mono8 mode at " + fmt("%.0f", fps)); return; }
  Rec rec;
  Stats st;
  const std::string why = stream(c, ms[size_t(mi)], 3.0, rec, st);
  if (!why.empty()) { r.fail(name, why); return; }
  const auto s = rec.take();
  std::vector<std::string> bad;
  if (s.size() < 10) { r.fail(name, fmt("%zu frames in 3 s", s.size())); return; }
  // ai: the stamps: every fill handed over, the times one offset from the device's
  size_t unstamped = 0, gaps = 0, back = 0, looked = 0, wrong = 0;
  std::string firstWrong;
  int64_t offLo = INT64_MAX, offHi = INT64_MIN;
  std::vector<double> iv;
  for (size_t i = 0; i < s.size(); i++) {
    if (!s[i].t) { unstamped++; continue; }
    offLo = std::min(offLo, s[i].ns - s[i].t);
    offHi = std::max(offHi, s[i].ns - s[i].t);
    if (s[i].bad >= 0) {
      looked++;
      if (s[i].bad) { wrong++; if (firstWrong.empty()) firstWrong = s[i].first; }
    }
    if (i) {
      if (s[i].n != s[i - 1].n + 1) gaps++;
      if (s[i].ns <= s[i - 1].ns) back++;
      iv.push_back(double(s[i].ns - s[i - 1].ns) / 1e6);
    }
  }
  std::sort(iv.begin(), iv.end());
  const double med = iv.empty() ? 0 : iv[iv.size() / 2], period = 1000.0 / fps;
  const double span = double(s.back().t - s.front().t) / 1e9;
  const double made = span > 0 ? double(s.back().n - s.front().n) / span : 0;   // ai: the fake's own rate, by its count
  const double got = span > 0 ? double(s.size() - 1) / span : 0;
  if (unstamped) bad.push_back(fmt("%zu frames without the fake's stamp", unstamped));
  if (gaps || st.dropped || st.corrupt)
    bad.push_back(fmt("%zu gaps in the fake's count, %llu dropped, %llu corrupt", gaps, (unsigned long long)st.dropped, (unsigned long long)st.corrupt));
  if (back) bad.push_back(fmt("ns not rising %zu times", back));
  if (offHi != offLo) bad.push_back(fmt("the offset from the device's time moved %.3f ms", double(offHi - offLo) / 1e6));
  if (offLo < 0 || offLo > 100'000'000) bad.push_back(fmt("an offset of %.3f ms from the device's time", double(offLo) / 1e6));
  if (wrong || !looked) bad.push_back(fmt("%zu of %zu frames looked at off the gradient%s%s", wrong, looked, firstWrong.empty() ? "" : ", first ", firstWrong.c_str()));
  if (std::fabs(st.fps - fps) > fps * 0.001) bad.push_back(fmt("Stats::fps %.3f", st.fps));
  const bool paced = made >= fps * 0.97;
  if (paced && (std::fabs(med - period) > period * 0.02 || got < fps * 0.97)) bad.push_back(fmt("%.2f a second, the median interval %.3f ms", got, med));
  std::string all;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  const std::string said = fmt("%zu frames in %.2f s, %.2f a second (the fake made %.2f by its count, every one handed over), "
                               "Stats::fps %.3f, the median interval %.3f ms (%.3f a period), ns rising, %.3f ms after the device's "
                               "stamp each, %zu frames' luma the gradient exactly",
                               s.size(), span, got, made, st.fps, med, period, double(offLo) / 1e6, looked);
  if (!bad.empty()) r.fail(name, all + " (" + said + ")");
  else if (!paced) r.skip(name, "the fake made " + fmt("%.2f", made) + " a second, under the rate asked (its fill and send share one thread): " + said);
  else r.ok(name, said);
}

void formatsCase(Report& r, Camera& c) {
  const auto ms = c.modes();
  std::vector<std::string> bad, said;
  for (const char* f : {"Mono16", "BayerBG8", "BayerGB8", "BayerGR8", "BayerRG8", "RGB8"}) {
    const int mi = modeOf(ms, f, 60);
    if (mi < 0) { bad.push_back(std::string("no ") + f + " at 60"); continue; }
    Rec rec;
    Stats st;
    const std::string why = stream(c, ms[size_t(mi)], 1.0, rec, st);
    if (!why.empty()) { bad.push_back(std::string(f) + ": " + why); continue; }
    const auto s = rec.take();
    size_t looked = 0, wrong = 0;
    std::string first;
    for (const auto& x : s)
      if (x.bad >= 0) {
        looked++;
        if (x.bad) { wrong++; if (first.empty()) first = x.first; }
      }
    if (s.size() < 10 || !looked || wrong || st.corrupt)
      bad.push_back(fmt("%s: %zu frames, %zu of %zu looked at off the gradient%s%s, %llu corrupt", f, s.size(), wrong, looked,
                        first.empty() ? "" : ", first ", first.c_str(), (unsigned long long)st.corrupt));
    said.push_back(fmt("%s (%s) %zu frames, %zu looked at", f, pixName(ms[size_t(mi)].pix), s.size(), looked));
  }
  std::string all, ok;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  for (const auto& b : said) ok += (ok.empty() ? "" : "; ") + b;
  r.say(bad.empty(), "formats", bad.empty() ? ok + ": each one's luma the gradient's exactly (Bayer8 its 2 x 2 mean, convertMs counted)" : all);
}

// ai: The lock's lever on the gradient's fake (camera.h delayBar, captures, delayOnce): the bar "", the schedule begun
// ai: on its own, each Capture's ns a frame's own, fresh first, the deadlines a period apart and one delay of 5 ms
// ai: asked at the 120th capture landing as one interval of the period and 5 ms, the trigger's answers (the fake's
// ai: stamps) moved with it, the schedule's figures sane; then the trigger put back (a free run after it at the rate)
void triggerCase(Report& r, Camera& c) {
  const char* name = "trigger";
  const std::string bar = c.delayBar();
  if (!bar.empty()) { r.fail(name, "delayBar: " + bar); return; }
  const auto ms = c.modes();
  const int mi = modeOf(ms, "Mono8", 60);
  if (mi < 0) { r.fail(name, "no Mono8 at 60"); return; }
  struct Cap { int64_t ns, frameNs; bool delayed, fresh; };
  std::mutex mu;
  std::vector<Cap> caps;
  std::atomic<int> fates{0};
  std::atomic<bool> asked{false};
  const int64_t D = 5'000'000;
  c.captures(
      [&](const Capture& k) {
        size_t n;
        {
          std::lock_guard<std::mutex> l(mu);
          caps.push_back({k.ns, k.frameNs, k.delayed, k.fresh});
          n = caps.size();
        }
        if (n == 120) asked = c.delayOnce(D / 1000);
      },
      [&](bool) { fates++; });
  Rec rec;
  std::string why;
  if (!c.start(ms[size_t(mi)], rec.sink(), [](const std::string&) {}, why)) { r.fail(name, "start: " + why); c.captures(nullptr, nullptr); return; }
  sleepMs(4000);
  const std::string rep = c.report();
  const Stats st = c.stats();
  c.stop();
  c.captures(nullptr, nullptr);
  const auto seen = rec.take();
  std::vector<Cap> cs;
  {
    std::lock_guard<std::mutex> l(mu);
    cs = caps;
  }
  std::vector<std::string> bad;
  const auto j = nlohmann::json::parse(rep, nullptr, false);
  const auto sch = j.is_object() ? j.value("schedule", nlohmann::json()) : nlohmann::json();
  if (!j.is_object() || j.value("trigger", "") != "triggered" || !sch.is_object()) bad.push_back("the report " + rep);
  const int64_t P = sch.is_object() ? sch.value("periodNs", int64_t(0)) : 0;
  if (P != int64_t(1e9 / 60 * (1 - 0.001))) bad.push_back(fmt("a period of %lld ns", (long long)P));
  if (cs.size() < 150) bad.push_back(fmt("%zu captures in 4 s", cs.size()));
  // ai: each capture a frame's own time; the stamps' answer against the deadline before and after the delay
  std::vector<int64_t> frameNs;
  for (const auto& x : seen) frameNs.push_back(x.ns);
  std::sort(frameNs.begin(), frameNs.end());
  size_t notFrame = 0, fresh = 0, delayed = 0, offGrid = 0;
  int64_t delayedGap = 0, delayedFrame = 0;
  size_t at = 0;
  for (size_t i = 0; i < cs.size(); i++) {
    if (!std::binary_search(frameNs.begin(), frameNs.end(), cs[i].ns)) notFrame++;
    if (cs[i].fresh) fresh += i == 0 ? 1 : 100;
    if (cs[i].delayed) { delayed++; at = i; delayedFrame = cs[i].frameNs; }
    if (i) {
      const int64_t g = cs[i].ns - cs[i - 1].ns;
      if (cs[i].delayed) delayedGap = g;
      else if (g != P) offGrid++;
    }
  }
  std::vector<double> before, after;   // ai: ms the fake answered after each deadline
  for (const auto& x : seen) {
    if (!x.t) continue;
    for (size_t i = 0; i < cs.size(); i++)
      if (cs[i].ns == x.ns) { (at && i < at ? before : after).push_back(double(x.t - x.ns) / 1e6); break; }
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2]; };
  const double mb = med(before), ma = med(after);
  if (notFrame) bad.push_back(fmt("%zu captures at no frame's time", notFrame));
  if (fresh != 1) bad.push_back("fresh not on the first capture alone");
  if (!asked) bad.push_back("delayOnce refused");
  if (delayed != 1 || delayedGap != P + D || delayedFrame != P + D)
    bad.push_back(fmt("%zu delayed captures, its interval %.3f ms and frame %.3f ms (%.3f wanted)", delayed, delayedGap / 1e6, delayedFrame / 1e6, (P + D) / 1e6));
  if (offGrid) bad.push_back(fmt("%zu intervals off the period", offGrid));
  if (!(std::fabs(ma - mb) < 0.3)) bad.push_back(fmt("the fake answered %.3f ms after its deadline before the delay, %.3f after", mb, ma));
  if (fates) bad.push_back(fmt("%d fates told", fates.load()));
  const double lost = sch.is_object() ? sch.value("lost", -1.0) : -1, refused = sch.is_object() ? sch.value("refused", -1.0) : -1;
  const double late = sch.is_object() ? sch.value("lateUs", -1.0) : -1;
  const double lat = sch.is_object() && sch["latencyUs"].is_number() ? sch["latencyUs"].get<double>() : -1;
  if (lost > 1 || refused != 0 || late < 0 || late > 2000 || lat <= 0 || lat > 20000) bad.push_back("the schedule's figures " + sch.dump());
  // ai: the trigger put back: the next stream free-runs at the camera's rate (no feed: no trigger)
  Rec after2;
  Stats st2;
  const std::string w2 = stream(c, ms[size_t(mi)], 1.0, after2, st2);
  const size_t freeN = after2.take().size();
  if (!w2.empty() || freeN < 50) bad.push_back(fmt("a free run after the stop: %zu frames in 1 s%s%s", freeN, w2.empty() ? "" : ", ", w2.c_str()));
  std::string all;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  r.say(bad.empty(), name,
        bad.empty() ? fmt("delayBar \"\"; triggered %lld ms after the start, %zu captures in 4 s, each at its frame's time, fresh on the "
                          "first; the deadlines %.3f ms apart but one: the 5 ms delay asked at the 120th, its interval %.3f ms, "
                          "marked delayed, the fake's answers %.3f ms after their deadlines before it and %.3f after; schedule %s; "
                          "Stats::fps %.3f; then a free run of %zu frames in 1 s (the trigger put back)",
                          (long long)j.value("beganMs", int64_t(0)), cs.size(), P / 1e6, delayedGap / 1e6, mb, ma, sch.dump().c_str(),
                          st.fps, freeN)
                    : all);
}

// ai: One run of the session on a fake filming the simulated screen at a bad start (--start-phase 0.1: a capture
// ai: beginning a tenth into the picture on screen, which at the 8 ms exposure and the scene's scan reads near
// ai: nothing), the screen's clock as fast as the schedule (ppm 1000: no slide, so a free schedule stays where it
// ai: starts), the lock as LIZ_PHASE says; the stats every 250 ms and the receiver's series gathered
struct LockRun {
  std::string why, report;
  struct Poll { double t, blocks; bool stood; int delays; double landMs, pace; std::string state; };
  std::vector<Poll> polls;
  std::vector<std::array<double, 8>> frames;   // ai: the series' frames, each once
  double secs = 0;
  // ai: over the last `w` s: blocks a second, and the share of frames whose pilots read both near a whole picture
  double rate(double w) const {
    const Poll* a = nullptr;
    for (const auto& p : polls) if (p.t >= secs - w && !a) a = &p;
    return a && polls.back().t > a->t ? (polls.back().blocks - a->blocks) / (polls.back().t - a->t) : 0;
  }
  double clean(double w, size_t* n = nullptr) const {
    if (frames.empty()) return 0;
    const double end = frames.back()[0];
    size_t all = 0, good = 0;
    for (const auto& f : frames) {
      if (f[0] < end - w * 1000) continue;
      all++;
      if (f[3] == 1.0 && std::fabs(f[4]) >= 0.8 && std::fabs(f[6]) >= 0.8) good++;
    }
    if (n) *n = all;
    return all ? double(good) / double(all) : 0;
  }
};

LockRun lockRun(const fs::path& exe, const char* phase, double fps, double secs) {
  LockRun run;
  run.secs = secs;
  std::random_device rd;
  Fake fake;
  std::string why;
  if (!fake.start(exe, {"--serial", fmt("LK%04x", unsigned(rd() & 0xffff)), "--scene", "blocks=8 fps=60 hz=60 ppm=1000 w=960 h=540",
                        "--start-phase", "0.1"}, why)) { run.why = why; return run; }
  const fs::path store = fs::temp_directory_path() / fmt("lizard-check-aravis-%08x", unsigned(rd()));
  std::error_code ec;
  fs::create_directories(store, ec);
  const char* had = std::getenv("LIZ_PHASE");
  const std::string hadPhase = had ? had : "";
  if (phase) setEnv("LIZ_PHASE", phase);
  else unsetEnv("LIZ_PHASE");
  {
    Session::Config cfg;
    cfg.assets = LIZ_ASSETS;
    cfg.dataDir = store.u8string();
    cfg.decoder = "cpu";
    cfg.layout = "1:1";
    cfg.forgetSettings = true;
    auto s = Session::create(cfg, why);
    const std::string id = "aravis:" + fake.id;
    int mi = -1;
    if (s && done(s->open(id), why)) {
      const auto cj = nlohmann::json::parse(s->camera(), nullptr, false);
      if (cj.is_object() && cj.contains("modes"))
        for (size_t i = 0; i < cj["modes"].size(); i++)
          if (cj["modes"][i].value("format", "") == "Mono8" && std::fabs(cj["modes"][i].value("fps", 0.0) - fps) < 0.01) { mi = int(i); break; }
      if (mi < 0) why = "no Mono8 mode at " + fmt("%.0f", fps);
    }
    if (!s) run.why = "the session: " + why;
    else if (mi < 0) run.why = why;
    else if (!done(s->start(std::to_string(mi)), why)) run.why = "start: " + why;
    else {
      const auto t0 = std::chrono::steady_clock::now();
      double since = -1;
      for (;;) {
        sleepMs(250);
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto j = nlohmann::json::parse(s->stats(), nullptr, false);
        if (j.is_object()) {
          const auto k = j.value("lock", nlohmann::json::object());
          LockRun::Poll p{t, j.contains("totals") && j["totals"].is_object() ? j["totals"].value("blocks", 0.0) : 0.0,
                          k.is_object() && k.value("stood", false), k.is_object() ? k.value("delays", 0) : 0,
                          k.is_object() && k.contains("landMs") && k["landMs"].is_number() ? k["landMs"].get<double>() : NAN,
                          k.is_object() && k.contains("pace") && k["pace"].is_number() ? k["pace"].get<double>() : NAN,
                          k.is_object() ? k.value("state", std::string()) : std::string()};
          run.polls.push_back(p);
          if (j.contains("camera") && j["camera"].is_object() && j["camera"].contains("report")) run.report = j["camera"]["report"].dump();
        }
        const auto a = s->series(since);
        for (size_t o = 1; o + 8 <= a.size(); o += 8)
          if (a[o] > since) {
            std::array<double, 8> f;
            for (int q = 0; q < 8; q++) f[size_t(q)] = a[o + size_t(q)];
            run.frames.push_back(f);
          }
        for (const auto& f : run.frames) since = std::max(since, f[0]);
        if (t >= secs) break;
      }
      done(s->stop(), why);
    }
  }
  if (had) setEnv("LIZ_PHASE", hadPhase);
  else unsetEnv("LIZ_PHASE");
  fs::remove_all(store, ec);
  return run;
}

// ai: The lock on the bad start against the same run with it off (LIZ_PHASE=off), 14 s each, judged over the last
// ai: 8 s: the lock's hold standing, delays asked and landing, the captures clean (both pilots' readings at least
// ai: 0.8 of a whole picture's) in 30 points more of the frames than off, blocks a second twice off's and 400
void lockCase(Report& r, const fs::path& exe) {
  const char* name = "lock";
  const LockRun on = lockRun(exe, nullptr, 60, 14), off = lockRun(exe, "off", 60, 14);
  if (!on.why.empty() || !off.why.empty()) { r.fail(name, on.why.empty() ? "off: " + off.why : on.why); return; }
  double stoodAt = -1;
  for (const auto& p : on.polls) if (p.stood) { stoodAt = p.t; break; }
  const auto& last = on.polls.back();
  size_t nOn = 0, nOff = 0;
  const double cOn = on.clean(8, &nOn), cOff = off.clean(8, &nOff), bOn = on.rate(8), bOff = off.rate(8);
  // ai: the first second the lock's run read at nine tenths of its last 8 s
  double goodAt = -1;
  for (size_t i = 4; i < on.polls.size() && goodAt < 0; i++) {
    const auto& p = on.polls[i];
    const auto& q = on.polls[i - 4];
    if ((p.blocks - q.blocks) / (p.t - q.t) >= 0.9 * bOn) goodAt = p.t;
  }
  std::vector<std::string> bad;
  if (stoodAt < 0) bad.push_back("the hold never stood");
  if (last.delays <= 0 || !(last.landMs > 0)) bad.push_back(fmt("%d delays, landing in %.1f ms", last.delays, last.landMs));
  if (cOn < cOff + 0.3) bad.push_back(fmt("%.0f%% of captures clean with the lock, %.0f%% without", 100 * cOn, 100 * cOff));
  if (!(bOn >= 2 * bOff) || bOn < 400) bad.push_back(fmt("%.0f blocks a second with the lock, %.0f without", bOn, bOff));
  const std::string said = fmt("from a bad start (a capture a tenth into the picture on screen, the screen's clock the schedule's): with the "
                               "lock the hold stood %.1f s after the start, %d delays asked, landing in %.1f ms, the pace it holds "
                               "%.0f us a second (the schedule's against the screen's: about 0), the reading at 0.9 of its last by "
                               "%.1f s; over the last 8 s %.0f blocks a second, %.0f%% of %zu captures clean (both "
                               "pilots >= 0.8); with LIZ_PHASE=off %.0f blocks a second, %.0f%% of %zu clean; the camera's report "
                               "with the lock %s",
                               stoodAt, last.delays, last.landMs, last.pace, goodAt, bOn, 100 * cOn, nOn, bOff, 100 * cOff, nOff, on.report.c_str());
  std::string all;
  for (const auto& b : bad) all += (all.empty() ? "" : "; ") + b;
  r.say(bad.empty(), name, bad.empty() ? said : all + " (" + said + ")");
}

// ai: The camera at 120 against the 60 Hz screen, lock on (what the lock makes of a camera twice the display's rate:
// ai: said, not judged), held to the schedule running at 8.325 ms with no trigger lost and the stream reading
void lock120Case(Report& r, const fs::path& exe) {
  const char* name = "lock 120";
  const LockRun on = lockRun(exe, nullptr, 120, 10);
  if (!on.why.empty()) { r.fail(name, on.why); return; }
  const auto rep = nlohmann::json::parse(on.report, nullptr, false);
  const auto sch = rep.is_object() ? rep.value("schedule", nlohmann::json()) : nlohmann::json();
  const bool ran = sch.is_object() && sch.value("periodNs", int64_t(0)) == int64_t(1e9 / 120 * (1 - 0.001)) &&
                   sch.value("lost", 1e9) <= 0.01 * sch.value("fired", 0.0);
  size_t n = 0;
  const double b = on.rate(6), cl = on.clean(6, &n);
  const auto& last = on.polls.back();
  r.say(ran && b > 0, name,
        fmt("the schedule %s; %.0f blocks a second over the last 6 s, %.0f%% of %zu captures clean; the lock %s, %d delays "
            "asked, %s (it takes the camera's 8.325 ms frame for the display's refresh; each picture captured twice)",
            sch.dump().c_str(), b, 100 * cl, n, last.state.c_str(), last.delays, last.stood ? "its hold standing" : "its hold not yet standing"));
}

void sessionCase(Report& r, const fs::path& exe) {
  const char* c = "session";
  std::random_device rd;
  Fake fake;
  std::string why;
  const std::string serial = fmt("LS%04x", unsigned(rd() & 0xffff));
  if (!fake.start(exe, {"--serial", serial, "--scene", "blocks=8 fps=30 hz=60 w=960 h=540"}, why)) { r.fail(c, why); return; }
  const fs::path store = fs::temp_directory_path() / fmt("lizard-check-aravis-%08x", unsigned(rd()));
  std::error_code ec;
  fs::create_directories(store, ec);
  const char* vk = std::getenv("LIZ_VK_DEVICE");
  const std::string userVk = vk ? vk : "";
  setEnv("LIZ_VK_DEVICE", "llvmpipe");
  {
    Session::Config cfg;
    cfg.assets = LIZ_ASSETS;
    cfg.dataDir = store.u8string();
    cfg.decoder = "cpu";
    cfg.layout = "1:1";
    cfg.forgetSettings = true;
    auto s = Session::create(cfg, why);
    const std::string id = "aravis:" + fake.id;
    if (!s) r.fail(c, "the session: " + why);
    else if (!done(s->open(id), why)) r.fail(c, "open " + id + ": " + why);
    else {
      const auto cj = nlohmann::json::parse(s->camera(), nullptr, false);
      const int pick = cj.is_object() ? cj.value("pick", -1) : -1;
      std::string mode;
      if (pick >= 0 && cj.contains("modes") && size_t(pick) < cj["modes"].size()) {
        const auto& m = cj["modes"][size_t(pick)];
        mode = fmt("%s %d x %d at %.0f", m.value("format", "").c_str(), m.value("w", 0), m.value("h", 0), m.value("fps", 0.0));
      }
      if (pick < 0) r.fail(c, "no pick in " + s->camera().substr(0, 300));
      else if (!done(s->start(std::to_string(pick)), why)) r.fail(c, "start: " + why);
      else {
        double blocks = 0, firstRead = -1, camFps = 0, frames = 0, dropped = 0, perSec = 0;
        std::string state, decoder;
        const auto t0 = std::chrono::steady_clock::now();
        auto since = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
        while (since() < 20 && (firstRead < 0 || since() - firstRead < 2.5)) {
          sleepMs(250);
          const auto j = nlohmann::json::parse(s->stats(), nullptr, false);
          if (!j.is_object()) continue;
          if (j.contains("totals") && j["totals"].is_object()) blocks = j["totals"].value("blocks", 0.0);
          state = j.value("state", "");
          decoder = j.value("decoder", "");
          perSec = j.value("blocks", 0.0) / std::max(0.001, j.value("windowSecs", 0.0));
          if (j.contains("camera") && j["camera"].is_object()) {
            camFps = j["camera"].value("fps", 0.0);
            frames = j["camera"].value("frames", 0.0);
            dropped = j["camera"].value("dropped", 0.0);
          }
          if (firstRead < 0 && state == "test" && blocks > 0) firstRead = since();
        }
        done(s->stop(), why);
        if (firstRead >= 0)
          r.ok(c, fmt("%s (a fake filming the simulated screen: blocks=8 fps=30 hz=60 w=960 h=540) opened by an id no "
                      "discovery had listed, started in pick's %s: the test stream read %.1f s after the start, %.0f "
                      "blocks verified, %.0f a second on the %s decoder; the camera %.1f fps, %.0f frames, %.0f dropped",
                      id.c_str(), mode.c_str(), firstRead, blocks, perSec, decoder.c_str(), camFps, frames, dropped));
        else
          r.fail(c, fmt("state %s after 20 s, %.0f blocks verified, the camera %.1f fps, %.0f frames; %s", state.c_str(), blocks,
                        camFps, frames, s->stats().substr(0, 400).c_str()));
      }
    }
  }
  if (userVk.empty()) unsetEnv("LIZ_VK_DEVICE");
  else setEnv("LIZ_VK_DEVICE", userVk);
  fs::remove_all(store, ec);
}

}  // namespace

int aravis(const Args& a) {
  if (!a.empty()) {
    std::fprintf(stderr, "cam_check aravis takes no arguments\n");
    return kUsage;
  }
  Report r;
  const char* named = std::getenv("LIZ_ARAVIS");
  std::string why;
  const arv::Api* api = aravisApi(&why);
  if (!api) {
    if (named && *named) {
      r.fail("load", std::string("LIZ_ARAVIS names ") + named + ", which does not load: " + why);
      return r.code();
    }
    r.skip("load", named ? std::string("Aravis is off (LIZ_ARAVIS is set and empty)")
                         : why + " (the receiver does without it: no GenICam camera listed)");
    for (size_t i = 1; i < sizeof kCases / sizeof kCases[0]; i++) r.skip(kCases[i], "no Aravis (load's line)");
    return kHeld;
  }
  r.ok("load", fmt("Aravis %u.%u.%u from %s", api->vmajor, api->vminor, api->vmicro, api->from.c_str()));
  // ai: LIZ_CHECK_TIMING=0: the cases held to real time (a trigger's interval, the lock's schedule) skipped, said so,
  // ai: on a machine that is not held to it (a shared CI runner: one interval off the period, 171 of 994 triggers
  // ai: missed at 120, 2026-11-02, where this PC keeps them within microseconds); every other case runs
  const char* tv = std::getenv("LIZ_CHECK_TIMING");
  const bool timing = !(tv && std::string(tv) == "0");
  const std::string noTiming = "LIZ_CHECK_TIMING=0: a machine not held to real time";
  // ai: no USB camera enumerated or opened: GigE alone, before anything lists
  api->arv_disable_interface("USB3Vision");
  const fs::path exe = fakePath();
  std::error_code ec;
  if (exe.empty() || !fs::is_regular_file(exe, ec) || fs::file_size(exe, ec) < 4096) {
    const std::string w = "no lizard_fakegv beside cam_check (" + exe.u8string() + "): the build had no Aravis headers (LIZ_ARAVIS_ROOT)";
    for (size_t i = 1; i < sizeof kCases / sizeof kCases[0]; i++) named ? r.fail(kCases[i], w) : r.skip(kCases[i], w);
    return r.code();
  }
  {
    std::random_device rd;
    Fake fake;
    if (!fake.start(exe, {"--serial", fmt("LZ%04x", unsigned(rd() & 0xffff))}, why)) {
      for (size_t i = 1; i < 8; i++) r.fail(kCases[i], why);
    } else {
      listCase(r, fake);
      if (auto c = openCase(r, fake)) {
        controlsCase(r, *c);
        freeCase(r, *c, 60);
        freeCase(r, *c, 120);
        formatsCase(r, *c);
        if (timing) triggerCase(r, *c); else r.skip("trigger", noTiming);
      } else {
        for (const char* k : {"controls", "free 60", "free 120", "formats", "trigger"}) r.fail(k, "the camera did not open");
      }
    }
  }
  sessionCase(r, exe);
  if (timing) { lockCase(r, exe); lock120Case(r, exe); }
  else { r.skip("lock", noTiming); r.skip("lock 120", noTiming); }
  return r.code();
}

}  // namespace lizard::check
