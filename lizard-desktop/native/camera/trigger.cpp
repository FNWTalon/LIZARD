// ai: trigger.h's clock (2026-10-25). sleepUntil sleeps with each system's precise sleep, then spins: Linux's
// ai: clock_nanosleep on CLOCK_MONOTONIC's absolute time (libstdc++'s and libc++'s steady_clock there, so t is that
// ai: clock's own), Windows' high-resolution waitable timer (Windows 10 1803 on; before it Sleep under
// ai: timeBeginPeriod(1), winmm looked up at run time) and macOS's mach_wait_until, both on the time left converted to
// ai: their units (their clocks need not be steady_clock's). The loop's wait to 2 ms before a deadline is a
// ai: condition variable's timed wait, but on Windows: there libstdc++ (MinGW-w64, no pthread_cond_clockwait) hands it
// ai: to winpthreads as a system-clock deadline, which waits whole milliseconds against GetSystemTimeAsFileTime and
// ai: retries until GetTickCount64 passes them, both moving in the system's 15.6 ms ticks, which no timeBeginPeriod
// ai: changes (a Windows 11 VM: such a wait 12.9 ms late at the median, so triggers went out up to 16.5 ms late and
// ai: deadlines were skipped). There the loop sets its own waitable timer (high resolution, else a plain one on the
// ai: 1 ms tick) and waits on it together with an event that begin, end, delay and the destructor set.
#include "trigger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#if defined(__linux__)
#include <cerrno>
#include <ctime>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <thread>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace lizard::cam {

namespace {

constexpr int64_t kWakeNs = 2'000'000;      // ai: the loop's wait (waitUntil) ends this far before a deadline
constexpr int64_t kKeepNs = 1'000'000'000;  // ai: a trigger unmatched this long is lost
constexpr size_t kLat = 64;                 // ai: the matches the latency's median is over

inline void relax() {
#if defined(_MSC_VER)
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield");
#endif
}

#if defined(_WIN32)
// ai: the 1 ms scheduler tick the Sleep fallback needs, asked once a process
void coarseTick() {
  static const bool once = [] {
    if (HMODULE m = LoadLibraryW(L"winmm.dll")) {
      using Fn = unsigned (WINAPI*)(unsigned);
      if (auto f = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(m, "timeBeginPeriod")))) f(1);
    }
    return true;
  }();
  (void)once;
}

// ai: the loop's timer: a high-resolution one (Windows 10 1803 on), else a plain one, whose waits end on the 1 ms
// ai: tick coarseTick asks for
HANDLE loopTimer() {
  if (HANDLE t = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) return t;
  coarseTick();
  return CreateWaitableTimerW(nullptr, FALSE, nullptr);
}
#endif

// ai: x to one decimal with a '.': whole tenths printed as integers, which no locale changes (%.1f follows
// ai: LC_NUMERIC, which the JVM sets to the user's: under a decimal-comma locale stats() read "lateUs":0,5, and the
// ai: session's parse of it gave the trigger's figures as null, 2026-10-25)
std::string tenths(double x) {
  const long long t = std::llround(x * 10);
  const unsigned long long a = t < 0 ? 0ULL - static_cast<unsigned long long>(t) : static_cast<unsigned long long>(t);
  char b[32];
  std::snprintf(b, sizeof b, "%s%llu.%llu", t < 0 ? "-" : "", a / 10, a % 10);
  return b;
}

}  // namespace

int64_t steadyNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sleepUntil(int64_t t, int64_t spinNs) {
  const int64_t wake = t - std::max<int64_t>(spinNs, 0);
#if defined(__linux__)
  if (steadyNs() < wake) {
    timespec ts{};
    ts.tv_sec = time_t(wake / 1'000'000'000);
    ts.tv_nsec = long(wake % 1'000'000'000);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
  }
#elif defined(_WIN32)
  if (steadyNs() < wake) {
    // ai: A high-resolution timer made for this sleep and closed after it (0.5 us the pair, median, in a Windows 11
    // ai: VM), its due time taken once it is made; none before Windows 10 1803. 2026-10-25: it was one timer a thread,
    // ai: a thread_local closed by its destructor, and under MinGW-w64 GCC's emulated TLS threads that end holding a
    // ai: thread_local with a destructor corrupt the heap.
    if (HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS)) {
      LARGE_INTEGER due;
      due.QuadPart = -((wake - steadyNs()) / 100);   // ai: relative, in 100 ns
      if (due.QuadPart < 0 && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
      CloseHandle(timer);
    } else {
      coarseTick();
      // ai: Sleep ends up to a tick late: the last millisecond and a half is spun
      const int64_t ms = (wake - steadyNs() - 1'500'000) / 1'000'000;
      if (ms > 0) Sleep(DWORD(ms));
    }
  }
#elif defined(__APPLE__)
  const int64_t left = wake - steadyNs();
  if (left > 0) {
    static const mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t i{}; mach_timebase_info(&i); return i; }();
    const uint64_t ticks = uint64_t(double(left) * tb.denom / tb.numer);
    mach_wait_until(mach_absolute_time() + ticks);
  }
#else
  if (steadyNs() < wake) std::this_thread::sleep_for(std::chrono::nanoseconds(wake - steadyNs()));
#endif
  while (steadyNs() < t) relax();
}

TriggerClock::TriggerClock(std::function<bool(std::string&)> fire, std::function<void(const std::string&)> log)
    : fire_(std::move(fire)), log_(std::move(log)) {
#if defined(_WIN32)
  woken_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  timer_ = loopTimer();
#endif
}

TriggerClock::~TriggerClock() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    quit_ = true;
    run_ = false;
  }
  wake();
  if (th_.joinable()) th_.join();
#if defined(_WIN32)
  if (timer_) CloseHandle(timer_);
  if (woken_) CloseHandle(woken_);
#endif
}

void TriggerClock::wake() {
  cv_.notify_all();
#if defined(_WIN32)
  if (woken_) SetEvent(woken_);
#endif
}

void TriggerClock::waitUntil(std::unique_lock<std::mutex>& lk, int64_t t) {
#if defined(_WIN32)
  // ai: the event is a latch: a wake() between the unlock and the wait is not lost, and one that came while the loop
  // ai: was not waiting here ends this wait at once, after which the loop looks again
  if (timer_ && woken_) {
    LARGE_INTEGER due;
    due.QuadPart = -std::max<int64_t>((t - steadyNs()) / 100, 1);   // ai: relative, in 100 ns
    if (SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE)) {
      HANDLE both[2] = {woken_, timer_};
      lk.unlock();
      const DWORD r = WaitForMultipleObjects(2, both, FALSE, INFINITE);
      lk.lock();
      if (r != WAIT_FAILED) return;
    }
  }
#endif
  cv_.wait_for(lk, std::chrono::nanoseconds(t - steadyNs()));
}

void TriggerClock::begin(int64_t periodNs) {
  if (periodNs <= 0) { end(); return; }
  {
    std::unique_lock<std::mutex> lk(mu_);
    // ai: a trigger being fired belongs to the old schedule: it finishes first (not from the fire callback itself)
    if (std::this_thread::get_id() != th_.get_id()) cv_.wait(lk, [&] { return !firing_; });
    period_ = periodNs;
    next_ = steadyNs() + periodNs;
    moved_ = false;
    fired_.clear();
    lat_.clear();
    nFired_ = nLost_ = nRefused_ = nMissed_ = 0;
    lateMax_ = 0;
    lateSum_ = lateSq_ = 0;
    run_ = true;
    if (!th_.joinable()) th_ = std::thread([this] { loop(); });
  }
  wake();
}

void TriggerClock::delay(int64_t ns) {
  if (ns <= 0) return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!run_) return;
    next_ += ns;
    moved_ = true;
  }
  wake();
}

void TriggerClock::end() {
  {
    std::unique_lock<std::mutex> lk(mu_);
    if (std::this_thread::get_id() != th_.get_id()) cv_.wait(lk, [&] { return !firing_; });
    run_ = false;
    moved_ = false;
    fired_.clear();
  }
  wake();
}

void TriggerClock::loseOldest() {
  const bool d = fired_.front().delayed;
  fired_.pop_front();
  nLost_++;
  if (d) {
    if (!fired_.empty()) fired_.front().delayed = true;
    else moved_ = true;
  }
}

void TriggerClock::loop() {
  std::unique_lock<std::mutex> lk(mu_);
  while (!quit_) {
    if (!run_) { cv_.wait(lk); continue; }
    const int64_t now = steadyNs();
    if (next_ - now > kWakeNs) {
      waitUntil(lk, next_ - kWakeNs);
      continue;   // ai: woken early or on time: begin, end and delay may have moved things, so look again
    }
    // ai: the deadline is taken: a delay from here on moves the one after it
    int64_t due = next_;
    const int64_t behind = now - due;
    int64_t skipped = 0;
    if (behind >= period_) {
      skipped = behind / period_;
      due += skipped * period_;
      nMissed_ += uint64_t(skipped);
    }
    next_ = due + period_;
    const bool moved = moved_;
    moved_ = false;
    firing_ = true;
    due_ = due;
    lk.unlock();
    sleepUntil(due);
    const int64_t at = steadyNs();
    // ai: kept before it goes out, so a frame answering it while the fire is still out (a command's own round trip,
    // ai: or this thread held off the CPU after it) finds it (2026-10-25: kept after the fire, with two busy loops on
    // ai: each CPU a simulated camera answering 3 ms after its trigger matched nothing, and the trigger went as lost)
    lk.lock();
    while (!fired_.empty() && fired_.front().at < at - kKeepNs) loseOldest();
    fired_.push_back({due, at, moved});
    lk.unlock();
    std::string why;
    const bool ok = fire_(why);
    lk.lock();
    firing_ = false;
    nFired_++;
    const double late = double(at - due);
    lateMax_ = std::max(lateMax_, at - due);
    lateSum_ += late;
    lateSq_ += late * late;
    if (!ok) {
      // ai: refused: answered by no frame, taken off the kept ones where it still is (a match meanwhile may have passed
      // ai: it over, counting it lost; a begin from the fire forgot it), its delayed mark passing to the next trigger
      nRefused_++;
      const auto it = std::find_if(fired_.begin(), fired_.end(), [&](const Fired& f) { return f.at == at; });
      if (it != fired_.end()) {
        if (it->delayed) moved_ = true;
        fired_.erase(it);
        nLost_++;
      }
    }
    if (log_ && (skipped > 0 || !ok) && at - saidNs_ >= 1'000'000'000) {
      saidNs_ = at;
      char line[256];
      if (!ok)
        std::snprintf(line, sizeof line, "trigger: refused (%s), %llu of %llu refused", why.c_str(),
                      static_cast<unsigned long long>(nRefused_), static_cast<unsigned long long>(nFired_));
      else
        std::snprintf(line, sizeof line, "trigger: %lld deadlines missed, the clock %s ms behind",
                      static_cast<long long>(skipped), tenths(double(behind) / 1e6).c_str());
      lk.unlock();   // ai: log_ is set once, at construction
      log_(line);
      lk.lock();
    }
    cv_.notify_all();   // ai: a begin or end may wait on this fire
  }
}

bool TriggerClock::match(int64_t arrivedNs, int64_t& ns, bool& delayed) {
  std::lock_guard<std::mutex> lk(mu_);
  while (!fired_.empty() && fired_.front().at < arrivedNs - kKeepNs) loseOldest();
  size_t n = 0;   // ai: fired_ is oldest first: the first n were fired at or before the arrival
  while (n < fired_.size() && fired_[n].at <= arrivedNs) n++;
  if (n == 0) return false;
  size_t best = 0;
  if (!lat_.empty()) {
    std::vector<int64_t> s(lat_);
    std::nth_element(s.begin(), s.begin() + (s.size() - 1) / 2, s.end());
    const int64_t l = s[(s.size() - 1) / 2];
    int64_t err = INT64_MAX;
    for (size_t i = 0; i < n; i++) {
      const int64_t d = arrivedNs - fired_[i].at - l, e = d < 0 ? -d : d;
      if (e < err) { err = e; best = i; }
    }
  }
  for (size_t i = 0; i < best; i++) loseOldest();
  ns = fired_.front().due;
  delayed = fired_.front().delayed;
  const int64_t out = fired_.front().at;
  fired_.pop_front();
  if (lat_.size() == kLat) lat_.erase(lat_.begin());
  lat_.push_back(arrivedNs - out);
  return true;
}

std::string TriggerClock::stats() {
  std::lock_guard<std::mutex> lk(mu_);
  const double n = double(nFired_);
  const double mean = n > 0 ? lateSum_ / n : 0;
  const double sd = n > 1 ? std::sqrt(std::max(0.0, lateSq_ / n - mean * mean)) : 0;
  std::string lat = "null";
  if (!lat_.empty()) {
    std::vector<int64_t> s(lat_);
    std::nth_element(s.begin(), s.begin() + (s.size() - 1) / 2, s.end());
    lat = tenths(double(s[(s.size() - 1) / 2]) / 1e3);
  }
  char out[320];
  std::snprintf(out, sizeof out,
                "{\"periodNs\":%lld,\"fired\":%llu,\"lost\":%llu,\"refused\":%llu,\"missed\":%llu,\"lateUs\":%s,"
                "\"jitterUs\":%s,\"latencyUs\":%s}",
                static_cast<long long>(period_), static_cast<unsigned long long>(nFired_),
                static_cast<unsigned long long>(nLost_), static_cast<unsigned long long>(nRefused_),
                static_cast<unsigned long long>(nMissed_), tenths(double(lateMax_) / 1e3).c_str(),
                tenths(sd / 1e3).c_str(), lat.c_str());
  return out;
}

}  // namespace lizard::cam
