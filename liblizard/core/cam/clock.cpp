// ai: clock.h (2026-10-25). sleepUntil sleeps with each system's precise sleep, then spins: Linux's clock_nanosleep on
// ai: CLOCK_MONOTONIC's absolute time (libstdc++'s and libc++'s steady_clock there, so t is that clock's own), Windows'
// ai: high-resolution waitable timer (Windows 10 1803 on; before it Sleep under timeBeginPeriod(1), winmm looked up at
// ai: run time) and macOS's mach_wait_until, both on the time left converted to their units (their clocks need not be
// ai: steady_clock's).
#include "clock.h"

#include <algorithm>
#include <chrono>

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
#endif

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

}  // namespace lizard::cam
