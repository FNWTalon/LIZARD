// ai: The phase lock's host (2026-10-31): what a receive session does around the lock (phase_lock.h) for a camera that
// ai: can delay one capture, as the Android app's Engine.kt did around PhaseLock.kt (its capture callback's order, the
// ai: mode read every 30 captures, the feed, the log line every 300). The camera backend supplies the lever:
// ai:   delayOnce(us)  one capture made us microseconds longer, so every capture after it starts that much later: on
// ai:                  Camera2 one request with auto-exposure off, the newest undelayed result's exposure and sensitivity,
// ai:                  and SENSOR_FRAME_DURATION that result's frame plus us, the repeating request going on after it;
// ai:                  false where it could not be asked (no undelayed result yet, the request refused)
// ai: and reports, per capture result in order: its sensor timestamp (the clock the receiver's series keeps, since the
// ai: frame pushed carries the same timestamp) and whether it was that delayed capture; and the delayed capture's fate
// ai: where it is not a completed result (its request failed: taken where its image was captured all the same; its
// ai: sequence aborted: not taken). A camera without per-frame manual sensor control cannot delay a capture where it
// ai: was asked (opened's bar), and runs free with the lock Off.
// ai: Threads: opened, want, captured and fate on the camera's callback thread (one thread for a capture session's
// ai: callbacks), delayOnce never calling back into the host before it returns; state() from any thread.
#pragma once
#include "phase_lock.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lizard {

class PhaseLockHost {
 public:
  // ai: series: the receiver's frames after sinceMs (Receiver::series's vector; {0} with no receiver)
  using Series = std::function<std::vector<double>(double sinceMs)>;
  using DelayOnce = std::function<bool(int64_t us)>;

  PhaseLockHost(PhaseLock::Say say, Series series, DelayOnce delayOnce, bool debug = false)
      : say_(say), lock_(std::move(say), debug), series_(std::move(series)), delayOnce_(std::move(delayOnce)) {}

  // ai: A capture session begun (each open of the camera, each restart of its stream): the lock's camera and hold
  // ai: afresh (reopened), the feed from the series' start. bar: why this camera cannot delay a capture where it was
  // ai: asked ("" where it can), which holds the lock Off for the session.
  void opened(const std::string& bar) {
    lock_.reopened();
    fedMs_ = -1.0;
    n_ = 0;
    bar_ = bar;
    if (!bar_.empty()) {
      say_("phase: no lock on this lens: " + bar_);
      lock_.set(PhaseLock::Mode::Off);
    }
  }
  // ai: The mode a tool asks (Track unless told; Android's debug.lizard.phase), taken at the session's next 30th capture
  void want(PhaseLock::Mode m) { want_ = m; }
  // ai: A capture's result: the mode (every 30 captures), the receiver's new frames to the lock, then the capture, and
  // ai: the delay it asks to the camera (a refusal told to the lock at once); the lock's line every 300 captures
  void captured(int64_t tsNs, bool delayed) {
    if (n_ % 30 == 15) lock_.set(bar_.empty() ? want_ : PhaseLock::Mode::Off);
    feed();
    const int64_t us = lock_.onCapture(tsNs, delayed);
    if (us > 0 && !delayOnce_(us)) lock_.delayFate(false);
    if (n_ % 300 == 0)
      if (const auto l = lock_.line(tsNs)) say_(*l);
    n_++;
  }
  void fate(bool taken) { lock_.delayFate(taken); }
  // ai: for the stats and a replay's rows (PhaseLock::State), and the bar ("" none) where the lock is off for the camera
  std::optional<PhaseLock::State> state() const { return lock_.state(); }
  const std::string& bar() const { return bar_; }

 private:
  // ai: the frames after the newest fed, Engine.feed's way: none asked while the lock is Off
  void feed() {
    if (lock_.mode() != PhaseLock::Mode::Track) return;
    const std::vector<double> a = series_(fedMs_);
    int version = 0;
    const std::vector<PhaseLock::Frame> fs = PhaseLock::framesOf(a, version);
    if (fs.empty()) return;
    fedMs_ = a[1 + 8 * (fs.size() - 1)];
    lock_.onWindow(fs, version);
  }

  PhaseLock::Say say_;
  PhaseLock lock_;
  Series series_;
  DelayOnce delayOnce_;
  PhaseLock::Mode want_ = PhaseLock::Mode::Track;
  std::string bar_;
  double fedMs_ = -1.0;
  uint64_t n_ = 0;
};

}  // namespace lizard
