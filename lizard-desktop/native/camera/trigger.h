// ai: A software trigger fired on a schedule the host keeps (2026-10-25): the phase lock's lever on a camera that
// ai: exposes one frame at each trigger (a GenICam camera's TriggerSoftware; the simulated camera's until a GenICam
// ai: backend exists). The schedule's deadlines are T0 + k P plus every delay asked, on the host's steady clock,
// ai: never re-based on what the frames say or on when a trigger
// ai: went out: the phase lock reads the frames and asks the delays (liblizard/kotlin/PhaseLock.kt, the desktop's
// ai: Lock.kt), as it asks Camera2 for one long frame on Android. A thread of its own waits to about 2 ms before each
// ai: deadline, woken sooner by begin, end and delay: a condition variable's timed wait on Linux and macOS; on Windows
// ai: a high-resolution waitable timer and an event those calls set, waited together (2026-10-25: MinGW-w64's posix
// ai: threads time a condition variable's wait in whole ticks of the system's 15.6 ms clock, which ended the wait
// ai: 12.5 to 15 ms late and sent triggers up to a period late). Then it sleeps to within a fraction of a millisecond
// ai: of the deadline with the system's precise sleep (clock_nanosleep on its absolute monotonic time, a
// ai: high-resolution waitable timer on Windows, mach_wait_until on macOS), spins the rest and fires. A loop more
// ai: than a period behind (its thread held off the CPU) skips the deadlines it missed, counted, and never fires a
// ai: burst to catch up.
// ai: Each frame that arrives is matched to the trigger that exposed it by its arrival time against each trigger's
// ai: time out plus the camera's latency, the median of the last 64 matches (a trigger the camera dropped leaves a gap,
// ai: not a shift of every frame after it); its time is that trigger's deadline, so the lock reads the schedule it
// ai: moves exactly: not the host's lateness in sending the trigger (the stats' lateUs and jitterUs), which the
// ai: exposure still carries as a jitter of its phase that only the pilots see, nor the transport's jitter in
// ai: bringing the frame back. Stamped with the time
// ai: the trigger went out until 2026-10-25, a trigger the host's timer sent late read as a capture that late, which
// ai: PhaseLock answers with a delay of a frame less that (a capture skipped), and the camera's frame it takes from one
// ai: interval carried two triggers' lateness: in a Windows VM whose timers run 1 to 3 ms late the camera ran 26.9 of 30
// ai: captures a second and the pace read its bound, 5,000 us a second, where the schedule's own arithmetic is 1,001.
// ai: With no match yet the oldest trigger fired before the arrival is taken: the
// ai: first frame after begin answers the first trigger. Arrival times alone cannot tell a latency L from L + P, so a
// ai: first trigger whose frame never comes (a camera that ignores triggers until its stream runs, or one that
// ai: answers several early triggers with one frame) matches every later frame to a trigger k periods before its
// ai: own, which no later frame corrects: begin the schedule with the camera already streaming.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lizard::cam {

// ai: The host's steady clock in ns (std::chrono::steady_clock: CLOCK_MONOTONIC on Linux, QPC on Windows, libc++'s
// ai: monotonic clock on macOS), the clock every trigger and arrival is on.
int64_t steadyNs();
// ai: Back at or just after t (steadyNs's clock), never before it: the system's precise sleep to within `spinNs` of
// ai: it, then a spin on steadyNs. A t already past returns at once.
void sleepUntil(int64_t t, int64_t spinNs = 300'000);

class TriggerClock {
 public:
  // ai: fire: the camera's trigger command, called on the clock's thread, true when the camera took it (false with
  // ai: why ends nothing: a refused trigger counts as lost, and a delayed mark it carried passes to the next). log:
  // ai: what the clock has to say (a refusal, deadlines missed), at most a line a second; may be null.
  explicit TriggerClock(std::function<bool(std::string& why)> fire, std::function<void(const std::string&)> log = nullptr);
  ~TriggerClock();
  TriggerClock(const TriggerClock&) = delete;
  TriggerClock& operator=(const TriggerClock&) = delete;
  // ai: Fires every periodNs from a period from now: the first deadline now + periodNs, the thread started at the
  // ai: first begin and kept. A running schedule restarts at the new period, its triggers, matches and counts
  // ai: forgotten. Waits for a trigger being fired to finish (at most about 2 ms and the fire's own time). A period
  // ai: of 0 or less is end().
  void begin(int64_t periodNs);
  // ai: The next deadline not yet being fired and every one after it ns later (camera.h Trigger::delay), the trigger
  // ai: at that deadline the one reported delayed. Any thread, the fire callback's included; ns of 0 or less, or no
  // ai: schedule running, does nothing.
  void delay(int64_t ns);
  // ai: Stops the schedule, waiting for a trigger being fired to finish; triggers not yet matched are forgotten, so a
  // ai: frame arriving after end() matches nothing.
  void end();
  bool running() const { return run_.load(); }
  // ai: The deadline of the trigger being fired, for the fire callback to read (0 before the first): the time its
  // ai: frame will carry (match), where the moment the callback runs carries the trigger's lateness.
  int64_t firing() const { return due_.load(); }
  // ai: A frame that arrived at arrivedNs (steadyNs's clock, as the backend stamps a frame it is handed), matched to
  // ai: the trigger that exposed it: of the triggers fired at or before arrivedNs, the one whose time out plus the
  // ai: latency is nearest the arrival. Its deadline into ns, and delayed for the first trigger a delay moved; the triggers fired
  // ai: before it go as lost (counted), a delayed mark among them passing to it, so the lock sees one delayed frame
  // ai: for each delay even where the moved trigger's own frame was lost. Triggers unmatched for a second go as lost
  // ai: too. False where no trigger fits (none kept fired before the arrival: a free-running frame, or one after
  // ai: end()), nothing consumed.
  bool match(int64_t arrivedNs, int64_t& ns, bool& delayed);
  // ai: Trigger::stats's JSON, since begin: periodNs; fired (triggers sent, refused ones included), lost (fired and
  // ai: answered by no frame: refused, or passed over by a match, or a second old), refused, missed (deadlines skipped
  // ai: by a loop more than a period behind); lateUs (the worst a trigger went out after its deadline) and jitterUs
  // ai: (the standard deviation of that lateness); latencyUs (the median of the matches' arrival less the trigger's
  // ai: time out, null before the first). Its numbers carry a '.' whatever the process's locale.
  std::string stats();

 private:
  struct Fired { int64_t due, at; bool delayed; };   // ai: its deadline (its frame's time); when it went out
  void loop();
  void wake();                                  // ai: the loop woken (after a change made under mu_)
  void waitUntil(std::unique_lock<std::mutex>& lk, int64_t t);   // ai: mu_ held; back at t or when woken
  void loseOldest();   // ai: mu_: the oldest unmatched trigger counted lost, a delayed mark passing on
  std::function<bool(std::string&)> fire_;
  std::function<void(const std::string&)> log_;
  std::atomic<bool> run_{false};
  std::atomic<int64_t> due_{0};                 // ai: the deadline of the trigger being fired (firing())
  std::thread th_;
  mutable std::mutex mu_;
  std::condition_variable cv_;                  // ai: wakes the loop (begin, end, delay, the destructor) and a begin
                                                // ai: or end waiting on a fire
#if defined(_WIN32)
  void* timer_ = nullptr;                       // ai: the loop's waitable timer (high resolution where the system has one)
  void* woken_ = nullptr;                       // ai: an auto-reset event: wake()'s, waited with the timer
#endif
  int64_t period_ = 0, next_ = 0;               // ai: mu_: the period; the next deadline not yet taken by the loop
  bool moved_ = false;                          // ai: mu_: the next trigger fired is reported delayed
  bool firing_ = false;                         // ai: mu_: the loop has taken a deadline and its fire is not yet done
  bool quit_ = false;                           // ai: mu_: the destructor's
  std::deque<Fired> fired_;                     // ai: mu_: fired and not yet matched, oldest first
  std::vector<int64_t> lat_;                    // ai: mu_: the latest matches' latencies, for their median
  uint64_t nFired_ = 0, nLost_ = 0, nRefused_ = 0, nMissed_ = 0;
  int64_t lateMax_ = 0;
  double lateSum_ = 0, lateSq_ = 0;
  int64_t saidNs_ = 0;                          // ai: when log_ was last called (a line a second at most)
};

}  // namespace lizard::cam
