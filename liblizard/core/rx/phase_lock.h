// ai: The camera's phase lock (2026-10-31): liblizard/kotlin/PhaseLock.kt (D6 of the pilots-only, self-healing
// ai: rework, the Android app's lock since 2026-10-28) in C++, decision for decision, so a receive session that owns a
// ai: camera able to delay one capture (a capture made longer, which moves every capture after it) steers it here.
// ai: Camera-agnostic: every time is on the camera's own clock (Camera2's SENSOR_TIMESTAMP on Android), and the lock
// ai: neither opens nor drives a camera: its host feeds it the receiver's series (Receiver::series), each capture's
// ai: timestamp, the fate of each delay it asked and a reopen, and applies the delay it returns to the next capture.
// ai: phase_lock.cpp's header holds the model; tools/phase_lock_check.cpp replays a trace of those calls.
// ai: Threads: every call on one thread (the host's capture thread), state() from any.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace lizard {

class PhaseLock {
 public:
  enum class Mode { Off, Track };
  // ai: A frame of the receiver's series: its capture's timestamp (ns), verified blocks, whether a symbol was
  // ai: registered; r, rSd: the pilots' reading over the even blocks and its standard error; r2, r2Sd: over the odd;
  // ai: row: the code's mean row as a share of the sensor's rows, in the order the camera reads them (NaN: not known)
  struct Frame {
    int64_t ts = 0;
    int blocks = 0;
    bool found = false;
    double r = NAN, rSd = NAN, r2 = NAN, r2Sd = NAN, row = NAN;
  };
  // ai: The lock's state for the stats row and a replay's meta: a snapshot made at each decision, read from any thread.
  // ai: `arm`: pilots; `what`: the word line() uses (holds, no frames yet); the newest estimate of the leaks (a and b
  // ai: the shares of the pictures before and after in a capture, k what a capture of one picture reads, se the error
  // ai: of a - b, n its frames; NaN with none); the gain (ms a unit of a - b) and the pace (us a second); whether the
  // ai: hold stands; the edge of a flat top it stepped in from (1 early, -1 late, 0 none) and the top's width (ms); the
  // ai: delays asked since the mode was set; the display's refresh on the camera's clock (ms, NaN before a hold), the
  // ai: refreshes a capture spans, the camera's frame (ms); the longest landing of a delay in the newest LONG_NS (ms
  // ai: from its ask, NaN none); the code's row.
  struct State {
    std::string arm, what;
    double a = NAN, b = NAN, k = NAN, se = NAN;
    int n = 0;
    double gain = 0, pace = 0;
    bool stood = false;
    int edge = 0;
    double flatMs = 0;
    int delays = 0;
    double refreshMs = NAN;
    int multiple = 1;
    double frameMs = 0, landMs = NAN, row = 0;
  };
  using Say = std::function<void(const std::string&)>;

  // ai: say: the lock's log lines; debug: its readings and looks too (the Kotlin's -Dphase.debug=1)
  explicit PhaseLock(Say say, bool debug = false);

  // ai: Track starts its hold and the clocks afresh, Off stops it; what the camera's timestamps have measured
  // ai: stands either way (the intervals are the camera's, not the mode's).
  void set(Mode m);
  Mode mode() const { return mode_; }
  // ai: The camera was opened again (a setting changed, the app came back, a logical camera switched its lens): a new
  // ai: session, its frames starting anywhere in the display's refresh and maybe on a clock restarted below the old
  // ai: one; the display's refresh is the display's and stands, and the view's and the hold's are learned afresh.
  void reopened();
  // ai: A capture's result: its timestamp, whether it was a delayed frame, and the camera's readout skew over the
  // ai: sensor's rows (0 where the host has none). Returns the microseconds to delay the next frame by, 0 for none.
  int64_t onCapture(int64_t tsNs, bool delayed, int64_t skewNs = 0);
  // ai: What became of the delay asked, as the host hears it: taken, its frame captured though its result failed (its
  // ai: interval then shows the delay as a marked frame's would); not taken (the host refused it, its request failed
  // ai: before its frame, its sequence was aborted): given up at once.
  void delayFate(bool taken);
  // ai: The receiver's newest frames and the version the word names (0 before any word). A frame captured before
  // ai: this session's first capture is another session's and is left out.
  void onWindow(const std::vector<Frame>& given, int version);
  // ai: Receiver::series's vector ([the held version, then 8 doubles a frame: ms on the camera's clock, verified,
  // ai: new, found, r, its error, r2, its error]) as the lock's frames, each ms made ns by truncation and no code
  // ai: row, as the Android app's feed read it; version the vector's first value (0 for an empty one).
  static std::vector<Frame> framesOf(const std::vector<double>& series, int& version);
  // ai: The newest snapshot (State's header); none before the first decision of a hold.
  std::optional<State> state() const;
  // ai: For the log, every few seconds: what is held or owed, the delays since the last call, the hold's newest
  // ai: leaks, and what a bad start is told by (the prefix "phase: track" stays, as the tools' readers match it).
  // ai: None in Off.
  std::optional<std::string> line(int64_t tsNs);

  // ai: the most the pace may be (and how far a camera's frame may be off a whole number of refreshes); how late a
  // ai: capture may start of the line before a period less is asked; the span of frames, places and landings the
  // ai: hold keeps; how many of its errors a reading must be; the least time a flat top's edge reached again is
  // ai: reckoned over; the frames of camera time after which a delay nothing was heard of is given up (and the
  // ai: captures after its mark a taken delay may land while none has landed)
  static constexpr double PACE_MOST = 5e-3;
  static constexpr int64_t TOOTH_NS = 1'000'000;
  static constexpr int64_t LONG_NS = 12'000'000'000;
  static constexpr double Z = 3.0;
  static constexpr int64_t PACED_NS = 1'000'000'000;
  static constexpr int LOST = 60;
  // ai: the pilots: |r| at which a reading's sign is not noise; the least error a reading is given (its own at
  // ai: LIZARD-512 is about 0.02 over half the blocks); the frames an estimate of the leaks needs; the median |r|
  // ai: under which the frames are deep in a mix; the ms the line moves a unit of a - b before any move has measured
  // ai: it (under twice the least a display's own is, so a first move cannot grow the error: a window of 9 ms that
  // ai: changes pictures at one instant has 9, the S26's arithmetic about 20), and the least and most a measurement
  // ai: may say; the step in from the edge of a flat top; the span of the newest frames the leaks are read from (and
  // ai: the intervals the camera's frame is read from); the share of a neighbour under which a capture is taken to
  // ai: hold none of it; what a pair of readings keeps of its weight in the fit with each pair after it (unused); how
  // ai: early a capture is delayed back onto the line; the code's row before a frame has told it (the frame's middle,
  // ai: where the receiver's crops are)
  static constexpr double R_FLASH = 0.25;
  static constexpr double R_NOISE_MIN = 0.005;
  static constexpr int L_NEED = 12;
  static constexpr double DEEP = 0.35;
  static constexpr double GAIN_MS = 12.0;
  static constexpr double GAIN_LEAST = 2.0;
  static constexpr double GAIN_MOST = 40.0;
  static constexpr int64_t CROSS_NS = 2'000'000;
  static constexpr int64_t RECENT_NS = 1'000'000'000;
  static constexpr double LEAK = 0.02;
  static constexpr double FADE = 0.85;
  static constexpr int64_t BAND_NS = 250'000;
  static constexpr double ROW = 0.5;
  // ai: the estimate (observe). Before any look: the pace as far off as two clocks may be, the gain anywhere between
  // ai: its least and most (a uniform's variance). The view's wander, a random walk of the code's read time: a
  // ai: hand-held code wanders over half the frame's readout within a minute, re-aims and all (the S26's 6.26 ms
  // ai: readout: (3.13 ms)² a minute); a hand still aiming sweeps the code across half the frame in a second, 3 ms a
  // ai: second of read time, for the session's first three seconds (AIM_NS), three readings of RECENT_NS, which the
  // ai: newest N_EVID looks' median outvotes (seven outvote three). Process noise: the gain is half the window over
  // ai: the code, which a hand moves about a millisecond in ten seconds, so (0.5 ms a unit)² in ten seconds; the
  // ai: slide is the clocks' (which part by about 1 ppm a minute as a phone warms) and a hand's slow turn (about a
  // ai: degree a second, 0.15 ms of read time a second on the S26, which changes within about ten seconds): (0.15 ms
  // ai: a second)² in ten seconds.
  static constexpr double PACE_VAR0 = PACE_MOST * PACE_MOST;
  static constexpr double GAIN_VAR0 = (GAIN_MOST - GAIN_LEAST) * (GAIN_MOST - GAIN_LEAST) / 12;
  static constexpr double VIEW_MS2 = 3.13 * 3.13 / 60;
  static constexpr double AIM_MS2 = 9.0;
  static constexpr int N_EVID = 7;
  static constexpr int64_t AIM_NS = 3'000'000'000;
  static constexpr double GAIN_Q = 0.025;
  static constexpr double PACE_Q = 0.15e-3 * 0.15e-3 / 10;

 private:
  // ai: t: the frames' mean time; k: what a capture of one picture reads; x: the frames' mean place against the
  // ai: line (ns)
  struct Leak {
    double a, b, sa, sb;
    int n;
    int64_t t;
    double k, x;
  };

  void hold();
  void steps();
  void landed(int64_t ts, int64_t dt, bool delayed);
  void giveUp();
  std::vector<Frame> captures(const std::vector<Frame>& fs) const;
  Frame merged(const std::vector<Frame>& g, size_t from, size_t to) const;
  int64_t track(int64_t ts, bool delayed);
  void here();
  std::optional<Leak> leaks(int64_t from, int64_t to = INT64_MAX) const;
  void tracked(const std::vector<Frame>& got);
  void decide();
  void learn(int64_t t, double de, double dt, double h, double paceMs, double w);
  void observe(int64_t t, double z, double hg, double hp, double rRead, double dt, bool both);
  void age(int64_t t);
  void check();
  void held();
  void snap();
  double sure() const { return std::sqrt(paceVar); }
  double refresh() const { return base * (1 + pace); }
  double period() const { return refresh() * multiple; }
  const char* standing() const { return line_ < 0 ? "no frames yet" : "holds"; }
  double landMs() const;

  Say say;
  bool debug;
  mutable std::mutex stateMu;
  std::optional<State> state_;   // ai: stateMu

  Mode mode_ = Mode::Off;
  int delays = 0;
  int delaysAll = 0;   // ai: delays asked since the mode was set (State)

  // ai: the camera's: the session's first capture and its newest (frames captured outside them are another
  // ai: session's), the newest frame of the series counted, whether the newest capture was a delayed one, its readout
  // ai: skew (ns, the sensor's first row to one past its last); the newest RECENT_NS of intervals between two frames
  // ai: not delayed (end, length) and their median, the camera's frame
  int64_t first = 0;
  int64_t now = 0;
  int64_t seen = 0;
  bool wasDelayed = false;
  int64_t skew = 0;
  std::deque<std::array<int64_t, 2>> plains;
  int64_t frame = 0;
  // ai: the delay out (ns; 0: none), whether its frame was marked delayed (or the camera said it took it) and the
  // ai: captures since, whether the capture before ended an interval that long, the camera time it was asked at; and
  // ai: the newest LONG_NS of landings (when, ns from the ask, captures from the mark)
  int64_t owed = 0;
  bool flagged = false;
  int flagAt = 0;
  bool longPrev = false;
  int64_t askT = 0;
  std::deque<std::array<int64_t, 3>> lands;
  // ai: the clocks': the display's refresh on the camera's clock is base (1 + pace) (base: the camera's frame when the
  // ai: refresh was last taken from it; 0: not yet), how well the pace is known (ns a ns; PACE_VAR0: not at all),
  // ai: and the whole number of refreshes in the camera's frame, which the line steps
  int64_t base = 0;
  double pace = 0.0;
  double paceVar = PACE_VAR0;   // ai: (ns a ns)²
  int64_t paceAt = 0;           // ai: the camera time the estimate's process noise was last brought to
  int multiple = 1;
  // ai: the hold
  double row = ROW;         // ai: the code's row, the newest frame's that has one
  int64_t settle = 0;       // ai: frames after this one were captured after the last move landed
  int64_t line_ = -1;       // ai: the read time of a capture on the phase held, the one nearest the newest capture; -1: none
  bool move = false;        // ai: the line was moved: the camera follows at once
  bool moving = false;      // ai: the delay that follows it is on its way
  std::deque<Frame> window;  // ai: the hold's frames, the newest LONG_NS
  int64_t since = 0;        // ai: frames captured after this are the hold's
  double level = 0.0;       // ai: blocks a frame (the log's)
  // ai: each capture's timestamp against the line when it was taken (ts - line, ns) and its skew, the newest
  // ai: LONG_NS: a frame's place is that and the skew times its own row
  std::map<int64_t, std::array<int64_t, 2>> placed;
  // ai: the pilots' fit: ms the line moves a unit of a - b. The reading before (a - b, its error, its frames' mean
  // ai: time and place against the line; 0: none that showed a neighbour), the move made on it (ms) and the pace it
  // ai: was made at.
  double gain = GAIN_MS;
  double gainVar = GAIN_VAR0;   // ai: (ms a unit)²
  double gpCov = 0.0;           // ai: the gain's and the pace's covariance (ms a unit, ns a ns)
  double prevE = 0.0;
  double prevSe = 0.0;
  int64_t prevT = 0;
  double prevHop = 0.0;
  double prevPace = 0.0;
  double prevX = 0.0;
  int prevSide = 0;   // ai: which neighbours it showed: 1 the one before, 2 the one after, 3 both
  // ai: the session's newest N_EVID looks at the gain and the pace (z ms, its gain's and pace's coefficients, its
  // ai: variance ms², its readings' variance, the gain and its variance before it, its camera time)
  std::deque<std::array<double, 8>> evid;
  int64_t aimFrom = 0;    // ai: the session's first hold began (the code in view): a hand may still aim for AIM_NS
  bool balanced = false;  // ai: this hold has said it stands (a line for the log, once a hold)
  std::optional<Leak> lastLeak;   // ai: the newest estimate of the leaks (State, line)
  // ai: a top that is flat (a capture there holds neither neighbour): the edge the hold last stepped in from (1: the
  // ai: early edge, so the step was later; -1: the late; 0: none), how far in it stepped (ns) and the frame it did so
  // ai: at; half the top's width where a step in from one edge met the other (0: not known)
  int edgeDir = 0;
  int64_t edgeIn = 0;
  int64_t edgeT = 0;
  bool edgeClean = false;   // ai: frames since that step have held neither neighbour (it landed inside the top)
  int64_t flatHalf = 0;
  // ai: whether the hold has asked no move (State.stood), until a mix too deep to read or a new hold
  bool stood = false;

  friend struct PhaseLockProbe;   // ai: tools/phase_lock_check.cpp: every field above, for the side-by-side check
};

}  // namespace lizard
