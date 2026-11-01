// ai: The camera's phase against the display (2026-09-30; in C++ since 2026-10-31, liblizard/kotlin/PhaseLock.kt
// ai: decision for decision: tools/phase_lock_check.cpp holds the two to the same trace).
// ai: A capture that holds a change of the painted frame is two symbols and reads little, and whether captures hold
// ai: one depends on where they start in the display's refresh. A camera and a display each keep their own clock, so
// ai: that place slides: the S26's frame is 16.650 ms, 1.1 ms a second against a 60.00 Hz display, every phase in
// ai: 16 s. One frame made longer moves every frame after it (the host's delay). That is the only lever: a delay of x
// ai: moves the phase x later, and a delay of a whole period less x moves it x earlier (a capture skipped).
// ai:
// ai: Track holds by the pilots (research/SPEC.md 7.3, signed). The sender paints its count of pictures mod 4 into the
// ai: block tails, bit 0 on the even blocks and bit 1 on the odd, and a capture reads r (the even blocks) and r2 (the
// ai: odd): their signs the count of the picture it mostly holds, and their sizes what it holds of the pictures either
// ai: side, each neighbour lowering the reading whose bit it does not share. Every frame of the format carries both
// ai: readings, and a frame without them is no evidence. So the hold needs no search and no sweep:
// ai:   the hold begins where the camera is (`here`), on the first frames;
// ai:   the leaks (`leaks`): over the frames since the last move, each reading is K less twice K times the share
// ai:     of each neighbour whose bit differs (which neighbours do is read off the neighbouring captures' counts,
// ai:     so a picture the screen showed twice or skipped is counted for what it was, not guessed): K, the share
// ai:     a of the picture before and b of the one after by least squares, each reading by its own error;
// ai:   the move: a capture that starts too early holds some of the picture before, one too late some of the
// ai:     next, so the line goes later by `gain` ms a unit of a - b wherever that is Z of its errors from 0; the
// ai:     gain is measured by the moves (what the captures' places moved over what it changed a - b by);
// ai:   deep in a mix (no capture mostly one picture, so no count): a quarter of a refresh later, and look again;
// ai:   the pace: from pairs of readings (`learn`) and from a flat top's edge reached again.
// ai: Nothing it learns is ever certain. The pace and the gain are one estimate with its covariance (`observe`): every
// ai: pair of readings and every edge reached again is one look at them, with the variance its readings, the gain's
// ai: own error, the view's wander over its span (a hand's, re-aims and all: VIEW_MS2) and, over what of it was the
// ai: AIM_NS after the session's first hold began (the code in view), a hand still aiming (AIM_MS2) give it; a look
// ai: beyond Z of what is expected moves them only as far as Z (the view jumps where a hand aims; a jump is the
// ai: view's, not the clocks'); both drift by their own process noise between looks (GAIN_Q: the window over the code
// ai: moves with a hand; PACE_Q: the slide the hold sees is the clocks' and a hand's slow turn, which changes within
// ai: seconds). The newest N_EVID looks (of the newest LONG_NS) re-check the pace: where their median (a vote each, at
// ai: the gain as it is) is beyond Z of its error and the pace's own, the pace is that median with its error (a bad
// ai: start, a view's jump taken for a slide, a kept pace another screen refutes), and the gain goes back to what it
// ai: was before those looks (they told it at a pace that was wrong). So the pace a reopen keeps is a prior the new
// ai: session's looks confirm or overturn, and what the log claims is the estimate's own error.
// ai: A capture's place is the time its code is read: the camera reads its rows down the sensor over its readout skew
// ai: (Camera2's SENSOR_ROLLING_SHUTTER_SKEW, its first row to one past its last) and stamps a capture with its first
// ai: row's (SENSOR_TIMESTAMP), so the code's mean row (the receiver's quad, a frame of the series each, as a share of
// ai: the sensor's rows) is read at the timestamp plus the skew times that share. A code that moves up or down the
// ai: picture (a turn of the phone, an aim moved) moves where it is read and so the mix; held on its read time, the
// ai: camera follows it by delays, and no balance moves and no slide is left for the pace to learn.
// ai: Self-healing: what the hold knows belongs to what it describes, and goes when that does.
// ai:   The camera's (its frame, the delays' landings, the session's first and newest capture, the newest frame
// ai:     counted): from its own timestamps whatever the mode; afresh for a new session (`reopened`), never at a set.
// ai:   The clocks' (the display's refresh on the camera's clock and how well it is known): the state the line steps
// ai:     by, apart from the camera's frame, which only says how many refreshes a capture spans; kept through a
// ai:     reopen, afresh at a set of the mode; a frame that is no whole number of refreshes is another display's.
// ai:   The view's (the gain and its fit, a flat top's width, the code's row) and the hold's place: afresh at a
// ai:     reopen, and the place where the camera's frame changes; a flat top's width measured again once the hold has
// ai:     stood clear of its edges for LONG_NS, and the gain bounded by a step that crossed the top.
// ai: A delay lands where the timestamps show it (an interval longer than the camera's frame by over half of it).
// ai: What became of the request the host says (`delayFate`: the camera took it, or it never will: refused, failed
// ai: before its frame, its sequence aborted), so a delay the camera did not take is known, not waited for. One the
// ai: camera took whose interval has not shown is given up once its mark is further behind than the camera's own
// ai: newest landings were behind theirs, and one more capture (a frame's duration sets the interval to the frame
// ai: after it). Of a delay nothing is heard of, the camera's clock past its ask by LOST frames (taken at the newest
// ai: time the camera is known to have reached, so a backlog after a stall of the host's thread gives up nothing on
// ai: its way) gives it up: Camera2 promises one or the other for every request, so only a host that breaks that
// ai: leaves one there.
// ai: The arithmetic is the JVM's, so the two decide alike to the bit: Math.round (ties toward positive infinity),
// ai: Math.max and Math.min (NaN and signed zeros), a double's conversions to integers (truncated, NaN 0, saturated),
// ai: the sums in the Kotlin's order, and fdlibm's hypot as the JVM has it.
#include "phase_lock.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>

namespace lizard {

namespace {

// ai: Java's Math.round: the closest integer, ties toward positive infinity; NaN 0, saturated at the ends
int64_t jround(double a) {
  uint64_t u;
  std::memcpy(&u, &a, sizeof u);
  const int64_t bits = (int64_t)u;
  const int64_t biased = (int64_t)((u & 0x7FF0000000000000ULL) >> 52);
  const int64_t shift = (53 - 2 + 1023) - biased;
  if ((shift & -64) == 0) {
    int64_t r = (int64_t)((u & 0x000FFFFFFFFFFFFFULL) | 0x0010000000000000ULL);
    if (bits < 0) r = -r;
    return ((r >> shift) + 1) >> 1;
  }
  if (std::isnan(a)) return 0;
  if (a >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (a <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return (int64_t)a;
}

// ai: Kotlin's Double.toLong and Double.toInt: toward zero, NaN 0, saturated
int64_t jlong(double a) {
  if (std::isnan(a)) return 0;
  if (a >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (a <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return (int64_t)a;
}
int jint(double a) {
  if (std::isnan(a)) return 0;
  if (a >= 2147483647.0) return std::numeric_limits<int>::max();
  if (a <= -2147483648.0) return std::numeric_limits<int>::min();
  return (int)a;
}

// ai: Java's Math.max and Math.min: NaN where either is, -0.0 under 0.0
double jmax(double a, double b) {
  if (a != a) return a;
  if (a == 0.0 && b == 0.0 && std::signbit(a)) return b;
  return a >= b ? a : b;
}
double jmin(double a, double b) {
  if (a != a) return a;
  if (a == 0.0 && b == 0.0 && std::signbit(b)) return b;
  return a <= b ? a : b;
}

// ai: Kotlin's coerceIn
double coerce(double v, double lo, double hi) { return v < lo ? lo : v > hi ? hi : v; }
int64_t coerce(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }

// ai: Java's sort order for doubles (Arrays.sort): -0.0 before 0.0, NaN last
bool jless(double a, double b) {
  if (a < b) return true;
  if (a > b) return false;
  const bool na = a != a, nb = b != b;
  if (na || nb) return !na && nb;
  return std::signbit(a) && !std::signbit(b);
}

// ai: fdlibm's hypot as the JVM has it (StrictMath.hypot, FdLibm.Hypot, which Math.hypot is): sqrt(x² + y²) with no
// ai: overflow or underflow, under one ulp, the same bits on every platform
int32_t hi32(double x) {
  uint64_t u;
  std::memcpy(&u, &x, 8);
  return (int32_t)(u >> 32);
}
double withHi(double x, int32_t hi) {
  uint64_t u;
  std::memcpy(&u, &x, 8);
  u = ((uint64_t)(uint32_t)hi << 32) | (u & 0xFFFFFFFFULL);
  std::memcpy(&x, &u, 8);
  return x;
}
double jhypot(double x, double y) {
  double a = std::fabs(x), b = std::fabs(y);
  if (!std::isfinite(a) || !std::isfinite(b)) {
    if (a == INFINITY || b == INFINITY) return INFINITY;
    return a + b;   // ai: NaN's significand carried
  }
  if (b > a) std::swap(a, b);
  int32_t ha = hi32(a), hb = hi32(b);
  if (ha - hb > 0x3c00000) return a + b;   // ai: x / y > 2**60
  int k = 0;
  if (a > 0x1.00000ffffffffp500) {   // ai: a > ~2**500: a and b scaled by 2**-600
    ha -= 0x25800000;
    hb -= 0x25800000;
    a = a * 0x1.0p-600;
    b = b * 0x1.0p-600;
    k += 600;
  }
  double t1, t2;
  if (b < 0x1.0p-500) {   // ai: b < 2**-500
    if (b < std::numeric_limits<double>::min()) {   // ai: subnormal b or 0
      if (b == 0.0) return a;
      t1 = 0x1.0p1022;
      b *= t1;
      a *= t1;
      k -= 1022;
    } else {   // ai: a and b scaled by 2**600
      ha += 0x25800000;
      hb += 0x25800000;
      a = a * 0x1.0p600;
      b = b * 0x1.0p600;
      k -= 600;
    }
  }
  // ai: medium size a and b
  double w = a - b;
  if (w > b) {
    t1 = withHi(0.0, ha);
    t2 = a - t1;
    w = std::sqrt(t1 * t1 - (b * (-b) - t2 * (a + t1)));
  } else {
    a = a + a;
    const double y1 = withHi(0.0, hb);
    const double y2 = b - y1;
    t1 = withHi(0.0, ha + 0x00100000);
    t2 = a - t1;
    w = std::sqrt(t1 * y1 - (w * (-w) - (t1 * y2 + t2 * b)));
  }
  return k != 0 ? std::ldexp(1.0, k) * w : w;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  const int n = std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  if (n < 0) return std::string();
  if ((size_t)n < sizeof buf) return std::string(buf, (size_t)n);
  std::string s((size_t)n + 1, '\0');
  va_start(ap, f);
  std::vsnprintf(&s[0], s.size(), f, ap);
  va_end(ap);
  s.resize((size_t)n);
  return s;
}

double median(std::vector<double> s) {
  std::sort(s.begin(), s.end(), jless);
  return s.size() % 2 == 1 ? s[s.size() / 2] : (s[s.size() / 2 - 1] + s[s.size() / 2]) / 2;
}

// ai: three equations in three unknowns, m a 3 x 7 array (the matrix, the right side, the identity): solved in place
// ai: by elimination, the answers m[k][3] / m[k][k] and the inverse's diagonal m[k][4 + k] / m[k][k]; false where
// ai: they do not determine the unknowns
using M37 = std::array<std::array<double, 7>, 3>;
bool solve3(M37& m) {
  for (int c = 0; c < 3; c++) {
    int r = c;
    for (int k = c + 1; k < 3; k++)
      if (std::fabs(m[k][c]) > std::fabs(m[r][c])) r = k;
    std::swap(m[c], m[r]);
    if (std::fabs(m[c][c]) < 1e-9) return false;
    for (int k = 0; k < 3; k++)
      if (k != c) {
        const double f = m[k][c] / m[c][c];
        for (int j = 0; j <= 6; j++) m[k][j] -= f * m[c][j];
      }
  }
  return true;
}

}  // namespace

PhaseLock::PhaseLock(Say s, bool d) : say(std::move(s)), debug(d) {
  if (!say) say = [](const std::string&) {};
}

void PhaseLock::set(Mode m) {
  if (m == mode_) return;
  mode_ = m;
  base = 0;
  pace = 0.0;
  paceVar = PACE_VAR0;
  paceAt = 0;
  multiple = 1;
  hold();
  delaysAll = 0;
  say(m == Mode::Track ? "phase: track" : "phase: off");
}

void PhaseLock::reopened() {
  first = 0;
  now = 0;
  seen = 0;
  wasDelayed = false;
  skew = 0;
  plains.clear();
  frame = 0;
  owed = 0;
  flagged = false;
  longPrev = false;
  askT = 0;
  lands.clear();
  hold();
}

void PhaseLock::hold() {
  row = ROW;
  settle = 0;
  line_ = -1;
  move = false;
  moving = false;
  window.clear();
  since = 0;
  level = 0.0;
  placed.clear();
  gain = GAIN_MS;
  gainVar = GAIN_VAR0;
  gpCov = 0.0;
  evid.clear();
  aimFrom = 0;
  prevT = 0;
  balanced = false;
  lastLeak.reset();
  edgeDir = 0;
  flatHalf = 0;
  stood = false;
  std::lock_guard<std::mutex> l(stateMu);
  state_.reset();
}

int64_t PhaseLock::onCapture(int64_t tsNs, bool delayed, int64_t skewNs) {
  if (tsNs == 0) return 0;
  if (now == 0) {
    first = tsNs;
    settle = tsNs - 1;
  }
  const int64_t dt = now != 0 ? tsNs - now : 0;
  const bool plain = dt > 0 && !delayed && !wasDelayed;
  // ai: the camera's frame is the median of the newest RECENT_NS of intervals, and it is that only once the session
  // ai: has run that long (a session's first intervals are often irregular, and a median of one is the one, a dropped
  // ai: frame among them): until then a session that kept the clocks takes their period for it and nothing is taken
  // ai: from it; a cold start, with nothing kept, begins from what is in and is put right then
  const bool known = tsNs - first >= RECENT_NS;
  if (plain) {
    plains.push_back({tsNs, dt});
    while (plains.front()[0] < tsNs - RECENT_NS) plains.pop_front();
    std::vector<int64_t> d;
    d.reserve(plains.size());
    for (const auto& p : plains) d.push_back(p[1]);
    std::sort(d.begin(), d.end());
    frame = !known && base != 0 ? jround(period()) : d[(d.size() - 1) / 2];
  }
  wasDelayed = delayed;
  now = tsNs;
  skew = std::max<int64_t>(0, skewNs);
  landed(tsNs, dt, delayed);
  if (plain && base != 0 && known) steps();
  if (mode_ == Mode::Off) return 0;
  return track(tsNs, delayed);
}

// ai: The refreshes a capture spans: the whole number nearest the camera's frame over the refresh. A frame no whole
// ai: number of refreshes makes to within PACE_MOST (the most two clocks differ) is not this refresh's (one taken
// ai: from a camera at another rate, or another screen): the refresh taken from the camera's frame afresh. Either way
// ai: the hold begins again where the camera is: its frames so far were captured a refresh or more apart from what the
// ai: line steps now, so neither their places nor their neighbours (a capture's neighbours are the pictures either
// ai: side only where it spans one refresh) say anything of the new hold.
void PhaseLock::steps() {
  const int64_t m = jround((double)frame / refresh());
  if (m >= 1 && std::fabs((double)frame / ((double)m * refresh()) - 1) <= PACE_MOST) {
    if ((int)m == multiple) return;
    say(fmt("phase: the camera's frame %.3f ms is %lld refreshes of %.3f ms", frame / 1e6, (long long)m, refresh() / 1e6));
    multiple = (int)m;
  } else {
    say(fmt("phase: the camera's frame %.3f ms is no whole number of refreshes of %.3f ms: the refresh taken from it", frame / 1e6,
            refresh() / 1e6));
    base = frame;
    pace = 0.0;
    paceVar = PACE_VAR0;
    gpCov = 0.0;
    multiple = 1;
    evid.clear();
  }
  if (line_ >= 0) here();
}

// ai: The delay out, whatever the mode: landed at the first interval longer than the camera's frame by over half the
// ai: delay, from the capture before its marked frame on (a camera that applies the longer frame captures after the
// ai: mark lands it there; on the S26 at the mark): only frames after it are the move's. Given up (the header says
// ai: when), and the camera asked onto the line afresh (a pair of readings across it counts what the captures' places
// ai: show it did, nothing if it never landed).
void PhaseLock::landed(int64_t ts, int64_t dt, bool delayed) {
  if (owed == 0) return;
  const bool lng = dt - frame > owed / 2;
  const bool was = longPrev;
  longPrev = lng;
  if (delayed) {
    flagged = true;
    flagAt = 0;
  }
  if (flagged && (lng || (delayed && was))) {
    owed = 0;
    lands.push_back({ts, ts - askT, (int64_t)flagAt});
    while (lands.front()[0] < ts - LONG_NS) lands.pop_front();
    if (moving) {
      moving = false;
      settle = ts;
    }
    return;
  }
  bool lost;
  if (flagged) {
    int most = LOST;
    if (!lands.empty()) {
      int64_t m = lands.front()[2];
      for (const auto& l : lands) m = std::max(m, l[2]);
      most = (int)m;
    }
    lost = ++flagAt > most + 1;
  } else {
    lost = ts - askT > LOST * frame;
  }
  if (lost) {
    if (debug) {
      const std::string how = flagged ? fmt("taken, %d captures on", flagAt) : std::string("nothing heard");
      say(fmt("phase: DEBUG a delay of %.3f ms not landed %.0f ms after its ask (%s): given up", owed / 1e6, (ts - askT) / 1e6, how.c_str()));
    }
    giveUp();
  }
}

void PhaseLock::giveUp() {
  owed = 0;
  if (moving) {
    moving = false;
    move = true;
  }
}

void PhaseLock::delayFate(bool taken) {
  if (owed == 0) return;
  if (taken) {
    flagged = true;
    flagAt = 0;
  } else {
    giveUp();
  }
}

void PhaseLock::onWindow(const std::vector<Frame>& given, int version) {
  if (now == 0) return;
  std::vector<Frame> mine;
  mine.reserve(given.size());
  for (const Frame& f : given)
    if (f.ts >= first) mine.push_back(f);
  const std::vector<Frame> frames = captures(mine);
  if (frames.empty()) return;
  int64_t newest = frames.front().ts;
  for (const Frame& f : frames) newest = std::max(newest, f.ts);
  if (mode_ == Mode::Track && version > 0) {
    std::vector<Frame> got;
    for (const Frame& f : frames)
      if (f.ts > seen && f.found && !std::isnan(f.r) && !std::isnan(f.r2)) got.push_back(f);
    std::stable_sort(got.begin(), got.end(), [](const Frame& a, const Frame& b) { return a.ts < b.ts; });
    tracked(got);
  }
  seen = std::max(seen, newest);
}

std::vector<PhaseLock::Frame> PhaseLock::framesOf(const std::vector<double>& a, int& version) {
  version = a.empty() ? 0 : jint(a[0]);
  std::vector<Frame> out;
  if (a.empty()) return out;
  const size_t n = (a.size() - 1) / 8;
  out.reserve(n);
  for (size_t i = 0; i < n; i++) {
    const size_t o = 1 + 8 * i;
    Frame f;
    f.ts = jlong(a[o] * 1e6);
    f.blocks = jint(a[o + 1]);
    f.found = a[o + 3] == 1.0;
    f.r = a[o + 4];
    f.rSd = a[o + 5];
    f.r2 = a[o + 6];
    f.r2Sd = a[o + 7];
    out.push_back(f);
  }
  return out;
}

// ai: One frame a capture. The 2:1 crop gives the receiver two frames a capture (its halves, the capture's timestamp
// ai: both), and they are one look at one mix of pictures: as two frames, each half's neighbouring capture in `leaks`
// ai: was its own twin and each capture's mix counted twice as evidence. Frames of one timestamp are one: each pilot
// ai: reading their mean by inverse variance, the blocks and the code's row their mean, found where either was.
std::vector<PhaseLock::Frame> PhaseLock::captures(const std::vector<Frame>& fs) const {
  if (fs.size() < 2) return fs;
  std::vector<Frame> out;
  out.reserve(fs.size());
  size_t i = 0;
  while (i < fs.size()) {
    size_t j = i + 1;
    while (j < fs.size() && fs[j].ts == fs[i].ts) j++;
    out.push_back(j - i == 1 ? fs[i] : merged(fs, i, j));
    i = j;
  }
  return out;
}

PhaseLock::Frame PhaseLock::merged(const std::vector<Frame>& g, size_t from, size_t to) const {
  auto mean = [&](double Frame::*v, double Frame::*sd, double& m, double& e) {
    double w = 0.0, s = 0.0;
    for (size_t i = from; i < to; i++) {
      const double x = g[i].*v;
      if (std::isnan(x)) continue;
      double q = g[i].*sd;
      if (std::isnan(q) || q <= 0) q = R_NOISE_MIN;
      w += 1 / (q * q);
      s += x / (q * q);
    }
    if (w == 0.0) {
      m = NAN;
      e = NAN;
    } else {
      m = s / w;
      e = 1 / std::sqrt(w);
    }
  };
  Frame f;
  f.ts = g[from].ts;
  mean(&Frame::r, &Frame::rSd, f.r, f.rSd);
  mean(&Frame::r2, &Frame::r2Sd, f.r2, f.r2Sd);
  int blocks = 0;
  bool found = false;
  double rows = 0.0;
  int nRows = 0;
  for (size_t i = from; i < to; i++) {
    blocks += g[i].blocks;
    found = found || g[i].found;
    if (!std::isnan(g[i].row)) {
      rows += g[i].row;
      nRows++;
    }
  }
  f.blocks = (int)jround((double)blocks / (double)(to - from));
  f.found = found;
  f.row = nRows == 0 ? NAN : rows / nRows;
  return f;
}

// ai: A capture (track): the delay to ask, us. Where a capture sat comes from its own timestamp and its code's row,
// ai: never from what was asked of the camera, so a delay that lands late, short or not at all costs nothing but time.
int64_t PhaseLock::track(int64_t ts, bool delayed) {
  if (base == 0 || line_ < 0) return 0;
  // ai: its code's read time, and the line's point nearest it
  const int64_t rt = ts + jround((double)skew * row);
  const int64_t k = jround((double)(rt - line_) / period());
  if (k != 0) line_ += jround((double)k * period());
  placed[ts] = {ts - line_, skew};
  while (placed.begin()->first < ts - LONG_NS) placed.erase(placed.begin());
  // ai: nothing is asked on a delayed frame's own timestamp, nor while a delay is out
  if (owed > 0 || delayed) return 0;
  int64_t ns = 0;
  // ai: early: delayed onto the line as soon as it is a band early (a move's, at once). Late (a camera slower than the
  // ai: display, a line moved earlier, a code moved down the picture): a period less what it is late, a capture skipped
  const int64_t early = line_ - rt;
  const int64_t far = move ? jround(refresh() / 256) : TOOTH_NS;
  if (early >= std::min(far, BAND_NS)) {
    ns = early;
  } else if (early <= -far) {
    ns = jround(period()) + early;
  }
  if (move) {
    if (ns != 0) {
      moving = true;
    } else {
      settle = ts;
    }
    move = false;
  }
  if (ns <= 0) return 0;
  owed = ns;
  flagged = false;
  flagAt = 0;
  longPrev = false;
  askT = std::max(ts, seen);
  delays++;
  delaysAll++;
  return std::max<int64_t>(1, ns / 1000);
}

// ai: The hold begun where the camera is: the line through the newest capture's read time, so the camera is on it and
// ai: its frames from then on are the hold's; the clocks as they are (from the camera's frame, where the refresh is not
// ai: known yet).
void PhaseLock::here() {
  if (frame == 0) return;
  if (base == 0) {
    base = frame;
    pace = 0.0;
    paceVar = PACE_VAR0;
    gpCov = 0.0;
    multiple = 1;
  }
  prevT = 0;
  edgeDir = 0;
  flatHalf = 0;
  balanced = false;
  stood = false;
  lastLeak.reset();
  placed.clear();
  if (aimFrom == 0) aimFrom = now;
  line_ = now + jround((double)skew * row);
  move = false;
  moving = false;
  window.clear();
  since = now;
  settle = now;
  level = 0.0;
  say(fmt("phase: the hold begins: the refresh %.4f ms on the camera's clock, %d a frame of %.4f ms, the code at row %.2f", refresh() / 1e6,
          multiple, frame / 1e6, row));
}

// ai: What the hold's frames after `from` hold of their neighbours. A frame's count is its two signs (bit 0 from r,
// ai: bit 1 from r2), and so are the captures' before and after it, a capture either side (a frame whose neighbour is
// ai: not in the series, or any of the three with a reading under R_FLASH, whose sign may be noise, is left out). Each
// ai: of its readings is then, in size, K - 2 K a [the bit differs from the count before] - 2 K b [it differs from the
// ai: count after]: K what a capture of one picture reads, a and b the shares of the pictures before and after. Least
// ai: squares over both readings of every frame, each by its own error, for K, K a and K b. None: too few frames, or
// ai: frames that do not tell the three apart (a sender whose count does not change: no bit ever differs).
std::optional<PhaseLock::Leak> PhaseLock::leaks(int64_t from, int64_t to) const {
  const std::deque<Frame>& fs = window;
  const double t = period();
  M37 m{};
  for (int k = 0; k < 3; k++) m[k][4 + k] = 1.0;
  int n = 0;
  double ts = 0.0, xs = 0.0;
  int xn = 0;
  auto sure = [](const Frame& f) { return std::fabs(f.r) >= R_FLASH && std::fabs(f.r2) >= R_FLASH; };
  auto count = [](const Frame& f) { return (f.r < 0 ? 1 : 0) | (f.r2 < 0 ? 2 : 0); };
  for (size_t i = 1; i + 1 < fs.size(); i++) {
    const Frame& f = fs[i];
    const Frame& p = fs[i - 1];
    const Frame& q = fs[i + 1];
    if (f.ts <= from || f.ts > to || (double)(f.ts - p.ts) > t * 3 / 2 || (double)(q.ts - f.ts) > t * 3 / 2 || !sure(f) || !sure(p) || !sure(q))
      continue;
    const int c = count(f), before = count(p) ^ c, after = count(q) ^ c;
    for (int bit = 0; bit <= 1; bit++) {
      const double y = std::fabs(bit == 0 ? f.r : f.r2);
      double e = bit == 0 ? f.rSd : f.r2Sd;
      if (std::isnan(e)) e = R_NOISE_MIN;
      const double sd = jmax(R_NOISE_MIN, e);
      const double w = 1 / (sd * sd);
      const double x[3] = {1.0, -2.0 * ((before >> bit) & 1), -2.0 * ((after >> bit) & 1)};
      for (int u = 0; u < 3; u++) {
        for (int v = 0; v < 3; v++) m[u][v] += w * x[u] * x[v];
        m[u][3] += w * x[u] * y;
      }
    }
    n++;
    ts += (double)(f.ts - from);
    // ai: the frame's place as its capture had it, at the frame's own row (the series' times are the captures' to the
    // ai: ns, give or take one)
    auto it = placed.upper_bound(f.ts + frame / 2);
    if (it != placed.begin()) {
      --it;
      if (it->first > f.ts - frame / 2) {
        xs += (double)it->second[0] + (double)it->second[1] * (std::isnan(f.row) ? row : f.row);
        xn++;
      }
    }
  }
  if (n < L_NEED || !solve3(m)) return std::nullopt;
  const double k = m[0][3] / m[0][0];
  if (k < R_FLASH) return std::nullopt;
  return Leak{m[1][3] / m[1][1] / k,
              m[2][3] / m[2][2] / k,
              std::sqrt(jmax(0.0, m[1][5] / m[1][1])) / k,
              std::sqrt(jmax(0.0, m[2][6] / m[2][2])) / k,
              n,
              from + jround(ts / n),
              k,
              xn > 0 ? xs / xn : 0.0};
}

// ai: The receiver's new frames (track), in capture order: the code's row from the newest that has one, the hold
// ai: begun on the first, then each batch read.
void PhaseLock::tracked(const std::vector<Frame>& got) {
  if (got.empty()) return;
  for (auto it = got.rbegin(); it != got.rend(); ++it)
    if (!std::isnan(it->row)) {
      row = coerce(it->row, 0.0, 1.0);
      break;
    }
  if (line_ < 0) {
    here();
    return;
  }
  for (const Frame& f : got)
    if (f.ts > since) window.push_back(f);
  while (!window.empty() && window.front().ts < window.back().ts - LONG_NS) window.pop_front();
  decide();
}

// ai: What the hold's frames say, once a batch of them is in.
void PhaseLock::decide() {
  if (move || moving || window.empty()) return;
  held();
  snap();
}

// ai: The gain and the display's pace from the hold's own history. Between two readings that both showed a
// ai: neighbour, the captures went later by h (ms: what their own places against the line show, with the line's move)
// ai: and earlier by the slide (the display's pace p less the line's, over the time dt), and a - b fell by that over
// ai: the gain G: G (e2 - e1) - p dt = -h - pace dt, one look at (G, p) with the readings' variance through the gain.
// ai: The gain is told only by a pair whose move is Z of what its readings resolve (the gain times the error of e2 -
// ai: e1): a pair chosen by a - b's own change is chosen by its noise, and the gain told by those falls toward
// ai: GAIN_LEAST the longer a hold lasts; elsewhere the look tells the pace alone.
void PhaseLock::learn(int64_t t, double de, double dt, double h, double paceMs, double w) {
  const double z = -h - paceMs * dt;
  const double hp = -dt * 1e3;
  const double rRead = gain * gain / w;
  const bool both = h * h * w >= Z * Z * gain * gain;
  observe(t, z, de, hp, rRead, dt, both);
}

// ai: One look: z (ms) = hg gain + hp pace, its readings' variance rRead (ms²) over dt seconds to camera time t. The
// ai: view's wander over dt (and, over what of it was the session's first AIM_NS, an aiming hand's) is added; a look
// ai: beyond Z of its expected spread counts as Z (the gain and the pace move only as far as that); `both`: the gain is
// ai: told too, else the pace alone with the gain's error carried (its covariance kept whole either way).
void PhaseLock::observe(int64_t t, double z, double hg, double hp, double rRead, double dt, bool both) {
  age(t);
  // ai: the aim: the part of the look's span within the session's first AIM_NS (a hand still aiming)
  const double aim =
      aimFrom == 0 ? 0.0 : jmax(0.0, (double)(std::min(t, aimFrom + AIM_NS) - std::max(t - jround(dt * 1e9), aimFrom)) / 1e9);
  // ai: a look says the pace's mean over its span, which is off the pace now by the pace's own process noise over the
  // ai: span (a random walk's mean over a span T is off its end by a third of its variance over T)
  const double span = hp * hp * PACE_Q * dt / 3;
  const double r = rRead + span + VIEW_MS2 * dt + AIM_MS2 * aim * aim;
  evid.push_back({z, hg, hp, r, rRead + span, gain, gainVar, (double)t});
  while ((int)evid.size() > N_EVID) evid.pop_front();
  const double nu = z - hg * gain - hp * pace;
  if (debug)
    say(fmt("phase: DEBUG a look over %.1f s (%s): the pace %.0f us a second (to %.0f) at the gain %.1f, against %.0f (to %.0f)", dt,
            both ? "gain too" : "pace", (z - hg * gain) / hp * 1e6, std::sqrt((r + hg * hg * gainVar) / (hp * hp)) * 1e6, gain, pace * 1e6,
            sure() * 1e6));
  const double pg = gainVar * hg + gpCov * hp;
  const double pp = gpCov * hg + paceVar * hp;
  const double hph = hg * pg + hp * pp;
  const double s = jmax(hph + r, nu * nu / (Z * Z));
  if (both) {
    const double kg = pg / s, kp = pp / s;
    gain = coerce(gain + kg * nu, GAIN_LEAST, GAIN_MOST);
    pace = coerce(pace + kp * nu, -PACE_MOST, PACE_MOST);
    gainVar -= kg * pg;
    gpCov -= kg * pp;
    paceVar -= kp * pp;
  } else {
    // ai: the pace's row alone (Joseph's form: the gain kept as it is, its error in the pace's)
    const double kp = pp / s;
    pace = coerce(pace + kp * nu, -PACE_MOST, PACE_MOST);
    const double a = -kp * hg, b = 1 - kp * hp;
    const double gp = a * gainVar + b * gpCov;
    paceVar = a * a * gainVar + 2 * a * b * gpCov + b * b * paceVar + kp * kp * (s - hph);
    gpCov = gp;
  }
  gainVar = jmax(gainVar, 1e-9);
  paceVar = jmax(paceVar, 1e-18);
  check();
}

// ai: the process noise up to camera time t: the gain's (GAIN_Q) and the pace's (PACE_Q), each a second
void PhaseLock::age(int64_t t) {
  if (paceAt != 0 && t > paceAt) {
    const double s = (double)(t - paceAt) / 1e9;
    gainVar += GAIN_Q * s;
    paceVar = jmin(PACE_VAR0, paceVar + PACE_Q * s);
  }
  paceAt = std::max(paceAt, t);
}

// ai: The newest N_EVID looks (of the newest LONG_NS) re-check the pace: each says the pace at the gain as it is, a
// ai: vote each (a jump of the view's spoils a look whatever its own error says, so none outweighs the rest); where
// ai: their median is beyond Z of its error (a median's: pi / 2 of a look's variance over their count, a look's the
// ai: larger of what its readings and the gain say, their median, and what the looks' spread about the median says:
// ai: the view's wander as it is, not as a hand's is supposed) and the pace's own, the pace is the median and its
// ai: variance the median's.
void PhaseLock::check() {
  // ai: (the looks of the newest LONG_NS: an older one says what the pace was)
  while (!evid.empty() && evid.front()[7] < (double)(paceAt - LONG_NS)) evid.pop_front();
  if ((int)evid.size() < N_EVID) return;
  const size_t n = evid.size();
  std::vector<double> ps(n), vs(n);
  for (size_t i = 0; i < n; i++) {
    const auto& e = evid[i];
    ps[i] = (e[0] - e[1] * gain) / e[2];
    vs[i] = (e[4] + e[1] * e[1] * gainVar) / (e[2] * e[2]);
  }
  const double m = median(ps);
  std::vector<double> dev(n);
  for (size_t i = 0; i < n; i++) dev[i] = std::fabs(ps[i] - m);
  const double mad = median(dev);
  const double err2 = 3.141592653589793 / 2 * jmax(median(vs), (1.4826 * mad) * (1.4826 * mad)) / (double)n;
  if (std::fabs(m - pace) > Z * std::sqrt(err2 + paceVar)) {
    if (debug)
      say(fmt("phase: DEBUG the newest %d looks say the pace is %.0f us a second (to %.0f), not %.0f (to %.0f)", (int)n, m * 1e6,
              std::sqrt(err2) * 1e6, pace * 1e6, sure() * 1e6));
    pace = coerce(m, -PACE_MOST, PACE_MOST);
    paceVar = jmin(PACE_VAR0, err2);
    gpCov = 0.0;
    // ai: what those looks told the gain they told it at a pace that was wrong: the gain goes back to what it was
    // ai: before the oldest of them, its variance then and its process noise since
    const auto o = evid.front();
    gain = o[5];
    gainVar = o[6] + GAIN_Q * jmax(0.0, ((double)paceAt - o[7]) / 1e9);
    evid.clear();
  }
}

// ai: The pilots' hold: what the newest frames hold of their neighbours, and the move it asks.
// ai:   both neighbours in them (a top that bends: no phase is clean): later by the gain times a - b;
// ai:   one neighbour only (the edge of a top that is flat, where a capture holds neither): out of it and a step of
// ai:     CROSS in. Back at the same edge after standing clear of it, the two clocks' slide is the step over the time
// ai:     it took, and the pace takes it (in the flat no reading shows a slide). At the other edge straight from the
// ai:     step, the top is narrower than the step: its width is known, and the hold goes to its middle;
// ai:   neither: the hold stands; LONG_NS of it from the last step, and the width a step met may be the top's no
// ai:     longer (a window that shortened widens it): the next step is CROSS again and measures it afresh.
// ai: Every reading that shows a neighbour, with the one before, also tells the gain and the pace (`learn`), and the
// ai: move allows for what the hold has slid since its frames were captured.
void PhaseLock::held() {
  std::vector<Frame> fs;
  for (const Frame& f : window)
    if (f.ts > settle) fs.push_back(f);
  if ((int)fs.size() < L_NEED) return;
  int sum = 0;
  for (const Frame& f : fs) sum += f.blocks;
  level = jmax(level * 0.98, (double)sum / (double)fs.size());
  // ai: the newest second's frames where they are enough (what the hold holds now, not a mean over a long stand),
  // ai: else all since the last move
  std::optional<Leak> est = leaks(std::max(settle, fs.back().ts - RECENT_NS));
  if (!est) est = leaks(settle);
  if (est) lastLeak = est;
  if (!est) {
    // ai: no frame tells its neighbours apart. Deep in a mix (half the frames' r under DEEP: no capture is mostly one
    // ai: picture) nothing has a side: a quarter of a refresh later, and look again. Only where half the frames have
    // ai: no sure sign: frames that have one and are just too few yet (a small batch, its ends without a neighbour)
    // ai: will tell the side with the next batch, and a blind step is not it
    std::vector<double> rs;
    rs.reserve(fs.size());
    for (const Frame& f : fs) rs.push_back(std::fabs(f.r));
    std::sort(rs.begin(), rs.end(), jless);
    const double mid = rs[fs.size() / 2];
    size_t sig = 0;
    for (const Frame& f : fs)
      if (std::fabs(f.r) >= R_FLASH && std::fabs(f.r2) >= R_FLASH) sig++;
    if (mid < DEEP && 2 * sig < fs.size()) {
      line_ += jround(refresh() / 4);
      move = true;
      prevT = 0;
      edgeDir = 0;
      flatHalf = 0;
      stood = false;
      if (debug) say(fmt("phase: DEBUG deep in a mix (|r| %.2f): a quarter later", mid));
    }
    return;
  }
  const double e = est->a - est->b;
  const double se = jhypot(est->sa, est->sb);
  // ai: a neighbour is in the frames where its share is Z of its errors and LEAK at least (under that it costs no
  // ai: block, and at a top that is nearly flat both shares hover about nothing)
  const bool sigA = est->a > jmax(LEAK, Z * est->sa);
  const bool sigB = est->b > jmax(LEAK, Z * est->sb);
  // ai: readings of frames that overlap say one thing twice: a reading counts once its frames are a RECENT on from
  // ai: the last one's (or a move has been made since, which starts the frames afresh)
  if (prevT != 0 && prevHop == 0.0 && est->t - prevT < RECENT_NS) return;
  const double was = pace;
  // ai: a pair tells the gain and the pace where a - b went straight with the hold's place between them: both
  // ai: readings showing the same neighbours (across a flat top it stands still whatever the hold does)
  const int side = (sigA ? 1 : 0) | (sigB ? 2 : 0);
  if (prevT != 0 && side != 0 && side == prevSide && est->t > prevT)
    learn(est->t, e - prevE, (double)(est->t - prevT) / 1e9, prevHop + (est->x - prevX) / 1e6, prevPace * 1e3, 1 / (se * se + prevSe * prevSe));
  if (debug) {
    std::string top;
    if (flatHalf > 0)
      top = fmt(", a flat top %.1f ms wide", (double)(2 * flatHalf) / 1e6);
    else if (edgeDir != 0)
      top = fmt(", %.1f ms in from the %s edge", edgeIn / 1e6, edgeDir > 0 ? "early" : "late");
    say(fmt("phase: DEBUG leaks %d frames: before %.3f (to %.3f), after %.3f (to %.3f), gain %.1f ms, pace %.0f%s", est->n, est->a, est->sa, est->b,
            est->sb, gain, pace * 1e6, top.c_str()));
  }
  int64_t hop = 0;
  if (sigA && sigB) {
    edgeDir = 0;
    flatHalf = 0;
    if (std::fabs(e) >= Z * se) {
      hop = jround(gain * e * 1e6);
    } else if (!balanced && paceVar < PACE_VAR0) {
      balanced = true;
      const std::string to = fmt("%.0f", sure() * 1e6);
      say(fmt("phase: the hold stands: %.0f%% of the picture before and %.0f%% of the one after in a capture; the pace %.0f us a second (to %s), the "
              "line %.1f ms a unit",
              100 * est->a, 100 * est->b, pace * 1e6, to.c_str(), gain));
    }
  } else if (sigA || sigB) {
    const int dir = sigA ? 1 : -1;
    int64_t edge = jround(gain * (sigA ? est->a : est->b) * 1e6);
    if (edgeDir == dir && edgeClean && est->t - edgeT >= PACED_NS) {
      // ai: back at the edge it stepped in from, having stood inside the top meanwhile: the slide is the step (and
      // ai: what it is past the edge now) over the time since the step was reckoned from as a look at (G, p):
      // ai: p dt - dir share G = dir edgeIn + pace dt (ms), the two edges' readings' errors
      const double dt = (double)(est->t - edgeT);
      const double dtS = dt / 1e9;
      const double sd = 2 * gain * jmax(est->sa, est->sb);
      observe(est->t, (double)(dir * edgeIn) / 1e6 + pace * dtS * 1e3, (double)(-dir) * (sigA ? est->a : est->b), dtS * 1e3, sd * sd, dtS, false);
      say(fmt("phase: the hold slid %.1f ms in %.1f s: the pace %.0f us a second (to %.0f)", (double)(edgeIn + edge) / 1e6, dt / 1e9, pace * 1e6,
              sure() * 1e6));
    } else if (edgeDir == -dir && est->t > edgeT && !edgeClean) {
      // ai: the other edge, straight from a step in from the first: the top is narrower than the step, as wide as what
      // ai: was stepped, and the hold goes to its middle. What the captures moved (their own places) is the top's width
      // ai: and both edges' depths, each the gain times its share; a gain that puts the depths alone past the move is
      // ai: too large (a width is never less than 0), and the move over the change of a - b is the most it can be
      // ai: (else every step from one edge overshoots to the other, and readings of one neighbour each never pair to
      // ai: tell the gain)
      const double moved = std::fabs(prevHop * 1e6 + (est->x - prevX));
      const double de = std::fabs(e - prevE);
      if (prevT != 0 && prevSide == 3 - side && gain * de * 1e6 > moved) {
        gain = coerce(moved / de / 1e6, GAIN_LEAST, GAIN_MOST);
        edge = jround(gain * (sigA ? est->a : est->b) * 1e6);
      }
      flatHalf = std::max(BAND_NS, (edgeIn - edge) / 2);
      say(fmt("phase: the top is %.1f ms wide", (double)(2 * flatHalf) / 1e6));
    } else if (edgeDir == -dir) {
      flatHalf = 0;
    }
    const int64_t step = flatHalf > 0 ? flatHalf : CROSS_NS;
    hop = dir * (edge + step);
    // ai: the step is reckoned from now where the pace was just put right (the move below allows for the slide since
    // ai: the frames); where it was not, the hold slides on as it did, and the step is reckoned from the frames' own time
    edgeDir = dir;
    edgeIn = step;
    edgeT = pace != was ? now : est->t;
    edgeClean = false;
  } else {
    if (edgeDir != 0 && est->t > edgeT) edgeClean = true;   // ai: (its frames are after the step landed: settle)
    if (flatHalf > 0 && est->t - edgeT >= LONG_NS) flatHalf = 0;
  }
  // ai: the frames are old by the time they are read, and the hold slid on meanwhile at what the pace was off by: the
  // ai: move allows for it (the pace is right from now on)
  hop = coerce(hop, -jround(refresh() / 2), jround(refresh() / 2));
  if (hop == 0) stood = true;
  // ai: (the next pair's move is this one without that allowance: its line reckons the slide at the new pace from the
  // ai: frames' own time)
  if (side != 0) {
    prevE = e;
    prevSe = se;
    prevT = est->t;
    prevHop = (double)hop / 1e6;
    prevPace = pace;
    prevSide = side;
    prevX = est->x;
  } else {
    prevT = 0;
  }
  if (hop != 0 || pace != was) hop += jround((pace - was) * (double)(now - est->t));
  if (hop != 0) {
    line_ += hop;
    move = true;
  }
}

double PhaseLock::landMs() const {
  if (lands.empty()) return NAN;
  int64_t m = lands.front()[1];
  for (const auto& l : lands) m = std::max(m, l[1]);
  return (double)m / 1e6;
}

void PhaseLock::snap() {
  State s;
  s.arm = "pilots";
  s.what = standing();
  if (lastLeak) {
    s.a = lastLeak->a;
    s.b = lastLeak->b;
    s.k = lastLeak->k;
    s.se = jhypot(lastLeak->sa, lastLeak->sb);
    s.n = lastLeak->n;
  }
  s.gain = gain;
  s.pace = pace * 1e6;
  s.stood = stood;
  s.edge = edgeDir;
  s.flatMs = (double)(2 * flatHalf) / 1e6;
  s.delays = delaysAll;
  s.refreshMs = base == 0 ? NAN : refresh() / 1e6;
  s.multiple = multiple;
  s.frameMs = (double)frame / 1e6;
  s.landMs = landMs();
  s.row = row;
  std::lock_guard<std::mutex> l(stateMu);
  state_ = std::move(s);
}

std::optional<PhaseLock::State> PhaseLock::state() const {
  std::lock_guard<std::mutex> l(stateMu);
  return state_;
}

std::optional<std::string> PhaseLock::line(int64_t tsNs) {
  (void)tsNs;
  const int d = delays;
  delays = 0;
  if (mode_ == Mode::Off) return std::nullopt;
  const bool holds = line_ >= 0;
  const std::string to = paceVar >= PACE_VAR0 ? std::string("?") : fmt("%.0f", sure() * 1e6);
  const std::string leak = lastLeak && holds
                               ? fmt(", %.0f%% of the picture before and %.0f%% of the one after in a capture", 100 * lastLeak->a, 100 * lastLeak->b)
                               : std::string();
  const std::string ref = base == 0 ? std::string("?") : fmt("%.4f", refresh() / 1e6);
  std::string land("?");
  if (!lands.empty()) {
    int64_t most = lands.front()[2];
    for (const auto& l : lands) most = std::max(most, l[2]);
    land = fmt("%.0f ms, %lld captures after the mark", landMs(), (long long)most);
  }
  return fmt("phase: track%s, %s, the pace %.0f us a second (to %s), %.1f blocks a frame, %d delays%s; the refresh %s ms, %d a frame of %.3f ms, "
             "delays landing in %s, the code at row %.2f",
             holds ? " by the pilots" : "", standing(), pace * 1e6, to.c_str(), level, d, leak.c_str(), ref.c_str(), multiple, (double)frame / 1e6,
             land.c_str(), row);
}

}  // namespace lizard
