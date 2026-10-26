// ai: The simulated screen and camera (sim/scene.h's model, 2026-10-25). The painters take each picture's blocks in
// ai: order under the ring's lock (liz_tx_next: the test stream or a file's schedule, one sender, as a real one paints
// ai: them) and paint them in parallel (an encoder each, liz_encoder_paint with the picture's number: the pilots'
// ai: count) into a ring of about a second, never more than `ahead` pictures past the one on screen. A render works
// ai: out, for each sensor row and each band of kBand screen rows, which pictures its exposure saw and by how much
// ai: (the step response's integral over the window, exactly, settling included), then for each pixel the screen
// ai: point its ray meets (the frame-to-screen homography), that point's light in each of those pictures (bilinear in
// ai: the picture's pixels, in linear light) weighted, the blur, the sensor's black, gain, noise and tone.
#include "scene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <locale>
#include <sstream>

#include "lizard.h"

namespace lizard::sim {
namespace {

// ai: steadyNs's clock (camera/clock.h: std::chrono::steady_clock), read here so the model needs no camera code
int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::chrono::steady_clock::time_point timeOf(int64_t ns) {
  return std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::nanoseconds(ns)));
}

uint64_t mix(uint64_t x) {   // ai: splitmix64's finaliser
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
double unit(uint64_t x) { return double(mix(x) >> 11) * (1.0 / 9007199254740992.0); }   // ai: [0, 1)

// ai: a JSON number whatever the process's locale (the JVM sets LC_NUMERIC to the user's: presenter.cpp's fmt)
std::string num(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  for (char* q = b; *q; q++) if (*q == ',') *q = '.';
  return b;
}

// ai: A fork-join pool: run(jobs, f) calls f(0) .. f(jobs - 1) over its threads and the caller's, back when all are
// ai: done. One caller at a time (render's lock).
class Pool {
 public:
  explicit Pool(int threads) {
    for (int i = 1; i < threads; i++) th_.emplace_back([this] { work(); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> l(mu_); quit_ = true; }
    go_.notify_all();
    for (auto& t : th_) t.join();
  }
  int threads() const { return int(th_.size()) + 1; }
  void run(int jobs, const std::function<void(int)>& f) {
    {
      std::lock_guard<std::mutex> l(mu_);
      f_ = &f; jobs_ = jobs; next_ = 0; busy_ = int(th_.size()); gen_++;
    }
    go_.notify_all();
    for (int j; (j = next_.fetch_add(1)) < jobs;) f(j);
    std::unique_lock<std::mutex> l(mu_);
    done_.wait(l, [&] { return busy_ == 0; });
  }

 private:
  void work() {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> l(mu_);
    for (;;) {
      go_.wait(l, [&] { return quit_ || gen_ != seen; });
      if (quit_) return;
      seen = gen_;
      const std::function<void(int)>* f = f_;
      const int jobs = jobs_;
      l.unlock();
      for (int j; (j = next_.fetch_add(1)) < jobs;) (*f)(j);
      l.lock();
      if (--busy_ == 0) done_.notify_all();
    }
  }
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable go_, done_;
  const std::function<void(int)>* f_ = nullptr;
  std::atomic<int> next_{0};
  int jobs_ = 0, busy_ = 0;
  uint64_t gen_ = 0;
  bool quit_ = false;
};

// ai: The pictures in one pixel's light: at most kMix (a window holding more folds the oldest into the next: windows
// ai: past about 100 ms at 60 a second, which nothing here asks); pic indexes the render's pictures.
constexpr int kMix = 8;
struct Mix {
  int n = 0;
  int pic[kMix];
  float w[kMix];
};
// ai: A change older than kTail time constants is taken as complete (e^-8: 3.4e-4 of the step, 0.09 of a grey level
// ai: at full scale). kBand screen rows share a change time: 2 rows of a 1080-row scan of 16 ms is 30 us, against
// ai: exposures of milliseconds.
constexpr double kTail = 8.0;
constexpr int kBand = 2;
constexpr int kRowsAJob = 8;
constexpr double kRoom = 0.02;   // ai: the room's light off the screen, of a white screen's

}  // namespace

struct Scene::Impl {
  // ai: what is painted
  liz_geometry g{};
  liz_tx* tx = nullptr;
  std::vector<liz_encoder*> enc;
  // ai: the display's clock: ns past t0, as doubles
  int64_t t0 = 0;
  double phase = 0, tref = 0, fps = 60, hz = 60, scanNs = 0, tau = 0;
  double repeat = 0;
  uint64_t repSeed = 0;
  mutable std::mutex repMu;
  mutable std::vector<uint32_t> repE;   // ai: repE[k]: the pictures before k shown a refresh long
  // ai: the ring (mu): slot k % slots holds picture k once ready; pinned pictures are not painted over
  int slots = 0, ahead = 0;
  std::vector<std::vector<uint8_t>> ring;
  std::vector<int64_t> pic;
  std::vector<char> ready;
  std::vector<uint8_t> blank;
  int64_t next = 0, pinLo = 0, pinHi = -1;
  std::mutex mu;
  std::condition_variable cv;
  bool quit = false, started = false;   // ai: started: t0 set, the display's clock running
  std::vector<std::thread> painters;
  uint64_t painted = 0, late = 0, stale = 0, renders = 0;
  double paintMs = 0, renderMs = 0;
  // ai: the camera
  uint32_t W = 0, H = 0, sw = 0, sh = 0;
  double hm[9] = {};          // ai: frame point (pixel centres at + 0.5) to screen point (pixels from its top left)
  float lut[256] = {};        // ai: a grey level's light
  std::vector<uint8_t> tone;  // ai: the sensor's value in 65536 steps to 8 bits
  float sur = 1;
  double ox = 0, oy = 0, inv = 1;   // ai: the code's top left on the screen, picture pixels a screen pixel
  std::vector<float> light, tmp, gk;
  int gr = 0;
  std::mutex renderMu;
  std::unique_ptr<Pool> pool;

  ~Impl() {
    { std::lock_guard<std::mutex> l(mu); quit = true; }
    cv.notify_all();
    for (auto& t : painters) t.join();
    for (auto* e : enc) liz_encoder_free(e);
    liz_tx_free(tx);
  }

  // ai: the display: the picture refresh j shows without repeats, floor(j fps / hz) (an epsilon so a whole ratio's
  // ai: boundary is not lost to rounding: 60 on 59.94 is 1000 / 999, its fractions at least 1e-3)
  int64_t base(int64_t j) const { return j <= 0 ? 0 : int64_t(std::floor(double(j) * fps / hz + 1e-7)); }
  int64_t due(int64_t k) const {   // ai: the first refresh base() shows k or later on
    if (k <= 0) return 0;
    int64_t d = int64_t(std::ceil(double(k) * hz / fps));
    while (d > 0 && base(d - 1) >= k) d--;
    while (base(d) < k) d++;
    return d;
  }
  uint32_t repeats(int64_t k) const {
    if (repeat <= 0 || k <= 0) return 0;
    std::lock_guard<std::mutex> l(repMu);
    if (repE.empty()) repE.push_back(0);
    while (int64_t(repE.size()) <= k) {
      const uint64_t i = repE.size() - 1;
      repE.push_back(repE.back() + (unit(repSeed ^ (i * 0xd1b54a32d192ed03ull)) < repeat ? 1u : 0u));
    }
    return repE[size_t(k)];
  }
  int64_t first(int64_t k) const { return due(k) + repeats(k); }   // ai: picture k's first refresh
  int64_t shown(int64_t j) const {                                  // ai: the picture of refresh j
    if (j <= 0) return 0;
    if (repeat <= 0) return base(j);
    int64_t lo = 0, hi = base(j);
    while (lo < hi) {
      const int64_t m = lo + (hi - lo + 1) / 2;
      if (first(m) <= j) lo = m; else hi = m - 1;
    }
    return lo;
  }
  int64_t refreshAt(double rel) const { return int64_t(std::floor((rel - phase) / tref)); }   // ai: begun at or before rel (ns past t0)
  // ai: the first whole ns of refresh j (past t0): its start rounded up, a step on where rounding left it in j - 1
  int64_t refreshNs(int64_t j) const {
    int64_t t = int64_t(std::ceil(phase + double(j) * tref));
    while (refreshAt(double(t)) < j) t++;
    return t;
  }

  void paintLoop(int which) {
    std::vector<uint8_t> blocks(size_t(g.frame_blocks) * LIZ_BLOCK);
    std::unique_lock<std::mutex> l(mu);
    while (!quit) {
      const int64_t k = next;
      if (k >= ahead) {   // ai: at most `ahead` past the picture on screen
        if (!started) { cv.wait(l); continue; }
        const int64_t when = t0 + refreshNs(first(k - ahead + 1));
        if (nowNs() < when) { cv.wait_until(l, timeOf(when)); continue; }
      }
      const int s = int(k % slots);
      if (pic[s] >= 0 && (!ready[s] || (pic[s] >= pinLo && pic[s] <= pinHi))) { cv.wait(l); continue; }
      next = k + 1;
      pic[s] = k;
      ready[s] = 0;
      if (liz_tx_next(tx, g.frame_blocks, blocks.data()) < 0) std::memset(blocks.data(), 0, blocks.size());
      l.unlock();
      const int64_t a = nowNs();
      if (liz_encoder_paint(enc[size_t(which)], blocks.data(), uint32_t(k), ring[size_t(s)].data(), g.width, LIZ_GREY8) != LIZ_OK)
        std::memset(ring[size_t(s)].data(), 255, ring[size_t(s)].size());
      const double ms = double(nowNs() - a) * 1e-6;
      l.lock();
      ready[s] = 1;
      painted++;
      paintMs += ms;
      cv.notify_all();
    }
  }

  // ai: (mu) the ready picture nearest k, preferring the newest at or before it where `before`; blank if none is
  const uint8_t* nearest(int64_t k, bool before) const {
    int best = -1;
    int64_t bestD = INT64_MAX;
    for (int s = 0; s < slots; s++) {
      if (pic[s] < 0 || !ready[s]) continue;
      const int64_t d = pic[s] <= k ? k - pic[s] : (before ? INT64_MAX / 2 : pic[s] - k);
      if (d < bestD) { bestD = d; best = s; }
    }
    return best < 0 ? blank.data() : ring[size_t(best)].data();
  }

  // ai: (mu) pictures kLo .. kHi pinned and found: a picture not yet painted waited for until a refresh from now, then
  // ai: the newest painted; one painted over, the nearest held
  void acquire(int64_t kLo, int64_t kHi, std::vector<const uint8_t*>& out, bool& wasLate, bool& wasStale,
               std::unique_lock<std::mutex>& l) {
    pinLo = kLo;
    pinHi = kHi;
    const int64_t deadline = nowNs() + int64_t(tref);
    out.assign(size_t(kHi - kLo + 1), nullptr);
    for (int64_t k = kLo; k <= kHi; k++) {
      const int s = int(k % slots);
      for (;;) {
        if (pic[s] == k && ready[s]) { out[size_t(k - kLo)] = ring[size_t(s)].data(); break; }
        if (pic[s] > k) { wasStale = true; out[size_t(k - kLo)] = nearest(k, false); break; }
        if (wasLate || nowNs() >= deadline) { wasLate = true; out[size_t(k - kLo)] = nearest(k, true); break; }
        cv.wait_until(l, timeOf(deadline));
      }
    }
  }

  // ai: the integral over [a, b] of a unit step at c settling with tau: (b - lo) - tau (e^-(lo-c)/tau - e^-(b-c)/tau)
  double settled(double c, double a, double b) const {
    if (c >= b) return 0;
    const double lo = std::max(a, c);
    if (tau <= 0) return b - lo;
    return (b - lo) - tau * (std::exp(-(lo - c) / tau) - std::exp(-(b - c) / tau));
  }

  // ai: which pictures a screen row's light holds over the window [a, b] (ns past t0), and their shares: the state
  // ai: kTail time constants before the window taken as settled, each change after it a step whose share of the window
  // ai: S is the step response's mean over it, so the picture a change leaves keeps S_before - S
  void mixOf(double yoff, double a, double b, int64_t jLo, const std::vector<int64_t>& byRefresh, int64_t kLo, int64_t kHi,
             Mix& m) const {
    const auto at = [&](int64_t j) { return j >= jLo && j - jLo < int64_t(byRefresh.size()) ? byRefresh[size_t(j - jLo)] : shown(j); };
    const auto add = [&](int64_t k, double w) {
      if (w < 1e-6) return;
      const int i = int(std::min(std::max(k, kLo), kHi) - kLo);
      if (m.n && m.pic[m.n - 1] == i) { m.w[m.n - 1] += float(w); return; }
      if (m.n == kMix) {   // ai: fold the oldest into the next
        m.w[1] += m.w[0];
        std::memmove(m.pic, m.pic + 1, sizeof(int) * (kMix - 1));
        std::memmove(m.w, m.w + 1, sizeof(float) * (kMix - 1));
        m.n--;
      }
      m.pic[m.n] = i;
      m.w[m.n] = float(w);
      m.n++;
    };
    m.n = 0;
    const double span = b - a;
    const int64_t jA = refreshAt(a - kTail * tau - yoff);
    int64_t cur = at(jA);
    double before = 1;
    for (int64_t j = jA + 1;; j++) {
      const double c = phase + double(j) * tref + yoff;
      if (c >= b) break;
      const int64_t kj = at(j);
      if (kj == cur) continue;
      const double S = settled(c, a, b) / span;
      add(cur, before - S);
      cur = kj;
      before = S;
    }
    add(cur, before);
    if (!m.n) { m.n = 1; m.pic[0] = int(std::min(std::max(cur, kLo), kHi) - kLo); m.w[0] = 1; }
  }

  // ai: sensor row r's light into light[r]: its window [a, a + span]; mixes the caller's (a job's) buffer for the
  // ai: bands' mixes, grown as needed
  void sampleRow(uint32_t r, double a, double span, int64_t jLo, const std::vector<int64_t>& byRefresh, int64_t kLo,
                 int64_t kHi, const std::vector<const uint8_t*>& pics, std::vector<Mix>& mixes) {
    const double v = r + 0.5;
    const double ax = hm[1] * v + hm[2], ay = hm[4] * v + hm[5], aw = hm[7] * v + hm[8];
    // ai: the bands the row's screen line crosses: a homography keeps a line's points in order between its ends, unless
    // ai: the line meets the horizon (w changes sign), then every band
    const int nb = int((sh + kBand - 1) / kBand);
    int bLo = 0, bHi = nb - 1;
    {
      const double u0 = 0.5, u1 = W - 0.5, w0 = hm[6] * u0 + aw, w1 = hm[6] * u1 + aw;
      if (w0 > 0 && w1 > 0) {
        const double y0 = (hm[3] * u0 + ay) / w0, y1 = (hm[3] * u1 + ay) / w1;
        const double lo = std::min(y0, y1), hi = std::max(y0, y1);
        bLo = int(std::floor(std::max(0.0, std::min(double(sh - 1), lo)) / kBand));
        bHi = int(std::floor(std::max(0.0, std::min(double(sh - 1), hi)) / kBand));
      }
    }
    mixes.resize(size_t(bHi - bLo + 1));
    for (int b = bLo; b <= bHi; b++) {
      const double y = b * kBand + (kBand - 1) * 0.5;
      mixOf(y / sh * scanNs, a, a + span, jLo, byRefresh, kLo, kHi, mixes[size_t(b - bLo)]);
    }
    const float h0 = float(hm[0]), h3 = float(hm[3]), h6 = float(hm[6]);
    const float fax = float(ax), fay = float(ay), faw = float(aw);
    const float fsw = float(sw), fsh = float(sh), fox = float(ox), foy = float(oy), finv = float(inv);
    const int gw = g.width, gh = g.height;
    const float gwf = float(gw), ghf = float(gh), room = float(kRoom), bandInv = 1.f / kBand;
    const int last = bHi - bLo;
    float* o = light.data() + size_t(r) * W;
    for (uint32_t c = 0; c < W; c++) {
      const float u = float(c) + 0.5f, w = h6 * u + faw;
      float L = room;
      if (w > 0) {
        const float iw = 1.f / w, sx = (h0 * u + fax) * iw, sy = (h3 * u + fay) * iw;
        if (sx >= 0 && sy >= 0 && sx < fsw && sy < fsh) {
          const float qx = (sx - fox) * finv, qy = (sy - foy) * finv;
          if (qx < 0 || qy < 0 || qx >= gwf || qy >= ghf) {
            L = sur;
          } else {
            float fx = qx - 0.5f, fy = qy - 0.5f;   // ai: at least -0.5, so int(f + 1) - 1 is the floor
            int ix = int(fx + 1.f) - 1, iy = int(fy + 1.f) - 1;
            fx -= float(ix);
            fy -= float(iy);
            if (ix < 0) { ix = 0; fx = 0; } else if (ix > gw - 2) { ix = gw - 2; fx = 1; }
            if (iy < 0) { iy = 0; fy = 0; } else if (iy > gh - 2) { iy = gh - 2; fy = 1; }
            const int bi = std::min(std::max(int(sy * bandInv) - bLo, 0), last);
            const Mix& m = mixes[size_t(bi)];
            const size_t off = size_t(iy) * size_t(gw) + size_t(ix);
            L = 0;
            for (int i = 0; i < m.n; i++) {
              const uint8_t* q = pics[size_t(m.pic[i])] + off;
              const float p00 = lut[q[0]], p01 = lut[q[1]], p10 = lut[q[gw]], p11 = lut[q[gw + 1]];
              const float top = p00 + fx * (p01 - p00), bot = p10 + fx * (p11 - p10);
              L += m.w[i] * (top + fy * (bot - top));
            }
          }
        }
      }
      o[c] = L;
    }
  }
};

// ai: ---- the spec ----

namespace {

bool number(const std::string& text, double& v) {
  std::istringstream is(text);
  is.imbue(std::locale::classic());
  is >> v;
  return !is.fail() && is.peek() == std::char_traits<char>::eof() && std::isfinite(v);
}

}  // namespace

bool parse(const std::string& spec, Sender& s, Display& d, Optics& o, std::string& why) {
  std::istringstream in(spec);
  std::string tok;
  while (in >> tok) {
    const size_t eq = tok.find('=');
    if (eq == std::string::npos || !eq) { why = "\"" + tok + "\" is not key=value"; return false; }
    const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
    double x = 0;
    // ai: a number within [lo, hi] (whole where `whole`), else why
    const auto take = [&](double lo, double hi, bool whole) {
      if (!number(val, x) || x < lo || x > hi || (whole && x != std::floor(x))) {
        why = key + "=" + val + ": " + (whole ? "a whole number" : "a number") + " from " + num("%g", lo) + " to " + num("%g", hi);
        return false;
      }
      return true;
    };
    if (key == "blocks") { if (!take(LIZ_MIN_BLOCKS, LIZ_MAX_BLOCKS, true)) return false; s.blocks = int(x); }
    else if (key == "ring") {
      // ai: by its cells a side, as the sender hook's ring= names it (an index would mean
      // ai: another ring once the list changes)
      int r = -1;
      std::string cells;
      for (int i = 0; i < LIZ_RINGS; i++) {
        cells += (i ? (i + 1 < LIZ_RINGS ? ", " : " or ") : "") + std::to_string(liz_ring_cells(i));
        if (number(val, x) && x == liz_ring_cells(i)) r = i;
      }
      if (r < 0 && val != "auto") { why = "ring=" + val + ": " + cells + " cells a side, or auto"; return false; }
      s.ring = r;
    }
    else if (key == "fps") { if (!take(1, 255, true)) return false; s.fps = int(x); }
    else if (key == "codes") { if (!take(1, 2, true)) return false; s.codes = int(x); }
    else if (key == "file") { if (val.empty()) { why = "file=: a path"; return false; } s.file = val; }
    else if (key == "type") { s.type = val; }
    else if (key == "id") { if (!take(0, 4294967295.0, true)) return false; s.firstId = uint32_t(x); }
    else if (key == "hz") { if (!take(1, 1000, false)) return false; d.hz = x; }
    else if (key == "ppm") { if (!take(-100000, 100000, false)) return false; d.ppm = x; }
    else if (key == "phase") { if (!take(-1, 10000, false)) return false; d.phaseMs = x; }
    else if (key == "scan") { if (!take(0, 1, false)) return false; d.scan = x; }
    else if (key == "response") { if (!take(0, 50, false)) return false; d.responseMs = x; }
    else if (key == "sw") { if (!take(16, 16384, true)) return false; d.width = uint32_t(x); }
    else if (key == "sh") { if (!take(16, 16384, true)) return false; d.height = uint32_t(x); }
    else if (key == "share") { if (!take(0.01, 1, false)) return false; d.share = x; }
    else if (key == "surround") { if (!take(0, 255, true)) return false; d.surround = uint8_t(x); }
    else if (key == "repeat") { if (!take(0, 1, false)) return false; d.repeat = x; }
    else if (key == "w") { if (!take(16, 16384, true)) return false; o.width = uint32_t(x); }
    else if (key == "h") { if (!take(16, 16384, true)) return false; o.height = uint32_t(x); }
    else if (key == "fill") { if (!take(0.05, 10, false)) return false; o.fill = x; }
    else if (key == "yaw") { if (!take(-80, 80, false)) return false; o.yawDeg = x; }
    else if (key == "tilt") { if (!take(-80, 80, false)) return false; o.tiltDeg = x; }
    else if (key == "roll") { if (!take(-180, 180, false)) return false; o.rollDeg = x; }
    else if (key == "blur") { if (!take(0, 20, false)) return false; o.blurPx = x; }
    else if (key == "exposure") { if (!take(0.01, 200, false)) return false; o.exposureMs = x; }
    else if (key == "shutter") {
      if (val == "rolling") o.rolling = true;
      else if (val == "global") o.rolling = false;
      else { why = "shutter=" + val + ": rolling or global"; return false; }
    }
    else if (key == "readout") { if (!take(0, 100, false)) return false; o.readoutMs = x; }
    else if (key == "gain") { if (!take(0.001, 1000, false)) return false; o.gain = x; }
    else if (key == "black") { if (!take(0, 0.9, false)) return false; o.black = x; }
    else if (key == "noise") { if (!take(0, 1, false)) return false; o.readNoise = x; }
    else if (key == "shot") { if (!take(0, 1, false)) return false; o.shotNoise = x; }
    else if (key == "gamma") { if (!take(0.1, 5, false)) return false; o.gamma = x; }
    else if (key == "seed") { if (!take(0, 4294967295.0, true)) return false; o.seed = uint32_t(x); }
    else { why = "\"" + key + "\" is not a key of the simulated screen"; return false; }
  }
  return true;
}

// ai: ---- the scene ----

std::unique_ptr<Scene> Scene::create(const Sender& s, const Display& d, const Optics& o, int painters, std::string& why) {
  if (d.hz <= 0 || d.width < 16 || d.height < 16 || d.share <= 0 || d.share > 1 || d.scan < 0 || d.scan > 1 ||
      d.responseMs < 0 || d.repeat < 0 || d.repeat > 1) {
    why = "the display's settings are out of range";
    return nullptr;
  }
  if (o.width < 16 || o.height < 16 || o.fill <= 0 || o.exposureMs <= 0 || o.readoutMs < 0 || o.gamma <= 0 ||
      o.blurPx < 0 || o.gain <= 0 || std::abs(o.yawDeg) >= 89 || std::abs(o.tiltDeg) >= 89) {
    why = "the camera's settings are out of range";
    return nullptr;
  }
  const liz_format f{s.blocks, s.ring, s.fps, s.codes};
  liz_geometry g{};
  if (liz_geometry_of(&f, &g) != LIZ_OK) { why = std::string("the format: ") + liz_last_error(); return nullptr; }
  std::unique_ptr<Scene> sc(new Scene(s, d, o));
  sc->p_ = std::make_unique<Impl>();
  Impl& p = *sc->p_;
  p.g = g;
  painters = std::max(1, std::min(painters, 16));
  for (int i = 0; i < painters; i++) {
    liz_encoder* e = liz_encoder_new(&f);
    if (!e) { why = std::string("the encoder: ") + liz_last_error(); return nullptr; }
    p.enc.push_back(e);
  }
  p.tx = s.file.empty() ? liz_tx_new_test(s.firstId) : liz_tx_new_path(s.file.c_str(), nullptr, s.type.empty() ? nullptr : s.type.c_str());
  if (!p.tx) { why = (s.file.empty() ? std::string("the test stream: ") : s.file + ": ") + liz_last_error(); return nullptr; }

  // ai: the display's clock
  p.fps = s.fps;
  p.hz = d.hz;
  p.tref = 1e9 / (d.hz * (1 + d.ppm * 1e-6));
  p.phase = d.phaseMs >= 0 ? d.phaseMs * 1e6 : unit(mix(o.seed) ^ 0x7068617365ull) * p.tref;
  p.scanNs = d.scan * p.tref;
  p.tau = d.responseMs * 1e6;
  p.repeat = d.repeat;
  p.repSeed = mix(uint64_t(o.seed) ^ 0x726570656174ull);
  // ai: the ring: `ahead` (three quarters of a second) past the picture on screen, a quarter second behind it, and
  // ai: the most a capture's window can span (its exposure up to 50 ms, its readout, its settling) besides
  p.ahead = std::max(4, int(std::ceil(0.75 * s.fps)));
  const double windowS = (std::max(50.0, o.exposureMs) + o.readoutMs) * 1e-3 + kTail * p.tau * 1e-9;
  p.slots = p.ahead + std::max(4, int(std::ceil(0.25 * s.fps))) + int(std::ceil(windowS * s.fps)) + 4;
  const double ringGb = double(p.slots) * g.width * g.height / 1e9;
  if (ringGb > 2) { why = "the ring of pictures would take " + num("%.1f", ringGb) + " GB (a smaller format or rate)"; return nullptr; }
  p.ring.assign(size_t(p.slots), std::vector<uint8_t>(size_t(g.width) * size_t(g.height)));
  p.pic.assign(size_t(p.slots), -1);
  p.ready.assign(size_t(p.slots), 0);
  p.blank.assign(size_t(g.width) * size_t(g.height), 255);

  // ai: the camera: two screen heights from the screen's centre, the focal length so the screen's height fills
  // ai: `fill` of the frame's head on; R = Ry(yaw) Rx(tilt) Rz(roll) the camera's axes in the screen's (x right,
  // ai: y down, z away from the viewer); a screen point P (from its centre) is at R^T P + (0, 0, D) from the camera
  p.W = o.width; p.H = o.height; p.sw = d.width; p.sh = d.height;
  const double D = 2.0 * d.height, fl = o.fill * o.height * D / d.height, deg = 3.14159265358979323846 / 180;
  const double cy = std::cos(o.yawDeg * deg), sy = std::sin(o.yawDeg * deg), ct = std::cos(o.tiltDeg * deg),
               st = std::sin(o.tiltDeg * deg), cr = std::cos(o.rollDeg * deg), sr = std::sin(o.rollDeg * deg);
  const double Ry[9] = {cy, 0, sy, 0, 1, 0, -sy, 0, cy}, Rx[9] = {1, 0, 0, 0, ct, -st, 0, st, ct},
               Rz[9] = {cr, -sr, 0, sr, cr, 0, 0, 0, 1};
  const auto mul = [](const double* a, const double* b, double* c) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) c[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
  };
  double t[9], R[9];
  mul(Ry, Rx, t);
  mul(t, Rz, R);
  // ai: screen (top-left pixels) to camera: columns R^T e_x, R^T e_y, and the centre's shift
  const double hw = d.width * 0.5, hh = d.height * 0.5;
  double M[9];
  for (int i = 0; i < 3; i++) {
    M[3 * i + 0] = R[0 * 3 + i];
    M[3 * i + 1] = R[1 * 3 + i];
    M[3 * i + 2] = -hw * R[0 * 3 + i] - hh * R[1 * 3 + i] + (i == 2 ? D : 0);
  }
  const double K[9] = {fl, 0, o.width * 0.5, 0, fl, o.height * 0.5, 0, 0, 1};
  double A[9];
  mul(K, M, A);
  // ai: its inverse, frame to screen
  const double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (std::abs(det) < 1e-300) { why = "the camera sees the screen edge on"; return nullptr; }
  const double I[9] = {A[4] * A[8] - A[5] * A[7], A[2] * A[7] - A[1] * A[8], A[1] * A[5] - A[2] * A[4],
                       A[5] * A[6] - A[3] * A[8], A[0] * A[8] - A[2] * A[6], A[2] * A[3] - A[0] * A[5],
                       A[3] * A[7] - A[4] * A[6], A[1] * A[6] - A[0] * A[7], A[0] * A[4] - A[1] * A[3]};
  // ai: A^-1 maps a frame point to (X, Y, 1) / z, z its depth from the camera: w > 0 where the ray meets the screen's
  // ai: plane in front; scaled by the frame centre's w (positive: the camera looks at the screen's centre)
  for (int i = 0; i < 9; i++) p.hm[i] = I[i] / det;
  const double wc = p.hm[6] * o.width * 0.5 + p.hm[7] * o.height * 0.5 + p.hm[8];
  if (!(wc > 0)) { why = "the camera does not face the screen"; return nullptr; }
  for (double& h : p.hm) h /= wc;
  // ai: the code's place: centred, its frame `share` of the largest the screen holds
  const double scale = d.share * std::min(double(d.height) / g.height, double(d.width) / g.width);
  p.inv = 1 / scale;
  p.ox = (d.width - scale * g.width) * 0.5;
  p.oy = (d.height - scale * g.height) * 0.5;
  for (int i = 0; i < 256; i++) p.lut[i] = float(std::pow(i / 255.0, o.gamma));
  p.sur = p.lut[d.surround];
  p.tone.resize(65536);
  for (int i = 0; i < 65536; i++) p.tone[size_t(i)] = uint8_t(std::lround(255 * std::pow(i / 65535.0, 1 / o.gamma)));
  if (o.blurPx >= 0.05) {
    p.gr = int(std::ceil(3 * o.blurPx));
    p.gk.resize(size_t(p.gr) + 1);
    double sum = 0;
    for (int i = 0; i <= p.gr; i++) sum += (i ? 2 : 1) * (p.gk[size_t(i)] = float(std::exp(-0.5 * i * i / (o.blurPx * o.blurPx))));
    for (float& k : p.gk) k = float(k / sum);
  }
  p.light.resize(size_t(o.width) * o.height);
  if (p.gr) p.tmp.resize(p.light.size());
  const unsigned hc = std::max(2u, std::thread::hardware_concurrency());
  p.pool = std::make_unique<Pool>(int(std::min(16u, hc / 2)));

  // ai: the first pictures painted before the display's clock starts (an encoder's first paint makes its tables), so
  // ai: a capture at the start is not late: two a painter, at most `ahead`, within 5 s
  for (int i = 0; i < painters; i++) p.painters.emplace_back([&p, i] { p.paintLoop(i); });
  {
    std::unique_lock<std::mutex> l(p.mu);
    const uint64_t want = uint64_t(std::min(p.ahead, 2 * painters));
    p.cv.wait_until(l, timeOf(nowNs() + 5'000'000'000LL), [&] { return p.painted >= want; });
    p.t0 = nowNs();
    p.started = true;
  }
  p.cv.notify_all();
  return sc;
}

Scene::Scene(const Sender& s, const Display& d, const Optics& o) : s_(s), d_(d), o_(o) {}
Scene::~Scene() = default;

int64_t Scene::startNs() const { return p_->t0; }

int64_t Scene::pictureAt(int64_t t) const { return p_->shown(p_->refreshAt(double(t - p_->t0))); }

int64_t Scene::shownAt(int64_t k) const { return p_->t0 + p_->refreshNs(p_->first(std::max<int64_t>(k, 0))); }

void Scene::render(int64_t t0, uint8_t* out, uint32_t stride, double exposureMs, double gain) {
  Impl& p = *p_;
  std::lock_guard<std::mutex> one(p.renderMu);
  const int64_t began = nowNs();
  const double span = (exposureMs > 0 ? exposureMs : o_.exposureMs) * 1e6;
  const double readout = o_.rolling && p.H > 1 ? o_.readoutMs * 1e6 : 0;
  const double rel = double(t0 - p.t0);
  // ai: the refreshes any row's window or its settling reaches, and their pictures
  const int64_t jLo = p.refreshAt(rel - kTail * p.tau - p.scanNs) - 1, jHi = p.refreshAt(rel + readout + span) + 1;
  std::vector<int64_t> byRefresh(size_t(jHi - jLo + 1));
  for (int64_t j = jLo; j <= jHi; j++) byRefresh[size_t(j - jLo)] = p.shown(j);
  const int64_t kHi = byRefresh.back(), kLo = std::max(byRefresh.front(), kHi - p.slots + 3);
  std::vector<const uint8_t*> pics;
  bool wasLate = false, wasStale = false;
  {
    std::unique_lock<std::mutex> l(p.mu);
    p.acquire(kLo, kHi, pics, wasLate, wasStale, l);
  }
  const uint32_t H = p.H, W = p.W;
  const int jobs = int((H + kRowsAJob - 1) / kRowsAJob);
  // ai: Each job's scratch is its own, made in the job (2026-10-25): these were the pool threads' thread_locals, and
  // ai: under MinGW-w64 GCC's emulated TLS threads that end holding a thread_local with a destructor corrupt the heap,
  // ai: as the pool's did when a scene ended on Windows.
  p.pool->run(jobs, [&](int job) {
    std::vector<Mix> mixes;
    for (uint32_t r = uint32_t(job) * kRowsAJob, e = std::min(H, r + kRowsAJob); r < e; r++)
      p.sampleRow(r, rel + (H > 1 ? double(r) * readout / (H - 1) : 0), span, jLo, byRefresh, kLo, kHi, pics, mixes);
  });
  {
    std::lock_guard<std::mutex> l(p.mu);
    p.pinLo = 0;
    p.pinHi = -1;
  }
  p.cv.notify_all();
  // ai: the blur on the light, separable, the edges held
  const int gr = p.gr;
  if (gr) {
    p.pool->run(jobs, [&](int job) {
      for (uint32_t r = uint32_t(job) * kRowsAJob, e = std::min(H, r + kRowsAJob); r < e; r++) {
        const float* in = p.light.data() + size_t(r) * W;
        float* o = p.tmp.data() + size_t(r) * W;
        for (int x = 0; x < int(W); x++) {
          float acc = p.gk[0] * in[x];
          if (x >= gr && x + gr < int(W)) {
            for (int i = 1; i <= gr; i++) acc += p.gk[size_t(i)] * (in[x - i] + in[x + i]);
          } else {
            for (int i = 1; i <= gr; i++) acc += p.gk[size_t(i)] * (in[std::max(x - i, 0)] + in[std::min(x + i, int(W) - 1)]);
          }
          o[x] = acc;
        }
      }
    });
  }
  // ai: the sensor: black + gain light (1 - black), read and shot noise (a normal from one 64-bit draw: the popcount of
  // ai: 52 bits, binomial(52, 1/2), smoothed by a 12-bit uniform; seeded by the seed, the capture's time and the row,
  // ai: so a capture is the same whatever the threads), clamped, to 8 bits through 1 / gamma
  const float black = float(o_.black), k1 = float((gain > 0 ? gain : o_.gain) * (1 - o_.black));
  const float read2 = float(o_.readNoise * o_.readNoise), shot2 = float(o_.shotNoise * o_.shotNoise);
  const bool noisy = read2 > 0 || shot2 > 0;
  const uint64_t seed = mix(uint64_t(o_.seed) * 0x9e3779b97f4a7c15ull ^ mix(uint64_t(t0)));
  constexpr float kNorm = 0.27639f;   // ai: 1 / sqrt(13 + 1 / 12)
  p.pool->run(jobs, [&](int job) {
    std::vector<float> acc(gr ? W : 0);   // ai: the blur's vertical pass, a row (the job's own, as above)
    for (uint32_t r = uint32_t(job) * kRowsAJob, e = std::min(H, r + kRowsAJob); r < e; r++) {
      const float* L = p.light.data() + size_t(r) * W;
      if (gr) {
        const float* c0 = p.tmp.data() + size_t(r) * W;
        for (uint32_t x = 0; x < W; x++) acc[x] = p.gk[0] * c0[x];
        for (int i = 1; i <= gr; i++) {
          const float* up = p.tmp.data() + size_t(std::max(int(r) - i, 0)) * W;
          const float* dn = p.tmp.data() + size_t(std::min(int(r) + i, int(H) - 1)) * W;
          const float k = p.gk[size_t(i)];
          for (uint32_t x = 0; x < W; x++) acc[x] += k * (up[x] + dn[x]);
        }
        L = acc.data();
      }
      uint64_t st = mix(seed ^ (uint64_t(r) * 0xd1b54a32d192ed03ull));
      uint8_t* o = out + size_t(r) * stride;
      for (uint32_t x = 0; x < W; x++) {
        const float sig = k1 * L[x];
        float val = black + sig;
        if (noisy) {
          st += 0x9e3779b97f4a7c15ull;
          uint64_t z = st;
          z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
          z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
          z ^= z >> 31;
          const float n = (float(__builtin_popcountll(z & ((1ull << 52) - 1)) - 26) + (float(z >> 52) + 0.5f) * (1.f / 4096) - 0.5f) * kNorm;
          val += std::sqrt(read2 + shot2 * std::max(sig, 0.f)) * n;
        }
        val = std::min(std::max(val, 0.f), 1.f);
        o[x] = p.tone[size_t(val * 65535.f + 0.5f)];
      }
    }
  });
  const double ms = double(nowNs() - began) * 1e-6;
  std::lock_guard<std::mutex> l(p.mu);
  p.renders++;
  p.renderMs += ms;
  if (wasLate) p.late++;
  if (wasStale) p.stale++;
}

std::string Scene::stats() {
  Impl& p = *p_;
  std::lock_guard<std::mutex> l(p.mu);
  return std::string("{\"painted\":") + std::to_string(p.painted) + ",\"late\":" + std::to_string(p.late) +
         ",\"stale\":" + std::to_string(p.stale) + ",\"renders\":" + std::to_string(p.renders) +
         ",\"paintMs\":" + num("%.3f", p.painted ? p.paintMs / double(p.painted) : 0) +
         ",\"renderMs\":" + num("%.3f", p.renders ? p.renderMs / double(p.renders) : 0) +
         ",\"frameBlocks\":" + std::to_string(p.g.frame_blocks) + ",\"n\":" + std::to_string(p.g.n) +
         ",\"side\":" + std::to_string(p.g.side) + ",\"width\":" + std::to_string(p.g.width) +
         ",\"height\":" + std::to_string(p.g.height) + ",\"slots\":" + std::to_string(p.slots) + "}";
}

}  // namespace lizard::sim
