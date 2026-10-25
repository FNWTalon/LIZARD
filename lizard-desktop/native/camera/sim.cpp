// ai: The simulated camera (2026-10-25, backends.h simList and simOpen): sim/scene.h's screen filmed by a camera that
// ai: runs as a real backend's does, for every test of the receiver where no camera is. Listed only where
// ai: LIZ_CAMERA_SIM is set, its value the scene's spec plus the camera's own keys: camfps (its rate, 60), camppm (its
// ai: clock against the host's, 0), trigger (1: it takes a software trigger, one exposure a trigger: camera.h
// ai: Trigger), latency (ms from a trigger to the exposure, 0.05), jitter (ms, that latency's spread, a seeded normal,
// ai: 0.02), format (grey, yuyv, nv12 or mjpeg: the bytes its frames come in, so luma.h's and jpeg.h's paths run as a
// ai: real camera's would).
// ai: Free run: capture k begins at the stream's start + k / (camfps (1 + camppm 1e-6)) and is handed over once its
// ai: exposure and readout are over (sleepUntil, then the render); one handed over past the next one's hand-over time
// ai: drops that next one, counted. With a trigger begun, each trigger's fire time is queued and its capture begins
// ai: latency + jitter after it, handed over with the time and the delayed mark TriggerClock::match gives for its
// ai: arrival (steadyNs at hand-over), Frame::triggered set; end() goes back to free run. A schedule begun or ended
// ai: forgets the captures of the one before, queued or being exposed, as the clock forgets their triggers; a free
// ai: run's capture being exposed is handed over as it is. Frame::ns is on steadyNs's clock always.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <locale>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "backends.h"
#include "jpeg.h"
#include "luma.h"
#include "sim/scene.h"
#include "trigger.h"

namespace lizard::cam {
namespace {

constexpr int kJpegQuality = 85;   // ai: a webcam's MJPEG

struct Form { const char* key; Pix pix; const char* name; uint32_t fourcc; };
constexpr uint32_t fourcc(char a, char b, char c, char d) { return uint32_t(a) | uint32_t(b) << 8 | uint32_t(c) << 16 | uint32_t(d) << 24; }
const Form kForms[] = {
    {"grey", Pix::Grey8, "GREY", fourcc('G', 'R', 'E', 'Y')},
    {"yuyv", Pix::YUYV, "YUYV", fourcc('Y', 'U', 'Y', 'V')},
    {"nv12", Pix::NV12, "NV12", fourcc('N', 'V', '1', '2')},
    {"mjpeg", Pix::MJPEG, "MJPG", fourcc('M', 'J', 'P', 'G')},
};

bool number(const std::string& text, double& v) {
  std::istringstream is(text);
  is.imbue(std::locale::classic());
  is >> v;
  return !is.fail() && is.peek() == std::char_traits<char>::eof() && std::isfinite(v);
}

// ai: the camera's own keys out of the spec, the rest left for sim::parse
struct Own {
  double fps = 60, ppm = 0, latencyMs = 0.05, jitterMs = 0.02;
  bool trigger = false;
  const Form* form = &kForms[0];
};
bool split(const std::string& spec, Own& own, std::string& rest, std::string& why) {
  std::istringstream in(spec);
  std::string tok;
  rest.clear();
  while (in >> tok) {
    const size_t eq = tok.find('=');
    const std::string key = eq == std::string::npos ? tok : tok.substr(0, eq), val = eq == std::string::npos ? "" : tok.substr(eq + 1);
    double x = 0;
    const auto take = [&](double lo, double hi, const char* what) {
      if (!number(val, x) || x < lo || x > hi) { why = key + "=" + val + ": " + what; return false; }
      return true;
    };
    if (key == "camfps") { if (!take(1, 1000, "a rate from 1 to 1000")) return false; own.fps = x; }
    else if (key == "camppm") { if (!take(-100000, 100000, "ppm from -100000 to 100000")) return false; own.ppm = x; }
    else if (key == "trigger") {
      if (val != "0" && val != "1") { why = "trigger=" + val + ": 0 or 1"; return false; }
      own.trigger = val == "1";
    }
    else if (key == "latency") { if (!take(0, 1000, "ms from 0 to 1000")) return false; own.latencyMs = x; }
    else if (key == "jitter") { if (!take(0, 100, "ms from 0 to 100")) return false; own.jitterMs = x; }
    else if (key == "format") {
      own.form = nullptr;
      for (const Form& f : kForms) if (val == f.key) own.form = &f;
      if (!own.form) { why = "format=" + val + ": grey, yuyv, nv12 or mjpeg"; return false; }
    }
    else { if (!rest.empty()) rest += ' '; rest += tok; }
  }
  return true;
}

class SimCamera;

class SimTrigger : public Trigger {
 public:
  explicit SimTrigger(SimCamera& cam);
  bool begin(int64_t periodNs, std::string& why) override;
  void delay(int64_t ns) override { clock_.delay(ns); }
  void end() override;
  bool running() const override { return clock_.running(); }
  std::string stats() override { return clock_.stats(); }
  bool match(int64_t arrivedNs, int64_t& ns, bool& delayed) { return clock_.match(arrivedNs, ns, delayed); }

 private:
  SimCamera& cam_;
  TriggerClock clock_;
};

class SimCamera : public Camera {
 public:
  SimCamera(const std::string& spec, const Own& own, const sim::Sender& s, const sim::Display& d, const sim::Optics& o)
      : own_(own), s_(s), d_(d), o_(o),
        exposureUs_(std::min(33333.0, std::max(100.0, o.exposureMs * 1000))), gain_(1.0) {
    info_.id = "sim:0";
    info_.name = "Simulated camera";
    info_.backend = "sim";
    info_.detail = spec;
    if (own.trigger) trig_ = std::make_unique<SimTrigger>(*this);
  }
  ~SimCamera() override {
    stop();
    trig_.reset();   // ai: its clock's thread fires into this camera: ended before the rest goes
  }

  const Info& info() const override { return info_; }

  std::vector<Mode> modes() override {
    std::vector<Mode> m;
    for (const uint32_t div : {1u, 2u}) {
      Mode x;
      x.width = o_.width / div;
      x.height = o_.height / div;
      x.fps = x.fpsMin = own_.fps;
      x.format = own_.form->name;
      x.pix = own_.form->pix;
      x.compressed = x.pix == Pix::MJPEG;
      x.key = own_.form->fourcc;
      m.push_back(x);
    }
    return m;
  }

  std::vector<Control> controls() override {
    Control e;
    e.id = "exposure"; e.name = "Exposure"; e.unit = "us";
    e.min = 100; e.max = 33333; e.step = 1;
    e.def = std::min(33333.0, std::max(100.0, o_.exposureMs * 1000));
    e.value = exposureUs_.load();
    Control g;
    g.id = "gain"; g.name = "Gain"; g.unit = "x";
    g.min = 0.25; g.max = 4; g.step = 0.01; g.def = 1; g.value = gain_.load();
    return {e, g};
  }

  bool set(const std::string& id, double value, bool automatic, std::string& why) override {
    if (id != "exposure" && id != "gain") { why = "the simulated camera has no control " + id; return false; }
    if (automatic) { why = "the simulated camera's " + id + " has no automatic"; return false; }
    if (id == "exposure") {
      if (!(value >= 100 && value <= 33333)) { why = "exposure is 100 to 33333 us"; return false; }
      exposureUs_ = value;
    } else {
      if (!(value >= 0.25 && value <= 4)) { why = "gain is 0.25 to 4"; return false; }
      gain_ = value;
    }
    return true;
  }

  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override {
    stop();
    const auto ms = modes();
    const auto it = std::find_if(ms.begin(), ms.end(), [&](const Mode& m) { return m.width == mode.width && m.height == mode.height && m.pix == mode.pix; });
    if (it == ms.end()) { why = "the simulated camera has no mode " + std::to_string(mode.width) + " x " + std::to_string(mode.height) + " " + mode.format; return false; }
    if (mode.pix == Pix::MJPEG && !(jpeg_ = Jpeg::load(why))) return false;
    sim::Optics o = o_;
    o.width = mode.width;
    o.height = mode.height;
    scene_ = sim::Scene::create(s_, d_, o, 3, why);
    if (!scene_) return false;
    mode_ = *it;
    sink_ = std::move(sink);
    ended_ = std::move(ended);
    {
      std::lock_guard<std::mutex> l(mu_);
      stats_ = Stats{};
      convertSum_ = 0;
      fires_.clear();
      streaming_ = true;
      quit_ = false;
    }
    th_ = std::thread([this] { loop(); });
    return true;
  }

  void stop() override {
    {
      std::lock_guard<std::mutex> l(mu_);
      quit_ = true;
      streaming_ = false;
      fires_.clear();
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    scene_.reset();
    jpeg_.reset();
  }

  Stats stats() override {
    std::lock_guard<std::mutex> l(mu_);
    Stats s = stats_;
    s.convertMs = s.frames ? convertSum_ / double(s.frames) : 0;
    return s;
  }

  Trigger* trigger() override { return trig_.get(); }

  // ai: SimTrigger's: a fire queued (its time) while streaming; the schedule's start and end
  void fired(int64_t at) {
    {
      std::lock_guard<std::mutex> l(mu_);
      if (!streaming_ || !triggered_) return;
      fires_.push_back(at);
    }
    cv_.notify_all();
  }
  // ai: a schedule begun or ended, the clock stopped meanwhile (SimTrigger): the captures of the one before forgotten,
  // ai: as the clock forgets their triggers: those queued here, and one being exposed, which its hand-over finds of an
  // ai: older schedule_
  void triggered(bool on) {
    {
      std::lock_guard<std::mutex> l(mu_);
      triggered_ = on;
      fires_.clear();
      schedule_++;
    }
    cv_.notify_all();
  }

 private:
  // ai: the capture thread
  void loop() {
    const double period = 1e9 / (own_.fps * (1 + own_.ppm * 1e-6));
    const uint32_t w = mode_.width, h = mode_.height;
    std::mt19937_64 rng(uint64_t(o_.seed) * 0x9e3779b97f4a7c15ull + 0x6a6974746572ull);
    std::normal_distribution<double> normal(0, 1);
    // ai: the frame as the camera sends it (packed), and the luma a backend makes of it
    std::vector<uint8_t> grey(size_t(w) * h), bytes, luma;
    if (mode_.pix == Pix::YUYV) bytes.assign(size_t(w) * 2 * h, 128);
    if (mode_.pix == Pix::NV12) bytes.assign(size_t(w) * h + size_t(w) * ((h + 1) / 2), 128);
    int64_t base = steadyNs();   // ai: free run's capture 0
    int64_t k = 0;
    uint64_t seq = 0;
    bool wasTriggered = false;
    std::unique_lock<std::mutex> l(mu_);
    while (!quit_) {
      int64_t t0 = 0;
      bool trig = false;
      uint64_t sched = 0;   // ai: a triggered capture's schedule (schedule_ when its fire was taken)
      if (triggered_) {
        wasTriggered = true;
        if (fires_.empty()) { cv_.wait(l); continue; }
        t0 = fires_.front();
        fires_.pop_front();
        trig = true;
        sched = schedule_;
        t0 += int64_t(std::max(0.0, own_.latencyMs + own_.jitterMs * normal(rng)) * 1e6);
        // ai: a trigger whose hand-over passed more than a period ago (the thread behind) is dropped, its trigger lost
        if (steadyNs() > t0 + int64_t((exposureUs_.load() * 1e-3 + (o_.rolling ? o_.readoutMs : 0)) * 1e6 + period)) {
          stats_.dropped++;
          seq++;
          continue;
        }
      } else {
        if (wasTriggered) {   // ai: back to free run: its schedule anew from a period from now
          wasTriggered = false;
          base = steadyNs() + int64_t(period);
          k = 0;
        }
        t0 = base + int64_t(double(k) * period);
        k++;
      }
      const double expMs = exposureUs_.load() * 1e-3, gain = gain_.load() * o_.gain;
      const int64_t window = int64_t((expMs + (o_.rolling ? o_.readoutMs : 0)) * 1e6);
      l.unlock();
      // ai: handed over once its exposure and readout are over; a stop meanwhile ends it (waits are short: at most
      // ai: the 33 ms exposure and the readout)
      sleepUntil(t0 + window);
      l.lock();
      if (quit_) break;
      // ai: a schedule begun or ended while it was exposed: its trigger is forgotten, so is it (2026-10-25: a capture
      // ai: of a 30 a second schedule begun again at 60 took the new schedule's first trigger, every later one the
      // ai: trigger after its own)
      if (trig && sched != schedule_) continue;
      l.unlock();
      uint8_t* into = mode_.pix == Pix::NV12 ? bytes.data() : grey.data();
      scene_->render(t0, into, w, expMs, gain);
      Frame f;
      f.width = w;
      f.height = h;
      f.ns = t0;
      bool good = true;
      double convertMs = 0;
      std::string why;
      switch (mode_.pix) {
        case Pix::Grey8:
          f.luma = grey.data();
          f.stride = w;
          break;
        case Pix::NV12:
          // ai: a planar layout's Y plane is the luma in place, once the buffer holds the frame
          good = bytes.size() >= frameBytes(Pix::NV12, w, h, w);
          f.luma = bytes.data();
          f.stride = w;
          break;
        case Pix::YUYV:
          for (uint32_t y = 0; y < h; y++) {
            const uint8_t* g = grey.data() + size_t(y) * w;
            uint8_t* o = bytes.data() + size_t(y) * w * 2;
            for (uint32_t x = 0; x < w; x++) o[2 * x] = g[x];
          }
          {
            const int64_t a = steadyNs();
            good = toLuma(Pix::YUYV, 8, bytes.data(), bytes.size(), w, h, w * 2, luma, &why);
            convertMs = double(steadyNs() - a) * 1e-6;
          }
          f.luma = luma.data();
          f.stride = w;
          break;
        case Pix::MJPEG: {
          good = jpeg_->encode(grey.data(), w, h, w, kJpegQuality, true, bytes, &why);
          uint32_t jw = 0, jh = 0;
          const int64_t a = steadyNs();
          good = good && jpeg_->grey(bytes.data(), bytes.size(), luma, jw, jh, &why) && jw == w && jh == h;
          convertMs = double(steadyNs() - a) * 1e-6;
          f.luma = luma.data();
          f.stride = w;
          break;
        }
        default:
          good = false;
      }
      if (trig) {
        // ai: the frame's time and delayed mark as a real backend reads them: its arrival matched to a trigger, its
        // ai: schedule looked at again first (one begun meanwhile forgets it: the render is a few ms, a schedule's
        // ai: first trigger a period after its begin)
        l.lock();
        if (sched != schedule_) continue;   // ai: with l held, as the loop's top has it
        l.unlock();
        bool delayed = false;
        int64_t ns = 0;
        if (trig_ && trig_->match(steadyNs(), ns, delayed)) {
          f.ns = ns;
          f.delayed = delayed;
          f.triggered = true;
        }
      }
      f.seq = seq++;
      if (good) sink_(f);
      l.lock();
      if (good) { stats_.frames++; convertSum_ += convertMs; } else stats_.corrupt++;
      // ai: free run: a capture whose hand-over time has passed is dropped (the camera had no buffer for it)
      if (!triggered_ && !trig) {
        const int64_t now = steadyNs();
        while (base + int64_t(double(k) * period) + window < now) { k++; stats_.dropped++; seq++; }
      }
    }
  }

  Info info_;
  Own own_;
  sim::Sender s_;
  sim::Display d_;
  sim::Optics o_;
  std::atomic<double> exposureUs_, gain_;
  Mode mode_;
  Sink sink_;
  Ended ended_;
  std::unique_ptr<sim::Scene> scene_;
  std::unique_ptr<Jpeg> jpeg_;
  std::thread th_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool quit_ = false, streaming_ = false, triggered_ = false;
  uint64_t schedule_ = 0;   // ai: mu_: schedules begun and ended (triggered), a capture's generation
  std::deque<int64_t> fires_;
  Stats stats_;
  double convertSum_ = 0;
  std::unique_ptr<SimTrigger> trig_;
};

SimTrigger::SimTrigger(SimCamera& cam) : cam_(cam), clock_([this](std::string&) { cam_.fired(steadyNs()); return true; }) {}

// ai: The schedule before stopped first (end waits out a trigger being fired), then the camera's captures of it
// ai: forgotten, then the new one begun: no trigger fires between the forgetting and the begin, so the camera's queue
// ai: holds the clock's triggers alone (2026-10-25: forgotten before the begin, a trigger of the old schedule being
// ai: fired was queued after it while the clock forgot it, and its capture took the new schedule's first trigger,
// ai: every later one the trigger after its own)
bool SimTrigger::begin(int64_t periodNs, std::string& why) {
  if (periodNs <= 0) { why = "a period of at least a ns"; return false; }
  clock_.end();
  cam_.triggered(true);
  clock_.begin(periodNs);
  return true;
}

void SimTrigger::end() {
  clock_.end();
  cam_.triggered(false);
}

}  // namespace

std::vector<Info> simList(std::vector<std::string>* notes) {
  (void)notes;
  const char* spec = std::getenv("LIZ_CAMERA_SIM");
  if (!spec) return {};
  Info i;
  i.id = "sim:0";
  i.name = "Simulated camera";
  i.backend = "sim";
  i.detail = spec;
  return {i};
}

std::unique_ptr<Camera> simOpen(const std::string& id, std::string& why) {
  const char* spec = std::getenv("LIZ_CAMERA_SIM");
  if (id != "sim:0") { why = "no simulated camera " + id; return nullptr; }
  if (!spec) { why = "LIZ_CAMERA_SIM is not set"; return nullptr; }
  Own own;
  std::string rest;
  if (!split(spec, own, rest, why)) { why = "LIZ_CAMERA_SIM: " + why; return nullptr; }
  sim::Sender s;
  sim::Display d;
  sim::Optics o;
  if (!sim::parse(rest, s, d, o, why)) { why = "LIZ_CAMERA_SIM: " + why; return nullptr; }
  return std::make_unique<SimCamera>(spec, own, s, d, o);
}

}  // namespace lizard::cam
