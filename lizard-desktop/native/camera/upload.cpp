// ai: The Upload camera (2026-10-28, backends.h uploadOpen): a replay played as a camera, a .lizard (the phone's
// ai: Download, liblizard/core/rx/lizfile.h) or a run's folder as Save replays leaves it, id upload:<path>. Never
// ai: listed: Receive's Camera menu asks for the file. Its frames are the crops a receiver read: a 1:1 run's frames as
// ai: they are; a 2:1 run's halves, two frames of one capture's time, set side by side (or one over the other, where
// ai: their camera axes say the halves were stacked) into the frame the receiver's 2:1 crop takes them from again, a
// ai: half with no partner dropped. One mode, the frames' size at the run's rate (its captures' median interval, else
// ai: 60), no controls. Each capture is handed over at its recorded time after the start (meta.json's ms; k / 60 s
// ai: where it has none), Frame::ns that time on steadyNs's clock; the file's end ends the stream ("the upload ended").
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

#include "backends.h"
#include "clock.h"
#include "lizfile.h"

namespace lizard::cam {
namespace {

// ai: a capture: its frames' indices (one, or a 2:1 run's two halves in their places: first left or top)
struct Capture { size_t a = 0, b = SIZE_MAX; double ms = NAN; };

class UploadCamera : public Camera {
 public:
  UploadCamera(std::string id, std::unique_ptr<lizfile::Reader> r) : r_(std::move(r)) {
    info_.id = std::move(id);
    const auto& fr = r_->frames();
    const bool two = r_->layout() == "2:1";
    // ai: the captures, in order; a 2:1 run's halves paired by their capture's time
    for (size_t i = 0; i < fr.size();) {
      const bool pair = two && i + 1 < fr.size() && fr[i].w == fr[i + 1].w && fr[i].h == fr[i + 1].h && fr[i].w == fr[i].h &&
                        (std::isnan(fr[i].ms) || std::fabs(fr[i].ms - fr[i + 1].ms) < 1e-3);
      if (two && !pair) { lone_++; i++; continue; }
      Capture c;
      c.ms = fr[i].ms;
      if (pair) {
        // ai: a half's axis is the middle of its inner edge (core/rx/receiver.cpp axisOf): x = side on the left half,
        // ai: y = side on the top one; without axes the run's order, left first
        const float s = float(fr[i].w);
        const bool firstIsSecond = (fr[i].ax == 0 && fr[i + 1].ax == s) || (fr[i].ay == 0 && fr[i + 1].ay == s);
        stacked_ = stacked_ || fr[i].ay == s || fr[i].ay == 0;
        c.a = firstIsSecond ? i + 1 : i;
        c.b = firstIsSecond ? i : i + 1;
        i += 2;
      } else {
        c.a = i++;
      }
      caps_.push_back(c);
    }
    if (!caps_.empty()) {
      const lizfile::FrameInfo& f = fr[caps_[0].a];
      w_ = two && !stacked_ ? 2 * f.w : f.w;
      h_ = two && stacked_ ? 2 * f.h : f.h;
    }
    std::vector<double> gaps;
    for (size_t k = 1; k < caps_.size(); k++)
      if (!std::isnan(caps_[k].ms) && !std::isnan(caps_[k - 1].ms) && caps_[k].ms > caps_[k - 1].ms) gaps.push_back(caps_[k].ms - caps_[k - 1].ms);
    if (!gaps.empty()) {
      std::nth_element(gaps.begin(), gaps.begin() + long(gaps.size() / 2), gaps.end());
      fps_ = 1000.0 / gaps[gaps.size() / 2];
    }
    const std::string name = std::filesystem::u8path(info_.id.substr(7)).filename().u8string();
    info_.name = name.empty() ? "Upload" : name;
    info_.backend = "upload";
    info_.detail = r_->layout() + ", " + std::to_string(caps_.size()) + " captures";
  }
  ~UploadCamera() override { stop(); }

  const Info& info() const override { return info_; }
  bool empty() const { return caps_.empty(); }

  std::vector<Mode> modes() override {
    Mode m;
    m.width = w_;
    m.height = h_;
    m.fps = m.fpsMin = fps_;
    m.format = "GREY";
    m.pix = Pix::Grey8;
    return {m};
  }
  std::vector<Control> controls() override { return {}; }
  bool set(const std::string& id, double, bool, std::string& why) override {
    why = "an upload has no control " + id;
    return false;
  }

  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override {
    stop();
    if (mode.width != w_ || mode.height != h_) { why = "an upload has one mode, " + std::to_string(w_) + " x " + std::to_string(h_); return false; }
    sink_ = std::move(sink);
    ended_ = std::move(ended);
    {
      std::lock_guard<std::mutex> l(mu_);
      stats_ = Stats{};
      stats_.fps = fps_;
      stats_.dropped = lone_;
      convertSum_ = 0;
    }
    quit_ = false;
    th_ = std::thread([this] { loop(); });
    return true;
  }
  void stop() override {
    quit_ = true;
    if (th_.joinable()) th_.join();
  }
  Stats stats() override {
    std::lock_guard<std::mutex> l(mu_);
    Stats s = stats_;
    s.convertMs = s.frames ? convertSum_ / double(s.frames) : 0;
    return s;
  }

 private:
  Info info_;
  std::unique_ptr<lizfile::Reader> r_;
  std::vector<Capture> caps_;
  uint32_t w_ = 0, h_ = 0;
  double fps_ = 60;
  bool stacked_ = false;
  uint64_t lone_ = 0;
  Sink sink_;
  Ended ended_;
  std::thread th_;
  std::atomic<bool> quit_{false};
  std::mutex mu_;
  Stats stats_;
  double convertSum_ = 0;

  // ai: the capture thread: each capture unpacked, then handed over at its time (waits in slices, so a stop is prompt)
  void loop() {
    std::vector<uint8_t> frame(size_t(w_) * h_), half;
    const int64_t base = steadyNs();
    const double ms0 = caps_.empty() ? 0 : caps_[0].ms;
    for (size_t k = 0; k < caps_.size() && !quit_; k++) {
      const Capture& c = caps_[k];
      const int64_t t0 = steadyNs();
      std::string why;
      bool ok = true;
      for (int part = 0; part < (c.b == SIZE_MAX ? 1 : 2) && ok; part++) {
        ok = r_->frame(part ? c.b : c.a, half, why);
        if (!ok) break;
        const lizfile::FrameInfo& fi = r_->frames()[part ? c.b : c.a];
        if (c.b == SIZE_MAX) {
          if (fi.w == w_ && fi.h == h_) frame.swap(half);
          else { ok = false; why = "a frame of another size"; }
          continue;
        }
        // ai: a half into its place: the second right of the first, or under it
        const uint32_t x0 = !stacked_ && part ? fi.w : 0, y0 = stacked_ && part ? fi.h : 0;
        for (uint32_t y = 0; y < fi.h; y++) std::copy_n(half.data() + size_t(y) * fi.w, fi.w, frame.data() + size_t(y0 + y) * w_ + x0);
      }
      const double convert = double(steadyNs() - t0) * 1e-6;
      if (!ok) {
        std::lock_guard<std::mutex> l(mu_);
        stats_.corrupt++;
        if (why.find("CRC") != std::string::npos || why.find("cut short") != std::string::npos) { ended_(why); return; }
        continue;
      }
      const int64_t at = base + int64_t(std::isnan(c.ms) || std::isnan(ms0) ? double(k) * 1e9 / fps_ : (c.ms - ms0) * 1e6);
      while (!quit_) {
        const int64_t now = steadyNs();
        if (now >= at) break;
        sleepUntil(std::min(at, now + 50'000'000));
      }
      if (quit_) break;
      Frame f;
      f.luma = frame.data();
      f.width = w_;
      f.height = h_;
      f.stride = w_;
      f.ns = at;
      f.seq = k;
      sink_(f);
      std::lock_guard<std::mutex> l(mu_);
      stats_.frames++;
      convertSum_ += convert;
    }
    if (!quit_) ended_("the upload ended");
  }
};

}  // namespace

std::unique_ptr<Camera> uploadOpen(const std::string& id, std::string& why) {
  const std::string path = id.substr(std::strlen("upload:"));
  if (path.empty()) { why = "no file to upload"; return nullptr; }
  auto r = lizfile::Reader::open(path, why);
  if (!r) return nullptr;
  auto c = std::make_unique<UploadCamera>(id, std::move(r));
  if (c->empty()) { why = path + " holds no capture to play"; return nullptr; }
  return c;
}

}  // namespace lizard::cam
