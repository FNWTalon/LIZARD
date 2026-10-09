// ai: A stand-in Receiver (LIZ_STUB_RECEIVER=ON) in place of core's lizard_rx: every frame is counted and handed back
// ai: at once, from the stub's own thread, as the real one releases from its threads (the JNI attach path runs). It
// ai: decodes nothing: its stats say "looking" with the camera's rates, so the app's camera, push and release are
// ai: checked on a phone before the decoder is.
//
// ai: The stats JSON, the fields the app reads (MainActivity.kt rxOf into Readout.kt Rx; the web receiver's stats row names where one
// ai: exists, lizard-web/recv.mjs):
// ai:   state        loading | looking | test | found | receiving | received | error
// ai:   error        why, with state error, else absent
// ai:   decoder      what decodes: gpu:<variant>, cpu, stub
// ai:   capturedFps  frames pushed a second, over the last second
// ai:   processedFps frames decoded a second (the stub: the frames it released)
// ai:   goodputKBs   new blocks' bytes a second (1 KB = 1000 B), the rate now
// ai:   blocks       CRC-verified blocks since the start
// ai:   foundShare   share of the last second's frames registered, 0 to 1
// ai:   bandVersion  the version the last read word names (sub-channels / 8), 0 before any
// ai:   file         null before a transfer's header, else {name, size, received, verified, root, type, sent, sentIn}
// ai:                (bytes, bytes, bool, the BLAKE3 root in hex, the header's media type, the bytes as sent and those
// ai:                of them in, 2026-10-05)
// ai:   word         the last word read: version, fps (the rate it states)
// ai: and, for the Developer Tools readout only: frames, dropped, res, input (hb or luma), gpuMs.
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

#include "receiver.h"

namespace lizard {
namespace {

using Clock = std::chrono::steady_clock;

class StubReceiver final : public Receiver {
 public:
  StubReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release)
      : cfg_(c), release_(std::move(release)), worker_([this] { run(); }) {
    say("stub receiver: assets " + cfg_.assets + ", decoder " + cfg_.decoder + ", precision " + cfg_.precision);
  }

  ~StubReceiver() override {
    { std::lock_guard<std::mutex> l(m_); stop_ = true; }
    cv_.notify_one();
    worker_.join();
    for (uint64_t t : queue_) release_(t);   // ai: none left after run(); kept so the contract holds if run changes
  }

  void push(const CameraFrame& f) override {
    {
      std::lock_guard<std::mutex> l(m_);
      queue_.push_back(f.tag);
      ++pushed_;
      w_ = f.width, h_ = f.height;
      hb_ = f.hb != nullptr;
    }
    cv_.notify_one();
  }

  std::string stats() override {
    std::lock_guard<std::mutex> l(m_);
    char s[512];
    snprintf(s, sizeof s,
             "{\"state\":\"looking\",\"decoder\":\"stub\",\"capturedFps\":%.2f,\"processedFps\":%.2f,"
             "\"goodputKBs\":0,\"blocks\":0,\"foundShare\":0,\"bandVersion\":0,\"file\":null,"
             "\"frames\":%llu,\"dropped\":0,\"res\":\"%ux%u\",\"input\":\"%s\",\"gpuMs\":0}",
             capturedFps_, processedFps_, static_cast<unsigned long long>(pushed_), w_, h_, hb_ ? "hb" : "luma");
    return s;
  }

  // ai: no frame is decoded: no word, no rows
  std::vector<double> series(double) override { return {0}; }
  void batchCap(int) override {}
  std::string file() override { return {}; }
  void clear() override {}
  bool wantsLuma() const override { return false; }   // ai: it reads nothing: the camera's buffers as they are

 private:
  void say(const std::string& s) { if (cfg_.log) cfg_.log(s); }

  // ai: releases whatever is queued, and once a second works out the rates and logs them (the logcat check)
  void run() {
    auto t0 = Clock::now();
    uint64_t pushed0 = 0, released0 = 0;
    std::unique_lock<std::mutex> l(m_);
    for (;;) {
      cv_.wait_for(l, std::chrono::milliseconds(250), [this] { return stop_ || !queue_.empty(); });
      while (!queue_.empty()) {
        uint64_t t = queue_.front();
        queue_.pop_front();
        l.unlock();
        release_(t);
        l.lock();
        ++released_;
      }
      if (stop_) return;
      double dt = std::chrono::duration<double>(Clock::now() - t0).count();
      if (dt >= 1.0) {
        capturedFps_ = (pushed_ - pushed0) / dt;
        processedFps_ = (released_ - released0) / dt;
        pushed0 = pushed_, released0 = released_, t0 = Clock::now();
        char s[160];
        snprintf(s, sizeof s, "stub: push %.1f fps, released %.1f fps, %ux%u %s, %llu frames", capturedFps_,
                 processedFps_, w_, h_, hb_ ? "hb" : "luma", static_cast<unsigned long long>(pushed_));
        bool any = capturedFps_ > 0;   // ai: quiet while the camera is off
        l.unlock();
        if (any) say(s);
        l.lock();
      }
    }
  }

  ReceiverConfig cfg_;
  std::function<void(uint64_t)> release_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<uint64_t> queue_;
  bool stop_ = false, hb_ = false;
  uint64_t pushed_ = 0, released_ = 0;
  uint32_t w_ = 0, h_ = 0;
  double capturedFps_ = 0, processedFps_ = 0;
  std::thread worker_;   // ai: last: started once every member above is built
};

}  // namespace

std::unique_ptr<Receiver> Receiver::create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release) {
  return std::make_unique<StubReceiver>(c, std::move(release));
}

}  // namespace lizard
