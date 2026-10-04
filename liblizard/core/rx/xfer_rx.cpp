// ai: XferRx (xfer_rx.h): the receiving end's threaded form, the app's: the judge on the producer's thread, a queue,
// ai: and the core (xfer_rx_core.h) on a worker thread with files under the store directory, so take never waits on a
// ai: solve. The chain itself is xfer_rx_core.cpp's.
#include "xfer_rx.h"

#include <cstring>

namespace lizard {

// ai: the queue's bound, 31 MB: a 4 MiB chunk's solve and hash are tens of ms, so this is seconds of backlog at any
// ai: format's rate; past it a block is lost (counted), which the fountain covers as any other loss
static constexpr size_t QUEUE_MAX = 1 << 16;

XferRx::XferRx(const std::string& storeDir, std::function<void(const std::string&)> log) : log_(std::move(log)) {
  rx_ = std::make_unique<XferRxCore>(fileStore(storeDir + "/lizard-xfer"), log_);
  judge_ = std::make_unique<XferJudge>(rx_->rootGen());
  rx_->progress(snap_);
  worker_ = std::thread([this] { run(); });
}

XferRx::~XferRx() {
  { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
  cv_.notify_all();
  worker_.join();
}

void XferRx::push(uint8_t kind, const uint8_t* b) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (kind == 0 && queue_.size() >= QUEUE_MAX) { lost_++; return; }
    queue_.push_back(Item{kind, {}});
    if (b) memcpy(queue_.back().b.data(), b, XFER_BLOCK_BYTES);
  }
  cv_.notify_one();
}

void XferRx::run() {
  std::deque<Item> batch;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      busy_ = false;
      idle_.notify_all();
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      batch.swap(queue_);
      busy_ = true;
    }
    for (size_t k = 0; k < batch.size(); k++) {
      if (batch[k].kind == 1) rx_->clearAll(); else rx_->block(batch[k].b.data());
      // ai: a snapshot every 256 blocks as well as at the end, so progress moves through a long backlog
      if ((k & 255) == 255) { std::lock_guard<std::mutex> lk(pmu_); rx_->progress(snap_); }
    }
    batch.clear();
    std::lock_guard<std::mutex> lk(pmu_);
    rx_->progress(snap_);
  }
}

bool XferRx::take(const uint8_t* b) { return judge_->take(b, [this](const uint8_t* p) { push(0, p); }); }

FrameVerdict XferRx::frame(const std::vector<Block>& verified) {
  return judge_->frame(verified.empty() ? nullptr : verified.front().bytes.data(), verified.size(), [this](const uint8_t* p) { push(0, p); });
}

XferProgress XferRx::progress() {
  XferProgress p;
  { std::lock_guard<std::mutex> lk(pmu_); p = snap_; }
  std::lock_guard<std::mutex> lk(mu_);
  p.queued = uint32_t(queue_.size());
  p.lost = lost_;
  return p;
}

void XferRx::clear() {
  judge_->clear();
  push(1, nullptr);
}

void XferRx::drain() {
  std::unique_lock<std::mutex> lk(mu_);
  idle_.wait(lk, [&] { return queue_.empty() && !busy_; });
}

}  // namespace lizard
