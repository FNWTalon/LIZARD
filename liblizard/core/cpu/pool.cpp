#include "pool.h"
#include "codec.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace lizard {

static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool PoolPolicy::lost(bool allReady, int size, int ceiling, double repeatShare) {
  // ai: a worker still starting is capacity already on its way
  if (!allReady) return false;
  if (repeatShare > repeatCeiling) { pressure = 0; return false; }
  if (++pressure < threshold || size >= ceiling) return false;
  pressure = 0; slack = 0;
  return true;
}

bool PoolPolicy::tick(int busySkips, int size, bool hasIdle, bool autoSize) {
  if (!busySkips) pressure = 0;
  slack = busySkips <= slackSkips && size > 1 && hasIdle ? slack + 1 : 0;
  if (slack < slackSecs || !autoSize) return false;
  slack = 0; pressure = 0;
  return true;
}

// ai: A worker: its thread, its decoder (made on the thread itself, so the codec's per-thread tables are its own),
// ai: and the one frame it holds. state is the pool's (mu_); the frame's fields are the worker's once it is Busy.
struct CpuPool::Worker {
  enum State { Booting, Idle, Busy, Dead };
  State state = Booting;
  std::thread thread;
  std::condition_variable cv;
  std::vector<uint8_t> luma;
  uint32_t w = 0, h = 0;
  uint64_t tag = 0;
  int held = 0;
  double at = 0;
  bool go = false;
};

CpuPool::CpuPool(int nmax, int ceiling, int fixed, std::function<void(CpuFrameOut&&)> done, std::function<void(const std::string&)> log, std::string tiers)
    : nmax_(nmax), tiers_(std::move(tiers)), done_(std::move(done)), log_(std::move(log)) {
  const int cores = std::max(1u, std::thread::hardware_concurrency());
  ceiling_ = ceiling > 0 ? ceiling : cores;
  fixed_ = std::min(fixed, ceiling_);
  std::lock_guard<std::mutex> l(mu_);
  for (int i = 0; i < std::max(1, fixed_); i++) grow();
}

CpuPool::~CpuPool() { stop(); }

void CpuPool::stop() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
    for (auto& w : workers_) w->cv.notify_all();
  }
  for (auto& w : workers_) if (w->thread.joinable()) w->thread.join();
}

// ai: mu_ held. A parked worker comes back, else a new one starts.
void CpuPool::grow() {
  const int n = size_.load();
  if (n < (int)workers_.size()) { size_ = n + 1; return; }
  auto w = std::make_unique<Worker>();
  Worker* p = w.get();
  workers_.push_back(std::move(w));
  size_ = n + 1;
  p->thread = std::thread([this, p] { run(p); });
}

int CpuPool::ready() const {
  std::lock_guard<std::mutex> l(mu_);
  int r = 0;
  for (int i = 0; i < size_.load(); i++) r += workers_[i]->state == Worker::Idle || workers_[i]->state == Worker::Busy;
  return r;
}

bool CpuPool::offer(const uint8_t* plane, uint32_t stride, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag, int held, double at) {
  Worker* got = nullptr;
  {
    std::lock_guard<std::mutex> l(mu_);
    for (int i = 0; i < size_.load() && !got; i++) if (workers_[i]->state == Worker::Idle) got = workers_[i].get();
    if (!got) return false;
    got->state = Worker::Busy;
  }
  // ai: the worker is this caller's until go: the copy runs outside the lock
  got->luma.resize((size_t)w * h);
  for (uint32_t r = 0; r < h; r++) memcpy(got->luma.data() + (size_t)r * w, plane + (size_t)(y + r) * stride + x, w);
  got->w = w; got->h = h; got->tag = tag; got->held = held; got->at = at;
  {
    std::lock_guard<std::mutex> l(mu_);
    got->go = true;
  }
  got->cv.notify_one();
  return true;
}

void CpuPool::lost(double repeatShare) {
  std::lock_guard<std::mutex> l(mu_);
  if (fixed_) return;
  bool allReady = true;
  for (int i = 0; i < size_.load(); i++) allReady = allReady && workers_[i]->state != Worker::Booting;
  if (policy_.lost(allReady, size_.load(), ceiling_, repeatShare)) {
    grow();
    if (log_) log_("cpu pool: " + std::to_string(size_.load()) + " workers (frames were being lost to busy ones)");
  }
}

void CpuPool::tick(int busySkips) {
  std::lock_guard<std::mutex> l(mu_);
  bool hasIdle = false;
  for (int i = 0; i < size_.load(); i++) hasIdle = hasIdle || workers_[i]->state == Worker::Idle;
  if (policy_.tick(busySkips, size_.load(), hasIdle, !fixed_)) {
    // ai: the last worker is parked: it finishes the frame it holds and is given no other
    size_ = size_.load() - 1;
    if (log_) log_("cpu pool: " + std::to_string(size_.load()) + " workers (ten slack seconds)");
  }
}

void CpuPool::run(Worker* w) {
  cpu_dec_t* dec = cpu_dec_new(nmax_);
  std::vector<uint8_t> blocks, ok;
  int bb = 0;
  if (dec && !tiers_.empty()) {
    char why[160];
    if (cpu_dec_tiers(dec, tiers_.c_str(), why, sizeof why) < 0 && log_) log_(std::string("cpu pool: the rate profile was refused, one rate: ") + why);
  }
  if (dec) {
    bb = cpu_dec_block_bytes(dec);
    blocks.resize((size_t)cpu_dec_top(dec) * bb);
    ok.resize((size_t)cpu_dec_top(dec));
  } else if (log_) log_("cpu pool: the codec refused a decoder to n = " + std::to_string(nmax_));
  std::unique_lock<std::mutex> l(mu_);
  w->state = dec ? Worker::Idle : Worker::Dead;
  while (dec) {
    w->cv.wait(l, [&] { return stop_ || w->go; });
    if (stop_) break;
    w->go = false;
    l.unlock();
    CpuFrameOut out;
    out.tag = w->tag; out.at = w->at; out.blockBytes = bb;
    cpu_frame_t fr;
    const double t0 = nowMs();
    cpu_dec_frame(dec, w->luma.data(), (int)w->w, (int)w->h, w->held, blocks.data(), ok.data(), &fr);
    out.ms = nowMs() - t0;
    out.found = fr.found != 0; out.ring = fr.ring; out.n = fr.n; out.held = fr.held != 0;
    out.hasWord = fr.word != 0; out.version = fr.version; out.fps = fr.fps;
    memcpy(out.quad, fr.quad, sizeof out.quad);
    out.pilotBlocks = fr.pilot_blocks; out.pilotR = fr.pilot_r[0]; out.pilotSd = fr.pilot_sd[0]; out.pilotR2 = fr.pilot_r[1]; out.pilotSd2 = fr.pilot_sd[1];
    if (fr.n) for (int b = 0; b < fr.total; b++) if (ok[b]) out.blocks.insert(out.blocks.end(), blocks.begin() + (size_t)b * bb, blocks.begin() + (size_t)(b + 1) * bb);
    done_(std::move(out));
    l.lock();
    w->state = Worker::Idle;
  }
  l.unlock();
  cpu_dec_free(dec);
}

}  // namespace lizard
