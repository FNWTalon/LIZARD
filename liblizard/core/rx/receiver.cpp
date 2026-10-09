// ai: The receiver (receiver.h): the web's GpuQueue (lizard-web/gpuqueue.mjs) over the native decoder, with the
// ai: camera's frames ingested on arrival (core/ingest/camera.h). Three threads besides the caller's:
// ai:   the camera's (push): a frame onto the ring at once (its ingest submitted on the ingest queue), or dropped;
// ai:   the releaser: each camera buffer handed back (release(tag)) once its ingest has run, so a buffer is held for
// ai:     about a millisecond, never behind a batch;
// ai:   the decoder: plans on the first frame's shape, launches as soon as a frame is staged and a lane is free, with
// ai:     every frame then staged (since 2026-10-08: nothing held back to fill a batch), two in flight, finishes them,
// ai:     keeps the held word, and hands each frame's verified blocks to the transfer.
#include "receiver.h"
#include "replay.h"
#include "front.h"
#include "camera.h"
#include "codec.h"
#include "pool.h"
#include "xfer_rx.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace lizard {

namespace {

// ai: Batches' worth staged past which a new frame is dropped (gpuqueue.mjs). A batch is never waited for (2026-10-08):
// ai: a launch goes as soon as a frame is staged and a lane is free and takes every frame staged then, so a GPU that
// ai: keeps up decodes each frame alone, its reading at the phase lock a frame after its capture, and one that falls
// ai: behind decodes what arrived meanwhile together. Waiting for 32 held a reading half a second and more and a hot
// ai: phone stalled behind it; one frame a launch dropped a quarter of the halves at 120 a second (a half alone cost
// ai: the S26 11.0 ms of GPU at the 1536 picture against 8.3 ms of arrivals).
constexpr int STAGE_BATCHES = 2;

// ai: The crops of a camera frame (ReceiverConfig.layout): 1:1 the centre square (the web's cropAtSource or
// ai: baseRect); 2:1 the centre region twice as long as it is high along the frame's long side, its two halves (each a
// ai: square, each submitted as a frame).
struct Crops { int n; uint32_t side, x[2], y[2]; };
Crops cropsOf(const CameraFrame& f, const std::string& layout) {
  Crops c{};
  const bool two = layout == "2:1", wide = f.width >= f.height;
  const uint32_t lo = std::min(f.width, f.height), hi = std::max(f.width, f.height);
  c.side = two ? std::min(lo, hi / 2) : lo;
  c.n = two ? 2 : 1;
  for (int k = 0; k < c.n; k++) {
    const uint32_t along = (hi - c.n * c.side) / 2 + k * c.side, across = (lo - c.side) / 2;
    c.x[k] = wide ? along : across;
    c.y[k] = wide ? across : along;
  }
  return c;
}

// ai: the symbol's side in camera pixels, the mean of its quad's four edges
double sideOf(const float* c) {
  double e = 0;
  for (int k = 0; k < 4; k++) { const int j = (k + 1) % 4; e += std::hypot(c[2 * j] - c[2 * k], c[2 * j + 1] - c[2 * k + 1]); }
  return e / 4;
}

// ai: Each pushed frame's camera timestamp by its tag, until its result comes back, and the newest SERIES frames as
// ai: [ms on the camera's clock, verified blocks, new blocks, 1 where the finder registered a symbol, the pilots' r and
// ai: its standard error over the even blocks, then over the odd (SPEC 7.3; null where none were read)] for the stats
// ai: ("series", about two seconds, so a reader
// ai: that asks twice a second misses none and sees a frame more than once: it keeps the newest it has counted).
// ai: Which captures read and which did not, against the camera's own clock, is how the camera's phase against the
// ai: display shows (the app's PhaseLock.kt; STATUS "A delay for the camera's phase").
constexpr size_t SERIES = 150;
struct Stamps {
  std::map<uint64_t, int64_t> ns;
  std::deque<std::array<double, 8>> series;
  void put(uint64_t tag, int64_t t) { ns[tag] = t; while (ns.size() > 512) ns.erase(ns.begin()); }
  std::array<double, 8> row(uint64_t tag, size_t blocks, int fresh, bool found, int pilotBlocks, float pilotR, float pilotSd, float pilotR2, float pilotSd2) const {
    auto it = ns.find(tag);
    return {it == ns.end() ? 0 : it->second / 1e6, (double)blocks, (double)fresh, found ? 1.0 : 0.0, pilotBlocks ? (double)pilotR : NAN, pilotBlocks ? (double)pilotSd : NAN,
            pilotBlocks ? (double)pilotR2 : NAN, pilotBlocks ? (double)pilotSd2 : NAN};
  }
  void add(const std::array<double, 8>& r) { series.push_back(r); while (series.size() > SERIES) series.pop_front(); }
  void frame(uint64_t tag, size_t blocks, int fresh, bool found, int pilotBlocks, float pilotR, float pilotSd, float pilotR2, float pilotSd2) {
    add(row(tag, blocks, fresh, found, pilotBlocks, pilotR, pilotSd, pilotR2, pilotSd2));
  }
  // ai: Receiver::series: the version, then the rows captured after sinceMs, oldest first (the series is in the order
  // ai: the results came, which on the CPU's pool is not always the captures')
  std::vector<double> since(int version, double sinceMs) const {
    std::vector<const std::array<double, 8>*> rows;
    for (auto& r : series) if (r[0] > sinceMs) rows.push_back(&r);
    std::stable_sort(rows.begin(), rows.end(), [](const std::array<double, 8>* a, const std::array<double, 8>* b) { return (*a)[0] < (*b)[0]; });
    std::vector<double> out{(double)version};
    out.reserve(1 + 8 * rows.size());
    for (auto* r : rows) out.insert(out.end(), r->begin(), r->end());
    return out;
  }
};

class GpuReceiver : public Receiver {
 public:
  GpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release) : cfg(c), releaseFn(std::move(release)) {}
  ~GpuReceiver() override;
  bool start(std::string& why);
  void push(const CameraFrame& f) override;
  std::string stats() override;
  std::vector<double> series(double sinceMs) override { std::lock_guard<std::mutex> l(mu); return stamps.since(held, sinceMs); }
  void batchCap(int n) override { cap = n; cv.notify_all(); }
  std::string file() override;
  void clear() override;
  bool wantsLuma() const override { return false; }
  void record(std::shared_ptr<Replay> r) override { std::lock_guard<std::mutex> l(mu); rec = std::move(r); }
  void cameraClosed() override {
    std::unique_lock<std::shared_mutex> planHeld(planMu);   // ai: no push inside an enqueue meanwhile
    const size_t n = ci ? ci->cameraClosed() : 0;
    std::lock_guard<std::mutex> l(mu);
    if (cfg.log) cfg.log("camera closed: " + std::to_string(n) + " imported buffers let go" + (error.empty() ? "" : ", the error \"" + error + "\"" + (error.rfind("ingest: ", 0) == 0 ? " cleared" : " kept")));
    if (error.rfind("ingest: ", 0) == 0) error.clear();
  }

 private:
  ReceiverConfig cfg;
  std::function<void(uint64_t)> releaseFn;
  std::unique_ptr<wg::Device> dev;
  std::unique_ptr<FrontHalf> fh;
  std::unique_ptr<CameraIngest> ci;
  std::unique_ptr<XferRx> xfer;
  std::string variant, error;
  std::atomic<bool> stop{false};
  std::thread decoderThread, releaserThread;
  std::mutex mu;                          // ai: guards staging, the shape, the stats window, the held word
  std::condition_variable cv;
  std::deque<Slot*> staging;
  // ai: Save replays: beside each staged frame, the replay that held it (Replay::hold) as it was staged, or none; the
  // ai: frame goes to that replay with its batch, whatever replay runs by then, so a run's end gets every frame the
  // ai: decoder took before it (Replay::finish waits for them)
  std::deque<std::shared_ptr<Replay>> stagingReplay;
  // ai: The crop the plan must cover: the largest side the camera has sent (gpuqueue.mjs needsPlan: a frame larger
  // ai: than the ring plans the lanes and the ring again for it; a smaller one goes onto the ring as it is, at its
  // ai: own size). planned: the lanes and ring cover shapeW; false from the first frame, and again when a larger
  // ai: crop arrives (the camera reopened at a higher resolution), until the decoder thread has planned. failedW: a
  // ai: side whose plan threw (out of memory): its frames are dropped without asking again, until the side changes.
  uint32_t shapeW = 0, failedW = 0;
  bool planned = false;
  // ai: push holds it shared around an enqueue and its staging; the decoder thread holds it exclusive while a plan
  // ai: clears the lanes and rebuilds the ring, so no enqueue is under way on a ring that goes
  std::shared_mutex planMu;
  int held = 0;
  int batchSize = 1;                      // ai: the batcher's size as the decoder thread last read it (push is another thread)
  std::atomic<int> cap{0};                // ai: Receiver::batchCap, set by the caller's thread and read by the decoder's; 0 for none
  // ai: camera buffers waiting for their ingest to finish before release: (timeline value, tag)
  std::mutex relMu;
  std::condition_variable relCv;
  struct Rel { uint64_t value, tag; double at; };
  std::deque<Rel> toRelease;
  // ai: the stats window (a second), and totals
  Stamps stamps;
  struct Win { int arrived = 0, dropped = 0, processed = 0, found = 0, words = 0, test = 0; long blocks = 0, fresh = 0; double gpuMs = 0, gpuFrames = 0, side = 0, held = 0, heldMax = 0; int heldN = 0; } win, last;
  double winStart = 0;
  long totalBlocks = 0, totalFrames = 0;
  json lastWord;
  std::shared_ptr<Replay> rec;   // ai: Save replays (record), under mu
  void decoderLoop();
  void releaserLoop();
  void drop(uint64_t tag) { releaseFn(tag); }
  // ai: the last window as it was: its length (a window ends at the first batch delivered past a second) and the
  // ai: frames it processed. `last.arrived` and `last.processed` are rates; a count over the window is
  // ai: a rate only over lastSecs, and a share only over lastFrames (to 2026-10-01 the goodput counted the window as
  // ai: a second, at 1024 B a KB, and read 4.2% over the page's; the registered share read 107%)
  double lastSecs = 1;
  int lastFrames = 0;
  // ai: a batch launched with its leading n frames kept for `replay`
  struct Kept { std::shared_ptr<Replay> replay; int n = 0; };
  void deliver(BatchOut& out, const Kept& k);
  void rollWindow(double now);
};

GpuReceiver::~GpuReceiver() {
  stop = true;
  cv.notify_all();
  relCv.notify_all();
  if (decoderThread.joinable()) decoderThread.join();
  if (releaserThread.joinable()) releaserThread.join();
  if (dev) dev->waitIdle();
  for (auto& r : toRelease) releaseFn(r.tag);
  for (auto& r : stagingReplay) if (r) r->release(1);
}

bool GpuReceiver::start(std::string& why) {
  auto log = cfg.log ? cfg.log : [](const std::string&) {};
  // ai: LIZ_VK_DEVICE (a substring of a device's name) picks the device on a machine with several, as the check
  // ai: tools take it; unset on a phone. Before 2026-09-30 the receiver took the first device whatever it named, so
  // ai: `receive` on the desktop ran on the 4090 under LIZ_VK_DEVICE=AMD.
  dev = wg::Device::create(!cfg.device.empty() ? cfg.device : getenv("LIZ_VK_DEVICE") ? getenv("LIZ_VK_DEVICE") : "", false, log);
  if (!cfg.cacheDir.empty()) dev->loadCache(cfg.cacheDir + "/pipeline.cache");   // ai: none, no cache (the library's default)
  // ai: The best variant the device runs of those the assets hold (2026-10-04): an install may hold some of the six
  // ai: (variantsIn). Before, the device's best was asked for whatever the assets held: an install of f32-sg alone
  // ai: asked for int8-sg's setup on the 4090 and lavapipe, which it lacks, so gpu failed and auto decoded on the C.
  const auto runs = variantsFor(*dev);
  auto variants = variantsIn(cfg.assets, runs);
  // ai: the device's best where the assets lack it and auto settles for less (said in the log)
  const std::string lacking = cfg.precision == "auto" && !variants.empty() && variants[0] != runs[0] ? runs[0] : "";
  if (cfg.precision != "auto") {
    std::vector<std::string> keep;
    for (auto& v : variants) if (v.rfind(cfg.precision, 0) == 0) keep.push_back(v);
    variants = keep;
  }
  if (variants.empty()) {
    std::string list;
    for (auto& v : runs) list += (list.empty() ? "" : " ") + v;
    why = runs.empty() ? "no GPU variant on " + dev->name
                       : cfg.assets + " holds no " + (cfg.precision != "auto" ? cfg.precision + " " : "") + "variant " + dev->name + " runs (" + list + ")";
    return false;
  }
  variant = variants[0];
  // ai: the web's budget on the S26 (Chrome's maxBufferSize, 2 GiB): the lanes, the ring and the tables within it
  const double budget = std::min<double>(2.0 * (1ull << 30), (double)dev->limits.maxBufferSize);
  fh = FrontHalf::create(*dev, readDir(cfg.assets), variant, budget, 0, log);
  ci = CameraIngest::create(*dev, readDir(cfg.assets));
  fh->ingest = ci.get();
  xfer = std::make_unique<XferRx>(cfg.storeDir, log);
  dev->saveCache();
  winStart = fh->now();
  batchSize = std::max(1, fh->size());
  decoderThread = std::thread([this] { decoderLoop(); });
  releaserThread = std::thread([this] { releaserLoop(); });
  log("receiver: the GPU decoder, " + variant + (lacking.empty() ? "" : " (" + lacking + " not in the assets)") + (ci ? ", zero-copy camera" : ", camera luma uploaded"));
  return true;
}

// ai: gpuqueue.mjs push, place and stage: the shape planned on the decoder thread, a frame
// ai: dropped before it touches the device once STAGE_BATCHES batches' worth are staged, else onto the ring.
void GpuReceiver::push(const CameraFrame& f) {
  const double now = fh->now();
  const Crops cr = cropsOf(f, cfg.layout);
  const uint32_t side = cr.side;
  const int crops = cr.n;
  const uint32_t* cx = cr.x;
  const uint32_t* cy = cr.y;
  // ai: a frame dropped is handed back after the lock is let go (2026-10-03, for the library: a binding's release
  // ai: handler may call back into the receiver)
  bool dropped = false;
  {
    std::lock_guard<std::mutex> l(mu);
    win.arrived++;
    stamps.put(f.tag, f.timestampNs);
    if (side == failedW) dropped = true;
    else {
      if (side > shapeW) { shapeW = side; planned = false; failedW = 0; }
      if (!planned) {
        // ai: no ring covers this crop yet (the first frame, or a larger one than the plan's): the decoder thread
        // ai: plans on shapeW, and frames meanwhile are dropped, never held (the web holds HOLD_MAX; the camera's pool
        // ai: here is the reader's images, handed back at once)
        cv.notify_all();
        dropped = true;
      } else if ((int)staging.size() + crops > STAGE_BATCHES * batchSize) dropped = true;
    }
    if (dropped) win.dropped++;
  }
  if (dropped) { drop(f.tag); return; }
  std::shared_lock<std::shared_mutex> planHeld(planMu);
  Slot* got[2] = {nullptr, nullptr};
  uint64_t ready = 0;
  for (int k = 0; k < crops; k++) {
    try {
      if (f.hb && ci) got[k] = fh->enqueueCamera(*ci, f.hb, cx[k], cy[k], side, side, f.tag);
      else if (f.luma) got[k] = fh->enqueueLuma(f.luma + (size_t)cy[k] * f.stride + cx[k], side, side, f.stride, f.tag);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(mu);
      error = std::string("ingest: ") + e.what();
    }
    if (got[k]) ready = std::max(ready, got[k]->ready);
  }
  if (!got[0] && !got[1]) { { std::lock_guard<std::mutex> l(mu); win.dropped++; } drop(f.tag); return; }
  // ai: the camera's buffer back once the last of its ingests has run (they complete in order on their queue); a
  // ai: luma frame was copied into the slots' staging, so its buffer is free at once
  if (ready) { std::lock_guard<std::mutex> l(relMu); toRelease.push_back({ready, f.tag, now}); relCv.notify_one(); }
  else drop(f.tag);
  std::lock_guard<std::mutex> l(mu);
  for (auto* s : got) if (s) { staging.push_back(s); stagingReplay.push_back(rec && rec->hold(1) ? rec : nullptr); }
  cv.notify_all();
}

void GpuReceiver::releaserLoop() {
  while (!stop) {
    Rel next;
    {
      std::unique_lock<std::mutex> l(relMu);
      relCv.wait_for(l, std::chrono::milliseconds(50), [&] { return stop || !toRelease.empty(); });
      if (toRelease.empty()) continue;
      next = toRelease.front();
    }
    // ai: the oldest ingest's value: ingests complete in order on their queue
    while (!stop && !ci->done(next.value)) std::this_thread::sleep_for(std::chrono::microseconds(300));
    {
      std::lock_guard<std::mutex> l(relMu);
      toRelease.pop_front();
    }
    releaseFn(next.tag);
    // ai: how long the camera's buffer was held (the app's reader has 32: past 32 frames' time the camera drops)
    const double heldMs = fh->now() - next.at;
    std::lock_guard<std::mutex> l(mu);
    win.held += heldMs; win.heldN++; win.heldMax = std::max(win.heldMax, heldMs);
  }
}

// ai: The plan on the first shape, then a launch whenever a frame is staged while fewer than inflight compute and a
// ai: lane is free, taking every frame staged then up to the batcher's size or the cap (gpuqueue.mjs maybeLaunch's
// ai: wait for a full batch went 2026-10-08), and the oldest finished in order.
void GpuReceiver::decoderLoop() {
  std::deque<std::unique_ptr<InFlight>> flight;
  std::deque<Kept> flightKept;   // ai: beside each batch in flight, the frames it keeps and for which replay
  // ai: The oldest batch read back and delivered: taken off both deques first, so a finish that throws leaves nothing
  // ai: behind (before 2026-10-03 an emptied entry left at the front was dereferenced by the next loop), and the
  // ai: frames it kept let go.
  auto finishFront = [&] {
    auto f = std::move(flight.front());
    Kept k = std::move(flightKept.front());
    flight.pop_front();
    flightKept.pop_front();
    BatchOut out;
    try { out = fh->finish(std::move(f)); }
    catch (...) { if (k.replay) k.replay->release(k.n); throw; }
    deliver(out, k);
  };
  while (!stop) {
    uint32_t want = 0;
    try {
      {
        std::lock_guard<std::mutex> l(mu);
        if (!planned && shapeW) want = shapeW;
      }
      if (want) {
        // ai: A plan clears the lanes and the ring (front.cpp plan: grown to the largest side seen), so first the
        // ai: batches in flight finish on their lanes and the frames staged on the old ring are let go (their camera
        // ai: buffers went back once their ingests ran), with no push inside an enqueue meanwhile (planMu). The
        // ai: first plan and a re-plan for a larger crop are one path. A plan that fails leaves the receiver with no
        // ai: lanes: its side is remembered (failedW) and a smaller crop later plans again.
        std::unique_lock<std::shared_mutex> planHeld(planMu);
        while (!flight.empty()) {
          flight.front()->ticket->wait();
          finishFront();
        }
        std::vector<Slot*> old;
        std::vector<std::shared_ptr<Replay>> oldReplay;
        {
          std::lock_guard<std::mutex> l(mu);
          old.assign(staging.begin(), staging.end()); oldReplay.assign(stagingReplay.begin(), stagingReplay.end());
          staging.clear(); stagingReplay.clear();
        }
        for (auto* s : old) fh->release(s);
        for (auto& r : oldReplay) if (r) r->release(1);
        fh->plan(want, want);
        std::lock_guard<std::mutex> l(mu);
        planned = true;
        error.clear();
        if (cfg.log) cfg.log("plan: " + std::to_string(want) + " x " + std::to_string(want) + ", batches of " + std::to_string(fh->size()));
        want = 0;
      }
      std::vector<Slot*> batch;
      std::vector<std::shared_ptr<Replay>> batchReplay;
      int heldNow = 0;
      std::shared_ptr<Replay> recNow;
      {
        std::unique_lock<std::mutex> l(mu);
        // ai: a batch goes as soon as a frame is staged and a lane is free, with every frame staged then, up to the
        // ai: batcher's size or the cap where one is set (Receiver::batchCap; STAGE_BATCHES above)
        batchSize = std::max(1, fh->size());
        const int c = cap.load();
        const bool room = (int)flight.size() < fh->inflightMax() && fh->freeLane();
        if (room && !staging.empty()) {
          const int n = std::min<int>(c > 0 ? std::min(batchSize, c) : batchSize, (int)staging.size());
          batch.assign(staging.begin(), staging.begin() + n);
          batchReplay.assign(stagingReplay.begin(), stagingReplay.begin() + n);
          staging.erase(staging.begin(), staging.begin() + n);
          stagingReplay.erase(stagingReplay.begin(), stagingReplay.begin() + n);
        }
        heldNow = held;
        recNow = rec;
        if (batch.empty() && flight.empty()) cv.wait_for(l, std::chrono::milliseconds(5));
      }
      if (!batch.empty()) {
        // ai: Save replays: the batch's leading frames held for one replay are read off the ring with it and go to
        // ai: that replay as the batch delivers; frames held for another (a run begun mid-batch) are let go. A batch
        // ai: that cannot be recorded gives its ring slots back (released twice, a slot is simply free).
        Kept k{batchReplay[0], 0};
        if (k.replay) while (k.n < (int)batchReplay.size() && batchReplay[k.n] == k.replay) k.n++;
        for (size_t i = k.n; i < batchReplay.size(); i++) if (batchReplay[i]) batchReplay[i]->release(1);
        try {
          flight.push_back(fh->run(batch, heldNow, false, k.n));
        } catch (...) {
          for (auto* s : batch) fh->release(s);
          if (k.replay) k.replay->release(k.n);
          throw;
        }
        flightKept.push_back(std::move(k));
      } else if (flight.empty() && !recNow) fh->dropKeep();
      if (!flight.empty() && (flight.front()->ticket->done() || (int)flight.size() >= fh->inflightMax() || batch.empty())) {
        if (!flight.front()->ticket->done()) { std::this_thread::sleep_for(std::chrono::microseconds(500)); continue; }
        finishFront();
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(mu);
      error = (want ? "plan " + std::to_string(want) + " x " + std::to_string(want) + ": " : std::string()) + e.what();
      if (want) { failedW = want; shapeW = 0; planned = false; }
      if (cfg.log) cfg.log(std::string(want ? "plan: " : "decoder: ") + e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  for (auto& f : flight) f->ticket->wait();
  for (auto& k : flightKept) if (k.replay) k.replay->release(k.n);
}

// ai: a batch's frames: the held word from each frame's own word in capture order (a frame whose word does not read
// ai: leaves it as is), the verified blocks to the transfer frame by frame (its judge and dedupe), the window's counts.
void GpuReceiver::deliver(BatchOut& out, const Kept& k) {
  int index = 0;
  for (auto& fo : out.frames) {
    // ai: Save replays: a kept frame as its batch read it off the ring, to the replay that held it, the batches in the
    // ai: order they went; one that did not come back (its readback refused) let go
    if (index++ < k.n) {
      if (fo.luma.empty()) k.replay->release(1);
      else {
        double ms = 0;
        {
          std::lock_guard<std::mutex> l(mu);
          auto it = stamps.ns.find(fo.tag);
          if (it != stamps.ns.end()) ms = it->second / 1e6;
        }
        k.replay->frame(fo.luma.data(), fo.w, fo.h, fo.w, ms, true);
      }
    }
    if (fo.empty) continue;
    std::vector<Block> blocks;
    for (auto& r : fo.records) { Block b; memcpy(b.bytes.data(), r.payload.data(), b.bytes.size()); blocks.push_back(b); }
    auto v = xfer->frame(blocks);
    std::lock_guard<std::mutex> l(mu);
    win.processed++;
    win.found += fo.finder.found == 1;
    if (fo.finder.found == 1) win.side += sideOf(fo.finder.corners.data());
    win.blocks += (long)fo.records.size();
    win.fresh += v.fresh;
    stamps.frame(fo.tag, fo.records.size(), v.fresh, fo.finder.found == 1, fo.pilotBlocks, fo.pilotR, fo.pilotSd, fo.pilotR2, fo.pilotSd2);
    if (v.test) win.test++;
    totalBlocks += (long)fo.records.size();
    totalFrames++;
    if (fo.hasWord) { win.words++; held = fo.word.version; lastWord = {{"version", fo.word.version}, {"fps", fo.word.fps}, {"ring", fo.ring}}; }
  }
  std::lock_guard<std::mutex> l(mu);
  if (out.gpuMs) { win.gpuMs += *out.gpuMs; win.gpuFrames += out.carried; }
  // ai: a batch delivered is the decoder working: an error from before (an ingest a plan has since covered, a batch
  // ai: that failed once) no longer describes it. One that keeps failing sets it again before the next delivery.
  error.clear();
  rollWindow(fh->now());
}

void GpuReceiver::rollWindow(double now) {
  if (now - winStart < 1000) return;
  last = win;
  const double dt = (now - winStart) / 1000;
  lastSecs = dt; lastFrames = win.processed;
  last.arrived = (int)std::lround(win.arrived / dt); last.processed = (int)std::lround(win.processed / dt);
  win = Win{};
  winStart = now;
}

std::string GpuReceiver::stats() {
  auto p = xfer->progress();
  std::lock_guard<std::mutex> l(mu);
  const double n = std::max(1, lastFrames);
  json j = {
    // ai: the test stream in the last window's frames comes first (2026-10-05): what the camera reads now is the state, over
    // ai: a file received or in progress, which comes back once the camera is on its frames again
    {"state", !error.empty() ? "error" : !planned ? "starting" : last.test ? "test" : p.done ? "received" : p.live ? "receiving" : lastWord.is_null() ? "looking" : "found"},
    {"error", error}, {"decoder", "gpu " + variant}, {"zeroCopy", !!ci}, {"layout", cfg.layout},
    {"capturedFps", last.arrived}, {"processedFps", last.processed}, {"dropped", last.dropped},
    {"foundShare", last.found / n}, {"side", last.found ? last.side / last.found : 0},
    {"heldMs", last.heldN ? last.held / last.heldN : 0}, {"heldMaxMs", last.heldMax}, {"blocks", last.blocks}, {"windowSecs", lastSecs}, {"goodputKBs", last.fresh * 469 / 1000.0 / lastSecs},
    {"gpuMs", last.gpuFrames ? last.gpuMs / last.gpuFrames : 0}, {"B", fh ? batchSize : 0}, {"cap", cap.load()},
    {"bandVersion", lastWord.is_null() ? json() : lastWord["version"]}, {"word", lastWord},
    {"totals", {{"frames", totalFrames}, {"blocks", totalBlocks}}}, {"series", stamps.series},
    // ai: a file only once a header named one: the page reads a file's presence as a transfer under way; type, the
    // ai: header's media type, is what the app opens and shares the file as (2026-10-01)
    {"file", p.header ? json{{"name", p.name}, {"size", p.length}, {"received", p.bytesIn}, {"sent", p.sent}, {"sentIn", p.sentIn}, {"fraction", p.fraction}, {"verified", p.done}, {"chunks", p.chunks}, {"chunksVerified", p.verified}, {"root", p.root}, {"type", p.type}} : json()},
    {"test", last.test > 0},
  };
  return j.dump();
}

std::string GpuReceiver::file() { auto p = xfer->progress(); return p.done ? p.path : ""; }
void GpuReceiver::clear() { xfer->clear(); std::lock_guard<std::mutex> l(mu); lastWord = nullptr; held = 0; }

// ai: The C on the CPU (cpu/pool.h: the web's worker pool natively): a frame's crop is copied to an idle worker or the
// ai: frame is lost, never queued; each worker decodes blind on its own decoder; the held word is the last word any
// ai: frame read; a frame's verified blocks go into the transfer one frame at a time. The camera's buffer is handed
// ai: back inside push, once the crop is copied.
constexpr int CPU_NMAX = 1536;   // ai: every picture of the ladder (sim/phy.mjs BLIND_NMAX)

class CpuReceiver : public Receiver {
 public:
  CpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release, std::string gpuWhy);
  // ai: The workers stop before the transfer they feed goes, and while `pool` still points at the pool: a frame in a
  // ai: worker's hand ends in done(), whose rollWindow ticks the pool (a reset alone cleared the pointer first, a race
  // ai: ThreadSanitizer found through the library, 2026-10-03).
  ~CpuReceiver() override { if (pool) pool->stop(); pool.reset(); }
  void push(const CameraFrame& f) override;
  std::string stats() override;
  std::vector<double> series(double sinceMs) override { std::lock_guard<std::mutex> l(mu); return stamps.since(held.load(), sinceMs); }
  void batchCap(int) override {}   // ai: a frame at a time already
  std::string file() override { auto p = xfer->progress(); return p.done ? p.path : ""; }
  void clear() override { xfer->clear(); std::lock_guard<std::mutex> l(mu); lastWord = nullptr; held = 0; }
  bool wantsLuma() const override { return true; }
  void record(std::shared_ptr<Replay> r) override { std::lock_guard<std::mutex> l(mu); rec = std::move(r); }

 private:
  ReceiverConfig cfg;
  std::function<void(uint64_t)> releaseFn;
  std::string gpuWhy, error;
  std::unique_ptr<XferRx> xfer;
  std::unique_ptr<CpuPool> pool;
  std::mutex mu;           // ai: guards the stats window, the word and the error
  std::mutex deliverMu;    // ai: one frame's blocks into the transfer at a time (XferRx takes one producer)
  // ai: The series in capture order. The pool's workers finish out of it (a capture that holds a change runs its
  // ai: LDPC to the cap while the next one is done), and a reader that asks at every capture (the phase lock since
  // ai: 2026-10-01) would take the later frame and never see the earlier. `flight`: the tags offered and not yet
  // ai: in the series, in the order offered; `ready`: results that came before an earlier frame's. mu guards both.
  std::deque<uint64_t> flight;
  std::multimap<uint64_t, std::array<double, 8>> ready;
  void flushRows() {
    // ai: (a frame the pool took and never answered would hold every later row back: past 64 waiting, it is let go)
    while (flight.size() > 64) flight.pop_front();
    while (!flight.empty()) {
      auto it = ready.find(flight.front());
      if (it == ready.end()) break;
      stamps.add(it->second);
      ready.erase(it);
      flight.pop_front();
    }
    if (flight.empty()) { for (auto& r : ready) stamps.add(r.second); ready.clear(); }
  }
  std::atomic<int> held{0};
  // ai: the stats window (a second), and totals; repeat: frames that decoded blocks, none of them new; skips: frames
  // ai: lost to busy workers
  Stamps stamps;
  struct Win { int arrived = 0, dropped = 0, processed = 0, found = 0, words = 0, test = 0, repeat = 0, skips = 0; long blocks = 0, fresh = 0; double ms = 0, side = 0; } win, last;
  double winStart = 0, repeatShare = 0;
  int lastMsFrames = 0;    // ai: the frames last.ms is summed over (last.processed is a rate)
  double lastSecs = 1;     // ai: the last window's length: what a count over it is a rate of (GpuReceiver's note)
  long totalBlocks = 0, totalFrames = 0;
  json lastWord;
  std::shared_ptr<Replay> rec;   // ai: Save replays (record), under mu
  static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  void done(CpuFrameOut&& out);
  void rollWindow(double t);
};

CpuReceiver::CpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release, std::string why)
    : cfg(c), releaseFn(std::move(release)), gpuWhy(std::move(why)) {
  auto log = cfg.log ? cfg.log : [](const std::string&) {};
  xfer = std::make_unique<XferRx>(cfg.storeDir, log);
  winStart = now();
  pool = std::make_unique<CpuPool>(CPU_NMAX, 0, cfg.cpuThreads, [this](CpuFrameOut&& o) { done(std::move(o)); }, log);
  log(std::string("receiver: the C on the CPU, ") + (cpu_simd() ? "vector" : "scalar") + " paths, up to " + std::to_string(pool->ceiling()) + " threads" +
      (gpuWhy.empty() ? "" : " (no GPU decoder: " + gpuWhy + ")"));
}

void CpuReceiver::push(const CameraFrame& f) {
  const double t = now();
  int lost = 0;
  std::shared_ptr<Replay> r;
  { std::lock_guard<std::mutex> l(mu); stamps.put(f.tag, f.timestampNs); r = rec; }
  if (!f.luma) {
    std::lock_guard<std::mutex> l(mu);
    error = "the CPU decoder reads a luma plane, and the camera's frame came without one";
    win.arrived++; win.dropped++;
  } else {
    const Crops c = cropsOf(f, cfg.layout);
    for (int k = 0; k < c.n; k++) {
      { std::lock_guard<std::mutex> l(mu); flight.push_back(f.tag); }
      if (pool->offer(f.luma, f.stride, c.x[k], c.y[k], c.side, c.side, f.tag, held.load(), t)) {
        // ai: Save replays: the crop a worker took, as it took it
        if (r) r->frame(f.luma + (size_t)c.y[k] * f.stride + c.x[k], c.side, c.side, f.stride, f.timestampNs / 1e6);
        continue;
      }
      lost++;
      // ai: not taken: its place in the order goes (the newest of that tag: this thread put it there)
      std::lock_guard<std::mutex> l(mu);
      for (auto it = flight.rbegin(); it != flight.rend(); ++it) if (*it == f.tag) { flight.erase(std::next(it).base()); break; }
      flushRows();
    }
  }
  releaseFn(f.tag);
  double share;
  {
    std::lock_guard<std::mutex> l(mu);
    if (f.luma) { win.arrived++; win.dropped += lost; win.skips += lost; }
    share = repeatShare;
    rollWindow(t);
  }
  for (int k = 0; k < lost; k++) pool->lost(share);
}

void CpuReceiver::done(CpuFrameOut&& out) {
  std::vector<Block> blocks(out.blockBytes ? out.blocks.size() / out.blockBytes : 0);
  for (size_t b = 0; b < blocks.size(); b++) memcpy(blocks[b].bytes.data(), out.blocks.data() + b * out.blockBytes, blocks[b].bytes.size());
  FrameVerdict v;
  {
    std::lock_guard<std::mutex> l(deliverMu);
    v = xfer->frame(blocks);
  }
  std::lock_guard<std::mutex> l(mu);
  win.processed++;
  win.found += out.found;
  if (out.found) win.side += sideOf(out.quad);
  win.blocks += (long)blocks.size();
  win.fresh += v.fresh;
  ready.emplace(out.tag, stamps.row(out.tag, blocks.size(), v.fresh, out.found, out.pilotBlocks, out.pilotR, out.pilotSd, out.pilotR2, out.pilotSd2));
  flushRows();
  win.ms += out.ms;
  if (v.test) win.test++;
  if (!v.fresh && v.seen) win.repeat++;
  totalBlocks += (long)blocks.size();
  totalFrames++;
  if (out.hasWord) { win.words++; held = out.version; lastWord = {{"version", out.version}, {"fps", out.fps}, {"ring", out.ring}}; }
  rollWindow(now());
}

// ai: mu held. A second's window closed: its rates for the page, its repeat share and lost frames for the pool.
void CpuReceiver::rollWindow(double t) {
  if (t - winStart < 1000) return;
  const double dt = (t - winStart) / 1000;
  last = win;
  lastMsFrames = win.processed; lastSecs = dt;
  last.arrived = (int)std::lround(win.arrived / dt); last.processed = (int)std::lround(win.processed / dt);
  repeatShare = win.processed ? (double)win.repeat / win.processed : 0;
  const int skips = win.skips;
  win = Win{};
  winStart = t;
  pool->tick(skips);
}

std::string CpuReceiver::stats() {
  auto p = xfer->progress();
  std::lock_guard<std::mutex> l(mu);
  const double n = std::max(1, lastMsFrames);
  json j = {
    {"state", !error.empty() ? "error" : last.test ? "test" : p.done ? "received" : p.live ? "receiving" : lastWord.is_null() ? "looking" : "found"},
    {"error", error}, {"decoder", "cpu"}, {"zeroCopy", false}, {"layout", cfg.layout},
    {"threads", pool->size()}, {"threadsReady", pool->ready()}, {"threadsMax", pool->ceiling()}, {"simd", cpu_simd() != 0}, {"gpuWhy", gpuWhy},
    {"capturedFps", last.arrived}, {"processedFps", last.processed}, {"dropped", last.dropped},
    {"foundShare", last.found / n}, {"side", last.found ? last.side / last.found : 0},
    {"blocks", last.blocks}, {"windowSecs", lastSecs}, {"goodputKBs", last.fresh * 469 / 1000.0 / lastSecs}, {"repeatShare", repeatShare},
    {"cpuMs", lastMsFrames ? last.ms / lastMsFrames : 0}, {"gpuMs", 0}, {"B", 0},
    {"bandVersion", lastWord.is_null() ? json() : lastWord["version"]}, {"word", lastWord},
    {"totals", {{"frames", totalFrames}, {"blocks", totalBlocks}}}, {"series", stamps.series},
    {"file", p.header ? json{{"name", p.name}, {"size", p.length}, {"received", p.bytesIn}, {"sent", p.sent}, {"sentIn", p.sentIn}, {"fraction", p.fraction}, {"verified", p.done}, {"chunks", p.chunks}, {"chunksVerified", p.verified}, {"root", p.root}, {"type", p.type}} : json()},
    {"test", last.test > 0},
  };
  return j.dump();
}

}  // namespace

// ai: cpu: the C. gpu: the GPU decoder, or an error. auto: the GPU decoder where the device runs a variant the assets
// ai: hold, else the C, which every device runs.
std::unique_ptr<Receiver> Receiver::create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release) {
  std::string why;
  if (c.decoder != "cpu") {
    try {
      auto r = std::make_unique<GpuReceiver>(c, release);
      if (r->start(why)) return r;
    } catch (const std::exception& e) { why = e.what(); }
    if (c.decoder == "gpu") throw std::runtime_error(why);
  }
  return std::make_unique<CpuReceiver>(c, std::move(release), why);
}

}  // namespace lizard
