// ai: The receiver (receiver.h): a queue of camera frames over the native decoder, the frames ingested on arrival
// ai: (core/ingest/camera.h). Three threads besides the caller's:
// ai:   the camera's (push): a frame onto the ring at once (its ingest submitted on the ingest queue), or dropped;
// ai:   the releaser: each camera buffer handed back (release(tag)) once its ingest has run, so a buffer is held for
// ai:     about a millisecond, never behind a batch;
// ai:   the decoder: plans on the first frame's shape, launches as soon as a frame is staged and a lane is free, with
// ai:     every frame then staged (since 2026-10-08: nothing held back to fill a batch), two in flight, finishes them,
// ai:     keeps the held word, and hands each frame's verified blocks to the transfer.
#include "receiver.h"
#include "develop.h"
#include "crops.h"
#include "stream.h"
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

CameraFrame developedOnCpu(const CameraFrame& f, std::vector<uint8_t>& luma) {
  if (!f.raw) return f;
  luma.resize((size_t)f.width * f.height);
  developLuma(*f.raw, 0, 0, f.width, f.height, luma.data(), f.width);
  CameraFrame g = f;
  g.raw = nullptr; g.hb = nullptr;
  g.luma = luma.data(); g.stride = f.width;
  return g;
}

namespace {

// ai: Batches' worth staged past which a new frame is dropped. A batch is never waited for (2026-10-08):
// ai: a launch goes as soon as a frame is staged and a lane is free and takes every frame staged then, so a GPU that
// ai: keeps up decodes each frame alone, its reading at the phase lock a frame after its capture, and one that falls
// ai: behind decodes what arrived meanwhile together. Waiting for 32 held a reading half a second and more and a hot
// ai: phone stalled behind it; one frame a launch dropped a quarter of the halves at 120 a second (a half alone cost
// ai: the S26 11.0 ms of GPU at the 1536 picture against 8.3 ms of arrivals).
constexpr int STAGE_BATCHES = 2;

// ai: Where the camera's optical axis falls in crop k: the frame's centre less the crop's corner (the lens's own
// ai: centre is the array's within a few pixels on the S26's calibration). A 2:1 half's is the middle of its inner
// ai: edge, a 1:1 square's its centre; both decoders fit the lens about it (liblizard/src/acquire.c, register.mjs LENS).
inline void axisOf(const CameraFrame& f, const Crops& c, int k, float* a) { a[0] = 0.5f * f.width - c.x[k]; a[1] = 0.5f * f.height - c.y[k]; }
Crops cropsOf(const CameraFrame& f, const std::string& layout) { return lizard::cropsOf(f.width, f.height, layout == "2:1"); }

// ai: the symbol's side in camera pixels, the mean of its quad's four edges
double sideOf(const float* c) {
  double e = 0;
  for (int k = 0; k < 4; k++) { const int j = (k + 1) % 4; e += std::hypot(c[2 * j] - c[2 * k], c[2 * j + 1] - c[2 * k + 1]); }
  return e / 4;
}

// ai: Each pushed frame's camera timestamp and stream (the receiver's count of cameraClosed calls when it came) by its
// ai: tag, until its result comes back, and the newest SERIES frames as
// ai: [ms on the camera's clock, verified blocks, new blocks, 1 where the finder registered a symbol, the pilots' r and
// ai: its standard error over the even blocks, then over the odd (SPEC 7.3; null where none were read)] for the stats
// ai: ("series", about two seconds, so a reader
// ai: that asks twice a second misses none and sees a frame more than once: it keeps the newest it has counted).
// ai: Which captures read and which did not, against the camera's own clock, is how the camera's phase against the
// ai: display shows (the app's PhaseLock.kt; STATUS "A delay for the camera's phase").
constexpr size_t SERIES = 150;
struct Stamps {
  struct At { int64_t ns; uint64_t stream; };
  std::map<uint64_t, At> ns;
  std::deque<std::array<double, 8>> series;
  void put(uint64_t tag, int64_t t, uint64_t stream) { ns[tag] = {t, stream}; while (ns.size() > 512) ns.erase(ns.begin()); }
  // ai: whether the frame tagged `tag` came in stream s (one no longer kept, 512 pushes on, is taken for an earlier
  // ai: stream's: no frame waits in either decoder that long)
  bool in(uint64_t tag, uint64_t s) const { auto it = ns.find(tag); return it != ns.end() && it->second.stream == s; }
  std::array<double, 8> row(uint64_t tag, size_t blocks, int fresh, bool found, int pilotBlocks, float pilotR, float pilotSd, float pilotR2, float pilotSd2) const {
    auto it = ns.find(tag);
    return {it == ns.end() ? 0 : it->second.ns / 1e6, (double)blocks, (double)fresh, found ? 1.0 : 0.0, pilotBlocks ? (double)pilotR : NAN, pilotBlocks ? (double)pilotSd : NAN,
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
  // ai: transfer: another receiver's to go on with (Receiver::create's from), null for one of its own (made in start)
  GpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release, std::shared_ptr<XferRx> transfer)
      : cfg(c), releaseFn(std::move(release)), xfer(std::move(transfer)) {}
  ~GpuReceiver() override;
  // ai: software: whether a device of type CPU may run it (Receiver::create: gpu's, not auto's)
  bool start(std::string& why, bool software);
  void push(const CameraFrame& f) override;
  std::string stats() override;
  std::vector<double> series(double sinceMs) override { std::lock_guard<std::mutex> l(mu); return stamps.since(held, sinceMs); }
  void batchCap(int n) override { cap = n; cv.notify_all(); }
  std::string file() override;
  void clear() override;
  void forget(const std::string& root) override;
  // ai: the camera's buffer is read in place only through the zero-copy ingest; a device without it (camera.h) takes
  // ai: the frame's luma plane, uploaded (enqueueLuma)
  bool wantsLuma() const override { return !ci; }
  void record(std::shared_ptr<Replay> r) override { std::lock_guard<std::mutex> l(mu); rec = std::move(r); }
  void cameraClosed() override {
    std::unique_lock<std::shared_mutex> planHeld(planMu);   // ai: no push inside an enqueue meanwhile
    const size_t n = ci ? ci->cameraClosed() : 0;
    std::lock_guard<std::mutex> l(mu);
    if (cfg.log) cfg.log("camera closed: " + std::to_string(n) + " imported buffers let go" + (error.empty() ? "" : ", the error \"" + error + "\"" + (error.rfind("ingest: ", 0) == 0 ? " cleared" : " kept")));
    if (error.rfind("ingest: ", 0) == 0) error.clear();
    endWindow();
  }

 private:
  std::shared_ptr<XferRx> transfer() const override { return xfer; }
  ReceiverConfig cfg;
  std::function<void(uint64_t)> releaseFn;
  std::unique_ptr<wg::Device> dev;
  std::unique_ptr<FrontHalf> fh;
  std::unique_ptr<CameraIngest> ci;
  std::shared_ptr<XferRx> xfer;
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
  // ai: The crop the plan must cover: the largest side the camera has sent (a frame larger than the ring plans the
  // ai: lanes and the ring again for it; a smaller one goes onto the ring as it is, at its own size). planned: the lanes and ring cover shapeW; false from the first frame, and again when a larger
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
  float lensNow[4] = {0, 0, 0, 0};   // ai: the newest frame's lens report (CameraFrame::lens), for the replay's GPU-kept frames; under mu
  // ai: fresh: data ids new to the dedupe (the series' and the lock's); useful: what goodputKBs counts (rateFold)
  // ai: bx0..by1: the box of the codes found in the window, camera pixels from the frame's centre (the app's focus region)
  // ai: anchors: the anchors each frame the GPU's finder found read
  struct Win { int arrived = 0, dropped = 0, processed = 0, found = 0, words = 0, test = 0; long blocks = 0, fresh = 0, useful = 0; double gpuMs = 0, gpuFrames = 0, side = 0, held = 0, heldMax = 0; int heldN = 0; double bx0 = 1e30, by0 = 1e30, bx1 = -1e30, by1 = -1e30; std::vector<int> anchors; } win, last;
  uint64_t takenSeen = 0;   // ai: the transfer's taken() at the last window's close, or at its opening
  double winStart = 0;
  // ai: the window's stream (receiver.h cameraClosed): stream counts the camera's closes, and a frame's figures go to
  // ai: the window of the stream it came in (Stamps); opened: a window runs, from a stream's first frame (openWindow,
  // ai: in push) to the camera's close (endWindow), so none before a receiver's first frame or between two streams
  uint64_t stream = 0;
  bool opened = false;
  long totalBlocks = 0, totalFrames = 0;
  json lastWord;
  std::shared_ptr<Replay> rec;   // ai: Save replays (record), under mu
  // ai: the decoder thread's frames and clear(), which the app calls from another thread, go into the transfer one at
  // ai: a time under its own lock (xfer_rx.h; 2026-10-10 a lock here: the dedupe was cleared unlocked while a batch
  // ai: was delivered)
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
  void openWindow(double now);
  void endWindow();
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

bool GpuReceiver::start(std::string& why, bool software) {
  auto log = cfg.log ? cfg.log : [](const std::string&) {};
  // ai: LIZ_VK_DEVICE (a substring of a device's name) picks the device on a machine with several, as the check
  // ai: tools take it; unset on a phone. Before 2026-09-30 the receiver took the first device whatever it named, so
  // ai: `receive` on the desktop ran on the 4090 under LIZ_VK_DEVICE=AMD.
  wg::DeviceExtras ex;
  ex.lowPriority = cfg.yieldGpu;
  dev = wg::Device::create(!cfg.device.empty() ? cfg.device : getenv("LIZ_VK_DEVICE") ? getenv("LIZ_VK_DEVICE") : "", false, log, ex);
  // ai: A device of type CPU (lavapipe, SwiftShader) runs the kernels as threads on the cores the C decodes on, slower
  // ai: by far: auto takes the C there, before any pipeline is built (2026-10-25: the PC receiver's auto on a machine
  // ai: whose one Vulkan device is llvmpipe, the Windows test VM's, took 11.6 s for a batch of one frame and the screen
  // ai: read nothing). gpu still runs on it, as the checks ask.
  if (!software && dev->type == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    why = "a software Vulkan device: " + dev->name;
    return false;
  }
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
  fh = FrontHalf::create(*dev, readDir(cfg.assets), variant, budget, log);
  ci = CameraIngest::create(*dev, readDir(cfg.assets));
  fh->ingest = ci.get();
  if (!xfer) xfer = std::make_shared<XferRx>(cfg.storeDir, log);
  dev->saveCache();
  batchSize = std::max(1, fh->size());
  decoderThread = std::thread([this] { decoderLoop(); });
  releaserThread = std::thread([this] { releaserLoop(); });
  log("receiver: the GPU decoder, " + variant + (lacking.empty() ? "" : " (" + lacking + " not in the assets)") + (ci ? ", zero-copy camera" : ", camera luma uploaded"));
  return true;
}

// ai: A frame staged: the shape planned on the decoder thread, a frame dropped before it touches the device once
// ai: STAGE_BATCHES batches' worth are staged, else onto the ring.
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
    if (!opened) openWindow(now);
    win.arrived++;
    stamps.put(f.tag, f.timestampNs, stream);
    for (int i = 0; i < 4; i++) lensNow[i] = f.lens[i];
    if (side == failedW) dropped = true;
    else {
      if (side > shapeW) { shapeW = side; planned = false; failedW = 0; }
      if (!planned) {
        // ai: no ring covers this crop yet (the first frame, or a larger one than the plan's): the decoder thread
        // ai: plans on shapeW, and frames meanwhile are dropped, never held (the camera's pool is the reader's images,
        // ai: handed back at once)
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
      if (f.raw) got[k] = fh->enqueueRaw(*f.raw, cx[k], cy[k], side, side, f.tag);
      else if (f.hb && ci) got[k] = fh->enqueueCamera(*ci, f.hb, cx[k], cy[k], side, side, f.tag);
      else if (f.luma) got[k] = fh->enqueueLuma(f.luma + (size_t)cy[k] * f.stride + cx[k], side, side, f.stride, f.tag);
      if (got[k]) { float a[2]; axisOf(f, cr, k, a); got[k]->axis = axisWord(a[0], a[1]); }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(mu);
      error = std::string("ingest: ") + e.what();
    }
    if (got[k]) ready = std::max(ready, got[k]->ready);
  }
  if (!got[0] && !got[1]) { { std::lock_guard<std::mutex> l(mu); win.dropped++; } drop(f.tag); return; }
  // ai: the camera's buffer back once the last of its ingests has run (they complete in order on their queue); a
  // ai: luma or raw frame was copied into the slots' staging, so its buffer is free at once
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
// ai: lane is free, taking every frame staged then up to the batcher's size or the cap (no wait for a full batch since
// ai: 2026-10-08), and the oldest finished in order.
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
          if (it != stamps.ns.end()) ms = it->second.ns / 1e6;
        }
        // ai: the axis word back to pixels for the replay's meta (0, the centre, as none)
        const float ax = fo.axis ? 0.5f * ((float)(fo.axis & 0xffffu) - 32768) : NAN, ay = fo.axis ? 0.5f * ((float)(fo.axis >> 16) - 32768) : NAN;
        float ln[4];
        { std::lock_guard<std::mutex> l(mu); for (int i = 0; i < 4; i++) ln[i] = lensNow[i]; }
        k.replay->frame(fo.luma.data(), fo.w, fo.h, fo.w, ms, true, ax, ay, ln);
      }
    }
    if (fo.empty) continue;
    std::vector<Block> blocks;
    for (auto& r : fo.records) { Block b; memcpy(b.bytes.data(), r.payload.data(), b.bytes.size()); blocks.push_back(b); }
    const FrameVerdict v = xfer->frame(blocks);
    std::lock_guard<std::mutex> l(mu);
    // ai: a frame of a stream the camera has closed since (staged or in flight at the close) counts in no window; its
    // ai: blocks, its row, the totals and the word as every frame's
    if (stamps.in(fo.tag, stream)) {
      win.processed++;
      win.found += fo.finder.found == 1;
      if (fo.finder.found == 1) win.side += sideOf(fo.finder.corners.data());
      if (fo.finder.found == 1) win.anchors.push_back(fo.anchors);
      // ai: the code's corners from the frame's centre: the crop's corners less its axis point (the axis word: half
      // ai: pixels each + 32768; 0 the crop's centre)
      if (fo.finder.found == 1) {
        double ax = fo.w / 2.0, ay = fo.h / 2.0;
        if (fo.axis) { ax = ((double)(fo.axis & 0xffff) - 32768) / 2; ay = ((double)(fo.axis >> 16) - 32768) / 2; }
        for (int q = 0; q < 4; q++) {
          const double x = fo.finder.corners[2 * q] - ax, y = fo.finder.corners[2 * q + 1] - ay;
          win.bx0 = std::min(win.bx0, x); win.by0 = std::min(win.by0, y); win.bx1 = std::max(win.bx1, x); win.by1 = std::max(win.by1, y);
        }
      }
      win.blocks += (long)fo.records.size();
      win.fresh += v.fresh;
      if (v.test) { win.useful += v.fresh; win.test++; }
      win.words += fo.hasWord;
    }
    stamps.frame(fo.tag, fo.records.size(), v.fresh, fo.finder.found == 1, fo.pilotBlocks, fo.pilotR, fo.pilotSd, fo.pilotR2, fo.pilotSd2);
    totalBlocks += (long)fo.records.size();
    totalFrames++;
    if (fo.hasWord) { held = fo.word.version; lastWord = {{"version", fo.word.version}, {"fps", fo.word.fps}, {"ring", fo.ring}}; }
  }
  std::lock_guard<std::mutex> l(mu);
  // ai: the batch's GPU time, a frame's share for each of its frames of the window's stream (an empty one's too)
  if (out.gpuMs && out.carried > 0) {
    int own = 0;
    for (const auto& fo : out.frames) own += stamps.in(fo.tag, stream);
    own = std::min(own, out.carried);
    win.gpuMs += *out.gpuMs * own / out.carried; win.gpuFrames += own;
  }
  // ai: a batch delivered is the decoder working: an error from before (an ingest a plan has since covered, a batch
  // ai: that failed once) no longer describes it. One that keeps failing sets it again before the next delivery.
  error.clear();
  rollWindow(fh->now());
}

// ai: The window's rate (goodputKBs, 2026-10-10): the test stream's new blocks, and a file's blocks as the transfer took
// ai: them (XferRx::taken), not the ids new to the dedupe, which also counted a file's blocks held while its header was
// ai: not being read (and dropped there) and blocks of a chunk already verified: a stalled transfer read the light's full
// ai: rate. mu held.
static long rateFold(XferRx& x, uint64_t& seen) { const uint64_t t = x.taken(); const long d = (long)(t - seen); seen = t; return d; }

void GpuReceiver::rollWindow(double now) {
  if (!opened || now - winStart < 1000) return;
  win.useful += rateFold(*xfer, takenSeen);
  last = win;
  const double dt = (now - winStart) / 1000;
  lastSecs = dt; lastFrames = win.processed;
  last.arrived = (int)std::lround(win.arrived / dt); last.processed = (int)std::lround(win.processed / dt);
  win = Win{};
  winStart = now;
}

// ai: push, mu held: a stream's first frame opens its window at its arrival, nothing in it from before (the releaser's
// ai: hold times of the stream before, the transfer's blocks taken before it)
void GpuReceiver::openWindow(double now) { opened = true; win = Win{}; winStart = now; takenSeen = xfer->taken(); }

// ai: cameraClosed, mu held: the stream's figures end with it (receiver.h cameraClosed)
void GpuReceiver::endWindow() { stream++; opened = false; win = last = Win{}; lastSecs = 1; lastFrames = 0; }

std::string GpuReceiver::stats() {
  auto p = xfer->progress();
  std::lock_guard<std::mutex> l(mu);
  const double n = std::max(1, lastFrames);
  // ai: the median of the anchors read over the last window's found frames (0 where it found none)
  json anchors;
  {
    std::vector<int> a = last.anchors;
    std::sort(a.begin(), a.end());
    const size_t k = a.size();
    anchors = k ? (k % 2 ? (double)a[k / 2] : 0.5 * (a[k / 2 - 1] + a[k / 2])) : 0.0;
  }
  json j = {
    // ai: the test stream in the last window's frames comes first (2026-10-05): what the camera reads now is the state, over
    // ai: a file received or in progress, which comes back once the camera is on its frames again
    {"state", !error.empty() ? "error" : !planned ? "starting" : last.test ? "test" : p.done ? "received" : p.live ? "receiving" : lastWord.is_null() ? "looking" : "found"},
    {"error", error}, {"decoder", "gpu " + variant}, {"zeroCopy", !!ci}, {"layout", cfg.layout},
    {"capturedFps", last.arrived}, {"processedFps", last.processed}, {"dropped", last.dropped},
    {"foundShare", last.found / n}, {"side", last.found ? last.side / last.found : 0},
    {"codeBox", last.bx1 > last.bx0 ? json::array({last.bx0, last.by0, last.bx1, last.by1}) : json()},
    {"heldMs", last.heldN ? last.held / last.heldN : 0}, {"heldMaxMs", last.heldMax}, {"blocks", last.blocks}, {"windowSecs", lastSecs}, {"goodputKBs", last.useful * 469 / 1000.0 / lastSecs},
    {"gpuMs", last.gpuFrames ? last.gpuMs / last.gpuFrames : 0}, {"B", fh ? batchSize : 0}, {"cap", cap.load()},
    {"bandVersion", lastWord.is_null() ? json() : lastWord["version"]}, {"word", lastWord}, {"anchors", anchors},
    {"totals", {{"frames", totalFrames}, {"blocks", totalBlocks}}}, {"series", stamps.series},
    // ai: a file only once a header named one: the page reads a file's presence as a transfer under way; type, the
    // ai: header's media type, is what the app opens and shares the file as (2026-10-01)
    {"file", p.header ? json{{"name", p.name}, {"size", p.length}, {"received", p.bytesIn}, {"sent", p.sent}, {"sentIn", p.sentIn}, {"fraction", p.fraction}, {"verified", p.done}, {"secs", p.secs}, {"chunks", p.chunks}, {"chunksVerified", p.verified}, {"root", p.root}, {"type", p.type}} : json()},
    {"test", last.test > 0},
  };
  return j.dump();
}

std::string GpuReceiver::file() { auto p = xfer->progress(); return p.done ? p.path : ""; }
void GpuReceiver::clear() { xfer->clear(); std::lock_guard<std::mutex> l(mu); lastWord = nullptr; held = 0; }
void GpuReceiver::forget(const std::string& root) { xfer->forget(root); }

// ai: The C on the CPU (cpu/pool.h: the web's worker pool natively): a frame's crop is copied to an idle worker or the
// ai: frame is lost, never queued; each worker decodes blind on its own decoder; the held word is the last word any
// ai: frame read; a frame's verified blocks go into the transfer one frame at a time. The camera's buffer is handed
// ai: back inside push, once the crop is copied.
constexpr int CPU_NMAX = 1536;   // ai: every picture of the ladder (sim/phy.mjs BLIND_NMAX)

class CpuReceiver : public Receiver {
 public:
  // ai: transfer: GpuReceiver's note
  CpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release, std::string gpuWhy, std::shared_ptr<XferRx> transfer);
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
  void forget(const std::string& root) override { xfer->forget(root); }
  bool wantsLuma() const override { return true; }
  void record(std::shared_ptr<Replay> r) override { std::lock_guard<std::mutex> l(mu); rec = std::move(r); }
  void cameraClosed() override { std::lock_guard<std::mutex> l(mu); endWindow(); }

 private:
  std::shared_ptr<XferRx> transfer() const override { return xfer; }
  ReceiverConfig cfg;
  std::function<void(uint64_t)> releaseFn;
  std::string gpuWhy, error;
  // ai: the workers' frames go into it one at a time under its own lock (xfer_rx.h)
  std::shared_ptr<XferRx> xfer;
  std::unique_ptr<CpuPool> pool;
  std::mutex devMu;              // ai: guards `developed`, a raw frame's luma while its push runs
  std::vector<uint8_t> developed;
  std::mutex mu;           // ai: guards the stats window, the word and the error
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
  struct Win { int arrived = 0, dropped = 0, processed = 0, found = 0, words = 0, test = 0, repeat = 0, skips = 0; long blocks = 0, fresh = 0, useful = 0; double ms = 0, side = 0; } win, last;
  uint64_t takenSeen = 0;   // ai: the transfer's taken() at the last window's close, or at its opening
  double winStart = 0, repeatShare = 0;
  int lastMsFrames = 0;    // ai: the frames last.ms is summed over (last.processed is a rate)
  double lastSecs = 1;     // ai: the last window's length: what a count over it is a rate of (GpuReceiver's note)
  uint64_t stream = 0;     // ai: the window's stream and whether one runs (GpuReceiver's note)
  bool opened = false;
  long totalBlocks = 0, totalFrames = 0;
  json lastWord;
  std::shared_ptr<Replay> rec;   // ai: Save replays (record), under mu
  static double now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  void done(CpuFrameOut&& out);
  void rollWindow(double t);
  // ai: mu held, as GpuReceiver's: a stream's first frame opens the window (push); cameraClosed ends it
  void openWindow(double t) { opened = true; win = Win{}; winStart = t; takenSeen = xfer->taken(); }
  void endWindow() { stream++; opened = false; win = last = Win{}; lastMsFrames = 0; lastSecs = 1; repeatShare = 0; }
};

CpuReceiver::CpuReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release, std::string why, std::shared_ptr<XferRx> transfer)
    : cfg(c), releaseFn(std::move(release)), gpuWhy(std::move(why)), xfer(std::move(transfer)) {
  auto log = cfg.log ? cfg.log : [](const std::string&) {};
  if (!xfer) xfer = std::make_shared<XferRx>(cfg.storeDir, log);
  pool = std::make_unique<CpuPool>(CPU_NMAX, 0, cfg.cpuThreads, [this](CpuFrameOut&& o) { done(std::move(o)); }, log);
  log(std::string("receiver: the C on the CPU, ") + (cpu_simd() ? "vector" : "scalar") + " paths, up to " + std::to_string(pool->ceiling()) + " threads" +
      (gpuWhy.empty() ? "" : " (no GPU decoder: " + gpuWhy + ")"));
}

void CpuReceiver::push(const CameraFrame& in) {
  std::unique_lock<std::mutex> dl(devMu, std::defer_lock);
  if (in.raw) dl.lock();   // ai: the develop's buffer, one push's at a time (the frame is copied out before the push ends)
  const CameraFrame f = developedOnCpu(in, developed);
  const double t = now();
  int lost = 0;
  std::shared_ptr<Replay> r;
  {
    std::lock_guard<std::mutex> l(mu);
    if (!opened) openWindow(t);
    stamps.put(f.tag, f.timestampNs, stream);
    r = rec;
  }
  if (!f.luma) {
    std::lock_guard<std::mutex> l(mu);
    error = "the CPU decoder reads a luma plane, and the camera's frame came without one";
    win.arrived++; win.dropped++;
  } else {
    const Crops c = cropsOf(f, cfg.layout);
    for (int k = 0; k < c.n; k++) {
      { std::lock_guard<std::mutex> l(mu); flight.push_back(f.tag); }
      float a[2];
      axisOf(f, c, k, a);
      if (pool->offer(f.luma, f.stride, c.x[k], c.y[k], c.side, c.side, f.tag, held.load(), t, a[0], a[1], f.lens)) {
        // ai: Save replays: the crop a worker took, as it took it
        if (r) r->frame(f.luma + (size_t)c.y[k] * f.stride + c.x[k], c.side, c.side, f.stride, f.timestampNs / 1e6, false, a[0], a[1], f.lens);
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
  const FrameVerdict v = xfer->frame(blocks);
  std::lock_guard<std::mutex> l(mu);
  // ai: a frame of a stream the camera has closed since (a worker's at the close) counts in no window (GpuReceiver's
  // ai: deliver)
  if (stamps.in(out.tag, stream)) {
    win.processed++;
    win.found += out.found;
    if (out.found) win.side += sideOf(out.quad);
    win.blocks += (long)blocks.size();
    win.fresh += v.fresh;
    if (v.test) { win.useful += v.fresh; win.test++; }
    win.ms += out.ms;
    if (!v.fresh && v.seen) win.repeat++;
    win.words += out.hasWord;
  }
  ready.emplace(out.tag, stamps.row(out.tag, blocks.size(), v.fresh, out.found, out.pilotBlocks, out.pilotR, out.pilotSd, out.pilotR2, out.pilotSd2));
  flushRows();
  totalBlocks += (long)blocks.size();
  totalFrames++;
  if (out.hasWord) { held = out.version; lastWord = {{"version", out.version}, {"fps", out.fps}, {"ring", out.ring}}; }
  rollWindow(now());
}

// ai: mu held. A second's window closed: its rates for the page, its repeat share and lost frames for the pool.
void CpuReceiver::rollWindow(double t) {
  if (!opened || t - winStart < 1000) return;
  const double dt = (t - winStart) / 1000;
  win.useful += rateFold(*xfer, takenSeen);
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
    {"threads", pool->size()}, {"threadsMax", pool->ceiling()}, {"simd", cpu_simd() != 0}, {"gpuWhy", gpuWhy},
    {"capturedFps", last.arrived}, {"processedFps", last.processed}, {"dropped", last.dropped},
    {"foundShare", last.found / n}, {"side", last.found ? last.side / last.found : 0},
    {"blocks", last.blocks}, {"windowSecs", lastSecs}, {"goodputKBs", last.useful * 469 / 1000.0 / lastSecs}, {"repeatShare", repeatShare},
    {"cpuMs", lastMsFrames ? last.ms / lastMsFrames : 0}, {"gpuMs", 0}, {"B", 0},
    {"bandVersion", lastWord.is_null() ? json() : lastWord["version"]}, {"word", lastWord},
    {"totals", {{"frames", totalFrames}, {"blocks", totalBlocks}}}, {"series", stamps.series},
    {"file", p.header ? json{{"name", p.name}, {"size", p.length}, {"received", p.bytesIn}, {"sent", p.sent}, {"sentIn", p.sentIn}, {"fraction", p.fraction}, {"verified", p.done}, {"secs", p.secs}, {"chunks", p.chunks}, {"chunksVerified", p.verified}, {"root", p.root}, {"type", p.type}} : json()},
    {"test", last.test > 0},
  };
  return j.dump();
}

}  // namespace

// ai: cpu: the C. gpu: the GPU decoder, or an error. auto: the GPU decoder where the device runs a variant the assets
// ai: hold and is not the CPU itself (VK_PHYSICAL_DEVICE_TYPE_CPU, 2026-10-25), else the C, which every device runs;
// ai: why it is not the GPU's goes to the C's stats as gpuWhy ("a software Vulkan device: llvmpipe (...)"). transfer:
// ai: the one to go on with (create's from), null for one of its own.
static std::unique_ptr<Receiver> made(const ReceiverConfig& c, std::function<void(uint64_t tag)> release, std::shared_ptr<XferRx> transfer) {
  // ai: pc: this end decodes nothing, its frames go to a PC that does (stream.h); the transfer is the PC's
  if (c.decoder == "pc") return makeStreamReceiver(c, std::move(release));
  std::string why;
  if (c.decoder != "cpu") {
    try {
      auto r = std::make_unique<GpuReceiver>(c, release, transfer);
      if (r->start(why, c.decoder == "gpu")) return r;
    } catch (const std::exception& e) { why = e.what(); }
    if (c.decoder == "gpu") throw std::runtime_error(why);
  }
  return std::make_unique<CpuReceiver>(c, std::move(release), why, std::move(transfer));
}

std::unique_ptr<Receiver> Receiver::create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release) {
  return made(c, std::move(release), nullptr);
}

std::unique_ptr<Receiver> Receiver::create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release, Receiver& from) {
  return made(c, std::move(release), from.transfer());
}

}  // namespace lizard
