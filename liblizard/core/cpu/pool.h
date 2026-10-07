// ai: The CPU decoder's threads: the web's worker pool (lizard-web/pool.mjs, recv.mjs)
// ai: natively. A worker is a thread with a decoder of its own (codec.h); a frame goes to an idle worker or is lost,
// ai: never queued; the pool starts at one worker and the policy (the web's PoolPolicy, its numbers) adds one when
// ai: frames are being lost to busy workers and those frames would have carried new data, and gives one back after
// ai: ten slack seconds. A worker given back is parked, not destroyed: its decoder is kept for the next time.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lizard {

// ai: lizard-web/pool.mjs PoolPolicy, its defaults: 4 frames lost before a worker is added; 10 slack seconds before
// ai: one is given back; above 15% repeats a lost frame was a duplicate and buys nothing; a second with up to 2 lost
// ai: frames still counts as slack.
struct PoolPolicy {
  int threshold = 4, slackSecs = 10, slackSkips = 2;
  double repeatCeiling = 0.15;
  int pressure = 0, slack = 0;
  void reset() { pressure = 0; slack = 0; }
  // ai: A frame was lost because every worker was busy: true to add a worker.
  bool lost(bool allReady, int size, int ceiling, double repeatShare);
  // ai: A second passed: true to give a worker back.
  bool tick(int busySkips, int size, bool hasIdle, bool autoSize);
};

// ai: One frame's decode, as the receiver takes it.
struct CpuFrameOut {
  uint64_t tag = 0;
  double at = 0, ms = 0;          // ai: when the frame was offered, and the decode's wall ms
  bool found = false;             // ai: the finder registered a symbol
  int ring = -1, n = 0;           // ai: the ring, and the picture the frame was finished at (0: not finished)
  bool held = false;              // ai: the held word stood in for the frame's own
  bool hasWord = false;
  int version = 0, fps = 0;       // ai: the frame's own word
  float quad[8] = {};             // ai: the symbol's corners in the crop, TL TR BR BL
  int pilotBlocks = 0;            // ai: the pilots (codec.h pilot_*): the blocks read (0 none), r and its standard
  float pilotR = 0, pilotSd = 0, pilotR2 = 0, pilotSd2 = 0;   // ai: error over the even blocks, then over the odd
  int blockBytes = 0;
  std::vector<uint8_t> blocks;    // ai: the verified blocks, blockBytes each
};

class CpuPool {
 public:
  // ai: nmax: the largest picture decoded. ceiling: workers at most (0: the machine's cores). fixed: that many
  // ai: workers always (0: the policy decides). done: called on a worker's thread with each frame it decoded.
  // ai: tiers: a rate profile every worker's decoder takes (codec.h cpu_dec_tiers; the lab's rate-by-ring arm,
  // ai: 2026-10-07), "" for one rate; one refused is logged and the worker decodes at one rate.
  CpuPool(int nmax, int ceiling, int fixed, std::function<void(CpuFrameOut&&)> done, std::function<void(const std::string&)> log, std::string tiers = "");
  ~CpuPool();
  // ai: The workers stopped and joined (each frame in hand finished, its done() returned); idempotent, the destructor's
  // ai: first step. An owner whose done() reaches back to the pool calls it before it lets go of its pointer.
  void stop();
  CpuPool(const CpuPool&) = delete;
  CpuPool& operator=(const CpuPool&) = delete;
  // ai: A frame's crop (w x h at (x, y) of a plane whose rows are `stride` bytes apart) to an idle worker, copied
  // ai: before the call returns; false when every worker is busy (the frame is lost and the caller counts it).
  // ai: held: the version of the last word read, 0 none.
  bool offer(const uint8_t* plane, uint32_t stride, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag, int held, double at);
  // ai: The policy's two hooks, the receiver's to call: a frame was lost to busy workers (repeatShare: the share of
  // ai: the last second's decoded frames that carried nothing new); a second passed with busySkips frames lost.
  void lost(double repeatShare);
  void tick(int busySkips);
  int size() const { return size_.load(); }   // ai: workers frames are given to
  int ready() const;                          // ai: those of them that have their decoder
  int ceiling() const { return ceiling_; }

 private:
  struct Worker;
  void run(Worker* w);
  void grow();
  int nmax_, ceiling_, fixed_;
  std::string tiers_;
  std::function<void(CpuFrameOut&&)> done_;
  std::function<void(const std::string&)> log_;
  mutable std::mutex mu_;                       // ai: guards the workers' states and the policy
  std::vector<std::unique_ptr<Worker>> workers_;   // ai: never shrinks: a worker past size_ is parked
  std::atomic<int> size_{0};
  PoolPolicy policy_;
  bool stop_ = false;
};

}  // namespace lizard
