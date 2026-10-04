// ai: The app's sender (2026-10-01): lizard-web/send-worker.mjs natively, the C on the CPU (NEON). One thread makes each
// ai: frame's blocks in order (a file's ids by XferTx's schedule, or the test stream's from a random first id, SHAKE256 of
// ai: each id, as the page's); a few painters, each with its own codec (src/focus.c keeps its scratch in the codec),
// ai: encode and paint them ahead (focus_encode, focus_paint_rgba: grey RGBA with the margin, the pilots' count the
// ai: frame's number mod 4); the app takes them in order (take) and puts them on the screen at the display's pace.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu_painter.h"
#include "xfer_tx.h"

namespace lizard {

// ai: The symbol (lizard-web/send.mjs spec): n the picture, subch the sub-channels (the version is subch / 8), span
// ai: the ring's (2 x its cells), fps the display rate the word states; threads the painters. painter (2026-10-02, the
// ai: web's encoder switch): 0 the C on the CPU, 1 the GPU (gpu_painter.h; its absence an error), 2 auto: the GPU where
// ai: it builds and its first frames are the C's (the border to the byte, the square within a grey level), else the
// ai: CPU; assets the tree the GPU's kernels and tables come from (liblizard/out as the app ships it). codes (2026-10-03, the
// ai: desktop sender's; the web's #codes): 1, or 2 symbols side by side gap modules apart, one format, one word
// ai: and one pilot count, a frame's ids split between them (the first code the first blocks, as send-worker.mjs).
// ai: device (2026-10-03, the library): the GPU painter's Vulkan device, a substring of its name (empty: LIZ_VK_DEVICE's,
// ai: else the first discrete). gap (2026-10-04, the desktop sender's setting; the web's gap slider): the modules between
// ai: two codes, 0 to 64, GAP_MODULES unless set; one code has none.
struct TxFormat { int n = 512, subch = 96, span = 128, fps = 60, threads = 3, painter = 0, codes = 1, gap = GAP_MODULES; std::string assets, device; };

class Sender {
 public:
  // ai: A file (data alive as long as this: the app maps it) or, with data null, the test stream.
  Sender(const uint8_t* data, size_t length, const std::string& name, const std::string& type);
  ~Sender();
  Sender(const Sender&) = delete;
  Sender& operator=(const Sender&) = delete;

  // ai: (Re)starts the painters on a format; the transfer goes on where it was (a re-pick, as the page's respec). The
  // ai: frames painted ahead under the old format are dropped (their ids are lost to the fountain, as a frame never
  // ai: shown is). Returns an error, or "".
  std::string configure(const TxFormat& f);
  // ai: The GPU painter made ahead of a configure (its device and pipelines: off the thread that configures, which on
  // ai: the phone is the main one); "" or why there is none (configure then sends from the CPU, or refuses painter 1).
  std::string prepareGpu(const std::string& assets, const std::string& device = "");
  // ai: The painted frame's pixels a side (a symbol and its margin: the frame's height), 0 before a configure; width(),
  // ai: the frame's width (codes symbols and the gaps between them; side() with one code).
  int side() const { return side_; }
  int width() const { return width_; }
  // ai: The next frame in order, RGBA width() x side(), into dst rows `stride` bytes apart, if it is painted: true, and
  // ai: it is the screen's now (its count advances); false and dst untouched if the painters are behind. A GPU frame is
  // ai: grey (one byte a pixel), made RGBA here as it is copied.
  bool take(uint8_t* dst, int stride);
  // ai: Whether take would give a frame now (the one consumer asks before it locks a window buffer).
  bool ready();
  // ai: The stats JSON (Sender::stats in sender.cpp names the fields).
  std::string stats();
  // ai: Where the frames are painted: "gpu" or "cpu", and why not the GPU where it was asked for or auto and is not.
  std::string painter() const { return onGpu_ ? "gpu" : "cpu"; }
  const std::string& gpuWhy() const { return gpuWhy_; }
  const XferTx* xfer() const { return xfer_.get(); }

 private:
  struct Job { uint64_t seq; std::vector<uint8_t> blocks; int data; };
  void stopAll();
  void produce();
  void paint(int k);
  void paintGpu();

  std::unique_ptr<XferTx> xfer_;
  uint32_t nextId_;                 // ai: the test stream's next id (a random first one)
  TxFormat fmt_;
  int side_ = 0, width_ = 0, gap_ = 0, blocks_ = 0, blockBytes_ = 0, gen_ = 0;
  std::string label_, error_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = true;
  uint64_t made_ = 0;               // ai: frames made (their blocks), the next to make
  uint64_t shown_ = 0;              // ai: frames taken, the next to take; its count mod 4 is the pilots'
  std::deque<Job> jobs_;            // ai: made, not painted
  size_t inFlight_ = 0;             // ai: in a painter's hands
  std::map<uint64_t, std::vector<uint8_t>> ready_;   // ai: painted, not taken
  std::map<uint64_t, int> dataOf_;  // ai: a frame's data blocks (its header and manifest not counted), for the lap
  std::vector<std::vector<uint8_t>> spare_;
  std::thread producer_;
  std::vector<std::thread> painters_;
  std::unique_ptr<GpuPainter> gpu_;   // ai: kept across configures (its device made once a send)
  bool onGpu_ = false;
  std::string gpuWhy_, gpuDevice_;
  // ai: the stats' window: frames taken, data blocks shown, paint ms summed
  std::atomic<uint64_t> takenWin_{0}, dataWin_{0}, dataAll_{0}, paintedWin_{0};
  std::atomic<double> paintMsWin_{0};
  double winStart_ = 0, lastFps_ = 0, lastKBs_ = 0, lastPaintMs_ = 0;
};

}  // namespace lizard
