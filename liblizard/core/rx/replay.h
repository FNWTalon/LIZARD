// ai: Save replays (Developer Tools, 2026-10-03; the web receiver's Record 300 frames, lizard-web/recv.mjs saveRun, as
// ai: a rolling window): the newest frames the decoder is handed, kept as raw luma in a scratch folder beside the
// ai: replay's file while the run records (<file>.frames/, a frame a file, cheap on the camera's side), then at the
// ai: run's end packed into the replay, a .lizard (lizfile.h, 2026-10-28: a folder of NNNN.gray, meta.json and
// ai: stats.jsonl before), the scratch removed. The frames are the crops the decoder reads: the GPU decoder's off its
// ai: ring as the camera's ingest left them (FrontHalf::run's keep), the C's as a worker takes them; in 2:1 each half
// ai: is a frame of its own. Any thread (the locks: below).
#pragma once
#include <array>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace lizard {

class Replay {
 public:
  // ai: file: the replay, a .lizard (refused where it exists, or its scratch folder holds anything); keep: the frames
  // ai: kept, the newest
  Replay(std::string file, std::string run, int keep);
  Replay(const Replay&) = delete;
  Replay& operator=(const Replay&) = delete;
  // ai: whether it still takes frames: false once the run is ending or ended, or a write failed
  bool live();
  // ai: The GPU decoder's frames come a batch after it takes them (read back with their batch): it holds each as it
  // ai: stages it (false: the replay takes no more), hands it over with frame(..., true), and releases the ones that
  // ai: will not come (dropped before their batch, a batch that failed). finish waits for the held ones.
  bool hold(int n);
  void release(int n);
  // ai: a frame: w x h bytes, rows `stride` apart; ms, its capture's time on the camera's clock; held, one hold()'s;
  // ai: ax, ay where the camera's optical axis fell in it (NaN: its centre); lens, the camera's report of its lens
  // ai: (k1, k2, k3, f px; null or f 0 none). Written as it comes, and the oldest past `keep` deleted.
  void frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, double ms, bool held = false, float ax = NAN, float ay = NAN, const float* lens = nullptr);
  // ai: The run's end, once the held frames have come (or a few seconds have passed): the file written, the frames
  // ai: kept oldest first (each scratch frame removed once it is in), then stats.jsonl (`rows`, a row a line), then
  // ai: meta.json (run, w, h, frames, of, sizes as the web's; ms, each frame's capture time; axes, each frame's camera
  // ai: axis ([x, y], null the centre); lens, each frame's lens report ([k1, k2, k3, f], null none); taken, the frames
  // ai: the run saw; config, the format the light last named, from the receiver's stats JSON `rxStats` (its word) as
  // ai: the web names it; decoder and layout from the same; error, where a write failed and ended the frames early;
  // ai: then the fields of the object `more`), its frames those in the file. The frames kept, or -1 when the file
  // ai: could not be made (error() says why).
  int finish(const std::string& rxStats, const std::string& rows, const std::string& more);
  int frames();
  int64_t bytes();   // ai: the frames' raw bytes while recording, the file's once finished
  std::string error();

 private:
  struct Kept { uint64_t i; uint32_t w, h; double ms; float ax, ay; std::array<float, 4> lens; };
  // ai: Two locks: hm guards the holds alone (a push holds a frame under the receiver's lock, which must never wait on
  // ai: a frame's write), mu the files and the window.
  std::mutex hm, mu;
  std::condition_variable cv;
  int pending = 0;                   // ai: hm: frames held and not yet handed over
  std::atomic<bool> closing{false};  // ai: the run's end has begun: no more holds, no more frames but held ones
  std::atomic<bool> dead{false};     // ai: a write failed or the folder could not be made: no more frames
  const std::string file, dir, run;   // ai: dir: the scratch folder
  const int keep;
  bool ended = false;       // ai: mu: renamed and written; nothing comes in after
  std::string err;          // ai: mu: what stopped the frames (a write, the folder), kept for meta.json
  bool broken = false;      // ai: mu: the folder could not be made: nothing to finish
  uint64_t taken = 0;
  int64_t held = 0;         // ai: the kept frames' bytes
  std::deque<Kept> kept;
  std::vector<uint8_t> rowsBuf;   // ai: a frame's rows packed where they come `stride` apart
  std::string path(uint64_t i) const;
  bool put(const std::string& p, const void* data, size_t n);
};

}  // namespace lizard
