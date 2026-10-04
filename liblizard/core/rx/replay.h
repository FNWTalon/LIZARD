// ai: Save replays (Developer Tools, 2026-10-03; the web receiver's Record 300 frames, lizard-web/recv.mjs saveRun, as
// ai: a rolling window): the newest frames the decoder is handed, kept as raw luma in a folder of the
// ai: web's run layout, NNNN.gray (w x h bytes, rows of w) then meta.json and stats.jsonl as the run ends, so the
// ai: folder is a run the replay tools read where it lands in research/captures/v0.3/. The frames are the crops
// ai: the decoder reads: the GPU decoder's off its ring as the camera's ingest left them (FrontHalf::run's keep), the
// ai: C's as a worker takes them; in 2:1 each half is a frame of its own. Any thread (the locks: below).
#pragma once
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
  // ai: dir: the replay's folder, made here (emptied first if it holds anything) and the replay's alone; keep: the
  // ai: frames kept, the newest
  Replay(std::string dir, std::string run, int keep);
  Replay(const Replay&) = delete;
  Replay& operator=(const Replay&) = delete;
  // ai: whether it still takes frames: false once the run is ending or ended, or a write failed
  bool live();
  // ai: The GPU decoder's frames come a batch after it takes them (read back with their batch): it holds each as it
  // ai: stages it (false: the replay takes no more), hands it over with frame(..., true), and releases the ones that
  // ai: will not come (dropped before their batch, a batch that failed). finish waits for the held ones.
  bool hold(int n);
  void release(int n);
  // ai: a frame: w x h bytes, rows `stride` apart; ms, its capture's time on the camera's clock; held, one hold()'s.
  // ai: Written as it comes, and the oldest past `keep` deleted.
  void frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, double ms, bool held = false);
  // ai: The run's end, once the held frames have come (or a few seconds have passed): the frames kept renamed
  // ai: 0000.gray on, oldest first; meta.json (run, w, h, frames, of, sizes as the web's; ms, each frame's capture
  // ai: time; taken, the frames the run saw; config, the format the light last named, from the receiver's stats JSON
  // ai: `rxStats` (its word) as the web names it; decoder and layout from the same; error, where a write failed and
  // ai: ended the frames early; then the fields of the object `more`); stats.jsonl (`rows`, a row a line). The frames
  // ai: kept, or -1 when the folder, a rename or the meta failed (error() says why).
  int finish(const std::string& rxStats, const std::string& rows, const std::string& more);
  int frames();
  int64_t bytes();
  std::string error();

 private:
  struct Kept { uint64_t i; uint32_t w, h; double ms; };
  // ai: Two locks: hm guards the holds alone (a push holds a frame under the receiver's lock, which must never wait on
  // ai: a frame's write), mu the files and the window.
  std::mutex hm, mu;
  std::condition_variable cv;
  int pending = 0;                   // ai: hm: frames held and not yet handed over
  std::atomic<bool> closing{false};  // ai: the run's end has begun: no more holds, no more frames but held ones
  std::atomic<bool> dead{false};     // ai: a write failed or the folder could not be made: no more frames
  const std::string dir, run;
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
