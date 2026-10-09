// ai: The receiver as the app sees it: the camera's frames in, a file out, a stats
// ai: snapshot for the page. Everything behind it runs on its own threads: the GPU decoder (dec/, a batch at a time,
// ai: launched as soon as a frame is staged), or the C on the CPU (cpu/: asked for, or where the device has no GPU
// ai: variant); the transfer (xfer_rx.h) on a thread of its own. This header is the contract between core/ and app/'s
// ai: JNI glue (lizard-android/app/src/main/cpp/jni.cpp): nothing in it names Java or Vulkan.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct AHardwareBuffer;

namespace lizard {

class Replay;   // ai: replay.h

struct ReceiverConfig {
  // ai: the generated tree (liblizard/out's setup, blobs, spv, or an install's share/lizard, which may hold some of the
  // ai: decoder variants), read-only
  std::string assets;
  std::string cacheDir;    // ai: the pipeline cache and scratch; empty, no pipeline cache
  std::string storeDir;    // ai: the transfer's chunks and the received file
  std::string decoder = "auto";   // ai: auto (the best GPU variant, else the C on the CPU), gpu, cpu
  int cpuThreads = 0;             // ai: the CPU decoder's workers: 0 the pool's own rule (cpu/pool.h), else that many
  std::string precision = "auto"; // ai: auto (the best the device runs of the variants the assets hold), int8, f16, f32
  std::string device;             // ai: a substring of the Vulkan device's name; empty, LIZ_VK_DEVICE's, else the first discrete
  // ai: 1:1 the frame's centre square; 2:1 its centre 2:1 region along the long side, cut in half, each half a
  // ai: frame of its own (two symbols a camera frame: the sender's two codes side by side)
  std::string layout = "1:1";
  std::function<void(const std::string&)> log;
};

// ai: A frame handed over. Its buffer stays the caller's until release(tag) is called (from a receiver thread): the
// ai: camera's Image is closed then, and not before, since the GPU reads the buffer in place.
struct CameraFrame {
  AHardwareBuffer* hb = nullptr;   // ai: the camera's buffer (zero copy), or null with luma set
  const uint8_t* luma = nullptr;   // ai: a Y plane (the CPU fallback, and desktop replays)
  uint32_t width = 0, height = 0, stride = 0;
  int64_t timestampNs = 0;
  uint64_t tag = 0;
};

class Receiver {
 public:
  static std::unique_ptr<Receiver> create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release);
  virtual ~Receiver() = default;
  // ai: Never blocks on the device: a frame that cannot be taken is released at once (dropped, counted).
  virtual void push(const CameraFrame& f) = 0;
  // ai: The page's snapshot as JSON: state, decoder, fps (camera, processed), goodput, blocks, the transfer's
  // ai: progress and file, the format last read, the GPU's ms a frame; the field names the web receiver's stats row uses
  // ai: where they mean the same (lizard-web/recv.mjs). `windowSecs`: the length of the window the goodput and
  // ai: the registered share are over; `B` and `cap` (the GPU decoder): the batcher's size and batchCap's n.
  virtual std::string stats() = 0;
  // ai: The newest frames with no JSON, for a reader that asks at every capture (the app's phase lock, Engine.kt):
  // ai: [the version the last read word names, 0 before any, then 8 doubles a frame for every frame of the stats'
  // ai: series captured after sinceMs, oldest first: ms on the camera's clock, verified blocks, new blocks, found (1
  // ai: or 0), the pilots' r and its standard error over the even blocks, then over the odd (NaN where the frame read
  // ai: none)]. One lock and a copy; a reader passes the newest ms it has and gets each frame once.
  virtual std::vector<double> series(double sinceMs) = 0;
  // ai: The most frames a GPU launch takes, 1 to 32 (the app's Batch size, 2026-10-02; this meaning since 2026-10-08):
  // ai: a launch still goes as soon as a frame is staged and a lane is free, with every frame then staged up to n; 0,
  // ai: the batcher's size (32 where memory allows). 1 decodes each frame alone. The C decodes a frame at a time.
  virtual void batchCap(int n) = 0;
  // ai: The received file's path once its root is verified, else empty.
  virtual std::string file() = 0;
  // ai: Forget the transfer and its file (the page's Clear).
  virtual void clear() = 0;
  // ai: Whether frames must come with their luma plane (the CPU decoder reads bytes; the GPU's takes the camera's
  // ai: buffer in place): the caller sets its camera up by this.
  virtual bool wantsLuma() const = 0;
  // ai: The camera's reader closed, every frame of it handed back (2026-10-02): the GPU decoder drops its imports of
  // ai: that reader's buffers (CameraIngest::cameraClosed) and an ingest error with them, so a camera started again
  // ai: reads afresh; nothing for the C, which copies each frame's luma.
  virtual void cameraClosed() {}
  // ai: Save replays (replay.h, 2026-10-03): from now on each frame the decoder is handed goes to r as well, in the
  // ai: order the decoder takes them, while r->live(); null stops. The GPU decoder's are read off its ring with their
  // ai: batch (FrontHalf::run's keep), the C's copied as a worker takes them; 2:1's halves are a frame each. The caller
  // ai: ends the run (Replay::finish).
  virtual void record(std::shared_ptr<Replay>) {}
};

}  // namespace lizard
