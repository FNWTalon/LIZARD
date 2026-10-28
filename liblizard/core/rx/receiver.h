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

struct RawFrame;   // ai: ingest/develop.h

class Replay;   // ai: replay.h
class XferRx;   // ai: xfer_rx.h

struct ReceiverConfig {
  // ai: the generated tree (liblizard/out's setup, blobs, spv, or an install's share/lizard, which may hold some of the
  // ai: decoder variants), read-only
  std::string assets;
  std::string cacheDir;    // ai: the pipeline cache and scratch; empty, no pipeline cache
  std::string storeDir;    // ai: the transfer's chunks and the received file
  // ai: auto: the GPU decoder's best variant where the device runs one and is no software device (Vulkan's CPU type:
  // ai: lavapipe, SwiftShader; 2026-10-25), else the C on the CPU, why in its stats' gpuWhy; gpu: the GPU decoder on
  // ai: any device, a software one too, else an error; cpu: the C; pc: none here, the frames served to a PC that
  // ai: decodes them (stream.h, 2026-10-27)
  std::string decoder = "auto";
  int cpuThreads = 0;             // ai: the CPU decoder's workers: 0 the pool's own rule (cpu/pool.h), else that many
  std::string precision = "auto"; // ai: auto (the best the device runs of the variants the assets hold), int8, f16, f32
  std::string device;             // ai: a substring of the Vulkan device's name; empty, LIZ_VK_DEVICE's, else the first discrete
  // ai: 1:1 the frame's centre square; 2:1 its centre 2:1 region along the long side, cut in half, each half a
  // ai: frame of its own (two symbols a camera frame: the sender's two codes side by side)
  std::string layout = "1:1";
  // ai: the GPU decoder's queues at the global priority LOW where the device offers it (wg::DeviceExtras::lowPriority):
  // ai: the screen, the preview and the compositor first, the decode in its slack (2026-10-27: on the S26, the lag
  // ai: replay beside the app's camera and screen, the screen's 99th percentile frame 23 to 81 ms to 17 to 19, as with
  // ai: no decoder, for 5.92 to 6.38 ms of GPU a half, every half decoded)
  bool yieldGpu = true;
  std::function<void(const std::string&)> log;
};

// ai: A frame handed over. Its buffer stays the caller's until release(tag) is called (from a receiver thread): the
// ai: camera's Image is closed then, and not before, since the GPU reads the buffer in place.
struct CameraFrame {
  AHardwareBuffer* hb = nullptr;   // ai: the camera's buffer (zero copy), or null with luma set
  const uint8_t* luma = nullptr;   // ai: a Y plane (the CPU fallback, and desktop replays)
  // ai: or a frame to develop to luma (ingest/develop.h: P010, RAW16, RAW10, RAW12; width x height the developed
  // ai: frame's): the GPU decoder develops it on its device, the others on the pushing thread (developedOnCpu)
  const RawFrame* raw = nullptr;
  uint32_t width = 0, height = 0, stride = 0;
  int64_t timestampNs = 0;
  uint64_t tag = 0;
  // ai: the camera's report of its lens in this frame's px: k1, k2, k3 (Android's LENS_DISTORTION form) and the focal
  // ai: length f; f 0 none. The C's lens model takes it (src/focus.h focus_set_lens); a replay's meta records it.
  float lens[4] = {0, 0, 0, 0};
};

// ai: f with its raw frame developed to luma on the calling thread (a buffer of that thread's, good until its next
// ai: call), or f itself where it has none
CameraFrame developedOnCpu(const CameraFrame& f);

class Receiver {
 public:
  static std::unique_ptr<Receiver> create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release);
  // ai: A receiver that goes on with from's transfer (2026-10-25, for a change of decoder or layout while a transfer is
  // ai: in hand, the PC receiver's): the transfer in hand, its store with a file it finished, the dedupe and the blocks
  // ai: taken shared with from, and no store opened on c.storeDir (one of its own would let the finished folders go, a
  // ai: file from finishes meanwhile with them, and the file would be taken again from the light). from may go on
  // ai: feeding the transfer until it is let go: the transfer takes one frame at a time from either (xfer_rx.h). The
  // ai: transfer keeps the log it was made with (the first receiver's config's), which must outlive the new receiver.
  static std::unique_ptr<Receiver> create(const ReceiverConfig& c, std::function<void(uint64_t tag)> release, Receiver& from);
  virtual ~Receiver() = default;
  // ai: Never blocks on the device: a frame that cannot be taken is released at once (dropped, counted).
  virtual void push(const CameraFrame& f) = 0;
  // ai: The page's snapshot as JSON: state, decoder, fps (camera, processed), goodput, blocks, the transfer's
  // ai: progress and file, the format last read, the GPU's ms a frame; the field names the web receiver's stats row uses
  // ai: where they mean the same (lizard-web/recv.mjs). `windowSecs`: the length of the window the goodput and
  // ai: the registered share are over (a window opens at a stream's first frame and closes a second or more on, at the
  // ai: next frame the C is handed or decodes, the next batch the GPU delivers; the figures 0 until a stream's first
  // ai: closes: cameraClosed); `B` and `cap` (the GPU decoder): the batcher's size and batchCap's n; `anchors` (the GPU
  // ai: decoder): the median of the anchors it read over the window's found frames (SPEC 6.13), null with the C.
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
  // ai: Forget one transfer by its root, the one in hand or one parked unfinished (2026-10-22, an app's Delete of an
  // ai: unfinished transfer; xfer_rx.h forget). Done when it returns.
  virtual void forget(const std::string& root) { (void)root; }
  // ai: Whether frames must come with their luma plane (the CPU decoder reads bytes; the GPU's takes the camera's
  // ai: buffer in place where the device has the zero-copy ingest, and luma where it has not): the caller sets its camera
  // ai: up by this.
  virtual bool wantsLuma() const = 0;
  // ai: The camera's stream closed (its reader, a stop), every frame of it handed back (2026-10-02): the GPU decoder
  // ai: drops its imports of that reader's buffers (CameraIngest::cameraClosed) and an ingest error with them, so a
  // ai: camera started again reads afresh. On both decoders the stats window ends with the stream (2026-10-25): the
  // ai: window open at the close is dropped and the last one's figures with it (they read 0, as a new receiver's,
  // ai: until the next stream's first window closes), and the next window opens at the next frame's arrival, as a
  // ai: receiver's first does at its first frame: the first window after a restart is that stream's own, never one
  // ai: over the time off. A frame of the stream before still in the decoder at the close counts in no window (its
  // ai: blocks go to the transfer and its row to the series as ever). The transfer, the held word, the totals and the
  // ai: series stay.
  virtual void cameraClosed() {}
  // ai: Save replays (replay.h, 2026-10-03): from now on each frame the decoder is handed goes to r as well, in the
  // ai: order the decoder takes them, while r->live(); null stops. The GPU decoder's are read off its ring with their
  // ai: batch (FrontHalf::run's keep), the C's copied as a worker takes them; 2:1's halves are a frame each. The caller
  // ai: ends the run (Replay::finish).
  virtual void record(std::shared_ptr<Replay>) {}

 private:
  // ai: the transfer this receiver feeds, for create's from; null where it has none, and create then gives the new
  // ai: receiver a transfer of its own
  virtual std::shared_ptr<XferRx> transfer() const { return nullptr; }
};

}  // namespace lizard
