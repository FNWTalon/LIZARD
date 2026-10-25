// ai: The PC receiver's session (2026-10-25): one camera (camera/camera.h) feeding one Receiver (liblizard/core/rx/
// ai: receiver.h), with what the app's screen and the phase lock read beside the receiver's own: the camera's line in
// ai: the stats, the decoder's crop as a preview, each capture's time for the lock, the software trigger it steers.
// ai: rx_jni.cpp is its face for the desktop app (ReceiveState.kt), receive.cpp's lizard_receive the headless one.
// ai: Each frame goes to the receiver on the camera's capture thread with a tag of its own (a count: the receiver
// ai: keys its stamps by tag), its luma handed over where the camera's bytes are one (a planar YUV layout's Y plane, a
// ai: grey mode's bytes) or converted once; the receiver copies a frame's crop inside push and releases it there, so
// ai: the camera's buffer goes back as soon as push returns.
// ai: The receiver outlives the camera (stop, open another, start), and its transfer the receiver: a reconfigure's
// ai: receiver goes on with it (receiver.h create's from). A finished file stays in the store, in file() and the stats,
// ai: until the caller moves it out, or clear() (or a session made later on the same store) lets it go.
// ai: Threads: every call from any thread; the camera calls (open, close, start, stop, set, trigger, reconfigure) one
// ai: at a time under a lock of their own, which can wait seconds on a camera's open or start; camera, stats, series,
// ai: preview, captures, delay and the receiver's calls never wait on them. No call during the destructor.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lizard {

class Replay;
namespace cam { class Camera; }

// ai: A camera as RxSession::camera() describes it, from the camera itself (lizard_receive list's line): {"info",
// ai: "modes", "pick", "controls", "trigger"}; its controls and modes read off the camera, a camera call.
std::string describeCamera(cam::Camera& c);

class RxSession {
 public:
  struct Config {
    std::string assets;      // ai: the GPU tree (ReceiverConfig.assets)
    std::string cacheDir;    // ai: the pipeline cache's folder; "" none
    std::string storeDir;    // ai: the transfer's (XferRx: storeDir/lizard-xfer)
    std::string decoder = "auto", precision = "auto", layout = "1:1", device;
    int cpuThreads = 0;
    std::function<void(const std::string&)> log;
  };
  // ai: The receiver made (Receiver::create: about 12 ms with a warm pipeline cache to under a second cold); null with
  // ai: why where it cannot be (a decoder, precision or layout it does not know; gpu where no GPU decoder runs).
  static std::unique_ptr<RxSession> create(const Config& c, std::string& why);
  // ai: The cameras every backend lists (camera.h list), no session needed: {"cameras": [{id, name, backend, detail}],
  // ai: "notes": [why a backend saw none]}.
  static std::string cameras();
  // ai: The camera stopped and closed, the captures() waits ended and waited out, then the receiver.
  ~RxSession();
  RxSession(const RxSession&) = delete;
  RxSession& operator=(const RxSession&) = delete;

  // ai: A new receiver where the decoder or the layout changed (the same both: nothing to do), made while the old one
  // ai: still reads, on the old one's transfer (receiver.h create's from: a transfer under way goes on, and a file
  // ai: finished before the change or during it stays in file() and the stats, never taken again), the batch cap and
  // ai: the replay handed on as it is swapped in; the camera stopped around the swap and started again in the mode it
  // ai: had (its trigger and the settings kept with it). False with why: the old receiver kept where the new one could
  // ai: not be made; where the camera would not start again, the new receiver kept and the camera stopped.
  bool reconfigure(const std::string& decoder, const std::string& layout, std::string& why);

  // ai: The camera (camera.h open; the one before stopped and closed). The camera rules are the backend's, at start.
  bool open(const std::string& id, std::string& why);
  void close();
  // ai: JSON: {"info": {id, name, backend, detail}, "modes": [{w, h, fps, fpsMin, format, pix, compressed}], "pick":
  // ai: camera.h pick's index for 1920 x 1080 at 60, "controls": [{id, name, unit, min, max, step, def, value,
  // ai: canAuto, isAuto}], "trigger": whether it takes a software trigger, "mode": the index started last (-1 none),
  // ai: "streaming"}; {} with no camera open. Read off the camera at open, start, stop and a set while streaming, and
  // ai: kept: a call here never waits on the camera (the controls' values as of the last of those).
  std::string camera();
  // ai: A user's setting (camera.h set: a control to a value, or to the camera's automatic), kept for the camera open
  // ai: and set again after each start (start, a restart, a reconfigure's), since every backend puts back at stop what
  // ai: set changed and a start applies the camera rules: the user's zoom, exposure and gain hold across every stream
  // ai: until the camera closes, which forgets them. While a stream runs it is set at once (false with the camera's
  // ai: refusal); with none it is kept alone, the camera not touched, and set at the next start (false only for a
  // ai: control the camera does not list, or an automatic it has not). A setting the camera refuses at a start is said
  // ai: in the log ("camera: set again after the start: ..."), the start standing.
  bool set(const std::string& control, double value, bool automatic, std::string& why);
  // ai: Streams modes()[mode] into the receiver until stop, the settings kept set again once it runs. A stream already
  // ai: running is stopped first and its trigger begun again at its period once the new one runs.
  bool start(int mode, std::string& why);
  // ai: The trigger ended, then the stream (its thread joined: no frame after it), then the receiver told the camera
  // ai: closed (receiver.h cameraClosed: what it kept of the stream let go, an ingest error with it).
  void stop();
  bool streaming() const;

  // ai: The receiver's stats JSON (receiver.h stats) and the camera's beside it: "camera": {id, name, backend, detail,
  // ai: streaming, starts (since the camera opened: a restart's new schedule shows here), mode (an object {w, h,
  // ai: format, fps}: the mode started last, null before one; camera()'s "mode" is its index), fps (frames a second
  // ai: the session measured over the last second; in a stream's first second over its frames so far; 0 before its
  // ai: fourth frame: not measured yet; 0 with no stream), agreedFps (the rate the camera agreed to at start, 0 where it
  // ai: names none), frames, dropped, corrupt, convertMs (those four and agreedFps Camera::stats since the last start,
  // ai: read where no camera call runs, else as last read), ended (why the stream stopped by itself, "" none), trigger
  // ai: (Trigger::stats parsed, or null where the camera has none)}, null with no camera open.
  // ai: "live" (2026-10-25): whether a stream runs, so that the receiver's figures of its last window are this
  // ai: stream's. With no stream (no camera, stopped, ended) they read 0 and "test" false: capturedFps, processedFps,
  // ai: dropped, foundShare, side, heldMs, heldMaxMs, blocks, windowSecs, goodputKBs, gpuMs, cpuMs, repeatShare (after
  // ai: a stream's own end the receiver keeps its last window until the next stop). From a start they are the
  // ai: receiver's own: each stop ends its window (receiver.h cameraClosed), so they read 0 until the stream's first
  // ai: window closes, about a second after its first frame, and that window is the stream's alone. The transfer's
  // ai: (file), the word, the totals, the series and the state stay as the receiver says.
  std::string stats();
  // ai: The receiver's own (receiver.h), on the receiver in hand; batchCap's n and record's replay (null: no more) are
  // ai: handed to a receiver a reconfigure makes too, under the lock it swaps under: the last asked is the receiver's.
  std::vector<double> series(double sinceMs);
  std::string file();
  void clear();
  void forget(const std::string& root);
  void batchCap(int n);
  void record(std::shared_ptr<Replay> r);

  // ai: The phase lock's feed (Lock.kt, PhaseLock.onCapture): the captures pushed since the last call, oldest first, as
  // ai: pairs (Frame::ns, 1 where Frame::delayed else 0), waiting up to timeoutMs for one; empty on the timeout, at a
  // ai: stop or the stream's own end during the wait, or once wake() was called (a closing reader: the waits under way
  // ai: end, a later call waits as before). A wait begun with the camera stopped waits out its timeout (a reader's loop
  // ai: never spins). Emptied at each start and each trigger begun; while a trigger runs, its own schedule's captures
  // ai: alone, so a lock reads them alone: a capture the camera's backend matched to one of its triggers
  // ai: (cam::Frame::triggered) stamped at or after its begin (2026-10-25: the free run's capture in flight at the begin
  // ai: came after the emptying, stamped before the schedule); in free run every capture. The newest 1024 kept for a
  // ai: reader that fell behind.
  std::vector<int64_t> captures(int timeoutMs);
  void wake();
  // ai: The camera's software trigger (camera.h Trigger): fired every periodNs from now (> 0), or back to free run (0);
  // ai: false with why where the camera has none or refuses. A stop ends it; a restart (start while streaming, a
  // ai: reconfigure) begins it again at its period, a new schedule (a new phase: the stats' camera "starts" moves, so a
  // ai: lock reading captures can start afresh). delay moves the schedule (Trigger::delay), on the caller's thread,
  // ai: never waiting on a camera call.
  bool trigger(int64_t periodNs, std::string& why);
  void delay(int64_t ns);

  // ai: The newest frame's crop as the decoder reads it (the centre square, or the 2:1 region whole), area-averaged to
  // ai: at most `side` pixels on its long side (never enlarged), into out (w x h, rows of w), with that frame's count
  // ai: in seq (the frame's tag: rising across starts); false where no frame newer than `since` came. The capture
  // ai: thread copies the crop only while a preview was asked within the last second (none for lizard_receive), so a
  // ai: first call after a pause answers false and the next one the newest; the scaling is the caller's thread's.
  bool preview(uint64_t since, int side, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, uint64_t& seq);

  struct Impl;

 private:
  explicit RxSession(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

}  // namespace lizard
