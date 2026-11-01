// ai: The receiving camera (2026-10-25, the PC's; Android's since 2026-10-31), one interface over each system's own camera
// ai: API: UVC devices through V4L2 on Linux, Media Foundation on Windows, AVFoundation on macOS; Camera2 through the
// ai: NDK on Android (android.cpp); a phone streaming to a PC, a replay played, and the simulated camera the tests read
// ai: (sim.cpp). A camera hands over each frame as luma (luma.h): a YUV mode's Y bytes as they are, an MJPEG frame
// ai: decoded to grey, a Bayer mosaic's 2 x 2 sums, a colour mode's luma; or, where the system has them (Android), the
// ai: camera's own buffer for the GPU decoder to sample in place (Frame::hb, held until release) or a frame to develop
// ai: (Frame::raw: RAW16, RAW10, RAW12; ingest/develop.h). The receiver (core/rx/receiver.h) takes those three. The camera rules (camera.cpp's header)
// ai: are applied at start and the controls a backend changed, the rules' and set's, are put back at stop, since a UVC
// ai: device keeps them for the next app: the user's settings last one stream, and the session sets them again after
// ai: each start (session.h set).
// ai: Threads: list and open from any thread; one Camera is driven by one thread at a time (the session's), and its
// ai: frames come on the backend's own capture thread.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct AHardwareBuffer;

namespace lizard {
struct RawFrame;   // ai: ingest/develop.h
}

namespace lizard::cam {

// ai: How a mode's bytes become luma (luma.h convert): the layouts the backends meet. Grey16 is little-endian
// ai: samples of `bits` significant bits (Mode::bits), the top 8 kept.
enum class Pix : uint8_t {
  Unknown, Grey8, Grey16, YUYV, UYVY, YVYU, VYUY, NV12, NV21, I420, YV12, Bayer8, RGB24, BGR24, BGRA32, MJPEG,
};
const char* pixName(Pix p);

struct Mode {
  uint32_t width = 0, height = 0;
  double fps = 0;           // ai: the rate it runs at: a UVC interval's, the top of a range (AVFoundation)
  double fpsMin = 0;        // ai: a range's bottom; fps where the mode has one rate
  // ai: the camera's own name for its pixels, as its backend writes it: V4L2's fourcc (YUYV, MJPG, GREY, RGGB), Media
  // ai: Foundation's subtype (NV12, YUY2, L8), AVFoundation's FourCharCode (420v, 2vuy), the simulated camera's (GREY)
  std::string format;
  Pix pix = Pix::Unknown;
  uint8_t bits = 8;         // ai: Grey16: the significant bits of a sample
  bool compressed = false;  // ai: MJPEG: decoded on the host (TurboJPEG on Linux; the system's decoder elsewhere)
  // ai: the backend's own: V4L2's fourcc, Media Foundation's media type index, AVFoundation's format index, Android's
  // ai: ImageFormat (0 for its "auto")
  uint64_t key = 0;
  // ai: the size of the second stream a camera with one writes for the platform's preview (Android's, previewTo), 0
  // ai: none: the app sizes its preview surface to it before the stream starts (the surface's size is what the camera
  // ai: service reads)
  uint32_t previewW = 0, previewH = 0;
};

// ai: A camera setting a user may change (Settings) or the camera rules set. ids are the same on every backend:
// ai: zoom, focus, exposure (us), gain (dB, or the camera's units where it names none), fps, sharpness, backlight,
// ai: brightness, contrast; a backend lists the ones the camera has.
struct Control {
  std::string id, name, unit;
  double min = 0, max = 0, step = 0, def = 0, value = 0;
  bool canAuto = false, isAuto = false;
};

struct Info {
  // ai: unique across backends and stable across runs where the system allows (a saved choice finds it again):
  // ai: v4l2:<the /dev/v4l/by-id link, else the node>, mf:<symbolic link>, avf:<unique id>,
  // ai: sim:0 (the simulated camera, LIZ_CAMERA_SIM), upload:<path> (a replay played, never listed)
  std::string id;
  std::string name;         // ai: the device's own
  std::string backend;      // ai: uvc, phone, upload or sim
  std::string detail;       // ai: the bus, the driver
};

struct Frame {
  const uint8_t* luma = nullptr;   // ai: the Y plane, valid during the sink's call alone
  // ai: or the camera's own buffer (Android's PRIVATE or YUV_420_888 AHardwareBuffer), which the GPU decoder samples
  // ai: in place: held by the backend past the sink's call, until release(hold)
  AHardwareBuffer* hb = nullptr;
  // ai: or a frame to develop to luma (RAW16, RAW10, RAW12: ingest/develop.h, width x height the developed frame's),
  // ai: valid during the sink's call alone
  const lizard::RawFrame* raw = nullptr;
  uint32_t width = 0, height = 0, stride = 0;
  // ai: in ns on one clock for the whole stream: when the capture began, the system's time for the frame (V4L2's
  // ai: monotonic buffer time, Media Foundation's QPC, AVFoundation's host clock), moved onto steadyNs's (clock.h);
  // ai: Android's the sensor's own timestamp, the clock its capture results carry (Capture::ns), not moved
  int64_t ns = 0;
  uint64_t seq = 0;                // ai: the camera's frame count
  uint64_t hold = 0;               // ai: hb's token for release, 0 for a frame nothing holds past the sink's call
};

// ai: A capture's result as the camera reports it (Android's Camera2), on the camera's own thread: its timestamp (the
// ai: frames' clock, Frame::ns), whether it was the one delayOnce asked, its frame, exposure and readout skew (ns) and
// ai: its sensitivity; fresh: the first result of a capture session made anew (a start, the preview's change), whose
// ai: frames may start anywhere in the display's refresh.
struct Capture {
  int64_t ns = 0;
  bool delayed = false, fresh = false;
  int64_t frameNs = 0, exposureNs = 0, skewNs = 0;
  int32_t iso = 0;
};

// ai: Since start: frames handed over, dropped (the system's or the camera's: a buffer the stream lost, a sequence
// ai: gap), corrupt (a frame that would not convert: a broken JPEG, a short buffer); the mean time to luma;
// ai: fps, the rate the camera agreed to at start (V4L2's answer to the interval asked), 0 where it names none.
struct Stats {
  uint64_t frames = 0, dropped = 0, corrupt = 0;
  double convertMs = 0, fps = 0;
  // ai: a camera that is another device's (a phone, phone.cpp): the times its own Start or Stop was pressed there
  // ai: since the open, and the last one's way (true Start), for the receiver to follow
  uint64_t asks = 0;
  bool askOn = false;
};

class Camera {
 public:
  virtual ~Camera() = default;
  virtual const Info& info() const = 0;
  virtual std::vector<Mode> modes() = 0;
  virtual std::vector<Control> controls() = 0;
  // ai: One control (Control::id) to value, or to the camera's automatic where `automatic` and the control can be;
  // ai: false with why otherwise. Live while streaming where the camera allows it; put back at the stream's stop (a
  // ai: set on a stopped camera at the next stop, or at close), so a caller that wants it to last sets it again after
  // ai: each start (the session does: session.h set).
  virtual bool set(const std::string& id, double value, bool automatic, std::string& why) = 0;
  using Sink = std::function<void(const Frame&)>;
  using Ended = std::function<void(const std::string& why)>;
  // ai: Streams mode (one of modes()) until stop(): each frame's luma to sink on the capture thread; a stream that ends
  // ai: by itself (unplugged, the device's error) calls ended once, on that thread, and stops. False with why where the
  // ai: mode cannot start.
  virtual bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) = 0;
  // ai: Ends the stream and joins its thread (no sink call after it returns); puts back the controls start and set
  // ai: changed. The simulated camera (sim.cpp) has nothing to put back: its exposure and gain are the model's own.
  virtual void stop() = 0;
  virtual Stats stats() = 0;
  // ai: What the receiver reads of a started stream, for a camera that hands it back to its source (a phone, phone.cpp,
  // ai: 2026-10-27: its phase lock steers by the pilots read here): the receiver's series and its stats JSON, callable
  // ai: from any thread until stop. The session gives them after each start; every other camera ignores them.
  using Series = std::function<std::vector<double>(double sinceMs)>;
  using StatsJson = std::function<std::string()>;
  virtual void readBack(Series series, StatsJson stats) { (void)series; (void)stats; }

  // ai: What a camera that can hand over either is told before each start (the session's, from the receiver: receiver.h
  // ai: wantsLuma): luma bytes (the C, the PC stream) or the camera's own buffer (the GPU decoder's zero copy). Android
  // ai: picks its reader's format and usage by it; every other camera hands luma whatever this says.
  virtual void receiverWants(bool luma) { (void)luma; }
  // ai: A frame's buffer (Frame::hold) let go: the receiver is done with it (its ingest ran, or it was dropped). From any
  // ai: thread, also after stop (a hold the camera no longer knows is nothing).
  virtual void release(uint64_t hold) { (void)hold; }
  // ai: The platform's preview surface (Android: an ANativeWindow*, the TextureView's; null drops it): written as a
  // ai: second stream at the mode's preview size (Mode::previewW, previewH) from the next start, or at once where a
  // ai: stream runs (Android: its capture session made again with it). Cameras with no second stream ignore it; their
  // ai: preview is the session's crop (session.h preview).
  virtual void previewTo(void* window) { (void)window; }
  // ai: The code's box (camera pixels from the frame's centre: x0, y0, x1, y1; the GPU stats' codeBox) for a camera
  // ai: that focuses on a region (Android: the autofocus region), else ignored
  virtual void focusOn(const double box[4]) { (void)box; }
  // ai: What the camera reports of its running stream beyond Stats, a JSON object (Android: the format and sizes it
  // ai: configured, the fps range, the newest capture's exposure, ISO, readout, frame and physical lens): the stats'
  // ai: camera.report and a replay's meta. "{}" where it has nothing to add.
  virtual std::string report() { return "{}"; }

  // ai: The phase lock's actuator (Android's Camera2): one frame made longer moves every frame after it. delayBar: ""
  // ai: where this camera can (manual sensor control applied per frame), else why not (the lock stays off on it).
  // ai: captures: each capture's result and each delayOnce's fate (taken: its frame was captured), on the camera's own
  // ai: thread, from the next start until stop. delayOnce: from that thread, the next frame us microseconds longer
  // ai: (auto-exposure off for that frame alone, at the newest result's exposure and sensitivity); false where it was not
  // ai: asked (no result yet, the camera refused).
  using Captures = std::function<void(const Capture&)>;
  using Fate = std::function<void(bool taken)>;
  virtual std::string delayBar() { return "the camera cannot delay a capture"; }
  virtual void captures(Captures each, Fate fate) { (void)each; (void)fate; }
  virtual bool delayOnce(int64_t us) { (void)us; return false; }
};

// ai: Every backend's cameras: the system's (UVC; Android's back cameras) first, then the simulated one where
// ai: LIZ_CAMERA_SIM is set. A
// ai: backend that could not look (Media Foundation absent, a node named by LIZ_V4L2_NODES missing) says why in notes.
std::vector<Info> list(std::vector<std::string>* notes = nullptr);
// ai: A cheap mark of what list() would find (2026-10-27): the USB devices libusb sees (none opened), the V4L2 nodes,
// ai: the phones whose beacons are heard. It changes where a camera may have come or gone, so a caller watches it and
// ai: lists again only then. Media Foundation's and AVFoundation's devices are in it only through libusb.
std::string listStamp();
std::unique_ptr<Camera> open(const std::string& id, std::string& why);

// ai: The mode a receiver starts with (camera.cpp: the rule): of those at wantW x wantH (0 any), the fastest up to
// ai: wantFps (0 any), then the largest, then uncompressed before MJPEG; none of that size, the nearest by pixels.
// ai: Index into modes, -1 where it is empty.
int pick(const std::vector<Mode>& modes, uint32_t wantW, uint32_t wantH, double wantFps);

}  // namespace lizard::cam
