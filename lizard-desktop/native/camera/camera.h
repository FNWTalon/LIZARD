// ai: The receiving camera on a PC (2026-10-25), one interface over each system's own camera API for UVC devices (V4L2
// ai: on Linux, Media Foundation on Windows, AVFoundation on macOS), and the simulated camera the tests read (sim.cpp).
// ai: A camera hands over each frame's
// ai: luma (luma.h): a YUV mode's Y bytes as they are, an MJPEG frame decoded to grey, a Bayer mosaic's 2 x 2 sums, a
// ai: colour mode's luma; the receiver (core/rx/receiver.h) takes nothing else. The camera rules (camera.cpp's header)
// ai: are applied at start and the controls a backend changed, the rules' and set's, are put back at stop, since a UVC
// ai: device keeps them for the next app: the user's settings last one stream, and the session sets them again after
// ai: each start (rx_session.h set).
// ai: Threads: list and open from any thread; one Camera is driven by one thread at a time (the session's), and its
// ai: frames come on the backend's own capture thread.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
  // ai: the backend's own: V4L2's fourcc, Media Foundation's media type index, AVFoundation's format index
  uint64_t key = 0;
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
  // ai: sim:0 (the simulated camera, LIZ_CAMERA_SIM)
  std::string id;
  std::string name;         // ai: the device's own
  std::string backend;      // ai: uvc, phone or sim
  std::string detail;       // ai: the bus, the driver
};

struct Frame {
  const uint8_t* luma = nullptr;   // ai: the Y plane, valid during the sink's call alone
  uint32_t width = 0, height = 0, stride = 0;
  // ai: in ns on one clock for the whole stream: when the capture began, the system's time for the frame (V4L2's
  // ai: monotonic buffer time, Media Foundation's QPC, AVFoundation's host clock), moved onto steadyNs's (clock.h)
  int64_t ns = 0;
  uint64_t seq = 0;                // ai: the camera's frame count
};

// ai: Since start: frames handed over, dropped (the system's or the camera's: a buffer the stream lost, a sequence
// ai: gap), corrupt (a frame that would not convert: a broken JPEG, a short buffer); the mean time to luma;
// ai: fps, the rate the camera agreed to at start (V4L2's answer to the interval asked), 0 where it names none.
struct Stats { uint64_t frames = 0, dropped = 0, corrupt = 0; double convertMs = 0, fps = 0; };

class Camera {
 public:
  virtual ~Camera() = default;
  virtual const Info& info() const = 0;
  virtual std::vector<Mode> modes() = 0;
  virtual std::vector<Control> controls() = 0;
  // ai: One control (Control::id) to value, or to the camera's automatic where `automatic` and the control can be;
  // ai: false with why otherwise. Live while streaming where the camera allows it; put back at the stream's stop (a
  // ai: set on a stopped camera at the next stop, or at close), so a caller that wants it to last sets it again after
  // ai: each start (the session does: rx_session.h set).
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
};

// ai: Every backend's cameras: UVC (the system's) first, then the simulated one where LIZ_CAMERA_SIM is set. A
// ai: backend that could not look (Media Foundation absent, a node named by LIZ_V4L2_NODES missing) says why in notes.
std::vector<Info> list(std::vector<std::string>* notes = nullptr);
std::unique_ptr<Camera> open(const std::string& id, std::string& why);

// ai: The mode a receiver starts with (camera.cpp: the rule): of those at wantW x wantH (0 any), the fastest up to
// ai: wantFps (0 any), then the largest, then uncompressed before MJPEG; none of that size, the nearest by pixels.
// ai: Index into modes, -1 where it is empty.
int pick(const std::vector<Mode>& modes, uint32_t wantW, uint32_t wantH, double wantFps);

}  // namespace lizard::cam
