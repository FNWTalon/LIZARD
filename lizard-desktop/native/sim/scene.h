// ai: A simulated screen filmed by a simulated camera (2026-10-25): the camera every receiver test uses where no camera
// ai: is (the synthetic tests the PC receiver is held to). The sender's pictures are LIZARD's own, painted ahead by the
// ai: C API (lizard.h liz_tx_next, liz_encoder_paint; the test stream or a file's transfer, the pilots' count as each
// ai: picture's number), shown by a display with a clock of its own and filmed by a camera whose exposure, shutter and
// ai: pose are given. What the model keeps, since the phase lock and the decoders live on it:
// ai:   the display: a refresh every 1 / hz on its own clock (ppm off the host's steady clock), the picture of refresh j
// ai:     floor(j fps / hz) (a share `repeat` of pictures, drawn from the seed, held a refresh longer, the later ones
// ai:     shifted by it); each refresh scanned out top to bottom over `scan` of the refresh, so a screen row changes
// ai:     picture at its own time; each pixel then settles to its new value with a time constant responseMs (a
// ai:     grey-to-grey LCD), in linear light (the grey levels through gamma);
// ai:   the camera: each sensor row integrates the light over its exposure, the rows' exposures starting `readoutMs`
// ai:     apart top to bottom (a rolling shutter) or together (global); so a capture holds the pictures each of its
// ai:     rows saw, by the share of its exposure each was on screen, settling included: the mix the pilots measure;
// ai:   the pose: the screen seen through a pinhole two screen heights from it (a phone 60 cm from a 24-inch monitor;
// ai:     `fill` so sets the focal length) looking at its centre from a yaw and a tilt, rolled about its axis, the
// ai:     screen's height `fill` of the frame's head on; the code's frame (margins in) centred on the screen at `share`
// ai:     of the largest size the screen holds (of its height for one code on a landscape screen), sampled bilinearly
// ai:     in the picture's pixels; off the screen a dim room (2% of white); a Gaussian blur on the light; shot and
// ai:     read noise; the camera's tone back to 8 bits through 1 / gamma.
// ai: What it leaves out, and so what no figure from it may claim: a real lens's distortion and vignetting, a real
// ai: sensor's noise spectrum and its ISP, a real panel's overdrive and backlight PWM, the display's resample of the
// ai: picture, glare, a real camera's clocks and its trigger path. Its numbers are the model's.
// ai: Threads: the painters (paint ahead of the display clock, a ring of pictures), and render on the caller's thread,
// ai: split over a pool of its own; render from one thread at a time (a second caller waits for the first).
// ai: Clock: steadyNs's (camera/trigger.h), std::chrono::steady_clock read directly, so the model links without the
// ai: cameras.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lizard::sim {

// ai: What the screen shows: the sender's format and payload.
struct Sender {
  int blocks = 60;          // ai: the format's size (lizard.h liz_format.blocks)
  int ring = -1;            // ai: 0 to 3, or -1 the default (the 128)
  int fps = 60;             // ai: pictures a second, the rate the word states
  int codes = 1;            // ai: 1, or 2 side by side
  std::string file;         // ai: "" the test stream, else this file's transfer
  std::string type;         // ai: the file's media type as its header carries it ("" none; application/x-tar a bundle)
  uint32_t firstId = 0;     // ai: the test stream's first id (0 random)
};

struct Display {
  double hz = 60.0;         // ai: refreshes a second on the display's own clock
  double ppm = 0;           // ai: that clock against the host's (+: the display runs fast)
  double phaseMs = -1;      // ai: the first refresh's start past the scene's start; < 0: drawn from the seed
  double scan = 0.96;       // ai: the scan-out's share of a refresh, its first row to its last (1080 of 1125 lines)
  double responseMs = 3.0;  // ai: a pixel's settling after its row is scanned; 0 none
  uint32_t width = 1920, height = 1080;
  double share = 0.9;       // ai: the code's frame (margins in) over the largest the screen holds
  uint8_t surround = 255;   // ai: the screen around the code
  double repeat = 0;        // ai: the share of pictures shown a refresh too long (a busy desktop's compositor)
};

struct Optics {
  uint32_t width = 1920, height = 1080;   // ai: the frame
  double fill = 0.8;        // ai: the screen's height over the frame's, head on
  double yawDeg = 6, tiltDeg = 4, rollDeg = 1.0;
  double blurPx = 0.7;      // ai: Gaussian sigma in frame pixels; 0 none
  double exposureMs = 8.0;
  bool rolling = true;
  double readoutMs = 10.0;  // ai: a rolling shutter's first row to its last
  double gain = 1.0;        // ai: the light a white screen gives over the sensor's full scale
  double black = 0.03;      // ai: the black level, of full scale
  double readNoise = 0.004; // ai: of full scale
  double shotNoise = 0.006; // ai: of full scale at full scale, by the square root of the signal
  double gamma = 2.2;       // ai: the screen's grey to light, and the camera's tone from light
  uint32_t seed = 1;
};

// ai: The space-separated key=value form the test hooks take (LIZ_CAMERA_SIM, cam_check, sim/fake_v4l2's devices):
// ai: blocks, ring, fps, codes, file, type, id; hz, ppm, phase, scan, response, sw, sh, share, surround, repeat; w, h,
// ai: fill, yaw, tilt, roll, blur, exposure, shutter (rolling|global), readout, gain, black, noise (read), shot, gamma,
// ai: seed. False with why for a key it does not know or a value out of range; keys it does not name are left as they
// ai: were.
bool parse(const std::string& spec, Sender& s, Display& d, Optics& o, std::string& why);

class Scene {
 public:
  // ai: Paints ahead on `painters` threads from now (the scene's start, steadyNs's clock: camera/trigger.h). Null with
  // ai: why where the format or the file is refused.
  static std::unique_ptr<Scene> create(const Sender& s, const Display& d, const Optics& o, int painters, std::string& why);
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;
  // ai: The capture whose first row's exposure begins at t0 (steadyNs's clock): its luma into out, o.width x o.height,
  // ai: rows `stride` apart; exposureMs > 0 in place of the optics' (a camera's exposure set live), gain > 0 in place
  // ai: of the optics' gain (a camera's gain set live). Waits for a picture not yet painted at most a refresh, then takes
  // ai: the newest painted (and counts it late); a picture the ring no longer holds (t0 more than a quarter second
  // ai: behind the screen) is taken as the nearest it holds (and counted stale).
  void render(int64_t t0, uint8_t* out, uint32_t stride, double exposureMs = 0, double gain = 0);
  // ai: The picture on screen at t (its number from the start, the pilots' count; its first row's, the refresh begun
  // ai: at or before t; 0 before the first refresh), for a test's ground truth.
  int64_t pictureAt(int64_t t) const;
  // ai: When picture k is first on screen: the start of its first refresh (the first after it, for a picture the
  // ai: display skips), so pictureAt(shownAt(k)) >= k.
  int64_t shownAt(int64_t k) const;
  int64_t startNs() const;  // ai: the scene's start, when create returned it
  // ai: JSON: painted, late (renders that waited past a refresh), stale, renders, paintMs (a picture's paint, one
  // ai: painter's ms), renderMs (a render's wall ms), frameBlocks, n, side, width, height (the painted frame), slots
  std::string stats();
  const Sender& sender() const { return s_; }
  const Display& display() const { return d_; }
  const Optics& optics() const { return o_; }

  struct Impl;

 private:
  Scene(const Sender& s, const Display& d, const Optics& o);   // ai: in scene.cpp, where Impl is whole
  Sender s_;
  Display d_;
  Optics o_;
  std::unique_ptr<Impl> p_;
};

}  // namespace lizard::sim
