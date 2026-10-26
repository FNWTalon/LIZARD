// ai: A fake webcam behind Media Foundation's IMFMediaSource (2026-10-25, sim/fake_mf.cpp), so cam_check runs the MF
// ai: backend's own path (camera/mf_source.h) through Media Foundation's source reader and its MJPEG decoder with no
// ai: camera. One live video stream offers a webcam's native types: MJPG 1920x1080 at 30 (a range of 5 to 30, as a
// ai: UVC camera's types carry), YUY2 1280x720 at 30, NV12 640x480 at 60 (in a 2D buffer), L8 640x480 at 30 (no
// ai: default stride named: the backend's own), and two bottom-up RGB ones, RGB24 320x240 at 30 (a plain buffer)
// ai: and RGB32 320x240 at 15 (a 2D buffer: Lock2D's pitch negative). The type current on the stream's handler at
// ai: Start is the one sent.
// ai: A frame falls due every 1 / fps from Start on the host's steady clock (clock.h's sleepUntil) and goes out as
// ai: a sample where a request waits, else it is lost, as a camera's are; its time is the capture's in 100 ns since
// ai: Start, as a device source stamps it. The pictures are a moving gradient (fakeMfGradient, 17 a cycle) or the
// ai: caller's paint (the simulated screen); MJPG frames are WIC's JPEG encoder at 4:2:2 from the grey picture, their
// ai: Huffman tables cut as UVC cameras send them where WIC's are the standard ones (JPEG's Annex K counts).
// ai: Windows only.
#pragma once
#if defined(_WIN32)
#include <mfidl.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace lizard::sim {

constexpr uint32_t kFakeMfPictures = 17;

// ai: picture p's luma at (x, y): a triangle wave along x + 2y, moved 30 levels a picture (no edge for JPEG to blur)
inline uint8_t fakeMfGradient(uint32_t x, uint32_t y, uint32_t p) {
  const uint32_t v = (x + 2 * y + 30 * (p % kFakeMfPictures)) % 510;
  return uint8_t(v < 256 ? v : 509 - v);
}

// ai: picture p into luma (w x h, rows stride apart), as the fake paints its uncompressed frames
void fakeMfPaint(uint32_t p, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride);

// ai: A fake device's controls (FakeMfConfig::controls), kept across its sources as a UVC device keeps them:
// ai: IAMCameraControl's and IAMVideoProcAmp's properties by their ids, each its range, step, default, caps, value
// ai: and flags (auto 1, manual 2), and the auto-exposure priority IKsControl reaches (PROPSETID_VIDCAP_CAMERACONTROL
// ai: id 19); a Set out of range, or automatic where the property has no automatic, refused.
struct FakeMfControls {
  struct Prop { long min = 0, max = 0, step = 1, def = 0, caps = 0, value = 0, flags = 0; };
  std::mutex mu;
  std::map<long, Prop> cam, amp;
  long aep = 1, aepFlags = 2;
  int sets = 0;   // ai: Set calls taken, the priority's included
  // ai: a webcam's: Zoom 100 to 500, Focus 0 to 250 by 5 (automatic), Exposure -11 to -2 (log2 s, automatic at -5);
  // ai: Gain 0 to 255 at 0, Sharpness 0 to 255 at 128, BacklightCompensation 0 to 1 at 1, Brightness and Contrast 0
  // ai: to 255 at 128; the priority 1 (the camera may lower its rate)
  static std::shared_ptr<FakeMfControls> webcam();
};

struct FakeMfConfig {
  // ai: the luma of the capture whose exposure begins at t0 (steadyNs's clock), w x h, rows stride apart, called on the
  // ai: fake's own thread; null: the gradient, picture n for the n-th frame since Start
  std::function<void(int64_t t0, uint32_t w, uint32_t h, uint8_t* luma, uint32_t stride)> paint;
  bool stripTables = true;   // ai: MJPG frames without their DHT segments where WIC's tables are the standard ones
  int corruptEvery = 0;      // ai: every n-th MJPG frame cut to half its bytes; 0 none
  // ai: the device's controls; null: none (no IAMCameraControl, IAMVideoProcAmp or IKsControl offered)
  std::shared_ptr<FakeMfControls> controls;
};

class FakeMf : public IMFMediaSource {
 public:
  // ai: a new source with one reference, the caller's; null with why where Media Foundation or WIC is missing
  static FakeMf* create(const FakeMfConfig& c, std::string& why);
  // ai: requests left unanswered (a camera gone quiet), or answered again
  virtual void stall(bool on) = 0;
  // ai: MEError with hr on the source, as a device's source sends when its device goes
  virtual void fail(HRESULT hr) = 0;
  // ai: JSON: starts, stops, shutdown, requests, samples, lost (frames due with no request), stalled, type, tables
  // ai: (cut, kept or none)
  virtual std::string stats() = 0;
};

}  // namespace lizard::sim
#endif
