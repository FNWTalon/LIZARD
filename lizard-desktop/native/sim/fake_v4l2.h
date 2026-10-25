// ai: Fake UVC devices behind a V4l2Sys table (2026-10-25), the V4L2 backend's test bench: nothing of /dev is touched.
// ai: Device i is the node /dev/video<90 + i> (a name only) with the by-id link
// ai: /dev/v4l/by-id/usb-LIZARD_Fake_<profile>-video-index0 (index1 for a metadata node; a profile met again gets
// ai: _<k> after its name). Profiles:
// ai:   c930e: MJPG 1920x1080, 1280x720, 640x480 at 30 down to 5; YUYV 1920x1080 at 5, 1280x720 at 10 down, 640x480 at
// ai:     30 down (the intervals a C930e lists: 30, 24, 20, 15, 10, 7.5, 5); a webcam's controls (brightness, contrast,
// ai:     gain, sharpness 0..255, backlight_compensation 0..1 at 1, exposure_auto {1 manual, 3 aperture priority},
// ai:     exposure_absolute 3..2047, exposure_auto_priority at 1, focus_automatic_continuous, focus_absolute,
// ai:     zoom_absolute 100..500, power_line_frequency);
// ai:   uvc60: MJPG 1280x720 at 60 and 30, 1920x1080 at 30; YUYV 640x480 at 60 and 30; NV12 1280x720 at 30;
// ai:     exposure_auto {0 auto, 1 manual} (no aperture priority);
// ai:   mono: GREY 1280x1024 at 60 and 30, Y16 1280x1024 at 30; gain and exposure;
// ai:   bayer: SRGGB8 1920x1200 at 60, an old driver's: no extended controls (QUERY_EXT_CTRL, G/S_EXT_CTRLS: ENOTTY),
// ai:     no device caps, buffers stamped with no clock (the backend's dequeue time stands in);
// ai:   stepwise: GREY sizes 160..2048 step 16 by 120..1536 step 8, intervals continuous 1/120 to 1/1; no controls;
// ai:   formats: UYVY, YVYU, VYUY, NV21, YUV420, YVU420, Y10, Y12, RGB24, BGR24 at 320x240 and 30, and H264 (which no
// ai:     receiver takes);
// ai:   card: an old capture card's YUYV 1280x720 at 60 that enumerates no size and no interval and has no G/S_PARM
// ai:     (ENOTTY): its size known from TRY_FMT alone, its rate unstated;
// ai:   meta: a UVC camera's metadata node (device caps META_CAPTURE alone), which a backend skips.
// ai: The kernel's semantics as uvcvideo and videobuf2 have them: QUERYCAP; ENUM_FMT, ENUM_FRAMESIZES,
// ai: ENUM_FRAMEINTERVALS; G/S/TRY_FMT (a format not offered becomes the first, a size not offered the nearest by
// ai: uvcvideo's measure; S_FMT busy with buffers allocated); G/S_PARM (the nearest interval; busy while streaming);
// ai: QUERY_EXT_CTRL and QUERYCTRL with NEXT_CTRL, QUERYMENU (sparse menus), G/S_CTRL, G/S/TRY_EXT_CTRLS (values
// ai: clamped to their range and step; a menu index not offered EINVAL; an absolute exposure or focus EACCES unless its
// ai: master is manual, the same call's master counting; values kept per device across opens, as hardware keeps them);
// ai: REQBUFS (MMAP only; 2 to 8; 0 frees, a mapped buffer's memory kept until unmapped; another fd's queue EBUSY),
// ai: QUERYBUF, mmap by offset (MAP_SHARED), QBUF, DQBUF (EAGAIN on a non-blocking fd with none done), STREAMON,
// ai: STREAMOFF (every buffer back to the program), poll (POLLIN with one done, POLLERR not streaming or in error,
// ai: POLLERR | POLLHUP once unplugged), close (the queue freed where that fd held it); an unknown ioctl ENOTTY.
// ai: Frames come from a producer thread at the interval on the device's own clock (an offset and a ppm against
// ai: steadyNs's), stamped CLOCK_MONOTONIC at the frame's start (start of exposure, as uvcvideo), each delivered an
// ai: interval later; the sequence counts every frame the device makes, so a frame with no buffer queued is a gap, as
// ai: on uvcvideo. The picture: a scene (sim/scene.h, rendered at each frame's start) where the device's config names
// ai: one, else a moving gradient (gradient()); the luma packed into the format (YUV with chroma 128, Y16 the luma
// ai: shifted to the top, Bayer the luma at every site, RGB grey, MJPEG through jpeg.h's encoder with its tables cut).
// ai: Faults per device: a share of frames flagged V4L2_BUF_FLAG_ERROR, a share of MJPEG frames cut short (10 to 60%
// ai: of their bytes kept), a share skipped (sequence gaps), an unplug N frames into a stream (every later call
// ai: ENODEV, open ENOENT, the node gone from nodes()), the stream's error N frames into it (DQBUF and QBUF EIO until
// ai: STREAMOFF; the next stream runs N frames again).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "camera/v4l2_sys.h"

namespace lizard::sim {

struct FakeFaults {
  double errors = 0;          // ai: the share of frames flagged V4L2_BUF_FLAG_ERROR
  double truncated = 0;       // ai: the share of MJPEG frames cut short
  double skipped = 0;         // ai: the share of frames the device skips
  int64_t unplugAfter = -1;   // ai: frames a stream delivers before the device goes; -1 never
  int64_t eioAfter = -1;      // ai: frames a stream delivers before its error (EIO); -1 never
  uint32_t seed = 1;
};

struct FakeDevice {
  std::string profile;        // ai: c930e, uvc60, mono, bayer, stepwise, formats, card, meta
  std::string scene;          // ai: sim/scene.h parse's spec, its frame size the format's; "" the gradient
  double offsetMs = 0;        // ai: the device clock's first frame past the stream's start
  double ppm = 0;             // ai: the device clock against steadyNs's (+: fast, its frames closer together)
  int quality = 90;           // ai: its MJPEG frames' JPEG quality
  FakeFaults faults;
};

// ai: A device's counts since it was made; the sequences, the format and the state are the latest stream's and now
struct FakeServed {
  uint64_t opens = 0, streams = 0;
  uint64_t frames = 0;        // ai: delivered into the done queue
  uint64_t errors = 0, truncated = 0, skipped = 0;
  uint64_t starved = 0;       // ai: frames the device made with no buffer queued (a gap, as on uvcvideo)
  uint64_t late = 0;          // ai: frames the producer fell behind on (a gap)
  uint64_t dequeued = 0, dequeuedErrors = 0, dequeuedTruncated = 0;
  // ai: the latest stream's: frames delivered and dequeued, the first and last sequence dequeued, so its gaps are
  // ai: lastSeq - firstSeq + 1 - streamDequeued (what a reader of every frame counts dropped)
  uint64_t streamFrames = 0, streamDequeued = 0;
  int64_t firstSeq = -1, lastSeq = -1;
  bool streaming = false, unplugged = false, eio = false;
  uint32_t fourcc = 0, width = 0, height = 0, num = 0, den = 0;   // ai: the format and interval now
  uint32_t buffers = 0;       // ai: allocated now
  uint32_t mapped = 0;        // ai: mappings not yet unmapped, freed buffers' included
  std::string error;          // ai: a refusal of the device's own (a scene that would not start, no JPEG encoder)
};

class FakeV4l2 {
 public:
  explicit FakeV4l2(const std::vector<FakeDevice>& devices);
  ~FakeV4l2();
  FakeV4l2(const FakeV4l2&) = delete;
  FakeV4l2& operator=(const FakeV4l2&) = delete;

  // ai: The table over these devices (cam::v4l2UseSys); it lives as long as this object.
  const cam::V4l2Sys* sys() const;
  size_t size() const;
  std::string node(size_t i) const;
  std::string byId(size_t i) const;
  // ai: A control's value now (false where the device has none), and one set as another program would leave it
  bool control(size_t i, uint32_t id, int64_t& value) const;
  bool setControl(size_t i, uint32_t id, int64_t value);
  FakeServed served(size_t i) const;

  // ai: The gradient's luma at (x, y) in the frame of this sequence number: a triangle wave 0..254..0 along x + y,
  // ai: moved 7 a frame (a check's truth: one step a pixel, so a 2 x 2 window stays within 1 of its corner)
  static uint8_t gradient(uint64_t seq, uint32_t x, uint32_t y);

  struct Impl;

 private:
  std::unique_ptr<Impl> p_;
};

}  // namespace lizard::sim
