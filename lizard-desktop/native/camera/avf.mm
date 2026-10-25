// ai: AVFoundation's cameras on macOS (2026-10-25), camera.h's interface over AVCaptureDevice, AVCaptureSession and
// ai: AVCaptureVideoDataOutput. Built with ARC (CMakeLists.txt: -fobjc-arc), for macOS 12 on
// ai: (CMAKE_OSX_DEPLOYMENT_TARGET): an API newer than 12 is behind @available.
// ai: List: a discovery session for video over the external cameras (AVCaptureDeviceTypeExternal from macOS 14,
// ai: AVCaptureDeviceTypeExternalUnknown before it, which 14 deprecates) and the built-in ones (wide angle); a
// ai: Continuity Camera iPhone is its own type on 14 and is not asked for. Ids "avf:<uniqueID>", which persist across
// ai: reconnects and reboots; the backend is "uvc" (the system's own camera API, as V4L2 and Media Foundation are).
// ai: Open: the device by its unique id, then the camera permission (TCC): asked where it was never answered, the
// ai: caller blocked until the answer, a minute at most.
// ai: Modes: one a frame-rate range of each device format (fps its top, fpsMin its bottom, key the format's index),
// ai: format the media subtype's FourCharCode ('420v', '2vuy', 'yuvs', 'dmb1' a UVC camera's MJPEG); compressed for
// ai: JPEG, H.264 and HEVC, which the system decodes. Whatever the device sends, the video data output hands over one
// ai: layout it converts to: '420f', else '420v', else '2vuy' (pickFormat), so Mode::pix is NV12 (the Y plane in
// ai: place) or UYVY (luma.h), the same for every mode.
// ai: Start: the format and its rate's frame duration set as active (both frame-duration bounds: the rate held, not
// ai: left to the camera to lower in the dark), in the configuration that adds the input and the output, so they apply
// ai: together at commit, under the device's configuration lock, which is held until stop: macOS has no input-priority
// ai: preset (AVCaptureSessionPresetInputPriority is iOS's, tvOS's and Catalyst's alone), and AVFoundation documents
// ai: the lock as the way to keep every settable property of the device unchanged; a format or rate the session's
// ai: start moved anyway is set again while running. Nothing here persists in the device for the next app, as V4L2's
// ai: controls do, so stop puts nothing back.
// ai: Frames: the output's delegate on a serial queue of the camera's own (user-interactive QoS), late frames
// ai: discarded; each sample buffer's pixel buffer locked read-only around the sink's call (avf::frame). A session's
// ai: runtime error and the device's disconnection end the stream: ended once on the frame queue, nothing sunk after;
// ai: stop releases it.
// ai: Controls: none. AVFoundation on macOS exposes no UVC processing-unit control (sharpness, backlight compensation,
// ai: auto-exposure priority) and no zoom (videoZoomFactor is iOS's); reaching them would take USB control requests
// ai: through IOKit, not tried.
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "avf_test.h"
#include "backends.h"
#include "luma.h"
#include "trigger.h"

namespace lizard::cam {
class AvfCamera;
}

// ai: The video data output's delegate: its callbacks on the camera's frame queue, where alone `cam_` is read or
// ai: cleared (detach), so a callback queued before stop finds it null. lastError: the session's latest runtime error,
// ai: set on the thread that posts it, read by start.
@interface LIZAvfFrames : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@property(atomic, copy) NSString* lastError;
- (instancetype)initWithCamera:(lizard::cam::AvfCamera*)cam;
- (void)detach;
- (void)endWith:(NSString*)why;
@end

namespace lizard::cam {

namespace {

// ai: the frame queue's mark (dispatch_queue_set_specific), its value the camera: stop on that queue cannot wait on it
const char kFrameQueue = 0;

std::string str(NSString* s) {
  const char* c = s ? s.UTF8String : nullptr;
  return c ? std::string(c) : std::string();
}

std::string text(NSError* e) { return e ? str(e.localizedDescription) : std::string("no reason given"); }

std::vector<uint32_t> offered(AVCaptureVideoDataOutput* o) {
  std::vector<uint32_t> v;
  for (NSNumber* n in o.availableVideoCVPixelFormatTypes) v.push_back(n.unsignedIntValue);
  return v;
}

NSArray<AVCaptureDevice*>* devicesNow() {
  NSMutableArray<AVCaptureDeviceType>* types = [NSMutableArray array];
  if (@available(macOS 14.0, *)) {
    [types addObject:AVCaptureDeviceTypeExternal];
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [types addObject:AVCaptureDeviceTypeExternalUnknown];
#pragma clang diagnostic pop
  }
  [types addObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
  AVCaptureDeviceDiscoverySession* s =
      [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:types
                                                             mediaType:AVMediaTypeVideo
                                                              position:AVCaptureDevicePositionUnspecified];
  return s.devices;
}

Info infoOf(AVCaptureDevice* d) {
  Info i;
  i.id = "avf:" + str(d.uniqueID);
  i.name = str(d.localizedName);
  i.backend = "uvc";
  i.detail = str(d.modelID);   // ai: a UVC camera's reads "UVC Camera VendorID_<vid> ProductID_<pid>"
  return i;
}

// ai: The camera permission. An app's bundle whose Info.plist names no NSCameraUsageDescription is stopped by the
// ai: system at the request (AVFoundation: "raises an exception"), so that is said instead; a bare executable
// ai: (lizard_receive, cam_check, Gradle's java) is asked for by the app it runs under, a terminal, as TCC attributes
// ai: it (not tried). The completion runs on a queue of the system's, so a caller on any thread can wait for it.
bool allowed(std::string& why) {
  const AVAuthorizationStatus s = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
  if (s == AVAuthorizationStatusAuthorized) return true;
  if (s == AVAuthorizationStatusNotDetermined) {
    NSBundle* b = [NSBundle mainBundle];
    if ([b.bundlePath hasSuffix:@".app"] && ![b objectForInfoDictionaryKey:@"NSCameraUsageDescription"]) {
      why = "This build cannot ask for the camera: its Info.plist has no NSCameraUsageDescription.";
      return false;
    }
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block BOOL granted = NO;
    [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                             completionHandler:^(BOOL ok) {
                               granted = ok;
                               dispatch_semaphore_signal(done);
                             }];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 60 * static_cast<int64_t>(NSEC_PER_SEC))) != 0) {
      why = "The camera's permission was not answered.";
      return false;
    }
    if (granted) return true;
  }
  why = "The camera is blocked: allow LIZARD in System Settings, Privacy & Security, Camera.";
  return false;
}

}  // namespace

// ai: The parts that need no camera (avf_test.h), check_avf.cpp's.
namespace avf {

namespace {

// ai: the host clock in ns: CMClockGetHostTimeClock, mach_absolute_time's (it stops while the Mac sleeps)
int64_t hostNs() {
  const CMTime t = CMClockGetTime(CMClockGetHostTimeClock());
  return CMTimeConvertScale(t, 1000000000, kCMTimeRoundingMethod_RoundHalfAwayFromZero).value;
}

// ai: steadyNs's clock less the host clock, the tightest of four back-to-back brackets (the reads tens of ns apart, so
// ai: a preempted one is outvoted). steadyNs is std::chrono::steady_clock (trigger.h), which Apple's libc++ reads from
// ai: CLOCK_MONOTONIC_RAW, mach_continuous_time's: it runs on while the Mac sleeps, so the two differ by the sleep
// ai: since boot. Read at each frame, so a sleep while streaming moves it too.
int64_t hostToSteady() {
  int64_t best = 0, span = INT64_MAX;
  for (int i = 0; i < 4; i++) {
    const int64_t a = hostNs(), s = steadyNs(), b = hostNs();
    if (b - a < span) {
      span = b - a;
      best = s - (a + (b - a) / 2);
    }
  }
  return best;
}

}  // namespace

bool frame(void* pixelBuffer, int64_t ns, uint64_t seq, std::vector<uint8_t>& scratch, const Camera::Sink& sink,
           double& convertMs, std::string& why) {
  CVPixelBufferRef pb = static_cast<CVPixelBufferRef>(pixelBuffer);
  if (!pb) {
    why = "the sample carries no pixel buffer";
    return false;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const OSType type = CVPixelBufferGetPixelFormatType(pb);
  const bool biPlanar =
      type == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || type == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  if (!biPlanar && type != kCVPixelFormatType_422YpCbCr8) {
    why = "pixel format " + fourcc(type) + " is not one the receiver takes";
    return false;
  }
  if (CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
    why = "the frame's pixel buffer did not lock";
    return false;
  }
  Frame f;
  f.ns = ns;
  f.seq = seq;
  bool ok = false;
  if (biPlanar) {
    // ai: plane 0 is the Y plane, its own width, height and pitch; '420v' carries Y in 16..235, which the decoders take
    // ai: as they take any contrast
    if (CVPixelBufferIsPlanar(pb) && CVPixelBufferGetPlaneCount(pb) >= 1) {
      f.luma = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(pb, 0));
      f.width = static_cast<uint32_t>(CVPixelBufferGetWidthOfPlane(pb, 0));
      f.height = static_cast<uint32_t>(CVPixelBufferGetHeightOfPlane(pb, 0));
      f.stride = static_cast<uint32_t>(CVPixelBufferGetBytesPerRowOfPlane(pb, 0));
      ok = f.luma && f.width && f.height && f.stride >= f.width;
      if (!ok) why = "the frame's Y plane is empty";
    } else {
      why = "a bi-planar frame without its planes";
    }
  } else {
    // ai: '2vuy' is Cb Y0 Cr Y1 (camera.h's UYVY), rows CVPixelBufferGetBytesPerRow apart
    const auto* base = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const auto w = static_cast<uint32_t>(CVPixelBufferGetWidth(pb));
    const auto h = static_cast<uint32_t>(CVPixelBufferGetHeight(pb));
    const size_t pitch = CVPixelBufferGetBytesPerRow(pb);
    if (!base) why = "the frame has no pixels";
    else ok = toLuma(Pix::UYVY, 8, base, pitch * h, w, h, static_cast<uint32_t>(pitch), scratch, &why);
    if (ok) {
      f.luma = scratch.data();
      f.width = w;
      f.height = h;
      f.stride = w;
    }
  }
  if (ok) {
    convertMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sink(f);
  }
  CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
  return ok;
}

// ai: The presentation time is on the session's synchronization clock ("all capture output sample buffer timestamps
// ai: are on the synchronization clock's timebase"): for a session of one camera the host clock, or a device clock
// ai: CoreMediaIO keeps on it; either way one clock for the stream, then moved onto steadyNs's (hostToSteady).
int64_t frameNs(int64_t value, int32_t timescale) {
  if (timescale <= 0) return steadyNs();
  const CMTime t =
      CMTimeConvertScale(CMTimeMake(value, timescale), 1000000000, kCMTimeRoundingMethod_RoundHalfAwayFromZero);
  if (!CMTIME_IS_NUMERIC(t)) return steadyNs();
  return t.value + hostToSteady();
}

uint32_t pickFormat(const std::vector<uint32_t>& offered) {
  // ai: full range first: the Y bytes as the sensor's tone gave them; video range squeezes them into 16..235
  static const uint32_t kWant[] = {kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                   kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_422YpCbCr8};
  for (uint32_t w : kWant)
    if (std::find(offered.begin(), offered.end(), w) != offered.end()) return w;
  return 0;
}

std::string fourcc(uint32_t code) {
  const char c[4] = {static_cast<char>(code >> 24), static_cast<char>(code >> 16), static_cast<char>(code >> 8),
                     static_cast<char>(code)};
  for (char x : c)
    if (static_cast<unsigned char>(x) < 0x20 || static_cast<unsigned char>(x) > 0x7e) {
      char h[12];
      std::snprintf(h, sizeof h, "0x%08x", static_cast<unsigned>(code));
      return h;
    }
  return std::string(c, 4);
}

bool compressed(uint32_t code) {
  return code == kCMVideoCodecType_JPEG || code == kCMVideoCodecType_JPEG_OpenDML || code == kCMVideoCodecType_H264 ||
         code == kCMVideoCodecType_HEVC;
}

}  // namespace avf

class AvfCamera final : public Camera {
 public:
  AvfCamera(AVCaptureDevice* dev, Info info);
  ~AvfCamera() override;
  AvfCamera(const AvfCamera&) = delete;
  AvfCamera& operator=(const AvfCamera&) = delete;

  const Info& info() const override { return info_; }
  std::vector<Mode> modes() override;
  std::vector<Control> controls() override { return {}; }
  bool set(const std::string& control, double value, bool automatic, std::string& why) override;
  bool start(const Mode& mode, Sink sink, Ended ended, std::string& why) override;
  void stop() override;
  Stats stats() override;

  // ai: the delegate's, on the frame queue
  void sample(CMSampleBufferRef sb);
  void dropped();
  void end(const std::string& why);

 private:
  // ai: a mode's device format and the frame duration of its rate (its range's minimum, the top rate's exact period)
  struct Fmt {
    AVCaptureDeviceFormat* format = nil;
    CMTime duration{};
  };
  Fmt find(const Mode& m);
  bool begin(const Mode& mode, Sink sink, Ended ended, std::string& why);
  std::string configure(const Fmt& f, uint32_t pixelFormat);
  void observe();
  std::string run(const Fmt& f);
  void teardown();
  bool onQueue() const { return dispatch_get_specific(&kFrameQueue) == this; }

  AVCaptureDevice* dev_;
  Info info_;
  dispatch_queue_t q_;                        // ai: the frame queue: every delegate callback, serially
  // ai: from here to locked_: the session's thread alone (start, stop)
  AVCaptureSession* session_ = nil;
  AVCaptureDeviceInput* input_ = nil;
  AVCaptureVideoDataOutput* output_ = nil;
  LIZAvfFrames* del_ = nil;
  id<NSObject> errTok_ = nil, goneTok_ = nil;  // ai: the notification observers
  bool locked_ = false;                       // ai: the device's configuration lock held
  Sink sink_;                                 // ai: set before the delegate is, cleared after it is let go
  Ended ended_;
  std::vector<uint8_t> scratch_;              // ai: the frame queue's: '2vuy' frames' luma
  std::mutex mu_;
  bool started_ = false;                      // ai: mu_: start returned true
  std::string endWhy_;                        // ai: mu_: why the stream ended before start returned
  std::atomic<bool> over_{false};             // ai: ended or stopping: nothing more is sunk
  std::atomic<uint64_t> seq_{0}, frames_{0}, dropped_{0}, corrupt_{0}, convNs_{0};
  std::atomic<double> fps_{0};                // ai: the rate the device took at start (Stats::fps)
};

}  // namespace lizard::cam

@implementation LIZAvfFrames {
  lizard::cam::AvfCamera* cam_;
}

- (instancetype)initWithCamera:(lizard::cam::AvfCamera*)cam {
  if ((self = [super init])) cam_ = cam;
  return self;
}

- (void)detach {
  cam_ = nullptr;
}

- (void)endWith:(NSString*)why {
  if (cam_) cam_->end(lizard::cam::str(why));
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  @autoreleasepool {
    if (cam_) cam_->sample(sampleBuffer);
  }
}

- (void)captureOutput:(AVCaptureOutput*)output
    didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
         fromConnection:(AVCaptureConnection*)connection {
  if (cam_) cam_->dropped();
}

@end

namespace lizard::cam {

AvfCamera::AvfCamera(AVCaptureDevice* dev, Info info) : dev_(dev), info_(std::move(info)) {
  dispatch_queue_attr_t a =
      dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
  q_ = dispatch_queue_create("dev.lizard.avf.frames", a);
  dispatch_queue_set_specific(q_, &kFrameQueue, this, nullptr);
}

AvfCamera::~AvfCamera() { stop(); }

std::vector<Mode> AvfCamera::modes() {
  std::vector<Mode> out;
  @autoreleasepool {
    // ai: the layout start will ask for, from an output not yet in a session: AVFoundation's converter offers the same
    // ai: ones whatever the device sends (its list may change with the active format; start asks again)
    const uint32_t pf = avf::pickFormat(offered([[AVCaptureVideoDataOutput alloc] init]));
    const Pix pix = pf == kCVPixelFormatType_422YpCbCr8 ? Pix::UYVY : pf ? Pix::NV12 : Pix::Unknown;
    NSArray<AVCaptureDeviceFormat*>* fs = dev_.formats;
    for (NSUInteger i = 0; i < fs.count; i++) {
      AVCaptureDeviceFormat* f = fs[i];
      if (![f.mediaType isEqualToString:AVMediaTypeVideo]) continue;
      const CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
      if (d.width <= 0 || d.height <= 0) continue;
      const uint32_t sub = CMFormatDescriptionGetMediaSubType(f.formatDescription);
      for (AVFrameRateRange* r in f.videoSupportedFrameRateRanges) {
        Mode m;
        m.width = static_cast<uint32_t>(d.width);
        m.height = static_cast<uint32_t>(d.height);
        m.fps = r.maxFrameRate;
        m.fpsMin = r.minFrameRate;
        m.format = avf::fourcc(sub);
        m.pix = pix;
        m.compressed = avf::compressed(sub);
        m.key = i;
        // ai: a format listed twice (the same size, subtype and rates, differing in what a receiver does not read)
        // ai: is one mode, the first's
        const bool seen = std::any_of(out.begin(), out.end(), [&](const Mode& o) {
          return o.width == m.width && o.height == m.height && o.fps == m.fps && o.fpsMin == m.fpsMin &&
                 o.format == m.format;
        });
        if (!seen) out.push_back(m);
      }
    }
  }
  return out;
}

bool AvfCamera::set(const std::string& control, double, bool, std::string& why) {
  why = control + ": AVFoundation gives no control of this camera on macOS";
  return false;
}

AvfCamera::Fmt AvfCamera::find(const Mode& m) {
  Fmt out;
  NSArray<AVCaptureDeviceFormat*>* fs = dev_.formats;
  auto fits = [&](AVCaptureDeviceFormat* f) {
    if (![f.mediaType isEqualToString:AVMediaTypeVideo]) return false;
    const CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
    if (static_cast<uint32_t>(d.width) != m.width || static_cast<uint32_t>(d.height) != m.height ||
        avf::fourcc(CMFormatDescriptionGetMediaSubType(f.formatDescription)) != m.format)
      return false;
    for (AVFrameRateRange* r in f.videoSupportedFrameRateRanges)
      if (std::fabs(r.maxFrameRate - m.fps) < 0.01) {
        out.format = f;
        out.duration = r.minFrameDuration;
        return true;
      }
    return false;
  };
  // ai: the format at the mode's key, else (the device's list changed since modes()) the first that matches it
  if (m.key < fs.count && fits(fs[m.key])) return out;
  for (AVCaptureDeviceFormat* f in fs)
    if (fits(f)) return out;
  return Fmt{};
}

bool AvfCamera::start(const Mode& mode, Sink sink, Ended ended, std::string& why) {
  stop();
  bool ok = false;
  @autoreleasepool {
    ok = begin(mode, std::move(sink), std::move(ended), why);
  }
  return ok;
}

bool AvfCamera::begin(const Mode& mode, Sink sink, Ended ended, std::string& why) {
  const Fmt f = find(mode);
  if (!f.format) {
    char rate[32];
    std::snprintf(rate, sizeof rate, "%.2f", mode.fps);
    why = "the camera has no mode " + std::to_string(mode.width) + " x " + std::to_string(mode.height) + " " +
          mode.format + " at " + rate + " fps";
    return false;
  }
  AVCaptureVideoDataOutput* out = [[AVCaptureVideoDataOutput alloc] init];
  const uint32_t pf = avf::pickFormat(offered(out));
  if (!pf) {
    why = "the camera's frames come in no layout the receiver takes ('420f', '420v' or '2vuy')";
    return false;
  }
  NSError* err = nil;
  AVCaptureDeviceInput* in = [AVCaptureDeviceInput deviceInputWithDevice:dev_ error:&err];
  if (!in) {
    why = "the camera did not open: " + text(err);
    return false;
  }
  sink_ = std::move(sink);
  ended_ = std::move(ended);
  seq_ = 0;
  frames_ = 0;
  dropped_ = 0;
  corrupt_ = 0;
  convNs_ = 0;
  fps_ = 0;
  over_ = false;
  {
    std::lock_guard<std::mutex> l(mu_);
    started_ = false;
    endWhy_.clear();
  }
  session_ = [[AVCaptureSession alloc] init];
  input_ = in;
  output_ = out;
  del_ = [[LIZAvfFrames alloc] initWithCamera:this];
  std::string bad;
  // ai: a device refusing a format or a rate raises an exception (NSInvalidArgumentException, the frame durations'
  // ai: documented case); it is the start's why, not the JVM's end
  @try {
    bad = configure(f, pf);
    if (bad.empty()) {
      observe();
      bad = run(f);
    }
  } @catch (NSException* e) {
    bad = "the camera refused the mode: " + str(e.reason);
  }
  if (bad.empty()) {
    std::lock_guard<std::mutex> l(mu_);
    if (over_) bad = endWhy_.empty() ? std::string("the camera stopped as it started") : endWhy_;
    else started_ = true;
  }
  if (!bad.empty()) {
    teardown();
    why = bad;
    return false;
  }
  return true;
}

std::string AvfCamera::configure(const Fmt& f, uint32_t pixelFormat) {
  AVCaptureSession* s = session_;
  std::string bad;
  [s beginConfiguration];
  if ([s canAddInput:input_]) [s addInput:input_];
  else bad = "the camera could not join a capture session";
  if (bad.empty()) {
    output_.videoSettings = @{(__bridge NSString*)kCVPixelBufferPixelFormatTypeKey : @(pixelFormat)};
    output_.alwaysDiscardsLateVideoFrames = YES;
    [output_ setSampleBufferDelegate:del_ queue:q_];
    if ([s canAddOutput:output_]) [s addOutput:output_];
    else bad = "the capture session took no video output";
  }
  if (bad.empty()) {
    NSError* err = nil;
    locked_ = [dev_ lockForConfiguration:&err];
    if (!locked_) bad = "the camera's mode could not be set: " + text(err);
  }
  if (bad.empty()) {
    // ai: the format first: setting it puts the frame durations back to its defaults, and macOS 15's automatic video
    // ai: frame rate (which lowers the rate in low light, and makes setting a duration throw) back to off
    dev_.activeFormat = f.format;
    dev_.activeVideoMinFrameDuration = f.duration;
    dev_.activeVideoMaxFrameDuration = f.duration;
  }
  [s commitConfiguration];
  return bad;
}

void AvfCamera::observe() {
  // ai: The blocks hold the delegate and the queue, never the camera: a notification in flight while stop removes its
  // ai: observer reaches a delegate let go (its camera null) on a queue that still exists
  NSNotificationCenter* nc = [NSNotificationCenter defaultCenter];
  LIZAvfFrames* del = del_;
  dispatch_queue_t q = q_;
  errTok_ = [nc addObserverForName:AVCaptureSessionRuntimeErrorNotification
                            object:session_
                             queue:nil
                        usingBlock:^(NSNotification* n) {
                          id e = n.userInfo[AVCaptureSessionErrorKey];
                          NSString* reason = @"no reason given";
                          if ([e isKindOfClass:[NSError class]]) reason = [(NSError*)e localizedDescription];
                          NSString* why = [@"the capture session failed: " stringByAppendingString:reason];
                          del.lastError = why;
                          dispatch_async(q, ^{
                            [del endWith:why];
                          });
                        }];
  goneTok_ = [nc addObserverForName:AVCaptureDeviceWasDisconnectedNotification
                             object:dev_
                              queue:nil
                         usingBlock:^(NSNotification* n) {
                           dispatch_async(q, ^{
                             [del endWith:@"the camera was disconnected"];
                           });
                         }];
}

std::string AvfCamera::run(const Fmt& f) {
  // ai: synchronous: it returns once the session runs, or has failed with a runtime error posted
  [session_ startRunning];
  if (!session_.running) {
    NSString* e = del_.lastError;
    return e ? str(e) : std::string("the capture session did not start");
  }
  if (![dev_.activeFormat isEqual:f.format]) dev_.activeFormat = f.format;
  if (CMTimeCompare(dev_.activeVideoMinFrameDuration, f.duration) != 0) dev_.activeVideoMinFrameDuration = f.duration;
  if (CMTimeCompare(dev_.activeVideoMaxFrameDuration, f.duration) != 0) dev_.activeVideoMaxFrameDuration = f.duration;
  // ai: the rate the device runs at now (Stats::fps), from its frame duration as it reads back
  const CMTime d = dev_.activeVideoMinFrameDuration;
  fps_ = CMTIME_IS_NUMERIC(d) && d.value > 0 ? static_cast<double>(d.timescale) / static_cast<double>(d.value) : 0.0;
  return {};
}

void AvfCamera::stop() {
  @autoreleasepool {
    teardown();
  }
}

void AvfCamera::teardown() {
  if (!session_) return;
  // ai: Nothing is sunk from here on, even by a callback already queued; then the session stopped (synchronous),
  // ai: the delegate let go and the frame queue drained, so no sink call outlives stop. On the frame queue itself
  // ai: (stop from inside sink or ended, which camera.h's threads rule out) there is nothing queued ahead to wait for.
  over_ = true;
  NSNotificationCenter* nc = [NSNotificationCenter defaultCenter];
  if (errTok_) [nc removeObserver:errTok_];
  if (goneTok_) [nc removeObserver:goneTok_];
  errTok_ = nil;
  goneTok_ = nil;
  [session_ stopRunning];
  [output_ setSampleBufferDelegate:nil queue:nil];
  LIZAvfFrames* del = del_;
  const bool here = onQueue();
  if (here) {
    [del detach];
  } else {
    dispatch_sync(q_, ^{
      [del detach];
    });
  }
  if (locked_) {
    [dev_ unlockForConfiguration];
    locked_ = false;
  }
  if ([session_.outputs containsObject:output_]) [session_ removeOutput:output_];
  if ([session_.inputs containsObject:input_]) [session_ removeInput:input_];
  session_ = nil;
  input_ = nil;
  output_ = nil;
  del_ = nil;
  {
    std::lock_guard<std::mutex> l(mu_);
    started_ = false;
  }
  // ai: a sink or ended running now (stop from inside one) is not destroyed under itself
  if (!here) {
    sink_ = nullptr;
    ended_ = nullptr;
  }
}

Stats AvfCamera::stats() {
  Stats s;
  s.frames = frames_;
  s.dropped = dropped_;
  s.corrupt = corrupt_;
  s.convertMs = s.frames ? static_cast<double>(convNs_) / 1e6 / static_cast<double>(s.frames) : 0;
  s.fps = fps_;
  return s;
}

void AvfCamera::sample(CMSampleBufferRef sb) {
  const uint64_t seq = seq_++;
  if (over_ || !sink_) return;
  CVImageBufferRef image = CMSampleBufferGetImageBuffer(sb);
  const CMTime t = CMSampleBufferGetPresentationTimeStamp(sb);
  const int64_t ns = CMTIME_IS_NUMERIC(t) ? avf::frameNs(t.value, t.timescale) : steadyNs();
  double ms = 0;
  std::string why;
  if (image && avf::frame(image, ns, seq, scratch_, sink_, ms, why)) {
    frames_++;
    convNs_ += static_cast<uint64_t>(ms * 1e6);
  } else {
    corrupt_++;
  }
}

void AvfCamera::dropped() {
  seq_++;
  if (!over_) dropped_++;
}

void AvfCamera::end(const std::string& why) {
  if (over_.exchange(true)) return;
  {
    std::lock_guard<std::mutex> l(mu_);
    if (!started_) {
      endWhy_ = why;
      return;
    }
  }
  if (ended_) ended_(why);
}

std::vector<Info> avfList(std::vector<std::string>* notes) {
  std::vector<Info> out;
  @autoreleasepool {
    for (AVCaptureDevice* d in devicesNow()) out.push_back(infoOf(d));
  }
  return out;
}

std::unique_ptr<Camera> avfOpen(const std::string& cid, std::string& why) {
  std::unique_ptr<Camera> cam;
  @autoreleasepool {
    AVCaptureDevice* d = nil;
    if (cid.compare(0, 4, "avf:") == 0) {
      NSString* uid = [NSString stringWithUTF8String:cid.c_str() + 4];
      if (uid) d = [AVCaptureDevice deviceWithUniqueID:uid];
    }
    if (!d) why = "the camera is not connected: " + cid;
    else if (allowed(why)) cam = std::make_unique<AvfCamera>(d, infoOf(d));
  }
  return cam;
}

}  // namespace lizard::cam
