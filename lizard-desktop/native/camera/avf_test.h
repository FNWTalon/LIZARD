// ai: The AVFoundation backend's parts that need no camera (2026-10-25), declared in plain C++ for cam_check's avf
// ai: check (check_avf.cpp): avf.mm defines them and its capture delegate runs them, so the check holds the code a
// ai: camera's frames go through. macOS only.
#pragma once
#if defined(__APPLE__)
#include <cstdint>
#include <string>
#include <vector>

#include "camera.h"

namespace lizard::cam::avf {

// ai: One pixel buffer (a CVPixelBufferRef) to sink as a Frame with time ns and count seq: the bi-planar 4:2:0
// ai: layouts ('420f', '420v') hand over plane 0 in place, at the buffer's own row pitch; '2vuy' (UYVY) becomes luma
// ai: in scratch (luma.h). The buffer is locked read-only around the sink's call alone. False with why, and no sink
// ai: call, for a layout of another kind or a buffer that does not lock; convertMs: lock to luma, the sink not counted.
bool frame(void* pixelBuffer, int64_t ns, uint64_t seq, std::vector<uint8_t>& scratch, const Camera::Sink& sink,
           double& convertMs, std::string& why);

// ai: A capture's presentation time, value / timescale seconds on the host clock, as the Frame::ns it is given: on
// ai: steadyNs's clock (trigger.h). A timescale of 0 or less (no time) gives steadyNs's now.
int64_t frameNs(int64_t value, int32_t timescale);

// ai: The pixel format start() asks the video data output for, of those the output offers: '420f', else '420v', else
// ai: '2vuy'; 0 where it offers none of the three.
uint32_t pickFormat(const std::vector<uint32_t>& offered);

// ai: A FourCharCode as Mode::format names it ('420v' as "420v"; a code with a byte outside printable ASCII as
// ai: 0x%08x), and whether a device format of that media subtype reaches the host compressed (JPEG, Motion JPEG's
// ai: 'dmb1', H.264, HEVC: the system decodes it before the output's pixel format)
std::string fourcc(uint32_t code);
bool compressed(uint32_t code);

}  // namespace lizard::cam::avf
#endif
