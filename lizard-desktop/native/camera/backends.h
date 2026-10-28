// ai: The backends behind camera.h (2026-10-25), each in its own file, each built where its system is: list and open
// ai: for the ids of its prefix (camera.h Info::id), the registry in camera.cpp calling them in this order. A backend
// ai: that cannot look (its library absent, its system service refusing) returns no cameras and says why in notes.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "camera.h"

namespace lizard::cam {

#if defined(__linux__)
// ai: v4l2.cpp: UVC and every other V4L2 capture device (v4l2:)
std::vector<Info> v4l2List(std::vector<std::string>* notes);
std::string v4l2Stamp();   // ai: the nodes it would probe (camera.h listStamp)
std::unique_ptr<Camera> v4l2Open(const std::string& id, std::string& why);
#endif
#if defined(_WIN32)
// ai: mf.cpp: Media Foundation's video capture devices (mf:), its DLLs loaded at run time
std::vector<Info> mfList(std::vector<std::string>* notes);
std::unique_ptr<Camera> mfOpen(const std::string& id, std::string& why);
#endif
#if defined(__APPLE__)
// ai: avf.mm: AVFoundation's video devices (avf:)
std::vector<Info> avfList(std::vector<std::string>* notes);
std::unique_ptr<Camera> avfOpen(const std::string& id, std::string& why);
#endif
// ai: phone.cpp: a phone streaming its frames over TCP (phone:<ipv4>:<port>, 2026-10-27; liblizard/core/rx/stream.h),
// ai: listed while its beacon is heard, or over USB as an Android Open Accessory (phone:usb:<place>, aoa.h); every system
std::vector<Info> phoneList(std::vector<std::string>* notes);
std::string phoneStamp();   // ai: the beacons heard and the USB devices (camera.h listStamp)
std::unique_ptr<Camera> phoneOpen(const std::string& id, std::string& why);
// ai: upload.cpp: a replay played as a camera (upload:<path>: a .lizard, or a run's folder; 2026-10-28), never listed:
// ai: the receiver names the file; every system
std::unique_ptr<Camera> uploadOpen(const std::string& id, std::string& why);
// ai: sim.cpp: the simulated camera (sim:0), listed only where LIZ_CAMERA_SIM is set; its value is sim/scene.h parse's
// ai: spec, plus the camera's own keys: camfps (its frame rate, 60), camppm (its clock against the host's, 0), format
// ai: (the Pix its frames come in, grey by default: yuyv, nv12, mjpeg exercise luma.h and jpeg.h)
std::vector<Info> simList(std::vector<std::string>* notes);
std::unique_ptr<Camera> simOpen(const std::string& id, std::string& why);

}  // namespace lizard::cam
