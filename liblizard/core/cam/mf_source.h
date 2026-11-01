// ai: Media Foundation's hooks beside mfList and mfOpen (2026-10-25): a camera over media sources the caller makes, so
// ai: a source of its own (sim/fake_mf.cpp's) runs the backend's whole path (the source reader, Media Foundation's
// ai: MJPEG decoder, the capture thread, the controls); and such a source listed under an id of its own, so the
// ai: registry and the session open it as they open a device. Windows only.
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "camera.h"

struct IMFMediaSource;

namespace lizard::cam {

// ai: Media Foundation's DLLs loaded and MFStartup called (mf.cpp): false with why where they are absent (Windows N
// ai: without the Media Feature Pack, a Server without the feature) or refuse.
bool mfReady(std::string* why);

// ai: A new source with one reference, which the caller takes; null with why where it cannot be made.
using MakeSource = std::function<IMFMediaSource*(std::string& why)>;

// ai: A camera over the sources make returns, described by info: the first made at once, a new one for each later
// ai: stream (and a set on a stopped camera), since a stream's source reader shuts its source down when it goes (a
// ai: device's source is activated anew the same way); a look at the modes or controls after a stop makes none. Null
// ai: with why where Media Foundation is absent, make fails or its source has no video stream.
std::unique_ptr<Camera> mfOpenSource(MakeSource make, const Info& info, std::string& why);

// ai: A source listed by mfList as "mf:test:<name>" (name label, detail "test source") and opened by mfOpen through
// ai: make; make empty takes the name off the list. The process's own (cam_check's), never the app's.
void mfTestSource(const std::string& name, const std::string& label, MakeSource make);

}  // namespace lizard::cam
