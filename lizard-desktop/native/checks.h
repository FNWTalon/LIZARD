// ai: cam_check's checks (2026-10-25), one file each beside the part it holds to account: check_luma.cpp (the
// ai: conversions, TurboJPEG, the trigger's clock), check_scene.cpp (the simulated screen and camera), check_session.cpp
// ai: (the session over the receiver, on the simulated camera), check_v4l2.cpp (V4L2 against sim/fake_v4l2's devices),
// ai: check_mf.cpp (Media Foundation against sim/fake_mf's source), check_avf.cpp (AVFoundation). Each takes the
// ai: arguments after its name and returns the process's exit code, one rule for all (kHeld and the rest below); each
// ai: case a line "ok <case>: ...", "FAIL <case>: ..." or "skip <case>: <why>" on stdout.
#pragma once
#include <string>
#include <vector>

namespace lizard::check {

// ai: A check's exit code: kHeld every case held; kFailed one did not; kUsage an argument the check does not take (a
// ai: case it has not, or any argument to a check that takes none); kNotHere the check is another system's (v4l2 off
// ai: Linux, mf off Windows, avf off macOS), 77, the code test harnesses (Automake's, Meson's) read as skipped
// ai: (2026-10-25: 2 had meant both a usage error and not here, so `cam_check all clean` read a session refusing its
// ai: argument as "not on this system").
constexpr int kHeld = 0, kFailed = 1, kUsage = 2, kNotHere = 77;

// ai: Why the MJPEG cases cannot run here where the receiver does without TurboJPEG (2026-10-25, check_luma.cpp):
// ai: on Windows and macOS Media Foundation and AVFoundation decode a camera's MJPEG themselves (camera/jpeg.h), so
// ai: where no TurboJPEG loads the checks' MJPEG cases skip with load's words and this; "" where it loads, on Linux
// ai: (the receiver decodes V4L2's MJPEG with it: there its absence fails them) and wherever LIZ_TURBOJPEG names a
// ai: library (one asked for must load).
std::string jpegOptional();

using Args = std::vector<std::string>;
int luma(const Args& a);
int trigger(const Args& a);
int scene(const Args& a);
int session(const Args& a);
int v4l2(const Args& a);
int mf(const Args& a);
int avf(const Args& a);

}  // namespace lizard::check
