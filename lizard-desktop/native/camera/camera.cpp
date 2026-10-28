// ai: The cameras' registry (2026-10-25): every backend's list in camera.h's order, open by an id's prefix, pick's
// ai: rule, the layouts' names. The camera rules each backend applies at start, where the camera offers the control
// ai: (the receiving camera is a machine vision camera, as the Android app's request since 2026-10-06):
// ai:   the ISP's enhancements off: sharpness at its minimum, backlight compensation off (UVC's, and Media
// ai:   Foundation's and AVFoundation's where they reach them);
// ai:   the frame rate held: no exposure priority that lowers it in the dark (V4L2's exposure_dynamic_framerate 0, the
// ai:   others' equivalent);
// ai:   exposure, gain, focus and zoom as the device has them at start, no backend sets them: the camera's own
// ai:   automatic where it runs one, else the value it holds, which on a UVC device may be one another program left
// ai:   (a manual focus stays manual); the user's values (Settings: zoom, exposure, gain) set by the session after
// ai:   each start (rx_session.h set);
// ai:   what start and set changed put back at stop, since a UVC device keeps its controls for the next app (so a
// ai:   user's setting lasts one stream: the session sets the user's values again after each start, rx_session.h set).
#include "camera.h"

#include <cmath>
#include <cstring>

#include "backends.h"

namespace lizard::cam {

const char* pixName(Pix p) {
  switch (p) {
    case Pix::Unknown: return "Unknown";
    case Pix::Grey8: return "Grey8";
    case Pix::Grey16: return "Grey16";
    case Pix::YUYV: return "YUYV";
    case Pix::UYVY: return "UYVY";
    case Pix::YVYU: return "YVYU";
    case Pix::VYUY: return "VYUY";
    case Pix::NV12: return "NV12";
    case Pix::NV21: return "NV21";
    case Pix::I420: return "I420";
    case Pix::YV12: return "YV12";
    case Pix::Bayer8: return "Bayer8";
    case Pix::RGB24: return "RGB24";
    case Pix::BGR24: return "BGR24";
    case Pix::BGRA32: return "BGRA32";
    case Pix::MJPEG: return "MJPEG";
  }
  return "Unknown";
}

std::vector<Info> list(std::vector<std::string>* notes) {
  std::vector<Info> all;
  auto add = [&](std::vector<Info> v) { for (auto& i : v) all.push_back(std::move(i)); };
#if defined(__linux__)
  add(v4l2List(notes));
#elif defined(_WIN32)
  add(mfList(notes));
#elif defined(__APPLE__)
  add(avfList(notes));
#endif
  add(phoneList(notes));
  add(simList(notes));
  return all;
}

// ai: the whole id goes to its backend (the id its list gave), the prefix included; a backend that gives no why for
// ai: a refusal is named by the id
std::unique_ptr<Camera> open(const std::string& id, std::string& why) {
  why.clear();
  auto is = [&](const char* prefix) { return id.compare(0, std::strlen(prefix), prefix) == 0; };
  auto said = [&](std::unique_ptr<Camera> c) {
    if (!c && why.empty()) why = "no camera " + id;
    return c;
  };
#if defined(__linux__)
  if (is("v4l2:")) return said(v4l2Open(id, why));
#elif defined(_WIN32)
  if (is("mf:")) return said(mfOpen(id, why));
#elif defined(__APPLE__)
  if (is("avf:")) return said(avfOpen(id, why));
#endif
  if (is("phone:")) return said(phoneOpen(id, why));
  if (is("sim:")) return said(simOpen(id, why));
  why = "no camera " + id;
  return nullptr;
}

// ai: The rule (camera.h pick), exactly:
// ai:   the candidates are the modes at wantW x wantH (a want of 0 matches any); where none is, those of the size
// ai:   nearest by pixels, |w h - W H| with a want of 0 standing for the mode's own side, a tie to the larger size;
// ai:   a mode's rate at the want is its fps, or the want itself where its range fpsMin .. fps holds the want (the
// ai:   range's bottom where the want is under it); with no want, its fps;
// ai:   with a want, the modes whose rate is at most the want (0.5% over it counted at it: 60.05 is 60) come first,
// ai:   the fastest of them; where none is, the slowest; rates within 0.1% of each other are equal;
// ai:   then the most pixels, then uncompressed before compressed (MJPEG), then the first listed.
int pick(const std::vector<Mode>& modes, uint32_t wantW, uint32_t wantH, double wantFps) {
  if (modes.empty()) return -1;
  std::vector<int> c;
  for (size_t i = 0; i < modes.size(); i++)
    if ((!wantW || modes[i].width == wantW) && (!wantH || modes[i].height == wantH)) c.push_back(int(i));
  auto px = [](const Mode& m) { return double(m.width) * double(m.height); };
  if (c.empty()) {
    auto dist = [&](const Mode& m) {
      return std::fabs(px(m) - double(wantW ? wantW : m.width) * double(wantH ? wantH : m.height));
    };
    size_t near = 0;
    for (size_t i = 1; i < modes.size(); i++) {
      const double d = dist(modes[i]), n = dist(modes[near]);
      if (d < n || (d == n && px(modes[i]) > px(modes[near]))) near = i;
    }
    for (size_t i = 0; i < modes.size(); i++)
      if (modes[i].width == modes[near].width && modes[i].height == modes[near].height) c.push_back(int(i));
  }
  auto rate = [&](const Mode& m) {
    if (wantFps <= 0) return m.fps;
    const double lo = m.fpsMin > 0 ? std::fmin(m.fpsMin, m.fps) : m.fps;
    return wantFps > m.fps ? m.fps : wantFps < lo ? lo : wantFps;
  };
  const double top = wantFps * 1.005;
  auto same = [](double a, double b) { return std::fabs(a - b) <= 0.001 * std::fmax(std::fabs(a), std::fabs(b)); };
  // ai: whether mode a comes before mode b
  auto before = [&](int ia, int ib) {
    const Mode& a = modes[size_t(ia)];
    const Mode& b = modes[size_t(ib)];
    const double ra = rate(a), rb = rate(b);
    if (!same(ra, rb)) {
      if (wantFps <= 0) return ra > rb;
      const bool ua = ra <= top, ub = rb <= top;
      if (ua != ub) return ua;
      return ua ? ra > rb : ra < rb;
    }
    if (px(a) != px(b)) return px(a) > px(b);
    if (a.compressed != b.compressed) return !a.compressed;
    return ia < ib;
  };
  int best = c[0];
  for (size_t k = 1; k < c.size(); k++)
    if (before(c[k], best)) best = c[k];
  return best;
}

}  // namespace lizard::cam
