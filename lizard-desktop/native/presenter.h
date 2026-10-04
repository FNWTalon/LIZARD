// ai: The desktop sender's presenter (2026-10-04; its stack, 2026-10-03, for portability and vsync): present.cpp's FIFO
// ai: loop, which held every picture exactly one refresh on the 4090, as a class over any window it is handed (the app's
// ai: AWT canvas through JAWT, lizard_present's own full-screen window).
// ai: One thread of its own does every Vulkan and window call: it opens its own X display connection on X11 (a window id
// ai: is good on any connection, and Xlib stays off AWT's thread), makes the window's surface and a FIFO swapchain, and
// ai: at every refresh clears the area white and blits the picture on screen into it, centred; a new picture is taken
// ai: from the Sender (core/tx) into a mapped staging buffer when one is due (each picture held round(refresh / fps)
// ai: refreshes where that is whole within 1%, else a grid). Vulkan is reached through its own function tables (from its
// ai: own instance and device), never through volk's globals, which the GPU painter's wg reloads for its instance.
#pragma once
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace lizard {

class Sender;

// ai: The window the code goes in: one of them (X11: the drawable's id; Windows: the HWND, built under _WIN32 and not
// ai: run yet).
struct PresentTarget { unsigned long x11Window = 0; void* hwnd = nullptr; };

struct PresentOptions {
  int fps = 60;            // ai: pictures a second asked; each held round(refreshHz / fps) refreshes where that is whole
                           // ai: within 1% (present.cpp's rule), else a grid
  double size = 1.0;       // ai: the code's share of the area's room, 0.25 to 1 (the web's #size)
  bool whole = false;      // ai: whole device pixels a sample, nearest (present.cpp --fit whole); else stretch, linear
  std::string device;      // ai: a substring of a Vulkan device's name, "" the first discrete (LIZ_VK_DEVICE where unset)
  std::function<void(const std::string&)> log;
  std::string csv;         // ai: a CSV line a present (lizard_present --log), "" none
};

class Presenter {
 public:
  // ai: senderLock serializes every Sender call: take() here, configure() on the UI's thread (tx_jni.cpp holds it). The
  // ai: sender and the window must outlive the presenter (stop it first). Null, with why in err, where it cannot start.
  static std::unique_ptr<Presenter> start(const PresentTarget& target, Sender* sender, std::mutex* senderLock,
                                          const PresentOptions& opt, std::string* err);
  void setFps(int fps);
  void setSize(double size);
  void setWhole(bool on);
  // ai: Paused: the last picture stays (no new pictures taken; presents go on a few times a second, uncounted).
  void setPaused(bool on);
  // ai: X11: _NET_WM_BYPASS_COMPOSITOR (1 on, 0 the manager's choice) on the target's top-level window (the ancestor the
  // ai: window manager manages, which carries WM_STATE; else the root's child); a no-op elsewhere.
  void setBypassCompositor(bool on);
  // ai: The last second's window, JSON: shownFps (fresh pictures on screen a second), presentsPerSec, hz (the refresh
  // ai: measured, else the mode's, else 60), modeHz (XRandR's for the monitor holding the window, 0 unknown), held
  // ai: ({"refreshes": pictures}), missed (refreshes the loop missed), behind (pictures due and not yet painted), paused,
  // ai: surface and frame ("WxH": the swapchain's, the picture on screen's), presentWait, device, error, secs (the
  // ai: window's length).
  std::string stats();
  // ai: The whole run's, JSON: secs, pictures, presents, held, missed, behind, unknown (presents with no time), hz, modeHz.
  std::string totals();
  // ai: Stops and joins (again a no-op); the destructor calls it.
  void stop();
  ~Presenter();
  Presenter(const Presenter&) = delete;
  Presenter& operator=(const Presenter&) = delete;

  struct Impl;

 private:
  explicit Presenter(std::unique_ptr<Impl> p);
  std::unique_ptr<Impl> p_;
};

}  // namespace lizard
