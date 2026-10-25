// ai: The desktop sender's presenter (presenter.h): present.cpp's swapchain, slots, waits and tallies (2026-10-03) as a
// ai: class with one thread of its own.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include "presenter.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

// ai: Vulkan loaded at run time through volk, as the engine's wg (liblizard/core/third_party/volk): the loader's entry
// ai: points from it, everything else from this presenter's own instance and device (the tables below)
#include "volk.h"

#include "sender.h"
#include "wg.h"

#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif LIZ_X11
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xrandr.h>
#include <X11/extensions/shape.h>
#include <vulkan/vulkan_xlib.h>
#elif defined(__APPLE__)
#include <vulkan/vulkan_metal.h>

#include "macos.h"
#endif

namespace lizard {

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void vkCheck(VkResult r, const char* what) {
  if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed (VkResult " + std::to_string(static_cast<int>(r)) + ")");
}

// ai: a string's text fit for JSON (a driver's message may hold quotes or a backslash)
std::string esc(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if (static_cast<unsigned char>(ch) >= 0x20) o += ch;
  }
  return o;
}

// ai: a number as the JSON and the CSV want it whatever the process's locale: the JVM sets LC_NUMERIC to the user's,
// ai: and a decimal-comma locale (de_DE, fr_FR, en_DK) made the stats JSON {"shownFps":54,0}, which the app could not
// ai: read, so the rate line stayed empty (2026-10-04). A %.Nf conversion differs from C's only in that mark.
std::string fmt(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  for (char* q = b; *q; q++) if (*q == ',') *q = '.';
  return b;
}

// ai: The functions this presenter calls, fetched from the sender's instance and device (wg's, which made them with
// ai: Presenter::deviceExtras()): its own tables, since volk's globals are wg's and loaded for its own use.
#define LIZ_INST_FNS(X)                                                                                                \
  X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) X(vkGetDeviceProcAddr)                 \
  X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)          \
  X(vkGetPhysicalDeviceSurfaceFormatsKHR)
#define LIZ_DEV_FNS(X)                                                                                                 \
  X(vkQueueWaitIdle) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR)                                                   \
  X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkCreateSemaphore)      \
  X(vkDestroySemaphore) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateCommandPool)  \
  X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkResetCommandBuffer)                  \
  X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdClearColorImage) X(vkCmdBlitImage)      \
  X(vkAllocateMemory) X(vkFreeMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)               \
  X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateShaderModule) X(vkDestroyShaderModule)     \
  X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
  X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool)               \
  X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCmdBindPipeline)                  \
  X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch)
#define LIZ_DECL(f) PFN_##f f = nullptr;
struct InstFns {
  LIZ_INST_FNS(LIZ_DECL)
};
struct DevFns {
  LIZ_DEV_FNS(LIZ_DECL)
  PFN_vkWaitForPresentKHR vkWaitForPresentKHR = nullptr;   // ai: where the device has present_wait
};
#undef LIZ_DECL

// ai: The swapchain: FIFO (each present one refresh, the queue's back-pressure the loop's pace), 8 images where the
// ai: surface allows (the presents queued up to 7 refreshes ahead of the screen, 2026-10-07; 3 before), an 8-bit UNORM
// ai: format so a grey level reaches the screen as painted, written by transfers (a clear and a blit).
struct Chain {
  VkSwapchainKHR sc = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  std::vector<VkImage> images;
  std::vector<VkSemaphore> done;   // ai: per image: its commands finished, which its present waits on
};

// ai: A picture's home on the device: an RGBA8 image the expand kernel fills from the sender's ring (grey on the
// ai: device, no host copy; 2026-10-07) once, and the blit reads at every refresh it is held. Three, so the one being
// ai: filled is never one in flight; each remade at the frame's size when a re-pick changes it.
struct Slot {
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory imgMem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  int w = 0, h = 0;
  int64_t usedBy = -2;   // ai: the last frame whose commands read or wrote it
};

// ai: one present: its id, the picture it carried (the sender's count, whose mod 4 the pilots paint), whether that
// ai: picture was new with it, when (ms from the start) its acquire returned, it was submitted and it was on screen (-1:
// ai: not known), and its epoch (a new one after a pause or a new swapchain: no gap is counted across it)
struct Shown {
  uint64_t id = 0, picture = 0;
  bool fresh = false;
  double acquiredMs = 0, submittedMs = 0, shownMs = -1;
  uint32_t epoch = 0;
  int queued = -1;   // ai: presents from the last one on screen to this one at its submit (the chain's slack); -1 unknown
};

// ai: The second thread that waits for each present to reach the screen (vkWaitForPresentKHR), in id order. Without
// ai: present_wait a present's time is its acquire's.
struct Waiter {
  const DevFns* D = nullptr;
  VkDevice dev = VK_NULL_HANDLE;
  bool wait = false;
  VkSwapchainKHR sc = VK_NULL_HANDLE;
  double t0 = 0;
  std::atomic<uint64_t> shownId{0};   // ai: the last present id seen on screen
  std::mutex m;
  std::condition_variable cv;
  std::deque<Shown> todo;
  std::vector<Shown> done;
  bool stop = false;
  std::thread t;

  void start(const DevFns& fns, VkDevice d, bool presentWait, VkSwapchainKHR s, double origin) {
    D = &fns; dev = d; wait = presentWait; sc = s; t0 = origin;
    {
      std::lock_guard<std::mutex> l(m);
      stop = false;
    }
    if (wait) t = std::thread([this] { run(); });
  }
  void push(Shown s) {
    if (!wait) {
      s.shownMs = s.acquiredMs;
      std::lock_guard<std::mutex> l(m);
      done.push_back(s);
      return;
    }
    {
      std::lock_guard<std::mutex> l(m);
      todo.push_back(s);
    }
    cv.notify_one();
  }
  std::vector<Shown> take() {
    std::lock_guard<std::mutex> l(m);
    std::vector<Shown> o;
    o.swap(done);
    return o;
  }
  void finish() {
    {
      std::lock_guard<std::mutex> l(m);
      stop = true;
    }
    cv.notify_one();
    if (t.joinable()) t.join();
    std::lock_guard<std::mutex> l(m);
    for (auto& s : todo) done.push_back(s);
    todo.clear();
  }
  void run() {
    for (;;) {
      Shown s;
      {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return stop || !todo.empty(); });
        if (stop) return;
        s = todo.front();
        todo.pop_front();
      }
      VkResult r;
      for (;;) {
        r = D->vkWaitForPresentKHR(dev, sc, s.id, 50'000'000ull);
        if (r != VK_TIMEOUT) break;
        std::lock_guard<std::mutex> l(m);
        if (stop) break;
      }
      if (r == VK_SUCCESS) { s.shownMs = nowMs() - t0; shownId = s.id; }
      std::lock_guard<std::mutex> l(m);
      done.push_back(s);
    }
  }
};

// ai: The presents counted: the refreshes between presents in a row (one, unless the loop missed a refresh and the
// ai: image before stayed up), and the refreshes each picture held (from its first present to the next picture's). A
// ai: hold of 0 is the waits' doing, not the screen's: after a miss the queued presents' waits return together.
struct Tally {
  std::map<long, uint64_t> held;
  uint64_t presents = 0, pictures = 0, missed = 0, unknown = 0, behind = 0;
  double lastShown = -1, lastFresh = -1;
  uint32_t epoch = 0;
  std::vector<double> gaps;
  double queuedSum = 0;   // ai: the chain's slack at each submit, summed, and the least (-1: none known)
  uint64_t queuedN = 0;
  int queuedMin = -1;
  void add(const Shown& s, double refreshMs) {
    presents++;
    if (s.queued >= 0) {
      queuedSum += s.queued;
      queuedN++;
      if (queuedMin < 0 || s.queued < queuedMin) queuedMin = s.queued;
    }
    if (s.epoch != epoch) { epoch = s.epoch; lastShown = lastFresh = -1; }
    if (s.shownMs < 0) { unknown++; lastShown = lastFresh = -1; return; }
    if (lastShown >= 0) {
      const double gap = s.shownMs - lastShown;
      gaps.push_back(gap);
      const long k = std::lround(gap / refreshMs);
      if (k >= 2) missed += static_cast<uint64_t>(k - 1);
    }
    lastShown = s.shownMs;
    if (s.fresh) {
      pictures++;
      if (lastFresh >= 0) held[std::lround((s.shownMs - lastFresh) / refreshMs)]++;
      lastFresh = s.shownMs;
    }
  }
  std::string heldJson() const {
    std::string o = "{";
    for (const auto& [k, n] : held) o += (o.size() > 1 ? ",\"" : "\"") + std::to_string(k) + "\":" + std::to_string(n);
    return o + "}";
  }
  // ai: the refresh the presents show, Hz: the mean of the gaps of about one refresh (a wait's wake-up moves each time
  // ai: by a few tenths of a ms either way, which a median of gaps leans on and a mean cancels). One refresh is the
  // ai: mode's where it is known, else the median gap (FIFO presents one a refresh while the loop keeps up), so a
  // ai: 144 Hz screen with no mode to read is not judged against 60.
  // ai: Where fewer than half the gaps are near the mode's refresh (a windowed swapchain paced by another monitor's
  // ai: CRTC than the one under the window's centre: a mixed 60/120 Hz pair), the median gap is the refresh
  // ai: instead (2026-10-04: the mode alone left the measurement at 0 and paced 120 presents against 60).
  double measuredHz(double modeHz) const {
    if (gaps.size() < 8) return 0;
    auto aroundRef = [&](double ref, double& hz) {
      double sum = 0;
      size_t n = 0;
      for (double g : gaps)
        if (g > 0.5 * ref && g < 1.5 * ref) { sum += g; n++; }
      hz = n ? 1000.0 * static_cast<double>(n) / sum : 0;
      return n;
    };
    double hz = 0;
    if (modeHz > 0 && 2 * aroundRef(1000.0 / modeHz, hz) >= gaps.size()) return hz;
    std::vector<double> g = gaps;
    std::nth_element(g.begin(), g.begin() + static_cast<long>(g.size() / 2), g.end());
    if (g[g.size() / 2] <= 0) return 0;
    aroundRef(g[g.size() / 2], hz);
    return hz;
  }
  // ai: a new window: its counts zero, the last times kept so the first gap of the next one counts
  void reset() {
    Tally t;
    t.lastShown = lastShown;
    t.lastFresh = lastFresh;
    t.epoch = epoch;
    *this = std::move(t);
  }
};

// ai: the mutex taken for at most `ms`: the present thread never waits out a configure (the sender's painters joined
// ai: and restarted, the GPU checked), only a take or a stats call's few microseconds
bool lockFor(std::mutex& m, double ms) {
  const double end = nowMs() + ms;
  while (!m.try_lock()) {
    if (nowMs() >= end) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

void barrier(const DevFns& D, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
             VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = sa;
  b.dstAccessMask = da;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  D.vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ai: whether a refresh shows fps pictures a second evenly, each held a whole number of refreshes (period()'s 1%)
bool evenAt(double hz, int fps) {
  const double p = hz / std::max(1, fps), r = std::round(p);
  return r >= 1 && std::fabs(p - r) < 0.01 * r;
}
// ai: VSync's choice among a monitor's refreshes at its size: one fps divides, the nearest 60 Hz in whole Hz (the rate
// ai: every monitor has, and the camera's), a tie to the higher (60.00 over 59.94, 120.00 over 119.88); -1 for none
int pickRate(const std::vector<double>& hz, int fps) {
  int best = -1;
  for (int k = 0; k < static_cast<int>(hz.size()); k++) {
    if (!evenAt(hz[k], fps)) continue;
    const long a = std::lround(std::fabs(hz[k] - 60)), b = best < 0 ? 0 : std::lround(std::fabs(hz[best] - 60));
    if (best < 0 || a < b || (a == b && hz[k] > hz[best])) best = k;
  }
  return best;
}

#if LIZ_X11 && !defined(_WIN32)
// ai: a RandR mode's refresh as xrandr reckons it; 0 for an interlaced or doubled one, which VSync never takes
double modeRefresh(const XRRModeInfo& m) {
  if (!m.hTotal || !m.vTotal || (m.modeFlags & (RR_Interlace | RR_DoubleScan))) return 0;
  return static_cast<double>(m.dotClock) / (static_cast<double>(m.hTotal) * m.vTotal);
}
const XRRModeInfo* modeById(const XRRScreenResources* res, RRMode id) {
  for (int k = 0; k < res->nmode; k++) if (res->modes[k].id == id) return &res->modes[k];
  return nullptr;
}
// ai: the CRTC showing a point of the root window, 0 none
RRCrtc crtcAt(Display* d, XRRScreenResources* res, int x, int y) {
  RRCrtc found = 0;
  for (int i = 0; i < res->ncrtc && !found; i++) {
    XRRCrtcInfo* ci = XRRGetCrtcInfo(d, res, res->crtcs[i]);
    if (!ci) continue;
    if (ci->mode && x >= ci->x && x < ci->x + static_cast<int>(ci->width) && y >= ci->y && y < ci->y + static_cast<int>(ci->height)) found = res->crtcs[i];
    XRRFreeCrtcInfo(ci);
  }
  return found;
}

// ai: the window the manager manages above w: the ancestor with WM_STATE (a reparenting manager puts its own frame
// ai: between it and the root), else the root's child (no manager)
Window topLevel(Display* d, Window w) {
  const Atom wmState = XInternAtom(d, "WM_STATE", True);
  Window cur = w;
  for (;;) {
    if (wmState != None) {
      Atom type = None;
      int format = 0;
      unsigned long n = 0, after = 0;
      unsigned char* data = nullptr;
      if (XGetWindowProperty(d, cur, wmState, 0, 0, False, AnyPropertyType, &type, &format, &n, &after, &data) == Success) {
        if (data) XFree(data);
        if (type != None) return cur;
      }
    }
    Window root = 0, parent = 0, *kids = nullptr;
    unsigned int nk = 0;
    if (!XQueryTree(d, cur, &root, &parent, &kids, &nk)) return cur;
    if (kids) XFree(kids);
    if (!parent || parent == root) return cur;
    cur = parent;
  }
}
#endif

}  // namespace

// ai: the window manager's window around w (topLevel above), through a Display of its own (2026-10-23: the parent the
// ai: desktop's file chooser is attached to, tx_jni.cpp windowOf); w itself off X11
unsigned long x11TopLevel(unsigned long window) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return window;
  const unsigned long top = topLevel(d, window);
  XCloseDisplay(d);
  return top;
#else
  return window;
#endif
}

// ai: Full screen as other apps ask for it (2026-10-23): the window manager's own, _NET_WM_STATE_FULLSCREEN added or
// ai: removed by the EWMH client message on the managed window (x11TopLevel's), the way GTK, Qt and the browsers do it;
// ai: the manager keeps the window's place, size and maximized state and gives them back. Until then the presenter set
// ai: _NET_WM_BYPASS_COMPOSITOR and _NET_WM_STATE_ABOVE with the app's full screen (2026-10-03 to 2026-10-23), for a
// ai: vsync whose losses were the machine's power-saving profile. false off X11 (the app then asks Compose).
bool x11Fullscreen(unsigned long top, bool on) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return false;
  XEvent e{};
  e.xclient.type = ClientMessage;
  e.xclient.window = top;
  e.xclient.message_type = XInternAtom(d, "_NET_WM_STATE", False);
  e.xclient.format = 32;
  e.xclient.data.l[0] = on ? 1 : 0;   // ai: _NET_WM_STATE_ADD, _NET_WM_STATE_REMOVE
  e.xclient.data.l[1] = static_cast<long>(XInternAtom(d, "_NET_WM_STATE_FULLSCREEN", False));
  e.xclient.data.l[2] = 0;
  e.xclient.data.l[3] = 1;   // ai: the source: an application
  const Status sent = XSendEvent(d, DefaultRootWindow(d), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
  XFlush(d);
  XCloseDisplay(d);
  return sent != 0;
#else
  (void)top; (void)on;
  return false;
#endif
}
// ai: whether the manager has the window full screen: _NET_WM_STATE_FULLSCREEN among its _NET_WM_STATE
bool x11IsFullscreen(unsigned long top) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return false;
  const Atom state = XInternAtom(d, "_NET_WM_STATE", False), full = XInternAtom(d, "_NET_WM_STATE_FULLSCREEN", False);
  Atom type = None;
  int format = 0;
  unsigned long n = 0, after = 0;
  unsigned char* data = nullptr;
  bool is = false;
  if (XGetWindowProperty(d, top, state, 0, 64, False, XA_ATOM, &type, &format, &n, &after, &data) == Success && data) {
    const Atom* atoms = reinterpret_cast<const Atom*>(data);
    for (unsigned long i = 0; i < n; i++) is = is || atoms[i] == full;
    XFree(data);
  }
  XCloseDisplay(d);
  return is;
#else
  (void)top;
  return false;
#endif
}

#if defined(_WIN32)
namespace {
// ai: the windows full screen now, each with what it was before: its style, its rectangle as a normal window, maximized
struct Windowed { LONG_PTR style = 0; RECT rect{}; bool maximized = false; };
std::mutex gFullMu;
std::map<HWND, Windowed> gFull;
}  // namespace
#endif

// ai: Windows' full screen (2026-10-24; Compose's placement before, which left the window unmaximized and the app's
// ai: own state wrong), as Chromium makes it: a maximized window restored first (left maximized, its placement's
// ai: rectangle stays the monitor's and it comes back over the taskbar), its normal rectangle kept, WS_CAPTION and
// ai: WS_THICKFRAME off and the window over its monitor's whole rectangle, which also hides the taskbar; on leaving,
// ai: the style and the rectangle back and the window maximized again where it was. Called from the app's thread,
// ai: these reach the window's (AWT's) synchronously, as AWT's own peers do.
bool win32Fullscreen(void* hwnd, bool on) {
#if defined(_WIN32)
  HWND w = static_cast<HWND>(hwnd);
  if (!w || !IsWindow(w)) return false;
  std::lock_guard<std::mutex> l(gFullMu);
  auto at = gFull.find(w);
  if (on) {
    if (at != gFull.end()) return true;
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    if (!GetMonitorInfoW(MonitorFromWindow(w, MONITOR_DEFAULTTONEAREST), &mi)) return false;
    Windowed was;
    was.maximized = IsZoomed(w) != 0;
    if (was.maximized) ShowWindow(w, SW_RESTORE);
    was.style = GetWindowLongPtrW(w, GWL_STYLE);
    GetWindowRect(w, &was.rect);
    gFull[w] = was;
    SetWindowLongPtrW(w, GWL_STYLE, was.style & ~static_cast<LONG_PTR>(WS_CAPTION | WS_THICKFRAME));
    const RECT& r = mi.rcMonitor;
    SetWindowPos(w, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
  } else {
    if (at == gFull.end()) return true;
    const Windowed was = at->second;
    gFull.erase(at);
    SetWindowLongPtrW(w, GWL_STYLE, was.style);
    const RECT& r = was.rect;
    SetWindowPos(w, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    if (was.maximized) ShowWindow(w, SW_MAXIMIZE);
  }
  return true;
#else
  (void)hwnd; (void)on;
  return false;
#endif
}
bool win32IsFullscreen(void* hwnd) {
#if defined(_WIN32)
  std::lock_guard<std::mutex> l(gFullMu);
  return gFull.count(static_cast<HWND>(hwnd)) > 0;
#else
  (void)hwnd;
  return false;
#endif
}

struct Presenter::Impl {
  PresentTarget target;
  Sender* sender = nullptr;
  std::mutex* senderLock = nullptr;
  PresentOptions opt;
  std::atomic<int> fps{60};
  std::atomic<double> size{1.0};
  std::atomic<double> edge{0.0};
  std::atomic<bool> whole{false}, paused{false}, quit{false};
  // ai: VSync (setVsync), and a change of it or of fps for the loop to take at its next present
  std::atomic<bool> vsync{false}, vsyncDirty{false};
  std::mutex wakeMu;
  std::condition_variable wake;
  std::thread thread;

#if LIZ_X11 && !defined(_WIN32)
  // ai: two connections of the present thread's own: dpy the surface's (the driver's alone; AWT never calls
  // ai: XInitThreads, so no Xlib call of ours shares it with a driver thread's xcb), xq for the window queries (the size,
  // ai: the mode, whether a compositor draws the window)
  Display* dpy = nullptr;
  Display* xq = nullptr;
  Window xwin = 0;
  int randr = -1;   // ai: XRandR on xq: -1 not asked yet, 0 no, 1 yes
  int shapes = -1;  // ai: Composite and Shape on xq: -1 not asked yet, 0 no, 1 yes
#endif
  // ai: whether the screen shows the window direct (the swapchain's flips reach the scanout: no compositor, or the
  // ai: window unredirected) or a compositor draws it: -1 not known (no X11, no Composite), 0 composited, 1 direct.
  // ai: Read once a second (queryDirect); a change is logged.
  int direct = -1;
  // ai: the sender's device (wg's): the painter's frames and the presents on one GPU (2026-10-07); its handles below
  // ai: are that device's, freed by it, and this presenter frees only what it made on them
  wg::Device* W = nullptr;
  bool shareQueue = false;   // ai: one queue in the family: the painter's too, taken under W->mu around a submit or present
  InstFns I;
  DevFns D;
  VkInstance inst = VK_NULL_HANDLE;
  VkSurfaceKHR surf = VK_NULL_HANDLE;
  VkPhysicalDevice pd = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  VkQueue q = VK_NULL_HANDLE;
  uint32_t qf = 0;
  std::string name;
  bool presentWait = false;
  VkPhysicalDeviceMemoryProperties mem{};
  VkSurfaceCapabilitiesKHR caps{};
  double capsAt = -1e9;   // ai: when the surface's capabilities were last read (once a second, and at a remake)
  Chain c;
  std::vector<Slot> slots = std::vector<Slot>(3);
  VkCommandPool pool = VK_NULL_HANDLE;
  // ai: one command buffer, fence and acquire semaphore an image, made with the chain: only its back-pressure paces
  // ai: the loop (two in flight before, which held the presents to two refreshes ahead whatever the chain)
  std::vector<VkCommandBuffer> cbs;
  std::vector<VkFence> fences;
  std::vector<VkSemaphore> acquired;
  // ai: the expand kernel (liblizard/core/tx/expand.comp, send_expand.spv in the sender's assets): a ring frame to a
  // ai: slot image; a descriptor set an image in flight (the ring's buffer and the slot's view), and the ring buffer
  // ai: each holds while its commands may read it (a configure makes a new ring; the old one lives on until then)
  VkShaderModule expandSm = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  VkPipeline expandPipe = VK_NULL_HANDLE;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> dsets;
  std::vector<std::shared_ptr<wg::Buffer>> ringHeld;
  Waiter waiter;
  Tally win, all;
  FILE* csv = nullptr;

  // ai: the loop's state (the present thread's alone)
  double t0 = 0, lastWin = 0, due = 0, modeHz = 0, measured = 0, early = 0, lastRemake = -1e9;
  std::vector<double> firstAcq;   // ai: the first presents' acquire times, for `early`
  int cur = -1;                         // ai: the slot holding the picture on screen
  uint64_t picture = 0, presents = 0, pid = 0, winPresents = 0;
  int64_t frame = 0;
  uint32_t epoch = 0;
  bool resized = false, suboptimal = false, ignoreSuboptimal = false, wasPaused = false;
  VkExtent2D suboptimalAt{};
  // ai: VSync's hold (holdMode): the monitor looked at for fps (nothing more to do until either moves), and the mode
  // ai: this presenter set on it, to put back. where: X11 the CRTC, Windows the HMONITOR, macOS the display.
  struct Held {
    bool on = false, decided = false;
    uint64_t where = 0, orig = 0, mode = 0;   // ai: X11: the mode before ours, and ours
    double origHz = 0, hz = 0;
    int fps = 0;
#if defined(_WIN32)
    std::wstring device;
#endif
  } held;

  std::mutex statsMu;
  std::string statsJson, totalsJson, error;

  void say(const std::string& m) { if (opt.log) opt.log(m); }
  uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want) return i;
    throw std::runtime_error("no memory type for the frame");
  }
  // ai: the refresh: measured (each second's window), else the mode's, else `early`, the median gap between the first
  // ai: presents' acquires (FIFO paces them at the refresh), else 60 for those few presents (2026-10-04:
  // ai: with no mode, always so on Xvfb, the first second took a picture every present, 144 on a 144 Hz screen)
  double refreshHz() const { return measured > 0 ? measured : modeHz > 0 ? modeHz : early > 0 ? early : 60.0; }
  // ai: refreshes a picture: the refresh over the asked rate, a whole number where it is within 1% of one (a 60 Hz
  // ai: monitor at 60 or 59.94 asked: one; 120 Hz at 60: two), else the due rule spreads them (the web's and Send.kt's)
  double period() const {
    double p = refreshHz() / std::max(1, fps.load());
    const double r = std::round(p);
    if (r >= 1 && std::fabs(p - r) < 0.01 * r) p = r;
    return p;
  }

  void setup();
  void run();
  void teardown();
  void presentOnce(bool live);
  // ai: the queue's submit, present and idle wait, under the device's lock where the painter shares the queue
  VkResult qSubmit(const VkSubmitInfo& si, VkFence f);
  VkResult qPresent(const VkPresentInfoKHR& pi);
  void qIdle();
  void makeChain(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext);
  void remake(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext);
  void makeSlot(Slot& s, int w, int h);
  void freeSlot(Slot& s);
  VkExtent2D windowSize();
  void queryMode();
  void queryDirect();
  void holdMode();
  void releaseMode();
  void rateMoved();
  void drain();
  void window(double secs, bool last = false);
};

VkResult Presenter::Impl::qSubmit(const VkSubmitInfo& si, VkFence f) {
  if (!shareQueue) return D.vkQueueSubmit(q, 1, &si, f);
  std::lock_guard<std::mutex> l(W->mu);
  return D.vkQueueSubmit(q, 1, &si, f);
}

VkResult Presenter::Impl::qPresent(const VkPresentInfoKHR& pi) {
  if (!shareQueue) return D.vkQueuePresentKHR(q, &pi);
  std::lock_guard<std::mutex> l(W->mu);
  return D.vkQueuePresentKHR(q, &pi);
}

void Presenter::Impl::qIdle() {
  if (!shareQueue) { D.vkQueueWaitIdle(q); return; }
  std::lock_guard<std::mutex> l(W->mu);
  D.vkQueueWaitIdle(q);
}

// ai: The sender's device (wg's, made by Sender::prepareGpu with deviceExtras()): the surface on its instance, the
// ai: presents on the family's second queue where it has one (the 4090's), else on the painter's own under its lock
// ai: (RADV, lavapipe); the swapchain, pool, fences and semaphores this presenter's own.
void Presenter::Impl::setup() {
  W = sender->device();
  if (!W) throw std::runtime_error("the sender has no Vulkan device to present from" + (sender->prepareWhy().empty() ? std::string() : ": " + sender->prepareWhy()));
  inst = W->instance;
  pd = W->phys;
  dev = W->dev;
  qf = W->family;
  name = W->name;
  shareQueue = !W->twoQueues;
  q = shareQueue ? W->queue : W->queue2;
#define LIZ_LOADI(f)                                                                     \
  I.f = reinterpret_cast<PFN_##f>(vkGetInstanceProcAddr(inst, #f));                      \
  if (!I.f) throw std::runtime_error("the sender's Vulkan instance has no " #f " (no window surface extension)");
  LIZ_INST_FNS(LIZ_LOADI)
#undef LIZ_LOADI
  if (!W->has(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) throw std::runtime_error(name + " has no swapchain");
#define LIZ_LOADD(f)                                                                     \
  D.f = reinterpret_cast<PFN_##f>(I.vkGetDeviceProcAddr(dev, #f));                       \
  if (!D.f) throw std::runtime_error("the Vulkan device has no " #f);
  LIZ_DEV_FNS(LIZ_LOADD)
#undef LIZ_LOADD
  presentWait = W->has(VK_KHR_PRESENT_ID_EXTENSION_NAME) && W->has(VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
  if (presentWait) D.vkWaitForPresentKHR = reinterpret_cast<PFN_vkWaitForPresentKHR>(I.vkGetDeviceProcAddr(dev, "vkWaitForPresentKHR"));
  if (!D.vkWaitForPresentKHR) presentWait = false;

#if defined(_WIN32)
  if (!target.hwnd) throw std::runtime_error("no window to present to");
  auto createSurface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateWin32SurfaceKHR"));
  if (!createSurface) throw std::runtime_error("the sender's Vulkan instance has no VK_KHR_win32_surface");
  VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
  si.hinstance = GetModuleHandleW(nullptr);
  si.hwnd = static_cast<HWND>(target.hwnd);
  vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateWin32SurfaceKHR");
#elif LIZ_X11
  if (!target.x11Window) throw std::runtime_error("no window to present to");
  dpy = XOpenDisplay(nullptr);
  xq = dpy ? XOpenDisplay(nullptr) : nullptr;
  if (!dpy || !xq) throw std::runtime_error("no X display");
  xwin = static_cast<Window>(target.x11Window);
  auto createSurface = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateXlibSurfaceKHR"));
  if (!createSurface) throw std::runtime_error("the sender's Vulkan instance has no VK_KHR_xlib_surface");
  VkXlibSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  si.dpy = dpy;
  si.window = xwin;
  vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateXlibSurfaceKHR");
#elif defined(__APPLE__)
  if (!target.metalLayer) throw std::runtime_error("no layer to present to");
  auto createSurface = reinterpret_cast<PFN_vkCreateMetalSurfaceEXT>(vkGetInstanceProcAddr(inst, "vkCreateMetalSurfaceEXT"));
  if (!createSurface) throw std::runtime_error("the sender's Vulkan instance has no VK_EXT_metal_surface");
  VkMetalSurfaceCreateInfoEXT si{VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
  si.pLayer = static_cast<const CAMetalLayer*>(target.metalLayer);
  vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateMetalSurfaceEXT");
#else
  throw std::runtime_error("no window surface for this platform yet");
#endif
  VkBool32 ok = VK_FALSE;
  I.vkGetPhysicalDeviceSurfaceSupportKHR(pd, qf, surf, &ok);
  if (!ok) throw std::runtime_error(name + " cannot present to this window from its queue family (LIZ_VK_DEVICE names another GPU)");
  I.vkGetPhysicalDeviceMemoryProperties(pd, &mem);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = qf;
  vkCheck(D.vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
  {
    // ai: by its UTF-8 path (fs::u8path): Windows reads a narrow one in the ANSI code page, so under a profile folder
    // ai: with a letter past ASCII (the app's tree under %LOCALAPPDATA%) the code was never shown (2026-10-25)
    const std::string path = sender->assets() + "/spv/send_expand.spv";
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (code.empty() || code.size() % 4) throw std::runtime_error("no expand kernel at " + path);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size();
    smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    vkCheck(D.vkCreateShaderModule(dev, &smi, nullptr, &expandSm), "vkCreateShaderModule");
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2;
    dli.pBindings = b;
    vkCheck(D.vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    vkCheck(D.vkCreatePipelineLayout(dev, &pli, nullptr, &pl), "vkCreatePipelineLayout");
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = expandSm;
    cpi.stage.pName = "main";
    cpi.layout = pl;
    vkCheck(D.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &expandPipe), "vkCreateComputePipelines");
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 16;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = ps;
    vkCheck(D.vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool), "vkCreateDescriptorPool");
  }
  if (!opt.csv.empty()) {
#if defined(_WIN32)
    csv = _wfopen(std::filesystem::u8path(opt.csv).c_str(), L"w");   // ai: a UTF-8 path, as the expand kernel's
#else
    csv = std::fopen(opt.csv.c_str(), "w");
#endif
    if (!csv) throw std::runtime_error(opt.csv + " could not be written");
    std::fprintf(csv, "id,picture,fresh,acquired_ms,submitted_ms,shown_ms,queued\n");
  }
  queryMode();
  t0 = lastWin = nowMs();
  window(0);
  say("presenter: " + name + (shareQueue ? ", the painter's queue shared" : ", the family's second queue") +
      (presentWait ? ", times from present_wait" : ", times from acquire (no present_wait)") +
      (modeHz > 0 ? ", the mode's refresh " + fmt("%.3f", modeHz) + " Hz" : ""));
}

// ai: the window's size where the surface does not say (currentExtent left to the swapchain)
VkExtent2D Presenter::Impl::windowSize() {
#if defined(_WIN32)
  RECT r{};
  if (GetClientRect(static_cast<HWND>(target.hwnd), &r)) return {static_cast<uint32_t>(r.right - r.left), static_cast<uint32_t>(r.bottom - r.top)};
#elif LIZ_X11
  XWindowAttributes wa{};
  if (XGetWindowAttributes(xq, xwin, &wa)) return {static_cast<uint32_t>(wa.width), static_cast<uint32_t>(wa.height)};
#elif defined(__APPLE__)
  uint32_t w = 0, h = 0;
  if (mac::layerPixels(target.metalLayer, &w, &h)) return {w, h};
#endif
  return {0, 0};
}

// ai: X11: the mode's refresh of the monitor (a CRTC) holding the window's centre; Windows: the display settings of the
// ai: monitor nearest the window (whole Hz: 59 for 59.94, the measurement corrects it); macOS: the main display's mode
void Presenter::Impl::queryMode() {
#if defined(_WIN32)
  HMONITOR mon = MonitorFromWindow(static_cast<HWND>(target.hwnd), MONITOR_DEFAULTTONEAREST);
  MONITORINFOEXW mi{};
  mi.cbSize = sizeof mi;
  DEVMODEW dm{};
  dm.dmSize = sizeof dm;
  if (mon && GetMonitorInfoW(mon, &mi) && EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1)
    modeHz = dm.dmDisplayFrequency;
#elif LIZ_X11
  if (randr < 0) {
    int ev = 0, er = 0;
    randr = XRRQueryExtension(xq, &ev, &er) ? 1 : 0;
  }
  if (!randr) return;
  XWindowAttributes wa{};
  if (!XGetWindowAttributes(xq, xwin, &wa)) return;
  int cx = 0, cy = 0;
  Window child = 0;
  XTranslateCoordinates(xq, xwin, wa.root, wa.width / 2, wa.height / 2, &cx, &cy, &child);
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, wa.root);
  if (!res) return;
  double hz = 0;
  for (int i = 0; i < res->ncrtc && hz == 0; i++) {
    XRRCrtcInfo* ci = XRRGetCrtcInfo(xq, res, res->crtcs[i]);
    if (!ci) continue;
    if (ci->mode && cx >= ci->x && cx < ci->x + static_cast<int>(ci->width) && cy >= ci->y && cy < ci->y + static_cast<int>(ci->height))
      for (int k = 0; k < res->nmode; k++)
        if (res->modes[k].id == ci->mode && res->modes[k].hTotal && res->modes[k].vTotal)
          hz = static_cast<double>(res->modes[k].dotClock) / (static_cast<double>(res->modes[k].hTotal) * res->modes[k].vTotal);
    XRRFreeCrtcInfo(ci);
  }
  XRRFreeScreenResources(res);
  modeHz = hz;
#elif defined(__APPLE__)
  modeHz = mac::layerRefreshHz(target.metalLayer);
#endif
}

// ai: X11: is the window on the screen direct, or drawn by a compositor? A compositing manager owns _NET_WM_CM_S<screen>
// ai: and draws into the composite overlay window; a window it unredirects (Mutter, Muffin, KWin and others cut it out
// ai: of the overlay's bounding shape) shows direct, its swapchain flipping on its own monitor's vblank. The window's
// ai: centre inside the overlay's shape: composited. No manager: direct. The overlay is asked for and released around
// ai: each query, so this process never keeps it mapped past the manager's own hold (2026-10-07: full screen under
// ai: Muffin was composited whenever another window stood above the sender in the stack, even on the other monitor,
// ai: and its composite tore 3% of frames; the app keeps the window above the others while full screen).
void Presenter::Impl::queryDirect() {
#if LIZ_X11 && !defined(_WIN32)
  if (shapes < 0) {
    int ev = 0, er = 0;
    shapes = XCompositeQueryExtension(xq, &ev, &er) && XShapeQueryExtension(xq, &ev, &er) ? 1 : 0;
  }
  int now = -1;
  if (shapes) {
    XWindowAttributes wa{};
    if (!XGetWindowAttributes(xq, xwin, &wa)) return;
    char sel[32];
    std::snprintf(sel, sizeof sel, "_NET_WM_CM_S%d", XScreenNumberOfScreen(wa.screen));
    if (!XGetSelectionOwner(xq, XInternAtom(xq, sel, False))) now = 1;
    else {
      int cx = 0, cy = 0;
      Window child = 0;
      XTranslateCoordinates(xq, xwin, wa.root, wa.width / 2, wa.height / 2, &cx, &cy, &child);
      const Window cow = XCompositeGetOverlayWindow(xq, wa.root);
      int n = 0, order = 0;
      XRectangle* rs = cow ? XShapeGetRectangles(xq, cow, ShapeBounding, &n, &order) : nullptr;
      bool covered = false;
      for (int i = 0; i < n; i++)
        if (cx >= rs[i].x && cx < rs[i].x + static_cast<int>(rs[i].width) && cy >= rs[i].y && cy < rs[i].y + static_cast<int>(rs[i].height)) covered = true;
      if (rs) XFree(rs);
      if (cow) XCompositeReleaseOverlayWindow(xq, wa.root);
      XFlush(xq);
      now = covered ? 0 : 1;
    }
  }
  if (now != direct) {
    direct = now;
    if (now == 0) say("presenter: the compositor draws the window");
    else if (now == 1) say("presenter: the screen shows the window direct");
  }
#endif
}

// ai: VSync's hold (presenter.h setVsync), at a change of VSync or fps and once a second: the monitor under the window's
// ai: centre and the refresh of its mode before ours; where fps divides that, its own mode (ours put back), else the
// ai: mode at the same size pickRate takes, set. Each monitor and fps is looked at once (held.decided).
void Presenter::Impl::holdMode() {
  if (!vsync) { releaseMode(); held.decided = false; return; }
  const int f = std::max(1, fps.load());
  auto fpsText = [&] { return std::to_string(f) + " pictures a second"; };
#if defined(_WIN32)
  HMONITOR mon = MonitorFromWindow(static_cast<HWND>(target.hwnd), MONITOR_DEFAULTTONEAREST);
  MONITORINFOEXW mi{};
  mi.cbSize = sizeof mi;
  if (!mon || !GetMonitorInfoW(mon, &mi)) return;
  const uint64_t where = reinterpret_cast<uint64_t>(mon);
  if (held.on && held.where != where) { releaseMode(); held.decided = false; }
  if (held.decided && held.where == where && held.fps == f) return;
  DEVMODEW cur{};
  cur.dmSize = sizeof cur;
  if (!EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &cur) || cur.dmDisplayFrequency <= 1) return;
  held.decided = true; held.where = where; held.fps = f;
  const double origHz = held.on ? held.origHz : cur.dmDisplayFrequency;
  if (evenAt(origHz, f)) { releaseMode(); return; }
  // ai: the monitor's modes at its size and depth, progressive (Windows lists each refresh in whole Hz: 59 is 59.94)
  std::vector<DEVMODEW> modes;
  std::vector<double> hz;
  DEVMODEW dm{};
  dm.dmSize = sizeof dm;
  for (DWORD i = 0; EnumDisplaySettingsW(mi.szDevice, i, &dm); i++) {
    if (dm.dmPelsWidth == cur.dmPelsWidth && dm.dmPelsHeight == cur.dmPelsHeight && dm.dmBitsPerPel == cur.dmBitsPerPel &&
        !(dm.dmDisplayFlags & DM_INTERLACED) && dm.dmDisplayFrequency > 1) {
      modes.push_back(dm);
      hz.push_back(dm.dmDisplayFrequency);
    }
    dm = DEVMODEW{};
    dm.dmSize = sizeof dm;
  }
  const int b = pickRate(hz, f);
  if (b < 0) { say("presenter: VSync: no mode of this monitor shows " + fpsText() + " evenly"); releaseMode(); return; }
  if (hz[b] == cur.dmDisplayFrequency) return;
  DEVMODEW want = modes[b];
  want.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL | DM_DISPLAYFREQUENCY;
  if (ChangeDisplaySettingsExW(mi.szDevice, &want, nullptr, CDS_FULLSCREEN, nullptr) != DISP_CHANGE_SUCCESSFUL) {
    say("presenter: VSync: the monitor refused " + fmt("%.0f", hz[b]) + " Hz");
    return;
  }
  if (!held.on) { held.origHz = origHz; held.device = mi.szDevice; }
  held.on = true;
  held.hz = hz[b];
#elif LIZ_X11
  if (randr <= 0) return;
  XWindowAttributes wa{};
  if (!XGetWindowAttributes(xq, xwin, &wa)) return;
  int cx = 0, cy = 0;
  Window child = 0;
  XTranslateCoordinates(xq, xwin, wa.root, wa.width / 2, wa.height / 2, &cx, &cy, &child);
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, wa.root);
  if (!res) return;
  RRCrtc crtc = crtcAt(xq, res, cx, cy);
  if (held.on && crtc && held.where != crtc) {
    releaseMode();
    held.decided = false;
    XRRFreeScreenResources(res);
    res = XRRGetScreenResourcesCurrent(xq, wa.root);
    if (!res) return;
    crtc = crtcAt(xq, res, cx, cy);
  }
  if (!crtc || (held.decided && held.where == crtc && held.fps == f)) { XRRFreeScreenResources(res); return; }
  XRRCrtcInfo* ci = XRRGetCrtcInfo(xq, res, crtc);
  const XRRModeInfo* om = ci ? modeById(res, held.on ? held.orig : ci->mode) : nullptr;
  if (!om || ci->noutput < 1) {
    if (ci) XRRFreeCrtcInfo(ci);
    XRRFreeScreenResources(res);
    return;
  }
  held.decided = true; held.where = crtc; held.fps = f;
  const double origHz = held.on ? held.origHz : modeRefresh(*om);
  // ai: the CRTC's outputs: modes all of them list, at the size of the mode before ours. Xwayland's are emulated (its
  // ai: modes scale one client's view; the refresh is the Wayland compositor's), so VSync leaves them alone.
  std::vector<RRMode> ids;
  std::vector<double> hz;
  bool xwayland = false;
  for (int o = 0; o < ci->noutput; o++) {
    XRROutputInfo* oi = XRRGetOutputInfo(xq, res, ci->outputs[o]);
    if (!oi) continue;
    if (std::strncmp(oi->name, "XWAYLAND", 8) == 0) xwayland = true;
    if (o == 0) {
      for (int k = 0; k < oi->nmode; k++) {
        const XRRModeInfo* m = modeById(res, oi->modes[k]);
        const double r = m && m->width == om->width && m->height == om->height ? modeRefresh(*m) : 0;
        if (r > 0) { ids.push_back(m->id); hz.push_back(r); }
      }
    } else {
      for (size_t k = 0; k < ids.size();) {
        bool listed = false;
        for (int j = 0; j < oi->nmode; j++) listed = listed || oi->modes[j] == ids[k];
        if (listed) k++;
        else { ids.erase(ids.begin() + static_cast<long>(k)); hz.erase(hz.begin() + static_cast<long>(k)); }
      }
    }
    XRRFreeOutputInfo(oi);
  }
  int b = -1;
  if (xwayland) say("presenter: VSync: under Wayland the desktop sets the refresh");
  else if (!evenAt(origHz, f) && (b = pickRate(hz, f)) < 0) say("presenter: VSync: no mode of this monitor shows " + fpsText() + " evenly");
  if (xwayland || b < 0 || ids[static_cast<size_t>(b)] == ci->mode) {
    XRRFreeCrtcInfo(ci);
    XRRFreeScreenResources(res);
    if (xwayland || b < 0) releaseMode();
    return;
  }
  const RRMode want = ids[static_cast<size_t>(b)], before = ci->mode;
  const Status st = XRRSetCrtcConfig(xq, res, crtc, CurrentTime, ci->x, ci->y, want, ci->rotation, ci->outputs, ci->noutput);
  XRRFreeCrtcInfo(ci);
  XRRFreeScreenResources(res);
  if (st != RRSetConfigSuccess) {
    say("presenter: VSync: the monitor refused " + fmt("%.2f", hz[static_cast<size_t>(b)]) + " Hz");
    return;
  }
  if (!held.on) { held.orig = before; held.origHz = origHz; }
  held.on = true;
  held.mode = want;
  held.hz = hz[static_cast<size_t>(b)];
#elif defined(__APPLE__)
  const uint32_t display = mac::mainDisplay();
  if (held.on && held.where != display) { releaseMode(); held.decided = false; }
  if (held.decided && held.where == display && held.fps == f) return;
  double now = 0;
  const std::vector<double> hz = mac::displayRates(display, &now);
  // ai: a built-in panel's mode says 0: its refresh is the one measured
  const double origHz = held.on ? held.origHz : now > 0 ? now : measured;
  if (origHz <= 0) return;
  held.decided = true; held.where = display; held.fps = f;
  if (evenAt(origHz, f)) { releaseMode(); return; }
  const int b = pickRate(hz, f);
  if (b < 0) { say("presenter: VSync: no mode of this display shows " + fpsText() + " evenly"); releaseMode(); return; }
  if (now > 0 && std::fabs(hz[static_cast<size_t>(b)] - now) < 0.01) return;
  if (!mac::setDisplayRate(display, hz[static_cast<size_t>(b)])) {
    say("presenter: VSync: the display refused " + fmt("%.2f", hz[static_cast<size_t>(b)]) + " Hz");
    return;
  }
  if (!held.on) held.origHz = origHz;
  held.on = true;
  held.hz = hz[static_cast<size_t>(b)];
#else
  return;
#endif
  rateMoved();
  say("presenter: VSync: the monitor moved from " + fmt("%.2f", held.origHz) + " to " + fmt("%.2f", held.hz) + " Hz for " + fpsText());
}

// ai: the monitor's own mode back, where it still shows ours (a mode the desktop set since is left as it is)
void Presenter::Impl::releaseMode() {
  if (!held.on) return;
  held.on = false;
#if defined(_WIN32)
  DEVMODEW cur{};
  cur.dmSize = sizeof cur;
  if (EnumDisplaySettingsW(held.device.c_str(), ENUM_CURRENT_SETTINGS, &cur) && cur.dmDisplayFrequency == held.hz)
    ChangeDisplaySettingsExW(held.device.c_str(), nullptr, nullptr, 0, nullptr);   // ai: the registry's mode, the desktop's
#elif LIZ_X11
  if (XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, DefaultRootWindow(xq))) {
    if (XRRCrtcInfo* ci = XRRGetCrtcInfo(xq, res, static_cast<RRCrtc>(held.where))) {
      if (ci->mode == held.mode &&
          XRRSetCrtcConfig(xq, res, static_cast<RRCrtc>(held.where), CurrentTime, ci->x, ci->y, static_cast<RRMode>(held.orig), ci->rotation,
                           ci->outputs, ci->noutput) != RRSetConfigSuccess)
        say("presenter: VSync: the monitor's own mode could not be put back");
      XRRFreeCrtcInfo(ci);
    }
    XRRFreeScreenResources(res);
  }
#elif defined(__APPLE__)
  mac::restoreDisplays();
#endif
  rateMoved();
  say("presenter: VSync: the monitor back at " + fmt("%.2f", held.origHz) + " Hz");
}

// ai: the monitor's refresh moved: its mode read again, the refresh measured afresh, no gap counted across the change, a
// ai: picture due at the next present
void Presenter::Impl::rateMoved() {
  queryMode();
  measured = 0;
  early = 0;
  firstAcq.clear();
  win.gaps.clear();
  epoch++;
  due = static_cast<double>(presents);
}

void Presenter::Impl::makeChain(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext) {
  if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) throw std::runtime_error("the swapchain cannot be written by transfers");
  uint32_t fn = 0;
  I.vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surf, &fn, nullptr);
  std::vector<VkSurfaceFormatKHR> fs(fn);
  I.vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surf, &fn, fs.data());
  VkSurfaceFormatKHR pick{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  for (const auto& f : fs)
    if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { pick = f; break; }
  if (pick.format == VK_FORMAT_UNDEFINED) throw std::runtime_error("no 8-bit UNORM swapchain format");
  VkFormatProperties fp;
  I.vkGetPhysicalDeviceFormatProperties(pd, pick.format, &fp);
  if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) throw std::runtime_error("the swapchain's format takes no blit");
  // ai: deep: 8 images where the surface allows, the presents queued as far ahead of the screen as the chain holds, so
  // ai: a stall of several refreshes on this thread or the GPU (a painter batch, an X round trip, a fence) never leaves
  // ai: a refresh without its present (2026-10-07; 3 images held about two refreshes of slack)
  uint32_t count = std::max(8u, caps.minImageCount);
  if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
  count = std::min(count, 16u);
  VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  if (!(caps.supportedCompositeAlpha & alpha))
    for (uint32_t b = 1; b; b <<= 1)
      if (caps.supportedCompositeAlpha & b) { alpha = static_cast<VkCompositeAlphaFlagBitsKHR>(b); break; }
  VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  ci.surface = surf;
  ci.minImageCount = count;
  ci.imageFormat = pick.format;
  ci.imageColorSpace = pick.colorSpace;
  ci.imageExtent = ext;
  ci.imageArrayLayers = 1;
  ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.preTransform = caps.currentTransform;
  ci.compositeAlpha = alpha;
  ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  ci.clipped = VK_TRUE;
  ci.oldSwapchain = c.sc;
  VkSwapchainKHR sc;
  vkCheck(D.vkCreateSwapchainKHR(dev, &ci, nullptr, &sc), "vkCreateSwapchainKHR");
  if (c.sc) D.vkDestroySwapchainKHR(dev, c.sc, nullptr);
  for (VkSemaphore s : c.done) D.vkDestroySemaphore(dev, s, nullptr);
  c.done.clear();
  c.sc = sc;
  c.format = pick.format;
  c.extent = ext;
  uint32_t n = 0;
  D.vkGetSwapchainImagesKHR(dev, sc, &n, nullptr);
  c.images.resize(n);
  D.vkGetSwapchainImagesKHR(dev, sc, &n, c.images.data());
  c.done.assign(n, VK_NULL_HANDLE);
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (auto& s : c.done) vkCheck(D.vkCreateSemaphore(dev, &sci, nullptr, &s), "vkCreateSemaphore");
  // ai: one command buffer, fence and acquire semaphore an image (the queue is idle here: remake waited it out)
  for (VkFence f : fences) if (f) D.vkDestroyFence(dev, f, nullptr);
  for (VkSemaphore a : acquired) if (a) D.vkDestroySemaphore(dev, a, nullptr);
  if (!cbs.empty()) D.vkFreeCommandBuffers(dev, pool, static_cast<uint32_t>(cbs.size()), cbs.data());
  cbs.assign(n, VK_NULL_HANDLE);
  fences.assign(n, VK_NULL_HANDLE);
  acquired.assign(n, VK_NULL_HANDLE);
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = n;
  vkCheck(D.vkAllocateCommandBuffers(dev, &cai, cbs.data()), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (uint32_t i = 0; i < n; i++) {
    vkCheck(D.vkCreateFence(dev, &fci, nullptr, &fences[i]), "vkCreateFence");
    vkCheck(D.vkCreateSemaphore(dev, &sci, nullptr, &acquired[i]), "vkCreateSemaphore");
  }
  for (auto& sl : slots) sl.usedBy = -2;
  // ai: a descriptor set an image in flight, from the pool emptied (nothing is in flight here)
  vkCheck(D.vkResetDescriptorPool(dev, dpool, 0), "vkResetDescriptorPool");
  std::vector<VkDescriptorSetLayout> layouts(n, dsl);
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = dpool;
  dai.descriptorSetCount = n;
  dai.pSetLayouts = layouts.data();
  dsets.assign(n, VK_NULL_HANDLE);
  vkCheck(D.vkAllocateDescriptorSets(dev, &dai, dsets.data()), "vkAllocateDescriptorSets");
  ringHeld.assign(n, nullptr);
}

// ai: The swapchain made again (a new size, out of date, suboptimal): the waits stopped first (a wait on a swapchain
// ai: that is gone is undefined), the presents so far counted, a new epoch so no gap is counted across it.
void Presenter::Impl::remake(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext) {
  waiter.finish();
  drain();
  qIdle();
  makeChain(caps, ext);
  epoch++;
  waiter.start(D, dev, presentWait, c.sc, t0);
  resized = false;
  lastRemake = nowMs();
  say("presenter: the swapchain at " + std::to_string(ext.width) + " x " + std::to_string(ext.height) + ", " +
      (c.format == VK_FORMAT_B8G8R8A8_UNORM ? "BGRA8" : "RGBA8") + " UNORM, " + std::to_string(c.images.size()) + " images, FIFO");
}

void Presenter::Impl::freeSlot(Slot& s) {
  if (s.view) D.vkDestroyImageView(dev, s.view, nullptr);
  if (s.img) D.vkDestroyImage(dev, s.img, nullptr);
  if (s.imgMem) D.vkFreeMemory(dev, s.imgMem, nullptr);
  s = Slot{};
}

// ai: a slot's image: RGBA8, written by the expand kernel (storage), read by the blit (transfer source)
void Presenter::Impl::makeSlot(Slot& s, int w, int h) {
  freeSlot(s);
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = VK_FORMAT_R8G8B8A8_UNORM;
  ii.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  vkCheck(D.vkCreateImage(dev, &ii, nullptr, &s.img), "vkCreateImage");
  VkMemoryRequirements r;
  D.vkGetImageMemoryRequirements(dev, s.img, &r);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = r.size;
  ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkCheck(D.vkAllocateMemory(dev, &ai, nullptr, &s.imgMem), "vkAllocateMemory");
  vkCheck(D.vkBindImageMemory(dev, s.img, s.imgMem, 0), "vkBindImageMemory");
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = s.img;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = VK_FORMAT_R8G8B8A8_UNORM;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCheck(D.vkCreateImageView(dev, &vi, nullptr, &s.view), "vkCreateImageView");
  s.w = w;
  s.h = h;
}

void Presenter::Impl::drain() {
  const double refreshMs = 1000.0 / refreshHz();
  for (const Shown& s : waiter.take()) {
    win.add(s, refreshMs);
    all.add(s, refreshMs);
    if (csv)
      std::fprintf(csv, "%llu,%llu,%d,%s,%s,%s,%d\n", static_cast<unsigned long long>(s.id), static_cast<unsigned long long>(s.picture),
                   s.fresh ? 1 : 0, fmt("%.3f", s.acquiredMs).c_str(), fmt("%.3f", s.submittedMs).c_str(), fmt("%.3f", s.shownMs).c_str(), s.queued);
  }
}

// ai: A second's window closed: the presents' times counted, the refresh measured, the mode read again (the window may
// ai: have moved to another monitor) and VSync's hold looked at (not in the last window, at a stop), the stats made,
// ai: the counts zeroed.
void Presenter::Impl::window(double secs, bool last) {
  drain();
  const double m = win.measuredHz(modeHz);
  if (m > 0) measured = m;
  queryMode();
  queryDirect();
  if (!last) holdMode();
  const Slot* s = cur >= 0 ? &slots[cur] : nullptr;
  const double per = secs > 0 ? secs : 1;
  std::string st = "{\"shownFps\":" + fmt("%.1f", secs > 0 ? win.pictures / per : 0) +
                   ",\"presentsPerSec\":" + fmt("%.1f", secs > 0 ? winPresents / per : 0) + ",\"hz\":" + fmt("%.3f", refreshHz()) +
                   ",\"modeHz\":" + fmt("%.3f", modeHz) + ",\"held\":" + win.heldJson() + ",\"missed\":" + std::to_string(win.missed) +
                   ",\"behind\":" + std::to_string(win.behind) + ",\"queued\":" + fmt("%.1f", win.queuedN ? win.queuedSum / static_cast<double>(win.queuedN) : 0.0) +
                   ",\"queuedMin\":" + std::to_string(std::max(0, win.queuedMin)) + ",\"images\":" + std::to_string(c.images.size()) +
                   ",\"paused\":" + (paused ? "true" : "false") + ",\"surface\":\"" +
                   std::to_string(c.extent.width) + "x" + std::to_string(c.extent.height) + "\",\"frame\":\"" +
                   std::to_string(s ? s->w : 0) + "x" + std::to_string(s ? s->h : 0) + "\",\"presentWait\":" + (presentWait ? "true" : "false") +
                   (direct >= 0 ? std::string(",\"direct\":") + (direct ? "true" : "false") : std::string()) +
                   ",\"vsync\":" + (vsync ? "true" : "false") + ",\"modeFrom\":" + fmt("%.3f", held.on ? held.origHz : 0.0) +
                   ",\"device\":\"" + esc(name) + "\",\"error\":\"";
  const double total = (nowMs() - t0) / 1000;
  std::string tt = "{\"secs\":" + fmt("%.3f", total) + ",\"pictures\":" + std::to_string(all.pictures) + ",\"presents\":" +
                   std::to_string(all.presents) + ",\"held\":" + all.heldJson() + ",\"missed\":" + std::to_string(all.missed) +
                   ",\"behind\":" + std::to_string(all.behind) + ",\"queued\":" + fmt("%.1f", all.queuedN ? all.queuedSum / static_cast<double>(all.queuedN) : 0.0) +
                   ",\"queuedMin\":" + std::to_string(std::max(0, all.queuedMin)) + ",\"unknown\":" + std::to_string(all.unknown) + ",\"hz\":" +
                   fmt("%.3f", refreshHz()) + ",\"modeHz\":" + fmt("%.3f", modeHz) + "}";
  {
    std::lock_guard<std::mutex> l(statsMu);
    statsJson = st + esc(error) + "\",\"secs\":" + fmt("%.3f", secs) + "}";
    totalsJson = tt;
  }
  win.reset();
  winPresents = 0;
}

void Presenter::Impl::presentOnce(bool live) {
  // ai: the surface's capabilities once a second and when the chain must be made again (a resize shows first as out
  // ai: of date or suboptimal), not every present
  const double nowT = nowMs();
  if (!c.sc || resized || suboptimal || nowT - capsAt >= 1000) {
    vkCheck(I.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surf, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    capsAt = nowT;
  }
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == UINT32_MAX) {
    ext = windowSize();
    ext.width = std::clamp(ext.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  // ai: no room (minimised, not laid out yet): nothing to present to
  if (!ext.width || !ext.height) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return;
  }
  const bool sized = ext.width != c.extent.width || ext.height != c.extent.height;
  if (sized) ignoreSuboptimal = false;
  // ai: suboptimal remakes the swapchain once; where it says so again at the same size (a compositor's copy, a
  // ai: driver's steady state), it is that platform's way and left alone until the size changes
  if (suboptimal && !sized && !resized) {
    if (lastRemake > 0 && suboptimalAt.width == ext.width && suboptimalAt.height == ext.height && nowMs() - lastRemake < 2000) {
      ignoreSuboptimal = true;
      say("presenter: the swapchain stays suboptimal at this size; left as it is");
    } else resized = true;
    suboptimalAt = ext;
  }
  suboptimal = false;
  if (!c.sc || sized || resized) remake(caps, ext);

  const int N = static_cast<int>(cbs.size());
  const int fl = static_cast<int>(frame % N);
  D.vkWaitForFences(dev, 1, &fences[fl], VK_TRUE, UINT64_MAX);
  ringHeld[fl].reset();
  uint32_t idx = 0;
  // ai: an acquire waits at most 100 ms (a window the screen does not show, minimised or on no monitor, can stall
  // ai: FIFO): then the loop comes round again, so a stop or a stats window is never held up behind it
  VkResult r = D.vkAcquireNextImageKHR(dev, c.sc, 100'000'000ull, acquired[fl], VK_NULL_HANDLE, &idx);
  if (r == VK_TIMEOUT || r == VK_NOT_READY) return;
  if (r == VK_ERROR_OUT_OF_DATE_KHR) { resized = true; return; }
  if (r == VK_SUBOPTIMAL_KHR) { if (!ignoreSuboptimal) suboptimal = true; }
  else vkCheck(r, "vkAcquireNextImageKHR");
  const double acq = nowMs() - t0;
  D.vkResetFences(dev, 1, &fences[fl]);
  // ai: the first chain-full of acquires return at once (the images are free): the early refresh reads the ones after
  if (measured <= 0 && modeHz <= 0 && frame >= N && firstAcq.size() < 16) {
    firstAcq.push_back(acq);
    if (firstAcq.size() >= 6) {
      std::vector<double> g;
      for (size_t k = 1; k < firstAcq.size(); k++) g.push_back(firstAcq[k] - firstAcq[k - 1]);
      std::nth_element(g.begin(), g.begin() + static_cast<long>(g.size() / 2), g.end());
      if (g[g.size() / 2] > 0.5) early = 1000.0 / g[g.size() / 2];
    }
  }

  // ai: a new picture when one is due and painted: its slot in the sender's ring taken (no copy), expanded below
  // ai: into a slot image; a due one not yet painted keeps the last up (behind)
  bool fresh = false;
  int ringSlot = -1;
  std::shared_ptr<wg::Buffer> ring;
  uint32_t ringFS = 0, ringRW = 0;
  const double i = static_cast<double>(presents);
  if (live && i + 0.5 >= due) {
    const int p = (cur + 1) % static_cast<int>(slots.size());
    if (slots[p].usedBy >= 0 && slots[p].usedBy > frame - N) D.vkWaitForFences(dev, 1, &fences[slots[p].usedBy % N], VK_TRUE, UINT64_MAX);
    if (lockFor(*senderLock, 2.0)) {
      std::lock_guard<std::mutex> held(*senderLock, std::adopt_lock);
      const int fw = sender->width(), fh = sender->side();
      uint64_t seq = 0;
      if (fw > 0 && fh > 0 && sender->takeSlot(&seq, &ringSlot)) {
        ring = sender->ring();
        ringFS = sender->ringFrameWords();
        ringRW = sender->ringRowWords();
        if (slots[p].w != fw || slots[p].h != fh) makeSlot(slots[p], fw, fh);
        if (cur >= 0) picture++;
        cur = p;
        fresh = true;
        // ai: the schedule: present i shows this picture, and the next is due a period on, from now where this
        // ai: one came late (a late picture holds its whole period; the same as before at one refresh a picture)
        const double per = period();
        due = std::max(due + per, i + per);
      }
    }
    if (!fresh && cur >= 0) { win.behind++; all.behind++; }
  }

  VkCommandBuffer cb = cbs[fl];
  D.vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  D.vkBeginCommandBuffer(cb, &bi);
  if (fresh && ring) {
    Slot& s = slots[cur];
    ringHeld[fl] = ring;
    VkDescriptorBufferInfo bi{ring->buf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, s.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};
    w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = dsets[fl];
    w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi;
    w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &ii;
    D.vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    // ai: the ring's frame, painted on the painter's queue and complete before the sender marked it ready (its fence
    // ai: waited there): made visible to this queue's compute stage
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = 0;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    D.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    barrier(D, cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    D.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, expandPipe);
    D.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dsets[fl], 0, nullptr);
    const uint32_t pc[4] = {static_cast<uint32_t>(ringSlot) * ringFS, ringRW, static_cast<uint32_t>(s.w), static_cast<uint32_t>(s.h)};
    D.vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    D.vkCmdDispatch(cb, (static_cast<uint32_t>(s.w) + 15) / 16, (static_cast<uint32_t>(s.h) + 15) / 16, 1);
    barrier(D, cb, s.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  }
  const VkImage img = c.images[idx];
  barrier(D, cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  const VkClearColorValue white{{1.f, 1.f, 1.f, 1.f}};
  const VkImageSubresourceRange whole1{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  D.vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &whole1);
  if (cur >= 0) {
    // ai: where the picture goes: centred, the largest that fits with the edge's light round it, times the size share,
    // ai: stretched (linear) as the page's fit=stretch; or whole device pixels a sample (nearest) where at least one fits
    const Slot& s = slots[cur];
    const double e = 2 * edge.load() * s.h;
    const double fit = std::min(static_cast<double>(c.extent.width) / (s.w + e), static_cast<double>(c.extent.height) / (s.h + e));
    const double share = std::clamp(size.load(), 0.25, 1.0);
    double scale = fit * share;
    VkFilter filter = VK_FILTER_LINEAR;
    if (whole && fit >= 1) {
      scale = std::max(1.0, std::floor(fit * share));
      filter = VK_FILTER_NEAREST;
    }
    const int dw = std::min(static_cast<int>(c.extent.width), static_cast<int>(std::lround(s.w * scale)));
    const int dh = std::min(static_cast<int>(c.extent.height), static_cast<int>(std::lround(s.h * scale)));
    if (dw > 0 && dh > 0) {
      barrier(D, cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
      VkImageBlit blit{};
      blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      blit.srcOffsets[1] = {s.w, s.h, 1};
      blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      blit.dstOffsets[0] = {(static_cast<int>(c.extent.width) - dw) / 2, (static_cast<int>(c.extent.height) - dh) / 2, 0};
      blit.dstOffsets[1] = {blit.dstOffsets[0].x + dw, blit.dstOffsets[0].y + dh, 1};
      D.vkCmdBlitImage(cb, s.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, filter);
    }
    slots[cur].usedBy = frame;
  }
  barrier(D, cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  D.vkEndCommandBuffer(cb);

  const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &acquired[fl];
  si.pWaitDstStageMask = &waitStage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &c.done[idx];
  vkCheck(qSubmit(si, fences[fl]), "vkQueueSubmit");
  const double sub = nowMs() - t0;

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &c.done[idx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &c.sc;
  pi.pImageIndices = &idx;
  const uint64_t id = ++pid;
  VkPresentIdKHR pidInfo{VK_STRUCTURE_TYPE_PRESENT_ID_KHR};
  pidInfo.swapchainCount = 1;
  pidInfo.pPresentIds = &id;
  if (presentWait) pi.pNext = &pidInfo;
  r = qPresent(pi);
  if (r == VK_ERROR_OUT_OF_DATE_KHR) resized = true;
  else if (r == VK_SUBOPTIMAL_KHR) { if (!ignoreSuboptimal) suboptimal = true; }
  else vkCheck(r, "vkQueuePresentKHR");
  if (live && cur >= 0) {
    Shown s;
    s.id = id;
    s.picture = picture;
    s.fresh = fresh;
    s.acquiredMs = acq;
    s.submittedMs = sub;
    s.epoch = epoch;
    s.queued = presentWait ? static_cast<int>(id - waiter.shownId.load()) : -1;
    waiter.push(s);
  }
  if (live) presents++;
  winPresents++;
  frame++;
}

void Presenter::Impl::run() {
  try {
    while (!quit) {
      if (vsyncDirty.exchange(false)) holdMode();
      const double now = nowMs();
      if (now - lastWin >= 1000) {
        window((now - lastWin) / 1000);
        lastWin = now;
      }
      const bool p = paused;
      if (p) {
        // ai: paused: the last picture presented again a few times a second (a resize or an expose shows it), none taken
        wasPaused = true;
        std::unique_lock<std::mutex> l(wakeMu);
        wake.wait_for(l, std::chrono::milliseconds(100), [&] { return quit.load() || !paused.load(); });
        if (quit) break;
        if (!paused) continue;
      } else if (wasPaused) {
        // ai: resumed: a new picture due now, no gap counted across the pause
        due = static_cast<double>(presents);
        epoch++;
        wasPaused = false;
      }
      presentOnce(!p);
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> l(statsMu);
    error = e.what();
  }
  if (!error.empty()) say("presenter: stopped: " + error);
}

void Presenter::Impl::teardown() {
  waiter.finish();
  // ai: the device is the sender's: this presenter frees what it made on it and leaves the device and the instance
  if (dev && D.vkQueueWaitIdle) {
    qIdle();
    // ai: the last window, however short (its secs say), and the run's totals with every present counted
    if (t0 > 0) window((nowMs() - lastWin) / 1000, true);
    for (VkFence f : fences) if (f) D.vkDestroyFence(dev, f, nullptr);
    for (VkSemaphore a : acquired) if (a) D.vkDestroySemaphore(dev, a, nullptr);
    if (!cbs.empty() && pool) D.vkFreeCommandBuffers(dev, pool, static_cast<uint32_t>(cbs.size()), cbs.data());
    fences.clear();
    acquired.clear();
    cbs.clear();
    if (pool) D.vkDestroyCommandPool(dev, pool, nullptr);
    pool = VK_NULL_HANDLE;
    for (auto& sl : slots) freeSlot(sl);
    for (VkSemaphore a : c.done) D.vkDestroySemaphore(dev, a, nullptr);
    if (c.sc) D.vkDestroySwapchainKHR(dev, c.sc, nullptr);
    c = Chain{};
    ringHeld.clear();
    dsets.clear();
    if (dpool) D.vkDestroyDescriptorPool(dev, dpool, nullptr);
    if (expandPipe) D.vkDestroyPipeline(dev, expandPipe, nullptr);
    if (pl) D.vkDestroyPipelineLayout(dev, pl, nullptr);
    if (dsl) D.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    if (expandSm) D.vkDestroyShaderModule(dev, expandSm, nullptr);
    dpool = VK_NULL_HANDLE; expandPipe = VK_NULL_HANDLE; pl = VK_NULL_HANDLE; dsl = VK_NULL_HANDLE; expandSm = VK_NULL_HANDLE;
  }
  if (surf && I.vkDestroySurfaceKHR) I.vkDestroySurfaceKHR(inst, surf, nullptr);
  surf = VK_NULL_HANDLE;
  dev = VK_NULL_HANDLE;
  inst = VK_NULL_HANDLE;
  releaseMode();
#if LIZ_X11 && !defined(_WIN32)
  if (xq) XCloseDisplay(xq);
  if (dpy) XCloseDisplay(dpy);
  xq = dpy = nullptr;
#endif
  if (csv) std::fclose(csv);
  csv = nullptr;
}

const wg::DeviceExtras& Presenter::deviceExtras() {
  static const wg::DeviceExtras extras = [] {
    wg::DeviceExtras x;
    x.instanceExts = {VK_KHR_SURFACE_EXTENSION_NAME,
#if defined(_WIN32)
                      VK_KHR_WIN32_SURFACE_EXTENSION_NAME
#elif LIZ_X11
                      VK_KHR_XLIB_SURFACE_EXTENSION_NAME
#elif defined(__APPLE__)
                      VK_EXT_METAL_SURFACE_EXTENSION_NAME
#endif
    };
    x.deviceExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_KHR_PRESENT_WAIT_EXTENSION_NAME};
    x.graphics = true;
    return x;
  }();
  return extras;
}

std::unique_ptr<Presenter> Presenter::start(const PresentTarget& target, Sender* sender, std::mutex* senderLock,
                                            const PresentOptions& opt, std::string* err) {
  if (!sender || !senderLock) {
    if (err) *err = "no sender to present";
    return nullptr;
  }
  auto p = std::make_unique<Impl>();
  p->target = target;
  p->sender = sender;
  p->senderLock = senderLock;
  p->opt = opt;
  p->fps = std::clamp(opt.fps, 1, 240);
  p->size = std::clamp(opt.size, 0.25, 1.0);
  p->edge = std::clamp(opt.edge, 0.0, 1.0);
  p->whole = opt.whole;
  // ai: VSync's first look on the present thread's first turn, after start returns (a mode change takes a second or
  // ai: two, which the caller, the UI's thread, would wait out here)
  p->vsync = opt.vsync;
  p->vsyncDirty = opt.vsync;
  std::promise<std::string> ready;
  std::future<std::string> started = ready.get_future();
  Impl* raw = p.get();
  // ai: the setup on the present thread too, so every window and Vulkan call is that thread's
  raw->thread = std::thread([raw, pr = std::move(ready)]() mutable {
    try {
      raw->setup();
    } catch (const std::exception& e) {
      raw->teardown();
      pr.set_value(e.what());
      return;
    }
    pr.set_value("");
    raw->run();
    raw->teardown();
  });
  const std::string why = started.get();
  if (!why.empty()) {
    raw->thread.join();
    if (err) *err = why;
    return nullptr;
  }
  return std::unique_ptr<Presenter>(new Presenter(std::move(p)));
}

Presenter::Presenter(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

void Presenter::setFps(int fps) {
  p_->fps = std::clamp(fps, 1, 240);
  p_->vsyncDirty = true;
}
void Presenter::setVsync(bool on) {
  p_->vsync = on;
  p_->vsyncDirty = true;
}
void Presenter::setSize(double size, double edge) {
  p_->size = std::clamp(size, 0.25, 1.0);
  p_->edge = std::clamp(edge, 0.0, 1.0);
}
void Presenter::setPaused(bool on) {
  {
    std::lock_guard<std::mutex> l(p_->wakeMu);
    p_->paused = on;
  }
  p_->wake.notify_all();
}

std::string Presenter::stats() {
  std::lock_guard<std::mutex> l(p_->statsMu);
  return p_->statsJson;
}

std::string Presenter::totals() {
  std::lock_guard<std::mutex> l(p_->statsMu);
  return p_->totalsJson;
}

void Presenter::stop() {
  {
    std::lock_guard<std::mutex> l(p_->wakeMu);
    p_->quit = true;
  }
  p_->wake.notify_all();
  if (p_->thread.joinable()) p_->thread.join();
}

Presenter::~Presenter() { stop(); }

}  // namespace lizard
