// ai: The desktop sender's presenter (presenter.h): the test sender lizard_present's swapchain, slots, waits and tallies
// ai: (2026-10-03) as a class with one thread of its own.
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
#include "vk_present_timing.h"
#include "wg.h"

#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif LIZ_X11
#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include <vulkan/vulkan_xlib.h>
#include <vulkan/vulkan_xlib_xrandr.h>
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
  X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCreateBuffer) X(vkDestroyBuffer)          \
  X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkCmdCopyBufferToImage)
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
  // ai: on present timing (2026-10-31): the refresh this present was for (ms, -1 none) and its shown time the display's
  // ai: first pixel; a present then stands for its picture's whole hold, so no refresh between two is a miss
  double targetMs = -1;
  bool timed = false;
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
  // ai: present timing's (2026-10-31): pictures on screen a refresh or more past their target, the schedule moved to
  // ai: the display's times, the largest first-pixel time off its target (ms)
  uint64_t late = 0, slips = 0;
  double errMaxMs = 0;
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
    if (s.timed && s.targetMs >= 0) {
      const double err = s.shownMs - s.targetMs;
      errMaxMs = std::max(errMaxMs, std::fabs(err));
      if (s.fresh && err >= 0.5 * refreshMs) late++;
    }
    if (lastShown >= 0) {
      const double gap = s.shownMs - lastShown;
      gaps.push_back(gap);
      const long k = std::lround(gap / refreshMs);
      if (k >= 2 && !s.timed) missed += static_cast<uint64_t>(k - 1);
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

std::string x11OutputAt(unsigned long window) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return "";
  std::string name;
  int ev = 0, er = 0;
  XWindowAttributes wa{};
  if (XRRQueryExtension(d, &ev, &er) && XGetWindowAttributes(d, static_cast<Window>(window), &wa)) {
    int cx = 0, cy = 0;
    Window child = 0;
    XTranslateCoordinates(d, static_cast<Window>(window), wa.root, wa.width / 2, wa.height / 2, &cx, &cy, &child);
    if (XRRScreenResources* res = XRRGetScreenResourcesCurrent(d, wa.root)) {
      if (const RRCrtc crtc = crtcAt(d, res, cx, cy)) {
        if (XRRCrtcInfo* ci = XRRGetCrtcInfo(d, res, crtc)) {
          if (ci->noutput > 0)
            if (XRROutputInfo* oi = XRRGetOutputInfo(d, res, ci->outputs[0])) { name = oi->name; XRRFreeOutputInfo(oi); }
          XRRFreeCrtcInfo(ci);
        }
      }
      XRRFreeScreenResources(res);
    }
  }
  XCloseDisplay(d);
  return name;
#else
  (void)window;
  return "";
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
  std::atomic<bool> paused{false}, quit{false};
  // ai: VSync (setVsync), and a change of it or of fps for the loop to take at its next present
  std::atomic<bool> vsync{false}, vsyncDirty{false};
  std::mutex wakeMu;
  std::condition_variable wake;
  std::thread thread;

#if LIZ_X11 && !defined(_WIN32)
  // ai: two connections of the present thread's own: dpy the surface's (the driver's alone; AWT never calls
  // ai: XInitThreads, so no Xlib call of ours shares it with a driver thread's xcb), xq for the window queries (the size,
  // ai: the mode)
  Display* dpy = nullptr;
  Display* xq = nullptr;
  Window xwin = 0;
  int randr = -1;   // ai: XRandR on xq: -1 not asked yet, 0 no, 1 yes
#endif
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
  Tally win;
  FILE* csv = nullptr;
  // ai: Present timing (VK_EXT_present_timing, 2026-10-31): each present carries the refresh it is for, an absolute
  // ai: time on the display's own refresh grid, and the display reports when its first pixel left for the screen.
  // ai: timingDev: the device took the extensions (wg); timing: this swapchain presents on them (its surface takes
  // ai: absolute targets and reports the first-pixel stage); refreshNs the display's refresh as it reports it; the
  // ai: grid: refresh slot k at anchorNs + (k - anchorSlot) refreshNs in the swapchain's time domain, set from a
  // ai: first-pixel time. Without them the presents count refreshes and present_wait times them, as before.
  struct TimingFns {
    PFN_vkSetSwapchainPresentTimingQueueSizeEXT setQueueSize = nullptr;
    PFN_vkGetSwapchainTimingPropertiesEXT props = nullptr;
    PFN_vkGetSwapchainTimeDomainPropertiesEXT domains = nullptr;
    PFN_vkGetPastPresentationTimingEXT past = nullptr;
    PFN_vkGetCalibratedTimestampsKHR calibrate = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR caps2 = nullptr;
  } T;
  bool timingDev = false, timing = false, anchored = false;
  VkPresentStageFlagsEXT stages = 0;
  VkTimeDomainKHR domain = VK_TIME_DOMAIN_DEVICE_KHR;
  uint64_t domainId = 0, refreshNs = 0, propsCounter = 0, domainsCounter = 0, anchorNs = 0;
  double anchorSlot = 0;
  // ai: a present's record until its timing comes back: what it showed, the slot it was for (refreshes on the grid,
  // ai: or the present's count before the grid is set), its target (0 none)
  struct Sent { Shown s; double slot = 0; uint64_t targetNs = 0; };
  std::map<uint64_t, Sent> sent;
  std::vector<Shown> timedDone;   // ai: presents whose timing came back, for drain
  uint64_t lastDone = 0;   // ai: the last present whose first pixel was reported
  int lateRun = 0;         // ai: fresh pictures late in a row (three move the grid: the display's times win)
  // ai: the domain against the host's steady clock (vkGetCalibratedTimestampsKHR, once a second): ms from t0 of a
  // ai: domain time; calOk false where it cannot be read
  double calHostMs = 0, calAt = -1e9;
  uint64_t calNs = 0;
  bool calOk = false;
  std::vector<VkPastPresentationTimingEXT> past;
  std::vector<VkPresentStageTimeEXT> pastStages;
  // ai: the display taken (PresentOptions::direct): its handle, the mode's size and refresh
  VkDisplayKHR display = VK_NULL_HANDLE;
  bool displayTaken = false;
  VkExtent2D displayExtent{};
  double displayHz = 0;
  PFN_vkReleaseDisplayEXT releaseDisplay = nullptr;
#if LIZ_X11 && !defined(_WIN32)
  // ai: the desktop's layout before the display was taken (every CRTC's mode, place, rotation and outputs, the primary
  // ai: output, the screen's size), put back once it is given back: taking an output turns it off first, and the
  // ai: window manager may lay the other monitors out again meanwhile
  struct CrtcWas { RRCrtc crtc = 0; RRMode mode = 0; int x = 0, y = 0; Rotation rot = RR_Rotate_0; std::vector<RROutput> outs; };
  std::vector<CrtcWas> layoutWas;
  RROutput primaryWas = 0;
  int fbW = 0, fbH = 0, fbMmW = 0, fbMmH = 0;
  void saveLayout();
  void restoreLayout();
  // ai: the guard for a crash (armGuard): a shell waiting on a pipe this process holds the other end of; its end
  // ai: closed without the word "given-back" (the process gone while the display was taken), it puts the saved
  // ai: layout back with xrandr (the lease itself ends with the process; the output would stay off without this)
  int guardFd = -1;
  void armGuard();
  void disarmGuard();
  // ai: while the only monitor is taken the keyboard and the pointer are this presenter's (XGrabKeyboard, XGrabPointer
  // ai: on the app's window, else the root where it is not viewable): nothing else takes the input no screen shows, and
  // ai: a key press or a click, any, gives the display back (leaveDisplay). Where another monitor stays on the desktop
  // ai: nothing is grabbed: the app's window there is the control (its Stop, Pause and Fullscreen, and its keys)
  bool grabbed = false;
  double grabTried = -1e9;
  void grabInput();
  void ungrabInput();
  bool inputArrived();
#endif
  // ai: (outside the X11 block: the start line and the stats read them on every system; set where a display is taken)
  bool grabToLeave = true;
  std::string directName;   // ai: the taken monitor's own name (its EDID's), else the output's; the stats' directName
  bool leaseEnded = false;   // ai: the display given back by a key or a click (the stats' leaseEnded), the window since
  void leaveDisplay(const std::string& why);
  // ai: the mark (PresentOptions::mark): a host-visible strip an image in flight, copied into the image's corner
  static constexpr int MARK_BITS = 12, MARK_PX = 8, MARK_W = 2 * MARK_BITS * MARK_PX;
  VkBuffer markBuf = VK_NULL_HANDLE;
  VkDeviceMemory markMem = VK_NULL_HANDLE;
  uint8_t* markPtr = nullptr;
  size_t markSlots = 0;

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
  std::string statsJson, error;

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
  void freeMark();
  VkExtent2D windowSize();
  void queryMode();
  void holdMode();
  void releaseMode();
  void rateMoved();
  void drain();
  void window(double secs, bool last = false);
  // ai: present timing (2026-10-31)
  void timingSetup(VkSwapchainKHR sc);
  void readTimingProps();
  bool calibrate();
  double hostMs(uint64_t ns) const { return calHostMs + (static_cast<double>(ns) - static_cast<double>(calNs)) / 1e6; }
  uint64_t gridNs(double slot) const;
  double nowSlot();
  void pollTiming();
  std::string takeDisplay();
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
  // ai: present timing where wg took it (its extensions and features): its calls from the device, the surface query
  // ai: from the instance (VK_KHR_get_surface_capabilities2); any missing, none of it
  timingDev = W->has(VK_EXT_PRESENT_TIMING_EXTENSION_NAME) && W->has(VK_KHR_PRESENT_ID_2_EXTENSION_NAME);
  if (timingDev) {
    T.setQueueSize = reinterpret_cast<PFN_vkSetSwapchainPresentTimingQueueSizeEXT>(I.vkGetDeviceProcAddr(dev, "vkSetSwapchainPresentTimingQueueSizeEXT"));
    T.props = reinterpret_cast<PFN_vkGetSwapchainTimingPropertiesEXT>(I.vkGetDeviceProcAddr(dev, "vkGetSwapchainTimingPropertiesEXT"));
    T.domains = reinterpret_cast<PFN_vkGetSwapchainTimeDomainPropertiesEXT>(I.vkGetDeviceProcAddr(dev, "vkGetSwapchainTimeDomainPropertiesEXT"));
    T.past = reinterpret_cast<PFN_vkGetPastPresentationTimingEXT>(I.vkGetDeviceProcAddr(dev, "vkGetPastPresentationTimingEXT"));
    T.calibrate = reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(I.vkGetDeviceProcAddr(dev, "vkGetCalibratedTimestampsKHR"));
    T.caps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
    timingDev = T.setQueueSize && T.props && T.domains && T.past && T.calibrate && T.caps2;
  }

#if !(LIZ_X11 && !defined(_WIN32))
  if (!opt.direct.empty()) say("presenter: the display " + opt.direct + " not taken (no direct display on this system): the window instead");
#endif
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
  if (!opt.direct.empty()) {
    const std::string why = takeDisplay();
    if (!why.empty()) say("presenter: the display " + opt.direct + " not taken (" + why + "): the window instead");
  }
  auto createSurface = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateXlibSurfaceKHR"));
  if (!createSurface) throw std::runtime_error("the sender's Vulkan instance has no VK_KHR_xlib_surface");
  VkXlibSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  si.dpy = dpy;
  si.window = xwin;
  if (!surf) vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateXlibSurfaceKHR");
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
    std::fprintf(csv, "id,picture,fresh,acquired_ms,submitted_ms,shown_ms,queued,target_ms\n");
  }
  queryMode();
  t0 = lastWin = nowMs();
  // ai: the CSV's origin on the steady clock (CLOCK_MONOTONIC on Linux), so tools on other clocks' readings line up
  if (csv) std::fprintf(csv, "# t0_steady_ms=%s\n", fmt("%.3f", t0).c_str());
  window(0);
  say("presenter: " + name + (shareQueue ? ", the painter's queue shared" : ", the family's second queue") +
      (displayTaken ? ", the display " + opt.direct + " taken (" + std::to_string(displayExtent.width) + " x " +
                          std::to_string(displayExtent.height) + " at " + fmt("%.3f", displayHz) + " Hz; " +
                          (grabToLeave ? "the only monitor: any key or click gives it back" : "the window on another monitor the control") + ")" : "") +
      (timingDev ? ", present timing on the device" : "") +
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
  // ai: the display taken: its mode is the presenter's own choice (takeDisplay)
  if (displayTaken) { modeHz = displayHz; return; }
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

// ai: VSync's hold (presenter.h setVsync), at a change of VSync or fps and once a second: the monitor under the window's
// ai: centre and the refresh of its mode before ours; where fps divides that, its own mode (ours put back), else the
// ai: mode at the same size pickRate takes, set. Each monitor and fps is looked at once (held.decided).
void Presenter::Impl::holdMode() {
  // ai: the display taken shows a mode chosen for the asked rate already (takeDisplay); the desktop's modes are not ours
  if (displayTaken) return;
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
  anchored = false;
}

// ai: Present timing on a new swapchain (2026-10-31): a results queue of four chain-lengths (a present asking for its
// ai: times takes a slot until they are read), the display's refresh, the time domain (the host's steady clock where
// ai: the swapchain offers it, else the device's, else the swapchain's own: each converted to the host's by
// ai: calibration), and the grid unset until a first-pixel time comes back
void Presenter::Impl::timingSetup(VkSwapchainKHR sc) {
  const uint32_t q = static_cast<uint32_t>(std::max<size_t>(64, 4 * c.images.size()));
  if (T.setQueueSize(dev, sc, q) != VK_SUCCESS) { timing = false; say("presenter: present timing refused its results queue"); return; }
  past.assign(q, VkPastPresentationTimingEXT{});
  pastStages.assign(q, VkPresentStageTimeEXT{});
  refreshNs = 0;
  readTimingProps();
  domainId = 0;
  domain = VK_TIME_DOMAIN_DEVICE_KHR;
  uint32_t n = 0;
  VkSwapchainTimeDomainPropertiesEXT dp{VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT};
  if (T.domains(dev, sc, &dp, nullptr) == VK_SUCCESS && (n = dp.timeDomainCount) > 0) {
    std::vector<VkTimeDomainKHR> doms(n);
    std::vector<uint64_t> ids(n);
    dp.pTimeDomains = doms.data();
    dp.pTimeDomainIds = ids.data();
    T.domains(dev, sc, &dp, &domainsCounter);
#if defined(_WIN32)
    const VkTimeDomainKHR host = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
    const VkTimeDomainKHR host = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
    int pick = -1;
    for (const VkTimeDomainKHR want : {host, VK_TIME_DOMAIN_DEVICE_KHR, VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT})
      for (uint32_t k = 0; k < dp.timeDomainCount && pick < 0; k++)
        if (doms[k] == want) pick = static_cast<int>(k);
    if (pick < 0) pick = 0;
    domain = doms[static_cast<size_t>(pick)];
    domainId = ids[static_cast<size_t>(pick)];
  }
  // ai: a grid dropped: the schedule counts presents again (the grid's slots ran ahead of the count at one present a
  // ai: picture), from the next present on
  if (anchored) due = static_cast<double>(presents);
  anchored = false;
  sent.clear();
  lastDone = 0;
  lateRun = 0;
  calAt = -1e9;
  calOk = false;
}

// ai: the display's refresh (VK_NOT_READY until a present has gone out on some platforms: asked again then); a
// ai: refresh that moved drops the grid
void Presenter::Impl::readTimingProps() {
  VkSwapchainTimingPropertiesEXT tp{VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT};
  if (T.props(dev, c.sc, &tp, &propsCounter) != VK_SUCCESS || !tp.refreshDuration) return;
  // ai: a variable refresh (refreshInterval UINT64_MAX) has no grid to hold pictures on: the present count instead
  if (tp.refreshInterval == UINT64_MAX) {
    if (anchored) due = static_cast<double>(presents);
    timing = anchored = false;
    say("presenter: a variable refresh: presents count refreshes");
    return;
  }
  if (refreshNs && tp.refreshDuration != refreshNs && anchored) { anchored = false; due = static_cast<double>(presents); }
  refreshNs = tp.refreshDuration;
}

// ai: the domain's time and the host's steady clock read together (vkGetCalibratedTimestampsKHR), once a second
bool Presenter::Impl::calibrate() {
  if (nowMs() - calAt < 1000 && calOk) return true;
  calAt = nowMs();
  VkSwapchainCalibratedTimestampInfoEXT sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT};
  sci.swapchain = c.sc;
  sci.timeDomainId = domainId;
  VkCalibratedTimestampInfoKHR ti[2]{{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}, {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}};
  ti[0].timeDomain = domain;
  if (domain == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT) ti[0].pNext = &sci;
#if defined(_WIN32)
  ti[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
#else
  ti[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR;
#endif
  uint64_t ts[2] = {0, 0}, dev2 = 0;
  const bool hostDomain = domain == ti[1].timeDomain;
  if (T.calibrate(dev, hostDomain ? 1 : 2, hostDomain ? &ti[1] : ti, ts, &dev2) != VK_SUCCESS) { calOk = false; return false; }
  if (hostDomain) ts[1] = ts[0];
  double hostNs = static_cast<double>(ts[1]);
#if defined(_WIN32)
  LARGE_INTEGER f;
  QueryPerformanceFrequency(&f);
  hostNs = hostNs * 1e9 / static_cast<double>(f.QuadPart);
#endif
  // ai: the steady clock's ms (nowMs: CLOCK_MONOTONIC on Linux, QPC on Windows) from t0
  calNs = ts[0];
  calHostMs = hostNs / 1e6 - t0;
  calOk = true;
  return true;
}

uint64_t Presenter::Impl::gridNs(double slot) const {
  const double k = std::round(slot) - anchorSlot;
  return static_cast<uint64_t>(static_cast<double>(anchorNs) + k * static_cast<double>(refreshNs));
}

// ai: the grid's slot now (fractional), from the calibration
double Presenter::Impl::nowSlot() {
  if (!calibrate()) return -1e18;
  const double nowNs = static_cast<double>(calNs) + (nowMs() - t0 - calHostMs) * 1e6;
  return anchorSlot + (nowNs - static_cast<double>(anchorNs)) / static_cast<double>(refreshNs);
}

// ai: The timings back so far: each present's first-pixel time against its target, into the tally as a Shown with the
// ai: display's own time (ms on the host's clock); the first one sets the grid (its time is its slot's); three fresh
// ai: pictures late in a row move the grid to where the display put them (the schedule slips once, counted, rather
// ai: than every picture after it counting late)
void Presenter::Impl::pollTiming() {
  if (!timing || past.empty()) return;
  if (!refreshNs) readTimingProps();
  for (size_t k = 0; k < past.size(); k++) {
    past[k] = VkPastPresentationTimingEXT{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT};
    past[k].presentStageCount = 1;
    past[k].pPresentStages = &pastStages[k];
  }
  VkPastPresentationTimingInfoEXT info{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT};
  info.swapchain = c.sc;
  VkPastPresentationTimingPropertiesEXT pp{VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT};
  pp.presentationTimingCount = static_cast<uint32_t>(past.size());
  pp.pPresentationTimings = past.data();
  const VkResult r = T.past(dev, &info, &pp);
  if (r != VK_SUCCESS && r != VK_INCOMPLETE) return;
  if (pp.timingPropertiesCounter != propsCounter) readTimingProps();
  if (pp.timeDomainsCounter != domainsCounter) { timingSetup(c.sc); return; }
  if (!calibrate()) return;
  for (uint32_t k = 0; k < pp.presentationTimingCount; k++) {
    const VkPastPresentationTimingEXT& t = past[k];
    auto it = sent.find(t.presentId);
    if (it == sent.end()) continue;
    Sent x = it->second;
    sent.erase(it);
    uint64_t out = 0;
    for (uint32_t j = 0; j < t.presentStageCount; j++)
      if (t.pPresentStages[j].stage == VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) out = t.pPresentStages[j].time;
    lastDone = std::max(lastDone, t.presentId);
    x.s.timed = true;
    x.s.shownMs = out ? hostMs(out) : -1;
    x.s.targetMs = x.targetNs ? hostMs(x.targetNs) : -1;
    if (out && refreshNs && !anchored) {
      anchored = true;
      anchorNs = out;
      anchorSlot = std::round(x.slot);
    } else if (out && x.targetNs && x.s.fresh) {
      const double lateBy = (static_cast<double>(out) - static_cast<double>(x.targetNs)) / static_cast<double>(refreshNs);
      if (lateBy >= 0.5) {
        if (++lateRun >= 3) {
          anchorNs += static_cast<uint64_t>(std::llround(lateBy)) * refreshNs;
          win.slips++;
          lateRun = 0;
        }
      } else lateRun = 0;
    }
    timedDone.push_back(x.s);
  }
}

#if LIZ_X11 && !defined(_WIN32)
// ai: The display taken from the X server (PresentOptions::direct): the RandR output by name, its Vulkan display
// ai: (vkGetRandROutputDisplayEXT), taken (vkAcquireXlibDisplayEXT: the server leases it away from the desktop), a mode
// ai: of the output's present size at a refresh the asked rate divides (pickRate), else the nearest 60 Hz there, a
// ai: plane that shows this display, and the display's own surface on them (vkCreateDisplayPlaneSurfaceKHR). "" or why
// ai: not; given back (vkReleaseDisplayEXT) at teardown, or here where a later step fails.
// ai: a monitor's own name from its EDID (the display descriptor tagged 0xFC, up to 13 characters ended by a line
// ai: feed), as a settings dialog shows it; empty where the output has no EDID or none names it
static std::string monitorName(Display* xq, RROutput out) {
  std::string name;
  const Atom edid = XInternAtom(xq, "EDID", True);
  if (edid == None) return name;
  Atom type = None;
  int format = 0;
  unsigned long n = 0, after = 0;
  unsigned char* data = nullptr;
  if (XRRGetOutputProperty(xq, out, edid, 0, 64, False, False, AnyPropertyType, &type, &format, &n, &after, &data) == Success && data) {
    for (int d = 54; d + 18 <= 126 && static_cast<unsigned long>(d + 18) <= n; d += 18)
      if (!data[d] && !data[d + 1] && !data[d + 2] && data[d + 3] == 0xFC) {
        for (int k = 5; k < 18 && data[d + k] != '\n' && data[d + k]; k++) name += static_cast<char>(data[d + k]);
        while (!name.empty() && name.back() == ' ') name.pop_back();
        break;
      }
    XFree(data);
  }
  return name;
}

std::string Presenter::Impl::takeDisplay() {
  auto getOutput = reinterpret_cast<PFN_vkGetRandROutputDisplayEXT>(vkGetInstanceProcAddr(inst, "vkGetRandROutputDisplayEXT"));
  auto acquire = reinterpret_cast<PFN_vkAcquireXlibDisplayEXT>(vkGetInstanceProcAddr(inst, "vkAcquireXlibDisplayEXT"));
  releaseDisplay = reinterpret_cast<PFN_vkReleaseDisplayEXT>(vkGetInstanceProcAddr(inst, "vkReleaseDisplayEXT"));
  auto modesOf = reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(vkGetInstanceProcAddr(inst, "vkGetDisplayModePropertiesKHR"));
  auto planesOf = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR>(vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR"));
  auto planeDisplays = reinterpret_cast<PFN_vkGetDisplayPlaneSupportedDisplaysKHR>(vkGetInstanceProcAddr(inst, "vkGetDisplayPlaneSupportedDisplaysKHR"));
  auto createSurface = reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateDisplayPlaneSurfaceKHR"));
  if (!getOutput || !acquire || !releaseDisplay || !modesOf || !planesOf || !planeDisplays || !createSurface)
    return "the Vulkan instance has no VK_KHR_display with VK_EXT_acquire_xlib_display";
  int ev = 0, er = 0;
  if (!XRRQueryExtension(xq, &ev, &er)) return "no XRandR";
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, DefaultRootWindow(xq));
  if (!res) return "no XRandR resources";
  RROutput out = 0;
  uint32_t w = 0, h = 0;
  for (int i = 0; i < res->noutput && !out; i++) {
    XRROutputInfo* oi = XRRGetOutputInfo(xq, res, res->outputs[i]);
    if (!oi) continue;
    if (opt.direct == oi->name && oi->connection == RR_Connected) {
      out = res->outputs[i];
      if (XRRCrtcInfo* ci = oi->crtc ? XRRGetCrtcInfo(xq, res, oi->crtc) : nullptr) {
        w = ci->width;
        h = ci->height;
        XRRFreeCrtcInfo(ci);
      }
    }
    XRRFreeOutputInfo(oi);
  }
  XRRFreeScreenResources(res);
  if (!out) return "no connected output of that name";
  VkDisplayKHR d = VK_NULL_HANDLE;
  if (getOutput(pd, dpy, out, &d) != VK_SUCCESS || !d) return name + " shows no Vulkan display on that output";
  // ai: an output the desktop shows is not leased (NVIDIA: VK_ERROR_UNKNOWN): its CRTC turned off first, the screen's
  // ai: size kept, the layout kept to put back
  saveLayout();
  if (XRRScreenResources* r2 = XRRGetScreenResourcesCurrent(xq, DefaultRootWindow(xq))) {
    if (XRROutputInfo* oi = XRRGetOutputInfo(xq, r2, out)) {
      if (oi->crtc) XRRSetCrtcConfig(xq, r2, oi->crtc, CurrentTime, 0, 0, None, RR_Rotate_0, nullptr, 0);
      XRRFreeOutputInfo(oi);
    }
    XRRFreeScreenResources(r2);
  }
  XSync(xq, False);
  const VkResult ar = acquire(pd, dpy, d);
  if (ar != VK_SUCCESS) {
    restoreLayout();
    return "the X server kept it (VkResult " + std::to_string(static_cast<int>(ar)) + ")";
  }
  display = d;
  displayTaken = true;
  grabToLeave = layoutWas.size() <= 1;
  directName = monitorName(xq, out);
  if (directName.empty()) directName = opt.direct;
  armGuard();
  auto giveBack = [&](const std::string& why) {
    releaseDisplay(pd, display);
    displayTaken = false;
    display = VK_NULL_HANDLE;
    restoreLayout();
    disarmGuard();
    return why;
  };
  uint32_t nm = 0;
  modesOf(pd, display, &nm, nullptr);
  std::vector<VkDisplayModePropertiesKHR> modes(nm);
  modesOf(pd, display, &nm, modes.data());
  if (!w || !h) {
    for (const auto& m : modes) if (m.parameters.visibleRegion.width * m.parameters.visibleRegion.height > w * h) { w = m.parameters.visibleRegion.width; h = m.parameters.visibleRegion.height; }
  }
  std::vector<double> hz;
  std::vector<size_t> at;
  for (size_t k = 0; k < modes.size(); k++)
    if (modes[k].parameters.visibleRegion.width == w && modes[k].parameters.visibleRegion.height == h && modes[k].parameters.refreshRate) {
      hz.push_back(modes[k].parameters.refreshRate / 1000.0);
      at.push_back(k);
    }
  if (hz.empty()) return giveBack("no mode at " + std::to_string(w) + " x " + std::to_string(h));
  int b = pickRate(hz, std::max(1, fps.load()));
  if (b < 0) {
    b = 0;
    for (size_t k = 1; k < hz.size(); k++) if (std::fabs(hz[k] - 60) < std::fabs(hz[static_cast<size_t>(b)] - 60)) b = static_cast<int>(k);
  }
  const VkDisplayModePropertiesKHR& mode = modes[at[static_cast<size_t>(b)]];
  uint32_t np = 0;
  planesOf(pd, &np, nullptr);
  std::vector<VkDisplayPlanePropertiesKHR> planes(np);
  planesOf(pd, &np, planes.data());
  uint32_t plane = UINT32_MAX;
  for (uint32_t p = 0; p < np && plane == UINT32_MAX; p++) {
    if (planes[p].currentDisplay && planes[p].currentDisplay != display) continue;
    uint32_t nd = 0;
    planeDisplays(pd, p, &nd, nullptr);
    std::vector<VkDisplayKHR> ds(nd);
    planeDisplays(pd, p, &nd, ds.data());
    for (VkDisplayKHR x : ds) if (x == display) plane = p;
  }
  if (plane == UINT32_MAX) return giveBack("no plane shows it");
  VkDisplaySurfaceCreateInfoKHR dsi{VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR};
  dsi.displayMode = mode.displayMode;
  dsi.planeIndex = plane;
  dsi.planeStackIndex = planes[plane].currentStackIndex;
  dsi.transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  dsi.alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
  dsi.imageExtent = mode.parameters.visibleRegion;
  const VkResult sr = createSurface(inst, &dsi, nullptr, &surf);
  if (sr != VK_SUCCESS) { surf = VK_NULL_HANDLE; return giveBack("its surface refused (VkResult " + std::to_string(static_cast<int>(sr)) + ")"); }
  displayExtent = mode.parameters.visibleRegion;
  displayHz = mode.parameters.refreshRate / 1000.0;
  modeHz = displayHz;
  return "";
}

void Presenter::Impl::saveLayout() {
  layoutWas.clear();
  const Window root = DefaultRootWindow(xq);
  fbW = DisplayWidth(xq, DefaultScreen(xq));
  fbH = DisplayHeight(xq, DefaultScreen(xq));
  fbMmW = DisplayWidthMM(xq, DefaultScreen(xq));
  fbMmH = DisplayHeightMM(xq, DefaultScreen(xq));
  primaryWas = XRRGetOutputPrimary(xq, root);
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, root);
  if (!res) return;
  for (int i = 0; i < res->ncrtc; i++) {
    XRRCrtcInfo* ci = XRRGetCrtcInfo(xq, res, res->crtcs[i]);
    if (!ci) continue;
    if (ci->mode) {
      CrtcWas w;
      w.crtc = res->crtcs[i];
      w.mode = ci->mode;
      w.x = ci->x;
      w.y = ci->y;
      w.rot = ci->rotation;
      w.outs.assign(ci->outputs, ci->outputs + ci->noutput);
      layoutWas.push_back(w);
    }
    XRRFreeCrtcInfo(ci);
  }
  XRRFreeScreenResources(res);
}

// ai: the layout as saveLayout found it: the screen's size, then each CRTC that differs, then the primary output
void Presenter::Impl::restoreLayout() {
  if (layoutWas.empty()) return;
  const Window root = DefaultRootWindow(xq);
  XSync(xq, False);
  if (DisplayWidth(xq, DefaultScreen(xq)) != fbW || DisplayHeight(xq, DefaultScreen(xq)) != fbH) XRRSetScreenSize(xq, root, fbW, fbH, fbMmW, fbMmH);
  // ai: just after the display is released the server may refuse its CRTC (NVIDIA 615, a DisplayPort monitor:
  // ai: RRSetConfigFailed on the first try, where an HDMI one took it), and a monitor left without a signal sleeps in
  // ai: seconds and drops off the link until it is switched off and on: each CRTC asked again every 100 ms for 3 s
  for (CrtcWas& w : layoutWas) {
    for (int tries = 0;; tries++) {
      XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, root);
      if (!res) break;
      XRRCrtcInfo* ci = XRRGetCrtcInfo(xq, res, w.crtc);
      const bool same = ci && ci->mode == w.mode && ci->x == w.x && ci->y == w.y && ci->rotation == w.rot &&
                        ci->noutput == static_cast<int>(w.outs.size()) && std::equal(w.outs.begin(), w.outs.end(), ci->outputs);
      if (ci) XRRFreeCrtcInfo(ci);
      const bool done = same || XRRSetCrtcConfig(xq, res, w.crtc, CurrentTime, w.x, w.y, w.mode, w.rot, w.outs.data(), static_cast<int>(w.outs.size())) == RRSetConfigSuccess;
      XRRFreeScreenResources(res);
      if (done) break;
      if (tries == 30) { say("presenter: the desktop's layout not put back for a monitor"); break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  if (primaryWas && XRRGetOutputPrimary(xq, root) != primaryWas) XRRSetOutputPrimary(xq, root, primaryWas);
  XSync(xq, False);
  layoutWas.clear();
}

void Presenter::Impl::grabInput() {
  if (grabbed || nowMs() - grabTried < 250) return;
  grabTried = nowMs();
  Window top = topLevel(xq, xwin);
  XWindowAttributes wa{};
  if (!XGetWindowAttributes(xq, top, &wa) || wa.map_state != IsViewable) top = DefaultRootWindow(xq);
  const int k = XGrabKeyboard(xq, top, False, GrabModeAsync, GrabModeAsync, CurrentTime);
  const int m = XGrabPointer(xq, top, False, ButtonPressMask, GrabModeAsync, GrabModeAsync, None, None, CurrentTime);
  XFlush(xq);
  if (k == GrabSuccess && m == GrabSuccess) { grabbed = true; return; }
  // ai: another client holds one (a menu, the manager's switcher): both let go and asked again in a quarter second
  if (k == GrabSuccess) XUngrabKeyboard(xq, CurrentTime);
  if (m == GrabSuccess) XUngrabPointer(xq, CurrentTime);
  XFlush(xq);
}

void Presenter::Impl::ungrabInput() {
  if (!grabbed) return;
  XUngrabKeyboard(xq, CurrentTime);
  XUngrabPointer(xq, CurrentTime);
  XFlush(xq);
  grabbed = false;
}

bool Presenter::Impl::inputArrived() {
  bool hit = false;
  while (XPending(xq)) {
    XEvent e;
    XNextEvent(xq, &e);
    hit = hit || e.type == KeyPress || e.type == ButtonPress;
  }
  return hit;
}

void Presenter::Impl::armGuard() {
  // ai: xrandr's words for the saved layout: the screen's size, then each output's mode (by id), place and rotation,
  // ai: the primary marked
  std::string cmd = "xrandr --fb " + std::to_string(fbW) + "x" + std::to_string(fbH);
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(xq, DefaultRootWindow(xq));
  if (!res) return;
  for (const CrtcWas& w : layoutWas)
    for (RROutput o : w.outs)
      if (XRROutputInfo* oi = XRRGetOutputInfo(xq, res, o)) {
        char mode[32];
        std::snprintf(mode, sizeof mode, "0x%lx", static_cast<unsigned long>(w.mode));
        const Rotation r = w.rot & 0xf;
        const char* rot = r == RR_Rotate_90 ? "left" : r == RR_Rotate_180 ? "inverted" : r == RR_Rotate_270 ? "right" : "normal";
        cmd += std::string(" --output '") + oi->name + "' --mode " + mode + " --pos " + std::to_string(w.x) + "x" + std::to_string(w.y) +
               " --rotate " + rot + (o == primaryWas ? " --primary" : "");
        XRRFreeOutputInfo(oi);
      }
  XRRFreeScreenResources(res);
  int fds[2];
  if (pipe2(fds, O_CLOEXEC) != 0) return;
  // ai: asked again every 100 ms for 3 s, as restoreLayout does: the server may refuse a CRTC just after the release
  const std::string script = "read x; [ \"$x\" = given-back ] || { command -v xrandr >/dev/null || exit 0; i=0; until " + cmd +
                             "; do i=$((i+1)); [ $i -ge 30 ] && exit 1; sleep 0.1; done; }";
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, fds[0], 0);
  // ai: none of this process's other files in the guard: an X connection held open there would keep the lease alive
  // ai: past a crash
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
  posix_spawn_file_actions_addclosefrom_np(&fa, 3);
#endif
  const char* argv[] = {"sh", "-c", script.c_str(), nullptr};
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, "/bin/sh", &fa, nullptr, const_cast<char* const*>(argv), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(fds[0]);
  if (rc != 0) { close(fds[1]); say("presenter: no guard for the display (the layout stays off if this process dies)"); return; }
  guardFd = fds[1];
}

// ai: the display given back by this process: the guard told so and let go (it exits without a word to xrandr)
void Presenter::Impl::disarmGuard() {
  if (guardFd < 0) return;
  const char word[] = "given-back\n";
  if (write(guardFd, word, sizeof word - 1) < 0) {}
  close(guardFd);
  guardFd = -1;
}
#else
std::string Presenter::Impl::takeDisplay() { return "no direct display on this system"; }
#endif

// ai: The display given back while presenting (a key or a click): the chain and the display's surface let go, the
// ai: display released, the desktop's layout put back, the input let go, and the window's surface made, on which the
// ai: next present makes a chain; the app follows the stats' leaseEnded and puts its window back.
void Presenter::Impl::leaveDisplay(const std::string& why) {
  if (!displayTaken) return;
  waiter.finish();
  drain();
  qIdle();
  for (VkSemaphore a : c.done) D.vkDestroySemaphore(dev, a, nullptr);
  if (c.sc) D.vkDestroySwapchainKHR(dev, c.sc, nullptr);
  c = Chain{};
  if (surf) I.vkDestroySurfaceKHR(inst, surf, nullptr);
  surf = VK_NULL_HANDLE;
  if (releaseDisplay) releaseDisplay(pd, display);
  displayTaken = false;
  display = VK_NULL_HANDLE;
  leaseEnded = true;
#if LIZ_X11 && !defined(_WIN32)
  restoreLayout();
  disarmGuard();
  ungrabInput();
  auto createSurface = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateXlibSurfaceKHR"));
  VkXlibSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  si.dpy = dpy;
  si.window = xwin;
  vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateXlibSurfaceKHR");
#endif
  VkBool32 ok = VK_FALSE;
  I.vkGetPhysicalDeviceSurfaceSupportKHR(pd, qf, surf, &ok);
  if (!ok) throw std::runtime_error(name + " cannot present to this window");
  capsAt = -1e9;
  rateMoved();
  say("presenter: the display " + opt.direct + " given back (" + why + "): the window");
  // ai: the stats made now, so the app sees leaseEnded at its next read rather than a second on
  window((nowMs() - lastWin) / 1000);
  lastWin = nowMs();
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
  // ai: present timing on this surface: absolute targets and the first-pixel stage reported (VK_EXT_present_timing's
  // ai: surface query), present_id2 for the ids its reports carry; else the present count and present_wait
  timing = false;
  if (timingDev) {
    VkPresentTimingSurfaceCapabilitiesEXT ptc{VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
    VkSurfaceCapabilitiesPresentId2KHR idc{VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR};
    idc.pNext = &ptc;
    VkSurfaceCapabilities2KHR c2{VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR};
    c2.pNext = &idc;
    VkPhysicalDeviceSurfaceInfo2KHR si2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR};
    si2.surface = surf;
    if (T.caps2(pd, &si2, &c2) == VK_SUCCESS && ptc.presentTimingSupported && ptc.presentAtAbsoluteTimeSupported && idc.presentId2Supported &&
        (ptc.presentStageQueries & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT)) {
      timing = true;
      stages = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT;
    }
  }
  VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  if (timing) ci.flags = VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT | VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
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
  if (timing) timingSetup(sc);
  if (opt.mark) {
    freeMark();
    const VkDeviceSize slot = static_cast<VkDeviceSize>(MARK_W) * MARK_PX * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = slot * n;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCheck(D.vkCreateBuffer(dev, &bci, nullptr, &markBuf), "vkCreateBuffer");
    VkMemoryRequirements r;
    D.vkGetBufferMemoryRequirements(dev, markBuf, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkCheck(D.vkAllocateMemory(dev, &ai, nullptr, &markMem), "vkAllocateMemory");
    vkCheck(D.vkBindBufferMemory(dev, markBuf, markMem, 0), "vkBindBufferMemory");
    void* p = nullptr;
    vkCheck(D.vkMapMemory(dev, markMem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory");
    markPtr = static_cast<uint8_t*>(p);
    markSlots = n;
  }
}

void Presenter::Impl::freeMark() {
  if (markBuf) D.vkDestroyBuffer(dev, markBuf, nullptr);
  if (markMem) D.vkFreeMemory(dev, markMem, nullptr);
  markBuf = VK_NULL_HANDLE;
  markMem = VK_NULL_HANDLE;
  markPtr = nullptr;
  markSlots = 0;
}

// ai: The swapchain made again (a new size, out of date, suboptimal): the waits stopped first (a wait on a swapchain
// ai: that is gone is undefined), the presents so far counted, a new epoch so no gap is counted across it.
void Presenter::Impl::remake(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext) {
  waiter.finish();
  drain();
  qIdle();
  makeChain(caps, ext);
  epoch++;
  waiter.start(D, dev, presentWait && !timing, c.sc, t0);
  resized = false;
  lastRemake = nowMs();
  say("presenter: the swapchain at " + std::to_string(ext.width) + " x " + std::to_string(ext.height) + ", " +
      (c.format == VK_FORMAT_B8G8R8A8_UNORM ? "BGRA8" : "RGBA8") + " UNORM, " + std::to_string(c.images.size()) + " images, FIFO" +
      (timing ? ", present timing (refresh " + fmt("%.4f", refreshNs / 1e6) + " ms, time domain " + std::to_string(static_cast<int>(domain)) + ")" : ""));
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
  std::vector<Shown> all = waiter.take();
  all.insert(all.end(), timedDone.begin(), timedDone.end());
  timedDone.clear();
  for (const Shown& s : all) {
    win.add(s, refreshMs);
    if (csv)
      std::fprintf(csv, "%llu,%llu,%d,%s,%s,%s,%d,%s\n", static_cast<unsigned long long>(s.id), static_cast<unsigned long long>(s.picture),
                   s.fresh ? 1 : 0, fmt("%.3f", s.acquiredMs).c_str(), fmt("%.3f", s.submittedMs).c_str(), fmt("%.3f", s.shownMs).c_str(), s.queued,
                   fmt("%.3f", s.targetMs).c_str());
  }
}

// ai: A second's window closed: the presents' times counted, the refresh measured, the mode read again (the window may
// ai: have moved to another monitor) and VSync's hold looked at (not in the last window, at a stop), the stats made,
// ai: the counts zeroed.
void Presenter::Impl::window(double secs, bool last) {
  drain();
  // ai: on present timing the display's own refresh; else measured from the presents' gaps
  const double m = timing && refreshNs ? 1e9 / static_cast<double>(refreshNs) : win.measuredHz(modeHz);
  if (m > 0) measured = m;
  queryMode();
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
                   ",\"vsync\":" + (vsync ? "true" : "false") + ",\"modeFrom\":" + fmt("%.3f", held.on ? held.origHz : 0.0) +
                   ",\"timing\":" + (timing ? "true" : "false") + ",\"late\":" + std::to_string(win.late) + ",\"slips\":" + std::to_string(win.slips) +
                   ",\"errMs\":" + fmt("%.3f", win.errMaxMs) + ",\"direct\":\"" + (displayTaken ? esc(opt.direct) : std::string()) + "\",\"directName\":\"" + (displayTaken ? esc(directName) : std::string()) + "\",\"directPx\":\"" +
                   (displayTaken ? std::to_string(displayExtent.width) + "x" + std::to_string(displayExtent.height) : std::string()) + "\",\"leaseEnded\":" + (leaseEnded ? "true" : "false") +
                   ",\"device\":\"" + esc(name) + "\",\"error\":\"";
  {
    std::lock_guard<std::mutex> l(statsMu);
    statsJson = st + esc(error) + "\",\"secs\":" + fmt("%.3f", secs) + "}";
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
  // ai: on the display's grid (present timing, its first time back): one present a picture, for its slot's refresh,
  // ai: which the display holds until the next one's; a slot already near or past (the loop stalled) gives way to
  // ai: the first one a refresh and a half on, those between counted missed. Before the grid (or without timing):
  // ai: a present a refresh, a picture where one is due, as before.
  pollTiming();
  const bool onGrid = live && timing && anchored && refreshNs;
  double slot = i;
  uint64_t targetNs = 0;
  if (onGrid) {
    const double ns = nowSlot();
    if (due < ns + 1.5) {
      const double to = std::ceil(ns + 1.5);
      if (cur >= 0 && ns > -1e17) win.missed += static_cast<uint64_t>(std::max(0.0, std::round(to - due)));
      due = to;
    }
    slot = std::round(due);
    targetNs = gridNs(slot);
  }
  if (live && (onGrid || i + 0.5 >= due)) {
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
        due = onGrid ? due + per : std::max(due + per, i + per);
      }
    }
    if (!fresh && cur >= 0) win.behind++;
    // ai: on the grid a slot with nothing painted keeps the picture on screen for it and the next is a period on
    if (onGrid && !fresh) due += period();
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
    // ai: stretched (linear) as the page's fit=stretch
    const Slot& s = slots[cur];
    const double e = 2 * edge.load() * s.h;
    const double fit = std::min(static_cast<double>(c.extent.width) / (s.w + e), static_cast<double>(c.extent.height) / (s.h + e));
    const double share = std::clamp(size.load(), 0.25, 1.0);
    const double scale = fit * share;
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
      D.vkCmdBlitImage(cb, s.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    }
    slots[cur].usedBy = frame;
  }
  if (markPtr && static_cast<size_t>(fl) < markSlots && c.extent.width >= static_cast<uint32_t>(MARK_W) && c.extent.height >= static_cast<uint32_t>(MARK_PX)) {
    const size_t slot = static_cast<size_t>(MARK_W) * MARK_PX * 4;
    uint8_t* m = markPtr + static_cast<size_t>(fl) * slot;
    const uint32_t v = static_cast<uint32_t>(picture) & ((1u << MARK_BITS) - 1);
    for (int b = 0; b < 2 * MARK_BITS; b++) {
      const uint32_t bit = b < MARK_BITS ? (v >> (MARK_BITS - 1 - b)) & 1u : ((v >> (2 * MARK_BITS - 1 - b)) & 1u) ^ 1u;
      const uint8_t g = bit ? 0 : 255;
      for (int yy = 0; yy < MARK_PX; yy++)
        for (int xx = 0; xx < MARK_PX; xx++) {
          uint8_t* q = m + (static_cast<size_t>(yy) * MARK_W + static_cast<size_t>(b) * MARK_PX + xx) * 4;
          q[0] = q[1] = q[2] = g;
          q[3] = 255;
        }
    }
    barrier(D, cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy rg{};
    rg.bufferOffset = static_cast<VkDeviceSize>(fl) * slot;
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rg.imageExtent = {static_cast<uint32_t>(MARK_W), static_cast<uint32_t>(MARK_PX), 1};
    D.vkCmdCopyBufferToImage(cb, markBuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);
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
  // ai: present timing: the id its report carries (present_id2), the target refresh (absolute, the nearest refresh
  // ai: to it) and the first-pixel stage asked for; a full results queue presents once more without asking
  VkPresentTimingInfoEXT pti{VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
  pti.flags = targetNs ? static_cast<VkPresentTimingInfoFlagsEXT>(VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT) : 0u;
  pti.targetTime = targetNs;
  pti.timeDomainId = domainId;
  pti.presentStageQueries = stages;
  pti.targetTimeDomainPresentStage =
      domain == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ? static_cast<VkPresentStageFlagsEXT>(VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) : 0u;
  VkPresentTimingsInfoEXT ptis{VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
  ptis.swapchainCount = 1;
  ptis.pTimingInfos = &pti;
  VkPresentId2KHR pid2{VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
  pid2.swapchainCount = 1;
  pid2.pPresentIds = &id;
  pid2.pNext = &ptis;
  if (timing) pi.pNext = &pid2;
  else if (presentWait) pi.pNext = &pidInfo;
  r = qPresent(pi);
  if (timing && r == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) {
    pid2.pNext = nullptr;
    r = qPresent(pi);
  }
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
    if (timing) {
      s.queued = static_cast<int>(id - lastDone);
      sent[id] = Sent{s, slot, targetNs};
    } else {
      s.queued = presentWait ? static_cast<int>(id - waiter.shownId.load()) : -1;
      waiter.push(s);
    }
  }
  if (live) presents++;
  winPresents++;
  frame++;
}

void Presenter::Impl::run() {
  try {
    while (!quit) {
      if (vsyncDirty.exchange(false)) holdMode();
#if LIZ_X11 && !defined(_WIN32)
      if (displayTaken && grabToLeave) {
        grabInput();
        if (inputArrived()) leaveDisplay("a key or a click");
      }
#endif
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
        // ai: resumed: a new picture due now (on the grid, the first slot a refresh and a half on), no gap counted
        // ai: across the pause
        due = timing && anchored ? std::ceil(nowSlot() + 1.5) : static_cast<double>(presents);
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
    // ai: the last window, however short (its secs say)
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
    freeMark();
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
  // ai: the display given back to the desktop (the X server's lease ends; it also ends with this process)
  if (displayTaken && releaseDisplay) {
    const VkResult rr = releaseDisplay(pd, display);
    say("presenter: the display " + opt.direct + (rr == VK_SUCCESS ? " given back" : " not given back (VkResult " + std::to_string(static_cast<int>(rr)) + ")"));
  }
  displayTaken = false;
  display = VK_NULL_HANDLE;
#if LIZ_X11 && !defined(_WIN32)
  if (xq) restoreLayout();
  disarmGuard();
  if (xq) ungrabInput();
#endif
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
    // ai: the display's own surface where a host asks for it (PresentOptions::direct): VK_KHR_display, and on X11 an
    // ai: output taken from the X server (VK_EXT_acquire_xlib_display, given back by VK_EXT_direct_mode_display);
    // ai: present_timing's surface query (VK_KHR_get_surface_capabilities2). Each enabled where offered.
    x.instanceExts.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
    x.instanceExts.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
    x.instanceExts.push_back(VK_EXT_DIRECT_MODE_DISPLAY_EXTENSION_NAME);
#if LIZ_X11 && !defined(_WIN32)
    x.instanceExts.push_back(VK_EXT_ACQUIRE_XLIB_DISPLAY_EXTENSION_NAME);
#endif
    // ai: present_id2, present_timing and calibrated timestamps (2026-10-31): each present's target refresh and the
    // ai: time its first pixel left for the display (wg enables them with their features, or none of them)
    x.deviceExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME, VK_KHR_PRESENT_WAIT_EXTENSION_NAME,
                    VK_KHR_PRESENT_ID_2_EXTENSION_NAME, VK_EXT_PRESENT_TIMING_EXTENSION_NAME, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME};
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
