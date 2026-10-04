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
#include <future>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

// ai: Vulkan loaded at run time through volk, as the engine's wg (liblizard/core/third_party/volk): the loader's entry
// ai: points from it, everything else from this presenter's own instance and device (the tables below)
#include "volk.h"

#include "sender.h"

#if defined(_WIN32)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif LIZ_X11
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include <vulkan/vulkan_xlib.h>
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

std::string lower(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

// ai: The functions this presenter calls, fetched from its own instance and device. volk's globals are not used past
// ai: the loader's entry points: wg (the GPU painter) loads them from its instance, which has no surface extension, so
// ai: a surface function from there would be null.
#define LIZ_INST_FNS(X)                                                                                                \
  X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                                  \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties)                                   \
  X(vkGetPhysicalDeviceFormatProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice)                     \
  X(vkGetDeviceProcAddr) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR)                                \
  X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR)
#define LIZ_DEV_FNS(X)                                                                                                 \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR)          \
  X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkCreateSemaphore)      \
  X(vkDestroySemaphore) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateCommandPool)  \
  X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer) X(vkBeginCommandBuffer)                  \
  X(vkEndCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) X(vkCmdClearColorImage) X(vkCmdBlitImage)    \
  X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)            \
  X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)              \
  X(vkBindImageMemory)
#define LIZ_DECL(f) PFN_##f f = nullptr;
struct InstFns {
  LIZ_INST_FNS(LIZ_DECL)
  PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;   // ai: Vulkan 1.1; present_wait's check
};
struct DevFns {
  LIZ_DEV_FNS(LIZ_DECL)
  PFN_vkWaitForPresentKHR vkWaitForPresentKHR = nullptr;   // ai: where the device has present_wait
};
#undef LIZ_DECL

constexpr int FLIGHT = 2;

// ai: The swapchain: FIFO (each present one refresh, the queue's back-pressure the loop's pace), at least 3 images, an
// ai: 8-bit UNORM format so a grey level reaches the screen as painted, written by transfers (a clear and a blit).
struct Chain {
  VkSwapchainKHR sc = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  std::vector<VkImage> images;
  std::vector<VkSemaphore> done;   // ai: per image: its commands finished, which its present waits on
};

// ai: A picture's home on the device: the staging buffer the sender writes (mapped), and the image it is copied to once
// ai: and blitted from at every refresh it is held. Three, so the one being filled is never one in flight; each remade
// ai: at the frame's size when a re-pick changes it.
struct Slot {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory bufMem = VK_NULL_HANDLE;
  uint8_t* map = nullptr;
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory imgMem = VK_NULL_HANDLE;
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
};

// ai: The second thread that waits for each present to reach the screen (vkWaitForPresentKHR), in id order. Without
// ai: present_wait a present's time is its acquire's.
struct Waiter {
  const DevFns* D = nullptr;
  VkDevice dev = VK_NULL_HANDLE;
  bool wait = false;
  VkSwapchainKHR sc = VK_NULL_HANDLE;
  double t0 = 0;
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
      if (r == VK_SUCCESS) s.shownMs = nowMs() - t0;
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
  void add(const Shown& s, double refreshMs) {
    presents++;
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

#if LIZ_X11 && !defined(_WIN32)
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

struct Presenter::Impl {
  PresentTarget target;
  Sender* sender = nullptr;
  std::mutex* senderLock = nullptr;
  PresentOptions opt;
  std::atomic<int> fps{60};
  std::atomic<double> size{1.0};
  std::atomic<bool> whole{false}, paused{false}, quit{false};
  std::atomic<int> bypass{-1};   // ai: a bypass asked and not yet set: 1 on, 0 off
  std::mutex wakeMu;
  std::condition_variable wake;
  std::thread thread;

#if LIZ_X11 && !defined(_WIN32)
  // ai: two connections of the present thread's own: dpy the surface's (the driver's alone; AWT never calls
  // ai: XInitThreads, so no Xlib call of ours shares it with a driver thread's xcb), xq for the window queries (the size,
  // ai: the mode, the bypass)
  Display* dpy = nullptr;
  Display* xq = nullptr;
  Window xwin = 0;
  int randr = -1;   // ai: XRandR on xq: -1 not asked yet, 0 no, 1 yes
#endif
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
  Chain c;
  std::vector<Slot> slots = std::vector<Slot>(3);
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cbs[FLIGHT]{};
  VkFence fences[FLIGHT]{};
  VkSemaphore acquired[FLIGHT]{};
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
  void makeChain(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext);
  void remake(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext);
  void makeSlot(Slot& s, int w, int h);
  void freeSlot(Slot& s);
  VkExtent2D windowSize();
  void queryMode();
  void applyBypass();
  void drain();
  void window(double secs);
};

void Presenter::Impl::setup() {
  static const VkResult loader = volkInitialize();
  if (loader != VK_SUCCESS) throw std::runtime_error("no Vulkan loader on this machine");
  uint32_t version = VK_API_VERSION_1_0;
  auto enumVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (enumVersion) enumVersion(&version);
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "lizard_presenter";
  app.apiVersion = std::min(version, static_cast<uint32_t>(VK_API_VERSION_1_2));
#if defined(_WIN32)
  const char* ie[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
#elif LIZ_X11
  const char* ie[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
#else
  const char* ie[] = {VK_KHR_SURFACE_EXTENSION_NAME};
  throw std::runtime_error("no window surface for this platform yet");
#endif
  VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ii.pApplicationInfo = &app;
  ii.enabledExtensionCount = static_cast<uint32_t>(sizeof ie / sizeof ie[0]);
  ii.ppEnabledExtensionNames = ie;
  vkCheck(vkCreateInstance(&ii, nullptr, &inst), "vkCreateInstance");
#define LIZ_LOADI(f)                                                                     \
  I.f = reinterpret_cast<PFN_##f>(vkGetInstanceProcAddr(inst, #f));                      \
  if (!I.f) throw std::runtime_error("the Vulkan instance has no " #f);
  LIZ_INST_FNS(LIZ_LOADI)
#undef LIZ_LOADI
  I.vkGetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2"));

#if defined(_WIN32)
  if (!target.hwnd) throw std::runtime_error("no window to present to");
  auto createSurface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(vkGetInstanceProcAddr(inst, "vkCreateWin32SurfaceKHR"));
  if (!createSurface) throw std::runtime_error("no VK_KHR_win32_surface");
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
  if (!createSurface) throw std::runtime_error("no VK_KHR_xlib_surface");
  VkXlibSurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
  si.dpy = dpy;
  si.window = xwin;
  vkCheck(createSurface(inst, &si, nullptr, &surf), "vkCreateXlibSurfaceKHR");
#endif

  // ai: The device that presents to the window (a graphics queue that can present: a blit needs graphics): the one
  // ai: named like opt.device (or LIZ_VK_DEVICE), else the discrete one first; present_id and present_wait where it has both.
  std::string want = opt.device;
  if (want.empty() && std::getenv("LIZ_VK_DEVICE")) want = std::getenv("LIZ_VK_DEVICE");
  uint32_t n = 0;
  I.vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n);
  I.vkEnumeratePhysicalDevices(inst, &n, pds.data());
  int best = -1;
  std::string names;
  uint32_t api = 0;
  for (VkPhysicalDevice p : pds) {
    VkPhysicalDeviceProperties pp;
    I.vkGetPhysicalDeviceProperties(p, &pp);
    names += (names.empty() ? "" : ", ") + std::string(pp.deviceName);
    if (!want.empty() && lower(pp.deviceName).find(lower(want)) == std::string::npos) continue;
    uint32_t qn = 0;
    I.vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    I.vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, qs.data());
    for (uint32_t i = 0; i < qn; i++) {
      VkBool32 ok = VK_FALSE;
      I.vkGetPhysicalDeviceSurfaceSupportKHR(p, i, surf, &ok);
      if (!(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !ok) continue;
      const int score = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
      if (score > best) { best = score; pd = p; qf = i; name = pp.deviceName; api = pp.apiVersion; }
      break;
    }
  }
  if (!pd) throw std::runtime_error(want.empty() ? "no Vulkan device presents to this window (there are: " + names + ")"
                                                 : "no Vulkan device named like \"" + want + "\" presents to this window (there are: " + names + ")");
  uint32_t en = 0;
  I.vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
  std::vector<VkExtensionProperties> ex(en);
  I.vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, ex.data());
  auto has = [&](const char* e) {
    for (const auto& x : ex)
      if (!std::strcmp(x.extensionName, e)) return true;
    return false;
  };
  if (!has(VK_KHR_SWAPCHAIN_EXTENSION_NAME)) throw std::runtime_error(name + " has no swapchain");
  if (I.vkGetPhysicalDeviceFeatures2 && app.apiVersion >= VK_API_VERSION_1_1 && api >= VK_API_VERSION_1_1 &&
      has(VK_KHR_PRESENT_ID_EXTENSION_NAME) && has(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
    VkPhysicalDevicePresentWaitFeaturesKHR pw{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
    VkPhysicalDevicePresentIdFeaturesKHR pi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    pi.pNext = &pw;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &pi;
    I.vkGetPhysicalDeviceFeatures2(pd, &f2);
    presentWait = pi.presentId && pw.presentWait;
  }
  std::vector<const char*> de{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkPhysicalDevicePresentWaitFeaturesKHR pwOn{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
  VkPhysicalDevicePresentIdFeaturesKHR piOn{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
  pwOn.presentWait = VK_TRUE;
  piOn.presentId = VK_TRUE;
  piOn.pNext = &pwOn;
  if (presentWait) { de.push_back(VK_KHR_PRESENT_ID_EXTENSION_NAME); de.push_back(VK_KHR_PRESENT_WAIT_EXTENSION_NAME); }
  const float prio = 1;
  VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qi.queueFamilyIndex = qf;
  qi.queueCount = 1;
  qi.pQueuePriorities = &prio;
  VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  di.pNext = presentWait ? &piOn : nullptr;
  di.queueCreateInfoCount = 1;
  di.pQueueCreateInfos = &qi;
  di.enabledExtensionCount = static_cast<uint32_t>(de.size());
  di.ppEnabledExtensionNames = de.data();
  vkCheck(I.vkCreateDevice(pd, &di, nullptr, &dev), "vkCreateDevice");
#define LIZ_LOADD(f)                                                                     \
  D.f = reinterpret_cast<PFN_##f>(I.vkGetDeviceProcAddr(dev, #f));                       \
  if (!D.f) throw std::runtime_error("the Vulkan device has no " #f);
  LIZ_DEV_FNS(LIZ_LOADD)
#undef LIZ_LOADD
  if (presentWait) D.vkWaitForPresentKHR = reinterpret_cast<PFN_vkWaitForPresentKHR>(I.vkGetDeviceProcAddr(dev, "vkWaitForPresentKHR"));
  if (!D.vkWaitForPresentKHR) presentWait = false;
  D.vkGetDeviceQueue(dev, qf, 0, &q);
  I.vkGetPhysicalDeviceMemoryProperties(pd, &mem);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = qf;
  vkCheck(D.vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool");
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = FLIGHT;
  vkCheck(D.vkAllocateCommandBuffers(dev, &cai, cbs), "vkAllocateCommandBuffers");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (int i = 0; i < FLIGHT; i++) {
    vkCheck(D.vkCreateFence(dev, &fci, nullptr, &fences[i]), "vkCreateFence");
    vkCheck(D.vkCreateSemaphore(dev, &sci, nullptr, &acquired[i]), "vkCreateSemaphore");
  }
  if (!opt.csv.empty()) {
    csv = std::fopen(opt.csv.c_str(), "w");
    if (!csv) throw std::runtime_error(opt.csv + " could not be written");
    std::fprintf(csv, "id,picture,fresh,acquired_ms,submitted_ms,shown_ms\n");
  }
  queryMode();
  t0 = lastWin = nowMs();
  window(0);
  say("presenter: " + name + (presentWait ? ", times from present_wait" : ", times from acquire (no present_wait)") +
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
#endif
  return {0, 0};
}

// ai: X11: the mode's refresh of the monitor (a CRTC) holding the window's centre; Windows: the display settings of the
// ai: monitor nearest the window (whole Hz: 59 for 59.94, the measurement corrects it); elsewhere none
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
#endif
}

// ai: X11: asks the window manager to take the window out of the compositor (_NET_WM_BYPASS_COMPOSITOR, 1 on, 0 the
// ai: manager's choice), so the code's presents reach the screen as the swapchain makes them, not as the compositor
// ai: redraws (2026-10-03: Chrome under Cinnamon left 8.6% of pictures one refresh or less)
void Presenter::Impl::applyBypass() {
  const int b = bypass.exchange(-1);
  if (b < 0) return;
#if LIZ_X11 && !defined(_WIN32)
  const Window top = topLevel(xq, xwin);
  const unsigned long v = b ? 1 : 0;
  XChangeProperty(xq, top, XInternAtom(xq, "_NET_WM_BYPASS_COMPOSITOR", False), XA_CARDINAL, 32, PropModeReplace,
                  reinterpret_cast<const unsigned char*>(&v), 1);
  XFlush(xq);
  char t[32];
  std::snprintf(t, sizeof t, "0x%lx", static_cast<unsigned long>(top));
  say("presenter: compositor bypass " + std::string(b ? "on" : "off") + " on window " + t);
#endif
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
  uint32_t count = std::max(3u, caps.minImageCount);
  if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
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
}

// ai: The swapchain made again (a new size, out of date, suboptimal): the waits stopped first (a wait on a swapchain
// ai: that is gone is undefined), the presents so far counted, a new epoch so no gap is counted across it.
void Presenter::Impl::remake(const VkSurfaceCapabilitiesKHR& caps, VkExtent2D ext) {
  waiter.finish();
  drain();
  D.vkDeviceWaitIdle(dev);
  makeChain(caps, ext);
  epoch++;
  waiter.start(D, dev, presentWait, c.sc, t0);
  resized = false;
  lastRemake = nowMs();
  say("presenter: the swapchain at " + std::to_string(ext.width) + " x " + std::to_string(ext.height) + ", " +
      (c.format == VK_FORMAT_B8G8R8A8_UNORM ? "BGRA8" : "RGBA8") + " UNORM, " + std::to_string(c.images.size()) + " images, FIFO");
}

void Presenter::Impl::freeSlot(Slot& s) {
  if (s.img) D.vkDestroyImage(dev, s.img, nullptr);
  if (s.imgMem) D.vkFreeMemory(dev, s.imgMem, nullptr);
  if (s.buf) D.vkDestroyBuffer(dev, s.buf, nullptr);
  if (s.bufMem) D.vkFreeMemory(dev, s.bufMem, nullptr);
  s = Slot{};
}

void Presenter::Impl::makeSlot(Slot& s, int w, int h) {
  freeSlot(s);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = static_cast<VkDeviceSize>(w) * h * 4;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCheck(D.vkCreateBuffer(dev, &bi, nullptr, &s.buf), "vkCreateBuffer");
  VkMemoryRequirements r;
  D.vkGetBufferMemoryRequirements(dev, s.buf, &r);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = r.size;
  ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkCheck(D.vkAllocateMemory(dev, &ai, nullptr, &s.bufMem), "vkAllocateMemory");
  vkCheck(D.vkBindBufferMemory(dev, s.buf, s.bufMem, 0), "vkBindBufferMemory");
  void* m = nullptr;
  vkCheck(D.vkMapMemory(dev, s.bufMem, 0, VK_WHOLE_SIZE, 0, &m), "vkMapMemory");
  s.map = static_cast<uint8_t*>(m);
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = VK_FORMAT_R8G8B8A8_UNORM;
  ii.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  vkCheck(D.vkCreateImage(dev, &ii, nullptr, &s.img), "vkCreateImage");
  D.vkGetImageMemoryRequirements(dev, s.img, &r);
  ai.allocationSize = r.size;
  ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkCheck(D.vkAllocateMemory(dev, &ai, nullptr, &s.imgMem), "vkAllocateMemory");
  vkCheck(D.vkBindImageMemory(dev, s.img, s.imgMem, 0), "vkBindImageMemory");
  s.w = w;
  s.h = h;
}

void Presenter::Impl::drain() {
  const double refreshMs = 1000.0 / refreshHz();
  for (const Shown& s : waiter.take()) {
    win.add(s, refreshMs);
    all.add(s, refreshMs);
    if (csv)
      std::fprintf(csv, "%llu,%llu,%d,%s,%s,%s\n", static_cast<unsigned long long>(s.id), static_cast<unsigned long long>(s.picture),
                   s.fresh ? 1 : 0, fmt("%.3f", s.acquiredMs).c_str(), fmt("%.3f", s.submittedMs).c_str(), fmt("%.3f", s.shownMs).c_str());
  }
}

// ai: A second's window closed: the presents' times counted, the refresh measured, the mode read again (the window may
// ai: have moved to another monitor), the stats made, the counts zeroed.
void Presenter::Impl::window(double secs) {
  drain();
  const double m = win.measuredHz(modeHz);
  if (m > 0) measured = m;
  queryMode();
  const Slot* s = cur >= 0 ? &slots[cur] : nullptr;
  const double per = secs > 0 ? secs : 1;
  std::string st = "{\"shownFps\":" + fmt("%.1f", secs > 0 ? win.pictures / per : 0) +
                   ",\"presentsPerSec\":" + fmt("%.1f", secs > 0 ? winPresents / per : 0) + ",\"hz\":" + fmt("%.3f", refreshHz()) +
                   ",\"modeHz\":" + fmt("%.3f", modeHz) + ",\"held\":" + win.heldJson() + ",\"missed\":" + std::to_string(win.missed) +
                   ",\"behind\":" + std::to_string(win.behind) + ",\"paused\":" + (paused ? "true" : "false") + ",\"surface\":\"" +
                   std::to_string(c.extent.width) + "x" + std::to_string(c.extent.height) + "\",\"frame\":\"" +
                   std::to_string(s ? s->w : 0) + "x" + std::to_string(s ? s->h : 0) + "\",\"presentWait\":" + (presentWait ? "true" : "false") +
                   ",\"device\":\"" + esc(name) + "\",\"error\":\"";
  const double total = (nowMs() - t0) / 1000;
  std::string tt = "{\"secs\":" + fmt("%.3f", total) + ",\"pictures\":" + std::to_string(all.pictures) + ",\"presents\":" +
                   std::to_string(all.presents) + ",\"held\":" + all.heldJson() + ",\"missed\":" + std::to_string(all.missed) +
                   ",\"behind\":" + std::to_string(all.behind) + ",\"unknown\":" + std::to_string(all.unknown) + ",\"hz\":" +
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
  VkSurfaceCapabilitiesKHR caps;
  vkCheck(I.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surf, &caps), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
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

  const int fl = static_cast<int>(frame % FLIGHT);
  D.vkWaitForFences(dev, 1, &fences[fl], VK_TRUE, UINT64_MAX);
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
  if (measured <= 0 && modeHz <= 0 && firstAcq.size() < 16) {
    firstAcq.push_back(acq);
    if (firstAcq.size() >= 6) {
      std::vector<double> g;
      for (size_t k = 1; k < firstAcq.size(); k++) g.push_back(firstAcq[k] - firstAcq[k - 1]);
      std::nth_element(g.begin(), g.begin() + static_cast<long>(g.size() / 2), g.end());
      if (g[g.size() / 2] > 0.5) early = 1000.0 / g[g.size() / 2];
    }
  }

  // ai: a new picture when one is due and painted; a due one not yet painted keeps the last up (behind)
  bool fresh = false;
  const double i = static_cast<double>(presents);
  if (live && i + 0.5 >= due) {
    const int p = (cur + 1) % static_cast<int>(slots.size());
    if (slots[p].usedBy == frame - 1) D.vkWaitForFences(dev, 1, &fences[(frame - 1) % FLIGHT], VK_TRUE, UINT64_MAX);
    if (lockFor(*senderLock, 2.0)) {
      std::lock_guard<std::mutex> held(*senderLock, std::adopt_lock);
      if (sender->ready()) {
        const int fw = sender->width(), fh = sender->side();
        if (fw > 0 && fh > 0) {
          if (slots[p].w != fw || slots[p].h != fh) makeSlot(slots[p], fw, fh);
          if (sender->take(slots[p].map, fw * 4)) {
            if (cur >= 0) picture++;
            cur = p;
            fresh = true;
            const double per = period();
            due = i - due > per ? i + per : due + per;
          }
        }
      }
    }
    if (!fresh && cur >= 0) { win.behind++; all.behind++; }
  }

  VkCommandBuffer cb = cbs[fl];
  D.vkResetCommandBuffer(cb, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  D.vkBeginCommandBuffer(cb, &bi);
  if (fresh) {
    Slot& s = slots[cur];
    barrier(D, cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {static_cast<uint32_t>(s.w), static_cast<uint32_t>(s.h), 1};
    D.vkCmdCopyBufferToImage(cb, s.buf, s.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier(D, cb, s.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  }
  const VkImage img = c.images[idx];
  barrier(D, cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  const VkClearColorValue white{{1.f, 1.f, 1.f, 1.f}};
  const VkImageSubresourceRange whole1{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  D.vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &whole1);
  if (cur >= 0) {
    // ai: where the picture goes: centred, the largest that fits times the size share, stretched (linear) as the page's
    // ai: fit=stretch; or whole device pixels a sample (nearest) where at least one fits
    const Slot& s = slots[cur];
    const double fit = std::min(static_cast<double>(c.extent.width) / s.w, static_cast<double>(c.extent.height) / s.h);
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
  vkCheck(D.vkQueueSubmit(q, 1, &si, fences[fl]), "vkQueueSubmit");
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
  r = D.vkQueuePresentKHR(q, &pi);
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
    waiter.push(s);
  }
  if (live) presents++;
  winPresents++;
  frame++;
}

void Presenter::Impl::run() {
  try {
    while (!quit) {
      applyBypass();
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
  // ai: a device whose functions did not all load made nothing else (they load before anything is made)
  if (dev) {
    if (D.vkDeviceWaitIdle) D.vkDeviceWaitIdle(dev);
    // ai: the last window, however short (its secs say), and the run's totals with every present counted
    if (t0 > 0) window((nowMs() - lastWin) / 1000);
    for (int i = 0; i < FLIGHT; i++) {
      if (fences[i]) D.vkDestroyFence(dev, fences[i], nullptr);
      if (acquired[i]) D.vkDestroySemaphore(dev, acquired[i], nullptr);
    }
    if (pool) D.vkDestroyCommandPool(dev, pool, nullptr);
    for (auto& s : slots) freeSlot(s);
    for (VkSemaphore s : c.done) D.vkDestroySemaphore(dev, s, nullptr);
    if (c.sc) D.vkDestroySwapchainKHR(dev, c.sc, nullptr);
    if (D.vkDestroyDevice) D.vkDestroyDevice(dev, nullptr);
    dev = VK_NULL_HANDLE;
  }
  if (surf && I.vkDestroySurfaceKHR) I.vkDestroySurfaceKHR(inst, surf, nullptr);
  if (inst && I.vkDestroyInstance) I.vkDestroyInstance(inst, nullptr);
  surf = VK_NULL_HANDLE;
  inst = VK_NULL_HANDLE;
#if LIZ_X11 && !defined(_WIN32)
  if (xq) XCloseDisplay(xq);
  if (dpy) XCloseDisplay(dpy);
  xq = dpy = nullptr;
#endif
  if (csv) std::fclose(csv);
  csv = nullptr;
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
  p->whole = opt.whole;
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

void Presenter::setFps(int fps) { p_->fps = std::clamp(fps, 1, 240); }
void Presenter::setSize(double size) { p_->size = std::clamp(size, 0.25, 1.0); }
void Presenter::setWhole(bool on) { p_->whole = on; }
void Presenter::setPaused(bool on) {
  {
    std::lock_guard<std::mutex> l(p_->wakeMu);
    p_->paused = on;
  }
  p_->wake.notify_all();
}
void Presenter::setBypassCompositor(bool on) { p_->bypass = on ? 1 : 0; }

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
