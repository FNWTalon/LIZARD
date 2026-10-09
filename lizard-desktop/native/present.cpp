// ai: A test sender for the desktop (2026-10-03, to test whether the browser was to blame): the web sender's frames
// ai: with no browser.
// ai: liblizard/core's Sender paints them (the GPU or the C, as the page's encoder switch; a file or the test stream), and
// ai: they go on one monitor full screen through the presenter (presenter.h, 2026-10-04: this program's FIFO loop made
// ai: the desktop app's, this program now a thin main over it): each picture held a whole number of refreshes (the
// ai: refresh over the asked rate, snapped to a whole number within 1%), on white, stretched into the room as the page's
// ai: fit=stretch (linear) or at whole pixels (--fit whole, nearest). The window asks X11's window manager for full
// ai: screen and _NET_WM_BYPASS_COMPOSITOR. No UI: the arguments below; Escape or q (or Ctrl+C in the terminal) ends it
// ai: with a summary.
// ai: Where the device has VK_KHR_present_wait, each present's time on screen is waited for on a second thread, so the
// ai: line each second says how many refreshes every picture actually held and how many refreshes the loop missed;
// ai: elsewhere the times are when each acquire returned, which FIFO paces at the refresh. The phone's rx lines are the
// ai: A/B's other half.
// ai:   lizard_present [--output DP-0] [--subch 416] [--codes 2] [--fps 60] [--ring 128] [--painter auto|gpu|cpu]
// ai:                  [--threads 8] [--n N] [--file PATH] [--fit stretch|whole] [--secs S] [--log CSV] [--assets DIR]
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "presenter.h"
#include "sender.h"

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xrandr.h>
#include <X11/keysym.h>

#ifndef LIZ_ASSETS
#define LIZ_ASSETS "liblizard/out"
#endif

namespace {

std::atomic<bool> gQuit{false};

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Args {
  std::string output, file, fit = "stretch", painter = "auto", log, assets = LIZ_ASSETS;
  int subch = 416, codes = 2, fps = 60, ring = 128, threads = 8, n = 0;
  double secs = 0;
};

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr,
               "%s\nusage: lizard_present [--output NAME] [--subch K] [--codes 1|2] [--fps F] [--ring 32|64|96|128]\n"
               "  [--painter auto|gpu|cpu] [--threads T] [--n N] [--file PATH] [--fit stretch|whole] [--secs S] [--log CSV]\n"
               "  [--assets DIR]\n",
               why.c_str());
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    if (k == "-h" || k == "--help") usage("");
    if (i + 1 >= argc) usage("no value for " + k);
    const std::string v = argv[++i];
    try {
      if (k == "--output") a.output = v;
      else if (k == "--file") a.file = v;
      else if (k == "--fit") a.fit = v;
      else if (k == "--painter") a.painter = v;
      else if (k == "--log") a.log = v;
      else if (k == "--assets") a.assets = v;
      else if (k == "--subch") a.subch = std::stoi(v);
      else if (k == "--codes") a.codes = std::stoi(v);
      else if (k == "--fps") a.fps = std::stoi(v);
      else if (k == "--ring") a.ring = std::stoi(v);
      else if (k == "--threads") a.threads = std::stoi(v);
      else if (k == "--n") a.n = std::stoi(v);
      else if (k == "--secs") a.secs = std::stod(v);
      else usage("unknown " + k);
    } catch (const std::logic_error&) { usage("not a number: " + k + " " + v); }
  }
  if (a.subch < 8 || a.subch > 1024 || a.subch % 8) usage("--subch is 8 to 1024 in steps of 8 (LIZARD-k)");
  if (a.codes != 1 && a.codes != 2) usage("--codes is 1 or 2");
  if (a.fps < 1 || a.fps > 240) usage("--fps is 1 to 240");
  if (a.ring != 32 && a.ring != 64 && a.ring != 96 && a.ring != 128) usage("--ring is 32, 64, 96 or 128");
  if (a.fit != "stretch" && a.fit != "whole") usage("--fit is stretch or whole");
  if (a.painter != "auto" && a.painter != "gpu" && a.painter != "cpu") usage("--painter is auto, gpu or cpu");
  return a;
}

// ai: the picture for a count of sub-channels: desktop Pick.kt nFor (liblizard/sim/lizard_pick.mjs), the first size with n >= 3 R
int nFor(int subch) {
  const double r = std::sqrt(2 * 320.0 * subch / 3.14159265358979323846);
  for (int p : {256, 384, 512, 768, 1024, 1536})
    if (p >= 3 * r) return p;
  return 1536;
}

// ai: a stats JSON's held histogram as text: "1: 58, 2: 1" (refreshes: pictures, in the refreshes' order)
std::string heldText(const nlohmann::json& held) {
  std::map<long, long long> byRefreshes;
  for (auto it = held.begin(); it != held.end(); ++it) byRefreshes[std::stol(it.key())] = it.value().get<long long>();
  std::string o;
  for (const auto& [k, n] : byRefreshes) o += (o.empty() ? "" : ", ") + std::to_string(k) + ": " + std::to_string(n);
  return o.empty() ? "none yet" : o;
}

// ai: the file to send, mapped for as long as the sender lives (the Sender reads it in place)
struct Mapped {
  const uint8_t* data = nullptr;
  size_t len = 0;
  explicit Mapped(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    struct stat st{};
    if (fd < 0 || fstat(fd, &st)) { if (fd >= 0) close(fd); throw std::runtime_error(path + " could not be read"); }
    len = static_cast<size_t>(st.st_size);
    if (len) {
      void* m = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
      if (m == MAP_FAILED) { close(fd); throw std::runtime_error(path + " could not be mapped"); }
      data = static_cast<const uint8_t*>(m);
    }
    close(fd);
  }
  ~Mapped() { if (data) munmap(const_cast<uint8_t*>(data), len); }
};

// ai: the monitor (an XRandR output) the code goes on: its place on the root window and its mode's refresh
struct Monitor { std::string name; int x = 0, y = 0, w = 0, h = 0; double hz = 0; };

Monitor findMonitor(Display* d, const std::string& want) {
  const Window root = DefaultRootWindow(d);
  XRRScreenResources* res = XRRGetScreenResourcesCurrent(d, root);
  if (!res) throw std::runtime_error("XRandR gave no screen resources");
  const RROutput primary = XRRGetOutputPrimary(d, root);
  Monitor m, first;
  bool found = false, any = false;
  std::string names;
  for (int i = 0; i < res->noutput && !found; i++) {
    XRROutputInfo* o = XRRGetOutputInfo(d, res, res->outputs[i]);
    if (o->connection == RR_Connected && o->crtc) {
      XRRCrtcInfo* c = XRRGetCrtcInfo(d, res, o->crtc);
      Monitor here{o->name, c->x, c->y, static_cast<int>(c->width), static_cast<int>(c->height), 0};
      for (int k = 0; k < res->nmode; k++)
        if (res->modes[k].id == c->mode && res->modes[k].hTotal && res->modes[k].vTotal)
          here.hz = static_cast<double>(res->modes[k].dotClock) / (static_cast<double>(res->modes[k].hTotal) * res->modes[k].vTotal);
      XRRFreeCrtcInfo(c);
      names += (names.empty() ? "" : ", ") + here.name;
      if (!any) { first = here; any = true; }
      if (want.empty() ? res->outputs[i] == primary : want == here.name) { m = here; found = true; }
    }
    XRRFreeOutputInfo(o);
  }
  XRRFreeScreenResources(res);
  if (!found && want.empty() && any) return first;
  if (!found) throw std::runtime_error("no connected output " + want + " (there are: " + names + ")");
  return m;
}

// ai: A window over the monitor, full screen and out of the compositor where the window manager agrees; no cursor.
Window makeWindow(Display* d, const Monitor& m, Atom& del, const std::string& title) {
  const int s = DefaultScreen(d);
  XSetWindowAttributes a{};
  a.background_pixel = WhitePixel(d, s);
  a.event_mask = KeyPressMask | StructureNotifyMask | ExposureMask;
  const Window w = XCreateWindow(d, RootWindow(d, s), m.x, m.y, static_cast<unsigned>(m.w), static_cast<unsigned>(m.h), 0,
                                 CopyFromParent, InputOutput, CopyFromParent, CWBackPixel | CWEventMask, &a);
  XStoreName(d, w, title.c_str());
  XChangeProperty(d, w, XInternAtom(d, "_NET_WM_NAME", False), XInternAtom(d, "UTF8_STRING", False), 8, PropModeReplace,
                  reinterpret_cast<const unsigned char*>(title.data()), static_cast<int>(title.size()));
  XSizeHints sh{};
  sh.flags = USPosition | USSize;
  sh.x = m.x; sh.y = m.y; sh.width = m.w; sh.height = m.h;
  XSetWMNormalHints(d, w, &sh);
  const Atom state = XInternAtom(d, "_NET_WM_STATE", False), full = XInternAtom(d, "_NET_WM_STATE_FULLSCREEN", False);
  XChangeProperty(d, w, state, XA_ATOM, 32, PropModeReplace, reinterpret_cast<const unsigned char*>(&full), 1);
  const unsigned long one = 1;
  XChangeProperty(d, w, XInternAtom(d, "_NET_WM_BYPASS_COMPOSITOR", False), XA_CARDINAL, 32, PropModeReplace,
                  reinterpret_cast<const unsigned char*>(&one), 1);
  del = XInternAtom(d, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(d, w, &del, 1);
  char zero = 0;
  const Pixmap p = XCreateBitmapFromData(d, w, &zero, 1, 1);
  XColor black{};
  XDefineCursor(d, w, XCreatePixmapCursor(d, p, p, &black, &black, 0, 0));
  XFreePixmap(d, p);
  XMapRaised(d, w);
  // ai: the state asked again once mapped, for a manager that reads it only from a message
  XEvent e{};
  e.xclient.type = ClientMessage;
  e.xclient.window = w;
  e.xclient.message_type = state;
  e.xclient.format = 32;
  e.xclient.data.l[0] = 1;
  e.xclient.data.l[1] = static_cast<long>(full);
  e.xclient.data.l[3] = 1;
  XSendEvent(d, DefaultRootWindow(d), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
  XFlush(d);
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  signal(SIGINT, [](int) { gQuit = true; });
  signal(SIGTERM, [](int) { gQuit = true; });
  XInitThreads();
  Display* d = XOpenDisplay(nullptr);
  if (!d) { std::fprintf(stderr, "lizard_present: no X display\n"); return 1; }
  int status = 0;
  try {
    const Monitor mon = findMonitor(d, a.output);
    std::unique_ptr<Mapped> file;
    if (!a.file.empty()) file = std::make_unique<Mapped>(a.file);
    static const uint8_t empty = 0;
    const std::string base = a.file.substr(a.file.find_last_of('/') + 1);
    lizard::Sender sender(file ? (file->data ? file->data : &empty) : nullptr, file ? file->len : 0, base, "");
    std::mutex senderLock;
    lizard::TxFormat f;
    f.n = a.n ? a.n : nFor(a.subch);
    f.subch = a.subch;
    f.span = 2 * a.ring;
    f.fps = a.fps;
    f.codes = a.codes;
    f.threads = a.threads;
    f.painter = a.painter == "cpu" ? 0 : a.painter == "gpu" ? 1 : 2;
    f.assets = a.assets;
    f.hostFrames = false;
    f.aheadSecs = 1.0;
    f.aheadBytes = 256ull << 20;
    f.margin = 9;
    // ai: the device first, with what the presenter needs of it (the window's surface, the swapchain, present_wait):
    // ai: the code is shown from the painter's device (2026-10-07)
    const std::string prep = sender.prepareGpu(a.assets, "", &lizard::Presenter::deviceExtras());
    if (!prep.empty()) throw std::runtime_error("no Vulkan device to present from: " + prep);
    const std::string err = sender.configure(f);
    if (!err.empty()) throw std::runtime_error(err);
    std::printf("sender: LIZARD-%d%s, n %d in the %d ring, %d a second asked, a frame %d x %d px, painted on the %s%s%s, %s, clip %g\n",
                a.subch, a.codes > 1 ? " x 2" : "", f.n, a.ring, a.fps, sender.width(), sender.side(), sender.painter().c_str(),
                sender.gpuWhy().empty() ? "" : " (the GPU not: ", sender.gpuWhy().empty() ? "" : (sender.gpuWhy() + ")").c_str(),
                file ? ("the file " + a.file).c_str() : "the test stream", lizard::txClip());

    // ai: the window, until it is mapped and the size the manager gives it has settled
    Atom del = 0;
    const Window w = makeWindow(d, mon, del, "LIZARD");
    unsigned W = static_cast<unsigned>(mon.w), H = static_cast<unsigned>(mon.h);
    bool mapped = false;
    for (const double t = nowMs(); nowMs() - t < 1500;) {
      while (XPending(d)) {
        XEvent e;
        XNextEvent(d, &e);
        if (e.type == MapNotify) mapped = true;
        if (e.type == ConfigureNotify && e.xconfigure.window == w) {
          W = static_cast<unsigned>(e.xconfigure.width);
          H = static_cast<unsigned>(e.xconfigure.height);
        }
      }
      if (mapped && nowMs() - t > 400) break;
      usleep(5000);
    }

    lizard::PresentOptions o;
    o.fps = a.fps;
    o.whole = a.fit == "whole";
    o.csv = a.log;
    o.log = [](const std::string& m) { std::printf("%s\n", m.c_str()); std::fflush(stdout); };
    std::string why;
    lizard::PresentTarget target;
    target.x11Window = static_cast<unsigned long>(w);
    auto p = lizard::Presenter::start(target, &sender, &senderLock, o, &why);
    if (!p) throw std::runtime_error(why);
    std::printf("present: %s (%d x %d at %d,%d), %.3f Hz the mode's; window %u x %u; the frame %s\n", mon.name.c_str(), mon.w,
                mon.h, mon.x, mon.y, mon.hz, W, H, a.fit == "whole" ? "at whole pixels (nearest)" : "stretched (linear)");
    std::fflush(stdout);

    // ai: a line each time the presenter closes a second's window, beside the sender's stats
    const double t0 = nowMs();
    std::string last = p->stats();
    while (!gQuit && (a.secs <= 0 || nowMs() - t0 < a.secs * 1000)) {
      while (XPending(d)) {
        XEvent e;
        XNextEvent(d, &e);
        if (e.type == KeyPress) {
          const KeySym k = XLookupKeysym(&e.xkey, 0);
          if (k == XK_Escape || k == XK_q) gQuit = true;
        } else if (e.type == ClientMessage && static_cast<Atom>(e.xclient.data.l[0]) == del) gQuit = true;
      }
      const std::string st = p->stats();
      if (st != last) {
        last = st;
        const auto j = nlohmann::json::parse(st);
        if (!j.value("error", std::string()).empty()) throw std::runtime_error(j.value("error", std::string()));
        std::string ss;
        {
          std::lock_guard<std::mutex> l(senderLock);
          ss = sender.stats();
        }
        const auto s = nlohmann::json::parse(ss);
        std::printf("present: %.1f pictures a second, held (refreshes: pictures) %s; %lld refreshes missed, %lld behind; "
                    "%.2f Hz measured, %.1f presents a second, surface %s; the sender %s, paint %.1f ms, %d ahead\n",
                    j.value("shownFps", 0.0), heldText(j["held"]).c_str(), j.value("missed", 0LL), j.value("behind", 0LL),
                    j.value("hz", 0.0), j.value("presentsPerSec", 0.0), j.value("surface", std::string()).c_str(),
                    s.value("painter", std::string()).c_str(), s.value("paintMs", 0.0), s.value("ahead", 0));
        std::fflush(stdout);
      }
      usleep(20000);
    }

    p->stop();
    const auto t = nlohmann::json::parse(p->totals());
    const double secs = t.value("secs", 0.0);
    std::printf("summary: %.1f s, %lld pictures (%.2f a second), held (refreshes: pictures) %s; %lld refreshes missed, %lld behind, "
                "%lld presents with no time; %.3f Hz measured against the mode's %.3f\n",
                secs, t.value("pictures", 0LL), secs > 0 ? t.value("pictures", 0LL) / secs : 0.0, heldText(t["held"]).c_str(),
                t.value("missed", 0LL), t.value("behind", 0LL), t.value("unknown", 0LL), t.value("hz", 0.0), t.value("modeHz", 0.0));
    const auto j = nlohmann::json::parse(p->stats());
    if (!j.value("error", std::string()).empty()) {
      std::fprintf(stderr, "lizard_present: %s\n", j.value("error", std::string()).c_str());
      status = 1;
    }
    p.reset();
    XDestroyWindow(d, w);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "lizard_present: %s\n", e.what());
    status = 1;
  }
  XCloseDisplay(d);
  return status;
}
