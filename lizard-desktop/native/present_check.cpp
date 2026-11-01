// ai: The presenter's own check on X11 (2026-10-31): the sender's test stream painted on the GPU and shown by
// ai: lizard::Presenter in a window of this process's own, with no JVM, so a run measures the presenter alone. A window
// ai: at x, y of w x h on the root (the window manager's full screen there with full=1), the presenter's stats a second
// ai: on stderr, its CSV a line a present (csv=), and with mark=1 the picture's count drawn into the window's top-left
// ai: corner (PresentOptions::mark) for tools that read the screen back.
// ai:   present_check [secs=30] [fps=60] [version=41] [ring=256] [codes=2] [gap=12] [x=0] [y=0] [w=1600] [h=900]
// ai:                 [full=0] [above=0] [direct=<output>] [size=0.9] [margin=2] [vsync=0] [mark=0] [csv=<path>] [device=<name>]
// ai:                 [assets=<liblizard/out>]
// ai: direct=<output>: the presenter takes that monitor (PresentOptions::direct), this window stays unmapped, and while
// ai: it is taken any key press or click gives it back and ends the run (the presenter grabs the keyboard and the
// ai: pointer); TERM or INT end it too. Exit 0 after secs with a summary line on stdout (the presenter's last stats and the sender's), 1 with FAIL <why>.
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "presenter.h"
#include "sender.h"

namespace {

// ai: a TERM or an INT ends the run as its time would (the presenter stopped, a display taken given back)
std::atomic<bool> gQuit{false};
void onSignal(int) { gQuit = true; }

std::map<std::string, std::string> parse(int argc, char** argv) {
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; i++) {
    const char* eq = std::strchr(argv[i], '=');
    if (eq) kv[std::string(argv[i], static_cast<size_t>(eq - argv[i]))] = eq + 1;
  }
  return kv;
}

double num(const std::map<std::string, std::string>& kv, const char* k, double d) {
  auto it = kv.find(k);
  return it == kv.end() ? d : std::atof(it->second.c_str());
}

std::string str(const std::map<std::string, std::string>& kv, const char* k, const std::string& d) {
  auto it = kv.find(k);
  return it == kv.end() ? d : it->second;
}

int fail(const std::string& why) {
  std::printf("FAIL %s\n", why.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const auto kv = parse(argc, argv);
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  const double secs = num(kv, "secs", 30);
  const int fps = static_cast<int>(num(kv, "fps", 60));
  const int version = static_cast<int>(num(kv, "version", 41));
  const int ring = static_cast<int>(num(kv, "ring", 256));
  const int codes = static_cast<int>(num(kv, "codes", 2));
  const int gap = static_cast<int>(num(kv, "gap", 12));
  const int x = static_cast<int>(num(kv, "x", 0)), y = static_cast<int>(num(kv, "y", 0));
  const int w = static_cast<int>(num(kv, "w", 1600)), h = static_cast<int>(num(kv, "h", 900));
  const bool full = num(kv, "full", 0) != 0;
  const std::string assets = str(kv, "assets", "../liblizard/out");

  Display* d = XOpenDisplay(nullptr);
  if (!d) return fail("no X display");
  XSetWindowAttributes swa{};
  swa.background_pixel = WhitePixel(d, DefaultScreen(d));
  swa.event_mask = StructureNotifyMask | ExposureMask;
  const Window win = XCreateWindow(d, DefaultRootWindow(d), x, y, static_cast<unsigned>(w), static_cast<unsigned>(h), 0, CopyFromParent,
                                   InputOutput, CopyFromParent, CWBackPixel | CWEventMask, &swa);
  XStoreName(d, win, "LIZARD present_check");
  XSizeHints hints{};
  hints.flags = USPosition | USSize | PPosition | PSize;
  hints.x = x; hints.y = y; hints.width = w; hints.height = h;
  XSetWMNormalHints(d, win, &hints);
  // ai: direct=: the window stays unmapped (the presenter's fallback target only), so nothing shows on the desktop's
  // ai: other monitors while the output is taken
  const bool direct = !str(kv, "direct", "").empty();
  if (!direct) {
    XMapWindow(d, win);
    for (;;) {
      XEvent e;
      XNextEvent(d, &e);
      if (e.type == MapNotify) break;
    }
  }
  XMoveWindow(d, win, x, y);
  XSync(d, False);
  if (full) lizard::x11Fullscreen(lizard::x11TopLevel(win), true);
  if (num(kv, "above", 0) != 0) {
    // ai: the check's window kept above the others (_NET_WM_STATE_ABOVE), so a tool reading the screen back sees it
    XEvent e{};
    e.xclient.type = ClientMessage;
    e.xclient.window = win;
    e.xclient.message_type = XInternAtom(d, "_NET_WM_STATE", False);
    e.xclient.format = 32;
    e.xclient.data.l[0] = 1;
    e.xclient.data.l[1] = static_cast<long>(XInternAtom(d, "_NET_WM_STATE_ABOVE", False));
    e.xclient.data.l[3] = 1;
    XSendEvent(d, DefaultRootWindow(d), False, SubstructureRedirectMask | SubstructureNotifyMask, &e);
    XFlush(d);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(full ? 800 : 300));
  {
    int rx = 0, ry = 0;
    Window child = 0;
    XWindowAttributes wa{};
    XGetWindowAttributes(d, win, &wa);
    XTranslateCoordinates(d, win, DefaultRootWindow(d), 0, 0, &rx, &ry, &child);
    std::fprintf(stderr, "present_check: window 0x%lx at %d,%d, %d x %d\n", win, rx, ry, wa.width, wa.height);
  }

  lizard::Sender sender(nullptr, 0, "test", "application/octet-stream");
  std::mutex lock;
  std::string why;
  {
    std::lock_guard<std::mutex> l(lock);
    why = sender.prepareGpu(assets, str(kv, "device", ""), &lizard::Presenter::deviceExtras());
  }
  if (!why.empty()) return fail("prepareGpu: " + why);
  lizard::TxFormat f;
  f.subch = version * 16;
  f.span = 2 * ring;
  f.fps = fps;
  f.codes = codes;
  f.gap = gap;
  f.painter = 1;
  f.assets = assets;
  f.hostFrames = false;
  f.aheadSecs = 1.0;
  f.aheadBytes = 256ull << 20;
  f.margin = 9;
  {
    std::lock_guard<std::mutex> l(lock);
    why = sender.configure(f);
  }
  if (!why.empty()) return fail("configure: " + why);

  lizard::PresentTarget target;
  target.x11Window = win;
  lizard::PresentOptions o;
  o.fps = fps;
  o.size = num(kv, "size", 0.9);
  // ai: margin=<modules>: the light kept past the codec's quiet zone (2 modules) at each edge of the area, as the app's
  // ai: Margin (Room.edgeShare: a share of the frame's height, the ring's span and its border and quiet zones)
  o.edge = (num(kv, "margin", 2) - 2) / (2.0 * ring + 2 * 15 + 2 * 2);
  o.vsync = num(kv, "vsync", 0) != 0;
  o.mark = num(kv, "mark", 0) != 0;
  o.direct = str(kv, "direct", "");
  o.csv = str(kv, "csv", "");
  o.log = [](const std::string& m) { std::fprintf(stderr, "lizard: %s\n", m.c_str()); };
  auto p = lizard::Presenter::start(target, &sender, &lock, o, &why);
  if (!p) return fail("presenter: " + why);

  const auto t0 = std::chrono::steady_clock::now();
  auto next = t0 + std::chrono::seconds(1);
  std::string last;
  while (!gQuit && std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(secs)) {
    while (XPending(d)) {
      XEvent e;
      XNextEvent(d, &e);
    }
    std::this_thread::sleep_until(next);
    next += std::chrono::seconds(1);
    last = p->stats();
    std::fprintf(stderr, "present: %s\n", last.c_str());
    // ai: direct=: a key or a click gave the display back (the presenter's grab): the run ends there
    if (direct && last.find("\"leaseEnded\":true") != std::string::npos) break;
  }
  p->stop();
  last = p->stats();
  std::string tx;
  {
    std::lock_guard<std::mutex> l(lock);
    tx = sender.stats();
  }
  std::printf("{\"present\":%s,\"sender\":%s}\n", last.c_str(), tx.c_str());
  p.reset();
  if (full) lizard::x11Fullscreen(lizard::x11TopLevel(win), false);
  XDestroyWindow(d, win);
  XCloseDisplay(d);
  return 0;
}
