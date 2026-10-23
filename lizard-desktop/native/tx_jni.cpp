// ai: The desktop sender's JNI face (dev.lizard.desktop.Native): lizard-android/app/src/main/cpp/jni.cpp's Tx without its
// ai: window code: a sender made on a mapped file or the test stream, configured, and its frames put on the screen by the
// ai: native presenter (presenter.h, 2026-10-04) in the app's code area, an AWT canvas whose native window JAWT hands over.
// ai: The Kotlin side loads jawt before this library (it links libjawt).
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <jni.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "portal.h"
#include "presenter.h"
#include "sender.h"
#if defined(__APPLE__)
#include "macos.h"
#endif

#include <jawt.h>
#include <jawt_md.h>

namespace {
// ai: A sender and the mutex every call into it holds: configure here on the UI's thread, take on the presenter's
// ai: thread, which only waits a moment for it (presenter.cpp lockFor)
struct Tx {
  std::unique_ptr<lizard::Sender> s;
  std::mutex mu;
  const uint8_t* map = nullptr;
  size_t mapLen = 0;
};
Tx* tx(jlong h) { return reinterpret_cast<Tx*>(h); }

// ai: the file to send mapped read-only for as long as the sender lives (the Sender reads it in place): "" or why not;
// ai: an empty file maps nothing
std::string mapFile(const std::string& path, const uint8_t*& map, size_t& len) {
#if defined(_WIN32)
  const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring wp(wn > 0 ? wn : 0, L'\0');
  if (wn > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wp.data(), wn);
  const HANDLE f = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  LARGE_INTEGER size{};
  if (f == INVALID_HANDLE_VALUE || !GetFileSizeEx(f, &size)) { if (f != INVALID_HANDLE_VALUE) CloseHandle(f); return "the file could not be read"; }
  len = static_cast<size_t>(size.QuadPart);
  if (len) {
    const HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    void* v = m ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (m) CloseHandle(m);
    if (!v) { CloseHandle(f); len = 0; return "the file could not be mapped"; }
    map = static_cast<const uint8_t*>(v);
  }
  CloseHandle(f);
#else
  const int fd = open(path.c_str(), O_RDONLY);
  struct stat st{};
  if (fd < 0 || fstat(fd, &st)) { if (fd >= 0) close(fd); return "the file could not be read"; }
  len = static_cast<size_t>(st.st_size);
  if (len) {
    void* m = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); len = 0; return "the file could not be mapped"; }
    map = static_cast<const uint8_t*>(m);
  }
  close(fd);
#endif
  return "";
}
void unmapFile(const uint8_t* map, size_t len) {
  if (!map) return;
#if defined(_WIN32)
  UnmapViewOfFile(map);
#else
  munmap(const_cast<uint8_t*>(map), len);
#endif
}
// ai: a presenter and, on macOS, the layer it presents to (made with it, released after it)
struct Pres {
  std::unique_ptr<lizard::Presenter> p;
  void* layer = nullptr;
  ~Pres() {
    p.reset();
#if defined(__APPLE__)
    lizard::mac::detachLayer(layer);
#endif
  }
};
lizard::Presenter* pr(jlong h) { return reinterpret_cast<Pres*>(h)->p.get(); }
std::string str(JNIEnv* e, jstring s) {
  if (!s) return "";
  const char* c = e->GetStringUTFChars(s, nullptr);
  std::string o(c ? c : "");
  if (c) e->ReleaseStringUTFChars(s, c);
  return o;
}
#if defined(__APPLE__)
// ai: the five numbers of a place (macos.h Place), a missing or short array the empty place
lizard::mac::Place placeOf(JNIEnv* e, jfloatArray a) {
  lizard::mac::Place p;
  if (!a || e->GetArrayLength(a) < 5) return p;
  jfloat v[5];
  e->GetFloatArrayRegion(a, 0, 5, v);
  p.x = v[0]; p.y = v[1]; p.w = v[2]; p.h = v[3]; p.scale = v[4];
  return p;
}
#endif
std::mutex gErrorMu;
std::string gError;
void setError(const std::string& m) {
  std::lock_guard<std::mutex> l(gErrorMu);
  gError = m;
}
}  // namespace

#define FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_dev_lizard_desktop_Native_##name

// ai: path: the file to send ("" for the test stream); name and type as its header carries them. 0, with the reason in
// ai: txError(), where it cannot go.
FN(jlong, txCreate)(JNIEnv* e, jclass, jstring path, jstring name, jstring type) {
  auto t = std::make_unique<Tx>();
  const std::string p = str(e, path);
  try {
    if (!p.empty()) {
      const std::string why = mapFile(p, t->map, t->mapLen);
      if (!why.empty()) { setError(why); return 0; }
    }
    static const uint8_t empty = 0;
    t->s = std::make_unique<lizard::Sender>(p.empty() ? nullptr : (t->map ? t->map : &empty), t->mapLen, str(e, name), str(e, type));
  } catch (const std::exception& x) {
    unmapFile(t->map, t->mapLen);
    setError(x.what());
    return 0;
  }
  return reinterpret_cast<jlong>(t.release());
}

FN(jstring, txError)(JNIEnv* e, jclass) {
  std::lock_guard<std::mutex> l(gErrorMu);
  return e->NewStringUTF(gError.c_str());
}

// ai: The device and the GPU painter made ahead, off the UI thread: "" or why there is none. The device carries what the
// ai: presenter needs of it (the window's surface, the swapchain, present_wait): the code is shown from it (2026-10-07).
FN(jstring, txPrepare)(JNIEnv* e, jclass, jlong h, jstring assets) {
  const std::string a = str(e, assets);
  std::lock_guard<std::mutex> l(tx(h)->mu);
  return e->NewStringUTF(tx(h)->s->prepareGpu(a, "", &lizard::Presenter::deviceExtras()).c_str());
}

// ai: The format to paint (n, sub-channels, span, the rate the word states, codes side by side and the gap between them
// ai: in modules, painters) and where (painter: 0 the CPU, 1 the GPU, 2 auto; assets: the GPU's kernels and tables): ""
// ai: or the refusal. A presenter running on this sender shows the last picture until the new format's first is painted.
FN(jstring, txConfigure)(JNIEnv* e, jclass, jlong h, jint n, jint subch, jint span, jint fps, jint codes, jint gap, jint threads, jint painter,
                         jstring assets) {
  lizard::TxFormat f;
  f.n = n; f.subch = subch; f.span = span; f.fps = fps; f.codes = codes; f.gap = gap; f.threads = threads; f.painter = painter;
  f.assets = str(e, assets);
  // ai: the frames stay on the device for the presenter (no host copies), a second of them ahead within 256 MB, the
  // ai: swapchain's images + 1 of margin (2026-10-07)
  f.hostFrames = false;
  f.aheadSecs = 1.0;
  f.aheadBytes = 256ull << 20;
  f.margin = 9;
  std::string r;
  // ai: no C++ exception crosses into the JVM (a thread or an allocation refused inside configure would abort it):
  // ai: the reason comes back as the refusal (2026-10-04)
  try {
    std::lock_guard<std::mutex> l(tx(h)->mu);
    r = tx(h)->s->configure(f);
  } catch (const std::exception& x) {
    r = std::string("the format could not be set: ") + x.what();
  } catch (...) {
    r = "the format could not be set";
  }
  return e->NewStringUTF(r.c_str());
}

// ai: The blocks a frame of subch sub-channels carries under the format's rate profile (src/focus.h focus_blocks_for),
// ai: for the Blocks slider's figures before a configure
FN(jint, blocksFor)(JNIEnv*, jclass, jint subch) { return focus_blocks_for(subch); }

FN(jstring, txStats)(JNIEnv* e, jclass, jlong h) {
  std::string s;
  {
    std::lock_guard<std::mutex> l(tx(h)->mu);
    s = tx(h)->s->stats();
  }
  return e->NewStringUTF(s.c_str());
}

// ai: After presentStop on every presenter of this sender.
FN(void, txDestroy)(JNIEnv*, jclass, jlong h) {
  Tx* t = tx(h);
  if (!t) return;
  t->s.reset();
  unmapFile(t->map, t->mapLen);
  delete t;
}

// ai: a heavyweight canvas's native window (X11's drawable, Windows' HWND), JAWT locked only to read it; false, with
// ai: why in txError(), where it has none yet
#if !defined(__APPLE__)
static bool canvasTarget(JNIEnv* e, jobject canvas, lizard::PresentTarget& target) {
  JAWT awt;
  awt.version = JAWT_VERSION_9;
#if defined(_WIN32)
  // ai: Windows: JAWT_GetAWT from jawt.dll, which the app loads before this library (Native.kt): no import library is
  // ai: needed to link, so the DLL cross-builds from Linux with MinGW (2026-10-04)
  using GetAWT = jboolean(JNICALL*)(JNIEnv*, JAWT*);
  HMODULE jm = GetModuleHandleW(L"jawt.dll");
  const auto getAwt = jm ? reinterpret_cast<GetAWT>(reinterpret_cast<void*>(GetProcAddress(jm, "JAWT_GetAWT"))) : nullptr;
  if (!getAwt || getAwt(e, &awt) == JNI_FALSE) { setError("JAWT is not available"); return false; }
#else
  if (JAWT_GetAWT(e, &awt) == JNI_FALSE) { setError("JAWT is not available"); return false; }
#endif
  JAWT_DrawingSurface* ds = awt.GetDrawingSurface(e, canvas);
  if (!ds) { setError("the canvas has no drawing surface"); return false; }
  const jint lock = ds->Lock(ds);
  if (lock & JAWT_LOCK_ERROR) { awt.FreeDrawingSurface(ds); setError("the canvas's drawing surface could not be locked"); return false; }
  JAWT_DrawingSurfaceInfo* dsi = ds->GetDrawingSurfaceInfo(ds);
  if (dsi) {
#if defined(_WIN32)
    target.hwnd = static_cast<JAWT_Win32DrawingSurfaceInfo*>(dsi->platformInfo)->hwnd;
#else
    target.x11Window = static_cast<unsigned long>(static_cast<JAWT_X11DrawingSurfaceInfo*>(dsi->platformInfo)->drawable);
#endif
    ds->FreeDrawingSurfaceInfo(dsi);
  }
  ds->Unlock(ds);
  awt.FreeDrawingSurface(ds);
  if (!target.x11Window && !target.hwnd) { setError("the canvas has no native window yet"); return false; }
  return true;
}
#endif

// ai: The presenter on a canvas (a heavyweight java.awt.Component, displayable: its native window made), the sender's
// ai: frames on it: fps pictures a second asked, size the code's share of the room (0.25 to 1), edge the light kept
// ai: at each edge (presenter.h PresentOptions), whole pixels or stretched, the device the sender's (txPrepare). JAWT
// ai: is held only to read the window's handle. 0, with why in txError(), where it cannot start. presentStop before the
// ai: canvas goes or the sender is destroyed.
// ai: place: macOS's layer over the canvas (x, y, w, h in points from the top left of the window's content, the
// ai: backing scale; presentPlace when it moves), ignored elsewhere.
// ai: vsync: the monitor held at a refresh fps divides while presenting (presenter.h setVsync; presentVsync after).
FN(jlong, presentStart)(JNIEnv* e, jclass, jobject canvas, jlong h, jint fps, jfloat size, jfloat edge, jboolean whole, jboolean vsync, jfloatArray place) {
  if (!h) { setError("no sender"); return 0; }
  auto out = std::make_unique<Pres>();
  lizard::PresentTarget target;
#if defined(__APPLE__)
  std::string why;
  out->layer = lizard::mac::attachLayer(e, canvas, placeOf(e, place), &why);
  if (!out->layer) { setError(why); return 0; }
  target.metalLayer = out->layer;
#else
  (void)place;
  if (!canvasTarget(e, canvas, target)) return 0;
#endif
  lizard::PresentOptions o;
  o.fps = fps;
  o.size = size;
  o.edge = edge;
  o.whole = whole == JNI_TRUE;
  o.vsync = vsync == JNI_TRUE;
  // ai: LIZ_PRESENT_CSV=<path>: a line a present (lizard_present --log's), for reading a run's cadence around a resize
  if (const char* c = std::getenv("LIZ_PRESENT_CSV")) o.csv = c;
  o.log = [](const std::string& m) { std::fprintf(stderr, "lizard: %s\n", m.c_str()); };
  std::string err;
  try {
    out->p = lizard::Presenter::start(target, tx(h)->s.get(), &tx(h)->mu, o, &err);
    if (!out->p) { setError(err); return 0; }
    return reinterpret_cast<jlong>(out.release());
  } catch (const std::exception& x) {
    setError(std::string("the presenter could not start: ") + x.what());
  } catch (...) {
    setError("the presenter could not start");
  }
  return 0;
}

// ai: macOS: the layer moved to where the canvas is now (presentStart's place); nothing elsewhere, where the canvas's
// ai: own window carries the swapchain
FN(void, presentPlace)(JNIEnv* e, jclass, jlong p, jfloatArray place) {
#if defined(__APPLE__)
  if (p) lizard::mac::placeLayer(reinterpret_cast<Pres*>(p)->layer, placeOf(e, place));
#else
  (void)e; (void)p; (void)place;
#endif
}

FN(void, presentFps)(JNIEnv*, jclass, jlong p, jint fps) { if (p) pr(p)->setFps(fps); }
FN(void, presentVsync)(JNIEnv*, jclass, jlong p, jboolean on) { if (p) pr(p)->setVsync(on == JNI_TRUE); }
FN(void, presentSize)(JNIEnv*, jclass, jlong p, jfloat size, jfloat edge) { if (p) pr(p)->setSize(size, edge); }
FN(void, presentPause)(JNIEnv*, jclass, jlong p, jboolean on) { if (p) pr(p)->setPaused(on == JNI_TRUE); }
// ai: presenter.h stats(): the last second's window as JSON
FN(jstring, presentStats)(JNIEnv* e, jclass, jlong p) { return e->NewStringUTF(p ? pr(p)->stats().c_str() : ""); }
FN(void, presentStop)(JNIEnv*, jclass, jlong p) { delete reinterpret_cast<Pres*>(p); }

// ai: The desktop's file chooser (2026-10-23, portal.cpp). windowOf: the window manager's window around the canvas (the
// ai: ancestor with WM_STATE; Windows: the canvas's root window), for the dialog's parent; 0 where there is none
// ai: (macOS, no window yet). On the UI thread.
FN(jlong, windowOf)(JNIEnv* e, jclass, jobject canvas) {
#if defined(__linux__)
  lizard::PresentTarget target;
  if (!canvasTarget(e, canvas, target)) return 0;
  return static_cast<jlong>(lizard::x11TopLevel(target.x11Window));
#elif defined(_WIN32)
  lizard::PresentTarget target;
  if (!canvasTarget(e, canvas, target)) return 0;
  return static_cast<jlong>(reinterpret_cast<uintptr_t>(GetAncestor(static_cast<HWND>(target.hwnd), GA_ROOT)));
#else
  (void)e; (void)canvas;
  return 0;
#endif
}
// ai: portalPick: the portal's OpenFile, parent the window windowOf gave (0 none), files or folders, several where
// ai: multiple; blocks until the dialog closes, so never on the UI thread. The file:// URIs chosen (none: cancelled), or
// ai: null with why in txError() where the portal could not be asked.
FN(jobjectArray, portalPick)(JNIEnv* e, jclass, jlong window, jstring title, jboolean directory, jboolean multiple) {
  char parent[40] = "";
#if defined(_WIN32)
  if (window) std::snprintf(parent, sizeof parent, "win32:%llx", static_cast<unsigned long long>(window));
#else
  if (window) std::snprintf(parent, sizeof parent, "x11:%lx", static_cast<unsigned long>(window));
#endif
  const lizard::PortalPick r = lizard::portalOpen(parent, str(e, title), directory == JNI_TRUE, multiple == JNI_TRUE);
  if (!r.answered) { setError(r.error); return nullptr; }
  jobjectArray out = e->NewObjectArray(static_cast<jsize>(r.uris.size()), e->FindClass("java/lang/String"), nullptr);
  for (size_t i = 0; i < r.uris.size(); i++) {
    jstring u = e->NewStringUTF(r.uris[i].c_str());
    e->SetObjectArrayElement(out, static_cast<jsize>(i), u);
    e->DeleteLocalRef(u);
  }
  return out;
}
// ai: full screen as other apps ask for it (2026-10-23, presenter.cpp x11Fullscreen): the window manager's own for the
// ai: window windowOf gave; false where it could not be asked (not X11), for Compose's placement instead
FN(jboolean, windowFullscreen)(JNIEnv*, jclass, jlong window, jboolean on) {
  return window && lizard::x11Fullscreen(static_cast<unsigned long>(window), on == JNI_TRUE) ? JNI_TRUE : JNI_FALSE;
}
FN(jboolean, windowIsFullscreen)(JNIEnv*, jclass, jlong window) {
  return window && lizard::x11IsFullscreen(static_cast<unsigned long>(window)) ? JNI_TRUE : JNI_FALSE;
}
