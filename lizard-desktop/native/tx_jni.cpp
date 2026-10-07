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

#include "presenter.h"
#include "sender.h"

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
lizard::Presenter* pr(jlong h) { return reinterpret_cast<lizard::Presenter*>(h); }
std::string str(JNIEnv* e, jstring s) {
  if (!s) return "";
  const char* c = e->GetStringUTFChars(s, nullptr);
  std::string o(c ? c : "");
  if (c) e->ReleaseStringUTFChars(s, c);
  return o;
}
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
  // ai: LIZ_TIERS in the app's environment (the lab's rate-by-ring arm, 2026-10-07): the profile painted in place of
  // ai: the format asked for (TxFormat.tiers); the window's figures still count the blocks it asked for
  if (const char* t = getenv("LIZ_TIERS"); t && *t) f.tiers = t;
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

FN(jint, txSide)(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> l(tx(h)->mu);
  return tx(h)->s->side();
}
FN(jint, txWidth)(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> l(tx(h)->mu);
  return tx(h)->s->width();
}
FN(jboolean, txReady)(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> l(tx(h)->mu);
  return tx(h)->s->ready() ? JNI_TRUE : JNI_FALSE;
}

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

// ai: The presenter on a canvas (a heavyweight java.awt.Component, displayable: its native window made), the sender's
// ai: frames on it: fps pictures a second asked, size the code's share of the room (0.25 to 1), whole pixels or
// ai: stretched, the device the sender's (txPrepare). JAWT
// ai: is held only to read the window's handle. 0, with why in txError(), where it cannot start. presentStop before the
// ai: canvas goes or the sender is destroyed.
FN(jlong, presentStart)(JNIEnv* e, jclass, jobject canvas, jlong h, jint fps, jfloat size, jboolean whole) {
  if (!h) { setError("no sender"); return 0; }
#if defined(__APPLE__)
  // ai: macOS needs a CAMetalLayer through JAWT's surface layers and MoltenVK: not built yet
  (void)e; (void)canvas; (void)fps; (void)size; (void)whole;
  setError("presenting is not built for macOS yet");
  return 0;
#else
  JAWT awt;
  awt.version = JAWT_VERSION_9;
#if defined(_WIN32)
  // ai: Windows: JAWT_GetAWT from jawt.dll, which the app loads before this library (Native.kt): no import library is
  // ai: needed to link, so the DLL cross-builds from Linux with MinGW (2026-10-04)
  using GetAWT = jboolean(JNICALL*)(JNIEnv*, JAWT*);
  HMODULE jm = GetModuleHandleW(L"jawt.dll");
  const auto getAwt = jm ? reinterpret_cast<GetAWT>(reinterpret_cast<void*>(GetProcAddress(jm, "JAWT_GetAWT"))) : nullptr;
  if (!getAwt || getAwt(e, &awt) == JNI_FALSE) { setError("JAWT is not available"); return 0; }
#else
  if (JAWT_GetAWT(e, &awt) == JNI_FALSE) { setError("JAWT is not available"); return 0; }
#endif
  JAWT_DrawingSurface* ds = awt.GetDrawingSurface(e, canvas);
  if (!ds) { setError("the canvas has no drawing surface"); return 0; }
  const jint lock = ds->Lock(ds);
  if (lock & JAWT_LOCK_ERROR) { awt.FreeDrawingSurface(ds); setError("the canvas's drawing surface could not be locked"); return 0; }
  lizard::PresentTarget target;
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
  if (!target.x11Window && !target.hwnd) { setError("the canvas has no native window yet"); return 0; }
  lizard::PresentOptions o;
  o.fps = fps;
  o.size = size;
  o.whole = whole == JNI_TRUE;
  // ai: LIZ_PRESENT_CSV=<path>: a line a present (lizard_present --log's), for reading a run's cadence around a resize
  if (const char* c = std::getenv("LIZ_PRESENT_CSV")) o.csv = c;
  o.log = [](const std::string& m) { std::fprintf(stderr, "lizard: %s\n", m.c_str()); };
  std::string err;
  try {
    auto p = lizard::Presenter::start(target, tx(h)->s.get(), &tx(h)->mu, o, &err);
    if (!p) { setError(err); return 0; }
    return reinterpret_cast<jlong>(p.release());
  } catch (const std::exception& x) {
    setError(std::string("the presenter could not start: ") + x.what());
  } catch (...) {
    setError("the presenter could not start");
  }
  return 0;
#endif
}

FN(void, presentFps)(JNIEnv*, jclass, jlong p, jint fps) { if (p) pr(p)->setFps(fps); }
FN(void, presentSize)(JNIEnv*, jclass, jlong p, jfloat size) { if (p) pr(p)->setSize(size); }
FN(void, presentWhole)(JNIEnv*, jclass, jlong p, jboolean on) { if (p) pr(p)->setWhole(on == JNI_TRUE); }
FN(void, presentPause)(JNIEnv*, jclass, jlong p, jboolean on) { if (p) pr(p)->setPaused(on == JNI_TRUE); }
// ai: Full screen's part here: the compositor bypass on the window's top level (X11; the app makes the window full screen)
FN(void, presentFullscreen)(JNIEnv*, jclass, jlong p, jboolean on) { if (p) pr(p)->setBypassCompositor(on == JNI_TRUE); }
// ai: presenter.h stats(): the last second's window as JSON
FN(jstring, presentStats)(JNIEnv* e, jclass, jlong p) { return e->NewStringUTF(p ? pr(p)->stats().c_str() : ""); }
FN(void, presentStop)(JNIEnv*, jclass, jlong p) { delete pr(p); }
