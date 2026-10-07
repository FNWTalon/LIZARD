// ai: The JNI glue between the app (Native.kt) and core's receiver (liblizard/core/rx/receiver.h), and nothing else: a handle is
// ai: a Receiver; a frame's HardwareBuffer goes in as its AHardwareBuffer (no copy); the receiver's release(tag), called
// ai: from its own threads, comes back to Native.release, which closes the camera's Image.
#include <android/hardware_buffer_jni.h>
#include <android/log.h>
#include <jni.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/system_properties.h>

#include <memory>
#include <string>
#include <vector>

#include "receiver.h"
#include "replay.h"
#include "sender.h"

#include <android/native_window_jni.h>
#include <android/surface_control.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mutex>

#define TAG "lizard"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

JavaVM* gVm = nullptr;
jclass gNative = nullptr;     // ai: dev.lizard.receiver.Native, a global ref
jmethodID gRelease = nullptr; // ai: Native.release(long), static
pthread_key_t gDetach;        // ai: set on the threads attached here, so they detach when they end

// ai: The calling thread's JNIEnv: a receiver thread is attached on its first release and stays attached (an attach per
// ai: frame costs more than the release), detached by gDetach's destructor when the thread exits.
JNIEnv* envHere() {
  JNIEnv* e = nullptr;
  if (gVm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) == JNI_OK) return e;
  JavaVMAttachArgs a{JNI_VERSION_1_6, "lizard-rx", nullptr};
  if (gVm->AttachCurrentThread(&e, &a) != JNI_OK) return nullptr;
  pthread_setspecific(gDetach, e);
  return e;
}

void release(uint64_t tag) {
  JNIEnv* e = envHere();
  if (!e) { LOGE("release %llu: no JNIEnv", static_cast<unsigned long long>(tag)); return; }
  e->CallStaticVoidMethod(gNative, gRelease, static_cast<jlong>(tag));
  if (e->ExceptionCheck()) { e->ExceptionDescribe(); e->ExceptionClear(); }
}

std::string str(JNIEnv* e, jstring s) {
  if (!s) return {};
  const char* c = e->GetStringUTFChars(s, nullptr);
  std::string r(c ? c : "");
  if (c) e->ReleaseStringUTFChars(s, c);
  return r;
}

lizard::Receiver* rx(jlong h) { return reinterpret_cast<lizard::Receiver*>(h); }

void throwJava(JNIEnv* e, const char* cls, const std::string& msg) {
  jclass c = e->FindClass(cls);
  if (c) e->ThrowNew(c, msg.c_str());
}

}  // namespace

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  gVm = vm;
  JNIEnv* e = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
  jclass c = e->FindClass("dev/lizard/receiver/Native");
  if (!c) return JNI_ERR;
  gNative = static_cast<jclass>(e->NewGlobalRef(c));
  gRelease = e->GetStaticMethodID(gNative, "release", "(J)V");
  if (!gRelease) return JNI_ERR;
  pthread_key_create(&gDetach, [](void*) { gVm->DetachCurrentThread(); });
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL Java_dev_lizard_receiver_Native_create(JNIEnv* e, jclass, jstring assets,
    jstring cacheDir, jstring storeDir, jstring decoder, jstring precision, jstring layout) {
  lizard::ReceiverConfig c;
  c.assets = str(e, assets);
  c.cacheDir = str(e, cacheDir);
  c.storeDir = str(e, storeDir);
  c.decoder = str(e, decoder);
  c.precision = str(e, precision);
  c.layout = str(e, layout);
  // ai: the core's LIZ_* switches for a run driven over adb: `adb shell setprop debug.lizard.env "LIZ_PARTS=0 ..."`,
  // ai: read as each receiver is made
  {
    char v[PROP_VALUE_MAX] = "";
    __system_property_get("debug.lizard.env", v);
    std::string all(v);
    for (size_t a = 0; a < all.size();) {
      size_t b = all.find(' ', a);
      if (b == std::string::npos) b = all.size();
      const std::string kv = all.substr(a, b - a);
      const size_t eq = kv.find('=');
      if (eq != std::string::npos && eq > 0) { setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1); __android_log_print(ANDROID_LOG_INFO, "lizard", "env %s", kv.c_str()); }
      a = b + 1;
    }
  }
  c.log = [](const std::string& s) { LOGI("%s", s.c_str()); };
  try {
    auto r = lizard::Receiver::create(c, release);
    if (!r) { throwJava(e, "java/lang/IllegalStateException", "Receiver::create returned nothing"); return 0; }
    return reinterpret_cast<jlong>(r.release());
  } catch (const std::exception& x) {
    throwJava(e, "java/lang/IllegalStateException", std::string("Receiver::create: ") + x.what());
    return 0;
  }
}

// ai: Throws (IllegalArgumentException) only before the receiver has the frame, so the caller closes the Image at once;
// ai: once pushed, the frame is the receiver's until its release(tag), even when it drops it.
extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_push(JNIEnv* e, jclass, jlong h, jobject hb,
    jobject luma, jint width, jint height, jint stride, jlong timestampNs, jlong tag) {
  lizard::CameraFrame f;
  if (hb) f.hb = AHardwareBuffer_fromHardwareBuffer(e, hb);
  if (luma) f.luma = static_cast<const uint8_t*>(e->GetDirectBufferAddress(luma));
  if (!h || (!f.hb && !f.luma)) {
    throwJava(e, "java/lang/IllegalArgumentException", !h ? "no receiver" : "a frame with neither a buffer nor luma");
    return;
  }
  f.width = static_cast<uint32_t>(width);
  f.height = static_cast<uint32_t>(height);
  f.stride = static_cast<uint32_t>(stride);
  f.timestampNs = timestampNs;
  f.tag = static_cast<uint64_t>(tag);
  rx(h)->push(f);
}

// ai: a system property's value ("" when unset): the switches a run driven over adb sets while the app runs
// ai: (Engine.kt's delay test, debug.lizard.nudge)
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_prop(JNIEnv* e, jclass, jstring name) {
  const char* n = e->GetStringUTFChars(name, nullptr);
  char v[PROP_VALUE_MAX] = "";
  __system_property_get(n, v);
  e->ReleaseStringUTFChars(name, n);
  return e->NewStringUTF(v);
}

extern "C" JNIEXPORT jboolean JNICALL Java_dev_lizard_receiver_Native_wantsLuma(JNIEnv*, jclass, jlong h) {
  return h && rx(h)->wantsLuma() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_stats(JNIEnv* e, jclass, jlong h) {
  return e->NewStringUTF(h ? rx(h)->stats().c_str() : "{}");
}

// ai: Receiver::series as a double[]: [version, then 8 doubles a frame captured after sinceMs]; asked at every
// ai: capture result, so no JSON is made or parsed on that path
extern "C" JNIEXPORT jdoubleArray JNICALL Java_dev_lizard_receiver_Native_series(JNIEnv* e, jclass, jlong h, jdouble sinceMs) {
  const std::vector<double> v = h ? rx(h)->series(sinceMs) : std::vector<double>{0};
  jdoubleArray a = e->NewDoubleArray(static_cast<jsize>(v.size()));
  if (a) e->SetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
  return a;   // ai: null only with the VM's OutOfMemoryError pending
}

extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_soon(JNIEnv*, jclass, jlong h, jboolean on) {
  if (h) rx(h)->soon(on == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_batchCap(JNIEnv*, jclass, jlong h, jint n) {
  if (h) rx(h)->batchCap(n);
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_file(JNIEnv* e, jclass, jlong h) {
  return e->NewStringUTF(h ? rx(h)->file().c_str() : "");
}

// ai: The camera's reader closed (Engine.closeReader, every frame of it handed back): Receiver::cameraClosed.
extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_cameraClosed(JNIEnv*, jclass, jlong h) { if (h) rx(h)->cameraClosed(); }
// ai: the transfer in hand forgotten, and the held word with it (Receiver::clear): the app's delete of the file it shows
// ai: (2026-10-07), so the same file in the light is received anew, as the web's Clear did
extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_clear(JNIEnv*, jclass, jlong h) { if (h) rx(h)->clear(); }

// ai: The receiver's destructor releases every frame it still holds (on this thread or its own) before it returns.
extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_destroy(JNIEnv*, jclass, jlong h) {
  delete rx(h);
}

// ai: Save replays (Developer Tools, 2026-10-03; liblizard/core/rx/replay.h): a handle is a shared_ptr to the replay, the receiver
// ai: holding another while it takes frames, so the run outlives either; replayEnd lets the handle's go.
namespace {
std::shared_ptr<lizard::Replay>& replayOf(jlong r) { return *reinterpret_cast<std::shared_ptr<lizard::Replay>*>(r); }
}

extern "C" JNIEXPORT jlong JNICALL Java_dev_lizard_receiver_Native_replayNew(JNIEnv* e, jclass, jstring dir, jstring run, jint frames) {
  return reinterpret_cast<jlong>(new std::shared_ptr<lizard::Replay>(std::make_shared<lizard::Replay>(str(e, dir), str(e, run), frames)));
}

extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_record(JNIEnv*, jclass, jlong h, jlong replay) {
  if (h) rx(h)->record(replay ? replayOf(replay) : nullptr);
}

// ai: The run's end (Replay::finish, which waits for the frames the GPU decoder still holds for it), the handle let
// ai: go: {"frames", "bytes", "error"} as JSON, error the write that ended the frames early ("" none); an
// ai: IllegalStateException where the folder, a rename or the meta failed.
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_replayEnd(JNIEnv* e, jclass, jlong replay, jstring rxStats,
    jstring rows, jstring more) {
  if (!replay) { throwJava(e, "java/lang/IllegalStateException", "no replay"); return nullptr; }
  auto* p = &replayOf(replay);
  const int n = (*p)->finish(str(e, rxStats), str(e, rows), str(e, more));
  const long long bytes = (*p)->bytes();
  const std::string why = (*p)->error();
  delete p;
  if (n < 0) { throwJava(e, "java/lang/IllegalStateException", why); return nullptr; }
  std::string quoted;
  for (const char ch : why) {
    if (ch == '"' || ch == '\\') { quoted += '\\'; quoted += ch; }
    else if ((unsigned char)ch < 0x20) quoted += ' ';
    else quoted += ch;
  }
  const std::string out = "{\"frames\":" + std::to_string(n) + ",\"bytes\":" + std::to_string(bytes) + ",\"error\":\"" + quoted + "\"}";
  return e->NewStringUTF(out.c_str());
}

// ai: The sender (2026-10-01; liblizard/core/tx/sender.h): a handle is a Tx, the
// ai: Sender over the picked file mapped read-only (the app copies the document into its cache first), or over nothing
// ai: for the test stream; each frame goes straight into the SurfaceView's window, its buffers sized to the painted
// ai: frame and scaled to the view by the compositor.
namespace {
// ai: The vsync-locked present (2026-10-02). Android never
// ai: tears, but a buffer posted into the window from the main thread is shown at whichever refresh the compositor
// ai: latches it for, and the S26's panel (adaptive refresh, a 240 Hz grid) shows a buffer as soon as it can, so the
// ai: main thread's jitter became each picture's length. Here every frame goes in a transaction on a child surface of
// ai: the SurfaceView, named for the vsync the Choreographer's frame timeline gives it (ASurfaceTransaction_
// ai: setFrameTimeline, Android 13), so the compositor presents it at that vsync's time; its frame rate voted on that
// ai: surface with the switch always made. A ring of hardware buffers the CPU paints into, each free again once the
// ai: compositor's release fence for it has signalled (the next transaction's completion hands that fence over).
struct Ring {
  struct Slot { AHardwareBuffer* b = nullptr; bool busy = false; int fence = -1; };
  std::mutex m;
  std::vector<Slot> slots;
  ASurfaceControl* sc = nullptr;
  int last = -1;   // ai: the slot the last transaction set (on screen, or about to be)
  ~Ring() {
    for (auto& x : slots) { if (x.fence >= 0) close(x.fence); if (x.b) AHardwareBuffer_release(x.b); }
    if (sc) ASurfaceControl_release(sc);
  }
};
// ai: a transaction's completion: the buffer it replaced (prev) free once its release fence signals
struct Done { std::shared_ptr<Ring> r; int prev; };
void onDone(void* ctx, ASurfaceTransactionStats* st) {
  auto* d = static_cast<Done*>(ctx);
  if (d->prev >= 0) {
    int fence = -1;
    ASurfaceControl** scs = nullptr;
    size_t n = 0;
    ASurfaceTransactionStats_getASurfaceControls(st, &scs, &n);
    for (size_t i = 0; i < n; i++) if (scs[i] == d->r->sc) fence = ASurfaceTransactionStats_getPreviousReleaseFenceFd(st, d->r->sc);
    if (scs) ASurfaceTransactionStats_releaseASurfaceControls(scs);
    std::lock_guard<std::mutex> l(d->r->m);
    auto& x = d->r->slots[d->prev];
    if (x.fence >= 0) close(x.fence);
    x.fence = fence; x.busy = false;
  }
  delete d;
}
// ai: the calls newer than the app's minimum (29), looked up once
using SetFrameRate = void (*)(ASurfaceTransaction*, ASurfaceControl*, float, int8_t);
using SetFrameRateAlways = void (*)(ASurfaceTransaction*, ASurfaceControl*, float, int8_t, int8_t);
using SetFrameTimeline = void (*)(ASurfaceTransaction*, int64_t);
template <class F> F sym(const char* n) { return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, n)); }
SetFrameRate setFrameRate() { static auto f = sym<SetFrameRate>("ASurfaceTransaction_setFrameRate"); return f; }
SetFrameRateAlways setFrameRateAlways() { static auto f = sym<SetFrameRateAlways>("ASurfaceTransaction_setFrameRateWithChangeStrategy"); return f; }
SetFrameTimeline setFrameTimeline() { static auto f = sym<SetFrameTimeline>("ASurfaceTransaction_setFrameTimeline"); return f; }


struct Tx {
  std::unique_ptr<lizard::Sender> s;
  const uint8_t* map = nullptr;
  size_t mapLen = 0;
  ANativeWindow* win = nullptr;
  jobject surface = nullptr;   // ai: the global ref win was made from
  int geom = 0, geomW = 0;     // ai: the frame size the window's buffers were set to (height, width)
  int fps = 0;                 // ai: the rate configured, voted on the ring's surface
  std::shared_ptr<Ring> ring;  // ai: the vsync-locked present's surface and buffers (none on the window path)
  int ringSide = 0, ringW = 0, ringFps = 0, dstW = 0, dstH = 0;
  bool ringFailed = false;     // ai: no child surface or buffers here: the window path for this sender
  std::string error;
};
Tx* tx(jlong h) { return reinterpret_cast<Tx*>(h); }
void dropRing(Tx* t) {
  if (!t->ring) return;
  ASurfaceTransaction* tr = ASurfaceTransaction_create();
  ASurfaceTransaction_reparent(tr, t->ring->sc, nullptr);
  ASurfaceTransaction_apply(tr);
  ASurfaceTransaction_delete(tr);
  t->ring.reset();   // ai: the buffers and the surface go with the last completion still owed
  t->ringSide = t->ringFps = t->dstW = t->dstH = 0;
}
void dropWindow(JNIEnv* e, Tx* t) {
  dropRing(t);
  if (t->win) ANativeWindow_release(t->win);
  if (t->surface) e->DeleteGlobalRef(t->surface);
  t->win = nullptr; t->surface = nullptr; t->geom = 0; t->geomW = 0;
}
// ai: The next painted frame (width x side: two codes side by side are wider than tall, 2026-10-05) for vsync `vsync`,
// ai: scaled into the view's w x h keeping its shape, centred: true when one was posted. False when none is painted yet,
// ai: or every buffer is still the compositor's (the frame waits for the next vsync).
bool presentVsynced(Tx* t, int side, int width, int64_t vsync, int w, int h) {
  if (!t->ring || t->ringSide != side || t->ringW != width) {
    dropRing(t);
    auto r = std::make_shared<Ring>();
    r->sc = ASurfaceControl_createFromWindow(t->win, "lizard-send");
    if (!r->sc) { t->ringFailed = true; return false; }
    AHardwareBuffer_Desc d{};
    d.width = static_cast<uint32_t>(width); d.height = static_cast<uint32_t>(side); d.layers = 1;
    d.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    d.usage = AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY;
    r->slots.resize(4);
    for (auto& x : r->slots) if (AHardwareBuffer_allocate(&d, &x.b)) { x.b = nullptr; t->ringFailed = true; return false; }
    t->ring = r; t->ringSide = side; t->ringW = width; t->dstW = t->dstH = 0;
  }
  if (!t->s->ready()) return false;
  Ring& r = *t->ring;
  int k = -1, fence = -1;
  {
    std::lock_guard<std::mutex> l(r.m);
    for (int i = 0; i < static_cast<int>(r.slots.size()) && k < 0; i++) {
      auto& x = r.slots[i];
      if (x.busy || i == r.last) continue;
      if (x.fence >= 0) {
        pollfd p{x.fence, POLLIN, 0};
        if (poll(&p, 1, 0) != 1) continue;   // ai: still being read: another slot, or the next vsync
        close(x.fence); x.fence = -1;
      }
      k = i;
    }
    if (k < 0) return false;
    r.slots[k].busy = true;
  }
  AHardwareBuffer* b = r.slots[k].b;
  AHardwareBuffer_Desc d{};
  AHardwareBuffer_describe(b, &d);
  void* px = nullptr;
  bool ok = !AHardwareBuffer_lock(b, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &px) && t->s->take(static_cast<uint8_t*>(px), static_cast<int>(d.stride) * 4);
  if (px) AHardwareBuffer_unlock(b, &fence);
  if (!ok) { if (fence >= 0) close(fence); std::lock_guard<std::mutex> l(r.m); r.slots[k].busy = false; return false; }
  ASurfaceTransaction* tr = ASurfaceTransaction_create();
  ASurfaceTransaction_setBuffer(tr, r.sc, b, fence);   // ai: the transaction owns the acquire fence
  if (t->dstW != w || t->dstH != h) {
    // ai: the frame's shape kept: the view is laid out at about the frame's shape (Send.kt CodeBox), not to the pixel
    const double sc = std::min(static_cast<double>(w) / width, static_cast<double>(h) / side);
    const int dw = static_cast<int>(width * sc + 0.5), dh = static_cast<int>(side * sc + 0.5), x0 = (w - dw) / 2, y0 = (h - dh) / 2;
    const ARect src{0, 0, width, side}, dst{x0, y0, x0 + dw, y0 + dh};
    ASurfaceTransaction_setGeometry(tr, r.sc, src, dst, 0);
    ASurfaceTransaction_setBufferTransparency(tr, r.sc, ASURFACE_TRANSACTION_TRANSPARENCY_OPAQUE);
    ASurfaceTransaction_setVisibility(tr, r.sc, ASURFACE_TRANSACTION_VISIBILITY_SHOW);
    t->dstW = w; t->dstH = h;
  }
  if (t->ringFps != t->fps && t->fps > 0) {
    // ai: 1: ANATIVEWINDOW_CHANGE_FRAME_RATE_ALWAYS, whose name the headers fence at Android 12 (the call is looked up)
    if (auto f = setFrameRateAlways()) f(tr, r.sc, static_cast<float>(t->fps), ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE, 1);
    else if (auto g = setFrameRate()) g(tr, r.sc, static_cast<float>(t->fps), ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE);
    t->ringFps = t->fps;
  }
  if (auto f = setFrameTimeline(); f && vsync > 0) f(tr, vsync);
  {
    std::lock_guard<std::mutex> l(r.m);
    ASurfaceTransaction_setOnComplete(tr, new Done{t->ring, r.last}, onDone);
    r.last = k;
  }
  ASurfaceTransaction_apply(tr);
  ASurfaceTransaction_delete(tr);
  return true;
}
}  // namespace

// ai: path: the file to send ("" for the test stream); name and type as its header carries them. 0 with the reason in
// ai: txError(0) where it cannot go.
static std::string gTxError;
extern "C" JNIEXPORT jlong JNICALL Java_dev_lizard_receiver_Native_txCreate(JNIEnv* e, jclass, jstring path, jstring name, jstring type) {
  auto t = std::make_unique<Tx>();
  const std::string p = str(e, path);
  try {
    if (!p.empty()) {
      const int fd = open(p.c_str(), O_RDONLY);
      struct stat st{};
      if (fd < 0 || fstat(fd, &st)) { if (fd >= 0) close(fd); gTxError = "the file could not be read"; return 0; }
      t->mapLen = static_cast<size_t>(st.st_size);
      if (t->mapLen) {
        void* m = mmap(nullptr, t->mapLen, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); gTxError = "the file could not be mapped"; return 0; }
        t->map = static_cast<const uint8_t*>(m);
      }
      close(fd);
    }
    static const uint8_t empty = 0;
    t->s = std::make_unique<lizard::Sender>(p.empty() ? nullptr : (t->map ? t->map : &empty), t->mapLen, str(e, name), str(e, type));
  } catch (const std::exception& x) {
    if (t->map) munmap(const_cast<uint8_t*>(t->map), t->mapLen);
    gTxError = x.what();
    return 0;
  }
  return reinterpret_cast<jlong>(t.release());
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txError(JNIEnv* e, jclass) { return e->NewStringUTF(gTxError.c_str()); }

// ai: The format to paint (n, sub-channels, span, the rate the word states, painters) and where (painter: 0 the CPU, 1
// ai: the GPU, 2 auto; assets: the app's generated tree, the GPU's kernels and tables): "" or the codec's refusal.
// ai: codes: 1, or 2 side by side for a receiver's 2:1 crop, gap modules apart (2026-10-05; the app sent one before)
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txConfigure(JNIEnv* e, jclass, jlong h, jint n, jint subch, jint span, jint fps, jint threads, jint painter, jstring assets, jint codes, jint gap) {
  lizard::TxFormat f;
  f.n = n; f.subch = subch; f.span = span; f.fps = fps; f.threads = threads; f.painter = painter; f.assets = str(e, assets); f.codes = codes; f.gap = gap;
  tx(h)->fps = fps;
  return e->NewStringUTF(tx(h)->s->configure(f).c_str());
}

// ai: The GPU painter made ahead (its device and pipelines), off the main thread: "" or why there is none.
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txPrepare(JNIEnv* e, jclass, jlong h, jstring assets) {
  return e->NewStringUTF(tx(h)->s->prepareGpu(str(e, assets)).c_str());
}

extern "C" JNIEXPORT jint JNICALL Java_dev_lizard_receiver_Native_txSide(JNIEnv*, jclass, jlong h) { return tx(h)->s->side(); }

// ai: The next frame onto the surface, if it is painted: true when one was posted. On the main thread, at the display's
// ai: pace (Send.kt's Choreographer); a surface other than the last one is taken up afresh. vsync: the frame timeline's
// ai: id to present at (Android 13 on; the vsync-locked present, Ring above, into the view's w x h), or 0 for the
// ai: window's own queue as before.
extern "C" JNIEXPORT jboolean JNICALL Java_dev_lizard_receiver_Native_txPresent(JNIEnv* e, jclass, jlong h, jobject surface, jlong vsync, jint w, jint hh) {
  Tx* t = tx(h);
  const int side = t->s->side(), width = t->s->width();
  if (!surface || !side || !width) return JNI_FALSE;
  if (!t->surface || !e->IsSameObject(t->surface, surface)) {
    dropWindow(e, t);
    t->win = ANativeWindow_fromSurface(e, surface);
    if (!t->win) return JNI_FALSE;
    t->surface = e->NewGlobalRef(surface);
  }
  if (vsync > 0 && !t->ringFailed && w > 0 && hh > 0) return presentVsynced(t, side, width, vsync, w, hh) ? JNI_TRUE : JNI_FALSE;
  // ai: the window's own queue: its buffers the frame's size, stretched to the view, which is laid out at about its shape
  if (t->geom != side || t->geomW != width) {
    if (ANativeWindow_setBuffersGeometry(t->win, width, side, WINDOW_FORMAT_RGBA_8888)) return JNI_FALSE;
    t->geom = side; t->geomW = width;
  }
  // ai: a buffer is locked only for a frame in hand: one posted unfilled would show an older frame of the window's queue
  if (!t->s->ready()) return JNI_FALSE;
  ANativeWindow_Buffer b;
  if (ANativeWindow_lock(t->win, &b, nullptr)) return JNI_FALSE;
  const bool ok = b.width == width && b.height == side && t->s->take(static_cast<uint8_t*>(b.bits), b.stride * 4);
  ANativeWindow_unlockAndPost(t->win);
  return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txStats(JNIEnv* e, jclass, jlong h) { return e->NewStringUTF(tx(h)->s->stats().c_str()); }

extern "C" JNIEXPORT void JNICALL Java_dev_lizard_receiver_Native_txDestroy(JNIEnv* e, jclass, jlong h) {
  Tx* t = tx(h);
  if (!t) return;
  dropWindow(e, t);
  t->s.reset();
  if (t->map) munmap(const_cast<uint8_t*>(t->map), t->mapLen);
  delete t;
}
