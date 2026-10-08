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
#include <condition_variable>
#include <deque>
#include <fstream>
#include <thread>
#include <linux/sync_file.h>
#include <sys/ioctl.h>

#include "wg.h"

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

// ai: The core's LIZ_* switches for a run driven over adb: `adb shell setprop debug.lizard.env "LIZ_PARTS=0 ..."`, read
// ai: as each receiver and each sender is made; the names set the last time are unset first, so a property cleared
// ai: takes effect at the next one (2026-10-07; they stayed set before).
static void applyDebugEnv() {
  static std::vector<std::string> set;
  for (const auto& k : set) unsetenv(k.c_str());
  set.clear();
  char v[PROP_VALUE_MAX] = "";
  __system_property_get("debug.lizard.env", v);
  std::string all(v);
  for (size_t a = 0; a < all.size();) {
    size_t b = all.find(' ', a);
    if (b == std::string::npos) b = all.size();
    const std::string kv = all.substr(a, b - a);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && eq > 0) { setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1); set.push_back(kv.substr(0, eq)); __android_log_print(ANDROID_LOG_INFO, "lizard", "env %s", kv.c_str()); }
    a = b + 1;
  }
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
  applyDebugEnv();
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

// ai: when a sync fence signalled (the kernel's sync_file info, ns on CLOCK_MONOTONIC), -1 while it has not
int64_t fenceSignalledAt(int fd) {
  if (fd < 0) return -1;
  pollfd p{fd, POLLIN, 0};
  if (poll(&p, 1, 0) != 1) return -1;
  sync_file_info info{};
  if (ioctl(fd, SYNC_IOC_FILE_INFO, &info) || !info.num_fences) return 0;
  std::vector<sync_fence_info> fs(info.num_fences);
  info.sync_fence_info = reinterpret_cast<uint64_t>(fs.data());
  if (ioctl(fd, SYNC_IOC_FILE_INFO, &info)) return 0;
  int64_t t = 0;
  for (const auto& f : fs) t = std::max<int64_t>(t, static_cast<int64_t>(f.timestamp_ns));
  return t;
}

// ai: The GPU-filled ring (2026-10-07: "generate ahead, buffer generously, present on an exact fixed-refresh schedule";
// ai: GPU only). The painter's frames stay on its device (sender.h ring()); a filler thread expands each into one of the
// ai: ring's AHardwareBuffers (RGBA8, imported into the painter's device as a storage image, core/tx/expand.comp) as
// ai: soon as one is free, so filled buffers wait ahead of the screen; at a vsync where a picture is due the main
// ai: thread posts the next one for that frame timeline (setFrameTimeline), with no fence (its GPU work is complete)
// ai: and no copy. A buffer is free again once the compositor's release fence for it has signalled. Each transaction's
// ai: present fence says when it reached the screen: the tally (posted, held by vsyncs, late, behind) is the
// ai: schedule's truth, as present_wait's is on the desktop. Whether a storage image can live on an AHardwareBuffer
// ai: here is probed (probeStorage); where not, the CPU copy path below (Ring) carries on, as before.
struct GpuRing {
  struct Slot {
    AHardwareBuffer* b = nullptr;
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    int fence = -1;         // ai: the compositor's release fence, once posted and replaced
    bool posted = false;    // ai: the compositor's (queued or on screen) until its release fence comes
    bool filled = false;    // ai: a frame expanded in, not yet posted
    uint64_t seq = 0;
    VkBuffer bound = VK_NULL_HANDLE;   // ai: the ring buffer the set binds
  };
  wg::Device* dev = nullptr;
  int w = 0, h = 0;
  std::mutex m;
  std::condition_variable cv;
  std::vector<Slot> slots;
  ASurfaceControl* sc = nullptr;
  int last = -1;
  bool stop = false;
  std::thread filler;
  // ai: the expand pipeline on the painter's device, the filler's own pool and fence
  VkCommandPool pool = VK_NULL_HANDLE;
  VkDescriptorPool dpool = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout pl = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  // ai: the tally: transactions posted, each with the vsync it was posted for and its present fence until that
  // ai: signals; held (vsyncs a picture stayed, by the present times), late (shown after its vsync by over half a
  // ai: vsync), behind (a due vsync with no frame filled)
  struct Posted { int64_t expected; int fd; };
  std::deque<Posted> pending;
  int64_t lastShown = -1, vsyncNs = 0;
  uint64_t posts = 0, late = 0, behind = 0, shownN = 0;
  std::map<int, uint64_t> held;
  std::string error;
  ~GpuRing() {
    if (filler.joinable()) {
      { std::lock_guard<std::mutex> l(m); stop = true; }
      cv.notify_all();
      filler.join();
    }
    if (dev) {
      vkDeviceWaitIdle(dev->dev);
      for (auto& x : slots) {
        if (x.view) vkDestroyImageView(dev->dev, x.view, nullptr);
        if (x.img) vkDestroyImage(dev->dev, x.img, nullptr);
        if (x.mem) vkFreeMemory(dev->dev, x.mem, nullptr);
      }
      if (fence) vkDestroyFence(dev->dev, fence, nullptr);
      if (dpool) vkDestroyDescriptorPool(dev->dev, dpool, nullptr);
      if (pipe) vkDestroyPipeline(dev->dev, pipe, nullptr);
      if (pl) vkDestroyPipelineLayout(dev->dev, pl, nullptr);
      if (dsl) vkDestroyDescriptorSetLayout(dev->dev, dsl, nullptr);
      if (pool) vkDestroyCommandPool(dev->dev, pool, nullptr);
    }
    for (auto& x : slots) { if (x.fence >= 0) close(x.fence); if (x.b) AHardwareBuffer_release(x.b); }
    for (auto& p : pending) if (p.fd >= 0) close(p.fd);
    if (sc) ASurfaceControl_release(sc);
  }
};

// ai: which AHardwareBuffer usage a storage image of RGBA8 needs on this device, or false where none can
bool probeStorage(wg::Device& d, uint64_t* usage) {
  VkPhysicalDeviceExternalImageFormatInfo ext{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
  ext.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
  VkPhysicalDeviceImageFormatInfo2 info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
  info.pNext = &ext;
  info.format = VK_FORMAT_R8G8B8A8_UNORM;
  info.type = VK_IMAGE_TYPE_2D;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
  VkAndroidHardwareBufferUsageANDROID au{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID};
  VkExternalImageFormatProperties ep{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
  ep.pNext = &au;
  VkImageFormatProperties2 p2{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
  p2.pNext = &ep;
  if (!d.features.ahb || vkGetPhysicalDeviceImageFormatProperties2(d.phys, &info, &p2) != VK_SUCCESS) return false;
  if (!(ep.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) return false;
  *usage = au.androidHardwareBufferUsage;
  return true;
}

std::vector<char> readAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}


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
  std::shared_ptr<GpuRing> gring;   // ai: the GPU-filled ring (GpuRing), where the device takes storage images on hardware buffers
  bool gpuExpand = false;      // ai: probeStorage's answer, at txPrepare
  uint64_t ahbUsage = 0;
  std::string assets;
  std::string error;
};
Tx* tx(jlong h) { return reinterpret_cast<Tx*>(h); }
void dropRing(Tx* t) {
  if (t->gring) {
    ASurfaceTransaction* tr = ASurfaceTransaction_create();
    ASurfaceTransaction_reparent(tr, t->gring->sc, nullptr);
    ASurfaceTransaction_apply(tr);
    ASurfaceTransaction_delete(tr);
    t->gring.reset();
  }
  if (t->ring) {
    ASurfaceTransaction* tr = ASurfaceTransaction_create();
    ASurfaceTransaction_reparent(tr, t->ring->sc, nullptr);
    ASurfaceTransaction_apply(tr);
    ASurfaceTransaction_delete(tr);
    t->ring.reset();   // ai: the buffers and the surface go with the last completion still owed
  }
  t->ringSide = t->ringW = t->ringFps = t->dstW = t->dstH = 0;
}

// ai: the GPU ring's filler: a free buffer (not the compositor's, its release fence signalled), the next painted
// ai: frame's slot from the sender, the expand recorded and run on the painter's queue (under the device's lock, as
// ai: the painter's own submits), waited, then the buffer filled for the main thread to post
void fillRing(std::shared_ptr<GpuRing> r, Tx* t) {
  wg::Device& d = *r->dev;
  for (;;) {
    int k = -1;
    {
      std::unique_lock<std::mutex> l(r->m);
      for (;;) {
        if (r->stop) return;
        for (int i = 0; i < static_cast<int>(r->slots.size()) && k < 0; i++) {
          auto& x = r->slots[i];
          if (x.posted || x.filled) continue;
          if (x.fence >= 0) {
            pollfd p{x.fence, POLLIN, 0};
            if (poll(&p, 1, 0) != 1) continue;
            close(x.fence);
            x.fence = -1;
          }
          k = i;
        }
        if (k >= 0) break;
        r->cv.wait_for(l, std::chrono::milliseconds(2));
      }
    }
    uint64_t seq = 0;
    int slot = -1;
    while (!t->s->takeSlot(&seq, &slot)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      std::lock_guard<std::mutex> l(r->m);
      if (r->stop) return;
    }
    auto ring = t->s->ring();
    const uint32_t FS = t->s->ringFrameWords(), RW = t->s->ringRowWords();
    if (!ring || t->s->side() != r->h || t->s->width() != r->w) { std::lock_guard<std::mutex> l(r->m); r->error = "the frame's size changed under the ring"; return; }
    auto& x = r->slots[k];
    if (x.bound != ring->buf) {
      VkDescriptorBufferInfo bi{ring->buf, 0, VK_WHOLE_SIZE};
      VkDescriptorImageInfo ii{VK_NULL_HANDLE, x.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet w[2]{};
      w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[0].dstSet = w[1].dstSet = x.set;
      w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi;
      w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &ii;
      vkUpdateDescriptorSets(d.dev, 2, w, 0, nullptr);
      x.bound = ring->buf;
    }
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = r->pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    if (vkAllocateCommandBuffers(d.dev, &cai, &cb) != VK_SUCCESS) { std::lock_guard<std::mutex> l(r->m); r->error = "no command buffer for the expand"; return; }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    // ai: the painter's frame visible to the compute stage (its submit's fence was waited before it was marked ready)
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = 0; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    // ai: the buffer acquired from the compositor (foreign), written, released back to it
    const uint32_t foreign = d.features.foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
    VkImageMemoryBarrier acq{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    acq.srcAccessMask = 0; acq.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    acq.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; acq.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    acq.srcQueueFamilyIndex = foreign; acq.dstQueueFamilyIndex = d.family;
    acq.image = x.img; acq.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->pl, 0, 1, &x.set, 0, nullptr);
    const uint32_t pc[4] = {static_cast<uint32_t>(slot) * FS, RW, static_cast<uint32_t>(r->w), static_cast<uint32_t>(r->h)};
    vkCmdPushConstants(cb, r->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cb, (static_cast<uint32_t>(r->w) + 15) / 16, (static_cast<uint32_t>(r->h) + 15) / 16, 1);
    VkImageMemoryBarrier rel = acq;
    rel.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; rel.dstAccessMask = 0;
    rel.oldLayout = VK_IMAGE_LAYOUT_GENERAL; rel.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    rel.srcQueueFamilyIndex = d.family; rel.dstQueueFamilyIndex = foreign;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &rel);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkResult sr;
    {
      std::lock_guard<std::mutex> l(d.mu);
      vkResetFences(d.dev, 1, &r->fence);
      sr = vkQueueSubmit(d.queue, 1, &si, r->fence);
    }
    if (sr == VK_SUCCESS) vkWaitForFences(d.dev, 1, &r->fence, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(d.dev, r->pool, 1, &cb);
    std::lock_guard<std::mutex> l(r->m);
    if (sr != VK_SUCCESS) { r->error = "the expand's submit failed (VkResult " + std::to_string(static_cast<int>(sr)) + ")"; return; }
    x.filled = true;
    x.seq = seq;
  }
}

// ai: a transaction's completion on the GPU ring: the buffer it replaced free once its release fence signals, and
// ai: the present fences of the transactions before it read for when each reached the screen (the tally)
struct GDone { std::shared_ptr<GpuRing> r; int prev; int64_t expected; };
void onGDone(void* ctx, ASurfaceTransactionStats* st) {
  auto* d = static_cast<GDone*>(ctx);
  GpuRing& r = *d->r;
  int fence = -1;
  if (d->prev >= 0) {
    ASurfaceControl** scs = nullptr;
    size_t n = 0;
    ASurfaceTransactionStats_getASurfaceControls(st, &scs, &n);
    for (size_t i = 0; i < n; i++) if (scs[i] == r.sc) fence = ASurfaceTransactionStats_getPreviousReleaseFenceFd(st, r.sc);
    if (scs) ASurfaceTransactionStats_releaseASurfaceControls(scs);
  }
  const int present = ASurfaceTransactionStats_getPresentFenceFd(st);
  std::lock_guard<std::mutex> l(r.m);
  if (d->prev >= 0) {
    auto& x = r.slots[d->prev];
    if (x.fence >= 0) close(x.fence);
    x.fence = fence;
    x.posted = false;
  }
  r.pending.push_back({d->expected, present});
  // ai: the present times known so far, oldest first; one not yet signalled holds the rest
  while (!r.pending.empty()) {
    auto& p = r.pending.front();
    const int64_t at = fenceSignalledAt(p.fd);
    if (at < 0 && p.fd >= 0) break;
    if (p.fd >= 0) close(p.fd);
    if (at > 0) {
      if (r.vsyncNs > 0 && at > p.expected + r.vsyncNs / 2) r.late++;
      if (r.lastShown > 0 && r.vsyncNs > 0) r.held[static_cast<int>(std::lround(static_cast<double>(at - r.lastShown) / r.vsyncNs))]++;
      r.lastShown = at;
      r.shownN++;
    }
    r.pending.pop_front();
  }
  r.cv.notify_all();
  delete d;
}

// ai: The GPU ring made for the frame's size: the child surface, the buffers (the probe's usage, never CPU-written),
// ai: each imported as a storage image, the expand pipeline, the filler thread. False with the reason in t->error.
bool makeGpuRing(Tx* t, int side, int width) {
  wg::Device* dev = t->s->device();
  if (!dev) { t->error = "no device for the ring"; return false; }
  auto r = std::make_shared<GpuRing>();
  r->dev = dev;
  r->w = width;
  r->h = side;
  r->sc = ASurfaceControl_createFromWindow(t->win, "lizard-send");
  if (!r->sc) { t->error = "no child surface"; return false; }
  const uint64_t frameBytes = static_cast<uint64_t>(width) * side * 4;
  const int depth = std::max(4, std::min(12, static_cast<int>((96ull << 20) / std::max<uint64_t>(1, frameBytes))));
  AHardwareBuffer_Desc d{};
  d.width = static_cast<uint32_t>(width); d.height = static_cast<uint32_t>(side); d.layers = 1;
  d.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
  d.usage = t->ahbUsage | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY;
  r->slots.resize(depth);
  for (auto& x : r->slots) if (AHardwareBuffer_allocate(&d, &x.b)) { x.b = nullptr; t->error = "no hardware buffers for the ring"; return false; }
  try {
    const auto code = readAll(t->assets + "/spv/send_expand.spv");
    if (code.empty() || code.size() % 4) throw std::runtime_error("no expand kernel in the assets");
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm;
    wg::check(vkCreateShaderModule(dev->dev, &smi, nullptr, &sm), "vkCreateShaderModule (expand)");
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2; dli.pBindings = b;
    wg::check(vkCreateDescriptorSetLayout(dev->dev, &dli, nullptr, &r->dsl), "vkCreateDescriptorSetLayout (expand)");
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &r->dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    wg::check(vkCreatePipelineLayout(dev->dev, &pli, nullptr, &r->pl), "vkCreatePipelineLayout (expand)");
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = r->pl;
    const VkResult pr = vkCreateComputePipelines(dev->dev, dev->cache, 1, &cpi, nullptr, &r->pipe);
    vkDestroyShaderModule(dev->dev, sm, nullptr);
    wg::check(pr, "vkCreateComputePipelines (expand)");
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 16; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
    wg::check(vkCreateDescriptorPool(dev->dev, &dpi, nullptr, &r->dpool), "vkCreateDescriptorPool (expand)");
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = dev->family;
    wg::check(vkCreateCommandPool(dev->dev, &cpci, nullptr, &r->pool), "vkCreateCommandPool (expand)");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    wg::check(vkCreateFence(dev->dev, &fci, nullptr, &r->fence), "vkCreateFence (expand)");
    for (auto& x : r->slots) {
      VkAndroidHardwareBufferPropertiesANDROID pr2{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
      wg::check(vkGetAndroidHardwareBufferPropertiesANDROID(dev->dev, x.b, &pr2), "vkGetAndroidHardwareBufferPropertiesANDROID (ring)");
      VkExternalMemoryImageCreateInfo emi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
      VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      ici.pNext = &emi;
      ici.imageType = VK_IMAGE_TYPE_2D;
      ici.format = VK_FORMAT_R8G8B8A8_UNORM;
      ici.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(side), 1};
      ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
      ici.tiling = VK_IMAGE_TILING_OPTIMAL;
      ici.usage = VK_IMAGE_USAGE_STORAGE_BIT;
      ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      wg::check(vkCreateImage(dev->dev, &ici, nullptr, &x.img), "vkCreateImage (ring)");
      VkImportAndroidHardwareBufferInfoANDROID imp{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
      imp.buffer = x.b;
      VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
      ded.pNext = &imp; ded.image = x.img;
      VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      mai.pNext = &ded;
      mai.allocationSize = pr2.allocationSize;
      mai.memoryTypeIndex = dev->memoryType(pr2.memoryTypeBits, 0);
      wg::check(vkAllocateMemory(dev->dev, &mai, nullptr, &x.mem), "vkAllocateMemory (ring import)");
      wg::check(vkBindImageMemory(dev->dev, x.img, x.mem, 0), "vkBindImageMemory (ring)");
      VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vci.image = x.img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_R8G8B8A8_UNORM;
      vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      wg::check(vkCreateImageView(dev->dev, &vci, nullptr, &x.view), "vkCreateImageView (ring)");
      VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      dsa.descriptorPool = r->dpool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &r->dsl;
      wg::check(vkAllocateDescriptorSets(dev->dev, &dsa, &x.set), "vkAllocateDescriptorSets (ring)");
    }
  } catch (const std::exception& e) {
    t->error = e.what();
    return false;
  }
  r->filler = std::thread([r, t] { fillRing(r, t); });
  t->gring = r;
  t->ringSide = side; t->ringW = width; t->dstW = t->dstH = 0;
  __android_log_print(ANDROID_LOG_INFO, "lizard", "send: a GPU ring of %d buffers (%d x %d), filled ahead on %s", depth, width, side, dev->name.c_str());
  return true;
}

// ai: The next filled buffer posted for the vsync `vsync` (its frame timeline id), expected on screen at `expected`
// ai: ns; false, counted behind, where none is filled (the picture on screen stays).
bool postGpu(Tx* t, int side, int width, int64_t vsync, int64_t expected, int64_t vsyncNs, int w, int h) {
  if (!t->gring || t->ringSide != side || t->ringW != width) {
    dropRing(t);
    if (!makeGpuRing(t, side, width)) { t->ringFailed = true; return false; }
  }
  GpuRing& r = *t->gring;
  int k = -1;
  {
    std::lock_guard<std::mutex> l(r.m);
    if (!r.error.empty()) { t->error = r.error; t->ringFailed = true; return false; }
    r.vsyncNs = vsyncNs;
    uint64_t best = ~0ull;
    for (int i = 0; i < static_cast<int>(r.slots.size()); i++) if (r.slots[i].filled && r.slots[i].seq < best) { best = r.slots[i].seq; k = i; }
    if (k < 0) { r.behind++; return false; }
    r.slots[k].filled = false;
    r.slots[k].posted = true;
  }
  ASurfaceTransaction* tr = ASurfaceTransaction_create();
  ASurfaceTransaction_setBuffer(tr, r.sc, r.slots[k].b, -1);   // ai: the expand was waited: no acquire fence
  if (t->dstW != w || t->dstH != h) {
    const double sc = std::min(static_cast<double>(w) / width, static_cast<double>(h) / side);
    const int dw = static_cast<int>(width * sc + 0.5), dh = static_cast<int>(side * sc + 0.5), x0 = (w - dw) / 2, y0 = (h - dh) / 2;
    const ARect src{0, 0, width, side}, dst{x0, y0, x0 + dw, y0 + dh};
    ASurfaceTransaction_setGeometry(tr, r.sc, src, dst, 0);
    ASurfaceTransaction_setBufferTransparency(tr, r.sc, ASURFACE_TRANSACTION_TRANSPARENCY_OPAQUE);
    ASurfaceTransaction_setVisibility(tr, r.sc, ASURFACE_TRANSACTION_VISIBILITY_SHOW);
    t->dstW = w; t->dstH = h;
  }
  if (t->ringFps != t->fps && t->fps > 0) {
    if (auto f = setFrameRateAlways()) f(tr, r.sc, static_cast<float>(t->fps), ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE, 1);
    else if (auto g = setFrameRate()) g(tr, r.sc, static_cast<float>(t->fps), ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_FIXED_SOURCE);
    t->ringFps = t->fps;
  }
  if (auto f = setFrameTimeline(); f && vsync > 0) f(tr, vsync);
  {
    std::lock_guard<std::mutex> l(r.m);
    ASurfaceTransaction_setOnComplete(tr, new GDone{t->gring, r.last, expected}, onGDone);
    r.last = k;
    r.posts++;
  }
  ASurfaceTransaction_apply(tr);
  ASurfaceTransaction_delete(tr);
  return true;
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
  applyDebugEnv();
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
// ai: vsynced: the frames posted for their vsyncs (Send.kt, Android 13 on): with the GPU ring the painters keep no
// ai: host copies (the frames go from the painter's ring to the compositor's buffers on the device); otherwise take()
// ai: needs them. The ring: a second of frames within 96 MB, 2 slots of margin (an expand is waited before its slot
// ai: is free again).
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txConfigure(JNIEnv* e, jclass, jlong h, jint n, jint subch, jint span, jint fps, jint threads, jint painter, jstring assets, jint codes, jint gap, jboolean vsynced) {
  lizard::TxFormat f;
  f.n = n; f.subch = subch; f.span = span; f.fps = fps; f.threads = threads; f.painter = painter; f.assets = str(e, assets); f.codes = codes; f.gap = gap;
  f.aheadSecs = 1.0;
  f.aheadBytes = 96ull << 20;
  f.margin = 2;
  f.hostFrames = !(tx(h)->gpuExpand && vsynced == JNI_TRUE);
  tx(h)->fps = fps;
  tx(h)->assets = f.assets;
  return e->NewStringUTF(tx(h)->s->configure(f).c_str());
}

// ai: The device and the GPU painter made ahead, off the main thread: "" or why there is none. Then the probe: whether
// ai: this device takes a storage image on a hardware buffer (the GPU ring's way), logged.
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txPrepare(JNIEnv* e, jclass, jlong h, jstring assets) {
  Tx* t = tx(h);
  t->assets = str(e, assets);
  const std::string why = t->s->prepareGpu(t->assets);
  if (wg::Device* d = t->s->device()) {
    t->gpuExpand = probeStorage(*d, &t->ahbUsage);
    __android_log_print(ANDROID_LOG_INFO, "lizard", "send: %s", t->gpuExpand ? "the GPU fills the compositor's buffers (storage images on hardware buffers)" : "no storage images on hardware buffers here: the CPU copies frames in");
  }
  return e->NewStringUTF(why.c_str());
}

// ai: A picture due at the vsync `vsync` (its frame timeline, expected on screen at `expected` ns; `vsyncNs` the
// ai: display's period): the next filled buffer of the GPU ring posted for it (postGpu), or on a device without the
// ai: ring the CPU-copied present as before (presentVsynced). True when one was posted.
extern "C" JNIEXPORT jboolean JNICALL Java_dev_lizard_receiver_Native_txPost(JNIEnv* e, jclass, jlong h, jobject surface, jlong vsync, jlong expected, jlong vsyncNs, jint w, jint hh) {
  Tx* t = tx(h);
  const int side = t->s->side(), width = t->s->width();
  if (!surface || !side || !width || w <= 0 || hh <= 0) return JNI_FALSE;
  if (!t->surface || !e->IsSameObject(t->surface, surface)) {
    dropWindow(e, t);
    t->win = ANativeWindow_fromSurface(e, surface);
    if (!t->win) return JNI_FALSE;
    t->surface = e->NewGlobalRef(surface);
  }
  if (t->ringFailed) return JNI_FALSE;
  if (t->gpuExpand) return postGpu(t, side, width, vsync, expected, vsyncNs, w, hh) ? JNI_TRUE : JNI_FALSE;
  return presentVsynced(t, side, width, vsync, w, hh) ? JNI_TRUE : JNI_FALSE;
}

// ai: The GPU ring's tally since the last call, JSON: posted (transactions), shown (present fences read), held
// ai: ({vsyncs: pictures}), late, behind, filled (buffers waiting), slots, gpu (the ring in use), error.
extern "C" JNIEXPORT jstring JNICALL Java_dev_lizard_receiver_Native_txPostStats(JNIEnv* e, jclass, jlong h) {
  Tx* t = tx(h);
  std::string o = "{\"gpu\":" + std::string(t->gring ? "true" : "false");
  if (t->gring) {
    GpuRing& r = *t->gring;
    std::lock_guard<std::mutex> l(r.m);
    int filled = 0;
    for (auto& x : r.slots) if (x.filled) filled++;
    o += ",\"posted\":" + std::to_string(r.posts) + ",\"shown\":" + std::to_string(r.shownN) + ",\"late\":" + std::to_string(r.late) + ",\"behind\":" + std::to_string(r.behind) +
         ",\"filled\":" + std::to_string(filled) + ",\"slots\":" + std::to_string(r.slots.size()) + ",\"held\":{";
    bool first = true;
    for (auto& [k, n] : r.held) { o += std::string(first ? "" : ",") + "\"" + std::to_string(k) + "\":" + std::to_string(n); first = false; }
    o += "},\"error\":\"" + r.error + "\"";
    r.posts = r.shownN = r.late = r.behind = 0;
    r.held.clear();
  }
  return e->NewStringUTF((o + "}").c_str());
}

extern "C" JNIEXPORT jint JNICALL Java_dev_lizard_receiver_Native_txSide(JNIEnv*, jclass, jlong h) { return tx(h)->s->side(); }
// ai: The blocks a frame of subch sub-channels carries under the format's rate profile (src/focus.h focus_blocks_for),
// ai: for the Blocks slider's figures before a configure
extern "C" JNIEXPORT jint JNICALL Java_dev_lizard_receiver_Native_blocksFor(JNIEnv*, jclass, jint subch) { return focus_blocks_for(subch); }

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
