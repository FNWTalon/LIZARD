// ai: lizard_fakegv (2026-10-25; on today's cam/scene.h since 2026-11-01): a GigE Vision camera on loopback for the
// ai: receiver's tests (cam_check aravis), Aravis's own ArvGvFakeCamera (its GVCP and GVSP over real sockets,
// ai: AcquisitionStart and Stop, TriggerMode honoured) with a fill of ours, so the receiver loads Aravis at run time
// ai: and sees an ordinary GigE camera.
// ai:   lizard_fakegv [--serial LZ01] [--interface 127.0.0.1] [--tick 10000] [--scene "<cam/scene.h spec>"]
// ai:                 [--start-phase F]
// ai: The spec is scene.h parse's (blocks 4 to 64, 30 by default, 1 to 3 refused; ring in cells 64, 128, 256 or auto;
// ai: w and h the sensor's; the rest of its keys). Its device id is Aravis-Fake-<serial>, printed as
// ai: "ready Aravis-Fake-<serial>" on stdout once it serves; SIGINT or SIGTERM ends it cleanly (exit 0), and so does the
// ai: end of the process that started it (POSIX), so a check that dies leaves no camera holding the port. Only one fake
// ai: GigE camera can bind <interface>:3956 at a time (the control socket takes no reuse, and Aravis's fake runs deaf
// ai: where its bind fails), so the port is tried first and a taken one ends it (exit 1).
// ai: Free run: the camera's own AcquisitionFrameRate paces it (whole us periods on CLOCK_REALTIME's grid, the fill
// ai: and the frame's packets sent on the fake's one thread). A software trigger (TriggerMode On, TriggerSource
// ai: Software: the phase lock's lever, cam/aravis.cpp) is answered at the fake's tick, --tick Hz
// ai: (arv_fake_camera_set_trigger_frequency; Aravis's own 25 Hz would answer a trigger up to 40 ms late and collapse
// ai: those between two ticks into one frame): at 10 kHz within 0.1 ms, a core's half spent on the tick's spin.
// ai: The sensor and the region are the scene's optics size (1920 x 1080 without one: sim::Optics' default), written
// ai: into the fake's registers (SensorWidth 0x11c, SensorHeight 0x118, Width 0x100, Height 0x104), and the camera's
// ai: exposure (ExposureTimeAbs, 0x120) the optics' (8 ms by default).
// ai: The fill runs on the fake's thread, the GVCP replies behind it, so it is kept short. It is called at t (steadyNs,
// ai: the fill's entry). With --scene the frame is the simulated screen (scene.h render) of the capture that began at
// ai: t less its exposure and readout in free run (as cam/sim.cpp hands a capture over once both are over), or at t
// ai: where a trigger asked it (the capture the trigger starts, sent at once: before its exposure would have ended),
// ai: at the exposure and gain the camera hands it (ExposureTimeAbs and GainRaw, as the receiver set them, live:
// ai: GainRaw g the optics' gain times 1 + g), the buffer's timestamp that capture's start. --start-phase F (0 to 1,
// ai: with a scene; a test's): at the second triggered capture (the first is the receiver's probe, aravis.cpp, the
// ai: second its schedule's first) the screen's clock is moved, once, by under a picture's time on screen, so that
// ai: capture begins F of the way through the picture on screen: the schedule's phase against the screen the test's
// ai: from its first capture, not the moment the schedule began. Without --scene, a moving gradient at t,
// ai: luma(x, y) = tri((x + y + floor(t / 1 ms)) mod 512) with tri(k) = k for k < 256, else 511 - k (no edge, so a
// ai: Bayer or colour conversion of it stays within a level), its first row starting with a stamp, little-endian:
// ai: "LZGV", t (8 bytes, ns), the exposure in us (4), GainRaw (4), the fill's count from 0 (8); the timestamp t. The
// ai: device clock is so the host's steady clock.
// ai: Pixels: Mono8 and the four Bayer8 layouts take the luma at every site (a grey scene through a colour filter
// ai: array of equal channels), RGB8 the luma in each channel, Mono16 the luma times 257.
#include <arv.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cam/clock.h"
#include "cam/scene.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <arpa/inet.h>
#include <csignal>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

struct Fill {
  ArvFakeCamera* fc = nullptr;
  lizard::sim::Scene* scene = nullptr;
  double startPhase = -1;         // ai: --start-phase, < 0 none
  uint64_t triggered = 0;         // ai: triggered captures so far
  int64_t shift = 0;              // ai: the screen's clock against the camera's (ns, --start-phase's)
  uint32_t w = 0, h = 0;          // ai: the sensor's (the optics')
  double gain = 1;                // ai: the optics' gain, GainRaw 0's
  int64_t readoutNs = 0;          // ai: the optics' rolling readout, 0 for a global shutter
  std::vector<uint8_t> full;      // ai: a render at the sensor's size, where the region is smaller
  std::vector<uint8_t> luma;      // ai: the region's luma, where the pixels are not one byte of luma each
  std::atomic<uint64_t> n{0};     // ai: fills so far
};

std::atomic<bool> gStop{false};   // ai: a stop signal came

void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = uint8_t(v >> (8 * i)); }
void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = uint8_t(v >> (8 * i)); }

// ai: the gradient frame (the header's formula), rows of w; the stamp where the first row holds it
void gradient(uint8_t* o, uint32_t w, uint32_t h, uint32_t stride, int64_t t, uint32_t exposureUs, uint32_t gain,
              uint64_t n) {
  const uint32_t phase = uint32_t((t / 1'000'000) % 512);
  for (uint32_t y = 0; y < h; y++) {
    uint8_t* r = o + size_t(y) * stride;
    const uint32_t k0 = y + phase;
    for (uint32_t x = 0; x < w; x++) {
      const uint32_t k = (k0 + x) & 511;
      r[x] = uint8_t(k < 256 ? k : 511 - k);
    }
  }
  if (w < 28) return;
  std::memcpy(o, "LZGV", 4);
  put64(o + 4, uint64_t(t));
  put32(o + 12, exposureUs);
  put32(o + 16, gain);
  put64(o + 20, n);
}

// ai: GainRaw (0 to 10) as the scene's gain: the optics' own at 0, (1 + g) times it above, as the stock fake's ramp
// ai: scales by 1 + g
double sceneGain(const Fill& f, guint32 g) { return g ? f.gain * (1.0 + g) : 0.0; }

void fill(ArvBuffer* b, void* data, guint32 exposureUs, guint32 gain, ArvPixelFormat pf) {
  const int64_t t = lizard::cam::steadyNs();
  Fill& f = *static_cast<Fill*>(data);
  size_t size = 0;
  auto* d = static_cast<uint8_t*>(const_cast<void*>(arv_buffer_get_data(b, &size)));
  const gint w = arv_buffer_get_image_width(b), h = arv_buffer_get_image_height(b);
  const gint x0 = arv_buffer_get_image_x(b), y0 = arv_buffer_get_image_y(b);
  if (!d || w <= 0 || h <= 0) return;
  const uint32_t W = uint32_t(w), H = uint32_t(h);
  // ai: the fake sends its buffer whole, sized at AcquisitionStart from the registers then (arv_fake_camera_fill_buffer
  // ai: refuses a payload that grew since), so W x H of the format fits
  const bool oneByte = pf == ARV_PIXEL_FORMAT_MONO_8 || pf == ARV_PIXEL_FORMAT_BAYER_BG_8 || pf == ARV_PIXEL_FORMAT_BAYER_GB_8 ||
                       pf == ARV_PIXEL_FORMAT_BAYER_GR_8 || pf == ARV_PIXEL_FORMAT_BAYER_RG_8;
  if (!oneByte && pf != ARV_PIXEL_FORMAT_RGB_8_PACKED && pf != ARV_PIXEL_FORMAT_MONO_16) return;
  uint8_t* luma = d;
  if (!oneByte) {
    f.luma.resize(size_t(W) * H);
    luma = f.luma.data();
  }
  const uint64_t n = f.n.fetch_add(1);
  int64_t stamp = t;
  if (!f.scene) {
    gradient(luma, W, H, W, t, exposureUs, gain, n);
  } else {
    // ai: free run: the capture whose exposure and readout end now; triggered: the one the trigger starts now
    const bool trig = arv_fake_camera_is_in_software_trigger_mode(f.fc);
    if (!trig) stamp = t - int64_t(exposureUs) * 1000 - f.readoutNs;
    if (trig && f.startPhase >= 0 && ++f.triggered == 2) {
      const int64_t k = f.scene->pictureAt(stamp), r0 = f.scene->shownAt(k), r = f.scene->shownAt(k + 1) - r0;
      if (r > 0) f.shift = ((int64_t(f.startPhase * double(r)) - (stamp - r0)) % r + r) % r;
    }
    const int64_t seen = stamp + f.shift;   // ai: the screen's clock
    const double ms = exposureUs / 1000.0, g = sceneGain(f, gain);
    if (W == f.w && H == f.h && x0 == 0 && y0 == 0) {
      f.scene->render(seen, luma, W, ms, g);
    } else {
      // ai: a region smaller than the sensor: the whole frame rendered, the region copied out of it
      f.full.resize(size_t(f.w) * f.h);
      f.scene->render(seen, f.full.data(), f.w, ms, g);
      for (uint32_t y = 0; y < H; y++) {
        const uint32_t sy = std::min(uint32_t(std::max(y0, 0)) + y, f.h - 1);
        for (uint32_t x = 0; x < W; x++) {
          const uint32_t sx = std::min(uint32_t(std::max(x0, 0)) + x, f.w - 1);
          luma[size_t(y) * W + x] = f.full[size_t(sy) * f.w + sx];
        }
      }
    }
  }
  if (pf == ARV_PIXEL_FORMAT_RGB_8_PACKED) {
    for (size_t i = 0, e = size_t(W) * H; i < e; i++) d[3 * i] = d[3 * i + 1] = d[3 * i + 2] = luma[i];
  } else if (pf == ARV_PIXEL_FORMAT_MONO_16) {
    for (size_t i = 0, e = size_t(W) * H; i < e; i++) { d[2 * i] = luma[i]; d[2 * i + 1] = luma[i]; }   // ai: v * 257, little-endian
  }
  arv_buffer_set_timestamp(b, guint64(stamp));
}

int usage() {
  std::fprintf(stderr, "usage: lizard_fakegv [--serial LZ01] [--interface 127.0.0.1] [--tick 10000] [--scene \"<spec>\"] "
                       "[--start-phase F]\n");
  return 2;
}

#if !defined(_WIN32)
// ai: whether <ip>:3956 can be bound (no other fake camera holds it); a name in place of an address is not tried
bool portFree(const std::string& iface, std::string& why) {
  in_addr a{};
  if (inet_pton(AF_INET, iface.c_str(), &a) != 1) return true;
  const int s = socket(AF_INET, SOCK_DGRAM, 0);
  if (s < 0) return true;
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_port = htons(3956);
  sa.sin_addr = a;
  const bool ok = bind(s, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0;
  if (!ok) why = std::strerror(errno);
  close(s);
  return ok;
}
sigset_t gSigs;   // ai: SIGINT and SIGTERM, taken by the waiting thread
#else
BOOL WINAPI onCtrl(DWORD) { gStop = true; return TRUE; }
#endif

}  // namespace

int main(int argc, char** argv) {
#if !defined(_WIN32)
  // ai: SIGINT and SIGTERM blocked before any thread exists (the scene's painters, Aravis's: they inherit it), taken by
  // ai: sigwait below
  sigemptyset(&gSigs);
  sigaddset(&gSigs, SIGINT);
  sigaddset(&gSigs, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &gSigs, nullptr);
#endif
  std::string serial = "LZ01", iface = "127.0.0.1", spec;
  double tick = 10000, startPhase = -1;
  bool haveScene = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (i + 1 >= argc) return usage();
    const std::string v = argv[++i];
    if (a == "--serial") serial = v;
    else if (a == "--interface") iface = v;
    else if (a == "--tick") { tick = std::atof(v.c_str()); if (!(tick > 0)) return usage(); }
    else if (a == "--start-phase") { startPhase = std::atof(v.c_str()); if (!(startPhase >= 0 && startPhase < 1)) return usage(); }
    else if (a == "--scene") { spec = v; haveScene = true; }
    else return usage();
  }
  lizard::sim::Sender s;
  lizard::sim::Display d;
  lizard::sim::Optics o;
  std::unique_ptr<lizard::sim::Scene> scene;
  if (startPhase >= 0 && !haveScene) return usage();
  if (haveScene) {
    std::string why;
    if (!lizard::sim::parse(spec, s, d, o, why)) { std::fprintf(stderr, "lizard_fakegv: --scene: %s\n", why.c_str()); return 2; }
    scene = lizard::sim::Scene::create(s, d, o, 2, why);
    if (!scene) { std::fprintf(stderr, "lizard_fakegv: the scene: %s\n", why.c_str()); return 1; }
  }
#if !defined(_WIN32)
  std::string why;
  if (!portFree(iface, why)) {
    std::fprintf(stderr, "lizard_fakegv: %s:3956 is taken (%s): another fake GigE camera serves there\n", iface.c_str(), why.c_str());
    return 1;
  }
  const pid_t parent = getppid();
#else
  SetConsoleCtrlHandler(onCtrl, TRUE);
#endif
  ArvGvFakeCamera* gv = arv_gv_fake_camera_new_full(iface.c_str(), serial.c_str(), nullptr);
  if (!gv || !arv_gv_fake_camera_is_running(gv)) {
    std::fprintf(stderr, "lizard_fakegv: no fake camera on %s\n", iface.c_str());
    if (gv) g_object_unref(gv);
    return 1;
  }
  ArvFakeCamera* fc = arv_gv_fake_camera_get_fake_camera(gv);
  arv_fake_camera_write_register(fc, ARV_FAKE_CAMERA_REGISTER_SENSOR_WIDTH, o.width);
  arv_fake_camera_write_register(fc, ARV_FAKE_CAMERA_REGISTER_SENSOR_HEIGHT, o.height);
  arv_fake_camera_write_register(fc, ARV_FAKE_CAMERA_REGISTER_WIDTH, o.width);
  arv_fake_camera_write_register(fc, ARV_FAKE_CAMERA_REGISTER_HEIGHT, o.height);
  arv_fake_camera_write_register(fc, ARV_FAKE_CAMERA_REGISTER_EXPOSURE_TIME_US,
                                 guint32(std::min(1e7, std::max(10.0, std::round(o.exposureMs * 1000)))));
  arv_fake_camera_set_trigger_frequency(fc, tick);
  Fill f;
  f.fc = fc;
  f.startPhase = startPhase;
  f.scene = scene.get();
  f.w = o.width;
  f.h = o.height;
  f.gain = o.gain;
  f.readoutNs = o.rolling ? int64_t(o.readoutMs * 1e6) : 0;
  arv_fake_camera_set_fill_pattern(fc, fill, &f);
  std::fprintf(stderr, "lizard_fakegv: %u x %u on %s, trigger tick %g Hz, %s\n", o.width, o.height, iface.c_str(), tick,
               scene ? "the simulated screen" : "the gradient");
  std::printf("ready Aravis-Fake-%s\n", serial.c_str());
  std::fflush(stdout);
#if !defined(_WIN32)
  // ai: a thread of its own waits for the signal (macOS has no sigtimedwait); this one watches it and the starter
  std::thread([] {
    int sig = 0;
    sigwait(&gSigs, &sig);
    gStop = true;
  }).detach();
  while (!gStop && getppid() == parent) std::this_thread::sleep_for(std::chrono::milliseconds(100));
#else
  while (!gStop) Sleep(100);
#endif
  arv_fake_camera_set_fill_pattern(fc, nullptr, nullptr);   // ai: under the fill's lock: no fill of ours after it
  g_object_unref(gv);
  std::fprintf(stderr, "lizard_fakegv: %llu fills\n", static_cast<unsigned long long>(f.n.load()));
  return 0;
}
