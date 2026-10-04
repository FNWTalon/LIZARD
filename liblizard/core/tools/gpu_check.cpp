// ai: The native GPU decoder's checks, on any Vulkan device: the desktop's (the 4090 for
// ai: blocks, the iGPU for times, lavapipe for a small device) and the phone's over adb.
//
//   lizard_gpu_check info                         the device, its features, and the variants it can run
//   lizard_gpu_check build [variant] [B]          every pipeline and buffer of a variant built, timed
//   lizard_gpu_check selftest [variant]           the web self-test's 8 frames (research/build/gpu_selftest_frames) in
//                                                 one batch, each frame's finder, word and blocks against the web's
//                                                 reference at its tolerances (research/build/gpu_selftest_ref.json, or
//                                                 _int8.json for an int8 variant); exit 1 on a difference
//   lizard_gpu_check same vote <variant> <spv>    an exact change to a kernel, held to the kernel before it on one
//                                                 state: the self-test's batch, then VOTE alone on that lane by the
//                                                 tree's kernel and by the SPIR-V given, acc word for word; exit 1
//                                                 on a difference (a batch's peaks differ run to run, SELECT's order,
//                                                 so two runs' dumps cannot show it)
//   lizard_gpu_check replay <run> [frames] [variant]   a recording (research/captures/v0.3/<run>) through the
//                                                 decoder as the web's replay runs it (batches of the batcher's size,
//                                                 two in flight, the held word carried in capture order): frames
//                                                 found, words, verified blocks, GPU ms a frame
//
// ai: info needs only a Vulkan device, build and compile gen's tree too (LIZ_OUT); bench out/spv/bench_*.spv, which no
// ai: build step makes (core/nets/README.md has the line). selftest, same vote and dump read the self-test's frames and
// ai: references, replay and receive a recording, ingest (Android) either: the lab's research/ (LIZ_BARCODE), which is
// ai: not published. A run lizard_engine paint makes reads as a recording put at <LIZ_BARCODE>/captures/v0.3/<run>.
//
// ai: Env: LIZ_OUT (the generated tree, liblizard/out; on the phone /data/local/tmp/lizard/out), LIZ_BARCODE (research/,
// ai: for its build/ frames and references and its captures/ recordings), LIZ_VK_DEVICE (a substring of the device's name), LIZ_VALIDATE=1 (the
// ai: Khronos validation layer), LIZ_TRACE=<path> (wg's trace, a JSON line an event), LIZ_CACHE=<path> (the pipeline
// ai: cache file), LIZ_BUDGET=<MB> (the decoder's memory budget; 2048 by default, the web's on the S26).
#include "front.h"
#include "receiver.h"
#include "replay.h"
#include <algorithm>
#include <atomic>
#include <thread>

#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>

using namespace lizard;

static std::string env(const char* k, const char* d = "") { const char* v = getenv(k); return v ? v : d; }

static std::vector<uint8_t> readAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw wg::Error("no " + p);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ai: Each of a run's first `frames` frames' w x h: meta's sizes where it has them (the web's format: a run may hold
// ai: frames of several sizes, a tracked crop's or a camera reopened at another resolution), else the run's w x h.
static std::vector<std::pair<uint32_t, uint32_t>> frameSizes(const json& meta, int frames) {
  std::vector<std::pair<uint32_t, uint32_t>> out(frames, {meta["w"].get<uint32_t>(), meta["h"].get<uint32_t>()});
  if (meta.contains("sizes") && meta["sizes"].is_array())
    for (int i = 0; i < frames && i < (int)meta["sizes"].size(); i++)
      if (meta["sizes"][i].is_array()) out[i] = {meta["sizes"][i][0].get<uint32_t>(), meta["sizes"][i][1].get<uint32_t>()};
  return out;
}

struct Ctx {
  std::unique_ptr<wg::Device> dev;
  std::string out, barcode;
  std::vector<std::string> variants;
};

static std::unique_ptr<FrontHalf> open(Ctx& c, const std::string& v) {
  const double budget = std::stod(env("LIZ_BUDGET", "2048")) * (1 << 20);
  auto fh = FrontHalf::create(*c.dev, readDir(c.out), v, budget, 0, [](const std::string& m) { std::cout << "  " << m << std::endl; });
  return fh;
}

// ai: dump: the self-test's batch, then the batch's lane buffers written to <dir>/<name>.bin, to hold one device's
// ai: stages against another's (the same variant on the 4090 and the phone) and find the first that parts.
static std::vector<uint8_t> readBack(wg::Device& d, wg::Buffer& b) {
  auto rd = d.createBuffer(b.size, wg::MAP_READ | wg::COPY_DST);
  auto e = d.encoder();
  e.copyBufferToBuffer(b, 0, *rd, 0, b.size);
  d.submit(e)->wait();
  const uint8_t* p = d.read(*rd, 0, b.size);
  return std::vector<uint8_t>(p, p + b.size);
}

static int dump(Ctx& c, std::string v, const std::string& dir) {
  auto fh = open(c, v);
  auto refBytes = readAll(c.barcode + "/build/gpu_selftest_ref.json");
  json ref = json::parse(refBytes.begin(), refBytes.end());
  const auto& frames = ref["frames"];
  const uint32_t W = frames[0]["w"], H = frames[0]["h"];
  fh->plan(W, H);
  std::vector<std::vector<uint8_t>> px;
  std::vector<Slot*> slots;
  for (auto& f : frames) px.push_back(readAll(c.barcode + "/build/" + ref["source"].get<std::string>() + "/" + f["name"].get<std::string>()));
  for (size_t i = 0; i < px.size(); i++) slots.push_back(fh->enqueueLuma(px[i].data(), W, H, W, i));
  auto out = fh->finish(fh->run(slots, 0));
  Lane& ln = *fh->lanes[0];
  std::vector<std::pair<std::string, BufP>> bufs = {{"peaks", ln.peaksBuf}, {"counts", ln.countsBuf}, {"readings", ln.readingsBuf}, {"raw", ln.rawBuf}, {"acc", ln.accBuf},
    {"quads", ln.quadsBuf}, {"score", ln.scoreBuf}, {"result", ln.resultBuf}, {"maps", ln.mapsBuf}, {"resid", ln.residBuf}, {"meas", ln.measBuf}, {"logit", ln.logitBuf}, {"list", ln.listBuf}};
  for (size_t i = 0; i < ln.level.size(); i++) bufs.push_back({"level" + std::to_string(i + 1), ln.level[i].buf});
  for (size_t i = 0; i < ln.fcn2Levels.size(); i++) if (ln.fcn2Levels[i].plane) bufs.push_back({"plane" + std::to_string(i + 1), ln.fcn2Levels[i].plane});
  auto& b = *ln.back;
  for (auto& [k, x] : std::initializer_list<std::pair<const char*, BufP>>{{"lists", b.lists}, {"args", b.args}, {"part", b.part}, {"coef", b.coef}, {"y", b.y}, {"s", b.s}, {"bcounts", b.counts}, {"L", b.L}, {"EST", b.EST}, {"BLK", b.BLK}, {"V", b.V}, {"ITS", b.ITS}}) bufs.push_back({k, x});
  for (auto& [name, bp] : bufs) {
    if (!bp) continue;
    auto data = readBack(*c.dev, *bp);
    std::ofstream f(dir + "/" + name + ".bin", std::ios::binary);
    f.write((const char*)data.data(), (std::streamsize)data.size());
  }
  int blocks = 0;
  for (auto& fo : out.frames) blocks += (int)fo.records.size();
  printf("%s: dumped %zu buffers to %s, %d blocks\n", v.c_str(), bufs.size(), dir.c_str(), blocks);
  return 0;
}

#ifdef __ANDROID__
#include "camera.h"
#include <android/hardware_buffer.h>
// ai: ingest [variant] [run]: the self-test's frames (or a run's, below) through the camera's zero-copy path with no camera: each frame's luma written
// ai: into the Y plane of a YUV 4:2:0 hardware buffer of a camera frame's size (1920 x 1080, the frame in its centre
// ai: square, chroma 128), then sampled into the ring by CameraIngest as a camera frame is; the batch's blocks must
// ai: be the luma path's (selftest).
// ai: ingest <variant> <run> (2026-10-01, the S26 at 3840x2160 finding nothing): a recording's first frames (at most
// ai: a batch) instead, each in the centre of a 16:9 camera buffer as tall as the frame (3840 x 2160 for a 2160 run),
// ai: the plan made at 1080 first and grown to the frame's size, as the app's camera switched from 1080 live; the
// ai: blocks printed, no reference.
static int ingestCheck(Ctx& c, std::string v, const std::string& run = "") {
  auto fh = open(c, v);
  auto ci = CameraIngest::create(*c.dev, readDir(c.out));
  if (!ci) throw wg::Error("no zero-copy ingest on this device");
  fh->ingest = ci.get();
  json ref;
  std::vector<std::vector<uint8_t>> frames;
  uint32_t S;
  if (run.empty()) {
    auto refBytes = readAll(c.barcode + "/build/gpu_selftest_ref.json");
    ref = json::parse(refBytes.begin(), refBytes.end());
    S = ref["frames"][0]["w"];
    for (auto& f : ref["frames"]) frames.push_back(readAll(c.barcode + "/build/" + ref["source"].get<std::string>() + "/" + f["name"].get<std::string>()));
  } else {
    const std::string dir = c.barcode + "/captures/v0.3/" + run;
    auto metaBytes = readAll(dir + "/meta.json");
    json meta = json::parse(metaBytes.begin(), metaBytes.end());
    S = meta["w"];
    char name[32];
    for (int i = 0; i < std::min<int>(meta["frames"].get<int>(), 16); i++) { snprintf(name, sizeof name, "/%04d.gray", i); frames.push_back(readAll(dir + name)); }
    fh->plan(1080, 1080);
  }
  const uint32_t CH = std::max<uint32_t>(S, 1080), CW = std::max<uint32_t>(1920, (CH * 16 / 9 + 1) & ~1u);
  fh->plan(S, S);
  std::vector<AHardwareBuffer*> hbs;
  std::vector<Slot*> slots;
  for (auto& px : frames) {
    AHardwareBuffer_Desc d{};
    d.width = CW; d.height = CH; d.layers = 1; d.format = AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420;
    d.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    AHardwareBuffer* hb = nullptr;
    if (AHardwareBuffer_allocate(&d, &hb)) throw wg::Error("AHardwareBuffer_allocate");
    AHardwareBuffer_Planes pl{};
    if (AHardwareBuffer_lockPlanes(hb, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &pl)) throw wg::Error("AHardwareBuffer_lockPlanes");
    const uint32_t x0 = (CW - S) / 2, y0 = (CH - S) / 2;
    for (uint32_t y = 0; y < CH; y++) {
      uint8_t* row = (uint8_t*)pl.planes[0].data + (size_t)y * pl.planes[0].rowStride;
      for (uint32_t x = 0; x < CW; x++) row[(size_t)x * pl.planes[0].pixelStride] = 128;
      if (y >= y0 && y < y0 + S) for (uint32_t x = 0; x < S; x++) row[(size_t)(x0 + x) * pl.planes[0].pixelStride] = px[(size_t)(y - y0) * S + x];
    }
    for (int p = 1; p < 3; p++)
      for (uint32_t y = 0; y < CH / 2; y++)
        for (uint32_t x = 0; x < CW / 2; x++) ((uint8_t*)pl.planes[p].data)[(size_t)y * pl.planes[p].rowStride + (size_t)x * pl.planes[p].pixelStride] = 128;
    AHardwareBuffer_unlock(hb, nullptr);
    hbs.push_back(hb);
    slots.push_back(fh->enqueueCamera(*ci, hb, x0, y0, S, S, slots.size()));
  }
  auto out = fh->finish(fh->run(slots, 0));
  int blocks = 0, found = 0;
  for (auto& fo : out.frames) { blocks += (int)fo.records.size(); found += fo.finder.found == 1; }
  int refBlocks = 0;
  if (!run.empty()) printf("%s ingest of %s (%u x %u in a %u x %u buffer, planned at 1080 then grown): %d of %zu found, %d blocks\n", v.c_str(), run.c_str(), S, S, CW, CH, found, hbs.size(), blocks);
  else {
    for (auto& r : ref["summaries"]) refBlocks += r["back"]["blocks"].get<int>();
    printf("%s ingest: %d of %zu found, %d blocks (the reference %d)\n", v.c_str(), found, hbs.size(), blocks, refBlocks);
  }
  fh.reset();
  ci.reset();
  for (auto* hb : hbs) AHardwareBuffer_release(hb);
  return 0;
}
#endif

// ai: receive: a recording through the whole receiver (rx/receiver.h) at a camera's pace (fps, 60 by default):
// ai: the queue, the batches, the held word, the judge, the dedupe and the transfer, the stats printed each second.
// ai: layout 2:1: each camera frame is two recorded frames side by side (2W x H), which the receiver cuts in half.
// ai: A frame's timestamp is its push time on the steady clock, and the receiver's series is asked after every push as
// ai: the app's phase lock asks it (Receiver::series): the last line but one says how old a frame's row was when it
// ai: came, and the verified blocks of every row. LIZ_SOON=1: results asked for soon from the first push
// ai: (Receiver::soon: the GPU decoder's batches at 8 frames). LIZ_BATCH=<n>: the most frames a batch waits for
// ai: (Receiver::batchCap, the app's Batch size).
static int receive(Ctx& c, const std::string& run, double fps, const std::string& storeDir, const std::string& layout = "1:1") {
  const std::string dir = c.barcode + "/captures/v0.3/" + run;
  auto metaBytes = readAll(dir + "/meta.json");
  json meta = json::parse(metaBytes.begin(), metaBytes.end());
  const uint32_t W = meta["w"], H = meta["h"];
  // ai: the frames present (a phone holds the first n of a recording), LIZ_LAPS laps over them (1 by default)
  int frames = meta["frames"];
  char name[32];
  for (int i = 0; i < frames; i++) { snprintf(name, sizeof name, "/%04d.gray", i); if (!std::ifstream(dir + name).good()) { frames = i; break; } }
  if (!frames) throw wg::Error(run + ": no frames");   // ai: a replay that kept none (the camera stopped at once)
  // ai: LIZ_GROW=1 (layout 1:1): the first lap pushes each frame's centre three quarters as a frame of its own, the
  // ai: laps after it the whole frame, so the receiver plans small and must plan again for the larger crop (the
  // ai: phone's camera reopened at a higher resolution; STATUS "Higher resolutions in the Android app"): the file
  // ai: must still come, from the later laps. Two laps at least.
  const bool grow = getenv("LIZ_GROW") && atoi(getenv("LIZ_GROW")) && layout == "1:1";
  const int laps = std::max(grow ? 2 : 1, getenv("LIZ_LAPS") ? std::max(1, atoi(getenv("LIZ_LAPS"))) : 1);
  // ai: LIZ_PLAN=<side>: one blank frame of that side pushed first, so the receiver plans for it and then decodes the
  // ai: recording's smaller frames on that plan: what a plan grown for a higher resolution costs the frames of a
  // ai: lower one (the phone's camera reopened smaller), read off the stats' gpuMs against a run without it.
  const uint32_t planSide = getenv("LIZ_PLAN") ? (uint32_t)atoi(getenv("LIZ_PLAN")) : 0;
  ReceiverConfig cfg;
  cfg.assets = c.out; cfg.cacheDir = storeDir; cfg.storeDir = storeDir; cfg.layout = layout;
  // ai: LIZ_DECODER: auto (the default), gpu or cpu; LIZ_CPU_THREADS: the CPU decoder's workers (0: its pool's rule)
  if (getenv("LIZ_DECODER")) cfg.decoder = getenv("LIZ_DECODER");
  if (getenv("LIZ_CPU_THREADS")) cfg.cpuThreads = atoi(getenv("LIZ_CPU_THREADS"));
  cfg.log = [](const std::string& m) { std::cout << "  " << m << std::endl; };
  std::atomic<int> released{0};
  auto rx = Receiver::create(cfg, [&](uint64_t) { released++; });
  if (getenv("LIZ_SOON") && atoi(getenv("LIZ_SOON"))) rx->soon(true);
  if (getenv("LIZ_BATCH")) rx->batchCap(atoi(getenv("LIZ_BATCH")));
  // ai: LIZ_REPLAY=<dir> (Save replays, 2026-10-03; rx/replay.h): the newest LIZ_REPLAY_FRAMES (300) frames the
  // ai: decoder is handed kept in <dir> as the app's Developer Tools keeps them, the run ended once the pushes are done
  std::shared_ptr<Replay> replay;
  if (getenv("LIZ_REPLAY")) {
    replay = std::make_shared<Replay>(getenv("LIZ_REPLAY"), "run-check", getenv("LIZ_REPLAY_FRAMES") ? atoi(getenv("LIZ_REPLAY_FRAMES")) : 300);
    rx->record(replay);
  }
  auto nowNs = [] { return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  // ai: the rows not yet seen: counted, their verified blocks summed, and (timed) each one's age now
  double since = 0;
  long pushes = 0, rows = 0, verified = 0;
  std::vector<double> ages;
  auto poll = [&](bool timed) {
    const std::vector<double> v = rx->series(since);
    const double now = nowNs() / 1e6;
    for (size_t at = 1; at + 8 <= v.size(); at += 8) {
      rows++;
      verified += (long)v[at + 1];
      if (timed) ages.push_back(now - v[at]);
      since = std::max(since, v[at]);
    }
    return v.size() > 1;
  };
  std::vector<std::vector<uint8_t>> px(frames);
  const auto sizes = frameSizes(meta, frames);
  for (int i = 0; i < frames; i++) {
    snprintf(name, sizeof name, "/%04d.gray", i);
    px[i] = readAll(dir + name);
    if (px[i].size() != (size_t)sizes[i].first * sizes[i].second)
      throw wg::Error(std::string(name + 1) + ": " + std::to_string(px[i].size()) + " bytes for " + std::to_string(sizes[i].first) + " x " + std::to_string(sizes[i].second));
    // ai: 2:1 lays two frames side by side and LIZ_GROW crops by the run's size: both want frames of one size
    if ((layout == "2:1" || grow) && sizes[i] != sizes[0]) throw wg::Error(std::string(layout == "2:1" ? "2:1" : "LIZ_GROW") + " needs frames of one size; " + run + " holds several");
  }
  if (planSide) {
    std::vector<uint8_t> blank((size_t)planSide * planSide, 128);
    CameraFrame f;
    f.luma = blank.data(); f.width = planSide; f.height = planSide; f.stride = planSide; f.tag = ~0ull;
    // ai: the first push only records the shape (the decoder thread plans on it); pushed until it is taken
    for (int k = 0; k < 200; k++) {
      f.timestampNs = nowNs();
      rx->push(f);
      pushes++;
      poll(true);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (json::parse(rx->stats())["state"] != "starting") break;
    }
    std::cout << "  planned for " << planSide << " x " << planSide << " first" << std::endl;
  }
  const auto t0 = std::chrono::steady_clock::now();
  double nextPrint = 1;
  for (int i = 0; i < laps * frames + (int)(3 * fps); i++) {
    const int k = i % frames;
    CameraFrame f;
    std::vector<uint8_t> wide;
    if (layout == "2:1") {
      const auto& a = px[(2 * i) % frames]; const auto& b = px[(2 * i + 1) % frames];
      wide.resize((size_t)2 * W * H);
      for (uint32_t y = 0; y < H; y++) { memcpy(&wide[(size_t)y * 2 * W], &a[(size_t)y * W], W); memcpy(&wide[(size_t)y * 2 * W + W], &b[(size_t)y * W], W); }
      f.luma = wide.data(); f.width = 2 * W; f.height = H; f.stride = 2 * W;
    } else if (grow && i < frames) {
      const uint32_t s = std::min(W, H) * 3 / 4, x = (W - s) / 2, y = (H - s) / 2;
      f.luma = px[k].data() + (size_t)y * W + x; f.width = s; f.height = s; f.stride = W;
    } else { f.luma = px[k].data(); f.width = sizes[k].first; f.height = sizes[k].second; f.stride = sizes[k].first; }
    f.tag = (uint64_t)i;
    f.timestampNs = nowNs();
    rx->push(f);
    pushes++;
    poll(true);
    std::this_thread::sleep_until(t0 + std::chrono::microseconds((int64_t)((i + 1) * 1e6 / fps)));
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (t >= nextPrint) { std::cout << rx->stats() << std::endl; nextPrint += 1; }
    if (!rx->file().empty() && (laps == 1 || (grow && i >= frames))) break;   // ai: laps run to their end (a load to measure, not a file to get)
  }
  // ai: LIZ_REPLAY: the run ended as the app ends one when its camera stops, at once after the last push: what the
  // ai: decoder still holds for it comes in first (Replay::finish waits for it)
  if (replay) {
    rx->record(nullptr);
    const int n = replay->finish(rx->stats(), "", "{}");
    printf("replay: %d frames kept, %lld B%s\n", n, (long long)replay->bytes(), replay->error().empty() ? "" : (" (" + replay->error() + ")").c_str());
  }
  // ai: the frames still staged or in flight when the pushes stop: their rows and blocks counted, so two runs count
  // ai: the same frames, but not their ages (no frame is coming to fill their batch); until no row has come for
  // ai: longer than the receiver waits to launch a part batch (1.5 batches of 32 at this pace)
  for (auto quiet = std::chrono::steady_clock::now(); std::chrono::duration<double>(std::chrono::steady_clock::now() - quiet).count() < 1.5 * 32 / fps + 0.25;) {
    if (poll(false)) quiet = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  std::sort(ages.begin(), ages.end());
  const size_t na = ages.size();
  printf("series: %ld rows of %ld pushes, a row's age when it came median %.0f ms, 90th percentile %.0f ms (over the %zu that came while frames were pushed), %ld verified blocks\n",
         rows, pushes, na ? ages[na / 2] : 0.0, na ? ages[std::min(na - 1, (size_t)(0.9 * na))] : 0.0, na, verified);
  fflush(stdout);
  const json last = json::parse(rx->stats());
  std::cout << "file: " << (rx->file().empty() ? "(none)" : rx->file()) << "; frames released " << released
            << "; state " << last["state"].get<std::string>() << (last["error"].get<std::string>().empty() ? "" : " (" + last["error"].get<std::string>() + ")") << std::endl;
  // ai: the file, and the receiver not left saying error (a plan that grew clears one; STATUS "Higher resolutions")
  return rx->file().empty() || last["state"] == "error" ? 1 : 0;
}

// ai: bench: the device's int8 and f16 matrix units (VK_KHR_cooperative_matrix) against the packed int8 dot product
// ai: today's int8 kernels use, each a kernel of multiply-adds in registers (core/nets/bench_*.comp), timed by
// ai: timestamps: the ceiling a rewrite of the nets can reach.
static int bench(Ctx& c) {
  auto& d = *c.dev;
  auto bgl = d.createBindGroupLayout({wg::Bind::RW, wg::Bind::RW});
  const uint32_t G = 2048;
  auto src = d.createBuffer(1 << 20, wg::STORAGE | wg::COPY_DST);
  auto dst = d.createBuffer((uint64_t)G * 64 * 64 * 4, wg::STORAGE);
  auto group = d.createBindGroup(bgl, {src, dst});
  auto qs = d.createQuerySet(2);
  auto res = d.createBuffer(16, wg::QUERY_RESOLVE | wg::COPY_SRC);
  auto rd = d.createBuffer(16, wg::MAP_READ | wg::COPY_DST);
  d.flush();
  struct Case { const char* name; const char* spv; uint32_t N, R; double macs; };
  std::vector<Case> cases;
  const uint32_t R = 512;
  cases.push_back({"dot4x8 int8", "bench_dot", 0, R, (double)G * 64 * 8 * R * 4});
  if (d.features.coopmat) {
    for (uint32_t N : {16u, 32u, 64u}) cases.push_back({"coopmat int8 64xNx32", "bench_coop", N, R, (double)G * 64 * N * 32 * R});
    for (uint32_t N : {16u, 32u, 64u}) cases.push_back({"coopmat f16 64xNx16", "bench_h16", N, R, (double)G * 64 * N * 16 * R});
  }
  for (auto& k : cases) {
    auto b = readAll(c.out + "/spv/" + k.spv + ".spv");
    std::vector<uint32_t> w(b.size() / 4);
    memcpy(w.data(), b.data(), b.size());
    std::shared_ptr<wg::Pipeline> p;
    try { p = d.createPipeline(w, bgl, "", k.name, {k.N ? k.N : 32u, k.R}); }
    catch (const std::exception& e) { printf("%-24s N %2u: %s\n", k.name, k.N, e.what()); continue; }
    double best = 1e9;
    for (int rep = 0; rep < 4; rep++) {
      auto e = d.encoder();
      e.beginComputePass(qs.get(), 0, 1);
      e.setPipeline(*p); e.setBindGroup(*group); e.dispatch(G);
      e.endPass();
      e.resolveQuerySet(*qs, 0, 2, *res, 0);
      e.copyBufferToBuffer(*res, 0, *rd, 0, 16);
      d.submit(e)->wait();
      const uint64_t* t = (const uint64_t*)d.read(*rd, 0, 16);
      if (rep) best = std::min(best, (double)((t[1] - t[0]) & d.timestampMask) * d.timestampPeriod / 1e6);
    }
    printf("%-24s N %2u: %.3f ms, %.2f T multiply-adds a second\n", k.name, k.N, best, k.macs / (best / 1e3) / 1e12);
  }
  // ai: What a dispatch costs past its work: bench_dot at one round and 32 workgroups (a small stage's size at a
  // ai: batch of 32), once and K times in one pass, each behind the barrier every dispatch here has (wg.cpp
  // ai: hazard). (t(K) - t(1)) / (K - 1) is the floor a stage merged into another can save (2026-09-30).
  {
    auto b = readAll(c.out + "/spv/bench_dot.spv");
    std::vector<uint32_t> w(b.size() / 4);
    memcpy(w.data(), b.data(), b.size());
    auto p = d.createPipeline(w, bgl, "", "overhead", {32u, 1u});
    auto time = [&](int K) {
      double best = 1e9;
      for (int rep = 0; rep < 6; rep++) {
        auto e = d.encoder();
        e.beginComputePass(qs.get(), 0, 1);
        e.setPipeline(*p); e.setBindGroup(*group);
        for (int k = 0; k < K; k++) e.dispatch(32);
        e.endPass();
        e.resolveQuerySet(*qs, 0, 2, *res, 0);
        e.copyBufferToBuffer(*res, 0, *rd, 0, 16);
        d.submit(e)->wait();
        const uint64_t* t = (const uint64_t*)d.read(*rd, 0, 16);
        if (rep) best = std::min(best, (double)((t[1] - t[0]) & d.timestampMask) * d.timestampPeriod / 1e6);
      }
      return best;
    };
    const double t1 = time(1), t64 = time(64);
    printf("%-24s: one %.4f ms, 64 %.4f ms, %.2f us a dispatch past the first\n", "dispatch and barrier", t1, t64, (t64 - t1) / 63 * 1e3);
  }
  return 0;
}

// ai: same vote: the self-test's batch decoded once, then VOTE alone twice on the lane as the batch left it (its
// ai: frames and readings), acc cleared before each: the tree's kernel, then the SPIR-V at spvPath over the same
// ai: layout and group. Every word of acc must agree.
static int sameVote(Ctx& c, std::string v, const std::string& spvPath) {
  auto fh = open(c, v);
  auto refBytes = readAll(c.barcode + "/build/gpu_selftest_ref.json");
  json ref = json::parse(refBytes.begin(), refBytes.end());
  const auto& frames = ref["frames"];
  const uint32_t W = frames[0]["w"], H = frames[0]["h"], n = (uint32_t)frames.size();
  fh->plan(W, H);
  std::vector<std::vector<uint8_t>> px;
  std::vector<Slot*> slots;
  for (auto& f : frames) px.push_back(readAll(c.barcode + "/build/" + ref["source"].get<std::string>() + "/" + f["name"].get<std::string>()));
  for (size_t i = 0; i < px.size(); i++) slots.push_back(fh->enqueueLuma(px[i].data(), W, H, W, i));
  auto out = fh->finish(fh->run(slots, 0));
  Lane& ln = *fh->lanes[0];
  const Pipe now = fh->S->pair(fh->S->tree["front"]["vote"]);
  auto b = readAll(spvPath);
  std::vector<uint32_t> words(b.size() / 4);
  memcpy(words.data(), b.data(), words.size() * 4);
  auto other = c.dev->createPipeline(words, now.bgl, "vote@other", "vote, the other build", {});
  const uint64_t size = ln.accBuf->size / ln.B * n;
  auto keep = c.dev->createBuffer(2 * size, wg::MAP_READ | wg::COPY_DST);
  auto e = c.dev->encoder();
  for (int k = 0; k < 2; k++) {
    e.clearBuffer(*ln.accBuf, 0, size);
    e.beginComputePass();
    e.setPipeline(k ? *other : *now.p);
    e.setBindGroup(*ln.voteGroup);
    e.dispatch(((uint32_t)fh->cap + 63) / 64, n);
    e.endPass();
    e.copyBufferToBuffer(*ln.accBuf, 0, *keep, k * size, size);
  }
  c.dev->submit(e)->wait();
  const uint32_t* a = (const uint32_t*)c.dev->read(*keep, 0, 2 * size);
  const uint64_t nw = size / 4;
  uint64_t differ = 0, massA = 0, massB = 0, cells = 0;
  for (uint64_t i = 0; i < nw; i++) { differ += a[i] != a[nw + i]; massA += a[i]; massB += a[nw + i]; cells += a[i] != 0; }
  int blocks = 0;
  for (auto& fo : out.frames) blocks += (int)fo.records.size();
  printf("%s: VOTE on %u frames (%d blocks decoded), %llu words of acc: %llu differ; votes %llu in %llu cells by the tree's kernel, %llu by %s\n", v.c_str(), n, blocks,
    (unsigned long long)nw, (unsigned long long)differ, (unsigned long long)massA, (unsigned long long)cells, (unsigned long long)massB, spvPath.c_str());
  return differ || !massA ? 1 : 0;
}

// ai: The self-test: the eight frames in one batch, as the web's page runs them, against its reference.
static int selftest(Ctx& c, std::string v) {
  auto fh = open(c, v);
  const bool int8 = v.rfind("int8", 0) == 0;
  auto refBytes = readAll(c.barcode + "/build/" + (int8 ? "gpu_selftest_ref_int8.json" : "gpu_selftest_ref.json"));
  json ref = json::parse(refBytes.begin(), refBytes.end());
  const auto& frames = ref["frames"];
  const int n = (int)frames.size();
  std::vector<std::vector<uint8_t>> px;
  for (auto& f : frames) px.push_back(readAll(c.barcode + "/build/" + ref["source"].get<std::string>() + "/" + f["name"].get<std::string>()));
  const uint32_t W = frames[0]["w"], H = frames[0]["h"];
  fh->plan(W, H);
  std::vector<Slot*> slots;
  for (int i = 0; i < n; i++) slots.push_back(fh->enqueueLuma(px[i].data(), W, H, W, i));
  auto out = fh->finish(fh->run(slots, 0));
  const double TOLpx = 1, TOLcount = 0.1;
  int bad = 0, blocks = 0, refBlocks = 0;
  std::cout << "frame  found ring  corners(px)  word(v)   version  blocks(ref)  ids shared  raw(ref)" << std::endl;
  for (int i = 0; i < n; i++) {
    const json& R = ref["summaries"][i];
    const auto& fo = out.frames[i];
    const json& A = R["finder"];
    std::string why;
    double cmax = 0;
    if (A["found"].get<int>() != (int)fo.finder.found) why += " found";
    if (A["found"].get<int>() == 1) {
      if (A["ring"].get<int>() != (int)fo.finder.ring) why += " ring";
      for (int k = 0; k < 8; k++) cmax = std::max(cmax, std::abs(A["corners"][k].get<double>() - fo.finder.corners[k]));
      if (cmax > TOLpx) why += " corners";
    }
    const json& Wd = R["word"];
    const bool wordRef = !Wd["word"].is_null();
    if (wordRef != fo.hasWord || (wordRef && (Wd["word"]["version"].get<int>() != fo.word.version || Wd["word"]["fps"].get<int>() != fo.word.fps))) why += " word";
    if (Wd["version"].get<int>() != fo.version || Wd["n"].get<int>() != fo.n || Wd["held"].get<bool>() != fo.held) why += " version";
    const int rb = R["back"]["blocks"];
    std::set<int> ids;
    for (auto& x : R["back"]["ids"]) ids.insert(x.get<int>());
    int shared = 0;
    for (auto& r : fo.records) shared += ids.count((int)r.id);
    if (std::abs((double)fo.records.size() - rb) > TOLcount * std::max(rb, 1)) why += " blocks";
    const int raw = R["proposer"]["raw"];
    if (std::abs((double)fo.counts[1] - raw) > TOLcount * std::max(raw, 1)) why += " raw";
    blocks += (int)fo.records.size();
    refBlocks += rb;
    printf("%5d  %5d %4d  %10.2f  %8d  %7d  %6zu(%3d)  %10d  %6u(%6d)%s%s\n", i, (int)fo.finder.found, fo.ring, cmax, fo.hasWord ? fo.word.version : 0, fo.version, fo.records.size(), rb, shared, fo.counts[1], raw, why.empty() ? "" : "  DIFFERS:", why.c_str());
    if (!why.empty()) bad++;
  }
  printf("%s: %d blocks (reference %d), %d of %d frames differ; batch %.2f ms of GPU (%d frames)\n", v.c_str(), blocks, refBlocks, bad, n, out.gpuMs.value_or(-1), n);
  return bad ? 1 : 0;
}

// ai: A recording through the decoder as the web's replay runs it (scripts/exp/gpu_captures.mjs): the frames in capture
// ai: order, batches of the batcher's size, two in flight, each launched with the last word read so far.
static int replay(Ctx& c, const std::string& run, int limit, std::string v) {
  auto fh = open(c, v);
  const std::string dir = c.barcode + "/captures/v0.3/" + run;
  auto metaBytes = readAll(dir + "/meta.json");
  json meta = json::parse(metaBytes.begin(), metaBytes.end());
  const int frames = std::min<int>(limit > 0 ? limit : 1 << 30, meta["frames"].get<int>());
  const auto sizes = frameSizes(meta, frames);
  uint32_t W = meta["w"], H = meta["h"];
  for (auto& wh : sizes) { W = std::max(W, wh.first); H = std::max(H, wh.second); }
  fh->plan(W, H);
  std::deque<std::unique_ptr<InFlight>> flight;
  int held = 0, found = 0, words = 0, heldFrames = 0, next = 0;
  long blocks = 0;
  std::vector<float> pilots, pilots2;   // ai: each frame's pilot readings where read (SPEC 7.3): the even blocks', the odd blocks'
  double gpuMs = 0, gpuFrames = 0;
  std::vector<std::pair<std::string, double>> stageSum;
  const double t0 = fh->now();
  auto drain = [&]() {
    auto out = fh->finish(std::move(flight.front()));
    flight.pop_front();
    for (auto& fo : out.frames) {
      if (fo.empty) continue;
      found += fo.finder.found == 1;
      blocks += (long)fo.records.size();
      if (fo.hasWord) { words++; held = fo.word.version; }
      heldFrames += fo.held;
      if (fo.pilotBlocks) { pilots.push_back(fo.pilotR); pilots2.push_back(fo.pilotR2); }
    }
    if (out.gpuMs) { gpuMs += *out.gpuMs; gpuFrames += out.carried; }
    for (auto& kv : out.stageMs) {
      auto it = std::find_if(stageSum.begin(), stageSum.end(), [&](auto& x) { return x.first == kv.first; });
      if (it == stageSum.end()) stageSum.push_back(kv); else it->second += kv.second;
    }
  };
  char name[32];
  std::vector<std::vector<uint8_t>> px;
  while (next < frames || !flight.empty()) {
    if (next < frames && (int)flight.size() < fh->inflightMax() && fh->freeLane()) {
      const int nb = std::min(fh->size(), frames - next);
      std::vector<Slot*> slots;
      for (int i = 0; i < nb; i++, next++) {
        snprintf(name, sizeof name, "/%04d.gray", next);
        auto p = readAll(dir + name);
        const uint32_t w = sizes[next].first, h = sizes[next].second;
        if (p.size() != (size_t)w * h) throw wg::Error(dir + name + ": " + std::to_string(p.size()) + " bytes for " + std::to_string(w) + " x " + std::to_string(h));
        // ai: A slot comes back once the batch that took it has copied it out (front.cpp takeSlot): the batch just
        // ai: launched may not have on a slow device (the iGPU, 32 frames a batch, a ring of 36), so wait for it.
        Slot* s = fh->enqueueLuma(p.data(), w, h, w, (uint64_t)next);
        for (int t = 0; !s && t < 20000; t++) { std::this_thread::sleep_for(std::chrono::microseconds(250)); s = fh->enqueueLuma(p.data(), w, h, w, (uint64_t)next); }
        if (!s) throw wg::Error("the ring is full");
        slots.push_back(s);
      }
      flight.push_back(fh->run(slots, held, env("PROFILE") == "1"));
    } else drain();
  }
  const double ms = fh->now() - t0;
  if (!stageSum.empty()) {
    printf("stages, ms a frame:");
    for (auto& [k, v] : stageSum) printf(" %s %.2f", k.c_str(), v / std::max(1.0, gpuFrames));
    printf("\n");
  }
  printf("%s: %d frames, %d found, %d words, %d held, %ld verified blocks; %.2f ms of GPU a frame, %.0f frames a second end to end (%s, batches of %d)\n", run.c_str(), frames, found, words, heldFrames, blocks,
         gpuFrames ? gpuMs / gpuFrames : -1, frames / (ms / 1000), v.c_str(), fh->B);
  for (auto* pv : {&pilots, &pilots2}) if (!pv->empty()) {
    std::vector<float> v = *pv;
    std::sort(v.begin(), v.end());
    const int neg = (int)std::count_if(v.begin(), v.end(), [](float x) { return x < 0; });
    printf("pilots: read on %zu of %d frames; %s from %.3f to %.3f, median %.3f, %d negative\n", v.size(), frames, pv == &pilots ? "r (the even blocks)" : "r2 (the odd blocks)", v.front(), v.back(), v[v.size() / 2], neg);
  }
  return 0;
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "info";
  try {
    Ctx c;
    auto log = [](const std::string& m) { std::cout << m << std::endl; };
    c.dev = wg::Device::create(env("LIZ_VK_DEVICE"), env("LIZ_VALIDATE") == "1", log);
    if (!env("LIZ_TRACE").empty()) c.dev->traceOut = fopen(env("LIZ_TRACE").c_str(), "w");
    if (!env("LIZ_CACHE").empty()) c.dev->loadCache(env("LIZ_CACHE"));
    c.variants = variantsFor(*c.dev);
    c.out = env("LIZ_OUT", "out");
    c.barcode = env("LIZ_BARCODE", "../research");
    std::cout << "variants:";
    for (auto& v : c.variants) std::cout << " " << v;
    std::cout << (c.variants.empty() ? " none (the C decodes here)" : "") << std::endl;
    auto pick = [&](int at) { return argc > at ? std::string(argv[at]) : (c.variants.empty() ? std::string("f32") : c.variants[0]); };
    int rc = 0;
    if (cmd == "info") rc = 0;
    else if (cmd == "build") {
      std::string v = pick(2);
      int B = argc > 3 ? atoi(argv[3]) : 32;
      auto s = Setup::load(*c.dev, readDir(c.out), v);
      s->build(B);
      std::cout << "built " << v << " at B " << B << ": " << s->pipelines.size() << " pipelines, " << s->buffers.size() << " buffers, " << (c.dev->allocated >> 10) << " KB of device memory, " << (int)s->buildMs << " ms" << std::endl;
    } else if (cmd == "compile") {
      // ai: one pipeline of a variant (its layout with it), timed: which module a driver takes long over
      if (argc < 4) throw wg::Error("compile <variant> <pipeline id>...");
      auto s = Setup::load(*c.dev, readDir(c.out), argv[2]);
      for (int i = 3; i < argc; i++) {
        auto t0 = std::chrono::steady_clock::now();
        s->makeOne(argv[i]);
        printf("%s %s: %.0f ms\n", argv[2], argv[i], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
    } else if (cmd == "selftest") rc = selftest(c, pick(2));
#ifdef __ANDROID__
    else if (cmd == "ingest") rc = ingestCheck(c, pick(2), argc > 3 ? argv[3] : "");
#endif
    else if (cmd == "receive") {
      if (argc < 3) throw wg::Error("receive <run> [fps] [store dir] [1:1|2:1]");
      rc = receive(c, argv[2], argc > 3 ? atof(argv[3]) : 60, argc > 4 ? argv[4] : "/tmp", argc > 5 ? argv[5] : "1:1");
    }
    else if (cmd == "bench") rc = bench(c);
    else if (cmd == "dump") {
      if (argc < 4) throw wg::Error("dump <variant> <dir>");
      rc = dump(c, argv[2], argv[3]);
    }
    else if (cmd == "same") {
      if (argc < 5 || std::string(argv[2]) != "vote") throw wg::Error("same vote <variant> <spv>");
      rc = sameVote(c, argv[3], argv[4]);
    }
    else if (cmd == "replay") {
      if (argc < 3) throw wg::Error("replay <run> [frames] [variant]");
      rc = replay(c, argv[2], argc > 3 ? atoi(argv[3]) : 0, pick(4));
    } else { std::cerr << "unknown command " << cmd << std::endl; return 2; }
    c.dev->saveCache();
    return rc;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
}
