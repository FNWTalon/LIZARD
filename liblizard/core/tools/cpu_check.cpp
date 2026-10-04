// ai: The CPU decoder's check: recorded frames through the native C (cpu/codec.h), on one thread or several.
//   lizard_cpu_check info
//   lizard_cpu_check replay <run> [frames] [threads] [seq|none]
// ai: replay: the frames of research/captures/v0.3/<run> that are there (a phone holds the first 96), each
// ai: decoded once. seq (one thread, the default there): the held word is the last word read in frame order, as the
// ai: web's blind replay carries it (sim/phy.mjs makeBlind), so the block total is comparable with
// ai: `ARM=blind node scripts/exp/capture_check.mjs`. none (the default on several threads): nothing held, so a frame's
// ai: result does not depend on the order frames finish in, and the digest over every frame's result (what it found,
// ai: its word, its blocks, each block's iterations and estimate) is the same at any thread count: the race check. On
// ai: one thread a second digest, "floats", is the codec's own hash of the floats and soft values behind each decode:
// ai: with the first, the vector build's check against the scalar one.
// ai: info runs on its own; replay reads a recording, the lab's research/ (LIZ_BARCODE), which is not published. A run
// ai: lizard_engine paint makes reads as a recording put at <LIZ_BARCODE>/captures/v0.3/<run>.
// ai: Env: LIZ_BARCODE (../research), LIZ_PASSES (1: timing passes over the frames), LIZ_NMAX (1024).
#include "codec.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::string envOr(const char* k, const char* d) { const char* v = getenv(k); return v ? v : d; }

static std::vector<uint8_t> readAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ai: the first integer after "key": in a small JSON text (meta.json's w, h, frames)
static int jsonInt(const std::string& s, const std::string& key) {
  const size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return 0;
  return atoi(s.c_str() + s.find(':', k) + 1);
}

struct FrameResult { cpu_frame_t fr{}; int got = 0; double ms = 0; uint64_t digest = 0; uint32_t deep = 0; };

static void fnv(uint64_t& h, const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
}

// ai: What a frame's decode produced, less its timings.
static uint64_t digestOf(const cpu_dec_t* d, const cpu_frame_t& fr, int got, const uint8_t* blocks, const uint8_t* ok, int bb) {
  uint64_t h = 1469598103934665603ull;
  const int head[8] = {fr.found, fr.ring, fr.n, fr.held, fr.word, fr.version, fr.fps, got};
  fnv(h, head, sizeof head);
  if (fr.found) fnv(h, fr.quad, sizeof fr.quad);
  if (fr.n) {
    fnv(h, ok, (size_t)fr.total);
    for (int b = 0; b < fr.total; b++) if (ok[b]) fnv(h, blocks + (size_t)b * bb, (size_t)bb);
    const int8_t* its; const float* est;
    cpu_dec_block_stats(d, &its, &est);
    if (its) fnv(h, its, (size_t)fr.total);
    if (est) fnv(h, est, (size_t)fr.total * sizeof(float));
  }
  return h;
}

static int replay(const std::string& run, int want, int threads, std::string mode) {
  const std::string dir = envOr("LIZ_BARCODE", "../research") + "/captures/v0.3/" + run;
  auto mb = readAll(dir + "/meta.json");
  if (mb.empty()) { fprintf(stderr, "no %s/meta.json\n", dir.c_str()); return 2; }
  const std::string meta(mb.begin(), mb.end());
  const int W = jsonInt(meta, "w"), H = jsonInt(meta, "h");
  int frames = jsonInt(meta, "frames");
  if (want > 0) frames = std::min(frames, want);
  // ai: each frame's size from meta's sizes where it has them (the web's format: a run may hold frames of several
  // ai: sizes, a tracked crop's or a camera reopened at another resolution), else the run's w x h
  const nlohmann::json mj = nlohmann::json::parse(meta, nullptr, false);
  std::vector<std::vector<uint8_t>> px;
  std::vector<std::pair<int, int>> dims;
  char name[32];
  for (int i = 0; i < frames; i++) {
    snprintf(name, sizeof name, "/%04d.gray", i);
    int w = W, h = H;
    if (mj.is_object() && mj.contains("sizes") && mj["sizes"].is_array() && i < (int)mj["sizes"].size() && mj["sizes"][i].is_array())
      { w = mj["sizes"][i][0].get<int>(); h = mj["sizes"][i][1].get<int>(); }
    auto b = readAll(dir + name);
    if (b.size() != (size_t)w * h) { if (!b.empty()) fprintf(stderr, "%s: %zu bytes for %d x %d: the frames stop there\n", name + 1, b.size(), w, h); break; }
    px.push_back(std::move(b));
    dims.push_back({w, h});
  }
  frames = (int)px.size();
  if (!frames) { fprintf(stderr, "no frames in %s\n", dir.c_str()); return 2; }
  if (mode.empty()) mode = threads == 1 ? "seq" : "none";
  if (mode == "seq" && threads != 1) { fprintf(stderr, "seq carries the held word in frame order: one thread\n"); return 2; }
  const int passes = std::max(1, atoi(envOr("LIZ_PASSES", "1").c_str())), nmax = atoi(envOr("LIZ_NMAX", "1024").c_str());
  std::vector<FrameResult> res(frames);
  std::vector<double> passMs;
  std::vector<double> prof(32, 0.0);
  std::atomic<bool> failed{false};
  for (int pass = 0; pass < passes; pass++) {
    const double t0 = nowMs();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++) th.emplace_back([&, t] {
      cpu_dec_t* d = cpu_dec_new(nmax);
      if (!d) { failed = true; return; }
      const int bb = cpu_dec_block_bytes(d), top = cpu_dec_top(d);
      std::vector<uint8_t> blocks((size_t)top * bb), ok(top);
      int held = 0;
      cpu_prof_reset();
      for (int i = t; i < frames; i += threads) {
        FrameResult r;
        if (threads == 1) cpu_hash_arm();
        const double a = nowMs();
        r.got = cpu_dec_frame(d, px[i].data(), dims[i].first, dims[i].second, mode == "seq" ? held : 0, blocks.data(), ok.data(), &r.fr);
        r.ms = nowMs() - a;
        if (r.fr.word) held = r.fr.version;
        r.digest = digestOf(d, r.fr, r.got, blocks.data(), ok.data(), bb);
        if (threads == 1) r.deep = cpu_hash_take();
        res[i] = r;
      }
      if (threads == 1) for (int k = 0; k < 32; k++) prof[k] = cpu_prof_ms(k);
      cpu_dec_free(d);
    });
    for (auto& x : th) x.join();
    passMs.push_back(nowMs() - t0);
  }
  if (failed) { fprintf(stderr, "the codec refused a decoder\n"); return 1; }
  long blocks = 0;
  int found = 0, words = 0, held = 0, finished = 0;
  uint64_t all = 1469598103934665603ull, deep = 1469598103934665603ull;
  std::vector<double> ms;
  for (auto& r : res) {
    blocks += r.got; found += r.fr.found != 0; words += r.fr.word != 0; held += r.fr.held != 0; finished += r.fr.n != 0;
    fnv(all, &r.digest, sizeof r.digest);
    fnv(deep, &r.deep, sizeof r.deep);
    ms.push_back(r.ms);
  }
  std::sort(ms.begin(), ms.end());
  std::sort(passMs.begin(), passMs.end());
  double sum = 0;
  for (double v : ms) sum += v;
  char deepText[40] = "";
  if (threads == 1) snprintf(deepText, sizeof deepText, ", floats %016llx", (unsigned long long)deep);
  printf("%s: %d frames, %d found, %d words, %d held, %d finished, %ld verified blocks; digest %016llx%s (%s, %s)\n", run.c_str(), frames, found, words, held, finished, blocks,
         (unsigned long long)all, deepText, mode.c_str(), cpu_simd() ? "vector" : "scalar");
  printf("  %d thread%s: %.2f ms a frame on its thread (median %.2f, slowest %.1f); a pass of the frames %.0f ms, %.1f frames a second\n", threads, threads == 1 ? "" : "s", sum / frames, ms[ms.size() / 2],
         ms.back(), passMs[passMs.size() / 2], 1000.0 * frames / passMs[passMs.size() / 2]);
  if (threads == 1) {
    // ai: internal.h PROF_*: 13 sample, 14 detrend, 15 transform, 16 LLR, 17 LDPC; 20 to 25 the finder's parts; 2, 3 the binarizer and the mesh
    const double per = 1.0 / frames;
    printf("  stages, ms a frame: binarize %.2f, finder (points %.2f, turn %.2f, fit %.2f, score %.2f, marks %.2f, track %.2f), mesh %.2f, sample %.2f, detrend %.2f, transform %.2f, LLR %.2f, LDPC %.2f\n",
           prof[1] * per, prof[20] * per, prof[21] * per, prof[22] * per, prof[23] * per, prof[24] * per, prof[25] * per, prof[3] * per, prof[13] * per, prof[14] * per, prof[15] * per, prof[16] * per, prof[17] * per);
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "info";
  if (cmd == "info") {
    cpu_dec_t* d = cpu_dec_new(1024);
    if (!d) { printf("the codec refused a decoder\n"); return 1; }
    printf("C decoder: blocks of %d bytes, %d a frame at most to n = 1024, %s paths, %u cores\n", cpu_dec_block_bytes(d), cpu_dec_top(d), cpu_simd() ? "vector" : "scalar", std::thread::hardware_concurrency());
    cpu_dec_free(d);
    return 0;
  }
  if (cmd == "replay" && argc > 2) return replay(argv[2], argc > 3 ? atoi(argv[3]) : 0, argc > 4 ? std::max(1, atoi(argv[4])) : 1, argc > 5 ? argv[5] : "");
  fprintf(stderr, "lizard_cpu_check info | replay <run> [frames] [threads] [seq|none]  (the run at $LIZ_BARCODE/captures/v0.3/<run>, ../research by default)\n");
  return 2;
}
