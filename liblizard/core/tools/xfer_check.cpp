// ai: The native transfer's check: the stream liblizard/gen/xfer_stream.mjs wrote (a 9 MB file of
// ai: 3 chunks from the web's sender, 20% of blocks lost, frames repeated, blocks and frames reordered) through XferRx in
// ai: frames of 40 blocks, the file rebuilt byte-exact and its BLAKE3 root verified, every count against the web's own
// ai: chain on the same stream (ref.json); the same stream again after clear(); the stream with one wrong block (a chunk
// ai: rejected, its symbols banned, the counts against the web's again); and the SHAKE256 test stream, every frame
// ai: judged test, its one wrong block bad, nothing passed to the transfer. A reader thread polls progress() throughout.
//
//   xfer_check [dir]      dir: xfer_stream.mjs's output (default $LIZ_XFER, else build/xfer); exit 1 on any difference
#include "xfer_rx.h"

#include <json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

extern "C" {
#include "xfer.h"
}

using namespace lizard;
using json = nlohmann::json;

static std::vector<uint8_t> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Run { json got; XferProgress p; double ms = 0; uint32_t maxQueued = 0; };

// ai: the stream in frames of 40, as xfer_stream.mjs's reference chain cut it
static Run feed(XferRx& rx, const std::vector<uint8_t>& s) {
  Run r;
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> maxQ{0};
  std::thread reader([&] {
    while (!stop) { auto p = rx.progress(); if (p.queued > maxQ) maxQ = p.queued; std::this_thread::sleep_for(std::chrono::microseconds(500)); }
  });
  const size_t n = s.size() / XFER_BLOCK_BYTES;
  int frames = 0, seen = 0, bad = 0, test = 0, judged = 0, fresh = 0;
  const double t = nowMs();
  for (size_t f = 0; f * 40 < n; f++) {
    std::vector<Block> fr(std::min<size_t>(40, n - f * 40));
    for (size_t k = 0; k < fr.size(); k++) memcpy(fr[k].bytes.data(), s.data() + (f * 40 + k) * XFER_BLOCK_BYTES, XFER_BLOCK_BYTES);
    auto v = rx.frame(fr);
    frames++; seen += v.seen; bad += v.bad; test += v.test; judged += v.judged; fresh += v.fresh;
  }
  const double fed = nowMs() - t;
  rx.drain();
  r.ms = nowMs() - t;
  stop = true;
  reader.join();
  r.maxQueued = maxQ;
  r.p = rx.progress();
  r.got = {{"frames", frames}, {"seen", seen}, {"bad", bad}, {"test", test}, {"judged", judged}, {"fresh", fresh}, {"forwarded", r.p.forwarded},
           {"doneAt", r.p.done ? json(r.p.doneAt) : json(nullptr)}, {"done", r.p.done}, {"verified", r.p.verified}, {"rejected", r.p.rejected},
           {"manifestRejected", r.p.manifestRejected}, {"refused", r.p.refused}, {"over", r.p.over}, {"fraction", r.p.fraction}};
  printf("  fed %.1f ms, drained %.1f ms (solves %.1f ms), queue at most %u, lost %llu\n", fed, r.ms, r.p.solveMs, r.maxQueued, (unsigned long long)r.p.lost);
  return r;
}

static int fails = 0;
static void compare(const char* what, const json& got, const json& ref) {
  printf("  %-18s %14s %14s\n", "", "native", "js");
  for (auto& [k, v] : got.items()) {
    const json& w = ref.contains(k) ? ref[k] : json(nullptr);
    const bool same = k == "fraction" ? w.is_number() && std::abs(v.get<double>() - w.get<double>()) < 1e-9 : v == w;
    if (!same) fails++;
    printf("  %-18s %14s %14s%s\n", k.c_str(), v.dump().c_str(), w.dump().c_str(), same ? "" : "   DIFFERS");
  }
  (void)what;
}
static void expect(bool ok, const std::string& what) { if (!ok) fails++; printf("  %-60s %s\n", what.c_str(), ok ? "ok" : "FAIL"); }

// ai: the rebuilt file byte-exact against the sent one, and its own BLAKE3 the header's root
static void fileChecks(const XferProgress& p, const std::vector<uint8_t>& file, const std::string& root) {
  auto out = slurp(p.path);
  uint8_t h[XFER_CV];
  xfer_b3_hash(out.data(), out.size(), h);
  char hex[2 * XFER_CV + 1];
  for (int i = 0; i < XFER_CV; i++) snprintf(hex + 2 * i, 3, "%02x", h[i]);
  expect(p.done && !p.path.empty(), "done, file at " + p.path);
  expect(out == file, "byte-exact: " + std::to_string(out.size()) + " of " + std::to_string(file.size()) + " B");
  expect(p.root == root && root == hex, "root verified: BLAKE3(file) = header root = sender root " + root.substr(0, 16));
  expect(p.name == "check-9MB.bin" && p.length == file.size() && p.chunks == 3, "header: " + p.name + ", " + std::to_string(p.length) + " B, " + std::to_string(p.chunks) + " chunks");
}

int main(int argc, char** argv) {
  const char* e = getenv("LIZ_XFER");
  const std::string dir = argc > 1 ? argv[1] : e ? e : "build/xfer";
  const auto stream = slurp(dir + "/stream.bin"), file = slurp(dir + "/file.bin"), test = slurp(dir + "/test.bin");
  std::ifstream rf(dir + "/ref.json");
  if (stream.empty() || file.empty() || test.empty() || !rf) { fprintf(stderr, "no stream in %s: node liblizard/gen/xfer_stream.mjs first\n", dir.c_str()); return 2; }
  const json ref = json::parse(rf);
  const std::string root = ref["root"];
  auto log = [](const std::string& s) { printf("  [log] %s\n", s.c_str()); };
  printf("stream: %zu blocks (%zu frames of 40), a lap %d blocks, joined at frame %d; file %zu B\n", stream.size() / XFER_BLOCK_BYTES,
         (stream.size() / XFER_BLOCK_BYTES + 39) / 40, ref["lap"].get<int>(), ref["join"].get<int>(), file.size());

  {
    XferRx rx(dir + "/store", log);
    printf("file stream:\n");
    Run r = feed(rx, stream);
    compare("file", r.got, ref["file"]);
    fileChecks(r.p, file, root);
    const std::string first = r.p.path;
    // ai: Clear, then the same light again: the dedupe and the worker start over, the finished file goes, and the file
    // ai: comes back at the same block. Progress is read straight after clear(), with no drain: clear() returns once the
    // ai: worker has cleared (2026-10-10), the app's poll reads it so.
    rx.clear();
    auto p0 = rx.progress();
    expect(!p0.header && !p0.done && !std::filesystem::exists(first), "clear(): no header, the finished file removed, read straight after it");
    printf("file stream again, after clear():\n");
    Run r2 = feed(rx, stream);
    fileChecks(r2.p, file, root);
    expect(r2.p.doneAt == r.p.doneAt && r2.got["fresh"] == r.got["fresh"], "done at the same block (" + std::to_string(r2.p.doneAt) + "), as many fresh");
  }
  if (const auto bad = slurp(dir + "/bad.bin"); !bad.empty()) {
    XferRx rx(dir + "/store-bad", log);
    printf("file stream with one wrong block of chunk 1:\n");
    Run r = feed(rx, bad);
    compare("bad", r.got, ref["bad"]);
    expect(r.p.rejected >= 1 && r.p.per.size() == 3 && r.p.per[1] < 254, "chunk 1 rejected, collected again from fresh ids");
  }
  {
    XferRx rx(dir + "/store-test", log);
    printf("test stream:\n");
    Run r = feed(rx, test);
    compare("test", r.got, ref["test"]);
    expect(r.got["test"] == r.got["frames"] && r.got["bad"] == 1 && r.p.forwarded == 0 && !r.p.header && r.p.held == 0,
           "every frame test, its one wrong block bad, none forwarded");
  }
  printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
