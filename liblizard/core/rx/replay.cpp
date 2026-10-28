// ai: Save replays (replay.h): a frame a scratch file as it comes, the window of the newest `keep` held by deleting the
// ai: oldest, the run's .lizard written at its end.
#include "replay.h"

#include "json.hpp"
#include "lizfile.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

extern "C" {
#include "focus.h"
}

namespace lizard {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ai: how long the run's end waits for frames the GPU decoder took and has not handed over: a frame staged when the
// ai: camera stops launches once a lane is free (2026-10-08; before, at its bound of 1.5 batches of arrivals), then runs
constexpr auto HELD_WAIT = std::chrono::seconds(4);

// ai: A file and a scratch folder of its own: a file there already is refused, as is a scratch folder that holds anything,
// ai: never emptied (emptying it, a proposed fix for a reused folder, would have deleted whatever a replay was pointed at
// ai: by mistake: a run, the captures; 2026-10-03).
Replay::Replay(std::string file_, std::string run_, int keep_)
    : file(std::move(file_)), dir(file + ".frames"), run(std::move(run_)), keep(std::max(1, keep_)) {
  std::error_code ec;
  const fs::path p = fs::u8path(dir);
  if (fs::exists(fs::u8path(file), ec)) err = "the replay " + file + " exists already";
  else if (fs::exists(p, ec) && !fs::is_empty(p, ec)) err = "the replay's scratch folder " + dir + " holds files already";
  else {
    fs::create_directories(p, ec);
    if (ec) err = "the replay's folder: " + ec.message();
  }
  if (!err.empty()) { broken = true; dead = true; }
}

std::string Replay::path(uint64_t i) const {
  char b[32];
  snprintf(b, sizeof b, "/%04llu.gray", (unsigned long long)i);
  return dir + b;
}

// ai: mu held. A whole file, or none: a write the file system refuses removes what it left, and says why in err.
bool Replay::put(const std::string& p, const void* data, size_t n) {
  {
    std::ofstream o(fs::u8path(p), std::ios::binary | std::ios::trunc);
    if (o) o.write((const char*)data, (std::streamsize)n);
    if (o) o.close();
    if (o) return true;
  }
  err = "write " + fs::u8path(p).filename().string() + ": " + strerror(errno);
  std::error_code ec;
  fs::remove(fs::u8path(p), ec);
  return false;
}

bool Replay::live() { return !closing && !dead; }

bool Replay::hold(int n) {
  std::lock_guard<std::mutex> l(hm);
  if (closing || dead) return false;
  pending += n;
  return true;
}

void Replay::release(int n) {
  std::lock_guard<std::mutex> l(hm);
  pending = std::max(0, pending - n);
  cv.notify_all();
}

int Replay::frames() {
  std::lock_guard<std::mutex> l(mu);
  return (int)kept.size();
}

int64_t Replay::bytes() {
  std::lock_guard<std::mutex> l(mu);
  return held;
}

std::string Replay::error() {
  std::lock_guard<std::mutex> l(mu);
  return err;
}

void Replay::frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, double ms, bool isHeld, float ax, float ay, const float* lens) {
  // ai: a held frame's hold goes once it is written or refused, after mu is let go (the guard outlives the lock): let go
  // ai: first, the run's end could overtake the newest frame (2026-10-03)
  struct Done { Replay* r; bool on; ~Done() { if (on) r->release(1); } } done{this, isHeld};
  // ai: past the run's end only the frames held before it come in
  if (dead || (closing && !isHeld)) return;
  std::lock_guard<std::mutex> l(mu);
  if (ended || !err.empty()) return;
  const uint8_t* b = px;
  if (stride != w) {
    rowsBuf.resize((size_t)w * h);
    for (uint32_t y = 0; y < h; y++) memcpy(rowsBuf.data() + (size_t)y * w, px + (size_t)y * stride, w);
    b = rowsBuf.data();
  }
  if (!put(path(taken), b, (size_t)w * h)) { dead = true; return; }
  std::array<float, 4> ln{0, 0, 0, 0};
  if (lens) for (int i = 0; i < 4; i++) ln[i] = lens[i];
  kept.push_back({taken++, w, h, ms, ax, ay, ln});
  held += (int64_t)w * h;
  while ((int)kept.size() > keep) {
    std::error_code ec;
    fs::remove(fs::u8path(path(kept.front().i)), ec);
    held -= (int64_t)kept.front().w * kept.front().h;
    kept.pop_front();
  }
}

int Replay::finish(const std::string& rxStats, const std::string& rows, const std::string& more) {
  {
    std::unique_lock<std::mutex> h(hm);
    closing = true;
    cv.wait_for(h, HELD_WAIT, [&] { return pending <= 0; });
  }
  std::lock_guard<std::mutex> l(mu);
  if (ended) { if (err.empty()) err = "the run has ended"; return -1; }
  ended = true;
  if (broken) return -1;
  using ojson = nlohmann::ordered_json;
  const json st = json::parse(rxStats.empty() ? "{}" : rxStats, nullptr, false);
  const json word = st.is_object() && st.contains("word") ? st["word"] : json();
  // ai: the web's saveRun config, what a replay reads (scripts/exp/capture_check.mjs, through sim/phy.mjs recordedSpec): the
  // ai: format the light last named under today's rule, its stream the test stream's; null where no word was read
  ojson config;
  const int v = word.is_object() && word.contains("version") && word["version"].is_number() ? word["version"].get<int>() : 0;
  if (v > 0) {
    const int ring = word.contains("ring") && word["ring"].is_number() ? word["ring"].get<int>() : -1;
    const int r = ring >= 0 && ring < FOCUS_RINGS ? ring : FOCUS_RING_DEFAULT;
    config = {{"from", "band"}, {"spec", {{"phy", "focus"}, {"n", focus_n_for(8 * v)}, {"subch", 8 * v}, {"mode", 1}, {"span", 2 * FOCUS_RING[r]}, {"bitmap", 3}, {"stream", "shake256"}}}};
  }
  const ojson extra = ojson::parse(more.empty() ? "{}" : more, nullptr, false);
  // ai: The file: the frames oldest first, each scratch frame removed once it is in, so the disk holds no more than
  // ai: before as it is written (the file is smaller); a frame that does not go in (the disk full) ends the frames
  // ai: there, the file still finished with the ones before it, its meta saying why.
  std::string why;
  auto w = lizfile::Writer::create(file, why);
  std::error_code ec;
  if (!w) { err = why; fs::remove_all(fs::u8path(dir), ec); return -1; }
  std::string stopped = err, px;
  size_t n = 0;
  for (; n < kept.size(); n++) {
    const Kept& f = kept[n];
    std::ifstream in(fs::u8path(path(f.i)), std::ios::binary);
    px.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (px.size() != (size_t)f.w * f.h) { stopped = "read " + fs::u8path(path(f.i)).filename().string() + ": cut short"; break; }
    if (!w->frame((const uint8_t*)px.data(), f.w, f.h, f.w, why)) { stopped = why; break; }
    fs::remove(fs::u8path(path(f.i)), ec);
  }
  kept.resize(n);
  ojson sizes = ojson::array(), ms = ojson::array(), axes = ojson::array(), lenses = ojson::array();
  for (auto& f : kept) {
    sizes.push_back({f.w, f.h}); ms.push_back(f.ms); axes.push_back(std::isnan(f.ax) || std::isnan(f.ay) ? ojson() : ojson({f.ax, f.ay}));
    lenses.push_back(f.lens[3] > 0 ? ojson({f.lens[0], f.lens[1], f.lens[2], f.lens[3]}) : ojson());
  }
  ojson meta = {{"run", run}, {"w", kept.empty() ? 0 : kept[0].w}, {"h", kept.empty() ? 0 : kept[0].h}, {"frames", kept.size()}, {"of", keep},
                {"sizes", sizes}, {"ms", ms}, {"axes", axes}, {"lens", lenses}, {"taken", taken}, {"config", config},
                {"decoder", st.is_object() ? st.value("decoder", "") : ""}, {"layout", st.is_object() ? st.value("layout", "") : ""}};
  if (!stopped.empty()) meta["error"] = stopped;
  if (extra.is_object()) for (auto& [k, val] : extra.items()) meta[k] = val;
  std::string s = rows;
  if (!s.empty() && s.back() != '\n') s += '\n';
  const int64_t bytes = w->finish(meta.dump(1), s, why);
  w.reset();
  fs::remove_all(fs::u8path(dir), ec);
  err = stopped;
  if (bytes < 0) { err = stopped.empty() ? why : stopped + "; " + why; return -1; }
  held = bytes;
  return (int)kept.size();
}

}  // namespace lizard
