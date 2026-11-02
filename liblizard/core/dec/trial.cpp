// ai: The forms' trial (trial.h).
#include "trial.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

extern "C" {
#include "lizard_phy.h"
}

namespace lizard {

namespace {
constexpr int TRIAL_BATCH = 8;   // ai: frames a launch at most (every picture size in one on a desktop GPU)
constexpr uint32_t ID_TAG = 0x7a000000u;   // ai: a known block's id: ID_TAG | frame << 8 | block
// ai: a known frame's least pixels a module, in its own paint: at one, the finder misses LIZARD-80 in the 256 ring at
// ai: some noises and not others (a camera never samples a module once, square on); and a paint at one shown at a whole
// ai: multiple is a picture of sharp pixel steps, which the GPU's read below the picture's size (TRANSFORM_DOWN) folds
// ai: into the lowest frequencies, a few percent of the first block's soft values turned (a camera's lens blurs such
// ai: steps away), so a frame takes a ring whose paint has this many
constexpr int MIN_PXM = 2;

double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t fnv(const std::vector<uint8_t>& b) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t c : b) { h ^= c; h *= 1099511628211ull; }
  return h;
}
std::string hex(uint64_t v) { char b[17]; snprintf(b, sizeof b, "%016llx", (unsigned long long)v); return b; }
std::string deviceKey(const wg::Device& d) {
  return d.name + " " + hex(((uint64_t)d.vendorId << 32) | d.deviceId) + " " + hex(d.driverVersion) + (d.driver.empty() ? "" : " " + d.driver);
}
struct Rng {
  uint64_t s;
  uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ull; }
  double unit() { return ((next() >> 11) + 1) * (1.0 / 9007199254740993.0); }   // ai: (0, 1]
};
std::string named(const std::string& v, const std::string& form) { return form.empty() ? v : v + "/" + form; }
}  // namespace

// ai: ---- the record ----

FormRecord::FormRecord(const std::string& dir, const wg::Device& d, Log log) : log_(std::move(log)) {
  const std::string key = deviceKey(d);
  if (!dir.empty()) {
    path_ = dir + "/gpu-forms.json";
    std::ifstream f(std::filesystem::u8path(path_), std::ios::binary);
    if (f) j_ = json::parse(f, nullptr, false);
  }
  if (!j_.is_object() || j_.value("device", "") != key || !j_["variants"].is_object()) j_ = {{"device", key}, {"variants", json::object()}};
  if (j_.contains("trial")) {
    const json t = j_["trial"];
    j_.erase("trial");
    if (t.is_array() && t.size() == 2 && t[0].is_string() && t[1].is_string())
      refuse(t[0], t[1], "its trial did not finish: the process ended in it (a hang the driver never gave back, or a crash)");
  }
}

void FormRecord::setup(const std::string& variant, const std::vector<uint8_t>& bytes) {
  const std::string h = hex(fnv(bytes));
  json& v = j_["variants"][variant];
  if (v.is_object() && v.value("setup", "") == h && v["forms"].is_object()) return;
  v = {{"setup", h}, {"forms", json::object()}};
  save();
}

bool FormRecord::passed(const std::string& variant, const std::string& form) const {
  const json& vs = j_["variants"];
  auto v = vs.find(variant);
  if (v == vs.end()) return false;
  auto f = (*v)["forms"].find(form);
  return f != (*v)["forms"].end() && *f == "pass";
}

bool FormRecord::refused(const std::string& variant, const std::string& form) const {
  const json& vs = j_["variants"];
  auto v = vs.find(variant);
  if (v == vs.end()) return false;
  auto f = (*v)["forms"].find(form);
  return f != (*v)["forms"].end() && *f != "pass";
}

void FormRecord::begin(const std::string& variant, const std::string& form) {
  j_["trial"] = {variant, form};
  save();
}

void FormRecord::pass(const std::string& variant, const std::string& form) {
  j_["variants"][variant]["forms"][form] = "pass";
  j_.erase("trial");
  save();
}

void FormRecord::refuse(const std::string& variant, const std::string& form, const std::string& why) {
  j_["variants"][variant]["forms"][form] = why;
  j_.erase("trial");
  save();
  if (log_) log_("trial: " + named(variant, form) + " refused: " + why);
}

void FormRecord::lost(const std::string& variant, const std::string& why) {
  if (j_.contains("trial")) { const json t = j_["trial"]; refuse(t[0], t[1], "hung or lost the device in its trial: " + why); }
  else refuse(variant, "", "hung or lost the device: " + why);
}

void FormRecord::cancel() {
  if (!j_.contains("trial")) return;
  j_.erase("trial");
  save();
}

std::vector<std::string> FormRecord::refusals() const {
  std::vector<std::string> out;
  for (auto& [v, e] : j_["variants"].items())
    for (auto& [f, o] : e["forms"].items()) if (o != "pass") out.push_back(named(v, f) + ": " + o.get<std::string>());
  return out;
}

// ai: written whole beside the file and renamed over it (std::filesystem::rename replaces on every system), the bytes
// ai: on the disk first: a trial's mark must survive the process it marks
void FormRecord::save() {
  if (path_.empty()) return;
  const std::string tmp = path_ + ".tmp", s = j_.dump(1);
#ifdef _WIN32
  FILE* f = _wfopen(std::filesystem::u8path(tmp).c_str(), L"wb");
#else
  FILE* f = fopen(tmp.c_str(), "wb");
#endif
  bool ok = f && fwrite(s.data(), 1, s.size(), f) == s.size() && fflush(f) == 0;
#ifdef _WIN32
  ok = ok && _commit(_fileno(f)) == 0;
#else
  ok = ok && fsync(fileno(f)) == 0;
#endif
  if (f) fclose(f);
  std::error_code e;
  if (ok) std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path_), e);
  if ((!ok || e) && log_) log_("trial: cannot write " + path_ + (e ? ": " + e.message() : ""));
}

// ai: ---- the known frames ----

// ai: Frame i: the top format of the setup's picture size i, in ring (i + 1) mod 3, or the next smaller ring where that
// ai: one's paint would have fewer than MIN_PXM pixels a module (the 256 ring's at 384 and 512: every ring at one size or
// ai: more), painted as the senders paint (lizard_encode, lizard_paint_grey: the border to the byte, whole pixels a module,
// ai: the quiet margin) with the pilots' count i mod 4 and the display rate 60, then gaussian noise of standard deviation
// ai: `noise` grey levels added (a fixed seed a frame), clipped to 0..255.
TrialFrames paintTrial(const json& pictures, double noise) {
  TrialFrames t;
  const double t0 = nowMs();
  for (size_t i = 0; i < pictures.size(); i++) {
    TrialFrames::Frame fr;
    fr.subch = pictures[i]["subch"].get<int>();
    const int n = pictures[i]["n"].get<int>();
    int ri = (int)((i + 1) % LIZARD_RINGS);
    while (ri > 0 && lizard_pxm_for(n, 2 * LIZARD_RING[ri]) < MIN_PXM) ri--;
    if (lizard_pxm_for(n, 2 * LIZARD_RING[ri]) < MIN_PXM) throw wg::Error("trial: no ring paints a " + std::to_string(n) + " picture at " + std::to_string(MIN_PXM) + " pixels a module");
    fr.ring = LIZARD_RING[ri];
    lizard_t f{};
    if (lizard_init(&f, n, fr.subch, 2.0f, 2 * fr.ring, 0.f)) { throw wg::Error("trial: no codec for LIZARD-" + std::to_string(fr.subch)); }
    if (f.block_bytes != XFER_BLOCK) { lizard_free(&f); throw wg::Error("trial: blocks of " + std::to_string(f.block_bytes) + " bytes"); }
    lizard_fmt_fps(&f, 60);
    lizard_parity(&f, (int)(i & 3));
    fr.blocks.resize(f.blocks);
    std::vector<uint8_t> blocks((size_t)f.blocks * XFER_BLOCK);
    for (int b = 0; b < f.blocks; b++) {
      auto& o = fr.blocks[b];
      const uint32_t id = ID_TAG | (uint32_t)i << 8 | (uint32_t)b;
      for (int k = 0; k < 4; k++) o[k] = (uint8_t)(id >> 8 * k);
      Rng r{0x9e3779b97f4a7c15ull ^ id};
      for (size_t k = 4; k < o.size(); k++) o[k] = (uint8_t)(r.next() >> 56);
      memcpy(blocks.data() + (size_t)b * XFER_BLOCK, o.data(), XFER_BLOCK);
    }
    std::vector<float> drive((size_t)f.px * f.px);
    lizard_encode(&f, blocks.data(), drive.data());
    fr.side = (uint32_t)(f.px + 2 * LIZARD_QUIET * f.pxm);
    fr.px.resize((size_t)fr.side * fr.side);
    lizard_paint_grey(&f, drive.data(), fr.px.data());
    lizard_free(&f);
    Rng r{0x5eed5eedull + i};
    for (auto& v : fr.px) {
      const double z = std::sqrt(-2.0 * std::log(r.unit())) * std::cos(6.283185307179586 * r.unit());
      v = (uint8_t)std::clamp<long>(std::lround(v + noise * z), 0, 255);
    }
    t.side = std::max(t.side, fr.side);
    t.frames.push_back(std::move(fr));
  }
  t.paintMs = nowMs() - t0;
  return t;
}

// ai: ---- a run ----

TrialRun trialRun(FrontHalf& fh, const TrialFrames& t) {
  TrialRun r;
  const double t0 = nowMs();
  fh.trialling = true;
  struct Off { FrontHalf& fh; ~Off() { fh.trialling = false; } } off{fh};
  const size_t n = t.frames.size();
  const size_t per = (size_t)std::max(1, std::min({fh.size(), TRIAL_BATCH, (int)n}));
  long its = 0, itsN = 0;
  for (size_t k = 0; k < n; k += per) {
    std::vector<Slot*> slots;
    for (size_t i = k; i < std::min(n, k + per); i++) {
      const auto& fr = t.frames[i];
      Slot* s = fh.enqueueLuma(fr.px.data(), fr.side, fr.side, fr.side, i);
      if (!s) throw wg::Error("trial: no free slot on the ring");
      slots.push_back(s);
    }
    BatchOut out = fh.finish(fh.run(slots, 0));
    for (size_t j = 0; j < slots.size(); j++) {
      const uint32_t i = (uint32_t)(k + j);
      const auto& fr = t.frames[i];
      const auto& fo = out.frames[j];
      std::vector<char> seen(fr.blocks.size(), 0);
      int wrong = 0;
      for (const auto& rec : fo.records) {
        const uint32_t b = rec.id & 0xff;
        if ((rec.id & ~0xffu) != (ID_TAG | i << 8) || b >= fr.blocks.size() || memcmp(rec.payload.data(), fr.blocks[b].data(), XFER_BLOCK) != 0) { wrong++; continue; }
        if (!seen[b]) { seen[b] = 1; r.verified++; its += rec.its; itsN++; }
      }
      const int missing = (int)std::count(seen.begin(), seen.end(), 0);
      r.painted += (int)fr.blocks.size();
      r.wrong += wrong;
      if ((missing || wrong) && r.why.empty()) {
        char m[240];
        snprintf(m, sizeof m, "LIZARD-%d in the %d ring: %d of %zu blocks missing, %d verified wrong (found %d, version %d)", fr.subch, fr.ring, missing, fr.blocks.size(), wrong, (int)fo.finder.found, fo.version);
        r.why = m;
      }
    }
  }
  r.ok = r.verified == r.painted && r.wrong == 0;
  r.its = itsN ? (double)its / itsN : 0;
  r.ms = nowMs() - t0;
  return r;
}

// ai: ---- a variant ----

// ai: The variant's LDPC forms in order, each with the float classifier, then the int8 twin with the first LDPC form
// ai: that passed; an outcome the record holds is taken from it, so the frames are painted (frames()) and the lanes
// ai: planned only when one runs. The variant is refused when no LDPC form passes (its other kernels are then what
// ai: fails). A run that throws wg::Lost leaves its mark in the record for the caller.
static bool trialVariant(FrontHalf& fh, FormRecord& rec, const std::function<const TrialFrames&()>& frames, const Log& log) {
  const std::string v = fh.variant;
  const auto names = fh.ldpcFormNames();
  const TrialFrames* t = nullptr;
  auto run = [&](const std::string& form) {
    if (!t) { t = &frames(); fh.plan(t->side, t->side); }
    rec.begin(v, form);
    const TrialRun r = trialRun(fh, *t);
    char m[200];
    snprintf(m, sizeof m, ": %d of %d blocks, %d wrong, %.2f LDPC iterations a block, %.0f ms", r.verified, r.painted, r.wrong, r.its, r.ms);
    log("trial: " + named(v, form) + m);
    if (r.ok) rec.pass(v, form);
    else rec.refuse(v, form, r.why);
    return r.ok;
  };
  int first = -1;
  for (int k = 0; k < (int)names.size(); k++) {
    if (!fh.ldpcBuilt[k]) continue;
    const std::string form = "ldpc:" + names[k];
    bool ok = !rec.refused(v, form);
    if (ok && !rec.passed(v, form)) { fh.useLdpc(k); ok = run(form); }
    if (!ok) { fh.refuseLdpc(k); continue; }
    if (first < 0) first = k;
  }
  if (first < 0) {
    if (!rec.refused(v, "")) rec.refuse(v, "", "no LDPC form read the known frames");
    if (t) fh.unplan();
    return false;
  }
  // ai: LIZ_LDPC_FORM=<k> (an A/B's pin, front.cpp create) holds where its form passed
  const char* lf = getenv("LIZ_LDPC_FORM");
  const int pin = lf && *lf ? atoi(lf) : -1;
  fh.ldpcBase = first;
  fh.useLdpc(pin >= 0 && pin < (int)fh.ldpcBuilt.size() && fh.ldpcBuilt[pin] ? pin : first);
  if (fh.hasTwin()) {
    const std::string form = "classifier:int8";
    bool ok = !rec.refused(v, form);
    if (ok && !rec.passed(v, form)) { fh.useTwin(true); ok = run(form); fh.useTwin(false); }
    if (!ok) fh.refuseTwin();
  }
  if (!rec.passed(v, "")) rec.pass(v, "");
  if (t) fh.unplan();
  return true;
}

bool buildTrialled(GpuDecoder& d, const DeviceMaker& make, ReadFile read, const std::vector<std::string>& variants, double budget, FormRecord& rec, Log log, std::string& why) {
  std::unique_ptr<TrialFrames> frames;
  for (const auto& v : variants) {
    rec.setup(v, read(setupName(v)));
    while (!rec.refused(v, "")) {
      try {
        if (!d.dev) d.dev = make();
        d.fh = FrontHalf::create(*d.dev, read, v, budget, log);
        auto paint = [&]() -> const TrialFrames& {
          if (!frames) {
            frames = std::make_unique<TrialFrames>(paintTrial(d.fh->S->tree["pictures"]));
            char m[120];
            snprintf(m, sizeof m, "trial: %zu known frames painted in %.0f ms (at most %u px a side)", frames->frames.size(), frames->paintMs, frames->side);
            log(m);
          }
          return *frames;
        };
        if (trialVariant(*d.fh, rec, paint, log)) return true;
        d.fh.reset();
      } catch (const wg::Lost& e) {
        // ai: what was in trial refused (else the variant), and the device made again
        rec.lost(v, e.what());
        d.fh.reset();
        d.dev.reset();
      } catch (const std::exception& e) {
        // ai: a setup that does not build here (memory, a pipeline) or the host's own error: the next variant, no
        // ai: outcome recorded
        rec.cancel();
        log("trial: " + v + " not built: " + e.what());
        d.fh.reset();
        break;
      }
    }
  }
  const auto refusals = rec.refusals();
  why = "no GPU variant read the known frames";
  for (const auto& r : refusals) why += "; " + r;
  return false;
}

}  // namespace lizard
