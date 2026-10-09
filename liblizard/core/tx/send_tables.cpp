#include "send_tables.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include "json.hpp"

namespace lizard {

std::string SendConsts::load(const std::string& manifest, const std::vector<uint8_t>& tabBytes) {
  nlohmann::json j;
  try { j = nlohmann::json::parse(manifest); } catch (const std::exception& e) { return std::string("the sender's manifest: ") + e.what(); }
  if (!j.contains("tab") || !j.contains("codes")) return "the sender's manifest predates the rate profile (no tab or codes): regenerate liblizard/out";
  slots = j["slots"]; tposeTile = j["tposeTile"]; rsvThreads = j["rsvThreads"]; rshThreads = j["rshThreads"];
  blockBytes = j["blockBytes"]; bitmap = j["bitmap"];
  const auto& p = j["paramsAt"];
  paramsSizes = p["sizes"]; paramsDims = p["dims"]; paramsPw = p["pw"]; paramsWords = p["words"];
  pw = j["pw"].get<std::vector<uint32_t>>();
  sizes = j["sizes"].get<std::vector<int>>();
  codeRate.clear(); codeSubs.clear();
  for (const auto& c : j["codes"]) { codeRate.push_back(c["rate"]); codeSubs.push_back(c["subs"]); }
  const size_t words = j["tab"]["words"];
  if (tabBytes.size() != 4 * words) return "the sender's TAB is " + std::to_string(tabBytes.size()) + " B, not " + std::to_string(4 * words);
  tab.resize(words);
  std::memcpy(tab.data(), tabBytes.data(), 4 * words);
  if (pw.size() != 120) return "the sender's pw words are not 120";
  if (slots < 1 + static_cast<int>(codeRate.size())) return "the paint's uniform holds " + std::to_string(slots) + " sizes, too few for the codes' tiers";
  return "";
}

namespace {
// ai: gpu/back/transform.mjs uvOf: the coefficient at pos p as (u, v), the C's half-plane block layout
void uvOf(int p, int n, int bw, int& u, int& v) {
  const int nb = n * bw;
  v = (p / nb) * bw + ((p % nb) % bw);
  const int uw = (p % nb) / bw;
  u = uw < n / 2 ? uw : uw - n;
}
uint32_t bitsOf(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
}  // namespace

float txClip() {
  static const float c = [] {
    const char* e = getenv("LIZ_CLIP");
    const float v = e ? strtof(e, nullptr) : 2.0f;
    return v >= 1.0f && v <= 4.0f ? v : 2.0f;
  }();
  return c;
}

std::string sendTables(const focus_t& f, const SendConsts& k, int frames, SendTables& t, int codes, int gap) {
  t = SendTables{};
  t.n = f.n; t.subch = f.subch; t.blocks = f.blocks; t.blockBytes = f.block_bytes; t.npos = f.subch * FOCUS_SUB;
  const int n = f.n, bw = f.bw;
  // ai: discLayout: row v holds u = ustart .. umaxP then -umaxN .. -1 (ustart 1 on v = 0), packed at off[v]
  std::map<int, std::vector<int>> byV;
  for (int i = 0; i < t.npos; i++) { int u, v; uvOf(f.pos[i], n, bw, u, v); byV[v].push_back(u); }
  if (byV.empty()) return "no coefficients";
  t.Vr = byV.rbegin()->first + 1;
  std::vector<uint32_t> off(t.Vr), np(t.Vr), un(t.Vr);
  t.rows.assign(2048, 0);
  if (4 * t.Vr > 2048) return "the disc has " + std::to_string(t.Vr) + " rows, past the uniform's 512";
  uint32_t count = 0;
  for (int v = 0; v < t.Vr; v++) {
    const std::vector<int> us = byV.count(v) ? byV[v] : std::vector<int>{};
    const int ustart = v == 0 ? 1 : 0;
    int maxP = ustart - 1, maxN = 0;
    for (int u : us) { if (u >= 0) maxP = std::max(maxP, u); else maxN = std::max(maxN, -u); }
    const std::set<int> have(us.begin(), us.end());
    bool ok = static_cast<int>(us.size()) == maxP + 1 - ustart + maxN;
    for (int u = ustart; u <= maxP && ok; u++) ok = have.count(u) != 0;
    for (int u = -maxN; u < 0 && ok; u++) ok = have.count(u) != 0;
    if (!ok) return "disc row v = " + std::to_string(v) + " is not contiguous";
    off[v] = count; np[v] = static_cast<uint32_t>(maxP + 1 - ustart); un[v] = static_cast<uint32_t>(maxN);
    t.rows[4 * v] = count; t.rows[4 * v + 1] = static_cast<uint32_t>(us.size()); t.rows[4 * v + 2] = np[v]; t.rows[4 * v + 3] = un[v];
    count += static_cast<uint32_t>(us.size());
  }
  if (static_cast<int>(count) != t.npos) return "the disc holds " + std::to_string(count) + " entries for " + std::to_string(t.npos);
  t.uv.resize(t.npos);
  for (int i = 0; i < t.npos; i++) {
    int u, v; uvOf(f.pos[i], n, bw, u, v);
    const int ustart = v == 0 ? 1 : 0;
    t.uv[i] = off[v] + (u >= 0 ? static_cast<uint32_t>(u - ustart) : np[v] + un[v] + static_cast<uint32_t>(u));
  }
  // ai: the twiddles as the web computes them, in double, kept as f32
  t.tw.resize(2 * n);
  for (int m = 0; m < n; m++) { const double th = -2 * M_PI * m / n; t.tw[2 * m] = static_cast<float>(std::cos(th)); t.tw[2 * m + 1] = static_cast<float>(std::sin(th)); }
  // ai: gpu/back/paint.mjs clipLimit at the codec's clip (src/focus.c's level at tilt 0): fround(fround(2 clip) x
  // ai: fround(sqrt(fround(0.5 subch 320)))); its reciprocal halved in double
  const float lim = 2.0f * f.clip * std::sqrt(static_cast<float>(0.5 * f.subch * 320));
  t.su = {static_cast<uint32_t>(n), static_cast<uint32_t>(t.Vr), static_cast<uint32_t>(t.Vr * n), static_cast<uint32_t>(t.npos),
          static_cast<uint32_t>(n * n / 4), 0, 0, 0, bitsOf(lim), bitsOf(static_cast<float>(0.5 / lim)), 0, 0};
  // ai: the resampler (the C's taps): q pixels from the square's first, six taps each from i0, wrapped mod n
  int geo[4];
  const int* i0 = nullptr;
  const float* w = nullptr;
  if (focus_resample_geom(&f, geo, &i0, &w)) return "the codec has no resampler tables";
  t.q = geo[0];
  if (geo[2] != n) return "the resampler's n " + std::to_string(geo[2]) + " is not the codec's " + std::to_string(n);
  t.taps.resize(12 * static_cast<size_t>(t.q));
  t.copy = true;
  for (int x = 0; x < t.q; x++) for (int s = 0; s < 6; s++) {
    t.taps[6 * x + s] = static_cast<uint32_t>((((i0[x] + s) % n) + n) % n);
    const float wt = w[6 * x + s];
    t.taps[6 * t.q + 6 * x + s] = bitsOf(wt);
    if (wt != (s == 2 ? 1.f : 0.f)) t.copy = false;
  }
  t.W = f.px + 2 * FOCUS_QUIET * f.pxm;
  t.sq = geo[1] + FOCUS_QUIET * f.pxm;
  if (codes < 1 || codes > 2) return "codes are 1 or 2, not " + std::to_string(codes);
  if (gap < 0 || gap > 64) return "the gap is 0 to 64 modules, not " + std::to_string(gap);
  t.codes = codes;
  t.gap = codes > 1 ? gap * f.pxm : 0;
  t.FW = codes * t.W + (codes - 1) * t.gap; t.RW = (t.FW + 3) / 4;
  t.FS = 64 * ((t.RW * t.W + 63) / 64);
  t.g = {static_cast<uint32_t>(n), static_cast<uint32_t>(t.q), static_cast<uint32_t>(t.sq), static_cast<uint32_t>(t.W), static_cast<uint32_t>(codes),
         static_cast<uint32_t>(t.FW), static_cast<uint32_t>(t.RW), static_cast<uint32_t>(t.FS), 1, 0, static_cast<uint32_t>(t.gap), 0};
  // ai: the paint's uniform: block count at slot 0; each tier at slot 1 + its code's place in TAB (first block, first
  // ai: sub-channel, blocks; src/focus.c init lays the tiers out from the lowest frequencies in order); dims (blocks,
  // ai: npos, symbols an encode, parity | codes << 2: set an encode)
  if (f.bitmap != k.bitmap) return "the codec's bit map " + std::to_string(f.bitmap) + " is not the paint's " + std::to_string(k.bitmap);
  t.pu.assign(k.paramsWords, 0);
  t.pu[k.paramsSizes] = static_cast<uint32_t>(f.blocks);
  for (int q = 0, b = 0, sub = 0; q < f.tiers; q++) {
    int c = -1;
    for (size_t i = 0; i < k.codeRate.size(); i++) if (k.codeRate[i] == f.tier[q].rate && k.codeSubs[i] == f.tier[q].subs) c = static_cast<int>(i);
    if (c < 0) return "a tier at rate " + std::to_string(f.tier[q].rate) + " of " + std::to_string(f.tier[q].subs) + " sub-channels, which the paint has no code for";
    uint32_t* at = t.pu.data() + k.paramsSizes + 4 * (1 + c);
    if (at[2]) return "two tiers of one code";
    at[0] = static_cast<uint32_t>(b); at[1] = static_cast<uint32_t>(sub); at[2] = static_cast<uint32_t>(f.tier[q].blocks);
    b += f.tier[q].blocks; sub += f.tier[q].blocks * f.tier[q].subs;
  }
  t.pu[k.paramsDims] = static_cast<uint32_t>(f.blocks); t.pu[k.paramsDims + 1] = static_cast<uint32_t>(t.npos); t.pu[k.paramsDims + 2] = static_cast<uint32_t>(frames * codes);
  std::copy(k.pw.begin(), k.pw.end(), t.pu.begin() + k.paramsPw);
  // ai: the border: the C's paint of the symbol with every block zero (only the square depends on them)
  std::vector<uint8_t> zero(static_cast<size_t>(f.blocks) * f.block_bytes, 0), rgba(static_cast<size_t>(t.W) * t.W * 4);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  focus_encode(&f, zero.data(), drive.data());
  focus_paint_rgba(&f, drive.data(), rgba.data());
  t.border.assign(4 * ((static_cast<size_t>(t.W) * t.W + 3) / 4), 0);
  for (size_t i = 0; i < static_cast<size_t>(t.W) * t.W; i++) t.border[i] = rgba[4 * i];
  return "";
}

}  // namespace lizard
