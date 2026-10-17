// ai: Where the camera's optical axis fell in each frame of a recording (the lens model's, liblizard/src/acquire.c
// ai: and gpu/wgsl/register.mjs LENS): meta.json's `axes` ([x, y] px, null the frame's centre) where the run has it
// ai: (Save replays since 2026-10-17), else from its layout: a 2:1 run's frames are the halves of one capture, the left
// ai: first (the pair shares its `ms`: a frame whose `ms` the next shares is a left, one the frame before shares a
// ai: right; a lone first frame a right, whose left the kept window cut off), the left's axis the middle of its right
// ai: edge and the right's of its left (core/rx/receiver.cpp axisOf, for a camera frame wider than high); else the
// ai: centre. NaN for the centre.
// ai: replayLens: meta.json's `lens`, the camera's report of its lens a frame ([k1, k2, k3, f] in the frame's px, null
// ai: none; Save replays since 2026-10-17), zeros where the run has none.
#pragma once
#include "json.hpp"
#include <array>
#include <cmath>
#include <utility>
#include <vector>

inline std::vector<std::array<float, 2>> replayAxes(const nlohmann::json& meta, const std::vector<std::pair<int, int>>& sizes) {
  const int n = (int)sizes.size();
  std::vector<std::array<float, 2>> out(n, {NAN, NAN});
  if (!meta.is_object()) return out;
  if (meta.contains("axes") && meta["axes"].is_array()) {
    for (int i = 0; i < n && i < (int)meta["axes"].size(); i++)
      if (meta["axes"][i].is_array() && meta["axes"][i].size() == 2) out[i] = {meta["axes"][i][0].get<float>(), meta["axes"][i][1].get<float>()};
    return out;
  }
  if (!(meta.contains("layout") && meta["layout"] == "2:1")) return out;
  const bool ms = meta.contains("ms") && meta["ms"].is_array() && (int)meta["ms"].size() >= n;
  for (int i = 0; i < n; i++) {
    bool first = true;
    if (ms) {
      const auto& t = meta["ms"];
      if (i + 1 < n && t[i] == t[i + 1]) first = true;
      else if (i > 0 && t[i] == t[i - 1]) first = false;
      else first = i > 0;
    }
    out[i] = {first ? (float)sizes[i].first : 0.0f, 0.5f * sizes[i].second};
  }
  return out;
}

inline std::vector<std::array<float, 4>> replayLens(const nlohmann::json& meta, int n) {
  std::vector<std::array<float, 4>> out(n, {0, 0, 0, 0});
  if (!meta.is_object() || !meta.contains("lens") || !meta["lens"].is_array()) return out;
  for (int i = 0; i < n && i < (int)meta["lens"].size(); i++) {
    const auto& l = meta["lens"][i];
    if (l.is_array() && l.size() == 4) out[i] = {l[0].get<float>(), l[1].get<float>(), l[2].get<float>(), l[3].get<float>()};
  }
  return out;
}
