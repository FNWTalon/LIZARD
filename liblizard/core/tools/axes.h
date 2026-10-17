// ai: Where the camera's optical axis fell in each frame of a recording (the lens model's, liblizard/src/acquire.c
// ai: and gpu/wgsl/register.mjs LENS): meta.json's `axes` ([x, y] px, null the frame's centre) where the run has it
// ai: (Save replays since 2026-10-17), else from its layout: a 2:1 run's frames are the halves of one capture, the left
// ai: first (the pair shares its `ms`), the left's axis the middle of its right edge and the right's of its left
// ai: (core/rx/receiver.cpp axisOf, for a camera frame wider than high); else the centre. NaN for the centre.
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
  const bool ms = meta.contains("ms") && meta["ms"].is_array();
  for (int i = 0; i < n; i++) {
    const bool first = !ms || i == 0 || i >= (int)meta["ms"].size() || meta["ms"][i] != meta["ms"][i - 1];
    out[i] = {first ? (float)sizes[i].first : 0.0f, 0.5f * sizes[i].second};
  }
  return out;
}
