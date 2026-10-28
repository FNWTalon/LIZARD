// ai: A recording as the tools read it (2026-10-28): a .lizard (rx/lizfile.h), named by its run under the lab's
// ai: research/ (LIZ_BARCODE)/replays/<run>.lizard, or by a path (one with a '/').
#pragma once
#include <memory>
#include <string>

#include "lizfile.h"

inline std::string recordingPath(const std::string& barcode, const std::string& run) {
  if (run.find('/') != std::string::npos) return run;
  const bool named = run.size() > 7 && run.compare(run.size() - 7, 7, ".lizard") == 0;
  return barcode + "/replays/" + run + (named ? "" : ".lizard");
}
// ai: null with why
inline std::unique_ptr<lizard::lizfile::Reader> openRecording(const std::string& barcode, const std::string& run, std::string& why) {
  return lizard::lizfile::Reader::open(recordingPath(barcode, run), why);
}
