// ai: cam_check (2026-10-25): the PC receiver's camera checks, synthetic all (no real camera is opened), one check a
// ai: run with its arguments, `cam_check <luma|scene|session|v4l2|mf|avf> [its arguments]`, or every one whole,
// ai: `cam_check all` (no arguments: each check names its own cases). Exit codes as checks.h's kHeld and the rest:
// ai: all's kFailed where any check gave other than kHeld, a check of another system's part (kNotHere) skipped and
// ai: said so.
// ai: GPU work inside them (the session's GPU decoder) runs under scripts/tools/gpulock.sh share; every figure is
// ai: the simulated camera's, never a real one's.
#include <cstdio>
#include <cstring>
#include <string>

#include "checks.h"

namespace {

struct Check { const char* name; int (*run)(const lizard::check::Args&); };
const Check kChecks[] = {
    {"luma", lizard::check::luma}, {"scene", lizard::check::scene},
    {"session", lizard::check::session}, {"v4l2", lizard::check::v4l2},
    {"mf", lizard::check::mf}, {"avf", lizard::check::avf},
};

int usage() {
  std::fprintf(stderr, "usage: cam_check <");
  for (size_t i = 0; i < sizeof kChecks / sizeof kChecks[0]; i++) std::fprintf(stderr, "%s|", kChecks[i].name);
  std::fprintf(stderr, "all> [arguments]; all takes none\n");
  return lizard::check::kUsage;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace lizard::check;
  if (argc < 2) return usage();
  const Args rest(argv + 2, argv + argc);
  if (!std::strcmp(argv[1], "all")) {
    if (!rest.empty()) return usage();
    int worst = kHeld;
    for (auto& c : kChecks) {
      std::printf("== %s\n", c.name);
      std::fflush(stdout);
      const int rc = c.run(rest);
      if (rc == kNotHere) {
        std::printf("skip %s: not on this system\n", c.name);
        std::fflush(stdout);
        continue;
      }
      // ai: anything but kHeld fails all: a case's failure, or a check that refused no arguments
      if (rc != kHeld) worst = kFailed;
    }
    return worst;
  }
  for (auto& c : kChecks)
    if (!std::strcmp(argv[1], c.name)) return c.run(rest);
  return usage();
}
