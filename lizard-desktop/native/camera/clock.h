// ai: The host's steady clock and a precise sleep on it (2026-10-25): every camera frame's arrival, the simulated
// ai: camera's and the fake devices' schedules are on this clock.
#pragma once
#include <cstdint>

namespace lizard::cam {

// ai: The host's steady clock in ns (std::chrono::steady_clock: CLOCK_MONOTONIC on Linux, QPC on Windows, libc++'s
// ai: monotonic clock on macOS).
int64_t steadyNs();
// ai: Back at or just after t (steadyNs's clock), never before it: the system's precise sleep to within `spinNs` of
// ai: it, then a spin on steadyNs. A t already past returns at once.
void sleepUntil(int64_t t, int64_t spinNs = 300'000);

}  // namespace lizard::cam
