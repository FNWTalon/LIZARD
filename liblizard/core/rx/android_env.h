// ai: The core's LIZ_* switches for a run driven over adb on Android: `adb shell setprop debug.lizard.env "LIZ_X=1
// ai: LIZ_Y=2"`, read as each receiver (the receive session's) and each sender (the app's) is made; the names set the
// ai: last time are unset first, so a property cleared takes effect at the next one (2026-10-07). One set of names a
// ai: process (an inline function's static). Android alone.
#pragma once
#if defined(__ANDROID__)
#include <android/log.h>
#include <stdlib.h>
#include <sys/system_properties.h>

#include <string>
#include <vector>

namespace lizard {

inline void applyDebugEnv() {
  static std::vector<std::string> set;
  for (const auto& k : set) unsetenv(k.c_str());
  set.clear();
  char v[PROP_VALUE_MAX] = "";
  __system_property_get("debug.lizard.env", v);
  const std::string all(v);
  for (size_t a = 0; a < all.size();) {
    size_t b = all.find(' ', a);
    if (b == std::string::npos) b = all.size();
    const std::string kv = all.substr(a, b - a);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos && eq > 0) {
      setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
      set.push_back(kv.substr(0, eq));
      __android_log_print(ANDROID_LOG_INFO, "lizard", "env %s", kv.c_str());
    }
    a = b + 1;
  }
}

}  // namespace lizard
#endif
