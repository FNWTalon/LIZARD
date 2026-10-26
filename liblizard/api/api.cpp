// ai: The library's own: its version, the last error, the codec's vector paths.
#include "api.hpp"

#include <cstring>

extern "C" {
#include "codec.h"
}

namespace liz {

// ai: The calling thread's last failure. The message is a fixed array, so no thread_local here has a destructor
// ai: (2026-10-25): under the Windows builds' toolchain (MinGW-w64 GCC 13, posix threads, emulated TLS), threads that
// ai: end holding a thread_local with a destructor corrupt the heap, as threads that had called the library did with
// ai: the std::string held here. A longer message is cut before a character, never inside one.
static thread_local char lastError[4096];
static thread_local int lastCode = LIZ_OK;

int fail(int code, const std::string& message) {
  size_t n = message.size() < sizeof lastError ? message.size() : sizeof lastError - 1;
  while (n < message.size() && n > 0 && (static_cast<unsigned char>(message[n]) & 0xC0) == 0x80) n--;
  std::memcpy(lastError, message.data(), n);
  lastError[n] = '\0';
  lastCode = code;
  return code;
}
void clearError() { lastError[0] = '\0'; lastCode = LIZ_OK; }

}  // namespace liz

extern "C" {

// ai: the library's release, which is also the format's (the header says why)
LIZ_API const char* liz_version(void) { return LIZ_VERSION_STRING; }
LIZ_API const char* liz_last_error(void) { return liz::lastError; }
LIZ_API int liz_last_error_code(void) { return liz::lastCode; }
LIZ_API int liz_simd(void) { return cpu_simd(); }

}  // extern "C"
