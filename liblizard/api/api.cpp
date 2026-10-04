// ai: The library's own: its version, the last error, the codec's vector paths.
#include "api.hpp"

extern "C" {
#include "codec.h"
}

namespace liz {

static thread_local std::string lastError;
static thread_local int lastCode = LIZ_OK;

int fail(int code, const std::string& message) {
  lastError = message;
  lastCode = code;
  return code;
}
void clearError() { lastError.clear(); lastCode = LIZ_OK; }

}  // namespace liz

extern "C" {

LIZ_API int liz_abi(void) { return LIZ_ABI; }
// ai: the library's release, which is also the format's (the header says why)
LIZ_API const char* liz_version(void) { return LIZ_VERSION_STRING; }
LIZ_API const char* liz_last_error(void) { return liz::lastError.c_str(); }
LIZ_API int liz_last_error_code(void) { return liz::lastCode; }
LIZ_API int liz_simd(void) { return cpu_simd(); }

}  // extern "C"
