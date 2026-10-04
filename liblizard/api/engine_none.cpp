// ai: Layer 4 where the engines are not built (a wasm build, or LIZ_GPU off): every call says so (LIZ_E_UNSUPPORTED),
// ai: so lizard.h means the same in every build.
#include "api.hpp"

using namespace liz;

static int none() { return fail(LIZ_E_UNSUPPORTED, "the engines are not in this build (no threads or no GPU targets)"); }

extern "C" {
LIZ_API liz_receiver* liz_receiver_new(const liz_receiver_config*, void (*)(void*, uint64_t)) { none(); return nullptr; }
LIZ_API void liz_receiver_free(liz_receiver*) {}
LIZ_API int liz_receiver_push(liz_receiver*, const uint8_t*, int, int, int, int64_t, uint64_t) { return none(); }
LIZ_API int liz_receiver_push_hardware_buffer(liz_receiver*, void*, int, int, int64_t, uint64_t) { return none(); }
LIZ_API int liz_receiver_wants_luma(liz_receiver*) { return none(); }
LIZ_API const char* liz_receiver_stats(liz_receiver*) { none(); return ""; }
LIZ_API int liz_receiver_series(liz_receiver*, double, double*, int) { return none(); }
LIZ_API void liz_receiver_soon(liz_receiver*, int) { none(); }
LIZ_API void liz_receiver_batch_cap(liz_receiver*, int) { none(); }
LIZ_API const char* liz_receiver_file(liz_receiver*) { none(); return ""; }
LIZ_API void liz_receiver_clear(liz_receiver*) { none(); }
LIZ_API void liz_receiver_camera_closed(liz_receiver*) { none(); }
LIZ_API liz_sender* liz_sender_new(const uint8_t*, size_t, const char*, const char*, int) { none(); return nullptr; }
LIZ_API void liz_sender_free(liz_sender*) {}
LIZ_API int liz_sender_configure(liz_sender*, const liz_format*, int, int, const char*, const char*) { return none(); }
LIZ_API int liz_sender_geometry(liz_sender*, liz_geometry*) { return none(); }
LIZ_API int liz_sender_ready(liz_sender*) { return none(); }
LIZ_API int liz_sender_take(liz_sender*, uint8_t*, int, liz_pixfmt) { return none(); }
LIZ_API const char* liz_sender_stats(liz_sender*) { none(); return ""; }
}
