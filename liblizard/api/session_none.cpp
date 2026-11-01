// ai: Layer 5 where the session is not built (a wasm build, LIZ_GPU off):
// ai: every call says so (LIZ_E_UNSUPPORTED), so lizard.h means the same in every build.
#include "api.hpp"

using namespace liz;

static int none() { return fail(LIZ_E_UNSUPPORTED, "the receive session is not in this build (no cameras here yet)"); }

extern "C" {
LIZ_API liz_session* liz_session_new(const liz_session_config*) { none(); return nullptr; }
LIZ_API void liz_session_free(liz_session*) {}
LIZ_API size_t liz_cameras(char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API size_t liz_cameras_stamp(char* out, size_t cap) { if (out && cap) *out = '\0'; none(); return 0; }
LIZ_API void liz_session_open(liz_session*, const char*) { none(); }
LIZ_API void liz_session_close(liz_session*) { none(); }
LIZ_API void liz_session_set(liz_session*, const char*, double, int) { none(); }
LIZ_API void liz_session_start(liz_session*, const char*) { none(); }
LIZ_API void liz_session_stop(liz_session*) { none(); }
LIZ_API void liz_session_reconfigure(liz_session*, const char*, const char*) { none(); }
LIZ_API void liz_session_batch_cap(liz_session*, int) { none(); }
LIZ_API size_t liz_session_stats(liz_session*, char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API size_t liz_session_camera(liz_session*, char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API int liz_session_preview(liz_session*, uint64_t, int, uint8_t*, size_t, uint32_t*, uint32_t*, uint64_t*) { return none(); }
LIZ_API void liz_session_forget(liz_session*, const char*) { none(); }
LIZ_API void liz_session_clear(liz_session*) { none(); }
LIZ_API void liz_session_replays(liz_session*, int, const char*) { none(); }
LIZ_API void liz_session_extra(liz_session*, const char*) { none(); }
LIZ_API size_t liz_kept(const char*, char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API int liz_kept_placed(const char*, const char*) { return none(); }
LIZ_API size_t liz_partials(const char*, char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API int liz_partial_delete(const char*, const char*) { return none(); }
LIZ_API int liz_session_preview_window(liz_session*, void*) { return none(); }
LIZ_API int liz_link(int, const char*) { return none(); }
LIZ_API int liz_link_pack(int) { return none(); }
LIZ_API size_t liz_link_state(char* json, size_t cap) { if (json && cap) *json = '\0'; none(); return 0; }
LIZ_API int liz_usb_accessory(int) { return none(); }
}
