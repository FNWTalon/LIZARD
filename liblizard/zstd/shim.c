// zstd as the transfer uses it (shim.h): one frame a chunk, parameters pinned.
#define ZSTD_STATIC_LINKING_ONLY   // ZSTD_c_useRowMatchFinder and ZSTD_paramSwitch_e, static link only
#include "shim.h"
#include "zstd.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

EMSCRIPTEN_KEEPALIVE
uint32_t lizard_zstd_bound(uint32_t n) { return (uint32_t)ZSTD_compressBound(n); }

// Why the parameters are set and not left to zstd's defaults: the output must be the same bytes from every build.
// Level 9 (shim.h). The row match finder on: zstd's automatic choice turns it on by the window size, with a threshold
// that depends on whether the build has SSE2 or NEON, and the two match finders find different matches, so a chunk
// under 128 KB would compress differently in wasm and natively; forced on, every build uses it where its strategy
// allows (the SIMD and scalar paths of the finder itself agree). No checksum: BLAKE3 verifies the chunk. The content
// size in the frame, so a receiver refuses a frame of the wrong size before decompressing it. One thread.
EMSCRIPTEN_KEEPALIVE
int32_t lizard_zstd_compress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap) {
  ZSTD_CCtx *c = ZSTD_createCCtx();
  if (!c) return -1;
  int ok = !ZSTD_isError(ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, LIZARD_ZSTD_LEVEL)) &&
           !ZSTD_isError(ZSTD_CCtx_setParameter(c, ZSTD_c_useRowMatchFinder, ZSTD_ps_enable)) &&
           !ZSTD_isError(ZSTD_CCtx_setParameter(c, ZSTD_c_checksumFlag, 0)) &&
           !ZSTD_isError(ZSTD_CCtx_setParameter(c, ZSTD_c_contentSizeFlag, 1));
  size_t r = 0;
  if (ok) { r = ZSTD_compress2(c, dst, cap, src, n); ok = !ZSTD_isError(r); }
  ZSTD_freeCCtx(c);
  return ok && r <= 0x7fffffffu ? (int32_t)r : -1;
}

EMSCRIPTEN_KEEPALIVE
int32_t lizard_zstd_decompress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t len) {
  if (ZSTD_getFrameContentSize(src, n) != (unsigned long long)len) return -1;
  const size_t r = ZSTD_decompress(dst, len, src, n);
  return !ZSTD_isError(r) && r == len ? 0 : -1;
}
