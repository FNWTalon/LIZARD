// zstd for the transfer (src/xfer.h: a chunk goes as one zstd frame where that is shorter than its bytes), as every
// LIZARD build calls it: the parameters are pinned here, so the web's sender in wasm and the native senders write the
// same frame for the same chunk, and any receiver reads any sender's. Changing a parameter changes what senders paint,
// not what receivers can read: a zstd frame is a zstd frame.
#ifndef LIZARD_ZSTD_SHIM_H
#define LIZARD_ZSTD_SHIM_H
#include <stdint.h>

// The compression level, the one choice. zstd's levels trade time for size along a knee: from 3 to 9 about 7% fewer
// bytes for five times the work, from 9 to 19 about 5% fewer for twelve times. A sender compresses the whole file
// before its first frame (the header needs every chunk's size), so 9 keeps a phone's start to seconds on a hundred
// megabytes where 19 would be a minute, and gives up a few percent of bytes against it.
#define LIZARD_ZSTD_LEVEL 9

#ifdef __cplusplus
extern "C" {
#endif
// The most bytes a frame of n bytes can take (ZSTD_compressBound).
uint32_t lizard_zstd_bound(uint32_t n);
// One frame of src[0..n) into dst (cap bytes): the frame's bytes, or -1 where zstd refuses (cap too small).
int32_t lizard_zstd_compress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
// The len bytes a frame of n bytes holds, into dst: 0, or -1 where the frame does not say it holds len bytes, is
// damaged, or decompresses to anything but len.
int32_t lizard_zstd_decompress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t len);
#ifdef __cplusplus
}
#endif
#endif
