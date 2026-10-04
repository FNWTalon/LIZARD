# BLAKE3, the official C implementation

The transfer's hash (`liblizard/src/xfer.h`): the file's root is what `b3sum` prints, and each chunk is checked against
its subtree chaining value.

From <https://github.com/BLAKE3-team/BLAKE3>, release 1.8.7 (2026-08-20), `c/` and `test_vectors/`, unmodified:
`https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.8.7.tar.gz`, sha256
`c6782a28842b1c0478524ac06a4f2ede784038ee298d6e2162c0b089c4306a3c`.

Kept: `blake3.c`, `blake3_dispatch.c`, `blake3_portable.c`, `blake3.h`, `blake3_impl.h`, the three licence files, and
`test_vectors/test_vectors.json`. Left out: the SSE, AVX and NEON kernels, the TBB file (threads) and the build files.
The portable build is all wasm can run of it, and `liblizard/build.sh` compiles it with `-DBLAKE3_NO_SSE2
-DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 -DBLAKE3_USE_NEON=0` so the native check compiles the same paths.

Licence: public domain under CC0 1.0 (`LICENSE_CC0`), or Apache 2.0 (`LICENSE_A2`), or Apache 2.0 with LLVM exceptions
(`LICENSE_A2LLVM`), at the user's choice.

The C has no call for a subtree's chaining value, so `liblizard/src/xfer.c` builds one on this code's own compression
function (`blake3_compress_in_place`, from `blake3_impl.h`). `liblizard/test/xfer_test.c` holds it to
`test_vectors.json`.
