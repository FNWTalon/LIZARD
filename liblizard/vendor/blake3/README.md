# BLAKE3, the official C implementation

The transfer's hash (`liblizard/src/xfer.h`): the file's root is what `b3sum` prints, and each chunk is checked against
its subtree chaining value.

From <https://github.com/BLAKE3-team/BLAKE3>, release 1.8.7 (2026-08-20), `c/` and `test_vectors/`, unmodified:
`https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.8.7.tar.gz`, sha256
`c6782a28842b1c0478524ac06a4f2ede784038ee298d6e2162c0b089c4306a3c`.

Kept: `blake3.c`, `blake3_dispatch.c`, `blake3_portable.c`, `blake3.h`, `blake3_impl.h`, the vector kernels
`blake3_sse41_x86-64_unix.S`, `blake3_sse41_x86-64_windows_gnu.S`, `blake3_avx2_x86-64_unix.S`,
`blake3_avx2_x86-64_windows_gnu.S` and `blake3_neon.c`, the three licence files, and `test_vectors/test_vectors.json`.
Left out: the SSE2 kernels (SSE4.1 is in x86-64-v2, the floor of every native build, so the dispatcher never chooses
SSE2), the AVX-512 kernels (no build targets x86-64-v4), the kernels as C intrinsics for x86 and the MSVC assembly
(the GNU assembly is faster: on a Ryzen 9 9950X, gcc 13 built the intrinsics to 1.5 GiB/s for SSE4.1 and 3.0 for
AVX2, the assembly runs 2.2 and 4.3), the TBB file (threads) and the build files.

`liblizard/core/CMakeLists.txt` (the transfer's library) builds it per target, the dispatcher choosing at run time:
on x86-64 the SSE4.1 and AVX2 assembly (`_unix` on Linux and macOS, `_windows_gnu` under MinGW), AVX2 taken where
cpuid offers it; on arm64 `blake3_neon.c`; under WebAssembly SIMD `blake3_neon.c` through emscripten's NEON headers
(`-mfpu=neon`); the portable code alone for the WebAssembly 1.0 build, `LIZ_SCALAR` and MSVC. Every build defines
`BLAKE3_NO_SSE2` and `BLAKE3_NO_AVX512`.

Licence: public domain under CC0 1.0 (`LICENSE_CC0`), or Apache 2.0 (`LICENSE_A2`), or Apache 2.0 with LLVM exceptions
(`LICENSE_A2LLVM`), at the user's choice.

The C has no public call for a subtree's chaining value, so `liblizard/src/xfer.c` builds one on this code's own
internals from `blake3_impl.h`: `blake3_compress_subtree_wide` (the hasher's wide path, the kernels' lanes at once)
and `blake3_compress_in_place` for the nodes above it. `liblizard/test/xfer_test.c` holds it to `test_vectors.json`.
