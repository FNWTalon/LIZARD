#!/bin/sh
# ./build.sh          -> the codec's WebAssembly module (needs emsdk, source <emsdk>/emsdk_env.sh, and CMake 3.22 or later):
#   the library's wasm target (CMakeLists.txt; the vector paths on WebAssembly SIMD) in build-native/wasm, written to
#   bindings/wasm/dist/lizard.mjs and lizard.wasm: the C API and the codec's own exports (src/wasm.c), Wirehair and zstd
#   inside. build/ob.mjs, the path the tree imports the codec by (sim/ob.mjs, the native generator), re-exports it.
# ./build.sh mvp      -> the same module as WebAssembly 1.0 (dist/lizard-mvp.mjs and .wasm), the library's fallback for an
#   engine without SIMD
# ./build.sh scalar   -> build/nosimd/lizard.mjs: the module with the codec's scalar twins in place of its vector paths, for a
#   lab check that is not published (scripts/exp/simd_check.mjs)
# ./build.sh verify   -> build/verify and build/verify-scalar, for a lab check that is not published (scripts/exp/points_check.mjs)
# ./build.sh wirehair, ./build.sh zstd -> the default build: Wirehair and zstd are in the module (for build lines that
#   still name them)
# ./build.sh native   -> the C compiled natively under ASan and UBSan, warnings on, nothing kept (a compile check)
# ./build.sh app      -> ../lizard-web/app/, the installed web app (../lizard-web/pwa/build.mjs: content-addressed, its service worker; node only)
# -ffp-contract=off (every target, CMakeLists.txt and core/CMakeLists.txt): the reference is exact (2026-09-24), and a
# compiler that fuses a multiply and an add into one FMA rounds once where the source rounds twice, so an FMA machine's
# build would paint other bytes.
set -e
cd "$(dirname "$0")"
# CMake 3.22 or later for the module's builds: the PATH's where it is one, else fetched with pip into .tools/cmake (no
# root; a build image with Python and no CMake, as Cloudflare Pages' are, builds unattended)
if [ "$1" != app ] && [ "$1" != native ]; then
  v=$(cmake --version 2>/dev/null | sed -n 's/^cmake version \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')
  if [ -z "$v" ] || [ "$(echo "$v" | awk '{ print ($1 > 3 || ($1 == 3 && $2 >= 22)) }')" != 1 ]; then
    [ -x .tools/cmake/cmake/data/bin/cmake ] || python3 -m pip install -q --target .tools/cmake "cmake>=3.22"
    PATH="$PWD/.tools/cmake/cmake/data/bin:$PATH"; export PATH
  fi
fi
if [ "$1" = app ]; then
  # The installed app (2026-10-01, a content-hashed PWA): the codec's module as the default build left it, so build that
  # first when the C or a shim changed. An optional second argument names another out dir (relative to liblizard/).
  OUTDIR="${2:-../lizard-web/app}"
  case "$OUTDIR" in /*) ;; *) OUTDIR="$PWD/$OUTDIR" ;; esac
  exec node ../lizard-web/pwa/build.mjs "$OUTDIR"
fi
if [ "$1" = native ]; then
  # BLAKE3 for the transfer's hash (src/xfer.h): the official C, portable only, which is all wasm can run of it, so the
  # native check compiles the same paths. No threads.
  SRC="src/ldpc.c src/layout.c src/acquire.c src/fft.c src/rs.c src/fmt.c src/focus.c src/any.c src/shake.c src/xfer.c src/wasm.c vendor/blake3/blake3.c vendor/blake3/blake3_dispatch.c vendor/blake3/blake3_portable.c"
  B3="-Ivendor/blake3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 -DBLAKE3_USE_NEON=0"
  gcc -O2 -g -ffp-contract=off -Wall -Wextra -fsanitize=address,undefined $B3 -c $SRC && rm -f ./*.o
  exit 0
fi
# wasm <build tree> <out dir> <cache settings>: the module configured and built (emcmake, as the wasm presets' tree is);
# every setting given, so a tree's cache never carries another build's
wasm() {
  tree=$1 out=$2; shift 2
  mkdir -p "$out"
  emcmake cmake -S . -B "$tree" -DCMAKE_BUILD_TYPE=Release -DLIZ_TESTS=OFF -DLIZ_WASM_DIST="$out" "$@" > /dev/null
  cmake --build "$tree" -j "$(nproc 2>/dev/null || echo 4)"
}
case "$1" in
  ""|wirehair|zstd)
    wasm build-native/wasm "$PWD/bindings/wasm/dist" -DLIZ_WASM_MVP=OFF -DLIZ_SCALAR=OFF -DLIZ_VERIFY_POINTS=OFF
    mkdir -p build
    printf '%s\n%s\n' "// The codec's module (bindings/wasm/dist/lizard.mjs, liblizard/build.sh) at the path the tree imports it by." \
      'export { default } from "../bindings/wasm/dist/lizard.mjs";' > build/ob.mjs
    ls -l bindings/wasm/dist/lizard.wasm ;;
  mvp)
    wasm build-native/wasm-mvp "$PWD/bindings/wasm/dist" -DLIZ_WASM_MVP=ON -DLIZ_SCALAR=OFF -DLIZ_VERIFY_POINTS=OFF
    ls -l bindings/wasm/dist/lizard-mvp.wasm ;;
  scalar)
    wasm build-native/wasm-scalar "$PWD/build/nosimd" -DLIZ_WASM_MVP=OFF -DLIZ_SCALAR=ON -DLIZ_VERIFY_POINTS=OFF
    ls -l build/nosimd/lizard.wasm ;;
  verify)
    # The border finder's point scans next to the plain scan they replaced, which aborts on any difference: scripts/exp/points_check.mjs.
    wasm build-native/wasm-verify "$PWD/build/verify" -DLIZ_WASM_MVP=OFF -DLIZ_SCALAR=OFF -DLIZ_VERIFY_POINTS=ON
    wasm build-native/wasm-verify-scalar "$PWD/build/verify-scalar" -DLIZ_WASM_MVP=OFF -DLIZ_SCALAR=ON -DLIZ_VERIFY_POINTS=ON
    ls -l build/verify/lizard.wasm build/verify-scalar/lizard.wasm ;;
  *) echo "build.sh: no build $1 (none, mvp, scalar, verify, native, app; wirehair and zstd the default)" >&2; exit 2 ;;
esac
