#!/usr/bin/env bash
# ai: The Android decoder's build steps; core/, gen/, out/ and .tools/ are liblizard's:
#   ./build.sh tools     naga-cli (its version line) and ../liblizard/gen/wgsl2spv (naga 30 as a library), once
#                        (cargo: Rust 1.87 or later)
#   ./build.sh gen       the setup from the web host: ../liblizard/out/{setup,blobs,wgsl,naga,spv}, and with LIZ_NETS=1
#                        the native nets (gen/nets.mjs: out/nets; glslang 16, placed by hand in ../liblizard/.tools/glslang).
#                        Needs Node, ../liblizard/build/ob.wasm (liblizard's ./build.sh), naga and wgsl2spv (tools), and
#                        spirv-val and glslc: the NDK's shader-tools (ANDROID_NDK_HOME, else $ANDROID_HOME/ndk/27.1.12297006,
#                        else ~/Android/Sdk/ndk/27.1.12297006),
#                        else SPIRV-Tools and shaderc on the PATH
#   ./build.sh linux     ../liblizard/core for this desktop: build/linux/lizard_gpu_check (the 4090, the iGPU, lavapipe)
#   ./build.sh android   ../liblizard/core for the phone with the NDK: build/android/lizard_gpu_check (arm64, API 29)
#   ./build.sh phone [cmd...]   the phone build and out/ pushed to /data/local/tmp/lizard over adb, then cmd run there
#                        (default: info); a selftest also pushes its frames and references from the lab's
#                        ../research/build, which is not published
#   ./build.sh apk       the receiver app (app/, Gradle): app/build/outputs/apk/debug/app-debug.apk, out/ as its assets
#   ./build.sh install   the app installed on the phone over adb, the camera granted (extra args go to Gradle)
set -euo pipefail
cd "$(dirname "$0")"
LIB=../liblizard; OUT=$LIB/out; RESEARCH=../research
NDK=${ANDROID_NDK_HOME:-${ANDROID_HOME:-$HOME/Android/Sdk}/ndk/27.1.12297006}
PHONE=/data/local/tmp/lizard
step=${1:-}; shift || true
case $step in
  tools)
    cargo install naga-cli --version 30.0.1 --root $LIB/.tools --locked
    (cd $LIB/gen/wgsl2spv && cargo build --release)
    ;;
  gen)
    node $LIB/gen/gen.mjs "$@"
    # ai: the native nets only when asked: they need $LIB/.tools/glslang, and none ships by default
    if [ "${LIZ_NETS:-0}" = 1 ]; then node $LIB/gen/nets.mjs "$@"; fi
    ;;
  linux)
    cmake -S $LIB/core -B build/linux -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
    cmake --build build/linux -j"$(nproc)"
    ;;
  android)
    cmake -S $LIB/core -B build/android -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-29 -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build/android -j"$(nproc)"
    ;;
  phone)
    adb get-state >/dev/null || { echo "no phone on adb (plug it in, USB debugging on)" >&2; exit 1; }
    adb shell mkdir -p $PHONE/out/spv $PHONE/research/build
    adb push build/android/lizard_gpu_check $PHONE/ >/dev/null
    adb push $OUT/setup $OUT/blobs $PHONE/out/ >/dev/null
    if [ -d $OUT/nets ]; then adb push $OUT/nets $PHONE/out/ >/dev/null; fi
    # ai: the SPIR-V the device loads, not naga's raw copies
    tmp=$(mktemp -d); cp $OUT/spv/*.spv "$tmp"/; rm -f "$tmp"/*.raw.spv; adb push "$tmp"/. $PHONE/out/spv/ >/dev/null; rm -rf "$tmp"
    if [ "${1:-}" = selftest ]; then
      adb push $RESEARCH/build/gpu_selftest_frames $RESEARCH/build/gpu_selftest_ref.json $RESEARCH/build/gpu_selftest_ref_int8.json $PHONE/research/build/ >/dev/null
    fi
    adb shell "cd $PHONE && LIZ_OUT=out LIZ_BARCODE=research LIZ_CACHE=$PHONE/pipeline.cache ./lizard_gpu_check ${*:-info}"
    ;;
  apk)
    ./gradlew assembleDebug "$@"
    ;;
  install)
    adb get-state >/dev/null || { echo "no phone on adb (plug it in, USB debugging on)" >&2; exit 1; }
    ./gradlew installDebug "$@"
    adb shell pm grant dev.lizard.receiver android.permission.CAMERA
    ;;
  *)
    sed -n 2,16p ./build.sh; exit 2   # ai: the script's own folder, where it has changed to
    ;;
esac
