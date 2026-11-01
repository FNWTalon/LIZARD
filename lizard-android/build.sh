#!/usr/bin/env bash
# ai: The Android decoder's build steps; core/, gen/, out/ and .tools/ are liblizard's:
#   ./build.sh tools     ../liblizard/gen/wgsl2spv (naga 30 as a library), once (cargo: Rust 1.87 or later); and
#                        glslang 16.6.0 into ../liblizard/.tools/glslang, the Khronos release build (Linux x86-64 or
#                        macOS), its sha256 checked, unless that glslang is there already (curl, tar)
#   ./build.sh gen       the setup from the JS host (liblizard/gpu/decoder.mjs): ../liblizard/out/{setup,blobs,wgsl,naga,glsl,spv}. Needs Node,
#                        ../liblizard/build/ob.mjs and ../liblizard/bindings/wasm/dist/lizard.wasm (liblizard's
#                        ./build.sh), wgsl2spv and glslang 16.6.0 (tools), and spirv-val and glslc: the NDK's
#                        shader-tools (ANDROID_NDK_HOME, else $ANDROID_HOME/ndk/27.1.12297006, else
#                        ~/Android/Sdk/ndk/27.1.12297006), else SPIRV-Tools and shaderc on the PATH
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
# ai: glslang for gen's GLSL modules (../liblizard/gen/gen.mjs GLSLANG): a module is named by its source, so the
# ai: compiler is pinned; the release's own builds, each checked by the sha256 the release states for it
GLSLANG=16.6.0
glslang() {
  local dir=$LIB/.tools/glslang asset sum tmp got
  [ "$("$dir/bin/glslang" --version 2>/dev/null | sed -n 's/^Glslang Version: [0-9]*://p')" = "$GLSLANG" ] && return 0
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) asset=linux-x86_64-release.tar.gz; sum=2c34071f56ecf39d16233294d3739cf43f981fd39080f2fd9dccb782101191ce ;;
    Darwin-*) asset=macos-universal-release.tar.gz; sum=4f8c05f2888a73ed0b7b274bc524af9130e09ae94b43c0f3f9ed83cfae852834 ;;
    *) echo "glslang $GLSLANG: no release build for $(uname -sm) here; put glslang $GLSLANG at $dir/bin/glslang" >&2; return 1 ;;
  esac
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 -o "$tmp/glslang.tar.gz" "https://github.com/KhronosGroup/glslang/releases/download/$GLSLANG/glslang-$GLSLANG-$asset"
  got=$( (sha256sum "$tmp/glslang.tar.gz" 2>/dev/null || shasum -a 256 "$tmp/glslang.tar.gz") | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then echo "glslang-$GLSLANG-$asset: sha256 $got, the release's is $sum" >&2; rm -rf "$tmp"; return 1; fi
  rm -rf "$dir"; mkdir -p "$dir"
  tar xzf "$tmp/glslang.tar.gz" -C "$dir"
  rm -rf "$tmp"
  echo "glslang $("$dir/bin/glslang" --version | sed -n 's/^Glslang Version: [0-9]*://p') in $dir"
}
case $step in
  tools)
    (cd $LIB/gen/wgsl2spv && cargo build --release)
    glslang
    ;;
  gen)
    node $LIB/gen/gen.mjs "$@"
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
    adb shell pm grant dev.fosslabs.lizard android.permission.CAMERA
    ;;
  *)
    sed -n 2,17p ./build.sh; exit 2   # ai: the script's own folder, where it has changed to
    ;;
esac
