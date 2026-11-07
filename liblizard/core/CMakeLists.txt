# ai: The Android decoder's native core: wg (a WebGPU-shaped layer over Vulkan), the decoder
# ai: host (dec/), and the check tools (tools/). One project for android-arm64 (the NDK, from the app's Gradle or
# ai: lizard-android/build.sh phone) and linux-x86_64 (lizard-android/build.sh linux, the desktop's 4090, iGPU and lavapipe).
cmake_minimum_required(VERSION 3.22)
project(lizard_core C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(TP ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
# ai: -ffp-contract=off on every line, as the C codec is built (liblizard/build.sh): clang on arm64 fuses a*b+c otherwise.
# ai: GCC and clang (MinGW's included), or clang-cl on Windows (2026-10-03, the library's builds); MSVC's own compiler
# ai: has none of the builtins the C codec uses (__builtin_ctz and the like).
if(MSVC AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
  add_compile_options(/W3 /clang:-ffp-contract=off)
  set(LIZ_NOWARN /w)
  set(LIZ_FAST /O2)
elseif(MSVC)
  message(FATAL_ERROR "the C codec needs GCC or clang: build with clang-cl (-T ClangCL) or MinGW")
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -ffp-contract=off)
  set(LIZ_NOWARN -w)
  set(LIZ_FAST -O3)
endif()
# ai: The GPU's targets (wg, the decoder host, the GPU painter, the engines that hold them; 2026-10-03, for builds with
# ai: no Vulkan, wasm's above all): off, the codec and the transfer alone build.
if(EMSCRIPTEN)
  option(LIZ_GPU "build the Vulkan targets" OFF)
else()
  option(LIZ_GPU "build the Vulkan targets" ON)
endif()
# ai: Always optimised. Gradle's debug variant passes CMAKE_BUILD_TYPE=Debug, which is no -O at all: the app then ran
# ai: its Vulkan host, the transfer, Wirehair, BLAKE3 and the block judge's SHAKE256 unoptimised (found 2026-09-30;
# ai: only the codec carried its own -O3). A debug APK is for the Kotlin side, and -g stays. LIZ_O0=ON is the way back
# ai: for stepping through the native code.
option(LIZ_O0 "leave a Debug build unoptimised" OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LIZ_O0)
  add_compile_options($<IF:$<BOOL:${MSVC}>,/O2,-O2>)
  add_compile_definitions(NDEBUG)
endif()
# ai: No link-time optimisation: measured 2026-09-30 and removed (the C on one thread, 96 frames, identical digests:
# ai: the desktop 36.7 ms a frame either way, the S26 17.9 and 19.3 with it against 16.9 and 17.8 without).
# ai: LIZ_SAN=thread (or address): the whole build under that sanitizer, for the CPU pool's race check
if(LIZ_SAN)
  add_compile_options(-fsanitize=${LIZ_SAN} -g)
  add_link_options(-fsanitize=${LIZ_SAN})
endif()

if(LIZ_GPU)
# ai: Vulkan through volk (third_party/volk, 2026-10-03): loaded at run time, no libvulkan to link
add_library(wg STATIC wg/wg.cpp ${TP}/volk/volk.c)
set_source_files_properties(${TP}/volk/volk.c PROPERTIES COMPILE_OPTIONS "${LIZ_NOWARN}")
target_include_directories(wg PUBLIC wg)
target_include_directories(wg SYSTEM PUBLIC ${TP}/volk)
if(ANDROID)
  target_compile_definitions(wg PUBLIC VK_USE_PLATFORM_ANDROID_KHR=1)
  target_link_libraries(wg PUBLIC log android nativewindow)
else()
  target_include_directories(wg SYSTEM PUBLIC ${TP}/vulkan-headers)
  target_link_libraries(wg PUBLIC ${CMAKE_DL_LIBS})
endif()

add_library(dec STATIC dec/setup.cpp dec/front.cpp dec/trial.cpp ingest/camera.cpp ingest/develop.cpp)
target_include_directories(dec PUBLIC dec ingest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(dec SYSTEM PUBLIC ${TP})
# ai: the codec for its ring table: a setup is held to the rings the C reads (dec/front.cpp create)
target_link_libraries(dec PUBLIC wg lizard_codec)

add_executable(lizard_gpu_check tools/gpu_check.cpp)
target_link_libraries(lizard_gpu_check PRIVATE dec lizard_rx)
endif()

# ai: The transfer (rx/xfer_rx.h): src's xfer and shake, BLAKE3 with its vector kernels (below), Wirehair and the
# ai: web's own shim over it (wirehair/shim.cpp), zstd as one file under the project's shim (zstd/shim.c, the
# ai: parameters pinned; 2026-10-05), compiled natively; vendor code without our warnings.
# ai: LZ is liblizard/, the folder above this one.
set(LZ ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(WH ${LZ}/vendor/wirehair)
# ai: BLAKE3's vector kernels (2026-11-06), upstream's own, its dispatcher choosing at run time: on x86-64 SSE4.1 (in
# ai: the v2 floor) and AVX2 (by cpuid) as upstream's assembly (gcc 13's build of its intrinsics ran 1.5 and 3.0 GiB/s
# ai: on the 9950X, the assembly 2.2 and 4.3; the portable 1.36), no SSE2 (below the floor, never chosen) and no
# ai: AVX-512 (no v4); NEON on arm64; under wasm SIMD the NEON kernel through emscripten's NEON headers; the portable
# ai: code alone for the wasm MVP, LIZ_SCALAR and MSVC (no build of the tree's).
set(B3 ${LZ}/vendor/blake3)
set(B3_SRC ${B3}/blake3.c ${B3}/blake3_dispatch.c ${B3}/blake3_portable.c)
set(B3_DEFS BLAKE3_NO_SSE2 BLAKE3_NO_AVX512)
set(B3_NEON "")
if(LIZ_SCALAR OR LIZ_WASM_MVP OR MSVC)
  list(APPEND B3_DEFS BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_USE_NEON=0)
elseif(EMSCRIPTEN OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(B3_NEON ${B3}/blake3_neon.c)
  list(APPEND B3_SRC ${B3_NEON})
  list(APPEND B3_DEFS BLAKE3_USE_NEON=1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  enable_language(ASM)
  if(WIN32)
    set(B3_ABI windows_gnu)
  else()
    set(B3_ABI unix)
  endif()
  list(APPEND B3_SRC ${B3}/blake3_sse41_x86-64_${B3_ABI}.S ${B3}/blake3_avx2_x86-64_${B3_ABI}.S)
  list(APPEND B3_DEFS BLAKE3_USE_NEON=0)
else()
  list(APPEND B3_DEFS BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_USE_NEON=0)
endif()
set(XFER_VENDOR ${B3_SRC}
  ${LZ}/vendor/zstd/zstd.c ${LZ}/zstd/shim.c
  ${WH}/wirehair.cpp ${WH}/gf256.cpp ${WH}/WirehairCodec.cpp ${WH}/WirehairTools.cpp ${WH}/codec/WirehairV2Codec.cpp
  ${WH}/codec/WirehairV2GF16.cpp ${WH}/codec/WirehairV2Peel.cpp ${WH}/codec/WirehairV2Plan.cpp ${WH}/codec/WirehairV2Policy.cpp
  ${WH}/codec/WirehairV2Precode.cpp ${WH}/codec/WirehairV2PrecodeDecode.cpp ${WH}/codec/WirehairV2PrecodeEncode.cpp
  ${WH}/codec/WirehairV2Profile.cpp ${WH}/codec/WirehairV2Seeds.cpp ${WH}/codec/WirehairV2Solve.cpp ${LZ}/wirehair/shim.cpp)
# ai: lizard_xfer_core (2026-10-03): both ends of the transfer with no thread of their own (rx/xfer_rx_core.h, the
# ai: stores, tx/xfer_tx.h; the sender alone runs workers of its own, a physical core each, that make its chunks while
# ai: frames go and are joined when it goes, none under emscripten), what the library and a wasm build take;
# ai: lizard_xfer adds the app's threaded receiving end.
add_library(lizard_xfer_core STATIC rx/xfer_rx_core.cpp rx/store.cpp tx/xfer_tx.cpp ${LZ}/src/xfer.c ${LZ}/src/shake.c ${XFER_VENDOR})
set_source_files_properties(${XFER_VENDOR} PROPERTIES COMPILE_OPTIONS "${LIZ_NOWARN}")
if(EMSCRIPTEN AND B3_NEON)
  set_property(SOURCE ${B3_NEON} APPEND PROPERTY COMPILE_OPTIONS -mfpu=neon)   # ai: emscripten's arm_neon.h, on wasm SIMD
endif()
target_include_directories(lizard_xfer_core PUBLIC rx tx ${LZ}/src ${LZ}/zstd PRIVATE ${LZ}/vendor/blake3 ${LZ}/vendor/zstd ${WH}/include ${WH})
target_compile_definitions(lizard_xfer_core PRIVATE ${B3_DEFS} WIREHAIR_BUILDING=1)
if(NOT EMSCRIPTEN)
find_package(Threads REQUIRED)
target_link_libraries(lizard_xfer_core PUBLIC Threads::Threads)   # ai: std::call_once (Wirehair's init) on some C libraries
add_library(lizard_xfer STATIC rx/xfer_rx.cpp)
target_link_libraries(lizard_xfer PUBLIC lizard_xfer_core Threads::Threads)

add_executable(xfer_check tools/xfer_check.cpp)
target_include_directories(xfer_check SYSTEM PRIVATE ${TP})
target_link_libraries(xfer_check PRIVATE lizard_xfer)
endif()

# ai: The camera's phase lock (rx/phase_lock.h, 2026-10-31: liblizard/kotlin/PhaseLock.kt in C++), depending on
# ai: nothing, for a receive session whose camera can delay one capture; phase_lock_check replays a trace of its calls
add_library(lizard_phase STATIC rx/phase_lock.cpp)
target_include_directories(lizard_phase PUBLIC rx)
if(NOT EMSCRIPTEN)
# ai: the threads' priority (sys/priority.h): the sender's, the camera's and the decoder's threads raise their own;
# ai: libdbus-1 (rtkit on Linux) loaded at run time
add_library(lizard_sys STATIC sys/priority.cpp)
target_include_directories(lizard_sys PUBLIC sys)
target_link_libraries(lizard_sys PUBLIC ${CMAKE_DL_LIBS})

add_executable(phase_lock_check tools/phase_lock_check.cpp)
target_link_libraries(phase_lock_check PRIVATE lizard_phase)
endif()

# ai: The C codec natively (src/, the format's reference): the CPU decoder's (cpu/codec.h). -O3 as the wasm is
# ai: built; its vector paths come with the platform (NEON on arm64, src/simd.h), and LIZ_SCALAR=ON builds the scalar
# ai: twins in their place, for the check that the two give the same bytes.
option(LIZ_SCALAR "the C codec's scalar paths only" OFF)
set(CODEC ${LZ}/src)
add_library(lizard_codec STATIC cpu/codec.c ${CODEC}/lizard_phy.c ${CODEC}/acquire.c ${CODEC}/layout.c ${CODEC}/fmt.c ${CODEC}/rs.c
  ${CODEC}/ldpc.c ${CODEC}/fft.c ${CODEC}/any.c)
target_include_directories(lizard_codec PUBLIC cpu PRIVATE ${CODEC})
target_compile_options(lizard_codec PRIVATE ${LIZ_FAST})
# ai: x86-64: the x86-64-v2 level (SSE4.2, POPCNT and below), the floor every native build is compiled for (2026-11-05;
# ai: Windows 11 24H2 asks the same), so the vector paths take src/simd.h's x86 form; a site's x86-64-v3 form (AVX2) is
# ai: chosen at run time (simd.h ob_v3); the scalar twins only under LIZ_SCALAR
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  if(MSVC)
    target_compile_options(lizard_codec PRIVATE /clang:-march=x86-64-v2)
  else()
    target_compile_options(lizard_codec PRIVATE -march=x86-64-v2)
  endif()
endif()
target_compile_definitions(lizard_codec PRIVATE NDEBUG)
if(LIZ_SCALAR)
  target_compile_definitions(lizard_codec PRIVATE OB_SCALAR)
endif()
if(NOT MSVC)
  target_link_libraries(lizard_codec PUBLIC m)
endif()

if(NOT EMSCRIPTEN)
# ai: the CPU decoder's threads (cpu/pool.h)
add_library(lizard_cpu STATIC cpu/pool.cpp)
target_include_directories(lizard_cpu PUBLIC cpu)
target_link_libraries(lizard_cpu PUBLIC lizard_codec Threads::Threads)

add_executable(lizard_cpu_check tools/cpu_check.cpp)
target_include_directories(lizard_cpu_check SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_cpu_check PRIVATE lizard_codec lizard_replay Threads::Threads)
endif()

# ai: the sender (tx/sender.h, 2026-10-01): the transfer's sending end (tx/xfer_tx.h) and the frames painted by the C,
# ai: or since 2026-10-02 on the GPU (tx/gpu_painter.h: the web's encoder kernels on wg, the tables from the C)
if(LIZ_GPU)
add_library(lizard_tx STATIC tx/sender.cpp tx/send_tables.cpp tx/gpu_painter.cpp)
target_include_directories(lizard_tx PUBLIC tx)
target_include_directories(lizard_tx SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_tx PUBLIC lizard_xfer lizard_codec wg lizard_sys Threads::Threads)

add_executable(tx_check tools/tx_check.cpp)
target_include_directories(tx_check SYSTEM PRIVATE ${TP})
target_link_libraries(tx_check PRIVATE lizard_tx lizard_xfer lizard_codec lizard_replay)

# ai: Save replays (rx/replay.h, 2026-10-03): the newest frames a receiver decodes, kept as a run's folder; its own
# ai: target, which the app's libliz.so links beside lizard_rx. With it a replay as one file (rx/lizfile.h, .lizard,
# ai: 2026-10-28), its frames through zstd (lizard_xfer_core's)
add_library(lizard_replay STATIC rx/replay.cpp rx/lizfile.cpp)
target_include_directories(lizard_replay PUBLIC rx PRIVATE ${CODEC} ${LZ}/vendor/zstd)
target_include_directories(lizard_replay SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_replay PUBLIC lizard_codec lizard_xfer_core)

# ai: the receiver (rx/receiver.h): a decoder (the GPU's, or the C on the CPU), the camera ingest and the transfer
# ai: behind one object, for the app
add_library(lizard_rx STATIC rx/receiver.cpp rx/stream.cpp)
target_include_directories(lizard_rx PUBLIC rx)
target_include_directories(lizard_rx SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_rx PUBLIC dec lizard_xfer lizard_cpu lizard_replay lizard_sys)
endif()

# ai: The receive session (rx/session.h, 2026-10-31: the desktop's RxSession and cameras moved into the engine; Android's
# ai: camera the same day, cam/android.cpp on Camera2's NDK): the cameras (cam/camera.h: each system's own backend, a
# ai: phone (not on Android, the phone end), an upload, the simulated camera of LIZ_CAMERA_SIM) and the session over
# ai: the receiver, which the C API's layer 5 (api/session.cpp), the apps, lizard_receive and cam_check share, the phase
# ai: lock (lizard_phase, rx/phase_host.h) fed from a camera that can delay a capture. lizard_api_static: the C API's
# ai: layers 1 to 3 (../api) as a static library for the simulated screen's painter (cam/scene.cpp: liz_tx_next,
# ai: liz_encoder_paint), LIZ_STATIC leaving LIZ_API empty; a shared library that holds the API itself pulls none of it.
# ai: TurboJPEG, libusb, Aravis (the GenICam cameras, cam/aravis_api.h; not on Android) and Media Foundation's functions
# ai: are loaded at run time; Media Foundation's GUIDs (Windows), AVFoundation's frameworks (macOS) and Camera2's and the
# ai: image reader's NDK libraries (Android) linked.
if(LIZ_GPU AND NOT EMSCRIPTEN)
  add_library(lizard_api_static STATIC ${LZ}/api/api.cpp ${LZ}/api/format.cpp ${LZ}/api/codec.cpp ${LZ}/api/xfer.cpp)
  target_include_directories(lizard_api_static PUBLIC ${LZ}/include)
  target_compile_definitions(lizard_api_static PUBLIC LIZ_STATIC PRIVATE LIZ_VERSION_STRING="static")
  target_link_libraries(lizard_api_static PUBLIC lizard_codec lizard_xfer_core)
  set(LIZ_CAM_SOURCES cam/camera.cpp cam/luma.cpp cam/jpeg.cpp cam/clock.cpp cam/sim.cpp cam/scene.cpp cam/upload.cpp)
  if(ANDROID)
    list(APPEND LIZ_CAM_SOURCES cam/android.cpp)
  else()
    list(APPEND LIZ_CAM_SOURCES cam/phone.cpp cam/aoa.cpp cam/aravis.cpp cam/trigger.cpp)
  endif()
  if(WIN32)
    list(APPEND LIZ_CAM_SOURCES cam/mf.cpp)
  elseif(APPLE)
    enable_language(OBJCXX)   # ai: AVFoundation's backend is Objective-C++, which this project does not enable itself
    list(APPEND LIZ_CAM_SOURCES cam/avf.mm)
    set_source_files_properties(cam/avf.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  elseif(UNIX AND NOT ANDROID)
    list(APPEND LIZ_CAM_SOURCES cam/v4l2.cpp)
  endif()
  add_library(lizard_cam STATIC ${LIZ_CAM_SOURCES})
  target_include_directories(lizard_cam PUBLIC cam PRIVATE ingest)
  target_include_directories(lizard_cam SYSTEM PRIVATE ${TP})
  target_compile_options(lizard_cam PRIVATE ${LIZ_FAST})
  find_package(Threads REQUIRED)
  # ai: lizard_replay: the .lizard reader (upload.cpp); lizard_rx's headers: stream.h, the phone's wire (phone.cpp)
  target_link_libraries(lizard_cam PUBLIC lizard_api_static lizard_replay lizard_rx Threads::Threads ${CMAKE_DL_LIBS})
  if(WIN32)
    # ai: Media Foundation's GUIDs and DirectShow's camera-control IIDs are static data (libmfuuid, libstrmiids); its
    # ai: functions are looked up in mfplat, mf and mfreadwrite at run time, so a library with them loads where Media
    # ai: Foundation is absent (Windows N without the Media Feature Pack, a Server without the feature)
    target_link_libraries(lizard_cam PUBLIC mfuuid strmiids ole32 uuid ws2_32 iphlpapi)   # ai: ws2_32, iphlpapi: the phone's stream
  elseif(APPLE)
    target_link_libraries(lizard_cam PUBLIC "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo" "-framework Foundation")
  elseif(ANDROID)
    target_link_libraries(lizard_cam PUBLIC camera2ndk mediandk nativewindow android log)
  endif()
  add_library(lizard_session STATIC rx/session.cpp)
  target_include_directories(lizard_session PUBLIC rx)
  target_include_directories(lizard_session SYSTEM PRIVATE ${TP})
  target_link_libraries(lizard_session PUBLIC lizard_cam lizard_rx lizard_replay lizard_phase)
endif()
