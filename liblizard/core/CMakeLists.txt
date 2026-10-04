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

add_library(dec STATIC dec/setup.cpp dec/front.cpp ingest/camera.cpp)
target_include_directories(dec PUBLIC dec ingest)
target_include_directories(dec SYSTEM PUBLIC ${TP})
target_link_libraries(dec PUBLIC wg)

add_executable(lizard_gpu_check tools/gpu_check.cpp)
target_link_libraries(lizard_gpu_check PRIVATE dec lizard_rx)
endif()

# ai: The transfer (rx/xfer_rx.h): src's xfer and shake, BLAKE3 portable with the defines build.sh gives it,
# ai: Wirehair and the web's own shim over it (wirehair/shim.cpp), compiled natively; vendor code without our warnings.
# ai: LZ is liblizard/, the folder above this one.
set(LZ ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(WH ${LZ}/vendor/wirehair)
set(XFER_VENDOR ${LZ}/vendor/blake3/blake3.c ${LZ}/vendor/blake3/blake3_dispatch.c ${LZ}/vendor/blake3/blake3_portable.c
  ${WH}/wirehair.cpp ${WH}/gf256.cpp ${WH}/WirehairCodec.cpp ${WH}/WirehairTools.cpp ${WH}/codec/WirehairV2Codec.cpp
  ${WH}/codec/WirehairV2GF16.cpp ${WH}/codec/WirehairV2Peel.cpp ${WH}/codec/WirehairV2Plan.cpp ${WH}/codec/WirehairV2Policy.cpp
  ${WH}/codec/WirehairV2Precode.cpp ${WH}/codec/WirehairV2PrecodeDecode.cpp ${WH}/codec/WirehairV2PrecodeEncode.cpp
  ${WH}/codec/WirehairV2Profile.cpp ${WH}/codec/WirehairV2Seeds.cpp ${WH}/codec/WirehairV2Solve.cpp ${LZ}/wirehair/shim.cpp)
# ai: lizard_xfer_core (2026-10-03): both ends of the transfer with no thread of their own (rx/xfer_rx_core.h, the
# ai: stores, tx/xfer_tx.h), what the library and a wasm build take; lizard_xfer adds the app's threaded receiving end.
add_library(lizard_xfer_core STATIC rx/xfer_rx_core.cpp rx/store.cpp tx/xfer_tx.cpp ${LZ}/src/xfer.c ${LZ}/src/shake.c ${XFER_VENDOR})
set_source_files_properties(${XFER_VENDOR} PROPERTIES COMPILE_OPTIONS "${LIZ_NOWARN}")
target_include_directories(lizard_xfer_core PUBLIC rx tx ${LZ}/src PRIVATE ${LZ}/vendor/blake3 ${WH}/include ${WH})
target_compile_definitions(lizard_xfer_core PRIVATE BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512 BLAKE3_USE_NEON=0 WIREHAIR_BUILDING=1)
if(NOT EMSCRIPTEN)
find_package(Threads REQUIRED)
target_link_libraries(lizard_xfer_core PUBLIC Threads::Threads)   # ai: std::call_once (Wirehair's init) on some C libraries
add_library(lizard_xfer STATIC rx/xfer_rx.cpp)
target_link_libraries(lizard_xfer PUBLIC lizard_xfer_core Threads::Threads)

add_executable(xfer_check tools/xfer_check.cpp)
target_include_directories(xfer_check SYSTEM PRIVATE ${TP})
target_link_libraries(xfer_check PRIVATE lizard_xfer)
endif()

# ai: The C codec natively (src/, the format's reference): the CPU decoder's (cpu/codec.h). -O3 as the wasm is
# ai: built; its vector paths come with the platform (NEON on arm64, src/simd.h), and LIZ_SCALAR=ON builds the scalar
# ai: twins in their place, for the check that the two give the same bytes.
option(LIZ_SCALAR "the C codec's scalar paths only" OFF)
set(CODEC ${LZ}/src)
add_library(lizard_codec STATIC cpu/codec.c ${CODEC}/focus.c ${CODEC}/acquire.c ${CODEC}/layout.c ${CODEC}/fmt.c ${CODEC}/rs.c
  ${CODEC}/ldpc.c ${CODEC}/fft.c ${CODEC}/any.c)
target_include_directories(lizard_codec PUBLIC cpu PRIVATE ${CODEC})
target_compile_options(lizard_codec PRIVATE ${LIZ_FAST})
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
target_link_libraries(lizard_cpu_check PRIVATE lizard_codec Threads::Threads)
endif()

# ai: the sender (tx/sender.h, 2026-10-01): the transfer's sending end (tx/xfer_tx.h) and the frames painted by the C,
# ai: or since 2026-10-02 on the GPU (tx/gpu_painter.h: the web's encoder kernels on wg, the tables from the C)
if(LIZ_GPU)
add_library(lizard_tx STATIC tx/sender.cpp tx/send_tables.cpp tx/gpu_painter.cpp)
target_include_directories(lizard_tx PUBLIC tx)
target_include_directories(lizard_tx SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_tx PUBLIC lizard_xfer lizard_codec wg Threads::Threads)

add_executable(tx_check tools/tx_check.cpp)
target_include_directories(tx_check SYSTEM PRIVATE ${TP})
target_link_libraries(tx_check PRIVATE lizard_tx lizard_xfer lizard_codec)

# ai: Save replays (rx/replay.h, 2026-10-03): the newest frames a receiver decodes, kept as a run's folder; its own
# ai: target so the app's stub receiver links it too
add_library(lizard_replay STATIC rx/replay.cpp)
target_include_directories(lizard_replay PUBLIC rx PRIVATE ${CODEC})
target_include_directories(lizard_replay SYSTEM PRIVATE ${TP})
target_link_libraries(lizard_replay PUBLIC lizard_codec)

# ai: the receiver (rx/receiver.h): a decoder (the GPU's, or the C on the CPU), the camera ingest and the transfer
# ai: behind one object, for the app
add_library(lizard_rx STATIC rx/receiver.cpp)
target_include_directories(lizard_rx PUBLIC rx)
target_link_libraries(lizard_rx PUBLIC dec lizard_xfer lizard_cpu lizard_replay)
endif()
