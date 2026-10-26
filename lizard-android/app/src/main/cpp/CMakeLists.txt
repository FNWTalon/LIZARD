# ai: libliz.so, the app's native side: jni.cpp over liblizard/core's receiver (lizard_rx, rx/receiver.h), sender and
# ai: replay.
cmake_minimum_required(VERSION 3.22)
project(liz C CXX)
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../liblizard/core)
# ai: EXCLUDE_FROM_ALL: only what libliz links is built (not core's check tools)
add_subdirectory(${CORE} core EXCLUDE_FROM_ALL)

add_library(liz SHARED jni.cpp)
target_compile_features(liz PRIVATE cxx_std_17)
target_compile_options(liz PRIVATE -Wall -Wextra -Wno-unused-parameter)
# ai: optimised in Gradle's debug variant too, as core is (liblizard/core/CMakeLists.txt, LIZ_O0)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LIZ_O0)
  target_compile_options(liz PRIVATE -O2)
endif()
target_include_directories(liz PRIVATE ${CORE}/rx)
target_link_libraries(liz PRIVATE lizard_rx)
# ai: the sender (liblizard/core/tx, 2026-10-01): the C on the CPU or the GPU painting, and the transfer's sending end
target_link_libraries(liz PRIVATE lizard_tx)
# ai: Save replays (liblizard/core/rx/replay.h, 2026-10-03)
target_link_libraries(liz PRIVATE lizard_replay)
# ai: android: AHardwareBuffer_fromHardwareBuffer; 16 KB pages (Android 15's devices load nothing aligned below them)
target_link_libraries(liz PRIVATE android log)
target_link_options(liz PRIVATE -Wl,-z,max-page-size=16384)
