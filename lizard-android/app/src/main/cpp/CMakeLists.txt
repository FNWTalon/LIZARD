# ai: libliz.so, the app's native side: jni.cpp over liblizard/core's sender (lizard_tx), and the receive session
# ai: (lizard_session: the camera through Camera2's NDK, the receiver, the transfer, Save replays) behind the C API's
# ai: layer 5 (liblizard/api/session.cpp) and its JNI (liblizard/api/jni_session.cpp, registered by jni.cpp's
# ai: JNI_OnLoad), as the desktop app's library carries them: one JNI surface for receiving.
cmake_minimum_required(VERSION 3.22)
project(liz C CXX)
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../liblizard/core)
set(LZ ${CORE}/..)
# ai: EXCLUDE_FROM_ALL: only what libliz links is built (not core's check tools)
add_subdirectory(${CORE} core EXCLUDE_FROM_ALL)

add_library(liz SHARED jni.cpp ${LZ}/api/session.cpp ${LZ}/api/jni_session.cpp ${LZ}/api/jni_text.cpp)
target_compile_features(liz PRIVATE cxx_std_17)
target_compile_options(liz PRIVATE -Wall -Wextra -Wno-unused-parameter)
# ai: optimised in Gradle's debug variant too, as core is (liblizard/core/CMakeLists.txt, LIZ_O0)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LIZ_O0)
  target_compile_options(liz PRIVATE -O2)
endif()
target_include_directories(liz PRIVATE ${CORE}/rx ${LZ}/api)
# ai: the receive session (liblizard/core/rx/session.h, 2026-10-31; the receiver, lizard_rx, and Save replays with it)
target_link_libraries(liz PRIVATE lizard_session)
# ai: the sender (liblizard/core/tx, 2026-10-01): the C on the CPU or the GPU painting, and the transfer's sending end
target_link_libraries(liz PRIVATE lizard_tx)
# ai: android: the surfaces; 16 KB pages (Android 15's devices load nothing aligned below them)
target_link_libraries(liz PRIVATE android log)
target_link_options(liz PRIVATE -Wl,-z,max-page-size=16384)
