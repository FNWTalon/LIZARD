# ai: liblizard, the native library (2026-10-03; include/lizard.h, api/, the engine in core/ beside it): one shared
# ai: library exporting the C API alone (LIZ_API), every engine object hidden in it. Presets in CMakePresets.json:
# ai: cmake --preset linux && cmake --build --preset linux && ctest --preset linux.
cmake_minimum_required(VERSION 3.22)
project(liblizard VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(LIZ_TESTS "the C API's tests" ON)
# ai: the Kotlin binding's JNI (api/jni.cpp) in the library: always on Android, else asked for (the "jvm" preset; the
# ai: JDK's headers from JAVA_HOME)
if(ANDROID)
  option(LIZ_JNI "the Kotlin binding's JNI in the library" ON)
else()
  option(LIZ_JNI "the Kotlin binding's JNI in the library" OFF)
endif()
# ai: LIZ_SAN=address,undefined (or thread): the library, the engine and the tests under that sanitizer
if(LIZ_SAN)
  add_compile_options(-fsanitize=${LIZ_SAN} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${LIZ_SAN})
endif()

# ai: The wasm build (bindings/wasm; emcmake, the "wasm" preset): no threads and no GPU (the core leaves them out under
# ai: emscripten), the codec's vector paths on WebAssembly SIMD, C++ exceptions as wasm's own (XferTx throws; every
# ai: entry point catches), one ES module with its .wasm.
if(EMSCRIPTEN)
  add_compile_options(-fwasm-exceptions -msimd128)
  add_link_options(-fwasm-exceptions)
endif()

# ai: the engine built for a shared library and hidden in it
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# ai: the engine: core/ beside this file
# ai: the tree's root named "." in what the compiler writes (debug info, __FILE__): no builder's home folder in a
# ai: built library (as lizard-desktop/native)
get_filename_component(LIZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_compile_options("-ffile-prefix-map=${LIZ_ROOT}=.")
add_subdirectory(core EXCLUDE_FROM_ALL)

set(LIZ_API_SOURCES api/api.cpp api/format.cpp api/codec.cpp api/xfer.cpp)
if(EMSCRIPTEN)
  add_executable(lizard ${LIZ_API_SOURCES})
  set(LIZ_WASM_EXPORTS _liz_abi _liz_version _liz_last_error _liz_last_error_code _liz_simd _liz_ring_cells _liz_geometry_of _liz_room_for
      _liz_pick _liz_encoder_new _liz_encoder_free _liz_encoder_geometry _liz_encoder_paint _liz_decoder_new
      _liz_decoder_free _liz_decoder_max_blocks _liz_decode _liz_layout_rects _liz_stream_fill _liz_tx_new
      _liz_tx_new_test _liz_tx_free _liz_tx_next _liz_tx_info_get _liz_rx_new _liz_rx_free _liz_rx_frame
      _liz_rx_progress _liz_rx_chunks _liz_rx_meta _liz_rx_data _liz_rx_clear _malloc _free)
  string(REPLACE ";" "," LIZ_WASM_EXPORTS "${LIZ_WASM_EXPORTS}")
  set_target_properties(lizard PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bindings/wasm/dist)
  target_link_options(lizard PRIVATE -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1
      -sINITIAL_MEMORY=33554432 -sEXPORTED_FUNCTIONS=${LIZ_WASM_EXPORTS}
      -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAP32,HEAPF32,HEAPF64,UTF8ToString,stringToUTF8,lengthBytesUTF8)
else()
  add_library(lizard SHARED ${LIZ_API_SOURCES})
  if(LIZ_JNI)
    if(NOT ANDROID)
      find_package(JNI REQUIRED)
      target_include_directories(lizard PRIVATE ${JNI_INCLUDE_DIRS})
    endif()
    target_sources(lizard PRIVATE api/jni.cpp)
  endif()
endif()
target_include_directories(lizard PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_definitions(lizard PRIVATE LIZ_BUILDING LIZ_VERSION_STRING="${PROJECT_VERSION}")
target_link_libraries(lizard PRIVATE lizard_codec lizard_xfer_core)
if(NOT EMSCRIPTEN)
  set_target_properties(lizard PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
endif()
if(WIN32)
  set_target_properties(lizard PROPERTIES PREFIX "")   # ai: lizard.dll, as the Kotlin binding's loader names it
endif()
if(NOT MSVC)
  target_compile_options(lizard PRIVATE -Wall -Wextra -Wno-unused-parameter -ffp-contract=off)
endif()
# ai: exactly the C API exported: the engine's own symbols (BLAKE3, Wirehair, the codec) and the standard library's
# ai: template instances kept out (libstdc++ marks its templates visible whatever the preset). On Windows only what
# ai: LIZ_API marks dllexport is exported.
if(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
  target_link_options(lizard PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/api/lizard.map)
  set_target_properties(lizard PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/api/lizard.map)
elseif(APPLE)
  target_link_options(lizard PRIVATE -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/api/lizard.exp)
elseif(MINGW)
  target_link_options(lizard PRIVATE -Wl,--exclude-libs,ALL)
endif()
# ai: layer 4, the engines (api/engine.cpp over core/rx/receiver.h and core/tx/sender.h): where the GPU targets are
# ai: built (LIZ_GPU, the core's option: on natively, off under emscripten); else stubs saying LIZ_E_UNSUPPORTED
if(LIZ_GPU AND NOT EMSCRIPTEN)
  target_sources(lizard PRIVATE api/engine.cpp)
  target_link_libraries(lizard PRIVATE lizard_rx lizard_tx)
else()
  target_sources(lizard PRIVATE api/engine_none.cpp)
endif()
install(TARGETS lizard LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/lizard.h DESTINATION include)
# ai: the engines' GPU files, installed under share/lizard (the config's assets): the decoder variants LIZ_GPU_VARIANTS
# ai: names (all six by default, 21 MB; f32-sg alone 10 MB with the sender's; the receiver takes the best the device
# ai: runs of those) and the sender's, out of the generated tree LIZ_GPU_TREE
if(LIZ_GPU AND NOT EMSCRIPTEN AND NOT ANDROID)
  include(cmake/gpu_files.cmake)
  set(LIZ_GPU_TREE ${CMAKE_CURRENT_SOURCE_DIR}/out CACHE PATH "the generated GPU tree (gen's out/)")
  set(LIZ_GPU_VARIANTS "int8-sg;int8;f16-sg;f16;f32-sg;f32" CACHE STRING "the decoder variants installed")
  if(EXISTS ${LIZ_GPU_TREE}/setup/send.json)
    liz_gpu_files(gpu_files ${LIZ_GPU_TREE} ${LIZ_GPU_VARIANTS})
    foreach(f ${gpu_files})
      get_filename_component(d ${f} DIRECTORY)
      install(FILES ${LIZ_GPU_TREE}/${f} DESTINATION share/lizard/${d})
    endforeach()
    list(LENGTH gpu_files n)
    message(STATUS "lizard: ${n} GPU files from ${LIZ_GPU_TREE} (${LIZ_GPU_VARIANTS})")
  else()
    message(STATUS "lizard: no GPU files to install, no generated tree at ${LIZ_GPU_TREE} (lizard-android/build.sh tools, then gen; then configure again)")
  endif()
endif()

if(LIZ_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  add_executable(lizard_roundtrip api/tests/roundtrip.c)
  target_link_libraries(lizard_roundtrip PRIVATE lizard)
  # ai: the web picker's table, where node and the codec's wasm are there to print it (build.sh)
  find_program(LIZ_NODE node)
  set(LIZ_PICK "")
  if(LIZ_NODE AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/ob.mjs)
    set(LIZ_PICK ${CMAKE_CURRENT_BINARY_DIR}/pick.txt)
    add_test(NAME pick_table COMMAND ${LIZ_NODE} ${CMAKE_CURRENT_SOURCE_DIR}/api/tests/pick_ref.mjs ${LIZ_PICK})
    set_tests_properties(pick_table PROPERTIES FIXTURES_SETUP pick)
  elseif(NOT LIZ_NODE)
    message(STATUS "lizard: no pick_table test (roundtrip's picker against the web sender's): no node")
  else()
    message(STATUS "lizard: no pick_table test (roundtrip's picker against the web sender's): no build/ob.mjs (liblizard/build.sh; then configure again)")
  endif()
  add_test(NAME roundtrip COMMAND lizard_roundtrip "${LIZ_PICK}" ${CMAKE_CURRENT_BINARY_DIR}/store)
  # ai: the engines on a recording and the GPU (api/tests/engine.c says how): run by hand, not by ctest
  add_executable(lizard_engine api/tests/engine.c)
  target_link_libraries(lizard_engine PRIVATE lizard)
  if(LIZ_PICK)
    set_tests_properties(roundtrip PROPERTIES FIXTURES_REQUIRED pick)
  endif()
endif()
