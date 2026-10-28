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

# ai: The wasm build, the codec's one WebAssembly module (2026-10-25; liblizard/build.sh drives it, or the "wasm" preset): no
# ai: threads and no GPU (the core leaves them out under emscripten), the codec's vector paths on WebAssembly SIMD, C++
# ai: exceptions as wasm's own (XferTx throws; every entry point catches), one ES module with its .wasm in LIZ_WASM_DIST.
# ai: It exports the C API and, beside it, the codec's own interface (src/wasm.c: the tree's pages, the native generator
# ai: and the lab through sim/ob.mjs; no stable interface), with Wirehair's and zstd's shims inside.
# ai: LIZ_WASM_MVP (liblizard/build.sh mvp, the "wasm-mvp" preset): the same module as WebAssembly 1.0 with no extension,
# ai: the library's fallback for an engine without SIMD (bindings/wasm loads it there): the codec's scalar paths (no
# ai: -msimd128, so src/simd.h defines no OB_SIMD), exceptions through JS, lizard-mvp.mjs and its .wasm. Emscripten's
# ai: libc carries sign extension, bulk memory and saturating float-to-int, which every browser it supports has:
# ai: binaryen lowers those three after the link, and the module is validated against the MVP feature set.
# ai: LIZ_SCALAR (the core's option, build.sh scalar) builds the codec's scalar twins in place of its vector paths, the
# ai: rest as the SIMD module, LIZ_VERIFY_POINTS (build.sh verify) the border finder's point scans beside the plain scan they
# ai: replaced (src/acquire.c OB_VERIFY_POINTS): lab builds for scripts/exp/simd_check.mjs and points_check.mjs.
option(LIZ_WASM_MVP "the wasm build as WebAssembly 1.0 (MVP): no SIMD, JS exceptions" OFF)
option(LIZ_VERIFY_POINTS "the border finder's point scans checked against the plain scan (a lab build)" OFF)
set(LIZ_WASM_DIST ${CMAKE_CURRENT_SOURCE_DIR}/bindings/wasm/dist CACHE PATH "where the wasm build writes its module")
if(EMSCRIPTEN)
  if(LIZ_WASM_MVP)
    add_compile_options(-fexceptions -mcpu=mvp)
    add_link_options(-fexceptions)
  else()
    add_compile_options(-fwasm-exceptions)
    add_link_options(-fwasm-exceptions)
    # ai: LIZ_SCALAR: no -msimd128, so src/simd.h defines no OB_SIMD and the scalar twins are built (OB_SCALAR leaves
    # ai: only the NEON paths out)
    if(NOT LIZ_SCALAR)
      add_compile_options(-msimd128)
    endif()
  endif()
endif()

# ai: the engine built for a shared library and hidden in it (a static wasm module has no use for position independence)
if(NOT EMSCRIPTEN)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
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
  add_executable(lizard ${LIZ_API_SOURCES} src/wasm.c)
  target_include_directories(lizard PRIVATE src)
  # ai: the C API, then the codec's own exports (src/wasm.c and what it reaches; an in-tree interface, no stability
  # ai: promise), then the allocator; Wirehair's and zstd's shims export themselves (EMSCRIPTEN_KEEPALIVE)
  set(LIZ_WASM_EXPORTS _liz_version _liz_last_error _liz_last_error_code _liz_simd _liz_ring_cells _liz_geometry_of _liz_room_for
      _liz_pick _liz_blocks_for _liz_encoder_new _liz_encoder_free _liz_encoder_geometry _liz_encoder_paint _liz_decoder_new
      _liz_decoder_free _liz_decoder_max_blocks _liz_decode _liz_layout_rects _liz_stream_fill _liz_tx_new
      _liz_tx_new_test _liz_tx_free _liz_tx_next _liz_tx_info_get _liz_rx_new _liz_rx_free _liz_rx_frame
      _liz_rx_progress _liz_rx_chunks _liz_rx_meta _liz_rx_data _liz_rx_clear
      _focus_setup _focus_setup_tiers _focus_any_setup _focus_any_rx _focus_any_held _focus_any_max_blocks _focus_any_which
      _focus_any_n _focus_any_blocks _focus_any_fmt _focus_any_builds _focus_bitmap_get _focus_subch
      _focus_blocks _focus_side _focus_cell _focus_quiet _focus_fmt_rx_ptr _focus_fmt_fps_set _focus_parity_set
      _focus_pilot_get _focus_any_pilot_get _focus_align_get _focus_tx _focus_tx_rgba _focus_rs_geom _focus_rx _focus_dbg
      _focus_dbg_sym _focus_dbg_coef _focus_blk_its _focus_blk_est _ob_heap_top
      _ob_last _ob_prof _ob_prof_reset _ob_luma_strip _ob_test_binarize _ob_test_sample _ob_test_marks
      _ob_test_mesh_tables_out _ob_test_fmt_plan _ob_test_fmt_cells _ob_test_thin_plan _ob_test_thin_score_out
      _ob_test_finder_out _focus_rx_acquire _focus_rx_acquire_quad _focus_rx_finish _focus_shape _focus_pos_ptr
      _focus_tables_out _focus_rx_ldpc _focus_block_subs _focus_ldpc_tables _focus_tiers_for _focus_set_axis _focus_set_lens
      _focus_lens_set _focus_anchors_found _focus_anchor_geom_get _focus_fmt_check_out _focus_fmt_soft_out _focus_grid_out _ob_test_crc_hash
      _ob_test_crc_gate_out _ob_dbg_hash _stream_fill _xfer_layout _xfer_id_of _xfer_kind_of _xfer_hdr_write _xfer_hdr_parse
      _xfer_mf_parse _xfer_chunk_ok _xfer_manifest_blocks _xfer_manifest_write _xfer_manifest_check _xfer_root
      _xfer_chunk_cv _xfer_b3_hash _xfer_b3_sub
      _malloc _free)
  string(REPLACE ";" "," LIZ_WASM_EXPORTS "${LIZ_WASM_EXPORTS}")
  set_target_properties(lizard PROPERTIES SUFFIX ".mjs" RUNTIME_OUTPUT_DIRECTORY ${LIZ_WASM_DIST})
  target_link_options(lizard PRIVATE -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1
      -sINITIAL_MEMORY=8388608 -sEXPORTED_FUNCTIONS=${LIZ_WASM_EXPORTS}
      -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPU32,HEAP32,HEAPF32,HEAPF64,UTF8ToString,stringToUTF8,lengthBytesUTF8)
  if(LIZ_SCALAR)
    target_compile_definitions(lizard PRIVATE OB_SCALAR)
  endif()
  if(LIZ_WASM_MVP)
    set_target_properties(lizard PROPERTIES OUTPUT_NAME lizard-mvp)
    find_program(LIZ_WASM_OPT wasm-opt HINTS "${EMSCRIPTEN_ROOT_PATH}/../bin" REQUIRED)
    set(LIZ_MVP_WASM ${LIZ_WASM_DIST}/lizard-mvp.wasm)
    add_custom_command(TARGET lizard POST_BUILD
      COMMAND ${LIZ_WASM_OPT} ${LIZ_MVP_WASM} --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int
              --enable-mutable-globals --llvm-memory-copy-fill-lowering --signext-lowering --llvm-nontrapping-fptoint-lowering
              -o ${LIZ_MVP_WASM}
      COMMAND ${LIZ_WASM_OPT} -q --mvp-features ${LIZ_MVP_WASM} -o ${CMAKE_CURRENT_BINARY_DIR}/mvp-validated.wasm
      COMMENT "lizard-mvp.wasm: sign extension, bulk memory and saturating float-to-int lowered; validated as MVP"
      VERBATIM)
  endif()
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
if(LIZ_VERIFY_POINTS)
  target_compile_definitions(lizard_codec PRIVATE OB_VERIFY_POINTS)
endif()
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
