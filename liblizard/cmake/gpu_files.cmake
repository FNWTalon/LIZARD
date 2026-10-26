# ai: The GPU files the engines read (2026-10-03), out of the generated tree (out/: gen's setup, SPIR-V and blobs):
# ai: each decoder variant's setup and the modules and blobs it names (core/dec/setup.cpp reads objects.pipelines[].module
# ai: as spv/<module>.spv and objects.buffers[].blob as blobs/<blob>.bin), the camera's ingest kernel, the sender's setup
# ai: with its blob (tab.blob, the paint's codes and whitening) and its send_*.spv (core/tx/gpu_painter.cpp). Nothing
# ai: else of the tree: its raw SPIR-V and the benchmark's kernel are not read. The one copy of the rule: the library's
# ai: install takes it by the function, and a build that is not CMake's (the AAR, bindings/kotlin/lizard-android) runs
# ai: this file as a script for the list.
# ai:   liz_gpu_files(<out var> <tree> <variants...>): the files, relative to the tree
# ai:   cmake -DLIZ_GPU_TREE=<tree> "-DLIZ_GPU_VARIANTS=<v;v...>" -DLIZ_GPU_LIST=<file> -P gpu_files.cmake: the files
# ai:     written to <file>, one a line
function(liz_gpu_files out tree)
  set(files setup/send.json spv/ingest_camera.spv)
  file(GLOB sends RELATIVE ${tree} ${tree}/spv/send_*.spv)
  list(FILTER sends EXCLUDE REGEX "\\.raw\\.spv$")
  list(APPEND files ${sends})
  file(READ ${tree}/setup/send.json send)
  string(JSON tab GET "${send}" tab blob)
  list(APPEND files ${tab})
  foreach(v ${ARGN})
    if(NOT EXISTS ${tree}/setup/${v}.json)
      message(FATAL_ERROR "no GPU setup ${v} in ${tree}")
    endif()
    list(APPEND files setup/${v}.json)
    file(READ ${tree}/setup/${v}.json j)
    string(JSON pipes GET "${j}" objects pipelines)
    string(JSON np LENGTH "${pipes}")
    math(EXPR last "${np} - 1")
    foreach(i RANGE ${last})
      string(JSON id MEMBER "${pipes}" ${i})
      string(JSON m GET "${pipes}" ${id} module)
      list(APPEND files spv/${m}.spv)
    endforeach()
    string(JSON bufs GET "${j}" objects buffers)
    string(JSON nb LENGTH "${bufs}")
    math(EXPR last "${nb} - 1")
    foreach(i RANGE ${last})
      string(JSON id MEMBER "${bufs}" ${i})
      string(JSON type ERROR_VARIABLE none TYPE "${bufs}" ${id} blob)
      if(type STREQUAL "STRING")
        string(JSON b GET "${bufs}" ${id} blob)
        list(APPEND files blobs/${b}.bin)
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES files)
  set(${out} ${files} PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  if(NOT LIZ_GPU_TREE OR NOT LIZ_GPU_VARIANTS OR NOT LIZ_GPU_LIST)
    message(FATAL_ERROR "cmake -DLIZ_GPU_TREE=<tree> -DLIZ_GPU_VARIANTS=<v;v...> -DLIZ_GPU_LIST=<file> -P gpu_files.cmake")
  endif()
  if(NOT EXISTS ${LIZ_GPU_TREE}/setup/send.json)
    message(FATAL_ERROR "no generated GPU tree in ${LIZ_GPU_TREE}: lizard-android/build.sh tools, then gen")
  endif()
  liz_gpu_files(files ${LIZ_GPU_TREE} ${LIZ_GPU_VARIANTS})
  list(JOIN files "\n" text)
  file(WRITE ${LIZ_GPU_LIST} "${text}\n")
endif()
