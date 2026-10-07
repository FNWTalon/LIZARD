# ai: The desktop sender's native library (2026-10-03): liblizard/core's sender (lizard_tx: the transfer, the C's
# ai: painters, the Vulkan painter) behind a JNI face for the Kotlin app (lizard-desktop/src), and the presenter that
# ai: puts its frames on screen (presenter.h, 2026-10-04: the app's canvas through JAWT; lizard_present's own window).
cmake_minimum_required(VERSION 3.22)
project(lizard_desktop C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ai: the core's static libraries go into a shared one
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
# ai: the tree's root named "." in what the compiler writes (debug info, __FILE__), so a built library carries no
# ai: builder's home folder (2026-10-04: 56 such paths in the .deb's library), by its real path too (a symlinked tree)
get_filename_component(LIZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(LIZ_ROOT_REAL "${LIZ_ROOT}" REALPATH)
add_compile_options("-ffile-prefix-map=${LIZ_ROOT}=." "-ffile-prefix-map=${LIZ_ROOT_REAL}=.")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../liblizard/core ${CMAKE_CURRENT_BINARY_DIR}/core EXCLUDE_FROM_ALL)
if(NOT MSVC)
  set(LIZ_DESKTOP_WARN -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# ai: the presenter: X11 (XRandR for the mode's refresh; Composite and Shape to tell a composited window from one
# ai: shown direct) on Linux; Win32 under _WIN32 (built, not run yet)
if(UNIX AND NOT APPLE)
  find_package(X11 REQUIRED)
  if(NOT X11_Xrandr_FOUND)
    message(FATAL_ERROR "the presenter needs XRandR (libxrandr-dev)")
  endif()
  if(NOT X11_Xcomposite_FOUND OR NOT X11_Xshape_FOUND)
    message(FATAL_ERROR "the presenter needs Xcomposite and Xshape (libxcomposite-dev, libxext-dev)")
  endif()
endif()
add_library(lizard_presenter STATIC presenter.cpp)
target_include_directories(lizard_presenter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lizard_presenter PRIVATE ${LIZ_DESKTOP_WARN})
target_link_libraries(lizard_presenter PUBLIC lizard_tx)
if(UNIX AND NOT APPLE)
  target_compile_definitions(lizard_presenter PRIVATE LIZ_X11=1)
  target_include_directories(lizard_presenter SYSTEM PRIVATE ${X11_INCLUDE_DIR})
  target_link_libraries(lizard_presenter PUBLIC ${X11_X11_LIB} ${X11_Xrandr_LIB} ${X11_Xcomposite_LIB} ${X11_Xext_LIB})
endif()

# ai: the JNI library: JAWT for the canvas's window (JAVA_AWT_LIBRARY; the app loads jawt first). MinGW from Linux
# ai: (2026-10-04): the build JDK's jni.h and jawt.h with Windows' jni_md.h and jawt_md.h (win32/), JAWT_GetAWT looked
# ai: up in jawt.dll at run time, so no Windows JDK is needed to link; LIZ_JDK_INCLUDE names the build JDK's include,
# ai: JAVA_HOME's by default
if(MINGW)
  if(NOT LIZ_JDK_INCLUDE AND DEFINED ENV{JAVA_HOME})
    set(LIZ_JDK_INCLUDE $ENV{JAVA_HOME}/include)
  endif()
  if(NOT EXISTS "${LIZ_JDK_INCLUDE}/jawt.h")
    message(FATAL_ERROR "the JNI library needs a JDK's jni.h and jawt.h (JAVA_HOME a JDK, or -DLIZ_JDK_INCLUDE=<its include>)")
  endif()
  set(JNI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/win32 ${LIZ_JDK_INCLUDE})
  set(JAVA_AWT_LIBRARY "")
else()
  find_package(JNI REQUIRED)
endif()
add_library(lizard_desktop SHARED tx_jni.cpp)
if(WIN32)
  set_target_properties(lizard_desktop PROPERTIES PREFIX "")   # ai: lizard_desktop.dll, the name the Gradle bundle takes
endif()
target_include_directories(lizard_desktop SYSTEM PRIVATE ${JNI_INCLUDE_DIRS})
# ai: the JDK's include folders named "jdk" in the debug info too (a JDK in a home folder named it, as the tree's root did)
foreach(d IN LISTS JNI_INCLUDE_DIRS)
  get_filename_component(r "${d}" REALPATH)
  target_compile_options(lizard_desktop PRIVATE "-ffile-prefix-map=${d}=jdk" "-ffile-prefix-map=${r}=jdk")
endforeach()
target_compile_options(lizard_desktop PRIVATE ${LIZ_DESKTOP_WARN})
target_link_libraries(lizard_desktop PRIVATE lizard_presenter lizard_tx ${JAVA_AWT_LIBRARY})
# ai: no RUNPATH to the builder's JDK (a JDK in a home folder named it in the bundled library); the app's JVM has
# ai: loaded jawt before this library (Native.kt)
set_target_properties(lizard_desktop PROPERTIES SKIP_BUILD_RPATH ON)
# ai: Windows: built with liblizard's toolchain file (`-DCMAKE_TOOLCHAIN_FILE=../../liblizard/cmake/mingw-w64.cmake`),
# ai: whose flags link MinGW's runtime into the DLL, so the one file the app bundles loads on a clean Windows machine
# ai: (2026-10-04: libstdc++-6, libgcc_s_seh-1 and libwinpthread-1 were needed beside it)

# ai: lizard_present (2026-10-03, to test whether the browser was to blame): the sender's frames full screen on one monitor
# ai: through the presenter, no browser and no UI; present.cpp's header has its arguments. The GPU painter's kernels and
# ai: tables from ../../liblizard/out unless --assets names another tree.
if(UNIX AND NOT APPLE)
  add_executable(lizard_present present.cpp)
  target_include_directories(lizard_present SYSTEM PRIVATE ${X11_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../liblizard/core/third_party)
  target_compile_options(lizard_present PRIVATE ${LIZ_DESKTOP_WARN})
  target_compile_definitions(lizard_present PRIVATE LIZ_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/../../liblizard/out")
  target_link_libraries(lizard_present PRIVATE lizard_presenter lizard_tx ${X11_X11_LIB} ${X11_Xrandr_LIB})
endif()
