# ai: The desktop app's native library (2026-10-03): liblizard/core's sender (lizard_tx: the transfer, the C's
# ai: painters, the Vulkan painter) behind a JNI face for the Kotlin app (lizard-desktop/src), and the presenter that
# ai: puts its frames on screen (presenter.h, 2026-10-04: the app's canvas through JAWT);
# ai: since 2026-10-25 the receiver beside it: since 2026-10-31 the engine's receive session (liblizard/core/rx/session.h,
# ai: the cameras in liblizard/core/cam/) behind the C API's layer 5 and the Kotlin binding's JNI (liblizard/api).
cmake_minimum_required(VERSION 3.22)
# ai: macOS 12 on, as the app's Info.plist says (LSMinimumSystemVersion): with no target set the library was built for
# ai: the SDK's own macOS (15.0 in CI's 2026-10-25 .dmg) and would not load before it. Set before project(); no effect
# ai: elsewhere.
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "the oldest macOS the library loads on")
if(APPLE)
  project(lizard_desktop C CXX OBJCXX)
else()
  project(lizard_desktop C CXX)
endif()
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

# ai: the presenter: X11 (XRandR for the mode's refresh) on Linux; Win32 under _WIN32; on macOS a CAMetalLayer over
# ai: the canvas (macos.mm) for MoltenVK's surface
if(UNIX AND NOT APPLE)
  find_package(X11 REQUIRED)
  if(NOT X11_Xrandr_FOUND)
    message(FATAL_ERROR "the presenter needs XRandR (libxrandr-dev)")
  endif()
endif()
# ai: the JNI library: JAWT for the canvas's window (JAVA_AWT_LIBRARY; the app loads jawt first). MinGW from Linux
# ai: (2026-10-04): the build JDK's jni.h and jawt.h with Windows' jni_md.h and jawt_md.h (win32/), JAWT_GetAWT looked
# ai: up in jawt.dll at run time, so no Windows JDK is needed to link; LIZ_JDK_INCLUDE names the build JDK's include,
# ai: JAVA_HOME's by default
# ai: macOS: JAWT_GetAWT looked up at run time too (macos.mm), so no libjawt is linked
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
  if(APPLE)
    set(JAVA_AWT_LIBRARY "")
  endif()
endif()
add_library(lizard_presenter STATIC presenter.cpp)
target_include_directories(lizard_presenter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lizard_presenter PRIVATE ${LIZ_DESKTOP_WARN})
target_link_libraries(lizard_presenter PUBLIC lizard_tx)
if(APPLE)
  target_sources(lizard_presenter PRIVATE macos.mm)
  set_source_files_properties(macos.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_include_directories(lizard_presenter SYSTEM PRIVATE ${JNI_INCLUDE_DIRS})
  target_link_libraries(lizard_presenter PUBLIC "-framework AppKit" "-framework QuartzCore" "-framework CoreGraphics" "-framework Metal")
endif()
if(UNIX AND NOT APPLE)
  target_compile_definitions(lizard_presenter PRIVATE LIZ_X11=1)
  target_include_directories(lizard_presenter SYSTEM PRIVATE ${X11_INCLUDE_DIR})
  target_link_libraries(lizard_presenter PUBLIC ${X11_X11_LIB} ${X11_Xrandr_LIB})
endif()

# ai: The receiver (2026-10-25; the session and the cameras the engine's since 2026-10-31: liblizard/core's
# ai: lizard_session, lizard_cam, lizard_api_static). The app's library carries layer 5 of the C API (api/session.cpp)
# ai: and the session's JNI (api/jni_session.cpp: dev.lizard.SessionNative, registered at the library's load by
# ai: jni_util.cpp's JNI_OnLoad), the same the Kotlin binding's liblizard carries.
set(LZ ${CMAKE_CURRENT_SOURCE_DIR}/../../liblizard)

# ai: portal.cpp the file chooser; jni_util.cpp the error channel and the load; priority.cpp the launch's priority
add_library(lizard_desktop SHARED tx_jni.cpp jni_util.cpp portal.cpp priority.cpp ${LZ}/api/session.cpp ${LZ}/api/jni_session.cpp ${LZ}/api/jni_text.cpp)
target_include_directories(lizard_desktop PRIVATE ${LZ}/api)
if(WIN32)
  set_target_properties(lizard_desktop PROPERTIES PREFIX "")   # ai: lizard_desktop.dll, the name the Gradle bundle takes
  target_link_libraries(lizard_desktop PRIVATE ole32 uuid)   # ai: the shell's file dialog (portal.cpp)
endif()
target_include_directories(lizard_desktop SYSTEM PRIVATE ${JNI_INCLUDE_DIRS})
# ai: the JDK's include folders named "jdk" in the debug info too (a JDK in a home folder named it, as the tree's root did)
foreach(d IN LISTS JNI_INCLUDE_DIRS)
  get_filename_component(r "${d}" REALPATH)
  target_compile_options(lizard_desktop PRIVATE "-ffile-prefix-map=${d}=jdk" "-ffile-prefix-map=${r}=jdk")
endforeach()
target_compile_options(lizard_desktop PRIVATE ${LIZ_DESKTOP_WARN})
target_link_libraries(lizard_desktop PRIVATE lizard_presenter lizard_tx lizard_session ${JAVA_AWT_LIBRARY})
# ai: no RUNPATH to the builder's JDK (a JDK in a home folder named it in the bundled library); the app's JVM has
# ai: loaded jawt before this library (Native.kt)
set_target_properties(lizard_desktop PROPERTIES SKIP_BUILD_RPATH ON)
# ai: macOS: the folder the library is in searched for the libMoltenVK.dylib Native.kt puts beside it (volk's dlopen by
# ai: leaf name, after a system Vulkan loader)
if(APPLE)
  target_link_options(lizard_desktop PRIVATE "-Wl,-rpath,@loader_path")
endif()
# ai: Windows: built with liblizard's toolchain file (`-DCMAKE_TOOLCHAIN_FILE=../../liblizard/cmake/mingw-w64.cmake`),
# ai: whose flags link MinGW's runtime into the DLL, so the one file the app bundles loads on a clean Windows machine
# ai: (2026-10-04: libstdc++-6, libgcc_s_seh-1 and libwinpthread-1 were needed beside it)

# ai: lizard_receive (2026-10-25, receive.cpp): the receiver headless on every system: the cameras listed, one read into
# ai: the session with a stats line a second, a file written out; its header has its arguments. cam_check (cam_check.cpp):
# ai: the cameras' checks, each backend against a fake device (sim/fake_v4l2.cpp, sim/fake_mf.cpp, and lizard_fakegv
# ai: below) painting the simulated screen, the conversions and the clock's sleep.
add_executable(lizard_receive receive.cpp)
target_compile_options(lizard_receive PRIVATE ${LIZ_DESKTOP_WARN})
target_compile_definitions(lizard_receive PRIVATE LIZ_ASSETS="${LZ}/out")
target_link_libraries(lizard_receive PRIVATE lizard_session)
if(MINGW)
  target_link_options(lizard_receive PRIVATE -municode)   # ai: its wmain (receive.cpp): the arguments in UTF-16
endif()
set(LIZ_CHECK_SOURCES cam_check.cpp check_luma.cpp check_scene.cpp check_session.cpp check_v4l2.cpp check_mf.cpp check_avf.cpp check_aravis.cpp)
if(UNIX AND NOT APPLE)
  list(APPEND LIZ_CHECK_SOURCES sim/fake_v4l2.cpp)
elseif(WIN32)
  list(APPEND LIZ_CHECK_SOURCES sim/fake_mf.cpp)
endif()
add_executable(cam_check ${LIZ_CHECK_SOURCES})
target_include_directories(cam_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LZ}/core)   # ai: sim/'s fakes; cam/'s headers as "cam/..."
target_include_directories(cam_check SYSTEM PRIVATE ${LZ}/core/third_party)
target_compile_options(cam_check PRIVATE ${LIZ_DESKTOP_WARN})
target_compile_definitions(cam_check PRIVATE LIZ_ASSETS="${LZ}/out")
target_link_libraries(cam_check PRIVATE lizard_session)
if(WIN32)
  target_link_libraries(cam_check PRIVATE windowscodecs)   # ai: the fake source's MJPEG frames (WIC's JPEG encoder)
endif()

# ai: lizard_fakegv (2026-10-25, sim/fake_gv.cpp): a GigE Vision camera on loopback for cam_check aravis, Aravis's own
# ai: fake filming the simulated screen, so it builds against Aravis's headers and library (the receiver's backend
# ai: needs neither: it loads Aravis at run time). Built only where Aravis 0.8 is found, beside cam_check, which runs it
# ai: from its own folder: LIZ_ARAVIS_ROOT, an unpacked prefix (the lab's ../../liblizard/.tools/aravis/usr: its
# ai: include/aravis-0.8 and lib[/<multiarch>]) with GLib's own pkg-config; else pkg-config's aravis-0.8 (a system's
# ai: libaravis-dev, MSYS2's, Homebrew's). Not in a cross build (pkg-config would name the build machine's GLib).
set(LIZ_ARAVIS_ROOT "" CACHE PATH "an Aravis 0.8 prefix (include/aravis-0.8, lib) for lizard_fakegv; empty: pkg-config's aravis-0.8")
set(LIZ_FAKEGV_WITH "")
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT CMAKE_CROSSCOMPILING)
  if(LIZ_ARAVIS_ROOT)
    pkg_check_modules(LIZ_GLIB QUIET IMPORTED_TARGET glib-2.0 gobject-2.0 gio-2.0)
    find_path(LIZ_ARAVIS_INCLUDE arv.h PATHS ${LIZ_ARAVIS_ROOT}/include/aravis-0.8 NO_DEFAULT_PATH)
    find_library(LIZ_ARAVIS_LIB NAMES aravis-0.8 PATHS ${LIZ_ARAVIS_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE} ${LIZ_ARAVIS_ROOT}/lib NO_DEFAULT_PATH)
    if(LIZ_GLIB_FOUND AND LIZ_ARAVIS_INCLUDE AND LIZ_ARAVIS_LIB)
      set(LIZ_FAKEGV_WITH ${LIZ_ARAVIS_LIB} PkgConfig::LIZ_GLIB)
    else()
      message(WARNING "LIZ_ARAVIS_ROOT ${LIZ_ARAVIS_ROOT} holds no arv.h or libaravis-0.8, or GLib's headers are missing: no lizard_fakegv")
    endif()
  else()
    pkg_check_modules(LIZ_ARAVIS QUIET IMPORTED_TARGET aravis-0.8)
    if(LIZ_ARAVIS_FOUND)
      set(LIZ_FAKEGV_WITH PkgConfig::LIZ_ARAVIS)
    endif()
  endif()
endif()
if(LIZ_FAKEGV_WITH)
  add_executable(lizard_fakegv sim/fake_gv.cpp)
  target_include_directories(lizard_fakegv PRIVATE ${LZ}/core)   # ai: cam/scene.h, cam/clock.h
  if(LIZ_ARAVIS_INCLUDE)
    target_include_directories(lizard_fakegv SYSTEM PRIVATE ${LIZ_ARAVIS_INCLUDE})
  endif()
  target_compile_options(lizard_fakegv PRIVATE ${LIZ_DESKTOP_WARN})
  target_link_libraries(lizard_fakegv PRIVATE lizard_cam ${LIZ_FAKEGV_WITH})
  add_dependencies(cam_check lizard_fakegv)
else()
  message(STATUS "no Aravis 0.8 headers (LIZ_ARAVIS_ROOT, or pkg-config's aravis-0.8): no lizard_fakegv, cam_check aravis skips its cases")
endif()

# ai: present_check (2026-10-31, present_check.cpp): the presenter on the test stream in a window of its own (X11), no
# ai: JVM: the presenter's cadence measured alone, the mark read back off the screen by a tool
if(UNIX AND NOT APPLE)
  add_executable(present_check present_check.cpp)
  target_compile_options(present_check PRIVATE ${LIZ_DESKTOP_WARN})
  target_link_libraries(present_check PRIVATE lizard_presenter lizard_tx ${X11_X11_LIB})
endif()
