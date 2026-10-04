# ai: The android-arm64 preset's toolchain: the NDK's own, from the NDK at ANDROID_NDK_HOME, else the SDK's 27.1 (the SDK
# ai: at ANDROID_HOME, else ~/Android/Sdk: lizard-android/build.sh's rule); a preset alone cannot fall back from an unset
# ai: variable.
set(LIZ_NDK "$ENV{ANDROID_NDK_HOME}")
if(NOT LIZ_NDK AND DEFINED ENV{ANDROID_HOME})
  set(LIZ_NDK "$ENV{ANDROID_HOME}/ndk/27.1.12297006")
elseif(NOT LIZ_NDK)
  set(LIZ_NDK "$ENV{HOME}/Android/Sdk/ndk/27.1.12297006")
endif()
if(NOT EXISTS "${LIZ_NDK}/build/cmake/android.toolchain.cmake")
  message(FATAL_ERROR "no Android NDK at ${LIZ_NDK} (ANDROID_NDK_HOME names one, or ANDROID_HOME an SDK with NDK 27.1.12297006)")
endif()
include("${LIZ_NDK}/build/cmake/android.toolchain.cmake")
