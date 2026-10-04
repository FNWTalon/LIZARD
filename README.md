# Lizard

Lizard is an animated 2D code for moving files from a screen to a phone camera: a sender shows a stream of grey
pictures, and a receiver films them and rebuilds the file. Each frame is a grey picture whose spectrum carries the
data (an OFDM-style design derived from Focus, Hermans et al., MobiSys 2016), inside a self-describing border. Blocks
are protected by a soft-decision LDPC code and a CRC, and a file goes as chunks of a fountain code, each verified by
BLAKE3 against the file's root.

- **Demo:** [fosslabs.dev](https://fosslabs.dev)
- **How it works:** [an example on video](https://www.youtube.com/watch?v=F-Mie4m9gBQ)
- **Android app:** [0.1 (alpha)](https://github.com/FNWTalon/lizard/releases/tag/v0.1)

Design choices:
- **Luminance only.** Grey levels, never colour, so any screen and any camera read it the same way.
- **Degrades instead of failing.** Each frame carries many independently checked blocks. Blur, glare or distance cost
  some blocks, not the whole frame, where a QR code either decodes or does not.
- **Self-describing.** A border around the picture names the format, so a receiver reads any stream with no settings
  and no server; every frame decodes on its own.
- **GPU first.** A WebGPU decoder (also compiled to Vulkan for Android) does the whole decode on the device, with a
  small learned finder; a C decoder is the reference and the CPU fallback.

## Status

A research project, not a released product, and the format still changes. Measured on one phone (Samsung S26 Ultra)
and the author's monitors:

| receiver | rate |
|---|---|
| Android app, one code | 1.3 to 1.6 MB/s logged |
| Android app, two codes side by side (2:1 crop) | 2.0 to 2.2 MB/s logged; 2.9 held with the monitor's contrast at its maximum (reported: 99% of what the sender offered) |
| Chrome on the same phone | about 0.55 to 0.8 MB/s |

Once the Android app's phase lock holds, the rate stays: on a tripod, a 150 s hold of the lock before the pilots read
no short captures, and the current lock's holds, by hand, read 0 to 3% short (one phone, two monitors). The rate depends heavily on the sending
screen (contrast and brightness at their maximum, a steady frame rate) and on the receiving camera: phone cameras
through a native app work best, browser cameras on laptops and webcams poorly.

Everything was built and run on Linux x86-64. The desktop sender has run on Linux (X11) only: its native library
cross-builds for Windows with MinGW-w64 but has not been run there, and macOS is not supported yet.

## Layout

| folder | what it is |
|---|---|
| `liblizard/` | the library: the C codec (`src/`, the format's reference), the GPU decoder (`gpu/`, WebGPU), the native engine (`core/`: the decoder on Vulkan, the C on the CPU, the transfer, the sender), the generator that compiles the GPU decoder for Vulkan (`gen/`), and a C API with WebAssembly and Kotlin bindings (`include/`, `api/`, `bindings/`) |
| `lizard-web/` | the web app: sender and receiver pages, installable as a PWA |
| `lizard-android/` | the Android app: receiver and sender, native Vulkan decoding |
| `lizard-desktop/` | a desktop sender: a Compose Desktop app drawing the code through a native Vulkan presenter |
| `research/SPEC.md` | the format specification |

Comments and docs also cite the project's lab files (`scripts/`, `archive/`, `research/` other than `SPEC.md`,
`STATUS.md`, the per-folder notes, and its phone and rig tools), which are not published; the checks that need its recordings
or reference files say so.

## Building

Commands run from the repository's root unless they `cd`. The web pages, the Android app and the desktop sender all
start from the codec built to WebAssembly; the Android app and the desktop sender also need the GPU kernels generated
from it. Node 22 or later (the web checks drive Chrome through Node's own WebSocket); tested with Node 22, Rust 1.93,
CMake 3.28 and emsdk 6.0.9.

**The codec** ([Emscripten](https://emscripten.org)):

```
source <emsdk>/emsdk_env.sh
liblizard/build.sh             # liblizard/build/ob.mjs and ob.wasm
liblizard/build.sh wirehair    # liblizard/build/wirehair.mjs, the fountain code
```

**The web app.** Serve the repository's root over HTTP and open `lizard-web/index.html`; a camera needs HTTPS or
localhost. `node lizard-web/server.mjs` serves it on 8080 and, with a self-signed certificate made by openssl, on 8443
for a phone on the same network (`RIG_HTTP` and `RIG_HTTPS` name other ports); it serves the repository, apart from
dot folders, to that network. `node lizard-web/pwa/build.mjs` builds the installable app into `lizard-web/app/`, a
static site. Its checks (`lizard-web/check_send.mjs`, `check_ui.mjs`, `check_app.mjs`, `check_rates.mjs`) drive
Google Chrome, as `google-chrome` on the PATH.

**The GPU kernels**, into `liblizard/out/`, need Rust 1.87 or later for [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga),
and SPIRV-Tools' `spirv-val` and shaderc's `glslc`: the Android NDK 27.1's (`ANDROID_NDK_HOME`, else
`$ANDROID_HOME/ndk/27.1.12297006`, else `~/Android/Sdk/ndk/27.1.12297006`) or any on the PATH. No GPU is needed.

```
lizard-android/build.sh tools  # naga 30.0.1, once
lizard-android/build.sh gen    # the GPU decoder's and encoder's kernels and tables
```

Every Gradle build below runs on JDK 17, which Gradle finds among the JDKs installed (each project's
`gradle/gradle-daemon-jvm.properties`): `JAVA_HOME` is not needed.

**The Android app** needs the Android SDK, through `ANDROID_HOME` or `sdk.dir` in `lizard-android/local.properties`,
with the packages `platforms;android-36`, `build-tools;35.0.0`, `ndk;27.1.12297006` and `cmake;3.22.1`.
`lizard-android/build.sh apk` writes `lizard-android/app/build/outputs/apk/debug/app-debug.apk` (arm64, Android 10
or later).

**The desktop sender** needs CMake 3.22 or later (with `JAVA_HOME` naming a JDK where CMake does not find one for its
JNI headers) and, on Linux, the X11 and XRandR development files; `packageDeb`
also needs `dpkg-deb` and `fakeroot`.

```
cd lizard-desktop
cmake -S native -B build/native && cmake --build build/native
./gradlew run                  # or ./gradlew packageDeb: a .deb with its own Java runtime
```

The Windows library cross-builds from Linux, with MinGW-w64's posix-thread compilers (`x86_64-w64-mingw32-gcc-posix`
and `g++-posix`, as Debian names them) and `JAVA_HOME` a JDK, for its JNI headers. From `lizard-desktop/`:
`cmake -S native -B build/win -DCMAKE_TOOLCHAIN_FILE=$PWD/../liblizard/cmake/mingw-w64.cmake && cmake --build build/win --target lizard_desktop`.

**The library**: CMake presets for Linux, Windows, Android, WebAssembly and the JVM, in
[liblizard/README.md](liblizard/README.md).

## License

Apache License 2.0 (LICENSE). Third-party code and attributions are listed in NOTICE.
