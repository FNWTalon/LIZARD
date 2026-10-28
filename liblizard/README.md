# liblizard

The LIZARD code as a library: a luminance-only animated barcode that carries a file from a screen to a camera. A
frame is a grey picture whose FFT coefficients carry blocks of 473 bytes (a 4-byte id and 469 of payload, CRC-checked),
each LDPC-coded at a rate that follows the frequency (7/8 inside, then 3/4 and 2/3, 1/2 outside), inside a border
that says what is inside; a file goes as a fountain-coded transfer verified by its BLAKE3 root.

One C API (`include/lizard.h`), three bindings over it:

| | what | where |
|---|---|---|
| C | the shared library: `liblizard.so` (Linux, Android), `lizard.dll` (Windows: cross-built from Linux, not yet run) | `include/`, `api/` |
| WebAssembly | one ES module and its `.wasm` in two builds (WebAssembly SIMD with wasm exceptions; WebAssembly 1.0 for engines without them), no threads, no GPU | `bindings/wasm/` |
| Kotlin | a JVM library (desktops; its natives Linux x86-64 only), and an AAR (Android, arm64, the GPU files in it) | `bindings/kotlin/` |

macOS has no preset.

Beside them, `kotlin/` holds plain-JVM code the apps compile into themselves. Both the Android app and the desktop app
compile `Tar.kt`, how several files or a folder become one transfer (a tar, `research/SPEC.md` 9.6) and how a receiver
unpacks one; `Readout.kt`, the receive screen's words and figures; and `Pick.kt` (`dev.lizard.pick`), the sender's
ladder, rings, rates and the room a format needs. The Android app alone compiles `PhaseLock.kt`, the camera's phase
lock against the display.

## Layers

1. **Format arithmetic.** A format is its size (2 to 128: LIZARD-16 to -1024; a symbol of it carries `liz_blocks_for`
   blocks, the rate profile's count), a ring (the border: 64, 128 or 256 cells, the 128 by default), the display
   rate its word states, and one code or two side by side.
   `liz_geometry_of` gives what it paints; `liz_room_for` and `liz_pick` the room a symbol needs on a display and the
   most blocks that fit a room.
2. **The per-frame codec**, on the CPU (NEON on arm64, WebAssembly SIMD in the wasm build, scalar elsewhere and in the
   wasm MVP build).
   `liz_encoder_paint`: a frame's blocks to its picture, grey or four bytes a pixel, the frame's count for the pilots.
   `liz_decode`: a captured image, or one region of it (`liz_layout_rects`: the centre square, or the two halves of a
   2:1 region), to its verified blocks; the caller keeps the held word (the last word read), the one thing carried from
   frame to frame.
3. **The transfer.** `liz_tx`: a file (or the test stream) to each frame's blocks. `liz_rx`: a frame's verified blocks
   in, the file out once every chunk is verified against the root, in memory or under a directory.
4. **The engines** (native builds): `liz_receiver` decodes camera frames on its own threads (the GPU decoder through
   Vulkan in batches, else the C on a pool) into a transfer; `liz_sender` paints frames ahead on its own threads (the
   GPU's encoder, else the C). The GPU paths read their kernels and nets from an assets directory (`share/lizard` as
   installed); with none, or no Vulkan device that runs them, the CPU does the work. Vulkan is loaded at run time.

Errors are codes (`LIZ_E_*`) with `liz_last_error()` and `liz_last_error_code()` for the calling thread; the bindings
throw (`LizardException` and the JVM's own, `LizardError` in JS). A handle is used from one thread at a time.

The format is never frozen: a sender and a receiver need the same release (`liz_version`).

## Build

From `liblizard/`, with CMake 3.22 or later:

```
cmake --preset linux && cmake --build --preset linux && ctest --preset linux     # the host (Linux); share/lizard on install
cmake --preset linux-asan ...                                                    # the codec, transfer and API under ASan and UBSan (no engines)
cmake --preset jvm && cmake --build --preset jvm                                 # with the Kotlin binding's JNI (JAVA_HOME a JDK)
source <emsdk>/emsdk_env.sh && cmake --preset wasm && cmake --build --preset wasm   # bindings/wasm/dist/lizard.*
cmake --preset wasm-mvp && cmake --build --preset wasm-mvp                       # bindings/wasm/dist/lizard-mvp.*, WebAssembly 1.0
./build.sh && ./build.sh mvp                                                     # the same two builds, as the repository's tree uses them
cmake --preset windows && cmake --build --preset windows                         # lizard.dll, cross-built with MinGW-w64 (posix threads)
cmake --preset android-arm64 && cmake --build --preset android-arm64             # the NDK at ANDROID_NDK_HOME, else $ANDROID_HOME/ndk/27.1.12297006, else ~/Android/Sdk/ndk/27.1.12297006
```

Two inputs of the linux preset come from other builds and are looked for when it is configured (configure again after
making them; the configure says what it left out): ctest's `pick_table`, which holds the picker to the web sender's,
needs Node and the wasm module (`./build.sh`, with emsdk: it writes `bindings/wasm/dist/lizard.mjs` and `build/ob.mjs`,
the path the repository's tools import it by, which re-exports it); the GPU files installed under
`share/lizard` are the generated tree `out/` (`../lizard-android/build.sh tools`, then `gen`).

Options: `LIZ_GPU` (the Vulkan engines; off in wasm), `LIZ_JNI`, `LIZ_TESTS`, `LIZ_SAN`, `LIZ_WASM_MVP` (the wasm build as
WebAssembly 1.0, the `wasm-mvp` preset), `LIZ_WASM_DIST` (where the wasm build writes its module, `bindings/wasm/dist`
by default), `LIZ_VERIFY_POINTS` (a lab build: the border finder's point scans checked against the plain scan they
replaced, `./build.sh verify`), `LIZ_GPU_TREE` (the generated GPU
files) and `LIZ_GPU_VARIANTS` (the decoder variants installed: all six by default). The receiver decodes with the best variant the device runs of
those its assets hold; where they hold none it runs, `auto` decodes on the CPU (the log says why) and `gpu` fails. On a
software Vulkan device (Vulkan's CPU type: lavapipe, SwiftShader) `auto` decodes on the CPU too, its stats' `gpuWhy`
naming the device, and `gpu` still runs there.
Build trees go in `build-native/<preset>`.

The bindings:

```
cd bindings/wasm && npm test                                    # after the wasm preset; LIZ_NATIVE_DUMP=<dir> to hold its paint to native's
cd bindings/wasm && npm run test:mvp                            # the same on the MVP build, after the wasm-mvp preset
cd bindings/kotlin && ./gradlew :lizard:test                    # after the jvm preset
cd bindings/kotlin && ./gradlew :lizard-android:assembleRelease # the AAR
```

The wasm package holds two builds of one module. `dist/lizard.mjs` runs the codec's vector paths on WebAssembly SIMD
and throws C++ exceptions as wasm's own (wasm exception handling); `dist/lizard-mvp.mjs` is WebAssembly 1.0 with no
extension (the codec's scalar paths, exceptions through JS; emscripten's libc has its sign extension, bulk memory and
saturating float-to-int lowered by binaryen, and the module is validated against the MVP feature set), for an engine
without either. `init()` loads the SIMD build where the engine validates both SIMD and wasm exceptions (`hasSimd()`
answers for SIMD alone) and the MVP build otherwise; `init(options, { build: "simd" | "mvp" })` names one. Both
decode the same blocks from the same frames; the MVP build, on the codec's scalar paths, is several times slower. Both builds also
carry the codec's own exports and the Wirehair and zstd shims, which the repository's pages and tools reach through
`sim/`; those are no interface of the package. The repository's web pages load the SIMD build alone, so they need both WebAssembly SIMD and wasm
exceptions: of the browsers with SIMD, that leaves out Chrome 91 to 94 and Firefox 89 to 99.

`npm test`'s comparison with the web sender's picker also needs `build/ob.mjs` (`./build.sh`, the default build's
re-export); without it that test is skipped, saying why. The AAR needs the Android SDK (`ANDROID_HOME`, or `sdk.dir` in `bindings/kotlin/local.properties`)
with the NDK 27.1.12297006 and CMake 3.22.1 in it, and the generated tree `out/`; `-PlizardVariants=f32-sg,...` packs
some of the decoder variants, chosen among as an install's are.

## Tests

- `api/tests/roundtrip.c` (ctest): every format's encoder, the picker against the web sender's (`api/tests/pick_ref.mjs`),
  the test stream in every ring, a file through two codes into memory and through one into a directory.
  `lizard_roundtrip --dump <dir>` writes reference frames for the wasm test.
- `api/tests/engine.c` (by hand, on a machine with the GPU files): `lizard_engine receive <assets> <run> <gpu|cpu|auto>
  [device]` pushes a run through the receiver until its file is whole; `lizard_engine send <assets> <gpu|cpu|auto>
  [device]` paints a file and reads it back. `<assets>` is `out/` or an install's `share/lizard`. A run is a folder of
  `meta.json` (`{"w": ..., "h": ..., "frames": ...}`) and the frames `0000.gray` on, `w` x `h` grey bytes each. The
  project's camera recordings (`.lizard` files, `../research/SPEC.md` 15, which this C API does not read yet) are not published; `lizard_engine paint <assets> <run dir> [frames] [blocks]` makes a run
  with the library alone (a file painted on the CPU through the GPU's frame ring, so the assets too; each frame on
  white).
- `bindings/wasm/test/`, `bindings/kotlin/lizard/src/test/`: the same round trips through each binding.
