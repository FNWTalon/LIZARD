# LIZARD

LIZARD is an animated 2D code for moving files from a screen to a phone camera: a sender shows a stream of grey
pictures, and a receiver films them and rebuilds the file. Each frame is a grey picture whose spectrum carries the
data (an OFDM-style design derived from Focus, Hermans et al., MobiSys 2016), inside a self-describing border. Blocks
are protected by a soft-decision LDPC code whose rate follows the frequency (7/8 on the lowest, where a capture holds
the most, 1/2 on the highest) and a CRC, and a file goes as chunks of a fountain code, each chunk
compressed with zstd where that makes it smaller and each verified by BLAKE3 against the file's root.

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

## How it works

**A frame is a picture of its spectrum.** The sender writes the data as QPSK symbols (two bits each) onto the 2D
Fourier coefficients of an n x n picture, taken in order of rising spatial frequency, and an inverse transform turns
them into a grey picture. The picture is 256 to 1536 samples a side, depending on how much the frame carries.

**Low frequencies survive the most.** Blur, distance, a small camera and resampling all eat the high frequencies
first. So the coefficients are grouped into sub-channels of 320, and blocks are laid out from the centre of the
spectrum outwards. Each block is 473 bytes: a 4-byte id and 469 bytes of payload, with a CRC-32. Each block is
LDPC-coded over whole sub-channels at a rate that follows the frequency: 7/8 on the innermost, then 3/4 and 2/3, and
1/2 on the outermost. A poor capture loses blocks from the outside in, and every block that reads is checked on its
own.

**The border says what is inside.** Round the picture sits a black and white border 15 modules deep: a corner mark
at each corner, a dark ring, and a band of cells. The band carries a timing track and the format word, which names
the version and the display rate under a Reed-Solomon code. The border comes in four sizes (rings of 32, 64, 96 or
128 band cells a side), and any ring can carry any picture. A receiver finds the border, reads the word, and from it
knows the picture size, the blocks and their code rates. It needs no settings and no server.

**A file is a fountain.** A file is cut into chunks of 4 MiB. Each chunk is compressed with zstd where that makes it
shorter and sent as a Wirehair fountain, so any large enough set of its blocks rebuilds it, from any frames in any
order. Header and manifest blocks in the same stream describe the file, and each chunk is checked against the file's
BLAKE3 root. Nothing comes back from the receiver.

**Pilots tie a frame to its time.** Spare slots at the end of the 7/8 and 3/4 blocks carry a known pattern that
flips with the frame's count. A receiver uses them to correct the grid's alignment, and to tell which neighbouring
picture leaked into a capture. The Android app uses that to hold its camera in phase with the display.

## The format

[research/SPEC.md](research/SPEC.md) is the specification. The C codec in `liblizard/src/` is the reference: the
spec describes what it paints, and where the two disagree the code wins. The spec's test vectors are that code's own
regression check. Another encoder conforms by painting a symbol that decodes: the border exactly, the picture by the
format's arithmetic. Its sections:

| section | covers |
|---|---|
| 1 to 2 | what LIZARD is, its terms, its fixed constraints (luminance only, generic screens and cameras, a border that grows outwards only, a self-describing symbol), and what conformance means |
| 3. The formats | the 128 versions (LIZARD-8 to LIZARD-1024: sub-channels in steps of 8), the rate profile that sets each block's code rate, the picture size each version takes, the four rings, capacity, and how a sender picks a version for a screen |
| 4. The border | module by module: rim, ring, gap, band, guard, the corner marks, the timing track and the margin |
| 5. The format word | its fields (magic, version, display rate), its Reed-Solomon code and where its cells sit in each ring |
| 6. The picture | the coefficient order, QPSK, the transform, clipping to grey, resampling onto the painted grid and where a receiver samples |
| 7. Blocks and the transfer | the block, its bit map and whitening, the pilots, the block id, chunks and their fountains, the header and manifest blocks, and when a file is accepted |
| 8. The LDPC code | the four quasi-cyclic codes (7/8, 3/4, 2/3, 1/2) as printed base matrices, their encoding, and soft-decision decoding |
| 9 to 10 | frames over time (the display rate, the test stream, the fountain) and the two reference receivers |
| 11 to 14 | measured facts, what was settled or rejected and why, open questions, and the changes over time |

The format is not frozen: a sender and a receiver need the same release.

## The pipelines

Two decoders read the format, one on the CPU and one on the GPU, and each sender paints on either. Every frame is
decoded on its own. The one thing a receiver carries between frames is the last format word it read: a frame whose
own word is unreadable is decoded at that held word.

### Receiving on the CPU (`liblizard/src/`)

The reference decoder, in C. It runs as WebAssembly with SIMD in a pool of web workers, and natively in the Android
app on NEON with a pool of threads:

1. **Find.** The image goes to luma, and a local threshold makes a black and white copy for the finder alone. The
   corner marks are searched for at full, half and quarter scale. Each candidate quad is scored by reading the timing
   track against every ring and orientation. The fallbacks are fitting the border's lines, then cropping round a mark
   that was seen.
2. **Register.** A homography from the quad, then every border node refined by correlation against the border as
   painted. Each side's nodes are smoothed along it, which absorbs a display's fractional scaling. The interior is
   filled from the border alone.
3. **Read the word** in the ring that registered, and take the picture size, blocks and rates it names.
4. **Sample and transform.** The n x n picture is sampled through the registration and detrended, then goes through
   a 2D FFT (radix 2, with one radix-3 stage for the 3 x 2^k sizes).
5. **Decode.** Align the grid from the pilots, make soft values for each block's sub-channels, skip blocks too weak to
   try, run layered min-sum LDPC, and check the CRC-32.

### Receiving on the GPU (`liblizard/gpu/`)

Designed from the format for the GPU, not ported from the C. It is written in WGSL and runs on WebGPU in a browser.
Natively, the same WGSL is compiled to SPIR-V by naga and run by a C++ Vulkan host (`liblizard/core/`), which in the
Android app takes camera frames with no copy. Frames go in batches: one batch is one submission and one readback. The
nets and kernels run in int8 where a device supports it, else f16, else f32.

1. **Front half: find and register.**
   - A pyramid of each frame feeds a trained fully convolutional proposer, which marks likely corner marks.
   - Two trained classifiers in a cascade say, for each candidate, whether it is a mark, which way is out, and how
     big a module is.
   - Votes along the symbol's diagonals find each centre.
   - Every quad, orientation and ring is scored against the timing track.
   - The border nodes are fitted to the ring that won, each side's curve chosen by its own evidence.
   - The format word is read.
2. **Back half: decode, per picture size.** Sample the picture and take its 2D DFT on the device, align the grid
   from the pilots, and make soft values for each code rate's blocks. Then run layered min-sum LDPC, one dispatch per
   code, each stopping a hopeless codeword early by a small trained net, and check the CRC-32.

Both decoders count a block only when every parity check holds and its CRC passes.

### Sending

A sender turns a file into each frame's blocks: chunks, zstd, the fountain, the header and the manifest. It then
paints each frame:
- Encode each block with LDPC and spread its bits over its slots, with whitening and pilots.
- Place them as QPSK on the picture's coefficients and take the inverse transform.
- Clip to grey and resample onto whole pixels per module.
- Put the border round it.

The C paints on the CPU: in a web worker, or on threads in the native senders. The GPU encoder paints the same
symbol in WebGPU on the web page, and as the same kernels on Vulkan in the Android app and the desktop sender. The
native senders paint about a second of frames ahead on the GPU and present them on a fixed schedule. A symbol from
either painter decodes the same; the GPU's picture is within a grey level of the C's.

## Status

A research project, not a released product, and the format still changes. Measured on one phone (Samsung S26 Ultra)
and the author's monitors:

| receiver | rate |
|---|---|
| Android app, two codes side by side (2:1 crop) | 3.8 MB/s median and 4.0 max over 13 minutes on a 1080p monitor (logged 2026-10-08): two LIZARD-592 codes of 70 blocks each at 60 pictures a second from the desktop sender, 96% of the blocks read on a clean capture |
| Android app, one code | 1.3 to 1.6 MB/s logged (2026-10-01, before the code rate followed the frequency) |
| Chrome on the same phone | about 0.55 to 0.8 MB/s (one code, before the code rate followed the frequency) |

Once the Android app's phase lock holds, the rate stays: its holds, by hand, read 0 to 3% of captures short (one
phone, two monitors). What moves the rate from there is the camera, not the decoder: the app asks the camera for its
picture with no sharpening, noise reduction or other enhancement (each is a filter over the code), and a phone that
has warmed delivers fewer frames. The rate also depends heavily on the sending screen (contrast and brightness at
their maximum, a steady frame rate) and on the receiving camera: phone cameras through a native app work best,
browser cameras on laptops and webcams poorly.

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
liblizard/build.sh zstd        # liblizard/build/zstd.mjs, the transfer's compression
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
JNI headers) and, on Linux, the X11, XRandR, Xcomposite and Xext development files; `packageDeb`
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
