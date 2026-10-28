# LIZARD

LIZARD moves files from a screen to a phone camera, one way, with no network between them. The sending screen plays
a stream of grey pictures; the phone films them and rebuilds the file. From a 1080p monitor to a recent phone it has
carried up to about 4 MB/s (see [Status](#status)).

- **Demo:** [fosslabs.dev](https://fosslabs.dev)
- **How it works:** [an example on video](https://www.youtube.com/shorts/pjcAtQ7GcVQ)
- **Apps:** [0.5 (alpha)](https://github.com/FNWTalon/LIZARD/releases/tag/v0.5): the Android app, and the desktop app for Linux, Windows and macOS, which sends and, on `main` since 0.5 (not yet in a release), also receives through a USB or built-in camera

## The idea

Most screen-to-camera codes (animated QR codes, colour-cell codes) store bits in small squares. The camera has to
see every square clearly: once the squares blur into each other, through distance, focus or a small camera, the
whole frame fails at once.

LIZARD stores bits in the picture's **spatial frequencies** instead. Think of the picture as many striped wave
patterns added together, from broad, smooth gradients to very fine grain. Where each wave's stripes sit (its phase)
carries two bits. Blur and distance wipe out fine detail first and leave the broad patterns alone, so a poor capture
loses only the data on the finest waves and keeps the rest. The code degrades gradually instead of failing all at
once.

Around the picture sits a black and white border. It lets the receiver find the code, work out exactly where it is
and at what angle, and read a small header that says how the picture is laid out. Every frame decodes on its own,
and a file is spread across frames with a fountain code, so it does not matter which frames the camera happens to
catch. Everything is checked: a block counts only if its error correction and its CRC pass, and the file only if its
BLAKE3 hash matches.

In signal-processing terms this is OFDM, the modulation used by Wi-Fi and 4G/5G, applied to a picture (see
[Prior work](#prior-work)).

## What a frame looks like

```
 ▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
 ██▀▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
 ██  ▄▄▄▄▄▄    ▄▄  ▄▄    ▄▄    ▄▄  ▄▄  ▄▄▄▄    ▄▄    ▄▄▄▄▄▄▄▄▄▄▄
 ██  ██████    ▀▀  ▀▀    ▀▀    ▀▀  ▀▀  ▀▀▀▀    ▀▀    ▀▀▀▀▀▀▀▀▀▀▀
 ██  ██████
 ▀▀  ▀▀▀▀▀▀
 ██         ┌───────────────────────────────────────────────────
 ██         │
 ██  ▄▄     │
 ██  ██     │
 ██  ▀▀     │   the picture
 ██         │
 ██         │   grey, looks like noise;
 ██         │   its spatial frequencies
 ██  ▄▄     │   carry the data
 ██  ██     │
 ██  ▀▀     │
 ██         │
 ██  ▄▄     │
 ██  ▀▀     │
 ██  ▄▄     │
 ██  ▀▀     │
 ██  ▄▄     │
 ██  ▀▀     │
 ██  ▄▄     │
 ██  ▀▀     │
 ██  ▄▄     │
 ██  ▀▀     │
 ██         │
 ██  ▄▄     │
 ██  ██     │
 ██  ▀▀     │
```

The top-left corner of a real symbol, module for module: LIZARD-16 in the 64 ring (158 modules a side), each
character one module wide and two tall (`▀` the top module dark, `▄` the bottom, `█` both). The 128 ring, the
default, and the 256 ring have the same border with longer sides.

- **The picture** is n x n samples (256 to 1536 a side, depending on how much the frame carries), drawn on the
  screen at whole pixels. Only data is painted inside it, apart from the anchors.
- **The anchors** are 49 small squares on a 7 x 7 grid painted over the picture (a light middle, a dark ring, a light
  rim; 1.2% of the picture), whose middles show the receiver how a lens bends the picture between the border's nodes,
  which the border alone cannot see. A receiver erases them before it reads the data.
- **The border** is 15 modules deep, a module being the border's unit square, as in QR codes. Going inwards:
  - a **corner mark** at each corner, for finding the code and fixing its perspective;
  - a **dark ring** all the way round, the line the receiver aligns to;
  - a **band of cells** holding a **timing track** (a known pattern, different on each side, which tells the
    receiver which way up the code is and which border size it is looking at) and the **format word** (a 3-byte
    header under Reed-Solomon protection naming the version, meaning how much data the picture holds, and the
    sender's frame rate);
  - a **guard**: the picture's own edges wrapped round it, so that blur affects every part of the picture the same
    way. In OFDM this is called a cyclic prefix.

## Sending, step by step

Several files or a folder are first packed into one tar archive and sent as that file; the Android app unpacks a
received archive into a folder, and the web receiver offers it as the `.tar` it is.

1. **Split the file** into 4 MiB chunks, each compressed with zstd where that makes it smaller.
2. **Fountain-code each chunk** (Wirehair). The encoder can produce as many 469-byte pieces as needed, and any set
   slightly larger than the chunk rebuilds it, whichever pieces they are. A missed frame costs only time, and nothing
   ever has to go back to the sender.
3. **Describe the file** in a header block (name, size, BLAKE3 hash) and manifest blocks (each chunk's hash and
   size), sent in the same stream as the data.
4. **Make blocks.** A block is a 4-byte id (which chunk, which piece), 469 bytes of payload and a CRC-32.
5. **Add error correction.** Each block becomes one LDPC codeword. How much protection it gets depends on where it
   will sit in the spectrum:

   | where in the spectrum | what it looks like in the picture | survives blur and distance | code rate |
   |---|---|---|---|
   | lowest frequencies | broad, smooth shading | best | 7/8 (1 bit in 8 is protection) |
   | lower middle | | | 3/4 |
   | upper middle | | | 2/3 |
   | highest frequencies | the finest grain | worst | 1/2 (half the bits are protection) |

   The code rate is the share of the sent bits that is data. About 39% of the frequencies get 7/8, 15% get 2/3,
   20% get 1/2, and the rest 3/4.
6. **Interleave and scramble.** Each codeword's bits are spread over all of its frequencies by a fixed permutation,
   then XORed with a pseudo-random sequence, so that neither a burst of damage nor repetitive data lines up.
7. **Turn bits into waves.** Each pair of bits picks one of four phases (QPSK) for one frequency, the lowest
   frequencies first. A few known symbols, the **pilots**, fill each block's spare slots.
8. **Make the picture.** An inverse 2D FFT adds all the waves into one grey image; the rare extremes are clipped, the
   image is scaled onto the screen's pixels, and the anchors are painted over it.
9. **Add the border and show the frame**, typically 60 pictures a second on a steady schedule.

## Receiving, step by step

1. **Take brightness only:** the camera's luma (Y) plane, read straight from its buffer. The phone's image
   enhancements (sharpening, noise reduction and the like) are switched off, since each one is a filter over the code.
2. **Find the code.** Locate the corner marks, then test each candidate against the timing track in every
   orientation and border size.
3. **Register (deskew).** Work out exactly where every point of the code falls in the camera image, allowing for
   perspective, lens distortion and the display's own scaling, to about a tenth of a module. The border gives the
   outline; the anchors give the lens's bend inside it.
4. **Read the format word.** It gives the version, and from that where every block lies and which code rate it uses.
   If a frame's word cannot be read, the last word that was read is used.
5. **Sample and transform.** Read the picture at the sender's sample positions, erase the anchors, and take a 2D
   FFT, back to frequencies. Remove the screen's uneven brightness, and use the pilots to correct any offset of a fraction of a
   sample that registration left behind.
6. **Estimate confidence.** For each group of 320 frequencies (a sub-channel), estimate the signal and the noise, and
   turn each received bit into a confidence value rather than a bare 0 or 1 (a soft decision).
7. **Decode.** Undo the scrambling and the interleaving, run the LDPC decoder, and keep a block only if every parity
   check holds and its CRC matches.
8. **Rebuild the file.** Blocks go to their chunk's fountain decoder, in any order and from any frames. Each rebuilt
   chunk is decompressed and checked against its hash; the file is handed over only when every chunk checks.
9. **Stay in step with the screen** (Android app). The camera and the screen both run near 60 Hz but drift apart, and
   a capture taken across a change of picture sees two pictures blended. The pilots show how much of the previous and
   of the next picture leaked into each capture, and the app delays one camera frame now and then to keep its
   captures in the middle of each picture (the algebra is
   [below](#the-pilots-and-the-phase-lock-in-algebra)).

## A phone as a PC's camera

The desktop app can decode what an Android phone films: the phone keeps its own camera, crop and phase lock, and
sends each capture's crop to the PC, which decodes it on its own GPU or CPU. It helps a phone too slow or too hot to
decode at the camera's rate, and a PC whose own camera is poor. Each frame goes as a `.lizard` frame (see
[Replays](#replays-lizard-files)), about three quarters of its raw bytes; where the link carries fewer frames than the
camera makes, the phone skips evenly and sends the rest in runs of four, so the phase lock still sees neighbouring
captures. Packing a frame costs the phone about 1.4 ms of a big core for a 960 x 960 half; Receive › Settings ›
Compression (shown with the PC decoder) turns it off for a phone too weak or too warm, and the frames then go raw. Start and Stop on either end start and stop both, and either end stops when it has not heard from the other
for 3 seconds.

**Over a USB tether** (no driver on Linux or Windows):

1. Connect the phone to the PC with a USB data cable and open LIZARD's Receive on the phone (the PC's Start reaches the
   app there).
2. Turn on USB tethering (the phone's Settings, or the button under Receive › Settings › PC link in the app).
3. On the PC, open Receive › Settings › Camera: the phone is listed by its name. Press Start. The phone's decoder
   switches to PC and its camera starts.

The phone announces itself on the tether alone (UDP 47231) and serves the stream on TCP 47230. Where the PC does not
hear it, Camera › Phone by address takes the address the phone's PC link field shows.

**Over the cable alone, as an Android Open Accessory.** The PC switches the phone into accessory mode and reads the
stream over USB bulk transfers, with no network at all. The desktop app loads libusb-1.0 at run time:
`libusb-1.0.so.0` on Linux (Debian's and Ubuntu's `libusb-1.0-0`), `libusb-1.0.dylib` from Homebrew on macOS, and
`libusb-1.0.dll` on Windows (from [libusb's releases](https://github.com/libusb/libusb/releases), beside the app's
executable or on the `PATH`). The phone then shows under Camera as USB; Start switches it (the phone asks once whether
LIZARD may open the accessory) and the phone reappears on the bus as Google's accessory device, 18d1:2d00 or 18d1:2d01.

- **Linux**: the logged-in user needs access to both USB devices, the phone as it plugs in and the accessory it
  becomes. Desktops grant it to most phones already; where the camera list does not show the phone, a udev rule for the
  phone's vendor (Samsung's is 04e8) and Google's, for example `/etc/udev/rules.d/51-lizard.rules`:

  ```
  SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0660", TAG+="uaccess"
  SUBSYSTEM=="usb", ATTR{idVendor}=="04e8", MODE="0660", TAG+="uaccess"
  ```

  then `sudo udevadm control --reload-rules` and plug the phone in again.
- **Windows**: libusb reaches a device only through the WinUSB driver, which [Zadig](https://zadig.akeo.ie) binds:
  1. Plug the phone in. In Zadig, choose Options › List All Devices, pick the phone, set the driver to WinUSB and press
     Replace Driver. This takes the phone's file transfer (MTP) away from Explorer while WinUSB holds it.
  2. In LIZARD, press Start with the phone chosen under Camera. The phone switches into accessory mode and Windows
     finds a new device.
  3. In Zadig again, pick that device (18d1:2d00 or 2d01, interface 0), choose WinUSB and press Install Driver. Press
     Start in LIZARD again. Windows remembers both drivers for those devices.

  To undo it, uninstall the device in Device Manager with its driver deleted, and plug it in again. This route has
  not been tried on Windows yet; the USB tether needs none of it.
- **macOS**: nothing to install beyond libusb; not tried yet.

On a link of 42 MB/s, what a USB 2 cable carries, a 2:1 crop at 1080p and 60 captures a second (110 MB/s raw) arrives at
about 31 captures a second, where raw luma got about 22 (measured over a throttled loopback); a USB 3 cable and port
carry all of them.

## Replays (.lizard files)

A replay is a run of captures as a receiver's decoder was handed them, kept for playing again into a decoder. It is
one `.lizard` file: a ZIP archive of stored entries, each frame's luma as JPEG-LS's median prediction residuals
compressed with zstd (about three quarters of the raw bytes, lossless), with the run's `meta.json` (sizes, capture
times, camera axes, the crop) and its statistics. The format is specified in
[research/SPEC.md](research/SPEC.md) section 15; any ZIP tool lists and tests one.

- **Recording.** Receive › Developer Tools › Save replays, in the Android app and the desktop app, keeps the newest
  300 frames while the camera runs and writes the `.lizard` when it stops. The Android app's Download puts it in the
  phone's Downloads folder; the desktop app's is in its data folder (Show in folder).
- **Playing.** The desktop app's Receive › Settings › Camera › Upload asks for a `.lizard` and plays it as a camera at
  its recorded times, a 2:1 run's halves put side by side again; the crop is the file's. Headless:
  `lizard_receive --camera upload:<file.lizard> --layout <its crop>`.
- **Looking inside.** `lizard_receive unpack <file.lizard> <folder>` writes the frames as raw `NNNN.gray` files with
  `meta.json`, and `lizard_receive pack <folder> <file.lizard>` makes a `.lizard` of such a folder.

## The pilots and the phase lock, in algebra

**The problem.** The screen shows a new picture every $T_d$ (16.67 ms at 60 Hz), and the camera starts a capture
every $T_c$. The two clocks run independently, so each capture's exposure window slides across the screen's picture
changes; on the Samsung S26, $T_c = 16.650$ ms against $T_d = 16.667$ ms, a slide of about 1 ms a second. A capture
whose window crosses a change holds a share $a$ of the previous picture or a share $b$ of the next one, and reads
fewer blocks. The pilots let the receiver measure $a$ and $b$ on every capture, and the Android app moves its camera's
timing to keep both near zero.

**What the sender paints.** The sender counts its pictures modulo 4, $c = 0, 1, 2, 3, 0, \dots$, with low bit $c_0$
and high bit $c_1$. The spare slots at the end of each block hold known symbols $s_i = \pm 1$, which the sender
flips (multiplies by $-1$) when the block's bit is 1: bit $c_0$ in the even-numbered blocks, bit $c_1$ in the odd
ones. So the even blocks' pilots flip on every picture, and the odd blocks' on every second picture:

| picture $c$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $c_0$, even blocks | 0 | 1 | 0 | 1 |
| $c_1$, odd blocks | 0 | 0 | 1 | 1 |

**What a capture reads.** For each block $b$, the receiver correlates the received values $y_i$ of its $T$ spare
slots with the known symbols, scaled by the block's own signal level ($m_2$ is the mean power of the sub-channel the
slots sit in):

$$
r_b = \frac{1}{T} \sum_{i=1}^{T} \frac{s_i \, y_i}{\sqrt{m_2 / 2}}
$$

The mean over the even blocks is $r$, over the odd blocks $r_2$. A capture of a single picture reads $r = \pm K$ and
$r_2 = \pm K$, where $K = \sqrt{S / (S + N)}$ is just under 1 at a good signal-to-noise ratio, and the two signs give
the count of the picture it holds:

$$
\hat c = [\, r < 0 \,] + 2 \, [\, r_2 < 0 \,]
$$

**What a blended capture reads.** Suppose a capture of picture $c$ also holds a share $a$ of picture $c - 1$ and a
share $b$ of picture $c + 1$. A neighbour whose bit differs contributes flipped symbols, which cancel part of the
correlation:

$$
|r| = K \, (1 - 2a - 2b)
\qquad
|r_2| = \begin{cases} K \, (1 - 2a) & c \text{ even} \\ K \, (1 - 2b) & c \text{ odd} \end{cases}
$$

Bit $c_0$ differs from both neighbours. Bit $c_1$ differs only from the previous picture when $c$ is even, and only
from the next one when $c$ is odd (see the table). Two readings, two unknowns: for $c$ even,

$$
a = \frac{1}{2} \left( 1 - \frac{|r_2|}{K} \right) \qquad b = \frac{|r_2| - |r|}{2K}
$$

and the same with $a$ and $b$ exchanged for $c$ odd. So a capture tells not only how much it is blended but with
which neighbour, which is what a controller needs to know: whether to move later or earlier. A single flipping bit
would only give $|r| = K(1 - 2a - 2b)$, the sum.

**Fitting over many captures.** One capture's readings are noisy, so the app fits $K$, $a$ and $b$ over the last
second of captures by weighted least squares. For each capture and each bit $\beta$ (0 or 1):

$$
|r_\beta| = K - 2Ka \, d^-_\beta - 2Kb \, d^+_\beta + \varepsilon
$$

Here $d^-_\beta$ is 1 when bit $\beta$ of this capture's count differs from the previous capture's count, and
$d^+_\beta$ the same against the next capture. The counts are read from the neighbouring captures' own signs, so a
picture the screen showed twice (no bit differs) or skipped is accounted for as it happened, not assumed. Each
reading is weighted by $1/\sigma^2$, its standard error from the spread of $r_b$ over the blocks.

**The correction.** A capture that starts too early holds some of the previous picture ($a > b$); one that starts
too late holds some of the next ($b > a$). The error is $e = a - b$, with standard error
$\sigma_e = \sqrt{\sigma_a^2 + \sigma_b^2}$. When both neighbours show and $|e| \ge 3 \sigma_e$, the app moves its
captures later by

$$
h = G \, e
$$

where $G$, in milliseconds per unit of $e$, is learned as the lock runs (below). The only timing control the camera
API gives is to make one frame longer: a single capture request with frame duration $T_c + h$ shifts every capture
after it by $h$. A move earlier by $x$ is therefore made as a move later by $T_d - x$, one picture less $x$.

- Neither neighbour shows (both shares under 2%): the captures are clean, and the hold stands.
- Only one shows: the captures sit at the edge of a clean region; the hold steps out of that edge and 2 ms in.
- No capture is mostly one picture (the median $|r|$ under 0.35): there is no sign to trust, so the hold jumps a
  quarter of a picture later and looks again.

**Learning the gain and the drift.** Take two fits a time $\tau$ apart, with a move $h$ made between them. The error
changed by what the move and the clocks' slide did between them:

$$
G \, \Delta e - p \, \tau = -h - p_{\mathrm{line}} \, \tau
$$

where $p$ is the display's pace against the camera (its slide per second) and $p_{\mathrm{line}}$ is the pace the
hold assumed over that interval. Least squares over successive pairs of fits, the older ones fading, gives $G$ and
$p$, and the hold cancels the measured slide as it goes instead of waiting for it to show up as an error.

The pilots do one more job in both decoders: the known symbols sit at every frequency, so they also measure any
offset of a fraction of a sample left in the sampling grid (a grid offset turns each coefficient by an angle that
grows with its frequency), which is turned back before decoding.

## Why it is built this way

- **Grey only, never colour.** Colour passes through the camera's colour filter mosaic, demosaicing and colour
  processing, and every screen shows colours differently. Brightness is what any camera reports most reliably.
- **Frequencies rather than cells.** In the frequency domain, blur simply weakens each frequency by some amount,
  which the receiver measures and allows for, and it takes the finest frequencies first. A code made of cells needs
  every cell resolved, and below that resolution it reads nothing. Measured side by side, a cell code carried more
  while every cell was sharp and nothing past that point; LIZARD keeps reading, with fewer blocks.
- **Code rate by frequency.** The lower frequencies arrive cleaner than the higher ones, so they carry more data and
  less protection, and the higher ones the reverse.
- **Soft-decision LDPC.** The decoder uses how sure it is of each bit, not just its value, which lets the same parity
  correct far more errors.
- **One large code with many blocks**, rather than many small codes side by side: each small code would need its own
  finder marks and quiet zone, while one code spends its border once.
- **Everything in the border, nothing inside.** The picture is a sum of waves that each span all of it, so cutting a
  hole in it for an alignment mark costs much more than the hole's area (about eight times, measured). Finding and
  deskewing use the border alone.
- **A fountain and hashes, no back channel.** The camera can miss any frames; the file still arrives, and it is
  verified end to end.
- **The decoder on the GPU.** Each frame needs finding, registering, a 2D FFT of up to 1536 x 1536 samples and LDPC
  decoding, up to 120 frames a second on a phone; a phone's GPU does that in a few milliseconds a frame. A C decoder
  is the reference and runs where there is no suitable GPU, and in the web receiver.

## Words used here

| term | meaning | usual name elsewhere |
|---|---|---|
| frame | one picture shown on the screen | symbol |
| module | the border's unit square | module (as in QR) |
| picture | the grey square inside the border | data area |
| coefficient | one 2D spatial frequency of the picture, carrying two bits as one of four phases | subcarrier carrying a QPSK symbol |
| sub-channel | 320 consecutive coefficients, ordered from coarse to fine | group of subcarriers |
| block | 473 bytes (a 4-byte id and 469 of payload) plus a CRC-32, sent as one LDPC codeword over 7 to 12 sub-channels | codeword, packet |
| code rate | the share of the sent bits that is data | code rate |
| rate profile | which code rate each block gets, by its frequencies | unequal error protection |
| version, LIZARD-k | how much a frame holds: LIZARD-k has k sub-channels and is version k / 8. LIZARD-592 holds 72 blocks, about 33.8 KB | QR version (capacity) |
| ring | the border's size: 64, 128 or 256 band cells a side, 128 by default. It does not depend on the version: any ring can carry any picture. A larger ring gives the picture more of the screen but asks for a steadier camera | none (a QR code's size follows its version) |
| format word | the short Reed-Solomon-coded header in the border, naming the version and the frame rate | format information |
| timing track | the known pattern in the border's band | timing pattern |
| corner mark | the square at each corner of the border | finder pattern |
| guard | the picture's edges wrapped round it | cyclic prefix |
| pilots | known symbols in each block's spare slots, flipped by the picture's count | pilot symbols |
| bit map, whitening | the fixed permutation and the scrambling of a codeword's bits | interleaver, scrambler |
| registration | working out where each point of the code falls in the camera image | deskewing, geometric correction |
| soft values | how confident the receiver is of each bit | log-likelihood ratios (LLRs) |
| chunk | a 4 MiB piece of the file, with its own fountain and hash | |
| fountain | an erasure code that rebuilds a chunk from any set of pieces slightly larger than it | fountain (rateless) code |
| header, manifest | the blocks that describe the file and each chunk | |
| held word | the last format word read, used for a frame whose own word cannot be read | |
| 2:1 | two codes side by side, read as the two halves of each camera frame | |
| test stream | frames of known content, used for measuring | |

## Prior work

- **PixNet** (Perli, Ahmed and Katabi, MobiCom 2010) sent OFDM from an LCD to a camera: many small codes tiled across
  the screen, each with a cyclic prefix, and error correction set by how strongly blur weakens each frequency. It
  used DSLR cameras with optical zoom.
- **FOCUS** (Hermans et al., MobiSys 2016) made one OFDM code whose spectrum is split into sub-channels ordered by
  frequency, so a weaker camera decodes the inner ones; each sub-channel carried its own Reed-Solomon-coded fragment.

LIZARD keeps FOCUS's modulation (frequency-ordered sub-channels of QPSK) and adds the self-describing border, soft
LDPC with a code rate that follows the frequency, the interleaving, the pilots and the camera's phase lock, the
transfer in the light, and the GPU decoder.

## The format

[research/SPEC.md](research/SPEC.md) is the specification. The C codec in `liblizard/src/` is the reference: the
spec describes what it paints, and where the two disagree the code wins. The spec's test vectors are that code's own
regression check. Another encoder conforms by painting a symbol that decodes: the border exactly, the picture by the
format's arithmetic. Its sections:

| section | covers |
|---|---|
| 1 to 2 | what LIZARD is, its terms, its fixed constraints (luminance only, generic screens and cameras, a border that grows outwards only, a self-describing symbol), and what conformance means |
| 3. The formats | the 127 versions (LIZARD-16 to LIZARD-1024: sub-channels in steps of 8), the rate profile that sets each block's code rate, the picture size each version takes, the three rings, capacity, and how a sender picks a version for a screen |
| 4. The border | module by module: rim, ring, gap, band, guard, the corner marks, the timing track and the margin |
| 5. The format word | its fields (magic, version, display rate), its Reed-Solomon code and where its cells sit in each ring |
| 6. The picture | the coefficient order, QPSK, the transform, clipping to grey, resampling onto the painted grid and where a receiver samples |
| 7. Blocks and the transfer | the block, its bit map and whitening, the pilots, the block id, chunks and their fountains, the header and manifest blocks, and when a file is accepted |
| 8. The LDPC code | the four quasi-cyclic codes (7/8, 3/4, 2/3, 1/2) as printed base matrices, their encoding, and soft-decision decoding |
| 9 to 10 | frames over time (the display rate, the test stream, the fountain) and the two reference receivers |
| 11 to 14 | measured facts, what was settled or rejected and why, open questions, and the changes over time |

The format is not frozen: a sender and a receiver need the same release.

## The pipelines

The technical detail behind the two step-by-step lists above. Two decoders read the format, one on the CPU and one
on the GPU, and each sender paints on either. Every frame is decoded on its own. The one thing a receiver carries
between frames is the last format word it read: a frame whose own word is unreadable is decoded at that held word.

### Receiving on the CPU (`liblizard/src/`)

The reference decoder, in C. It runs as WebAssembly with SIMD in a pool of web workers (the web receiver's one
decoder), and natively in the Android and desktop apps on a pool of threads (NEON on arm64):

1. **Find.** The image goes to luma, and a local threshold makes a black and white copy for the finder alone. The
   corner marks are searched for at full, half and quarter scale. Each candidate quad is scored by reading the timing
   track against every ring and orientation. The fallbacks are fitting the border's lines, then cropping round a mark
   that was seen.
2. **Register.** A homography from the quad, then every border node refined by correlation against the border as
   painted. Each side's nodes are smoothed along it, which absorbs a display's fractional scaling. The anchors are
   found through that map, and where 12 or more hold, the lens is fitted again with them inside the border.
3. **Read the word** in the ring that registered, and take the picture size, blocks and rates it names.
4. **Sample and transform.** The n x n picture is sampled through the registration and detrended, then goes through
   a 2D FFT (radix 2, with one radix-3 stage for the 3 x 2^k sizes).
5. **Decode.** Align the grid from the pilots, make soft values for each block's sub-channels, skip blocks too weak to
   try, run layered min-sum LDPC, and check the CRC-32.

### Receiving on the GPU (`liblizard/gpu/`)

Designed from the format for the GPU, not ported from the C. It is written in WGSL; its setup (the kernels' sizes,
tables and nets) is built by its JavaScript host in `liblizard/gpu/`, and the WGSL is compiled to SPIR-V by naga and
run by a C++ Vulkan host (`liblizard/core/`) in the Android and desktop apps; in the Android app it takes camera
frames with no copy. The web receiver does not run it: it decodes with the C. Frames go in batches: one batch is one
submission and one readback. The nets and kernels run in int8 where a device supports it, else f16, else f32.

1. **Front half: find and register.**
   - A pyramid of each frame feeds a trained fully convolutional proposer, which marks likely corner marks.
   - A trained classifier says, for each candidate, whether it is a mark, which way is out, and how big a module is.
   - Votes along the symbol's diagonals find each centre.
   - Every quad, orientation and ring is scored against the timing track.
   - The border nodes are fitted to the ring that won, each side's curve chosen by its own evidence.
   - The anchors are found through that fit, a mesh is laid through their middles, and the border is searched again
     from what they say.
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
- Paint the anchors over it and put the border round it.

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
| Chrome on the same phone, through the web receiver's former GPU decoder | about 0.55 to 0.8 MB/s (one code, 2026-09-27, before the code rate followed the frequency). That decoder left the page on 2026-10-25; the page now decodes with the C, and no current figure for the page's C decoder on a phone exists |

Once the Android app's phase lock holds, the rate stays: its holds, by hand, read 0 to 3% of captures short (one
phone, two monitors). What moves the rate from there is the camera, not the decoder: the app asks the camera for its
picture with no sharpening, noise reduction or other enhancement (each is a filter over the code), and a phone that
has warmed delivers fewer frames. The rate also depends heavily on the sending screen (contrast and brightness at
their maximum, a steady frame rate) and on the receiving camera: phone cameras through a native app work best,
browser cameras on laptops and webcams poorly.

Everything was built and measured on Linux x86-64. The desktop app also runs on Windows and on macOS (Apple
Silicon, through MoltenVK); there its sender has been checked only on virtual machines (Windows on a software Vulkan
driver, macOS on a build service's machine), each screenshot of it read back by the decoder, which says it sends but
not how steadily it paces on real hardware. The desktop app's receiver (UVC cameras through V4L2, Media Foundation or
AVFoundation) is checked against fake camera devices and a simulated camera filming a simulated screen; with a real
camera it has run once, a 2013 USB webcam (a Logitech C930e on Linux) reading 90 to 120 kB/s (reported, not logged),
which no phase lock helps: a webcam cannot delay a single frame, so a sender for one paints at most half its rate.
It also takes a phone as its camera ([A phone as a PC's camera](#a-phone-as-a-pcs-camera)): the phone sends its
camera's crop as `.lizard` frames over a USB tether or, as an Android Open Accessory, over the cable alone. The
accessory has run over a USB 2 cable on Linux (the link steady at 41 to 42 MB/s); the tether and the accessory on
Windows and macOS have not run on a cable yet.

## Layout

| folder | what it is |
|---|---|
| `liblizard/` | the library: the C codec (`src/`, the format's reference), the GPU decoder's WGSL and its host and the sender's WebGPU encoder (`gpu/`), the native engine (`core/`: the decoder on Vulkan, the C on the CPU, the transfer, the sender), the generator that compiles the GPU decoder for Vulkan (`gen/`), and a C API with WebAssembly and Kotlin bindings (`include/`, `api/`, `bindings/`) |
| `lizard-web/` | the web app: sender and receiver pages, installable as a PWA |
| `lizard-android/` | the Android app: receiver and sender, native Vulkan decoding |
| `lizard-desktop/` | the desktop app: sender and receiver, a Compose Desktop app drawing the code through a native Vulkan presenter and reading a camera (V4L2, Media Foundation or AVFoundation), a phone's stream or a `.lizard` replay into the native decoders |
| `research/SPEC.md` | the format specification |

Comments and docs also cite the project's lab files (`scripts/`, `archive/`, `research/` other than `SPEC.md`,
`STATUS.md`, the per-folder notes, and its phone and rig tools), which are not published; the checks that need its recordings
or reference files say so.

## Building

Commands run from the repository's root unless they `cd`. The web pages, the Android app and the desktop app all
start from the codec built to WebAssembly; the Android app and the desktop app also need the GPU kernels generated
from it. Node 22 or later (the web checks drive Chrome through Node's own WebSocket); tested with Node 22, Rust 1.93,
CMake 3.28 and emsdk 6.0.9.

**The codec** ([Emscripten](https://emscripten.org) and CMake 3.22 or later):

```
source <emsdk>/emsdk_env.sh
liblizard/build.sh             # liblizard/bindings/wasm/dist/lizard.mjs and lizard.wasm: the codec, the transfer, Wirehair and zstd
liblizard/build.sh mvp         # the same module as WebAssembly 1.0 (lizard-mvp.*), the library's fallback without SIMD
```

`liblizard/build/ob.mjs`, the path the pages and tools import the codec by, re-exports the module. The web pages
load its SIMD build, which needs WebAssembly SIMD and wasm exception handling (the library's `init()` falls back to
the MVP build elsewhere; [liblizard/README.md](liblizard/README.md)). `liblizard/build.sh wirehair` and
`liblizard/build.sh zstd`, the steps that once built those two separately, now run the default build.

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

**The desktop app** needs CMake 3.22 or later (with `JAVA_HOME` naming a JDK where CMake does not find one for its
JNI headers) and, on Linux, the X11 and XRandR development files; `packageDeb`
also needs `dpkg-deb` and `fakeroot`. On Linux its receiver loads TurboJPEG (`libturbojpeg.so.0`) at run time for
cameras that send MJPEG; Windows and macOS decode MJPEG themselves.

```
cd lizard-desktop
cmake -S native -B build/native && cmake --build build/native
./gradlew run                  # or ./gradlew packageDeb: a .deb with its own Java runtime
```

The receiver's checks run against fake camera devices and a simulated camera, never a real one:
`cmake --build build/native --target cam_check lizard_receive`, then `build/native/cam_check all` (with
`LIZ_V4L2_NODES=` set empty on Linux, so no camera node is probed). `lizard_receive` is the receiver without the
window: `lizard_receive list`, or `lizard_receive --camera <id>` with a line of figures a second (`upload:<file.lizard>`
plays a replay), and `lizard_receive pack` and `unpack` turn a `.lizard` into a folder of frames and back.

On Windows the native library is built with MinGW-w64, most simply from Linux with its posix-thread compilers
(`x86_64-w64-mingw32-gcc-posix` and `g++-posix`, as Debian names them) and `JAVA_HOME` a JDK, for its JNI headers. From
`lizard-desktop/`:
`cmake -S native -B build/win -DCMAKE_TOOLCHAIN_FILE=$PWD/../liblizard/cmake/mingw-w64.cmake && cmake --build build/win --target lizard_desktop`.
The Windows build must hold no C++ `thread_local` with a destructor: MinGW-w64's emulated thread-local storage
corrupts the heap when a thread holding one exits (`x86_64-w64-mingw32-nm` finds `__cxa_thread_atexit` where one is).
Then, on Windows, with `build/win/lizard_desktop.dll` copied to `build/native/`, `gradlew.bat packageMsi` writes an
installer (WiX, which Gradle fetches) with its own Java runtime. Windows needs a GPU driver with Vulkan.

On macOS (Apple Silicon, macOS 12 or later) the code is presented through MoltenVK: `libMoltenVK.dylib` from a
[MoltenVK release](https://github.com/KhronosGroup/MoltenVK/releases) (`MoltenVK/dynamic/dylib/macOS/`) and its
`LICENSE`, as `MoltenVK.LICENSE`, go in `build/native/` beside the library CMake builds there (Xcode's command line
tools); then `./gradlew packageDmg`.

`.github/workflows/desktop.yml` builds all three packages from a clean checkout.

**The library**: CMake presets for Linux, Windows, Android, WebAssembly and the JVM, in
[liblizard/README.md](liblizard/README.md).

## License

Apache License 2.0 (LICENSE). Third-party code and attributions are listed in NOTICE.
