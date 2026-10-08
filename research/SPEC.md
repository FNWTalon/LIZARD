# Lizard format specification

2026-09-24. The C in `liblizard/src/` is normative: the format is what `liblizard/src/enc.c`, `liblizard/src/focus.c`,
`liblizard/src/layout.c`, `liblizard/src/fmt.c`, `liblizard/src/rs.c`, `liblizard/src/ldpc.c`, `liblizard/src/demap.c`,
`liblizard/src/dec.c`, `liblizard/src/shake.c`, `liblizard/src/xfer.c` and their headers do, with the official BLAKE3 C
1.8.7 (`liblizard/vendor/blake3/`) under `liblizard/src/xfer.c`. Where this text
and the code disagree, the code defines the format. Decisions that the code does not yet carry are
stated as decisions and listed in section 13. Definitions cite their source as `file:function`. The arithmetic and the
test vectors were checked against the code or the wasm build (`liblizard/build/ob.wasm`, through `liblizard/sim/ob.mjs`) on 2026-09-24,
and 5.2, 6.12, 7.4 and 9.3 again on 2026-09-26 after FOCUS's finder frame was deleted. Updated 2026-09-27 for the
rings: three that morning (32, 48, 64), four that evening (32, 64, 96, 128), with the 64 the sender's default
for every picture. Sections 1 to 6, 10 and 12 to 14 follow the evening's code, with 3.3 to 3.6, 4.3 to 4.10, 5.1 to
5.4, 6.2, 6.8, 6.9 and 6.12 recomputed from `liblizard/build/ob.wasm` and `liblizard/src/fmt.c`; section 4 is the border the code paints
(a 12-module corner mark, a 3-module guard). Measured figures cite the record they come from.

## Contents

1. [What Lizard is](#1-what-lizard-is)
2. [Constraints](#2-constraints-normative)
3. [The formats](#3-the-formats)
4. [The border](#4-the-border)
5. [The format word](#5-the-format-word)
6. [The picture](#6-the-picture)
7. [Blocks, the bit mapping and the transfer](#7-blocks-the-bit-mapping-and-the-transfer)
8. [The LDPC code](#8-the-ldpc-code)
9. [Animation and transport](#9-animation-and-transport-non-normative) (non-normative)
10. [Reference receivers](#10-reference-receivers-non-normative) (non-normative)
11. [Measured facts](#11-measured-facts-non-normative) (non-normative)
12. [Settled and rejected](#12-settled-and-rejected-non-normative) (non-normative)
13. [Open questions](#13-open-questions)
14. [Changes since the 2026-09-22 spec](#14-changes-since-the-2026-09-22-spec)

## 1. What Lizard is

Lizard is a luminance-only animated 2D barcode for one-way transfer from a screen to a camera, derived from FOCUS
(Hermans et al., MobiSys 2016). A generic screen paints a sequence of frames, a generic phone camera films them, and
nothing comes back. Each frame is one symbol: a square grey picture of n x n samples whose 2D spectrum carries QPSK
symbols on the coefficients of the upper half-plane, taken in order of rising spatial frequency, inside a black and
white border 15 modules deep. The border comes in four sizes, the rings, whose band holds 32, 64, 96 or 128 cells a
side; any ring may carry any picture, and the sender paints the 64 ring unless told otherwise. The ring registers the
symbol and carries a format word that says what is inside (the sub-channel count, so the picture size and the blocks,
and the display rate the sender intends): one Reed-Solomon codeword over every word cell of the border, 8, 16, 24 or 32
bytes by ring. The
payload travels in blocks of 473 bytes (a 32-bit block id and 469 payload bytes) with a CRC-32, each coded by a
QC-LDPC code onto consecutive sub-channels of 320 coefficients at a rate that follows the frequency: 7/8 on 7
sub-channels a block in the inner tier, 3/4 on 8 in the middle, 1/2 on 12 in the outer (the rate profile, a function of
the sub-channel count, 3.1). Block 0 holds the lowest frequencies and the
blocks run outwards, and blocks from any frame feed a Wirehair fountain, so the receiver needs no particular frame. A
file travels in chunks, a fountain each, described by a header block and verified chunk by chunk against its BLAKE3
hash, all in the light (7.5 to 7.10).

The format is built on one property: blur, distance and resampling take the outermost blocks first, so a frame loses
blocks gradually instead of stopping dead at one resolution the way QR, Aztec and binary grid codes do. That is why
Lizard was chosen over the binary grid code, which carries more above its cliff (2026-09-20).

### 1.1 Terms

| term | meaning |
|---|---|
| frame | one symbol as displayed; the sender paints a sequence of them |
| symbol | one frame's square: the picture inside the border, S = 2B + 30 modules a side in ring B (94, 158, 222, 286; 3.3). S, "modules a side" and module coordinates exclude the margin |
| ring B | one of the four border sizes, named by its band cells a side: B = 32, 64, 96 or 128 (`liblizard/src/focus.h:FOCUS_RING`). "The 96 ring" is a symbol whose band holds 96 cells a side. It locates the symbol and syncs its grid; the word it carries says what is inside (3.3, 5.5) |
| painted image | the symbol and its margin as the codec paints them: S + 4 modules, (S + 4) pxm pixels a side (4.8) |
| module | the border's unit of length; the symbol is S modules a side |
| picture | the n x n grey image inside the border that carries the data (section 6); it spans span = 2B modules |
| sample | one of the picture's n x n values; a module is n / span samples, a whole number in the 32 and 64 rings, and in the 96 and 128 rings where span divides n (3.3) |
| pixel | a pixel of the painted image (pxm to a module) or of a camera image; the text says which |
| drive | the symbol as the encoder's float image, S x pxm pixels a side, 0 dark and 1 light, margin excluded (`liblizard/src/focus.c:focus_encode`) |
| border | the 15 modules round the picture: rim, ring, gap, band and guard (4.3) |
| outer border | depths 0 to 11 of the border: rim, ring, gap, band, the light depths 7 to 11 and the corner marks; everything but the guard. Painted from the layout, each module one level, dark or light (4.2) |
| ring | the dark line at depths 1 and 2 of the border, the reference receiver's anchor; by extension the border's size, as in "ring B" |
| top ring | the outermost frequency ring a format uses, radius R in cycles across the picture (3.2); not the border's ring |
| guard | depths 12 to 14, the picture's periodic extension (4.4) |
| margin | the 2 modules of light round the symbol, `FOCUS_QUIET`, which the codec paints (4.8). The C's variable `margin` is the 15-module border depth, not this |
| mark | a corner mark, the 12 x 12 module square at each corner (4.5) |
| band | depths 5 and 6 of the border, in cells (4.6) |
| cell | a 2 x 2 module square of the band: a track cell or a word cell. A test cell is one capture condition of a simulator sweep |
| word | the format word, one Reed-Solomon codeword over the band's W word cells a side (5.1) |
| coefficient | one frequency (u, v) of the picture's 2D DFT; it carries one QPSK symbol, two slots |
| slot | one transmitted bit: the sign of a coefficient's real or imaginary part |
| sub-channel | 320 consecutive coefficients of the frequency order (6.3) |
| block | 473 bytes, one codeword on consecutive sub-channels: 7 at rate 7/8, 8 at 3/4, 12 at 1/2 (3.1) |
| codeword | the LDPC codeword of one block: 4464 bits at 7/8, 5088 at 3/4, 7680 at 1/2 (section 8) |
| rate profile | the blocks' code rates by frequency, inner first, a function of subch alone (3.1) |
| subch | the number of sub-channels in a frame; LIZARD-512 has subch = 512 |
| version | V = subch / 8, the format's size; what the word's second byte carries. A frame of it carries B(V) blocks, the rate profile's count. It says what is inside the ring and nothing about the ring (3.1) |
| block id | bytes 0 to 3 of a block: 14 bits of chunk over 18 of fountain symbol (7.5) |
| chunk | 2^k bytes of the file (the last may be shorter), its own Wirehair fountain (7.6) |
| fountain symbol | the low 18 bits of a block id: the block's Wirehair block id in its chunk's fountain (7.5). `liblizard/src/xfer.h` calls it the symbol; it is not a symbol in the sense above |
| control block | the header block or a manifest block, chunk 16383 (7.5, 7.7, 7.8) |
| chaining value | BLAKE3's 32-byte value of a subtree of the file's hash tree; each chunk has one (7.9) |

## 2. Constraints (normative)

These bind every encoder and decoder of the format. This document records the format and proposes nothing.

1. **Luminance only.** A sender MUST carry information in luminance alone. Every pixel it paints, margin included, is
   a grey level: `liblizard/src/focus.c:focus_paint_rgba` writes one value to R, G and B. Colour MUST NOT carry information; it
   is what made cimbar unreliable. The one exception is a two-colour ramp: a sender MAY paint the dark and light ends
   of the ramp in two brand colours, the data still only in luminance, if the pair meets both halves of the palette
   rule (STATUS.md, "Two brand colours"):
   - the luma difference is at least 60% of full range, about 153 of 255, with luma
     Y = (77 R + 150 G + 29 B + 128) >> 8 as the receiver computes it (`liblizard/src/acquire.c:ob_luma`);
   - the three channel swings (light minus dark in R, in G and in B) are within about 15% of each other, so a module
     boundary carries no chroma edge.

   The rule was measured in the simulator (`scripts/exp/focus_colour.mjs`), whose camera has no Bayer mosaic, demosaic or
   chromatic aberration. Neither `liblizard/src/` nor the sender paints two colours, and no code defines how a grey level maps
   to a colour between the two ends (section 13).
2. **Generic screen, generic phone.** No hardware-specific tricks. A sender MUST NOT rely on, and a receiver MUST NOT
   require, a property of one display or one camera.
3. **The border grows outwards only.** Nothing is ever painted into the picture. Its span = 2B modules a side hold
   the picture and nothing else, and every mark, band and margin the format has lies outside them. The picture is an
   inverse FFT, so a hole in it is not a local cost: it was measured at about 8 times its own area in lost payload,
   and half the blocks at n = 1024 (`liblizard/src/focus.c:init`, simulator).
4. **The symbol is self-describing.** Everything a receiver needs to decode a frame is in the frame. The ring, one of
   four, locates the symbol and syncs its grid, and states nothing else; the format word it carries gives the
   sub-channel count and so the picture size n and the block count (3.1, 3.2, section 5). Anything about the transfer
   belongs in the payload and never in the border, as in QR and Aztec (`liblizard/src/fmt.h`): the fountain's header is a
   block of its own, the header block (7.7). The C around the word is open to heavy change.
5. **The margin is part of the symbol.** An encoder MUST paint 2 modules of light round the symbol's S x S modules, so
   a page may put anything at the canvas edge (4.8). The margin became part of the symbol on 2026-09-23 and the
   codec's to paint on 2026-09-24: `liblizard/src/focus.h:FOCUS_QUIET` = 2, painted by `liblizard/src/focus.c:focus_paint_rgba`. It belongs
   to the painted image, S + 4 modules a side, but lies outside the module grid, so S and module coordinates exclude it
   (1.1). With the border's light rim, which stays (2026-09-27), the ring then has 3 modules of light outside it,
   which the reference finder assumes.
6. **The channel.** The symbol and its margin are undisturbed. Noise is incidental and lies outside them: a busy page,
   a noisy surround, clutter up to the margin. Nothing adversarial, nothing painted over the symbol. A noisy
   background is a standard test condition for any decoder, not a special case.
7. **Scope.** This specification is the 2D code. The Wirehair fountain and the zstd compression of each chunk (7.6)
   are fixed codecs, not under test; the layouts that carry them (7.5 to 7.9) are the format's.

**Conformance** (2026-09-24; revised 2026-09-29). The reference build is exact: `liblizard/src/` as `liblizard/build.sh` builds it
paints the same bytes on every machine, and its test vectors (5.2, 6.12, 7.4, 9.3) are its own regression check.
`liblizard/build.sh` compiles with `-ffp-contract=off`, so no compiler fuses a multiply and an add into one rounding and a build
on a machine with FMA paints the same bytes. Another encoder is not held to the reference's bytes (2026-09-29). A sender
conforms by painting a symbol that decodes: the border (sections 4 and 5, the margin included) exactly as the format
specifies it, and the picture by the format's arithmetic, its coded bits exactly as 6.3 to 6.5 and sections 7 and 8
define them, its real arithmetic (6.6 to 6.8) to the formula, not to the reference's last bits (13, item 9). A receiver
never checks conformance.

**Not the format.** Each of these exists in the code or the record. None is part of the format, and the format word
cannot state any of them. A receiver told nothing reads a frame of other rate tiers, `FOCUS_RS` or another bit map wrongly.

- A power tilt across the sub-channels (`focus_t.tilt`, in dB; 6.5). Sender only: the receiver measures each
  sub-channel's amplitude.
- Rate tiers other than the profile's (`liblizard/src/focus.c:focus_init_tiers`): up to three runs of blocks at any of the
  generator's rates, a block taking 7 sub-channels at 7/8, 8 at 3/4, 9 at 2/3 and 12 at 1/2. The format's tiers are the
  rate profile's alone (3.1).
- `FOCUS_RS`, mode 0 of `liblizard/src/focus.c:focus_init`: the FOCUS paper's RS(80,64) per sub-channel with hard QPSK.
- Bit map modes other than LINEAR (7.3).
- Straddle cancellation, reading a capture that straddles a frame change by subtracting what the receiver already
  holds: out for good (2026-09-23), archived in `archive/straddle-cancel/`.
- FOCUS's own finder frame ("finders and margin", `thin = 0`): corner finders, orientation strips and a 20-module
  margin, the frame the FOCUS reimplementation began with. Deleted from `liblizard/src/` and `liblizard/sim/` (2026-09-26; section
  13, item 12). A symbol has one frame, the border of section 4.
- A 2 x 2 picture split, four smaller symbols edge to edge in place of one, each with a word stating its own version
  or with none (n = 128). It is gone from the sender page (2026-09-24); `archive/lizard-2/lizard2.mjs` still builds it
  through `liblizard/sim/phy.mjs:makeGrid` for measurement, and the receiver still accepts a grid spec.
- The other borders `liblizard/src/` can paint for experiments (4.11).

**Implementation rules.** These bind the project's decoders, not the symbol.

- No SharedArrayBuffer, so no wasm threads. A browser decoder runs one thread per worker, each with its own heap, and
  per-worker memory is the figure that matters on a phone.
- The format is never gated on a decoder. Where a decoder falls short on the format, the decoder is fixed; a format
  choice is not reverted or bent for it (2026-09-23).
- No idealised benchmarks. A number that assumes exact registration, a still camera or a white screen carries that
  caveat or is not reported. Every measured figure in this document carries its conditions.

## 3. The formats

### 3.1 Names, versions and the ladder (normative)

- A **sub-channel** is 320 consecutive coefficients of the picture's coefficient order (`liblizard/src/focus.h:FOCUS_SUB`, 6.3).
- A **block** is 473 bytes coded as one codeword on consecutive sub-channels, as many as its rate needs: 7 at 7/8, 8 at
  3/4 (`FOCUS_GROUP`), 12 at 1/2. Blocks are laid from sub-channel 0 outwards, each beginning where the one before
  ended (`liblizard/src/focus.c:init`, `block_sub`).
- A **format** is named after its sub-channel count subch: LIZARD-512 has subch = 512. Its **version** is
  V = subch / 8, its size. The format word's version byte carries V (`liblizard/src/focus.c:fmt_paint`). A frame of it
  carries B(V) blocks, the rate profile's count (below).

**The rate profile** (2026-10-07). The code rate follows the frequency. A capture's signal-to-noise ratio falls with the
frequency, so the lowest sub-channels carry more than rate 3/4 needs and the highest less. Inner first, a frame of
subch sub-channels carries a blocks at 7/8 (7 sub-channels each), m at 3/4 (8 each) and c at 1/2 (12 each), with
7a + 8m + 12c = subch and m >= 1. Of the (a, c) that fit, the profile is the one with the least
(700 a - 32 subch)^2 + (1200 c - 30 subch)^2, in integers, the first of equals in rising a, then rising c: 7a near 32%
of the sub-channels and 12c near 30% (`liblizard/src/focus.c:focus_tiers_for`, `FOCUS_TIER_IN` = 32, `FOCUS_TIER_OUT` =
30; `liblizard/sim/lizard_pick.mjs:tiersFor`, the same integers). A tier of no blocks is absent: LIZARD-8 to -40 are one
tier of 3/4, and every larger format has all three. B(V) = a + m + c (`focus_blocks_for`): V up to V = 10, then at most
7 under V (121 at LIZARD-1024), never falling as V rises. Every block is 473 bytes whatever its rate, so the profile
moves a frame's capacity by its block count alone. It is a function of subch, which the word names, so a receiver told
the version knows every block's place and code, and nothing else is signalled.

| format | tiers, inner first | blocks B(V) |
|---|---|---|
| LIZARD-8 to -40 | 3/4 x V | 1 to 5 |
| LIZARD-48 | 7/8 x 4, 3/4 x 1, 1/2 x 1 | 6 |
| LIZARD-128 | 7/8 x 4, 3/4 x 8, 1/2 x 3 | 15 |
| LIZARD-256 | 7/8 x 12, 3/4 x 11, 1/2 x 7 | 30 |
| LIZARD-432 | 7/8 x 20, 3/4 x 20, 1/2 x 11 | 51 |
| LIZARD-512 | 7/8 x 24, 3/4 x 25, 1/2 x 12 | 61 |
| LIZARD-568 | 7/8 x 24, 3/4 x 29, 1/2 x 14 | 67 |
| LIZARD-1024 | 7/8 x 48, 3/4 x 47, 1/2 x 26 | 121 |

Why (non-normative). Before 2026-10-07 every block was 3/4 on 8 sub-channels and B(V) = V (section 12). On two
recorded 2:1 phone captures (one phone at 1080; section 11), each frame's own residuals re-modulated under a profile and
read by the reference decoder's soft values and LDPC gave +17.5% bytes a frame at LIZARD-432 under this profile and
+25% at LIZARD-416 under one of its shape (7/8 x 16, 3/4 x 20, 1/2 x 12), against one rate 3/4 at the same sub-channel
count, though the profile offers 5 to 7% fewer blocks; 7/8 everywhere read a third less.

**Versions collapse** (2026-09-27). A version names what is inside the symbol, its sub-channel count, and nothing
else: not a symbol size, not a module count. The symbol's size is its ring's, one of four (3.3), and any ring may carry
any version. The names stay: LIZARD-k already states the sub-channel count.

**Every whole number of blocks** (2026-10-01). A sender MAY paint any version from 1 to 128: every multiple of 8 sub-channels
from 8 to 1024, LIZARD-8, -16, ..., -1024 (`liblizard/sim/lizard_pick.mjs:VERSIONS`, 128 values). One block is the smallest step
a symbol can take: the word names subch / 8, a step of 8 sub-channels. The reference sender's control is a slider of
versions, each labelled with the blocks its profile carries, with an automatic pick beside it (`lizard-web/send.html`),
and its picker reaches all 128. A
receiver reads the version the word names with no flag: the word states any version from 1 to 128
(`liblizard/src/fmt.c:ob_fmt_encode`, `ob_fmt_decode`, `OB_FMT_VERSION_MAX` = 128), and `liblizard/src/focus.c:focus_init` builds any
positive multiple of 8 sub-channels up to 1024. Checked 2026-10-01: every one of the 128 in each of the four rings
painted and read blind byte for byte by the C (`scripts/exp/ring_pairs.mjs`, 512 of 512) and by the GPU decoder
(`scripts/exp/gpu_rings.mjs`, 511 of 512: LIZARD-8 in the 128 ring at 2 px a module not found; STATUS "Blocks a frame, a
slider"). A frame of one block has no diversity (one failed block is an empty frame), which was the reason for the
floor at 16 before; a sender that names one block takes that.

**The ladder before** (2026-09-23 to 2026-10-01): every multiple of 16 from 16 to 1024, 64 formats, versions 2,
4, ..., 128, and from 2026-09-24 an odd version was not painted by the reference sender (a preset of one fell back to
the automatic pick). A clean round trip of LIZARD-8, -24, -136 and -568 through the painted pixels decoded every block
and read the word's version (1, 3, 17, 71) on 2026-09-24, on the n / 8 + 60 border.

**Past the top.** A version past 128 is one the word cannot state, so the symbol could not describe itself.
`liblizard/src/focus.c:init` refuses to build one: more than 1024 sub-channels returns -1 (LIZARD-1032 and -1040, and a
rate-tiered frame of 1029 sub-channels, all refused; LIZARD-1024 builds). Until 2026-09-24 LIZARD-1032 built at
n = 2048 with a blank word.

### 3.2 The picture size follows from the sub-channels (normative)

The top ring radius, in cycles across the picture:

```
R = sqrt(2 * 320 * subch / pi) = sqrt(5120 * V / pi)
```

R is the radius of the half-disc that holds 320 subch coefficients (area pi R^2 / 2 = 320 subch). The picture size n
MUST be the first of 256, 384, 512, 768, 1024 and 1536 with n >= 3R, or 1536 where none holds, computed in double
precision (`liblizard/src/focus.c:focus_n_for` over `FOCUS_PICTURES`; `liblizard/sim/lizard_pick.mjs:N_FOR` over `PICTURE_SIZES` is the
same rule):

```
for n in 256, 384, 512, 768, 1024, 1536: if n >= 3 * R, return n
return 1536
```

The sizes are 2^k and 3 x 2^k, so each is at most 1.5 times the one before (2026-09-27). n >= 3R oversamples the top
ring by at least 1.5: its period, n / R samples, is at least 3. The fallback to 1536 never binds inside the word's
range (LIZARD-1024 has 3R = 1370.2). Since n >= 3R is V <= pi n^2 / 46080:

| n | versions the word can state | ladder formats | ladder count |
|---|---|---|---|
| 256 | 1 to 4 | LIZARD-16, -32 | 2 |
| 384 | 5 to 10 | LIZARD-48 to -80 | 3 |
| 512 | 11 to 17 | LIZARD-96 to -128 | 3 |
| 768 | 18 to 40 | LIZARD-144 to -320 | 12 |
| 1024 | 41 to 71 | LIZARD-336 to -560 | 15 |
| 1536 | 72 to 128 | LIZARD-576 to -1024 | 29 |

Until 2026-09-27 n was the smallest power of two from 256 to 2048 with n >= 3R: LIZARD-48 to -128 at 512, -144 to
-560 at 1024, -576 to -1024 at 2048. At each step to the next power of two a format took four times the samples for
one more block (LIZARD-144 at n / 2R = 2.99). 44 of the 64 formats moved n that day. None moved ring (3.3), and none
moved a coefficient (below; checked on all 44).

**Why these sizes.** The payload a picture carries is set by R, not by n; n only has to oversample the top ring, and a
smaller n costs less to sample and transform. Measured when the sizes came in (2026-09-27, section 11): in the
simulator, eight formats painted at their 3 x 2^k size read 0.85% fewer blocks than at the power of two above it (the
most -1.4%, LIZARD-80 at n / 2R = 1.50), while the C's sampler and transform took 0.62 to 0.69 of their time (one
camera model, clean poses, told, 16 frames a cell, `scripts/exp/focus_nfit.mjs`). The GPU decoder on the desktop's iGPU took
5.53 ms of compute a frame for LIZARD-192 at 768 against 5.68 at 1024, and 16.25 for LIZARD-1024 at 1536 against 18.21
at 2048 (means of two rounds), blocks within 0.2%, bad 0 (synthetic frames, one cell each).

n is derived and never chosen, and only the word states it: the ring says nothing about n (3.3). A receiver reads the
word, takes n = `focus_n_for(8 * version)`, and finishes the frame at that picture in the ring it registered
(`liblizard/src/wasm.c:focus_any_rx`, 10.1). Until 2026-09-27 each picture size had its own module count, and the receiver told
nothing dropped a word whose version implied another n than the count it registered; nothing checks the version against
the ring now.

**n only sets how finely the same coefficients are sampled.** Every coefficient a format uses lies at a radius of at
most R + 0.07 (the largest overshoot on the ladder, LIZARD-48's 0.068), and n / 2 >= 1.5 R. A coefficient that a larger
n adds has a horizontal or vertical frequency of magnitude at least n / 2 of the smaller n, so its radius is at least
1.5 R and it sorts after all of them. So:

- a format's coefficient positions are the same at every n that can hold it;
- within one n, each format's coefficients are the first 320 subch of the top format's, so block b of any version sits
  on the same coefficients, with the same bit map and whitening, as block b of the top version of that n;
- a decoder built for the top version of an n reads any lower version of that n, whose missing blocks fail
  (`liblizard/src/wasm.c:focus_any_setup`, `pic_for`).

The argument holds for any even n, so for the 3 x 2^k sizes as for the powers of two.

On the ladder the oversampling n / 2R runs from 1.50 (LIZARD-80 and -320) to 2.24 (LIZARD-16, -144 and -576).

`liblizard/src/focus.c:init` itself only checks that n is 2^k or 3 x 2^k and a multiple of 64 (the transform's sizes and the
64-column strips), and that the half-plane has room (320 subch at most (n / 2)(n - 2) coefficients, so 96 sub-channels
as a multiple of 8 at n = 256). It does not check n >= 3R; the derived n keeps well inside that limit.

### 3.3 The rings (normative)

The border comes in four sizes, the rings (2026-09-27). Ring B has B band cells of 2 x 2 modules a side, B = 32, 64, 96
or 128 (`liblizard/src/focus.h:FOCUS_RING`), and its geometry does not depend on the picture:

- Modules across the picture: span = 2B (64, 128, 192, 256). `focus_init`'s span argument names the ring (span = 2B); 0
  takes the default ring below (`liblizard/src/focus.c:init`).
- The border: 15 modules a side, the corner mark's 12 (`OB_THIN_CORNER`) plus the guard's 3 (`FOCUS_CP`)
  (`liblizard/src/focus.c:init`, where the C calls it `margin`). Section 4 defines its content.
- Modules a side: S = span + 30 = 2B + 30 (94, 158, 222, 286). With the 2-module margin (4.8) the painted square is
  S + 4 modules (98, 162, 226, 290).

On the morning of 2026-09-27 the rings were three, B = 32, 48 and 64, and that evening 32, 64 and 128 before the 96
ring joined them (section 12).

**Any ring may carry any picture.** A sender MAY paint a picture of any n in any of the four rings. The reference sender
paints the 128 ring for every picture since 2026-10-01 (the 64 from 2026-09-27
evening) (`liblizard/src/focus.h:FOCUS_RING_DEFAULT`, `liblizard/sim/lizard_pick.mjs:RING_DEFAULT`, the Android app's `Pick.kt`); the
Developer panel's ring menu names another (`lizard-web/send.html`; `lizard-web/send.mjs` paints `SPAN(n, ring)`). No ring follows
the version, even by default: the ring is a sync and bootstrap layer, like 5G's sync block, chosen for the channel and
the room, and what is inside is the word's to say (2026-09-27). Until the evening of 2026-09-27 the default followed n
(`focus_ring_for`: 32 for 256, 48 for 384 and 512, 64 above). A receiver MUST NOT infer n from the ring: the
four module counts identify the ring and nothing else, and the word says what is inside (3.2, section 5).

In ring B carrying a picture of n samples:

- Pixels a module in the painted image: pxm = ceil(n / span - 0.0001), in single precision (`liblizard/src/focus.c:init`). It is
  a whole number, so no module is split across pixels.
- Picture samples a module: scale = n / span. The picture keeps its n samples and is resampled to fill its span
  modules (span x pxm pixels), so it is never shrunk; the upscale is pxm x span / n (6.8). Where span divides n, pxm =
  scale and the resampling is an exact copy, one pixel a sample: every picture in the 32 and 64 rings, 384, 768 and
  1536 in the 96 ring, and every picture but 384 in the 128 ring. The 96 ring upscales 256 by 1.5 and 512 and 1024 by
  1.125; the 128 ring upscales 384 by 1.333.

| ring B | span | S | with margin | n | samples a module | pxm | upscale | symbol, px | with margin, px |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 64 | 94 | 98 | 256 | 4 | 4 | 1 (copy) | 376 | 392 |
| 32 | 64 | 94 | 98 | 384 | 6 | 6 | 1 (copy) | 564 | 588 |
| 32 | 64 | 94 | 98 | 512 | 8 | 8 | 1 (copy) | 752 | 784 |
| 32 | 64 | 94 | 98 | 768 | 12 | 12 | 1 (copy) | 1128 | 1176 |
| 32 | 64 | 94 | 98 | 1024 | 16 | 16 | 1 (copy) | 1504 | 1568 |
| 32 | 64 | 94 | 98 | 1536 | 24 | 24 | 1 (copy) | 2256 | 2352 |
| 64 | 128 | 158 | 162 | 256 | 2 | 2 | 1 (copy) | 316 | 324 |
| 64 | 128 | 158 | 162 | 384 | 3 | 3 | 1 (copy) | 474 | 486 |
| 64 | 128 | 158 | 162 | 512 | 4 | 4 | 1 (copy) | 632 | 648 |
| 64 | 128 | 158 | 162 | 768 | 6 | 6 | 1 (copy) | 948 | 972 |
| 64 | 128 | 158 | 162 | 1024 | 8 | 8 | 1 (copy) | 1264 | 1296 |
| 64 | 128 | 158 | 162 | 1536 | 12 | 12 | 1 (copy) | 1896 | 1944 |
| 96 | 192 | 222 | 226 | 256 | 1.333 | 2 | 1.5 | 444 | 452 |
| 96 | 192 | 222 | 226 | 384 | 2 | 2 | 1 (copy) | 444 | 452 |
| 96 | 192 | 222 | 226 | 512 | 2.667 | 3 | 1.125 | 666 | 678 |
| 96 | 192 | 222 | 226 | 768 | 4 | 4 | 1 (copy) | 888 | 904 |
| 96 | 192 | 222 | 226 | 1024 | 5.333 | 6 | 1.125 | 1332 | 1356 |
| 96 | 192 | 222 | 226 | 1536 | 8 | 8 | 1 (copy) | 1776 | 1808 |
| 128 | 256 | 286 | 290 | 256 (default) | 1 | 1 | 1 (copy) | 286 | 290 |
| 128 | 256 | 286 | 290 | 384 (default) | 1.5 | 2 | 1.333 | 572 | 580 |
| 128 | 256 | 286 | 290 | 512 (default) | 2 | 2 | 1 (copy) | 572 | 580 |
| 128 | 256 | 286 | 290 | 768 (default) | 3 | 3 | 1 (copy) | 858 | 870 |
| 128 | 256 | 286 | 290 | 1024 (default) | 4 | 4 | 1 (copy) | 1144 | 1160 |
| 128 | 256 | 286 | 290 | 1536 (default) | 6 | 6 | 1 (copy) | 1716 | 1740 |

The pixel columns are the painted image's own size (`M._focus_side()` and `M._focus_cell()` on every row, 2026-09-27
evening); a page then scales it to the room (3.5). The 128 ring puts every picture on 286 modules, the count every
format had before 2026-09-23.

### 3.4 Capacity (normative)

A block (section 7 defines its layout), by its rate (3.1):

| rate | sub-channels | slots | codeword n | z | known slots | information k |
|---|---|---|---|---|---|---|
| 7/8 | 7 | 4480 | 4464 | 93 | 16 | 3906 |
| 3/4 | 8 | 5120 | 5088 | 106 | 32 | 3816 |
| 1/2 | 12 | 7680 | 7680 | 160 | 0 | 3840 |

- Slots: the sub-channels x 320 coefficients x 2 bits. The codeword: 48 base columns lifted by z = floor(slots / 48)
  (`liblizard/src/ldpc.c:ldpc_init`). The slots past it carry known values (7.3).
- The information word's first 3816 bits, 477 bytes, are the 473 bytes the codec is handed for the block, then their
  CRC-32 (`liblizard/src/focus.c:focus_encode`); at 7/8 and 1/2 the 90 and 24 bits past them are zeros, which a decoder
  knows. `block_bytes` = k / 8 - 4 = 473 for the smallest k of the format's codes, the 3/4 code's, which every format
  has (`liblizard/src/focus.c:init`).
- The first 4 of those 473 bytes are the block id, a uint32, little-endian (7.5, `liblizard/src/xfer.h`), which leaves 469
  useful bytes (7.1).

A frame of version V has B(V) blocks (3.1), 2560 V coefficients, 5120 V slots and 469 B(V) useful bytes, in
whichever ring it is painted (3752 V bits, 73.3% of the slots, under one rate 3/4 before 2026-10-07; 3536 bits a block
of 7/8, 3752 of 3/4 and 3752 of 1/2 on its 4480, 5120 and 7680 slots). Its ceiling in bytes a second is 469 B(V) times
the display rate; goodput, what the receiver's decoded blocks deliver, is lower.

| n | versions | blocks a frame | useful bytes a frame |
|---|---|---|---|
| 256 | 1 to 4 | 1 to 4 | 469 to 1876 |
| 384 | 5 to 10 | 5 to 10 | 2345 to 4690 |
| 512 | 11 to 17 | 10 to 16 | 4690 to 7504 |
| 768 | 18 to 40 | 17 to 38 | 7973 to 17822 |
| 1024 | 41 to 71 | 39 to 67 | 18291 to 31423 |
| 1536 | 72 to 128 | 68 to 121 | 31892 to 56749 |

**Worked example, LIZARD-512.** subch = 512, V = 64. R = sqrt(5120 x 64 / pi) = 322.96 and 3R = 968.9, so n = 1024
(768 < 968.9 <= 1024). Its default ring is 64: span = 128, so 158 modules a side and 162 with the margin; pxm =
ceil(1024 / 128) = 8, an exact copy, so the symbol paints 1264 x 1264 pixels, 1296 x 1296 with the margin. It carries 61
blocks (7/8 x 24, 3/4 x 25, 1/2 x 12) on 163,840 coefficients (327,680 slots), 28,609 useful bytes a frame, and its
version byte is 64 (0x40). At 24 frames a second its ceiling is 686,616 B/s (720,384 under one rate).

To reproduce, in node with `init()` from `liblizard/sim/ob.mjs`: `M._focus_setup(1024, 512, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0)` returns
473 (block bytes), then `M._focus_blocks()` gives 61, `M._focus_side()` 1264, `M._focus_cell()` 8 and `M._focus_quiet()`
2 (2026-09-27). Painting a frame with `M._focus_tx_rgba(blocks, drive, rgba)` gives 1296 x 1296 pixels, the codec's
2-module margin (16 px) included. Every (ring, picture) pair painted this way reads back blind byte-exact
(`scripts/exp/ring_pairs.mjs` with `SUBCH=16,32,48,80,96,128,144,320,336,560,576,1024`, the first and last format of each
picture, in each of the four rings: 48 of 48, 2026-09-27 evening). That round trip has no camera in it: it checks the
arithmetic, not the channel.

### 3.5 Choosing a version (non-normative)

Guidance from the library's picker (`liblizard/sim/lizard_pick.mjs:pickVersion`) and the sender page (`lizard-web/send.mjs`). The
automatic pick is a preview (2026-09-23); nothing in the format depends on it.

- **The room is the only input**: the device pixels the symbol may take. There is no camera assumption. The sender
  cannot see the capture, and the operator frames the camera on the symbol.
- **The room a format needs**, the margin included, in ring `ring` (the default ring, the 128 since 2026-10-01, unless named):

  ```
  ROOM_FOR(subch, ring) = (MODULES(n, ring) + 2 M) / SPAN(n, ring) * 2.7 * R(subch)
  n = N_FOR(subch), SPAN = 2 B, MODULES = 2 B + 30, B = RINGS[ring], M = 2
  ```

  M is `PAINTED_MARGIN()`: the codec's `FOCUS_QUIET` read from the wasm. It throws until `liblizard/sim/ob.mjs` `init()` has
  loaded the codec (4.8).

  2.7 device px a cycle at the top ring (`T_DISPLAY_CYCLE`) is the highest of the floors measured in
  `scripts/exp/display_scale.mjs`, where three formats of 2026-09-20 read their whole payload down to 2.7, 2.2 and 2.0 device
  px a cycle (simulator, nearest-neighbour display).
- **The pick** is the largest ladder format whose room fits, or LIZARD-16 when none does (flagged `tight`).
  `pickVersion` walks the whole ladder; in one ring the room rises with the version (below), so the largest that fits
  is also the last before the first that fails. `pickVersion(roomPx, top, ring)` returns the ring and span with the
  format; a ring left out is the 64.
- **The cap.** The library picks over the whole ladder and does not cap (`pickVersion`'s `top` defaults to LIZARD-1024).
  An app is strongly advised to stop at LIZARD-560 unless it is a fixed rig: the largest pictures, n = 1536 (2048 until
  2026-09-27), are for industrial use (2026-09-23). A receiver reads every version the word names (the reference
  receiver builds pictures to 1536, `liblizard/sim/phy.mjs` `BLIND_NMAX`, since 2026-09-29; to 1024 by default before). The reference sender capped its automatic pick there from 2026-09-26,
  `pickVersion(room, APP_TOP, ring)` with `APP_TOP` = 560, and picks over the whole ladder since 2026-09-29, once every
  receiver read it; an app may still name a `top`. In the default ring a cap
  at LIZARD-560 binds only from 1170.6 device px of room, where LIZARD-576 first fits: a 1080 px room (a fullscreen 1080p
  monitor, a 412 CSS px phone at 2.625) picks LIZARD-480 (n = 1024 in the 64 ring, 60 blocks) capped and uncapped
  (`scripts/exp/pick_check.mjs`, 2026-09-27 evening; on the n / 8 + 60 border it picked LIZARD-624 uncapped and -528 capped). In
  the simulator a 2048 picture registered nothing at 594 camera px with 1.5 px of defocus, where LIZARD-560 read 3752 B
  a frame (6 frames a test cell, one camera model). On one 1080p monitor and phone the 2048 formats decoded but
  did worse than the 1024 range. Both were seen under the n / 4 + 30 border (542 modules a side at n = 2048) and have
  not been repeated on the rings (158 modules in the 64 ring) or at n = 1536.
- **How the reference page scales.** `lizard-web/send.mjs:layout` draws the painted image, margin included, onto a canvas at
  `floor(room / side)` canvas pixels a painted pixel, nearest-neighbour (`imageSmoothingEnabled = false`). In its
  default mode ("stretch", `lizard-web/send.html`) it then sets the canvas's CSS size to the room, a further factor under 2,
  and sets `canvas.style.imageRendering = "auto"`, which overrides the `image-rendering: pixelated` in
  `lizard-web/send.html`. That last step uses the browser's default filter, which smooths. The record has two simulator
  measurements of smoothing. `scripts/exp/display_bilinear.mjs` held a bilinear display at scale 2.5 and moved only the
  monitor: the frame kept 38 to 88% of its payload depending on where the camera's grid fell, where nearest and area
  kept 100% (STATUS.md, 2026-09-20). `research/09` had put smoothed fractional scaling at 2% of the mean and 11% at
  worst (2026-09-19). Which the page should do is open (section 13, item 15).
- **Changing format mid-transfer is allowed**, for example on a window resize. Every format's block carries the same
  469 useful bytes, so the fountain is unaffected (9.1).
- **What a format asks of the camera**, not an input to the pick:
  `CAMERA_PX_FOR(subch, ring) = MODULES(n, ring) / SPAN(n, ring) * 2.12 * R(subch)`, the camera px across the symbol
  (margin excluded) that give the top ring 2.12 camera px a cycle. 2.12 was fitted by `scripts/exp/focus_sweep.mjs` on five
  resolution-limited simulated test cells (2026-09-20, before the border took its present size).

In the default ring, the 64:

| n | ring | room, device px | camera px |
|---|---|---|---|
| 256 | 64 | 195.1 to 275.9 | 149.4 to 211.3 |
| 384 | 64 | 337.9 to 436.2 | 258.8 to 334.1 |
| 512 | 64 | 477.9 to 551.8 | 366.0 to 422.6 |
| 768 | 64 | 585.3 to 872.5 | 448.2 to 668.1 |
| 1024 | 64 | 894.0 to 1154.2 | 684.6 to 883.9 |
| 1536 | 64 | 1170.6 to 1560.8 | 896.4 to 1195.2 |

The room depends on the ring and R only. LIZARD-16 to -128 defaulted to the 32 and 48 rings until the evening of
2026-09-27, and in the 64 ring they need less of both (LIZARD-16 195.1 device px of room and 149.4 camera px, against
236.0 and 177.8 in the 32 ring); from LIZARD-144 up no figure moved.

**The room rises with the version.** In one ring the room is a constant times R: the border's share of the symbol,
MODULES / SPAN, is 1.469, 1.234, 1.156 and 1.117 in the 32, 64, 96 and 128 rings, and does not change with n. So the
picker reaches every format (`scripts/exp/pick_check.mjs`: 64 of 64, 2026-09-27 evening), one capped at LIZARD-560, as the
reference sender's is, all 35 up to 560, and every room of 1170.6 device px or more, and none below, lands on n = 1536.
While the default ring followed n (2026-09-27 morning), the share fell at n = 768 (48 ring to 64) by more than R rose:
LIZARD-144 needed 585.3 px against LIZARD-128's 590.4, and the picker never returned LIZARD-128 (63 of 64; 58 on the
n / 8 + 60 border).

### 3.6 Every format

The columns from format to n / 2R follow from 3.1 to 3.4 and are normative, except ring, which is the reference
sender's default (3.3); the last two are the picker's guidance (3.5), taken in that ring.

| format | V | n | ring | useful B a frame | R | n / 2R | room, device px | camera px |
|---|---|---|---|---|---|---|---|---|
| LIZARD-16 | 2 | 256 | 64 | 938 | 57.09 | 2.24 | 195.1 | 149.4 |
| LIZARD-32 | 4 | 256 | 64 | 1876 | 80.74 | 1.59 | 275.9 | 211.3 |
| LIZARD-48 | 6 | 384 | 64 | 2814 | 98.89 | 1.94 | 337.9 | 258.8 |
| LIZARD-64 | 8 | 384 | 64 | 3752 | 114.18 | 1.68 | 390.2 | 298.8 |
| LIZARD-80 | 10 | 384 | 64 | 4690 | 127.66 | 1.50 | 436.2 | 334.1 |
| LIZARD-96 | 12 | 512 | 64 | 5628 | 139.85 | 1.83 | 477.9 | 366.0 |
| LIZARD-112 | 14 | 512 | 64 | 6566 | 151.05 | 1.69 | 516.2 | 395.3 |
| LIZARD-128 | 16 | 512 | 64 | 7504 | 161.48 | 1.59 | 551.8 | 422.6 |
| LIZARD-144 | 18 | 768 | 64 | 8442 | 171.28 | 2.24 | 585.3 | 448.2 |
| LIZARD-160 | 20 | 768 | 64 | 9380 | 180.54 | 2.13 | 616.9 | 472.5 |
| LIZARD-176 | 22 | 768 | 64 | 10318 | 189.35 | 2.03 | 647.1 | 495.5 |
| LIZARD-192 | 24 | 768 | 64 | 11256 | 197.77 | 1.94 | 675.8 | 517.5 |
| LIZARD-208 | 26 | 768 | 64 | 12194 | 205.85 | 1.87 | 703.4 | 538.7 |
| LIZARD-224 | 28 | 768 | 64 | 13132 | 213.62 | 1.80 | 730.0 | 559.0 |
| LIZARD-240 | 30 | 768 | 64 | 14070 | 221.12 | 1.74 | 755.6 | 578.6 |
| LIZARD-256 | 32 | 768 | 64 | 15008 | 228.37 | 1.68 | 780.4 | 597.6 |
| LIZARD-272 | 34 | 768 | 64 | 15946 | 235.40 | 1.63 | 804.4 | 616.0 |
| LIZARD-288 | 36 | 768 | 64 | 16884 | 242.22 | 1.59 | 827.7 | 633.9 |
| LIZARD-304 | 38 | 768 | 64 | 17822 | 248.86 | 1.54 | 850.4 | 651.2 |
| LIZARD-320 | 40 | 768 | 64 | 18760 | 255.32 | 1.50 | 872.5 | 668.1 |
| LIZARD-336 | 42 | 1024 | 64 | 19698 | 261.63 | 1.96 | 894.0 | 684.6 |
| LIZARD-352 | 44 | 1024 | 64 | 20636 | 267.79 | 1.91 | 915.1 | 700.8 |
| LIZARD-368 | 46 | 1024 | 64 | 21574 | 273.80 | 1.87 | 935.6 | 716.5 |
| LIZARD-384 | 48 | 1024 | 64 | 22512 | 279.69 | 1.83 | 955.8 | 731.9 |
| LIZARD-400 | 50 | 1024 | 64 | 23450 | 285.46 | 1.79 | 975.5 | 747.0 |
| LIZARD-416 | 52 | 1024 | 64 | 24388 | 291.11 | 1.76 | 994.8 | 761.8 |
| LIZARD-432 | 54 | 1024 | 64 | 25326 | 296.66 | 1.73 | 1013.7 | 776.3 |
| LIZARD-448 | 56 | 1024 | 64 | 26264 | 302.10 | 1.69 | 1032.3 | 790.6 |
| LIZARD-464 | 58 | 1024 | 64 | 27202 | 307.45 | 1.67 | 1050.6 | 804.6 |
| LIZARD-480 | 60 | 1024 | 64 | 28140 | 312.71 | 1.64 | 1068.6 | 818.3 |
| LIZARD-496 | 62 | 1024 | 64 | 29078 | 317.87 | 1.61 | 1086.2 | 831.8 |
| LIZARD-512 | 64 | 1024 | 64 | 30016 | 322.96 | 1.59 | 1103.6 | 845.1 |
| LIZARD-528 | 66 | 1024 | 64 | 30954 | 327.97 | 1.56 | 1120.7 | 858.3 |
| LIZARD-544 | 68 | 1024 | 64 | 31892 | 332.90 | 1.54 | 1137.6 | 871.2 |
| LIZARD-560 | 70 | 1024 | 64 | 32830 | 337.76 | 1.52 | 1154.2 | 883.9 |
| LIZARD-576 | 72 | 1536 | 64 | 33768 | 342.55 | 2.24 | 1170.6 | 896.4 |
| LIZARD-592 | 74 | 1536 | 64 | 34706 | 347.28 | 2.21 | 1186.7 | 908.8 |
| LIZARD-608 | 76 | 1536 | 64 | 35644 | 351.94 | 2.18 | 1202.6 | 921.0 |
| LIZARD-624 | 78 | 1536 | 64 | 36582 | 356.54 | 2.15 | 1218.4 | 933.0 |
| LIZARD-640 | 80 | 1536 | 64 | 37520 | 361.08 | 2.13 | 1233.9 | 944.9 |
| LIZARD-656 | 82 | 1536 | 64 | 38458 | 365.57 | 2.10 | 1249.2 | 956.6 |
| LIZARD-672 | 84 | 1536 | 64 | 39396 | 370.00 | 2.08 | 1264.4 | 968.2 |
| LIZARD-688 | 86 | 1536 | 64 | 40334 | 374.38 | 2.05 | 1279.3 | 979.7 |
| LIZARD-704 | 88 | 1536 | 64 | 41272 | 378.71 | 2.03 | 1294.1 | 991.0 |
| LIZARD-720 | 90 | 1536 | 64 | 42210 | 382.98 | 2.01 | 1308.7 | 1002.2 |
| LIZARD-736 | 92 | 1536 | 64 | 43148 | 387.22 | 1.98 | 1323.2 | 1013.3 |
| LIZARD-752 | 94 | 1536 | 64 | 44086 | 391.40 | 1.96 | 1337.5 | 1024.3 |
| LIZARD-768 | 96 | 1536 | 64 | 45024 | 395.54 | 1.94 | 1351.7 | 1035.1 |
| LIZARD-784 | 98 | 1536 | 64 | 45962 | 399.64 | 1.92 | 1365.7 | 1045.8 |
| LIZARD-800 | 100 | 1536 | 64 | 46900 | 403.70 | 1.90 | 1379.5 | 1056.4 |
| LIZARD-816 | 102 | 1536 | 64 | 47838 | 407.72 | 1.88 | 1393.2 | 1066.9 |
| LIZARD-832 | 104 | 1536 | 64 | 48776 | 411.70 | 1.87 | 1406.8 | 1077.4 |
| LIZARD-848 | 106 | 1536 | 64 | 49714 | 415.64 | 1.85 | 1420.3 | 1087.7 |
| LIZARD-864 | 108 | 1536 | 64 | 50652 | 419.54 | 1.83 | 1433.6 | 1097.9 |
| LIZARD-880 | 110 | 1536 | 64 | 51590 | 423.41 | 1.81 | 1446.9 | 1108.0 |
| LIZARD-896 | 112 | 1536 | 64 | 52528 | 427.24 | 1.80 | 1459.9 | 1118.0 |
| LIZARD-912 | 114 | 1536 | 64 | 53466 | 431.03 | 1.78 | 1472.9 | 1128.0 |
| LIZARD-928 | 116 | 1536 | 64 | 54404 | 434.80 | 1.77 | 1485.8 | 1137.8 |
| LIZARD-944 | 118 | 1536 | 64 | 55342 | 438.53 | 1.75 | 1498.5 | 1147.6 |
| LIZARD-960 | 120 | 1536 | 64 | 56280 | 442.23 | 1.74 | 1511.2 | 1157.3 |
| LIZARD-976 | 122 | 1536 | 64 | 57218 | 445.90 | 1.72 | 1523.7 | 1166.9 |
| LIZARD-992 | 124 | 1536 | 64 | 58156 | 449.54 | 1.71 | 1536.2 | 1176.4 |
| LIZARD-1008 | 126 | 1536 | 64 | 59094 | 453.15 | 1.69 | 1548.5 | 1185.8 |
| LIZARD-1024 | 128 | 1536 | 64 | 60032 | 456.74 | 1.68 | 1560.8 | 1195.2 |

## 4. The border

Normative except 4.10 and 4.11. Sources: `liblizard/src/layout.h`; `liblizard/src/layout.c` `thin_init`, `thin_corner`, `ob_thin_track`,
`ob_thin_reserve`, `ob_layout_set_fmt`; `liblizard/src/focus.c` `init`, `focus_encode`, `resample_init`, `focus_paint_rgba`;
`liblizard/src/focus.h` `FOCUS_RING`, `FOCUS_RING_DEFAULT`, `FOCUS_QUIET`; `liblizard/src/layout.h` `OB_THIN_CORNER`, `FOCUS_CP`. The border
is what the code paints: a 12-module corner mark with a 1-module inner edge, a 3-module guard, 15 modules in all. An
even border (a 13-module mark, a 2-module guard, every feature but the rim whole 2 x 2 cells) was built on 2026-09-27
and reverted that day (section 12).

### 4.1 Units and coordinates

- The border's unit is the module. A symbol in ring B is `S = 2B + 30` modules a side: `span = 2B` modules of picture
  inside 15 modules of border on each side (`liblizard/src/focus.h:FOCUS_RING`, B = 32, 64, 96, 128). The module count (94, 158,
  222, 286) identifies the ring before the word is read, and says nothing about n. The sizes in modules and pixels for
  each ring and picture are in 3.3.
- Module (x, y): x grows to the right and y grows downwards, and (0, 0) is the top-left module as displayed. In
  continuous module coordinates, module (x, y) covers `[x, x + 1) x [y, y + 1)` and its centre is `(x + 0.5, y + 0.5)`.
- The depth of a module is `d = min(x, y, S - 1 - x, S - 1 - y)`, which is 0 on the outer edge.
- Sides are numbered 0 top, 1 right, 2 bottom, 3 left. The position t along a side is x on the top and bottom and y on
  the left and right, so every side counts left to right or top to bottom; none runs round the symbol. The module at
  along t and depth d is `(t, d)` on the top, `(S - 1 - d, t)` on the right, `(t, S - 1 - d)` on the bottom and
  `(d, t)` on the left.
- Levels: in the outer border (depths 0 to 11) the drive (`focus_encode`) is 0.0 for dark and 1.0 for light. It is
  painted as grey `(int)(v * 255 + 0.5)`, clamped to 0 to 255 (`focus_paint_rgba`, 6.8), so every pixel of the outer
  border is 0 or 255, as is the margin. The guard (depths 12 to 14) is resampled picture and takes grey levels (4.4).
  Colour is never used.

### 4.2 Whole pixels a module

- The border is painted at `pxm = ceil(n / span - 1e-4)` pixels a module (`liblizard/src/focus.c:init`). The `1e-4` keeps a span
  that divides n at exactly n / span.
- Every module of the outer border (depths 0 to 11, the corner marks included) MUST be painted as a `pxm x pxm` block
  of one level, covering drive pixels `[x * pxm, (x + 1) * pxm) x [y * pxm, (y + 1) * pxm)`. No module is split. The
  guard's modules are not one level: the resampler paints them pixel by pixel, as it paints the picture (4.4, 6.8).
- The picture is on its own grid. It keeps its n samples and is resampled (Lanczos-3, periodic) to fill its span
  modules, upscaled by `pxm / scale` and never shrunk: 1, an exact copy, for every picture in the 32 and 64 rings, for
  n = 384, 768 and 1536 in the 96 ring and for every n but 384 in the 128 ring; 1.5 (n = 256) or 1.125 (n = 512, 1024)
  in the 96 ring and 4/3 (n = 384) in the 128 ring (3.3). Picture
  sample x (0 to n - 1) is centred at module coordinate `15 + (x + 0.5) / scale` on each axis (6.8, 6.9). Nothing
  light lies between the border and the picture: the guard (4.4) touches both.

### 4.3 Depths, outside in

Along the middle of any side:

| depth | name | content |
|---|---|---|
| -2, -1 | margin | light, painted by the codec (4.8) |
| 0 | rim | light |
| 1, 2 | ring | dark all the way round, broken only beside the corner marks (4.5) |
| 3, 4 | gap | light |
| 5, 6 | band | 2 x 2 cells: timing track and format word (4.6) |
| 7 to 11 | | light |
| 12 to 14 | guard | the picture's periodic extension (4.4) |
| 15 to S - 16 | picture | span = 2B modules |

The table has two exceptions. The four 12 x 12 corner squares are the corner marks (4.5). In the band's reserve, the 15
modules at each end of each side, depths 5 and 6 are light (4.6). The ring is the reference receiver's anchor. Among
symbols of one ring, the outer border differs only in the word's cells, whatever picture the ring carries; the guard
follows the picture.

### 4.4 Guard

Depths 12 to 14 (3 modules, `FOCUS_CP` = 3) are the picture wrapped round its own edge. The encoder resamples over
`span + 6` modules from module 12 and reads the picture as periodic in both axes, so the guard is a torus wrap, corners
included, and not four strips. In drive pixels, with `o = 12 * pxm`: for `0 <= X < 3 * pxm`, column `o + X` equals
column `o + X + span * pxm` over the whole resampled region, row `o + X` equals row `o + X + span * pxm`, and the
corners follow. An encoder MUST paint the guard this way. The decoder models blur as a gain on each coefficient, which
holds for a circular convolution and fails for a picture with anything else beside it.

### 4.5 Corner marks

A mark fills the 12 x 12 square at each corner (`OB_THIN_CORNER` = 12), and there the depth table does not apply. For
a module (x, y), let `cx = min(x, S - 1 - x)` and `cy = min(y, S - 1 - y)`. If both are under 12, the module belongs to
a mark and is (`thin_corner`):

- light if `cx` or `cy` is 0 (the rim's row) or 11 (the inner edge, 1 module);
- otherwise, with `e = min(cx, cy)`: dark if e is 1 or 2 (the ring, carried in), light if e is 3 or 4 (the gap, carried
  in), dark if e is 5 or more (the core).

Each mark is therefore a 6 x 6 dark core at modules 5 to 10 from its corner on both axes. The gap and the ring wrap its
two outer sides, and its inner sides are light at module 11. The ring is broken there: at depths 1 and 2, modules 11 and
`S - 12` along each side are light (1 module). A scan through the core from the corner (any row or column 5 to 10 of the
mark) sees light 1, dark 2, light 2, dark 6, then light (the inner edge and what lies past the mark), which gives
ring : gap : core = 2 : 2 : 6 (`liblizard/src/acquire.c:mark_ratio`).

| form | when | what a scan finds | extent from the corner | centre from the corner |
|---|---|---|---|---|
| gapped | sharp | the 2 : 2 : 6 cross-section | core, modules 5 to 10 | (8.0, 8.0) |
| merged | blur has closed the gap | one dark square, 10 x 10 | modules 1 to 10 | (6.0, 6.0) |

Centres are continuous module coordinates measured in from the mark's own corner. The gapped centres are (8, 8),
(S - 8, 8), (S - 8, S - 8) and (8, S - 8), with adjacent ones `S - 16` modules apart. The merged centres are 6 in from
each edge, `S - 12` apart (`liblizard/src/acquire.c:mark_form_mid`, `ob_mark_geom`). The reference receiver tries the gapped form
first and the merged form only when the gapped one settles nothing. There are four marks and not three, because a
fourth measured corner is what carries perspective.

The border is 15 modules because the mark (12) and the guard (3) may not share depths. A mark never reaches into the
picture or its guard, and the border grows outwards only (`liblizard/src/focus.c:init`).

### 4.6 The band

- The band is depths 5 and 6, in cells of 2 x 2 modules. On each side it runs from along 15 to `S - 16`. The reserve,
  15 modules at each end, is `ob_thin_reserve(12)` = 12 + 3 (14, `(corner + 3) & ~1`, until 2026-09-27). So no band
  cell touches a mark: along 11 is the mark's light inner edge and along 12 to 14 are light.
- Band cells a side: `B = (S - 30) / 2`, the ring's own number (32, 64, 96, 128). Band cell b (0 to B - 1) covers along
  `[15 + 2b, 17 + 2b)`.
- Runs of 4 cells alternate, starting with the track (`OB_BAND_RUN` 4). Cell b is a track cell when `floor(b / 4)` is
  even and a word cell when it is odd. Track cell j is band cell `8 * floor(j / 4) + j mod 4`, and word cell i is band
  cell `8 * floor(i / 4) + 4 + i mod 4` (`liblizard/src/layout.h` `OB_TRACK_AT`, `OB_WORD_AT`). B is a multiple of 8, so each
  side holds B / 2 of each kind (16, 32, 48, 64) and ends on a whole word run.
- Inside a run, the track never has more than 2 equal cells (4 modules) in a row, because each Manchester pair holds a
  bit and its complement. The band as a whole has no such bound, since word cells carry data.

### 4.7 Timing track

The track is Manchester coded from a hash that differs per side (`ob_thin_track`). Track cells 2p and 2p + 1, which
are adjacent and in one run, carry a bit and its complement. The arithmetic is unsigned 32-bit and `>>` is a logical
shift. s is the side (0 to 3) and `p = floor(j / 2)`:

```
v = (s * 4099 + p) * 2654435761 + 40503     mod 2^32   (0x9E3779B1, 0x9E37)
v = v ^ (v >> 15)
v = v * 2246822519                          mod 2^32   (0x85EBCA77)
v = v ^ (v >> 13)
track cell j is dark  iff  ((v ^ j) & 1) == 1
```

Worked example: side 0, pair 0 gives v = 0x00009E37, then 0x00009E36, 0xC6B2271A, 0xC6B4128B. Bit 0 is 1, so track cell
0 is dark and track cell 1 is light.

Track cells 0 to 15 of each side (1 is dark). This is the 32 ring's whole track and the start of the 64, 96 and 128
rings' (read from the painted cells in each ring, 2026-09-27 evening):

| side | track cells 0 to 15 |
|---|---|
| 0 | `1010 1010 1001 1010` |
| 1 | `1010 0110 1010 0110` |
| 2 | `0101 0110 1010 0101` |
| 3 | `0110 1001 0101 0110` |

The track does not depend on n, the version or the payload, and a ring only sets its length. The four sequences
differ, so the track alone tells a receiver the rotation and mirroring (8 hypotheses).

### 4.8 Margin

- The margin is the codec's (2026-09-24): `liblizard/src/focus.h:FOCUS_QUIET` = 2 modules of light, `2 * pxm` pixels (4, 6,
  8, 12, 16 and 24 at n = 256 to 1536 in the default 64 ring; 2 to 48 over every ring and picture, 3.3), outside depth 0
  on every side. With the rim, that puts 3 modules of light outside the ring, and the page may put anything at the
  canvas's edge. An encoder MUST paint the margin. The painted image, symbol and margin, is `(S + 4) * pxm` pixels a
  side. The margin is outside the module grid, whose origin stays the symbol's outer corner (4.1), so nothing a decoder
  reads moves with it: the picture grid (6.9), the kind map, the word's cells and the marks are where they were before
  the margin moved into the codec.
- `focus_paint_rgba(f, drive, rgba)` is how the reference paints (`liblizard/src/focus.h`): the drive as grey (6.8), then
  `FOCUS_QUIET` modules of white, every one pxm pixels, as RGBA with R = G = B and alpha 255, `W = S pxm + 4 pxm`
  pixels a side. The drive
  `focus_encode` writes has no margin; the paint adds it. The wasm exports it as `focus_tx_rgba` and the margin as
  `focus_quiet`, so the JS keeps no copy of the number: `liblizard/sim/ob.mjs` reads it into `QUIET` when `init()` loads the
  codec, a live binding and not a top-level await (a module worker whose imports await drops the messages that arrive
  meanwhile), and `liblizard/sim/lizard_pick.mjs` re-exports it as `OB_QUIET`.
- The reference finder assumes light outside the ring. With dark noise painted right up to the ring, LIZARD-256 kept
  64 to 388 of 1446 blocks (`archive/build-scratch-2026-09/ring_race.mjs`, simulator, STATUS.md 2026-09-23, measured under the previous
  n / 4 + 30 border).
- A guard ring (a closed black ring round 3 modules of light) was a sender option from 2026-09-23 and was deleted on
  2026-09-26, judged useless; no painter or page offers it.
- `liblizard/sim/phy.mjs` takes `spec.quiet`, a margin in modules other than the codec's, for simulator experiments on the float
  drive. The page-pixel path (`frameRGBA`, the codec's own paint) refuses it rather than paint a margin the spec does
  not say.

### 4.9 One corner

LIZARD-16 in the 32 ring (S = 94) at fps 0, top-left corner, modules x 0 to 37 and y 0 to 17, taken from the layout's
cells (`ob_test_mesh_tables_out`, 2026-09-27 evening). Legend: `#` dark, `.` light, `T` / `t` track cell dark /
light, `W` / `w` word cell dark / light, `g` guard, `P` picture. The other corners are this one reflected, except for
the band's cells, which follow 4.6 and 4.7. Every symbol in the 32 ring has this border except for its word's cells,
whatever picture it carries, and this corner, word cells aside, is the same in all four rings (read from the cells of
LIZARD-16, -64 and -256 in each).

```
                1         2         3
   x  01234567890123456789012345678901234567
   0  ......................................
   1  .##########.##########################
   2  .##########.##########################
   3  .##...................................
   4  .##...................................
   5  .##..######....TTttTTttwwWWwwwwTTttTTt
   6  .##..######....TTttTTttwwWWwwwwTTttTTt
   7  .##..######...........................
   8  .##..######...........................
   9  .##..######...........................
  10  .##..######...........................
  11  ......................................
  12  .##.........gggggggggggggggggggggggggg
  13  .##.........gggggggggggggggggggggggggg
  14  .##.........gggggggggggggggggggggggggg
  15  .##..tt.....gggPPPPPPPPPPPPPPPPPPPPPPP
  16  .##..tt.....gggPPPPPPPPPPPPPPPPPPPPPPP
  17  .##..TT.....gggPPPPPPPPPPPPPPPPPPPPPPP
```

The same symbol's whole band at depth 5 (depth 6 is identical), along 0 to 93 of each side, so the marks' cores show at
both ends:

```
top    .##..######....TTttTTttwwWWwwwwTTttTTttWWWWwwwwTTttttTTwwwwWWWWTTttTTttWWWWWWWW....######..##.
right  .##..######....TTttTTttwwwwwwwwttTTTTttwwwwWWwwTTttTTttwwwwwwWWttTTTTttwwWWwwww....######..##.
bottom .##..######....ttTTttTTwwwwwwwwttTTTTttwwwwwwwwTTttTTttWWwwwwwwttTTttTTwwwwWWWW....######..##.
left   .##..######....ttTTTTttwwWWWWWWTTttttTTwwwwWWwwttTTttTTWWwwwwWWttTTTTttwwWWwwww....######..##.
```

### 4.10 Mesh node sites (non-normative)

Nothing is painted for these. The layout defines a lattice of registration nodes, the same on both axes (`thin_init`):
positions `3.5 + 16k` for every k with `3.5 + 16k < S - 11.5`, then `S - 3.5`: B / 8 + 2 a side. They depend on the ring
alone.

| ring B | S | node positions | count |
|---|---|---|---|
| 32 | 94 | 3.5, 19.5, 35.5, 51.5, 67.5, 90.5 | 6 |
| 64 | 158 | 3.5 to 131.5 in steps of 16, then 154.5 | 10 |
| 96 | 222 | 3.5 to 195.5 in steps of 16, then 218.5 | 14 |
| 128 | 286 | 3.5 to 259.5 in steps of 16, then 282.5 | 18 |

The last step is 23 modules in every ring (read from `ob_test_mesh_tables_out`, 2026-09-27 evening). The nodes a
receiver measures are on the border: the rows y = 3.5 and y = S - 3.5 and the columns x = 3.5 and x = S - 3.5. That is
depth 3.5, the middle of the gap's outer row (`OB_NODE_LINE = (5 + 2) / 2`). The four corner nodes, (3.5, 3.5) and its
reflections, are the corners of the registration quad (`liblizard/src/acquire.c:corner_coords`). The reference receiver refines
the corner nodes first and then every other border node. Each is matched against the layout's own modules in a window 13
modules along the side and 5 across it (depths 1 to 5); at a corner node the window reaches `OB_THIN_CORNER - 1` = 11
modules either way, so covers modules 0 to 14 on both axes (`refine_node`). Word cells and the picture are left out of
the window. Interior lattice points are filled from the border (`refine_node`, `border_fill`), since the symbol has no
marks inside the picture.

### 4.11 Settings in liblizard/src/ that are not the format (non-normative)

`ob_cfg_t` and `focus_init` take switches that paint other borders, and some build macros change the border too. The
switches are a corner mark of another size or none (`corner`), a filled mark (`corner_filled`), edge and centre marks
(`edge`, `centre`), a 1 : 1 clock track (`track_alt`), a wider border (`border`), and a span other than the four rings'
64, 128, 192 and 256 (`span`; one under 32 or over n falls back to `min(n, 256)`, `liblizard/src/focus.c:init`). The macros are
`OB_RING_DEEP`, `OB_CELL`, `OB_BAND_RUN` and `FOCUS_CP`; `FOCUS_SLOPE` and `FOCUS_SPAN0`, which set the side from n,
went on 2026-09-27. They exist for experiments. The format is their defaults: a gapped 12-module mark with a 1-module
inner edge at each corner, no other marks, the hashed Manchester track, a 15-module border, bands 2 modules deep, 2 x 2
cells, runs of 4, a 3-module guard and 2B modules of picture in one of the four rings.

## 5. The format word

Normative except 5.4 and 5.5. Sources: `liblizard/src/fmt.h`; `liblizard/src/fmt.c` `ob_fmt_side_cells`, `ob_fmt_bytes`, `ob_fmt_encode`,
`ob_fmt_bit`, `ob_fmt_decode`; `liblizard/src/rs.c` `init`, `rs_encode`; painted by `liblizard/src/layout.c:ob_layout_set_fmt`, called
from `liblizard/src/focus.c:fmt_paint`. One code a ring over every word cell (2026-09-24, one code over every word cell;
2026-09-27, the rings).

The word is the only information the border carries, and it is what makes the symbol self-describing: the ring is
found, and the word it carries says what is inside it. Everything about the content (sub-channels, picture size n,
blocks, code, rate) follows from the word's second byte (section 3, `liblizard/src/focus.c:focus_n_for`); nothing about it
follows from the ring.

### 5.1 Fields

`bytes` bytes, `cw[0]` to `cw[bytes - 1]`. The size comes from the band, never from a table of n. With B the band cells a
side (4.6), the word cells a side are the whole word runs, and the codeword fills whole bytes of them on every side:

```
W = 4 floor(B / 8)        bytes = min(36, 4 floor(W / 8))
```

(`liblizard/src/fmt.c` `ob_fmt_side_cells`, `ob_fmt_bytes`; `OB_FMT_BYTES_MAX`). In the four rings B is a multiple of 8, so W = B
/ 2 = (S - 30) / 4 (16, 32, 48, 64) and bytes = W / 2 (8, 16, 24, 32). A receiver has B from the ring it registered
(4.1) before it reads the word, and the same picture in two rings carries two different codewords. A lab layout at
another module count gets the code its band holds (S = 126, the morning's 48 ring: B = 48, W = 24, RS(12, 3)); a word
run its band cuts short is not a word run, and stays light. The data is the same 3 bytes in every ring:

| byte | field | value |
|---|---|---|
| `cw[0]` | magic | `0x4C` (ASCII `L`) |
| `cw[1]` | version | sub-channels / 8, 1 to 128; the blocks a frame and their rates follow from it (3.1). It names the picture: n = `focus_n_for(8 * version)` (3.2) |
| `cw[2]` | fps | whole frames a second the sender means to paint at, 1 to 255; 0 means not stated |
| `cw[3]` to `cw[bytes - 1]` | parity | Reed-Solomon, 5.2 |

- The 128 formats, LIZARD-8 to LIZARD-1024 in steps of 8, have versions 1 to 128: LIZARD-k has version k / 8. An
  encoder MUST write version = subch / 8 (3.1; until 2026-10-01 only the 64 even versions were painted).
- A decoder MUST accept every ladder version in every ring, and reads an odd version as it reads a ladder one, with no
  flag (3.1). Any ring may carry any picture (3.3), so a decoder MUST NOT reject a word for the ring it was read in. The
  reference decoder accepts every version from 1 to 128 (`OB_FMT_VERSION_MAX`), and the codec builds any multiple of 8
  sub-channels up to 1024 and refuses more.
- fps is the rate the sender intends, not the rate it achieves. No decode decision depends on it. A sender MAY change
  it between frames: only the word's cells change (`liblizard/src/focus.c:focus_fmt_fps`, 9.2).
- Nothing else is in the word. The mode is always LDPC, the rates are the profile's for the version (3.1,
  `liblizard/src/focus.c:focus_tiers_for`; one rate 3/4 before 2026-10-07), and the bit map is always LINEAR, the only one the format has, so the word names none (2026-09-24; 7.3). Everything about
  the transfer (fountain seeds, file length, hash, name) is in the header block (7.7).

### 5.2 Reed-Solomon (bytes, 3)

- Field: GF(2^8). Addition is XOR. Multiplication is modulo `p(x) = x^8 + x^4 + x^3 + x^2 + 1` (0x11D), with primitive
  element alpha = 0x02 (`liblizard/src/rs.c:init`).
- One code a ring, over every word cell, with `nroots = bytes - 3` parity bytes:

| ring B | modules a side S | word cells a side W | code | nroots | distance | bytes a side |
|---|---|---|---|---|---|---|
| 32 | 94 | 16 | RS(8, 3) | 5 | 6 | 2 |
| 64 | 158 | 32 | RS(16, 3) | 13 | 14 | 4 |
| 96 | 222 | 48 | RS(24, 3) | 21 | 22 | 6 |
| 128 | 286 | 64 | RS(32, 3) | 29 | 30 | 8 |

- `bytes` and W are 5.1's; the table is the four rings. Until 2026-09-27 the code followed the picture size, RS(8, 3),
  (12, 3), (20, 3) and (36, 3) at n = 256 to 2048, and that morning's rings (32, 48, 64) took RS(8, 3), (12, 3) and
  (16, 3). RS(8, 3) is unchanged since 2026-09-24; RS(12, 3), RS(20, 3) and RS(36, 3) are gone from the format.
- Generator: roots alpha^0 to alpha^(nroots - 1), so the first consecutive root is alpha^0 (`rs_encode`). Subtraction
  is addition in this field. `g(x) = (x + 1)(x + alpha) ... (x + alpha^(nroots - 1))`, coefficients from x^nroots down
  to x^0, in hex:

  ```
  nroots  5: 01 1f c6 3f 93 74
  nroots 13: 01 89 49 e3 11 b1 11 34 0d 2e 2b 53 84 78
  nroots 21: 01 2c f3 0d 83 31 84 c2 43 d6 1c 59 7c 52 9e f4 25 ec 8e 52 ff 59
  nroots 29: 01 e4 c1 c4 30 aa 56 50 d9 36 8f 4f 20 58 ff 57 18 0f fb 55 52 c9 3a 70 bf 99 6c 84 8f aa
  ```

  The first is `x^5 + 0x1F x^4 + 0xC6 x^3 + 0x3F x^2 + 0x93 x + 0x74`. Each line is `rs_encode`'s parity of x^nroots
  (data `00 00 01`), with the leading 01 (2026-09-27 evening, `liblizard/src/rs.c` compiled natively).
- Byte order: `cw[0]` is the highest-degree coefficient, `c(x) = cw[0] x^(bytes - 1) + cw[1] x^(bytes - 2) + ... +
  cw[bytes - 1]`.
- Systematic: `cw[3]` to `cw[bytes - 1]` are the coefficients of x^(nroots - 1) down to x^0 of
  `(cw[0] x^(bytes - 1) + cw[1] x^(bytes - 2) + cw[2] x^(bytes - 3)) mod g(x)`. Equivalently `c(alpha^i) = 0` for
  i = 0 to nroots - 1, which is what a decoder's syndromes check.
- Each code is RS(255, 255 - nroots) shortened to `bytes`. Its distance is nroots + 1: it corrects e errors and f
  erasures with `2e + f <= nroots`.
- **Acceptance.** A decoder MUST NOT take a correction past the code's bound, `2e + f <= nroots` for e errors and f
  erasures (`liblizard/src/rs.c` `solve` refuses it), and the garbage rates in 5.4 assume it. A decoder MUST reject a corrected
  word whose `cw[0]` is not `0x4C` or whose version is outside 1 to 128. RS can miscorrect past its distance, and a wrong
  version sends the receiver to another picture: that frame's blocks fail their CRCs, and a receiver that holds the
  last word (5.5) decodes the wordless frames after it at the wrong picture until a word reads again.

**Test vectors.** `cw[0]` to `cw[bytes - 1]`, in hex:

```
LIZARD-16    n  256  ring  32  version   2  fps   0  RS(8, 3)   4c 02 00 72 3f 14 83 94
LIZARD-512   n 1024  ring  64  version  64  fps  24  RS(16, 3)  4c 40 18 43 c2 36 ee 95 dc 07 2a 1c 61 cb 63 fc
LIZARD-1024  n 1536  ring  64  version 128  fps 240  RS(16, 3)  4c 80 f0 1a 03 b0 6a 27 2b f1 25 d4 09 be 9d d9
LIZARD-576   n 1536  ring  64  version  72  fps  30  RS(16, 3)  4c 48 1e 8a 58 89 b2 ff dc 13 58 1b 53 70 20 83
LIZARD-16    n  256  ring  64  version   2  fps   0  RS(16, 3)  4c 02 00 bf 8a 27 26 53 d9 ff 5f 11 92 79 47 ed
LIZARD-256   n  768  ring  96  version  32  fps 120  RS(24, 3)  4c 20 78 2d 28 b4 96 c1 7b 9c 80 f7 3a 2b 38 5d
                                                                b0 ea 9d 62 86 a4 bf 2e
LIZARD-64    n  384  ring 128  version   8  fps  60  RS(32, 3)  4c 08 3c 78 7c 7c 41 6d 2b 6b 3f c1 80 75 00 4a
                                                                d5 cd 92 b1 94 5a b6 16 5f 18 ed 02 1c 05 3f f6
```

How they were made (2026-09-27 evening): each symbol painted by `liblizard/build/ob.wasm` through `liblizard/sim/ob.mjs` in the ring named
(`_focus_setup(n, subch, 1, 2, 2B, ...)`, `_focus_fmt_fps_set(fps)`), the word
read back out of the layout's cells (`ob_test_mesh_tables_out`) with the placement in 5.3, and compared with `liblizard/src/fmt.c`
compiled natively and with `liblizard/gpu/wordcode.mjs`, a separate encoder in JS; all three agree. `liblizard/test/vectors.mjs`
reads all seven back out of the painted cells. `scripts/gpu/word_check.mjs` checked the JS encoder against the wasm's painted
cells on the four rings (256 words, 2026-09-27, evening; 192 on the morning's rings). The word does not depend on n: when the picture sizes changed
(3.2), LIZARD-64, -1024 and -576 painted the same outer border, module for module, at 384 and 1536 as at 512 and 2048
(2026-09-27). The RS(8, 3) word is unchanged from 2026-09-24. `liblizard/test/fmt_test.c` holds these seven since 2026-10-04 (that
day's five before, four of them in codes no ring uses: RS(12, 3), RS(20, 3), RS(36, 3)).

### 5.3 Placement

- A cell shows one bit: dark is 1, light is 0.
- Byte k goes to side `k mod 4` (0 top, 1 right, 2 bottom, 3 left), on word cells `8 * floor(k / 4)` to
  `8 * floor(k / 4) + 7` of that side, most significant bit first (`ob_fmt_bit`). Put the other way: word cell c (0 to
  W - 1) of side s shows bit `7 - (c mod 8)` of byte `4 * floor(c / 8) + s`. Side s holds bytes s, s + 4, s + 8, ...:

| side | word cells 0 to 7 | 8 to 15 | 16 to 23 | 24 to 31 | 32 to 39 | 40 to 47 | 48 to 55 | 56 to 63 |
|---|---|---|---|---|---|---|---|---|
| 0 top | `cw[0]` magic | `cw[4]` | `cw[8]` | `cw[12]` | `cw[16]` | `cw[20]` | `cw[24]` | `cw[28]` |
| 1 right | `cw[1]` version | `cw[5]` | `cw[9]` | `cw[13]` | `cw[17]` | `cw[21]` | `cw[25]` | `cw[29]` |
| 2 bottom | `cw[2]` fps | `cw[6]` | `cw[10]` | `cw[14]` | `cw[18]` | `cw[22]` | `cw[26]` | `cw[30]` |
| 3 left | `cw[3]` | `cw[7]` | `cw[11]` | `cw[15]` | `cw[19]` | `cw[23]` | `cw[27]` | `cw[31]` |

  The 32 ring uses the first two columns, the 64 ring four, the 96 ring six, the 128 ring all eight.
- Word cells are numbered along the side in the direction its modules count (4.1), and where they sit in the band is
  in 4.6. A side has W word cells (5.1), and every one carries a bit of the codeword: there are no copies and no
  spare cells. An encoder MUST paint every word cell from the codeword. (A lab band whose W is
  not a multiple of 8, or is past 72, paints the cells past the codeword light; no ring has any.)
- The reference layout flags every word cell `CELL_WORD` (`liblizard/src/layout.h`), so no registration template treats one as
  fixed (5.4).
- The shortest word, RS(8, 3), needs 16 word cells a side, which the 32 ring holds exactly.
- A side lost costs a quarter of the bytes (2, 4, 6 or 8). A faded side reads at its local level, so its bytes are the
  least confident and the erasure steps take them (5.4): the word reads with two faded sides in every ring, and with
  three in the 96 and 128 rings (5.4's table). A side under glare is taken the same way. In the simulator on the n / 8 +
  60 border (`scripts/exp/fmt_word.mjs` DAMAGE, glare saturating one side's band and ring, 32 frames a cell, 2026-09-24), two
  glared sides at n = 256, RS(8, 3) as in today's 32 ring, are 4 wrong bytes, twice what errors-only RS corrects; they
  read on 31 and 32 of 32 frames with the steps to 4 erasures (the frame lost failed orientation) and on 16 of 32 with
  the steps stopped at 2. Those 16 are the frames where side 3 is one of the pair, which fits a glared side's cells
  reading light: `cw[7]` of that word is `0x00`, so one glared byte reads right. Glare on one side read on 32 of 32 at
  every size, on two 32 of 32 at n >= 512, and on three almost never, orientation failing first on most (10.1;
  STATUS.md, "C decoder fixes"). On the morning's three rings (32, 48, 64; `DAMAGE=1`, 2,048 frames, told, against the
  build before, 2026-09-27) the told words went from 1,476 to 1,481 and wrong words stayed 0; not repeated on the four
  rings.

LIZARD-16 in the 32 ring at fps 0, the 16 word cells of each side in order (the whole band is drawn in 4.9):

| side | word cells 0 to 15 |
|---|---|
| 0 | `0100 1100 0011 1111` |
| 1 | `0000 0010 0001 0100` |
| 2 | `0000 0000 1000 0011` |
| 3 | `0111 0010 1001 0100` |

LIZARD-64 in the 128 ring at fps 60, the 64 word cells of each side in order (read from the painted cells, 2026-09-27
evening):

| side | word cells 0 to 63 |
|---|---|
| 0 | `0100 1100 0111 1100 0010 1011 1000 0000 1101 0101 1001 0100 0101 1111 0001 1100` |
| 1 | `0000 1000 0111 1100 0110 1011 0111 0101 1100 1101 0101 1010 0001 1000 0000 0101` |
| 2 | `0011 1100 0100 0001 0011 1111 0000 0000 1001 0010 1011 0110 1110 1101 0011 1111` |
| 3 | `0111 1000 0110 1101 1100 0001 0100 1010 1011 0001 0001 0110 0000 0010 1111 0110` |

### 5.4 Reading the word (non-normative)

This is what the reference receiver does: `liblizard/src/acquire.c:ob_thin_read_fmt` for the soft values, then
`liblizard/src/fmt.c:ob_fmt_decode`. Any read that recovers the codeword will do. The word is read after registration, in the
ring that registered, and before the picture. Points are in continuous module coordinates (4.1), mapped through the
registration and mesh (`ob_map_point`), and sampled bilinearly as luminance in 0 to 1 (`liblizard/src/internal.h:sample`). A
point at along t and depth d is `(t, d)` on the top, `(S - d, t)` on the right, `(t, S - d)` on the bottom and
`(d, t)` on the left (`side_xy`). Here t and d are continuous: d is the distance in from the symbol's outer edge, so
depth 2.0 is the ring's middle line, 4.0 the gap's and 6.0 the band's. That is why the right and bottom sides use
`S - d`, where 4.1's integer module index uses `S - 1 - d`.

1. **Contrast.** On every side, for t = 15, 31, 47, ... while `t < S - 15`, sample the ring at (along `t + 0.5`, depth
   2.0) and the gap at (along `t + 0.5`, depth 4.0). `con` is the mean gap sample minus the mean ring sample. Below
   0.08 the reader gives up.
2. **Local level.** For each Manchester pair p of the side's track (track cells 2p and 2p + 1, 4.7), `mid[p]` is the
   mean of the two cells' samples at their centres (depth 6.0). A side has `npair` = B / 4 = 8, 16, 24, 32 pairs in
   the 32, 64, 96 and 128 rings.
3. **Cells.** Word cell i (0 to W - 1) of side s sits in band cell b (4.6) and is sampled at its centre, along
   `15 + 2b + 1`, depth 6.0. With `a = floor(2b / 8)`, its level is the mean of `mid[p]` for p from `max(0, a - 3)` to
   `min(npair - 1, a + 3)`. `q[s][i] = sample - level`.
4. **Threshold.** Start from `ref = 0`, then up to four times: split the 4W values of q at ref, a value equal to ref
   going with the lighter part (`q >= ref`), stop if either part has fewer than W / 2, and set ref to the midpoint of
   the two parts' means.
5. **Soft value.** `z[s][i] = clamp((ref - q[s][i]) / con, -0.5, 0.5)` for side s and word cell i, positive for
   dark.

Decoding (`ob_fmt_decode`):

1. Byte k is built from the 8 soft values of side `k mod 4` starting at cell `8 * floor(k / 4)`, most significant bit
   first, with a bit of 1 where `z > 0`. The byte's confidence is the smallest `|z|` among its 8 bits.
2. Errors-only RS: up to `floor(nroots / 2)` byte errors (2, 6, 10, 14).
3. Then erase the least confident bytes (the lower index wins a tie), 2, then 4, and so on up to nroots - 1 (4) in the
   32 ring and nroots - 3 (10, 18, 26) in the 64, 96 and 128 rings, decoding errors and erasures at each step within
   `2e + f <= nroots` (5.2). Every nroots is odd, so the last step leaves one root over the erasures in the 32 ring and
   three in the others. A side read faded, near its local track level, is W / 8 of the least confident bytes; where
   three roots are left, one byte past the last step is corrected as an error.
4. Each result, from step 2 on, is taken only if `cw[0] = 0x4C` and the version is 1 to 128 (5.2); the first that
   passes is the word. fps is taken as it comes. Nothing is checked against the ring: the receiver calls
   `ob_fmt_decode` with versions 1 to 128 in every ring (`liblizard/src/focus.c`). Until 2026-09-27 the receiver told nothing also
   required the version to imply the picture size of the module count it registered; that check went with the rings.

The steps stop at nroots - 3 in the 64, 96 and 128 rings, where going on to nroots - 1, as they did until 2026-09-24,
let garbage through 43 times as often. In the 32 ring they go on to nroots - 1 (2026-09-24): stopping at 2 lost
two faded or glared sides there (`scripts/exp/fmt_word.mjs` DAMAGE, LIZARD-16 on the n / 8 + 60 border: glare on two sides 16 of
32 on each cell against 31 and 32, defocus 4 px 12 against 26). With the size check gone, that step lets garbage through
at 7.8e-6 a read in the 32 ring (2.4e-7 with the check). A step of nroots leaves no root, so RS takes every read and
only the magic and the version stand behind it. What a read of garbage (soft values uniform in -0.5 to 0.5, what a wrong
grid or a quad that is not the symbol gives) is accepted as, exact, with the data bytes of a false codeword uniform:

| per garbage read | RS(8, 3) | RS(16, 3) | RS(24, 3) | RS(32, 3) |
|---|---|---|---|---|
| errors-only RS decodes it | 1.7e-6 | 1.1e-13 | 6.1e-21 | 3.4e-28 |
| some step up to the last (4, 10, 18, 26) decodes it | 4.0e-3 | 9.3e-5 | 9.3e-5 | 9.3e-5 |
| accepted: magic, version 1 to 128 | 7.8e-6 | 1.8e-7 | 1.8e-7 | 1.8e-7 |
| measured over 2,000,000: accepted | 13 | 1 | 0 | 0 |
| with the other last step (2 in the 32 ring, nroots - 1 in the others): magic and version | 1.8e-7 | 7.8e-6 | 7.8e-6 | 7.8e-6 |
| with a step of nroots: magic and version | 2.0e-3 | 2.0e-3 | 2.0e-3 | 2.0e-3 |

The last step sets the rate, not the length: a step that leaves r roots over its erasures passes garbage at the same
rate at every length (r = 3: 9.1e-5 for the step itself; r = 1: 1 in 256). The exact rows are the bounded-distance count
of `liblizard/test/fmt_test.c` (`p_pass`), recomputed for RS(16, 3), RS(24, 3) and RS(32, 3) on 2026-09-27. The measured row's
RS(8, 3) cell is `liblizard/test/fmt_test.c` through `liblizard/src/fmt.c` (2026-09-24, the same schedule measured before the steps were cut
that evening, STATUS.md, "Miscorrection"; 12 in a rerun on 2026-09-27); its RS(16, 3) cell is a scratch copy of that
test through `liblizard/src/fmt.c` (2026-09-27, about 17 s), and its RS(24, 3) and RS(32, 3) cells a scratch harness of the same
reads through the evening's `liblizard/src/fmt.c` (2026-09-27). The reader before the code grew (RS(8, 3) in copies, errors-only
then 2 erasures) accepted 1.8e-7.

**Known limit, the 32 ring.** Side 0 holds `cw[0]` and `cw[4]`. A frame whose side 0 reads clean and whose other three
sides read garbage keeps the real magic, so the version byte is all that stands behind a miscorrection. Measured over
10^7 such reads through `liblizard/src/fmt.c` (side 0 at magnitude 0.4, the rest uniform): accepted 2.0e-3 a read.
That is the steps' rate with the magic already right, 4.0e-3, times the chance the garbage version passes, 1/2. In the
64 ring one clean side leaves 12 garbage bytes, one past the steps' reach of 11 (the table below), and the same counting
gives 9.3e-5 times 1/2, 4.6e-5 a read; the harness below measured 2 wrong words (and 7,699 right) in 10^6 such reads.
In the 96 and 128 rings one clean side leaves 18 and 24 garbage bytes, inside the steps' reach of 19 and 27, so the
steps erase them and the clean side's 6 or 8 bytes give back the sent word: 10^6 of 10^6 read right in each, none
wrong (2026-09-27). Until 2026-09-27 the size check cut the 32 ring's rate to 6.0e-5, and the 48 ring's, counted at
4.6e-5, to 4.7e-6.

Registration should not correlate against word cells. The fps byte is the sender's choice, so a receiver cannot predict
those cells. The layout flags every word cell `CELL_WORD` (`liblizard/src/layout.h`) and the reference mesh skips them. When they
were in its template, one 720 px capture fell from 22 blocks of 24 to 6 with fps mis-stated by one (STATUS.md,
2026-09-21).

`liblizard/src/fmt.c:ob_fmt_decode` compiled natively (a scratch harness, 2026-09-27 evening), versions 1 to 128, given soft
values built from a codeword (magnitude 0.4; a byte in error 0.45, so no erasure lands on it, or 0.05 for one made least
confident; a faded side uniform within 0.02 of 0), version 2 and fps 0 unless a row says otherwise, 400 trials a cell:

| input | RS(8, 3) | RS(16, 3) | RS(24, 3) | RS(32, 3) |
|---|---|---|---|---|
| clean | read | read | read | read |
| floor(nroots / 2) byte errors (2, 6, 10, 14) | read | read | read | read |
| one byte error more | rejected | rejected | rejected | rejected |
| nroots - 2 byte errors, all in the least confident bytes | read | read | read | read |
| nroots - 1 byte errors, all in the least confident bytes | read | rejected | rejected | rejected |
| 1 byte error, and nroots - 3 in the least confident bytes | read | read | read | read |
| 1 byte error, and nroots - 2 in the least confident bytes | rejected | rejected | rejected | rejected |
| side 0 faded | read | read | read | read |
| sides 0 and 1 faded | read | read | read | read |
| sides 0 and 2 faded | read | read | read | read |
| sides 1, 2 and 3 faded | rejected | rejected (5 of 400 read) | read | read |
| sides 0, 1 and 2 faded | rejected | rejected (5 of 400 read) | read | read |
| valid codeword, magic `0x4D` | rejected | rejected | rejected | rejected |
| valid codeword, version 0 or 129 | rejected | rejected | rejected | rejected |
| valid codeword, version 128 | read | read | read | read |
| valid codeword, version 3 (off the ladder), fps 255 | read | read | read | read |

No trial read a wrong word. Three faded sides in the 64 ring are 12 bytes, one past the steps' reach of 11; the few
that read are those whose faded cells happened to fall on the right side of the threshold. In the 96 and 128 rings they
are 18 and 24 bytes, inside the steps' reach of 19 and 27, and read. Until 2026-09-27 RS(20, 3) and RS(36, 3) read
three faded sides at n = 1024 and 2048.

### 5.5 What the ring bootstraps (non-normative)

The rule since 2026-09-27: the ring tells what is inside it and bootstraps the receiver's configuration, and a frame
whose word later fails to decode leaves that configuration as it was. Both reference receivers do this (10.1, 10.2):

- **The ring is found first.** The finder registers the symbol against the four rings' layouts, a bounded search of 4
  like the 8 orientations, and the ring that registered is the frame's. The ring says nothing about the picture.
- **The word says what is inside.** It is read in that ring (5.4), and its version names the sub-channels and so n
  (3.2). The frame is finished at that picture, in that ring.
- **A failed word: the held configuration.** A receiver keeps the version of the last word it read, and decodes a
  frame whose word does not read at that configuration, in the ring the frame registered in. The held word is the one
  state a receiver carries from frame to frame (`liblizard/src/wasm.c:focus_any_rx` `held`; `liblizard/sim/phy.mjs:makeBlind`).
- **Nothing before the first word.** With no word held, a frame whose word does not read is not decoded.
- **No retry at the other rings.** When a word fails, the receiver does not register again at another ring
  (2026-09-27): by the time the word fails, the rest of the frame has long failed. On the recordings before the rings
  (the build before, told, 10,656 registered frames) the word read on 96.0% of frames, a block on 76.6%, and a block
  without the word on none. In the simulator, on the morning's three rings (32,
  48, 64), a frame registered at a wrong ring now and then: 90 of 2,048 under `scripts/exp/fmt_word.mjs` `DAMAGE=1`, 82 of them
  under glare over three sides, where nothing reads; 1 of 426 in `RINGS=all` (LIZARD-576 painted in the 48 ring at 45
  degrees: 2 blocks blind, 6 told); 1 or 2 frames in 9 cells of `scripts/exp/border_scale.mjs`, none with payload (2026-09-27).
  Not repeated on the four rings.

## 6. The picture

Normative, except the subsections marked non-normative. The definitions are in `liblizard/src/focus.c` (`init`, `ws_init`,
`focus_encode`, `turn_pack`, `clip_row`, `resample_init`, `resample`, `focus_paint_rgba`, `focus_n_for`) and
`liblizard/src/focus.h`; the transform conventions are in `liblizard/src/fft.h`.

The picture is an n x n grid of grey samples. Its 2D spectrum carries the frame's bits as QPSK symbols, two bits a
coefficient, lowest spatial frequency first. The transmitter fills the spectrum, takes a real inverse DFT, clips the
result and maps it to grey, then resamples it onto the symbol's pixel grid. The receiver samples the captured symbol on
the same n x n grid, in module coordinates, and takes the forward DFT.

### 6.1 Coordinates

- Picture sample (x, y) has 0 <= x, y < n. x grows to the right and y grows downward, as the symbol is painted: row 0
  is the top row (`focus_encode` stores the picture row-major, row y at offset y * n; the drive and
  `focus_paint_rgba` are row-major, top row first).
- u is the spatial frequency along x and v the frequency along y, in cycles across the picture.
- Module coordinates are those of 4.1: origin at the outer top-left corner of the symbol (the margin lies outside it),
  x to the right, y downward, one unit a module. "Top-left" is the symbol's logical corner, which the receiver learns
  from the border.
- The picture occupies modules [15, 15 + span) on each axis, span = 2B in ring B (3.3).

The directions matter. A transmitter that paints y upward, or swaps u and v, paints the mirror image of the symbol,
whose spectrum puts each coefficient's symbol on a different coefficient.

### 6.2 Picture size and scale

n follows from subch by the rule in 3.2, span from the ring (3.3), and pxm and the drive's size from both. The resampler
(6.8) and the receiver's grid (6.9) use these values:

| ring B | n | scale = n / span (samples a module) | pxm | upscale pxm / scale | resampled square: side, first pixel |
|---|---|---|---|---|---|
| 32 | 256 | 4 | 4 | 1 (copy) | 280, from 48 |
| 32 | 384 | 6 | 6 | 1 (copy) | 420, from 72 |
| 32 | 512 | 8 | 8 | 1 (copy) | 560, from 96 |
| 32 | 768 | 12 | 12 | 1 (copy) | 840, from 144 |
| 32 | 1024 | 16 | 16 | 1 (copy) | 1120, from 192 |
| 32 | 1536 | 24 | 24 | 1 (copy) | 1680, from 288 |
| 64 | 256 (default) | 2 | 2 | 1 (copy) | 268, from 24 |
| 64 | 384 (default) | 3 | 3 | 1 (copy) | 402, from 36 |
| 64 | 512 (default) | 4 | 4 | 1 (copy) | 536, from 48 |
| 64 | 768 (default) | 6 | 6 | 1 (copy) | 804, from 72 |
| 64 | 1024 (default) | 8 | 8 | 1 (copy) | 1072, from 96 |
| 64 | 1536 (default) | 12 | 12 | 1 (copy) | 1608, from 144 |
| 96 | 256 | 4/3 = 1.3333 | 2 | 1.5 | 396, from 24 |
| 96 | 384 | 2 | 2 | 1 (copy) | 396, from 24 |
| 96 | 512 | 8/3 = 2.6667 | 3 | 1.125 | 594, from 36 |
| 96 | 768 | 4 | 4 | 1 (copy) | 792, from 48 |
| 96 | 1024 | 16/3 = 5.3333 | 6 | 1.125 | 1188, from 72 |
| 96 | 1536 | 8 | 8 | 1 (copy) | 1584, from 96 |
| 128 | 256 | 1 | 1 | 1 (copy) | 262, from 12 |
| 128 | 384 | 3/2 = 1.5 | 2 | 4/3 = 1.3333 | 524, from 24 |
| 128 | 512 | 2 | 2 | 1 (copy) | 524, from 24 |
| 128 | 768 | 3 | 3 | 1 (copy) | 786, from 36 |
| 128 | 1024 | 4 | 4 | 1 (copy) | 1048, from 48 |
| 128 | 1536 | 6 | 6 | 1 (copy) | 1572, from 72 |

pxm = ceil(n / span) is the painter's whole number of pixels a module (`focus.c` `init` computes
`ceilf(scale - 1e-4)`, which is the same on every row; the `1e-4` keeps a scale that is a whole number at that number).
The square is (span + 2 cp) pxm pixels from (15 - cp) pxm, with cp = 3 the guard in modules (6.8): (span + 6) pxm from
12 pxm, read from the painted layout's cells on every row (2026-09-27 evening).

### 6.3 The coefficients

The candidate coefficients are the upper half-plane without DC, and without the Nyquist row and column, which are
their own reflections and cannot carry a free phase (`focus.c` `init`):

```
-n/2 < u < n/2,   0 <= v < n/2,   and not (v = 0 and u <= 0)
```

They are sorted by u^2 + v^2 ascending, then v ascending, then u ascending (`focus.c` `by_freq`). Call the sorted list
C_0, C_1, ... A frame of subch sub-channels uses C_0 to C_(320 subch - 1) and nothing else; every other coefficient of
the half-plane is zero.

**The order does not depend on n** (3.2 gives the argument). There is one order, and every format's coefficients are a
prefix of LIZARD-1024's. Its first 24 entries, as (u, v):

```
(1,0) (0,1) (-1,1) (1,1) (2,0) (0,2) (-2,1) (2,1) (-1,2) (1,2) (-2,2) (2,2)
(3,0) (0,3) (-3,1) (3,1) (-1,3) (1,3) (-3,2) (3,2) (-2,3) (2,3) (4,0) (0,4)
```

**Sub-channels.** Sub-channel j is C_(320 j) to C_(320 j + 319) (`FOCUS_SUB` = 320). Each holds 320 coefficients of the
half-plane, so sub-channel 0 is a half-disc of radius 14.2 and each later one a thin half-annulus about
640 / pi = 203.7 wide in u^2 + v^2. Higher j is higher spatial frequency. The last coefficient of some sub-channels:

| sub-channels | last coefficient (u, v) | u^2 + v^2 | radius |
|---|---|---|---|
| 1 | (9, 11) | 202 | 14.2 |
| 2 | (9, 18) | 405 | 20.1 |
| 8 (block 0) | (20, 35) | 1625 | 40.3 |
| 16 | (57, 3) | 3258 | 57.1 |
| 64 | (6, 114) | 13032 | 114.2 |
| 128 | (145, 71) | 26066 | 161.4 |
| 256 | (218, 68) | 52148 | 228.4 |
| 512 | (25, 322) | 104309 | 323.0 |
| 1024 | (26, 456) | 208612 | 456.7 |

**The disc.** A frame's coefficients fill a half-disc of radius R. The top coefficient's radius matches R to within 0.07
on every format of the ladder; the largest miss is LIZARD-48's, 0.068 outside R. Samples a cycle at the top ring,
n / r_top:

| format | n | R | top radius | n / r_top |
|---|---|---|---|---|
| LIZARD-16 | 256 | 57.09 | 57.08 | 4.49 |
| LIZARD-32 | 256 | 80.74 | 80.75 | 3.17 |
| LIZARD-48 | 384 | 98.89 | 98.95 | 3.88 |
| LIZARD-80 | 384 | 127.66 | 127.63 | 3.01 |
| LIZARD-96 | 512 | 139.85 | 139.85 | 3.66 |
| LIZARD-128 | 512 | 161.48 | 161.45 | 3.17 |
| LIZARD-144 | 768 | 171.28 | 171.27 | 4.48 |
| LIZARD-320 | 768 | 255.32 | 255.33 | 3.01 |
| LIZARD-336 | 1024 | 261.63 | 261.63 | 3.91 |
| LIZARD-560 | 1024 | 337.76 | 337.76 | 3.03 |
| LIZARD-576 | 1536 | 342.55 | 342.54 | 4.48 |
| LIZARD-1024 | 1536 | 456.74 | 456.74 | 3.36 |

The rows are the first and last format of each n. Before 2026-09-27 (3.2) the ratio ran to 5.98 (LIZARD-576 at 2048);
it now stays between 3.01 and 4.49.

Block 0 runs from the centre out to radius 37.8 from LIZARD-48 up (the 7/8 tier's 7 sub-channels) and 40.3 below (8
sub-channels of 3/4).

### 6.4 Blocks

Block b's rate and place come from the rate profile (3.1): it takes g_b sub-channels, 7, 8 or 12 by its rate, from
sub-channel s_b, the sum of the sub-channels of the blocks before it; coefficients C_(320 s_b) to C_(320 (s_b + g_b) -
1), 640 g_b slots. Its codeword (section 8) fills 4464, 5088 or 7680 of them and leaves the last 16, 32 or none to the
whitening (7.3). Under one rate (before 2026-10-07, and LIZARD-8 to -40 now) g_b = 8 and s_b = 8b. Other rate tiers and
`FOCUS_RS` build other frames, which are not the format (section 2).

### 6.5 Symbols

Frame slot 2s lies on the real axis of coefficient C_s and frame slot 2s + 1 on its imaginary axis, 0 <= s < 320 subch,
so a frame has 640 subch slots. Block b's slot k is frame slot 640 s_b + k (6.4). The slot bits x_0, x_1, ... are the frame's
whitened, permuted codeword bits (7.3). Coefficient s carries

```
H(C_s) = (a_j / sqrt 2) * ( (1 - 2 x_(2s)) + i (1 - 2 x_(2s+1)) ),      j = floor(s / 320),  i = sqrt(-1)
```

that is, slot bit 0 gives +, bit 1 gives -, on each axis independently (Gray-mapped QPSK) (`focus_encode`). The
reference computes the magnitude as `0.70710678f * a_j` in float32.

a_j is the amplitude of sub-channel j:

```
a_j = 10^( -t * j / (20 * (subch - 1)) )        t = tilt in dB; t = 0 gives a_j = 1 for every j
```

(`focus.c` `ws_init`). The tilt slopes power from the first sub-channel down to the last. It is a sender parameter
that no receiver needs, because the receiver measures each sub-channel's amplitude itself. The reference sender uses
t = 0 (equal power; `lizard-web/send.mjs` passes none, `liblizard/sim/phy.mjs` defaults to 0). Whether a conforming sender may use
another tilt is open (section 13).

### 6.6 The transform

The picture before clipping is the real inverse DFT of the half-plane and its conjugate mirror:

```
p(x, y) = sum over s < 320 subch of  2 Re[ H(C_s) exp( +2 pi i (u_s x + v_s y) / n ) ]
```

Equivalently: set H(-u, -v) = conj(H(u, v)) (indices mod n), zero everywhere else, and take the unscaled inverse 2D DFT
with kernel exp(+2 pi i (u x + v y) / n). No 1 / n^2 factor is applied (`fft.h`: inverse is exp(+i) and not scaled).
The reference evaluates it in float32 over the half-plane only (`focus_encode`, `turn_pack`, `fft.c` `fft_cols`),
built with `-ffp-contract=off`, from one twiddle table W^j = exp(-2 pi i j / n), j < n / 2, computed in double and
stored as float32 (conjugated for the inverse). For n a power of two the FFT is radix 2. For n = 3 x 2^k (384, 768,
1536) it is one radix-3 stage, the shortest, then k radix-2 stages, on the input permuted to digit-reversed order in
those radices (`fft.c` `stage_list`, `plan_for`, `rows3`, `fft_cols_mixed`). With b and c already multiplied by their
twiddles, the radix-3 butterfly is

```
t1 = b + c,   t2 = b - c,   h = a - 0.5 t1
a' = a + t1,  b' = h - i s k t2,  c' = h + i s k t2
k = 0.8660254 (sqrt 3 / 2 in float32),  s = +1 forward, -1 inverse
```

Both are implementations of the formula above, not a different definition; the reference's bytes are what its code
computes (section 2; 13, item 9). Every power of two stays on the radix-2 path, so the 3 x 2^k sizes moved no byte of a
format whose n did not change (2026-09-27).

**Normalisation.** Let P = sum over s of |H(C_s)|^2 = 320 * sum over j of a_j^2. The mirror doubles it, so by Parseval
the mean of p^2 over the n^2 samples is 2P and the picture's rms is

```
sigma = sqrt(2 P)          at t = 0:  sigma = sqrt(640 subch)
```

The reference never measures the rms; it uses this value. At LIZARD-16 sigma = sqrt(10240) = 101.19.

### 6.7 Clipping and grey

```
L = c * sigma,     c = 2 (the clipping ratio)
g(x, y) = 1/2 + clamp(p(x, y), -L, L) / (2 L)
```

g is in [0, 1], 0 black and 1 white. The reference computes the level in float32 as `lim = 2 * c * sqrtf(0.5 * 320
subch)` at t = 0, or `2 * c * sqrt(0.5 * P)` with P summed in double when t > 0, and g as `0.5f + clamp(p) * (0.5f /
lim)` (`focus_encode`, `clip_row`); both equal c * sigma. Since p is close to Gaussian, about 4.5% of the samples are
clipped.

c = 2 is the value every sender in the tree uses (`focus.c` `init` default, `liblizard/sim/phy.mjs` default). A receiver does not
depend on it, for the same reason as the tilt. Whether a conforming sender may use another c is open (section 13).

### 6.8 Resampling onto the painted grid

The drive is S = span + 30 = 2B + 30 modules a side at pxm pixels a module, so S x pxm pixels square (3.3). Drive pixel (X, Y)
covers module coordinates [X / pxm, (X + 1) / pxm) and its centre is at (X + 0.5) / pxm.

**The resampled square.** Modules [12, span + 18) on both axes are the resampler's alone: the picture's span modules and
the 3-module guard on every side of it (`layout.h` `FOCUS_CP` = 3; `focus_encode` takes cp = min(FOCUS_CP, margin -
OB_THIN), where the C's `margin` is the 15-module border depth and `OB_THIN` is 7, so cp = 3 in the format). In pixels
that is a square of q = (span + 6) * pxm pixels whose top-left pixel is (12 pxm, 12 pxm) (6.2). Nothing else is painted
there: the outer border's modules outside it are painted from the layout (section 4). `focus_encode` stamps a corner
mark back over the square only when the mark is wider than the border, which the format's 12-module mark in a 15-module
border never is, and the format has no centre or edge marks.

**Where each pixel reads the picture.** Along each axis, the pixel with local index t = X - 12 pxm (0 <= t < q) reads
the picture at the continuous sample coordinate

```
w(t) = (t + 0.5) * sc / pxm - 3 * sc - 0.5,        sc = n / span,   evaluated in double
```

which is w = ((X + 0.5) / pxm - 15) * sc - 0.5. Its inverse is the rule that places the picture: **sample x sits at
module coordinate 15 + (x + 0.5) / sc**, the same on both axes. Where sc is a whole number it equals pxm, each pixel
lands on one sample and the resampler copies: every picture in the 32 and 64 rings, n = 384, 768 and 1536 in the 96
ring (sc = 2, 4, 8) and every n but 384 in the 128 ring (sc = 1, 2, 3, 4, 6). At n = 256, 512 and 1024 in the 96 ring
(sc = 1.33, 2.67, 5.33) and at n = 384 in the 128 ring (sc = 1.5) sc is not a whole number, which is why those four are
resampled rather than copied. Either way there is no light between the border and the picture.

**The kernel.** Lanczos-3:

```
K3(d) = 3 sin(pi d) sin(pi d / 3) / (pi d)^2     for 0 < |d| < 3,     K3(d) = 0 for |d| >= 3
```

For each t (`resample_init`):

- If |w - r| < 1e-6, with r = floor(w + 0.5), the pixel lands on a sample: one tap, weight 1, on sample r. In the
  six-tap form of the passes below that is i0 = r - 2 and weights (0, 0, 1, 0, 0, 0).
- Otherwise the taps are the six samples i0 + k, k = 0 to 5, with i0 = floor(w) - 2, and their weights are
  K3(w - (i0 + k)) divided by the sum of the six, computed in double and stored as float32.

Sample indices are taken mod n: the picture is read as **periodic**. The same weights serve both axes.

In the pairs that copy, every pixel lands on a sample (w = x exactly, one tap), so the resampler is an exact copy. In
the four that resample no pixel does: the nearest miss is 0.167 samples at n = 256 and 0.056 at 512 and 1024 in the 96
ring, and 0.125 at 384 in the 128 ring (w in double over every pixel of the square, as `resample_init` computes it,
2026-09-27 evening). The morning's 48 ring resampled 256, 512 and 1024.

**The passes.** Vertical first, then horizontal, each accumulated in float32 in tap order. Only the horizontal result
is clamped; T is not:

```
T(ty, x) = sum over l = 0..5 of  wy_l * g(x, (i0(ty) + l) mod n)             for every picture column x
D(tx, ty) = clamp( sum over k = 0..5 of  wx_k * T(ty, (i0(tx) + k) mod n),  0, 1 )
```

D is the drive at pixel (12 pxm + tx, 12 pxm + ty). Lanczos overshoot past [0, 1] at clipped plateaus is what the clamp
removes: about 3% of the square's pixels, measured on the n / 8 + 60 border, where every picture was upscaled. Clamping T
as well paints a different picture: on LIZARD-256 with a zero payload, then upscaled by 1.08, it changed 37,690 of the
square's 1,317,904 grey bytes, 10,923 of them by more than one level (at most 13). A copy has no overshoot, so both
clamps matter only for the four pairs that resample: 256, 512 and 1024 in the 96 ring and 384 in the 128 ring.

**The guard** is not a separate step. Because the picture is read as periodic, the 3 modules round it are the
picture's own far side, wrapped round its edges as a torus, corners included (4.4). It makes the camera's blur act on the
picture as a circular convolution, which is the decoder's model (a per-coefficient gain). A pixel at horizontal index t
and the pixel at t + span * pxm read samples exactly n apart, so the two carry the same value.

**Painting.** Each drive value becomes one 8-bit grey level, round half up:

```
G = floor(D * 255 + 0.5), clamped to [0, 255];     R = G = B = G, alpha 255
```

(`focus_paint_rgba`, which evaluates it as float32 `D * 255.0f + 0.5f` truncated). The outer border's drive values are
exactly 0 or 1, so it paints only 0 and 255. The picture and its guard use all 256 levels.

**How exact.** An independent implementation of 6.5 to 6.8 in double precision (the Hermitian spectrum built from the
reference's slot bits, a double-precision inverse FFT times n^2, the clip, and the taps and passes as written) was
held against the reference's drive and grey bytes on seven frames, each with every payload byte zero and fps 0:
LIZARD-16 (P2 in 6.12), LIZARD-16 with t = 6 dB, LIZARD-48, LIZARD-64 (P3), LIZARD-144, LIZARD-256 and LIZARD-576.
The drive matched to within 4.2e-7 at every pixel of the square (the largest, 4.17e-7, at LIZARD-576). The grey bytes
were identical on both LIZARD-16 frames and differed by one level, never more, at 4 of 360,000 pixels (LIZARD-48), 2
of 360,000 (LIZARD-64), 11 of 1,317,904 (LIZARD-144), 18 of 1,317,904 (LIZARD-256) and 55 of 5,456,896 (LIZARD-576).
The residue is float32 rounding in the reference's FFT. That comparison was made on the n / 8 + 60 border (2026-09-24)
and not repeated on the rings. To reproduce, dump the drive, `_focus_dbg_sym` and
`_focus_pos_ptr` as in 6.12 and decode the positions as in P1. A conforming transmitter follows the formula; it need
not match the reference's bytes in the picture, only in the border (2026-09-29, revising 2026-09-24's rule that
it match them all; section 2, Conformance; section 13, item 9).

### 6.9 Where the receiver samples

Picture sample (x, y) is at module coordinates

```
( 15 + (x + 0.5) * span / n,   15 + (y + 0.5) * span / n )
```

so the grid starts at 15 + 0.5 / scale and steps by 1 / scale modules. In the default ring, the 64 (`_focus_grid_out`,
float32, 2026-09-27 evening): 15.25 and 0.5 at n = 256, 15.166667 and 0.3333333 (15 + 1/6 and 1/3, rounded) at 384,
15.125 and 0.25 at 512, 15.083333 and 0.1666667 at 768, 15.0625 and 0.125 at 1024, and 15.041667 and 0.0833333 (15 +
1/24 and 1/12) at 1536. In the 96 ring: 15.375 and 0.75 at 256, 15.25 and 0.5 at 384, 15.1875 and 0.375 at 512, 15.125
and 0.25 at 768, 15.09375 and 0.1875 at 1024, 15.0625 and 0.125 at 1536. Any other pair follows from 6.2's scale
(`focus_finish_bits`: `g0 = margin + 0.5f / scale` with the C's `margin` = 15, `step = 1.0f / scale`; `wasm.c`
`focus_grid_out`). A receiver that takes the forward DFT of an n x n grid MUST take it at these points: a grid displaced
by (dx, dy) samples turns coefficient (u, v) by 2 pi (u dx + v dy) / n.

The receiver's transform is the unscaled forward DFT

```
Y(u, v) = sum over x, y of s(x, y) exp( -2 pi i (u x + v y) / n )
```

(`fft.h`: forward is exp(-i), not scaled). Its scale does not matter: the reference receiver estimates each
sub-channel's amplitude and noise from the second and fourth moments of |Y| over its 320 coefficients (8.5), so neither
the clip level nor the tilt nor the capture's gain is signalled.

**Reading a slot.** For an unclipped picture, g = 1/2 + p / (2L), the forward DFT at a used coefficient is
Y(C_s) = n^2 H(C_s) / (2L), with (u, v) taken mod n as an index; the constant 1/2 lands on DC only. Clipping and the
channel scale this, to first order, by a positive gain for each sub-channel and add noise. So a receiver MUST read
Re Y(C_s) > 0 as bit 0 of frame slot 2s and Re Y(C_s) < 0 as bit 1, and Im Y(C_s) the same way for slot 2s + 1 (6.5:
bit 0 is sent as +). The reference's soft values follow it: `focus.c:focus_finish_bits` sets each LLR to a non-negative
multiple of the component, and positive means bit 0 (8.5, `ldpc.h`). The signs of the forward DFT of the clipped
picture g itself, with no resampling, give back every slot bit of all seven frames in 6.8's "How exact" (0 of 368,640
at LIZARD-576).

The reference receiver reaches these points through its registration (the homography and the border mesh, 10.1) and
reads the camera image bilinearly, pixel centres at integer + 0.5, through the lookup (i / 255)^gamma (`acquire.c`
`ob_sample_grid`, `internal.h` `sample`, `acquire.c` `ob_image_init`).

Reading the reference drive itself bilinearly at these points (module coordinate times pxm, pixel centres at
integer + 0.5), with no camera, and taking the signs of Y at the frame's coefficients recovered every slot bit: 0 of
10,240 wrong at LIZARD-16 and 0 of 30,720 at LIZARD-48 (the n / 8 + 60 border, 2026-09-24; not repeated on the
rings).

### 6.10 What the receiver removes before reading (non-normative)

A screen is never evenly lit. A brightness ramp across the picture is not periodic, so its spectrum falls off as 1 / k
along both axes and lands on the coefficients nearest zero frequency, which carry block 0. The reference receiver
removes the least-squares surface in 1, x, y, xy, x^2, y^2 (`focus_finish_bits`, basis built in `ws_init`).

With c_i = (2 i + 1 - n) / n and q_i = c_i^2 - mean(c^2) for i = 0 to n - 1, the five non-constant functions c(x),
c(y), c(x) c(y), q(x), q(y) are orthogonal on the grid and to the constant, so each coefficient is one projection of
the samples s:

```
kx  = sum s(x,y) c_x / (n Sc)        ky  = sum s(x,y) c_y / (n Sc)        kxy = sum s(x,y) c_x c_y / Sc^2
kxx = sum s(x,y) q_x / (n Sq)        kyy = sum s(x,y) q_y / (n Sq)
Sc  = sum c_i^2 = (n^2 - 1) / (3 n)  Sq  = sum q_i^2
```

The surface is separable, so it comes out in the spectrum. With X1(k) = sum c_i exp(-2 pi i k i / n) and X2(k) the same
for q:

```
Y(u, v) -= kxy * X1(u) * X1(v)                         every (u, v)
Y(u, 0) -= n * (kx * X1(u) + kxx * X2(u))              the row v = 0
Y(0, v) -= n * (ky * X1(v) + kyy * X2(v))              the column u = 0
```

The constant term is DC, which carries nothing, and X1(0) = X2(0) = 0, so DC is untouched. The sender does nothing to
make room for this; whatever part of the picture itself projects onto the five functions is removed with the ramp.

### 6.11 Grey levels, gamma and palette

- **Levels.** The reference paints 256 levels. In the simulator 16 levels carried within 1% of the payload of 256, and 4
  levels lost up to 15% (`scripts/exp/focus_media.mjs`, `research/07-focus-future-work.md`), so nothing in the format needs a
  fine grey scale.
- **Gamma.** The sender writes the levels as they are, with no gamma pre-compensation and no linearisation; the panel's
  transfer curve and the camera's tone curve are part of the channel. The format assumes nothing about either. The
  reference receiver is told gamma 1 for a grey picture (`lizard-web/send.mjs` spec, `variants`) and uses the identity
  lookup. Measured in the simulator with panel gamma drawn from 1.8 to 2.6 and camera tone curve from 1.8 to 2.4 a
  frame, no linearisation beat gamma 2.2 by 7 to 15%, because the two curves roughly cancel (`research/04-wave1.md`,
  wave 1).
- **Luminance only** (section 2, item 1). The data stays in luminance; two brand colours, where used, only move where
  that luminance sits. The reference painter in `liblizard/src/` paints neutral grey only.
- **Dithered black and white** was deleted on 2026-09-26 (section 12): a Lizard picture is grey levels.

### 6.12 Test vectors

Computed with `liblizard/build/ob.wasm` through `liblizard/sim/ob.mjs` (`init`, then the raw exports): `_focus_setup(n, subch, 1, 2, 128, 0,
0, 0, 0, 0, 0, 0)` (LDPC mode, clip 2, span 128: the 64 ring, named; it was the default ring when these were made, and
the default has been the 128 since 2026-10-01; no tilt, default corner mark), `_focus_dbg(1)`,
`_focus_tx` on a payload of blocks x 473 zero bytes, then the drive (float32, S x pxm pixels square, row-major, top row
first), the slot bits from `_focus_dbg_sym` (int8 a slot, +1 for bit 0, -1 for bit 1, frame slot order) and the
coefficient positions from `_focus_pos_ptr`. Grey bytes are G as defined under Painting (6.8). The format word states
fps 0, and the pilot count is 0 (`_focus_parity_set` not called): since 2026-09-30 the tail slots, and so the slot
bits and the drive, change with the count (7.3, 9.1). The slot bits and 7.4 do not depend on the ring or on n. The drive and grey values below are the build of the
evening of 2026-09-27: the four rings with the 64 the default for every picture, the picture sizes of 3.2 and the
border of section 4 (12-module corner marks and a 3-module guard, so the resampler paints modules [12, span + 18)).
The values of the n / 8 + 60 border, and those of that morning's default rings, are gone.

**P1, the order.** The first 24 coefficients and the sub-channel ends in 6.3, from `_focus_pos_ptr` at every format of
the ladder. `_focus_pos_ptr` holds 320 subch int32, entry i for C_i, each an offset into the reference's strip layout
(`focus.c:init`). With `bw = min(n / 2, 64)` it is

```
p = floor(v / bw) * n * bw + ((u + n) mod n) * bw + (v mod bw)
```

and it decodes as `v = floor(p / (n * bw)) * bw + (p mod bw)`, `u' = floor(p / bw) mod n`, and `u = u'` when
`u' < n / 2`, otherwise `u' - n`. Decoded this way, the entries equal the sort of 6.3 at every coefficient of
LIZARD-16, -48, -64, -144, -256 and -576, each at its n of 3.2 (2026-09-27). The offsets depend on n and the (u, v)
entries do not: all 44 formats that moved n on 2026-09-27 decode to the same (u, v) list at the new n as at the old.

**P2, LIZARD-16 (n = 256, 2 blocks) in the 64 ring (span 128), every payload byte zero.**

- slot bits, one byte (0 or 1) a slot, 10,240 bytes: SHA-256
  `726c1e022ebb8c61cd6ded139be3a029b0102d39f81fdceeb08c70410edea142`
- the same packed 8 to a byte, first slot in the most significant bit, first 16 bytes:
  `00203e010efc0be0f8bfbdce000c0c03`
- sigma = 101.192885, L = 202.385770
- drive 316 x 316 (S = 158, pxm = 2; scale = pxm, so the picture is copied one pixel a sample). Drive values at row 30,
  columns 30 to 33 (the picture's first pixel row and column): 0.7166191, 1.0000000, 1.0000000, 1.0000000
- grey bytes at row 30, columns 30 to 45: `183 255 255 255 255 177 121 120 138 131 96 57 41 61 104 135`
- grey bytes at row 230, columns 230 to 245: `255 255 119 22 9 42 74 89 93 104 132 176 211 213 182 148`
- grey bytes of the resampled square, pixels [24, 292) on both axes (modules [12, 146)), row by row, 71,824 bytes:
  SHA-256 `4b0fece66fb3a1963ae8d322893800afedcd2164914f6c519b75fc2033b76116`; 1619 of them are 0 and 1698 are 255
- grey bytes of the whole drive, border included, 99,856 bytes: SHA-256
  `49f87e1569ffd094642136cb1025ab20e038f23b8d068a0f2ce774e851ff377a`

**P3, LIZARD-64 (n = 384, 8 blocks) in the 64 ring (span 128), every payload byte zero.** The first 16 packed slot bytes
are those of P2, since block 0's payload, codeword and whitening are the same.

- slot bits, one byte a slot, 40,960 bytes: SHA-256 `530e72f118c1c3f3b90b6017548c13039b3c9415922ff234a6a10853af28250e`
- drive 474 x 474 (S = 158, pxm = 3; scale = 3, so the picture is copied one pixel a sample, as in P2); the picture
  starts at pixel 45
- grey bytes at row 45, columns 45 to 60: `94 159 255 255 196 145 208 229 141 71 91 129 132 127 138 156`
- grey bytes of the resampled square, pixels [36, 438) (modules [12, 146)), 161,604 bytes: SHA-256
  `3da487dd12bb2fd61466373bb6377ff80a1afa2deae35d4ddc21dc36db074a1a`
- grey bytes of the whole drive, 474 x 474, 224,676 bytes: SHA-256
  `ec6c4e4cd91acb3aaaac44e9ab13ce99078a25aed4784d7baa60daaf0ad0593f`. The border holds the RS(16, 3) word (5.2)

P2 and P3 moved to the 64 ring when it became the default for every picture (2026-09-27 evening); that morning they were
in the 32 and 48 rings, and P3 was at n = 512 before the picture sizes of 3.2. Both are copies, so neither goes through
the Lanczos taps: in the 64 ring every picture is a copy, and only four (ring, picture) pairs resample (6.8), 384 in
the 128 ring, the default since 2026-10-01, among them.

`test/vectors.mjs <ob.mjs>`, run from `liblizard/`, checks 5.2 (the seven words read back out of the painted
cells), P2, P3, 7.4 and 9.3 against a build (all ok on the evening build of 2026-09-27 and on that of 2026-10-04, 20 s)
and prints three digests over the whole ladder to diff two builds: every format's cell map at five rates, its drive on
a fixed payload, and `_focus_pos_ptr`, each format in the 64 ring (span 128). "Every format" there is the 64 formats of
before 2026-10-01, LIZARD-16 to -1024 in steps of 16 (two blocks), not every whole number of blocks. They are cells `e8ae1656c90e4552`, drives `3f8e49cfe13a5596`, order
`60842c81479e470a`. With the 64 ring the default, the cells and drives of LIZARD-16 to -128 moved (that morning they
were `fcd70aefcce26613` and `f0b6d16dbac39887`) and the order did not. When the picture sizes changed the cells did not
move and the drives and the order did, by design: a drive is painted from an n x n picture, and the order digest hashes
the offsets of P1, which depend on n.

The double-precision implementation of 6.8 was held against P2 and P3 on the n / 8 + 60 border (2026-09-24) and has
not been run on these.

## 7. Blocks, the bit mapping and the transfer

Normative, except where a paragraph says otherwise. Sources: `liblizard/src/focus.c` `init`, `focus_encode`, `whiten_fill`,
`perm_fill`, `map_out`, `map_in`, `focus_finish_bits`; `liblizard/src/focus.h` `FOCUS_BITMAP`; `liblizard/src/layout.c` `ob_crc32`. For the
transfer (7.5 to 7.10): `liblizard/src/xfer.h`, `liblizard/src/xfer.c`, and their wasm exports in `liblizard/src/wasm.c`.

A frame of version V carries B(V) blocks (3.1). Each block is one LDPC codeword with its own CRC-32. Blocks are independent, so a
frame delivers any subset of them. Block 0 sits at the lowest spatial frequencies and the blocks run outwards, so blur,
distance and resampling take the last blocks first.

In this section and in section 8, n, k and m are the block's code's length, information length and parity count (5088,
3816 and 1272 at 3/4; 4464, 3906 and 558 at 7/8; 7680, 3840 and 3840 at 1/2), not the picture size.

### 7.1 The block

A block is 473 bytes at every rate. The number comes from the code: `focus.c:init` sets block_bytes = k/8 - 4 for the
smallest k of the format's codes, the 3/4 code's (every format has a 3/4 tier), 3816/8 - 4, and the 4 bytes held back
are the CRC.

| bytes | content |
|---|---|
| 0 to 3 | block id, uint32, little-endian (7.5) |
| 4 to 472 | 469 payload bytes: one Wirehair block, the header block or a manifest block |

The picture codec (`liblizard/src/focus.c`) treats the 473 bytes as opaque; the id's layout is the transfer's (7.5, `liblizard/src/xfer.h`,
2026-09-24). `liblizard/sim/phy.mjs:makeFocus` writes the id in `frame` and reads it in `tally`. A frame's payload is
B(V) x 469 bytes.

**Information word**, k bits (`focus.c:focus_encode`): 3816 at 3/4, 3906 at 7/8, 3840 at 1/2.

| bits | content |
|---|---|
| 0 to 3783 | block bytes 0 to 472. Bit i is bit 7 - (i mod 8) of byte floor(i / 8): most significant bit first. |
| 3784 to 3815 | CRC-32 of bytes 0 to 472. Bit 3784 + i is bit 31 - i of the CRC: most significant bit first. |
| 3816 to k - 1 | zeros: 90 bits at 7/8, 24 at 1/2, none at 3/4. |

At 3/4 the word fills k exactly. At 7/8 and 1/2 the zeros past the CRC are known to the decoder (the reference feeds
them as certain zeros, `focus.c:focus_finish_bits`), so the roomier code spends them on protection.

**CRC-32** (`layout.c:ob_crc32`) is the CRC of IEEE 802.3 and zlib: polynomial 0x04C11DB7 processed reflected
(0xEDB88320, each byte least significant bit first), register initialised to 0xFFFFFFFF, result complemented. The CRC-32
of the ASCII string `123456789` is 0xCBF43926. It covers the id and the payload.

The two multi-byte fields have opposite byte orders. The id is little-endian. The CRC, read as bytes 473 to 476 of the
477-byte word, is big-endian (gzip stores the same CRC little-endian).

**Acceptance.** A receiver MUST hand a block to the fountain only when the CRC-32 it computes over the 473 decoded bytes
equals the 32 decoded CRC bits. LDPC convergence alone is not acceptance. The reference requires both, a zero syndrome
and then the CRC (`focus.c:focus_finish_bits`).

### 7.2 Where a block lies in the frame

6.3 orders the frame's coefficients by rising frequency, 320 to a sub-channel. Block b (b = 0 to B(V) - 1), of g_b
sub-channels from sub-channel s_b (6.4), owns:

- sub-channels s_b to s_b + g_b - 1;
- coefficients 320 s_b to 320 (s_b + g_b) - 1 in that order;
- 640 g_b slots, two a coefficient. Slot i of the block is the real part of coefficient 320 s_b + floor(i / 2) when i
  is even, and its imaginary part when i is odd. It lies in sub-channel s_b + floor(i / 640), and its place in the
  frame is slot 640 s_b + i.

Under one rate (g_b = 8, s_b = 8b) these are sub-channels 8b to 8b + 7, coefficients 2560b to 2560b + 2559 and frame
slots 5120b + i.

A slot bit 0 is sent as a positive component and a 1 as a negative one (`focus.c:focus_encode`). 6.5 gives the
magnitude.

### 7.3 The bit map

A codeword has n bits and its block 640 g_b slots: 4464 of 4480 at 7/8, 5088 of 5120 at 3/4, 7680 of 7680 at 1/2. The
bit map puts the codeword on the slots and whitens it. The format's map is LINEAR (`focus.h:FOCUS_BITMAP`). For block b,
its codeword c (section 8), the whitening sequence w and the picture's parity q (9.1):

```
slot[i] = (i < n ? c[(st * i) mod n] : q) XOR w[640 s_b + i],    i = 0 .. 640 g_b - 1
```

with st 1709 at 7/8, 1943 at 3/4 and 2933 at 1/2 (below). At 3/4, the one rate before 2026-10-07:

```
slot[i] = (i < 5088 ? c[(1943 * i) mod 5088] : q) XOR w[640 s_b + i],    i = 0 .. 5119
```

**The permutation** (`focus.c:perm_fill`). Slot i carries codeword bit perm(i) = i * st mod n. The stride st is
floor(0.3819660113 n + 0.5), then raised by 1 until gcd(st, n) = 1 (0.3819660113 is 1/phi^2, phi the golden ratio). For
n = 5088 the first value, 1943, is already prime to n (1943 = 29 x 67, 5088 = 2^5 x 3 x 53), so st = 1943. Inverse:
codeword bit t lies on slot 1799 t mod 5088, since 1943 x 1799 = 1 mod 5088. perm(0) to perm(11) are 0, 1943, 3886, 741,
2684, 4627, 1482, 3425, 280, 2223, 4166, 1021. For n = 4464 (7/8) the first value, 1705, shares 31 with n, and 1706,
1707 and 1708 share 2, 3 and 4, so st = 1709; codeword bit t lies on slot 3077 t mod 4464, and perm(0) to perm(11) are
0, 1709, 3418, 663, 2372, 4081, 1326, 3035, 280, 1989, 3698, 943. For n = 7680 (1/2) the first value, 2933, is prime
to n, so st = 2933; codeword bit t lies on slot 1757 t mod 7680, and perm(0) to perm(11) are 0, 2933, 5866, 1119, 4052,
6985, 2238, 5171, 424, 3357, 6290, 1543. Every run of the codeword (a circulant's z bits, the parity chain) is spread
across the block's coefficients.

**The pilots** (the block's slots past its codeword, the top of its last sub-channel: 5088 to 5119 at 3/4,
coefficients 2544 to 2559 of the block; 4464 to 4479 at 7/8, coefficients 2232 to 2239; none at 1/2, whose codeword
fills its block; 2026-09-30; signed 2026-10-01). The slots past the codeword carry no code bit: the whitening XOR a bit
of the picture's count c (9.1), bit 0 of it on an even block (b even, b the block's place in the frame) and bit 1 on an
odd one. So an even block's tail flips sign picture to picture, and an odd block's every second picture; where the bit
is 1 the tail's coefficients are painted inverted. They
cost no capacity (the codeword never used them) and no power (the same QPSK points, turned by 180 degrees). The LDPC
ignores them. What a receiver reads from them, per block b with a tail, from the coefficients after its transform and
detrend: the T axis values y_i of the tail (T = 32 at 3/4, 16 at 7/8; slot n + 2k the real part of coefficient n / 2 +
k, n + 2k + 1 its imaginary), against the known signs, in units of the rms axis value of the tail's sub-channel (m2
its mean |y|^2 over the 320 coefficients; a 1/2 block has no tail and reads none):

```
r_b = (1 / T) sum_i (1 - 2 w[640 s_b + n + i]) y_i / sqrt(m2 / 2)
```

and a frame has two readings: r, the mean of r_b over the even blocks its format carries, and r2, the mean over the
odd ones (every format has both: LIZARD-16 one block of each). A capture of one picture reads r = +1 where bit 0 of its
count is 0 and -1 where it is 1, and r2 the same by bit 1 (sqrt(A2 / m2) of that at a signal-to-noise ratio A2 / N,
0.98 to 0.99 at the decoders' usual ones). So the two signs are the count of the picture a capture mostly holds: a
capture whose count is the one before's saw a picture the screen showed twice, and one whose count is two on saw a
picture skipped.

A capture that also holds a share f of pictures whose bit differs reads, on that bit's blocks, 1 - 2 f where each
sample came from one picture or another (a tear), and (1 - 2 f) / sqrt(1 - 2 f + 2 f^2) where every sample is the same
blend (the blend lowers m2, the pictures' data being independent; a blend of pictures that share the bit reads over 1
for the same reason); either way 0 at an even mix and largest in size unmixed. Bit 0 differs between any two
consecutive pictures, bit 1 only across every second change. So for a capture of picture c holding a share a of the
picture before and b of the one after, in size: r is 1 - 2 a - 2 b, and r2 is 1 - 2 a where c is even and 1 - 2 b where
c is odd. The readings say which neighbour a capture holds, not only how much of one: what a receiver steering its
camera's phase needs (10).

A reading's standard error is its blocks' spread of r_b over the square root of their count (0.003 to 0.007 on a clean
capture at LIZARD-32 to -512, where a reading has half the blocks; 1/sqrt(M) sqrt(N / A2) in general, M its pilots' real
symbols; LIZARD-16's readings are one block each and have no spread to state). Both reference decoders report the two
(`focus.h:focus_pilot`; the GPU's soft stage, `liblizard/gpu/wgsl/back_soft.mjs` `PILOT`, a value a block, averaged by its host)
and the Android receiver steers its camera's phase by them (10). A sender that paints c = 0 throughout (every sender
before 2026-09-30) reads +1 on both, mixed or not: the receiver can tell, since consecutive pictures then never read
opposite signs of r. Until 2026-10-01 every block carried bit 0 (the count mod 2), which says how much of a neighbour a
capture holds and not which.

**The whitening** (`focus.c:whiten_fill`) is PRBS-23 on x^23 + x^18 + 1, started from all ones, not inverted:

```
w[i] = w[i - 23] XOR w[i - 18],    w[j] = 1 for j < 0
```

The C does the same with a 23-bit register: s = 0x7FFFFF, and for each i, w[i] = ((s >> 22) XOR (s >> 17)) AND 1, then
s = ((s << 1) OR w[i]) AND 0x7FFFFF. The sequence starts with 18 zeros and then 5 ones. Its first 128 bits are
`00003e000ffc03e0f8ffffce000c1c03` (hex, the first bit of each digit most significant).

w is indexed by the slot's place in the frame, 640 s_b + i, and restarts at slot 0 in every frame. It depends on
nothing else: not the payload, the frame number, the format or n. So at one picture size n, the same block bytes in the
same code at the same place paint the same slots in any format: a smaller format's slots are a prefix of a larger
format's as far as their tiers agree (LIZARD-256 and LIZARD-512 share their first 12 blocks, the 7/8 ones, slot for
slot; under one rate the whole of the smaller frame), and the same 473 bytes paint differently in different block
positions. The period, 2^23 - 1, is longer than the
largest frame (LIZARD-1024: 655,360 slots).

**Why** (non-normative; STATUS.md, "The bit map", 2026-09-23). Unwhitened, the payload's own bits were the picture's
phases. A block of runs painted thousands of coefficients at one phase into a peak that the clip flattened: LIZARD-64
with every block 100 bytes then zeros decoded 6 of 64 blocks on a clean capture, and 64 of 64 whitened. Without the
permutation, the systematic codeword's 1272 parity bits (degree 2, its weakest) sat on the block's outermost quarter,
the slots blur and distance take first. `scripts/exp/bitmap_race.mjs` chose LINEAR over INNER, which tied with it, because
LINEAR spreads every run of the code instead of betting on which way the damage runs.

**One map** (2026-09-24). LINEAR is the only bit map the format has, so the format word names none. The other
values of `focus.h:FOCUS_BITMAP_*` (NONE, WHITE, SPREAD, INNER) are lab settings for `scripts/exp/bitmap_race.mjs`, not the format.

### 7.4 Test vector

LIZARD-16 (n = 256, 2 blocks, one tier of 3/4 under the rate profile, so these are the 3/4 code's vectors). Block b
is the id b as uint32 little-endian, then 469 payload bytes, byte j = j mod 256:

```
block 0: 00 00 00 00 | 00 01 02 ... fe ff 00 01 ... d4
block 1: 01 00 00 00 | 00 01 02 ... fe ff 00 01 ... d4
```

Bit strings are hex, four bits a digit, the first bit the most significant.

| | block 0 | block 1 |
|---|---|---|
| CRC-32 of bytes 0 to 472 | 0x9F59BCE3 | 0x90C7A3FE |
| codeword bits 3776 to 3815 (last payload byte, CRC) | `d49f59bce3` | `d490c7a3fe` |
| codeword bits 3816 to 3943 (parity p_0 to p_127) | `689b3806ae2ff8272a67778bebce6096` | `689b47f951d007d8d598887414319f96` |
| whitening under slots 0 to 127 (frame slots 5120b to 5120b + 127) | `00003e000ffc03e0f8ffffce000c1c03` | `79923e97abd8c5a180e323390e8831bf` |
| slots 0 to 127 | `2a6d2d409722af16f5f2c5ec83cda6fb` | `77df2df63b06611785ae191b9d491b47` |
| slots 5088 to 5119 (the pilots), c = 0 | `a909ab10` | `8afe77aa` |
| slots 5088 to 5119 (the pilots), c = 1 | `56f654ef` | `8afe77aa` |
| slots 5088 to 5119 (the pilots), c = 2 | `a909ab10` | `75018855` |
| slots 5088 to 5119 (the pilots), c = 3 | `56f654ef` | `75018855` |

So block 0's coefficient 0 is sent as (+, +) and its coefficients 1 to 3 as (-, +).

SHA-256 of block 0's whole codeword, 5088 bits packed most significant bit first (636 bytes):
`01c4ddf4d95d858e261b9bdc6a02ec9080268cb73bcfc9ce95c613631097d167`. SHA-256 of the frame's 10,240 slots packed the same
way (1280 bytes), c = 0: `8881113e5edfc7b2c3ac85c58d00b57d4186021e9c3a66a045715263e95cbd1f`. At c = 1, 2 and 3
(`f.setParity(c)` before the encode below; only the pilot slots of the blocks whose bit is 1 differ):
`1d78ab5959cafac00fa21f962887172051f365483cf3dd757a0d48316700e3d3`,
`23b8f121af7afe4541b1b8b3c41e70bebbbab7c74bf50773ecf7ae108fd9cbd4`,
`d6401b795aaf1d4992cd6cf9e15bfb39f03232261a1fe3400f466acad4676e08`.

How it was computed. `liblizard/build/ob.wasm` reports the bit pair it put on every coefficient. From `liblizard/`:

```js
import { init, Focus } from "./sim/ob.mjs";
await init();
const f = new Focus(256, 16, 1);                         // LIZARD-16, rate 3/4, LINEAR
const B = f.blockBytes, blocks = new Uint8Array(2 * B);  // B = 473
for (let b = 0; b < 2; b++) { blocks[b * B] = b; for (let j = 0; j < 469; j++) blocks[b * B + 4 + j] = j & 255; }
f.measure(true); f.encode(blocks);
const slot = Array.from(f.sent(), (s) => (s < 0 ? 1 : 0));   // 10240 slot bits, frame order
```

The CRC is the wasm's `ob_test_crc_hash` and agrees with node's `zlib.crc32`. The codeword, parity and whitening rows
come from a separate JavaScript implementation of this section and section 8 (CRC, shift table construction, encoder,
PRBS, permutation). It reproduces the wasm's slots bit for bit on LIZARD-16, on LIZARD-64 (then at n = 512; the slots
do not depend on n) and on LIZARD-512 (n = 1024, 327,680 slots), and each of the other four modes agrees with its row
of the table in 7.3.

### 7.5 The block id

The transfer is in the light (2026-09-24): a camera alone recovers a file, with nothing from the network. Bytes 0
to 3 of every block are its id, a uint32 read little-endian (`liblizard/src/xfer.h:xfer_id_get`), in two fields:

| bits | field | values |
|---|---|---|
| 31 to 18 | chunk | 0 to 16382: a chunk of the file (7.6). 16383 (`XFER_CONTROL`): a control block |
| 17 to 0 | fountain symbol | a data block: its Wirehair block id in its chunk's fountain. A control block: 0 the header (7.7), 1 + m manifest block m for m = 0 to 1170 (7.8), 1172 to 262143 reserved |

- id = chunk x 2^18 + symbol. The header's id is 0xFFFC0000 (`XFER_ID_HEADER`); manifest block m's is 0xFFFC0001 + m.
- A reserved id is one this version never sends, and a receiver MUST ignore a block that carries one.
- A data block's symbols count from 0 in each chunk: 0 to K - 1 are the chunk's K source blocks verbatim, K up its
  repair blocks (7.6, 9.4). The field holds 2^18 = 262,144 symbols; a sender that runs past them wraps to 0.
- The header's id is the same in every transfer, and a manifest block's in every transfer with that many chunks, so a
  receiver that drops a block whose id it has seen misses the next file. A control block is new when its bytes are:
  the reference receiver never deduplicates one (`liblizard/sim/xfer.mjs`, `lizard-web/recv.mjs`).

Why these widths (non-normative, `liblizard/src/xfer.h`). Wirehair takes at most 64,000 source blocks, so the largest chunk is
16 MiB, 35,773 blocks (32 MiB would be 71,546). 2^18 symbols is 7.3 times that, so a sender repeats an id only once a
receiver has missed 86% of a 16 MiB chunk's blocks; 16 bits would repeat at 45%. 14 bits leave 16,383 chunks: about
16 GiB at 1 MiB a chunk, 64 GiB at 4 MiB and 256 GiB at 16 MiB. The control range is at the top so that chunk 0's ids
are 0 to 2^18 - 1, the ids the one-fountain file mode sent before.

### 7.6 Chunks and their fountains

- A file of `length` bytes is cut into chunks of 2^k bytes, k = `chunk_log2` from 10 to 24 (1 KiB to 16 MiB,
  `XFER_LOG2_MIN`, `XFER_LOG2_MAX`). Chunk c is bytes c 2^k up to the smaller of (c + 1) 2^k and length. Every chunk
  but the last is 2^k bytes on a 2^k boundary, which makes it a whole subtree of the file's BLAKE3 tree (7.9); the last
  may be shorter.
- chunks = ceil(length / 2^k), at most 16,383 (`XFER_MAX_CHUNKS`), and 0 for an empty file. A file may therefore be
  up to 16,383 x 2^k bytes: 274,861,129,728 at k = 24.
- **Compression** (2026-10-05). Each chunk is compressed on its own before it is fountained: one zstd frame
  (`liblizard/zstd/shim.c` pins the parameters: level 9, the row match finder on, no checksum, the content size in the
  frame; `liblizard/vendor/zstd/`, Zstandard 1.5.7) where that is shorter than the chunk's bytes, else the bytes as
  they are. What a chunk's fountain carries is its bytes **as sent**, 1 to len; sent < len means a frame. A receiver
  decompresses such a chunk the moment it is recovered, before its hash (7.9), with the chunk's memory alone (the
  frame's window is at most the chunk). The file's length, root and chaining values are the file's own bytes, so
  b3sum of the received file is the root whatever was compressed. Under codec 0 (7.7) every chunk is sent as it is;
  under codec 1 a sender MAY send any chunk as it is (one that does not shrink) and MUST send a frame only where it is
  shorter. A receiver MUST accept a stored chunk under either codec and MUST refuse a frame under codec 0.
- Chunk c has K = ceil(sent / 469) source blocks (`xfer_blocks`). With K >= 2 it is its own Wirehair fountain (9.4),
  its blocks carrying chunk c and their Wirehair block id. A chunk with K = 1 (sent 469 bytes or less) is not
  fountained, since Wirehair refuses a one-block message: its sent bytes go zero padded to 469 as symbol 0, repeated,
  and a receiver takes the first sent of them.
- **Seed attempts.** Wirehair picks a message's seed attempt (0 to 255) from its block count alone, and its decoder is
  built from the message length, the block size and that attempt (9.4). Compressed, every chunk has its own count, so
  every chunk's attempt travels: in the manifest (7.8), or for a one-chunk file in the header (7.7). A receiver
  therefore builds no chunk's decoder before the manifest is in; the control cycle (9.1) puts the manifest within an
  eighth of a lap, and under the interleaved schedule no chunk completes sooner. (Until 2026-10-05 the header carried
  two attempts, one for every chunk but the last, and refused a file whose full chunks seeded apart.)
- Header version 2 fixes the fountain, Wirehair V2, profile `WIREHAIR_V2_PROFILE_CERTIFIED_2026_07`, 469-byte blocks,
  and the codec, zstd frames. A change to any of them is a new version.
- k is the sender's choice inside the range. The reference sender uses 22 (4 MiB) unless `send.html?chunk=` says
  otherwise (9.1).

### 7.7 The header block

Id 0xFFFC0000. Its 469 payload bytes, integers little-endian (`liblizard/src/xfer.h`, `liblizard/src/xfer.c` `xfer_header_write`,
`xfer_header_parse`); version 2 since 2026-10-05 (version 1 carried two seed attempts at bytes 3 and 4, a type to 162
bytes, no codec and no sent):

| offset | bytes | field | value |
|---|---|---|---|
| 0 | 1 | version | 2 (`XFER_VERSION`) |
| 1 | 1 | hash | 1, BLAKE3 (`XFER_HASH_BLAKE3`); 0 and 2 to 255 are refused, the room for another hash |
| 2 | 1 | chunk_log2 | k, 10 to 24 |
| 3 | 1 | codec | 0, none (`XFER_CODEC_NONE`): every chunk sent as its bytes; 1, zstd (`XFER_CODEC_ZSTD`): a chunk sent shorter than its bytes is one zstd frame of them (7.6); 2 to 255 refused |
| 4 | 1 | seed | the Wirehair seed attempt of the one chunk of a one-chunk file of 2 blocks or more; 0 otherwise |
| 5 | 1 | name_len | bytes of name, 0 to 255 |
| 6 | 1 | type_len | bytes of media type, 0 to 158 |
| 7 | 1 | reserved | 0 |
| 8 | 8 | length | the file's bytes, u64 |
| 16 | 4 | chunks | ceil(length / 2^k), u32 |
| 20 | 32 | root | BLAKE3 of the whole file, what `b3sum` prints (7.9) |
| 52 | 255 | name | the file name, UTF-8, zero padded; empty for none |
| 307 | 158 | type | the media type, ASCII, zero padded; empty for none |
| 465 | 4 | sent | the one chunk's bytes as sent (its frame, or its own), u32, 1 to its length; 0 when the file is not one chunk |

The fields fill the 469 bytes exactly, and every byte a field does not use is zero. A receiver MUST refuse a header
block that breaks this table. The reference parse (`xfer_header_parse`) returns:

- -2 (`XFER_ERR_VERSION`) for a version other than 2, -3 (`XFER_ERR_HASH`) for a hash other than 1, and -4
  (`XFER_ERR_CODEC`) for a codec other than 0 or 1;
- -1 for chunk_log2 outside 10 to 24; chunks other than ceil(length / 2^k), past 16,383, or 0 for a file that is not
  empty; type_len over 158; a nonzero byte 7, or a nonzero byte past name_len in the name or past type_len in the
  type; with one chunk, a sent of 0 or past the chunk's length, a sent under it with codec 0, or a nonzero seed with a
  sent of 469 bytes or less (`xfer_sent_ok`); with any other number of chunks, a nonzero sent or seed.

So a block that is not a header rarely passes for one. The parse does not check that the name is UTF-8 or the type
ASCII: the reference sender cuts a name to 255 bytes on a character boundary and drops a type that is not printable
ASCII (`liblizard/sim/xfer.mjs`).

### 7.8 The manifest blocks

A file of 2 chunks or more has ceil(chunks / 12) manifest blocks (`xfer_manifest_blocks`, 1 to 1,366; 14 a block and
1,171 until 2026-10-05). A file of one chunk or none has none: its root is the one chunk's own hash and is checked
from the chunk, and its sent and seed are in the header. Manifest block m, id 0xFFFC0001 + m, 469 bytes:

| offset | bytes | field |
|---|---|---|
| 0 | 8 | tag: the root's first 8 bytes |
| 8 | 444 | the entries of chunks 12 m to 12 m + 11, 37 bytes each, in order; zero past the last chunk |
| 452 | 17 | reserved, 0 |

An entry: the chunk's chaining value (32 bytes, 7.9), its bytes as sent (u32, little-endian, 7.6) and its Wirehair
seed attempt (1 byte, 0 where it is one block). A receiver MUST refuse a manifest block whose tag is not the first 8
bytes of the root in the header it holds, whose m is past the file's last manifest block, with an entry no chunk of
that file could have (a sent of 0 or past the chunk's length, a sent under it with codec 0, a nonzero seed on a chunk
of one block: `xfer_sent_ok`), or with a nonzero byte after its last entry (`xfer_manifest_parse`). The tag keeps
another transfer's block out before it is used.

### 7.9 The chunk tree, and when a file is accepted

BLAKE3 here is the official function (the official C 1.8.7, `liblizard/vendor/blake3/`, CC0 or Apache 2.0); the chunks'
chaining values are nodes of its tree of the whole file.

- **A chunk's chaining value** (`xfer_chunk_cv`). BLAKE3 hashes in 1 KiB chunks of its own, numbered from 0 across the
  file, so chunk c's first is number c 2^(k - 10). Its chaining value is the one BLAKE3 computes for the subtree over
  the chunk's bytes, those 1 KiB chunks at those numbers and merged by BLAKE3's rule (the largest power of two of
  1 KiB chunks that leaves a byte on the right), with the ROOT flag nowhere. The official C has no call for a
  subtree, so `liblizard/src/xfer.c` builds one on its compression function; the vendored files are unmodified.
- **The root of the list** (`xfer_root`), for n >= 2 chaining values c[0 .. n): split at p, the largest power of two
  below n; the root is PARENT(root of c[0 .. p), root of c[p .. n)), where PARENT(l, r) is BLAKE3's compression of the
  64-byte block l || r with BLAKE3's IV as the key, counter 0, flag PARENT, and at the top PARENT and ROOT. One
  chaining value is its own subtree.
- This is BLAKE3's own tree: every chunk but the last is 2^k bytes on a 2^k boundary, k >= 10, so BLAKE3's split
  always falls on a chunk boundary. The root of the list is therefore BLAKE3 of the file. With one chunk the root is
  BLAKE3 of the chunk, and with none BLAKE3 of the empty input.

**Acceptance.** A chunk sent shorter than its length (7.6) is first decompressed as one zstd frame; a frame that does
not say it holds the chunk's length, is damaged, or yields anything but that many bytes is a wrong chunk, refused as
a wrong hash is (its symbols banned, the chunk collected again). A chunk is verified when its length is what the
header gives and, with one chunk, its BLAKE3 is the root, or, with two or more, its chaining value is entry c of a
list that combines to the root (`xfer_chunk_check`, `xfer_manifest_check`). A receiver MUST NOT hand over a file until every chunk is verified, and an empty file until
the root is BLAKE3 of the empty input. The block CRC (7.1) still gates every block, control blocks included. A header
or manifest block that passes its CRC-32 wrongly (1 in 2^32) is caught only when the root fails.

### 7.10 Transfer test vectors

**Ids.**

| chunk | symbol | id | bytes 0 to 3 | kind |
|---|---|---|---|---|
| 0 | 0 | 0x00000000 | `00 00 00 00` | data |
| 1 | 5 | 0x00040005 | `05 00 04 00` | data |
| 16382 | 262143 | 0xFFFBFFFF | `ff ff fb ff` | data, the last |
| 16383 | 0 | 0xFFFC0000 | `00 00 fc ff` | header |
| 16383 | 1 | 0xFFFC0001 | `01 00 fc ff` | manifest block 0 |
| 16383 | 1366 | 0xFFFC0556 | `56 05 fc ff` | manifest block 1365, the last |
| 16383 | 1367 | 0xFFFC0557 | `57 05 fc ff` | reserved |

**Files.** Each is BLAKE3's official test input, byte i = i mod 251, so each root is the official vector for its length
(`liblizard/vendor/blake3/test_vectors/test_vectors.json`). That input repeats every 251 bytes, so under codec 1 every
1 KiB chunk compresses to a 271-byte frame, one block, unfountained (the one chunk of 1,025 bytes at k = 11 the
same); the two rows under codec 0 send the same files as they are, three blocks a chunk with Wirehair's seed attempt
2, so the fountained path and the seeds have vectors too. Header version 2 (2026-10-05); version 1's bytes are not
kept. Per chunk: its bytes, its bytes as sent, K and its seed attempt.

| length | k | codec | chunks | first chunk | last chunk | manifest blocks | header bytes 0 to 19 |
|---|---|---|---|---|---|---|---|
| 0 | 22 | 1 | 0 | | | 0 | `02011601 00000000 00000000 00000000 00000000` |
| 1025 | 11 | 1 | 1 | 1025, 271, 1, 0 | | 0 | `02010b01 00000000 01040000 00000000 01000000` |
| 8193 | 10 | 1 | 9 | 1024, 271, 1, 0 | 1, 1, 1, 0 | 1 | `02010a01 00000000 01200000 00000000 09000000` |
| 31744 | 10 | 1 | 31 | 1024, 271, 1, 0 | 1024, 271, 1, 0 | 3 | `02010a01 000a1800 007c0000 00000000 1f000000` |
| 102400 | 10 | 1 | 100 | 1024, 271, 1, 0 | 1024, 271, 1, 0 | 9 | `02010a01 00000000 00900100 00000000 64000000` |
| 1025 | 11 | 0 | 1 | 1025, 1025, 3, 2 | | 0 | `02010b00 02000000 01040000 00000000 01000000` |
| 31744 | 10 | 0 | 31 | 1024, 1024, 3, 2 | 1024, 1024, 3, 2 | 3 | `02010a00 000a1800 007c0000 00000000 1f000000` |

Roots, bytes 20 to 51 of each header (the file's own bytes, the same under either codec):

```
0        af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
1025     d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444
8193     bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b
31744    62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47
102400   bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085
```

The 31,744-byte file is named `vector.bin` (10 bytes at 52, `766563746f722e62696e`, then 245 zero bytes) with type
`application/octet-stream` (24 bytes at 307, then 134 zero bytes); the others have neither. Bytes 465 to 468, the one
chunk's sent: `0f010000` (271) for 1,025 bytes under codec 1, `01040000` (1,025) under codec 0, zero in every other
row. SHA-256 of each whole 469-byte header:

```
0               3cf38facead9e13b3b9818eee2069c3c14cfd0a55a2a82344133ae4fb9a81db4
1025            8a57f03e55fd710bd493d5a6f1fa9d3458a911d8194f340be1e12452b190a941
8193            81e4e71bc4e6034e59be3896e6ead03d2202e5aef6b9ffc385deb25c2df69ff0
31744           546b119153a74764939a22ab5b0e35d2a77cae3426c90e9d25e2c075606e4dd5
102400          ca763d22a3ebf289b461062e28a1d4593dabe6e05e842a4da9e4996f3f80ee09
1025, codec 0   e7bd57580853ea8b6207d33468e526ffc635fb8dc6014e5bb6f631305d291e27
31744, codec 0  eefee20ecfafc72f8c154ecd12e2c0a1505d64f05cb122de11ee19a6c5018a97
```

Chaining values of 1 KiB chunks of this input (the file's own bytes, unchanged from version 1):

```
chunk 0, bytes 0 to 1023 (the same in every file above)   5c9e654411e393d1f4bec710ccd5bc5669ab177d610a0eb691fcfee92fb4e8b1
chunk 30 of the 31,744-byte file                          a6b76fabdd73c356e0a57cd84e53c562b819b21f140425029385c1574fe985f7
chunk 8 of the 8,193-byte file, its one byte a0           0c89b23aaf0f396fe25995d42f7d6f3c7b1e18b59d9c778b06688325f44016de
```

Manifest blocks, SHA-256 of each 469 bytes (7.8). Block 0 of the 31,744-byte file begins with the tag `62b6960e1a44bcc1`,
then chunk 0's entry: its chaining value, sent `0f010000` and seed 0 under codec 1, sent `00040000` and seed 2 under
codec 0; its three blocks hold 12, 12 and 7 entries, the 102,400-byte file's nine 12 each but the last's 4.

```
31744 m 0            62ce9f91d4b9cb6763c9540c37c0e2bcd33bb09a6427731acf80d20b391c0da6
31744 m 1            68d121c3b60340ece781152965535bd1d557feb1a93fd22cac243128691dc616
31744 m 2            1228754ba7562b9cd46d3d06374554bf300f4526dcd93b894e092f2c173be0de
8193  m 0            d89af098949dbe13d1b6e7b16834100adf3fe4fe4e9ad565ad1ed56314230826
102400 m 0           b269362930980a83cfe4148c36fc4acb625d17f603954a6fb269018288480e8e
102400 m 8           523ae573a05805e8aa7ebe518e605f5a1cbb819e4367576cfbd8e64e665c8a21
31744 m 0, codec 0   622b1095da8f8059b146706ebc17d6cc74bdd3742c5c8c24072145594d69989d
31744 m 1, codec 0   db34b752b781674f1639b2279853127614420f4287dc62961d3a5e8ee2352c7e
31744 m 2, codec 0   48fb5759408c0276086d0893b2a6a61c3fac33d8af542b9a8d66ebac9fcbb20a
```

**Refusals.** The 31,744-byte header with one field changed: version 1 gives -2; hash 0 or 2 gives -3; codec 2 gives
-4; chunk_log2 9 or 25, byte 7 set, a byte of the name's or the type's padding set, chunks one more, type_len 159, or
a nonzero sent (the file is not one chunk) gives -1. The 1,025-byte header under codec 0 with sent 300 (shorter than
the chunk under codec 0), or with sent 0, gives -1. Its last manifest block with a padding byte set, with its tag
changed, with an entry's sent past its chunk's bytes, or with a seed on a chunk of one block, is refused.

How they were made: `liblizard/build/ob.wasm` (sha256 409299a866c8...) through its `xfer_*` exports, the frames from
zstd itself (`liblizard/build/zstd.mjs` through `liblizard/sim/zstd.mjs`) and the seed attempts from Wirehair itself
(`liblizard/build/wirehair.mjs` through `liblizard/sim/fountain.mjs`). Three of the headers (1,025 bytes under both
codecs, 31,744 under codec 1) were packed again in Python from the table in 7.7 and came out byte-identical. Every
chaining value and root above is as in version 1's vectors, where a separate BLAKE3 written in Python from its
compression function, chunk and parent rules recomputed and matched them, and every root is also the official
vector. `liblizard/test/xfer_test.c` (native) and `liblizard/test/xfer_wasm.mjs` hold the C and the wasm to all 35
official vectors, whole and split into chunks of 2^10 to 2^17 bytes, and the wasm test to the compressed path on
prose files (STATUS.md, 2026-10-05).

## 8. The LDPC code

Normative except 8.3 and 8.5. Sources: `liblizard/src/ldpc_base.h` `LDPC_BASE`; `liblizard/src/ldpc.h`; `liblizard/src/ldpc.c` `ldpc_init`,
`ldpc_generate`, `count_cycles`, `ldpc_encode`; `liblizard/src/focus.h` `FOCUS_RATE`; `liblizard/src/focus.c` `init`.

A block's code is one of three quasi-cyclic irregular repeat-accumulate (QC-IRA) codes, H = [Hd | Hp], by its tier
(3.1): rate 7/8 (z = 93), 3/4 (z = 106) and 1/2 (z = 160). Hd is built from z x z circulant permutation matrices; Hp is
a bit-level accumulator (a staircase). The rates are the profile's, fixed by the version (`focus.c:focus_tiers_for`;
indices 6, 4 and 2 of the profile table in `ldpc.c`), and the format word carries none. 8.1 to 8.5 give the 3/4 code,
the one rate before 2026-10-07; 8.6 gives the 7/8 and 1/2 codes, which are built, encoded and decoded the same way at
their own sizes.

### 8.1 Parameters

| | |
|---|---|
| codeword n | 5088 = 48 z |
| information k | 3816 = 36 z |
| parity bits and checks m | 1272 = 12 z |
| lifting z | 106 |
| base graph | 12 block rows by 48 block columns: 36 data columns of circulants (Hd), then the 1272 parity bits as a staircase (Hp) |
| data column degree | 12 for columns 0 to 7, 3 for columns 8 to 35 |
| data circulants | 180, 15 in each block row |
| check degree | 17: 15 data bits and 2 parity bits. Check 0 has 16. |
| parity bit degree | 2. The last parity bit has 1. |
| edges | 21,623 |
| cycles | none of length 4. Every variable node but the last parity bit lies on a cycle of length 6. |

The size comes from `ldpc_init(code, n_max, rate, seed)` with n_max = 8 sub-channels x 320 coefficients x 2 = 5120, so
z = floor(5120 / 48) = 106 and n = 48 z = 5088. The 32 slots left over are the block's tail (7.3). `focus.c:init` passes
rate index 4 and seed 1, and for z = 106 at rate index 4 and seed 1 `ldpc_init` takes the base matrix from
`liblizard/src/ldpc_base.h` (`LDPC_BASE_Z`, `LDPC_BASE_RATE`, `LDPC_BASE_SEED`) instead of generating it. The table has the
3/4 profile's shape (`ldpc.c:PROFILE` row 4): 12 block rows, 8 heavy columns of degree 12, degree 3 for the rest.

### 8.2 The parity-check matrix

Codeword bit 106 j + t (j = 0 to 35, t = 0 to 105) is bit t of data column j. Codeword bit 3816 + c is parity bit c
(c = 0 to 1271). Every "mod" below is the non-negative remainder.

Checks are numbered c = 12 i + r, where r (0 to 11) is the block row and i (0 to 105) the position in it. Check c
contains:

- for each data column j that has a shift sh[r][j] in block row r: data bit 106 j + ((i - sh[r][j]) mod 106);
- parity bit c - 1, when c > 0;
- parity bit c.

Read the other way, data bit t of column j is in check 12 ((t + sh[r][j]) mod 106) + r of every block row r where
sh[r][j] exists. Hp is the 1272 x 1272 matrix with ones on its diagonal and just below it.

The numbering interleaves the block rows along the staircase, and the code needs it. With checks numbered 106 r + i
instead, data bits t and t + 1 of a degree-3 column land on adjacent staircase checks in every block row, and those two
bits with the three parity bits between them form a codeword of weight 5 (`ldpc.c`, the comment above `count_cycles`).

**The shift table sh** (a dash: no circulant). This table is H, and it is normative (2026-09-24): the format's
code is this table, not a construction. The reference ships it as `liblizard/src/ldpc_base.h:LDPC_BASE`, 12 rows of 36 shifts
with -1 for a dash, and `ldpc_init` builds the format's code from it; the table below is that array, entry for entry.
No profile, seed or score moves it.

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0       3  87  22  87  41  20  44  67   -   -   - 102   -   -   -   7  81   -
1      80  30  28  45  42  23  37   0   -   -   -  70   -  13   -   -  62   -
2      58  72  29  92  57  83  53  51   -  56   -   -   -   -  11   -   -   -
3      91  15  61  49  79  95  88  18   -   -   -  41   -   5   -   -   -  81
4      77  71  71  77  64  54  46 102   -   -  30   -   -   -   -  50   -   -
5     100  93  44  43  11  51  63  36   -  86   -   -   -   -   -   9   -   -
6      41  64  87  31  52  43  19   6  80   -   -   -  48   -   -   -   -   -
7      46  22  10  41  22  46  49  78   -  63   -   -   -   6   -   -   -  28
8       9   7  18  25  72  79  60   8   2   -   -   -   -   -  63   -   -   -
9      71  26  56  38  51  70   4  23  73   -   -   -  36   -   -   -   -   -
10     85  75  42  51  92  31  47  27   -   -  41   -   -   -  79   -   0   -
11      9  21   4  48  43  77  67 105   -   - 105   -  66   -   -   -   -  50

r \ j  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
0       -   -   -  96   -   -   -  15   -   -  27   -   -   -   7   -   -   -
1       -   -   -   -  11   -  74   -   -   -   -   -   -  51   -   -   -  68
2      63   -   -  19   -   -   -  88   -   -   - 100   -   -   -  79   -   -
3       -   -   -   -   -  67   -   -   -  77   -  68   -   -   -   1   -   -
4      93   -  19   -   -   -   -   -  27   -   -   -   -  36   -   -  68   -
5       -  55  67   -   -   -   -   -   -  47  20   -   -   -   -   - 105   -
6       -  52   -   -   -  88   -   -   -  62   -   -  52   -  16   -   -   -
7       -   -   -  98   -   -   -   -  51   -   -   - 105   -   -   -   -  96
8      15   -   -   -  88   -  39   -   -   -   -  21   -   -  46   -   -   -
9       -  81   -   -   -  28   -   -  31   -  49   -   -   -   -  37   -   -
10      -   -   -   - 105   -  83   -   -   -   -   -   -  15   -   -  40   -
11      -   -  90   -   -   -   -  46   -   -   -   -  94   -   -   -   -  50
```

### 8.3 How the table was made (non-normative)

`liblizard/src/ldpc_base.h` was written out on 2026-09-24 from what the generator made for this code, and the generator stays:
`ldpc.c:ldpc_generate` builds every code outside the format (the binary code at its layout's z, the rates the
experiments tier with, `liblizard/test/ldpc_ablate.c`'s profiles), and the format's own when a test sets
`ldpc_override_heavy`. `liblizard/test/ldpc_table.c` prints the table from the generator and fails if the two differ, or if
`ldpc_init`'s H is not the one generated before the table existed (FNV-1a over its row pointers and column indices,
2ca2bc24; 21,623 edges); a table altered in one entry fails it. A change to the generator now moves only codes outside
the format. `ldpc.c` also has `ldpc_norm`, a test global that changes the decoder's scale and not H.

The construction, as `ldpc.c:ldpc_generate` and `ldpc.c:count_cycles` run it. Columns are filled in order, one edge at
a time. Each edge goes in the
least-used block row the column is not yet in, and its shift is the best of up to 12 random candidates by a cycle score.
Arithmetic on s is unsigned 32-bit with logical shifts. md(a) is the remainder of a modulo 106, in 0 to 105.

```
s = 1 * 2654435761 + 4 * 97 + 1 (mod 2^32) = 2654436150       seed 1, rate index 4 (s = 1 if this were 0)
rnd(): s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s         xorshift32

sh[r][j] = none for all r, j; rowdeg[r] = 0 for all r
for j = 0 .. 35:
  deg = 12 if j < 8 else 3
  repeat deg times:
    best = none; bd = infinity; tie = 0
    for r = 0 .. 11 where sh[r][j] is none:
      t = rnd()                                                  drawn for every such row, tie or not
      if rowdeg[r] < bd or (rowdeg[r] == bd and t > tie): best = r; bd = rowdeg[r]; tie = t
    p4 = infinity; p6 = infinity
    repeat 12 times:
      cand = rnd() mod 106
      (four, six) = score(best, j, cand)
      if four < p4 or (four == p4 and six < p6): pick = cand; p4 = four; p6 = six
      if four == 0 and six == 0: stop repeating
    sh[best][j] = pick; rowdeg[best] += 1
```

score(r, j, x) counts the short cycles that shift x in block row r of column j would close against the shifts already
placed. It includes cycles through the staircase: checks c and c + 1 share parity bit c, so a data bit in both closes a
4-cycle. The counts are the construction's own score, not exact cycle counts, so reproduce them as written:

```
four = 0; six = 0
for r2 = 0 .. 11 where r2 != r and sh[r2][j] exists:
  d = x - sh[r2][j]
  if (r2 == r + 1 or r2 == r - 1) and sh[r2][j] == x: four += 1
  if r == 11 and r2 == 0 and md(sh[r2][j] - x) == 1: four += 1
  if r == 0 and r2 == 11 and md(x - sh[r2][j]) == 1: four += 1
  for j2 = 0 .. 35 where j2 != j and sh[r2][j2] exists:
    if sh[r][j2] exists and md(d - (sh[r][j2] - sh[r2][j2])) == 0: four += 1
    for r3 = 0 .. 11 where r3 != r, r3 != r2 and sh[r3][j2] exists:
      d2 = d + sh[r2][j2] - sh[r3][j2]
      for j3 = 0 .. 35 where j3 != j, j3 != j2, sh[r3][j3] exists and sh[r][j3] exists:
        if md(d2 + sh[r3][j3] - sh[r][j3]) == 0: six += 1
return (four, six)
```

The C loops j2 and j3 over all 48 base columns. Columns 36 to 47 never hold a shift, so 0 to 35 is the same.

### 8.4 Encoding

Systematic and linear-time (`ldpc.c:ldpc_encode`). The codeword is the k information bits, then p_0 to p_(m-1) (at 3/4,
3816 bits, then p_0 to p_1271):

```
p_c = p_(c-1) XOR (XOR of the data bits in check c),    p_(-1) = 0
```

### 8.5 Soft values into the code, and the reference decoder (non-normative)

What the reference decoder feeds the code, and how it decodes (`focus.c:focus_finish_bits`, `focus.c:axis_info`,
`focus.c:map_in`, `ldpc.c:ldpc_decode_stall`). The format fixes only the sign convention and the bit map. A decoder may
compute and scale its soft values any way it likes.

**Per sub-channel**, over its 320 received coefficients y, after the transform and the detrend (6.9, 6.10):

- m2 = mean of |y|^2, m4 = mean of |y|^4.
- A2 = sqrt(max(0, 2 m2^2 - m4)), the signal power. For constant-modulus QPSK in complex Gaussian noise of power N,
  E[m2] = A2 + N and E[m4] = A2^2 + 4 A2 N + 2 N^2, so the estimate is exact at the expected moments; from the sample
  moments of 320 coefficients it is biased, since E[m2^2] is not E[m2]^2.
- N = max(m2 - A2, 0.02 m2).
- Each axis gets LLR = 0.7 x 2 sqrt(2 A2) / N x y_axis, in natural-log units. 2 sqrt(2 A2) / N is the exact BPSK LLR
  for an axis amplitude of sqrt(A2 / 2) in noise of variance N / 2. The 0.7 damps it.
- Each LLR is clipped to [-10, 10] and stored as int8 = round(8 x LLR), ties to even (`nearbyintf`). Full scale is -80
  to 80.
- Positive means slot bit 0.

**Per block**, slot order to codeword order: for i < n, L_code[st i mod n] = L_slot[i], negated where
w[640 s_b + i] = 1 (at 3/4, L_code[1943 i mod 5088]). The tail slots are dropped. At 7/8 and 1/2 the information bits
past the CRC, 3816 to k - 1, are fed as certain zeros (+127).

**Decline gate.** Each sub-channel's information per coded bit is estimated as J(2 sqrt(A2 / N)), with ten Brink's
approximation J(sigma) = (1 - 2^(-0.3073 sigma^1.787))^1.1064. A block's estimate is the mean over its sub-channels.
A block under 0.2 x k / n (`FOCUS_DECLINE`: 0.175 at 7/8, 0.15 at 3/4, 0.10 at 1/2) is not decoded.

**LDPC decoder**, layered normalised min-sum:

- A layer is one block row. Its z checks share no bit, so they run side by side.
- Check-to-bit messages are int8, magnitude at most 127. Posteriors are int16, clamped to [-8191, 8191].
- min1 and min2 are scaled by 13/16, as (x x 13) >> 4.
- The sweep over the mb layers alternates direction every iteration, because the staircase carries parity information
  only one step against the sweep per pass.
- Check 0 has no parity bit c - 1. The layer feeds that slot as a certain 0 (posterior 8191).
- Decoding stops when the syndrome is zero, after at most 30 iterations. A block whose count of unsatisfied checks after
  iteration 9 is above 0.95 of its count after iteration 1 is given up (`FOCUS_STALL_IT`, `FOCUS_STALL_RATIO`).
- A posterior below 0 decides bit 1.

This describes `liblizard/src/ldpc.c` as of 2026-09-22 and `liblizard/src/focus.c` as of 2026-09-23. An LDPC early stop under test in the
speed round has not landed; if it does, the stopping rule above changes and the format does not.

### 8.6 The 7/8 and 1/2 codes

The rate profile's other two codes (3.1), normative like the 3/4 code. Each is H = [Hd | Hp] as in 8.2, at its own z:
codeword bit z j + t (t = 0 to z - 1) is bit t of data column j, codeword bit k + c is parity bit c, check c = mb i + r
(r the block row, 0 to mb - 1, i the position, 0 to z - 1) holds data bit z j + ((i - sh[r][j]) mod z) for each shift
of block row r, parity bit c - 1 when c > 0, and parity bit c. Encoding is 8.4's accumulator over the m checks.

| | 7/8 | 1/2 |
|---|---|---|
| sub-channels a block, slots | 7, 4480 | 12, 7680 |
| codeword n | 4464 = 48 z | 7680 = 48 z |
| information k | 3906 = 42 z | 3840 = 24 z |
| parity bits and checks m | 558 = 6 z | 3840 = 24 z |
| lifting z | 93 | 160 |
| base graph | 6 block rows, 42 data columns | 24 block rows, 24 data columns |
| data column degree | 4 for columns 0 to 5, 3 for 6 to 41 | 8 for columns 0 to 9, 3 for 10 to 23 |
| data circulants | 132, 22 in each block row | 122, 5 in each block row but rows 4 and 20, which have 6 |
| check degree | 24: 22 data bits and 2 parity bits (check 0: 23) | 7, 8 in block rows 4 and 20 (check 0 one fewer) |
| edges | 13,391 | 27,199 |
| cycles of length 4 | none | none |
| H, FNV-1a over row pointers and column indices (`liblizard/test/ldpc_table.c`) | d942b2a6 | 83296c5e |

They were written out on 2026-10-07 from what `ldpc.c:ldpc_generate` made for (z = 93, rate index 6, seed 1) and
(z = 160, rate index 2, seed 1): 8.3's construction with the profile table's rows 6 (6 block rows, 6 heavy columns of
degree 4) and 2 (24 block rows, 10 heavy columns of degree 8), s starting at 2654435761 + 6 x 97 + 1 and + 2 x 97 + 1,
md modulo z, and the loops over each code's own block rows and data columns. The reference ships them as
`liblizard/src/ldpc_base.h:LDPC_BASE78` and `LDPC_BASE12`, which `ldpc_init` builds from, as it does the 3/4 code from
`LDPC_BASE`; `liblizard/test/ldpc_table.c` holds all three to the generator and to the hashes above. 8.5 applies at each
code's own z, mb and decline bar.

**The 7/8 shift table** (a dash: no circulant):

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0      89  62   -  62   -  76  21   -   -  40  52   -   -  24  69   -   -  89
1       8   -  62  72   -  32   -  77  71   -   -  54  64   -   -  57  45   -
2       -  78  40   9  66   -  33   -  14   -  54   -   -  69   -  23   -  72
3      39   -  64   -  34  53   -  92   -  89  24   -   3   -  72   -  32   -
4      32  49   -  20  25   -   -  37   -  86   -  27   -  81   -  90  30   -
5       -  36  25   -  33  28  21   -  45   -   -  34  29   -  27   -   -  82

r \ j  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
0      37   -  81   -   0   -  31   -   -   6  83   -  58   -   -  50   -  83
1       -  43   -  30   6   -   -  35  32   -  11   -  81   -  84   -   -  58
2      74   -  49   -   -  23   1   -   -  34   -  20  61   -  17   -   -  89
3      19   -   -  41  18   -   -   3   -  82   0   -   -  11   -  26  42   -
4       -  60  13   -   -  55   -  91   0   -   -  92   -  82  55   -  19   -
5       -  51   -  30   -  24  83   -  55   -   -  58   -  78   -  39  57   -

r \ j  36  37  38  39  40  41
0       -  34  13   -   6   -
1      31   -   -  51  23   -
2       -  54  23   -  62   -
3      47   -  85   -   -  70
4      74   -   -  42   -  42
5       -  70   -  89   -  28
```

**The 1/2 shift table**:

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0       -   - 126   -  13   -   -   -  99  31   -   -   -   -   -   -   - 148
1       - 125   -   -   -  26 155   -   -  57   -   -   -   -   -   -   -   -
2       -   -  31   -   -  10   -  26   -   -   -   -   -  27   -   -   -   -
3       -   - 125   -   - 108   - 102   -   -  33   -   -   -   -  17   -   -
4     138   -   - 142   -   - 108   -   -   -   - 144   -   -   -   -   -   -
5      70   -   - 136   -   -   -   -  48   -   -   -  14   -   -   -   -   -
6       -   -  83   - 124   -   -  97   - 136   -   -   -   -   -   -   -   -
7     128   -   -  23   -   -  50   -   -   -   -  37   -   -   -   -   -   -
8       - 131   -  71   -   -  93   -   -  62   -   -   -   -   -   -   -  64
9     110   -   -   -  20   -   -   - 140   -   -   - 115   -   -   -  93   -
10      -   -  15   -   -  93 112   -   -   -   -  12   -   -   -   -  42   -
11      - 106   -  80   -   -   -  47   -   -  65   -   -   -   -   -   -  35
12      -   - 150   -   - 122   -   -  15  96   -   -   -   -   -   -   -   -
13     55   -   -   - 110   -   -   -  63   -   -   -   - 142   -   -   -   -
14     85   -   -   -  69   -   -   - 157  15   -   -   -   -   -   -   -   -
15      -   -   8   -   -  74  44   -   -   -   -   -   -   -   -  43 140   -
16      - 124   -   - 127   -   - 120   -   -   -   -   -   -  90 142   -   -
17      -  19   -   -   - 129   -   -   2 123   -   -   -   -   -   -   -   -
18      -  89   -   -  27   - 107   -   -  23   -   -   -   -   -   -   -   -
19      - 136   -   2   -   -   - 123   -   -   -   -   -   - 107   -   -   -
20     10   -   -  93   -   - 134   -   -   -   -   -   -  69   -   -   -   -
21    159   -   -  99   -   -   - 109   -   -   -   - 131   -   -   -   -   -
22      - 108   -   -   - 135   - 145   -   -  41   -   -   -   -   -   -   -
23      -   -   7   - 105   -   -   -  72   -   -   -   -   - 158   -   -   -

r \ j  18  19  20  21  22  23
0       -   -   -   -   -   -
1      58   -   -   -   -   -
2       -   -   -  67   -   -
3       -   -   -   -   -   -
4      70   -   -   -   - 133
5      15   -   -   -   -   -
6       -   -   -  52   -   -
7       -   -   -   -  44   -
8       -   -   -   -   -   -
9       -   -   -   -   -   -
10      -   -   -   -   -   -
11      -   -   -   -   -   -
12      -  67   -   -   -   -
13      -   -   -   -  21   -
14      -   - 121   -   -   -
15      -   -   -   -   -   -
16      -   -   -   -   -   -
17      -   -   -  88   -   -
18      -   -   -   -   -  19
19      -   -   -   -  82   -
20      -   - 120   -   - 104
21      -  36   -   -   -   -
22      -   -  48   -   -   -
23      - 107   -   -   -   -
```

## 9. Animation and transport (non-normative)

A Lizard symbol is one frame. This section is how the project's sender and receiver run a stream of them. The display
rate travels in the format word and the transfer's layout in its blocks (7.5 to 7.9); how the rig schedules them lives in
`liblizard/sim/` and `lizard-web/`, outside `liblizard/src/`.

### 9.1 Frames and block ids

- **The count** (the pilots, 7.3; 2026-09-30, mod 4 since 2026-10-01). c is the sender's count of the pictures it has
  painted, mod 4: 0 on its first picture, then 1, 2, 3 and 0 again. A picture the display shows for two refreshes keeps
  its c, so a receiver reads a repeat as the same count twice, and a picture never shown as a count two on. Both codes
  of a two-code frame carry the frame's c. The count restarts with the sender (`lizard-web/send.mjs` `seq`, the page's
  painted count; `liblizard/gpu/encoder.mjs` passes each encode's first frame's).
- **The block id.** Bytes 0 to 3 of a block are its id (7.5), bytes 4 to 472 its 469 bytes of payload (7.1). The
  picture codec never reads the id: to `liblizard/src/focus.c` all 473 bytes are payload.
- **A file** (`liblizard/sim/xfer.mjs:XferSender`, used by the sender page and its worker). The ids come from the transfer's
  schedule, and the payload is what each id names (7.5 to 7.8):
  - **Laps.** Every chunk is interleaved with every other over the whole file, in proportion to its blocks, and laps
    repeat for as long as the sender runs, each chunk's symbols counting on, so every lap sends fresh repair ids. A
    lap is a full chunk's K rounds, and a round visits every full-size chunk once. A shorter last chunk of K blocks gets
    K + ceil(2 sqrt(K)) visits a lap, at least 64 and at most a full chunk's K, spread over the rounds by an
    accumulator: its count received wanders more than a big chunk's, and the file waits on whichever chunk is last. A
    chunk of one block sends symbol 0 at every visit.
  - **Control blocks.** Slot 0 of a frame, block 0, the lowest ring and the last a frame loses, carries a control block
    once `every` data blocks have gone since the last one: the header on even turns and the manifest's blocks in turn
    on odd ones. every = 63 (1.6% of the airtime), or lap / (8 m) data blocks when that is less, m the manifest blocks
    (at least 1), so a short file gets them more often. At LIZARD-512, 61 blocks a frame, that is a control block every
    frame and the header every other frame. An empty file's frames are all header.
  - **Shuffle.** Each frame's data slots are shuffled (Fisher-Yates, xorshift32 seeded from the frame count), because
    slot k is ring k, and a round robin whose chunk count divides the frame's would pin a chunk to rings a distant
    camera never reads.
  - **Chunk size.** 4 MiB unless `send.html?chunk=10..24` says otherwise. Interleaved, the chunk size does not move the
    rate (Wirehair costs 7 to 9 ns a byte from 1 to 16 MiB in this wasm), so the receiver's memory sets it: a 4 MiB
    solve fits the fountain worker's heaps as they start (Wirehair 16 MB, codec 8 MB), where a 16 MiB decoder alone is
    about 24 MB.
- **The test stream.** Block b of a frame (b = 0 the lowest-frequency block) carries id base + b, where base is the
  sender's id counter, which starts at a random id under 2^30 at each start (so a receiver takes a restarted stream as
  new with no signal) and advances by V after each frame (`lizard-web/send.mjs:start`, `nextId`). No id is sent twice, so no
  frame repeats. Since block 0 survives longest, blur and distance take a frame's highest ids first. These ids do not
  follow the chunk layout; a receiver tells a test frame from a file's by its blocks' bytes (9.3), and
  `liblizard/sim/phy.mjs:tally` sets a control-range id aside rather than compare it to the stream.
- **A change of format mid-transfer.** The sender re-picks the format when its room changes. The test stream's counter
  is kept apart from the frame count so a change of V never walks back onto ids already sent, and a file's schedule
  simply fills the new frame size. Every format has the same 469-byte payload, so the fountain's block size never
  moves; the sender refuses a change that would move it (`lizard-web/send-worker.mjs`, the `respec` message).

Worked example (encoded with `liblizard/sim/phy.mjs:makePhy`, the drive scaled 2x, nearest, on white, decoded by the told
receiver): LIZARD-16 painted with base 1000 returns ids 1000 and 1001; LIZARD-512 with base 1000 returns 1000 to 1063;
both 0 bad, their words reading version 2 and version 64 at 24 fps.

What the id does:

1. In a file transfer it names the chunk and the Wirehair block within it, or a control block (7.5).
2. In a test transfer it seeds the payload, so a receiver can check any block without knowing which frame carried it
   (9.3).
3. The receiver counts a data block as new once per id (`lizard-web/recv.mjs`, two generations of 2^16 ids). A capture whose
   verified blocks are all old is a repeat of a displayed frame, not a decode; one with none is empty. Control blocks
   are handed on every time (7.5).

### 9.2 The display rate

- The rig's sender paints on `requestAnimationFrame` at a period of 1000 / fps ms (`lizard-web/send.mjs:start`, `tick`), so
  each frame stays up a whole number of panel refreshes: 2 or 3 at 24 fps on a 60 Hz panel.
- The sender states the rate in the format word's third byte: 1 to 255 whole frames a second, 0 for none (5.1,
  `liblizard/src/fmt.c:ob_fmt_encode`). `liblizard/src/focus.c:focus_fmt_fps` refuses anything outside 0 to 255. It is the rate the sender
  means to paint at, not the rate it achieved, so a sender falling behind shows as one. Changing it repaints only the
  word's cells (`liblizard/src/layout.c:ob_layout_set_fmt`), so it may change between any two frames without touching
  registration.
- No decode decision depends on it (`liblizard/src/fmt.c:ob_fmt_decode` takes it unchecked). A receiver uses it to judge its
  capture rate against the display (a camera below the display rate cannot see every frame; the rig's receiver warns
  when that happens) and to tell a repeat from a missed frame.
- **Throughput.** The ceiling is V x 469 x fps bytes a second (3.4): LIZARD-512 at 24 fps is 64 x 469 x 24 = 720,384
  B/s. A 60 fps camera sees each 24 fps frame 2.5 times on average, and since blocks count once per id, a receiver gets
  the union of what its captures of each displayed frame verify.

### 9.3 The test stream

In a test transfer, block id i carries the first 469 bytes of SHAKE256 (FIPS 202) of i as four little-endian bytes
(`liblizard/src/shake.c:stream_fill`). The stream is a function of the id alone, so the receiver needs nothing but the id to check
a block. Test vectors, the first 16 bytes of output, from the wasm `stream_fill` and matched by Python's
`hashlib.shake_256`:

| id | SHAKE256(LE32(id)), first 16 bytes |
|---|---|
| 0 | `b4a21e7939beb125bd3c10237ed5ea31` |
| 1 | `220ee668c9ea457d0330c21554da9e11` |
| 12345 | `08781f021b171adcc9fd8886013dffc2` |
| 4294967295 | `8c82cb0926f0275dc7650c348eae46b2` |

Id 0's 469 bytes end in `2149dd1e8e99a4fc`.

**Judged from the light** (2026-09-26: the receiver is told nothing, not even that a test stream is on screen). A block's
id is in the light, so its test payload is known, but a file's ids cover the same range, so a single block does not say
which it is. A frame does: the sender paints the test stream or a file, never both in one frame. The reference receiver
(`liblizard/sim/phy.mjs:blockJudge`, both decoders) takes a frame to be the test stream's when one of its CRC-verified data blocks
carries its id's SHAKE256 (a file's block does so with probability 2^-3752), and then counts every other verified data
block of that frame as bad (a wrong block the CRC passed) and drops it; a frame with none is a file's, judged by the
CRC and BLAKE3 alone, and a test frame's blocks go to no transfer. The one bad block not counted is in a test frame
whose every verified block is wrong. A negative control (`lizard-web/check_rates.mjs` `LIE=1`: one block of every 24th frame
painted with a bit flipped, its CRC right) is counted bad by both decoders.

`liblizard/sim/phy.mjs:sourceFor` maps a spec's `stream` field to a generator: `"shake256"`, which `lizard-web/send.mjs` always
paints, or `"legacy"`, which is also what a missing field means, for a replay of a recording. Legacy (`liblizard/sim/phy.mjs:blockBytesLegacy`) is an xorshift32 walk
kept for recordings made before 2026-09-21: s = ((id + 1) x 0x9E3779B1 mod 2^32) XOR 0x85EBCA6B, then for each byte
s ^= s << 13, s ^= s >> 17, s ^= s << 5 (32-bit, logical shifts) and the byte is s & 255; id 12345 gives
`f278e81edbb4b2a215c9014f630baa69`. All 19 recordings in `research/captures` are SHAKE256.

### 9.4 The fountain: Wirehair as the project uses it

- **Codec.** Wirehair V2 (`liblizard/vendor/wirehair`, through `liblizard/wirehair/shim.cpp`, driven by `liblizard/sim/fountain.mjs`), with the
  profile pinned to `WIREHAIR_V2_PROFILE_CERTIFIED_2026_07`. A sender and receiver on different profiles would disagree
  about the equations with no way to notice, so the shim refuses any other.
- **Sizes.** Block size 469 bytes. A message must span 2 to 64,000 blocks, so 470 to 30,016,000 bytes; the encoder
  refuses anything else (`InvalidDimensions`). A chunk is at most 16 MiB, 35,773 blocks, and a chunk of one block is
  not fountained (7.6), so a Lizard file of any length up to 16,383 chunks goes, an empty one as a header alone.
- **Seed.** The encoder chooses a `seed_attempt` of 0 to 255 (`liblizard/wirehair/shim.cpp:lizard_wh_encoder_create`). A
  decoder is built from the message length, the block size and that seed alone
  (`liblizard/wirehair/shim.cpp:lizard_wh_decoder_create`). Every chunk's travels in the manifest, a one-chunk file's
  in the header (7.6 to 7.8); the message is the chunk's bytes as sent.
- **Symbols.** In a chunk of K blocks, symbols 0 to K - 1 are its blocks verbatim; the last is zero-padded to 469 bytes
  on the wire and trimmed back to its true length before the decoder takes it (`liblizard/sim/fountain.mjs:Decoder.add`).
  Symbols from K up are repair blocks. A duplicate is idempotent.
- **Recovery.** Enough distinct symbols recover a chunk whichever they are, so a missed frame costs only time.
  Checked: a 10,000-byte message (N = 22, seed_attempt 0) recovered from 22 repair blocks, ids 1,000,000 to
  1,000,021, and again from ids 4,294,967,255 to 4,294,967,276 (one fountain over the whole 32-bit id, before chunks).
- **The receiver** (`liblizard/sim/xfer.mjs:XferReceiver`, in `lizard-web/fountain-worker.mjs`). It takes blocks in any order, control
  blocks included. It keeps each chunk's blocks in the origin private file system (sync access handles in the
  fountain worker) where the browser has one, otherwise in memory (`recv.html?store=memory` forces memory), since
  interleaving means no chunk completes before the first lap does. A chunk is solved once the manifest has said its
  block count and K distinct symbols are in, fed more as they come if Wirehair wants them, decompressed where it was
  sent as a frame, and checked the moment it decodes (7.9). A chunk that fails is refused, every symbol that went into it banned, and it is collected again
  from later laps. The file is offered only when every chunk verified.
- **The header** comes from the light only (2026-09-26: the receiver is told nothing, and no server is assumed). A
  header with other bytes is a new transfer, everything held dropped; a receiver lets go of the ids it has seen when a
  header's root changes, since a new transfer's ids repeat the last one's.
- The binary grid code's file path, whose header went over the rig's network, was removed with its receiver path on
  2026-09-26, and the Aztec and QR paths with the baselines' archive the same day (12).
- Compression: zstd, a chunk at a time (7.6), applied by every reference sender since 2026-10-05 (`liblizard/sim/xfer.mjs`
  through `liblizard/sim/zstd.mjs`, the native `XferTx` through `liblizard/zstd/shim.c`). Until then the project fixed
  LZMA on paper and no sender applied it.

## 10. Reference receivers (non-normative)

Two decoders read the format. Neither defines it. Both report each frame's two pilot readings (7.3), and where the
frames flash the Android app's phase lock holds by what they say a capture holds of its neighbours.

### 10.1 The C decoder (`liblizard/src/`)

The decoder the phone rig carries: wasm with SIMD128, one thread a worker, a pool of workers that grows only while
frames are lost to busy ones (`lizard-web/recv.mjs`, `lizard-web/pool.mjs`). The rig's receiver is told nothing (2026-09-26): each
worker runs `liblizard/sim/phy.mjs:makeBlind` over `liblizard/sim/ob.mjs:FocusAny(nmax)`, the wasm's `focus_any_*` (`liblizard/src/wasm.c`), with
nmax 1536, the ladder's top (`BLIND_NMAX`, since 2026-09-29; 1024 by default before, 1536 from the receiver's sizes menu,
which went that day; 2048 until 2026-09-27). Since 2026-09-27 it bootstraps from the ring (5.5):

- `focus_any_setup(nmax)` builds four ring codecs, one a ring (32, 64, 96, 128), each at n = 256.
- `focus_any_rx(img, iw, ih, gamma, mesh, blocks, ok, held)` registers the frame against the four rings' layouts
  (`liblizard/src/focus.c:focus_acquire_ring`, a bounded search of 4 inside the finder, like the 8 orientations), and the ring
  that registered reads the word. Its version names n (3.2); the codec for that (ring, n) pair is built the first time
  the pair is seen and cached (`ANY_PICS` = 2, the least recently used replaced), each at the top format of its n,
  since a lower format's blocks are a prefix of it and the rest decline (3.2). The registration is handed to it
  (`focus_hand_over`), which reads the word again and finishes the frame.
- `held` is the version of the last word the caller read, 0 for none. A frame whose word does not read is finished at
  the held configuration in the ring it registered; with none held it is not decoded (`focus_release`). A picture past
  nmax is not decoded. No frame is registered again at another ring when its word fails. `makeBlind` holds the last
  word across a run's frames, each pool worker its own; that is the receiver's one state between frames.
- `_focus_any_which` gives the ring (-1 none), `_focus_any_n` the picture finished (0 none), `_focus_any_held` whether
  the held configuration stood in, `_focus_any_fmt` the word this frame read (never the held one),
  `_focus_any_blocks` and `_focus_any_max_blocks` the block counts.
- A frame is read once. The decision-directed shift pass the blind codecs ran from 2026-09-27 (the failed blocks read
  again through the shift the verified blocks' codewords gave) was deleted on 2026-09-28 (section 12): on the two
  recordings on the rings the C reads 19,960 and 22,266 blocks without it, 20,256 and 23,934 with it, 0 bad either way.

Checks (2026-09-27 evening, the four rings): every (ring, picture) pair read blind byte-exact, 48 of 48
(`scripts/exp/ring_pairs.mjs`: the first and last format of each picture, twelve, in each ring); a wordless frame decoded at
the held word and not without one, 5 of 5 (`scripts/exp/ring_held.mjs`). On the morning's three rings `scripts/exp/fmt_word.mjs` read
287 of 287 words, 0 wrong, blind equal to told in every row; not repeated on the four. `liblizard/src/focus.c:focus_decode`, the
decode of one known codec, is `focus_acquire` then `focus_finish`:

1. **Luma.** (77 R + 150 G + 29 B + 128) >> 8 (`liblizard/src/acquire.c:ob_luma`), then an optional gamma table (1 for the grey
   picture, 6.11).
2. **Binarize, for the finder only.** 8 x 8 block means; each block's threshold is the mean of the 5 x 5 blocks round
   it, less 3 (`liblizard/src/acquire.c:binarize`). Picture samples are never thresholded.
3. **Find** (`liblizard/src/acquire.c:find_frame`). The corner marks are scanned for directly in the image (`mark_scan`: a gapped
   form, then a merged one, then the merged squares read as mark cores; 4.5). Quads come from the four marks' mutual
   agreement (`mark_rect`), and each is scored by reading the timing track against its own local levels
   (`thin_levels`, `thin_score`; 4.7) against every ring's layout (a told decode has one), under the 4 rotations,
   and under the 4 mirror images only while no rotation holds the best. A rotation holds at 0.15 x max(1,
   sqrt(128 / c)) or more, c the ring's track cells (`holds_in`): 0.212 in the 32 ring, 0.15 in the others, the same
   significance against a picture's texture on every ring (since 2026-10-04; at a flat 0.15, mark-like squares in a
   picture settled a 32-ring quad before the true ring was scored). The best reading of a form's quads over every ring
   wins, not the first to hold. A hold settles it; otherwise the search goes on through the forms and on the image
   halved and quartered, and a mirror image never ends it. If no mark quad settles, the border's lines are fitted
   (`row_points`, `col_points`, `fit_lines`) and every combination is scored the same way, ring by ring until one
   holds, one read of each side in both directions serving all 8 hypotheses (`thin_read_both`), the best of all of
   them winning. A best score of 0.06 from either path is accepted. If nothing is and a mark was seen, the image is
   cropped round each box the mark implies (up to 4 a layout) and searched again, the best crop's reading winning and
   a rotation at a flat 0.15 ending the search (`mark_crop`).
4. **Register** (`liblizard/src/acquire.c:acquire_core`). A homography from the quad. The corner nodes, then the border nodes in
   rings outward, are each refined by correlation against the border as painted (`refine_node`, 4.10), reading a
   box-blurred copy of the border band once a module spans 2 px or more (`border_blur`: whenever its box has a radius, floor(mpx / 2) >= 1). Each
   side's nodes are then moved along the side onto a cubic fitted to their along-side displacement, the across-side
   part and the corners kept as measured (`smooth_along`, since 2026-09-27): a display that scales the symbol by a
   fraction moves the border's sharp cells along each side in a ripple of period 1 / frac(device px a module)
   modules that the picture does not share, and nodes 16 modules apart alias it. The interior comes from the border
   alone (`border_fill`): a radial k |q|^2 q plus affine fit to the border residuals, then a Coons patch of what the
   fit leaves.
5. **Format word**, before anything is sampled (5.4), in the ring that registered. The blind receiver hands the
   registration over to the picture the word names, or the held one (5.5).
6. **Sample** (`liblizard/src/acquire.c:ob_sample_grid`). The n x n picture at the points of 6.9, through the homography plus the
   bilinear node correction, a strip of 64 columns at a time.
7. **Detrend** (6.10), fitted while the strips are sampled and subtracted in the spectrum.
8. **Transform.** Along y a strip at a time, two real columns packed as one complex transform; along x only the kept
   rows, v < n / 2.
9. **Soft values, decline gate and LDPC** (8.5).
10. **CRC-32** over the block's 473 bytes (7.1). A block that passes goes to the page, which keeps the new ids and hands
    their payload to the fountain.

`liblizard/src/focus.c` also accepts a quad, a sampled grid, soft values or hard decisions made elsewhere (`focus_acquire_quad`,
`focus_finish`, `focus_finish_ext`, `focus_finish_bits`), so another front or back half is held to the C's acceptance
rules.

### 10.2 The GPU decoder (`liblizard/gpu/`)

A second decoder, designed from the format for WebGPU rather than ported from the C (a port that copied the C's control
flow was deleted on 2026-09-23). A batch of frames is one command encoder and one readback: a trained proposer and a
trained classifier find the corner marks, votes along the symbol's diagonals find its centre, the timing track scores
every quad, orientation and ring (`liblizard/gpu/wgsl/finder.mjs` `RING_COUNT` 4, 128 hypotheses: 4 quads x 8 orientations x 4
rings), PICK picks the ring, the border nodes are refitted against that ring's nodes, the word is read and the picture
sampled, and a back half on the device does the transform with the detrend, the soft values and the decline gate, the
LDPC and the CRC-32. The rate profile is one stage there (`liblizard/gpu/back/tiers.mjs`,
`liblizard/gpu/wgsl/back_tiers.mjs`): every version's tiers a row of a table, each block's soft values, estimate,
decline and pilot over its own sub-channels and code, a dispatch a code. The format's tables come from the wasm codec at start: the rings' borders (built at n = 256 in each
ring) and the pictures apart (`liblizard/gpu/tables.mjs` `formatTables()` returns `{ rings, pictures }`, the pictures from
`liblizard/sim/lizard_pick.mjs` `PICTURE_SIZES`; a frame's sample grid is its (ring, picture) pair's, `gridOf`). A device with
32 KB of workgroup memory and no `shader-f16` builds no 1536 picture (pass 1 needs 33,792 B in f32), as it built no
2048 before.

The word stage, F8 (`liblizard/gpu/wgsl/word.mjs`), bootstraps as the C does (5.5): one code a ring, read over every version 1 to
128, sets the frame's selection to (ring, 1 + version, picture slot); with no word taken, the batch's held version (the
last word the caller read, `decoder.mjs` `run(..., { held })`, a held uniform a lane); with none held, the frame is
dropped. Its acceptance distance is set in bits by W, 6 of 64, 24 of 128, 36 of 192 and 48 of 256 in the 32, 64, 96
and 128 rings, with union bounds on a garbage read of 1.5e-7, 7.5e-9, 8.6e-15 and 1.1e-20.

It is judged only on CRC-verified blocks, on throughput and on a negative control (captures with no symbol and empty
batch slots must give nothing), never on parity with the C at any stage. On the four rings (2026-09-27 evening,
synthetic frames, no camera, an RTX 4090): every (ring, picture) pair read told nothing, 24 of 24 at
each picture's top and 48 of 48 over the twelve formats of 10.1 (`scripts/exp/gpu_rings.mjs`), a wordless frame read at the held
word and not without one, 5 of 5; `scripts/exp/gpu_front.mjs`, 28 cells, found the right ring and picture on every symbol frame,
the no-symbol controls 0 blocks and 0 words, 0 wrong, 0 bad. On the desktop's two-CU RDNA-2 iGPU, on the morning's rings
(f16, 48 synthetic 1080 frames, LIZARD-192 in the 64 ring, then at n = 1024, batches of 32), it took 5.67 ms of compute
a frame, 0.107 ms of it the word, and verified 1,152 of 1,152 blocks, bad 0; at n = 768, 5.53 ms (section 11). Before
the rings, on every frame of the 19 recorded phone runs (5,942 frames, the 4090) it verified 93,124 blocks against the
C's 84,116, 0 bad on either, with the classifier weights installed on the afternoon of 2026-09-24 (STATUS.md,
2026-09-24), and it decoded live on the S26 Ultra from 2026-09-25. On the one recording on the rings (v0.3
07-56-38, LIZARD-336 at n = 1024 in the 64 ring, 600 frames, the 4090) it verified 20,133 blocks, bad 0, every frame
found in its ring and every word right, with the shift stage it ran until 2026-09-28; it has not run on a phone on the
rings. `liblizard/gpu/README.md` has the shape rules, the stages and how to run it.

## 11. Measured facts (non-normative)

The simulator (`scripts/sim/camera.mjs`, one camera model) is for ranking configurations. The phone figures are one phone
(Samsung S26 Ultra, Firefox for Android) filming one ordinary 1080p office monitor (an OSU lab display, not a gaming panel), and where they disagree with the simulator the phone wins;
the simulator has been badly wrong once, on dithered black and white. A test cell is one capture condition of a
simulator sweep. Every figure is from STATUS.md unless its line says it was measured for this document.

| fact | conditions | kind |
|---|---|---|
| 270 KB/s sustained and not finicky: 48 fully decoded frames a second of 5,628 B | 96 sub-channels (12 blocks) at n = 256, which is not a current format (LIZARD-96 now takes n = 512); 720 x 720 crop; the 286-module border, before the bit map; reported, 2026-09-20 | phone |
| Decode 13.7 ms a frame; the phone is about 4x the desktop | the same configuration | phone |
| LIZARD-256: 281 KB/s (13,268 B a frame, 88% of frames found) at a 2160 crop; 269 KB/s (9,955 B, 96%) at 1440 | the 286-module border; 5 to 6 s of stats from an uncontrolled session; 2026-09-23 | phone |
| LIZARD-16 at about 21 camera px a module: about 21 KB/s, against its 22.5 KB/s ceiling at 24 fps | 2160 crop, the 94-module border, 2026-09-23 | phone |
| LIZARD-512: the C decoder told the format finds 600 of 600 frames and verifies 42.8 of 64 blocks a frame (20,081 useful B, 67% of 30,016), 0 bad | recording 2026-09-23T21-34-51: 1440 crop, sender at 24 fps, painted on the n / 4 + 30 border (286 modules); replayed on desktop wasm, decision-directed pilots off; measured for this document | phone recording |
| The GPU decoder verifies 93,124 blocks, the C 84,116, 0 bad on either | all 19 recordings (5,942 frames), every one on a white surround and on a border older than n / 8 + 60; an RTX 4090; the classifier weights installed on the afternoon of 2026-09-24 (93,178 with the weights before them) | phone recording |
| Dithered black and white: about 20 KB/s where grey did 270; the simulator had put it at 80 to 100% of grey | 2026-09-20 | phone |
| The n / 8 + 60 border (2026-09-23 to 2026-09-27): reported to work; 14 recordings of it followed (v0.2, 2026-09-27) | 2026-09-23 | phone |
| The morning's three rings (32, 48, 64) against the n / 8 + 60 border, told: blocks -0.8% at LIZARD-16, -1.1% at -64 and -128, -6.8% at -256, -3.7% in all (229,108 to 220,724); blind equal to told in bytes in every cell. Under damage (`scripts/exp/fmt_word.mjs DAMAGE=1`, 2,048 frames) payload frames 1,185 to 1,028, the loss at n = 1024 and 2048 in cells set in pixels a module, where the new symbol has 158 modules against 188 and 316; wrong words 0 in both | `scripts/exp/border_scale.mjs FULL=1`, 16 frames a cell, old build against new, 2026-09-27; not repeated on the four rings | simulator |
| A decision-directed second pass recovered blocks in every recording whose shift exceeded 0.13 samples, and nothing below | ten recordings; the pass and its gate (`DD_GATE`) deleted 2026-09-28 (section 12) | phone recording |
| Contrast floor: no loss from 100% down to 60% of luma range, the first at 50%, collapse by 30%; the C's frame finder failed, not the payload | neutral squeezes, 7 test cells x 8 frames, 2026-09-20 | simulator |
| Narrow luma: neither decoder shows that cliff now. At 30% of range (luma 89 to 166) the C reads 2,308 blocks and the GPU 2,382, against 2,498 and 2,557 at full range; the loss follows the SNR-limited test cells (the phone test cell 84 to 31, 1.5 px defocus 90 to 62) and the easy test cells stay whole | 17 test cells with a symbol x 6 frames, both no-symbol controls at 0; a luminance-only model, no Bayer or demosaic; 2026-09-24 | simulator |
| Palette rule: a luma difference of at least 60% of full range (153 of 255 under the 77/150/29 weights) and R, G, B swings within about 15% of each other; the brand pairs measured cost about 2.1 dB and nothing measurable except about 12% at fill 0.4 | 7 test cells, 2026-09-20; the model has no chroma, so a pair with a large channel spread is not covered | simulator |
| Noise painted up to the ring with the margin cut away: LIZARD-256 kept 64 to 388 of 1,446 blocks, LIZARD-64 214 to 496 of 614 | `archive/build-scratch-2026-09/ring_race.mjs`, 18 test cells x 6 frames, three kinds of noise, the n / 4 + 30 border, 2026-09-23 | simulator |
| The guard ring (1 module of black round 3 of white) with the noise painted up to it: LIZARD-256 1,224 to 1,345 of 1,446; not recovered: what only the line fallback carried (270 px and below, 45 degrees of yaw) and 2.5 px of defocus | the same harness | simulator |
| A 2048 picture stops dead where a 1024 one degrades: at 594 camera px and 1.5 px of defocus LIZARD-576 and -1024 register nothing, LIZARD-560 reads 3,752 B a frame | the n / 4 + 30 border (542 modules), 6 frames a test cell; not re-run on 316 modules, on the rings (158) or at 1536, the top picture since 2026-09-27 | simulator |
| Decode by format: LIZARD-256 22.4 ms, -568 (71 blocks, off the ladder) 19.6, -576 40.1, -1024 39.4; heap 20 MB through n = 1024, 38 to 41 MB at 2048 | simulated 2160 capture, clean pose, told, desktop wasm on one thread, 3 frames, the n / 4 + 30 border | simulator |
| The picture sizes between the powers of two, each format at its 3 x 2^k size against the power of two above, same ring and pose: 29,461 blocks against 29,714 over eight formats (-0.85%), per format -0.1% (LIZARD-144, 2.24) to -1.4% (LIZARD-80, 1.50); every frame registered in both arms; the C's sampler and transform 0.62 to 0.69 of their time, its whole decode 0.77 to 0.89 | `FULL=1 scripts/exp/focus_nfit.mjs`, 1,536 captures: one camera model, clean poses, told, one physical size, 16 frames a cell, 6 cells (0.8, 0.9, 1.0 and 1.25 of `CAMERA_PX_FOR`, 2.5 with 1.5 px of defocus, 1.25 through the page's smoothing), 2160 px captures (1080 for LIZARD-48 and -80); times from one worker among 30, read as ratios; 2026-09-27 | simulator |
| The GPU decoder at those sizes: LIZARD-192 at 1080, 5.53 ms of compute a frame at 768 against 5.68 at 1024 (pass 1 0.545 against 0.744), 1,152 of 1,152 blocks in both; LIZARD-1024 at 2160, 16.25 at 1536 against 18.21 at 2048 (pass 1 1.85 against 3.78), 6,024 against 6,035 of 6,144 blocks; bad 0 | the desktop's two-CU RDNA-2 iGPU, f16, 48 synthetic frames a cell, one cell each, batch auto, two rounds interleaved (means), the cancel stage off; tree copies with `PICTURE_SIZES` edited; 2026-09-27 | desktop |
| Encode: LIZARD-1024 takes 17.5 ms in C (`focus_tx`) and 19.2 ms with the RGBA, so a 60 fps sender misses its frame by about 3 ms | node on the desktop, the n / 8 + 60 border | desktop |
| Rate: 7/8 everywhere is +10% where the capture matches the sender's assumption and -42% at 360 px | ten test cells (`liblizard/src/focus.h`, `FOCUS_RATE`) | simulator |
| The rate profile against one rate 3/4 at the same sub-channel count: +17.5% bytes a frame read at LIZARD-432 under the profile (7/8 x 20, 3/4 x 20, 1/2 x 11), +25% at LIZARD-416 under one of its shape (7/8 x 16, 3/4 x 20, 1/2 x 12), though they offer 5.6 and 7.7% fewer blocks; 7/8 everywhere 33 to 36% less | `scripts/exp/rate_tiers.mjs`: two recorded 2:1 captures of one phone (S26 Ultra at 1080, zoom 1.5, 960 x 960 halves, 236 and 299 frames whose bytes are known), each frame's residuals re-modulated under each profile and read by the C decoder's own soft values and LDPC (the one-rate profile through this path reads the real decode's bytes exactly); one phone, one distance; 2026-10-07 | phone recording |
| Bit map: whitening took a frame of 100-byte payloads from 6 of 64 blocks to 64; at 8 px of motion blur LINEAR reads 870 B a frame against 674 in order (LIZARD-64) and 791 against 557 (LIZARD-128) | 48 frames (`liblizard/src/focus.h`, `FOCUS_BITMAP`) | simulator |
| n / 8 + 60 against n / 4 + 30: useful bytes summed over 17 test cells -1% (LIZARD-16) to -13% (LIZARD-560), the losses where the picture is short of pixels; more frames registered at the far end and at 45 degrees | 6 frames a test cell, told receiver | simulator |
| A file in chunks through the rig's pages, the header from the light: 98.3% of the one fountain's rate at 10 MB and 98.7% at 600 KB (the control blocks are 1.6% of the airtime); 64 MB, past what one fountain carries, in 16 chunks at 698.3 KB/s, 97.0% of the 720.1 KB/s the display painted; the receiver about 40 MB with the blocks in OPFS, 230 MB with them in memory | `lizard-web/check_rates.mjs`: a clean synthetic clip, display 24 fps, camera 30, headless desktop Chrome, the header kept off the network, the file held to BLAKE3 by two implementations; LIZARD-96 at 1280 x 720, LIZARD-512 at 2560 x 1440; no phone; OPFS tried in Chrome only; 2026-09-24 | desktop |

## 12. Settled and rejected (non-normative)

Each of these was built and measured, or decided. The reason given is the one on record.

**Settled**

| decision | reason | record |
|---|---|---|
| The ladder is every multiple of 16 from LIZARD-16 to LIZARD-1024: 64 formats, version = blocks a frame, versions 2 to 128; the names stay under the rings (every whole number of blocks since 2026-10-01, next row) | a symbol a computer paints can afford more versions than QR's 40 (2026-09-23); it replaced the five formats of 2026-09-21; LIZARD-k already states the sub-channel count (2026-09-27) | STATUS "64 formats" |
| Every whole number of blocks, 1 to 128, a slider on the sender in place of the menu of 64 | one version (8 sub-channels) is the smallest step the word names; the blocks a frame are the rate profile's since 2026-10-07 | 2026-10-01; `liblizard/sim/lizard_pick.mjs:VERSIONS`, `lizard-web/send.html` |
| The largest picture (LIZARD-576 to -1024; n = 1536 since 2026-09-27, 2048 before) is for fixed rigs; apps are advised to stop at LIZARD-560; the library does not cap | a 2048 picture doubles decode and heap and stops dead under blur where 1024 degrades (both measured on the 542-module border, section 11; neither measured at 1536); the library's picker takes the largest format a room holds, so in the default ring any room of 1,170.6 device px or more gets the largest picture | 2026-09-23; `liblizard/sim/lizard_pick.mjs:ROOM_FOR` |
| The picture size is the first of 256, 384, 512, 768, 1024 and 1536 with n >= 3R: 2^k and 3 x 2^k, each at most 1.5 times the one before; a 3 x 2^k transform takes one radix-3 stage, and every power of two stays radix 2 | on the powers of two each step gave a format four times the samples for one more block (LIZARD-144 at n / 2R = 2.99); at the smaller sizes the simulator read 0.85% fewer blocks, the C's sampler and transform took 0.62 to 0.69 of their time, and the iGPU 2.6 and 10.8% less compute a frame (section 11); no format moved a coefficient or its ring | 2026-09-27; STATUS "Picture sizes between the powers of two"; 3.2 |
| Four rings: the band holds B = 32, 64, 96 or 128 cells of 2 x 2 modules a side, a side is 2B + 30 modules (94, 158, 222, 286), the picture spans 2B; any ring may carry any picture | the ring bootstraps what is inside, and versions collapse (the rows below) | 2026-09-27 (three rings that morning, four that evening); STATUS "Three rings: the ring bootstraps what is inside, versions collapse" |
| The sender's default ring is the 64 for every picture (the 128 since 2026-10-01, next row); no ring follows the version, even by default | unlike QR, whose size is fixed by its version, Lizard decouples the two, closer to 5G than to QR or Aztec: the ring is a sync and bootstrap layer, like 5G's sync block, chosen for the channel and the room; a rule that picks it from the room waits on a measured module-size floor | 2026-09-27 evening; `liblizard/src/focus.h:FOCUS_RING_DEFAULT`, `liblizard/sim/lizard_pick.mjs:RING_DEFAULT` |
| The sender's default ring is the 128 for every picture | the best phone runs (1,570 to 1,640 KB/s at LIZARD-480) were painted in the 128 ring (the page's `ring=3`) | 2026-10-01; `liblizard/src/focus.h:FOCUS_RING_DEFAULT`, `liblizard/sim/lizard_pick.mjs:RING_DEFAULT` |
| Versions collapse: the word's second byte, sub-channels / 8, names what is inside the ring and nothing about the ring; n follows from it, and nothing checks it against the ring | the ring only locates the symbol and syncs its grid, and the word it carries says what is inside it | 2026-09-27; `liblizard/src/wasm.c:focus_any_rx` |
| The ring bootstraps; a failed word is decoded at the last word read, and before any word nothing is decoded | the ring bootstraps the configuration, and a later word that fails to decode leaves it as it was | 2026-09-27; 5.5 |
| No retry at the other rings when a word fails | by the time the word fails, the rest of the frame has long failed: on the recordings before the rings no frame of 10,656 registered ones read a block without its word; a wrong ring registered on 90 of 2,048 damaged simulated frames, 82 under glare over three sides where nothing reads | 2026-09-27; 5.5 |
| The ring at a whole number of pixels a module, pxm = ceil(n / span); the picture keeps its n samples, resampled (Lanczos-3, periodic) to fill its 2B modules, with no dead light between border and picture | a whole 8 samples a module left 15 modules of dead white (judged broken); in the 32 and 64 rings, at 384, 768 and 1536 in the 96 and at every n but 384 in the 128, the resampler copies (6.8) | STATUS 2026-09-23 |
| The border grows outwards only | the picture is an inverse FFT, so a hole in it costs about 8x its own area in payload | `liblizard/src/focus.c:init` |
| The symbol is self-describing through the format word | a receiver told nothing reads it; the version as an anchor count was built and lost (fill 0.3: 2,521 B to 381) and cannot carry error correction | `liblizard/src/fmt.h` |
| The word carries the display rate | the one property of an animated symbol that no single frame shows | `liblizard/src/fmt.h` |
| The margin, 2 modules of light, is part of the symbol, and the codec paints it (`FOCUS_QUIET`) | the finder assumes light outside the ring: noise up to the ring cost LIZARD-256 73 to 96% of its blocks; in the codec, every encoder built from `liblizard/src/` paints it and no JS keeps a copy | 2026-09-23 and 2026-09-24; `liblizard/src/focus.h` |
| Only ladder versions are painted; a receiver reads any version the word states, with no flag; the codec refuses a symbol past LIZARD-1024 | an odd version is unsupported; past version 128 the word cannot name the symbol | 2026-09-24; `liblizard/sim/lizard_pick.mjs:VERSIONS`, `liblizard/src/focus.c:init` |
| H is the printed shift table, `liblizard/src/ldpc_base.h`, not a construction (the 7/8 and 1/2 codes' too since 2026-10-07) | with H generated at start-up, a change to the profile, the seed or the cycle score would have changed the format without warning | 2026-09-24; 8.2, 8.6 |
| The transfer is in the light: the block id as 14 bits of chunk over 18 of fountain symbol, a header block, manifest blocks, and chunks of 2^10 to 2^24 bytes each verified by BLAKE3 against the file's root | a camera alone recovers a file; a chunk aligned at its own size is a subtree of BLAKE3's tree, so each is checked on arrival, as Bao and iroh stream; one fountain stops at 30 MB | 2026-09-24; `liblizard/src/xfer.h` |
| The rig's sender interleaves every chunk over the whole file, laps with fresh repair ids, not chunks in turn | in turn needs a budget per chunk a lap, since nothing says a chunk arrived: counted over 64 MB at LIZARD-512, 15% random loss took 1.18 laps interleaved against 1.25 to 2.18 in turn, and nothing lost 1.001 against 1.10 to 1.49 | STATUS 2026-09-24; a count, not a camera |
| No guard ring: the symbol ends at its 2-module margin | the ring round the margin was a sender option (2026-09-23) that recovered part of what noise at the ring cost, not the line fallback's cells; judged useless | 2026-09-26 |
| Luminance only: grey levels, never colour; two brand colours are allowed as a luma ramp under the palette rule | colour is what made cimbar unreliable | project rules |
| The code rate follows the frequency: the rate profile, 7/8 inner, 3/4 between, 1/2 outer, a function of the sub-channel count (3.1) | a capture's signal-to-noise ratio falls with the frequency, and on recorded phone captures re-modulated under each profile it read +17.5 and +25% bytes a frame over one rate (section 11). One rate 3/4 was the format before 2026-10-07: the rate a bet on a capture the sender cannot see, after profiles set on the simulator lost 4 to 16% where the capture did not match them; the recorded channel's own noise settled it the other way | 2026-10-07; `liblizard/src/focus.c:focus_tiers_for` |
| The bit map is LINEAR, whitened, and the only one, so the word names none | whitening fixes small payloads; LINEAR spreads every run of the code across the block; the word has no free bit for a mode, and the other modes are lab settings | `liblizard/src/focus.h:FOCUS_BITMAP`; 2026-09-24 |
| The format is never gated on a decoder | a weak decoder is fixed; it never decides or reverts a format choice | 2026-09-23; section 2 |
| Lizard over the binary grid code | not throughput (the binary code is ahead above its cliff): Lizard degrades where the binary code stops dead; no further work on the binary code, which `scripts/exp/` still reaches through `liblizard/sim/phy.mjs`; the rig has no path for it since 2026-09-26 (its receiver was told by the server) | 2026-09-20 and 2026-09-26 |
| The receiver is blind and the sender stands alone: a page talks to a server only to send it development logs, and nothing comes back into decoding or painting | Lizard's format, geometry and file header come from the light, the test stream's bad blocks are judged from the light (9.3), and Aztec and QR describe themselves | 2026-09-26; STATUS "The whole receiver blind" |
| QR and Aztec are archived, not baselines: Lizard is compared against other apps | the research had settled that a beefier QR is not the option; the baselines dated from when a new format was in doubt, and FOCUS showed that OFDM, in luma only, does better and accelerates well on a GPU | 2026-09-26; `archive/qr-aztec/README.md` |
| Wirehair and zstd are fixed and out of scope (LZMA on paper until 2026-10-05, never applied; zstd chosen over it and GDeflate: ratio within 5 to 10% of xz at level 19 at compressors a phone keeps ahead of the channel with, where GDeflate keeps DEFLATE's ratio for a GPU decode speed the channel cannot use) | the scope is the 2D code | project rules; STATUS "zstd, a chunk at a time" |
| No SharedArrayBuffer, so no wasm threads | workers cannot share a heap, and per-worker memory is the number that matters on a phone | project rules |

**Rejected**

| rejected | reason | record |
|---|---|---|
| Three rings, B = 32, 48, 64, the sender's default following n (32 for 256, 48 for 384 and 512, 64 above), and the top ring kept at 64, the morning of 2026-09-27 | the top ring was kept at 64 after the A/Bs that put the n = 1024 and 2048 pictures on 158 modules against 188 and 316 lost 6.8% at LIZARD-256 (`scripts/exp/border_scale.mjs`, simulator, told) and payload frames only at those sizes under damage (LIZARD-256 312 to 269, -576 354 to 240); that evening the rings became 32, 64, 128, then four, and the default the 64 for every picture | 2026-09-27 |
| Middle marks on the larger rings | looked at and set aside when the rings became four | 2026-09-27 evening |
| The even border: a 13-module mark with a 2-module inner edge, a 2-module guard, ring breaks of 2, light depths 7 to 12, every feature but the rim whole 2 x 2 cells | built and reverted the same day; the code paints the 12-module mark with 1-module inner edge and breaks and the 3-module guard (section 4); the rim stays either way | 2026-09-27 |
| A side that follows the picture, n / 8 + 60 modules (92, 124, 188, 316), from 2026-09-23 | replaced by the rings: the ring bootstraps what is inside, and versions collapse. It had replaced borders whose ring was too thin and which shrank to almost nothing beside the picture (92 = 28 modules of corner clearance + 32 band cells of 2) | 2026-09-27 |
| The receiver's size check: a word whose version implied another picture size than the module count registered was dropped, and the frame registered again at the other counts | the module count no longer says anything about the picture | 2026-09-27 |
| Straddle cancellation, out for good | too complicated for its gain, on one phone's evidence; restored as a receiver menu, it spiked the phone's lag (2026-09-23) | `archive/straddle-cancel/README.md` |
| Dithered black and white as the default | about 20 KB/s on the phone against 270 for grey; a sender option with no further work | STATUS 2026-09-20 |
| Dithered black and white Lizard, the sender option, deleted | it ruins the range and the pixel scaling; it was also decoded told at gamma 2.2, which no blind receiver is | 2026-09-26 |
| Reed-Solomon in place of the LDPC | 70 to 81% of the LDPC's payload on the mean, since it cannot use grey values; RS stays for small fixed fields such as the word | `research/08` |
| 16-QAM | the SNR floor is about 10 dB of distortion | `research/10` |
| A sender-side power tilt as the default | the +12 and +15% on the mean came from profiles that joined a 9 dB tilt to 320 sub-channels in place of 192, or to rate tiers, and every profile that added payload lost 4 to 16% in the 720, phone and fill 0.4 test cells; the tilt stays an option | `research/10` |
| QR's three finder squares in place of the corner marks | they lost the finder race | STATUS 2026-09-21 |
| A filled corner mark | it moves the edge the line fit reads; half the payload lost at 720 | STATUS 2026-09-20 |
| Edge marks | a wash at 12 modules, a loss at 16 from the wider border | STATUS 2026-09-20 |
| A centre mark | wherever it goes it is in the picture | STATUS 2026-09-20 |
| A 4 x 4 border cell | blur limits the mesh's sampling, not the band, and a coarser band made the mesh worse | STATUS 2026-09-21 |
| Bursts as the way to run a stream, and then the receiver option itself | with a fountain, more distinct frames beat better frames; the rig never used it, and the codec's option (summed soft values of several captures of one frame) was deleted on 2026-09-28 | `research/07`; 2026-09-28; `archive/bursts/` |
| The decision-directed shift pass: a frame's failed blocks read again through the registration shift its verified blocks' codewords give, in both decoders, and the shift carried between frames | an opaque filter patching a problem, where what had helped was the basic design, the rings among it; the loss is accepted until the registration bias it corrected is fixed at its source: on the two recordings on the rings (v0.3 07-56-38 and 17-59-29, 600 frames each) the blind C reads 19,960 and 22,266 blocks without it against 20,256 and 23,934 with it (-4.4%), 0 bad either way, 11.2 ms a frame against 12.0 (`scripts/exp/capture_check.mjs ARM=blind`, this desktop's CPU, one thread, both runs beside the GPU jobs). The carry had gone from the receiver on 2026-09-26 (the blind rule) | 2026-09-28; `archive/shift-pass/` |
| A faster CPU transform (radix 4, radix 8) | radix 2 is the better all-rounder on the CPU (2026-09-23); better FFTs belong to a GPU decoder; the kernels are in `archive/stash/fft-gen2/`. The one radix-3 stage of a 3 x 2^k size (2026-09-27) is a size, not a faster transform: every power of two stays radix 2 | STATUS 2026-09-23 |
| The first WebGPU decoder, a port of the C | the wrong shape: it copied the CPU's control flow; deleted 2026-09-23 | STATUS 2026-09-23 |
| Polar codes; the 5G NR base graphs and shift tables | patents (CRC-aided SC list decoding to 2035, NR shift tables to about 2036 to 2038); polar is also weaker than LDPC at 4k to 16k bits with soft input | `research/01` |
| The camera's exposure set from the pilots' leaks, and any exposure shorter than the camera's own (the Android app) | a shorter exposure takes the neighbours out of a capture and reads fewer blocks all the same: on the S26 at LIZARD-480, 8.33, 6, 4, 3 and 2 ms left 17, 15, 5, 4 and 4% of the neighbours and read 57.4, 55.8, 52.6, 50.9 and 47.2 blocks of 60 (10 s by hand, one screen). The rule drafted for it, its script (`lizard-android/tools/phone/exposure.sh`) and the app's `debug.lizard.exposure` switch are deleted (2026-10-01). Revised 2026-10-04: a longer exposure than the camera's 8.33 ms is what a dim screen gets from auto-exposure, and it widens the capture's window past the refresh at both ends (a 2:1 replay of a laptop's screen read 12.5% of each neighbour at the lock's balanced phase against 5 to 7% on a monitor, and half the blocks), two controls were built and removed the same day: the exposure held at half the frame once auto-exposure asked for more (Android 16's exposure-time priority, the sensitivity automatic), which read no better on that laptop's screen (no log), and the exposure learned from the blocks by trials (not run). The camera's exposure stays auto-exposure's; the stats rows carry it | STATUS "The lock fed sooner; the exposure from the leaks measured and dropped", `research/results/phase/phone_exposure_quick.txt`; STATUS "The 22:02 replay: the lock held; the capture window halved the blocks" |
| More pilots a block (2026-10-01) | the pilots measure the mix and do not remove it: a reading is 0.8% of a picture exact where blocks fall past a 15 to 20% share, the model's lock does not move at 2 and 4 times the pilots, and a second 32 a block would come out of the code's bits. The simulator's option for it (`phasesim.pilots`) is deleted | STATUS "More pilots would not fix the mix", `research/results/phase/model_pilots_x2.txt` |
| Soft values that take a capture's two neighbours as signals of known size (their shares by the pilots), not as noise (2026-10-01) | no gain even given the true shares, which a receiver would have to estimate: LIZARD-480 in the 128 ring through a simulated camera, 24 frames a cell, 15,107 blocks to 15,109 and 12,174 to 12,171 on mixes held on every row (the rule's best case), 12,214 to 12,213 on the rows model; no cell moved by more than 0.13 of a block a frame. Where a neighbour is small against the noise the two rules give the same numbers. The experimental build and its script are deleted | STATUS "Soft values that know the two leaks", `research/results/mix/` |
| The pilots as a meter on \|r\| alone (2026-09-30 night): a sweep folded for the display's period, a hold on the slope of \|r\| against phase, repeats guessed by a spike | \|r\| is the same whichever neighbour a capture leaks from, so the hold was blind near its top and slow to start; replaced by the signed pilots (the count mod 4), whose two readings say which neighbour, and deleted | STATUS "The pilots' phase meter", "Signed pilots" |
| A learned fine round for F7's nodes, and a learned side model (GPU decoder, 2026-09-29) | nodes 35% more accurate on neutral data and no more blocks (-1.1% there, -4.0% and +6.8% on the two recordings); the side model +20 to +46% on neutral and -2.4 to -7.9% on the recordings; thrown out | STATUS "The F7 node net", "The F7 side model" |
| The proposer's network on the Adreno's cooperative-matrix units, and an NPU port (Android app, 2026-09-30) | the units multiply int8 2.5 to 2.8 times as fast and the kernel ran 5.4 times slower (4.95 ms a frame against 0.92), around ten recorded compiler faults; the NPU judged too unstable | STATUS "Native nets on the matrix units", `archive/stash/coopmat-proposer/` |

## 13. Open questions

None is decided here. An item marked closed was decided and its answer moved into the normative text; it keeps its
number so references to the others hold.

**Where `liblizard/src/` and the decisions differ, or the code leaves a choice open**

1. **Versions off the ladder. Closed (2026-09-24): unsupported.** A sender never paints one, and a receiver
   reads one as it reads the ladder, with no flag (3.1, 5.1).
2. **A symbol with no word. Closed (2026-09-24): refused.** `liblizard/src/focus.c:init` refuses a symbol past
   LIZARD-1024, whose version the word cannot state (3.1). The wordless tiles of the 2 x 2 split are gone from the
   sender page (section 2).
3. **The margin is outside `liblizard/src/`. Closed (2026-09-24): it is the codec's.** `liblizard/src/focus.h:FOCUS_QUIET` = 2,
   painted by `focus_paint_rgba`, read by the JS from the wasm (2, item 5; 4.8).
4. **The block id is outside `liblizard/src/`. Closed (2026-09-24): it is the format's.** 14 bits of chunk over 18 of
   fountain symbol, with a control range for the header and manifest (7.5, `liblizard/src/xfer.h`).
5. **The bit map is not in the word. Closed (2026-09-24): LINEAR is the only bit map the format has**, so the
   word names none (5.1, 7.3). No bit of the word was free for one.
6. **Table or construction. Closed (2026-09-24): the table.** H is the shift table printed in 8.2, shipped as
   `liblizard/src/ldpc_base.h`; the generator builds only codes outside the format (8.3).
7. **Spare word cells. Closed (2026-09-24): there are none.** The word is one code over every word cell, a code
   a ring since 2026-09-27 (a picture size before), and every word cell is flagged `CELL_WORD` (5.2, 5.3).
8. **Clip ratio and tilt.** `liblizard/src/` treats both as sender arguments (`focus_init` `clip`, `tilt`) that no receiver
   needs, and every sender in the tree uses c = 2 and t = 0 (6.5, 6.7). No decision says whether they are format
   constants or free for a conforming sender.
9. **Bit-exactness. Decided (2026-09-24; revised 2026-09-29): the reference is exact, and a sender need not
   be.** The reference encoder (`liblizard/src/`, as `liblizard/build.sh` builds it) paints the same bytes on every machine, and its test
   vectors are its own regression check; the text's real arithmetic (6.6 to 6.8) describes the picture, it does not
   define its last bits. On 2026-09-24 only a symbol with the reference's bytes conformed, and the sender could use
   WebGL to paint the reference's pixels, never to encode or take the FFT. On 2026-09-29 that rule was replaced. A
   sender may encode anywhere, a GPU included, provided its symbol decodes: the border exactly as the format specifies
   it, the picture by the format's arithmetic, not the reference's last bits (section 2, Conformance). A receiver never
   checks. The sender's GPU encoder
   (`liblizard/gpu/encoder.mjs`, 2026-09-29) paints the reference's own border bytes and computes the picture in float32 on the
   device: in the 33 cases of `scripts/gpu/encoder_check.mjs` (every ring with each picture's top, each picture's lowest
   ladder version, two codes side by side, frames encoded in batches), 508 of 52,562,612 pixels one grey level off the
   reference's on SwiftShader and 456 on an NVIDIA 4090, none by more and none in the border, and every block, 1,748
   in 39 symbols, read back by the C decoder told nothing (STATUS.md, "The sender encodes on the GPU"). `fft_set_radix(4)` is a lab setting.
10. **Two brand colours.** Allowed by the palette rule (2, item 1), but neither `liblizard/src/` nor the sender paints them, and
    no code defines the grey-to-colour mapping between the two ends (linear in 8-bit values, or in light).
    `scripts/exp/focus_colour.mjs` and `scripts/gpu/harness/scenes.mjs` `range` simulate only a narrower luma range.
11. **The palette rule's 60% floor** came from a 2026-09-20 finder cliff that neither decoder shows now (the simulator
    reads a 30% range, 2026-09-24, section 11). Whether 60% still binds is not measured on a phone or on a model with
    chroma.
12. **A sender page option that breaks the format. Closed (2026-09-26): removed.** FOCUS's finder
    frame ("finders and margin", thin = 0) is gone from `liblizard/src/`, `liblizard/sim/ob.mjs` and `liblizard/sim/phy.mjs`: `focus_init`,
    `focus_init_tiers` and the wasm's `focus_setup` and `focus_setup_tiers` take no `thin`, and a harness spec that
    names no frame gets the format's border. The page's other such option, the 2 x 2 picture split, was removed
    (2026-09-24; section 2): its tiles were off the ladder, or at n = 128 with no word.
13. **The fountain's header has no place in light. Closed (2026-09-24): the header block.** Its layout is 7.7,
    with the manifest (7.8) and the chunk tree (7.9); the rig's network copy is only a fallback (9.4).
14. **Message sizes Wirehair refuses. Closed with 13:** a file is cut into chunks of at most 16 MiB, well inside
    Wirehair's 64,000 blocks, and a chunk of one block is sent as it is, not fountained (7.6). An empty file is a header
    alone.
15. **The display resamples again.** The sender page stretches the painted image to the room (a CSS scale of the
    canvas, `lizard-web/send.mjs:layout`), so on a real screen the picture is resampled twice, once by 6.8 and once by the
    page. The page's stretch is smoothed (`image-rendering: auto`, 3.5), while the record's last word on smoothing
    (`scripts/exp/display_bilinear.mjs`: bilinear at scale 2.5 kept 38 to 88% of the payload, nearest 100%) says a page must
    not smooth, and `research/09` said the opposite. Only the simulator's nearest-neighbour display has seen the
    double resample (STATUS.md, 2026-09-23).
16. **Patent-clean ingredients.** The 2026-09-22 spec listed them as a constraint (`research/01-prior-art.md`,
    section 8). The project's rules do not, so section 2 leaves it out. Answered on 2026-09-26, with a library to
    publish in view: FOCUS, the only new technique the format uses, is not patented, and neither are
    Wirehair and LDPC. The code's own LDPC base matrix (`liblizard/src/ldpc_base.h`, generated from seed 1 and frozen as a
    table) is not a standard's; the 5G NR base graphs and shift tables were rejected for their patents (section 12).
    BLAKE3 is public domain or Apache 2.0, and the format word's Reed-Solomon code long out of patent.
17. **Stale text.**
    - `liblizard/src/acquire.c`, above `mark_form_mid`, gives the gapped core's middle as (S + 3) / 2 and the merged square's as
      S / 2 in from the corner. For the 12-module mark the first is 7.5, where the code gives (5 + 12 - 1) / 2 = 8.0,
      the core's true centre; the second is 6.0, as the code gives; 4.5 follows the code.
    - `liblizard/src/layout.h` (the thin-frame comment) and the 2026-09-22 spec say no run in the band is longer than 4 modules.
      That holds for the track only (4.6).
    - `liblizard/sim/lizard_pick.mjs` comments still give n = 2048 as 542 modules and old room figures (LIZARD-16 needs 226 px;
      camera 178, 299, 423); `scripts/exp/pick_check.mjs` says LIZARD-1024 needs 1305 px (1560.8 in the 64 ring).
    - STATUS.md's "64 formats" table (94, 158, 286, 542 modules; rooms 226 to 1305 px) and its never-picked list (128,
      528, 544, 560) predate both later borders. In the default ring every format is picked by some room, and the
      largest picture (1536 since 2026-09-27, 2048 before) starts at 1170.6 px of room (1034.8 on the n / 8 + 60
      border, 979 before).
    - `liblizard/test/fmt_test.c` tests the four rings' codes since 2026-10-04 (5.2's seven words; all passed, its exact
      garbage rates 5.4's). 5.4's measured row still comes from the scratch harnesses of 2026-09-27; `build/fmt_test
      2000000` remakes it (about 10 minutes).
18. **Test vectors outside the repository.** `liblizard/test/vectors.mjs` checks 5.2, 6.12, 7.4 and 9.3 against a build
    (in the repository since 2026-10-04). The scripts that computed the vectors in 4.7, 4.9, 5.3 and 7.10, and the
    independent implementations that matched them, are outside the repository, not in `scripts/exp/`. Those vectors can be
    recomputed from `liblizard/build/ob.wasm` as each section describes; whether the scripts belong in `scripts/exp/` is not
    decided.
19. **The decoder text may move.** 8.5 describes `liblizard/src/ldpc.c` as of 2026-09-22 and `liblizard/src/focus.c` as of 2026-09-23. The
    speed round's LDPC early stop has not landed; if it does, the stopping rule in 8.5 changes. The format does not.

**Not yet measured**

20. The rings on a phone. One recording exists, in the 64 ring (`research/captures/v0.3` 07-56-38, LIZARD-336 at
    n = 1024, 600 frames at a 1080 crop, the S26 Ultra; 10.1, 10.2), none in the 32, 96 or 128 ring. The recordings of
    the borders before (v0.1, 19 runs and more to 2026-09-26; v0.2, 14 runs on n / 8 + 60, 2026-09-27) do not register
    on today's decoders; the build before (`archive/before-copies/rings/before/`) still reads them. Nor a transfer at speed, or heat.
21. The largest pictures under blur in the 64 ring (158 modules): 1536 since 2026-09-27, 2048 before. The dead stop
    was measured at 2048 on 542 modules, and whether the border or the decoder caused it was not separated. The advice
    to cap apps at LIZARD-560 (3.5) rests on it. In the simulator the rings cost payload frames under damage only at
    n = 1024 and 2048 (section 11).
22. The temporal model: display rate against capture rate, the rolling-shutter blend, and how often phones other than
    the S26 capture a frame change (a capture straddling two frames reads nothing, since every block spans the whole
    picture's spectrum). Measured on the S26 against a 60.00 Hz monitor on 2026-09-30 (STATUS "The phase lock on the
    S26, and track"): at 60 captures and 60 painted a second, 8 ms of the 16.7 ms refresh read at 8.33 ms of
    exposure, and the Android receiver holds the camera there by delaying frames (its two arms, `auto` and `track`,
    both kept). A per-frame reading of the phase from the light: the pilots (7.3, 9.1), built 2026-09-30 (STATUS "The
    pilots"), signed 2026-10-01 so a capture reads which neighbour it holds (STATUS "Signed
    pilots"); the Android receiver's `track` holds by them. Open: the two readings on a real display (the first night's
    single reading read 0.76 at the best phase of this desk's monitor, about 6% of each neighbour; the simulator's
    model of the leaks is arithmetic).
23. The GPU decoder on a phone on the rings. It ran on the S26 Ultra on the n / 8 + 60 border from 2026-09-25 (10.2).
24. A recording with a noisy surround. Every recording has a white one; the sender's surround menu (`lizard-web/send.html`,
    `?bg=`) exists for it.
25. The C decoder's block total on the 19 recordings depends on the harness: `scripts/exp/capture_check.mjs` reads 81,390
    (0 bad, 2026-09-24) where the C column of `scripts/exp/gpu_captures.mjs` reads 84,116. The settings differ and were not
    reconciled. Section 11 cites 84,116 beside the GPU's 93,124, as STATUS.md does.
26. The transfer on a phone. Every figure for it is desktop Chrome on a clean synthetic clip (section 11): not a real
    camera's loss against the interleaved schedule, OPFS in Firefox or Safari, or a phone's BLAKE3 and Wirehair
    solve times. The sender holds every chunk's encoder, about 1.75 times the file in Wirehair's heap, which puts its
    limit at a few hundred MB; not measured on a phone either.
27. The rate profile beyond one phone at 1080. Its split (about 32% of the sub-channels at 7/8 inside, 30% at 1/2
    outside) was set on two recorded captures of one phone at one distance and zoom (section 11). At 720p or at range
    the outer sub-channels read worse and the knee moves inward; a profile set for one capture lost 4 to 16% where the
    capture did not match it on the simulator (research/10). A second phone and a capture at range are owed.

## 14. Changes since the 2026-09-22 spec

The 2026-09-22 text is kept at `archive/stash/spec-2026-09-22/SPEC.md`.

- **The rate profile** (2026-10-07). Every block was coded at rate 3/4 on 8 sub-channels, and a frame of version V
  carried V blocks. The rate now follows the frequency: 7/8 on 7 sub-channels a block in the inner tier, 3/4 on 8 in the
  middle, 1/2 on 12 in the outer, by the sub-channel count alone (3.1), so a frame carries B(V) blocks: V up to LIZARD-80,
  at most 7 fewer above (121 at LIZARD-1024). The word is unchanged: the version names the size, and the profile follows
  from it. The 7/8 and 1/2 codes are printed tables as the 3/4 code is (8.6). The pilots ride in the 7/8 and 3/4
  blocks' tails; a 1/2 block has none (7.3). A one-rate frame of LIZARD-48 or more, painted before, reads as the profile
  today and decodes nothing; LIZARD-8 to -40 are unchanged.
- **Every whole number of blocks** (2026-10-01). The 64 formats by 16 became 128 by 8, versions 1 to 128, all of which the word
  already named and every receiver already read; the reference sender sets blocks a frame on a slider (3.1, 5.1). Tables
  below that list 64 formats list the ladder as it was.
- **The default ring is the 128** (2026-10-01). The reference sender, the web's and the Android app's, paints every
  picture in the 128 ring unless one is named; a receiver reads any ring as before, and nothing in a symbol changed.
  Worked examples below that say "the 64" or
  "the default ring, the 64" were computed in the 64 ring and stay true of it; 6.12's vectors are the 64 ring's, span
  128 named. The room a format needs falls in the 128 ring (more modules a side, so fewer device pixels a module), so
  the sender's auto picks a larger format for the same room (3.5).
- **Ladder.** Five formats (LIZARD-16, -64, -128, -192, -256) became 64: every multiple of 16 from LIZARD-16 to
  LIZARD-1024 (3.1, 3.6). The word's version range went from 1 to 63 to 1 to 128 (5.1). The picture size cap went from
  1024 to 2048; n = 2048 carries LIZARD-576 to -1024 and is for fixed rigs, apps advised to stop at LIZARD-560
  (3.2, 3.5). The top picture is 1536 since 2026-09-27 (below).
- **Picture sizes** (2026-09-27). n was the smallest
  power of two from 256 to 2048 with n >= 3R. It is now the first of 256, 384, 512, 768, 1024 and 1536 with n >= 3R,
  and 2048 is no ladder size. 44 of the 64 formats moved n; none moved its ring, its coefficients or its word, and no
  room or camera figure moved. On the ladder n / 2R runs from 1.50 to 2.24 (1.52 to 2.99 before). A 3 x 2^k transform
  takes one radix-3 stage, the powers of two stay radix 2 (6.6). That morning's 48 ring copied 384, 768 and 1536.
  6.12's P3 moved to n = 384. The receivers' largest picture is 1536 (10.1, 10.2). Sections 3.2 to 3.6, 6.2, 6.3, 6.9
  and 11 carry the new numbers.
- **Versions off the ladder** (2026-09-24). An odd version is unsupported: a sender never paints one, and a
  receiver reads it as it reads the ladder, with no flag. The codec refuses a symbol past LIZARD-1024, which it used to
  build with a blank word. The sender page's 2 x 2 picture split, whose tiles were off the ladder or had no word, is
  gone (3.1, 5.1, section 2).
- **Border rule.** Every format was 286 modules a side (span = min(n, 256), so 1, 2 or 4 samples a module). On
  2026-09-23 a side became n / 8 + 60 modules (92, 124, 188, 316), the picture n / 8 + 30, and the module count
  identified n. Since 2026-09-27 a side is its ring's (below).
- **Rings** (2026-09-27). The border is one of four rings, B = 32, 64, 96 or 128 band cells a side, S = 2B + 30
  modules (94, 158, 222, 286), the picture spanning 2B. Any ring carries any picture, and the word, one code a ring,
  says what is inside (3.3, 5.1 to 5.5). The sender paints the 64 ring for every picture. That morning there were three
  (32, 48, 64), the default following n; that evening 32, 64 and 128, then the four. The receivers register against
  every ring, read the word in the ring that registered, and decode a wordless frame at the last word read (5.5, 10).
  The border is the code's, a 12-module corner mark and a 3-module guard (section 4); an even border (13 and 2) was
  built and reverted that day. 6.12's P2 and P3 moved to the 64 ring.
- **Resampling.** The border and the picture are on separate grids. The border is painted at pxm = ceil(n / span) whole
  pixels a module; the picture keeps its n samples and is resampled, Lanczos-3 over the picture read as periodic, to
  fill its modules, with no light between border and picture. The guard is the same resampler's periodic wrap (4.2,
  4.4, 6.8). The receiver samples picture sample x at module 15 + (x + 0.5) / scale (6.9).
- **Margin.** "A light zone of 2 modules is required outside the symbol" became: 2 modules of light are part of the
  symbol and the codec paints them (`liblizard/src/focus.h:FOCUS_QUIET`, `focus_paint_rgba`, 2026-09-24) (2, item 5;
  4.8). A guard ring round 3 modules of light was a sender option from 2026-09-23 until it was deleted on
  2026-09-26.
- **Band and word placement.** 129 cells a side (65 track, 64 word, 4 copies of the word) became half track and half
  word: n / 16 + 16 cells a side on the n / 8 + 60 border, B in ring B since 2026-09-27 (4.6). The old claim that no run
  in the band exceeds 4 modules holds for the track only.
- **The word's code.** An 8-byte RS(8, 3) word in 1, 1, 2 or 4 whole copies, summed before RS, with 8 spare word cells
  a side painted light at n >= 512, became one code a picture size over every word cell: RS(8, 3), RS(12, 3),
  RS(20, 3), RS(36, 3), no copies and no spare cells (2026-09-24), and one code a ring since 2026-09-27:
  RS(8, 3), RS(16, 3), RS(24, 3), RS(32, 3). The data is the same 3 bytes and the placement rule is the same, extended.
  The reference reader erases the least confident bytes in steps of 2 up to nroots - 3, or nroots - 1 in RS(8, 3),
  where it erased 2 (5.2 to 5.4).
- **Bit map default.** There was no interleaver: codeword bit t sat on slot t, the parity on the block's outer quarter,
  and the tail slots were zero. The format's map is now LINEAR, slot i carrying codeword bit 1943 i mod 5088, whitened
  by PRBS-23 indexed by frame slot, with the tail slots carrying the whitening (7.3). Recordings before 2026-09-23 were
  painted with NONE. LINEAR is the only map, and the word names none (2026-09-24).
- **Block.** The id was a fountain block id in the information word. It is now the transfer's: 14 bits of chunk over
  18 of fountain symbol, chunk 16383 the control range (7.5). The picture codec treats the 473 bytes as opaque.
  Acceptance requires a zero syndrome and the CRC (7.1).
- **The transfer** (2026-09-24). A file was one Wirehair fountain, at most 30,016,000 bytes, and its header
  (length, seed attempt, CRC-32, name) went over the rig's network, with no layout in the format. A file is now chunks
  of 2^10 to 2^24 bytes, a fountain each (a one-block chunk sent as it is), described in the light by a 469-byte header
  block and by manifest blocks of chunk chaining values, each chunk verified on arrival against the file's BLAKE3 root
  (7.5 to 7.10). The rig interleaves the chunks over the whole file and kept the network's header as a fallback until 2026-09-26 (9.1,
  9.4).
- **LDPC.** "Both ends derive H from (n_max, rate, seed). No matrix is stored or shipped" became: H is the printed shift
  table, shipped as `liblizard/src/ldpc_base.h` (2026-09-24), and the generator that made it builds only codes outside
  the format (8.2, 8.3). The profile table for the six rates the format does not use is dropped.
- **Coefficient order.** The old text claimed the order was identical for n = 256 and 512 over the first 80.4
  sub-channels. It is independent of n for every format, and every format's coefficients are a prefix of
  LIZARD-1024's (3.2, 6.3).
- **Display rule.** "Device px needed = (samples / n at n = 512) x 2.7 x R" became `ROOM_FOR`, which counts the border
  and the margin at each n. With the 64 ring the default for every picture (2026-09-27) the room rises with the
  version and the picker returns every format; on the n / 8 + 60 border it never returned six (3.5).
- **Constraints.** The patent-clean constraint is no longer listed (13, item 16). The software-decoding constraint
  became the implementation rules. Added: the margin is part of the symbol, the format is never gated on a decoder, two
  brand colours as a luma ramp, and a list of what is not the format (section 2).
- **Exact definitions and test vectors.** Bit and byte orders, the resampler, the painting rule, the track hash with
  per-side vectors, the RS generator, the LDPC construction, and test vectors for the picture (6.12), the block (7.4),
  the transfer (7.10), the word (5.2) and the test stream (9.3). The clip level is stated as the analytic rms sqrt(2P), which is what the
  reference uses (6.6, 6.7).
- **Receivers.** Registration moved into the C decoder's description (10.1). The GPU decoder is described (10.2).
- **Straddle cancellation** is out for good, and dithered black and white, a sender option since 2026-09-20, was deleted
  on 2026-09-26 (12).
- **FOCUS's finder frame** (2026-09-26). The frame the FOCUS reimplementation started from (corner finders,
  orientation strips, a 20-module margin; `thin = 0`, the harness default) is deleted, and with it the `thin`
  argument of `focus_init` and the wasm's `focus_setup`. The border of section 4 is the only frame; the vectors did
  not move (13, item 12).
- **The rig's pages** (2026-09-26). The receiver was told the test stream's truth and a baseline's format by the
  rig's server; it is now blind: bad test blocks are judged from the light (9.3), and the network header's fallback
  and the binary grid code's receiver path are gone. The sender stands alone. A page sends the server development logs
  only (12).
- **QR and Aztec archived** (2026-09-26), with `lib/`: Lizard is compared against other apps, no longer against
  them (12).
- **Left out.** The 2026-09-22 research questions on LDPC decode speed (int8 posteriors, the degree profile, base-graph
  width, rotations, batching blocks across lanes), which concern a decoder; its bit-mapping question is answered by
  LINEAR. Also its measured facts that were not rechecked for this rewrite (LDPC on BI-AWGN, capacity per ring, decode
  cost per stage).
