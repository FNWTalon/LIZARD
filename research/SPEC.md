# LIZARD format specification

The C in `liblizard/src/` is normative: the format is what `focus.c`, `layout.c`, `fmt.c`, `rs.c`, `ldpc.c`, `shake.c`,
`xfer.c` and their headers do, with the official BLAKE3 C 1.8.7 (`liblizard/vendor/blake3/`) under `xfer.c`. Where
this text and the code disagree, the code defines the format. Definitions cite their source as `file:function`, paths
under `liblizard/src/` unless given in full. The test vectors are checked against the codec's wasm build
(`liblizard/bindings/wasm/dist/lizard.wasm`, built by `liblizard/build.sh`, loaded through `liblizard/sim/ob.mjs`) by
`liblizard/test/vectors.mjs`. Sections and items marked non-normative describe the reference sender and receivers.

## Contents

1. [What LIZARD is](#1-what-lizard-is)
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
14. [Changes](#14-changes)
15. [Replays: the .lizard file](#15-replays-the-lizard-file)

## 1. What LIZARD is

LIZARD is a luminance-only animated 2D barcode for one-way transfer from a screen to a camera, derived from FOCUS
(Hermans et al., MobiSys 2016). A screen paints a sequence of frames, a camera films them, and nothing comes back. Each
frame is one symbol: a square grey picture of n x n samples whose 2D spectrum carries QPSK symbols on the coefficients
of the upper half-plane, in order of rising spatial frequency, inside a black and white border 15 modules deep. The
border is one of three rings, whose band holds 64, 128 or 256 cells a side; any ring may carry any picture. The ring
registers the symbol and carries the format word, one Reed-Solomon codeword that states the sub-channel count (so the
picture size and the blocks) and the display rate. The payload travels in blocks of 476 bytes (a 32-bit block id and
472 payload bytes) with a CRC-32, each coded by a QC-LDPC code onto consecutive sub-channels of 320 coefficients at a
rate that follows the frequency: 7/8, 3/4, 2/3 and 1/2 from the inside out (the rate profile, 3.1). Every symbol also
carries pilots in its blocks' tails (7.3) and 49 anchors painted over its picture (6.13). Blocks from any frame feed a
Wirehair fountain per chunk of the file, described by header and manifest blocks and verified against the file's
BLAKE3 hash, all in the light (7.5 to 7.10).

Blur, distance and resampling take the outermost blocks first, so a frame loses blocks gradually instead of stopping
dead at one resolution.

### 1.1 Terms

| term | meaning |
|---|---|
| ring B | one of the three border sizes, named by its band cells a side: B = 64, 128 or 256 (`focus.h:FOCUS_RING`) |
| symbol | the picture inside the border, S = 2B + 30 modules a side (158, 286, 542). S and module coordinates exclude the margin |
| module | the border's unit of length |
| span | the picture's width in modules, 2B |
| picture | the n x n grey image inside the border that carries the data (section 6) |
| sample | one of the picture's n x n values; scale = n / span samples a module |
| pxm | painted pixels a module, a whole number (3.3) |
| painted image | the symbol and its margin as the codec paints them, (S + 4) pxm pixels a side |
| drive | the symbol as the encoder's float image, S pxm pixels a side, 0 dark and 1 light, margin excluded (`focus.c:focus_encode`) |
| border | the 15 modules round the picture: rim, ring, gap, band, light depths and guard (4.3) |
| outer border | depths 0 to 11, everything but the guard; every module one level, dark or light |
| ring (line) | the dark line at depths 1 and 2; by extension the border's size, as in "ring B" |
| guard | depths 12 to 14, the picture's periodic extension (4.4) |
| margin | 2 modules of light round the symbol, `FOCUS_QUIET`, painted by the codec (4.8). The C's variable `margin` is the 15-module border depth, not this |
| mark | a corner mark, the 12 x 12 module square at each corner (4.5) |
| band, cell | depths 5 and 6, in cells of 2 x 2 modules: track cells and word cells (4.6) |
| word | the format word, one Reed-Solomon codeword over the band's word cells (section 5) |
| anchor | one of 49 squares of span / 64 modules painted over the picture on a 7 x 7 grid (6.13) |
| top ring | the outermost frequency a format uses, radius R in cycles across the picture (3.2) |
| coefficient | one frequency (u, v) of the picture's 2D DFT; it carries one QPSK symbol, two slots |
| slot | one transmitted bit: the sign of a coefficient's real or imaginary part |
| sub-channel | 320 consecutive coefficients of the frequency order (6.3) |
| subch | the number of sub-channels in a frame; LIZARD-512 has subch = 512 |
| version | V = subch / 16, 2 to 64, the word's second byte. A frame of it carries B(V) blocks (3.1) |
| block | 476 bytes, one LDPC codeword and its pilots on 4496, 5152, 5760 or 7680 consecutive slots by its rate (3.1) |
| rate profile | the blocks' code rates by frequency, inner first, a function of subch alone (3.1) |
| padding | the frame's slots past its last block, which carry the whitening alone (6.5) |
| block id | bytes 0 to 3 of a block: 14 bits of chunk over 18 of fountain symbol (7.5) |
| chunk | 2^k bytes of the file (the last may be shorter), its own Wirehair fountain (7.6) |
| fountain symbol | the low 18 bits of a block id, its Wirehair block id in its chunk (`xfer.h` calls it the symbol) |
| control block | the header block or a manifest block, chunk 16383 (7.5, 7.7, 7.8) |
| chaining value | BLAKE3's 32-byte value of a subtree of the file's hash tree; each chunk has one (7.9) |
| count c | the sender's count of painted pictures mod 4, carried by the pilots (7.3, 9.1) |

## 2. Constraints (normative)

1. **Luminance only.** A sender MUST carry information in luminance alone. Every pixel it paints, margin included, is
   a grey level: `focus.c:focus_paint_rgba` writes one value to R, G and B. Colour MUST NOT carry information. A sender
   MAY paint the dark and light ends of the grey ramp in two brand colours, the data still only in luminance, if the
   pair meets both halves of the palette rule:
   - the luma difference is at least 60% of full range, about 153 of 255, with luma
     Y = (77 R + 150 G + 29 B + 128) >> 8 as the receiver computes it (`acquire.c:ob_luma`);
   - the three channel swings (light minus dark in R, in G and in B) are within about 15% of each other, so a module
     boundary carries no chroma edge.

   No code defines the mapping of a grey level to a colour between the two ends (13, items 10 and 11).
2. **Generic screen, generic camera.** A sender MUST NOT rely on, and a receiver MUST NOT require, a property of one
   display or one camera.
3. **The border grows outwards only.** Every mark, band and margin lies outside the picture's span = 2B modules. Inside
   the picture only the anchors are painted (6.13), and a receiver erases them before reading it.
4. **The symbol is self-describing.** Everything a receiver needs to decode a frame is in the frame. The ring locates
   the symbol and syncs its grid; the format word it carries gives the sub-channel count, and so the picture size and
   the blocks (3.1, 3.2, section 5). Anything about the transfer belongs in the payload, never in the border (7.5 to
   7.8).
5. **The margin is part of the symbol.** An encoder MUST paint 2 modules of light round the symbol's S x S modules
   (`focus.h:FOCUS_QUIET` = 2, 4.8). With the border's light rim, the ring has 3 modules of light outside it.
6. **The channel.** The symbol and its margin are undisturbed; noise lies outside them (a busy page, a noisy surround,
   clutter up to the margin). Nothing is painted over the symbol.
7. **Scope.** This specification is the 2D code and the layouts that carry the transfer (7.5 to 7.9). Wirehair and
   zstd are fixed codecs used as they are.

**Conformance.** The reference build is exact: `liblizard/src/` as `liblizard/build.sh` builds it (with
`-ffp-contract=off`, so no fused multiply-add changes a rounding) paints the same bytes on every machine, and its test
vectors (5.2, 6.12, 7.4, 9.3) are its regression check. Another encoder is not held to those bytes. A sender conforms
by painting a symbol that decodes: the border (sections 4 and 5, the margin included) and the anchors' pattern exactly,
the coded bits exactly as 6.3 to 6.5 and sections 7 and 8 define them, and the real arithmetic of the picture (6.6 to
6.8) to the formula, not to the reference's last bits. A receiver never checks conformance.

**Not the format.** `focus_init` takes two sender parameters that no receiver needs and the word cannot state: the
clipping ratio c (6.7) and a power tilt t across the sub-channels (6.5). Every sender paints c = 2 and t = 0, and
whether a conforming sender may use others is open (13, item 8). `focus.c:focus_init_tiers` builds rate tiers other
than the profile's for experiments; a frame painted with them is not LIZARD and a receiver told nothing reads it
wrongly.

**Implementation rules** for the project's decoders, not the symbol: no SharedArrayBuffer (so no wasm threads; one
heap a worker); a decoder that falls short on the format is fixed, and the format is not bent for it; a measured
figure states its conditions (a still camera, exact registration or a white screen are caveats, not defaults).

## 3. The formats

### 3.1 Names, versions and the ladder (normative)

- A **sub-channel** is 320 consecutive coefficients of the coefficient order (`focus.h:FOCUS_SUB`, 6.3), 640 slots. A
  format counts its sub-channels; the receiver's soft values are estimated a sub-channel at a time (8.5).
- A **block** is 476 bytes coded as one codeword, followed by its pilots, on consecutive slots, as many as its rate
  needs (`focus.c:focus_block_slots`): 4496 at 7/8, 5152 at 3/4, 5760 at 2/3, 7680 at 1/2 (3.4). Blocks are laid from
  slot 0 outwards, each beginning on the slot after the one before ended, whatever the sub-channels (`focus.c:init`,
  `block_slot`); the slots past the last block are the padding (6.5).
- A **format** is named after its sub-channel count: LIZARD-k has subch = k, a multiple of 16. Its **version** is
  V = subch / 16 (`focus.h:FOCUS_STEP`), which the word's version byte carries (`focus.c:fmt_paint`).
- The **ladder** is every version from 4 to 64: LIZARD-64, -80, ..., -1024, 61 formats
  (`liblizard/sim/lizard_pick.mjs:VERSIONS`). A sender MAY paint any of them. A receiver reads any version the word
  names, with no flag (`fmt.c:ob_fmt_decode`, `OB_FMT_VERSION_MIN` = 4, `OB_FMT_VERSION_MAX` = 64).
- The smallest format, LIZARD-64, carries seven blocks, its first four 7/8 blocks, so every format carries two pilot
  blocks of each parity (7.3). Versions 1 to 3 are no formats: LIZARD-16's 10,240 slots hold one 7/8 and one 3/4
  block, and the odd blocks' pilots of LIZARD-32 and -48 are one 7/8 block's 8 coefficients, which read past the
  pilots' model on captures mixed with a neighbouring picture and under 0.9 on clean ones (7.3, 9.1). A sub-channel
  count that is not a multiple of 16 is no format: its version would not be whole, and `focus_tiers_for` has no
  profile for it.
- A version past 64 cannot be stated by the word: `focus.c:init` refuses more than 1024 sub-channels.

**The rate profile.** A frame of subch sub-channels has S = 640 subch slots. Inner first, it carries a blocks at 7/8
(s7 = 4496 slots each), m at 3/4 (s3 = 5152), t at 2/3 (s2 = 5760) and c at 1/2 (s1 = 7680), with
a s7 + t s2 + c s1 + s3 <= S and m = floor((S - a s7 - t s2 - c s1) / s3) >= 1: the 3/4 tier fills, and the
S - (a s7 + m s3 + t s2 + c s1) slots it leaves are the padding. Of the (a, t, c) that fit, the profile is the one with
the least

```
(1000 a s7 - 390 S)^2 + (1000 t s2 - 150 S)^2 + (1000 c s1 - 200 S)^2
```

in integers, the first of equals in rising a, then rising t, then rising c (`focus.c:focus_tiers_for`, `FOCUS_TIER_IN`
= 390, `FOCUS_TIER_23` = 150, `FOCUS_TIER_OUT` = 200, thousandths of the slots; `liblizard/sim/lizard_pick.mjs:tiersFor`,
which counts in units of 16 slots, every block's slots being a multiple of 16, so that each cost is an exact integer
under 2^53). The targets are the profile's split over the sub-channels before the blocks were packed by slots (39, 15 and 20%
at 7/8, 2/3 and 1/2), now shares of the slots (12, 13). Every format has all four tiers. B(V) = a + m + t + c
(`focus_blocks_for`) is under 2V at every version and never falls as V rises. The padding runs from none (LIZARD-320 and -1008) to 4976 slots
(LIZARD-768), at most 10.7% of a frame (LIZARD-64's 4384), under 5% from LIZARD-96 up and under 2% from LIZARD-320 up
(0.46% at LIZARD-656). Nothing about the profile is signalled: it
follows from the version.

| format | tiers, inner first | blocks B(V) | slots padded |
|---|---|---|---|
| format | tiers, inner first | blocks B(V) | slots padded |
| LIZARD-64 | 7/8 x 4, 3/4 x 1, 2/3 x 1, 1/2 x 1 | 7 | 4384 |
| LIZARD-80 | 7/8 x 4, 3/4 x 3, 2/3 x 1, 1/2 x 1 | 9 | 4320 |
| LIZARD-128 | 7/8 x 7, 3/4 x 4, 2/3 x 2, 1/2 x 2 | 15 | 2960 |
| LIZARD-256 | 7/8 x 14, 3/4 x 9, 2/3 x 4, 1/2 x 4 | 31 | 768 |
| LIZARD-432 | 7/8 x 24, 3/4 x 14, 2/3 x 7, 1/2 x 7 | 52 | 2368 |
| LIZARD-512 | 7/8 x 28, 3/4 x 15, 2/3 x 9, 1/2 x 9 | 61 | 3552 |
| LIZARD-560 | 7/8 x 31, 3/4 x 19, 2/3 x 9, 1/2 x 9 | 68 | 176 |
| LIZARD-592 | 7/8 x 33, 3/4 x 18, 2/3 x 10, 1/2 x 10 | 71 | 3376 |
| LIZARD-656 | 7/8 x 36, 3/4 x 21, 2/3 x 11, 1/2 x 11 | 79 | 1952 |
| LIZARD-1024 | 7/8 x 57, 3/4 x 33, 2/3 x 17, 1/2 x 17 | 124 | 592 |

A version names what is inside the symbol and nothing about its size: the symbol's size is its ring's (3.3).

### 3.2 The picture size follows from the sub-channels (normative)

The top ring radius, in cycles across the picture, is the radius of the half-disc that holds 320 subch coefficients:

```
R = sqrt(2 * 320 * subch / pi) = sqrt(10240 * V / pi)
```

n MUST be the first of 256, 384, 512, 768, 1024, 1152, 1280 and 1536 with n >= 3R, computed in double precision, or 1536 where
none holds (`focus.c:focus_n_for` over `FOCUS_PICTURES`; `liblizard/sim/lizard_pick.mjs:N_FOR`). The fallback never
binds on the ladder (LIZARD-1024 has 3R = 1370.2). Since n >= 3R is V <= pi n^2 / 92160 (the 256 picture holds
versions 1 and 2, which are no formats, 3.1; no format's picture is 256):

| n | versions | formats | count | blocks a frame | useful bytes a frame (3.4) |
|---|---|---|---|---|---|
| n | versions | formats | count | blocks a frame | useful bytes a frame (3.4) |
| 256 | none | | 0 | | |
| 384 | 4 to 5 | LIZARD-64 to -80 | 2 | 7 to 9 | 3304 to 4248 |
| 512 | 6 to 8 | LIZARD-96 to -128 | 3 | 11 to 15 | 5192 to 7080 |
| 768 | 9 to 20 | LIZARD-144 to -320 | 12 | 17 to 39 | 8024 to 18408 |
| 1024 | 21 to 35 | LIZARD-336 to -560 | 15 | 40 to 68 | 18880 to 32096 |
| 1152 | 36 to 45 | LIZARD-576 to -720 | 10 | 69 to 87 | 32568 to 41064 |
| 1280 | 46 to 55 | LIZARD-736 to -880 | 10 | 89 to 106 | 42008 to 50032 |
| 1536 | 56 to 64 | LIZARD-896 to -1024 | 9 | 108 to 124 | 50976 to 58528 |

n / 2R runs from 1.50 to 2.24 on the ladder. Only the word states n: a receiver takes n = `focus_n_for(16 * version)`
and nothing checks the version against the ring.

**n sets only how finely the same coefficients are sampled.** Every coefficient a format uses lies within R + 0.07,
and n / 2 >= 1.5 R, so a coefficient that a larger n adds sorts after all of them. Therefore a format's coefficients
are the same at every n that can hold it, and within one n each format's coefficients are the first 320 subch of the
top format's: block b of any version sits on the same coefficients, with the same bit map and whitening, as block b of
the top version of that n wherever their tiers agree (`wasm.c:focus_any_setup`, `pic_for`).

`focus.c:init` checks only that n is one of the eight sizes (each with n / 2 a whole number of the transform's 64-column strips)
and that the half-plane holds 320 subch coefficients ((n / 2)(n - 2) at most).

### 3.3 The rings (normative)

Ring B has B band cells of 2 x 2 modules a side, B = 64, 128 or 256 (`focus.h:FOCUS_RING`), and its geometry does not
depend on the picture:

- span = 2B modules across the picture (128, 256, 512); `focus_init`'s span argument names the ring, 0 the default,
  any other value is refused (`focus.c:init`).
- The border is 15 modules a side: the corner mark's 12 (`OB_THIN_CORNER`) and the guard's 3 (`FOCUS_CP`).
- S = 2B + 30 modules a side (158, 286, 542); the painted square, with the margin, is S + 4 (162, 290, 546).

**Any ring may carry any picture.** A sender MAY paint a picture of any n in any ring. The reference senders paint the
128 ring unless another is named (`focus.h:FOCUS_RING_DEFAULT`, `liblizard/sim/lizard_pick.mjs:RING_DEFAULT`,
`liblizard/kotlin/Pick.kt`). A receiver MUST NOT infer n from the ring: the module count identifies the ring and
nothing else.

In ring B carrying a picture of n samples:

```
pxm = max( ceil(n / span - 0.0001), ceil(384 / span) )        the first term in single precision (focus.c:init)
scale = n / span                                              samples a module
```

pxm is the painted pixels a module, a whole number, so no module is split across pixels. The second term is the
anchors' floor (6.13): a square of span / 64 modules is then at least 6 pixels. The picture keeps its n samples and is
resampled to fill its span modules, upscaled by pxm / scale, never shrunk (6.8); where pxm = scale the resampling is an
exact copy. Nine pairs resample: n = 256 in the 64 ring; 256, 384 and 1152 in the 128 ring; 256, 384, 768, 1152 and
1280 in the 256 ring.
The resampled square is (span + 6) pxm pixels a side from pixel 12 pxm on both axes (the picture and its guard, 6.8).

| ring B | span | S | n | scale | pxm | upscale | symbol, px | painted, px | resampled square, from |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 158 | 256 | 2 | 3 | 3/2 | 474 | 486 | 402, 36 |
| 64 | 128 | 158 | 384 | 3 | 3 | copy | 474 | 486 | 402, 36 |
| 64 | 128 | 158 | 512 | 4 | 4 | copy | 632 | 648 | 536, 48 |
| 64 | 128 | 158 | 768 | 6 | 6 | copy | 948 | 972 | 804, 72 |
| 64 | 128 | 158 | 1024 | 8 | 8 | copy | 1264 | 1296 | 1072, 96 |
| 64 | 128 | 158 | 1152 | 9 | 9 | copy | 1422 | 1458 | 1206, 108 |
| 64 | 128 | 158 | 1280 | 10 | 10 | copy | 1580 | 1620 | 1340, 120 |
| 64 | 128 | 158 | 1536 | 12 | 12 | copy | 1896 | 1944 | 1608, 144 |
| 128 | 256 | 286 | 256 | 1 | 2 | 2 | 572 | 580 | 524, 24 |
| 128 | 256 | 286 | 384 | 3/2 | 2 | 4/3 | 572 | 580 | 524, 24 |
| 128 | 256 | 286 | 512 | 2 | 2 | copy | 572 | 580 | 524, 24 |
| 128 | 256 | 286 | 768 | 3 | 3 | copy | 858 | 870 | 786, 36 |
| 128 | 256 | 286 | 1024 | 4 | 4 | copy | 1144 | 1160 | 1048, 48 |
| 128 | 256 | 286 | 1152 | 9/2 | 5 | 10/9 | 1430 | 1450 | 1310, 60 |
| 128 | 256 | 286 | 1280 | 5 | 5 | copy | 1430 | 1450 | 1310, 60 |
| 128 | 256 | 286 | 1536 | 6 | 6 | copy | 1716 | 1740 | 1572, 72 |
| 256 | 512 | 542 | 256 | 1/2 | 1 | 2 | 542 | 546 | 518, 12 |
| 256 | 512 | 542 | 384 | 3/4 | 1 | 4/3 | 542 | 546 | 518, 12 |
| 256 | 512 | 542 | 512 | 1 | 1 | copy | 542 | 546 | 518, 12 |
| 256 | 512 | 542 | 768 | 3/2 | 2 | 4/3 | 1084 | 1092 | 1036, 24 |
| 256 | 512 | 542 | 1024 | 2 | 2 | copy | 1084 | 1092 | 1036, 24 |
| 256 | 512 | 542 | 1152 | 9/4 | 3 | 4/3 | 1626 | 1638 | 1554, 36 |
| 256 | 512 | 542 | 1280 | 5/2 | 3 | 6/5 | 1626 | 1638 | 1554, 36 |
| 256 | 512 | 542 | 1536 | 3 | 3 | copy | 1626 | 1638 | 1554, 36 |

The pixel columns are the painted image's own size (`_focus_side()`, `_focus_cell()`); a sender then scales it to the
room (3.5, 9.5).

### 3.4 Capacity (normative)

| rate | slots a block | codeword n (sent nt) | z | tail slots (pilots) | information k | zero bits past the CRC |
|---|---|---|---|---|---|---|
| rate | slots a block | codeword n (sent nt) | z | tail slots (pilots) | information k | zero bits past the CRC |
| 7/8 | 4496 | 4608 (4480) | 128 | 16 | 3968 | 128 |
| 3/4 | 5152 | 5120 (5120) | 128 | 32 | 3840 | 0 |
| 2/3 | 5760 | 5760 (5760) | 128 | 0 | 3840 | 0 |
| 1/2 | 7680 | 7680 (7680) | 128 | 0 | 3840 | 0 |

- A block's slots are its code's nt sent bits and then its tail, the slots that carry the pilots (7.3): 16 at 7/8 and
  32 at 3/4; a 2/3 or 1/2 block has none. Every code is lifted by z = 128 (section 8); the 7/8 code's last data
  column, the 128 zero information bits past the CRC, is never sent (np = 128, 8.6).
- The information word is the 476 block bytes and their CRC-32, 3840 bits, the k of the 3/4, 2/3 and 1/2 codes; the
  7/8 code's k holds 128 zero bits past them (7.1). A block is 476 bytes whatever its rate (`focus.h:FOCUS_BLOCK`, `focus.c:init`, `block_bytes`).
- The first 4 of the 476 bytes are the block id (7.5), which leaves 472 useful bytes (3776 bits).

A frame of version V has 5120 V coefficients, 10240 V slots, B(V) blocks, the padding past them (3.1) and 472 B(V)
useful bytes in any ring (3.2's table by n, 3.6 by format). Its ceiling in bytes a second is 472 B(V) times the display
rate. To reproduce a row: `M._focus_setup(n, subch, 2, 2B, 0)` (n, subch, clip, span, tilt) returns 476, then
`M._focus_blocks()`, `M._focus_side()`, `M._focus_cell()` and `M._focus_quiet()` give B(V), S pxm, pxm and 2;
`M._focus_tiers_for(subch, p)` writes the tiers as (rate index, blocks, slots a block) triplets.

### 3.5 Choosing a version (non-normative)

The reference picker (`liblizard/sim/lizard_pick.mjs:pickVersion`) takes one input, the device pixels the symbol may
take (the room), and makes no camera assumption.

- **Room a format needs**, margin included, in ring B:
  `ROOM_FOR(subch, ring) = (MODULES + 2 M) / SPAN * 2.7 * R(subch)` with SPAN = 2B, MODULES = 2B + 30, M = 2
  (`PAINTED_MARGIN()`, the codec's `FOCUS_QUIET`), and 2.7 device px a cycle at the top ring (`T_DISPLAY_CYCLE`). In one
  ring the room is a constant times R, so it rises with the version.
- **The pick** is the largest ladder format whose room fits, or LIZARD-64 where none does (flagged `tight`); the ring
  is the default unless named. The library does not cap; an app is advised to stop at LIZARD-560 unless it is a fixed
  rig (13, item 21).
- **What a format asks of the camera**, not an input: `CAMERA_PX_FOR(subch, ring) = MODULES / SPAN * 2.12 * R(subch)`,
  camera px across the symbol, margin excluded.
- A sender MAY change format mid-transfer (on a resize, say): every block carries the same 472 useful bytes (9.1).

### 3.6 Every format

Derived from 3.1 to 3.4 (normative) for every version; the room and camera columns are 3.5's guidance in the default
128 ring.

| format | V | n | blocks B(V) | useful B a frame | R | n / 2R | room, device px | camera px |
|---|---|---|---|---|---|---|---|---|
| format | V | n | blocks B(V) | useful B a frame | R | n / 2R | room, device px | camera px |
| LIZARD-64 | 4 | 384 | 7 | 3304 | 114.18 | 1.68 | 349.2 | 270.4 |
| LIZARD-80 | 5 | 384 | 9 | 4248 | 127.66 | 1.50 | 390.5 | 302.4 |
| LIZARD-96 | 6 | 512 | 11 | 5192 | 139.85 | 1.83 | 427.7 | 331.2 |
| LIZARD-112 | 7 | 512 | 13 | 6136 | 151.05 | 1.69 | 462.0 | 357.8 |
| LIZARD-128 | 8 | 512 | 15 | 7080 | 161.48 | 1.59 | 493.9 | 382.5 |
| LIZARD-144 | 9 | 768 | 17 | 8024 | 171.28 | 2.24 | 523.9 | 405.7 |
| LIZARD-160 | 10 | 768 | 19 | 8968 | 180.54 | 2.13 | 552.2 | 427.6 |
| LIZARD-176 | 11 | 768 | 21 | 9912 | 189.35 | 2.03 | 579.2 | 448.5 |
| LIZARD-192 | 12 | 768 | 23 | 10856 | 197.77 | 1.94 | 604.9 | 468.4 |
| LIZARD-208 | 13 | 768 | 25 | 11800 | 205.85 | 1.87 | 629.6 | 487.5 |
| LIZARD-224 | 14 | 768 | 26 | 12272 | 213.62 | 1.80 | 653.4 | 505.9 |
| LIZARD-240 | 15 | 768 | 29 | 13688 | 221.12 | 1.74 | 676.3 | 523.7 |
| LIZARD-256 | 16 | 768 | 31 | 14632 | 228.37 | 1.68 | 698.5 | 540.9 |
| LIZARD-272 | 17 | 768 | 32 | 15104 | 235.40 | 1.63 | 720.0 | 557.5 |
| LIZARD-288 | 18 | 768 | 34 | 16048 | 242.22 | 1.59 | 740.9 | 573.7 |
| LIZARD-304 | 19 | 768 | 36 | 16992 | 248.86 | 1.54 | 761.2 | 589.4 |
| LIZARD-320 | 20 | 768 | 39 | 18408 | 255.32 | 1.50 | 780.9 | 604.7 |
| LIZARD-336 | 21 | 1024 | 40 | 18880 | 261.63 | 1.96 | 800.2 | 619.7 |
| LIZARD-352 | 22 | 1024 | 42 | 19824 | 267.79 | 1.91 | 819.0 | 634.2 |
| LIZARD-368 | 23 | 1024 | 44 | 20768 | 273.80 | 1.87 | 837.5 | 648.5 |
| LIZARD-384 | 24 | 1024 | 46 | 21712 | 279.69 | 1.83 | 855.5 | 662.4 |
| LIZARD-400 | 25 | 1024 | 48 | 22656 | 285.46 | 1.79 | 873.1 | 676.1 |
| LIZARD-416 | 26 | 1024 | 50 | 23600 | 291.11 | 1.76 | 890.4 | 689.5 |
| LIZARD-432 | 27 | 1024 | 52 | 24544 | 296.66 | 1.73 | 907.4 | 702.6 |
| LIZARD-448 | 28 | 1024 | 54 | 25488 | 302.10 | 1.69 | 924.0 | 715.5 |
| LIZARD-464 | 29 | 1024 | 56 | 26432 | 307.45 | 1.67 | 940.4 | 728.2 |
| LIZARD-480 | 30 | 1024 | 58 | 27376 | 312.71 | 1.64 | 956.4 | 740.6 |
| LIZARD-496 | 31 | 1024 | 60 | 28320 | 317.87 | 1.61 | 972.2 | 752.9 |
| LIZARD-512 | 32 | 1024 | 61 | 28792 | 322.96 | 1.59 | 987.8 | 764.9 |
| LIZARD-528 | 33 | 1024 | 63 | 29736 | 327.97 | 1.56 | 1003.1 | 776.8 |
| LIZARD-544 | 34 | 1024 | 65 | 30680 | 332.90 | 1.54 | 1018.2 | 788.5 |
| LIZARD-560 | 35 | 1024 | 68 | 32096 | 337.76 | 1.52 | 1033.1 | 800.0 |
| LIZARD-576 | 36 | 1152 | 69 | 32568 | 342.55 | 1.68 | 1047.7 | 811.3 |
| LIZARD-592 | 37 | 1152 | 71 | 33512 | 347.28 | 1.66 | 1062.2 | 822.5 |
| LIZARD-608 | 38 | 1152 | 73 | 34456 | 351.94 | 1.64 | 1076.4 | 833.5 |
| LIZARD-624 | 39 | 1152 | 75 | 35400 | 356.54 | 1.62 | 1090.5 | 844.4 |
| LIZARD-640 | 40 | 1152 | 77 | 36344 | 361.08 | 1.60 | 1104.4 | 855.2 |
| LIZARD-656 | 41 | 1152 | 79 | 37288 | 365.57 | 1.58 | 1118.1 | 865.8 |
| LIZARD-672 | 42 | 1152 | 81 | 38232 | 370.00 | 1.56 | 1131.7 | 876.3 |
| LIZARD-688 | 43 | 1152 | 83 | 39176 | 374.38 | 1.54 | 1145.1 | 886.7 |
| LIZARD-704 | 44 | 1152 | 85 | 40120 | 378.71 | 1.52 | 1158.3 | 896.9 |
| LIZARD-720 | 45 | 1152 | 87 | 41064 | 382.98 | 1.50 | 1171.4 | 907.1 |
| LIZARD-736 | 46 | 1280 | 89 | 42008 | 387.22 | 1.65 | 1184.3 | 917.1 |
| LIZARD-752 | 47 | 1280 | 90 | 42480 | 391.40 | 1.64 | 1197.1 | 927.0 |
| LIZARD-768 | 48 | 1280 | 92 | 43424 | 395.54 | 1.62 | 1209.8 | 936.8 |
| LIZARD-784 | 49 | 1280 | 95 | 44840 | 399.64 | 1.60 | 1222.3 | 946.5 |
| LIZARD-800 | 50 | 1280 | 97 | 45784 | 403.70 | 1.59 | 1234.8 | 956.1 |
| LIZARD-816 | 51 | 1280 | 98 | 46256 | 407.72 | 1.57 | 1247.0 | 965.7 |
| LIZARD-832 | 52 | 1280 | 100 | 47200 | 411.70 | 1.55 | 1259.2 | 975.1 |
| LIZARD-848 | 53 | 1280 | 102 | 48144 | 415.64 | 1.54 | 1271.3 | 984.4 |
| LIZARD-864 | 54 | 1280 | 104 | 49088 | 419.54 | 1.53 | 1283.2 | 993.7 |
| LIZARD-880 | 55 | 1280 | 106 | 50032 | 423.41 | 1.51 | 1295.0 | 1002.8 |
| LIZARD-896 | 56 | 1536 | 108 | 50976 | 427.24 | 1.80 | 1306.7 | 1011.9 |
| LIZARD-912 | 57 | 1536 | 110 | 51920 | 431.03 | 1.78 | 1318.4 | 1020.9 |
| LIZARD-928 | 58 | 1536 | 112 | 52864 | 434.80 | 1.77 | 1329.9 | 1029.8 |
| LIZARD-944 | 59 | 1536 | 114 | 53808 | 438.53 | 1.75 | 1341.3 | 1038.6 |
| LIZARD-960 | 60 | 1536 | 116 | 54752 | 442.23 | 1.74 | 1352.6 | 1047.4 |
| LIZARD-976 | 61 | 1536 | 118 | 55696 | 445.90 | 1.72 | 1363.8 | 1056.1 |
| LIZARD-992 | 62 | 1536 | 119 | 56168 | 449.54 | 1.71 | 1375.0 | 1064.7 |
| LIZARD-1008 | 63 | 1536 | 122 | 57584 | 453.15 | 1.69 | 1386.0 | 1073.3 |
| LIZARD-1024 | 64 | 1536 | 124 | 58528 | 456.74 | 1.68 | 1397.0 | 1081.8 |

## 4. The border

Normative except 4.10 and 4.11. Sources: `layout.h`, `layout.c` (`thin_init`, `thin_corner`, `ob_thin_track`,
`ob_layout_set_fmt`), `focus.c` (`init`, `focus_encode`, `resample_init`, `focus_paint_rgba`), `focus.h`.

### 4.1 Units and coordinates

- The unit is the module. A symbol in ring B is S = 2B + 30 modules a side: span = 2B modules of picture inside 15
  modules of border on each side.
- Module (x, y): x grows to the right, y downwards, (0, 0) the top-left module as displayed. In continuous module
  coordinates module (x, y) covers `[x, x + 1) x [y, y + 1)`, its centre `(x + 0.5, y + 0.5)`.
- The depth of a module is `d = min(x, y, S - 1 - x, S - 1 - y)`, 0 on the outer edge.
- Sides are numbered 0 top, 1 right, 2 bottom, 3 left. The position t along a side is x on the top and bottom and y on
  the left and right (every side counts left to right or top to bottom). The module at along t and depth d is `(t, d)`
  on the top, `(S - 1 - d, t)` on the right, `(t, S - 1 - d)` on the bottom and `(d, t)` on the left.
- Levels: in the outer border the drive is 0.0 for dark and 1.0 for light, painted as grey 0 or 255 (6.8), as is the
  margin. The guard is resampled picture and takes grey levels.

### 4.2 Whole pixels a module

- The border is painted at pxm pixels a module (3.3).
- Every module of the outer border (depths 0 to 11, the marks included) MUST be painted as a pxm x pxm block of one
  level, covering drive pixels `[x pxm, (x + 1) pxm) x [y pxm, (y + 1) pxm)`.
- The picture is on its own grid: sample x (0 to n - 1) is centred at module coordinate `15 + (x + 0.5) / scale` on
  each axis (6.8, 6.9). The guard touches both the border and the picture; nothing light lies between them.

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

Two exceptions: the four 12 x 12 corner squares are the marks (4.5), and in the band's reserve, the 15 modules at
each end of each side, depths 5 and 6 are light (4.6). Among symbols of one ring the outer border differs only in the
word's cells.

### 4.4 Guard

Depths 12 to 14 (`FOCUS_CP` = 3) are the picture wrapped round its own edge. The encoder resamples over span + 6
modules from module 12 and reads the picture as periodic on both axes, so the guard is a torus wrap, corners included.
In drive pixels, with `o = 12 pxm`, for `0 <= X < 3 pxm` column `o + X` equals column `o + X + span pxm` over the
whole resampled square, row `o + X` equals row `o + X + span pxm`, and the corners follow. An encoder MUST paint the
guard this way: it makes a camera's blur act on the picture as a circular convolution, a gain on each coefficient.

### 4.5 Corner marks

A mark fills the 12 x 12 square at each corner (`OB_THIN_CORNER` = 12), where the depth table does not apply. For a
module (x, y) let `cx = min(x, S - 1 - x)` and `cy = min(y, S - 1 - y)`. If both are under 12 the module belongs to a
mark and is (`thin_corner`):

- light if `cx` or `cy` is 0 (the rim) or 11 (the inner edge);
- otherwise, with `e = min(cx, cy)`: dark if e is 1 or 2 (the ring carried in), light if e is 3 or 4 (the gap carried
  in), dark if e is 5 or more (the core).

So each mark is a 6 x 6 dark core at modules 5 to 10 from its corner, wrapped on its two outer sides by the gap and
the ring, light at module 11 on its inner sides. The ring is broken there: at depths 1 and 2, modules 11 and S - 12
along each side are light. A scan through the core from the corner (any row or column 5 to 10 of the mark) sees light
1, dark 2, light 2, dark 6, then light: ring : gap : core = 2 : 2 : 6 (`acquire.c:mark_ratio`).

| form | when | what a scan finds | extent from the corner | centre from the corner |
|---|---|---|---|---|
| gapped | sharp | the 2 : 2 : 6 cross-section | core, modules 5 to 10 | (8.0, 8.0) |
| merged | blur has closed the gap | one dark square, 10 x 10 | modules 1 to 10 | (6.0, 6.0) |

Centres are continuous module coordinates measured in from the mark's own corner (`acquire.c:mark_form_mid`).

### 4.6 The band

- The band is depths 5 and 6, in cells of 2 x 2 modules, from along 15 to `S - 16` on each side. The reserve, 15
  modules at each end, is `OB_THIN_RESERVE` = `OB_THIN_CORNER` + 3, so no band cell touches a mark.
- Band cells a side: B = (S - 30) / 2. Band cell b (0 to B - 1) covers along `[15 + 2b, 17 + 2b)`.
- Runs of 4 cells alternate, starting with the track (`OB_BAND_RUN` = 4): cell b is a track cell when `floor(b / 4)` is
  even and a word cell when it is odd. Track cell j is band cell `8 floor(j / 4) + j mod 4`, word cell i is band cell
  `8 floor(i / 4) + 4 + i mod 4` (`layout.h` `OB_TRACK_AT`, `OB_WORD_AT`). Each side holds B / 2 of each kind (32, 64,
  128) and ends on a whole word run.

### 4.7 Timing track

Track cells 2p and 2p + 1, adjacent in one run, carry a bit and its complement (Manchester), from a hash that differs
by side (`ob_thin_track`). Arithmetic is unsigned 32-bit, `>>` logical; s is the side (0 to 3), `p = floor(j / 2)`:

```
v = (s * 4099 + p) * 2654435761 + 40503     mod 2^32   (0x9E3779B1, 0x9E37)
v = v ^ (v >> 15)
v = v * 2246822519                          mod 2^32   (0x85EBCA77)
v = v ^ (v >> 13)
track cell j is dark  iff  ((v ^ j) & 1) == 1
```

Side 0, pair 0: v = 0x00009E37, then 0x00009E36, 0xC6B2271A, 0xC6B4128B; bit 0 is 1, so track cell 0 is dark and 1
light. Track cells 0 to 15 of each side (1 is dark), the same in every ring:

| side | track cells 0 to 15 |
|---|---|
| 0 | `1010 1010 1001 1010` |
| 1 | `1010 0110 1010 0110` |
| 2 | `0101 0110 1010 0101` |
| 3 | `0110 1001 0101 0110` |

The track does not depend on n, the version or the payload; a ring sets only its length. The four sequences differ,
so the track alone tells the rotation and mirroring (8 hypotheses).

### 4.8 Margin

`focus.h:FOCUS_QUIET` = 2 modules of light, 2 pxm pixels, outside depth 0 on every side: 6 to 24 pixels in the 64
ring, 4 to 12 in the 128 and 2 to 6 in the 256 (pxm from 3.3). An encoder MUST paint it. The painted image is
(S + 4) pxm pixels a side. The margin lies outside the module grid, whose origin stays the symbol's outer corner
(4.1). `focus_paint_rgba(f, drive, rgba)` paints the drive as grey (6.8) inside `FOCUS_QUIET` modules of white, as RGBA
with R = G = B and alpha 255; the wasm exports it as `focus_tx_rgba` and the margin as `focus_quiet`.

### 4.9 One corner

LIZARD-64 in the 64 ring (S = 158) at fps 0, top-left corner, modules x 0 to 37 and y 0 to 17, from the layout's cells
(`ob_test_mesh_tables_out`). `#` dark, `.` light, `T` / `t` track cell dark / light, `W` / `w` word cell dark / light,
`g` guard, `P` picture. The other corners are this one reflected, except for the band's cells (4.6, 4.7). Word cells
aside, this corner is the same in every ring and for every picture.

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

The same symbol's band at depth 5 (depth 6 is identical), along 0 to 157 of each side:

```
top    .##..######....TTttTTttwwWWwwwwTTttTTttWWWWwwwwTTttttTTwwWWwwWWTTttTTttwwWWWWWWTTttttTTwwWWWWWWTTttttTTWWWWwwWWttTTttTTWWWWwwwwttTTTTttwwWWwwww....######..##.
right  .##..######....TTttTTttwwwwwwwwttTTTTttwwWWwwwwTTttTTttwwwwWWWWttTTTTttwwWWwwwwTTttttTTWWwwwwwwttTTttTTwwwwWWwwttTTTTttwwWWwwwwTTttttTTWWWWwwww....######..##.
bottom .##..######....ttTTttTTwwwwwwwwttTTTTttwwwwwwwwTTttTTttWWwwWWWWttTTttTTWWwwWWWWTTttTTttwwWWWWwwttTTTTttwwWWWWwwttTTttTTwwWWWWwwttTTttTTwwWWWWww....######..##.
left   .##..######....ttTTTTttWWWWWWwwTTttttTTwwWWwwWWttTTttTTwwWWwwWWttTTTTttWWwwWWWWttTTTTttwwwwwwWWTTttTTttWWWWWWWWttTTTTttwwWWwwwwttTTTTttwwWWWWww....######..##.
```

### 4.10 Mesh node sites (non-normative)

Nothing is painted for these. The layout defines a lattice of registration nodes, the same on both axes
(`thin_init`): `3.5 + 16k` for every k with `3.5 + 16k < S - 11.5`, then `S - 3.5`, B / 8 + 2 a side.

| ring B | S | node positions | count |
|---|---|---|---|
| 64 | 158 | 3.5 to 131.5 in steps of 16, then 154.5 | 10 |
| 128 | 286 | 3.5 to 259.5 in steps of 16, then 282.5 | 18 |
| 256 | 542 | 3.5 to 515.5 in steps of 16, then 538.5 | 34 |

The nodes a receiver measures lie on the border at depth 3.5, the middle of the gap's outer row (`OB_NODE_LINE`); the
four corner nodes are the corners of the registration quad (`acquire.c:corner_coords`). Word cells are left out of
any registration template: the fps byte is the sender's choice.

### 4.11 Settings in liblizard/src/ that are not the format (non-normative)

None: `liblizard/src/` paints this border alone, and its constants (`OB_RING_DEEP`, `OB_CELL`, `OB_BAND_RUN`,
`FOCUS_CP`, `OB_THIN_CORNER`) are plain definitions in `layout.h`, not build settings.

## 5. The format word

Normative except 5.4 and 5.5. Sources: `fmt.h`; `fmt.c` (`ob_fmt_side_cells`, `ob_fmt_bytes`, `ob_fmt_encode`,
`ob_fmt_bit`, `ob_fmt_decode`); `rs.c` (`init`, `rs_encode`); painted by `layout.c:ob_layout_set_fmt` from
`focus.c:fmt_paint`. The word is the only information the border carries.

### 5.1 Fields

The word is `bytes` bytes, `cw[0]` to `cw[bytes - 1]`. With B the band cells a side (4.6):

```
W = 4 floor(B / 8)        bytes = min(64, 4 floor(W / 8))
```

(`ob_fmt_side_cells`, `ob_fmt_bytes`, `OB_FMT_BYTES_MAX` = 64). In the three rings W = B / 2 (32, 64, 128) word cells a
side and bytes = W / 2 (16, 32, 64). A receiver has B from the ring it registered before it reads the word.

| byte | field | value |
|---|---|---|
| `cw[0]` | magic | `0x4C` (ASCII `L`) |
| `cw[1]` | version | sub-channels / 16, 2 to 64; 0, 1 and 65 to 255 are no version. The picture is n = `focus_n_for(16 * version)` (3.2), the blocks and their rates the profile's (3.1) |
| `cw[2]` | fps | whole frames a second the sender means to paint at, 1 to 255; 0 means not stated |
| `cw[3]` to `cw[bytes - 1]` | parity | Reed-Solomon, 5.2 |

- An encoder MUST write version = subch / 16.
- A decoder MUST accept every version from 2 to 64 in every ring, and MUST NOT reject a word for the ring it was read
  in.
- fps is the rate the sender intends, not the rate it achieves; no decode decision depends on it. A sender MAY change
  it between frames, which repaints only the word's cells (`focus.c:focus_fmt_fps`).
- Nothing else is in the word: the code is always LDPC at the profile's rates, the bit map always LINEAR (7.3), and
  everything about the transfer is in the header block (7.7).

### 5.2 Reed-Solomon (bytes, 3)

- Field GF(2^8): addition is XOR, multiplication modulo `p(x) = x^8 + x^4 + x^3 + x^2 + 1` (0x11D), primitive element
  alpha = 0x02 (`rs.c:init`).
- One code a ring over every word cell, with `nroots = bytes - 3` parity bytes:

| ring B | modules a side S | word cells a side W | code | nroots | distance | bytes a side |
|---|---|---|---|---|---|---|
| 64 | 158 | 32 | RS(16, 3) | 13 | 14 | 4 |
| 128 | 286 | 64 | RS(32, 3) | 29 | 30 | 8 |
| 256 | 542 | 128 | RS(64, 3) | 61 | 62 | 16 |

- Generator: roots alpha^0 to alpha^(nroots - 1),
  `g(x) = (x + 1)(x + alpha) ... (x + alpha^(nroots - 1))`, coefficients from x^nroots down to x^0, in hex
  (`rs_encode`):

  ```
  nroots 13: 01 89 49 e3 11 b1 11 34 0d 2e 2b 53 84 78
  nroots 29: 01 e4 c1 c4 30 aa 56 50 d9 36 8f 4f 20 58 ff 57 18 0f fb 55 52 c9 3a 70 bf 99 6c 84 8f aa
  nroots 61: 01 d1 fa 1a 7c 5f 3a 45 cb 4c 4d 52 4e a8 6e 87 b4 8e cf 94 9c 70 65 10 79 73 b2 91 a9 ad 6c
                03 7f 60 3b 48 fb 5b 0e 65 80 72 2b f5 ee 33 ab e4 f1 97 77 11 c0 5d 22 dd 5f ff 24 e5 9a c1
  ```

- Byte order: `cw[0]` is the highest-degree coefficient, `c(x) = cw[0] x^(bytes - 1) + ... + cw[bytes - 1]`.
- Systematic: `cw[3]` to `cw[bytes - 1]` are the coefficients of x^(nroots - 1) down to x^0 of
  `(cw[0] x^(bytes - 1) + cw[1] x^(bytes - 2) + cw[2] x^(bytes - 3)) mod g(x)`; equivalently `c(alpha^i) = 0` for
  i = 0 to nroots - 1.
- Each code is RS(255, 255 - nroots) shortened to `bytes`, distance nroots + 1.
- **Acceptance.** A decoder MUST NOT take a correction past `2e + f <= nroots` for e errors and f erasures (`rs.c`
  `solve` refuses one, and a locator whose degree is not its length), and MUST reject a corrected word whose `cw[0]` is
  not `0x4C` or whose version is outside 2 to 64.

**Test vectors**, `cw[0]` to `cw[bytes - 1]` in hex, read back out of the painted cells by `liblizard/test/vectors.mjs`
(`_focus_setup(n, subch, 2, 2B, 0)`, `_focus_fmt_fps_set(fps)`) and held by `liblizard/test/fmt_test.c`; the word does
not depend on n:

```
LIZARD-512   n 1024  ring  64  version 32  fps  24  RS(16, 3)  4c 20 18 8a 93 1b cb 15 4e 84 9d fc 68 bc 49 9e
LIZARD-1024  n 1536  ring  64  version 64  fps 240  RS(16, 3)  4c 40 f0 95 a1 ea 20 3a 12 ea 56 09 1b 50 c9 1d
LIZARD-576   n 1152  ring  64  version 36  fps  30  RS(16, 3)  4c 24 1e f7 ae 82 b0 6f 1b 6a 9d e7 f6 6d 48 aa
LIZARD-64    n  384  ring  64  version  4  fps   0  RS(16, 3)  4c 04 00 e5 57 34 bb 5b 7d 82 66 1f c4 4c 66 46
LIZARD-64    n  384  ring 128  version  4  fps  60  RS(32, 3)  4c 04 3c 76 18 ef 93 52 e5 25 c5 10 45 b7 22 df
                                                               9e 80 64 f9 f0 dd ad 60 a7 61 d0 30 c9 96 32 06
LIZARD-256   n  768  ring 256  version 16  fps 120  RS(64, 3)  4c 10 78 3d 45 04 e4 34 1b 70 33 13 d3 86 14 a0
                                                               a2 d8 94 f9 ca 7e 94 d8 81 95 95 61 fe c1 87 68
                                                               9d 90 17 a5 7e 59 ff eb 67 5e 0b 2d 07 4a 08 0b
                                                               48 ab 41 f8 c3 a9 c9 cd c8 8d ea 6b d9 8c 99 1c
```

### 5.3 Placement

- A cell shows one bit: dark is 1, light is 0.
- Byte k goes to side `k mod 4`, on word cells `8 floor(k / 4)` to `8 floor(k / 4) + 7` of that side, most
  significant bit first (`ob_fmt_bit`). Equivalently, word cell c of side s shows bit `7 - (c mod 8)` of byte
  `4 floor(c / 8) + s`.

| side | word cells 0 to 7 | 8 to 15 | 16 to 23 | 24 to 31 | 32 to 39 | 40 to 47 | 48 to 55 | 56 to 63 |
|---|---|---|---|---|---|---|---|---|
| 0 top | `cw[0]` magic | `cw[4]` | `cw[8]` | `cw[12]` | `cw[16]` | `cw[20]` | `cw[24]` | `cw[28]` |
| 1 right | `cw[1]` version | `cw[5]` | `cw[9]` | `cw[13]` | `cw[17]` | `cw[21]` | `cw[25]` | `cw[29]` |
| 2 bottom | `cw[2]` fps | `cw[6]` | `cw[10]` | `cw[14]` | `cw[18]` | `cw[22]` | `cw[26]` | `cw[30]` |
| 3 left | `cw[3]` | `cw[7]` | `cw[11]` | `cw[15]` | `cw[19]` | `cw[23]` | `cw[27]` | `cw[31]` |

  The 64 ring uses the first four columns, the 128 ring all eight, the 256 ring sixteen (`cw[4j]` to `cw[4j + 3]` at
  word cells 8j to 8j + 7, to `cw[60]` to `cw[63]` at 120 to 127).
- Word cells are numbered along the side as its modules count (4.1). Every word cell carries a bit of the codeword:
  no copies and no spare cells. An encoder MUST paint every word cell from the codeword. The layout flags each one
  `CELL_WORD` (`layout.h`).

LIZARD-64 in the 64 ring at fps 0, the 32 word cells of each side in order:

| side | word cells 0 to 31 |
|---|---|
| 0 | `0100 1100 0101 0111 0111 1101 1100 0100` |
| 1 | `0000 0100 0011 0100 1000 0010 0100 1100` |
| 2 | `0000 0000 1011 1011 0110 0110 0110 0110` |
| 3 | `1110 0101 0101 1011 0001 1111 0100 0110` |

LIZARD-64 in the 128 ring at fps 60, the 64 word cells of each side in order:

| side | word cells 0 to 63 |
|---|---|
| 0 | `0100 1100 0001 1000 1110 0101 0100 0101 1001 1110 1111 0000 1010 0111 1100 1001` |
| 1 | `0000 0100 1110 1111 0010 0101 1011 0111 1000 0000 1101 1101 0110 0001 1001 0110` |
| 2 | `0011 1100 1001 0011 1100 0101 0010 0010 0110 0100 1010 1101 1101 0000 0011 0010` |
| 3 | `0111 0110 0101 0010 0001 0000 1101 1111 1111 1001 0110 0000 0011 0000 0000 0110` |

### 5.4 Reading the word (non-normative)

The reference reader (`acquire.c:ob_thin_read_fmt`, then `fmt.c:ob_fmt_decode`) runs after registration, in the ring
that registered, before the picture. Points are continuous module coordinates mapped through the registration and
sampled bilinearly as luminance in 0 to 1. A point at along t and depth d is `(t, d)` on the top, `(S - d, t)` on the
right, `(t, S - d)` on the bottom and `(d, t)` on the left (`side_xy`; d continuous, so the band's middle is depth
6.0).

1. **Contrast.** On every side, for t = 15, 31, 47, ... while `t < S - 15`, sample the ring at (`t + 0.5`, 2.0) and the
   gap at (`t + 0.5`, 4.0); `con` is the mean gap sample minus the mean ring sample. Below 0.08 the reader gives up.
2. **Local level.** For each Manchester pair p of the side's track, `mid[p]` is the mean of its two cells' samples at
   their centres (depth 6.0); a side has `npair` = B / 4 pairs.
3. **Cells.** Word cell i of side s, in band cell b, is sampled at along `15 + 2b + 1`, depth 6.0. With
   `a = floor(2b / 8)` its level is the mean of `mid[p]` for p from `max(0, a - 3)` to `min(npair - 1, a + 3)`;
   `q[s][i] = sample - level`.
4. **Threshold.** From `ref = 0`, up to four times: split the 4W values of q at ref (`q >= ref` the lighter part),
   stop if either part has fewer than W / 2, else set ref to the midpoint of the two parts' means.
5. **Soft value.** `z[s][i] = clamp((ref - q[s][i]) / con, -0.5, 0.5)`, positive for dark.

Decoding: byte k is built from the 8 soft values of side `k mod 4` from cell `8 floor(k / 4)`, MSB first, 1 where
`z > 0`, its confidence the smallest `|z|` of its bits. First errors-only RS (up to floor(nroots / 2) errors); then the
least confident bytes erased (the lower index first on a tie), 2, then 4, and so on up to nroots - 3 (10, 26, 58),
decoding errors and erasures within `2e + f <= nroots` at each step. Each result is taken only if it passes 5.2's
acceptance; the first that passes is the word.

A read of garbage (soft values uniform in -0.5 to 0.5) is accepted at 9.3e-5 (the steps) x 1/256 (the magic) x
64/256 (the version) = 9.1e-8 a read in every ring (`liblizard/test/fmt_test.c` `p_pass`, exact; 0 of 20,000 measured
a ring). The word reads with any two sides faded in every ring and with three in the 128 and 256 rings. In the 64 ring
a read whose side 0 is clean and the other three garbage keeps the real magic, and is accepted wrongly at 2.3e-5.

### 5.5 What the ring bootstraps (non-normative)

The rule both reference receivers follow (10.1, 10.2):

- The finder registers the symbol against the three rings' layouts; the ring that registered is the frame's.
- The word is read in that ring, and its version names the sub-channels and so n. The frame is finished at that
  picture, in that ring.
- A receiver holds the version of the last word it read and decodes a frame whose word does not read at that held
  configuration, in the ring the frame registered in. This is the one state a receiver carries between frames.
- With no word held, a frame whose word does not read is not decoded. No frame is registered again at another ring.

## 6. The picture

Normative except where marked. Sources: `focus.c` (`init`, `ws_init`, `focus_encode`, `turn_pack`, `clip_row`,
`resample_init`, `resample`, `focus_paint_rgba`, `focus_n_for`), `focus.h`, `fft.h`.

### 6.1 Coordinates

- Picture sample (x, y), 0 <= x, y < n: x to the right, y downwards, row 0 the top row; the picture, the drive and the
  paint are row-major, top row first.
- u is the spatial frequency along x and v along y, in cycles across the picture.
- Module coordinates are those of 4.1. The picture occupies modules [15, 15 + span) on each axis.

A transmitter that paints y upwards, or swaps u and v, paints the mirror image, whose spectrum puts each symbol on
another coefficient.

### 6.2 Picture size and scale

n follows from subch (3.2), span from the ring, and pxm, scale and the resampled square from both (3.3, whose table
lists every pair).

### 6.3 The coefficients

The candidates are the upper half-plane without DC and without the Nyquist row and column:

```
-n/2 < u < n/2,   0 <= v < n/2,   and not (v = 0 and u <= 0)
```

sorted by u^2 + v^2 ascending, then v ascending, then u ascending (`focus.c:by_freq`): C_0, C_1, ... A frame of subch
sub-channels uses C_0 to C_(320 subch - 1); every other coefficient of the half-plane is zero. The order does not
depend on n (3.2): every format's coefficients are a prefix of LIZARD-1024's. Its first 24 entries, as (u, v):

```
(1,0) (0,1) (-1,1) (1,1) (2,0) (0,2) (-2,1) (2,1) (-1,2) (1,2) (-2,2) (2,2)
(3,0) (0,3) (-3,1) (3,1) (-1,3) (1,3) (-3,2) (3,2) (-2,3) (2,3) (4,0) (0,4)
```

**Sub-channels.** Sub-channel j is C_(320 j) to C_(320 j + 319) (`FOCUS_SUB` = 320); higher j is higher frequency.
The last coefficient of some sub-channel counts:

| sub-channels | last coefficient (u, v) | u^2 + v^2 | radius |
|---|---|---|---|
| 1 | (9, 11) | 202 | 14.2 |
| 2 | (9, 18) | 405 | 20.1 |
| 8 | (20, 35) | 1625 | 40.3 |
| 16 | (57, 3) | 3258 | 57.1 |
| 64 | (6, 114) | 13032 | 114.2 |
| 128 | (145, 71) | 26066 | 161.4 |
| 256 | (218, 68) | 52148 | 228.4 |
| 512 | (25, 322) | 104309 | 323.0 |
| 1024 | (26, 456) | 208612 | 456.7 |

A frame's top coefficient lies within 0.07 of R on every format; n / r_top runs from 3.01 to 4.49. Block 0, a 7/8
block in every format, runs to C_2247, (37, 8), radius 37.9.

### 6.4 Blocks

Block b's rate and place come from the rate profile (3.1): it takes L_b slots, 4496, 5152, 5760 or 7680 by its rate
(3.4), from frame slot S_b, the sum of the slots of the blocks before it (S_0 = 0): coefficients C_(S_b / 2) to
C_((S_b + L_b) / 2 - 1), every S_b and L_b being even (a multiple of 16), so no coefficient is shared by two blocks. A
block runs across sub-channels as its slots fall: its first and last sub-channels may hold the end of the block before
or the start of the block after. Its sent codeword bits fill the first 4480, 5120, 5760 or 7680 of its slots, and the
last 16, 32, 0 or 0 are its tail (7.3). The slots from S_B(V) (the end of the last block) to 640 subch - 1 are the
padding.

### 6.5 Symbols

Frame slot 2s lies on the real axis of coefficient C_s and slot 2s + 1 on its imaginary axis, 0 <= s < 320 subch;
block b's slot k is frame slot S_b + k. With x the frame's slot bits (7.3: a block's slots, and on the padding
x_i = w[i], the whitening alone), coefficient s carries

```
H(C_s) = (a_j / sqrt 2) * ( (1 - 2 x_(2s)) + i (1 - 2 x_(2s+1)) ),      j = floor(s / 320),  i = sqrt(-1)
```

that is, bit 0 gives + and bit 1 gives -, on each axis independently (`focus_encode`; the magnitude
`0.70710678f * a_j` in float32). a_j is sub-channel j's amplitude:

```
a_j = 10^( -t * j / (20 * (subch - 1)) )        t = tilt in dB; t = 0 gives a_j = 1 for every j
```

(`focus.c:ws_init`). Every sender uses t = 0 (section 2, Not the format).

### 6.6 The transform

The picture before clipping is the real inverse DFT of the half-plane and its conjugate mirror:

```
p(x, y) = sum over s < 320 subch of  2 Re[ H(C_s) exp( +2 pi i (u_s x + v_s y) / n ) ]
```

Equivalently: set H(-u, -v) = conj(H(u, v)) (indices mod n), zero everywhere else, and take the unscaled inverse 2D
DFT with kernel exp(+2 pi i (u x + v y) / n); no 1 / n^2 factor (`fft.h`). The reference evaluates it in float32 over
the half-plane (`focus_encode`, `turn_pack`, `fft.c:fft_cols`) from one twiddle table W^j = exp(-2 pi i j / n),
j < n / 2, computed in double and stored as float32 (conjugated for the inverse). For n a power of two the FFT is radix
2; otherwise its radix-3 stages come first (one for n = 3 x 2^k, two for 1152 = 9 x 2^7), then its radix-5 stage (one
for 1280 = 5 x 2^8), then the radix-2 stages, on the input in digit-reversed order in those radices (`fft.c`
`stage_list`, `plan_for`, `rows3`, `rows5`, `fft_cols_mixed`). With b and c already multiplied by their twiddles, the
radix-3 butterfly is

```
t1 = b + c,   t2 = b - c,   h = a - 0.5 t1
a' = a + t1,  b' = h - i s k t2,  c' = h + i s k t2
k = 0.8660254 (sqrt 3 / 2 in float32),  s = +1 forward, -1 inverse
```

and with b, c, d and e multiplied by theirs, the radix-5 butterfly, every product and sum in float32 in this order:

```
t1 = b + e,   t2 = c + d,   t3 = b - e,   t4 = c - d
m1 = (a + c1 t1) + c2 t2,   m2 = (a + c2 t1) + c1 t2
n1 = s1 t3 + s2 t4,          n2 = s2 t3 - s1 t4
a' = (a + t1) + t2,  b' = m1 - i n1,  e' = m1 + i n1,  c' = m2 - i n2,  d' = m2 + i n2
c1 = cos(2 pi / 5), c2 = cos(4 pi / 5), s1 = s sin(2 pi / 5), s2 = s sin(4 pi / 5), each in float32
(0.30901699, -0.80901699, 0.95105652, 0.58778525),  s = +1 forward, -1 inverse
```

**Normalisation.** P = sum over s of |H(C_s)|^2 = 320 sum over j of a_j^2: every coefficient of the frame is sent, the
padding's too (6.5). By Parseval the mean of p^2 is 2P, so the
picture's rms is

```
sigma = sqrt(2 P)          at t = 0:  sigma = sqrt(640 subch)
```

The reference uses this value and never measures the rms. At LIZARD-64, sigma = sqrt(40960) = 202.39.

### 6.7 Clipping and grey

```
L = c * sigma,     c = 2 (the clipping ratio)
g(x, y) = 1/2 + clamp(p(x, y), -L, L) / (2 L)
```

g is in [0, 1], 0 black and 1 white. The reference computes the level in float32 as `lim = 2 * c * sqrtf(0.5 * 320
subch)` at t = 0, or `2 * c * sqrt(0.5 * P)` with P summed in double when t > 0, and g as `0.5f + clamp(p) * (0.5f /
lim)` (`focus_encode`, `clip_row`). Every sender uses c = 2; a receiver does not depend on it.

### 6.8 Resampling onto the painted grid

Drive pixel (X, Y) covers module coordinates [X / pxm, (X + 1) / pxm), centre (X + 0.5) / pxm.

**The resampled square.** Modules [12, span + 18) on both axes are the resampler's alone: the picture and the 3-module
guard round it (`FOCUS_CP` = 3; `focus_encode` takes cp = min(FOCUS_CP, margin - OB_THIN), 3 in the format), a square
of q = (span + 6) pxm pixels from pixel (12 pxm, 12 pxm). The outer border outside it is painted from the layout
(section 4).

**Where each pixel reads the picture.** Along each axis, the pixel with local index t = X - 12 pxm (0 <= t < q) reads
the picture at the continuous sample coordinate

```
w(t) = (t + 0.5) * sc / pxm - 3 * sc - 0.5,        sc = n / span,   evaluated in double
```

that is w = ((X + 0.5) / pxm - 15) sc - 0.5, which places **sample x at module coordinate 15 + (x + 0.5) / sc** on
both axes. Where pxm = sc every pixel lands on a sample and the resampler copies. In the nine pairs where pxm > sc
(3.3: n = 256 in the 64 ring, 256, 384 and 1152 in the 128 ring, 256, 384, 768, 1152 and 1280 in the 256 ring) the
picture is enlarged through the taps below.

**The kernel.** Lanczos-3:

```
K3(d) = 3 sin(pi d) sin(pi d / 3) / (pi d)^2     for 0 < |d| < 3,     K3(d) = 0 for |d| >= 3
```

For each t (`resample_init`):

- If |w - r| < 1e-6, with r = floor(w + 0.5), the pixel lands on a sample: one tap, weight 1, on sample r (in the
  six-tap form below, i0 = r - 2 and weights (0, 0, 1, 0, 0, 0)).
- Otherwise the taps are the six samples i0 + k, k = 0 to 5, i0 = floor(w) - 2, with weights K3(w - (i0 + k)) divided
  by the sum of the six, computed in double and stored as float32.

Sample indices are taken mod n: the picture is read as periodic, which paints the guard (4.4). The same weights serve
both axes.

**The passes.** Vertical first, then horizontal, each accumulated in float32 in tap order; only the horizontal result
is clamped:

```
T(ty, x) = sum over l = 0..5 of  wy_l * g(x, (i0(ty) + l) mod n)             for every picture column x
D(tx, ty) = clamp( sum over k = 0..5 of  wx_k * T(ty, (i0(tx) + k) mod n),  0, 1 )
```

D is the drive at pixel (12 pxm + tx, 12 pxm + ty). Clamping T as well paints a different picture; in a copy neither
clamp acts. The anchors are painted over D afterwards (6.13).

**Painting.** Each drive value becomes one 8-bit grey level, round half up:

```
G = floor(D * 255 + 0.5), clamped to [0, 255];     R = G = B = G, alpha 255
```

(`focus_paint_rgba`, as float32 `D * 255.0f + 0.5f` truncated). The outer border paints only 0 and 255; the picture
and its guard use all 256 levels.

**How exact.** The reference evaluates 6.5 to 6.8 in float32, so an implementation in double precision meets its grey
bytes only to that rounding; a conforming transmitter follows the formula (section 2, Conformance).

### 6.9 Where the receiver samples

Picture sample (x, y) is at module coordinates

```
( 15 + (x + 0.5) * span / n,   15 + (y + 0.5) * span / n )
```

so the grid starts at 15 + 0.5 / scale and steps by 1 / scale modules (`focus_finish_ext`: `g0 = margin + 0.5f /
scale`, `step = 1.0f / scale`, float32; `wasm.c:focus_grid_out`). A receiver that takes the forward DFT of an n x n
grid MUST take it at these points: a grid displaced by (dx, dy) samples turns coefficient (u, v) by
2 pi (u dx + v dy) / n.

The receiver's transform is the unscaled forward DFT

```
Y(u, v) = sum over x, y of s(x, y) exp( -2 pi i (u x + v y) / n )
```

For an unclipped picture Y(C_s) = n^2 H(C_s) / (2L); clipping and the channel scale it, to first order, by a positive
gain a sub-channel and add noise. So a receiver MUST read Re Y(C_s) > 0 as bit 0 of frame slot 2s and Re Y(C_s) < 0 as
bit 1, and Im Y(C_s) the same way for slot 2s + 1. Neither the clip level, the tilt nor the capture's gain is
signalled: a receiver estimates each sub-channel's amplitude and noise itself (8.5).

### 6.10 What the receiver removes before reading (non-normative)

A brightness ramp across the picture lands on the coefficients nearest zero frequency, which carry block 0. The
reference receiver removes the least-squares surface in 1, x, y, xy, x^2, y^2 (`focus_finish_ext`, basis in `ws_init`).
With c_i = (2 i + 1 - n) / n and q_i = c_i^2 - mean(c^2), i = 0 to n - 1, the five non-constant functions c(x), c(y),
c(x) c(y), q(x), q(y) are orthogonal on the grid and to the constant, so each coefficient is one projection of the
samples s:

```
kx  = sum s(x,y) c_x / (n Sc)        ky  = sum s(x,y) c_y / (n Sc)        kxy = sum s(x,y) c_x c_y / Sc^2
kxx = sum s(x,y) q_x / (n Sq)        kyy = sum s(x,y) q_y / (n Sq)
Sc  = sum c_i^2 = (n^2 - 1) / (3 n)  Sq  = sum q_i^2
```

With X1(k) = sum c_i exp(-2 pi i k i / n) and X2(k) the same for q, the surface comes out in the spectrum:

```
Y(u, v) -= kxy * X1(u) * X1(v)                         every (u, v)
Y(u, 0) -= n * (kx * X1(u) + kxx * X2(u))              the row v = 0
Y(0, v) -= n * (ky * X1(v) + kyy * X2(v))              the column u = 0
```

DC carries nothing and X1(0) = X2(0) = 0, so DC is untouched.

### 6.11 Grey levels, gamma and palette

- The reference paints 256 levels; nothing in the format needs a fine grey scale.
- A sender writes the levels as they are, with no gamma pre-compensation; the panel's and the camera's curves are part
  of the channel. The reference receiver reads a grey picture at gamma 1.
- The data stays in luminance; two brand colours, where used, only move where that luminance sits (section 2, item 1).

### 6.12 Test vectors

Computed with the codec's wasm through `liblizard/sim/ob.mjs`: `_focus_setup(n, subch, 2, 128, 0)` (clip 2, span 128:
the 64 ring; no tilt), `_focus_dbg(1)`, `_focus_tx` on a payload of blocks x 476 zero bytes, then the drive (float32,
S pxm pixels square, row-major), the slot bits from `_focus_dbg_sym` (int8 a slot, +1 for bit 0, -1 for bit 1, frame
slot order) and the coefficient positions from `_focus_pos_ptr`. Grey bytes are G of 6.8. The word states fps 0 and
the count is 0 (`_focus_parity_set` not called). The slot bits do not depend on the ring or on n.

**P1, the order.** `_focus_pos_ptr` holds 320 subch int32, entry i for C_i, an offset into the reference's strip
layout. With `bw = min(n / 2, 64)`:

```
p = floor(v / bw) * n * bw + ((u + n) mod n) * bw + (v mod bw)
```

decoded as `v = floor(p / (n bw)) bw + (p mod bw)`, `u' = floor(p / bw) mod n`, u = u' when u' < n / 2, else u' - n.
Decoded, the entries equal 6.3's sort at every coefficient of LIZARD-64, -144, -256 and -576, each at its n.

**P2, LIZARD-64 (n = 384, 7 blocks) in the 128 ring, every payload byte zero.** pxm = 2, scale 1.5, so the picture is
enlarged by 4/3 through the taps of 6.8.

- slot bits, one byte (0 or 1) a slot, 40,960 bytes: P3's (the slots do not depend on the ring), SHA-256
  `04e27f794b080a1ebc30d409405851e011e18ac904c713fa75f9bb6917799353`
- the same packed 8 to a byte, first slot in the most significant bit, first 16 bytes:
  `00043e000efd03e0f0ffffcc000c0c03`
- sigma = 202.385773, L = 404.771545
- drive 572 x 572 (S = 286, pxm = 2; scale = 1.5, so the picture is upscaled by 4/3 through the taps of 6.8). Drive
  values at row 30, columns 30 to 33 (the picture's first pixel row and column): 0.4950773, 0.4724883, 0.6975312, 0.9888784
- grey bytes at row 30, columns 30 to 45: `126 120 178 252 255 209 139 131 188 242 231 160 91 65 89 129`
- grey bytes at row 345, columns 345 to 360: `113 94 95 99 110 140 183 217 207 151 75 18 5 35 118 211`
- the anchors (6.13): a = 4 modules, 8 pixels, from modules 31, 68, 104, 141, 178, 214, 251 on either axis. Grey bytes at
  row 64 (anchor (0, 0)'s third pixel row), columns 60 to 71, the square at columns 62 to 69:
  `62 111 255 255 0 0 0 0 255 255 106 114`
- grey bytes of the resampled square, pixels [24, 548) on both axes (modules [12, 274)), row by row, 274,576 bytes:
  SHA-256 `8aeda0414b3b27d7711a5acd994531a16c14acffb8f0a4f025cb65a59d4de4e3`; 4897 of them are 0 and 6900 are 255
- grey bytes of the whole drive, border included, 327,184 bytes: SHA-256
  `27c1a4bdd0794e7b955a94a7336f78a51e1186b0a38b17e772ec05d8b12ddb77`

**P3, LIZARD-64 (n = 384, 7 blocks) in the 64 ring, every payload byte zero.** pxm = 3, scale 3: a copy. Its slot bits
are P2's.

- slot bits, one byte a slot, 40,960 bytes: SHA-256 `04e27f794b080a1ebc30d409405851e011e18ac904c713fa75f9bb6917799353`
- drive 474 x 474 (S = 158, pxm = 3; scale = 3, so the picture is copied one pixel a sample); the picture starts at
  pixel 45
- grey bytes at row 45, columns 45 to 60: `136 154 247 255 163 126 206 249 176 91 81 119 148 139 109 122`
- the anchors (6.13): a = 2 modules, 6 pixels, from modules 23, 41, 60, 78, 96, 115, 133 on either axis. Grey bytes at
  row 71 (anchor (0, 0)'s third pixel row), columns 66 to 77, the square at columns 69 to 74: `62 96 126 255 0 255 255 0 255 86 96 91`
- grey bytes of the resampled square, pixels [36, 438) (modules [12, 146)), 161,604 bytes: SHA-256
  `35967d2409d3eff8c40d94b1e449bd6a70c1bbf432d532c6074c6600f7209f3b`
- grey bytes of the whole drive, 474 x 474, 224,676 bytes: SHA-256
  `db6422590e50b169cfee2561e925bec21be730286e32b3e68f11168aecf7a1a6`. The border holds the RS(16, 3) word (5.2)

**Digests over the ladder.** `node test/vectors.mjs [ob.mjs]`, run from `liblizard/`, checks 5.2, P2, P3, 7.4 and 9.3
against a build and prints three digests to diff two builds, each over LIZARD-64 to -1024 in steps of 16 in the 64
ring: every format's cell map at five rates (with its layout dimensions), its drive on a fixed payload, and the P1
offsets: cells `7e4f838d46272fd7`, drives `6d6a18a74b53b038`, order `830c1e79ca55f85b`.

### 6.13 Anchors (normative)

Every symbol carries 49 anchors: squares painted over the picture on a 7 x 7 grid, whose middles give a receiver the
geometry inside the border.

- **Geometry.** In ring B (span = 2B) a square is a = span / 64 modules a side: 2, 4 and 8 in the 64, 128 and 256
  rings. Square (i, j), 0 <= i, j < 7, covers module columns [m0(i), m0(i) + a) and rows [m0(j), m0(j) + a), with
  m0(i) = floor(15 + span (i + 0.5) / 7 - a / 2 + 0.5) in double precision, the same on both axes
  (`acquire.c:ob_anchor_m0`). The squares cover 1.2% of the picture.

| ring B | a, modules | m0(0) to m0(6) |
|---|---|---|
| 64 | 2 | 23, 41, 60, 78, 96, 115, 133 |
| 128 | 4 | 31, 68, 104, 141, 178, 214, 251 |
| 256 | 8 | 48, 121, 194, 267, 340, 413, 486 |

- **Pattern.** A square is a pxm drive pixels a side, never under 6 (3.3). Its drive pixel (x, y), 0 <= x, y < a pxm,
  with h = a pxm / 2 and d = max(|x + 0.5 - h|, |y + 0.5 - h|) / h, is light (1.0) where d < 0.25 or d >= 0.6 and dark
  (0.0) otherwise: a light middle, a dark ring, a light rim.
- **Painting.** After the picture is resampled into the drive (6.8), every square overwrites the drive pixels it
  covers. No slot, block, coefficient or word bit changes for it.
- **Erasing (a receiver MUST).** Before the detrend (6.10) and the transform, on every frame, whether or not it found
  the anchors, a receiver sets every picture sample a square touches and 2 samples beyond it on each side to the
  picture's mean: on either axis the samples from floor((m0(i) - 15) scale) - 2 to ceil((m0(i) + a - 15) scale) - 1 + 2,
  clamped to [0, n - 1], for each i, the 49 products of the 7 ranges on the two axes (`focus.c:focus_finish_ext`). The
  mean need not be exact (the reference takes it over every eighth sample each way).
- **Reading them** is the receiver's (10.1, 10.2).

## 7. Blocks, the bit mapping and the transfer

Normative except where marked. Sources: `focus.c` (`init`, `focus_encode`, `whiten_fill`, `perm_fill`, `map_out`,
`map_in`, `focus_finish_ext`), `focus.h`, `layout.c:ob_crc32`; for the transfer (7.5 to 7.10) `xfer.h`, `xfer.c` and
their wasm exports in `wasm.c`.

A frame of version V carries B(V) blocks, each one LDPC codeword with its own CRC-32, independent of the others. In
this section and section 8, n, k and m are the block code's length, information length and parity count, not the
picture size.

### 7.1 The block

| bytes | content |
|---|---|
| 0 to 3 | block id, uint32, little-endian (7.5) |
| 4 to 475 | 472 payload bytes: one Wirehair block, the header block or a manifest block |

The picture codec treats the 476 bytes as opaque. A frame's payload is 472 B(V) bytes.

**Information word**, k bits (`focus.c:focus_encode`): 3968 at 7/8, 3840 at 3/4, 2/3 and 1/2.

| bits | content |
|---|---|
| 0 to 3807 | block bytes 0 to 475, bit i the bit 7 - (i mod 8) of byte floor(i / 8): most significant bit first |
| 3808 to 3839 | CRC-32 of bytes 0 to 475, bit 3808 + i the bit 31 - i of the CRC: most significant bit first |
| 3840 to k - 1 | at 7/8 alone, 128 zeros. A decoder knows them (the reference feeds them as certain zeros); the 3/4, 2/3 and 1/2 codes' k is 3840 |

**CRC-32** (`layout.c:ob_crc32`) is that of IEEE 802.3 and zlib: polynomial 0x04C11DB7 reflected (0xEDB88320),
register initialised to 0xFFFFFFFF, result complemented; the CRC-32 of ASCII `123456789` is 0xCBF43926. It covers the
id and the payload. The id is little-endian; the CRC, read as bytes 476 to 479 of the 480-byte word, is big-endian.

**Acceptance.** A receiver MUST hand a block on only when the CRC-32 it computes over the 476 decoded bytes equals the
32 decoded CRC bits; LDPC convergence alone is not acceptance. The reference requires a zero syndrome and then the CRC.

### 7.2 Where a block lies in the frame

Block b (0 to B(V) - 1), of L_b slots from frame slot S_b (6.4), owns frame slots S_b to S_b + L_b - 1 and coefficients
S_b / 2 to (S_b + L_b) / 2 - 1: block slot i is the real part of coefficient (S_b + i) / 2 when i is even and the
imaginary part of coefficient (S_b + i - 1) / 2 when i is odd, frame slot S_b + i. A slot bit 0 is sent as a positive
component (6.5). In LIZARD-656 the blocks start at frame slots 0, 4496, ..., 157,360 (the 7/8 tier's 36), 161,856 to
264,896 by 5152 (the 3/4 tier's 21), 270,048 to 327,648 by 5760 (the 2/3 tier's 11) and 333,408 to 410,208 by 7680 (the
1/2 tier's 11); slots 417,888 to 419,839 are the padding.

### 7.3 The bit map

A codeword has n bits, of which np are never sent: codeword bits k - np to k - 1, the 7/8 code's zero information bits
past the CRC (np = 128 at 7/8, 0 otherwise, 8.6). Sent bit t is codeword bit s(t) = t for t < k - np and t + np
otherwise; the nt = n - np sent bits fill the first nt of the block's L_b slots, and the rest is the tail. The format's one map is LINEAR
(`focus.h:FOCUS_BITMAP` = 3). For block b, its codeword c (section 8), the whitening w and the pilot bit q:

```
slot[i] = (i < nt ? c[s((st * i) mod nt)] : q) XOR w[S_b + i],    i = 0 .. L_b - 1
```

Frame slot i of the padding (S_B(V) <= i < 640 subch) is w[i]: a coefficient there carries the whitening alone.

**The permutation** (`focus.c:perm_fill`). The stride st is floor(0.3819660113 nt + 0.5) (1/phi^2), then raised by 1
until gcd(st, nt) = 1. Sent bit t (codeword bit s(t)) lies on slot st^-1 t mod nt.

| rate | nt | np | st | st^-1 mod nt | tail slots T | tail coefficients of the block |
|---|---|---|---|---|---|---|
| 7/8 | 4480 | 128 | 1711 | 2511 | 16 | 2240 to 2247 |
| 3/4 | 5120 | 0 | 1957 | 2093 | 32 | 2560 to 2575 |
| 2/3 | 5760 | 0 | 2201 | 3881 | 0 | none |
| 1/2 | 7680 | 0 | 2933 | 1757 | 0 | none |

At 3/4, perm(0) to perm(5) are 0, 1957, 3914, 751, 2708, 4665.

**The pilots.** The tail slots carry no code bit: q is a bit of the count c of the picture painted (9.1), bit 0 of c
on an even block (b even, b the block's place in the frame) and bit 1 on an odd one. Where q is 1 the tail's
coefficients are painted inverted. The LDPC ignores them. Only 7/8 and 3/4 blocks have tails; every format has a block
with a tail of each parity (its block 0 is a 7/8 block and its block 1 a 7/8 or 3/4 block). A tail lies at the end of
its block and runs across two sub-channels where its block's last slots do (a 3/4 tail can: 5152 is not a multiple of
640).

A receiver reads, per block b with a tail, the T axis values y_i of the tail (frame slot S_b + nt + i: the real part of
coefficient (S_b + nt + i) / 2 when i is even, the imaginary part when odd), after the transform and detrend, against
the known signs, each in units of the rms axis value of its own sub-channel j (m2_j the mean |y|^2 over that
sub-channel's 320 coefficients):

```
r_b = (1 / T) sum_i (1 - 2 w[S_b + nt + i]) y_i / sqrt(m2_j / 2),    j = floor((S_b + nt + i) / 640)
```

A frame has two readings: r, the mean of r_b over the even blocks, and r2, over the odd ones. A reading over one 7/8
block averages 8 coefficients, one over a 3/4 block 16: LIZARD-64's r2 is two 7/8 blocks' 16, its r two 7/8
blocks' and the 3/4 block's 32. A capture of one picture
reads r = +1 where bit 0 of its count is 0 and -1 where it is 1, and r2 the same by bit 1 (scaled by sqrt(A2 / m2) at a
signal-to-noise ratio A2 / N). A capture of picture c that also holds a share a of the picture before and b of the one
after reads, in size, r = 1 - 2a - 2b, and r2 = 1 - 2a where c is even and 1 - 2b where c is odd (where samples come
from one picture or the other; a uniform blend reads (1 - 2f) / sqrt(1 - 2f + 2f^2) for share f). The readings say
which neighbour a capture holds, which a receiver steering its camera's phase uses (10). A reading's standard error is
its blocks' spread of r_b over the square root of their count. Both reference decoders report the two readings
(`focus.h:focus_pilot`).

**The whitening** (`focus.c:whiten_fill`) is PRBS-23 on x^23 + x^18 + 1, started from all ones, not inverted:

```
w[i] = w[i - 23] XOR w[i - 18],    w[j] = 1 for j < 0
```

As a 23-bit register: s = 0x7FFFFF, and for each i, w[i] = ((s >> 22) XOR (s >> 17)) AND 1, then
s = ((s << 1) OR w[i]) AND 0x7FFFFF. The sequence starts with 18 zeros and then 5 ones; its first 128 bits are
`00003e000ffc03e0f8ffffce000c1c03` (hex, the first bit of each digit most significant). w is indexed by frame slot,
S_b + i, and restarts at slot 0 in every frame; it depends on nothing else (not the payload, frame, format or n).
Its period, 2^23 - 1, exceeds the largest frame (655,360 slots).

### 7.4 Test vector

LIZARD-64 (n = 384, 7 blocks: four 7/8 blocks on frame slots 0 to 17,983, a 3/4 block on 17,984 to 23,135, a 2/3 block
on 23,136 to 28,895, a 1/2 block on 28,896 to 36,575, the padding on 36,576 to 40,959). Block b is the id b as uint32
little-endian, then 472 payload bytes, byte j = j mod 256:

```
block 0: 00 00 00 00 | 00 01 02 ... fe ff 00 01 ... d7
block 1: 01 00 00 00 | 00 01 02 ... fe ff 00 01 ... d7
...
block 6: 06 00 00 00 | 00 01 02 ... fe ff 00 01 ... d7
```

Bit strings are hex, four bits a digit, the first bit the most significant. The 7/8 blocks:

| | block 0 (7/8) | block 1 (7/8) | block 2 (7/8) | block 3 (7/8) |
|---|---|---|---|---|
| first frame slot | 0 | 4,496 | 8,992 | 13,488 |
| CRC-32 of bytes 0 to 475 | 0x4AB56DA5 | 0x84A6ED54 | 0x0DE36A06 | 0xC3F0EAF7 |
| codeword bits 3800 to 3839 (last payload byte, CRC) | `d74ab56da5` | `d784a6ed54` | `d70de36a06` | `d7c3f0eaf7` |
| codeword bits 3840 to k - 1 | 128 zeros | 128 zeros | 128 zeros | 128 zeros |
| codeword bits k to k + 127 (parity p_0 to p_127) | `e9266e19f9d8d7cc96857e9f8fc572c9` | `e6d991e606272833697a8160703a8d36` | `e9266e19f9d8d7cc96857e9f8fc57236` | `e6d991e606272833697a8160703a8dc9` |
| whitening under its slots 0 to 127 (frame slots S_b to S_b + 127) | `00003e000ffc03e0f8ffffce000c1c03` | `602e8acbffa7681694d5881cc9170bd7` | `ab494584c3ea397d5a2dac3f3057b275` | `4531d3c6175609fb2e6d3dc7950a6a68` |
| slots 0 to 127 | `75327e236757d37a0084f2c064a9db97` | `1518eac8960db08464eec512ada2cc43` | `de7f05a7aa40e9efa256a13154f265e1` | `3003b3c57ffdd161de5670c9f1bfbdfc` |
| the tail (the pilots), c = 0 | slots 4480 to 4495 `5a69` | slots 4480 to 4495 `6e57` | slots 4480 to 4495 `80c1` | slots 4480 to 4495 `bc79` |
| the tail, c = 1 | `a596` | `6e57` | `7f3e` | `bc79` |
| the tail, c = 2 | `5a69` | `91a8` | `80c1` | `4386` |
| the tail, c = 3 | `a596` | `91a8` | `7f3e` | `4386` |

The 3/4, 2/3 and 1/2 blocks:

| | block 4 (3/4) | block 5 (2/3) | block 6 (1/2) |
|---|---|---|---|
| first frame slot | 17,984 | 23,136 | 28,896 |
| CRC-32 of bytes 0 to 475 | 0xC41962E3 | 0x0A0AE212 | 0x834F6540 |
| codeword bits 3800 to 3839 (last payload byte, CRC) | `d7c41962e3` | `d70a0ae212` | `d7834f6540` |
| codeword bits 3840 to k - 1 | none | none | none |
| codeword bits k to k + 127 (parity p_0 to p_127) | `8e1fee07a8f1080cfcaec24e3654b527` | `f19ae6dd86677815a1b9c7f83863c1ef` | `f284e220c7972b3c6a97444d895fdcc7` |
| whitening under its slots 0 to 127 (frame slots S_b to S_b + 127) | `3866bc6962628a5c6783a12fef09a41c` | `a60df6cf665e475b2d587d0cafb932b1` | `3d4ee329250c1b091ef45580fdcb3e89` |
| slots 0 to 127 | `0603a4229de65af270a8f2db1111d193` | `a90b8ae3e08dd639660698828a536799` | `060e92c7b025451292c7ec2e384e367b` |
| the tail (the pilots), c = 0 | slots 5120 to 5151 `c22fe70f` | none | none |
| the tail, c = 1 | `3dd018f0` | none | none |
| the tail, c = 2 | `c22fe70f` | none | none |
| the tail, c = 3 | `3dd018f0` | none | none |

The padding's first 128 slots, frame slots 36,576 to 36,703, are the whitening there:
`332c1ead5f960d5aaf4c1e8d9f9e7cd8`.

SHA-256 of each whole codeword, packed most significant bit first (a 7/8 codeword's 4608 bits include the 128 never
sent): block 0 (4608 bits) `a62fac54602b23b03e30eb47aaa7f8e983f7d19bc8c7e13713eba2640a23c088`, block 1 (4608 bits) `152ae7463835c31782ca4980d5c592c817b1b66a3a7b976c47fd0f5f436a6700`, block 2 (4608 bits) `602e5bbe31440df25be5562e371c15eb771046800dd05375348c84bf7b182e07`, block 3 (4608 bits) `3fda58ac7bd787b16da65e3da29ea12528f98a3a8bf41020f8fffd30517d95d4`, block 4 (5120 bits) `3be83a78fef345f13645e840469519501a7b01da4576f78d8a5b5713d7fd7035`, block 5 (5760 bits) `958111ce99624a166d811a1f75c056f3a09127a58238d7d84c3b4abe28886671`, block 6 (7680 bits) `2925aaea9a62cbe2c1b4b6c546071eaf2e96a5cd70b9edcb5e4de979155609b8`. SHA-256 of the frame's 40,960 slots
packed the same way (5120 bytes), c = 0: `994f4750a6596def85a66fdbcacb9866edbbc827c1c408ad6ba5e010793b46cb`. At c = 1, 2 and 3 (`f.setParity(c)` before the encode below;
only the pilot slots of the blocks whose bit is 1 differ):
`e5bbfe8c31afa6ba38bfa27dbd3c621f1d51f8982f6be2db08dda65231b117b5`,
`26d07dc9a5636e452679c3429d31b7ea3b7f0a11064dc5831dd5571552232ebe`,
`1e08c661206a2133065b6637cd8a6df727c4d32256f35172274bd722f3a96712`.

From `liblizard/`:

```js
import { init, Focus } from "./sim/ob.mjs";
await init();
const f = new Focus(384, 64, 1);                         // LIZARD-64, LINEAR
const B = f.blockBytes, blocks = new Uint8Array(7 * B);  // B = 476
for (let b = 0; b < 7; b++) { blocks[b * B] = b; for (let j = 0; j < 472; j++) blocks[b * B + 4 + j] = j & 255; }
f.measure(true); f.encode(blocks);
const slot = Array.from(f.sent(), (s) => (s < 0 ? 1 : 0));   // 40960 slot bits, frame order
```

The CRC is the wasm's `ob_test_crc_hash` and agrees with node's `zlib.crc32`; the codewords, parity and whitening rows
were made by a separate JavaScript implementation of this section and section 8 (`liblizard/gpu/back/bitmap.mjs`
`encodeCodeword`, `perm`, `whiten`), whose slots equal the wasm's on every slot of the seven blocks.

### 7.5 The block id

The transfer is in the light: a camera alone recovers a file. Bytes 0 to 3 of every block are its id, a uint32 read
little-endian (`xfer.h:xfer_id_get`):

| bits | field | values |
|---|---|---|
| 31 to 18 | chunk | 0 to 16382: a chunk of the file (7.6). 16383 (`XFER_CONTROL`): a control block |
| 17 to 0 | fountain symbol | a data block: its Wirehair block id in its chunk's fountain. A control block: 0 the header (7.7), 1 + m manifest block m for m = 0 to 1365 (7.8), 1367 to 262143 reserved |

- id = chunk x 2^18 + symbol. The header's id is 0xFFFC0000 (`XFER_ID_HEADER`); manifest block m's is 0xFFFC0001 + m.
- A receiver MUST ignore a block with a reserved id.
- A data block's symbols count from 0 in each chunk: 0 to K - 1 are the chunk's K source blocks verbatim, K up its
  repair blocks (7.6). A sender that runs past 2^18 symbols wraps to 0.
- The header's id is the same in every transfer, and a manifest block's in every transfer with that many chunks, so a
  receiver MUST NOT drop a control block for its id: a control block is new when its bytes are.

### 7.6 Chunks and their fountains

- A file of `length` bytes is cut into chunks of 2^k bytes, k = `chunk_log2` from 10 to 24 (`XFER_LOG2_MIN`,
  `XFER_LOG2_MAX`): chunk c is bytes c 2^k up to the smaller of (c + 1) 2^k and length. Every chunk but the last is
  2^k bytes on a 2^k boundary, a whole subtree of the file's BLAKE3 tree (7.9).
- chunks = ceil(length / 2^k), at most 16,383 (`XFER_MAX_CHUNKS`), 0 for an empty file.
- **Compression.** Under codec 1 a chunk is sent as one zstd frame (`liblizard/zstd/shim.c` pins level 9, the row match
  finder on, no checksum, the content size in the frame; Zstandard 1.5.7) where that is shorter than the chunk, else as
  its bytes. What a chunk's fountain carries is its bytes **as sent**, 1 to len; sent < len means a frame. A sender MAY
  send any chunk as it is and MUST send a frame only where it is shorter; under codec 0 every chunk is sent as it is. A
  receiver MUST accept a stored chunk under either codec and MUST refuse a frame under codec 0. The length, root and
  chaining values are of the file's own bytes.
- Chunk c has K = ceil(sent / 472) source blocks (`xfer_blocks`). With K >= 2 it is its own Wirehair fountain over its
  sent bytes, the last block zero padded to 472. A chunk with K = 1 is not fountained: its sent bytes, zero padded to
  472, go as symbol 0, repeated, and a receiver takes the first sent of them.
- Wirehair's seed attempt (0 to 255) is chosen by the encoder; a decoder is built from the message length, the block
  size and that attempt. Every chunk's attempt travels in the manifest (7.8), a one-chunk file's in the header (7.7).
- Header version 3 fixes the fountain (Wirehair V2, profile `WIREHAIR_V2_PROFILE_CERTIFIED_2026_07`, 472-byte blocks)
  and the codec (zstd frames). A change to any of them is a new version.

### 7.7 The header block

Id 0xFFFC0000. Its 472 payload bytes, integers little-endian (`xfer.c` `xfer_header_write`, `xfer_header_parse`),
version 3:

| offset | bytes | field | value |
|---|---|---|---|
| 0 | 1 | version | 3 (`XFER_VERSION`) |
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
| 469 | 3 | reserved | 0 |

The fields fill the 472 bytes, and every byte a field does not use is zero. A receiver MUST refuse a header block that
breaks this table. `xfer_header_parse` returns -2 (`XFER_ERR_VERSION`) for a version other than 3, -3
(`XFER_ERR_HASH`) for a hash other than 1, -4 (`XFER_ERR_CODEC`) for a codec other than 0 or 1, and -1 for: chunk_log2
outside 10 to 24; chunks other than ceil(length / 2^k), past 16,383, or 0 for a file that is not empty; type_len over
158; a nonzero byte 7 or 469 to 471, or a nonzero byte past name_len in the name or past type_len in the type; with one chunk, a
sent of 0 or past the chunk's length, a sent under it with codec 0, or a nonzero seed with a sent of 472 bytes or less
(`xfer_sent_ok`); with any other number of chunks, a nonzero sent or seed. The parse does not check that the name is
UTF-8 or the type ASCII.

### 7.8 The manifest blocks

A file of 2 chunks or more has ceil(chunks / 12) manifest blocks (`xfer_manifest_blocks`, 1 to 1,366); a file of one
chunk or none has none. Manifest block m, id 0xFFFC0001 + m, 472 bytes:

| offset | bytes | field |
|---|---|---|
| 0 | 8 | tag: the root's first 8 bytes |
| 8 | 444 | the entries of chunks 12 m to 12 m + 11, 37 bytes each, in order; zero past the last chunk |
| 452 | 20 | reserved, 0 |

An entry is the chunk's chaining value (32 bytes, 7.9), its bytes as sent (u32, little-endian) and its Wirehair seed
attempt (1 byte, 0 where it is one block). A receiver MUST refuse a manifest block whose tag is not the first 8 bytes
of the root in the header it holds, whose m is past the file's last manifest block, with an entry no chunk of that
file could have (`xfer_sent_ok`), or with a nonzero byte after its last entry (`xfer_manifest_parse`).

### 7.9 The chunk tree, and when a file is accepted

BLAKE3 is the official function (the official C 1.8.7, `liblizard/vendor/blake3/`, unmodified).

- **A chunk's chaining value** (`xfer_chunk_cv`). BLAKE3 hashes in 1 KiB chunks numbered from 0 across the file, so
  chunk c's first is number c 2^(k - 10). Its chaining value is the one BLAKE3 computes for the subtree over the
  chunk's bytes, those 1 KiB chunks at those numbers merged by BLAKE3's rule, with the ROOT flag nowhere
  (`xfer.c` builds it on BLAKE3's compression function).
- **The root of the list** (`xfer_root`), for n >= 2 chaining values c[0 .. n): split at p, the largest power of two
  below n; the root is PARENT(root of c[0 .. p), root of c[p .. n)), PARENT(l, r) BLAKE3's compression of the 64-byte
  block l || r with BLAKE3's IV as the key, counter 0, flag PARENT, and at the top PARENT and ROOT. One chaining value
  is its own subtree.
- Since every chunk but the last is 2^k bytes on a 2^k boundary, k >= 10, this is BLAKE3's own tree: the root is
  BLAKE3 of the file (of the one chunk with one chunk, of the empty input with none).

**Acceptance.** A chunk sent shorter than its length is first decompressed as one zstd frame; a frame that does not
state the chunk's length, is damaged, or yields anything but that many bytes is a wrong chunk, refused as a wrong hash
is (its symbols discarded, the chunk collected again). A chunk is verified when its length is what the header gives
and, with one chunk, its BLAKE3 is the root, or, with two or more, its chaining value is entry c of a list that
combines to the root (`xfer_chunk_check`, `xfer_manifest_check`). A receiver MUST NOT hand over a file until every
chunk is verified, and an empty file until the root is BLAKE3 of the empty input.

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
(`liblizard/vendor/blake3/test_vectors/test_vectors.json`). Under codec 1 every 1 KiB chunk of that input compresses to
a 271-byte frame, one block, unfountained; the two rows under codec 0 send the same files as they are, three blocks a
chunk with seed attempt 1. Header version 3. Per chunk: its bytes, its bytes as sent, K and its seed attempt.

| length | k | codec | chunks | first chunk | last chunk | manifest blocks | header bytes 0 to 19 |
|---|---|---|---|---|---|---|---|
| 0 | 22 | 1 | 0 |  |  | 0 | `03011601 00000000 00000000 00000000 00000000` |
| 1025 | 11 | 1 | 1 | 1025, 271, 1, 0 |  | 0 | `03010b01 00000000 01040000 00000000 01000000` |
| 8193 | 10 | 1 | 9 | 1024, 271, 1, 0 | 1, 1, 1, 0 | 1 | `03010a01 00000000 01200000 00000000 09000000` |
| 31744 | 10 | 1 | 31 | 1024, 271, 1, 0 | 1024, 271, 1, 0 | 3 | `03010a01 000a1800 007c0000 00000000 1f000000` |
| 102400 | 10 | 1 | 100 | 1024, 271, 1, 0 | 1024, 271, 1, 0 | 9 | `03010a01 00000000 00900100 00000000 64000000` |
| 1025 | 11 | 0 | 1 | 1025, 1025, 3, 1 |  | 0 | `03010b00 01000000 01040000 00000000 01000000` |
| 31744 | 10 | 0 | 31 | 1024, 1024, 3, 1 | 1024, 1024, 3, 1 | 3 | `03010a00 000a1800 007c0000 00000000 1f000000` |

Roots, bytes 20 to 51 of each header:

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
row. SHA-256 of each whole 472-byte header:

```
0               7d037bf8c824ec661b7ab3eed17980fc7959e1ac6ea58d1301a4b254b966c643
1025            6f50037377d51026bd7c9c9f0c18126da5a5960804d7dfabfce8366533521fc1
8193            3d476678686d0801bb8c824bc7f119df020d991a58adc500b9f48382bd49740c
31744           b3277c33e376caab1d7139234054c75202f4d5629a40a939882790e7cf7f1b03
102400          770386a9351c12a9ee84a828a278cf309373de47c08b725473ce46576ebf998d
1025, codec 0   1790552b5cf24bcec969613c8bc09e108474b8afc04eeee1a2a33d433084a34f
31744, codec 0  8d57984a4009b87eefcff9e17452984448a1996ff70a4287cc14cde38ddb2b35
```

Chaining values of 1 KiB chunks of this input:

```
chunk 0, bytes 0 to 1023 (the same in every file above)   5c9e654411e393d1f4bec710ccd5bc5669ab177d610a0eb691fcfee92fb4e8b1
chunk 30 of the 31,744-byte file                          a6b76fabdd73c356e0a57cd84e53c562b819b21f140425029385c1574fe985f7
chunk 8 of the 8,193-byte file, its one byte a0           0c89b23aaf0f396fe25995d42f7d6f3c7b1e18b59d9c778b06688325f44016de
```

Manifest blocks, SHA-256 of each 472 bytes. Block 0 of the 31,744-byte file begins with the tag `62b6960e1a44bcc1`,
then chunk 0's entry: its chaining value, sent `0f010000` and seed 0 under codec 1, sent `00040000` and seed 1 under
codec 0; its three blocks hold 12, 12 and 7 entries, the 102,400-byte file's nine 12 each but the last's 4.

```
31744 m 0            c4ba4a2096a40df0468508c48f1e49a1e95990ff384389d2a2e86dccf2fda607
31744 m 1            87ddd478798fe50c02d66396d0aff226af1d5b0d7002903bf9b07c03d486b131
31744 m 2            bf0907c0dfeaeaa622f68550bdbc4e1ffd49ba8b41c36e9a2a2bc3573d02a3ec
8193  m 0            f0d7b12dd3ca55ed55756b0242f47ef082f8f9bff1c94714922aeaf1bb28fb25
102400 m 0           4b6a3983dc6b82a31debccc70f6eddec4119f456d8a0d0652bb0e4d8639b36c9
102400 m 8           48f2fc154e3d7f893dacc1bc17d11e131cdd0e3fb999960c3759dc936dc5251f
31744 m 0, codec 0   0f9591921d928d444453b54c9bb78e1bb6122881c583520a30f31fbd8d6a5943
31744 m 1, codec 0   8e6cdc007900173aa3b6f78a98c0e075fdd9fce09f9db482f14d315c86798cb7
31744 m 2, codec 0   2e652489193f3a4c1bd51e54406bf8b04d3c91bc9dfce02b6adcf7cb82c7b8cd
```

**Refusals.** The 31,744-byte header with one field changed: version 2 gives -2; hash 0 or 2 gives -3; codec 2 gives
-4; chunk_log2 9 or 25, byte 7 or a byte of 469 to 471 set, a byte of the name's or the type's padding set, chunks one more, type_len 159, or
a nonzero sent gives -1. The 1,025-byte header under codec 0 with sent 300, or with sent 0, gives -1. Its last manifest
block with a padding byte set, with its tag changed, with an entry's sent past its chunk's bytes, or with a seed on a
chunk of one block, is refused.

`liblizard/test/xfer_test.c` and `liblizard/test/xfer_wasm.mjs` hold the C and the wasm to BLAKE3's 35 official
vectors, whole and split into chunks of 2^10 to 2^17 bytes.

## 8. The LDPC code

Normative except 8.3 and 8.5. Sources: `ldpc_base.h`, `ldpc.h`, `ldpc.c` (`ldpc_init`, `ldpc_encode`), `focus.c:init`.

A block's code is one of four quasi-cyclic irregular repeat-accumulate codes, H = [Hd | Hp], by its tier (3.1): rate
7/8, 3/4, 2/3 and 1/2, every one lifted by z = 128, each over its own base graph. Hd is built from z x z circulant
permutation matrices; Hp is a bit-level accumulator (a staircase). 8.1 and 8.2 give the 3/4 code; 8.6 the other three,
built and encoded the same way at their own base graphs.

### 8.1 Parameters

| | |
|---|---|
| codeword n | 5120 = 40 z |
| information k | 3840 = 30 z: the 3840 bits of 7.1 |
| parity bits and checks m | 1280 = 10 z |
| lifting z | 128 |
| base graph | 10 block rows by 40 block columns: 30 data columns of circulants (Hd), then the 1280 parity bits as a staircase (Hp) |
| data column degree | 10 for columns 0 to 5, 3 for columns 6 to 29 |
| data circulants | 132, 14 in rows 0 and 2 and 13 in the other eight |
| check degree | 15 or 16: 13 or 14 data bits and 2 parity bits; check 0 has one fewer |
| parity bit degree | 2; the last parity bit has 1 |
| edges | 19,455 |
| cycles | none of length 4 |
| H, FNV-1a over row pointers and column indices (`liblizard/test/ldpc_table.c`) | 3f726d8e |

### 8.2 The parity-check matrix

Codeword bit 128 j + t (j = 0 to 29, t = 0 to 127) is bit t of data column j; codeword bit 3840 + c is parity bit c
(c = 0 to 1279). Every "mod" is the non-negative remainder. Checks are numbered c = 10 i + r, r (0 to 9) the block
row and i (0 to 127) the position in it. Check c contains:

- for each data column j with a shift sh[r][j] in block row r: data bit 128 j + ((i - sh[r][j]) mod 128);
- parity bit c - 1, when c > 0;
- parity bit c.

So data bit t of column j is in check 10 ((t + sh[r][j]) mod 128) + r of every block row r where sh[r][j] exists, and
Hp is the 1280 x 1280 matrix with ones on its diagonal and just below it. The interleaved check numbering is part of
the code (numbered 128 r + i, a degree-3 column would form codewords of weight 5).

**The shift table sh** (a dash: no circulant). This table is H: the reference ships it as `ldpc_base.h:LDPC_BASE34`, 10
rows of 30 shifts with -1 for a dash (`LDPC_BASE34_RATE` = 4, the rate index; `LDPC_BASE_Z` = 128), and `ldpc_init`
builds the code from it by its rate.

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0      38  44  23 124  14  68   -  99   -   -   2   -   -   -  41   -   5   -
1      24  97  74  61  42  65   -   -   5  38   -   -   -   -   -  88   -  46
2      34  43  82  90 106  37   -   -  25   -   -   -  35   -   -  64   -   -
3      76  29  45  64  61  98  74   -   -   -   -  57   -   -  70   -   -   -
4     104  87  99 124  37  59   -   -   -  23 102   -   - 124   -   -  57   -
5      11  34 109  53  60   1   -  12   -   -   - 123   -  43   -   -   -   -
6       6  39  65  93  20  91   -  75   -   -   -  91   -   7   -   -   -  31
7      24 120 116  62  15  56   -   -  35   -  96   -   -   -   -  52   -   -
8      33 121  49  49  31  59  56   -   -  35   -   - 107   -   -   -  78   -
9      93 105  97  64  41 101  87   -   -   -   -   -  64   - 110   -   -  32

r \ j  18  19  20  21  22  23  24  25  26  27  28  29
0       -  79   -   -   -   -   -  78   -   - 110  55
1       -   -  65   -   - 104   -   -   -   -   -  23
2      37   -   -  97   -   -  98   -  42   -   - 113
3       - 105   -  17   -   -   -   0   - 111   -   -
4       -   -   4   -   -   -  96   - 104   -   -   -
5      50  77   -   -   -   0   -   -   -   -  50   -
6       -   -   -  32  95   -   -   - 116   -   -   -
7      89   -   -   - 127   -   -  13   -  26   -   -
8       -   -   -   -  53  41   -   -   -   -  88   -
9       -   -  34   -   -   -  35   -   -  17   -   -
```

### 8.3 How the tables were made (non-normative)

All four come from `scripts/exp/ldpc_opt.py` `generate(z, mb, kb, degrees(kb, heavy), seed, np)` at z = 128, which
regenerates them exactly: 7/8 (5 block rows, 31 data columns, 5 of degree 5, seed 1), 3/4
(10, 30, 6 of degree 10, seed 2), 2/3 (15, 30, 7 of degree 11, seed 1), 1/2 (30, 30, 12 of degree 8, seed 1); the other
data columns of degree 3. The columns are placed in order, each edge in the lightest block row, and every shift of 0 to
z - 1 is scored by the exact count of 4- and 6-cycles through the whole base matrix, the staircase included. The heavy
columns are the profiles of the codes before them (2026-10-08) scaled to these widths. Their 10% codeword-loss
thresholds on BI-AWGN (QPSK Es/N0, the exact int8 decoder of 8.5 with no stop, 30 iterations): 6.023, 4.038, 3.005 and
1.087 dB. The printed tables are the format, and `liblizard/test/ldpc_table.c` holds each to its hash (8.1, 8.6) and
`ldpc_init`'s block-row view to the table entry by entry.

### 8.4 Encoding

Systematic and linear-time (`ldpc.c:ldpc_encode`). The codeword is the k information bits, then p_0 to p_(m-1):

```
p_c = p_(c-1) XOR (XOR of the data bits in check c),    p_(-1) = 0
```

### 8.5 Soft values into the code, and the reference decoder (non-normative)

The format fixes only the sign convention (positive means bit 0) and the bit map; a decoder computes its soft values
as it likes. The reference (`focus.c` `focus_finish_ext`, `axis_info`, `map_in`; `ldpc.c:ldpc_decode_stall`):

- **Per sub-channel**, over its 320 received coefficients y after the transform, detrend and pilot alignment: m2 and m4
  the means of |y|^2 and |y|^4; A2 = sqrt(max(0, 2 m2^2 - m4)), the signal power; N = max(m2 - A2, 0.02 m2); each
  axis LLR = 0.7 x 2 sqrt(2 A2) / N x y_axis, clipped to [-10, 10] and stored as int8 = round(8 x LLR), ties to even,
  for every slot of the frame. A block reads its slots' values whatever sub-channels they fall in.
- **Per block**, slot order to codeword order: L_code[s(st i mod nt)] = L_slot[i] for i < nt, negated where
  w[S_b + i] = 1; the tail dropped; at 7/8 the 128 bits never sent, the zero information bits past the CRC, fed as
  certain zeros (+127).
- **Decline gate.** Each sub-channel's information per coded bit is J(2 sqrt(A2 / N)), with
  J(sigma) = (1 - 2^(-0.3073 sigma^1.787))^1.1064; a block's estimate is the mean over its slots of their sub-channels'
  information (each sub-channel weighted by the block's slots in it), and a block whose estimate is under 0.2 k / nt
  (`FOCUS_DECLINE`: 0.177 at 7/8, 0.15 at 3/4, 0.133 at 2/3, 0.10 at 1/2) is not decoded.
- **LDPC**, layered normalised min-sum: a layer is one block row (its z checks share no bit); int8 check-to-bit
  messages, int16 posteriors clamped to [-8191, 8191]; min1 and min2 scaled by 13/16 as (x x 13) >> 4; the sweep's
  direction alternates every iteration; check 0's missing parity bit fed as a certain 0; at most 30 iterations, stopping
  at a zero syndrome, and a block whose unsatisfied checks after iteration 9 are above 0.95 of those after iteration 1
  given up (`FOCUS_STALL_IT`, `FOCUS_STALL_RATIO`); a posterior below 0 decides bit 1.

### 8.6 The 7/8, 2/3 and 1/2 codes

Each is H = [Hd | Hp] as in 8.2 over its own base graph at z = 128: codeword bit 128 j + t (t = 0 to 127) is bit t of
data column j, codeword bit k + c is parity bit c, and check c = mb i + r (r the block row, 0 to mb - 1, i the position,
0 to 127) holds data bit 128 j + ((i - sh[r][j]) mod 128) for each shift of block row r, parity bit c - 1 when c > 0,
and parity bit c. Encoding is 8.4's accumulator over the m checks.

The 7/8 code's data column 30 (codeword bits 3840 to 3967, the 128 zero information bits past the CRC) is never sent.
A block's 4480 code slots carry codeword bits 0 to 3839 and 3968 to 4607 (7.3); a receiver feeds bits 3840 to 3967 as
certain zeros.

| | 7/8 | 2/3 | 1/2 |
|---|---|---|---|
| slots a block (sent bits, tail) | 4496 (4480, 16) | 5760 (5760, 0) | 7680 (7680, 0) |
| codeword n | 4608 = 36 z, of which 4480 = 35 z sent | 5760 = 45 z | 7680 = 60 z |
| information k | 3968 = 31 z | 3840 = 30 z | 3840 = 30 z |
| parity bits and checks m | 640 = 5 z | 1920 = 15 z | 3840 = 30 z |
| base graph | 5 block rows, 31 data columns, column 30 (the zeros) never sent | 15 block rows, 30 data columns | 30 block rows, 30 data columns |
| data column degree | 5 for columns 0 to 4, 3 for 5 to 30 | 11 for columns 0 to 6, 3 for 7 to 29 | 8 for columns 0 to 11, 3 for 12 to 29 |
| data circulants | 103, 21 in rows 0, 1, 3 and 20 in rows 2, 4 | 146, 10 in rows 0, 3, 4, 6 to 12, 14 and 9 in rows 1, 2, 5, 13 | 150, 5 in every block row |
| check degree | 22 or 23: 20 or 21 data bits and 2 parity bits (check 0 one fewer) | 11 or 12 (check 0 one fewer) | 7 (check 0 one fewer) |
| edges | 14,463 | 22,527 | 26,879 |
| H, FNV-1a over row pointers and column indices (`liblizard/test/ldpc_table.c`) | 6f67223a | 27d3a72e | 721499de |

The reference ships them as `ldpc_base.h:LDPC_BASE78` (`LDPC_BASE78_NP` = 1, the last data columns never sent), `LDPC_BASE23` and
`LDPC_BASE12`, from which `ldpc_init` builds the codes by their rates.

**The 7/8 shift table** (a dash: no circulant):

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0      18  83   3  51  98   -  59 124  43   -   -   7  91   -  15   -  65 108
1      31 111  59  16  25   -  37  56   - 108   -  42 110   -  63   -  24  24
2      65  35  43 126  34  14   -  20  59   -   5   -   - 107  96  82   -  25
3     120  69  63  15  53   3   -   -  37 105  70 118   -  88   -  57   -   -
4     121  29  94  39  26  99  39   -   -  46  54   -  27  73   -   8  56   -

r \ j  18  19  20  21  22  23  24  25  26  27  28  29  30
0       -  60 102  29   - 102   -  83   - 121   -  57  72
1       -  45   - 106  88   -  97   - 111 113   -  66  92
2      25   -   - 120   - 123 114 114  47   -  60   -   -
3      85  14  67   -  95   -  65   - 121   -  31 106 105
4      95   -   0   - 107   3   - 110   -   9  18   -   -
```

**The 2/3 shift table**:

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0      18   -  87  73  47  77  10   -   -   -   -   -   -  80   -   -  53   -
1       3  45   -   4  93 105   -   -  44   -   -   -  26   -   -   -   -   -
2       - 100 119  69   -  62 108   -   -   -  81   -   -   -  40   -   -  77
3      39  65  19   -  46   6   -   -   -   -   -  27   -   -  71   -   -   -
4     106  14 120   -   -  21  59   -   -   -  34   -   -   -   -  35   -   -
5      52   3  60   -   -  88  94  39   -   -   -   8   -   -   -   -   -   -
6       -  15  63  70 125   -  34   -   -   -  85   -  16   -   -   -   -   -
7      65 124   -   6  79  50   -  44   -   -   -   -   -   -  60   -   -  83
8       -  41  37  79  74   -  75   - 114   -   -   -   -   -   -  70   -   -
9      70 116   -  99 106   -  32   -   -  43   -   -   -  48   -   -   -   -
10      -  68  16   - 102  63  45   -  75   -   -   -   -  40   -   -  65   -
11    121   -  19 105  14   - 125   -   -  24   -   - 105   -   -   -   -   -
12    120   -  10 116  73 127   -  95   -   -   -   -   -   -   -  84   -   -
13     53   9   -  60  22   1 107   -   -   -   - 113   -   -   -   -   - 118
14     96   -  93  50   -  45  77   -   - 121   -   -   -   -   -   -  81   -

r \ j  18  19  20  21  22  23  24  25  26  27  28  29
0       -   -   -   -   -   -   -  78  33   -   -   -
1     103   -   -   -  21   -   -   -   -   -   -   -
2       -   -   -   -   -   -   -   -  47   -   -   -
3       -   -  31 119   -   -   -   -  19   -   -   -
4       -   -  21   -   -   -   -  46   -  30   -   -
5      80   -   -   -   -  66   -   -   -   -   -   -
6      28   -   -   -   3   -   -   -   -   - 117   -
7       -   -   -   -   -  61   -   -   -  36   -   -
8       -   -   4   -   -   -  66   -   -   -   -  91
9       - 121   -   -   -  69   -   -   -   -   -  42
10      -   -   -  86   -   -   -   -   -   -  19   -
11      -   -   - 114  99   -   -   -   -  14   -   -
12      -  13   -   -   -   -  58   -   -   -   -  20
13      -   -   -   -   -   -   -  50   -   -   -   -
14      -  69   -   -   -   -  81   -   -   - 108   -
```

**The 1/2 shift table**:

```
r \ j   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
0      18   -   -   -  93   -   -   -   -  61   -  87   -   -   -   -   -   -
1       -   -  96   -   -   -  62   -   -   -  84   -   -   -   -   -   - 113
2       -   3   -   -   -  58   -   -   -   -   -  21   -   -   -   -   -   -
3       -  39   -   -  54   -   -   -   -   -   6   -  45   -   -   -   -   -
4       -  59   -   -   -   -  69   -   -   -  32   -   -   -   -   -  96   -
5       -   -  35   -   -   -  72   -   -   - 113  74   -   -   -   -   -   -
6      52   -   -   -   -   -   -  28  92   -   -   -   -   -   -   -   -  18
7      39   -   -   -   -  44   -   -   - 114   -   -   -   -   - 124   -   -
8     106   -   -   -   -   -   - 112   - 118   -   -  70   -   -   -   -   -
9       -   - 125   -   -   -   - 107   -  84   -   4   -   -   -   -   -   -
10      -   -   -  92   - 124   -   -   -   -   -  18   -   -   -   -   -  36
11      -   -  51   -  43   -   -  60   -   -   -   -   -   -   -   -   -   -
12      - 100   -  36   -   -   -   - 105   -   -  98   -   -   -   -   -   -
13      -   -   -  41   -  88   -   -  28   -   -   -   -   2   -   -   -   -
14     65   -   -   - 111   -   -   -   -  72   -  64   -   -   -   -   -   -
15      -   -  33   -   -   -  45   -   -   - 101   -  77   -   -   -   -   -
16      -   -  62   -   - 103   -   -   -   -  98   -   -   - 113   -   -   -
17      -   -   -  69  97   -   -   -   -   - 102   -   -   -   -   - 111   -
18      -   -   -  20  57   -   -   -   -  53   -   -   -   -   -   -   -   -
19      -  70   -   -   -   -  96  32   -   -   -   -   -   -   -  10   -   -
20      -   -   - 123   -  97   -   -  81   -   -   -   -   -  20   -   -   -
21      -  41   -   -   -   - 112   -   -   -  24  96   -   -   -   -   -   -
22    121   -   -   -   -   -   6   -   -   4   -   -   -  98   -   -   -   -
23      -  68   -   -  45   -   -   -  98   -   -   -   -   -   -   -   -   -
24      -   -  26   -   -  73   -  82   -   -   -   -   -  63   -   -   -   -
25    121   -   -   -   -  27   -   -  20   -   -   -   -   -   -   -   -   -
26     53   -   -   -   -   -   9   -   - 109   -   -   -   -   -  34   -   -
27      -  96   -   -  51   -   -   1   -   -   -   -   -   -  36   -   -   -
28      -   -   -  35   -   -   -  85 118   -   -   -   -   -   -   -   -   -
29      -   -  17  54   -   -   -   -  34   -   -   -   -   -   -   -   8   -

r \ j  18  19  20  21  22  23  24  25  26  27  28  29
0       -   -   -   -   -  26   -   -   -   -   -   -
1       -   -   -   -   -   -   -   -   -  58   -   -
2       -  16   -   -   -   -   -   -   -   -   - 100
3       -   -   -   -  12   -   -   -   -   -   -   -
4       -   -   -   -   -   - 125   -   -   -   -   -
5       -   -   -   -   -  34   -   -   -   -   -   -
6       -   -  69   -   -   -   -   -   -   -   -   -
7       -   -   -   -  57   -   -   -   -   -   -   -
8       -   -   -   -   -   -   -  99   -   -   -   -
9       -   -   -   -   -   -   -   -   -   -  14   -
10      -   -   -   -   -   -   -   - 117   -   -   -
11    117   -   -   -   -   -   -   -   -  64   -   -
12      -   -   -   -   -  36   -   -   -   -   -   -
13      -   -   - 107   -   -   -   -   -   -   -   -
14      -   -  85   -   -   -   -   -   -   -   -   -
15      -   -   -   -   -   -   -   -   -   -   - 110
16      -   -   -   -   -   -   -   -   -   8   -   -
17      -   -  55   -   -   -   -   -   -   -   -   -
18      -   8   -  44   -   -   -   -   -   -   -   -
19      -   -   -   -  48   -   -   -   -   -   -   -
20      -   -   -   -   -   -   -   -   -   -  27   -
21      -   -   -   -   -   -   -   -  76   -   -   -
22      -   -   -  92   -   -   -   -   -   -   -   -
23     99   -   -   -   -   -   -  97   -   -   -   -
24      -   -   -   -   -   -  40   -   -   -   -   -
25     23   -   -   -   -   -   -   -   -   -   -  35
26      -   -   -   -   -   -   -   -   -   -  67   -
27      -   -   -   -   -   -   -  99   -   -   -   -
28      -  55   -   -   -   -  74   -   -   -   -   -
29      -   -   -   -   -   -   -   -  86   -   -   -
```

## 9. Animation and transport (non-normative)

What the reference senders and receivers do with a stream of symbols. The display rate is in the word (5.1) and the
transfer's layout in the blocks (7.5 to 7.9).

### 9.1 Frames and block ids

- **The count.** c is the sender's count of the pictures it has painted, mod 4: 0 on its first picture, then 1, 2, 3,
  0. A picture the display shows twice keeps its c, so a receiver reads a repeat as the same count twice and a skipped
  picture as a count two on. A sender painting two codes a frame gives both the frame's c. The pilots carry it (7.3);
  a sender that paints c = 0 throughout reads +1 on both readings.
- **A file** (`liblizard/sim/xfer.mjs:XferSender` and the native `XferTx`):
  - Every chunk is interleaved with every other over the whole file in proportion to its blocks, in laps that repeat
    while the sender runs, each chunk's symbols counting on so every lap sends fresh repair ids. A lap is a full
    chunk's K rounds; a shorter last chunk of K blocks gets K + ceil(2 sqrt(K)) visits a lap, at least 64 and at most a
    full chunk's K. A chunk of one block sends symbol 0 at every visit.
  - A control block goes out once `every` data blocks have gone since the last: the header on even turns, the manifest
    blocks in turn on odd ones. every = 63, or lap / (8 m) data blocks when that is less, m the manifest blocks (at
    least 1). An empty file's frames are all header.
  - Each frame's slots, the control block's among them, are shuffled (Fisher-Yates, xorshift32 seeded from the frame
    count), so no chunk or control block is pinned to one block position.
  - Chunks of 4 MiB (k = 22) unless set otherwise.
- **The test stream.** Block b of a frame carries id base + b, base a counter that starts at a random id under 2^30 at
  each start and advances by the frame's blocks after each frame; no id repeats.
- **A change of format** keeps the 472-byte payload, so the fountain is unaffected; the test stream's counter keeps
  counting.
- A receiver counts a data block as new once per id; control blocks are handed on every time (7.5).

### 9.2 The display rate

The sender paints at a period of 1000 / fps ms and states fps in the word (5.1). fps may change between any two frames
(only the word's cells change). A receiver uses it to judge its capture rate against the display. The ceiling is
472 B(V) fps bytes a second: LIZARD-512 at 24 fps, 61 x 472 x 24 = 691,008 B/s.

### 9.3 The test stream

In a test transfer, block id i carries the first 472 bytes of SHAKE256 (FIPS 202) of i as four little-endian bytes
(`shake.c:stream_fill`). The first 16 bytes of output:

| id | SHAKE256(LE32(id)), first 16 bytes |
|---|---|
| 0 | `b4a21e7939beb125bd3c10237ed5ea31` |
| 1 | `220ee668c9ea457d0330c21554da9e11` |
| 12345 | `08781f021b171adcc9fd8886013dffc2` |
| 4294967295 | `8c82cb0926f0275dc7650c348eae46b2` |

Id 0's 472 bytes end in `1e8e99a4fc614f4e`.

A receiver told nothing judges a test frame from the light: a sender paints the test stream or a file, never both in
one frame, so a frame is the test stream's when one of its CRC-verified data blocks carries its id's SHAKE256 (a
file's block does so with probability 2^-3776). Every other verified data block of that frame that does not is
counted bad and dropped; a test frame's blocks go to no transfer (`liblizard/sim/phy.mjs:blockJudge`).

### 9.4 The fountain: Wirehair as the project uses it

- Wirehair V2 (`liblizard/vendor/wirehair`, through `liblizard/wirehair/shim.cpp`), the profile pinned to
  `WIREHAIR_V2_PROFILE_CERTIFIED_2026_07`; the shim refuses any other, since two profiles disagree about the equations
  silently.
- Block size 472 bytes; a message spans 2 to 64,000 blocks. A chunk is at most 16 MiB (35,545 blocks), and a chunk of
  one block is not fountained (7.6).
- In a chunk of K blocks, symbols 0 to K - 1 are its blocks verbatim, the last zero padded on the wire and trimmed
  before the decoder takes it; symbols from K up are repair blocks; a duplicate is idempotent. Enough distinct symbols
  recover a chunk whichever they are.
- The receiver (`liblizard/sim/xfer.mjs:XferReceiver`) takes blocks in any order, keeps each chunk's blocks (in the
  origin private file system where a browser has one), solves a chunk once the manifest has given its count and K
  distinct symbols are in, decompresses it, and checks it the moment it decodes (7.9); a chunk that fails is refused,
  its symbols discarded, and collected again from later laps. A header with other bytes is a new transfer.

### 9.5 Putting the symbol on a screen

How a sender brings the painted image to a display is outside the format. Every resample between the painted image and
the display's pixels attenuates the outer sub-channels, where blocks fail first, so a sender should resample once, keep
the border's modules whole where the room allows, and add no other scaling (13, item 15).

### 9.6 Several files and folders

A transfer carries one file. Several files or a folder go as one POSIX tar sent as that file with the media type
`application/x-tar` (`lizard-web/tar.mjs`, `liblizard/kotlin/Tar.kt`, one rule in both):

- One file alone is sent as itself, never as a tar.
- ustar headers, the name and prefix fields where a path fits them, a pax extended header (typeflag `x`) with `path`
  where it fits neither or is not plain ASCII and `size` past 8 GiB; folders as entries of their own (typeflag `5`);
  files mode 644, folders 755, owner 0; every entry in the order of its path, folders before what they hold; two zero
  blocks at the end. Links are skipped.
- The name: one folder is `<folder>.tar`, its entries under `<folder>/`; several items `<first> and <N - 1> more.tar`,
  the items at the archive's top. A second file of the same path is numbered (`b (2).txt`). The name is cut on a
  character so it and its `.tar` fit 255 bytes.
- Times: a file's modification time; a folder's where the platform gives one, else the newest of what it holds, else
  the newest of the choice, so the same choice makes the same archive.
- A receiver may offer the tar or unpack it. An unpacker makes only plain files and folders, refuses an absolute path,
  a path through `..` and any other entry type, and on a refusal keeps the tar as the file it is.

## 10. Reference receivers (non-normative)

Two decoders read the format; neither defines it. Both are blind (5.5), erase the anchors (6.13), report each frame's
two pilot readings (7.3), and the Android app holds its camera's phase against the display by them.

### 10.1 The C decoder (`liblizard/src/`)

The reference, built to wasm for the web receiver (a worker pool, one thread a worker) and natively for the apps
(NEON on arm64, SSE4.1 on x86-64, `liblizard/core/cpu/`). `wasm.c:focus_any_setup(nmax)` builds a codec a ring;
`focus_any_rx(img, iw, ih, gamma, mesh, blocks, ok, held)` registers the frame against the three rings' layouts, reads
the word in the ring that registered, builds (and caches, `FOCUS_ANY_PICS` = 2) the codec for that ring and version,
and finishes the frame there, or at the held version `held` (0 for none) when the word does not read. The steps of one
frame (`focus.c:focus_decode`, `focus_acquire` then `focus_finish`):

1. **Luma.** (77 R + 150 G + 29 B + 128) >> 8 (`acquire.c:ob_luma`), then an optional gamma table.
2. **Binarize**, for the finder only: 8 x 8 block means, each block's threshold the mean of the 5 x 5 blocks round it
   less 3.
3. **Find** (`acquire.c:find_frame`): corner marks scanned in the image in their gapped and merged forms (4.5), quads
   from their agreement, each scored by reading the timing track against its local levels in every ring's layout under
   the rotations and mirror images; a rotation holds at 0.15. Failing that, the border's lines are fitted and scored the
   same way.
4. **Register** (`acquire.c:acquire_core`): a homography from the quad with its corner nodes refined; the anchors (6.13)
   found by the pattern's normalised correlation, accepted at 0.6, held to a robust rule, and used where 12 or more
   remain; the border nodes refined against the painted border (4.10); the interior from the border (and the anchors) as
   a homography's first-order change plus radial terms about the camera's axis, their count chosen by AICc, then a Coons
   patch and, with the anchors, a cubic mesh.
5. **Format word** (5.4) in the ring that registered; the blind receiver hands the registration over to the picture
   the word names, or the held one (5.5).
6. **Sample** the n x n picture at 6.9's points through the registration (`acquire.c:ob_sample_grid`).
7. **Erase and detrend**: the anchors' squares erased (6.13), the surface fitted (6.10).
8. **Transform** along y a strip of 64 columns at a time, along x only the rows v < n / 2.
9. **Pilot alignment, soft values, decline gate and LDPC** (8.5): the grid's shift fitted from the pilots' known
   symbols and turned back where its chi-square passes 13.8 (`focus.c:pilot_align`).
10. **CRC-32** (7.1).

`focus_acquire_quad`, `focus_finish` and `focus_finish_ext` take a quad, a sampled grid or soft values made elsewhere,
so another front or back half is held to the C's acceptance rules.

### 10.2 The GPU decoder (`liblizard/gpu/`)

A second decoder designed for a GPU: GLSL generated by a JavaScript host for its setup, compiled through glslang to
SPIR-V and run natively on Vulkan by `liblizard/core/` in the Android and desktop apps. A batch of frames is one submission: learned
nets find the corner marks, the timing track scores every quad, orientation and ring, the border nodes are refitted,
the anchors are found and steer the map, the word is read (F8, by ordered statistics over the ring's code with a
side-aware gate held to the C's false-word rate), and a back half does the sampling, transform, detrend, pilot
alignment, soft values, decline gate, LDPC (its stop a small learned net a code, a z = 128 code a 128-lane workgroup)
and CRC-32 for every tier of the profile. It is judged on CRC-verified blocks, throughput and a negative control, never on parity with the C.
`liblizard/gpu/README.md` has its stages.

## 11. Measured facts (non-normative)

Measurements, with their conditions, are kept in the project's log, not here. The one figure behind a constant: the
clip ratio c = 2 (6.7) read more blocks a clean capture than c = 1.6 in every live session (76.4 against 74.9 of 81,
one phone, 2:1, the 256 ring).

## 12. Settled and rejected (non-normative)

| decision | reason |
|---|---|
| Luminance only, never colour | colour is what made earlier colour barcodes unreliable across screens and cameras |
| Data on frequencies (FOCUS's OFDM), not on cells | a frame degrades block by block under blur and distance where a cell code stops dead at its cliff |
| QPSK on every tier; no 16-QAM or other constellation | the capture's distortion floor; denser constellations cost more decoding than they carry |
| Soft LDPC for the payload, Reed-Solomon only for the word | LDPC uses the grey values; RS suits a small fixed field read with erasures |
| The rate profile, not one rate | a capture's signal-to-noise ratio falls with the frequency |
| The profile's targets 39 / 15 / 20% of the slots at 7/8, 2/3 and 1/2 | the split the profile had over the sub-channels, as shares of the slots. Live, on one phone's 2:1 captures of LIZARD-656 in the 256 ring, it read above 4.3 MB/s, where the split searched for these codes on re-modulated recorded captures (27.8 / 16.5 / 32.9%) read at most 4.03 MB/s and 89.8% of its blocks (the codes before at their split: 4.26 MB/s, 95.1%) |
| Every code at z = 128, each its own base graph; the block 476 bytes | a z that fills 64-lane waves and 128-lane workgroups: on recorded captures re-modulated under the codes (the integrated GPU, the learned stops on) 6 to 7% more blocks per GPU millisecond than the codes before (z 93, 106, 120, 160) at the same split, 9 to 10% on the same captures 1 dB worse; block lengths from half to twice 473 bytes moved that by a few percent at most. 476 bytes and the CRC-32 fill the 3/4, 2/3 and 1/2 codes' k: no information bit there is a known zero |
| Blocks packed by slots, the remainder padded | whole sub-channels a block would fix each code's length to a multiple of 640 slots; the padding is under 2% of a frame from LIZARD-320 up |
| H a printed table, not a construction | a construction would let a change of seed or score move the format silently |
| No 5G NR tables, no polar codes | patents |
| Three rings, any ring for any picture, version decoupled from size | the ring is the sync layer chosen for the room, the word says what is inside |
| No retry at the other rings when a word fails | a frame whose word fails has failed throughout |
| Picture sizes 256, 384, 512, 768, 1024, 1152, 1280, 1536 | no step more than 1.5 times the samples; 1152 and 1280 cut the samples a sender paints and the C reads by 0.56 and 0.69 for the formats between 1024 and 1536; radix 2 with radix-3 and radix-5 stages |
| Whole pixels a module, the picture resampled to fill its span | no dead light between border and picture; a copy where pxm = scale |
| Four corner marks, gapped; no QR finders, centre or edge marks, filled marks | four measured corners carry perspective; the others lost on the finder or cut into the picture |
| 2 x 2 band cells | a coarser band did not help the mesh |
| The margin is the codec's; no guard ring | the finder needs light outside the ring; a further ring bought nothing |
| Versions of 16 sub-channels, 4 to 64; LIZARD-64 the smallest format | a format at the ladder's foot that fails the pilots' checks leaves it: LIZARD-16 (one 7/8 and one 3/4 block), then LIZARD-32 and -48, whose odd blocks' pilots are one 7/8 block's 8 coefficients (on captures mixed with a neighbour up to 0.13 past the pilots' model, 0.82 and 0.84 on clean ones); a frame of one block has no diversity and one pilot parity alone; 61 of the version byte's 256 values a format (5.4) |
| LINEAR the only bit map, whitened | whitening breaks up payload runs; LINEAR spreads every run of the code over the block |
| Pilots in every format; no more of them | the tails are free; more pilots would come out of the code and not remove a mix |
| Anchors in every symbol, unsignalled | the border cannot see a lens's bend inside the picture |
| The display rate in the word | the one property of an animated symbol no single frame shows |
| The transfer in the light: chunk ids, header and manifest blocks, BLAKE3 per chunk | a camera alone recovers and verifies a file, chunk by chunk |
| zstd a chunk at a time | a ratio near the best at a speed a phone keeps ahead of the channel with |
| No straddle cancellation, no shift pass, no bursts | patches; the pilots' phase lock and grid fit do their work at the source |
| No dithered black and white picture | it ruins range and pixel scaling |
| Clip ratio c = 2 | measured against 1.6 live (section 11) |

## 13. Open questions

Numbers are kept so references hold; closed items are gone.

- **Item 8. Clip ratio and tilt.** Every sender paints c = 2 and t = 0; whether they are format constants or free for a
  sender is not decided. The tilt waits on a replay's SNR by sub-channel (`scripts/exp/capture_snr.mjs`).
- **Item 10. Two brand colours.** Allowed by the palette rule (section 2, item 1); the mapping from grey to the two colours
  (linear in 8-bit values or in light) is not defined, and no sender paints them.
- **Item 11. The palette rule's 60% floor** is not measured on a phone; narrower luma ranges and a brand pair are to be read
  live.
- **Item 15. The display's second resample** is a sender's matter (9.5); the desktop sender's bilinear shrink is to be
  measured against a single resample.
- **Item 18. Test vectors.** A set for every code of the profile (a four-rate format in the 128 ring and one in the 256 ring,
  a resampled picture) is to be added to `liblizard/test/vectors.mjs`, with the computations behind 4.7, 4.9, 5.3 and
  7.10.
- **Item 20.** A 1:1 capture in the 64 or 256 ring, and a file transfer in either, recorded in today's format.
- **Item 21.** The largest pictures (n = 1536) under blur; the advice to cap apps at LIZARD-560 (3.5) rests on an older border.
- **Item 22.** Frame-change capture and phase holding on more than one phone.
- **Item 23.** The GPU decoder on other phones' GPUs.
- **Item 24.** A recording with a noisy surround.
- **Item 26.** The transfer on a phone: OPFS in other browsers, and the sender's memory (about 1.75 times the file).
- **Item 27.** The rate profile beyond one phone at 1080, at 720p and at range; its targets are the sub-channel split
  carried to the slots, read live at LIZARD-656 in the 256 ring on one phone and one monitor, and need paired live runs.
- **Item 28.** The pilots' readings: LIZARD-64's r2 rests on two 7/8 blocks' 16 coefficients and passes the
  reference's check (`scripts/exp/pilot_check.mjs`, `liblizard/test/pilot_wasm.mjs`); LIZARD-288's r2 read by the GPU
  decoder on clean captures of one paint, 0.83 to 0.86, is under the check's 0.9 where the C's is not. Whether the
  256 picture, which no format has since LIZARD-32 left, stays in the ladder of picture sizes (3.2).

## 14. Changes

The format is not frozen: a sender and a receiver need the same release, and older frames, words or replays may not
decode on a newer one. The history of the format is in the repository's log.

## 15. Replays: the .lizard file

A replay is a run of captures as a receiver's decoder was handed them: the crops it read, in order, with each one's
capture time. A receiver records one (Save replays), one device hands it to another to decode, and the reference tools
replay it. Every program that writes or reads one follows this section; the reference is `liblizard/core/rx/lizfile.h`
and `lizfile.cpp` (with `lizard_receive pack` and `unpack`), with a reader in `liblizard/sim/lizfile.mjs`.

### 15.1 The archive

- A ZIP archive (PKWARE's APPNOTE) of stored entries: compression method 0 for every entry, its compressed size equal
  to its uncompressed size, no encryption, no data descriptor, no ZIP64. A file is therefore under 4 GiB and holds at
  most 65,535 entries; a writer refuses past either.
- The end of central directory record's comment is the 15 ASCII bytes `LIZARD replay 1`: the kind of archive and the
  version of this section. A reader refuses an archive whose comment does not begin `LIZARD replay `, and one of a
  version it does not read.
- Its entries are exactly: `NNNN.zst` for each frame (its index from 0, in decimal, at least four digits with leading
  zeros: `0000.zst`, `0001.zst`, ..., `10000.zst`), then `stats.jsonl` where the run has rows, then `meta.json`. A writer
  puts them in that order, the frames as they are written and `meta.json` at the end; a reader finds each by the
  central directory and depends on no order. A frame entry missing for an index under meta.json's `frames` makes the
  file invalid.
- Every entry carries ZIP's CRC-32 of its bytes, which a reader checks. The reference writes the date 1980-01-01 00:00
  on every entry, so a run packs to the same bytes; a reader ignores dates and attributes.

### 15.2 meta.json

A JSON object in UTF-8. What a reader takes from it:

| field | meaning |
|---|---|
| `frames` | N, the frames in the file: entries 0 to N - 1 |
| `w`, `h` | a frame's size in samples where `sizes` gives none |
| `sizes` | `[w, h]` a frame, in order |
| `ms` | each frame's capture time, ms on one clock for the run (the camera's) |
| `axes` | each frame's `[x, y]`, in samples, where the camera's optical axis fell in it, or null (its centre) |
| `lens` | each frame's lens as the camera reported it, `[k1, k2, k3, f]` (f in the frame's samples), or null (none) |
| `layout` | `"1:1"` (a frame a capture) or `"2:1"` (a capture's two halves, below) |
| `config` | the format the light last named, `{"from": "band", "spec": {...}}` (`phy`, `n`, `subch`, `mode`, `span`, `bitmap`, `stream`), or null where no word was read |

In a `"2:1"` run each capture's two halves are two consecutive frames of one size with the same `ms`, the left half
first (the top half, where the halves lay along the camera's height); a half's axis is the middle of its inner edge,
so `axes` tells the halves apart where a run lost one. The other fields are the recorder's own and inform: `run` (its
name), `of` (the frames it kept at most), `taken` (the frames it saw), `decoder`, `error` (what ended its frames
early), `camera`, `zoom`, `batch`, `thermal`, `app`. `stats.jsonl` is the receiver's statistics, a JSON object a line,
a second apart, for reading the run's phase lock and rates; no frame depends on it.

### 15.3 A frame

Frame i is its capture's luma, w x h samples of 8 bits (`sizes[i]`, else `w` and `h`), in rows from the top. Its entry
is one zstd frame (RFC 8878) whose content is the frame's w h residuals in the same order:

    r(x, y) = (p(x, y) - pred(x, y)) mod 256

with pred JPEG-LS's median edge detector (ITU-T T.87, LOCO-I) over the neighbours to the left, above and above-left,
a = p(x - 1, y), b = p(x, y - 1), c = p(x - 1, y - 1), each 0 where it falls outside the frame (this edge rule, not
JPEG-LS's own, which repeats the row above):

    pred = min(a, b)      where c >= max(a, b)
           max(a, b)      where c <= min(a, b)
           a + b - c      otherwise

A reader decompresses the entry, refuses it unless it holds exactly w h bytes, and restores p(x, y) = (r(x, y) +
pred(x, y)) mod 256 in raster order. Every valid zstd frame decodes; the level is the writer's (the reference's 1).

### 15.4 Measured (non-normative)

A 960 x 960 half of a 2:1 replay packs to about 74% of its raw bytes at zstd level 1, near its residuals' order-0
entropy.
