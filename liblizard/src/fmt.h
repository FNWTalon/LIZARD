// The format word: the one thing in the border that carries information, and the only thing a receiver cannot
// derive for itself. It says WHICH FORMAT the symbol is and HOW FAST the sender means to paint.
//
// It carries the version because that is what precedent carries. QR's format word is 15 bits, 5 of data (error
// level and mask) and 10 of BCH, drawn twice. Aztec's mode message is 28 bits compact, 7 of data (layers and
// codeword count) with Reed-Solomon over the rest. Neither says anything about the TRANSFER: a QR symbol's mode
// indicator and character count live in the data stream, inside the payload's own error correction. LIZARD's do
// the same. Sub-channels are 8 * version, the picture size is the smallest that holds them (sim/lizard_pick.mjs
// N_FOR), the mode is always LDPC, the code rates are the rate profile the sub-channels give (focus.h
// focus_tiers_for), and the fountain's header is payload.
//
// It carries the display rate because that is the one thing about an ANIMATED symbol that no single frame shows.
// The version is a property of the picture in front of the camera; the rate is a property of a sequence the
// receiver only ever sees part of, and light is the only channel it has. What it buys the receiver: how long a
// frame is on screen, so a capture rate can be judged against it (a camera below the display rate cannot see
// every frame, whatever else is right), and the difference between a repeat and a frame that was missed. It is
// the rate the sender MEANS to paint at, not the rate it achieved, so a sender falling behind reads as what it
// is: the word says 60, the receiver counts 15 distinct frames a second, and the sender is the problem.
// 1 to 255 whole frames a second, and 0 for a sender that does not state one.
//
// Why it is a whole byte. 240 fps panels exist, so the field has to reach past the 60 a code table would have
// covered. A miscorrection that lands in the rate byte alone mis-states the rate, which no decode decision depends
// on, where one in the version byte costs every block.
//
// Why one code over every word cell (2026-09-24). The band has W word cells a side, B / 2 in ring B: 16, 32,
// 64 and 128 over the four rings (16, 32, 48, 64 while the 96 ring was the third, to 2026-10-10) (2B + 30 modules a side; S = n / 8 + 60 and W 16 to 72 by picture size before
// 2026-09-27). Whole copies of an 8-byte word summed before RS spent the extra cells on averaging; a word as long as
// the band spends them on parity, RS(W / 2, 3), so a side lost to glare is W / 8 erasures out of W / 2 - 3 roots: two
// faded sides read in every ring, three in the 128 and 256 (src/fmt.c, test/fmt_test.c; SPEC.md 5.4).
//
// Why it is not an anchor count. Counting anchors was built and measured on 2026-09-21: the version as the
// number of edge marks a side, three corner marks QR-style, no word at all. It works in a good capture and it
// costs what the word never did. The marks have to be 16 modules to be detected (a 12-module edge mark has a
// 2-module core, under the detector's floor), which pushes the border from 15 to 19; and five a side blank a
// quarter of the timing track. Measured over the standard twelve cells: fill 0.3 fell from 2521 B on 16 of 16
// frames to 381 B on 8, 45 degrees lost 24%, the phone cell 18%. The version was read on 93 of 129 captures
// against the word's 99 of 99, and a miscount loses every block of the frame where a missed word loses nothing.
// A count has no error correction and no way to gain any; Reed-Solomon does. It could not have carried the
// display rate at all.
#ifndef OB_FMT_H
#define OB_FMT_H
#include <stdint.h>

// OB_FMT_MAGIC: a fixed byte the decoder checks after Reed-Solomon. RS can miscorrect past its distance, and a
// wrong word is worse than none: it would send the receiver to another picture size and cost every block of
// every frame until it changed again. 0x4c is 'L'.
// OB_FMT_VERSION_MAX: the top version a word may state, LIZARD-1024's 128 (121 blocks under the rate profile). A byte
// could say 255; keeping the range to the formats that exist keeps it a check on a miscorrected word, with the magic.
// OB_FMT_BYTES: the shortest word, RS(8, 3), the 32 ring's, and OB_FMT_SIDE_CELLS the cells a side it needs.
// OB_FMT_BYTES_MAX: the longest a band holds, RS(64, 3), the 256 ring's (W = 128; RS(36, 3) until 2026-10-10, when the
// 128 ring's RS(32, 3) was the longest). A wider border paints its remaining word cells light.
enum { OB_FMT_MAGIC = 0x4c, OB_FMT_DATA = 3, OB_FMT_BYTES = 8, OB_FMT_BYTES_MAX = 64, OB_FMT_SIDE_CELLS = 16, OB_FMT_VERSION_MAX = 128 };

typedef struct { int version, fps; } ob_fmt_t;

// Cells a side the word takes (every word cell, W), and the codeword's length there: RS(4 floor(W / 8), 3), so 8,
// 16, 32 and 64 bytes in the 32, 64, 128 and 256 rings. Taken from the layout the receiver registered, which the module
// count gives before the word is read. 0 where the side is too short to hold the shortest word.
int ob_fmt_side_cells(int side, int reserve);
int ob_fmt_bytes(int side, int reserve);
// The word as `bytes` of codeword, `bytes` being ob_fmt_bytes(side): 3 data bytes and bytes - 3 roots. 0, or -1
// where the version does not fit. fps is 0 to 255, 0 being a sender that states no rate.
int ob_fmt_encode(const ob_fmt_t *f, uint8_t *cw, int bytes);
// Bit `cell` of frame side `edge` (0..3), as the word is laid out: byte k goes to side k % 4. 0 past the codeword.
int ob_fmt_bit(const uint8_t *cw, int bytes, int edge, int cell);
// q: soft values, q[side * cells + i] in -0.5 .. 0.5, positive for dark. A word is accepted only with the magic and a
// version in vlo .. vhi (within 1 .. OB_FMT_VERSION_MAX): every receiver passes 1 .. OB_FMT_VERSION_MAX, since any
// ring carries any version (2026-09-27). 0, or -1 where nothing decoded.
int ob_fmt_decode(const float *q, int cells, int bytes, int vlo, int vhi, ob_fmt_t *f);

#endif
