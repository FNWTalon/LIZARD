// ai: A camera frame in a format the decoders do not read as it is (2026-10-27, the Android app's Format: P010, RAW16,
// ai: RAW10, RAW12), and its development to the 8-bit luma they read. One arithmetic, run in two places: on the CPU
// ai: (developLuma: the C decoder's frames, and any receiver without a GPU develop), and on the GPU decoder's device
// ai: inside its batch (stageRaw here, develop.comp there: the crop's samples staged as they are, developed into the
// ai: slot's layer), byte for byte alike.
// ai:
// ai: P010: each sample's top byte (its 10 bits sit in the top of a little-endian 16-bit sample). RAW16, RAW10, RAW12:
// ai: each sample less its 2 x 2 position's black level, times that position's white balance gain over (white -
// ai: black), clipped to 1 and taken to gamma 1 / 2.2 on 4096 steps (gammaLut); bin 2 averages each 2 x 2 quad (one
// ai: sample of each CFA position: a luma with no mosaic left), bin 1 keeps every sample (the screen's neutral grey
// ai: reads alike in every channel once balanced).
#pragma once
#include <cstddef>
#include <cstdint>

namespace lizard {

// ai: ImageFormat's values (android.graphics.ImageFormat), as the app hands them over
enum : int { RAW_FMT_RAW16 = 0x20, RAW_FMT_RAW10 = 0x25, RAW_FMT_RAW12 = 0x26, RAW_FMT_P010 = 0x36 };

// ai: The frame's plane 0 (rowStride bytes a row; pixelStride bytes a sample for RAW16 and P010; RAW10 four samples in
// ai: five bytes, the low bits last; RAW12 two in three), read only during the push that hands it over. The developed
// ai: frame (the CameraFrame's width x height) has its (0, 0) at sample (x0, y0) of the plane (both even), bin samples
// ai: a pixel each way (P010: 1). levels: the black level per 2 x 2 position (row-major, its (0, 0) the plane's first
// ai: sample), the white level, the gain per position (the app's Engine.levelsOf).
struct RawFrame {
  const uint8_t* plane = nullptr;
  uint32_t rowStride = 0, pixelStride = 2;
  int format = 0;
  uint32_t x0 = 0, y0 = 0, bin = 1;
  float levels[9] = {0, 0, 0, 0, 1023, 1, 1, 1, 1};
};

// ai: Whether the format is one this develops.
bool rawFormatKnown(int format);
// ai: The bytes a row of w samples from the row's start takes in the format (the plane's bound).
size_t rawRowBytes(int format, uint32_t w, uint32_t pixelStride);
// ai: The 4097 steps of gamma 1 / 2.2: round(255 (i / 4096)^(1 / 2.2)).
const uint8_t* gammaLut();
// ai: Each position's scale to the LUT's steps: gain x 4096 / max(1, white - black). Both develops use these floats.
void rawScales(const RawFrame& r, float k[4]);
// ai: The developed frame's w x h from (ox, oy) into out, rows outStride apart (the CPU's develop).
void developLuma(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint8_t* out, size_t outStride);

// ai: The GPU's develop (develop.comp) reads a staging buffer of a header of RAW_STAGE_HEADER words, then the rows of
// ai: the plane about the crop, each `pitch` bytes (a multiple of 4), its first sample an even column of the plane.
// ai: The header, in u32 words: the format's kind (0 P010, 1 RAW16, 2 RAW10, 3 RAW12), w, h, bin, pitch, pixelStride,
// ai: the crop's first sample in a staged row, the CFA row of the first staged row (0 or 1), the black levels (4
// ai: floats), the scales (rawScales, 4 floats). Out: w x h bytes, row after row.
enum : uint32_t { RAW_STAGE_HEADER = 16 };
// ai: The staging bytes the crop of w x h developed pixels at (ox, oy) takes, the header included.
size_t rawStageBytes(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h);
// ai: That crop staged into dst (rawStageBytes of room); the bytes written.
size_t stageRaw(const RawFrame& r, uint32_t ox, uint32_t oy, uint32_t w, uint32_t h, uint8_t* dst);

}  // namespace lizard
