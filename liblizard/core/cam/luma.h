// ai: A frame's bytes to the luma plane the receiver reads (2026-10-25), for every layout camera.h's Pix names but
// ai: MJPEG (jpeg.h). The Y plane of a YUV layout is taken as it is: planar ones in place (no copy), packed ones every
// ai: other byte. Higher bit depths keep their top 8 bits. A Bayer mosaic becomes the mean of each 2 x 2 window of it,
// ai: which always holds one red, two green and one blue sample whatever the mosaic's phase (a luma blurred by half a
// ai: pixel and moved by a quarter of one, the same everywhere in the frame, which registration absorbs). RGB becomes
// ai: (77 R + 150 G + 29 B + 128) / 256: a grey code's channels are alike, so the weights barely matter.
// ai: Each layout, a sample's luma as toLuma computes it (rows `stride` apart, any width unless it says):
// ai:   Grey8, NV12, NV21, I420, YV12: the first plane's bytes (the chroma planes after it are not read);
// ai:   Grey16: each little-endian sample v as min(255, v >> (bits - 8)), bits 8 to 16 (Mode::bits);
// ai:   YUYV, YVYU: bytes 0 and 2 of every 4; UYVY, VYUY: bytes 1 and 3 (two bytes a pixel: pixel x at 2x, 2x + 1);
// ai:   Bayer8: out(x, y) = (in(x, y) + in(x + 1, y) + in(x, y + 1) + in(x + 1, y + 1) + 2) >> 2, x + 1 and y + 1
// ai:     clamped to the last column and row (which so take their own samples twice);
// ai:   RGB24, BGR24, BGRA32: (77 R + 150 G + 29 B + 128) >> 8, the alpha byte not read.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "camera.h"

namespace lizard::cam {

// ai: Whether a frame of this layout starts with its luma plane at the frame's own stride (Grey8 and the planar YUV
// ai: layouts): the sink can then be handed the camera's bytes with no copy.
bool lumaInPlace(Pix pix);

// ai: The bytes of one row of the layout's first plane, the least stride a frame of width w can have: w for Grey8,
// ai: Bayer8 and the planar YUV layouts, 2w for Grey16 and the packed 4:2:2 ones, 3w for RGB24 and BGR24, 4w for
// ai: BGRA32; 0 for MJPEG and Unknown.
size_t lumaRowBytes(Pix pix, uint32_t w);

// ai: The bytes the luma of a w x h frame at `stride` reads: (h - 1) * stride plus the last row's lumaRowBytes. A
// ai: planar layout's chroma planes come after these and are not counted, so a frame this long converts and a shorter
// ai: one is refused. 0 for MJPEG and Unknown, which have no size of their own, and for an empty frame.
size_t frameBytes(Pix pix, uint32_t w, uint32_t h, uint32_t stride);

// ai: The frame's luma into out (w x h, rows of w, resized as needed), as this file's header gives each layout. bits:
// ai: Grey16's significant bits (8 to 16), not read for the others. False with why where the layout is
// ai: MJPEG or Unknown, the frame is empty, its stride is under lumaRowBytes, it is shorter than frameBytes, or bits is
// ai: out of range. One pass over the rows, no branch a pixel.
bool toLuma(Pix pix, uint8_t bits, const uint8_t* data, size_t bytes, uint32_t w, uint32_t h, uint32_t stride,
            std::vector<uint8_t>& out, std::string* why);

}  // namespace lizard::cam
