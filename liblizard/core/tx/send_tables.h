// ai: The GPU sender's tables a format (2026-10-02, gpu_painter.h): what liblizard/gpu/encoder.mjs configure and
// ai: allocate build from the wasm, from the C's own codec here: the disc's rows and each coefficient's entry
// ai: (gpu/back/transform.mjs discLayout), the twiddles, the inverse's sizes and the clip (SU), the resampler's taps
// ai: (sim/ob.mjs resampleGeom: the C's), the geometry word G, the paint's uniform (gpu/wgsl/back_ldpc.mjs PARAMS_AT: the
// ai: format's block count, each tier's first block, first sub-channel and blocks (2026-10-07, the format's rate
// ai: profile), the stride, and the CRC powers gen/sender.mjs baked), and the border, the C's
// ai: paint of the symbol (its pixels outside the square: ring, marks, band, word, margin) as grey bytes. Held to the
// ai: web's byte for byte by `tx_check gputables` against gen/send_tables_ref.mjs (the twiddles to a float's last bit).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include "focus.h"
}

namespace lizard {

// ai: gen/sender.mjs's manifest (liblizard/out/setup/send.json): what no format changes.
struct SendConsts {
  int slots = 0, tposeTile = 16, rsvThreads = 64, rshThreads = 64, blockBytes = 480, clip = 2, bitmap = 0;
  int paramsSizes = 0, paramsDims = 0, paramsPw = 0, paramsWords = 0;
  std::vector<uint32_t> pw;
  std::vector<int> codeRate, codeSubs;   // ai: the paint's codes in TAB's order (gpu/wgsl/back_tiers.mjs CODES): ldpc.h rate, sub-channels a block
  std::vector<uint32_t> tab;    // ai: the paint's TAB (gpu/encoder.mjs sendTab): the codes' shapes, rows, entries and bit maps, the whitening
  std::vector<int> sizes;       // ai: the picture sizes, which the kernels irows<n> and ipic<n> are built for
  // ai: from the manifest's JSON text and the TAB blob's bytes; "" or why not
  std::string load(const std::string& manifest, const std::vector<uint8_t>& tab);
};

struct SendTables {
  int n = 0, subch = 0, blocks = 0, blockBytes = 0, npos = 0, Vr = 0;   // ai: Vr: the disc's rows (the web's tb.V)
  int q = 0, sq = 0, W = 0, FW = 0, RW = 0, FS = 0;                       // ai: the square, a symbol's side, the frame
  int codes = 1, gap = 0;       // ai: symbols a frame side by side, and the pixels between them (gpu/encoder.mjs shape)
  bool copy = false;            // ai: every tap the C's on-sample copy: rsh's copy form, no rsv
  std::vector<uint32_t> rows;   // ai: 2048 words (the uniform's size): (off, count, nonnegative, umaxN) a disc row
  std::vector<uint32_t> uv;     // ai: npos: each coefficient's entry of S
  std::vector<float> tw;        // ai: 2 n: cos, sin of -2 pi m / n
  std::vector<uint32_t> su;     // ai: 12 words: n, Vr, Vr n, npos, n n / 4, 0, 0, 0, then lim and 0.5 / lim as f32
  std::vector<uint32_t> g;      // ai: 12 words: n, q, sq, W, codes, FW, RW, FS, 1, the encode's first frame, gap, 0
  std::vector<uint32_t> taps;   // ai: 12 q words: 6 q sample indices (mod n), then 6 q weights as f32
  std::vector<uint8_t> border;  // ai: W x W grey, padded to a whole word
  std::vector<uint32_t> pu;     // ai: PARAMS_AT.words: sizes[0] (blocks), sizes[1 + c] (code c's tier: first block, first sub-channel, blocks), dims (blocks, npos, symbols an encode, 0), pw
};

// ai: The gap between two codes of a frame, in modules, where none is set (liblizard/gpu/encoder.mjs GAP_MODULES,
// ai: lizard-web/send-worker.mjs the same, since 2026-09-30); TxFormat.gap sets it
constexpr int GAP_MODULES = 12;

// ai: The tables of f's format (focus_init'd, its rate set: the border's word states it), encodes of up to `frames`
// ai: frames of `codes` symbols side by side (2026-10-03, the desktop sender's two codes: the frame codes W + gap wide, the
// ai: gap the margin's byte, rsh stepping W + gap; the paint's symbols frames x codes, a frame's codes one pilot count),
// ai: `gap` modules apart (2026-10-04). "" or why not (the disc not as the shader reads it).
std::string sendTables(const focus_t& f, const SendConsts& k, int frames, SendTables& t, int codes = 1, int gap = GAP_MODULES);

}  // namespace lizard
