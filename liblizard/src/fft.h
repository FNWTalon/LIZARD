#ifndef OB_FFT_H
#define OB_FFT_H
// Radix-2 FFT built around one kernel that transforms along the FIRST index of a rows x cols block, every
// column at once. A butterfly is then two whole rows and one twiddle, so its inner loop runs over contiguous
// columns, four floats to a vector, and a block of 512 x 64 complex samples (256 KB) stays in cache however
// large the picture is. The textbook column pass, one column at a time with a stride of a row, spends most of
// its time missing the cache: 512 x 512 took six times as long as 256 x 256 for four times the data.
//
// re, im: the block's first row; row r starts at r * pitch (pitch >= cols, so a block can be a slice of a wider
// array). Forward is exp(-i), inverse exp(+i) and NOT scaled: callers that need the
// identity divide by rows themselves, and LIZARD normalizes by the picture's RMS anyway.
// ai: rows: 2^a 3^b 5^c. A power of two is radix 2 alone; 384, 768 and 1536 add one radix-3 stage, 1152 two (fft.c
// ai: rows3) and 1280 one radix-5 stage (rows5); any other size is not transformed. Radix 2 is the CPU transform (2026-09-23: radix 4 was 4 to 15% slower on the desktop and
// ai: a third slower on the S26 at a strip of 64, research/10 s12, wasm both; it left the code on 2026-10-25). The
// ai: format's: the encoder paints with it, and the decoder off arm64 and x86-64.
void fft_cols(float *re, float *im, int rows, int cols, int pitch, int inverse);
// ai: The decoder's forward transforms, every column of a rows x cols block. On arm64 (OB_NEON) their own kernels (radix
// ai: 12, 6, 5, 4, 3 and 2, fused multiply-adds, each stage's twiddles tabled), and on x86-64 (OB_X86) the same kernels
// ai: and plans in SSE4.1 (a multiply and an add where NEON fuses), which round otherwise than the format's
// ai: transform, the encoder's (test/fft_dec.c holds them to the definition), and keep rows in the digit-reversed order
// ai: of their own plan, so no row is moved for it: fft_dec_dif leaves frequency k in row place[k]; fft_dec_dit takes
// ai: sample t from row place[t] and leaves the frequencies in order; where keep < rows / 2 its last stage writes only
// ai: rows [0, keep] and [rows - keep, rows) (the others hold what the stage before left). place = fft_dec_place(rows),
// ai: the calling thread's, valid until it transforms a ninth size. Elsewhere both are fft_cols, rows in order, keep
// ai: unread, and fft_dec_place is NULL.
void fft_dec_dif(float *re, float *im, int rows, int cols, int pitch);
void fft_dec_dit(float *re, float *im, int rows, int cols, int pitch, int keep);
const int *fft_dec_place(int rows);

#endif
