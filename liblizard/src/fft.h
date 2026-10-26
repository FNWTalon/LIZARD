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
// identity divide by rows themselves, and FOCUS normalizes by the picture's RMS anyway.
// ai: rows: a power of two, or 3 times one (384, 768, 1536), which takes one radix-3 stage (fft.c rows3); any other
// ai: size is not transformed. Radix 2 is the CPU transform (2026-09-23: radix 4 was 4 to 15% slower on the desktop and
// ai: a third slower on the S26 at a strip of 64, research/10 s12; it left the code on 2026-10-25).
void fft_cols(float *re, float *im, int rows, int cols, int pitch, int inverse);

#endif
