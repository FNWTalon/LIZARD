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
// ai: size is not transformed.
void fft_cols(float *re, float *im, int rows, int cols, int pitch, int inverse);

// Which radix fft_cols uses from here on: 2 (the default) or 4; any other value takes 2 (radix 8 has no kernel,
// fft.c). A size the radix does not divide takes one leading radix-2 stage and the rest at the full radix. The output
// differs between radices in its last bits, the order of the additions not being the same, so a build that wants
// to be compared byte for byte against another must be on the same one. scripts/exp/fft_bench.mjs and scripts/pages/fft.html time
// them; a phone is the only place the choice can be made.
void fft_set_radix(int r);
int fft_get_radix(void);

// n x n complex samples in place, rows then columns. inverse = 1 divides by n * n, so forward then inverse is the
// identity. For tests and for anything that wants the whole plane; FOCUS only ever needs half of it and uses
// fft_cols directly (src/focus.c).
void fft2d(float *re, float *im, int n, int inverse);
#endif
