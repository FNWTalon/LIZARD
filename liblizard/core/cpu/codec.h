// ai: The C decoder as the CPU path takes it: liblizard/src's blind receiver (any.h) behind a
// ai: C++-safe face, one object a thread. A frame in (tight 8-bit luma), its verified blocks out.
#ifndef CPU_CODEC_H
#define CPU_CODEC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpu_dec cpu_dec_t;

// ai: What a frame's decode found. ring: the ring that registered (0 to 3), -1 none. n: the picture it was finished
// ai: at, 0 not finished. held: the caller's held word stood in for the frame's own. word: the frame's own word read
// ai: (version = sub-channels / 8, fps the display rate it states). total: the blocks the finished picture's top
// ai: carries (ok[] and blocks[] hold that many). quad: the symbol's corners in the image, TL TR BR BL. pilot_*: what
// ai: the block tails read (liblizard/src/any.h focus_any_pilot: r of the even blocks and of the odd, their standard
// ai: errors, the blocks read, 0 none).
typedef struct {
  int found, ring, n, held, word, version, fps, total;
  float quad[8];
  int pilot_blocks;
  float pilot_r[2], pilot_sd[2];
  float ms_detect, ms_sample, ms_decode;
} cpu_frame_t;

// ai: nmax: the largest picture decoded (1024; 1536 for an industrial rig). NULL where the codec refuses.
cpu_dec_t *cpu_dec_new(int nmax);
void cpu_dec_free(cpu_dec_t *d);
// ai: The most blocks a frame can carry (the size of ok[], and of blocks[] in cpu_dec_block_bytes each).
int cpu_dec_top(const cpu_dec_t *d);
// ai: A rate profile (any.h focus_any_tiers; the lab's rate-by-ring arm, 2026-10-07): spec as focus_tiers_parse reads
// ai: it ("7/8:20,3/4:20,1/2:11"), NULL or "" for one rate. The frames whose word names the profile's sub-channels
// ai: decode on its tiers. Returns the tiers set (0 none), or -1 with why in `why` (why_len bytes); cpu_dec_top may
// ai: grow with it (size the buffers after).
int cpu_dec_tiers(cpu_dec_t *d, const char *spec, char *why, int why_len);
// ai: What a spec names, without a decoder: its tiers (1 to 3), subch its sub-channels, blocks its blocks a frame,
// ai: label its text ("7/8 x 20, 3/4 x 20, 1/2 x 11"); or 0, why in label.
int cpu_tiers_check(const char *spec, int *subch, int *blocks, char *label, int label_len);
int cpu_dec_block_bytes(const cpu_dec_t *d);
// ai: One frame: img is iw x ih luma, rows iw apart, alive for the call. held: the version field of the last word the
// ai: caller read, 0 none (a frame whose own word does not read is decoded at it, and with none held is not decoded).
// ai: Returns the blocks that verified; block b is good where ok[b] is 1.
int cpu_dec_frame(cpu_dec_t *d, const uint8_t *img, int iw, int ih, int held, uint8_t *blocks, uint8_t *ok, cpu_frame_t *out);
// ai: The last frame's per-block LDPC iterations and information estimates (out->total of each), for a check.
void cpu_dec_block_stats(const cpu_dec_t *d, const int8_t **its, const float **est);
// ai: The codec's own digest of a frame's decode: arm before cpu_dec_frame, take after (a hash of the floats and soft
// ai: values behind the decode where the frame was finished, else 1). From one thread at a time only.
void cpu_hash_arm(void);
uint32_t cpu_hash_take(void);
// ai: The codec's stage timers for the calling thread (internal.h PROF_*), ms since the last reset: acquire's parts,
// ai: sample, detrend, transform, LLR, LDPC.
double cpu_prof_ms(int i);
void cpu_prof_reset(void);
// ai: 1 where the codec's vector paths are compiled in (simd.h), else 0.
int cpu_simd(void);

#ifdef __cplusplus
}
#endif
#endif
