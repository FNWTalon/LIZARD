// FOCUS (Hermans et al., MobiSys 2016) as the paper specifies it, plus a soft-decision variant.
//
// Data lives in the spectrum of a grey image: QPSK symbols (phase only) on the coefficients of
// the upper half-plane, taken in order of increasing frequency, 320 to a sub-channel. The
// lower half-plane is the conjugate reflection, the inverse FFT is real, and it is clipped to
// tame the peak-to-average ratio and shown as grey levels.
//   FOCUS_RS    the paper: 64-byte fragment + 16 RS parity bytes per sub-channel, hard QPSK
//   FOCUS_LDPC  same modulator; per-coefficient LLRs into the QC-IRA LDPC, 8 sub-channels a block
// What the paper leaves open and this file chooses: image size n (the paper says "e.g. 512"),
// the clipping ratio, equal magnitude on every used coefficient, and a CRC per fragment (the
// paper says "e.g. by using a checksum"). The paper's 16 circle markers and calibrated lens
// undistortion are replaced by the border (layout.h: the ring, four corner marks, and a band of
// timing track and format word), which is all the receiver registers on.
#ifndef OB_FOCUS_H
#define OB_FOCUS_H
#include "ob.h"
#include "fmt.h"

// THE code rate, not a parameter. 3/4 (ldpc.h index 4) and nothing else, because the rate is a bet on a capture
// the sender cannot see and 3/4 is the only setting that is good on both sides of it. Measured over ten cells
// (focus_ringrate.mjs, focus_ratemix.mjs, archived in archive/rates-by-ring/exp/): 7/8 everywhere is +10% where
// the capture matches the sender's assumption and -42% at 360 px, the protected plans are the reverse, and cycling
// plans frame to frame is exactly averaging (+0.5% mean, 0.84x worst). 3/4 sits on the efficient frontier between the
// two failure modes.
// A rig that KNOWS its capture can still spend that +10%: focus_init_tiers takes rates explicitly and is what
// scripts/exp/focus_sweep.mjs uses (and focus_ringrate.mjs did). It is not in the format word, so no receiver is ever told.
enum { FOCUS_RATE = 4 };

// ai: THE RINGS (2026-09-27: three ring sizes; that evening 32, 64 and 128, then four: 32, 64, 96 and 128). The
// ai: dotted band holds FOCUS_RING[r] cells of 2 x 2 modules on a side, the border round it is 15 modules (a 12-module
// ai: mark, a 3-module guard) and the corner clearance 15, so a side is S = 2 B + 30 modules (94, 158, 222, 286) and
// ai: the picture spans 2 B (64, 128, 192, 256). The ring only locates the symbol and syncs its grid, as 5G's sync block
// ai: does; which picture is inside is the format word's to say (sub-channels, so n), and any ring may
// ai: carry any picture: a caller names a ring by its span. With none named the ring is FOCUS_RING_DEFAULT, the 128
// ai: since 2026-10-01 (the 64 before), whatever the picture: no ring follows the version, even by default. Until that evening the default followed n
// ai: (32 / 48 / 64, rings 32, 48, 64, that morning), before that the side itself, n / 8 + 60 (92 / 124 / 188 / 316, 2026-09-23),
// ai: and n / 4 + 30 that morning.
//
// The ring and the picture are independent grids. The ring is painted at pxm = ceil(n / span) pixels a module, a
// whole number, so no module is ever split; the picture keeps its n samples and is resampled to fill its span
// modules (focus_encode, Lanczos-3), never shrunk. The decoder reads the picture on its own n-sample grid in module
// coordinates, a step of 1 / scale modules, and never sees the painter's pixels.
// ai: A span that divides n has scale == pxm and the resampling is an exact copy (which ones, below).
// ai: The 32 and 64 rings' spans (64, 128) divide every picture size, so their pictures are exact copies; the 96
// ai: ring's (192) divides 384, 768 and 1536 and resamples 256, 512 and 1024; the 128 ring's (256) divides all but 384.
enum { FOCUS_RINGS = 4, FOCUS_RING_DEFAULT = 3 };
extern const int FOCUS_RING[FOCUS_RINGS];

// How a block's codeword lies on its coefficients (FOCUS_LDPC). A block is 8 sub-channels, a thin ring of the
// spectrum taken in order of rising frequency, two slots a coefficient. Until 2026-09-23 slot i carried codeword bit
// i and nothing else: the payload's own bits were the picture's phases, so a payload of runs (a small file padded
// out to its blocks) painted thousands of coefficients at one phase, a peak the clip flattened, and the frame did
// not decode (LIZARD-64, every block 100 payload bytes then zeros: 6 of 64 blocks on a clean capture, against 64);
// and the codeword being systematic, data first, its 1272 weakest bits, the degree-2 parity, sat on the ring's
// outermost slots, the ones blur and distance take first.
//   NONE     the map as it was, slot i = bit i. Only for captures painted before this (sim/phy.mjs recordedSpec).
//   WHITE    slot i = bit i XOR the frame's whitening sequence at that slot (focus.c whiten_fill, PRBS-23).
//   SPREAD   whitened, and a parity bit on every n / m-th slot, data in order between.
//   LINEAR   whitened, and slot i = bit i P mod n, P near n over the golden ratio: every run of the code spread.
//   INNER    whitened, and the parity on the innermost slots.
// FOCUS_BITMAP is the format's: LINEAR. The others are here for scripts/exp/bitmap_race.mjs, which chose it.
//
// Whitening is what fixes the small payload (every block 100 bytes then zeros: 6 of 64 blocks to 64 of 64, and
// all zero the same). Where the bits go matters only where a frame is losing blocks, and most of all where only
// its first one or two survive: every format's first block runs from the centre of the spectrum out to 40 cycles,
// so its reliability falls steeply across it. At 8 px of motion blur, 48 frames, LIZARD-64 read 674 B a frame with
// the bits in order and 870 linear, LIZARD-128 557 and 791; INNER did as well (899, 723) and over the whole race the
// four whitened maps were within 1% of each other elsewhere. LINEAR over INNER because it spreads every run of the
// code over the ring rather than betting on which way the damage runs, which also covers the narrow-band damage
// the simulator cannot make: moire at one spatial frequency lands on consecutive slots of one block.
enum { FOCUS_BITMAP_NONE = 0, FOCUS_BITMAP_WHITE = 1, FOCUS_BITMAP_SPREAD = 2, FOCUS_BITMAP_LINEAR = 3, FOCUS_BITMAP_INNER = 4,
       FOCUS_BITMAP = FOCUS_BITMAP_LINEAR };
enum { FOCUS_RS = 0, FOCUS_LDPC = 1, FOCUS_SUB = 320, FOCUS_FRAG = 64, FOCUS_PAR = 16, FOCUS_GROUP = 8, FOCUS_STRIP = 64, FOCUS_TIERS = 3 };

// A run of blocks sent at one code rate (FOCUS_LDPC). Low frequencies survive blur and distance and high ones do
// not, so a frame can spend a high rate where a capture is nearly always clean and a low one further out
// (research/10). Every block carries the same payload, because the fountain above wants blocks of one size: a block
// at a lower rate is a longer codeword and takes more sub-channels, 8 at 3/4, 9 at 2/3, 12 at 1/2.
typedef struct { int rate, blocks, subs; } focus_tier_t;   // LDPC rate index (ldpc.h), how many blocks, sub-channels to a block

// Named, so internal.h can forward-declare it for the two accessors that hand back what focus_acquire is holding.
typedef struct focus_s {
  int n, subch, mode, blocks, block_bytes;
  float scale;            // picture samples a frame module, n / span (span = 2 FOCUS_RING[ring])
  int pxm, px;            // the painter's whole pixels a module, and the drive's side in them
  int margin;             // border modules between the symbol's edge and the picture: OB_THIN at least, 15 in the format
  float clip;
  // The picture's power is fixed (the display's range over the clipping ratio) and shared by every coefficient sent.
  // tilt, in dB, slopes the share from the first sub-channel down to the last, linearly in dB: the low frequencies
  // are what a blurred or distant capture still reads, and a sharp one has SNR to spare at the high ones. Only the
  // sender needs to know: the receiver scales each sub-channel by what it measures there. 0 is the paper's equal share.
  float tilt;
  // The transform never holds the whole plane. The picture goes through a strip of FOCUS_STRIP columns at a time
  // (transformed along y), and only the rows v < n / 2 are kept, since the lower half-plane is the upper's mirror.
  // They are kept as blocks of bw columns v, each n rows u, which is the shape the second pass (along x) wants:
  // its butterflies then run over contiguous memory. Blocks past the last used coefficient do not exist.
  int bw, vblocks;
  int *pos;               // subch * 320 indices into that array ((v / bw) * n * bw + u * bw + v % bw, u wrapped), low frequency first
  ob_layout_t frame;      // the border round the picture: ring, corner marks, band (layout.h)
  int tiers;                                  // FOCUS_LDPC: the frame's blocks from the lowest frequencies outwards
  focus_tier_t tier[FOCUS_TIERS];
  ldpc_t code[FOCUS_TIERS];
  int *block_tier, *block_sub, *block_bit;    // per block: its tier, its first sub-channel, its first bit among all the frame's bits
  struct focus_ws *ws;    // ai: transform arrays and tables: allocated once, written by encode and decode
  ob_fmt_t fmt;           // what the frame's format word says (fmt.h); the PHY half restates this struct
  uint8_t fmt_cw[OB_FMT_BYTES_MAX];   // the word as painted, ob_fmt_bytes of it, all zero where the configuration has none
  int bitmap;             // FOCUS_BITMAP_*, set by focus_bitmap; FOCUS_BITMAP from init
  int parity;             // ai: the painted picture's count mod 4, which its blocks' tails carry (the pilots: focus_parity)
} focus_t;

// The word the last decode read out of the symbol, or 0 where it read none.
const ob_fmt_t *focus_fmt_rx(const focus_t *f);
// What the band says about the display rate: 1 to 255 whole frames a second the sender MEANS to paint at, 0 for
// a sender that states none (the default, so nothing that does not care is changed). -1 where fps is out of
// range. Repaints the word, which is safe at any time.
int focus_fmt_fps(focus_t *f, int fps);
// ai: The pilots (SPEC 7.3; 2026-09-30): the slots of a block past its codeword (FOCUS_LDPC, a
// ai: bit map on: 32 at the top of its last sub-channel) carry the whitening XOR a bit of c, the sender's count of
// ai: painted pictures mod 4: bit 0 on an even block, so its tail flips sign picture to picture, and bit 1 on an odd
// ai: block, whose tail flips every second picture (signed since 2026-10-01; one bit, the count mod 2, on
// ai: every block the day before). So a capture of picture k that holds a share a of the picture before and b of the
// ai: one after reads, in size, 1 - 2 a - 2 b on the even blocks and 1 - 2 a (k even) or 1 - 2 b (k odd) on the odd
// ai: ones: which neighbour leaks, and from the two signs k mod 4, so a picture shown twice or skipped is read, not
// ai: guessed. focus_parity sets the c the next encode paints (0, the default, paints what every sender painted
// ai: before the pilots). 0, or -1.
int focus_parity(focus_t *f, int c);
// ai: What the last decode read from them (FOCUS_LDPC, a bit map on, the soft values its own), over its first `blocks`
// ai: blocks (0: every block; a receiver finishing on a codec bigger than the frame's format passes the format's):
// ai: r[0] over the even blocks and r[1] over the odd, each the mean over its blocks of a tail's axis values against
// ai: the whitening's signs, in units of its sub-channel's rms axis value sqrt(m2 / 2); +1 where the picture's bit is
// ai: 0 and -1 where it is 1 at a high signal-to-noise ratio (sqrt(A2 / m2) of it at any). On a capture a share f of
// ai: whose light came from pictures of the other bit: 1 - 2 f where each sample came from one picture or the other
// ai: (a tear), (1 - 2 f) / sqrt(1 - 2 f + 2 f^2) where every sample is the same blend (the blend lowers m2); either
// ai: way 0 at an even mix and largest in size unmixed. sd[0], sd[1]: their standard errors, the blocks' spread over
// ai: the square root of their count. Returns the blocks read (0: none, e.g. soft values made elsewhere).
int focus_pilot(const focus_t *f, int blocks, float r[2], float sd[2]);
// ai: The grid's shift the last decode read off the pilots and turned back before reading its soft values (focus.c
// ai: pilot_align): samples along u and v, 0, 0 where the pilots did not tell one from none (or soft values made
// ai: elsewhere). A decoder's own doing, not the format's: the pilots are known symbols at every radius.
void focus_align(const focus_t *f, float d[2]);
// corner: side in modules of the solid corner mark (layout.h). 0 takes the default, negative asks for none.
// corner_filled: keep depth 3 light through it, which is what makes the mark detectable. Ignored when corner is 0.
int focus_init(focus_t *f, int n, int subch, int mode, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border);
// The general form of FOCUS_LDPC: up to FOCUS_TIERS runs of blocks. focus_init is one tier of subch / 8 blocks, 8 sub-channels each.
int focus_init_tiers(focus_t *f, int n, const focus_tier_t *tier, int tiers, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border);
void focus_free(focus_t *f);
// blocks: blocks * block_bytes. drive: px^2 values in 0..1, the frame at pxm pixels a module and the picture resampled into it.
void focus_encode(const focus_t *f, const uint8_t *blocks, float *drive);
// ai: The resampler focus_encode paints the picture with, for an encoder elsewhere (the sender's GPU encoder,
// ai: 2026-09-29): out[0] q, the pixels across the picture and its guard interval; out[1] their first pixel in the drive
// ai: on either axis, (margin - cp) pxm; out[2] n; out[3] px. *i0 is q first samples (unwrapped: read mod n) and *w
// ai: q x 6 weights, the taps both passes read. 0, or -1.
int focus_resample_geom(const focus_t *f, int *out, const int **i0, const float **w);
// Measurement: keeps the bit pair (+1 / -1 per axis) the last encode put on each coefficient and the coefficient
// the last decode read there, both in pos order, two values a coefficient. For experiments, off by default.
int focus_debug(focus_t *f, int on, const int8_t **sym, const float **coef);
// Per block of the last decode (FOCUS_LDPC): LDPC iterations used (-1 never converged, -2 declined, 0 not tried), and
// the information per coded bit its sub-channels were estimated to hold before decoding.
void focus_block_stats(const focus_t *f, const int8_t **its, const float **est);
// ok[b] = 1 where block b passed its checksum. Returns the count.
int focus_decode(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The same decode in two halves, so the sampler can run somewhere the caller cannot wait on inline: a GPU fence
// costs more than the stage does unless several frames are outstanding, and one blocking call can have one.
// focus_acquire registers the symbol and holds the result; focus_finish takes it up again, sampling the held
// image itself when grid is NULL or reading the caller's grid when it is not. focus_decode is the two in a row.
// The image the caller passed must outlive the matching focus_finish, and a supplied grid is scratch: the
// transform runs in place over it. grid layout: strip s of FOCUS_STRIP columns at s * n * FOCUS_STRIP, n rows.
int focus_acquire(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res);
// ai: A receiver told nothing (2026-09-27). rings: a
// ai: codec a ring (FOCUS_RING), whose layouts the finder registers against, a bounded search over the side counts
// ai: like the eight orientations; the one that registered comes back in *which, holding the registration, with the
// ai: format word read (focus_fmt_rx). The word names the sub-channels, so the picture; the caller hands the
// ai: registration to the codec for that picture in that ring (focus_hand_over) and finishes there.
int focus_acquire_ring(const focus_t *const *rings, int k, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res, int *which);
// ai: The registration `from` holds, moved to `to`, a codec of the same ring (the same layout) at the picture the word
// ai: named; `to` reads the word again and finishes the frame. 0 when from holds nothing or the rings differ.
int focus_hand_over(const focus_t *from, const focus_t *to);
// ai: A held registration let go of unfinished (a frame whose picture is not known).
void focus_release(const focus_t *f);
// The picture size a format takes (sim/lizard_pick.mjs N_FOR).
// ai: FOCUS_PICTURES: the sizes the ladder takes, smallest first (sim/lizard_pick.mjs PICTURE_SIZES).
enum { FOCUS_NPICTURES = 6 };
extern const int FOCUS_PICTURES[FOCUS_NPICTURES];
int focus_n_for(int subch);
// The margin: modules of white round the frame, part of the symbol (2026-09-23) and painted by the codec, not
// asked of the page. With the frame's own outermost module, which is light, the ring the decoder reads has three
// modules of light outside it, and the finder assumes light there: noise painted right up to the ring left LIZARD-256
// 64 to 388 of its 1446 blocks (archive/build-scratch-2026-09/ring_race.mjs). It lies outside the module grid, whose origin stays the ring's
// outer corner, so nothing a decoder reads moves with it.
enum { FOCUS_QUIET = 2 };
// The symbol as a page paints it, RGBA, grey levels round(drive * 255), the margin round it (FOCUS_QUIET modules at
// pxm pixels a module; lizard-web/send.mjs through lizard-web/send-worker.mjs). rgba holds W^2 * 4 bytes, W = px + 2 FOCUS_QUIET pxm.
void focus_paint_rgba(const focus_t *f, const float *drive, uint8_t *rgba);
// The bit map both ends use, FOCUS_BITMAP_* (FOCUS_LDPC only). 0, or -1 for a mode that does not exist.
// ai: The same symbol as grey levels, one byte a pixel (what the GPU painter writes; the senders keep frames grey,
// ai: 2026-10-07): grey holds W^2 bytes, the margin 255, every level as focus_paint_rgba's.
void focus_paint_grey(const focus_t *f, const float *drive, uint8_t *grey);
int focus_bitmap(focus_t *f, int mode);
// The same with the quad found elsewhere: eight floats in image pixels, its orientation and its track score.
int focus_acquire_quad(const focus_t *f, const uint8_t *img, int iw, int ih, float gamma, int mesh, ob_result_t *res,
                       const float *quad, int orient, float score);
int focus_finish(const focus_t *f, float *grid, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The same with the soft values already quantised somewhere else: ext_llr is subch runs of 2 * FOCUS_SUB int8,
// one per global sub-channel, and ext_est the information estimate of each, which the decline gate reads. The
// grid is then unused and the transform is not run, so a caller supplying these supplies the spectrum's whole
// product. Everything from the decline gate onwards is shared with the ordinary path.
int focus_finish_ext(const focus_t *f, float *grid, const int8_t *ext_llr, const float *ext_est, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The same with the LDPC run elsewhere too: ext_bits is blocks * code[0].n hard decisions, one a byte, and
// ai: ext_its what each block spent. The decline gate and the CRC stay here, so acceptance is identical.
int focus_finish_bits(const focus_t *f, float *grid, const int8_t *ext_llr, const float *ext_est, const uint8_t *ext_bits, const int8_t *ext_its, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// The tail alone, from verdicts and bytes worked out elsewhere; and the format word from soft values alone.
int focus_assemble(const focus_t *f, const uint8_t *bytes, int stride, const uint8_t *verdicts, const int8_t *its, const float *est,
                   uint8_t *blocks, uint8_t *ok, ob_result_t *res, const float *quad, int orient, float score);
int focus_fmt_check(const focus_t *f, const float *q);
// What focus_acquire is holding is declared in internal.h, where image_t and ob_reg_t live: it is the sampler's
// business, not the format's, and this header is the format's.
// For a caller running the transform elsewhere: the detrend basis it needs, and the spectrum to be held to.
void focus_tables(const focus_t *f, float *out);          // cx[n], qx[n], x1r[n], x1i[n], x2r[n], x2i[n], sx2, sq2
void focus_spectrum(const focus_t *f, float *re, float *im);   // vblocks * n * bw of each

#endif
