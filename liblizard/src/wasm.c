// C ABI for the JS harness and the browser.
#include "ob.h"
#include "internal.h"
#include "focus.h"
#include "any.h"
#include "fft.h"
#include "ldpc.h"
#include "xfer.h"
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static ob_layout_t L;
static ob_result_t R;
static ob_opts_t O;
static int ready;

int ob_setup(int w, int h, int tx, int ty, int rate, int map, int pilot_step) {
  if (ready) { ob_layout_free(&L); ready = 0; }
  ob_cfg_t cfg = { .w = w, .h = h, .tx = tx, .ty = ty, .rate = rate, .map = map, .pilot_step = pilot_step };
  if (ob_layout_init(&L, &cfg)) return -1;
  ob_default_opts(&O);
  ready = 1;
  return L.block_bytes;
}
int ob_tiles(void) { return L.tiles; }
int ob_code_n(void) { return L.code.n; }
int ob_code_k(void) { return L.code.k; }
void ob_set_opts(float gamma, int eq_mode, int eq_taps, int eq_regions, int mesh, int max_iter, float gain, float clip) {
  O.gamma = gamma; O.eq_mode = eq_mode; O.eq_taps = eq_taps; O.eq_regions = eq_regions; O.mesh = mesh;
  O.max_iter = max_iter; O.llr_gain = gain; O.llr_clip = clip;
}
int ob_pilots(void) { return L.pilots; }
void ob_tx(const uint8_t *blocks, uint8_t *modules) { ob_encode(&L, blocks, modules); }
int ob_rx(const uint8_t *img, int iw, int ih, uint8_t *blocks, const uint8_t *truth) { return ob_decode(&L, img, iw, ih, &O, blocks, &R, truth); }
// Camera frames arrive as RGBA. Converting here keeps the 4 bytes a pixel out of JS loops.
static uint8_t *lum; static int lum_cap;
static const uint8_t *to_luma(const uint8_t *rgba, int n) {
  if (n > lum_cap) { free(lum); lum = malloc((size_t)n); lum_cap = lum ? n : 0; }
  if (!lum) return NULL;
  PROF(PROF_LUMA, ob_luma(rgba, n, lum));
  return lum;
}
// The same a strip at a time, so that the heap never holds a frame at four bytes a pixel: n pixels of RGBA go to
// luma[offset ..] of a buffer of `total`. Returns the buffer, which the last strip completes.
const uint8_t *ob_luma_strip(const uint8_t *rgba, int n, int offset, int total) {
  if (total > lum_cap) { free(lum); lum = malloc((size_t)total); lum_cap = lum ? total : 0; }
  if (!lum || offset < 0 || n < 0 || offset + n > total) return NULL;
  PROF(PROF_LUMA, ob_luma(rgba, n, lum + offset));
  return lum;
}
int ob_rx_rgba(const uint8_t *rgba, int iw, int ih, uint8_t *blocks, const uint8_t *truth) { const uint8_t *y = to_luma(rgba, iw * ih); return y ? ob_decode(&L, y, iw, ih, &O, blocks, &R, truth) : 0; }
const uint8_t *ob_luma_of(const uint8_t *rgba, int n) { return to_luma(rgba, n); }
const ob_result_t *ob_last(void) { return &R; }
extern uint32_t ob_debug_hash;
extern int8_t *ob_debug_llr;
// Test hooks: a buffer of w * h bytes to receive each decode's per-cell LLRs (0 to stop), and the layout's cell kinds.
void ob_dbg_llr(int8_t *buf) { ob_debug_llr = buf; }
const uint8_t *ob_kind(void) { return L.kind; }
uint32_t ob_dbg_hash(int arm) { uint32_t v = ob_debug_hash; ob_debug_hash = (uint32_t)arm; return v; }
double ob_prof(int i) { return i >= 0 && i < PROF_N ? ob_prof_ms[i] : 0; }
void ob_prof_reset(void) { for (int i = 0; i < PROF_N; i++) ob_prof_ms[i] = 0; }
int ob_result_size(void) { return (int)sizeof(ob_result_t); }

static focus_t F;
static int fready;
int focus_setup(int n, int subch, int mode, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border) {
  if (fready) { focus_free(&F); fready = 0; }
  if (focus_init(&F, n, subch, mode, clip, span, tilt, corner, corner_filled, centre, edge, track_alt, border)) return -1;
  fready = 1;
  return F.block_bytes;
}
// Up to three runs of blocks at different code rates (focus.h): rate index, blocks, sub-channels a block; a run of 0 blocks ends the list.
int focus_setup_tiers(int n, float clip, int span, float tilt, int corner, int corner_filled, int centre, int edge, int track_alt, int border, int r0, int b0, int s0, int r1, int b1, int s1, int r2, int b2, int s2) {
  const focus_tier_t all[FOCUS_TIERS] = { { r0, b0, s0 }, { r1, b1, s1 }, { r2, b2, s2 } };
  int tiers = 0;
  while (tiers < FOCUS_TIERS && all[tiers].blocks > 0) tiers++;
  if (fready) { focus_free(&F); fready = 0; }
  if (focus_init_tiers(&F, n, all, tiers, clip, span, tilt, corner, corner_filled, centre, edge, track_alt, border)) return -1;
  fready = 1;
  return F.block_bytes;
}
// ai: A receiver told nothing: the module's one blind receiver (any.h), behind the exports.
static focus_any_t ANY = { .which = -1 };
// ai: nmax: the largest picture this receiver decodes (1024 by default, 1536 for an industrial rig). Returns the
// ai: block bytes (473), or -1.
int focus_any_setup(int nmax) { return focus_any_init(&ANY, nmax); }
// ai: blocks and ok hold focus_any_max_blocks() blocks. held: the sub-channel count / 8 of the last word the caller
// ai: read (its version field), 0 for none. Returns the blocks decoded; focus_any_which says which ring registered
// ai: (-1 none), focus_any_n the picture finished (0 none), focus_any_held whether the held configuration stood in.
int focus_any_rx(const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok, int held) {
  return focus_any_frame(&ANY, img, iw, ih, gamma, mesh, blocks, ok, held, &R);
}
int focus_any_which(void) { return ANY.which; }
int focus_any_n(void) { return ANY.n; }
int focus_any_held(void) { return ANY.held; }
// ai: Pictures built since the setup, for a caller timing its first frames (the rig's workersFirst).
int focus_any_builds(void) { return ANY.builds; }
int focus_any_blocks(void) { return ANY.last ? ANY.last->blocks : 0; }
int focus_any_max_blocks(void) { return focus_any_top(&ANY); }
// ai: The word this frame's ring read, or 0: the held configuration is the caller's, never reported as read.
const ob_fmt_t *focus_any_fmt(void) { return focus_any_word(&ANY); }
// ai: the pilots the blind receiver's last frame read: out[0], out[1] r of the even and of the odd blocks, out[2],
// ai: out[3] their standard errors; the blocks read, 0 for none
int focus_any_pilot_get(float *out) { return focus_any_pilot(&ANY, out, out + 2); }

// The bit map (focus.h FOCUS_BITMAP_*) of the codec focus_setup made. A replay of a capture painted before
// 2026-09-23 sets FOCUS_BITMAP_NONE; nothing else needs to.
int focus_bitmap_set(int mode) { return fready ? focus_bitmap(&F, mode) : -1; }
int focus_bitmap_get(void) { return fready ? F.bitmap : -1; }

// What the last decode read out of the band, as the ints of an ob_fmt_t (fmt.h), or 0 for nothing read.
const ob_fmt_t *focus_fmt_rx_ptr(void) { return fready ? focus_fmt_rx(&F) : 0; }
// The display rate the painted band states (focus.h). A sender may call this between frames.
int focus_fmt_fps_set(int fps) { return fready ? focus_fmt_fps(&F, fps) : -1; }
// ai: The pilots (focus.h focus_parity, focus_pilot): the count mod 4 the next encode paints, and what the last told
// ai: decode read, out[0], out[1] r of the even and of the odd blocks, out[2], out[3] their standard errors; the
// ai: blocks read, 0 for none.
int focus_parity_set(int c) { return fready ? focus_parity(&F, c) : -1; }
int focus_pilot_get(float *out) { return fready ? focus_pilot(&F, 0, out, out + 2) : 0; }
// ai: the grid's shift the last decode read off the pilots and turned back (focus.h focus_align), for a measurement
// ai: that re-reads the coefficients focus_dbg_coef holds, which are the ones before the turn (2026-10-07)
int focus_align_get(float *out) { if (!fready) return 0; focus_align(&F, out); return 1; }
int focus_subch(void) { return F.subch; }
int focus_blocks(void) { return F.blocks; }
int focus_side(void) { return F.px; }
int focus_cell(void) { return F.pxm; }   // whole pixels a module in the painted symbol
int focus_quiet(void) { return FOCUS_QUIET; }   // the margin the codec paints, in modules, so the JS has no copy of its own
void focus_tx(const uint8_t *blocks, float *drive) { focus_encode(&F, blocks, drive); }
// The same encode and the page's pixels in one call, margin included: drive is the scratch the encode writes, rgba
// what is painted.
void focus_tx_rgba(const uint8_t *blocks, float *drive, uint8_t *rgba) { focus_encode(&F, blocks, drive); focus_paint_rgba(&F, drive, rgba); }
// ai: The resampler's geometry and taps (focus.h focus_resample_geom): out[0..3] q, the square's first drive pixel, n
// ai: and px, out[4] and out[5] the heap offsets of the q first samples and the q x 6 weights. 0, or -1 before a setup.
int focus_rs_geom(int *out) {
  const int *i0; const float *w;
  if (!fready || focus_resample_geom(&F, out, &i0, &w)) return -1;
  out[4] = (int)(intptr_t)i0; out[5] = (int)(intptr_t)w;
  return 0;
}
int focus_rx(const uint8_t *img, int iw, int ih, float gamma, int mesh, uint8_t *blocks, uint8_t *ok) { return focus_decode(&F, img, iw, ih, gamma, mesh, blocks, ok, &R); }
static const int8_t *dbg_sym; static const float *dbg_coef;
int focus_dbg(int on) { dbg_sym = 0; dbg_coef = 0; return fready ? focus_debug(&F, on, &dbg_sym, &dbg_coef) : -1; }
const int8_t *focus_dbg_sym(void) { return dbg_sym; }
const float *focus_dbg_coef(void) { return dbg_coef; }
const int8_t *focus_blk_its(void) { const int8_t *i; const float *e; focus_block_stats(&F, &i, &e); return i; }
const float *focus_blk_est(void) { const int8_t *i; const float *e; focus_block_stats(&F, &i, &e); return e; }
// One transform's cost on THIS machine, for scripts/exp/fft_bench.mjs and scripts/pages/fft.html: `reps` forward transforms of a
// rows x cols block, ms for all of them. A phone is the only place the radix and the strip width can be chosen,
// and this needs no camera, no sender and no second build.
// Repeated forward transforms grow the magnitudes, so the block is refilled every BATCH of them, outside the
// timing, and starts small enough that nothing reaches infinity inside a batch: white noise grows by about
// sqrt(rows) a transform, but a correlated fill concentrates the energy and grows by up to rows, and an infinity
// would make the rest of the batch measure something else. Hence a real PRNG and 1e-9: even at the correlated
// rate, 512^8 * 1e-9 is still finite. Eight transforms of a 512 x 64 strip is half a millisecond, well above
// clock()'s resolution.
double ob_fft_bench(int rows, int cols, int reps) {
  enum { BATCH = 8 };
  if (rows < 2 || cols < 1 || reps < 1) return -1;
  float *re = malloc((size_t)rows * cols * sizeof(float)), *im = malloc((size_t)rows * cols * sizeof(float));
  if (!re || !im) { free(re); free(im); return -1; }
  double ms = 0;
  uint32_t s = 2463534242u;
  for (int done = 0; done < reps; ) {
    for (int i = 0; i < rows * cols; i++) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; re[i] = 1e-9f * (float)(int32_t)s;
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; im[i] = 1e-9f * (float)(int32_t)s;
    }
    int n = reps - done < BATCH ? reps - done : BATCH;
    double t0 = ob_now_ms();
    for (int r = 0; r < n; r++) fft_cols(re, im, rows, cols, cols, 0);
    ms += ob_now_ms() - t0;
    done += n;
  }
  free(re); free(im);
  return ms;
}
// The top of the heap. sbrk never gives memory back here, so this is the most the module has had in use.
unsigned ob_heap_top(void) { return (unsigned)(uintptr_t)sbrk(0); }

// Test hook for a second implementation of ob_sample_grid (the first WebGPU port's, since deleted). Acquires
// one capture with the current FOCUS setup, writes out everything the sampler reads, then the grid it produces,
// so the two can be held to the same floats. A GPU cannot be byte-identical to the scalar build the way
// scripts/exp/simd_check.mjs holds the vector build, so what this exists for is a tolerance and an ulp count.
//   dims:  nx, ny, n, found
//   state: H[9], g0, step, node_x[nx], node_y[ny], nx * ny pairs of (dx, dy), lut[256], the sampler's own ms
//   grid:  n * n samples, row major, pitch n
// Returns n, or 0 if the capture did not register. The caller sizes the buffers: nx and ny are at most
// OB_MAX_NODES, so 9 + 2 + 2 * 40 + 2 * 40 * 40 + 256 + 1 floats is always enough.
static void state_dump(const image_t *im, const ob_reg_t *reg, int32_t *dims, float *state) {
  const ob_layout_t *L2 = &F.frame;
  const int nx = L2->nx, ny = L2->ny;
  dims[0] = nx; dims[1] = ny; dims[2] = F.n; dims[3] = 1;
  float *s = state;
  for (int i = 0; i < 9; i++) *s++ = reg->H.h[i];
  *s++ = F.margin + 0.5f / F.scale; *s++ = 1.0f / F.scale;
  for (int i = 0; i < nx; i++) *s++ = L2->node_x[i];
  for (int i = 0; i < ny; i++) *s++ = L2->node_y[i];
  for (int i = 0; i < nx * ny; i++) { *s++ = reg->nodes[i].dx; *s++ = reg->nodes[i].dy; }
  for (int i = 0; i < 256; i++) *s++ = im->lut[i];
}

int ob_test_sample(const uint8_t *img, int iw, int ih, float gamma, int mesh, int32_t *dims, float *state, float *grid) {
  if (!fready) return 0;
  const int n = F.n;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = n; dims[3] = 0;
  image_t im;
  ob_image_init(&im, img, iw, ih, gamma);
  ob_reg_t reg;
  if (!ob_acquire(&F.frame, &im, mesh, &reg, &R)) return 0;
  state_dump(&im, &reg, dims, state);
  const float g0 = F.margin + 0.5f / F.scale, step = 1.0f / F.scale;
  const double t0 = ob_now_ms();
  ob_sample_grid(&F.frame, &im, &reg, g0, g0, step, 0, n, 0, n, grid, n);
  state[9 + 2 + F.frame.nx + F.frame.ny + 2 * F.frame.nx * F.frame.ny + 256] = (float)(ob_now_ms() - t0);
  ob_reg_free(&reg);
  return n;
}

// The decode in two halves (focus.h), for a receiver that samples on the GPU. focus_rx_acquire registers and
// writes out what the sampler needs, in state_dump's layout above minus the trailing ms; focus_rx_finish takes
// the grid back. A caller that passes grid = 0 gets the sampler the decoder has always had, which is how the
// two arms are compared on one build.
int focus_rx_acquire(const uint8_t *img, int iw, int ih, float gamma, int mesh, int32_t *dims, float *state) {
  if (!fready) return 0;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = F.n; dims[3] = 0;
  if (!focus_acquire(&F, img, iw, ih, gamma, mesh, &R)) return 0;
  state_dump(focus_held_image(&F), focus_held_reg(&F), dims, state);
  return 1;
}
// Acquisition from a quad found elsewhere, so a GPU finder can be held to the decode it produces. Writes the
// same state focus_rx_acquire does.
int focus_rx_acquire_quad(const uint8_t *img, int iw, int ih, float gamma, int mesh, const float *quad, int orient, float score, int32_t *dims, float *state) {
  if (!fready) return 0;
  dims[0] = F.frame.nx; dims[1] = F.frame.ny; dims[2] = F.n; dims[3] = 0;
  if (!focus_acquire_quad(&F, img, iw, ih, gamma, mesh, &R, quad, orient, score)) return 0;
  state_dump(focus_held_image(&F), focus_held_reg(&F), dims, state);
  return 1;
}
// What the finder settled on for the last acquisition: the quad in image pixels, then orient and score.
void focus_rx_quad(float *out) {
  for (int k = 0; k < 8; k++) out[k] = ob_raw_quad[k];
  out[8] = (float)R.orient; out[9] = R.mark_score;
}
int focus_rx_finish(float *grid, uint8_t *blocks, uint8_t *ok) { return fready ? focus_finish(&F, grid, blocks, ok, &R) : 0; }

// Shapes the GPU pipeline has to allocate for: n, bw, vblocks, subch, blocks, and the coefficient index table's
// length. The table itself is focus_pos_ptr: subch * FOCUS_SUB ints into the spectrum (src/focus.c).
// The mark scan and the constants it runs on, for the first WebGPU port's finder to be held to (since deleted).
int ob_test_mark_scan_out(const uint8_t *bin, int w, int h, int merged, float *out, int cap) {
  return fready ? ob_test_mark_scan(&F.frame, bin, w, h, merged, out, cap) : -1;
}
int ob_test_mark_cfg_out(int32_t *out) { return fready ? ob_test_mark_cfg(&F.frame, out) : 0; }
// ai: The mesh tables, for a mesh built elsewhere (the first WebGPU port's, since deleted).
int ob_test_mesh_tables_out(int32_t *dims, float *nodeX, float *nodeY, uint8_t *kind, int32_t *mark) {
  return fready ? ob_test_mesh_tables(&F.frame, dims, nodeX, nodeY, kind, mark) : 0;
}
void ob_test_corner_coords_out(float *src) { if (fready) ob_test_corner_coords(&F.frame, src); }
int ob_test_fmt_plan(int cells, float *pts, int32_t *meta, int32_t *widx) {
  return fready ? ob_thin_fmt_plan(&F.frame, cells, pts, meta, widx) : 0;
}
// The track read's sample plan and a single hypothesis through it, and what the finder settled on last time,
// for the first WebGPU port's track read to be held to (since deleted).
int ob_test_thin_plan(float *pts, int32_t *meta, int32_t *signs) { return fready ? ob_thin_score_plan(&F.frame, pts, meta, signs) : 0; }
int ob_test_thin_score_out(const uint8_t *img, int w, int h, const float *centres, int hyp, int merged, float *out) {
  return fready ? ob_test_thin_score(&F.frame, img, w, h, centres, hyp, merged, out) : 0;
}
void ob_test_finder_out(float *out) { for (int k = 0; k < 8; k++) out[k] = ob_finder_info[k]; }
int ob_test_fmt_cells(void) { return fready ? ob_fmt_side_cells(F.frame.w < F.frame.h ? F.frame.w : F.frame.h, F.frame.reserve) : 0; }
int ob_test_mark_quads_out(const float *cands, int n, int merged, float *out) {
  return fready ? ob_test_mark_quads(&F.frame, cands, n, merged, out) : -1;
}

// The current layout's mark geometry, for archive/opencv-finder/opencv_finder.mjs.
int ob_test_marks(float *out) { if (!fready) return 0; ob_mark_geom(&F.frame, out); return 1; }
int focus_shape(int32_t *out) {
  if (!fready) return 0;
  out[0] = F.n; out[1] = F.bw; out[2] = F.vblocks; out[3] = F.subch; out[4] = F.blocks; out[5] = F.subch * FOCUS_SUB;
  return 1;
}
const int *focus_pos_ptr(void) { return fready ? F.pos : 0; }
void focus_tables_out(float *out) { if (fready) focus_tables(&F, out); }
void focus_spectrum_out(float *re, float *im) { if (fready) focus_spectrum(&F, re, im); }
// The decode finished from soft values quantised elsewhere: llr is subch runs of
// 2 * FOCUS_SUB int8, est one information estimate per sub-channel. The transform is not run.
int focus_rx_ldpc(const int8_t *llr, const float *est, uint8_t *blocks, uint8_t *ok) {
  return fready ? focus_finish_ext(&F, 0, llr, est, blocks, ok, &R) : 0;
}
// Which global sub-channel each block starts at, and how many it spans: what maps a flat per-sub-channel
// soft-value array onto the blocks.
void focus_block_subs(int32_t *first, int32_t *count) {
  if (!fready) return;
  for (int b = 0; b < F.blocks; b++) { first[b] = F.block_sub[b]; count[b] = F.tier[F.block_tier[b]].subs; }
}

// The LDPC code behind a block, for a decoder running elsewhere. tier is a block's tier
// index; focus_block_tiers says which block is on which. See ldpc.c ldpc_tables for the layout.
void focus_ldpc_tables(int tier, int32_t *dims, int32_t *lay) {
  if (fready && tier >= 0 && tier < FOCUS_TIERS) ldpc_tables(&F.code[tier], dims, lay);
}
void focus_block_tiers(int32_t *out) { if (fready) for (int b = 0; b < F.blocks; b++) out[b] = F.block_tier[b]; }
// The decode finished from soft values AND hard decisions worked out elsewhere. bits: blocks * code[0].n, one
// decision a byte. its: what each block spent, or 0 to say one iteration each.
int focus_rx_bits(const int8_t *llr, const float *est, const uint8_t *bits, const int8_t *its, uint8_t *blocks, uint8_t *ok) {
  return fready ? focus_finish_bits(&F, 0, llr, est, bits, its, blocks, ok, &R) : 0;
}
// The whole tail done on the device: verdicts and bytes in, blocks out, no registration.
int focus_rx_assemble(const uint8_t *bytes, int stride, const uint8_t *verdicts, const int8_t *its, const float *est, uint8_t *blocks, uint8_t *ok, const float *quad, int orient, float score) {
  return fready ? focus_assemble(&F, bytes, stride, verdicts, its, est, blocks, ok, &R, quad, orient, score) : 0;
}
int focus_fmt_check_out(const float *q) { return fready ? focus_fmt_check(&F, q) : 0; }
// The sampled grid's origin and step in module coordinates, what state_dump hands a sampler after an
// acquisition, for a sampler that runs without one.
void focus_grid_out(float *out) { out[0] = F.margin + 0.5f / F.scale; out[1] = 1.0f / F.scale; }

// The decline gate against the live configuration, for the first WebGPU port's gate to be held to (since deleted).
// Packing and the CRC itself need no state, so they are exported straight out of src/focus.c.
int ob_test_crc_gate_out(const float *est, float *blk_est, float *bar, uint8_t *declined) {
  return fready ? ob_test_crc_gate(&F, est, blk_est, bar, declined) : 0;
}

// ---- The transfer (xfer.h): the block id, the header and manifest blocks, BLAKE3 ----
// The C functions with plain arguments are exported as they are (xfer_manifest_blocks, _write, _parse, _check,
// xfer_root, xfer_chunk_cv, xfer_b3_hash); these wrap the rest for JS. Every id and uint32 comes back as a signed
// int32, so JS reads it with >>> 0. Lengths past 2^32 cross as doubles, exact to 2^53.

// The layout's numbers, so the JS keeps no copy: id bytes, payload bytes, symbol bits, chunk bits, the header's id,
// manifest block 0's id, entries a manifest block, name bytes at most, media type bytes at most, chunk_log2 least and
// most, chunks at most, layout version, BLAKE3's hash byte.
void xfer_layout(int32_t *out) {
  const int32_t v[] = { XFER_ID_BYTES, XFER_PAYLOAD, XFER_SYMBOL_BITS, XFER_CHUNK_BITS, (int32_t)XFER_ID_HEADER, (int32_t)XFER_ID_MANIFEST,
                        XFER_PER_BLOCK, XFER_NAME_MAX, XFER_TYPE_MAX, XFER_LOG2_MIN, XFER_LOG2_MAX, XFER_MAX_CHUNKS, XFER_VERSION, XFER_HASH_BLAKE3 };
  for (unsigned i = 0; i < sizeof v / sizeof *v; i++) out[i] = v[i];
}
uint32_t xfer_id_of(uint32_t chunk, uint32_t symbol) { return xfer_id(chunk, symbol); }
int xfer_kind_of(uint32_t id) { return xfer_id_kind(id); }

// The header's 469 bytes from its fields: length in bytes, the chunk size's log2, the codec, the one chunk's seed
// attempt and bytes as sent (0 and 0 unless the file is one chunk), the root's 32 bytes, and name and media type as
// bytes with their lengths (0 for none). 0, or -1 for fields the layout refuses.
int xfer_hdr_write(uint8_t *payload, double length, int chunk_log2, int codec, int seed, double sent, const uint8_t *root,
                   const uint8_t *name, int name_len, const uint8_t *type, int type_len) {
  xfer_header_t h;
  if (!(length >= 0 && length <= 9007199254740992.0) || (double)(uint64_t)length != length) return -1;
  if (!(sent >= 0 && sent <= 4294967295.0) || (double)(uint32_t)sent != sent) return -1;
  if (codec < 0 || codec > 255 || seed < 0 || seed > 255 || name_len < 0 || name_len > XFER_NAME_MAX || type_len < 0 || type_len > XFER_TYPE_MAX) return -1;
  if (xfer_header_init(&h, (uint64_t)length, chunk_log2)) return -1;
  h.codec = (uint8_t)codec; h.seed = (uint8_t)seed; h.sent = (uint32_t)sent;
  memcpy(h.root, root, XFER_CV);
  h.name_len = (uint8_t)name_len; if (name_len) memcpy(h.name, name, (size_t)name_len);
  h.type_len = (uint8_t)type_len; if (type_len) memcpy(h.type, type, (size_t)type_len);
  return xfer_header_write(&h, payload);
}
// A header block's payload read back. scalars, 13 int32: version, hash, chunk_log2, codec, the one chunk's seed,
// name_len, type_len, length's low 32 bits, its high 32 bits, chunks, manifest blocks, the chunk size in bytes, the one
// chunk's bytes as sent. root (32), name (255) and type (158) take the bytes, or may be 0. 0, or XFER_ERR (-1),
// XFER_ERR_VERSION (-2), XFER_ERR_HASH (-3), XFER_ERR_CODEC (-4), and nothing is written then.
int xfer_hdr_parse(const uint8_t *payload, int32_t *scalars, uint8_t *root, uint8_t *name, uint8_t *type) {
  xfer_header_t h;
  const int rc = xfer_header_parse(payload, &h);
  if (rc) return rc;
  const int32_t v[] = { h.version, h.hash, h.chunk_log2, h.codec, h.seed, h.name_len, h.type_len, (int32_t)(uint32_t)h.length,
                        (int32_t)(uint32_t)(h.length >> 32), (int32_t)h.chunks, xfer_manifest_blocks(h.chunks), 1 << h.chunk_log2, (int32_t)h.sent };
  for (unsigned i = 0; i < sizeof v / sizeof *v; i++) scalars[i] = v[i];
  if (root) memcpy(root, h.root, XFER_CV);
  if (name) memcpy(name, h.name, XFER_NAME_MAX);
  if (type) memcpy(type, h.type, XFER_TYPE_MAX);
  return 0;
}
// Manifest block m's payload read under the header payload `hdr` (xfer_manifest_parse): its entries into cvs (32 a
// chunk), sent (uint32 a chunk) and seeds (a byte a chunk) at 12 m. -1 for a block that is not this transfer's
// manifest block m, or a header that does not parse.
int xfer_mf_parse(const uint8_t *payload, const uint8_t *hdr, uint32_t m, uint8_t *cvs, uint32_t *sent, uint8_t *seeds) {
  xfer_header_t h;
  return xfer_header_parse(hdr, &h) ? -1 : xfer_manifest_parse(payload, &h, m, cvs, sent, seeds);
}
// 0 where chunk `index`'s bytes are right for the file the header payload describes (xfer_chunk_check): cvs is the
// chaining-value list once xfer_manifest_check has passed it, or 0 for a file of one chunk.
int xfer_chunk_ok(const uint8_t *payload, uint32_t index, const uint8_t *data, uint32_t len, const uint8_t *cvs) {
  xfer_header_t h;
  return xfer_header_parse(payload, &h) ? -1 : xfer_chunk_check(&h, index, data, len, cvs);
}
// xfer_b3_subtree with the 1 KiB chunk counter as a double (xfer_chunk_cv is the same thing by chunk index).
void xfer_b3_sub(const uint8_t *in, uint32_t len, double counter, int root, uint8_t *out) {
  xfer_b3_subtree(in, len, counter > 0 && counter < 18446744073709551616.0 ? (uint64_t)counter : 0, root, out);
}
