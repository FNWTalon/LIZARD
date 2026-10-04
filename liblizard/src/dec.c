// Receiver top level: register, sample every cell in grey, demap to LLRs, decode each block.
#include "internal.h"
#include <stdlib.h>
#include <string.h>

// Test hook (scripts/exp/simd_check.mjs): nonzero asks each decode to leave a hash of its samples and LLRs here.
uint32_t ob_debug_hash;
// Test hook (fec_pipe.mjs, archived in archive/binary-grid/exp/): when set, each decode leaves its per-cell LLRs
// here (w * h, > 0 = light).
int8_t *ob_debug_llr;

// Give up on a block that is making no headway, the way focus.c already does (FOCUS_STALL_IT). A block that
// will never decode used to burn all max_iter iterations and a cleared test with each of them, and a capture
// where nothing decodes is where the decoder is slowest: over scripts/sim/camera.mjs, FEC went from 0.17 ms a frame
// with every block landing to 1.02 with 40% of them failing. The two counts replace those iterations' own
// cleared test, so a block that decodes normally pays nothing for this.
#define OB_STALL_IT 9
#define OB_STALL_RATIO 0.95f

void ob_default_opts(ob_opts_t *o) {
  o->gamma = 1.0f; o->eq_mode = OB_EQ_BLIND; o->eq_taps = 3; o->eq_regions = 1; o->mesh = 1;
  o->max_iter = 50; o->llr_gain = 0.7f; o->llr_clip = 10.0f;
}

int ob_decode(const ob_layout_t *L, const uint8_t *img, int iw, int ih, const ob_opts_t *o, uint8_t *blocks, ob_result_t *res, const uint8_t *truth) {
  memset(res, 0, sizeof *res);
  double t0 = ob_now_ms();
  image_t im;
  ob_image_init(&im, img, iw, ih, o->gamma);
  ob_reg_t reg;
  int got = ob_acquire(L, &im, o->mesh, &reg, res);
  res->ms_detect = (float)(ob_now_ms() - t0);
  if (!got) { ob_reg_free(&reg); return 0; }
  t0 = ob_now_ms();

  int cells = L->w * L->h;
  float *s = malloc((size_t)cells * sizeof(float));
  int8_t *llr_cell = malloc((size_t)cells);
  PROF(PROF_SAMPLE, ob_sample_cells(L, &im, &reg, s));
  ob_reg_free(&reg);
  PROF(PROF_DEMAP, ob_demap(L, s, o, truth, llr_cell, res));
  if (ob_debug_llr) memcpy(ob_debug_llr, llr_cell, (size_t)cells);
  if (ob_debug_hash) { uint32_t hsh = 2166136261u; for (int i = 0; i < cells; i++) hsh = (hsh ^ (uint8_t)llr_cell[i]) * 16777619u; for (int i = 0; i < cells; i++) { uint32_t b; memcpy(&b, s + i, 4); hsh = (hsh ^ b) * 16777619u; } ob_debug_hash = hsh | 1; }
  free(s);
  res->ms_sample = (float)(ob_now_ms() - t0);
  t0 = ob_now_ms();

  const ldpc_t *code = &L->code;
  int8_t *llr = malloc((size_t)code->n);
  uint8_t *bits = malloc((size_t)code->n), *buf = malloc((size_t)L->block_bytes + 4);
  for (int t = 0; t < L->tiles; t++) {
    for (int b = 0; b < code->n; b++) { int cell = L->tile_cells[t][b]; int8_t v = llr_cell[cell]; llr[b] = ob_scramble(cell) ? (int8_t)-v : v; }
    int it;
    PROF(PROF_LDPC, it = ldpc_decode_stall(code, llr, bits, o->max_iter, OB_STALL_IT, OB_STALL_RATIO));
    res->iters[t] = (int8_t)it;
    if (it < 0) continue;
    int B = L->block_bytes;
    memset(buf, 0, (size_t)B + 4);
    for (int i = 0; i < (B + 4) * 8; i++) buf[i >> 3] |= (uint8_t)(bits[i] << (7 - (i & 7)));
    uint32_t crc = ob_crc32(buf, B), got = (uint32_t)buf[B] | (uint32_t)buf[B + 1] << 8 | (uint32_t)buf[B + 2] << 16 | (uint32_t)buf[B + 3] << 24;
    if (crc != got) continue;
    memcpy(blocks + (size_t)t * B, buf, (size_t)B);
    res->ok[t] = 1; res->tiles_ok++;
  }
  free(llr); free(bits); free(buf); free(llr_cell);
  res->ms_decode = (float)(ob_now_ms() - t0);
  return res->tiles_ok;
}
