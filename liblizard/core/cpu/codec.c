// ai: codec.h over liblizard/src/any.h. The web decodes blind with gamma 1 and mesh 2 (sim/ob.mjs FocusAny.decode), and
// ai: so does this.
#include "codec.h"
#include "any.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

struct cpu_dec { focus_any_t any; ob_result_t res; int block_bytes; };

cpu_dec_t *cpu_dec_new(int nmax) {
  cpu_dec_t *d = calloc(1, sizeof *d);
  if (!d) return NULL;
  d->any.which = -1;
  d->block_bytes = focus_any_init(&d->any, nmax);
  if (d->block_bytes < 0) { free(d); return NULL; }
  return d;
}
void cpu_dec_free(cpu_dec_t *d) {
  if (!d) return;
  focus_any_free(&d->any);
  free(d);
}
int cpu_dec_top(const cpu_dec_t *d) { return focus_any_top(&d->any); }
int cpu_dec_block_bytes(const cpu_dec_t *d) { return d->block_bytes; }
int cpu_tiers_check(const char *spec, int *subch, int *blocks, char *label, int label_len) {
  focus_tier_t tier[FOCUS_TIERS];
  int s = 0;
  const int tiers = focus_tiers_parse(spec, tier, &s, label, label_len);
  if (subch) *subch = tiers ? s : 0;
  if (blocks) { int b = 0; for (int t = 0; t < tiers; t++) b += tier[t].blocks; *blocks = b; }
  return tiers;
}
int cpu_dec_tiers(cpu_dec_t *d, const char *spec, char *why, int why_len) {
  if (why && why_len > 0) why[0] = 0;
  if (!spec || !*spec) { focus_any_tiers(&d->any, NULL, 0); return 0; }
  focus_tier_t tier[FOCUS_TIERS];
  const int tiers = focus_tiers_parse(spec, tier, NULL, why, why_len);
  if (!tiers) return -1;
  if (focus_any_tiers(&d->any, tier, tiers)) { if (why && why_len > 0) { const char *t = "the receiver refused the profile"; int i = 0; for (; t[i] && i + 1 < why_len; i++) why[i] = t[i]; why[i] = 0; } return -1; }
  return tiers;
}

int cpu_dec_frame(cpu_dec_t *d, const uint8_t *img, int iw, int ih, int held, uint8_t *blocks, uint8_t *ok, cpu_frame_t *out) {
  memset(&d->res, 0, sizeof d->res);
  const int got = focus_any_frame(&d->any, img, iw, ih, 1.0f, 2, blocks, ok, held, &d->res);
  if (out) {
    const ob_fmt_t *fm = focus_any_word(&d->any);
    out->found = d->res.found; out->ring = d->any.which; out->n = d->any.n; out->held = d->any.held;
    out->word = fm != 0; out->version = fm ? fm->version : 0; out->fps = fm ? fm->fps : 0;
    out->total = d->any.last ? d->any.last->blocks : 0;
    memcpy(out->quad, d->res.quad, sizeof out->quad);
    out->ms_detect = d->res.ms_detect; out->ms_sample = d->res.ms_sample; out->ms_decode = d->res.ms_decode;
    out->pilot_blocks = focus_any_pilot(&d->any, out->pilot_r, out->pilot_sd);
  }
  return got;
}
void cpu_dec_block_stats(const cpu_dec_t *d, const int8_t **its, const float **est) {
  *its = 0; *est = 0;
  if (d->any.last) focus_block_stats(d->any.last, its, est);
}
// ai: focus.c's test hook, which the wasm build takes from dec.c (the binary grid code, left out here): armed
// ai: (nonzero) before a frame, it comes back as a hash of the floats and soft values behind that frame's decode. One
// ai: global, so it is armed from one thread only (the check tool's single-thread runs); the pool never arms it.
uint32_t ob_debug_hash;
void cpu_hash_arm(void) { ob_debug_hash = 1; }
uint32_t cpu_hash_take(void) { const uint32_t v = ob_debug_hash; ob_debug_hash = 0; return v; }
double cpu_prof_ms(int i) { return i >= 0 && i < PROF_N ? ob_prof_ms[i] : 0; }
void cpu_prof_reset(void) { for (int i = 0; i < PROF_N; i++) ob_prof_ms[i] = 0; }
int cpu_simd(void) {
#ifdef OB_SIMD
  return 1;
#else
  return 0;
#endif
}
