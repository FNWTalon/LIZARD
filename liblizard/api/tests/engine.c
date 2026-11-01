/* ai: The engines through lizard.h (2026-10-03), on a machine with the GPU's assets:
 * ai:   lizard_engine send <assets> <gpu|cpu|auto> [device]: a 3 MB file painted by liz_sender (LIZARD-480, two codes;
 * ai:     the frames through the GPU's ring, so the assets, whichever painter) and every frame read back by liz_decode
 * ai:     into a memory liz_rx: whole, its bytes and root the sender's.
 * ai: The folder runs (meta.json and NNNN.gray frames; receive and paint) went 2026-10-29: replays are .lizard files.
 * ai: Exit 1 on a failure. The lab runs GPU jobs under its scripts/tools/gpulock.sh share. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lizard.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static void sleep_ms(double ms) {
  if (ms <= 0) return;
  struct timespec t = {(time_t)(ms / 1e3), (long)((ms - 1e3 * (time_t)(ms / 1e3)) * 1e6)};
  nanosleep(&t, NULL);
}
static int painter_of(const char *s) { return !strcmp(s, "gpu") ? LIZ_PAINTER_GPU : !strcmp(s, "cpu") ? LIZ_PAINTER_CPU : LIZ_PAINTER_AUTO; }

static int send(const char *assets, const char *painter, const char *device) {
  const size_t bytes = 3000000;
  uint8_t *data = malloc(bytes);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < bytes; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; data[i] = (uint8_t)(s >> 32); }
  liz_sender *snd = liz_sender_new(data, bytes, "engine.bin", "", 0);
  liz_format f = {30, LIZ_RING_DEFAULT, 60, 2};
  if (!snd || liz_sender_configure(snd, &f, painter_of(painter), 4, assets, device) != LIZ_OK) {
    printf("FAIL sender: %s\n", liz_last_error()); return 1;
  }
  liz_geometry g;
  liz_sender_geometry(snd, &g);
  liz_decoder *d = liz_decoder_new(0);
  liz_rx *rx = liz_rx_new(NULL);
  uint8_t *frame = malloc((size_t)g.width * g.height), *verified = malloc((size_t)liz_decoder_max_blocks(d) * 2 * LIZ_BLOCK);
  int held = 0, frames = 0, missed = 0;
  liz_progress p = {0};
  const double t0 = now_ms();
  while (!p.done && frames < 400) {
    if (!liz_sender_take(snd, frame, g.width, LIZ_GREY8)) { sleep_ms(1); continue; }
    frames++;
    int total = 0;
    for (int k = 0; k < 2; k++) {   /* ai: each code read where it is painted, with no surround: the paint itself */
      const int got = liz_decode(d, frame + k * (g.side + g.gap), g.side, g.side, g.width, LIZ_GREY8, &held,
                                 verified + (size_t)total * LIZ_BLOCK, NULL);
      if (got > 0) total += got;
    }
    missed += g.frame_blocks - total;
    liz_verdict v;
    liz_rx_frame(rx, verified, total, &v);
    liz_rx_progress(rx, &p);
  }
  const char *stats = liz_sender_stats(snd);
  const uint8_t *out;
  size_t len;
  const int same = p.done && liz_rx_data(rx, &out, &len) == LIZ_OK && len == bytes && !memcmp(out, data, bytes);
  printf("send (%s): %d frames in %.1f s, %d blocks missed, %s; root %s\n  stats: %.300s\n", painter, frames, (now_ms() - t0) / 1e3,
         missed, same ? "the file whole" : "NOT whole", liz_rx_meta(rx, LIZ_META_ROOT), stats);
  liz_rx_free(rx); liz_decoder_free(d); liz_sender_free(snd);
  free(frame); free(verified); free(data);
  printf(same ? "send ok\n" : "send FAILED\n");
  return same ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc >= 4 && !strcmp(argv[1], "send")) return send(argv[2], argv[3], argc > 4 ? argv[4] : NULL);
  fprintf(stderr, "lizard_engine send <assets> <gpu|cpu|auto> [device]\n");
  return 2;
}
