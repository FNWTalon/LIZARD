/* ai: The engines through lizard.h (2026-10-03, the library's P4 check), on a machine with the GPU's assets:
 * ai:   lizard_engine receive <assets> <run dir> <gpu|cpu|auto> [device]: a run (meta.json {"w", "h", "frames"} and the
 * ai:     frames 0000.gray on, w x h grey bytes each: a recording, or paint's) pushed at 60 frames a second through
 * ai:     liz_receiver, laps over it until its file is whole and verified (the receiver's own BLAKE3 check against the
 * ai:     header's root), or 4 laps;
 * ai:   lizard_engine paint <run dir> [frames] [blocks]: a run made with the library alone (the project's recordings are
 * ai:     not published): a 400 kB file painted by liz_sender on the CPU (64 frames of 40 blocks by default), each frame
 * ai:     on white a quarter taller than it, the code in the centre square, as the C test reads one;
 * ai:   lizard_engine send <assets> <gpu|cpu|auto> [device]: a 3 MB file painted by liz_sender (LIZARD-480, two codes)
 * ai:     and every frame read back by liz_decode into a memory liz_rx: whole, its bytes and root the sender's.
 * ai: Exit 1 on a failure. The lab runs GPU jobs under its scripts/tools/gpulock.sh share. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir(p, 0777)
#endif

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
static void say(void *user, const char *line) { (void)user; printf("  %s\n", line); }
static int released = 0;
static void release(void *user, uint64_t tag) { (void)user; (void)tag; __atomic_add_fetch(&released, 1, __ATOMIC_RELAXED); }
static int decoder_of(const char *s) { return !strcmp(s, "gpu") ? LIZ_DECODER_GPU : !strcmp(s, "cpu") ? LIZ_DECODER_CPU : LIZ_DECODER_AUTO; }
static int painter_of(const char *s) { return !strcmp(s, "gpu") ? LIZ_PAINTER_GPU : !strcmp(s, "cpu") ? LIZ_PAINTER_CPU : LIZ_PAINTER_AUTO; }

static uint8_t *read_all(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *n = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n + 1);
  *n = fread(b, 1, *n, f);
  b[*n] = 0;
  fclose(f);
  return b;
}
static int json_int(const char *json, const char *key) {
  char k[64];
  snprintf(k, sizeof k, "\"%s\"", key);
  const char *p = strstr(json, k);
  return p ? atoi(strchr(p, ':') + 1) : 0;
}

static int receive(const char *assets, const char *run, const char *decoder, const char *device) {
  char path[4096];
  size_t n;
  snprintf(path, sizeof path, "%s/meta.json", run);
  uint8_t *meta = read_all(path, &n);
  if (!meta) { printf("FAIL no %s\n", path); return 1; }
  const int w = json_int((char *)meta, "w"), h = json_int((char *)meta, "h"), frames = json_int((char *)meta, "frames");
  free(meta);
  if (w < 1 || h < 1 || frames < 1) { printf("FAIL %s: w %d, h %d, frames %d\n", path, w, h, frames); return 1; }
  uint8_t **px = calloc((size_t)frames, sizeof *px);
  for (int i = 0; i < frames; i++) {
    snprintf(path, sizeof path, "%s/%04d.gray", run, i);
    px[i] = read_all(path, &n);
    if (!px[i] || n != (size_t)w * h) { printf("FAIL %s\n", path); return 1; }
  }
  char store[4096];
  snprintf(store, sizeof store, "%s/lizard-engine-%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)(now_ms() / 1000) % 100000);
  liz_receiver_config c = {0};
  c.assets = assets; c.store_dir = store; c.device = device; c.decoder = decoder_of(decoder); c.layout = 1; c.log = say;
  liz_receiver *r = liz_receiver_new(&c, release);
  if (!r) { printf("FAIL no receiver: %s\n", liz_last_error()); return 1; }
  const double t0 = now_ms();
  int pushed = 0;
  const char *file = "";
  for (int i = 0; i < 4 * frames + 180 && !*file; i++) {
    sleep_ms(t0 + i * 1000.0 / 60 - now_ms());
    liz_receiver_push(r, px[i % frames], w, h, w, (int64_t)(now_ms() * 1e6), (uint64_t)i);
    pushed++;
    file = liz_receiver_file(r);
  }
  for (int k = 0; k < 300 && !*file; k++) { sleep_ms(10); file = liz_receiver_file(r); }   /* ai: the last batch in flight */
  const char *stats = liz_receiver_stats(r), *totals = strstr(stats, "\"totals\"");
  printf("receive (%s): %d frames pushed in %.1f s, %d released, %d decoded, %d blocks verified, file %s\n  stats: %.300s\n",
         decoder, pushed, (now_ms() - t0) / 1e3, released, totals ? json_int(totals, "frames") : -1,
         totals ? json_int(totals, "blocks") : -1, *file ? file : "NOT whole", stats);
  const int ok = *file != 0;
  liz_receiver_free(r);
  for (int i = 0; i < frames; i++) free(px[i]);
  free(px);
  printf(ok ? "receive ok\n" : "receive FAILED\n");
  return ok ? 0 : 1;
}

/* ai: dir and the folders above it, made where missing (mkdir -p; a failure shows as the frames' writes failing) */
static void make_dirs(const char *dir) {
  char p[4096];
  snprintf(p, sizeof p, "%s", dir);
  for (char *c = p; *c; c++)
    if (c > p && (*c == '/' || *c == '\\')) { const char k = *c; *c = 0; make_dir(p); *c = k; }
  if (*p) make_dir(p);
}

static int paint(const char *run, int frames, int blocks) {
  if (frames < 1) { printf("FAIL %d frames\n", frames); return 1; }
  const size_t bytes = 400000;
  uint8_t *data = malloc(bytes);
  uint64_t s = 0x2545f4914f6cdd1dull;
  for (size_t i = 0; i < bytes; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; data[i] = (uint8_t)(s >> 32); }
  liz_sender *snd = liz_sender_new(data, bytes, "run.bin", "", 0);
  liz_format f = {blocks, LIZ_RING_DEFAULT, 60, 1};
  if (!snd || liz_sender_configure(snd, &f, LIZ_PAINTER_CPU, 4, NULL, NULL) != LIZ_OK) {
    printf("FAIL sender: %s\n", liz_last_error()); return 1;
  }
  liz_geometry g;
  liz_sender_geometry(snd, &g);
  const int side = g.height + g.height / 4, x0 = (side - g.width) / 2, y0 = (side - g.height) / 2;
  const size_t n = (size_t)side * side;
  uint8_t *frame = malloc((size_t)g.width * g.height), *canvas = malloc(n);
  char path[4096];
  const char *failed = NULL;
  make_dirs(run);
  for (int i = 0; i < frames && !failed;) {
    if (!liz_sender_take(snd, frame, g.width, LIZ_GREY8)) { sleep_ms(1); continue; }
    memset(canvas, 255, n);
    for (int y = 0; y < g.height; y++) memcpy(canvas + (size_t)(y0 + y) * side + x0, frame + (size_t)y * g.width, (size_t)g.width);
    snprintf(path, sizeof path, "%s/%04d.gray", run, i++);
    FILE *fp = fopen(path, "wb");
    if (!fp || fwrite(canvas, 1, n, fp) != n) failed = path;
    if (fp) fclose(fp);
  }
  if (!failed) {
    snprintf(path, sizeof path, "%s/meta.json", run);
    FILE *fp = fopen(path, "w");
    if (!fp || fprintf(fp, "{\"w\": %d, \"h\": %d, \"frames\": %d}\n", side, side, frames) < 0) failed = path;
    if (fp) fclose(fp);
  }
  liz_sender_free(snd);
  free(frame); free(canvas); free(data);
  if (failed) { printf("FAIL %s not written\n", failed); return 1; }
  printf("paint: %d frames of %d x %d into %s, LIZARD-%d (n %d), a file of %zu B\n", frames, side, side, run, 8 * blocks, g.n, bytes);
  return 0;
}

static int send(const char *assets, const char *painter, const char *device) {
  const size_t bytes = 3000000;
  uint8_t *data = malloc(bytes);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < bytes; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; data[i] = (uint8_t)(s >> 32); }
  liz_sender *snd = liz_sender_new(data, bytes, "engine.bin", "", 0);
  liz_format f = {60, LIZ_RING_DEFAULT, 60, 2};
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
  if (argc >= 5 && !strcmp(argv[1], "receive")) return receive(argv[2], argv[3], argv[4], argc > 5 ? argv[5] : NULL);
  if (argc >= 3 && !strcmp(argv[1], "paint")) return paint(argv[2], argc > 3 ? atoi(argv[3]) : 64, argc > 4 ? atoi(argv[4]) : 40);
  if (argc >= 4 && !strcmp(argv[1], "send")) return send(argv[2], argv[3], argc > 4 ? argv[4] : NULL);
  fprintf(stderr, "lizard_engine receive <assets> <run dir> <gpu|cpu|auto> [device] | paint <run dir> [frames] [blocks] |"
                  " send <assets> <gpu|cpu|auto> [device]\n");
  return 2;
}
