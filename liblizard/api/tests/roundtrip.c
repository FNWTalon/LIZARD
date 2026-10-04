/* ai: The C API end to end, through lizard.h alone (2026-10-03, the library's P2 check):
 * ai:   1. every format (128 block counts x 4 rings x 1 or 2 codes): an encoder made, its geometry the arithmetic's;
 * ai:   2. the picker and the room against the web's (a table tests/pick_ref.mjs printed from sim/lizard_pick.mjs), when
 * ai:      its path is given;
 * ai:   3. the test stream in every ring, one code and two, painted and read back: every block verified, every frame the
 * ai:      test stream's, none bad; then one payload byte wrong: one bad;
 * ai:   4. a file (LIZ_TEST_MB, 9 by default: three 4 MiB chunks) through two codes in a 2:1 frame (each half found by
 * ai:      liz_layout_rects) into a memory store, and through one code into a directory store: whole, its root the
 * ai:      sender's, its bytes the file's.
 * ai: Usage: lizard_roundtrip [pick table] [store dir]. Exit 1 on any failure.
 * ai: lizard_roundtrip --dump <dir>: the test stream's first frame (first id 1, picture 0) of 26 blocks in every ring, one
 * ai: code and two, painted grey into <dir>/paint-26-<ring>-<codes>.grey (dir made where missing), for the wasm
 * ai: binding's test to hold its own paint to (bindings/wasm/test). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define make_dir(p) mkdir(p, 0777)
#endif

#include "lizard.h"

static int faults = 0;
#define CHECK(ok, ...) do { if (!(ok)) { faults++; printf("FAIL "); printf(__VA_ARGS__); printf(" (%s)\n", liz_last_error()); } } while (0)

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint8_t next_byte(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)(rng >> 32); }

static void hex(const uint8_t *p, int n, char *out) {
  for (int i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", p[i]);
}

static void formats(void) {
  int made = 0;
  for (int codes = 1; codes <= 2; codes++)
    for (int ring = 0; ring < LIZ_RINGS; ring++)
      for (int b = 1; b <= LIZ_MAX_BLOCKS; b++) {
        liz_format f = {b, ring, 60, codes};
        liz_geometry g, e;
        liz_encoder *enc = liz_encoder_new(&f);
        CHECK(enc, "LIZARD-%d in ring %d, %d codes: no encoder", 8 * b, ring, codes);
        if (!enc) continue;
        CHECK(liz_geometry_of(&f, &g) == LIZ_OK && liz_encoder_geometry(enc, &e) == LIZ_OK && !memcmp(&g, &e, sizeof g),
              "LIZARD-%d ring %d: geometry", 8 * b, ring);
        CHECK(g.width == codes * g.side + (codes - 1) * g.gap && g.frame_blocks == codes * b, "LIZARD-%d: frame size", 8 * b);
        liz_encoder_free(enc);
        made++;
      }
  liz_format bad = {0, 0, 60, 1};
  CHECK(!liz_encoder_new(&bad) && liz_last_error()[0], "a format of 0 blocks refused, with a message");
  printf("formats: %d encoders made, their geometry the arithmetic's\n", made);
}

static void picker(const char *path) {
  FILE *f = fopen(path, "r");
  CHECK(f, "the pick table %s", path);
  if (!f) return;
  char kind[8];
  int rows = 0, worst = 0;
  double room_err = 0;
  while (fscanf(f, "%7s", kind) == 1) {
    if (!strcmp(kind, "room")) {
      int b, r; double want;
      if (fscanf(f, "%d %d %lf", &b, &r, &want) != 3) break;
      const double got = liz_room_for(b, r), d = got > want ? got - want : want - got;
      if (d > room_err) room_err = d;
      CHECK(d < 1e-9 * want, "room for %d blocks in ring %d: %.12f, the web's %.12f", b, r, got, want);
    } else if (!strcmp(kind, "pick")) {
      double w, h; int codes, r, top, want;
      if (fscanf(f, "%lf %lf %d %d %d %d", &w, &h, &codes, &r, &top, &want) != 6) break;
      const int got = liz_pick(w, h, codes, r, top);
      if (got != want) worst++;
      CHECK(got == want, "pick %.0f x %.0f, %d codes, ring %d, top %d: %d, the web's %d", w, h, codes, r, top, got, want);
    } else break;
    rows++;
  }
  fclose(f);
  CHECK(rows > 0, "rows in the pick table");
  printf("picker: %d rows against the web's, %d picks differ, rooms within %.3g px\n", rows, worst, room_err);
}

/* ai: a frame painted, then read as a camera frame would hold it: on white, a quarter taller than the frame (a symbol cut
 * ai: to its 2-module margin is missed now and then, 2 of 200 in the 32 ring, where a surround reads every one), one code
 * ai: in a square read by its centre (layout 1), two in a 2:1 canvas read by its halves (layout 2) */
static int read_frame(liz_decoder *d, const uint8_t *frame, const liz_geometry *g, int codes, int *held, uint8_t *verified,
                      uint8_t *canvas) {
  const int ch = g->height + g->height / 4, cw = codes * ch;
  memset(canvas, 255, (size_t)cw * ch);
  const int x0 = (cw - g->width) / 2, y0 = (ch - g->height) / 2;
  for (int y = 0; y < g->height; y++) memcpy(canvas + (size_t)(y0 + y) * cw + x0, frame + (size_t)y * g->width, (size_t)g->width);
  int rect[2][4], total = 0;
  const int n = liz_layout_rects(cw, ch, codes, rect);
  for (int k = 0; k < n; k++) {
    const int got = liz_decode(d, canvas + (size_t)rect[k][1] * cw + rect[k][0], rect[k][2], rect[k][3], cw, LIZ_GREY8, held,
                               verified + (size_t)total * LIZ_BLOCK, NULL);
    if (got > 0) total += got;
  }
  return total;
}

static void test_stream(liz_decoder *d) {
  int frames = 0, blocks = 0, missed = 0, judged_bad = 0;
  for (int codes = 1; codes <= 2; codes++)
    for (int ring = 0; ring < LIZ_RINGS; ring++) {
      liz_format f = {26, ring, 60, codes};
      liz_geometry g;
      liz_geometry_of(&f, &g);
      liz_encoder *enc = liz_encoder_new(&f);
      /* ai: a fixed first id: the C's own misses of a clean frame (about 1 in 200) are
       * ai: measured elsewhere, and this checks the API's path */
      liz_tx *tx = liz_tx_new_test(1u + 1000u * (unsigned)ring + 100u * (unsigned)codes);
      liz_rx *rx = liz_rx_new(NULL);
      uint8_t *in = malloc((size_t)g.frame_blocks * LIZ_BLOCK), *frame = malloc((size_t)g.width * g.height);
      uint8_t *verified = malloc((size_t)liz_decoder_max_blocks(d) * 2 * LIZ_BLOCK);
      uint8_t *canvas = malloc((size_t)(g.height + g.height / 4) * 2 * (g.height + g.height / 4));
      int held = 0;
      for (int i = 0; i < 4; i++) {
        liz_tx_next(tx, g.frame_blocks, in);
        CHECK(liz_encoder_paint(enc, in, (uint32_t)i, frame, g.width, LIZ_GREY8) == LIZ_OK, "paint");
        const int got = read_frame(d, frame, &g, codes, &held, verified, canvas);
        liz_verdict v;
        liz_rx_frame(rx, verified, got, &v);
        frames++; blocks += got; missed += g.frame_blocks - got; judged_bad += v.bad;
        CHECK(got == g.frame_blocks, "ring %d, %d codes, frame %d: %d of %d blocks", ring, codes, i, got, g.frame_blocks);
        CHECK(v.test && !v.bad && v.judged == got, "ring %d, %d codes, frame %d: judged the test stream's", ring, codes, i);
        CHECK(held == 26, "ring %d: the held word %d", ring, held);
        if (i == 3 && got) {   /* ai: one payload byte wrong: that block is bad */
          verified[LIZ_ID_BYTES + 100] ^= 1;
          liz_rx_frame(rx, verified, got, &v);
          CHECK(v.test && v.bad == 1, "a wrong byte judged bad (%d bad)", v.bad);
        }
      }
      free(in); free(frame); free(verified); free(canvas);
      liz_rx_free(rx); liz_tx_free(tx); liz_encoder_free(enc);
    }
  printf("test stream: %d frames in every ring, one code and two, %d blocks read, %d missed, %d bad\n", frames, blocks, missed, judged_bad);
}

static void file(liz_decoder *d, int codes, const char *dir, size_t bytes) {
  uint8_t *data = malloc(bytes);
  for (size_t i = 0; i < bytes; i++) data[i] = next_byte();
  liz_tx *tx = liz_tx_new(data, bytes, "roundtrip.bin", "application/octet-stream", 0);
  CHECK(tx, "a sender for %zu bytes", bytes);
  if (!tx) { free(data); return; }
  liz_tx_info info;
  liz_tx_info_get(tx, &info);
  liz_format f = {60, LIZ_RING_DEFAULT, 60, codes};
  liz_geometry g;
  liz_geometry_of(&f, &g);
  liz_encoder *enc = liz_encoder_new(&f);
  liz_store store = {dir, 0};
  liz_rx *rx = liz_rx_new(&store);
  CHECK(rx, "a receiver (%s)", dir ? dir : "memory");
  uint8_t *in = malloc((size_t)g.frame_blocks * LIZ_BLOCK), *frame = malloc((size_t)g.width * g.height);
  uint8_t *verified = malloc((size_t)liz_decoder_max_blocks(d) * 2 * LIZ_BLOCK);
  uint8_t *canvas = malloc((size_t)(g.height + g.height / 4) * 2 * (g.height + g.height / 4));
  int held = 0, frames = 0, lost = 0;
  liz_progress p = {0};
  /* ai: at most three laps of the schedule's frames */
  const int cap = (int)(3 * (info.lap + 64) / (uint32_t)g.frame_blocks) + 16;
  for (; frames < cap && rx && !p.done; frames++) {
    liz_tx_next(tx, g.frame_blocks, in);
    liz_encoder_paint(enc, in, (uint32_t)frames, frame, g.width, LIZ_GREY8);
    const int got = read_frame(d, frame, &g, codes, &held, verified, canvas);
    lost += g.frame_blocks - got;
    liz_verdict v;
    liz_rx_frame(rx, verified, got, &v);
    CHECK(!v.test && !v.bad, "frame %d judged a file's", frames);
    liz_rx_progress(rx, &p);
  }
  char want[65], got_root[65];
  hex(info.root, 32, want);
  CHECK(p.done, "%s: the file whole after %d frames (%.0f%%)", dir ? "directory" : "memory", frames, 100 * p.fraction);
  if (p.done && rx) {
    snprintf(got_root, sizeof got_root, "%s", liz_rx_meta(rx, LIZ_META_ROOT));
    CHECK(!strcmp(want, got_root), "root %s, the sender's %s", got_root, want);
    CHECK(!strcmp(liz_rx_meta(rx, LIZ_META_NAME), "roundtrip.bin"), "the name");
    if (!dir) {
      const uint8_t *out; size_t len;
      CHECK(liz_rx_data(rx, &out, &len) == LIZ_OK && len == bytes && !memcmp(out, data, bytes), "the bytes in memory");
    } else {
      const char *path = liz_rx_meta(rx, LIZ_META_PATH);
      FILE *fp = fopen(path, "rb");
      uint8_t *back = malloc(bytes + 1);
      const size_t n = fp ? fread(back, 1, bytes + 1, fp) : 0;
      if (fp) fclose(fp);
      CHECK(n == bytes && !memcmp(back, data, bytes), "the bytes at %s", path);
      free(back);
    }
  }
  printf("file: %zu B, %u chunks, %d code%s, %s store: %s after %d frames (%d blocks missed), root %s\n", bytes, info.chunks,
         codes, codes > 1 ? "s" : "", dir ? "directory" : "memory", p.done ? "whole" : "NOT whole", frames, lost, want);
  free(in); free(frame); free(verified); free(canvas); free(data);
  liz_rx_free(rx); liz_encoder_free(enc); liz_tx_free(tx);
}

/* ai: dir and the folders above it, made where missing (mkdir -p; a failure shows as the frames' writes failing) */
static void make_dirs(const char *dir) {
  char p[1024];
  snprintf(p, sizeof p, "%s", dir);
  for (char *c = p; *c; c++)
    if (c > p && (*c == '/' || *c == '\\')) { const char k = *c; *c = 0; make_dir(p); *c = k; }
  if (*p) make_dir(p);
}

static int dump(const char *dir) {
  make_dirs(dir);
  for (int codes = 1; codes <= 2; codes++)
    for (int ring = 0; ring < LIZ_RINGS; ring++) {
      liz_format f = {26, ring, 60, codes};
      liz_geometry g;
      liz_geometry_of(&f, &g);
      liz_encoder *enc = liz_encoder_new(&f);
      liz_tx *tx = liz_tx_new_test(1);
      uint8_t *in = malloc((size_t)g.frame_blocks * LIZ_BLOCK), *frame = malloc((size_t)g.width * g.height);
      liz_tx_next(tx, g.frame_blocks, in);
      liz_encoder_paint(enc, in, 0, frame, g.width, LIZ_GREY8);
      char path[1024];
      snprintf(path, sizeof path, "%s/paint-26-%d-%d.grey", dir, ring, codes);
      FILE *fp = fopen(path, "wb");
      CHECK(fp && fwrite(frame, 1, (size_t)g.width * g.height, fp) == (size_t)g.width * g.height, "%s written", path);
      if (fp) fclose(fp);
      free(in); free(frame);
      liz_tx_free(tx); liz_encoder_free(enc);
    }
  if (faults) printf("%d FAILED\n", faults);
  else printf("dumped 8 frames into %s\n", dir);
  return faults ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc > 2 && !strcmp(argv[1], "--dump")) return dump(argv[2]);
  printf("lizard %s, ABI %d, vector paths %d\n", liz_version(), liz_abi(), liz_simd());
  CHECK(liz_abi() == LIZ_ABI, "the ABI the header names");
  formats();
  if (argc > 1 && argv[1][0]) picker(argv[1]);
  liz_decoder *d = liz_decoder_new(0);
  CHECK(d, "a decoder");
  if (!d) return 1;
  test_stream(d);
  const char *mb = getenv("LIZ_TEST_MB");
  const size_t bytes = (size_t)((mb ? atof(mb) : 9.0) * 1000000);
  file(d, 2, NULL, bytes);
  if (argc > 2) file(d, 1, argv[2], bytes / 3);
  liz_decoder_free(d);
  printf(faults ? "%d FAILED\n" : "all passed\n", faults);
  return faults ? 1 : 0;
}
