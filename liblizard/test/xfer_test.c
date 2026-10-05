// The transfer's C side (src/xfer.h): BLAKE3 and the chunk-aligned subtree hash against the official test vectors,
// the block id, and files of 3 chunks and of 1 written as header and manifest blocks, read back and verified chunk by
// chunk, with a corrupted chunk, a corrupted manifest and a manifest entry no chunk could have caught. The chunks'
// bytes as sent and seed attempts are made up here (no zstd and no Wirehair in this test: the layouts carry them, the
// wasm test exercises the codecs). From liblizard/, about 3 s (one line):
//   gcc -O2 -g -Wall -Wextra -fsanitize=address,undefined -Isrc -Ivendor/blake3 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41
//   -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512 -o build/xfer_test test/xfer_test.c src/xfer.c vendor/blake3/blake3.c
//   vendor/blake3/blake3_dispatch.c vendor/blake3/blake3_portable.c && build/xfer_test
#include "xfer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks, fails;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rng = 2463534242u;
static void fill(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; p[i] = (uint8_t)rng; } }
static int unhex(const char *s, uint8_t *out, int n) {
  for (int i = 0; i < n; i++) { unsigned v; if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1; out[i] = (uint8_t)v; }
  return 0;
}

// The chaining values of a file's chunks, and its root through them (or the chunk's own hash for one chunk).
static uint8_t *chunk_cvs(const uint8_t *file, uint64_t len, int log2, uint32_t *chunks) {
  *chunks = xfer_chunk_count(len, log2);
  uint8_t *cvs = calloc(*chunks ? *chunks : 1, XFER_CV);
  for (uint32_t i = 0; i < *chunks; i++) {
    const uint64_t at = (uint64_t)i << log2, n = len - at < (1ull << log2) ? len - at : 1ull << log2;
    if (xfer_chunk_cv(file + at, (size_t)n, i, log2, cvs + (size_t)i * XFER_CV)) { free(cvs); return NULL; }
  }
  return cvs;
}
static int root_via_chunks(const uint8_t *file, uint64_t len, int log2, uint8_t root[XFER_CV]) {
  uint32_t chunks;
  uint8_t *cvs = chunk_cvs(file, len, log2, &chunks);
  if (!cvs) return -1;
  int rc = 0;
  if (chunks >= 2) rc = xfer_root(cvs, chunks, root);
  else xfer_b3_subtree(file, (size_t)len, 0, 1, root);
  free(cvs);
  return rc;
}

static void vectors(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { CHECK(0, "cannot open %s", path); return; }
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *js = malloc((size_t)n + 1);
  js[fread(js, 1, (size_t)n, f)] = 0;
  fclose(f);
  int cases = 0, splits = 0;
  for (const char *p = strstr(js, "\"input_len\""); p; p = strstr(p + 1, "\"input_len\"")) {
    const int len = atoi(strchr(p, ':') + 1);
    const char *h = strstr(p, "\"hash\"");
    uint8_t want[XFER_CV], got[XFER_CV];
    if (!h || unhex(strchr(h + 6, '"') + 1, want, XFER_CV)) { CHECK(0, "vector %d unreadable", len); continue; }
    uint8_t *in = malloc((size_t)len + 1);
    for (int i = 0; i < len; i++) in[i] = (uint8_t)(i % 251);
    xfer_b3_hash(in, (size_t)len, got);
    CHECK(!memcmp(got, want, XFER_CV), "official hasher, input_len %d", len);
    xfer_b3_subtree(in, (size_t)len, 0, 1, got);
    CHECK(!memcmp(got, want, XFER_CV), "subtree as root, input_len %d", len);
    // Every chunk size that splits it: the chunks' chaining values must combine to the vector's hash.
    for (int log2 = XFER_LOG2_MIN; log2 <= 17 && len; log2++) {
      if (root_via_chunks(in, (uint64_t)len, log2, got)) { CHECK(0, "chunks of 2^%d, input_len %d", log2, len); continue; }
      CHECK(!memcmp(got, want, XFER_CV), "chunks of 2^%d, input_len %d", log2, len);
      splits += xfer_chunk_count((uint64_t)len, log2) >= 2;
    }
    free(in);
    cases++;
  }
  free(js);
  printf("official vectors: %d cases, each as one buffer, as the root subtree, and at chunk sizes 2^10 to 2^17 (%d splits into 2 or more chunks)\n", cases, splits);
  CHECK(cases == 35, "expected the 35 cases of test_vectors.json, read %d", cases);
}

// Lengths either side of the chunk boundaries at several chunk sizes, against the official hasher on the whole file.
static void boundaries(void) {
  static const int logs[] = { 10, 11, 13, 16 };
  int n = 0;
  for (int li = 0; li < 4; li++) {
    const uint64_t c = 1ull << logs[li];
    for (uint64_t k = 1; k <= 20; k += (k < 5 ? 1 : 5)) {
      const uint64_t lens[] = { k * c - 1, k * c, k * c + 1, k * c + 469, k * c + c / 2 };
      for (int j = 0; j < 5; j++) {
        const uint64_t len = lens[j];
        uint8_t *file = malloc((size_t)len), want[XFER_CV], got[XFER_CV];
        fill(file, (size_t)len);
        xfer_b3_hash(file, (size_t)len, want);
        CHECK(!root_via_chunks(file, len, logs[li], got) && !memcmp(got, want, XFER_CV), "length %llu at 2^%d", (unsigned long long)len, logs[li]);
        free(file);
        n++;
      }
    }
  }
  printf("chunk boundaries: %d files at 4 chunk sizes, each root through its chunks against the official hasher\n", n);
}

static void ids(void) {
  CHECK(xfer_id(0, 5) == 5, "chunk 0 keeps today's ids");
  CHECK(xfer_id(1, 0) == 1u << 18, "chunk 1 starts at 2^18");
  CHECK(xfer_id(16382, (1u << 18) - 1) == 0xfffbffffu, "last data id");
  CHECK(XFER_ID_HEADER == 0xfffc0000u && XFER_ID_MANIFEST == 0xfffc0001u, "control ids");
  CHECK(xfer_id_kind(0xfffbffffu) == XFER_KIND_DATA && xfer_id_kind(XFER_ID_HEADER) == XFER_KIND_HEADER, "kinds");
  CHECK(xfer_id_kind(XFER_ID_MANIFEST + 1365) == XFER_KIND_MANIFEST && xfer_id_kind(XFER_ID_MANIFEST + 1366) == XFER_KIND_RESERVED &&
        xfer_id_kind(0xffffffffu) == XFER_KIND_RESERVED, "manifest and reserved kinds");
  CHECK(xfer_id_chunk(xfer_id(123, 4567)) == 123 && xfer_id_symbol(xfer_id(123, 4567)) == 4567, "fields");
  uint8_t b[4];
  xfer_id_put(b, 0x12345678u);
  CHECK(b[0] == 0x78 && b[3] == 0x12 && xfer_id_get(b) == 0x12345678u, "little-endian");
  CHECK(xfer_chunk_count(16383ull << 24, 24) == 16383 && !xfer_chunk_count((16383ull << 24) + 1, 24), "the largest file");
  CHECK(!xfer_chunk_count(1, 9) && !xfer_chunk_count(1, 25), "chunk size range");
  xfer_header_t h;
  CHECK(xfer_header_init(&h, (16383ull << 24) + 1, 24) && !xfer_header_init(&h, 0, 10) && h.chunks == 0, "init limits");
  CHECK(xfer_manifest_blocks(0) == 0 && xfer_manifest_blocks(1) == 0 && xfer_manifest_blocks(2) == 1 && xfer_manifest_blocks(12) == 1 &&
        xfer_manifest_blocks(13) == 2 && xfer_manifest_blocks(16383) == 1366, "manifest block counts");
  CHECK(xfer_blocks(0) == 0 && xfer_blocks(1) == 1 && xfer_blocks(469) == 1 && xfer_blocks(470) == 2, "blocks of the bytes sent");
}

// A file through the whole C side: chaining values, root, header and manifest blocks written as 473-byte blocks and
// read back into a receiver's state, every chunk verified, then corruption. codec: XFER_CODEC_ZSTD makes up each chunk's
// bytes as sent (three quarters of its own past 1,000 bytes, as a frame would be) and a seed attempt (its index mod 3
// where it is fountained); XFER_CODEC_NONE sends every chunk as it is. Returns the file's manifest block count.
static int transfer(const char *label, uint64_t len, int log2, int codec, const char *name, const char *type) {
  uint8_t *file = malloc(len ? (size_t)len : 1), b3sum[XFER_CV];
  fill(file, (size_t)len);
  xfer_b3_hash(file, (size_t)len, b3sum);

  // Sender.
  xfer_header_t tx;
  CHECK(!xfer_header_init(&tx, len, log2), "%s: init", label);
  uint32_t chunks;
  uint8_t *cvs = chunk_cvs(file, len, log2, &chunks);
  CHECK(cvs && chunks == tx.chunks, "%s: chunk count", label);
  if (chunks >= 2) CHECK(!xfer_root(cvs, chunks, tx.root), "%s: root", label);
  else memcpy(tx.root, b3sum, XFER_CV);
  CHECK(!memcmp(tx.root, b3sum, XFER_CV), "%s: the root through the chunks is the file's b3sum", label);
  uint32_t *sent = calloc(chunks ? chunks : 1, sizeof *sent);
  uint8_t *seeds = calloc(chunks ? chunks : 1, 1);
  tx.codec = (uint8_t)codec;
  for (uint32_t i = 0; i < chunks; i++) {
    const size_t n = xfer_chunk_len(&tx, i);
    sent[i] = codec == XFER_CODEC_ZSTD && n > 1000 ? (uint32_t)(n * 3 / 4) : (uint32_t)n;
    seeds[i] = xfer_blocks(sent[i]) >= 2 ? (uint8_t)(i % 3) : 0;
    CHECK(!xfer_sent_ok(&tx, i, sent[i], seeds[i]), "%s: chunk %u sent %u with seed %d refused", label, i, sent[i], seeds[i]);
    xfer_header_t none = tx;
    none.codec = XFER_CODEC_NONE;
    CHECK(!xfer_sent_ok(&none, i, sent[i], seeds[i]) == (sent[i] == n), "%s: chunk %u sent shorter under codec none", label, i);
    CHECK(xfer_sent_ok(&tx, i, (uint32_t)n + 1, 0) && xfer_sent_ok(&tx, i, 0, 0), "%s: chunk %u sent past its bytes, or as none, passed", label, i);
  }
  if (chunks == 1) { tx.sent = sent[0]; tx.seed = seeds[0]; }
  tx.name_len = (uint8_t)strlen(name); memcpy(tx.name, name, tx.name_len);
  tx.type_len = (uint8_t)strlen(type); memcpy(tx.type, type, tx.type_len);
  const int mblocks = xfer_manifest_blocks(chunks);
  uint8_t hblock[XFER_BLOCK], *mf = calloc((size_t)(mblocks ? mblocks : 1), XFER_BLOCK);
  xfer_id_put(hblock, XFER_ID_HEADER);
  CHECK(!xfer_header_write(&tx, hblock + XFER_ID_BYTES), "%s: header write", label);
  for (int m = 0; m < mblocks; m++) {
    xfer_id_put(mf + (size_t)m * XFER_BLOCK, XFER_ID_MANIFEST + (uint32_t)m);
    CHECK(!xfer_manifest_write(cvs, sent, seeds, chunks, tx.root, (uint32_t)m, mf + (size_t)m * XFER_BLOCK + XFER_ID_BYTES), "%s: manifest %d write", label, m);
  }

  // Receiver: the blocks by id, in reverse order to show nothing hangs on order.
  xfer_header_t rx;
  CHECK(xfer_id_kind(xfer_id_get(hblock)) == XFER_KIND_HEADER, "%s: header id", label);
  CHECK(!xfer_header_parse(hblock + XFER_ID_BYTES, &rx), "%s: header parse", label);
  CHECK(!memcmp(&rx, &tx, sizeof rx) || (rx.version == tx.version && rx.hash == tx.hash && rx.chunk_log2 == tx.chunk_log2 && rx.length == tx.length &&
        rx.chunks == tx.chunks && rx.codec == tx.codec && rx.seed == tx.seed && rx.sent == tx.sent && !memcmp(rx.root, tx.root, XFER_CV) &&
        rx.name_len == tx.name_len && !memcmp(rx.name, tx.name, XFER_NAME_MAX) && rx.type_len == tx.type_len && !memcmp(rx.type, tx.type, XFER_TYPE_MAX)),
        "%s: header read back differs", label);
  uint8_t *got = calloc(rx.chunks ? rx.chunks : 1, XFER_CV);
  uint32_t *gsent = calloc(rx.chunks ? rx.chunks : 1, sizeof *gsent);
  uint8_t *gseeds = calloc(rx.chunks ? rx.chunks : 1, 1);
  if (rx.chunks == 1) { gsent[0] = rx.sent; gseeds[0] = rx.seed; }
  for (int m = mblocks - 1; m >= 0; m--) {
    const uint8_t *blk = mf + (size_t)m * XFER_BLOCK;
    const uint32_t id = xfer_id_get(blk);
    CHECK(xfer_id_kind(id) == XFER_KIND_MANIFEST && id - XFER_ID_MANIFEST == (uint32_t)m, "%s: manifest id", label);
    CHECK(!xfer_manifest_parse(blk + XFER_ID_BYTES, &rx, id - XFER_ID_MANIFEST, got, gsent, gseeds), "%s: manifest %d parse", label, m);
  }
  if (rx.chunks >= 2) {
    CHECK(!memcmp(got, cvs, (size_t)chunks * XFER_CV), "%s: list read back differs", label);
    CHECK(!xfer_manifest_check(got, rx.chunks, rx.root), "%s: list against the root", label);
  }
  for (uint32_t i = 0; i < rx.chunks; i++) {
    const size_t n = xfer_chunk_len(&rx, i);
    CHECK(!xfer_chunk_check(&rx, i, file + ((uint64_t)i << log2), n, got), "%s: chunk %u", label, i);
    CHECK(gsent[i] == sent[i] && gseeds[i] == seeds[i], "%s: chunk %u sent %u and seed %d read back as %u and %d", label, i, sent[i], seeds[i], gsent[i], gseeds[i]);
    CHECK((gsent[i] < n) == (codec == XFER_CODEC_ZSTD && n > 1000), "%s: chunk %u compressed where it should be", label, i);
  }

  // A corrupted chunk: one bit of the middle one, caught there and nowhere else.
  if (rx.chunks) {
    const uint32_t bad = rx.chunks / 2;
    uint8_t *p = file + ((uint64_t)bad << log2) + xfer_chunk_len(&rx, bad) / 2;
    *p ^= 0x10;
    for (uint32_t i = 0; i < rx.chunks; i++)
      CHECK(!xfer_chunk_check(&rx, i, file + ((uint64_t)i << log2), xfer_chunk_len(&rx, i), got) == (i != bad), "%s: corrupt chunk %u, checking %u", label, bad, i);
    *p ^= 0x10;
    CHECK(xfer_chunk_check(&rx, bad, file + ((uint64_t)bad << log2), xfer_chunk_len(&rx, bad) - 1, got), "%s: a short chunk passed", label);
  }
  // A corrupted manifest: a bit of the last chaining value carried. The block parses (it is only data), the list
  // fails against the root, and so does the chunk it names.
  if (mblocks) {
    uint8_t *blk = mf + (size_t)(mblocks - 1) * XFER_BLOCK + XFER_ID_BYTES, keep = blk[XFER_M_ENTRIES];
    const uint32_t hit = (uint32_t)(mblocks - 1) * XFER_PER_BLOCK;
    blk[XFER_M_ENTRIES] ^= 1;
    CHECK(!xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: flipped manifest parse", label);
    CHECK(xfer_manifest_check(got, rx.chunks, rx.root), "%s: a corrupted list passed against the root", label);
    CHECK(xfer_chunk_check(&rx, hit, file + ((uint64_t)hit << log2), xfer_chunk_len(&rx, hit), got), "%s: a chunk passed against a corrupted list", label);
    blk[XFER_M_ENTRIES] = keep;
    CHECK(!xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds) && !xfer_manifest_check(got, rx.chunks, rx.root), "%s: repaired", label);
    xfer_header_t other = rx;
    other.root[3] ^= 0x80;
    CHECK(xfer_manifest_parse(blk, &other, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: a block of another transfer parsed", label);
    CHECK(xfer_manifest_parse(blk, &rx, (uint32_t)mblocks, got, gsent, gseeds), "%s: a manifest index past the list parsed", label);
    blk[XFER_PAYLOAD - 1] = 1;
    CHECK(xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: nonzero padding parsed", label);
    blk[XFER_PAYLOAD - 1] = 0;
    // An entry no chunk could have: sent past the chunk's bytes, then a seed on a chunk of one block.
    uint8_t *e = blk + XFER_M_ENTRIES + XFER_CV, save4[XFER_SENT_BYTES + 1];
    memcpy(save4, e, sizeof save4);
    e[0] = e[1] = e[2] = 0xff; e[3] = 0x7f;
    CHECK(xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: a manifest entry past the chunk's bytes parsed", label);
    e[0] = 1; e[1] = e[2] = e[3] = 0; e[4] = 1;
    CHECK(xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: a seed on a chunk of one block parsed", label);
    memcpy(e, save4, sizeof save4);
    CHECK(!xfer_manifest_parse(blk, &rx, (uint32_t)mblocks - 1, got, gsent, gseeds), "%s: repaired again", label);
  }
  // A header that is not one.
  uint8_t *hp = hblock + XFER_ID_BYTES, save[XFER_PAYLOAD];
  memcpy(save, hp, XFER_PAYLOAD);
  hp[XFER_H_VERSION] = 1; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR_VERSION, "%s: version 1", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[XFER_H_HASH] = 2; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR_HASH, "%s: hash 2", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[XFER_H_CODEC] = 2; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR_CODEC, "%s: codec 2", label); memcpy(hp, save, XFER_PAYLOAD);
  if (chunks != 1) { hp[XFER_H_SENT] = 1; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: sent for a file of %u chunks", label, chunks); memcpy(hp, save, XFER_PAYLOAD); }
  else { hp[XFER_H_SENT] = 0; hp[XFER_H_SENT + 1] = 0; hp[XFER_H_SENT + 2] = 0; hp[XFER_H_SENT + 3] = 0; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: one chunk sent as nothing", label); memcpy(hp, save, XFER_PAYLOAD); }
  hp[XFER_H_CHUNKS]++; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: chunk count off", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[XFER_H_LOG2] = 25; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: 32 MiB chunks", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[7] = 1; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: reserved byte", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[XFER_H_NAME + XFER_NAME_MAX - 1] = 'x'; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: name padding", label); memcpy(hp, save, XFER_PAYLOAD);
  hp[XFER_H_TYPE_LEN] = XFER_TYPE_MAX + 1; CHECK(xfer_header_parse(hp, &rx) == XFER_ERR, "%s: type length", label); memcpy(hp, save, XFER_PAYLOAD);
  CHECK(!xfer_header_parse(hp, &rx), "%s: restored", label);

  printf("%-9s %10llu B, chunks of 2^%d: %u chunk%s, %d manifest block%s, root %02x%02x%02x%02x... = b3sum\n", label, (unsigned long long)len, log2,
         chunks, chunks == 1 ? "" : "s", mblocks, mblocks == 1 ? "" : "s", b3sum[0], b3sum[1], b3sum[2], b3sum[3]);
  free(file); free(cvs); free(mf); free(got); free(sent); free(seeds); free(gsent); free(gseeds);
  return mblocks;
}

int main(int argc, char **argv) {
  vectors(argc > 1 ? argv[1] : "vendor/blake3/test_vectors/test_vectors.json");
  boundaries();
  ids();
  // Two transfers: three chunks (the last short) and one, compressed.
  CHECK(transfer("3 chunks", (2u << 20) + 304000, 20, XFER_CODEC_ZSTD, "IMG_2041.jpg", "image/jpeg") == 1, "3 chunks: one manifest block");
  CHECK(transfer("1 chunk", 700001, 20, XFER_CODEC_ZSTD, "notes.txt", "text/plain; charset=utf-8") == 0, "1 chunk: no manifest");
  // Four manifest blocks (12, 12, 12, 5) and a last chunk of one block, which is not fountained, sent as it is; a
  // power of two of chunks with the last one full, compressed; a file of one block; an empty file.
  CHECK(transfer("41 chunks", 40 * 1024 + 100, 10, XFER_CODEC_NONE, "", "") == 4, "41 chunks: four manifest blocks");
  CHECK(transfer("16 chunks", 16u << 16, 16, XFER_CODEC_ZSTD, "a", "application/octet-stream") == 2, "16 chunks: two manifest blocks");
  CHECK(transfer("1 block", 469, 10, XFER_CODEC_ZSTD, "tiny", "") == 0, "one block");
  CHECK(transfer("empty", 0, 10, XFER_CODEC_NONE, "empty", "") == 0, "empty");
  xfer_header_t h;
  uint8_t p[XFER_PAYLOAD];
  xfer_header_init(&h, 469, 10); h.sent = 469; h.seed = 1;
  CHECK(xfer_header_write(&h, p) == XFER_ERR, "a seed for an unfountained chunk was written");
  xfer_header_init(&h, 470, 10); h.sent = 470; h.seed = 1;
  CHECK(!xfer_header_write(&h, p), "a seed for a chunk of two blocks refused");
  h.sent = 300; CHECK(xfer_header_write(&h, p) == XFER_ERR, "a one-chunk file sent shorter under codec none was written");
  h.codec = XFER_CODEC_ZSTD; h.seed = 0; CHECK(!xfer_header_write(&h, p), "a one-chunk file sent shorter under zstd refused");
  xfer_header_init(&h, 40 * 1024 + 100, 10); h.sent = 1;
  CHECK(xfer_header_write(&h, p) == XFER_ERR, "a sent for a file of many chunks was written");
  printf("%d checks, %d failed\n", checks, fails);
  return fails != 0;
}
