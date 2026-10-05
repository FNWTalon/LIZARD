// The transfer's blocks and its BLAKE3 tree (xfer.h has the layouts).
#include "xfer.h"
#include "blake3.h"
#include "blake3_impl.h"   // the official compression function, which the chunk-aligned subtree hash is built on
#include <string.h>

static void put_le(uint8_t *p, uint64_t v, int n) { for (int i = 0; i < n; i++) p[i] = (uint8_t)(v >> 8 * i); }
static uint64_t get_le(const uint8_t *p, int n) { uint64_t v = 0; for (int i = n - 1; i >= 0; i--) v = v << 8 | p[i]; return v; }
static int zero(const uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) if (p[i]) return 0; return 1; }

uint32_t xfer_chunk_count(uint64_t length, int chunk_log2) {
  if (chunk_log2 < XFER_LOG2_MIN || chunk_log2 > XFER_LOG2_MAX) return 0;
  const uint64_t n = (length >> chunk_log2) + ((length & ((1ull << chunk_log2) - 1)) != 0);
  return n > XFER_MAX_CHUNKS ? 0 : (uint32_t)n;
}

size_t xfer_chunk_len(const xfer_header_t *h, uint32_t index) {
  if (index >= h->chunks) return 0;
  const uint64_t size = 1ull << h->chunk_log2, at = (uint64_t)index << h->chunk_log2;
  return (size_t)(h->length - at < size ? h->length - at : size);
}

uint32_t xfer_blocks(uint32_t sent) { return (sent + XFER_PAYLOAD - 1) / XFER_PAYLOAD; }

int xfer_sent_ok(const xfer_header_t *h, uint32_t index, uint32_t sent, int seed) {
  const size_t len = xfer_chunk_len(h, index);
  if (!len || !sent || sent > len) return XFER_ERR;
  if (sent < len && h->codec == XFER_CODEC_NONE) return XFER_ERR;
  if (seed < 0 || seed > 255 || (seed && xfer_blocks(sent) < 2)) return XFER_ERR;
  return 0;
}

int xfer_manifest_blocks(uint32_t chunks) {
  return chunks < 2 || chunks > XFER_MAX_CHUNKS ? 0 : (int)((chunks + XFER_PER_BLOCK - 1) / XFER_PER_BLOCK);
}

int xfer_header_init(xfer_header_t *h, uint64_t length, int chunk_log2) {
  memset(h, 0, sizeof *h);
  const uint32_t chunks = xfer_chunk_count(length, chunk_log2);
  if (chunk_log2 < XFER_LOG2_MIN || chunk_log2 > XFER_LOG2_MAX || (length && !chunks)) return -1;
  h->version = XFER_VERSION; h->hash = XFER_HASH_BLAKE3; h->chunk_log2 = (uint8_t)chunk_log2; h->codec = XFER_CODEC_NONE;
  h->length = length; h->chunks = chunks;
  return 0;
}

// What write and parse both hold a header to, the one chunk's sent and seed included: a sent or a seed no chunk could
// have (xfer_sent_ok), or either for a file that is not one chunk, says the block is not what it claims.
static int consistent(const xfer_header_t *h) {
  if (h->version != XFER_VERSION) return XFER_ERR_VERSION;
  if (h->hash != XFER_HASH_BLAKE3) return XFER_ERR_HASH;
  if (h->codec != XFER_CODEC_NONE && h->codec != XFER_CODEC_ZSTD) return XFER_ERR_CODEC;
  if (h->chunk_log2 < XFER_LOG2_MIN || h->chunk_log2 > XFER_LOG2_MAX) return XFER_ERR;
  if (h->chunks != xfer_chunk_count(h->length, h->chunk_log2) || (h->length && !h->chunks)) return XFER_ERR;
  if (h->type_len > XFER_TYPE_MAX || !zero(h->name + h->name_len, XFER_NAME_MAX - h->name_len) || !zero(h->type + h->type_len, XFER_TYPE_MAX - h->type_len)) return XFER_ERR;
  if (h->chunks == 1) { if (xfer_sent_ok(h, 0, h->sent, h->seed)) return XFER_ERR; }
  else if (h->sent || h->seed) return XFER_ERR;
  return 0;
}

int xfer_header_write(const xfer_header_t *h, uint8_t *payload) {
  if (consistent(h)) return XFER_ERR;
  memset(payload, 0, XFER_PAYLOAD);
  payload[XFER_H_VERSION] = h->version; payload[XFER_H_HASH] = h->hash; payload[XFER_H_LOG2] = h->chunk_log2;
  payload[XFER_H_CODEC] = h->codec; payload[XFER_H_SEED] = h->seed;
  payload[XFER_H_NAME_LEN] = h->name_len; payload[XFER_H_TYPE_LEN] = h->type_len;
  put_le(payload + XFER_H_LENGTH, h->length, 8);
  put_le(payload + XFER_H_CHUNKS, h->chunks, 4);
  memcpy(payload + XFER_H_ROOT, h->root, XFER_CV);
  memcpy(payload + XFER_H_NAME, h->name, h->name_len);
  memcpy(payload + XFER_H_TYPE, h->type, h->type_len);
  put_le(payload + XFER_H_SENT, h->sent, 4);
  return 0;
}

int xfer_header_parse(const uint8_t *payload, xfer_header_t *h) {
  memset(h, 0, sizeof *h);
  h->version = payload[XFER_H_VERSION];
  if (h->version != XFER_VERSION) return XFER_ERR_VERSION;
  h->hash = payload[XFER_H_HASH]; h->chunk_log2 = payload[XFER_H_LOG2];
  h->codec = payload[XFER_H_CODEC]; h->seed = payload[XFER_H_SEED];
  h->name_len = payload[XFER_H_NAME_LEN]; h->type_len = payload[XFER_H_TYPE_LEN];
  if (payload[7] || h->type_len > XFER_TYPE_MAX) return XFER_ERR;
  h->length = get_le(payload + XFER_H_LENGTH, 8);
  h->chunks = (uint32_t)get_le(payload + XFER_H_CHUNKS, 4);
  memcpy(h->root, payload + XFER_H_ROOT, XFER_CV);
  memcpy(h->name, payload + XFER_H_NAME, XFER_NAME_MAX);
  memcpy(h->type, payload + XFER_H_TYPE, XFER_TYPE_MAX);
  h->sent = (uint32_t)get_le(payload + XFER_H_SENT, 4);
  return consistent(h);
}

// The entries of manifest block m: chunks 12 m .. 12 m + n - 1.
static uint32_t entries(uint32_t chunks, uint32_t m) {
  const uint32_t first = m * XFER_PER_BLOCK;
  return chunks - first < XFER_PER_BLOCK ? chunks - first : XFER_PER_BLOCK;
}

int xfer_manifest_write(const uint8_t *cvs, const uint32_t *sent, const uint8_t *seeds, uint32_t chunks, const uint8_t root[XFER_CV],
                        uint32_t m, uint8_t *payload) {
  if (m >= (uint32_t)xfer_manifest_blocks(chunks)) return -1;
  const uint32_t first = m * XFER_PER_BLOCK, n = entries(chunks, m);
  memset(payload, 0, XFER_PAYLOAD);
  memcpy(payload + XFER_M_TAG, root, XFER_M_TAG_BYTES);
  for (uint32_t i = 0; i < n; i++) {
    uint8_t *e = payload + XFER_M_ENTRIES + (size_t)i * XFER_ENTRY;
    memcpy(e, cvs + (size_t)(first + i) * XFER_CV, XFER_CV);
    put_le(e + XFER_CV, sent[first + i], XFER_SENT_BYTES);
    e[XFER_CV + XFER_SENT_BYTES] = seeds[first + i];
  }
  return 0;
}

int xfer_manifest_parse(const uint8_t *payload, const xfer_header_t *h, uint32_t m, uint8_t *cvs, uint32_t *sent, uint8_t *seeds) {
  const uint32_t chunks = h->chunks;
  if (m >= (uint32_t)xfer_manifest_blocks(chunks) || memcmp(payload + XFER_M_TAG, h->root, XFER_M_TAG_BYTES)) return -1;
  const uint32_t first = m * XFER_PER_BLOCK, n = entries(chunks, m);
  const size_t used = XFER_M_ENTRIES + (size_t)n * XFER_ENTRY;
  if (!zero(payload + used, XFER_PAYLOAD - used)) return -1;
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *e = payload + XFER_M_ENTRIES + (size_t)i * XFER_ENTRY;
    if (xfer_sent_ok(h, first + i, (uint32_t)get_le(e + XFER_CV, XFER_SENT_BYTES), e[XFER_CV + XFER_SENT_BYTES])) return -1;
  }
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *e = payload + XFER_M_ENTRIES + (size_t)i * XFER_ENTRY;
    memcpy(cvs + (size_t)(first + i) * XFER_CV, e, XFER_CV);
    sent[first + i] = (uint32_t)get_le(e + XFER_CV, XFER_SENT_BYTES);
    seeds[first + i] = e[XFER_CV + XFER_SENT_BYTES];
  }
  return 0;
}

// ---- BLAKE3 ----

static void cv_bytes(uint8_t out[XFER_CV], const uint32_t cv[8]) { for (int i = 0; i < 8; i++) put_le(out + 4 * i, cv[i], 4); }

static void parent(const uint8_t block[BLAKE3_BLOCK_LEN], int root, uint8_t out[XFER_CV]) {
  uint32_t cv[8];
  memcpy(cv, IV, sizeof cv);
  blake3_compress_in_place(cv, block, BLAKE3_BLOCK_LEN, 0, (uint8_t)(PARENT | (root ? ROOT : 0)));
  cv_bytes(out, cv);
}

// One 1 KiB chunk, or the shorter last one. As the root it is the whole input, so its counter is 0, which is also the
// output block counter the official root output compresses with.
static void chunk(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  uint32_t cv[8];
  memcpy(cv, IV, sizeof cv);
  const size_t blocks = len ? (len + BLAKE3_BLOCK_LEN - 1) / BLAKE3_BLOCK_LEN : 1;
  for (size_t b = 0; b < blocks; b++) {
    const size_t at = b * BLAKE3_BLOCK_LEN, n = len - at < BLAKE3_BLOCK_LEN ? len - at : BLAKE3_BLOCK_LEN;
    uint8_t pad[BLAKE3_BLOCK_LEN] = { 0 };
    if (n < BLAKE3_BLOCK_LEN && n) memcpy(pad, in + at, n);
    const uint8_t flags = (uint8_t)((b == 0 ? CHUNK_START : 0) | (b + 1 == blocks ? CHUNK_END | (root ? ROOT : 0) : 0));
    blake3_compress_in_place(cv, n == BLAKE3_BLOCK_LEN ? in + at : pad, (uint8_t)n, counter, flags);
  }
  cv_bytes(out, cv);
}

static void subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  if (len <= BLAKE3_CHUNK_LEN) { chunk(in, len, counter, root, out); return; }
  const size_t left = (size_t)round_down_to_power_of_2((len - 1) / BLAKE3_CHUNK_LEN) * BLAKE3_CHUNK_LEN;
  uint8_t block[BLAKE3_BLOCK_LEN];
  subtree(in, left, counter, 0, block);
  subtree(in + left, len - left, counter + left / BLAKE3_CHUNK_LEN, 0, block + XFER_CV);
  parent(block, root, out);
}

void xfer_b3_subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]) {
  subtree(in, len, root ? 0 : counter, root, out);
}

void xfer_b3_hash(const uint8_t *in, size_t len, uint8_t out[XFER_CV]) {
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, in, len);
  blake3_hasher_finalize(&h, out, XFER_CV);
}

static void merge(const uint8_t *cvs, uint32_t n, int root, uint8_t out[XFER_CV]) {
  if (n == 1) { memcpy(out, cvs, XFER_CV); return; }
  const uint32_t p = (uint32_t)round_down_to_power_of_2(n - 1);
  uint8_t block[BLAKE3_BLOCK_LEN];
  merge(cvs, p, 0, block);
  merge(cvs + (size_t)p * XFER_CV, n - p, 0, block + XFER_CV);
  parent(block, root, out);
}

int xfer_root(const uint8_t *cvs, uint32_t chunks, uint8_t root[XFER_CV]) {
  if (chunks < 2 || chunks > XFER_MAX_CHUNKS) return -1;
  merge(cvs, chunks, 1, root);
  return 0;
}

int xfer_manifest_check(const uint8_t *cvs, uint32_t chunks, const uint8_t root[XFER_CV]) {
  uint8_t got[XFER_CV];
  return xfer_root(cvs, chunks, got) || memcmp(got, root, XFER_CV) ? -1 : 0;
}

int xfer_chunk_cv(const uint8_t *data, size_t len, uint32_t index, int chunk_log2, uint8_t cv[XFER_CV]) {
  if (chunk_log2 < XFER_LOG2_MIN || chunk_log2 > XFER_LOG2_MAX || !len || len > (size_t)1 << chunk_log2 || index >= XFER_MAX_CHUNKS) return -1;
  subtree(data, len, (uint64_t)index << (chunk_log2 - 10), 0, cv);
  return 0;
}

int xfer_chunk_check(const xfer_header_t *h, uint32_t index, const uint8_t *data, size_t len, const uint8_t *cvs) {
  uint8_t got[XFER_CV];
  if (index >= h->chunks || len != xfer_chunk_len(h, index)) return -1;
  if (h->chunks == 1) { xfer_b3_hash(data, len, got); return memcmp(got, h->root, XFER_CV) ? -1 : 0; }
  if (!cvs || xfer_chunk_cv(data, len, index, h->chunk_log2, got)) return -1;
  return memcmp(got, cvs + (size_t)index * XFER_CV, XFER_CV) ? -1 : 0;
}
