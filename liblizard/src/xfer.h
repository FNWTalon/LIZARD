// The transfer in the light: the block id, the header block and the manifest blocks, so a camera alone can recover a
// file. Before this the fountain's header went from the sender page to the receiver page over the rig's network, and
// the block id was a convention of sim/phy.mjs.
//
// A file is cut into chunks of 2^k bytes (k 10 to 24; the last one shorter), each its own Wirehair fountain and each
// verified on arrival against the file's one BLAKE3 root, the way Bao and iroh stream: a chunk of 2^k bytes aligned
// at its own size is a whole subtree of BLAKE3's tree, so its chaining value comes from the chunk alone, and the
// chaining values of all the chunks combine to the root (2026-09-24).
//
// ---- The block ----
// Every LIZARD block is 473 bytes (src/focus.c block_bytes, the same in all 128 formats and at every rate): the id, uint32
// little-endian in bytes 0 to 3, then XFER_PAYLOAD = 469 bytes of payload.
//
//   id bits 31..18  chunk   0 .. 16382 a chunk of the file, 16383 (XFER_CONTROL) a control block
//   id bits 17..0   symbol  a data block: the Wirehair block id within its chunk's fountain
//                           a control block: 0 the header, 1 + m manifest block m (m < 1171), the rest reserved
//
// Why 18 bits of symbol. Wirehair takes at most 64,000 source blocks (liblizard/wirehair/shim.cpp), so the largest chunk
// is 16 MiB (35,773 blocks of 469 bytes; 32 MiB would be 71,546). Repair ids run past the source count without end,
// and a sender that runs out of field wraps and repeats ids the receiver may already hold. 2^18 is 7.3 times the
// source blocks of a 16 MiB chunk and 29 times those of a 4 MiB one: a receiver has to miss 86% of a 16 MiB chunk's
// blocks before the sender repeats one. 16 bits would repeat at 45%.
// Why 14 bits of chunk. 16,383 chunks: 16 GiB at 1 MiB a chunk, 64 GiB at 4 MiB, 256 GiB at 16 MiB. A file of
// several GB at a few MiB a chunk is two thousand or so.
// Why the control range is at the top: chunk 0's ids are then 0 .. 2^18 - 1, the ids today's one-fountain file mode
// sends, so a one-chunk file goes out under the ids it always had.
// The header's id is the same for every transfer and a manifest block's for every transfer with that many chunks, so
// a receiver that drops a block whose id it has seen misses the next file: a control block is new when its bytes are.
//
// ---- Compression (2026-10-05) ----
// A chunk is compressed on its own before it is fountained: one zstd frame (zstd/shim.c pins the parameters) where
// that is shorter than the chunk's own bytes, else the bytes as they are. What a chunk's fountain carries is its
// "sent" bytes, 1 .. its length; sent < length means a frame. Each chunk its own frame, so a receiver decompresses a
// chunk the moment it is recovered with the chunk's memory alone (the frame's window is at most the chunk), and a
// file of pictures and archives costs nothing: every chunk that does not shrink goes as it is. The file's length,
// its root and the chaining values are the file's own bytes, the ones handed over, so b3sum of the received file is
// the root whatever was compressed. The receiver reads sent and the seed attempt of every chunk from the manifest
// (one chunk: from the header), so it builds no chunk's decoder before the manifest is in; the control cycle puts the
// manifest within an eighth of a lap, and under the interleaved schedule no chunk completes sooner anyway.
//
// ---- The header block (id XFER_ID_HEADER = 0xfffc0000), 469 bytes, little-endian ----
//     0    1  version      XFER_VERSION = 2; a receiver refuses any other (-2). 1 until 2026-10-05: seeds at 3 and
//                          4 (one for every chunk but the last, one for the last), type to 162, no codec, no sent
//     1    1  hash         XFER_HASH_BLAKE3 = 1; 0 and 2..255 refused (-3), the room for another hash
//     2    1  chunk_log2   the chunk size is 1 << chunk_log2 bytes, 10 .. 24
//     3    1  codec        XFER_CODEC_NONE = 0: every chunk is sent as its bytes; XFER_CODEC_ZSTD = 1: a chunk sent
//                          shorter than its bytes is one zstd frame of them; 2..255 refused (-4)
//     4    1  seed         Wirehair seed attempt of the one chunk of a one-chunk file of 2+ blocks; 0 otherwise
//     5    1  name_len     bytes of name, 0 .. 255
//     6    1  type_len     bytes of media type, 0 .. 158
//     7    1  reserved, 0
//     8    8  length       the file's bytes, u64
//    16    4  chunks       ceil(length / chunk size), u32, at most 16,383; 0 for an empty file
//    20   32  root         BLAKE3 of the whole file, what b3sum prints
//    52  255  name         the file name, UTF-8, zero padded; empty for none
//   307  158  type         the media type, ASCII, zero padded; empty for none
//   465    4  sent         the one chunk's bytes as sent (its frame, or its own), u32, 1 .. its length; 0 when the
//                          file is not one chunk
// Every byte past a length is zero and the parse insists, so a block that is not a header rarely passes for one.
// Version 2 also fixes the fountain: Wirehair V2, profile WIREHAIR_V2_PROFILE_CERTIFIED_2026_07 (the shim pins it), at
// 469-byte blocks, and the codec, zstd frames. A change to any is a new version.
//
// The seed attempt. Wirehair picks the attempt (0 .. 255) from the block count alone, and with compression every
// chunk has its own count, so every chunk's attempt travels: in the manifest, or for a one-chunk file in the header.
// A chunk of one block (sent 469 bytes or less) is not fountained, since Wirehair refuses a one-block message (-9):
// its sent bytes go zero padded as symbol 0, repeated as the sender cycles, and its seed is 0.
//
// ---- The manifest blocks (id XFER_ID_MANIFEST + m, m = 0 .. xfer_manifest_blocks(chunks) - 1) ----
//     0    8  tag          the root's first 8 bytes, so a block of another transfer is refused before it is used
//     8  444  entries      chunks 12 m .. 12 m + 11, 37 bytes each, in order: the chaining value (32), sent (4,
//                          u32), the seed attempt (1); zero past the last chunk
//   452   17  reserved, 0
// A file of two chunks or more has ceil(chunks / 12) of them; a file of one chunk or none has none, since its root is
// the one chunk's own hash and is checked from the chunk, and its sent and seed are in the header.
//
// ---- The tree ----
// The root of chunks c[0..n), n >= 2: split at p, the largest power of two below n; left the root of c[0..p), right
// that of c[p..n); a parent node is BLAKE3's compression of left || right under the key with the PARENT flag, and the
// top one also has ROOT. A single chaining value is its own subtree. This is BLAKE3's own tree: every chunk but the
// last is 2^k bytes on a 2^k boundary, so BLAKE3's split (the largest power of two in 1 KiB chunks that leaves a byte
// for the right) always falls on a chunk boundary. test/xfer_test.c proves it on the official test vectors.
//
// ---- In the wasm (src/wasm.c) ----
// Exported as declared below: xfer_manifest_blocks, xfer_manifest_write, xfer_manifest_check, xfer_root,
// xfer_chunk_cv, xfer_b3_hash. Wrapped for JS, with flat arguments: xfer_layout, xfer_id_of, xfer_kind_of,
// xfer_hdr_write, xfer_hdr_parse, xfer_mf_parse, xfer_chunk_ok, xfer_b3_sub. test/xfer_wasm.mjs drives every one.
#ifndef XFER_H
#define XFER_H
#include <stddef.h>
#include <stdint.h>

enum {
  XFER_ID_BYTES = 4, XFER_PAYLOAD = 469, XFER_BLOCK = XFER_ID_BYTES + XFER_PAYLOAD,
  XFER_SYMBOL_BITS = 18, XFER_CHUNK_BITS = 14,
  XFER_CONTROL = (1 << XFER_CHUNK_BITS) - 1, XFER_MAX_CHUNKS = XFER_CONTROL,
  XFER_VERSION = 2, XFER_HASH_BLAKE3 = 1, XFER_CODEC_NONE = 0, XFER_CODEC_ZSTD = 1,
  XFER_LOG2_MIN = 10, XFER_LOG2_MAX = 24,
  XFER_CV = 32, XFER_SENT_BYTES = 4, XFER_SEED_BYTES = 1, XFER_ENTRY = XFER_CV + XFER_SENT_BYTES + XFER_SEED_BYTES,
  XFER_PER_BLOCK = 12,
  XFER_MAX_MANIFEST = (XFER_MAX_CHUNKS + XFER_PER_BLOCK - 1) / XFER_PER_BLOCK,
  XFER_NAME_MAX = 255, XFER_TYPE_MAX = 158,
  // header offsets
  XFER_H_VERSION = 0, XFER_H_HASH = 1, XFER_H_LOG2 = 2, XFER_H_CODEC = 3, XFER_H_SEED = 4,
  XFER_H_NAME_LEN = 5, XFER_H_TYPE_LEN = 6, XFER_H_LENGTH = 8, XFER_H_CHUNKS = 16, XFER_H_ROOT = 20,
  XFER_H_NAME = 52, XFER_H_TYPE = XFER_H_NAME + XFER_NAME_MAX, XFER_H_SENT = XFER_H_TYPE + XFER_TYPE_MAX,
  // manifest offsets
  XFER_M_TAG = 0, XFER_M_TAG_BYTES = 8, XFER_M_ENTRIES = 8,
  // what the parses return besides 0
  XFER_ERR = -1, XFER_ERR_VERSION = -2, XFER_ERR_HASH = -3, XFER_ERR_CODEC = -4,
};
#define XFER_ID_HEADER ((uint32_t)XFER_CONTROL << XFER_SYMBOL_BITS)
#define XFER_ID_MANIFEST (XFER_ID_HEADER + 1u)
#define XFER_SYMBOL_MASK ((1u << XFER_SYMBOL_BITS) - 1u)

// XFER_KIND_RESERVED: a control id past the manifest's, which a receiver of this version ignores.
enum { XFER_KIND_RESERVED = -1, XFER_KIND_DATA = 0, XFER_KIND_HEADER = 1, XFER_KIND_MANIFEST = 2 };
static inline uint32_t xfer_id(uint32_t chunk, uint32_t symbol) { return chunk << XFER_SYMBOL_BITS | (symbol & XFER_SYMBOL_MASK); }
static inline uint32_t xfer_id_chunk(uint32_t id) { return id >> XFER_SYMBOL_BITS; }
static inline uint32_t xfer_id_symbol(uint32_t id) { return id & XFER_SYMBOL_MASK; }
static inline int xfer_id_kind(uint32_t id) {
  const uint32_t s = xfer_id_symbol(id);
  return xfer_id_chunk(id) != XFER_CONTROL ? XFER_KIND_DATA : !s ? XFER_KIND_HEADER : s <= XFER_MAX_MANIFEST ? XFER_KIND_MANIFEST : XFER_KIND_RESERVED;
}
static inline uint32_t xfer_id_get(const uint8_t *block) {
  return block[0] | (uint32_t)block[1] << 8 | (uint32_t)block[2] << 16 | (uint32_t)block[3] << 24;
}
static inline void xfer_id_put(uint8_t *block, uint32_t id) { for (int i = 0; i < 4; i++) block[i] = (uint8_t)(id >> 8 * i); }

typedef struct {
  uint8_t version, hash, chunk_log2, codec, seed, name_len, type_len;
  uint64_t length;
  uint32_t chunks, sent;
  uint8_t root[XFER_CV];
  uint8_t name[XFER_NAME_MAX], type[XFER_TYPE_MAX];
} xfer_header_t;

// ceil(length / 2^chunk_log2), or 0 where that is past XFER_MAX_CHUNKS or chunk_log2 is out of range (an empty file
// is 0 chunks too: the caller tells them apart by length).
uint32_t xfer_chunk_count(uint64_t length, int chunk_log2);
// Bytes of chunk `index` under h (the file's own, not as sent), 0 past the last.
size_t xfer_chunk_len(const xfer_header_t *h, uint32_t index);
// The Wirehair source blocks of a chunk sent as `sent` bytes, ceil(sent / 469); 1 means sent unfountained as symbol 0.
uint32_t xfer_blocks(uint32_t sent);
// 0 where `sent` and `seed` can be chunk `index`'s under h: sent 1 .. its length, shorter only under a codec, and a
// seed only for 2 blocks or more. What the header (one chunk) and the manifest entries are held to.
int xfer_sent_ok(const xfer_header_t *h, uint32_t index, uint32_t sent, int seed);
int xfer_manifest_blocks(uint32_t chunks);

// Fills a header for a file of `length` bytes: version, hash, chunk count, codec none; everything else is the
// caller's (the codec, and for a one-chunk file its sent and seed). 0, or -1 for a length or chunk size the layout
// cannot carry.
int xfer_header_init(xfer_header_t *h, uint64_t length, int chunk_log2);
// The 469-byte payload from h, or -1 where h is inconsistent (the same checks the parse makes).
int xfer_header_write(const xfer_header_t *h, uint8_t *payload);
// 0, or XFER_ERR / XFER_ERR_VERSION / XFER_ERR_HASH / XFER_ERR_CODEC.
int xfer_header_parse(const uint8_t *payload, xfer_header_t *h);

// Manifest block m of a file of `chunks` chunks whose root is `root`: the chaining values cvs (chunks * 32 bytes),
// each chunk's bytes as sent and its seed attempt.
int xfer_manifest_write(const uint8_t *cvs, const uint32_t *sent, const uint8_t *seeds, uint32_t chunks, const uint8_t root[XFER_CV],
                        uint32_t m, uint8_t *payload);
// Its entries into cvs, sent and seeds at 12 m: 0, or -1 for a block of another transfer (tag), past the list, with
// an entry no chunk of h could have (xfer_sent_ok), or with nonzero padding. Nothing is written on -1.
int xfer_manifest_parse(const uint8_t *payload, const xfer_header_t *h, uint32_t m, uint8_t *cvs, uint32_t *sent, uint8_t *seeds);
// The root the chaining values of chunks >= 2 chunks combine to (the tree above). -1 for fewer than 2.
int xfer_root(const uint8_t *cvs, uint32_t chunks, uint8_t root[XFER_CV]);
// 0 where the list combines to root.
int xfer_manifest_check(const uint8_t *cvs, uint32_t chunks, const uint8_t root[XFER_CV]);

// The chaining value of chunk `index` of 2^chunk_log2 bytes, len being its bytes (the chunk size, or less for the
// last). Not the root: a file of one chunk is checked with xfer_b3_hash.
int xfer_chunk_cv(const uint8_t *data, size_t len, uint32_t index, int chunk_log2, uint8_t cv[XFER_CV]);
// 0 where chunk `index` is right: len is what h says it is, and its chaining value is cvs[index] (a list checked
// against the root first), or with one chunk its hash is the root (cvs unused, may be NULL).
int xfer_chunk_check(const xfer_header_t *h, uint32_t index, const uint8_t *data, size_t len, const uint8_t *cvs);

// BLAKE3 of a buffer, 32 bytes, through the official hasher (vendor/blake3).
void xfer_b3_hash(const uint8_t *in, size_t len, uint8_t out[XFER_CV]);
// The chaining value of the subtree BLAKE3 would make of `len` bytes starting at 1 KiB chunk `counter` (byte offset
// 1024 counter), or with root = 1 the hash, the range then being the whole input (counter ignored). Right only for a
// range that is a node of the input's tree: 2^j KiB on a 2^j KiB boundary, or all that is left of the input from
// such a boundary when that is less.
void xfer_b3_subtree(const uint8_t *in, size_t len, uint64_t counter, int root, uint8_t out[XFER_CV]);

#endif
