/* ai: liblizard: the Lizard code as a library (2026-10-03): the C (NEON on arm64) and the GPU through Vulkan behind one
 * ai: portable API, with no UI. A luminance-only animated barcode: a frame is a grey picture whose FFT
 * ai: coefficients carry blocks of 473 bytes (a 4-byte little-endian id and 469 of payload, CRC-checked), inside a
 * ai: border that says what is inside; a file goes as a fountain-coded transfer verified by its BLAKE3 root.
 * ai:
 * ai: Layers, each usable alone:
 * ai:   1. format arithmetic: a format's geometry, the room it needs, the largest that fits a room;
 * ai:   2. the per-frame codec: blocks to a painted frame, a captured image to its verified blocks (the C reference,
 * ai:      NEON on arm64, WebAssembly SIMD in a wasm build, scalar elsewhere);
 * ai:   3. the transfer: a file (or the test stream) to each frame's blocks, verified blocks back to the file.
 * ai: Layers 1 to 3 start no thread and keep no state between calls but their handles'. A handle is used from one
 * ai: thread at a time; different handles from different threads at once.
 * ai:
 * ai: The format is never frozen: a sender and a receiver need the same release of this library (liz_version), and
 * ai: LIZ_ABI changes with any change to these declarations.
 * ai:
 * ai: Errors: a function returns LIZ_OK (0) or a count, or a negative LIZ_E_*; one returning a handle returns NULL. In
 * ai: either case liz_last_error() is a message for the calling thread's last failure. Strings the library returns are
 * ai: valid until the next call on the same handle. Buffers are the caller's, sized by the functions that say how big. */
#ifndef LIZARD_H
#define LIZARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(LIZ_BUILDING)
#define LIZ_API __declspec(dllexport)
#elif defined(LIZ_STATIC)
#define LIZ_API
#else
#define LIZ_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define LIZ_API __attribute__((visibility("default")))
#else
#define LIZ_API
#endif

#define LIZ_ABI 1

LIZ_API int liz_abi(void);
LIZ_API const char *liz_version(void);
LIZ_API const char *liz_last_error(void);
/* ai: the calling thread's last failure's LIZ_E_* code (LIZ_OK after a call that succeeded): what a NULL handle meant */
LIZ_API int liz_last_error_code(void);
/* ai: 1 where the codec's vector paths are compiled in (NEON, WebAssembly SIMD), else 0 */
LIZ_API int liz_simd(void);

enum {
  LIZ_OK = 0,
  LIZ_E_ARG = -1,          /* ai: an argument out of range */
  LIZ_E_NOMEM = -2,
  LIZ_E_IO = -3,           /* ai: a file could not be read or written */
  LIZ_E_UNSUPPORTED = -4,  /* ai: not in this build, or not on this device */
  LIZ_E_FORMAT = -5,       /* ai: a format the codec refuses */
  LIZ_E_TOOBIG = -6,       /* ai: a file past what the transfer or the store holds */
  LIZ_E_STATE = -7,        /* ai: not now (no file yet, no frame ready) */
  LIZ_E_INTERNAL = -8
};

enum {
  LIZ_BLOCK = 473,         /* ai: a block: the id and the payload */
  LIZ_ID_BYTES = 4,        /* ai: the id, little-endian */
  LIZ_PAYLOAD = 469,
  LIZ_MAX_BLOCKS = 128,    /* ai: blocks a symbol at most (LIZARD-1024) */
  LIZ_RINGS = 4,           /* ai: the borders: rings of 32, 64, 96 or 128 cells a side */
  LIZ_RING_DEFAULT = -1,   /* ai: the senders' default ring, the 128 */
  LIZ_GAP_MODULES = 12,    /* ai: two codes side by side are this many modules apart */
  LIZ_QUIET = 2            /* ai: the margin painted round a symbol, in modules */
};

typedef enum {
  LIZ_GREY8 = 1,  /* ai: one byte a pixel */
  LIZ_RGBX8 = 2,  /* ai: four bytes a pixel, grey in each of the first three; the fourth 255 (painting) or ignored */
  LIZ_RGBA8 = 3,  /* ai: as RGBX8 when painting; a captured image's luma by its R, G and B */
  LIZ_BGRA8 = 4   /* ai: as RGBX8 when painting; a captured image's luma by its B, G and R */
} liz_pixfmt;

/* ---- 1. format arithmetic ---------------------------------------------------------------------------------------- */

/* ai: blocks: the format's size, 1 to 128 (LIZARD-8 to LIZARD-1024: the number is 8 x blocks), the version its word
 * ai: names; a symbol of it carries liz_blocks_for(blocks) blocks, the code's rate following the frequency (7/8 on its
 * ai: lowest sub-channels, 1/2 on its highest, 3/4 between: 51 at LIZARD-432); ring: 0 to 3 or LIZ_RING_DEFAULT;
 * ai: fps: 1 to 255, the display rate the symbol's word states; codes: 1, or 2 symbols side by side (one format, one
 * ai: word, the frame's blocks split between them, the first code the first ones). */
typedef struct {
  int blocks, ring, fps, codes;
} liz_format;

/* ai: What a format paints. n: the picture's samples a side; span: the picture's modules a side (2 x the ring's
 * ai: cells); pxm: pixels a module; side: a symbol's pixels a side, its margin in; width, height: the frame's pixels
 * ai: (codes symbols and the gaps between them); gap: the pixels between two codes; frame_blocks: blocks a frame
 * ai: (codes x liz_blocks_for(blocks)). */
typedef struct {
  int n, span, pxm, side, width, height, gap, frame_blocks;
} liz_geometry;

/* ai: the cells a side of ring r (0 to 3; LIZ_RING_DEFAULT the default's), or LIZ_E_ARG */
LIZ_API int liz_ring_cells(int ring);
LIZ_API int liz_geometry_of(const liz_format *format, liz_geometry *out);
/* ai: The pixels a side a symbol of this many blocks needs on a display for its outer coefficients to keep 2.7 pixels
 * ai: a cycle (the web sender's ROOM_FOR); a negative error for an argument out of range. */
LIZ_API double liz_room_for(int blocks, int ring);
/* ai: The blocks a symbol of a format's size (blocks, 1 to 128) carries: the format's rate profile's count; 0 outside. */
LIZ_API int liz_blocks_for(int blocks);
/* ai: The most blocks a symbol may carry, at most top_blocks, for codes symbols in a room of w x h display pixels (the
 * ai: web sender's pickVersion: each symbol's room is the width over codes and the gaps, against the height); at least 1. */
LIZ_API int liz_pick(double w, double h, int codes, int ring, int top_blocks);

/* ---- 2. the per-frame codec -------------------------------------------------------------------------------------- */

typedef struct liz_encoder liz_encoder;
LIZ_API liz_encoder *liz_encoder_new(const liz_format *format);
LIZ_API void liz_encoder_free(liz_encoder *e);
LIZ_API int liz_encoder_geometry(const liz_encoder *e, liz_geometry *out);
/* ai: One frame: blocks, frame_blocks x LIZ_BLOCK bytes (each an id and a payload; the codec adds the CRC);
 * ai: picture, the frame's count since the start (its value mod 4 is what the pilots paint: count every frame shown,
 * ai: once, in order); out, width x height pixels of fmt, rows stride bytes apart. Grey on white margins. */
LIZ_API int liz_encoder_paint(liz_encoder *e, const uint8_t *blocks, uint32_t picture, uint8_t *out, int stride, liz_pixfmt fmt);

typedef struct liz_decoder liz_decoder;
/* ai: nmax: the largest picture to decode, 256 to 1536 (0: 1536, every format); smaller saves memory. */
LIZ_API liz_decoder *liz_decoder_new(int nmax);
LIZ_API void liz_decoder_free(liz_decoder *d);
/* ai: the most blocks a frame can verify: the decode's buffer is this many x LIZ_BLOCK bytes */
LIZ_API int liz_decoder_max_blocks(const liz_decoder *d);

typedef struct {
  int found;          /* ai: a symbol registered */
  int ring;           /* ai: the ring that registered, 0 to 3; -1 none */
  int n;              /* ai: the picture it was finished at; 0 not finished */
  int word;           /* ai: the frame's own word read */
  int blocks;         /* ai: the size of the format it was decoded at, its version (0 none) */
  int fps;            /* ai: the display rate its word states */
  int held_used;      /* ai: decoded at the held word, its own not read */
  int total;          /* ai: the blocks the symbol carries */
  int verified;       /* ai: the blocks verified */
  float quad[8];      /* ai: its corners in the image, x y x y ..., top left, top right, bottom right, bottom left */
  int pilot_blocks;   /* ai: blocks the pilots were read over (0 none) */
  float pilot_r[2];   /* ai: the pilots' readings over the even blocks and the odd: their signs the picture's count */
  float pilot_sd[2];  /* ai: their standard errors */
} liz_decoded;

/* ai: One captured frame (or one region of it: px at the region's first pixel, w x h, rows stride bytes apart): its
 * ai: verified blocks packed into verified (liz_decoder_max_blocks x LIZ_BLOCK bytes), their count returned. held: the
 * ai: caller's, the blocks a symbol of the last word read (0 none), shared by every decoder reading one stream; a frame
 * ai: whose own word does not read is decoded at it, and with none held is not decoded; it is updated whenever a
 * ai: frame's own word reads. Every frame is otherwise decoded on its own. out may be NULL. */
LIZ_API int liz_decode(liz_decoder *d, const uint8_t *px, int w, int h, int stride, liz_pixfmt fmt, int *held,
                       uint8_t *verified, liz_decoded *out);

/* ai: Where a camera frame's symbols are looked for: layout 1, the centre square; 2, the centre region twice as long as
 * ai: it is high along the frame's long side, cut into two squares (a sender's two codes). rect[k] = x, y, w, h; returns
 * ai: how many (1 or 2). */
LIZ_API int liz_layout_rects(int w, int h, int layout, int rect[2][4]);

/* ai: the test stream's payload for an id (SHAKE256 of its four bytes): a receiver judges a frame of it by its bytes */
LIZ_API void liz_stream_fill(uint32_t id, uint8_t payload[LIZ_PAYLOAD]);

/* ---- 3. the transfer --------------------------------------------------------------------------------------------- */

typedef struct liz_tx liz_tx;
enum { LIZ_TX_COPY = 1 };  /* ai: liz_tx_new copies the bytes; without it they must outlive the liz_tx */
/* ai: A file: its bytes, its name and media type as its header carries them (a name cut to 255 bytes, a type kept
 * ai: where it is printable ASCII of at most 162). */
LIZ_API liz_tx *liz_tx_new(const uint8_t *data, size_t length, const char *name, const char *type, int flags);
/* ai: A file read from a path (into memory); name NULL takes the path's last part. */
LIZ_API liz_tx *liz_tx_new_path(const char *path, const char *name, const char *type);
/* ai: The test stream from first_id (0: a random one): each block the stream's payload for its id. */
LIZ_API liz_tx *liz_tx_new_test(uint32_t first_id);
LIZ_API void liz_tx_free(liz_tx *t);
/* ai: The next frame's n blocks (n x LIZ_BLOCK bytes): a frame asked for twice is two frames. Returns how many of
 * ai: them carry the file's data (the others its header and manifest). */
LIZ_API int liz_tx_next(liz_tx *t, int n, uint8_t *blocks);

typedef struct {
  int test;              /* ai: the test stream */
  uint64_t length;       /* ai: the file's bytes */
  uint32_t chunks;       /* ai: its 4 MiB chunks */
  uint32_t lap;          /* ai: data blocks a lap of the schedule */
  uint8_t root[32];      /* ai: BLAKE3 of the file (b3sum's) */
  uint64_t sent;         /* ai: the file's bytes as they go: every chunk's zstd frame, or its own bytes (2026-10-05) */
} liz_tx_info;
LIZ_API int liz_tx_info_get(const liz_tx *t, liz_tx_info *out);

typedef struct liz_rx liz_rx;
/* ai: Where a receiver keeps what it holds: dir NULL, in memory; else files under dir (emptied first), the finished
 * ai: file at a path under it. max_bytes (memory only, 0 no cap) bounds the stored blocks and the file together, which
 * ai: peak near twice the file's size (its bytes and every chunk's blocks before it solves); a chunk's fountain decoder,
 * ai: up to a chunk's 4 MiB, is outside it. A store that cannot hold a transfer (over the cap, a write refused, a disk
 * ai: full) ends it: never done, its reason in LIZ_META_ERROR, until a new file's header begins afresh. */
typedef struct {
  const char *dir;
  uint64_t max_bytes;
} liz_store;
LIZ_API liz_rx *liz_rx_new(const liz_store *store);
LIZ_API void liz_rx_free(liz_rx *r);

/* ai: What a frame's verified blocks came to. seen: blocks taken (the header's and manifest's in, repeats in); bad:
 * ai: a test frame's blocks that are not the stream's; judged: a test frame's data blocks; fresh: data blocks new to
 * ai: this receiver; test: the frame was the test stream's (none of its blocks go to a file). */
typedef struct {
  int seen, bad, judged, fresh, test;
} liz_verdict;
/* ai: One frame's verified blocks, as liz_decode packed them (of both of a frame's regions, one call each or one
 * ai: together). A chunk completed by them is solved and checked inside the call (tens of ms for 4 MiB). */
LIZ_API int liz_rx_frame(liz_rx *r, const uint8_t *verified, int count, liz_verdict *out);

typedef struct {
  int header;            /* ai: the file's header read */
  int done;              /* ai: the file whole and verified by its root, and kept whole by the store */
  uint64_t length;       /* ai: its bytes */
  uint64_t bytes_in;     /* ai: about how many are in */
  double fraction;       /* ai: bytes_in over length */
  uint32_t chunks, verified, rejected;
  double solve_ms;       /* ai: the fountain's and the hash's time so far */
  uint64_t sent;         /* ai: the file's bytes as they go (every chunk's zstd frame or its own); 0 until the manifest has said them */
  uint64_t sent_in;      /* ai: how many of them are in: what a rate and a time left should count */
} liz_progress;
LIZ_API int liz_rx_progress(liz_rx *r, liz_progress *out);
/* ai: each chunk's state into per (at most cap): 0 to 99 the share in (a floor until the manifest is in), 255
 * ai: verified; returns the chunks */
LIZ_API int liz_rx_chunks(liz_rx *r, uint8_t *per, int cap);
enum { LIZ_META_NAME = 0, LIZ_META_TYPE = 1, LIZ_META_ROOT = 2, LIZ_META_PATH = 3, LIZ_META_ERROR = 4 };
/* ai: the header's name and type, the root as hex, the finished file's path (a directory store's), the store's failure
 * ai: in this transfer (the reason it ended); "" where there is none */
LIZ_API const char *liz_rx_meta(liz_rx *r, int which);
/* ai: The finished file's bytes in a memory store (valid until the next call on r); LIZ_E_STATE before it is done. */
LIZ_API int liz_rx_data(liz_rx *r, const uint8_t **data, size_t *length);
/* ai: The transfer and every file received let go, so the same file still in the light is received again. */
LIZ_API void liz_rx_clear(liz_rx *r);

/* ---- 4. engines: threads, the GPU (Vulkan) where a device runs it, else the CPU; native builds only --------------- */
/* ai: What the native apps run: a receiver decoding camera frames on its own threads (the GPU decoder in batches, or
 * ai: the C on a pool of workers) into a transfer, and a sender painting frames ahead on its own threads (the GPU's
 * ai: encoder, or the C on several). The GPU paths read their kernels and nets from an assets directory (share/lizard
 * ai: as installed); without one, or with no Vulkan device that runs them, the CPU does the work. A wasm build has no
 * ai: engines (LIZ_E_UNSUPPORTED). Their stats are JSON, the field names the apps' (not a stable interface). */

enum { LIZ_DECODER_AUTO = 0, LIZ_DECODER_GPU = 1, LIZ_DECODER_CPU = 2 };
typedef struct {
  const char *assets;      /* ai: the GPU's kernels and nets; NULL: the CPU decoder alone */
  const char *cache_dir;   /* ai: the GPU's pipeline cache; NULL none */
  const char *store_dir;   /* ai: where the file is kept, under store_dir/lizard-xfer (that folder emptied first) */
  const char *device;      /* ai: a substring of the Vulkan device's name to use; NULL the first discrete, else the first */
  const char *precision;   /* ai: the GPU decoder's nets: "auto" (NULL), "int8", "f16" or "f32" */
  int decoder;             /* ai: LIZ_DECODER_* */
  int cpu_threads;         /* ai: the CPU decoder's workers; 0 its own rule (one, more while frames are lost) */
  int layout;              /* ai: 1 the centre square, 2 the two squares of a 2:1 region (liz_layout_rects) */
  void (*log)(void *user, const char *line);   /* ai: NULL none; called from the receiver's threads */
  void *user;
} liz_receiver_config;

typedef struct liz_receiver liz_receiver;
/* ai: release(user, tag): a frame pushed is the caller's again (from a receiver thread, or inside the push). */
LIZ_API liz_receiver *liz_receiver_new(const liz_receiver_config *config, void (*release)(void *user, uint64_t tag));
LIZ_API void liz_receiver_free(liz_receiver *r);
/* ai: One camera frame's luma plane, w x h, rows stride bytes apart, at timestamp_ns on the camera's clock: copied (or
 * ai: dropped) before the call returns, release(tag) called by then. Never blocks on the device. */
LIZ_API int liz_receiver_push(liz_receiver *r, const uint8_t *luma, int w, int h, int stride, int64_t timestamp_ns, uint64_t tag);
/* ai: Android: a camera AHardwareBuffer (YUV), read by the GPU in place; it stays the caller's until release(tag). */
LIZ_API int liz_receiver_push_hardware_buffer(liz_receiver *r, void *ahardwarebuffer, int w, int h, int64_t timestamp_ns, uint64_t tag);
/* ai: 1 where frames must come as luma planes (the CPU decoder), 0 where hardware buffers are read in place */
LIZ_API int liz_receiver_wants_luma(liz_receiver *r);
LIZ_API const char *liz_receiver_stats(liz_receiver *r);
/* ai: The frames decoded since since_ms (on the camera's clock), 8 doubles each into out (at most cap doubles), after
 * ai: the held word's version: ms, verified blocks, new blocks, found, the pilots' r and its error over the even blocks,
 * ai: then over the odd (NaN none); returns the doubles written. For a camera's phase lock. */
LIZ_API int liz_receiver_series(liz_receiver *r, double since_ms, double *out, int cap);
/* ai: results wanted soon (smaller GPU batches), and the most frames a GPU batch waits for (0: its own size) */
LIZ_API void liz_receiver_soon(liz_receiver *r, int on);
LIZ_API void liz_receiver_batch_cap(liz_receiver *r, int frames);
/* ai: the received file's path once whole and verified, else "" */
LIZ_API const char *liz_receiver_file(liz_receiver *r);
LIZ_API void liz_receiver_clear(liz_receiver *r);
/* ai: Android: the camera's reader closed, every frame handed back (the GPU lets go of that reader's buffers) */
LIZ_API void liz_receiver_camera_closed(liz_receiver *r);

enum { LIZ_PAINTER_CPU = 0, LIZ_PAINTER_GPU = 1, LIZ_PAINTER_AUTO = 2 };
typedef struct liz_sender liz_sender;
/* ai: A file (data NULL: the test stream), its bytes copied with LIZ_TX_COPY, else alive as long as the sender. */
LIZ_API liz_sender *liz_sender_new(const uint8_t *data, size_t length, const char *name, const char *type, int flags);
LIZ_API void liz_sender_free(liz_sender *s);
/* ai: The format to paint (a re-pick keeps the transfer), on painter (LIZ_PAINTER_*: auto takes the GPU where its first
 * ai: frames are the C's), threads CPU painters, the GPU's assets and device as the receiver's. */
LIZ_API int liz_sender_configure(liz_sender *s, const liz_format *format, int painter, int threads, const char *assets, const char *device);
LIZ_API int liz_sender_geometry(liz_sender *s, liz_geometry *out);
/* ai: 1 where the next frame is painted */
LIZ_API int liz_sender_ready(liz_sender *s);
/* ai: The next frame in order (width x height of fmt, rows stride bytes apart): 1, it is the screen's now; 0, not painted
 * ai: yet (dst untouched). Each frame taken once; show them in order, each for as long as the display rate asks. */
LIZ_API int liz_sender_take(liz_sender *s, uint8_t *dst, int stride, liz_pixfmt fmt);
LIZ_API const char *liz_sender_stats(liz_sender *s);

#ifdef __cplusplus
}
#endif
#endif
