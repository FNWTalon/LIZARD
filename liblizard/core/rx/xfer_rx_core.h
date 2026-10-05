// ai: The transfer's receiving end with no thread of its own (2026-10-03, for the library, which runs it on the caller's
// ai: thread and in a wasm build with no threads at all; xfer_rx.h is the app's threaded form over the same two parts):
// ai:   XferRxCore  lizard-web/fountain-worker.mjs take (the live-header rule, the held ring) and sim/xfer.mjs XferReceiver
// ai:               (chunk states, bans, a new transfer on a header with other bytes): CRC-verified blocks in, a file out,
// ai:               each chunk checked against the file's BLAKE3 root, kept in a Store (store.h: files or memory)
// ai:   XferJudge   sim/phy.mjs blockJudge (a frame is the test stream's when a verified data block carries its id's
// ai:               SHAKE256, and its other data blocks are then bad) and lizard-web/recv.mjs take (the dedupe: two rotating id
// ai:               sets of 65,536, control ids always passed on, a test frame's blocks never, the sets cleared when the
// ai:               core reads a header of another root)
// ai: The layouts are src/xfer.h's, with Wirehair (liblizard/vendor/wirehair, liblizard/wirehair/shim.cpp), BLAKE3 (liblizard/vendor/blake3)
// ai: and zstd (liblizard/vendor/zstd, liblizard/zstd/shim.c; 2026-10-05: a chunk sent as a frame is decompressed on recovery, before its hash).
// ai: A solve (Wirehair and BLAKE3 over a 4 MiB chunk) is tens of ms, run inside the block() that completes the chunk.
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "store.h"

namespace lizard {

constexpr int XFER_BLOCK_BYTES = 473;   // ai: src/xfer.h XFER_BLOCK: the id (4, little-endian) and the payload (469)
struct Block { std::array<uint8_t, XFER_BLOCK_BYTES> bytes; };
static_assert(sizeof(Block) == XFER_BLOCK_BYTES, "a vector of Blocks is the packed blocks");

// ai: What a frame's verified blocks came to: seen, those not judged bad (control blocks and repeats in); bad, a test
// ai: frame's data blocks that are not the stream's; test, the frame was the test stream's; judged, the data blocks
// ai: held to the stream (all of them on a test frame, else 0); fresh, data ids new to the dedupe (a test frame's
// ai: counted too, for goodput, but none of its blocks passed on).
struct FrameVerdict { int seen = 0, bad = 0, judged = 0, fresh = 0; bool test = false; };

struct XferProgress {
  bool header = false;        // ai: a header read (this transfer's)
  bool live = false;          // ai: and read within the last LIVE_BLOCKS data blocks: data blocks go to the transfer
  std::string name, type, root;   // ai: the header's name and media type; root, BLAKE3 of the file as hex (b3sum's)
  uint64_t length = 0;        // ai: the file's bytes
  uint32_t chunks = 0, verified = 0, manifest = 0, manifestHave = 0;
  std::vector<uint8_t> per;   // ai: each chunk's share in hand, 0 to 99 (a floor until the manifest says its blocks), 255 verified
  double fraction = 0;        // ai: sim/xfer.mjs fractionDone: the share of the file in, 0 to 1 (by the bytes as sent once known)
  uint64_t bytesIn = 0;       // ai: fraction x length
  // ai: the file's bytes as they go, every chunk's zstd frame or its own (0 until the manifest has said them), and how
  // ai: many are in: a verified chunk's all, another's blocks in hand (2026-10-05, what a time left should count)
  uint64_t sent = 0, sentIn = 0;
  bool done = false;
  std::string path;           // ai: the verified file, once done (a file store's; "" for memory)
  std::string error;          // ai: the store's failure in this transfer (store.h), "" none
  bool failed = false;        // ai: the store could not keep the file: the transfer ended there, never done
  // ai: counts, the web's names: rejected chunks, manifests rejected, headers refused, chunks Wirehair wanted a block
  // ai: more than K for, solve ms; forwarded blocks handed to the worker (control ones in), doneAt how many had been
  // ai: when the file finished (0 before), held the blocks waiting for a header, queued those not yet taken, lost those
  // ai: the queue had no room for (the last two the threaded form's)
  uint32_t rejected = 0, manifestRejected = 0, refused = 0, over = 0;
  double solveMs = 0;
  uint64_t forwarded = 0, doneAt = 0, lost = 0;
  uint32_t held = 0, queued = 0;
};

class XferRxCore {
 public:
  XferRxCore(std::unique_ptr<Store> store, std::function<void(const std::string&)> log = nullptr);
  ~XferRxCore();
  XferRxCore(const XferRxCore&) = delete;
  XferRxCore& operator=(const XferRxCore&) = delete;

  // ai: One block a judge passed on (the fountain worker's take).
  void block(const uint8_t* block473);
  // ai: The transfer in hand and every file received let go (the page's Clear).
  void clearAll();
  void progress(XferProgress& p) const;
  // ai: Counts the headers of another root read (a new transfer), which clears a judge's dedupe.
  const std::atomic<uint64_t>& rootGen() const;
  // ai: The finished file's bytes where the store keeps them in memory, else null.
  const std::vector<uint8_t>* data() const;

  struct Rx;

 private:
  std::unique_ptr<Rx> rx_;
};

class XferJudge {
 public:
  using Pass = std::function<void(const uint8_t*)>;
  explicit XferJudge(const std::atomic<uint64_t>& rootGen) : gen_(rootGen) {}
  // ai: A frame's verified blocks, packed (473 B each): judged, deduplicated, and each one the transfer should take
  // ai: handed to pass, control blocks first.
  FrameVerdict frame(const uint8_t* blocks, size_t count, const Pass& pass);
  // ai: One block past the judge: true when its id is new data. A control block is always passed on (its id is every
  // ai: transfer's; only its bytes say which) and is never new.
  bool take(const uint8_t* block473, const Pass& pass);
  // ai: The ids forgotten (the page's Clear).
  void clear();

 private:
  const std::atomic<uint64_t>& gen_;
  std::unordered_set<uint32_t> seenIds_, seenOld_;
  uint64_t genSeen_ = 0;
  void remember(uint32_t id);
};

}  // namespace lizard
