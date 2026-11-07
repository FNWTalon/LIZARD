// ai: The transfer's sending end (2026-10-01): liblizard/sim/xfer.mjs XferSender ported name for name, over
// ai: liblizard/src/xfer.c (the chunks' chaining values, their root, the header and the manifest) and Wirehair
// ai: through the web's shim (liblizard/wirehair/shim.cpp), as the receiving end (rx/xfer_rx.h) takes them. A file
// ai: goes in 2^chunk_log2 chunks (4 MiB), each zstd-compressed on its own where that is shorter (2026-10-05,
// ai: liblizard/zstd/shim.c, the parameters pinned so the frame is the JS sender's), a fountain over each chunk's sent
// ai: bytes; the header every other control slot and the manifest (each chunk's chaining value, bytes as sent and seed
// ai: attempt) between, the control cycle within an eighth of a lap; data blocks by the schedule, shuffled in a frame
// ai: by a generator seeded by the frame count. The same ids and bytes as the JS sender for the same file
// ai: (tools/tx_check.cpp holds it to one). The chunks are made at once, a worker a core (2026-11-06): a chunk's zstd
// ai: frame, Wirehair encoder and chaining value depend on no other chunk, so the bytes are the serial sender's.
// ai: Streaming (2026-11-06): the constructor hashes the chunks' own bytes (their chaining values and the root, all the
// ai: header needs) and returns; the workers make the chunks (zstd, Wirehair) behind it while frames go. Until the
// ai: last is made, a frame carries one control block (the header, or a manifest block once its 12 chunks are made)
// ai: and the data of the chunks made so far in turn, fresh symbols each; then the schedule as before, every chunk's
// ai: symbols going on from where they were. A receiver keeps a chunk's blocks from the header on and solves once the
// ai: manifest is whole, so it needs nothing new. A file of one chunk is made before the constructor returns (its
// ai: header carries its sent bytes and seed), and so is every file under emscripten (no threads).
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "xfer.h"
}

namespace lizard {

class XferTx {
 public:
  // ai: data: the file's bytes, alive as long as this (the app maps the file); name and type as the header carries them
  // ai: (a name cut to 255 bytes at a character, a type kept only where it is printable ASCII of at most 158).
  // ai: Throws std::runtime_error where the file cannot go (too many chunks, a chunk Wirehair seeded apart). Returns
  // ai: with the root and the header made (the chaining values hashed a thread a physical core); the chunks are made
  // ai: after it by a worker a physical core, each taking the next chunk, until every one is (fault() says a failure);
  // ai: the destructor stops and joins them.
  XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2 = 22);
  ~XferTx();
  XferTx(const XferTx&) = delete;
  XferTx& operator=(const XferTx&) = delete;

  // ai: The ids of the next frame of n blocks (advances the schedule: a frame asked for twice is two frames).
  void frameIds(uint32_t* out, int n);
  // ai: What block `id` carries: 472 bytes into out.
  void block(uint32_t id, uint8_t* out) const;

  uint32_t chunks() const { return chunks_; }
  uint32_t lap() const { return lap_; }           // ai: data blocks a lap of the schedule (0 until every chunk is made)
  uint64_t sentBytes() const { return sentBytes_; }   // ai: the file's bytes as they go (every chunk's frame, or its own; 0 until made)
  const uint8_t* root() const { return root_; }   // ai: 32 bytes, b3sum's answer for the file
  std::string rootHex() const;
  bool made() const { return made_.load(std::memory_order_acquire) == chunks_; }   // ai: every chunk made
  // ai: Waits until every chunk is made (tx_check ids, whose frames from then are the serial sender's); throws a
  // ai: worker's failure.
  void waitMade();
  std::string fault() const;   // ai: a worker's failure, "" none: the transfer cannot finish

 private:
  uint32_t next();   // ai: the schedule's next data id (sim/xfer.mjs Schedule.next)
  void make(uint32_t c, std::vector<uint8_t>& frame);   // ai: chunk c made (a worker's)
  void work();        // ai: a worker: the next chunk until none is left, or a failure stops them
  void schedule();    // ai: the schedule, once every chunk is made (frameIds')
  void warm(uint32_t* out, int n);   // ai: a frame's ids before that

  size_t size_;
  uint32_t chunks_ = 0;
  const uint8_t* data_ = nullptr;
  std::atomic<uint32_t> lap_{0};
  std::atomic<uint64_t> sentBytes_{0};
  std::vector<uint8_t> cvs_;     // ai: every chunk's chaining value (two chunks or more)
  std::vector<size_t> len_;      // ai: a chunk's own bytes
  std::vector<uint32_t> sent_;   // ai: its bytes as sent: its zstd frame, or its own
  std::vector<uint8_t> seed_;    // ai: its Wirehair seed attempt (0 where K is 1)
  std::vector<uint32_t> K_;      // ai: Wirehair source blocks a chunk, of the sent bytes (1: sent unfountained)
  struct WhFree { void operator()(void* e) const; };
  std::vector<std::unique_ptr<void, WhFree>> enc_;   // ai: a chunk's encoder, null where K is 1
  std::vector<std::vector<uint8_t>> one_;            // ai: the sent bytes of a chunk of one block, else empty
  uint8_t root_[32]{}, header_[XFER_PAYLOAD]{};
  std::vector<std::vector<uint8_t>> manifest_;
  // ai: the schedule (Schedule): symbols a chunk sent, the round's place, each chunk's weight a round and accumulator
  std::vector<uint32_t> sym_, w_, acc_;
  uint32_t i_ = 0, max_ = 0;
  // ai: the control cycle: how many data blocks between control slots, since the last, which control is next
  uint32_t every_ = 1, since_ = UINT32_MAX, ctl_ = 0, frame_ = 0;
  // ai: streaming: a chunk's slots written before its ready_ is set, a manifest block's before its mready_ (the
  // ai: workers'); left_, chunks of a manifest group not yet made; next_, the next chunk a worker takes
  std::unique_ptr<std::atomic<uint8_t>[]> ready_, mready_;
  std::unique_ptr<std::atomic<uint32_t>[]> left_;
  std::atomic<uint32_t> made_{0}, next_{0};
  std::atomic<bool> stop_{false};
  std::vector<std::thread> workers_;
  mutable std::mutex faultMu_;
  std::string fault_;
  bool scheduled_ = false;   // ai: frameIds' own: the schedule built
  uint32_t turn_ = 0;        // ai: the warm frames' turn over the chunks made
};

}  // namespace lizard
