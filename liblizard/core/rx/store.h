// ai: Where the transfer's receiving end keeps what it holds (2026-10-03, for the library: xfer_rx_core.h): each
// ai: chunk's stored blocks until it is solved, and the file being rebuilt. lizard-web/fountain-worker.mjs has two, OpfsStore
// ai: and sim/xfer.mjs MemoryStore; so does this: files under a directory (the app's, as before), or memory with a cap
// ai: (a wasm build, or a caller with no directory). Used from one thread at a time, the receiving end's own.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lizard {

class Store {
 public:
  virtual ~Store() = default;
  // ai: The transfer in hand let go whole: what it held goes, its finished file too (a failed transfer's, the
  // ai: receiver's Clear). A memory store's finished file goes with it, being unreachable once a new header is read.
  virtual void clear() = 0;
  // ai: OpfsStore.removeAll: the receiver's Clear, the transfer in hand and every finished file; a file store keeps
  // ai: the transfers it parked
  virtual void removeAll() = 0;

  // ai: Transfers kept across receivers (2026-10-22). A store that keeps them (a file store) holds each transfer in a
  // ai: folder of its own, named by its id (the root's first 16 hex digits): its header, the manifest blocks it took,
  // ai: each chunk's stored blocks, the output so far, the chunks verified into it, and meta.json for an app's list. A
  // ai: header of another root parks the transfer in hand (park: its folder left as it is); its own header read again,
  // ai: by this receiver or a later one, resumes it (open). A memory store keeps one transfer at a time, as before:
  // ai: open begins it afresh, park is clear, and the notes are nothing.
  virtual bool keeps() const { return false; }
  // ai: What a transfer's folder held from before: the manifest blocks taken (index, payload), the chunks verified into
  // ai: the output, and the chunks with stored blocks (read() gives them). found: there was a folder to resume.
  struct Before {
    bool found = false;
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> manifest;
    std::vector<uint32_t> verified, chunks;
  };
  // ai: The transfer whose header is header (469 B) and whose file is length bytes: its folder resumed where it holds
  // ai: this header, else begun afresh, its output length bytes; false when the store cannot hold it (error says why).
  // ai: open, append and writeOut return false on a failure, which ends the transfer (xfer_rx_core.cpp): a file the
  // ai: store cannot keep whole is never reported received (before 2026-10-03 it was, with 0 bytes or a short file).
  virtual bool open(const std::string& id, const uint8_t* header, uint64_t length, Before& before) = 0;
  // ai: a manifest block taken; a chunk verified into the output (the output's bytes made durable first, so a chunk
  // ai: noted is a chunk on disk); the transfer's line for an app's list (meta.json)
  virtual void noteManifest(uint32_t m, const uint8_t* payload) { (void)m; (void)payload; }
  virtual void noteVerified(uint32_t c) { (void)c; }
  virtual void noteMeta(const std::string& json) { (void)json; }
  // ai: chunk c's stored blocks cut back to its first n (a park's: the blocks since the transfer's last header may be
  // ai: the next file's, xfer_rx_core.cpp)
  virtual void truncate(uint32_t c, uint32_t n) { (void)c; (void)n; }
  // ai: the transfer in hand set aside for a later header of its own (a memory store lets it go)
  virtual void park() { clear(); }
  // ai: a parked transfer let go, by its id (the transfer in hand is the caller's to clear)
  virtual void forget(const std::string& id) { (void)id; }
  // ai: one block of chunk c (its symbol, then the 469 B payload)
  virtual bool append(uint32_t c, uint32_t sym, const uint8_t* payload) = 0;
  // ai: chunk c's stored records (symbol then payload, 473 B each), in arrival order
  virtual std::vector<uint8_t> read(uint32_t c) = 0;
  virtual void drop(uint32_t c) = 0;
  virtual bool writeOut(uint64_t off, const uint8_t* p, size_t len) = 0;
  // ai: the output verified: its path for a file store, "" for memory (the bytes then in data()); a failure sets error
  virtual std::string finish(const std::string& name) = 0;
  // ai: the finished file's bytes where the store keeps them in memory, else null
  virtual const std::vector<uint8_t>* data() const { return nullptr; }
  std::function<void(const std::string&)> log;
  // ai: this transfer's failure (a write refused, the memory cap reached), "" none; cleared as a transfer begins and by
  // ai: removeAll; the transfer reports it
  std::string error;

 protected:
  void fail(const std::string& s) { error = s; if (log) log(s); }
};

// ai: Files under dir, a folder a transfer (keeps()): <id>/header, manifest (index and payload a block), c<k> (a chunk's
// ai: stored blocks), out (the output), verified (a chunk index each), meta.json; once verified the output is
// ai: <id>/done/<its name> and the rest goes. Opening it keeps the folders that hold a header (parked transfers) and
// ai: lets everything else go (one receiver a directory: a finished file nobody moved, the layout before 2026-10-22).
std::unique_ptr<Store> fileStore(const std::string& dir);
// ai: Memory, at most maxBytes held at once (the stored blocks and the output together; 0, no cap; a chunk's fountain
// ai: decoder, up to a chunk's size, is outside it). A transfer that would pass it ends and says so.
std::unique_ptr<Store> memoryStore(uint64_t maxBytes);

}  // namespace lizard
