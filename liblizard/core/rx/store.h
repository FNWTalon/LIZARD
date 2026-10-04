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
  // ai: OpfsStore.clear, a new transfer's: what the last one held goes (a file store keeps its finished files, which the
  // ai: app moves out; a memory store's finished file goes too, being unreachable once a new header is read)
  virtual void clear() = 0;
  // ai: OpfsStore.removeAll: the receiver's Clear, finished files too
  virtual void removeAll() = 0;
  // ai: A new transfer's output, length bytes; false when the store cannot hold it (error says why). begin, append and
  // ai: writeOut return false on a failure, which ends the transfer (xfer_rx_core.cpp): a file the store cannot keep
  // ai: whole is never reported received (before 2026-10-03 it was, with 0 bytes or a short file).
  virtual bool begin(uint64_t length) = 0;
  // ai: one block of chunk c (its symbol, then the 469 B payload)
  virtual bool append(uint32_t c, uint32_t sym, const uint8_t* payload) = 0;
  // ai: chunk c's stored records (symbol then payload, 473 B each), in arrival order
  virtual std::vector<uint8_t> read(uint32_t c) = 0;
  virtual void drop(uint32_t c) = 0;
  virtual bool writeOut(uint64_t off, const uint8_t* p, size_t len) = 0;
  virtual std::vector<uint8_t> readOut(uint64_t off, size_t len) = 0;
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

// ai: Files under dir, emptied here (one receiver a directory): a file of stored blocks a chunk (c<k>), one output
// ai: file a transfer (out<k>), moved to done<k>/<its name> once verified.
std::unique_ptr<Store> fileStore(const std::string& dir);
// ai: Memory, at most maxBytes held at once (the stored blocks and the output together; 0, no cap; a chunk's fountain
// ai: decoder, up to a chunk's size, is outside it). A transfer that would pass it ends and says so.
std::unique_ptr<Store> memoryStore(uint64_t maxBytes);

}  // namespace lizard
