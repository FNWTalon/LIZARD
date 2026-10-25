// ai: The transfer's receiving end, native, the app's threaded form: CRC-verified
// ai: blocks in, a file out, each chunk checked against the file's BLAKE3 root. Its two parts are xfer_rx_core.h's
// ai: (since 2026-10-03; the same chain, no thread of its own): XferJudge on the producer's thread (frame, take) and
// ai: XferRxCore on this object's worker thread, its store files under storeDir. Threads: take, frame, clear and
// ai: forget from any thread, one at a time under the transfer's own lock (2026-10-25: two receivers feed one transfer
// ai: while one takes over from the other, receiver.h create's from; each receiver held a lock of its own until then);
// ai: progress, taken and drain from any thread. Wirehair solves and BLAKE3 run on the worker, so take never waits on
// ai: them.
#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "xfer_rx_core.h"

namespace lizard {

class XferRx {
 public:
  // ai: storeDir: a directory the receiver may write; it works in storeDir/lizard-xfer, a folder a transfer (store.h
  // ai: fileStore): the transfers a receiver before this one parked stay there to resume, anything else is let go (one
  // ai: receiver a directory). A finished file is storeDir/lizard-xfer/<id>/done/<its name>.
  XferRx(const std::string& storeDir, std::function<void(const std::string&)> log = nullptr);
  ~XferRx();
  XferRx(const XferRx&) = delete;
  XferRx& operator=(const XferRx&) = delete;

  // ai: One CRC-verified block, past the judge: true when its id is new data. A control block is always passed on
  // ai: (its id is every transfer's; only its bytes say which) and is never new.
  bool take(const uint8_t* block473);
  FrameVerdict frame(const std::vector<Block>& verified);
  XferProgress progress();
  // ai: XferRxCore::taken, read from any thread: the blocks the transfer took, ever (a receiver's file rate)
  uint64_t taken() const { return rx_->taken().load(std::memory_order_relaxed); }
  // ai: The page's Clear: the ids, the transfer in hand and every file received let go, so the same file still in the
  // ai: light is received again. Done when it returns (2026-10-10): it waits for the worker to clear (after whatever was
  // ai: queued before it), so progress() read after it never shows the transfer let go. It queued the clear and returned
  // ai: until then, and an app's poll in between read the file it had just forgotten as still received.
  void clear();
  // ai: A transfer let go by its root (xfer_rx_core.h forget), the ids forgotten with it; done when it returns, as clear.
  void forget(const std::string& root);
  // ai: Waits until the worker has taken everything queued (tools and tests; the app never needs it).
  void drain();

 private:
  struct Item { uint8_t kind; std::array<uint8_t, XFER_BLOCK_BYTES> b; };   // ai: kind 0 a block, 1 a clear, 2 a forget (b the root, NUL-ended)
  void push(uint8_t kind, const uint8_t* b);
  void ask(uint8_t kind, const uint8_t* b);
  void run();

  std::function<void(const std::string&)> log_;
  std::mutex prod_;   // ai: the producers' side: take, frame, clear and forget, the judge with them, one at a time
  // ai: shared: the queue, the worker's progress snapshot
  std::mutex mu_;
  std::condition_variable cv_, idle_;
  std::deque<Item> queue_;
  bool stop_ = false, busy_ = false;
  uint64_t clearsAsked_ = 0, clearsDone_ = 0;   // ai: clear()'s tickets, the worker's count of clears done
  uint64_t lost_ = 0;
  std::mutex pmu_;
  XferProgress snap_;
  std::unique_ptr<XferRxCore> rx_;     // ai: the worker's
  std::unique_ptr<XferJudge> judge_;   // ai: the producer's: the dedupe (lizard-web/recv.mjs seenIds, seenOld)
  std::thread worker_;
};

}  // namespace lizard
