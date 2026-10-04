// ai: The transfer's sending end (2026-10-01): liblizard/sim/xfer.mjs XferSender ported name for name, over
// ai: liblizard/src/xfer.c (the chunks' chaining values, their root, the header and the manifest) and Wirehair
// ai: through the web's shim (liblizard/wirehair/shim.cpp), as the receiving end (rx/xfer_rx.h) takes them. A file
// ai: goes in 2^chunk_log2 chunks (4 MiB), a fountain each; the header every other control slot and the manifest
// ai: between, the control cycle within an eighth of a lap; data blocks by the schedule, shuffled in a frame by a
// ai: generator seeded by the frame count. The same ids and bytes as the JS sender for the same file
// ai: (tools/tx_check.cpp holds it to one).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lizard {

class XferTx {
 public:
  // ai: data: the file's bytes, alive as long as this (the app maps the file); name and type as the header carries them
  // ai: (a name cut to 255 bytes at a character, a type kept only where it is printable ASCII of at most 162).
  // ai: Throws std::runtime_error where the file cannot go (too many chunks, a chunk Wirehair seeded apart).
  XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2 = 22);
  ~XferTx();
  XferTx(const XferTx&) = delete;
  XferTx& operator=(const XferTx&) = delete;

  // ai: The ids of the next frame of n blocks (advances the schedule: a frame asked for twice is two frames).
  void frameIds(uint32_t* out, int n);
  // ai: What block `id` carries: 469 bytes into out.
  void block(uint32_t id, uint8_t* out) const;

  uint32_t chunks() const { return chunks_; }
  uint32_t lap() const { return lap_; }           // ai: data blocks a lap of the schedule
  const uint8_t* root() const { return root_; }   // ai: 32 bytes, b3sum's answer for the file
  std::string rootHex() const;

 private:
  uint32_t next();   // ai: the schedule's next data id (sim/xfer.mjs Schedule.next)

  const uint8_t* data_;
  size_t size_;
  uint32_t chunks_ = 0;
  std::vector<size_t> len_;
  std::vector<uint32_t> K_;      // ai: Wirehair source blocks a chunk (1: sent unfountained)
  std::vector<void*> enc_;       // ai: a chunk's encoder, null where K is 1
  uint8_t root_[32]{}, header_[469]{};
  std::vector<std::vector<uint8_t>> manifest_;
  // ai: the schedule (Schedule): symbols a chunk sent, the round's place, the last chunk's accumulator
  std::vector<uint32_t> sym_;
  uint32_t i_ = 0, acc_ = 0, full_ = 0, w_ = 0, lap_ = 0;
  // ai: the control cycle: how many data blocks between control slots, since the last, which control is next
  uint32_t every_ = 1, since_ = UINT32_MAX, ctl_ = 0, frame_ = 0;
};

}  // namespace lizard
