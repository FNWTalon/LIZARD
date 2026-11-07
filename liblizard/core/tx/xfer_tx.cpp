// ai: The transfer's sending end (xfer_tx.h), liblizard/sim/xfer.mjs XferSender and Schedule line for line.
#include "xfer_tx.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <set>
#include <unistd.h>
#include <fstream>
#endif

extern "C" {
#include "xfer.h"
#include "shim.h"   // ai: liblizard/zstd/shim.h, the pinned zstd every build compresses with
int lizard_wh_init(void);
int lizard_wh_encoder_create(const void* message, uint32_t message_bytes, uint32_t block_bytes, void** codec_out);
int lizard_wh_encode(void* codec, uint32_t block_id, void* out, uint32_t capacity);
void lizard_wh_free(void* codec);
}

namespace lizard {

namespace {
constexpr uint32_t PAYLOAD = XFER_PAYLOAD, SYMBOL_BITS = XFER_SYMBOL_BITS, SYMBOL_MASK = (1u << SYMBOL_BITS) - 1;
constexpr uint32_t CONTROL = XFER_CONTROL, ID_HEADER = XFER_ID_HEADER, CONTROL_EVERY = 63;
uint32_t idOf(uint32_t c, uint32_t s) { return c << SYMBOL_BITS | s; }

// ai: sim/xfer.mjs nameBytes: UTF-8, cut to 255 bytes at a character's start
std::string nameBytes(const std::string& name) {
  size_t n = std::min<size_t>(name.size(), XFER_NAME_MAX);
  while (n < name.size() && n > 0 && (static_cast<uint8_t>(name[n]) & 0xc0) == 0x80) n--;
  return name.substr(0, n);
}
// ai: The prepare's workers: one a physical core, not a hardware thread (2026-11-06, 1 GiB on the 9950X, 16 cores of
// ai: 2 threads: 16 workers 0.46 s random and 1.09 s compressible, 32 workers 0.62 and 1.18 s at twice the CPU time;
// ai: two chunks' Wirehair solves on one core's caches go slower than one). The hardware threads where the system
// ai: does not say; one under emscripten (no threads in the wasm build).
unsigned workers() {
  unsigned n = 0;
#if defined(__EMSCRIPTEN__)
  n = 1;
#elif defined(_WIN32)
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  std::vector<uint8_t> b(bytes);
  if (bytes && GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(b.data()), &bytes))
    for (DWORD o = 0; o < bytes; o += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(b.data() + o)->Size) n++;
#elif defined(__APPLE__)
  int c = 0;
  size_t len = sizeof c;
  if (!sysctlbyname("hw.physicalcpu", &c, &len, nullptr, 0) && c > 0) n = static_cast<unsigned>(c);
#elif defined(__linux__)
  // ai: the online CPUs' distinct sibling sets (an offline CPU has no topology)
  std::set<std::string> cores;
  const long conf = sysconf(_SC_NPROCESSORS_CONF);
  for (long i = 0; i < conf; i++) {
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/thread_siblings_list");
    std::string list;
    if (f >> list) cores.insert(list);
  }
  n = static_cast<unsigned>(cores.size());
#endif
  return n ? n : std::max(1u, std::thread::hardware_concurrency());
}
// ai: typeBytes: printable ASCII of at most 158, else none
std::string typeBytes(const std::string& t) {
  if (t.size() > XFER_TYPE_MAX) return "";
  for (char c : t) if (c < 0x20 || c > 0x7e) return "";
  return t;
}
}  // namespace

XferTx::XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2)
    : size_(size_t{1} << chunkLog2), data_(data) {
  static std::once_flag wh;
  std::call_once(wh, [] { lizard_wh_init(); });
  xfer_header_t h{};
  if (xfer_header_init(&h, length, chunkLog2)) throw std::runtime_error("the file is too large for 2^" + std::to_string(chunkLog2) + " chunks");
  const uint32_t C = static_cast<uint32_t>((length + size_ - 1) / size_), M = static_cast<uint32_t>(xfer_manifest_blocks(C));
  chunks_ = C;
  h.codec = XFER_CODEC_ZSTD;
  len_.resize(C); sent_.resize(C); seed_.resize(C); K_.resize(C); enc_.resize(C); one_.resize(C); sym_.assign(C, 0);
  for (uint32_t c = 0; c < C; c++) len_[c] = std::min(size_, length - c * size_);
  ready_.reset(new std::atomic<uint8_t>[std::max<uint32_t>(C, 1)]());
  mready_.reset(new std::atomic<uint8_t>[std::max<uint32_t>(M, 1)]());
  left_.reset(new std::atomic<uint32_t>[std::max<uint32_t>(M, 1)]());
  for (uint32_t m = 0; m < M; m++) left_[m] = std::min<uint32_t>(XFER_PER_BLOCK, C - m * XFER_PER_BLOCK);
  manifest_.assign(M, std::vector<uint8_t>(PAYLOAD));
  const uint32_t W = std::min<uint32_t>(std::max<uint32_t>(C, 1), workers());
  if (C >= 2) {
    // ai: every chunk's chaining value first (its own bytes, BLAKE3's wide hash), a thread a physical core, then the
    // ai: root: all the header needs, so the first frame waits on a hash of the file and not on its compression
    cvs_.resize(32 * size_t{C});
    std::atomic<uint32_t> nc{0}, bad{UINT32_MAX};
    auto hash = [&] {
      for (uint32_t c; (c = nc.fetch_add(1, std::memory_order_relaxed)) < C;)
        if (xfer_chunk_cv(data + c * size_, len_[c], c, chunkLog2, cvs_.data() + 32 * c)) { uint32_t none = UINT32_MAX; bad.compare_exchange_strong(none, c); }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < W; w++) try { pool.emplace_back(hash); } catch (const std::system_error&) { break; }   // ai: a thread refused: fewer do the work
    hash();
    for (auto& t : pool) t.join();
    if (bad != UINT32_MAX) throw std::runtime_error("chaining value of chunk " + std::to_string(bad.load()));
    if (xfer_root(cvs_.data(), C, root_)) throw std::runtime_error("root");
  } else {
    if (C == 1) { std::vector<uint8_t> frame; make(0, frame); }   // ai: one chunk: its header carries its sent bytes and seed
    xfer_b3_hash(data, length, root_);
  }
  std::memcpy(h.root, root_, 32);
  if (C == 1) { h.sent = sent_[0]; h.seed = seed_[0]; }
  const std::string nm = nameBytes(name), ty = typeBytes(type);
  h.name_len = static_cast<uint8_t>(nm.size()); std::memcpy(h.name, nm.data(), nm.size());
  h.type_len = static_cast<uint8_t>(ty.size()); std::memcpy(h.type, ty.data(), ty.size());
  if (xfer_header_write(&h, header_)) throw std::runtime_error("header refused");
  if (C < 2) { schedule(); return; }   // ai: made already: its lap and sent bytes known at once
  // ai: the chunks made behind the constructor, a worker a physical core; on this thread where no other can start
  // ai: (emscripten has no threads), every chunk then before it returns
#ifndef __EMSCRIPTEN__
  for (uint32_t w = 0; w < W; w++) try { workers_.emplace_back([this] { work(); }); } catch (const std::system_error&) { break; }
#endif
  if (workers_.empty()) { work(); const std::string f = fault(); if (!f.empty()) throw std::runtime_error(f); schedule(); }
}

XferTx::~XferTx() {
  stop_ = true;
  for (auto& t : workers_) if (t.joinable()) t.join();
}

// ai: Chunk c: compressed on its own (2026-10-05) and sent as its frame where that is shorter, else as it is; a
// ai: fountain over the sent bytes where they are 2 blocks or more, else the one block kept (one_). Its slots written,
// ai: then ready_; the last of a manifest group writes the group's block, then mready_.
void XferTx::make(uint32_t c, std::vector<uint8_t>& frame) {
  const uint8_t* plain = data_ + c * size_;
  const uint8_t* out = plain;
  uint32_t n = static_cast<uint32_t>(len_[c]);
  frame.resize(lizard_zstd_bound(n));
  const int32_t z = lizard_zstd_compress(plain, n, frame.data(), static_cast<uint32_t>(frame.size()));
  if (z > 0 && static_cast<uint32_t>(z) < n) { out = frame.data(); n = static_cast<uint32_t>(z); }
  sent_[c] = n;
  K_[c] = xfer_blocks(n);
  if (K_[c] > 1) {
    void* e = nullptr;
    const int seed = lizard_wh_encoder_create(out, n, PAYLOAD, &e);
    if (seed < 0) throw std::runtime_error("wirehair encoder: " + std::to_string(seed));
    enc_[c].reset(e);
    seed_[c] = static_cast<uint8_t>(seed);
  } else one_[c].assign(out, out + n);
  ready_[c].store(1, std::memory_order_release);
  if (chunks_ >= 2) {
    const uint32_t m = c / XFER_PER_BLOCK;
    if (left_[m].fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (xfer_manifest_write(cvs_.data(), sent_.data(), seed_.data(), chunks_, root_, m, manifest_[m].data())) throw std::runtime_error("manifest block " + std::to_string(m));
      mready_[m].store(1, std::memory_order_release);
    }
  }
  made_.fetch_add(1, std::memory_order_acq_rel);
}

// ai: A worker: the next chunk until none is left; a failure is kept (the first) and stops every worker
void XferTx::work() {
  std::vector<uint8_t> frame;
  for (uint32_t c; !stop_.load(std::memory_order_relaxed) && (c = next_.fetch_add(1, std::memory_order_relaxed)) < chunks_;) {
    try { make(c, frame); } catch (const std::exception& e) {
      std::lock_guard<std::mutex> l(faultMu_);
      if (fault_.empty()) fault_ = "chunk " + std::to_string(c) + ": " + e.what();
      stop_ = true;
    }
  }
}

void XferTx::waitMade() {
  for (auto& t : workers_) if (t.joinable()) t.join();
  const std::string f = fault();
  if (!f.empty()) throw std::runtime_error(f);
}

std::string XferTx::fault() const {
  std::lock_guard<std::mutex> l(faultMu_);
  return fault_;
}

// ai: the schedule (Schedule's constructor), once every chunk is made: a round visits every chunk of the most blocks
// ai: once; a chunk of fewer in proportion to its blocks, plus 2 sqrt(K) visits a lap and at least 64. The symbols a
// ai: chunk sent while the others were made go on (sym_), so none is sent twice.
void XferTx::schedule() {
  scheduled_ = true;
  const uint32_t C = chunks_;
  uint64_t sent = 0;
  for (uint32_t c = 0; c < C; c++) sent += sent_[c];
  acc_.assign(C, 0);
  w_.clear();
  max_ = C ? *std::max_element(K_.begin(), K_.end()) : 0;
  uint32_t lap = 0;
  for (uint32_t c = 0; c < C; c++) {
    const uint32_t k = K_[c];
    w_.push_back(k >= max_ ? max_ : std::min(max_, std::max<uint32_t>(64, k + static_cast<uint32_t>(std::ceil(2 * std::sqrt(static_cast<double>(k)))))));
    lap += w_[c];
  }
  // ai: the whole control cycle (the header every other block, the manifest between) within an eighth of a lap
  every_ = std::max<uint32_t>(1, std::min<uint32_t>(CONTROL_EVERY, lap / (8 * std::max<uint32_t>(1, static_cast<uint32_t>(manifest_.size())))));
  i_ = 0; since_ = UINT32_MAX;
  sentBytes_ = sent; lap_ = lap;
}

// ai: A frame before every chunk is made: a control block (where it has more than one slot, or nothing is made yet, or
// ai: every other frame of one), the header every other time and the manifest blocks written so far between; the rest
// ai: the chunks made so far in turn, each its next symbol (a chunk of one block its one); with none made, every slot a
// ai: control block.
void XferTx::warm(uint32_t* out, int n) {
  std::vector<uint32_t> ready, ms;
  for (uint32_t c = 0; c < chunks_; c++) if (ready_[c].load(std::memory_order_acquire)) ready.push_back(c);
  for (uint32_t m = 0; m < manifest_.size(); m++) if (mready_[m].load(std::memory_order_acquire)) ms.push_back(m);
  auto ctl = [&] { const uint32_t j = ctl_++; return ms.empty() || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ms[(j >> 1) % ms.size()]; };
  int k = 0;
  if (n > 1 || ready.empty() || !(frame_ & 1)) out[k++] = ctl();
  for (; k < n; k++) {
    if (ready.empty()) { out[k] = ctl(); continue; }
    const uint32_t c = ready[turn_++ % ready.size()];
    if (K_[c] == 1) { out[k] = idOf(c, 0); continue; }
    out[k] = idOf(c, sym_[c]);
    sym_[c] = (sym_[c] + 1) & SYMBOL_MASK;
  }
}

void XferTx::WhFree::operator()(void* e) const { lizard_wh_free(e); }

uint32_t XferTx::next() {
  uint32_t c;
  for (;;) {
    if (i_ >= chunks_) i_ = 0;
    c = i_++;
    if ((acc_[c] += w_[c]) >= max_) { acc_[c] -= max_; break; }
  }
  if (K_[c] == 1) return idOf(c, 0);   // ai: unfountained: the one block, again
  const uint32_t s = sym_[c];
  sym_[c] = (s + 1) & SYMBOL_MASK;     // ai: past 2^18 ids a chunk the sender repeats (src/xfer.h)
  return idOf(c, s);
}

void XferTx::frameIds(uint32_t* out, int n) {
  if (!chunks_) { std::fill(out, out + n, ID_HEADER); frame_++; return; }   // ai: an empty file is its header
  if (!scheduled_ && made()) schedule();
  if (!scheduled_) warm(out, n);
  else {
    int first = 0;
    if (since_ >= every_ && n > 1) {
      const uint32_t m = static_cast<uint32_t>(manifest_.size()), j = ctl_++;
      out[first++] = !m || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ((j >> 1) % m);
      since_ = 0;
    }
    for (int k = first; k < n; k++) out[k] = next();
    if (since_ != UINT32_MAX) since_ += static_cast<uint32_t>(n - first);
  }
  // ai: Fisher-Yates over every slot, the control block's too, from a generator seeded by the frame count (xorshift32).
  // ai: Until 2026-10-10 the control block kept slot 0 and only the data slots were shuffled: slot 0 is block 0, on
  // ai: the innermost sub-channel, which since the rate profile carries the 7/8 code and decodes for some contents
  // ai: never (the header among them), so a file's header stopped arriving and its transfer starved (STATUS
  // ai: "A file's header pinned to slot 0").
  uint32_t s = ((frame_ + 1) * 0x9e3779b1u) ^ 0x5bd1e995u;
  for (int i = n - 1; i > 0; i--) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const int j = static_cast<int>(s % static_cast<uint32_t>(i + 1));
    std::swap(out[i], out[j]);
  }
  frame_++;
}

void XferTx::block(uint32_t id, uint8_t* out) const {
  const uint32_t c = id >> SYMBOL_BITS, s = id & SYMBOL_MASK;
  if (c == CONTROL) {
    if (s == 0) std::memcpy(out, header_, PAYLOAD);
    else if (s - 1 < manifest_.size() && mready_[s - 1].load(std::memory_order_acquire)) std::memcpy(out, manifest_[s - 1].data(), PAYLOAD);
    else std::memset(out, 0, PAYLOAD);
    return;
  }
  std::memset(out, 0, PAYLOAD);
  if (c >= chunks_ || !ready_[c].load(std::memory_order_acquire)) return;   // ai: frameIds names only chunks made
  if (enc_[c]) {
    uint8_t buf[PAYLOAD];
    const int n = lizard_wh_encode(enc_[c].get(), s, buf, PAYLOAD);
    if (n > 0) std::memcpy(out, buf, static_cast<size_t>(n));
    return;
  }
  std::memcpy(out, one_[c].data(), one_[c].size());   // ai: a chunk of one block, zero padded
}

std::string XferTx::rootHex() const {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : root_) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

}  // namespace lizard
