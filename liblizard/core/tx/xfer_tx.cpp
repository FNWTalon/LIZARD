// ai: The transfer's sending end (xfer_tx.h), liblizard/sim/xfer.mjs XferSender and Schedule line for line.
#include "xfer_tx.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

extern "C" {
#include "xfer.h"
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
// ai: typeBytes: printable ASCII of at most 162, else none
std::string typeBytes(const std::string& t) {
  if (t.size() > XFER_TYPE_MAX) return "";
  for (char c : t) if (c < 0x20 || c > 0x7e) return "";
  return t;
}
}  // namespace

XferTx::XferTx(const uint8_t* data, size_t length, const std::string& name, const std::string& type, int chunkLog2)
    : data_(data), size_(size_t{1} << chunkLog2) {
  static std::once_flag wh;
  std::call_once(wh, [] { lizard_wh_init(); });
  xfer_header_t h{};
  if (xfer_header_init(&h, length, chunkLog2)) throw std::runtime_error("the file is too large for 2^" + std::to_string(chunkLog2) + " chunks");
  const uint32_t C = static_cast<uint32_t>((length + size_ - 1) / size_);
  chunks_ = C;
  for (uint32_t c = 0; c < C; c++) {
    len_.push_back(std::min(size_, length - c * size_));
    K_.push_back(static_cast<uint32_t>((len_[c] + PAYLOAD - 1) / PAYLOAD));
    void* e = nullptr;
    int seed = 0;
    if (K_[c] > 1) {
      seed = lizard_wh_encoder_create(data + c * size_, static_cast<uint32_t>(len_[c]), PAYLOAD, &e);
      if (seed < 0) throw std::runtime_error("wirehair encoder: " + std::to_string(seed));
    }
    enc_.push_back(e);
    if (c == 0) h.seed_full = C >= 2 ? static_cast<uint8_t>(seed) : 0;
    if (c == C - 1) h.seed_last = static_cast<uint8_t>(e ? seed : 0);
    // ai: the header carries one attempt for every chunk but the last (src/xfer.h): a full chunk seeded apart is a
    // ai: file it cannot describe
    if (C >= 2 && c > 0 && c < C - 1 && seed != h.seed_full)
      throw std::runtime_error("chunk " + std::to_string(c) + " took seed attempt " + std::to_string(seed));
  }
  std::vector<uint8_t> cvs;
  if (C >= 2) {
    cvs.resize(32 * size_t{C});
    for (uint32_t c = 0; c < C; c++)
      if (xfer_chunk_cv(data + c * size_, len_[c], c, chunkLog2, cvs.data() + 32 * c)) throw std::runtime_error("chaining value of chunk " + std::to_string(c));
    if (xfer_root(cvs.data(), C, root_)) throw std::runtime_error("root");
  } else xfer_b3_hash(data, length, root_);
  std::memcpy(h.root, root_, 32);
  const std::string nm = nameBytes(name), ty = typeBytes(type);
  h.name_len = static_cast<uint8_t>(nm.size()); std::memcpy(h.name, nm.data(), nm.size());
  h.type_len = static_cast<uint8_t>(ty.size()); std::memcpy(h.type, ty.data(), ty.size());
  if (xfer_header_write(&h, header_)) throw std::runtime_error("header refused");
  for (int m = 0; m < xfer_manifest_blocks(C); m++) {
    std::vector<uint8_t> p(PAYLOAD);
    if (xfer_manifest_write(cvs.data(), C, root_, static_cast<uint32_t>(m), p.data())) throw std::runtime_error("manifest block " + std::to_string(m));
    manifest_.push_back(std::move(p));
  }
  // ai: the schedule (Schedule's constructor): a round visits every full-size chunk once; a shorter last chunk in
  // ai: proportion to its blocks, plus 2 sqrt(K) visits a lap and at least 64
  sym_.assign(C, 0);
  full_ = C && K_[C - 1] < K_[0] ? C - 1 : C;
  w_ = full_ < C ? std::min(K_[0], std::max<uint32_t>(64, K_[C - 1] + static_cast<uint32_t>(std::ceil(2 * std::sqrt(static_cast<double>(K_[C - 1])))))) : 0;
  lap_ = C ? full_ * K_[0] + w_ : 0;
  // ai: the whole control cycle (the header every other block, the manifest between) within an eighth of a lap
  every_ = std::max<uint32_t>(1, std::min<uint32_t>(CONTROL_EVERY, lap_ / (8 * std::max<uint32_t>(1, static_cast<uint32_t>(manifest_.size())))));
}

XferTx::~XferTx() { for (void* e : enc_) if (e) lizard_wh_free(e); }

uint32_t XferTx::next() {
  uint32_t c;
  for (;;) {
    if (i_ < full_) { c = i_++; break; }
    i_ = 0;
    if (w_ && (acc_ += w_) >= K_[0]) { acc_ -= K_[0]; c = chunks_ - 1; break; }
  }
  if (K_[c] == 1) return idOf(c, 0);   // ai: unfountained: the one block, again
  const uint32_t s = sym_[c];
  sym_[c] = (s + 1) & SYMBOL_MASK;     // ai: past 2^18 ids a chunk the sender repeats (src/xfer.h)
  return idOf(c, s);
}

void XferTx::frameIds(uint32_t* out, int n) {
  if (!chunks_) { std::fill(out, out + n, ID_HEADER); frame_++; return; }   // ai: an empty file is its header
  int first = 0;
  if (since_ >= every_ && n > 1) {
    const uint32_t m = static_cast<uint32_t>(manifest_.size()), j = ctl_++;
    out[first++] = !m || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ((j >> 1) % m);
    since_ = 0;
  }
  for (int k = first; k < n; k++) out[k] = next();
  if (since_ != UINT32_MAX) since_ += static_cast<uint32_t>(n - first);
  // ai: Fisher-Yates over the data slots, from a generator seeded by the frame count (xorshift32)
  uint32_t s = ((frame_ + 1) * 0x9e3779b1u) ^ 0x5bd1e995u;
  for (int i = n - 1; i > first; i--) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const int j = first + static_cast<int>(s % static_cast<uint32_t>(i - first + 1));
    std::swap(out[i], out[j]);
  }
  frame_++;
}

void XferTx::block(uint32_t id, uint8_t* out) const {
  const uint32_t c = id >> SYMBOL_BITS, s = id & SYMBOL_MASK;
  if (c == CONTROL) {
    if (s == 0) std::memcpy(out, header_, PAYLOAD);
    else if (s - 1 < manifest_.size()) std::memcpy(out, manifest_[s - 1].data(), PAYLOAD);
    else std::memset(out, 0, PAYLOAD);
    return;
  }
  std::memset(out, 0, PAYLOAD);
  if (c >= chunks_) return;
  if (enc_[c]) {
    uint8_t buf[PAYLOAD];
    const int n = lizard_wh_encode(enc_[c], s, buf, PAYLOAD);
    if (n > 0) std::memcpy(out, buf, static_cast<size_t>(n));
    return;
  }
  std::memcpy(out, data_ + c * size_, len_[c]);   // ai: a chunk of one block, zero padded
}

std::string XferTx::rootHex() const {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (uint8_t b : root_) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

}  // namespace lizard
