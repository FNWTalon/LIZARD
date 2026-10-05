// ai: XferRxCore and XferJudge (xfer_rx_core.h): the web's transfer chain ported, with no thread of its own (split out of
// ai: xfer_rx.cpp 2026-10-03, every line of the chain as it was). Rx is sim/xfer.mjs XferReceiver and
// ai: lizard-web/fountain-worker.mjs take; the judge is sim/phy.mjs blockJudge and lizard-web/recv.mjs take's dedupe. Each function
// ai: names the JS it ports; where the port differs, an ai: comment says so and why.
#include "xfer_rx_core.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>

extern "C" {
#include "shake.h"
#include "shim.h"   // ai: liblizard/zstd/shim.h: a chunk sent as a zstd frame decompressed (2026-10-05)
#include "xfer.h"
// ai: liblizard/wirehair/shim.cpp, the web's own ABI to Wirehair (the profile pinned there), compiled natively
int lizard_wh_init(void);
int lizard_wh_decoder_create(uint32_t message_bytes, uint32_t block_bytes, uint32_t seed_attempt, void** codec_out);
int lizard_wh_decode(void* codec, uint32_t block_id, const void* data, uint32_t bytes);
int lizard_wh_recover(void* codec, void* out, uint32_t capacity);
void lizard_wh_free(void* codec);
}

namespace lizard {

static_assert(XFER_BLOCK_BYTES == XFER_BLOCK, "src/xfer.h moved");
static constexpr uint32_t PAYLOAD = XFER_PAYLOAD, REC = XFER_BLOCK;
// ai: lizard-web/fountain-worker.mjs LIVE_BLOCKS, HOLD_BLOCKS; lizard-web/recv.mjs SEEN_KEEP
static constexpr int64_t LIVE_BLOCKS = 4096, NEVER = INT64_MAX / 2;
static constexpr size_t HOLD_BLOCKS = 1024, SEEN_KEEP = 1 << 16;
enum { COLLECTING = 0, HOT = 1, VERIFIED = 3 };   // ai: 2 was a chunk decoded before the manifest, which no longer happens (a chunk's block count waits on it)

static std::string hexOf(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s(2 * n, '0');
  for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
  return s;
}
static double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ai: sim/xfer.mjs Bits: a set of symbols grown to the highest one seen
struct Bits {
  std::vector<uint8_t> a = std::vector<uint8_t>(64);
  uint32_t n = 0;
  bool has(uint32_t i) const { return (i >> 3) < a.size() && ((a[i >> 3] >> (i & 7)) & 1); }
  void add(uint32_t i) {
    if ((i >> 3) >= a.size()) a.resize(std::max(2 * a.size(), size_t(i >> 3) + 1));
    if (!has(i)) { a[i >> 3] |= uint8_t(1 << (i & 7)); n++; }
  }
  void orWith(const Bits& o) { for (size_t i = 0; i < o.a.size(); i++) if (o.a[i]) for (int b = 0; b < 8; b++) if ((o.a[i] >> b) & 1) add(uint32_t(8 * i + b)); }
};

// ai: sim/fountain.mjs Decoder over the shim
struct Wh {
  void* h = nullptr;
  uint32_t bytes = 0, lastId = 0;
  ~Wh() { if (h) lizard_wh_free(h); }
  int create(uint32_t messageBytes, uint32_t seed) {
    bytes = messageBytes;
    lastId = (messageBytes + PAYLOAD - 1) / PAYLOAD - 1;
    return lizard_wh_decoder_create(messageBytes, PAYLOAD, seed, &h);
  }
  // ai: 1 once enough blocks are in, 0 for more, negative on Wirehair's refusal; the last source block travels padded
  // ai: and Wirehair wants its true length
  int add(uint32_t id, const uint8_t* p) { return lizard_wh_decode(h, id, p, id == lastId ? bytes - lastId * PAYLOAD : PAYLOAD); }
  std::vector<uint8_t> recover() {
    std::vector<uint8_t> o(bytes);
    const int n = lizard_wh_recover(h, o.data(), bytes);
    o.resize(n < 0 ? 0 : size_t(n));
    return o;
  }
};

struct XferRxCore::Rx {
  struct Ch { Bits have, ban; int state = COLLECTING; std::unique_ptr<Wh> dec; };
  struct Hdr {
    std::array<uint8_t, XFER_PAYLOAD> bytes;
    xfer_header_t h;
    std::string name, type, root;
    uint64_t size = 0;
    uint32_t mcount = 0, mhave = 0;
    // ai: len: each chunk's own bytes (the header's); sent, seed and K: its bytes as sent, its seed attempt and its
    // ai: blocks, from the header for one chunk and from the manifest for more (known once the list checks)
    std::vector<uint32_t> K, len, sent;
    std::vector<uint8_t> seed, mgot, list;
    bool listOk = false, known = false;
  };
  std::unique_ptr<Store> store;
  std::function<void(const std::string&)> log;
  std::atomic<uint64_t> gen{0};   // ai: headers of another root read (a judge's dedupe clears on a change)
  std::string lastRoot;   // ai: lizard-web/recv.mjs xferRoot: the root the dedupe was last cleared for
  std::unique_ptr<Hdr> hdr;
  // ai: ordered where the JS Map is in insertion order: only the order of solves within one block's take differs
  std::map<uint32_t, Ch> ch;
  std::map<uint32_t, std::array<uint8_t, XFER_PAYLOAD>> raw;
  uint32_t verified = 0, rejected = 0, manifestRejected = 0, refused = 0, over = 0;
  double ms = 0;
  bool done = false;
  // ai: the store could not keep this transfer (store.h: over the memory cap, a write refused): it ends here, never
  // ai: done, its blocks no longer taken; the store's error says why. A new header begins afresh.
  bool failed = false;
  void stop() { if (!failed) say("xfer: " + (hdr ? hdr->name : std::string("the transfer")) + " stopped: " + store->error); failed = true; }
  std::string path;
  // ai: the fountain worker's
  int64_t sinceHeader = NEVER;
  std::vector<std::pair<uint32_t, std::array<uint8_t, XFER_PAYLOAD>>> held;
  size_t heldAt = 0;
  uint64_t forwarded = 0, doneAt = 0;

  Rx(std::unique_ptr<Store> s, std::function<void(const std::string&)> l) : store(std::move(s)), log(std::move(l)) {
    store->log = log;
    static const int wh = lizard_wh_init();
    if (wh) say("xfer: wirehair init failed");
  }
  void say(const std::string& s) { if (log) log(s); }

  // ai: lizard-web/fountain-worker.mjs chunked (a config) and handle's clear: a new receiver, the held blocks let go, and on
  // ai: a clear the finished files too (refused kept, as the web's receiver keeps it)
  void clearAll() {
    reset();
    store->removeAll();
    lastRoot.clear();
    sinceHeader = NEVER; held.clear(); heldAt = 0; forwarded = 0; doneAt = 0;
  }

  // ai: lizard-web/fountain-worker.mjs take
  void block(const uint8_t* b) {
    forwarded++;
    const uint32_t id = xfer_id_get(b);
    const uint8_t* p = b + XFER_ID_BYTES;
    if (xfer_id_chunk(id) == XFER_CONTROL) {
      add(id, p);
      if (id == XFER_ID_HEADER && hdr) { sinceHeader = 0; if (!held.empty()) release(); }
    } else if (hdr && sinceHeader++ < LIVE_BLOCKS) {
      add(id, p);
    } else {
      std::pair<uint32_t, std::array<uint8_t, XFER_PAYLOAD>> r{id, {}};
      memcpy(r.second.data(), p, PAYLOAD);
      if (held.size() < HOLD_BLOCKS) held.push_back(r); else { held[heldAt] = r; heldAt = (heldAt + 1) % HOLD_BLOCKS; }
    }
    if (done && !doneAt) doneAt = forwarded;
  }
  // ai: the held blocks, oldest first, into the transfer whose header was just read
  void release() {
    auto ring = std::move(held);
    held.clear();
    const size_t at = ring.size() < HOLD_BLOCKS ? 0 : heldAt;
    heldAt = 0;
    for (size_t k = 0; k < ring.size(); k++) { auto& r = ring[(at + k) % ring.size()]; add(r.first, r.second.data()); }
  }

  // ai: XferReceiver.reset (refused is kept, as there)
  void reset() {
    ch.clear(); raw.clear(); hdr.reset();
    verified = rejected = manifestRejected = over = 0;
    ms = 0; done = false; failed = false; path.clear();
    store->clear();
  }

  Ch& state(uint32_t c) { return ch[c]; }

  // ai: XferReceiver.add
  void add(uint32_t id, const uint8_t* p) {
    const uint32_t c = xfer_id_chunk(id), sym = xfer_id_symbol(id);
    if (c == XFER_CONTROL) {
      if (sym == 0) return header(p);
      if (sym <= XFER_MAX_MANIFEST) return manifestBlock(sym - 1, p);
      return;   // ai: reserved: this version ignores it
    }
    if (done || failed || (hdr && c >= hdr->h.chunks)) return;
    Ch& s = state(c);
    if (s.state == VERIFIED || s.have.has(sym) || s.ban.has(sym)) return;
    s.have.add(sym);
    if (s.dec) {
      const int rc = s.dec->add(sym, p);
      if (rc < 0) say("xfer: wirehair decode " + std::to_string(rc) + " on chunk " + std::to_string(c));
      if (rc == 1) recovered(c);
      return;
    }
    if (!store->append(c, sym, p)) return stop();
    if (hdr && hdr->known && s.have.n >= hdr->K[c]) solve(c);   // ai: K known once the manifest is in (one chunk: from the header)
  }

  // ai: XferReceiver.header, the light's only: a different header is a new transfer, and everything held is let go
  void header(const uint8_t* p) {
    if (hdr && !memcmp(p, hdr->bytes.data(), PAYLOAD)) return;
    auto h = std::make_unique<Hdr>();
    if (xfer_header_parse(p, &h->h)) { refused++; return; }   // ai: not a header this receiver reads
    if (hdr) reset();
    memcpy(h->bytes.data(), p, PAYLOAD);
    h->name.assign((const char*)h->h.name, h->h.name_len);
    h->type.assign((const char*)h->h.type, h->h.type_len);
    h->root = hexOf(h->h.root, XFER_CV);
    h->size = uint64_t(1) << h->h.chunk_log2;
    const uint32_t chunks = h->h.chunks;
    h->mcount = uint32_t(xfer_manifest_blocks(chunks));
    h->mgot.assign(h->mcount, 0);
    h->list.assign(std::max<size_t>(1, size_t(XFER_CV) * chunks), 0);
    h->K.assign(chunks, 0); h->len.resize(chunks); h->sent.assign(chunks, 0); h->seed.assign(chunks, 0);
    for (uint32_t c = 0; c < chunks; c++) h->len[c] = uint32_t(xfer_chunk_len(&h->h, c));
    if (chunks == 1) { h->sent[0] = h->h.sent; h->seed[0] = h->h.seed; h->K[0] = xfer_blocks(h->h.sent); h->known = true; }
    // ai: lizard-web/recv.mjs: a header of another root clears the page's dedupe (the producer's, told by the generation)
    if (!lastRoot.empty() && lastRoot != h->root) gen.fetch_add(1);
    lastRoot = h->root;
    hdr = std::move(h);
    say("xfer: header " + hdr->name + ", " + std::to_string(hdr->h.length) + " B, " + std::to_string(chunks) + " chunks, root " + hdr->root);
    for (auto it = ch.begin(); it != ch.end();) {   // ai: blocks of another transfer
      if (it->first >= chunks) { store->drop(it->first); it = ch.erase(it); } else ++it;
    }
    if (!store->begin(hdr->h.length)) return stop();
    if (!chunks) {   // ai: an empty file: its root is BLAKE3 of nothing, which is all there is to check
      uint8_t e[XFER_CV];
      xfer_b3_hash(nullptr, 0, e);
      if (!memcmp(e, hdr->h.root, XFER_CV)) finish(); else refused++;
      return;
    }
    auto pending = std::move(raw);
    raw.clear();
    for (auto& [m, b] : pending) manifestBlock(m, b.data());
    if (hdr && hdr->known) solveReady();
  }
  // ai: XferReceiver.solveReady: every chunk with its blocks in hand, now that their counts are known
  void solveReady() {
    std::vector<uint32_t> ready;
    for (auto& [c, s] : ch) if (s.state == COLLECTING && s.have.n >= hdr->K[c]) ready.push_back(c);
    for (uint32_t c : ready) if (hdr && !failed && !done && ch[c].state == COLLECTING) solve(c);
  }

  // ai: XferReceiver.manifestBlock
  void manifestBlock(uint32_t m, const uint8_t* p) {
    if (!hdr) { if (raw.size() < 4096) memcpy(raw[m].data(), p, PAYLOAD); return; }   // ai: kept until there is a root to check it by
    Hdr& h = *hdr;
    if (failed || h.listOk || m >= h.mcount || h.mgot[m]) return;
    if (xfer_manifest_parse(p, &h.h, m, h.list.data(), h.sent.data(), h.seed.data())) return;   // ai: another transfer's (its tag), not a manifest block, or entries no chunk could have
    h.mgot[m] = 1;
    if (++h.mhave < h.mcount) return;
    if (xfer_manifest_check(h.list.data(), h.h.chunks, h.h.root)) {
      // ai: every block passed its parse and the list still does not make the root: one passed its CRC wrongly, which
      // ai: cannot be told, so all are collected again; the sender repeats them
      std::fill(h.mgot.begin(), h.mgot.end(), 0);
      h.mhave = 0; manifestRejected++;
      return;
    }
    h.listOk = true;
    for (uint32_t c = 0; c < h.h.chunks; c++) h.K[c] = xfer_blocks(h.sent[c]);
    h.known = true;
    solveReady();
  }

  // ai: XferReceiver.solve and solveOnce
  void solve(uint32_t c) {
    const double t = nowMs();
    solveOnce(c);
    ms += nowMs() - t;
  }
  void solveOnce(uint32_t c) {
    Hdr& h = *hdr;
    Ch& s = ch[c];
    auto recs = store->read(c);
    if (h.K[c] == 1) {
      std::vector<uint8_t> out;
      if (recs.size() >= REC) out.assign(recs.begin() + XFER_ID_BYTES, recs.begin() + XFER_ID_BYTES + h.sent[c]);
      store->drop(c);
      return unpack(c, out);
    }
    auto d = std::make_unique<Wh>();
    if (int rc = d->create(h.sent[c], h.seed[c]); rc < 0) {
      // ai: the JS throws here; a header that passed its CRC wrongly is the only way, and its chunk is collected afresh
      say("xfer: wirehair decoder " + std::to_string(rc) + " for chunk " + std::to_string(c));
      store->drop(c);
      return reject(c, s);
    }
    bool ok = false;
    for (size_t o = 0; o + REC <= recs.size() && !ok; o += REC) {
      const int rc = d->add(xfer_id_get(recs.data() + o), recs.data() + o + XFER_ID_BYTES);
      if (rc < 0) say("xfer: wirehair decode " + std::to_string(rc) + " on chunk " + std::to_string(c));
      ok = rc == 1;
    }
    store->drop(c);   // ai: the decoder holds them now
    s.dec = std::move(d);
    if (!ok) { s.state = HOT; over++; return; }   // ai: Wirehair wanted a block more than K (a few percent of chunks): fed as they come
    recovered(c);
  }
  void recovered(uint32_t c) {
    Ch& s = ch[c];
    auto out = s.dec->recover();
    s.dec.reset();
    unpack(c, out);
  }
  // ai: XferReceiver.unpack: the chunk's bytes from what was sent, its zstd frame decompressed where it was sent
  // ai: shorter than it is (a frame that does not give exactly the chunk's bytes is a wrong chunk, rejected as a
  // ai: wrong hash is), else as they came
  void unpack(uint32_t c, const std::vector<uint8_t>& raw) {
    Hdr& h = *hdr;
    if (h.sent[c] >= h.len[c]) return check(c, raw);
    std::vector<uint8_t> out(h.len[c]);
    if (lizard_zstd_decompress(raw.data(), uint32_t(raw.size()), out.data(), h.len[c])) return reject(c, ch[c]);
    check(c, out);
  }
  void check(uint32_t c, const std::vector<uint8_t>& out) {
    Hdr& h = *hdr;
    Ch& s = ch[c];
    if (!verify(c, out)) return reject(c, s);
    if (!store->writeOut(uint64_t(c) * h.size, out.data(), out.size())) return stop();
    accept(c, s);
  }
  // ai: xfer_chunk_check: the chunk's chaining value against the list (already checked against the root), or, for a
  // ai: file of one chunk, its hash against the root
  bool verify(uint32_t c, const std::vector<uint8_t>& out) {
    Hdr& h = *hdr;
    return xfer_chunk_check(&h.h, c, out.data(), out.size(), h.h.chunks >= 2 ? h.list.data() : nullptr) == 0;
  }
  void accept(uint32_t c, Ch& s) {
    s.state = VERIFIED; s.have = Bits(); s.ban = Bits(); verified++;
    if (verified == hdr->h.chunks) finish();
  }
  // ai: a chunk that decoded to the wrong bytes: which block passed its CRC wrongly cannot be told, so every symbol that
  // ai: went into it is refused from now on and the chunk collected again from fresh ids
  void reject(uint32_t c, Ch& s) {
    s.ban.orWith(s.have); s.have = Bits(); s.state = COLLECTING; rejected++;
    say("xfer: chunk " + std::to_string(c) + " rejected");
  }
  void finish() {
    path = store->finish(hdr->name);
    if (!store->error.empty()) { path.clear(); return stop(); }
    done = true;
    say("xfer: " + hdr->name + " verified, " + path);
  }

  // ai: XferReceiver.progress, fractionDone and the fountain worker's report
  void progress(XferProgress& p) const {
    p = XferProgress{};
    p.error = store->error;
    p.refused = refused; p.forwarded = forwarded; p.doneAt = doneAt; p.held = uint32_t(held.size());
    if (!hdr) return;
    const Hdr& h = *hdr;
    p.header = true; p.live = sinceHeader < LIVE_BLOCKS;
    p.name = h.name; p.type = h.type; p.root = h.root; p.length = h.h.length; p.chunks = h.h.chunks;
    p.verified = verified; p.manifest = h.mcount; p.manifestHave = h.mhave;
    p.rejected = rejected; p.manifestRejected = manifestRejected; p.over = over; p.solveMs = ms;
    p.done = done; p.failed = failed; p.path = path;
    p.per.resize(h.h.chunks);
    double got = 0;
    for (uint32_t c = 0; c < h.h.chunks; c++) {
      auto it = ch.find(c);
      const Ch* s = it == ch.end() ? nullptr : &it->second;
      // ai: until the manifest says a chunk's blocks, its share is over the blocks its bytes would take as they are, a floor
      const uint32_t k = h.known ? h.K[c] : xfer_blocks(h.len[c]);
      const uint8_t v = !s ? 0 : s->state == VERIFIED ? 255 : uint8_t(std::min<uint64_t>(99, 100ull * s->have.n / std::max<uint32_t>(1, k)));
      p.per[c] = v;
      got += h.len[c] * (v >= 254 ? 1.0 : v / 100.0);
      if (h.known) { p.sent += h.sent[c]; if (s) p.sentIn += s->state == VERIFIED ? h.sent[c] : std::min<uint64_t>(h.sent[c], uint64_t(s->have.n) * PAYLOAD); }
    }
    // ai: XferReceiver.progress and fractionDone: by the bytes as sent once the manifest has said them, the file's own until then
    p.fraction = done ? 1 : p.sent ? double(p.sentIn) / double(p.sent) : h.h.length ? got / double(h.h.length) : 0;
    p.bytesIn = uint64_t(p.fraction * double(h.h.length));
  }
};

XferRxCore::XferRxCore(std::unique_ptr<Store> store, std::function<void(const std::string&)> log)
    : rx_(std::make_unique<Rx>(std::move(store), std::move(log))) {}
XferRxCore::~XferRxCore() = default;
void XferRxCore::block(const uint8_t* b) { rx_->block(b); }
void XferRxCore::clearAll() { rx_->clearAll(); }
void XferRxCore::progress(XferProgress& p) const { rx_->progress(p); }
const std::atomic<uint64_t>& XferRxCore::rootGen() const { return rx_->gen; }
const std::vector<uint8_t>* XferRxCore::data() const { return rx_->done ? rx_->store->data() : nullptr; }

void XferJudge::remember(uint32_t id) {
  seenIds_.insert(id);
  if (seenIds_.size() >= SEEN_KEEP) { seenOld_ = std::move(seenIds_); seenIds_ = {}; }
}

// ai: lizard-web/recv.mjs take's dedupe, and its reset on a new root (the page's xfer message; here the core's generation)
bool XferJudge::take(const uint8_t* b, const Pass& pass) {
  if (const uint64_t g = gen_.load(); g != genSeen_) { genSeen_ = g; seenIds_.clear(); seenOld_.clear(); }
  const uint32_t id = xfer_id_get(b);
  if (xfer_id_chunk(id) == XFER_CONTROL) { pass(b); return false; }
  if (seenIds_.count(id) || seenOld_.count(id)) return false;
  remember(id);
  pass(b);
  return true;
}

// ai: sim/phy.mjs blockJudge, then lizard-web/recv.mjs take: a test frame's new ids are counted and remembered as the page
// ai: does, and none of its blocks passed on
FrameVerdict XferJudge::frame(const uint8_t* blocks, size_t count, const Pass& pass) {
  FrameVerdict v;
  std::vector<uint8_t> same(count, 0);
  uint8_t want[XFER_PAYLOAD];
  int data = 0;
  for (size_t k = 0; k < count; k++) {
    const uint8_t* b = blocks + k * XFER_BLOCK_BYTES;
    const uint32_t id = xfer_id_get(b);
    if (xfer_id_chunk(id) == XFER_CONTROL) continue;
    data++;
    stream_fill(id, want, XFER_PAYLOAD);
    same[k] = !memcmp(want, b + XFER_ID_BYTES, XFER_PAYLOAD);
    v.test |= same[k] != 0;
  }
  v.judged = v.test ? data : 0;
  // ai: blockJudge's got holds the frame's control blocks first, then its data blocks: the order the worker takes them in
  for (int pass2 = 0; pass2 < 2; pass2++)
    for (size_t k = 0; k < count; k++) {
      const uint8_t* b = blocks + k * XFER_BLOCK_BYTES;
      const uint32_t id = xfer_id_get(b);
      const bool control = xfer_id_chunk(id) == XFER_CONTROL;
      if (control != (pass2 == 0)) continue;
      if (v.test && !control && !same[k]) { v.bad++; continue; }
      v.seen++;
      if (!v.test) { v.fresh += take(b, pass); continue; }
      if (!control && !seenIds_.count(id) && !seenOld_.count(id)) { remember(id); v.fresh++; }
    }
  return v;
}

void XferJudge::clear() {
  seenIds_.clear(); seenOld_.clear();
  genSeen_ = gen_.load();
}

}  // namespace lizard
