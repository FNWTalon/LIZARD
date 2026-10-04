// ai: Layer 3, the transfer (lizard.h): the sending end (core/tx/xfer_tx.h: the file's chunks, header, manifest and
// ai: schedule, or the test stream as the senders make it) and the receiving end with no thread (core/rx/
// ai: xfer_rx_core.h: the judge, the dedupe, the fountain and the BLAKE3 checks, in a memory or directory store).
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "api.hpp"
#include "xfer_rx_core.h"
#include "xfer_tx.h"

extern "C" {
#include "shake.h"
#include "xfer.h"
}

struct liz_tx {
  std::vector<uint8_t> owned;
  std::unique_ptr<lizard::XferTx> x;   // ai: null for the test stream
  uint32_t next = 0;
  uint64_t length = 0;
  std::vector<uint32_t> ids;
};

struct liz_rx {
  std::unique_ptr<lizard::XferRxCore> core;
  std::unique_ptr<lizard::XferJudge> judge;
  lizard::XferProgress p;
  std::string meta;
};

using namespace liz;

static liz_tx* fileTx(std::vector<uint8_t> owned, const uint8_t* data, size_t length, const char* name, const char* type) {
  static const uint8_t empty = 0;
  auto t = std::make_unique<liz_tx>();
  t->owned = std::move(owned);
  if (!t->owned.empty()) data = t->owned.data();
  try {
    t->x = std::make_unique<lizard::XferTx>(length ? data : &empty, length, name ? name : "", type ? type : "");
  } catch (const std::runtime_error& e) {
    throw Error(LIZ_E_TOOBIG, e.what());   // ai: XferTx refuses a file only for its size (chunks, Wirehair's seeds)
  }
  t->length = length;
  return t.release();
}

extern "C" {

LIZ_API liz_tx* liz_tx_new(const uint8_t* data, size_t length, const char* name, const char* type, int flags) {
  return guardNew<liz_tx>([&] {
    need(data || !length, "the file's bytes");
    std::vector<uint8_t> owned;
    if (flags & LIZ_TX_COPY) owned.assign(data, data + length);
    return fileTx(std::move(owned), data, length, name, type);
  });
}

LIZ_API liz_tx* liz_tx_new_path(const char* path, const char* name, const char* type) {
  return guardNew<liz_tx>([&] {
    need(path, "a path");
    const std::filesystem::path p = std::filesystem::u8path(path);
    std::ifstream f(p, std::ios::binary);
    if (!f) throw Error(LIZ_E_IO, std::string("cannot read ") + path);
    std::vector<uint8_t> owned((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) throw Error(LIZ_E_IO, std::string("cannot read ") + path);
    const std::string base = p.filename().u8string();
    const size_t length = owned.size();
    return fileTx(std::move(owned), nullptr, length, name ? name : base.c_str(), type);
  });
}

LIZ_API liz_tx* liz_tx_new_test(uint32_t first) {
  return guardNew<liz_tx>([&] {
    auto t = std::make_unique<liz_tx>();
    // ai: the senders' rule: a random first id well below the control chunk's, odd
    if (!first) { std::random_device rd; first = (rd() & 0x3fffffffu) | 1u; }
    t->next = first;
    return t.release();
  });
}

LIZ_API void liz_tx_free(liz_tx* t) { delete t; }

LIZ_API int liz_tx_next(liz_tx* t, int n, uint8_t* blocks) {
  return guard([&] {
    need(t && blocks && n >= 1, "a sender, at least a block and a place for them");
    t->ids.resize(size_t(n));
    if (t->x) t->x->frameIds(t->ids.data(), n);
    else for (int i = 0; i < n; i++) t->ids[i] = t->next++;
    int data = 0;
    for (int i = 0; i < n; i++) {
      uint8_t* o = blocks + size_t(i) * LIZ_BLOCK;
      xfer_id_put(o, t->ids[i]);
      if (t->x) t->x->block(t->ids[i], o + LIZ_ID_BYTES);
      else stream_fill(t->ids[i], o + LIZ_ID_BYTES, LIZ_PAYLOAD);
      data += xfer_id_chunk(t->ids[i]) != XFER_CONTROL;
    }
    return data;
  });
}

LIZ_API int liz_tx_info_get(const liz_tx* t, liz_tx_info* out) {
  return guard([&] {
    need(t && out, "a sender and a place for its info");
    liz_tx_info i{};
    i.test = !t->x;
    if (t->x) {
      i.length = t->length; i.chunks = t->x->chunks(); i.lap = t->x->lap();
      std::memcpy(i.root, t->x->root(), sizeof i.root);
    }
    *out = i;
    return LIZ_OK;
  });
}

LIZ_API liz_rx* liz_rx_new(const liz_store* store) {
  return guardNew<liz_rx>([&] {
    auto r = std::make_unique<liz_rx>();
    std::unique_ptr<lizard::Store> s;
    // ai: a directory's own subfolder, as the app's receiver keeps it: only that folder is emptied
    if (store && store->dir) s = lizard::fileStore(std::string(store->dir) + "/lizard-xfer");
    else s = lizard::memoryStore(store ? store->max_bytes : 0);
    if (!s->error.empty()) throw Error(LIZ_E_IO, s->error);
    r->core = std::make_unique<lizard::XferRxCore>(std::move(s));
    r->judge = std::make_unique<lizard::XferJudge>(r->core->rootGen());
    return r.release();
  });
}

LIZ_API void liz_rx_free(liz_rx* r) { delete r; }

LIZ_API int liz_rx_frame(liz_rx* r, const uint8_t* verified, int count, liz_verdict* out) {
  return guard([&] {
    need(r && count >= 0 && (verified || !count), "a receiver and the frame's blocks");
    const lizard::FrameVerdict v = r->judge->frame(verified, size_t(count), [r](const uint8_t* b) { r->core->block(b); });
    if (out) *out = liz_verdict{v.seen, v.bad, v.judged, v.fresh, v.test};
    return LIZ_OK;
  });
}

LIZ_API int liz_rx_progress(liz_rx* r, liz_progress* out) {
  return guard([&] {
    need(r && out, "a receiver and a place for its progress");
    r->core->progress(r->p);
    const auto& p = r->p;
    liz_progress o{};
    o.header = p.header; o.done = p.done; o.length = p.length; o.bytes_in = p.bytesIn; o.fraction = p.fraction;
    o.chunks = p.chunks; o.verified = p.verified; o.rejected = p.rejected; o.solve_ms = p.solveMs;
    *out = o;
    return LIZ_OK;
  });
}

LIZ_API int liz_rx_chunks(liz_rx* r, uint8_t* per, int cap) {
  return guard([&] {
    need(r && (per || cap <= 0), "a receiver and a place for its chunks");
    r->core->progress(r->p);
    const int n = int(r->p.per.size());
    if (cap > 0) std::memcpy(per, r->p.per.data(), size_t(std::min(n, cap)));
    return n;
  });
}

LIZ_API const char* liz_rx_meta(liz_rx* r, int which) {
  const char* s = "";
  guard([&] {
    need(r, "a receiver");
    r->core->progress(r->p);
    const auto& p = r->p;
    switch (which) {
      case LIZ_META_NAME: r->meta = p.name; break;
      case LIZ_META_TYPE: r->meta = p.type; break;
      case LIZ_META_ROOT: r->meta = p.root; break;
      case LIZ_META_PATH: r->meta = p.path; break;
      case LIZ_META_ERROR: r->meta = p.error; break;
      default: need(false, "which is a LIZ_META_* value");
    }
    s = r->meta.c_str();
    return LIZ_OK;
  });
  return s;
}

LIZ_API int liz_rx_data(liz_rx* r, const uint8_t** data, size_t* length) {
  return guard([&] {
    need(r && data && length, "a receiver and places for the file");
    const std::vector<uint8_t>* d = r->core->data();
    if (!d) throw Error(LIZ_E_STATE, "no file yet, or the store is a directory's");
    *data = d->data();
    *length = d->size();
    return LIZ_OK;
  });
}

LIZ_API void liz_rx_clear(liz_rx* r) {
  guard([&] {
    need(r, "a receiver");
    r->judge->clear();
    r->core->clearAll();
    return LIZ_OK;
  });
}

}  // extern "C"
