// ai: The receiving end's stores (store.h). FileStore is lizard-web/fountain-worker.mjs OpfsStore on plain files, as
// ai: xfer_rx.cpp held it until 2026-10-03, now on stdio alone (FILE*, 64-bit seeks, std::filesystem) so it builds on
// ai: Windows as on Linux and Android; MemoryStore is sim/xfer.mjs MemoryStore.
#include "store.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <unordered_map>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

extern "C" {
#include "xfer.h"
}

namespace fs = std::filesystem;

namespace lizard {

static constexpr uint32_t PAYLOAD = XFER_PAYLOAD, REC = XFER_BLOCK;

// ai: paths are UTF-8 everywhere; Windows' narrow stdio would read them in the ANSI code page
static fs::path pathOf(const std::string& utf8) { return fs::u8path(utf8); }
static FILE* openFile(const std::string& utf8, const char* mode) {
#ifdef _WIN32
  std::wstring m(mode, mode + strlen(mode));
  return _wfopen(pathOf(utf8).c_str(), m.c_str());
#else
  return fopen(utf8.c_str(), mode);
#endif
}
static bool seek64(FILE* f, uint64_t off) {
#ifdef _WIN32
  return _fseeki64(f, int64_t(off), SEEK_SET) == 0;
#else
  return fseeko(f, off_t(off), SEEK_SET) == 0;
#endif
}
// ai: false where the bytes may not have reached the disk (a write deferred until now that failed)
static bool syncFile(FILE* f) {
  if (fflush(f) != 0) return false;
#ifdef _WIN32
  return _commit(_fileno(f)) == 0;
#else
  return fsync(fileno(f)) == 0 || errno == EINVAL || errno == EROFS;
#endif
}

// ai: the header's name as one path component: no separators, no control bytes, never empty, . or ..; on Windows also
// ai: none of <>:"|?* nor a trailing dot or space, and no device name (CON, NUL, COM1, ...) as its stem
static std::string safeName(const std::string& s) {
  std::string r = s;
  for (char& ch : r) if (ch == '/' || ch == '\\' || (unsigned char)ch < 0x20 || ch == 0x7f) ch = '_';
#ifdef _WIN32
  for (char& ch : r) if (strchr("<>:\"|?*", ch)) ch = '_';
  while (!r.empty() && (r.back() == '.' || r.back() == ' ')) r.back() = '_';
  std::string stem = r.substr(0, r.find('.'));
  for (char& ch : stem) ch = char(toupper((unsigned char)ch));
  static const char* dev[] = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
                              "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};
  for (const char* d : dev) if (stem == d) { r = "_" + r; break; }
#endif
  return r.empty() || r == "." || r == ".." ? "file" : r;
}

// ai: Chunk files are kept open through a small LRU of FILE*s, since a file of many chunks has every one in flight
// ai: until the first lap ends and a phone's descriptors are few.
class FileStore final : public Store {
 public:
  explicit FileStore(std::string d) : dir(std::move(d)) {
    std::error_code e;
    fs::remove_all(pathOf(dir), e);
    fs::create_directories(pathOf(dir), e);
    if (e) fail("store: " + dir + ": " + e.message());
  }
  ~FileStore() override { for (auto& [c, o] : open) fclose(o.f); if (out) fclose(out); }
  void clear() override {
    std::vector<uint32_t> cs;
    for (auto& [c, n] : sizes) cs.push_back(c);
    for (uint32_t c : cs) drop(c);
    if (out) { fclose(out); out = nullptr; std::error_code e; fs::remove(pathOf(outPath), e); }
  }
  void removeAll() override {
    clear();
    error.clear();
    std::error_code e;
    for (auto& p : fs::directory_iterator(pathOf(dir), e)) fs::remove_all(p.path(), e);
  }
  bool begin(uint64_t length) override {
    error.clear();
    if (out) fclose(out);
    outPath = dir + "/out" + std::to_string(n++);
    out = openFile(outPath, "w+b");
    // ai: the file at its full length (zeros), as ftruncate made it: a chunk is written at its own offset
    if (!out || (length && (!seek64(out, length - 1) || fputc(0, out) == EOF))) { fail("store: " + outPath + ": " + strerror(errno)); return false; }
    return true;
  }
  bool append(uint32_t c, uint32_t sym, const uint8_t* payload) override {
    FILE* f = handle(c);
    uint8_t r[REC];
    xfer_id_put(r, sym);
    memcpy(r + 4, payload, PAYLOAD);
    if (!f || fwrite(r, 1, REC, f) != REC) { fail("store: a block of chunk " + std::to_string(c) + " not written"); return false; }
    sizes[c] += REC;
    return true;
  }
  std::vector<uint8_t> read(uint32_t c) override {
    auto s = sizes.find(c);
    if (s == sizes.end()) return {};
    auto o = open.find(c);
    if (o != open.end()) fflush(o->second.f);
    std::vector<uint8_t> b(s->second);
    FILE* f = openFile(chunkPath(c), "rb");
    size_t got = f ? fread(b.data(), 1, b.size(), f) : 0;
    if (f) fclose(f);
    b.resize(got - got % REC);
    return b;
  }
  void drop(uint32_t c) override {
    auto o = open.find(c);
    if (o != open.end()) { fclose(o->second.f); open.erase(o); }
    if (sizes.erase(c)) { std::error_code e; fs::remove(pathOf(chunkPath(c)), e); }
  }
  bool writeOut(uint64_t off, const uint8_t* p, size_t len) override {
    if (!out || !seek64(out, off) || fwrite(p, 1, len, out) != len) { fail("store: " + outPath + " not written: " + strerror(errno)); return false; }
    return true;
  }
  std::vector<uint8_t> readOut(uint64_t off, size_t len) override {
    std::vector<uint8_t> b(len);
    if (!out) return {};
    fflush(out);
    size_t got = seek64(out, off) ? fread(b.data(), 1, len, out) : 0;
    b.resize(got);
    return b;
  }
  // ai: OpfsStore.finish, and the file given its own name in a directory of its own, so the app shares it as sent
  std::string finish(const std::string& name) override {
    if (!out) return outPath;
    const bool synced = syncFile(out);
    const bool closed = fclose(out) == 0;
    out = nullptr;
    if (!synced || !closed) { fail("store: " + outPath + " not written to the end: " + strerror(errno)); return outPath; }
    std::string d = dir + "/done" + std::to_string(n - 1), path = d + "/" + safeName(name);
    std::error_code e;
    fs::create_directories(pathOf(d), e);
    fs::rename(pathOf(outPath), pathOf(path), e);
    if (e) { fail("store: " + path + ": " + e.message()); return outPath; }
    return path;
  }

 private:
  struct Open { FILE* f; uint64_t used; };
  std::string dir, outPath;
  FILE* out = nullptr;
  int n = 0;
  uint64_t tick = 0;
  std::unordered_map<uint32_t, uint64_t> sizes;   // ai: chunks with a file, and its bytes
  std::unordered_map<uint32_t, Open> open;
  static constexpr size_t OPEN_MAX = 32;
  std::string chunkPath(uint32_t c) const { return dir + "/c" + std::to_string(c); }
  FILE* handle(uint32_t c) {
    auto o = open.find(c);
    if (o != open.end()) { o->second.used = ++tick; return o->second.f; }
    if (open.size() >= OPEN_MAX) {
      auto old = std::min_element(open.begin(), open.end(), [](auto& a, auto& b) { return a.second.used < b.second.used; });
      fclose(old->second.f);
      open.erase(old);
    }
    // ai: OpfsStore.handle truncates a chunk's file when it first opens it; a descriptor the LRU let go reopens to append
    const bool fresh = !sizes.count(c);
    FILE* f = openFile(chunkPath(c), fresh ? "wb" : "ab");
    if (!f) return nullptr;
    if (fresh) sizes[c] = 0;
    open[c] = {f, ++tick};
    return f;
  }
};

class MemoryStore final : public Store {
 public:
  explicit MemoryStore(uint64_t cap) : cap(cap) {}
  // ai: a new transfer's: the finished file goes too (XferRxCore::data no longer reaches it, and it counted against the
  // ai: cap; the web's MemoryStore.clear lets its output go)
  void clear() override { chunks.clear(); held = 0; out.clear(); out.shrink_to_fit(); done.clear(); done.shrink_to_fit(); }
  void removeAll() override { clear(); error.clear(); }
  bool begin(uint64_t length) override {
    error.clear();
    out.clear();
    if (length > uint64_t(SIZE_MAX) || !fits(length)) {
      fail("store: a file of " + std::to_string(length) + " B is over the memory store's " + (length > uint64_t(SIZE_MAX) ? std::string("address space") : std::to_string(cap) + " B"));
      return false;
    }
    try { out.assign(size_t(length), 0); }
    catch (const std::bad_alloc&) { fail("store: no memory for a file of " + std::to_string(length) + " B"); return false; }
    return true;
  }
  bool append(uint32_t c, uint32_t sym, const uint8_t* payload) override {
    if (!fits(out.size() + REC)) { fail("store: the memory store is full (" + std::to_string(cap) + " B)"); return false; }
    auto& v = chunks[c];
    const size_t at = v.size();
    v.resize(at + REC);
    xfer_id_put(v.data() + at, sym);
    memcpy(v.data() + at + 4, payload, PAYLOAD);
    held += REC;
    return true;
  }
  std::vector<uint8_t> read(uint32_t c) override {
    auto it = chunks.find(c);
    return it == chunks.end() ? std::vector<uint8_t>{} : it->second;
  }
  void drop(uint32_t c) override {
    auto it = chunks.find(c);
    if (it == chunks.end()) return;
    held -= it->second.size();
    chunks.erase(it);
  }
  bool writeOut(uint64_t off, const uint8_t* p, size_t len) override {
    if (off + len > out.size()) { fail("store: a chunk past the file's end"); return false; }
    memcpy(out.data() + off, p, len);
    return true;
  }
  std::vector<uint8_t> readOut(uint64_t off, size_t len) override {
    if (off >= out.size()) return {};
    len = std::min<size_t>(len, size_t(out.size() - off));
    return std::vector<uint8_t>(out.begin() + off, out.begin() + off + len);
  }
  std::string finish(const std::string&) override { done = std::move(out); out.clear(); return ""; }
  const std::vector<uint8_t>* data() const override { return &done; }

 private:
  uint64_t cap, held = 0;
  std::unordered_map<uint32_t, std::vector<uint8_t>> chunks;
  std::vector<uint8_t> out, done;
  // ai: the stored blocks, the output and a finished file all count (extra is what is about to be added)
  bool fits(uint64_t extra) const { return !cap || held + extra + done.size() <= cap; }
};

std::unique_ptr<Store> fileStore(const std::string& dir) { return std::make_unique<FileStore>(dir); }
std::unique_ptr<Store> memoryStore(uint64_t maxBytes) { return std::make_unique<MemoryStore>(maxBytes); }

}  // namespace lizard
