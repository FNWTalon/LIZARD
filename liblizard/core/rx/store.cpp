// ai: The receiving end's stores (store.h). FileStore is lizard-web/fountain-worker.mjs OpfsStore on plain files, as
// ai: xfer_rx.cpp held it until 2026-10-03, now on stdio alone (FILE*, 64-bit seeks, std::filesystem) so it builds on
// ai: Windows as on Linux and Android, and since 2026-10-22 a folder a transfer, kept across receivers (store.h);
// ai: MemoryStore is sim/xfer.mjs MemoryStore, one transfer at a time.
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

// ai: A folder a transfer under dir (store.h fileStore). Chunk files are kept open through a small LRU of FILE*s, since
// ai: a file of many chunks has every one in flight until the first lap ends and a phone's descriptors are few.
class FileStore final : public Store {
 public:
  explicit FileStore(std::string d) : dir(std::move(d)) {
    std::error_code e;
    fs::create_directories(pathOf(dir), e);
    if (e) { fail("store: " + dir + ": " + e.message()); return; }
    // ai: a parked transfer (a folder with its header) stays for its header to come again; anything else is nobody's
    for (auto& p : fs::directory_iterator(pathOf(dir), e))
      if (!p.is_directory() || !fs::exists(p.path() / "header")) { std::error_code r; fs::remove_all(p.path(), r); }
  }
  ~FileStore() override { closeAll(); }
  bool keeps() const override { return true; }

  // ai: the transfer in hand let go whole, its folder (a finished file not yet moved out of it too)
  void clear() override {
    const std::string was = cur;
    closeAll();
    if (!was.empty()) { std::error_code e; fs::remove_all(pathOf(was), e); }
  }
  // ai: the receiver's Clear: the transfer in hand and every finished file (the folders with no header); parked
  // ai: transfers stay, an app's list's to delete
  void removeAll() override {
    clear();
    error.clear();
    std::error_code e;
    for (auto& p : fs::directory_iterator(pathOf(dir), e))
      if (!fs::exists(p.path() / "header")) { std::error_code r; fs::remove_all(p.path(), r); }
  }
  void park() override { closeAll(); }
  void truncate(uint32_t c, uint32_t n) override {
    auto s = sizes.find(c);
    if (s == sizes.end() || s->second <= uint64_t(n) * REC) return;
    auto o = opened.find(c);
    if (o != opened.end()) { fclose(o->second.f); opened.erase(o); }
    std::error_code e;
    fs::resize_file(pathOf(chunkPath(c)), uint64_t(n) * REC, e);
    s->second = uint64_t(n) * REC;
  }
  void forget(const std::string& id) override {
    if (id.empty() || id.find('/') != std::string::npos || id.find('\\') != std::string::npos || id == "." || id == "..") return;
    if (cur == dir + "/" + id) return clear();
    std::error_code e;
    fs::remove_all(pathOf(dir + "/" + id), e);
  }

  bool open(const std::string& id, const uint8_t* header, uint64_t length, Before& before) override {
    before = Before{};
    error.clear();
    closeAll();
    cur = dir + "/" + id;
    std::error_code e;
    // ai: the folder resumed only where it holds this very header (a header is the transfer's whole description)
    const std::vector<uint8_t> had = slurp(cur + "/header");
    const bool same = header && had.size() == PAYLOAD && !memcmp(had.data(), header, PAYLOAD) && fs::exists(pathOf(cur + "/out"));
    if (!same) {
      fs::remove_all(pathOf(cur), e);
      fs::create_directories(pathOf(cur), e);
      if (e) { fail("store: " + cur + ": " + e.message()); return false; }
      out = openFile(cur + "/out", "w+b");
      // ai: the file at its full length (zeros), as ftruncate made it: a chunk is written at its own offset
      if (!out || (length && (!seek64(out, length - 1) || fputc(0, out) == EOF))) { fail("store: " + cur + "/out: " + strerror(errno)); return false; }
      // ai: the header last, so a folder with one holds the output it describes
      if (header && !writeAll(cur + "/header", header, PAYLOAD)) { fail("store: " + cur + "/header: " + strerror(errno)); return false; }
      return true;
    }
    out = openFile(cur + "/out", "r+b");
    if (!out) { fail("store: " + cur + "/out: " + strerror(errno)); return false; }
    before.found = true;
    const std::vector<uint8_t> m = slurp(cur + "/manifest");
    for (size_t o = 0; o + REC <= m.size(); o += REC) before.manifest.push_back({xfer_id_get(m.data() + o), std::vector<uint8_t>(m.begin() + o + 4, m.begin() + o + REC)});
    const std::vector<uint8_t> v = slurp(cur + "/verified");
    for (size_t o = 0; o + 4 <= v.size(); o += 4) before.verified.push_back(xfer_id_get(v.data() + o));
    for (auto& p : fs::directory_iterator(pathOf(cur), e)) {
      const std::string f = p.path().filename().string();
      if (f.size() < 2 || f[0] != 'c' || f.find_first_not_of("0123456789", 1) != std::string::npos) continue;
      const uint32_t c = uint32_t(std::stoul(f.substr(1)));
      const uint64_t size = p.file_size(e);
      // ai: a record cut short by a stop goes, so the next one appended lands on a record's boundary
      if (size % REC) fs::resize_file(p.path(), size - size % REC, e);
      sizes[c] = size - size % REC;
      before.chunks.push_back(c);
    }
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
    auto o = opened.find(c);
    if (o != opened.end()) fflush(o->second.f);
    std::vector<uint8_t> b(s->second);
    FILE* f = openFile(chunkPath(c), "rb");
    size_t got = f ? fread(b.data(), 1, b.size(), f) : 0;
    if (f) fclose(f);
    b.resize(got - got % REC);
    return b;
  }
  void drop(uint32_t c) override {
    auto o = opened.find(c);
    if (o != opened.end()) { fclose(o->second.f); opened.erase(o); }
    if (sizes.erase(c)) { std::error_code e; fs::remove(pathOf(chunkPath(c)), e); }
  }
  bool writeOut(uint64_t off, const uint8_t* p, size_t len) override {
    if (!out || !seek64(out, off) || fwrite(p, 1, len, out) != len) { fail("store: " + cur + "/out not written: " + strerror(errno)); return false; }
    return true;
  }
  void noteManifest(uint32_t m, const uint8_t* payload) override {
    uint8_t r[REC];
    xfer_id_put(r, m);
    memcpy(r + 4, payload, PAYLOAD);
    appendTo(cur + "/manifest", r, REC);
  }
  // ai: the output's bytes on the disk before the chunk is noted: a chunk noted and lost to a power cut would come back
  // ai: verified and wrong; a chunk on the disk and not noted is only collected again
  void noteVerified(uint32_t c) override {
    if (!out || !syncFile(out)) return;
    uint8_t r[4];
    xfer_id_put(r, c);
    appendTo(cur + "/verified", r, 4);
  }
  void noteMeta(const std::string& json) override {
    if (cur.empty()) return;
    const std::string tmp = cur + "/meta.json.tmp";
    std::error_code e;
    if (writeAll(tmp, json.data(), json.size())) fs::rename(pathOf(tmp), pathOf(cur + "/meta.json"), e);
  }
  // ai: OpfsStore.finish, and the file given its own name in a folder of its own, so the app shares it as sent; the
  // ai: transfer whole, nothing of it is kept to resume
  std::string finish(const std::string& name) override {
    if (!out) return "";
    const bool synced = syncFile(out);
    const bool closed = fclose(out) == 0;
    out = nullptr;
    const std::string from = cur + "/out";
    if (!synced || !closed) { fail("store: " + from + " not written to the end: " + strerror(errno)); return from; }
    std::string d = cur + "/done", path = d + "/" + safeName(name);
    std::error_code e;
    fs::create_directories(pathOf(d), e);
    fs::rename(pathOf(from), pathOf(path), e);
    if (e) { fail("store: " + path + ": " + e.message()); return from; }
    for (const char* f : {"header", "manifest", "verified", "meta.json"}) fs::remove(pathOf(cur + "/" + f), e);
    return path;
  }

 private:
  struct Open { FILE* f; uint64_t used; };
  std::string dir, cur;   // ai: cur: the transfer in hand's folder, "" none
  FILE* out = nullptr;
  uint64_t tick = 0;
  std::unordered_map<uint32_t, uint64_t> sizes;   // ai: chunks with a file, and its bytes
  std::unordered_map<uint32_t, Open> opened;
  static constexpr size_t OPEN_CAP = 32;
  std::string chunkPath(uint32_t c) const { return cur + "/c" + std::to_string(c); }
  void closeAll() {
    for (auto& [c, o] : opened) fclose(o.f);
    opened.clear();
    sizes.clear();
    if (out) { fclose(out); out = nullptr; }
    cur.clear();
  }
  static std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> b;
    FILE* f = openFile(path, "rb");
    if (!f) return b;
    uint8_t buf[1 << 16];
    for (size_t k; (k = fread(buf, 1, sizeof buf, f)) > 0;) b.insert(b.end(), buf, buf + k);
    fclose(f);
    return b;
  }
  static bool writeAll(const std::string& path, const void* p, size_t len) {
    FILE* f = openFile(path, "wb");
    if (!f) return false;
    const bool ok = fwrite(p, 1, len, f) == len;
    return fclose(f) == 0 && ok;
  }
  static void appendTo(const std::string& path, const void* p, size_t len) {
    FILE* f = openFile(path, "ab");
    if (!f) return;
    fwrite(p, 1, len, f);
    fclose(f);
  }
  FILE* handle(uint32_t c) {
    auto o = opened.find(c);
    if (o != opened.end()) { o->second.used = ++tick; return o->second.f; }
    if (opened.size() >= OPEN_CAP) {
      auto old = std::min_element(opened.begin(), opened.end(), [](auto& a, auto& b) { return a.second.used < b.second.used; });
      fclose(old->second.f);
      opened.erase(old);
    }
    // ai: OpfsStore.handle truncates a chunk's file when it first opens it; a descriptor the LRU let go, or a chunk
    // ai: resumed, reopens to append
    const bool fresh = !sizes.count(c);
    FILE* f = openFile(chunkPath(c), fresh ? "wb" : "ab");
    if (!f) return nullptr;
    if (fresh) sizes[c] = 0;
    opened[c] = {f, ++tick};
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
  bool open(const std::string&, const uint8_t*, uint64_t length, Before& before) override {
    before = Before{};
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
