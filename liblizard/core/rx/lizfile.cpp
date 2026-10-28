// ai: .lizard (lizfile.h): the frame codec, the ZIP writer and reader.
#include "lizfile.h"

#include "json.hpp"
#include "zstd.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>

namespace lizard::lizfile {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr int LEVEL = 1;   // ai: zstd's level: 1 packs as small as 3 on these frames (74.4% both), in under half the time
constexpr char COMMENT[] = "LIZARD replay 1";
constexpr uint32_t LOCAL = 0x04034b50, CENTRAL = 0x02014b50, END = 0x06054b50;
constexpr uint16_t DOS_DATE = (0 << 9) | (1 << 5) | 1;   // ai: every entry 1980-01-01 00:00, so a run packs to the same bytes

// ai: ZIP's CRC-32 (the IEEE polynomial, reflected), eight bytes a step
struct Crc {
  uint32_t t[8][256];
  Crc() {
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1)));
      t[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; i++)
      for (int s = 1; s < 8; s++) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xff];
  }
  uint32_t of(const uint8_t* p, size_t n) const {
    uint32_t c = 0xffffffffu;
    for (; n >= 8; p += 8, n -= 8) {
      const uint32_t a = c ^ (uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24);
      const uint32_t b = uint32_t(p[4]) | uint32_t(p[5]) << 8 | uint32_t(p[6]) << 16 | uint32_t(p[7]) << 24;
      c = t[7][a & 0xff] ^ t[6][(a >> 8) & 0xff] ^ t[5][(a >> 16) & 0xff] ^ t[4][a >> 24] ^
          t[3][b & 0xff] ^ t[2][(b >> 8) & 0xff] ^ t[1][(b >> 16) & 0xff] ^ t[0][b >> 24];
    }
    while (n--) c = (c >> 8) ^ t[0][(c ^ *p++) & 0xff];
    return ~c;
  }
};
const Crc& crc() { static const Crc c; return c; }

// ai: JPEG-LS's median predictor (lizfile.h)
inline int med(int a, int b, int c) {
  const int mx = a > b ? a : b, mn = a < b ? a : b;
  return c >= mx ? mn : c <= mn ? mx : a + b - c;
}
void residuals(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, uint8_t* r) {
  std::vector<uint8_t> zero(w, 0);
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* cur = px + size_t(y) * stride;
    const uint8_t* up = y ? cur - stride : zero.data();
    uint8_t* o = r + size_t(y) * w;
    o[0] = uint8_t(cur[0] - med(0, up[0], 0));
    for (uint32_t i = 1; i < w; i++) o[i] = uint8_t(cur[i] - med(cur[i - 1], up[i], up[i - 1]));
  }
}
void restore(const uint8_t* r, uint32_t w, uint32_t h, uint8_t* px) {
  std::vector<uint8_t> zero(w, 0);
  for (uint32_t y = 0; y < h; y++) {
    uint8_t* cur = px + size_t(y) * w;
    const uint8_t* up = y ? cur - w : zero.data();
    const uint8_t* in = r + size_t(y) * w;
    cur[0] = uint8_t(in[0] + med(0, up[0], 0));
    for (uint32_t i = 1; i < w; i++) cur[i] = uint8_t(in[i] + med(cur[i - 1], up[i], up[i - 1]));
  }
}

// ai: into z (resized to the packed bytes), r the residuals' buffer; false where zstd refused
bool packInto(ZSTD_CCtx* cc, const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::vector<uint8_t>& r, std::vector<uint8_t>& z) {
  const size_t n = size_t(w) * h;
  r.resize(n);
  z.resize(ZSTD_compressBound(n));
  residuals(px, w, h, stride, r.data());
  const size_t m = ZSTD_compressCCtx(cc, z.data(), z.size(), r.data(), n, LEVEL);
  if (ZSTD_isError(m)) { z.clear(); return false; }
  z.resize(m);
  return true;
}
std::vector<uint8_t> packWith(ZSTD_CCtx* cc, const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride) {
  std::vector<uint8_t> r, z;
  packInto(cc, px, w, h, stride, r, z);
  return z;
}
bool unpackInto(ZSTD_DCtx* dc, const uint8_t* z, size_t n, uint32_t w, uint32_t h, std::vector<uint8_t>& r, uint8_t* out, std::string& why) {
  const size_t want = size_t(w) * h;
  r.resize(want);
  const size_t got = ZSTD_decompressDCtx(dc, r.data(), want, z, n);
  if (ZSTD_isError(got)) { why = std::string("zstd: ") + ZSTD_getErrorName(got); return false; }
  if (got != want) { why = "a frame of " + std::to_string(got) + " B where " + std::to_string(w) + " x " + std::to_string(h) + " is " + std::to_string(want); return false; }
  restore(r.data(), w, h, out);
  return true;
}
bool unpackWith(ZSTD_DCtx* dc, const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why) {
  std::vector<uint8_t> r;
  return unpackInto(dc, z, n, w, h, r, out, why);
}

std::FILE* openFile(const std::string& path, const char* mode) {
#if defined(_WIN32)
  std::wstring m(mode, mode + std::strlen(mode));
  return _wfopen(fs::u8path(path).wstring().c_str(), m.c_str());
#else
  return std::fopen(path.c_str(), mode);
#endif
}
bool seekTo(std::FILE* f, uint64_t at) {
#if defined(_WIN32)
  return _fseeki64(f, int64_t(at), SEEK_SET) == 0;
#else
  return fseeko(f, off_t(at), SEEK_SET) == 0;
#endif
}
int64_t sizeOf(std::FILE* f) {
#if defined(_WIN32)
  if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
  return _ftelli64(f);
#else
  if (fseeko(f, 0, SEEK_END) != 0) return -1;
  return int64_t(ftello(f));
#endif
}
bool readFile(const fs::path& p, std::string& out) {
  std::ifstream i(p, std::ios::binary);
  if (!i) return false;
  out.assign(std::istreambuf_iterator<char>(i), std::istreambuf_iterator<char>());
  return !i.bad();
}
std::string frameName(size_t i, const char* ext) {
  char b[32];
  std::snprintf(b, sizeof b, "%04zu.%s", i, ext);
  return b;
}
void le16(std::string& b, uint32_t v) { b += char(v & 0xff); b += char((v >> 8) & 0xff); }
void le32(std::string& b, uint32_t v) { le16(b, v & 0xffff); le16(b, v >> 16); }
uint32_t rd16(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8; }
uint32_t rd32(const uint8_t* p) { return rd16(p) | rd16(p + 2) << 16; }

// ai: The archive as it is written: each entry's local header and bytes, then the central directory and its end. A
// ai: write the file system refuses puts the file back at the last whole entry, which the next write overwrites.
class ZipOut {
 public:
  explicit ZipOut(std::FILE* f) : f_(f) {}
  bool entry(const std::string& name, const void* data, size_t n, std::string& why) {
    if (cd_.size() >= 0xffff) { why = "over ZIP's 65,535 entries (a .lizard has no ZIP64)"; return false; }
    if (at_ + 30 + name.size() + n > 0xffffffffull) { why = "over ZIP's 4 GB (a .lizard has no ZIP64)"; return false; }
    const uint32_t c = crc().of(static_cast<const uint8_t*>(data), n);
    std::string h;
    le32(h, LOCAL); le16(h, 10); le16(h, 0); le16(h, 0); le16(h, 0); le16(h, DOS_DATE);
    le32(h, c); le32(h, uint32_t(n)); le32(h, uint32_t(n)); le16(h, uint32_t(name.size())); le16(h, 0);
    h += name;
    const uint64_t at = at_;
    if (!put(h.data(), h.size(), why) || !put(data, n, why)) { back(at); return false; }
    cd_.push_back({name, c, uint32_t(n), uint32_t(at)});
    return true;
  }
  bool finish(std::string& why) {
    const uint64_t start = at_;
    std::string d;
    for (const Cd& e : cd_) {
      le32(d, CENTRAL); le16(d, (3 << 8) | 20); le16(d, 10); le16(d, 0); le16(d, 0); le16(d, 0); le16(d, DOS_DATE);
      le32(d, e.crc); le32(d, e.size); le32(d, e.size); le16(d, uint32_t(e.name.size())); le16(d, 0); le16(d, 0);
      le16(d, 0); le16(d, 0); le32(d, 0100644u << 16); le32(d, e.at);
      d += e.name;
    }
    if (start + d.size() > 0xffffffffull) { why = "over ZIP's 4 GB (a .lizard has no ZIP64)"; return false; }
    std::string end;
    le32(end, END); le16(end, 0); le16(end, 0); le16(end, uint32_t(cd_.size())); le16(end, uint32_t(cd_.size()));
    le32(end, uint32_t(d.size())); le32(end, uint32_t(start)); le16(end, uint32_t(sizeof COMMENT - 1));
    end += COMMENT;
    if (!put(d.data(), d.size(), why) || !put(end.data(), end.size(), why)) { back(start); return false; }
    return true;
  }
  uint64_t bytes() const { return at_; }

 private:
  struct Cd { std::string name; uint32_t crc, size, at; };
  std::FILE* f_;
  uint64_t at_ = 0;
  std::vector<Cd> cd_;
  bool put(const void* p, size_t n, std::string& why) {
    if (n && std::fwrite(p, 1, n, f_) != n) { why = std::string("write: ") + std::strerror(errno); return false; }
    at_ += n;
    return true;
  }
  void back(uint64_t at) { std::clearerr(f_); seekTo(f_, at); at_ = at; }
};

// ai: the frames' sizes and count from a run's meta.json (replay.cpp finish: frames, sizes, w and h)
bool framesOf(const json& m, std::vector<std::pair<uint32_t, uint32_t>>& sizes, std::string& why) {
  const json* s = m.contains("sizes") && m["sizes"].is_array() ? &m["sizes"] : nullptr;
  const size_t n = m.value("frames", s ? s->size() : size_t(0));
  const uint32_t w = m.value("w", 0u), h = m.value("h", 0u);
  sizes.clear();
  for (size_t i = 0; i < n; i++) {
    uint32_t fw = w, fh = h;
    if (s && i < s->size() && (*s)[i].is_array() && (*s)[i].size() == 2) { fw = (*s)[i][0].get<uint32_t>(); fh = (*s)[i][1].get<uint32_t>(); }
    if (!fw || !fh || fw > 16384 || fh > 16384) { why = "meta.json gives frame " + std::to_string(i) + " no size"; return false; }
    sizes.emplace_back(fw, fh);
  }
  return true;
}

}  // namespace

std::vector<uint8_t> packFrame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride) {
  ZSTD_CCtx* cc = ZSTD_createCCtx();
  if (!cc) return {};
  std::vector<uint8_t> z = packWith(cc, px, w, h, stride);
  ZSTD_freeCCtx(cc);
  return z;
}
bool unpackFrame(const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why) {
  ZSTD_DCtx* dc = ZSTD_createDCtx();
  if (!dc) { why = "zstd: no context"; return false; }
  const bool ok = unpackWith(dc, z, n, w, h, out, why);
  ZSTD_freeDCtx(dc);
  return ok;
}

FrameCodec::FrameCodec() : cctx_(ZSTD_createCCtx()), dctx_(ZSTD_createDCtx()) {}
FrameCodec::~FrameCodec() {
  ZSTD_freeCCtx(static_cast<ZSTD_CCtx*>(cctx_));
  ZSTD_freeDCtx(static_cast<ZSTD_DCtx*>(dctx_));
}
bool FrameCodec::pack(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::vector<uint8_t>& out) {
  return cctx_ && packInto(static_cast<ZSTD_CCtx*>(cctx_), px, w, h, stride, res_, out);
}
bool FrameCodec::unpack(const uint8_t* z, size_t n, uint32_t w, uint32_t h, uint8_t* out, std::string& why) {
  if (!dctx_) { why = "zstd: no context"; return false; }
  return unpackInto(static_cast<ZSTD_DCtx*>(dctx_), z, n, w, h, res_, out, why);
}

std::unique_ptr<Writer> Writer::create(const std::string& path, std::string& why) {
  std::unique_ptr<Writer> w(new Writer());
  w->path_ = path;
  w->f_ = openFile(path + ".part", "wb");
  if (!w->f_) { why = "open " + path + ".part: " + std::strerror(errno); return nullptr; }
  w->cctx_ = ZSTD_createCCtx();
  if (!w->cctx_) { why = "zstd: no context"; return nullptr; }
  w->zip_ = new ZipOut(w->f_);
  return w;
}

Writer::~Writer() {
  delete static_cast<ZipOut*>(zip_);
  if (cctx_) ZSTD_freeCCtx(static_cast<ZSTD_CCtx*>(cctx_));
  if (f_) {
    std::fclose(f_);
    std::error_code ec;
    if (!done_) fs::remove(fs::u8path(path_ + ".part"), ec);
  }
}

bool Writer::frame(const uint8_t* px, uint32_t w, uint32_t h, uint32_t stride, std::string& why) {
  if (done_) { why = "the file is finished"; return false; }
  const std::vector<uint8_t> z = packWith(static_cast<ZSTD_CCtx*>(cctx_), px, w, h, stride);
  if (z.empty()) { why = "zstd refused frame " + std::to_string(frames_); return false; }
  if (!static_cast<ZipOut*>(zip_)->entry(frameName(frames_, "zst"), z.data(), z.size(), why)) return false;
  frames_++;
  return true;
}

int64_t Writer::finish(const std::string& meta, const std::string& stats, std::string& why) {
  if (done_) { why = "the file is finished"; return -1; }
  ZipOut& z = *static_cast<ZipOut*>(zip_);
  if (!stats.empty() && !z.entry("stats.jsonl", stats.data(), stats.size(), why)) return -1;
  if (!z.entry("meta.json", meta.data(), meta.size(), why) || !z.finish(why)) return -1;
  const int64_t n = int64_t(z.bytes());
  const int closed = std::fclose(f_);
  f_ = nullptr;
  std::error_code ec;
  const fs::path part = fs::u8path(path_ + ".part"), to = fs::u8path(path_);
  if (closed != 0) { why = std::string("close: ") + std::strerror(errno); fs::remove(part, ec); return -1; }
  fs::resize_file(part, uint64_t(n), ec);   // ai: past a write refused before the last whole entry
  fs::remove(to, ec);   // ai: Windows' rename does not replace
  fs::rename(part, to, ec);
  if (ec) { why = "rename " + path_ + ".part: " + ec.message(); fs::remove(part, ec); return -1; }
  done_ = true;
  return n;
}

bool unpack(const std::string& path, const std::string& dir, std::string& why) {
  auto r = Reader::open(path, why);
  if (!r) return false;
  std::error_code ec;
  const fs::path d = fs::u8path(dir);
  if (fs::exists(d, ec) && !fs::is_empty(d, ec)) { why = dir + " holds files already"; return false; }
  fs::create_directories(d, ec);
  if (ec) { why = "the folder: " + ec.message(); return false; }
  auto put = [&](const std::string& name, const void* p, size_t n) {
    std::ofstream o(d / name, std::ios::binary | std::ios::trunc);
    o.write(static_cast<const char*>(p), std::streamsize(n));
    if (!o) { why = "write " + name + ": " + std::strerror(errno); return false; }
    return true;
  };
  if (!put("meta.json", r->meta().data(), r->meta().size())) return false;
  if (!r->stats().empty() && !put("stats.jsonl", r->stats().data(), r->stats().size())) return false;
  std::vector<uint8_t> luma;
  for (size_t i = 0; i < r->frames().size(); i++)
    if (!r->frame(i, luma, why) || !put(frameName(i, "gray"), luma.data(), luma.size())) return false;
  return true;
}

int64_t pack(const std::string& dir, const std::string& path, std::string& why) {
  try {
    const fs::path d = fs::u8path(dir);
    std::string meta, stats, px;
    if (!readFile(d / "meta.json", meta)) { why = dir + " holds no meta.json"; return -1; }
    const json m = json::parse(meta, nullptr, false);
    if (m.is_discarded() || !m.is_object()) { why = "its meta.json does not read"; return -1; }
    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    if (!framesOf(m, sizes, why)) return -1;
    std::error_code ec;
    if (fs::is_regular_file(d / "stats.jsonl", ec)) readFile(d / "stats.jsonl", stats);
    auto w = Writer::create(path, why);
    if (!w) return -1;
    for (size_t i = 0; i < sizes.size(); i++) {
      const auto [fw, fh] = sizes[i];
      if (!readFile(d / frameName(i, "gray"), px)) { why = dir + " holds no " + frameName(i, "gray"); return -1; }
      if (px.size() != size_t(fw) * fh) { why = frameName(i, "gray") + " holds " + std::to_string(px.size()) + " B, not " + std::to_string(fw) + " x " + std::to_string(fh); return -1; }
      if (!w->frame(reinterpret_cast<const uint8_t*>(px.data()), fw, fh, fw, why)) return -1;
    }
    return w->finish(meta, stats, why);
  } catch (const std::exception& x) { why = std::string("its meta.json: ") + x.what(); return -1; }
}

Reader::~Reader() {
  if (f_) std::fclose(f_);
  if (dctx_) ZSTD_freeDCtx(static_cast<ZSTD_DCtx*>(dctx_));
}

bool Reader::parseMeta(std::string& why) {
  try {
    const json m = json::parse(meta_, nullptr, false);
    if (m.is_discarded() || !m.is_object()) { why = "its meta.json does not read"; return false; }
    std::vector<std::pair<uint32_t, uint32_t>> sizes;
    if (!framesOf(m, sizes, why)) return false;
    const json* ms = m.contains("ms") && m["ms"].is_array() ? &m["ms"] : nullptr;
    const json* ax = m.contains("axes") && m["axes"].is_array() ? &m["axes"] : nullptr;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    frames_.clear();
    for (size_t i = 0; i < sizes.size(); i++) {
      FrameInfo fi;
      fi.w = sizes[i].first;
      fi.h = sizes[i].second;
      fi.ms = ms && i < ms->size() && (*ms)[i].is_number() ? (*ms)[i].get<double>() : nan;
      fi.ax = fi.ay = float(nan);
      if (ax && i < ax->size() && (*ax)[i].is_array() && (*ax)[i].size() == 2 && (*ax)[i][0].is_number() && (*ax)[i][1].is_number()) {
        fi.ax = (*ax)[i][0].get<float>();
        fi.ay = (*ax)[i][1].get<float>();
      }
      frames_.push_back(fi);
    }
    layout_ = m.value("layout", "") == "2:1" ? "2:1" : "1:1";
    return true;
  } catch (const std::exception& x) { why = std::string("its meta.json: ") + x.what(); return false; }
}

std::unique_ptr<Reader> Reader::open(const std::string& path, std::string& why) {
  std::unique_ptr<Reader> r(new Reader());
  r->path_ = path;
  r->f_ = openFile(path, "rb");
  if (!r->f_) { why = "open " + path + ": " + std::strerror(errno); return nullptr; }
  const int64_t size = sizeOf(r->f_);
  // ai: the end of the central directory: its 22 bytes and the comment, in the file's last 64 KB
  const uint64_t tail = uint64_t(std::min<int64_t>(size, 22 + 0xffff));
  std::vector<uint8_t> t(tail);
  if (size < 22 || !seekTo(r->f_, uint64_t(size) - tail) || std::fread(t.data(), 1, tail, r->f_) != tail) { why = path + " is not a ZIP archive"; return nullptr; }
  size_t e = std::string::npos;
  for (size_t i = tail - 22 + 1; i-- > 0;)
    if (rd32(&t[i]) == END && i + 22 + rd16(&t[i + 20]) == tail) { e = i; break; }
  if (e == std::string::npos) { why = path + " is not a ZIP archive"; return nullptr; }
  const std::string comment(reinterpret_cast<const char*>(&t[e + 22]), rd16(&t[e + 20]));
  if (comment.rfind("LIZARD replay ", 0) != 0) { why = path + " is not a .lizard"; return nullptr; }
  if (comment != COMMENT) { why = path + " is a " + comment + ", newer than this reader's " + COMMENT; return nullptr; }
  const uint32_t count = rd16(&t[e + 10]), cdSize = rd32(&t[e + 12]), cdAt = rd32(&t[e + 16]);
  std::vector<uint8_t> cd(cdSize);
  if (!seekTo(r->f_, cdAt) || std::fread(cd.data(), 1, cdSize, r->f_) != cdSize) { why = path + ": its central directory is cut short"; return nullptr; }
  struct Named { std::string name; Entry e; };
  std::vector<Named> all;
  for (size_t at = 0, k = 0; k < count; k++) {
    if (at + 46 > cd.size() || rd32(&cd[at]) != CENTRAL) { why = path + ": its central directory does not read"; return nullptr; }
    const uint32_t method = rd16(&cd[at + 10]), c = rd32(&cd[at + 16]), csize = rd32(&cd[at + 20]), usize = rd32(&cd[at + 24]);
    const uint32_t nl = rd16(&cd[at + 28]), el = rd16(&cd[at + 30]), cl = rd16(&cd[at + 32]), local = rd32(&cd[at + 42]);
    if (at + 46 + nl > cd.size()) { why = path + ": its central directory does not read"; return nullptr; }
    Named n{std::string(reinterpret_cast<const char*>(&cd[at + 46]), nl), {}};
    if (method != 0 || csize != usize) { why = path + ": " + n.name + " is compressed by the ZIP (a .lizard stores its entries)"; return nullptr; }
    uint8_t lh[30];
    if (!seekTo(r->f_, local) || std::fread(lh, 1, 30, r->f_) != 30 || rd32(lh) != LOCAL) { why = path + ": " + n.name + "'s header does not read"; return nullptr; }
    n.e = {uint64_t(local) + 30 + rd16(lh + 26) + rd16(lh + 28), csize, c};
    all.push_back(std::move(n));
    at += 46 + nl + el + cl;
  }
  auto find = [&](const std::string& name) -> const Entry* {
    for (const Named& n : all) if (n.name == name) return &n.e;
    return nullptr;
  };
  std::vector<uint8_t> b;
  const Entry* me = find("meta.json");
  if (!me || !r->readEntry(*me, b, why)) { if (why.empty()) why = path + " holds no meta.json"; return nullptr; }
  r->meta_.assign(b.begin(), b.end());
  if (const Entry* se = find("stats.jsonl")) {
    if (!r->readEntry(*se, b, why)) return nullptr;
    r->stats_.assign(b.begin(), b.end());
  }
  if (!r->parseMeta(why)) return nullptr;
  for (size_t i = 0; i < r->frames_.size(); i++) {
    const Entry* fe = find(frameName(i, "zst"));
    if (!fe) { why = path + " holds no " + frameName(i, "zst"); return nullptr; }
    r->entries_.push_back(*fe);
  }
  return r;
}

bool Reader::readEntry(const Entry& e, std::vector<uint8_t>& out, std::string& why) {
  out.resize(size_t(e.size));
  if (!seekTo(f_, e.at) || std::fread(out.data(), 1, out.size(), f_) != out.size()) { why = path_ + " is cut short"; return false; }
  if (crc().of(out.data(), out.size()) != e.crc) { why = path_ + ": an entry's CRC does not hold"; return false; }
  return true;
}

bool Reader::frame(size_t i, std::vector<uint8_t>& luma, std::string& why) {
  if (i >= frames_.size()) { why = "no frame " + std::to_string(i); return false; }
  const FrameInfo& fi = frames_[i];
  const size_t n = size_t(fi.w) * fi.h;
  if (!readEntry(entries_[i], buf_, why)) return false;
  if (!dctx_) dctx_ = ZSTD_createDCtx();
  if (!dctx_) { why = "zstd: no context"; return false; }
  luma.resize(n);
  return unpackWith(static_cast<ZSTD_DCtx*>(dctx_), buf_.data(), buf_.size(), fi.w, fi.h, luma.data(), why);
}

}  // namespace lizard::lizfile
