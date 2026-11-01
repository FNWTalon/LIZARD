// ai: jpeg.h over TurboJPEG's v2 API (2026-10-25), declared here from turbojpeg.h (build machines may lack the
// ai: header; the library's ABI is stable across 2.x and 3.x) and looked up at run time, as portal.cpp does sd-bus.
// ai: `unsigned long` is turbojpeg.h's own type for sizes (32 bits on Windows, 64 elsewhere), kept so on every system.
#include "jpeg.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lizard::cam {

namespace {

using tjhandle = void*;
// ai: turbojpeg.h's values: TJPF_RGB, TJPF_GRAY, TJSAMP_422, TJERR_WARNING
constexpr int kRgb = 0, kGray = 6, kSamp422 = 1, kWarning = 0;
// ai: the largest frame decoded, against a corrupt header's size: 16384 a side, 2^26 pixels (8K is 2^25)
constexpr int kSide = 16384;
constexpr size_t kPixels = size_t(1) << 26;

}  // namespace

struct Jpeg::Api {
  tjhandle (*tjInitDecompress)() = nullptr;
  int (*tjDecompressHeader3)(tjhandle, const unsigned char*, unsigned long, int*, int*, int*, int*) = nullptr;
  int (*tjDecompress2)(tjhandle, const unsigned char*, unsigned long, unsigned char*, int, int, int, int,
                       int) = nullptr;
  tjhandle (*tjInitCompress)() = nullptr;
  int (*tjCompress2)(tjhandle, const unsigned char*, int, int, int, int, unsigned char**, unsigned long*, int, int,
                     int) = nullptr;
  void (*tjFree)(unsigned char*) = nullptr;
  int (*tjDestroy)(tjhandle) = nullptr;
  char* (*tjGetErrorStr2)(tjhandle) = nullptr;
  int (*tjGetErrorCode)(tjhandle) = nullptr;
  bool ok = false;
  std::string why;   // ai: where not ok: each library tried and what the system said
};

namespace {

#if defined(_WIN32)
using Path = std::wstring;   // ai: a library's path as the system takes it

std::string utf8(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n > 0 ? n : 0), '\0');
  if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), &s[0], n, nullptr, nullptr);
  return s;
}

std::string lastError() {
  const DWORD e = GetLastError();
  char* msg = nullptr;
  FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e,
                 0, reinterpret_cast<char*>(&msg), 0, nullptr);
  std::string s = msg ? msg : "";
  if (msg) LocalFree(msg);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
  return "error " + std::to_string(e) + (s.empty() ? "" : ": " + s);
}
#else
using Path = std::string;
#endif

// ai: one library's calls, or why not: `path` alone where given (LIZ_TURBOJPEG), else the system's names in turn
std::unique_ptr<Jpeg::Api> openApi(const Path& path) {
  auto a = std::make_unique<Jpeg::Api>();
  std::vector<Path> names;
  if (!path.empty()) names = {path};
#if defined(_WIN32)
  else names = {L"turbojpeg.dll", L"libturbojpeg.dll"};
#elif defined(__APPLE__)
  else names = {"/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.0.dylib",
                "/usr/local/opt/jpeg-turbo/lib/libturbojpeg.0.dylib", "libturbojpeg.0.dylib"};
#else
  else names = {"libturbojpeg.so.0"};
#endif
  std::string tried;
  for (const auto& name : names) {
#if defined(_WIN32)
    const std::string shown = utf8(name);
    HMODULE h = LoadLibraryW(name.c_str());
    if (!h) { tried += (tried.empty() ? "" : "; ") + shown + ": " + lastError(); continue; }
    auto sym = [&](const char* s) { return reinterpret_cast<void*>(GetProcAddress(h, s)); };
#else
    const std::string& shown = name;
    void* h = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      const char* e = dlerror();
      tried += (tried.empty() ? "" : "; ") + (e ? std::string(e) : name + ": not loaded");
      continue;
    }
    auto sym = [&](const char* s) { return dlsym(h, s); };
#endif
    std::string missing;
    auto get = [&](auto& fn, const char* s) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym(s));
      if (!fn) missing += (missing.empty() ? "" : ", ") + std::string(s);
    };
    get(a->tjInitDecompress, "tjInitDecompress");
    get(a->tjDecompressHeader3, "tjDecompressHeader3");
    get(a->tjDecompress2, "tjDecompress2");
    get(a->tjInitCompress, "tjInitCompress");
    get(a->tjCompress2, "tjCompress2");
    get(a->tjFree, "tjFree");
    get(a->tjDestroy, "tjDestroy");
    get(a->tjGetErrorStr2, "tjGetErrorStr2");
    get(a->tjGetErrorCode, "tjGetErrorCode");
    if (missing.empty()) { a->ok = true; return a; }
    // ai: tjGetErrorCode and tjGetErrorStr2 came with libjpeg-turbo 2.0
    tried += (tried.empty() ? "" : "; ") + shown + " lacks " + missing + " (libjpeg-turbo 2.0 or later is needed)";
#if defined(_WIN32)
    FreeLibrary(h);
#else
    dlclose(h);
#endif
  }
  a->why = "TurboJPEG is not loaded: " + tried;
  return a;
}

// ai: Whether a frame's own bytes hold a whole header: SOI at its start, then a frame header (an SOFn segment) and
// ai: the first scan's SOS segment, every segment before the scan inside the frame; "" where they do, else why. The
// ai: walk is libjpeg's marker reader's (bytes other than 0xFF before a marker skipped, 0xFF fill, FF00 no marker;
// ai: SOI, EOI, RSTn and TEM stand alone, every other marker has a length), so a frame libjpeg reads is never refused
// ai: here. TurboJPEG cannot be asked this (2026-10-25): with libjpeg-turbo 3 (3.0.4, 3.2.0) a frame cut before its
// ai: SOF reads as a stream of tables alone, tjDecompressHeader3 succeeds with the size of the last header its handle
// ai: read, and tjDecompress2's error comes back as a warning, so the previous frame's picture was taken as this one's.
std::string headerWhole(const uint8_t* d, size_t n) {
  if (n < 2 || d[0] != 0xFF || d[1] != 0xD8) return "no SOI at its start";
  bool sof = false;
  size_t i = 2;
  for (;;) {
    for (;;) {
      while (i < n && d[i] != 0xFF) i++;
      while (i < n && d[i] == 0xFF) i++;
      if (i >= n) return "cut before its scan";
      if (d[i] != 0) break;
      i++;
    }
    const uint8_t m = d[i++];
    if (m == 0xD9) return "no image: its end (EOI) before a scan";
    if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7)) continue;
    if (i + 2 > n) return "cut before its scan";
    const size_t len = size_t(d[i]) << 8 | d[i + 1];
    if (len < 2) return "a segment of length " + std::to_string(len);
    if (i + len > n) return "cut before its scan";
    if (m == 0xDA) return sof ? "" : "no image: a scan before any frame header (SOF)";
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) sof = true;
    i += len;
  }
}

// ai: the process's table for LIZ_TURBOJPEG's value now ("" unset): each loaded once and kept, never closed
const Jpeg::Api* loadedApi() {
  static std::mutex mu;
  static std::map<std::string, std::unique_ptr<Jpeg::Api>> apis;
#if defined(_WIN32)
  const wchar_t* e = _wgetenv(L"LIZ_TURBOJPEG");
  const Path path = e ? e : L"";
  const std::string key = utf8(path);
#else
  const char* e = std::getenv("LIZ_TURBOJPEG");
  const Path path = e ? e : "";
  const std::string& key = path;
#endif
  std::lock_guard<std::mutex> lk(mu);
  auto& a = apis[key];
  if (!a) a = openApi(path);
  return a.get();
}

}  // namespace

std::unique_ptr<Jpeg> Jpeg::load(std::string& why) {
  const Api* a = loadedApi();
  if (!a->ok) { why = a->why; return nullptr; }
  std::unique_ptr<Jpeg> j(new Jpeg());
  j->api = a;
  j->dec = a->tjInitDecompress();
  if (!j->dec) {
    const char* e = a->tjGetErrorStr2(nullptr);
    why = std::string("TurboJPEG's decompressor did not start: ") + (e ? e : "");
    return nullptr;
  }
  return j;
}

Jpeg::~Jpeg() {
  if (dec) api->tjDestroy(dec);
  if (enc) api->tjDestroy(enc);
}

bool Jpeg::grey(const uint8_t* data, size_t n, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h, std::string* why) {
  warning_.clear();
  auto fail = [&](const std::string& s) { if (why) *why = s; return false; };
  auto said = [&]() {
    const char* e = api->tjGetErrorStr2(dec);
    return std::string(e ? e : "TurboJPEG gave no reason");
  };
  if (!data || n == 0) return fail("an empty frame");
  if (static_cast<unsigned long>(n) != n)
    return fail("a frame of " + std::to_string(n) + " bytes, past TurboJPEG's sizes");
  // ai: The header (SOI to the first scan's SOS) must be whole in the frame's own bytes (headerWhole) and then read
  // ai: with a size: TurboJPEG takes a header cut short for a stream of tables alone (success; the size unset in 2.x,
  // ai: the last header's in 3.x), and reports a warning then a fatal error as a warning ("missing SOS marker" with
  // ai: TJERR_WARNING); a size of 0 handed to tjDecompress2 would mean the JPEG's own. Whole, the header holds; what
  // ai: can still go wrong is in the scan, where libjpeg-turbo warns (corrupt data, a missing tail) and fills the rest,
  // ai: every row written.
  if (const std::string cut = headerWhole(data, n); !cut.empty()) return fail("not a frame that decodes: " + cut);
  // ai: A header that fails leaves TurboJPEG 2.x's handle inside it (tjDecompressHeader3's error path skips
  // ai: jpeg_abort_decompress, where tjDecompress2's does not), and the next frame is read on from there ("Bogus DQT
  // ai: index" on a good frame, 2026-10-25): the handle is made anew.
  if (!dec && !(dec = api->tjInitDecompress())) return fail("TurboJPEG's decompressor did not start again");
  int iw = 0, ih = 0, samp = 0, space = 0;
  if (api->tjDecompressHeader3(dec, data, static_cast<unsigned long>(n), &iw, &ih, &samp, &space) != 0) {
    const std::string s = "not a frame that decodes: " + said();
    api->tjDestroy(dec);
    dec = api->tjInitDecompress();
    return fail(s);
  }
  if (iw < 1 || ih < 1) return fail("a JPEG stream with no image in it");
  if (iw > kSide || ih > kSide || size_t(iw) * size_t(ih) > kPixels)
    return fail("a frame of " + std::to_string(iw) + " x " + std::to_string(ih) + ", past the largest decoded");
  out.resize(size_t(iw) * size_t(ih));
  if (api->tjDecompress2(dec, data, static_cast<unsigned long>(n), out.data(), iw, iw, ih, kGray, 0) != 0) {
    if (api->tjGetErrorCode(dec) != kWarning) return fail("a frame that does not decode: " + said());
    warning_ = said();
  }
  w = uint32_t(iw);
  h = uint32_t(ih);
  return true;
}

bool Jpeg::encode(const uint8_t* grey, uint32_t w, uint32_t h, uint32_t stride, int q, bool stripTables,
                  std::vector<uint8_t>& out, std::string* why) {
  auto fail = [&](const std::string& s) { if (why) *why = s; return false; };
  if (!grey || w == 0 || h == 0 || stride < w) return fail("no picture to encode");
  if (w > 65535 || h > 65535) return fail("a picture past JPEG's 65535 a side");
  if (q < 1 || q > 100) return fail("quality " + std::to_string(q) + " (1 to 100)");
  if (!enc && !(enc = api->tjInitCompress())) {
    const char* e = api->tjGetErrorStr2(nullptr);
    return fail(std::string("TurboJPEG's compressor did not start: ") + (e ? e : ""));
  }
  rgb.resize(size_t(w) * 3 * h);
  for (uint32_t y = 0; y < h; y++) {
    const uint8_t* g = grey + size_t(y) * stride;
    uint8_t* p = rgb.data() + size_t(y) * 3 * w;
    for (uint32_t x = 0; x < w; x++) p[3 * size_t(x)] = p[3 * size_t(x) + 1] = p[3 * size_t(x) + 2] = g[x];
  }
  unsigned char* buf = nullptr;
  unsigned long size = 0;
  if (api->tjCompress2(enc, rgb.data(), int(w), int(w) * 3, int(h), kRgb, &buf, &size, kSamp422, q, 0) != 0) {
    const char* e = api->tjGetErrorStr2(enc);
    const std::string s = std::string("TurboJPEG did not encode: ") + (e ? e : "");
    if (buf) api->tjFree(buf);
    return fail(s);
  }
  const unsigned char* j = buf;
  const size_t n = size;
  bool good = n >= 4 && j[0] == 0xFF && j[1] == 0xD8;
  if (good && !stripTables) out.assign(j, j + n);
  else if (good) {
    // ai: the segments from SOI to SOS as they are but DHT (FFC4), then SOS and the scan whole: the frame a UVC camera
    // ai: sends, decoded with the standard tables
    out.clear();
    out.reserve(n);
    out.insert(out.end(), j, j + 2);
    size_t i = 2;
    good = false;
    while (i + 4 <= n && j[i] == 0xFF) {
      const unsigned char mk = j[i + 1];
      if (mk == 0xFF) { i++; continue; }   // ai: a fill byte before a marker
      if (mk == 0xDA) { out.insert(out.end(), j + i, j + n); good = true; break; }
      const size_t len = size_t(j[i + 2]) << 8 | j[i + 3];
      if (len < 2 || i + 2 + len > n) break;
      if (mk != 0xC4) out.insert(out.end(), j + i, j + i + 2 + len);
      i += 2 + len;
    }
  }
  api->tjFree(buf);
  if (!good) return fail("TurboJPEG's frame is not a JPEG stream this reads (no SOS after its segments)");
  return true;
}

}  // namespace lizard::cam
