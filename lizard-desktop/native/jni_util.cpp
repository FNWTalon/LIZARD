// ai: jni_util.h: the strings' conversions and the shared error channel.
#include "jni_util.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace lizard::jni {

namespace {
std::mutex gErrorMu;
std::string gError;

void put(std::string& o, uint32_t c) {
  if (c < 0x80) {
    o += char(c);
  } else if (c < 0x800) {
    o += char(0xC0 | (c >> 6));
    o += char(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    o += char(0xE0 | (c >> 12));
    o += char(0x80 | ((c >> 6) & 0x3F));
    o += char(0x80 | (c & 0x3F));
  } else {
    o += char(0xF0 | (c >> 18));
    o += char(0x80 | ((c >> 12) & 0x3F));
    o += char(0x80 | ((c >> 6) & 0x3F));
    o += char(0x80 | (c & 0x3F));
  }
}
}  // namespace

std::string utf8(const jchar* s, size_t n) {
  std::string o;
  o.reserve(n + n / 2);
  for (size_t i = 0; i < n; i++) {
    uint32_t c = s[i];
    if (c >= 0xD800 && c <= 0xDBFF && i + 1 < n && s[i + 1] >= 0xDC00 && s[i + 1] <= 0xDFFF) {
      c = 0x10000 + ((c - 0xD800) << 10) + (uint32_t(s[i + 1]) - 0xDC00);
      i++;
    } else if (c >= 0xD800 && c <= 0xDFFF) {
      c = 0xFFFD;
    }
    put(o, c);
  }
  return o;
}

namespace {
// ai: UTF-8 to UTF-16, jstr's
std::vector<jchar> utf16(const std::string& s) {
  std::vector<jchar> o;
  o.reserve(s.size());
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t b = uint8_t(s[i]);
    uint32_t c = 0;
    size_t len = 0;
    // ai: the second byte's range by the first (Unicode's table 3-7: no overlong form, no surrogate, nothing past
    // ai: U+10FFFF), every later byte 80 to BF
    uint8_t lo = 0x80, hi = 0xBF;
    if (b < 0x80) { c = b; len = 1; }
    else if (b >= 0xC2 && b <= 0xDF) { c = b & 0x1F; len = 2; }
    else if (b >= 0xE0 && b <= 0xEF) { c = b & 0x0F; len = 3; lo = b == 0xE0 ? 0xA0 : 0x80; hi = b == 0xED ? 0x9F : 0xBF; }
    else if (b >= 0xF0 && b <= 0xF4) { c = b & 0x07; len = 4; lo = b == 0xF0 ? 0x90 : 0x80; hi = b == 0xF4 ? 0x8F : 0xBF; }
    else { o.push_back(0xFFFD); i++; continue; }
    size_t k = 1;
    for (; k < len && i + k < n; k++) {
      const uint8_t t = uint8_t(s[i + k]);
      if (t < (k == 1 ? lo : 0x80) || t > (k == 1 ? hi : 0xBF)) break;
      c = (c << 6) | (t & 0x3F);
    }
    if (k < len) { o.push_back(0xFFFD); i += k; continue; }
    i += len;
    if (c >= 0x10000) {
      c -= 0x10000;
      o.push_back(jchar(0xD800 + (c >> 10)));
      o.push_back(jchar(0xDC00 + (c & 0x3FF)));
    } else {
      o.push_back(jchar(c));
    }
  }
  return o;
}
}  // namespace

std::string str(JNIEnv* e, jstring s) {
  if (!s) return "";
  const jsize n = e->GetStringLength(s);
  if (n <= 0) return "";
  std::vector<jchar> b(static_cast<size_t>(n));
  e->GetStringRegion(s, 0, n, b.data());
  return utf8(b.data(), b.size());
}

jstring jstr(JNIEnv* e, const std::string& s) {
  const std::vector<jchar> u = utf16(s);
  static const jchar none = 0;
  return e->NewString(u.empty() ? &none : u.data(), jsize(u.size()));
}

void setError(const std::string& m) {
  std::lock_guard<std::mutex> l(gErrorMu);
  gError = m;
}

std::string lastError() {
  std::lock_guard<std::mutex> l(gErrorMu);
  return gError;
}

}  // namespace lizard::jni
