// ai: What the C API's sources share (api/*.cpp): the calling thread's last error, and the guards every entry point
// ai: runs its body in, so no C++ exception crosses the ABI.
#pragma once
#include <new>
#include <stdexcept>
#include <string>

#include "lizard.h"

namespace liz {

// ai: thrown inside the library for a failure with its own code; anything else is LIZ_E_INTERNAL, bad_alloc NOMEM
struct Error : std::runtime_error {
  int code;
  Error(int c, const std::string& m) : std::runtime_error(m), code(c) {}
};

int fail(int code, const std::string& message);
void clearError();

template <class F>
int guard(F&& f) {
  try {
    clearError();
    return f();
  } catch (const Error& e) {
    return fail(e.code, e.what());
  } catch (const std::bad_alloc&) {
    return fail(LIZ_E_NOMEM, "out of memory");
  } catch (const std::exception& e) {
    return fail(LIZ_E_INTERNAL, e.what());
  } catch (...) {
    return fail(LIZ_E_INTERNAL, "an unknown failure");
  }
}

template <class T, class F>
T* guardNew(F&& f) {
  T* out = nullptr;
  guard([&] { out = f(); return 0; });
  return out;
}

inline void need(bool ok, const std::string& what) { if (!ok) throw Error(LIZ_E_ARG, what); }

// ai: the ring's index from the API's (0 to 3, or LIZ_RING_DEFAULT)
int ringIndex(int ring);
// ai: a format's geometry, its arguments checked (format.cpp)
liz_geometry geometryOf(const liz_format& f);

}  // namespace liz
