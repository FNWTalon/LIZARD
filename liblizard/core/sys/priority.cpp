// ai: priority.h's raises. rtkit's rate limit (by default 25 requests in 20 s) is why only the threads that keep
// ai: frames on time ask, not every thread of the process.
#include "priority.h"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#elif !defined(__ANDROID__)
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <mutex>
#include <type_traits>
#endif

namespace lizard::sys {
namespace {

bool off() {
  const char* e = std::getenv("LIZ_PRIORITY");
  return e && !std::strcmp(e, "normal");
}

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__ANDROID__)
// ai: libdbus-1's calls the request needs, loaded at run time (no D-Bus headers on the build machines); DBusError's
// ai: layout is libdbus's public struct (two strings, five one-bit flags, a pointer)
struct DBusError { const char* name; const char* message; unsigned int flags; void* padding; };
constexpr int BUS_SYSTEM = 1, TYPE_INVALID = 0, TYPE_INT32 = 'i', TYPE_UINT64 = 't';
struct Rtkit {
  void (*errorInit)(DBusError*) = nullptr;
  void (*errorFree)(DBusError*) = nullptr;
  void* (*busGetPrivate)(int, DBusError*) = nullptr;
  void* (*newMethodCall)(const char*, const char*, const char*, const char*) = nullptr;
  int (*appendArgs)(void*, int, ...) = nullptr;
  void* (*sendBlock)(void*, void*, int, DBusError*) = nullptr;
  void (*messageUnref)(void*) = nullptr;
  void* bus = nullptr;   // ai: the process's own connection, kept
  int nice = 0;          // ai: what rtkit granted the first thread that asked (0: nothing yet)
  bool dead = false;     // ai: no libdbus, no bus, or rtkit granted nothing: not asked again

  bool open() {
    if (bus) return true;
    if (dead) return false;
    void* h = dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_LOCAL);
    auto get = [&](auto& fn, const char* s) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, s)); return fn != nullptr; };
    if (!h || !get(errorInit, "dbus_error_init") || !get(errorFree, "dbus_error_free") || !get(busGetPrivate, "dbus_bus_get_private") ||
        !get(newMethodCall, "dbus_message_new_method_call") || !get(appendArgs, "dbus_message_append_args") ||
        !get(sendBlock, "dbus_connection_send_with_reply_and_block") || !get(messageUnref, "dbus_message_unref")) {
      dead = true;
      return false;
    }
    DBusError e;
    errorInit(&e);
    bus = busGetPrivate(BUS_SYSTEM, &e);
    errorFree(&e);
    dead = !bus;
    return bus != nullptr;
  }

  // ai: MakeThreadHighPriority for one thread of this process: granted or not
  bool ask(long tid, int n) {
    void* m = newMethodCall("org.freedesktop.RealtimeKit1", "/org/freedesktop/RealtimeKit1", "org.freedesktop.RealtimeKit1",
                            "MakeThreadHighPriority");
    if (!m) return false;
    uint64_t t = static_cast<uint64_t>(tid);
    int32_t p = n;
    bool ok = false;
    if (appendArgs(m, TYPE_UINT64, &t, TYPE_INT32, &p, TYPE_INVALID)) {
      DBusError e;
      errorInit(&e);
      if (void* r = sendBlock(bus, m, 2000, &e)) { messageUnref(r); ok = true; }
      errorFree(&e);
    }
    messageUnref(m);
    return ok;
  }
};

std::string raiseLinux() {
  const long tid = static_cast<long>(syscall(SYS_gettid));
  if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), -20) == 0) return "nice -20";
  static std::mutex mu;
  static Rtkit rk;
  std::lock_guard<std::mutex> l(mu);
  if (!rk.open()) return "";
  if (!rk.nice) {
    // ai: the lowest nice rtkit grants, found by the first thread that asks
    for (int n : {-15, -10, -5})
      if (rk.ask(tid, n)) { rk.nice = n; return "nice " + std::to_string(n) + " by rtkit"; }
    rk.dead = true;
    return "";
  }
  return rk.ask(tid, rk.nice) ? "nice " + std::to_string(rk.nice) + " by rtkit" : "";
}
#endif

}  // namespace

std::string raiseProcess() {
  if (off()) return "";
#if defined(_WIN32)
  return SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) ? "priority class high" : "";
#else
  return raiseThread();
#endif
}

std::string raiseThread() {
  if (off()) return "";
#if defined(_WIN32) || defined(__ANDROID__)
  return "";
#elif defined(__APPLE__)
  return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0 ? "QoS user-interactive" : "";
#else
  return raiseLinux();
#endif
}

}  // namespace lizard::sys
