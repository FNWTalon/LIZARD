// ai: The desktop's own file and folder chooser (2026-10-23): the XDG desktop portal's FileChooser, OpenFile, over the
// ai: session bus through systemd's sd-bus, which is loaded at run time (libsystemd.so.0: no build-time dependency, and
// ai: a machine without it falls back in SendState.kt). The call returns a request object at once; the choice comes as
// ai: that object's Response signal once the person closes the dialog, so the match is made before the call, on the
// ai: request path the portal is documented to use (/org/freedesktop/portal/desktop/request/<sender>/<token>), and the
// ai: bus is read until the signal comes. The portal shows the desktop's dialog (GTK's on Cinnamon and GNOME, KDE's on
// ai: Plasma), attached to the parent window named "x11:<id>". On Windows (2026-10-23) the shell's own dialog,
// ai: IFileOpenDialog, attached to the window named "win32:<hex HWND>", its choice as paths; on macOS every call answers
// ai: "unavailable" (AWT's FileDialog is the system's panel there).
#include "portal.h"

#if defined(__linux__)
#include <dlfcn.h>
#include <atomic>
#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// ai: the parts of sd-bus used here, declared from its stable ABI (sd-bus.h, which build machines may lack)
struct sd_bus;
struct sd_bus_message;
struct sd_bus_slot;
struct sd_bus_error { const char* name; const char* message; int need_free; };
using Handler = int (*)(sd_bus_message*, void*, sd_bus_error*);

struct Bus {
  int (*open_user)(sd_bus**) = nullptr;
  int (*get_unique_name)(sd_bus*, const char**) = nullptr;
  int (*match_signal)(sd_bus*, sd_bus_slot**, const char*, const char*, const char*, const char*, Handler, void*) = nullptr;
  int (*new_method_call)(sd_bus*, sd_bus_message**, const char*, const char*, const char*, const char*) = nullptr;
  int (*append)(sd_bus_message*, const char*, ...) = nullptr;
  int (*open_container)(sd_bus_message*, char, const char*) = nullptr;
  int (*close_container)(sd_bus_message*) = nullptr;
  int (*call)(sd_bus*, sd_bus_message*, uint64_t, sd_bus_error*, sd_bus_message**) = nullptr;
  int (*read)(sd_bus_message*, const char*, ...) = nullptr;
  int (*enter_container)(sd_bus_message*, char, const char*) = nullptr;
  int (*exit_container)(sd_bus_message*) = nullptr;
  int (*skip)(sd_bus_message*, const char*) = nullptr;
  int (*process)(sd_bus*, sd_bus_message**) = nullptr;
  int (*wait)(sd_bus*, uint64_t) = nullptr;
  sd_bus_message* (*message_unref)(sd_bus_message*) = nullptr;
  sd_bus_slot* (*slot_unref)(sd_bus_slot*) = nullptr;
  sd_bus* (*flush_close_unref)(sd_bus*) = nullptr;
  void (*error_free)(sd_bus_error*) = nullptr;
  bool ok = false;
};

const Bus& bus() {
  static const Bus b = [] {
    Bus x;
    void* h = dlopen("libsystemd.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!h) return x;
    bool all = true;
    auto get = [&](auto& fn, const char* name) { fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(h, name)); all = all && fn; };
    get(x.open_user, "sd_bus_open_user");
    get(x.get_unique_name, "sd_bus_get_unique_name");
    get(x.match_signal, "sd_bus_match_signal");
    get(x.new_method_call, "sd_bus_message_new_method_call");
    get(x.append, "sd_bus_message_append");
    get(x.open_container, "sd_bus_message_open_container");
    get(x.close_container, "sd_bus_message_close_container");
    get(x.call, "sd_bus_call");
    get(x.read, "sd_bus_message_read");
    get(x.enter_container, "sd_bus_message_enter_container");
    get(x.exit_container, "sd_bus_message_exit_container");
    get(x.skip, "sd_bus_message_skip");
    get(x.process, "sd_bus_process");
    get(x.wait, "sd_bus_wait");
    get(x.message_unref, "sd_bus_message_unref");
    get(x.slot_unref, "sd_bus_slot_unref");
    get(x.flush_close_unref, "sd_bus_flush_close_unref");
    get(x.error_free, "sd_bus_error_free");
    x.ok = all;
    return x;
  }();
  return b;
}

struct Answer {
  bool done = false;
  uint32_t code = 2;
  std::vector<std::string> uris;
};

// ai: Response(u response, a{sv} results): 0 chosen (results' "uris" as), 1 cancelled, 2 ended otherwise
int onResponse(sd_bus_message* m, void* user, sd_bus_error*) {
  const Bus& b = bus();
  auto* a = static_cast<Answer*>(user);
  a->done = true;
  if (b.read(m, "u", &a->code) < 0 || a->code != 0) return 0;
  if (b.enter_container(m, 'a', "{sv}") <= 0) return 0;
  while (b.enter_container(m, 'e', "sv") > 0) {
    const char* key = nullptr;
    if (b.read(m, "s", &key) < 0) break;
    if (key && std::strcmp(key, "uris") == 0 && b.enter_container(m, 'v', "as") > 0) {
      if (b.enter_container(m, 'a', "s") > 0) {
        const char* u = nullptr;
        while (b.read(m, "s", &u) > 0) a->uris.emplace_back(u);
        b.exit_container(m);
      }
      b.exit_container(m);
    } else if (b.skip(m, "v") < 0) break;
    b.exit_container(m);
  }
  b.exit_container(m);
  return 0;
}

// ai: one {sv} entry of the options
int entry(const Bus& b, sd_bus_message* m, const char* key, const char* type, const void* value) {
  int r = b.open_container(m, 'e', "sv");
  if (r >= 0) r = b.append(m, "s", key);
  if (r >= 0) r = b.open_container(m, 'v', type);
  if (r >= 0) r = type[0] == 'b' ? b.append(m, "b", *static_cast<const int*>(value)) : b.append(m, type, static_cast<const char*>(value));
  if (r >= 0) r = b.close_container(m);
  if (r >= 0) r = b.close_container(m);
  return r;
}

}  // namespace

namespace lizard {

PortalPick portalOpen(const std::string& parent, const std::string& title, bool directory, bool multiple) {
  PortalPick out;
  const Bus& b = bus();
  if (!b.ok) { out.error = "no libsystemd.so.0 with sd-bus"; return out; }
  sd_bus* s = nullptr;
  if (b.open_user(&s) < 0 || !s) { out.error = "no session bus"; return out; }
  static std::atomic<unsigned> n{0};
  const std::string token = "lizard" + std::to_string(n++);
  const char* unique = nullptr;
  b.get_unique_name(s, &unique);
  std::string sender = unique && unique[0] == ':' ? unique + 1 : (unique ? unique : "");
  for (char& c : sender) if (c == '.') c = '_';
  const std::string path = "/org/freedesktop/portal/desktop/request/" + sender + "/" + token;
  Answer answer;
  sd_bus_slot* slot = nullptr;
  int r = b.match_signal(s, &slot, "org.freedesktop.portal.Desktop", path.c_str(), "org.freedesktop.portal.Request", "Response", onResponse, &answer);
  sd_bus_message* m = nullptr;
  if (r >= 0) r = b.new_method_call(s, &m, "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.FileChooser", "OpenFile");
  if (r >= 0) r = b.append(m, "ss", parent.c_str(), title.c_str());
  if (r >= 0) r = b.open_container(m, 'a', "{sv}");
  const int yes = 1, many = multiple ? 1 : 0, dirs = directory ? 1 : 0;
  if (r >= 0) r = entry(b, m, "handle_token", "s", token.c_str());
  if (r >= 0) r = entry(b, m, "modal", "b", &yes);
  if (r >= 0) r = entry(b, m, "multiple", "b", &many);
  if (r >= 0) r = entry(b, m, "directory", "b", &dirs);
  if (r >= 0) r = b.close_container(m);
  sd_bus_error err{nullptr, nullptr, 0};
  sd_bus_message* reply = nullptr;
  if (r >= 0) r = b.call(s, m, 0, &err, &reply);
  if (r < 0) {
    out.error = err.message ? err.message : err.name ? err.name : ("the portal's OpenFile failed (" + std::to_string(r) + ")");
  } else {
    // ai: a portal that answers on another path (one from before the handle_token rule) is followed there
    const char* handle = nullptr;
    if (b.read(reply, "o", &handle) >= 0 && handle && path != handle) {
      b.slot_unref(slot); slot = nullptr;
      b.match_signal(s, &slot, "org.freedesktop.portal.Desktop", handle, "org.freedesktop.portal.Request", "Response", onResponse, &answer);
    }
    while (!answer.done) {
      const int p = b.process(s, nullptr);
      if (p < 0) { out.error = "the session bus closed while the chooser was open"; break; }
      if (p == 0 && b.wait(s, UINT64_MAX) < 0) { out.error = "the session bus closed while the chooser was open"; break; }
    }
    if (answer.done) { out.answered = true; out.uris = std::move(answer.uris); }
  }
  b.error_free(&err);
  if (reply) b.message_unref(reply);
  if (m) b.message_unref(m);
  if (slot) b.slot_unref(slot);
  b.flush_close_unref(s);
  return out;
}

}  // namespace lizard

#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shobjidl.h>

#include <cstdlib>

namespace lizard {

namespace {
std::wstring wide(const std::string& s) {
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n > 0 ? n - 1 : 0, L'\0');
  if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}
std::string utf8(const wchar_t* w) {
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s(n > 0 ? n - 1 : 0, '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
  return s;
}
}  // namespace

// ai: the shell's Open dialog on the calling thread (COM initialised there for the call), files or folders, several
// ai: where multiple; the paths chosen, none when cancelled
PortalPick portalOpen(const std::string& parent, const std::string& title, bool directory, bool multiple) {
  PortalPick out;
  HWND owner = nullptr;
  if (parent.rfind("win32:", 0) == 0) owner = reinterpret_cast<HWND>(static_cast<uintptr_t>(std::strtoull(parent.c_str() + 6, nullptr, 16)));
  const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  if (FAILED(init) && init != RPC_E_CHANGED_MODE) { out.error = "COM could not start"; return out; }
  IFileOpenDialog* d = nullptr;
  if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&d)))) {
    out.error = "no shell file dialog";
  } else {
    FILEOPENDIALOGOPTIONS o = 0;
    d->GetOptions(&o);
    o |= FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST | FOS_NOCHANGEDIR;
    if (directory) o |= FOS_PICKFOLDERS;
    if (multiple) o |= FOS_ALLOWMULTISELECT;
    d->SetOptions(o);
    if (!title.empty()) d->SetTitle(wide(title).c_str());
    const HRESULT shown = d->Show(owner);
    out.answered = SUCCEEDED(shown) || shown == HRESULT_FROM_WIN32(ERROR_CANCELLED);
    if (!out.answered) out.error = "the shell file dialog failed";
    IShellItemArray* items = nullptr;
    if (SUCCEEDED(shown) && SUCCEEDED(d->GetResults(&items))) {
      DWORD n = 0;
      items->GetCount(&n);
      for (DWORD i = 0; i < n; i++) {
        IShellItem* it = nullptr;
        if (FAILED(items->GetItemAt(i, &it))) continue;
        PWSTR path = nullptr;
        if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &path))) { out.uris.push_back(utf8(path)); CoTaskMemFree(path); }
        it->Release();
      }
      items->Release();
    }
    d->Release();
  }
  if (SUCCEEDED(init)) CoUninitialize();
  return out;
}

}  // namespace lizard

#else

namespace lizard {
PortalPick portalOpen(const std::string&, const std::string&, bool, bool) {
  PortalPick out;
  out.error = "the portal is Linux's and Windows'";
  return out;
}
}  // namespace lizard

#endif
