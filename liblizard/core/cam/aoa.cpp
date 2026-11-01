// ai: Android Open Accessory (aoa.h, 2026-10-27): a phone on the cable as the phone camera's link, with no IP route,
// ai: beside a USB tether (which needs no driver on the PC, and the phone's own switch). The PC asks the phone's accessory protocol (control request 51),
// ai: hands it the accessory's strings (52) and starts it (53); the phone comes back on the same port as Google's
// ai: accessory device (18D1:2D00, 2D01 with adb), the app opens the accessory (lizard-android/.../PcLink.kt) and the
// ai: stream's bridge (liblizard/core/rx/stream.h streamAccessory) carries the stream's messages over the accessory
// ai: interface's two bulk endpoints. libusb-1.0 is loaded at run time (LIZ_LIBUSB: a path, or empty for none): on
// ai: Linux the user needs access to the device (a udev rule, as adb's), on Windows a WinUSB driver bound to the
// ai: phone's interface and to the accessory's (Zadig), on macOS nothing more.
#include "aoa.h"
#include "clock.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define LUSB __stdcall
#else
#include <dlfcn.h>
#define LUSB
#endif

namespace lizard::cam {
namespace {

// ai: libusb.h's structs and calls (libusb 1.0.16 and up), declared here: build machines may lack the header, and the
// ai: library's ABI is fixed by its soname (checked against this box's libusb-1.0.so.0 and lsusb, 2026-10-27)
struct DevDesc {
  uint8_t bLength, bDescriptorType;
  uint16_t bcdUSB;
  uint8_t bDeviceClass, bDeviceSubClass, bDeviceProtocol, bMaxPacketSize0;
  uint16_t idVendor, idProduct, bcdDevice;
  uint8_t iManufacturer, iProduct, iSerialNumber, bNumConfigurations;
};
struct EpDesc {
  uint8_t bLength, bDescriptorType, bEndpointAddress, bmAttributes;
  uint16_t wMaxPacketSize;
  uint8_t bInterval, bRefresh, bSynchAddress;
  const unsigned char* extra;
  int extraLength;
};
struct AltDesc {
  uint8_t bLength, bDescriptorType, bInterfaceNumber, bAlternateSetting, bNumEndpoints, bInterfaceClass, bInterfaceSubClass,
      bInterfaceProtocol, iInterface;
  const EpDesc* endpoint;
  const unsigned char* extra;
  int extraLength;
};
struct IfDesc {
  const AltDesc* altsetting;
  int numAltsetting;
};
struct CfgDesc {
  uint8_t bLength, bDescriptorType;
  uint16_t wTotalLength;
  uint8_t bNumInterfaces, bConfigurationValue, iConfiguration, bmAttributes, MaxPower;
  const IfDesc* interface;
  const unsigned char* extra;
  int extraLength;
};
static_assert(sizeof(DevDesc) == 18, "libusb_device_descriptor");

constexpr int ERR_ACCESS = -3, ERR_TIMEOUT = -7, ERR_NOT_SUPPORTED = -12;
constexpr uint16_t GOOGLE = 0x18D1;
bool accessoryPid(uint16_t p) { return p >= 0x2D00 && p <= 0x2D05; }
// ai: the accessory protocol's requests (AOA 1.0)
constexpr uint8_t GET_PROTOCOL = 51, SEND_STRING = 52, START = 53;
constexpr uint8_t VENDOR_IN = 0xC0, VENDOR_OUT = 0x40;
// ai: the strings the app's filter matches (res/xml/accessory_filter.xml, PcLink.kt): manufacturer, model,
// ai: description, version, URI (Android offers it where no app matches), serial
const char* const STRINGS[6] = {"LIZARD", "PC receiver", "LIZARD's PC receiver", "1", "https://github.com/FNWTalon/LIZARD", ""};
constexpr int SWITCH_MS = 8000;   // ai: the phone's way back onto the bus as an accessory
constexpr int64_t STALL_NS = 1'500'000'000;   // ai: a send that moves nothing this long has no phone to go to

struct Usb {
  bool ok = false;
  std::string why;
  void* ctx = nullptr;
  int(LUSB* init)(void**) = nullptr;
  std::ptrdiff_t(LUSB* getDeviceList)(void*, void***) = nullptr;
  void(LUSB* freeDeviceList)(void**, int) = nullptr;
  int(LUSB* getDeviceDescriptor)(void*, DevDesc*) = nullptr;
  int(LUSB* getActiveConfigDescriptor)(void*, CfgDesc**) = nullptr;
  void(LUSB* freeConfigDescriptor)(CfgDesc*) = nullptr;
  uint8_t(LUSB* getBusNumber)(void*) = nullptr;
  int(LUSB* getPortNumbers)(void*, uint8_t*, int) = nullptr;
  int(LUSB* open)(void*, void**) = nullptr;
  void(LUSB* close)(void*) = nullptr;
  void*(LUSB* getDevice)(void*) = nullptr;
  int(LUSB* controlTransfer)(void*, uint8_t, uint8_t, uint16_t, uint16_t, unsigned char*, uint16_t, unsigned) = nullptr;
  int(LUSB* bulkTransfer)(void*, unsigned char, unsigned char*, int, int*, unsigned) = nullptr;
  int(LUSB* claimInterface)(void*, int) = nullptr;
  int(LUSB* releaseInterface)(void*, int) = nullptr;
  int(LUSB* getStringDescriptorAscii)(void*, uint8_t, unsigned char*, int) = nullptr;
  const char*(LUSB* errorName)(int) = nullptr;

  std::string error(int r) const { return errorName ? errorName(r) : std::to_string(r); }
};

// ai: the process's libusb, loaded once and kept: LIZ_LIBUSB alone where set (empty: none), else the system's names
const Usb* usb() {
  static const Usb u = [] {
    Usb a;
    std::vector<std::string> names;
    if (const char* e = std::getenv("LIZ_LIBUSB")) {
      if (!*e) { a.why = ""; return a; }   // ai: off: no note
      names = {e};
    } else {
#if defined(_WIN32)
      names = {"libusb-1.0.dll"};
#elif defined(__APPLE__)
      names = {"/opt/homebrew/lib/libusb-1.0.0.dylib", "/usr/local/lib/libusb-1.0.0.dylib", "libusb-1.0.0.dylib"};
#else
      names = {"libusb-1.0.so.0"};
#endif
    }
    std::string tried;
    for (const auto& name : names) {
#if defined(_WIN32)
      std::wstring wide(name.size() + 1, L'\0');
      wide.resize(static_cast<size_t>(std::max(0, MultiByteToWideChar(CP_UTF8, 0, name.c_str(), -1, wide.data(), static_cast<int>(wide.size())) - 1)));
      HMODULE h = LoadLibraryW(wide.c_str());
      if (!h) { tried += (tried.empty() ? "" : "; ") + name + ": not loaded"; continue; }
      auto sym = [&](const char* s) { return reinterpret_cast<void*>(GetProcAddress(h, s)); };
#else
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
      get(a.init, "libusb_init");
      get(a.getDeviceList, "libusb_get_device_list");
      get(a.freeDeviceList, "libusb_free_device_list");
      get(a.getDeviceDescriptor, "libusb_get_device_descriptor");
      get(a.getActiveConfigDescriptor, "libusb_get_active_config_descriptor");
      get(a.freeConfigDescriptor, "libusb_free_config_descriptor");
      get(a.getBusNumber, "libusb_get_bus_number");
      get(a.getPortNumbers, "libusb_get_port_numbers");
      get(a.open, "libusb_open");
      get(a.close, "libusb_close");
      get(a.getDevice, "libusb_get_device");
      get(a.controlTransfer, "libusb_control_transfer");
      get(a.bulkTransfer, "libusb_bulk_transfer");
      get(a.claimInterface, "libusb_claim_interface");
      get(a.releaseInterface, "libusb_release_interface");
      get(a.getStringDescriptorAscii, "libusb_get_string_descriptor_ascii");
      get(a.errorName, "libusb_error_name");
      if (!missing.empty()) { tried += (tried.empty() ? "" : "; ") + name + " lacks " + missing; continue; }
      const int r = a.init(&a.ctx);
      if (r != 0) { tried += (tried.empty() ? "" : "; ") + name + ": libusb_init " + a.error(r); continue; }
      a.ok = true;
      return a;
    }
    a.why = "phones on USB: libusb-1.0 is not loaded (" + tried + ")";
    return a;
  }();
  return &u;
}

// ai: a device's place on the bus, "<bus>-<port>[.<port>...]"; "" for a root hub
std::string placeOf(const Usb* u, void* dev) {
  uint8_t ports[8];
  const int n = u->getPortNumbers(dev, ports, 8);
  if (n <= 0) return "";
  std::string s = std::to_string(u->getBusNumber(dev)) + "-";
  for (int i = 0; i < n; i++) s += (i ? "." : "") + std::to_string(ports[i]);
  return s;
}

// ai: an interface a phone shows: MTP (vendor FF/FF/00, as Android's MTP function and the accessory's own), PTP
// ai: (06/01/01), adb (FF/42/01), a USB tether (RNDIS E0/01/03 or EF/04/01, NCM 02/0D)
bool phoneLike(const Usb* u, void* dev) {
  CfgDesc* c = nullptr;
  if (u->getActiveConfigDescriptor(dev, &c) != 0 || !c) return false;
  bool yes = false;
  for (int i = 0; i < c->bNumInterfaces && !yes; i++)
    for (int k = 0; k < c->interface[i].numAltsetting && !yes; k++) {
      const AltDesc& a = c->interface[i].altsetting[k];
      const uint8_t cl = a.bInterfaceClass, sub = a.bInterfaceSubClass, pr = a.bInterfaceProtocol;
      yes = (cl == 0xFF && sub == 0xFF && pr == 0x00) || (cl == 0x06 && sub == 0x01) || (cl == 0xFF && sub == 0x42 && pr == 0x01) ||
            (cl == 0xE0 && sub == 0x01 && pr == 0x03) || (cl == 0xEF && sub == 0x04 && pr == 0x01) || (cl == 0x02 && sub == 0x0D);
    }
  u->freeConfigDescriptor(c);
  return yes;
}

std::string stringOf(const Usb* u, void* h, uint8_t index) {
  if (!index) return "";
  unsigned char b[256] = {};
  const int n = u->getStringDescriptorAscii(h, index, b, sizeof b - 1);
  return n > 0 ? std::string(reinterpret_cast<char*>(b), static_cast<size_t>(n)) : "";
}

// ai: the accessory protocol's version the device answers, 0 where it answers none
int protocolOf(const Usb* u, void* h) {
  unsigned char b[2] = {};
  return u->controlTransfer(h, VENDOR_IN, GET_PROTOCOL, 0, 0, b, 2, 1000) == 2 ? b[0] | b[1] << 8 : 0;
}

std::string openHint(int r) {
  if (r == ERR_ACCESS) return " (no access: a udev rule for its vendor, as adb's)";
  if (r == ERR_NOT_SUPPORTED) return " (no WinUSB driver on it: Zadig binds one)";
  return "";
}

// ai: the device at place opened (only Google's accessory ids where accessory), with its descriptor; null with err
void* openAt(const Usb* u, const std::string& place, bool accessory, DevDesc& dd, int& err) {
  void** devs = nullptr;
  const std::ptrdiff_t n = u->getDeviceList(u->ctx, &devs);
  void* h = nullptr;
  err = 0;
  bool seen = false;
  for (std::ptrdiff_t i = 0; i < n && !h && !seen; i++) {
    if (placeOf(u, devs[i]) != place || u->getDeviceDescriptor(devs[i], &dd) != 0) continue;
    if (accessory && !(dd.idVendor == GOOGLE && accessoryPid(dd.idProduct))) continue;
    seen = true;
    err = u->open(devs[i], &h);
    if (err != 0) h = nullptr;
  }
  if (n >= 0) u->freeDeviceList(devs, 1);
  return h;
}

// ai: The accessory interface's two bulk endpoints. recv reads 1 MiB transfers into its buffer (whole packets, so no
// ai: transfer overflows), or straight into the caller's memory for a large read; send ends a message on a whole
// ai: packet with a zero-length one, since the phone's read (f_accessory's 16 KB request) returns at a short packet
class UsbWire : public Wire {
 public:
  UsbWire(const Usb* u, void* h, int ifc, uint8_t in, uint8_t out, int inMax, int outMax)
      : u_(u), h_(h), ifc_(ifc), in_(in), out_(out), inMax_(std::max(1, inMax)), outMax_(std::max(1, outMax)), buf_(1 << 20) {}
  ~UsbWire() override {
    u_->releaseInterface(h_, ifc_);
    u_->close(h_);
  }

  // ai: false where the phone takes nothing for STALL_NS (its app gone: the accessory's reads stop with the cable in)
  bool send(const void* p, size_t n) override {
    auto* b = static_cast<unsigned char*>(const_cast<void*>(p));
    size_t left = n;
    int64_t stall = steadyNs() + STALL_NS;
    while (left) {
      if (cut_) return false;
      int done = 0;
      const int r = u_->bulkTransfer(h_, out_, b, static_cast<int>(std::min<size_t>(left, 1 << 20)), &done, 500);
      if (r != 0 && r != ERR_TIMEOUT) return false;
      if (done > 0) stall = steadyNs() + STALL_NS;
      else if (steadyNs() > stall) return false;
      b += done;
      left -= static_cast<size_t>(done);
    }
    if (n % static_cast<size_t>(outMax_) == 0) {
      int done = 0;
      u_->bulkTransfer(h_, out_, b, 0, &done, 500);
    }
    return !cut_;
  }

  bool recv(void* p, size_t n, const std::atomic<bool>& stop, int64_t deadlineNs) override {
    auto* b = static_cast<unsigned char*>(p);
    while (n) {
      if (at_ < len_) {
        const size_t k = std::min(n, len_ - at_);
        std::memcpy(b, &buf_[at_], k);
        at_ += k;
        b += k;
        n -= k;
        continue;
      }
      if (stop || cut_ || (deadlineNs && steadyNs() > deadlineNs)) return false;
      const bool direct = n >= buf_.size();
      unsigned char* to = direct ? b : buf_.data();
      const size_t want = direct ? std::min<size_t>(n - n % static_cast<size_t>(inMax_), 16u << 20) : buf_.size();
      int done = 0;
      const int r = u_->bulkTransfer(h_, in_, to, static_cast<int>(want), &done, 200);
      if (r != 0 && r != ERR_TIMEOUT) return false;
      if (direct) {
        b += done;
        n -= static_cast<size_t>(done);
      } else {
        at_ = 0;
        len_ = static_cast<size_t>(done);
      }
    }
    return true;
  }

  void cut() override { cut_ = true; }

 private:
  const Usb* u_;
  void* h_;
  const int ifc_;
  const uint8_t in_, out_;
  const int inMax_, outMax_;
  std::vector<unsigned char> buf_;
  size_t at_ = 0, len_ = 0;
  std::atomic<bool> cut_{false};
};

}  // namespace

namespace aoa {

std::vector<Found> list(std::vector<std::string>* notes) {
  const Usb* u = usb();
  if (!u->ok) {
    if (notes && !u->why.empty()) notes->push_back(u->why);
    return {};
  }
  std::vector<Found> out;
  void** devs = nullptr;
  const std::ptrdiff_t n = u->getDeviceList(u->ctx, &devs);
  for (std::ptrdiff_t i = 0; i < n; i++) {
    DevDesc dd{};
    const std::string place = placeOf(u, devs[i]);
    if (place.empty() || u->getDeviceDescriptor(devs[i], &dd) != 0 || dd.bDeviceClass == 0x09) continue;
    const bool acc = dd.idVendor == GOOGLE && accessoryPid(dd.idProduct);
    if (!acc && !phoneLike(u, devs[i])) continue;
    void* h = nullptr;
    const int r = u->open(devs[i], &h);
    if (r != 0) {
      if (notes) notes->push_back("phones on USB: the device at " + place + " was not opened: " + u->error(r) + openHint(r));
      continue;
    }
    const std::string name = stringOf(u, h, dd.iProduct);
    const bool speaks = acc || protocolOf(u, h) >= 1;
    u->close(h);
    if (speaks) out.push_back({"phone:usb:" + place, name.empty() ? "Phone" : name, acc ? "USB accessory" : "USB"});
  }
  if (n >= 0) u->freeDeviceList(devs, 1);
  return out;
}

std::string stamp() {
  const Usb* u = usb();
  if (!u->ok) return "";
  std::string s;
  void** devs = nullptr;
  const std::ptrdiff_t n = u->getDeviceList(u->ctx, &devs);
  char ids[16];
  for (std::ptrdiff_t i = 0; i < n; i++) {
    DevDesc dd{};
    if (u->getDeviceDescriptor(devs[i], &dd) != 0) continue;
    std::snprintf(ids, sizeof ids, "%04x:%04x", dd.idVendor, dd.idProduct);
    s += placeOf(u, devs[i]) + "=" + ids + ";";
  }
  if (n >= 0) u->freeDeviceList(devs, 1);
  return s;
}

std::unique_ptr<Wire> open(const std::string& id, std::string& name, std::string& why) {
  const Usb* u = usb();
  if (!u->ok) { why = u->why.empty() ? "phones on USB are off (LIZ_LIBUSB empty)" : u->why; return nullptr; }
  const std::string prefix = "phone:usb:";
  if (id.compare(0, prefix.size(), prefix) != 0) { why = "not a phone on USB: " + id; return nullptr; }
  const std::string place = id.substr(prefix.size());
  DevDesc dd{};
  int err = 0;
  void* h = openAt(u, place, false, dd, err);
  if (!h) { why = err ? "the phone at USB " + place + " was not opened: " + u->error(err) + openHint(err) : "no device at USB " + place; return nullptr; }
  if (!(dd.idVendor == GOOGLE && accessoryPid(dd.idProduct))) {
    // ai: into accessory mode: the strings, then start; the phone leaves the bus and comes back on the same port
    const int v = protocolOf(u, h);
    if (v < 1) { u->close(h); why = "the device at USB " + place + " does not speak Android Open Accessory"; return nullptr; }
    for (uint16_t i = 0; i < 6; i++) {
      const int r = u->controlTransfer(h, VENDOR_OUT, SEND_STRING, 0, i, reinterpret_cast<unsigned char*>(const_cast<char*>(STRINGS[i])),
                                       static_cast<uint16_t>(std::strlen(STRINGS[i]) + 1), 1000);
      if (r < 0) { u->close(h); why = "the phone refused the accessory's strings: " + u->error(r); return nullptr; }
    }
    const int r = u->controlTransfer(h, VENDOR_OUT, START, 0, 0, nullptr, 0, 1000);
    u->close(h);
    h = nullptr;
    if (r < 0) { why = "the phone refused to start as an accessory: " + u->error(r); return nullptr; }
    const int64_t until = steadyNs() + SWITCH_MS * 1'000'000LL;
    while (!h && steadyNs() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      h = openAt(u, place, true, dd, err);
    }
    if (!h) {
      why = err ? "the phone's accessory was not opened: " + u->error(err) + openHint(err)
                : "the phone did not come back as an accessory (on the phone: LIZARD installed, its USB prompt allowed)";
      return nullptr;
    }
  }
  // ai: the accessory interface: vendor FF/FF/00 with a bulk endpoint each way (the other, where there is one, adb's)
  CfgDesc* c = nullptr;
  int ifc = -1, inMax = 512, outMax = 512;
  uint8_t in = 0, out = 0;
  if (u->getActiveConfigDescriptor(u->getDevice(h), &c) == 0 && c) {
    for (int i = 0; i < c->bNumInterfaces && ifc < 0; i++) {
      if (c->interface[i].numAltsetting < 1) continue;
      const AltDesc& a = c->interface[i].altsetting[0];
      if (a.bInterfaceClass != 0xFF || a.bInterfaceSubClass != 0xFF || a.bInterfaceProtocol != 0x00) continue;
      uint8_t ei = 0, eo = 0;
      int mi = 0, mo = 0;
      for (int e = 0; e < a.bNumEndpoints; e++) {
        const EpDesc& ep = a.endpoint[e];
        if ((ep.bmAttributes & 3) != 2) continue;
        if (ep.bEndpointAddress & 0x80) { ei = ep.bEndpointAddress; mi = ep.wMaxPacketSize & 0x7FF; }
        else { eo = ep.bEndpointAddress; mo = ep.wMaxPacketSize & 0x7FF; }
      }
      if (ei && eo) { ifc = a.bInterfaceNumber; in = ei; out = eo; inMax = mi; outMax = mo; }
    }
    u->freeConfigDescriptor(c);
  }
  if (ifc < 0) { u->close(h); why = "the phone's accessory shows no accessory interface"; return nullptr; }
  if (const int r = u->claimInterface(h, ifc); r != 0) {
    u->close(h);
    why = "the phone's accessory interface was not claimed: " + u->error(r) + openHint(r);
    return nullptr;
  }
  name = stringOf(u, h, dd.iProduct);
  return std::make_unique<UsbWire>(u, h, ifc, in, out, inMax, outMax);
}

}  // namespace aoa

}  // namespace lizard::cam
