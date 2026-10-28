// ai: The phone camera's links (phone.cpp, 2026-10-27): a Wire carries the phone stream's messages
// ai: (liblizard/core/rx/stream.h) both ways, over TCP (phone.cpp) or over USB as an Android Open Accessory (aoa.cpp).
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lizard::cam {

class Wire {
 public:
  virtual ~Wire() = default;
  // ai: all n bytes, or false (the link gone, or cut)
  virtual bool send(const void* p, size_t n) = 0;
  // ai: n bytes, or false at the link's end, an error, stop, a cut, or once steadyNs passes deadlineNs (0: none)
  virtual bool recv(void* p, size_t n, const std::atomic<bool>& stop, int64_t deadlineNs = 0) = 0;
  // ai: a send or recv in another thread gives up (false), and every one after
  virtual void cut() = 0;
};

namespace aoa {
// ai: a phone on USB: id phone:usb:<bus>-<port>[.<port>...] (its place on the bus, which a switch into accessory
// ai: mode keeps), the device's product string, "USB accessory" where it is one already, else "USB"
struct Found { std::string id, name, detail; };
// ai: the phones on USB: Google's accessory devices, and the devices with a phone's interfaces (MTP, PTP, adb, a USB
// ai: tether) that answer the accessory protocol's version; notes say why one could not be asked (no access, no driver)
std::vector<Found> list(std::vector<std::string>* notes);
// ai: the phone at id as an accessory, switched into accessory mode where it is not one yet, its accessory interface
// ai: claimed; name its product string. Null with why where it cannot be
std::unique_ptr<Wire> open(const std::string& id, std::string& name, std::string& why);
}  // namespace aoa

}  // namespace lizard::cam
