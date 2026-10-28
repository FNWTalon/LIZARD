// ai: The phone's frames to a PC over TCP (2026-10-27): a phone whose own decode is too slow (or absent) keeps its
// ai: camera, its phase lock and its pilots, and a PC's receiver decodes. The link is any IP route between them: USB
// ai: tethering (the phone's own switch; the PC's system drivers), Wi-Fi, adb reverse. The phone is the "pc" decoder
// ai: (ReceiverConfig::decoder, makeStreamReceiver): it serves its frames, the receiver's crop of each camera frame's Y
// ai: plane (crops.h: the centre square, or the 2:1 region whose halves the PC reads as frames), newest first, a frame
// ai: the link has no room for dropped; the PC (lizard-desktop/native/camera/phone.cpp) decodes them and sends back its
// ai: receiver's series (receiver.h series, its ms on the phone camera's clock) and its stats, so the phone's lock
// ai: and screen read the PC's decode as they read their own receiver's.
// ai: Wire (both ends little-endian: arm64 and x86-64): a message is u32 type, u32 payload bytes, the payload.
// ai:   HELLO  phone to PC, once a connection: JSON {"proto", "name", "layout", "fps"}
// ai:   FRAME  phone to PC: u64 seq (the camera's frame count: a gap the frames not sent, skipped evenly where the
// ai:          link carries fewer than the camera makes, or dropped), i64 ns (the phone camera's clock), u32 w, u32 h,
// ai:          then w x h luma bytes, rows of w
// ai:   SERIES PC to phone: doubles, receiver.h series' array (the version, then 8 a frame), ms on the phone's clock
// ai:   STATS  PC to phone: the PC receiver's stats JSON
// ai:   OPEN   PC to phone, no payload: send a HELLO again (the PC's start on a link it did not open fresh: a USB
// ai:          accessory's, whose bytes may be partway through a message; the PC skips to the HELLO)
// ai: The phone also says where it is: a UDP beacon a second to each IPv4 interface's broadcast address on
// ai: BEACON_PORT, "LIZARD-STREAM <proto> port=<tcp port> w=<w> h=<h> fps=<fps> layout=<layout> name=<name>" (w, h,
// ai: fps those of the frames sent last, 0 before any).
// ai: Or over USB by Android Open Accessory (2026-10-27; on Windows a WinUSB driver bound by the user, Zadig): the PC
// ai: switches the phone into accessory mode (lizard-desktop/native/camera/aoa.cpp), the app opens the accessory and hands its fd to
// ai: streamAccessory, and a bridge carries the same messages between it and the stream receiver: each session a
// ai: socket pair the receiver serves as it serves a TCP connection (peer "USB"), whole messages written to the
// ai: accessory, so a session that ends leaves the accessory's bytes at a message's end. The bridge lives with the
// ai: accessory, not with a receiver: its reads and writes cannot be woken (f_accessory has no poll and no timeout),
// ai: so nothing waits on them; it ends when they fail (the cable out).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "receiver.h"

namespace lizard::stream {
constexpr uint32_t PROTO = 1;
constexpr uint16_t TCP_PORT = 47230;      // ai: the phone listens (LIZ_STREAM_PORT another)
constexpr uint16_t BEACON_PORT = 47231;   // ai: the PC listens for beacons
enum Type : uint32_t { HELLO = 1, FRAME = 2, SERIES = 3, STATS = 4, OPEN = 5 };
constexpr uint32_t FRAME_HEAD = 24;
constexpr uint32_t MAX_PAYLOAD = 64u << 20;   // ai: past it a message is a broken stream
}  // namespace lizard::stream

namespace lizard {
// ai: Receiver::create's for decoder "pc": throws where it cannot listen (the port taken, no sockets: Windows has none
// ai: here, the phone side being a phone's)
std::unique_ptr<Receiver> makeStreamReceiver(const ReceiverConfig& c, std::function<void(uint64_t tag)> release);
// ai: the accessory's fd (UsbManager.openAccessory, detached), the bridge's from here: it closes it at its end. False
// ai: where a bridge is up already (the fd closed) or this system has none (Windows)
bool streamAccessory(int fd, std::function<void(const std::string&)> log);
// ai: whether a bridge is up (the accessory open and not failed)
bool accessoryOpen();
}  // namespace lizard
