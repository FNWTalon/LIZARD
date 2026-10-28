// ai: A phone as the camera (2026-10-27; liblizard/core/rx/stream.h): a phone running the app's "pc" decoder serves
// ai: its frames, the receiver's crop of its camera's Y plane, over TCP on any IP route to this PC (USB tethering,
// ai: Wi-Fi), with its own camera control, phase lock and pilots; this end reads them as a camera's frames and sends
// ai: back what its receiver reads of them (readBack: the series, on the phone camera's clock, and the stats), which
// ai: the phone's lock and screen read. Phones are found by their beacons (a UDP listener on stream::BEACON_PORT,
// ai: started at the first list), each listed while its beacon was heard in the last BEACON_LIVE_NS, and by asking
// ai: each of this PC's IPv4 gateways for a phone's hello (2026-10-27: a phone is the gateway of its own USB tether, and
// ai: an outgoing connection passes a firewall that keeps
// ai: the beacons out, as Windows' does on a network it calls public); an id phone:<ipv4>:<port> opens one whether or
// ai: not it was found (an address typed in, a route neither takes: adb forward). Or over USB with no IP route, as an
// ai: Android Open Accessory (aoa.h, phone:usb:<place>), listed where a phone on the bus answers the protocol.
#if defined(_WIN32) && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600   // ai: GetAdaptersAddresses' gateways
#endif
#include "aoa.h"
#include "backends.h"
#include "clock.h"
#include "stream.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <net/route.h>
#include <sys/sysctl.h>
#endif
#endif

namespace lizard::cam {
namespace {

using json = nlohmann::json;

#if defined(_WIN32)
using Sock = SOCKET;
const Sock NONE = INVALID_SOCKET;
void closeSock(Sock s) { closesocket(s); }
void shutSock(Sock s) { shutdown(s, SD_BOTH); }
bool netUp() {
  static const bool up = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }();
  return up;
}
#else
using Sock = int;
const Sock NONE = -1;
void closeSock(Sock s) { close(s); }
void shutSock(Sock s) { shutdown(s, SHUT_RDWR); }
bool netUp() { return true; }
#endif

#if defined(MSG_NOSIGNAL)
const int SEND_FLAGS = MSG_NOSIGNAL;
#else
const int SEND_FLAGS = 0;
#endif

constexpr int64_t BEACON_LIVE_NS = 3'000'000'000;

// ai: whether s can be read (or written) within ms; false on a timeout or an error
bool ready(Sock s, int ms, bool write = false) {
  fd_set f;
  FD_ZERO(&f);
  FD_SET(s, &f);
  timeval t{ms / 1000, (ms % 1000) * 1000};
  const int r = select(static_cast<int>(s) + 1, write ? nullptr : &f, write ? &f : nullptr, nullptr, &t);
  return r > 0;
}

bool sendAll(Sock s, const void* p, size_t n) {
  const char* b = static_cast<const char*>(p);
  while (n) {
    const int k = send(s, b, static_cast<int>(std::min<size_t>(n, 1 << 20)), SEND_FLAGS);
    if (k <= 0) return false;
    b += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

// ai: n bytes, or false at the stream's end, an error, stop (polled every 200 ms), or past deadlineNs (0: none)
bool recvAll(Sock s, void* p, size_t n, const std::atomic<bool>& stop, int64_t deadlineNs = 0) {
  char* b = static_cast<char*>(p);
  while (n) {
    if (stop || (deadlineNs && steadyNs() > deadlineNs)) return false;
    if (!ready(s, 200)) continue;
    const int k = recv(s, b, static_cast<int>(std::min<size_t>(n, 1 << 20)), 0);
    if (k <= 0) return false;
    b += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

// ai: A TCP connection to ip:port within ms, NONE with why otherwise
Sock dial(const std::string& ip, uint16_t port, int ms, std::string& why) {
  if (!netUp()) { why = "no sockets"; return NONE; }
  addrinfo hint{}, *res = nullptr;
  hint.ai_family = AF_INET;
  hint.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(ip.c_str(), std::to_string(port).c_str(), &hint, &res) != 0 || !res) { why = "no address " + ip; return NONE; }
  Sock s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == NONE) { freeaddrinfo(res); why = "no socket"; return NONE; }
#if defined(_WIN32)
  u_long nb = 1;
  ioctlsocket(s, FIONBIO, &nb);
#else
  const int fl = fcntl(s, F_GETFL, 0);
  fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif
  connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen));
  freeaddrinfo(res);
  int err = 0;
  socklen_t el = sizeof err;
  if (!ready(s, ms, true) || getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &el) != 0 || err != 0) {
    closeSock(s);
    why = "the phone at " + ip + ":" + std::to_string(port) + " did not answer";
    return NONE;
  }
#if defined(_WIN32)
  nb = 0;
  ioctlsocket(s, FIONBIO, &nb);
#else
  fcntl(s, F_SETFL, fl);
#endif
  int one = 1, buf = 8 << 20;
  setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
  setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof buf);
#if defined(SO_NOSIGPIPE)
  setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  return s;
}

// ai: The beacons heard: a listener thread for the process, from the first list on
struct Heard { std::string ip, name, layout; uint16_t port = 0; uint32_t w = 0, h = 0; double fps = 0; int64_t at = 0; };
class Beacons {
 public:
  static Beacons& get() { static Beacons b; return b; }
  // ai: the phones heard lately; the first call waits a beacon's period for them
  std::vector<Heard> heard(std::vector<std::string>* notes) {
    bool first;
    {
      std::lock_guard<std::mutex> l(mu_);
      first = !started_;
      if (!started_) { started_ = true; thread_ = std::thread([this] { listen(); }); thread_.detach(); }
    }
    if (first) std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    std::lock_guard<std::mutex> l(mu_);
    if (!error_.empty() && notes) notes->push_back("phone: " + error_);
    std::vector<Heard> v;
    const int64_t now = steadyNs();
    for (const auto& [k, h] : seen_)
      if (now - h.at < BEACON_LIVE_NS) v.push_back(h);
    return v;
  }
  bool find(const std::string& ip, uint16_t port, Heard& out) {
    std::lock_guard<std::mutex> l(mu_);
    auto it = seen_.find(ip + ":" + std::to_string(port));
    if (it == seen_.end()) return false;
    out = it->second;
    return true;
  }

 private:
  void listen() {
    if (!netUp()) { std::lock_guard<std::mutex> l(mu_); error_ = "no sockets"; return; }
    Sock s = socket(AF_INET, SOCK_DGRAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof one);
#if defined(SO_REUSEPORT)
    setsockopt(s, SOL_SOCKET, SO_REUSEPORT, &one, sizeof one);
#endif
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(stream::BEACON_PORT);
    if (s == NONE || bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
      std::lock_guard<std::mutex> l(mu_);
      error_ = "cannot listen for phones on UDP port " + std::to_string(stream::BEACON_PORT);
      if (s != NONE) closeSock(s);
      return;
    }
    char buf[512];
    for (;;) {
      sockaddr_in from{};
      socklen_t fl = sizeof from;
      const int n = recvfrom(s, buf, sizeof buf - 1, 0, reinterpret_cast<sockaddr*>(&from), &fl);
      if (n <= 0) continue;
      buf[n] = 0;
      std::istringstream in(buf);
      std::string magic, word;
      int proto = 0;
      in >> magic >> proto;
      if (magic != "LIZARD-STREAM" || proto != static_cast<int>(stream::PROTO)) continue;
      Heard h;
      h.ip = inet_ntoa(from.sin_addr);
      while (in >> word) {
        const size_t eq = word.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = word.substr(0, eq), v = word.substr(eq + 1);
        if (k == "port") h.port = static_cast<uint16_t>(std::atoi(v.c_str()));
        else if (k == "w") h.w = static_cast<uint32_t>(std::atoi(v.c_str()));
        else if (k == "h") h.h = static_cast<uint32_t>(std::atoi(v.c_str()));
        else if (k == "fps") h.fps = std::atof(v.c_str());
        else if (k == "layout") h.layout = v;
        else if (k == "name") { h.name = v; std::string rest; std::getline(in, rest); h.name += rest; break; }   // ai: the name last, spaces in it
      }
      if (!h.port) continue;
      h.at = steadyNs();
      std::lock_guard<std::mutex> l(mu_);
      seen_[h.ip + ":" + std::to_string(h.port)] = h;
    }
  }
  std::mutex mu_;
  bool started_ = false;
  std::thread thread_;
  std::string error_;
  std::map<std::string, Heard> seen_;
};

// ai: A phone's hello at ip:port within ms (stream.h HELLO): its name and crop; false where nothing there says it
bool hello(const std::string& ip, uint16_t port, int ms, Heard& out) {
  std::string why;
  const Sock s = dial(ip, port, ms, why);
  if (s == NONE) return false;
  std::atomic<bool> never{false};
  uint32_t head[2] = {0, 0};
  std::string h;
  const bool ok = ready(s, ms) && recvAll(s, head, sizeof head, never) && head[0] == stream::HELLO && head[1] <= 65536 &&
                  (h.resize(head[1]), recvAll(s, h.data(), h.size(), never));
  closeSock(s);
  if (!ok) return false;
  const json j = json::parse(h, nullptr, false);
  if (j.is_discarded() || !j.is_object() || j.value("proto", 0) != static_cast<int>(stream::PROTO)) return false;
  out.ip = ip;
  out.port = port;
  out.name = j.value("name", std::string());
  out.layout = j.value("layout", std::string());
  return true;
}

// ai: This PC's IPv4 gateways, one each: a phone's USB tether among them (the phone routes the tether)
std::vector<std::string> gateways() {
  std::vector<std::string> out;
  auto add = [&](const std::string& ip) { if (!ip.empty() && ip != "0.0.0.0" && std::find(out.begin(), out.end(), ip) == out.end()) out.push_back(ip); };
#if defined(_WIN32)
  ULONG len = 16 << 10;
  std::vector<unsigned char> buf;
  ULONG r = ERROR_BUFFER_OVERFLOW;
  for (int k = 0; k < 3 && r == ERROR_BUFFER_OVERFLOW; k++) {
    buf.resize(len);
    r = GetAdaptersAddresses(AF_INET, GAA_FLAG_INCLUDE_GATEWAYS | GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                             nullptr, reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()), &len);
  }
  if (r != NO_ERROR) return out;
  for (auto* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data()); a; a = a->Next) {
    if (a->OperStatus != IfOperStatusUp) continue;
    for (auto* g = a->FirstGatewayAddress; g; g = g->Next)
      if (g->Address.lpSockaddr && g->Address.lpSockaddr->sa_family == AF_INET)
        add(inet_ntoa(reinterpret_cast<sockaddr_in*>(g->Address.lpSockaddr)->sin_addr));
  }
#elif defined(__APPLE__)
  int mib[6] = {CTL_NET, PF_ROUTE, 0, AF_INET, NET_RT_FLAGS, RTF_GATEWAY};
  size_t len = 0;
  if (sysctl(mib, 6, nullptr, &len, nullptr, 0) != 0 || !len) return out;
  std::vector<char> buf(len);
  if (sysctl(mib, 6, buf.data(), &len, nullptr, 0) != 0) return out;
  for (char* p = buf.data(); p + sizeof(rt_msghdr) <= buf.data() + len;) {
    const auto* rtm = reinterpret_cast<const rt_msghdr*>(p);
    if (rtm->rtm_msglen == 0) break;
    const char* sp = p + sizeof(rt_msghdr);
    for (int i = 0; i < RTAX_MAX && sp < p + rtm->rtm_msglen; i++) {
      if (!(rtm->rtm_addrs & (1 << i))) continue;
      const auto* sa = reinterpret_cast<const sockaddr*>(sp);
      if (i == RTAX_GATEWAY && sa->sa_family == AF_INET) add(inet_ntoa(reinterpret_cast<const sockaddr_in*>(sa)->sin_addr));
      sp += sa->sa_len ? ((sa->sa_len + sizeof(uint32_t) - 1) & ~(sizeof(uint32_t) - 1)) : sizeof(uint32_t);   // ai: Darwin's ROUNDUP
    }
    p += rtm->rtm_msglen;
  }
#else
  std::ifstream f("/proc/net/route");
  std::string line;
  std::getline(f, line);   // ai: the header
  while (std::getline(f, line)) {
    std::istringstream in(line);
    std::string iface, dst, gw;
    unsigned flags = 0;
    in >> iface >> dst >> gw >> std::hex >> flags;
    if (!(flags & 0x2)) continue;   // ai: RTF_GATEWAY
    in_addr a{};
    a.s_addr = static_cast<uint32_t>(std::strtoul(gw.c_str(), nullptr, 16));   // ai: the kernel's bytes, network order
    if (a.s_addr) add(inet_ntoa(a));
  }
#endif
  return out;
}

class TcpWire : public Wire {
 public:
  explicit TcpWire(Sock s) : s_(s) {}
  ~TcpWire() override { closeSock(s_); }
  bool send(const void* p, size_t n) override { return !cut_ && sendAll(s_, p, n); }
  bool recv(void* p, size_t n, const std::atomic<bool>& stop, int64_t deadlineNs) override { return !cut_ && recvAll(s_, p, n, stop, deadlineNs); }
  void cut() override { cut_ = true; shutSock(s_); }

 private:
  const Sock s_;
  std::atomic<bool> cut_{false};
};

// ai: The phone's hello on w within ms, its JSON in out: on TCP the connection's first message. On USB asked for
// ai: (OPEN) and found past what the accessory still held of an earlier session (a frame partway, a write the phone
// ai: could not finish): a header of type 1 and a length up to 64 KiB whose payload parses as this stream's hello,
// ai: read a byte at a time so nothing past it is taken from the frames after it
bool readHello(Wire& w, bool usb, int ms, json& out, std::string& why, std::mutex* sendMu = nullptr) {
  const int64_t until = steadyNs() + int64_t(ms) * 1'000'000;
  std::atomic<bool> never{false};
  auto ours = [&](const uint8_t* p, size_t n) {
    out = json::parse(p, p + n, nullptr, false);
    return !out.is_discarded() && out.is_object() && out.value("proto", 0) == static_cast<int>(stream::PROTO);
  };
  if (!usb) {
    uint32_t head[2] = {0, 0};
    std::string body;
    if (!w.recv(head, sizeof head, never, until) || head[0] != stream::HELLO || head[1] > 65536) { why = "the phone sent no hello"; return false; }
    body.resize(head[1]);
    if (head[1] && !w.recv(body.data(), body.size(), never, until)) { why = "the phone's hello was cut"; return false; }
    if (!ours(reinterpret_cast<const uint8_t*>(body.data()), body.size())) { why = "the phone's hello is not this stream's"; return false; }
    return true;
  }
  const uint32_t open[2] = {stream::OPEN, 0};
  {
    std::unique_lock<std::mutex> l;
    if (sendMu) l = std::unique_lock<std::mutex>(*sendMu);   // ai: a link whose sender runs: one message at a time
    if (!w.send(open, sizeof open)) { why = "the phone's accessory took nothing"; return false; }
  }
  std::vector<uint8_t> b;
  size_t at = 0;   // ai: the next place a hello may start
  for (;;) {
    uint8_t c;
    if (!w.recv(&c, 1, never, until)) { why = "the phone sent no hello (on the phone: LIZARD on Receive, its Decoder PC)"; return false; }
    b.push_back(c);
    while (at + 9 <= b.size()) {
      uint32_t t, n;
      std::memcpy(&t, &b[at], 4);
      std::memcpy(&n, &b[at + 4], 4);
      if (t != stream::HELLO || n < 2 || n > 65536 || b[at + 8] != '{') { at++; continue; }
      if (at + 8 + n > b.size()) break;   // ai: more of it to come
      if (ours(&b[at + 8], n)) return true;
      at++;
    }
    // ai: what no hello can start in any more, let go
    if (at > (1u << 20)) { b.erase(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(at)); at = 0; }
  }
}

// ai: A phone as the camera. One link a camera (2026-10-27): opened by the first start (on USB at the open, its hello
// ai: naming the phone and its crop) and kept until close, so either end's Start and Stop follow the other's (stream.h
// ai: CAMERA): start says the phone's camera on, stop says it off once the stop has lasted CAMERA_OFF_MS (the session's
// ai: own stop and start, a reconfigure, says nothing), and the phone's own switches are counted in stats() (asks,
// ai: askOn) for Receive to follow. While stopped the link's frames go nowhere; its stats a second go on, so the
// ai: phone knows a receiver is there. Either end silent QUIET_MS is gone: here the stream ends (Receive stops).
class PhoneCamera : public Camera {
 public:
  PhoneCamera(std::string ip, uint16_t port) : ip_(std::move(ip)), port_(port) {
    info_.id = "phone:" + ip_ + ":" + std::to_string(port_);
    info_.backend = "phone";
    Heard h;
    if (Beacons::get().find(ip_, port_, h)) heard_ = h;
    else if (hello(ip_, port_, 1500, h)) heard_ = h;   // ai: not heard (a typed address, a gateway's): its hello names it
    info_.name = heard_.name.empty() ? "Phone" : heard_.name;
    info_.detail = (heard_.layout.empty() ? std::string() : heard_.layout + ", ") + ip_;
  }
  // ai: a phone on USB (aoa.h): the link made here, its hello naming the phone and its crop
  static std::unique_ptr<PhoneCamera> overUsb(const std::string& id, std::string& why) {
    std::unique_ptr<PhoneCamera> c(new PhoneCamera());
    c->usb_ = true;
    c->info_.id = id;
    c->info_.backend = "phone";
    if (!c->link(why)) return nullptr;
    if (c->info_.name.empty()) c->info_.name = c->usbName_.empty() ? "Phone" : c->usbName_;
    c->info_.detail = c->layout_ + ", USB";
    return c;
  }
  ~PhoneCamera() override { unlink(); }
  const Info& info() const override { return info_; }
  // ai: one mode, the frames' (the beacon's last word on them; 0 x 0 before the phone sent any)
  std::vector<Mode> modes() override {
    Heard h;
    if (!usb_ && Beacons::get().find(ip_, port_, h)) heard_ = h;
    Mode m;
    m.width = heard_.w;
    m.height = heard_.h;
    m.fps = m.fpsMin = heard_.fps > 0 ? heard_.fps : 60;
    m.format = "GREY";
    m.pix = Pix::Grey8;
    return {m};
  }
  std::vector<Control> controls() override { return {}; }
  bool set(const std::string&, double, bool, std::string& why) override { why = "the phone's camera is set on the phone"; return false; }

  bool start(const Mode&, Sink sink, Ended ended, std::string& why) override {
    if (!link(why)) return false;
    {
      std::lock_guard<std::mutex> l(mu_);
      const uint64_t asks = st_.asks;
      const bool askOn = st_.askOn;
      st_ = Stats{};
      st_.fps = heard_.fps;
      st_.asks = asks;
      st_.askOn = askOn;
    }
    {
      std::lock_guard<std::mutex> l(sinkMu_);
      sink_ = std::move(sink);
      ended_ = std::move(ended);
      started_ = true;
    }
    offAt_ = 0;
    if (said_ != 1 && sayCamera(true)) said_ = 1;
    return true;
  }

  // ai: what this end's receiver reads, back to the phone (the link's sender, back()): kept for the link's life
  void readBack(Series series, StatsJson stats) override {
    std::lock_guard<std::mutex> l(cbMu_);
    series_ = std::move(series);
    stats_ = std::move(stats);
  }

  void stop() override {
    {
      std::lock_guard<std::mutex> l(sinkMu_);
      if (!started_) return;
      started_ = false;
      sink_ = nullptr;
      ended_ = nullptr;
    }
    offAt_ = steadyNs() + int64_t(stream::CAMERA_OFF_MS) * 1'000'000;
  }

  Stats stats() override {
    std::lock_guard<std::mutex> l(mu_);
    return st_;
  }

 private:
  PhoneCamera() : port_(0) {}

  // ai: the hello's crop and name
  void takeHello(const json& h) {
    layout_ = h.value("layout", std::string("1:1"));
    if (h.contains("name") && h["name"].is_string() && !h["name"].get<std::string>().empty()) info_.name = h["name"].get<std::string>();
  }

  // ai: The link up, or why not: TCP dialled, or the accessory opened (switched into accessory mode where it is not one
  // ai: yet); its hello read; the reader and the link's sender started. A link that went is made again.
  bool link(std::string& why) {
    if (w_ && !gone_) return true;
    unlink();
    json h;
    if (usb_) {
      std::string name;
      w_ = aoa::open(info_.id, name, why);
      if (!w_ || !readHello(*w_, true, 4000, h, why)) { w_.reset(); return false; }
      usbName_ = name;
    } else {
      const Sock s = dial(ip_, port_, 3000, why);
      if (s == NONE) return false;
      w_ = std::make_unique<TcpWire>(s);
      if (!readHello(*w_, false, 4000, h, why)) { w_.reset(); why += " at " + ip_; return false; }
    }
    takeHello(h);
    closing_ = false;
    gone_ = false;
    offset_ = 0;
    offsetSet_ = false;
    phoneOn_ = -1;
    said_ = -1;
    offAt_ = 0;
    reader_ = std::thread([this] { read(); });
    back_ = std::thread([this] { back(); });
    return true;
  }

  // ai: the link let go; a stop not yet said (a close straight after it: another camera, Receive left) said first
  void unlink() {
    if (w_ && !gone_ && offAt_ && said_ != 0) sayCamera(false);
    closing_ = true;
    if (w_) w_->cut();
    if (reader_.joinable()) reader_.join();
    if (back_.joinable()) back_.join();
    w_.reset();
  }

  // ai: a message whole in one send (one transfer on USB); every sender's sends one at a time
  bool say(uint32_t type, const void* p, size_t n) {
    std::lock_guard<std::mutex> l(sendMu_);
    const uint32_t head[2] = {type, static_cast<uint32_t>(n)};
    msg_.resize(sizeof head + n);
    std::memcpy(msg_.data(), head, sizeof head);
    if (n) std::memcpy(msg_.data() + sizeof head, p, n);
    return w_ && w_->send(msg_.data(), msg_.size());
  }
  bool sayCamera(bool on) {
    const uint32_t v = on ? 1 : 0;
    return say(stream::CAMERA, &v, sizeof v);
  }

  // ai: the phone's camera as it says: a change after the first word is the phone's own Start or Stop
  void phone(bool on) {
    const int was = phoneOn_.exchange(on ? 1 : 0);
    if (was == -1 || was == (on ? 1 : 0)) return;
    std::lock_guard<std::mutex> l(mu_);
    st_.asks++;
    st_.askOn = on;
  }

  // ai: The link's sender: the receiver's series (each row's ms moved from this clock to the phone camera's by the
  // ai: stream's offset) every 10 ms, its stats a second ({} before a start hands them over), and a stop's off once due
  void back() {
    double since = -1;
    int64_t lastStats = 0;
    std::vector<double> out;
    while (!closing_ && !gone_) {
      {
        std::lock_guard<std::mutex> l(cbMu_);
        if (series_ && offsetSet_) {
          const std::vector<double> v = series_(since);
          const size_t n = v.size() > 1 ? (v.size() - 1) / 8 : 0;
          if (n) {
            since = v[1 + 8 * (n - 1)];
            out.assign(v.begin(), v.begin() + 1 + 8 * n);
            const double off = offset_ / 1e6;
            for (size_t i = 0; i < n; i++) out[1 + 8 * i] -= off;
            if (!say(stream::SERIES, out.data(), out.size() * sizeof(double))) break;
          }
        }
        const int64_t now = steadyNs();
        if (now - lastStats >= 1'000'000'000) {
          lastStats = now;
          const std::string j = stats_ ? stats_() : std::string("{}");
          if (!say(stream::STATS, j.data(), j.size())) break;
        }
      }
      const int64_t due = offAt_;
      if (due && steadyNs() >= due && !started_) {
        offAt_ = 0;
        if (said_ != 0 && sayCamera(false)) said_ = 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void read() {
    std::vector<uint8_t> b;
    uint64_t lastSeq = 0, winSeq = 0;
    int64_t winAt = 0;
    std::string why = "the phone went";
    while (!closing_) {
      uint32_t head[2];
      // ai: the phone says something every HEARTBEAT_MS (stream.h): QUIET_MS of nothing, it has gone (its app closed with
      // ai: the cable in, which the accessory never says)
      const int64_t until = steadyNs() + int64_t(stream::QUIET_MS) * 1'000'000;
      if (!w_->recv(head, sizeof head, closing_, until)) {
        if (steadyNs() >= until) why = "the phone went quiet";
        break;
      }
      if (head[0] < stream::HELLO || head[0] > stream::CAMERA || head[1] > stream::MAX_PAYLOAD) {
        if (!usb_) { why = "the phone's stream broke"; break; }
        // ai: the accessory's bytes broke (a write the phone could not finish): read on from a hello asked for
        json h;
        if (!readHello(*w_, true, 4000, h, why, &sendMu_)) break;
        continue;
      }
      b.resize(head[1]);
      if (head[1] && !w_->recv(b.data(), head[1], closing_)) break;
      if (head[0] == stream::CAMERA && head[1] >= 4) {
        uint32_t v;
        std::memcpy(&v, b.data(), 4);
        phone(v != 0);
        continue;
      }
      // ai: a hello mid-link (the phone's session begun again) changes nothing here
      if (head[0] != stream::FRAME || head[1] < stream::FRAME_HEAD) continue;
      uint64_t seq;
      int64_t ns;
      uint32_t w, h;
      std::memcpy(&seq, b.data(), 8);
      std::memcpy(&ns, b.data() + 8, 8);
      std::memcpy(&w, b.data() + 16, 4);
      std::memcpy(&h, b.data() + 20, 4);
      std::lock_guard<std::mutex> k(sinkMu_);
      if (!started_ || !sink_) continue;
      if (static_cast<uint64_t>(w) * h != head[1] - stream::FRAME_HEAD) {
        std::lock_guard<std::mutex> l(mu_);
        st_.corrupt++;
        continue;
      }
      // ai: the phone camera's clock onto this one's by one offset for the link, so back() can take it off again
      if (!offsetSet_) { offset_ = steadyNs() - ns; offsetSet_ = true; }
      {
        std::lock_guard<std::mutex> l(mu_);
        st_.frames++;
        if (lastSeq && seq > lastSeq + 1) st_.dropped += seq - lastSeq - 1;   // ai: the phone's: frames not sent (the link's room)
        // ai: the phone camera's own rate, from its count a second at a time (the session's agreedFps; frames that
        // ai: arrive under it are the link's: ReceiveState's hint)
        const int64_t now = steadyNs();
        if (!winAt) { winAt = now; winSeq = seq; }
        else if (now - winAt >= 1'000'000'000) { st_.fps = (seq - winSeq) * 1e9 / double(now - winAt); winAt = now; winSeq = seq; }
      }
      lastSeq = seq;
      Frame f;
      f.luma = b.data() + stream::FRAME_HEAD;
      f.width = w;
      f.height = h;
      f.stride = w;
      f.ns = ns + offset_;
      f.seq = seq;
      sink_(f);
    }
    gone_ = true;
    Ended e;
    {
      std::lock_guard<std::mutex> l(sinkMu_);
      if (started_) {
        e = std::move(ended_);
        started_ = false;
        sink_ = nullptr;
        ended_ = nullptr;
      }
    }
    if (!closing_ && e) e(why);
  }

  const std::string ip_;
  const uint16_t port_;
  bool usb_ = false;
  Info info_;
  Heard heard_;
  std::string layout_ = "1:1";
  std::string usbName_;
  std::unique_ptr<Wire> w_;
  std::atomic<bool> closing_{false}, gone_{false}, started_{false};
  std::atomic<int64_t> offset_{0}, offAt_{0};   // ai: offAt_: when a stop says the phone's camera off (0 none due)
  std::atomic<bool> offsetSet_{false};
  std::atomic<int> phoneOn_{-1}, said_{-1};     // ai: the phone's camera as it last said, and as this end last said
  std::thread reader_, back_;
  std::mutex mu_;
  Stats st_;
  std::mutex sinkMu_;   // ai: the sink and ended, which a frame is handed to under it
  Sink sink_;
  Ended ended_;
  std::mutex cbMu_;
  Series series_;
  StatsJson stats_;
  std::mutex sendMu_;
  std::vector<uint8_t> msg_;
};

}  // namespace

std::vector<Info> phoneList(std::vector<std::string>* notes) {
  std::vector<Info> v;
  std::vector<Heard> all = Beacons::get().heard(notes);
  // ai: each gateway not heard asked at once, in parallel, a short wait each (a gateway that is a router refuses at
  // ai: once or times out)
  std::vector<std::string> ask;
  for (const auto& g : gateways())
    if (std::none_of(all.begin(), all.end(), [&](const Heard& h) { return h.ip == g; })) ask.push_back(g);
  std::vector<Heard> got(ask.size());
  std::vector<char> ok(ask.size(), 0);
  std::vector<std::thread> t;
  for (size_t i = 0; i < ask.size(); i++) t.emplace_back([&, i] { ok[i] = hello(ask[i], stream::TCP_PORT, 700, got[i]) ? 1 : 0; });
  for (auto& x : t) x.join();
  for (size_t i = 0; i < ask.size(); i++) if (ok[i]) all.push_back(got[i]);
  for (const auto& h : all) {
    Info i;
    i.id = "phone:" + h.ip + ":" + std::to_string(h.port);
    i.name = h.name.empty() ? "Phone" : h.name;
    i.backend = "phone";
    i.detail = (h.layout.empty() ? std::string() : h.layout + ", ") + h.ip;
    v.push_back(std::move(i));
  }
  for (const auto& f : aoa::list(notes)) {
    Info i;
    i.id = f.id;
    i.name = f.name;
    i.backend = "phone";
    i.detail = f.detail;
    v.push_back(std::move(i));
  }
  return v;
}

std::string phoneStamp() {
  std::string s;
  for (const auto& h : Beacons::get().heard(nullptr)) s += h.ip + ":" + std::to_string(h.port) + ";";
  return s + aoa::stamp();
}

std::unique_ptr<Camera> phoneOpen(const std::string& id, std::string& why) {
  if (id.rfind("phone:usb:", 0) == 0) return PhoneCamera::overUsb(id, why);
  // ai: phone:<ipv4>:<port>
  const size_t a = id.find(':'), b = id.rfind(':');
  if (a == std::string::npos || b == a) { why = "not a phone's id: " + id; return nullptr; }
  const std::string ip = id.substr(a + 1, b - a - 1);
  const int port = std::atoi(id.c_str() + b + 1);
  if (ip.empty() || port <= 0 || port > 65535) { why = "not a phone's id: " + id; return nullptr; }
  return std::make_unique<PhoneCamera>(ip, static_cast<uint16_t>(port));
}

}  // namespace lizard::cam
