// ai: The "pc" decoder (stream.h): the phone's side of the stream, a Receiver that decodes nothing.
#include "stream.h"
#include "crops.h"
#include "develop.h"
#include "json.hpp"

#include <stdexcept>

#if defined(_WIN32)

namespace lizard {
std::unique_ptr<Receiver> makeStreamReceiver(const ReceiverConfig&, std::function<void(uint64_t)>) {
  throw std::runtime_error("the PC stream serves from a phone; this system has no server for it");
}
bool streamAccessory(int, std::function<void(const std::string&)>) { return false; }
bool accessoryOpen() { return false; }
}  // namespace lizard

#else

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lizard {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

#if defined(MSG_NOSIGNAL)
constexpr int SEND_FLAGS = MSG_NOSIGNAL;
#else
constexpr int SEND_FLAGS = 0;   // ai: macOS: SO_NOSIGPIPE on the socket instead
#endif

bool sendAll(int fd, const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  while (n) {
    const ssize_t k = send(fd, b, n, SEND_FLAGS);
    if (k <= 0) return false;
    b += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

// ai: n bytes, or false at the stream's end or error; gives up when stop turns true (polled every 200 ms), or after
// ai: quietMs with no byte where quietMs > 0
bool recvAll(int fd, void* p, size_t n, const std::atomic<bool>& stop, int quietMs = 0) {
  auto* b = static_cast<uint8_t*>(p);
  int idle = 0;
  while (n) {
    pollfd q{fd, POLLIN, 0};
    const int r = poll(&q, 1, 200);
    if (stop) return false;
    if (r == 0) {
      if (quietMs > 0 && (idle += 200) >= quietMs) return false;
      continue;
    }
    if (r < 0) return false;
    const ssize_t k = recv(fd, b, n, 0);
    if (k <= 0) return false;
    b += k;
    n -= static_cast<size_t>(k);
    idle = 0;
  }
  return true;
}

// ai: A PC silent this long has gone: it sends its stats a second (lizard-desktop/native/camera/phone.cpp readBack),
// ai: and an accessory's session has no other end when the PC's program goes with the cable in
constexpr int QUIET_MS = 5000;

class StreamReceiver;
// ai: the receiver that takes the accessory's sessions (the newest made) and the bridge: a bridge offers under gMu and
// ai: a receiver leaves under it, so no offer reaches one gone; gCv: a receiver came, or a bridge ended
std::mutex gMu;
std::condition_variable gCv;
StreamReceiver* gServer = nullptr;

std::string deviceName() {
#if defined(__ANDROID__)
  char v[PROP_VALUE_MAX] = "";
  __system_property_get("ro.product.model", v);
  if (v[0]) return v;
#endif
  char h[256] = "";
  if (gethostname(h, sizeof h - 1) == 0 && h[0]) return h;
  return "phone";
}

class StreamReceiver : public Receiver {
 public:
  StreamReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release)
      : layout_(c.layout == "2:1" ? "2:1" : "1:1"), log_(c.log), release_(std::move(release)), name_(deviceName()) {
    const char* p = std::getenv("LIZ_STREAM_PORT");
    port_ = p && std::atoi(p) > 0 ? static_cast<uint16_t>(std::atoi(p)) : stream::TCP_PORT;
    lfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd_ < 0) throw std::runtime_error(std::string("the PC stream has no socket: ") + std::strerror(errno));
    int one = 1;
    setsockopt(lfd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(port_);
    if (bind(lfd_, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0 || listen(lfd_, 1) != 0) {
      const std::string e = std::strerror(errno);
      close(lfd_);
      throw std::runtime_error("the PC stream cannot listen on port " + std::to_string(port_) + ": " + e);
    }
    say("stream: serving frames on port " + std::to_string(port_) + " as " + name_ + ", " + layout_);
    net_ = std::thread([this] { serve(); });
    writer_ = std::thread([this] { write(); });
    beacon_ = std::thread([this] { announce(); });
    {
      std::lock_guard<std::mutex> l(gMu);
      gServer = this;
    }
    gCv.notify_all();
  }

  ~StreamReceiver() override {
    {
      std::lock_guard<std::mutex> l(gMu);
      if (gServer == this) gServer = nullptr;
    }
    stop_ = true;
    cv_.notify_all();
    {
      std::lock_guard<std::mutex> l(mu_);
      if (cfd_ >= 0) shutdown(cfd_, SHUT_RDWR);
    }
    if (net_.joinable()) net_.join();
    if (writer_.joinable()) writer_.join();
    if (beacon_.joinable()) beacon_.join();
    if (lfd_ >= 0) close(lfd_);
    for (const int fd : offered_) close(fd);
  }

  // ai: an accessory's session (the bridge's, under gMu), served as a connection is once the one in hand goes
  void offer(int fd) {
    std::lock_guard<std::mutex> l(mu_);
    offered_.push_back(fd);
  }

  // ai: The share_ of the camera's frames the link carries, spread evenly by a running credit (2026-10-27; the transfer
  // ai: is fountain coded, so any share reads, 58 of 60 as well as 1 of 2; 1 of every k until then): a frame
  // ai: whose credit comes to a whole one has its crop copied as the frame to send, the camera's buffer released at once;
  // ai: the rest skipped. A frame the link still had no room for when the next came is
  // ai: replaced by it (dropped, counted). seq is the camera's count, so the PC reads the gaps. Nothing kept while no
  // ai: PC is connected. A frame to develop (CameraFrame::raw: P010, RAW) is cropped first and only the crop developed,
  // ai: only where it is sent (ingest/develop.h developLuma), so a skipped frame costs no develop.
  void push(const CameraFrame& f) override {
    if ((!f.luma && !f.raw) || !f.width || !f.height || (f.luma && f.stride < f.width)) { release_(f.tag); return; }
    const Crops k = cropsOf(f.width, f.height, layout_ == "2:1");
    const bool wide = f.width >= f.height;
    const uint32_t w = wide ? k.n * k.side : k.side, h = wide ? k.side : k.n * k.side;
    {
      std::lock_guard<std::mutex> l(mu_);
      arrived_++;
      lastW_ = w; lastH_ = h;
      pace();
      if (cfd_ >= 0) credit_ += share_;
      if (cfd_ >= 0 && credit_ >= 1.0 - 1e-9) {
        credit_ -= 1.0;
        if (pending_) dropped_++;
        next_.resize(static_cast<size_t>(w) * h);
        if (f.raw) developLuma(*f.raw, k.x[0], k.y[0], w, h, next_.data(), w);
        else for (uint32_t y = 0; y < h; y++) std::memcpy(&next_[static_cast<size_t>(y) * w], f.luma + static_cast<size_t>(k.y[0] + y) * f.stride + k.x[0], w);
        nextW_ = w; nextH_ = h; nextNs_ = f.timestampNs; nextSeq_ = arrived_;
        pending_ = true;
      } else if (cfd_ >= 0) skipped_++;
    }
    release_(f.tag);
    cv_.notify_all();
  }

  // ai: the PC's stats (its receiver's JSON, the file and the rate the PC's), "stream" beside them: this end's
  // ai: figures since the last call; before a PC connects, this end's alone
  std::string stats() override {
    std::lock_guard<std::mutex> l(mu_);
    const auto now = Clock::now();
    const double secs = std::max(1e-3, std::chrono::duration<double>(now - lastStats_).count());
    lastStats_ = now;
    const double sentFps = (sent_ - sentLast_) / secs, mbps = (bytes_ - bytesLast_) / secs / 1e6, inFps = (arrived_ - arrivedLast_) / secs;
    const uint64_t drops = dropped_ - droppedLast_;
    sentLast_ = sent_; bytesLast_ = bytes_; arrivedLast_ = arrived_; droppedLast_ = dropped_;
    json j = json::object();
    if (cfd_ >= 0 && !pcStats_.empty()) {
      j = json::parse(pcStats_, nullptr, false);
      if (j.is_discarded() || !j.is_object()) j = json::object();
    }
    const std::string pcDecoder = j.value("decoder", std::string());
    j["decoder"] = pcDecoder.empty() ? std::string("pc") : "pc: " + pcDecoder;
    if (cfd_ < 0) j["state"] = "waiting for the PC";
    else if (!j.contains("state")) j["state"] = "streaming";
    if (!j.contains("error") || !j["error"].is_string()) j["error"] = "";
    j["stream"] = {{"port", port_}, {"name", name_}, {"peer", peer_}, {"connected", cfd_ >= 0}, {"cameraFps", inFps},
                   {"sentFps", sentFps}, {"MBps", mbps}, {"dropped", drops}, {"share", share_}, {"w", lastW_}, {"h", lastH_}};
    return j.dump();
  }

  // ai: the PC's rows, already on this camera's clock, after sinceMs
  std::vector<double> series(double sinceMs) override {
    std::lock_guard<std::mutex> l(mu_);
    std::vector<double> v{version_};
    for (const auto& r : rows_)
      if (r[0] > sinceMs) v.insert(v.end(), r.begin(), r.end());
    return v;
  }

  void batchCap(int) override {}
  std::string file() override { return ""; }   // ai: the file is the PC's
  void clear() override {}
  bool wantsLuma() const override { return true; }

 private:
  void say(const std::string& s) { if (log_) log_(s); }

  // ai: The skip, a second at a time (under mu_). The link's rate is read only from a second it was full (busy, the
  // ai: share of the second spent in send, past 0.8, or a frame dropped: send blocks once the socket's buffers are
  // ai: full, so the MB/s over the time in send is what it carries), and share_ set then, down at once: the frames sent need 95% of
  // ai: it. Three seconds in a row that kept up raise it: to 90% of the rate known, or, with none known, a tenth more
  // ai: and a frame a second (a probe: the next full second reads the rate); after 30 such seconds the rate is read
  // ai: afresh (a faster link found). 2026-10-27: a rate read from every second, the buffers' slack and a part second in
  // ai: it, took the skip from 1 of 5 to 1 of 2 and back, then stuck at 1 of 8, on a 30 MB/s reader.
  void pace() {
    const auto now = Clock::now();
    const double secs = std::chrono::duration<double>(now - paceAt_).count();
    if (secs < 1.0) return;
    const double camFps = (arrived_ - paceArrived_) / secs;
    const double mbps = (bytes_ - paceBytes_) / secs / 1e6;
    const double busy = std::chrono::duration<double>(std::chrono::nanoseconds(busyNs_ - paceBusy_)).count() / secs;
    const bool drops = dropped_ != paceDropped_;
    paceAt_ = now; paceArrived_ = arrived_; paceBytes_ = bytes_; paceBusy_ = busyNs_; paceDropped_ = dropped_;
    if (cfd_ < 0) { share_ = 1.0; credit_ = 0; easier_ = 0; calm_ = 0; linkMBps_ = 0; return; }
    const double frameMB = static_cast<double>(lastW_) * lastH_ / 1e6;
    if (frameMB <= 0 || camFps <= 0) return;
    if (busy > 0.8 || drops) {
      // ai: what the link carried while the writer sent (never under it: a share is cut only by this, and an over-read
      // ai: cuts nothing; over 0.8 as the floor until the share could fall by any amount, which read a 30 MB/s reader at
      // ai: 13 from seconds a frame was dropped in)
      linkMBps_ = mbps / std::min(1.0, std::max(busy, 0.05));
      easier_ = 0;
      calm_ = 0;
      const double want = std::min(1.0, 0.95 * linkMBps_ / (camFps * frameMB));
      if (want < share_ - 1e-9) {
        share_ = want;
        say("stream: the link carries about " + std::to_string(static_cast<int>(linkMBps_)) + " MB/s: " + std::to_string(static_cast<int>(std::lround(share_ * camFps))) + " of " +
            std::to_string(static_cast<int>(std::lround(camFps))) + " frames a second sent");
      }
      return;
    }
    if (++calm_ >= 30) { calm_ = 0; linkMBps_ = 0; }
    if (share_ < 1.0) {
      const double up = linkMBps_ > 0 ? std::min(1.0, 0.9 * linkMBps_ / (camFps * frameMB)) : std::min(1.0, share_ * 1.1 + 1.0 / camFps);
      if (up > share_ + 1e-9) { if (++easier_ >= 3) { share_ = up; easier_ = 0; } }
      else easier_ = 0;
    }
  }

  // ai: one PC at a time, a TCP connection or an accessory's session (offer, peer "USB"): its hello, then its rows and
  // ai: stats read until it goes
  void serve() {
    while (!stop_) {
      int fd = -1;
      std::string peer = "USB";
      {
        std::lock_guard<std::mutex> l(mu_);
        if (!offered_.empty()) { fd = offered_.front(); offered_.erase(offered_.begin()); }
      }
      if (fd < 0) {
        pollfd q{lfd_, POLLIN, 0};
        if (poll(&q, 1, 200) <= 0) continue;
        sockaddr_in a{};
        socklen_t al = sizeof a;
        fd = accept(lfd_, reinterpret_cast<sockaddr*>(&a), &al);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        char ip[INET_ADDRSTRLEN] = "";
        inet_ntop(AF_INET, &a.sin_addr, ip, sizeof ip);
        peer = ip;
      }
      session(fd, peer);
    }
  }

  void session(int fd, const std::string& peer) {
    const int buf = 4 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
#if defined(SO_NOSIGPIPE)
    const int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    const std::string who = peer == "USB" ? "the PC on USB" : "the PC at " + peer;
    const std::string hello = json{{"proto", stream::PROTO}, {"name", name_}, {"layout", layout_}, {"fps", 0}}.dump();
    const uint32_t head[2] = {stream::HELLO, static_cast<uint32_t>(hello.size())};
    if (!sendAll(fd, head, sizeof head) || !sendAll(fd, hello.data(), hello.size())) { close(fd); return; }
    {
      std::lock_guard<std::mutex> l(mu_);
      cfd_ = fd;
      peer_ = peer;
      pcStats_.clear();
      rows_.clear();
    }
    say("stream: " + who + " connected");
    readBack(fd);
    {
      std::lock_guard<std::mutex> l(mu_);
      cfd_ = -1;
      pending_ = false;
      pcStats_.clear();
    }
    // ai: the writer may be in a send on it: shut down first, closed once the writer has let it go (sending_)
    shutdown(fd, SHUT_RDWR);
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [this] { return !sending_; });
    }
    close(fd);
    say("stream: " + who + " went");
  }

  void readBack(int fd) {
    std::vector<uint8_t> b;
    while (!stop_) {
      uint32_t head[2];
      if (!recvAll(fd, head, sizeof head, stop_, QUIET_MS) || head[1] > stream::MAX_PAYLOAD) return;
      b.resize(head[1]);
      if (head[1] && !recvAll(fd, b.data(), head[1], stop_, QUIET_MS)) return;
      std::lock_guard<std::mutex> l(mu_);
      if (head[0] == stream::STATS) pcStats_.assign(b.begin(), b.end());
      else if (head[0] == stream::SERIES && head[1] >= sizeof(double) && (head[1] - sizeof(double)) % (8 * sizeof(double)) == 0) {
        const auto* d = reinterpret_cast<const double*>(b.data());
        version_ = d[0];
        const size_t n = (head[1] / sizeof(double) - 1) / 8;
        for (size_t i = 0; i < n; i++) {
          std::array<double, 8> r;
          std::memcpy(r.data(), d + 1 + 8 * i, sizeof r);
          if (rows_.empty() || r[0] > rows_.back()[0]) rows_.push_back(r);
        }
        // ai: rows kept for 4 s of the camera's clock (a reader asks at every capture)
        while (!rows_.empty() && rows_.back()[0] - rows_.front()[0] > 4000) rows_.pop_front();
      }
    }
  }

  void write() {
    std::vector<uint8_t> out;
    for (;;) {
      int fd = -1;
      uint32_t w = 0, h = 0;
      int64_t ns = 0;
      uint64_t seq = 0;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return stop_ || (pending_ && cfd_ >= 0); });
        if (stop_) return;
        out.swap(next_);
        w = nextW_; h = nextH_; ns = nextNs_; seq = nextSeq_;
        pending_ = false;
        fd = cfd_;
        sending_ = true;
      }
      uint8_t head[8 + stream::FRAME_HEAD];
      const uint32_t th[2] = {stream::FRAME, static_cast<uint32_t>(stream::FRAME_HEAD + out.size())};
      std::memcpy(head, th, 8);
      std::memcpy(head + 8, &seq, 8);
      std::memcpy(head + 16, &ns, 8);
      std::memcpy(head + 24, &w, 4);
      std::memcpy(head + 28, &h, 4);
      const auto t0 = Clock::now();
      const bool ok = sendAll(fd, head, sizeof head) && sendAll(fd, out.data(), out.size());
      {
        std::lock_guard<std::mutex> l(mu_);
        sending_ = false;
        busyNs_ += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
        if (ok) { sent_++; bytes_ += sizeof head + out.size(); }
      }
      cv_.notify_all();
      if (!ok) shutdown(fd, SHUT_RDWR);   // ai: serve's read ends with it and lets the PC go
    }
  }

  // ai: the beacon a second to each IPv4 broadcast address (the tether's, the Wi-Fi's)
  void announce() {
    const int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { say("stream: no beacon socket"); return; }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);
    auto last = Clock::now();
    uint64_t sentThen = 0;
    while (!stop_) {
      uint32_t w, h;
      double fps;
      {
        std::lock_guard<std::mutex> l(mu_);
        w = lastW_; h = lastH_;
        const double secs = std::chrono::duration<double>(Clock::now() - last).count();
        fps = secs > 0 ? (arrived_ - sentThen) / secs : 0;
        sentThen = arrived_;
        last = Clock::now();
      }
      const std::string msg = "LIZARD-STREAM " + std::to_string(stream::PROTO) + " port=" + std::to_string(port_) + " w=" + std::to_string(w) +
                              " h=" + std::to_string(h) + " fps=" + std::to_string(static_cast<int>(std::lround(fps))) + " layout=" + layout_ + " name=" + name_;
      ifaddrs* all = nullptr;
      if (getifaddrs(&all) == 0) {
        for (ifaddrs* i = all; i; i = i->ifa_next) {
          if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || !(i->ifa_flags & IFF_UP) || (i->ifa_flags & IFF_LOOPBACK)) continue;
          if (!(i->ifa_flags & IFF_BROADCAST) || !i->ifa_broadaddr) continue;
          sockaddr_in to = *reinterpret_cast<sockaddr_in*>(i->ifa_broadaddr);
          to.sin_port = htons(stream::BEACON_PORT);
          sendto(fd, msg.data(), msg.size(), 0, reinterpret_cast<sockaddr*>(&to), sizeof to);
        }
        freeifaddrs(all);
      }
      for (int k = 0; k < 10 && !stop_; k++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    close(fd);
  }

  const std::string layout_;
  std::function<void(const std::string&)> log_;
  std::function<void(uint64_t)> release_;
  const std::string name_;
  uint16_t port_ = stream::TCP_PORT;
  int lfd_ = -1;
  std::atomic<bool> stop_{false};
  std::thread net_, writer_, beacon_;
  std::mutex mu_;
  std::condition_variable cv_;
  int cfd_ = -1;              // ai: the PC's connection, -1 none
  std::string peer_;
  bool pending_ = false, sending_ = false;
  std::vector<uint8_t> next_;
  uint32_t nextW_ = 0, nextH_ = 0, lastW_ = 0, lastH_ = 0;
  int64_t nextNs_ = 0;
  uint64_t nextSeq_ = 0, arrived_ = 0, sent_ = 0, bytes_ = 0, dropped_ = 0, skipped_ = 0;
  // ai: the share of the camera's frames sent (pace), the running credit that spreads them, the seconds in a row that
  // ai: allowed a larger share, and those since the rate was read
  double share_ = 1.0, credit_ = 0;
  int easier_ = 0, calm_ = 0;
  int64_t busyNs_ = 0, paceBusy_ = 0;
  uint64_t paceArrived_ = 0, paceBytes_ = 0, paceDropped_ = 0;
  double linkMBps_ = 0;   // ai: the link's rate as pace last read it (0 none yet)
  Clock::time_point paceAt_ = Clock::now();
  uint64_t arrivedLast_ = 0, sentLast_ = 0, bytesLast_ = 0, droppedLast_ = 0;
  Clock::time_point lastStats_ = Clock::now();
  std::string pcStats_;
  double version_ = 0;
  std::deque<std::array<double, 8>> rows_;
  std::vector<int> offered_;   // ai: the accessory's sessions not yet served
};

// ai: an fd closed when its last holder lets it go: the bridge's end of a session, which the reader writes the PC's
// ai: messages into while the writer may be ending it
struct Fd {
  explicit Fd(int f) : fd(f) {}
  ~Fd() { close(fd); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  const int fd;
};

// ai: The accessory's bridge (stream.h): toPc offers the receiver a session at a time and writes each whole message it
// ai: sends to the accessory; fromPc reads the accessory 16 KB at a time (f_accessory's request), forwards the PC's
// ai: messages into the session, and at an OPEN ends the session, the next beginning with a HELLO. Both threads are
// ai: detached and hold the bridge; the accessory's fd closes with the last.
class Bridge {
 public:
  Bridge(int fd, std::function<void(const std::string&)> log) : acc_(fd), log_(std::move(log)) {}
  ~Bridge() { close(acc_); }
  Bridge(const Bridge&) = delete;
  Bridge& operator=(const Bridge&) = delete;

  static void start(const std::shared_ptr<Bridge>& b) {
    b->say("stream: a USB accessory open");
    std::thread([b] { b->toPc(); }).detach();
    std::thread([b] { b->fromPc(); }).detach();
  }
  bool alive() const { return !dead_; }

 private:
  void say(const std::string& s) { if (log_) log_(s); }

  void toPc() {
    std::vector<uint8_t> m;
    while (!dead_) {
      std::shared_ptr<Fd> s;
      {
        std::unique_lock<std::mutex> l(gMu);
        gCv.wait_for(l, std::chrono::milliseconds(250), [this] { return gServer || dead_; });
        if (dead_ || !gServer) continue;
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
          l.unlock();
          say(std::string("stream: no socket pair for the accessory: ") + std::strerror(errno));
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
        gServer->offer(sv[1]);
        s = std::make_shared<Fd>(sv[0]);
      }
      {
        std::lock_guard<std::mutex> l(mu_);
        sess_ = s;
      }
      // ai: a message written once all of it is in: a session that ends partway drops the rest of that one
      while (!dead_) {
        uint32_t head[2];
        if (!recvAll(s->fd, head, sizeof head, dead_) || head[1] > stream::MAX_PAYLOAD) break;
        m.resize(sizeof head + head[1]);
        std::memcpy(m.data(), head, sizeof head);
        if (head[1] && !recvAll(s->fd, m.data() + sizeof head, head[1], dead_)) break;
        if (!writeAcc(m.data(), m.size())) break;
      }
      {
        std::lock_guard<std::mutex> l(mu_);
        sess_.reset();
      }
      shutdown(s->fd, SHUT_RDWR);   // ai: the receiver's session ends with it
    }
  }

  // ai: false where the accessory failed (the bridge ends). Interrupted, how much went is not known: the rest of the
  // ai: message is dropped and the PC, reading a broken stream, asks for a HELLO (OPEN)
  bool writeAcc(const uint8_t* p, size_t n) {
    while (n) {
      const ssize_t k = ::write(acc_, p, n);
      if (k < 0 && errno == EINTR) { say("stream: a write to the accessory interrupted"); return true; }
      if (k <= 0) { dead_ = true; return false; }
      p += k;
      n -= static_cast<size_t>(k);
    }
    return true;
  }

  void fromPc() {
    std::vector<uint8_t> in, chunk(16384);
    while (!dead_) {
      const ssize_t k = ::read(acc_, chunk.data(), chunk.size());
      if (k < 0 && errno == EINTR) continue;
      if (k <= 0) break;
      in.insert(in.end(), chunk.begin(), chunk.begin() + k);
      size_t at = 0;
      while (in.size() - at >= 8) {
        uint32_t head[2];
        std::memcpy(head, &in[at], sizeof head);
        if (head[1] > stream::MAX_PAYLOAD) { at = in.size(); break; }   // ai: not the stream: what came is dropped
        if (in.size() - at < sizeof head + head[1]) break;
        if (head[0] == stream::OPEN) endSession();
        else forward(&in[at], sizeof head + head[1]);
        at += sizeof head + head[1];
      }
      in.erase(in.begin(), in.begin() + static_cast<std::ptrdiff_t>(at));
    }
    dead_ = true;
    endSession();
    gCv.notify_all();
    say("stream: the USB accessory closed");
  }

  void endSession() {
    std::lock_guard<std::mutex> l(mu_);
    if (sess_) shutdown(sess_->fd, SHUT_RDWR);
  }

  void forward(const uint8_t* p, size_t n) {
    std::shared_ptr<Fd> s;
    {
      std::lock_guard<std::mutex> l(mu_);
      s = sess_;
    }
    if (s && !sendAll(s->fd, p, n)) shutdown(s->fd, SHUT_RDWR);
  }

  const int acc_;
  std::function<void(const std::string&)> log_;
  std::atomic<bool> dead_{false};
  std::mutex mu_;
  std::shared_ptr<Fd> sess_;   // ai: the bridge's end of the session in hand
};

std::weak_ptr<Bridge> gBridge;   // ai: under gMu

}  // namespace

std::unique_ptr<Receiver> makeStreamReceiver(const ReceiverConfig& c, std::function<void(uint64_t)> release) {
  return std::make_unique<StreamReceiver>(c, std::move(release));
}

bool streamAccessory(int fd, std::function<void(const std::string&)> log) {
  if (fd < 0) return false;
  std::lock_guard<std::mutex> l(gMu);
  if (const auto b = gBridge.lock(); b && b->alive()) { close(fd); return false; }
  const auto b = std::make_shared<Bridge>(fd, std::move(log));
  gBridge = b;
  Bridge::start(b);
  return true;
}

bool accessoryOpen() {
  std::lock_guard<std::mutex> l(gMu);
  const auto b = gBridge.lock();
  return b && b->alive();
}

}  // namespace lizard

#endif
