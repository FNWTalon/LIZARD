// ai: The app's sender (sender.h): lizard-web/send-worker.mjs and the page's paint ahead, natively.
#include "sender.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <sstream>

extern "C" {
#include "focus.h"
#include "shake.h"
}

namespace lizard {

namespace {
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int PAYLOAD = 469, ID_BYTES = 4;
bool control(uint32_t id) { return (id >> 18) == 0x3fff; }   // ai: sim/xfer.mjs isControlId
// ai: a string's text fit for the stats JSON (a driver's message may hold quotes or a backslash)
std::string esc(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (static_cast<unsigned char>(c) >= 0x20) o += c; }
  return o;
}
}  // namespace

Sender::Sender(const uint8_t* data, size_t length, const std::string& name, const std::string& type) {
  if (data) xfer_ = std::make_unique<XferTx>(data, length, name, type);
  // ai: the test stream starts at a random id each start, as the page's (lizard-web/send.mjs), so a receiver takes a restarted
  // ai: one as new with no signal
  std::random_device rd;
  nextId_ = (rd() & 0x3fffffffu) | 1u;
  winStart_ = nowMs();
}

Sender::~Sender() { stopAll(); }

void Sender::stopAll() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (producer_.joinable()) producer_.join();
  for (auto& t : painters_) if (t.joinable()) t.join();
  painters_.clear();
}

std::string Sender::configure(const TxFormat& f) {
  stopAll();
  if (f.codes < 1 || f.codes > 2) return "codes are 1 or 2, not " + std::to_string(f.codes);
  if (f.gap < 0 || f.gap > 64) return "the gap is 0 to 64 modules, not " + std::to_string(f.gap);
  // ai: one codec made here to say what the format is (and that it is one); each painter makes its own
  focus_t probe{};
  if (focus_init(&probe, f.n, f.subch, 1, 2.0f, f.span, 0.f, 0, 0, 0, 0, 0, 0)) return "the codec refused LIZARD-" + std::to_string(f.subch) + " at n = " + std::to_string(f.n);
  // ai: the GPU where asked (or auto), its device made once a send; auto keeps it only where two frames of the test
  // ai: stream's blocks come out as the C paints them (gpu/encoder.mjs checkNow's rule), so a driver that returns
  // ai: garbage sends from the CPU instead
  bool gpu = false;
  gpuWhy_.clear();
  if (f.painter != 0) {
    try {
      if (!gpu_) gpu_ = GpuPainter::create(f.assets, nullptr, f.device);
      gpuWhy_ = gpu_->configure(f.n, f.subch, f.span, f.fps, 4, f.codes, f.gap);
      if (gpuWhy_.empty() && f.painter == 2) {
        const int k = std::min(2, gpu_->frames()), T = probe.blocks * f.codes, B = probe.block_bytes;
        std::vector<uint8_t> blocks(static_cast<size_t>(k) * T * B);
        for (int t = 0; t < k * T; t++) {
          uint8_t* o = blocks.data() + static_cast<size_t>(t) * B;
          for (int i = 0; i < ID_BYTES; i++) o[i] = static_cast<uint8_t>(t >> 8 * i);
          stream_fill(static_cast<uint32_t>(t), o + ID_BYTES, PAYLOAD);
        }
        const auto c = gpu_->check(blocks.data(), k);
        if (!c.ok) gpuWhy_ = "its frames are not the C's (border " + std::to_string(c.borderDiff) + " pixels apart, the square up to " + std::to_string(c.squareMax) + " levels)";
      }
    } catch (const std::exception& e) { gpuWhy_ = e.what(); gpu_.reset(); }
    if (gpuWhy_.empty()) gpu = true;
    else if (f.painter == 1) { focus_free(&probe); return "the GPU cannot paint here: " + gpuWhy_; }
  }
  std::lock_guard<std::mutex> l(mu_);
  fmt_ = f;
  fmt_.threads = gpu ? gpu_->frames() : std::max(1, f.threads);
  onGpu_ = gpu;
  gpuDevice_ = gpu ? gpu_->device() : "";
  blocks_ = probe.blocks;
  blockBytes_ = probe.block_bytes;
  side_ = probe.px + 2 * FOCUS_QUIET * probe.pxm;
  gap_ = f.codes > 1 ? f.gap * probe.pxm : 0;
  width_ = f.codes * side_ + (f.codes - 1) * gap_;
  focus_free(&probe);
  label_ = "LIZARD-" + std::to_string(f.subch) + (f.codes > 1 ? " x " + std::to_string(f.codes) : "");
  // ai: what was made or painted under the old format goes; the count goes on from the screen's
  jobs_.clear(); ready_.clear(); dataOf_.clear(); spare_.clear();
  made_ = shown_;
  stop_ = false;
  gen_++;
  producer_ = std::thread([this] { produce(); });
  if (gpu) painters_.emplace_back([this] { paintGpu(); });
  else for (int k = 0; k < fmt_.threads; k++) painters_.emplace_back([this, k] { paint(k); });
  return "";
}

std::string Sender::prepareGpu(const std::string& assets, const std::string& device) {
  if (gpu_) return "";
  try { gpu_ = GpuPainter::create(assets, nullptr, device); return ""; }
  catch (const std::exception& e) { return e.what(); }
}

// ai: Frames made ahead of the screen: enough for every painter to have one and the screen a few in hand.
void Sender::produce() {
  std::unique_lock<std::mutex> l(mu_);
  const int T = blocks_ * fmt_.codes, B = blockBytes_;   // ai: a frame's blocks: its codes' one after the other
  std::vector<uint32_t> ids(T);
  for (;;) {
    const size_t ahead = static_cast<size_t>(2 * fmt_.threads + 3);
    cv_.wait(l, [&] { return stop_ || jobs_.size() + ready_.size() + inFlight_ < ahead; });
    if (stop_) return;
    Job j{made_++, std::vector<uint8_t>(static_cast<size_t>(T) * B), 0};
    if (xfer_) xfer_->frameIds(ids.data(), T);
    else for (int t = 0; t < T; t++) ids[t] = nextId_++;
    for (int t = 0; t < T; t++) {
      uint8_t* o = j.blocks.data() + static_cast<size_t>(t) * B;
      for (int i = 0; i < ID_BYTES; i++) o[i] = static_cast<uint8_t>(ids[t] >> 8 * i);
      if (xfer_) xfer_->block(ids[t], o + ID_BYTES);
      else stream_fill(ids[t], o + ID_BYTES, PAYLOAD);
      if (!xfer_ || !control(ids[t])) j.data++;
    }
    dataOf_[j.seq] = j.data;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
  }
}

// ai: A painter: its own codec, the oldest frame made, the pilots' count its number mod 4 (src/focus.h focus_parity).
void Sender::paint(int) {
  focus_t f{};
  TxFormat fm;
  {
    std::lock_guard<std::mutex> l(mu_);
    fm = fmt_;
  }
  if (focus_init(&f, fm.n, fm.subch, 1, 2.0f, fm.span, 0.f, 0, 0, 0, 0, 0, 0)) {
    std::lock_guard<std::mutex> l(mu_);
    error_ = "a painter's codec failed";
    return;
  }
  focus_fmt_fps(&f, fm.fps);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  const size_t bytes = static_cast<size_t>(width_) * side_ * 4;
  // ai: with two codes each symbol is painted on its own and copied row by row into its place, the gap the margin's
  // ai: colour beside it (its row's last pixel), as lizard-web/send-worker.mjs does
  const int W = side_, codes = fm.codes, gap = gap_;
  const size_t per = static_cast<size_t>(blocks_) * blockBytes_;
  std::vector<uint8_t> part(codes > 1 ? static_cast<size_t>(W) * W * 4 : 0);
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    Job j = std::move(jobs_.front());
    jobs_.pop_front();
    std::vector<uint8_t> out;
    if (!spare_.empty()) { out = std::move(spare_.back()); spare_.pop_back(); }
    inFlight_++;
    l.unlock();
    out.resize(bytes);
    const double t0 = nowMs();
    focus_parity(&f, static_cast<int>(j.seq & 3));
    if (codes == 1) {
      focus_encode(&f, j.blocks.data(), drive.data());
      focus_paint_rgba(&f, drive.data(), out.data());
    } else for (int k = 0; k < codes; k++) {
      focus_encode(&f, j.blocks.data() + k * per, drive.data());
      focus_paint_rgba(&f, drive.data(), part.data());
      const size_t row = static_cast<size_t>(W) * 4, frow = static_cast<size_t>(width_) * 4, x0 = static_cast<size_t>(k) * (W + gap) * 4;
      for (int y = 0; y < W; y++) {
        uint8_t* o = out.data() + y * frow + x0;
        const uint8_t* src = part.data() + y * row;
        std::memcpy(o, src, row);
        if (k + 1 < codes) for (int x = 0; x < gap; x++) std::memcpy(o + row + 4 * x, src + row - 4, 4);
      }
    }
    const double ms = nowMs() - t0;
    l.lock();
    inFlight_--;
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_++;
    if (j.seq >= shown_) ready_[j.seq] = std::move(out);
    cv_.notify_all();
  }
  l.unlock();
  focus_free(&f);
}

// ai: The GPU's painter: the oldest frames made, up to an encode's worth (consecutive, so the first's count and one more
// ai: each is every frame's own), one submission, their grey pixels into ready_.
void Sender::paintGpu() {
  const int R = gpu_->frames();
  std::vector<std::vector<uint8_t>> outs(R);
  std::vector<uint8_t> blocks;
  std::vector<Job> js;
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    const int k = std::min<int>(R, static_cast<int>(jobs_.size()));
    js.clear();
    for (int i = 0; i < k; i++) { js.push_back(std::move(jobs_.front())); jobs_.pop_front(); }
    for (int i = 0; i < k; i++) if (outs[i].empty() && !spare_.empty()) { outs[i] = std::move(spare_.back()); spare_.pop_back(); }
    inFlight_ += k;
    l.unlock();
    const size_t per = js[0].blocks.size();
    blocks.resize(per * k);
    for (int i = 0; i < k; i++) std::memcpy(blocks.data() + per * i, js[i].blocks.data(), per);
    double ms = 0;
    std::string err;
    try { ms = gpu_->encode(blocks.data(), k, static_cast<int>(js[0].seq & 3), outs.data()); }
    catch (const std::exception& e) { err = e.what(); }
    l.lock();
    inFlight_ -= k;
    if (!err.empty()) { error_ = "the GPU painter stopped: " + err; cv_.notify_all(); break; }
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_ += k;
    for (int i = 0; i < k; i++) if (js[i].seq >= shown_) ready_[js[i].seq] = std::move(outs[i]);
    cv_.notify_all();
  }
}

bool Sender::ready() {
  std::lock_guard<std::mutex> l(mu_);
  return ready_.count(shown_) != 0;
}

bool Sender::take(uint8_t* dst, int stride) {
  std::vector<uint8_t> buf;
  int data = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    auto it = ready_.find(shown_);
    if (it == ready_.end()) return false;
    buf = std::move(it->second);
    ready_.erase(it);
    auto d = dataOf_.find(shown_);
    if (d != dataOf_.end()) { data = d->second; dataOf_.erase(d); }
    shown_++;
  }
  const size_t row = static_cast<size_t>(width_) * 4;
  if (buf.size() == static_cast<size_t>(width_) * side_) {
    // ai: a GPU frame: grey to RGBA on the way (the pixel's grey in R, G and B, opaque)
    for (int y = 0; y < side_; y++) {
      uint32_t* d = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(y) * stride);
      const uint8_t* g = buf.data() + static_cast<size_t>(y) * width_;
      for (int x = 0; x < width_; x++) d[x] = 0xff000000u | g[x] * 0x010101u;
    }
  } else for (int y = 0; y < side_; y++) std::memcpy(dst + static_cast<size_t>(y) * stride, buf.data() + y * row, row);
  takenWin_++;
  dataWin_ += data;
  dataAll_ += data;
  {
    std::lock_guard<std::mutex> l(mu_);
    spare_.push_back(std::move(buf));
  }
  cv_.notify_all();
  return true;
}

// ai: The stats a second (the app asks): label, n, subch, fps (asked), shownFps (frames on the screen a second over the
// ai: window), offeredKBs (data blocks shown a second x 469 B, 1000 B a KB, as the page's), paintMs (a frame's encode
// ai: and paint on one painter; on the GPU its ms a frame, an encode's timestamps over its frames), painters (the
// ai: GPU's: frames an encode), painter ("gpu" or "cpu"), device (the GPU's), gpuWhy (why not the GPU where it was
// ai: asked for), ahead (painted, not shown), side, test, and for a file chunks, lap (data blocks a lap), pass (laps
// ai: shown, 1 the first), root; error, the last failure.
std::string Sender::stats() {
  const double now = nowMs(), secs = std::max(1e-3, (now - winStart_) / 1000);
  if (secs >= 0.5) {
    lastFps_ = static_cast<double>(takenWin_.exchange(0)) / secs;
    lastKBs_ = static_cast<double>(dataWin_.exchange(0)) * PAYLOAD / 1000 / secs;
    const uint64_t p = paintedWin_.exchange(0);
    const double ms = paintMsWin_.exchange(0);
    if (p) lastPaintMs_ = ms / static_cast<double>(p);
    winStart_ = now;
  }
  std::lock_guard<std::mutex> l(mu_);
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(1);
  o << "{\"label\":\"" << label_ << "\",\"n\":" << fmt_.n << ",\"subch\":" << fmt_.subch << ",\"fps\":" << fmt_.fps
    << ",\"shownFps\":" << lastFps_ << ",\"offeredKBs\":" << lastKBs_ << ",\"paintMs\":" << lastPaintMs_
    << ",\"painters\":" << fmt_.threads << ",\"painter\":\"" << (onGpu_ ? "gpu" : "cpu") << "\",\"device\":\"" << esc(gpuDevice_)
    << "\",\"gpuWhy\":\"" << esc(gpuWhy_) << "\",\"ahead\":" << ready_.size() << ",\"side\":" << side_ << ",\"width\":" << width_
    << ",\"codes\":" << fmt_.codes << ",\"shown\":" << shown_
    << ",\"test\":" << (xfer_ ? "false" : "true");
  if (xfer_) {
    const uint32_t lap = xfer_->lap();
    o.precision(2);
    o << ",\"chunks\":" << xfer_->chunks() << ",\"lap\":" << lap << ",\"pass\":" << (lap ? 1.0 + static_cast<double>(dataAll_) / lap : 1.0)
      << ",\"root\":\"" << xfer_->rootHex() << "\"";
  }
  o << ",\"error\":\"" << error_ << "\"}";
  return o.str();
}

}  // namespace lizard
