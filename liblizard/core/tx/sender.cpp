// ai: The app's sender (sender.h): lizard-web/send-worker.mjs and the page's paint ahead, natively.
#include "sender.h"

#include "wg.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
  if (uploader_.joinable()) uploader_.join();
}

// ai: A painter's codec of the format: the profile's tiers where one is set (focus_init_tiers), else the one rate.
int Sender::makeCodec(focus_t* f, const TxFormat& fm) const {
  if (tiers_ > 0) return focus_init_tiers(f, fm.n, tier_, tiers_, 2.0f, fm.span, 0.f, 0, 0, 0, 0, 0, 0);
  return focus_init(f, fm.n, fm.subch, 1, 2.0f, fm.span, 0.f, 0, 0, 0, 0, 0, 0);
}

std::string Sender::configure(const TxFormat& given) {
  stopAll();
  if (given.codes < 1 || given.codes > 2) return "codes are 1 or 2, not " + std::to_string(given.codes);
  if (given.gap < 0 || given.gap > 64) return "the gap is 0 to 64 modules, not " + std::to_string(given.gap);
  // ai: the rate profile (TxFormat.tiers): the frame's sub-channels are its sum, the picture follows, the C paints
  TxFormat f = given;
  {
    std::lock_guard<std::mutex> l(mu_);
    tiers_ = 0; tiersLabel_.clear();
    if (!f.tiers.empty()) {
      char label[160];
      int subch = 0;
      const int tiers = focus_tiers_parse(f.tiers.c_str(), tier_, &subch, label, sizeof label);
      if (tiers <= 0) return std::string("the rate profile was refused: ") + label;
      tiers_ = tiers; tiersLabel_ = label;
      f.subch = subch; f.n = focus_n_for(subch);
      f.painter = 0;
    }
  }
  // ai: one codec made here to say what the format is (and that it is one); each painter makes its own
  focus_t probe{};
  if (makeCodec(&probe, f)) return "the codec refused LIZARD-" + std::to_string(f.subch) + " at n = " + std::to_string(f.n);
  // ai: The device and the painter are made here where no prepare did: the frame ring lives in the painter whatever
  // ai: paints (2026-10-07), so a sender needs a Vulkan device. The GPU paints where asked (or auto); auto keeps it only
  // ai: where two frames of the test stream's blocks come out as the C paints them (gpu/encoder.mjs checkNow's rule), so
  // ai: a driver that returns garbage sends from the CPU instead, into the same ring.
  bool gpu = false;
  gpuWhy_.clear();
  if (tiers_ > 0 && given.painter != 0) gpuWhy_ = "the GPU painter has one rate (the profile is painted by the C)";
  if (!gpu_) {
    try { gpu_ = GpuPainter::create(f.assets, nullptr, f.device, extras_.get()); }
    catch (const std::exception& e) { focus_free(&probe); return std::string("no Vulkan device for the frames: ") + e.what(); }
  }
  assets_ = f.assets;
  const int aheadFrames = std::max(1, static_cast<int>(std::lround(f.fps * f.aheadSecs)));
  if (f.painter != 0) {
    try {
      gpuWhy_ = gpu_->configure(f.n, f.subch, f.span, f.fps, 4, f.codes, f.gap, aheadFrames, f.aheadBytes, f.margin, false);
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
    } catch (const std::exception& e) { gpuWhy_ = e.what(); }
    if (gpuWhy_.empty()) gpu = true;
    else if (f.painter == 1) { focus_free(&probe); return "the GPU cannot paint here: " + gpuWhy_; }
  }
  if (!gpu) {
    std::string e;
    try { e = gpu_->configure(f.n, f.subch, f.span, f.fps, 1, f.codes, f.gap, aheadFrames, f.aheadBytes, f.margin, true); }
    catch (const std::exception& x) { e = x.what(); }
    if (!e.empty()) { focus_free(&probe); return "no frame ring on the device: " + e; }
  }
  std::lock_guard<std::mutex> l(mu_);
  fmt_ = f;
  fmt_.threads = gpu ? gpu_->frames() : std::max(1, f.threads);
  onGpu_ = gpu;
  gpuDevice_ = gpu_->device();
  depth_ = gpu_->depth();
  blocks_ = probe.blocks;
  blockBytes_ = probe.block_bytes;
  side_ = probe.px + 2 * FOCUS_QUIET * probe.pxm;
  gap_ = f.codes > 1 ? f.gap * probe.pxm : 0;
  width_ = f.codes * side_ + (f.codes - 1) * gap_;
  focus_free(&probe);
  label_ = "LIZARD-" + std::to_string(f.subch) + (tiers_ > 0 ? " (" + tiersLabel_ + ")" : "") + (f.codes > 1 ? " x " + std::to_string(f.codes) : "");
  // ai: what was made or painted under the old format goes; the count goes on from the screen's
  jobs_.clear(); ready_.clear(); hostOf_.clear(); toUpload_.clear(); dataOf_.clear(); spare_.clear();
  made_ = shown_;
  stop_ = false;
  gen_++;
  producer_ = std::thread([this] { produce(); });
  if (gpu) painters_.emplace_back([this] { paintGpu(); });
  else {
    for (int k = 0; k < fmt_.threads; k++) painters_.emplace_back([this, k] { paint(k); });
    uploader_ = std::thread([this] { upload(); });
  }
  return "";
}

std::string Sender::prepareGpu(const std::string& assets, const std::string& device, const wg::DeviceExtras* extras) {
  if (extras) extras_ = std::make_shared<wg::DeviceExtras>(*extras);
  if (gpu_) return "";
  try { gpu_ = GpuPainter::create(assets, nullptr, device, extras_.get()); prepareWhy_.clear(); return ""; }
  catch (const std::exception& e) { prepareWhy_ = e.what(); return prepareWhy_; }
}

wg::Device* Sender::device() const { return gpu_ ? gpu_->vk() : nullptr; }

// ai: Frames made ahead of the screen, up to the ring's depth less the margin (TxFormat): frame seq goes to slot seq
// ai: mod depth, so the slot's last frame (seq - depth) must be taken and past every reader's hold before it is made.
// ai: A frame's blocks are filled outside mu_ (under xferMu_, the transfer's), so a deep queue never holds the lock
// ai: against the screen's take.
void Sender::produce() {
  std::unique_lock<std::mutex> l(mu_);
  const int T = blocks_ * fmt_.codes, B = blockBytes_, room = std::max(1, depth_ - fmt_.margin);   // ai: a frame's blocks: its codes' one after the other
  std::vector<uint32_t> ids(T);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || made_ < shown_ + static_cast<uint64_t>(room); });
    if (stop_) return;
    Job j{made_++, std::vector<uint8_t>(static_cast<size_t>(T) * B), 0};
    l.unlock();
    {
      std::lock_guard<std::mutex> x(xferMu_);
      if (xfer_) xfer_->frameIds(ids.data(), T);
      else for (int t = 0; t < T; t++) ids[t] = nextId_++;
      for (int t = 0; t < T; t++) {
        uint8_t* o = j.blocks.data() + static_cast<size_t>(t) * B;
        for (int i = 0; i < ID_BYTES; i++) o[i] = static_cast<uint8_t>(ids[t] >> 8 * i);
        if (xfer_) xfer_->block(ids[t], o + ID_BYTES);
        else stream_fill(ids[t], o + ID_BYTES, PAYLOAD);
        if (!xfer_ || !control(ids[t])) j.data++;
      }
    }
    l.lock();
    dataOf_[j.seq] = j.data;
    jobs_.push_back(std::move(j));
    cv_.notify_all();
  }
}

// ai: A painter: its own codec, the oldest frame made, the pilots' count its number mod 4 (src/focus.h focus_parity).
// ai: The frame grey, packed as the ring's rows (4 x rowWords bytes a row; with two codes each symbol painted on its
// ai: own and copied into its place, the gap the margin's byte beside it, as lizard-web/send-worker.mjs does), handed
// ai: to the uploader.
void Sender::paint(int) {
  focus_t f{};
  TxFormat fm;
  size_t rowBytes = 0, bytes = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    fm = fmt_;
    rowBytes = 4ull * gpu_->rowWords();
    bytes = 4ull * gpu_->frameWords();
  }
  if (makeCodec(&f, fm)) {
    std::lock_guard<std::mutex> l(mu_);
    error_ = "a painter's codec failed";
    return;
  }
  focus_fmt_fps(&f, fm.fps);
  std::vector<float> drive(static_cast<size_t>(f.px) * f.px);
  const int W = side_, codes = fm.codes, gap = gap_;
  const size_t per = static_cast<size_t>(blocks_) * blockBytes_;
  std::vector<uint8_t> part(static_cast<size_t>(W) * W);
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    Job j = std::move(jobs_.front());
    jobs_.pop_front();
    std::vector<uint8_t> out;
    if (!spare_.empty()) { out = std::move(spare_.back()); spare_.pop_back(); }
    l.unlock();
    out.resize(bytes);
    const double t0 = nowMs();
    focus_parity(&f, static_cast<int>(j.seq & 3));
    for (int k = 0; k < codes; k++) {
      focus_encode(&f, j.blocks.data() + k * per, drive.data());
      focus_paint_grey(&f, drive.data(), part.data());
      const size_t x0 = static_cast<size_t>(k) * (W + gap);
      for (int y = 0; y < W; y++) {
        uint8_t* o = out.data() + y * rowBytes + x0;
        const uint8_t* src = part.data() + static_cast<size_t>(y) * W;
        std::memcpy(o, src, W);
        if (k + 1 < codes) std::memset(o + W, src[W - 1], gap);
      }
    }
    const double ms = nowMs() - t0;
    l.lock();
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_++;
    if (j.seq >= shown_) toUpload_[j.seq] = std::move(out);
    cv_.notify_all();
  }
  l.unlock();
  focus_free(&f);
}

// ai: The uploader (the CPU painters' mode): each painted frame into its slot of the ring, in order, then ready; the
// ai: one thread recording on the device then (wg records on one thread).
void Sender::upload() {
  bool host = false;
  int depth = 1;
  {
    std::lock_guard<std::mutex> l(mu_);
    host = fmt_.hostFrames;
    depth = std::max(1, depth_);
  }
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !toUpload_.empty(); });
    if (stop_) break;
    auto it = toUpload_.begin();
    const uint64_t seq = it->first;
    std::vector<uint8_t> buf = std::move(it->second);
    toUpload_.erase(it);
    l.unlock();
    const int slot = static_cast<int>(seq % static_cast<uint64_t>(depth));
    std::string err;
    try { gpu_->upload(slot, buf.data()); } catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!err.empty()) { error_ = "the frame upload stopped: " + err; cv_.notify_all(); break; }
    if (seq >= shown_) {
      ready_[seq] = slot;
      if (host) hostOf_[seq] = std::move(buf); else spare_.push_back(std::move(buf));
    }
    cv_.notify_all();
  }
}

// ai: The GPU's painter: the oldest frames made, up to an encode's worth (consecutive, so the first's count and one more
// ai: each is every frame's own; never past the ring's end), one submission into their slots; their host copies too
// ai: where the host takes frames by take().
void Sender::paintGpu() {
  const int R = gpu_->frames(), D = std::max(1, gpu_->depth());
  bool host = false;
  {
    std::lock_guard<std::mutex> l(mu_);
    host = fmt_.hostFrames;
  }
  std::vector<std::vector<uint8_t>> outs(R);
  std::vector<uint8_t> blocks;
  std::vector<Job> js;
  std::unique_lock<std::mutex> l(mu_);
  for (;;) {
    cv_.wait(l, [&] { return stop_ || !jobs_.empty(); });
    if (stop_) break;
    const int room = D - static_cast<int>(jobs_.front().seq % static_cast<uint64_t>(D));
    const int k = std::min({R, static_cast<int>(jobs_.size()), room});
    js.clear();
    for (int i = 0; i < k; i++) { js.push_back(std::move(jobs_.front())); jobs_.pop_front(); }
    if (host) for (int i = 0; i < k; i++) if (outs[i].empty() && !spare_.empty()) { outs[i] = std::move(spare_.back()); spare_.pop_back(); }
    l.unlock();
    const size_t per = js[0].blocks.size();
    blocks.resize(per * k);
    for (int i = 0; i < k; i++) std::memcpy(blocks.data() + per * i, js[i].blocks.data(), per);
    double ms = 0;
    std::string err;
    try { ms = gpu_->encode(blocks.data(), k, static_cast<int>(js[0].seq & 3), host ? outs.data() : nullptr, js[0].seq); }
    catch (const std::exception& e) { err = e.what(); }
    l.lock();
    if (!err.empty()) { error_ = "the GPU painter stopped: " + err; cv_.notify_all(); break; }
    paintMsWin_ = paintMsWin_ + ms;
    paintedWin_ += k;
    for (int i = 0; i < k; i++) if (js[i].seq >= shown_) {
      ready_[js[i].seq] = static_cast<int>(js[i].seq % static_cast<uint64_t>(D));
      if (host) hostOf_[js[i].seq] = std::move(outs[i]);
    }
    cv_.notify_all();
  }
}

bool Sender::ready() {
  std::lock_guard<std::mutex> l(mu_);
  return ready_.count(shown_) != 0;
}

bool Sender::takeSlot(uint64_t* seq, int* slot) {
  std::lock_guard<std::mutex> l(mu_);
  auto it = ready_.find(shown_);
  if (it == ready_.end()) return false;
  *seq = shown_;
  *slot = it->second;
  ready_.erase(it);
  auto h = hostOf_.find(shown_);
  if (h != hostOf_.end()) { spare_.push_back(std::move(h->second)); hostOf_.erase(h); }
  auto d = dataOf_.find(shown_);
  int data = 0;
  if (d != dataOf_.end()) { data = d->second; dataOf_.erase(d); }
  shown_++;
  takenWin_++;
  dataWin_ += data;
  dataAll_ += data;
  cv_.notify_all();
  return true;
}

bool Sender::take(uint8_t* dst, int stride) {
  std::vector<uint8_t> buf;
  int data = 0;
  size_t srcRow = 0;
  {
    std::lock_guard<std::mutex> l(mu_);
    auto it = ready_.find(shown_);
    if (it == ready_.end()) return false;
    auto h = hostOf_.find(shown_);
    if (h == hostOf_.end()) { error_ = "take() needs TxFormat.hostFrames"; return false; }
    buf = std::move(h->second);
    hostOf_.erase(h);
    ready_.erase(it);
    auto d = dataOf_.find(shown_);
    if (d != dataOf_.end()) { data = d->second; dataOf_.erase(d); }
    shown_++;
    // ai: a GPU frame read back is width_ bytes a row; a CPU frame is packed as the ring's rows
    srcRow = buf.size() == static_cast<size_t>(width_) * side_ ? static_cast<size_t>(width_) : 4ull * gpu_->rowWords();
  }
  for (int y = 0; y < side_; y++) {
    uint32_t* d = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(y) * stride);
    const uint8_t* g = buf.data() + static_cast<size_t>(y) * srcRow;
    for (int x = 0; x < width_; x++) d[x] = 0xff000000u | g[x] * 0x010101u;
  }
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

std::shared_ptr<wg::Buffer> Sender::ring() const { return gpu_ ? gpu_->ring() : nullptr; }
uint32_t Sender::ringFrameWords() const { return gpu_ ? gpu_->frameWords() : 0; }
uint32_t Sender::ringRowWords() const { return gpu_ ? gpu_->rowWords() : 0; }

// ai: The stats a second (the app asks): label, n, subch, fps (asked), shownFps (frames on the screen a second over the
// ai: window), offeredKBs (data blocks shown a second x 469 B, 1000 B a KB, as the page's), paintMs (a frame's encode
// ai: and paint on one painter; on the GPU its ms a frame, an encode's timestamps over its frames), painters (the
// ai: GPU's: frames an encode), painter ("gpu" or "cpu"), device (the GPU's), gpuWhy (why not the GPU where it was
// ai: asked for), ahead (painted, not shown), side, test, and for a file chunks, lap (data blocks a lap), pass (laps
// ai: shown, 1 the first), root, sentBytes (the file's bytes as they go, every chunk's zstd frame or its own;
// ai: 2026-10-05); error, the last failure.
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
    << "\",\"gpuWhy\":\"" << esc(gpuWhy_) << "\",\"ahead\":" << ready_.size() << ",\"depth\":" << depth_ << ",\"side\":" << side_ << ",\"width\":" << width_
    << ",\"codes\":" << fmt_.codes << ",\"shown\":" << shown_ << ",\"blocks\":" << blocks_ << ",\"tiers\":\"" << esc(tiersLabel_) << "\""
    << ",\"test\":" << (xfer_ ? "false" : "true");
  if (xfer_) {
    std::lock_guard<std::mutex> x(xferMu_);
    const uint32_t lap = xfer_->lap();
    o.precision(2);
    o << ",\"chunks\":" << xfer_->chunks() << ",\"lap\":" << lap << ",\"pass\":" << (lap ? 1.0 + static_cast<double>(dataAll_) / lap : 1.0)
      << ",\"root\":\"" << xfer_->rootHex() << "\",\"sentBytes\":" << xfer_->sentBytes();
  }
  o << ",\"error\":\"" << error_ << "\"}";
  return o.str();
}

}  // namespace lizard
