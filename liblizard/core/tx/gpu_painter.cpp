#include "gpu_painter.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "json.hpp"
#include "wg.h"

namespace lizard {

namespace {
std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("the GPU sender's " + path + " is missing");
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}
std::vector<uint32_t> spirv(const std::string& path) {
  const auto b = readFile(path);
  std::vector<uint32_t> w(b.size() / 4);
  std::memcpy(w.data(), b.data(), w.size() * 4);
  return w;
}
double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

std::unique_ptr<GpuPainter> GpuPainter::create(const std::string& assets, std::function<void(const std::string&)> log, const std::string& device,
                                               const wg::DeviceExtras* extras) {
  std::unique_ptr<GpuPainter> p(new GpuPainter());
  p->assets_ = assets;
  p->log_ = log ? log : [](const std::string&) {};
  const auto manifest = readFile(assets + "/setup/send.json");
  const std::string text(manifest.begin(), manifest.end());
  const auto json = nlohmann::json::parse(text);
  if (!json.contains("tab")) throw std::runtime_error("the GPU sender's " + assets + "/setup/send.json predates the rate profile: regenerate liblizard/out");
  const std::string blob = json["tab"]["blob"];
  const std::string err = p->k_.load(text, readFile(assets + "/" + blob));
  if (!err.empty()) throw std::runtime_error(err);
  const char* want = getenv("LIZ_VK_DEVICE");
  p->dev_ = wg::Device::create(!device.empty() ? device : want ? want : "", false, p->log_, extras ? *extras : wg::DeviceExtras());
  p->name_ = p->dev_->name;
  using wg::Bind;
  auto& d = *p->dev_;
  // ai: gpu/encoder.mjs buildShared's layouts, binding for binding
  p->bglPaint_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send paint");
  p->bglIrows_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform, Bind::Uniform}, "send irows");
  p->bglIpic_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send ipic");
  p->bglTpose_ = d.createBindGroupLayout({Bind::RO, Bind::RW, Bind::Uniform}, "send tpose");
  p->bglRsv_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send rsv");
  p->bglRsh_ = d.createBindGroupLayout({Bind::RO, Bind::RO, Bind::RO, Bind::RW, Bind::Uniform}, "send rsh");
  const std::string spv = assets + "/spv/send_";
  p->paint_ = d.createPipeline(spirv(spv + "paint.spv"), p->bglPaint_, "send paint");
  p->tpose_ = d.createPipeline(spirv(spv + "tpose.spv"), p->bglTpose_, "send tpose");
  p->rsv_ = d.createPipeline(spirv(spv + "rsv.spv"), p->bglRsv_, "send rsv");
  p->rsh_ = d.createPipeline(spirv(spv + "rsh.spv"), p->bglRsh_, "send rsh");
  p->rshCopy_ = d.createPipeline(spirv(spv + "rshCopy.spv"), p->bglRsh_, "send rsh copy");
  p->qs_ = d.createQuerySet(2, "send time");
  // ai: TAB, the paint's codes and whitening: the same for every format, made once
  p->TAB_ = d.createBuffer(4ull * p->k_.tab.size(), wg::STORAGE | wg::COPY_DST, "send TAB", false);
  d.upload(*p->TAB_, 0, p->k_.tab.data(), 4ull * p->k_.tab.size());
  return p;
}

GpuPainter::~GpuPainter() {
  if (dev_) dev_->waitIdle();
  gPaint_.reset(); gIrows_.reset(); gIpic_.reset(); gTpose_.reset(); gRsv_.reset(); gRsh_.reset();
  if (fReady_) focus_free(&f_);
}

// ai: The two inverse transforms of a picture size (gpu/encoder.mjs pipesFor): ipic holds two exchange buffers of n
// ai: vec2f in workgroup memory, and n / 8 invocations a workgroup.
std::string GpuPainter::pipes(int n) {
  if (pipesN_ == n) return "";
  const auto& L = dev_->limits;
  if (16u * n > L.maxComputeSharedMemorySize) return "the " + std::to_string(n) + " picture's inverse needs " + std::to_string(16 * n) + " B of workgroup memory, the device has " + std::to_string(L.maxComputeSharedMemorySize);
  if (static_cast<uint32_t>(n / 8) > L.maxComputeWorkGroupInvocations) return "the " + std::to_string(n) + " picture's inverse needs " + std::to_string(n / 8) + " invocations a workgroup";
  const std::string spv = assets_ + "/spv/send_", s = std::to_string(n);
  irows_ = dev_->createPipeline(spirv(spv + "irows" + s + ".spv"), bglIrows_, "send irows " + s);
  ipic_ = dev_->createPipeline(spirv(spv + "ipic" + s + ".spv"), bglIpic_, "send ipic " + s);
  pipesN_ = n;
  return "";
}

std::string GpuPainter::configure(int n, int subch, int span, int fps, int frames, int codes, int gap, int aheadFrames, uint64_t aheadBytes, int margin, bool ringOnly) {
  if (fReady_) { focus_free(&f_); fReady_ = false; }
  if (focus_init(&f_, n, subch, 1, txClip(), span, 0.f, 0, 0, 0, 0, 0, 0)) return "the codec refused LIZARD-" + std::to_string(subch) + " at n = " + std::to_string(n);
  fReady_ = true;
  focus_fmt_fps(&f_, fps);
  fps_ = fps;
  try {
    if (!ringOnly) {
      const std::string pe = pipes(n);
      if (!pe.empty()) return pe;
    }
    ringOnly_ = ringOnly;
    using namespace wg;
    auto& d = *dev_;
    for (int R = std::max(1, frames); R >= 1; R /= 2) {
      const std::string te = sendTables(f_, k_, R, t_, codes, gap);
      if (!te.empty()) return te;
      const SendTables& t = t_;
      // ai: Z: the symbols an encode (frames x codes): every stage before rsh works a symbol a workgroup z, rsh a frame
      const uint64_t V = t.blocks, np = t.npos, Z = uint64_t(R) * t.codes;
      // ai: the ring's depth: the budget's, floored at 2R + margin + 1 (an encode's frames twice over and the slots a
      // ai: reader may still hold); halved on refusal down to that floor before R is
      const int floorDepth = 2 * R + std::max(0, margin) + 1;
      int depth = floorDepth;
      if (aheadFrames > 0 && aheadBytes > 0) {
        const uint64_t byBytes = aheadBytes / (4ull * t.FS);
        depth = static_cast<int>(std::min<uint64_t>(static_cast<uint64_t>(aheadFrames), byBytes));
        depth = std::max(depth, floorDepth);
      }
      for (; depth >= floorDepth; depth = depth / 2 >= floorDepth ? depth / 2 : (depth > floorDepth ? floorDepth : 0)) {
      try {
        gPaint_.reset(); gIrows_.reset(); gIpic_.reset(); gTpose_.reset(); gRsv_.reset(); gRsh_.reset();
        BLOCKS_.reset(); S_.reset(); Y_.reset(); PQT_.reset(); PIC_.reset(); FRAME_.reset(); READ_.reset();
        FRAME_ = d.createBuffer(uint64_t(depth) * 4ull * t.FS, STORAGE | COPY_SRC | COPY_DST, "send FRAME");
        READ_ = d.createBuffer(R * 4ull * t.FS, MAP_READ | COPY_DST, "send READ");
        TIME_ = d.createBuffer(16, MAP_READ | COPY_DST, "send TIME");
        if (ringOnly) {
          d.flush();
          frames_ = R;
          depth_ = depth;
          log_("gpu sender: LIZARD-" + std::to_string(subch) + (t.codes > 1 ? " x " + std::to_string(t.codes) : "") + " at " + std::to_string(n) + ", " + std::to_string(t.FW) + " x " + std::to_string(t.W) + " px, a ring of " +
               std::to_string(depth) + " frames for the CPU's painters, " + std::to_string(d.allocated.load() >> 20) + " MB on " + name_);
          return "";
        }
        BLOCKS_ = d.createBuffer(Z * V * k_.blockBytes, STORAGE | COPY_DST, "send BLOCKS");
        UV_ = d.createBuffer(4 * np, STORAGE | COPY_DST, "send UV", false);
        d.upload(*UV_, 0, t.uv.data(), 4 * np);
        S_ = d.createBuffer(Z * np * 8, STORAGE | COPY_DST, "send S");
        PU_ = d.createBuffer(4 * t.pu.size(), UNIFORM | COPY_DST, "send PU", false);
        d.writeBuffer(*PU_, 0, t.pu.data(), 4 * t.pu.size());
        TW_ = d.createBuffer(8ull * n, STORAGE | COPY_DST, "send TW", false);
        d.upload(*TW_, 0, t.tw.data(), 8ull * n);
        ROWS_ = d.createBuffer(4 * t.rows.size(), UNIFORM | COPY_DST, "send ROWS", false);
        d.writeBuffer(*ROWS_, 0, t.rows.data(), 4 * t.rows.size());
        SU_ = d.createBuffer(48, UNIFORM | COPY_DST, "send SU", false);
        d.writeBuffer(*SU_, 0, t.su.data(), 48);
        Y_ = d.createBuffer(Z * uint64_t(t.Vr) * n * 8, STORAGE, "send Y");
        PQT_ = d.createBuffer(Z * 4ull * std::max<uint64_t>(uint64_t(n) * n, t.copy ? 0 : uint64_t(t.q) * n), STORAGE, "send PQT");
        PIC_ = d.createBuffer(Z * 4ull * n * n, STORAGE, "send PIC");
        TAPS_ = d.createBuffer(4 * t.taps.size(), STORAGE | COPY_DST, "send TAPS", false);
        d.upload(*TAPS_, 0, t.taps.data(), 4 * t.taps.size());
        BORDER_ = d.createBuffer(t.border.size(), STORAGE | COPY_DST, "send BORDER", false);
        d.upload(*BORDER_, 0, t.border.data(), t.border.size());
        GU_ = d.createBuffer(48, UNIFORM | COPY_DST, "send GU", false);
        d.writeBuffer(*GU_, 0, t.g.data(), 48);
        gPaint_ = d.createBindGroup(bglPaint_, {BLOCKS_, TAB_, UV_, S_, PU_});
        gIrows_ = d.createBindGroup(bglIrows_, {S_, TW_, Y_, ROWS_, SU_});
        gIpic_ = d.createBindGroup(bglIpic_, {Y_, TW_, PQT_, SU_});
        gTpose_ = d.createBindGroup(bglTpose_, {PQT_, PIC_, GU_});
        if (!t.copy) gRsv_ = d.createBindGroup(bglRsv_, {PIC_, TAPS_, PQT_, GU_});
        gRsh_ = d.createBindGroup(bglRsh_, {t.copy ? PIC_ : PQT_, TAPS_, BORDER_, FRAME_, GU_});
        d.flush();
        frames_ = R;
        depth_ = depth;
        staging_.assign(Z * V * k_.blockBytes, 0);
        log_("gpu sender: LIZARD-" + std::to_string(subch) + (t.codes > 1 ? " x " + std::to_string(t.codes) : "") + " at " + std::to_string(n) + ", " + std::to_string(t.FW) + " x " + std::to_string(t.W) + " px, encodes of " + std::to_string(R) +
             (t.copy ? ", rsh's copy" : "") + ", a ring of " + std::to_string(depth) + " frames, " + std::to_string(d.allocated.load() >> 20) + " MB on " + name_);
        return "";
      } catch (const OutOfMemory&) {
        log_("gpu sender: encodes of " + std::to_string(R) + " with a ring of " + std::to_string(depth) + " do not fit; fewer");
      }
      }
    }
    return "no encode of LIZARD-" + std::to_string(subch) + " fits this device's memory";
  } catch (const std::exception& e) {
    return std::string("the GPU sender: ") + e.what();
  }
}

double GpuPainter::encode(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq) {
  const SendTables& t = t_;
  const int V = t.blocks, B = t.blockBytes, BB = k_.blockBytes;
  if (ringOnly_) throw std::runtime_error("the GPU painter holds the ring alone here (the CPU paints)");
  if (count < 1 || count > frames_) throw std::runtime_error("an encode is 1 to " + std::to_string(frames_) + " frames");
  const uint32_t first = static_cast<uint32_t>(seq % static_cast<uint64_t>(depth_));
  if (static_cast<int>(first) + count > depth_) throw std::runtime_error("an encode of " + std::to_string(count) + " frames from slot " + std::to_string(first) + " would wrap the ring of " + std::to_string(depth_));
  auto& d = *dev_;
  const int Z = count * t.codes;   // ai: the symbols, a frame's codes one after the other
  for (int i = 0; i < Z * V; i++) std::memcpy(staging_.data() + static_cast<size_t>(i) * BB, blocks + static_cast<size_t>(i) * B, B);
  d.writeBuffer(*BLOCKS_, 0, staging_.data(), static_cast<uint64_t>(Z) * V * BB);
  const uint32_t pc = static_cast<uint32_t>(parity & 3) | (static_cast<uint32_t>(t.codes) << 2);
  d.writeBuffer(*GU_, 36, &first, 4);
  d.writeBuffer(*PU_, 4ull * (k_.paramsDims + 3), &pc, 4);
  const uint32_t R = static_cast<uint32_t>(count), RZ = static_cast<uint32_t>(Z), n = static_cast<uint32_t>(t.n);
  auto enc = d.encoder();
  enc.clearBuffer(*S_, 0, uint64_t(RZ) * t.npos * 8);
  enc.beginComputePass(qs_.get(), 0, 1);
  enc.setPipeline(*paint_); enc.setBindGroup(*gPaint_); enc.dispatch(V, 1, RZ);
  enc.setPipeline(*irows_); enc.setBindGroup(*gIrows_); enc.dispatch(t.Vr, 1, RZ);
  enc.setPipeline(*ipic_); enc.setBindGroup(*gIpic_); enc.dispatch(n / 4, 1, RZ);
  enc.setPipeline(*tpose_); enc.setBindGroup(*gTpose_); enc.dispatch(n / 4 / k_.tposeTile, n / k_.tposeTile, RZ);
  if (!t.copy) { enc.setPipeline(*rsv_); enc.setBindGroup(*gRsv_); enc.dispatch((n / 4 + k_.rsvThreads - 1) / k_.rsvThreads, t.q, RZ); }
  enc.setPipeline(t.copy ? *rshCopy_ : *rsh_); enc.setBindGroup(*gRsh_); enc.dispatch((t.RW + k_.rshThreads - 1) / k_.rshThreads, t.W, R);
  enc.endPass();
  enc.resolveQuerySet(*qs_, 0, 2, *TIME_, 0);
  if (out) enc.copyBufferToBuffer(*FRAME_, uint64_t(first) * 4 * t.FS, *READ_, 0, uint64_t(R) * 4 * t.FS);
  const double t0 = nowMs();
  auto ticket = d.submit(enc);
  ticket->wait();
  const double wall = nowMs() - t0;
  if (out) {
    const uint8_t* px = d.read(*READ_, 0, uint64_t(R) * 4 * t.FS);
    for (int f = 0; f < count; f++) {
      out[f].resize(static_cast<size_t>(t.FW) * t.W);
      for (int y = 0; y < t.W; y++) std::memcpy(out[f].data() + static_cast<size_t>(y) * t.FW, px + 4 * (static_cast<size_t>(f) * t.FS + static_cast<size_t>(y) * t.RW), t.FW);
    }
  }
  if (!d.features.timestamps) return wall;
  uint64_t ts[2];
  std::memcpy(ts, d.read(*TIME_, 0, 16), 16);
  return double((ts[1] - ts[0]) & d.timestampMask) * d.timestampPeriod / 1e6;
}

void GpuPainter::upload(int slot, const uint8_t* packed) {
  auto& d = *dev_;
  const uint64_t bytes = 4ull * t_.FS;
  d.upload(*FRAME_, static_cast<uint64_t>(slot) * bytes, packed, bytes);
  auto enc = d.encoder();
  auto ticket = d.submit(enc);
  ticket->wait();
}

GpuPainter::Check GpuPainter::check(const uint8_t* blocks, int count) {
  Check c;
  const SendTables& t = t_;
  const size_t per = static_cast<size_t>(t.blocks) * t.blockBytes, perFrame = per * t.codes;
  std::vector<std::vector<uint8_t>> got(frames_);
  std::vector<float> drive(static_cast<size_t>(f_.px) * f_.px);
  std::vector<uint8_t> rgba(static_cast<size_t>(t.W) * t.W * 4);
  for (int f0 = 0; f0 < count; f0 += frames_) {
    const int k = std::min(frames_, count - f0);
    encode(blocks + f0 * perFrame, k, f0 & 3, got.data());
    for (int i = 0; i < k; i++) {
      // ai: each code against the C's paint of its own blocks, the frame's count; the gap's pixels the margin's byte
      focus_parity(&f_, (f0 + i) & 3);
      for (int cd = 0; cd < t.codes; cd++) {
        focus_encode(&f_, blocks + (f0 + i) * perFrame + cd * per, drive.data());
        focus_paint_rgba(&f_, drive.data(), rgba.data());
        const int x0 = cd * (t.W + t.gap);
        for (int y = 0; y < t.W; y++) for (int x = 0; x < t.W; x++) {
          const int dv = std::abs(int(got[i][static_cast<size_t>(y) * t.FW + x0 + x]) - int(rgba[4 * (static_cast<size_t>(y) * t.W + x)]));
          if (!dv) continue;
          if (x >= t.sq && x < t.sq + t.q && y >= t.sq && y < t.sq + t.q) { c.squareDiff++; c.squareMax = std::max(c.squareMax, dv); }
          else c.borderDiff++;
        }
        if (cd + 1 < t.codes) for (int y = 0; y < t.W; y++) for (int x = 0; x < t.gap; x++)
          if (got[i][static_cast<size_t>(y) * t.FW + x0 + t.W + x] != rgba[4 * (static_cast<size_t>(y) * t.W + t.W - 1)]) c.borderDiff++;
      }
    }
  }
  focus_parity(&f_, 0);
  c.ok = c.borderDiff == 0 && c.squareMax <= 1;
  return c;
}

}  // namespace lizard
