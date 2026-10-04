// ai: The sender's painter on the GPU (2026-10-02; the CPU's four painters already held 60 a second at LIZARD-1024 on
// ai: the S26, about 1.5 cores busy): liblizard/gpu/encoder.mjs natively, on the app's Vulkan host (core/wg). Its
// ai: kernels are the web's, compiled by liblizard/gen (gen/sender.mjs: out/spv/send_*.spv, setup/send.json); its
// ai: tables the C's (send_tables.h). An encode is up to frames() frames' blocks in, one submission (paint, irows,
// ai: ipic, tpose, rsv unless the taps are copies, rsh), their grey W x W pixels read back: on a phone's shared
// ai: memory a copy, not a transfer. The square is within a grey level of the C's (the transform's rounding), the
// ai: border the C's to the byte; check() holds a device to that, as the web's auto does before it takes the GPU
// ai: (2026-09-29: a sender's symbol need only decode).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "send_tables.h"

namespace wg { struct Device; struct Buffer; struct BindGroupLayout; struct Pipeline; struct BindGroup; struct QuerySet; }

namespace lizard {

class GpuPainter {
 public:
  // ai: assets: liblizard/out's tree as the app ships it (filesDir/lizard: setup/send.json, blobs/, spv/send_*.spv). The
  // ai: device: the one named like `device` (a substring of its name), else LIZ_VK_DEVICE's, else the first discrete,
  // ai: else the first. Throws std::runtime_error saying why there is none.
  static std::unique_ptr<GpuPainter> create(const std::string& assets, std::function<void(const std::string&)> log = nullptr, const std::string& device = "");
  ~GpuPainter();

  // ai: A format (n, sub-channels, the ring's span, the rate the word states), encodes of up to `frames` frames (halved
  // ai: while the device cannot hold them), each `codes` symbols side by side `gap` modules apart (send_tables.h). ""
  // ai: or why not.
  std::string configure(int n, int subch, int span, int fps, int frames = 4, int codes = 1, int gap = GAP_MODULES);
  int frames() const { return frames_; }
  int side() const { return t_.W; }      // ai: a symbol's side, the frame's height
  int width() const { return t_.FW; }    // ai: the frame's width: codes symbols and the gaps between them
  int codes() const { return t_.codes; }
  const std::string& device() const { return name_; }

  // ai: count frames (1 to frames()), each codes x blocks x blockBytes (a block its id then its payload, as Sender makes
  // ai: them; a frame's first code's blocks, then its second's), into out[0 .. count) as width() x side() grey, the first
  // ai: painted as picture `parity` (its count mod 4, the pilots'; each frame after it one more, its codes alike). Returns the GPU's ms for the whole encode (timestamps), or the submission's wall ms where
  // ai: the device has none. Throws on a lost device.
  double encode(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out);

  // ai: frames painted here and by the C (its own codec, told the same counts): the border's pixels equal, the square's
  // ai: within one grey level (gpu/encoder.mjs checkNow's rule).
  struct Check { bool ok = false; long borderDiff = 0, squareDiff = 0; int squareMax = 0; };
  Check check(const uint8_t* blocks, int count);

 private:
  GpuPainter() = default;
  std::string pipes(int n);
  std::unique_ptr<wg::Device> dev_;
  std::string name_, assets_;
  std::function<void(const std::string&)> log_;
  SendConsts k_;
  focus_t f_{};
  bool fReady_ = false;
  SendTables t_;
  int frames_ = 0, fps_ = 0;
  std::shared_ptr<wg::BindGroupLayout> bglPaint_, bglIrows_, bglIpic_, bglTpose_, bglRsv_, bglRsh_;
  std::shared_ptr<wg::Pipeline> paint_, tpose_, rsv_, rsh_, rshCopy_, irows_, ipic_;
  int pipesN_ = 0;
  // ai: a configuration's buffers and groups
  std::shared_ptr<wg::Buffer> BLOCKS_, PERMW_, UV_, S_, PU_, TW_, ROWS_, SU_, Y_, PQT_, PIC_, TAPS_, BORDER_, GU_, FRAME_, READ_, TIME_;
  std::shared_ptr<wg::BindGroup> gPaint_, gIrows_, gIpic_, gTpose_, gRsv_, gRsh_;
  std::shared_ptr<wg::QuerySet> qs_;
  std::vector<uint8_t> staging_;
};

}  // namespace lizard
