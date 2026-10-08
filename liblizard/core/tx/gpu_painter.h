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

namespace wg { struct Device; struct DeviceExtras; struct Buffer; struct BindGroupLayout; struct Pipeline; struct BindGroup; struct QuerySet; }

namespace lizard {

class GpuPainter {
 public:
  // ai: assets: liblizard/out's tree as the app ships it (filesDir/lizard: setup/send.json, blobs/, spv/send_*.spv). The
  // ai: device: the one named like `device` (a substring of its name), else LIZ_VK_DEVICE's, else the first discrete,
  // ai: else the first. Throws std::runtime_error saying why there is none.
  // ai: extras: what the host asks of the device beyond the kernels (the desktop presenter's surface and swapchain).
  static std::unique_ptr<GpuPainter> create(const std::string& assets, std::function<void(const std::string&)> log = nullptr, const std::string& device = "",
                                            const wg::DeviceExtras* extras = nullptr);
  ~GpuPainter();
  // ai: the device, for a host that presents from it (its handles are wg::Device's public fields)
  wg::Device* vk() const { return dev_.get(); }

  // ai: A format (n, sub-channels, the ring's span, the rate the word states), encodes of up to `frames` frames (halved
  // ai: while the device cannot hold them), each `codes` symbols side by side `gap` modules apart (send_tables.h), and
  // ai: the frame ring (2026-10-07): FRAME holds depth() frames, the most of `aheadFrames` and `aheadBytes` allow and at
  // ai: least 2 frames + margin + 1 (halved while the device cannot hold them, never under the floor), frame seq at slot
  // ai: seq mod depth. ringOnly: the ring alone (the CPU's painters fill it), no pipelines' buffers. "" or why not.
  std::string configure(int n, int subch, int span, int fps, int frames = 4, int codes = 1, int gap = GAP_MODULES,
                        int aheadFrames = 0, uint64_t aheadBytes = 0, int margin = 0, bool ringOnly = false);
  int frames() const { return frames_; }
  int depth() const { return depth_; }
  // ai: the ring: FRAME (grey, four pixels a u32; a frame frameWords() u32 from slot * frameWords(), its rows rowWords()
  // ai: u32 apart), shared so a host that reads it on another queue keeps it alive across a configure
  std::shared_ptr<wg::Buffer> ring() const { return FRAME_; }
  uint32_t frameWords() const { return static_cast<uint32_t>(t_.FS); }
  uint32_t rowWords() const { return static_cast<uint32_t>(t_.RW); }
  // ai: a frame painted elsewhere (the C), packed as the ring's rows (4 * rowWords() bytes a row, 4 * frameWords() in
  // ai: all), into its slot; returns once it is there. The CPU mode's uploader thread calls it, no other recorder.
  void upload(int slot, const uint8_t* packed);
  int side() const { return t_.W; }      // ai: a symbol's side, the frame's height
  int width() const { return t_.FW; }    // ai: the frame's width: codes symbols and the gaps between them
  int codes() const { return t_.codes; }
  const std::string& device() const { return name_; }

  // ai: count frames (1 to frames()), each codes x blocks x blockBytes (a block its id then its payload, as Sender makes
  // ai: them; a frame's first code's blocks, then its second's), into out[0 .. count) as width() x side() grey, the first
  // ai: painted as picture `parity` (its count mod 4, the pilots'; each frame after it one more, its codes alike). Returns the GPU's ms for the whole encode (timestamps), or the submission's wall ms where
  // ai: the device has none. Throws on a lost device.
  // ai: seq: the first frame's number: the frames go to slots seq mod depth() on (an encode never wraps the ring:
  // ai: count is at most depth() - seq mod depth()). out null: the frames stay on the device (the ring's readers take
  // ai: them there); else each is read back into out[f] as width() x side() grey.
  double encode(const uint8_t* blocks, int count, int parity, std::vector<uint8_t>* out, uint64_t seq = 0);

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
  int frames_ = 0, fps_ = 0, depth_ = 0;
  bool ringOnly_ = false;
  std::shared_ptr<wg::BindGroupLayout> bglPaint_, bglIrows_, bglIpic_, bglTpose_, bglRsv_, bglRsh_;
  std::shared_ptr<wg::Pipeline> paint_, tpose_, rsv_, rsh_, rshCopy_, irows_, ipic_;
  int pipesN_ = 0;
  std::shared_ptr<wg::Buffer> TAB_;   // ai: the paint's codes and whitening (send_tables.h SendConsts tab), made at create
  // ai: a configuration's buffers and groups
  std::shared_ptr<wg::Buffer> BLOCKS_, UV_, S_, PU_, TW_, ROWS_, SU_, Y_, PQT_, PIC_, TAPS_, BORDER_, GU_, FRAME_, READ_, TIME_;
  std::shared_ptr<wg::BindGroup> gPaint_, gIrows_, gIpic_, gTpose_, gRsv_, gRsh_;
  std::shared_ptr<wg::QuerySet> qs_;
  std::vector<uint8_t> staging_;
};

}  // namespace lizard
