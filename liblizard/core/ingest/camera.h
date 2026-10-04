// ai: The camera's frames into the decoder's ring with no copy (2026-09-29). Each camera buffer (an AHardwareBuffer, in
// ai: the camera's own layout, UBWC-compressed on a Qualcomm) is imported into Vulkan once and kept while the camera
// ai: cycles through its pool; a frame's centre square is sampled from it through a YCbCr sampler into a ring layer by
// ai: camera.comp, on the device's second queue, so a camera buffer is held only for its own ingest and never behind a
// ai: batch. The batch that copies the layer waits on the ingest's timeline value (FrontHalf::run).
//
// ai: One thread calls ingest() (the camera's); done() and the semaphore are read from any.
#pragma once
#include "wg.h"
#include "setup.h"

#include <mutex>
#include <unordered_map>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#else
struct AHardwareBuffer;
#endif

namespace lizard {

class CameraIngest {
 public:
  // ai: Null where the device lacks YCbCr sampling, r8 storage images, timeline semaphores or the AHB import: the
  // ai: receiver then reads the frame's Y plane on the CPU and uploads it (FrontHalf::enqueueLuma).
  static std::unique_ptr<CameraIngest> create(wg::Device& d, ReadFile read);
  ~CameraIngest();
  // ai: The buffer's crop (x, y, w, h) into layer `layer` of `ring`: the timeline value its ingest signals.
  uint64_t ingest(AHardwareBuffer* hb, uint32_t x, uint32_t y, uint32_t w, uint32_t h, wg::Texture& ring, uint32_t layer);
  bool done(uint64_t value) const;
  uint64_t completed() const;
  VkSemaphore timeline = VK_NULL_HANDLE;
  // ai: Every import made again (a camera reconfigured, a ring remade).
  void forget();
  // ai: The camera's reader closed (2026-10-02): its buffers will not come again, and a reopened camera hands over new
  // ai: ones. Waits (up to a second) for the ingests already submitted, frees their command buffers, then forgets every
  // ai: import. Until then each reopen added its reader's buffers to the imports, holding their memory, and the third
  // ai: opening in one receiver ran the 64-set descriptor pool out: an ingest error, the receiver's state "error".
  // ai: Returns the imports it let go.
  size_t cameraClosed();

 private:
  CameraIngest(wg::Device& d) : dev(d) {}
  struct Imported { VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; VkImageView ringView = VK_NULL_HANDLE; uint32_t w = 0, h = 0; };
  wg::Device& dev;
  std::vector<uint32_t> spirv;
  uint64_t externalFormat = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkSamplerYcbcrConversion conv = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  std::unordered_map<AHardwareBuffer*, Imported> imports;
  struct Pending { VkCommandBuffer cb; uint64_t value; };
  std::vector<Pending> pending;
  uint64_t last = 0;
  bool build(AHardwareBuffer* hb);
  Imported& importOf(AHardwareBuffer* hb);
  void destroyPipe();
};

}  // namespace lizard
