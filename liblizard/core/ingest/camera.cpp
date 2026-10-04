#include "camera.h"

#include <cstring>

namespace lizard {

#ifdef __ANDROID__

std::unique_ptr<CameraIngest> CameraIngest::create(wg::Device& d, ReadFile read) {
  const auto& F = d.features;
  if (!(F.ycbcr && F.storageR8 && F.timeline && F.ahb)) return nullptr;
  std::unique_ptr<CameraIngest> c(new CameraIngest(d));
  auto b = read("spv/ingest_camera.spv");
  c->spirv.resize(b.size() / 4);
  memcpy(c->spirv.data(), b.data(), b.size());
  VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &tci;
  wg::check(vkCreateSemaphore(d.dev, &sci, nullptr, &c->timeline), "vkCreateSemaphore (ingest timeline)");
  return c;
}

CameraIngest::~CameraIngest() {
  vkDeviceWaitIdle(dev.dev);
  forget();
  destroyPipe();
  for (auto& p : pending) vkFreeCommandBuffers(dev.dev, dev.pool2, 1, &p.cb);
  if (timeline) vkDestroySemaphore(dev.dev, timeline, nullptr);
}

void CameraIngest::forget() {
  for (auto& [hb, im] : imports) {
    if (im.view) vkDestroyImageView(dev.dev, im.view, nullptr);
    if (im.img) vkDestroyImage(dev.dev, im.img, nullptr);
    if (im.mem) vkFreeMemory(dev.dev, im.mem, nullptr);
    AHardwareBuffer_release(hb);
  }
  imports.clear();
  if (pool) vkResetDescriptorPool(dev.dev, pool, 0);
}

void CameraIngest::destroyPipe() {
  if (pipe) vkDestroyPipeline(dev.dev, pipe, nullptr);
  if (layout) vkDestroyPipelineLayout(dev.dev, layout, nullptr);
  if (dsl) vkDestroyDescriptorSetLayout(dev.dev, dsl, nullptr);
  if (pool) vkDestroyDescriptorPool(dev.dev, pool, nullptr);
  if (sampler) vkDestroySampler(dev.dev, sampler, nullptr);
  if (conv) vkDestroySamplerYcbcrConversion(dev.dev, conv, nullptr);
  pipe = VK_NULL_HANDLE; layout = VK_NULL_HANDLE; dsl = VK_NULL_HANDLE; pool = VK_NULL_HANDLE; sampler = VK_NULL_HANDLE; conv = VK_NULL_HANDLE;
}

// ai: The conversion, sampler, layout and pipeline for the camera's format: made on its first buffer, and again
// ai: only if a buffer of another format comes (a camera reconfigured). The sampler is immutable in the layout, as
// ai: Vulkan requires of a YCbCr conversion.
bool CameraIngest::build(AHardwareBuffer* hb) {
  VkAndroidHardwareBufferFormatPropertiesANDROID fp{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
  VkAndroidHardwareBufferPropertiesANDROID pr{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
  pr.pNext = &fp;
  wg::check(vkGetAndroidHardwareBufferPropertiesANDROID(dev.dev, hb, &pr), "vkGetAndroidHardwareBufferPropertiesANDROID");
  if (pipe && fp.format == format && fp.externalFormat == externalFormat) return false;
  forget();
  destroyPipe();
  format = fp.format;
  externalFormat = fp.externalFormat;
  VkExternalFormatANDROID ef{VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID};
  ef.externalFormat = format == VK_FORMAT_UNDEFINED ? externalFormat : 0;
  VkSamplerYcbcrConversionCreateInfo yc{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO};
  yc.pNext = &ef;
  yc.format = format;
  // ai: RGB_IDENTITY: the planes' values as stored, no model or range conversion: G is Y
  yc.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY;
  yc.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
  yc.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
  yc.xChromaOffset = fp.suggestedXChromaOffset;
  yc.yChromaOffset = fp.suggestedYChromaOffset;
  yc.chromaFilter = VK_FILTER_NEAREST;
  yc.forceExplicitReconstruction = VK_FALSE;
  wg::check(vkCreateSamplerYcbcrConversion(dev.dev, &yc, nullptr, &conv), "vkCreateSamplerYcbcrConversion");
  VkSamplerYcbcrConversionInfo ci{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
  ci.conversion = conv;
  VkSamplerCreateInfo sm{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sm.pNext = &ci;
  sm.magFilter = sm.minFilter = VK_FILTER_NEAREST;
  sm.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sm.addressModeU = sm.addressModeV = sm.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sm.maxLod = 0;
  wg::check(vkCreateSampler(dev.dev, &sm, nullptr, &sampler), "vkCreateSampler (ycbcr)");
  VkDescriptorSetLayoutBinding bs[2] = {};
  bs[0].binding = 0; bs[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; bs[0].descriptorCount = 1; bs[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; bs[0].pImmutableSamplers = &sampler;
  bs[1].binding = 1; bs[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; bs[1].descriptorCount = 1; bs[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dci.bindingCount = 2; dci.pBindings = bs;
  wg::check(vkCreateDescriptorSetLayout(dev.dev, &dci, nullptr, &dsl), "vkCreateDescriptorSetLayout (ingest)");
  VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pc;
  wg::check(vkCreatePipelineLayout(dev.dev, &pli, nullptr, &layout), "vkCreatePipelineLayout (ingest)");
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size() * 4; smi.pCode = spirv.data();
  VkShaderModule mod;
  wg::check(vkCreateShaderModule(dev.dev, &smi, nullptr, &mod), "vkCreateShaderModule (ingest)");
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main";
  cpi.layout = layout;
  VkResult r = vkCreateComputePipelines(dev.dev, dev.cache, 1, &cpi, nullptr, &pipe);
  vkDestroyShaderModule(dev.dev, mod, nullptr);
  wg::check(r, "vkCreateComputePipelines (ingest)");
  // ai: a set a camera buffer; the pool sized for more than a camera keeps in flight (ImageReader's maxImages)
  // ai: YCbCr descriptors can take several slots each (combinedImageSamplerDescriptorCount); 4 covers three planes
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 * 4}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 64; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
  wg::check(vkCreateDescriptorPool(dev.dev, &dpi, nullptr, &pool), "vkCreateDescriptorPool (ingest)");
  char m[200];
  snprintf(m, sizeof m, "camera ingest: format %d, external format 0x%llx, model %d range %d (RGB identity taken), %s", (int)format, (unsigned long long)externalFormat, (int)fp.suggestedYcbcrModel,
           (int)fp.suggestedYcbcrRange, dev.twoQueues ? "its own queue" : "the decoder's queue");
  dev.log(m);
  return true;
}

// ai: A camera buffer as a Vulkan image, once: the camera cycles through a fixed pool, so each is imported on first
// ai: sight and kept (its reference held, AHardwareBuffer_acquire) until forget().
CameraIngest::Imported& CameraIngest::importOf(AHardwareBuffer* hb) {
  build(hb);
  auto it = imports.find(hb);
  if (it != imports.end()) return it->second;
  Imported im;
  AHardwareBuffer_Desc desc;
  AHardwareBuffer_describe(hb, &desc);
  im.w = desc.width; im.h = desc.height;
  VkAndroidHardwareBufferFormatPropertiesANDROID fp{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
  VkAndroidHardwareBufferPropertiesANDROID pr{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
  pr.pNext = &fp;
  wg::check(vkGetAndroidHardwareBufferPropertiesANDROID(dev.dev, hb, &pr), "vkGetAndroidHardwareBufferPropertiesANDROID");
  VkExternalFormatANDROID ef{VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID};
  ef.externalFormat = format == VK_FORMAT_UNDEFINED ? externalFormat : 0;
  VkExternalMemoryImageCreateInfo emi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  emi.pNext = &ef;
  emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.pNext = &emi;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = format;
  ici.extent = {desc.width, desc.height, 1};
  ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  wg::check(vkCreateImage(dev.dev, &ici, nullptr, &im.img), "vkCreateImage (camera)");
  VkImportAndroidHardwareBufferInfoANDROID imp{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
  imp.buffer = hb;
  VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  ded.pNext = &imp;
  ded.image = im.img;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &ded;
  mai.allocationSize = pr.allocationSize;
  mai.memoryTypeIndex = dev.memoryType(pr.memoryTypeBits, 0);
  wg::check(vkAllocateMemory(dev.dev, &mai, nullptr, &im.mem), "vkAllocateMemory (camera import)");
  wg::check(vkBindImageMemory(dev.dev, im.img, im.mem, 0), "vkBindImageMemory (camera)");
  VkSamplerYcbcrConversionInfo ci{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO};
  ci.conversion = conv;
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.pNext = &ci;
  vci.image = im.img;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = format;
  vci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  wg::check(vkCreateImageView(dev.dev, &vci, nullptr, &im.view), "vkCreateImageView (camera)");
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsa.descriptorPool = pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
  wg::check(vkAllocateDescriptorSets(dev.dev, &dsa, &im.set), "vkAllocateDescriptorSets (camera)");
  VkDescriptorImageInfo cam{VK_NULL_HANDLE, im.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = im.set; w.dstBinding = 0; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &cam;
  vkUpdateDescriptorSets(dev.dev, 1, &w, 0, nullptr);
  AHardwareBuffer_acquire(hb);
  return imports.emplace(hb, im).first->second;
}

uint64_t CameraIngest::ingest(AHardwareBuffer* hb, uint32_t x, uint32_t y, uint32_t w, uint32_t h, wg::Texture& ring, uint32_t layer) {
  // ai: command buffers of ingests that completed go back to the pool
  const uint64_t done = completed();
  for (size_t i = 0; i < pending.size();) {
    if (pending[i].value <= done) { vkFreeCommandBuffers(dev.dev, dev.pool2, 1, &pending[i].cb); pending[i] = pending.back(); pending.pop_back(); }
    else i++;
  }
  Imported& im = importOf(hb);
  if (x + w > im.w || y + h > im.h) throw wg::Error("ingest crop outside the camera's " + std::to_string(im.w) + " x " + std::to_string(im.h));
  if (im.ringView != ring.view) {
    VkDescriptorImageInfo ri{VK_NULL_HANDLE, ring.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = im.set; wr.dstBinding = 1; wr.descriptorCount = 1; wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr.pImageInfo = &ri;
    vkUpdateDescriptorSets(dev.dev, 1, &wr, 0, nullptr);
    im.ringView = ring.view;
  }
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = dev.pool2; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
  VkCommandBuffer cb;
  wg::check(vkAllocateCommandBuffers(dev.dev, &cai, &cb), "vkAllocateCommandBuffers (ingest)");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  // ai: the camera's buffer acquired from outside Vulkan (its producer wrote it), then released back to it after
  const uint32_t foreign = dev.features.foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
  VkImageMemoryBarrier acq{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  acq.srcAccessMask = 0; acq.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  acq.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; acq.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  acq.srcQueueFamilyIndex = foreign; acq.dstQueueFamilyIndex = dev.family;
  acq.image = im.img; acq.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &im.set, 0, nullptr);
  struct { int32_t ox, oy, sw, sh; float ix, iy; int32_t layer, pad; } pc{(int32_t)x, (int32_t)y, (int32_t)w, (int32_t)h, 1.0f / im.w, 1.0f / im.h, (int32_t)layer, 0};
  vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
  vkCmdDispatch(cb, (w + 15) / 16, (h + 15) / 16, 1);
  VkImageMemoryBarrier rel = acq;
  rel.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; rel.dstAccessMask = 0;
  rel.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; rel.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  rel.srcQueueFamilyIndex = dev.family; rel.dstQueueFamilyIndex = foreign;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &rel);
  wg::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer (ingest)");
  const uint64_t value = ++last;
  VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &value;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &ts;
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  si.signalSemaphoreCount = 1; si.pSignalSemaphores = &timeline;
  {
    // ai: a family with one queue: the decoder's thread submits on the same VkQueue (wg.h "Threads")
    std::unique_lock<std::mutex> l(dev.mu, std::defer_lock);
    if (!dev.twoQueues) l.lock();
    wg::check(vkQueueSubmit(dev.queue2, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit (ingest)");
  }
  pending.push_back({cb, value});
  return value;
}

// ai: a Vulkan 1.2 entry point, which the NDK's libvulkan at API 29 does not export: taken from the device
size_t CameraIngest::cameraClosed() {
  const size_t n = imports.size();
  if (last) {
    static PFN_vkWaitSemaphores wait = (PFN_vkWaitSemaphores)vkGetDeviceProcAddr(dev.dev, "vkWaitSemaphores");
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1; wi.pSemaphores = &timeline; wi.pValues = &last;
    if (wait(dev.dev, &wi, 1000000000ull) != VK_SUCCESS) vkQueueWaitIdle(dev.queue2);
  }
  for (auto& p : pending) vkFreeCommandBuffers(dev.dev, dev.pool2, 1, &p.cb);
  pending.clear();
  forget();
  return n;
}

uint64_t CameraIngest::completed() const {
  static PFN_vkGetSemaphoreCounterValue get = (PFN_vkGetSemaphoreCounterValue)vkGetDeviceProcAddr(dev.dev, "vkGetSemaphoreCounterValue");
  uint64_t v = 0;
  get(dev.dev, timeline, &v);
  return v;
}

bool CameraIngest::done(uint64_t value) const { return completed() >= value; }

#else

std::unique_ptr<CameraIngest> CameraIngest::create(wg::Device&, ReadFile) { return nullptr; }
CameraIngest::~CameraIngest() {}
uint64_t CameraIngest::ingest(AHardwareBuffer*, uint32_t, uint32_t, uint32_t, uint32_t, wg::Texture&, uint32_t) { return 0; }
bool CameraIngest::done(uint64_t) const { return true; }
uint64_t CameraIngest::completed() const { return 0; }
void CameraIngest::forget() {}
size_t CameraIngest::cameraClosed() { return 0; }

#endif

}  // namespace lizard
