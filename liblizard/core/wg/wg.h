// ai: wg: the part of WebGPU the decoder's host uses, over Vulkan. The host was written
// ai: against WebGPU (liblizard/gpu/decoder.mjs), so its port keeps WebGPU's rules here rather than in every caller:
// ai:   - every buffer and texture reads as zeros until written (a fill at creation);
// ai:   - queue writes land before the commands of the next submit and after everything submitted before them;
// ai:   - each dispatch sees everything the commands before it wrote (a memory barrier between commands; WebGPU on
// ai:     Vulkan runs a pass's dispatches one after the other too, so this is the web's order, not a new one);
// ai:   - textures are 2D arrays of r8unorm, sampled with textureLoad, kept in the GENERAL layout for their life;
// ai:   - timestamps come back in nanoseconds.
// ai: Threads: one thread at a time records and submits (the decoder's: encoder(), an Encoder's commands, submit(),
// ai: flush()), and the command pool is touched by that thread alone. Any thread may create and destroy buffers and
// ai: textures, write a buffer and drop a Ticket: the pending writes and fills and the queue are under `mu`, and a
// ai: dropped Ticket's command buffers wait in `retired` for the recording thread to free (Vulkan wants a pool's
// ai: allocations, frees and recording in one thread at a time). The camera ingest has its own pool and, where the
// ai: family has two, its own queue (core/ingest/camera.cpp); where it has one, its submit takes `mu` too.
// ai: (Until 2026-09-30 nothing was locked: the receiver's camera thread freed command buffers through a slot's
// ai: ticket and, on the host-luma path, recorded and submitted its uploads, against the decoder thread on the
// ai: same pool and queue; the 4090 lost the device once in a run of the receiver on it, Xid 69.)
#pragma once
#if defined(__ANDROID__) && !defined(VK_USE_PLATFORM_ANDROID_KHR)
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif
// ai: Vulkan loaded at run time (2026-10-03, for the library: third_party/volk): no link to libvulkan, so a machine with
// ai: no Vulkan loads the code and decodes on the CPU, and Windows and macOS need no import library. Every vk* name is
// ai: volk's pointer, set by Device::create (volkLoadInstance: device calls dispatch through the loader, so two devices
// ai: share a process, the decoder's and the painter's).
#include "volk.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace wg {

// ai: WebGPU's usage bits (GPUBufferUsage), as the setup and the host's code state them.
enum : uint32_t { MAP_READ = 1, MAP_WRITE = 2, COPY_SRC = 4, COPY_DST = 8, UNIFORM = 64, STORAGE = 128, INDIRECT = 256, QUERY_RESOLVE = 512 };
// ai: A bind group layout entry: what the shader binds there.
enum class Bind : uint8_t { Tex, RO, RW, Uniform };

struct Error : std::runtime_error { using std::runtime_error::runtime_error; VkResult code = VK_SUCCESS; };
// ai: An allocation the device refused: the host halves its batch on this, as the web does on an out-of-memory scope.
struct OutOfMemory : Error { using Error::Error; };

struct Limits {
  uint64_t maxBufferSize = 0, maxStorageBufferRange = 0, deviceLocalBytes = 0;
  uint32_t maxComputeSharedMemorySize = 0, maxPerStageStorageBuffers = 0, maxImageArrayLayers = 0, maxComputeWorkGroupInvocations = 0;
  uint32_t maxUniformBufferRange = 0, minStorageAlign = 0, minUniformAlign = 0;
};

struct Features {
  bool f16 = false, storage16 = false, int8 = false, dot = false, subgroupShuffle = false, timestamps = false;
  bool zeroInit = false;   // ai: shaderZeroInitializeWorkgroupMemory: the SPIR-V zeroes workgroup memory by initializer
  // ai: the camera's zero-copy ingest (core/ingest/): YCbCr sampling, an r8 storage image, timeline semaphores, and
  // ai: on Android the AHardwareBuffer import; all four or the ingest copies luma instead
  bool ycbcr = false, storageR8 = false, timeline = false, ahb = false, foreign = false;
  // ai: VK_KHR_cooperative_matrix (the nets' GEMMs on the device's matrix units, core/nets/): the shapes it offers,
  // ai: "MxNxK A,B,C,Result scope" a shape (KHR types: 0 f16, 1 f32, 2 f64, 3 i8, 4 i16, 5 i32, 6 i64, 7 u8, ...)
  bool coopmat = false;
  struct CoopShape { uint32_t M, N, K; int A, B, C, R; bool subgroup; };
  std::vector<CoopShape> coopShapes;
  uint32_t subgroupSize = 0;
};

struct Device;

struct Buffer {
  Device* dev = nullptr;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  uint64_t size = 0;
  uint32_t usage = 0;
  uint8_t* mapped = nullptr;   // ai: MAP_READ buffers, host-visible and mapped for life
  bool coherent = true;
  std::string id;
  ~Buffer();
};

struct Texture {
  Device* dev = nullptr;
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;   // ai: the whole array, 2D_ARRAY, as texture_2d_array reads it
  uint32_t w = 0, h = 0, layers = 0;
  bool storage = false;
  std::string id;
  ~Texture();
};

struct BindGroupLayout {
  Device* dev = nullptr;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  std::vector<Bind> entries;
  std::string id;
  ~BindGroupLayout();
};

struct Pipeline {
  Device* dev = nullptr;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  std::shared_ptr<BindGroupLayout> bgl;
  std::string id, label;
  ~Pipeline();
};

// ai: A resource bound at a binding: a buffer (whole, as every group here binds them) or a texture's array view.
struct Resource {
  Buffer* buf = nullptr;
  Texture* tex = nullptr;
  Resource(Buffer* b) : buf(b) {}
  Resource(Texture* t) : tex(t) {}
  Resource(const std::shared_ptr<Buffer>& b) : buf(b.get()) {}
  Resource(const std::shared_ptr<Texture>& t) : tex(t.get()) {}
};

struct BindGroup {
  Device* dev = nullptr;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  std::shared_ptr<BindGroupLayout> bgl;
  std::string id;
  ~BindGroup();
};

struct QuerySet {
  Device* dev = nullptr;
  VkQueryPool pool = VK_NULL_HANDLE;
  uint32_t count = 0;
  std::string id;
  ~QuerySet();
};

// ai: A command buffer being recorded: WebGPU's command encoder and compute pass in one. A pass is only a timestamp
// ai: pair's scope here (Vulkan has no compute pass); every command is behind a memory barrier.
struct Encoder {
  Device* dev = nullptr;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  std::vector<std::string> trace;   // ai: the commands as the mock records them, when the device traces
  int commands = 0;
  bool inPass = false;
  QuerySet* passQs = nullptr;
  int passEnd = -1;
  std::vector<QuerySet*> reset;     // ai: query sets reset at the head of this command buffer
  const Pipeline* pipe = nullptr;

  void beginComputePass(QuerySet* qs = nullptr, int begin = -1, int end = -1);
  void endPass();
  void setPipeline(const Pipeline& p);
  void setBindGroup(const BindGroup& g);
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
  void dispatchIndirect(const Buffer& b, uint64_t offset);
  void copyBufferToBuffer(const Buffer& src, uint64_t so, const Buffer& dst, uint64_t dof, uint64_t size);
  // ai: rows of w bytes, `stride` apart in src from `offset`, into layer `layer` of tex at (0, 0).
  void copyBufferToTexture(const Buffer& src, uint64_t offset, uint32_t stride, const Texture& tex, uint32_t layer, uint32_t w, uint32_t h);
  // ai: layer to layer, w x h from (0, 0) (the ring's slot into a lane's frame, as the web's copyTextureToTexture).
  void copyTextureToTexture(const Texture& src, uint32_t srcLayer, const Texture& dst, uint32_t dstLayer, uint32_t w, uint32_t h);
  // ai: a layer's w x h from (0, 0) into a buffer, rows of w bytes (an r8 texture: LIZ_DUMP's look at the ring)
  void copyTextureToBuffer(const Texture& src, uint32_t layer, uint32_t w, uint32_t h, const Buffer& dst, uint64_t offset);
  void clearBuffer(const Buffer& b, uint64_t offset = 0, uint64_t size = ~0ull);
  void resolveQuerySet(const QuerySet& qs, uint32_t first, uint32_t count, const Buffer& dst, uint64_t offset);
  void hazard();
};

// ai: A submitted command buffer's fence: done() polls it, wait() blocks on it.
struct Ticket {
  Device* dev = nullptr;
  VkFence fence = VK_NULL_HANDLE;
  VkCommandBuffer cbs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  std::vector<std::shared_ptr<Buffer>> keep;   // ai: staging the head copied from
  bool done();
  void wait();
  ~Ticket();
};

// ai: A host's extras for Device::create: instance and device extensions (a window surface, the swapchain, present_id
// ai: and present_wait, whose features are enabled with them where the device has both), and a queue family that draws
// ai: (the first family that both computes and draws is taken anyway; this refuses a device with none).
struct DeviceExtras {
  std::vector<std::string> instanceExts, deviceExts;
  bool graphics = false;
};

struct Device {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  // ai: the ingest's queue (the family's second where it has one, else queue itself) and its own command pool, for
  // ai: the camera thread: a camera frame is sampled into the ring there without waiting behind a batch
  VkQueue queue2 = VK_NULL_HANDLE;
  VkCommandPool pool2 = VK_NULL_HANDLE;
  bool twoQueues = false;
  uint32_t family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkPipelineCache cache = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties mem{};
  std::string name, driver;
  uint32_t apiVersion = 0, vendorId = 0, deviceId = 0;
  Limits limits;
  Features features;
  double timestampPeriod = 1.0;   // ai: ns a tick
  uint64_t timestampMask = ~0ull;
  std::atomic<uint64_t> allocated{0};   // ai: bytes of device memory this device holds (the web's fh.allocated)
  // ai: guards writes, inits, staging and retired, and the queue's submits where the ingest shares the queue
  std::mutex mu;
  // ai: command buffers of dropped Tickets, freed by the recording thread at its next encoder()
  std::vector<VkCommandBuffer> retired;
  std::function<void(const std::string&)> log;
  FILE* traceOut = nullptr;       // ai: LIZ_TRACE: a JSON line an event, as liblizard/gen/mockgpu.mjs records them
  std::string cachePath;

  // ai: The queue's writes waiting for the next submit (WebGPU's queue.writeBuffer), with their staging.
  struct Write { Buffer* dst; uint64_t off; std::vector<uint8_t> data; };
  std::vector<Write> writes;
  // ai: fills and layout changes of new objects, by the object they are for (its destructor drops its own)
  // ai: phase 0 (fills, clears, layout changes) runs before phase 1 (uploads' copies), a barrier between.
  struct Init { const void* owner; std::function<void(VkCommandBuffer)> f; int phase = 0; };
  std::vector<Init> inits;

  // ai: want: a substring of the device name to pick (LIZ_VK_DEVICE), else the first discrete, else the first.
  // ai: extras (2026-10-07, the desktop sender presents from the painter's device): what a host asks beyond the
  // ai: kernels' needs, each enabled where offered and left out where not (`has` says which took).
  static std::unique_ptr<Device> create(const std::string& want = "", bool validate = false, std::function<void(const std::string&)> log = nullptr,
                                        const DeviceExtras& extras = DeviceExtras());
  // ai: the device extensions enabled (the kernels' and the extras' that the device had)
  std::vector<std::string> enabled;
  bool has(const char* ext) const;
  ~Device();

  // ai: zero: filled with zeros at the next submit (WebGPU's rule); false only where an upload covers it whole.
  std::shared_ptr<Buffer> createBuffer(uint64_t size, uint32_t usage, const std::string& id = "", bool zero = true);
  // ai: storage: also written by a compute shader (the camera's ingest into the ring), as an r8 storage image.
  std::shared_ptr<Texture> createTexture(uint32_t w, uint32_t h, uint32_t layers, const std::string& id = "", bool storage = false);
  std::shared_ptr<BindGroupLayout> createBindGroupLayout(const std::vector<Bind>& entries, const std::string& id = "");
  // ai: spec: values for specialisation constants 0, 1, ... (u32 each), the native kernels' shapes (core/nets/)
  std::shared_ptr<Pipeline> createPipeline(const std::vector<uint32_t>& spirv, std::shared_ptr<BindGroupLayout> bgl, const std::string& id = "", const std::string& label = "", const std::vector<uint32_t>& spec = {});
  std::shared_ptr<BindGroup> createBindGroup(std::shared_ptr<BindGroupLayout> bgl, const std::vector<Resource>& entries, const std::string& id = "");
  std::shared_ptr<QuerySet> createQuerySet(uint32_t count, const std::string& id = "");

  void writeBuffer(Buffer& b, uint64_t off, const void* data, uint64_t size);
  // ai: A large write (the setup's tables): through a host-visible staging buffer, copied at the head of the next
  // ai: submit; the staging lives until that submit is done.
  void upload(Buffer& b, uint64_t off, const void* data, uint64_t size);
  std::vector<std::shared_ptr<Buffer>> staging;
  Encoder encoder();
  // ai: The pending writes and inits first (their own command buffer, behind a barrier), then enc's commands; with
  // ai: wait set, the submit waits for that timeline semaphore to reach waitValue (the camera ingest's) before it
  // ai: copies anything.
  std::shared_ptr<Ticket> submit(Encoder& enc, VkSemaphore wait = VK_NULL_HANDLE, uint64_t waitValue = 0);
  // ai: The pending writes and inits alone, waited for (the setup's uploads).
  void flush();
  void waitIdle();
  // ai: A MAP_READ buffer's bytes once its ticket is done (invalidated where the memory is not coherent).
  const uint8_t* read(Buffer& b, uint64_t off, uint64_t size);

  void loadCache(const std::string& path);
  void saveCache();

  void traceLine(const std::string& json);
  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0);
  std::atomic<int> nextId{0};
  std::string autoId(char kind);
};

void check(VkResult r, const char* what);
std::string jsonStr(const std::string& s);

}  // namespace wg
