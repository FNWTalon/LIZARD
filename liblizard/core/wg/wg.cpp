// ai: wg over Vulkan (wg.h). What each WebGPU rule costs here: a buffer's zero fill and a texture's clear and layout
// ai: change are recorded when it is made and run at the head of the next submit (`inits`), with the queue's writes
// ai: (`writes`, vkCmdUpdateBuffer in pieces of 64 KB or less); a global memory barrier sits before every command
// ai: of an encoder but its first (`hazard`), and after the head. Nothing waits on the device but Ticket::wait,
// ai: flush() and waitIdle().
#include "wg.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace wg {

static const char* resultName(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
    default: return "VkResult";
  }
}

void check(VkResult r, const char* what) {
  if (r == VK_SUCCESS) return;
  std::string m = std::string(what) + ": " + resultName(r) + " (" + std::to_string((int)r) + ")";
  if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) { OutOfMemory e(m); e.code = r; throw e; }
  Error e(m);
  e.code = r;
  throw e;
}

std::string jsonStr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o + "\"";
}

static std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }

std::string Device::autoId(char kind) { return std::string("n") + kind + std::to_string(nextId++); }

void Device::traceLine(const std::string& json) {
  if (!traceOut) return;
  fputs(json.c_str(), traceOut);
  fputc('\n', traceOut);
}

uint32_t Device::memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
  for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want && !(mem.memoryTypes[i].propertyFlags & avoid)) return i;
  return UINT32_MAX;
}

// ai: The layers and extensions an instance or device offers, by name.
static bool hasLayer(const char* name) {
  uint32_t n = 0;
  vkEnumerateInstanceLayerProperties(&n, nullptr);
  std::vector<VkLayerProperties> v(n);
  vkEnumerateInstanceLayerProperties(&n, v.data());
  for (auto& l : v) if (!strcmp(l.layerName, name)) return true;
  return false;
}
static std::vector<std::string> deviceExtensions(VkPhysicalDevice p) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(p, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> v(n);
  vkEnumerateDeviceExtensionProperties(p, nullptr, &n, v.data());
  std::vector<std::string> out;
  for (auto& e : v) out.push_back(e.extensionName);
  return out;
}

bool Device::has(const char* ext) const { return std::find(enabled.begin(), enabled.end(), ext) != enabled.end(); }

std::unique_ptr<Device> Device::create(const std::string& want, bool validate, std::function<void(const std::string&)> log, const DeviceExtras& extras) {
  auto d = std::make_unique<Device>();
  d->log = log ? log : [](const std::string&) {};
  static const VkResult loader = volkInitialize();
  if (loader != VK_SUCCESS) throw Error("no Vulkan loader on this machine");
  uint32_t instVersion = VK_API_VERSION_1_1;
  auto enumVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
  if (enumVersion) enumVersion(&instVersion);
  if (instVersion < VK_API_VERSION_1_1) throw Error("Vulkan 1.1 is needed; the loader offers 1.0");
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "lizard";
  app.apiVersion = std::min(instVersion, (uint32_t)VK_API_VERSION_1_3);
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  // ai: devices that are a portability layer's (MoltenVK on macOS) are listed only when asked for
  std::vector<const char*> iexts;
  {
    uint32_t ni = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ni, nullptr);
    std::vector<VkExtensionProperties> ie(ni);
    vkEnumerateInstanceExtensionProperties(nullptr, &ni, ie.data());
    for (auto& e : ie)
      if (!strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        iexts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
      }
    // ai: the host's instance extensions (a window surface), where the loader offers them
    for (const std::string& x : extras.instanceExts)
      for (auto& e : ie)
        if (x == e.extensionName) { iexts.push_back(x.c_str()); break; }
  }
  ici.enabledExtensionCount = (uint32_t)iexts.size();
  ici.ppEnabledExtensionNames = iexts.data();
  const char* layer = "VK_LAYER_KHRONOS_validation";
  if (validate && hasLayer(layer)) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; d->log("validation layer on"); }
  else if (validate) d->log("validation asked, but no VK_LAYER_KHRONOS_validation here");
  check(vkCreateInstance(&ici, nullptr, &d->instance), "vkCreateInstance");
  volkLoadInstance(d->instance);

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(d->instance, &n, nullptr);
  if (!n) throw Error("no Vulkan device");
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(d->instance, &n, devs.data());
  VkPhysicalDevice pick = VK_NULL_HANDLE;
  std::string names;
  for (auto p : devs) {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(p, &pp);
    names += std::string(names.empty() ? "" : ", ") + pp.deviceName;
    if (!want.empty()) { if (!pick && lower(pp.deviceName).find(lower(want)) != std::string::npos) pick = p; }
    else if (!pick && pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pick = p;
  }
  if (!pick && !want.empty()) throw Error("no Vulkan device named like \"" + want + "\" among " + names);
  if (!pick) pick = devs[0];
  d->phys = pick;

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pick, &props);
  d->name = props.deviceName;
  d->apiVersion = std::min(props.apiVersion, app.apiVersion);
  d->vendorId = props.vendorID;
  d->deviceId = props.deviceID;
  if (d->apiVersion < VK_API_VERSION_1_1) throw Error(d->name + " offers Vulkan 1.0; 1.1 is needed");
  auto exts = deviceExtensions(pick);
  auto has = [&](const char* e) { return std::find(exts.begin(), exts.end(), e) != exts.end(); };
  const bool v12 = d->apiVersion >= VK_API_VERSION_1_2, v13 = d->apiVersion >= VK_API_VERSION_1_3;
  const bool dotExt = v13 || has(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
  const bool f16Ext = v12 || has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  const bool m4 = v13 || has(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
  const bool zeroExt = v13 || has(VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME);

  // ai: Features: f16 and 16-bit storage, int8, the integer dot product; the rest of the decoder is Vulkan 1.1.
  VkPhysicalDeviceShaderIntegerDotProductFeatures dotF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
  VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures zeroF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES};
  VkPhysicalDeviceSamplerYcbcrConversionFeatures ycF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  const bool cmExt = has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  VkPhysicalDeviceTimelineSemaphoreFeatures tlF{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  const bool tlExt = v12 || has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  VkPhysicalDeviceShaderFloat16Int8Features f16F{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice16BitStorageFeatures s16F{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  void** tail = &f2.pNext;
  auto chain = [&](auto& s) { *tail = &s; tail = &s.pNext; };
  chain(s16F);
  if (f16Ext) chain(f16F);
  if (dotExt) chain(dotF);
  if (zeroExt) chain(zeroF);
  chain(ycF);
  if (tlExt) chain(tlF);
  if (cmExt) chain(cmF);
  vkGetPhysicalDeviceFeatures2(pick, &f2);

  VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceMaintenance3Properties m3P{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
  VkPhysicalDeviceMaintenance4Properties m4P{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
  VkPhysicalDeviceDriverProperties drvP{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  tail = &p2.pNext;
  chain(sub);
  chain(m3P);
  if (m4) chain(m4P);
  if (v12) chain(drvP);
  vkGetPhysicalDeviceProperties2(pick, &p2);
  d->driver = v12 ? std::string(drvP.driverName) + " " + drvP.driverInfo : "";

  auto& F = d->features;
  F.storage16 = s16F.storageBuffer16BitAccess;
  F.f16 = f16Ext && f16F.shaderFloat16 && F.storage16;
  F.int8 = f16Ext && f16F.shaderInt8;
  F.dot = dotExt && dotF.shaderIntegerDotProduct;
  F.subgroupShuffle = (sub.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) && (sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
  F.subgroupSize = sub.subgroupSize;
  F.zeroInit = zeroExt && zeroF.shaderZeroInitializeWorkgroupMemory;
  F.ycbcr = ycF.samplerYcbcrConversion;
  F.coopmat = cmExt && cmF.cooperativeMatrix;
  if (F.coopmat) {
    auto get = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(d->instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t n = 0;
    if (get && get(pick, &n, nullptr) == VK_SUCCESS && n) {
      std::vector<VkCooperativeMatrixPropertiesKHR> v(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
      get(pick, &n, v.data());
      for (auto& p : v) F.coopShapes.push_back({p.MSize, p.NSize, p.KSize, (int)p.AType, (int)p.BType, (int)p.CType, (int)p.ResultType, p.scope == VK_SCOPE_SUBGROUP_KHR});
    }
  }
  F.timeline = tlExt && tlF.timelineSemaphore;
  F.storageR8 = f2.features.shaderStorageImageExtendedFormats;
#ifdef __ANDROID__
  F.ahb = has(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
  F.foreign = has(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
#endif

  auto& L = d->limits;
  L.maxStorageBufferRange = props.limits.maxStorageBufferRange;
  L.maxBufferSize = m4 ? m4P.maxBufferSize : m3P.maxMemoryAllocationSize;
  L.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
  L.maxPerStageStorageBuffers = props.limits.maxPerStageDescriptorStorageBuffers;
  L.maxImageArrayLayers = props.limits.maxImageArrayLayers;
  L.maxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
  L.maxUniformBufferRange = props.limits.maxUniformBufferRange;
  L.minStorageAlign = (uint32_t)props.limits.minStorageBufferOffsetAlignment;
  L.minUniformAlign = (uint32_t)props.limits.minUniformBufferOffsetAlignment;
  vkGetPhysicalDeviceMemoryProperties(pick, &d->mem);
  for (uint32_t i = 0; i < d->mem.memoryHeapCount; i++)
    if (d->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) L.deviceLocalBytes = std::max<uint64_t>(L.deviceLocalBytes, d->mem.memoryHeaps[i].size);
  d->timestampPeriod = props.limits.timestampPeriod;

  // ai: A queue that computes; the first such family, preferring one that also draws (the universal queue, where
  // ai: drivers put their timestamps).
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pick, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(pick, &qn, qf.data());
  int fam = -1;
  for (uint32_t i = 0; i < qn && fam < 0; i++) if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) fam = (int)i;
  for (uint32_t i = 0; i < qn && fam < 0; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) fam = (int)i;
  if (fam < 0) throw Error(d->name + ": no compute queue");
  if (extras.graphics && !(qf[fam].queueFlags & VK_QUEUE_GRAPHICS_BIT)) throw Error(d->name + ": no queue that both computes and draws");
  d->family = (uint32_t)fam;
  F.timestamps = qf[fam].timestampValidBits > 0 && props.limits.timestampComputeAndGraphics;
  d->timestampMask = qf[fam].timestampValidBits >= 64 ? ~0ull : ((1ull << qf[fam].timestampValidBits) - 1);

  // ai: Enable what exists of what the modules ask (liblizard/gen/gen.mjs lists the capabilities); nothing else.
  VkPhysicalDeviceShaderIntegerDotProductFeatures dotE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
  VkPhysicalDeviceShaderFloat16Int8Features f16E{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice16BitStorageFeatures s16E{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceFeatures2 e2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  tail = &e2.pNext;
  s16E.storageBuffer16BitAccess = s16F.storageBuffer16BitAccess;
  chain(s16E);
  std::vector<const char*> enable;
  if (f16Ext) {
    f16E.shaderFloat16 = f16F.shaderFloat16;
    f16E.shaderInt8 = f16F.shaderInt8;
    chain(f16E);
    if (!v12) enable.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
  }
  if (F.dot) {
    dotE.shaderIntegerDotProduct = VK_TRUE;
    chain(dotE);
    if (!v13) enable.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
  }
  if (m4 && !v13) enable.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
  VkPhysicalDeviceZeroInitializeWorkgroupMemoryFeatures zeroE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES};
  if (F.zeroInit) {
    zeroE.shaderZeroInitializeWorkgroupMemory = VK_TRUE;
    chain(zeroE);
    if (!v13) enable.push_back(VK_KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_EXTENSION_NAME);
  }
  VkPhysicalDeviceSamplerYcbcrConversionFeatures ycE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
  if (F.ycbcr) { ycE.samplerYcbcrConversion = VK_TRUE; chain(ycE); }
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  if (F.coopmat) { cmE.cooperativeMatrix = VK_TRUE; chain(cmE); enable.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); }
  VkPhysicalDeviceTimelineSemaphoreFeatures tlE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  if (F.timeline) { tlE.timelineSemaphore = VK_TRUE; chain(tlE); if (!v12) enable.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME); }
  // ai: the host's device extensions, where the device has them; present_id and present_wait with their features
  // ai: where both are there (else neither: a wait needs the id)
  auto enabledHas = [&](const char* e) {
    for (const char* s : enable) if (!strcmp(s, e)) return true;
    return false;
  };
  for (const std::string& x : extras.deviceExts)
    if (has(x.c_str()) && !enabledHas(x.c_str())) enable.push_back(x.c_str());
  VkPhysicalDevicePresentIdFeaturesKHR pidE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
  VkPhysicalDevicePresentWaitFeaturesKHR pwE{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
  if (enabledHas(VK_KHR_PRESENT_ID_EXTENSION_NAME) || enabledHas(VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
    VkPhysicalDevicePresentIdFeaturesKHR pidQ{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
    VkPhysicalDevicePresentWaitFeaturesKHR pwQ{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR};
    pidQ.pNext = &pwQ;
    VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q2.pNext = &pidQ;
    vkGetPhysicalDeviceFeatures2(pick, &q2);
    const bool both = enabledHas(VK_KHR_PRESENT_ID_EXTENSION_NAME) && enabledHas(VK_KHR_PRESENT_WAIT_EXTENSION_NAME) && pidQ.presentId && pwQ.presentWait;
    if (both) { pidE.presentId = VK_TRUE; pwE.presentWait = VK_TRUE; chain(pidE); chain(pwE); }
    else enable.erase(std::remove_if(enable.begin(), enable.end(), [](const char* s) {
      return !strcmp(s, VK_KHR_PRESENT_ID_EXTENSION_NAME) || !strcmp(s, VK_KHR_PRESENT_WAIT_EXTENSION_NAME); }), enable.end());
  }
  e2.features.shaderStorageImageExtendedFormats = F.storageR8;
  // ai: a portability layer's device (MoltenVK) must have its subset enabled when it lists it
  if (has("VK_KHR_portability_subset")) enable.push_back("VK_KHR_portability_subset");
#ifdef __ANDROID__
  if (F.ahb) enable.push_back(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
  if (F.foreign) enable.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
#endif
  float prio[2] = {1.0f, 1.0f};
  d->twoQueues = qf[fam].queueCount > 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = d->family;
  qci.queueCount = d->twoQueues ? 2 : 1;
  qci.pQueuePriorities = prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &e2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)enable.size();
  dci.ppEnabledExtensionNames = enable.data();
  check(vkCreateDevice(pick, &dci, nullptr, &d->dev), "vkCreateDevice");
  for (const char* e : enable) d->enabled.push_back(e);
  vkGetDeviceQueue(d->dev, d->family, 0, &d->queue);
  if (d->twoQueues) vkGetDeviceQueue(d->dev, d->family, 1, &d->queue2); else d->queue2 = d->queue;
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = d->family;
  check(vkCreateCommandPool(d->dev, &cpi, nullptr, &d->pool), "vkCreateCommandPool");
  check(vkCreateCommandPool(d->dev, &cpi, nullptr, &d->pool2), "vkCreateCommandPool");
  VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  check(vkCreatePipelineCache(d->dev, &pci, nullptr, &d->cache), "vkCreatePipelineCache");

  char line[512];
  snprintf(line, sizeof line, "device %s (Vulkan %u.%u.%u%s%s): f16 %d, int8 %d, dot %d, subgroup shuffle %d (size %u), zero-init %d, ingest ycbcr %d r8 %d timeline %d ahb %d, queues %d, timestamps %d (%.3f ns a tick); shared memory %u B, storage range %llu MB, buffer %llu MB, device memory %llu MB",
           d->name.c_str(), VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), d->driver.empty() ? "" : ", ", d->driver.c_str(),
           F.f16, F.int8, F.dot, F.subgroupShuffle, F.subgroupSize, F.zeroInit, F.ycbcr, F.storageR8, F.timeline, F.ahb, d->twoQueues ? 2 : 1, F.timestamps, d->timestampPeriod, L.maxComputeSharedMemorySize,
           (unsigned long long)(L.maxStorageBufferRange >> 20), (unsigned long long)(L.maxBufferSize >> 20), (unsigned long long)(L.deviceLocalBytes >> 20));
  d->log(line);
  if (F.coopmat) {
    std::string cm = "cooperative matrix:";
    for (auto& c : F.coopShapes) cm += " " + std::to_string(c.M) + "x" + std::to_string(c.N) + "x" + std::to_string(c.K) + " " + std::to_string(c.A) + "," + std::to_string(c.B) + "," + std::to_string(c.C) + "," + std::to_string(c.R) + (c.subgroup ? "" : " (not subgroup)") + ";";
    d->log(cm);
  }
  return d;
}

Device::~Device() {
  if (dev) vkDeviceWaitIdle(dev);
  writes.clear();
  inits.clear();
  if (cache) { saveCache(); vkDestroyPipelineCache(dev, cache, nullptr); }
  if (pool) vkDestroyCommandPool(dev, pool, nullptr);
  if (pool2) vkDestroyCommandPool(dev, pool2, nullptr);
  if (dev) vkDestroyDevice(dev, nullptr);
  if (instance) vkDestroyInstance(instance, nullptr);
  if (traceOut && traceOut != stdout && traceOut != stderr) fclose(traceOut);
}

void Device::waitIdle() { check(vkDeviceWaitIdle(dev), "vkDeviceWaitIdle"); }

// ai: The pipeline cache from a file this device wrote before (Vulkan refuses another device's or driver's through
// ai: the header), and back to it at the end: a second start compiles nothing (STATUS "The ramp at the start").
void Device::loadCache(const std::string& path) {
  cachePath = path;
  std::ifstream f(path, std::ios::binary);
  if (!f) return;
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.size() < 32) return;
  VkPipelineCache c = VK_NULL_HANDLE;
  VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  pci.initialDataSize = data.size();
  pci.pInitialData = data.data();
  if (vkCreatePipelineCache(dev, &pci, nullptr, &c) == VK_SUCCESS) {
    vkDestroyPipelineCache(dev, cache, nullptr);
    cache = c;
    log("pipeline cache: " + std::to_string(data.size() / 1024) + " KB from " + path);
  }
}

void Device::saveCache() {
  if (cachePath.empty() || !cache) return;
  size_t n = 0;
  if (vkGetPipelineCacheData(dev, cache, &n, nullptr) != VK_SUCCESS || !n) return;
  std::vector<char> data(n);
  if (vkGetPipelineCacheData(dev, cache, &n, data.data()) != VK_SUCCESS) return;
  std::ofstream f(cachePath + ".tmp", std::ios::binary);
  f.write(data.data(), (std::streamsize)n);
  f.close();
  std::rename((cachePath + ".tmp").c_str(), cachePath.c_str());
}

// ai: A buffer: device-local, or host-visible and cached for MAP_READ (the readbacks); zero-filled at the next
// ai: submit, as WebGPU gives every buffer.
std::shared_ptr<Buffer> Device::createBuffer(uint64_t size, uint32_t usage, const std::string& id, bool zero) {
  auto b = std::make_shared<Buffer>();
  b->dev = this;
  b->size = size;
  b->usage = usage;
  b->id = id.empty() ? autoId('b') : id;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = std::max<uint64_t>(size, 16);
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (usage & STORAGE) bci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (usage & UNIFORM) bci.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (usage & INDIRECT) bci.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(dev, &bci, nullptr, &b->buf), "vkCreateBuffer");
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, b->buf, &req);
  const bool host = usage & (MAP_READ | MAP_WRITE);
  // ai: A readback wants cached memory (the host reads it); a staging buffer coherent memory (the host writes it).
  const VkMemoryPropertyFlags hostWant = (usage & MAP_READ) ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t type = host ? memoryType(req.memoryTypeBits, hostWant) : memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (type == UINT32_MAX) type = host ? memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) : memoryType(req.memoryTypeBits, 0);
  if (type == UINT32_MAX) { vkDestroyBuffer(dev, b->buf, nullptr); b->buf = VK_NULL_HANDLE; throw Error("no memory type for a buffer of " + std::to_string(size) + " B"); }
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  VkResult r = vkAllocateMemory(dev, &mai, nullptr, &b->mem);
  if (r != VK_SUCCESS) { vkDestroyBuffer(dev, b->buf, nullptr); b->buf = VK_NULL_HANDLE; check(r, ("vkAllocateMemory " + std::to_string(req.size) + " B").c_str()); }
  check(vkBindBufferMemory(dev, b->buf, b->mem, 0), "vkBindBufferMemory");
  allocated += size;
  if (host) {
    b->coherent = mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    void* p = nullptr;
    check(vkMapMemory(dev, b->mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory");
    b->mapped = (uint8_t*)p;
    memset(p, 0, (size_t)size);
    if (!b->coherent) {
      VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      mr.memory = b->mem; mr.size = VK_WHOLE_SIZE;
      vkFlushMappedMemoryRanges(dev, 1, &mr);
    }
  } else if (zero) {
    VkBuffer vb = b->buf;
    std::lock_guard<std::mutex> l(mu);
    inits.push_back({b.get(), [vb](VkCommandBuffer cb) { vkCmdFillBuffer(cb, vb, 0, VK_WHOLE_SIZE, 0); }});
  }
  if (traceOut) traceLine("{\"op\":\"buffer\",\"id\":" + jsonStr(b->id) + ",\"size\":" + std::to_string(size) + ",\"usage\":" + std::to_string(usage) + "}");
  return b;
}

Buffer::~Buffer() {
  if (!dev) return;
  // ai: A write or a fill still waiting for a submit names this buffer's handle, which goes now: drop them.
  {
    std::lock_guard<std::mutex> l(dev->mu);
    auto& w = dev->writes;
    w.erase(std::remove_if(w.begin(), w.end(), [this](const Device::Write& x) { return x.dst == this; }), w.end());
    auto& in = dev->inits;
    in.erase(std::remove_if(in.begin(), in.end(), [this](const Device::Init& x) { return x.owner == this; }), in.end());
  }
  if (buf) vkDestroyBuffer(dev->dev, buf, nullptr);
  if (mem) { vkFreeMemory(dev->dev, mem, nullptr); dev->allocated -= size; }
  if (dev->traceOut) dev->traceLine("{\"op\":\"destroy\",\"id\":" + jsonStr(id) + "}");
}

// ai: A texture: r8unorm, a 2D array of `layers`, sampled (textureLoad) and copied into; cleared and moved to the
// ai: GENERAL layout at the next submit, where it stays.
std::shared_ptr<Texture> Device::createTexture(uint32_t w, uint32_t h, uint32_t layers, const std::string& id, bool storage) {
  auto t = std::make_shared<Texture>();
  t->dev = this;
  t->w = w; t->h = h; t->layers = layers;
  t->storage = storage;
  t->id = id.empty() ? autoId('t') : id;
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_R8_UNORM;
  ici.extent = {w, h, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = layers;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  check(vkCreateImage(dev, &ici, nullptr, &t->img), "vkCreateImage");
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(dev, t->img, &req);
  uint32_t type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (type == UINT32_MAX) type = memoryType(req.memoryTypeBits, 0);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  VkResult r = vkAllocateMemory(dev, &mai, nullptr, &t->mem);
  if (r != VK_SUCCESS) { vkDestroyImage(dev, t->img, nullptr); t->img = VK_NULL_HANDLE; check(r, "vkAllocateMemory (texture)"); }
  check(vkBindImageMemory(dev, t->img, t->mem, 0), "vkBindImageMemory");
  allocated += (uint64_t)w * h * layers;
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t->img;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  vci.format = VK_FORMAT_R8_UNORM;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
  check(vkCreateImageView(dev, &vci, nullptr, &t->view), "vkCreateImageView");
  VkImage img = t->img;
  std::unique_lock<std::mutex> l(mu);
  inits.push_back({t.get(), [img, layers](VkCommandBuffer cb) {
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
  }});
  l.unlock();
  if (traceOut) traceLine("{\"op\":\"texture\",\"id\":" + jsonStr(t->id) + ",\"size\":[" + std::to_string(w) + "," + std::to_string(h) + "," + std::to_string(layers) + "]}");
  return t;
}

Texture::~Texture() {
  if (!dev) return;
  {
    std::lock_guard<std::mutex> l(dev->mu);
    auto& in = dev->inits;
    in.erase(std::remove_if(in.begin(), in.end(), [this](const Device::Init& x) { return x.owner == this; }), in.end());
  }
  if (view) vkDestroyImageView(dev->dev, view, nullptr);
  if (img) vkDestroyImage(dev->dev, img, nullptr);
  if (mem) { vkFreeMemory(dev->dev, mem, nullptr); dev->allocated -= (uint64_t)w * h * layers; }
  if (dev->traceOut) dev->traceLine("{\"op\":\"destroy\",\"id\":" + jsonStr(id) + "}");
}

static VkDescriptorType descriptorOf(Bind b) {
  switch (b) {
    case Bind::Tex: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case Bind::Uniform: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
}
static const char* bindName(Bind b) { return b == Bind::Tex ? "tex" : b == Bind::RO ? "ro" : b == Bind::RW ? "rw" : "uniform"; }

std::shared_ptr<BindGroupLayout> Device::createBindGroupLayout(const std::vector<Bind>& entries, const std::string& id) {
  auto L = std::make_shared<BindGroupLayout>();
  L->dev = this;
  L->entries = entries;
  L->id = id.empty() ? autoId('L') : id;
  std::vector<VkDescriptorSetLayoutBinding> bs;
  for (uint32_t i = 0; i < entries.size(); i++) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = i;
    b.descriptorType = descriptorOf(entries[i]);
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bs.push_back(b);
  }
  VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  ci.bindingCount = (uint32_t)bs.size();
  ci.pBindings = bs.data();
  check(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &L->dsl), "vkCreateDescriptorSetLayout");
  if (traceOut) {
    std::string e;
    for (auto b : entries) e += std::string(e.empty() ? "" : ",") + "\"" + bindName(b) + "\"";
    traceLine("{\"op\":\"bgl\",\"id\":" + jsonStr(L->id) + ",\"entries\":[" + e + "]}");
  }
  return L;
}

BindGroupLayout::~BindGroupLayout() { if (dev && dsl) vkDestroyDescriptorSetLayout(dev->dev, dsl, nullptr); }

std::shared_ptr<Pipeline> Device::createPipeline(const std::vector<uint32_t>& spirv, std::shared_ptr<BindGroupLayout> bgl, const std::string& id, const std::string& label, const std::vector<uint32_t>& spec) {
  auto p = std::make_shared<Pipeline>();
  p->dev = this;
  p->bgl = bgl;
  p->id = id.empty() ? autoId('p') : id;
  p->label = label;
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size() * 4;
  smi.pCode = spirv.data();
  VkShaderModule sm;
  check(vkCreateShaderModule(dev, &smi, nullptr, &sm), ("vkCreateShaderModule " + p->id).c_str());
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &bgl->dsl;
  check(vkCreatePipelineLayout(dev, &pli, nullptr, &p->layout), "vkCreatePipelineLayout");
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = sm;
  cpi.stage.pName = "main";
  std::vector<VkSpecializationMapEntry> me;
  for (uint32_t i = 0; i < spec.size(); i++) me.push_back({i, 4 * i, 4});
  VkSpecializationInfo si{(uint32_t)me.size(), me.data(), spec.size() * 4, spec.data()};
  if (!spec.empty()) cpi.stage.pSpecializationInfo = &si;
  cpi.layout = p->layout;
  VkResult r = vkCreateComputePipelines(dev, cache, 1, &cpi, nullptr, &p->pipe);
  vkDestroyShaderModule(dev, sm, nullptr);
  check(r, ("vkCreateComputePipelines " + p->id + (label.empty() ? "" : " (" + label + ")")).c_str());
  if (traceOut) traceLine("{\"op\":\"pipeline\",\"id\":" + jsonStr(p->id) + ",\"groups\":[" + jsonStr(bgl->id) + "]}");
  return p;
}

Pipeline::~Pipeline() {
  if (!dev) return;
  if (pipe) vkDestroyPipeline(dev->dev, pipe, nullptr);
  if (layout) vkDestroyPipelineLayout(dev->dev, layout, nullptr);
}

std::shared_ptr<BindGroup> Device::createBindGroup(std::shared_ptr<BindGroupLayout> bgl, const std::vector<Resource>& entries, const std::string& id) {
  if (entries.size() != bgl->entries.size()) throw Error("bind group of " + std::to_string(entries.size()) + " entries on layout " + bgl->id + " of " + std::to_string(bgl->entries.size()));
  auto g = std::make_shared<BindGroup>();
  g->dev = this;
  g->bgl = bgl;
  g->id = id.empty() ? autoId('g') : id;
  uint32_t nTex = 0, nSto = 0, nUni = 0;
  for (auto b : bgl->entries) { if (b == Bind::Tex) nTex++; else if (b == Bind::Uniform) nUni++; else nSto++; }
  std::vector<VkDescriptorPoolSize> sizes;
  if (nTex) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, nTex});
  if (nSto) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nSto});
  if (nUni) sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nUni});
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1;
  dpi.poolSizeCount = (uint32_t)sizes.size();
  dpi.pPoolSizes = sizes.data();
  check(vkCreateDescriptorPool(dev, &dpi, nullptr, &g->pool), "vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsa.descriptorPool = g->pool;
  dsa.descriptorSetCount = 1;
  dsa.pSetLayouts = &bgl->dsl;
  check(vkAllocateDescriptorSets(dev, &dsa, &g->set), "vkAllocateDescriptorSets");
  std::vector<VkDescriptorBufferInfo> bi(entries.size());
  std::vector<VkDescriptorImageInfo> ii(entries.size());
  std::vector<VkWriteDescriptorSet> ws;
  std::string te;
  for (uint32_t i = 0; i < entries.size(); i++) {
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = g->set;
    w.dstBinding = i;
    w.descriptorCount = 1;
    w.descriptorType = descriptorOf(bgl->entries[i]);
    if (bgl->entries[i] == Bind::Tex) {
      if (!entries[i].tex) throw Error("bind group on " + bgl->id + ": binding " + std::to_string(i) + " wants a texture");
      ii[i] = {VK_NULL_HANDLE, entries[i].tex->view, VK_IMAGE_LAYOUT_GENERAL};
      w.pImageInfo = &ii[i];
      if (traceOut) te += std::string(te.empty() ? "" : ",") + "{\"binding\":" + std::to_string(i) + ",\"view\":" + jsonStr(entries[i].tex->id) + "}";
    } else {
      Buffer* b = entries[i].buf;
      if (!b) throw Error("bind group on " + bgl->id + ": binding " + std::to_string(i) + " wants a buffer");
      bi[i] = {b->buf, 0, VK_WHOLE_SIZE};
      w.pBufferInfo = &bi[i];
      if (traceOut) te += std::string(te.empty() ? "" : ",") + "{\"binding\":" + std::to_string(i) + ",\"buf\":" + jsonStr(b->id) + "}";
    }
    ws.push_back(w);
  }
  vkUpdateDescriptorSets(dev, (uint32_t)ws.size(), ws.data(), 0, nullptr);
  if (traceOut) traceLine("{\"op\":\"group\",\"id\":" + jsonStr(g->id) + ",\"layout\":" + jsonStr(bgl->id) + ",\"entries\":[" + te + "]}");
  return g;
}

BindGroup::~BindGroup() { if (dev && pool) vkDestroyDescriptorPool(dev->dev, pool, nullptr); }

std::shared_ptr<QuerySet> Device::createQuerySet(uint32_t count, const std::string& id) {
  auto q = std::make_shared<QuerySet>();
  q->dev = this;
  q->count = count;
  q->id = id.empty() ? autoId('q') : id;
  VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qci.queryCount = count;
  check(vkCreateQueryPool(dev, &qci, nullptr, &q->pool), "vkCreateQueryPool");
  return q;
}

QuerySet::~QuerySet() { if (dev && pool) vkDestroyQueryPool(dev->dev, pool, nullptr); }

void Device::writeBuffer(Buffer& b, uint64_t off, const void* data, uint64_t size) {
  if (off + size > b.size) throw Error("writeBuffer past " + b.id + "'s " + std::to_string(b.size) + " bytes");
  if ((off & 3) || (size & 3)) throw Error("writeBuffer to " + b.id + " at " + std::to_string(off) + ", " + std::to_string(size) + " B: WebGPU's writes are whole u32s");
  if (!size) return;
  Write w{&b, off, std::vector<uint8_t>((const uint8_t*)data, (const uint8_t*)data + size)};
  { std::lock_guard<std::mutex> l(mu); writes.push_back(std::move(w)); }
  if (traceOut) traceLine("{\"op\":\"write\",\"buf\":" + jsonStr(b.id) + ",\"off\":" + std::to_string(off) + ",\"size\":" + std::to_string(size) + "}");
}

void Device::upload(Buffer& b, uint64_t off, const void* data, uint64_t size) {
  if (off + size > b.size) throw Error("upload past " + b.id + "'s " + std::to_string(b.size) + " bytes");
  if (!size) return;
  auto s = createBuffer(size, MAP_WRITE, autoId('s'));
  memcpy(s->mapped, data, (size_t)size);
  if (!s->coherent) {
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    mr.memory = s->mem; mr.size = VK_WHOLE_SIZE;
    vkFlushMappedMemoryRanges(dev, 1, &mr);
  }
  VkBuffer src = s->buf, dst = b.buf;
  {
    std::lock_guard<std::mutex> l(mu);
    inits.push_back({&b, [src, dst, off, size](VkCommandBuffer cb) { VkBufferCopy c{0, off, size}; vkCmdCopyBuffer(cb, src, dst, 1, &c); }, 1});
    staging.push_back(s);
  }
  if (traceOut) traceLine("{\"op\":\"write\",\"buf\":" + jsonStr(b.id) + ",\"off\":" + std::to_string(off) + ",\"size\":" + std::to_string(size) + "}");
}

Encoder Device::encoder() {
  Encoder e;
  e.dev = this;
  // ai: the recording thread: what dropped Tickets left goes back to the pool here, not where they were dropped
  {
    std::lock_guard<std::mutex> l(mu);
    if (!retired.empty()) { vkFreeCommandBuffers(dev, pool, (uint32_t)retired.size(), retired.data()); retired.clear(); }
  }
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  check(vkAllocateCommandBuffers(dev, &cai, &e.cb), "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(e.cb, &bi), "vkBeginCommandBuffer");
  return e;
}

// ai: Everything an encoder records sees what came before it: WebGPU's usage scopes, a dispatch each, and its
// ai: copies' ordering, as one barrier over every kind of access the decoder makes.
static void fullBarrier(VkCommandBuffer cb) {
  // ai: LIZ_BARRIER=all: every stage and every access (a driver experiment, 2026-09-29)
  static const bool all = getenv("LIZ_BARRIER") && std::string(getenv("LIZ_BARRIER")) == "all";
  if (all) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    return;
  }
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  // ai: No host access here: made visible to the host a barrier each (RADV writes its L2 back for it), the batch cost
  // ai: the iGPU 15.7 ms a frame against the web's 7.1 (2026-09-29); submit() makes the batch's writes host-visible once.
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ai: Before the first command too: Vulkan orders nothing across submits on one queue without a barrier, where
// ai: WebGPU's queue orders everything (a slot's upload before the batch that copies it, a batch's copy of a slot
// ai: before the next upload into it). Its first scope is everything submitted before, so batches run one after the
// ai: other on the device, as Dawn runs them; two lanes still hide the readback's round trip.
void Encoder::hazard() {
  commands++;
  fullBarrier(cb);
}

void Encoder::beginComputePass(QuerySet* qs, int begin, int end) {
  if (inPass) throw Error("a pass begun inside a pass");
  inPass = true;
  passQs = qs;
  passEnd = end;
  if (dev->traceOut) trace.push_back(std::string("{\"c\":\"pass\",\"ts\":") + (qs ? "[" + jsonStr(qs->id) + "," + std::to_string(begin) + "," + std::to_string(end) + "]" : "null") + "}");
  if (qs && dev->features.timestamps) {
    if (std::find(reset.begin(), reset.end(), qs) == reset.end()) {
      vkCmdResetQueryPool(cb, qs->pool, 0, qs->count);
      reset.push_back(qs);
    }
    hazard();
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qs->pool, (uint32_t)begin);
  }
}

void Encoder::endPass() {
  if (!inPass) throw Error("end with no pass");
  inPass = false;
  if (dev->traceOut) trace.push_back("{\"c\":\"end\"}");
  if (passQs && dev->features.timestamps) {
    hazard();
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, passQs->pool, (uint32_t)passEnd);
  }
  passQs = nullptr;
}

void Encoder::setPipeline(const Pipeline& p) {
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
  pipe = &p;
  if (dev->traceOut) trace.push_back("{\"c\":\"pipe\",\"id\":" + jsonStr(p.id) + "}");
}

void Encoder::setBindGroup(const BindGroup& g) {
  if (!pipe) throw Error("a bind group before a pipeline");
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout, 0, 1, &g.set, 0, nullptr);
  if (dev->traceOut) trace.push_back("{\"c\":\"group\",\"i\":0,\"id\":" + jsonStr(g.id) + "}");
}

void Encoder::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (dev->traceOut) trace.push_back("{\"c\":\"dispatch\",\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"z\":" + std::to_string(z) + "}");
  if (!x || !y || !z) return;
  hazard();
  vkCmdDispatch(cb, x, y, z);
}

void Encoder::dispatchIndirect(const Buffer& b, uint64_t offset) {
  if (dev->traceOut) trace.push_back("{\"c\":\"indirect\",\"buf\":" + jsonStr(b.id) + ",\"off\":" + std::to_string(offset) + "}");
  hazard();
  vkCmdDispatchIndirect(cb, b.buf, offset);
}

void Encoder::copyBufferToBuffer(const Buffer& src, uint64_t so, const Buffer& dst, uint64_t dof, uint64_t size) {
  if (dev->traceOut) trace.push_back("{\"c\":\"b2b\",\"src\":" + jsonStr(src.id) + ",\"so\":" + std::to_string(so) + ",\"dst\":" + jsonStr(dst.id) + ",\"do\":" + std::to_string(dof) + ",\"size\":" + std::to_string(size) + "}");
  if (!size) return;
  hazard();
  VkBufferCopy c{so, dof, size};
  vkCmdCopyBuffer(cb, src.buf, dst.buf, 1, &c);
}

void Encoder::copyBufferToTexture(const Buffer& src, uint64_t offset, uint32_t stride, const Texture& tex, uint32_t layer, uint32_t w, uint32_t h) {
  if (dev->traceOut) trace.push_back("{\"c\":\"b2t\",\"src\":" + jsonStr(src.id) + ",\"so\":" + std::to_string(offset) + ",\"dst\":" + jsonStr(tex.id) + ",\"layer\":" + std::to_string(layer) + ",\"size\":[" + std::to_string(w) + "," + std::to_string(h) + "]}");
  hazard();
  VkBufferImageCopy c{};
  c.bufferOffset = offset;
  c.bufferRowLength = stride;
  c.bufferImageHeight = 0;
  c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
  c.imageOffset = {0, 0, 0};
  c.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(cb, src.buf, tex.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

void Encoder::copyTextureToTexture(const Texture& src, uint32_t srcLayer, const Texture& dst, uint32_t dstLayer, uint32_t w, uint32_t h) {
  if (dev->traceOut) trace.push_back("{\"c\":\"t2t\",\"src\":" + jsonStr(src.id) + ",\"so\":[0,0," + std::to_string(srcLayer) + "],\"dst\":" + jsonStr(dst.id) + ",\"do\":[0,0," + std::to_string(dstLayer) + "],\"size\":[" + std::to_string(w) + "," + std::to_string(h) + ",1]}");
  hazard();
  VkImageCopy c{};
  c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, srcLayer, 1};
  c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, dstLayer, 1};
  c.extent = {w, h, 1};
  vkCmdCopyImage(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, dst.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

void Encoder::copyTextureToBuffer(const Texture& src, uint32_t layer, uint32_t w, uint32_t h, const Buffer& dst, uint64_t offset) {
  if (dev->traceOut) trace.push_back("{\"c\":\"t2b\",\"src\":" + jsonStr(src.id) + ",\"layer\":" + std::to_string(layer) + ",\"dst\":" + jsonStr(dst.id) + ",\"do\":" + std::to_string(offset) + ",\"size\":[" + std::to_string(w) + "," + std::to_string(h) + "]}");
  hazard();
  VkBufferImageCopy c{};
  c.bufferOffset = offset;
  c.bufferRowLength = w;
  c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
  c.imageExtent = {w, h, 1};
  vkCmdCopyImageToBuffer(cb, src.img, VK_IMAGE_LAYOUT_GENERAL, dst.buf, 1, &c);
}

void Encoder::clearBuffer(const Buffer& b, uint64_t offset, uint64_t size) {
  if (size == ~0ull) size = b.size - offset;
  if (dev->traceOut) trace.push_back("{\"c\":\"clear\",\"buf\":" + jsonStr(b.id) + ",\"off\":" + std::to_string(offset) + ",\"size\":" + std::to_string(size) + "}");
  if (!size) return;
  hazard();
  vkCmdFillBuffer(cb, b.buf, offset, size, 0);
}

void Encoder::resolveQuerySet(const QuerySet& qs, uint32_t first, uint32_t count, const Buffer& dst, uint64_t offset) {
  if (dev->traceOut) trace.push_back("{\"c\":\"resolve\",\"qs\":" + jsonStr(qs.id) + ",\"first\":" + std::to_string(first) + ",\"count\":" + std::to_string(count) + ",\"dst\":" + jsonStr(dst.id) + ",\"off\":" + std::to_string(offset) + "}");
  hazard();
  if (!dev->features.timestamps) { vkCmdFillBuffer(cb, dst.buf, offset, 8ull * count, 0); return; }
  vkCmdCopyQueryPoolResults(cb, qs.pool, first, count, dst.buf, offset, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

// ai: The head: the new objects' fills and clears, then the queue's writes (vkCmdUpdateBuffer takes 64 KB or less a
// ai: call), each behind a barrier; ends behind one, so the commands after it see all of it.
static bool recordHead(Device& d, VkCommandBuffer cb) {
  if (d.inits.empty() && d.writes.empty()) return false;
  for (int phase = 0; phase < 2; phase++) {
    bool any = false;
    for (auto& f : d.inits) if (f.phase == phase) { f.f(cb); any = true; }
    if (any) fullBarrier(cb);
  }
  for (auto& w : d.writes) {
    for (uint64_t o = 0; o < w.data.size(); o += 65536) {
      uint64_t n = std::min<uint64_t>(65536, w.data.size() - o);
      vkCmdUpdateBuffer(cb, w.dst->buf, w.off + o, n, w.data.data() + o);
    }
  }
  fullBarrier(cb);
  d.inits.clear();
  d.writes.clear();
  return true;
}

std::shared_ptr<Ticket> Device::submit(Encoder& enc, VkSemaphore wait, uint64_t waitValue) {
  if (enc.inPass) throw Error("submit with a pass open");
  // ai: Whatever the batch wrote reaches the host's mapped readbacks.
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(enc.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  check(vkEndCommandBuffer(enc.cb), "vkEndCommandBuffer");
  auto t = std::make_shared<Ticket>();
  t->dev = this;
  VkCommandBuffer head = VK_NULL_HANDLE;
  // ai: the pending writes and fills, and the queue, under the device's lock (wg.h "Threads")
  std::lock_guard<std::mutex> lock(mu);
  if (!inits.empty() || !writes.empty()) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev, &cai, &head), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(head, &bi), "vkBeginCommandBuffer");
    recordHead(*this, head);
    check(vkEndCommandBuffer(head), "vkEndCommandBuffer");
  }
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  check(vkCreateFence(dev, &fci, nullptr, &t->fence), "vkCreateFence");
  VkCommandBuffer cbs[2];
  uint32_t n = 0;
  if (head) cbs[n++] = head;
  cbs[n++] = enc.cb;
  t->cbs[0] = enc.cb;
  t->cbs[1] = head;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = n;
  si.pCommandBuffers = cbs;
  VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  if (wait) {
    ts.waitSemaphoreValueCount = 1;
    ts.pWaitSemaphoreValues = &waitValue;
    si.pNext = &ts;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &wait;
    si.pWaitDstStageMask = &stage;
  }
  check(vkQueueSubmit(queue, 1, &si, t->fence), "vkQueueSubmit");
  if (traceOut) {
    std::string c;
    for (auto& s : enc.trace) c += std::string(c.empty() ? "" : ",") + s;
    traceLine("{\"op\":\"submit\",\"cmds\":[" + c + "]}");
  }
  enc.cb = VK_NULL_HANDLE;
  t->keep = std::move(staging);
  staging.clear();
  return t;
}

void Device::flush() {
  { std::lock_guard<std::mutex> l(mu); if (inits.empty() && writes.empty()) return; }
  Encoder e = encoder();
  submit(e)->wait();
}

bool Ticket::done() {
  if (!fence) return true;
  VkResult r = vkGetFenceStatus(dev->dev, fence);
  if (r == VK_NOT_READY) return false;
  check(r, "vkGetFenceStatus");
  return true;
}

void Ticket::wait() {
  if (!fence) return;
  check(vkWaitForFences(dev->dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

Ticket::~Ticket() {
  if (!dev) return;
  if (fence) {
    vkWaitForFences(dev->dev, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(dev->dev, fence, nullptr);
  }
  // ai: freed by the recording thread (Device::encoder), whichever thread drops the ticket
  std::lock_guard<std::mutex> l(dev->mu);
  for (auto cb : cbs) if (cb) dev->retired.push_back(cb);
}

const uint8_t* Device::read(Buffer& b, uint64_t off, uint64_t size) {
  if (!b.mapped) throw Error("read of " + b.id + ", which is not MAP_READ");
  if (!b.coherent) {
    VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    mr.memory = b.mem;
    mr.offset = 0;
    mr.size = VK_WHOLE_SIZE;
    check(vkInvalidateMappedMemoryRanges(dev, 1, &mr), "vkInvalidateMappedMemoryRanges");
  }
  (void)size;
  return b.mapped + off;
}

}  // namespace wg
