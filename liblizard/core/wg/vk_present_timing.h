// ai: VK_KHR_present_id2 and VK_EXT_present_timing as Vulkan-Headers 1.4.367 declares them, for the vendored headers
// ai: (third_party/vulkan-headers, 1.3.275), which predate both. Each block stands only where the headers lack it, so a
// ai: newer vendored copy takes over by itself; the values are the registry's (vk.xml, extensions 480 and 209).
#pragma once
#include "volk.h"

#ifndef VK_KHR_present_id2
#define VK_KHR_present_id2 1
#define VK_KHR_PRESENT_ID_2_EXTENSION_NAME "VK_KHR_present_id2"
#define VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR static_cast<VkStructureType>(1000479000)
#define VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR static_cast<VkStructureType>(1000479001)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR static_cast<VkStructureType>(1000479002)
#define VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR static_cast<VkSwapchainCreateFlagBitsKHR>(0x00000040)
typedef struct VkSurfaceCapabilitiesPresentId2KHR {
  VkStructureType sType;
  void* pNext;
  VkBool32 presentId2Supported;
} VkSurfaceCapabilitiesPresentId2KHR;
typedef struct VkPresentId2KHR {
  VkStructureType sType;
  const void* pNext;
  uint32_t swapchainCount;
  const uint64_t* pPresentIds;
} VkPresentId2KHR;
typedef struct VkPhysicalDevicePresentId2FeaturesKHR {
  VkStructureType sType;
  void* pNext;
  VkBool32 presentId2;
} VkPhysicalDevicePresentId2FeaturesKHR;
#endif

#ifndef VK_EXT_present_timing
#define VK_EXT_present_timing 1
#define VK_EXT_PRESENT_TIMING_EXTENSION_NAME "VK_EXT_present_timing"
#define VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT static_cast<VkResult>(-1000208000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT static_cast<VkStructureType>(1000208000)
#define VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT static_cast<VkStructureType>(1000208001)
#define VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT static_cast<VkStructureType>(1000208002)
#define VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT static_cast<VkStructureType>(1000208003)
#define VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT static_cast<VkStructureType>(1000208004)
#define VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT static_cast<VkStructureType>(1000208005)
#define VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT static_cast<VkStructureType>(1000208006)
#define VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT static_cast<VkStructureType>(1000208007)
#define VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT static_cast<VkStructureType>(1000208008)
#define VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT static_cast<VkStructureType>(1000208009)
#define VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT static_cast<VkSwapchainCreateFlagBitsKHR>(0x00000200)
#define VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT static_cast<VkTimeDomainKHR>(1000208000)
#define VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT static_cast<VkTimeDomainKHR>(1000208001)
typedef VkFlags VkPresentStageFlagsEXT;
typedef VkFlags VkPastPresentationTimingFlagsEXT;
typedef VkFlags VkPresentTimingInfoFlagsEXT;
enum : VkFlags {
  VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT = 0x1,
  VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT = 0x2,
  VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT = 0x4,
  VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT = 0x8,
  VK_PAST_PRESENTATION_TIMING_ALLOW_PARTIAL_RESULTS_BIT_EXT = 0x1,
  VK_PAST_PRESENTATION_TIMING_ALLOW_OUT_OF_ORDER_RESULTS_BIT_EXT = 0x2,
  VK_PRESENT_TIMING_INFO_PRESENT_AT_RELATIVE_TIME_BIT_EXT = 0x1,
  VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT = 0x2,
};
typedef struct VkPhysicalDevicePresentTimingFeaturesEXT {
  VkStructureType sType;
  void* pNext;
  VkBool32 presentTiming;
  VkBool32 presentAtAbsoluteTime;
  VkBool32 presentAtRelativeTime;
} VkPhysicalDevicePresentTimingFeaturesEXT;
typedef struct VkPresentTimingSurfaceCapabilitiesEXT {
  VkStructureType sType;
  void* pNext;
  VkBool32 presentTimingSupported;
  VkBool32 presentAtAbsoluteTimeSupported;
  VkBool32 presentAtRelativeTimeSupported;
  VkPresentStageFlagsEXT presentStageQueries;
} VkPresentTimingSurfaceCapabilitiesEXT;
typedef struct VkSwapchainCalibratedTimestampInfoEXT {
  VkStructureType sType;
  const void* pNext;
  VkSwapchainKHR swapchain;
  VkPresentStageFlagsEXT presentStage;
  uint64_t timeDomainId;
} VkSwapchainCalibratedTimestampInfoEXT;
typedef struct VkSwapchainTimingPropertiesEXT {
  VkStructureType sType;
  void* pNext;
  uint64_t refreshDuration;
  uint64_t refreshInterval;
} VkSwapchainTimingPropertiesEXT;
typedef struct VkSwapchainTimeDomainPropertiesEXT {
  VkStructureType sType;
  void* pNext;
  uint32_t timeDomainCount;
  VkTimeDomainKHR* pTimeDomains;
  uint64_t* pTimeDomainIds;
} VkSwapchainTimeDomainPropertiesEXT;
typedef struct VkPastPresentationTimingInfoEXT {
  VkStructureType sType;
  const void* pNext;
  VkPastPresentationTimingFlagsEXT flags;
  VkSwapchainKHR swapchain;
} VkPastPresentationTimingInfoEXT;
typedef struct VkPresentStageTimeEXT {
  VkPresentStageFlagsEXT stage;
  uint64_t time;
} VkPresentStageTimeEXT;
typedef struct VkPastPresentationTimingEXT {
  VkStructureType sType;
  void* pNext;
  uint64_t presentId;
  uint64_t targetTime;
  uint32_t presentStageCount;
  VkPresentStageTimeEXT* pPresentStages;
  VkTimeDomainKHR timeDomain;
  uint64_t timeDomainId;
  VkBool32 reportComplete;
} VkPastPresentationTimingEXT;
typedef struct VkPastPresentationTimingPropertiesEXT {
  VkStructureType sType;
  void* pNext;
  uint64_t timingPropertiesCounter;
  uint64_t timeDomainsCounter;
  uint32_t presentationTimingCount;
  VkPastPresentationTimingEXT* pPresentationTimings;
} VkPastPresentationTimingPropertiesEXT;
typedef struct VkPresentTimingInfoEXT {
  VkStructureType sType;
  const void* pNext;
  VkPresentTimingInfoFlagsEXT flags;
  uint64_t targetTime;
  uint64_t timeDomainId;
  VkPresentStageFlagsEXT presentStageQueries;
  VkPresentStageFlagsEXT targetTimeDomainPresentStage;
} VkPresentTimingInfoEXT;
typedef struct VkPresentTimingsInfoEXT {
  VkStructureType sType;
  const void* pNext;
  uint32_t swapchainCount;
  const VkPresentTimingInfoEXT* pTimingInfos;
} VkPresentTimingsInfoEXT;
typedef VkResult(VKAPI_PTR* PFN_vkSetSwapchainPresentTimingQueueSizeEXT)(VkDevice, VkSwapchainKHR, uint32_t);
typedef VkResult(VKAPI_PTR* PFN_vkGetSwapchainTimingPropertiesEXT)(VkDevice, VkSwapchainKHR, VkSwapchainTimingPropertiesEXT*, uint64_t*);
typedef VkResult(VKAPI_PTR* PFN_vkGetSwapchainTimeDomainPropertiesEXT)(VkDevice, VkSwapchainKHR, VkSwapchainTimeDomainPropertiesEXT*, uint64_t*);
typedef VkResult(VKAPI_PTR* PFN_vkGetPastPresentationTimingEXT)(VkDevice, const VkPastPresentationTimingInfoEXT*, VkPastPresentationTimingPropertiesEXT*);
#endif
