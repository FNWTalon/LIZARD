// ai: The decoder's setup, as liblizard/gen/gen.mjs wrote it from the web host: one
// ai: variant's tree (setup/<variant>.json), its buffers' bytes (blobs/<hash>.bin) and its pipelines' SPIR-V
// ai: (spv/<module>.spv), built on a wg device. Only the front half and the one back half in use (a batch size) are
// ai: built; every object keeps the id the web host's recorder gave it, so a trace names the same objects.
#pragma once
#include "wg.h"
#include "json.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lizard {

using json = nlohmann::json;

// ai: A file's bytes by its path under the generated tree (a directory, or the app's assets).
using ReadFile = std::function<std::vector<uint8_t>(const std::string&)>;
ReadFile readDir(const std::string& root);

// ai: A pipeline and its one bind group layout, as the web host's { p, bgl } pairs.
struct Pipe {
  std::shared_ptr<wg::Pipeline> p;
  std::shared_ptr<wg::BindGroupLayout> bgl;
  // ai: a classifier's patches a workgroup (its dispatch is ceil(slots / P)); 1 on every other pipeline
  uint32_t P = 1;
  explicit operator bool() const { return !!p; }
};

// ai: The variants in the web's precision chain order, and which one a device can run: int8 needs f16, int8 and the
// ai: integer dot product (the int8 proposer); f16 needs f16 with 16-bit storage; -sg needs subgroup shuffles in
// ai: compute. Empty when the device misses the floors every variant needs (32 KB of shared memory, 10 storage
// ai: buffers a stage): the C decodes there.
std::vector<std::string> variantsFor(const wg::Device& d, bool subgroups = true);
// ai: Of variants, in their order, those a generated tree holds (setup/<variant>.json; an install carries a variant
// ai: whole, its modules and blobs with it, cmake/gpu_files.cmake): an install may hold only some of the six
// ai: (LIZ_GPU_VARIANTS, the AAR's lizardVariants), and the receiver chooses among those.
std::vector<std::string> variantsIn(const std::string& root, const std::vector<std::string>& variants);

struct Setup {
  json tree;       // ai: the variant's tree less its objects
  json objects;    // ai: buffers, bgls, pipelines by id
  wg::Device* dev = nullptr;
  ReadFile read;
  std::map<std::string, std::shared_ptr<wg::Buffer>> buffers;
  std::map<std::string, std::shared_ptr<wg::BindGroupLayout>> bgls;
  std::map<std::string, std::shared_ptr<wg::Pipeline>> pipelines;
  std::string variant;
  double buildMs = 0;
  std::map<std::string, double> compileMs;   // ai: each pipeline's vkCreateComputePipelines, ms

  static std::unique_ptr<Setup> load(wg::Device& d, ReadFile read, const std::string& variant);
  // ai: The front half's objects and the back half at batch size B, built (a back half built before is dropped).
  void build(int B);
  void buildBack(int B);
  // ai: The rate profile whose stage a back half builds (gpu/back/tiers.mjs; the back's "tiers" by this key, the
  // ai: lab's rate-by-ring arm, 2026-10-07): "" none, and no profile's objects are made.
  std::string tiersKey;
  const json& back() const { return tree["backs"][std::to_string(backB)]; }
  int backB = 0;

  wg::Buffer& buf(const std::string& id);
  std::shared_ptr<wg::Buffer> bufPtr(const std::string& id);
  std::shared_ptr<wg::BindGroupLayout> bgl(const std::string& id);
  std::shared_ptr<wg::Pipeline> pipe(const std::string& id);
  Pipe pair(const json& pb);   // ai: a tree's { p, bgl } (null gives an empty Pipe)
  void makeOne(const std::string& id) { make({id}); }
  // ai: The native nets (core/nets/, out/nets/manifest.json): a kernel on the device's cooperative-matrix units that
  // ai: stands for one of the web's pipelines over the same bind group layout, with its own weights packing, built
  // ai: where the device offers every matrix shape it needs. By role (the pipeline's path in the tree:
  // ai: front.twins.small.pipe). The host adopts one only after checking it on the device (FrontHalf::measureTwins:
  // ai: the same outputs as the web's int8 kernel on the first batch, bit for bit, and faster), since these kernels
  // ai: lean on a driver's compiler that has miscompiled them before (core/nets/README.md).
  struct NativeNet { Pipe pipe; std::shared_ptr<wg::Buffer> weights; };
  std::map<std::string, NativeNet> natives;
  // ai: Builds them; the roles built are listed. LIZ_NATIVE_NETS=0: none. LIZ_NATIVE_NETS=1 (a check's): each takes
  // ai: its web pipeline's and weights' place in the setup outright, unchecked, and `natives` stays empty.
  std::vector<std::string> applyNativeNets();

 private:
  void make(const std::vector<std::string>& ids);
  std::vector<uint32_t> spirv(const std::string& module);
};

}  // namespace lizard
