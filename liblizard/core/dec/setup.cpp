#include "setup.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace lizard {

// ai: The tree's files by their UTF-8 paths (fs::u8path): Windows reads a narrow path in the ANSI code page, so a tree
// ai: under a folder named outside it (the app's, under %LOCALAPPDATA%) held no variant there (2026-10-25).
ReadFile readDir(const std::string& root) {
  return [root](const std::string& path) {
    std::ifstream f(std::filesystem::u8path(root + "/" + path), std::ios::binary);
    if (!f) throw wg::Error("no " + root + "/" + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  };
}

std::vector<std::string> variantsFor(const wg::Device& d) {
  std::vector<std::string> out;
  const auto& F = d.features;
  const auto& L = d.limits;
  // ai: the SPIR-V zeroes workgroup memory by initializer (gen/wgsl2spv), which needs shaderZeroInitializeWorkgroupMemory
  if (L.maxComputeSharedMemorySize < 32768 || L.maxPerStageStorageBuffers < 10 || !F.zeroInit) return out;
  const bool sg = F.subgroupShuffle;
  auto add = [&](const std::string& p) { if (sg) out.push_back(p + "-sg"); out.push_back(p); };
  const bool halfOk = F.f16;
  if (halfOk && F.int8 && F.dot) add("int8");
  if (halfOk) add("f16");
  add("f32");
  return out;
}

// ai: A variant's setup: setup/<variant>.json.
std::string setupName(const std::string& variant) { return "setup/" + variant + ".json"; }

std::vector<std::string> variantsIn(const std::string& root, const std::vector<std::string>& variants) {
  std::vector<std::string> out;
  for (const auto& v : variants) if (std::ifstream(std::filesystem::u8path(root + "/" + setupName(v)))) out.push_back(v);
  return out;
}

std::unique_ptr<Setup> Setup::load(wg::Device& d, ReadFile read, const std::string& variant) {
  auto s = std::make_unique<Setup>();
  s->dev = &d;
  s->read = read;
  s->variant = variant;
  auto bytes = read(setupName(variant));
  s->tree = json::parse(bytes.begin(), bytes.end());
  s->objects = std::move(s->tree["objects"]);
  s->tree.erase("objects");
  return s;
}

std::vector<uint32_t> Setup::spirv(const std::string& module) {
  static const std::string dir = getenv("LIZ_SPV") ? getenv("LIZ_SPV") : "spv";
  auto b = read(dir + "/" + module + ".spv");
  if (b.size() % 4 || b.size() < 20) throw wg::Error("spv/" + module + ".spv: not SPIR-V");
  std::vector<uint32_t> w(b.size() / 4);
  memcpy(w.data(), b.data(), b.size());
  return w;
}

static wg::Bind bindOf(const std::string& t) {
  if (t == "tex") return wg::Bind::Tex;
  if (t == "ro") return wg::Bind::RO;
  if (t == "rw") return wg::Bind::RW;
  if (t == "uniform") return wg::Bind::Uniform;
  throw wg::Error("bind group layout entry " + t);
}

// ai: The objects named, those a pipeline names (its layout) first; each made once.
void Setup::make(const std::vector<std::string>& ids) {
  for (const auto& id : ids) {
    if (id.empty()) continue;
    const char k = id[0];
    if (k == 'L' && !bgls.count(id)) {
      std::vector<wg::Bind> e;
      for (auto& t : objects["bgls"].at(id)) e.push_back(bindOf(t.get<std::string>()));
      bgls[id] = dev->createBindGroupLayout(e, id);
    } else if (k == 'p' && !pipelines.count(id)) {
      const auto& o = objects["pipelines"].at(id);
      if (o["groups"].size() != 1) throw wg::Error(id + ": " + std::to_string(o["groups"].size()) + " bind groups; the decoder uses one");
      const std::string g = o["groups"][0];
      make({g});
      const std::string label = o.value("label", json()).is_string() ? o["label"].get<std::string>() : "";
      auto t0 = std::chrono::steady_clock::now();
      pipelines[id] = dev->createPipeline(spirv(o["module"]), bgls.at(g), id, label);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (ms > 200) dev->log("pipeline " + id + " (" + (label.empty() ? o["module"].get<std::string>() : label) + ", module " + o["module"].get<std::string>() + "): " + std::to_string((int)ms) + " ms to compile");
    } else if (k == 'b' && !buffers.count(id)) {
      const auto& o = objects["buffers"].at(id);
      const bool blob = o["blob"].is_string();
      auto b = dev->createBuffer(o["size"].get<uint64_t>(), o["usage"].get<uint32_t>(), id, !blob);
      if (blob) {
        auto data = read("blobs/" + o["blob"].get<std::string>() + ".bin");
        if (data.size() != b->size) throw wg::Error(id + ": blob of " + std::to_string(data.size()) + " B for a buffer of " + std::to_string(b->size));
        dev->upload(*b, 0, data.data(), data.size());
      }
      buffers[id] = b;
    }
  }
}

// ai: Every id a part of the tree names (strings like b12, L3, p40).
static void idsIn(const json& j, std::vector<std::string>& out) {
  if (j.is_string()) {
    const auto& s = j.get_ref<const std::string&>();
    if (s.size() > 1 && (s[0] == 'b' || s[0] == 'L' || s[0] == 'p') && std::all_of(s.begin() + 1, s.end(), ::isdigit)) out.push_back(s);
  } else if (j.is_array() || j.is_object()) {
    for (const auto& v : j) idsIn(v, out);
  }
}

void Setup::build(int B) {
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::string> ids;
  idsIn(tree["front"], ids);
  // ai: layouts before pipelines before buffers, so a pipeline's layout is there and its compile starts early
  std::stable_sort(ids.begin(), ids.end(), [](const std::string& a, const std::string& b) { auto r = [](char c) { return c == 'L' ? 0 : c == 'p' ? 1 : 2; }; return r(a[0]) < r(b[0]); });
  make(ids);
  buildBack(B);
  buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Setup::buildBack(int B) {
  if (!tree["backs"].contains(std::to_string(B))) throw wg::Error("no back half at B = " + std::to_string(B) + " in the setup");
  // ai: The previous back half's objects go (the front half's stay): the web's destroyBack.
  if (backB && backB != B) {
    std::vector<std::string> old;
    idsIn(back(), old);
    for (const auto& id : old) { buffers.erase(id); pipelines.erase(id); }
  }
  backB = B;
  std::vector<std::string> ids;
  idsIn(back(), ids);
  make(ids);
  dev->flush();
}

wg::Buffer& Setup::buf(const std::string& id) { return *bufPtr(id); }
std::shared_ptr<wg::Buffer> Setup::bufPtr(const std::string& id) {
  auto it = buffers.find(id);
  if (it == buffers.end()) throw wg::Error("setup buffer " + id + " not built");
  return it->second;
}
std::shared_ptr<wg::BindGroupLayout> Setup::bgl(const std::string& id) {
  auto it = bgls.find(id);
  if (it == bgls.end()) throw wg::Error("setup layout " + id + " not built");
  return it->second;
}
std::shared_ptr<wg::Pipeline> Setup::pipe(const std::string& id) {
  auto it = pipelines.find(id);
  if (it == pipelines.end()) throw wg::Error("setup pipeline " + id + " not built");
  return it->second;
}
Pipe Setup::pair(const json& pb) {
  if (pb.is_null()) return {};
  return {pipe(pb["p"]), bgl(pb["bgl"]), pb.value("P", 1u)};
}

}  // namespace lizard
