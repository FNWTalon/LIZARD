#include "setup.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>

namespace lizard {

ReadFile readDir(const std::string& root) {
  return [root](const std::string& path) {
    std::ifstream f(root + "/" + path, std::ios::binary);
    if (!f) throw wg::Error("no " + root + "/" + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  };
}

std::vector<std::string> variantsFor(const wg::Device& d, bool subgroups) {
  std::vector<std::string> out;
  const auto& F = d.features;
  const auto& L = d.limits;
  // ai: the SPIR-V zeroes workgroup memory by initializer (gen/wgsl2spv), which needs shaderZeroInitializeWorkgroupMemory
  if (L.maxComputeSharedMemorySize < 32768 || L.maxPerStageStorageBuffers < 10 || !F.zeroInit) return out;
  const bool sg = subgroups && F.subgroupShuffle;
  auto add = [&](const std::string& p) { if (sg) out.push_back(p + "-sg"); out.push_back(p); };
  const bool halfOk = F.f16;
  if (halfOk && F.int8 && F.dot) add("int8");
  if (halfOk) add("f16");
  add("f32");
  return out;
}

std::vector<std::string> variantsIn(const std::string& root, const std::vector<std::string>& variants) {
  std::vector<std::string> out;
  for (const auto& v : variants) if (std::ifstream(root + "/setup/" + v + ".json")) out.push_back(v);
  return out;
}

std::unique_ptr<Setup> Setup::load(wg::Device& d, ReadFile read, const std::string& variant) {
  auto s = std::make_unique<Setup>();
  s->dev = &d;
  s->read = read;
  s->variant = variant;
  auto bytes = read("setup/" + variant + ".json");
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
      compileMs[id] = ms;
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

// ai: out/nets/manifest.json: { "kernels": [ { "variants": [names], "role": a path into the tree to a { p, bgl } pair
// ai: (e.g. "front.fcn2", "front.twins.small.pipe"), "weights": the tree path to the buffer id it replaces or null,
// ai: "spv": its SPIR-V under out/, "spec": [u32...], "P": a classifier's patches a workgroup (Pipe::P; 1 when absent),
// ai: "blob": its weights' bytes under out/ or null, "needs": [ { M, N, K, A, B, C, R } ] the cooperative-matrix shapes
// ai: (wg::Features::CoopShape's types) } ] }.
static json* at(json& tree, const std::string& path) {
  json* j = &tree;
  size_t a = 0;
  while (a <= path.size()) {
    size_t b = path.find('.', a);
    if (b == std::string::npos) b = path.size();
    const std::string k = path.substr(a, b - a);
    if (!j->is_object() || !j->contains(k)) return nullptr;
    j = &(*j)[k];
    a = b + 1;
  }
  return j;
}

std::vector<std::string> Setup::applyNativeNets() {
  std::vector<std::string> done;
  const char* mode = getenv("LIZ_NATIVE_NETS");
  if (mode && std::string(mode) == "0") return done;
  const bool replace = mode && std::string(mode) == "1";
  // ai: the roles the host has a check for (FrontHalf::measureTwins); any other is built only to replace
  static const std::vector<std::string> checked = {"front.twins.small.pipe", "front.twins.large.pipe"};
  std::vector<uint8_t> mb;
  try { mb = read("nets/manifest.json"); } catch (const std::exception&) { return done; }
  const json man = json::parse(mb.begin(), mb.end());
  const auto& shapes = dev->features.coopShapes;
  for (const auto& k : man["kernels"]) {
    bool mine = false;
    for (const auto& v : k["variants"]) mine = mine || v.get<std::string>() == variant;
    if (!mine) continue;
    // ai: a kernel that names no matrix shape runs on any device
    bool ok = k["needs"].empty() || dev->features.coopmat;
    for (const auto& n : k["needs"]) {
      bool has = false;
      for (const auto& c : shapes)
        has = has || (c.subgroup && c.M == n["M"].get<uint32_t>() && c.N == n["N"].get<uint32_t>() && c.K == n["K"].get<uint32_t>() && c.A == n["A"].get<int>() && c.B == n["B"].get<int>() && c.C == n["C"].get<int>() && c.R == n["R"].get<int>());
      ok = ok && has;
    }
    const std::string role = k["role"];
    if (!ok) { dev->log("native net " + role + ": not on this device (its cooperative-matrix shapes)"); continue; }
    if (!replace && std::find(checked.begin(), checked.end(), role) == checked.end()) { dev->log("native net " + role + ": not built (the host has no check for it)"); continue; }
    json* pair = at(tree, role);
    if (!pair || !pair->is_object() || !(*pair)["p"].is_string()) throw wg::Error("native net: no pipeline at " + role);
    const std::string pid = (*pair)["p"], lid = (*pair)["bgl"];
    make({lid});
    auto b = read(k["spv"].get<std::string>());
    std::vector<uint32_t> w(b.size() / 4);
    memcpy(w.data(), b.data(), b.size());
    std::vector<uint32_t> spec;
    for (const auto& x : k.value("spec", json::array())) spec.push_back(x.get<uint32_t>());
    NativeNet nn;
    nn.pipe = {dev->createPipeline(w, bgls.at(lid), pid + "@native", "native " + role, spec), bgls.at(lid), k.value("P", 1u)};
    std::string bufId;
    if (k["weights"].is_string()) {
      const json* wid = at(tree, k["weights"].get<std::string>());
      if (!wid || !wid->is_string()) throw wg::Error("native net: no buffer at " + k["weights"].get<std::string>());
      bufId = wid->get<std::string>();
      auto data = read(k["blob"].get<std::string>());
      nn.weights = dev->createBuffer(data.size(), wg::STORAGE | wg::COPY_DST, bufId + "@native", false);
      dev->upload(*nn.weights, 0, data.data(), data.size());
    }
    if (replace) {
      pipelines[pid] = nn.pipe.p;
      (*pair)["P"] = nn.pipe.P;
      if (nn.weights) buffers[bufId] = nn.weights;
    } else natives[role] = nn;
    done.push_back(role + (replace ? " (in the web kernel's place, unchecked)" : ""));
  }
  dev->flush();
  return done;
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
