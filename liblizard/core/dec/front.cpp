// ai: FrontHalf and Batcher (front.h), ported from liblizard/gpu/decoder.mjs; the back half's lane and dispatches from
// ai: liblizard/gpu/back/back.mjs, transform.mjs, soft.mjs and ldpc.mjs; the proposer's from liblizard/gpu/bank_fcn2.mjs.
// ai: Each function names the JS it ports; where the port differs, an ai: comment says so and why.
#include "front.h"
#include "camera.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace lizard {

static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static constexpr double MiB = 1 << 20;

Consts Consts::of(const json& c) {
  Consts k;
  k.LEVELS = c["LEVELS"]; k.COUNTERS = c["COUNTERS"]; k.RESULT = c["RESULT"]; k.SEL_WORDS = c["SEL_WORDS"]; k.TS_MAX = c["TS_MAX"]; k.TS_PLAIN = c["TS_PLAIN"];
  k.TS_INGEST = c["TS_INGEST"]; k.RING_SLACK = c["RING_SLACK"]; k.K = c["K"]; k.CELL = c["CELL"]; k.HYP = c["HYP"]; k.RING_COUNT = c["RING_COUNT"];
  k.NODES_MAX = c["NODES_MAX"]; k.REFIT_DBG = c["REFIT_DBG"]; k.HIST_BINS = c["HIST_BINS"]; k.PYRAMID_BLOCK = c["PYRAMID_BLOCK"];
  k.TAU = c["TAU"]; k.EPS = c["EPS"]; k.TAU_TRACK = c["TAU_TRACK"];
  k.TILE = c["TILE"]; k.KEEP = c["KEEP"]; k.B_CEIL = c["B_CEIL"]; k.CAP_MS = c["CAP_MS"];
  k.TWIN_FRAMES = c["TWIN_FRAMES"]; k.TWIN_ROUNDS = c["TWIN_ROUNDS"]; k.VERSION_MAX = c["VERSION_MAX"]; k.SLOTS = c["SLOTS"];
  k.ARGS_WORDS = c["ARGS_WORDS"]; k.argsLdpc = c["argsLdpc"]; k.argsWords = c["argsWords"]; k.COUNTS = c["COUNTS"]; k.REC_BYTES = c["REC_BYTES"]; k.PAYLOAD = c["PAYLOAD"];
  for (auto& [n, v] : c["ARGS_SLOT"].items()) k.ARGS_SLOT[n] = v;
  return k;
}

std::vector<wg::Resource> Lane::lvEntries() const {
  std::vector<wg::Resource> e{tex.get()};
  for (auto& l : level) e.push_back(l.buf);
  e.push_back(framesBuf);
  e.push_back(ldUni);
  return e;
}

// ai: decoder.mjs FrontHalf.create, the part a native caller does: the setup's objects built (its create() ran in
// ai: gen.mjs), the nets taken as the tree names them, the back half at B_CEIL.
// ai: A rate profile's key (gpu/back/tiers.mjs parseTiers): "7/8:24,3/4:15,1/2:12" or "6:24,4:15,2:12" to
// ai: "6:24,4:15,2:12", the tree's name for its stage. What the profile may be is gen's to check (the tree holds valid
// ai: ones only); a text no key reads throws.
static std::string tiersKeyOf(const std::string& text) {
  static const char* names[] = {"1/4", "1/3", "1/2", "2/3", "3/4", "5/6", "7/8"};
  std::string key, tok;
  auto flush = [&]() {
    if (tok.empty()) return;
    const size_t c = tok.find(':');
    const std::string r = c == std::string::npos ? "" : tok.substr(0, c), n = c == std::string::npos ? "" : tok.substr(c + 1);
    int rate = -1;
    for (int i = 0; i < 7; i++) if (r == names[i]) rate = i;
    if (rate < 0 && r.size() == 1 && r[0] >= '0' && r[0] <= '6') rate = r[0] - '0';
    if (rate < 0 || n.empty() || n.size() > 4 || !std::all_of(n.begin(), n.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) throw wg::Error("LIZ_TIERS: a tier is <rate>:<blocks>, not " + tok);
    key += (key.empty() ? "" : ",") + std::to_string(rate) + ":" + std::to_string(std::stoi(n));
    tok.clear();
  };
  for (char ch : text) { if (ch == ',' || ch == ';' || ch == ' ' || ch == '\t') flush(); else tok += ch; }
  flush();
  return key;
}

std::unique_ptr<FrontHalf> FrontHalf::create(wg::Device& d, ReadFile read, const std::string& variant, double budget, int restart, std::function<void(const std::string&)> log) {
  std::unique_ptr<FrontHalf> fh(new FrontHalf(d));
  fh->log = log ? log : [](const std::string&) {};
  fh->variant = variant;
  const double t0 = fh->now();
  fh->S = Setup::load(d, read, variant);
  const json& T = fh->S->tree;
  fh->C = Consts::of(T["consts"]);
  fh->precision = T["precision"];
  fh->floatPrecision = T["floatPrecision"];
  fh->nets = T["nets"];
  fh->cap = T["cap"];
  fh->reg = T["register"];
  fh->nodesMax = T["nodesMax"];
  fh->bankKeep = T["bankKeep"];
  for (int i = 0; i < 4; i++) fh->modules[i] = T["modules"][i];
  fh->cascade = T["cascade"];
  // ai: LIZ_TIERS (the lab's rate-by-ring arm, 2026-10-07): the rate profile whose stage the back half runs, one the
  // ai: setup carries (liblizard/gen/gen.mjs GEN_TIERS); its objects alone are built
  if (const char* tv = getenv("LIZ_TIERS"); tv && *tv) fh->S->tiersKey = fh->tiersKey = tiersKeyOf(tv);
  fh->S->build(fh->C.B_CEIL);
  if (!fh->tiersKey.empty()) {
    const json& tiers = fh->S->back().value("tiers", json::object());
    if (!tiers.contains(fh->tiersKey)) {
      std::string have;
      for (auto it = tiers.begin(); it != tiers.end(); ++it) have += (have.empty() ? "" : "; ") + it.value()["label"].get<std::string>();
      throw wg::Error("LIZ_TIERS " + std::string(getenv("LIZ_TIERS")) + ": the setup carries no GPU stage for it (" + (have.empty() ? "none: gen.mjs before 2026-10-07" : have) + ")");
    }
    const json& st = tiers[fh->tiersKey];
    fh->tslot = st["slot"]; fh->tversion = st["version"]; fh->tblocks = st["blocks"]; fh->tiersLabel = st["label"];
    fh->log("tiers: " + fh->tiersLabel + ", " + std::to_string(fh->tblocks) + " blocks on LIZARD-" + std::to_string(8 * fh->tversion) + " at picture slot " + std::to_string(fh->tslot) + " (LIZ_TIERS); that slot decodes no other version");
  }
  for (auto& r : fh->S->applyNativeNets()) fh->log("native net built: " + r);
  fh->backShared = 0;
  fh->classify = fh->P("classify");
  fh->weightsBuf = fh->FBP("weightsBuf");
  if (!fh->cascade.is_null()) { fh->classify0 = fh->P("classify0"); fh->weightsBuf0 = fh->FBP("weightsBuf0"); }
  fh->twinForms = T["front"]["twins"];
  // ai: LIZ_TWINS=int8: the classifiers' int8 twins from the start, unmeasured (the web's twins "int8"): a check of
  // ai: the twins' own outputs (a native net's against the web's) reads them on the first batch. LIZ_TWINS=small (or
  // ai: large): that net's twin alone, the other net in its float form (what the S26 keeps: its large twin is 26
  // ai: times its f16 form), so a timing run is in its steady state from the first batch.
  const std::string tw = getenv("LIZ_TWINS") ? getenv("LIZ_TWINS") : "";
  if ((tw == "int8" || tw == "small" || tw == "large") && !fh->twinForms.is_null()) {
    const json& t = fh->twinForms;
    if (tw != "large" && !t["small"].is_null()) { fh->classify0 = fh->S->pair(t["small"]["pipe"]); fh->weightsBuf0 = fh->S->bufPtr(t["small"]["buf"]); fh->cascade["precision"] = "int8"; fh->nets["small"] = "int8"; }
    if (tw != "small" && !t["large"].is_null()) { fh->classify = fh->S->pair(t["large"]["pipe"]); fh->weightsBuf = fh->S->bufPtr(t["large"]["buf"]); fh->S->tree["weightsPrecision"] = "int8"; fh->nets["large"] = "int8"; }
    fh->twinForms = nullptr;
  }
  fh->bt.fh = fh.get();
  fh->bt.budget = budget;
  fh->bt.capMs = fh->C.CAP_MS;
  fh->bt.restart = restart;
  fh->log("setup " + variant + " built in " + std::to_string((int)(fh->now() - t0)) + " ms: nets bank " + fh->nets["bank"].dump() + ", small " + fh->nets["small"].dump() + ", large " + fh->nets["large"].dump() +
          (fh->twinForms.is_null() ? "" : " (the classifiers' int8 twins measured on the first batch)"));
  return fh;
}

// ai: The back half's static bytes (the web's backShared: the buffers BackHalf.build made).
static uint64_t backBytesOf(Setup& S) {
  uint64_t n = 0;
  std::function<void(const json&)> walk = [&](const json& j) {
    if (j.is_string()) { const std::string& s = j.get_ref<const std::string&>(); if (s.size() > 1 && s[0] == 'b' && S.buffers.count(s)) n += S.buffers[s]->size; }
    else if (j.is_array() || j.is_object()) for (const auto& v : j) walk(v);
  };
  walk(S.back());
  return n;
}

void FrontHalf::buildBack(int Bb) {
  S->buildBack(Bb);
  backShared = backBytesOf(*S);
}

// ai: decoder.mjs lane(B): the buffers a batch writes, sized for B frames, and the groups that do not depend on the
// ai: frame's size.
std::unique_ptr<Lane> FrontHalf::lane(int Bl) {
  auto ln = std::make_unique<Lane>();
  const uint64_t a0 = dev.allocated;
  const uint64_t B = Bl, cap = this->cap;
  ln->B = Bl;
  ln->slotsG = std::max(Bl, S->backB);
  using namespace wg;
  auto mk = [&](uint64_t size, uint32_t usage, const char* name) { return dev.createBuffer(size, usage, dev.traceOut ? std::string("n") + name + std::to_string(dev.nextId++) : ""); };
  ln->peaksBuf = mk(16 * cap * B, STORAGE | COPY_SRC | COPY_DST, "peaksBuf");
  ln->selUni = mk(16, UNIFORM | COPY_DST, "selUni");
  ln->framesBuf = mk(16ull * ln->slotsG, STORAGE | COPY_DST, "framesBuf");
  ln->mapsBuf = mk(64 * B, STORAGE | COPY_DST | COPY_SRC, "mapsBuf");
  ln->countsBuf = mk(4ull * C.COUNTERS * B, STORAGE | COPY_DST | COPY_SRC, "countsBuf");
  ln->readingsBuf = mk(48 * cap * B, STORAGE | COPY_DST | COPY_SRC, "readingsBuf");
  ln->candBuf = mk(16ull * C.K * B, STORAGE | COPY_DST | COPY_SRC, "candBuf");
  ln->quadsBuf = mk(16ull * 5 * C.K * B, STORAGE | COPY_DST | COPY_SRC, "quadsBuf");
  ln->scoreBuf = mk(4ull * C.HYP * B, STORAGE | COPY_DST | COPY_SRC, "scoreBuf");
  ln->selBuf = mk(4ull * C.SEL_WORDS * ln->slotsG, STORAGE | COPY_DST, "selBuf");
  ln->resultBuf = mk(4ull * C.RESULT * B, STORAGE | COPY_DST | COPY_SRC, "resultBuf");
  ln->heldUni = mk(16, UNIFORM | COPY_DST, "heldUni");
  ln->p5Uni = mk(32, UNIFORM | COPY_DST, "p5Uni");
  ln->ldUni = mk(80, UNIFORM | COPY_DST, "ldUni");
  ln->measBuf = mk(16ull * C.NODES_MAX * B, STORAGE | COPY_DST | COPY_SRC, "measBuf");
  ln->residBuf = mk(16ull * C.NODES_MAX * B, STORAGE | COPY_DST | COPY_SRC, "residBuf");
  ln->refitDbg = mk(4ull * C.REFIT_DBG * B, STORAGE | COPY_SRC | COPY_DST, "refitDbg");
  ln->histBuf = mk(4ull * C.HIST_BINS * B, STORAGE | COPY_DST | COPY_SRC, "histBuf");
  if (!cascade.is_null()) {
    ln->logitBuf = mk(4 * cap * B, STORAGE, "logitBuf");
    ln->listBuf = mk(4ull * cascade["keep"].get<uint32_t>() * B, STORAGE, "listBuf");
  }
  auto refit = P("refit"), pick = P("pick");
  ln->refitGroup = dev.createBindGroup(refit.bgl, {ln->framesBuf, FBP("nodesBuf"), ln->selBuf, ln->measBuf, ln->mapsBuf, ln->residBuf, FBP("fitUni"), ln->refitDbg});
  ln->pickGroup = dev.createBindGroup(pick.bgl, {ln->framesBuf, ln->quadsBuf, ln->scoreBuf, ln->mapsBuf, ln->selBuf, ln->resultBuf, FBP("pickUni")});
  if (dev.features.timestamps) {
    ln->qs = dev.createQuerySet(C.TS_PLAIN);
    ln->qsBuf = mk(8ull * C.TS_PLAIN, QUERY_RESOLVE | COPY_SRC, "qsBuf");
  }
  ln->laneBytes = dev.allocated - a0;
  return ln;
}

// ai: decoder.mjs netGroup: a classifier's group over the lane's levels, peaks, counts and readings, and its
// ai: weights; the cascade's small net writes its logits, the large one reads the list.
GroupP FrontHalf::netGroup(Lane& ln, const Pipe& pipe, wg::Buffer& buf, bool small) {
  auto e = ln.lvEntries();
  e.insert(e.end(), {ln.peaksBuf, ln.countsBuf, ln.readingsBuf, FBP("p4Uni"), &buf});
  if (small) e.push_back(ln.logitBuf);
  else if (!cascade.is_null()) e.push_back(ln.listBuf);
  return dev.createBindGroup(pipe.bgl, e);
}

void FrontHalf::bindNets(Lane& ln) {
  ln.classifyGroup = netGroup(ln, classify, *weightsBuf, false);
  if (!cascade.is_null()) ln.classify0Group = netGroup(ln, classify0, *weightsBuf0, true);
}

// ai: back.mjs lane: transform.lane, soft.lane, ldpc.lane, then transform.bindGate (the gate zeroes the counters,
// ai: verdicts and record count, so it binds V, ITS and REC).
void FrontHalf::backLane(Lane& ln) {
  using namespace wg;
  const json& bk = S->back();
  const json& T = bk["transform"];
  const json& so = bk["soft"];
  const json& ld = bk["ldpc"];
  const uint64_t Bb = bk["B"].get<uint64_t>(), cb = T["cbytes"].get<uint64_t>();
  const uint64_t a0 = dev.allocated;
  auto b = std::make_unique<BackLane>();
  auto mk = [&](uint64_t size, uint32_t usage, const char* name) { return dev.createBuffer(size, usage, dev.traceOut ? std::string("n") + name + std::to_string(dev.nextId++) : ""); };
  const uint64_t listsWords = C.SLOTS * (1 + Bb) + 5 * Bb;
  b->lists = mk(4 * listsWords, STORAGE | COPY_SRC, "lists");
  b->args = mk(4ull * C.argsWords, STORAGE | INDIRECT | COPY_SRC, "args");
  b->part = mk(16 * T["partStride"].get<uint64_t>() * Bb, STORAGE | COPY_SRC, "part");
  b->coef = mk(32 * Bb, STORAGE | COPY_SRC, "coef");
  b->y = mk(cb * T["yStride"].get<uint64_t>() * Bb, STORAGE | COPY_SRC, "y");
  b->s = mk(cb * T["sStride"].get<uint64_t>() * Bb, STORAGE | COPY_SRC, "s");
  b->counts = mk(4 * 8 * Bb, STORAGE | COPY_SRC | COPY_DST, "bcounts");
  b->params.assign(C.SLOTS, nullptr);
  b->red.assign(C.SLOTS, nullptr);
  b->p2.assign(C.SLOTS, nullptr);
  b->p1f.assign(C.SLOTS, nullptr);
  b->soft.assign(C.SLOTS, nullptr);
  for (auto& sj : T["built"]) {
    const int s = sj;
    const json& tb = T["sizes"][s];
    auto p = mk(48, UNIFORM | COPY_DST, "tparams");
    uint32_t u[12] = {};
    float* f = (float*)u;
    u[0] = (uint32_t)ln.gridStride; u[1] = T["yStride"]; u[2] = T["partStride"]; u[3] = (uint32_t)s; u[4] = tb["n"]; u[5] = 0;
    f[6] = tb["m2"].get<float>(); f[7] = tb["sx2"].get<float>(); f[8] = tb["sq2"].get<float>(); u[9] = T["sStride"]; u[10] = T["sampStride"];
    dev.writeBuffer(*p, 0, u, 48);
    b->params[s] = p;
    b->red[s] = dev.createBindGroup(S->bgl(T["redBgl"]), {b->lists, b->part, b->coef, p});
    b->p2[s] = dev.createBindGroup(S->bgl(T["p2Bgl"]), {b->lists, S->bufPtr(T["twBufs"][s]), b->part, b->y, b->coef, S->bufPtr(T["x12Bufs"][s]), S->bufPtr(T["rowsBufs"][s]), b->s, b->counts, p});
  }
  // ai: soft.mjs lane
  b->L = mk(so["bytes"]["L"].get<uint64_t>(), STORAGE | COPY_SRC, "L");
  b->EST = mk(so["bytes"]["EST"].get<uint64_t>(), STORAGE | COPY_SRC, "EST");
  b->BLK = mk(so["bytes"]["BLK"].get<uint64_t>(), STORAGE | COPY_SRC, "BLK");
  b->PILOT = mk(so["bytes"]["PILOT"].get<uint64_t>(), STORAGE | COPY_SRC, "PILOT");
  for (auto& sj : so["served"]) {
    const int s = sj;
    b->soft[s] = dev.createBindGroup(S->bgl(so["bgl"]), {b->s, S->bufPtr(so["uvBuf"]), b->lists, b->L, b->EST, b->BLK, b->counts, S->bufPtr(so["params"][s]), b->PILOT, S->bufPtr(so["twBuf"])});
  }
  // ai: ldpc.mjs lane
  b->V = mk(ld["bytes"]["V"].get<uint64_t>(), STORAGE | COPY_SRC | COPY_DST, "V");
  b->ITS = mk(ld["bytes"]["ITS"].get<uint64_t>(), STORAGE | COPY_SRC | COPY_DST, "ITS");
  b->REC = mk(ld["bytes"]["REC"].get<uint64_t>(), STORAGE | COPY_SRC | COPY_DST, "REC");
  b->ldpc = dev.createBindGroup(S->bgl(ld["bgl"]), {b->L, b->BLK, b->lists, S->bufPtr(ld["mapBuf"]), b->V, b->ITS, b->REC, b->counts, S->bufPtr(ld["params"])});
  // ai: transform.mjs bindGate (zero)
  b->gate = dev.createBindGroup(S->bgl(T["gateBgl"]), {ln.framesBuf, ln.selBuf, b->lists, b->args, S->bufPtr(T["gateUni"]), b->counts, b->V, b->ITS, b->REC});
  // ai: the rate profile's steps (gpu/back/tiers.mjs exportFor): a bind a lane buffer by name or a setup buffer by id
  if (const json* st = tierStage()) {
    b->PS = mk((*st)["bytes"]["PS"].get<uint64_t>(), STORAGE | COPY_SRC, "PS");
    const std::map<std::string, BufP> laneBufs = {{"S", b->s}, {"LISTS", b->lists}, {"L", b->L}, {"EST", b->EST}, {"BLK", b->BLK}, {"BCOUNTS", b->counts},
                                                  {"PILOT", b->PILOT}, {"PS", b->PS}, {"V", b->V}, {"ITS", b->ITS}, {"REC", b->REC}};
    auto group = [&](const json& step) {
      std::vector<wg::Resource> bufs;
      for (const auto& x : step["binds"]) {
        const std::string id = x;
        if (id.rfind("lane:", 0) == 0) bufs.push_back(laneBufs.at(id.substr(5))); else bufs.push_back(S->bufPtr(id));
      }
      return dev.createBindGroup(S->bgl(step["bgl"]), bufs);
    };
    for (const auto& step : (*st)["soft"]) b->tsoft.push_back(group(step));
    for (const auto& step : (*st)["ldpc"]) b->tldpc.push_back(group(step));
  }
  b->bytes = dev.allocated - a0;
  ln.back = std::move(b);
}

// ai: transform.mjs bindPicture: the fused pass 1 over the lane's texture, maps and residuals, again whenever the
// ai: lane's texture is replaced.
void FrontHalf::bindPicture(Lane& ln) {
  const json& T = S->back()["transform"];
  auto& b = *ln.back;
  for (auto& sj : T["built"]) {
    const int s = sj;
    b.p1f[s] = dev.createBindGroup(S->bgl(T["p1fBgl"]), {ln.tex.get(), b.lists, S->bufPtr(T["twBufs"][s]), b.part, b.y, b.counts, b.params[s], ln.mapsBuf, ln.residBuf, S->bufPtr(T["picUnis"][s])});
  }
  b.picture = ln.tex.get();
}

// ai: decoder.mjs ensure(W, H, nmax): capacity for frames up to W x H and pictures up to nmax; grows, never shrinks.
// ai: No grid (the fused pass 1 samples the picture), so gridStride only sizes the back half's params, as on the web.
void FrontHalf::ensure(Lane& ln, uint32_t W, uint32_t H, uint32_t nmax) {
  using namespace wg;
  const uint64_t a0 = dev.allocated, B = ln.B;
  auto mk = [&](uint64_t size, uint32_t usage, const char* name) { return dev.createBuffer(size, usage, dev.traceOut ? std::string("n") + name + std::to_string(dev.nextId++) : ""); };
  bool texChanged = false;
  if (W > ln.W || H > ln.H) {
    ln.W = std::max(W, ln.W);
    ln.H = std::max(H, ln.H);
    ln.tex = dev.createTexture(ln.W, ln.H, (uint32_t)B, dev.traceOut ? "ntex" + std::to_string(dev.nextId++) : "");
    texChanged = true;
    ln.level.clear();
    for (uint32_t l = 1; l < C.LEVELS; l++) {
      const uint32_t w = ceilDiv(ln.W, 1u << l), h = ceilDiv(ln.H, 1u << l);
      ln.level.push_back({w, h, mk(4ull * w * h * B, STORAGE, "level")});
    }
    auto dimsAt = [&](uint32_t l) { return l == 0 ? std::make_pair(ln.W, ln.H) : std::make_pair(ln.level[l - 1].w, ln.level[l - 1].h); };
    uint32_t slots = 0;
    std::vector<uint32_t> bases;
    for (uint32_t l = 0; l < C.LEVELS; l++) { auto [w, h] = dimsAt(l); bases.push_back(slots); slots += ceilDiv(w, 16) * ceilDiv(h, 16) * (uint32_t)bankKeep; }
    ln.slots = slots;
    ln.rawBuf = mk(16ull * slots * B, STORAGE | COPY_DST, "rawBuf");
    {
      uint32_t u[4] = {slots, (uint32_t)cap, 0, 0};
      memcpy(&u[2], &C.TAU, 4);
      dev.writeBuffer(*ln.selUni, 0, u, 16);
    }
    auto sel = P("select");
    ln.selGroup = dev.createBindGroup(sel.bgl, {ln.framesBuf, ln.rawBuf, ln.peaksBuf, ln.countsBuf, ln.selUni});
    // ai: bank_fcn2.mjs ensure: a uniform and (with the X plane) a plane a level, and its groups.
    const json& form = S->tree["front"]["fcn2Form"];
    const bool xplane = form["xplane"], int8 = form["int8"];
    const std::string store = form["store"];
    auto fcn2 = P("fcn2"), fcn2X = P("fcn2X");
    ln.fcn2Levels.clear();
    for (uint32_t l = 1; l < C.LEVELS; l++) {
      auto [w, h] = dimsAt(l);
      auto uni = mk(32, UNIFORM | COPY_DST, "fcn2Uni");
      uint32_t u[8] = {l, w, h, slots, 0, 0, bases[l], ceilDiv(w, C.TILE)};
      memcpy(&u[4], &C.TAU, 4);
      memcpy(&u[5], &C.EPS, 4);
      dev.writeBuffer(*uni, 0, u, 32);
      BufP plane;
      if (xplane) plane = mk(int8 ? (uint64_t)((w + 3) & ~3u) * h * B : (uint64_t)(store == "f32" ? 4 : 2) * (w + (w & 1)) * h * B, STORAGE, "fcn2Plane");
      auto group = dev.createBindGroup(fcn2.bgl, {plane ? plane : ln.level[l - 1].buf, ln.framesBuf, ln.rawBuf, ln.countsBuf, uni});
      GroupP xgroup = plane ? dev.createBindGroup(fcn2X.bgl, {ln.level[l - 1].buf, ln.framesBuf, plane, uni}) : nullptr;
      ln.fcn2Levels.push_back({w, h, group, xgroup, plane, uni});
    }
    {
      uint32_t ld[20] = {};
      for (size_t i = 0; i < ln.level.size(); i++) { ld[4 * (i + 1)] = ln.level[i].w; ld[4 * (i + 1) + 1] = ln.level[i].h; }
      dev.writeBuffer(*ln.ldUni, 0, ld, 80);
    }
    ln.gx = ceilDiv(ln.W, C.CELL);
    ln.gy = ceilDiv(ln.H, C.CELL);
    ln.accBuf = mk(4ull * C.RING_COUNT * ln.gx * ln.gy * B, STORAGE | COPY_DST, "accBuf");
    {
      uint32_t p5[8] = {(uint32_t)cap, ln.gx, ln.gy, 0};
      memcpy(&p5[4], modules.data(), 16);
      dev.writeBuffer(*ln.p5Uni, 0, p5, 32);
    }
    auto lv = ln.lvEntries();
    auto group = [&](const Pipe& pl, std::vector<wg::Resource> extra, bool withLevels) {
      std::vector<wg::Resource> e;
      if (withLevels) e = lv;
      e.insert(e.end(), extra.begin(), extra.end());
      return dev.createBindGroup(pl.bgl, e);
    };
    bindNets(ln);
    if (!cascade.is_null()) ln.rankGroup = group(P("rank"), {ln.framesBuf, ln.logitBuf, ln.countsBuf, ln.listBuf}, false);
    ln.scoreGroup = group(P("score"), {ln.quadsBuf, FBP("planBuf"), ln.scoreBuf, FBP("planUni")}, true);
    ln.voteGroup = group(P("vote"), {ln.framesBuf, ln.readingsBuf, ln.accBuf, ln.p5Uni}, false);
    ln.gatherGroup = group(P("gather"), {ln.framesBuf, ln.readingsBuf, ln.accBuf, ln.candBuf, ln.quadsBuf, ln.p5Uni}, false);
    ln.nodesGroups.clear();
    for (auto& r : S->tree["front"]["rounds"]) ln.nodesGroups.push_back(group(P("nodesP"), {FBP("nodesBuf"), FBP("tmplBuf"), ln.mapsBuf, ln.selBuf, ln.measBuf, S->bufPtr(r)}, true));
    ln.wordGroup = group(P("wordP"), {ln.mapsBuf, ln.selBuf, ln.residBuf, FBP("wordPtsBuf"), ln.resultBuf, FBP("wordUni"), FBP("wordLatUni"), ln.heldUni}, true);
    // ai: decoder.mjs pyrGroup: the layer, the frame records, levels 1 to 4, the histogram and the levels' dims
    ln.pyrGroup = dev.createBindGroup(P("pyramid").bgl, {ln.tex.get(), ln.framesBuf, ln.level[0].buf, ln.level[1].buf, ln.level[2].buf, ln.level[3].buf, ln.histBuf, ln.ldUni});
  }
  // ai: gridStride: the picture capacity a frame, which the back half's params are sized by; the back lane is made
  // ai: again when it grows (its params hold it). No grid buffer: F9 is not ported.
  if ((uint64_t)nmax * nmax > ln.gridStride) {
    ln.gridStride = (uint64_t)nmax * nmax;
    ln.readBuf.reset();
    const uint64_t b0 = dev.allocated;
    ln.back.reset();
    backLane(ln);
    ln.backBytes = dev.allocated > b0 ? ln.back->bytes : ln.back->bytes;
    texChanged = true;
  }
  if (texChanged || ln.back->picture != ln.tex.get()) bindPicture(ln);
  if (!ln.readBuf) {
    // ai: counters, finder results, histograms, the back half's readback, the timestamps (and the ring's ingest pairs,
    // ai: which the native ring has none of: kept in the size so the offsets are the web's)
    const uint64_t bytes = 4ull * C.COUNTERS * B + 4ull * C.RESULT * B + 4ull * C.HIST_BINS * B + S->back()["readBytes"].get<uint64_t>() + 8ull * C.TS_MAX + 8ull * C.TS_INGEST * B;
    ln.readBuf = mk(bytes, MAP_READ | COPY_DST, "readBuf");
  }
  ln.laneBytes += dev.allocated - a0;
}

bool FrontHalf::needsPlan(uint32_t w, uint32_t h) const {
  return !planned || w > planned->W || h > planned->H || nmaxFind > planned->nmax;
}

// ai: decoder.mjs plan: the lanes planned for every shape seen so far and this one, a frame priced with a one-frame
// ai: lane (its back lane's bytes shared out over bh.B), the batch size set from the budget and the device's limits.
void FrontHalf::plan(uint32_t w, uint32_t h) {
  const double t0 = now();
  const Planned p = planned.value_or(Planned{});
  const uint32_t W = std::max(p.W, w), H = std::max(p.H, h), nmax = std::max(p.nmax, nmaxFind);
  lanes.clear();
  destroyRing();
  const auto& L = dev.limits;
  for (;;) {
    const int Bb = S->backB;
    const uint64_t front = dev.allocated - backShared;
    std::unique_ptr<Lane> probe;
    try {
      probe = lane(1);
      ensure(*probe, W, H, nmax);
    } catch (const wg::OutOfMemory& e) {
      probe.reset();
      if (Bb == 1) throw wg::Error(std::string("out of memory for a one-frame lane: ") + e.what());
      buildBack(Bb >> 1);
      continue;
    }
    const uint64_t readBack = S->back()["readBytes"].get<uint64_t>();
    const double backBytes = (double)probe->back->bytes + readBack;
    const double perFrame = (double)probe->laneBytes - backBytes + backBytes / Bb;
    // ai: The largest buffer a frame (the back lane's at a share of bh.B, the readback's back half share likewise).
    double largest = 0;
    auto front1 = {probe->peaksBuf, probe->framesBuf, probe->mapsBuf, probe->countsBuf, probe->readingsBuf, probe->candBuf, probe->quadsBuf, probe->scoreBuf, probe->selBuf, probe->resultBuf, probe->measBuf, probe->residBuf, probe->refitDbg, probe->histBuf, probe->logitBuf, probe->listBuf, probe->rawBuf, probe->accBuf};
    for (auto& b : front1) if (b) largest = std::max(largest, (double)b->size);
    for (auto& l : probe->level) largest = std::max(largest, (double)l.buf->size);
    for (auto& l : probe->fcn2Levels) if (l.plane) largest = std::max(largest, (double)l.plane->size);
    largest = std::max(largest, (double)W * H);   // ai: the texture's layer
    largest = std::max(largest, (double)probe->readBuf->size - readBack + (double)readBack / Bb);
    auto& bk = *probe->back;
    for (auto& b : {bk.lists, bk.args, bk.part, bk.coef, bk.y, bk.s, bk.counts, bk.L, bk.EST, bk.BLK, bk.V, bk.ITS, bk.REC, bk.PILOT}) largest = std::max(largest, (double)b->size / Bb);
    probe.reset();
    int capB = INT32_MAX;
    std::string capWhy;
    const double byBuffer = std::floor((double)std::min(L.maxBufferSize, L.maxStorageBufferRange) / largest);
    if (byBuffer < capB) { capB = (int)byBuffer; char m[160]; snprintf(m, sizeof m, "the device's buffer size limit (a frame's largest buffer %.1f MB)", largest / MiB); capWhy = m; }
    if ((int)L.maxImageArrayLayers < capB) { capB = (int)L.maxImageArrayLayers; capWhy = "the device's texture layer limit"; }
    bt.perFrame = perFrame;
    bt.shared = (double)front + backShared;
    bt.room = bt.budget - (double)front - backShared;
    bt.cap = capB;
    bt.capWhy = capWhy;
    bt.slotBytes = (double)W * H;
    const int Bmax = bt.limitFor(bt.lanes).first;
    if (Bmax >= Bb) break;
    buildBack(Bmax);
  }
  planned = Planned{W, H, nmax};
  makeLanes(bt.lanes);
  bt.start();
  bt.planMs = now() - t0;
  log("batch auto: planned in " + std::to_string((int)bt.planMs) + " ms");
}

// ai: decoder.mjs makeLanes: n lanes of the largest batch the budget allows, and the ring beside them; an
// ai: allocation that fails is tried again at half.
void FrontHalf::makeLanes(int n) {
  lanes.clear();
  auto [Bmax, why] = bt.limitFor(n);
  for (;;) {
    try {
      if (S->backB != Bmax) buildBack(Bmax);
      B = Bmax;
      for (int i = 0; i < n; i++) lanes.push_back(lane(Bmax));
      for (auto& l : lanes) ensure(*l, planned->W, planned->H, planned->nmax);
      makeRing(Bmax + (int)C.RING_SLACK, planned->W, planned->H);
      dev.flush();
      break;
    } catch (const wg::OutOfMemory& e) {
      lanes.clear();
      destroyRing();
      if (Bmax == 1) throw wg::Error(std::string("out of memory for lanes of one frame: ") + e.what());
      why = "an allocation failed at " + std::to_string(Bmax) + " (" + e.what() + ")";
      Bmax /= 2;
    }
  }
  bt.laned(n, Bmax, why);
}

// ai: decoder.mjs makeRing: R layers at W x H, made with the lanes. The web writes a luma frame into a layer with
// ai: writeTexture; here a slot's host staging buffer is copied into it by a submit of its own (enqueueLuma), and the
// ai: phone's camera writes it on the device, so the ring is a storage image too.
void FrontHalf::makeRing(int R, uint32_t W, uint32_t H) {
  destroyRing();
  std::lock_guard<std::mutex> lock(ringMu);
  Ring r;
  r.R = R; r.W = W; r.H = H;
  r.tex = dev.createTexture(W, H, (uint32_t)R, dev.traceOut ? "nring" + std::to_string(dev.nextId++) : "", true);
  r.slots.assign(R, nullptr);
  r.staging.assign(R, nullptr);
  for (int i = 0; i < R; i++) r.store.push_back(std::make_unique<Slot>());
  ring = std::move(r);
}

void FrontHalf::destroyRing() {
  std::lock_guard<std::mutex> lock(ringMu);
  if (ring && ingest) vkQueueWaitIdle(dev.queue2);   // ai: no ingest may still write the ring that goes
  ring.reset();
}

// ai: decoder.mjs enqueue for a luma frame: a free slot (from the head on), the frame's rows copied to its staging
// ai: and a copy of that into the slot's layer submitted at once, so it lands before any batch that reads the slot.
int FrontHalf::takeSlot() {
  auto& r = *ring;
  for (int k = 0; k < r.R; k++) {
    const int id = (r.head + k) % r.R;
    if (r.slots[id]) continue;
    auto& rd = r.store[id]->reading;
    if (rd && !rd->done()) continue;
    rd.reset();
    r.head = (id + 1) % r.R;
    return id;
  }
  return -1;
}

Slot* FrontHalf::enqueueLuma(const uint8_t* bytes, uint32_t w, uint32_t h, uint32_t stride, uint64_t tag) {
  std::lock_guard<std::mutex> lock(ringMu);
  if (!ring) throw wg::Error("no ring: plan() before enqueue");
  auto& r = *ring;
  if (w > r.W || h > r.H) throw wg::Error(std::to_string(w) + " x " + std::to_string(h) + ": the ring holds " + std::to_string(r.W) + " x " + std::to_string(r.H));
  const int id = takeSlot();
  if (id < 0) return nullptr;
  Slot* s = r.store[id].get();
  // ai: Only the bytes here: this may be the camera's thread, and the device records on the decoder's (wg.h
  // ai: "Threads"). The batch that takes the slot copies the staging into its layer (run), and takeSlot hands a
  // ai: slot out again only once that batch's copies are done, so the staging is free to write.
  if (!r.staging[id]) r.staging[id] = dev.createBuffer((uint64_t)r.W * r.H, wg::MAP_WRITE, dev.traceOut ? "nstage" + std::to_string(dev.nextId++) : "");
  auto& st = *r.staging[id];
  for (uint32_t y = 0; y < h; y++) memcpy(st.mapped + (uint64_t)y * w, bytes + (uint64_t)y * stride, w);
  s->luma = true;
  s->id = id; s->w = w; s->h = h; s->at = now(); s->tag = tag; s->ready = 0;
  r.slots[id] = s;
  r.staged++;
  return s;
}

Slot* FrontHalf::enqueueCamera(CameraIngest& ci, AHardwareBuffer* hb, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag) {
  std::lock_guard<std::mutex> lock(ringMu);
  if (!ring) return nullptr;
  auto& r = *ring;
  if (w > r.W || h > r.H) throw wg::Error(std::to_string(w) + " x " + std::to_string(h) + ": the ring holds " + std::to_string(r.W) + " x " + std::to_string(r.H));
  const int id = takeSlot();
  if (id < 0) return nullptr;
  Slot* s = r.store[id].get();
  s->ready = ci.ingest(hb, x, y, w, h, *r.tex, (uint32_t)id);
  s->id = id; s->w = w; s->h = h; s->at = now(); s->tag = tag; s->luma = false;
  r.slots[id] = s;
  r.staged++;
  return s;
}

void FrontHalf::release(Slot* s) {
  std::lock_guard<std::mutex> lock(ringMu);
  if (!ring || !s || s->id < 0) return;
  auto& r = *ring;
  if (r.slots[s->id] == s) { r.slots[s->id] = nullptr; r.staged--; }
}

Lane* FrontHalf::freeLane() {
  for (auto& l : lanes) if (!l->busy) return l.get();
  return nullptr;
}

// ai: The back half's six dispatches (back.mjs encode: gate, pass 1 fused, reduce, pass 2, soft, LDPC from the gate).
void FrontHalf::encodeBack(wg::Encoder& e, Lane& ln, int frames, const std::function<void(const char*)>& begin) {
  const json& bk = S->back();
  const json& T = bk["transform"];
  const json& so = bk["soft"];
  auto& b = *ln.back;
  begin("gate");
  e.setPipeline(*S->pipe(T["gate"]));
  e.setBindGroup(*b.gate);
  e.dispatch(1);
  begin("pass1");
  for (auto& sj : T["built"]) { const int s = sj; e.setPipeline(*S->pipe(T["pass1f"][s])); e.setBindGroup(*b.p1f[s]); e.dispatchIndirect(*b.args, argsOffset(s, "pass1")); }
  begin("reduce");
  for (auto& sj : T["built"]) { const int s = sj; e.setPipeline(*S->pipe(T["reduce"][s])); e.setBindGroup(*b.red[s]); e.dispatchIndirect(*b.args, argsOffset(s, "reduce")); }
  begin("pass2");
  for (auto& sj : T["built"]) { const int s = sj; e.setPipeline(*S->pipe(T["pass2"][s])); e.setBindGroup(*b.p2[s]); e.dispatchIndirect(*b.args, argsOffset(s, "pass2")); }
  begin("soft");
  // ai: soft.mjs dispatch: the shift's fit by the pilots first (a workgroup a frame, a dispatch a size, on the soft
  // ai: stage's bind groups), then the soft values, which turn each coefficient back by it
  // ai: (LIZ_ALIGN=0 leaves the fit out, for an A/B: the shift then stays the 0, 0 its buffer was made with)
  static const bool align = !(getenv("LIZ_ALIGN") && std::string(getenv("LIZ_ALIGN")) == "0");
  const json* st = tierStage();
  if (align) {
    e.setPipeline(*S->pipe(so["align"]));
    for (auto& sj : so["served"]) { if ((int)sj == tslot) continue; e.setBindGroup(*b.soft[(int)sj]); e.dispatch(1, 1, (uint32_t)frames); }
  }
  e.setPipeline(*S->pipe(so["pipeline"]));
  for (auto& sj : so["served"]) {
    const int s = sj;
    if (s == tslot) continue;
    e.setBindGroup(*b.soft[s]);
    if (so["indirect"].get<bool>()) e.dispatchIndirect(*b.args, 4ull * (C.ARGS_WORDS * s + C.ARGS_SLOT.at("blocks")));
    else e.dispatch(so["blocks"][s].get<uint32_t>(), 1, (uint32_t)frames);
  }
  // ai: a step of the rate profile's (exportFor): its dispatch's numbers, "frames" the batch's
  auto stepOf = [&](const json& step, const GroupP& g) {
    e.setPipeline(*S->pipe(step["pipeline"]));
    e.setBindGroup(*g);
    if (step.contains("indirect")) { e.dispatchIndirect(*b.args, step["indirect"].get<uint64_t>()); return; }
    uint32_t d[3];
    for (int k = 0; k < 3; k++) d[k] = step["dispatch"][k].is_string() ? (uint32_t)frames : step["dispatch"][k].get<uint32_t>();
    e.dispatch(d[0], d[1], d[2]);
  };
  if (st) for (size_t k = 0; k < b.tsoft.size(); k++) stepOf((*st)["soft"][k], b.tsoft[k]);
  begin("ldpc");
  if (st) { for (size_t k = 0; k < b.tldpc.size(); k++) stepOf((*st)["ldpc"][k], b.tldpc[k]); return; }
  e.setPipeline(*S->pipe(bk["ldpc"]["pipeline"]));
  e.setBindGroup(*b.ldpc);
  e.dispatchIndirect(*b.args, 4ull * C.argsLdpc);
}

// ai: decoder.mjs batch, up to the submit: the frame records and held version written, each slot copied into its
// ai: lane layer, the buffers the batch writes cleared for its nb frames, one pass of every stage, the readback's
// ai: copies; the slots freed at the submit (the copies are queued before any later upload into them).
std::unique_ptr<InFlight> FrontHalf::run(const std::vector<Slot*>& slots, int held, bool profile, int keep) {
  const int nb = (int)slots.size();
  if (!nb) throw wg::Error("a batch of no slots");
  if (nb > B) throw wg::Error(std::to_string(nb) + " frames in a batch of " + std::to_string(B));
  if (held < 0 || held > (int)C.VERSION_MAX) throw wg::Error("held " + std::to_string(held));
  Lane* lnp = freeLane();
  if (!lnp) throw wg::Error(std::to_string(lanes.size()) + " batches in flight already");
  Lane& ln = *lnp;
  auto f = std::make_unique<InFlight>();
  f->t0 = now();
  uint32_t W = 1, H = 1;
  for (auto* s : slots) if (s) { W = std::max(W, s->w); H = std::max(H, s->h); }
  ensure(ln, W, H, nmaxFind);
  const uint64_t Bl = ln.B;
  { uint32_t hu[4] = {(uint32_t)held, 0, 0, 0}; dev.writeBuffer(*ln.heldUni, 0, hu, 16); }
  {
    std::vector<uint32_t> dims(4 * ln.slotsG, 0);
    for (int i = 0; i < nb; i++) if (slots[i]) { dims[4 * i] = slots[i]->w; dims[4 * i + 1] = slots[i]->h; dims[4 * i + 3] = 1; }
    dev.writeBuffer(*ln.framesBuf, 0, dims.data(), dims.size() * 4);
  }
  // ai: keep (Save replays, 2026-10-03) and LIZ_DUMP=<dir> (a debugging look, 2026-10-01: the S26 at 3840x2160
  // ai: found nothing where the same decoder reads a 2160 crop off a file): frames as the camera's ingest left them
  // ai: in the ring, copied into the lane's keepBuf with the slots' copies and read back with the batch (finish).
  // ai: LIZ_DUMP's is every 2 s the batch's first frame, written to <dir>/ingest-<w>x<h>.gray (w x h bytes, rows of
  // ai: w), overwritten each time. The lane is free, so nothing reads its keepBuf now. The buffer is made before
  // ai: anything is recorded: one the device refuses (its memory) keeps nothing this batch, said in the log, and the
  // ai: batch goes on (before 2026-10-03, thrown from the copies, it left the batch's ring slots taken for good).
  static const std::string dumpDir = getenv("LIZ_DUMP") ? getenv("LIZ_DUMP") : "";
  f->kept = std::clamp(keep, 0, nb);
  f->dump = !dumpDir.empty() && slots[0] && (!lastDump || now() - lastDump > 2000);
  if (f->dump) lastDump = now();
  uint64_t keepBytes = 0;
  for (int i = 0; i < std::max(f->kept, f->dump ? 1 : 0); i++) {
    f->keep.push_back({keepBytes, slots[i] ? slots[i]->w : 0, slots[i] ? slots[i]->h : 0});
    keepBytes += ((uint64_t)f->keep.back().w * f->keep.back().h + 255) & ~255ull;
  }
  if (!keepBytes) dropKeep();
  else if (!ln.keepBuf || ln.keepBuf->size < keepBytes) {
    try {
      ln.keepBuf = dev.createBuffer(keepBytes, wg::MAP_READ | wg::COPY_DST, "keep", false);
      keeping = true;
    } catch (const std::exception& x) {
      log("keep: " + std::to_string(keepBytes >> 20) + " MB for the readback refused (" + x.what() + "): this batch's frames not kept");
      ln.keepBuf.reset(); f->keep.clear(); f->kept = 0; f->dump = false; keepBytes = 0;
    }
  }
  // ai: The slots' copies in a submit of their own, waiting for the camera ingests they copy (the ingest's queue
  // ai: signals its timeline): the slots are free to be written again once it is done, not once the batch is. A
  // ai: host-luma slot's staging goes into its ring layer here first (enqueueLuma wrote only the bytes).
  std::shared_ptr<wg::Ticket> copies;
  {
    auto c = dev.encoder();
    uint64_t wait = 0;
    std::lock_guard<std::mutex> lock(ringMu);
    for (int i = 0; i < nb; i++) if (slots[i]) {
      if (slots[i]->luma) c.copyBufferToTexture(*ring->staging[slots[i]->id], 0, slots[i]->w, *ring->tex, (uint32_t)slots[i]->id, slots[i]->w, slots[i]->h);
      c.copyTextureToTexture(*ring->tex, (uint32_t)slots[i]->id, *ln.tex, (uint32_t)i, slots[i]->w, slots[i]->h);
      wait = std::max(wait, slots[i]->ready);
    }
    for (size_t i = 0; i < f->keep.size(); i++)
      if (f->keep[i].w) c.copyTextureToBuffer(*ring->tex, (uint32_t)slots[i]->id, f->keep[i].w, f->keep[i].h, *ln.keepBuf, f->keep[i].off);
    copies = dev.submit(c, wait && ingest ? ingest->timeline : VK_NULL_HANDLE, wait);
    if (keepBytes) f->keepTicket = copies;
    for (int i = 0; i < nb; i++) if (slots[i]) slots[i]->reading = copies;
  }
  auto e = dev.encoder();
  f->host["upload"] = now() - f->t0;
  auto clear = [&](const BufP& b) { e.clearBuffer(*b, 0, (b->size / Bl) * nb); };
  clear(ln.countsBuf); clear(ln.peaksBuf);
  clear(ln.rawBuf); clear(ln.resultBuf); clear(ln.residBuf); clear(ln.histBuf);
  for (auto& b : {ln.readingsBuf, ln.accBuf, ln.candBuf, ln.quadsBuf, ln.scoreBuf, ln.mapsBuf, ln.measBuf}) clear(b);
  e.clearBuffer(*ln.selBuf);
  // ai: decoder.mjs STAGES: unprofiled one pass (its pair at 0, 1); profiled a pass a stage group.
  static const std::vector<std::string> STAGES = {"pyramid", "bank", "select", "describe", "vote", "score", "nodes1", "refit1", "nodes2", "refit2", "word", "gate", "pass1", "reduce", "pass2", "soft", "ldpc"};
  if (profile && dev.features.timestamps && (!ln.qsProfile || ln.qsProfile->count < 2 * STAGES.size())) {
    ln.qsProfile = dev.createQuerySet(2 * (uint32_t)STAGES.size());
    ln.qsProfileBuf = dev.createBuffer(16ull * STAGES.size(), wg::QUERY_RESOLVE | wg::COPY_SRC);
  }
  const bool prof = profile && ln.qsProfile;
  bool open = false;
  auto begin = [&](const char* name) {
    if (open && !prof) return;
    if (open) e.endPass();
    if (prof) {
      const int k = (int)(std::find(STAGES.begin(), STAGES.end(), name) - STAGES.begin());
      e.beginComputePass(ln.qsProfile.get(), 2 * k, 2 * k + 1);
      f->stages.push_back(name);
    } else e.beginComputePass(ln.qs.get(), 0, 1);
    open = true;
  };
  const uint32_t n = (uint32_t)nb;
  auto step = [&](const Pipe& pl, const GroupP& g, uint32_t x, uint32_t y = 1, uint32_t z = 1) { e.setPipeline(*pl.p); e.setBindGroup(*g); e.dispatch(x, y, z); };
  // ai: the pyramid and the grey levels in one kernel (wgsl/pyramid.mjs), a PYRAMID_BLOCK square of a frame a workgroup
  begin("pyramid");
  step(P("pyramid"), ln.pyrGroup, ceilDiv(W, C.PYRAMID_BLOCK), ceilDiv(H, C.PYRAMID_BLOCK), n);
  begin("bank");
  // ai: bank_fcn2.mjs dispatch
  {
    const json& form = S->tree["front"]["fcn2Form"];
    const uint32_t TX = form["TX"], TY = form["TY"], walk = form["walk"];
    auto fcn2X = P("fcn2X"), fcn2 = P("fcn2");
    if (fcn2X) {
      e.setPipeline(*fcn2X.p);
      const uint32_t XB0 = form["XB"][0], XB1 = form["XB"][1];
      for (auto& lv : ln.fcn2Levels) { e.setBindGroup(*lv.xgroup); e.dispatch(ceilDiv(lv.w, XB0), ceilDiv(lv.h, XB1), n); }
    }
    e.setPipeline(*fcn2.p);
    for (auto& lv : ln.fcn2Levels) { e.setBindGroup(*lv.group); e.dispatch(ceilDiv(ceilDiv(lv.w, C.TILE * TX), walk), ceilDiv(lv.h, C.TILE * TY), n); }
  }
  begin("select");
  step(P("select"), ln.selGroup, n);
  begin("describe");
  // ai: a workgroup of a classifier takes its kernel's P patches: ceil(slots / P) workgroups a frame (decoder.mjs)
  auto net = [&](const Pipe& k, const GroupP& g, uint32_t slots) { step(k, g, ceilDiv(slots, k.P), n); };
  if (!cascade.is_null()) {
    net(classify0, ln.classify0Group, (uint32_t)cap);
    step(P("rank"), ln.rankGroup, n);
    net(classify, ln.classifyGroup, cascade["keep"].get<uint32_t>());
  } else net(classify, ln.classifyGroup, (uint32_t)cap);
  begin("vote");
  // ai: F5 (wgsl/finder.mjs): VOTE a lane a reading, then GATHER a workgroup a frame (its centre peaks and their quads)
  step(P("vote"), ln.voteGroup, ceilDiv((uint32_t)cap, 64), n);
  step(P("gather"), ln.gatherGroup, n);
  begin("score");
  step(P("score"), ln.scoreGroup, C.HYP * n);
  step(P("pick"), ln.pickGroup, n);
  if (reg) for (size_t r = 0; r < ln.nodesGroups.size(); r++) {
    begin(r ? "nodes2" : "nodes1");
    step(P("nodesP"), ln.nodesGroups[r], (uint32_t)nodesMax, n);
    begin(r ? "refit2" : "refit1");
    step(P("refit"), ln.refitGroup, n);
  }
  begin("word");
  step(P("wordP"), ln.wordGroup, n);
  encodeBack(e, ln, nb, begin);
  e.endPass();
  // ai: the readback's layout (the GPU back half's): counters, finder results, histograms, the back half's (REC, V,
  // ai: ITS, BCOUNTS), timestamps
  const uint64_t cntBytes = 4ull * C.COUNTERS * nb, resBytes = 4ull * C.RESULT * nb, histBytes = 4ull * C.HIST_BINS * nb;
  f->cntOff = 0; f->resOff = cntBytes; f->histOff = f->resOff + resBytes; f->backOff = f->histOff + histBytes;
  f->tsOff = f->backOff + S->back()["readBytes"].get<uint64_t>();
  e.copyBufferToBuffer(*ln.countsBuf, 0, *ln.readBuf, f->cntOff, cntBytes);
  e.copyBufferToBuffer(*ln.resultBuf, 0, *ln.readBuf, f->resOff, resBytes);
  e.copyBufferToBuffer(*ln.histBuf, 0, *ln.readBuf, f->histOff, histBytes);
  {
    // ai: back.mjs readback: REC, V, ITS, BCOUNTS, PILOT in that order
    const json& by = S->back()["bytes"];
    auto& b = *ln.back;
    uint64_t o = f->backOff;
    for (auto [buf, k] : std::initializer_list<std::pair<BufP, const char*>>{{b.REC, "REC"}, {b.V, "V"}, {b.ITS, "ITS"}, {b.counts, "BCOUNTS"}, {b.PILOT, "PILOT"}}) {
      const uint64_t sz = by[k];
      e.copyBufferToBuffer(*buf, 0, *ln.readBuf, o, sz);
      o += sz;
    }
  }
  if (prof) {
    f->nq = ln.qsProfile->count;
    e.resolveQuerySet(*ln.qsProfile, 0, f->nq, *ln.qsProfileBuf, 0);
    e.copyBufferToBuffer(*ln.qsProfileBuf, 0, *ln.readBuf, f->tsOff, 8ull * f->nq);
  } else {
    f->nq = ln.qs ? ln.qs->count : 0;
    if (ln.qs) { e.resolveQuerySet(*ln.qs, 0, f->nq, *ln.qsBuf, 0); e.copyBufferToBuffer(*ln.qsBuf, 0, *ln.readBuf, f->tsOff, 8ull * f->nq); }
  }
  f->readBytes = f->tsOff + 8ull * f->nq;
  f->ticket = dev.submit(e);
  for (auto* s : slots) if (s) { f->tags.push_back(s->tag); release(s); } else f->tags.push_back(0);
  f->slots = slots;
  f->submitted = now();
  f->host["encode"] = f->submitted - f->t0 - f->host["upload"];
  f->lane = &ln;
  f->nb = nb;
  f->W = W; f->H = H;
  ln.busy = true;
  return f;
}

// ai: decoder.mjs batch after the submit, and back.mjs parseReadback: the readback mapped once the ticket is done,
// ai: each frame's finder result, word, counters, histogram and verified records; the batcher told; the int8 twins
// ai: measured on this batch's lane if they wait to be.
BatchOut FrontHalf::finish(std::unique_ptr<InFlight> f) {
  Lane* lane = f->lane;
  try { return finishBatch(std::move(f)); }
  catch (...) { lane->busy = false; throw; }   // ai: a batch that could not be read back gives its lane back
}

// ai: The lanes' readback buffers for kept frames let go where no batch is reading one (Save replays off, or the
// ai: camera stopped): a batch keeping nothing calls it, and so does the receiver while it idles. Decoder thread.
void FrontHalf::dropKeep() {
  if (!keeping) return;
  bool any = false;
  for (auto& l : lanes) { if (!l->busy) l->keepBuf.reset(); else if (l->keepBuf) any = true; }
  keeping = any;
}

BatchOut FrontHalf::finishBatch(std::unique_ptr<InFlight> f) {
  Lane& ln = *f->lane;
  BatchOut out;
  f->ticket->wait();
  out.done = now();
  f->host["map"] = out.done - f->submitted;
  const uint8_t* all = dev.read(*ln.readBuf, 0, f->readBytes);
  const int nb = f->nb;
  const uint32_t* counts = (const uint32_t*)(all + f->cntOff);
  const float* res = (const float*)(all + f->resOff);
  const uint32_t* hist = (const uint32_t*)(all + f->histOff);
  if (!f->stages.empty()) {
    static const std::vector<std::string> STAGES = {"pyramid", "bank", "select", "describe", "vote", "score", "nodes1", "refit1", "nodes2", "refit2", "word", "gate", "pass1", "reduce", "pass2", "soft", "ldpc"};
    const uint64_t* ts = (const uint64_t*)(all + f->tsOff);
    double sum = 0;
    for (auto& name : f->stages) {
      const size_t k = std::find(STAGES.begin(), STAGES.end(), name) - STAGES.begin();
      const double ms = (double)((ts[2 * k + 1] - ts[2 * k]) & dev.timestampMask) * dev.timestampPeriod / 1e6;
      out.stageMs.push_back({name, ms});
      sum += ms;
    }
    out.gpuMs = sum;
  } else if (f->nq >= 2) {
    const uint64_t* ts = (const uint64_t*)(all + f->tsOff);
    const uint64_t d = (ts[1] - ts[0]) & dev.timestampMask;
    out.gpuMs = (double)d * dev.timestampPeriod / 1e6;
  }
  // ai: the back half's part: REC (a count, then REC_BYTES a record), V, ITS, BCOUNTS, PILOT
  const json& bk = S->back();
  const uint64_t Bb = bk["B"], blocksMax = bk["dims"]["blocksMax"], recCap = bk["dims"]["recCap"];
  const uint8_t* rec = all + f->backOff;
  const uint64_t recBytes = bk["bytes"]["REC"], vBytes = bk["bytes"]["V"];
  const uint32_t* V = (const uint32_t*)(rec + recBytes);
  const uint32_t* bcounts = (const uint32_t*)(rec + recBytes + 2 * vBytes);
  const float* pilots = (const float*)(rec + recBytes + 2 * vBytes + bk["bytes"]["BCOUNTS"].get<uint64_t>());
  uint32_t count;
  memcpy(&count, rec, 4);
  count = std::min<uint32_t>(count, (uint32_t)recCap);
  out.frames.resize(nb);
  for (uint32_t r = 0; r < count; r++) {
    const uint8_t* o = rec + 4 + (uint64_t)C.REC_BYTES * r;
    uint32_t tag; int32_t its;
    memcpy(&tag, o, 4); memcpy(&its, o + 4, 4);
    const int fr = tag & 0xffff, b = tag >> 16;
    if (fr >= (int)Bb || b >= (int)blocksMax || fr >= nb) continue;
    Record x;
    x.frame = fr; x.block = b; x.its = its;
    memcpy(&x.id, o + 8, 4);
    memcpy(x.payload.data(), o + 8, C.PAYLOAD);
    out.frames[fr].records.push_back(x);
  }
  const auto& pics = S->tree["pictures"];
  auto slotOf = [&](int version) { for (size_t i = 0; i < pics.size(); i++) if (version >= pics[i]["lo"].get<int>() && version <= pics[i]["hi"].get<int>()) return (int)i; return -1; };
  for (int i = 0; i < nb; i++) {
    auto& fo = out.frames[i];
    fo.counts.assign(counts + C.COUNTERS * i, counts + C.COUNTERS * (i + 1));
    fo.backCounts.assign(bcounts + C.COUNTS * i, bcounts + C.COUNTS * (i + 1));
    for (uint64_t b = 0; b < blocksMax; b++) fo.verdicts[std::min<uint32_t>(V[i * blocksMax + b], 4)]++;
    fo.tag = f->tags[i];
    if (!f->slots[i]) { fo.empty = true; continue; }
    const float* r = res + C.RESULT * i;
    fo.ring = r[0] == 1 ? (int)r[2] : -1;
    fo.version = (int)r[29];
    fo.size = fo.version > 0 ? slotOf(fo.version) : -1;
    fo.n = fo.size >= 0 ? pics[fo.size]["n"].get<int>() : 0;
    fo.held = r[30] == 1;
    fo.sampled = fo.version > 0 && r[0] == 1;
    fo.finder.found = r[0]; fo.finder.score = r[1]; fo.finder.ring = r[2]; fo.finder.orient = r[3]; fo.finder.quad = r[4];
    for (int k = 0; k < 9; k++) fo.finder.H[k] = r[8 + k];
    for (int k = 0; k < 8; k++) fo.finder.corners[k] = r[17 + k];
    for (int k = 0; k < 4; k++) fo.finder.depths[k] = r[25 + k];
    if (r[5] > 0) { fo.hasWord = true; fo.word.version = (int)r[5]; fo.word.fps = (int)r[6]; fo.word.score = r[7]; }
    fo.hist.assign(hist + C.HIST_BINS * i, hist + C.HIST_BINS * (i + 1));
    // ai: decoder.mjs pilotOf: over the blocks the version carries, where the soft stage ran (its count column 6)
    // ai: the even blocks (bit 0 of the painted count) and the odd (bit 1), each its own mean and standard error
    // ai: the blocks the frame carries: its version's, or a rate profile's at its slot (gpu/back/tiers.mjs blocksAt)
    const int nblk = !tiersKey.empty() && fo.size == tslot && fo.version == tversion ? tblocks : fo.version;
    if (fo.version > 0 && fo.backCounts.size() > 6 && fo.backCounts[6] > 0) for (int g = 0; g < 2; g++) {
      double s = 0, s2 = 0;
      int k = 0;
      for (int b = g; b < std::min<int>(nblk, (int)blocksMax); b += 2) { const float v = pilots[i * blocksMax + b]; if (std::isfinite(v)) { s += v; s2 += (double)v * v; k++; } }
      if (!k) continue;
      const double m = s / k, var = k > 1 ? (s2 - s * m) / (k - 1) : 0;
      fo.pilotBlocks += k;
      (g ? fo.pilotR2 : fo.pilotR) = (float)m; (g ? fo.pilotSd2 : fo.pilotSd) = (float)std::sqrt(std::max(var, 0.0) / k);
    }
  }
  // ai: run's keep: each kept frame's bytes off the lane's keepBuf; LIZ_DUMP's written, and handed on only where the
  // ai: caller asked for it too
  if (f->keepTicket) {
    f->keepTicket->wait();
    const uint8_t* kb = dev.read(*ln.keepBuf, 0, ln.keepBuf->size);
    for (size_t i = 0; i < f->keep.size(); i++) if (f->keep[i].w) {
      auto& fo = out.frames[i];
      fo.w = f->keep[i].w; fo.h = f->keep[i].h;
      fo.luma.assign(kb + f->keep[i].off, kb + f->keep[i].off + (size_t)fo.w * fo.h);
    }
    if (f->dump && !out.frames[0].luma.empty()) {
      const auto& fo = out.frames[0];
      const std::string path = std::string(getenv("LIZ_DUMP")) + "/ingest-" + std::to_string(fo.w) + "x" + std::to_string(fo.h) + ".gray";
      if (FILE* fp = fopen(path.c_str(), "wb")) { fwrite(fo.luma.data(), 1, fo.luma.size(), fp); fclose(fp); log("dump: " + path); }
      else log("dump: cannot write " + path);
    }
    for (size_t i = f->kept; i < f->keep.size(); i++) out.frames[i].luma = {};
  }
  out.carried = nb;
  out.submitted = f->submitted;
  out.deviceMs = out.gpuMs ? *out.gpuMs : out.done - std::max(f->submitted, lastDone);
  out.idle = f->submitted >= lastDone;
  lastDone = std::max(lastDone, out.done);
  f->host["read"] = now() - out.done;
  out.host = f->host;
  bt.observe(nb, out.deviceMs, !ln.watched);
  ln.watched = true;
  ln.busy = false;
  int lead = 0;
  while (lead < std::min<int>(nb, (int)C.TWIN_FRAMES) && f->slots[lead]) lead++;
  if (!twinForms.is_null() && lead) {
    const double t1 = now();
    try { measureTwins(ln, lead); } catch (const std::exception& ex) { log(std::string("int8 twins not measured (") + ex.what() + "): the float files"); twinForms = nullptr; }
    out.twinMs = now() - t1;
  }
  return out;
}

// ai: decoder.mjs measureTwins and timeArms: each classifier with an int8 twin timed in both forms on this batch's
// ai: first nb frames (still on its lane), by timestamp pairs, three rounds in alternating order, the first untimed;
// ai: the faster kept, the lanes' groups remade. A device without timestamps keeps the float forms (the web times
// ai: submits there; every device this runs on has timestamps).
void FrontHalf::measureTwins(Lane& ln, int nb) {
  json tf = twinForms;
  twinForms = nullptr;
  if (!dev.features.timestamps) { log("int8 twins not measured (no timestamps): the float files"); return; }
  struct Arm { std::string net, precision; Pipe pipe; BufP buf; GroupP group; uint32_t x; std::vector<double> ms; };
  std::vector<Arm> arms;
  for (const char* net : {"small", "large"}) {
    if (tf[net].is_null()) continue;
    const bool small = std::string(net) == "small";
    const uint32_t slots = small || cascade.is_null() ? (uint32_t)cap : cascade["keep"].get<uint32_t>();
    Arm now{net, small ? cascade["precision"].get<std::string>() : S->tree["weightsPrecision"].get<std::string>(), small ? classify0 : classify, small ? weightsBuf0 : weightsBuf, nullptr, 0, {}};
    now.group = netGroup(ln, now.pipe, *now.buf, small);
    now.x = ceilDiv(slots, now.pipe.P);
    Arm twin{net, "int8", S->pair(tf[net]["pipe"]), S->bufPtr(tf[net]["buf"]), nullptr, 0, {}};
    twin.group = netGroup(ln, twin.pipe, *twin.buf, small);
    twin.x = ceilDiv(slots, twin.pipe.P);
    arms.push_back(now);
    arms.push_back(twin);
    // ai: A native kernel for this twin (Setup::natives) is a third form, if it is the twin's arithmetic on this
    // ai: device: both run on the lane's first nb frames (the same peaks, and for the large net the same list) with
    // ai: the outputs cleared first, and every word they wrote must be the same. Not the same: not used.
    auto nn = S->natives.find(std::string("front.twins.") + net + ".pipe");
    if (nn != S->natives.end()) {
      Arm nat{net, "int8 native", nn->second.pipe, nn->second.weights ? nn->second.weights : twin.buf, nullptr, ceilDiv(slots, nn->second.pipe.P), {}};
      nat.group = netGroup(ln, nat.pipe, *nat.buf, small);
      const uint64_t rb = ln.readingsBuf->size / ln.B * nb, lb = small ? ln.logitBuf->size / ln.B * nb : 0;
      auto keep = dev.createBuffer(2 * (rb + lb), wg::MAP_READ | wg::COPY_DST);
      auto e = dev.encoder();
      for (int k = 0; k < 2; k++) {
        const Arm& a = k ? nat : twin;
        e.clearBuffer(*ln.readingsBuf, 0, rb);
        if (small) e.clearBuffer(*ln.logitBuf, 0, lb);
        e.beginComputePass();
        e.setPipeline(*a.pipe.p);
        e.setBindGroup(*a.group);
        e.dispatch(a.x, (uint32_t)nb);
        e.endPass();
        e.copyBufferToBuffer(*ln.readingsBuf, 0, *keep, k * (rb + lb), rb);
        if (small) e.copyBufferToBuffer(*ln.logitBuf, 0, *keep, k * (rb + lb) + rb, lb);
      }
      dev.submit(e)->wait();
      const uint8_t* got = dev.read(*keep, 0, keep->size);
      const bool same = !memcmp(got, got + rb + lb, rb + lb);
      log(std::string("native net front.twins.") + net + (same ? ": the int8 twin's outputs on " + std::to_string(nb) + " frames, bit for bit" : ": not the int8 twin's outputs on this device, not used"));
      if (same) arms.push_back(nat);
    }
  }
  const uint32_t R = C.TWIN_ROUNDS, A = (uint32_t)arms.size();
  auto qs = dev.createQuerySet(2 * A * R);
  auto res = dev.createBuffer(16ull * A * R, wg::QUERY_RESOLVE | wg::COPY_SRC);
  auto rd = dev.createBuffer(res->size, wg::MAP_READ | wg::COPY_DST);
  auto e = dev.encoder();
  for (uint32_t r = 0; r < R; r++) {
    for (uint32_t j = 0; j < A; j++) {
      const uint32_t i = r % 2 ? A - 1 - j : j, k = r * A + i;
      e.beginComputePass(qs.get(), (int)(2 * k), (int)(2 * k + 1));
      e.setPipeline(*arms[i].pipe.p);
      e.setBindGroup(*arms[i].group);
      e.dispatch(arms[i].x, (uint32_t)nb);
      e.endPass();
    }
  }
  e.resolveQuerySet(*qs, 0, qs->count, *res, 0);
  e.copyBufferToBuffer(*res, 0, *rd, 0, res->size);
  dev.submit(e)->wait();
  const uint64_t* t = (const uint64_t*)dev.read(*rd, 0, rd->size);
  for (uint32_t r = 1; r < R; r++) for (uint32_t i = 0; i < A; i++) { const uint32_t k = r * A + i; arms[i].ms.push_back((double)((t[2 * k + 1] - t[2 * k]) & dev.timestampMask) * dev.timestampPeriod / 1e6); }
  auto best = [&](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()) / nb; };
  json out = {{"frames", nb}, {"by", "timestamps"}};
  for (const char* net : {"small", "large"}) {
    // ai: the form in use first, then its twins: the fastest is kept (the one in use on a tie)
    Arm* fl = nullptr; Arm* keep = nullptr;
    json row = json::object();
    for (auto& a : arms) if (a.net == net) {
      if (!fl) fl = keep = &a;
      else if (best(a.ms) < best(keep->ms)) keep = &a;
      row[a.precision] = best(a.ms);
    }
    if (!fl) continue;
    row["kept"] = keep->precision;
    out[net] = row;
    if (keep != fl) {
      if (std::string(net) == "small") { classify0 = keep->pipe; weightsBuf0 = keep->buf; cascade["precision"] = "int8"; }
      else { classify = keep->pipe; weightsBuf = keep->buf; S->tree["weightsPrecision"] = "int8"; }
    }
    nets[net] = keep->precision;
  }
  twins = out;
  for (auto& l : lanes) bindNets(*l);
  log("int8 twins measured on " + std::to_string(nb) + " frames: " + out.dump());
  log("nets: bank " + nets["bank"].dump() + ", small " + nets["small"].dump() + ", large " + nets["large"].dump());
}

json FrontHalf::report() const {
  return {{"size", bt.size}, {"ceiling", bt.ceiling}, {"Bmax", bt.Bmax}, {"lanes", bt.lanes}, {"inflight", bt.inflight}, {"bound", bt.bound}, {"perFrame", bt.perFrame}, {"capMs", bt.capMs}, {"budget", bt.budget}, {"planMs", bt.planMs}};
}

// ai: decoder.mjs Batcher
void Batcher::log(const std::string& m) {
  events.push_back(m);
  if (events.size() > 16) events.pop_front();
  fh->log("batch auto: " + m);
}

std::pair<int, std::string> Batcher::limitFor(int n) const {
  const double slot = slotBytes;
  char m[400];
  int Bm = (int)std::floor((room - fh->C.RING_SLACK * slot) / (n * perFrame + slot));
  snprintf(m, sizeof m, "memory: %d lanes of %d and a ring of %d fill the %.1f MB budget (a frame %.1f MB a lane and %.1f MB on the ring, %.1f MB shared)", n, Bm, Bm + (int)fh->C.RING_SLACK, budget / MiB, perFrame / MiB, slot / MiB, shared / MiB);
  std::string why = m;
  if (cap < Bm) { Bm = cap; why = capWhy; }
  if (Bm > (int)fh->C.B_CEIL) { Bm = (int)fh->C.B_CEIL; why = "the ceiling of " + std::to_string(fh->C.B_CEIL) + " a batch (past it there is nothing left to amortise)"; }
  if (Bm < 1) { Bm = 1; why = "memory: batches of 1"; }
  // ai: the setup holds back halves at gen's BACK_B alone (32, 16, 8, 4, 2, 1; the web builds one at any size): the
  // ai: largest of them under the limit (lavapipe at 2,100 px allowed 28, and planned nothing, 2026-10-04)
  const int limit = Bm;
  while (Bm > 1 && !fh->S->tree["backs"].contains(std::to_string(Bm))) Bm--;
  if (Bm < limit) why += "; the setup's back halves round it to " + std::to_string(Bm);
  return {Bm, why};
}

void Batcher::laned(int n, int B, const std::string& why) {
  lanes = n; Bmax = B; BmaxWhy = why;
  if (size > Bmax) { size = Bmax; bound = why; }
  if (ceiling > Bmax) ceiling = Bmax;
}

void Batcher::start() {
  const int prev = size;
  const int best = restart ? std::min(restart, Bmax) : prev > 0 ? std::min(prev, Bmax) : Bmax;
  bound = restart ? "restarted at " + std::to_string(best) + " after the device was lost" : best < Bmax ? std::to_string(best) + ", the size the watch had come to before this plan" : BmaxWhy;
  size = best;
  ceiling = restart ? best : Bmax;
  over = 0; spare = 0;
  char m[300];
  snprintf(m, sizeof m, "batch auto: %d frames a batch (lanes of %d, %.0f MB each; %s)", best, Bmax, perFrame * Bmax / MiB, bound.c_str());
  fh->log(m);
}

void Batcher::resize(int n, const std::string& why) {
  if (n == size) return;
  log(std::to_string(size) + " to " + std::to_string(n) + " frames: " + why);
  size = n;
  over = 0; spare = 0;
}

void Batcher::observe(int frames, double ms, bool first) {
  if (!size) return;
  const int s = size;
  char m[200];
  if (ms > capMs) { snprintf(m, sizeof m, "a batch of %d took %.0f ms, over the %.0f ms cap", frames, ms, capMs); resize(std::max(1, s >> 1), m); }
  else if (first) return;
  else if (ms > 0.9 * capMs && frames >= s) { if (++over >= 3) { snprintf(m, sizeof m, "three batches running over 90%% of the %.0f ms cap (%.0f ms)", capMs, ms); resize(std::max(1, s >> 1), m); } }
  else over = 0;
  if (size == s && s < ceiling && frames >= s && ms * 2 <= capMs) {
    if (++spare >= 16) { snprintf(m, sizeof m, "sixteen batches running with room for a double (%.0f ms)", ms); resize(std::min(ceiling, 2 * s), m); }
  } else spare = 0;
}

}  // namespace lizard
