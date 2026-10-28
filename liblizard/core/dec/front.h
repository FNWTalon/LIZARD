// ai: The GPU decoder's host, ported from the web's (liblizard/gpu/decoder.mjs FrontHalf and Batcher; the back half's
// ai: lanes and dispatches from liblizard/gpu/back/{back,transform,soft,ldpc}.mjs; the proposer's from
// ai: liblizard/gpu/bank_fcn2.mjs). What create() builds comes from the setup (setup.h); what a lane holds and what a
// ai: batch records are ported here, name for name, so the two can be read side by side.
// ai: Not ported: the probe, F9 and the grid (the fused pass 1 samples the frame), told frames (maps: the receiver is
// ai: blind), the wasm back half.
//
// ai: One thread drives it (the decoder's). A batch is recorded and submitted by run(), and its readback parsed by
// ai: finish() once its ticket is done, so two lanes can be in flight as on the web.
#pragma once
#include "setup.h"
#include "develop.h"

#include <cmath>
#include <array>
#include <chrono>
#include <mutex>
#include <optional>

struct AHardwareBuffer;

namespace lizard {

// ai: The constants the web host states (gen.mjs exports them with the setup).
struct Consts {
  uint32_t VOTE_READS, LEVELS, COUNTERS, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, RING_SLACK, K, CELL, HYP, RING_COUNT, NODES_MAX, RESID, REFIT_DBG, HIST_BINS, PYRAMID_BLOCK;
  float TAU, EPS;
  uint32_t TILE, B_CEIL;
  double CAP_MS;
  uint32_t TWIN_FRAMES, TWIN_ROUNDS, VERSION_MIN, VERSION_MAX, SLOTS, ARGS_WORDS, argsWords, COUNTS, REC_BYTES, PAYLOAD;
  // ai: the anchors' (gpu/wgsl/anchors.mjs): anchors a frame, and vec4f a frame of its record
  uint32_t ANCH_K = 0, ANCH_REC = 0;
  std::map<std::string, uint32_t> ARGS_SLOT;
  static Consts of(const json& c);
};

using BufP = std::shared_ptr<wg::Buffer>;
using GroupP = std::shared_ptr<wg::BindGroup>;

// ai: The back half's lane (back.mjs lane: transform.lane, soft.lane, ldpc.lane, bindGate; bindPicture).
struct BackLane {
  BufP lists, args, part, coef, y, s, counts, L, BLK, V, ITS, REC, PILOT;
  std::vector<BufP> params;   // ai: a size slot each (null where the size is not built)
  GroupP gate, ldpc;
  std::vector<GroupP> p1f, red, p2, soft;
  // ai: the format's rate profile's stage (gpu/back/tiers.mjs exportFor): its lane buffer and its steps' groups, in the
  // ai: tree's order (the soft stage's and the LDPC's in their places)
  BufP EP;
  std::vector<GroupP> rsoft, rldpc;
  const wg::Texture* picture = nullptr;   // ai: the texture p1f's groups are over
  uint64_t bytes = 0;
};

struct Lane {
  int B = 0, slotsG = 0;
  BufP dumpBuf;   // ai: LIZ_DUMP_S's copy of the back half's spectrum, made on the first batch that dumps
  bool busy = false, watched = false;
  uint32_t W = 0, H = 0;
  BufP peaksBuf, selUni, framesBuf, mapsBuf, countsBuf, readingsBuf, candBuf, quadsBuf, scoreBuf, selBuf, resultBuf, heldUni, p5Uni, ldUni, measBuf, residBuf, refitDbg, histBuf;
  GroupP refitGroup, refitPostGroup, pickGroup;   // ai: refitPostGroup: the tree's front.refitPost
  // ai: the anchors (SPEC 6.13): ANCH_FIND's records (a vec4f an anchor) and ANCH_FIT's (ANCH_REC vec4f a frame,
  // ai: which pass 1 reads), no clear (every one a batch reads is written in it); their groups bind the texture
  // ai: (ensure's); ANCH_REFIT's does not (made with the lane)
  BufP anchBuf, arecBuf;
  GroupP anchFindGroup, anchFitGroup, anchRefitGroup, anchNodesGroup;
  // ai: the finder's second round (decoder.mjs, gpu/wgsl/finder.mjs REDO): its frame records and its groups over them
  BufP frames2Buf;
  GroupP pickGroup2, redoGroup, selGroup2, classifyGroup2, scoreGroup2, voteGroup2, gatherGroup2, completeGroup2;
  std::shared_ptr<wg::QuerySet> qs, qsProfile;
  BufP qsBuf, qsProfileBuf;
  // ai: ensure()'s, at the lane's W x H
  std::shared_ptr<wg::Texture> tex;
  struct Level { uint32_t w, h; BufP buf; };
  std::vector<Level> level;
  uint32_t slots = 0, gx = 0, gy = 0;
  BufP rawBuf, accBuf;
  GroupP selGroup, classifyGroup, scoreGroup, voteGroup, gatherGroup, wordGroup, pyrGroup;
  GroupP nodesGroup;
  struct Fcn2Level { uint32_t l, w, h; GroupP group, xgroup, group2, xgroup2; BufP plane, uni; };   // ai: group2, xgroup2: a level below coarseFrom's
  std::vector<Fcn2Level> fcn2Levels;
  BufP readBuf;
  BufP keepBuf;   // ai: run's keep: the kept frames' bytes, read back with the batch; let go by a batch that keeps none
  std::unique_ptr<BackLane> back;
  uint64_t laneBytes = 0;
  std::vector<wg::Resource> lvEntries(bool second = false) const;
};

// ai: A frame staged on the ring (decoder.mjs Slot): layer `id` holds its w x h at the origin until a batch copies it.
struct Slot {
  int id = -1;
  uint32_t w = 0, h = 0;
  bool luma = false;      // ai: the host-luma path: the frame is in the slot's staging buffer, and its batch copies it into the layer
  bool raw = false;       // ai: the develop path (enqueueRaw): the crop's samples in the slot's raw staging, which its batch develops and copies into the layer
  // ai: the batch copy that last read the layer: the slot is not written again until it is done (Vulkan orders
  // ai: nothing between the ingest's queue and the decoder's; the web's one queue ordered the next write for free)
  std::shared_ptr<wg::Ticket> reading;
  uint64_t ready = 0;     // ai: the camera ingest's timeline value (0: a luma upload, ordered on the queue)
  uint64_t tag = 0;       // ai: the caller's
  // ai: where the camera's optical axis falls in the frame, axisWord's (0: its centre); enqueue sets 0, the caller
  // ai: after it (core/rx/receiver.cpp: each crop's)
  uint32_t axis = 0;
};

// ai: The axis word (gpu/wgsl/common.mjs axisWord): x and y in half pixels, each + 32768 in 16 bits, x low; 0 the
// ai: frame's centre.
inline uint32_t axisWord(double x, double y) {
  auto h = [](double v) { const double t = std::floor(2 * v + 0.5) + 32768; return (uint32_t)(t < 0 ? 0 : t > 65535 ? 65535 : t); };
  return h(x) | (h(y) << 16);
}

struct Record { int frame, block, its; uint32_t id; std::array<uint8_t, 473> payload; };

struct FrameOut {
  bool empty = false;
  uint64_t tag = 0;   // ai: the caller's, as enqueued
  int ring = -1, version = 0, n = 0, size = -1;
  bool held = false;
  struct { float found = 0, ring = -1; std::array<float, 8> corners{}; } finder;
  bool hasWord = false;
  struct { int version = 0, fps = 0; } word;
  int anchors = 0;   // ai: result word 31: the anchors ANCH_FIT found on the frame (0: the frame not found)
  std::vector<uint32_t> counts;       // ai: COUNTERS a frame
  std::vector<uint32_t> backCounts;   // ai: the back half's COUNTS a frame
  std::vector<Record> records;        // ai: the frame's verified blocks
  // ai: the pilots (SPEC 7.3; decoder.mjs pilotOf): the mean of the soft stage's per-block r over the even blocks the
  // ai: frame's version carries (pilotR: bit 0 of the painted count) and over the odd (pilotR2: bit 1), with their
  // ai: standard errors, where the soft stage ran on it (pilotBlocks 0: none; NaN where no block of a parity was read)
  int pilotBlocks = 0;
  float pilotR = 0, pilotSd = 0, pilotR2 = 0, pilotSd2 = 0;
  // ai: the frame as the ring held it, w x h bytes in rows of w, where its batch kept it (run's keep: Save replays)
  std::vector<uint8_t> luma;
  uint32_t w = 0, h = 0;
  uint32_t axis = 0;   // ai: the slot's axis word (Slot::axis), for the replay's meta
};

struct BatchOut {
  std::vector<FrameOut> frames;
  std::optional<double> gpuMs;
  double deviceMs = 0, done = 0;
  int carried = 0;
  std::vector<std::pair<std::string, double>> stageMs;
};

class FrontHalf;

// ai: The batch controller behind B auto (decoder.mjs Batcher), unchanged in its rules: lanes sized by memory for
// ai: Bmax; the stream starts at Bmax; a batch over the cap halves the size, three over 90% halve it, sixteen with
// ai: room for a double grow it back toward Bmax; a lane's first batch is watched for the cap alone.
struct Batcher {
  FrontHalf* fh = nullptr;
  double budget = 0, capMs = 400;
  int size = 0, ceiling = 0, Bmax = 0, lanes = 2, inflight = 2;
  double perFrame = 0, shared = 0, room = 0, slotBytes = 0, planMs = 0;
  int cap = 0;
  std::string capWhy, bound, BmaxWhy;
  int over = 0, spare = 0;
  void log(const std::string& m);
  std::pair<int, std::string> limitFor(int n) const;
  void laned(int n, int B, const std::string& why);
  void start();
  void resize(int n, const std::string& why);
  void observe(int frames, double ms, bool first);
};

struct InFlight {
  Lane* lane = nullptr;
  std::shared_ptr<wg::Ticket> ticket;
  std::vector<Slot*> slots;   // ai: freed at submit already; kept for their shapes and tags
  std::vector<uint64_t> tags;
  std::vector<uint32_t> axes;   // ai: each slot's axis word, as the batch ran it
  int nb = 0;
  uint32_t W = 0, H = 0;
  uint64_t cntOff = 0, resOff = 0, backOff = 0, tsOff = 0, readBytes = 0;
  uint64_t dumpBytes = 0;   // ai: LIZ_DUMP_S: the spectrum's bytes this batch copied out, 0 when not dumping
  uint32_t nq = 0;
  std::vector<std::string> stages;   // ai: profiled: the stage a timestamp pair each
  double submitted = 0, t0 = 0;
  // ai: run's keep: the batch's first `kept` frames as the ring held them, copied out with the slots' copies into the
  // ai: lane's keepBuf (`keep`: each one's place and size), and the first where LIZ_DUMP wants it (dump)
  int kept = 0;
  bool dump = false;
  struct Kept { uint64_t off; uint32_t w, h; };
  std::vector<Kept> keep;
  std::shared_ptr<wg::Ticket> keepTicket;
};

class FrontHalf {
 public:
  // ai: variant: from variantsFor(); B auto always (the receiver's); budget: device memory the decoder may hold
  // ai: (the web's is maxBufferSize; here a share of the device's own memory, Decoder chooses).
  static std::unique_ptr<FrontHalf> create(wg::Device& d, ReadFile read, const std::string& variant, double budget, std::function<void(const std::string&)> log = nullptr);

  wg::Device& dev;
  std::unique_ptr<Setup> S;
  Consts C;
  Batcher bt;
  std::function<void(const std::string&)> log;
  std::string variant;
  json nets;
  int cap = 256, nodesMax = 0, bankKeep = 4;
  uint32_t coarseFrom = 0;   // ai: the tree's front.coarseFrom: the bank's first round from this level
  std::array<float, 4> modules{};

  // ai: the nets now in use (measureTwins may swap a classifier for its int8 twin)
  Pipe classify;
  BufP weightsBuf;
  json twinForms;   // ai: the twins still to measure (null once measured)
  // ai: the LDPC kernel's form in use (gpu/back/tiers.mjs exportFor: 0 the rule's "ldpc" steps, k its "ldpcAlts"[k - 1]),
  // ai: and whether the forms are still to be timed on the first batch
  int ldpcForm = 0;
  bool ldpcPending = false;
  // ai: The anchors (SPEC 6.13; gpu/wgsl/anchors.mjs, the tree's `anch`): ANCH_FIND, ANCH_PREDICT (on ANCH_REFIT's
  // ai: layout and group), the border's second search (anchNodes: F7's NODES centred, on its layout, its own round's
  // ai: group, anchNodesGroup), ANCH_REFIT and ANCH_FIT after F7's REFIT_POST in its refit's pass, the C's order; a
  // ai: batch with no F7 runs ANCH_FIND and ANCH_FIT alone. Pass 1 reads their records.
  Pipe anchFind, anchFit, anchRefit, anchPredict, anchNodes;
  BufP anchUni;
  // ai: The develop (enqueueRaw; the tree's spv/ingest_develop.spv, from core/ingest/develop.comp): its pipeline, its
  // ai: layout (the staged samples, the gamma table, the luma out) and the table (develop.h gammaLut)
  std::shared_ptr<wg::Pipeline> developPipe;
  std::shared_ptr<wg::BindGroupLayout> developBgl;
  BufP developLut;

  int B = 0;        // ai: the lanes' size
  std::vector<std::unique_ptr<Lane>> lanes;
  struct Planned { uint32_t W = 0, H = 0; };
  std::optional<Planned> planned;
  // ai: a slot's buffers by path: staging (enqueueLuma's bytes), raw (enqueueRaw's staged samples), developed (the
  // ai: develop's luma, device memory) and develop (the develop's bind group over raw, the table and developed)
  struct Ring { int R = 0; uint32_t W = 0, H = 0; std::shared_ptr<wg::Texture> tex; std::vector<Slot*> slots; std::vector<std::unique_ptr<Slot>> store; std::vector<BufP> staging, raw, developed; std::vector<std::shared_ptr<wg::BindGroup>> develop; int head = 0; };
  std::optional<Ring> ring;
  double lastDone = 0;
  double lastDump = 0;   // ai: LIZ_DUMP's last frame written (front.cpp run)
  bool keeping = false;  // ai: some lane holds a keepBuf (dropKeep)
  uint64_t backShared = 0;

  int size() const { return bt.size; }
  int inflightMax() const { return bt.inflight; }
  double now() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - epoch).count(); }

  void plan(uint32_t w, uint32_t h);
  // ai: A frame of luma onto the ring (the web's writeTexture path: recordings, the desktop): rows of w bytes,
  // ai: `stride` apart. Null when every slot is staged.
  Slot* enqueueLuma(const uint8_t* bytes, uint32_t w, uint32_t h, uint32_t stride, uint64_t tag = 0);
  // ai: A camera frame onto the ring with no copy (core/ingest/camera.h): its buffer's crop sampled into a free
  // ai: slot's layer on the ingest's queue; any thread (the ring is under ringMu). Null when every slot is taken.
  Slot* enqueueCamera(class CameraIngest& ci, AHardwareBuffer* hb, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag = 0);
  // ai: A frame in a format developed to luma (ingest/develop.h, 2026-10-27): the w x h crop at (x, y) of the
  // ai: developed frame, its samples staged here as they are (any thread, as enqueueLuma), developed on the device by
  // ai: the batch that takes the slot (develop.comp) and copied into its layer. Null when every slot is staged.
  Slot* enqueueRaw(const RawFrame& r, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag = 0);
  CameraIngest* ingest = nullptr;   // ai: the camera ingest whose timeline a batch waits on (set by the receiver)
  std::mutex ringMu;
  void release(Slot* s);
  Lane* freeLane();
  // ai: A batch of staged slots on a free lane: recorded and submitted. held: the held version (0 none).
  // ai: profile: a pass and a timestamp pair a stage group (decoder.mjs batch's profile: STAGES), so the batch's
  // ai: out.stageMs says which stage costs what; its gpuMs is their sum. keep: the batch's first `keep` frames read
  // ai: back as the ring holds them (FrameOut::luma), for a replay.
  std::unique_ptr<InFlight> run(const std::vector<Slot*>& slots, int held, bool profile = false, int keep = 0);
  // ai: The batch read back once its ticket is done (waits for it), the batcher told, the twins measured if due; a
  // ai: throw gives the lane back.
  BatchOut finish(std::unique_ptr<InFlight> f);
  // ai: the readback buffers of kept frames let go on every lane no batch is using (decoder thread)
  void dropKeep();

 private:
  explicit FrontHalf(wg::Device& d) : dev(d) {}
  std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
  Pipe P(const char* k) { return S->pair(S->tree["front"][k]); }
  wg::Buffer& FB(const char* k) { return S->buf(S->tree["front"]["buffers"][k]); }
  BufP FBP(const char* k) { return S->bufPtr(S->tree["front"]["buffers"][k]); }
  std::unique_ptr<Lane> lane(int B);
  void ensure(Lane& ln, uint32_t W, uint32_t H);
  void bindNets(Lane& ln);
  GroupP netGroup(Lane& ln, const Pipe& pipe, wg::Buffer& buf, bool second = false);
  void backLane(Lane& ln);
  void bindPicture(Lane& ln);
  void encodeBack(wg::Encoder& e, Lane& ln, int frames, const std::function<void(const char*)>& begin);
  void makeLanes(int n);
  void makeRing(int R, uint32_t W, uint32_t H);
  void destroyRing();
  int takeSlot();   // ai: under ringMu: a free slot whose last copy is done, from the head on; -1 none
  void buildBack(int B);
  void measureTwins(Lane& ln, int nb);
  void measureLdpc(Lane& ln, int nb);
  BatchOut finishBatch(std::unique_ptr<InFlight> f);
  uint64_t argsOffset(int s, const char* slot) const { return 4ull * (C.ARGS_WORDS * s + C.ARGS_SLOT.at(slot)); }
};

}  // namespace lizard
