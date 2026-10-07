// ai: The GPU decoder's host, ported from the web's (liblizard/gpu/decoder.mjs FrontHalf and Batcher; the back half's
// ai: lanes and dispatches from liblizard/gpu/back/{back,transform,soft,ldpc}.mjs; the proposer's from
// ai: liblizard/gpu/bank_fcn2.mjs). What create() builds comes from the setup (setup.h); what a lane holds and what a
// ai: batch records are ported here, name for name, so the two can be read side by side and their traces compared.
// ai: Not ported: the probe, F9 and the grid (the fused pass 1 samples the frame), the cancel stage (off by default on
// ai: the web), told frames (maps: the receiver is blind), the wasm back half.
//
// ai: One thread drives it (the decoder's). A batch is recorded and submitted by run(), and its readback parsed by
// ai: finish() once its ticket is done, so two lanes can be in flight as on the web.
#pragma once
#include "setup.h"

#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>

struct AHardwareBuffer;

namespace lizard {

// ai: The constants the web host states (gen.mjs exports them with the setup).
struct Consts {
  uint32_t LEVELS, COUNTERS, RESULT, SEL_WORDS, TS_MAX, TS_PLAIN, TS_INGEST, RING_SLACK, K, CELL, HYP, RING_COUNT, NODES_MAX, REFIT_DBG, HIST_BINS, PYRAMID_BLOCK;
  float TAU, EPS, TAU_TRACK;
  uint32_t TILE, KEEP, B_CEIL;
  double CAP_MS;
  uint32_t TWIN_FRAMES, TWIN_ROUNDS, VERSION_MAX, SLOTS, ARGS_WORDS, argsLdpc, argsWords, COUNTS, REC_BYTES, PAYLOAD;
  std::map<std::string, uint32_t> ARGS_SLOT;
  static Consts of(const json& c);
};

using BufP = std::shared_ptr<wg::Buffer>;
using GroupP = std::shared_ptr<wg::BindGroup>;

// ai: The back half's lane (back.mjs lane: transform.lane, soft.lane, ldpc.lane, bindGate; bindPicture).
struct BackLane {
  BufP lists, args, part, coef, y, s, counts, L, EST, BLK, V, ITS, REC, PILOT;
  std::vector<BufP> params;   // ai: a size slot each (null where the size is not built)
  GroupP gate, ldpc;
  std::vector<GroupP> p1f, red, p2, soft;
  // ai: a rate profile's (FrontHalf::tiersKey): its lane buffer and its steps' groups, in the tree's order
  BufP PS;
  std::vector<GroupP> tsoft, tldpc;
  const wg::Texture* picture = nullptr;   // ai: the texture p1f's groups are over
  uint64_t bytes = 0;
};

struct Lane {
  int B = 0, slotsG = 0;
  bool busy = false, watched = false, reserved = false;
  uint32_t W = 0, H = 0;
  uint64_t gridStride = 0;
  BufP peaksBuf, selUni, framesBuf, mapsBuf, countsBuf, readingsBuf, candBuf, quadsBuf, scoreBuf, selBuf, resultBuf, heldUni, p5Uni, ldUni, measBuf, residBuf, refitDbg, histBuf, logitBuf, listBuf;
  GroupP refitGroup, pickGroup;
  std::shared_ptr<wg::QuerySet> qs, qsProfile;
  BufP qsBuf, qsProfileBuf;
  // ai: ensure()'s, at the lane's W x H
  std::shared_ptr<wg::Texture> tex;
  struct Level { uint32_t w, h; BufP buf; };
  std::vector<Level> level;
  uint32_t slots = 0, gx = 0, gy = 0;
  BufP rawBuf, accBuf;
  GroupP selGroup, classifyGroup, classify0Group, rankGroup, scoreGroup, voteGroup, gatherGroup, wordGroup, pyrGroup;
  std::vector<GroupP> nodesGroups;
  struct Fcn2Level { uint32_t w, h; GroupP group, xgroup; BufP plane, uni; };
  std::vector<Fcn2Level> fcn2Levels;
  BufP readBuf;
  BufP keepBuf;   // ai: run's keep: the kept frames' bytes, read back with the batch; let go by a batch that keeps none
  std::unique_ptr<BackLane> back;
  uint64_t backBytes = 0;   // ai: the back lane's device bytes (plan prices them a frame of bh.B)
  uint64_t laneBytes = 0;
  std::vector<wg::Resource> lvEntries() const;
};

// ai: A frame staged on the ring (decoder.mjs Slot): layer `id` holds its w x h at the origin until a batch copies it.
struct Slot {
  int id = -1;
  uint32_t w = 0, h = 0;
  double at = 0;          // ai: ms on the decoder's clock at enqueue (the queue's launch bound runs from it)
  bool luma = false;      // ai: the host-luma path: the frame is in the slot's staging buffer, and its batch copies it into the layer
  // ai: the batch copy that last read the layer: the slot is not written again until it is done (Vulkan orders
  // ai: nothing between the ingest's queue and the decoder's; the web's one queue ordered the next write for free)
  std::shared_ptr<wg::Ticket> reading;
  uint64_t ready = 0;     // ai: the camera ingest's timeline value (0: a luma upload, ordered on the queue)
  uint64_t tag = 0;       // ai: the caller's
};

struct Record { int frame, block, its; uint32_t id; std::array<uint8_t, 473> payload; };

struct FrameOut {
  bool empty = false;
  uint64_t tag = 0;   // ai: the caller's, as enqueued
  int ring = -1, version = 0, n = 0, size = -1;
  bool held = false, sampled = false;
  struct { float found = 0, score = 0, ring = -1, orient = 0, quad = 0; std::array<float, 9> H{}; std::array<float, 8> corners{}; std::array<float, 4> depths{}; } finder;
  bool hasWord = false;
  struct { int version = 0, fps = 0; float score = 0; } word;
  std::vector<uint32_t> counts;       // ai: COUNTERS a frame
  std::vector<uint32_t> backCounts;   // ai: the back half's COUNTS a frame
  std::array<int, 5> verdicts{};      // ai: blocks a verdict: 0 not run, 1 verified, 2 declined, 3 stall or cap, 4 CRC
  std::vector<Record> records;        // ai: the frame's verified blocks
  std::vector<uint32_t> hist;         // ai: HIST_BINS
  // ai: the pilots (SPEC 7.3; decoder.mjs pilotOf): the mean of the soft stage's per-block r over the even blocks the
  // ai: frame's version carries (pilotR: bit 0 of the painted count) and over the odd (pilotR2: bit 1), with their
  // ai: standard errors, where the soft stage ran on it (pilotBlocks 0: none)
  int pilotBlocks = 0;
  float pilotR = 0, pilotSd = 0, pilotR2 = 0, pilotSd2 = 0;
  // ai: the frame as the ring held it, w x h bytes in rows of w, where its batch kept it (run's keep: Save replays)
  std::vector<uint8_t> luma;
  uint32_t w = 0, h = 0;
};

struct BatchOut {
  std::vector<FrameOut> frames;
  std::optional<double> gpuMs;
  double deviceMs = 0, submitted = 0, done = 0;
  int carried = 0;
  bool idle = false;
  std::map<std::string, double> host;
  std::vector<std::pair<std::string, double>> stageMs;
  double twinMs = -1;
};

class FrontHalf;

// ai: The batch controller behind B auto (decoder.mjs Batcher), unchanged in its rules: lanes sized by memory for
// ai: Bmax; the stream starts at Bmax; a batch over the cap halves the size, three over 90% halve it, sixteen with
// ai: room for a double grow it back toward Bmax; a lane's first batch is watched for the cap alone.
struct Batcher {
  FrontHalf* fh = nullptr;
  double budget = 0, capMs = 400;
  int restart = 0;
  int size = 0, ceiling = 0, Bmax = 0, lanes = 2, inflight = 2;
  double perFrame = 0, shared = 0, room = 0, slotBytes = 0, planMs = 0;
  int cap = 0;
  std::string capWhy, bound, BmaxWhy;
  int over = 0, spare = 0;
  std::deque<std::string> events;
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
  int nb = 0;
  uint32_t W = 0, H = 0;
  uint64_t cntOff = 0, resOff = 0, histOff = 0, backOff = 0, tsOff = 0, readBytes = 0;
  uint32_t nq = 0;
  std::vector<std::string> stages;   // ai: profiled: the stage a timestamp pair each
  double submitted = 0, t0 = 0;
  std::map<std::string, double> host;
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
  // ai: (the web's is maxBufferSize; here a share of the device's own memory, Decoder chooses); nmax: the largest
  // ai: picture a frame may name (1536, the ladder's top).
  static std::unique_ptr<FrontHalf> create(wg::Device& d, ReadFile read, const std::string& variant, double budget, int restart = 0, std::function<void(const std::string&)> log = nullptr);

  wg::Device& dev;
  std::unique_ptr<Setup> S;
  Consts C;
  Batcher bt;
  std::function<void(const std::string&)> log;
  std::string precision, floatPrecision, variant;
  // ai: The rate profile (the lab's rate-by-ring arm, 2026-10-07; gpu/back/tiers.mjs): LIZ_TIERS's key, "" none. Its
  // ai: frames are those at picture slot tslot whose version is tversion; they carry tblocks blocks.
  std::string tiersKey, tiersLabel;
  int tslot = -1, tversion = 0, tblocks = 0;
  const json* tierStage() const { return tiersKey.empty() ? nullptr : &S->back()["tiers"][tiersKey]; }
  json nets;
  int cap = 256, nodesMax = 0, bankKeep = 4;
  bool reg = true;
  std::array<float, 4> modules{};
  json cascade;   // ai: { keep, rest, precision } or null
  uint32_t nmaxFind = 1536;

  // ai: the nets now in use (measureTwins may swap a classifier for its int8 twin)
  Pipe classify, classify0;
  BufP weightsBuf, weightsBuf0;
  json twinForms;   // ai: the twins still to measure (null once measured)
  json twins;       // ai: the measurement

  int B = 0;        // ai: the lanes' size
  std::vector<std::unique_ptr<Lane>> lanes;
  struct Planned { uint32_t W = 0, H = 0, nmax = 0; };
  std::optional<Planned> planned;
  struct Ring { int R = 0; uint32_t W = 0, H = 0; std::shared_ptr<wg::Texture> tex; std::vector<Slot*> slots; std::vector<std::unique_ptr<Slot>> store; std::vector<BufP> staging; int head = 0, staged = 0; };
  std::optional<Ring> ring;
  double lastDone = 0;
  double lastDump = 0;   // ai: LIZ_DUMP's last frame written (front.cpp run)
  bool keeping = false;  // ai: some lane holds a keepBuf (dropKeep)
  uint64_t backShared = 0;

  int size() const { return bt.size; }
  int inflightMax() const { return bt.inflight; }
  double now() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - epoch).count(); }

  bool needsPlan(uint32_t w, uint32_t h) const;
  void plan(uint32_t w, uint32_t h);
  // ai: A frame of luma onto the ring (the web's writeTexture path: recordings, the desktop): rows of w bytes,
  // ai: `stride` apart. Null when every slot is staged.
  Slot* enqueueLuma(const uint8_t* bytes, uint32_t w, uint32_t h, uint32_t stride, uint64_t tag = 0);
  // ai: A camera frame onto the ring with no copy (core/ingest/camera.h): its buffer's crop sampled into a free
  // ai: slot's layer on the ingest's queue; any thread (the ring is under ringMu). Null when every slot is taken.
  Slot* enqueueCamera(class CameraIngest& ci, AHardwareBuffer* hb, uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint64_t tag = 0);
  CameraIngest* ingest = nullptr;   // ai: the camera ingest whose timeline a batch waits on (set by the receiver)
  std::mutex ringMu;
  int staged() { std::lock_guard<std::mutex> l(ringMu); return ring ? ring->staged : 0; }
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
  json report() const;

 private:
  explicit FrontHalf(wg::Device& d) : dev(d) {}
  std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
  Pipe P(const char* k) { return S->pair(S->tree["front"][k]); }
  wg::Buffer& FB(const char* k) { return S->buf(S->tree["front"]["buffers"][k]); }
  BufP FBP(const char* k) { return S->bufPtr(S->tree["front"]["buffers"][k]); }
  std::unique_ptr<Lane> lane(int B);
  void ensure(Lane& ln, uint32_t W, uint32_t H, uint32_t nmax);
  void bindNets(Lane& ln);
  GroupP netGroup(Lane& ln, const Pipe& pipe, wg::Buffer& buf, bool small);
  void backLane(Lane& ln);
  void bindPicture(Lane& ln);
  void encodeBack(wg::Encoder& e, Lane& ln, int frames, const std::function<void(const char*)>& begin);
  void makeLanes(int n);
  void makeRing(int R, uint32_t W, uint32_t H);
  void destroyRing();
  int takeSlot();   // ai: under ringMu: a free slot whose last copy is done, from the head on; -1 none
  void buildBack(int B);
  void measureTwins(Lane& ln, int nb);
  BatchOut finishBatch(std::unique_ptr<InFlight> f);
  uint64_t argsOffset(int s, const char* slot) const { return 4ull * (C.ARGS_WORDS * s + C.ARGS_SLOT.at(slot)); }
};

}  // namespace lizard
