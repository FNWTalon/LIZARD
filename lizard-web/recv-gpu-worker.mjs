// ai: The receiver's GPU decode worker: the whole decode on the GPU (gpu/decoder.mjs through gpuqueue.mjs), told nothing
// ai: (the ring found, each frame's version and picture from its own word, F8, or from the last word read, which
// ai: the worker holds for the queue's next batch: the held configuration, done below), and answering every frame with
// ai: recv-worker.mjs's own result message, so the page's dedupe, fountain and tracking take either without
// ai: knowing which.
// ai:   page to worker: { type: "config", config } ({ version, spec: sim/phy.mjs blindSpec }; any other spec is answered
// ai:     unavailable); { type: "frame", tag, w, h, luma | rgba } for each grabbed frame, the buffer transferred (the
// ai:     other fields a CPU worker gets are ignored); or, F0 (lizard-web/vframes.mjs), { type: "frame", tag, frame, x,
// ai:     y, w, h }: a VideoFrame, transferred, and the crop the page would have grabbed, which the decoder takes to luma
// ai:     on the device.
// ai:     Or the camera track itself (Chrome): { type: "track", track, crop: { x, y, w, h }, fps, pageTimeOrigin }, a
// ai:     clone transferred, read here by a MediaStreamTrackProcessor, every frame queued with the latest crop
// ai:     ({ type: "crop", x, y, w, h } moves it) and tagged on the page's clock; { type: "track", track: null } ends it.
// ai:   worker to page: { type: "ready", version, frames, framesWhy, processor, gpu: { adapter, precision, nets, bank, B, inflight } }
// ai:     (nets: the arithmetic each net runs in, gpu/decoder.mjs fh.nets { bank, small, large })
// ai:     (frames: this device imports a VideoFrame; framesWhy: why not; processor: a track can be read here) or
// ai:     { type: "unavailable", reason } per config; { type: "source", kind: "worker-track" } once the track's first
// ai:     frame is read, or { type: "source", kind: "page", why } when the track gives none in SOURCE_WAIT, fails, or
// ai:     ?gsrc=page declines it (the page keeps its tap); one { type: "result", ... } per frame (a track frame's with
// ai:     box: { x, y, w, h }, the crop used; with gained, the blocks among its own that straddle cancellation, pass two,
// ai:     verified, where it verified any); { type: "stats",
// ai:     ms, lag, B, inflight, queued, held, dropped, f0, vf, luma, levelsFrom, batches, carried, host, submitToMap,
// ai:     mapLatency, doneMs, ingestMs, arrived, gapMax, openMax, cancelled, gained } at most once a second
// ai:     (vf and luma: { n, found, blocks } of the window's VideoFrame frames and its luma frames; cancelled and
// ai:     gained: the window's frames pass two ran on with a solved fit, and the blocks it verified, sums a second;
// ai:     levelsFrom: where the last grey levels reported came from, "gpu" for a VideoFrame's histogram read back with
// ai:     its batch, "luma" for a luma frame measured here, "" before any; the rest: where a batch's wall time goes,
// ai:     at WIN0 below);
// ai:     { type: "lost", reason } once it gives up; { type: "error", tag, message } as the CPU worker's. A frame not
// ai:     decoded (dropped from the queue, or no decoder yet) is answered with the CPU worker's no-decoder result plus
// ai:     dropped: true.
// ai: ?pref=high-performance|low-power on the worker's URL asks for that adapter (a machine with two GPUs); ?cancel=1
// ai: puts straddle cancellation in the decoder (the receiver's ?gcancel=on; off by default).
import { blockJudge } from "../liblizard/sim/phy.mjs";
import { init as initOb, heapTop } from "../liblizard/sim/ob.mjs";
import { PAYLOAD } from "../liblizard/sim/xfer.mjs";
import { openGpu, GpuQueue, Unavailable, unsupported, quadOf } from "./gpuqueue.mjs";
import { importsWhyNot, LUMA_EVERY } from "./vframes.mjs";
import { Ingest } from "../liblizard/gpu/ingest.mjs";

const Q = new URLSearchParams(globalThis.location?.search ?? "");
const PREF = Q.get("pref");
// ai: Diagnostic switches (?prec=int8|f16|f32, that precision alone; ?sg=0) for a device whose int8, f16 or subgroup
// ai: arithmetic is suspect; ?cancel=1 puts straddle cancellation (pass two, gpu/back/cancel.mjs) in the decoder (the
// ai: receiver's ?gcancel=on).
const OPEN = { pref: PREF, prec: Q.get("prec"), subgroups: Q.get("sg") !== "0", test: Q.get("test"), cancel: Q.get("cancel") === "1" };
// ai: ?stages=1 (the receiver's ?gstages=on): every batch profiled a pass a stage, its stage ms a frame in
// ai: the stats (stageMs). Off by default: the profile splits the one compute pass, so it is not the default's timing.
const STAGES = Q.get("stages") === "1";
// ai: ?gsrc=page (passed on from the receiver's own): the page's tap feeds this worker, and a track sent anyway is declined.
const PAGE_TAP = Q.get("gsrc") === "page";
const SOURCE_WAIT = 3000;    // ai: ms a track may give no frame after arriving before the page is told to keep its tap
const log = (m) => console.info(`gpu worker: ${m}`);

// ai: ctx: what a config means for a frame's answer ({ cfg, judge }), carried on each frame, since a batch may be read
// ai: back after the next config. queue and key: the decoder and the spec it was opened for.
// ai: framesWhy: why that decoder's device takes no VideoFrame (vframes.mjs importsWhyNot), null when it does.
let ctx = null, queue = null, key = null, latest = null, configs = Promise.resolve(), framesWhy = "no decoder";
let dropped = 0, sinceLevels = LUMA_EVERY - 1;
// ai: twinMs: the int8 twins' measurement on this decoder's first batch, ms (gpu/decoder.mjs batch() out.twinMs), null before.
let twinMs = null;

// ai: The spec as the decoder takes it; a new config that leaves it alone keeps the decoder and its batch size.
const keyOf = (s) => JSON.stringify(s);

async function configure(c) {
  if (c !== latest) return;   // ai: a newer config is queued behind this one
  try {
    if (!c.spec?.blind) throw new Unavailable(unsupported(c.spec) ?? "a told LIZARD spec: the receiver decodes blind (sim/phy.mjs blindSpec)");
    const k = keyOf(c.spec);
    if (k !== key) {
      const old = queue;
      queue = null; key = null;
      await endQueue(old);
      const t0 = performance.now();
      const dec = await openGpu(c.spec, { ...OPEN, log });
      const t1 = performance.now();
      const why = await importsWhyNot(dec.fh.device, dec.adapter).catch((e) => `the import probe failed: ${e?.message ?? e}`);
      // ai: openGpu's phases and the import probe's, ms, for the page's stats row (gpuBoot).
      dec.boot = { ...dec.boot, importProbe: Math.round(performance.now() - t1), open: Math.round(t1 - t0) };
      if (c !== latest) { dec.fh.device.destroy(); return; }
      const q = new GpuQueue(dec, { reopen: (restart) => openGpu(c.spec, { ...OPEN, log, restart }), done, drop, lost: (reason) => lost(q, reason), profile: STAGES });
      queue = q; key = k; framesWhy = why; twinMs = null;
      if (trackFps) q.setRate(trackFps);   // ai: the track outlives the decoder: a new queue is told the camera's rate again
    }
    // ai: The judge (sim/phy.mjs blockJudge) judges from the light, the test stream's bytes made in the codec's wasm; it
    // ai: keeps nothing between frames (the page says which blocks are new).
    const tj = performance.now();
    await initOb();
    const judge = blockJudge(PAYLOAD);
    ctx = { cfg: c, judge };
    const fh = queue.fh;
    postMessage({ type: "ready", version: c.version, frames: framesWhy === null, framesWhy, processor: typeof MediaStreamTrackProcessor === "function", gpu: { adapter: queue.dec.adapter, precision: fh.precision, nets: queue.dec.nets, bank: fh.bankName, B: fh.auto ? "auto" : fh.B, inflight: fh.inflight, boot: { ...queue.dec.boot, judge: Math.round(performance.now() - tj) } } });
  } catch (e) {
    const old = queue;
    queue = null; key = null; ctx = null;
    await endQueue(old);
    if (c === latest) postMessage({ type: "unavailable", reason: e instanceof Unavailable ? e.message : `the GPU worker failed: ${e?.message ?? e}` });
  }
}

// ai: A queue stopped: its computing batches delivered, or the device gone.
async function endQueue(q) { await q?.stop(); }

// ai: A frame's arrival, counted for the stats window with the largest gap since the one before (kept across windows,
// ai: so a pause between two sessions shows once, in the first window after it).
let lastArrive = 0;
function arrival() {
  const now = performance.now();
  win.arrived++;
  if (lastArrive) win.gapMax = Math.max(win.gapMax, now - lastArrive);
  lastArrive = now;
  return now;
}

function frame(m) {
  if (m.frame) return videoFrame(m);
  const { tag, w, h } = m, isRgba = m.rgba !== undefined, px = new Uint8Array(isRgba ? m.rgba : m.luma);
  const luma = isRgba ? toLuma(px, w * h) : px;
  const item = { tag, w, h, luma, ctx, at: arrival(), levels: levelsOf(px, w, h, isRgba), levelsFrom: "luma" };
  // ai: A frame whose bytes are not w x h would fail its whole batch's upload, and a failed batch ends the GPU path.
  if (queue && ctx && w > 0 && h > 0 && px.length === (isRgba ? 4 : 1) * w * h) queue.push(item);
  else answerEmpty(item);
  stats();
}

// ai: F0 from the page: a VideoFrame the page's tap took, with the crop it would have grabbed.
function videoFrame(m) {
  const { tag, w, h, x, y, frame: source } = m;
  return pushVideoFrame({ tag, w, h, x, y, source, f0: true, ctx, at: arrival(), levels: null, levelsFrom: "gpu", box: null });
}

// ai: A VideoFrame into the queue, from the page or off the track here. The queue closes it once the decoder has it,
// ai: or once it is dropped; one that never reaches the queue is closed here. A crop outside its frame would fail the
// ai: whole batch, ending the GPU path, so it is checked first (Ingest.check, the decoder's own test). Its luma never
// ai: reaches this worker, so its grey levels come back with its batch (done: the decoder's histogram of the layer).
function pushVideoFrame(item) {
  const { source } = item;
  sinceLevels++;
  let ok = !!(queue && ctx) && framesWhy === null;
  if (ok) try { Ingest.check(item); } catch { ok = false; }
  if (ok) queue.push(item);
  else { source.close(); answerEmpty(item); }
  stats();
}

// ai: The camera track in the worker: the page's tap took only the newest frame between two callbacks and lost one each
// ai: time two came, where a reader here sees every frame the camera hands over. trk: the track in hand, its reader, the
// ai: crop the page last posted, the previous frame's arrival on the page's clock (the next frame's tag) and the timer
// ai: for a track that gives no frame; trackFps: the camera's rate the page sent, for the queue's launch bound.
let trk = null, trackFps = 0;
// ai: Now on the page's clock: both clocks count from their own timeOrigin on the same monotonic time.
const pageNow = (t) => performance.timeOrigin + performance.now() - t.pageTimeOrigin;

function trackMsg(m) {
  stopTrack();
  if (!m.track) return;
  if (m.fps > 0) { trackFps = m.fps; queue?.setRate(trackFps); }
  const t = { track: m.track, crop: m.crop, pageTimeOrigin: m.pageTimeOrigin, prev: 0, reader: null, timer: 0 };
  if (PAGE_TAP) return sourcePage(t, "the menu keeps the page's tap");
  try { t.reader = new MediaStreamTrackProcessor({ track: t.track }).readable.getReader(); }
  catch (e) { return sourcePage(t, `no track processor here: ${e?.message ?? e}`); }
  trk = t;
  t.timer = setTimeout(() => sourcePage(t, `no frame off the track in ${SOURCE_WAIT} ms`), SOURCE_WAIT);
  readTrack(t);
}

// ai: The page keeps its tap: the clone is stopped (the page's own track goes on feeding its preview) and the reader,
// ai: if one ran, cancelled.
function sourcePage(t, why) {
  if (t === trk) stopTrack(); else t.track.stop();
  postMessage({ type: "source", kind: "page", why });
}

function stopTrack() {
  const t = trk;
  if (!t) return;
  trk = null;
  clearTimeout(t.timer);
  t.reader.cancel().catch(() => {});
  t.track.stop();
}

async function readTrack(t) {
  for (;;) {
    let r;
    try { r = await t.reader.read(); } catch (e) { if (t === trk) sourcePage(t, `the track's reader failed: ${e?.message ?? e}`); return; }
    if (t !== trk) { r.value?.close(); return; }
    if (r.done) return sourcePage(t, "the track ended");
    // ai: The first frame says whether the page's crop fits the track's frames (a VideoFrame need not be the <video>'s
    // ai: size); one that does not hands the page back its tap, which holds the frame to the video's size and grabs.
    if (!t.prev) {
      try { Ingest.check({ source: r.value, ...t.crop }); } catch (e) { r.value.close(); return sourcePage(t, `${e.message} (the track's frame)`); }
      clearTimeout(t.timer);
      postMessage({ type: "source", kind: "worker-track" });
    }
    await trackFrame(t, r.value);
  }
}

// ai: A track frame into the queue with the crop the page last posted, tagged with the previous frame's arrival on the
// ai: page's clock (the page tags its own frames with the callback before, so tFirst and the tags' order hold), the crop
// ai: echoed as box for the page's tracker.
function trackFrame(t, source) {
  const now = pageNow(t), tag = t.prev || now - 33;
  t.prev = now;
  const { x, y, w, h } = t.crop;
  return pushVideoFrame({ tag, w, h, x, y, source, f0: true, ctx, at: arrival(), levels: null, levelsFrom: "gpu", box: { x, y, w, h } });
}

// ai: The canvas grab's RGBA as luma, with the arithmetic the codec uses (src/acquire.c).
function toLuma(px, n) {
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) out[i] = (77 * px[j] + 150 * px[j + 1] + 29 * px[j + 2] + 128) >> 8;
  return out;
}

// ai: recv-worker.mjs's grey range, the middle half of the frame as it arrived, from the first luma frame once
// ai: LUMA_EVERY frames of either kind have come since the last (a page that sends luma).
function levelsOf(px, w, h, isRgba) {
  if (++sinceLevels < LUMA_EVERY) return null;
  sinceLevels = 0;
  const hist = new Uint32Array(256), step = isRgba ? 4 : 1, x0 = w >> 2, x1 = (3 * w) >> 2, y0 = h >> 2, y1 = (3 * h) >> 2;
  for (let y = y0; y < y1; y += 2) for (let x = x0; x < x1; x += 2) hist[px[(y * w + x) * step]]++;
  return levelsFromHist(hist);
}

// ai: The page's grey fields from a 256-bin histogram (levelsOf's, or the decoder's of the same pixels on the device,
// ai: gpu/wgsl/pyramid.mjs): the p5..p95 range, the median, the shares pinned at 255 and at 0. Null when empty.
function levelsFromHist(hist) {
  let n = 0;
  for (let v = 0; v < 256; v++) n += hist[v];
  if (!n) return null;
  const at = (p) => { let acc = 0; for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * p) return v; } return 255; };
  return { range: at(0.95) - at(0.05), mid: at(0.5), hi: hist[255] / n, lo: hist[0] / n };
}

// ai: A frame answered without a decode (dropped from the queue, or no decoder yet): the CPU worker's answer when it has
// ai: no decoder, marked so the page counts it as skipped, not as a frame that read nothing.
function answerEmpty(item) {
  dropped++;
  post(item, { type: "result", tag: item.tag, found: 0, seen: 0, bad: 0, ms: 0, ids: [], dropped: true });
}

// ai: A frame's verified blocks (the back half's records, at whatever size and version the frame was read), laid out as
// ai: the codec lays a frame's blocks (the id first, as the payload carries it), through the config's judge (sim/phy.mjs
// ai: blockJudge, the CPU worker's): every block it does not call bad, repeats included, with its bytes, and bad.
function tally(c, records, found) {
  const BB = PAYLOAD + 4, T = found ? records.reduce((m, x) => Math.max(m, x.block + 1), 0) : 0;
  const ok = new Uint8Array(T), blocks = new Uint8Array(T * BB);
  for (const x of found ? records : []) if (x.payload.length === BB) { ok[x.block] = 1; blocks.set(x.payload, x.block * BB); }
  const r = c.judge.frame(ok, blocks);
  const out = { found, seen: r.seen, bad: r.bad, test: r.test ? 1 : 0, judged: r.judged, ids: r.got.map((g) => g.id), bytes: null };
  if (r.got.length) {
    const bytes = new Uint8Array(PAYLOAD * r.got.length);
    r.got.forEach((g, k) => bytes.set(g.bytes, k * PAYLOAD));
    out.bytes = bytes.buffer;
  }
  return out;
}

function answerDecoded(item, fr, ms, dec) {
  const c = item.ctx, found = fr.finder?.found === 1 ? 1 : 0;
  const r = tally(c, fr.back?.records ?? [], found);
  const proved = r.seen > 0 || !!fr.word;
  const out = { type: "result", tag: item.tag, found: r.found, seen: r.seen, bad: r.bad, test: r.test, judged: r.judged, code: proved ? "lizard" : null, ms, heap: heapTop(), quad: found ? quadOf(dec.fh.tables.rings[fr.ring], fr.finder.H) : null, ids: r.ids };
  if (r.bytes) out.bytes = r.bytes;
  if (fr.back?.gained) out.gained = fr.back.gained;
  // ai: The band's word as F8 read it (gpu/wgsl/word.mjs), the shape the pool sends (recv-worker.mjs fmt): the page keeps
  // ai: the last one and takes it as proof of the symbol for the tracker, as it does the pool's.
  if (fr.word) out.fmt = { version: fr.word.version, fps: fr.word.fps, ring: fr.ring };
  post(item, out);
}

// ai: The grey range rides on the next decoded answer: the page reads nothing from a dropped one. levelsFrom: where
// ai: the last levels reported came from, for the stats message; the first levels of each source are logged once, so a
// ai: session's log holds the device's histogram beside a luma frame's count where both ran.
let levels = null, levelsFrom = "";
const levelsLogged = new Set();
function post(item, out) {
  if (item.levels) {
    levels = item.levels; levelsFrom = item.levelsFrom;
    if (!levelsLogged.has(levelsFrom)) { levelsLogged.add(levelsFrom); log(`grey levels from ${levelsFrom === "gpu" ? "the device's histogram" : "a luma frame"}: p5..p95 range ${levels.range}, median ${levels.mid}, ${(100 * levels.hi).toFixed(1)}% at 255, ${(100 * levels.lo).toFixed(1)}% at 0`); }
  }
  if (!out.dropped && levels) { out.levels = levels; levels = null; }
  if (item.box) out.box = item.box;   // ai: a track frame's crop, which the page's tracker never saw
  postMessage(out, out.bytes ? [out.bytes] : []);
}

// ai: stats: ms is the device's time a frame and lag a frame's wait from arrival to its answer, both over the frames
// ai: decoded since the last; B the frames the controller wants a batch to carry (0 until the first batch is planned);
// ai: queued the frames staged on the device awaiting a batch, held those waiting for the ring to exist; dropped every
// ai: frame answered without a decode since the worker started.
// ai: raw, peaks, cascade: the decoder's counters a frame (gpu/decoder.mjs COUNT); found, blocks: frames found and
// ai: verified blocks a frame. Together they show where candidates disappear on a device. f0: the share of frames
// ai: decoded that came as VideoFrames (F0), the rest as luma.
// ai: luma: the frames that came as luma (the grab), their found and their blocks, so F0's frames and the grab's can be
// ai: held side by side from one run (vf: the same for the VideoFrames): a device whose import gives a different luma
// ai: than its grab shows it there, in blocks a frame.
// ai: Where a batch's wall time goes, from the decoder's own laps (gpu/decoder.mjs batch() host), means a batch over
// ai: the window: batches; carried, the frames a batch really held (B is the controller's target); host, its laps
// ai: (upload, encode, scopes, map, read); submitToMap, scopes + map, the submit to the readback in hand; mapLatency,
// ai: that less deviceMs over the batches submitted to an idle device (out.idle), the readback's round trip alone,
// ai: null when none was; doneMs, answering the batch's frames here. ingestMs: F0's device time a frame (0 without
// ai: timestamps, or where no frame came as a VideoFrame). arrived: frames that reached this worker in the window,
// ai: from the page or off the track, gapMax the largest gap in ms between two; openMax: the most VideoFrames open at
// ai: once (gpuqueue.mjs).
// ai: cancelled, gained: the cancel stage's tally over the window's frames (gpu/back/cancel.mjs): frames it ran on with
// ai: a solved fit, and the blocks it verified (in blocks too), posted as sums, not means.
// ai: stageMs (only with ?stages=1): each profiled stage's device ms a frame over the window's batches (gpu/decoder.mjs
// ai: STAGES names them: bank is the proposer, describe both classifiers and RANK), null when no batch was profiled.
const WIN0 = () => ({ stage: {}, stageFrames: 0, n: 0, ms: 0, lag: 0, raw: 0, peaks: 0, cascade: 0, found: 0, blocks: 0, score: 0, top: 0, f0: 0, lumaN: 0, lumaFound: 0, lumaBlocks: 0,
  batches: 0, carried: 0, host: { upload: 0, encode: 0, scopes: 0, map: 0, read: 0 }, submitToMap: 0, idle: 0, mapLatency: 0, doneMs: 0, ingestMs: 0, arrived: 0, gapMax: 0, cancelled: 0, gained: 0 });
let win = WIN0(), statAt = 0;
// ai: A batch's frames answered.
function done(items, out, dec) {
  if (out.twinMs != null) twinMs = Math.round(out.twinMs);
  const now = performance.now(), frames = out.frames, ms = out.deviceMs / items.length, h = out.host ?? {}, toMap = (h.scopes ?? 0) + (h.map ?? 0);
  win.batches++; win.carried += out.carried ?? items.length; win.submitToMap += toMap; win.ingestMs += out.ingestMs ?? 0;
  for (const k in win.host) win.host[k] += h[k] ?? 0;
  if (out.idle) { win.idle++; win.mapLatency += toMap - out.deviceMs; }
  if (out.stageMs) { win.stageFrames += items.length; for (const k in out.stageMs) win.stage[k] = (win.stage[k] ?? 0) + out.stageMs[k]; }
  items.forEach((it, i) => {
    win.n++; win.ms += ms; win.lag += now - it.at; win.f0 += it.f0 ? 1 : 0;
    const fr = frames[i], k = fr?.counts ?? [];
    win.raw += k[1] ?? 0; win.peaks += k[2] ?? 0; win.cascade += k[3] ?? 0;
    const found = fr?.finder?.found === 1 ? 1 : 0, blocks = fr?.back?.records?.length ?? 0;
    win.found += found; win.blocks += blocks;
    win.cancelled += fr?.back?.cancelled ? 1 : 0; win.gained += fr?.back?.gained ?? 0;
    if (!it.f0) { win.lumaN++; win.lumaFound += found; win.lumaBlocks += blocks; }
    // ai: A VideoFrame's grey levels, from the decoder's histogram of its layer; a luma frame keeps levelsOf's.
    else if (fr?.hist) it.levels = levelsFromHist(fr.hist);
    // ai: the best hypothesis's score even when it is under the accept line: 0 means no quad was scored at all
    const sc = fr?.finder?.score ?? 0; win.score += sc; win.top = Math.max(win.top, sc);
    // ai: The held configuration: the last word read, in capture order, for the next batch the queue launches (a
    // ai: frame whose own word does not read decodes at it and leaves it as it was, 2026-09-27). A stopped queue's last
    // ai: batch leaves it alone.
    if (fr?.word && queue?.dec === dec) queue.word = fr.word.version;
    try { answerDecoded(it, frames[i], ms, dec); } catch (e) { postMessage({ type: "error", tag: it.tag, message: `GPU frame: ${e?.message ?? e}` }); answerEmpty(it); }
  });
  win.doneMs += performance.now() - now;
  stats();
}
function drop(items) { for (const it of items) answerEmpty(it); }
// ai: Only the decoder in use can be lost: one a config replaced was already let go.
function lost(q, reason) {
  if (q !== queue) return;
  queue = null; key = null;
  postMessage({ type: "lost", reason });
}
function stats() {
  const now = performance.now();
  if (!queue || now - statAt < 1000) return;
  statAt = now;
  const fh = queue.fh;
  const per = (v) => (win.n ? v / win.n : 0), perB = (v) => (win.batches ? v / win.batches : 0);
  postMessage({ type: "stats", ms: per(win.ms), lag: per(win.lag), B: fh.size, inflight: fh.inflight, queued: queue.staged, held: queue.held, dropped,
    raw: per(win.raw), peaks: per(win.peaks), cascade: per(win.cascade), found: per(win.found), blocks: per(win.blocks), score: per(win.score), top: win.top, f0: per(win.f0),
    vf: { n: win.n - win.lumaN, found: win.found - win.lumaFound, blocks: win.blocks - win.lumaBlocks }, luma: { n: win.lumaN, found: win.lumaFound, blocks: win.lumaBlocks }, levelsFrom,
    batches: win.batches, carried: perB(win.carried), host: Object.fromEntries(Object.entries(win.host).map(([k, v]) => [k, perB(v)])), submitToMap: perB(win.submitToMap),
    mapLatency: win.idle ? win.mapLatency / win.idle : null, doneMs: perB(win.doneMs), ingestMs: per(win.ingestMs), arrived: win.arrived, gapMax: win.gapMax, openMax: queue.takeOpenMax(),
    cancelled: win.cancelled, gained: win.gained,
    // ai: the last plan's ms (fh.batcher.planMs: the first, unless a larger crop planned again) and the int8 twins'
    // ai: measurement on the first batch (its ms and fh.twins), this decoder's
    planMs: fh.batcher ? Math.round(fh.batcher.planMs) : null, twinMs, twins: fh.twins ?? null,
    stageMs: win.stageFrames ? Object.fromEntries(Object.entries(win.stage).map(([k, v]) => [k, v / win.stageFrames])) : null });
  win = WIN0();
}

onmessage = ({ data: m }) => {
  if (m.type === "config") { latest = m.config; configs = configs.then(() => configure(m.config)); }
  else if (m.type === "frame") frame(m);
  else if (m.type === "track") trackMsg(m);
  else if (m.type === "crop") { if (trk) trk.crop = { x: m.x, y: m.y, w: m.w, h: m.h }; }
};
