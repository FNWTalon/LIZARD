// Sender: paints the chosen code at a whole number of device pixels per cell.
// ai: Standalone (2026-09-26): it picks its format,
// ai: encodes and paints the test stream or a file picked from this device with no server.
// ai: For a user the page is a file, several or a folder (bundled as one tar, tar.mjs): choose or drop them, Start,
// ai: Fullscreen, Stop, the version picked from the window over
// ai: the whole ladder (up to LIZARD-560 until 2026-09-29, APP_TOP). The lab's switches are in Settings and Developer
// ai: Tools, their ids the URL presets.
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { DEFAULT_LOG2 } from "../liblizard/sim/xfer.mjs";
import { VERSIONS, NAME, SAMPLES, SPAN, N_FOR, RINGS, RING_DEFAULT, FOCUS_BITMAP, MODULES, OB_QUIET, pickVersion, blocksFor } from "../liblizard/sim/lizard_pick.mjs";
import { GAP_MODULES } from "../liblizard/gpu/encoder.mjs";
import { bundle } from "./tar.mjs";
import { remember, persist, say, bytes, rate, tipsDone, about, registerApp, collapser, sideResizer, icon, sayName } from "./ui.mjs";

// The room a format needs holds the codec's margin (src/focus.h FOCUS_QUIET), and only the codec knows it.
await initOb();
let offeredKBs = 0;
let fileBytes = 0, sentBytes = 0;   // ai: the file's bytes and its bytes as sent (every chunk's zstd frame or its own; 2026-10-05), for the sending line
// ai: ", 420 KB": the bytes that go, what compression left of the file (2026-10-05; the file's own size, and "compressed
// ai: to" beside it, are #tx's since), at module scope: start() names its file's bytes `bytes`, which there shadows
// ai: ui.mjs's formatter (a TypeError on the first frame left the line at "Preparing")
const sizeLine = () => (fileBytes ? `, ${sizeText(sentBytes, fileBytes)}` : "");
// ai: what goes, "<compressed> (<raw> raw)" once the transfer is made, "<raw> raw" while it is being made (2026-10-19;
// ai: the native senders' the same)
// ai: each figure kept whole on its line, "raw" with its own (no-break spaces, 2026-10-23: "(47 KB / raw)" broke there)
const nbsp = (t) => t.replace(/ /g, "\u00a0");
const sizeText = (sent, raw) => (sent ? `${nbsp(bytes(sent))} ${nbsp(`(${bytes(raw)} raw)`)}` : nbsp(`${bytes(raw)} raw`));
const $ = (id) => document.getElementById(id), canvas = $("c"), ctx = canvas.getContext("2d"), tmp = document.createElement("canvas"), tctx = tmp.getContext("2d"), gcanvas = $("cg");
// nextId is the block id space and seq only counts painted frames. They are separate because
// blocksPerFrame changes when repick() swaps the layout mid-stream: seq * blocksPerFrame would
// then jump backwards onto ids already sent, and every one of them would read as a repeat.
// ai: cfg: what is painted. running: from Start until Stop (preparing, then sending); live: a frame is painted.
// ai: lap: a file's data blocks a lap (sim/xfer.mjs), 0 for the test stream; dataSent: data blocks painted this run.
let timer = 0, seq = 0, nextId = 0, image = null, painted = 0, paintMs = 0, encMs = 0, lastReport = 0, cfg = null, shown = null;
let running = false, live = false, lap = 0, dataSent = 0, dataWin = 0;
// ai: Paused (2026-10-02): the run kept, nothing asked or
// ai: painted, the last picture left on screen; a receiver holds what it has, and a resume paints on at once.
let paused = false, sendingLine = () => {};   // ai: says the sending line again (after a pause)

// A plain grey LIZARD symbol, the default and the one that gets expensive, is encoded and turned into pixels in a
// worker (lizard-web/send-worker.mjs), in wasm, a few frames ahead: at LIZARD-1024 that was 45 ms a frame on this thread
// ai: (2026-09-23) and is now a putImageData here.
const AHEAD = 3;
// ai: The rejection a stop hands a reply still awaited: not an error, so nothing is said about it.
const STOPPED = new Error("stopped");
let sw = null, gen = 0, pending = null, ahead = new Map(), inflight = 0, reqSeq = 0, spare = [], respecPending = false, behind = 0;
function swInit() {
  if (sw) return;
  sw = new Worker(new URL("./send-worker.mjs", import.meta.url), { type: "module" });
  sw.onmessage = ({ data: m }) => {
    // ai: a pick's transfer made (prepare): its own number, not a run's
    if (m.type === "prepared") { madeFor(m); return; }
    if (m.gen !== gen) return;   // from before a stop or a restart
    if (m.type === "frame") { inflight--; ahead.set(m.seq, m); return; }
    // ai: A batch for the GPU encoder: queued, and encoded a few frames at a time as the paint loop's time allows (pump).
    if (m.type === "frames") {
      inflight--;
      const b = batchOf.get(m.seq);
      batchOf.delete(m.seq);
      if (!b || !genc) { gpuGiveUp(new Error(`frames ${m.seq} came back with no GPU configuration`)); return; }
      encodeQueue.push({ ...b, seq0: m.seq, blocks: new Uint8Array(m.buf), buf: m.buf, per: m.codes * b.cfg.V * b.cfg.B, next: 0, codes: m.codes, ms: m.ms / m.count, data: m.data });
      pump();
      return;
    }
    if (m.type === "error") { if (pending) { pending.rej(new Error(m.message)); pending = null; } else fail("The encoder stopped", new Error(m.message)); return; }
    if (pending && m.type === pending.type) { const p = pending; pending = null; p.res(m); }
  };
  sw.onerror = (e) => {
    const err = new Error(e.message || "the sender worker did not load");
    if (pending) { pending.rej(err); pending = null; } else fail("The encoder stopped", err);
  };
}
// A request that waits for its reply: start answers "ready", respec answers "respec".
function swCall(msg, transfer = []) {
  return new Promise((res, rej) => { pending = { type: msg.type === "start" ? "ready" : msg.type, res, rej }; sw.postMessage({ ...msg, gen }, transfer); });
}

// ai: The GPU encoder (Settings' #enc, 2026-09-29; gpu/encoder.mjs), for a PC with a good GPU: the sender
// ai: had lagged at LIZARD-1024. Its latency matters little and a dropped frame does, so frames go in batches (16, fewer
// ai: where the device's memory holds fewer: cfg.batch): the worker makes a batch's blocks alone, the page encodes the batch on
// ai: the device into one of its configuration's two halves as it arrives, and presents a frame's slot on #cg when the
// ai: frame is due, two batches asked for ahead (up to 32 frames: a stall anywhere is absorbed there, not dropped);
// ai: the wasm path puts the worker's pixels on #c, AHEAD frames ahead. auto takes it where a device builds and its first frame
// ai: at each format and ring matches the C (the border to the byte, the picture within a grey level: GpuEncoder.check,
// ai: a guard against a driver's garbage, not a conformance rule); anything wrong (no WebGPU, a failed check, a GPU
// ai: error, a lost device) starts the run again on the wasm path, auto staying there until the menu moves, #tx saying
// ai: why. WebGPU asked for by name fails instead. ?pref=low-power|high-performance names the adapter (high-performance
// ai: unless asked: a PC's good GPU; scripts/exp/send_rate.mjs takes the iGPU), ?encprof a stage a compute pass, each timed (#tx's
// ai: "GPU stages" line). A configuration (its buffers and slots) serves the batches asked for
// ai: while it was the current one (batchOf) and is released once the last of them is presented and it is not current.
let genc = null, gctx = null, gcfg = null, gpuOff = null, gpuMs = null;
const batchOf = new Map(), checked = new Set(), encodeQueue = [];
// ai: The encodes' pace. WebGPU runs one queue in order, so a present waits behind every encode submitted before it: an
// ai: animation frame submits encodes (whole sub-batches, gpu/encoder.mjs) while their measured GPU time (the
// ai: configuration's frameMs) fits half the interval between animation frames, at least one frame an animation frame
// ai: (one a frame until the configuration's first measurement), and a frame is ready once its encode is submitted.
// ai: Encoding a batch of 16 in one submission (85 ms at LIZARD-1024 on the iGPU) showed 53.5 of 60 frames a second with
// ai: none late on the page (2026-09-29): the presents waited on the device. false: the GPU was given up (the run restarts).
// ai: tickMs is the display's refresh, measured: a running mean of the animation frames' intervals, each over the
// ai: refreshes it spans (a frame the browser missed counts as two), so it stays the refresh when frames drop.
let tickMs = 1000 / 60, lastTick = 0, tickBudget = 0;
// ai: The paint's pace (2026-10-01). Where the asked rate is the display's over a whole number m (60 on a 60 Hz
// ai: screen: 1; 60 on 120 Hz or 30
// ai: on 60: 2), every picture stays m refreshes: m = 1 paints at every animation frame, a larger m at the m-th
// ai: refresh after the last paint, counted from the frames' own times. The schedule before (`next`, a grid 1000 / fps
// ai: ms apart and a paint at the first frame at or past a point of it) set the grid's phase against the refreshes at
// ai: Start, and where a grid point fell within the frame times' jitter of a refresh each paint went a refresh early or
// ai: late at random: pictures of 1 and 3 refreshes at 120 Hz, a repeat and a skip to a camera at 60, for as long as the
// ai: clocks held that phase. Firefox's frame times step in whole milliseconds and Chrome's in 0.1, so Firefox's band of
// ai: bad phases is ten times wider: 60 on 120 Hz, 5.6% of pictures wrong (18% at the worst start) against Chrome's 0.3%
// ai: (1.7), 0.0% under this rule in both (scripts/exp/paint_cadence.mjs). A ratio that is no whole number (24 on 60, 60 on 144)
// ai: keeps the grid, a picture going up at the refresh nearest its time. Whole within 5%: a display's clock is within
// ai: 0.1% of its rate and the measured refresh within 2% (Firefox's millisecond steps at 120 Hz), and no ratio met in
// ai: practice is nearer a whole number than 8% (165 Hz at 60). lastPaint: the frame time of the last paint; held: the
// ai: report's second's pictures by the refreshes each stayed as the page sees them (1, 2, 3, 4 or more), logged.
let lastPaint = 0;
const held = [0, 0, 0, 0];
// ai: The animation frames the browser gave this second, and their intervals (2026-10-04, replay 077, a Surface Pro 9 in
// ai: Edge: "painted 30.0 out of 60 asked" at about 3 ms of GPU a frame, so not lag).
// ai: The page paints at most once an animation frame, and Chromium's battery saver (Edge's efficiency mode, Chrome's
// ai: Energy Saver, on by default when a laptop is unplugged) gives a page 30 a second: every picture then stays two
// ai: refreshes and a camera at 60 reads each twice (STATUS "Replay 077"). tickMs above reads such a page as a 60 Hz
// ai: screen whose every other frame was missed, so the readout said "screen 60.0 Hz" while 30 frames came: the count
// ai: here is the browser's own, which the rail's capacity takes (drawFps, showRail). A gap over 250 ms (a pause, a hidden
// ai: tab, a stall) is left out, so the rate is the pace the browser keeps while it draws.
const rafGaps = [];
const rafRate = () => { const t = rafGaps.reduce((a, b) => a + b, 0); return t > 0 ? 1000 * rafGaps.length / t : 0; };
function due(now, next, period) {
  const per = period / tickMs, m = Math.round(per);
  if (m >= 1 && Math.abs(per - m) <= 0.05 * m) return m === 1 || !lastPaint || Math.round((now - lastPaint) / tickMs) >= m;
  return now >= next - tickMs / 2;
}
function pump() {
  while (genc && encodeQueue.length && tickBudget > 0) {
    const b = encodeQueue[0], cost = b.cfg.frameMs;
    const k = Math.max(1, Math.min(b.count - b.next, b.cfg.sub, cost ? Math.floor(tickBudget / cost) : 1));
    // ai: the count the pilots carry: the painted count (seq) of the encode's first frame, mod 4 (SPEC 7.3)
    try { genc.encode(b.half, b.next, b.blocks.subarray(b.next * b.per, (b.next + k) * b.per), k, b.cfg, (b.seq0 + b.next) & 3); } catch (e) { gpuGiveUp(e); return false; }
    for (let f = b.next; f < b.next + k; f++) ahead.set(b.seq0 + f, { gcfg: b.cfg, slot: b.half * b.cfg.batch + f, w: b.cfg.W, codes: b.codes, ms: b.ms, data: b.data[f] });
    b.next += k;
    tickBudget -= cost ? k * cost : tickBudget;
    if (b.next === b.count) { encodeQueue.shift(); spare.push(b.buf); }
  }
  return true;
}
const wantGpu = () => $("enc").value === "gpu" || ($("enc").value === "auto" && !gpuOff);
// ai: The encoder's menu lists GPU and CPU alone (2026-10-05): with neither chosen, its hidden auto option names the one
// ai: auto runs on (the run's, else the one it would try: the GPU where the browser has WebGPU and it has not failed here).
function showEnc() {
  const gpuNow = cfg ? !!gcfg : !gpuOff && "gpu" in navigator;
  $("enc").querySelector('option[value="auto"]').textContent = gpuNow ? "GPU" : "CPU";
}
// ai: The encoder, one a page, made once however many ask at a time (a start and a re-pick), and made again in place of
// ai: one whose device was lost while nothing ran on it; lost under a running configuration, the caller gives up.
let gencMaking = null;
async function gpuEncoder() {
  // ai: one made by name on a software GPU is not auto's (genc.software): auto goes to wasm
  if (genc && !genc.lost) { if (genc.software && $("enc").value === "auto") throw new Error("only a software GPU here"); return genc; }
  if (genc && gcfg) throw new Error(`device lost: ${genc.lost}`);
  if (genc) { try { genc.destroy(); } catch {} genc = gctx = null; }
  gencMaking ??= (async () => {
    const powerPreference = URLP.get("pref") ?? "high-performance";
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference }).catch(() => null);
    // ai: auto takes no software GPU (SwiftShader, llvmpipe, 2026-10-01): the C in its worker paints faster than one;
    // ai: the GPU asked for by name still runs on it (check_send ENC=gpu)
    const i = adapter?.info ?? {};
    const software = !!adapter && (i.isFallbackAdapter || adapter.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(`${i.vendor} ${i.architecture} ${i.description}`));
    if ($("enc").value === "auto" && software) throw new Error("only a software GPU here");
    const { GpuEncoder } = await import("../liblizard/gpu/encoder.mjs");
    const e = await GpuEncoder.create({ adapter, powerPreference, profile: URLP.has("encprof"), log: (m) => console.log(`sender: ${m}`) });
    e.software = software;
    gctx = e.attach(gcanvas);
    checked.clear();
    return (genc = e);
  })().finally(() => { gencMaking = null; });
  return gencMaking;
}
async function gpuConfigure(s, codes, fps, gap) {
  const e = await gpuEncoder();
  const cfg = await e.configure({ spec: s, codes, fps, gap }), key = `${s.n}/${s.span}/${codes}/${codes > 1 ? gap : 0}`;
  cfg.pending = 0; cfg.half = 0;
  if (!checked.has(key)) {
    const blocks = new Uint8Array(codes * cfg.V * cfg.B);
    for (let i = 0; i < blocks.length; i++) blocks[i] = (7 * i + (i >> 9)) & 255;
    let r;
    try { r = await e.check(blocks, 1, cfg); } catch (err) { e.release(cfg); throw err; }
    if (!r.ok) { e.release(cfg); throw new Error(`its frame is not the C's: ${r.borderDiff} border pixels differ, the picture by up to ${r.squareMax} levels${e.lost ? `, device lost: ${e.lost}` : e.error ? `, ${e.error}` : ""}`); }
    console.log(`sender: the GPU encoder at ${cfg.label}: the border the C's, the picture within ${r.squareMax} level (${r.squareDiff} of ${r.pixels} px differ)`);
    checked.add(key);
  }
  return cfg;
}
function gpuRelease(cfg) { if (cfg && cfg !== gcfg && cfg.pending <= 0) genc?.release(cfg); }
function gpuGiveUp(e) {
  console.warn("sender: the GPU encoder:", e);
  try { genc?.destroy(); } catch {}
  genc = gctx = gcfg = null; batchOf.clear(); encodeQueue.length = 0;
  if ($("enc").value === "gpu") { fail("The GPU encoder stopped", e); return; }
  gpuOff = String(e?.message ?? e);
  showEnc();
  if (running) go();
}

// ai: Blocks a frame, a slider (2026-10-01): any whole number of blocks from 1 to 128, a block 8 sub-channels
// ai: (src/focus.h FOCUS_GROUP), which is what the format word names, so every step is a symbol every receiver reads; the
// ai: menu of 64 versions before stepped by two blocks. #subch holds what the page reads, "auto" or the sub-channels; the
// ai: slider (#blocks) is a view over it, kept out of the saved settings, its first step (0) auto (2026-10-05; an Auto box
// ai: beside it before). The slider commits on release (a re-pick for every step dragged would rebuild the encoder each
// ai: time); under auto its label says what auto took, once known ("Auto, 57, 26.7 KB").
function showBlocks() {
  const auto = $("subch").value === "auto", b = auto ? (shown ? shown.subch / 8 : 0) : +$("subch").value / 8;
  $("blocks").value = auto ? "0" : String(b);
  $("blocksOut").textContent = auto ? (b ? `Auto, ${blocksText(b)}` : "Auto") : blocksText(b);
}
// ai: the title says "Blocks a frame", so the value is the count and its bytes alone, which fit beside it in the 20rem column
// ai: "57, 26.7 KB": the blocks a frame of the slider's size carries (the format's rate profile, 2026-10-07: 57 at
// ai: its 60) and what they hold, under the title "Blocks" (2026-10-05; "60 blocks, ..." before)
const blocksText = (v) => { const b = blocksFor(8 * v); return `${b}, ${((b * 469) / 1000).toFixed(1)} KB`; };
$("blocks").addEventListener("input", () => { const v = +$("blocks").value; $("blocksOut").textContent = v ? blocksText(v) : "Auto"; });
$("blocks").addEventListener("change", () => { const v = +$("blocks").value; $("subch").value = v ? String(8 * v) : "auto"; $("subch").dispatchEvent(new Event("change")); });
$("subch").addEventListener("change", showBlocks);
// ai: The page's defaults are LIZARD-480 at 60 painted a second (2026-10-01, the night the Android app, its camera's
// ai: phase held, read 93 to 95% of it; auto and 24 a second before), in send.html
// ai: itself since 2026-10-02 (#subch 480, #fps 60: set here, after the page was drawn, they flashed auto first). A
// ai: setting the user changed is kept over these (persist, below), and a URL preset over that.

// ai: The display rate goes in the BAND (src/fmt.h). The word is the only channel a receiver has, and the rate is the
// ai: one thing about an animated symbol that no single frame shows.
// It is what the box asks for, not what the loop achieved: a sender falling behind then reads as
// what it is, the band saying 60 where the receiver counts 15 distinct frames a second. Re-stating it is free,
// since only the word's cells are repainted (src/focus.c focus_fmt_fps), so the box can move mid-transfer.
let bandFps = 0;
function stateFps(force = false) {
  const v = Math.max(0, Math.min(255, Math.round(+$("fps").value) || 0));
  if (!force && v === bandFps) return;
  sw?.postMessage({ type: "fps", fps: v, gen });
  if (gcfg && v !== bandFps) genc.setFps(v).catch(gpuGiveUp);
  bandFps = v;
}

// A LIZARD file goes as a chunked transfer (sim/xfer.mjs): chunks of 2^chunk bytes (?chunk=10..24, 4 MiB unless
// ai: asked), a fountain each, the header and manifest in the light, and only there: the receiver reads it blind
// ai: (2026-09-26).
const URLP = new URLSearchParams(location.search), CHUNK_LOG2 = +URLP.get("chunk") || DEFAULT_LOG2;
// ai: The size on screen and the surround, test-only menus until 2026-10-01, URL parameters since: fit=stretch|whole|fit,
// ai: bg=none|noise|text|photo|checker|squares|mixed.
const FIT = URLP.get("fit") ?? "stretch", SURROUND = URLP.get("bg") ?? "none";

// Which generator fills a test block (sim/phy.mjs STREAMS). It rides in the spec so the receiver
// and any later replay of research/captures take it from the same place; a capture recorded before
// this field existed has no stream and replays against the old walk.
// ai: The receiver judges it from the light (SPEC 9.3), so it is SHAKE256 and nothing else.
const STREAM = "shake256";

function spec() {
  // The 2 x 2 picture split is gone from this page (2026-09-24): its tiles were n = 128 pictures off the ladder
  // with no format word, a symbol that cannot describe itself.
  // ai: The format's rate profile (the codec's, from the sub-channel count: src/focus.c focus_tiers_for, four rates
  // ai: since 2026-10-08), equal power, soft LDPC: the power tilt and RS(80,64) lost (research/10).
  // n is derived from the version (N_FOR in sim/lizard_pick.mjs) and is never chosen here.
  const subch = $("subch").value === "auto" ? pick(room()).subch : +$("subch").value, n = N_FOR(subch);
  // span written out, not 0, so a recording says which symbol it holds (sim/phy.mjs recordedSpec).
  // ai: The ring Settings names, or the default ring, the same for every picture (sim/lizard_pick.mjs RING_DEFAULT).
  return { phy: "focus", n, subch, mode: 1, span: SPAN(n, ringChosen() ?? RING_DEFAULT), bitmap: FOCUS_BITMAP,
    variants: [{ name: "rx" }], stream: STREAM };
}

// How large a LIZARD symbol to paint, and how many sub-channels to fill, from the window alone. Nothing comes back
// from the receiver: a back channel would make the optical link pointless, so both numbers are the sender's guess.
//
// The symbol is the picture resampled into its ring's span (SPAN) plus a 15-module border a side
// (sim/lizard_pick.mjs OB_MARGIN), so SAMPLES(n) samples, and the margin the codec paints outside that (src/focus.h
// FOCUS_QUIET), which ROOM_FOR counts. NEVER let the browser
// squeeze a format below the room it needs: downscaling past it throws the outer rings away, which is the whole payload
// above the first few. So take the largest format that fits, and stretch up to
// fill the rest of the room (an upscale only interpolates; research/09 measures smoothed fractional scaling at 2% of
// the mean and 11% at worst).
//
// Which format, from the WINDOW alone. There is no camera assumption: the sender cannot see the capture, and a
// guess at it was a control on this page that nobody could answer. Each format states the room it needs
// (ROOM_FOR, from its top ring), so the largest that fits is the pick. sim/lizard_pick.mjs has the arithmetic so
// scripts/exp/pick_check.mjs can exercise it.
// ai: The Margin, Settings' slider (2026-10-23; a gap between two codes before it, from 2026-10-03, shown with two
// ai: codes): the light round a code in modules from its ring, the quiet zone (OB_QUIET) at least, 34 at most (a
// ai: painter's widest gap, 64 modules), MARGIN_DEFAULT the encoders' own gap (GAP_MODULES) as a margin. Two codes stand
// ai: two margins apart (the painters' gap, gapOf(), past their quiet zones) and the area's edges a margin from the
// ai: codes, so each code takes its painted side and two margins: marginScale() times its side in the ring, which the
// ai: room and the canvas (layout) take and the checks read (#tx dataset.over). Commits on release, as Size.
const MARGIN_DEFAULT = GAP_MODULES / 2 + OB_QUIET;
const marginOf = () => { const m = Math.round(+$("margin").value); return Number.isFinite(m) ? Math.max(OB_QUIET, Math.min(34, m)) : MARGIN_DEFAULT; };
const gapOf = () => 2 * (marginOf() - OB_QUIET);
const marginScale = (ring = ringChosen() ?? RING_DEFAULT) => { const side = MODULES(256, ring) + 2 * OB_QUIET; return (side + gapOf()) / side; };
// ai: With two codes each symbol gets half the width: the room is one symbol's.
function room() {
  const box = canvas.parentElement, dpr = devicePixelRatio || 1, codes = codesOf(), k = marginScale();
  $("tx").dataset.over = k;
  return Math.max(64, Math.min(box.clientWidth / codes, box.clientHeight) / k * dpr * sizeOf());
}
// ai: The code's size, Settings' slider (2026-10-03, against instability at a monitor's edges): a share of <main>'s
// ai: room, 25 to 100%, which both the pick (room) and the canvas (layout)
// ai: take, so the code sits smaller in the middle with the surround round it; under auto the pick follows the smaller
// ai: room, as it follows a smaller window. Commits on release, as blocks a frame does.
const sizeOf = () => Math.max(0.25, Math.min(1, (+$("size").value || 100) / 100));
// ai: Symbols a frame, Settings' "codes": two side by side for a native receiver reading a 2:1 crop. No
// ai: browser decodes two (the receiver's 2:1 path was archived 2026-09-28 and deleted 2026-09-29, the receiver's side
// ai: planned as two 960 x 960 frames divided by a line); the option says so.
const codesOf = () => (+$("codes").value === 2 ? 2 : 1);
// The controls read here, the arithmetic in liblizard/sim/lizard_pick.mjs so that scripts/exp/pick_check.mjs can exercise it.
// ai: The library's pick over the whole ladder, at the first pick and every re-pick (capped at LIZARD-560, APP_TOP, until
// ai: 2026-09-29).
function pick(roomPx) { return pickVersion(roomPx, undefined, ringChosen()); }
// ai: Settings' ring (an index into RINGS), or null for the default ring (RING_DEFAULT): the menu has no auto
// ai: since 2026-10-05 (the 128 selected), so null only where it holds no ring at all. Its options are the rings'
// ai: cells since 2026-10-10 (indices before, when the 96 was the third), so a saved index no longer matches one.
function ringChosen() { const v = $("ring").value, i = RINGS.indexOf(+v); return v === "" || v === "auto" || i < 0 ? null : i; }
const ringOf = (s) => RINGS.indexOf(s.span / 2);

// The canvas laid out for a symbol of n samples a side, quiet zone included, whenever that or the room changed.
// ai: codes symbols side by side, gap paint px between them (the encoder's, gapOf() modules): the canvas is
// ai: codes n + (codes - 1) gap wide and n high, stretched as one. gpu: the GPU encoder's canvas (#cg) is the one
// ai: shown, else #c; canvasWhole is its integer factor, which the GPU's present draws at. #tx dataset.gap is the gap
// ai: in canvas px, for a check that cuts the halves.
let canvasWhole = 1;
function layout(n, codes = 1, gpu = false, gap = 0) {
  const fw = codes * n + (codes - 1) * gap;
  if (!image || image.width !== fw || image.height !== n || image.gpu !== gpu) {
    image = { width: fw, height: n, gpu }; tmp.width = fw; tmp.height = n;
    const cv = gpu ? gcanvas : canvas;
    canvas.hidden = gpu; gcanvas.hidden = !gpu;
    // ai: the frame with a margin less the quiet zone at each edge (marginScale() - 1 of a symbol's side across both)
    const k = marginScale(shown ? ringOf(shown) : undefined), box = canvas.parentElement;
    const dpr = devicePixelRatio || 1, room = Math.min(box.clientWidth / (fw / n + k - 1), box.clientHeight / k) * dpr * sizeOf();
    // Three ways to put a symbol of n samples on screen. "stretch" is the default and what a small screen wants:
    // paint whole device pixels a sample, then let the browser stretch that up to fill the room. Stretching UP only
    // interpolates, so no ring is lost; squeezing DOWN below the format's room (ROOM_FOR) would throw the outer rings
    // away, which is most of the payload, and that is the one thing never to do (auto's pick sees to it).
    const mode = FIT, whole = Math.max(1, Math.floor(room / n));
    const scale = mode === "fit" ? 1 : whole, css = mode === "whole" ? n * scale : room;
    cv.width = fw * scale; cv.height = n * scale;
    cv.style.width = `${(fw * css / n) / dpr}px`; cv.style.height = `${css / dpr}px`;
    cv.style.imageRendering = "auto";
    ctx.imageSmoothingEnabled = false;
    canvasWhole = scale;
    // ai: scale: what a sample ends up as on screen, fractional once stretched.
    $("tx").dataset.scale = (css / n).toFixed(3); $("tx").dataset.gap = gap * scale;
  }
}

// A frame the worker made: its pixels are final, so it is one putImageData, and at one canvas pixel a sample
// ai: (the large pictures on most screens) straight onto the canvas. CSS scales the canvas to the room as before.
function paintRGBA(fr) {
  const t0 = performance.now(), n = fr.w, codes = fr.codes ?? 1, gap = fr.gap ?? 0, fw = codes * n + (codes - 1) * gap;
  layout(n, codes, false, gap);
  const img = new ImageData(new Uint8ClampedArray(fr.buf), fw, n);
  if (canvas.width === fw) ctx.putImageData(img, 0, 0);
  else { tctx.putImageData(img, 0, 0); ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height); }
  spare.push(fr.buf);   // copied onto the canvas, so the worker can fill it again
  paintMs += performance.now() - t0;
}
// ai: A frame the GPU encoded: its slot presented on #cg at the canvas's whole factor; its configuration released once
// ai: no frame asked for under it is left and it is not the current one. false: the GPU was given up (the run restarts).
function paintGPU(fr) {
  const t0 = performance.now();
  layout(fr.w, fr.codes ?? 1, true, fr.gcfg.gap ?? 0);
  try { genc.present(fr.slot, gctx, canvasWhole, fr.gcfg); } catch (e) { gpuGiveUp(e); return false; }
  fr.gcfg.pending--; gpuRelease(fr.gcfg);
  paintMs += performance.now() - t0;
  return true;
}

// ai: What Start sends: the payload chosen (tar.mjs bundle: one file as itself, several files or a folder as one tar),
// ai: or the test stream. A pick through #file or #folder, or a drop, sets it; nothing else does.
let chosen = null;
const isTest = () => $("payload").value === "test";

// ai: The states, few on purpose: idle (nothing to send) and ready (Start offered), then preparing and sending (Stop
// ai: offered in Start's place, the same slot, so the controls keep their height), and an error (red, until the next
// ai: start or pick).
function showReady() {
  const f = chosen;
  $("go").hidden = false; $("go").disabled = !isTest() && !f; $("stop").hidden = true;
  showChoice(false); $("nums").textContent = "";
  if (f && !isTest()) sayName($("state"), "", f.name, `, ${f.bundled ? `${f.files} ${f.files === 1 ? "file" : "files"}, ` : ""}${sizeText(f.sent, f.size)}`);
  else say($("state"), isTest() ? "The test stream" : "");
  showRail();
}
// ai: Pause and resume (the rail's button, and Resume beside Stop while paused, so the open column has it too).
function setPaused(p) {
  if (!running) return;
  paused = p;
  $("go").hidden = !p; $("go").disabled = !p; $("go").textContent = p ? "Resume" : "Start";
  if (p) $("nums").textContent = "";
  if (p) say($("state"), "Paused"); else sendingLine();
  showRail();
}
// ai: The collapsed rail (send.html #rail): pause while sending, play to resume or to start, and the code's capacity,
// ai: what it offers at most (codes x blocks a frame x 469 B x pictures a second), while a run is configured. Written
// ai: only when it changes (tick calls it every frame). The pictures a second are the asked rate, or the browser's own
// ai: animation frames where fewer (drawFps, measured each second while sending; since 2026-10-04, so an external
// ai: limiter shows in the MB/s: Edge's efficiency mode drew 30 of 60 while the rail offered 60's capacity).
let railKey = "", drawFps = 0;
function showRail() {
  const asked = Math.max(1, +$("fps").value), shows = drawFps ? Math.min(asked, Math.round(drawFps)) : asked;
  const kbs = cfg ? cfg.codes * cfg.blocksPerFrame * cfg.usefulBytes * shows / 1000 : null;
  const key = `${running}|${paused}|${kbs}|${$("go").disabled}`;
  if (key === railKey) return;
  railKey = key;
  const b = $("railSend");
  b.classList.toggle("off", !running || paused); b.title = !running ? "Start" : paused ? "Resume" : "Pause";
  b.disabled = !running && $("go").disabled;
  const [v, u] = kbs != null ? rate(kbs).split(/\s/) : ["", ""];   // ai: \s takes rate()'s no-break space
  $("railRate").innerHTML = v ? `${v}<small>${u}</small>` : "";
}
function showRunning(before, name) {
  $("go").hidden = true; $("go").disabled = true; $("stop").hidden = false;
  if (name) sayName($("state"), before, name); else say($("state"), before);
}
// ai: Every error ends here: the stream stops, the state line says what went wrong in red, the console has the rest.
function fail(what, e) {
  console.error(`sender: ${what}:`, e);
  stop(); showReady();
  say($("state"), `${what}: ${e?.message ?? e}`, "bad");
}

// ai: The screen stays on while a symbol is painted. The browser lets the lock go when the page is hidden, so it is
// ai: taken again on coming back; a browser without one, or one that refuses, paints on regardless.
let wake = null;
// ai: A lock is kept only by the run that asked for it (gen), and only in place of none or one the browser let go: a
// ai: stop, or a stop and a start, while a request was out would otherwise leave a lock nothing releases.
async function holdWake() {
  const my = gen;
  try {
    const w = await navigator.wakeLock?.request("screen");
    if (running && my === gen && (!wake || wake.released)) wake = w ?? null; else w?.release().catch(() => {});
  } catch {}
}
function dropWake() {
  try { wake?.release().catch(() => {}); } catch {}
  wake = null;
}
// ai: A sender in a background tab pauses (2026-10-03): hidden,
// ai: it gets no animation frames and shows nothing anyway, and paused it asks nothing more of its worker or GPU. Shown
// ai: again, it resumes, unless it was paused by hand before (hiddenPause: the pause was the background's).
let hiddenPause = false;
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") { if (running && !paused) { hiddenPause = true; setPaused(true); } return; }
  if (hiddenPause) { hiddenPause = false; if (running) setPaused(false); }
  if (running) holdWake();
});

async function start() {
  stop();
  const test = isTest(), file = test ? null : chosen;
  if (!test && !file) { showReady(); return; }
  const what = test ? "the test stream" : file.name, my = gen;
  running = true;
  showRunning(test ? `Preparing ${what}` : "Preparing ", test ? null : what);
  holdWake();
  // ai: A browser reads a chosen file only as it was when chosen: once it has been saved again, moved or replaced
  // ai: on disk the read is refused (NotReadableError, in the browser's own words "permission problems that have
  // ai: occurred after a reference to a file was acquired"). The page says what to do about it.
  // ai: the file's transfer is the worker's, made at the pick (prepare; a start waits there for one still being made,
  // ai: once the page has handed the file over); one whose making failed is read and made here, which says why
  if (!test) await file.posted;
  if (my !== gen) return;
  const bytes = test || !file.failed ? null : await readOf(file);
  if (my !== gen) return;   // ai: stopped, or started again, while the file was read
  // ai: The spec after the read, so the room is the one the preparing controls left.
  const s = spec(), xfer = bytes ? { name: file.name, type: file.type, chunkLog2: CHUNK_LOG2 } : null;
  // The worker holds the codec and, for a file, the fountain: the page keeps neither.
  swInit();
  ahead.clear(); inflight = 0; reqSeq = 0; spare = []; respecPending = false; behind = 0;
  const fps = Math.max(0, Math.min(255, Math.round(+$("fps").value) || 0));
  const codes = codesOf(), gap = gapOf(), made = await swCall({ type: "start", spec: s, pick: file && !file.failed ? file.pick : null, bytes: bytes ? bytes.slice().buffer : null, fps, xfer, codes, gap });
  bandFps = fps;
  // ai: The GPU encoder for this run, where the menu asks for it and it builds (WebGPU by name: its failure is the start's).
  // ai: Its configuration is this run's only if the run is still this one when it comes back.
  if (wantGpu()) {
    let c = null;
    try { c = await gpuConfigure(s, codes, fps, gap); } catch (e) {
      if (my !== gen) return;
      if ($("enc").value === "gpu") throw e;
      console.warn("sender: the GPU encoder:", e);
      try { genc?.destroy(); } catch {}
      genc = gctx = null; gpuOff = String(e?.message ?? e);
    }
    if (my !== gen) { if (c) genc?.release(c); return; }
    gcfg = c;
  }
  // ai: What is painted, for the development log only: nothing reads it to decode. transfer tells one start from the
  // ai: next in the development log; a re-pick keeps it.
  cfg = { spec: s, label: made.label, mode: file ? "file" : "test", usefulBytes: made.usefulBytes, blocksPerFrame: made.blocksPerFrame, codes, gap, fps: +$("fps").value, encoder: gcfg ? "gpu" : "wasm", transfer: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`, ...(file ? { fileBytes: file.size, name: file.name, ...(file.bundled ? { files: file.files } : {}) } : {}) };
  shown = s; lap = made.xfer?.lap ?? 0; dataSent = 0; dataWin = 0; fileBytes = file?.size ?? 0; sentBytes = made.xfer?.sent ?? 0;
  showEnc();
  showBlocks();
  // ai: sent: the file's bytes as they go, each chunk zstd-compressed where that is shorter (sim/xfer.mjs, 2026-10-05)
  const xferLine = made.xfer ? `\nfile ${file.size} B${made.xfer.sent !== file.size ? `, ${made.xfer.sent} B as sent (zstd)` : ""}: ${made.xfer.chunks} chunk${made.xfer.chunks === 1 ? "" : "s"} of 2^${made.xfer.chunkLog2}, ${made.xfer.manifest} manifest block${made.xfer.manifest === 1 ? "" : "s"}, BLAKE3 ${made.xfer.root.slice(0, 16)}..., header in the light` : "";
  // What the window chose, so the guess is visible rather than implied. The capture is an assumption, not a measurement.
  const auto = () => {
    const r = room(), squeezed = SAMPLES(shown.n, ringOf(shown)) > r;
    const how = $("subch").value !== "auto" ? ", set by hand" : `, chosen from a ${r.toFixed(0)} px window and nothing else`;
    return `\n${NAME(shown.subch)} (${blocksFor(shown.subch)} blocks), picture ${shown.n} samples in the ${RINGS[ringOf(shown)]}-cell ring${how}` +
      (squeezed ? `\nshown at ${(r / SAMPLES(shown.n, ringOf(shown))).toFixed(2)} device px a sample, so the page is downscaling. That is fine while it AVERAGES; painting nearest drops samples.` : "");
  };
  // ai: The test stream starts at a random id each start (under 2^30, clear of the control ids), so a receiver that saw an
  // ai: earlier start reads the new one as new without being told: the blind receiver's dedupe has no other signal. A
  // ai: file's ids are its transfer's (sim/xfer.mjs), told apart by its header's root.
  seq = 0; nextId = cfg.mode === "test" ? Math.floor(Math.random() * 2 ** 30) : 0; image = null; painted = 0; paintMs = encMs = 0; lastReport = performance.now(); lastTick = 0; lastPaint = 0; held.fill(0); rafGaps.length = 0; drawFps = 0;
  let next = performance.now(), samples = 0;
  const tick = (now) => {
    timer = requestAnimationFrame(tick);
    // ai: A lost device or a GPU error: the run gives the GPU up if it runs on it; an idle encoder is only dropped.
    if (genc && (genc.lost || genc.error)) {
      if (gcfg) { gpuGiveUp(new Error(genc.lost ? `device lost: ${genc.lost}` : genc.error)); return; }
      try { genc.destroy(); } catch {}
      genc = gctx = null;
    }
    // ai: This animation frame's encode time: half the refresh (measured, a running mean).
    if (lastTick && now > lastTick) { const dt = Math.min(now - lastTick, 100); tickMs = 0.9 * tickMs + 0.1 * dt / Math.max(1, Math.round(dt / tickMs)); if (!paused && now - lastTick < 250) rafGaps.push(now - lastTick); }
    lastTick = now; tickBudget = tickMs / 2;
    showRail();
    if (paused) { next = now; return; }
    // Keep AHEAD frames asked for. Not while a re-pick is in flight: a frame's ids are counted in the code it will
    // be encoded with, and the worker only has the new one after it answers.
    // ai: Under the GPU encoder a whole batch at a time, two outstanding, each remembering the configuration and the
    // ai: half it was asked under (a half is asked for again only once its last frame is presented).
    if (gcfg) while (!respecPending && reqSeq - seq <= gcfg.batch) {
      const n = gcfg.batch, buf = spare.pop();
      batchOf.set(reqSeq, { cfg: gcfg, half: gcfg.half, count: n });
      gcfg.half ^= 1; gcfg.pending += n;
      sw.postMessage({ type: "frames", gen, seq: reqSeq, count: n, baseId: nextId, buf }, buf ? [buf] : []);
      reqSeq += n; nextId += n * cfg.blocksPerFrame * cfg.codes; inflight++;
    }
    else while (!respecPending && inflight + ahead.size < AHEAD) {
      const buf = spare.pop();
      sw.postMessage({ type: "frame", gen, seq: reqSeq++, baseId: nextId, buf }, buf ? [buf] : []);
      nextId += cfg.blocksPerFrame * cfg.codes; inflight++;
    }
    const period = 1000 / Math.max(1, +$("fps").value);
    if (!due(now, next, period)) { pump(); return; }
    stateFps();
    // Not encoded yet: paint it on the next animation frame rather than skip it, and count that.
    const fr = ahead.get(seq);
    if (!fr) { behind++; pump(); return; }
    ahead.delete(seq); seq++;
    next = Math.max(next + period, now - period);
    if (lastPaint) held[Math.min(4, Math.max(1, Math.round((now - lastPaint) / tickMs))) - 1]++;
    lastPaint = now;
    encMs += fr.ms;
    if (fr.gcfg) { if (!paintGPU(fr)) return; } else paintRGBA(fr);
    painted++; dataSent += fr.data; dataWin += fr.data;
    if (!pump()) return;
    samples = fr.w;
    // ai: the first frame painted: a sender has seen what the tips say (ui.mjs), so they go on every page of this origin
    // ai: the file's size and, where compression shrank it, the bytes that actually go (2026-10-05)
    if (!live) { live = true; showChoice(true); const sz = sizeLine(); sendingLine = file ? () => sayName($("state"), "Sending ", what, sz) : () => say($("state"), `Sending ${what}`); sendingLine(); tipsDone(); }
    if (now - lastReport > 1000 && painted) {
      const g = gcfg ? genc.takeGpuMs() : null;
      gpuMs = gcfg ? g ?? gpuMs : null;
      const how = gcfg ? `blocks ${(encMs / painted).toFixed(1)} ms in the worker + GPU encode ${gpuMs == null ? "untimed" : `${gpuMs.toFixed(2)} ms`} + present ${(paintMs / painted).toFixed(1)} ms` : `encode ${(encMs / painted).toFixed(1)} ms in the worker + paint ${(paintMs / painted).toFixed(1)} ms`;
      const st = gcfg ? genc.takeStageMs() : null;
      const encLine = (gcfg ? `encoder: WebGPU, ${genc.adapterName}` : `encoder: wasm in the worker${gpuOff && $("enc").value !== "wasm" ? ` (WebGPU off: ${gpuOff})` : ""}`) + (st ? `\nGPU stages, ms a frame: ${Object.entries(st).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")}` : "");
      const dt = (now - lastReport) / 1000, afps = rafRate();
      if (rafGaps.length >= 10) drawFps = afps;
      // ai: which pass of the file shows and how long a pass takes (a lap, at this second's rate of data blocks): #tx's
      // ai: since 2026-10-05 (#nums' until then)
      const passSecs = lap && dataWin ? lap / (dataWin / dt) : 0;
      const passLine = lap ? `\npass ${Math.floor(dataSent / lap) + 1}${passSecs ? `, ${passSecs.toFixed(1)} s a pass` : ""}` : "";
      const line = `${auto().slice(1)}\n${cfg.label}${xferLine}${passLine}\n${samples} x ${samples} samples with the quiet zone${cfg.codes > 1 ? `, ${cfg.codes} side by side` : ""}, ${$("tx").dataset.scale} device px per cell\n${cfg.codes > 1 ? `${cfg.codes} x ` : ""}${cfg.blocksPerFrame} x ${cfg.usefulBytes} B = ${cfg.codes * cfg.blocksPerFrame * cfg.usefulBytes} B per frame\npainted ${(painted / dt).toFixed(1)}/s of ${$("fps").value} asked (${how})` +
        `\nworker ${ahead.size} ready ahead, ${behind} animation frames found nothing ready\nanimation frames ${afps.toFixed(1)} a second, the screen ${(1000 / tickMs).toFixed(1)} Hz by their intervals; pictures held 1, 2, 3, 4+ refreshes: ${held.join(", ")}\n${encLine}`;
      $("tx").textContent = line;
      offeredKBs = (painted / dt) * cfg.codes * cfg.blocksPerFrame * cfg.usefulBytes / 1000;
      // ai: #tx's first line names the format (auto(): "LIZARD-480 (57 blocks), picture ..."; the lab line #lab did until
      // ai: 2026-10-02, repeating #nums' rate and pass: deleted). The user's figure (#nums): the offered rate alone (the
      // ai: pass went to #tx on 2026-10-05).
      $("nums").textContent = rate(offeredKBs);
      dataWin = 0;
      painted = 0; paintMs = encMs = 0; behind = 0; held.fill(0); rafGaps.length = 0; lastReport = now;
    }
  };
  timer = requestAnimationFrame(tick);
  // ai: A room change that came while the start was making its encoder: taken now.
  if (repickLater) { repickLater = false; roomChanged(); }
}
// ai: Halts the stream and clears the canvas; the caller shows the state that follows.
function stop() {
  offeredKBs = 0; drawFps = 0; cancelAnimationFrame(timer); cfg = shown = null; bandFps = 0; running = live = paused = hiddenPause = false; $("go").textContent = "Start";
  showEnc();
  // The worker's frames from this run are dropped by generation; a reply still awaited is refused, not left hanging.
  sw?.postMessage({ type: "stop", gen });
  gen++; if (pending) { pending.rej(STOPPED); pending = null; }
  ahead.clear(); inflight = 0; spare = []; respecPending = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height); image = null;
  // ai: The GPU encoder's configurations go (the device stays for the next start); its canvas is hidden.
  if (genc) for (const c of [...genc.cfgs]) genc.release(c);
  gcfg = null; batchOf.clear(); encodeQueue.length = 0; gcanvas.hidden = true; canvas.hidden = false;
  repickAgain = repickLater = false;
  dropWake();
}

// A resize, a rotation or going fullscreen changes what fits, and both numbers are picked from it.
// ai: Re-pick; the receiver reads the new version from the next frame's word.
// This is safe mid-transfer because block_bytes is the same at every rate
// (src/focus.c: the smallest k / 8 - 4 among the rate profile's codes, held to 473 B), not from n or the sub-channel
// count, so bytes per block does not move and Wirehair, being rateless, does not mind that a frame now carries a
// different number of blocks. The check below refuses the change if that ever stops being true.
//
// What Wirehair does mind is an id it has already seen. blocksPerFrame moves here, so the id
// ai: comes from nextId rather than seq * blocksPerFrame. The worker builds the new code.
// ai: One re-pick at a time (2026-09-29: the GPU encoder's configure made the window seconds wide): a room
// ai: change while one runs is taken once it ends (repickAgain), and one while a start is still making its encoder,
// ai: before there is a cfg, once the start ends (repickLater, start()), so neither is lost; a re-pick whose run was
// ai: stopped or started again meanwhile (gen) keeps nothing it made.
let repicking = false, repickAgain = false, repickLater = false;
async function repick() {
  // ai: The layout is re-applied whatever happens: layout() only re-sizes the canvas when the sample count changes, so
  // ai: a room that moved without changing the version would otherwise keep the old CSS size, and shrinking that way
  // ai: is what squeezes the picture.
  image = null;
  if (!cfg) { if (running) repickLater = true; return; }
  if (repicking) { repickAgain = true; return; }
  repicking = true;
  try { await repickOnce(); } finally { repicking = false; }
  if (repickAgain && running) { repickAgain = false; await repick(); }
}
async function repickOnce() {
  const my = gen, s = spec(), codes = codesOf(), gap = gapOf();
  if (s.n === shown.n && s.subch === shown.subch && s.span === shown.span && codes === cfg.codes && (codes < 2 || gap === cfg.gap)) return;   // same code: the layout refresh above was the whole job
  respecPending = true;
  // ai: Under the GPU encoder the new format's configuration is made (and checked, the first time) before a frame is
  // ai: asked for under it; the frames already asked for present from the old one's slots.
  let r, next = null;
  try {
    r = await swCall({ type: "respec", spec: s, codes, gap });
    if (my !== gen) return;
    if (r.ok && gcfg) {
      try { next = await gpuConfigure(s, codes, bandFps, gap); } catch (e) { if (my === gen) gpuGiveUp(e); return; }
      if (my !== gen) { genc?.release(next); return; }
    }
  } finally { if (my === gen) respecPending = false; }
  if (!r.ok) return;   // would break the stream: keep what is running
  if (next) { const old = gcfg; gcfg = next; gpuRelease(old); }
  shown = s;
  showBlocks();
  stateFps(true);
  Object.assign(cfg, { spec: s, label: r.label, blocksPerFrame: r.blocksPerFrame, codes, gap });
}
// One debounced entry point for every way the room can change, so they cannot race.
//   ResizeObserver on <main>: the fullscreen button regrows it by toggling a class, which fires NO resize event; it
//     works today only because it also calls requestFullscreen(), and that is rejected or unsupported on iOS Safari
//     for a non-video element.
//   window resize: browser zoom changes devicePixelRatio, and so room(), without changing clientWidth.
let repickSoon = 0;
const roomChanged = () => { clearTimeout(repickSoon); repickSoon = setTimeout(() => repick().catch((e) => { if (e !== STOPPED) fail("The version could not change", e); }), 300); };
addEventListener("resize", roomChanged);
new ResizeObserver(roomChanged).observe(canvas.parentElement);

// ai: Start as the person presses it (Start, the rail's play): before the first send on this browser, the brightness tip
// ai: (#first; 2026-10-04; the desktop sender's the same): the display's contrast and brightness are the channel's and
// ai: no page sets them (the
// ai: contrast at its maximum took 2:1 past 2.90 MB/s, STATUS "The oscillation in 2:1"). Shown until a send goes from it
// ai: (its Start, kept as send:tipSeen; send:brightnessSeen until 2026-10-23, when the tip gained power saving and was due
// ai: once more), Cancel or Escape leaving it for the next press; ?auto (the checks', the
// ai: rig's), Resume and a running send's restart (the encoder changed, the GPU given up) never show it.
const SEEN = "send:tipSeen";
let seenHere = false;
const brightnessSeen = () => { try { return seenHere || localStorage.getItem(SEEN) === "1"; } catch { return seenHere; } };
const begin = () => start().catch((e) => { if (e !== STOPPED) fail("Could not start", e); });
const go = () => (paused ? setPaused(false) : running || brightnessSeen() ? begin() : $("first").showModal());
$("firstStart").onclick = () => {
  seenHere = true;
  try { localStorage.setItem(SEEN, "1"); } catch {}
  $("first").close();
  begin();
};
$("firstCancel").onclick = () => $("first").close();
$("go").onclick = go;
$("stop").onclick = () => { stop(); showReady(); };
$("railSend").onclick = () => (running ? setPaused(!paused) : go());
// ai: The selection (2026-10-23; until then a pick replaced the last): items added by each pick, drop or share, in
// ai: order, each a file or a folder with its entries under its own name (tar.mjs bundle's), so files and folders mix.
// ai: The payload is their bundle (one file alone is itself, else one tar). An item already listed is not added again
// ai: (a file by its name, size and time, a folder by its name, files and bytes). Every change stops a run first and
// ai: makes the transfer again at once in the worker (prepare: every chunk compressed), so the state line says what
// ai: goes before Start; a reply for a selection since changed is ignored by its pick number.
let picks = 0, items = [];
function itemOf(name, entries) {
  const files = entries.filter((e) => !e.dir), size = files.reduce((n, f) => n + f.blob.size, 0);
  const dir = entries.length !== 1 || !!entries[0].dir || entries[0].path.includes("/");
  return { name, dir, files: files.length, size, entries, key: dir ? `d|${name}|${files.length}|${size}` : `f|${name}|${size}|${entries[0].mtime}` };
}
function add(fresh) {
  const have = new Set(items.map((i) => i.key)), more = fresh.filter((i) => !have.has(i.key) && have.add(i.key));
  if (!more.length) { if (!running) showReady(); return; }
  items = [...items, ...more];
  changed();
}
function remove(key) { items = items.filter((i) => i.key !== key); changed(); }
function clearPick() { items = []; changed(); }
function changed() {
  if (running) stop();
  $("payload").value = "file";
  const p = items.length ? bundle(items.flatMap((i) => i.entries)) : null;
  chosen = p;
  if (p) { p.pick = ++picks; p.sent = 0; p.failed = null; }
  else ++picks;
  showReady();
  if (p) p.posted = prepare(p);
}
// ai: the chosen file's bytes, with the page's words for one the browser no longer reads
const readOf = (file) => file.blob.arrayBuffer().then((b) => new Uint8Array(b), (e) => {
  throw e?.name === "NotReadableError" || e?.name === "NotFoundError" ? new Error(`${file.bundled ? `A file in ${file.name}` : file.name} has changed, moved or gone since it was chosen: choose it again`) : e;
});
async function prepare(p) {
  let b;
  try { b = await readOf(p); } catch (e) { p.failed = e; return; }   // ai: said at Start, which reads it again
  if (p !== chosen) return;
  swInit();
  sw.postMessage({ type: "prepare", pick: p.pick, bytes: b.buffer, xfer: { name: p.name, type: p.type, chunkLog2: CHUNK_LOG2 } }, [b.buffer]);
}
function madeFor(m) {
  const p = chosen;
  if (!p || p.pick !== m.pick) return;
  if (m.error) { p.failed = new Error(m.error); return; }
  p.sent = m.sent;
  if (!running) showReady();
}
// ai: The list over <main> (#pick) while the selection holds something and no send paints, the hint while it holds
// ai: nothing: a row an item (its icon, its name, "N files · size" for a folder or its size, Remove), then Add files,
// ai: Add a folder and Clear.
function renderPick() {
  const list = $("pickList");
  list.replaceChildren(...items.map((it) => {
    const li = document.createElement("li"), lead = document.createElement("span"), body = document.createElement("span");
    const t = document.createElement("span"), sub = document.createElement("span"), rm = document.createElement("button");
    li.className = "item";
    lead.className = "lead"; lead.append(icon(it.dir ? "folder" : "file"));
    body.className = "body"; t.className = "t"; t.textContent = it.name; t.title = it.name; sub.className = "s";
    sub.textContent = it.dir ? `${it.files} ${it.files === 1 ? "file" : "files"} · ${bytes(it.size)}` : bytes(it.size);
    body.append(t, sub);
    rm.type = "button"; rm.className = "btn icon"; rm.title = `Remove ${it.name}`; rm.append(icon("x"));
    rm.onclick = (e) => { e.stopPropagation(); remove(it.key); };
    li.append(lead, body, rm);
    return li;
  }));
}
function showChoice(painting) {
  $("hint").hidden = painting || isTest() || items.length > 0;
  $("pick").hidden = painting || isTest() || items.length === 0;
  renderPick();
}
// ai: An input's files are taken as they are picked and the input emptied, so the same pick again is a new pick. Each
// ai: file picked is an item; a folder's files carry their path under it (webkitRelativePath), one item, whose name is
// ai: its first part (a browser lists no empty folder there).
$("file").onchange = () => {
  const fs = [...$("file").files];
  $("file").value = "";
  add(fs.map((f) => itemOf(f.name, [{ path: f.name, blob: f, mtime: f.lastModified / 1000 }])));
};
$("folder").onchange = () => {
  const fs = [...$("folder").files];
  $("folder").value = "";
  if (!fs.length) { if (!running) showReady(); return; }
  const entries = fs.map((f) => ({ path: f.webkitRelativePath || f.name, blob: f, mtime: f.lastModified / 1000 }));
  add([itemOf(entries[0].path.split("/")[0], entries)]);
};
// ai: Files from outside the page (2026-10-23), each added as a file picked: what another app shared to the installed
// ai: app (the worker keeps them in the cache "lizard-share" and opens send.html?shared: pwa/service-worker.js shareIn),
// ai: and the system's Open with (the manifest's file_handlers, handed over through launchQueue).
async function takeShared() {
  if (!new URLSearchParams(location.search).has("shared") || typeof caches === "undefined") return;
  history.replaceState(null, "", location.pathname);
  const cache = await caches.open("lizard-share"), got = [];
  for (const req of await cache.keys()) {
    const r = await cache.match(req);
    if (r) {
      const name = decodeURIComponent(r.headers.get("x-name") || "file"), at = +r.headers.get("x-modified") || Date.now();
      const f = new File([await r.blob()], name, { type: r.headers.get("content-type") || "", lastModified: at });
      got.push(itemOf(name, [{ path: name, blob: f, mtime: at / 1000 }]));
    }
    await cache.delete(req);
  }
  if (got.length) add(got);
}
const canFolders = "webkitdirectory" in $("folder");
$("orFolder").hidden = $("addFolder").hidden = !canFolders;
$("folderBtn").onclick = (e) => { e.stopPropagation(); $("folder").click(); };
$("addFiles").onclick = (e) => { e.stopPropagation(); $("file").click(); };
$("addFolder").onclick = (e) => { e.stopPropagation(); $("folder").click(); };
$("clearPick").onclick = (e) => { e.stopPropagation(); clearPick(); };
$("pick").addEventListener("click", (e) => e.stopPropagation());   // ai: a tap on the list is not a tap on <main>
$("payload").onchange = () => { if (running) stop(); showReady(); };
// ai: Files and folders dropped anywhere on the page are added, each dropped item one. The counter pairs enter with
// ai: leave across child elements, so the outline goes only when the drag leaves the page.
const main = canvas.parentElement, hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
let drags = 0;
document.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); drags++; main.classList.add("drop"); });
document.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
document.addEventListener("dragleave", (e) => { if (hasFiles(e) && --drags <= 0) { drags = 0; main.classList.remove("drop"); } });
document.addEventListener("drop", (e) => {
  e.preventDefault(); drags = 0; main.classList.remove("drop");
  // ai: the entries are taken inside the event: the drop's items are gone once it returns
  const roots = [...(e.dataTransfer?.items ?? [])].filter((i) => i.kind === "file").map((i) => i.webkitGetAsEntry?.() ?? null);
  const flat = [...(e.dataTransfer?.files ?? [])];
  if (!flat.length && !roots.length) return;
  const walked = roots.length && roots.every(Boolean) ? walkDrop(roots)
    : Promise.resolve(flat.map((f) => itemOf(f.name, [{ path: f.name, blob: f, mtime: f.lastModified / 1000 }])));
  if (!running) say($("state"), "Reading the files");
  // ai: a drop that cannot be read leaves the list and a run as they were, and says so where nothing is sending
  walked.then(add, (err) => {
    console.error("sender: the dropped files could not be read:", err);
    if (!running) { showReady(); say($("state"), `The dropped files could not be read: choose them instead: ${err?.message ?? err}`, "bad"); }
  });
});
// ai: A drop's items, each dropped root with every folder walked, folders included (an empty one survives), under its
// ai: own name. readEntries hands a folder's listing over in pieces, so it is asked until it hands none.
async function walkDrop(roots) {
  const listAll = async (dir) => {
    const r = dir.createReader(), all = [];
    for (;;) { const got = await new Promise((ok, no) => r.readEntries(ok, no)); if (!got.length) return all; all.push(...got); }
  };
  const walk = async (en, path, out) => {
    if (en.isFile) { const f = await new Promise((ok, no) => en.file(ok, no)); out.push({ path, blob: f, mtime: f.lastModified / 1000 }); }
    else if (en.isDirectory) { out.push({ path, dir: true }); for (const c of await listAll(en)) await walk(c, `${path}/${c.name}`, out); }
  };
  const got = [];
  for (const r of roots) { const out = []; await walk(r, r.name, out); if (out.length) got.push(itemOf(r.name, out)); }
  return got;
}

// ai: Fullscreen: body.full shows <main> alone (the room grows, the ResizeObserver re-picks), and the page asks for
// ai: real fullscreen where there is one. Its end, by Escape or the browser, clears body.full through fullscreenchange
// ai: (until 2026-09-26 the first Escape left body.full set); a tap on the code or Escape leaves too. Where there is no
// ai: Fullscreen API (an iPhone) or it is refused, body.full alone is the fallback.
const fsElement = () => document.fullscreenElement ?? document.webkitFullscreenElement ?? null;
function enterFull() {
  document.body.classList.add("full"); image = null;
  const el = document.documentElement, ask = el.requestFullscreen ?? el.webkitRequestFullscreen;
  try { Promise.resolve(ask?.call(el)).catch(() => {}); } catch {}
}
function leaveFull() {
  document.body.classList.remove("full"); image = null;
  const quit = document.exitFullscreen ?? document.webkitExitFullscreen;
  if (fsElement()) try { Promise.resolve(quit?.call(document)).catch(() => {}); } catch {}
}
for (const ev of ["fullscreenchange", "webkitfullscreenchange"]) document.addEventListener(ev, () => { if (!fsElement()) leaveFull(); });
$("fs").onclick = enterFull;
// ai: the sidebar collapser (ui.mjs): <main> grows, and the ResizeObserver re-picks as for Fullscreen
collapser($("collapse"), "send:collapsed");
sideResizer("send:side");   // ai: the column's edge dragged to resize it (2026-10-05); <main> re-picks as for a resized window
// ai: A tap on the middle leaves fullscreen, or, while its hint asks for a file, opens the picker (the page's only one).
main.addEventListener("click", () => { if (document.body.classList.contains("full")) leaveFull(); else if (!$("hint").hidden) $("file").click(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && document.body.classList.contains("full")) leaveFull(); });
// ai: Another encoder asked for: the run starts again under it, and auto tries the GPU again.
$("enc").onchange = () => { gpuOff = null; showEnc(); if (running) go(); };
// ai: One code or two changes the room a symbol has, so it re-picks as a resize does; so does another version or ring
// ai: (2026-10-01: until then they waited for the next start or resize).
$("codes").onchange = () => { image = null; roomChanged(); };
$("subch").onchange = $("ring").onchange = () => roomChanged();
// ai: the display fps slider's label (2026-10-02, a slider 1 to 60 in place of the number box): live while it is dragged,
// ai: and the page's pace and the word's rate follow it as they followed the box (both read #fps as they go)
const showFps = () => { $("fpsOut").textContent = `${+$("fps").value}`; };   // ai: "FPS" over the bare count (2026-10-05; "Pictures a second", "60 a second" before)
for (const ev of ["input", "change"]) $("fps").addEventListener(ev, showFps);
// ai: the size slider (sizeOf): its label live while dragged, the room changed on release
const showSize = () => { $("sizeOut").textContent = `${Math.round(sizeOf() * 100)}%`; };
$("size").addEventListener("input", showSize);
$("size").addEventListener("change", () => { showSize(); roomChanged(); });
// ai: the margin slider (marginOf): its label live while dragged, a new canvas and the room changed on release
const showMargin = () => { $("marginOut").textContent = `${marginOf()} modules`; };
$("margin").addEventListener("input", showMargin);
$("margin").addEventListener("change", () => { showMargin(); image = null; roomChanged(); });
// ai: The app's shell (2026-10-01): the bar's Settings, About and the installed app's worker.
about($("about"));
registerApp(() => about($("about"), true));

// The surround: clutter painted on the canvas behind the symbol, filling <main> up to the symbol's own painted
// margin. A busy page is the channel the decoder is judged on (the training scenes carry the same kinds,
// gpu/cnn/assemble.mjs), and every recorded run so far had a white surround, so this is how a recording gets
// the honest version. Drawn from a fixed seed (?bgseed=n) so a recording's surround can be made again; redrawn
// when the room changes. The symbol's canvas is above it and untouched.
const bgc = $("bgc"), bgx = bgc.getContext("2d");
function paintSurround() {
  const kind = SURROUND, box = bgc.parentElement, dpr = devicePixelRatio || 1;
  const W = Math.max(1, Math.round(box.clientWidth * dpr)), H = Math.max(1, Math.round(box.clientHeight * dpr));
  bgc.width = W; bgc.height = H;
  bgx.filter = "none"; bgx.fillStyle = "#fff"; bgx.fillRect(0, 0, W, H);
  if (kind === "none") return;
  let s = (+(new URLSearchParams(location.search).get("bgseed")) || 1) >>> 0;
  const rnd = () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 8) / 0x1000000;
  const grey = (v) => `rgb(${Math.round(255 * v)},${Math.round(255 * v)},${Math.round(255 * v)})`;
  // Sizes follow the symbol on screen: a corner mark is 12 modules, and a module is about W / 200 at any window.
  const mark = Math.max(6, Math.round(W / 16));
  const one = (k) => {
    if (k === "noise") {
      const img = bgx.createImageData(W, H), amp = 0.5 + 0.5 * rnd();
      for (let i = 0; i < W * H; i++) { const v = Math.round(255 * (1 - amp * rnd())); img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255; }
      bgx.putImageData(img, 0, 0);
    } else if (k === "text") {
      const line = Math.round(mark * (0.4 + 0.6 * rnd())), tall = Math.max(2, Math.round(line / 2.8));
      bgx.fillStyle = grey(0.1);
      for (let y = 4; y < H - 4; y += line) { let x = 4; while (x < W - 3 * line) { const w = tall + Math.floor(rnd() * 4 * line); bgx.fillRect(x, y, w, tall); x += w + tall + Math.floor(rnd() * tall); } }
    } else if (k === "checker") {
      const b = Math.round(mark * (0.2 + 1.5 * rnd()));
      for (let y = 0; y < H; y += b) for (let x = 0; x < W; x += b) { bgx.fillStyle = grey(rnd() < 0.5 ? 0.06 : 0.95); bgx.fillRect(x, y, b, b); }
    } else if (k === "squares") {
      for (let k2 = 0; k2 < 60; k2++) { const sz = Math.round(mark * (0.3 + 1.5 * rnd())); bgx.fillStyle = grey(0.3 * rnd()); bgx.fillRect(Math.floor(rnd() * (W - sz)), Math.floor(rnd() * (H - sz)), sz, sz); }
    } else if (k === "photo") {
      // Noise at three scales, blurred: blobs, edges and gradients at every size a mark can have.
      for (const [g, a] of [[Math.round(mark * 2), 0.5], [Math.round(mark / 2), 0.3], [Math.max(2, Math.round(mark / 8)), 0.2]]) {
        for (let y = 0; y < H; y += g) for (let x = 0; x < W; x += g) { bgx.fillStyle = `rgba(0,0,0,${(a * rnd()).toFixed(3)})`; bgx.fillRect(x, y, g, g); }
      }
      bgx.filter = `blur(${(mark / 12).toFixed(1)}px)`; bgx.drawImage(bgc, 0, 0); bgx.filter = "none";
    }
  };
  if (kind === "mixed") {
    // A dense kind (covers every pixel) goes under a sparse one, or the second would just replace the first.
    const dense = ["noise", "checker", "photo"], sparse = ["text", "squares"];
    one(dense[Math.floor(rnd() * dense.length)]); one(sparse[Math.floor(rnd() * sparse.length)]);
  } else one(kind);
}
new ResizeObserver(paintSurround).observe(bgc.parentElement);
// ai: Any control can be preset from the URL: send.html?payload=test&subch=512&fps=30&auto (&bg=noise&bgseed=3 for a
// ai: surround). The file input is not one: a page cannot be handed a file by its URL.
// ai: Settings' controls as the user last left them (localStorage, since 2026-09-29), then
// ai: the URL's presets over them for this load.
// ai: and Developer Tools' payload (there since 2026-10-03), under the same keys
for (const el of document.querySelectorAll("#dev select, #dev input, #logs select")) if (el.type !== "file" && !("view" in el.dataset)) persist(el, `send:${el.id}`);
// ai: a menu takes a preset only where it is one of its options (?ring=auto, from before the menu lost its auto, keeps the default)
for (const [k, v] of URLP) { const el = $(k); if (el && "value" in el && el.type !== "file" && (el.tagName !== "SELECT" || [...el.options].some((o) => o.value === v))) el.value = v; }
// ai: A preset or a kept value that is no whole number of blocks from 1 to 128 (?subch=20) is no symbol: back to auto,
// ai: and say so, rather than paint it.
if ($("subch").value !== "auto" && !VERSIONS.includes(+$("subch").value)) { $("tx").textContent = `${$("subch").value} sub-channels is no whole number of blocks (8 each, 8 to 1024): showing auto`; $("subch").value = "auto"; }
showBlocks();
showEnc();
showFps();
showSize();
showMargin();
remember($("dev"), "send:dev");
remember($("logs"), "send:logs");
paintSurround();
showReady();
if (URLP.has("auto")) begin();
// ai: what was shared or opened with the app, once the page is set up (takeShared above)
takeShared().catch((e) => console.error("sender: the shared files could not be read:", e));
if ("launchQueue" in window) window.launchQueue.setConsumer(async (params) => {
  const fs = await Promise.all((params.files ?? []).map((h) => h.getFile()));
  if (fs.length) add(fs.map((f) => itemOf(f.name, [{ path: f.name, blob: f, mtime: f.lastModified / 1000 }])));
});
