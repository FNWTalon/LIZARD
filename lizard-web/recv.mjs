// Receiver page: camera frames to a pool of decode workers. A frame that arrives while every worker
// is busy is skipped, not queued: a stale frame is worth less than the next one.
//
// The pool starts at one worker and grows only while frames are being lost to busy workers, up to
// navigator.hardwareConcurrency: a phone that keeps up with one worker runs one, and stays cooler.
// What is shared between workers lives here: the set of block ids already seen (two workers often
// decode two captures of the same display frame) and, through one fountain worker, the file.
// ai: The decoder (Settings: GPU or CPU, auto while neither is chosen; ?dec=) swaps the pool for one GPU worker (recv-gpu-worker.mjs, the whole decode on WebGPU,
// ai: gpu/). It is sent every frame and batches them; its results take the same path as the pool's.
// ai: The whole receiver is blind (2026-09-26): both decoders are told nothing, each frame is read at the picture
// ai: size and version its own word names, its blocks judged from the light (a test frame is known by its blocks'
// ai: bytes), nothing carries between frames beyond a batch, and a file comes from the light's header. Nothing is fetched
// ai: from a server: stats rows and received files are posted as development logs (devlog.mjs), best effort,
// ai: never awaited.
// One canvas, #shot, is both what is on screen and what is read back for the decoder. It used to be two answers:
// the screen got the <video> under CSS object-fit: cover, resolved by the compositor, and the decoder got a
// drawImage source rect, resolved by the canvas. On Firefox for Android above 720p they disagreed, so the square
// aimed with was not the square decoded (another project's scanner had the same bug on another engine). Which of the
// two was wrong cannot be told from JS and does not matter: a crop the preview cannot show cannot exist when the
// preview is the crop.
// ai: Neither is needed where the camera's frame goes over as a VideoFrame (vframes.mjs): to the GPU worker under F0,
// ai: or to a pool worker, which copies the crop to luma itself (recv-worker.mjs lumaOf). Then the camera element is the
// ai: preview and the canvas is drawn only wherever no VideoFrame goes (preview). Under the GPU decoder on Chrome the
// ai: worker reads the camera track itself (sendTrack): this page's callback then only counts frames.
import { makeGlGrab } from "./glgrab.mjs";
import { frameTap, cornerProbe } from "./vframes.mjs";
import { verdict } from "./verdict.mjs";
import { PoolPolicy } from "./pool.mjs";
import { logPost } from "./devlog.mjs";
import { R_RING, SPAN, MODULES, N_FOR, NAME, FOCUS_BITMAP, PICTURE_SIZES } from "../liblizard/sim/lizard_pick.mjs";
import { isControlId, PAYLOAD, fractionDone } from "../liblizard/sim/xfer.mjs";
import { remember, persist, restoreSaved, say, meter, bytes, rate, left, about, registerApp, collapser, sideResizer } from "./ui.mjs";
import { open as openKept, list as listKept, watch as watchLibrary } from "./library.mjs";
const $ = (id) => document.getElementById(id), video = $("v"), shot = $("shot"), sctx = shot.getContext("2d", { willReadFrequently: true });
// ai: Settings and Developer Tools (the Developer panel until 2026-10-01) open as they were left, a phone session
// ai: being a series of reloads; restored here, at the top of the module, so nothing is seen to move. Errors a user must
// ai: see go to #state instead.
remember($("dev"), "recv:dev");
remember($("logs"), "recv:logs");
collapser($("collapse"), "recv:collapsed");   // ai: the sidebar collapser (ui.mjs); the camera box takes the width
// ai: The column's floor (ui.mjs sideResizer, 2026-10-06): what the actions need in their two-column form (recv.html
// ai: #top) with no word broken: Open over Save beside Share over Start camera, each label in the buttons' font plus
// ai: their padding, the gap between the columns, and the column's own padding. Every label counts whether or not a
// ai: file is in, so the column does not move when one arrives; the labels are measured on a canvas, so a hidden
// ai: button measures too ("Start camera" stands for #go's two, the longer).
function actionsFloor() {
  const b = getComputedStyle($("go")), side = getComputedStyle($("side")), top = getComputedStyle($("top"));
  const c = (actionsFloor.ctx ??= document.createElement("canvas").getContext("2d"));
  c.font = `${b.fontStyle} ${b.fontWeight} ${b.fontSize} ${b.fontFamily}`;
  const w = (label) => c.measureText(label).width + parseFloat(b.paddingLeft) + parseFloat(b.paddingRight);
  return Math.ceil(Math.max(w("Open"), w("Save")) + parseFloat(top.columnGap) + Math.max(w("Share"), w("Start camera")) + parseFloat(side.paddingLeft) + parseFloat(side.paddingRight));
}
sideResizer("recv:side", actionsFloor);   // ai: the column's edge dragged to resize it (2026-10-05), no narrower than the actions need
// ai: The app's shell (2026-10-01): the bar's Settings, the first-run tip, About and the installed app's worker.
about($("about"));
registerApp(() => about($("about"), true));
const PARAMS = new URLSearchParams(location.search), VERIFY = PARAMS.has("verify"), PROF = PARAMS.has("prof");
// ai: The lab's switches that were test-only menus until 2026-10-01, URL parameters alone since, each read where its
// ai: menu was: grab=gl, srccrop=full, workers=<n>, and the
// ai: GPU worker's gprec=int8|f16|f32, gsg=off, gtest=nocascade|b1, gcancel=on, gsrc=page, gstages=on. The stale stored
// ai: values of those menus go, so none comes back if a menu ever does. The GPU self-test went with them
// ai: (gpu_selftest.html by its URL). The page records no replays since 2026-10-09: the Android app's Save replays does.
const LAB = (k, d) => PARAMS.get(k) ?? d;
for (const k of ["gprec", "gsg", "gtest", "gcancel", "gsrc", "gstages", "grab", "srccrop", "workers"]) try { localStorage.removeItem(`recv:${k}`); } catch {}
// ai: A chunked file's header comes in the light, and only from there. ?save=post hands a finished file to the
// rig (lizard-web/server.mjs /api/file) for a scripted run to check (lizard-web/check_rates.mjs).
// ?store=memory keeps a chunked transfer's blocks in memory even where OPFS is there, to measure what OPFS saves.
const SAVE_POST = PARAMS.get("save") === "post", STORE = PARAMS.get("store") === "memory" ? "memory" : undefined;
const CORES = navigator.hardwareConcurrency || 4, DELAY = +PARAMS.get("delay") || 0;
// The crop is the centred square: the code is square, and the sides of a 16:9 frame are 44% of its pixels.
// ?crop=full takes the whole frame, which lizard-web/check_rates.mjs uses to check the crop from outside.
const CROP = PARAMS.get("crop") === "full" ? "full" : "square";
// Which browser this run came from. research/rig/stats.jsonl held 4130 rows of phone data and nothing said, so no
// question about one browser against another could be answered from it afterwards. Short on purpose: the
// engine and its version are what a run needs to be sorted by, and the rest is noise in every row.
const UA = (() => {
  const u = navigator.userAgent;
  const m = /(Firefox|SamsungBrowser|Edg|OPR)\/([\d.]+)/.exec(u) || /(Chrome)\/([\d.]+)/.exec(u) || /Version\/([\d.]+).*(Safari)/.exec(u);
  const name = m ? (m[1] === "Edg" ? "Edge" : m[1] === "OPR" ? "Opera" : m[1]) : "unknown";
  return `${name} ${m ? (m[2] ?? m[1]) : ""}${/Android/.test(u) ? " Android" : /iPhone|iPad/.test(u) ? " iOS" : ""}`.trim();
})();
// A per-video-frame callback or the display's: the two hand out a very different number of chances at one
// camera frame, and the pool's behaviour when it is full depends on which.
const VFC = typeof HTMLVideoElement !== "undefined" && !!HTMLVideoElement.prototype.requestVideoFrameCallback;
// The GL arm (glgrab.mjs) takes the frame through WebGL2 instead of drawImage + getImageData: the luma
// conversion happens in a fragment shader and 518 KB comes back through a fence where 2.07 MB of RGBA came back
// synchronously. ?glcs=default lets the browser colour-manage the upload, which is the one thing on that path
// that can move a pixel's value; the arms are meant to be compared, so it is a switch and not a decision.
//
// It is a switch (a menu until 2026-10-01, the URL's grab=gl since) and not a build, because the question it exists
// for is about a WARM phone: one that has been running a minute is a different machine from a cold one, and switching arms in place
// compares them on the same machine instead of across two runs with a cooldown in between.
//
// The arm is NOT chosen automatically, and that is deliberate. It was tried: the page ran both for 1.5 s and
// kept whichever delivered more frames a second, which picks correctly and would be right for something being
// used rather than measured. It is wrong here. This is an experiment, so which arm produced a figure has to be
// a fact about the run and not the outcome of a three-second race that could have gone the other way, and a
// run whose arm changed under it is a run that cannot be compared with any other. The grab costs 40 to 60 ms
// on Chrome for Android against single digits on Firefox (STATUS.md), so on that browser the GL arm is what to
// ask for; asking is the point.
// glGrab holds the GL arm (lizard-web/glgrab.mjs) while it is the one running, so glFrame below drives it; the WebGPU
// arm that shared its shape was deleted on 2026-09-23 (STATUS.md, "The WebGPU decoder archived").
let glGrab = null, webglGrab = null, grabMode = "canvas", grabNote = "";
// Which arm to grab with is asked for, never guessed from the browser's name. The canvas grab
// is fine on some phones and ruinous on others: `willReadFrequently` is a hint, and a canvas that is also on
// screen is the one a browser is most likely to keep GPU-backed, where every getImageData stalls the pipeline
// to pull the surface back. Measured on a phone, the canvas arm cost the frame on Chrome and the GL arm did
// not; on Firefox, the browser every figure in STATUS.md came from, the canvas arm is what those figures used.
//
// So neither browser is named: the canvas arm runs unless `?grab=gl` asks for the GL one (the calibration that ran
// both and kept the cheaper, above, is gone), which is what an experiment comparing the arms wants.
function setGrab(want) {
  if (want === "gl" && !webglGrab) {
    webglGrab = makeGlGrab(video, { colorSpace: PARAMS.get("glcs") ?? "none" });
    // #view's CSS lays out any canvas inside it, so this one needs no rules of its own. #shot stays in the
    // document and only stops being shown: blit, stopCamera and the source-rect check still refer to it.
    if (webglGrab) { webglGrab.canvas.id = "glshot"; $("view").appendChild(webglGrab.canvas); }
  }
  const arm = want === "gl" ? webglGrab : null;
  glGrab = arm && !arm.lost ? arm : null;
  grabMode = glGrab ? want : "canvas";
  grabNote = want !== "canvas" && grabMode === "canvas" ? "  (asked for the GL grab; this browser has no WebGL2)" : "";
  shot.style.display = grabMode === "canvas" ? "" : "none";
  if (webglGrab) { webglGrab.canvas.style.display = webglGrab === glGrab ? "" : "none"; if (webglGrab !== glGrab) webglGrab.reset(); }
  // The rates and the best-of belong to the arm that produced them. best.grab is the unthrottled reference the
  // ai: readout shows beside the live figure, so carrying it across a switch would compare one arm's heat with the
  // other's cold start.
  cur = fresh(); buckets = []; best = freshBest();
  if (track && video.videoWidth) { if (glGrab) glGrab.paint(cropRect()); else blit(sctx, cropRect()); lastPaint = performance.now(); }
}
const pool = [], fountain = new Worker(new URL("./fountain-worker.mjs", import.meta.url), { type: "module" });
// ai: config: what the decoders are told, always set (lizardConfig): nothing but the picture sizes to build.
let config = null, track = null, file = "", next = 0, busySkips = 0;
let photoCaps = {}, levels = null;   // what the camera says it can be told, and the grey range it actually delivers
// The last format word read out of a symbol (src/fmt.h): the version, and the rate the sender means to paint at.
// ai: LIZARD's decoders read everything from the light; this is what the light last said of the format, as shown
// ai: and recorded. It moves only when two results in a row that read a word agree on the new one (bandSeen is the
// ai: last word read), so a stray wrong word (a random frame passes the word check at about 3e-7 at n = 256) is never
// ai: shown and a real change shows two frames later. Decoding is untouched: every frame decodes by its own word.
let band = null, bandSeen = null;
// ai: What the light says: LIZARD's word.
function showCfg() {
  const light = !band ? "no format word read yet" : `${NAME(8 * band.version)}, ${band.fps ? `${band.fps} fps` : "no rate"} from the band`;
  $("cfg").textContent = `light: ${light}`;
}
// ai: What the side column says (showState). err: why a start, or the transfer, failed, a plain sentence kept until the
// ai: next start; starting: a start under way; received: the last file finished ({ name, n, secs, root }); testAt: when a
// ai: frame last showed the test stream; bandAt: when a frame last read a format word; stoppedAt: when the camera last
// ai: stopped (the pause is taken off a resumed transfer's clock, tFirst); lab: the stats tick's lab line (#lab, in
// ai: Developer Tools: the numbers line's until 2026-10-01); why: its hints in plain words.
// ai: fileAt: when a frame last brought blocks that were not the test stream's (2026-10-05: the test stream owns the
// ai: display only while it is the newer of the two)
const ui = { err: "", starting: false, received: null, testAt: -Infinity, fileAt: -Infinity, bandAt: -Infinity, stoppedAt: 0, lab: "", why: "" };
const LIVE_MS = 3000;   // ai: the test stream, or a code's word, is still "being read" this long after its last frame
const recent = (at) => performance.now() - at < LIVE_MS;
// ai: One state at a time, the first that holds: error, starting, loading (the camera on and no decoder answer yet since
// ai: it or the decoder started: `answered`, the box covered by #loading, a loading indicator in place of the camera,
// ai: since 2026-09-29), receiving (a header read, its file not yet offered),
// ai: received (the camera off, or the transfer in hand the file offered), idle (camera off), then scanning: the test
// ai: stream, a code found with no file yet, or looking. Drawn from what the page holds, on every change and once a
// ai: second from the stats tick. The received file's buttons stay whatever the state, until the next file replaces them.
// ai: The words and figures are the Android app's too (Readout.kt; 2026-10-01): the user's figures are progress, size,
// ai: speed and time left; the format's name and the registered
// ai: share are the lab line's, in Developer Tools (#lab).
let stateShown = "";
// ai: New bytes a second over the last closed one-second bucket (KB/s), null before a second has closed (the last two
// ai: buckets from 2026-09-29 to 2026-09-30).
function recentKBs() {
  const last = buckets[buckets.length - 1];
  return last && last.dt >= 0.5 ? Math.round(last.freshBytes / last.dt / 1000) : null;
}
const WEB_FPS = 24;   // ai: the display rate a browser's receiver is for (24 was chosen to avoid phasing against a 60 fps camera)
// ai: verdict.mjs's wording, the lab's (#out, the stats row), put plainly for the page's hints (#why).
function plainVerdict(text) {
  if (/reaching the decoders/.test(text)) return "The camera's pictures are not reaching the reader.";
  if (/not being registered/.test(text)) return "Hold the phone so the code fills the square, steady and in focus.";
  if (/not changing/.test(text)) return "The code on the other screen is not changing: is the sender running?";
  if (/survives FEC/.test(text)) return "The code is seen but cannot be read: move closer, or cut the glare.";
  return text;
}
function showState() {
  const on = !!track, h = xferNow?.header, go = $("go"), offered = !!h && xferNow.done && h.root === ui.received?.root;
  let line, tone = null, frac = null, nums = "";
  const loading = on && !ui.err && !ui.starting && !answered;
  // ai: the test stream first while the camera reads it, the newer of it and a file's frames (2026-10-05): it takes over
  // ai: from a file received or in progress, whose line, buttons and rail squares come back once the camera is on its frames
  const testing = on && recent(ui.testAt) && ui.testAt >= ui.fileAt;
  if (ui.err) { line = ui.err; tone = "bad"; }
  else if (ui.starting) line = "Starting the camera";
  else if (loading) line = "Getting ready";
  else if (testing) line = "Reading the test stream";   // ai: its rate the rail's and Developer Tools' (#lab) since 2026-10-05
  else if (on && h && !offered) {
    frac = fractionDone(xferNow);
    line = `Receiving ${h.name || "a file"}`;
    // ai: the last second's new bytes a second (recentKBs) and nothing else: the average since the first decode (avgAt)
    // ai: stood beside it as "(M KB/s average)" from 2026-09-29 to 2026-09-30 and is the stats row's alone since (avgKBs:
    // ai: after a wait it often read wrong); the time left is what is missing at that rate
    const now = recentKBs();
    // ai: the bytes as sent where the manifest has said them (2026-10-05): what the light carries, which the rate and
    // ai: the time left are of; the file's own bytes until then, and in the received line
    const total = h.sent ?? h.length, got = h.sent ? xferNow.sentIn : frac * h.length;
    // ai: progress, size and time left; the rate itself is the collapsed rail's and Developer Tools' (#lab) since 2026-10-05
    nums = `${Math.floor(100 * frac)}%, ${partOf(got, total)}${now > 0 ? `, ${left((total - got) / (now * 1000))}` : ""}`;
  } else if (ui.received && (!on || offered)) {
    const r = ui.received;
    line = `Received ${r.name}, ${nb(bytes(r.n))} in ${nb(`${r.secs.toFixed(1)} s`)}${r.sent && r.sent < r.n ? `, ${nb(bytes(r.sent))} sent` : ""}`; tone = "good"; frac = 1;
  } else if (!on) line = "";   // ai: idle says nothing (2026-10-02: an instruction that obvious only takes space)
  else line = "Looking for a code";   // ai: a word read with no file yet says this too (2026-10-05; "Found the code, waiting for the file" before)
  // ai: rewritten only when it changes: #state is a live region, and a screen reader reads out every rewrite
  if (`${tone} ${line}` !== stateShown) { stateShown = `${tone} ${line}`; say($("state"), line, tone); }
  meter($("meter"), frac);
  $("nums").textContent = nums;
  $("lab").textContent = on ? ui.lab : "";
  $("why").textContent = on && !ui.err && !loading ? ui.why : "";
  $("deliver").hidden = !ui.received || testing;
  $("idle").hidden = on;
  $("loading").hidden = !loading;
  // ai: one solid button on the page at a time: Start while nothing waits, else Open
  go.textContent = on ? "Stop camera" : "Start camera";
  go.classList.toggle("primary", !on && !ui.received);
  go.disabled = ui.starting;
  // ai: a file's actions first, the camera's button after them, in the order a keyboard tabs as well
  const top = $("top"), last = !$("deliver").hidden;
  if (last ? top.lastElementChild !== go : top.firstElementChild !== go) last ? top.append(go) : top.prepend(go);
  // ai: the collapsed rail (recv.html #rail): the camera's pause or play, the last second's rate while a file or the
  // ai: test stream is read (its figure over its unit), and once the file is in a green check in the rate's square (a
  // ai: mark, no action), then Open (Feather's external-link) and Save (its download), the deliver row's own actions, in
  // ai: black (2026-10-05; the green tick opened the file until then)
  const cam = $("railCam");
  cam.classList.toggle("off", !on); cam.title = on ? "Stop camera" : "Start camera"; cam.disabled = ui.starting;
  const now = (on && h && !offered) || testing ? recentKBs() : null, [v, u] = now != null ? rate(now).split(/\s/) : ["", ""];   // ai: \s takes rate()'s no-break space
  $("railRate").innerHTML = v ? `${v}<small>${u}</small>` : "";
  $("railRate").hidden = !!ui.received && !testing;   // ai: the check takes its square, except while the test stream is read
  $("railDone").hidden = $("railOpen").hidden = $("railSave").hidden = !ui.received || testing;
}
const nb = (s) => s.replace(/ /g, "\u00a0");   // ai: a figure and its unit kept on one line
// ai: "3.1 of 7.4 MB": the part in the unit ui.mjs bytes() gives the whole, at its precision.
function partOf(n, whole) {
  const all = bytes(whole), unit = all.split(" ")[1], v = n / { B: 1, KB: 1e3, MB: 1e6, GB: 1e9 }[unit];
  return `${unit === "B" ? Math.round(v) : v < 9.95 ? v.toFixed(1) : Math.round(v)} of ${nb(all)}`;
}
// Block ids already forwarded. Ids only ever increase (wirehair is rateless, so the sender keeps minting them),
// which used to make this grow for as long as the session lasted: about 500 a second here, and the same again
// inside every worker. Two generations instead of one set, rolled on INSERTION, so what ages out is only ids from
// far enough back that they cannot recur. A frozen sender inserts nothing, so it forgets nothing and still reads
// as repeats. Forwarding a duplicate would be harmless anyway (liblizard/wirehair/shim.cpp: duplicate ids are
// idempotent); what this set is really for is the withNew / repeat split and the bytes it saves.
// Two generations of this many: at LIZARD-1024's 121 blocks a frame and 60 frames a second that is 9 to 18 s of ids,
// where 1 << 14 was 2 to 4.5 s and a repeat older than that counted as new.
const SEEN_KEEP = 1 << 16;
let seenIds = new Set(), seenOld = new Set();
const seenHas = (id) => seenIds.has(id) || seenOld.has(id);
const seenAdd = (id) => { seenIds.add(id); if (seenIds.size >= SEEN_KEEP) { seenOld = seenIds; seenIds = new Set(); } };
// How many workers to run is lizard-web/pool.mjs's decision; this page only carries it out. repeatShare
// is what the policy needs from the last window, stuck counts workers the watchdog below had to
// recover, and both are reported so neither can hide.
const policy = new PoolPolicy();
let repeatShare = 0, stuck = 0, workerErrs = 0, lastErr = "";
// Rates come from one-second buckets, the last five summed: a window that slides instead of one that restarts.
// msFound and msMiss are split because a frame that registers nothing returns after detect (src/dec.c, src/focus.c)
// and costs a quarter of one that goes on to sample and FEC. Over the recorded sessions that is 8 ms against 31 at
// the 1080 crop, so a mix that moves reads as the decoder getting slower when nothing about it changed.
// ai: tapEmpty: callbacks on which the tap had no new VideoFrame; tapDelivered, tapClosedUnseen, tapGapMax: the tap's own
// ai: count of frames off the track, those closed before a take, and the largest gap between two (vframes.mjs frameTap
// ai: stat, folded in as the bucket closes); longTasks, longTaskMs: main-thread tasks of 50 ms and over (a callback that
// ai: lands late shows here first); cbGapMax: the largest gap between two callbacks with a new frame. The gaps are
// ai: maxed over the window's buckets, the rest summed.
const fresh = () => ({ t0: performance.now(), dt: 0, callbacks: 0, captured: 0, missed: 0, sent: 0, processed: 0, withNew: 0, repeat: 0, empty: 0, skipped: 0, found: 0, useful: 0, freshBytes: 0, grab: 0, draw: 0, read: 0, glTake: 0, glSubmit: 0, vfSent: 0, vfMs: 0, ms: 0, msFound: 0, msMiss: 0, bad: 0, judged: 0, prof: {},
  tapEmpty: 0, tapDelivered: 0, tapClosedUnseen: 0, tapGapMax: 0, longTasks: 0, longTaskMs: 0, cbGapMax: 0, loafN: 0, loafMs: 0, loafScriptMs: 0, loafRenderMs: 0, loaf: {} });
const MAXED = new Set(["tapGapMax", "cbGapMax"]);
let cur = fresh(), buckets = [];
// ai: Since the page loaded, never reset, so a scripted run takes the difference of two rows (lizard-web/check_rates.mjs): frames
// ai: answered and dropped unanswered by the GPU queue; found, with new ids, new bytes, blocks verified (repeats in), bad; the
// ai: decoders' ms (the GPU's device ms a frame); and the crop each frame was decoded from: how many were tracked
// ai: (smaller than the base crop), their pixels summed, and the count of every size seen ("WxH", 64 named, the rest "other");
// ai: judged, the verified data blocks of frames the light showed to be the test stream's, each held to its id's bytes.
const totals = { frames: 0, dropped: 0, found: 0, withNew: 0, fresh: 0, blocks: 0, bad: 0, judged: 0, ms: 0, tracked: 0, px: 0, sizes: {} };
// ai: A browser without the longtask entry type leaves the two fields at 0.
try { new PerformanceObserver((list) => { for (const e of list.getEntries()) { cur.longTasks++; cur.longTaskMs += e.duration; } }).observe({ type: "longtask" }); } catch {}
// ai: Long animation frames (Chrome 123 and up), the same stalls with what ran in them: loafN, loafMs, the frames of 50 ms
// ai: and over; loafScriptMs, the scripts' part of them (a garbage collection inside a script counts to it); loafRenderMs,
// ai: their rendering (style, layout, paint) from renderStart; loaf, by script (the invoker, its file and char position,
// ai: its function): count and ms. What is neither script nor rendering is the browser's own. A browser without the
// ai: entry type leaves them 0.
try {
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      cur.loafN++; cur.loafMs += e.duration;
      if (e.renderStart) cur.loafRenderMs += e.startTime + e.duration - e.renderStart;
      for (const x of e.scripts ?? []) {
        cur.loafScriptMs += x.duration;
        const key = `${x.invokerType} ${x.invoker} ${(x.sourceURL || "").split("/").pop()}:${x.sourceCharPosition} ${x.sourceFunctionName || ""}`.trim(), a = (cur.loaf[key] ??= { n: 0, ms: 0 });
        a.n++; a.ms += x.duration;
      }
    }
  }).observe({ type: "long-animation-frame" });
} catch {}
// The phone throttles under sustained load and the desktop does not, so a single reading from a warm phone is not
// the steady state: a 40 ms grab measured that way sent this project's notes down a wrong path once. The best seen
// this run is the unthrottled cost, and showing it beside the live figure makes heat obvious instead of silent.
//
// best.ms is per pool size, and over found frames only. Workers contend for the same cores, so the same decode
// costs more at 6 workers than at 3 (24.6 ms at 3, 52.4 at 6, at the same registered share, research/rig/stats.jsonl).
// ai: One best across sizes would make the bigger pool look hot for doing exactly what it was added to do.
const freshBest = () => ({ grab: Infinity, ms: new Map() });
let best = freshBest();
// Where the decoder's wasm heap break sits, the highest any worker has reported. ALLOW_MEMORY_GROWTH only ever
// moves it up and a growth copies the whole heap, so a step here is a stall some decode has already paid for.
let heapTop = 0;
// How big the symbol is in CAMERA pixels, and what that leaves the finest ring. This is the variable that
// decides how much of a LIZARD frame comes back, and the finder has always measured it and thrown it away.
// Smoothed, because a frame that registered but carries nothing has a quad worth nothing either.
// For scale: the format's own constant says the finest ring needs 2.12 camera px a cycle (T_PX_PER_CYCLE in
// sim/lizard_pick.mjs, fitted on the simulator). Phone captures put it nearer 4.
let symPx = 0, ringPx = 0;
// ai: camAt: when the camera last started; firstDecodeMs: from then to the first decoded frame's answer (either decoder),
// ai: in the stats row, for the start of a receive (STATUS "The ramp at the start").
let camAt = 0, firstDecodeMs = null;
// ai: answered: a decoder has answered a frame since the camera, or the decoder in use, last started; the camera box is
// ai: covered by #loading until then (showState).
let answered = false;
// File time starts at the arrival of the frame BEFORE the first useful one: that frame's light was on the
// screen during the interval that ended with it. Starting at its decode would flatter short files by a frame.
let lastPresented = 0, lastStamp, repeats = 0, looping = false, prevArrive = 0, tFirst = 0, lastPaint = 0;
// ai: The stats row's average (avgKBs; the receiving line's until 2026-09-30, when it left the page for the log): from the
// ai: first decoded answer that brought a transfer new bytes (avgAt, the page's clock) and the page's new-byte count then
// ai: (avgFresh, that answer's own bytes included, so they are not counted in no time). Reset and shifted with tFirst;
// ai: the file line's time (showFile) stays tFirst's, end to end.
let avgAt = 0, avgFresh = 0;

// ai: ?workers=<n> holds the pool at n (check_rates' WORKERS); without it the pool grows and shrinks by itself
const FIXED = Math.min(CORES, Math.max(0, Math.floor(+LAB("workers", 0)) || 0));
const limit = () => FIXED || CORES;

// The worker in a slot: a new one, or a replacement for one that stopped answering. terminate() is what makes
// replacing safe, since nothing more can arrive from the old one. Merely clearing busy did not: on a worker that
// was only slow, its late reply cleared busy a SECOND time and freed a slot that already had the next frame
// queued behind the first, so the slot ran a frame behind for good; on one that had really died, the slot went on
// being handed a frame per rotation for the rest of the session, and those lost frames are what grow the pool.
// Every terminate goes through here.
function kill(slot) {
  slot.w?.terminate();
}
function newWorker(slot) {
  kill(slot);
  Object.assign(slot, { w: new Worker(new URL("./recv-worker.mjs", import.meta.url), { type: "module" }), busy: false, ready: false, frames: false, sentAt: 0, bootAt: performance.now(), readyAt: null, secMs: 0, secN: 0, setupMs: null, firstMs: null, buildMs: 0 });
  slot.w.onmessage = ({ data: m }) => onResult(slot, m);
  // ai: counted, and named on the readout's workers line (workerErrs, lastErr), as a worker's own error message is
  slot.w.onerror = (e) => { e.preventDefault?.(); workerErrs++; lastErr = `decoder worker: ${e.message ?? "failed to start"}`; revive(slot); };
  if (config) slot.w.postMessage({ type: "config", config });
  return slot;
}
// How long a frame may be out before the worker holding it is treated as gone. Decodes run in tens of ms, so
// two seconds is already far outside the range, but a phone slow enough to want a longer leash should be given
// one rather than killed for it: each replacement doubles this slot's wait, and a frame that comes back keeps
// the wait the device has shown it needs while clearing the death count. So a worker that is merely slow is
// replaced once and then left alone, and one that is really gone is replaced a few times and dropped.
const STUCK_WAIT = 2000, STUCK_WAIT_MAX = 32000;
// Replace the worker in a slot. One that cannot start at all would be replaced for ever, so after a few goes
// with no frame decoded in between the slot is dropped and the pool runs one short, which the workers line on
// screen says out loud.
function revive(slot) {
  const i = pool.indexOf(slot);
  if (i < 0) return;
  stuck++;
  slot.wait = Math.min(slot.wait * 2, STUCK_WAIT_MAX);
  if (++slot.deaths > 3 && pool.length > 1) { kill(pool.splice(i, 1)[0]); return; }
  newWorker(slot);
}
function spawn() { pool.push(newWorker({ deaths: 0, wait: STUCK_WAIT })); }
// In auto the size is the allocator's to choose, so this only holds it inside the ceiling: a menu
// that drops the ceiling must still bring the pool down. Growing is busy()'s job and shrinking the tick's.
function resize() {
  if (gpuWanted()) { while (pool.length) kill(pool.pop()); if (!gpu) startGpu(); return; }
  if (gpu) stopGpu();
  const want = !FIXED ? Math.min(pool.length, limit()) : limit();
  while (pool.length > want) kill(pool.pop());   // a frame in flight there is lost, which the counters show as one not processed
  while (pool.length < want) spawn();
  if (!pool.length) spawn();
}

// ai: The GPU decoder: one worker in place of the pool, chosen in Settings so a figure names its
// ai: decoder. It batches, so it is never busy and never busy-skipped, and it keeps its own queue limit (the dropped
// ai: count in its stats): the pool policy does not apply. When it cannot run the pool takes over (a GPU asked for by
// ai: name snaps the menu back to CPU; auto stays auto and keeps the pool for the page's life) and
// ai: gpuNote says why, in the console, the readout and every stats row.
let gpu = null, gpuNote = "", gpuStalls = 0;
// ai: The decoder (Settings, the user's to switch): auto, the default since 2026-10-01, takes the GPU
// ai: decoder where WebGPU gives a real adapter, else the CPU pool, and the pool for the rest of the page's life once the
// ai: GPU decoder failed (gpuFail). A software adapter (SwiftShader, llvmpipe) is no GPU here: it decodes many times
// ai: slower than the pool. autoGpu: null until askGpu's adapter has answered; autoWhy: why auto took the pool.
let autoGpu = null, autoWhy = "";
const gpuWanted = () => $("dec").value === "gpu" || ($("dec").value === "auto" && autoGpu === true);
async function askGpu() {
  if (autoGpu !== null) return;
  const why = await (async () => {
    if (!("gpu" in navigator)) return "this browser has no WebGPU";
    const pref = PARAMS.get("pref");
    const a = await Promise.race([navigator.gpu.requestAdapter(pref ? { powerPreference: pref } : {}).catch(() => null), new Promise((r) => setTimeout(() => r(undefined), 3000))]);
    if (a === undefined) return "the GPU did not answer";
    if (!a) return "no GPU adapter here";
    const i = a.info ?? {};
    if (i.isFallbackAdapter || a.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(`${i.vendor} ${i.architecture} ${i.description}`)) return "only a software GPU here";
    return "";
  })();
  autoGpu = !why; autoWhy = why;
  if (why) console.info(`decoder: auto on the CPU: ${why}`);
  showDecoder();
}
// ai: What auto takes, named by its hidden option (2026-10-05: the menu lists GPU and CPU alone; "Auto (GPU)" from
// ai: 2026-10-02, the line #decwhy under the menu before): until the adapter has answered, the GPU where the browser
// ai: has WebGPU; the why goes to the console.
function showDecoder() {
  const gpuNow = autoGpu === null ? "gpu" in navigator : autoGpu;
  $("dec").querySelector('option[value="auto"]').textContent = gpuNow ? "GPU" : "CPU";
}
// ai: The camera and its rate as the page's default opened them (2026-10-05: the menus list the cameras, and 30 and
// ai: 60, alone; their hidden auto options named after what opened: the track's label, its rate, 60 before one has).
// ai: The camera's field shows once there is a camera to name: one opened on this page, or one chosen here before.
let camLabel = "", camRate = 0;
function showCamera() {
  const a = $("cam").querySelector('option[value="auto"]');
  if (a) a.textContent = camLabel || "Camera";
  $("camRow").hidden = $("cam").value === "auto" && !camLabel;
  $("fps").querySelector('option[value="auto"]').textContent = String(camRate || 60);
}
// ai: A worker sent frames that decodes none of them for GPU_WAIT is stuck: replaced once, then given up on if the
// ai: replacement is stuck before it decodes anything. Its first batch after a config plans the lanes and runs at the
// ai: largest batch (SwiftShader: tens of seconds), so it gets GPU_FIRST. GPU_BOOT bounds a config's answer (ready or
// ai: unavailable), which builds the device and every pipeline, the back half's since 2026-09-29 (SwiftShader 32 s of
// ai: it; 60 s before, when the back half compiled in the first batch's window).
const GPU_WAIT = 20000, GPU_FIRST = 120000, GPU_BOOT = 120000;
function startGpu() {
  if (!("gpu" in navigator)) return gpuFail("unavailable: this browser has no WebGPU");
  // ai: The worker's switches: ?pref=high-performance|low-power asks for that adapter (on a machine with two GPUs
  // ai: Chrome's default is the low-power one); the rest are the lab's, from this page's URL (LAB): a precision or no
  // ai: subgroups for a device whose int8, f16 or subgroup arithmetic is suspect (precision auto: the chain, int8 first
  // ai: where it builds), a stage test, straddle cancellation, the page's tap for frames, stage timing.
  const q = new URLSearchParams([...PARAMS].filter(([k]) => k === "pref"));
  if (LAB("gprec", "auto") !== "auto") q.set("prec", LAB("gprec"));
  if (LAB("gsg") === "off") q.set("sg", "0");
  if (LAB("gtest", "none") !== "none") q.set("test", LAB("gtest"));
  if (LAB("gcancel") === "on") q.set("cancel", "1");   // ai: straddle cancellation (pass two), off by default (2026-09-25: not worth its cost)
  if (LAB("gsrc") === "page") q.set("gsrc", "page");   // ai: the worker declines a track too, so both sides agree
  if (LAB("gstages") === "on") q.set("stages", "1");   // ai: every batch profiled a pass a stage, the stage ms into the stats row
  // ai: the worker named by a literal, its switches as the query: the built app renames every module to a hash of its
  // ai: bytes and rewrites only literal references (lizard-web/pwa/hash.mjs), so a name put together at run time 404s there
  const url = new URL("./recv-gpu-worker.mjs", import.meta.url);
  url.search = q.toString();
  // ai: processor: the worker can read a track; trackSent: a clone is in its hands; source: "worker-track" once its
  // ai: first frame off it landed; trackWhy: why it declined one, or why none could be sent; crop: the last one posted.
  const me = { w: new Worker(url, { type: "module" }), ready: false, info: null, stats: null, bootAt: 0, waitingSince: 0, processor: false, trackSent: false, source: "", trackWhy: "", crop: null };
  gpu = me;
  me.w.onmessage = ({ data: m }) => { if (gpu === me) onGpu(m); };
  // ai: A load or build failure before ready means no GPU decoder here. After it, an error is counted and the
  // ai: watchdog below decides whether the worker still answers.
  me.w.onerror = (e) => { e.preventDefault?.(); if (gpu !== me) return; if (!me.ready) gpuFail(`failed to start: ${e.message ?? "the worker did not load"}`); else { workerErrs++; lastErr = e.message ?? "GPU worker error"; } };
  if (config) configureGpu();
}
function configureGpu() {
  answered = false;
  Object.assign(gpu, { ready: false, decoded: false, bootAt: performance.now(), waitingSince: 0 });
  gpu.w.postMessage({ type: "config", config });
}
function stopGpu() { const g = gpu; dropTrack(); gpu = null; tap?.stop(); tap = null; if (g) kill(g); answered = false; }
function gpuFail(why) {
  stopGpu();
  gpuNote = `GPU decoder ${why}; decoding on the CPU pool`;
  if ($("dec").value === "auto") { autoGpu = false; autoWhy = `the GPU decoder ${why}`; } else $("dec").value = "cpu";
  cur = fresh(); buckets = []; best = freshBest();
  resize();
  console.warn(gpuNote);
  showDecoder();
}
function onGpu(m) {
  if (m.type === "ready") { gpu.ready = m.version === config?.version; gpu.info = m.gpu ?? null; gpu.bootMs = Math.round(performance.now() - gpu.bootAt); const n = gpu.info?.nets; if (n) console.info(`gpu nets: bank ${n.bank}, small ${n.small}, large ${n.large}`); gpu.frames = !!m.frames; gpu.framesWhy = m.framesWhy ?? ""; gpu.processor = !!m.processor; sendTrack(); return; }
  if (m.type === "unavailable") return gpuFail(`unavailable: ${m.reason}`);
  // ai: A device lost while VideoFrames went to it may be F0's import failing on this browser and driver: the GPU
  // ai: worker is started again once, on luma, before the page gives it up for the pool.
  if (m.type === "lost" && gpu.sentVf && !vfOff) { vfOff = `off since the GPU was lost with VideoFrames going to it (${m.reason})`; stopGpu(); return startGpu(); }
  if (m.type === "lost") return gpuFail(`lost: ${m.reason}`);
  // ai: The worker's first frame off the track landed: the page's tap has nothing more to send and is stopped (its reader
  // ai: would go on closing frames unseen). Or the worker declined the track and stopped the clone: the tap stays.
  if (m.type === "source") {
    if (m.kind === "worker-track") { gpu.source = "worker-track"; gpu.sentVf = true; tap?.stop(); tap = null; }
    else { gpu.source = ""; gpu.trackSent = false; gpu.crop = null; gpu.trackWhy = m.why || "the worker declined the track"; }
    return;
  }
  // ai: Under the worker's track this page hands over no frame: the frames that reached the worker are the window's
  // ai: VideoFrames sent (vfSent, and sent, so none reads as grabbed), and frames arriving with no decode back is what the
  // ai: watchdog watches (waitingSince, as a send sets it on the tap).
  if (m.type === "stats") { gpu.stats = m; if (gpu.source === "worker-track" && m.arrived) { cur.vfSent += m.arrived; cur.sent += m.arrived; gpu.waitingSince ||= performance.now(); } return; }
  if (m.type === "error") { workerErrs++; lastErr = m.message; return; }
  if (m.type !== "result") return;
  // ai: A frame its queue had no room for is a skip, as a busy pool's is, and proves the worker alive but not decoding:
  // ai: only a decoded frame clears the watchdog.
  if (m.dropped) { cur.skipped++; totals.dropped++; trackTags.delete(m.tag); return; }
  gpu.waitingSince = 0; gpu.decoded = true; gpuStalls = 0;
  take(m);
}
function watchGpu() {
  const now = performance.now();
  if (!gpu.ready && gpu.bootAt && now - gpu.bootAt > GPU_BOOT) return gpuFail(`did not answer its config in ${GPU_BOOT / 1000} s`);
  if (!gpu.waitingSince || now - gpu.waitingSince < (gpu.decoded ? GPU_WAIT : GPU_FIRST)) return;
  stuck++;
  if (gpuStalls++) return gpuFail("stopped decoding twice");
  stopGpu(); startGpu();
}
$("dec").onchange = async () => { gpuNote = ""; gpuStalls = 0; vfOff = ""; poolVfOff = ""; cur = fresh(); buckets = []; best = freshBest(); if ($("dec").value === "auto") await askGpu(); showDecoder(); resize(); };
$("gcrop").onchange = () => postCrop();   // ai: the worker's track takes the new crop at once; the tap's next frame does anyway

// ai: The camera's frame as a VideoFrame (lizard-web/vframes.mjs), with the crop the grab would have read back, in place of the
// ai: grab. Under the GPU decoder (F0) the decoder takes it to luma on the device (gpu/ingest.mjs); under the pool the
// ai: worker copies the crop to luma (recv-worker.mjs lumaOf: 0.3 ms at a 720 crop, where the canvas grab was 41 ms on
// ai: the S26 Ultra and the GL grab 29). Either way the readback is skipped and the camera element is the preview. Under
// ai: F0 no frame is grabbed: the grey range comes with each batch (the device's histogram of the ingested luma,
// ai: gpu/wgsl/pyramid.mjs) and the worker's queue copies its own planning luma; the pool measures the range on the
// ai: luma it copies, so every frame goes as a VideoFrame (a luma frame in rotation cost the GL arm a frame in 32:
// ai: its readback landed a callback later and took the one worker). The grab stays wherever this page makes no
// ai: VideoFrame or the worker takes none (it says why when ready). vfOff: F0 given up for the session; poolVfOff: the
// ai: pool's VideoFrames given up for the session (a worker could not read one, or the page's tap gave none).
// ai: vfFormat: the pixel format the pool's workers last copied from, for the stats row. tapNone: callbacks in a row on
// ai: which the pool's tap had nothing, against TAP_EMPTY_MAX.
let tap = null, vfOff = "", poolVfOff = "", vfNote = "", vfFormat = "", tapNone = 0;
function vfWhyNot(slot = null) {
  if (gpu) {
    if (!gpu.ready) return "no GPU decoder ready";
    if (!gpu.frames) return `the GPU worker takes no VideoFrame: ${gpu.framesWhy}`;
    if (vfOff) return vfOff;
  } else {
    if (!pool.some((c) => c.ready)) return "no worker ready";
    if (slot ? !slot.frames : !pool.some((c) => c.ready && c.frames)) return "the workers take no VideoFrame";
    if (poolVfOff) return poolVfOff;
  }
  if (typeof VideoFrame !== "function") return "no VideoFrame on this page";
  return "";
}
// ai: The newest VideoFrame the page can hand over, at the video's size (the crop is the video's). False: it is not
// ai: that size, so the caller grabs instead. Null: the tap has nothing new since the last take.
function vfTake() {
  tap ??= frameTap(track, video);
  const vf = tap.take();
  if (!vf) return null;
  if (vf.displayWidth !== video.videoWidth || vf.displayHeight !== video.videoHeight) {
    vfNote = `the camera's VideoFrame is ${vf.displayWidth}x${vf.displayHeight}, the video ${video.videoWidth}x${video.videoHeight}`;
    vf.close();
    return false;
  }
  vfNote = "";
  return vf;
}
// ai: True when the frame went to the GPU worker as a VideoFrame. False sends the caller on to the grab: F0 is off, it
// ai: is the luma frame, or the frame is not the video's size. Null: the track has delivered nothing new since the
// ai: last take, so this callback shows the picture before again and the caller skips it.
function sendVideoFrame(sub, tag) {
  if (vfWhyNot()) return false;
  const vf = vfTake();
  if (!vf) return vf;
  trackTags.set(tag, sub);
  gpu.sentVf = true;
  cur.vfSent++;
  send(gpu, { type: "frame", tag, frame: vf, x: sub.x, y: sub.y, w: sub.w, h: sub.h }, [vf]);
  return true;
}
// ai: The tap's sink under the GPU decoder (vframes.mjs frameTap): every frame the track delivers goes to the worker
// ai: from the reader loop, tagged with the arrival before it as the callback tags its frames (tFirst counts on that),
// ai: cropped as the decoder's current rect says. A frame the worker cannot take (vfWhyNot) or of another size than
// ai: the video's is closed and the sink comes off, so the callback's own path answers from the next frame on.
let tapPrev = 0;
function tapSink(vf) {
  const why = vfWhyNot(), sized = vf.displayWidth === video.videoWidth && vf.displayHeight === video.videoHeight;
  if (why || !sized) {
    if (!why) vfNote = `the camera's VideoFrame is ${vf.displayWidth}x${vf.displayHeight}, the video ${video.videoWidth}x${video.videoHeight}`;
    vf.close();
    if (tap) tap.sink = null;
    return;
  }
  const arrive = performance.now(), tag = tapPrev || arrive - 33, sub = decodeRect(cropRect());
  tapPrev = arrive;
  trackTags.set(tag, sub);
  gpu.sentVf = true;
  cur.vfSent++;
  send(gpu, { type: "frame", tag, frame: vf, x: sub.x, y: sub.y, w: sub.w, h: sub.h }, [vf]);
}
// ai: The same for a pool worker, with the same three answers, and the frame's decode settings as the grab sends them.
// ai: The crop goes in the video's pixels, as the GPU worker's does; the worker maps it into the frame's coded pixels
// ai: (vframes.mjs lumaOf). A tap that gives nothing is a refusal, not a wait, when it is the <video> kind
// ai: (new VideoFrame(video) threw) or when the track's processor has delivered nothing for TAP_EMPTY_MAX callbacks:
// ai: the pool would starve with no watchdog, so both hand the session to the grab.
const TAP_EMPTY_MAX = 120;
function sendPoolFrame(slot, sub, tag) {
  if (vfWhyNot(slot)) return false;
  const vf = vfTake();
  if (vf === null) {
    if (tap.kind === "video" || ++tapNone >= TAP_EMPTY_MAX) { poolVfOff = tap.kind === "video" ? "off since new VideoFrame(video) gave none" : `off since the track's processor delivered no frame in ${TAP_EMPTY_MAX} callbacks`; return false; }
    return null;
  }
  tapNone = 0;
  if (!vf) return false;
  trackTags.set(tag, sub);
  cur.vfSent++;
  send(slot, { type: "frame", tag, frame: vf, x: sub.x, y: sub.y, w: sub.w, h: sub.h, delay: DELAY, prof: PROF, verify: VERIFY }, [vf]);
  return true;
}

// ai: The camera track into the GPU worker, where a MediaStreamTrackProcessor reads every frame with no callback between:
// ai: the tap here kept only the newest frame and lost one each time two came between callbacks (a third of the camera's
// ai: 60 on the S26 Ultra). A clone goes, transferred, so this page's own track feeds the preview as before and stopping
// ai: the clone never stops the camera. Until the worker says its first frame off it landed ("source"), the tap goes on
// ai: sending, so the hand-over loses nothing. A browser that cannot transfer a track throws here (DataCloneError): the
// ai: clone is stopped and the tap stays. Once from a worker's ready and from a camera start, whichever comes second.
function sendTrack() {
  if (!gpu?.ready || !gpu.frames || !gpu.processor || gpu.trackSent || gpu.trackWhy || vfOff || !track || !video.videoWidth || LAB("gsrc") === "page") return;
  const t = track.clone(), crop = decodeRect(cropRect());
  try { gpu.w.postMessage({ type: "track", track: t, crop, fps: track.getSettings().frameRate || 0, pageTimeOrigin: performance.timeOrigin }, [t]); }
  catch (e) { t.stop(); gpu.trackWhy = `the track could not be transferred: ${e?.message ?? e}`; return; }
  gpu.trackSent = true; gpu.crop = crop;
}
// ai: The worker stops the clone and its reader; the page's tap is made again on the next callback that needs it.
function dropTrack() {
  if (!gpu?.trackSent) return;
  gpu.w.postMessage({ type: "track", track: null });
  gpu.trackSent = false; gpu.source = ""; gpu.crop = null;
}
// ai: The worker crops each track frame with the last crop it was told, so a moved decode rect is posted as it moves; a
// ai: frame in its hands meanwhile is cropped a frame stale, inside the 10% margin the crop carries (TRACK_MARGIN).
function postCrop() {
  if (!gpu?.trackSent) return;
  const r = decodeRect(cropRect()), c = gpu.crop;
  if (c && c.x === r.x && c.y === r.y && c.w === r.w && c.h === r.h) return;
  gpu.crop = r;
  gpu.w.postMessage({ type: "crop", ...r });
}
// ai: Why the GPU worker's frames come off this page's tap and not the track in the worker; "" under the worker's track.
function trackNote() {
  if (!gpu || gpu.source === "worker-track") return "";
  if (LAB("gsrc") === "page") return "?gsrc=page keeps the page's tap";
  if (gpu.trackWhy) return gpu.trackWhy;
  return gpu.ready && !gpu.processor ? "no MediaStreamTrackProcessor in the worker" : "";
}

// ai: What is on screen to aim with: "shot", the canvas as drawn (the pixels the decoder gets), or "video" while frames
// ai: go as VideoFrames, the camera element beneath it, composited for nothing and cropped to the box's ratio
// ai: (recv.html #view.f0): the decoder then takes the same crop of the same frames. Chosen by the mode, not the frame,
// ai: so a frame that falls to the grab does not flick the canvas on.
let previewKind = "";
function preview(kind) {
  if (kind === previewKind) return;
  previewKind = kind;
  $("view").classList.toggle("f0", kind === "video");
  if (kind === "video") showCrop();
}
const previewNow = () => (!vfWhyNot() ? "video" : "shot");

// A finished file, as bytes or as a Blob (a chunked transfer's comes from OPFS as a File, on disk).
// ai: Offered, never downloaded unasked (2026-10-01): Open (#open, in a tab of its own), Save (#save, a link to a File of
// ai: it under its name), Share (#share, the system's share sheet, where the browser shares such a file; it went for
// ai: Clear on 2026-09-29 and came back with the received files), until the next file. It
// ai: stays through a stop and a newer transfer. On disk the file is also kept in the received files (lizard-web/library.mjs),
// ai: listed on Home. `file`, the readout's line, is what lizard-web/check_rates.mjs reads.
let fileUrl = "", fileOut = null;
function showFile(name, data, how = "checksum ok", type = "", sent = 0) {
  const seconds = (performance.now() - tFirst) / 1000, blob = data instanceof Blob ? data : new Blob([data]), n = blob.size;
  file = `${name}: ${n} B in ${seconds.toFixed(2)} s (${(n / seconds / 1000).toFixed(1)} KB/s), complete, ${how}`;
  $("file").textContent = file;
  if (fileUrl) URL.revokeObjectURL(fileUrl);
  fileOut = new File([blob], name, { type: type || blob.type });
  const save = $("save");
  save.href = fileUrl = URL.createObjectURL(fileOut); save.download = name; save.textContent = "Save";
  let shares = false;
  try { shares = !!navigator.canShare?.({ files: [fileOut] }); } catch {}
  $("share").hidden = !shares;
  ui.received = { name, n, secs: seconds, root: xferRoot, sent };   // ai: sent: the bytes the light carried, the file's where nothing shrank
  ui.err = "";   // ai: a fountain error is not fatal (fountain-worker.mjs goes on), so a file can still finish after one
  // ai: The camera off once the file is in (2026-10-01):
  // ai: nothing is left to read, and the phone cools. Once a file (the fountain offers it once), so Start camera after
  // ai: it reads on.
  if (track) { second(); stopCamera(); }
  showState();
  if (SAVE_POST) logPost(`/api/file?name=${encodeURIComponent(name)}`, blob);
}
// ai: The received file deleted on Home (2026-10-07, as the app's delete): the library's channel says it changed, and
// ai: where the file this page shows as received is no longer kept, the green line, its buttons and the fountain's
// ai: transfer go (forgetTransfer), so the same file still in the light is received and kept anew. A change that
// ai: keeps the file (another deleted, one filed) touches nothing.
watchLibrary(async () => {
  const r = ui.received?.root;
  if (!r) return;
  const kept = await listKept().catch(() => null);
  if (!kept || kept.some((e) => e.root === r)) return;
  ui.received = null;
  if (fileUrl) { URL.revokeObjectURL(fileUrl); fileUrl = ""; }
  fileOut = null;
  forgetTransfer();
  showState();
});
$("open").onclick = () => { if (fileUrl) window.open(fileUrl, "_blank"); };
$("share").onclick = () => { if (fileOut) navigator.share({ files: [fileOut], title: fileOut.name }).catch(() => {}); };
// A chunked transfer as it goes (lizard-web/fountain-worker.mjs): where the header came from, each chunk's share of its blocks,
// and what the worker holds. xferStats rides in the stats, per chunk left out.
// ai: xferNow: the last such message whole, per included, for the state line and its meter (showState).
let xferStats = null, xferNow = null;
const MB = (b) => (b / 1048576).toFixed(1);
function showXfer(m) {
  const { per, ...rest } = m;
  xferStats = rest; xferNow = m;
  showState();
  if (m.done) return;   // the file line takes over
  const whence = !m.header ? (m.lightSeen ? "a header was read from the light and refused" : "waiting for the header, none read from the light yet") : "header read from the light";
  const h = m.header, lines = [`file: ${whence}`];
  if (h) {
    // A glyph a chunk: # verified, + decoded and waiting for the manifest, 0 to 9 tenths of its blocks in hand.
    const glyphs = per.length <= 256 ? Array.from(per, (v) => (v === 255 ? "#" : v === 254 ? "+" : String(Math.floor(v / 10)))).join("") : "";
    lines.push(`${h.name || "(no name)"}, ${h.length} B, ${h.chunks} chunk${h.chunks === 1 ? "" : "s"} of 2^${h.log2}, BLAKE3 ${h.root.slice(0, 16)}...`,
      `verified ${m.verified} of ${h.chunks}${m.manifest ? `, manifest ${m.manifestHave} of ${m.manifest}${m.listOk ? ", checked against the root" : ""}` : ""}${m.rejected ? `, ${m.rejected} chunk decode${m.rejected > 1 ? "s" : ""} REJECTED and collected again` : ""}${m.manifestRejected ? `, manifest refused ${m.manifestRejected}x` : ""}`);
    if (glyphs) lines.push(glyphs);
  }
  lines.push(`held: blocks ${m.store === "opfs" ? "on disk (OPFS)" : `in memory, ${MB(m.mem.storeBytes)} MB${m.storeWhy ? ` (no OPFS: ${m.storeWhy})` : ""}`}; heaps: Wirehair ${MB(m.mem.wirehairHeap)} MB, codec ${MB(m.mem.codecHeap)} MB`);
  $("file").textContent = lines.join("\n");
}
async function offerXfer(m) {
  let data = m.bytes;
  // ai: lib: kept in the received files (fountain-worker.mjs fileAway); opfs: left in the scratch folder (no keeping)
  if (m.lib) data = await openKept(m.lib);
  else if (m.opfs) { let d = await navigator.storage.getDirectory(); for (const p of m.opfs.slice(0, -1)) d = await d.getDirectoryHandle(p); data = await (await d.getFileHandle(m.opfs.at(-1))).getFile(); }
  showFile(m.name || "received.bin", data, `every chunk verified, BLAKE3 ${m.root}`, m.mediaType, m.sent ?? 0);
}
// ai: A header with another root is another transfer, whose ids repeat the last one's (chunk and symbol count from 0):
// ai: the page's set of ids goes, and the file line and its clock with it. The first header keeps what came before it.
let xferRoot = "";
fountain.onmessage = ({ data: m }) => {
  if (m.type === "xfer" && m.header?.root && m.header.root !== xferRoot) {
    if (xferRoot) { seenIds = new Set(); seenOld = new Set(); file = ""; tFirst = 0; avgAt = 0; }
    xferRoot = m.header.root;
  }
  if (m.type === "xfer") showXfer(m);
  else if (m.type === "file") offerXfer(m).catch((e) => { file = `${m.name}: FAILED to offer: ${e?.stack ?? e}`; console.error(file); ui.err = "The file arrived but could not be opened for saving."; showState(); });
  else if (m.type === "error") { console.error(`fountain: ${m.message}`); ui.err = "Putting the file together failed. Reload the page to receive it again."; showState(); }
};

function onResult(slot, m) {
  if (m.type === "ready") { slot.ready = m.version === config?.version; slot.frames = !!m.frames; slot.readyAt ??= performance.now(); slot.setupMs ??= m.setupMs ?? null; return; }
  slot.busy = false; slot.sentAt = 0; slot.deaths = 0;
  if (m.type === "result" && m.ms) { slot.secMs += m.ms; slot.secN++; slot.firstMs ??= Math.round(m.ms); if (m.built) slot.buildMs += Math.round(m.ms); }
  // Kept, because the stats line below rewrites $("out") every second and an error shown there is gone before
  // it can be read. A worker that failed before it was ever ready failed to BUILD, so it is replaced, not left.
  if (m.type === "error") { workerErrs++; lastErr = m.message; if (!slot.ready) revive(slot); return; }
  // ai: A VideoFrame the worker could not read (a format it does not know, or copyTo refused): that frame is a skip, and
  // ai: the pool gets the grab for the rest of the session. Every worker sees the same frames, so one answer settles it.
  if (m.noFrames) { poolVfOff = `off since a worker could not read a VideoFrame: ${m.noFrames}`; cur.skipped++; trackTags.delete(m.tag); return; }
  if (m.vf) vfFormat = m.vf;
  if (m.probe) cropProbe = m.probe;
  take(m);
}
// ai: One decoded frame, from a pool worker or the GPU worker: the two send the same message. B: a block's payload bytes.
function take(m) {
  const B = config.usefulBytes;
  if (!answered) { answered = true; showState(); }
  // ai: an answer to a frame from before the camera's last start (a stop and start in quick succession) is not the first
  if (camAt && firstDecodeMs == null && (m.tag ?? camAt) >= camAt - 100) firstDecodeMs = Math.round(performance.now() - camAt);
  let news = 0;
  if (m.test) ui.testAt = performance.now();
  else if (m.ids?.length) ui.fileAt = performance.now();
  // ai: A LIZARD frame's blocks new to the page go to the fountain, but not a test frame's (m.test): the light says they
  // ai: are the test stream's, so no transfer is theirs.
  if (m.ids) {
    // New to the transfer, not just to that worker. A control block (a transfer's header or manifest, sim/xfer.mjs) is
    // passed on each time and is not new data: its id is every transfer's, and only its bytes say which one it is.
    const keep = [];
    m.ids.forEach((id, k) => { if (isControlId(id)) keep.push(k); else if (!seenHas(id)) { seenAdd(id); keep.push(k); news++; } });
    if (m.bytes && keep.length && !m.test) {
      const all = new Uint8Array(m.bytes), bytes = new Uint8Array(B * keep.length);
      keep.forEach((k, j) => bytes.set(all.subarray(k * B, (k + 1) * B), j * B));
      fountain.postMessage({ type: "blocks", ids: keep.map((k) => m.ids[k]), bytes: bytes.buffer }, [bytes.buffer]);
    }
  }
  // A capture of a display frame that is already in is work done for nothing: it is a repeat, not a decode.
  if (news) cur.withNew++; else if (m.seen) cur.repeat++; else cur.empty++;
  cur.processed++; cur.found += m.found ? 1 : 0; cur.useful += m.seen * B; cur.freshBytes += news * B; cur.ms += m.ms; cur.bad += m.bad; cur.judged += m.judged ?? 0;
  totals.frames++; totals.found += m.found ? 1 : 0; totals.withNew += news ? 1 : 0; totals.fresh += news * B; totals.blocks += m.seen; totals.bad += m.bad; totals.judged += m.judged ?? 0; totals.ms += m.ms;
  if (!avgAt && news && !m.test) { avgAt = performance.now(); avgFresh = totals.fresh; }
  if (m.found) cur.msFound += m.ms; else cur.msMiss += m.ms;
  if (m.heap > heapTop) heapTop = m.heap;
  if (m.prof) for (const k in m.prof) cur.prof[k] = (cur.prof[k] ?? 0) + m.prof[k];
  if (m.levels) levels = m.levels;
  // ai: The band: the only place a receiver learns the sender's format and rate.
  if (m.fmt) {
    ui.bandAt = performance.now();
    const same = (a, b) => a?.version === b?.version && a?.fps === b?.fps && a?.ring === b?.ring;
    if (!same(m.fmt, band) && same(m.fmt, bandSeen)) { band = m.fmt; showCfg(); showState(); }
    bandSeen = m.fmt;
  }
  // Aim the next frame at where this one found the symbol, but only where the frame proved it WAS the symbol: a
  // block that passed its CRC, or the format word read. Anything else gives the whole frame back for one look.
  // Following any quad at all was how a lost track stayed lost: the finder settles for a poor quad rather than
  // none (a track read of 0.06 is enough), a crop around a poor quad holds only part of the symbol or none of it,
  // and inside it the next poor quad is always there. Recorded on the S26, 2026-09-23 (run ...06-26-06-874Z): past
  // 25 degrees of turn the crop walked down to 96 x 96 px and came back to the whole frame only when a capture
  // found nothing at all, which inside a crop almost never happens.
  // ai: The crop a frame was taken with: echoed by the GPU worker for a frame off its track (box), else the page's own.
  const shotBox = m.box ?? trackTags.get(m.tag);
  if (shotBox !== undefined) {
    trackTags.delete(m.tag);
    // ai: at most 64 sizes named, the rest "other": crops clipped at the frame's edge take any size, and a row a second carries them
    const base = cropRect(), named = `${shotBox.w}x${shotBox.h}`, size = named in totals.sizes || Object.keys(totals.sizes).length < 64 ? named : "other";
    totals.px += shotBox.w * shotBox.h; totals.tracked += shotBox.w * shotBox.h < base.w * base.h ? 1 : 0; totals.sizes[size] = (totals.sizes[size] ?? 0) + 1;
    if (m.quad && (m.seen > 0 || m.fmt)) trackFrom(m.quad, shotBox); else trackBox = null;
    postCrop();
  }
  if (trackTags.size > 256) trackTags.clear();   // ai: results that never came back (the GPU worker alone may owe a few batches); the next frame re-aims from whole
  if (m.quad) {
    const q = m.quad, side = Math.hypot(q[2] - q[0], q[3] - q[1]);
    // ai: The format is the last word read (none read yet: no ring to size).
    const subch = band ? 8 * band.version : 0, n = subch ? N_FOR(subch) : 0;
    if (side > 0) {
      symPx = symPx ? 0.7 * symPx + 0.3 * side : side;
      ringPx = subch ? (symPx * SPAN(n, band.ring) / MODULES(n, band.ring)) / R_RING(subch) : 0;   // the picture's px over its top ring's cycles
    }
  }
  if (!tFirst && m.seen && !m.test) tFirst = m.tag;   // ai: the test stream's frames are no file's, so its time is not the file's
}
// ai: LIZARD's decoders, told nothing (sim/phy.mjs blindSpec: every picture of the ladder, each at its top: the word names
// ai: which, and a receiver reads any version it names, with no flag; the #sizes menu that capped it at 1024 by default
// ai: went 2026-09-29).
// ai: Versions are this page's own, which the workers' ready echoes.
let cfgVersion = 0;
const lizardConfig = () => ({ version: ++cfgVersion, spec: { phy: "focus", blind: 1, nmax: PICTURE_SIZES.at(-1) }, usefulBytes: PAYLOAD });
// ai: New decoder settings: every worker told, the window's counters restarted. newTransfer: what is held goes (the ids
// ai: seen, the file line, the fountain's transfer: sim/xfer.mjs, its header from the light).
// ai: The transfer in hand let go: its ids, file line and clocks, and the fountain's receiver (a new one).
function forgetTransfer() {
  file = ""; tFirst = 0; avgAt = 0; seenIds = new Set(); seenOld = new Set(); xferStats = null; xferNow = null; xferRoot = ""; $("file").textContent = "";
  fountain.postMessage({ type: "config", config: { store: STORE } });
}
function setConfig(c, newTransfer) {
  config = c;
  if (newTransfer) { band = null; forgetTransfer(); }
  cur = fresh(); buckets = [];
  for (const slot of pool) { slot.ready = false; slot.busy = false; slot.w.postMessage({ type: "config", config: c }); }
  if (gpu && gpuWanted()) configureGpu();
  resize();
  showCfg();
}


// ai: Once a second while the camera runs: the pool's watch, the GPU's, the policy, and the stats row; also once more by
// ai: showFile just before the camera stops itself, so the row that carries the finished file is posted (a stopped
// ai: camera posts none; the rates are over the time the window really took).
function second() {
  if (!track) return;
  // A worker that dies mid-frame, or one whose result is lost, never clears its busy flag, and the pool
  // quietly runs a worker short for the rest of the session while the allocator adds more to make up for
  // it. How long is long enough to say so is STUCK_WAIT. Copied first, because reviving a slot can drop it.
  for (const c of [...pool]) if (c.busy && c.sentAt && performance.now() - c.sentAt > c.wait) revive(c);
  // And a slot that never came up at all, which costs the pool a worker just as quietly.
  for (const c of [...pool]) if (!c.ready && performance.now() - c.bootAt > BOOT_DEADLINE) revive(c);
  if (gpu) watchGpu();
  if (policy.tick({ busySkips, size: pool.length, hasIdle: pool.some((c) => c.ready && !c.busy), auto: !FIXED })) {
    const i = pool.findIndex((c) => c.ready && !c.busy);   // never one with a frame still out, nor one still starting up
    if (i >= 0) kill(pool.splice(i, 1)[0]);
  }
  busySkips = 0;
  if (tap?.stat) { cur.tapDelivered += tap.stat.delivered; cur.tapClosedUnseen += tap.stat.closedUnseen; cur.tapGapMax = Math.max(cur.tapGapMax, tap.stat.gapMax); tap.resetStat(); }
  cur.dt = (performance.now() - cur.t0) / 1000; buckets.push(cur); cur = fresh();
  if (buckets.length > 5) buckets.shift();
  const w = fresh(); for (const b of buckets) { for (const k in w) if (k !== "t0" && k !== "prof" && k !== "loaf") w[k] = MAXED.has(k) ? Math.max(w[k], b[k]) : w[k] + b[k]; for (const k in b.prof) w.prof[k] = (w.prof[k] ?? 0) + b.prof[k]; for (const k in b.loaf) { const a = (w.loaf[k] ??= { n: 0, ms: 0 }); a.n += b.loaf[k].n; a.ms += b.loaf[k].ms; } }
  const dt = w.dt, s = track.getSettings(), per = (v) => (w.processed ? v / w.processed : 0);
  // Goodput is THIS second's payload over this second's own measured length, not the window's. Both come from
  // counters that are exact: freshBytes is a count of blocks new to the transfer times the block size, and the
  // length is two performance.now() readings. A five-second mean lags the link by seconds and smooths away
  // exactly the moments worth looking at, and it is what made one stretch read 67% and another 34%.
  const last = buckets[buckets.length - 1];
  // Per found frame and per missed frame, never mixed: the two cost a factor of four apart, so their mean moves
  // with the registered share alone.
  const miss = w.processed - w.found, msFound = w.found ? w.msFound / w.found : 0, msMiss = miss ? w.msMiss / miss : 0;
  // Only once a window holds enough frames to mean anything, and the minimum over windows rather than over frames,
  // ai: which would just find the luckiest frame. The decode's best is kept per pool size, so a live figure above it
  // ai: means this machine got slower and not that the allocator added a worker.
  // ai: The grab's costs are per frame handed to a decoder, not per result: the GPU worker drops frames it was handed.
  // ai: Under F0, per frame grabbed (those sent less those that went as VideoFrames); F0's own per VideoFrame.
  const grabbed = w.sent - w.vfSent, perGrab = (v) => (grabbed ? v / grabbed : 0);
  if (grabbed >= 10) best.grab = Math.min(best.grab, perGrab(w.grab));
  const bestKey = gpu ? "gpu" : pool.length;   // ai: the GPU worker's best is its own, never a pool size's
  if (w.found >= 10) best.ms.set(bestKey, Math.min(best.ms.get(bestKey) ?? Infinity, msFound));
  const bestMs = best.ms.get(bestKey) ?? Infinity;
  const avgSecs = avgAt ? (performance.now() - avgAt) / 1000 : 0;
  const stats = {
    res: `${video.videoWidth}x${video.videoHeight}${s.frameRate ? " @" + Math.round(s.frameRate) : ""}${CROP === "square" ? `, centre ${cropRect().w}x${cropRect().h}` : ""}${decodeRect(cropRect()).w !== cropRect().w ? `, decoding ${decodeRect(cropRect()).w}x${decodeRect(cropRect()).h} of it` : ", decoding all of it"}`, window: dt.toFixed(0), captured: w.captured, capturedFps: w.captured / dt, callbackFps: w.callbacks / dt, processedFps: w.processed / dt, decodedFps: w.withNew / dt, repeatFps: w.repeat / dt, emptyFps: w.empty / dt, skippedFps: w.skipped / dt, missedFps: w.missed / dt,
    foundShare: per(w.found),
    // Per frame the decoder actually saw. Dividing by captured mixed an exact byte count with an inferred frame
    // count: captured includes the frames presentedFrames says the browser never handed over, which no decoder
    // ever had a chance at. Those stay visible on the skipped line, where they belong.
    usefulPerFrame: per(w.useful), freshPerFrame: per(w.freshBytes),
    goodputKBs: last.dt ? last.freshBytes / last.dt / 1000 : 0, goodputBytes: last.freshBytes, goodputSecs: last.dt,
    // ai: the transfer's average since its first decode (avgAt), new bytes a second, null under a second of it: for the
    // ai: log alone since 2026-09-30 (after a wait it often read wrong)
    avgSecs: avgSecs ? +avgSecs.toFixed(1) : null, avgKBs: avgSecs >= 1 ? +((totals.fresh - avgFresh) / avgSecs / 1000).toFixed(1) : null,
    msGrab: perGrab(w.grab), msGlTake: perGrab(w.glTake), msGlSubmit: perGrab(w.glSubmit), ms: per(w.ms), msFound, msMiss, foundFrames: w.found, bestGrab: best.grab, bestMs,
    // The two that say whether a climb is this page's doing: the decoder's heap break only ever rises, and the
    // id set is the one thing here that grows for as long as the session lasts.
    symbolPx: Math.round(symPx), ringPx, heapTop, seenIds: seenIds.size + seenOld.size, prof: Object.fromEntries(Object.entries(w.prof).map(([k, v]) => [k, per(v)])), bad: w.bad, judged: w.judged, file, xfer: xferStats, workers: pool.length, workersReady: pool.filter((c) => c.ready).length, workersMax: limit(), stuck, workerErrs, lastErr,
    cropDraw: drawMode, cropChecked: cropCheck.checked, cropDiffer: cropCheck.differ, cropProbe,
    // Which browser, whether it gives a per-frame video callback, and how many cores it admits to.
    ua: UA, vfc: VFC ? 1 : 0, cores: CORES,
    // ai: The camera's own state this second (track.getSettings(), Image Capture fields where the browser has them): its zoom,
    // ai: focus mode and focus distance, and which camera (the #cam menu's pick, and its device id's head). For the view
    // ai: that zooms in on a code filling the box (2026-09-29): nothing on this page scales the preview, so a jump in
    // ai: symbolPx here with the zoom unmoved is the camera's own (a lens or sensor-mode switch, or focus breathing near its
    // ai: closest focus).
    camZoom: s.zoom ?? null, camFocusMode: s.focusMode ?? null, camFocusDistance: s.focusDistance ?? null, camPick: $("cam").value === "auto" ? "auto" : "chosen", camId: (s.deviceId ?? "").slice(0, 8),
    // The canvas grab split in two. On a software canvas the draw costs and the read is a memcpy; on a
    // GPU-backed one it is the other way round and the read is a stall.
    msDraw: perGrab(w.draw), msRead: perGrab(w.read),
    // ai: Which sampler this second ran: "wasm" (the pool) or "gpu-decoder" (the GPU worker); "gpu" in old records is the
    // ai: WebGPU sampler in the wasm workers, stashed on 2026-09-22.
    sampleMode: gpu ? "gpu-decoder" : "wasm",
    // ai: Which decoder ran this second, what the GPU worker says of itself, and why a GPU decoder asked for is not running.
    decoder: gpu ? "gpu" : "cpu", decoderNote: gpuNote, gpuReady: !!gpu?.ready,
    // ai: the #gcrop menu (tracked, or the full square under the GPU decoder), and the run's totals with the page's clock in s
    gpuCrop: $("gcrop").value, totals: { ...totals, sizes: { ...totals.sizes }, t: performance.now() / 1000 },
    // ai: How the decoders read ("blind", the only way since 2026-09-26, kept so rows before and after can be told
    // ai: apart); the picture sizes to which they are built; judged (above), the window's verified data blocks of test
    // ai: frames, held to the stream from the light, of which bad were wrong.
    decode: "blind",
    ...(gpu?.info ? { gpuAdapter: gpu.info.adapter, gpuPrecision: gpu.info.precision, gpuNets: gpu.info.nets ?? null, gpuBank: gpu.info.bank, gpuBootMs: gpu.bootMs ?? null, gpuBoot: gpu.info.boot ?? null } : {}),
    ...(gpu?.stats ? { gpuMs: gpu.stats.ms, gpuLag: gpu.stats.lag, gpuB: gpu.stats.B, gpuInflight: gpu.stats.inflight, gpuQueued: gpu.stats.queued, gpuHeld: gpu.stats.held ?? 0, gpuDropped: gpu.stats.dropped,
      // ai: The window's frames that went as VideoFrames and as luma, each with its found and blocks (F0 against the grab).
      gpuVf: gpu.stats.vf ?? null, gpuLuma: gpu.stats.luma ?? null,
      // ai: where the GPU worker's last grey levels came from: "gpu" (its histogram) or "luma" (a grabbed frame)
      gpuLevelsFrom: gpu.stats.levelsFrom ?? "",
      // ai: a frame's raw proposer peaks, peaks kept, cascade candidates, found share and verified blocks: where a device loses them
      gpuRaw: gpu.stats.raw, gpuPeaks: gpu.stats.peaks, gpuCascade: gpu.stats.cascade, gpuFound: gpu.stats.found, gpuBlocks: gpu.stats.blocks,
      gpuScore: gpu.stats.score, gpuTop: gpu.stats.top, gpuF0: gpu.stats.f0, gpuTest: LAB("gtest", "none"), gpuSg: LAB("gsg", "on"),
      // ai: cancellation (?gcancel=on), and in the worker's last second the frames whose pass two came back with a solved fit and the
      // ai: blocks it verified there (repeats included), both inside each frame's own answer since 2026-09-27
      gpuCancel: LAB("gcancel", "off"), gpuCancelled: gpu.stats.cancelled ?? 0, gpuGained: gpu.stats.gained ?? 0,
      gpuPlanMs: gpu.stats.planMs ?? null, gpuTwinMs: gpu.stats.twinMs ?? null, gpuTwins: gpu.stats.twins ?? null,
      gpuStages: LAB("gstages", "off"), gpuStageMs: gpu.stats.stageMs ?? null,
      // ai: where a batch's wall time goes in the worker's last second (recv-gpu-worker.mjs WIN0): batches, the frames a
      // ai: batch carried against gpuB, the host's laps, submit to readback, the readback's round trip on an idle device,
      // ai: answering a batch, F0's device time a frame, frames that reached the worker and their largest gap, VideoFrames open at once
      gpuBatches: gpu.stats.batches ?? 0, gpuCarried: gpu.stats.carried ?? 0, gpuHost: gpu.stats.host ?? null, gpuSubmitToMap: gpu.stats.submitToMap ?? 0, gpuMapLatency: gpu.stats.mapLatency ?? null,
      gpuDoneMs: gpu.stats.doneMs ?? 0, gpuIngestMs: gpu.stats.ingestMs ?? 0, gpuArrived: gpu.stats.arrived ?? 0, gpuGapMax: gpu.stats.gapMax ?? 0, gpuOpenMax: gpu.stats.openMax ?? 0 } : {}),
    // LIZARD's data is in grey LEVELS, so a squeezed or clipped range loses signal at every frequency at once.
    // 41% of full range with 13% of the picture pinned at 255 is what a badly overexposed run looked like, against
    // 64% and 87% for runs that read well, so this is worth having on screen while the phone is still in hand.
    // ai: Read through the lens: bandFps is all a receiver knows of the sender's rate, and research/rig/stats.jsonl carries it so
    // ai: a run can be checked against the sender's line in the same row after.
    bandVersion: band?.version ?? 0, bandFps: band?.fps ?? 0,
    rateAsked: $("fps").value, rateNote, srcCrop, photoCaps, greyRange: levels?.range ?? 0, greyClipHi: levels?.hi ?? 0, greyClipLo: levels?.lo ?? 0, greyMedian: levels?.mid ?? 0,
    // Which grab produced this second. research/rig/stats.jsonl already holds every field of this object, so the two
    // arms of the experiment can be split apart afterwards by this one.
    // ai: Frames handed to `decoder` (the GPU worker under F0, or the pool's workers) as VideoFrames in the window, and as
    // ai: luma by the grab; the page's ms for each (the whole callback; 0 under the worker's track, which costs this page
    // ai: nothing a frame); where they came from (the track read in the GPU worker, the page's track processor or the
    // ai: <video>) and why not the worker's track; the pixel format the pool copied from (the GPU reports none); and why
    // ai: none went, if none. Under the worker's track vfSent is the frames that reached the worker (its arrived).
    vfSent: w.vfSent, lumaSent: grabbed, msVideoFrame: w.vfSent ? w.vfMs / w.vfSent : 0, vfSource: gpu?.source === "worker-track" ? "worker-track" : tap?.kind ?? "", vfSourceNote: trackNote(), vfFormat, vfNote: vfWhyNot() || vfNote,
    // ai: The tap against the callback, a second: frames off the track, those closed before a take, callbacks that found
    // ai: none new; the largest gaps (ms) between two deliveries and two callbacks; and the main thread's long tasks.
    tapDelivered: w.tapDelivered / dt, tapClosedUnseen: w.tapClosedUnseen / dt, tapEmptyFps: w.tapEmpty / dt, tapGapMax: w.tapGapMax, cbGapMax: w.cbGapMax, longTasks: w.longTasks, longTaskMs: w.longTaskMs,
    loafN: w.loafN, loafMs: Math.round(w.loafMs), loafScriptMs: Math.round(w.loafScriptMs), loafRenderMs: Math.round(w.loafRenderMs), loafTop: Object.entries(w.loaf).sort((a, b) => b[1].ms - a[1].ms).slice(0, 8).map(([k, a]) => [k, a.n, Math.round(a.ms)]),
    // ai: each pool worker's boot (created to ready, ms), its last second (frames answered and their mean decode ms), and
    // ai: its start: [its decoder's build in configure, its first frame's decode, the frames that built a picture's codec
    // ai: summed] (ms)
    firstDecodeMs, workersBootMs: pool.map((c) => (c.readyAt ? Math.round(c.readyAt - c.bootAt) : null)), workersSec: pool.map((c) => [c.secN, c.secN ? +(c.secMs / c.secN).toFixed(1) : null]), workersFirst: pool.map((c) => [c.setupMs, c.firstMs, c.buildMs]),
    grabMode, glUploadMs: glGrab?.stat.uploads ? glGrab.stat.uploadMs / glGrab.stat.uploads : 0, glSubmitted: glGrab?.stat.submitted ?? 0, glNoSlot: glGrab?.stat.noSlot ?? 0, glFailed: glGrab?.stat.failed ?? 0,
    ...(cmp.px ? { cmpFrames: cmp.frames, cmpMax: cmp.max, cmpMean: cmp.sum / cmp.px, cmpOver1: cmp.over1 / cmp.px } : {}),
  };
  repeatShare = w.processed ? w.repeat / w.processed : 0;
  const why = verdict(stats);
  stats.verdict = why?.text ?? null;
  // ai: The flags, each worded once: on its line of the readout below, and with the verdict on #why.
  const clipping = levels && stats.greyRange < 0.5 * 255 ? "TOO LITTLE RANGE: the picture is clipping, not the code's fault" : "";
  // What the band said, next to what the camera is managing. A camera below the stated display rate cannot see
  // every frame whatever else is right, and that is worth saying while the phone is still in hand.
  const slow = band?.fps && stats.capturedFps < band.fps - 1 ? `CAMERA BELOW THE DISPLAY RATE at ${stats.capturedFps.toFixed(0)} fps: some frames are never seen` : "";
  const bandLine = band
    ? `band: ${NAME(8 * band.version)}${band.fps ? `, ${band.fps} fps stated` : ", no rate stated"}${slow ? `  ${slow}` : ""}`
    : "band: no format word read yet";
  const g = gpu?.stats, num = (v) => (typeof v === "number" ? v.toFixed(1) : "?");
  const decLine = gpu
    ? `decoder GPU${gpu.info ? ` (${stats.gpuAdapter}, ${stats.gpuPrecision}, bank ${stats.gpuBank})` : ""}, ${gpu.ready ? "ready" : "starting"}${g ? `: ${num(g.ms)} ms a frame, ${num(g.lag)} ms from arrival to answer, B ${g.B || "not planned yet"}, ${g.inflight} in flight, ${g.queued} queued, ${g.dropped} dropped, ${stats.vfSent ? `frames as VideoFrames (F0, from the ${stats.vfSource}${stats.vfSourceNote ? `: ${stats.vfSourceNote}` : ""})${stats.lumaSent ? `, ${stats.lumaSent} as luma this window` : ", grey levels from the device"}` : `frames as luma${stats.vfNote ? ` (${stats.vfNote})` : ""}`}${g && g.raw != null ? `; a frame: ${num(g.raw)} raw peaks, ${num(g.peaks)} kept, ${num(g.cascade)} to the large net, ${num(g.found)} found, ${num(g.blocks)} blocks, best score ${num(g.score)} (top ${num(g.top)})${g.vf?.n && g.luma?.n ? `; VideoFrames ${(g.vf.blocks / g.vf.n).toFixed(1)} blocks a frame (${g.vf.n}), luma ${(g.luma.blocks / g.luma.n).toFixed(1)} (${g.luma.n})` : ""}` : ""}` : ""}`
    : `${gpuNote ? `${gpuNote}\n` : ""}workers ${stats.workersReady} ready of ${stats.workers}, at most ${stats.workersMax}, ${stats.vfSent ? `frames as VideoFrames (from the ${stats.vfSource}${stats.vfFormat ? `, ${stats.vfFormat}` : ""}, copied to luma in the worker)${stats.lumaSent ? `, ${stats.lumaSent} by the grab` : ""}` : `frames by the grab${stats.vfNote ? ` (${stats.vfNote})` : ""}`}`;
  const cropLine = grabMode !== "canvas"
    ? `crop taken on the GPU by texel coordinate, so there is no source rect to disagree with${glGrab.lost ? "  GL CONTEXT LOST" : ""}`
    : `crop drawn ${drawMode === "rect" ? "with a source rect" : "whole and clipped: this browser's source rect disagreed with the whole frame"}, checked on ${cropCheck.checked} frames, ${cropCheck.differ} differed${grabNote}`;
  const grabPart = stats.vfSent && !stats.lumaSent ? "grab: none (VideoFrames)"
    : `${grabMode} grab ${stats.msGrab.toFixed(1)}${grabMode === "canvas" ? ` (draw ${stats.msDraw.toFixed(1)}, read ${stats.msRead.toFixed(1)})` : ` (out ${stats.msGlSubmit.toFixed(1)}, back ${stats.msGlTake.toFixed(1)})`}`;
  const bestPart = best.grab < Infinity ? ` (best this run ${gpu ? "on the GPU" : `at ${pool.length} worker${pool.length > 1 ? "s" : ""}`}: ${best.grab.toFixed(1)} + ${bestMs < Infinity ? bestMs.toFixed(1) : "-"})` : "";
  // ai: The readout, an entry a line; an entry that does not apply this second is false and left out.
  // ai: The last second first, then the window's lines (up to 5 s): a start that is over shows here at once, where the
  // ai: window climbs for its length (2026-09-29).
  const perSec = (v) => (last.dt ? (v / last.dt).toFixed(1) : "0.0");
  $("out").textContent = [
    why?.text,
    `this second: camera ${perSec(last.captured)}, processed ${perSec(last.processed)}, new data ${perSec(last.withNew)}, skipped ${perSec(last.skipped)} fps; below, the last ${dt.toFixed(0)} s`,
    `camera ${stats.res}: ${stats.capturedFps.toFixed(1)} fps${rateNote ? `  ${rateNote}` : ""}`,
    cropLine,
    cmp.px && `GPU luma against the canvas's, ${cmp.frames} frames: max ${cmp.max}, mean ${(cmp.sum / cmp.px).toFixed(3)}, ${(100 * cmp.over1 / cmp.px).toFixed(3)}% differ by more than 1`,
    `decoded ${stats.decodedFps.toFixed(1)} with new data, ${stats.repeatFps.toFixed(1)} repeats of a frame already in, ${stats.emptyFps.toFixed(1)} nothing read`,
    `skipped ${stats.skippedFps.toFixed(1)} (${stats.missedFps.toFixed(1)} never reached the page, ${stats.callbackFps.toFixed(1)} callbacks/s)`,
    `${decLine}${stats.stuck ? `, ${stats.stuck} restarted` : ""}${stats.workerErrs ? `, ${stats.workerErrs} FAILED: ${stats.lastErr}` : ""}`,
    `registered ${(100 * stats.foundShare).toFixed(0)}%${symPx ? `, symbol ${Math.round(symPx)} px of the ${cropRect().w} crop, finest ring ${ringPx ? ringPx.toFixed(2) + " camera px a cycle" : "-"}` : ""}`,
    bandLine,
    `per decoded frame: ${Math.round(stats.freshPerFrame)} B new, ${Math.round(stats.usefulPerFrame)} B with repeats`,
    `goodput ${stats.goodputKBs.toFixed(1)} KB/s (${stats.goodputBytes} B of new blocks in ${stats.goodputSecs.toFixed(2)} s, 1 KB = 1000 B)`,
    `per frame: ${grabPart}${stats.vfSent ? `, VideoFrame ${stats.msVideoFrame.toFixed(1)}` : ""} + decode ${stats.msFound.toFixed(1)} ms registered, ${stats.msMiss.toFixed(1)} not${bestPart}`,
    PROF && `stages, ms: ${Object.entries(stats.prof).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")}`,
    `grey ${levels ? `${(100 * stats.greyRange / 255).toFixed(0)}% of range (p5..p95), median ${stats.greyMedian}, ${(100 * stats.greyClipHi).toFixed(1)}% pinned white, ${(100 * stats.greyClipLo).toFixed(1)}% black${clipping ? `  ${clipping}` : ""}` : "not measured yet"}`,
    `camera accepts: ${Object.keys(stats.photoCaps).join(", ") || "nothing beyond size and rate"}`,
    `bad blocks ${stats.bad} of ${stats.judged} held to the test stream`,
  ].filter(Boolean).join("\n");
  // ai: The lab line (#lab, in Developer Tools; the numbers line until 2026-10-01): the last second's rate, the registered
  // ai: share and the format's name while its word is live. The Android app's Developer Tools shows the same line.
  const live = band && recent(ui.bandAt), named = live ? `, ${NAME(8 * band.version)}${band.fps ? ` at ${nb(`${band.fps} fps`)}` : ""}` : "";
  ui.lab = live || w.found ? `${nb(`${Math.round(stats.goodputKBs)} KB/s`)}, ${Math.round(100 * stats.foundShare)}% registered${named}` : "no code yet";
  // ai: The hints (#why) in plain words, one a line (2026-10-01): the verdict's and the flags' lab wording stays in #out
  // ai: and the stats row. With no code seen, "not registered" only repeats "Looking for a code", so the verdict waits for
  // ai: one, unless no frame reaches a decoder.
  const lead = why && (live || w.found || (stats.processedFps ?? 0) < 0.5) ? plainVerdict(why.text) : "";
  // ai: A sender painting faster than WEB_FPS is one for the native app (2026-10-01): a browser cannot delay a camera
  // ai: frame, so its captures fall anywhere in the display's refresh and above 24 a second too many hold two pictures;
  // ai: the Android app holds the camera's phase by the pilots (lizard-android/, PhaseLock.kt) and reads 60 painted. Said
  // ai: while the word is live; the sender's own control for it is Settings' FPS slider ("Receiving with" went
  // ai: 2026-10-02).
  const fast = live && band.fps > WEB_FPS ? `${band.fps} pictures a second is too fast for a browser: use the LIZARD app, or send at ${WEB_FPS}.` : "";
  ui.why = [lead, clipping && "The picture is washed out: avoid glare, or turn the other screen's brightness down.", live && slow ? "This camera runs slower than the code changes, so some of it is missed." : "", fast].filter(Boolean).join("\n");
  showState();
  // ai: The frame path's diagnostics go to the console (read over adb DevTools) and the stats row, not the page, which
  // ai: keeps what a user aims and judges by (2026-09-25).
  console.info(`frames: camera ${stats.capturedFps.toFixed(1)}, callbacks ${stats.callbackFps.toFixed(1)}, processed ${stats.processedFps.toFixed(1)}, new data ${stats.decodedFps.toFixed(1)} (${stats.goodputKBs.toFixed(1)} KB/s, average ${num(stats.avgKBs)} over ${num(stats.avgSecs)} s, band v${stats.bandVersion}, registered ${(100 * stats.foundShare).toFixed(0)}%, bad ${stats.bad} of ${stats.judged} judged), skipped ${stats.skippedFps.toFixed(1)} (merged ${stats.missedFps.toFixed(1)}, tap empty ${num(stats.tapEmptyFps)}, closed unseen ${num(stats.tapClosedUnseen)}); long tasks ${stats.longTasks} (${num(stats.longTaskMs)} ms)${g ? `; gpu: batches ${g.batches}, carrying ${num(g.carried)}, ${num(g.ms)} ms a frame, map ${num(g.mapLatency)}, done ${num(g.doneMs)}, ingest ${num(g.ingestMs)}, arrived ${g.arrived}, open ${g.openMax}, held ${g.held ?? 0}, dropped ${g.dropped}, cancelled ${g.cancelled ?? 0}, gained ${g.gained ?? 0}` : ""}`);
  // ai: A development log (devlog.mjs): posted, never awaited, its answer unread.
  logPost("/api/stats", stats);
  for (const c of pool) { c.secMs = 0; c.secN = 0; }
}
setInterval(second, 1000);

// THE crop, in camera pixels: the centred square of the frame's short side in either orientation (a phone held
// upright loses rows, not columns), or the whole frame. Computed here and nowhere else. Whole-pixel offsets: a
// fractional source origin makes the browser interpolate, and a smeared module edge costs the read.
// Where the symbol was last seen, in camera pixels. crop_ceiling.mjs (archived: ../archive/build-scratch-2026-09/)
// measured a tight crop returning the
// whole payload in every noise cell; locate.mjs (archived: ../archive/locate-first/exp/) then measured four BLIND ways to find that crop and all of
// them failed for one reason, that the symbol cannot be located with the detector that cannot find it. A
// previous frame's quad is not blind. The symbol is found from the whole frame once and tracked after that.
//
// Measured over the ten recorded runs (scripts/exp/ddce_track.mjs): where the symbol fills the frame nothing changes,
// and where it does not the crop falls to a third of the frame, the decode gets 41% faster AND reads more
// (78.3% to 100.0% on run 21-27, 81.9% to 96.7% on 21-26), because a tight crop leaves the line fit nothing but
// border to fit. Nothing measured got worse. On a phone it should be worth more than it is here, since the GRAB
// scales with the crop too and is 8 ms of a 29 ms frame.
let trackBox = null, trackTags = new Map();
const TRACK_MARGIN = 0.10, TRACK_QUANT = 32, TRACK_MIN = 96;

function baseRect() {
  const vw = video.videoWidth, vh = video.videoHeight, side = Math.min(vw, vh);
  return CROP === "square" ? { x: Math.floor((vw - side) / 2), y: Math.floor((vh - side) / 2), w: side, h: side } : { x: 0, y: 0, w: vw, h: vh };
}

// What is DRAWN, and so what is on screen to aim with. Tracking must not touch this: the preview is the thing
// the phone is pointed by, and a view that jumped and changed shape every time the symbol's apparent size moved
// would be unusable for exactly the job it exists for.
function cropRect() { return baseRect(); }

// What is actually READ BACK and handed to wasm, always INSIDE cropRect(). Keeping the two apart this way costs
// nothing and keeps the property the preview was built for: the decoder's pixels are a sub-rectangle of the ones
// on screen, so the two still cannot silently disagree the way they did on Firefox for Android above 720p.
// The readback is the expensive half of a canvas grab, so narrowing it is where the saving was anyway.
// ai: The #gcrop menu at "full square" gives the GPU worker the base crop alone (the camera's square at the source, or
// ai: the centre square taken here), never a tracked one, to measure what the tracker saves and costs the GPU decoder
// ai: under hand motion (STATUS "What the tracker costs the GPU decoder"). The pool always tracks.
function decodeRect(base) {
  if (!trackBox || VERIFY || (gpu && $("gcrop").value === "full")) return base;   // VERIFY is there to prove the decoded region IS the centre square
  const x = Math.max(base.x, trackBox.x), y = Math.max(base.y, trackBox.y);
  const w = Math.min(base.x + base.w, trackBox.x + trackBox.w) - x, h = Math.min(base.y + base.h, trackBox.y + trackBox.h) - y;
  if (w < TRACK_MIN || h < TRACK_MIN || w * h >= base.w * base.h * 0.95) return base;
  return { x, y, w, h };
}

// The quad the decoder settled on, turned into the next frame's crop. Sizes are quantised so a hand that moves
// a pixel does not resize the canvas, the GL render target and the preview every frame.
function trackFrom(quad, box) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = 0; k < 4; k++) {
    const px = quad[2 * k] + box.x, py = quad[2 * k + 1] + box.y;
    x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  if (!(x1 > x0 && y1 > y0)) return;
  const mx = (x1 - x0) * TRACK_MARGIN, my = (y1 - y0) * TRACK_MARGIN, q = (v) => Math.ceil(v / TRACK_QUANT) * TRACK_QUANT;
  const bx = Math.floor(x0 - mx), by = Math.floor(y0 - my);
  trackBox = { x: Math.max(0, bx), y: Math.max(0, by), w: q(x1 + mx - bx), h: q(y1 + my - by) };
}
// The box on the page takes the crop's shape; its size is the stylesheet's (a column in portrait, the full height in
// landscape). The canvas fills it, so nothing here decides which pixels are seen.
function showCrop() {
  const r = cropRect(), ratio = r.w && r.h ? r.w / r.h : CROP === "square" ? 1 : 16 / 9;
  $("view").style.setProperty("--ratio", String(ratio));   // lizard-web/recv.html sizes the box from it, per orientation
}

// The one drawImage from the camera on this page. It paints #shot, which is looked at and read back, so the
// square on screen and the square the decoder gets are the same pixels, not two crops that should agree.
// drawMode "rect" crops in the draw (the readback and the colour conversion shrink with it). "offset" draws the
// whole frame shifted and lets the canvas clip it, for a browser whose source rect is not what it says (below).
let drawMode = PARAMS.get("draw") === "offset" ? "offset" : "rect";   // ?draw=offset forces the fallback, to try it on a device
function blit(ctx, r, mode = drawMode) {
  const c = ctx.canvas;
  if (c.width !== r.w || c.height !== r.h) {
    c.width = r.w; c.height = r.h;
    ctx.imageSmoothingEnabled = false;   // assigning a size resets the context. Nothing here scales; this is for the day something does
    if (c === shot) showCrop();
  }
  if (mode === "rect") ctx.drawImage(video, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
  else ctx.drawImage(video, -r.x, -r.y, video.videoWidth, video.videoHeight);
}
// Canvas path only. A shader crops by texel coordinate, so the GL grab cannot have the bug this tests for and
// does not pay for the test: three draws and three getImageData on every frame it runs on.
// Is the source rect the rect it says? The same frame drawn both ways must give the same picture. Where it does
// not, the whole-frame draw is the reference (it has no crop to get wrong) and the page switches to it for good.
// Run on the first frames of every frame size, since a lens cap or a blank wall agrees with anything, and with
// ?verify every 30th frame after that, so the count can be read off the phone under test.
const cropCheck = { checked: 0, differ: 0, key: "", left: 0, frame: 0, next: 0, moved: 0 }, probe = document.createElement("canvas"), pctx = probe.getContext("2d", { willReadFrequently: true });
function checkSourceRect(r) {
  // drawImage takes whatever frame is current when it is called, and that can move on inside one callback: at
  // 3840 x 2160, or on a slow phone, a draw and a readback are most of a frame period. So the source rect is drawn
  // before AND after the whole-frame draw. If those two differ the frame changed under the test, which says
  // nothing about the rect: the attempt is given back and run on a later frame.
  // Every fourth pixel each way is kept and the readback let go, so one frame's worth of memory is held at a time.
  const grab = (mode) => {
    blit(pctx, r, mode);
    const px = pctx.getImageData(0, 0, r.w, r.h).data, out = new Uint8Array(3 * Math.ceil(r.w / 4) * Math.ceil(r.h / 4));
    for (let y = 0, k = 0; y < r.h; y += 4) for (let x = 0; x < r.w; x += 4, k += 3) { const i = 4 * (y * r.w + x); out[k] = px[i]; out[k + 1] = px[i + 1]; out[k + 2] = px[i + 2]; }
    return out;
  };
  // Two draws may round a colour conversion differently, so a level or two is not a difference; a crop that is off
  // by a pixel moves every edge in the picture, which is.
  const differ = (a, b) => { let off = 0; for (let k = 0; k < a.length; k += 3) if (Math.abs(a[k] - b[k]) > 24 || Math.abs(a[k + 1] - b[k + 1]) > 24 || Math.abs(a[k + 2] - b[k + 2]) > 24) off++; return off > a.length / 600; };
  try {
    const a = grab("rect"), b = grab("offset"), a2 = grab("rect");
    if (differ(a, a2)) return false;
    cropCheck.checked++;
    if (differ(a, b)) { cropCheck.differ++; drawMode = "offset"; }
    return true;
  } catch { return true; }   // no memory for it (a 4K frame on a small phone): the test is given up, the frame loop is not
  finally { probe.width = probe.height = 0; }   // up to a whole frame of memory, wanted a few times a session
}
function cropTest(rect) {
  const key = `${video.videoWidth}x${video.videoHeight} ${CROP}`;
  if (key !== cropCheck.key) { cropCheck.key = key; cropCheck.left = 3; cropCheck.frame = 0; cropCheck.next = 0; cropCheck.moved = 0; }   // a new size (the phone was turned, the resolution changed): test again
  // Three tests a frame size, twenty frames apart. One that the camera moved under is run again ten frames later,
  // a few times at most: a camera faster than this machine can test is not a reason to test for ever.
  // With nothing cropped the two draws are one and the same, so there is nothing to test.
  const frame = cropCheck.frame++, cropped = rect.x > 0 || rect.y > 0;
  if (cropped && cropCheck.left > 0 && frame >= cropCheck.next) {
    if (checkSourceRect(rect) || ++cropCheck.moved > 12) { cropCheck.left--; cropCheck.next = frame + 20; } else cropCheck.next = frame + 10;
  } else if (cropped && VERIFY && frame % 30 === 29) checkSourceRect(rect);
}
// For lizard-web/check_rates.mjs (?verify): one channel of the frame sent to the decoder at three points of each
// corner (vframes.mjs cornerProbe). stride 4 reads the red channel of RGBA, stride 1 the luma the GL grab hands
// over; a pool worker given a VideoFrame answers the probe itself. check_rates paints its markers into a grey
// clip, so the paths agree, and its near() already allows either video range.
let cropProbe = null;
function probeCorners(px, w, h, stride = 4) { cropProbe = cornerProbe(px, w, h, stride); }

let frameHandle = 0;
function onFrame(now, meta) {
  frameHandle = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
  if (!video.videoWidth) return;
  // One camera frame must count once. A callback that repeats the previous frame's timestamp is the same
  // picture again (seen when a second loop was started, and possible on compositors faster than the camera).
  const stamp = meta?.captureTime ?? meta?.mediaTime;
  cur.callbacks++;
  if (stamp !== undefined && stamp === lastStamp && ++repeats < 30) return;   // 30 in a row: the stamp is not moving, stop trusting it
  if (stamp !== lastStamp) repeats = 0;
  lastStamp = stamp;
  // The callback can also coalesce under load, so frames that never reached the page come from the browser's counter.
  const n = meta?.presentedFrames && lastPresented ? Math.max(1, meta.presentedFrames - lastPresented) : 1;
  lastPresented = meta?.presentedFrames ?? 0;
  const arrive = performance.now(), before = prevArrive || arrive - 33;
  if (prevArrive) cur.cbGapMax = Math.max(cur.cbGapMax, arrive - prevArrive);
  prevArrive = arrive;
  // ai: A merged callback loses a frame only while this callback carries frames; under the tap's sink or the worker's
  // ai: own track every frame reaches the worker from the reader loop, and the merge is the compositor's business.
  cur.captured += n;
  if (!(tap?.sink || gpu?.source === "worker-track")) { cur.missed += n - 1; cur.skipped += n - 1; }
  // ai: The GPU worker reads the track itself (sendTrack): the callback only counts, and the camera element is the preview.
  if (gpu?.source === "worker-track") { preview("video"); return; }
  const rect = cropRect(), sub = decodeRect(rect);
  // ai: Under the GPU decoder on a Chrome whose workers have no track processor (the S26's Chrome 153), the tap's reader
  // ai: loop hands every frame to the worker itself (tapSink): this callback is then off the frame path and only counts
  // ai: and previews, as it is under the worker's own track, and no frame is closed unseen between two callbacks.
  if (gpu?.ready && gpu.frames && !vfOff && !vfNote && gpu.source !== "worker-track" && typeof VideoFrame === "function") {
    tap ??= frameTap(track, video);
    if (tap.kind === "track") { tap.sink ??= tapSink; preview("video"); return; }
  }
  if (grabMode !== "canvas") return glFrame(rect, sub, arrive, before);
  let tGrab = performance.now();
  const slot = freeSlot();
  // ai: The VideoFrame first: a frame a worker takes as one is never drawn here (preview: the camera element shows
  // ai: instead). The blit below, into a canvas Chrome for Android keeps in software, converted the whole crop on the
  // ai: main thread at 41 ms a frame and left the callback 20 of the camera's 60 frames a second (STATUS "The GPU
  // ai: decoder decoding on the phone"); its cost is the whole callback.
  if (slot) {
    const went = slot === gpu ? sendVideoFrame(sub, before) : sendPoolFrame(slot, sub, before);
    if (went) { preview("video"); cur.vfMs += performance.now() - tGrab; return; }
    if (went === null) { cur.skipped++; cur.tapEmpty++; return; }
  }
  // ai: The source-rect check tests the canvas grab's crop, so it runs only on a frame the grab takes for a decoder. It
  // ai: ran on every callback before 2026-09-29, ahead of the VideoFrame path that takes the pool's frames and never
  // ai: draws: three whole-crop draws and readbacks a test into a software canvas, about 70 ms at the 1080 crop on the
  // ai: S26, up to 15 tests after a load (the live picture moves under most), stalling the callback for 7 to 9 s
  // ai: (STATUS "The ramp after a reload").
  if (slot) { cropTest(rect); tGrab = performance.now(); }
  preview(previewNow());
  let tDraw = tGrab;
  // The code is square, so the sides of a 16:9 frame are pixels to convert, read back and search for nothing:
  // 44% of a 1920 x 1080 frame. Cropping happens in the draw, so all of that shrinks with it.
  // A frame nobody will decode is still painted if the last paint is 100 ms old: the preview is what is aimed with,
  // and it should not freeze exactly when the pool is full and aiming is hardest.
  // ai: Unless the camera element is the preview: then a frame nobody decodes costs nothing.
  if (slot || (previewKind !== "video" && arrive - lastPaint > 100)) { blit(sctx, rect); lastPaint = arrive; }
  // The draw and the readback are timed apart because they trade places between browsers, and which of them
  // costs says what the canvas is. A software canvas does the YUV to RGB conversion in the DRAW and hands
  // getImageData a memcpy; a GPU-backed one draws for almost nothing and pays for it in the READ, where
  // getImageData has to stall the pipeline and pull the surface back. willReadFrequently is a hint, not a
  // promise, and a canvas that is also on screen is the case a browser is most likely to keep on the GPU.
  tDraw = performance.now();
  cur.draw += tDraw - tGrab;
  if (!config) { cur.skipped++; return; }
  if (!slot) return busy();
  // Drawn whole, read back narrow: the sub-rectangle's offset is within the canvas, which holds the base crop.
  const { w, h } = sub, px = sctx.getImageData(sub.x - rect.x, sub.y - rect.y, w, h).data, rgba = px.buffer;
  if (VERIFY) probeCorners(px, w, h);
  cur.read += performance.now() - tDraw;
  cur.grab += performance.now() - tGrab;
  trackTags.set(before, sub);
  send(slot, { type: "frame", rgba, w, h, tag: before, delay: DELAY, prof: PROF }, [rgba]);
}

// Round robin over the idle workers, so none goes cold and captures of one display frame spread out.
function freeSlot() {
  if (gpu) return config && gpu.ready ? gpu : null;   // ai: the GPU worker batches: once ready it is sent every frame
  for (let k = 0; config && k < pool.length; k++) { const c = pool[(next + k) % pool.length]; if (c.ready && !c.busy) { next = (next + k + 1) % pool.length; return c; } }
  return null;
}
function send(slot, msg, transfer) {
  if (slot === gpu) gpu.waitingSince ||= performance.now();   // ai: the first frame sent since it last decoded one, for watchGpu
  else {
    slot.busy = true;
    slot.sentAt = performance.now();   // the watchdog above needs to know how long it has been out
  }
  cur.sent++;
  slot.w.postMessage(msg, transfer);
}
// Capacity already on its way is a reason not to add more, but only while it really is on its way. A worker
// still not ready after this long is not capacity, it is one that failed to start, and waiting on it holds the
// whole allocator still: a pool sat at 2 with a ceiling of 8, dropping 37 frames a second at a quarter
// occupancy, for as long as one slot stayed unready (build/captures/run-2026-09-21T20-04-09-535Z).
// BOOT_DEADLINE is the other half: past it the worker is not slow, it is broken. One that throws while building
// its decoder posts an error, and that error used to go to the same line the next second's stats overwrite, so a
// slot could sit dead and silent for ever while the allocator kept adding more beside it. Seen doing exactly
// that: workers 2, 3, 4, 5 with ready stuck at 2 (build/captures/run-2026-09-21T20-20-58-789Z).
const BOOT_GRACE = 3000, BOOT_DEADLINE = 10000;
const booting = () => pool.some((c) => !c.ready && performance.now() - c.bootAt < BOOT_GRACE);
// A frame was lost because every worker was busy. Whether that is worth a worker is lizard-web/pool.mjs.
function busy() {
  cur.skipped++;
  if (gpu) return;   // ai: the GPU worker is never busy: this frame came before it was ready
  busySkips++;
  if (policy.lost({ allReady: !booting(), size: pool.length, ceiling: limit(), repeatShare })) spawn();
}

// recv.html?grabcmp: the GPU's luma against the software canvas's, on the same frame, every 30th frame. Both
// paths run the same arithmetic on different hardware and should agree to a level or two. A shifted
// distribution means one of them expanded video range (16-235 to 0-255) and the other did not, which would
// change every module's contrast and quietly cost the outer rings. Chrome on the desktop agrees exactly; the
// phone's browser is a different implementation, so this has to be askable there and not only here.
const CMP = PARAMS.has("grabcmp");
let cmpTick = 0;
const cmp = { frames: 0, max: 0, sum: 0, px: 0, over1: 0 };
// The GL arm can only answer this after a fence, so the canvas side is taken FIRST and held: both then
// describe the frame the video element had at this instant, which is the whole point of the comparison.
async function compareGrabs(rect) {
  blit(sctx, rect);
  const px = sctx.getImageData(0, 0, rect.w, rect.h).data;
  const a = await glGrab.readSync(rect);
  if (!a) return;
  let max = 0, sum = 0, over1 = 0;
  for (let i = 0, j = 0; i < a.length; i++, j += 4) {
    const d = Math.abs(a[i] - ((77 * px[j] + 150 * px[j + 1] + 29 * px[j + 2] + 128) >> 8));
    if (d > max) max = d;
    sum += d; if (d > 1) over1++;
  }
  cmp.frames++; cmp.max = Math.max(cmp.max, max); cmp.sum += sum; cmp.px += a.length; cmp.over1 += over1;
}

// The GL grab is a pipeline and not a call: this frame's luma lands a frame or two later, through a fence, and
// goes to whichever worker is free by then rather than to one held open for it. So finished readbacks are drained
// first, and a new frame is started only if a worker looks idle now, which is the gate the canvas path applies
// to itself.
//
// cur.grab therefore counts something different here: the main thread's own cost, the upload and two draws going
// out plus the getBufferSubData coming back, and not the wait for the GPU, which is the point of the fence. It is
// the honest figure for what the page spends, but it is NOT the canvas path's number with a smaller value, so the
// arms are judged on sustained goodput and the throttling curve in research/rig/stats.jsonl, not on this.
function glFrame(rect, sub, arrive, before) {
  const t0 = performance.now();
  // One saturated callback is one lost frame. The readback loop below and the submit at the bottom
  // both find the pool full and both used to say so, which counted every such callback twice: two
  // skips reported, and two of the four pressure steps that add a worker. The GL arm grew twice as
  // eagerly as the canvas arm for no reason but this.
  let said = false;
  const lost = () => { if (!said) { said = true; busy(); } };
  for (;;) {
    const got = glGrab.take();
    if (!got) break;
    const slot = freeSlot();
    if (!slot) { lost(); break; }   // finished with nowhere to put it, which the canvas path counts the same way
    if (VERIFY) probeCorners(got.luma, got.w, got.h, 1);
    send(slot, { type: "frame", luma: got.luma.buffer, w: got.w, h: got.h, tag: got.tag, delay: DELAY, prof: PROF }, [got.luma.buffer]);
  }
  // Split, because the two halves fail for different reasons: the first is the fence poll and the
  // getBufferSubData that follows it, the second the upload, the two draws and the readPixels going out.
  const tMid = performance.now();
  const idle = !!config && (gpu ? gpu.ready : pool.some((c) => c.ready && !c.busy));
  // ai: The VideoFrame first, as onFrame: a frame that goes as one is neither uploaded, painted nor read back here. A
  // ai: pool worker is picked now, since the frame goes to it now and not through the ring.
  if (idle) {
    const slot = gpu ?? freeSlot(), went = slot === gpu ? sendVideoFrame(sub, before) : slot ? sendPoolFrame(slot, sub, before) : false;
    if (went) { preview("video"); cur.vfMs += performance.now() - t0; return; }
    if (went === null) { cur.skipped++; cur.tapEmpty++; return; }
  }
  preview(previewNow());
  glGrab.newFrame();   // one upload serves the preview and the readback
  if (glGrab.canvas.width !== rect.w || glGrab.canvas.height !== rect.h) showCrop();   // blit does this for #shot
  // Same texture and same crop uniforms as the readback, so the square on screen is the square decoded, which is
  // what #shot was for. A frame nobody will decode is still painted if the last paint is 100 ms old.
  // ai: Unless the camera element is the preview: then a frame nobody decodes is neither uploaded nor painted.
  if (idle || (previewKind !== "video" && arrive - lastPaint > 100)) { glGrab.paint(rect); lastPaint = arrive; }
  if (!config) cur.skipped++;
  else if (!idle) lost();
  else { trackTags.set(before, sub); if (!glGrab.submit(sub, before)) cur.skipped++; }   // painted whole above, read back narrow
  const tEnd = performance.now();
  cur.glTake += tMid - t0; cur.glSubmit += tEnd - tMid; cur.grab += tEnd - t0;
  // After the timing, since it runs both grabs and is not part of either.
  if (CMP && cmpTick++ % 30 === 0) compareGrabs(rect);
}

// One button: start, then stop. Stopping releases the camera and the wake lock and shrinks the pool, so the phone can cool between runs.
let wake = null;

function stopCamera() {
  dropTrack();
  tap?.stop(); tap = null;
  track?.stop(); track = null; video.srcObject = null;
  if (looping) { (video.cancelVideoFrameCallback ? video.cancelVideoFrameCallback(frameHandle) : cancelAnimationFrame(frameHandle)); looping = false; }
  wake?.release().catch(() => {}); wake = null;
  while (pool.length > 1 && !FIXED) kill(pool.pop());
  for (const slot of pool) slot.busy = false;
  if (gpu) gpu.waitingSince = 0;   // ai: frames it still owes are no longer awaited, so its silence is not a stall
  cur = fresh(); buckets = []; policy.reset();
  shot.width = shot.height = 0; cropCheck.key = ""; trackBox = null; trackTags.clear();   // nothing of the last session left on screen, and the next one tests its source rect again
  if (glGrab) glGrab.canvas.width = glGrab.canvas.height = 0;
  $("out").textContent = "camera off";
  ui.lab = ui.why = ""; ui.stoppedAt = performance.now();
  showState();
}
// ai: start(): the camera started once at a time (ui.starting refuses a second while one is under way, and #go is off
// ai: meanwhile). A failure leaves nothing running and says why in a sentence a user can act on (startError); the raw
// ai: error goes to the console.
async function start() {
  if (ui.starting) return;
  // ai: the transfer in hand is let go from the state line (the fountain keeps its blocks): the light names it again
  ui.err = ""; ui.starting = true; xferNow = null;
  if (tFirst && ui.stoppedAt) tFirst += performance.now() - ui.stoppedAt;   // ai: a resumed transfer's rate leaves the pause out
  if (avgAt && ui.stoppedAt) avgAt += performance.now() - ui.stoppedAt;
  ui.stoppedAt = 0;
  showState();
  try { await startCamera(); }
  catch (e) { console.error("camera did not start:", e); if (track) stopCamera(); ui.err = startError(e); }
  finally { ui.starting = false; showState(); }
}
function startError(e) {
  if (!navigator.mediaDevices) return "The camera needs a secure page: open this one over https, or on localhost.";
  switch (e?.name) {
    case "NotAllowedError": return "The camera is blocked. Allow it for this site in the browser's site settings, then start again.";
    case "NotFoundError": case "OverconstrainedError": return "No camera was found on this device.";
    case "NotReadableError": return "The camera is in use by another app. Close that app, then start again.";
    default: return `The camera did not start (${e?.name ?? "unknown error"}).`;
  }
}
async function startCamera() {
  if (!navigator.mediaDevices) throw new Error("no navigator.mediaDevices: the page is not a secure context");
  const [w, h] = $("res").value.split("x").map(Number);
  // 60 fps first, then whatever is offered: more captures of each display frame steady a transfer more than more
  // pixels of one. The rate has to be REQUIRED to win. Ideals are traded by distance and the size outweighs the rate:
  // with 1080p picked, 1080p at 30 (0.5 away) beats 720p at 60 (0.67 away). 50 and not 60, since cameras report 59.94.
  // The rear camera is required with it, or a phone whose front camera alone does 60 hands that one over. A machine
  // with no rear camera fails on facingMode and gets the second try; a rear camera that cannot do 60 skips it.
  // A camera picked by hand says which one and nothing else, so the facingMode tries drop out and only the rate is relaxed.
  const size = { width: { ideal: w }, height: { ideal: h } }, fast = { min: 50, ideal: 60 }, id = $("cam").value;
  const at = (rate) => id === "auto" ? [{ ...size, facingMode: { exact: "environment" }, frameRate: rate }, { ...size, frameRate: rate }] : [{ ...size, deviceId: { exact: id }, frameRate: rate }];
  const ask = $("fps").value, key = `${id}|${$("res").value}|${ask}`;
  let stream = null;
  rateNote = "";
  // A rate from the menu is held at the size asked for or not at all, where 60 first trades the size away. No call
  // lists what a camera offers at each size, so the menu learns it by trying: refused outright, or given only at a
  // smaller size (either way round, since a phone held upright reports its frames turned), and the option is greyed
  // for this camera and size and the run goes on at 60 first. The choice stays, so a size that offers it gets it back.
  if (ask !== "auto" && !refused.has(key)) {
    stream = await openCamera([...at(RATES[ask]), null]);
    const t = stream?.getVideoTracks()[0].getSettings();
    if (t && (Math.max(t.width, t.height) < Math.max(w, h) || Math.min(t.width, t.height) < Math.min(w, h))) { stream.getTracks().forEach((x) => x.stop()); stream = null; }
    if (!stream) refused.add(key);
  }
  if (ask !== "auto" && !stream) rateNote = `${ask} fps is not offered at ${w}x${h}, running 60 first`;
  stream ??= await openCamera([...at(fast), id === "auto" ? { ...size, facingMode: { ideal: "environment" }, frameRate: { ideal: 60 } } : { ...size, deviceId: { exact: id }, frameRate: { ideal: 60 } }]);
  showRates();
  track = stream.getVideoTracks()[0];
  srcCrop = await cropAtSource(track);
  console.info(`camera source crop: ${srcCrop}`);
  // What this browser will let us say about exposure, recorded rather than assumed. The Image Capture extensions
  // (exposureMode, exposureTime, exposureCompensation, iso, whiteBalanceMode) are a Chromium thing in practice and
  // this rig's figures come from Firefox for Android, so the honest answer for any given phone is whatever its own
  // getCapabilities() says. It lands in research/rig/stats.jsonl, so a recorded run carries the answer with it.
  try {
    const c = track.getCapabilities?.() ?? {};
    photoCaps = Object.fromEntries(["exposureMode", "exposureCompensation", "exposureTime", "iso", "whiteBalanceMode", "colorTemperature", "brightness", "contrast", "focusMode", "focusDistance", "zoom", "torch"]
      .filter((k) => k in c).map((k) => [k, Array.isArray(c[k]) ? c[k] : c[k] && typeof c[k] === "object" ? `${c[k].min}..${c[k].max}${c[k].step ? " step " + c[k].step : ""}` : String(c[k])]));
  } catch { photoCaps = {}; }
  video.srcObject = stream; await video.play();
  await squareOrBack(track);
  try { wake = await navigator.wakeLock?.request("screen"); } catch {}
  camLabel = track.label || "Camera"; camRate = Math.round(track.getSettings().frameRate || 0);
  listCameras();   // labels are blank until a camera has been granted, so the menu is worth filling only now
  cur = fresh(); buckets = []; lastPresented = 0; prevArrive = 0; lastStamp = undefined;
  poolVfOff = ""; vfFormat = ""; tapNone = 0;   // ai: a new stream may carry another pixel format, so the pool tries VideoFrames again
  best = freshBest();   // per run: what matters is drift within one sitting
  resize();
  if (gpu) { gpu.trackWhy = ""; sendTrack(); }   // ai: a new track is offered to the worker whatever the last one came to
  camAt = performance.now(); firstDecodeMs = null; answered = false;
  if (!looping) { looping = true; onFrame(); }   // a second loop would count every frame twice
}
// ai: The centre square at the source (2026-09-25). Every camera frame reaches Chrome's GPU process
// ai: whole, and at 1440 and 4K that process fell half a second behind on frames whose pixels the crop mostly throws
// ai: away (STATUS "Why 1440 and 4K drop frames on the phone"). resizeMode crop-and-scale with a square of the short
// ai: side makes the track's frames that square, so no stage after the camera sees the rest, and the centre crop
// ai: becomes the whole frame. The rate is held. Where resizeMode is not a supported constraint (a browser that might
// ai: scale to the square and squash it) nothing is asked; a refusal, or an answer that is not the square or is
// ai: slower, puts the track's first constraints back, as does ?srccrop=full (a menu until 2026-10-01; the fallback by
// ai: hand, and the arm to compare against). Continuous focus rides in the same call, since a later applyConstraints
// ai: replaces the whole set and would drop the square. Returns what happened, for the stats row; the camera's start
// ai: checks the frames that then arrive (squareOrBack).
async function cropAtSource(track) {
  // A lens hunting between frames is the worst thing for a dense code; ask for continuous focus where the device offers it.
  const focus = track.getCapabilities?.().focusMode?.includes("continuous") ? [{ focusMode: "continuous" }] : [];
  const first = firstConstraints = track.getConstraints(), was = track.getSettings(), s = Math.min(was.width, was.height), fr = was.frameRate;
  const plain = async (why) => { try { if (focus.length) await track.applyConstraints({ ...first, advanced: focus }); } catch {} return why; };
  if (CROP !== "square") return plain("off: crop=full");
  if (LAB("srccrop") === "full") return plain("off: srccrop=full");
  if (!navigator.mediaDevices.getSupportedConstraints?.().resizeMode) return plain("off: no resizeMode here");
  if (!(s > 0) || was.width === was.height) return plain(`none needed: ${was.width}x${was.height}`);
  let why;
  try {
    await track.applyConstraints({ width: { exact: s }, height: { exact: s }, resizeMode: "crop-and-scale", ...(fr ? { frameRate: { min: Math.floor(0.9 * fr), ideal: fr } } : {}), advanced: focus });
    const now = track.getSettings();
    if (now.width === s && now.height === s && !(fr && now.frameRate < 0.9 * fr)) return `square ${s} of ${was.width}x${was.height}`;
    why = `answered ${now.width}x${now.height}${now.frameRate ? " @" + Math.round(now.frameRate) : ""}`;
  } catch (e) { why = `refused: ${e.name}${e.constraint ? " " + e.constraint : ""}`; }
  return putBack(track, why);
}
// ai: The track's first constraints again, with continuous focus where it was asked for: the camera as it was opened.
async function putBack(track, why) {
  const focus = track.getCapabilities?.().focusMode?.includes("continuous") ? [{ focusMode: "continuous" }] : [];
  try { await track.applyConstraints({ ...firstConstraints, advanced: focus }); } catch (e) { why += `; put back failed: ${e.name}`; }
  return `off: ${why}`;
}
// ai: The frames that arrive after a square was granted must be square: a browser that granted it and delivers
// ai: something else is put back to the full frame, the centre crop taken in the decoder as before.
async function squareOrBack(track) {
  if (!srcCrop.startsWith("square")) return;
  if (video.videoWidth > 0 && video.videoWidth === video.videoHeight) return;
  srcCrop = await putBack(track, `granted a square, delivered ${video.videoWidth}x${video.videoHeight}`);
  console.info(`camera source crop: ${srcCrop}`);
}
let firstConstraints = {};
let srcCrop = "";
// The last try is the fallback. A first try refused on anything but facingMode goes straight to it, since the one
// between only relaxes facingMode and would hand over a front camera that does what the rear one cannot. A null
// last try means there is no fallback, and the answer is null.
async function openCamera(tries) {
  for (let k = 0; ; k++) {
    if (!tries[k]) return null;
    try { return await navigator.mediaDevices.getUserMedia({ audio: false, video: tries[k] }); }
    catch (e) { if (e.name !== "OverconstrainedError" || k === tries.length - 1) throw e; if (k === 0 && e.constraint !== "facingMode") k = tries.length - 2; }
  }
}
// 30 is for a sender at 24 or 30: at 60 the phone decodes each display frame two or three times and heats for it.
// max as well as min, so no more than 30 arrive: Chrome thins a faster mode to it rather than refuse.
const RATES = { 30: { min: 25, ideal: 30, max: 30 }, 60: { min: 50, ideal: 60 } };
const refused = new Set();   // camera|size|rate, for this page load
let rateNote = "";
function showRates() {
  for (const o of $("fps").options) {
    if (o.value === "auto") continue;
    o.disabled = refused.has(`${$("cam").value}|${$("res").value}|${o.value}`);
    o.text = `${o.value}${o.disabled ? " (unavailable)" : ""}`;
  }
}
// ai: The browser's limit (#first, 2026-10-09), before the first camera start on this browser: shown until a start goes
// ai: from it (its Start camera, kept as recv:limitSeen), Cancel or Escape leaving it for the next press. Home's Receive
// ai: (?auto) shows it too, being how most starts come; ?seen (the checks') takes it as seen. A restart on a camera
// ai: setting's change never shows it.
const LIMIT_SEEN = "recv:limitSeen";
let limitHere = PARAMS.has("seen");
const limitSeen = () => { try { return limitHere || localStorage.getItem(LIMIT_SEEN) === "1"; } catch { return limitHere; } };
$("firstStart").onclick = () => {
  limitHere = true;
  try { localStorage.setItem(LIMIT_SEEN, "1"); } catch {}
  $("first").close();
  start();
};
$("firstCancel").onclick = () => $("first").close();
$("go").onclick = () => (track ? stopCamera() : limitSeen() ? start() : $("first").showModal());
$("railCam").onclick = () => $("go").onclick();
$("railOpen").onclick = () => $("open").onclick();
$("railSave").onclick = () => $("save").click();
// A new resolution, rate or camera needs a new stream.
// ai: Not in the middle of a start, which would be stopped under itself: the choice then takes effect at the next start.
$("res").onchange = $("cam").onchange = $("fps").onchange = () => { showRates(); if (track && !ui.starting) { stopCamera(); start(); } };
// Which cameras this device has. A phone reports its rear lenses separately (wide, ultrawide, tele) and
// facingMode picks one of them for you, which is often not the one that focuses closest; this menu says which.
// Before permission the browser hands back one unnamed entry with an empty id, so there is nothing to list yet.
async function listCameras() {
  const sel = $("cam"), want = sel.value;
  const devs = (await navigator.mediaDevices?.enumerateDevices?.().catch(() => []) ?? []).filter((d) => d.kind === "videoinput" && d.deviceId);
  const auto = new Option(camLabel || "Camera", "auto");
  auto.hidden = auto.disabled = true;
  sel.replaceChildren(auto, ...devs.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
  sel.value = [...sel.options].some((o) => o.value === want) ? want : "auto";
  // ai: the camera last chosen here, once the list has it (a list read at load, before any start, where the browser gives
  // ai: ids; never under a running camera, which would then not be the one the menu names)
  if (!track && sel.value === "auto" && !new URLSearchParams(location.search).has("cam")) restoreSaved(sel, "recv:cam");
  showCamera();
}
navigator.mediaDevices?.addEventListener?.("devicechange", listCameras);
// The browser drops the wake lock whenever the page is hidden; take it back on return.
document.addEventListener("visibilitychange", async () => { if (track && document.visibilityState === "visible") try { wake = await navigator.wakeLock?.request("screen"); } catch {} });
// Controls can be preset from the URL: recv.html?workers=4&res=1280x720&auto
// ai: Settings' menus as the user last left them (localStorage, since 2026-09-29);
// ai: the URL's presets after, so a scripted run still gets what it names. The camera's menu fills later (listCameras).
for (const el of $("dev").querySelectorAll("select")) persist(el, `recv:${el.id}`);
listCameras();
for (const [k, v] of new URLSearchParams(location.search)) if ($(k) && "value" in $(k) && [...($(k).options ?? [])].some((o) => o.value === v)) $(k).value = v;
// ai: Auto's adapter asked first, so the first decoder made is the one kept (a pool worker made and killed before)
if ($("dec").value === "auto") await askGpu();
// ai: LIZARD's decoders from the start, told nothing.
setConfig(lizardConfig(), true);
// ai: No WebGPU (Firefox for Android): the menu says so, and ?dec=gpu falls back with the reason on the page.
if (!("gpu" in navigator)) { const o = [...$("dec").options].find((x) => x.value === "gpu"); o.disabled = true; o.text = "GPU (no WebGPU here)"; }
setGrab(LAB("grab") === "gl" ? "gl" : "canvas");
showCrop();
showDecoder(); showCamera();
resize();   // ai: the pool's first worker, or the GPU worker when the decoder (or auto) asks for it
showState();
if (new URLSearchParams(location.search).has("auto")) $("go").onclick();
