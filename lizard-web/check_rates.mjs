// Self-check of the rig's rate arithmetic. Builds a camera clip whose true rates are known (display 24 fps
// filmed at 30 fps, every frame clean), plays it to the receiver page through Chrome's fake camera, and
// compares what the page reports with what was put in. Runs its own server instance on port 8090, so a
// live session on 8080 is not disturbed. About 40 s.
//
// ai: The receiver is blind (2026-09-26): nothing is posted to it, and the rig here only takes its development logs (the
// ai: stats rows this script reads, and the finished file, ?save=post). It counts bad blocks from the light (a test frame
// ai: known by its blocks' bytes, sim/phy.mjs blockJudge); a Lizard run fails if it held none to the stream. LIE=1, the
// ai: negative control: every LIE_EVERY-th display frame's block 1 carries its id's bytes with one bit flipped, its CRC
// ai: right, and the run passes only if the page counts bad blocks. REPICK=<subch> (the test stream): the clip's second
// ai: half at that version, as a sender re-picks mid-stream (ids go on from where the first half left them), held to
// ai: both halves' version read through the lens and decoded at the display's rate.
// Files (XFER=600k by default, a comma list of the cases below): a Lizard file as the sender page sends it, a chunked
// ai: transfer (sim/xfer.mjs), its header in the light only, as the sender sends it, so
// the receiver has the light and nothing else. The page hands the finished file back (recv.html?save=post) and it is
// ai: held to the file sent: the same bytes, and BLAKE3 from the wasm (the vendored C, proved against the official
// ai: vectors by test/xfer_wasm.mjs) equal to the sender's root; b3sum, where installed, must agree too but is not required.
// ai: STORE=memory runs the receiver without OPFS. DEC=gpu HWGPU=1 (below) needs an X display on :99, which the lab's
// ai: scripts/tools/gpu_display.sh brings up (not published: elsewhere start one there, e.g. Xvfb :99, first).
import { openSync, writeSync, closeSync, mkdirSync, rmSync, readFileSync, existsSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { makePhy, sourceFor } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
const zstd = async () => { await initZstd(); return Z; };   // ai: the sender compresses each chunk (sim/xfer.mjs, 2026-10-05)
import { XferSender, blake3, hex, DEFAULT_LOG2, PAYLOAD } from "../liblizard/sim/xfer.mjs";
import { N_FOR } from "../liblizard/sim/lizard_pick.mjs";

const at = (p) => fileURLToPath(new URL(p, import.meta.url));
const dir = process.argv[2] ?? at("../research/build/check"), CW = +(process.env.CW ?? 1920), CH = +(process.env.CH ?? 1080), FD = 24, FC = 30, ONLY_TEST = process.env.ONLY === "test", FRAMES = ONLY_TEST ? 330 : 420;
// ai: TIERS=1: rates by ring and a power tilt, the sender's dense option.
// stream: the same generator the live sender uses, so this measures what the demo measures.
// SUBCH picks the Lizard version (96 unless asked; a bigger one needs a clip tall enough for its picture, CW and CH).
const SUBCH = +(process.env.SUBCH ?? 96), REPICK = +(process.env.REPICK ?? 0), LIE = process.env.LIE === "1", LIE_EVERY = 24;
const spec = { phy: "focus", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: FD, ...(process.env.TIERS ? { tiers: [[6, 8], [4, 12], [2, 12]], tilt: 9 } : {}), variants: [{ name: "rx" }] };
// VERIFY=1: the receiver's crop, checked from outside. The clip gets a grey marker (60, 100, 140, 180, clockwise
// from the top left) 4 to 20 px inside each corner of what the page should be decoding (the centred square of the
// short side, or the whole frame with CROP=full), and the page reports what it sent to the decoder at three points
// of each corner: one on the marker, two on the last pixel of background beside it. A crop off by one pixel fails.
const VERIFY = !!process.env.VERIFY, MARK = [60, 100, 140, 180], SHIFT = +(process.env.SHIFT ?? 0);   // SHIFT=1 puts the markers a pixel off, which must fail: the check checked
mkdirSync(dir, { recursive: true });
// A port per run: a run started right after another would otherwise meet the dying server of the last.
const PORT = 8090 + (process.pid % 400), RECEIVED = resolve(dir, "received"), server = spawn("node", [at("./server.mjs")], { env: { ...process.env, RIG_HTTP: PORT, RIG_UPLOADS: "1", RIG_HTTPS: PORT + 363, RIG_FILES: RECEIVED }, stdio: "ignore" });
let browser = null;
process.on("exit", () => { server.kill(); if (browser) try { process.kill(-browser.pid, "SIGKILL"); } catch {} });
// One retry: building a clip takes longer than the server's keep-alive, and the next request can land on the socket it just closed.
const once = (p, body) => fetch(`http://localhost:${PORT}${p}`, body ? { method: "POST", body: JSON.stringify(body) } : undefined).then((r) => r.json());
const api = (p, body) => once(p, body).catch(() => once(p, body));

// ai: MOTION=<A> (clip px; 0 included): the symbol moved on every camera frame as a hand holding the phone moves it, in
// ai: whole clip pixels about the centre, each axis on its own: a drift of two sines (0.45 A at 0.15 to 0.25 Hz, 0.25 A
// ai: at 0.4 to 0.6 Hz) plus a shake of two (0.2 A at 2 to 3 Hz, 0.1 A at 5 to 7 Hz), each sine's frequency and phase
// ai: drawn from a generator seeded the same every run, so every arm sees the same path. |dx| and |dy| stay within A; an
// ai: axis steps at most 0.33 A a frame (the sum of 2 pi f a / 30). No blur, turn or scale: the symbol only moves. Set,
// ai: the test stream decodes MOTION_S longer (20 s on the GPU), so a run holds several periods of the drift; the clip
// ai: prints the path it drew. For what the receiver's tracker costs the GPU decoder (STATUS "What the tracker costs
// ai: the GPU decoder").
const MOTION = process.env.MOTION === undefined ? null : +process.env.MOTION, MOTION_S = MOTION === null ? 0 : 10;
function motionPath(A) {
  let s = 1;
  const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  const axis = () => [[0.45, 0.15, 0.25], [0.25, 0.4, 0.6], [0.2, 2, 3], [0.1, 5, 7]].map(([a, lo, hi]) => ({ a: a * A, f: lo + (hi - lo) * rnd(), p: 2 * Math.PI * rnd() }));
  const ax = axis(), ay = axis(), at = (parts, t) => parts.reduce((v, c) => v + c.a * Math.sin(2 * Math.PI * c.f * t + c.p), 0);
  const offsetAt = (i) => [Math.round(at(ax, i / FC)), Math.round(at(ay, i / FC))];
  offsetAt.parts = { x: ax, y: ay };
  return offsetAt;
}

// ai: frameOf(seq): the display frame seq, as phy.frame paints it (or a promise of it); called once a display frame, in order.
async function clip(path, frameOf, frames = FRAMES) {
  const fd = openSync(path, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
  writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
  // ai: The crop the receiver decodes: the centred square of the short side, or the whole clip under CROP=full.
  const cw = process.env.CROP === "full" ? CW : Math.min(CW, CH), ch = process.env.CROP === "full" ? CH : Math.min(CW, CH), x0 = Math.floor((CW - cw) / 2), y0 = Math.floor((CH - ch) / 2);
  const A = MOTION ?? 0, move = motionPath(A), drawn = { max: [0, 0], step: [0, 0], stepSum: 0 };
  let shown = -1, sym = null, sw = 0, sh = 0, ox = 0, oy = 0, prev = [0, 0];
  for (let i = 0; i < frames; i++) {
    const seq = Math.floor((i * FD) / FC), d = move(i);
    if (seq !== shown) {
      const fr = await frameOf(seq), PX = fr.drive ? 1 : 2;
      sw = fr.w * PX; sh = fr.h * PX; ox = (CW - sw) >> 1; oy = (CH - sh) >> 1;
      // ai: One clip pixel a sample (a grey frame), so the whole symbol, its margin included, must fit the square the
      // ai: receiver decodes (the full clip under CROP=full); rows past the clip were dropped silently before, and at
      // ai: CW=1280 CH=720 LIZARD-96's 744 px lost its margin on every side (found 2026-09-26). Under MOTION it must fit
      // ai: wherever the motion takes it.
      if (ox - A < x0 || oy - A < y0 || ox + sw + A > x0 + cw || oy + sh + A > y0 + ch) throw new Error(`clip: a ${sw} x ${sh} px frame${A ? ` moved up to ${A} px each way` : ""} does not fit the receiver's ${cw} x ${ch} crop of a ${CW} x ${CH} clip; raise CW and CH${A ? " or lower MOTION" : ""}`);
      if (sym?.length !== sw * sh) sym = Buffer.alloc(sw * sh);
      for (let r = 0; r < sh; r++) for (let c = 0; c < sw; c++) { const k = ((r / PX) | 0) * fr.w + ((c / PX) | 0); sym[r * sw + c] = fr.drive ? Math.round(20 + 215 * fr.drive[k]) : fr.dark[k] ? 20 : 235; }
    }
    if (seq !== shown || A) {
      y.fill(235);
      for (let r = 0; r < sh; r++) sym.copy(y, (oy + d[1] + r) * CW + ox + d[0], r * sw, (r + 1) * sw);
      if (VERIFY) {
        const xm = x0 + SHIFT;
        [[0, 0], [1, 0], [1, 1], [0, 1]].forEach(([cx, cy], k) => { for (let r = 4; r < 20; r++) for (let c = 4; c < 20; c++) y[(y0 + (cy ? ch - 1 - r : r)) * CW + xm + (cx ? cw - 1 - c : c)] = MARK[k]; });
      }
      shown = seq;
    }
    for (const k of [0, 1]) { drawn.max[k] = Math.max(drawn.max[k], Math.abs(d[k])); if (i) drawn.step[k] = Math.max(drawn.step[k], Math.abs(d[k] - prev[k])); }
    if (i) drawn.stepSum += Math.hypot(d[0] - prev[0], d[1] - prev[1]);
    prev = d;
    writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
  }
  closeSync(fd);
  if (A) {
    const hz = (p) => p.map((c) => `${c.a.toFixed(0)} px at ${c.f.toFixed(2)} Hz`).join(", ");
    console.log(`motion ${A} px, a camera frame at ${FC} fps: x ${hz(move.parts.x)}; y ${hz(move.parts.y)}; drawn over ${frames} frames: at most ${drawn.max[0]} and ${drawn.max[1]} px off centre, a step at most ${drawn.step[0]} and ${drawn.step[1]} px, ${(drawn.stepSum / (frames - 1)).toFixed(1)} px a frame on average`);
  }
}

// Headless defaults to SwiftShader, so without one of these any timing of the GL grab measures a software
// rasteriser and nothing else. HWGPU=1 is the discrete card, which reaches its memory across PCIe and is the
// WRONG shape for this question. IGPU=1 is the CPU's own graphics, which shares system memory with the CPU the
// way a phone's does, so it is the closer analogue of the case the GL grab exists for.
// ai: DEC=gpu opens the receiver with ?dec=gpu (the GPU decoder, recv-gpu-worker.mjs) and turns WebGPU on; with
// ai: neither IGPU nor HWGPU its adapter is SwiftShader, which checks the wiring and times nothing. Headless Chrome
// ai: offers WebGPU only SwiftShader with the NVIDIA driver in reach, so DEC=gpu HWGPU=1 opens a window on the nested
// ai: display :99 (the lab's scripts/tools/gpu_display.sh, not published), as scripts/gpu/harness/run.mjs does, and asks
// ai: for the high-performance adapter (the default there is the iGPU). IGPU=1 stays headless: with the AMD driver alone
// ai: WebGPU gets the iGPU.
const DEC = process.env.DEC === "gpu" ? "gpu" : "cpu", WINDOWED = DEC === "gpu" && !!process.env.HWGPU && !process.env.IGPU, DISPLAY_SH = at("../scripts/tools/gpu_display.sh");
if (WINDOWED && !existsSync("/tmp/.X11-unix/X99") && existsSync(DISPLAY_SH)) spawnSync(DISPLAY_SH, { stdio: "ignore", timeout: 15000 });
if (WINDOWED && !existsSync("/tmp/.X11-unix/X99")) throw new Error(`DEC=gpu HWGPU=1 needs an X display on :99: ${existsSync(DISPLAY_SH) ? "scripts/tools/gpu_display.sh did not bring it up" : "start one there first (e.g. Xvfb :99; the lab's scripts/tools/gpu_display.sh is not published)"}`);
const GPU = process.env.IGPU
  ? { args: ["--enable-gpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist"], env: { VK_ICD_FILENAMES: "/usr/share/vulkan/icd.d/radeon_icd.json" } }
  : WINDOWED
  ? { args: ["--enable-features=Vulkan", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--window-size=1280,720", "--window-position=0,0", "--no-first-run", "--no-default-browser-check", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"], env: { DISPLAY: ":99" }, query: "&pref=high-performance" }
  : process.env.HWGPU
  ? { args: ["--enable-gpu", "--use-gl=angle", "--use-angle=gl"], env: {} }
  : DEC === "gpu"
  ? { args: ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"], env: {} }
  : { args: ["--disable-gpu", "--enable-unsafe-swiftshader"], env: {} };
// until(stats): stop as soon as it says so, polled a second at a time, and never later than seconds. Chrome's resident
// memory is sampled meanwhile: the renderer's peak (the page and every worker) and the whole tree's.
// ai: Every stats row polled meanwhile comes back as the result's rows, [seconds since the start, row].
const chrome = (path, seconds, until = null, extra = "") => new Promise((done) => {
  const p = spawn("google-chrome", [...(WINDOWED ? [] : ["--headless=new"]), "--no-sandbox", ...(process.env.CHROME_LOG ? ["--enable-logging=stderr", "--v=0"] : []), ...GPU.args, "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${path}`, `--user-data-dir=${dir}/profile`, `http://localhost:${PORT}/lizard-web/${process.env.APP ? "app/" : ""}recv.html?auto${extra}${process.env.WORKERS ? "&workers=" + process.env.WORKERS : ""}${process.env.DELAY ? "&delay=" + process.env.DELAY : ""}${process.env.CROP ? "&crop=" + process.env.CROP : ""}${process.env.SRCCROP ? "&srccrop=" + process.env.SRCCROP : ""}${VERIFY ? "&verify" : ""}${process.env.PROF ? "&prof" : ""}${process.env.DRAW ? "&draw=" + process.env.DRAW : ""}${process.env.GRAB ? "&grab=" + process.env.GRAB : ""}${process.env.SAMPLE ? "&sample=" + process.env.SAMPLE : ""}${process.env.GLCS ? "&glcs=" + process.env.GLCS : ""}${process.env.GRABCMP ? "&grabcmp" : ""}${process.env.REC ? "&rec=" + process.env.REC : ""}&dec=${DEC}${GPU.query ?? ""}&res=${CW >= 3840 ? "3840x2160" : CW >= 2560 ? "2560x1440" : CW >= 1920 || CH >= 1920 ? "1920x1080" : "1280x720"}`], { stdio: process.env.CHROME_LOG ? ["ignore", "ignore", openSync(process.env.CHROME_LOG, "a")] : "ignore", detached: true, env: { ...process.env, ...GPU.env } });
  browser = p;
  const mem = { renderer: 0, total: 0 }, t0 = Date.now();
  const sample = () => {
    const rows = spawnSync("ps", ["-o", "rss=,args=", "-g", String(p.pid)], { encoding: "utf8" }).stdout.split("\n").filter(Boolean).map((l) => [+l.trim().split(/\s+/)[0], l]);
    mem.total = Math.max(mem.total, rows.reduce((a, [k]) => a + k, 0) * 1024);
    for (const [k, l] of rows) if (l.includes("--type=renderer")) mem.renderer = Math.max(mem.renderer, k * 1024);
  };
  // Chrome is a tree of processes. Killing only the parent leaves children writing the profile, and the cleanup below fails.
  const rows = [];
  const end = async () => { clearInterval(tick); const s = (await api("/api/stats")).stats; s.chromeMem = mem; s.rows = rows; p.once("exit", () => setTimeout(() => done(s), 500)); try { process.kill(-p.pid, "SIGKILL"); } catch { p.kill("SIGKILL"); } };
  // Without until, exactly at seconds, as the rates above are read from the seconds before the clip loops.
  if (!until) setTimeout(end, seconds * 1000);
  let busy = false;
  const tick = setInterval(async () => {
    if (busy) return;
    busy = true; sample();
    try {
      const st = (await api("/api/stats")).stats ?? {};
      if (st.processedFps !== undefined) rows.push([(Date.now() - t0) / 1000, st]);
      if (until && (Date.now() - t0 > seconds * 1000 || until(st))) { await new Promise((r) => setTimeout(r, 1500)); await end(); return; }
    } catch {}
    busy = false;
  }, 1000);
});
// ai: Which decoder the page ran, from its last stats row. Asked for the GPU and given the pool is a failure: the rates
// ai: printed would be the pool's under the GPU's name.
function decoderLine(s, indent = "") {
  const f = (v) => (typeof v === "number" ? v.toFixed(1) : "?");
  console.log(`${indent}decoder: ${s.decoder === "gpu"
    ? `GPU, ${s.gpuAdapter ?? "adapter not reported"}, ${s.gpuPrecision ?? "?"}, bank ${s.gpuBank ?? "?"}: ${f(s.gpuMs)} ms a frame, B ${s.gpuB ?? "?"}, ${s.gpuInflight ?? "?"} in flight, ${s.gpuQueued ?? "?"} queued, ${s.gpuDropped ?? "?"} dropped`
    : `CPU pool, ${s.workers ?? "?"} worker${s.workers === 1 ? "" : "s"}${s.decoderNote ? ` (${s.decoderNote})` : ""}`}${DEC === "gpu" && s.decoder !== "gpu" ? "  FAILED: DEC=gpu asked for the GPU decoder" : ""}`);
  if (DEC === "gpu" && s.decoder !== "gpu") process.exitCode = 1;
}
// ai: Under DEC=gpu the test stream's 10 s start at the first frame the GPU decoder answers (the page's totals): on
// ai: SwiftShader building its pipelines took about 40 s (2026-09-24), and its first batch of 32 about 17 s after the
// ai: plan (2026-10-04), so a window opened at the plan (gpuB above 0, until then) ended with no frame answered and
// ai: nothing judged. A page that fell back to the pool is let go at once. A file case gets GPU_SLACK_S more, as a
// ai: ceiling: SwiftShader decodes about a frame a second (600,000 B took 164.5 s). The test clip gets GPU_CLIP_S more,
// ai: so the rates are read before the fake camera loops it, which would make new frames repeats: a real adapter plans
// ai: within about 5 s (the 4090), SwiftShader does not, and its rates are not the GPU's anyway.
const GPU_BOOT_S = 120, GPU_SLACK_S = 900, GPU_CLIP_S = 15;
const gpuDecodingFor = (secs) => { let at = 0; return (st) => { if (st.decoder && st.decoder !== "gpu") return true; if (st.totals?.frames > 0 && !at) at = Date.now(); return !!at && Date.now() - at >= secs * 1000; }; };
// ai: How the frames reached the decoder (vframes.mjs) and the page's ms a frame each way, from the run's last 5 s
// ai: window: as VideoFrames (F0 to the GPU worker, which still takes 1 in LUMA_EVERY as luma; to a pool worker, which
// ai: copies the crop to luma, its pixel format named) or as luma by the grab. A pool run that sent no VideoFrame fails:
// ai: headless Chrome's fake camera gives I420 off the track, as the S26 does, so the copyTo path must have run. Under
// ai: DEC=gpu only a real adapter must take them; SwiftShader is sent luma (a VideoFrame import takes it down).
function framesLine(s, indent = "") {
  const gpu = s.decoder === "gpu", must = gpu ? !!(process.env.IGPU || process.env.HWGPU) : true, f = (v) => (typeof v === "number" ? v.toFixed(2) : "?");
  console.log(`${indent}frames to the ${gpu ? "GPU" : "CPU pool"}: ${s.vfSent
    ? `${s.vfSent} as VideoFrames (from the ${s.vfSource}${gpu ? ", F0" : `, ${s.vfFormat || "format not reported"}, copied to luma in the worker`}) and ${s.lumaSent} as luma by the ${s.grabMode} grab${gpu ? `; ${(100 * (s.gpuF0 ?? 0)).toFixed(0)}% of the worker's last second came as VideoFrames` : ""}. The page, a frame: VideoFrame ${f(s.msVideoFrame)} ms, ${s.grabMode} grab ${f(s.msGrab)} ms`
    : `luma only, the ${s.grabMode} grab (${s.vfNote || "no reason given"})`}${must && !s.vfSent ? `  FAILED: no VideoFrame ${gpu ? "on a real adapter" : "reached the pool"}` : ""}`);
  if (must && !s.vfSent) process.exitCode = 1;
}
// ai: The run's decoding from the page's totals (recv.mjs totals, the difference of two rows), from the first row 2 s after
// ai: the first frame answered (the tracker aimed by then) to the last: frames and frames a second, dropped, registered,
// ai: blocks and new bytes a frame, frames with new data a second, KB/s of new bytes on the page's clock, bad, the
// ai: decoder's ms a frame (the GPU's device time), the crop (the #gcrop menu, the share of frames decoded from a tracked
// ai: crop, the mean side, every size with its count), and over the same rows the batch size, the worker's lag from a
// ai: frame's arrival to its answer, and with the stage timing menu on every stage's device ms a frame (bank the proposer,
// ai: describe the classifiers) and their sum. Exact over the interval, where the 5 s window's rates move with
// ai: the batches that land in it.
function totalsLine(s, indent = "") {
  const rows = [...s.rows.map(([, r]) => r), s].filter((r) => r.totals?.frames > 0);
  if (rows.length < 2) return console.log(`${indent}totals: fewer than two rows with a frame answered`);
  const t0 = rows[0].totals.t, from = rows.findIndex((r) => r.totals.t >= t0 + 2), a = rows[from < 0 ? 0 : from].totals, b = rows[rows.length - 1].totals, span = rows.slice(from < 0 ? 0 : from);
  const d = (k) => b[k] - a[k], n = d("frames"), dt = b.t - a.t, per = (v) => (n ? v / n : 0), mean = (f) => { const v = span.map(f).filter((x) => typeof x === "number"); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null; };
  const sizes = Object.entries(b.sizes).map(([k, v]) => [k, v - (a.sizes[k] ?? 0)]).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]);
  const Bs = span.map((r) => r.gpuB).filter((x) => x > 0), st = span.map((r) => r.gpuStageMs).filter(Boolean), f = (v, k = 1) => (v == null ? "-" : v.toFixed(k));
  const stage = (k) => (st.length ? st.reduce((x, m) => x + (k ? m[k] ?? 0 : Object.values(m).reduce((p, q) => p + q, 0)), 0) / st.length : null);
  console.log(`${indent}totals over ${dt.toFixed(1)} s: ${n} frames (${f(n / dt)} a second), ${d("dropped")} dropped, registered ${f(100 * per(d("found")))}%, ${f(per(d("blocks")), 2)} blocks and ${f(per(d("fresh")), 0)} new bytes a frame, ${f(d("withNew") / dt)} frames with new data a second, ${f(d("fresh") / dt / 1000)} KB/s, bad ${d("bad")}, ${f(per(d("ms")), 2)} ms a frame${Bs.length ? `, B ${Math.min(...Bs)} to ${Math.max(...Bs)}, lag ${f(mean((r) => r.gpuLag), 0)} ms` : ""}${st.length ? `; stages (${st.length} rows, ms a frame): all ${f(stage(null), 2)}, ${Object.keys(st[0]).map((k) => `${k} ${f(stage(k), 2)}`).join(", ")}` : ""}`);
  console.log(`${indent}crop (${s.gpuCrop ?? "?"}): ${f(100 * per(d("tracked")))}% of frames from a tracked crop, mean side ${f(Math.sqrt(per(d("px"))), 0)} px; ${sizes.map(([k, v]) => `${k} ${v}`).join(", ") || "no sizes"}`);
}
const row = (name, want, got, unit) => console.log(`${name.padEnd(34)} expected ${want.toFixed(1).padStart(8)}  reported ${got.toFixed(1).padStart(8)} ${unit}  (${want ? ((100 * (got - want)) / want).toFixed(1) : "0"}%)`);

let up = false;
for (let i = 0; i < 20 && !up; i++) { try { up = (await api("/api/info")).http === PORT; } catch {} if (!up) await new Promise((r) => setTimeout(r, 250)); }
if (!up) { console.error(`check_rates: no answer from its own server on port ${PORT} (in use?)`); process.exit(1); }
// 1. Test stream. ONLY=files skips it.
if (process.env.ONLY !== "files") {
  const phy = await makePhy(spec), per = phy.usefulBytes * phy.blocksPerFrame, half = FRAMES + (DEC === "gpu" ? GPU_CLIP_S * FC : 0) + MOTION_S * FC;
  // ai: LIE: block 1 of every LIE_EVERY-th display frame (ids seq * T + 1) painted with a bit of its bytes flipped.
  let lies = 0;
  if (LIE) { const gen = sourceFor("shake256"), T = phy.blocksPerFrame; phy.setSource((id, out) => { gen(id, out); if (id % (LIE_EVERY * T) === 1) { out[0] ^= 1; lies++; } }); }
  // ai: REPICK: the second half at another version, its ids from where the first half's end. Made at the cut, not
  // ai: before: the wasm holds one Lizard codec, and a second Focus replaces the first's.
  let phy2 = null;
  const cut = Math.floor((half * FD) / FC), spec2 = { ...spec, n: N_FOR(REPICK), subch: REPICK };
  await clip(`${dir}/test.y4m`, async (seq) => {
    if (!REPICK || seq < cut) return phy.frame(seq);
    phy2 ??= await makePhy(spec2);
    return phy2.frame(seq, cut * phy.blocksPerFrame + (seq - cut) * phy2.blocksPerFrame);
  }, REPICK ? 2 * half : half);
  const s = REPICK ? await chrome(resolve(dir, "test.y4m"), (2 * half) / FC - 1) : DEC === "gpu" ? await chrome(resolve(dir, "test.y4m"), 10 + MOTION_S + GPU_BOOT_S, gpuDecodingFor(10 + MOTION_S)) : await chrome(resolve(dir, "test.y4m"), 10 + MOTION_S);
  console.log(`test stream, ${phy.label}${phy2 ? `, then ${phy2.label} from display frame ${cut}` : ""}, ${per.toFixed(0)} B per display frame, display ${FD} fps, camera ${FC} fps; nothing posted to the receiver${LIE ? `; ${lies} lying blocks painted (block 1 of every ${LIE_EVERY}th display frame)` : ""}`);
  // ai: Blind, and bad counted from the light: a Lizard run whose page held no block to the stream did not judge.
  const tot = s.totals ?? {}, judgedOk = tot.judged > 0;
  console.log(`decode: ${s.decode ?? "?"}; bad ${tot.bad ?? "?"} of ${tot.judged ?? "?"} blocks held to the test stream from the light over the run${s.decode !== "blind" ? "  FAILED: the receiver did not read blind" : ""}${judgedOk ? "" : "  FAILED: no block judged"}`);
  if (s.decode !== "blind" || !judgedOk) process.exitCode = 1;
  decoderLine(s);
  framesLine(s);
  totalsLine(s);
  row("camera fps", FC, s.capturedFps, "fps");
  row("decoded with new data", FD, s.decodedFps, "fps");
  row("repeats, not counted as decoded", FC - FD, s.repeatFps, "fps");
  // ai: Under REPICK the last window is the second half's, so its rates are held to that version's truth.
  const per2 = phy2 ? phy2.usefulBytes * phy2.blocksPerFrame : per;
  // ai: A batch lands in one second or the next, so the GPU decoder's goodput is held to truth over the 5 s window (new
  // ai: bytes a frame times frames a second), not over the last second.
  if (DEC === "gpu") row("goodput, the 5 s window", (FD * per2) / 1000, (s.freshPerFrame * s.processedFps) / 1000, "KB/s");
  else row("goodput", (FD * per2) / 1000, s.goodputKBs, "KB/s");
  // Per frame the decoder saw. The clip is played with nothing skipped, so that is every captured frame and
  // the expectations below are still the per-camera-frame ones.
  row("new bytes per decoded frame", (per2 * FD) / FC, s.freshPerFrame, "B");
  row("bytes per decoded frame, repeats", per2, s.usefulPerFrame, "B");
  // The rate the sender stated IN THE BAND, come back through the clip, the fake camera and a worker: the only
  // number here that did not travel over the network (src/fmt.h). The ob code has no band of ours, so it is
  // checked only where one was painted.
  // ai: Both decoders read the word (the GPU's F8 since 2026-09-26).
  if (spec.phy === "focus") {
    const v = (phy2 ? REPICK : SUBCH) / 8, ok = s.bandFps === FD && s.bandVersion === v;
    console.log(`band read through the lens: version ${s.bandVersion}, ${s.bandFps} fps stated  ${ok ? "as painted" : `WRONG, the clip was painted at version ${v}, ${FD} fps`}`);
    if (!ok) process.exitCode = 1;
  }
  // ai: A re-pick: each half's version read and decoded at the display's rate in some window of its own (a window is 5 s).
  if (phy2) for (const [v, lab] of [[SUBCH / 8, phy.label], [REPICK / 8, phy2.label]]) {
    const mine = s.rows.filter(([, r]) => r.bandVersion === v), best = Math.max(0, ...mine.map(([, r]) => r.decodedFps ?? 0)), ok = best >= 0.9 * FD;
    console.log(`re-pick, ${lab} (version ${v}): ${mine.length} rows read its word, at best ${best.toFixed(1)} of ${FD} fps decoded with new data  ${ok ? "ok" : "FAILED"}`);
    if (!ok) process.exitCode = 1;
  }
  if (phy2) console.log(`rows (s: version, decoded fps, registered, bad): ${s.rows.map(([t, r]) => `${t.toFixed(0)}: ${r.bandVersion ?? "-"} ${(r.decodedFps ?? 0).toFixed(1)} ${(100 * (r.foundShare ?? 0)).toFixed(0)}% ${r.bad ?? "-"}`).join("; ")}`);
  console.log(`${s.res}\nChrome's renderer peaked at ${(s.chromeMem.renderer / 1048576).toFixed(0)} MB resident, the whole tree ${(s.chromeMem.total / 1048576).toFixed(0)} MB\nworkers ${s.workers} of at most ${s.workersMax}, registered ${(100 * s.foundShare).toFixed(0)}%, skipped ${s.skippedFps.toFixed(1)}/s, bad ${s.bad}, grab ${s.msGrab.toFixed(1)}${s.grabMode === "gl" ? ` (out ${s.msGlSubmit.toFixed(1)} of which upload ${s.glUploadMs.toFixed(1)}, back ${s.msGlTake.toFixed(1)})` : ""} + decode ${s.ms.toFixed(1)} ms${Object.keys(s.prof ?? {}).length ? `\nstages, ms: ${Object.entries(s.prof).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ")}` : ""}\n`);
  if (VERIFY) {
    // Chrome's fake camera may read the clip's luma as limited range, which stretches it: compare with room for that.
    const near = (got, want) => Math.abs(got - want) <= 14 || Math.abs(got - (want - 16) * 1.164) <= 14, probe = s.cropProbe ?? [];
    const okCorners = probe.length === 12 && MARK.every((g, k) => near(probe[3 * k], g) && near(probe[3 * k + 1], 235) && near(probe[3 * k + 2], 235));
    // ai: A camera cropped to the square at the source (recv.mjs cropAtSource) hands over the square itself: the page's
    // ai: crop is the whole frame, so, as on the GL arm, there is no source rect to check.
    // ai: Frames that all went as VideoFrames are cropped in the worker (vframes.mjs lumaOf), which the corners above
    // ai: check; the page's source-rect test runs only on a frame the canvas grab takes (since 2026-09-29).
    const gl = s.grabMode === "gl", atSource = String(s.srcCrop ?? "").startsWith("square"), viaVf = s.vfSent > 0 && !s.lumaSent;
    console.log(atSource
      ? `crop: taken at the camera (${s.srcCrop}), so the page's crop is the whole frame and there is no source rect to disagree with`
      : viaVf
      ? `crop: every frame went as a VideoFrame, cropped in the worker, so there is no source rect to disagree with`
      : gl
      ? `crop: taken on the GPU by texel coordinate, so there is no source rect to disagree with`
      : `crop: drawn ${s.cropDraw === "rect" ? "with a source rect" : "whole and clipped"}, source rect against the whole frame on ${s.cropChecked} frames, ${s.cropDiffer} differed`);
    console.log(`crop: what the decoder got at each corner (marker, then background twice): ${probe.join(" ")}  ${okCorners ? "the true centre, to the pixel" : "NOT the expected square"}\n`);
    // A GL run, a camera cropped at the source or frames all sent as VideoFrames have no source rect to check, so the
    // absence of that check is not a failure there.
    if (!okCorners || s.cropDiffer || (!s.cropChecked && process.env.CROP !== "full" && !gl && !atSource && !viaVf)) process.exitCode = 1;
  }
  // GRABCMP=1: the GPU's luma held against the software canvas's on the same frames. A mean above half a level
  // is a systematic difference and not rounding, and the thing it would most likely be is one path expanding
  // video range while the other does not.
  if (s.cmpFrames) {
    const same = s.cmpMean <= 0.5 && s.cmpMax <= 8;
    console.log(`luma: GPU against canvas over ${s.cmpFrames} frames: max ${s.cmpMax}, mean ${s.cmpMean.toFixed(4)}, ${(100 * s.cmpOver1).toFixed(3)}% differ by more than 1  ${same ? "the same conversion" : "DIFFERENT: check the video range"}\n`);
    if (!same) process.exitCode = 1;
  }
  // ai: LIE: the planted blocks must show as bad; otherwise any bad fails.
  if (LIE) { const ok = tot.bad > 0; console.log(`negative control: ${lies} lying blocks painted, ${tot.bad} bad counted over the run  ${ok ? "caught" : "FAILED: none counted"}`); if (!ok) process.exitCode = 1; }
  else if (s.bad || tot.bad) process.exitCode = 1;
  phy.free(); phy2?.free();
}
// 2. Files. Each case is a file made here, sent through the clip and recovered by the page with nothing from the network
// but the decode settings.
const CASES = {
  "600k": { length: 600000 }, empty: { length: 0 }, one: { length: PAYLOAD }, small: { length: 50000 },
  "10m": { length: 10e6 }, "64m": { length: 64e6 },   // 64 MB is past Wirehair's one-fountain limit (64,000 blocks, 30 MB)
  // Chunk 1 painted wrong in the first lap, every 64th block from its fourth, each with its CRC right: the chunk decodes
  // to the wrong bytes, its chaining value says so, and the page must refuse it and take the chunk again from the
  // second lap. More than one block, so a frame the receiver happened to skip cannot hide the test.
  corrupt: { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1 },
};
const XFER = (process.env.XFER ?? "600k").split(",").filter(Boolean);
if (!ONLY_TEST) {
  await init();
  const M = await initOb();
  for (const name of XFER) {
    const c = CASES[name];
    if (!c) { console.log(`no case ${name}: ${Object.keys(CASES).join(", ")}`); process.exitCode = 1; continue; }
    const bytes = new Uint8Array(c.length), fname = `check-${name}.bin`;
    for (let i = 0, v = c.length + 1; i < bytes.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; bytes[i] = v >>> 16; }
    const phy = await makePhy(spec), B = phy.blocksPerFrame;
    const xs = new XferSender(bytes, { name: fname, type: "application/octet-stream", chunkLog2: c.log2 ?? DEFAULT_LOG2, M, Encoder, Z: await zstd() });
    const src = (id, out) => xs.block(id, out);
    phy.setSource(c.corrupt === undefined ? src : (id, out) => { src(id, out); if (id >>> 18 === c.corrupt && (id & 0x3ffff) < xs.K[c.corrupt] && (id & 63) === 3) out[5] ^= 1; return out; });
    const least = Math.ceil(xs.K.reduce((a, b) => a + b, 0) / B);
    // A lap and a little (two for the corrupt case), a control block a frame at most. The fake camera loops the clip,
    // which repeats ids: that only helps a receiver that missed them the first time.
    const frames = Math.ceil((Math.max(1, xs.sched.lap) * (c.corrupt === undefined ? 1.08 : 2.1)) / Math.max(1, B - 1)) + 24;
    const clipFrames = Math.ceil((frames * FC) / FD), path = `${dir}/file.y4m`, t0 = Date.now();
    await clip(path, (seq) => phy.frame(seq, xs.frameIds(B)), clipFrames);
    const clipSecs = clipFrames / FC;
    console.log(`${name}: ${c.length} B, ${xs.chunks} chunk${xs.chunks === 1 ? "" : "s"} of 2^${xs.chunkLog2}, ${xs.manifest.length} manifest block${xs.manifest.length === 1 ? "" : "s"}, header in the light only; clip ${clipFrames} frames (${clipSecs.toFixed(0)} s) made in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    rmSync(`${RECEIVED}/${fname}`, { force: true });
    const s = await chrome(resolve(path), clipSecs * (c.corrupt === undefined ? 2 : 1.5) + 30 + (DEC === "gpu" ? GPU_SLACK_S : 0), (st) => /complete|FAILED/.test(st.file ?? ""), `&save=post${process.env.STORE ? "&store=" + process.env.STORE : ""}`);
    const m = /in ([\d.]+) s \(([\d.]+) KB/.exec(s.file ?? ""), secs = least / FD;
    console.log(`  ${least} display frames is the least that can carry it`);
    decoderLine(s, "  ");
    framesLine(s, "  ");
    if (!m) { console.log(`  FAILED: no file reported: ${s.file ?? ""} ${JSON.stringify(s.xfer ?? {})}`); process.exitCode = 1; }
    else {
      if (c.length) { row("  transfer time", secs, +m[1], "s"); row("  file rate", c.length / secs / 1000, +m[2], "KB/s"); }
      console.log(`  ${s.file}`);
    }
    // The file the page handed back, against the one sent.
    let got = null;
    try { got = readFileSync(`${RECEIVED}/${fname}`); } catch {}
    const want = hex(xs.root), same = !!got && got.length === bytes.length && Buffer.compare(got, Buffer.from(bytes.buffer, bytes.byteOffset, bytes.length)) === 0;
    const hashes = {};
    if (got) {
      hashes["vendored C (wasm)"] = hex(blake3(M, new Uint8Array(got.buffer, got.byteOffset, got.length)));
      const b3 = spawnSync("b3sum", ["--no-names", `${RECEIVED}/${fname}`], { encoding: "utf8" });
      if (b3.status === 0) hashes.b3sum = b3.stdout.trim();
    }
    const agree = "vendored C (wasm)" in hashes && Object.values(hashes).every((h) => h === want);
    console.log(`  handed back: ${got ? `${got.length} B, ${same ? "the same bytes" : "NOT THE BYTES SENT"}` : "NOTHING"}; BLAKE3 ${want}: ${Object.entries(hashes).map(([k, h]) => `${k} ${h === want ? "agrees" : `DIFFERS (${h})`}`).join(", ") || "none computed"}`);
    const x = s.xfer;
    if (x) {
      const mem = x.mem ?? {};
      console.log(`  header from the ${x.source || "?"}${x.lightSeen ? "" : ", never read from the light"}; ${x.verified ?? "?"} of ${x.header?.chunks ?? "?"} chunks verified, ${x.rejected ?? 0} rejected, manifest ${x.manifestHave ?? 0} of ${x.manifest ?? 0}${x.listOk ? " checked against the root" : ""}; blocks held ${x.store === "opfs" ? "on disk (OPFS)" : `in memory (${(mem.storeBytes / 1048576).toFixed(1)} MB)`}`);
      console.log(`  solving and checking took ${((x.solveMs ?? 0) / 1000).toFixed(2)} s of the fountain worker's time in all`);
      console.log(`  fountain worker: Wirehair heap ${(mem.wirehairHeap / 1048576).toFixed(1)} MB, codec heap ${(mem.codecHeap / 1048576).toFixed(1)} MB; Chrome's renderer peaked at ${(s.chromeMem.renderer / 1048576).toFixed(0)} MB resident (the page and every worker, ${s.workers} decoding at the end), the whole tree ${(s.chromeMem.total / 1048576).toFixed(0)} MB`);
    }
    const ok = same && agree && x?.source === "light" && (c.corrupt === undefined || x?.rejected >= 1) && s.decode === "blind";
    console.log(`  decode: ${s.decode ?? "?"}${s.decode !== "blind" ? "  FAILED: the receiver did not read blind" : ""}; bad ${s.bad}`);
    if (!ok) process.exitCode = 1;
    console.log(`  ${ok ? "ok" : "FAILED"}${c.corrupt !== undefined ? `: the corrupted chunk was ${x?.rejected >= 1 ? "refused and taken again" : "NOT refused"}` : ""}\n`);
    xs.free(); phy.free();
    rmSync(path, { force: true });
  }
}
try { rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 }); } catch (e) { console.log(`could not remove ${dir}: ${e.code}. It holds the test clip, delete it by hand.`); }
process.exit(process.exitCode ?? 0);
