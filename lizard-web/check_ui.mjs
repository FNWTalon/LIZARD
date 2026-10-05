// ai: The pages' UI (send.html and recv.html, rebuilt 2026-09-26; the app's shell and Home, 2026-10-01) in
// ai: headless Chrome, driven through phone.mjs's DevTools calls, at four sizes: phone portrait 412 x 915 at 2.625,
// ai: phone landscape 915 x 412, laptop 1366 x 768, desktop 1920 x 1080. Screenshots of every step go to research/build/ui/.
// ai:   0. The sender: idle (Start disabled until a file, a tap on the middle opens the file chooser; Start and
// ai:      Fullscreen in its controls, the other tool and Settings in the bar), then the test stream on auto, Advanced
// ai:      closed then open: <main>'s box and the version unmoved by it (a resize re-picks mid-stream), nor by the bar's
// ai:      Settings, #c inside <main>; #tx's first line names the version (the lab line #lab did until 2026-10-02), the
// ai:      numbers line (#nums) the speed;
// ai:      the defaults LIZARD-480 at 60 ("Receiving with" went 2026-10-02); Fullscreen, then its end clearing body.full.
// ai:   1. The receiver on Chrome's fake camera, Advanced closed then open: #v visible and above 100 x 100 px with more
// ai:      than 10 frame callbacks a second; at the phone sizes #go and #view on screen with no scrolling; Settings in view
// ai:      from the bar.
// ai:   2. getUserMedia made to throw NotAllowedError, then navigator.mediaDevices taken away: each sentence in red.
// ai:   3. LIZARD-96 through a 1920 x 1080 clip as the fake camera: the test stream read (its state, speed and lab line),
// ai:      then a 300,000 B file: receiving (the meter, progress, size, speed and time left), received (Save holds the
// ai:      file's bytes) with the camera stopped by itself, Save kept, the file named again after a restart with the
// ai:      camera left on (Receive again, which received it anew, went 2026-10-05).
// ai:   4. Home in the same profile at the four sizes: the received file listed, Save holding its bytes, Open, the tips
// ai:      card shown on a fresh profile and remembered once dismissed, Settings, Delete emptying the received files.
// ai: All pages: no horizontal scroll, Advanced closed on a fresh profile and remembered over a reload, a menu's setting
// ai: kept over a reload with a URL preset winning its own load (the sender's surround, the receiver's GPU crop), no
// ai: console error but the ones the start errors put there on purpose.
// ai: node lizard-web/check_ui.mjs, from anywhere (about 70 s, a few of them making the clips). APP=1 runs the same
// ai: against the built app (lizard-web/app/, node lizard-web/pwa/build.mjs first), its worker serving the pages. Exit 1 on a failure.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, openSync, writeSync, closeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { tabs, open, evaluate, navigate, screenshot } from "./phone.mjs";
import { makePhy } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initWh, Encoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { XferSender, DEFAULT_LOG2 } from "../liblizard/sim/xfer.mjs";
import { N_FOR } from "../liblizard/sim/lizard_pick.mjs";

const at = (p) => fileURLToPath(new URL(p, import.meta.url));
const DIR = process.env.APP ? "lizard-web/app" : "lizard-web", OUT = at(process.env.APP ? "../research/build/ui/app" : "../research/build/ui"), PORT = 8700 + (process.pid % 200);
const BASE = `http://localhost:${PORT}/${DIR}/recv.html`, SEND = `http://localhost:${PORT}/${DIR}/send.html`, HOME = `http://localhost:${PORT}/${DIR}/index.html`;
const scratch = mkdtempSync(join(tmpdir(), "lizard-ui-"));
mkdirSync(OUT, { recursive: true });
const server = spawn("node", [at("./server.mjs")], { env: { ...process.env, RIG_HTTP: PORT, RIG_UPLOADS: "1", RIG_HTTPS: PORT + 363, RIG_FILES: join(scratch, "received") }, stdio: "ignore" });
const browsers = [];
process.on("exit", () => { server.kill(); for (const b of browsers) try { process.kill(-b.pid, "SIGKILL"); } catch {} rmSync(scratch, { recursive: true, force: true }); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };

async function chrome(name, extra = []) {
  const port = 9400 + browsers.length + (process.pid % 100);
  const b = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", ...extra, `--remote-debugging-port=${port}`, `--user-data-dir=${join(scratch, name)}`, "about:blank"], { stdio: "ignore", detached: true });
  browsers.push(b);
  for (let t = 0; t < 50; t++) {
    try { const list = await tabs(port); if (list.length) { const s = await open(list[0], port); return watch(s); } } catch {}
    await sleep(200);
  }
  throw new Error(`chrome ${name}: no DevTools on ${port}`);
}
// ai: Every console error, uncaught exception and error log entry, into s.errors.
async function watch(s) {
  s.errors = [];
  s.on("Runtime.consoleAPICalled", (p) => { if (p.type === "error") s.errors.push(`console: ${p.args.map((a) => a.value ?? a.description).join(" ")}`); });
  s.on("Runtime.exceptionThrown", (p) => s.errors.push(`exception: ${p.exceptionDetails.exception?.description ?? p.exceptionDetails.text}`));
  s.on("Log.entryAdded", ({ entry: e }) => { if (e.level === "error") s.errors.push(`log: ${e.text} ${e.url ?? ""}`); });
  await s.send("Runtime.enable"); await s.send("Log.enable"); await s.send("Page.enable");
  return s;
}
async function until(s, expr, ms, what) {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(100)) if (await evaluate(s, expr)) return true;
  // ai: what the page showed when it gave up, and its console errors so far, so a stall names itself
  const seen = await evaluate(s, `(() => { const t = (id) => document.getElementById(id)?.textContent ?? ""; return [t("state"), t("nums"), t("lab"), t("why"), t("file")].filter(Boolean).join(" | "); })()`).catch(() => "");
  throw new Error(`waited ${ms / 1000} s for ${what}; the page: ${seen}${s.errors?.length ? `; errors: ${s.errors.slice(-3).join("; ")}` : ""}`);
}
async function size(s, { w, h, dpr = 1, mobile = false }) {
  await s.send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: dpr, mobile, screenOrientation: w > h ? { type: "landscapePrimary", angle: 90 } : { type: "portraitPrimary", angle: 0 } });
  await s.send("Emulation.setTouchEmulationEnabled", { enabled: mobile, maxTouchPoints: mobile ? 5 : 1 });
}
// ai: The layout at the top of the page, and #v's frame callbacks a second over 2 s.
const MEASURE = `(async () => {
  scrollTo(0, 0);
  const $ = (id) => document.getElementById(id), box = (id) => { const b = $(id).getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, w: b.width, h: b.height }; };
  const v = $("v"), cs = getComputedStyle(v);
  const rate = await new Promise((ok) => { let n = 0; const t0 = performance.now(); const f = () => { n++; const t = performance.now() - t0; if (t < 2000) v.requestVideoFrameCallback(f); else ok(n / (t / 1000)); }; v.requestVideoFrameCallback(f); });
  return { rate, iw: innerWidth, ih: innerHeight, sw: document.documentElement.scrollWidth, view: box("view"), v: box("v"), vDisplay: cs.display, vVisibility: cs.visibility,
    state: $("state").textContent, nums: $("nums").textContent, lab: $("lab").textContent, why: $("why").textContent, go: $("go").textContent, goBox: box("go"), swapBox: box("swap"), swap: $("swap").textContent + " " + $("swap").getAttribute("href"), dev: $("dev").open };
})()`;
// ai: The bar's Settings button (#set) and its check went 2026-10-02.

// ai: A Developer menu's setting kept over a reload (localStorage, ui.mjs persist, 2026-09-29), and a URL preset of it
// ai: winning for its own load without overwriting what was kept; then set back so the rest runs on the defaults.
// ai: the entries in an OPFS folder (the received files' lizard-files), -1 where it is missing
const opfsFiles = (dir) => `(async () => { try { const d = await (await navigator.storage.getDirectory()).getDirectoryHandle("${dir}"); let n = 0; for await (const k of d.keys()) n++; return n; } catch { return -1; } })()`;
async function remembered(s, page, id, value, other, tag) {
  await evaluate(s, `(() => { const el = document.getElementById("${id}"); el.value = "${value}"; el.dispatchEvent(new Event("change")); })()`);
  await navigate(s, page);
  await sleep(300);
  const kept = await evaluate(s, `document.getElementById("${id}").value`);
  await navigate(s, `${page}?${id}=${other}`);
  await sleep(300);
  const url = await evaluate(s, `document.getElementById("${id}").value`);
  await navigate(s, page);
  await sleep(300);
  const after = await evaluate(s, `document.getElementById("${id}").value`);
  check(kept === value && url === other && after === value, `${tag}: #${id} "${value}" kept over a reload ("${kept}"), "${other}" from the URL for that load ("${url}"), "${value}" again after it ("${after}")`);
  await evaluate(s, `(() => { const el = document.getElementById("${id}"); el.value = el.options[[...el.options].findIndex((o) => o.defaultSelected)]?.value ?? el.options[0].value; el.dispatchEvent(new Event("change")); })()`);
}

const SIZES = [
  { name: "phone-portrait", w: 412, h: 915, dpr: 2.625, mobile: true, phone: true },
  { name: "phone-landscape", w: 915, h: 412, dpr: 2.625, mobile: true, phone: true },
  { name: "laptop", w: 1366, h: 768 },
  { name: "desktop", w: 1920, h: 1080 },
];

// ai: 0. The sender at the four sizes.
{
  const s = await chrome("send");
  const MEASURE_SEND = `(() => {
    const $ = (id) => document.getElementById(id), box = (el) => { const b = el.getBoundingClientRect(); return { top: b.top, bottom: b.bottom, left: b.left, right: b.right, w: b.width, h: b.height }; };
    return { iw: innerWidth, ih: innerHeight, sw: document.documentElement.scrollWidth, main: box(document.querySelector("main")), c: box($("c")),
      state: $("state").textContent, nums: $("nums").textContent, lab: $("tx").textContent.split("\\n")[0], go: $("go").disabled, full: document.body.classList.contains("full"), dev: $("dev").open, hint: !$("hint").hidden,
      buttons: [...document.querySelectorAll("#side button:not([hidden]), #side .btn")].map((b) => b.textContent), bar: [...document.querySelectorAll(".bar .brand, .bar h1, .bar > .btn, .bar > button")].filter((b) => b.checkVisibility()).map((b) => b.textContent).join(","), railHome: document.getElementById("railHome")?.getAttribute("href") ?? "" };
  })()`;
  const same = (a, b) => ["top", "left", "w", "h"].every((k) => Math.abs(a[k] - b[k]) < 0.5);
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, SEND);
    await sleep(300);
    const idle = await evaluate(s, MEASURE_SEND);
    if (k === 0) check(!idle.dev, `send ${z.name}: the Developer panel is closed on a fresh profile`);
    check(idle.state === "" && idle.hint && idle.go && idle.buttons.join() === "Start,Fullscreen", `send ${z.name}: idle, no state line, the hint shown, buttons ${idle.buttons.join(" and ")}, Start ${idle.go ? "disabled" : "ENABLED"}`);
    // ai: Home in the bar in portrait, in the rail's last square in landscape, where the bar has the collapser (2026-10-05)
    check(idle.bar === (z.w > z.h ? "Send,Receiver,Sidebar" : "Home,Send,Receiver") && idle.railHome === "index.html", `send ${z.name}: the bar "${idle.bar}", the rail's Home "${idle.railHome}"`);
    // ai: The middle of the page is the only file picker (2026-09-26: the Choose file button deleted): a tap on
    // ai: it must open the chooser, which the DevTools protocol intercepts here instead of showing.
    if (k === 0) {
      await s.send("Page.setInterceptFileChooserDialog", { enabled: true });
      const opened = new Promise((ok) => { s.on("Page.fileChooserOpened", () => ok(true)); setTimeout(() => ok(false), 3000); });
      await s.send("Runtime.evaluate", { expression: `document.querySelector("main").click()`, userGesture: true });
      check(await opened, `send ${z.name}: a tap on the middle opens the file chooser`);
      await s.send("Page.setInterceptFileChooserDialog", { enabled: false });
    }
    await screenshot(s, `${OUT}/send-${z.name}-idle.png`);
    await navigate(s, `${SEND}?subch=auto&fps=24&payload=test&auto`);
    await until(s, `/painted [\\d.]+\\/s/.test(document.getElementById("tx").textContent) && /^LIZARD-\\d+/.test(document.getElementById("tx").textContent) && document.getElementById("nums").textContent !== ""`, 20000, "the test stream painted");
    const seen = [];
    for (const openPanel of [false, true]) {
      await evaluate(s, `document.getElementById("dev").open = ${openPanel}`);
      await sleep(500);
      const m = await evaluate(s, MEASURE_SEND), tag = `send ${z.name}, Advanced ${openPanel ? "open" : "closed"}`, v = +m.lab.match(/LIZARD-(\d+)/)[1];
      seen.push({ ...m, v });
      await screenshot(s, `${OUT}/send-${z.name}-sending-${openPanel ? "open" : "closed"}.png`);
      console.log(`     ${tag}: ${m.iw} x ${m.ih}, main ${m.main.w.toFixed(0)} x ${m.main.h.toFixed(0)} at ${m.main.left.toFixed(0)},${m.main.top.toFixed(0)}; "${m.state}" / "${m.nums}" / "${m.lab}"`);
      check(m.sw <= m.iw, `${tag}: no horizontal scroll (${m.sw} of ${m.iw})`);
      check(m.c.left >= m.main.left - 0.5 && m.c.right <= m.main.right + 0.5 && m.c.top >= m.main.top - 0.5 && m.c.bottom <= m.main.bottom + 0.5, `${tag}: #c inside <main>`);
      // ai: the user's figure is the speed alone for the test stream; the version is #tx's first line's (the lab line's, 2026-10-01 to 2026-10-02)
      check(/^[\d.]+\s(KB|MB)\/s$/.test(m.nums), `${tag}: the numbers line "${m.nums}"`);
    }
    check(same(seen[0].main, seen[1].main) && seen[0].v === seen[1].v, `send ${z.name}: Advanced moves neither <main> nor the version (LIZARD-${seen[0].v})`);
    if (k === 0) {
      // ai: the page stores the panel's state on its toggle event, a task a busy page can run late: reload after it
      await until(s, `localStorage.getItem("send:dev") === "1"`, 5000, "the panel's open state stored");
      await navigate(s, SEND);
      // ai: send.mjs awaits the codec's wasm before anything else, and restores the panel at its end: read it once the
      // ai: module has run (the rate slider's label filled; About is empty on the rig's pages since 2026-10-02), not at the
      // ai: load event
      await until(s, `document.getElementById("fpsOut").textContent !== ""`, 10000, "send.mjs run");
      check(await evaluate(s, `document.getElementById("dev").open`), `send ${z.name}: Advanced's open state remembered over a reload`);
      await remembered(s, SEND, "ring", "3", "1", `send ${z.name}`);
      // ai: the page's defaults, LIZARD-480 at 60 (2026-10-01); "Receiving with", a preset view over them, went 2026-10-02
      const defaults = await evaluate(s, `[document.getElementById("subch").value, document.getElementById("fps").value, !document.getElementById("target")].join()`);
      check(defaults === "480,60,true", `send ${z.name}: the defaults LIZARD-480 at 60, no "Receiving with" (${defaults})`);
      await evaluate(s, `document.getElementById("dev").open = false`);
      await navigate(s, `${SEND}?subch=auto&fps=24&payload=test&auto`);
      await until(s, `/^LIZARD-\\d+/.test(document.getElementById("tx").textContent)`, 20000, "the test stream painted");
    }
    // ai: Fullscreen from a click (a user gesture, as the Fullscreen API asks), then its end from outside the page.
    await s.send("Runtime.evaluate", { expression: `document.getElementById("fs").click()`, userGesture: true });
    await sleep(500);
    const f = await evaluate(s, `({ full: document.body.classList.contains("full"), api: !!document.fullscreenElement, main: document.querySelector("main").getBoundingClientRect().width, iw: innerWidth })`);
    await screenshot(s, `${OUT}/send-${z.name}-full.png`);
    check(f.full && Math.abs(f.main - f.iw) < 1, `send ${z.name}: Fullscreen shows <main> alone (${f.main.toFixed(0)} of ${f.iw} px wide, the API ${f.api ? "in fullscreen" : "refused, body.full alone"})`);
    await evaluate(s, f.api ? `document.exitFullscreen()` : `document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))`);
    await sleep(500);
    check(!(await evaluate(s, `document.body.classList.contains("full")`)), `send ${z.name}: body.full cleared when ${f.api ? "fullscreen ends (fullscreenchange)" : "Escape is pressed"}`);
  }
  check(!s.errors.length, `send: no console errors${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

// ai: 1. The receiver at the four sizes, the panel closed and open.
{
  const s = await chrome("sizes");
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, `${BASE}?auto`);
    await until(s, `document.getElementById("v").videoWidth > 0 && !document.getElementById("go").disabled && document.getElementById("lab").textContent !== ""`, 20000, "the camera and a stats tick");
    if (k === 0) check(!(await evaluate(s, `document.getElementById("dev").open`)), `${z.name}: Advanced is closed on a fresh profile`);
    for (const openPanel of [false, true]) {
      await evaluate(s, `document.getElementById("dev").open = ${openPanel}`);
      await sleep(400);
      const m = await evaluate(s, MEASURE), tag = `${z.name}, Advanced ${openPanel ? "open" : "closed"}`;
      await screenshot(s, `${OUT}/recv-${z.name}-${openPanel ? "open" : "closed"}.png`);
      console.log(`     ${tag}: ${m.iw} x ${m.ih}, view ${m.view.w.toFixed(0)} x ${m.view.h.toFixed(0)} at ${m.view.top.toFixed(0)}, go ${m.goBox.top.toFixed(0)} to ${m.goBox.bottom.toFixed(0)}, ${m.rate.toFixed(1)} callbacks/s; "${m.state}" / "${m.nums}" / "${m.lab}" / "${m.why.replace(/\n/g, " | ")}"`);
      check(m.rate > 10, `${tag}: #v ${m.rate.toFixed(1)} frame callbacks a second`);
      check(m.v.w > 100 && m.v.h > 100 && m.vDisplay !== "none" && m.vVisibility !== "hidden", `${tag}: #v ${m.v.w.toFixed(0)} x ${m.v.h.toFixed(0)}, display ${m.vDisplay}, visibility ${m.vVisibility}`);
      check(m.sw <= m.iw, `${tag}: no horizontal scroll (${m.sw} of ${m.iw})`);
      if (z.phone) check(m.goBox.top >= 0 && m.goBox.bottom <= m.ih && m.view.top >= 0 && m.view.bottom <= m.ih, `${tag}: #go and #view on screen unscrolled`);
      // ai: the switch to the sender page (2026-09-30), in the bar since 2026-10-01, on screen with Start camera
      check(m.swap === "Sender send.html" && m.swapBox.w > 0 && (!z.phone || (m.swapBox.top >= 0 && m.swapBox.bottom <= m.ih)), `${tag}: "${m.swap}" in the bar`);
    }
    // ai: open now: remembered over a reload (then closed again for the next size)
    if (k === 0) {
      await until(s, `localStorage.getItem("recv:dev") === "1"`, 5000, "the panel's open state stored");
      await navigate(s, BASE);
      check(await evaluate(s, `document.getElementById("dev").open`), `${z.name}: Advanced's open state remembered over a reload`);
      await remembered(s, BASE, "gcrop", "tracked", "full", z.name);
      // ai: the decoder's switch, in Settings since 2026-10-01 (outside Advanced, so kept by its own persist)
      await remembered(s, BASE, "dec", "cpu", "auto", z.name);
    }
    await evaluate(s, `document.getElementById("dev").open = false`);
  }
  check(!s.errors.length, `no console errors over the four sizes${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

// ai: 2. Start errors, in red.
{
  const s = await chrome("errors");
  await size(s, SIZES[0]);
  const cases = [
    ["notallowed", `navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Permission denied", "NotAllowedError"));`, /Allow it for this site in the browser's site settings/],
    ["nomediadevices", `Object.defineProperty(Navigator.prototype, "mediaDevices", { get: () => undefined, configurable: true });`, /secure page: open this one over https, or on localhost/],
  ];
  for (const [name, source, want] of cases) {
    const { identifier } = await s.send("Page.addScriptToEvaluateOnNewDocument", { source });
    await navigate(s, BASE);
    await evaluate(s, `document.getElementById("go").click()`);
    await sleep(600);
    const r = await evaluate(s, `(() => { const e = document.getElementById("state"); return { text: e.textContent, bad: e.classList.contains("bad"), color: getComputedStyle(e).color, go: document.getElementById("go").disabled }; })()`);
    await screenshot(s, `${OUT}/recv-error-${name}.png`);
    check(want.test(r.text) && r.bad && r.color === "rgb(176, 0, 32)" && !r.go, `${name}: "${r.text}" in ${r.color}, #go ${r.go ? "disabled" : "enabled"}`);
    await s.send("Page.removeScriptToEvaluateOnNewDocument", { identifier });
  }
  const other = s.errors.filter((e) => !/camera did not start/.test(e));
  check(!other.length && s.errors.length === 2, `the start errors' only console errors are their own two (${s.errors.length})${other.length ? `: ${other.join("; ")}` : ""}`);
  s.close();
}

// ai: 3. The light through the fake camera: 1920 x 1080 clips at 30 fps of a sender at 24, LIZARD-96, one clip pixel a
// ai: sample, as check_rates.mjs builds them. First the test stream (2.5 s, looped), then a file (a lap and 8%).
{
  const CW = 1920, CH = 1080, FD = 24, FC = 30, SUBCH = 96, LEN = 300000, NAME = "holiday-photo.jpg";
  await initWh(); await initZstd();
  const M = await initOb(), bytes = new Uint8Array(LEN);
  for (let i = 0, v = 7; i < LEN; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; bytes[i] = v >>> 16; }
  const phy = await makePhy({ phy: "focus", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: FD, variants: [{ name: "rx" }] }), B = phy.blocksPerFrame;
  async function clip(path, frameOf, n) {
    const fd = openSync(path, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
    writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
    let shown = -1;
    for (let i = 0; i < n; i++) {
      const seq = Math.floor((i * FD) / FC);
      if (seq !== shown) {
        const fr = await frameOf(seq), ox = (CW - fr.w) >> 1, oy = (CH - fr.h) >> 1;
        y.fill(235);
        for (let r = 0; r < fr.h; r++) for (let c = 0; c < fr.w; c++) y[(oy + r) * CW + ox + c] = Math.round(20 + 215 * fr.drive[r * fr.w + c]);
        shown = seq;
      }
      writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
    }
    closeSync(fd);
  }
  const testPath = join(scratch, "test.y4m");
  await clip(testPath, (seq) => phy.frame(seq), 75);
  const xs = new XferSender(bytes, { name: NAME, type: "image/jpeg", chunkLog2: DEFAULT_LOG2, M, Encoder, Z });
  phy.setSource((id, out) => xs.block(id, out));
  const frames = Math.ceil((xs.sched.lap * 1.08) / (B - 1)) + 24, clipFrames = Math.ceil((frames * FC) / FD), path = join(scratch, "file.y4m");
  await clip(path, (seq) => phy.frame(seq, xs.frameIds(B)), clipFrames);
  console.log(`     clips: the test stream, 75 frames; a ${LEN} B file, ${clipFrames} frames (${frames} display frames of LIZARD-${SUBCH})`);

  const t = await chrome("test", [`--use-file-for-fake-video-capture=${testPath}`]);
  await size(t, SIZES[0]);
  await navigate(t, `${BASE}?auto`);
  await until(t, `document.getElementById("state").textContent === "Reading the test stream" && document.getElementById("lab").textContent.includes("% registered, LIZARD")`, 30000, "the test stream read");
  await sleep(1200);
  const ts = await evaluate(t, `(() => { const $ = (id) => document.getElementById(id); return { state: $("state").textContent, nums: $("nums").textContent, lab: $("lab").textContent, why: $("why").textContent, meter: !$("meter").hidden }; })()`);
  await screenshot(t, `${OUT}/recv-phone-portrait-test.png`);
  // ai: no figure for the test stream (its rate the rail's and the lab line's since 2026-10-05); the lab line the rate,
  // ai: the registered share and the format (Developer Tools)
  // ai: the rate in Developer Tools' lab line and the collapsed rail, not the figures (2026-10-05)
  check(ts.state === "Reading the test stream" && ts.nums === "" && /^\d+\sKB\/s, \d+% registered, LIZARD-96 at 24\sfps$/.test(ts.lab) && !ts.meter, `test stream: "${ts.state}" / "${ts.nums}" / "${ts.lab}" / "${ts.why}"`);
  // ai: The native app is recommended only to a sender painting faster than 24 (2026-10-01): not here, and
  // ai: on a clip of the same stream whose word says 30 (a picture a camera frame).
  check(!/Lizard app/.test(ts.why), `at 24 fps no word of the Lizard app: "${ts.why}"`);
  {
    const fastPhy = await makePhy({ phy: "focus", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: FC, variants: [{ name: "rx" }] }), fastPath = join(scratch, "fast.y4m");
    const fd = openSync(fastPath, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
    writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
    for (let i = 0; i < 60; i++) {
      const fr = await fastPhy.frame(i), ox = (CW - fr.w) >> 1, oy = (CH - fr.h) >> 1;
      y.fill(235);
      for (let r = 0; r < fr.h; r++) for (let c = 0; c < fr.w; c++) y[(oy + r) * CW + ox + c] = Math.round(20 + 215 * fr.drive[r * fr.w + c]);
      writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
    }
    closeSync(fd);
    const f = await chrome("fast", [`--use-file-for-fake-video-capture=${fastPath}`]);
    await size(f, SIZES[0]);
    await navigate(f, `${BASE}?auto`);
    await until(f, `/at 30\\sfps/.test(document.getElementById("lab").textContent)`, 30000, "the 30 fps test stream read");
    await sleep(1200);
    const fw = await evaluate(f, `document.getElementById("why").textContent`);
    check(/(^|\n)30 pictures a second is too fast for a browser/.test(fw) && /Lizard app/.test(fw) && /send at 24/.test(fw), `a sender at 30 fps is sent to the Lizard app: "${fw}"`);
  }
  // ai: The light gone (the video paused, as a camera turned to a wall): the state falls back to looking, the code's
  // ai: name leaves the lab line (until 2026-09-26 the band, once read, stayed on both for good).
  await evaluate(t, `document.getElementById("v").pause()`);
  await until(t, `document.getElementById("state").textContent === "Looking for a code"`, 8000, "Looking for a code once the light is gone");
  const gone = await evaluate(t, `[document.getElementById("nums").textContent, document.getElementById("lab").textContent].join(" / ")`);
  check(!/LIZARD/.test(gone), `the light gone: "Looking for a code" / "${gone}"`);
  check(!t.errors.length, `no console errors over the test stream${t.errors.length ? `: ${t.errors.join("; ")}` : ""}`);
  t.close();

  const s = await chrome("file", [`--use-file-for-fake-video-capture=${path}`]);
  await size(s, SIZES[0]);
  await navigate(s, `${BASE}?auto`);
  const STATE = `(() => { const $ = (id) => document.getElementById(id), m = $("meter"); return { state: $("state").textContent, good: $("state").classList.contains("good"), nums: $("nums").textContent, why: $("why").textContent, meter: m.hidden ? null : m.firstElementChild?.style.width, deliver: !$("deliver").hidden, clear: !!$("clear"), open: !$("open").hidden && $("open").classList.contains("primary"), go: $("go").textContent, goSolid: $("go").classList.contains("primary"), save: $("save").textContent, href: $("save").href, download: $("save").download,  railRate: $("railRate").textContent, railDone: !$("railDone").hidden && !$("railOpen").hidden && !$("railSave").hidden, railOff: $("railCam").classList.contains("off") }; })()`;
  await until(s, `/^Receiving/.test(document.getElementById("state").textContent) && parseFloat(document.getElementById("meter").firstElementChild?.style.width) > 40`, 60000, "Receiving, two fifths of the way");
  const a = await evaluate(s, STATE);
  await screenshot(s, `${OUT}/recv-phone-portrait-receiving.png`);
  console.log(`     receiving: "${a.state}" / "${a.nums}", meter ${a.meter}`);
  // ai: progress, size, speed and time left (2026-10-01); the speed and the time left once a second has closed
  check(a.state === `Receiving ${NAME}` && /^\d+%, [\d.]+ of 300\sKB(, \d+\s(s|min) left)?$/.test(a.nums) && a.meter && !a.deliver, `receiving: the state, the meter at ${a.meter} and "${a.nums}"`);
  // ai: the collapsed rail's squares follow the page whether or not it is shown (2026-10-02): the pause square, the
  // ai: last second's rate where the line has one, no tick yet
  check(!a.railOff && !a.railDone && (/ left$/.test(a.nums) ? /^[\d.]+(KB|MB)\/s$/.test(a.railRate) : true), `receiving: the rail's camera on, its rate "${a.railRate}", no check, Open or Save`);
  await until(s, `/^Received/.test(document.getElementById("state").textContent)`, 60000, "Received");
  const b = await evaluate(s, STATE);
  check(b.railOff && b.railDone && b.railRate === "", `received: the rail's play square, the green check, Open and Save, no rate`);
  await screenshot(s, `${OUT}/recv-phone-portrait-received.png`);
  const got = await evaluate(s, `fetch(document.getElementById("save").href).then((r) => r.arrayBuffer()).then((x) => { const u = new Uint8Array(x); let v = 7, same = u.length === ${LEN}; for (let i = 0; same && i < u.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; same = u[i] === ((v >>> 16) & 255); } return { n: u.length, same }; })`);
  console.log(`     received: "${b.state}" / "${b.nums}", meter ${b.meter}, "${b.save}" ${b.href.slice(0, 5)}... as ${b.download}`);
  check(new RegExp(`^Received ${NAME.replace(".", "\\.")}, 300\\sKB in [\\d.]+\\ss$`).test(b.state) && b.good && b.meter === "100%" && b.deliver && b.download === NAME && b.href.startsWith("blob:"), `received: green, the meter full, Save offered as ${b.download}`);
  check(b.open && !b.goSolid && !b.clear, `received: Open the one solid button, "${b.go}" outlined, no Receive again`);
  // ai: the camera off by itself once the file is in (2026-10-01, to save heat)
  check(b.go === "Start camera" && b.nums === "" && b.why === "", `received: the camera stopped by itself ("${b.go}")`);
  check(got.n === LEN && got.same, `Save's file is the ${LEN} bytes sent (${got.n}, ${got.same ? "the same" : "NOT the same"})`);
  await sleep(500);
  const c = await evaluate(s, STATE);
  await screenshot(s, `${OUT}/recv-phone-portrait-received-stopped.png`);
  check(c.state === b.state && c.deliver && c.go === "Start camera" && c.nums === "" && c.why === "", `stopped: "${c.state}", Save kept, "${c.go}"`);
  // ai: started again on the same clip: the finished file named again by the light, Save still there, and the camera left
  // ai: running (it stops itself once a file, not each time the light names one)
  await evaluate(s, `document.getElementById("go").click()`);
  await until(s, `document.getElementById("go").textContent === "Stop camera" && /^Received/.test(document.getElementById("state").textContent)`, 30000, "Received again after a restart");
  const d = await evaluate(s, STATE);
  await sleep(1500);
  const d2 = await evaluate(s, STATE);
  check(d.state === b.state && d.deliver && d.href === b.href && d2.go === "Stop camera", `started again: "${d.state}", the same Save, the camera still on 1.5 s later ("${d2.go}")`);
  await evaluate(s, `document.getElementById("go").click()`);
  await size(s, SIZES[2]);
  await sleep(300);
  await screenshot(s, `${OUT}/recv-laptop-received-stopped.png`);
  check(!s.errors.length, `no console errors over the transfer${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);

  // ai: 4. Home (2026-10-01) in the same profile: the file just received is listed, its Save holds its bytes and Open opens
  // ai: it in a tab of its own; the tips card is shown on this fresh profile, and once dismissed stays so over a reload
  // ai: until Settings shows it again; Delete (asked on the button, then done) empties the received files.
  await s.send("Target.setDiscoverTargets", { discover: true });
  const opened = [];
  for (const ev of ["Target.targetCreated", "Target.targetInfoChanged"]) s.on(ev, ({ targetInfo: i }) => { if (i.type === "page") opened.push(i.url); });
  const HOMEM = `(() => { const $ = (id) => document.getElementById(id), rows = [...document.querySelectorAll("#list li")];
    return { iw: innerWidth, sw: document.documentElement.scrollWidth, rows: rows.map((r) => r.querySelector(".name").textContent), acts: rows[0] ? [...rows[0].querySelectorAll(".acts > *")].map((a) => a.textContent).join(",") : "",
      href: rows[0]?.querySelector("a[download]")?.href ?? "", download: rows[0]?.querySelector("a[download]")?.download ?? "", tips: !$("tips").hidden, empty: !$("empty").hidden, usage: $("usage").textContent, solid: [...document.querySelectorAll(".primary")].map((b) => b.id).join(), bar: [...document.querySelectorAll(".bar .brand, .bar > button")].map((b) => b.textContent).join(",") }; })()`;
  for (const [k, z] of SIZES.entries()) {
    await size(s, z);
    await navigate(s, HOME);
    await until(s, `document.querySelectorAll("#list li").length > 0`, 10000, "the received file listed on Home");
    const h = await evaluate(s, HOMEM);
    await screenshot(s, `${OUT}/home-${z.name}.png`);
    console.log(`     home ${z.name}: ${h.rows.join(" | ")}; ${h.acts}; "${h.usage}"; solid ${h.solid}`);
    check(h.sw <= h.iw, `home ${z.name}: no horizontal scroll (${h.sw} of ${h.iw})`);
    check(h.rows.length === 1 && h.rows[0].startsWith(NAME) && /300\sKB/.test(h.rows[0]) && /^Open,Save,(Share,)?Delete$/.test(h.acts) && h.download === NAME && !h.empty, `home ${z.name}: "${h.rows[0]}" listed with ${h.acts}`);
    check(h.bar === "Lizard" && h.solid === (z.mobile ? "recv" : "send"), `home ${z.name}: the bar "${h.bar}", the solid one ${h.solid}`);
    if (k === 0) {
      const bytesOk = await evaluate(s, `fetch(${JSON.stringify(h.href)}).then((r) => r.arrayBuffer()).then((x) => { const u = new Uint8Array(x); let v = 7, same = u.length === ${LEN}; for (let i = 0; same && i < u.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; same = u[i] === ((v >>> 16) & 255); } return same; })`);
      check(bytesOk, `home: the listed file's Save holds the ${LEN} bytes sent`);
      await s.send("Runtime.evaluate", { expression: `document.querySelector("#list li .acts button").click()`, userGesture: true });
      await sleep(800);
      check(opened.some((u) => u.startsWith("blob:")), `home: Open opens the file in a tab of its own (${opened.join(", ") || "none"})`);
      // ai: back from the tab Open made: the return redraws the list (library.mjs watch) and keeps a row's link while its
      // ai: entry stands, so that tab goes on reading (2026-10-01; before, every drawing let the links go)
      const hidden = await evaluate(s, `document.querySelector("#list li").dataset.mark = "1"; document.visibilityState`);
      await s.send("Page.bringToFront");
      await until(s, `!document.querySelector("#list li")?.dataset.mark`, 5000, "Home redrawn on a return to the page");
      const redrawn = await evaluate(s, HOMEM), readable = await evaluate(s, `fetch(${JSON.stringify(h.href)}).then((r) => r.ok, () => false)`);
      check(hidden === "hidden" && redrawn.href === h.href && readable, `home: back from the opened tab (Home ${hidden} meanwhile), the list redrawn with the same link (${redrawn.href === h.href}), which still reads (${readable})`);
      check(h.tips, `home: the tips card shown on a fresh profile`);
      await evaluate(s, `document.querySelector("#tips [data-got]").click()`);
      await navigate(s, HOME); await sleep(400);
      const gone = await evaluate(s, `document.getElementById("tips").hidden`);
      await evaluate(s, `document.getElementById("tipsAgain").click()`);
      const back = await evaluate(s, `!document.getElementById("tips").hidden`);
      check(gone && back, `home: the tips dismissed stay so over a reload (${gone}), and Settings shows them again (${back})`);
    }
  }
  await evaluate(s, `(() => { const b = [...document.querySelectorAll("#list li .acts button")].find((x) => x.textContent === "Delete"); b.click(); return b.textContent; })()`);
  await sleep(200);
  const asked = await evaluate(s, `[...document.querySelectorAll("#list li .acts button")].map((x) => x.textContent).join(",")`);
  await evaluate(s, `[...document.querySelectorAll("#list li .acts button")].find((x) => x.textContent === "Delete it?").click()`);
  await until(s, `document.querySelectorAll("#list li").length === 0 && !document.getElementById("empty").hidden`, 5000, "the list empty after Delete");
  const left = await evaluate(s, opfsFiles("lizard-files"));
  check(/Delete it\?/.test(asked) && left === 0, `home: Delete asks on itself ("${asked}") and then empties the received files (${left} left)`);
  check(!s.errors.length, `no console errors on Home${s.errors.length ? `: ${s.errors.join("; ")}` : ""}`);
  s.close();
}

console.log(failed ? `${failed} FAILED` : "all passed");
process.exit(failed ? 1 : 0);
