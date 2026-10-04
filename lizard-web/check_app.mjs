// ai: The installed app end to end (2026-10-01, a content-hashed PWA; pwa/): what
// ai: only the built app does, in headless Chrome on a fresh profile. pwa/build.mjs builds it into research/build/check-app
// ai: (lizard-web/app is left alone), served by a static server here that logs every request. Since 2026-10-03 it is
// ai: served as Cloudflare Pages serves a deploy: the folder at
// ai: the root of a public host name (lizard.pages.test, mapped to this machine and taken as a secure origin), x.html
// ai: answered by a 308 to x and index.html by one to its folder, x served from x.html, _headers read for the headers
// ai: and not served, anything but GET refused:
// ai:   1. installed: the worker controls the page, Chrome's own verdict finds it installable (Page.getInstallabilityErrors,
// ai:      the manifest's errors), and the caches hold the shell exactly (the worker's lists, no more, no fewer);
// ai:   2. offline, the server gone (closed, not emulated): Home loads, the receiver decodes the test stream off the fake
// ai:      camera (a LIZARD-96 clip, as check_ui.mjs makes them), the sender paints it at its clean URL (send?...);
// ai:   3. a rebuild of the same sources (a new cache name, the same files): the next visit installs it fetching the
// ai:      worker, the pages and the manifest alone, nothing under assets/ (the differential install).
// ai: Over all three, no request but GET (a public host gets no development log, devlog.mjs) and no console error but
// ai: the refused connections of the server's absence. node lizard-web/check_app.mjs, from anywhere (about 40 s). Exit 1
// ai: on a failure.
import { spawn, spawnSync } from "node:child_process";
import http from "node:http";
import { readFileSync, statSync, mkdtempSync, rmSync, openSync, writeSync, closeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, extname, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { tabs, open, evaluate, navigate } from "./phone.mjs";
import { makePhy } from "../liblizard/sim/phy.mjs";
import { N_FOR } from "../liblizard/sim/lizard_pick.mjs";

const OUT = fileURLToPath(new URL("../research/build/check-app", import.meta.url)), ROOT = OUT, PORT = 8900 + (process.pid % 90), DEV = 9480 + (process.pid % 90);
const HOST = "lizard.pages.test", BASE = `http://${HOST}:${PORT}/`;
const scratch = mkdtempSync(join(tmpdir(), "lizard-app-"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0, browser = null;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
process.on("exit", () => { try { process.kill(-browser.pid, "SIGKILL"); } catch {} rmSync(scratch, { recursive: true, force: true }); });

function build() {
  const r = spawnSync(process.execPath, [fileURLToPath(new URL("./pwa/build.mjs", import.meta.url)), OUT], { encoding: "utf8" });
  if (r.status !== 0) { console.error(r.stdout, r.stderr); throw new Error("the build failed"); }
  console.log(`     ${r.stdout.trim()}`);
}
// ai: the shell as the built worker lists it: its two arrays' entries (the hashes object's lines carry a colon)
const shellOf = () => readFileSync(join(OUT, "service-worker.js"), "utf8").split("\n").filter((l) => /^  "\.\/[^"]*",$/.test(l)).map((l) => l.trim().slice(1, -2));

// ai: A static server over the built app, as Cloudflare Pages serves it (the header), that logs every request (every
// ai: across the three parts, log since the last serve()); stop() closes it and every connection, so the next request is
// ai: refused as by a host that is gone.
const MIME = { ".html": "text/html; charset=utf-8", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".svg": "image/svg+xml", ".css": "text/css", ".webmanifest": "application/manifest+json" };
let log = [], every = [], server = null;
const isFile = (f) => { try { return statSync(f).isFile(); } catch { return false; } };
function serve() {
  server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://x"), path = decodeURIComponent(url.pathname);
    log.push(`${req.method} ${path}`); every.push(`${req.method} ${path}`);
    if (req.method !== "GET") { res.writeHead(405).end(); return; }
    const to = path.endsWith("/index.html") ? path.slice(0, -10) : path.endsWith(".html") ? path.slice(0, -5) : null;
    if (to !== null) { res.writeHead(308, { location: `${to}${url.search}` }).end(); return; }
    let file = normalize(join(ROOT, path.endsWith("/") ? `${path}index.html` : path));
    if (!isFile(file) && isFile(`${file}.html`)) file = `${file}.html`;
    if (!file.startsWith(ROOT) || !isFile(file) || path === "/_headers") { res.writeHead(404).end(); return; }
    const cache = path.startsWith("/assets/") ? "public, max-age=31536000, immutable" : "public, max-age=0, must-revalidate";
    res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream", "cache-control": cache }).end(readFileSync(file));
  });
  return new Promise((ok) => server.listen(PORT, "127.0.0.1", ok));
}
const stop = () => new Promise((ok) => { server.closeAllConnections?.(); server.close(() => ok()); });

// ai: LIZARD-96's test stream at 24 painted, 1920 x 1080 at 30, as check_ui.mjs clips it
async function clip(path) {
  const CW = 1920, CH = 1080, FD = 24, FC = 30, SUBCH = 96;
  const phy = await makePhy({ phy: "focus", stream: "shake256", n: N_FOR(SUBCH), subch: SUBCH, mode: 1, fps: FD, variants: [{ name: "rx" }] });
  const fd = openSync(path, "w"), y = Buffer.alloc(CW * CH), chroma = Buffer.alloc((CW * CH) / 2, 128);
  writeSync(fd, `YUV4MPEG2 W${CW} H${CH} F${FC}:1 Ip A1:1 C420jpeg\n`);
  let shown = -1;
  for (let i = 0; i < 75; i++) {
    const seq = Math.floor((i * FD) / FC);
    if (seq !== shown) {
      const fr = await phy.frame(seq), ox = (CW - fr.w) >> 1, oy = (CH - fr.h) >> 1;
      y.fill(235);
      for (let r = 0; r < fr.h; r++) for (let c = 0; c < fr.w; c++) y[(oy + r) * CW + ox + c] = Math.round(20 + 215 * fr.drive[r * fr.w + c]);
      shown = seq;
    }
    writeSync(fd, "FRAME\n"); writeSync(fd, y); writeSync(fd, chroma);
  }
  closeSync(fd);
}
async function until(s, expr, ms, what) {
  for (const t0 = Date.now(); Date.now() - t0 < ms; await sleep(200)) if (await evaluate(s, expr).catch(() => false)) return true;
  const seen = await evaluate(s, `[document.getElementById("state")?.textContent, document.getElementById("nums")?.textContent].filter(Boolean).join(" | ")`).catch(() => "");
  throw new Error(`waited ${ms / 1000} s for ${what}; the page: ${seen}`);
}
const CACHED = `(async () => { const out = {}; for (const k of await caches.keys()) out[k] = (await (await caches.open(k)).keys()).map((r) => new URL(r.url).pathname); return out; })()`;

build();
await serve();
const testClip = join(scratch, "test.y4m");
await clip(testClip);
browser = spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${testClip}`, `--host-resolver-rules=MAP ${HOST} 127.0.0.1`, `--unsafely-treat-insecure-origin-as-secure=http://${HOST}:${PORT}`, `--remote-debugging-port=${DEV}`, `--user-data-dir=${join(scratch, "profile")}`, "about:blank"], { stdio: "ignore", detached: true });
let s = null;
for (let t = 0; t < 50 && !s; t++) { try { const l = await tabs(DEV); if (l.length) s = await open(l[0], DEV); } catch {} await sleep(200); }
if (!s) throw new Error("no DevTools");
const errors = [];
s.on("Runtime.consoleAPICalled", (p) => { if (p.type === "error") errors.push(`console: ${p.args.map((a) => a.value ?? a.description).join(" ")}`); });
s.on("Runtime.exceptionThrown", (p) => errors.push(`exception: ${p.exceptionDetails.exception?.description ?? p.exceptionDetails.text}`));
s.on("Log.entryAdded", ({ entry: e }) => { if (e.level === "error") errors.push(`log: ${e.text} ${e.url ?? ""}`); });
await s.send("Runtime.enable"); await s.send("Log.enable"); await s.send("Page.enable");
await s.send("Emulation.setDeviceMetricsOverride", { width: 412, height: 915, deviceScaleFactor: 2.625, mobile: true });

// ai: 1. installed
await navigate(s, BASE);
await until(s, `navigator.serviceWorker.ready.then(() => true)`, 30000, "the worker ready");
await navigate(s, BASE);
const controlled = await evaluate(s, `!!navigator.serviceWorker.controller`);
const shell = shellOf();
const cached = await evaluate(s, CACHED);
const held = Object.values(cached).flat().map((p) => `.${p}`);
const missing = shell.filter((p) => !held.includes(p)), extra = held.filter((p) => !shell.includes(p));
check(controlled, `the worker controls the page`);
check(missing.length === 0 && extra.length === 0, `the caches hold the shell exactly: ${held.length} of ${shell.length} (${Object.keys(cached).join(", ")})${missing.length ? `; missing ${missing.slice(0, 3)}` : ""}${extra.length ? `; extra ${extra.slice(0, 3)}` : ""}`);
const inst = await s.send("Page.getInstallabilityErrors").catch((e) => ({ installabilityErrors: [{ errorId: String(e) }] }));
const man = await s.send("Page.getAppManifest").catch(() => ({ errors: [{ message: "no manifest" }] }));
check(!inst.installabilityErrors?.length && !man.errors?.length, `installable by Chrome's own verdict${inst.installabilityErrors?.length ? `: ${inst.installabilityErrors.map((e) => e.errorId).join(", ")}` : ""}${man.errors?.length ? `; the manifest: ${man.errors.map((e) => e.message).join(", ")}` : ""}`);
// ai: a URL with a query is answered under its own URL, query and all: a worker's location is its response's URL, and
// ai: the GPU worker reads its switches from location.search (2026-10-01: answered with the stored
// ai: response, the worker read none). fetch's url is that same response URL.
const asked = await evaluate(s, `fetch("index.html?probe=1").then((r) => r.url)`);
check(asked.endsWith("/index.html?probe=1"), `a query kept through the cache (fetch's url ${asked.replace(BASE, "./")})`);

// ai: 2. offline: the server closed
await stop();
await navigate(s, BASE);
check(await evaluate(s, `!!document.getElementById("send") && !!document.getElementById("recv")`), `offline: Home loads`);
// ai: a page by its own name, no query: the stored response itself answers, and the host redirected it on install
await navigate(s, `${BASE}send.html`).catch(() => {});
check(await evaluate(s, `!!document.getElementById("go")`).catch(() => false), `offline: send.html loads by its own name (stored without the redirect that brought it)`);
await navigate(s, `${BASE}recv.html?auto`);
const read = await until(s, `document.getElementById("state").textContent === "Reading the test stream" && /LIZARD-96/.test(document.getElementById("lab").textContent)`, 40000, "offline: the test stream read").catch((e) => (console.log(`     ${e.message}`), false));
check(read, `offline: the receiver reads the test stream off the camera (the codec, the workers, every file from the cache)`);
await navigate(s, `${BASE}send?subch=auto&fps=24&payload=test&auto`);
const painted = await until(s, `/painted [\\d.]+\\/s/.test(document.getElementById("tx").textContent) && document.getElementById("state").textContent === "Sending the test stream"`, 30000, "offline: the test stream painted").catch((e) => (console.log(`     ${e.message}`), false));
check(painted, `offline: the sender paints the test stream, asked for at its clean URL (send?...)`);

// ai: 3. the differential install
await sleep(1100);   // ai: a new second, so the new build's cache name differs
const before = Object.keys(await evaluate(s, CACHED));
build();
log = [];
await serve();
await navigate(s, BASE);
const newName = (await evaluate(s, `fetch("service-worker.js", { cache: "no-store" }).then((r) => r.text()).then((t) => (t.match(/v(\\d+)\`;/) || [])[1])`)) ?? "";
const updated = await until(s, `caches.keys().then((k) => k.includes("lizard-v${newName}"))`, 30000, "the new build installed").catch(() => false);
await sleep(500);
const fetched = log.filter((l) => l.startsWith("GET ")).map((l) => `.${l.slice(4)}`);
const assets = fetched.filter((p) => p.startsWith("./assets/"));
check(updated && assets.length === 0, `a rebuild installs (${before.join(", ")} to lizard-v${newName}) fetching ${fetched.length} files, none under assets/ (${[...new Set(fetched)].join(", ")})`);

const notGet = every.filter((l) => !l.startsWith("GET "));
check(!notGet.length, `no request but GET (no development log on a public host)${notGet.length ? `: ${[...new Set(notGet)].join(", ")}` : ""}`);
const real = errors.filter((e) => !/ERR_CONNECTION_REFUSED/.test(e));
check(!real.length, `no console errors but the absent server's refused connections${real.length ? `: ${real.join("; ")}` : ""}`);
await stop();
console.log(failed ? `${failed} FAILED` : "all passed");
process.exit(failed ? 1 : 0);
