// ai: The receiver in two browsers' engines, on each one's fake camera: the page decodes on the CPU pool and its stats
// ai: row says how the frames reached it.
// ai:   chrome: headless with the GPU off; the pool's VideoFrames come off the track (MediaStreamTrackProcessor)
// ai:   firefox: headless; the pool's VideoFrames come from new VideoFrame(video) (no track processor); skipped when
// ai:     firefox is not installed
// ai: node lizard-web/check_browsers.mjs (BROWSERS=chrome,firefox); about 10 s a browser; exit 1 unless each decoded.
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = 8500 + (process.pid % 300), root = mkdtempSync(join(tmpdir(), "lizard-browsers-"));
const server = spawn("node", [fileURLToPath(new URL("./server.mjs", import.meta.url))], { env: { ...process.env, RIG_HTTP: PORT, RIG_HTTPS: PORT + 363 }, stdio: "ignore" });
process.on("exit", () => { server.kill(); rmSync(root, { recursive: true, force: true }); });
const api = (p, body) => fetch(`http://localhost:${PORT}${p}`, body ? { method: "POST", body: JSON.stringify(body) } : undefined).then((r) => r.json());
for (let i = 0; i < 40; i++) { try { await api("/api/info"); break; } catch { await new Promise((r) => setTimeout(r, 250)); } }
const url = `http://localhost:${PORT}/lizard-web/recv.html?auto&seen`;

function open(browser, prof) {
  if (browser === "chrome") return spawn("google-chrome", ["--headless=new", "--no-sandbox", "--disable-gpu", "--enable-unsafe-swiftshader", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--user-data-dir=${prof}`, url], { stdio: "ignore", detached: true });
  const prefs = { "media.navigator.streams.fake": true, "media.navigator.permission.disabled": true, "browser.shell.checkDefaultBrowser": false };
  writeFileSync(join(prof, "user.js"), Object.entries(prefs).map(([k, v]) => `user_pref("${k}", ${v});\n`).join(""));
  return spawn("firefox", ["--headless", "--no-remote", "--profile", prof, url], { stdio: "ignore", detached: true });
}

for (const browser of (process.env.BROWSERS ?? "chrome,firefox").split(",")) {
  if (browser === "firefox" && spawnSync("firefox", ["--version"]).status !== 0) { console.log("firefox: not installed, skipped"); continue; }
  const prof = mkdtempSync(join(root, `${browser}-`)), b = open(browser, prof);
  // ai: Polled until the pool is decoding frames. A row is this browser's by its ua (the last browser's may still be the
  // ai: server's last row).
  const mine = (r) => (r?.ua ?? "").startsWith(browser === "chrome" ? "Chrome" : "Firefox");
  const decoding = (r) => r?.decoder === "cpu" && r.workersReady > 0 && r.processedFps > 0;
  let row = null;
  for (let t = 0; t < 60 && !decoding(row); t++) {
    await new Promise((r) => setTimeout(r, 1000));
    const got = (await api("/api/stats")).stats ?? null;
    row = mine(got) ? got : null;
  }
  try { process.kill(-b.pid, "SIGKILL"); } catch {}
  const ok = decoding(row);
  // ai: How the pool got its frames: as VideoFrames (from where, the format, copied to luma in the worker) or by the grab.
  const frames = !row ? "" : row.vfSent ? `; ${row.vfSent} frames in the window as VideoFrames from the ${row.vfSource} (${row.vfFormat || "format not reported"}), ${row.lumaSent} by the canvas grab` : `; frames by the canvas grab (${row.vfNote || "no reason given"})`;
  console.log(`${browser}: ${row ? `${row.ua}, decoder ${row.decoder}, ${row.workersReady} of ${row.workers} workers ready, ${row.processedFps.toFixed(1)} frames/s decoded${frames}` : "no stats row"}  ${ok ? "ok" : "FAILED"}`);
  if (!ok) process.exitCode = 1;
}
process.exit(process.exitCode ?? 0);
