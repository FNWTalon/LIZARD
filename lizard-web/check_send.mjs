// ai: Does the sender page stand alone, its development logs aside (2026-09-26)?
// ai: A plain static file server (no /api: every such request is answered 404 and logged) serves lizard/ to headless
// ai: Chrome, driven over the DevTools protocol (phone.mjs on a local port), its window 1920 x 1080. Six cases: the
// ai: test stream at LIZARD-96 and at LIZARD-16, LIZARD-96 carrying a 50,000-byte file chosen through the file input
// ai: (DOM.setFileInputFiles), and the test stream at the page's own pick (?payload=test&auto), which must be the
// ai: library's pick for the room the page measured (sim/lizard_pick.mjs pickVersion over the whole ladder since 2026-09-29,
// ai: capped at LIZARD-560 before; on the rings a 1080 px room picks LIZARD-480 either way), then that pick with two codes (&codes=2): the canvas 2:1, each half read on its own,
// ai: the right half's word the same and none of its ids the left's (no browser decodes two codes; the C here reads
// ai: each half as a native receiver would), and the pick again, changed while it runs (below). Each time the page's canvas is
// ai: read back as luma, padded with white as the page's surround is, and decoded by the C told nothing (sim/phy.mjs
// ai: makeBlind): the word must name the version painted and the page's 24 fps, the blocks must verify, the state line
// ai: must say what is sent, and for the file one of the frames read must carry its header block, whose name, length and
// ai: BLAKE3 root (sim/xfer.mjs XferReceiver's parse) are the file's. Then the server's log: every request under /api a
// ai: POST (a development log), none to /api/config. Exit 1 on any failure. About 10 s; 21 s with ENC=gpu (SwiftShader).
// ai: The encoder (2026-09-29, gpu/encoder.mjs): plain, Chrome has no WebGPU adapter and the page's auto must fall back
// ai: to the wasm path and say why (#tx "encoder: wasm in the worker (WebGPU off: ...)"); ENC=gpu gives Chrome
// ai: SwiftShader's WebGPU and asks for the GPU encoder by name (&enc=gpu), and every case is then read off the GPU's
// ai: canvas (#cg) and #tx must name WebGPU. Two things about that canvas in headless Chrome (2026-09-29): with ANGLE on
// ai: SwiftShader's GL the first present loses the device ("A valid external Instance reference no longer exists", a
// ai: bare probe page the same), so the arm runs ANGLE on SwiftShader's Vulkan; and a WebGPU canvas keeps nothing after
// ai: it is presented (drawn later it reads transparent), so it is drawn into a 2D canvas inside an animation frame,
// ai: after the page's own tick in the same frame, on a frame in which the page presented (else the next one).
// ai: The sixth case (2026-09-29), in both arms, takes the two changes a running stream meets that the five above never
// ai: make: a re-pick and a new display rate. The pick at 1920 x 1080 is read once as the auto case reads it, then the
// ai: viewport goes to 1280 x 720 (Emulation.setDeviceMetricsOverride): <main>'s ResizeObserver, send.mjs roomChanged's
// ai: 300 ms debounce, repick, which is the worker's respec and, under the GPU encoder, a new configuration made and
// ai: checked against the C while the frames already asked for present from the old one's slots, the old one released
// ai: after its last. The canvas is grabbed from the resize on: every grab until the new version shows must read the old
// ai: version or the new at 24 fps with 0 bad (the frames in flight: AHEAD, 3, on wasm, up to two batches of 16 under
// ai: the GPU encoder; a GPU canvas with nothing presented for 120 animation frames, while the new configuration is made,
// ai: is counted, not judged), and the first frame at the new version, the library's pick for the room the page then
// ai: measures (a 720 px room: LIZARD-208, n = 768), must read it at 24 fps, its blocks verified, 0 bad, off the arm's
// ai: canvas with the state line still sending, and its ids all above every id read before (send.mjs nextId: a re-pick
// ai: changes the blocks a frame, and ids counted from seq would come back as repeats). The page's report after it (#tx
// ai: and #nums, written together once a second, so a grab can hold the one from before the change) must name it in #nums
// ai: within 3 s, #tx's encoder line still the arm's: in the GPU arm WebGPU, so the check at the new picture passed and
// ai: nothing stopped. Then the display fps box is set to 30 as a script sets it (no change event, so the page keeps
// ai: nothing): the tick restates it, to the worker's word and under the GPU encoder through GpuEncoder.setFps, which
// ai: repaints the border's word. Grabbed from the change on, every frame must read the new version at 24 until the first
// ai: that says 30, which is judged as the re-pick's was, and the report after it must say "of 30 asked", the same encoder.
// ai: Each wait is bounded (60 s for the re-pick, SwiftShader's compile and check at the new picture inside it; 30 s for
// ai: the rate), and how long each change took to reach the canvas is printed. The C's blind decoder misses a few clean
// ai: frames (2026-09-29, the C's own paint of the test stream, whole pixels in white: 8 of 2,000 at LIZARD-208 and 3 of
// ai: 1,000 at LIZARD-480, each registered in the 32 ring, no word, no block, where a decoder told the version reads every
// ai: block), which the case's twenty-odd grabs would meet: a grab on the way with no word is decoded told at the old and
// ai: new versions, and read so it is printed as missed, not failed; a frame no decoder reads still fails. The first five
// ai: cases judge one grab each blind, a missed one read told and grabbed again (below).
// ai: The gap between two codes (2026-10-03, send.html's Gap slider): two codes again at gap 0 and at 40 modules
// ai: (&gap=), each canvas's gap the modules asked within half a module and both halves read blind; then the gap moved
// ai: to 4 while it runs (the slider's change event), grabbed until the canvas's gap is 4 (5 s at most), both halves
// ai: every block, 0 bad.
// ai:   node lizard-web/check_send.mjs            (from anywhere; APP=1 the built app, lizard-web/app)
// ai:   ENC=gpu node lizard-web/check_send.mjs
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFileSync, existsSync, statSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { makePhy, blindSpec } from "../liblizard/sim/phy.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { init as initWh, Decoder } from "../liblizard/sim/fountain.mjs";
import { XferReceiver, MemoryStore, ID_HEADER, blake3, hex } from "../liblizard/sim/xfer.mjs";
import { N_FOR, NAME, SPAN, RING_DEFAULT, FOCUS_BITMAP, MODULES, OB_QUIET, pickVersion } from "../liblizard/sim/lizard_pick.mjs";
import { tabs, open, evaluate, navigate } from "./phone.mjs";

const ROOT = resolve(fileURLToPath(new URL("./..", import.meta.url))), PAD = 32, FPS = 24, FILE_BYTES = 50000, GRABS = 60, GPU = process.env.ENC === "gpu";
const MIME = { ".html": "text/html; charset=utf-8", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".svg": "image/svg+xml", ".css": "text/css" };

// ai: The static server: files under lizard/, nothing else; every request logged as "METHOD path status".
const log = [];
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = normalize(join(ROOT, p)), status = 200;
  if (!file.startsWith(ROOT) || p.startsWith("/api/")) status = 404;
  else {
    if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
    if (!existsSync(file)) status = 404;
  }
  log.push(`${req.method} ${p} ${status}`);
  if (status !== 200) { res.writeHead(status); return res.end(); }
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
  res.end(readFileSync(file));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const HTTP = server.address().port, DEVTOOLS = 9300 + (process.pid % 300);

const scratch = mkdtempSync(join(tmpdir(), "lizard-send-")), filePath = join(scratch, "check-send.bin");
const fileBytes = new Uint8Array(FILE_BYTES);
for (let i = 0, v = 7; i < fileBytes.length; i++) { v = (Math.imul(v, 1103515245) + 12345) >>> 0; fileBytes[i] = v >>> 16; }
writeFileSync(filePath, fileBytes);
const flags = GPU ? ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader", "--use-angle=vulkan", "--use-vulkan=swiftshader", "--enable-features=Vulkan"] : ["--disable-gpu", "--enable-unsafe-swiftshader"];
const chrome = spawn("google-chrome", ["--headless=new", "--no-sandbox", ...flags, `--remote-debugging-port=${DEVTOOLS}`, `--user-data-dir=${join(scratch, "profile")}`, "--window-size=1920,1080", "about:blank"], { stdio: "ignore", detached: true });
let cleaned = false;
const cleanup = () => { if (cleaned) return; cleaned = true; try { process.kill(-chrome.pid, "SIGKILL"); } catch {} server.close(); rmSync(scratch, { recursive: true, force: true }); };
// ai: A DevTools call that throws ends the run with its error; the detached Chrome and the scratch folder go with it.
process.on("exit", cleanup);

let s = null;
for (let t = 0; t < 50 && !s; t++) { try { const list = await tabs(DEVTOOLS); if (list.length) s = await open(list[0], DEVTOOLS); } catch { await new Promise((r) => setTimeout(r, 200)); } }
if (!s) { cleanup(); throw new Error("no DevTools session on the local Chrome"); }

const M = await initOb();
await initWh();
const phy = await makePhy(blindSpec());

// ai: The page's canvas as luma (the rig's formula, src/acquire.c), with the #tx line that says what was painted. The
// ai: canvas shown: #c (2D) or the GPU encoder's #cg, read through a 2D canvas it is drawn into.
const GRAB = `(async () => { const s = [document.getElementById("cg"), document.getElementById("c")].find((e) => !e.hidden);
  let c = s;
  if (s.id === "cg") for (let k = 0; ; k++) {
    if (k > 120) throw new Error("the GPU canvas was never read in a frame it was presented in");
    c = await new Promise((ok) => requestAnimationFrame(() => { const t = Object.assign(document.createElement("canvas"), { width: s.width, height: s.height }), x = t.getContext("2d"); x.drawImage(s, 0, 0); ok(x.getImageData(0, 0, 1, 1).data[3] ? t : null); }));
    if (c) break;
  }
  const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data, n = c.width * c.height, y = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) y[i] = (77 * d[j] + 150 * d[j + 1] + 29 * d[j + 2] + 128) >> 8;
  let b = ""; for (let i = 0; i < n; i += 8192) b += String.fromCharCode.apply(null, y.subarray(i, i + 8192));
  return { w: c.width, h: c.height, which: s.id, b64: btoa(b), gap: +document.getElementById("tx").dataset.gap || 0, tx: document.getElementById("tx").textContent, state: document.getElementById("state").textContent, nums: document.getElementById("nums").textContent }; })()`;
// ai: One grab decoded blind, the canvas set in PAD px of white as the page's white surround sets it. With two codes
// ai: the canvas is two squares side by side: each half is set in white and decoded on its own (halves), as a receiver
// ai: reading both would, and r is the left's.
// ai: Two codes sit a gap apart (the page's #tx dataset.gap, canvas px; send.mjs layout): each half is a symbol's
// ai: width, k (side + gap) in.
async function readOnce(codes = 1) {
  const g = await evaluate(s, GRAB), px = Buffer.from(g.b64, "base64"), gap = codes > 1 ? g.gap : 0, side = (g.w - (codes - 1) * gap) / codes, W = side + 2 * PAD, H = g.h + 2 * PAD, halves = [], imgs = [];
  for (let k = 0; k < codes; k++) {
    const img = new Uint8Array(W * H).fill(255);
    for (let y = 0; y < g.h; y++) img.set(px.subarray(y * g.w + k * (side + gap), y * g.w + k * (side + gap) + side), (y + PAD) * W + PAD);
    halves.push(phy.decode(img, W, H, null)[0]); imgs.push(img);
  }
  return { g, r: halves[0], halves, img: imgs[0], imgs, W, H };
}
// ai: A grab the blind decoder read no word on, decoded told (sim/phy.mjs makeFocus, the test stream's generator) at
// ai: each sub-channel count in subs, in the default ring: the first whose blocks verify, { sub, r }, or null. One told
// ai: codec at a time (the wasm holds one; the blind receiver's are its own), freed after.
async function readTold(x, subs) {
  for (const sub of subs) {
    const n = N_FOR(sub), p = await makePhy({ phy: "focus", n, subch: sub, mode: 1, span: SPAN(n, RING_DEFAULT), bitmap: FOCUS_BITMAP, stream: "shake256" });
    try { const r = p.decode(x.img, x.W, x.H, null)[0]; if (r.seen > 0) return { sub, r }; } finally { p.free(); }
  }
  return null;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const waitPainted = async () => { for (let t = 0; t < 100; t++) { if (/painted [\d.]+\/s/.test(await evaluate(s, `document.getElementById("tx").textContent`))) return true; await sleep(200); } return false; };
// ai: The window and the room the page measures, as send.mjs room() measures it (codes symbols across <main>).
// ai: (with two codes the gap between them, the page's #tx dataset.gapfrac of a symbol's width, is taken off first)
const measure = (codes = 1) => evaluate(s, `(() => { const b = document.querySelector("main"), gf = ${codes} > 1 ? +document.getElementById("tx").dataset.gapfrac || 0 : 0; return { w: innerWidth, h: innerHeight, room: Math.max(64, Math.min(b.clientWidth / (${codes} + gf), b.clientHeight) * devicePixelRatio) }; })()`);
// ai: #tx's encoder line, and whether it and the canvas read are the arm's: WebGPU off #cg, or wasm off #c saying why
// ai: WebGPU is off.
function encoderOf(g) {
  const line = g.tx.split("\n").find((l) => l.startsWith("encoder: ")) ?? "no encoder line";
  return { line, ok: GPU ? g.which === "cg" && line.startsWith("encoder: WebGPU") : g.which === "c" && /^encoder: wasm in the worker \(WebGPU off: .+\)$/.test(line) };
}
const decoded = (g, r) => `canvas ${g.w} x ${g.h}, picture n = ${r.n}, word ${r.fmt ? `version ${r.fmt.version} at ${r.fmt.fps} fps` : "not read"}, ${r.seen} blocks verified, ${r.bad} bad`;
const pageLine = (g) => `  page: ${g.state} | ${g.nums} | ${g.tx.split("\n").slice(0, 3).join(" | ")} | ${g.tx.split("\n").find((l) => l.startsWith("painted ")) ?? "no painted line"} | ${encoderOf(g).line} (#${g.which})`;

const faults = [];
// ai: 488 (61 blocks, 2026-10-01): a whole number of blocks the 16-step ladder never offered, set as the slider sets it
for (const c of [{ subch: 96 }, { subch: 16 }, { subch: 488 }, { subch: 96, file: true }, { subch: "auto" }, { subch: "auto", codes: 2 }, { subch: "auto", codes: 2, gap: 0 }, { subch: "auto", codes: 2, gap: 40, regap: 4 }, { subch: "auto", midStream: true }]) {
  const codes = c.codes ?? 1;
  let name = `LIZARD-${c.subch}${c.file ? `, a ${FILE_BYTES} B file` : ", test stream"}${c.gap !== undefined ? `, gap ${c.gap}` : ""}`;
  // ai: The window's frame takes rows even headless (1920 x 937 of 1920 x 1080), so the auto case sets the page's viewport.
  if (c.subch === "auto") await s.send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await navigate(s, `http://127.0.0.1:${HTTP}/lizard-web/${process.env.APP ? "app/" : ""}send.html?${c.subch === "auto" ? `subch=auto&fps=${FPS}&payload=test&auto` : `subch=${c.subch}&fps=${FPS}${c.file ? "&payload=file" : "&payload=test&auto"}`}${codes > 1 ? `&codes=${codes}` : ""}${c.gap !== undefined ? `&gap=${c.gap}` : ""}${GPU ? "&enc=gpu" : ""}`);
  if (c.file) {
    // ai: The file picked as a user picks one: into the file input, then the start button.
    const doc = await s.send("DOM.getDocument"), { nodeId } = await s.send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#file" });
    await s.send("DOM.setFileInputFiles", { nodeId, files: [filePath] });
    await evaluate(s, `document.getElementById("go").click()`);
    // ai: the first Start on a fresh profile asks about the display's brightness first (send.mjs go): it must, and its
    // ai: Start sends
    if (!(await evaluate(s, `document.getElementById("first").open`))) faults.push(`${name}: no brightness tip before the first send`);
    await evaluate(s, `document.getElementById("firstStart").click()`);
  }
  if (!(await waitPainted())) { faults.push(`${name}: the page never said it painted: ${await evaluate(s, `document.getElementById("tx").textContent`)}`); continue; }
  // ai: The auto case's version: the library's pick for the room the page measured, as send.mjs room() measures it.
  let subch = c.subch;
  if (subch === "auto") {
    const m = await measure(codes);
    subch = pickVersion(m.room).subch;
    name = `auto in a ${m.w} x ${m.h} window (a ${m.room} px room${codes > 1 ? ` a code, ${codes} codes side by side` : ""}): LIZARD-${subch}, test stream${c.midStream ? ", before a re-pick and a new rate" : ""}`;
    if (m.w !== 1920 || m.h !== 1080) faults.push(`${name}: the window is not 1920 x 1080`);
  }
  // ai: For the file, frames until one carries the header (a control block every few frames at this size); else one.
  // ai: A grab a half of which the C's blind decoder reads no word on (its miss on a few clean frames, above) is decoded
  // ai: told at the version painted: read so with 0 bad it is counted as missed and another grab taken, at most four
  // ai: (2026-09-29: one blind grab a case failed a good run about 2% of the time). Read by neither, it is
  // ai: judged as it is and fails.
  let out = null, header = null, frames = 0, blocks = 0, missed = 0;
  for (let k = 0; k < (c.file ? GRABS : 1) + missed; k++) {
    out = await readOnce(codes); frames++;
    const miss = out.halves.findIndex((h) => !h.fmt);
    if (miss >= 0 && missed < 4) {
      const t = await readTold({ img: out.imgs[miss], W: out.W, H: out.H }, [subch]);
      if (t && t.r.bad === 0) { missed++; console.log(`  a grab missed blind${codes > 1 ? ` (half ${miss})` : ""}, read told: ${t.r.seen} blocks, 0 bad; grabbed again`); continue; }
    }
    blocks += out.r.seen;
    const h = out.r.got.find((b) => b.id === ID_HEADER);
    if (h) { const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() }); await rx.add(ID_HEADER, h.bytes); header = rx.progress().header; break; }
  }
  const { g, r } = out, v = subch / 8, wordOk = r.fmt?.version === v && r.fmt?.fps === FPS, blocksOk = r.seen > 0 && r.bad === 0;
  // ai: a file's line carries its size, and what it was compressed to where zstd shrank it (2026-10-05): random bytes do not shrink
  const stateOk = g.state === (c.file ? "Sending check-send.bin, 50 KB" : "Sending the test stream");
  const root = hex(blake3(M, fileBytes)), headerOk = !c.file || (header && header.name === "check-send.bin" && header.length === FILE_BYTES && header.root === root);
  console.log(`${name}: canvas ${g.w} x ${g.h}, picture n = ${r.n} (want ${N_FOR(subch)}), word ${r.fmt ? `version ${r.fmt.version} at ${r.fmt.fps} fps` : "not read"}, ${r.seen} blocks verified, ${r.bad} bad${c.file ? `; ${frames} frames read, ${blocks} blocks, header ${header ? `${header.name}, ${header.length} B, root ${header.root.slice(0, 16)}...` : "not read"}` : ""}`);
  const enc = encoderOf(g);
  console.log(pageLine(g));
  if (!enc.ok) faults.push(`${name}: the encoder is not the one asked for: ${enc.line}, read off #${g.which}`);
  if (!wordOk) faults.push(`${name}: the word is not version ${v} at ${FPS} fps`);
  if (!blocksOk) faults.push(`${name}: ${r.seen} blocks, ${r.bad} bad`);
  if (r.n !== N_FOR(subch)) faults.push(`${name}: registered at n = ${r.n}`);
  if (!stateOk) faults.push(`${name}: the state line says "${g.state}"`);
  if (!headerOk) faults.push(`${name}: the header read is not the file's (want ${FILE_BYTES} B, root ${root.slice(0, 16)}...)`);
  // ai: Two codes: the right half read on its own as well, its word the same, its blocks verified and none of its ids
  // ai: the left's (the page gives the second symbol the frame's next blocksPerFrame ids).
  if (codes > 1) {
    const right = out.halves[1], ids = (x) => new Set(x.got.map((b) => b.id)), left = ids(r), shared = [...ids(right)].filter((id) => left.has(id));
    console.log(`  right half: word ${right.fmt ? `version ${right.fmt.version} at ${right.fmt.fps} fps` : "not read"}, ${right.seen} blocks verified, ${right.bad} bad, ${shared.length} ids shared with the left`);
    if (!(right.fmt?.version === v && right.fmt?.fps === FPS)) faults.push(`${name}: the right half's word is not version ${v} at ${FPS} fps`);
    if (!(right.seen > 0 && right.bad === 0)) faults.push(`${name}: the right half read ${right.seen} blocks, ${right.bad} bad`);
    if (shared.length) faults.push(`${name}: the halves share ${shared.length} ids`);
    // ai: two squares a gap apart (send.mjs gapOf modules, 12 unless the slider moved, the page's #tx dataset.gap in
    // ai: canvas px): 2:1 plus the gap, and the gap the modules asked (the 128 ring's symbol, margin included, the canvas's height)
    if (g.w !== 2 * g.h + g.gap) faults.push(`${name}: the canvas is ${g.w} x ${g.h}, not two squares ${g.gap} px apart`);
    const mods = g.gap / g.h * (MODULES(256, RING_DEFAULT) + 2 * OB_QUIET), asked = c.gap ?? 12;
    console.log(`  gap ${g.gap} canvas px, ${mods.toFixed(2)} modules (${asked} asked)`);
    if (Math.abs(mods - asked) > 0.5) faults.push(`${name}: the gap is ${mods.toFixed(2)} modules, not ${asked}`);
  }
  if (c.midStream) await midStream(subch, r);
  // ai: The gap slider moved while it runs (2026-10-03): its change event, as a release gives it, re-picks (send.mjs
  // ai: repickOnce: the worker's respec, and under the GPU encoder a new configuration) and the canvas's gap follows.
  // ai: Grabs until the gap is the one asked (5 s at most), then both halves read blind, every block, 0 bad.
  if (c.regap !== undefined) {
    await evaluate(s, `(() => { const e = document.getElementById("gap"); e.value = "${c.regap}"; e.dispatchEvent(new Event("change")); })()`);
    const T = MODULES(256, RING_DEFAULT) + 2 * OB_QUIET, t0 = Date.now();
    let o = null, mods = -1;
    while (Date.now() - t0 < 5000) {
      await sleep(100);
      o = await readOnce(codes);
      mods = o.g.gap / o.g.h * T;
      if (Math.abs(mods - c.regap) <= 0.5 && o.halves.every((h) => h.fmt)) break;
    }
    const n2 = `${name}, the gap moved to ${c.regap} while running`;
    console.log(`${n2}: ${((Date.now() - t0) / 1000).toFixed(1)} s, gap ${mods.toFixed(2)} modules, halves ${o.halves.map((h) => `version ${h.fmt?.version ?? "none"} ${h.seen} verified ${h.bad} bad`).join(", ")}`);
    if (Math.abs(mods - c.regap) > 0.5) faults.push(`${n2}: the gap is ${mods.toFixed(2)} modules after 5 s`);
    for (const [k, h] of o.halves.entries()) if (!(h.fmt && h.seen > 0 && h.bad === 0 && h.seen === h.fmt.version)) faults.push(`${n2}: half ${k} read ${h.seen} blocks, ${h.bad} bad, word ${h.fmt ? h.fmt.version : "none"}`);
    if (o.g.w !== 2 * o.g.h + o.g.gap) faults.push(`${n2}: the canvas is ${o.g.w} x ${o.g.h}, not two squares ${o.g.gap} px apart`);
  }
}

// ai: The canvas grabbed and decoded from the moment of a change until want(r) holds, or ms pass. Every grab on the way
// ai: must read a word `ok` allows (the frames the change left in flight, or the new ones) with 0 bad. A grab the blind
// ai: decoder reads no word on is decoded told at the sub-channel counts in subs (readTold): read so, with 0 bad and a
// ai: word ok allows, it is a good frame the blind decoder missed (the C's blind decoder registers the wrong ring on a
// ai: few clean frames, 2026-09-29, the C's own paint of the same ids alike), counted as missed; read by neither, it is
// ai: a stray, a fault. A grab of the GPU canvas that saw nothing presented in 120 animation frames (GRAB throws) is
// ai: counted as idle, not judged. before: the words read on the way, run-length ("60@24 x3"); ids: every id they verified.
async function grabUntil(want, ok, subs, ms) {
  const t0 = Date.now(), words = [], ids = [];
  let out = null, idle = 0, strays = 0, missed = 0, bad = 0;
  for (;;) {
    let x = null;
    try { x = await readOnce(); } catch (e) { if (!/never read in a frame it was presented in/.test(e.message)) throw e; idle++; }
    if (x) {
      out = x; bad += x.r.bad;
      if (want(x.r)) break;
      let word = x.r.fmt ? `${x.r.fmt.version}@${x.r.fmt.fps}` : "no word";
      if (!x.r.fmt) {
        const t = await readTold(x, subs);
        if (t) bad += t.r.bad;
        if (t && t.r.bad === 0 && ok(t.r)) { missed++; word = `missed blind, told ${t.r.fmt.version}@${t.r.fmt.fps}`; for (const b of t.r.got) ids.push(b.id); }
        else strays++;
      } else if (!ok(x.r)) strays++;
      words.push(word);
      for (const b of x.r.got) ids.push(b.id);
    }
    if (Date.now() - t0 > ms) { out = null; break; }
  }
  const before = [];
  for (const w of words) { const last = before.at(-1); if (last?.w === w) last.k++; else before.push({ w, k: 1 }); }
  return { ...(out ?? {}), found: !!out, before: before.map(({ w, k }) => (k > 1 ? `${w} x${k}` : w)).join(", ") || "none", grabs: words.length + (out ? 1 : 0), idle, strays, missed, bad, ids, secs: (Date.now() - t0) / 1000 };
}
// ai: The sixth case's second half (the header): the re-pick to the room of a 1280 x 720 viewport, then 30 fps asked.
// ai: r0: the case's first grab, at the old version. Block ids rise frame to frame (send.mjs nextId, never seq x
// ai: blocksPerFrame: a re-pick changes the blocks a frame), so each change's first frame must carry only ids above every
// ai: id read before it, or a receiver would count the new frames' blocks as repeats.
async function midStream(subch0, r0) {
  const v0 = subch0 / 8;
  let maxId = Math.max(-1, ...r0.got.map((b) => b.id));
  await s.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  const t0 = Date.now();
  let m = await measure();
  for (let t = 0; t < 50 && (m.w !== 1280 || m.h !== 720); t++) { await sleep(100); m = await measure(); }
  const subch = pickVersion(m.room).subch, v = subch / 8;
  let name = `re-pick in a ${m.w} x ${m.h} window (a ${m.room} px room): LIZARD-${subch0} to LIZARD-${subch}, test stream`;
  if (m.w !== 1280 || m.h !== 720) { faults.push(`${name}: the window is not 1280 x 720`); return; }
  if (subch === subch0) { faults.push(`${name}: the smaller room picks the same version, so nothing re-picks`); return; }
  // ai: The change's first frame on the canvas: its word, blocks, picture and ids, the canvas it was read off and the state
  // ai: line at that moment.
  const judge = (o, what, fps) => {
    if (!o.found) { faults.push(`${what}: no grab read version ${v} at ${fps} fps in ${o.secs.toFixed(0)} s (read on the way: ${o.before}; ${o.idle} idle)`); return false; }
    const { g, r } = o, ids = r.got.map((b) => b.id), lo = Math.min(...ids), prior = Math.max(maxId, ...o.ids);
    console.log(`${what}: ${o.grabs} grab${o.grabs === 1 ? "" : "s"} in ${o.secs.toFixed(1)} s until the word read it (on the way: ${o.before}${o.missed ? `; ${o.missed} missed by the blind decoder, read told` : ""}${o.idle ? `; ${o.idle} with nothing presented` : ""}); ${decoded(g, r)} (want n = ${N_FOR(subch)}), ids ${lo} to ${Math.max(...ids)}, the highest read before ${prior}`);
    if (!(r.fmt?.version === v && r.fmt?.fps === fps)) faults.push(`${what}: the word is not version ${v} at ${fps} fps`);
    if (!(r.seen > 0 && r.bad === 0)) faults.push(`${what}: ${r.seen} blocks, ${r.bad} bad`);
    if (o.bad) faults.push(`${what}: ${o.bad} bad blocks over the grabs until it`);
    if (o.strays) faults.push(`${what}: ${o.strays} grabs on the way that neither the blind decoder nor a told one reads as the old frames or the new (${o.before})`);
    if (r.n !== N_FOR(subch)) faults.push(`${what}: registered at n = ${r.n} (want ${N_FOR(subch)})`);
    if (g.state !== "Sending the test stream") faults.push(`${what}: the state line says "${g.state}"`);
    if (g.which !== (GPU ? "cg" : "c")) faults.push(`${what}: read off #${g.which}`);
    if (!(lo > prior)) faults.push(`${what}: its ids start at ${lo}, not above the ${prior} read before it`);
    maxId = Math.max(prior, ...ids);
    return true;
  };
  // ai: The page's own lines once its report after the change is written (#tx and #nums together, once a second, so a
  // ai: frame's grab can hold the report from before the change): said(p) must hold within 3 s, and then the encoder line
  // ai: must still be the arm's (under the GPU encoder no stop or fallback since) and the state line sending.
  const PAGE = `(() => ({ nums: document.getElementById("nums").textContent, lab: document.getElementById("tx").textContent.split("\\n")[0], tx: document.getElementById("tx").textContent, state: document.getElementById("state").textContent, which: ["cg", "c"].find((id) => !document.getElementById(id).hidden) }))()`;
  const reported = async (what, said, want) => {
    let p = await evaluate(s, PAGE);
    for (let t = 0; t < 30 && !said(p); t++) { await sleep(100); p = await evaluate(s, PAGE); }
    const enc = encoderOf(p);
    console.log(pageLine(p));
    if (!said(p)) faults.push(`${what}: 3 s after the canvas showed it the page does not say ${want}: #tx "${p.lab}", #nums "${p.nums}", #tx "${p.tx.split("\n").find((l) => l.startsWith("painted ")) ?? ""}"`);
    if (!enc.ok) faults.push(`${what}: the encoder is not the one asked for: ${enc.line}, on #${p.which}`);
    if (p.state !== "Sending the test stream") faults.push(`${what}: the state line says "${p.state}"`);
  };
  // ai: The re-pick: grabbed from the resize on (the old version until the worker, and under the GPU encoder the new
  // ai: configuration, has the new code, and the frames asked for before it have painted); the first frame at the new
  // ai: version is judged, then #nums must name it.
  let o = await grabUntil((r) => r.fmt?.version === v, (r) => (r.fmt?.version === v0 || r.fmt?.version === v) && r.fmt?.fps === FPS, [subch0, subch], 60000);
  if (!judge(o, name, FPS)) return;
  // ai: the version is #tx's first line's since 2026-10-02 (the lab line #lab's from 2026-10-01); #nums is the rate
  await reported(name, (p) => p.lab.split(/[ ,]/)[0] === NAME(subch), `${NAME(subch)} in #tx`);
  // ai: The display rate, set as a script sets it (no change event, so the page keeps nothing); the page's tick states it
  // ai: from the box. Grabbed from the change on; the first frame whose word says 30 is judged, then #tx must say 30 asked.
  name = `${NAME(subch)} with the display fps box moved from ${FPS} to 30`;
  await evaluate(s, `document.getElementById("fps").value = 30`);
  o = await grabUntil((r) => r.fmt?.fps === 30, (r) => r.fmt?.version === v && (r.fmt?.fps === FPS || r.fmt?.fps === 30), [subch], 30000);
  if (judge(o, name, 30)) await reported(name, (p) => / of 30 asked/.test(p.tx), `"of 30 asked" in #tx`);
}
s.close();
cleanup();
phy.free();

const api = log.filter((l) => / \/api\//.test(l)), gets = api.filter((l) => !l.startsWith("POST ")), config = api.filter((l) => l.includes("/api/config"));
console.log(`static server: ${log.length} requests; under /api: ${api.length} (${[...new Set(api)].join(", ") || "none"}); GETs there ${gets.length}, /api/config ${config.length}`);
if (gets.length) faults.push(`the page asked the server for ${gets.join(", ")}`);
if (config.length) faults.push("the page asked for /api/config");
if (faults.length) { console.log(`FAIL\n  ${faults.join("\n  ")}`); process.exit(1); }
console.log("ok");
process.exit(0);
