// Test rig for the PHY candidates: a sender page for the laptop and a receiver page for a phone.
//   node lizard-web/server.mjs      (from the project's root)
// Needs openssl (the https certificate, made once); python3's qrcode module, where installed, prints a QR of the
// phone's address. RIG_HTTP, RIG_HTTPS: other ports (8080, 8443); RIG_UPLOADS=1: take /api/file posts.
// It serves the repository to the LAN but no dot folder (its key) and none of research/rig/.
// Serves the repository over http (localhost) and https (LAN, self-signed: a phone's camera
// ai: needs a secure origin), and takes the pages' development logs, the only thing a page sends it (2026-09-26):
// ai: nothing it answers is read by a
// ai: page to decode or paint. POSTs: /api/stats (the receiver's row a second, the latest kept in memory alone: rows are
// ai: logged only in a replay since 2026-10-09, research/rig/stats.jsonl no longer written), /api/sender (the sender's line
// ai: and config), /api/selftest, /api/fft,
// ai: /api/file. GETs for node scripts: /api/info, /api/stats (the last row, for lizard-web/check_rates.mjs).
import { createServer as http } from "node:http";
import { createServer as https } from "node:https";
import { readFileSync, existsSync, mkdirSync, writeFileSync, statSync, appendFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { networkInterfaces } from "node:os";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url)), root = resolve(here, ".."), RIG = resolve(root, "research/rig"), HTTP = +process.env.RIG_HTTP || 8080, HTTPS = +process.env.RIG_HTTPS || 8443;   // the self-check runs its own instance on other ports
const lan = Object.values(networkInterfaces()).flat().find((a) => a.family === "IPv4" && !a.internal)?.address ?? "127.0.0.1";
const MIME = { ".html": "text/html; charset=utf-8", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".svg": "image/svg+xml", ".css": "text/css", ".webmanifest": "application/manifest+json", ".woff2": "font/woff2" };   // ai: the installed app's manifest (lizard-web/app/, 2026-10-01); the pages' fonts (2026-10-10)

const certDir = join(here, ".cert");
if (!existsSync(join(certDir, `${lan}.pem`))) {
  mkdirSync(certDir, { recursive: true });
  execFileSync("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365", "-keyout", join(certDir, `${lan}.key`), "-out", join(certDir, `${lan}.pem`),
    "-subj", "/CN=barcode-rig", "-addext", `subjectAltName=IP:${lan},IP:127.0.0.1,DNS:localhost`], { stdio: "ignore" });
}

const phoneUrl = `https://${lan}:${HTTPS}/lizard-web/recv.html`;
// ai: The phone's address as a QR, printed in this terminal at start (the pages show none), where python3 has the
// ai: qrcode module; without it the address alone (the server died at start there until 2026-10-04, and with it the
// ai: checks that start one).
let matrix = null;
try { matrix = JSON.parse(execFileSync("python3", ["-c", "import qrcode,json,sys; q=qrcode.QRCode(border=2); q.add_data(sys.argv[1]); q.make(); print(json.dumps(q.get_matrix()))", phoneUrl], { stdio: ["ignore", "pipe", "ignore"] }).toString()); } catch {}

let stats = {}, statsAt = 0, sender = {}, senderAt = 0;
const body = (req) => new Promise((ok) => { const c = []; req.on("data", (d) => c.push(d)); req.on("end", () => ok(Buffer.concat(c))); });

async function handle(req, res) {
  const url = new URL(req.url, "http://x"), p = url.pathname;
  const json = (o) => { res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" }); res.end(JSON.stringify(o)); };
  if (p === "/") { res.writeHead(302, { location: "/lizard-web/" }); return res.end(); }
  if (p === "/api/info") return json({ phoneUrl, lan, https: HTTPS, http: HTTP });
  if (p === "/api/stats") {
    if (req.method === "POST") {
      // ai: the latest row alone, for the checks' GET (2026-10-09: no row is appended to a log here; rows are logged
      // ai: only as part of a replay, the Android app's Save replays)
      stats = JSON.parse(await body(req)); statsAt = Date.now();
    }
    // senderAgeSecs is worked out here, against the rig's own clock. The receiver is a different
    // device and its clock can be minutes off, so it must never subtract one from the other.
    // ai: A post is answered { ok }: the page reads nothing back. A GET is a script's (the last row, 5 s at most old).
    if (req.method === "POST") return json({ ok: true });
    return json({ stats: Date.now() - statsAt < 5000 ? stats : {}, sender, senderAgeSecs: senderAt ? (Date.now() - senderAt) / 1000 : null });
  }   // a receiver that went away must not look live
  if (p === "/api/sender" && req.method === "POST") { sender = JSON.parse(await body(req)); senderAt = Date.now(); return json({ ok: true }); }
  // ai: The GPU self-test's report (gpu_selftest.html), a line each in research/rig/selftest.jsonl stamped with the rig's clock,
  // ai: and its verdict printed here.
  if (p === "/api/selftest" && req.method === "POST") {
    const rep = JSON.parse(await body(req)), at = Date.now();
    mkdirSync(RIG, { recursive: true });
    appendFileSync(join(RIG, "selftest.jsonl"), JSON.stringify({ rigAt: at, ...rep }) + "\n");
    const bad = rep.runs?.find((r) => r.error || r.errors?.length || r.lost || r.first), a = rep.adapter?.info ?? {};
    const why = rep.error ?? (bad ? `${bad.precision}: ${bad.error ?? (bad.lost ? `lost ${bad.lost}` : bad.first ? `first difference ${bad.first.stage}: ${bad.first.first.note}` : bad.errors[0])}` : "every stage matches");
    console.log(`\n--- GPU self-test, ${[a.vendor, a.architecture, a.description].filter(Boolean).join(" ") || "no adapter"}: ${why} (research/rig/selftest.jsonl)\n`);
    return json({ ok: true });
  }
  // scripts/pages/fft.html's table, printed here so the phone's numbers do not have to be transcribed off its screen.
  if (p === "/api/fft" && req.method === "POST") { console.log(`\n--- transform bench from the phone ---\n${await body(req)}\n`); return json({ ok: true }); }
  // ai: Uploads (a received file) only where asked for, RIG_UPLOADS=1 (rig.sh, the checks that post): a rig on the LAN
  // ai: otherwise takes no writes. The pages record no replays since 2026-10-09 (the Android app's Save replays does).
  if (p === "/api/file" && req.method === "POST" && process.env.RIG_UPLOADS !== "1") { res.writeHead(403); return res.end("uploads off (RIG_UPLOADS=1)"); }
  // A file a receiver finished (recv.html?save=post), for a scripted run to check against what was sent
  // (lizard-web/check_rates.mjs). Into RIG_FILES, or research/rig/received.
  if (p === "/api/file" && req.method === "POST") {
    const name = (url.searchParams.get("name") ?? "file").replace(/[^\w.-]/g, "_"), dir = process.env.RIG_FILES || join(RIG, "received");
    mkdirSync(dir, { recursive: true });
    const bytes = await body(req);
    writeFileSync(join(dir, name), bytes);
    return json({ saved: name, bytes: bytes.length });
  }
  // ai: Files under the repository's root, none in a dot folder (the rig's own key in lizard-web/.cert/, .git/, a "..")
  // ai: and none of the rig's records (research/rig/: the stats rows and what was posted): it serves the LAN.
  const rel = decodeURIComponent(p);
  if (rel.split("/").some((s) => s.startsWith("."))) { res.writeHead(404); return res.end("not found"); }
  let file = normalize(join(root, rel));
  if (file !== root && !file.startsWith(root + sep)) { res.writeHead(403); return res.end(); }
  if (file === RIG || file.startsWith(RIG + sep)) { res.writeHead(404); return res.end("not found"); }   // ai: as normalised: "//" collapses
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  if (!existsSync(file)) { res.writeHead(404); return res.end("not found"); }
  res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
  res.end(readFileSync(file));
}
const safe = (req, res) => handle(req, res).catch((e) => { res.writeHead(500); res.end(String(e)); });
const inUse = (port) => (e) => { console.error(`server: port ${port}: ${e.code === "EADDRINUSE" ? "in use (RIG_HTTP and RIG_HTTPS name other ports)" : e.message}`); process.exit(1); };
http(safe).on("error", inUse(HTTP)).listen(HTTP, "0.0.0.0");
https({ key: readFileSync(join(certDir, `${lan}.key`)), cert: readFileSync(join(certDir, `${lan}.pem`)) }, safe).on("error", inUse(HTTPS)).listen(HTTPS, "0.0.0.0");

const rows = [];
for (let y = 0; y < (matrix?.length ?? 0); y += 2) rows.push(matrix[y].map((t, x) => { const b = matrix[y + 1]?.[x]; return t && b ? " " : t ? "▄" : b ? "▀" : "█"; }).join(""));
console.log(`\nLAPTOP (sender):   http://localhost:${HTTP}/lizard-web/send.html`);
console.log(`PHONE  (receiver): ${phoneUrl}`);
console.log(`\n${rows.length ? "Scan this with the phone's camera app" : "Open that address on the phone"}. Same Wi-Fi as this machine. The certificate is\nself-signed, so Chrome warns once: tap Advanced, then Proceed. Then allow the camera.\n`);
console.log(rows.length ? "\x1b[30;47m" + rows.join("\x1b[0m\n\x1b[30;47m") + "\x1b[0m" : "(A QR of it here needs python3's qrcode module.)");
console.log(`\nNo Wi-Fi path? USB instead: enable USB debugging, open chrome://inspect/#devices on the laptop,\nPort forwarding ${HTTP} -> localhost:${HTTP}, then open http://localhost:${HTTP}/lizard-web/recv.html on the phone.\n`);
