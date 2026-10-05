// ai: The installed app (2026-10-01, a content-hashed PWA): lizard-web/app/, built from the rig's source pages
// ai: (lizard-web/index.html, send.html, recv.html) and the files they reach, on a scheme ported from another project:
// ai: build output, never edited by hand.
// ai:   1. crawl: from the three pages, every file a page or a shipped module names by a literal (an import specifier,
// ai:      new URL(..., import.meta.url), a fetched path; referrer-relative, under liblizard/ (the nets' paths,
// ai:      "gpu/cnn/weights.safetensors"), or a bare name beside the referrer), comments left out so a path in one
// ai:      ships nothing; the codec and the fountain (BUILT, liblizard/build.sh's output) among them, or no app;
// ai:   2. copy: the pages to the root, their references re-pointed, and every other file under assets/ at its path under
// ai:      the project's root (assets/lizard-web/recv.mjs, assets/liblizard/sim/phy.mjs,
// ai:      assets/liblizard/gpu/cnn/weights.safetensors), so the modules' relative references hold as they are; into each page's head the manifest, the theme colour and the build;
// ai:   3. strip comments (strip.mjs; not the emscripten glue), then parse every module again (node --check);
// ai:   4. content-address assets/ (hash.mjs): name.<hash>.ext, every reference rewritten, in dependency order;
// ai:   5. the worker (service-worker.js, the template here): its cache name, its shell (the pages, the manifest, every
// ai:      asset) and the SHA-256 of every byte that ships;
// ai:   6. verify: every name a true hash, no original name left, every listed path present.
// ai: node lizard-web/pwa/build.mjs [out] (from anywhere; out relative to lizard-web/, app by default). About a second.
import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync, statSync, copyFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { dirname, join, posix, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { strip } from "./strip.mjs";
import { hashTree, verifyTree, findTokens, walk } from "./hash.mjs";

// ai: ROOT: the project's root, which every source path below is relative to and assets/ mirrors; LIB: the base of a
// ai: path the nets' literals name ("gpu/cnn/weights.safetensors", liblizard/gpu's bank_fcn2.mjs and gpuqueue.mjs
// ai: load them from liblizard/).
const HERE = dirname(fileURLToPath(import.meta.url)), WEB = resolve(HERE, ".."), ROOT = resolve(WEB, ".."), LIB = "liblizard";
const OUT = resolve(WEB, process.argv[2] ?? "app");
const PAGES = ["lizard-web/index.html", "lizard-web/send.html", "lizard-web/recv.html"];
// ai: what ships: the pages' own kinds of file, the codec's wasm and the nets
const SHIP = new Set([".html", ".css", ".mjs", ".js", ".svg", ".wasm", ".safetensors", ".json"]);
const SCAN = new Set([".html", ".css", ".mjs", ".js", ".svg"]);
// ai: emscripten's output (-O3): no comments, and its own string literals are the bare names it loads
const GLUE = new Set(["liblizard/build/ob.mjs", "liblizard/build/wirehair.mjs", "liblizard/build/zstd.mjs"]);
// ai: liblizard/build.sh's output, built and not in git: the crawl drops a name whose file is not there, so before
// ai: 2026-10-04 an app was built without the codec and the fountain where they were not built. Each must ship.
const BUILT = { "liblizard/build/ob.mjs": "liblizard/build.sh", "liblizard/build/ob.wasm": "liblizard/build.sh", "liblizard/build/wirehair.mjs": "liblizard/build.sh wirehair",
  "liblizard/build/zstd.mjs": "liblizard/build.sh zstd" };
const fail = (msg) => { console.error(`build: ${msg}`); process.exit(1); };
const isFile = (p) => { try { return statSync(p).isFile(); } catch { return false; } };

// --- 1. crawl -----------------------------------------------------------------------
// ai: A token's file in the source tree, by the three bases hash.mjs takes in the built one; a query or fragment is the
// ai: page's, not the file's (Home's "recv.html?auto").
function sourceRef(rel, token) {
  const tok = token.split(/[?#]/)[0];
  if (!tok || /^[a-z][a-z0-9+.-]*:/i.test(tok) || tok.startsWith("/")) return null;
  const here = posix.dirname(rel) === "." ? "" : posix.dirname(rel);
  const cands = tok.includes("/") ? [posix.normalize(posix.join(here, tok)), posix.normalize(posix.join(LIB, tok))] : [posix.join(here, tok)];
  const hits = [...new Set(cands)].filter((c) => !c.startsWith("..") && SHIP.has(posix.extname(c)) && isFile(join(ROOT, c)));
  if (hits.length > 1) fail(`an ambiguous reference in ${rel}: ${token} -> ${hits.join(", ")}`);
  return hits[0] ?? null;
}
const shipped = new Set(), queue = [...PAGES];
while (queue.length) {
  const rel = queue.shift();
  if (shipped.has(rel)) continue;
  if (posix.extname(rel) === ".html" && !PAGES.includes(rel)) fail(`${rel} is named by a shipped file, and the app carries only ${PAGES.join(", ")}`);
  shipped.add(rel);
  const ext = posix.extname(rel);
  if (!SCAN.has(ext)) continue;
  let text = readFileSync(join(ROOT, rel), "utf8");
  if (!GLUE.has(rel)) try { text = strip(text, ext); } catch (e) { fail(`${rel}: ${e.message}`); }
  for (const t of findTokens(rel, text)) {
    const hit = sourceRef(rel, t.value);
    if (hit && !shipped.has(hit)) queue.push(hit);
  }
}
const unshipped = Object.keys(BUILT).filter((f) => !shipped.has(f)), unbuilt = unshipped.filter((f) => !isFile(join(ROOT, f)));
if (unbuilt.length) fail(`${unbuilt.join(", ")} not built: run ${[...new Set(unbuilt.map((f) => BUILT[f]))].join(" and ")} first (Emscripten; README.md, Building)`);
if (unshipped.length) fail(`${unshipped.join(", ")} named by no shipped file`);

// --- 2. copy ----------------------------------------------------------------------------
const unix = Math.floor(Date.now() / 1000), cacheName = `lizard-v${unix}`, built = new Date(unix * 1000).toISOString();
rmSync(OUT, { recursive: true, force: true });
for (const rel of shipped) {
  if (PAGES.includes(rel)) continue;
  mkdirSync(join(OUT, "assets", posix.dirname(rel)), { recursive: true });
  copyFileSync(join(ROOT, rel), join(OUT, "assets", rel));
}
const HEAD = `<link rel="manifest" href="manifest.webmanifest"><meta name="theme-color" content="#ffffff"><meta name="lizard-build" content="${built} ${cacheName}">`;
for (const rel of PAGES) {
  let html = readFileSync(join(ROOT, rel), "utf8");
  // ai: from the end, so the recorded offsets hold: a reference to a shipped file other than a page goes under assets/
  const hits = findTokens(rel, html).map((t) => ({ ...t, target: sourceRef(rel, t.value) })).filter((t) => t.target && !PAGES.includes(t.target));
  for (const t of hits.sort((a, b) => b.start - a.start)) html = html.slice(0, t.start) + `assets/${t.target}${t.value.slice(t.value.split(/[?#]/)[0].length)}` + html.slice(t.end);
  if (!html.includes("</title>")) fail(`${rel} has no <title> to put the app's head after`);
  html = html.replace("</title>", `</title>${HEAD}`);
  writeFileSync(join(OUT, posix.basename(rel)), html);
}
copyFileSync(join(HERE, "manifest.webmanifest"), join(OUT, "manifest.webmanifest"));

// --- 3. strip ---------------------------------------------------------------------------------
let before = 0, after = 0;
for (const rel of walk(OUT)) {
  const ext = posix.extname(rel);
  if (!SCAN.has(ext) || GLUE.has(rel.replace(/^assets\//, ""))) continue;
  const src = readFileSync(join(OUT, rel), "utf8");
  let out;
  try { out = strip(src, ext); } catch (e) { fail(`${rel}: ${e.message}`); }
  before += Buffer.byteLength(src); after += Buffer.byteLength(out);
  writeFileSync(join(OUT, rel), out);
}
// ai: the stripper's one judgement is regex against division; a module it misjudged no longer parses
const broken = walk(OUT).filter((rel) => /\.(mjs|js)$/.test(rel)).filter((rel) => spawnSync(process.execPath, ["--check", join(OUT, rel)], { encoding: "utf8" }).status !== 0);
if (broken.length) fail(`a stripped module no longer parses: ${broken.join(", ")}`);

// --- 4. hash ----------------------------------------------------------------------------------
let renamed;
try { ({ renamed } = hashTree(OUT)); } catch (e) { fail(e.message); }

// --- 5. the worker ------------------------------------------------------------------------------
const sha = (rel) => createHash("sha256").update(readFileSync(join(OUT, rel))).digest("hex");
const pages = ["./", "./index.html", "./send.html", "./recv.html"];
const assets = ["./manifest.webmanifest", ...walk(OUT).filter((f) => f.startsWith("assets/")).sort().map((f) => `./${f}`)];
const fileOf = (p) => (p === "./" ? "index.html" : p.slice(2));
const list = (paths) => `[\n${paths.map((p) => `  "${p}",`).join("\n")}\n]`;
let sw = strip(readFileSync(join(HERE, "service-worker.js"), "utf8"), ".js");
const fill = [
  ["const CACHE_NAME = `${CACHE_PREFIX}v0`;", `const CACHE_NAME = \`\${CACHE_PREFIX}v${unix}\`;`],
  ["const SHELL_PAGE_URLS = [];", `const SHELL_PAGE_URLS = ${list(pages)};`],
  ["const SHELL_ASSET_URLS = [];", `const SHELL_ASSET_URLS = ${list(assets)};`],
  ["const SHELL_HASHES = {};", `const SHELL_HASHES = {\n${[...pages, ...assets].map((p) => `  "${p}": "${sha(fileOf(p))}",`).join("\n")}\n};`],
];
for (const [mark, line] of fill) {
  if (sw.split(mark).length !== 2) fail(`service-worker.js: "${mark}" is not there exactly once`);
  sw = sw.replace(mark, () => line);
}
writeFileSync(join(OUT, "service-worker.js"), sw);
// ai: Cloudflare Pages' headers (2026-10-03, the app deployed there as a static folder): the
// ai: content-addressed assets kept a year without asking again (a name is its bytes); the pages, the manifest and the
// ai: worker left to Pages' default, revalidated on every load. Pages reads the file and does not serve it, and nothing
// ai: loads it, so it is outside the worker's shell. Other hosts ignore it.
writeFileSync(join(OUT, "_headers"), "/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n");
// ai: The licenses the app ships under, one text beside the pages, outside the worker's shell as _headers is: its wasm
// ai: carries the codec, BLAKE3 and Wirehair (whose BSD 3-Clause asks for its notice with a binary), and the build
// ai: strips every comment.
const NOTICES = [["", "LICENSE"], ["", "NOTICE"], ["Wirehair (liblizard/vendor/wirehair)", "liblizard/vendor/wirehair/LICENSE"],
  ["BLAKE3 (liblizard/vendor/blake3), under its Apache License 2.0 option", "liblizard/vendor/blake3/LICENSE_A2"]];
writeFileSync(join(OUT, "LICENSES.txt"), NOTICES.map(([t, f]) => (t ? `${t}\n\n` : "") + readFileSync(join(ROOT, f), "utf8").trimEnd()).join(`\n\n${"-".repeat(78)}\n\n`) + "\n");

// --- 6. verify ------------------------------------------------------------------------------------
const problems = verifyTree(OUT);
for (const p of [...pages, ...assets]) if (!existsSync(join(OUT, fileOf(p)))) problems.push(`listed, not there: ${p}`);
if (problems.length) { for (const p of problems) console.error(`  ${p}`); fail(`${problems.length} problem(s) in ${OUT}`); }
const bytes = walk(OUT).reduce((s, f) => s + statSync(join(OUT, f)).size, 0);
console.log(`built ${posix.relative(ROOT, OUT) || OUT}: ${pages.length - 1} pages and ${assets.length} shell files (${renamed.size} content-addressed), ${(bytes / 1e6).toFixed(2)} MB; comments ${((before - after) / 1e3).toFixed(0)} KB of ${(before / 1e3).toFixed(0)} KB; ${cacheName}, ${built}`);
