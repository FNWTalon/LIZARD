// ai: Content-addresses the installed app's tree (lizard-web/pwa/build.mjs): every file under assets/ renamed name.<hash>.ext
// ai: (the first 10 hex digits of the SHA-256 of the bytes it ships with) and every reference to it rewritten, so a URL
// ai: names one byte sequence for ever: two builds' files never collide, an unchanged file keeps its URL and its cached
// ai: copy, and a page is never handed another build's module. Ported from another project's asset hasher, changed
// ai: where LIZARD differs:
// ai:   - ES modules: import specifiers, new URL("...", import.meta.url), and a backtick literal with no ${} are
// ai:     references like any quoted string;
// ai:   - three bases, tried together (one that lands on a shipped file wins; two that land on two is an error):
// ai:       referrer-relative   "../liblizard/sim/phy.mjs", "./recv-worker.mjs"
// ai:       library-relative    "gpu/cnn/weights.safetensors", a path under liblizard/, which assets/liblizard/ mirrors
// ai:       a bare name         "ob.wasm" in the emscripten glue: a file in the referrer's own folder
// ai:   - verify also reads a template literal's head: `./recv-gpu-worker.mjs${q}` names a file the rewrite cannot reach
// ai:     (a literal with ${} is no reference), so it fails the build here rather than 404ing on a phone. A path built
// ai:     from pieces (`./bank_${name}.mjs`) names no file whole and passes verify: the built app's own runs are its
// ai:     check (lizard-web/check_app.mjs, APP=1 on the rig's checks). A file's name in a message or a shader's comment is
// ai:     no reference, and passes.
// ai: Two rules hold it together, both checked by verify: complete (no reference left unrewritten) and ordered (a file
// ai: hashed only once the references inside it are final, or its name stops being a hash of its bytes).
// ai:   node lizard-web/pwa/hash.mjs --verify <dir>    the checks alone, on a built tree
// ai:   node lizard-web/pwa/hash.mjs --self-test       the hazards, on a miniature tree
import { readFileSync, writeFileSync, renameSync, readdirSync, mkdirSync, mkdtempSync, rmSync, existsSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { join, posix } from "node:path";
import { tmpdir } from "node:os";

export const HASH_LEN = 10;
export const HASHED_RE = new RegExp(`\\.[0-9a-f]{${HASH_LEN}}\\.[^.]+$`);
const RENAME_PREFIX = "assets/";
const LIB_ROOT = "assets/liblizard";
const TEXT_EXT = new Set([".html", ".css", ".mjs", ".js", ".webmanifest", ".svg", ".json"]);

export class HashError extends Error {}

// --- references -----------------------------------------------------------------
// ai: Each pattern captures a prefix and then the token, so the token's offset is m.index + m[1].length: a rewrite at a
// ai: recorded offset cannot touch lookalike text ("fob.mjs" holds "ob.mjs").
const PATTERNS = {
  html: [/((?:src|href)\s*=\s*["'])([^"'<>]*)/gi],
  css: [/(url\(\s*["']?)([^"')]*)/gi],
  code: [/(["'`])((?:\\.|(?!\1)[^\\\n])*)\1/g],
};
function patternsFor(rel) {
  const ext = posix.extname(rel);
  if (ext === ".html") return [...PATTERNS.html, ...PATTERNS.code];
  if (ext === ".css") return PATTERNS.css;
  if (ext === ".svg") return [...PATTERNS.html, ...PATTERNS.css];
  return PATTERNS.code;
}
export function findTokens(rel, text) {
  const out = [], seen = new Set();
  for (const pattern of patternsFor(rel)) {
    pattern.lastIndex = 0;
    let m;
    while ((m = pattern.exec(text)) !== null) {
      const value = m[2];
      if (!value || value.includes("\\") || value.includes("${")) continue;
      const start = m.index + m[1].length, key = `${start}:${value.length}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ start, end: start + value.length, value });
    }
  }
  return out.sort((a, b) => a.start - b.start);
}

// ai: The shipped file a token names, or null. No scheme, fragment or root-absolute path is ours.
function resolveRef(referrerRel, token, known) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(token) || token.startsWith("#") || token.startsWith("/")) return null;
  const dir = posix.dirname(referrerRel), here = dir === "." ? "" : dir, candidates = [];
  if (!token.includes("/")) candidates.push(posix.join(here, token));
  else candidates.push(posix.normalize(posix.join(here, token)), posix.normalize(posix.join(LIB_ROOT, token)));
  const hits = [...new Set(candidates)].filter((c) => !c.startsWith("..") && c.startsWith(RENAME_PREFIX) && known.has(c));
  if (hits.length > 1) throw new HashError(`ambiguous reference in ${referrerRel}: ${token} -> ${hits.join(", ")}`);
  return hits[0] ?? null;
}

const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
function hashedName(base, digest) {
  const ext = posix.extname(base);
  return `${base.slice(0, base.length - ext.length)}.${digest.slice(0, HASH_LEN)}${ext}`;
}
function swapBaseName(token, base) {
  const cut = token.lastIndexOf("/");
  return cut === -1 ? base : token.slice(0, cut + 1) + base;
}
export function walk(root, rel = "") {
  const out = [];
  for (const e of readdirSync(join(root, rel), { withFileTypes: true })) {
    const child = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) out.push(...walk(root, child));
    else if (e.isFile()) out.push(child);
  }
  return out;
}
function applyRewrites(text, hits, finalName) {
  let out = text;
  for (const hit of [...hits].sort((a, b) => b.start - a.start)) {
    const next = finalName.get(hit.target);
    if (next) out = out.slice(0, hit.start) + swapBaseName(hit.value, posix.basename(next)) + out.slice(hit.end);
  }
  return out;
}

// --- the stage ----------------------------------------------------------------------
// ai: Returns { renamed, rewritten }: the assets renamed (old path -> new) and the files that keep their names but were
// ai: rewritten (the pages, the manifest).
export function hashTree(root) {
  const files = walk(root), known = new Set(files);
  const renameable = files.filter((f) => f.startsWith(RENAME_PREFIX) && !HASHED_RE.test(f));
  // ai: pass 1: every reference in every text file, as offsets into it
  const refs = new Map();
  for (const rel of files) {
    if (!TEXT_EXT.has(posix.extname(rel))) continue;
    const text = readFileSync(join(root, rel), "utf8"), hits = [];
    for (const t of findTokens(rel, text)) {
      const target = resolveRef(rel, t.value, known);
      if (target) hits.push({ ...t, target });
    }
    if (hits.length) refs.set(rel, { text, hits });
  }
  // ai: pass 2: the assets in dependency order, each hashed once the references inside it are final
  const finalName = new Map(), state = new Map();
  const visit = (rel, trail) => {
    if (finalName.has(rel)) return;
    if (state.get(rel) === "open") throw new HashError(`reference cycle: ${[...trail, rel].join(" -> ")}`);
    state.set(rel, "open");
    const entry = refs.get(rel);
    if (entry) for (const hit of entry.hits) if (hit.target !== rel) visit(hit.target, [...trail, rel]);
    const body = entry ? Buffer.from(applyRewrites(entry.text, entry.hits, finalName), "utf8") : readFileSync(join(root, rel));
    const next = posix.join(posix.dirname(rel), hashedName(posix.basename(rel), sha256(body)));
    writeFileSync(join(root, rel), body);
    renameSync(join(root, rel), join(root, next));
    state.set(rel, "done");
    finalName.set(rel, next);
  };
  for (const rel of renameable) visit(rel, []);
  // ai: pass 3: the files that keep their names
  let rewritten = 0;
  for (const [rel, entry] of refs) {
    if (finalName.has(rel)) continue;
    writeFileSync(join(root, rel), applyRewrites(entry.text, entry.hits, finalName));
    rewritten++;
  }
  return { renamed: finalName, rewritten };
}

// --- verify ---------------------------------------------------------------------------
// ai: The problems in a built tree, [] when there are none: a name under assets/ that is not a true hash of its bytes
// ai: (ordered), and a reference still naming a renamed file by its original name (complete): a token whose last path
// ai: segment is one (the ported hasher's check), or a template literal's head before its first ${ ending in one. The second
// ai: is inventory-driven, not resolution-driven: a reference that was missed names a file that no longer exists, so it
// ai: resolves to nothing, and a resolution check would pass it.
const TEMPLATE_HEAD = /`((?:\\.|[^\\`\n])*?)\$\{/g;
export function verifyTree(root) {
  const files = walk(root), problems = [], original = new Set();
  for (const rel of files) {
    if (!rel.startsWith(RENAME_PREFIX)) continue;
    const base = posix.basename(rel);
    if (!HASHED_RE.test(base)) { problems.push(`not content-addressed: ${rel}`); continue; }
    const claimed = base.split(".").slice(-2, -1)[0], actual = sha256(readFileSync(join(root, rel))).slice(0, HASH_LEN);
    if (claimed !== actual) problems.push(`the hash in its name is stale: ${rel} says ${claimed}, its bytes ${actual}`);
    original.add(base.replace(HASHED_RE, (m) => m.slice(HASH_LEN + 1)));
  }
  const lastSegment = (v) => v.split(/[?#]/)[0].split("/").pop();
  for (const rel of files) {
    if (!TEXT_EXT.has(posix.extname(rel))) continue;
    const text = readFileSync(join(root, rel), "utf8");
    for (const t of findTokens(rel, text)) {
      if (/^[a-z][a-z0-9+.-]*:/i.test(t.value) || t.value.startsWith("//")) continue;
      if (original.has(lastSegment(t.value))) problems.push(`a reference left unrewritten in ${rel}: ${t.value}`);
    }
    if (posix.extname(rel) !== ".mjs" && posix.extname(rel) !== ".js") continue;
    TEMPLATE_HEAD.lastIndex = 0;
    for (let m; (m = TEMPLATE_HEAD.exec(text)) !== null; ) {
      if (original.has(lastSegment(m[1]))) problems.push(`a name put together at run time in ${rel}: \`${m[1]}\${...}\``);
    }
  }
  return problems;
}

// --- self-test ----------------------------------------------------------------------------
function selfTest() {
  const tmp = mkdtempSync(join(tmpdir(), "lizard-hash-"));
  let failures = 0;
  const check = (name, ok, detail) => { console.log(`${ok ? "  ok  " : "  FAIL"} ${name}${ok || !detail ? "" : `: ${detail}`}`); if (!ok) failures++; };
  const write = (rel, body) => { mkdirSync(join(tmp, posix.dirname(rel)), { recursive: true }); writeFileSync(join(tmp, rel), body); };
  // ai: a miniature of the built app, with every hazard the real one has
  write("index.html", '<link rel="stylesheet" href="assets/lizard-web/ui.css"><link rel="icon" href="assets/lizard-web/icon.svg"><script type="module" src="assets/lizard-web/recv.mjs"></script><a href="send.html">Send</a><a href="https://example.test/x.mjs">x</a>');
  write("send.html", "<p>send</p>");
  write("manifest.webmanifest", '{ "icons": [{ "src": "./assets/lizard-web/icon.svg" }] }');
  write("assets/lizard-web/ui.css", '.a{background:url("icon.svg")}');
  write("assets/lizard-web/icon.svg", "<svg/>");
  write("assets/lizard-web/recv.mjs", [
    'import { phy } from "../liblizard/sim/phy.mjs";',
    'import { fob } from "../liblizard/sim/fob.mjs";',
    'const w = new Worker(new URL("./recv-worker.mjs", import.meta.url), { type: "module" });',
    'const NET = "gpu/cnn/weights.safetensors";',
    "const bank = () => import(`./bank.mjs`);",
    'const page = "send.html";',
  ].join("\n"));
  write("assets/lizard-web/recv-worker.mjs", 'import { phy } from "../liblizard/sim/phy.mjs";');
  write("assets/lizard-web/bank.mjs", "export const b = 1;");
  write("assets/liblizard/sim/phy.mjs", 'import create from "../build/ob.mjs"; export const phy = create;');
  write("assets/liblizard/sim/fob.mjs", "export const fob = 1;");
  write("assets/liblizard/build/ob.mjs", 'var f = "ob.wasm"; export default () => new URL("ob.wasm", import.meta.url);');
  write("assets/liblizard/build/ob.wasm", "WASM");
  write("assets/liblizard/gpu/cnn/weights.safetensors", "NET");
  let threw = null;
  try { hashTree(tmp); } catch (e) { threw = e; }
  check("the tree hashes", !threw, threw?.message);
  const read = (rel) => readFileSync(join(tmp, rel), "utf8");
  const find = (dir, prefix) => readdirSync(join(tmp, dir)).find((f) => f.startsWith(prefix) && HASHED_RE.test(f));
  const W = "assets/lizard-web", L = "assets/liblizard";
  const recv = find(W, "recv."), phy = find(`${L}/sim`, "phy."), fob = find(`${L}/sim`, "fob."), ob = find(`${L}/build`, "ob.") && readdirSync(join(tmp, `${L}/build`)).find((f) => /^ob\.[0-9a-f]{10}\.mjs$/.test(f));
  const wasm = readdirSync(join(tmp, `${L}/build`)).find((f) => /^ob\.[0-9a-f]{10}\.wasm$/.test(f)), net = find(`${L}/gpu/cnn`, "weights."), worker = find(W, "recv-worker."), bank = find(W, "bank."), css = find(W, "ui."), icon = find(W, "icon.");
  check("every asset renamed with its hash", [recv, phy, fob, ob, wasm, net, worker, bank, css, icon].every(Boolean));
  const r = recv ? read(`${W}/${recv}`) : "";
  check("an import specifier follows", r.includes(`"../liblizard/sim/${phy}"`), r);
  check("fob.mjs is not mangled by ob.mjs's rewrite", r.includes(`"../liblizard/sim/${fob}"`), r);
  check("a worker's new URL follows", r.includes(`new URL("./${worker}"`), r);
  check("a library-relative path follows", r.includes(`"gpu/cnn/${net}"`), r);
  check("a backtick literal follows", r.includes(`\`./${bank}\``), r);
  check("a page's name is left alone", r.includes('"send.html"'), r);
  const o = ob ? read(`${L}/build/${ob}`) : "";
  check("the glue's bare name follows, both times", (o.match(new RegExp(wasm?.replace(/\./g, "\\.") ?? "x", "g")) ?? []).length === 2, o);
  check("phy names the hashed glue", phy && read(`${L}/sim/${phy}`).includes(`"../build/${ob}"`));
  const html = read("index.html");
  check("the page keeps its name, its references follow", html.includes(`${W}/${css}`) && html.includes(`${W}/${recv}`) && html.includes(`${W}/${icon}`) && html.includes('href="send.html"') && html.includes("https://example.test/x.mjs"), html);
  check("the stylesheet's url() follows", css && read(`${W}/${css}`).includes(`url("${icon}")`));
  check("the manifest's icon follows", read("manifest.webmanifest").includes(`./${W}/${icon}`));
  const problems = verifyTree(tmp);
  check("verify finds nothing wrong", problems.length === 0, problems.join("; "));
  // ai: a file's name in a message or a shader's comment is no reference
  write(`${W}/note.0000000000.mjs`, "throw new Error(`tables disagree with sim/phy.mjs, ${x}`); const w = `// see recv-worker.mjs (phy.mjs)\n`;");
  const quiet = verifyTree(tmp).filter((p) => !/stale/.test(p));
  check("a name in a message or a comment passes", quiet.length === 0, quiet.join("; "));
  rmSync(join(tmp, `${W}/note.0000000000.mjs`));
  // ai: the negatives: a path put together at run time, a literal the rewrite missed, a file whose bytes no longer match
  // ai: its name, an unhashed file
  write(`${W}/late.mjs`, 'const u = new URL(`./recv-worker.mjs${q}`, import.meta.url); const v = "../nowhere/phy.mjs";');
  writeFileSync(join(tmp, W, bank), "export const b = 2;");
  const bad = verifyTree(tmp);
  check("verify catches a name put together at run time", bad.some((p) => /run time/.test(p) && /recv-worker\.mjs/.test(p)), bad.join("; "));
  check("verify catches a literal the rewrite missed", bad.some((p) => /unrewritten/.test(p) && /nowhere\/phy\.mjs/.test(p)), bad.join("; "));
  check("verify catches a stale hash", bad.some((p) => /stale/.test(p) && p.includes(bank)), bad.join("; "));
  check("verify catches an unhashed name", bad.some((p) => /not content-addressed: assets\/lizard-web\/late\.mjs/.test(p)), bad.join("; "));
  rmSync(tmp, { recursive: true, force: true });
  if (failures) { console.error(`${failures} self-test failure${failures === 1 ? "" : "s"}.`); process.exit(1); }
  console.log("hash self-test passed.");
}

if ((() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })()) {
  const argv = process.argv.slice(2);
  if (argv.includes("--self-test")) selfTest();
  else if (argv.includes("--verify")) {
    const dir = argv.find((a) => !a.startsWith("--"));
    if (!dir || !existsSync(dir)) { console.error("usage: node lizard-web/pwa/hash.mjs --verify <dir>"); process.exit(2); }
    const problems = verifyTree(dir);
    for (const p of problems) console.error(`  ${p}`);
    console.log(problems.length ? `verify: ${problems.length} problem(s)` : `verify: ${walk(dir).length} files, every name a true hash, no reference left unrewritten`);
    process.exit(problems.length ? 1 : 0);
  } else { console.error("usage: node lizard-web/pwa/hash.mjs --self-test | --verify <dir> (build.mjs runs the stage itself)"); process.exit(2); }
}
