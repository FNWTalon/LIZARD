// ai: Comments out of the installed app's copy of a file (lizard-web/pwa/build.mjs), the source keeping them: another
// ai: project's comment stripper, ported to a module for .mjs. Not a
// ai: minifier: whitespace, indentation, names, string contents and statement order are untouched; a line that held
// ai: only a comment goes, a trailing comment leaves its code. A bang comment (/*! ... */) stays, as a licence's does.
// ai:   strip(text, ext)                      ext ".mjs", ".js", ".css", ".html" or ".svg"; throws StripError on a
// ai:                                         construct it cannot scan (the build then keeps the file whole and fails)
// ai:   node lizard-web/pwa/strip.mjs --self-test   the tokenizer fixtures (the ported stripper's, and Lizard's own hazards)
// ai: The one judgement it makes is whether a "/" opens a regex literal (regexAllowedAfter): wrong there, it would desync
// ai: and eat live code, which is why build.mjs parses every stripped module again (node --check).

import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

export class StripError extends Error {}

// ai: U+0000 appears in no source file here, so it can mark "a comment stood here" for the line pass.
const SENTINEL = "\u0000";

const isBang = (span) => span.startsWith("/*!");

// ai: a span's newlines kept, so the lines round it do not merge, and every line it covered marked
function sentinelFor(span) {
  let lines = 1;
  for (let i = 0; i < span.length; i++) if (span[i] === "\n") lines++;
  return new Array(lines).fill(SENTINEL).join("\n");
}

// ai: the lines that held only a comment dropped; the ones that held code too, tidied
function applyLinePass(text) {
  if (!text.includes(SENTINEL)) return text;
  const out = [];
  for (const line of text.split("\n")) {
    if (!line.includes(SENTINEL)) { out.push(line); continue; }
    const bare = line.split(SENTINEL).join("");
    if (bare.trim() === "") continue;
    out.push(bare.replace(/\s+$/, ""));
  }
  return out.join("\n");
}

// --- JavaScript -------------------------------------------------------------

const REGEX_KEYWORDS = new Set(["return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else", "yield", "await"]);

// ai: A "/" opens a regex literal only where a value may start; anywhere else it is division.
function regexAllowedAfter(emitted) {
  let i = emitted.length - 1;
  while (i >= 0 && /\s/.test(emitted[i])) i--;
  if (i < 0) return true;
  const ch = emitted[i];
  if ("(,=:[!&|?{};+-*%~^<>".includes(ch)) return true;
  if (/[A-Za-z0-9_$]/.test(ch)) {
    let j = i;
    while (j >= 0 && /[A-Za-z0-9_$]/.test(emitted[j])) j--;
    return REGEX_KEYWORDS.has(emitted.slice(j + 1, i + 1));
  }
  return false;
}

function scanQuoted(src, start, quote) {
  let i = start + 1;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "\\") { i += 2; continue; }
    if (ch === quote) return i + 1;
    if (ch === "\n") throw new StripError("unterminated string literal");
    i++;
  }
  throw new StripError("unterminated string literal");
}

// ai: just past the closing delimiter and its flags, or -1 when this "/" was not a regex after all
function scanRegex(src, start) {
  let i = start + 1, inClass = false;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "\\") { i += 2; continue; }
    if (ch === "\n") return -1;
    if (inClass) { if (ch === "]") inClass = false; }
    else if (ch === "[") inClass = true;
    else if (ch === "/") { i++; while (i < src.length && /[a-z]/.test(src[i])) i++; return i; }
    i++;
  }
  return -1;
}

function stripJs(src) {
  let out = "", i = 0;
  const n = src.length;
  // ai: a frame an open template literal; "${" inside one counts braces, so a nested template or a regex in an
  // ai: interpolation is tracked rather than guessed at
  const templates = [];
  const comment = (span) => { out += isBang(span) ? span : sentinelFor(span); };
  while (i < n) {
    const ch = src[i], next = src[i + 1];
    // ai: template text first: inside it a "/" is literal, a quote opens nothing, a backtick closes
    if (templates.length && templates[templates.length - 1] === 0) {
      if (ch === "\\") { out += src.slice(i, i + 2); i += 2; continue; }
      if (ch === "$" && next === "{") { templates[templates.length - 1] = 1; out += "${"; i += 2; continue; }
      if (ch === "`") { templates.pop(); out += ch; i++; continue; }
      out += ch; i++; continue;
    }
    if (ch === "/" && next === "/") { let j = i + 2; while (j < n && src[j] !== "\n") j++; comment(src.slice(i, j)); i = j; continue; }
    if (ch === "/" && next === "*") {
      const end = src.indexOf("*/", i + 2);
      if (end === -1) throw new StripError("unterminated block comment");
      comment(src.slice(i, end + 2)); i = end + 2; continue;
    }
    if (ch === '"' || ch === "'") { const j = scanQuoted(src, i, ch); out += src.slice(i, j); i = j; continue; }
    if (ch === "`") { templates.push(0); out += ch; i++; continue; }
    if (templates.length && ch === "{") { templates[templates.length - 1]++; out += ch; i++; continue; }
    if (templates.length && ch === "}") { templates[templates.length - 1]--; out += ch; i++; continue; }
    if (ch === "/" && regexAllowedAfter(out)) {
      const j = scanRegex(src, i);
      if (j !== -1) { out += src.slice(i, j); i = j; continue; }
    }
    out += ch; i++;
  }
  if (templates.length) throw new StripError("unterminated template literal");
  return applyLinePass(out);
}

// --- CSS ----------------------------------------------------------------------

function scanCssQuoted(src, start, quote) {
  let i = start + 1;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "\\") { i += 2; continue; }
    if (ch === quote) return i + 1;
    i++;
  }
  throw new StripError("unterminated string literal");
}

function stripCss(src) {
  let out = "", i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === "/" && src[i + 1] === "*") {
      const end = src.indexOf("*/", i + 2);
      if (end === -1) throw new StripError("unterminated block comment");
      const span = src.slice(i, end + 2);
      out += isBang(span) ? span : sentinelFor(span);
      i = end + 2; continue;
    }
    if (ch === '"' || ch === "'") { const j = scanCssQuoted(src, i, ch); out += src.slice(i, j); i = j; continue; }
    out += ch; i++;
  }
  return applyLinePass(out);
}

// --- HTML (and SVG) -------------------------------------------------------------

const JS_TYPES = new Set(["", "text/javascript", "application/javascript", "module"]);

function startsWithTag(src, i, tag) {
  if (src[i] !== "<" || src.slice(i + 1, i + 1 + tag.length).toLowerCase() !== tag) return false;
  const after = src[i + 1 + tag.length];
  return after === ">" || after === " " || after === "\t" || after === "\n" || after === "\r";
}

function matchEmbedded(src, i) {
  for (const [tag, kind] of [["script", "js"], ["style", "css"]]) {
    if (!startsWithTag(src, i, tag)) continue;
    const openEnd = src.indexOf(">", i);
    if (openEnd === -1) return null;
    const attrs = src.slice(i + tag.length + 1, openEnd);
    if (/\/$/.test(attrs.trim())) return null;
    if (kind === "js") {
      const type = /type\s*=\s*["']?([^"'\s>]*)/i.exec(attrs);
      if (type && !JS_TYPES.has(type[1].toLowerCase())) return null;
    }
    const closeStart = src.toLowerCase().indexOf(`</${tag}`, openEnd + 1);
    if (closeStart === -1) return null;
    const closeEnd = src.indexOf(">", closeStart);
    if (closeEnd === -1) return null;
    return { openEnd: openEnd + 1, closeStart, closeEnd: closeEnd + 1, kind };
  }
  return null;
}

function stripHtml(src) {
  let out = "", i = 0;
  while (i < src.length) {
    if (src.startsWith("<!--", i)) {
      if (src.startsWith("<!--[if ", i)) { out += "<!--"; i += 4; continue; }
      const end = src.indexOf("-->", i + 4);
      if (end === -1) throw new StripError("unterminated HTML comment");
      out += sentinelFor(src.slice(i, end + 3)); i = end + 3; continue;
    }
    const e = matchEmbedded(src, i);
    if (e) {
      out += src.slice(i, e.openEnd);
      const inner = src.slice(e.openEnd, e.closeStart);
      out += e.kind === "js" ? stripJs(inner) : stripCss(inner);
      out += src.slice(e.closeStart, e.closeEnd);
      i = e.closeEnd; continue;
    }
    out += src[i]; i++;
  }
  return applyLinePass(out);
}

export function strip(text, ext) {
  switch (ext) {
    case ".mjs": case ".js": return stripJs(text);
    case ".css": return stripCss(text);
    case ".html": case ".svg": return stripHtml(text);
    default: return text;
  }
}

// --- self-test --------------------------------------------------------------------

// ai: The ported stripper's fixtures, then Lizard's: a WGSL comment inside a template literal is the shader's, not the module's;
// ai: a URL in a string; a division after a closing paren and one after an index; a regex after "return".
const FIXTURES = [
  ['const re = /[\\\\/:*?"<>|]/g; // gone', 'const re = /[\\\\/:*?"<>|]/g;'],
  ["const s = /(['\"])((?:\\\\.|(?!\\1)[^\\\\])*)\\1/g; // gone", "const s = /(['\"])((?:\\\\.|(?!\\1)[^\\\\])*)\\1/g;"],
  ['x.replace(/\\\\/g, "/"); // gone', 'x.replace(/\\\\/g, "/");'],
  ['x.replace(/^\\/+/, ""); // gone', 'x.replace(/^\\/+/, "");'],
  ["const t = `a${/* gone */ b}c`;", "const t = `a${ b}c`;"],
  ["const t = `a${`nested ${x}`}b`; // gone", "const t = `a${`nested ${x}`}b`;"],
  ['const u = "http://example.com"; // gone', 'const u = "http://example.com";'],
  ["const q = a / b / c; // gone", "const q = a / b / c;"],
  ["/*! keep */\nconst a = 1;", "/*! keep */\nconst a = 1;"],
  ["const a = 1;\n// gone\nconst b = 2;", "const a = 1;\nconst b = 2;"],
  ["a();\n/* gone\n   also gone */\nb();", "a();\nb();"],
  ["const a = 1; /* gone */ const b = 2;", "const a = 1;  const b = 2;"],
  ["const wgsl = `\n  // the shader's, kept\n  let x = 1u;\n`; // gone", "const wgsl = `\n  // the shader's, kept\n  let x = 1u;\n`;"],
  ["const r = (a + b) / 2 / n; // gone", "const r = (a + b) / 2 / n;"],
  ["const r = v[0] / 2; // gone", "const r = v[0] / 2;"],
  ["function f(s) { return /^\\d+$/.test(s); } // gone", "function f(s) { return /^\\d+$/.test(s); }"],
  ['import { a } from "../sim/phy.mjs"; // ai: gone', 'import { a } from "../sim/phy.mjs";'],
];
const HTML_FIXTURES = [
  ["<p>a</p>\n<!-- gone -->\n<p>b</p>", "<p>a</p>\n<p>b</p>"],
  ["<!DOCTYPE html>\n<p>a</p>", "<!DOCTYPE html>\n<p>a</p>"],
  ["<script>var a = 1; // gone\n</script>", "<script>var a = 1;\n</script>"],
  ['<script type="application/json">{"//": 1}</script>', '<script type="application/json">{"//": 1}</script>'],
  ["<style>a { color: red } /* gone */</style>", "<style>a { color: red }</style>"],
];
const THROWS = ["/* unterminated", "const s = 'unterminated"];

function selfTest() {
  let failures = 0;
  const check = (label, actual, expected) => {
    if (actual === expected) return;
    failures++;
    console.error(`FAIL ${label}\n  expected ${JSON.stringify(expected)}\n  actual   ${JSON.stringify(actual)}`);
  };
  for (const [src, want] of FIXTURES) check("js", stripJs(src), want);
  for (const [src, want] of HTML_FIXTURES) check("html", stripHtml(src), want);
  check("css", stripCss("a { color: red } /* gone */"), "a { color: red }");
  check("css bang", stripCss("/*! keep */\na {}"), "/*! keep */\na {}");
  check("svg", stripHtml('<svg>\n  <!-- gone -->\n  <rect/>\n</svg>'), "<svg>\n  <rect/>\n</svg>");
  for (const src of THROWS) {
    let threw = false;
    try { stripJs(src); } catch (e) { threw = e instanceof StripError; }
    if (!threw) { failures++; console.error(`FAIL throws: ${JSON.stringify(src)} did not raise StripError`); }
  }
  if (failures) { console.error(`${failures} self-test failure${failures === 1 ? "" : "s"}.`); process.exit(1); }
  console.log("strip self-test passed.");
}

if ((() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })() && process.argv.includes("--self-test")) selfTest();
