// ai: Several files or a folder as one POSIX tar, the way the sender bundles them into one transfer (the transfer's zstd
// ai: compresses each 4 MiB chunk of the tar as of any file). The twin of liblizard/kotlin/Tar.kt's writer, the same
// ai: headers: ustar (the name, and the prefix field where a path is long), a pax extended header (typeflag x) carrying
// ai: `path` where a path fits neither field or is not plain ASCII, and `size` past ustar's 8 GiB; folders as their own
// ai: entries (typeflag 5, so an empty folder survives); files mode 644, folders 755, owner 0; two zero blocks at the
// ai: end. The archive is a list of parts a Blob takes by reference: a file's bytes are never copied or read here.
// ai: Pure and importing nothing (the installed app's rule), so node can check it against GNU tar.

export const TAR_TYPE = "application/x-tar";
const BLOCK = 512, MAX_OCTAL_11 = 2 ** 33;
const enc = new TextEncoder();
const pad = (n) => (BLOCK - (n % BLOCK)) % BLOCK;
const isAscii = (s) => /^[\x20-\x7e]*$/.test(s);

function header(name, size, mtime, type, mode, prefix = "") {
  const h = new Uint8Array(BLOCK);
  const put = (off, len, s) => h.set(enc.encode(s).subarray(0, len), off);
  put(0, 100, name);
  put(100, 8, `${mode}\0`);
  put(108, 8, "0000000\0");
  put(116, 8, "0000000\0");
  put(124, 12, size.toString(8).padStart(11, "0") + "\0");
  put(136, 12, Math.max(0, Math.min(MAX_OCTAL_11 - 1, Math.floor(mtime))).toString(8).padStart(11, "0") + "\0");
  put(148, 8, "        ");                      // ai: the checksum's field is spaces while the sum is taken
  put(156, 1, type);
  put(257, 8, "ustar\x0000");
  put(345, 155, prefix);
  let sum = 0;
  for (const b of h) sum += b;
  put(148, 8, sum.toString(8).padStart(6, "0") + "\0 ");
  return h;
}

// ai: ustar's split of a plain ASCII path: the name field holds up to 100 bytes, the prefix up to 155 joined by a '/'.
function splitUstar(path) {
  if (path.length <= 100) return ["", path];
  for (let i = path.lastIndexOf("/", 155); i > 0; i = path.lastIndexOf("/", i - 1)) {
    const name = path.slice(i + 1);
    if (name && name.length <= 100) return [path.slice(0, i), name];
  }
  return null;
}

// ai: The ustar name a pax header stands in front of: the path's last characters as ASCII, for a reader with no pax.
const asciiTail = (path, max = 99) => [...path].map((c) => (/[\x20-\x7e]/.test(c) ? c : "_")).join("").slice(-max).replace(/^\/+/, "");

// ai: One pax record, "<length> <key>=<value>\n", its length counting its own digits.
function paxRecord(key, value) {
  const rest = enc.encode(` ${key}=${value}\n`).length;
  let n = rest + 1;
  while (String(n).length + rest !== n) n = String(n).length + rest;
  return `${n} ${key}=${value}\n`;
}

function entryParts(path, size, mtime, type, mode) {
  const parts = [];
  const split = isAscii(path) ? splitUstar(path) : null;
  let pax = "";
  if (!split) pax += paxRecord("path", path);
  const big = size >= MAX_OCTAL_11;
  if (big) pax += paxRecord("size", String(size));
  if (pax) {
    const body = enc.encode(pax);
    parts.push(header("PaxHeaders/" + asciiTail(path, 80), body.length, mtime, "x", "0000644"), body);
    if (pad(body.length)) parts.push(new Uint8Array(pad(body.length)));
  }
  const [prefix, name] = split ?? ["", asciiTail(path)];
  parts.push(header(name, big ? 0 : size, mtime, type, mode, prefix));
  return parts;
}

/**
 * entries: [{ path, dir: true, mtime } | { path, blob, mtime }] with '/' between a path's parts, relative, folders
 * before what they hold; mtime in seconds. Returns the archive's parts in order, for `new Blob(parts)`.
 */
export function tarParts(entries) {
  const parts = [];
  for (const e of entries) {
    if (!e.path || e.path.startsWith("/")) throw new Error(`tar path must be relative: ${e.path}`);
    const mtime = e.mtime ?? Date.now() / 1000;
    if (e.dir) { parts.push(...entryParts(e.path.replace(/\/+$/, "") + "/", 0, mtime, "5", "0000755")); continue; }
    const size = e.blob.size;
    parts.push(...entryParts(e.path, size, mtime, "0", "0000644"), e.blob);
    if (pad(size)) parts.push(new Uint8Array(pad(size)));
  }
  parts.push(new Uint8Array(2 * BLOCK));         // ai: the end of the archive, two zero blocks
  return parts;
}

// ai: "a/b.txt" as "a/b (k).txt": a second file of one path (two picks of the same name from different folders).
function numbered(path, k) {
  const slash = path.lastIndexOf("/") + 1, dot = path.lastIndexOf(".");
  const cut = dot > slash ? dot : path.length;
  return `${path.slice(0, cut)} (${k})${path.slice(cut)}`;
}

// ai: The transfer's header holds a name of 255 bytes of UTF-8 (liblizard/sim/xfer.mjs nameBytes): the stem is cut on
// ai: a character so ".tar" always ends it.
function fitName(stem, suffix) {
  const chars = [...stem];
  while (chars.length && enc.encode(chars.join("") + suffix).length > 255) chars.pop();
  return chars.join("") + suffix;
}

/**
 * The payload a choice of files and folders makes. One file alone is itself; anything else is a tar: one folder
 * named after it, several items after the first and how many more. entries as tarParts', in any order, each file's
 * folders added where missing, a path given twice numbered. Returns { name, type, size, files, bundled, blob }.
 */
export function bundle(entries) {
  const used = new Set();
  const all = entries.map((e) => {
    if (e.dir) return e;
    let p = e.path;
    for (let k = 2; used.has(p); k++) p = numbered(e.path, k);
    used.add(p);
    return p === e.path ? e : { ...e, path: p };
  });
  const files = all.filter((e) => !e.dir);
  if (entries.length === 1 && files.length === 1 && !entries[0].path.includes("/"))
    return { name: entries[0].path, type: files[0].blob.type ?? "", size: files[0].blob.size, files: 1, bundled: false, blob: files[0].blob };
  const dirs = new Map();
  for (const e of entries) {
    const parts = e.path.split("/").filter(Boolean);
    for (let i = 1; i < (e.dir ? parts.length + 1 : parts.length); i++) {
      const p = parts.slice(0, i).join("/");
      if (!dirs.has(p)) dirs.set(p, { path: p, dir: true, mtime: e.dir && i === parts.length ? e.mtime : undefined });
    }
  }
  // ai: a folder no time was given for (a picked folder's, which a browser does not list) takes the newest of what it
  // ai: holds, an empty one the newest of the choice, so the same choice makes the same archive, and so the same transfer
  const newest = (under) => {
    let t = -Infinity;
    for (const f of files) if ((under === null || f.path.startsWith(under + "/")) && f.mtime !== undefined) t = Math.max(t, f.mtime);
    return t > -Infinity ? t : undefined;
  };
  for (const d of dirs.values()) if (d.mtime === undefined) d.mtime = newest(d.path) ?? newest(null);
  const ordered = [...dirs.values(), ...files].sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
  const tops = [...new Set(all.map((e) => e.path.split("/").filter(Boolean)[0]))];
  const name = tops.length === 1 ? fitName(tops[0], ".tar") : fitName(tops[0], ` and ${tops.length - 1} more.tar`);
  const blob = new Blob(tarParts(ordered), { type: TAR_TYPE });
  return { name, type: TAR_TYPE, size: blob.size, files: files.length, bundled: true, blob };
}
