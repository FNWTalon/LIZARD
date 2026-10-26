// The file's fountain, one per transfer, fed the new blocks of every decode worker. Off the page's thread because a
// solve takes long enough to drop camera frames.
//
// A LIZARD file is a chunked transfer (sim/xfer.mjs): the header and manifest come in the light, each chunk is its own
// Wirehair fountain, and each is checked against the file's BLAKE3 root as it completes. Its blocks wait on disk,
// in the origin private file system (OPFS), where the browser has one: interleaved, every chunk is in flight until the
// ai: end of the first lap, and the file would otherwise be held in memory twice over. The header comes from the light
// ai: alone, and the page's config names only the store (?store=memory).
// ai: A blind receiver hands over every block it verifies, the test stream's too, and cannot tell a test block from a
// ai: file's by its id. So a data block goes to the transfer only while that transfer's header is in the light: read,
// ai: and read again within LIVE_BLOCKS data blocks (the sender puts one in every 128 at most, sim/xfer.mjs). Until a
// ai: header is read, and once it has gone quiet, the newest HOLD_BLOCKS wait here (0.5 MB) and go to whichever
// ai: transfer's header comes next; a transfer with no header in sight stores nothing.
import { init as initWh, Decoder } from "../liblizard/sim/fountain.mjs";
import { init as initZstd, Z } from "../liblizard/sim/zstd.mjs";
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { XferReceiver, MemoryStore, PAYLOAD, ID_BYTES, ID_HEADER, isControlId } from "../liblizard/sim/xfer.mjs";

let rx = null, store = null, chain = Promise.resolve(), lastPost = 0, posted = null;
const REC = ID_BYTES + PAYLOAD, DIR = "lizard-xfer", LIVE_BLOCKS = 4096, HOLD_BLOCKS = 1024;
let sinceHeader = Infinity, held = [], heldAt = 0;

// Blocks and the file on disk, through sync access handles (a dedicated worker's only).
// ai: A folder a transfer under lizard-xfer, its id the root's first 16 hex digits (2026-10-22: kept across page loads,
// ai: liblizard/core/rx/store.h's file store in OPFS): header, manifest (index and payload a block), c<k> (a chunk's
// ai: stored blocks, removed once the chunk is verified), out (the file, written at each chunk's offset), verified (a
// ai: chunk index each) and meta.json, which Home lists. A header of another root parks the transfer in hand (its
// ai: handles closed, its folder left); its own header read again resumes it, in this page or a later one. Opening the
// ai: store keeps the folders that hold a header and removes everything else (a finished file nobody filed, the layout
// ai: before). A finished transfer leaves its file alone in its folder, which fileAway moves out and removes.
const u32 = (b, o) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
const put32 = (b, o, v) => { b[o] = v & 255; b[o + 1] = (v >>> 8) & 255; b[o + 2] = (v >>> 16) & 255; b[o + 3] = v >>> 24; };
async function readAll(dir, name) {
  try { return new Uint8Array(await (await (await dir.getFileHandle(name)).getFile()).arrayBuffer()); } catch { return null; }
}
async function hasFile(dir, name) { try { await dir.getFileHandle(name); return true; } catch { return false; } }
class OpfsStore {
  static async open() {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    // Proved here, not assumed: a browser can have the directory and refuse the handle (a private window, an old engine).
    const probe = await (await dir.getFileHandle("probe", { create: true })).createSyncAccessHandle();
    probe.close();
    await dir.removeEntry("probe");
    for await (const [name, h] of dir.entries())
      if (h.kind !== "directory" || !(await hasFile(h, "header"))) await dir.removeEntry(name, { recursive: true }).catch(() => {});
    return new OpfsStore(dir);
  }
  constructor(dir) { this.kind = "opfs"; this.keeps = true; this.dir = dir; this.cur = null; this.id = ""; this.h = new Map(); this.out = null; this.man = null; this.ver = null; this.rec = new Uint8Array(REC); }
  async handle(c) {
    let e = this.h.get(c);
    if (!e) { e = { h: await (await this.cur.getFileHandle(`c${c}`, { create: true })).createSyncAccessHandle(), size: 0 }; e.h.truncate(0); this.h.set(c, e); }
    return e;
  }
  async sync(name) { const h = await (await this.cur.getFileHandle(name, { create: true })).createSyncAccessHandle(); return { h, size: h.getSize() }; }
  closeAll() {
    for (const e of this.h.values()) e.h.close();
    this.h.clear();
    for (const k of ["out", "man", "ver"]) { if (this[k]) { this[k].h.flush(); this[k].h.close(); } this[k] = null; }
    this.cur = null; this.id = "";
  }
  async open(id, header, length) {
    this.closeAll();
    let d = await this.dir.getDirectoryHandle(id, { create: true });
    const had = await readAll(d, "header");
    const same = had && had.length === header.length && had.every((v, i) => v === header[i]) && (await hasFile(d, "out"));
    if (!same) {
      await this.dir.removeEntry(id, { recursive: true }).catch(() => {});
      d = await this.dir.getDirectoryHandle(id, { create: true });
      this.cur = d; this.id = id;
      this.out = await this.sync("out"); this.out.h.truncate(length);
      const hh = await this.sync("header"); hh.h.write(header, { at: 0 }); hh.h.flush(); hh.h.close();   // ai: last, so a folder with one holds its output
      return { found: false };
    }
    this.cur = d; this.id = id;
    this.out = await this.sync("out");
    const m = (await readAll(d, "manifest")) ?? new Uint8Array(0), v = (await readAll(d, "verified")) ?? new Uint8Array(0);
    const manifest = [], verified = [], chunks = [];
    for (let o = 0; o + REC <= m.length; o += REC) manifest.push([u32(m, o), m.slice(o + ID_BYTES, o + REC)]);
    for (let o = 0; o + 4 <= v.length; o += 4) verified.push(u32(v, o));
    for await (const [name, h] of d.entries()) {
      if (h.kind !== "file" || !/^c\d+$/.test(name)) continue;
      const c = +name.slice(1), sh = await h.createSyncAccessHandle(), size = sh.getSize() - (sh.getSize() % REC);
      sh.truncate(size);   // ai: a record cut short by a closed page goes, so the next lands on a record's boundary
      this.h.set(c, { h: sh, size });
      chunks.push(c);
    }
    return { found: true, manifest, verified, chunks };
  }
  // ai: the transfer in hand let go whole, its folder too
  async clear() { const id = this.id; this.closeAll(); if (id) await this.dir.removeEntry(id, { recursive: true }).catch(() => {}); }
  async park() { this.closeAll(); }
  async forget(id) { if (id === this.id) return this.clear(); await this.dir.removeEntry(id, { recursive: true }).catch(() => {}); }
  async append(c, sym, bytes) {
    const e = await this.handle(c), r = this.rec;
    put32(r, 0, sym);
    r.set(bytes.subarray(0, PAYLOAD), ID_BYTES);
    e.h.write(r, { at: e.size }); e.size += REC;
  }
  async read(c) { const e = this.h.get(c), b = new Uint8Array(e?.size ?? 0); if (e) e.h.read(b, { at: 0 }); return b; }
  async drop(c) { const e = this.h.get(c); if (!e) return; e.h.close(); this.h.delete(c); await this.cur.removeEntry(`c${c}`).catch(() => {}); }
  async truncate(c, n) { const e = this.h.get(c); if (!e || e.size <= n * REC) return; e.h.truncate(n * REC); e.size = n * REC; }
  async writeOut(off, bytes) { this.out.h.write(bytes, { at: off }); }
  async noteManifest(m, bytes) {
    this.man ??= await this.sync("manifest");
    const r = new Uint8Array(REC); put32(r, 0, m); r.set(bytes.subarray(0, PAYLOAD), ID_BYTES);
    this.man.h.write(r, { at: this.man.size }); this.man.size += REC;
  }
  // ai: the output flushed before the chunk is noted: a chunk noted is a chunk on the disk
  async noteVerified(c) {
    this.out.h.flush();
    this.ver ??= await this.sync("verified");
    const r = new Uint8Array(4); put32(r, 0, c);
    this.ver.h.write(r, { at: this.ver.size }); this.ver.size += 4; this.ver.h.flush();
  }
  async noteMeta(json) {
    if (!this.cur) return;
    const h = await (await this.cur.getFileHandle("meta.json", { create: true })).createSyncAccessHandle(), b = ENC.encode(json);
    h.truncate(0); h.write(b, { at: 0 }); h.flush(); h.close();
  }
  // ai: the file whole: what was kept to resume goes, the file left in the folder for fileAway to move
  async finish() {
    const id = this.id, d = this.cur;
    this.out.h.flush(); this.out.h.close(); this.out = null;
    this.closeAll();
    for (const f of ["header", "manifest", "verified", "meta.json"]) await d.removeEntry(f).catch(() => {});
    return { opfs: [DIR, id, "out"] };
  }
  held() { return 0; }
}

async function chunked(c) {
  const [M] = await Promise.all([initOb(), initWh(), initZstd()]);
  if (c.store === "memory" && store?.kind !== "memory") { store = new MemoryStore(); store.why = "asked for (?store=memory)"; rx = null; }
  if (!store) try { store = await OpfsStore.open(); } catch (e) { store = new MemoryStore(); store.why = String(e?.message ?? e); }
  rx ??= new XferReceiver({ M, Decoder, store, Z });   // ai: Z decompresses a chunk sent as a zstd frame (2026-10-05)
  await rx.reset(true);   // ai: a transfer in hand parked, as a header of another file would (2026-10-22; let go before)
  sinceHeader = Infinity; held = []; heldAt = 0;
}

// ai: The held blocks, oldest first, into the transfer whose header was just read.
async function release() {
  const ring = held.length < HOLD_BLOCKS ? held : [...held.slice(heldAt), ...held.slice(0, heldAt)];
  held = []; heldAt = 0;
  for (const [id, bytes] of ring) await rx.add(id, bytes);
}
async function take(id, bytes) {
  if (isControlId(id)) {
    await rx.add(id, bytes);
    if (id === ID_HEADER && rx.hdr) { sinceHeader = 0; if (held.length) await release(); }
    return;
  }
  if (rx.hdr && sinceHeader++ < LIVE_BLOCKS) return rx.add(id, bytes);
  const rec = [id, bytes.slice()];
  if (held.length < HOLD_BLOCKS) held.push(rec); else { held[heldAt] = rec; heldAt = (heldAt + 1) % HOLD_BLOCKS; }
}

// What the page shows and reports: the receiver's progress, and what this worker holds, since holding little is the
// point of the store.
async function report(force) {
  const now = performance.now();
  if (!force && now - lastPost < 250) return;
  lastPost = now;
  const [M, W] = await Promise.all([initOb(), initWh()]);
  postMessage({ ...rx.progress(), type: "xfer", store: store.kind, storeWhy: store.why, mem: { wirehairHeap: W.HEAPU8.length, codecHeap: M.HEAPU8.length, storeBytes: store.held() } });
}

// Once a transfer, the moment its last chunk verifies. A finished receiver still reads control blocks: a header with
// other bytes is the next transfer.
// ai: On disk, the finished file is filed into the received files first (fileAway), and the page told where it is.
async function offer() {
  await report(rx.done);
  if (rx.done && posted !== rx.result) {
    posted = rx.result;
    // ai: a filing that fails (a full disk) leaves the file where it was, offered from there as before 2026-10-01
    const lib = rx.result.opfs ? await fileAway(rx.result).catch((e) => { console.warn("fountain: the file was not kept:", e); return null; }) : null;
    postMessage({ ...rx.result, ...(lib ? { opfs: null, lib } : {}), type: "file", ok: true, chunked: true });
  }
}

// ai: The received files (2026-10-01, lizard-web/library.mjs, which reads them): lizard-files/<id>/data and its meta.json, the
// ai: id the root's first 16 hex digits, so the same file received again replaces its entry. The finished file leaves
// ai: its transfer's folder under lizard-xfer (OpfsStore) by a move where the browser has one, else a copy through sync
// ai: access handles, and the folder goes after it: nothing of the transfer is left behind.
const LIB = "lizard-files", ENC = new TextEncoder();
async function fileAway(r) {
  const path = r.opfs, name = path.at(-1), root = await navigator.storage.getDirectory();
  let from = root;
  for (const d of path.slice(0, -1)) from = await from.getDirectoryHandle(d);
  const lib = await root.getDirectoryHandle(LIB, { create: true }), id = r.root.slice(0, 16);
  await lib.removeEntry(id, { recursive: true }).catch(() => {});
  const d = await lib.getDirectoryHandle(id, { create: true }), src = await from.getFileHandle(name);
  let moved = false;
  if (typeof src.move === "function") try { await src.move(d, "data"); moved = true; } catch {}
  if (!moved) {
    const a = await src.createSyncAccessHandle(), b = await (await d.getFileHandle("data", { create: true })).createSyncAccessHandle(), buf = new Uint8Array(1 << 20);
    b.truncate(0);
    for (let at = 0, n; (n = a.read(buf, { at })) > 0; at += n) b.write(buf.subarray(0, n), { at });
    b.flush(); b.close(); a.close();
    await from.removeEntry(name);
  }
  // ai: the transfer's folder, empty now (OpfsStore.finish took the rest)
  if (path.length > 2) { let parent = root; for (const d of path.slice(0, -2)) parent = await parent.getDirectoryHandle(d); await parent.removeEntry(path.at(-2), { recursive: true }).catch(() => {}); }
  const meta = { name: r.name || "received.bin", type: r.mediaType || "", size: r.length, root: r.root, at: Date.now() };
  const m = await (await d.getFileHandle("meta.json", { create: true })).createSyncAccessHandle(), bytes = ENC.encode(JSON.stringify(meta));
  m.truncate(0); m.write(bytes, { at: 0 }); m.flush(); m.close();
  if (typeof BroadcastChannel === "function") { const bc = new BroadcastChannel(LIB); bc.postMessage("changed"); bc.close(); }
  return { ...meta, id };
}

async function blocks({ ids, bytes }) {
  if (!rx) return;
  const all = new Uint8Array(bytes);
  for (let k = 0; k < ids.length; k++) await take(ids[k], all.subarray(k * PAYLOAD, (k + 1) * PAYLOAD));
  return offer();
}

// ai: A config starts a new transfer: what was held goes. { type: "config", config: { store } }. A finished file was
// ai: filed in the received files (fileAway) and stays there; the pages delete those (lizard-web/library.mjs). The
// ai: "clear" message, the receiver's Receive again, went with that button on 2026-10-05.
async function handle(m) {
  if (m.type !== "config") return blocks(m);
  await chunked(m.config);
  await offer();
}
onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", message: String(e?.message ?? e) }));
};
// ai: Home's Delete of an unfinished transfer whose files this worker holds open (library.mjs forgetUnfinished):
// ai: forgotten in turn with the blocks, then the received files' channel told, so Home draws again.
if (typeof BroadcastChannel === "function") new BroadcastChannel("lizard-xfer").onmessage = ({ data }) => {
  if (typeof data?.forget !== "string") return;
  chain = chain.then(async () => { if (rx) await rx.forget(data.forget); const bc = new BroadcastChannel(LIB); bc.postMessage("changed"); bc.close(); }).catch(() => {});
};
