// The file's fountain, one per transfer, fed the new blocks of every decode worker. Off the page's thread because a
// solve takes long enough to drop camera frames.
//
// A Lizard file is a chunked transfer (sim/xfer.mjs): the header and manifest come in the light, each chunk is its own
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
import { init as initOb } from "../liblizard/sim/ob.mjs";
import { XferReceiver, MemoryStore, PAYLOAD, ID_BYTES, ID_HEADER, isControlId } from "../liblizard/sim/xfer.mjs";

let rx = null, store = null, chain = Promise.resolve(), lastPost = 0, posted = null;
const REC = ID_BYTES + PAYLOAD, DIR = "lizard-xfer", LIVE_BLOCKS = 4096, HOLD_BLOCKS = 1024;
let sinceHeader = Infinity, held = [], heldAt = 0;

// Blocks and the file on disk, through sync access handles (a dedicated worker's only). One file of stored blocks a
// chunk, removed once the chunk is verified, and one output file a transfer, written at each chunk's offset. The
// directory is emptied when this worker starts: one receiver an origin, and what a page load before this one left
// there is nobody's.
class OpfsStore {
  static async open() {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(DIR, { recursive: true }).catch(() => {});
    const dir = await root.getDirectoryHandle(DIR, { create: true });
    // Proved here, not assumed: a browser can have the directory and refuse the handle (a private window, an old engine).
    const probe = await (await dir.getFileHandle("probe", { create: true })).createSyncAccessHandle();
    probe.close();
    await dir.removeEntry("probe");
    return new OpfsStore(dir);
  }
  constructor(dir) { this.kind = "opfs"; this.dir = dir; this.h = new Map(); this.out = null; this.rec = new Uint8Array(REC); this.n = 0; }
  async handle(c) {
    let e = this.h.get(c);
    if (!e) { e = { h: await (await this.dir.getFileHandle(`c${c}`, { create: true })).createSyncAccessHandle(), size: 0 }; e.h.truncate(0); this.h.set(c, e); }
    return e;
  }
  // A finished file is left where it is, since the page's link reads it; only what an unfinished transfer held goes.
  async clear() {
    for (const c of [...this.h.keys()]) await this.drop(c);
    if (this.out && !this.out.done) { this.out.h.close(); await this.dir.removeEntry(this.out.name).catch(() => {}); }
    this.out = null;
  }
  async begin(length) { const name = `out${this.n++}`, h = await (await this.dir.getFileHandle(name, { create: true })).createSyncAccessHandle(); h.truncate(length); this.out = { name, h, done: false }; }
  async append(c, sym, bytes) {
    const e = await this.handle(c), r = this.rec;
    r[0] = sym & 255; r[1] = (sym >>> 8) & 255; r[2] = (sym >>> 16) & 255; r[3] = sym >>> 24;
    r.set(bytes.subarray(0, PAYLOAD), ID_BYTES);
    e.h.write(r, { at: e.size }); e.size += REC;
  }
  async read(c) { const e = this.h.get(c), b = new Uint8Array(e?.size ?? 0); if (e) e.h.read(b, { at: 0 }); return b; }
  async drop(c) { const e = this.h.get(c); if (!e) return; e.h.close(); this.h.delete(c); await this.dir.removeEntry(`c${c}`).catch(() => {}); }
  async writeOut(off, bytes) { this.out.h.write(bytes, { at: off }); }
  async readOut(off, len) { const b = new Uint8Array(len); this.out.h.read(b, { at: off }); return b; }
  async finish() { this.out.h.flush(); this.out.h.close(); this.out.done = true; return { opfs: [DIR, this.out.name] }; }
  // ai: Everything this store holds: the receiver's "Receive again" (Clear from 2026-09-29). A finished file is no longer
  // ai: here (offer files it away, 2026-10-01), unless filing it failed.
  async removeAll() {
    await this.clear();
    for await (const name of this.dir.keys()) await this.dir.removeEntry(name).catch(() => {});
  }
  held() { return 0; }
}

async function chunked(c) {
  const [M] = await Promise.all([initOb(), initWh()]);
  if (c.store === "memory" && store?.kind !== "memory") { store = new MemoryStore(); store.why = "asked for (?store=memory)"; rx = null; }
  if (!store) try { store = await OpfsStore.open(); } catch (e) { store = new MemoryStore(); store.why = String(e?.message ?? e); }
  rx ??= new XferReceiver({ M, Decoder, store });
  await rx.reset();
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
// ai: lizard-xfer (the transfer's scratch folder, emptied at this worker's start and by the page's "Receive again") by a
// ai: move where the browser has one, else a copy through sync access handles; either way nothing of it is left behind.
const LIB = "lizard-files", ENC = new TextEncoder();
async function fileAway(r) {
  const [dirName, name] = r.opfs, root = await navigator.storage.getDirectory(), from = await root.getDirectoryHandle(dirName);
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

// ai: A config starts a new transfer: what was held goes. { type: "config", config: { store } }. A clear does the same and
// ai: empties the scratch folder (the receiver's "Receive again", its Clear from 2026-09-29), so the same file still in
// ai: the light is received and offered again. A finished file was
// ai: filed in the received files (fileAway) and stays there; the pages delete those (lizard-web/library.mjs).
async function handle(m) {
  if (m.type !== "config" && m.type !== "clear") return blocks(m);
  await chunked(m.config);
  if (m.type === "clear") { await store.removeAll?.(); posted = null; return; }
  await offer();
}
onmessage = ({ data: m }) => {
  chain = chain.then(() => handle(m)).catch((e) => postMessage({ type: "error", message: String(e?.message ?? e) }));
};
