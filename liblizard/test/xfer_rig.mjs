// The rig's chunked transfer (sim/xfer.mjs) without a camera: the sender's schedule, its blocks dropped the ways a
// camera drops them, and the receiver putting the file back from what is left, held to the file and to its BLAKE3
// (the vendored C through the wasm). Files of no bytes, one block, one chunk, several, 41 small chunks and 10 MB,
// random (sent as they are) and prose (each chunk sent as its zstd frame, so the chunks differ in blocks); loss at
// random, by ring, in bursts, a late join, a corrupted block that passes its CRC, a stale header from the network.
// Laps is the airtime from the join over one lap of the file's blocks as sent. About 8 s.
// ai: Since 2026-10-22, transfers kept across receivers (store keeps): a store whose folders outlive its receiver, as
// ai: OPFS outlives a page; part of a file into one receiver, which is dropped, the rest into the next; a file parked by
// ai: another's header and resumed by its own; a parked transfer forgotten.
//   node test/xfer_rig.mjs
import { init as initOb } from "../sim/ob.mjs";
import { init as initWh, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, Z } from "../sim/zstd.mjs";
import { XferSender, XferReceiver, MemoryStore, blake3, hex, isControlId, PAYLOAD } from "../sim/xfer.mjs";

const M = await initOb();
await initWh(); await initZstd();
let fails = 0, over = 0;
const rand = (n, seed) => { const a = new Uint8Array(n); let s = seed >>> 0 || 1; for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a[i] = s >>> 16; } return a; };
// Text-like bytes from a small vocabulary: zstd shrinks them by about half, each chunk by a different amount.
const prose = (n, seed) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const a = new Uint8Array(n); let s = seed >>> 0 || 1, k = 0; while (k < n) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; const t = w[(s >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) a[k++] = t.charCodeAt(i); } return a; };

// ring: the outer share of a frame's slots never read. burst: frames lost, 48 of every 240 (2 s of 10 at 24 fps).
// corrupt: chunk whose symbol 3 is sent with a flipped bit (in the first lap only, as symbols do not repeat).
// stale: a header of another file handed over first, as the network's. net: this file's header, as the network's.
// dark: every control block lost, so only the network's header can start the file.
async function run(label, { length, log2 = 22, B = 12, loss = 0, ring = 0, burst = 0, join = 0, corrupt = -1, stale = false, net = false, dark = false, text = false, maxLaps = 6, wantRejected = 0, wantSource = "light" }) {
  const file = text ? prose(length, length + 7) : rand(length, length + 7), xs = new XferSender(file, { name: "f.bin", type: "application/octet-stream", chunkLog2: log2, M, Encoder, Z });
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore(), Z });
  if (net) await rx.header(xs.header, "network");
  if (stale) { const other = new XferSender(rand(length + 1000, 3), { name: "old.bin", chunkLog2: log2, M, Encoder, Z }); await rx.header(other.header, "network"); other.free(); }
  let s = 12345;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const buf = new Uint8Array(PAYLOAD), lap = Math.max(1, xs.K.reduce((a, b) => a + b, 0)), limit = join + Math.ceil((maxLaps * lap) / B) + 50;
  let f = 0, sent = 0;
  for (; f < limit && !rx.done; f++) {
    const ids = xs.frameIds(B);
    if (f < join) continue;
    if (burst && f % 240 < 48) { sent += B; continue; }
    for (let k = 0; k < B; k++) {
      sent++;
      if (rnd() < loss || k >= B * (1 - ring) || (dark && isControlId(ids[k]))) continue;
      const b = xs.block(ids[k], buf).slice();
      if (!isControlId(ids[k]) && ids[k] >>> 18 === corrupt && (ids[k] & 0x3ffff) === 3) b[5] ^= 1;
      await rx.add(ids[k], b);
    }
  }
  const r = rx.result, p = rx.progress();
  const ok = !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && r.root === hex(xs.root) && p.rejected === wantRejected && r.source === wantSource;
  over += p.over ?? 0;
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${length} B, ${xs.chunks} chunk${xs.chunks === 1 ? "" : "s"} of 2^${log2}, ${xs.manifest.length} manifest; ${(sent / lap).toFixed(3)} laps, ${p.rejected ?? 0} rejected, ${p.over ?? 0} solve${p.over === 1 ? "" : "s"} past K`);
  xs.free();
}
await run("empty", { length: 0 });
await run("one block, unfountained", { length: PAYLOAD });
await run("two blocks", { length: PAYLOAD + 1 });
await run("small", { length: 50000 });
await run("small, 30% lost", { length: 50000, loss: 0.3 });
await run("3 chunks", { length: 3 * 2 ** 20 - 12345, log2: 20 });
await run("3 chunks, the outer 25% of rings lost", { length: 3 * 2 ** 20 - 12345, log2: 20, ring: 0.25 });
await run("3 chunks, joined 200 frames late", { length: 3 * 2 ** 20 - 12345, log2: 20, join: 200 });
await run("3 chunks, 2 s lost in every 10", { length: 3 * 2 ** 20 - 12345, log2: 20, burst: 1 });
await run("3 chunks, chunk 1 sent wrong in lap 1", { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1, wantRejected: 1 });
await run("3 chunks, a stale header from the network first", { length: 3 * 2 ** 20 - 12345, log2: 20, stale: true });
await run("small, the header from the network, none read from the light", { length: 50000, net: true, dark: true, wantSource: "network" });
await run("3 chunks, the network's header, then the light's", { length: 3 * 2 ** 20 - 12345, log2: 20, net: true });
await run("41 chunks of 1 KiB, 3 manifest blocks", { length: 40 * 1024 + 100, log2: 10 });
await run("12 chunks, as many as a frame's blocks, outer 25% lost", { length: 12 * 2 ** 16 - 5, log2: 16, ring: 0.25 });
await run("10 MB", { length: 10e6 });
await run("10 MB, 40% lost", { length: 10e6, loss: 0.4 });
// Prose: every chunk sent as its zstd frame, each a different number of blocks, so the schedule weighs them apart.
await run("small, prose", { length: 50000, text: true });
await run("one block of prose, unfountained once compressed", { length: 1200, text: true });
await run("3 chunks, prose", { length: 3 * 2 ** 20 - 12345, log2: 20, text: true });
await run("3 chunks, prose, 30% lost", { length: 3 * 2 ** 20 - 12345, log2: 20, loss: 0.3, text: true });
await run("3 chunks, prose, chunk 1 sent wrong in lap 1", { length: 3 * 2 ** 20 - 12345, log2: 20, corrupt: 1, wantRejected: 1, text: true });
await run("10 MB, prose, 40% lost", { length: 10e6, loss: 0.4, text: true });
await run("41 chunks of 1 KiB, prose, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100, text: true });
for (let k = 0; k < 20; k++) await run(`200 KB, 20% lost, seed ${k}`, { length: 200000 + k, log2: 16, loss: 0.2 });
// Joined after the first lap, so repair blocks only: Wirehair then wants a block past K now and again, and the chunk
// takes the rest as they come.
await run("41 chunks of 1 KiB, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100 });
for (let k = 0; k < 10; k++) await run(`400 KB in 16 KiB chunks, repair blocks only, seed ${k}`, { length: 400000 + 37 * k, log2: 14, join: 80 });

// ai: The fountain worker's OPFS store in memory (lizard-web/fountain-worker.mjs OpfsStore): a folder a transfer by its
// ai: id, in a map that outlives the receiver (the origin's OPFS outliving a page).
class KeepStore {
  constructor(folders) { this.kind = "keep"; this.keeps = true; this.folders = folders; this.cur = null; this.id = ""; }
  async open(id, header, length) {
    const f = this.folders.get(id);
    this.id = id;
    if (f && f.header.length === header.length && f.header.every((v, i) => v === header[i])) {
      this.cur = f;
      return { found: true, manifest: f.manifest.map(([m, b]) => [m, b.slice()]), verified: [...f.verified], chunks: [...f.recs.keys()] };
    }
    this.cur = { header: header.slice(), out: new Uint8Array(length), recs: new Map(), manifest: [], verified: [], meta: null };
    this.folders.set(id, this.cur);
    return { found: false };
  }
  async begin(length) { return this.open("t", new Uint8Array(0), length); }
  async append(c, sym, bytes) {
    const r = this.cur.recs.get(c) ?? [];
    const rec = new Uint8Array(4 + PAYLOAD);
    rec[0] = sym & 255; rec[1] = (sym >>> 8) & 255; rec[2] = (sym >>> 16) & 255; rec[3] = sym >>> 24;
    rec.set(bytes.subarray(0, PAYLOAD), 4);
    r.push(rec); this.cur.recs.set(c, r);
  }
  async read(c) { const r = this.cur.recs.get(c) ?? [], b = new Uint8Array(r.length * (4 + PAYLOAD)); r.forEach((x, i) => b.set(x, i * x.length)); return b; }
  async drop(c) { this.cur?.recs.delete(c); }
  async writeOut(off, bytes) { this.cur.out.set(bytes, off); }
  async readOut(off, len) { return this.cur.out.subarray(off, off + len); }
  async noteManifest(m, b) { this.cur.manifest.push([m, b.slice()]); }
  async noteVerified(c) { this.cur.verified.push(c); }
  async noteMeta(j) { this.cur.meta = JSON.parse(j); }
  async truncate(c, n) { const r = this.cur.recs.get(c); if (r) r.length = Math.min(r.length, n); }
  async finish() { const out = this.cur.out; this.folders.delete(this.id); return { bytes: out }; }
  async park() { this.cur = null; this.id = ""; }
  async clear() { if (this.id) this.folders.delete(this.id); this.cur = null; this.id = ""; }
  async forget(id) { if (id === this.id) return this.clear(); this.folders.delete(id); }
  held() { return 0; }
}
// ai: share of a lap of x's blocks into one receiver, then (other) a second file whole into the same receiver, then x
// ai: into a new receiver on the same folders (or the same one, other) until it is done
async function resume(label, { length, log2 = 20, B = 12, loss = 0, join = 0, share = 0.5, text = false, other = false, forget = false }) {
  const mk = (n, seed) => (text ? prose(n, seed) : rand(n, seed));
  const file = mk(length, length + 11), xs = new XferSender(file, { name: "x.bin", chunkLog2: log2, M, Encoder, Z });
  xs.name = "x.bin";
  const folders = new Map(), id = hex(xs.root).slice(0, 16), buf = new Uint8Array(PAYLOAD);
  let s = 4242;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s >>> 8) / 16777216; };
  const feed = async (rx, tx, frames) => {
    let f = 0;
    for (; f < frames && !(rx.done && rx.result?.name === tx.name); f++) { const ids = tx.frameIds(B); for (let k = 0; k < B; k++) if (rnd() >= loss) await rx.add(ids[k], tx.block(ids[k], buf).slice()); }
    return f;
  };
  for (let f = 0; f < join; f++) xs.frameIds(B);
  const lap = Math.max(1, xs.K.reduce((a, b) => a + b, 0)), first = Math.ceil((share * lap) / B);
  let rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders), Z });
  await feed(rx, xs, first);
  const part = rx.progress(), parked = () => folders.get(id);
  let ok = !rx.done && part.verified < part.header.chunks, extra = "";
  if (other) {
    const y = mk(length + 333, 99), ys = new XferSender(y, { name: "y.bin", chunkLog2: log2, M, Encoder, Z });
    ys.name = "y.bin";
    await feed(rx, ys, 50 * Math.ceil(ys.K.reduce((a, b) => a + b, 0) / B));
    ok &&= !!rx.result && rx.result.bytes.every((v, i) => v === y[i]) && rx.result.name === "y.bin";
    extra = `, y.bin whole between`;
    ys.free();
  } else rx = new XferReceiver({ M, Decoder, store: new KeepStore(folders), Z });   // ai: the page closed: the receiver dropped as it was
  ok &&= !!parked()?.meta && parked().meta.name === "x.bin";
  if (forget) {
    await rx.forget(hex(xs.root));
    ok &&= !folders.has(id);
    console.log(`${ok ? "ok  " : "FAIL"} ${label}: parked at ${Math.round(100 * part.verified / part.header.chunks)}% of chunks verified, then forgotten`);
    if (!ok) fails++;
    xs.free();
    return;
  }
  const more = await feed(rx, xs, 50 * Math.ceil(lap / B));
  const r = rx.result;
  ok &&= !!r && r.bytes.length === length && r.bytes.every((v, i) => v === file[i]) && r.root === hex(blake3(M, file)) && !folders.has(id);
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${xs.chunks} chunks, ${part.verified} verified and ${Math.round(100 * (first * B) / lap)}% of a lap in before${extra}; the rest in ${(more * B / lap).toFixed(3)} laps`);
  xs.free();
}
await resume("resumed by the next receiver", { length: 6 * 2 ** 20 - 99, log2: 20 });
await resume("resumed by the next receiver, prose, 30% lost", { length: 6 * 2 ** 20 - 99, log2: 20, loss: 0.3, text: true });
await resume("resumed by the next receiver, repair blocks only", { length: 400000, log2: 14, join: 80 });
await resume("parked by another file's header, resumed by its own", { length: 3 * 2 ** 20 - 12345, log2: 20, other: true });
await resume("a parked transfer forgotten", { length: 3 * 2 ** 20 - 12345, log2: 20, forget: true });
console.log(`${over} chunk solves needed more than K blocks and took them as they came`);
console.log(fails ? `${fails} FAILED` : "all ok");
process.exit(fails ? 1 : 0);
