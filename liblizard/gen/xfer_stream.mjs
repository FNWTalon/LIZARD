// ai: The native transfer's check stream (liblizard/core/tools/xfer_check.cpp): a random multi-chunk file sent by the web's
// ai: sender (liblizard/sim/xfer.mjs XferSender over Wirehair, liblizard/build/wirehair.mjs) through a lossy channel, written as
// ai: flat 473-byte blocks (the id little-endian, then 469 payload bytes), and the web's receiving chain run on the same
// ai: stream as the reference: blockJudge (sim/phy.mjs) a frame of 40 blocks, the page's dedupe (lizard-web/recv.mjs take),
// ai: the fountain worker's live-header rule (lizard-web/fountain-worker.mjs take) and XferReceiver in memory.
//
//   node liblizard/gen/xfer_stream.mjs [outDir] [bytes]    outDir defaults to liblizard/build/xfer, bytes to 9,000,000
//
// ai: Writes stream.bin (the file's blocks as received), file.bin (the file), bad.bin (stream.bin with one block of chunk 1
// ai: wrong), test.bin (the SHAKE256 test stream, one block of frame 7 wrong), ref.json (what the JS chain made of each). The channel, seeded: the receiver joins 30% into
// ai: the first lap; each block is lost with p 0.2; a frame is read twice with p 0.08 (a camera's repeat); a frame's
// ai: blocks come in shuffled, and neighbouring frames swap with p 0.1 (decode workers finishing out of order).
import { mkdirSync, writeFileSync } from "node:fs";
const LIB = new URL("../", import.meta.url);
const imp = (p) => import(new URL(p, LIB).href);
const { XferSender, XferReceiver, MemoryStore, PAYLOAD, ID_HEADER, isControlId, hex, fractionDone } = await imp("sim/xfer.mjs");
const { init: initWh, Encoder, Decoder } = await imp("sim/fountain.mjs");
const { init: initOb, streamBlock } = await imp("sim/ob.mjs");
const { blockJudge } = await imp("sim/phy.mjs");

const OUT = process.argv[2] ?? new URL("../build/xfer/", import.meta.url).pathname;
const LENGTH = Number(process.argv[3] ?? 9_000_000), B = 40, REC = 4 + PAYLOAD;
mkdirSync(OUT, { recursive: true });
const [M] = await Promise.all([initOb(), initWh()]);

let s = 0x2545f491;
const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 2 ** 32; };
const file = new Uint8Array(LENGTH);
for (let i = 0; i < LENGTH; i++) file[i] = (rnd() * 256) | 0;

const tx = new XferSender(file, { name: "check-9MB.bin", type: "application/octet-stream", M, Encoder });
const lap = tx.info().lap, frames = Math.ceil((1.8 * lap) / (B - 1)), join = Math.floor((0.3 * lap) / (B - 1));
const sent = [];
for (let f = 0; f < frames; f++) {
  const ids = tx.frameIds(B), blocks = [];
  if (f < join) continue;
  for (const id of ids) { const b = new Uint8Array(REC); new DataView(b.buffer).setUint32(0, id, true); tx.block(id, b.subarray(4)); blocks.push(b); }
  sent.push(blocks);
}
tx.free();
for (let f = 0; f + 1 < sent.length; f++) if (rnd() < 0.1) [sent[f], sent[f + 1]] = [sent[f + 1], sent[f]];
const got = [];
for (const blocks of sent) {
  for (let r = rnd() < 0.08 ? 2 : 1; r > 0; r--) {
    const kept = blocks.filter(() => rnd() >= 0.2);
    for (let i = kept.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [kept[i], kept[j]] = [kept[j], kept[i]]; }
    got.push(...kept);
  }
}
const stream = new Uint8Array(got.length * REC);
got.forEach((b, i) => stream.set(b, i * REC));
writeFileSync(`${OUT}/stream.bin`, stream);
writeFileSync(`${OUT}/file.bin`, file);

// ai: The test stream: ids from 0 as phy.mjs frame(seq) numbers them, each block its id's SHAKE256; frame 7's block 3 one
// ai: byte off, as a block that passed its CRC wrongly would be.
const TEST_FRAMES = 60, test = new Uint8Array(TEST_FRAMES * B * REC);
for (let i = 0; i < TEST_FRAMES * B; i++) {
  new DataView(test.buffer).setUint32(i * REC, i, true);
  test.set(streamBlock(i, PAYLOAD), i * REC + 4);
}
test[(7 * B + 3) * REC + 4 + 100] ^= 0x40;
writeFileSync(`${OUT}/test.bin`, test);

// ai: The web's chain, in the order its pages run it.
async function receive(bytes) {
  const judge = blockJudge(PAYLOAD), rx = new XferReceiver({ M, Decoder, store: new MemoryStore() });
  let seenIds = new Set(), seenOld = new Set(), root = "";
  const seenHas = (id) => seenIds.has(id) || seenOld.has(id);
  const seenAdd = (id) => { seenIds.add(id); if (seenIds.size >= 1 << 16) { seenOld = seenIds; seenIds = new Set(); } };
  let sinceHeader = Infinity, held = [], heldAt = 0, forwarded = 0, doneAt = null;
  const release = async () => {
    const ring = held.length < 1024 ? held : [...held.slice(heldAt), ...held.slice(0, heldAt)];
    held = []; heldAt = 0;
    for (const [id, b] of ring) await rx.add(id, b);
  };
  const take = async (id, b) => {
    if (isControlId(id)) {
      await rx.add(id, b);
      if (id === ID_HEADER && rx.hdr) { sinceHeader = 0; if (held.length) await release(); }
      return;
    }
    if (rx.hdr && sinceHeader++ < 4096) return rx.add(id, b);
    const r = [id, b.slice()];
    if (held.length < 1024) held.push(r); else { held[heldAt] = r; heldAt = (heldAt + 1) % 1024; }
  };
  const t = { frames: 0, seen: 0, bad: 0, test: 0, judged: 0, fresh: 0, forwarded: 0 };
  const n = bytes.length / REC;
  for (let f = 0; f * B < n; f++) {
    const blocks = bytes.subarray(f * B * REC, Math.min(n, (f + 1) * B) * REC), ok = new Uint8Array(blocks.length / REC).fill(1);
    const v = judge.frame(ok, blocks);
    t.frames++; t.seen += v.seen; t.bad += v.bad; t.test += v.test ? 1 : 0; t.judged += v.judged;
    const keep = [];
    for (const g of v.got) { if (isControlId(g.id)) keep.push(g); else if (!seenHas(g.id)) { seenAdd(g.id); keep.push(g); t.fresh++; } }
    if (v.test) continue;
    for (const g of keep) {
      forwarded++;
      await take(g.id, g.bytes);
      if (rx.done && doneAt == null) doneAt = forwarded;
    }
    const p = rx.progress();
    if (p.header?.root && p.header.root !== root) { if (root) { seenIds = new Set(); seenOld = new Set(); } root = p.header.root; }
  }
  t.forwarded = forwarded;
  const p = rx.progress();
  return { ...t, doneAt, done: rx.done, header: p.header ? { name: p.header.name, length: p.header.length, chunks: p.header.chunks, root: p.header.root } : null,
    verified: p.verified ?? 0, rejected: p.rejected ?? 0, manifestRejected: p.manifestRejected ?? 0, refused: p.refused, over: p.over ?? 0, fraction: fractionDone(p),
    exact: rx.done ? Buffer.compare(Buffer.from(rx.result.bytes), Buffer.from(file)) === 0 : false, rootOk: rx.done ? rx.result.root === hex(tx.root) : false };
}
// ai: The same stream with one data block of chunk 1 wrong (a block that passed its CRC wrongly): the chunk decodes to
// ai: other bytes, is rejected and its symbols banned, and both chains must agree on everything after.
const bad = stream.slice();
for (let i = 0; i < got.length; i++) if (new DataView(bad.buffer).getUint32(i * REC, true) >>> 18 === 1) { bad[i * REC + 4 + 200] ^= 0x01; break; }
writeFileSync(`${OUT}/bad.bin`, bad);
const ref = { blocks: got.length, lap, join, sentFrames: sent.length, root: hex(tx.root), file: await receive(stream), bad: await receive(bad), test: await receive(test) };
writeFileSync(`${OUT}/ref.json`, JSON.stringify(ref, null, 1));
console.log(JSON.stringify(ref));
