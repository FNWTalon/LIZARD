// The rig's chunked transfer (sim/xfer.mjs) without a camera: the sender's schedule, its blocks dropped the ways a
// camera drops them, and the receiver putting the file back from what is left, held to the file and to its BLAKE3
// (the vendored C through the wasm). Files of no bytes, one block, one chunk, several, 41 small chunks and 10 MB;
// loss at random, by ring, in bursts, a late join, a corrupted block that passes its CRC, a stale header from the
// network. Laps is the airtime from the join over one lap of the file's source blocks. About 5 s.
//   node test/xfer_rig.mjs
import { init as initOb } from "../sim/ob.mjs";
import { init as initWh, Encoder, Decoder } from "../sim/fountain.mjs";
import { XferSender, XferReceiver, MemoryStore, blake3, hex, isControlId, PAYLOAD } from "../sim/xfer.mjs";

const M = await initOb();
await initWh();
let fails = 0, over = 0;
const rand = (n, seed) => { const a = new Uint8Array(n); let s = seed >>> 0 || 1; for (let i = 0; i < n; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; a[i] = s >>> 16; } return a; };

// ring: the outer share of a frame's slots never read. burst: frames lost, 48 of every 240 (2 s of 10 at 24 fps).
// corrupt: chunk whose symbol 3 is sent with a flipped bit (in the first lap only, as symbols do not repeat).
// stale: a header of another file handed over first, as the network's. net: this file's header, as the network's.
// dark: every control block lost, so only the network's header can start the file.
async function run(label, { length, log2 = 22, B = 12, loss = 0, ring = 0, burst = 0, join = 0, corrupt = -1, stale = false, net = false, dark = false, maxLaps = 6, wantRejected = 0, wantSource = "light" }) {
  const file = rand(length, length + 7), xs = new XferSender(file, { name: "f.bin", type: "application/octet-stream", chunkLog2: log2, M, Encoder });
  const rx = new XferReceiver({ M, Decoder, store: new MemoryStore() });
  if (net) await rx.header(xs.header, "network");
  if (stale) { const other = new XferSender(rand(length + 1000, 3), { name: "old.bin", chunkLog2: log2, M, Encoder }); await rx.header(other.header, "network"); other.free(); }
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
for (let k = 0; k < 20; k++) await run(`200 KB, 20% lost, seed ${k}`, { length: 200000 + k, log2: 16, loss: 0.2 });
// Joined after the first lap, so repair blocks only: Wirehair then wants a block past K now and again, and the chunk
// takes the rest as they come.
await run("41 chunks of 1 KiB, repair blocks only", { length: 40 * 1024 + 100, log2: 10, join: 100 });
for (let k = 0; k < 10; k++) await run(`400 KB in 16 KiB chunks, repair blocks only, seed ${k}`, { length: 400000 + 37 * k, log2: 14, join: 80 });
console.log(`${over} chunk solves needed more than K blocks and took them as they came`);
console.log(fails ? `${fails} FAILED` : "all ok");
process.exit(fails ? 1 : 0);
