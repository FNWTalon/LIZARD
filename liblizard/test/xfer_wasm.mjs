// The transfer's exports in the wasm (src/xfer.h, src/wasm.c): BLAKE3 against the official test vectors, then files of
// 3 chunks and of 1, random and compressible, end to end through the exports, Wirehair (build/wirehair.mjs) and zstd
// (build/zstd.mjs): each chunk compressed where that is shorter, the header and manifest blocks written and read back,
// each chunk's fountain rebuilt from the manifest's sent bytes and seed alone (one chunk: the header's) and fed repair
// blocks only, each recovered chunk decompressed and verified, a corrupted chunk, a corrupted manifest and a manifest
// entry no chunk could have caught. Also the block size the layout assumes, and what hashing costs here. About 5 s.
//   node test/xfer_wasm.mjs [path/to/ob.mjs]      default build/ob.mjs
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { init as initWh, Encoder, Decoder } from "../sim/fountain.mjs";
import { init as initZstd, compress, decompress } from "../sim/zstd.mjs";

const M = await (await import(resolve(process.argv[2] ?? "build/ob.mjs"))).default();
await initWh(); await initZstd();
let checks = 0, fails = 0;
const check = (ok, what) => { checks++; if (!ok) { fails++; console.log("FAIL", what); } };
const hex = (a) => Buffer.from(a).toString("hex");
const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
const take = (p, n) => M.HEAPU8.slice(p, p + n);

const L = new Int32Array(14);
{ const p = M._malloc(4 * 14); M._xfer_layout(p); L.set(new Int32Array(M.HEAPU8.buffer, p, 14)); M._free(p); }
const [ID_BYTES, PAYLOAD, SYM_BITS, CHUNK_BITS, ID_HEADER, ID_MANIFEST, PER_BLOCK, NAME_MAX, TYPE_MAX, , , , VERSION] = L;
check(ID_BYTES === 4 && PAYLOAD === 469 && SYM_BITS === 18 && CHUNK_BITS === 14 && ID_HEADER >>> 0 === 0xfffc0000 && ID_MANIFEST >>> 0 === 0xfffc0001 && PER_BLOCK === 12 && TYPE_MAX === 158 && VERSION === 2, "layout");
check(M._focus_setup(512, 128, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0) === ID_BYTES + PAYLOAD, "a Lizard block is the id and the payload");
check(M._xfer_id_of(3, 7) >>> 0 === 3 * 2 ** 18 + 7 && M._xfer_kind_of(0xfffc0000) === 1 && M._xfer_kind_of(0xfffc0005) === 2 && M._xfer_kind_of(0xfffc0000 + 1366) === 2 && M._xfer_kind_of(0xfffc0000 + 1367) === -1 && M._xfer_kind_of(5) === 0, "ids");

const b3 = (bytes) => { const p = put(bytes), o = M._malloc(32); M._xfer_b3_hash(p, bytes.length, o); const h = take(o, 32); M._free(p); M._free(o); return h; };
const cvsOf = (file, log2) => {
  const C = 2 ** log2, n = Math.ceil(file.length / C), out = new Uint8Array(32 * n), o = M._malloc(32);
  for (let i = 0; i < n; i++) {
    const chunk = file.subarray(i * C, Math.min(file.length, (i + 1) * C)), p = put(chunk);
    check(M._xfer_chunk_cv(p, chunk.length, i, log2, o) === 0, `cv of chunk ${i}`);
    out.set(take(o, 32), 32 * i); M._free(p);
  }
  M._free(o);
  return out;
};
const rootOf = (cvs) => { const p = put(cvs), o = M._malloc(32); check(M._xfer_root(p, cvs.length / 32, o) === 0, "root"); const r = take(o, 32); M._free(p); M._free(o); return r; };

// The official vectors: whole, as the root subtree, and split at 2^10 .. 2^17.
const vec = JSON.parse(readFileSync(new URL("../vendor/blake3/test_vectors/test_vectors.json", import.meta.url), "utf8"));
let splits = 0;
for (const { input_len: n, hash } of vec.cases) {
  const input = Uint8Array.from({ length: n }, (_, i) => i % 251), want = hash.slice(0, 64);
  check(hex(b3(input)) === want, `hash of ${n}`);
  const p = put(input), o = M._malloc(32);
  M._xfer_b3_sub(p, n, 0, 1, o);
  check(hex(take(o, 32)) === want, `root subtree of ${n}`);
  M._free(p); M._free(o);
  for (let log2 = 10; log2 <= 17; log2++) if (n > 2 ** log2) { check(hex(rootOf(cvsOf(input, log2))) === want, `${n} in chunks of 2^${log2}`); splits++; }
}
console.log(`official vectors: ${vec.cases.length} cases through the wasm, ${splits} of them split into chunks`);

let s = 99;
const rand = (n) => Uint8Array.from({ length: n }, () => ((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) & 255);
// Text-like bytes: words from a small vocabulary, so zstd shrinks them by about half, and no two chunks alike.
const prose = (n) => { const w = ["lizard ", "frame ", "block ", "ring ", "pilot ", "camera ", "screen ", "the ", "a ", "of ", "and ", "reads ", "paints ", "\n"]; const out = new Uint8Array(n); let k = 0; while (k < n) { const t = w[((s = (Math.imul(s, 1103515245) + 12345) >>> 0) >>> 16) % w.length]; for (let i = 0; i < t.length && k < n; i++) out[k++] = t.charCodeAt(i); } return out; };
const enc = new TextEncoder();

function transfer(label, file, log2, name, type) {
  const length = file.length, C = 2 ** log2, chunks = Math.ceil(length / C);
  // Sender: each chunk's bytes as sent (its zstd frame where that is shorter), its blocks and Wirehair's seed attempt
  // for it; the chaining values and root of the file's own bytes; the header and manifest payloads.
  const sentOf = [], K = new Uint32Array(chunks), seeds = new Uint8Array(chunks), sent = new Uint32Array(chunks);
  let shrunk = 0;
  for (let i = 0; i < chunks; i++) {
    const plain = file.subarray(i * C, Math.min(length, (i + 1) * C)), f = compress(plain), out = f && f.length < plain.length ? f : plain;
    shrunk += out !== plain;
    sentOf.push(out); sent[i] = out.length; K[i] = Math.ceil(out.length / PAYLOAD);
    if (K[i] > 1) { const e = new Encoder(out, PAYLOAD); seeds[i] = e.seed; e.free(); }
  }
  const cvs = cvsOf(file, log2), root = chunks > 1 ? rootOf(cvs) : b3(file);
  check(hex(root) === hex(b3(file)), `${label}: the root is the file's b3sum`);
  const nm = enc.encode(name), ty = enc.encode(type), pr = put(root), pn = put(nm), pt = put(ty), ph = M._malloc(PAYLOAD);
  check(M._xfer_hdr_write(ph, length, log2, 1, chunks === 1 ? seeds[0] : 0, chunks === 1 ? sent[0] : 0, pr, pn, nm.length, pt, ty.length) === 0, `${label}: header write`);
  const header = take(ph, PAYLOAD);
  const mcount = M._xfer_manifest_blocks(chunks), pc = put(cvs), ps = put(new Uint8Array(sent.buffer)), pz = put(seeds), manifest = [];
  for (let m = 0; m < mcount; m++) { check(M._xfer_manifest_write(pc, ps, pz, chunks, pr, m, ph) === 0, `${label}: manifest ${m}`); manifest.push(take(ph, PAYLOAD)); }

  // Receiver: nothing but the payloads.
  const sc = M._malloc(52), rr = M._malloc(32), rn = M._malloc(NAME_MAX), rt = M._malloc(TYPE_MAX), hp = M._malloc(PAYLOAD);
  M.HEAPU8.set(header, hp);
  check(M._xfer_hdr_parse(hp, sc, rr, rn, rt) === 0, `${label}: header parse`);
  const f = new Int32Array(M.HEAPU8.buffer, sc, 13).slice();
  const got = { log2: f[2], codec: f[3], seed: f[4], length: (f[7] >>> 0) + (f[8] >>> 0) * 2 ** 32, chunks: f[9], mcount: f[10], sent: f[12] >>> 0, name: new TextDecoder().decode(take(rn, f[5])), type: new TextDecoder().decode(take(rt, f[6])) };
  check(got.log2 === log2 && got.codec === 1 && got.length === length && got.chunks === chunks && got.mcount === mcount && got.name === name && got.type === type && hex(take(rr, 32)) === hex(root), `${label}: header fields`);
  check(chunks === 1 ? got.sent === sent[0] && got.seed === seeds[0] : got.sent === 0 && got.seed === 0, `${label}: the one chunk's sent and seed in the header`);
  const list = M._malloc(32 * Math.max(1, chunks)), rsent = M._malloc(4 * Math.max(1, chunks)), rseed = M._malloc(Math.max(1, chunks));
  for (let m = mcount - 1; m >= 0; m--) { M.HEAPU8.set(manifest[m], ph); check(M._xfer_mf_parse(ph, hp, m, list, rsent, rseed) === 0, `${label}: manifest ${m} parse`); }
  if (mcount) check(M._xfer_manifest_check(list, chunks, rr) === 0, `${label}: list against the root`);
  const gsent = chunks === 1 ? Uint32Array.of(got.sent) : new Uint32Array(M.HEAPU8.buffer, rsent, chunks).slice(), gseed = chunks === 1 ? Uint8Array.of(got.seed) : take(rseed, chunks);
  check(gsent.every((v, i) => v === sent[i]) && gseed.every((v, i) => v === seeds[i]), `${label}: sent and seeds read back`);
  let rebuilt = 0, inflated = 0;
  for (let i = 0; i < chunks; i++) {
    const bytes = Math.min(C, length - i * C), blocks = Math.ceil(gsent[i] / PAYLOAD);
    let raw = sentOf[i];
    if (blocks > 1) {
      // The fountain from the manifest alone, fed repair blocks only (ids from the source count up), so a seed attempt
      // that is not the sender's cannot pass by the systematic blocks.
      const e = new Encoder(sentOf[i], PAYLOAD), d = new Decoder(gsent[i], PAYLOAD, gseed[i]);
      check(e.seed === gseed[i], `${label}: chunk ${i} seed ${gseed[i]}, Wirehair picked ${e.seed}`);
      let id = blocks, done = false;
      while (!done && id < blocks * 2 + 20) done = d.add(id, e.block(id++));
      raw = done ? d.recover() : new Uint8Array(0);
      e.free(); d.free();
      rebuilt++;
    }
    let out = raw;
    if (gsent[i] < bytes) { out = decompress(raw, bytes) ?? new Uint8Array(0); inflated++; }
    check(out.length === bytes, `${label}: chunk ${i} came back ${out.length} of ${bytes} bytes`);
    const pd = put(out);
    check(M._xfer_chunk_ok(hp, i, pd, out.length, mcount ? list : 0) === 0, `${label}: chunk ${i} verified`);
    if (i === Math.floor(chunks / 2)) {
      M.HEAPU8[pd + (out.length >> 1)] ^= 4;
      check(M._xfer_chunk_ok(hp, i, pd, out.length, mcount ? list : 0) !== 0, `${label}: corrupt chunk ${i} caught`);
      if (gsent[i] < bytes) { const bad = raw.slice(); bad[bad.length >> 1] ^= 4; check(decompress(bad, bytes) === null || M._xfer_chunk_ok(hp, i, put(decompress(bad, bytes)), bytes, mcount ? list : 0) !== 0, `${label}: a corrupt frame of chunk ${i} caught`); }
    }
    M._free(pd);
  }
  if (mcount) {
    M.HEAPU8[list + 32 * (chunks - 1)] ^= 1;
    check(M._xfer_manifest_check(list, chunks, rr) !== 0, `${label}: corrupt manifest caught`);
    // An entry that says a chunk was sent as more bytes than it has: no chunk could have it, so the block is refused.
    const bad = manifest[0].slice(), at = 8 + 32;
    bad[at] = 0xff; bad[at + 1] = 0xff; bad[at + 2] = 0xff; bad[at + 3] = 0x7f;
    M.HEAPU8.set(bad, ph);
    check(M._xfer_mf_parse(ph, hp, 0, list, rsent, rseed) !== 0, `${label}: a manifest entry past the chunk's bytes refused`);
  }
  [pr, pn, pt, ph, pc, ps, pz, sc, rr, rn, rt, list, rsent, rseed, hp].forEach((p) => M._free(p));
  console.log(`${label}: ${length} B in ${chunks} chunk${chunks > 1 ? "s" : ""} of 2^${log2}, ${mcount} manifest block${mcount === 1 ? "" : "s"}, ${shrunk} compressed to ${sent.reduce((a, b) => a + b, 0)} B, ${rebuilt} fountain${rebuilt === 1 ? "" : "s"} rebuilt from the manifest, ${inflated} decompressed`);
}
transfer("3 chunks", rand(2 * 2 ** 20 + 304000), 20, "IMG_2041.jpg", "image/jpeg");
transfer("1 chunk", rand(700001), 20, "notes.txt", "text/plain");
transfer("41 chunks", rand(40 * 1024 + 100), 10, "", "");
transfer("3 chunks, prose", prose(2 * 2 ** 20 + 304000), 20, "notes.txt", "text/plain");
transfer("1 chunk, prose", prose(700001), 20, "notes.txt", "text/plain");
transfer("1 block, prose", prose(900), 20, "tiny.txt", "text/plain");   // 900 bytes of prose go as one block once compressed

// What a chunk's chaining value and its compression cost here, for the pages to budget (a phone is slower).
{
  const n = 4 * 2 ** 20, p = put(rand(n)), o = M._malloc(32);
  const t = performance.now();
  for (let r = 0; r < 4; r++) M._xfer_chunk_cv(p, n, r, 22, o);
  const ms = (performance.now() - t) / 4;
  console.log(`chaining value of a 4 MiB chunk: ${ms.toFixed(1)} ms, ${(n / 1048576 / (ms / 1000)).toFixed(0)} MiB/s in this wasm on this machine`);
  M._free(p); M._free(o);
  const text = prose(n), t1 = performance.now(), z = compress(text), t2 = performance.now(), back = decompress(z, n), t3 = performance.now();
  check(back && back.every((v, i) => v === text[i]), "4 MiB of prose round trip");
  console.log(`zstd on 4 MiB of prose: ${(z.length / n * 100).toFixed(1)}% of the bytes, compress ${(t2 - t1).toFixed(0)} ms (${(n / 1048576 / ((t2 - t1) / 1000)).toFixed(0)} MiB/s), decompress ${(t3 - t2).toFixed(0)} ms`);
}
console.log(`${checks} checks, ${fails} failed`);
process.exit(fails ? 1 : 0);
