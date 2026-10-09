// The transfer in the light for the rig (src/xfer.h holds the layouts and their reasons): the sender's schedule, and a
// receiver that puts a file back together from blocks alone, each chunk checked against the file's BLAKE3 root.
//
// Pure JS. The codec's wasm (sim/ob.mjs init(), for BLAKE3 and the header), Wirehair (sim/fountain.mjs) and zstd
// (sim/zstd.mjs, as Z) are handed in, so sim/phy.mjs and the receiver page take the id test from here without loading
// any of them.
//
// COMPRESSION (2026-10-05, src/xfer.h): every chunk is compressed on its own, one zstd frame where that is shorter than
// its bytes, else sent as it is; its fountain carries the sent bytes. The manifest names each chunk's sent bytes and
// seed attempt (one chunk: the header), so a receiver solves no chunk before the manifest is in; the control cycle puts
// it within an eighth of a lap. The file's length, root and chaining values are the file's own bytes.
//
// THE SCHEDULE: every chunk interleaved with every other, in proportion to its blocks, over the whole file (a lap),
// and laps repeated for as long as the sender runs, each with fresh repair ids. Not chunks in turn, because a camera's
// loss is steady and bursty, not rare: LIZARD degrades instead of stopping, so at a distance the outer rings go and a
// frame brings a share of its blocks, and a hand that moves loses whole seconds. In turn, a chunk gets a fixed budget
// a lap (K plus a margin m, since nothing comes back to say it arrived); a receiver losing more than m / (1 + m) of
// its blocks finishes no chunk in the first lap, and the margin itself is airtime a receiver that loses nothing throws
// away. Interleaved, a file takes 1 / (1 - loss) laps at any loss. Counted over 64 MB at 4 MiB, LIZARD-512, laps of
// airtime from the join to the last chunk (1.000 the least possible; STATUS.md has the table):
//                         interleaved   in turn m=0.1   m=0.25   m=0.5   ideal
//   nothing lost             1.001          1.099        1.246    1.492   1.000
//   joined 50% into a lap    1.001          1.094        1.225    1.459   1.000
//   15% lost at random       1.184          2.183        1.249    1.495   1.176
//   outer 25% of rings       1.347          2.186        2.480    1.497   1.333
//   2 s lost every 10 s      1.271          2.126        3.743    2.977   1.250
// A late join costs neither schedule anything: in turn the chunk being sent when the camera arrived comes round again
// at the end of its lap, which is when the file would end anyway.
// What it costs: no chunk completes before the end of the first lap, so the receiver holds every chunk's blocks at
// once. It keeps them on disk (the origin private file system, OPFS) where the browser has one, so its memory is a
// chunk or two at a solve, not the file. The sender holds every chunk's Wirehair encoder, about 1.75 times the file,
// which is what the one-fountain file mode held too.
//
// Blocks are spread over a frame's slots by a shuffle drawn from the frame count: slot k of a LIZARD frame is ring k,
// the outer ones the first to go, and a plain round robin would pin chunk c to the same slots whenever the chunk count
// divides the frame's (every chunk on a ring that a distant camera never reads). Control blocks (the header, then the
// manifest) take slot 0, the lowest ring, the one read first and lost last.

// src/xfer.h, held against the codec's xfer_layout whenever a wasm is handed in. The receiver page has no codec and
// needs the id test anyway.
export const PAYLOAD = 469, ID_BYTES = 4, SYMBOL_BITS = 18, CONTROL = 0x3fff, PER_BLOCK = 12, MAX_CHUNKS = 0x3fff;
export const ID_HEADER = 0xfffc0000, SYMBOL_MASK = (1 << SYMBOL_BITS) - 1, MAX_MANIFEST = Math.ceil(MAX_CHUNKS / PER_BLOCK);
export const CODEC_NONE = 0, CODEC_ZSTD = 1;
const blocksOf = (sent) => Math.ceil(sent / PAYLOAD);
export const isControlId = (id) => id >>> SYMBOL_BITS === CONTROL;
export const idOf = (chunk, symbol) => ((chunk << SYMBOL_BITS) | symbol) >>> 0;

// 4 MiB. Interleaved, the chunk size does not move the rate: Wirehair costs about the same a byte from 1 to 16 MiB (7 to
// 9 ns here, STATUS.md), its overhead is a fraction of a block a chunk, and no chunk finishes before the lap does
// whatever its size. So memory decides it: the receiver holds one chunk at a solve several times over (the blocks read
// back, the decoder, the recovered bytes and their copy for the hash), and at 4 MiB that fits the fountain worker's
// heaps as they start (Wirehair's 16 MB and the codec's 8 MB did not grow over a 64 MB file); at the format's 16 MiB
// the decoder alone is about 24 MB. Smaller buys nothing a phone needs and costs chunks: a 64 MB file is 16 chunks and
// 2 manifest blocks at 4 MiB, 64 and 5 at 1 MiB, and an 8 GB one 1,908 against 7,630.
export const DEFAULT_LOG2 = 22;
// One control block for this many data blocks (1.6% of the airtime), more often for a short file, so a small file is
// not waiting on its header. A camera that joins sees the header within 128 blocks: 3 frames at LIZARD-512 (61 blocks), 12 at 96 (11).
const CONTROL_EVERY = 63;
const REC = ID_BYTES + PAYLOAD;   // a stored block: its symbol, then its payload

function wasm(M) {
  const put = (bytes) => { const p = M._malloc(Math.max(1, bytes.length)); M.HEAPU8.set(bytes, p); return p; };
  const take = (p, n) => M.HEAPU8.slice(p, p + n);
  const p = M._malloc(4 * 14);
  M._xfer_layout(p);
  const L = new Int32Array(M.HEAPU8.buffer, p, 14).slice();
  M._free(p);
  if (L[0] !== ID_BYTES || L[1] !== PAYLOAD || L[2] !== SYMBOL_BITS || L[3] !== 14 || L[4] >>> 0 !== ID_HEADER || L[6] !== PER_BLOCK || L[11] !== MAX_CHUNKS || L[12] !== 2)
    throw new Error(`src/xfer.h moved (layout ${L.join(",")}): sim/xfer.mjs is out of date`);
  const b3 = (bytes) => { const a = put(bytes), o = M._malloc(32); M._xfer_b3_hash(a, bytes.length, o); const h = take(o, 32); M._free(a); M._free(o); return h; };
  return { put, take, b3, logMin: L[9], logMax: L[10] };
}
export const hex = (a) => Array.from(a, (v) => v.toString(16).padStart(2, "0")).join("");
export const b64 = (a) => { let s = ""; for (const v of a) s += String.fromCharCode(v); return btoa(s); };
export const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
// BLAKE3 of a buffer through the vendored C, for a harness that has the codec's wasm to hand.
export const blake3 = (M, bytes) => wasm(M).b3(bytes);

// UTF-8 cut to 255 bytes on a character boundary, and a media type kept to printable ASCII or dropped.
function nameBytes(name) {
  const b = new TextEncoder().encode(name ?? "");
  let n = Math.min(b.length, 255);
  while (n < b.length && n > 0 && (b[n] & 0xc0) === 0x80) n--;
  return b.subarray(0, n);
}
const typeBytes = (t) => (/^[\x20-\x7e]{0,162}$/.test(t ?? "") ? new TextEncoder().encode(t ?? "") : new Uint8Array(0));

// Which chunk the next data block is for, and its symbol. A round visits every chunk of the most blocks once; a chunk
// of fewer is visited in proportion to its blocks by an accumulator, plus 2 sqrt(K) visits a lap and at least 64: a
// small chunk's count of blocks received wanders more than a big one's, and the file waits on whichever chunk is
// last. Counted (10 MB, 8.5 MB, 4.2 MB, 64 MB; 0, 15 and 30% lost) when only a last chunk could be short, that costs a
// receiver that loses nothing at most 0.55% and is level with or ahead of plain proportion everywhere else (4.2 MB at
// 30% lost: 1.431 laps against 1.472, the least being 1.429); 4 sqrt(K) + 64 cost 1.4% at 10 MB, which the rig
// measured. Compressed, chunks differ in blocks by their content and each short one takes the margin, 2 / sqrt(K) of
// its airtime at most (2.6% at 6,000 blocks), nothing where the file does not compress.
class Schedule {
  constructor(K) {
    const C = K.length;
    this.K = K; this.C = C; this.sym = new Uint32Array(C); this.i = 0; this.acc = new Uint32Array(C);
    this.max = C ? K.reduce((a, b) => Math.max(a, b), 0) : 0;
    this.w = Uint32Array.from(K, (k) => (k >= this.max ? this.max : Math.min(this.max, Math.max(64, k + Math.ceil(2 * Math.sqrt(k))))));
    this.lap = this.w.reduce((a, b) => a + b, 0);
  }
  next() {
    let c;
    for (;;) {
      if (this.i >= this.C) this.i = 0;
      c = this.i++;
      if ((this.acc[c] += this.w[c]) >= this.max) { this.acc[c] -= this.max; break; }
    }
    if (this.K[c] === 1) return idOf(c, 0);   // unfountained: the one block, again
    const s = this.sym[c];
    this.sym[c] = (s + 1) & SYMBOL_MASK;       // past 2^18 ids a chunk the sender repeats (src/xfer.h: 86% of a 16 MiB chunk missed)
    return idOf(c, s);
  }
}

export class XferSender {
  // bytes: the file. M: the codec's wasm module. Encoder: sim/fountain.mjs's, its init() already awaited. Z: sim/zstd.mjs's
  // Z, its init() awaited, or null to send every chunk as it is (codec none).
  constructor(bytes, { name = "", type = "", chunkLog2 = DEFAULT_LOG2, M, Encoder, Z = null }) {
    const W = wasm(M), size = 2 ** chunkLog2, length = bytes.length;
    if (!(chunkLog2 >= W.logMin && chunkLog2 <= W.logMax)) throw new Error(`chunk size 2^${chunkLog2} is outside 2^${W.logMin} to 2^${W.logMax}`);
    const C = Math.ceil(length / size);
    if (C > MAX_CHUNKS) throw new Error(`${length} B is ${C} chunks of 2^${chunkLog2}; the id carries ${MAX_CHUNKS}`);
    this.chunks = C; this.chunkLog2 = chunkLog2; this.size = size; this.length = length; this.codec = Z ? CODEC_ZSTD : CODEC_NONE;
    this.len = Array.from({ length: C }, (_, c) => Math.min(size, length - c * size));
    // Each chunk compressed on its own and sent as its frame where that is shorter, else as it is; a fountain over the
    // sent bytes where they are 2 blocks or more, else the one block kept here (one).
    this.sent = new Uint32Array(C); this.seed = new Uint8Array(C); this.K = new Uint32Array(C); this.enc = []; this.one = new Map();
    for (let c = 0; c < C; c++) {
      const plain = bytes.subarray(c * size, c * size + this.len[c]);
      let out = plain;
      if (Z) { const f = Z.compress(plain); if (f && f.length < plain.length) out = f; }
      this.sent[c] = out.length; this.K[c] = blocksOf(out.length);
      if (this.K[c] > 1) { const e = new Encoder(out, PAYLOAD); this.enc.push(e); this.seed[c] = e.seed; } else { this.enc.push(null); this.one.set(c, out.slice()); }
    }
    this.sentBytes = this.sent.reduce((a, b) => a + b, 0);
    // The chaining values, their root (b3sum's answer), the header and the manifest.
    let root;
    const o = M._malloc(32);
    if (C >= 2) {
      this.cvs = new Uint8Array(32 * C);
      for (let c = 0; c < C; c++) {
        const p = W.put(bytes.subarray(c * size, c * size + this.len[c]));
        if (M._xfer_chunk_cv(p, this.len[c], c, chunkLog2, o)) throw new Error(`chaining value of chunk ${c}`);
        this.cvs.set(M.HEAPU8.subarray(o, o + 32), 32 * c);
        M._free(p);
      }
      const pc = W.put(this.cvs);
      if (M._xfer_root(pc, C, o)) throw new Error("root");
      root = W.take(o, 32);
      M._free(pc);
    } else root = W.b3(bytes);
    M._free(o);
    this.root = root;
    const nm = nameBytes(name), ty = typeBytes(type), pr = W.put(root), pn = W.put(nm), pt = W.put(ty), ph = M._malloc(PAYLOAD);
    if (M._xfer_hdr_write(ph, length, chunkLog2, this.codec, C === 1 ? this.seed[0] : 0, C === 1 ? this.sent[0] : 0, pr, pn, nm.length, pt, ty.length)) throw new Error("header refused");
    this.header = W.take(ph, PAYLOAD);
    const pc = W.put(this.cvs ?? new Uint8Array(0)), ps = W.put(new Uint8Array(this.sent.buffer)), pz = W.put(this.seed);
    this.manifest = [];
    for (let m = 0; m < M._xfer_manifest_blocks(C); m++) {
      if (M._xfer_manifest_write(pc, ps, pz, C, pr, m, ph)) throw new Error(`manifest block ${m}`);
      this.manifest.push(W.take(ph, PAYLOAD));
    }
    [pr, pn, pt, ph, pc, ps, pz].forEach((p) => M._free(p));
    this.sched = new Schedule(this.K);
    // The whole control cycle (the header every other block, the manifest between) within an eighth of a lap, so a file
    // of many small chunks does not wait on its manifest.
    this.every = Math.max(1, Math.min(CONTROL_EVERY, Math.floor(this.sched.lap / (8 * Math.max(1, this.manifest.length)))));
    this.since = Infinity; this.ctl = 0; this.frame = 0;
  }
  // sent: the file's bytes as they go, every chunk's frame or its own bytes (the file's length where nothing shrank).
  info() { return { header: b64(this.header), chunks: this.chunks, chunkLog2: this.chunkLog2, manifest: this.manifest.length, root: hex(this.root), lap: this.sched.lap, codec: this.codec, sent: this.sentBytes }; }

  // The ids of the next frame of B blocks. Advances the schedule, so a frame asked for twice is two frames.
  frameIds(B, out = new Uint32Array(B)) {
    if (!this.chunks) { out.fill(ID_HEADER); this.frame++; return out; }   // an empty file is its header
    let first = 0;
    if (this.since >= this.every && B > 1) {
      const m = this.manifest.length, j = this.ctl++;
      out[first++] = !m || !(j & 1) ? ID_HEADER : ID_HEADER + 1 + ((j >> 1) % m);
      this.since = 0;
    }
    for (let k = first; k < B; k++) out[k] = this.sched.next();
    this.since += B - first;
    // Fisher-Yates over the data slots, from a generator seeded by the frame count.
    let s = (Math.imul(this.frame + 1, 0x9e3779b1) ^ 0x5bd1e995) >>> 0;
    for (let i = B - 1; i > first; i--) {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      const j = first + (s % (i - first + 1)), t = out[i];
      out[i] = out[j]; out[j] = t;
    }
    this.frame++;
    return out;
  }

  // What block `id` carries: 469 bytes into out.
  block(id, out) {
    const c = id >>> SYMBOL_BITS, s = id & SYMBOL_MASK;
    if (c === CONTROL) { out.set(s === 0 ? this.header : this.manifest[s - 1]); return out; }
    if (this.enc[c]) return this.enc[c].block(s, out);
    out.fill(0); out.set(this.one.get(c));   // a chunk of one block, zero padded
    return out;
  }
  free() { for (const e of this.enc) e?.free(); this.enc = []; this.one.clear(); }
}

// A set of symbols, as bits, grown to the highest one seen: ids are dense from 0 up, a lap at a time.
class Bits {
  constructor() { this.a = new Uint8Array(64); this.n = 0; }
  has(i) { return (i >> 3) < this.a.length && (this.a[i >> 3] >> (i & 7)) & 1; }
  add(i) {
    if ((i >> 3) >= this.a.length) { const b = new Uint8Array(Math.max(2 * this.a.length, (i >> 3) + 1)); b.set(this.a); this.a = b; }
    if (!this.has(i)) { this.a[i >> 3] |= 1 << (i & 7); this.n++; }
  }
  or(o) { for (let i = 0; i < o.a.length; i++) if (o.a[i]) for (let b = 0; b < 8; b++) if ((o.a[i] >> b) & 1) this.add(8 * i + b); }
}

// Blocks and the finished file, held in memory: the fallback where there is no origin private file system. The
// fountain worker (lizard-web/fountain-worker.mjs) has the OPFS one with the same calls.
export class MemoryStore {
  constructor() { this.kind = "memory"; this.recs = new Map(); this.out = null; }
  async clear() { this.recs.clear(); this.out = null; }
  async begin(length) { this.out = new Uint8Array(length); }
  async append(c, sym, bytes) {
    let r = this.recs.get(c);
    if (!r) this.recs.set(c, (r = { buf: new Uint8Array(REC * 256), n: 0 }));
    if ((r.n + 1) * REC > r.buf.length) { const b = new Uint8Array(2 * r.buf.length); b.set(r.buf); r.buf = b; }
    const o = r.n++ * REC;
    r.buf[o] = sym & 255; r.buf[o + 1] = (sym >>> 8) & 255; r.buf[o + 2] = (sym >>> 16) & 255; r.buf[o + 3] = sym >>> 24;
    r.buf.set(bytes.subarray(0, PAYLOAD), o + ID_BYTES);
  }
  async read(c) { const r = this.recs.get(c); return r ? r.buf.subarray(0, r.n * REC) : new Uint8Array(0); }
  async drop(c) { this.recs.delete(c); }
  async writeOut(off, bytes) { this.out.set(bytes, off); }
  async readOut(off, len) { return this.out.subarray(off, off + len); }
  async finish() { return { bytes: this.out }; }
  held() { let t = this.out?.length ?? 0; for (const r of this.recs.values()) t += r.buf.length; return t; }
}

const COLLECTING = 0, HOT = 1, VERIFIED = 3;   // 2 was a chunk decoded before the manifest, which no longer happens: a chunk's block count waits on it
const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// A file from blocks alone. add() every block a decoder hands over, control blocks included and in any order, then
// read progress() and, once done is set, result.
export class XferReceiver {
  // M: the codec's wasm. Decoder: sim/fountain.mjs's. store: a MemoryStore, or the fountain worker's OPFS store. Z:
  // sim/zstd.mjs's Z (null: a compressed chunk is rejected).
  constructor({ M, Decoder, store, Z = null }) {
    this.M = M; this.W = wasm(M); this.Decoder = Decoder; this.store = store; this.Z = Z;
    this.hdr = null; this.source = ""; this.lightSeen = false;
    this.ch = new Map(); this.raw = new Map();
    this.verified = 0; this.rejected = 0; this.manifestRejected = 0; this.refused = 0; this.over = 0; this.ms = 0; this.done = false; this.result = null;
  }

  async reset() {
    const M = this.M;
    for (const s of this.ch.values()) s.dec?.free();
    if (this.hdr) [this.hdr.p, this.hdr.root, this.hdr.list, this.hdr.psent, this.hdr.pseed].forEach((p) => M._free(p));
    this.hdr = null; this.source = ""; this.lightSeen = false; this.ch.clear(); this.raw.clear();
    this.verified = 0; this.rejected = 0; this.manifestRejected = 0; this.over = 0; this.ms = 0; this.done = false; this.result = null;
    await this.store.clear();
  }

  state(c) {
    let s = this.ch.get(c);
    if (!s) this.ch.set(c, (s = { have: new Bits(), ban: new Bits(), state: COLLECTING, dec: null }));
    return s;
  }

  async add(id, bytes) {
    const c = id >>> SYMBOL_BITS, sym = id & SYMBOL_MASK;
    if (c === CONTROL) {
      if (sym === 0) return this.header(bytes, "light");
      if (sym <= MAX_MANIFEST) return this.manifestBlock(sym - 1, bytes);
      return;   // reserved: this version ignores it
    }
    const h = this.hdr;
    if (this.done || (h && c >= h.chunks)) return;
    const s = this.state(c);
    if (s.state === VERIFIED || s.have.has(sym) || s.ban.has(sym)) return;
    s.have.add(sym);
    if (s.dec) { if (s.dec.add(sym, bytes)) await this.recovered(c, s.dec); return; }
    await this.store.append(c, sym, bytes);
    if (h && h.K && s.have.n >= h.K[c]) await this.solve(c);   // K: known once the manifest is in (one chunk: from the header)
  }

  // A header block, from the light or, as the fallback, the network (the sender page's copy of the same 469 bytes).
  // The light's always wins: a different one is a new transfer, and everything held is let go.
  async header(bytes, source) {
    const light = source === "light";
    if (light) this.lightSeen = true;
    if (this.hdr) {
      if (eq(bytes, this.hdr.bytes)) { if (light) this.source = "light"; return; }
      if (!light) return;
    }
    const M = this.M, W = this.W, ph = W.put(bytes), sc = M._malloc(52), root = M._malloc(32), nm = M._malloc(255), ty = M._malloc(158);
    const rc = M._xfer_hdr_parse(ph, sc, root, nm, ty);
    // ai: a failed parse writes nothing: name and type lengths read then were heap leftovers, megabytes decoded for nothing.
    if (rc) { this.refused++; [ph, sc, root, nm, ty].forEach((p) => M._free(p)); return; }   // not a header this receiver reads
    const f = new Int32Array(M.HEAPU8.buffer, sc, 13).slice(), name = new TextDecoder().decode(W.take(nm, f[5])), type = new TextDecoder().decode(W.take(ty, f[6]));
    [sc, nm, ty].forEach((p) => M._free(p));
    if (this.hdr) await this.reset();
    // sent, seed and K: each chunk's bytes as sent, its seed attempt and its blocks, from the header for one chunk and
    // from the manifest (psent, pseed, read back once the list checks) for more; null until then.
    const chunks = f[9], size = f[11], length = (f[7] >>> 0) + (f[8] >>> 0) * 2 ** 32;
    const h = { bytes: bytes.slice(), p: ph, root, list: M._malloc(Math.max(1, 32 * chunks)), psent: M._malloc(Math.max(4, 4 * chunks)), pseed: M._malloc(Math.max(1, chunks)),
      name, type, log2: f[2], size, length, chunks, codec: f[3], mcount: f[10], len: Uint32Array.from({ length: chunks }, (_, c) => Math.min(size, length - c * size)),
      sent: null, seed: null, K: null, mgot: new Uint8Array(f[10]), mhave: 0, listOk: false };
    if (chunks === 1) { h.sent = Uint32Array.of(f[12]); h.seed = Uint8Array.of(f[4]); h.K = Uint32Array.of(blocksOf(f[12])); }
    this.hdr = h; this.source = source; this.lightSeen ||= light;
    for (const c of [...this.ch.keys()]) if (c >= chunks) { this.ch.delete(c); await this.store.drop(c); }   // blocks of another transfer
    await this.store.begin(h.length);
    if (!chunks) {   // an empty file: its root is BLAKE3 of nothing, which is all there is to check
      if (eq(W.b3(new Uint8Array(0)), W.take(root, 32))) await this.finish(); else this.refused++;
      return;
    }
    for (const [m, b] of this.raw) await this.manifestBlock(m, b);
    this.raw.clear();
    if (h.K) await this.solveReady();
  }
  // Every chunk with its blocks in hand, now that their counts are known.
  async solveReady() {
    const h = this.hdr;
    for (const [c, s] of [...this.ch]) if (!this.done && s.state === COLLECTING && s.have.n >= h.K[c]) await this.solve(c);
  }

  async manifestBlock(m, bytes) {
    const h = this.hdr, M = this.M;
    if (!h) { if (this.raw.size < 4096) this.raw.set(m, bytes.slice()); return; }   // kept until there is a root to check it by
    if (h.listOk || m >= h.mcount || h.mgot[m]) return;
    const p = this.W.put(bytes), ok = M._xfer_mf_parse(p, h.p, m, h.list, h.psent, h.pseed) === 0;
    M._free(p);
    if (!ok) return;   // another transfer's (its tag), not a manifest block, or entries no chunk could have
    h.mgot[m] = 1;
    if (++h.mhave < h.mcount) return;
    if (M._xfer_manifest_check(h.list, h.chunks, h.root)) {
      // Every block passed its parse and the list still does not make the root: one of them passed its CRC wrongly.
      // Which one cannot be told, so all are collected again; the sender repeats them.
      h.mgot.fill(0); h.mhave = 0; this.manifestRejected++;
      return;
    }
    h.listOk = true;
    h.sent = new Uint32Array(M.HEAPU8.buffer, h.psent, h.chunks).slice(); h.seed = M.HEAPU8.slice(h.pseed, h.pseed + h.chunks); h.K = h.sent.map(blocksOf);
    await this.solveReady();
  }

  async solve(c) {
    const t = performance.now();
    await this.solveOnce(c);
    this.ms += performance.now() - t;   // interleaved, most of this falls after the last block: it is the transfer's tail
  }
  async solveOnce(c) {
    const h = this.hdr, s = this.ch.get(c), recs = await this.store.read(c);
    if (h.K[c] === 1) { const out = recs.slice(ID_BYTES, ID_BYTES + h.sent[c]); await this.store.drop(c); return this.unpack(c, out); }
    const d = new this.Decoder(h.sent[c], PAYLOAD, h.seed[c]);
    let done = false;
    for (let o = 0; o < recs.length && !done; o += REC) done = d.add((recs[o] | (recs[o + 1] << 8) | (recs[o + 2] << 16) | (recs[o + 3] << 24)) >>> 0, recs.subarray(o + ID_BYTES, o + REC));
    await this.store.drop(c);   // the decoder holds them now
    if (!done) { s.dec = d; s.state = HOT; this.over++; return; }   // Wirehair wanted a block more than K (a few percent of chunks): fed as they come
    await this.recovered(c, d);
  }
  async recovered(c, d) {
    const s = this.ch.get(c), out = d.recover();
    d.free(); s.dec = null;
    await this.unpack(c, out);
  }
  // The chunk's bytes from what was sent: its zstd frame decompressed where it was sent shorter than it is (a frame
  // that does not give exactly the chunk's bytes is a wrong chunk, rejected as a wrong hash is), else as they came.
  async unpack(c, raw) {
    const h = this.hdr, s = this.ch.get(c);
    let out = raw;
    if (h.sent[c] < h.len[c]) { out = this.Z ? this.Z.decompress(raw, h.len[c]) : null; if (!out) return this.reject(c, s); }
    await this.check(c, out);
  }
  async check(c, out) {
    const h = this.hdr, s = this.ch.get(c);
    if (this.verify(c, out)) { await this.store.writeOut(c * h.size, out); await this.accept(c, s); } else this.reject(c, s);
  }
  // xfer_chunk_check: the chunk's chaining value against the list (already checked against the root), or, for a file
  // of one chunk, its hash against the root.
  verify(c, out) {
    const M = this.M, h = this.hdr, p = this.W.put(out), ok = M._xfer_chunk_ok(h.p, c, p, out.length, h.chunks >= 2 ? h.list : 0) === 0;
    M._free(p);
    return ok;
  }
  async accept(c, s) {
    s.state = VERIFIED; s.have = s.ban = null; this.verified++;
    if (this.verified === this.hdr.chunks) await this.finish();
  }
  // A chunk that decoded to the wrong bytes: some block of it passed its CRC wrongly (1 in 2^32), or was sent wrong.
  // Which one cannot be told, so every symbol that went into it is refused from now on and the chunk is collected
  // again from fresh ids; the sender's next laps bring them.
  reject(c, s) {
    s.ban.or(s.have); s.have = new Bits(); s.state = COLLECTING; this.rejected++;
  }
  async finish() {
    this.done = true;
    const h = this.hdr;
    this.result = { ...(await this.store.finish(h)), name: h.name, mediaType: h.type, length: h.length, sent: this.sentTotal(), root: hex(this.W.take(h.root, 32)), source: this.source };
  }
  // The file's bytes as they go, every chunk's frame or its own (null until the manifest says), and how many of them
  // are in: a verified chunk's all, another's blocks in hand (2026-10-05, so a page's time left counts the bytes the
  // light carries, not the file's).
  sentTotal() { const h = this.hdr; return h?.sent ? h.sent.reduce((a, b) => a + b, 0) : null; }
  sentIn() {
    const h = this.hdr;
    if (!h?.sent) return 0;
    let got = 0;
    for (let c = 0; c < h.chunks; c++) { const s = this.ch.get(c); if (s) got += s.state === VERIFIED ? h.sent[c] : Math.min(h.sent[c], s.have.n * PAYLOAD); }
    return got;
  }

  // For the page: what is known, and each chunk's share of its blocks in hand (0 to 99, 255 verified; until the
  // manifest says how many blocks a chunk has, the share is over the blocks its bytes would take uncompressed, a floor).
  progress() {
    const h = this.hdr;
    if (!h) return { header: null, lightSeen: this.lightSeen, held: this.ch.size, refused: this.refused };
    const per = new Uint8Array(h.chunks);
    for (let c = 0; c < h.chunks; c++) {
      const s = this.ch.get(c);
      per[c] = !s ? 0 : s.state === VERIFIED ? 255 : Math.min(99, Math.floor((100 * s.have.n) / (h.K ? h.K[c] : blocksOf(h.len[c]))));
    }
    return { header: { name: h.name, type: h.type, length: h.length, chunks: h.chunks, log2: h.log2, root: hex(this.W.take(h.root, 32)), sent: this.sentTotal() }, sentIn: this.sentIn(), source: this.source, lightSeen: this.lightSeen,
      manifest: h.mcount, manifestHave: h.mhave, listOk: h.listOk, verified: this.verified, rejected: this.rejected, manifestRejected: this.manifestRejected, refused: this.refused, over: this.over, solveMs: this.ms, per, done: this.done };
  }
}

// ai: fractionDone(p): how much of the file is in, 0 to 1, from progress()'s answer (or the fountain worker's xfer
// ai: message, which carries it): a chunk verified (255; 254, decoded before the manifest, until 2026-10-05) counts whole, any other its per / 100, each
// ai: weighted by its length (2^log2 bytes, the last chunk the rest). 0 with no header; 1 once done, an empty file too.
export function fractionDone(p) {
  const h = p?.header;
  if (p?.done) return 1;
  if (!h?.length) return 0;
  if (h.sent) return Math.min(1, p.sentIn / h.sent);   // ai: the bytes the light carries, once the manifest has said them (2026-10-05)
  const size = 2 ** h.log2;
  let got = 0;
  for (let c = 0; c < h.chunks; c++) {
    const v = p.per[c], len = Math.min(size, h.length - c * size);
    got += len * (v >= 254 ? 1 : v / 100);
  }
  return got / h.length;
}
