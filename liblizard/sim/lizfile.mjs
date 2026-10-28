// A .lizard read in node (the format: core/rx/lizfile.h; SPEC 15): a ZIP of stored entries whose comment is
// "LIZARD replay 1", frames NNNN.zst (JPEG-LS median prediction residuals mod 256 as one zstd frame), stats.jsonl and
// meta.json. Synchronous; a frame read when asked. Needs node's zlib.zstdDecompressSync (node 22.15 and later).
import fs from 'node:fs';
import zlib from 'node:zlib';

const COMMENT = 'LIZARD replay 1';
const LOCAL = 0x04034b50, CENTRAL = 0x02014b50, END = 0x06054b50;

// ZIP's CRC-32
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); t[i] = c >>> 0; }
  return t;
})();
function crc32(b) {
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = (c >>> 8) ^ CRC[(c ^ b[i]) & 0xff];
  return (c ^ 0xffffffff) >>> 0;
}

function med(a, b, c) {
  const mx = a > b ? a : b, mn = a < b ? a : b;
  return c >= mx ? mn : c <= mn ? mx : a + b - c;
}
// the frame's luma from its residuals (w h bytes, rows of w)
function restore(r, w, h) {
  const px = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w, up = row - w;
    px[row] = (r[row] + (y ? px[up] : 0)) & 0xff;
    for (let i = 1; i < w; i++) {
      const a = px[row + i - 1], b = y ? px[up + i] : 0, c = y ? px[up + i - 1] : 0;
      px[row + i] = (r[row + i] + med(a, b, c)) & 0xff;
    }
  }
  return px;
}

// { path, meta (parsed), metaText, stats (text, '' none), layout, frames: [{ w, h, ms, ax, ay }], frame(i) -> Uint8Array, close() }
export function openLizard(path) {
  const fd = fs.openSync(path, 'r');
  const size = fs.fstatSync(fd).size;
  const read = (at, n) => { const b = Buffer.alloc(n); if (fs.readSync(fd, b, 0, n, at) !== n) throw new Error(`${path} is cut short`); return b; };
  const tailN = Math.min(size, 22 + 0xffff);
  if (size < 22) throw new Error(`${path} is not a ZIP archive`);
  const tail = read(size - tailN, tailN);
  let e = -1;
  for (let i = tailN - 22; i >= 0; i--) if (tail.readUInt32LE(i) === END && i + 22 + tail.readUInt16LE(i + 20) === tailN) { e = i; break; }
  if (e < 0) throw new Error(`${path} is not a ZIP archive`);
  const comment = tail.toString('latin1', e + 22, e + 22 + tail.readUInt16LE(e + 20));
  if (!comment.startsWith('LIZARD replay ')) throw new Error(`${path} is not a .lizard`);
  if (comment !== COMMENT) throw new Error(`${path} is a ${comment}, newer than this reader's ${COMMENT}`);
  const count = tail.readUInt16LE(e + 10), cdSize = tail.readUInt32LE(e + 12), cdAt = tail.readUInt32LE(e + 16);
  const cd = read(cdAt, cdSize);
  const entries = new Map();
  for (let at = 0, k = 0; k < count; k++) {
    if (cd.readUInt32LE(at) !== CENTRAL) throw new Error(`${path}: its central directory does not read`);
    const method = cd.readUInt16LE(at + 10), crc = cd.readUInt32LE(at + 16), csize = cd.readUInt32LE(at + 20), usize = cd.readUInt32LE(at + 24);
    const nl = cd.readUInt16LE(at + 28), el = cd.readUInt16LE(at + 30), cl = cd.readUInt16LE(at + 32), local = cd.readUInt32LE(at + 42);
    const name = cd.toString('utf8', at + 46, at + 46 + nl);
    if (method !== 0 || csize !== usize) throw new Error(`${path}: ${name} is compressed by the ZIP (a .lizard stores its entries)`);
    const lh = read(local, 30);
    if (lh.readUInt32LE(0) !== LOCAL) throw new Error(`${path}: ${name}'s header does not read`);
    entries.set(name, { at: local + 30 + lh.readUInt16LE(26) + lh.readUInt16LE(28), size: csize, crc });
    at += 46 + nl + el + cl;
  }
  const entry = name => {
    const x = entries.get(name);
    if (!x) return null;
    const b = read(x.at, x.size);
    if (crc32(b) !== x.crc) throw new Error(`${path}: ${name}'s CRC does not hold`);
    return b;
  };
  const metaB = entry('meta.json');
  if (!metaB) throw new Error(`${path} holds no meta.json`);
  const metaText = metaB.toString('utf8'), meta = JSON.parse(metaText);
  const statsB = entry('stats.jsonl');
  const n = meta.frames ?? meta.sizes?.length ?? 0;
  const frames = [];
  for (let i = 0; i < n; i++) {
    const s = meta.sizes?.[i];
    const ax = meta.axes?.[i];
    frames.push({ w: s ? s[0] : meta.w, h: s ? s[1] : meta.h, ms: typeof meta.ms?.[i] === 'number' ? meta.ms[i] : NaN,
      ax: Array.isArray(ax) ? ax[0] : NaN, ay: Array.isArray(ax) ? ax[1] : NaN });
    if (!entries.has(String(i).padStart(4, '0') + '.zst')) throw new Error(`${path} holds no ${String(i).padStart(4, '0')}.zst`);
  }
  return {
    path, meta, metaText, stats: statsB ? statsB.toString('utf8') : '', layout: meta.layout === '2:1' ? '2:1' : '1:1', frames,
    frame(i) {
      const f = frames[i];
      if (!f) throw new Error(`no frame ${i}`);
      const r = zlib.zstdDecompressSync(entry(String(i).padStart(4, '0') + '.zst'));
      if (r.length !== f.w * f.h) throw new Error(`${path}: frame ${i} is ${r.length} B where ${f.w} x ${f.h} is ${f.w * f.h}`);
      return restore(r, f.w, f.h);
    },
    close() { fs.closeSync(fd); },
  };
}
