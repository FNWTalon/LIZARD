// ai: liblizard's WebAssembly binding (2026-10-03): the C API (../../include/lizard.h) compiled with emscripten, no
// ai: threads and no GPU, the codec's vector paths on WebAssembly SIMD; this module is its JS face, the same layers:
// ai: format arithmetic, the per-frame codec (Encoder, Decoder) and the transfer (Tx, Rx), and two conveniences over
// ai: them, Sender (a file to painted frames) and Receiver (camera frames to a file). Everything is synchronous: a
// ai: page runs a Receiver in a worker of its own (several, for several decoders). Call init() once first.
// ai: Handles hold wasm memory: free() each, or let the module go.
import create from "./dist/lizard.mjs";

let M = null;

// ai: Thrown for a failed call: code, one of lizard.h's LIZ_E_* (negative), and the library's message.
export class LizardError extends Error {
  constructor(code, message) {
    super(message || `lizard error ${code}`);
    this.name = "LizardError";
    this.code = code;
  }
}

export async function init(moduleOptions) {
  M ??= await create(moduleOptions);
  return M;
}
const mod = () => {
  if (!M) throw new Error("lizard: await init() first");
  return M;
};
const check = (rc) => {
  if (rc < 0) throw new LizardError(rc, M.UTF8ToString(M._liz_last_error()));
  return rc;
};
const made = (ptr) => {
  if (!ptr) throw new LizardError(-1, M.UTF8ToString(M._liz_last_error()));
  return ptr;
};

export const BLOCK = 473, ID_BYTES = 4, PAYLOAD = 469, MAX_BLOCKS = 128, RINGS = 4, RING_DEFAULT = -1, GAP_MODULES = 12;
const PIXFMT = { grey: 1, rgbx: 2, rgba: 3, bgra: 4 };
const bpp = (f) => (f === "grey" ? 1 : 4);
const pixfmt = (f) => {
  const v = PIXFMT[f];
  if (!v) throw new LizardError(-1, `a pixel format of grey, rgbx, rgba or bgra, not ${f}`);
  return v;
};

// ai: Scratch memory a handle keeps (the heap's views are read afresh after every call: memory may grow under them).
class Heap {
  constructor() { this.ptr = 0; this.size = 0; }
  at(size) {
    if (size > this.size) {
      if (this.ptr) M._free(this.ptr);
      this.ptr = made(M._malloc(size));
      this.size = size;
    }
    return this.ptr;
  }
  free() { if (this.ptr) M._free(this.ptr); this.ptr = 0; this.size = 0; }
}
const scratch = new Heap();
const u64 = (p) => M.HEAPU32[p >> 2] + M.HEAPU32[(p >> 2) + 1] * 2 ** 32;

export const version = () => mod().UTF8ToString(M._liz_version());
export const abi = () => mod()._liz_abi();
export const simd = () => mod()._liz_simd() === 1;

// ---- 1. format arithmetic --------------------------------------------------------------------------------------------

// ai: format: { blocks (1 to 128), ring (0 to 3, -1 the default), fps (1 to 255), codes (1 or 2) }
function formatAt(p, { blocks, ring = RING_DEFAULT, fps = 60, codes = 1 }) {
  M.HEAP32.set([blocks, ring, fps, codes], p >> 2);
  return p;
}
function geometryFrom(p) {
  const [n, span, pxm, side, width, height, gap, frameBlocks] = M.HEAP32.subarray(p >> 2, (p >> 2) + 8);
  return { n, span, pxm, side, width, height, gap, frameBlocks };
}
export function ringCells(ring = RING_DEFAULT) { return check(mod()._liz_ring_cells(ring)); }
export function geometry(format) {
  mod();
  const p = scratch.at(48);
  check(M._liz_geometry_of(formatAt(p, format), p + 16));
  return geometryFrom(p + 16);
}
export function roomFor(blocks, ring = RING_DEFAULT) { return check(mod()._liz_room_for(blocks, ring)); }
// ai: the largest size (format.blocks) a symbol may have in a w x h room of display pixels (the senders' pick)
export function pick(w, h, { codes = 1, ring = RING_DEFAULT, top = MAX_BLOCKS } = {}) { return check(mod()._liz_pick(w, h, codes, ring, top)); }

// ---- 2. the per-frame codec ------------------------------------------------------------------------------------------

export class Encoder {
  constructor(format) {
    mod();
    this.format = { ring: RING_DEFAULT, fps: 60, codes: 1, ...format };
    const p = scratch.at(16);
    this.h = made(M._liz_encoder_new(formatAt(p, this.format)));
    const g = scratch.at(32);
    check(M._liz_encoder_geometry(this.h, g));
    this.geometry = geometryFrom(g);
    this.inHeap = new Heap();
    this.outHeap = new Heap();
  }
  // ai: blocks: frameBlocks x 473 bytes; picture: the frame's count (mod 4, the pilots); out: width x height pixels
  // ai: of fmt (grey, or four bytes a pixel: an ImageData's data takes "rgba"); made if not given
  paint(blocks, picture, out, fmt = "rgba") {
    const g = this.geometry, size = g.width * g.height * bpp(fmt);
    if (blocks.length !== g.frameBlocks * BLOCK) throw new LizardError(-1, `${g.frameBlocks} blocks of ${BLOCK} bytes`);
    out ??= new Uint8ClampedArray(size);
    if (out.length < size) throw new LizardError(-1, `a frame of ${size} bytes`);
    const ib = this.inHeap.at(blocks.length), ob = this.outHeap.at(size);
    M.HEAPU8.set(blocks, ib);
    check(M._liz_encoder_paint(this.h, ib, picture >>> 0, ob, g.width * bpp(fmt), pixfmt(fmt)));
    out.set(M.HEAPU8.subarray(ob, ob + size));
    return out;
  }
  free() { if (this.h) M._liz_encoder_free(this.h); this.h = 0; this.inHeap.free(); this.outHeap.free(); }
}

// ai: A decoded frame: its verified blocks (count x 473 bytes), and what the decode found.
function decodedFrom(p, count, blocks) {
  const i = M.HEAP32.subarray(p >> 2, (p >> 2) + 9), f = M.HEAPF32;
  return {
    count, blocks,
    found: !!i[0], ring: i[1], n: i[2], word: !!i[3], blocksPerSymbol: i[4], fps: i[5], heldUsed: !!i[6], total: i[7],
    quad: Array.from(f.subarray((p >> 2) + 9, (p >> 2) + 17)),
    pilot: { blocks: M.HEAP32[(p >> 2) + 17], r: Array.from(f.subarray((p >> 2) + 18, (p >> 2) + 20)), sd: Array.from(f.subarray((p >> 2) + 20, (p >> 2) + 22)) },
  };
}

export class Decoder {
  // ai: nmax: the largest picture decoded (0: every format). held: the held word (the size, format.blocks, the last
  // ai: word read names), the one thing carried from frame to frame; decoders reading one stream may share it through
  // ai: decode's held option.
  constructor(nmax = 0) {
    mod();
    this.h = made(M._liz_decoder_new(nmax));
    this.maxBlocks = check(M._liz_decoder_max_blocks(this.h));
    this.held = 0;
    this.img = new Heap();
    this.out = M._malloc(this.maxBlocks * BLOCK + 128);
  }
  // ai: px: an image (or ImageData), w x h pixels of fmt, rows stride bytes apart. Returns the decoded frame; held
  // ai: (a { held } object shared by several decoders, else this decoder's own) is updated when the frame's word reads.
  decode(px, w, h, { fmt = "rgba", stride, held } = {}) {
    if (px.data) ({ data: px, width: w, height: h } = px);
    stride ??= w * bpp(fmt);
    const size = stride * (h - 1) + w * bpp(fmt), ib = this.img.at(size);
    M.HEAPU8.set(px.subarray ? px.subarray(0, size) : px.slice(0, size), ib);
    const hp = (this.out + this.maxBlocks * BLOCK + 3) & ~3, rp = hp + 8;   // ai: the held word and the result, 4-aligned
    M.HEAP32[hp >> 2] = held ? held.held : this.held;
    const count = check(M._liz_decode(this.h, ib, w, h, stride, pixfmt(fmt), hp, this.out, rp));
    if (held) held.held = M.HEAP32[hp >> 2]; else this.held = M.HEAP32[hp >> 2];
    return decodedFrom(rp, count, M.HEAPU8.slice(this.out, this.out + count * BLOCK));
  }
  free() { if (this.h) { M._liz_decoder_free(this.h); M._free(this.out); } this.h = 0; this.img.free(); }
}

// ai: where a camera frame's symbols are looked for: layout 1, the centre square; 2, two squares of a centred 2:1
// ai: region (a sender's two codes). [{ x, y, w, h }]
export function layoutRects(w, h, layout = 1) {
  mod();
  const p = scratch.at(32), n = check(M._liz_layout_rects(w, h, layout, p));
  return Array.from({ length: n }, (_, k) => { const [x, y, rw, rh] = M.HEAP32.subarray((p >> 2) + 4 * k, (p >> 2) + 4 * k + 4); return { x, y, w: rw, h: rh }; });
}

export function streamFill(id) {
  mod();
  const p = scratch.at(PAYLOAD);
  M._liz_stream_fill(id >>> 0, p);
  return M.HEAPU8.slice(p, p + PAYLOAD);
}

// ---- 3. the transfer -------------------------------------------------------------------------------------------------

function cString(heap, s) {
  if (s == null) return 0;
  const n = M.lengthBytesUTF8(s) + 1, p = heap.at(n);
  M.stringToUTF8(s, p, n);
  return p;
}

export class Tx {
  // ai: Tx.file(bytes, name, type) or Tx.test(firstId): a file's frames (its bytes copied in), or the test stream's
  static file(bytes, name = "", type = "") {
    mod();
    const t = Object.create(Tx.prototype), b = new Heap(), nh = new Heap(), th = new Heap();
    const p = bytes.length ? b.at(bytes.length) : 0;
    if (bytes.length) M.HEAPU8.set(bytes, p);
    try {
      t.h = made(M._liz_tx_new(p, bytes.length, cString(nh, name), cString(th, type), 1));
    } finally { b.free(); nh.free(); th.free(); }
    t.blocks = new Heap();
    return t;
  }
  static test(firstId = 0) {
    mod();
    const t = Object.create(Tx.prototype);
    t.h = made(M._liz_tx_new_test(firstId >>> 0));
    t.blocks = new Heap();
    return t;
  }
  // ai: the next frame's n blocks (n x 473 bytes)
  next(n) {
    const p = this.blocks.at(n * BLOCK);
    check(M._liz_tx_next(this.h, n, p));
    return M.HEAPU8.slice(p, p + n * BLOCK);
  }
  get info() {
    const p = scratch.at(56);
    check(M._liz_tx_info_get(this.h, p));
    const root = Array.from(M.HEAPU8.subarray(p + 24, p + 56), (b) => b.toString(16).padStart(2, "0")).join("");
    return { test: !!M.HEAP32[p >> 2], length: u64(p + 8), chunks: M.HEAPU32[(p + 16) >> 2], lap: M.HEAPU32[(p + 20) >> 2], root };
  }
  free() { if (this.h) M._liz_tx_free(this.h); this.h = 0; this.blocks.free(); }
}

export class Rx {
  // ai: a receiver keeping what it holds in wasm memory, at most maxBytes (0: no cap)
  constructor({ maxBytes = 0 } = {}) {
    mod();
    const p = scratch.at(16);
    M.HEAPU32.set([0, 0, maxBytes % 2 ** 32, Math.floor(maxBytes / 2 ** 32)], p >> 2);
    this.h = made(M._liz_rx_new(p));
    this.in = new Heap();
  }
  // ai: a frame's verified blocks (a Decoder's blocks, or several regions' joined) -> { seen, bad, judged, fresh, test }
  frame(blocks) {
    const n = blocks.length / BLOCK, p = this.in.at(blocks.length + 24);
    M.HEAPU8.set(blocks, p);
    const v = (p + blocks.length + 3) & ~3;   // ai: the verdict after the blocks, 4-aligned
    check(M._liz_rx_frame(this.h, p, n, v));
    const [seen, bad, judged, fresh, test] = M.HEAP32.subarray(v >> 2, (v >> 2) + 5);
    return { seen, bad, judged, fresh, test: !!test };
  }
  get progress() {
    const p = scratch.at(56);
    check(M._liz_rx_progress(this.h, p));
    return {
      header: !!M.HEAP32[p >> 2], done: !!M.HEAP32[(p >> 2) + 1], length: u64(p + 8), bytesIn: u64(p + 16),
      fraction: M.HEAPF64[(p + 24) >> 3], chunks: M.HEAPU32[(p + 32) >> 2], verified: M.HEAPU32[(p + 36) >> 2],
      rejected: M.HEAPU32[(p + 40) >> 2], solveMs: M.HEAPF64[(p + 48) >> 3],
    };
  }
  // ai: each chunk: 0 to 99 the share in (a floor until the manifest is in), 255 verified
  get chunks() {
    const n = check(M._liz_rx_chunks(this.h, 0, 0)), p = scratch.at(Math.max(1, n));
    check(M._liz_rx_chunks(this.h, p, n));
    return M.HEAPU8.slice(p, p + n);
  }
  meta(which) { return M.UTF8ToString(M._liz_rx_meta(this.h, which)); }
  get name() { return this.meta(0); }
  get type() { return this.meta(1); }
  get root() { return this.meta(2); }
  get error() { return this.meta(4); }
  // ai: the finished file's bytes (a copy), null before it is done
  data() {
    const p = scratch.at(8);
    if (M._liz_rx_data(this.h, p, p + 4) < 0) return null;
    const at = M.HEAPU32[p >> 2], n = M.HEAPU32[(p + 4) >> 2];
    return M.HEAPU8.slice(at, at + n);
  }
  clear() { M._liz_rx_clear(this.h); }
  free() { if (this.h) M._liz_rx_free(this.h); this.h = 0; this.in.free(); }
}

// ---- conveniences ----------------------------------------------------------------------------------------------------

// ai: A file (or the test stream) to painted frames: frame() gives the next, in order, its pilots counted.
export class Sender {
  // ai: source: { bytes, name, type } or { test: true, firstId }; format as geometry()'s
  constructor(source, format) {
    this.tx = source.test ? Tx.test(source.firstId ?? 0) : Tx.file(source.bytes, source.name ?? "", source.type ?? "");
    this.encoder = new Encoder(format);
    this.picture = 0;
  }
  get geometry() { return this.encoder.geometry; }
  get info() { return this.tx.info; }
  // ai: { data, width, height, picture }: data the frame's pixels of fmt (rgba: an ImageData's)
  frame(fmt = "rgba", out) {
    const g = this.encoder.geometry, blocks = this.tx.next(g.frameBlocks), picture = this.picture++;
    return { data: this.encoder.paint(blocks, picture, out, fmt), width: g.width, height: g.height, picture };
  }
  free() { this.tx.free(); this.encoder.free(); }
}

// ai: Camera frames to a file: each frame's regions (layout 1 or 2) decoded, their blocks into the transfer.
export class Receiver {
  constructor({ layout = 1, nmax = 0, maxBytes = 0 } = {}) {
    this.layout = layout;
    this.decoder = new Decoder(nmax);
    this.rx = new Rx({ maxBytes });
  }
  // ai: one camera frame (an ImageData, or pixels with w and h): { decoded: [per region], verdict, progress }
  push(px, w, h, opts = {}) {
    if (px.data) ({ data: px, width: w, height: h } = px);
    const fmt = opts.fmt ?? "rgba", b = bpp(fmt), stride = opts.stride ?? w * b, decoded = [];
    for (const r of layoutRects(w, h, this.layout)) {
      const at = r.y * stride + r.x * b;
      decoded.push(this.decoder.decode(px.subarray(at), r.w, r.h, { fmt, stride }));
    }
    const all = new Uint8Array(decoded.reduce((s, d) => s + d.blocks.length, 0));
    let o = 0;
    for (const d of decoded) { all.set(d.blocks, o); o += d.blocks.length; }
    return { decoded, verdict: this.rx.frame(all), progress: this.rx.progress };
  }
  // ai: { name, type, root, bytes } once the file is whole, else null
  get file() {
    const bytes = this.rx.data();
    return bytes && { name: this.rx.name, type: this.rx.type, root: this.rx.root, bytes };
  }
  clear() { this.rx.clear(); }
  free() { this.decoder.free(); this.rx.free(); }
}
