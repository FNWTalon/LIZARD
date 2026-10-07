// JS face of build/ob.wasm for the harness.
import create from "../build/ob.mjs";

// OB_BUILD names ANOTHER build's ob.mjs, as a path relative to liblizard/, so a harness run can be held against a
// different build of the same source without rebuilding in between: an A/B otherwise costs a full emcc run each
// way round and the two arms cannot then run at once.
//
// THIS MODULE IS LOADED BY THE BROWSER. lizard-web/send.mjs reaches it through sim/phy.mjs and lizard-web/recv-worker.mjs
// imports it directly, so it may not carry a node: import or touch `process` at the top level; doing either
// stops both demo pages before they run. The static import above is what the browser takes, and the dynamic one
// is only ever reached under node with the variable set.
const ALT = typeof process !== "undefined" && process.env && process.env.OB_BUILD;
let M, createFn;
async function loader() {
  createFn ??= ALT ? (await import(`../${ALT}`)).default : create;
  return createFn;
}
// The margin the codec paints round the frame, in modules (src/focus.h FOCUS_QUIET), read from the codec by init(): a
// live binding, undefined until then. Not a top-level await: a worker whose imports await loses the messages that
// arrive meanwhile (lizard-web/recv-worker.mjs read nothing that way), and a page that only wants the arithmetic would pay
// for a codec instance.
export let QUIET;
export async function init() { M ??= await (await loader())(); QUIET ??= M._focus_quiet?.(); return M; }

// Stage times since the last call, ms, summed over the decodes in between (slots: PROF_* in src/internal.h).
// Both codes share one table: a stage one of them does not have stays at 0.
const STAGES = { luma: [0], binarize: [1], finder: [2], mesh: [3], sample: [4, 13], demap: [5, 14, 16], fft: [15], ldpc: [6, 17] };
export function prof() {
  const out = {};
  for (const [name, slots] of Object.entries(STAGES)) out[name] = slots.reduce((t, i) => t + M._ob_prof(i), 0);
  M._ob_prof_reset();
  return out;
}

// Where the heap break sits now. ALLOW_MEMORY_GROWTH only ever moves it up, and a growth copies the whole heap
// inside whichever decode asked for it, so a step here is a stall some decode has already been charged for.
export const heapTop = () => (M ? M._ob_heap_top() : 0);

// A camera frame's RGBA goes into wasm a strip at a time and comes out as luma there (vectorized). The heap then
// holds one byte a pixel and one strip, not four bytes a pixel: 4.4 MB less per decoder at 1080 x 1080.
const STRIP = 1 << 16;
let pStrip = 0;
// ai: The pilots the last decode read (src/focus.h focus_pilot): { r, sd, r2, sd2, blocks }, r of the even blocks (bit 0
// ai: of the painted count: it flips picture to picture) and r2 of the odd (bit 1: every second picture) with their
// ai: standard errors, or null where none were read or the build predates them (OB_BUILD).
let pPilot = 0;
function pilotOf(get) {
  if (!get) return null;
  if (!pPilot) pPilot = M._malloc(16);
  const k = get(pPilot), o = pPilot >> 2;
  return k ? { r: M.HEAPF32[o], sd: M.HEAPF32[o + 2], r2: M.HEAPF32[o + 1], sd2: M.HEAPF32[o + 3], blocks: k } : null;
}
function lumaOf(rgba, n) {
  pStrip ||= M._malloc(4 * STRIP);
  let p = 0;
  for (let off = 0; off < n; off += STRIP) {
    const c = Math.min(STRIP, n - off);
    M.HEAPU8.set(rgba.subarray(4 * off, 4 * (off + c)), pStrip);   // M.HEAPU8 afresh every time: the heap may have grown
    p = M._ob_luma_strip(pStrip, c, off, n);
  }
  return p;
}

// The test stream (src/shake.h): SHAKE256 of the block id. Both ends derive the same bytes from
// the id alone, so a receiver can check any block it decodes without knowing which frame it came
// from. It runs once per decoded block on the phone, so it lives in wasm: in JS the Keccak
// permutations cost about 0.5 ms a frame against a 4 ms decode.
let pStream = 0, streamCap = 0;
export function streamBlock(id, len) {
  if (len > streamCap) { if (pStream) M._free(pStream); pStream = M._malloc(len); streamCap = len; }
  M._stream_fill(id >>> 0, pStream, len);
  return M.HEAPU8.subarray(pStream, pStream + len);   // afresh every call: the heap may have grown
}

export class Codec {
  // map: 0 tiled, 1 interleaved. pilotStep: 0 or one known cell per pilotStep x pilotStep.
  constructor(w, h, tx, ty, rate, { map = 0, pilotStep = 0 } = {}) {
    this.blockBytes = M._ob_setup(w, h, tx, ty, rate, map, pilotStep);
    if (this.blockBytes < 0) throw new Error(`layout ${w}x${h} ${tx}x${ty} rate ${rate} rejected`);
    Object.assign(this, { w, h, tiles: M._ob_tiles(), n: M._ob_code_n(), k: M._ob_code_k() });
    this.pBlocks = M._malloc(this.tiles * this.blockBytes);
    this.pMods = M._malloc(w * h);
    this.pTruth = M._malloc(w * h);
    this.pImg = 0; this.imgCap = 0;
  }
  // eq: 0 none, 1 blind, 2 pilot, 3 genie (needs truth). taps 3 or 5. regions n = n x n local filters.
  opts({ gamma = 1, eq = 1, taps = 3, regions = 1, mesh = 1, maxIter = 50, gain = 0.7, clip = 10 } = {}) { M._ob_set_opts(gamma, eq, taps, regions, mesh, maxIter, gain, clip); }
  encode(blocks) {
    M.HEAPU8.set(blocks, this.pBlocks);
    M._ob_tx(this.pBlocks, this.pMods);
    return M.HEAPU8.slice(this.pMods, this.pMods + this.w * this.h);
  }
  // rgba: img is a canvas RGBA buffer, and the luma conversion happens in wasm (vectorized) instead of a JS loop.
  decode(img, iw, ih, truth = null, rgba = false) {
    let pLuma = 0;
    if (rgba) pLuma = lumaOf(img, iw * ih);
    else { if (img.length > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(img.length); this.imgCap = img.length; } M.HEAPU8.set(img, this.pImg); pLuma = this.pImg; }
    if (truth) M.HEAPU8.set(truth, this.pTruth);
    const ok = M._ob_rx(pLuma, iw, ih, this.pBlocks, truth ? this.pTruth : 0);
    const r = M._ob_last(), i32 = (o) => M.HEAP32[(r + o) >> 2], f32 = (o) => M.HEAPF32[(r + o) >> 2];
    return {
      tilesOk: ok, found: i32(0), finders: i32(4), quad: Array.from({ length: 8 }, (_, k) => f32(8 + 4 * k)), orient: i32(40), markScore: f32(44),
      ok: M.HEAPU8.slice(r + 52, r + 52 + this.tiles), iters: Array.from(new Int8Array(M.HEAPU8.buffer, r + 116, this.tiles)),
      msDetect: f32(180), msSample: f32(184), msDecode: f32(188), ber: f32(192), gmi: f32(196),
      blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.tiles * this.blockBytes),
    };
  }
}

// ai: A receiver told nothing (src/focus.h focus_acquire_ring, src/wasm.c focus_any_rx): it registers against the
// ai: rings (src/focus.h FOCUS_RING), reads the word there, and finishes the frame at the picture the word names, in whichever ring it was
// ai: painted (any ring may carry any picture). nmax: the largest picture it decodes (sim/phy.mjs BLIND_NMAX). Its
// ai: codecs are the module's own, apart from Focus's, and replaced by the next FocusAny.
export class FocusAny {
  constructor(nmax = 1024) {
    this.blockBytes = M._focus_any_setup(nmax);
    if (this.blockBytes < 0) throw new Error(`focus any to n = ${nmax} rejected`);
    this.nmax = nmax;
    this.maxBlocks = M._focus_any_max_blocks();
    this.pBlocks = M._malloc(this.maxBlocks * this.blockBytes); this.pOk = M._malloc(this.maxBlocks);
    this.pImg = 0; this.imgCap = 0;
  }
  // ai: held: the version field (sub-channels / 8) of the last word the caller read, 0 for none; a frame whose own word
  // ai: does not read is decoded at it. Returns n (blocks decoded), ring (the ring that registered, 0 to 2, or -1),
  // ai: pictureN (the picture finished, 0 for none), held (whether the held configuration stood in), fmt (the word this
  // ai: frame read, or null), builds (pictures built since the setup), ok and blocks (as many as the finished picture's top carries; a smaller sub-channel count's
  // ai: blocks are a prefix and the rest decline), pilot (what the pilots read: { r, sd, r2, sd2, blocks }, or null). rgba: img is a
  // ai: canvas's RGBA, made luma in wasm as Focus.decode does.
  decode(img, iw, ih, { gamma = 1, mesh = 2, held = 0, rgba = false } = {}) {
    let pLuma;
    if (rgba) pLuma = lumaOf(img, iw * ih);
    else { if (img.length > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(img.length); this.imgCap = img.length; } M.HEAPU8.set(img, this.pImg); pLuma = this.pImg; }
    const n = M._focus_any_rx(pLuma, iw, ih, gamma, mesh, this.pBlocks, this.pOk, held), r = M._ob_last();
    const T = M._focus_any_blocks();
    return { n, ring: M._focus_any_which(), pictureN: M._focus_any_n(), held: !!M._focus_any_held(), builds: M._focus_any_builds?.() ?? 0, found: M.HEAP32[r >> 2],
      quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_any_fmt()), ok: M.HEAPU8.slice(this.pOk, this.pOk + T), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + T * this.blockBytes),
      msDetect: M.HEAPF32[(r + 180) >> 2], pilot: pilotOf(M._focus_any_pilot_get) };
  }
  // ai: This object's heap buffers; the codecs are the module's, replaced by the next FocusAny.
  free() { for (const k of ["pBlocks", "pOk", "pImg"]) { if (this[k]) M._free(this[k]); this[k] = 0; } this.imgCap = 0; }
}

// FOCUS (src/focus.c). mode 0 = the paper (RS per sub-channel), 1 = soft LLRs into LDPC.
export class Focus {
  // Sub-channels to a block by LDPC rate index (1/2, 2/3, 3/4, 7/8), chosen so every rate carries about the same payload: 640 bits a sub-channel.
  static SUBS = { 2: 12, 3: 9, 4: 8, 6: 7 };
  // tilt: dB by which the last sub-channel is sent below the first (src/focus.h). A sender's choice: a receiver need not be told.
  // tiers: [[rate, blocks], ...] from the lowest frequencies outwards, in place of subch and rate: runs of blocks at
  // different code rates, every block with the same payload, so one at a lower rate takes more sub-channels (SUBS).
  // corner: side in modules of the solid corner mark (src/layout.h). Left out it is the default, 12 gapped;
  // NEGATIVE asks for none, which is what the experiments that measure the mark's own effect pass. cornerFilled:
  // keep depth 3 light through it, which is what makes it detectable. centre: one mark at the picture's
  // middle instead of four at the corners, for archive/build-scratch-2026-09/mark_race.mjs.
  // span: modules across the picture, 0 for the format's own (src/focus.h FOCUS_CELL); a number is an experiment's override.
  // bitmap: how a block's bits lie on its coefficients (src/focus.h FOCUS_BITMAP_*), left out for the format's own.
  // A capture painted before 2026-09-23 was painted with none (0): sim/phy.mjs recordedSpec says so for a replay.
  // ai: The border is always Lizard's (ring, marks, band): FOCUS's own finder frame and its `thin` switch were deleted
  // ai: on 2026-09-26, so a `thin` key is ignored. A build from before then (OB_BUILD, an A/B) still takes the switch
  // ai: after span, and gets 1, the border every build since paints.
  constructor(n, subch, mode, { clip = 2, span = 0, tilt = 0, tiers = null, corner = 0, cornerFilled = 0, centre = 0, edge = 0, trackAlt = 0, border = 0, bitmap } = {}) {
    const was = M._focus_setup.length === 13 ? [1] : [];
    if (tiers) { const a = [...tiers, [0, 0], [0, 0]].slice(0, 3).flatMap(([r, b]) => [r, b, b ? Focus.SUBS[r] : 0]); this.blockBytes = M._focus_setup_tiers(n, clip, span, ...was, tilt, corner, cornerFilled, centre, edge, trackAlt, border, ...a); }
    else this.blockBytes = M._focus_setup(n, subch, mode, clip, span, ...was, tilt, corner, cornerFilled, centre, edge, trackAlt, border);
    if (this.blockBytes < 0) throw new Error(`focus ${n}/${tiers ? JSON.stringify(tiers) : subch}/${mode} rejected`);
    // A build from before the bit map (OB_BUILD, an A/B) has no switch and paints with none.
    if (bitmap !== undefined && mode === 1 && M._focus_bitmap_set && M._focus_bitmap_set(bitmap)) throw new Error(`bit map ${bitmap} rejected`);
    this.bitmap = mode === 1 && M._focus_bitmap_get ? M._focus_bitmap_get() : 0;
    this.subch = M._focus_subch();
    this.blocks = M._focus_blocks(); this.side = M._focus_side(); this.cell = M._focus_cell ? M._focus_cell() : 0;   // samples a module
    this.quiet = M._focus_quiet ? M._focus_quiet() : undefined;   // the margin the codec paints, modules (src/focus.h FOCUS_QUIET)
    this.pBlocks = M._malloc(this.blocks * this.blockBytes); this.pOk = M._malloc(this.blocks);
    this.pDrive = M._malloc(4 * this.side * this.side); this.pImg = 0; this.imgCap = 0;
  }
  // A frame straight to the pixels a page paints (src/focus.h focus_paint_rgba): RGBA, the codec's margin round it. The
  // view is into the wasm heap and good until the next call; a caller that keeps it copies it.
  encodeRGBA(blocks) {
    this.fits(blocks);
    const W = this.side + 2 * this.quiet * this.cell, bytes = 4 * W * W;
    if (this.rgbaCap !== bytes) {
      if (this.pRGBA) M._free(this.pRGBA);
      this.pRGBA = M._malloc(bytes); this.rgbaCap = bytes;
      if (!this.pRGBA) { this.rgbaCap = 0; throw new Error("frame pixels: out of wasm memory"); }
    }
    M.HEAPU8.set(blocks, this.pBlocks);
    M._focus_tx_rgba(this.pBlocks, this.pDrive, this.pRGBA);
    return { w: W, rgba: M.HEAPU8.subarray(this.pRGBA, this.pRGBA + bytes) };
  }
  // ai: The resampler encodeRGBA paints the picture with (src/focus.h focus_resample_geom), for an encoder elsewhere
  // ai: (gpu/encoder.mjs): q pixels across the picture and its guard interval from drive pixel o on either axis (the
  // ai: drive is px a side, the RGBA quiet * cell more round it), i0[x] the first of the six picture samples pixel x
  // ai: reads (unwrapped: mod n) and w[6 x + t] their weights, both passes alike. Copies, good after the next setup.
  resampleGeom() {
    const p = M._malloc(24);
    try {
      if (M._focus_rs_geom(p)) throw new Error("focus_rs_geom: no resampler tables");
      const g = M.HEAP32.slice(p >> 2, (p >> 2) + 6), q = g[0];
      return { q, o: g[1], n: g[2], px: g[3], i0: M.HEAP32.slice(g[4] >> 2, (g[4] >> 2) + q), w: M.HEAPF32.slice(g[5] >> 2, (g[5] >> 2) + 6 * q) };
    } finally { M._free(p); }
  }
  // Every buffer this object took from the wasm heap. One replaced without this leaves them there for the worker's
  // life: the heap never shrinks, and a receiver builds a new one on every config. The codec's own workspace
  // belongs to the module and is reused.
  free() {
    const out = (p) => { if (typeof p === "number") { if (p) M._free(p); } else if (p && typeof p === "object") Object.values(p).forEach(out); };
    for (const k of Object.keys(this)) if (/^p[A-Z]/.test(k)) { out(this[k]); this[k] = 0; }
    this.gridN = 0; this.imgCap = 0; this.rgbaCap = 0;
  }
  // ai: A frame's blocks are exactly blocks x blockBytes: the heap holds that many at pBlocks, and a longer array would
  // ai: be written over whatever follows it (2026-09-29, with the sender's GPU encoder).
  fits(blocks) {
    if (blocks.length !== this.blocks * this.blockBytes) throw new Error(`a frame is ${this.blocks} blocks of ${this.blockBytes} B, not ${blocks.length} B`);
  }
  encode(blocks) {
    this.fits(blocks);
    M.HEAPU8.set(blocks, this.pBlocks);
    M._focus_tx(this.pBlocks, this.pDrive);
    return M.HEAPF32.slice(this.pDrive >> 2, (this.pDrive >> 2) + this.side * this.side);
  }
  // The word the last decode read out of the symbol's band, or null where it read none. This is all a receiver
  // that was told nothing goes on: the version, everything derived from it (sim/lizard_pick.mjs), and the rate
  // the sender means to paint at (fps 0 where it stated none).
  static fmtOf(p) {
    if (!p) return null;
    return { version: M.HEAP32[p >> 2], fps: M.HEAP32[(p >> 2) + 1] };
  }
  // The display rate the painted band states, whole frames a second, 0 for none (src/fmt.h). A sender may change
  // it between frames: only the word's cells move, so registration is untouched and no frame is invalidated.
  setFps(v) { if (M._focus_fmt_fps_set(v)) throw new Error(`fps ${v} out of range for the format word`); }
  // ai: The painted picture's count, 0 to 3 (the sender's count of painted pictures mod 4), which the next encode puts
  // ai: into the blocks' tails, bit 0 on the even blocks and bit 1 on the odd (the pilots, src/focus.h focus_parity;
  // ai: SPEC 7.3). 0 paints what every sender did before the pilots.
  setParity(c) { if (!M._focus_parity_set || M._focus_parity_set(c)) throw new Error(`count ${c}: 0 to 3, on a build with the signed pilots`); }
  // Measurement hooks: after measure(true), sent() is the bit pair (+1 / -1 per axis) the last encode put on every
  // coefficient and read() the coefficient the last decode found there, in order of frequency, 320 to a sub-channel.
  measure(on) { if (M._focus_dbg(on ? 1 : 0)) throw new Error("focus debug buffers"); }
  sent() { const p = M._focus_dbg_sym(); return new Int8Array(M.HEAPU8.buffer, p, 2 * 320 * this.subch).slice(); }
  // Per block of the last decode: LDPC iterations (-1 never converged, -2 declined, 0 not tried) and the estimate made before decoding.
  // ai: the grid's shift the last decode read off the pilots and turned back, samples along u and v ([0, 0] where
  // ai: none was taken); read() holds the coefficients before that turn (2026-10-07, scripts/exp/rate_tiers.mjs)
  align() { this.pAlign ??= M._malloc(8); if (!M._focus_align_get || !M._focus_align_get(this.pAlign)) return [0, 0]; return Array.from(M.HEAPF32.subarray(this.pAlign >> 2, (this.pAlign >> 2) + 2)); }
  blockStats() { const pi = M._focus_blk_its(), pe = M._focus_blk_est() >> 2; return { its: Array.from(new Int8Array(M.HEAPU8.buffer, pi, this.blocks)), est: Array.from(M.HEAPF32.subarray(pe, pe + this.blocks)) }; }
  read() { const p = M._focus_dbg_coef() >> 2; return M.HEAPF32.slice(p, p + 2 * 320 * this.subch); }
  // mesh: 0 off, 1 interior by neighbour average, 2 interior from the border model (research/06). 2 is never worse.
  decode(img, iw, ih, { gamma = 1, mesh = 2, rgba = false } = {}) {
    let pLuma = 0;
    if (rgba) pLuma = lumaOf(img, iw * ih);
    else { if (img.length > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(img.length); this.imgCap = img.length; } M.HEAPU8.set(img, this.pImg); pLuma = this.pImg; }
    const n = M._focus_rx(pLuma, iw, ih, gamma, mesh, this.pBlocks, this.pOk), r = M._ob_last();
    // quad: the lattice corners the frame finder settled on, TL TR BR BL in image pixels (src/ob.h). Same offset
    // as the binary codec reads: one ob_result_t serves both. A locate pass needs it to know what to crop to.
    return { n, found: M.HEAP32[r >> 2], quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_fmt_rx_ptr()), ok: M.HEAPU8.slice(this.pOk, this.pOk + this.blocks), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.blocks * this.blockBytes),
      // mark: the best corner or centre mark the finder saw, x, y and module size (src/ob.h), or null.
      mark: (() => { const m = [200, 204, 208].map((o) => M.HEAPF32[(r + o) >> 2]); return m[2] > 0 ? m : null; })(),
      msDetect: M.HEAPF32[(r + 180) >> 2], msSample: M.HEAPF32[(r + 184) >> 2], msDecode: M.HEAPF32[(r + 188) >> 2],
      // ai: pilot: what the block tails read (src/focus.h focus_pilot), { r, sd, r2, sd2, blocks }, or null
      pilot: pilotOf(M._focus_pilot_get) };
  }
  // The same decode in two halves (src/focus.h), for a caller that samples somewhere it has to wait for.
  // rxAcquire registers and hands back everything the sampler reads; rxFinish takes the grid up again. Between
  // the two this object holds one frame, so a caller with several in flight needs one Focus each, which is what
  // the receiver's pool of workers already gives it.
  //
  // The image buffer must stay put until rxFinish: src/focus.c holds a pointer to it, not a copy. Since it is
  // this object's own pImg, the only way to break that is to call rxAcquire twice.
  rxAcquire(img, iw, ih, { gamma = 1, mesh = 2, rgba = false } = {}) {
    let pLuma = 0;
    if (rgba) pLuma = lumaOf(img, iw * ih);
    else { if (img.length > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(img.length); this.imgCap = img.length; } M.HEAPU8.set(img, this.pImg); pLuma = this.pImg; }
    this.pDims ??= M._malloc(16);
    this.pState ??= M._malloc(4 * Focus.STATE);
    if (!M._focus_rx_acquire(pLuma, iw, ih, gamma, mesh, this.pDims, this.pState)) return null;
    const d = M.HEAP32.subarray(this.pDims >> 2, (this.pDims >> 2) + 4), nx = d[0], ny = d[1], n = d[2];
    const st = M.HEAPF32.subarray(this.pState >> 2, (this.pState >> 2) + Focus.STATE);
    let k = 0;
    const take = (q) => st.slice(k, k += q);
    const H = take(9), g0 = st[k++], step = st[k++];
    return { n, nx, ny, iw, ih, H, g0, step, nodeX: take(nx), nodeY: take(ny), nodeD: take(2 * nx * ny), lut: take(256) };
  }
  // ai: The registration with NOTHING of the frame copied in: the C registers on this object's own buffer, whatever it
  // ai: holds, so the nodes and the word it reads there mean nothing. It is a placeholder for a caller that hands
  // ai: rxFinish its own grid (gridView), the one thing rxFinish then reads. The buffer is this object's, never a null,
  // ai: so those reads land in its own memory and not at address zero.
  rxAcquireQuadNoImage(iw, ih, quad, orient = 0, score = 1, { gamma = 1, mesh = 2 } = {}) {
    const n = iw * ih;
    if (n > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(n); this.imgCap = n; }
    this.pDims ??= M._malloc(16);
    this.pState ??= M._malloc(4 * Focus.STATE);
    this.pQuad ??= M._malloc(32);
    M.HEAPF32.set(quad, this.pQuad >> 2);   // a raw export takes a pointer; an Array coerces to 0 and the C runs its own finder
    if (!M._focus_rx_acquire_quad(this.pImg, iw, ih, gamma, mesh, this.pQuad, orient, score, this.pDims, this.pState)) return null;
    const d = M.HEAP32.subarray(this.pDims >> 2, (this.pDims >> 2) + 4), nx = d[0], ny = d[1], nn = d[2];
    const st = M.HEAPF32.subarray(this.pState >> 2, (this.pState >> 2) + Focus.STATE);
    let k = 0;
    const take = (q) => st.slice(k, k += q);
    const H = take(9), g0 = st[k++], step = st[k++];
    return { n: nn, nx, ny, iw, ih, H, g0, step, nodeX: take(nx), nodeY: take(ny), nodeD: take(2 * nx * ny), lut: take(256) };
  }
  // The same registration from a quad found somewhere else.
  // The image is still needed: ob_acquire_quad reads the border for the track score, refines the corner nodes
  // and fills the mesh, so a GPU finder removes the SEARCH from the CPU and not the pixels.
  rxAcquireQuad(img, iw, ih, quad, orient = 0, score = 1, { gamma = 1, mesh = 2, rgba = false } = {}) {
    let pLuma = 0;
    if (rgba) pLuma = lumaOf(img, iw * ih);
    else { if (img.length > this.imgCap) { if (this.pImg) M._free(this.pImg); this.pImg = M._malloc(img.length); this.imgCap = img.length; } M.HEAPU8.set(img, this.pImg); pLuma = this.pImg; }
    this.pDims ??= M._malloc(16);
    this.pState ??= M._malloc(4 * Focus.STATE);
    this.pQuad ??= M._malloc(32);
    M.HEAPF32.set(quad, this.pQuad >> 2);
    if (!M._focus_rx_acquire_quad(pLuma, iw, ih, gamma, mesh, this.pQuad, orient, score, this.pDims, this.pState)) return null;
    const d = M.HEAP32.subarray(this.pDims >> 2, (this.pDims >> 2) + 4), nx = d[0], ny = d[1], n = d[2];
    const st = M.HEAPF32.subarray(this.pState >> 2, (this.pState >> 2) + Focus.STATE);
    let k = 0;
    const take = (q) => st.slice(k, k += q);
    const H = take(9), g0 = st[k++], step = st[k++];
    return { n, nx, ny, iw, ih, H, g0, step, nodeX: take(nx), nodeY: take(ny), nodeD: take(2 * nx * ny), lut: take(256) };
  }
  // Where the sampled grid has to land, as a view on this object's heap. Handing the sampler this instead of a
  // fresh array is one 4 MB copy a frame saved, which at n = 1024 is most of what sampling elsewhere gains.
  gridView(n) {
    if (!this.pGrid || this.gridN !== n) {
      if (this.pGrid) M._free(this.pGrid);
      // A growth build returns 0 when the heap cannot grow, and a grid written from 0 would land on static data.
      this.pGrid = M._malloc(4 * n * n); this.gridN = n;
      if (!this.pGrid) { this.gridN = 0; throw new Error("grid: out of wasm memory"); }
    }
    return M.HEAPF32.subarray(this.pGrid >> 2, (this.pGrid >> 2) + n * n);
  }
  // What a GPU chain has to allocate for, the detrend basis it needs, and the coefficient table it indexes
  // the spectrum with (src/focus.c). All fixed for a transfer, so a caller reads them once.
  shape() {
    this.pShape ??= M._malloc(24);
    M._focus_shape(this.pShape);
    const d = M.HEAP32.subarray(this.pShape >> 2, (this.pShape >> 2) + 6);
    return { n: d[0], bw: d[1], vblocks: d[2], subch: d[3], blocks: d[4], npos: d[5] };
  }
  tables() {
    const n = this.shape().n;
    this.pTab ??= M._malloc(4 * (6 * n + 2));
    M._focus_tables_out(this.pTab);
    return M.HEAPF32.slice(this.pTab >> 2, (this.pTab >> 2) + 6 * n + 2);
  }
  posTable() {
    const { subch } = this.shape();
    return new Uint32Array(M.HEAP32.buffer, M._focus_pos_ptr(), subch * 320).slice();
  }
  // Where the soft values have to land, as a view on this object's heap, and the decode finished from them.
  llrView() {
    const { subch } = this.shape();
    this.pLlr ??= M._malloc(subch * 320 * 2);
    this.pEst ??= M._malloc(subch * 4);
    return M.HEAPU8.subarray(this.pLlr, this.pLlr + subch * 320 * 2);
  }
  // est: one information estimate per sub-channel, which the decline gate reads. The GPU sends a2 and nv home
  // and the caller finishes it, so that arithmetic stays the C's.
  rxLdpc(est) {
    M.HEAPF32.set(est, this.pEst >> 2);
    const n = M._focus_rx_ldpc(this.pLlr, this.pEst, this.pBlocks, this.pOk), r = M._ob_last();
    return { n, found: M.HEAP32[r >> 2], quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_fmt_rx_ptr()), ok: M.HEAPU8.slice(this.pOk, this.pOk + this.blocks), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.blocks * this.blockBytes),
      mark: (() => { const m = [200, 204, 208].map((o) => M.HEAPF32[(r + o) >> 2]); return m[2] > 0 ? m : null; })(),
      msDetect: M.HEAPF32[(r + 180) >> 2], msSample: M.HEAPF32[(r + 184) >> 2], msDecode: M.HEAPF32[(r + 188) >> 2] };
  }
  // The code's own shape and its layered schedule, which gpu/back/ldpc.mjs allocates against, plus which
  // sub-channels each block owns. Read once a configuration, not once a frame.
  ldpcCode() {
    const pD = M._malloc(4 * 16);
    // lay = 0 asks for the dimensions only: ldpc_tables returns early on a null lay and writes nothing through
    // it. Passing a small buffer here instead smashes the heap, because the full schedule is mb + 1 + 2 * slots
    // ints and the function has no length to check against. That cost an afternoon: the corruption does not
    // fault, it just makes every later decode return nothing, which reads exactly like a bad capture.
    M._focus_ldpc_tables(0, pD, 0);
    const d = M.HEAP32.subarray(pD >> 2, (pD >> 2) + 11);
    const code = { n: d[0], k: d[1], m: d[2], z: d[3], zp: d[4], mb: d[5], kb: d[6], norm: d[7],
      slotsMax: d[8], slotsTotal: d[9], slots: d[10] };
    const pLay = M._malloc(4 * (code.mb + 1 + 2 * code.slots));
    M._focus_ldpc_tables(0, pD, pLay);
    code.lay = M.HEAP32.slice(pLay >> 2, (pLay >> 2) + code.mb + 1 + 2 * code.slots);
    const pFirst = M._malloc(4 * this.blocks), pCount = M._malloc(4 * this.blocks);
    M._focus_block_subs(pFirst, pCount);
    const first = M.HEAP32.slice(pFirst >> 2, (pFirst >> 2) + this.blocks);
    const count = M.HEAP32.slice(pCount >> 2, (pCount >> 2) + this.blocks);
    for (const q of [pD, pLay, pFirst, pCount]) M._free(q);
    return { code, first, count };
  }
  // The decode finished from bits a GPU LDPC decided, in place of rxLdpc finishing it from the soft values.
  // bits/its are what that LDPC gave back, already unpacked to one byte a coded bit.
  rxBits(est, bits, its, n) {
    this.pBits ??= M._malloc(this.blocks * n);
    this.pIts ??= M._malloc(this.blocks);
    M.HEAPU8.set(bits, this.pBits);
    M.HEAPU8.set(its, this.pIts);
    M.HEAPF32.set(est, this.pEst >> 2);
    const nb = M._focus_rx_bits(this.pLlr, this.pEst, this.pBits, this.pIts, this.pBlocks, this.pOk), r = M._ob_last();
    return { n: nb, found: M.HEAP32[r >> 2], quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_fmt_rx_ptr()), ok: M.HEAPU8.slice(this.pOk, this.pOk + this.blocks), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.blocks * this.blockBytes),
      mark: (() => { const m = [200, 204, 208].map((o) => M.HEAPF32[(r + o) >> 2]); return m[2] > 0 ? m : null; })(),
      msDetect: M.HEAPF32[(r + 180) >> 2], msSample: M.HEAPF32[(r + 184) >> 2], msDecode: M.HEAPF32[(r + 188) >> 2] };
  }
  // The decode finished from what a GPU back half left: packed payload bytes at `stride` a block, a verdict a
  // block (1 passed, 2 declined, 3 never converged, 4 CRC), the LDPC's iteration counts and the gate's estimates.
  // No registration is consumed; quad, orient and score say what was registered so ob_last reports it.
  rxAssemble(bytes, stride, verdicts, its, est, quad, orient = 0, score = 1) {
    const T = this.blocks;
    this.pAsm ??= { bytes: M._malloc(T * 512), verd: M._malloc(T), its: M._malloc(T), est: M._malloc(4 * T), quad: M._malloc(32) };
    const a = this.pAsm;
    M.HEAPU8.set(bytes.subarray(0, T * stride), a.bytes);
    M.HEAPU8.set(verdicts.subarray(0, T), a.verd);
    for (let b = 0; b < T; b++) M.HEAPU8[a.its + b] = Math.max(-128, Math.min(127, its ? its[b] : 1)) & 255;
    M.HEAPF32.set(est ? est.subarray(0, T) : new Float32Array(T), a.est >> 2);
    M.HEAPF32.set(quad, a.quad >> 2);
    const nb = M._focus_rx_assemble(a.bytes, stride, a.verd, a.its, a.est, this.pBlocks, this.pOk, a.quad, orient, score), r = M._ob_last();
    return { n: nb, found: M.HEAP32[r >> 2], quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_fmt_rx_ptr()), ok: M.HEAPU8.slice(this.pOk, this.pOk + this.blocks), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.blocks * this.blockBytes),
      mark: null, msDetect: 0, msSample: 0, msDecode: 0 };
  }
  // The format word from soft values read on the device, [ok, q0..]: feeds fmtOf and nothing in the decode.
  fmtCheck(q) {
    this.pFmtQ ??= M._malloc(4 * 4096);
    M.HEAPF32.set(q, this.pFmtQ >> 2);
    return M._focus_fmt_check_out(this.pFmtQ);
  }
  rxFinish() {
    const n = M._focus_rx_finish(this.pGrid ?? 0, this.pBlocks, this.pOk), r = M._ob_last();
    return { n, found: M.HEAP32[r >> 2], quad: Array.from({ length: 8 }, (_, k) => M.HEAPF32[(r + 8 + 4 * k) >> 2]),
      fmt: Focus.fmtOf(M._focus_fmt_rx_ptr()), ok: M.HEAPU8.slice(this.pOk, this.pOk + this.blocks), blocks: M.HEAPU8.slice(this.pBlocks, this.pBlocks + this.blocks * this.blockBytes),
      mark: (() => { const m = [200, 204, 208].map((o) => M.HEAPF32[(r + o) >> 2]); return m[2] > 0 ? m : null; })(),
      msDetect: M.HEAPF32[(r + 180) >> 2], msSample: M.HEAPF32[(r + 184) >> 2], msDecode: M.HEAPF32[(r + 188) >> 2] };
  }
}
// What ob_test_sample and focus_rx_acquire write: H, g0, step, the mesh nodes (OB_MAX_NODES = 40 each way) and
// their shifts, the decoder's gamma table, and one trailing ms the test hook uses.
Focus.STATE = 9 + 2 + 2 * 40 + 2 * 40 * 40 + 256 + 1;
