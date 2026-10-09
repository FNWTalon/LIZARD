// ai: The sender's encoder on the GPU (2026-09-29), for PCs with a good GPU, as the sender has lagged at LIZARD-1024;
// ai: bit exact only in the ring; batches of 16, since a sender's latency matters little and a dropped frame does.
// ai: Frames' blocks in, their symbols' pixels out, on the device, and onto a WebGPU canvas a frame at a time; the page
// ai: (lizard-web/send.mjs) keeps its timing and paces the encodes, and the worker keeps making the blocks.
// ai:
// ai: The kernels, one compute pass an encode of up to `sub` frames, a symbol a workgroup z (frames x codes, the codes
// ai: of a frame side by side):
// ai:   paint: a block a workgroup (wgsl/send.mjs paintSource, 2026-10-07): CRC-32, its code's systematic codeword
// ai:     (every code of the format's rate profile, src/focus.h focus_tiers_for; the codes' tables in TAB, sendTab,
// ai:     from the wasm's codec), the bit map and whitening, the pilots, QPSK onto the disc;
// ai:   irows, ipic: the decoder's own inverse transforms (wgsl/cancel_paint.mjs, mode "send"): the inverse along x,
// ai:     turn_pack's packing and the inverse along y, clipped to 0..1 in f32, left quad-major for whole-line stores;
// ai:   tpose: the picture row-major again (wgsl/send.mjs);
// ai:   rsv, rsh: the C's Lanczos-3 resample onto the ring's square with the C's own taps (sim/ob.mjs resampleGeom),
// ai:     grey as focus_paint_rgba rounds it, and every pixel outside the square the border's byte (wgsl/send.mjs);
// ai:     where every tap is the C's on-sample copy (the span divides n: every picture in the 32 and 64 rings, 384,
// ai:     768 and 1536 in the 96, all but 384 in the 128, 20 of the 24 pairs) rsh alone, reading the picture (rsh's
// ai:     copy form);
// ai:   present: a render pass onto the canvas, whole canvas pixels a frame pixel, a frame of a batch.
// ai: The border (ring, marks, band, word, quiet margin) is the C's paint, taken once a configuration and again at each
// ai: fps change from this module's own Focus on the importing thread's wasm, so it is the C's to the byte. The square
// ai: is not: the transform's rounding differs, within one grey level of the C's (scripts/gpu/encoder_check.mjs); `check`
// ai: holds a device to that on frames of its choosing (a driver that returns garbage fails it), not a conformance
// ai: rule: a sender's symbol need only decode (2026-09-29).
// ai:
// ai:   const enc = await GpuEncoder.create();                 // throws Unavailable: no WebGPU, no adapter
// ai:   const cfg = await enc.configure({ spec, codes, fps, gap }); // spec { n, subch, span, bitmap }; gap: modules between codes; again at a re-pick
// ai:   const ctx = enc.attach(canvas);                        // a canvas with no 2D context
// ai:   enc.encode(half, f0, blocks, count, cfg);              // frames f0 .. f0 + count - 1 of a half, count <= cfg.sub,
// ai:                                                          // each codes x blocks x 473 B
// ai:   enc.present(half * cfg.batch + f, ctx, whole, cfg);    // frame f of that half, at its due time
// ai:   enc.release(cfg);                                      // a configuration no frame still needs
// ai:   await enc.setFps(fps); await enc.check(blocks, count, cfg); cfg.frameMs; enc.takeGpuMs(); enc.destroy();
// ai: A configuration holds two batches' frames (half 0 and 1, `batch` frames each: one presenting while the next is
// ai: made; a frame is encoded again only after its present was submitted) and the scratch of `sub` frames. An encode
// ai: is one submission, and WebGPU runs a queue in order, so a present waits behind every encode submitted before it:
// ai: one encode of a whole batch of 16 at LIZARD-1024 on the iGPU (85 ms) cost 6 of 60 frames a second (2026-09-29),
// ai: which is why the encodes are small and the page paces them (cfg.frameMs, the measured GPU ms a frame of that
// ai: configuration: timestamps, or a submission's time to done where there are none). A configuration is its own
// ai: buffers, so frames encoded before a re-pick present from theirs while the next one's are made; cfg left out is
// ai: the latest. The wasm holds one codec at a time, the latest configuration's Focus (cfg.focus): configure, check
// ai: and setFps run one at a time on a chain shared by every encoder, and only the latest configuration's border is ever painted again (an fps
// ai: change leaves an older one's last frames at the old rate). batch 16 and sub 4 unless asked, sub then batch halved
// ai: while the device cannot hold a configuration's buffers (cfg.batch, cfg.sub say what it got).
import { irowsSource, ipicSource } from "./wgsl/cancel_paint.mjs";
import { paintSource, tposeSource, rsvSource, rshSource, presentSource, TPOSE_TILE, RSV_THREADS, RSH_THREADS } from "./wgsl/send.mjs";
import { transformTables } from "./back/transform.mjs";
import { whiten, perm } from "./back/bitmap.mjs";
import { ruleTables } from "./back/codes.mjs";
import { CODES, WHITE_SLOTS } from "./wgsl/back_tiers.mjs";
import { PARAMS_AT } from "./wgsl/back_ldpc.mjs";
// ai: the gap between two codes, in modules, unless configure names one (lizard-web/send-worker.mjs paints the same for the
// ai: wasm path; the page's slider gives both)
export const GAP_MODULES = 12;
import { crcPowers, PAYLOAD } from "./back/ref_ldpc.mjs";
import { clipLimit } from "./back/paint.mjs";
import { computePipeline } from "./pipeline.mjs";
import { init as initOb, Focus } from "../sim/ob.mjs";

// ai: A block's 473 bytes in BLOCKS, padded to 120 words (wgsl/send.mjs paintSource).
const BLOCK_BYTES = 480;

// ai: The paint's TAB (wgsl/send.mjs paintSource) from the codec's codes (back/codes.mjs ruleTables: { codes, bitmap },
// ai: the codes in CODES' order): word 0 the whitening's offset; 16 words a code at 16 + 16 c, (n, k, m, z, mb, rows,
// ai: lay, perm, subs, nt); a code's block rows' first entries (mb + 1), its entries (col | shift << 16), its bit map
// ai: (slot i's codeword bit, np + perm[i], in the half i & 1 of word i >> 1, over the nt sent bits); the whitening, a
// ai: bit a slot of the frame over 1024 sub-channels and a word past them. The same for every format: gen/sender.mjs
// ai: bakes it for the native painter.
export function sendTab({ codes, bitmap }) {
  if (codes.length !== CODES.length) throw new Error(`the paint takes ${CODES.length} codes, got ${codes.length}`);
  let o = 16 + 16 * codes.length;
  const at = codes.map((c) => { const rows = o; o += c.mb + 1; const lay = o; o += c.slots; const pm = o; o += c.nt / 2; return { rows, lay, pm }; });
  const WHITE = o;
  o += WHITE_SLOTS / 32 + 1;
  const tab = new Uint32Array(o);
  tab[0] = WHITE;
  codes.forEach((c, q) => {
    const a = at[q];
    if (c.nt % 2 || 640 * CODES[q].subs < c.nt || c.n / 32 > 240) throw new Error(`the ${CODES[q].name} code's n ${c.n} (${c.nt} sent) does not fit the paint`);
    tab.set([c.n, c.k, c.m, c.z, c.mb, a.rows, a.lay, a.pm, CODES[q].subs, c.nt], 16 + 16 * q);
    for (let r = 0; r <= c.mb; r++) tab[a.rows + r] = c.lay[r];
    for (let e = 0; e < c.slots; e++) tab[a.lay + e] = c.lay[c.mb + 1 + 2 * e] | (c.lay[c.mb + 2 + 2 * e] << 16);
    const p = perm(c.nt, c.k - c.np, bitmap);
    for (let i = 0; i < c.nt; i++) tab[a.pm + (i >> 1)] |= (c.np + p[i]) << (16 * (i & 1));
  });
  if (bitmap) { const w = whiten(WHITE_SLOTS); for (let i = 0; i < WHITE_SLOTS; i++) if (w[i]) tab[WHITE + (i >> 5)] |= 1 << (i & 31); }
  return tab;
}
// ai: The codec's codes and profiles, read once a page on the codec's chain (ruleTables sets up a codec of its own).
let codesOnce = null;
// ai: Frames a half unless asked (16), and frames an encode at most.
export const BATCH = 16, SUB = 4;
// ai: Timestamp readbacks in flight at most; a batch that finds none free goes untimed.
const TIMERS = 4;
// ai: The compute stages a batch, in order (a profile's names).
export const STAGES = ["paint", "irows", "ipic", "tpose", "rsv", "rsh"];

export class Unavailable extends Error {}

// ai: The wasm holds one codec for the whole module, whichever encoder set it up: the steps that use it (configure,
// ai: check, setFps) run one at a time on this chain across every GpuEncoder, and focus is the Focus that holds it
// ai: (null while it changes hands). A configuration paints its border only while its Focus is this one.
const codec = { chain: Promise.resolve(), focus: null };

export class GpuEncoder {
  // ai: adapter: one the caller asked for, else the one powerPreference names (high-performance: the sender is for a
  // ai: PC's good GPU). batch: frames a half; sub: frames an encode at most. profile: every stage its own compute pass
  // ai: and timestamp pair (takeStageMs), for a timing; the default is one pass an encode, timed whole (takeGpuMs).
  static async create({ adapter = null, powerPreference = "high-performance", batch = BATCH, sub = SUB, profile = false, log = () => {} } = {}) {
    if (!globalThis.navigator?.gpu) throw new Unavailable("no WebGPU in this browser");
    adapter ??= await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Unavailable("no WebGPU adapter");
    const L = adapter.limits, ts = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: ts ? ["timestamp-query"] : [],
      requiredLimits: { maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize },
    });
    const e = new GpuEncoder();
    const i = adapter.info ?? {};
    Object.assign(e, { adapter, device, batch, sub: Math.min(sub, batch), dead: false, log, ts, format: navigator.gpu.getPreferredCanvasFormat(), pipes: new Map(), cfg: null, cfgs: new Set(), lost: null, error: null, timers: [], timersMade: 0, gpuNs: [], profile: profile && ts, stageNs: [] });
    e.adapterName = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ").replace(/\s+/g, " ").trim() || "unknown adapter";
    device.lost.then((x) => { e.lost = x.message || x.reason || "lost"; log(`gpu encoder: device lost: ${e.lost}`); });
    device.addEventListener("uncapturederror", (x) => { e.error ??= x.error.message; log(`gpu encoder: ${x.error.message}`); });
    try { await e.buildShared(); } catch (err) { device.destroy(); throw err; }
    return e;
  }

  // ai: The layouts, and the pipelines no picture size shapes (paint, tpose, rsv, rsh twice, present), compiled at once.
  async buildShared() {
    const d = this.device, RO = "read-only-storage", RW = "storage", UN = "uniform";
    const layout = (types, visibility = GPUShaderStage.COMPUTE) => d.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility, buffer: { type } })) });
    this.bgl = {
      paint: layout([RO, RO, RO, RW, UN]),
      irows: layout([RO, RO, RW, UN, UN]),
      ipic: layout([RO, RO, RW, UN]),
      tpose: layout([RO, RW, UN]),
      rsv: layout([RO, RO, RW, UN]),
      rsh: layout([RO, RO, RO, RW, UN]),
      present: layout([RO, UN], GPUShaderStage.FRAGMENT),
    };
    const module = d.createShaderModule({ label: "send present", code: presentSource() });
    // ai: The pipelines under names of their own: `present` is the method.
    [this.paintPipe, this.tposePipe, this.rsvPipe, this.rshPipe, this.rshCopyPipe, this.presentPipe] = await Promise.all([
      computePipeline(d, { code: paintSource(), layout: this.pl("paint"), label: "send paint" }),
      computePipeline(d, { code: tposeSource(), layout: this.pl("tpose"), label: "send tpose" }),
      computePipeline(d, { code: rsvSource(), layout: this.pl("rsv"), label: "send rsv" }),
      computePipeline(d, { code: rshSource(), layout: this.pl("rsh"), label: "send rsh" }),
      computePipeline(d, { code: rshSource({ copy: true }), layout: this.pl("rsh"), label: "send rsh copy" }),
      d.createRenderPipelineAsync({ label: "send present", layout: this.pl("present"), vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [{ format: this.format }] }, primitive: { topology: "triangle-list" } }),
    ]);
  }
  pl(k) { return this.device.createPipelineLayout({ bindGroupLayouts: [this.bgl[k]] }); }

  // ai: The two inverse transforms of a picture size, built once and kept. ipic holds two exchange buffers of n vec2f
  // ai: (24,576 B at 1536): a device that offers less has no GPU encoder at that size (Unavailable).
  pipesFor(n) {
    if (!this.pipes.has(n)) {
      const lim = this.device.limits, need = 16 * n;
      if (need > lim.maxComputeWorkgroupStorageSize) throw new Unavailable(`the ${n} picture's inverse needs ${need} B of workgroup memory, the device has ${lim.maxComputeWorkgroupStorageSize}`);
      if (n / 8 > lim.maxComputeInvocationsPerWorkgroup) throw new Unavailable(`the ${n} picture's inverse needs ${n / 8} invocations a workgroup`);
      const d = this.device, job = Promise.all([
        computePipeline(d, { code: irowsSource({ n, prec: "f32", mode: "send" }), layout: this.pl("irows"), label: `send irows ${n}` }),
        computePipeline(d, { code: ipicSource({ n, prec: "f32", mode: "send" }), layout: this.pl("ipic"), label: `send ipic ${n}` }),
      ]);
      job.catch(() => this.pipes.delete(n));
      this.pipes.set(n, job);
    }
    return this.pipes.get(n);
  }

  // ai: The encoder's steps that use the wasm's one codec run one at a time, in call order (a failed one does not stop
  // ai: the next).
  serial(fn) {
    const p = codec.chain.then(fn);
    codec.chain = p.catch(() => {});
    return p;
  }

  // ai: A configuration: the format's tables, the border and the resampler from the C, every buffer a batch needs.
  // ai: spec as the page's (n, subch, span, bitmap); codes symbols a frame, side by side; fps the rate the word states.
  configure(opts) { return this.serial(() => this.configureNow(opts)); }
  async configureNow({ spec, codes = 1, fps = 0, gap = GAP_MODULES }) {
    const { n, subch, span = 0, bitmap } = spec;
    const alive = () => { if (this.dead) throw new Unavailable("the encoder was destroyed"); };
    await initOb();
    alive();
    const pipes = this.pipesFor(n);
    // ai: The codec changes hands from here (ruleTables, the first time, and transformTables set up and free their own,
    // ai: then this configuration's Focus), so the Focus before it can no longer paint, whether or not this
    // ai: configuration comes to be.
    codec.focus?.free();
    codec.focus = null;
    codesOnce ??= await ruleTables();
    const rt = codesOnce, profile = rt.profiles[subch / 8];
    if (!profile || (bitmap ?? rt.bitmap) !== rt.bitmap) throw new Unavailable(`the GPU encoder paints the format's codes and bit map, not LIZARD-${subch}'s${profile ? ` bit map ${bitmap}` : ""}`);
    // ai: TAB, the encoder's for its life (the same for every format; release leaves it, destroy takes the device)
    if (!this.TAB) { const tab = sendTab(rt); this.TAB = this.device.createBuffer({ size: tab.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.device.queue.writeBuffer(this.TAB, 0, tab); }
    const [tb] = await transformTables([{ n, subch, bitmap }]);
    const fc = new Focus(n, subch, 1, { span, bitmap });
    let c = null;
    try {
      alive();
      if (fps) fc.setFps(fps);
      const g = fc.resampleGeom(), V = fc.blocks, B = fc.blockBytes, W = fc.side + 2 * fc.quiet * fc.cell;
      const want = profile.reduce((a, t) => a + t.count, 0);
      if (g.n !== n || tb.blocks !== subch / 8 || V !== want || B !== PAYLOAD) throw new Error(`the tables (${tb.blocks} chunks at ${n}, a profile of ${want} blocks) and the codec (${V} of ${B} B at ${g.n}) disagree`);
      const [irows, ipic] = await pipes;
      alive();
      // ai: The sizes. Over the device's limits: FRAME (two halves of batch frames) halves batch, the scratch (sub
      // ai: frames) halves sub. Out of memory, which buffer is unknown: sub first, then batch. Both error scopes, popped
      // ai: together at once: a buffer out of memory is invalid and the bind groups made with it are validation errors,
      // ai: which would otherwise reach uncapturederror (this.error, which gives the encoder up) from an attempt thrown
      // ai: away, and a scope left open across an animation frame would catch the running configuration's errors.
      const lim = this.device.limits, most = Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize), d = this.device;
      let batch = this.batch, sub = this.sub;
      while (!c) {
        sub = Math.min(sub, batch);
        if (batch < 1) throw new Unavailable(`no configuration of LIZARD-${subch} at ${n} fits this device's memory`);
        const t = this.shape({ spec, codes, gapModules: gap, tb, g, fc, V, B, W, batch, sub, irows, ipic }), z = t.sizes;
        t.profile = profile;
        if (z.FRAME > most) { batch = Math.floor(batch / 2); continue; }
        if (Math.max(z.BLOCKS, z.S, z.Y, z.PQT, z.PIC) > most) {
          if (sub === 1) throw new Unavailable(`one frame of LIZARD-${subch} at ${n} needs a buffer over this device's limit`);
          sub = Math.floor(sub / 2); continue;
        }
        d.pushErrorScope("validation");
        d.pushErrorScope("out-of-memory");
        this.allocate(t, tb, g);
        const [oom, bad] = await Promise.all([d.popErrorScope(), d.popErrorScope()]);
        if (oom || bad || this.dead) {
          for (const b of t.bufs) b.destroy();
          alive();
          this.log(`gpu encoder: halves of ${batch}, encodes of ${sub}: ${(oom ?? bad).message}; smaller`);
          if (sub > 1) sub = Math.floor(sub / 2); else batch = Math.floor(batch / 2);
          continue;
        }
        c = t;
      }
    } catch (e) { fc.free(); throw e; }
    c.focus = codec.focus = fc;
    this.cfg = c;
    this.cfgs.add(c);
    this.paintBorder(c);
    this.log(`gpu encoder: ${c.label}, ${this.adapterName}, halves of ${c.batch} frames, encodes of up to ${c.sub}, square ${c.q} px from ${c.sq}${c.copy ? " (a copy)" : ""}, frame ${c.FW} x ${c.W}, ${(c.bufs.reduce((t, b) => t + b.size, 0) / 1048576).toFixed(1)} MB`);
    return c;
  }

  // ai: A configuration's numbers and its buffers' sizes: halves of `batch` frames, scratch for `sub` (R symbols).
  shape({ spec, codes, gapModules = GAP_MODULES, tb, g, fc, V, B, W, batch, sub, irows, ipic }) {
    const { n, subch, span = 0 } = spec, q = g.q, R = sub * codes;
    // ai: two codes a gap of gapModules modules apart (2026-09-30; a slider since 2026-10-03; lizard-web/send-worker.mjs
    // ai: paints the same gap): the frame is codes W + (codes - 1) gap wide, the gap the margin's colour (wgsl/send.mjs rsh pixel)
    const gap = codes > 1 ? gapModules * fc.cell : 0;
    const FW = codes * W + (codes - 1) * gap, RW = Math.ceil(FW / 4), words = RW * W, FS = 64 * Math.ceil(words / 64);
    // ai: A copy when every pixel takes one sample whole (the C's on-sample taps): no vertical pass, no TMP.
    const copy = g.w.every((w, i) => w === (i % 6 === 2 ? 1 : 0));
    const sStride = tb.npos, Vr = tb.V, yStride = Vr * n;
    const sizes = {
      BLOCKS: R * V * BLOCK_BYTES, S: R * sStride * 8, Y: R * yStride * 8,
      // ai: PICQ, ipic's quad-major picture, and TMP, the vertical pass's rows, one buffer: tpose reads PICQ before
      // ai: rsv writes TMP.
      PQT: R * 4 * Math.max(n * n, copy ? 0 : q * n), PIC: R * n * n * 4, FRAME: batch * FS * 4,
    };
    return { spec, n, subch, span, codes, gap, V, B, W, FW, RW, FS, q, sq: g.o + fc.quiet * fc.cell, words, Vr, sStride, yStride, batch, sub, R, copy, sizes, irows, ipic, bufs: [], whole: 0, frameMs: null, focus: null,
      label: `LIZARD-${subch} at ${n}, ${W} px a symbol${codes > 1 ? ` x ${codes}` : ""}` };
  }

  // ai: The buffers and bind groups of a shaped configuration.
  allocate(c, tb, g) {
    const d = this.device, U = GPUBufferUsage, { n, V, q, codes, sStride, Vr, yStride, W, R } = c;
    const buf = (size, usage, data) => {
      const b = d.createBuffer({ size: Math.max(16, 4 * Math.ceil(size / 4)), usage });
      if (data) d.queue.writeBuffer(b, 0, data);
      c.bufs.push(b);
      return b;
    };
    // ai: the paint's uniform: the blocks, each code's tier (first block, first sub-channel, blocks), dims, the CRC's powers
    const paintUni = new Uint32Array(PARAMS_AT.words);
    paintUni.set([V, 0, 0, 0], PARAMS_AT.sizes);
    for (const t of c.profile) paintUni.set([t.firstBlock, t.sub0, t.count, 0], PARAMS_AT.sizes + 4 * (1 + t.c));
    paintUni.set([V, sStride, R, 0], PARAMS_AT.dims);
    paintUni.set(crcPowers(PAYLOAD), PARAMS_AT.pw);
    const size = new ArrayBuffer(48), s32 = new Uint32Array(size), sf = new Float32Array(size), lim = clipLimit(c.subch);
    s32.set([n, Vr, yStride, sStride, (n * n) / 4, 0, 0, 0]); sf[8] = lim; sf[9] = Math.fround(0.5 / lim);
    const rows = new Uint32Array(2048); rows.set(tb.rows);
    // ai: The taps: each one's sample mod n (the C wraps it in resample), then the weights, 6 a pixel.
    const taps = new ArrayBuffer(48 * q), idx = new Uint32Array(taps, 0, 6 * q);
    for (let x = 0; x < q; x++) for (let t = 0; t < 6; t++) idx[6 * x + t] = (((g.i0[x] + t) % n) + n) % n;
    new Float32Array(taps, 24 * q, 6 * q).set(g.w);
    c.g = new Uint32Array([n, q, c.sq, W, codes, c.FW, c.RW, c.FS, 1, 0, c.gap, 0]);
    c.BLOCKS = buf(c.sizes.BLOCKS, U.STORAGE | U.COPY_DST);
    c.staging = new Uint8Array(c.sizes.BLOCKS);
    const UV = buf(4 * tb.UV.length, U.STORAGE | U.COPY_DST, tb.UV);
    c.S = buf(c.sizes.S, U.STORAGE | U.COPY_DST);
    const PU = buf(4 * PARAMS_AT.words, U.UNIFORM | U.COPY_DST, paintUni);
    const TW = buf(8 * n, U.STORAGE | U.COPY_DST, tb.tw);
    const ROWS = buf(4 * rows.length, U.UNIFORM | U.COPY_DST, rows);
    const SU = buf(48, U.UNIFORM | U.COPY_DST, new Uint8Array(size));
    const Y = buf(c.sizes.Y, U.STORAGE);
    const PQT = buf(c.sizes.PQT, U.STORAGE);
    const PIC = buf(c.sizes.PIC, U.STORAGE);
    const TAPS = buf(48 * q, U.STORAGE | U.COPY_DST, new Uint8Array(taps));
    c.BORDER = buf(W * W, U.STORAGE | U.COPY_DST);
    const GU = (c.GU = buf(48, U.UNIFORM | U.COPY_DST, c.g));
    c.PG = buf(48, U.UNIFORM | U.COPY_DST, c.g);
    c.FRAME = [buf(c.sizes.FRAME, U.STORAGE | U.COPY_SRC), buf(c.sizes.FRAME, U.STORAGE | U.COPY_SRC)];
    const group = (k, list) => d.createBindGroup({ layout: this.bgl[k], entries: list.map((r, binding) => ({ binding, resource: r.buffer ? r : { buffer: r } })) });
    c.paintGroup = group("paint", [c.BLOCKS, this.TAB, UV, c.S, PU]);
    c.PU = PU;
    c.irowsGroup = group("irows", [c.S, TW, Y, ROWS, SU]);
    c.ipicGroup = group("ipic", [Y, TW, PQT, SU]);
    c.tposeGroup = group("tpose", [PQT, PIC, GU]);
    c.rsvGroup = c.copy ? null : group("rsv", [PIC, TAPS, PQT, GU]);
    c.rshGroups = c.FRAME.map((F) => group("rsh", [c.copy ? PIC : PQT, TAPS, c.BORDER, F, GU]));
    // ai: A frame's present binds its own words of its half's FRAME, FS words apart (FS a multiple of 64 words: 256 B,
    // ai: the offset alignment WebGPU asks of a storage binding).
    c.presentGroups = [];
    for (const F of c.FRAME) for (let f = 0; f < c.batch; f++) c.presentGroups.push(group("present", [{ buffer: F, offset: 4 * c.FS * f, size: 4 * c.words }, c.PG]));
  }

  // ai: The C's paint of configuration c's symbol, its grey bytes (every pixel outside the square is read from them): at
  // ai: configure and at every fps change, when only the word's cells move. Only while c's Focus is the wasm's codec.
  paintBorder(c) {
    if (!c.focus || c.focus !== codec.focus) throw new Error(`${c.label}: its codec is no longer the wasm's`);
    const { rgba } = c.focus.encodeRGBA(new Uint8Array(c.V * c.B)), grey = new Uint8Array(4 * Math.ceil((c.W * c.W) / 4));
    for (let i = 0; i < c.W * c.W; i++) grey[i] = rgba[4 * i];
    this.device.queue.writeBuffer(c.BORDER, 0, grey);
  }
  // ai: The rate the latest configuration's word states, its border painted again (after any configure in flight).
  setFps(fps) {
    return this.serial(() => {
      const c = this.cfg;
      if (!c || c.focus !== codec.focus) return;
      c.focus.setFps(fps);
      this.paintBorder(c);
    });
  }

  // ai: count frames (1 to c.sub) into frames f0 .. f0 + count - 1 of half h (0 or 1), one submission: each frame codes x
  // ai: V blocks of 473 bytes, each block its id then its payload (sim/phy.mjs frameBlocks), frame after frame. The
  // ai: frame offset goes in G before the submission, in queue order. parity: the first frame's count (the sender's
  // ai: count of painted pictures mod 4; each frame after it one more), which the pilots carry (SPEC 7.3).
  encode(h, f0, blocks, count, c = this.cfg, parity = 0) {
    const d = this.device, per = c.codes * c.V, R = count * c.codes;
    if (count < 1 || count > c.sub || f0 + count > c.batch || blocks.length !== count * per * c.B) throw new Error(`an encode is 1 to ${c.sub} frames of ${per} blocks of ${c.B} B within a half of ${c.batch}, got ${count} frames from ${f0} in ${blocks.length} B`);
    for (let i = 0; i < count * per; i++) c.staging.set(blocks.subarray(i * c.B, (i + 1) * c.B), i * BLOCK_BYTES);
    d.queue.writeBuffer(c.BLOCKS, 0, c.staging, 0, count * per * BLOCK_BYTES);
    d.queue.writeBuffer(c.GU, 36, new Uint32Array([f0]));
    d.queue.writeBuffer(c.PU, 4 * (PARAMS_AT.dims + 3), new Uint32Array([(parity & 3) | (c.codes << 2)]));
    const enc = d.createCommandEncoder(), t = this.timer(), K = this.profile ? STAGES.length : 1;
    enc.clearBuffer(c.S, 0, R * c.sStride * 8);
    const stages = [
      [this.paintPipe, c.paintGroup, [c.V, 1, R]],
      [c.irows, c.irowsGroup, [c.Vr, 1, R]],
      [c.ipic, c.ipicGroup, [c.n / 4, 1, R]],
      [this.tposePipe, c.tposeGroup, [c.n / 4 / TPOSE_TILE, c.n / TPOSE_TILE, R]],
      c.copy ? null : [this.rsvPipe, c.rsvGroup, [Math.ceil(c.n / 4 / RSV_THREADS), c.q, R]],
      [c.copy ? this.rshCopyPipe : this.rshPipe, c.rshGroups[h], [Math.ceil(c.RW / RSH_THREADS), c.W, count]],
    ];
    // ai: One pass (timed whole), or with profile a pass a stage, a skipped stage (rsv on a copy) its pair around nothing.
    let p = null;
    stages.forEach((st, k) => {
      if (!p || this.profile) p = enc.beginComputePass(t ? { timestampWrites: { querySet: t.qs, beginningOfPassWriteIndex: this.profile ? 2 * k : 0, endOfPassWriteIndex: this.profile ? 2 * k + 1 : 1 } } : {});
      if (st) { const [pipe, group, n] = st; p.setPipeline(pipe); p.setBindGroup(0, group); p.dispatchWorkgroups(...n); }
      if (this.profile || k === stages.length - 1) p.end();
    });
    if (t) { enc.resolveQuerySet(t.qs, 0, 2 * K, t.res, 0); enc.copyBufferToBuffer(t.res, 0, t.read, 0, 16 * K); }
    const t0 = performance.now();
    d.queue.submit([enc.finish()]);
    // ai: c's GPU ms a frame, a running mean the page paces by: the timestamps; without timestamp-query, the submission's
    // ai: time to done (an upper bound: it holds whatever the queue held before it).
    const seen = (ms) => { c.frameMs = c.frameMs == null ? ms : 0.8 * c.frameMs + 0.2 * ms; };
    if (t) t.read.mapAsync(GPUMapMode.READ).then(() => {
      const ns = new BigInt64Array(t.read.getMappedRange(0, 16 * K)).slice();
      t.read.unmap();
      const dt = (k) => Number(ns[2 * k + 1] - ns[2 * k]) / count;
      let ns1 = 0;
      if (this.profile) { const v = STAGES.map((_, k) => dt(k)); if (v.every((x) => x >= 0)) { this.stageNs.push(v); ns1 = v.reduce((a, b) => a + b, 0); } }
      else if (dt(0) > 0) ns1 = dt(0);
      if (ns1 > 0) { this.gpuNs.push(ns1); seen(ns1 / 1e6); }
      this.timers.push(t);
    }).catch(() => {});
    else if (!this.ts) d.queue.onSubmittedWorkDone().then(() => seen((performance.now() - t0) / count)).catch(() => {});
  }
  // ai: A free timestamp set, made up to TIMERS; null without timestamp-query or with every set in flight.
  timer() {
    if (!this.ts) return null;
    if (this.timers.length) return this.timers.pop();
    if (this.timersMade >= TIMERS) return null;
    this.timersMade++;
    const d = this.device, U = GPUBufferUsage, K = this.profile ? STAGES.length : 1;
    return { qs: d.createQuerySet({ type: "timestamp", count: 2 * K }), res: d.createBuffer({ size: 16 * K, usage: U.QUERY_RESOLVE | U.COPY_SRC }), read: d.createBuffer({ size: 16 * K, usage: U.MAP_READ | U.COPY_DST }) };
  }
  // ai: The mean GPU ms a frame over the encodes timed by timestamps since the last call (null: none timed).
  takeGpuMs() {
    const a = this.gpuNs.splice(0);
    return a.length ? a.reduce((t, x) => t + x, 0) / a.length / 1e6 : null;
  }
  // ai: With profile: each stage's mean ms a frame since the last call ({ paint, irows, ... }), else null.
  takeStageMs() {
    const a = this.stageNs.splice(0);
    return a.length ? Object.fromEntries(STAGES.map((k, i) => [k, a.reduce((t, v) => t + v[i], 0) / a.length / 1e6])) : null;
  }

  // ai: The canvas the frames are presented on (it must have no 2D context).
  attach(canvas) {
    const ctx = canvas.getContext("webgpu");
    if (!ctx) throw new Unavailable("the canvas gave no WebGPU context");
    ctx.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    return ctx;
  }
  // ai: Slot h batch + f (frame f of half h) onto the canvas: whole canvas pixels a frame pixel (the page's layout).
  present(slot, ctx, whole = 1, c = this.cfg) {
    const d = this.device;
    if (c.whole !== whole) { c.whole = whole; const u = c.g.slice(); u[8] = whole; d.queue.writeBuffer(c.PG, 0, u); }
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 1, g: 1, b: 1, a: 1 } }] });
    pass.setPipeline(this.presentPipe); pass.setBindGroup(0, c.presentGroups[slot]); pass.draw(3); pass.end();
    d.queue.submit([enc.finish()]);
  }

  // ai: Slot h batch + f's pixels read back: FW x W grey bytes, the symbols side by side, row by row.
  async readback(slot, c = this.cfg) {
    const d = this.device, bytes = 4 * c.words, h = Math.floor(slot / c.batch), f = slot % c.batch;
    const rd = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(c.FRAME[h], 4 * c.FS * f, rd, 0, bytes);
    d.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const all = new Uint8Array(rd.getMappedRange()), out = new Uint8Array(c.FW * c.W);
    for (let y = 0; y < c.W; y++) out.set(all.subarray(4 * c.RW * y, 4 * c.RW * y + c.FW), c.FW * y);
    rd.unmap(); rd.destroy();
    return out;
  }

  // ai: This device against the C on count frames in half 0 (encodes of up to c.sub): each frame encoded here and by the
  // ai: C (its own paint, the same word), compared symbol by symbol. ok when every border and margin is equal to the byte
  // ai: and every square within one grey level. borderDiff: border pixels that differ; squareMax, squareDiff: the
  // ai: squares' largest difference and the pixels that differ at all; got: the frames' pixels, frame after frame.
  // ai: c: the configuration to check, which must still be the latest when the check's turn comes (a check queued
  // ai: behind another configure is refused, never run on the other configuration).
  check(blocks, count = 1, c = this.cfg) { return this.serial(() => this.checkNow(blocks, count, c)); }
  async checkNow(blocks, count, c) {
    if (!c || c !== this.cfg || c.focus !== codec.focus) throw new Error(`check: ${c?.label ?? "no configuration"} is no longer the latest`);
    const per = c.V * c.B, fb = c.codes * per;
    // ai: frame f painted as the f-th picture (count f mod 4), the C told the same before its encode
    for (let f0 = 0; f0 < count; f0 += c.sub) {
      const k = Math.min(c.sub, count - f0);
      this.encode(0, f0, blocks.subarray(f0 * fb, (f0 + k) * fb), k, c, f0 & 3);
    }
    let borderDiff = 0, squareMax = 0, squareDiff = 0;
    const got = new Uint8Array(count * c.FW * c.W);
    for (let f = 0; f < count; f++) {
      const px = await this.readback(f, c);
      got.set(px, f * c.FW * c.W);
      c.focus.setParity(f & 3);
      for (let k = 0; k < c.codes; k++) {
        const { rgba } = c.focus.encodeRGBA(blocks.subarray((f * c.codes + k) * per, (f * c.codes + k + 1) * per));
        for (let y = 0; y < c.W; y++) for (let x = 0; x < c.W; x++) {
          const dv = Math.abs(px[y * c.FW + k * (c.W + c.gap) + x] - rgba[4 * (y * c.W + x)]);
          if (!dv) continue;
          if (x >= c.sq && x < c.sq + c.q && y >= c.sq && y < c.sq + c.q) { squareDiff++; if (dv > squareMax) squareMax = dv; } else borderDiff++;
        }
      }
    }
    c.focus.setParity(0);
    return { ok: !this.lost && !this.error && borderDiff === 0 && squareMax <= 1, borderDiff, squareMax, squareDiff, pixels: count * c.codes * c.W * c.W, got };
  }

  // ai: A configuration's buffers freed (its Focus stays the wasm's codec while it is the latest: another configure frees it).
  release(c) {
    if (!c || !this.cfgs.delete(c)) return;
    for (const b of c.bufs) b.destroy();
    if (this.cfg === c) this.cfg = null;
  }
  // ai: A configure still running on the chain throws at its next step (dead) and frees its own Focus.
  destroy() {
    this.dead = true;
    if (this.cfg && this.cfg.focus === codec.focus) { codec.focus?.free(); codec.focus = null; }
    for (const c of [...this.cfgs]) this.release(c);
    this.device.destroy();
  }
}
