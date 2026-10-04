// ai: The GPU decoder (gpu/decoder.mjs FrontHalf) opened for the sender's format and fed from a frame queue, for the
// ai: receiver's GPU worker (recv-gpu-worker.mjs). Throughput over latency: the decoder's ms a frame falls with the
// ai: batch (17 ms alone, 9.5 at 7 on the S26 Ultra), so a batch waits to fill, and a stale frame is worth less than
// ai: the next one, so whatever cannot be taken is dropped, never queued behind. Each frame goes onto the device the
// ai: moment it arrives (fh.enqueue: a slot on the decoder's ring) and its VideoFrame is closed in the same task, so
// ai: a camera buffer is held only until its ingest executes on the device's in-order queue. A batch is cut from the
// ai: ring, the oldest staged frames first, once it holds the batch size (fh.size) or the oldest has waited the
// ai: bound (FILL batches of arrivals at the measured rate: a partial batch only when the camera is slower than the
// ai: size assumed). fh.inflight (2) batches compute at once: the second's compute hides the first's readback round
// ai: trip (18 to 68 ms on the S26 at 1080; one at a time sat at the edge of 60 a second and dropped frames at 1440).
// ai: Each frame is moved to the device as it arrives and its camera buffer freed once the ingest runs, at most two
// ai: batches' compute later; the camera held 60 a second with two batches of 32 ahead, its buffers no problem (STATUS
// ai: "The phone at the camera's rate"). Before the ring exists,
// ai: and while a plan (the first frame, a larger crop, a recovery)
// ai: runs, at most HOLD_MAX frames wait open. A lost device is recovered once; the second time, or on any other
// ai: failure, the queue gives up and the page falls back to the CPU pool.
import { FrontHalf, CAP, CAP_MS, precisionChain } from "../liblizard/gpu/decoder.mjs";
import { WEIGHTS_INT8, storeOptions } from "../liblizard/gpu/bank_fcn2.mjs";
import { formatTables, PICTURES } from "../liblizard/gpu/tables.mjs";
import { loadNet } from "../liblizard/gpu/cnn/netfile.mjs";
import { CASCADE } from "../liblizard/gpu/wgsl/classify.mjs";
import { pass1WorkgroupBytes } from "../liblizard/gpu/wgsl/back_transform.mjs";

export class Unavailable extends Error {}

export const HOLD_MAX = 4;   // ai: frames held open with no ring to take them: under what the camera's pool spares (about 5 on the S26)
// ai: batches' worth of staged frames past which a new frame is dropped before it touches the device (stage): the device is
// ai: behind the camera, and ingesting more only delays what is already staged.
const STAGE_BATCHES = 2;
const FILL = 1.5;            // ai: the bound in batches of arrivals: one to fill, half again for the readback and the ingests between
const RATE_TAU = 2000;       // ai: ms the arrival-rate EMA remembers: rides a dropped frame, follows a camera change within seconds
const RATE_MIN = 5;          // ai: fps the EMA never goes under: a longer gap is a pause, not the rate, and would stretch the bound past the page's watchdog
// ai: The installed classifiers, as paths under liblizard/: large, cnn-v1, the net the cascade runs on each frame's 64
// ai: best peaks; small, cnn-v3, the net on every kept peak (gpu/wgsl/classify.mjs CASCADE). Whole literals, which the
// ai: installed app rewrites to its files' hashed names (lizard-web/pwa/hash.mjs).
const INSTALLED = { large: "gpu/cnn/weights.safetensors", small: CASCADE.weights };

// ai: Why a spec is not the GPU decoder's to read, or null. It reads Lizard in the format (soft LDPC): any of the
// ai: rings, any of the picture sizes (gpu/tables.mjs PICTURES). A blind spec (sim/phy.mjs blindSpec) is the format by construction.
export function unsupported(spec) {
  if (spec?.phy !== "focus") return `${spec?.phy ?? "no"} code: the GPU decoder reads Lizard only`;
  if (spec.blind) return PICTURES.some((s) => s.n === spec.nmax) ? null : `picture sizes to ${spec.nmax}: none of ${PICTURES.map((s) => s.n).join(", ")}`;
  if (spec.grid) return "a split picture (grid) is not the format";
  if (spec.mode !== 1) return "RS blocks: the GPU back half is the soft LDPC only";
  if (!PICTURES.some((s) => s.n === spec.n)) return `picture size ${spec.n} is none of ${PICTURES.map((s) => s.n).join(", ")}`;
  return null;
}

const weights = new Map();
const weightsOf = (path) => {
  // ai: a path under liblizard/, or a candidate under research/ (../research/..., the checks')
  if (path.replace(/^\.\.\/research\//, "").includes("..") || path.startsWith("/")) throw new Unavailable(`classifier ${path}: a path under liblizard/ or ../research/`);
  if (!weights.has(path)) weights.set(path, loadNet(new URL(`../liblizard/${path}`, import.meta.url)).catch((e) => { weights.delete(path); throw e; }));
  return weights.get(path);
};

// ai: The decoder for a spec, every option named (create's defaults are the harness's: f32, no cascade). Every picture
// ai: at its top (gpu/tables.mjs PICTURES), so a frame's ring comes from the finder and its version and picture from
// ai: its own word (F8), or the held one the caller passes each batch (GpuQueue.word). A blind spec (the receiver's,
// ai: sim/phy.mjs blindSpec) builds the back half to its nmax; a told one (the self-test, the bench) to max(1024, n):
// ai: the decoder reads it blind all the same. restart: the batch size after a lost device (gpu/decoder.mjs create).
// ai: Throws Unavailable when there is no adapter or it fails.
// ai: prec ("int8", "f16" or "f32") and subgroups (false) override what the adapter offers: diagnostic switches for a
// ai: device whose int8, f16 or subgroup arithmetic is suspect (recv.html menus); prec "int8" also names the int8
// ai: form for a test. With no prec the decoder is built at the first precision of precisionChain(adapter)
// ai: (gpu/decoder.mjs: int8 where the browser has the packed integer dot product, then f16 where shader-f16, then
// ai: f32) that builds (receiverChain). An int8 build that fails is logged and the next
// ai: precision tried on a new adapter (an adapter makes one device); a float build that fails is Unavailable.
// ai: test bypasses one stage to find a device's failing one: "nocascade" (cnn-v1 on every peak), "b1" (batches of
// ai: 1).
// ai: probe: the self-test's readback of every stage (gpu/probe.mjs, scripts/pages/gpu_selftest.mjs); the receiver never sets
// ai: it. cancel: straddle cancellation as pass two of every batch, in the batch's own submit (gpu/back/cancel.mjs),
// ai: off unless the receiver's "GPU cancellation" menu turns it on; the self-test's judged runs keep it off, since
// ai: its reference predates pass two. nets: { large, small }, classifier files by path
// ai: under liblizard/ in place of the installed ones (INSTALLED), as gpu/harness/page.mjs takes a job's; a q8 file
// ai: (arch.quant) runs int8 at precision int8 and is refused at a float one. The GPU bench's arms name them
// ai: (archive/gpu-bench/gpu_bench.mjs); the receiver never does. The result's nets is fh.nets: the arithmetic each net runs in.
// ai: twins: what an installed float classifier runs at int8 (gpu/decoder.mjs create): null, the decoder's default
// ai: (under B "auto" each net timed in its float form and its int8 twin on the first batch's frames, the faster kept;
// ai: fh.nets says which once planned, fh.twins the times), "int8" the twins (the self-test's int8 run), "off" the float files.
// ai: The precisions openGpu tries on this adapter with no prec, best first: precisionChain's, int8 only where an int8
// ai: proposer file is installed or the page's ?fcnstore=weights= names one. The self-test judges the same list.
export function receiverChain(adapter) {
  return precisionChain(adapter).filter((p) => p !== "int8" || WEIGHTS_INT8 || storeOptions().weights);
}

export async function openGpu(spec, { pref = null, log = () => {}, restart = 0, prec = null, subgroups = true, test = null, probe = false, cancel = false, nets = null, twins = null } = {}) {
  const why = unsupported(spec);
  if (why) throw new Unavailable(why);
  if (!globalThis.navigator?.gpu) throw new Unavailable("this browser has no WebGPU");
  const getAdapter = async () => {
    const a = await navigator.gpu.requestAdapter(pref ? { powerPreference: pref } : {});
    if (!a) throw new Unavailable("no WebGPU adapter");
    return a;
  };
  // ai: boot: ms a phase (the adapter, the tables and nets, then create's own, fh.boot), for the receiver's stats row.
  const t0 = performance.now(), boot = {};
  let adapter = await getAdapter();
  boot.adapter = Math.round(performance.now() - t0);
  const nmax = spec.blind ? spec.nmax : Math.max(1024, spec.n);
  const files = { ...INSTALLED, ...nets };
  const [tables, large, small] = await Promise.all([formatTables(), weightsOf(files.large), weightsOf(files.small)]);
  boot.tablesNets = Math.round(performance.now() - t0 - boot.adapter);
  const chain = prec ? [prec] : receiverChain(adapter);
  let fh;
  for (const [k, precision] of chain.entries()) {
    try {
      if (k) adapter = await getAdapter();
      fh = await FrontHalf.create(adapter, tables, {
        B: test === "b1" ? 1 : "auto", capMs: CAP_MS, inflight: test === "b1" ? 1 : "auto", restart, cap: CAP, register: true,
        weights: large, precision, subgroups,
        bank: "fcn2", cascade: test === "nocascade" ? null : { weights: small, keep: CASCADE.keep, rest: CASCADE.rest },
        back: "gpu", backN: nmax, fuse: true, log, probe, cancel, twins,
      });
      break;
    } catch (e) {
      if (e instanceof Unavailable) throw e;
      if (precision === "int8" && k < chain.length - 1) { log(`the int8 decoder did not build (${e?.message ?? e}); ${chain[k + 1]} instead`); continue; }
      throw new Unavailable(`the GPU decoder did not build: ${e?.message ?? e}`);
    }
  }
  // ai: The back half leaves a size unbuilt whose pass 1 overflows workgroup memory (gpu/back/back.mjs), and says so
  // ai: only in its log, so its rule is asked here: a 32 KB device without shader-f16
  // ai: builds no 1536 picture (pass 1 needs 33,792 B in f32), and LIZARD-576 and up stay on the CPU there.
  // ai: A blind spec decodes every picture the device builds, the rest found and their words read (since 2026-09-29, when
  // ai: the receiver began building to the ladder's top; refused whole before, when the top was a menu's choice); a told
  // ai: one must build its own picture.
  const nbig = spec.blind ? nmax : spec.n, need = pass1WorkgroupBytes(nbig, fh.backOpts.precision), room = fh.device.limits.maxComputeWorkgroupStorageSize;
  if (need > room && !spec.blind) { fh.device.destroy(); throw new Unavailable(`n = ${nbig} needs ${need} B of workgroup memory, this device offers ${room}`); }
  if (need > room) log(`n = ${nbig} needs ${need} B of workgroup memory, this device offers ${room}: its pictures are found and not decoded here`);
  const info = adapter.info ?? {};
  boot.create = fh.boot ?? null; boot.total = Math.round(performance.now() - t0);
  return { fh, nmax, nets: fh.nets ?? null, boot, adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unnamed adapter" };
}

// ai: The symbol's lattice corners in the frame's pixels, logical TL TR BR BL: where the C's quad is (src/ob.h, its
// ai: corner nodes). table: the ring the finder picked (gpu/tables.mjs rings); H: its map, modules to pixels.
export function quadOf(table, H) {
  const { nodeX: x, nodeY: y } = table, at = (u, v) => { const w = H[6] * u + H[7] * v + H[8]; return [(H[0] * u + H[1] * v + H[2]) / w, (H[3] * u + H[4] * v + H[5]) / w]; };
  const x0 = x[0], x1 = x[x.length - 1], y0 = y[0], y1 = y[y.length - 1];
  return [...at(x0, y0), ...at(x1, y0), ...at(x1, y1), ...at(x0, y1)];
}

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

// ai: dec: openGpu's result. reopen(restart): openGpu again, for the one recovery. Callbacks: done(items, out, dec), a
// ai: batch read back (out is gpu/decoder.mjs batch()'s result whole: out.frames[i] for items[i], its blocks of both
// ai: passes with the cancel stage, deviceMs the batch's device time, carried, host, idle and ingestMs its measurement
// ai: fields); drop(items), frames answered
// ai: without a decode; lost(reason), the queue has given up. An item is { w, h, luma }, or { w, h, source, x, y } for
// ai: F0 (a VideoFrame and the crop, lizard-web/vframes.mjs), plus whatever the caller keeps on it. Its luma is let go and
// ai: its source closed once enqueued, or once dropped (closeSource); the item keeps its ring slot (item.slot) until
// ai: its batch launches. held: items waiting for a ring; staged (queued, the worker's word): frames on the ring
// ai: awaiting a batch. setRate(fps): the camera's stated rate, from which the measured one starts. flush(): the
// ai: staged frames as a batch as soon as a lane can take them, whatever the fill (a fixed-B caller's only way to
// ai: launch a partial batch). open counts the sources not yet closed (each a camera buffer), openMax the most at once
// ai: since the last takeOpenMax().
export class GpuQueue {
  // ai: profile: every batch timed stage by stage (gpu/decoder.mjs batch() profile: a compute pass a stage, each with
  // ai: its own timestamps, so its out.stageMs and a deviceMs that is their sum).
  constructor(dec, { reopen, done, drop, lost, profile = false }) {
    Object.assign(this, { dec, reopen, done, drop, lost, profile });
    this.holding = []; this.staging = [];
    this.planning = null; this.computing = new Set(); this.timer = null; this.flushed = false;
    this.recovering = false; this.dead = false; this.recoveries = 0; this.stopping = null;
    this.iv = 1000 / 30; this.lastPush = 0;
    this.open = 0; this.openMax = 0;
    // ai: The held configuration (gpu/wgsl/word.mjs): the version of the last word the caller read, 0 for none. Every
    // ai: batch is launched with it; the caller sets it from each batch's words as they come back (recv-gpu-worker.mjs
    // ai: done), the one state a decode carries from batch to batch (2026-09-27).
    this.word = 0;
    this.watch(dec.fh);
  }

  get fh() { return this.dec.fh; }
  get held() { return this.holding.length; }
  get staged() { return this.staging.length; }
  get queued() { return this.staging.length; }
  get target() { return Math.max(1, this.fh.size | 0); }
  get fps() { return 1000 / this.iv; }
  // ai: A fixed B has no bound: it launches full, or on flush().
  get bound() { return this.fh.auto ? FILL * this.target * 1000 / this.fps : 0; }
  opts() { return { nmaxFind: this.dec.nmax, held: this.word }; }

  setRate(fps) { if (fps > 0) this.iv = 1000 / fps; }
  // ai: The high-water mark of open sources, then the mark starts again from what is open now.
  takeOpenMax() { const m = this.openMax; this.openMax = this.open; return m; }

  // ai: The arrival interval is an EMA of the gaps between pushes, each weighted by its share of RATE_TAU, so it spans
  // ai: about RATE_TAU whatever the frame rate and a burst of near-simultaneous pushes barely moves it; it starts from
  // ai: the stated rate (setRate, 30 by default). A gap past 1 / RATE_MIN counts as that long.
  push(item) {
    const now = performance.now();
    if (this.lastPush) { const dt = Math.min(now - this.lastPush, 1000 / RATE_MIN); this.iv += (dt - this.iv) * Math.min(1, dt / RATE_TAU); }
    this.lastPush = now;
    if (item.source) { this.open++; this.openMax = Math.max(this.openMax, this.open); }
    if (this.dead) return this.discard([item]);
    if (this.recovering) return this.hold(item);
    this.place(this.fh, item);
    this.maybeLaunch();
  }

  // ai: Onto the ring, or held while the ring is not there for it: none yet, a plan running, or a frame larger than
  // ai: the ring (under B auto the lanes are planned again and the ring with them; at a fixed B ready() grows the
  // ai: ring at once, so needsPlan answers for the ring alone there).
  place(fh, it) {
    if (this.planning) return this.hold(it);
    if (!fh.ring || fh.needsPlan([frameOf(it)], null, this.opts())) { this.hold(it); return this.plan(fh); }
    this.stage(fh, it);
  }

  hold(it) {
    this.holding.push(it);
    if (this.holding.length > HOLD_MAX) this.discard([this.holding.shift()]);
  }

  // ai: The lanes and the ring planned on the first held frame's shape (the plan reads nothing else since 2026-09-29:
  // ai: no calibration, the back half built at create; a luma copy of the frame before). The held frames then go onto
  // ai: the new ring. Frames staged on the ring about to be rebuilt are launched now when a lane can take them, else
  // ai: dropped.
  plan(fh) {
    if (this.staging.length && this.computing.size < fh.inflight && fh.lanes.some((l) => !l.busy)) this.launch(fh, this.staging.splice(0, this.target));
    this.unstage(fh, this.staging.splice(0));
    // ai: A plan failing on a device being recovered keeps the held frames: the new device plans on them.
    const p = this.readyOn(fh).then(() => { this.settled(p); this.resume(fh); }, (e) => { this.settled(p); this.failed(fh, fh === this.fh && !this.recovering ? this.holding.splice(0) : [], e); });
    this.planning = p;
  }

  async readyOn(fh) {
    const it = this.holding[0];
    if (!it || this.dead || fh !== this.fh) return;
    return fh.ready([{ w: it.w, h: it.h }], null, this.opts());
  }

  // ai: A plan that ended after a recovery replaced its decoder, or on one that is being recovered, moves nothing:
  // ai: the recovery plans again on what is held.
  settled(p) { if (this.planning === p) this.planning = null; }
  resume(fh) {
    if (this.dead || this.recovering || fh !== this.fh) return;
    for (const it of this.holding.splice(0)) this.place(fh, it);
    this.maybeLaunch();
  }

  // ai: The frame onto the ring, the oldest staged dropped while the ring is full, and its VideoFrame closed at once:
  // ai: the ingest is submitted when enqueue returns. enqueue throws only on a decoder fault (the worker checked the
  // ai: crop), which ends the GPU path as a failed batch does. With STAGE_BATCHES batches' worth already staged the
  // ai: device is behind the camera, and a new frame is dropped before it touches the device: on the S26 at a 2052
  // ai: crop (20 ms a frame, 3.5 an ingest, 50 frames a second) ingesting every arrival and dropping the oldest spent
  // ai: the GPU on frames never decoded, and a batch's readback landed 13 s after its submit (STATUS "The phone at the
  // ai: camera's rate"). Dropping the new frame costs nothing and the staged ones are at most two batches old.
  stage(fh, it) {
    if (this.staging.length >= STAGE_BATCHES * this.target) return this.discard([it]);
    let slot = null;
    try { while (!(slot = fh.enqueue(frameOf(it))) && this.staging.length) this.unstage(fh, [this.staging.shift()]); }
    catch (e) { return this.failed(fh, [it], e); }
    it.luma = null; this.closeSource(it);
    if (!slot) return this.discard([it]);
    it.slot = slot;
    this.staging.push(it);
  }

  // ai: A batch while fewer than fh.inflight compute and a lane is free: the oldest staged frames, at most fh.size,
  // ai: once the ring holds the target or the oldest has waited the bound. A timer to the bound is armed otherwise;
  // ai: when it fires into a full device the readback's own call here re-arms it. Two in flight hide the readback's
  // ai: round trip (18 to 68 ms on the S26 at 1080) behind the next batch's compute; the camera's buffers then wait
  // ai: behind up to two batches, which the S26's pool tolerates at batches of 8 (capturedFps 60, STATUS).
  maybeLaunch() {
    if (this.dead || this.recovering || this.planning || this.computing.size >= this.fh.inflight || !this.staging.length) return;
    const fh = this.fh;
    if (!fh.lanes.some((l) => !l.busy)) return;
    const waited = performance.now() - this.staging[0].slot.at, bound = this.bound;
    if (this.staging.length < this.target && !this.flushed && !(bound > 0 && waited >= bound)) return this.arm(bound - waited);
    this.launch(fh, this.staging.splice(0, this.target));
  }

  arm(ms) { if (this.timer || !(ms > 0)) return; this.timer = setTimeout(() => { this.timer = null; this.maybeLaunch(); }, ms); }
  disarm() { if (this.timer) { clearTimeout(this.timer); this.timer = null; } }
  flush() { this.flushed = true; this.maybeLaunch(); }

  // ai: A batch of the items' slots.
  launch(fh, items) {
    let p;
    try { p = fh.run(items.map((it) => it.slot), null, { ...this.opts(), ...(this.profile ? { profile: true } : {}) }); } catch (e) { return this.failed(fh, items, e); }
    this.flushed = false;
    const dec = this.dec;
    this.computing.add(p);
    // ai: Only its own batch is cleared: a lost device's batch may settle after a recovery has launched another.
    p.then((out) => this.done(items, out, dec), (e) => this.failed(fh, items, e))
      .finally(() => { this.computing.delete(p); this.maybeLaunch(); });
  }

  // ai: Frames answered without a decode, their sources closed; unstage also gives their slots back to the ring (in a
  // ai: try: a ring being rebuilt, or a lost device's, has nothing to give back to).
  discard(items) {
    if (!items.length) return;
    this.drop(items);
    for (const it of items) this.closeSource(it);
  }
  unstage(fh, items) {
    for (const it of items) { try { fh.release(it.slot); } catch {} }
    this.discard(items);
  }

  closeSource(it) { if (it.source) { it.source.close(); it.source = null; this.open--; } }

  // ai: A batch (or a plan) that failed: the device, if it went, is recovered once; anything else is a fault in the
  // ai: decoder, and the CPU pool is the known good path. A failure on a decoder already replaced only costs its frames.
  async failed(fh, items, e) {
    this.discard(items);
    if (fh !== this.fh || this.dead) return;
    const info = fh.lost ?? (await Promise.race([fh.device.lost, sleep(500).then(() => null)]));
    if (info) this.recover(fh, `${info.reason}: ${info.message}`, items.length || this.target);
    else this.giveUp(`a GPU batch failed: ${e?.message ?? e}`);
  }

  watch(fh) {
    fh.device.lost.then((info) => { if (info.reason !== "destroyed") this.recover(fh, `${info.reason}: ${info.message}`, this.target); });
  }

  // ai: As gpu/harness/page.mjs does: a new device, batches of at most half the one that was running. The ring went
  // ai: with the device, so the frames staged on it are dropped; the held ones (HOLD_MAX at most, more arriving
  // ai: meanwhile push the oldest out) wait for the new device and plan it.
  async recover(fh, reason, at) {
    if (fh !== this.fh || this.recovering || this.dead) return;
    if (this.recoveries++) return this.giveUp(`the device was lost a second time (${reason})`);
    this.recovering = true;
    this.disarm();
    this.discard(this.staging.splice(0));
    // ai: The lost device's batch and plan may never settle: neither may keep the new device from launching.
    this.computing.clear(); this.planning = null;
    try {
      this.dec = await this.reopen(Math.max(1, at >> 1));
    } catch (e) {
      this.recovering = false;
      return this.giveUp(`the device was lost (${reason}) and a new one failed: ${e?.message ?? e}`);
    }
    this.recovering = false;
    if (this.dead) { this.dec.fh.device.destroy(); return; }
    this.watch(this.dec.fh);
    if (this.holding.length) this.plan(this.dec.fh);
  }

  // ai: Every frame answered, then the device let go. The batches computing are waited for (and delivered), 2 s at
  // ai: most: a lost device's batch might never settle.
  // ai: One stop however often asked, so a caller that asks while a give-up is stopping waits for the same end.
  stop() {
    this.stopping ??= (async () => {
      this.dead = true;
      this.disarm();
      this.discard(this.holding.splice(0));
      this.unstage(this.fh, this.staging.splice(0));
      if (this.computing.size) await Promise.race([Promise.allSettled([...this.computing]), sleep(2000)]);
      this.fh.device.destroy();
    })();
    return this.stopping;
  }

  async giveUp(reason) {
    if (this.dead) return;
    await this.stop();
    this.lost(reason);
  }
}

const frameOf = (it) => (it.source ? { w: it.w, h: it.h, source: it.source, x: it.x, y: it.y } : { w: it.w, h: it.h, bytes: it.luma });
