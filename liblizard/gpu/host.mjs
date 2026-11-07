// ai: The GPU decoder on a WebGPU device: the native host (core/dec/setup.cpp Setup, core/dec/front.cpp FrontHalf and
// ai: Batcher) ported name for name, over the same generated tree (liblizard/gen/gen.mjs): a variant's setup
// ai: (setup/<variant>.json), its buffers' bytes (blobs/<hash>.bin) and its modules as WGSL (webgpu/<module>.wgsl,
// ai: gen's naga translation of the SPIR-V the native host loads). What a lane, a batch and its readback do is the
// ai: native host's; where the port differs, an ai: comment says so and why.
// ai: Not ported: the camera ingests and the develop (enqueueCamera, enqueueRaw: a browser hands the page a frame of
// ai: its own), the kept frames (run's keep, LIZ_DUMP, LIZ_DUMP_S), the int8 twins (the WebGPU tree is f32, which has
// ai: none), the LDPC forms' cross-check and the trial (a WebGPU device validates every module before it runs).

export const TREE_VERSION = 8;   // ai: front.cpp TREE_VERSION
export const WEBGPU_VARIANT = "f32";   // ai: gen.mjs WEBGPU_VARIANT: the variant gen translates to WGSL

// ai: front.cpp STAGES, the profile's stage groups, a timestamp pair each.
const STAGES = ["pyramid", "bank", "select", "describe", "vote", "score", "find2", "nodes", "refit", "word", "gate", "pass1", "reduce", "pass2", "soft", "ldpc"];
const MiB = 1 << 20;
const ceilDiv = (a, b) => Math.floor((a + b - 1) / b);
const U = GPUBufferUsage;

// ai: setup.cpp idsIn: every id a part of the tree names (strings like b12, L3, p40).
function idsIn(j, out = []) {
  if (typeof j === "string") { if (/^[bLp]\d+$/.test(j)) out.push(j); }
  else if (Array.isArray(j)) for (const v of j) idsIn(v, out);
  else if (j && typeof j === "object") for (const v of Object.values(j)) idsIn(v, out);
  return out;
}

// ai: A device and the bytes it holds: front.cpp's dev.allocated, which plan prices a lane by. Every buffer and
// ai: texture the host makes goes through here, and is destroyed with the object that owns it (a lane, a back half),
// ai: not left to the collector.
class Dev {
  constructor(device, log) {
    this.d = device;
    this.log = log;
    this.allocated = 0;
    this.limits = device.limits;
    this.timestamps = device.features.has("timestamp-query");
  }
  buffer(size, usage, label) {
    size = Math.max(4, Math.ceil(size / 4) * 4);
    const b = this.d.createBuffer({ size, usage, label });
    this.allocated += size;
    return b;
  }
  texture(w, h, layers, label) {
    const t = this.d.createTexture({ size: [w, h, layers], format: "r8unorm", dimension: "2d", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC, label });
    t.bytes = w * h * layers;
    this.allocated += t.bytes;
    return t;
  }
  free(o) {
    if (!o) return;
    this.allocated -= o.bytes ?? o.size;
    o.destroy();
  }
  // ai: a bind group over a layout, from a list of resources: a buffer (whole), a texture (its 2d-array view) or the
  // ai: sampler
  group(bgl, entries) {
    return this.d.createBindGroup({ layout: bgl, entries: entries.map((r, i) => ({ binding: i, resource: r instanceof GPUTexture ? r.createView({ dimension: "2d-array" }) : r instanceof GPUSampler ? r : { buffer: r } })) });
  }
  write(buf, off, data) { this.d.queue.writeBuffer(buf, off, data.buffer ?? data, data.byteOffset ?? 0, data.byteLength); }
}

// ai: setup.cpp Setup: one variant's tree, its objects made on the device by id, each once.
export class Setup {
  // ai: read(path): the bytes of a file under the generated tree (a Uint8Array or an ArrayBuffer, or a promise of one)
  static async load(dev, read, variant) {
    const s = new Setup();
    s.dev = dev;
    s.read = async (p) => { const b = await read(p); return b instanceof Uint8Array ? b : new Uint8Array(b); };
    s.variant = variant;
    s.tree = JSON.parse(new TextDecoder().decode(await s.read(`setup/${variant}.json`)));
    s.objects = s.tree.objects;
    delete s.tree.objects;
    s.buffers = new Map(); s.bgls = new Map(); s.pipelines = new Map(); s.unbuilt = new Set();
    s.modules = new Map();   // ai: a module's GPUShaderModule by its hash, made once
    s.backB = 0;
    // ai: the device's one linear sampler, which every layout's `sampler` entry binds (wg.cpp's immutable sampler:
    // ai: linear, clamped, no mipmaps)
    s.sampler = dev.d.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge", lodMaxClamp: 0 });
    return s;
  }
  back() { return this.tree.backs[String(this.backB)]; }
  layoutOf(id) {
    const ent = (t, i) => {
      const e = { binding: i, visibility: GPUShaderStage.COMPUTE };
      if (t === "tex") e.texture = { sampleType: "float", viewDimension: "2d-array" };
      else if (t === "ro") e.buffer = { type: "read-only-storage" };
      else if (t === "rw") e.buffer = { type: "storage" };
      else if (t === "uniform") e.buffer = { type: "uniform" };
      else if (t === "sampler") e.sampler = { type: "filtering" };
      else throw new Error(`bind group layout entry ${t}`);
      return e;
    };
    return this.dev.d.createBindGroupLayout({ label: id, entries: this.objects.bgls[id].map(ent) });
  }
  // ai: setup.cpp make: the objects named, layouts first; the pipelines compiled together (createComputePipelineAsync,
  // ai: each its own promise: no error scope of the device's in between), a pipeline the device cannot hold left out
  async make(ids) {
    const compiles = [];
    const order = (c) => (c === "L" ? 0 : c === "p" ? 1 : 2);
    for (const id of [...ids].sort((a, b) => order(a[0]) - order(b[0]))) {
      const k = id[0];
      if (k === "L" && !this.bgls.has(id)) this.bgls.set(id, this.layoutOf(id));
      else if (k === "p" && !this.pipelines.has(id) && !this.unbuilt.has(id)) {
        const o = this.objects.pipelines[id];
        if (o.groups.length !== 1) throw new Error(`${id}: ${o.groups.length} bind groups; the decoder uses one`);
        const g = o.groups[0];
        if (!this.bgls.has(g)) this.bgls.set(g, this.layoutOf(g));
        const lanes = o.lanes ?? 0, label = o.label ?? "";
        if (lanes > this.dev.limits.maxComputeInvocationsPerWorkgroup) { this.dev.log(`pipeline ${id} (${label}) left out: ${lanes} lanes a workgroup, the device ${this.dev.limits.maxComputeInvocationsPerWorkgroup}`); this.unbuilt.add(id); continue; }
        // ai: a module that needs a feature WebGPU does not offer ("int16": the LDPC's reg+i16 forms) has no WGSL (gen.mjs
        // ai: webgpu.not); left out as setup.cpp leaves it out on a device without the feature
        if ((o.needs ?? []).length) { this.dev.log(`pipeline ${id} (${label}) left out: it needs ${o.needs.join(", ")}, which WebGPU does not offer`); this.unbuilt.add(id); continue; }
        this.pipelines.set(id, null);
        compiles.push((async () => {
          const module = await this.module(o.module, label);
          const t0 = performance.now();
          try {
            const p = await this.dev.d.createComputePipelineAsync({ label: `${id} ${label}`, layout: this.dev.d.createPipelineLayout({ bindGroupLayouts: [this.bgls.get(g)] }), compute: { module, entryPoint: o.entry ?? "main" } });
            this.pipelines.set(id, p);
          } catch (e) {
            const msgs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
            throw new Error(`pipeline ${id} (${label}, module ${o.module}): ${msgs.join("; ") || e.message}`);
          }
          const ms = performance.now() - t0;
          if (ms > 2000) this.dev.log(`pipeline ${id} (${label}, module ${o.module}): ${Math.round(ms)} ms to compile`);
        })());
      } else if (k === "b" && !this.buffers.has(id)) {
        const o = this.objects.buffers[id];
        const b = this.dev.buffer(o.size, o.usage, id);
        this.buffers.set(id, b);
        if (typeof o.blob === "string") {
          compiles.push((async () => {
            const data = await this.read(`blobs/${o.blob}.bin`);
            if (data.byteLength !== o.size) throw new Error(`${id}: blob of ${data.byteLength} B for a buffer of ${o.size}`);
            // ai: writeBuffer takes whole words: a blob of another length is padded (the buffer is, Dev.buffer)
            const padded = data.byteLength % 4 ? (() => { const p = new Uint8Array(Math.ceil(data.byteLength / 4) * 4); p.set(data); return p; })() : data;
            this.dev.write(b, 0, padded);
          })());
        }
      }
    }
    await Promise.all(compiles);
  }
  async module(hash, label) {
    if (!this.modules.has(hash)) this.modules.set(hash, (async () => {
      const code = new TextDecoder().decode(await this.read(`webgpu/${hash}.wgsl`));
      return this.dev.d.createShaderModule({ label: `${hash} ${label}`, code });
    })());
    return this.modules.get(hash);
  }
  async build(B) {
    const t0 = performance.now();
    await this.make([...idsIn(this.tree.front), ...idsIn(this.tree.anch)]);
    await this.buildBack(B);
    this.buildMs = performance.now() - t0;
  }
  async buildBack(B) {
    if (!this.tree.backs[String(B)]) throw new Error(`no back half at B = ${B} in the setup`);
    // ai: the previous back half's objects go (the front half's stay)
    if (this.backB && this.backB !== B) {
      for (const id of idsIn(this.back())) {
        if (this.buffers.has(id)) { this.dev.free(this.buffers.get(id)); this.buffers.delete(id); }
        this.pipelines.delete(id);
      }
    }
    this.backB = B;
    await this.make(idsIn(this.back()));
  }
  buf(id) { const b = this.buffers.get(id); if (!b) throw new Error(`setup buffer ${id} not built`); return b; }
  bgl(id) { const b = this.bgls.get(id); if (!b) throw new Error(`setup layout ${id} not built`); return b; }
  pipe(id) { const p = this.pipelines.get(id); if (!p) throw new Error(`setup pipeline ${id} not built`); return p; }
  built(id) { return !!this.pipelines.get(id); }
  // ai: a tree's { p, bgl } (null gives null), with a classifier's patches a workgroup
  pair(pb) { return pb == null ? null : { p: this.pipe(pb.p), bgl: this.bgl(pb.bgl), P: pb.P ?? 1 }; }
}

// ai: front.h Batcher: lanes sized by memory for Bmax; the stream starts at Bmax; a batch over the cap halves the size,
// ai: three over 90% halve it, sixteen with room for a double grow it back toward Bmax; a lane's first batch is
// ai: watched for the cap alone.
class Batcher {
  constructor(fh) { Object.assign(this, { fh, budget: 0, capMs: 400, size: 0, ceiling: 0, Bmax: 0, lanes: 2, inflight: 2, perFrame: 0, shared: 0, room: 0, slotBytes: 0, planMs: 0, cap: 0, capWhy: "", bound: "", BmaxWhy: "", over: 0, spare: 0 }); }
  log(m) { this.fh.log("batch auto: " + m); }
  limitFor(n) {
    const C = this.fh.C, slot = this.slotBytes;
    let Bm = Math.floor((this.room - C.RING_SLACK * slot) / (n * this.perFrame + slot));
    let why = `memory: ${n} lanes of ${Bm} and a ring of ${Bm + C.RING_SLACK} fill the ${(this.budget / MiB).toFixed(1)} MB budget (a frame ${(this.perFrame / MiB).toFixed(1)} MB a lane and ${(slot / MiB).toFixed(1)} MB on the ring, ${(this.shared / MiB).toFixed(1)} MB shared)`;
    if (this.cap < Bm) { Bm = this.cap; why = this.capWhy; }
    if (Bm > C.B_CEIL) { Bm = C.B_CEIL; why = `the ceiling of ${C.B_CEIL} a batch (past it there is nothing left to amortise)`; }
    if (Bm < 1) { Bm = 1; why = "memory: batches of 1"; }
    const limit = Bm;
    while (Bm > 1 && !this.fh.S.tree.backs[String(Bm)]) Bm--;
    if (Bm < limit) why += `; the setup's back halves round it to ${Bm}`;
    return [Bm, why];
  }
  laned(n, B, why) {
    this.lanes = n; this.Bmax = B; this.BmaxWhy = why;
    if (this.size > this.Bmax) { this.size = this.Bmax; this.bound = why; }
    if (this.ceiling > this.Bmax) this.ceiling = this.Bmax;
  }
  start() {
    const prev = this.size, best = prev > 0 ? Math.min(prev, this.Bmax) : this.Bmax;
    this.bound = best < this.Bmax ? `${best}, the size the watch had come to before this plan` : this.BmaxWhy;
    this.size = best; this.ceiling = this.Bmax; this.over = 0; this.spare = 0;
    this.fh.log(`batch auto: ${best} frames a batch (lanes of ${this.Bmax}, ${Math.round(this.perFrame * this.Bmax / MiB)} MB each; ${this.bound})`);
  }
  resize(n, why) {
    if (n === this.size) return;
    this.log(`${this.size} to ${n} frames: ${why}`);
    this.size = n; this.over = 0; this.spare = 0;
  }
  observe(frames, ms, first) {
    if (!this.size) return;
    const s = this.size, cap = this.capMs;
    if (ms > cap) this.resize(Math.max(1, s >> 1), `a batch of ${frames} took ${Math.round(ms)} ms, over the ${cap} ms cap`);
    else if (first) return;
    else if (ms > 0.9 * cap && frames >= s) { if (++this.over >= 3) this.resize(Math.max(1, s >> 1), `three batches running over 90% of the ${cap} ms cap (${Math.round(ms)} ms)`); }
    else this.over = 0;
    if (this.size === s && s < this.ceiling && frames >= s && ms * 2 <= cap) { if (++this.spare >= 16) this.resize(Math.min(this.ceiling, 2 * s), `sixteen batches running with room for a double (${Math.round(ms)} ms)`); }
    else this.spare = 0;
  }
}

class OutOfMemory extends Error {}

// ai: front.cpp FrontHalf: the lanes, the ring, a batch recorded and submitted by run() and read back by finish().
export class FrontHalf {
  // ai: device: a GPUDevice made with the adapter's limits (requiredLimitsFor) and timestamp-query where offered;
  // ai: read(path): the generated tree's files; budget: device memory the decoder may hold (default the device's
  // ai: largest buffer, as the web host had it)
  static async create(device, read, { variant = WEBGPU_VARIANT, budget, log = () => {}, ldpcForm } = {}) {
    const fh = new FrontHalf();
    fh.log = log;
    fh.dev = new Dev(device, log);
    fh.variant = variant;
    const t0 = performance.now();
    fh.S = await Setup.load(fh.dev, read, variant);
    const T = fh.S.tree;
    if ((T.treeVersion ?? 0) !== TREE_VERSION) throw new Error(`the setup is tree version ${T.treeVersion}, this host reads ${TREE_VERSION} (liblizard/gen/gen.mjs on another tree): generate it again`);
    fh.C = { ...T.consts };
    fh.nets = T.nets; fh.cap = T.cap; fh.nodesMax = T.nodesMax; fh.bankKeep = T.bankKeep;
    fh.coarseFrom = T.front.coarseFrom;
    fh.modules = new Float32Array(T.modules.slice(0, 4));
    await fh.S.build(fh.C.B_CEIL);
    const A = T.anch;
    fh.anchFind = fh.S.pair(A.find); fh.anchFit = fh.S.pair(A.fit); fh.anchRefit = fh.S.pair(A.refit);
    fh.anchPredict = fh.S.pair(A.predict); fh.anchNodes = fh.S.pair(A.nodes);
    fh.anchUni = fh.S.buf(A.buffers.anchUni);
    fh.backShared = 0;
    fh.classify = fh.P("classify");
    fh.weightsBuf = fh.FBP("weightsBuf");
    if (T.front.twins != null) fh.log("the tree's int8 twins are not measured here: the float classifier");
    // ai: the LDPC kernel's forms (front.cpp create): every form built is timed on the first batch with enough
    // ai: codewords and the fastest kept; ldpcForm: form k, untimed (LIZ_LDPC_FORM's)
    const rule = fh.S.back().rule;
    const forms = 1 + (rule.ldpcAlts?.length ?? 0);
    fh.ldpcBuilt = [];
    for (let k = 0; k < forms; k++) {
      const ok = (k ? rule.ldpcAlts[k - 1] : rule.ldpc).every((st) => fh.S.built(st.pipeline));
      fh.ldpcBuilt.push(ok);
      if (!ok) fh.log(`LDPC form ${rule.ldpcForms[k]} left out: a kernel WebGPU cannot build`);
    }
    fh.ldpcForm = 0; fh.ldpcPending = false;
    if (ldpcForm != null) { const k = Math.max(0, Math.min(forms - 1, ldpcForm)); fh.ldpcForm = fh.ldpcBuilt[k] ? k : 0; }
    else if (fh.ldpcBuilt.filter(Boolean).length > 1) fh.ldpcPending = true;
    fh.bt = new Batcher(fh);
    fh.bt.budget = budget ?? device.limits.maxBufferSize;
    fh.bt.capMs = fh.C.CAP_MS;
    fh.lanes = []; fh.ring = null; fh.planned = null; fh.B = 0; fh.lastDone = 0;
    fh.log(`setup ${variant} built in ${Math.round(performance.now() - t0)} ms: nets bank ${fh.nets.bank}, classifier ${fh.nets.large}`);
    return fh;
  }

  P(k) { return this.S.pair(this.S.tree.front[k]); }
  FBP(k) { return this.S.buf(this.S.tree.front.buffers[k]); }
  now() { return performance.now(); }
  size() { return this.bt.size; }
  inflightMax() { return this.bt.inflight; }

  // ai: front.cpp backBytesOf: the back half's static bytes, each buffer once
  backBytes() {
    let n = 0;
    const seen = new Set();
    for (const id of idsIn(this.S.back())) if (id[0] === "b" && this.S.buffers.has(id) && !seen.has(id)) { seen.add(id); n += this.S.buffers.get(id).size; }
    return n;
  }
  async buildBack(B) { await this.S.buildBack(B); this.backShared = this.backBytes(); }

  // ai: every buffer and texture of a lane (and its back lane), destroyed with it
  dropLane(ln) { for (const o of ln.own) this.dev.free(o); ln.own = []; }
  mk(ln, size, usage, label) { const b = this.dev.buffer(size, usage, label); ln.own.push(b); return b; }

  // ai: front.cpp lane(B): the buffers a batch writes, sized for B frames, and the groups that do not depend on the
  // ai: frame's size.
  lane(Bl) {
    const C = this.C, cap = this.cap, B = Bl, d = this.dev;
    const ln = { B: Bl, own: [], busy: false, watched: false, W: 0, H: 0, level: [], fcn2Levels: [], back: null };
    const a0 = d.allocated;
    ln.slotsG = Math.max(Bl, this.S.backB);
    const mk = (size, usage) => this.mk(ln, size, usage);
    ln.peaksBuf = mk(16 * cap * B, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.selUni = mk(16, U.UNIFORM | U.COPY_DST);
    ln.selUni2 = mk(16, U.UNIFORM | U.COPY_DST);
    ln.framesBuf = mk(16 * ln.slotsG, U.STORAGE | U.COPY_DST);
    ln.mapsBuf = mk(64 * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.countsBuf = mk(4 * C.COUNTERS * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.readingsBuf = mk(48 * cap * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.candBuf = mk(16 * C.K * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.quadsBuf = mk(16 * 5 * C.K * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.vlistBuf = mk(16 * C.PEAK_LIST * B, U.STORAGE);
    ln.vcountBuf = mk(4 * B, U.STORAGE | U.COPY_DST);
    ln.args2Buf = mk(12 * C.R2_N, U.STORAGE | U.INDIRECT | U.COPY_DST);
    ln.r2Uni = mk(16 * C.R2_N, U.UNIFORM | U.COPY_DST);
    ln.scoreBuf = mk(4 * C.HYP * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.selBuf = mk(4 * C.SEL_WORDS * ln.slotsG, U.STORAGE | U.COPY_DST);
    ln.resultBuf = mk(4 * C.RESULT * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.heldUni = mk(16, U.UNIFORM | U.COPY_DST);
    ln.p5Uni = mk(32, U.UNIFORM | U.COPY_DST);
    ln.ldUni = mk(80, U.UNIFORM | U.COPY_DST);
    ln.measBuf = mk(16 * C.NODES_MAX * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.residBuf = mk(16 * C.RESID * B, U.STORAGE | U.COPY_DST | U.COPY_SRC);
    ln.anchBuf = mk(16 * C.ANCH_K * B, U.STORAGE | U.COPY_SRC);
    ln.arecBuf = mk(16 * C.ANCH_REC * B, U.STORAGE | U.COPY_SRC);
    const refit = this.P("refit"), pick = this.P("pick");
    const fit = [ln.framesBuf, this.FBP("nodesBuf"), ln.selBuf, ln.measBuf, ln.mapsBuf, ln.residBuf, this.FBP("fitUni")];
    ln.refitGroup = d.group(refit.bgl, fit);
    ln.refitPostGroup = d.group(this.P("refitPost").bgl, fit);
    ln.anchRefitGroup = d.group(this.anchRefit.bgl, [...fit, ln.anchBuf, this.FBP("wordLatUni"), this.anchUni]);
    ln.pickGroup = d.group(pick.bgl, [ln.framesBuf, ln.quadsBuf, ln.scoreBuf, ln.mapsBuf, ln.selBuf, ln.resultBuf, this.FBP("pickUni")]);
    ln.frames2Buf = mk(16 * B, U.STORAGE);
    ln.pickGroup2 = d.group(pick.bgl, [ln.frames2Buf, ln.quadsBuf, ln.scoreBuf, ln.mapsBuf, ln.selBuf, ln.resultBuf, this.FBP("pickUni")]);
    if (d.timestamps) {
      ln.qs = d.d.createQuerySet({ type: "timestamp", count: C.TS_PLAIN });
      ln.qsBuf = mk(8 * C.TS_PLAIN, U.QUERY_RESOLVE | U.COPY_SRC);
    }
    ln.laneBytes = d.allocated - a0;
    return ln;
  }

  lvEntries(ln, second = false) { return [ln.tex, ...ln.level.map((l) => l.buf), second ? ln.frames2Buf : ln.framesBuf, ln.ldUni]; }
  netGroup(ln, pipe, buf, second = false) { return this.dev.group(pipe.bgl, [...this.lvEntries(ln, second), ln.peaksBuf, ln.countsBuf, ln.readingsBuf, this.FBP("p4Uni"), buf]); }
  bindNets(ln) { ln.classifyGroup = this.netGroup(ln, this.classify, this.weightsBuf); ln.classifyGroup2 = this.netGroup(ln, this.classify, this.weightsBuf, true); }

  // ai: front.cpp backLane: the transform's, the soft stage's and the LDPC's lane buffers, the gate's group and the
  // ai: rate profile's steps' groups.
  backLane(ln) {
    const d = this.dev, C = this.C, S = this.S, bk = S.back(), T = bk.transform, so = bk.soft, ld = bk.ldpc;
    const Bb = bk.B, cb = T.cbytes;
    const a0 = d.allocated;
    const mk = (size, usage) => this.mk(ln, size, usage);
    const b = {};
    const listsWords = C.SLOTS * (1 + Bb) + 6 * Bb;
    b.lists = mk(4 * listsWords, U.STORAGE | U.COPY_SRC);
    b.args = mk(4 * C.argsWords, U.STORAGE | U.INDIRECT | U.COPY_SRC);
    b.part = mk(16 * T.partStride * Bb, U.STORAGE | U.COPY_SRC);
    b.coef = mk(32 * Bb, U.STORAGE | U.COPY_SRC);
    b.y = mk(cb * T.yStride * Bb, U.STORAGE | U.COPY_SRC);
    b.s = mk(cb * T.sStride * Bb, U.STORAGE | U.COPY_SRC);
    b.counts = mk(4 * 8 * Bb, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    b.params = []; b.red = []; b.p2 = []; b.p1f = []; b.soft = [];
    for (const s of T.built) {
      const tb = T.sizes[s];
      const p = mk(48, U.UNIFORM | U.COPY_DST);
      const u = new Uint32Array(12), f = new Float32Array(u.buffer);
      u[0] = 0; u[1] = T.yStride; u[2] = T.partStride; u[3] = s; u[4] = 0; u[5] = 0;
      f[6] = tb.m2; f[7] = tb.sx2; f[8] = tb.sq2; u[9] = T.sStride;
      d.write(p, 0, u);
      b.params[s] = p;
      b.red[s] = d.group(S.bgl(T.redBgl), [b.lists, b.part, b.coef, p]);
      b.p2[s] = d.group(S.bgl(T.p2Bgl), [b.lists, S.buf(T.twBufs[s]), b.part, b.y, b.coef, S.buf(T.x12Bufs[s]), S.buf(T.rowsBufs[s]), b.s, b.counts, p]);
    }
    b.L = mk(so.bytes.L, U.STORAGE | U.COPY_SRC);
    b.BLK = mk(so.bytes.BLK, U.STORAGE | U.COPY_SRC);
    b.PILOT = mk(so.bytes.PILOT, U.STORAGE | U.COPY_SRC);
    b.V = mk(ld.bytes.V, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    b.ITS = mk(ld.bytes.ITS, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    b.REC = mk(ld.bytes.REC, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    b.gate = d.group(S.bgl(T.gateBgl), [ln.framesBuf, ln.selBuf, b.lists, b.args, S.buf(T.gateUni), b.counts, b.V, b.ITS, b.REC]);
    const st = bk.rule;
    b.EP = mk(st.bytes.EP, U.STORAGE | U.COPY_SRC);
    const laneBufs = { S: b.s, LISTS: b.lists, L: b.L, EP: b.EP, BLK: b.BLK, BCOUNTS: b.counts, PILOT: b.PILOT, V: b.V, ITS: b.ITS, REC: b.REC };
    const group = (step) => d.group(S.bgl(step.bgl), step.binds.map((id) => (id.startsWith("lane:") ? laneBufs[id.slice(5)] : S.buf(id))));
    b.rsoft = st.soft.map(group);
    b.rldpc = st.ldpc.map(group);
    b.bytes = d.allocated - a0;
    ln.back = b;
  }

  // ai: front.cpp bindPicture: the fused pass 1 over the lane's texture (and the sampler), again whenever it is replaced
  bindPicture(ln) {
    const T = this.S.back().transform, b = ln.back;
    for (const s of T.built) b.p1f[s] = this.dev.group(this.S.bgl(T.p1fBgl), [ln.tex, b.lists, this.S.buf(T.twBufs[s]), b.part, b.y, b.counts, b.params[s], ln.mapsBuf, ln.residBuf, this.S.buf(T.picUnis[s]), ln.arecBuf, this.S.sampler]);
    b.picture = ln.tex;
  }

  // ai: front.cpp ensure(W, H): capacity for frames up to W x H; grows, never shrinks.
  ensure(ln, W, H) {
    const C = this.C, d = this.dev, B = ln.B, cap = this.cap;
    const a0 = d.allocated;
    const mk = (size, usage) => this.mk(ln, size, usage);
    let texChanged = false;
    if (W > ln.W || H > ln.H) {
      ln.W = Math.max(W, ln.W); ln.H = Math.max(H, ln.H);
      // ai: what the grown lane replaces goes (the native's shared pointers let go of it)
      for (const o of [ln.tex, ...ln.level.map((l) => l.buf), ln.rawBuf, ln.accBuf, ...ln.fcn2Levels.flatMap((l) => [l.plane, l.uni])]) if (o) { d.free(o); ln.own = ln.own.filter((x) => x !== o); }
      ln.tex = d.texture(ln.W, ln.H, B); ln.own.push(ln.tex);
      texChanged = true;
      ln.level = [];
      for (let l = 1; l < C.LEVELS; l++) { const w = ceilDiv(ln.W, 1 << l), h = ceilDiv(ln.H, 1 << l); ln.level.push({ w, h, buf: mk(4 * w * h * B, U.STORAGE) }); }
      const dimsAt = (l) => (l === 0 ? [ln.W, ln.H] : [ln.level[l - 1].w, ln.level[l - 1].h]);
      let slots = 0;
      const bases = [];
      for (let l = 0; l < C.LEVELS; l++) { const [w, h] = dimsAt(l); bases.push(slots); if (l > 0) slots += ceilDiv(w, 16) * ceilDiv(h, 16) * this.bankKeep; }
      ln.slots = slots;
      ln.rawBuf = mk(16 * slots * B, U.STORAGE | U.COPY_DST);
      {
        const u = new Uint32Array([slots, cap, 0, bases[this.coarseFrom]]);
        new Float32Array(u.buffer, 8, 1)[0] = C.TAU;
        d.write(ln.selUni, 0, u);
        u[3] = bases[1];
        d.write(ln.selUni2, 0, u);
      }
      const sel = this.P("select");
      ln.selGroup = d.group(sel.bgl, [ln.framesBuf, ln.rawBuf, ln.peaksBuf, ln.countsBuf, ln.selUni]);
      ln.selGroup2 = d.group(sel.bgl, [ln.frames2Buf, ln.rawBuf, ln.peaksBuf, ln.countsBuf, ln.selUni2]);
      const form = this.S.tree.front.fcn2Form;
      const xplane = form.xplane, int8 = form.int8, store = form.store;
      const fcn2 = this.P("fcn2"), fcn2X = this.P("fcn2X");
      ln.fcn2Levels = [];
      for (let l = 1; l < C.LEVELS; l++) {
        const [w, h] = dimsAt(l);
        const uni = mk(32, U.UNIFORM | U.COPY_DST);
        const u = new Uint32Array([l, w, h, slots, 0, 0, bases[l], ceilDiv(w, C.TILE)]), f = new Float32Array(u.buffer);
        f[4] = C.TAU; f[5] = C.EPS;
        d.write(uni, 0, u);
        const plane = xplane ? mk(int8 ? ((w + 3) & ~3) * h * B : (store === "f32" ? 4 : 2) * (w + (w & 1)) * h * B, U.STORAGE) : null;
        const group = (frames) => d.group(fcn2.bgl, [plane ?? ln.level[l - 1].buf, frames, ln.rawBuf, ln.countsBuf, uni]);
        const xgroup = (frames) => (plane ? d.group(fcn2X.bgl, [ln.level[l - 1].buf, frames, plane, uni]) : null);
        const fine = l < this.coarseFrom;
        ln.fcn2Levels.push({ l, w, h, group: group(ln.framesBuf), xgroup: xgroup(ln.framesBuf), group2: fine ? group(ln.frames2Buf) : null, xgroup2: fine ? xgroup(ln.frames2Buf) : null, plane, uni });
      }
      {
        const ld = new Uint32Array(20);
        ln.level.forEach((lv, i) => { ld[4 * (i + 1)] = lv.w; ld[4 * (i + 1) + 1] = lv.h; });
        d.write(ln.ldUni, 0, ld);
      }
      ln.gx = ceilDiv(ln.W, C.CELL); ln.gy = ceilDiv(ln.H, C.CELL);
      ln.accBuf = mk(4 * C.RING_COUNT * ln.gx * ln.gy * B, U.STORAGE | U.COPY_DST);
      {
        const p5 = new Uint32Array(8);
        p5.set([cap, ln.gx, ln.gy, 0]);
        new Float32Array(p5.buffer, 16, 4).set(this.modules);
        d.write(ln.p5Uni, 0, p5);
      }
      const lv = this.lvEntries(ln), lv2 = this.lvEntries(ln, true);
      const group = (pl, extra, withLevels) => d.group(pl.bgl, [...(withLevels ? lv : []), ...extra]);
      this.bindNets(ln);
      ln.scoreGroup = group(this.P("score"), [ln.quadsBuf, this.FBP("planBuf"), ln.scoreBuf, this.FBP("planUni")], true);
      ln.traceGroup = group(this.P("trace"), [ln.quadsBuf], true);
      ln.voteGroup = group(this.P("vote"), [ln.framesBuf, ln.readingsBuf, ln.accBuf, ln.p5Uni], false);
      ln.peaksGroup = group(this.P("peaks"), [ln.framesBuf, ln.accBuf, ln.vlistBuf, ln.vcountBuf, ln.p5Uni], false);
      ln.gatherGroup = group(this.P("gather"), [ln.framesBuf, ln.readingsBuf, ln.accBuf, ln.candBuf, ln.quadsBuf, ln.p5Uni, ln.vlistBuf, ln.vcountBuf], false);
      ln.scoreGroup2 = d.group(this.P("score").bgl, [...lv2, ln.quadsBuf, this.FBP("planBuf"), ln.scoreBuf, this.FBP("planUni")]);
      ln.voteGroup2 = group(this.P("vote"), [ln.frames2Buf, ln.readingsBuf, ln.accBuf, ln.p5Uni], false);
      ln.peaksGroup2 = group(this.P("peaks"), [ln.frames2Buf, ln.accBuf, ln.vlistBuf, ln.vcountBuf, ln.p5Uni], false);
      ln.gatherGroup2 = group(this.P("gather"), [ln.frames2Buf, ln.readingsBuf, ln.accBuf, ln.candBuf, ln.quadsBuf, ln.p5Uni, ln.vlistBuf, ln.vcountBuf], false);
      ln.completeGroup2 = d.group(this.P("complete").bgl, [...lv2, ln.readingsBuf, ln.quadsBuf, ln.p5Uni]);
      ln.traceGroup2 = d.group(this.P("trace").bgl, [...lv2, ln.quadsBuf]);
      ln.redoGroup = group(this.P("redo"), [ln.framesBuf, ln.selBuf, ln.frames2Buf, ln.accBuf, ln.readingsBuf, ln.candBuf, ln.quadsBuf, ln.p5Uni, ln.args2Buf, ln.r2Uni], false);
      ln.nodesGroup = group(this.P("nodesP"), [this.FBP("nodesBuf"), this.FBP("tmplBuf"), ln.mapsBuf, ln.selBuf, ln.measBuf, this.S.buf(this.S.tree.front.round)], true);
      ln.anchNodesGroup = group(this.P("nodesP"), [this.FBP("nodesBuf"), this.FBP("tmplBuf"), ln.mapsBuf, ln.selBuf, ln.measBuf, this.S.buf(this.S.tree.anch.buffers.round)], true);
      ln.wordGroup = group(this.P("wordP"), [ln.mapsBuf, ln.selBuf, ln.residBuf, this.FBP("wordPtsBuf"), ln.resultBuf, this.FBP("wordUni"), this.FBP("wordLatUni"), ln.heldUni], true);
      {
        const head = [ln.tex, ln.framesBuf, ln.mapsBuf, ln.selBuf, ln.residBuf, this.FBP("wordLatUni"), this.anchUni];
        ln.anchFindGroup = d.group(this.anchFind.bgl, [...head, ln.anchBuf]);
        ln.anchFitGroup = d.group(this.anchFit.bgl, [...head, ln.anchBuf, ln.arecBuf, ln.resultBuf]);
      }
      ln.pyrGroup = d.group(this.P("pyramid").bgl, [ln.tex, ln.framesBuf, ln.level[0].buf, ln.level[1].buf, ln.level[2].buf, ln.level[3].buf, ln.ldUni]);
    }
    if (!ln.back) { this.backLane(ln); texChanged = true; }
    if (texChanged || ln.back.picture !== ln.tex) this.bindPicture(ln);
    if (!ln.readBuf) {
      const bytes = 4 * C.COUNTERS * B + 4 * C.RESULT * B + this.S.back().readBytes + 8 * C.TS_MAX;
      ln.readBuf = mk(bytes, U.MAP_READ | U.COPY_DST);
    }
    ln.laneBytes += d.allocated - a0;
  }

  // ai: a lane made and grown under an out-of-memory error scope: the native's OutOfMemory throw (WebGPU reports a
  // ai: refused allocation on the scope, not at the call)
  async guarded(fn) {
    this.dev.d.pushErrorScope("out-of-memory");
    let r, err;
    try { r = fn(); } catch (e) { err = e; }
    const oom = await this.dev.d.popErrorScope();
    if (err) throw err;
    if (oom) throw new OutOfMemory(oom.message);
    return r;
  }

  // ai: front.cpp plan: the lanes planned for every shape seen so far and this one, a frame priced with a one-frame
  // ai: lane, the batch size set from the budget and the device's limits.
  async plan(w, h) {
    const t0 = this.now();
    const p = this.planned ?? { W: 0, H: 0 };
    const W = Math.max(p.W, w), H = Math.max(p.H, h);
    for (const l of this.lanes) this.dropLane(l);
    this.lanes = [];
    this.destroyRing();
    const L = this.dev.limits;
    for (;;) {
      const Bb = this.S.backB;
      const front = this.dev.allocated - this.backShared;
      let probe;
      try {
        probe = await this.guarded(() => { const ln = this.lane(1); try { this.ensure(ln, W, H); } catch (e) { this.dropLane(ln); throw e; } return ln; });
      } catch (e) {
        if (!(e instanceof OutOfMemory)) throw e;
        if (Bb === 1) throw new Error(`out of memory for a one-frame lane: ${e.message}`);
        await this.buildBack(Bb >> 1);
        continue;
      }
      const readBack = this.S.back().readBytes;
      const backBytes = probe.back.bytes + readBack;
      const perFrame = probe.laneBytes - backBytes + backBytes / Bb;
      let largest = 0;
      for (const b of [probe.peaksBuf, probe.framesBuf, probe.mapsBuf, probe.countsBuf, probe.readingsBuf, probe.candBuf, probe.quadsBuf, probe.scoreBuf, probe.selBuf, probe.resultBuf, probe.measBuf, probe.residBuf, probe.rawBuf, probe.accBuf, probe.anchBuf, probe.arecBuf]) if (b) largest = Math.max(largest, b.size);
      for (const l of probe.level) largest = Math.max(largest, l.buf.size);
      for (const l of probe.fcn2Levels) if (l.plane) largest = Math.max(largest, l.plane.size);
      largest = Math.max(largest, W * H);
      largest = Math.max(largest, probe.readBuf.size - readBack + readBack / Bb);
      const bk = probe.back;
      for (const b of [bk.lists, bk.args, bk.part, bk.coef, bk.y, bk.s, bk.counts, bk.L, bk.BLK, bk.V, bk.ITS, bk.REC, bk.PILOT]) largest = Math.max(largest, b.size / Bb);
      this.dropLane(probe);
      let capB = 2 ** 31 - 1, capWhy = "";
      const byBuffer = Math.floor(Math.min(L.maxBufferSize, L.maxStorageBufferBindingSize) / largest);
      if (byBuffer < capB) { capB = byBuffer; capWhy = `the device's buffer size limit (a frame's largest buffer ${(largest / MiB).toFixed(1)} MB)`; }
      if (L.maxTextureArrayLayers < capB) { capB = L.maxTextureArrayLayers; capWhy = "the device's texture layer limit"; }
      Object.assign(this.bt, { perFrame, shared: front + this.backShared, room: this.bt.budget - front - this.backShared, cap: capB, capWhy, slotBytes: W * H });
      const [Bmax] = this.bt.limitFor(this.bt.lanes);
      if (Bmax >= Bb) break;
      await this.buildBack(Bmax);
    }
    this.planned = { W, H };
    await this.makeLanes(this.bt.lanes);
    this.bt.start();
    this.bt.planMs = this.now() - t0;
    this.log(`batch auto: planned in ${Math.round(this.bt.planMs)} ms`);
  }

  // ai: front.cpp makeLanes: n lanes of the largest batch the budget allows, and the ring beside them; an allocation
  // ai: that fails is tried again at half.
  async makeLanes(n) {
    let [Bmax, why] = this.bt.limitFor(n);
    for (;;) {
      try {
        if (this.S.backB !== Bmax) await this.buildBack(Bmax);
        this.B = Bmax;
        await this.guarded(() => {
          for (let i = 0; i < n; i++) this.lanes.push(this.lane(Bmax));
          for (const l of this.lanes) this.ensure(l, this.planned.W, this.planned.H);
          this.makeRing(Bmax + this.C.RING_SLACK, this.planned.W, this.planned.H);
        });
        break;
      } catch (e) {
        for (const l of this.lanes) this.dropLane(l);
        this.lanes = [];
        this.destroyRing();
        if (!(e instanceof OutOfMemory)) throw e;
        if (Bmax === 1) throw new Error(`out of memory for lanes of one frame: ${e.message}`);
        why = `an allocation failed at ${Bmax} (${e.message})`;
        Bmax >>= 1;
      }
    }
    this.bt.laned(n, Bmax, why);
  }

  // ai: front.cpp makeRing: R layers at W x H. A luma frame goes into its layer by writeTexture (enqueueLuma), on the
  // ai: device's one queue, so it lands before any batch submitted after it and after every batch submitted before
  // ai: it: the native's staging buffers and its copies' tickets are not needed.
  makeRing(R, W, H) {
    this.destroyRing();
    this.ring = { R, W, H, tex: this.dev.texture(W, H, R, "ring"), slots: new Array(R).fill(null), head: 0 };
  }
  destroyRing() { if (this.ring) { this.dev.free(this.ring.tex); this.ring = null; } }
  takeSlot() {
    const r = this.ring;
    for (let k = 0; k < r.R; k++) {
      const id = (r.head + k) % r.R;
      if (r.slots[id]) continue;
      r.head = (id + 1) % r.R;
      return id;
    }
    return -1;
  }
  // ai: front.cpp enqueueLuma: a frame of luma (rows of w bytes, `stride` apart) onto a free slot; null when every
  // ai: slot is taken. tag: the caller's.
  enqueueLuma(bytes, w, h, stride = w, tag = 0) {
    if (!this.ring) throw new Error("no ring: plan() before enqueue");
    const r = this.ring;
    if (w > r.W || h > r.H) throw new Error(`${w} x ${h}: the ring holds ${r.W} x ${r.H}`);
    const id = this.takeSlot();
    if (id < 0) return null;
    this.dev.d.queue.writeTexture({ texture: r.tex, origin: [0, 0, id] }, bytes, { bytesPerRow: stride, rowsPerImage: h }, [w, h, 1]);
    const s = { id, w, h, tag, axis: 0 };
    r.slots[id] = s;
    return s;
  }
  // ai: front.cpp enqueueTexture's web twin: a frame already on the device (a camera's, sampled to luma by the
  // ai: caller's own pass into `copy(encoder, texture, layer)`), recorded into the slot's layer; null when every slot is
  // ai: taken
  enqueueEncoded(w, h, record, tag = 0) {
    if (!this.ring) throw new Error("no ring: plan() before enqueue");
    const r = this.ring;
    if (w > r.W || h > r.H) throw new Error(`${w} x ${h}: the ring holds ${r.W} x ${r.H}`);
    const id = this.takeSlot();
    if (id < 0) return null;
    const e = this.dev.d.createCommandEncoder();
    record(e, r.tex, id);
    this.dev.d.queue.submit([e.finish()]);
    const s = { id, w, h, tag, axis: 0 };
    r.slots[id] = s;
    return s;
  }
  release(s) { if (this.ring && s && s.id >= 0 && this.ring.slots[s.id] === s) this.ring.slots[s.id] = null; }
  freeLane() { return this.lanes.find((l) => !l.busy) ?? null; }
  argsOffset(s, slot) { return 4 * (this.C.ARGS_WORDS * s + this.C.ARGS_SLOT[slot]); }

  // ai: front.cpp encodeBack: the back half's dispatches (gate, pass 1 fused, reduce, pass 2, soft, LDPC from the gate)
  encodeBack(pass, ln, begin) {
    const S = this.S, bk = S.back(), T = bk.transform, b = ln.back;
    let e = begin("gate");
    e.setPipeline(S.pipe(T.gate)); e.setBindGroup(0, b.gate); e.dispatchWorkgroups(1);
    const sized = (step, pipes, groups) => { for (const s of T.built) { e.setPipeline(S.pipe(pipes[s])); e.setBindGroup(0, groups[s]); e.dispatchWorkgroupsIndirect(b.args, this.argsOffset(s, step)); } };
    e = begin("pass1"); sized("pass1", T.pass1f, b.p1f);
    e = begin("reduce"); sized("reduce", T.reduce, b.red);
    e = begin("pass2"); sized("pass2", T.pass2, b.p2);
    const st = bk.rule;
    const stepOf = (step, g) => { e.setPipeline(S.pipe(step.pipeline)); e.setBindGroup(0, g); e.dispatchWorkgroupsIndirect(b.args, step.argsAt); };
    e = begin("soft");
    b.rsoft.forEach((g, k) => stepOf(st.soft[k], g));
    e = begin("ldpc");
    const lsteps = this.ldpcForm ? st.ldpcAlts[this.ldpcForm - 1] : st.ldpc;
    b.rldpc.forEach((g, k) => stepOf(lsteps[k], g));
  }

  // ai: front.cpp run: a batch of staged slots on a free lane, recorded and submitted. held: the held version (0
  // ai: none). profile: a pass and a timestamp pair a stage group.
  run(slots, held, profile = false) {
    const C = this.C, d = this.dev, nb = slots.length;
    if (!nb) throw new Error("a batch of no slots");
    if (nb > this.B) throw new Error(`${nb} frames in a batch of ${this.B}`);
    if (held < 0 || (held > 0 && held < C.VERSION_MIN) || held > C.VERSION_MAX) throw new Error(`held ${held}: 0, or a version ${C.VERSION_MIN} to ${C.VERSION_MAX}`);
    const ln = this.freeLane();
    if (!ln) throw new Error(`${this.lanes.length} batches in flight already`);
    const f = { t0: this.now(), lane: ln, nb, stages: [] };
    let W = 1, H = 1;
    for (const s of slots) if (s) { W = Math.max(W, s.w); H = Math.max(H, s.h); }
    this.ensure(ln, W, H);
    const Bl = ln.B;
    d.write(ln.heldUni, 0, new Uint32Array([held, 0, 0, 0]));
    {
      const dims = new Uint32Array(4 * ln.slotsG);
      slots.forEach((s, i) => { if (s) dims.set([s.w, s.h, s.axis, 1], 4 * i); });
      d.write(ln.framesBuf, 0, dims);
    }
    const enc = d.d.createCommandEncoder();
    // ai: each slot's layer into the lane's texture (the native's copies' own submit: here the one queue orders them)
    slots.forEach((s, i) => { if (s) enc.copyTextureToTexture({ texture: this.ring.tex, origin: [0, 0, s.id] }, { texture: ln.tex, origin: [0, 0, i] }, [s.w, s.h, 1]); });
    const clear = (b) => enc.clearBuffer(b, 0, (b.size / Bl) * nb);
    for (const b of [ln.countsBuf, ln.peaksBuf, ln.rawBuf, ln.resultBuf, ln.residBuf, ln.readingsBuf, ln.accBuf, ln.candBuf, ln.quadsBuf, ln.scoreBuf, ln.mapsBuf, ln.measBuf, ln.vcountBuf]) clear(b);
    enc.clearBuffer(ln.selBuf);
    enc.clearBuffer(ln.args2Buf);
    if (profile && d.timestamps && (!ln.qsProfile || ln.qsProfile.count < 2 * STAGES.length)) {
      ln.qsProfile = d.d.createQuerySet({ type: "timestamp", count: 2 * STAGES.length });
      ln.qsProfileBuf = this.mk(ln, 16 * STAGES.length, U.QUERY_RESOLVE | U.COPY_SRC);
    }
    const prof = profile && !!ln.qsProfile;
    let pass = null;
    // ai: unprofiled one pass (its pair at 0, 1); profiled a pass a stage group
    const begin = (name) => {
      if (pass && !prof) return pass;
      if (pass) pass.end();
      const ts = (qs, a, b) => (qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: a, endOfPassWriteIndex: b } } : {});
      if (prof) { const k = STAGES.indexOf(name); pass = enc.beginComputePass({ label: name, ...ts(ln.qsProfile, 2 * k, 2 * k + 1) }); f.stages.push(name); }
      else pass = enc.beginComputePass({ label: "batch", ...ts(ln.qs, 0, 1) });
      return pass;
    };
    const n = nb;
    let e = begin("pyramid");
    const step = (pl, g, x, y = 1, z = 1) => { e.setPipeline(pl.p); e.setBindGroup(0, g); e.dispatchWorkgroups(x, y, z); };
    step(this.P("pyramid"), ln.pyrGroup, ceilDiv(W, C.PYRAMID_BLOCK), ceilDiv(H, C.PYRAMID_BLOCK), n);
    e = begin("bank");
    const form = this.S.tree.front.fcn2Form;
    const TX = form.TX, TY = form.TY, walk = form.walk;
    const fcn2X = this.P("fcn2X"), fcn2 = this.P("fcn2");
    const XB0 = fcn2X ? form.XB[0] : 0, XB1 = fcn2X ? form.XB[1] : 0;
    {
      // ai: front.cpp run's grids2: the second round's grids for one frame (gpu/glsl/finder.mjs R2), in its order
      const g = [];
      if (fcn2X) for (const lv of ln.fcn2Levels) if (lv.l < this.coarseFrom) g.push([ceilDiv(lv.w, XB0), ceilDiv(lv.h, XB1), 1, 2]);
      for (const lv of ln.fcn2Levels) if (lv.l < this.coarseFrom) g.push([ceilDiv(ceilDiv(lv.w, C.TILE * TX), walk), ceilDiv(lv.h, C.TILE * TY), 1, 2]);
      const cp = this.cap;
      g.push([1, 1, 1, 0], [ceilDiv(cp, this.classify.P), 1, 1, 1], [ceilDiv(cp, C.VOTE_READS), 1, 1, 1], [ceilDiv(ln.gx * ln.gy, 256), 1, 1, 1], [1, 1, 1, 0], [C.K, 1, 1, 0], [C.K, 1, 1, 0], [C.HYP, 1, 1, 0], [1, 1, 1, 0]);
      if (g.length > C.R2_N) throw new Error(`the second round has ${g.length} grids, R2 holds ${C.R2_N}`);
      const u = new Uint32Array(4 * C.R2_N);
      for (let i = 0; i < C.R2_N; i++) { if (i < g.length) u.set(g[i], 4 * i); else u[4 * i + 3] = 3; }
      d.write(ln.r2Uni, 0, u);
    }
    const bank = (round) => {
      const inRound = (lv) => (lv.l < this.coarseFrom) === (round === 2);
      let r = 0;
      const go = (x, y) => { if (round === 2) { e.dispatchWorkgroupsIndirect(ln.args2Buf, 12 * r); r++; } else e.dispatchWorkgroups(x, y, n); };
      if (fcn2X) {
        e.setPipeline(fcn2X.p);
        for (const lv of ln.fcn2Levels) if (inRound(lv)) { e.setBindGroup(0, round === 2 ? lv.xgroup2 : lv.xgroup); go(ceilDiv(lv.w, XB0), ceilDiv(lv.h, XB1)); }
      }
      e.setPipeline(fcn2.p);
      for (const lv of ln.fcn2Levels) if (inRound(lv)) { e.setBindGroup(0, round === 2 ? lv.group2 : lv.group); go(ceilDiv(ceilDiv(lv.w, C.TILE * TX), walk), ceilDiv(lv.h, C.TILE * TY)); }
      return r;
    };
    bank(1);
    e = begin("select");
    step(this.P("select"), ln.selGroup, n);
    e = begin("describe");
    step(this.classify, ln.classifyGroup, ceilDiv(this.cap, this.classify.P), n);
    e = begin("vote");
    step(this.P("vote"), ln.voteGroup, ceilDiv(this.cap, C.VOTE_READS), n);
    step(this.P("peaks"), ln.peaksGroup, ceilDiv(ln.gx * ln.gy, 256), n);
    step(this.P("gather"), ln.gatherGroup, n);
    step(this.P("trace"), ln.traceGroup, C.K * n);
    e = begin("score");
    step(this.P("score"), ln.scoreGroup, C.HYP * n);
    step(this.P("pick"), ln.pickGroup, n);
    {
      e = begin("find2");
      step(this.P("redo"), ln.redoGroup, n);
      let r = bank(2);
      const again = (pl, g) => { e.setPipeline(pl.p); e.setBindGroup(0, g); e.dispatchWorkgroupsIndirect(ln.args2Buf, 12 * r++); };
      again(this.P("select"), ln.selGroup2);
      again(this.classify, ln.classifyGroup2);
      again(this.P("vote"), ln.voteGroup2);
      again(this.P("peaks"), ln.peaksGroup2);
      again(this.P("gather"), ln.gatherGroup2);
      again(this.P("complete"), ln.completeGroup2);
      again(this.P("trace"), ln.traceGroup2);
      again(this.P("score"), ln.scoreGroup2);
      again(this.P("pick"), ln.pickGroup2);
    }
    e = begin("nodes");
    step(this.P("nodesP"), ln.nodesGroup, this.nodesMax, n);
    e = begin("refit");
    step(this.P("refit"), ln.refitGroup, n);
    step(this.P("refitPost"), ln.refitPostGroup, n);
    step(this.anchFind, ln.anchFindGroup, C.ANCH_K, n);
    step(this.anchPredict, ln.anchRefitGroup, n);
    step(this.anchNodes, ln.anchNodesGroup, this.nodesMax, n);
    step(this.anchRefit, ln.anchRefitGroup, n);
    step(this.anchFit, ln.anchFitGroup, n);
    e = begin("word");
    step(this.P("wordP"), ln.wordGroup, n);
    this.encodeBack(e, ln, (name) => (e = begin(name)));
    pass.end();
    // ai: the readback's layout: counters, finder results, the back half's (REC, V, ITS, BCOUNTS, PILOT), timestamps
    const cntBytes = 4 * C.COUNTERS * nb, resBytes = 4 * C.RESULT * nb;
    f.cntOff = 0; f.resOff = cntBytes; f.backOff = f.resOff + resBytes;
    f.tsOff = f.backOff + this.S.back().readBytes;
    enc.copyBufferToBuffer(ln.countsBuf, 0, ln.readBuf, f.cntOff, cntBytes);
    enc.copyBufferToBuffer(ln.resultBuf, 0, ln.readBuf, f.resOff, resBytes);
    {
      const by = this.S.back().bytes, b = ln.back;
      let o = f.backOff;
      for (const [buf, k] of [[b.REC, "REC"], [b.V, "V"], [b.ITS, "ITS"], [b.counts, "BCOUNTS"], [b.PILOT, "PILOT"]]) { enc.copyBufferToBuffer(buf, 0, ln.readBuf, o, by[k]); o += by[k]; }
    }
    if (prof) {
      f.nq = ln.qsProfile.count;
      enc.resolveQuerySet(ln.qsProfile, 0, f.nq, ln.qsProfileBuf, 0);
      enc.copyBufferToBuffer(ln.qsProfileBuf, 0, ln.readBuf, f.tsOff, 8 * f.nq);
    } else {
      f.nq = ln.qs ? ln.qs.count : 0;
      if (ln.qs) { enc.resolveQuerySet(ln.qs, 0, f.nq, ln.qsBuf, 0); enc.copyBufferToBuffer(ln.qsBuf, 0, ln.readBuf, f.tsOff, 8 * f.nq); }
    }
    f.readBytes = Math.ceil((f.tsOff + 8 * f.nq) / 4) * 4;
    d.d.queue.submit([enc.finish()]);
    f.mapped = ln.readBuf.mapAsync(GPUMapMode.READ, 0, f.readBytes);
    f.tags = slots.map((s) => (s ? s.tag : 0));
    f.axes = slots.map((s) => (s ? s.axis : 0));
    f.present = slots.map((s) => !!s);
    for (const s of slots) if (s) this.release(s);
    f.submitted = this.now();
    f.W = W; f.H = H;
    ln.busy = true;
    return f;
  }

  // ai: front.cpp finish: the readback mapped once the batch is done, each frame's finder result, word, counters and
  // ai: verified records; the batcher told; a throw gives the lane back.
  async finish(f) {
    try { return await this.finishBatch(f); }
    catch (e) { f.lane.busy = false; throw e; }
  }

  async finishBatch(f) {
    const ln = f.lane, C = this.C, d = this.dev;
    await f.mapped;
    const out = { frames: [], gpuMs: null, deviceMs: 0, done: this.now(), carried: 0, stageMs: [] };
    const all = new Uint8Array(ln.readBuf.getMappedRange(0, f.readBytes).slice(0));
    ln.readBuf.unmap();
    const nb = f.nb, dv = new DataView(all.buffer);
    const u32 = (o) => dv.getUint32(o, true), f32 = (o) => dv.getFloat32(o, true);
    const ts = (k) => dv.getBigUint64(f.tsOff + 8 * k, true);
    const period = 1;   // ai: WebGPU timestamps are in nanoseconds
    if (f.stages.length) {
      let sum = 0;
      for (const name of f.stages) { const k = STAGES.indexOf(name); const ms = Number(ts(2 * k + 1) - ts(2 * k)) * period / 1e6; out.stageMs.push([name, ms]); sum += ms; }
      out.gpuMs = sum;
    } else if (f.nq >= 2) {
      const t0 = ts(0), t1 = ts(1);
      out.gpuMs = t1 >= t0 ? Number(t1 - t0) * period / 1e6 : null;
    }
    const bk = this.S.back();
    const Bb = bk.B, blocksMax = bk.dims.blocksMax, recCap = bk.dims.recCap;
    const rec = f.backOff, by = bk.bytes;
    const bcounts = rec + by.REC + by.V + by.ITS, pilots = bcounts + by.BCOUNTS;
    const count = Math.min(u32(rec), recCap);
    for (let i = 0; i < nb; i++) out.frames.push({ records: [] });
    for (let r = 0; r < count; r++) {
      const o = rec + 4 + C.REC_BYTES * r;
      const tag = u32(o), its = dv.getInt32(o + 4, true);
      const fr = tag & 0xffff, b = tag >>> 16;
      if (fr >= Bb || b >= blocksMax || fr >= nb) continue;
      out.frames[fr].records.push({ frame: fr, block: b, its, id: u32(o + 8), payload: all.slice(o + 8, o + 8 + C.PAYLOAD) });
    }
    const pics = this.S.tree.pictures;
    const slotOf = (v) => pics.findIndex((p) => v >= p.lo && v <= p.hi);
    const rb = bk.rule.blocks;
    for (let i = 0; i < nb; i++) {
      const fo = out.frames[i];
      fo.counts = Array.from({ length: C.COUNTERS }, (_, k) => u32(f.cntOff + 4 * (C.COUNTERS * i + k)));
      fo.backCounts = Array.from({ length: C.COUNTS }, (_, k) => u32(bcounts + 4 * (C.COUNTS * i + k)));
      fo.tag = f.tags[i]; fo.axis = f.axes[i];
      if (!f.present[i]) { fo.empty = true; continue; }
      const r = (k) => f32(f.resOff + 4 * (C.RESULT * i + k));
      fo.ring = r(0) === 1 ? r(2) : -1;
      fo.version = Math.trunc(r(29));
      fo.size = fo.version > 0 ? slotOf(fo.version) : -1;
      fo.n = fo.size >= 0 ? pics[fo.size].n : 0;
      fo.held = r(30) === 1;
      fo.finder = { found: r(0), ring: r(2), corners: Array.from({ length: 8 }, (_, k) => r(17 + k)) };
      fo.hasWord = r(5) > 0;
      if (fo.hasWord) fo.word = { version: Math.trunc(r(5)), fps: Math.trunc(r(6)) };
      fo.anchors = Math.trunc(r(31));
      fo.pilotBlocks = 0;
      const nblk = fo.version > 0 && fo.version < rb.length ? rb[fo.version] : fo.version;
      if (fo.version > 0 && fo.backCounts[6] > 0) {
        fo.pilotR = fo.pilotSd = fo.pilotR2 = fo.pilotSd2 = NaN;
        for (let g = 0; g < 2; g++) {
          let s = 0, s2 = 0, k = 0;
          for (let b = g; b < Math.min(nblk, blocksMax); b += 2) { const v = f32(pilots + 4 * (i * blocksMax + b)); if (Number.isFinite(v)) { s += v; s2 += v * v; k++; } }
          if (!k) continue;
          const m = s / k, v = k > 1 ? (s2 - s * m) / (k - 1) : 0;
          fo.pilotBlocks += k;
          if (g) { fo.pilotR2 = m; fo.pilotSd2 = Math.sqrt(Math.max(v, 0) / k); } else { fo.pilotR = m; fo.pilotSd = Math.sqrt(Math.max(v, 0) / k); }
        }
      }
    }
    out.carried = nb;
    out.deviceMs = out.gpuMs ?? out.done - Math.max(f.submitted, this.lastDone);
    this.lastDone = Math.max(this.lastDone, out.done);
    this.bt.observe(nb, out.deviceMs, !ln.watched);
    ln.watched = true;
    ln.busy = false;
    let codewords = 0;
    for (const fo of out.frames) codewords += fo.backCounts[3] ?? 0;
    if (this.ldpcPending && nb && codewords >= 64) {
      try { await this.measureLdpc(ln, nb); } catch (x) { this.ldpcPending = false; this.log(`LDPC forms not timed (${x.message}): the first`); }
    }
    return out;
  }

  // ai: front.cpp measureLdpc: the LDPC kernel's forms timed on this batch's frames, still on its lane, three rounds
  // ai: in alternating order, the first untimed; the fastest kept (the earlier on a tie)
  async measureLdpc(ln, nb) {
    this.ldpcPending = false;
    const d = this.dev;
    if (!d.timestamps) { this.log("LDPC forms not timed (no timestamps): the first"); return; }
    const st = this.S.back().rule, b = ln.back;
    const R = this.C.TWIN_ROUNDS, A = 1 + st.ldpcAlts.length;
    const qs = d.d.createQuerySet({ type: "timestamp", count: 2 * A * R });
    const res = d.d.createBuffer({ size: 16 * A * R, usage: U.QUERY_RESOLVE | U.COPY_SRC });
    const rd = d.d.createBuffer({ size: 16 * A * R, usage: U.MAP_READ | U.COPY_DST });
    const e = d.d.createCommandEncoder();
    for (let r = 0; r < R; r++) for (let j = 0; j < A; j++) {
      const i = r % 2 ? A - 1 - j : j, k = r * A + i;
      const steps = i ? st.ldpcAlts[i - 1] : st.ldpc;
      const p = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
      if (this.ldpcBuilt[i]) b.rldpc.forEach((g, q) => { p.setPipeline(this.S.pipe(steps[q].pipeline)); p.setBindGroup(0, g); p.dispatchWorkgroupsIndirect(b.args, steps[q].argsAt); });
      p.end();
    }
    e.resolveQuerySet(qs, 0, 2 * A * R, res, 0);
    e.copyBufferToBuffer(res, 0, rd, 0, res.size);
    d.d.queue.submit([e.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rd.getMappedRange().slice(0));
    rd.unmap(); rd.destroy(); res.destroy(); qs.destroy();
    const best = new Array(A).fill(Infinity);
    for (let r = 1; r < R; r++) for (let i = 0; i < A; i++) { const k = r * A + i; if (this.ldpcBuilt[i]) best[i] = Math.min(best[i], Number(t[2 * k + 1] - t[2 * k]) / 1e6 / nb); }
    this.ldpcForm = 0;
    for (let i = 1; i < A; i++) if (best[i] < best[this.ldpcForm]) this.ldpcForm = i;
    const times = { frames: nb, by: "timestamps" };
    st.ldpcForms.forEach((n, i) => { if (this.ldpcBuilt[i]) times[n] = best[i]; });
    times.kept = st.ldpcForms[this.ldpcForm];
    this.log(`LDPC forms timed on ${nb} frames: ${JSON.stringify(times)}`);
  }

  // ai: every lane, the ring and the setup's objects let go
  destroy() {
    for (const l of this.lanes) this.dropLane(l);
    this.lanes = [];
    this.destroyRing();
    for (const b of this.S.buffers.values()) this.dev.free(b);
    this.S.buffers.clear();
  }
}

// ai: The files of a packed tree (gen.mjs packWebgpu: "LZWT", u32 1, u32 count, then a u16 name length, the name, a
// ai: u32 size and the bytes a file), its zstd frame already undone: a FrontHalf.create `read` over them.
export function treeReader(raw) {
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength), files = new Map(), td = new TextDecoder();
  if (td.decode(raw.subarray(0, 4)) !== "LZWT" || dv.getUint32(4, true) !== 1) throw new Error("not a WebGPU tree (gen.mjs packWebgpu)");
  let o = 12;
  for (let i = 0, n = dv.getUint32(8, true); i < n; i++) {
    const nl = dv.getUint16(o, true), name = td.decode(raw.subarray(o + 2, o + 2 + nl)), size = dv.getUint32(o + 2 + nl, true);
    o += 6 + nl;
    files.set(name, raw.subarray(o, o + size));
    o += size;
  }
  return (path) => { const f = files.get(path); if (!f) throw new Error(`the WebGPU tree holds no ${path}`); return f; };
}

// ai: The limits and features a decoder's device asks of its adapter: the adapter's own limits (the tree needs 10
// ai: storage buffers a stage and 32 KB of workgroup memory, past WebGPU's defaults) and timestamps where offered.
// ai: Null with why where the adapter misses the floors (setup.cpp variantsFor's: 32 KB, 10 storage buffers).
export function deviceDescriptor(adapter) {
  const L = adapter.limits;
  if (L.maxComputeWorkgroupStorageSize < 32768) return { why: `${L.maxComputeWorkgroupStorageSize} B of workgroup memory, the decoder needs 32768` };
  if (L.maxStorageBuffersPerShaderStage < 10) return { why: `${L.maxStorageBuffersPerShaderStage} storage buffers a stage, the decoder needs 10` };
  const requiredLimits = {};
  for (const k of ["maxComputeWorkgroupStorageSize", "maxStorageBuffersPerShaderStage", "maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ", "maxBufferSize", "maxStorageBufferBindingSize", "maxUniformBuffersPerShaderStage", "maxSampledTexturesPerShaderStage", "maxTextureArrayLayers", "maxComputeWorkgroupsPerDimension"]) requiredLimits[k] = L[k];
  const requiredFeatures = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  return { requiredLimits, requiredFeatures };
}
