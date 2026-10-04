// ai: The paint stage's host (DESIGN.md 13.1): the cancel stage's paint, irows and ipic pipelines over a BackHalf
// ai: built with cancel, PERMW (slot to codeword bit a bit map mode, bitmap.mjs permTable), the bind groups a lane,
// ai: and the three dispatches plugged into Cancel.stages. It reads the transform's tables (twiddles, disc rows,
// ai: yStride), the soft stage's UV buffer and entry offsets and the cancel stage's buffers, so Cancel.wire builds it
// ai: once the back half's other stages exist and Cancel.bind gives it every lane (cancel.mjs); scripts/gpu/back/test_cancel_paint.mjs
// ai: runs it through those stages.
// ai:
// ai:   const paint = await Paint.build(device, bh, { log });   // ai: sets bh.cancel.stages paint, irows and ipic
// ai:   paint.lane(ln);                                          // ai: after bh.lane(...)
import { paintSource, irowsSource, ipicSource } from "../wgsl/cancel_paint.mjs";
import { permTable } from "./bitmap.mjs";
import { layWords, packMap } from "./ldpc.mjs";
import { PARAMS_AT } from "../wgsl/back_ldpc.mjs";
import { crcPowers, PAYLOAD } from "./ref_ldpc.mjs";

const U = globalThis.GPUBufferUsage;

// ai: Every sender's clip ratio and tilt today (SPEC.md 13 item 8): the clip level the paint reproduces, the C's
// ai: f32 arithmetic (focus.c 646): lim = 2 clip sqrt(0.5 subch 320) at tilt 0.
export const CLIP = 2;
export const clipLimit = (subch, clip = CLIP) => Math.fround(Math.fround(2 * clip) * Math.fround(Math.sqrt(Math.fround(0.5 * subch * 320))));
// ai: ipic's LIM uniform: version v's (lim, 0.5 / lim, 0, 0) at v, versions 1 to 128 (8 v sub-channels); 0 unused.
export function limTable() {
  const f = new Float32Array(4 * 129);
  for (let v = 1; v <= 128; v++) { const lim = clipLimit(8 * v); f[4 * v] = lim; f[4 * v + 1] = Math.fround(0.5 / lim); }
  return f;
}

export class Paint {
  static async build(device, bh, { log = () => {} } = {}) {
    const c = bh.cancel;
    if (!c) throw new Error("the back half was built without the cancel stage");
    const t = bh.transform, { dims, tables, code, precision, B } = bh, { served, blocksMax, refSlots, sStride, picStride, bitmaps } = dims;
    const p = new Paint();
    Object.assign(p, { device, bh, built: t.built, refSlots, tables });
    // ai: PERMW: a row of blocks x 5120 u16 a bit map mode in play, the most blocks any size of that mode holds (block
    // ai: b's row is the same in every format, as the LDPC's MAP), at its entry offset permOff.
    const modes = new Map(), permOff = new Map();
    for (const s of served) { const m = bitmaps[s]; modes.set(m, Math.max(modes.get(m) ?? 0, tables[s].blocks)); }
    let entries = 0;
    for (const [m, blocks] of modes) { permOff.set(m, entries); entries += blocks * 5120; }
    p.PERMW = device.createBuffer({ size: Math.max(16, 2 * entries), usage: U.STORAGE | U.COPY_DST });
    for (const [m, blocks] of modes) device.queue.writeBuffer(p.PERMW, 2 * permOff.get(m), packMap(permTable({ blocks, mode: m })));
    const { uvOff } = bh.soft;
    const up = (bytes) => { const b = device.createBuffer({ size: bytes.byteLength, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, bytes); return b; };
    // ai: The paint's uniform in the LDPC's shape: sizes, dims, lay, pw (wgsl/cancel_paint.mjs paintSource).
    const uni = new Uint32Array(PARAMS_AT.words);
    for (const s of served) uni.set([tables[s].blocks, uvOff[s], permOff.get(bitmaps[s]), 0], PARAMS_AT.sizes + 4 * s);
    uni.set([blocksMax, sStride, refSlots, 0], PARAMS_AT.dims);
    uni.set(layWords(code), PARAMS_AT.lay);
    uni.set(crcPowers(PAYLOAD), PARAMS_AT.pw);
    p.paintUni = up(uni);
    p.limUni = up(limTable());
    p.sizeUni = [];
    for (const s of t.built) {
      const tb = tables[s], ab = new ArrayBuffer(48), u32 = new Uint32Array(ab), f32 = new Float32Array(ab), lim = clipLimit(tb.subch);
      u32.set([tb.n, tb.V, t.yStride, sStride, picStride / 8, s, 0, 0]);
      f32[8] = lim; f32[9] = Math.fround(0.5 / lim);
      p.sizeUni[s] = up(new Uint8Array(ab));
    }
    const layout = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
    const RO = "read-only-storage", RW = "storage", UN = "uniform";
    p.paintBgl = layout([RO, RO, RO, RO, RW, UN]);
    p.irowsBgl = layout([RO, RO, RO, RW, UN, UN]);
    p.ipicBgl = layout([RO, RO, RO, RW, UN, UN]);
    const pipe = async (bgl, src, label) => {
      const module = device.createShaderModule({ label, code: src });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === "error");
      if (errs.length) throw new Error(errs.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + src.split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
      return device.createComputePipelineAsync({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: "main" } });
    };
    const common = { prec: precision, B, blocksMax, refSlots };
    const jobs = [pipe(p.paintBgl, paintSource(common), "cancel paint")];
    for (const s of t.built) {
      jobs.push(pipe(p.irowsBgl, irowsSource({ n: tables[s].n, ...common }), `cancel irows ${tables[s].n}`));
      jobs.push(pipe(p.ipicBgl, ipicSource({ n: tables[s].n, ...common }), `cancel ipic ${tables[s].n}`));
    }
    const done = await Promise.all(jobs);
    p.paint = done[0]; p.irows = []; p.ipic = [];
    t.built.forEach((s, i) => { p.irows[s] = done[1 + 2 * i]; p.ipic[s] = done[2 + 2 * i]; });
    c.stages.paint = (pass, ln) => p.dispatch(pass, ln, "paint");
    c.stages.irows = (pass, ln) => p.dispatch(pass, ln, "irows");
    c.stages.ipic = (pass, ln) => p.dispatch(pass, ln, "ipic");
    log(`paint: ${precision}, sizes ${t.built.map((s) => `${tables[s].n}/${tables[s].subch} (bit map ${bitmaps[s]}, lim ${clipLimit(tables[s].subch).toFixed(1)})`).join(", ")}; ${refSlots} slots; PERMW ${[...modes].map(([m, b]) => `mode ${m} x ${b} blocks`).join(", ")}`);
    return p;
  }

  // ai: The bind groups over a lane the back half made (REC, PLAN, S, Y, PIC on it).
  lane(ln) {
    const d = this.device, bh = this.bh, t = bh.transform, c = bh.cancel;
    const group = (bgl, bufs) => d.createBindGroup({ layout: bgl, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.paintGroup = group(this.paintBgl, [ln.REC, ln.PLAN, this.PERMW, bh.soft.uvBuf, ln.S, this.paintUni]);
    ln.irowsGroups = []; ln.ipicGroups = [];
    for (const s of this.built) {
      ln.irowsGroups[s] = group(this.irowsBgl, [ln.PLAN, ln.S, t.twBufs[s], ln.yBuf, t.rowsBufs[s], this.sizeUni[s]]);
      ln.ipicGroups[s] = group(this.ipicBgl, [ln.PLAN, ln.yBuf, t.twBufs[s], ln.PIC, this.sizeUni[s], this.limUni]);
    }
    return ln;
  }

  // ai: One stage on an open pass: paint (blocksMax, 1, R) once, irows (V, 1, R) and ipic (n / 4, 1, R) a built size.
  dispatch(p, ln, name) {
    const R = this.refSlots, tb = this.tables;
    if (name === "paint") { p.setPipeline(this.paint); p.setBindGroup(0, ln.paintGroup); p.dispatchWorkgroups(this.bh.dims.blocksMax, 1, R); return; }
    if (name === "irows") { for (const s of this.built) { p.setPipeline(this.irows[s]); p.setBindGroup(0, ln.irowsGroups[s]); p.dispatchWorkgroups(tb[s].V, 1, R); } return; }
    if (name === "ipic") { for (const s of this.built) { p.setPipeline(this.ipic[s]); p.setBindGroup(0, ln.ipicGroups[s]); p.dispatchWorkgroups(tb[s].n / 4, 1, R); } return; }
    throw new Error(`paint stage ${name}`);
  }
}
