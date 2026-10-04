// ai: The cancel stage's host (DESIGN.md section 13): straddle cancellation as pass two over a lane whose pass one
// ai: has run, behind it in the batch's own command buffer. From pass one's verified blocks (REC) the gate plans which
// ai: known pictures (references, keyed by id - block) each short frame is cancelled against; the paint encodes those
// ai: blocks into the disc and inverts them to pictures (PIC); the blur makes three bases and their row means a
// ai: reference (REF); the fit and solve regress each short frame on its references' bases (FITP, CANCEL); then pass
// ai: one's chain runs again over the short frames (LISTS2, ARGS2, BCOUNTS2) with the subtraction at the sample, and
// ai: the LDPC appends the blocks pass one missed to REC. The carry (CARRY, shared by the lanes) keeps the last
// ai: reference for the next batch.
// ai:
// ai: This host holds the buffers of DESIGN 13's table, their strides from dims.mjs, and the gate, zero and carry
// ai: shaders (wgsl/cancel_gate.mjs). The paint (paint.mjs) and the bases and fit (fit.mjs) plug in through
// ai: `stages`, filled by wire(bh) once the back half's other stages exist; a stage left unfilled fails the build,
// ai: so a built stage always runs whole.
// ai:
// ai:   const c = await Cancel.build(device, { B, tables, precision, code, dims, log });
// ai:   await c.wire(bh);   // the paint, bases and fit over the back half's transform and soft stage
// ai:   c.lane(ln);   // the buffers, before the soft and LDPC stages make their second bind groups
// ai:   c.bind(ln);   // the bind groups, once ln.REC exists
// ai:   c.prepare(enc, ln);   // the carry in, on the encoder before the lane's pass two
// ai:   c.dispatch(p, ln, name, picture);   // one of PASSES2 less SECOND, on an open pass; picture "grid" | "fused"
import { gate2Source, zeroSource, carrySource, planLayout, NONE, KEYTAB_WORDS, FR_WORDS, CANCEL_WORDS, CARRY_HEADER } from "../wgsl/cancel_gate.mjs";
import { CPW, ARGS_WORDS, SLOTS, argsWords, listsHead, listsWords, listsBlocks, listsRows, listsRing } from "../wgsl/back_transform.mjs";
import { Paint } from "./paint.mjs";
import { build as buildFit, STAGES as FIT_STAGES } from "./fit.mjs";

const U = globalThis.GPUBufferUsage;

// ai: Pass two's dispatches in order, appended to pass one's six (back.mjs PASSES). SECOND are pass one's own
// ai: pipelines over the second lists, dispatched by back.mjs; the rest are this stage's.
export const PASSES2 = ["gate2", "paint", "irows", "ipic", "blur", "bmeans", "fit", "solve", "pass1c", "reduce2", "pass2b", "soft2", "ldpc2", "carry"];
export const SECOND = ["pass1c", "reduce2", "pass2b", "soft2", "ldpc2"];
// ai: The stages other files supply, by pass name: a pipeline each (null until wired).
export const STAGES = ["paint", "irows", "ipic", "blur", "bmeans", "fit", "solve"];
export { NONE, CANCEL_WORDS, FR_WORDS, KEYTAB_WORDS, CARRY_HEADER };

export class Cancel {
  // ai: tables: transformTables' (n, V, blocks a built size); code: the LDPC code; dims: derived() with B and cap
  // ai: (refSlots, refStride, picStride, fitpStride, sStride, blocksMax, recCap, nmax, served).
  static async build(device, { B, tables, precision = "f32", code, dims, log = () => {} }) {
    if (precision !== "f32" && precision !== "f16") throw new Error(`precision ${precision}`);
    const c = new Cancel();
    const { served, blocksMax, recCap, refSlots, refStride, refMeans, picStride, fitpStride, sStride, nmax } = dims;
    const cbytes = precision === "f16" ? 4 : 8;
    Object.assign(c, { device, B, precision, cbytes, code, dims, log, stages: Object.fromEntries(STAGES.map((n) => [n, null])) });
    c.plan = planLayout({ blocksMax, B, refSlots });
    c.sVec4 = (sStride * cbytes) / 16;
    c.refVec4 = refStride / 16;
    c.bytes = {
      PLAN: 4 * c.plan.words, LISTS2: 4 * listsWords(B), ARGS2: 4 * argsWords, CANCEL: 4 * CANCEL_WORDS * B, BCOUNTS2: 4 * 8 * B,
      PIC: refSlots * picStride, REF: (refSlots + 1) * refStride, BPART: refSlots * 3 * nmax * (nmax / 16) * 4, FITP: B * fitpStride,
      CARRY: CARRY_HEADER + refStride,
    };
    // ai: What a size's dispatches need: (n / CPW, V, blocks, 0), as the first gate's uniform (its rows table apart:
    // ai: gate2 copies each frame's rows from LISTS).
    c.at = [];
    const gate = new Uint32Array(4 * SLOTS);
    for (const s of served) { c.at[s] = [tables[s].n / CPW, tables[s].V, tables[s].blocks, 0]; gate.set(c.at[s], 4 * s); }
    // ai: The rows a block count takes a size (transform.mjs rowsByCount), for hostLists.
    c.rowsBy = tables.map((tb) => tb?.rowsBy ?? null);
    c.gateUni = device.createBuffer({ size: gate.byteLength, usage: U.UNIFORM | U.COPY_DST });
    device.queue.writeBuffer(c.gateUni, 0, gate);
    // ai: The carry: one reference slot behind its header, shared by every lane (a batch's carry in is a copy from
    // ai: it, its carry out a write into it; the queue orders them). Zero at creation, so valid is 0.
    c.CARRY = device.createBuffer({ size: c.bytes.CARRY, usage: U.STORAGE | U.UNIFORM | U.COPY_SRC | U.COPY_DST });
    const layout = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
    const RO = "read-only-storage", RW = "storage", UN = "uniform";
    c.gate2Bgl = layout([RO, RO, RO, RW, RW, RW, RW, RW, UN, UN]);
    c.zeroBgl = layout([RO, RW]);
    c.carryBgl = layout([RO, RO, RW]);
    const pipe = async (bgl, code, label) => {
      const module = device.createShaderModule({ label, code });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === "error");
      if (errs.length) throw new Error(errs.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + code.split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
      return device.createComputePipelineAsync({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: "main" } });
    };
    [c.gate2, c.zero, c.carry] = await Promise.all([
      pipe(c.gate2Bgl, gate2Source({ B, blocksMax, refSlots, recCap }), "cancel gate2"),
      pipe(c.zeroBgl, zeroSource({ B, blocksMax, refSlots, sVec4: c.sVec4 }), "cancel zero"),
      pipe(c.carryBgl, carrySource({ B, blocksMax, refSlots, refVec4: c.refVec4 }), "cancel carry"),
    ]);
    c.sharedBytes = gate.byteLength + c.bytes.CARRY;
    const MB = (v) => (v / 1048576).toFixed(1);
    log(`cancel: ${refSlots} reference slots and a carry, REF ${MB(refStride)} MB a slot, PIC ${MB(picStride)} MB, a lane ${MB(c.laneBytes())} MB more, the carry ${MB(c.bytes.CARRY)} MB shared`);
    return c;
  }

  // ai: The paint (paint.mjs: paint, irows, ipic) and the bases and fit (fit.mjs: blur, bmeans, fit,
  // ai: solve; the fused fit when the transform samples the frame itself) over the back half bh, whose transform,
  // ai: soft stage and this object already exist. A stage still unfilled after both is a wiring bug, not a mode.
  async wire(bh) {
    if (bh.cancel !== this) throw new Error("wire: not this back half's cancel stage");
    this.paint = await Paint.build(this.device, bh, { log: this.log });
    this.fit = await buildFit(this.device, { B: this.B, tables: bh.tables, dims: this.dims, grid: bh.grid, picUnis: bh.fusable ? bh.transform.picUnis : null, log: this.log });
    for (const name of FIT_STAGES) this.stages[name] = (p, ln, picture) => this.fit.dispatch(p, ln, name, picture);
    const missing = STAGES.filter((name) => typeof this.stages[name] !== "function");
    if (missing.length) throw new Error(`cancel stages not wired: ${missing.join(" ")}`);
  }

  laneBytes() { const b = this.bytes; return b.PLAN + b.LISTS2 + b.ARGS2 + b.CANCEL + b.BCOUNTS2 + b.PIC + b.REF + b.BPART + b.FITP; }

  // ai: One lane's buffers (DESIGN 13.2), counted into ln.bytes. Before soft.lane and ldpc.lane, which bind LISTS2
  // ai: and BCOUNTS2 for their second dispatch.
  lane(ln) {
    const d = this.device, b = this.bytes;
    const mk = (size, usage) => { const buf = d.createBuffer({ size, usage }); ln.bytes = (ln.bytes ?? 0) + size; return buf; };
    ln.PLAN = mk(b.PLAN, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.LISTS2 = mk(b.LISTS2, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.ARGS2 = mk(b.ARGS2, U.STORAGE | U.INDIRECT | U.COPY_SRC | U.COPY_DST);
    ln.CANCEL = mk(b.CANCEL, U.STORAGE | U.UNIFORM | U.COPY_SRC | U.COPY_DST);
    ln.BCOUNTS2 = mk(b.BCOUNTS2, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.PIC = mk(b.PIC, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.REF = mk(b.REF, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    ln.BPART = mk(b.BPART, U.STORAGE);
    ln.FITP = mk(b.FITP, U.STORAGE | U.COPY_SRC);
    return ln;
  }

  // ai: The gate, zero and carry bind groups, once the lane holds REC (ldpc.lane); the paint's and the fit's too
  // ai: (the fit binds the lane's grid here when it has one; a picture is bound through bindPicture).
  bind(ln) {
    const d = this.device;
    const group = (bgl, entries) => d.createBindGroup({ layout: bgl, entries: entries.map((e, binding) => ({ binding, resource: e.buffer ? e : { buffer: e } })) });
    ln.gate2Group = group(this.gate2Bgl, [ln.REC, ln.BCOUNTS, ln.LISTS, ln.PLAN, ln.LISTS2, ln.ARGS2, ln.BCOUNTS2, ln.CANCEL, this.gateUni, { buffer: this.CARRY, offset: 0, size: 16 }]);
    ln.zeroGroup = group(this.zeroBgl, [ln.PLAN, ln.S]);
    ln.carryGroup = group(this.carryBgl, [ln.PLAN, ln.REF, this.CARRY]);
    this.paint?.lane(ln);
    this.fit?.lane(ln);
    return ln;
  }

  // ai: The fit's picture: the fused variant binds the texture, maps and residuals as pass 1 does; the grid one
  // ai: the grid. Both after bind(ln).
  bindPicture(ln, picture) { if (this.fit?.fusable) this.fit.bindPicture(ln, picture); }
  bindGrid(ln, gridBuf) { this.fit?.bindGrid(ln, gridBuf); }

  // ai: The carry in: the shared carry's slot into the lane's slot R, on the encoder before the lane's pass two.
  prepare(enc, ln) {
    enc.copyBufferToBuffer(this.CARRY, CARRY_HEADER, ln.REF, this.dims.refSlots * this.dims.refStride, this.dims.refStride);
  }

  // ai: One named dispatch of this stage on an open pass; picture ("grid" | "fused") is the batch's, which the fit
  // ai: samples as pass 1 does.
  dispatch(p, ln, name, picture = "grid") {
    const { refSlots } = this.dims, st = this.stages;
    switch (name) {
      case "gate2": p.setPipeline(this.gate2); p.setBindGroup(0, ln.gate2Group); p.dispatchWorkgroups(1); return;
      case "paint":
        p.setPipeline(this.zero); p.setBindGroup(0, ln.zeroGroup); p.dispatchWorkgroups(Math.ceil(this.sVec4 / 256), 1, refSlots);
        st.paint(p, ln, picture);
        return;
      case "carry": p.setPipeline(this.carry); p.setBindGroup(0, ln.carryGroup); p.dispatchWorkgroups(Math.ceil(this.refVec4 / 4096), 1, 1); return;
      default:
        if (!STAGES.includes(name)) throw new Error(`cancel pass ${name}`);
        st[name](p, ln, picture);
    }
  }

  // ai: Host builders for the stage tests (scripts/gpu/back/test_page.mjs "paint" and "fit"), laying PLAN, LISTS2, ARGS2 and
  // ai: CANCEL out as the gate does.
  // ai: slots[r] = { key, size, blocks: [[b, rec], ...] } (rec the record's index in REC) paints slot r from
  // ai: KEYTAB entry r; frames[f] = { fb, fa, slotA, slotB, nrefs, short, T, keyOwn } fills FR (unset fields NONE).
  hostPlan({ slots = [], frames = [] } = {}) {
    const L = this.plan, { blocksMax } = this.dims, w = new Uint32Array(L.words);
    w.fill(NONE, L.keytab, L.keytab + KEYTAB_WORDS * 128);
    for (let e = 0; e < 128; e++) w.fill(0, L.keytab + KEYTAB_WORDS * e + 1, L.keytab + KEYTAB_WORDS * e + 5);
    for (let e = 0; e < 128; e++) { w[L.keytab + KEYTAB_WORDS * e + 6] = 0; w[L.keytab + KEYTAB_WORDS * e + 7] = 0; }
    w.fill(0, L.ptr, L.fr);
    w.fill(NONE, L.fr, L.sel);
    w.set([NONE, 0, 0, 0], L.sel);
    w.fill(NONE, L.slots, L.slots + L.nslots);
    slots.forEach((sl, r) => {
      if (!sl) return;
      if (r >= this.dims.refSlots) throw new Error(`slot ${r}: ${this.dims.refSlots} paint slots`);
      const e = L.keytab + KEYTAB_WORDS * r;
      w[e] = sl.key >>> 0; w[e + 5] = r; w[e + 6] = sl.blocks.length; w[e + 7] = (1 << sl.size) | 256;
      for (const [b, rec] of sl.blocks) {
        if (b >= blocksMax) throw new Error(`block ${b} of ${blocksMax}`);
        w[e + 1 + (b >> 5)] |= 1 << (b & 31);
        w[L.ptr + r * blocksMax + b] = rec + 1;
      }
      w[L.slots + r] = r;
    });
    frames.forEach((fr, f) => {
      if (!fr) return;
      const o = L.fr + FR_WORDS * f, v = (x) => (x === undefined || x === null ? NONE : x >>> 0);
      w.set([v(fr.fb), v(fr.fa), v(fr.slotA), v(fr.slotB), fr.nrefs ?? 0, fr.short ?? 0, fr.T ?? 0, v(fr.keyOwn)], o);
    });
    return w;
  }
  // ai: frames[f] = { size, w, h, listed, blocks, ring } (null empty): LISTS2 and ARGS2 over the listed frames by size, a
  // ai: frame's block count its blocks or its size's, its rows the ones that count takes, its ring (0 unless named),
  // ai: pass 2's workgroups the most rows of a size's listed frames.
  hostLists(frames) {
    const B = this.B, LS = B + 1, head = listsHead(B), lists = new Uint32Array(listsWords(B)), args = new Uint32Array(argsWords), rowsMax = new Array(SLOTS).fill(0);
    frames.forEach((fr, f) => {
      if (!fr) return;
      lists[head + 2 * f] = fr.w; lists[head + 2 * f + 1] = fr.h;
      if (!fr.listed || !this.at[fr.size]) return;
      const s = fr.size, k = Math.min(fr.blocks ?? this.at[s][2], this.at[s][2]), r = this.rowsBy[s][k], c = lists[s * LS];
      lists[listsBlocks(B) + f] = k; lists[listsRows(B) + f] = r; lists[listsRing(B) + f] = fr.ring ?? 0; rowsMax[s] = Math.max(rowsMax[s], r);
      lists[s * LS + 1 + c] = f; lists[s * LS] = c + 1;
    });
    for (let s = 0; s < SLOTS; s++) {
      const c = lists[s * LS], at = this.at[s] ?? [0, 0, 0, 0], a = ARGS_WORDS * s;
      args.set([at[0], 1, c, 1, 1, c, rowsMax[s], 1, c, at[2], 1, c], a);
    }
    return { lists, args };
  }
  // ai: frames[f] = { slotA, slotB, nrefs } (null empty): CANCEL's headers, coefficients zero, ok 0.
  hostCancel(frames) {
    const w = new Uint32Array(CANCEL_WORDS * this.B);
    frames.forEach((fr, f) => { if (fr) w.set([fr.slotA ?? NONE, fr.slotB ?? fr.slotA ?? NONE, fr.nrefs ?? 0, 0], CANCEL_WORDS * f); });
    return w;
  }
}

// ai: What a lane's BCOUNTS2 and CANCEL read back as, a frame (back.mjs parseReadback): the second pass's counters, the
// ai: references it was given and whether it was cancelled (a solve that succeeded), and the blocks pass two verified.
export function parseCancel(buffer, offset, { B }) {
  const counts2 = new Uint32Array(buffer.slice(offset, offset + 32 * B)), cancel = new Uint32Array(buffer.slice(offset + 32 * B, offset + 32 * B + 4 * CANCEL_WORDS * B));
  const frames = [];
  for (let f = 0; f < B; f++) {
    const c2 = Array.from(counts2.subarray(8 * f, 8 * f + 8)), h = cancel.subarray(CANCEL_WORDS * f, CANCEL_WORDS * (f + 1));
    frames.push({ counts2: c2, refs: h[2], cancelled: h[2] > 0 && h[3] !== 0, gained: c2[4], slots: [h[0], h[1]], ok: h[3], coef: Array.from(new Float32Array(h.buffer, h.byteOffset + 16, 18)) });
  }
  return frames;
}
