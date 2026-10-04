// ai: The GPU back half composed (DESIGN.md section 11): the transform (A), the soft values (B) and the LDPC with its
// ai: CRC and compaction (C) as one object that owns the stages, their tables and one set of derived strides
// ai: (dims.mjs), makes a lane's buffers over the front half's grid, frames and sel buffers, and appends the chain's
// ai: six dispatches to a batch's command stream. What the host reads back is the result buffer of verified blocks
// ai: (REC: id and payload a record), the verdict a block (V, ITS) and the counters a frame (BCOUNTS); nothing else
// ai: leaves the device. Runs in the browser; node imports it for parseReadback and the constants.
//
//   const bh = await BackHalf.build(device, { B, sizes, precision, cap, picture });
//   const ln = bh.lane({ gridBuf, gridStride, framesBuf, selBuf });   // one a lane in flight
//   bh.bindPicture(ln, { texView, mapsBuf, residBuf });   // built with picture: pass 1 may sample the frame itself
//   bh.encode(enc, ln, { ts, fused });   // or bh.encode(pass, ln, { fused }) inside an open compute pass
//   bh.readback(enc, ln, readBuf, offset); ... parseReadback(mapped, offset, bh.dims)
// scripts/gpu/back/test_back.mjs runs exactly this on grids from the C's own registration (DESIGN section 11).
// ai: Built with cancel, pass two (cancel.mjs, DESIGN section 13) follows pass one on the same lane in the same
// ai: command buffer: bh.prepareCancel(enc, ln) on the encoder before the pass that holds it, then encode and
// ai: encodeCancel(target, ln, { fused, frames }). The readback then holds both passes: REC pass one's records and
// ai: pass two's after them, V and ITS the last verdict a block, BCOUNTS pass one's counters, and BCOUNTS2 and CANCEL
// ai: pass two's (a frame's references, its fit, the blocks gained).
import { Transform, transformTables } from "./transform.mjs";
import { build as buildSoft } from "./soft.mjs";
import { build as buildLdpc, unpackRecords, REC_BYTES } from "./ldpc.mjs";
import { Cancel, PASSES2, SECOND, parseCancel } from "./cancel.mjs";
import { derived } from "./dims.mjs";
import { PAYLOAD } from "./ref_ldpc.mjs";
import { pass1WorkgroupBytes, SLOTS } from "../wgsl/back_transform.mjs";
import { WG_BYTES, WG_BYTES_I16, VARIANTS } from "../wgsl/back_ldpc.mjs";

export const PASSES = ["gate", "pass1", "reduce", "pass2", "soft", "ldpc"];
export { PASSES2 };
// BCOUNTS: u32 [B][8] a batch, what each column holds and who writes it.
export const COUNT = { columns: 0, rows: 1, stored: 2, tried: 3, verified: 4, records: 5, subch: 6, iterations: 7 };
export const COUNTS = 8;

export class BackHalf {
  // ai: sizes: the decoder's picture sizes, [{ n, subch, bitmap }] with a size's index its position and sel[f].z
  // ai: (gpu/tables.mjs PICTURES, each at its largest sub-channel count; at most SLOTS), null where none; bitmap left
  // ai: out is the format's own. A size whose pass 1 does not fit the device's workgroup memory is left unbuilt
  // ai: (logged), and the gate leaves its frames, and any frame whose sel[f].z is SLOTS or more or whose sel[f].y is
  // ai: under 2, off every list.
  // precision: "f16" keeps Y and S as f16 where the device has shader-f16 (the iGPU, phones), else f32 with a log.
  // cap: records the result buffer holds a batch (0: every block of every frame, B x blocksMax).
  // variant: the LDPC row kernel (wgsl/back_ldpc.mjs VARIANTS). grid: the grid buffer's element ("f32" today).
  // indirect: B dispatched from the gate's blocks slot of ARGS instead of (blocks, 1, B).
  // ai: picture: the front half's rings, [{ span, margin, lattice }] (gpu/tables.mjs rings: span, margin, nodeX): builds
  // ai: the fused pass 1 too, which samples the frame through the front half's map, residuals and texture on its (ring,
  // ai: picture) pair's grid, so no grid is written or read.
  // ai: cancel: the cancel stage (cancel.mjs) is built, its buffers made a lane and its counters read back with the
  // ai: rest; encodeCancel runs it.
  static async build(device, { B, sizes, precision = "f32", cap = 0, variant = "i16", cpw, tw, grid = "f32", indirect = false, picture = null, cancel = false, log = () => {} }) {
    if (precision !== "f32" && precision !== "f16") throw new Error(`precision ${precision}`);
    if (precision === "f16" && !device.features.has("shader-f16")) { log("no shader-f16 on this adapter: back half in f32"); precision = "f32"; }
    if (sizes.length > SLOTS) throw new Error(`${sizes.length} sizes, ${SLOTS} slots`);
    if (!VARIANTS.includes(variant)) throw new Error(`ldpc variant ${variant}`);
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    const want = [];
    for (let s = 0; s < SLOTS; s++) {
      const z = sizes[s] ?? null;
      if (!z) { want.push(null); continue; }
      const need = pass1WorkgroupBytes(z.n, precision);
      if (need > limit) { log(`size ${s} (${z.n}/${z.subch}) not built: pass 1 needs ${need} B of workgroup memory, the device offers ${limit}`); want.push(null); continue; }
      want.push({ n: z.n, subch: z.subch, bitmap: z.bitmap });
    }
    const ldpcNeed = variant.startsWith("i16") ? WG_BYTES_I16 : WG_BYTES;
    if (ldpcNeed > limit) throw new Error(`the LDPC (${variant}) needs ${ldpcNeed} B of workgroup memory, the device offers ${limit}`);
    const tables = await transformTables(want);
    const bsizes = tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks, bitmap: tb.bitmap } : null));
    const dims = derived(bsizes, { B, cap });
    // One LDPC code for every format (DESIGN section 1); the tables say so or the build stops.
    const codes = tables.filter(Boolean).map((tb) => tb.code), key = (c) => JSON.stringify([c.n, c.k, c.m, c.z, c.mb, c.norm, Array.from(c.lay)]);
    if (codes.some((c) => key(c) !== key(codes[0]))) throw new Error("the built sizes do not share one LDPC code");
    const bh = new BackHalf();
    Object.assign(bh, { device, B, precision, cap, variant, grid, indirect, sizes: bsizes, tables, dims, code: codes[0], log });
    // ai: The stages built at once (2026-09-29, the GPU worker's start: STATUS "The ramp at the start"): none reads
    // ai: another's build, and none compiles under an error scope (../pipeline.mjs).
    [bh.transform, bh.soft, bh.ldpc, bh.cancel] = await Promise.all([
      Transform.build(device, { B, tables, precision, cpw, tw, grid, zero: true, blkStride: dims.blkStride, picture, cancel: cancel ? { refStride: dims.refStride, refMeans: dims.refMeans } : null }),
      buildSoft(device, { B, sizes: bsizes, tables, precision, indirect, cap, log }),
      buildLdpc(device, { B, sizes: bsizes, code: codes[0], cap, variant, second: !!cancel, fromGate: true, log }),
      cancel ? Cancel.build(device, { B, tables, precision, code: codes[0], dims, log }) : null,
    ]);
    bh.fusable = !!picture;
    if (bh.transform.sStride !== dims.sStride) throw new Error(`A's sStride ${bh.transform.sStride} against the chain's ${dims.sStride}`);
    // ai: The paint and the fit need the transform and the soft stage, so they are wired after every stage exists.
    if (bh.cancel) await bh.cancel.wire(bh);
    dims.cancel = !!bh.cancel;
    bh.bytes = { REC: bh.ldpc.bytes.REC, V: bh.ldpc.bytes.V, ITS: bh.ldpc.bytes.ITS, BCOUNTS: 4 * COUNTS * B, PILOT: bh.soft.bytes.PILOT, ...(bh.cancel ? { BCOUNTS2: 4 * COUNTS * B, CANCEL: bh.cancel.bytes.CANCEL } : {}) };
    // What one readback of the chain's outputs takes, timestamps and other copies aligned after it.
    bh.readBytes = (bh.bytes.REC + bh.bytes.V + bh.bytes.ITS + bh.bytes.BCOUNTS + bh.bytes.PILOT + (bh.bytes.BCOUNTS2 ?? 0) + (bh.bytes.CANCEL ?? 0) + 7) & ~7;
    log(`back half: ${precision}, sizes ${dims.served.map((s) => `${bsizes[s].n}/${bsizes[s].subch} (${bsizes[s].blocks} blocks, bit map ${bsizes[s].bitmap})`).join(", ")}; S ${dims.sStride} entries, L ${dims.lStride} words, ${dims.blkStride} blocks a frame; records cap ${dims.recCap}; readback ${(bh.readBytes / 1024).toFixed(0)} KB a batch`);
    return bh;
  }

  // One lane's buffers over the front half's: gridBuf (gridStride elements a frame, strip order; null when only the
  // ai: fused pass 1 will run, or until bindGrid), framesBuf (Frame { w, h, size, valid } a frame) and selBuf (vec4u
  // ai: (ring, 1 + version, picture slot, 0) a frame). Named as DESIGN section 2 names them (S, L,
  // EST, BLK, V, ITS, REC, LISTS, ARGS, BCOUNTS). The params hold gridStride, so a lane is made again (and the old
  // one destroyed) when the front half's picture capacity grows; bindGrid and bindPicture rebind in place.
  // ai: ln.bytes is the lane's device memory.
  lane({ gridBuf = null, gridStride, framesBuf, selBuf }) {
    const ln = this.transform.lane({ gridBuf, gridStride, framesBuf, selBuf });
    ln.S = ln.sBuf; ln.LISTS = ln.listsBuf; ln.ARGS = ln.argsBuf; ln.BCOUNTS = ln.countsBuf;
    // ai: The cancel stage's buffers first: the soft and LDPC stages bind LISTS2 and BCOUNTS2 for pass two.
    if (this.cancel) { this.cancel.lane(ln); this.transform.bindSecond(ln); }
    this.soft.lane(ln);
    this.ldpc.lane(ln);
    if (this.cancel) this.cancel.bind(ln);
    this.transform.bindGate(ln);
    return ln;
  }

  // The grid pass 1 over the front half's gridBuf, made after the lane (the decoder makes its grid only when a batch
  // asks for grids).
  bindGrid(ln, gridBuf) { this.transform.bindGrid(ln, gridBuf); this.cancel?.bindGrid(ln, gridBuf); }

  // The fused pass 1 over the front half's frame texture view (2d-array, a frame a layer), maps and residuals. Call
  // again whenever the front half replaces the texture.
  bindPicture(ln, picture) { this.transform.bindPicture(ln, picture); this.cancel?.bindPicture(ln, picture); }

  // ai: Before a lane's pass two, on the encoder: the cancel stage's carry in (a buffer copy into the lane's REF).
  prepareCancel(enc, ln) { this.cancel?.prepare(enc, ln); }

  // ai: The chain's six dispatches in order (PASSES). target: a GPUCommandEncoder (a compute pass a stage, a timestamp
  // ai: pair each at qs[base + 2k] when ts = { qs, base } is given), or an open GPUComputePassEncoder (the dispatches
  // ai: appended, no timestamps). Nothing is cleared from the host: the gate zeroes the counters, verdicts and record
  // ai: count. fused: pass 1 samples the frame (bindPicture) instead of reading the grid (bindGrid). frames: the
  // ai: batch's slots (at most B); the soft values and the LDPC dispatch only those.
  encode(target, ln, { ts = null, fused = false, frames = this.B } = {}) {
    const t = this.transform;
    this.steps(target, ts, [(p) => t.dispatchGate(p, ln), (p) => t.dispatchPass1(p, ln, fused ? "fused" : "grid"), (p) => t.dispatchReduce(p, ln), (p) => t.dispatchPass2(p, ln), (p) => this.soft.dispatch(p, ln, frames), (p) => this.ldpc.dispatch(p, ln, frames)]);
  }

  // ai: Pass two's fourteen dispatches (PASSES2) over a lane whose pass one has run, on a chain built with the cancel
  // ai: stage: the stage's own (c.dispatch) and its second chain (SECOND: pass 1's cancel variant, reduce, pass 2,
  // ai: soft, LDPC over LISTS2) from pass one's stages. fused and frames as pass one's: the fit samples the picture as
  // ai: its pass 1 did, soft2 and ldpc2 dispatch the batch's slots. target and ts as encode's.
  encodeCancel(target, ln, { ts = null, fused = false, frames = this.B } = {}) {
    if (!this.cancel) throw new Error("the back half was built without the cancel stage");
    const t = this.transform, c = this.cancel, picture = fused ? "fused" : "grid";
    const second = { pass1c: (p) => t.dispatchPass1(p, ln, picture, true), reduce2: (p) => t.dispatchReduce(p, ln, true), pass2b: (p) => t.dispatchPass2(p, ln, true), soft2: (p) => this.soft.dispatch(p, ln, frames, true), ldpc2: (p) => this.ldpc.dispatch(p, ln, frames, true) };
    this.steps(target, ts, PASSES2.map((name) => (SECOND.includes(name) ? second[name] : (p) => c.dispatch(p, ln, name, picture))));
  }

  steps(target, ts, steps) {
    if (typeof target.beginComputePass !== "function") { for (const step of steps) step(target); return; }
    steps.forEach((step, k) => {
      const p = target.beginComputePass(ts ? { timestampWrites: { querySet: ts.qs, beginningOfPassWriteIndex: ts.base + 2 * k, endOfPassWriteIndex: ts.base + 2 * k + 1 } } : {});
      step(p);
      p.end();
    });
  }

  // The copies of what the host reads, into readBuf at offset (a multiple of 4): REC, V, ITS, BCOUNTS in that order.
  // ai: then PILOT (a block's pilots, f32 blocksMax a frame, since 2026-09-30), then with the cancel stage BCOUNTS2
  // ai: and CANCEL. Encoded after the lane's last pass. Returns the bytes used (readBytes).
  readback(enc, ln, readBuf, offset = 0) {
    const parts = [[ln.REC, this.bytes.REC], [ln.V, this.bytes.V], [ln.ITS, this.bytes.ITS], [ln.BCOUNTS, this.bytes.BCOUNTS], [ln.PILOT, this.bytes.PILOT]];
    if (this.cancel) parts.push([ln.BCOUNTS2, this.bytes.BCOUNTS2], [ln.CANCEL, this.bytes.CANCEL]);
    let o = offset;
    for (const [buf, size] of parts) { enc.copyBufferToBuffer(buf, 0, readBuf, o, size); o += size; }
    return this.readBytes;
  }

  // Destroys a lane's buffers (the front half's three are the caller's).
  destroyLane(ln) {
    for (const [k, v] of Object.entries(ln)) if (v && typeof v.destroy === "function" && !["gridBuf", "framesBuf", "selBuf"].includes(k)) v.destroy();
  }
}

// One readback as the host reads it: the verified blocks as records (frame, block, iterations, id, the 473 payload
// bytes with the id first) and, a frame, the verdicts and iteration counts a block (blocksMax each) and the counters
// (COUNT). buffer: an ArrayBuffer holding a readback() range at offset. dims: { B, blocksMax, recCap } (BackHalf.dims
// with B, or a result header's).
// ai: With the cancel stage (dims.cancel) the records are pass one's, then from `first` on pass two's; a frame's
// ai: cancel is { counts2 (BCOUNTS2's columns), refs (references the gate gave it), cancelled (a solve that succeeded
// ai: on them), gained (blocks pass two verified), slots, ok, coef } (cancel.mjs parseCancel), null without it. V and
// ai: ITS are then pass two's on a block it gained.
export function parseReadback(buffer, offset, { B, blocksMax, recCap, cancel = false }) {
  const recBytes = 4 + REC_BYTES * recCap, vBytes = 4 * B * blocksMax, cBytes = 4 * COUNTS * B, pBytes = 4 * B * blocksMax + 8 * B;   // ai: the pilots, then a frame's shift
  const rec = new Uint8Array(buffer, offset, recBytes);
  const V = new Uint32Array(buffer.slice(offset + recBytes, offset + recBytes + vBytes));
  const ITS = new Int32Array(buffer.slice(offset + recBytes + vBytes, offset + recBytes + 2 * vBytes));
  const counts = new Uint32Array(buffer.slice(offset + recBytes + 2 * vBytes, offset + recBytes + 2 * vBytes + cBytes));
  const pilots = new Float32Array(buffer.slice(offset + recBytes + 2 * vBytes + cBytes, offset + recBytes + 2 * vBytes + cBytes + pBytes));
  const passTwo = cancel ? parseCancel(buffer, offset + recBytes + 2 * vBytes + cBytes + pBytes, { B }) : null;
  const { count, frames } = unpackRecords(rec, V, ITS, { B, blocksMax, recCap });
  const records = recordsOf(buffer, offset, count, { B, blocksMax });
  // ai: Pass one claims its records before pass two's first dispatch: as many as it verified.
  let first = count;
  if (passTwo) { let v = 0; for (let f = 0; f < B; f++) v += counts[COUNTS * f + COUNT.verified]; first = Math.min(v, count); }
  return {
    count, first, records, V, ITS, counts,
    frames: frames.map((fr, f) => ({ verdict: fr.verdict, its: fr.its, bytes: fr.bytes, records: fr.records, counts: Array.from(counts.subarray(COUNTS * f, COUNTS * (f + 1))), pilots: pilots.subarray(blocksMax * f, blocksMax * (f + 1)), shift: [pilots[B * blocksMax + 2 * f], pilots[B * blocksMax + 2 * f + 1]], cancel: passTwo ? passTwo[f] : null })),
  };
}

// ai: REC's first `count` records (the count word at offset, then REC_BYTES a record), a record's frame and block from
// ai: its tag word and index its place in REC; one past the batch's slots or the block count is skipped, as
// ai: unpackRecords skips it.
function recordsOf(buffer, offset, count, { B, blocksMax }) {
  const dv = new DataView(buffer, offset), records = [];
  for (let r = 0; r < count; r++) {
    const o = 4 + REC_BYTES * r, tag = dv.getUint32(o, true), f = tag & 0xffff, b = tag >>> 16;
    if (f >= B || b >= blocksMax) continue;
    const payload = new Uint8Array(buffer.slice(offset + o + 8, offset + o + 8 + PAYLOAD));
    records.push({ frame: f, block: b, its: dv.getInt32(o + 4, true), id: dv.getUint32(o + 8, true), payload, index: r });
  }
  return records;
}
