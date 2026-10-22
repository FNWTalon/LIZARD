// ai: The GPU back half composed (DESIGN.md section 11): the transform (A), the soft values (B) and the LDPC with its
// ai: CRC and compaction (C) as one object that owns the stages, their tables and one set of derived strides
// ai: (dims.mjs), makes a lane's buffers over the front half's frames and sel buffers, and appends the chain's
// ai: six dispatches to a batch's command stream. What the host reads back is the result buffer of verified blocks
// ai: (REC: id and payload a record), the verdict a block (V, ITS) and the counters a frame (BCOUNTS); nothing else
// ai: leaves the device. Runs in the browser; node imports it for parseReadback and the constants.
//
//   const bh = await BackHalf.build(device, { B, sizes, precision, cap, picture });
//   const ln = bh.lane({ gridStride, framesBuf, selBuf });   // one a lane in flight
//   bh.bindPicture(ln, { texView, mapsBuf, residBuf });   // pass 1 samples the frame itself
//   bh.encode(enc, ln, { ts });   // or bh.encode(pass, ln) inside an open compute pass
//   bh.readback(enc, ln, readBuf, offset); ... parseReadback(mapped, offset, bh.dims)
import { Transform, transformTables, TRANSFORM_DOWN } from "./transform.mjs";
import { build as buildSoft } from "./soft.mjs";
import { build as buildLdpc, unpackRecords, REC_BYTES } from "./ldpc.mjs";
import { derived } from "./dims.mjs";
import { PAYLOAD } from "./ref_ldpc.mjs";
import { pass1WorkgroupBytes, SLOTS } from "../wgsl/back_transform.mjs";
import { RuleStage } from "./tiers.mjs";

export const PASSES = ["gate", "pass1", "reduce", "pass2", "soft", "ldpc"];
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
  // ai: picture: the front half's rings, [{ span, margin, lattice }] (gpu/tables.mjs rings: span, margin, nodeX): pass 1
  // ai: samples the frame through the front half's map, residuals and texture on its (ring, picture) pair's grid.
  static async build(device, { B, sizes, precision = "f32", cap = 0, cpw, tw, picture, log = () => {} }) {
    if (precision !== "f32" && precision !== "f16") throw new Error(`precision ${precision}`);
    if (precision === "f16" && !device.features.has("shader-f16")) { log("no shader-f16 on this adapter: back half in f32"); precision = "f32"; }
    if (sizes.length > SLOTS) throw new Error(`${sizes.length} sizes, ${SLOTS} slots`);
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    const want = [];
    for (let s = 0; s < SLOTS; s++) {
      const z = sizes[s] ?? null;
      if (!z) { want.push(null); continue; }
      // ai: pass 1 reads a picture at the size below its own (transform.mjs TRANSFORM_DOWN)
      const nt = TRANSFORM_DOWN[z.n] ?? z.n;
      const need = pass1WorkgroupBytes(nt, precision);
      if (need > limit) { log(`size ${s} (${z.n}/${z.subch}) not built: pass 1 needs ${need} B of workgroup memory, the device offers ${limit}`); want.push(null); continue; }
      want.push({ n: z.n, subch: z.subch, bitmap: z.bitmap, nt });
    }
    const tables = await transformTables(want);
    const bsizes = tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks, bitmap: tb.bitmap } : null));
    const dims = derived(bsizes, { B, cap });
    const bh = new BackHalf();
    Object.assign(bh, { device, B, precision, cap, sizes: bsizes, tables, dims, log });
    // ai: The stages built at once (2026-09-29, the GPU worker's start: STATUS "The ramp at the start"): none reads
    // ai: another's build, and none compiles under an error scope (../pipeline.mjs).
    [bh.transform, bh.soft, bh.ldpc] = await Promise.all([
      Transform.build(device, { B, tables, precision, cpw, tw, zero: true, blkStride: dims.blkStride, picture }),
      buildSoft(device, { B, sizes: bsizes, tables, cap }),
      buildLdpc(device, { B, sizes: bsizes, cap }),
    ]);
    if (bh.transform.sStride !== dims.sStride) throw new Error(`A's sStride ${bh.transform.sStride} against the chain's ${dims.sStride}`);
    // ai: the format's rate profile (tiers.mjs): its stage decodes every version, over the soft and LDPC stages' tables
    // ai: and lane buffers
    bh.rule = await RuleStage.build(bh);
    bh.bytes = { REC: bh.ldpc.bytes.REC, V: bh.ldpc.bytes.V, ITS: bh.ldpc.bytes.ITS, BCOUNTS: 4 * COUNTS * B, PILOT: bh.soft.bytes.PILOT };
    // What one readback of the chain's outputs takes, timestamps and other copies aligned after it.
    bh.readBytes = (bh.bytes.REC + bh.bytes.V + bh.bytes.ITS + bh.bytes.BCOUNTS + bh.bytes.PILOT + 7) & ~7;
    log(`back half: ${precision}, sizes ${dims.served.map((s) => `${bsizes[s].n}/${bsizes[s].subch} (${bsizes[s].blocks} blocks, bit map ${bsizes[s].bitmap})`).join(", ")}; S ${dims.sStride} entries, L ${dims.lStride} words, ${dims.blkStride} blocks a frame; records cap ${dims.recCap}; readback ${(bh.readBytes / 1024).toFixed(0)} KB a batch`);
    return bh;
  }

  // One lane's buffers over the front half's: framesBuf (Frame { w, h, size, valid } a frame) and selBuf (vec4u
  // ai: (ring, 1 + version, picture slot, 0) a frame). Named as DESIGN section 2 names them (S, L,
  // BLK, V, ITS, REC, LISTS, ARGS, BCOUNTS). The params hold gridStride (the picture capacity a frame), so a lane is
  // made again (and the old one destroyed) when the front half's picture capacity grows; bindPicture rebinds in place.
  // ai: ln.bytes is the lane's device memory.
  lane({ gridStride, framesBuf, selBuf }) {
    const ln = this.transform.lane({ gridStride, framesBuf, selBuf });
    ln.S = ln.sBuf; ln.LISTS = ln.listsBuf; ln.ARGS = ln.argsBuf; ln.BCOUNTS = ln.countsBuf;
    this.soft.lane(ln);
    this.ldpc.lane(ln);
    this.rule.lane(ln);
    this.transform.bindGate(ln);
    return ln;
  }

  // The fused pass 1 over the front half's frame texture view (2d-array, a frame a layer), maps and residuals. Call
  // again whenever the front half replaces the texture.
  bindPicture(ln, picture) { this.transform.bindPicture(ln, picture); }

  // ai: The chain's six dispatches in order (PASSES). target: a GPUCommandEncoder (a compute pass a stage, a timestamp
  // ai: pair each at qs[base + 2k] when ts = { qs, base } is given), or an open GPUComputePassEncoder (the dispatches
  // ai: appended, no timestamps). Nothing is cleared from the host: the gate zeroes the counters, verdicts and record
  // ai: count. frames: the batch's slots (at most B); the soft values and the LDPC dispatch only those.
  encode(target, ln, { ts = null, frames = this.B } = {}) {
    const t = this.transform, r = this.rule;
    // ai: the soft values and the LDPC by the format's rate profile (tiers.mjs)
    this.steps(target, ts, [(p) => t.dispatchGate(p, ln), (p) => t.dispatchPass1(p, ln), (p) => t.dispatchReduce(p, ln), (p) => t.dispatchPass2(p, ln), (p) => r.dispatchSoft(p, ln, frames), (p) => r.dispatchLdpc(p, ln, frames)]);
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
  // ai: then PILOT (a block's pilots, f32 blocksMax a frame, since 2026-09-30). Encoded after the lane's last pass.
  // ai: Returns the bytes used (readBytes).
  readback(enc, ln, readBuf, offset = 0) {
    const parts = [[ln.REC, this.bytes.REC], [ln.V, this.bytes.V], [ln.ITS, this.bytes.ITS], [ln.BCOUNTS, this.bytes.BCOUNTS], [ln.PILOT, this.bytes.PILOT]];
    let o = offset;
    for (const [buf, size] of parts) { enc.copyBufferToBuffer(buf, 0, readBuf, o, size); o += size; }
    return this.readBytes;
  }

  // Destroys a lane's buffers (the front half's three are the caller's).
  destroyLane(ln) {
    for (const [k, v] of Object.entries(ln)) if (v && typeof v.destroy === "function" && !["framesBuf", "selBuf"].includes(k)) v.destroy();
  }
}

// One readback as the host reads it: the verified blocks as records (frame, block, iterations, id, the 473 payload
// bytes with the id first) and, a frame, the verdicts and iteration counts a block (blocksMax each) and the counters
// (COUNT). buffer: an ArrayBuffer holding a readback() range at offset. dims: { B, blocksMax, recCap } (BackHalf.dims
// with B, or a result header's).
export function parseReadback(buffer, offset, { B, blocksMax, recCap }) {
  const recBytes = 4 + REC_BYTES * recCap, vBytes = 4 * B * blocksMax, cBytes = 4 * COUNTS * B, pBytes = 4 * B * blocksMax + 8 * B;   // ai: the pilots, then a frame's shift
  const rec = new Uint8Array(buffer, offset, recBytes);
  const V = new Uint32Array(buffer.slice(offset + recBytes, offset + recBytes + vBytes));
  const ITS = new Int32Array(buffer.slice(offset + recBytes + vBytes, offset + recBytes + 2 * vBytes));
  const counts = new Uint32Array(buffer.slice(offset + recBytes + 2 * vBytes, offset + recBytes + 2 * vBytes + cBytes));
  const pilots = new Float32Array(buffer.slice(offset + recBytes + 2 * vBytes + cBytes, offset + recBytes + 2 * vBytes + cBytes + pBytes));
  const { count, frames } = unpackRecords(rec, V, ITS, { B, blocksMax, recCap });
  const records = recordsOf(buffer, offset, count, { B, blocksMax });
  return {
    count, records, V, ITS, counts,
    frames: frames.map((fr, f) => ({ verdict: fr.verdict, its: fr.its, bytes: fr.bytes, records: fr.records, counts: Array.from(counts.subarray(COUNTS * f, COUNTS * (f + 1))), pilots: pilots.subarray(blocksMax * f, blocksMax * (f + 1)), shift: [pilots[B * blocksMax + 2 * f], pilots[B * blocksMax + 2 * f + 1]] })),
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
