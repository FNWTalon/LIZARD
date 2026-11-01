// ai: The GPU back half composed (DESIGN.md section 9): the transform (A), the soft values (B) and the LDPC with its
// ai: CRC and compaction (C) as one object that owns the stages, their tables and one set of derived strides
// ai: (dims.mjs), built by the setup (../decoder.mjs, on ../../gen/mockgpu.mjs's recording device) and exported for the
// ai: native host (../../gen/gen.mjs exportBack, which reads every field built here). A lane's buffers, the chain's six
// ai: dispatches in a batch and its readback are the native host's (core/dec/front.cpp); what it reads back is the
// ai: result buffer of verified blocks (REC: id and payload a record), the verdict a block (V, ITS), the counters a
// ai: frame (BCOUNTS) and PILOT; nothing else leaves the device. The WebGPU host's lane, encode and readback went with
// ai: the WGSL decoder (archive/webgpu-decoder/).
//
//   const bh = await BackHalf.build(device, { B, sizes, precision, picture });
import { Transform } from "./transform.mjs";
import { transformTables, TRANSFORM_DOWN } from "./disc.mjs";
import { build as buildSoft } from "./soft.mjs";
import { build as buildLdpc } from "./ldpc.mjs";
import { derived } from "./dims.mjs";
import { pass1WorkgroupBytes, SLOTS } from "../plan/back_transform.mjs";
import { RuleStage } from "./tiers.mjs";
import { ruleTables } from "./codes.mjs";
import { NC } from "../plan/back_tiers.mjs";

// BCOUNTS: u32 [B][8] a batch, what each column holds and who writes it.
export const COUNT = { columns: 0, rows: 1, stored: 2, tried: 3, verified: 4, records: 5, subch: 6, iterations: 7 };
export const COUNTS = 8;

export class BackHalf {
  // ai: sizes: the decoder's picture sizes, [{ n, subch }] with a size's index its position and sel[f].z
  // ai: (gpu/tables.mjs PICTURES, each at its largest sub-channel count; at most SLOTS), null where none. A size whose
  // ai: pass 1 does not fit the device's workgroup memory is left unbuilt (logged), and the gate leaves its frames, and
  // ai: any frame whose sel[f].z is SLOTS or more or whose sel[f].y is under 2, off every list.
  // precision: "f16" keeps Y and S as f16 where the device has shader-f16 (the iGPU, phones), else f32 with a log.
  // ai: The result buffer holds every block of every frame (B x blocksMax records).
  // ai: picture: the front half's rings, [{ span, margin, lattice }] (gpu/tables.mjs rings: span, margin, nodeX): pass 1
  // ai: samples the frame through the front half's map, residuals and texture on its (ring, picture) pair's grid.
  static async build(device, { B, sizes, precision = "f32", picture, log = () => {} }) {
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
      want.push({ n: z.n, subch: z.subch, nt });
    }
    const tables = await transformTables(want);
    const bsizes = tables.map((tb) => (tb ? { n: tb.n, subch: tb.subch, blocks: tb.blocks } : null));
    const dims = derived(bsizes, { B });
    const bh = new BackHalf();
    Object.assign(bh, { device, B, precision, sizes: bsizes, tables, dims, log });
    // ai: The stages built at once (2026-09-29, the GPU worker's start: STATUS "The ramp at the start"): none reads
    // ai: another's build, and none compiles under an error scope (../pipeline.mjs).
    // ai: each version's blocks a code, a byte a code in CODES' order, the gate's codes table (the LDPC's dispatches)
    const rt = await ruleTables(), codes = new Uint32Array(rt.profiles.length);
    rt.profiles.forEach((p, v) => { for (const t of p ?? []) { if (t.count > 255 || t.c >= NC) throw new Error(`version ${v}: ${t.count} blocks of code ${t.c}: the gate's codes table holds a byte a code`); codes[v] |= t.count << (8 * t.c); } });
    [bh.transform, bh.soft, bh.ldpc] = await Promise.all([
      Transform.build(device, { B, tables, precision, blkStride: dims.blkStride, picture, codes }),
      buildSoft(device, { B, sizes: bsizes, tables }),
      buildLdpc(device, { B, sizes: bsizes }),
    ]);
    if (bh.transform.sStride !== dims.sStride) throw new Error(`A's sStride ${bh.transform.sStride} against the chain's ${dims.sStride}`);
    // ai: the format's rate profile (tiers.mjs): its stage decodes every version, over the soft and LDPC stages' tables
    // ai: and lane buffers
    bh.rule = await RuleStage.build(bh);
    bh.bytes = { REC: bh.ldpc.bytes.REC, V: bh.ldpc.bytes.V, ITS: bh.ldpc.bytes.ITS, BCOUNTS: 4 * COUNTS * B, PILOT: bh.soft.bytes.PILOT };
    // What one readback of the chain's outputs takes, timestamps and other copies aligned after it.
    bh.readBytes = (bh.bytes.REC + bh.bytes.V + bh.bytes.ITS + bh.bytes.BCOUNTS + bh.bytes.PILOT + 7) & ~7;
    log(`back half: ${precision}, sizes ${dims.served.map((s) => `${bsizes[s].n}/${bsizes[s].subch} (${bsizes[s].blocks} chunks)`).join(", ")}; S ${dims.sStride} entries, L ${dims.lStride} words, ${dims.blkStride} blocks a frame; records cap ${dims.recCap}; readback ${(bh.readBytes / 1024).toFixed(0)} KB a batch`);
    return bh;
  }
}
