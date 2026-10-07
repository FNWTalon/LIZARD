// Part B of the GPU back half, host side (DESIGN.md sections 4, 5 and 10): soft values, the per sub-channel
// estimate and the decline gate. In: S (part A), LISTS and ARGS (part A's gate), the UV table a size
// (bitmap.mjs discLayout or transform.mjs discLayout). Out: L (int8 slot order, 4 a u32, lStride words a frame),
// EST (f32 a sub-channel), BLK (a block: bitcast f32 estimate, u32 declined), and column 6 of BCOUNTS
// (sub-channels quantised). Every stride comes from dims.mjs derived(), the one place the chain sizes itself from.
// One dispatch a size: direct, (blocks_s, 1, B) with the kernel reading the size's frame count from LISTS, or
// indirect from the gate's blocks slot of ARGS, (blocks_s, 1, count_s), when `indirect` is set (measured within
// 3% of each other, DESIGN section "as built: B"; direct is the default).
// ai: The stage holds uvOff[s], size s's entry offset in the one UV buffer, for the cancel stage's paint (paint.mjs).
import { softSource, alignSource, THREADS, PILOT_FROM, PILOT_BLOCKS } from "../wgsl/back_soft.mjs";
import { whiten } from "./bitmap.mjs";
import { ARGS_WORDS, ARGS_SLOT, SLOTS, argsWords, listsWords, listsBlocks } from "../wgsl/back_transform.mjs";
import { derived } from "./dims.mjs";
import { computePipeline } from "../pipeline.mjs";

const U = globalThis.GPUBufferUsage;   // undefined under node, where only listsAndArgs and the constants are imported

export const DECLINE_BAR = Math.fround(Math.fround(0.2 * 3816) / 5088);   // FOCUS_DECLINE k / n, the C's float

// ai: The pilots' signs (SPEC 7.3): a word a block, bit i the whitening of slot PILOT_FROM + i of block b (5120 b +
// ai: PILOT_FROM + i of the frame, the whitening depending on the slot alone, so one table serves every format).
export function tailWords(blocks = PILOT_BLOCKS) {
  const w = whiten(5120 * blocks), out = new Uint32Array(blocks);
  for (let b = 0; b < blocks; b++) for (let i = 0; i < 5120 - PILOT_FROM; i++) out[b] |= w[5120 * b + PILOT_FROM + i] << i;
  return out;
}

// ai: Byte offset of size s's blocks slot in ARGS: u32 [SLOTS][ARGS_WORDS] with (blocks_s, 1, count_s) at
// ai: ARGS_SLOT.blocks.
export const argsOffset = (s) => 4 * (ARGS_WORDS * s + ARGS_SLOT.blocks);

// sizes[s] = { n, subch, blocks } (null for a size not served); tables[s] = { uv | UV: Uint32Array subch * 320 }.
// precision: "f32" or "f16" (S's element; the arithmetic is f32 either way). indirect: dispatch from ARGS.
export async function build(device, { B, sizes, tables, precision = "f32", indirect = false, cap = 0, log = () => {} }) {
  if (precision !== "f32" && precision !== "f16") throw new Error(`precision ${precision}`);
  const dims = derived(sizes, { B, cap }), { served, subchMax, blocksMax, sStride, lStride } = dims;
  for (const s of served) if (!(tables[s]?.uv ?? tables[s]?.UV)) throw new Error(`size ${s} (${sizes[s].n}/${sizes[s].subch}) has no UV table`);
  // The UV tables, one buffer, each size at its own entry offset.
  // ai: and after them all, at the same offsets again, each coefficient's (u, v) (transform.mjs discLayout UC) for
  // ai: the turn by the grid's shift; a table that carries none (a test's own) leaves zeros, which fit no shift.
  let uvTotal = 0;
  const uvOff = [];
  for (const s of served) { uvOff[s] = uvTotal; uvTotal += (tables[s].uv ?? tables[s].UV).length; }
  const uvBuf = device.createBuffer({ size: 8 * uvTotal, usage: U.STORAGE | U.COPY_DST });
  for (const s of served) {
    device.queue.writeBuffer(uvBuf, 4 * uvOff[s], tables[s].uv ?? tables[s].UV);
    const uc = tables[s].uc ?? tables[s].UC;
    if (uc) device.queue.writeBuffer(uvBuf, 4 * (uvTotal + uvOff[s]), uc);
  }
  // ai: Params a size: 12 words (48 bytes), wgsl/back_soft.mjs Params.
  const params = [];
  for (const s of served) {
    const z = sizes[s], buf = device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST });
    const w = new ArrayBuffer(48), u32 = new Uint32Array(w), f32 = new Float32Array(w);
    u32.set([s, z.subch, z.blocks, uvOff[s], sStride, lStride, subchMax, blocksMax, B]);
    f32[9] = DECLINE_BAR;
    u32[10] = uvTotal + uvOff[s]; u32[11] = z.n;
    device.queue.writeBuffer(buf, 0, w);
    params[s] = buf;
  }
  // ai: The pilots' signs, a uniform every lane's groups share (PILOT is each lane's, and pass two's its own scratch).
  const twBuf = device.createBuffer({ size: 4 * PILOT_BLOCKS, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(twBuf, 0, tailWords());
  const bgl = device.createBindGroupLayout({ entries: ["ro", "ro", "ro", "rw", "rw", "rw", "rw", "uniform", "rw", "uniform"].map((type, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type === "uniform" ? "uniform" : type === "ro" ? "read-only-storage" : "storage" } })) });
  // ai: No error scope (../pipeline.mjs): the back half builds its stages at once.
  const [pipeline, alignPipeline] = await Promise.all([
    computePipeline(device, { code: softSource({ f16: precision === "f16" }), layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), label: "soft" }),
    computePipeline(device, { code: alignSource({ f16: precision === "f16" }), layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), label: "align" })]);
  log(`soft: ${precision}, sizes ${served.map((s) => `${sizes[s].n}/${sizes[s].subch}`).join(" ")}, sStride ${sStride}, L ${lStride} words a frame, ${THREADS} threads a block, ${indirect ? "indirect" : "direct"} dispatch`);
  // ai: PILOT: a block's pilots, blocksMax a frame, then a frame's shift, two each (align)
  const bytes = { L: 4 * B * lStride, EST: 4 * B * subchMax, BLK: 8 * B * blocksMax, PILOT: 4 * B * blocksMax + 8 * B };
  return {
    precision, B, sizes, served, dims, sStride, lStride, subchMax, blocksMax, bytes, pipeline, alignPipeline, bgl, uvBuf, uvOff, uvTotal, params, indirect, twBuf,
    // lane: { S, LISTS, BCOUNTS, ARGS (indirect only) } from part A. Adds L, EST, BLK, PILOT and this stage's bind groups.
    // ai: With LISTS2 and BCOUNTS2 on the lane (the cancel stage), the second bind groups over them too, their pilots
    // ai: into PILOT2, which nothing reads (pass one's stand).
    lane(ln) {
      ln.L = device.createBuffer({ size: bytes.L, usage: U.STORAGE | U.COPY_SRC });
      ln.EST = device.createBuffer({ size: bytes.EST, usage: U.STORAGE | U.COPY_SRC });
      ln.BLK = device.createBuffer({ size: bytes.BLK, usage: U.STORAGE | U.COPY_SRC });
      ln.PILOT = device.createBuffer({ size: bytes.PILOT, usage: U.STORAGE | U.COPY_SRC });
      ln.bytes = (ln.bytes ?? 0) + bytes.L + bytes.EST + bytes.BLK + bytes.PILOT;
      if (ln.LISTS2) { ln.PILOT2 = device.createBuffer({ size: bytes.PILOT, usage: U.STORAGE }); ln.bytes += bytes.PILOT; }
      // ai: Keyed by the size index, as dispatch reads them: a dense list broke a build serving one size that is not
      // ai: the first (test_back's negN job), which every other build masked by serving size 0.
      const group = (lists, counts, pilot) => Object.fromEntries(served.map((s) => [s, device.createBindGroup({ layout: bgl, entries: [ln.S, uvBuf, lists, ln.L, ln.EST, ln.BLK, counts, params[s], pilot, twBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) })]));
      ln.softBind = group(ln.LISTS, ln.BCOUNTS, ln.PILOT);
      ln.softBind2 = ln.LISTS2 ? group(ln.LISTS2, ln.BCOUNTS2, ln.PILOT2) : null;
      return ln;
    },
    // One dispatch a served size, appended to an open compute pass. frames: the batch's slots, which bound every
    // list's count, so a part-filled batch dispatches no workgroup past them.
    // ai: second: over LISTS2 and ARGS2, counting into BCOUNTS2 (the cancel stage's pass two).
    // ai: Before it, on the same bind groups, the shift's fit (align): a workgroup a frame, a dispatch a size.
    // ai: skip: a picture slot left out (a rate profile's, tiers.mjs, whose own kernels read it), -1 none.
    dispatch(p, ln, frames = B, second = false, skip = -1) {
      const groups = second ? ln.softBind2 : ln.softBind, args = second ? ln.ARGS2 : ln.ARGS;
      if (!groups) throw new Error("soft: no bind groups over the second lists");
      p.setPipeline(alignPipeline);
      for (const s of served) { if (s === skip) continue; p.setBindGroup(0, groups[s]); p.dispatchWorkgroups(1, 1, frames); }
      p.setPipeline(pipeline);
      for (const s of served) {
        if (s === skip) continue;
        p.setBindGroup(0, groups[s]);
        if (indirect) p.dispatchWorkgroupsIndirect(args, argsOffset(s)); else p.dispatchWorkgroups(sizes[s].blocks, 1, frames);
      }
    },
    // One compute pass.
    encode(enc, ln, { timestampWrites } = {}) {
      const p = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
      this.dispatch(p, ln);
      p.end();
    },
  };
}

// The host side of LISTS and ARGS for a batch whose frames are already sized (the standalone tests; behind the
// front half part A's gate writes them on the device). frames[f] = size id or -1. Only the blocks slot is filled.
// ai: A listed frame's block count is its size's (no format word read).
export function listsAndArgs(frames, B, blocksOf) {
  const lists = new Uint32Array(listsWords(B)), args = new Uint32Array(argsWords);
  for (let s = 0; s < SLOTS; s++) {
    let c = 0;
    for (let f = 0; f < frames.length; f++) if (frames[f] === s) { lists[s * (1 + B) + 1 + c++] = f; lists[listsBlocks(B) + f] = blocksOf(s); }
    lists[s * (1 + B)] = c;
    const o = argsOffset(s) / 4;
    args[o] = c ? blocksOf(s) : 0; args[o + 1] = 1; args[o + 2] = c;
  }
  return { lists, args };
}
