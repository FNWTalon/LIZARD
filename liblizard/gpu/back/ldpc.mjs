// Part C of the GPU back half, host side (DESIGN.md sections 6, 7 and 10): the LDPC, the CRC-32 and the
// compaction of verified blocks into records. In: L, BLK (part B), LISTS (part A's gate), the MAP table a bit
// map mode (bitmap.mjs) and the code's block row tables (Focus.ldpcCode); behind the bit maps in MAP, the learned
// stop rule's words (stop/rule.mjs netWords). Out: V (u32 a block: 0 not run,
// 1 verified, 2 declined, 3 stall or cap, 4 CRC), ITS (i32 a block, the C's codes), REC (u32 count, then 484-byte
// records: frame + 65536 block, iterations, 473 payload bytes, 3 pad), and BCOUNTS columns 3 (blocks tried),
// ai: 4 (verified), 5 (records written), 7 (iterations run). One dispatch (blocksMax, B, SLOTS) covers every size: a
// workgroup past its size's block or frame count returns. Every stride comes from dims.mjs derived().
// ai: Built fromGate (the composed chain, back.mjs), pass one's dispatch is the gate's instead
// ai: (wgsl/back_transform.mjs argsLdpc: the most blocks and frames of a listed size), so an empty slot costs no
// ai: workgroup.
import { ldpcSource, THREADS, WG_BYTES, WG_BYTES_I16, PARAMS_AT, netWgBytes } from "../wgsl/back_ldpc.mjs";
import { SLOTS } from "../wgsl/common.mjs";
import { argsLdpc } from "../wgsl/back_transform.mjs";
import { mapTable } from "./bitmap.mjs";
import { crcPowers, rowTable, stopRule, DEFAULT_STOP, DEFAULT_STOP2, FIRST_READ, PAYLOAD } from "./ref_ldpc.mjs";
import { netWords } from "./stop/rule.mjs";
import { derived } from "./dims.mjs";
import { computePipeline } from "../pipeline.mjs";

const U = globalThis.GPUBufferUsage;

export const REC_WORDS = 121;
export const REC_BYTES = 4 * REC_WORDS;

// col | shift << 16 a data slot, 15 a block row, from the C's lay table.
export function layWords(code) {
  const { nd, rows } = rowTable(code);
  if (nd !== 15 || rows.length !== 12) throw new Error(`the shader is written for 12 rows of 15 data slots, the code has ${rows.length} of ${nd}`);
  const w = new Uint32Array(180);
  rows.forEach((row, r) => row.forEach(([col, s], e) => { w[r * 15 + e] = col | (s << 16); }));
  return w;
}

export const packMap = (map) => { const w = new Uint32Array((map.length + 1) >> 1); for (let i = 0; i < map.length; i++) w[i >> 1] |= map[i] << (16 * (i & 1)); return w; };

// ai: sizes[s] = { n, subch, blocks, bitmap } at the size's index (sel[f].z), null where not served. code = the
// wasm's ldpcCode().code (its lay table is what H is rebuilt from). cap: records the result buffer holds a batch
// (0: B x blocksMax, every block of every frame). MAP is built here for each bit map mode in play, with the
// largest block count that mode serves (block b's row is the same in every format: the whitening runs over the
// frame's slots in pos order).
// variant: the shader's row kernel (wgsl/back_ldpc.mjs VARIANTS). "i16" is the default: the fastest of the variants
// whose decisions and iteration counts equal the C's, at 20.8 KB of workgroup memory.
// ai: stop: when a codeword that has not cleared is given up (ref_ldpc.mjs stopRule): "learned" (the default since
// ai: 2026-09-28) the net of stop/, "c" the C's stall rule and cap alone (the check that the arithmetic is the C's). A
// ai: page's ?ldpcstop= overrides it, so a harness can choose without a change to the callers (BackHalf.build does not
// ai: pass stop).
// ai: second: the cancel stage's second dispatch (pass two) runs its own pipeline under ref_ldpc.mjs DEFAULT_STOP2
// ai: (the stall rule and funnel2's checkpoints); nothing overrides it, and a build without the stage compiles none.
// ai: fromGate: pass one's dispatch comes from the lane's ARGS, which the gate writes each batch (argsLdpc); without
// ai: it (the stage alone, scripts/gpu/back/test_page.mjs) (blocksMax, frames, SLOTS).
export async function build(device, { B, sizes, code, cap = 0, variant = "i16", stop = DEFAULT_STOP, second = false, fromGate = false, log = () => {} }) {
  const forced = globalThis.location ? new URLSearchParams(globalThis.location.search).get("ldpcstop") : null;
  stop = forced ?? stop;
  if (!FIRST_READ.includes(stop)) throw new Error(`ldpc stop ${stop}: one of ${FIRST_READ.join(", ")}`);
  const rule = await stopRule(stop, code.m), rule2 = second ? await stopRule(DEFAULT_STOP2, code.m) : null;
  const need = (variant.startsWith("i16") ? WG_BYTES_I16 : WG_BYTES) + (rule.net ? netWgBytes(rule.net.hidden) : 0);
  if (need > device.limits.maxComputeWorkgroupStorageSize) throw new Error(`the LDPC (${variant}, stop ${stop}) needs ${need} B of workgroup memory, the device offers ${device.limits.maxComputeWorkgroupStorageSize}`);
  const dims = derived(sizes, { B, cap }), { served, lStride, blocksMax, recCap } = dims;
  const modes = new Map();
  for (const s of served) { const m = sizes[s].bitmap | 0; modes.set(m, Math.max(modes.get(m) ?? 0, sizes[s].blocks)); }
  const mapParts = [], mapOff = new Map();
  let mapEntries = 0;
  for (const [mode, blocks] of modes) {
    const map = mapTable({ blocks, mode });
    mapOff.set(mode, mapEntries); mapParts.push(packMap(map)); mapEntries += 2 * ((map.length + 1) >> 1);
  }
  // ai: the learned rule's words behind the bit maps (mapEntries is even: packMap fills whole words)
  const net = rule.net ? { hidden: rule.net.hidden, at: mapEntries / 2, words: netWords(rule.net) } : null;
  if (net) mapParts.push(net.words);
  const mapBuf = device.createBuffer({ size: 2 * mapEntries + (net ? net.words.byteLength : 0), usage: U.STORAGE | U.COPY_DST });
  { let off = 0; for (const p of mapParts) { device.queue.writeBuffer(mapBuf, off, p); off += p.byteLength; } }
  const uni = new Uint32Array(PARAMS_AT.words);
  for (const s of served) { uni[PARAMS_AT.sizes + 4 * s] = sizes[s].blocks; uni[PARAMS_AT.sizes + 4 * s + 1] = mapOff.get(sizes[s].bitmap | 0); }
  uni.set([B, lStride, blocksMax, recCap], PARAMS_AT.dims);
  uni.set(layWords(code), PARAMS_AT.lay);
  uni.set(crcPowers(PAYLOAD), PARAMS_AT.pw);
  const params = device.createBuffer({ size: uni.byteLength, usage: U.UNIFORM | U.COPY_DST });
  device.queue.writeBuffer(params, 0, uni);
  const bgl = device.createBindGroupLayout({ entries: ["ro", "ro", "ro", "ro", "rw", "rw", "rw", "rw", "uniform"].map((type, binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type === "uniform" ? "uniform" : type === "ro" ? "read-only-storage" : "storage" } })) });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  // ai: No error scope (../pipeline.mjs): both rules' kernels, and the back half's other stages, compile at once.
  const compile = (r) => computePipeline(device, { code: ldpcSource({ variant, funnel: r.funnel, net: r.net ? { hidden: net.hidden, at: net.at } : null }), layout, label: "ldpc" });
  const [pipeline, pipeline2] = await Promise.all([compile(rule), rule2 ? compile(rule2) : null]);
  const show = (r) => (r.net ? `: net ${r.net.hidden}x${r.net.hidden} at 1 in ${Math.round(1 / r.net.budget)}, ${net.words.length} words in MAP` : r.funnel.length ? `: ${r.funnel.map(([j, c]) => `${c} after ${j}`).join(", ")}` : "");
  log(`ldpc (${variant}, stop ${stop}${show(rule)}${rule2 ? `; pass two ${DEFAULT_STOP2}${show(rule2)}` : ""}): sizes ${served.map((s) => `${sizes[s].n}/${sizes[s].subch} (${sizes[s].blocks} blocks, bit map ${sizes[s].bitmap | 0})`).join(", ")}; L ${lStride} words a frame, ${blocksMax} blocks max, records cap ${recCap}, MAP ${(2 * mapEntries / 1024).toFixed(0)} KB, ${THREADS} threads and ${need} B of workgroup memory a codeword`);
  const bytes = { V: 4 * B * blocksMax, ITS: 4 * B * blocksMax, REC: 4 + REC_BYTES * recCap };
  return {
    B, sizes, served, dims, lStride, blocksMax, recCap, bytes, pipeline, pipeline2, bgl, mapBuf, params, variant, stop, rule, rule2,
    // lane: { L, BLK, LISTS, BCOUNTS } from parts A and B. Adds V, ITS, REC and this stage's bind group.
    // ai: With LISTS2 and BCOUNTS2 on the lane (the cancel stage), the second bind group over them too.
    lane(ln) {
      ln.V = device.createBuffer({ size: bytes.V, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.ITS = device.createBuffer({ size: bytes.ITS, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.REC = device.createBuffer({ size: bytes.REC, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
      ln.bytes = (ln.bytes ?? 0) + bytes.V + bytes.ITS + bytes.REC;
      const group = (lists, counts) => device.createBindGroup({ layout: bgl, entries: [ln.L, ln.BLK, lists, mapBuf, ln.V, ln.ITS, ln.REC, counts, params].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      ln.ldpcBind = group(ln.LISTS, ln.BCOUNTS);
      ln.ldpcBind2 = ln.LISTS2 ? group(ln.LISTS2, ln.BCOUNTS2) : null;
      return ln;
    },
    // One dispatch over every size, appended to an open compute pass; V, ITS and REC's count must be zero already
    // (the composed chain's gate zeroes them, encode() below clears them).
    // frames: the batch's slots, which bound every list's count (a part-filled batch dispatches none past them).
    // ai: second: over LISTS2, counting into BCOUNTS2, under pass two's stop rule (the cancel stage's pass two: a block
    // ai: pass one verified returns at once, the records append to REC). Built fromGate, pass one takes the gate's
    // ai: dispatch (argsLdpc) and frames goes unused.
    dispatch(p, ln, frames = B, second = false) {
      const pipe = second ? pipeline2 : pipeline, group = second ? ln.ldpcBind2 : ln.ldpcBind;
      if (!pipe || !group) throw new Error("ldpc: pass two on a stage or lane built without it");
      p.setPipeline(pipe);
      p.setBindGroup(0, group);
      if (!second && fromGate) p.dispatchWorkgroupsIndirect(ln.ARGS, 4 * argsLdpc);
      else p.dispatchWorkgroups(blocksMax, frames, SLOTS);
    },
    // The verdicts and the record count cleared, then one compute pass over every size.
    encode(enc, ln, { timestampWrites } = {}) {
      enc.clearBuffer(ln.V);
      enc.clearBuffer(ln.ITS);
      enc.clearBuffer(ln.REC, 0, 4);
      const p = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
      this.dispatch(p, ln);
      p.end();
    },
  };
}

// The records of one readback laid out a block: REC bytes (count first) to per frame verdict, iterations and
// payload arrays the C's focus_assemble takes (Focus.rxAssemble). Frames beyond those in the batch are ignored.
export function unpackRecords(recBytes, vWords, itsWords, { B, blocksMax, recCap }) {
  const dv = new DataView(recBytes.buffer, recBytes.byteOffset, recBytes.byteLength);
  const count = Math.min(dv.getUint32(0, true), recCap);
  const frames = [];
  for (let f = 0; f < B; f++) frames.push({ verdict: new Uint8Array(blocksMax), its: new Int8Array(blocksMax), bytes: new Uint8Array(blocksMax * PAYLOAD), records: 0 });
  for (let f = 0; f < B; f++) for (let b = 0; b < blocksMax; b++) { frames[f].verdict[b] = vWords[f * blocksMax + b]; frames[f].its[b] = itsWords[f * blocksMax + b]; }
  for (let r = 0; r < count; r++) {
    const o = 4 + REC_BYTES * r, tag = dv.getUint32(o, true), f = tag & 0xffff, b = tag >>> 16;
    if (f >= B || b >= blocksMax) continue;
    frames[f].bytes.set(new Uint8Array(recBytes.buffer, recBytes.byteOffset + o + 8, PAYLOAD), b * PAYLOAD);
    frames[f].records++;
  }
  return { count, frames };
}
