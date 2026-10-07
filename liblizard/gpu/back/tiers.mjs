// ai: The rate profile on the GPU (the lab's rate-by-ring arm, 2026-10-07; the kernels and their contract in
// ai: ../wgsl/back_tiers.mjs). A profile is "7/8:24,3/4:15,1/2:12" (src/any.h focus_tiers_parse's text: the inner tier
// ai: first, a rate by name or ldpc.h index, its blocks), its key the indices "6:24,4:15,2:12". Its frames are the ones
// ai: whose word names its sub-channel count; they decode at its picture's slot, which then decodes no other version.
//
//   const st = await bh.addTiers("7/8:24,3/4:15,1/2:12");   // a BackHalf (back.mjs); tables, buffers, pipelines
//   bh.useTiers(st.key);    // encode() runs it from the next batch (null: one rate everywhere again)
//   st.exportFor(idOf)      // what the native host replays (../../gen/gen.mjs, core/dec/front.cpp)
import { init, Focus } from "../../sim/ob.mjs";
import { N_FOR } from "../../sim/lizard_pick.mjs";
import { whiten, perm, inverse } from "./bitmap.mjs";
import { packMap } from "./ldpc.mjs";
import { crcPowers, PAYLOAD } from "./ref_ldpc.mjs";
import { SLOTS } from "../wgsl/common.mjs";
import { argsLdpc } from "../wgsl/back_transform.mjs";
import { talignSource, tsoftSource, tblocksSource, tldpcSource, tierShape, tierWgBytes, tierUniformWords, TIER_BLOCKS, TJ_SUBS, KNOWN } from "../wgsl/back_tiers.mjs";
import { computePipeline } from "../pipeline.mjs";

const U = globalThis.GPUBufferUsage ?? { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
export const RATE_NAMES = ["1/4", "1/3", "1/2", "2/3", "3/4", "5/6", "7/8"];
// ai: The rates the GPU's tier kernels take (sub-channels a block from ../../sim/ob.mjs Focus.SUBS: 12, 9, 8, 7).
export const GPU_RATES = [2, 3, 4, 6];

// ai: A profile's text to its tiers [[rate, blocks], ...] and its key. Throws on what the C refuses too (src/any.c
// ai: focus_tiers_parse): more than three tiers, a sum not a multiple of 8, no 3/4 tier; and on a rate the GPU lacks.
export function parseTiers(text) {
  const tiers = [];
  for (const part of String(text).split(/[\s,;]+/).filter(Boolean)) {
    const m = /^(\d\/\d|\d):(\d+)$/.exec(part);
    if (!m) throw new Error(`profile ${text}: a tier is <rate>:<blocks>, not ${part}`);
    const rate = m[1].includes("/") ? RATE_NAMES.indexOf(m[1]) : +m[1], blocks = +m[2];
    if (!GPU_RATES.includes(rate)) throw new Error(`profile ${text}: the GPU takes 7/8, 3/4, 2/3 and 1/2, not ${m[1]}`);
    if (!(blocks >= 1 && blocks <= 1024)) throw new Error(`profile ${text}: ${blocks} blocks`);
    tiers.push([rate, blocks]);
  }
  if (tiers.length < 1 || tiers.length > 3) throw new Error(`profile ${text}: 1 to 3 tiers`);
  const subch = tiers.reduce((a, [r, b]) => a + b * Focus.SUBS[r], 0), blocks = tiers.reduce((a, [, b]) => a + b, 0);
  if (subch % 8) throw new Error(`profile ${text}: ${subch} sub-channels, not a multiple of 8 (the word's version)`);
  if (!tiers.some(([r]) => r === 4)) throw new Error(`profile ${text}: no 3/4 tier, so its block is not 473 B`);
  return { tiers, subch, blocks, key: tiers.map(([r, b]) => `${r}:${b}`).join(","), label: tiers.map(([r, b]) => `${RATE_NAMES[r]} x ${b}`).join(", ") };
}

// ai: The profile's tables, from the codec (../../sim/ob.mjs Focus with tiers, the wasm holding it for the call):
// ai: blocks [{ first, subs, tier }], codes (each tier's ldpcCode-shaped code), the whitening over its slots, the
// ai: bit map mode, its pos table (to hold against the slot's), n.
export async function tierTables(text) {
  const p = parseTiers(text), M = await init();
  const n = N_FOR(p.subch);
  const fc = new Focus(n, 0, 1, { tiers: p.tiers });
  try {
    const shape = fc.shape(), pos = fc.posTable(), bitmap = fc.bitmap;
    if (shape.subch !== p.subch || shape.blocks !== p.blocks || fc.blockBytes !== PAYLOAD) throw new Error(`profile ${p.label}: the codec made ${shape.subch} sub-channels, ${shape.blocks} blocks of ${fc.blockBytes} B`);
    const pFirst = M._malloc(4 * p.blocks), pCount = M._malloc(4 * p.blocks), pTier = M._malloc(4 * p.blocks), pD = M._malloc(64);
    M._focus_block_subs(pFirst, pCount);
    M._focus_block_tiers(pTier);
    const first = M.HEAP32.slice(pFirst >> 2, (pFirst >> 2) + p.blocks), count = M.HEAP32.slice(pCount >> 2, (pCount >> 2) + p.blocks), tierOf = M.HEAP32.slice(pTier >> 2, (pTier >> 2) + p.blocks);
    const codes = p.tiers.map((_, t) => {
      M._focus_ldpc_tables(t, pD, 0);
      const d = M.HEAP32.slice(pD >> 2, (pD >> 2) + 11);
      const code = { n: d[0], k: d[1], m: d[2], z: d[3], zp: d[4], mb: d[5], kb: d[6], norm: d[7], slotsMax: d[8], slotsTotal: d[9], slots: d[10] };
      const pLay = M._malloc(4 * (code.mb + 1 + 2 * code.slots));
      M._focus_ldpc_tables(t, pD, pLay);
      code.lay = M.HEAP32.slice(pLay >> 2, (pLay >> 2) + code.mb + 1 + 2 * code.slots);
      M._free(pLay);
      return code;
    });
    [pFirst, pCount, pTier, pD].forEach((q) => M._free(q));
    const blocks = Array.from(first, (f, b) => ({ first: f, subs: count[b], tier: tierOf[b] }));
    return { ...p, n, pos, bitmap, blocks, codes };
  } finally { fc.free(); }
}

// ai: The uniform and storage contents a profile's stage binds, from its tables. MAP: each tier's rows (a block's
// ai: codeword bit j: its slot, bit 15 its whitening), tier after tier; TB, TJ and a tier's T as ../wgsl/back_tiers.mjs
// ai: lays them out.
export function tierContents(tt, { slot, dims, B }) {
  const white = tt.bitmap ? whiten(tt.subch * 640) : null;
  const version = tt.subch / 8, nb = tt.blocks.length;
  const tiers = tt.tiers.map(([rate, count], t) => {
    const code = tt.codes[t], sh = tierShape(code);
    const firstBlock = tt.blocks.findIndex((b) => b.tier === t);
    return { t, rate, count, code, sh, firstBlock, sub0: tt.blocks[firstBlock].first, subs: tt.blocks[firstBlock].subs };
  });
  // ai: the rows, a tier's blocks in order
  const rows = [], mapOff = [];
  let entries = 0;
  for (const tr of tiers) {
    const { n, k } = tr.code, inv = inverse(perm(n, k, tt.bitmap)), m = new Uint16Array(tr.count * n);
    for (let x = 0; x < tr.count; x++) {
      const s0 = 640 * tt.blocks[tr.firstBlock + x].first;
      for (let j = 0; j < n; j++) m[x * n + j] = inv[j] | (white && white[s0 + inv[j]] ? 0x8000 : 0);
    }
    mapOff.push(entries); rows.push(m); entries += m.length;
  }
  const all = new Uint16Array(entries + (entries & 1));
  { let o = 0; for (const m of rows) { all.set(m, o); o += m.length; } }
  const map = packMap(all);
  // ai: TB and TJ
  const tb = new Uint32Array(4 + 8 * TIER_BLOCKS), tj = new Uint32Array(4 + 2 * TJ_SUBS);
  tb.set([version, nb, slot, 0]);
  tj.set([version, slot, 0, 0]);
  for (let j = 0; j < TJ_SUBS; j++) tj[4 + 2 * j] = 320;
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  tt.blocks.forEach((b, i) => {
    const { n, k } = tt.codes[b.tier], last = b.first + b.subs - 1, tail = (640 * b.subs - n) / 2, s0 = 640 * b.first;
    let tw = 0;
    for (let q = 0; q < 2 * tail && q < 32; q++) if (white && white[s0 + n + q]) tw |= 1 << q;
    tw >>>= 0;
    f32[0] = Math.fround(Math.fround(0.2 * k) / n);
    tb.set([b.first, b.subs, last, tail], 4 + 8 * i);
    tb.set([320 * b.first + n / 2, tw, u32[0], 0], 8 + 8 * i);
    if (tail > 0) { tj[4 + 2 * last] = n / 2 - 320 * (b.subs - 1); tj[5 + 2 * last] = tw; }
  });
  // ai: a tier's T
  const pw = crcPowers(PAYLOAD);
  const ts = tiers.map((tr, q) => {
    const { sh, code } = tr, w = new Uint32Array(tierUniformWords(sh));
    w.set([B, dims.lStride, dims.blocksMax, dims.recCap], 0);
    w.set([slot, version, tr.firstBlock, tr.sub0], 4);
    w.set([tr.subs, mapOff[q], tr.count, 0], 8);
    let o = 12;
    for (let r = 0; r <= code.mb; r++) w[o + r] = code.lay[r];
    o += 4 * sh.rowv;
    for (let e = 0; e < code.slots; e++) w[o + e] = code.lay[code.mb + 1 + 2 * e] | (code.lay[code.mb + 2 + 2 * e] << 16);
    o += 4 * sh.pairv;
    w.set(pw, o);
    return w;
  });
  return { version, tiers, map, tb, tj, ts };
}

// ai: The kernels a BackHalf shares among its profiles (the profile is the uniforms'): bind group layouts and
// ai: pipelines, made once a back half, a tier code's LDPC once a code.
async function kernelsOf(bh) {
  if (bh._tierKernels) return bh._tierKernels;
  const device = bh.device, f16 = bh.precision === "f16";
  const bgl = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type === "u" ? "uniform" : type === "r" ? "read-only-storage" : "storage" } })) });
  const k = {
    alignBgl: bgl(["r", "r", "r", "w", "u", "u"]),
    softBgl: bgl(["r", "r", "r", "w", "w", "w", "r", "w", "u", "u"]),
    blocksBgl: bgl(["r", "r", "r", "w", "w", "u", "u"]),
    ldpcBgl: bgl(["r", "r", "r", "r", "w", "w", "w", "w", "u"]),
    ldpc: new Map(),
  };
  const layout = (g) => device.createPipelineLayout({ bindGroupLayouts: [g] });
  [k.align, k.soft, k.blocks] = await Promise.all([
    computePipeline(device, { code: talignSource({ f16 }), layout: layout(k.alignBgl), label: "talign" }),
    computePipeline(device, { code: tsoftSource({ f16 }), layout: layout(k.softBgl), label: "tsoft" }),
    computePipeline(device, { code: tblocksSource(), layout: layout(k.blocksBgl), label: "tblocks" }),
  ]);
  k.ldpcFor = async (sh) => {
    const key = JSON.stringify(sh);
    if (!k.ldpc.has(key)) k.ldpc.set(key, computePipeline(device, { code: tldpcSource(sh), layout: layout(k.ldpcBgl), label: `tldpc ${sh.n}` }));
    return k.ldpc.get(key);
  };
  bh._tierKernels = k;
  return k;
}

export class TierStage {
  // ai: bh: a built BackHalf (its tables, dims, soft and LDPC stages); text: the profile.
  static async build(bh, text) {
    const device = bh.device, tt = await tierTables(text);
    const slot = bh.sizes.findIndex((z) => z && z.n === tt.n);
    if (slot < 0) throw new Error(`profile ${tt.label}: its picture, ${tt.n}, is not built`);
    const top = bh.tables[slot];
    for (let i = 0; i < tt.pos.length; i++) if (tt.pos[i] !== top.pos[i]) throw new Error(`profile ${tt.label}: its coefficient order is not slot ${slot}'s`);
    if (tt.blocks.length > Math.min(bh.dims.blocksMax, TIER_BLOCKS)) throw new Error(`profile ${tt.label}: ${tt.blocks.length} blocks, the back half holds ${Math.min(bh.dims.blocksMax, TIER_BLOCKS)} a frame`);
    if (tt.subch > TJ_SUBS) throw new Error(`profile ${tt.label}: ${tt.subch} sub-channels`);
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    const c = tierContents(tt, { slot, dims: bh.dims, B: bh.B });
    for (const tr of c.tiers) if (tierWgBytes(tr.sh) > limit) throw new Error(`profile ${tt.label}: the ${RATE_NAMES[tr.rate]} code needs ${tierWgBytes(tr.sh)} B of workgroup memory, the device offers ${limit}`);
    const k = await kernelsOf(bh);
    const st = new TierStage();
    const uni = (words) => { const b = device.createBuffer({ size: words.byteLength, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, words); return b; };
    const mapBuf = device.createBuffer({ size: c.map.byteLength, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(mapBuf, 0, c.map);
    // ai: the one-rate LDPC's uniform with this slot's blocks 0: its workgroups there return at once
    const p0 = bh.ldpc.uniWords.slice();
    p0[4 * slot] = 0;
    Object.assign(st, {
      bh, key: tt.key, label: tt.label, slot, version: c.version, blocks: tt.blocks.length, n: tt.n, subch: tt.subch, kernels: k,
      tbBuf: uni(c.tb), tjBuf: uni(c.tj), tBufs: c.ts.map(uni), mapBuf, ldpcParams0: uni(p0),
      tiers: await Promise.all(c.tiers.map(async (tr) => ({ rate: tr.rate, count: tr.count, sh: tr.sh, pipeline: await k.ldpcFor(tr.sh) }))),
      bytes: { PS: 4 * bh.B * bh.dims.subchMax },
    });
    bh.log(`tiers ${tt.label}: LIZARD-${tt.subch} at slot ${slot} (n ${tt.n}), ${tt.blocks.length} blocks; codes ${st.tiers.map((t) => `${RATE_NAMES[t.rate]} n ${t.sh.n} z ${t.sh.z}, ${t.sh.threads} lanes, ${tierWgBytes(t.sh)} B`).join("; ")}`);
    return st;
  }

  // ai: The lane's bind groups (ln is a BackHalf lane): PS, then each kernel's group, and the one-rate LDPC's group
  // ai: over the zeroed uniform. Named as exportFor binds them.
  lane(ln) {
    const device = this.bh.device, so = this.bh.soft, k = this.kernels;
    ln.PS ??= device.createBuffer({ size: this.bytes.PS, usage: U.STORAGE | U.COPY_SRC });
    const group = (layout, bufs) => device.createBindGroup({ layout, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const params = so.params[this.slot];
    const g = {
      align: group(k.alignBgl, [ln.S, so.uvBuf, ln.LISTS, ln.PILOT, params, this.tbBuf]),
      soft: group(k.softBgl, [ln.S, so.uvBuf, ln.LISTS, ln.L, ln.EST, ln.BCOUNTS, ln.PILOT, ln.PS, params, this.tjBuf]),
      blocks: group(k.blocksBgl, [ln.LISTS, ln.EST, ln.PS, ln.BLK, ln.PILOT, params, this.tbBuf]),
      ldpc: this.tBufs.map((t) => group(k.ldpcBgl, [ln.L, ln.BLK, ln.LISTS, this.mapBuf, ln.V, ln.ITS, ln.REC, ln.BCOUNTS, t])),
      ldpc0: group(this.bh.ldpc.bgl, [ln.L, ln.BLK, ln.LISTS, this.bh.ldpc.mapBuf, ln.V, ln.ITS, ln.REC, ln.BCOUNTS, this.ldpcParams0]),
    };
    (ln.tierGroups ??= {})[this.key] = g;
    return g;
  }

  // ai: In the soft stage's pass, after the one-rate soft dispatches (which skip this slot): the shift, the soft values
  // ai: and pilots a sub-channel, the blocks.
  dispatchSoft(p, ln, frames) {
    const g = ln.tierGroups?.[this.key] ?? this.lane(ln), k = this.kernels;
    p.setPipeline(k.align); p.setBindGroup(0, g.align); p.dispatchWorkgroups(1, 1, frames);
    p.setPipeline(k.soft); p.setBindGroup(0, g.soft); p.dispatchWorkgroups(this.version, 1, frames);
    p.setPipeline(k.blocks); p.setBindGroup(0, g.blocks); p.dispatchWorkgroups(1, 1, frames);
  }

  // ai: In the LDPC's pass: the one-rate pipeline over its uniform with this slot zeroed (from the gate's ARGS as the
  // ai: composed chain's), then each tier's blocks.
  dispatchLdpc(p, ln, frames, fromGate) {
    const g = ln.tierGroups?.[this.key] ?? this.lane(ln), ld = this.bh.ldpc;
    p.setPipeline(ld.pipeline); p.setBindGroup(0, g.ldpc0);
    if (fromGate) p.dispatchWorkgroupsIndirect(ln.ARGS, 4 * argsLdpc); else p.dispatchWorkgroups(ld.blocksMax, frames, SLOTS);
    this.tiers.forEach((t, q) => { p.setPipeline(t.pipeline); p.setBindGroup(0, g.ldpc[q]); p.dispatchWorkgroups(t.count, frames, 1); });
  }

  // ai: The native host's form (core/dec/front.cpp): every object by its recorder id, a lane buffer as "lane:<name>"
  // ai: (S, LISTS, L, EST, BLK, BCOUNTS, PILOT, PS, V, ITS, REC), a dispatch's "frames" the batch's frame count. soft runs
  // ai: after the one-rate soft (which skips `slot`), ldpc in the LDPC's place: its first step is the one-rate kernel
  // ai: on the zeroed uniform, dispatched from ARGS at argsLdpc.
  exportFor(idOf) {
    const k = this.kernels, so = this.bh.soft, ld = this.bh.ldpc, params = idOf(so.params[this.slot]);
    const step = (pipeline, bgl, binds, dispatch) => ({ pipeline: idOf(pipeline), bgl: idOf(bgl), binds, dispatch });
    return {
      key: this.key, label: this.label, slot: this.slot, version: this.version, blocks: this.blocks, n: this.n, subch: this.subch, bytes: this.bytes,
      soft: [
        step(k.align, k.alignBgl, ["lane:S", idOf(so.uvBuf), "lane:LISTS", "lane:PILOT", params, idOf(this.tbBuf)], [1, 1, "frames"]),
        step(k.soft, k.softBgl, ["lane:S", idOf(so.uvBuf), "lane:LISTS", "lane:L", "lane:EST", "lane:BCOUNTS", "lane:PILOT", "lane:PS", params, idOf(this.tjBuf)], [this.version, 1, "frames"]),
        step(k.blocks, k.blocksBgl, ["lane:LISTS", "lane:EST", "lane:PS", "lane:BLK", "lane:PILOT", params, idOf(this.tbBuf)], [1, 1, "frames"]),
      ],
      ldpc: [
        { ...step(ld.pipeline, ld.bgl, ["lane:L", "lane:BLK", "lane:LISTS", idOf(ld.mapBuf), "lane:V", "lane:ITS", "lane:REC", "lane:BCOUNTS", idOf(this.ldpcParams0)], null), indirect: 4 * argsLdpc },
        ...this.tiers.map((t, q) => step(t.pipeline, k.ldpcBgl, ["lane:L", "lane:BLK", "lane:LISTS", idOf(this.mapBuf), "lane:V", "lane:ITS", "lane:REC", "lane:BCOUNTS", idOf(this.tBufs[q])], [t.count, "frames", 1])),
      ],
    };
  }
}

// ai: The blocks a frame carries at slot `size` and version `version` under the active profile (st), else the
// ai: version's (one block a chunk): the count its pilots and blocks are read over.
export const blocksAt = (st, size, version) => (st && size === st.slot && version === st.version ? st.blocks : version);
