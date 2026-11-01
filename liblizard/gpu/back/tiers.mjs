// ai: The format's rate profile on the GPU (2026-10-07; src/focus.h focus_tiers_for; the kernels in
// ai: ../glsl/back_tiers.mjs, TAB's layout and the codes' shapes in ../plan/back_tiers.mjs): one stage a back half, its
// ai: soft values and LDPC. Every version's profile is a row of TAB, so any version at any picture slot decodes. A
// ai: lane's groups and the dispatches are the native host's (core/dec/front.cpp), from exportFor.
//
//   const st = await RuleStage.build(bh);   // BackHalf.build does it (back.mjs)
//   st.blocks[v]       // the blocks a frame of version v carries (its pilots and blocks are read over them)
//   st.exportFor(idOf) // what the native host replays (../../gen/gen.mjs, core/dec/front.cpp)
import { N_FOR, subchOf } from "../../sim/lizard_pick.mjs";
import { whiten, perm, inverse } from "./bitmap.mjs";
import { ruleTables } from "./codes.mjs";
import { crcPowers, PAYLOAD, stopFor } from "./ref_ldpc.mjs";
import { netWords } from "./stop/rule.mjs";
import { VERSION_MIN } from "../wordcode.mjs";
import { tierShape, tierWgBytes, tierUniformWords, tabLayout, VERSIONS, CODES, TTW, WHITE_SLOTS, LDPC_FORMS } from "../plan/back_tiers.mjs";
import { talignSource, tsoftSource, tblocksSource, tldpcSource } from "../glsl/back_tiers.mjs";
import { computePipeline } from "../pipeline.mjs";
import { argsOffset, ldpcArgsOffset } from "./transform.mjs";

const U = globalThis.GPUBufferUsage ?? { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };

// ai: TAB's words (../plan/back_tiers.mjs tabLayout) and the codes' constants the kernels take. nets: each code's learned
// ai: stop words in CODES' order, one after another in NET (L.netAt); none (scripts/gpu/back/check_tiers.mjs) leaves NET
// ai: empty.
export function ruleContents(rt, nets = []) {
  nets ??= [];
  const sizes = rt.codes.map((_, c) => nets[c]?.length ?? 0);
  const L = tabLayout(rt.codes, sizes.reduce((a, b) => a + b, 0)), tab = new Uint32Array(L.words);
  L.netAt = sizes.map((_, c) => L.NET + sizes.slice(0, c).reduce((a, b) => a + b, 0));
  // ai: TT keeps its rows 0 to VERSIONS; row 0 stays zero (blocks[0] 0), which the kernels read as a frame on no list
  const blocks = new Uint32Array(VERSIONS + 1);
  for (let v = VERSION_MIN; v <= VERSIONS; v++) {
    const o = L.TT + TTW * v;
    for (const t of rt.profiles[v]) { tab.set([t.firstBlock, t.slot0, t.count, 0], o + 4 + 4 * t.c); blocks[v] += t.count; }
    tab[o] = blocks[v];
    tab[o + 1] = rt.profiles[v].used;
  }
  rt.codes.forEach((code, c) => {
    // ai: codeword bit j's slot: the bit map over the nt sent bits, offset by the np never sent, which read 0xffff (no slot)
    const inv = inverse(perm(code.nt));
    for (let j = 0; j < code.n; j++) tab[L.inv[c] + (j >> 1)] |= (j < code.np ? 0xffff : inv[j - code.np]) << (16 * (j & 1));
  });
  const w = whiten(WHITE_SLOTS);
  for (let i = 0; i < WHITE_SLOTS; i++) if (w[i]) tab[L.WHITE + (i >> 5)] |= 1 << (i & 31);
  nets.forEach((w, c) => { if (w) tab.set(w, L.netAt[c]); });
  // ai: n here is the sent length (the codeword's slots; the tail, the pilots, follows it), slots a block's (CODES'),
  // ai: tail its pilots' coefficients ((slots - nt) / 2: 8 at 7/8, 16 at 3/4, none at 2/3 and 1/2), the bar at the sent
  // ai: rate k / nt
  const consts = rt.codes.map((code, c) => ({ n: code.nt, k: code.k, slots: CODES[c].slots, tail: (CODES[c].slots - code.nt) / 2, bar: Math.fround(Math.fround(0.2 * code.k) / code.nt) }));
  return { L, tab, blocks, consts };
}

// ai: A code's uniform T at batch size B (../glsl/back_tiers.mjs's TierCode block).
function tierUniform(code, sh, dims, B) {
  const w = new Uint32Array(tierUniformWords(sh));
  w.set([B, dims.lStride, dims.blocksMax, dims.recCap], 0);
  let o = 4;
  for (let r = 0; r <= code.mb; r++) w[o + r] = code.lay[r];
  o += 4 * sh.rowv;
  for (let e = 0; e < code.slots; e++) w[o + e] = code.lay[code.mb + 1 + 2 * e] | (code.lay[code.mb + 2 + 2 * e] << 16);
  o += 4 * sh.pairv;
  w.set(crcPowers(PAYLOAD), o);
  return w;
}

export class RuleStage {
  // ai: bh: a BackHalf with its transform, soft and LDPC stages built (their lane buffers, the soft stage's per-slot
  // ai: uniforms and UV are this stage's too).
  static async build(bh) {
    const device = bh.device, f16 = bh.precision === "f16";
    const rt = await ruleTables();
    // ai: every code gives up by its own learned stop (ref_ldpc.mjs stopFor)
    const nets = await Promise.all(rt.codes.map((code) => stopFor(code.m)));
    const { L, tab, blocks, consts } = ruleContents(rt, nets.map(netWords));
    // ai: the versions this back half serves: those whose picture it built
    const served = []; for (let v = VERSION_MIN; v <= VERSIONS; v++) if (bh.sizes.some((z) => z && z.n === N_FOR(subchOf(v)))) served.push(v);
    const maxBlocks = Math.max(...served.map((v) => blocks[v]));
    if (maxBlocks > bh.dims.blocksMax) throw new Error(`the rate profile's ${maxBlocks} blocks a frame pass the back half's ${bh.dims.blocksMax}`);
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    // ai: each code in the first of LDPC_FORMS, and in every other there is (the same uniform and groups: only the
    // ai: pipelines differ; the native host keeps the fastest on its device)
    const shapes = rt.codes.map((code) => tierShape(code, LDPC_FORMS[0]));
    const alts = LDPC_FORMS.slice(1).map((form) => rt.codes.map((code) => tierShape(code, form)));
    for (const set of [shapes, ...alts]) set.forEach((sh, c) => { const need = tierWgBytes(sh, nets[c]); if (need > limit) throw new Error(`the ${CODES[c].name} code (${sh.form}) needs ${need} B of workgroup memory, the device offers ${limit}`); });
    const bgl = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type === "u" ? "uniform" : type === "r" ? "read-only-storage" : "storage" } })) });
    const st = new RuleStage();
    Object.assign(st, { bh, blocks, layout: L, consts, codes: rt.codes, shapes, alts });
    st.alignBgl = bgl(["r", "r", "r", "w", "r", "u"]);
    st.softBgl = bgl(["r", "r", "r", "w", "w", "w", "r", "r", "u"]);
    st.blocksBgl = bgl(["r", "r", "w", "w", "r", "u"]);
    st.ldpcBgl = bgl(["r", "r", "r", "r", "w", "w", "w", "w", "u"]);
    const pl = (g) => device.createPipelineLayout({ bindGroupLayouts: [g] });
    st.tabBuf = device.createBuffer({ size: tab.byteLength, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(st.tabBuf, 0, tab);
    st.tBufs = rt.codes.map((code, c) => { const w = tierUniform(code, shapes[c], bh.dims, bh.B), b = device.createBuffer({ size: w.byteLength, usage: U.UNIFORM | U.COPY_DST }); device.queue.writeBuffer(b, 0, w); return b; });
    // ai: each code's most blocks in any version: its dispatch's x
    st.counts = rt.codes.map((_, c) => Math.max(0, ...served.map((v) => rt.profiles[v].find((t) => t.c === c)?.count ?? 0)));
    const args = { layout: L, codes: consts };
    // ai: every kernel in GLSL (../glsl/back_tiers.mjs)
    const ldpcCode = (sh, c) => ({ glsl: tldpcSource(sh, { c, inv: L.inv[c], ...args, net: { hidden: nets[c].hidden, at: L.netAt[c] } }) });
    [st.align, st.soft, st.blocksPipe, ...st.ldpc] = await Promise.all([
      computePipeline(device, { code: { glsl: talignSource({ f16, ...args }) }, layout: pl(st.alignBgl), label: "talign" }),
      computePipeline(device, { code: { glsl: tsoftSource({ f16, ...args }) }, layout: pl(st.softBgl), label: "tsoft" }),
      computePipeline(device, { code: { glsl: tblocksSource(args) }, layout: pl(st.blocksBgl), label: "tblocks" }),
      ...shapes.map((sh, c) => computePipeline(device, { code: ldpcCode(sh, c), layout: pl(st.ldpcBgl), label: `tldpc ${CODES[c].name}` })),
    ]);
    st.ldpcAlts = await Promise.all(alts.map((set) => Promise.all(set.map((sh, c) => computePipeline(device, { code: ldpcCode(sh, c), layout: pl(st.ldpcBgl), label: `tldpc ${CODES[c].name} ${sh.form}` })))));
    st.bytes = { EP: 8 * bh.B * bh.dims.subchMax };
    bh.log(`rate profile (LDPC ${LDPC_FORMS.join(", ")}): ${shapes.map((sh, c) => `${CODES[c].name} n ${sh.n} z ${sh.z}, ${sh.threads} lanes, ${tierWgBytes(sh, nets[c])} B, the learned stop ${nets[c].hidden} x ${nets[c].hidden}`).join("; ")}; TAB ${(tab.byteLength / 1024).toFixed(0)} KB; LIZARD-432 ${blocks[27]} blocks`);
    return st;
  }

  // ai: The native host's form (core/dec/front.cpp): every object by its recorder id, a lane buffer as "lane:<name>"
  // ai: (S, LISTS, L, EP, BLK, BCOUNTS, PILOT, V, ITS, REC), every dispatch indirect from the lane's ARGS at "argsAt"
  // ai: (its byte offset: dispatchSoft's, dispatchLdpc's); soft in the soft stage's place, ldpc in the LDPC's;
  // ai: blocks[v] a version's blocks.
  exportFor(idOf) {
    const so = this.bh.soft, uv = idOf(so.uvBuf), tab = idOf(this.tabBuf);
    const step = (pipeline, bgl, binds, argsAt) => ({ pipeline: idOf(pipeline), bgl: idOf(bgl), binds, argsAt });
    const soft = [];
    for (const s of so.served) soft.push(step(this.align, this.alignBgl, ["lane:S", uv, "lane:LISTS", "lane:PILOT", tab, idOf(so.params[s])], argsOffset(s, "reduce")));
    for (const s of so.served) soft.push(step(this.soft, this.softBgl, ["lane:S", uv, "lane:LISTS", "lane:L", "lane:EP", "lane:BCOUNTS", "lane:PILOT", tab, idOf(so.params[s])], argsOffset(s, "soft")));
    for (const s of so.served) soft.push(step(this.blocksPipe, this.blocksBgl, ["lane:LISTS", "lane:EP", "lane:BLK", "lane:PILOT", tab, idOf(so.params[s])], argsOffset(s, "reduce")));
    const ldpcOf = (pipes) => pipes.map((pipe, c) => step(pipe, this.ldpcBgl, ["lane:L", "lane:BLK", "lane:LISTS", tab, "lane:V", "lane:ITS", "lane:REC", "lane:BCOUNTS", idOf(this.tBufs[c])], ldpcArgsOffset(c))).filter((_, c) => this.counts[c] > 0);
    // ai: ldpcAlts: the same steps in each further form (the same groups), which core/dec/front.cpp times against ldpc
    return { bytes: this.bytes, blocks: Array.from(this.blocks), soft, ldpc: ldpcOf(this.ldpc), ldpcForms: LDPC_FORMS, ...(this.ldpcAlts.length ? { ldpcAlts: this.ldpcAlts.map(ldpcOf) } : {}) };
  }
}
