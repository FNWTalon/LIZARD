// ai: The format's rate profile on the GPU (2026-10-07; src/focus.h focus_tiers_for; the kernels and TAB's layout in
// ai: ../wgsl/back_tiers.mjs): one stage a back half, run in place of the one-rate soft and LDPC dispatches. Every
// ai: version's profile is a row of TAB, so any version at any picture slot decodes.
//
//   const st = await RuleStage.build(bh);   // BackHalf.build does it (back.mjs)
//   st.lane(ln); st.dispatchSoft(p, ln, frames); st.dispatchLdpc(p, ln, frames)
//   st.blocks[v]       // the blocks a frame of version v carries (its pilots and blocks are read over them)
//   st.exportFor(idOf) // what the native host replays (../../gen/gen.mjs, core/dec/front.cpp)
import { N_FOR } from "../../sim/lizard_pick.mjs";
import { whiten, perm, inverse } from "./bitmap.mjs";
import { ruleTables } from "./codes.mjs";
import { crcPowers, PAYLOAD, stopRule, DEFAULT_STOP, FIRST_READ, C_RULE } from "./ref_ldpc.mjs";
import { netWords, hasRule } from "./stop/rule.mjs";
import { SLOTS } from "../wgsl/common.mjs";
import { talignSource, tsoftSource, tblocksSource, tldpcSource, tierShape, tierWgBytes, tierUniformWords, tabLayout, VERSIONS, CODES, TTW, WHITE_SLOTS, LDPC_FORMS } from "../wgsl/back_tiers.mjs";
import { computePipeline } from "../pipeline.mjs";

const U = globalThis.GPUBufferUsage ?? { COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };

// ai: TAB's words (../wgsl/back_tiers.mjs tabLayout) and the codes' constants the kernels take. nets: a code's learned
// ai: stop words each in CODES' order (null: that code gives up by the stall rule), one after another in NET (L.netAt).
export function ruleContents(rt, nets = []) {
  nets ??= [];
  const sizes = rt.codes.map((_, c) => nets[c]?.length ?? 0);
  const L = tabLayout(rt.codes, sizes.reduce((a, b) => a + b, 0)), tab = new Uint32Array(L.words);
  L.netAt = sizes.map((_, c) => L.NET + sizes.slice(0, c).reduce((a, b) => a + b, 0));
  const blocks = new Uint32Array(VERSIONS + 1);
  for (let v = 1; v <= VERSIONS; v++) {
    const o = L.TT + TTW * v;
    for (const t of rt.profiles[v]) { tab.set([t.firstBlock, t.sub0, t.count, 0], o + 4 + 4 * t.c); blocks[v] += t.count; }
    tab[o] = blocks[v];
  }
  rt.codes.forEach((code, c) => {
    // ai: codeword bit j's slot: the bit map over the nt sent bits, offset by the np never sent, which read 0xffff (no slot)
    const inv = inverse(perm(code.nt, code.k - code.np, rt.bitmap));
    for (let j = 0; j < code.n; j++) tab[L.inv[c] + (j >> 1)] |= (j < code.np ? 0xffff : inv[j - code.np]) << (16 * (j & 1));
  });
  if (rt.bitmap) { const w = whiten(WHITE_SLOTS); for (let i = 0; i < WHITE_SLOTS; i++) if (w[i]) tab[L.WHITE + (i >> 5)] |= 1 << (i & 31); }
  nets.forEach((w, c) => { if (w) tab.set(w, L.netAt[c]); });
  // ai: n here is the sent length (the codeword's slots; the tail and the pilots follow it), the bar at the sent rate k / nt
  const consts = rt.codes.map((code, c) => ({ n: code.nt, k: code.k, subs: CODES[c].subs, tail: (640 * CODES[c].subs - code.nt) / 2, bar: Math.fround(Math.fround(0.2 * code.k) / code.nt) }));
  return { L, tab, blocks, consts };
}

// ai: A code's uniform T at batch size B (../wgsl/back_tiers.mjs tierUniformStruct).
function tierUniform(code, sh, dims, B) {
  const w = new Uint32Array(tierUniformWords(sh));
  w.set([B, dims.lStride, dims.blocksMax, dims.recCap], 0);
  let o = 4;
  for (let r = 0; r <= code.mb; r++) w[o + r] = code.lay[r];
  o += 4 * sh.rowv;
  for (let e = 0; e < code.slots; e++) w[o + e] = code.lay[code.mb + 1 + 2 * e] | (code.lay[code.mb + 2 + 2 * e] << 16);
  o += 4 * sh.pairv;
  w.set(crcPowers(PAYLOAD), o);
  o += 120;
  // ai: the "toggle" form (../wgsl/back_tiers.mjs LDPC_FORMS): each data column's checks, colPtr[kb + 1] then (row | shift << 16)
  if (sh.toggle) {
    const byCol = Array.from({ length: code.kb }, () => []);
    for (let r = 0; r < code.mb; r++) for (let e = code.lay[r]; e < code.lay[r + 1]; e++) byCol[code.lay[code.mb + 1 + 2 * e]].push(r | (code.lay[code.mb + 2 + 2 * e] << 16));
    let q = 0;
    for (let c = 0; c <= code.kb; c++) { w[o + c] = q; if (c < code.kb) q += byCol[c].length; }
    byCol.flat().forEach((v, j) => { w[o + code.kb + 1 + j] = v; });
  }
  return w;
}

export class RuleStage {
  // ai: bh: a BackHalf with its transform, soft and LDPC stages built (their lane buffers, the soft stage's per-slot
  // ai: uniforms and UV are this stage's too).
  static async build(bh) {
    const device = bh.device, f16 = bh.precision === "f16";
    const rt = await ruleTables();
    // ai: every code gives up by its own learned stop (since 2026-10-07; the 3/4 code's alone before, the 7/8 and 1/2
    // ai: codes by the C's stall rule), or all by the stall rule under a page's ?ldpcstop=c (ldpc.mjs's switch, the
    // ai: arithmetic check and the A/B's other arm)
    const forced = globalThis.location ? new URLSearchParams(globalThis.location.search).get("ldpcstop") : null, stop = forced ?? DEFAULT_STOP;
    if (!FIRST_READ.includes(stop)) throw new Error(`ldpc stop ${stop}: one of ${FIRST_READ.join(", ")}`);
    // ai: a code with no learned stop of its own gives up by the stall rule (every code of the format has one)
    const rules = await Promise.all(rt.codes.map((code) => (stop === "learned" && !hasRule(code.m) ? C_RULE : stopRule(stop, code.m))));
    const { L, tab, blocks, consts } = ruleContents(rt, rules.map((r) => (r.net ? netWords(r.net) : null)));
    // ai: the versions this back half serves: those whose picture it built (a 32 KB device in f32 builds no 1536)
    const served = []; for (let v = 1; v <= VERSIONS; v++) if (bh.sizes.some((z) => z && z.n === N_FOR(8 * v))) served.push(v);
    const maxBlocks = Math.max(...served.map((v) => blocks[v]));
    if (maxBlocks > bh.dims.blocksMax) throw new Error(`the rate profile's ${maxBlocks} blocks a frame pass the back half's ${bh.dims.blocksMax}`);
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    // ai: each code in the first of LDPC_FORMS, and in the second where there is one (the same uniform and groups: only
    // ai: the pipelines differ; the native host keeps the faster on its device)
    const shapes = rt.codes.map((code) => tierShape(code, LDPC_FORMS[0]));
    const alts = LDPC_FORMS[1] ? rt.codes.map((code) => tierShape(code, LDPC_FORMS[1])) : null;
    for (const set of [shapes, alts]) if (set) set.forEach((sh, c) => { const need = tierWgBytes(sh, rules[c].net); if (need > limit) throw new Error(`the ${CODES[c].name} code (${sh.form}) needs ${need} B of workgroup memory, the device offers ${limit}`); });
    if (alts && alts.some((sh, c) => sh.toggle !== shapes[c].toggle)) throw new Error("LDPC forms with different uniforms");
    const bgl = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: type === "u" ? "uniform" : type === "r" ? "read-only-storage" : "storage" } })) });
    const st = new RuleStage();
    Object.assign(st, { bh, blocks, layout: L, consts, codes: rt.codes, shapes, alts, stop });
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
    [st.align, st.soft, st.blocksPipe, ...st.ldpc] = await Promise.all([
      computePipeline(device, { code: talignSource({ f16, ...args }), layout: pl(st.alignBgl), label: "talign" }),
      computePipeline(device, { code: tsoftSource({ f16, ...args }), layout: pl(st.softBgl), label: "tsoft" }),
      computePipeline(device, { code: tblocksSource(args), layout: pl(st.blocksBgl), label: "tblocks" }),
      ...shapes.map((sh, c) => computePipeline(device, { code: tldpcSource(sh, { c, inv: L.inv[c], ...args, net: rules[c].net ? { hidden: rules[c].net.hidden, at: L.netAt[c] } : null }), layout: pl(st.ldpcBgl), label: `tldpc ${CODES[c].name}` })),
    ]);
    st.ldpcAlt = alts ? await Promise.all(alts.map((sh, c) => computePipeline(device, { code: tldpcSource(sh, { c, inv: L.inv[c], ...args, net: rules[c].net ? { hidden: rules[c].net.hidden, at: L.netAt[c] } : null }), layout: pl(st.ldpcBgl), label: `tldpc ${CODES[c].name} ${sh.form}` }))) : null;
    st.bytes = { EP: 8 * bh.B * bh.dims.subchMax };
    bh.log(`rate profile (LDPC ${LDPC_FORMS.join(" and ")}): ${shapes.map((sh, c) => `${CODES[c].name} n ${sh.n} z ${sh.z}, ${sh.threads} lanes, ${tierWgBytes(sh, rules[c].net)} B${rules[c].net ? `, the learned stop ${rules[c].net.hidden} x ${rules[c].net.hidden}` : ", the stall rule"}`).join("; ")}; TAB ${(tab.byteLength / 1024).toFixed(0)} KB; LIZARD-432 ${blocks[54]} blocks`);
    return st;
  }

  // ai: The lane's groups: EP, then a slot's align, soft and blocks groups and a code's LDPC group.
  lane(ln) {
    const device = this.bh.device, so = this.bh.soft;
    ln.EP = device.createBuffer({ size: this.bytes.EP, usage: U.STORAGE | U.COPY_SRC });
    ln.bytes = (ln.bytes ?? 0) + this.bytes.EP;
    const group = (layout, bufs) => device.createBindGroup({ layout, entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    ln.ruleGroups = {
      align: Object.fromEntries(so.served.map((s) => [s, group(this.alignBgl, [ln.S, so.uvBuf, ln.LISTS, ln.PILOT, this.tabBuf, so.params[s]])])),
      soft: Object.fromEntries(so.served.map((s) => [s, group(this.softBgl, [ln.S, so.uvBuf, ln.LISTS, ln.L, ln.EP, ln.BCOUNTS, ln.PILOT, this.tabBuf, so.params[s]])])),
      blocks: Object.fromEntries(so.served.map((s) => [s, group(this.blocksBgl, [ln.LISTS, ln.EP, ln.BLK, ln.PILOT, this.tabBuf, so.params[s]])])),
      ldpc: this.tBufs.map((t) => group(this.ldpcBgl, [ln.L, ln.BLK, ln.LISTS, this.tabBuf, ln.V, ln.ITS, ln.REC, ln.BCOUNTS, t])),
    };
    return ln;
  }

  // ai: The soft stage's pass: the shift by the pilots, the soft values and pilots a sub-channel, the blocks, a slot each.
  dispatchSoft(p, ln, frames) {
    const g = ln.ruleGroups, so = this.bh.soft;
    p.setPipeline(this.align); for (const s of so.served) { p.setBindGroup(0, g.align[s]); p.dispatchWorkgroups(1, 1, frames); }
    p.setPipeline(this.soft); for (const s of so.served) { p.setBindGroup(0, g.soft[s]); p.dispatchWorkgroups(so.sizes[s].blocks, 1, frames); }
    p.setPipeline(this.blocksPipe); for (const s of so.served) { p.setBindGroup(0, g.blocks[s]); p.dispatchWorkgroups(1, 1, frames); }
  }

  // ai: The LDPC's pass: a dispatch a code over every slot.
  dispatchLdpc(p, ln, frames) {
    this.ldpc.forEach((pipe, c) => { if (!this.counts[c]) return; p.setPipeline(pipe); p.setBindGroup(0, ln.ruleGroups.ldpc[c]); p.dispatchWorkgroups(this.counts[c], frames, SLOTS); });
  }

  // ai: The native host's form (core/dec/front.cpp): every object by its recorder id, a lane buffer as "lane:<name>"
  // ai: (S, LISTS, L, EP, BLK, BCOUNTS, PILOT, V, ITS, REC), a dispatch's "frames" the batch's frame count; soft in the
  // ai: soft stage's place, ldpc in the LDPC's; blocks[v] a version's blocks.
  exportFor(idOf) {
    const so = this.bh.soft, uv = idOf(so.uvBuf), tab = idOf(this.tabBuf);
    const step = (pipeline, bgl, binds, dispatch) => ({ pipeline: idOf(pipeline), bgl: idOf(bgl), binds, dispatch });
    const soft = [];
    for (const s of so.served) soft.push(step(this.align, this.alignBgl, ["lane:S", uv, "lane:LISTS", "lane:PILOT", tab, idOf(so.params[s])], [1, 1, "frames"]));
    for (const s of so.served) soft.push(step(this.soft, this.softBgl, ["lane:S", uv, "lane:LISTS", "lane:L", "lane:EP", "lane:BCOUNTS", "lane:PILOT", tab, idOf(so.params[s])], [so.sizes[s].blocks, 1, "frames"]));
    for (const s of so.served) soft.push(step(this.blocksPipe, this.blocksBgl, ["lane:LISTS", "lane:EP", "lane:BLK", "lane:PILOT", tab, idOf(so.params[s])], [1, 1, "frames"]));
    const ldpcOf = (pipes) => pipes.map((pipe, c) => step(pipe, this.ldpcBgl, ["lane:L", "lane:BLK", "lane:LISTS", tab, "lane:V", "lane:ITS", "lane:REC", "lane:BCOUNTS", idOf(this.tBufs[c])], [this.counts[c], "frames", SLOTS])).filter((_, c) => this.counts[c] > 0);
    // ai: ldpcAlt: the same steps in the second form (the same groups), which core/dec/front.cpp times against ldpc
    return { bytes: this.bytes, blocks: Array.from(this.blocks), soft, ldpc: ldpcOf(this.ldpc), ldpcForms: LDPC_FORMS, ...(this.ldpcAlt ? { ldpcAlt: ldpcOf(this.ldpcAlt) } : {}) };
  }
}
