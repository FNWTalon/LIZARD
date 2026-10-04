// ai: The host of the cancel stage's bases and fit (DESIGN.md section 13.5): BLUR and MEANS (wgsl/cancel_blur.mjs)
// ai: over the painted slots of PIC into REF, FIT and SOLVE (wgsl/cancel_fit.mjs) over the short frames of LISTS2
// ai: into FITP and CANCEL. Shaped as soft.mjs and ldpc.mjs are, so cancel.mjs plugs it into its stages:
// ai:
// ai:   const ft = await build(device, { B, tables, dims, grid, picUnis, log });
// ai:   ft.lane(ln);                        // after cancel.lane (PIC, REF, BPART, PLAN, LISTS2, FITP, CANCEL, BCOUNTS2)
// ai:   ft.bindGrid(ln, gridBuf);           // the grid fit (done by lane() when ln.gridBuf is there)
// ai:   ft.bindPicture(ln, { texView, mapsBuf, residBuf });   // the fused fit, built with picUnis
// ai:   ft.dispatch(p, ln, name, picture);  // name in STAGES, picture "grid" | "fused" for the fit
// ai:
// ai: picUnis: the transform's PIC uniform a size (Transform.picUnis: g0, step, the lattice), null for no fused fit.
// ai: The fit and solve dispatch direct, (fitWorkgroups(n), 1, B) and (B) a size, returning past LISTS2's count.
import { blurSource, meansSource, meansWorkgroups, PARAMS_WORDS } from "../wgsl/cancel_blur.mjs";
import { fitSource, solveSource, samplesARow, sampleStride, fitWorkgroups } from "../wgsl/cancel_fit.mjs";
import { planLayout } from "../wgsl/cancel_gate.mjs";

const U = globalThis.GPUBufferUsage;
export const STAGES = ["blur", "bmeans", "fit", "solve"];

export async function build(device, { B, tables, dims, grid = "f32", picUnis = null, log = () => {} }) {
  const { served, nmax, refSlots, refStride, refMeans, fitpStride, blocksMax } = dims;
  const L = planLayout({ blocksMax, B, refSlots });
  const layout = (types) => device.createBindGroupLayout({ entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })) });
  const RO = "read-only-storage", RW = "storage", UN = "uniform";
  const basesBgl = layout([RO, RO, RW, RW, UN]);
  const fitBgl = layout([RO, RO, RO, RO, RW, UN]);
  const solveBgl = layout([RO, RO, RO, RW, RW, UN]);
  // ai: The fused fit binds the picture as fused pass 1 does: the texture at 0, maps 7, resid 8, PIC 9.
  const fitfBgl = picUnis ? device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d-array" } },
    ...[[1, RO], [2, RO], [3, RO], [4, RW], [5, UN], [7, RO], [8, RO], [9, UN]].map(([binding, type]) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } }))] }) : null;
  const pipe = async (bgl, code, label) => {
    const module = device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error(errs.map((m) => `${label} ${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + code.split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
    return device.createComputePipelineAsync({ label, layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: "main" } });
  };
  const blur = [], means = [], fit = [], fitf = [];
  const jobs = [pipe(solveBgl, solveSource({ B }), "cancel solve")];
  for (const s of served) {
    const { n } = tables[s];
    jobs.push(pipe(basesBgl, blurSource({ n }), `cancel blur ${n}`).then((p) => { blur[s] = p; }));
    jobs.push(pipe(basesBgl, meansSource({ n }), `cancel means ${n}`).then((p) => { means[s] = p; }));
    jobs.push(pipe(fitBgl, fitSource({ n, B, picture: "grid", grid }), `cancel fit ${n}`).then((p) => { fit[s] = p; }));
    if (picUnis) jobs.push(pipe(fitfBgl, fitSource({ n, B, picture: "fused" }), `cancel fit fused ${n}`).then((p) => { fitf[s] = p; }));
  }
  const [solve] = await Promise.all(jobs);
  log(`cancel fit: sizes ${served.map((s) => `${tables[s].n} (${samplesARow(tables[s].n)} samples a row)`).join(", ")}, ${refSlots} slots, FITP ${(fitpStride / 1024).toFixed(0)} KB a frame${picUnis ? ", fused fit built" : ""}`);
  const group = (bgl, entries) => device.createBindGroup({ layout: bgl, entries: entries.map((e, binding) => ({ binding, resource: e.buffer ? e : { buffer: e } })) });
  return {
    B, served, dims, grid, fusable: !!picUnis, plan: L, pipelines: { blur, means, fit, fitf, solve },
    // ai: A lane's params a size (the strides in each buffer's own units) and the bases and solve groups; the grid
    // ai: fit's when the lane already holds its grid.
    lane(ln) {
      ln.fitParams = []; ln.basesGroups = []; ln.solveGroups = []; ln.fitGroups = null; ln.fitfGroups = null;
      for (const s of served) {
        const { n } = tables[s], ns = samplesARow(n);
        const u = new Uint32Array(PARAMS_WORDS);
        u.set([s, n, ns, sampleStride(n), nmax * nmax, refStride / 8, refMeans / 8, 3 * nmax * (nmax / 16), fitpStride / 4, L.fr, L.slots, ln.gridStride ?? 0, fitWorkgroups(n)]);
        const buf = device.createBuffer({ size: 4 * PARAMS_WORDS, usage: U.UNIFORM | U.COPY_DST });
        device.queue.writeBuffer(buf, 0, u);
        ln.fitParams[s] = buf;
        ln.basesGroups[s] = group(basesBgl, [ln.PIC, ln.PLAN, ln.REF, ln.BPART, buf]);
        ln.solveGroups[s] = group(solveBgl, [ln.LISTS2, ln.FITP, ln.PLAN, ln.CANCEL, ln.BCOUNTS2, buf]);
      }
      if (ln.gridBuf) this.bindGrid(ln, ln.gridBuf);
      return ln;
    },
    bindGrid(ln, gridBuf) {
      ln.fitGroups = [];
      for (const s of served) ln.fitGroups[s] = group(fitBgl, [gridBuf, ln.LISTS2, ln.REF, ln.PLAN, ln.FITP, ln.fitParams[s]]);
    },
    bindPicture(ln, { texView, mapsBuf, residBuf }) {
      if (!picUnis) throw new Error("the fit was built without picture tables: no fused fit");
      ln.fitfGroups = [];
      for (const s of served) {
        const bufs = [[1, ln.LISTS2], [2, ln.REF], [3, ln.PLAN], [4, ln.FITP], [5, ln.fitParams[s]], [7, mapsBuf], [8, residBuf], [9, picUnis[s]]];
        ln.fitfGroups[s] = device.createBindGroup({ layout: fitfBgl, entries: [{ binding: 0, resource: texView }, ...bufs.map(([binding, buffer]) => ({ binding, resource: { buffer } }))] });
      }
    },
    // ai: One stage over every served size, appended to an open compute pass.
    dispatch(p, ln, name, picture = "grid") {
      for (const s of served) {
        const { n } = tables[s];
        switch (name) {
          case "blur": p.setPipeline(blur[s]); p.setBindGroup(0, ln.basesGroups[s]); p.dispatchWorkgroups(n / 16, n / 16, refSlots); break;
          case "bmeans": p.setPipeline(means[s]); p.setBindGroup(0, ln.basesGroups[s]); p.dispatchWorkgroups(meansWorkgroups(n), 1, refSlots); break;
          case "fit": {
            const fused = picture === "fused", groups = fused ? ln.fitfGroups : ln.fitGroups;
            if (!groups) throw new Error(fused ? "fused fit before bindPicture" : "grid fit with no grid bound");
            p.setPipeline(fused ? fitf[s] : fit[s]); p.setBindGroup(0, groups[s]); p.dispatchWorkgroups(fitWorkgroups(n), 1, B);
            break;
          }
          case "solve": p.setPipeline(solve); p.setBindGroup(0, ln.solveGroups[s]); p.dispatchWorkgroups(B, 1, 1); break;
          default: throw new Error(`fit stage ${name}`);
        }
      }
    },
  };
}
