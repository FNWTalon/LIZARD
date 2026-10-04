// ai: The nets' timing probes on a device with no browser harness (the phone): where the proposer's and the small
// ai: classifier's time goes there. The web generators' own timing builds, compiled as gen.mjs compiles the web's
// ai: modules and listed in the manifest in the web kernel's place; run with LIZ_NATIVE_NETS=1, which swaps them in
// ai: unchecked, LIZ_TWINS=int8 and PROFILE=1 for the stage's ms. Timing only: a probe's output is wrong and nothing
// ai: decodes.
// ai:   LIZ_PROBE=<stage>:<probe>[,<stage>:<probe>] node gen/nets.mjs <tree>
// ai:   x (the bank's X pass, role front.fcn2X) and net (its network, front.fcn2): liblizard/gpu/wgsl/bank_fcn2.mjs
// ai:     fcn2Source's `probe`: nostats (no box sums), noconv (no convolutions), noload (no loads of the luma and X
// ai:     windows), and none, an empty kernel (the stage without that pass)
// ai:   small (the cascade's first net, int8: front.twins.small.pipe): liblizard/gpu/wgsl/classify_gemm.mjs
// ai:     classifySource's `stop`: cut, conv1 .. conv4, fc1 (the kernel returns after that layer), and none
// ai: The kernels are built at the tree's own forms (setup/<variant>.json); before any probe is written, each stage's
// ai: source built here with no probe must be the tree's own module byte for byte, or the build stops.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { nagaWgsl, stripCaps, patchBitcasts, TOOLS } from "./gen.mjs";

const LIB = new URL("../", import.meta.url);
const { fcn2Source } = await import(new URL("gpu/wgsl/bank_fcn2.mjs", LIB).href);
const { WEIGHTS_INT8 } = await import(new URL("gpu/bank_fcn2.mjs", LIB).href);
const { loadNet } = await import(new URL("gpu/cnn/netfile.mjs", LIB).href);
const { classifySource, classifyPlan, packWeightsInt8 } = await import(new URL("gpu/wgsl/classify_gemm.mjs", LIB).href);
const { CASCADE, INT8_TWINS } = await import(new URL("gpu/wgsl/classify.mjs", LIB).href);

const BANK = { x: "fcn2X", net: "fcn2" }, PROBES = ["nostats", "noconv", "noload", "none"];
const STOPS = ["cut", "conv1", "conv2", "conv3", "conv4", "fc1", "none"];
const NONE = "@compute @workgroup_size(1)\nfn main() {}\n";

function compile(code, spv) {
  const naga = spv.replace(/\.spv$/, ".naga.wgsl");
  writeFileSync(naga, nagaWgsl(code));
  execFileSync(TOOLS.wgsl2spv, [naga, spv], { stdio: ["ignore", "pipe", "inherit"] });
  stripCaps(spv);
  patchBitcasts(spv);
  execFileSync(TOOLS.val, ["--target-env", "vulkan1.3", spv], { stdio: ["ignore", "pipe", "inherit"] });
}

export async function build({ out }) {
  const asked = (process.env.LIZ_PROBE ?? "").split(",").filter(Boolean).map((s) => s.split(":"));
  if (!asked.length) return [];
  const treeOf = (v) => JSON.parse(readFileSync(new URL(`setup/${v}.json`, out), "utf8"));
  const same = (tree, pair, code, what) => {
    const mod = tree.objects.pipelines[pair.p].module;
    if (code !== readFileSync(new URL(`wgsl/${mod}.wgsl`, out), "utf8")) throw new Error(`nets_probe: ${what} built here is not the tree's module ${mod} (a stale tree, or another form)`);
  };
  mkdirSync(new URL("nets/", out), { recursive: true });
  const entries = [];
  const add = (name, code, entry) => {
    const spv = `nets/probe_${name}.spv`;
    compile(code, fileURLToPath(new URL(spv, out)));
    entries.push({ ...entry, weights: null, spv, spec: [], blob: null, needs: [] });
    console.log(`  probe ${name}: ${spv}`);
  };
  // ai: the bank: one int8 form for both variants (the proposer takes no subgroup built-in)
  if (asked.some(([stage]) => BANK[stage])) {
    const tree = treeOf("int8-sg"), form = tree.front.fcn2Form, json = await loadNet(new URL(WEIGHTS_INT8, LIB));
    const source = (stage, probe) => fcn2Source(json, { ...form, f16: false, packed: false, stage, probe });
    for (const stage of Object.keys(BANK)) same(tree, tree.front[BANK[stage]], source(stage, ""), `the bank's ${stage} stage`);
    for (const [stage, probe] of asked.filter(([stage]) => BANK[stage])) {
      if (!PROBES.includes(probe)) throw new Error(`LIZ_PROBE ${stage}:${probe}: a probe of ${PROBES.join(", ")}`);
      add(`${stage}_${probe}`, probe === "none" ? NONE : source(stage, probe), { variants: ["int8-sg", "int8"], role: `front.${BANK[stage]}` });
    }
  }
  // ai: the small classifier's int8 twin: a kernel a variant (subgroups or not), dispatched at its plan's P
  const small = asked.filter(([stage]) => stage === "small");
  if (small.length) {
    const w = packWeightsInt8(await loadNet(new URL(INT8_TWINS[CASCADE.weights], LIB)));
    for (const v of ["int8-sg", "int8"]) {
      const tree = treeOf(v), f = { int8: true, subgroups: v.endsWith("-sg") }, P = classifyPlan(w.arch, f).P;
      const source = (stop) => classifySource(w.offsets, w.arch, { ...f, role: "first", rest: CASCADE.rest, stop });
      same(tree, tree.front.twins.small.pipe, source(null), `${v}'s small classifier`);
      for (const [, stop] of small) {
        if (!STOPS.includes(stop)) throw new Error(`LIZ_PROBE small:${stop}: a stop of ${STOPS.join(", ")}`);
        add(`small_${stop}_${v}`, stop === "none" ? NONE : source(stop), { variants: [v], role: "front.twins.small.pipe", P });
      }
    }
  }
  const unknown = asked.filter(([stage]) => !BANK[stage] && stage !== "small");
  if (unknown.length) throw new Error(`LIZ_PROBE ${unknown.map((u) => u.join(":")).join(",")}: a stage of x, net or small`);
  return entries;
}
