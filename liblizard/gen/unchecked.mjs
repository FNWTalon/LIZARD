// ai: Every module of out/naga compiled with naga's bound checks left out by policy, into out/<dir>: what the checks
// ai: cost on a device (STATUS "Native speed": all off, the S26 2.80 to 2.51 ms of GPU a frame). The app's build
// ai: (out/spv, gen.mjs) has all three off since 2026-09-30; `none` here builds every check Restrict, the build before,
// ai: for an A/B. The modules go through gen.mjs's own patches and spirv-val; the ingest's SPIR-V is copied.
//   node liblizard/gen/unchecked.mjs <dir> [index,buffer,image|none]      (after lizard-android/build.sh gen; all three when none is named)
//   LIZ_SPV=<dir> lizard-android/build/linux/lizard_gpu_check replay <run> 48 int8-sg    (on the phone: push out/<dir> beside out/spv)
import { readdirSync, mkdirSync, copyFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { stripCaps, patchBitcasts, TOOLS } from "./gen.mjs";

const OUT = fileURLToPath(new URL("../out/", import.meta.url));
const dir = process.argv[2], policies = process.argv[3];
if (!dir || dir === "spv" || dir.includes("/")) throw new Error("unchecked.mjs <dir under out/, not spv> [index,buffer,image]");
mkdirSync(`${OUT}${dir}`, { recursive: true });
let n = 0;
for (const f of readdirSync(`${OUT}naga`).filter((f) => f.endsWith(".wgsl"))) {
  const spv = `${OUT}${dir}/${f.replace(/\.wgsl$/, "")}.spv`;
  execFileSync(TOOLS.wgsl2spv, [`${OUT}naga/${f}`, spv, ...(policies === "none" ? [] : [policies ? `--unchecked=${policies}` : "--unchecked"])]);
  stripCaps(spv);
  patchBitcasts(spv);
  execFileSync(TOOLS.val, ["--target-env", "vulkan1.3", spv]);
  n++;
}
for (const f of readdirSync(`${OUT}spv`).filter((f) => f.startsWith("ingest_"))) copyFileSync(`${OUT}spv/${f}`, `${OUT}${dir}/${f}`);
console.log(`${n} modules in out/${dir}, bound checks off: ${policies === "none" ? "none (every check Restrict)" : policies ?? "index, buffer, image"}`);
