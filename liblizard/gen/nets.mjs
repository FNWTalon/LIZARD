// ai: The native nets (core/nets/README.md): kernels on the device's cooperative-matrix units in place of the
// ai: web's int8 net kernels, over the same bind group layouts, so the host swaps a pipeline and a weights buffer by id
// ai: (core/dec/setup.cpp applyNativeNets). Each net's generator is a module here exporting `build({ out, glslang })`,
// ai: which writes its SPIR-V and weights under out/nets/ and returns its manifest entries; this driver runs every one
// ai: present and writes out/nets/manifest.json.
//   node liblizard/gen/nets.mjs [out]    (LIZ_NETS=1 lizard-android/build.sh gen runs it after gen.mjs)
// ai: GLSL with GL_KHR_cooperative_matrix needs glslang 16, placed by hand at liblizard/.tools/glslang/bin/glslang (a
// ai: KhronosGroup/glslang release; no build step fetches it); the NDK's glslc (2022) has no such extension.
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = new URL("./", import.meta.url);
const OUT = process.argv[2] ? pathToFileURL(`${resolve(process.argv[2])}/`) : new URL("../out/", HERE);
const glslang = fileURLToPath(new URL("../.tools/glslang/bin/glslang", HERE));
if (!existsSync(glslang)) throw new Error(`no ${glslang} (glslang 16, placed by hand: a KhronosGroup/glslang release's bin/glslang)`);
mkdirSync(new URL("nets/", OUT), { recursive: true });
const kernels = [];
for (const name of ["nets_cls.mjs", "nets_prop.mjs", "nets_probe.mjs"]) {
  if (!existsSync(new URL(name, HERE))) continue;
  const m = await import(new URL(name, HERE).href);
  const k = await m.build({ out: OUT, glslang });
  console.log(`${name}: ${k.length} kernel${k.length === 1 ? "" : "s"}`);
  kernels.push(...k);
}
writeFileSync(new URL("nets/manifest.json", OUT), JSON.stringify({ kernels }, null, 1));
console.log(`out/nets/manifest.json: ${kernels.length} kernels`);
