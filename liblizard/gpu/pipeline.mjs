// ai: A compute pipeline, compiled asynchronously with no error scope of its own, so any number of them can be in
// ai: flight at once (2026-09-29, the worker's start: STATUS "The ramp at the start"). An error scope is the device's,
// ai: pushed and popped in call order, and two compiles awaited across each other pop each other's; a failed creation
// ai: rejects its own promise instead (GPUPipelineError), and the module's own messages, where it has any, name the line.
// ai: device: a GPUDevice; layout: a GPUPipelineLayout; label: for the browser's own messages; code: a stage's source
// ai: (shaderModule).
export async function computePipeline(device, { code, layout, label }) {
  const module = shaderModule(device, code, label);
  try {
    return await device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: "main" } });
  } catch (e) {
    const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (!errs.length) throw e;
    throw new Error(errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + String(code).split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
  }
}

// ai: A stage's shader module from its source: WGSL text (the browser sender's kernels, encoder.mjs, on the page's
// ai: WebGPU device), or { glsl } (a decoder stage, glsl/, on the recording device of ../gen/mockgpu.mjs, which gen.mjs
// ai: compiles with glslang). The decoder's WGSL twins went 2026-10-31 (archive/webgpu-decoder/).
export function shaderModule(device, src, label) {
  if (typeof src === "string") return device.createShaderModule({ label, code: src });
  if (typeof src?.glsl !== "string") throw new Error(`${label ?? "a stage"}: a source is WGSL text (the sender's) or { glsl } (a decoder stage)`);
  return device.createShaderModule({ label, glsl: src.glsl });
}
