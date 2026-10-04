// ai: A compute pipeline from WGSL, compiled asynchronously with no error scope of its own, so any number of them can be
// ai: in flight at once (2026-09-29, the worker's start: STATUS "The ramp at the start"). An error scope is the device's,
// ai: pushed and popped in call order, and two compiles awaited across each other pop each other's; a failed creation
// ai: rejects its own promise instead (GPUPipelineError), and the module's own messages, where it has any, name the line.
// ai: device: a GPUDevice; layout: a GPUPipelineLayout; label: for the browser's own messages.
export async function computePipeline(device, { code, layout, label }) {
  const module = device.createShaderModule({ label, code });
  try {
    return await device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: "main" } });
  } catch (e) {
    const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (!errs.length) throw e;
    throw new Error(errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n") + "\n" + String(code).split("\n").map((l, i) => `${i + 1} ${l}`).join("\n"));
  }
}
