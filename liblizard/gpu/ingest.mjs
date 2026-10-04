// ai: F0's host (wgsl/ingest.mjs): a frame's crop to luma in an r8unorm layer, a render pass a frame. The decoder
// ai: runs it for a batch frame that carries a source in place of bytes (decoder.mjs batch()); the writeTexture of
// ai: bytes beside it stays, the fallback and the recording path.
import { INGEST } from "./wgsl/ingest.mjs";

export class Ingest {
  static async create(device) {
    const bgl = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} }] });
    const module = device.createShaderModule({ code: INGEST });
    const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error(`ingest: ${errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    const pipeline = await device.createRenderPipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "r8unorm" }] },
      primitive: { topology: "triangle-list" },
    });
    return new Ingest(device, bgl, pipeline);
  }

  // ai: zeroCopy: how the imports went, by Chrome's non-standard GPUExternalTexture.isZeroCopy (true: the GPU read the
  // ai: source's own buffer; false: Chrome copied it first; unknown: the attribute is absent, which it is unless
  // ai: chrome://flags/#enable-webgpu-developer-features is on). Asked 2026-09-26 for the phone's 1440 and 4K frame
  // ai: drops (STATUS "Why 1440 and 4K drop frames on the phone"): a copy of the whole camera frame per import is the
  // ai: suspect. A console line on the first import and whenever the answer or the source's size changes.
  constructor(device, bgl, pipeline) { Object.assign(this, { device, bgl, pipeline, zeroCopy: { yes: 0, no: 0, unknown: 0 }, zcLast: "" }); }

  #noteZeroCopy(ext, source) {
    const z = ext.isZeroCopy, k = z === true ? "yes" : z === false ? "no" : "unknown";
    this.zeroCopy[k]++;
    const size = `${source.codedWidth ?? source.videoWidth}x${source.codedHeight ?? source.videoHeight}${source.format ? " " + source.format : ""}`;
    if (`${k} ${size}` !== this.zcLast) {
      this.zcLast = `${k} ${size}`;
      console.info(`F0 import: isZeroCopy ${k === "unknown" ? "unknown (turn on chrome://flags/#enable-webgpu-developer-features)" : z} for a ${size} frame`);
    }
  }

  // ai: Throws unless the crop (x, y, w, h) lies inside its source, whose size is the VideoFrame's display size (a
  // ai: <video>'s videoWidth x videoHeight). A caller with error scopes open checks first, so a bad crop leaves none.
  static check({ source, x = 0, y = 0, w, h }) {
    const sw = source.displayWidth ?? source.videoWidth, sh = source.displayHeight ?? source.videoHeight;
    if (!(x >= 0 && y >= 0 && w >= 1 && h >= 1 && x + w <= sw && y + h <= sh && x === Math.floor(x) && y === Math.floor(y))) throw new Error(`crop ${x},${y} ${w}x${h}: not whole pixels inside the ${sw}x${sh} source`);
  }

  // ai: jobs: [{ source, x, y, w, h, view }]: the crop of source (a VideoFrame, or anything importExternalTexture
  // ai: takes) to view (one r8unorm layer, RENDER_ATTACHMENT, at least w x h), pixel (x, y) to the view's (0, 0).
  // ai: Each source is imported here, so enc must be submitted before the task ends and before a source is closed;
  // ai: the layer outside w x h is cleared (clear, not load: a tiled phone GPU would read the old layer in).
  // ai: ts ({ querySet, index }, for a measurement): the first pass's start at index, the last pass's end at index + 1.
  encode(enc, jobs, ts = null) {
    jobs.forEach((j, k) => {
      const ext = this.device.importExternalTexture({ source: j.source });
      this.#noteZeroCopy(ext, j.source);
      const group = this.device.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: ext }] });
      const tw = ts && (k === 0 || k === jobs.length - 1) ? { timestampWrites: { querySet: ts.querySet, ...(k === 0 ? { beginningOfPassWriteIndex: ts.index } : {}), ...(k === jobs.length - 1 ? { endOfPassWriteIndex: ts.index + 1 } : {}) } } : {};
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: j.view, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }], ...tw });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, group);
      pass.setViewport(0, 0, j.w, j.h, 0, 1);
      pass.draw(3, 1, 0, (j.x ?? 0) + 65536 * (j.y ?? 0));
      pass.end();
    });
  }
}
