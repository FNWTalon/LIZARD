// ai: A WebGPU device in node that runs nothing and records the objects asked of it, so the web decoder's own host
// ai: (liblizard/gpu/decoder.mjs FrontHalf) can build under node: every shader module's WGSL, every bind group layout,
// ai: pipeline, buffer (with the bytes written into it), texture and bind group. gen.mjs takes the static objects
// ai: create() makes as the native decoder's setup (its WGSL goes to naga).
//
// ai: Nothing is computed: a mapped range reads as zeros, so a batch reads back "nothing found" and a timestamp 0.
// ai: installGlobals() must run before any liblizard/gpu module is imported: several read GPUBufferUsage at load.
import { createHash } from "node:crypto";

export const BUFFER = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
export const TEXTURE = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
export const STAGE = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
export const INT8 = "packed_4x8_integer_dot_product";

export const hash = (s) => createHash("sha256").update(s).digest("hex").slice(0, 16);

// ai: The globals the liblizard/gpu modules read, and navigator.gpu with the WGSL language features the int8 nets ask for.
export function installGlobals({ wgsl = [INT8] } = {}) {
  globalThis.GPUBufferUsage = BUFFER;
  globalThis.GPUTextureUsage = TEXTURE;
  globalThis.GPUShaderStage = STAGE;
  globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
  Object.defineProperty(globalThis, "navigator", { value: { gpu: { wgslLanguageFeatures: new Set(wgsl) } }, configurable: true, writable: true });
}

// ai: The bytes a queue write or a writeTexture takes: an ArrayBuffer, a typed array or a DataView, and the WebGPU
// ai: offset and size (elements of a typed array, bytes otherwise).
function bytesOf(data, dataOffset = 0, size) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
  const el = data.BYTES_PER_ELEMENT ?? 1, len = size ?? (data.byteLength / el - dataOffset);
  return new Uint8Array(data.buffer, data.byteOffset + dataOffset * el, len * el);
}

// ai: What a bind group layout entry binds: tex (a 2d-array float texture), ro, rw or uniform.
function entryType(e) {
  if (e.texture) return "tex";
  const t = e.buffer?.type ?? "uniform";
  return t === "read-only-storage" ? "ro" : t === "storage" ? "rw" : "uniform";
}

export class Recorder {
  constructor() {
    this.seq = { b: 0, t: 0, v: 0, L: 0, P: 0, m: 0, p: 0, g: 0, q: 0 };
    this.objects = new Map();   // id -> object
    this.modules = new Map();   // hash -> WGSL
  }
  id(kind) { return `${kind}${this.seq[kind]++}`; }
  add(kind, o) { o.__id = this.id(kind); o.__kind = kind; this.objects.set(o.__id, o); return o; }
}

class MockBuffer {
  constructor(rec, { size, usage, label, mappedAtCreation }) {
    Object.assign(this, { rec, size, usage, label: label ?? null, data: null, live: true, mapState: "unmapped" });
    rec.add("b", this);
    if (mappedAtCreation) this.mapState = "mapped";
  }
  bytes() { return (this.data ??= new Uint8Array(this.size)); }
  destroy() { this.live = false; }
  mapAsync() { this.mapState = "mapped"; return Promise.resolve(); }
  getMappedRange(offset = 0, size) { return new ArrayBuffer(size ?? this.size - offset); }
  unmap() { this.mapState = "unmapped"; }
}

class MockTexture {
  constructor(rec, { size, format, usage }) {
    const [w, h, d = 1] = Array.isArray(size) ? size : [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];
    Object.assign(this, { rec, width: w, height: h, depthOrArrayLayers: d, format, usage, live: true });
    rec.add("t", this);
  }
  createView(desc = {}) { return this.rec.add("v", { tex: this, desc }); }
  destroy() { this.live = false; }
}

// ai: A command encoder that records nothing: a batch's commands are the native host's own (core/dec/front.cpp).
class MockEncoder {
  beginComputePass() {
    const nop = () => {};
    return { setPipeline: nop, setBindGroup: nop, dispatchWorkgroups: nop, dispatchWorkgroupsIndirect: nop, end: nop };
  }
  copyBufferToBuffer() {}
  copyTextureToTexture() {}
  clearBuffer() {}
  resolveQuerySet() {}
  finish() { return {}; }
}

export class MockDevice {
  constructor(rec, { features, limits }) {
    Object.assign(this, { rec, features: new Set(features), limits: { ...limits }, destroyed: false });
    this.lost = new Promise(() => {});
    this.queue = {
      writeBuffer: (b, off, data, dataOffset = 0, size) => {
        const src = bytesOf(data, dataOffset, size);
        if (off + src.byteLength > b.size) throw new Error(`writeBuffer past ${b.__id}'s ${b.size} bytes`);
        b.bytes().set(src, off);
      },
      writeTexture: () => {},
      submit: () => {},
      onSubmittedWorkDone: () => Promise.resolve(),
    };
  }
  addEventListener() {}
  pushErrorScope() {}
  popErrorScope() { return Promise.resolve(null); }
  destroy() { this.destroyed = true; }
  createBuffer(desc) { return new MockBuffer(this.rec, desc); }
  createTexture(desc) { return new MockTexture(this.rec, desc); }
  createQuerySet({ type, count }) { return this.rec.add("q", { type, count, destroy: () => {} }); }
  createShaderModule({ code, label }) {
    const h = hash(code), m = this.rec.add("m", { code, hash: h, label: label ?? null, getCompilationInfo: async () => ({ messages: [] }) });
    this.rec.modules.set(h, code);
    return m;
  }
  createBindGroupLayout({ entries }) {
    const types = [];
    for (const e of entries) types[e.binding] = entryType(e);
    return this.rec.add("L", { entries: types });
  }
  createPipelineLayout({ bindGroupLayouts }) {
    return this.rec.add("P", { groups: bindGroupLayouts });
  }
  createComputePipeline({ layout, compute, label }) {
    const p = this.rec.add("p", { layout, module: compute.module, entry: compute.entryPoint ?? "main", label: label ?? null });
    p.getBindGroupLayout = (i) => layout.groups[i];
    return p;
  }
  async createComputePipelineAsync(desc) { return this.createComputePipeline(desc); }
  createBindGroup({ layout, entries }) {
    const out = entries.map((e) => {
      const r = e.resource;
      if (r?.__kind === "v") return { binding: e.binding, view: r.__id };
      if (r?.buffer) return { binding: e.binding, buf: r.buffer.__id, off: r.offset ?? 0, size: r.size ?? null };
      return { binding: e.binding, other: true };
    }).sort((a, b) => a.binding - b.binding);
    return this.rec.add("g", { layout, entries: out });
  }
  createCommandEncoder() { return new MockEncoder(); }
}

// ai: An adapter over one recorder: requestDevice gives a MockDevice with the features asked and the adapter's limits.
export function mockAdapter(rec, { features, limits, info = {} }) {
  return {
    features: new Set(features),
    limits: { ...limits },
    info,
    requestDevice: async ({ requiredFeatures = [], requiredLimits = {} } = {}) => new MockDevice(rec, { features: requiredFeatures, limits: { ...limits, ...requiredLimits } }),
  };
}
