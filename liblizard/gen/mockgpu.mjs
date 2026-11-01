// ai: A WebGPU device in node that runs nothing and records the objects asked of it, so the decoder's setup
// ai: (liblizard/gpu/decoder.mjs FrontHalf.create) can build under node: every shader module's GLSL, every bind group
// ai: layout, pipeline layout, pipeline and buffer (with the bytes written into it). gen.mjs takes what create() makes
// ai: as the native decoder's setup (a module's GLSL to glslang). Only what the setup's build calls is here: a lane's
// ai: textures, bind groups, command encoders and readbacks are the native host's (core/dec/front.cpp).
// ai: installGlobals() must run before any liblizard/gpu module is imported: several read GPUBufferUsage at load.
import { createHash } from "node:crypto";

export const BUFFER = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
export const STAGE = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

export const hash = (s) => createHash("sha256").update(s).digest("hex").slice(0, 16);

// ai: The globals the liblizard/gpu modules read.
export function installGlobals() {
  globalThis.GPUBufferUsage = BUFFER;
  globalThis.GPUShaderStage = STAGE;
}

// ai: The bytes a queue write takes: an ArrayBuffer, a typed array or a DataView, and the WebGPU offset and size
// ai: (elements of a typed array, bytes otherwise).
function bytesOf(data, dataOffset = 0, size) {
  if (data instanceof ArrayBuffer) return new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
  const el = data.BYTES_PER_ELEMENT ?? 1, len = size ?? (data.byteLength / el - dataOffset);
  return new Uint8Array(data.buffer, data.byteOffset + dataOffset * el, len * el);
}

// ai: What a bind group layout entry binds: tex (a 2d-array float texture), sampler (a filtering sampler: natively the
// ai: device's linear sampler, immutable in the layout, core/wg Device::linearSampler), ro, rw or uniform.
function entryType(e) {
  if (e.texture) return "tex";
  if (e.sampler) return "sampler";
  const t = e.buffer?.type ?? "uniform";
  return t === "read-only-storage" ? "ro" : t === "storage" ? "rw" : "uniform";
}

export class Recorder {
  constructor() {
    this.seq = { b: 0, L: 0, P: 0, m: 0, p: 0 };
    this.objects = new Map();   // id -> object
    this.modules = new Map();   // hash -> { lang: "glsl", code }
  }
  id(kind) { return `${kind}${this.seq[kind]++}`; }
  add(kind, o) { o.__id = this.id(kind); o.__kind = kind; this.objects.set(o.__id, o); return o; }
}

class MockBuffer {
  constructor(rec, { size, usage, label }) {
    Object.assign(this, { rec, size, usage, label: label ?? null, data: null });
    rec.add("b", this);
  }
  bytes() { return (this.data ??= new Uint8Array(this.size)); }
}

export class MockDevice {
  constructor(rec, { features, limits }) {
    Object.assign(this, { rec, features: new Set(features), limits: { ...limits } });
    this.queue = {
      writeBuffer: (b, off, data, dataOffset = 0, size) => {
        const src = bytesOf(data, dataOffset, size);
        if (off + src.byteLength > b.size) throw new Error(`writeBuffer past ${b.__id}'s ${b.size} bytes`);
        b.bytes().set(src, off);
      },
    };
  }
  createBuffer(desc) { return new MockBuffer(this.rec, desc); }
  // ai: A decoder stage is GLSL (gpu/pipeline.mjs shaderModule's { glsl }); WGSL here is a mistake (the decoder's WGSL
  // ai: twins are archived, the sender's kernels reach gen.mjs through gen/sender.mjs).
  createShaderModule({ glsl, label }) {
    if (typeof glsl !== "string") throw new Error(`shader module ${label ?? ""}: no GLSL (a decoder stage is GLSL)`);
    const h = hash(glsl), m = this.rec.add("m", { code: glsl, lang: "glsl", hash: h, label: label ?? null, getCompilationInfo: async () => ({ messages: [] }) });
    this.rec.modules.set(h, { lang: "glsl", code: glsl });
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
