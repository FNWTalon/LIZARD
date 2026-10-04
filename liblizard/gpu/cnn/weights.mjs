// ai: The classifier's weights (scripts/gpu/cnn/README.md, "Weights" and "Architectures"): a classifier file's document (netfile.mjs)
// ai: packed into one storage buffer, layer by layer, weights then biases, and the element offset of each part. The
// ai: layer list comes from the file's "arch" block; the shader is generated from the same block and these offsets
// ai: (gpu/wgsl/classify.mjs), so a file whose shapes disagree with its arch is refused. PyTorch order inside a layer:
// ai: conv [out, in, ky, kx], fc [out, in] with the input flattened as (c, y, x).
import { PATCH } from "./patch.mjs";
import { loadNet } from "./netfile.mjs";

export const V1 = { name: "cnn-v1", layers: [
  { type: "conv", out: 8, stride: 2 }, { type: "conv", out: 16, stride: 1 }, { type: "conv", out: 32, stride: 2 }, { type: "conv", out: 32, stride: 2 },
  { type: "fc", out: 32 }, { type: "fc", out: 5 } ] };

const size = (shape) => shape.reduce((a, b) => a * b, 1);

// Every layer of an arch with its shapes and where its weights sit: convs are 3 x 3, padding 1, so an S-wide input
// gives ceil(S / stride); the first fc takes the last conv's output flattened. Names are conv1.., fc1.. in order,
// which is what the trainer writes.
export function planArch(arch = V1) {
  const layers = [];
  let C = 1, S = PATCH, nConv = 0, nFc = 0;
  for (const l of arch.layers) {
    if (l.type === "conv") {
      if (nFc) throw new Error(`arch ${arch.name}: conv after fc`);
      const stride = l.stride ?? 1, O = Math.floor((S - 1) / stride) + 1;
      layers.push({ name: `conv${++nConv}`, type: "conv", shape: [l.out, C, 3, 3], stride, inC: C, inS: S, outC: l.out, outS: O, macs: l.out * O * O * C * 9 });
      C = l.out; S = O;
    } else if (l.type === "fc") {
      const inN = nFc ? C : C * S * S;
      layers.push({ name: `fc${++nFc}`, type: "fc", shape: [l.out, inN], inN, outN: l.out, macs: l.out * inN });
      C = l.out;
    } else throw new Error(`arch ${arch.name}: layer type ${l.type}`);
  }
  const last = layers[layers.length - 1];
  if (!last || last.type !== "fc" || last.outN !== 5) throw new Error(`arch ${arch.name}: the last layer must be fc with 5 outputs`);
  const offsets = {};
  let total = 0;
  for (const l of layers) { offsets[`${l.name}w`] = total; total += size(l.shape); offsets[`${l.name}b`] = total; total += l.shape[0]; }
  return { name: arch.name, layers, offsets, total, macs: layers.reduce((a, l) => a + l.macs, 0) };
}

// cnn-v1's constants, for the scripts that replicate that network in JS (scripts/gpu/cnn/check_scaled.mjs).
export const SPEC = V1.name;
const v1 = planArch(V1);
export const LAYERS = v1.layers.map((l) => [l.name, l.shape]);
export const OFFSETS = v1.offsets;
export const TOTAL = v1.total;

// ai: A float packer: a q8 file (arch.quant, codes and no float w or b) is refused by name; its packer is
// ai: gpu/wgsl/classify_gemm.mjs packWeightsInt8.
function pack(json, data) {
  if (json.arch?.quant) throw new Error(`weights: ${json.arch.name} is an int8 file (arch.quant); classify_gemm.mjs packWeightsInt8 packs it`);
  const arch = json.arch, plan = planArch(arch);
  if (data.length < plan.total) throw new Error(`weights: ${data.length} slots for ${plan.total} parameters`);
  for (const { name, shape } of plan.layers) {
    const l = json.layers.find((x) => x.name === name);
    if (!l) throw new Error(`weights: no layer ${name}`);
    if (JSON.stringify(l.shape) !== JSON.stringify(shape) || l.w.length !== size(shape) || l.b.length !== shape[0]) throw new Error(`weights: ${name} is ${JSON.stringify(l.shape)} (${l.w.length} + ${l.b.length}), expected ${JSON.stringify(shape)}`);
    data.set(l.w, plan.offsets[`${name}w`]);
    data.set(l.b, plan.offsets[`${name}b`]);
  }
  return { data, offsets: plan.offsets, arch, val: json.val ?? null, total: plan.total, macs: plan.macs };
}

export const packWeights = (json) => pack(json, new Float32Array(planArch(json.arch).total));

// IEEE half, round to nearest even; the buffer is padded to a whole number of u32.
export function toHalf(x) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0], s = (u >>> 16) & 0x8000;
  let e = ((u >>> 23) & 0xff) - 127 + 15, m = u & 0x7fffff;
  if (((u >>> 23) & 0xff) === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  if (e >= 31) return s | 0x7c00;
  if (e <= 0) {
    if (e < -10) return s;
    m |= 0x800000;
    const shift = 14 - e, half = m >> shift, rem = m & ((1 << shift) - 1), mid = 1 << (shift - 1);
    return s | (half + (rem > mid || (rem === mid && (half & 1)) ? 1 : 0));
  }
  let h = s | (e << 10) | (m >> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return h;
}
export function packWeightsF16(json) {
  const f = packWeights(json), data = new Uint16Array((f.total + 1) & ~1);
  for (let i = 0; i < f.total; i++) data[i] = toHalf(f.data[i]);
  return { ...f, data };
}

export async function loadWeights(url = new URL("./weights.safetensors", import.meta.url)) {
  return packWeights(await loadNet(url));
}
