// ai: The nets' weights files in JS (scripts/gpu/cnn/README.md "Weights"; scripts/gpu/cnn/netfile.py writes them with the official safetensors
// ai: package and defines the names, dtypes and metadata): a hand-written reader of the safetensors standard, and the
// ai: document the decoder works on rebuilt from it, as the JSON files gave it before 2026-09-26: { spec, arch (with
// ai: quant), layers: [{ name, shape, dil (proposers only), then its fields as plain arrays of numbers }], val }. The
// ai: fields are w, b (float nets and the int8 head) or wq, sw, bq, m, s (int8 convs). No node import at the top, so
// ai: a page or worker can import it.

const DTYPES = { F64: [8, "getFloat64"], F32: [4, "getFloat32"], I64: [8, "getBigInt64"], I32: [4, "getInt32"], I16: [2, "getInt16"], I8: [1, "getInt8"] };
const TENSOR = { wq: "weight_q", sw: "weight_scale", bq: "bias_q", m: "requant_scale", s: "dequant_scale", w: "weight", b: "bias" };

// ai: The standard: an 8-byte little-endian header length, the JSON header ({ name: { dtype, shape, data_offsets } },
// ai: data_offsets relative to the byte after the header, and __metadata__), then the little-endian tensor bytes.
// ai: Returns { tensors: { name: { dtype, shape, values (a plain array) } }, metadata }.
export function readSafetensors(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = Number(dv.getBigUint64(0, true)), base = 8 + n;
  if (base > u8.byteLength) throw new Error(`safetensors: a header of ${n} bytes in a file of ${u8.byteLength}`);
  const { __metadata__: metadata = {}, ...entries } = JSON.parse(new TextDecoder().decode(u8.subarray(8, base)));
  const tensors = {};
  for (const [name, { dtype, shape, data_offsets: [a, b] }] of Object.entries(entries)) {
    const t = DTYPES[dtype];
    if (!t) throw new Error(`safetensors: ${name} is ${dtype}, which this reader does not read`);
    const count = shape.reduce((x, y) => x * y, 1), [size, get] = t;
    if (b - a !== count * size || base + b > u8.byteLength) throw new Error(`safetensors: ${name} ${dtype}${JSON.stringify(shape)} at bytes ${a} to ${b} of ${u8.byteLength - base}`);
    const values = new Array(count);
    for (let i = 0; i < count; i++) values[i] = dv[get](base + a + i * size, true);
    if (dtype === "I64") for (let i = 0; i < count; i++) values[i] = Number(values[i]);
    tensors[name] = { dtype, shape, values };
  }
  return { tensors, metadata };
}

// ai: [layer name, tensor prefix, dil or undefined] for each layer, in the document's order (scripts/gpu/cnn/netfile.py slots)
function slots(spec, arch) {
  if (spec.startsWith("proposer-")) return [...arch.layers.map((l, i) => [`conv${i + 1}`, `convs.${i}`, l.dil]), ["head", "head", 1]];
  // ai: a float classifier, or an int8 one (q8c.mjs), whose tensors take the same layer names
  if (spec !== "cnn-v1" && spec !== "cnn-v1-q8") throw new Error(`weights: spec ${spec}: cnn-v1 or cnn-v1-q8 (a classifier) or proposer-*`);
  const count = { conv: 0, fc: 0 };
  return arch.layers.map((l) => { const name = `${l.type}${++count[l.type]}`; return [name, name, undefined]; });
}

// ai: The document from a weights file's bytes (ArrayBuffer, Uint8Array or a node Buffer).
export function readNet(bytes) {
  const { tensors, metadata: m } = readSafetensors(bytes);
  if (m.format !== "pt" || !m["lizard.spec"] || !m["lizard.arch"]) throw new Error("weights: no lizard.spec and lizard.arch metadata: not a LIZARD weights file");
  const spec = m["lizard.spec"], arch = JSON.parse(m["lizard.arch"]), used = new Set();
  if (m["lizard.quant"]) arch.quant = JSON.parse(m["lizard.quant"]);
  const layers = slots(spec, arch).map(([name, prefix, dil]) => {
    const fields = Object.keys(TENSOR).filter((f) => `${prefix}.${TENSOR[f]}` in tensors);
    if (!fields.length) throw new Error(`weights: no tensors for ${name} (${prefix}.*)`);
    const L = { name, shape: [...tensors[`${prefix}.${TENSOR[fields[0]]}`].shape] };
    if (dil !== undefined) L.dil = dil;
    for (const f of fields) { const key = `${prefix}.${TENSOR[f]}`; L[f] = tensors[key].values; used.add(key); }
    return L;
  });
  const extra = Object.keys(tensors).filter((k) => !used.has(k));
  if (extra.length) throw new Error(`weights: tensors ${extra.join(", ")} belong to no layer`);
  const doc = { spec, arch, layers };
  if (m["lizard.val"] !== undefined) doc.val = JSON.parse(m["lizard.val"]);
  return doc;
}

// ai: A weights file's bytes by URL: node reads it, a page or worker fetches it (also the LDPC stop rule's,
// ai: gpu/back/stop/rule.mjs).
export async function readUrl(url) {
  if (!String(url).endsWith(".safetensors")) throw new Error(`weights ${url}: not a .safetensors file (an old JSON weights file converts with scripts/gpu/cnn/netfile.py, the lab's, not published)`);
  if (typeof process !== "undefined" && process.versions?.node) return (await import("node:fs/promises")).readFile(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`weights ${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

// ai: A weights file by URL. The document's `source` (not enumerable) is the URL it came from, by which the decoder
// ai: knows an installed float file and finds its int8 twin (gpu/wgsl/classify.mjs INT8_TWINS).
export async function loadNet(url) {
  return Object.defineProperty(readNet(await readUrl(url)), "source", { value: String(url) });
}

// ai: A classifier file's check reference, a path under liblizard/: for a bare name (an installed file under gpu/cnn/)
// ai: build/cnn/reference-X.bin (in research/build/cnn/) for weights-X.safetensors (the float trainer's reference.bin
// ai: or a q8 file's CQ81), for a path under liblizard/ reference-<name>.bin beside it, as the trainers write them
// ai: (scripts/gpu/cnn/check.mjs, scripts/gpu/cnn/test_gemm.mjs).
export function referenceOf(weights) {
  if (!weights.includes("/")) return `build/cnn/${weights.replace(/^weights/, "reference").replace(/\.safetensors$/, ".bin")}`;
  const i = weights.lastIndexOf("/");
  return `${weights.slice(0, i)}/reference-${weights.slice(i + 1).replace(/\.safetensors$/, ".bin")}`;
}
