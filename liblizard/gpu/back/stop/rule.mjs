// ai: The learned LDPC stop rule for the decoder (scripts/gpu/back/stop/README.md; the trainer is scripts/gpu/back/stop/fit.py): its weights file read and checked
// ai: (readStop, loadStop), folded into the form the kernel runs exactly as scripts/gpu/back/stop/net.py fold() folds it (the same float64
// ai: operations in the same order, then one rounding to float32), the words the kernel reads (netWords, laid out for
// ai: gpu/wgsl/back_ldpc.mjs), and that form's arithmetic in float32 (scoreF32, gives), which gpu/back/ref_ldpc.mjs's
// ai: decoders apply. The rule: after iteration t (1 to 29), with x = (bad, low, first, prev, t, est) (the violated
// ai: checks now, their running minimum, after iteration 1, after the iteration before, t, the block's estimate BLK.x),
// ai: h1 = relu(W1 x + b1), h2 = relu(W2 h1 + b2), s = w3 . h2; give up when s < T. The order of operations (the
// ai: kernel's): a hidden unit starts at its bias and adds weight times input in input order, s adds the products
// ai: w3_j h2_j in unit order, every product and sum rounded to float32. No node import at the top, so a page or
// ai: worker can import it.
import { readSafetensors, readUrl } from "../../cnn/netfile.mjs";

export const SPEC = "ldpc-stop-mlp-v1";
export const FEATURES = ["bad", "low", "first", "fall", "it", "est"];   // ai: the training form's inputs (scripts/gpu/back/stop/net.py)
export const INPUTS = ["bad", "low", "first", "prev", "t", "est"];       // ai: the kernel's raw inputs
export const RULE_URL = new URL("./stop.safetensors", import.meta.url);

// ai: The kernel form from a weights file's bytes: { hidden, W1 (H x 6, row-major), b1, W2 (H x H), b2, w3, T } as
// ai: Float32Arrays (T a number holding a float32 value), with the file's metadata (arch, budget, provenance, val).
export function readStop(bytes) {
  const { tensors: t, metadata: meta } = readSafetensors(bytes);
  if (meta["lizard.spec"] !== SPEC) throw new Error(`stop rule: spec ${meta["lizard.spec"]}, not ${SPEC}`);
  const arch = JSON.parse(meta["lizard.arch"]), H = arch.hidden[0], m = arch.m;
  if (JSON.stringify(arch.features) !== JSON.stringify(FEATURES) || JSON.stringify(arch.inputs) !== JSON.stringify(INPUTS)) throw new Error(`stop rule: features ${arch.features}, inputs ${arch.inputs}`);
  const want = { mu: [6], sd: [6], "fc1.weight": [H, 6], "fc1.bias": [H], "fc2.weight": [H, H], "fc2.bias": [H], "out.weight": [1, H], "out.bias": [1], theta: [1] };
  for (const [k, shape] of Object.entries(want)) if (JSON.stringify(t[k]?.shape) !== JSON.stringify(shape)) throw new Error(`stop rule: ${k} is ${JSON.stringify(t[k]?.shape)}, not ${JSON.stringify(shape)}`);
  if (Object.keys(t).length !== Object.keys(want).length) throw new Error(`stop rule: tensors ${Object.keys(t)}`);
  const v = (k) => t[k].values, W = v("fc1.weight"), b = v("fc1.bias"), mu = v("mu"), sd = v("sd");
  const W1 = new Float32Array(6 * H), b1 = new Float32Array(H), f = FEATURES.reduce((o, n, i) => ((o[n] = i), o), {});
  for (let j = 0; j < H; j++) {
    const c = (k) => W[6 * j + k] / sd[k];
    const row = [c(f.bad) / m - c(f.fall) / m, c(f.low) / m, c(f.first) / m, c(f.fall) / m, c(f.it), c(f.est)];
    row.forEach((x, k) => { W1[6 * j + k] = x; });
    let shift = 0;
    for (let k = 0; k < 6; k++) shift = shift + c(k) * mu[k];
    b1[j] = b[j] - shift;
  }
  return { hidden: H, W1, b1, W2: Float32Array.from(v("fc2.weight")), b2: Float32Array.from(v("fc2.bias")), w3: Float32Array.from(v("out.weight")),
    T: Math.fround(v("theta")[0] - v("out.bias")[0]), arch, budget: +meta["lizard.budget"], meta, h1: new Float64Array(H), h2: new Float64Array(H) };
}

// ai: The installed rule (RULE_URL) unless another file is named; read once a URL.
const loaded = new Map();
export function loadStop(url = RULE_URL) {
  const key = String(url);
  if (!loaded.has(key)) loaded.set(key, readUrl(url).then(readStop).catch((e) => { loaded.delete(key); throw e; }));
  return loaded.get(key);
}

// ai: The words the kernel reads, float32 bits: W1 input-major (word k H + j is W1[j][k]), b1, W2 input-major
// ai: (7 H + k H + j), b2, w3, then T: H^2 + 9 H + 1 words, so lane j reads consecutive words across the lanes.
export function netWords(rule) {
  const H = rule.hidden, f = new Float32Array(H * H + 9 * H + 1);
  for (let k = 0; k < 6; k++) for (let j = 0; j < H; j++) f[k * H + j] = rule.W1[6 * j + k];
  f.set(rule.b1, 6 * H);
  for (let k = 0; k < H; k++) for (let j = 0; j < H; j++) f[7 * H + k * H + j] = rule.W2[H * j + k];
  f.set(rule.b2, 7 * H + H * H);
  f.set(rule.w3, 8 * H + H * H);
  f[9 * H + H * H] = rule.T;
  return new Uint32Array(f.buffer);
}

// ai: s for x = [bad, low, first, prev, t, est] in the kernel's float32 order.
export function scoreF32(rule, x) {
  const r = Math.fround, H = rule.hidden, { W1, b1, W2, b2, w3, h1, h2 } = rule;
  const x5 = r(x[5]);
  for (let j = 0; j < H; j++) {
    let a = b1[j];
    for (let k = 0; k < 5; k++) a = r(a + r(W1[6 * j + k] * x[k]));
    a = r(a + r(W1[6 * j + 5] * x5));
    h1[j] = a > 0 ? a : 0;
  }
  let s = 0;
  for (let j = 0; j < H; j++) {
    let a = b2[j];
    for (let k = 0; k < H; k++) a = r(a + r(W2[H * j + k] * h1[k]));
    const p = r(w3[j] * (a > 0 ? a : 0));
    s = j === 0 ? p : r(s + p);
  }
  return s;
}

// ai: The rule's decision after iteration t: give up.
export const gives = (rule, x) => scoreF32(rule, x) < rule.T;
