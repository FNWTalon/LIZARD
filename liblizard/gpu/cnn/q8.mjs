// ai: The int8 contract, scheme "int8-v1", in JS: scripts/gpu/cnn/q8.py's twin, for the packers, the JS reference forwards
// ai: and the kernel checks (the definition, the file and PQ81 are in scripts/gpu/cnn/q8.py's header). readQuant checks a q8 weights
// ai: file and parses it; forwardLevelQ8 runs the integer proposer on one level; packI8x4 packs codes as WGSL's
// ai: pack4xI8 does. q8c.mjs (the classifiers) runs the arithmetic, quantiseInput and the readers in "shared by both
// ai: contracts" below. No node import at the top, so a page or worker can import it.
// ai:   node gpu/cnn/q8.mjs <weights.safetensors> <reference.bin>
// ai: checks this file against a PQ81 reference the trainer wrote (the lab's, not published): every code exact, logit
// ai: and s within 1e-5, and the reference's X within 1e-5 of reference_v2.mjs's standardisation of its luma.

export const KIND = "int8", VERSION = 1, SPEC = "proposer-v2-q8", BOUND = 2 ** 24;
// ai: the i8 container's largest magnitude; the 2^24 bound counts every input at it
export const CODE = 127;

export const qmax = (bits) => 2 ** (bits - 1) - 1;

// ai: Round half to even, as WGSL's round(), torch.round and numpy's rint; Math.round rounds halves up.
export function roundEven(v) {
  const f = Math.floor(v), d = v - f;
  return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1;
}

// ai: A q8 code from an exact accumulator: clamp(roundEven(f32(acc + bq) * m), 0, qm). acc + bq is an integer under
// ai: 2^24 and m an f32, so the f64 product is exact and Math.fround rounds it once, as an f32 multiply would.
export function requant(acc, bq, m, qm) {
  const v = roundEven(Math.fround(Math.fround(acc + bq) * m));
  return v < 0 ? 0 : v > qm ? qm : v;
}

// ai: The float output of the layer that feeds the head: relu(f32(acc + bq) * s).
export const dequant = (acc, bq, s) => Math.max(Math.fround(Math.fround(acc + bq) * s), 0);

// ai: Four codes in one u32, a in the low byte, each as a two's complement byte: WGSL's pack4xI8.
export const packI8x4 = (a, b, c, d) => ((a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24)) >>> 0;

// ai: ---- shared by both contracts (q8c.mjs reads the classifiers' files with these) ----

export const isF32 = (v) => typeof v === "number" && Number.isFinite(v) && v > 0 && Math.fround(v) === v;

// ai: fail(why) throws: each reader names itself in its errors ("q8:", "q8c:").
export function ints(v, len, lo, hi, what, fail) {
  if (!Array.isArray(v) || v.length !== len || !v.every((x) => Number.isInteger(x) && x >= lo && x <= hi)) fail(`${what}: ${len} integers in ${lo}..${hi} expected`);
  return v;
}

export function f32s(v, len, what, fail) {
  if (!Array.isArray(v) || v.length !== len || !v.every(isF32)) fail(`${what}: ${len} positive f32 values expected`);
  return Float32Array.from(v);
}

// ai: arch.quant's kind, version, bits and input checked: { q, qm }. The spec is the caller's to check. A float file
// ai: (no arch.quant) is refused here, as a q8 file is by the float loaders.
export function readHeader(json, fail) {
  const q = json?.arch?.quant;
  if (!q || typeof q !== "object") fail("no arch.quant: a float weights file, which the float loaders run");
  if (q.kind !== KIND || q.version !== VERSION) fail(`quant ${q.kind} version ${q.version}: this reader runs ${KIND} version ${VERSION}`);
  if (!Number.isInteger(q.bits) || q.bits < 2 || q.bits > 8) fail(`bits ${q.bits}: 2 to 8`);
  if (!isF32(q.input?.inv) || !isF32(q.input?.scale)) fail("quant.input: scale and inv must be positive f32 values");
  return { q, qm: qmax(q.bits) };
}

// ai: arch.quant.layers against want, [[name, in, out]] in the file's order.
export function checkLayers(q, want, fail) {
  const got = (q.layers ?? []).map((l) => [l.name, l.in, l.out]);
  if (JSON.stringify(got) !== JSON.stringify(want)) fail(`quant.layers ${JSON.stringify(got)}: expected ${JSON.stringify(want)}`);
}

// ai: One quantised layer checked: its name, codes and no float w or b, its shape, wq in -qm..qm, bq, sw, and m (out
// ai: "q8") or s (out "f32") as positive f32s, and the 2^24 bound on every channel. Returns { wq (Int8Array, flat),
// ai: bq (Int32Array), sw, mul (Float32Array), bound (the worst channel's |bq| + 127 sum|wq|) }.
export function readLayer(L, name, shape, qm, out, fail) {
  if (L?.name !== name || "w" in L || "b" in L) fail(`${name}: codes and no float w or b expected`);
  if (JSON.stringify(L.shape) !== JSON.stringify(shape)) fail(`${name}: shape ${JSON.stringify(L.shape)}, expected ${JSON.stringify(shape)}`);
  const cout = shape[0], k = shape.slice(1).reduce((a, b) => a * b, 1);
  const wq = Int8Array.from(ints(L.wq, cout * k, -qm, qm, `${name}.wq`, fail));
  const bq = Int32Array.from(ints(L.bq, cout, -(BOUND - 1), BOUND - 1, `${name}.bq`, fail));
  const sw = f32s(L.sw, cout, `${name}.sw`, fail);
  const [key, other] = out === "q8" ? ["m", "s"] : ["s", "m"];
  if (other in L) fail(`${name}: a ${out} output carries ${key}, not ${other}`);
  const mul = f32s(L[key], cout, `${name}.${key}`, fail);
  let bound = 0;
  for (let o = 0; o < cout; o++) {
    let sum = Math.abs(bq[o]);
    for (let j = 0; j < k; j++) sum += CODE * Math.abs(wq[o * k + j]);
    if (sum >= BOUND) fail(`${name} channel ${o}: |bq| + 127 sum|wq| reaches ${sum}, not under 2^24: f32 would round a partial sum`);
    bound = Math.max(bound, sum);
  }
  return { wq, bq, sw, mul, bound };
}

// ai: The float layer after the codes (the proposer's head, a classifier's fc2) checked: w of shape and b of
// ai: shape[0], every value finite. Returns { w, b } as Float64Arrays, w [shape[0]][the rest].
export function readFloat(L, name, shape, fail) {
  const n = shape.reduce((a, b) => a * b, 1);
  if (L?.name !== name || "wq" in L || JSON.stringify(L.shape) !== JSON.stringify(shape)) fail(`${name}: float w and b of shape ${JSON.stringify(shape)} expected`);
  if (!Array.isArray(L.w) || L.w.length !== n || !Array.isArray(L.b) || L.b.length !== shape[0] || ![...L.w, ...L.b].every(Number.isFinite)) fail(`${name}: w of shape ${JSON.stringify(shape)} and b of ${shape[0]} finite values expected`);
  return { w: Float64Array.from(L.w), b: Float64Array.from(L.b) };
}

// ai: A reference file's bytes (PQ81, CQ81) read in order from after its magic: u32(), bytes(Kind, n) for a u8 or i8
// ai: array padded to 4, f32(n), and end(), which refuses bytes left over.
export function refReader(bytes, magic) {
  // ai: a plain Uint8Array view, since a node Buffer's slice() shares memory where bytes() needs a copy
  const u8 = ArrayBuffer.isView(bytes) ? new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== magic) throw new Error(`q8: not a ${magic} reference`);
  const r = {
    o: 4,
    u32: () => { const v = dv.getUint32(r.o, true); r.o += 4; return v; },
    bytes: (Kind, n) => { const a = new Kind(u8.slice(r.o, r.o + n).buffer); r.o += n + ((4 - (n % 4)) % 4); return a; },
    f32: (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = dv.getFloat32(r.o + 4 * i, true); r.o += 4 * n; return a; },
    end: () => { if (r.o !== u8.byteLength) throw new Error(`q8: ${u8.byteLength - r.o} bytes past the ${magic} reference's last array`); },
  };
  return r;
}

// ai: ---- the proposer ----

// ai: A q8 proposer file checked and parsed, or an Error: readHeader, the spec and norm, every conv through readLayer
// ai: and the head through readFloat. Returns { bits, qmax, inv, scale, window, eps, halo, convs: [{ name, cin, cout,
// ai: dil, out ("q8" | "f32"), wq (Int8Array, [out][in][ky][kx]), bq (Int32Array), sw, mul (Float32Array: m for a q8
// ai: output, s for the float one) }], head: { cin, w (Float64Array, [2][cin]), b (Float64Array, 2) } }.
export function readQuant(json) {
  const fail = (why) => { throw new Error(`q8: ${why}`); };
  const { q, qm } = readHeader(json, fail), a = json.arch;
  if (json.spec !== SPEC || a.norm !== "pixel") fail(`spec ${json.spec}, norm ${a.norm}: only ${SPEC} (proposer v2, norm pixel) is quantised`);
  const arch = a.layers ?? [], layers = json.layers ?? [], n = arch.length;
  if (n < 2 || layers.length !== n + 1) fail(`${layers.length} layers for ${n} convs and a head`);
  const want = [...arch.map((_, i) => [`conv${i + 1}`, "q8", i < n - 1 ? "q8" : "f32"]), ["head", "f32", "f32"]];
  checkLayers(q, want, fail);
  const convs = [];
  let cin = 1;
  for (let i = 0; i < n; i++) {
    const [name, , out] = want[i], cout = arch[i].out, dil = arch[i].dil;
    if (layers[i].dil !== dil) fail(`${name}: dil ${layers[i].dil}, expected ${dil}`);
    const { wq, bq, sw, mul } = readLayer(layers[i], name, [cout, cin, 3, 3], qm, out, fail);
    convs.push({ name, cin, cout, dil, out, wq, bq, sw, mul });
    cin = cout;
  }
  const { w, b } = readFloat(layers[n], "head", [2, cin, 1, 1], fail);
  return {
    bits: q.bits, qmax: qm, inv: q.input.inv, scale: q.input.scale, window: a.window, eps: a.eps,
    halo: 1 + arch.reduce((s, l) => s + l.dil, 0), convs, head: { cin, w, b },
  };
}

// ai: Xq from an f32 standardised input X (a level, row major, or a classifier's patch): clamp(roundEven(f32(X * inv)),
// ai: -qmax, qmax).
export function quantiseInput(q, X) {
  const xq = new Int8Array(X.length);
  for (let i = 0; i < X.length; i++) {
    const v = roundEven(Math.fround(Math.fround(X[i]) * q.inv));
    xq[i] = v < -q.qmax ? -q.qmax : v > q.qmax ? q.qmax : v;
  }
  return xq;
}

// ai: The integer net on one level's Xq (w x h), read clamped outside the level as the kernel reads X. Returns
// ai: { xq, codes, a1, a2, a3, logit, s }: codes the q8 planes (Int8Array, [c][y][x] over level pixels -b .. w - 1 + b,
// ai: b = halo less the dilations so far: 5 for a1, 3 for a2), a3 the last conv's f32 output and logit, s the head in
// ai: f64 in scripts/gpu/cnn/q8.py's head64 order, all three over level pixels -1 .. w by -1 .. h.
export function forwardCodes(q, xq, w, h) {
  const halo = q.halo;
  let W = w + 2 * halo, H = h + 2 * halo, C = 1;
  let a = new Int32Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) a[j * W + i] = xq[Math.min(h - 1, Math.max(0, j - halo)) * w + Math.min(w - 1, Math.max(0, i - halo))];
  const codes = [];
  let a3 = null;
  for (const L of q.convs) {
    const d = L.dil, O = L.cout, OW = W - 2 * d, OH = H - 2 * d, q8out = L.out === "q8";
    const y = q8out ? new Int8Array(O * OW * OH) : new Float32Array(O * OW * OH);
    for (let oc = 0; oc < O; oc++) for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
      let acc = 0;
      for (let ic = 0; ic < C; ic++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) acc += L.wq[((oc * C + ic) * 3 + ky) * 3 + kx] * a[(ic * H + oy + ky * d) * W + ox + kx * d];
      y[(oc * OH + oy) * OW + ox] = q8out ? requant(acc, L.bq[oc], L.mul[oc], q.qmax) : dequant(acc, L.bq[oc], L.mul[oc]);
    }
    if (q8out) codes.push(y); else a3 = y;
    a = y; W = OW; H = OH; C = O;
  }
  const N = W * H, logit = new Float64Array(N), s = new Float64Array(N), hw = q.head.w, hb = q.head.b;
  for (let i = 0; i < N; i++) {
    let z = hb[0], t = hb[1];
    for (let c = 0; c < C; c++) { z = z + hw[c] * a3[c * N + i]; t = t + hw[C + c] * a3[c * N + i]; }
    logit[i] = z; s[i] = t;
  }
  return { xq, codes, a1: codes[0], a2: codes[1], a3, logit, s };
}

const parsed = new WeakMap();

// ai: The integer net on one level given its f32 standardised map X (w x h; reference_v2.mjs standardiseLevel
// ai: computes it): forwardCodes on quantiseInput(X). json is a q8 weights file, checked once by readQuant.
export function forwardLevelQ8(json, X, w, h) {
  let q = parsed.get(json);
  if (!q) { q = readQuant(json); parsed.set(json, q); }
  return forwardCodes(q, quantiseInput(q, X), w, h);
}

// ai: A PQ81 reference (scripts/gpu/cnn/q8.py write_reference) from its bytes: { K, halo, planes: [{ ch, border }], levels: [{ w, h,
// ai: px, x (Float32Array), xq, codes (Int8Array each), logit, s (Float32Array) }] }.
export function readReferenceQ8(bytes) {
  const r = refReader(bytes, "PQ81"), count = r.u32(), K = r.u32(), halo = r.u32(), P = r.u32(), planes = [];
  for (let i = 0; i < P; i++) planes.push({ ch: r.u32(), border: r.u32() });
  const levels = [];
  for (let k = 0; k < count; k++) {
    const w = r.u32(), h = r.u32();
    const px = r.bytes(Uint8Array, w * h), x = r.f32(w * h), xq = r.bytes(Int8Array, w * h);
    const codes = planes.map(({ ch, border }) => r.bytes(Int8Array, ch * (w + 2 * border) * (h + 2 * border)));
    const maps = r.f32(2 * (w + 2) * (h + 2));
    levels.push({ w, h, px, x, xq, codes, logit: maps.subarray(0, (w + 2) * (h + 2)), s: maps.subarray((w + 2) * (h + 2)) });
  }
  r.end();
  return { K, halo, planes, levels };
}

async function main(wf, rf) {
  const { readFileSync } = await import("node:fs");
  const { standardiseLevel } = await import("./proposer/reference_v2.mjs");
  const { readNet } = await import("./netfile.mjs");
  const json = readNet(readFileSync(wf)), q = readQuant(json), ref = readReferenceQ8(readFileSync(rf));
  const planes = q.convs.slice(0, -1).map((L, i) => ({ ch: L.cout, border: q.halo - q.convs.slice(0, i + 1).reduce((s, c) => s + c.dil, 0) }));
  if (ref.K !== q.window || ref.halo !== q.halo || JSON.stringify(ref.planes) !== JSON.stringify(planes)) throw new Error(`${rf}: K ${ref.K} halo ${ref.halo} planes ${JSON.stringify(ref.planes)} are not ${wf}'s`);
  let codes = 0, bad = 0, wo = 0, wx = 0;
  const same = (u, v) => { codes += u.length; for (let i = 0; i < u.length; i++) if (u[i] !== v[i]) bad++; };
  for (const L of ref.levels) {
    const x = standardiseLevel(L.px, L.w, L.h, q.window, q.eps);
    for (let i = 0; i < x.length; i++) wx = Math.max(wx, Math.abs(x[i] - L.x[i]));
    const r = forwardLevelQ8(json, L.x, L.w, L.h);
    same(r.xq, L.xq);
    r.codes.forEach((c, k) => same(c, L.codes[k]));
    // ai: to f32 first, as stored: the same f64 sum on both sides then differs by nothing
    for (let i = 0; i < r.logit.length; i++) wo = Math.max(wo, Math.abs(Math.fround(r.logit[i]) - L.logit[i]), Math.abs(Math.fround(r.s[i]) - L.s[i]));
  }
  const ok = bad === 0 && wo < 1e-5 && wx < 1e-5;
  console.log(`q8.mjs against ${rf}: ${ref.levels.length} levels, ${q.bits} bits; ${bad} of ${codes} codes differ; logit and s max |diff| ${wo.toExponential(2)}; X against reference_v2.mjs ${wx.toExponential(2)} ${ok ? "PASS" : "FAIL"}`);
  return ok;
}

if (typeof process !== "undefined" && process.argv?.[1]) {
  const { pathToFileURL } = await import("node:url");
  if (import.meta.url === pathToFileURL(process.argv[1]).href) {
    const [wf, rf] = process.argv.slice(2);
    if (!wf || !rf) { console.error("usage: node gpu/cnn/q8.mjs <weights.safetensors> <reference.bin> (a PQ81 reference the trainer wrote; the lab's, not published)"); process.exit(2); }
    process.exit((await main(wf, rf)) ? 0 : 1);
  }
}
