// ai: The int8 contract for the two classifiers (cnn-v1, cnn-v3), scheme "int8-v1", in JS: scripts/gpu/cnn/q8c.py's twin (the
// ai: definition, the file and CQ81 are in scripts/gpu/cnn/q8c.py's header), for the int8 GEMM's packer, its JS reference forward and
// ai: the kernel checks. The arithmetic, quantiseInput (Xq) and the readers' checks are q8.mjs's; this file is the
// ai: classifiers' layer walk and the CQ81 layout. No node import at the top, so a page or worker can import it.
// ai:
// ai: The interface, in the order a kernel check uses it:
// ai:   readQuantC(doc)            a q8 classifier document (netfile.mjs readNet) checked and parsed, or an Error: {
// ai:                              bits, qmax, inv, scale, layers: [{ name, type ("conv" | "fc"), stride, cin, cout,
// ai:                              side (input side), sideOut, k (inputs an output: cin * 9, or fc1's cin * side *
// ai:                              side), out ("q8" for a conv, "f32" for fc1), wq (Int8Array [cout][k], k in torch
// ai:                              order: (c, ky, kx) for a conv, (c, y, x) for fc1), bq (Int32Array), sw, mul
// ai:                              (Float32Array: m for a conv, s for fc1), bound }], fc2: { cin, w (Float64Array
// ai:                              [5][cin]), b (Float64Array, 5) } }. A float file (no arch.quant) is refused.
// ai:   standardisePatch(px)       a u8 patch (1,024 samples) to x, f64 statistics then f32 (the CQ81 reference's x;
// ai:                              a kernel computes its own x in f32, and its Xq may differ by 1 where x * inv sits
// ai:                              within rounding of a half)
// ai:   quantiseInput(q, x)        (q8.mjs) Xq = clamp(roundEven(f32(x * inv)), -qmax, qmax), Int8Array(1024)
// ai:   forwardCodesC(q, xq)       the integer net on one patch's Xq: { xq, codes (Int8Array a conv, [c][y][x]), a
// ai:                              (Float32Array, fc1's float output), out (Float64Array, the five outputs, fc2 in
// ai:                              f64 in scripts/gpu/cnn/q8c.py fc64's order) }
// ai:   forwardPatchQ8(doc, x)     forwardCodesC on quantiseInput(x), the document parsed once
// ai:   readReferenceCQ81(bytes)   a CQ81 file: { count, side, planes: [{ ch, side }], F, O, patches (Uint8Array),
// ai:                              x (Float32Array), xq (Int8Array), codes (Int8Array a plane), a, out (Float32Array) },
// ai:                              each array over every patch in turn
// ai: Accumulators are exact integers: every |bq| + 127 * sum|wq| is under 2^24 (readQuantC refuses a file where it is
// ai: not), so an i32 GEMM with any K-split reduces to the same acc, and requant rounds it once.
// ai:   node gpu/cnn/q8c.mjs <weights.safetensors> <reference.bin>
// ai: checks this file against a CQ81 reference the trainer wrote (the lab's, not published): every code exact, a
// ai: and the outputs within 1e-5, and the reference's x within 1e-5 of standardisePatch.

import { requant, dequant, quantiseInput, readHeader, checkLayers, readLayer, readFloat, refReader } from "./q8.mjs";

export const SPEC = "cnn-v1-q8", PATCH = 32;
// ai: scripts/gpu/cnn/train.py STD_EPS: x = (v - mean) / (std + STD_EPS)
export const STD_EPS = 0.02;

// ai: [name, type, stride, out] of an arch's layers: convs, then exactly fc1 and fc2 of 5 outputs (scripts/gpu/cnn/q8c.py layers_of)
function layersOf(arch) {
  const out = [];
  let nconv = 0, nfc = 0;
  for (const l of arch?.layers ?? []) {
    if (l.type === "conv" && !nfc) out.push([`conv${++nconv}`, "conv", l.stride, l.out]);
    else if (l.type === "fc") out.push([`fc${++nfc}`, "fc", 1, l.out]);
    else throw new Error(`q8c: layer ${out.length}: ${JSON.stringify(l)}: int8-v1 runs convs, then fc layers`);
  }
  if (!nconv || nfc !== 2 || out[out.length - 1][3] !== 5) throw new Error(`q8c: ${arch?.name}: int8-v1 runs one or more convs, then exactly fc1 and fc2 of 5 outputs`);
  return out;
}

export function readQuantC(doc) {
  const fail = (why) => { throw new Error(`q8c: ${why}`); };
  const { q, qm } = readHeader(doc, fail);
  if (doc.spec !== SPEC) fail(`spec ${doc.spec}: a q8 classifier is ${SPEC}`);
  const plan = layersOf(doc.arch), layers = doc.layers ?? [];
  if (layers.length !== plan.length) fail(`${layers.length} layers for an arch of ${plan.length}`);
  const last = plan[plan.length - 1][0];
  checkLayers(q, [...plan.slice(0, -1).map(([n, t]) => [n, "q8", t === "conv" ? "q8" : "f32"]), [last, "f32", "f32"]], fail);
  const out = [];
  let C = 1, S = PATCH;
  plan.slice(0, -1).forEach(([name, type, stride, cout], i) => {
    const conv = type === "conv", k = conv ? C * 9 : C * S * S;
    const { wq, bq, sw, mul, bound } = readLayer(layers[i], name, conv ? [cout, C, 3, 3] : [cout, k], qm, conv ? "q8" : "f32", fail);
    const sideOut = conv ? Math.floor((S - 1) / stride) + 1 : 1;
    out.push({ name, type, stride, cin: C, cout, side: S, sideOut, k, out: conv ? "q8" : "f32", wq, bq, sw, mul, bound });
    C = cout; S = sideOut;
  });
  const { w, b } = readFloat(layers[layers.length - 1], last, [5, C], fail);
  return { bits: q.bits, qmax: qm, inv: q.input.inv, scale: q.input.scale, layers: out, fc2: { cin: C, w, b } };
}

export function standardisePatch(px) {
  const n = PATCH * PATCH;
  let s = 0;
  for (let i = 0; i < n; i++) s += px[i] / 255;
  const mean = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = px[i] / 255 - mean; v += d * d; }
  const den = Math.sqrt(v / n) + STD_EPS, x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (px[i] / 255 - mean) / den;
  return x;
}

export function forwardCodesC(q, xq) {
  let h = xq;
  const codes = [];
  let a = null;
  for (const L of q.layers) {
    if (L.type === "conv") {
      const { cin: C, cout: O, side: S, sideOut: SO, stride: st, k } = L, y = new Int8Array(O * SO * SO);
      for (let o = 0; o < O; o++) for (let oy = 0; oy < SO; oy++) for (let ox = 0; ox < SO; ox++) {
        let acc = 0;
        for (let c = 0; c < C; c++) for (let ky = 0; ky < 3; ky++) {
          const iy = oy * st + ky - 1;
          if (iy < 0 || iy >= S) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = ox * st + kx - 1;
            if (ix >= 0 && ix < S) acc += L.wq[o * k + c * 9 + ky * 3 + kx] * h[(c * S + iy) * S + ix];
          }
        }
        y[(o * SO + oy) * SO + ox] = requant(acc, L.bq[o], L.mul[o], q.qmax);
      }
      codes.push(y);
      h = y;
    } else {
      a = new Float32Array(L.cout);
      for (let o = 0; o < L.cout; o++) {
        let acc = 0;
        for (let j = 0; j < L.k; j++) acc += L.wq[o * L.k + j] * h[j];
        a[o] = dequant(acc, L.bq[o], L.mul[o]);
      }
    }
  }
  const { cin, w, b } = q.fc2, out = new Float64Array(5);
  for (let o = 0; o < 5; o++) {
    let z = b[o];
    for (let c = 0; c < cin; c++) z = z + w[o * cin + c] * a[c];
    out[o] = z;
  }
  return { xq, codes, a, out };
}

const parsed = new WeakMap();

export function forwardPatchQ8(doc, x) {
  let q = parsed.get(doc);
  if (!q) { q = readQuantC(doc); parsed.set(doc, q); }
  return forwardCodesC(q, quantiseInput(q, x));
}

export function readReferenceCQ81(bytes) {
  const r = refReader(bytes, "CQ81"), count = r.u32(), side = r.u32(), P = r.u32(), planes = [];
  for (let i = 0; i < P; i++) planes.push({ ch: r.u32(), side: r.u32() });
  const F = r.u32(), O = r.u32(), n = count * side * side;
  const patches = r.bytes(Uint8Array, n), x = r.f32(n), xq = r.bytes(Int8Array, n);
  const codes = planes.map(({ ch, side: s }) => r.bytes(Int8Array, count * ch * s * s));
  const a = r.f32(count * F), out = r.f32(count * O);
  r.end();
  return { count, side, planes, F, O, patches, x, xq, codes, a, out };
}

async function main(wf, rf) {
  const { readFileSync } = await import("node:fs");
  const { readNet } = await import("./netfile.mjs");
  const doc = readNet(readFileSync(wf)), q = readQuantC(doc), ref = readReferenceCQ81(readFileSync(rf));
  const planes = q.layers.filter((L) => L.type === "conv").map((L) => ({ ch: L.cout, side: L.sideOut }));
  if (ref.side !== PATCH || JSON.stringify(ref.planes) !== JSON.stringify(planes) || ref.F !== q.fc2.cin || ref.O !== 5) throw new Error(`${rf}: planes ${JSON.stringify(ref.planes)} F ${ref.F} are not ${wf}'s`);
  const N = PATCH * PATCH;
  let codes = 0, bad = 0, wa = 0, wo = 0, wx = 0;
  for (let p = 0; p < ref.count; p++) {
    const x = ref.x.subarray(p * N, (p + 1) * N), sx = standardisePatch(ref.patches.subarray(p * N, (p + 1) * N));
    for (let i = 0; i < N; i++) wx = Math.max(wx, Math.abs(sx[i] - x[i]));
    const r = forwardCodesC(q, quantiseInput(q, x));
    const same = (u, v) => { codes += u.length; for (let i = 0; i < u.length; i++) if (u[i] !== v[i]) bad++; };
    same(r.xq, ref.xq.subarray(p * N, (p + 1) * N));
    r.codes.forEach((c, k) => same(c, ref.codes[k].subarray(p * c.length, (p + 1) * c.length)));
    for (let i = 0; i < ref.F; i++) wa = Math.max(wa, Math.abs(r.a[i] - ref.a[p * ref.F + i]));
    // ai: to f32 first, as stored: the same f64 sum on both sides then differs by nothing
    for (let i = 0; i < 5; i++) wo = Math.max(wo, Math.abs(Math.fround(r.out[i]) - ref.out[p * 5 + i]));
  }
  const ok = bad === 0 && wa < 1e-5 && wo < 1e-5 && wx < 1e-5;
  console.log(`q8c.mjs against ${rf}: ${ref.count} patches, ${q.bits} bits; ${bad} of ${codes} codes differ; fc1 a max |diff| ${wa.toExponential(2)}; outputs ${wo.toExponential(2)}; x against standardisePatch ${wx.toExponential(2)} ${ok ? "PASS" : "FAIL"}`);
  return ok;
}

if (typeof process !== "undefined" && process.argv?.[1]) {
  const { pathToFileURL } = await import("node:url");
  if (import.meta.url === pathToFileURL(process.argv[1]).href) {
    const [wf, rf] = process.argv.slice(2);
    if (!wf || !rf) { console.error("usage: node gpu/cnn/q8c.mjs <weights.safetensors> <reference.bin> (a CQ81 reference the trainer wrote; the lab's, not published)"); process.exit(2); }
    process.exit((await main(wf, rf)) ? 0 : 1);
  }
}
