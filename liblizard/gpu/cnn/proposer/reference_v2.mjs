// Version 2 of the proposer (scripts/gpu/cnn/proposer/README.md "Version 2") in plain JS on a whole level, in f64: what a kernel is checked
// against, and, run as a script, this file checked against the trainer's reference file.
//   node liblizard/gpu/cnn/proposer/reference_v2.mjs <weights.safetensors> <reference.bin>
// ai:   both named (a default went stale at two installs, and the references are the lab's, under research/): the
// ai:   installed pair is liblizard/gpu/cnn/proposer/weights2.safetensors and research/results/gpu/prop_blown/pb-s1.ref
// ai:   (gpu/bank_fcn2.mjs REFERENCE)
import { readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { readNet } from "../netfile.mjs";

// The standardised map of a level (px: luma 0..255 row major, w x h): each pixel over the K x K box centred on it,
// the level's edge replicated into the box, population std. Returns a Float64Array of w x h.
export function standardiseLevel(px, w, h, K, eps) {
  const R = (K - 1) / 2, W2 = w + 2 * R, n = K * K;
  const at = (x, y) => px[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))] / 255;
  // Column sums over the box's rows, for every column the boxes reach (-R .. w - 1 + R, clamped when read).
  const c1 = new Float64Array(h * W2), c2 = new Float64Array(h * W2);
  for (let y = 0; y < h; y++) for (let i = 0; i < W2; i++) {
    let a = 0, b = 0;
    for (let d = -R; d <= R; d++) { const v = at(i - R, y + d); a += v; b += v * v; }
    c1[y * W2 + i] = a; c2[y * W2 + i] = b;
  }
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0, b = 0;
    for (let i = x; i < x + K; i++) { a += c1[y * W2 + i]; b += c2[y * W2 + i]; }
    const mean = a / n, sd = Math.sqrt(Math.max(b / n - mean * mean, 0));
    out[y * w + x] = (px[y * w + x] / 255 - mean) / (sd + eps);
  }
  return out;
}

// The net on a whole level: { x, z, s } with x the standardised map (w x h) and z (logit), s (log module size) at
// level pixels -1 .. w by -1 .. h, (w + 2) x (h + 2) row major. The convs are valid over the map read clamped.
export function forwardLevel(json, px, w, h) {
  const arch = json.arch, halo = 1 + arch.layers.reduce((a, l) => a + l.dil, 0);
  const x = standardiseLevel(px, w, h, arch.window, arch.eps);
  let W = w + 2 * halo, H = h + 2 * halo, C = 1;
  let a = new Float64Array(W * H);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) a[j * W + i] = x[Math.min(h - 1, Math.max(0, j - halo)) * w + Math.min(w - 1, Math.max(0, i - halo))];
  for (let k = 0; k < arch.layers.length; k++) {
    const L = json.layers[k], d = L.dil, O = L.shape[0], OW = W - 2 * d, OH = H - 2 * d, y = new Float64Array(O * OW * OH);
    for (let oc = 0; oc < O; oc++) for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
      let acc = L.b[oc];
      for (let ic = 0; ic < C; ic++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) acc += L.w[((oc * C + ic) * 3 + ky) * 3 + kx] * a[(ic * H + oy + ky * d) * W + ox + kx * d];
      y[(oc * OH + oy) * OW + ox] = Math.max(acc, 0);
    }
    a = y; W = OW; H = OH; C = O;
  }
  const hd = json.layers[arch.layers.length], z = new Float64Array(W * H), s = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    let az = hd.b[0], as = hd.b[1];
    for (let c = 0; c < C; c++) { az += hd.w[c] * a[c * W * H + i]; as += hd.w[C + c] * a[c * W * H + i]; }
    z[i] = az; s[i] = as;
  }
  return { x, z, s };
}

// The trainer's reference file (scripts/gpu/cnn/proposer/train.py write_reference_pixel): u32 count, K, halo; per level u32 w, h, the
// luma (u8, padded to 4 bytes), the standardised map (f32 w x h), then logit and s (f32, 2 x (h + 2) x (w + 2)).
export function readReference(buf) {
  const count = buf.readUInt32LE(0), K = buf.readUInt32LE(4), halo = buf.readUInt32LE(8), levels = [];
  let o = 12;
  const f32 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = buf.readFloatLE(o + 4 * i); o += 4 * n; return a; };
  for (let k = 0; k < count; k++) {
    const w = buf.readUInt32LE(o), h = buf.readUInt32LE(o + 4);
    o += 8;
    const px = new Uint8Array(buf.subarray(o, o + w * h));
    o += w * h + ((4 - ((w * h) % 4)) % 4);
    const x = f32(w * h), maps = f32(2 * (w + 2) * (h + 2));
    levels.push({ w, h, px, x, z: maps.subarray(0, (w + 2) * (h + 2)), s: maps.subarray((w + 2) * (h + 2)) });
  }
  return { K, halo, levels };
}

if ((() => { try { return fileURLToPath(import.meta.url) === realpathSync(process.argv[1]); } catch { return false; } })()) {
  const [wf, rf] = process.argv.slice(2);
  if (!wf || !rf) { console.error("usage: node reference_v2.mjs <weights.safetensors> <reference.bin> (the installed pair: liblizard/gpu/cnn/proposer/weights2.safetensors research/results/gpu/prop_blown/pb-s1.ref)"); process.exit(2); }
  const json = readNet(readFileSync(wf)), ref = readReference(readFileSync(rf));
  if (json.arch.norm !== "pixel" || json.arch.window !== ref.K) throw new Error(`${wf} is not version 2 with the reference's K ${ref.K}`);
  let wx = 0, wm = 0;
  for (const L of ref.levels) {
    const { x, z, s } = forwardLevel(json, L.px, L.w, L.h);
    for (let i = 0; i < x.length; i++) wx = Math.max(wx, Math.abs(x[i] - L.x[i]));
    for (let i = 0; i < z.length; i++) wm = Math.max(wm, Math.abs(z[i] - L.z[i]), Math.abs(s[i] - L.s[i]));
  }
  const ok = wx < 1e-5 && wm < 1e-4;
  console.log(`JS v2 forward against ${rf}: ${ref.levels.length} levels, K ${ref.K}; max |diff| standardised ${wx.toExponential(2)}, maps ${wm.toExponential(2)} ${ok ? "PASS" : "FAIL"}`);
  process.exit(ok ? 0 : 1);
}
