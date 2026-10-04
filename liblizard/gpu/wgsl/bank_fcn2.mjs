// F2 as the trained proposer, version 2 (scripts/gpu/cnn/proposer/README.md "Version 2"), to the contract in
// gpu/wgsl/bank.mjs. Each level pixel is standardised over the K x K box centred on it, so the map is a function
// of the level alone and a workgroup may compute any region of it: one workgroup makes a region of TX x TY 16 x 16
// tiles of one pyramid level (1 to 4) of one frame, and neighbouring tiles in a region share their halo instead of
// recomputing it (v1 recomputes a 16 x 16 tile's 26 x 26 conv1 input; a 32 x 32 region needs 42 x 42 for four tiles).
//
// Two ways to X, the standardised map (stage):
//   "x" then "net" (the default): the X pass computes each level pixel's X once, a 64 x 32 or 32 x 32 block a
//     workgroup, into a plane (halves where the activations are halves, else f32); the network kernel loads its X
//     window from the plane, clamped to the level (the spec's step 4), and runs the convolutions. On the iGPU this is
//     2.63 ms a 1080 frame against the fused kernel's 2.81: the fused one recomputes the box sums on every halo.
//   "all": one kernel; it computes X on its own window, the region plus HALO, from the luma on the region plus
//     HALO + R.
// The box sums, either way: of v - c and (v - c)^2, c the loaded window's mean (f32 sums of plain v cancel on bright,
// near-flat boxes: 1e-3 on the probability against 2e-5 shifted, README), separably, a row pass sliding along four
// columns an item and a column pass sliding down a run of rows. X = (v - mean) / (sd + eps).
//
// The network as the speed round's v1 kernel found it fastest on the iGPU (VALU bound): weights as literals, each tap
// a chain of fma into the accumulator, a pixel's eight channels one 16-byte element (two in f32), conv1 four output
// pixels a lane. f16: activations stored and multiplied as f16, the statistics, head outputs, sigmoid and parabola
// f32. packed: f32 arithmetic, activations stored as halves through pack2x16float (core WGSL), for a device whose
// ai: workgroup memory will not hold f32. Peaks (v1's test without its slack, peakAt), the parabola, sigma and each
// ai: tile's KEEP strongest are v1's (wgsl/bank_fcn.mjs), per 16 x 16 tile of the region.
//
// Workgroup memory is two pools of 16-byte elements, each holding one thing at a time between barriers:
//   A: the mean's partial sums, then the row sums, then conv1's output, then conv2's, then the peaks;
//   B: the luma window (v - c, four pixels an element), then X (four an element), then the map (probability and s,
//      two pixels an element).
// conv2's outputs wait in registers for a barrier and are then written over conv1's (staged), so the two never
// coexist: a 32 x 32 region takes 37.5 KB in f16 against 51 KB unstaged, and three workgroups share the iGPU's WGP
// where two did (2.63 ms against 3.12).
// ai: int8 (store "int8", a q8 file: the contract in cnn/q8.mjs, whose forwardCodes this computes bit for bit on the
// ai: kernel's own Xq): the plane form only, staged. The X pass computes X as above and stores
// ai: Xq = clamp(round(X * INV), -QMAX, QMAX), four pixels a u32 (pack4xI8), a quarter of f32's plane. The network's
// ai: pools are 8-byte elements: A holds a pixel's eight codes as one vec2<u32> (conv1's output, then conv2's, then
// ai: the peaks, two elements each), B the X window's Xq words (two an element) and then the map (one a pixel). conv1
// ai: makes each tap row's word from two X words by shifts, three taps and a zero weight: 24 dot4I8Packed a pixel.
// ai: conv2 and conv3 load a pixel's codes once a tap: two dot4I8Packed an output channel, 144 a pixel, each added to
// ai: its accumulator on its own, the weights packed u32 literals from the file's wq, a zero word left out. Every
// ai: accumulator starts at the file's bq; conv1 and conv2 requantise (rq), conv3 dequantises into the fused f32 head
// ai: (relu(f32(acc) * s), then the head as an fma chain from its bias, channel 0 first). NMS and selection are the
// ai: float forms'. A 32 x 32 region takes 23,376 bytes, so a 32 KB device runs it (with the 32 x 32 X block).
import { DIMS } from "./common.mjs";
import { KEEP } from "./bank.mjs";
import { KAPPA } from "../cnn/patch.mjs";
import { toHalf } from "../cnn/weights.mjs";
import { TAU, INT8 } from "../constants.mjs";   // ai: not ../decoder.mjs, which imports this module (gpu/constants.mjs)
import { readQuant, packI8x4 } from "../cnn/q8.mjs";

export { KEEP };
export const TILE = 16, LOCAL = 64;
// ai: The frame's counter slot for peaks past a tile's LOCAL, summed over its tiles; decoder.mjs COUNT names 0 to 3.
export const OVERFLOW = 4;
// Regions a workgroup can take, in tiles x by y, the largest first; the X pass's blocks in pixels, the larger first
// (38 KB of workgroup memory against 21: fewer window pixels a pixel, and its column pass fills the workgroup).
export const REGIONS = [[2, 2], [3, 1], [2, 1], [1, 1]];
export const XBLOCKS = [[64, 32], [32, 32]];
// ai: int8 takes the same regions. A 64 x 32 int8 region (42,848 bytes) was timed against 32 x 32 on the iGPU and
// ai: lost (bank 2.30 against 2.11 ms a frame at 1080, 8.09 against 7.81 at 4K, 2026-09-26), so it was deleted.
// Regions a network workgroup walks along its row (stage net); the host dispatches ceil(regions a row / WALK).
export const WALK = 4;
const WG = 256;
// ai: The int8 shaders name the language feature their built-ins (pack4xI8, dot4I8Packed) come from.
const INT8_REQUIRES = `requires ${INT8};\n`;

function roundHalf(x) {
  const h = toHalf(x), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// The arch of a version 2 weights file: its convolutions, the head, the box (K, R), the halo the convs reach
// (1 + sum of dilations: the map at a pixel reads X within halo - 1, and the non-maximum ring adds one).
export function fcn2Plan(json) {
  const arch = json.arch;
  if (arch?.norm !== "pixel") throw new Error(`proposer weights ${json.spec ?? ""}: not version 2 (arch.norm ${JSON.stringify(arch?.norm)}); a file without norm is v1 (bank fcn)`);
  const K = arch.window, R = (K - 1) / 2;
  if (!(Number.isInteger(K) && K % 2 === 1 && K >= 3)) throw new Error(`proposer weights: window ${K} is not an odd box`);
  const halo = 1 + arch.layers.reduce((a, l) => a + l.dil, 0);
  if (arch.halo !== undefined && arch.halo !== halo) throw new Error(`proposer weights: halo ${arch.halo}, the convs reach ${halo}`);
  if (arch.layers[0].dil > 2) throw new Error("fcn2 bank: conv1's dilation above 2 (conv1 reads a row of taps as two elements)");
  if (halo % 2) throw new Error("fcn2 bank: an odd halo (the X window's pairs of halves need an even origin)");
  let C = 1;
  const convs = arch.layers.map((l, k) => {
    const layer = json.layers[k];
    if (layer.name !== `conv${k + 1}` || layer.shape[0] !== l.out || layer.shape[1] !== C || layer.shape[2] !== 3 || l.out % 4) throw new Error(`proposer weights: ${layer.name} is ${JSON.stringify(layer.shape)}, the arch says ${l.out} x ${C} x 3 x 3 (channels in fours)`);
    const c = { name: layer.name, dil: l.dil, inC: C, outC: l.out, w: layer.w, b: layer.b };
    C = l.out;
    return c;
  });
  const head = json.layers[convs.length];
  if (!head || head.name !== "head" || head.shape[0] !== 2 || head.shape[1] !== C) throw new Error(`proposer weights: head ${JSON.stringify(head?.shape)} after ${C} channels`);
  const macs = convs.reduce((a, l) => a + l.inC * l.outC * 9, 0) + 2 * C;
  return { convs, head, K, R, halo, eps: arch.eps, nominal: arch.nominal, floor: arch.floor ?? TAU, macs };
}

// ai: A WGSL f32 literal: nine significant digits give an f32 back.
const lit9 = (v) => { let s = (+v).toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return s; };
// ai: int8 literals. An f32 the contract multiplies by must reach the kernel as that f32 exactly (checked here); a
// ai: weight word is four codes packed as pack4xI8 packs them.
function exactF32(v) {
  const s = lit9(v);
  if (Math.fround(v) !== v || Math.fround(+s) !== v) throw new Error(`fcn2 bank int8: ${v} is not an f32 the literal ${s} gives back`);
  return s;
}
const vec4Exact = (a) => `vec4f(${Array.from(a, exactF32).join(", ")})`;
const hexWord = (u) => `0x${u.toString(16).padStart(8, "0")}u`;

// ai: A float form refuses a q8 file (its convs hold codes, no w or b) and the int8 form a float one: cnn/q8.mjs
// ai: readQuant refuses a file without arch.quant, another version, codes out of range or past the 2^24 bound.
// ai: Returns the parsed codes for int8, else null.
function quantOf(json, int8) {
  if (!int8) {
    if (json.arch?.quant) throw new Error(`fcn2 bank: ${json.spec ?? "this file"} is an int8 file (arch.quant); only the int8 form runs it`);
    return null;
  }
  const q = readQuant(json);
  if (q.convs[0].dil !== 1) throw new Error("fcn2 bank int8: conv1's dilation is not 1 (its taps are three adjacent bytes of a word)");
  if (q.convs.some((L) => L.cout !== 8)) throw new Error("fcn2 bank int8: every conv has 8 output channels (a pixel's codes are one vec2<u32>)");
  return q;
}

// The box sums on an XW x XH window of X: the luma window VW x VH (four pixels an element, VE a row), the row sums
// (XE elements a row, two planes), and the column pass's run RR.
function statsLayout(P, XW, XH, colrun) {
  const XE = Math.ceil(XW / 4), NE = Math.ceil((P.K + 3) / 4), VW = 4 * XE + 2 * P.R, VH = XH + 2 * P.R, VE = XE - 1 + NE;
  const RR = colrun || Math.ceil((XH * XE) / WG);
  return { XW, XH, XE, NE, VW, VH, VE, RR, nA: Math.max(WG / 4, 2 * VH * XE), nB: VH * VE };
}

// Sizes of a stage's planes, in 16-byte elements, and the workgroup memory they need.
// ai: int8 (net stage, staged only): in 8-byte elements (vec2<u32>), a pixel's codes one, a map pixel one, a peak two,
// ai: the X window RS Xq words a row, two an element.
export function fcn2Layout(P, { TX = 1, TY = 1, halves = false, int8 = false, stage = "all", colrun = 0, XB = XBLOCKS[1], staged = true } = {}) {
  if (stage === "x") {
    const S = statsLayout(P, XB[0], XB[1], colrun || 2);
    return { ...S, stage, HALO: 0, R: P.R, nA: S.nA, nB: S.nB, bytes: 16 * (S.nA + S.nB) + 4 * 16 };
  }
  const RW = TILE * TX, RH = TILE * TY, HALO = P.halo, R = P.R;
  const XW = RW + 2 * HALO, XH = RH + 2 * HALO, XE = Math.ceil(XW / 4);
  let w = XW, h = XH;
  const convs = P.convs.map((L) => { const c = { ...L, inW: w, inH: h, outW: w - 2 * L.dil, outH: h - 2 * L.dil }; w = c.outW; h = c.outH; return c; });
  if (w !== RW + 2 || h !== RH + 2) throw new Error(`fcn2 bank: the convs leave ${w} x ${h}, the region and its ring are ${RW + 2} x ${RH + 2}`);
  const G1 = Math.ceil(convs[0].outW / 4), RS = G1 + 1;
  if (int8) {
    if (stage !== "net" || !staged) throw new Error("fcn2 bank int8: the network reads the plane (stage net), staged");
    const NT = TX * TY, MW = RW + 2, MH = RH + 2;
    const nA = Math.max(...convs.slice(0, -1).map((L) => L.outW * L.outH), 2 * NT * LOCAL), nB = Math.max(Math.ceil((XH * RS) / 2), MW * MH);
    return { stage, RW, RH, HALO, R, XW, XH, XE, G1, RS, NT, MW, MH, convs, nA, nB, mapPool: "B", staged, int8, bytes: 8 * (nA + nB) + 4 * NT };
  }
  const epp = (C) => (halves ? Math.ceil(C / 8) : C / 4);
  const NT = TX * TY, MW = RW + 2, MH = RH + 2;
  // staged: every conv writes A, conv1 straight over nothing, the later ones from registers after a barrier (over the
  // output they read), so conv1's and conv2's outputs never coexist; the map (two pixels an element) goes in B over X,
  // the peaks in A. Else conv k's output is in A for even k, B for odd, the map in the pool the last conv does not read.
  const act = convs.slice(0, -1).map((L) => L.outW * L.outH * epp(L.outC));
  const S = stage === "all" ? statsLayout(P, XW, XH, colrun) : { nA: 0, nB: 0 };
  let mapPool, nA, nB;
  if (staged) {
    mapPool = "B";
    nA = Math.max(S.nA, ...act, NT * LOCAL);
    nB = Math.max(S.nB, XH * RS, Math.ceil((MW * MH) / 2));
  } else {
    mapPool = (convs.length - 1) % 2 ? "B" : "A";
    nA = Math.max(S.nA, ...act.filter((_, k) => k % 2 === 0), mapPool === "A" ? MW * MH : 0, mapPool === "B" ? NT * LOCAL : 0);
    nB = Math.max(S.nB, XH * RS, ...act.filter((_, k) => k % 2 === 1), mapPool === "B" ? MW * MH : 0, mapPool === "A" ? NT * LOCAL : 0);
  }
  const bytes = 16 * (nA + nB) + (stage === "all" ? 4 * 16 : 0) + 4 * NT;
  return { ...(stage === "all" ? S : {}), stage, RW, RH, HALO, R, XW, XH, XE, G1, RS, NT, MW, MH, convs, epp, nA, nB, mapPool, staged, bytes };
}

// The forms a device can be given, best first: f16 when the decoder runs f16; else f32 storage, then halves stored.
// Within a storage, the largest region first; the X plane before the fused kernel, its larger X block first. A form's
// bytes are its larger kernel's. `force`: { packed, f32, region: "32x16", fused, plane, xblock: "32x32" } narrows it;
// force.walk sets the regions a network workgroup walks.
// ai: int8 (a q8 file, which no float form runs): store "int8", the plane form only; force.packed, f32 or fused
// ai: leave it no form.
export function fcn2Forms(json, { f16 = false, int8 = false, force = {} } = {}) {
  if (f16 && int8) throw new Error("fcn2 bank: f16 or int8, not both");
  quantOf(json, int8);
  const P = fcn2Plan(json);
  const regions = REGIONS.filter(([tx, ty]) => !force.region || force.region === `${TILE * tx}x${TILE * ty}`);
  const stores = int8 ? (force.packed || force.f32 ? [] : ["int8"]) : f16 ? ["f16"] : force.packed ? ["packed"] : force.f32 ? ["f32"] : ["f32", "packed"];
  const planes = int8 ? (force.fused ? [] : [true]) : force.fused ? [false] : force.plane ? [true] : [true, false];
  const out = [];
  const xblocks = XBLOCKS.filter((b) => !force.xblock || force.xblock === `${b[0]}x${b[1]}`);
  for (const store of stores) for (const [TX, TY] of regions) for (const xplane of planes) for (const XB of xplane ? xblocks : [null]) {
    const halves = store === "f16" || store === "packed", net = fcn2Layout(P, { TX, TY, halves, int8, stage: xplane ? "net" : "all" });
    const bytes = Math.max(net.bytes, xplane ? fcn2Layout(P, { stage: "x", XB }).bytes : 0);
    out.push({ store, TX, TY, xplane, XB, walk: xplane ? +(force.walk ?? WALK) : 1, f16: store === "f16", packed: store === "packed", int8, bytes, name: `${store} ${TILE * TX}x${TILE * TY}${xplane ? ` plane ${XB[0]}x${XB[1]}` : ""}` });
  }
  return out;
}

// ai: What a dump build writes a frame, in f32 elements, at a W x H capacity (the kernel's P.stride x P.rows): X (a
// ai: float form) or Xq (int8) on the level's pixels at 0; then, int8 only, each requantised conv's codes, [c][y][x]
// ai: over level pixels -border .. W - 1 + border (border = the halo less the dilations so far: 5, then 3); then the
// ai: head's logit and s over -1 .. W by -1 .. H. { DS (a frame's elements), planes: [{ off, ch, border }], logit, s }.
export function fcn2Dump(json, W, H, { int8 = false } = {}) {
  const P = fcn2Plan(json), planes = [];
  let off = W * H, b = P.halo;
  if (int8) for (const L of P.convs.slice(0, -1)) { b -= L.dil; planes.push({ off, ch: L.outC, border: b }); off += L.outC * (W + 2 * b) * (H + 2 * b); }
  const M = (W + 2) * (H + 2);
  return { DS: off + 2 * M, planes, logit: off, s: off + M };
}

// The WGSL of one stage. stage "x" is the X pass; "net" the network reading the plane; "all" the fused kernel.
// Bindings: x: lev, frames, plane, P. net: plane, frames, raw, counts, P. all: lev, frames, raw, counts, P.
// The plane holds frame f's level pixel (x, y) at (f rows + y) SP + x, SP the stride rounded up to even; halves as
// pairs in a u32 (pack2x16float), else f32.
// dump: a sixth binding (net, all), dbg, receiving X on the level's own pixels and the head's raw logit and s on level
// pixels -1 .. w by -1 .. h, frame f at f * (stride rows + 2 (stride + 2)(rows + 2)); for cnn/proposer/check2.mjs.
// probe (timing only, the output is wrong): "nostats" leaves out the box sums, "noconv" the convolutions, "noload" the
// loads of the luma window and of the X window.
// ai: int8: the int8 form (fcn2Forms; a q8 file). Its plane holds Xq, four pixels a u32 at a row stride rounded up to
// ai: four; its dump build writes Xq, then each requantised conv's codes, then logit and s, as fcn2Dump lays them out.
export function fcn2Source(json, { f16 = false, packed = false, int8 = false, TX = 2, TY = 2, stage = "all", dump = false, probe = "", colrun = 0, XB = XBLOCKS[1], staged = true, walk = WALK } = {}) {
  if (f16 && packed) throw new Error("fcn2 bank: f16 already stores halves; packed is for f32");
  if (int8 && (f16 || packed || stage === "all" || !staged)) throw new Error("fcn2 bank: int8 is a store of its own with the plane form only (stage x, then net), staged");
  const Q = quantOf(json, int8);
  const P = fcn2Plan(json), halves = f16 || packed, Lz = fcn2Layout(P, { TX, TY, halves, int8, stage, colrun, XB, staged });
  const { HALO, R, nA, nB, bytes } = Lz;
  const T = f16 ? "f16" : "f32", V4 = `vec4<${T}>`, K = P.K, KK = K * K;
  const f32 = lit9;
  const num = (v) => { const r = f16 ? roundHalf(v) : v; let s = r.toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return f16 ? `${s}h` : s; };
  const vec = (a) => `${V4}(${a.map(num).join(", ")})`;
  const stats = stage !== "net", net = stage !== "x", withDump = dump && net;

  // ---- The box sums (stages x and all): the window of v - c, the row pass, the column pass into registers.
  let statsCode = "", statsFns = "";
  if (stats) {
    const { XH, XE, NE, VH, VE, RR } = Lz;
    const NRUN = Math.ceil(XH / RR), NCI = XE * NRUN, NCL = Math.ceil(NCI / WG), sums = !probe.includes("nostats");
    // Lane li loads elements li, li + WG, ... of the window (four pixels an element, VE a row, the last one's padding
    // included, so c is the mean of VH x 4 VE pixels) into registers.
    const NV = VH * VE, NLD = Math.ceil(NV / WG), guard = (r) => ((r + 1) * WG > NV ? `if (q < ${NV}u) ` : "");
    const loads = Array.from({ length: NLD }, (_, r) => `  var v${r} = vec4f(0.0);
  { let q = li + ${r * WG}u; ${guard(r)}{
    let row = (f * P.rows + u32(clamp(yo0 - ${R} + i32(q / VE), 0, h - 1))) * P.stride;
    let xs = vec4<u32>(clamp(vec4i(xo0 - ${R} + 4 * i32(q % VE)) + vec4i(0, 1, 2, 3), vec4i(0), vec4i(w - 1)));
    v${r} = ${probe.includes("noload") ? "vec4f(xs) * 1e-3" : "vec4f(lev[row + xs.x], lev[row + xs.y], lev[row + xs.z], lev[row + xs.w])"};
    s0 += (v${r}.x + v${r}.y) + (v${r}.z + v${r}.w);
  } }`).join("\n");
    const puts = Array.from({ length: NLD }, (_, r) => `  { let q = li + ${r * WG}u; ${guard(r)}{ poolB[q] = bitcast<vec4<u32>>(v${r} - vec4f(c)); } }`).join("\n");
    // The row pass: window row j, X columns 4g .. 4g + 3, from window pixels 4g .. 4g + K + 2 (a sliding sum).
    const a = Array.from({ length: 4 * NE }, (_, n) => `b${n >> 2}.${"xyzw"[n & 3]}`);
    const sum = (fn) => Array.from({ length: K }, (_, n) => fn(n)).join(" + ");
    const rowPass = !sums ? "" : `  for (var it = li; it < ${VH * XE}u; it += ${WG}u) {
    let j = it / XE;
    let g = it % XE;
${Array.from({ length: NE }, (_, n) => `    let b${n} = bitcast<vec4f>(poolB[j * VE + g + ${n}u]);`).join("\n")}
${Array.from({ length: K + 3 }, (_, n) => `    let a${n} = ${a[n]};\n    let q${n} = a${n} * a${n};`).join("\n")}
    let s10 = ${sum((n) => `a${n}`)};
    let s20 = ${sum((n) => `q${n}`)};
${[1, 2, 3].map((k) => `    let s1${k} = s1${k - 1} + a${K + k - 1} - a${k - 1};\n    let s2${k} = s2${k - 1} + q${K + k - 1} - q${k - 1};`).join("\n")}
    poolA[j * XE + g] = bitcast<vec4<u32>>(vec4f(s10, s11, s12, s13));
    poolA[HOFF + j * XE + g] = bitcast<vec4<u32>>(vec4f(s20, s21, s22, s23));
  }
  workgroupBarrier();
`;
    // The column pass: an item is X columns 4g .. 4g + 3 on a run of RR rows, the box sums slid down the run (a row
    // clamped to the level's edge keeps its box). Its values stay in registers (xs<c>_<k>) for the caller to store.
    const item = (c) => {
      const lines = [`  {
    let ci = li + ${c * WG}u;
    if (ci < ${NCI}u) {
      let g = ci % XE;
      let j0 = (ci / XE) * ${RR}u;
      let x0 = xo0 + i32(g) * 4;
      let cx = vec4<u32>(clamp(vec4i(x0) + vec4i(0, 1, 2, 3), vec4i(0), vec4i(w - 1)) - vec4i(xo0));
      let inner = x0 >= 0 && x0 + 3 < w;
      var s1 = vec4f(0.0);
      var s2 = vec4f(0.0);
      var r0 = u32(clamp(yo0 + i32(j0), 0, h - 1) - yo0);${sums ? `
      for (var r = 0u; r < KBOX; r++) { let hh = hrow(r0 + r, g, cx, inner); s1 += hh.a; s2 += hh.b; }` : ""}
      xs${c}_0 = xval(s1, s2, r0 + ${R}u, cx);`];
      for (let k = 1; k < RR; k++) lines.push(`      if (j0 + ${k}u < ${XH}u) {
        let rn = u32(clamp(yo0 + i32(j0 + ${k}u), 0, h - 1) - yo0);
        if (rn != r0) {${sums ? ` let hi = hrow(r0 + KBOX, g, cx, inner); let lo = hrow(r0, g, cx, inner); s1 += hi.a - lo.a; s2 += hi.b - lo.b;` : ""} r0 = rn; }
        xs${c}_${k} = xval(s1, s2, r0 + ${R}u, cx);
      }`);
      lines.push("    }\n  }");
      return Array.from({ length: RR }, (_, k) => `  var xs${c}_${k} = vec4f(0.0);`).join("\n") + "\n" + lines.join("\n");
    };
    // Each value of the column pass handed to put(row, group, value).
    const each = (put) => Array.from({ length: NCL }, (_, c) => `  {
    let ci = li + ${c * WG}u;
    let g = ci % XE;
    let j0 = (ci / XE) * ${RR}u;
${Array.from({ length: RR }, (_, k) => `    if (ci < ${NCI}u && j0 + ${k}u < ${XH}u) { ${put(`j0 + ${k}u`, "g", `xs${c}_${k}`)} }`).join("\n")}
  }`).join("\n");
    statsCode = `  var s0 = 0.0;
${loads}
  // c, the window's mean: partial sums in two rounds of sixteen, every lane adding the last sixteen in one order.
  poolA[li >> 2u][li & 3u] = bitcast<u32>(s0);
  workgroupBarrier();
  if (li < 16u) {
    var t = 0.0;
    for (var k = 0u; k < 16u; k++) { t += bitcast<f32>(poolA[(li * 16u + k) >> 2u][(li * 16u + k) & 3u]); }
    cst[li] = t;
  }
  let gs = workgroupUniformLoad(&cst);
  var tc = 0.0;
  for (var k = 0u; k < 16u; k++) { tc += gs[k]; }
  let c = tc / ${f32(4 * NV)};
${puts}
  workgroupBarrier();
${rowPass}${Array.from({ length: NCL }, (_, c) => item(c)).join("\n")}
`;
    statsCode += stage === "x"
      ? each((j, g, v) => `planePut(${j}, ${g}, ${v}, xo0, yo0, w, h, f);`)
      : `  workgroupBarrier();\n${each((j, g, v) => `xPut(${j}, ${g}, ${v}, xo0, yo0, w, h, f);`)}\n  workgroupBarrier();\n`;
    statsFns = `const VE = ${VE}u;
const HOFF = ${VH * XE}u;
const KBOX = ${K}u;
var<workgroup> cst: array<f32, 16>;
fn vGet(j: u32, i: u32) -> f32 { return bitcast<f32>(poolB[j * VE + (i >> 2u)][i & 3u]); }
// The row sums of box row r for X columns 4g .. 4g + 3: one element each where no column is clamped, else a column at
// a time, each column's box being the one centred on its pixel clamped into the level (the spec's step 4).
struct HS { a: vec4f, b: vec4f }
fn hrow(r: u32, g: u32, cx: vec4<u32>, inner: bool) -> HS {
  if (inner) { return HS(bitcast<vec4f>(poolA[r * XE + g]), bitcast<vec4f>(poolA[HOFF + r * XE + g])); }
  var o = HS(vec4f(0.0), vec4f(0.0));
  for (var k = 0u; k < 4u; k++) {
    let e = r * XE + (cx[k] >> 2u);
    o.a[k] = bitcast<f32>(poolA[e][cx[k] & 3u]);
    o.b[k] = bitcast<f32>(poolA[HOFF + e][cx[k] & 3u]);
  }
  return o;
}
// X from a box's sums of v - c and (v - c)^2 and its centre's window row vy.
fn xval(s1: vec4f, s2: vec4f, vy: u32, cx: vec4<u32>) -> vec4f {
  let mu = s1 / ${f32(KK)};
  let sd = sqrt(max(s2 / ${f32(KK)} - mu * mu, vec4f(0.0)));
  let vc = vec4f(vGet(vy, cx.x + ${R}u), vGet(vy, cx.y + ${R}u), vGet(vy, cx.z + ${R}u), vGet(vy, cx.w + ${R}u));
  return (vc - mu) / (sd + vec4f(EPS));
}
`;
  }

  // The plane: frame f's row y at (f rows + y) SP, SP even so a pair of halves never straddles two rows.
  // ai: int8: SP a multiple of four, so a word of four Xq never straddles two rows.
  const planeFns = stage === "all" ? "" : int8 ? `fn planeRow(f: u32, y: i32) -> u32 { return (f * P.rows + u32(y)) * ((P.stride + 3u) & ~3u); }
// ai: the Xq byte of level pixel (x, y), as its bits
fn planeByte(f: u32, x: i32, y: i32) -> u32 { let i = planeRow(f, y) + u32(x); return (xpl[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }` : `fn planeRow(f: u32, y: i32) -> u32 { return (f * P.rows + u32(y)) * ((P.stride + 1u) & ~1u); }
${halves ? `fn planeAt(f: u32, x: i32, y: i32) -> f32 { let i = planeRow(f, y) + u32(x); return unpack2x16float(xpl[i >> 1u])[i & 1u]; }` : `fn planeAt(f: u32, x: i32, y: i32) -> f32 { return xpl[planeRow(f, y) + u32(x)]; }`}`;
  const planePut = stage !== "x" ? "" : int8 ? `// ai: Xq of X columns xo0 + 4g .. + 3 (xo0 a multiple of 4: one word) of block row j into the plane, on the level; past
// ai: its right edge the word repeats the edge's X.
fn planePut(j: u32, g: u32, xv: vec4f, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) {
  let x = xo0 + i32(g) * 4;
  let y = yo0 + i32(j);
  if (x >= w || y >= h) { return; }
  xpl[(planeRow(f, y) + u32(x)) >> 2u] = pack4xI8(vec4<i32>(clamp(round(xv * INV), vec4f(-QMAX), vec4f(QMAX))));
}` : `// X columns xo0 + 4g .. + 3 (xo0 a multiple of 4, so two whole pairs) of block row j into the plane, on the level.
fn planePut(j: u32, g: u32, xv: vec4f, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) {
  let x = xo0 + i32(g) * 4;
  let y = yo0 + i32(j);
  if (x >= w || y >= h) { return; }
  let i = planeRow(f, y) + u32(x);
${halves ? `  xpl[i >> 1u] = pack2x16float(xv.xy);
  if (x + 2 < w) { xpl[(i >> 1u) + 1u] = pack2x16float(xv.zw); }` : `  for (var k = 0; k < 4; k++) { if (x + k < w) { xpl[i + u32(k)] = xv[k]; } }`}
}`;

  // ---- The network (stages net and all).
  let netCode = "", netFns = "", Qn = null;
  if (net) {
    const { RW, RH, XW, XH, XE, G1, RS, NT, MW, MH, convs, epp, mapPool } = Lz;
    // A pixel's output channels as the elements that store it: halves two groups an element, f32 one.
    const half2 = (a) => (a ? (f16 ? `bitcast<vec2<u32>>(${a})` : `pk(${a})`) : "vec2<u32>(0u)");
    const packPixel = (accs) => {
      const outs = accs.map((a) => `max(${a}, ${V4}(0.0))`);
      return halves ? Array.from({ length: epp(outs.length * 4) }, (_, e) => `vec4<u32>(${half2(outs[2 * e])}, ${half2(outs[2 * e + 1])})`) : outs.map((x) => `bitcast<vec4<u32>>(${x})`);
    };
    const col = (L, ky, kx, gi, go, j) => vec(Array.from({ length: 4 }, (_, i) => L.w[(((go * 4 + i) * L.inC + gi * 4 + j) * 3 + ky) * 3 + kx]));
    const grp = (es, g) => {
      if (!halves) return `bitcast<vec4<f32>>(${es[g]})`;
      const h = `${es[g >> 1]}.${g & 1 ? "zw" : "xy"}`;
      return f16 ? `bitcast<vec4<f16>>(${h})` : `upk(${h})`;
    };
    const storePixel = (dst, p, accs) => { const es = packPixel(accs); return es.map((x, e) => `${dst}[${p} * ${es.length}u + ${e}u] = ${x};`).join(" "); };
    const pool = (k) => (Lz.staged || k % 2 === 0 ? "poolA" : "poolB");
    // conv1: four output pixels a lane, each tap row read as two elements of X (four pixels an element).
    const conv1 = (L) => {
      const OG = L.outC / 4, N = G1 * L.outH, lines = [];
      for (let j = 0; j < 4; j++) for (let go = 0; go < OG; go++) lines.push(`      var a${j}_${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`);
      for (let ky = 0; ky < 3; ky++) {
        lines.push(`      {\n        let r0 = bitcast<vec4f>(poolB[(y + ${ky * L.dil}u) * ${RS}u + gk]);\n        let r1 = bitcast<vec4f>(poolB[(y + ${ky * L.dil}u) * ${RS}u + gk + 1u]);`);
        lines.push(`        let cv = array<${T}, 8>(${T}(r0.x), ${T}(r0.y), ${T}(r0.z), ${T}(r0.w), ${T}(r1.x), ${T}(r1.y), ${T}(r1.z), ${T}(r1.w));`);
        for (let j = 0; j < 4; j++) for (let kx = 0; kx < 3; kx++) for (let go = 0; go < OG; go++)
          lines.push(`        a${j}_${go} = fma(${vec(Array.from({ length: 4 }, (_, i) => L.w[(go * 4 + i) * 9 + ky * 3 + kx]))}, ${V4}(cv[${j + kx * L.dil}]), a${j}_${go});`);
        lines.push("      }");
      }
      const st = Array.from({ length: 4 }, (_, j) => `      if (x0 + ${j}u < ${L.outW}u) { let p = y * ${L.outW}u + x0 + ${j}u; ${storePixel(pool(0), "p", Array.from({ length: OG }, (_, go) => `a${j}_${go}`))} }`);
      return `    for (var i = li; i < ${N}u; i += ${WG}u) {
      let gk = i % ${G1}u;
      let y = i / ${G1}u;
      let x0 = gk * 4u;
${lines.join("\n")}
${st.join("\n")}
    }
    workgroupBarrier();
`;
    };
    const conv = (L, k) => {
      const last = k === convs.length - 1, OG = L.outC / 4, IG = L.inC / 4, N = L.outW * L.outH, Ei = epp(L.inC), src = pool(k - 1);
      const taps = [];
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        const es = Array.from({ length: Ei }, (_, e) => `e${e}`);
        const reads = es.map((e, n) => `let ${e} = ${src}[q + ${n}u];`).join(" ");
        const gs = Array.from({ length: IG }, (_, gi) => `let i${gi} = ${grp(es, gi)};`).join(" ");
        const body = Array.from({ length: OG }, (_, go) => Array.from({ length: IG }, (_, gi) => Array.from({ length: 4 }, (_, j) => `        acc${go} = fma(${col(L, ky, kx, gi, go, j)}, ${V4}(i${gi}[${j}]), acc${go});`).join("\n")).join("\n")).join("\n");
        taps.push(`      { let q = ((y + ${ky * L.dil}u) * ${L.inW}u + x + ${kx * L.dil}u) * ${Ei}u; ${reads} ${gs}\n${body} }`);
      }
      let store;
      if (last) {
        // The head, fused. The map keeps the probability, so the non-maximum test reads it nine times without nine exps.
        const H = P.head, dot = (o) => Array.from({ length: OG }, (_, go) => `dot(${vec(H.w.slice(o * L.outC + go * 4, o * L.outC + go * 4 + 4))}, max(acc${go}, ${V4}(0.0)))`).join(" + ");
        store = `      let zl = ${f32(H.b[0])} + f32(${dot(0)});
      let sv = ${f32(H.b[1])} + f32(${dot(1)});
      mapPut(i, 1.0 / (1.0 + exp(-zl)), sv);${withDump ? `
      {
        // The raw head a pixel, once: the region's own pixels and the level's outer ring (-1 and w, -1 and h).
        let lx = rx0 - 1 + i32(x);
        let ly = ry0 - 1 + i32(y);
        let ox = (lx >= rx0 && lx < rx0 + ${RW}) || lx == -1 || lx == w;
        let oy = (ly >= ry0 && ly < ry0 + ${RH}) || ly == -1 || ly == h;
        if (ox && oy && lx >= -1 && lx <= w && ly >= -1 && ly <= h) {
          let o = dbgBase(f) + P.stride * P.rows + u32(ly + 1) * (P.stride + 2u) + u32(lx + 1);
          dbg[o] = zl;
          dbg[o + (P.stride + 2u) * (P.rows + 2u)] = sv;
        }
      }` : ""}`;
      } else if (Lz.staged) {
        // Staged: the pixel's elements into registers o<r>_<e> for round r, written over the input after a barrier.
        const NR = Math.ceil(N / WG), es = packPixel(Array.from({ length: OG }, (_, go) => `acc${go}`));
        const regs = Array.from({ length: NR }, (_, r) => es.map((_, e) => `o${k}_${r}_${e}`));
        return `${regs.flat().map((o) => `    var ${o} = vec4<u32>(0u);`).join("\n")}
    for (var r = 0u; r < ${NR}u; r++) {
      let i = li + r * ${WG}u;
      if (i < ${N}u) {
      let x = i % ${L.outW}u;
      let y = i / ${L.outW}u;
${Array.from({ length: OG }, (_, go) => `      var acc${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`).join("\n")}
${taps.join("\n")}
${es.map((x, e) => `      let st${e} = ${x};`).join("\n")}
${regs.map((rr, r) => `      if (r == ${r}u) { ${rr.map((o, e) => `${o} = st${e};`).join(" ")} }`).join("\n")}
      }
    }
    workgroupBarrier();
${regs.map((rr, r) => `    if (li + ${r * WG}u < ${N}u) { ${rr.map((o, e) => `${pool(k)}[(li + ${r * WG}u) * ${es.length}u + ${e}u] = ${o};`).join(" ")} }`).join("\n")}
    workgroupBarrier();
`;
      } else {
        store = `      ${storePixel(pool(k), "i", Array.from({ length: OG }, (_, go) => `acc${go}`))}`;
      }
      return `    for (var i = li; i < ${N}u; i += ${WG}u) {
      let x = i % ${L.outW}u;
      let y = i / ${L.outW}u;
${Array.from({ length: OG }, (_, go) => `      var acc${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`).join("\n")}
${taps.join("\n")}
${store}
    }
    workgroupBarrier();
`;
    };
    // ai: int8: the network's pieces (int8Net).
    Qn = int8 ? int8Net(Q, Lz, { withDump, probe }) : null;
    const layers = probe.includes("noconv") ? "" : Qn ? Qn.layers : conv1(convs[0]) + convs.slice(1).map((L, k) => conv(L, k + 1)).join("");
    // Stage net: the X window from the plane, four pixels an item, clamped to the level; a group of four wholly in the
    // level is two pairs (the window's origin rx0 - HALO is even), else each pixel is read at its clamped place.
    // Stage net: the X window from the plane, four pixels an item, clamped to the level; a group of four wholly in the
    // level is two pairs (the window's origin rx0 - HALO is even), else each pixel is read at its clamped place. A
    // workgroup walks WALK regions along the row and fetches the next region's window into registers (xp<r>) before
    // the current one's convolutions, so the loads' latency hides behind them.
    const NXI = XH * XE, NXL = Math.ceil(NXI / WG);
    const xfetchFn = stage !== "net" ? "" : Qn ? Qn.xfetch : `fn xfetch(it: u32, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) -> vec4f {
  let jr = it / XE;
  let g = it % XE;
  let x0 = xo0 + i32(g) * 4;
  let y = clamp(yo0 + i32(jr), 0, h - 1);
  var xv = vec4f(0.0);
  if (${probe.includes("noload") ? "false" : "x0 >= 0 && x0 + 3 < w"}) {
${halves ? `    let i = (planeRow(f, y) + u32(x0)) >> 1u;
    xv = vec4f(unpack2x16float(xpl[i]), unpack2x16float(xpl[i + 1u]));` : `    let i = planeRow(f, y) + u32(x0);
    xv = vec4f(xpl[i], xpl[i + 1u], xpl[i + 2u], xpl[i + 3u]);`}
  } else {
    for (var k = 0; k < 4; k++) { xv[k] = ${probe.includes("noload") ? "f32(x0 + k) * 1e-3" : "planeAt(f, clamp(x0 + k, 0, w - 1), y)"}; }
  }
  return xv;
}
`;
    const xfetch = (xo) => Array.from({ length: NXL }, (_, r) => `      if (li + ${r * WG}u < ${NXI}u) { xp${r} = xfetch(li + ${r * WG}u, ${xo}, yo0, w, h, f); }`).join("\n");
    const xload = stage !== "net" ? "" : `${Array.from({ length: NXL }, (_, r) => `    if (li + ${r * WG}u < ${NXI}u) { xPut((li + ${r * WG}u) / XE, (li + ${r * WG}u) % XE, xp${r}, xo0, yo0, w, h, f); }`).join("\n")}
    workgroupBarrier();
    if (rr + 1u < WALK && rx0 + RW < w) {
${xfetch("xo0 + RW")}
    }
`;
    const foundName = Lz.staged ? "poolA" : mapPool === "A" ? "poolB" : "poolA", mp = mapPool === "A" ? "poolA" : "poolB";
    // The map: probability and s a pixel, two pixels an element when staged.
    const mapFns = Qn
      ? Qn.map
      : Lz.staged
      ? `fn mapPut(i: u32, z: f32, s: f32) { ${mp}[i >> 1u][(i & 1u) * 2u] = bitcast<u32>(z); ${mp}[i >> 1u][(i & 1u) * 2u + 1u] = bitcast<u32>(s); }
fn mapP(i: u32) -> f32 { return bitcast<f32>(${mp}[i >> 1u][(i & 1u) * 2u]); }
fn mapS(i: u32) -> f32 { return bitcast<f32>(${mp}[i >> 1u][(i & 1u) * 2u + 1u]); }`
      : `fn mapPut(i: u32, z: f32, s: f32) { ${mp}[i] = vec4<u32>(bitcast<u32>(z), bitcast<u32>(s), 0u, 0u); }
fn mapP(i: u32) -> f32 { return bitcast<f32>(${mp}[i].x); }
fn mapS(i: u32) -> f32 { return bitcast<f32>(${mp}[i].y); }`;
    // ai: A peak in the found list: a vec4 element of the float pools, two vec2 elements of int8's.
    const peakStore = (i, v) => (int8 ? `peakPut(${i}, ${v})` : `${foundName}[${i}] = bitcast<vec4<u32>>(${v})`);
    const peakLoad = (i) => (int8 ? `peakGet(${i})` : `bitcast<vec4f>(${foundName}[${i}])`);
    const peakResp = (i) => (int8 ? `peakW(${i})` : `bitcast<f32>(${foundName}[${i}].w)`);
    netFns = `const RW = ${RW};
const WALK = ${stage === "net" ? walk : 1}u;
const RH = ${RH};
const XE = ${XE}u;
const RS = ${RS}u;
var<workgroup> nfound: array<atomic<u32>, ${NT}>;${packed ? `
fn pk(v: vec4f) -> vec2u { return vec2u(pack2x16float(v.xy), pack2x16float(v.zw)); }
fn upk(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }` : ""}${Qn ? Qn.fns : ""}
${mapFns}
${xfetchFn}${Qn ? Qn.xPut : `// X columns xo0 + 4g .. + 3 of window row jr into the X pool (RS elements a row, the last ones zero).
fn xPut(jr: u32, g: u32, xv: vec4f, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) {
  poolB[jr * RS + g] = bitcast<vec4<u32>>(xv);
  if (g == XE - 1u) { for (var e = XE; e < RS; e++) { poolB[jr * RS + e] = vec4<u32>(0u); } }${withDump ? `
  let y = yo0 + i32(jr);
  for (var k = 0; k < 4; k++) {
    let x = xo0 + i32(g) * 4 + k;
    if (x >= xo0 + HALO && x < min(xo0 + HALO + RW, w) && y >= yo0 + HALO && y < min(yo0 + HALO + RH, h)) { dbg[dbgBase(f) + u32(y) * P.stride + u32(x)] = xv[k]; }
  }` : ""}
}
`}// ai: A peak of the probability map at map index cc from its 3 x 3 neighbourhood nb (row major): above the floor and
// ai: its eight neighbours, ties to the earlier one. Strict, where v1 keeps a slack for its tiles' disagreeing halos:
// ai: v2's map is the level's alone. Placed by a parabola each axis, into tile t's list.
fn peakAt(nb: array<f32, 9>, cc: u32, t: u32, lx: i32, ly: i32, w: i32, h: i32) {
  let v = nb[4];
  var peak = v > FLOOR && lx < w && ly < h;
  for (var k = 0u; k < 9u; k++) {
    if (k != 4u) { peak = peak && select(v > nb[k], v >= nb[k], k < 4u); }
  }
  if (peak) {
    let ex = nb[5] + nb[3] - 2.0 * v;
    let ey = nb[7] + nb[1] - 2.0 * v;
    let ox = select(0.0, clamp(0.5 * (nb[3] - nb[5]) / ex, -0.5, 0.5), ex < 0.0);
    let oy = select(0.0, clamp(0.5 * (nb[1] - nb[7]) / ey, -0.5, 0.5), ey < 0.0);
    let kk = f32(1u << P.level);
    let uu = kk * NOMINAL * exp(mapS(cc));
    let at = atomicAdd(&nfound[t], 1u);
    if (at < LOCAL) { ${peakStore("t * LOCAL + at", "vec4f((f32(lx) + ox + 0.5) * kk, (f32(ly) + oy + 0.5) * kk, KAPPA * uu, v)")}; }
  }
}
`;
    // The non-maximum test: a lane takes a column run of PPL pixels and reads the PPL + 2 map rows round it once,
    // where the region's width divides the workgroup; else a pixel a round.
    const PPL = (RW * RH) / WG, runs = WG % RW === 0 && (WG / RW) * PPL === RH;
    const nms = runs ? `  {
    let cx = li % ${RW}u;
    let y0 = (li / ${RW}u) * ${PPL}u;
${Array.from({ length: PPL + 2 }, (_, rr) => `    let m${rr} = vec3f(mapP((y0 + ${rr}u) * ${MW}u + cx), mapP((y0 + ${rr}u) * ${MW}u + cx + 1u), mapP((y0 + ${rr}u) * ${MW}u + cx + 2u));`).join("\n")}
${Array.from({ length: PPL }, (_, p) => `    peakAt(array<f32, 9>(m${p}.x, m${p}.y, m${p}.z, m${p + 1}.x, m${p + 1}.y, m${p + 1}.z, m${p + 2}.x, m${p + 2}.y, m${p + 2}.z), (y0 + ${p + 1}u) * ${MW}u + cx + 1u, ((y0 + ${p}u) / ${TILE}u) * ${TX}u + cx / ${TILE}u, rx0 + i32(cx), ry0 + i32(y0) + ${p}, w, h);`).join("\n")}
  }` : `  for (var r = 0u; r < ${PPL}u; r++) {
    let q = li + r * ${WG}u;
    let px = q % ${RW}u + 1u;
    let py = q / ${RW}u + 1u;
    let cc = py * ${MW}u + px;
    var nb = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var k = 0u; k < 9u; k++) { nb[k] = mapP(cc + (k / 3u) * ${MW}u + (k % 3u) - ${MW}u - 1u); }
    peakAt(nb, cc, ((py - 1u) / ${TILE}u) * ${TX}u + (px - 1u) / ${TILE}u, rx0 + i32(px) - 1, ry0 + i32(py) - 1, w, h);
  }`;
    const body = `${xload}${layers}
${nms}
  workgroupBarrier();
  // Each tile's KEEP strongest by rank (ties by position, so ranks are distinct) into its slots; empty slots stay as
  // the clear left them, response 0. An item is (tile, place in its list).
  for (var it = li; it < ${NT * LOCAL}u; it += ${WG}u) {
    let t = it / LOCAL;
    let j = it % LOCAL;
    let tx = rxi * ${TX}u + t % ${TX}u;
    let ty = wg.y * ${TY}u + t / ${TX}u;
    let n = atomicLoad(&nfound[t]);
    let kept = min(n, LOCAL);
    if (j == 0u && n > 0u) { atomicAdd(&counts[f * 16u + 1u], n); }
    // ai: Peaks past LOCAL were never stored, so which LOCAL a full tile keeps is the lanes' order; count them.
    if (j == 0u && n > LOCAL) { atomicAdd(&counts[f * 16u + ${OVERFLOW}u], n - LOCAL); }
    if (j < kept && tx < P.tilesX) {
      let e = ${peakLoad("t * LOCAL + j")};
      var rank = 0u;
      for (var q = 0u; q < kept; q++) {
        let o = ${peakResp("t * LOCAL + q")};
        if (o > e.w || (o == e.w && q < j)) { rank++; }
      }
      if (rank < KEEP) { raw[f * P.slots + P.slotBase + (ty * P.tilesX + tx) * KEEP + rank] = e; }
    }
  }
`;
    netCode = stage === "net" ? `  for (var rr = 0u; rr < WALK; rr++) {
    let rxi = wg.x * WALK + rr;
    let rx0 = i32(rxi) * RW;
    let xo0 = rx0 - HALO;
    // The same for every lane (the frame's width and the loop count), so the loop's barriers stay uniform.
    if (rx0 >= w) { break; }
    if (li < ${NT}u) { atomicStore(&nfound[li], 0u); }
${body}    workgroupBarrier();
  }
` : `  let rxi = wg.x;
${body}`;
  }

  const planeType = halves || int8 ? "array<u32>" : "array<f32>";
  // ai: The pools' element: 16 bytes, or 8 in the int8 network (fcn2Layout).
  const EL = int8 && stage === "net" ? "vec2<u32>" : "vec4<u32>";
  const bind = stage === "x"
    ? `@group(0) @binding(0) var<storage, read> lev: array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read_write> xpl: ${planeType};
@group(0) @binding(3) var<uniform> P: Bank;`
    : `@group(0) @binding(0) var<storage, read> ${stage === "net" ? `xpl: ${planeType}` : "lev: array<f32>"};
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read_write> raw: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> P: Bank;${withDump ? `
@group(0) @binding(5) var<storage, read_write> dbg: array<f32>;
${Qn ? Qn.dbg : "fn dbgBase(f: u32) -> u32 { return f * (P.stride * P.rows + 2u * (P.stride + 2u) * (P.rows + 2u)); }"}` : ""}`;
  const what = stage === "x" ? `the X pass, ${XB[0]} x ${XB[1]} pixels a workgroup${int8 ? ", Xq out" : ""}` : `${P.macs} MACs a map pixel, ${int8 ? `int8 codes of ${Q.bits} bits` : `${T}${packed ? " with halves stored" : ""}`}, ${TX} x ${TY} tiles a workgroup${stage === "net" ? `, ${int8 ? "Xq" : "X"} from the plane` : ""}`;
  // The region (or block) and the origin of its X window in level pixels.
  const origin = stage === "x"
    ? `  let xo0 = i32(wg.x) * ${XB[0]};
  let yo0 = i32(wg.y) * ${XB[1]};
  if (F.valid == 0u || xo0 >= w || yo0 >= h) { return; }`
    : stage === "net" ? `  let ry0 = i32(wg.y) * RH;
  let yo0 = ry0 - HALO;
  // A whole workgroup returns together, so the early exit keeps control flow uniform.
  if (F.valid == 0u || i32(wg.x * WALK) * RW >= w || ry0 >= h) { return; }
${Array.from({ length: Math.ceil((Lz.XH * Lz.XE) / WG) }, (_, r) => `  var xp${r} = ${int8 ? "0u" : "vec4f(0.0)"};`).join("\n")}
  {
    let xo0 = i32(wg.x * WALK) * RW - HALO;
${Array.from({ length: Math.ceil((Lz.XH * Lz.XE) / WG) }, (_, r) => `    if (li + ${r * WG}u < ${Lz.XH * Lz.XE}u) { xp${r} = xfetch(li + ${r * WG}u, xo0, yo0, w, h, f); }`).join("\n")}
  }`
    : `  let rx0 = i32(wg.x) * RW;
  let ry0 = i32(wg.y) * RH;
  let xo0 = rx0 - HALO;
  let yo0 = ry0 - HALO;
  // A whole workgroup returns together, so the early exit keeps control flow uniform.
  if (F.valid == 0u || rx0 >= w || ry0 >= h) { return; }
  if (li < ${Lz.NT}u) { atomicStore(&nfound[li], 0u); }`;
  return (f16 ? "enable f16;\n" : "") + (int8 ? INT8_REQUIRES : "") + DIMS + /* wgsl */ `
// slotBase: where this level's tiles start in the frame's slots; tilesX: tiles a row at this level; slots: a frame's
// slots over every level. The bank contract's uniform (wgsl/bank.mjs); eps and the floor are baked in.
struct Bank { level: u32, stride: u32, rows: u32, slots: u32, tau: f32, eps: f32, slotBase: u32, tilesX: u32 }
${bind}
const HALO = ${HALO};
const KEEP = ${KEEP}u;
const LOCAL = ${LOCAL}u;
const KAPPA = ${f32(KAPPA)};
const NOMINAL = ${f32(P.nominal)};
const EPS = ${f32(P.eps)};
const FLOOR = ${f32(P.floor)};${int8 ? `
const QMAX = ${Q.qmax}.0;
const INV = ${exactF32(Q.inv)};` : ""}
// ${what}, box ${K}: ${bytes} bytes of workgroup memory.
var<workgroup> poolA: array<${EL}, ${nA}>;
var<workgroup> poolB: array<${EL}, ${nB}>;
${stage === "x" ? `const XE = ${Lz.XE}u;\n` : ""}${netFns}${statsFns}${planeFns}
${planePut}

@compute @workgroup_size(${WG}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.z;
  let F = frames[f];
  let w = i32((F.w + (1u << P.level) - 1u) >> P.level);
  let h = i32((F.h + (1u << P.level) - 1u) >> P.level);
${origin}
${statsCode}${netCode}}
`;
}

// ai: The int8 network's WGSL for fcn2Source (stage net; Q the file's codes from readQuant, Lz fcn2Layout's int8
// ai: layout): { layers (the convolutions and the fused head, inside the walk loop), fns (the X words, the peak list,
// ai: rq, the dump's writers), map (one element a pixel), xfetch, xPut (Xq words), dbg (the dump's frame layout) }.
function int8Net(Q, Lz, { withDump, probe }) {
  const { HALO, G1, convs } = Lz;
  // ai: G is a conv's geometry (Lz.convs), C its codes (Q.convs). A pixel's eight codes are a vec2<u32>, channels 0
  // ai: to 3 in x (channel 0 the low byte), 4 to 7 in y.
  const acc = (a, lo) => `vec4<i32>(${[0, 1, 2, 3].map((j) => `${a}${lo + j}`).join(", ")})`;
  const mul = (C, lo) => vec4Exact(C.mul.slice(lo, lo + 4));
  // ai: The dump's code planes as fcn2Dump lays them out: plane k's offset in a frame (a WGSL expression) and border.
  const qplanes = [];
  let dumpOff = "P.stride * P.rows", border = HALO;
  for (const L of convs.slice(0, -1)) { border -= L.dil; qplanes.push({ off: dumpOff, b: border }); dumpOff = `${dumpOff} + ${L.outC}u * dumpPlane(${border}u)`; }
  const dbg = `// ai: The int8 dump a frame (fcn2Dump): Xq, the code planes, then logit and s (border 1).
fn dumpPlane(b: u32) -> u32 { return (P.stride + 2u * b) * (P.rows + 2u * b); }
fn dumpMap() -> u32 { return ${dumpOff}; }
fn dbgBase(f: u32) -> u32 { return f * (dumpMap() + 2u * dumpPlane(1u)); }`;
  // ai: The dump of a conv's codes q0, q1 at its output pixel (x, y) into plane k.
  const codesDump = (k, x, y) => (withDump ? `
        { let d = dumpAt(${qplanes[k].b}, ${x}, ${y}, rx0, ry0, w, h); if (d >= 0) { dumpCodes(${qplanes[k].off} + u32(d), dumpPlane(${qplanes[k].b}u), q0, q1, f); } }` : "");
  // ai: conv1 (one input channel, dilation 1): four output pixels a lane, as the float conv1. A tap row's word t<j> is
  // ai: X pixels x0 + j .. x0 + j + 3, cut from the row's words gk and gk + 1; its weight word holds the row's three
  // ai: taps in bytes 0 to 2 and 0 in byte 3.
  const conv1Q = (G, C) => {
    const N = G1 * G.outH, lines = [];
    for (let j = 0; j < 4; j++) for (let o = 0; o < C.cout; o++) lines.push(`      var a${j}_${o} = ${C.bq[o]}i;`);
    for (let ky = 0; ky < 3; ky++) {
      lines.push(`      {
        let w0 = xr((y + ${ky}u) * RS + gk);
        let w1 = xr((y + ${ky}u) * RS + gk + 1u);
        let t0 = w0;
        let t1 = (w0 >> 8u) | (w1 << 24u);
        let t2 = (w0 >> 16u) | (w1 << 16u);
        let t3 = (w0 >> 24u) | (w1 << 8u);`);
      for (let j = 0; j < 4; j++) for (let o = 0; o < C.cout; o++) {
        const W = packI8x4(...[0, 1, 2].map((kx) => C.wq[o * 9 + ky * 3 + kx]), 0);
        if (W) lines.push(`        a${j}_${o} += dot4I8Packed(t${j}, ${hexWord(W)});`);
      }
      lines.push("      }");
    }
    const st = Array.from({ length: 4 }, (_, j) => `      if (x0 + ${j}u < ${G.outW}u) {
        let q0 = rq(${acc(`a${j}_`, 0)}, ${mul(C, 0)});
        let q1 = rq(${acc(`a${j}_`, 4)}, ${mul(C, 4)});
        poolA[y * ${G.outW}u + x0 + ${j}u] = vec2<u32>(pack4xI8(q0), pack4xI8(q1));${codesDump(0, `x0 + ${j}u`, "y")}
      }`);
    return `    for (var i = li; i < ${N}u; i += ${WG}u) {
      let gk = i % ${G1}u;
      let y = i / ${G1}u;
      let x0 = gk * 4u;
${lines.join("\n")}
${st.join("\n")}
    }
    workgroupBarrier();
`;
  };
  // ai: conv2 and conv3: a pixel a lane, one load a tap, each output channel two dot4I8Packed (input channels 0 to 3,
  // ai: then 4 to 7), each added to the channel's accumulator in a statement of its own: a device whose dot product
  // ai: takes an accumulator (RDNA 2's) then needs no add, where a + (d0 + d1) keeps one (the same i32 either way).
  // ai: conv2 requantises into registers and is written over conv1's codes after a barrier (staged); conv3 feeds the
  // ai: head.
  const convQ = (G, C, k) => {
    const last = k === convs.length - 1, N = G.outW * G.outH, d = G.dil, taps = [];
    for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const body = [];
      for (let o = 0; o < C.cout; o++) {
        const terms = [0, 1].map((g) => [g, packI8x4(...[0, 1, 2, 3].map((j) => C.wq[((o * C.cin + 4 * g + j) * 3 + ky) * 3 + kx]))]).filter(([, W]) => W).map(([g, W]) => `dot4I8Packed(e.${"xy"[g]}, ${hexWord(W)})`);
        for (const t of terms) body.push(`        a${o} += ${t};`);
      }
      if (body.length) taps.push(`      {\n        let e = poolA[(y + ${ky * d}u) * ${G.inW}u + x + ${kx * d}u];\n${body.join("\n")}\n      }`);
    }
    const init = Array.from({ length: C.cout }, (_, o) => `      var a${o} = ${C.bq[o]}i;`).join("\n");
    if (last) {
      // ai: The head from its bias, channel 0 first, one fma a channel (q8.mjs forwardCodes' order, in f32).
      const H = Q.head, chain = (row) => Array.from({ length: H.cin }, (_, c) => c).reduce((z, c) => `fma(${exactF32(H.w[row * H.cin + c])}, h${c >> 2}.${"xyzw"[c & 3]}, ${z})`, exactF32(H.b[row]));
      return `    for (var i = li; i < ${N}u; i += ${WG}u) {
      let x = i % ${G.outW}u;
      let y = i / ${G.outW}u;
${init}
${taps.join("\n")}
      let h0 = max(vec4f(${acc("a", 0)}) * ${mul(C, 0)}, vec4f(0.0));
      let h1 = max(vec4f(${acc("a", 4)}) * ${mul(C, 4)}, vec4f(0.0));
      let zl = ${chain(0)};
      let sv = ${chain(1)};
      mapPut(i, 1.0 / (1.0 + exp(-zl)), sv);${withDump ? `
      {
        let d = dumpAt(1, x, y, rx0, ry0, w, h);
        if (d >= 0) {
          dbg[dbgBase(f) + dumpMap() + u32(d)] = zl;
          dbg[dbgBase(f) + dumpMap() + dumpPlane(1u) + u32(d)] = sv;
        }
      }` : ""}
    }
    workgroupBarrier();
`;
    }
    const NR = Math.ceil(N / WG);
    return `${Array.from({ length: NR }, (_, r) => `    var o${k}_${r} = vec2<u32>(0u);`).join("\n")}
    for (var r = 0u; r < ${NR}u; r++) {
      let i = li + r * ${WG}u;
      if (i < ${N}u) {
      let x = i % ${G.outW}u;
      let y = i / ${G.outW}u;
${init}
${taps.join("\n")}
      let q0 = rq(${acc("a", 0)}, ${mul(C, 0)});
      let q1 = rq(${acc("a", 4)}, ${mul(C, 4)});
      let st = vec2<u32>(pack4xI8(q0), pack4xI8(q1));${codesDump(k, "x", "y")}
${Array.from({ length: NR }, (_, r) => `      if (r == ${r}u) { o${k}_${r} = st; }`).join("\n")}
      }
    }
    workgroupBarrier();
${Array.from({ length: NR }, (_, r) => `    if (li + ${r * WG}u < ${N}u) { poolA[li + ${r * WG}u] = o${k}_${r}; }`).join("\n")}
    workgroupBarrier();
`;
  };
  // ai: xfetch: an item is one word of four Xq. The window starts HALO before a region (a multiple of four), so where
  // ai: HALO is not one either a word inside the level is cut from two plane words.
  const cut = 8 * (((-HALO % 4) + 4) % 4);
  const xfetch = `fn xfetch(it: u32, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) -> u32 {
  let jr = it / XE;
  let g = it % XE;
  let x0 = xo0 + i32(g) * 4;
${probe.includes("noload") ? "  return u32(x0 + i32(jr)) * 0x01010101u;" : `  let y = clamp(yo0 + i32(jr), 0, h - 1);
  if (x0 >= 0 && x0 + 3 < w) {
    let i = (planeRow(f, y) + u32(x0)) >> 2u;
    return ${cut ? `(xpl[i] >> ${cut}u) | (xpl[i + 1u] << ${32 - cut}u)` : "xpl[i]"};
  }
  var v = 0u;
  for (var k = 0; k < 4; k++) { v |= planeByte(f, clamp(x0 + k, 0, w - 1), y) << (8u * u32(k)); }
  return v;`}
}
`;
  const fns = `
// ai: The X window's Xq words, four pixels a u32, two an element of B.
fn xw(i: u32, v: u32) { poolB[i >> 1u][i & 1u] = v; }
fn xr(i: u32) -> u32 { return poolB[i >> 1u][i & 1u]; }
// ai: A peak (x, y, sigma, response) is two elements of A.
fn peakPut(i: u32, v: vec4f) { poolA[2u * i] = bitcast<vec2<u32>>(v.xy); poolA[2u * i + 1u] = bitcast<vec2<u32>>(v.zw); }
fn peakGet(i: u32) -> vec4f { return vec4f(bitcast<vec2f>(poolA[2u * i]), bitcast<vec2f>(poolA[2u * i + 1u])); }
fn peakW(i: u32) -> f32 { return bitcast<f32>(poolA[2u * i + 1u].y); }
// ai: The contract's requantisation (cnn/q8.mjs requant) of four channels' accumulators, the bias in: one f32
// ai: multiply, rounded half to even, clamped to the codes 0 .. QMAX.
fn rq(a: vec4<i32>, m: vec4f) -> vec4<i32> { return vec4<i32>(clamp(round(vec4f(a) * m), vec4f(0.0), vec4f(QMAX))); }${withDump ? `
// ai: A conv's output pixel (x, y) at border b, its level pixel rx0 - b + x, ry0 - b + y: the index in a dump plane
// ai: (fcn2Dump) where the region owns the pixel or it lies outside the level within b (a neighbouring region may
// ai: write the same value there), else -1.
fn dumpAt(b: i32, x: u32, y: u32, rx0: i32, ry0: i32, w: i32, h: i32) -> i32 {
  let lx = rx0 - b + i32(x);
  let ly = ry0 - b + i32(y);
  let ox = (lx >= rx0 && lx < rx0 + RW) || lx < 0 || lx >= w;
  let oy = (ly >= ry0 && ly < ry0 + RH) || ly < 0 || ly >= h;
  if (!(ox && oy && lx >= -b && lx < w + b && ly >= -b && ly < h + b)) { return -1; }
  return (ly + b) * (i32(P.stride) + 2 * b) + lx + b;
}
// ai: Eight codes at o, a plane of pp elements apart.
fn dumpCodes(o: u32, pp: u32, q0: vec4<i32>, q1: vec4<i32>, f: u32) {
  for (var c = 0u; c < 4u; c++) { dbg[dbgBase(f) + o + c * pp] = f32(q0[c]); dbg[dbgBase(f) + o + (c + 4u) * pp] = f32(q1[c]); }
}` : ""}`;
  // ai: The dump build also writes Xq on the region's own level pixels.
  const xPut = `// ai: X columns xo0 + 4g .. + 3 of window row jr, as Xq, into the X pool (RS words a row, the last ones zero).
fn xPut(jr: u32, g: u32, xv: u32, xo0: i32, yo0: i32, w: i32, h: i32, f: u32) {
  xw(jr * RS + g, xv);
  if (g == XE - 1u) { for (var e = XE; e < RS; e++) { xw(jr * RS + e, 0u); } }${withDump ? `
  let y = yo0 + i32(jr);
  for (var k = 0; k < 4; k++) {
    let x = xo0 + i32(g) * 4 + k;
    if (x >= xo0 + HALO && x < min(xo0 + HALO + RW, w) && y >= yo0 + HALO && y < min(yo0 + HALO + RH, h)) { dbg[dbgBase(f) + u32(y) * P.stride + u32(x)] = f32(extractBits(bitcast<i32>(xv), 8u * u32(k), 8u)); }
  }` : ""}
}
`;
  // ai: The map: one 8-byte element a pixel.
  const map = `fn mapPut(i: u32, z: f32, s: f32) { poolB[i] = vec2<u32>(bitcast<u32>(z), bitcast<u32>(s)); }
fn mapP(i: u32) -> f32 { return bitcast<f32>(poolB[i].x); }
fn mapS(i: u32) -> f32 { return bitcast<f32>(poolB[i].y); }`;
  const layers = conv1Q(convs[0], Q.convs[0]) + convs.slice(1).map((L, k) => convQ(L, Q.convs[k + 1], k + 1)).join("");
  return { layers, fns, map, xfetch, xPut, dbg };
}

// Workgroup memory a source declares, bytes (the comment the source carries says the same).
export function fcn2Bytes(src) {
  const m = src.match(/(\d+) bytes of workgroup memory/);
  return m ? +m[1] : 0;
}
