// ai: F2's plan (glsl/bank_fcn2.mjs fcn2Source, the proposer version 2): the net's arch from its file, the forms a
// ai: device can be given and their workgroup memory (fcn2Layout), the literals the int8 form needs exact, and the
// ai: constants the host (../bank_fcn2.mjs) and ../../gen/gen.mjs read.
import { KEEP } from "./bank.mjs";
import { toHalf } from "../cnn/weights.mjs";
import { TAU } from "../constants.mjs";
import { readQuant } from "../cnn/q8.mjs";

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
export const WG = 256;

export function roundHalf(x) {
  const h = toHalf(x), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// The arch of a version 2 weights file: its convolutions, the head, the box (K, R), the halo the convs reach
// (1 + sum of dilations: the map at a pixel reads X within halo - 1, and the non-maximum ring adds one).
export function fcn2Plan(json) {
  const arch = json.arch;
  if (arch?.norm !== "pixel") throw new Error(`proposer weights ${json.spec ?? ""}: not version 2 (arch.norm ${JSON.stringify(arch?.norm)})`);
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

// ai: An f32 literal (WGSL's and GLSL's): nine significant digits give an f32 back. roundHalf, lit9, exactF32, quantOf
// ai: and the plan and layout are the GLSL twin's too (../glsl/bank_fcn2.mjs).
export const lit9 = (v) => { let s = (+v).toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return s; };
// ai: int8 literals. An f32 the contract multiplies by must reach the kernel as that f32 exactly (checked here); a
// ai: weight word is four codes packed as pack4xI8 packs them.
export function exactF32(v) {
  const s = lit9(v);
  if (Math.fround(v) !== v || Math.fround(+s) !== v) throw new Error(`fcn2 bank int8: ${v} is not an f32 the literal ${s} gives back`);
  return s;
}

// ai: A float form refuses a q8 file (its convs hold codes, no w or b) and the int8 form a float one: cnn/q8.mjs
// ai: readQuant refuses a file without arch.quant, another version, codes out of range or past the 2^24 bound.
// ai: Returns the parsed codes for int8, else null.
export function quantOf(json, int8) {
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
export function fcn2Layout(P, { TX = 1, TY = 1, halves = false, int8 = false, stage = "all", colrun = 0, XB = XBLOCKS[1] } = {}) {
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
    if (stage !== "net") throw new Error("fcn2 bank int8: the network reads the plane (stage net)");
    const NT = TX * TY, MW = RW + 2, MH = RH + 2;
    const nA = Math.max(...convs.slice(0, -1).map((L) => L.outW * L.outH), 2 * NT * LOCAL), nB = Math.max(Math.ceil((XH * RS) / 2), MW * MH);
    return { stage, RW, RH, HALO, R, XW, XH, XE, G1, RS, NT, MW, MH, convs, nA, nB, mapPool: "B", int8, bytes: 8 * (nA + nB) + 4 * NT };
  }
  const epp = (C) => (halves ? Math.ceil(C / 8) : C / 4);
  const NT = TX * TY, MW = RW + 2, MH = RH + 2;
  // Staged: every conv writes A, conv1 straight over nothing, the later ones from registers after a barrier (over the
  // output they read), so conv1's and conv2's outputs never coexist; the map (two pixels an element) goes in B over X,
  // the peaks in A.
  const act = convs.slice(0, -1).map((L) => L.outW * L.outH * epp(L.outC));
  const S = stage === "all" ? statsLayout(P, XW, XH, colrun) : { nA: 0, nB: 0 };
  const mapPool = "B", nA = Math.max(S.nA, ...act, NT * LOCAL), nB = Math.max(S.nB, XH * RS, Math.ceil((MW * MH) / 2));
  const bytes = 16 * (nA + nB) + (stage === "all" ? 4 * 16 : 0) + 4 * NT;
  return { ...(stage === "all" ? S : {}), stage, RW, RH, HALO, R, XW, XH, XE, G1, RS, NT, MW, MH, convs, epp, nA, nB, mapPool, bytes };
}

// The forms a device can be given, best first: f16 when the decoder runs f16; else f32 storage, then halves stored.
// Within a storage, the largest region first; the X plane before the fused kernel, its larger X block first. A form's
// bytes are its larger kernel's. `force`: { packed, f32, region: "32x16", fused, xblock: "32x32" } narrows it (the
// forms a smaller device would be given, run on one where a larger fits).
// ai: int8 (a q8 file, which no float form runs): store "int8", the plane form only; force.packed, f32 or fused
// ai: leave it no form.
export function fcn2Forms(json, { f16 = false, int8 = false, force = {} } = {}) {
  if (f16 && int8) throw new Error("fcn2 bank: f16 or int8, not both");
  quantOf(json, int8);
  const P = fcn2Plan(json);
  const regions = REGIONS.filter(([tx, ty]) => !force.region || force.region === `${TILE * tx}x${TILE * ty}`);
  const stores = int8 ? (force.packed || force.f32 ? [] : ["int8"]) : f16 ? ["f16"] : force.packed ? ["packed"] : force.f32 ? ["f32"] : ["f32", "packed"];
  const planes = int8 ? (force.fused ? [] : [true]) : force.fused ? [false] : [true, false];
  const out = [];
  const xblocks = XBLOCKS.filter((b) => !force.xblock || force.xblock === `${b[0]}x${b[1]}`);
  for (const store of stores) for (const [TX, TY] of regions) for (const xplane of planes) for (const XB of xplane ? xblocks : [null]) {
    const halves = store === "f16" || store === "packed", net = fcn2Layout(P, { TX, TY, halves, int8, stage: xplane ? "net" : "all" });
    const bytes = Math.max(net.bytes, xplane ? fcn2Layout(P, { stage: "x", XB }).bytes : 0);
    out.push({ store, TX, TY, xplane, XB, walk: xplane ? WALK : 1, f16: store === "f16", packed: store === "packed", int8, bytes, name: `${store} ${TILE * TX}x${TILE * TY}${xplane ? ` plane ${XB[0]}x${XB[1]}` : ""}` });
  }
  return out;
}

// Workgroup memory a source declares, bytes (the comment the source carries says the same).
export function fcn2Bytes(src) {
  const m = src.match(/(\d+) bytes of workgroup memory/);
  return m ? +m[1] : 0;
}
