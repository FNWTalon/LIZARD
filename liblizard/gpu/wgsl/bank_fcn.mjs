// F2 as a trained proposer (scripts/gpu/cnn/proposer/README.md), to the contract in gpu/wgsl/bank.mjs: one workgroup a
// 16 x 16 tile of one pyramid level (1 to 4) of one frame (or a region of `region` tiles side by side). It loads
// the tile with a halo of HALO pixels, clamped to the frame's edge, standardises it over what it loaded, runs the
// arch's 3 x 3 convolutions (valid, dilated, ReLU) and the 1 x 1 head in workgroup memory, and gets an 18 x 18 map
// of mark logits and log module sizes: the tile and one pixel of halo for the non-maximum test. A peak is a
// sigmoid above the floor and above its eight neighbours (ties to the earlier one), placed by a parabola each axis;
// its response is the probability, its sigma KAPPA times the size the head gives (u = 2^level x nominal x exp(s)),
// so the classifier reads the module size back as sigma / KAPPA. The tile's KEEP strongest go to its slots.
//
// The iGPU runs this kernel VALU bound, so it is written for instruction count. The weights are literals. A kernel
// row's taps are one fma chain into a partial sum, added to the accumulator once a row: a mat4x4 product costs a
// multiply and two adds more a tap, and one chain over all 72 terms drifts a third further from f32 in f16. A
// pixel's eight channels are one 16-byte element, so a tap is one LDS read and one bounds clamp (WebGPU clamps
// every workgroup index). conv1 makes four output pixels a lane from each row of taps read once. Buffers never
// live at the same time share two pools, so more workgroups stay resident. With f16 the activations are stored
// and multiplied as f16 (packed on RDNA); the standardisation, the head's outputs, the sigmoid and the parabola
// are f32. With packed the arithmetic is f32 and the stored activations are halves through pack2x16float (core
// WGSL, no shader-f16), for a device whose workgroup memory will not hold f32.
//
// region 2 shares the halo between two tiles (13% fewer MACs a tile) and standardises both over the one 44 x 28
// window: a different function from the one trained. It is a variant, not the default (README, "The kernel").
import { DIMS } from "./common.mjs";
import { KEEP } from "./bank.mjs";
import { KAPPA } from "../cnn/patch.mjs";
import { toHalf } from "../cnn/weights.mjs";
import { TAU } from "../decoder.mjs";

export { KEEP };
export const TILE = 16, LOCAL = 64, MAP = TILE + 2;
export const NMS_SLACK = 0.02, NMS_STRONG = 0.5;   // see the non-maximum test
const WG = 256;

// The exactly representable half nearest x, as a number.
function roundHalf(x) {
  const h = toHalf(x), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// The plan of an arch: each conv's input and output region for a TILE x TILE tile (the output map is MAP x MAP),
// channel counts (multiples of 4) and MACs a map pixel.
// Only v1's arch: a tile standardised over its own window. v2 ("norm": "pixel", a box a pixel) is bank_fcn2.mjs, and
// run here it would compute v1's input and propose from it without a word.
export function fcnPlan(json) {
  const arch = json.arch, spec = json.spec ?? "";
  if (!arch?.layers?.length) throw new Error(`proposer weights ${spec}: no arch.layers`);
  if ((arch.norm ?? "tile") !== "tile") throw new Error(`proposer weights ${spec}: arch.norm ${JSON.stringify(arch.norm)}; the v1 bank (BANK=fcn) standardises a tile, a "pixel" file is v2 (BANK=fcn2)`);
  if ((arch.tile ?? TILE) !== TILE) throw new Error(`proposer weights ${spec}: tile ${arch.tile}; the v1 bank's is ${TILE}`);
  const reach = 1 + arch.layers.reduce((a, l) => a + l.dil, 0), halo = arch.halo ?? reach;
  if (halo !== reach) throw new Error(`proposer weights ${spec}: halo ${halo}, the convs reach ${reach}`);
  let S = TILE + 2 * halo, C = 1;
  const convs = arch.layers.map((l, k) => {
    const layer = json.layers[k], out = { name: layer.name, dil: l.dil, inC: C, outC: l.out, inS: S, outS: S - 2 * l.dil, w: layer.w, b: layer.b };
    if (layer.name !== `conv${k + 1}` || layer.shape[0] !== l.out || layer.shape[1] !== C || l.out % 4) throw new Error(`proposer weights: ${layer.name} is ${JSON.stringify(layer.shape)}, the arch says ${l.out} x ${C} (channels in fours)`);
    S = out.outS; C = l.out;
    return out;
  });
  const head = json.layers[convs.length];
  if (!head || head.name !== "head" || head.shape[0] !== 2 || head.shape[1] !== C || S !== MAP) throw new Error(`proposer weights: head ${JSON.stringify(head?.shape)} after a ${S} x ${S} map`);
  const macs = convs.reduce((a, l) => a + l.inC * l.outC * 9, 0) + 2 * C;
  return { convs, head, halo, S0: TILE + 2 * halo, macs, eps: arch.eps, nominal: arch.nominal, floor: arch.floor ?? TAU };
}

// region: tiles a workgroup along x, standardised together. flat > 0: a workgroup whose window's deviation is under
// it returns with its slots empty (a frame's cost then depends on what it shows; a variant, not the default).
export function fcnSource(json, { f16 = false, packed = false, region = 1, flat = 0 } = {}) {
  if (f16 && packed) throw new Error("fcn bank: f16 already stores halves; packed is for f32");
  const P = fcnPlan(json), T = f16 ? "f16" : "f32", V = `vec4<${T}>`;
  if (P.convs[0].inC !== 1) throw new Error("fcn bank: conv1 takes one channel");
  // conv1 writes poolX, the convs after it alternate pools, and the last one writes the map into poolX, so it must
  // read poolY: an odd count. With an even count it would read and write poolX in one pass; with one conv the head
  // never runs.
  const nc = P.convs.length;
  if (nc < 3 || nc % 2 === 0) throw new Error(`fcn bank: ${nc} conv layer${nc === 1 ? "" : "s"}; the v1 kernel runs an odd number, 3 or more`);
  if (P.convs[0].dil > 2) throw new Error(`fcn bank: conv1's dilation ${P.convs[0].dil}; conv1 reads a row of taps as two elements, so 2 at most`);
  const RX = region, OW = TILE * RX, OH = TILE, HALO = P.halo, W0 = OW + 2 * HALO, H0 = OH + 2 * HALO, NIN = W0 * H0;
  const MW = OW + 2, MH = OH + 2, NMAP = MW * MH, NLD = Math.ceil(NIN / WG);
  let cw = W0, ch = H0;
  const convs = P.convs.map((L) => { const c = { ...L, inW: cw, inH: ch, outW: cw - 2 * L.dil, outH: ch - 2 * L.dil }; cw = c.outW; ch = c.outH; return c; });
  const f32 = (v) => { let s = (+v).toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return s; };
  const num = (v) => { const r = f16 ? roundHalf(v) : v; let s = r.toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return f16 ? `${s}h` : s; };
  const vec = (a) => `${V}(${a.map(num).join(", ")})`;
  // Column j of the weights for (tap, input group gi, output group go): the four output channels' weights on input
  // channel gi * 4 + j. PyTorch weight order: [out][in][ky][kx].
  const col = (L, ky, kx, gi, go, j) => vec(Array.from({ length: 4 }, (_, i) => L.w[(((go * 4 + i) * L.inC + gi * 4 + j) * 3 + ky) * 3 + kx]));

  // Two pools of 16-byte elements, each holding one thing at a time across the barriers. X: the reduction's
  // partials (four an element), then conv1's output, then the map (sigmoid and s a pixel). Y: the standardised
  // window (four pixels an element, rows of RS elements, so conv1 reads a row of taps as two elements), then
  // conv2's output, then the found peaks. f16 and packed keep a pixel's channels as two groups of four halves an
  // element; f32 as one group an element.
  const halves = f16 || packed, E = halves ? "vec4<u32>" : "vec4<f32>";
  const epp = (C) => (halves ? Math.ceil(C / 8) : C / 4);
  const G1 = Math.ceil(convs[0].outW / 4), RS = G1 + 1;
  const act = convs.slice(0, -1).map((L) => L.outW * L.outH * epp(L.outC));
  const nX = Math.max(WG / 2, NMAP, ...act.filter((_, k) => k % 2 === 0));
  const nY = Math.max(H0 * RS, RX * LOCAL, ...act.filter((_, k) => k % 2 === 1));
  const bytes = 16 * (nX + nY) + 128 + 4 * RX;
  const bits = (e) => `bitcast<u32>(${e})`, fromBits = (e) => `bitcast<f32>(${e})`;
  const elem = (a, b) => (halves ? `vec4<u32>(${a}, ${b}, 0u, 0u)` : `vec4<f32>(${fromBits(a)}, ${fromBits(b)}, 0.0, 0.0)`);
  const asU = (e) => (halves ? e : `bitcast<vec4<u32>>(${e})`);
  const grp = (es, g) => {
    if (!halves) return es[g];
    const h = `${es[g >> 1]}.${g & 1 ? "zw" : "xy"}`;
    return f16 ? `bitcast<vec4<f16>>(${h})` : `upk(${h})`;
  };
  const pack2 = (a, b) => `vec4<u32>(${f16 ? `bitcast<vec2<u32>>(${a})` : `pk(${a})`}, ${b ? (f16 ? `bitcast<vec2<u32>>(${b})` : `pk(${b})`) : "vec2<u32>(0u)"})`;
  const pool = (k) => (k % 2 ? "poolY" : "poolX");
  const storePixel = (dst, p, accs) => {
    const outs = accs.map((a) => `max(${a}, ${V}(0.0))`), n = epp(outs.length * 4);
    return (halves ? Array.from({ length: n }, (_, e) => `${dst}[${p} * ${n}u + ${e}u] = ${pack2(outs[2 * e], outs[2 * e + 1])};`) : outs.map((x, go) => `${dst}[${p} * ${n}u + ${go}u] = ${x};`)).join(" ");
  };

  // conv1: four output pixels a lane, columns x0..x0+7 of each tap row read as two elements.
  const conv1 = (L) => {
    const OG = L.outC / 4, N = G1 * L.outH, lines = [];
    for (let j = 0; j < 4; j++) for (let go = 0; go < OG; go++) lines.push(`      var a${j}_${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`);
    for (let ky = 0; ky < 3; ky++) {
      lines.push(`      {\n        let r0 = inpRow((y + ${ky * L.dil}u) * ${RS}u + gk);\n        let r1 = inpRow((y + ${ky * L.dil}u) * ${RS}u + gk + 1u);`);
      lines.push(`        let c = array<${T}, 8>(${T}(r0.x), ${T}(r0.y), ${T}(r0.z), ${T}(r0.w), ${T}(r1.x), ${T}(r1.y), ${T}(r1.z), ${T}(r1.w));`);
      for (let j = 0; j < 4; j++) for (let kx = 0; kx < 3; kx++) for (let go = 0; go < OG; go++)
        lines.push(`        a${j}_${go} = fma(${vec(Array.from({ length: 4 }, (_, i) => L.w[((go * 4 + i) * 9 + ky * 3 + kx)]))}, ${V}(c[${j + kx * L.dil}]), a${j}_${go});`);
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
      const pre = kx === 0 ? Array.from({ length: OG }, (_, go) => `        p${go} = ${V}(0.0);\n`).join("") : "";
      const post = kx === 2 ? Array.from({ length: OG }, (_, go) => `\n        acc${go} += p${go};`).join("") : "";
      const body = Array.from({ length: OG }, (_, go) => Array.from({ length: IG }, (_, gi) => Array.from({ length: 4 }, (_, j) => `        p${go} = fma(${col(L, ky, kx, gi, go, j)}, ${V}(i${gi}[${j}]), p${go});`).join("\n")).join("\n")).join("\n");
      taps.push(`      { let q = ((y + ${ky * L.dil}u) * ${L.inW}u + x + ${kx * L.dil}u) * ${Ei}u; ${reads} ${gs}\n${pre}${body}${post} }`);
    }
    let store;
    if (last) {
      // The head, fused. The probability goes into the map, so the non-maximum test reads it nine times without
      // nine exps.
      const H = P.head, dot = (o) => Array.from({ length: OG }, (_, go) => `dot(${vec(H.w.slice(o * L.outC + go * 4, o * L.outC + go * 4 + 4))}, max(acc${go}, ${V}(0.0)))`).join(" + ");
      store = `      mapPut(i, 1.0 / (1.0 + exp(-(${f32(H.b[0])} + f32(${dot(0)})))), ${f32(H.b[1])} + f32(${dot(1)}));`;
    } else {
      store = `      ${storePixel(pool(k), "i", Array.from({ length: OG }, (_, go) => `acc${go}`))}`;
    }
    return `    for (var i = li; i < ${N}u; i += ${WG}u) {
      let x = i % ${L.outW}u;
      let y = i / ${L.outW}u;
${Array.from({ length: OG }, (_, go) => `      var acc${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};\n      var p${go} = ${V}(0.0);`).join("\n")}
${taps.join("\n")}
${store}
    }
    workgroupBarrier();
`;
  };
  const layers = conv1(convs[0]) + convs.slice(1).map((L, k) => conv(L, k + 1)).join("");

  // Lane li loads window pixels li, li + WG, ... into registers and sums them in that order.
  const guard = (r) => ((r + 1) * WG > NIN ? `if (li + ${r * WG}u < ${NIN}u) ` : "");
  const loads = Array.from({ length: NLD }, (_, r) => `  var v${r} = 0.0;\n  ${guard(r)}{ v${r} = lev[(f * P.rows + u32(clamp(ty0 - HALO + i32((li + ${r * WG}u) / W0), 0, h - 1))) * P.stride + u32(clamp(tx0 - HALO + i32((li + ${r * WG}u) % W0), 0, w - 1))]; s1 += v${r}; s2 += v${r} * v${r}; }`).join("\n");
  const stores = Array.from({ length: NLD }, (_, r) => `  ${guard(r)}{ let q = li + ${r * WG}u; inpPut(q % W0, q / W0, (v${r} - mu) * inv); }`).join("\n");

  return (f16 ? "enable f16;\n" : "") + DIMS + /* wgsl */ `
// slotBase: where this level's tiles start in the frame's slots; tilesX: tiles a row at this level; slots: a frame's
// slots over every level. The bank contract's uniform (wgsl/bank.mjs); eps and the floor are baked in.
struct Bank { level: u32, stride: u32, rows: u32, slots: u32, tau: f32, eps: f32, slotBase: u32, tilesX: u32 }
@group(0) @binding(0) var<storage, read> lev: array<f32>;
@group(0) @binding(1) var<storage, read> frames: array<Frame>;
@group(0) @binding(2) var<storage, read_write> raw: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> P: Bank;
const W0 = ${W0}u;
const HALO = ${HALO};
const MAPW = ${MW}u;
const KEEP = ${KEEP}u;
const LOCAL = ${LOCAL}u;
const KAPPA = ${f32(KAPPA)};
const NOMINAL = ${f32(P.nominal)};
const EPS = ${f32(P.eps)};
const FLOOR = ${f32(P.floor)};
const NMS_SLACK = ${f32(NMS_SLACK)};
const NMS_STRONG = ${f32(NMS_STRONG)};
// ${P.macs} MACs a map pixel, ${T}${packed ? " with halves stored" : ""}, ${RX} tile${RX > 1 ? "s" : ""} a workgroup: ${bytes} bytes of workgroup memory.
var<workgroup> poolX: array<${E}, ${nX}>;
var<workgroup> poolY: array<${E}, ${nY}>;
var<workgroup> sums: array<f32, 32>;
var<workgroup> nfound: array<atomic<u32>, ${RX}>;
fn redPut(i: u32, v: f32) { poolX[i >> 2u][i & 3u] = ${halves ? bits("v") : "v"}; }
fn redGet4(e: u32) -> vec4f { return ${halves ? "bitcast<vec4f>(poolX[e])" : "poolX[e]"}; }
fn mapPut(i: u32, z: f32, s: f32) { poolX[i] = ${elem(bits("z"), bits("s"))}; }
fn zGet(i: u32) -> f32 { return ${fromBits(`${asU("poolX[i]")}.x`)}; }
fn sGet(i: u32) -> f32 { return ${fromBits(`${asU("poolX[i]")}.y`)}; }
fn foundPut(i: u32, e: vec4f) { poolY[i] = ${halves ? "bitcast<vec4<u32>>(e)" : "e"}; }
fn foundGet(i: u32) -> vec4f { return ${halves ? "bitcast<vec4f>(poolY[i])" : "poolY[i]"}; }
fn inpPut(x: u32, y: u32, v: f32) { poolY[y * ${RS}u + (x >> 2u)][x & 3u] = ${halves ? bits("v") : "v"}; }
fn inpRow(e: u32) -> vec4f { return ${halves ? "bitcast<vec4f>(poolY[e])" : "poolY[e]"}; }${packed ? `
fn pk(v: vec4f) -> vec2u { return vec2u(pack2x16float(v.xy), pack2x16float(v.zw)); }
fn upk(p: vec2u) -> vec4f { return vec4f(unpack2x16float(p.x), unpack2x16float(p.y)); }` : ""}

@compute @workgroup_size(${WG}, 1, 1)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let f = wg.z;
  let F = frames[f];
  let w = i32((F.w + (1u << P.level) - 1u) >> P.level);
  let h = i32((F.h + (1u << P.level) - 1u) >> P.level);
  let tx0 = i32(wg.x) * ${OW};
  let ty0 = i32(wg.y) * ${OH};
  // A whole workgroup returns together, so the early exit keeps control flow uniform.
  if (F.valid == 0u || tx0 >= w || ty0 >= h) { return; }
  if (li < ${RX}u) { atomicStore(&nfound[li], 0u); }
  var s1 = 0.0;
  var s2 = 0.0;
${loads}
  // The sums in two rounds of sixteen; every lane then adds the last sixteen in the same order, so mu and sd are
  // uniform across the workgroup.
  redPut(li, s1);
  redPut(${WG}u + li, s2);
  workgroupBarrier();
  if (li < 16u) {
    var a = 0.0;
    var b = 0.0;
    for (var k = 0u; k < 4u; k++) {
      let ea = redGet4(li * 4u + k);
      let eb = redGet4(${WG / 4}u + li * 4u + k);
      a += ea.x; a += ea.y; a += ea.z; a += ea.w;
      b += eb.x; b += eb.y; b += eb.z; b += eb.w;
    }
    sums[li] = a;
    sums[16u + li] = b;
  }
  let g = workgroupUniformLoad(&sums);
  var t1 = 0.0;
  var t2 = 0.0;
  for (var k = 0u; k < 16u; k++) { t1 += g[k]; t2 += g[16u + k]; }
  let mu = t1 / f32(${NIN}u);
  let sd = sqrt(max(t2 / f32(${NIN}u) - mu * mu, 0.0));
${flat > 0 ? `  // No contrast in the window, so no mark in it: the slots stay as the clear left them.
  if (sd < ${f32(flat)}) { return; }
` : ""}  let inv = 1.0 / (sd + EPS);
${stores}
  workgroupBarrier();
${layers}
  // The non-maximum test on each tile's 16 x 16 interior of the map, against the probability; ties go to the
  // earlier neighbour, so a flat top gives one peak. A strong pixel (above NMS_STRONG) whose neighbour wins by less
  // than NMS_SLACK is still a peak: the map's halo row is the neighbouring tile's pixel computed under THIS tile's
  // standardisation, so at a tile edge the two tiles can each see the other's pixel as the larger one and neither
  // keep the peak (two recorded frames lost a corner that way). The slack makes a near-tie a peak in both, and the
  // classifier sees the mark twice, which costs nothing. Weak pixels get no slack: a flat region's noise would
  // otherwise be a peak at every pixel.
  for (var r = 0u; r < ${RX}u; r++) {
    let q = li + r * ${WG}u;
    let px = q % ${OW}u + 1u;
    let py = q / ${OW}u + 1u;
    let t = (px - 1u) / ${TILE}u;
    let c = py * MAPW + px;
    let lx = tx0 + i32(px) - 1;
    let ly = ty0 + i32(py) - 1;
    let v = zGet(c);
    // Most of a map is under the floor, and a wave whose pixels all are skips the neighbours.
    if (v > FLOOR && lx < w && ly < h) {
    var peak = true;
    let slack = select(0.0, NMS_SLACK, v > NMS_STRONG);
    var nb = array<f32, 9>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var k = 0u; k < 9u; k++) {
      let qq = c + (k / 3u) * MAPW + (k % 3u) - MAPW - 1u;
      nb[k] = zGet(qq);
      if (k != 4u) { peak = peak && select(v > nb[k] - slack, v >= nb[k] - slack, k < 4u); }
    }
    if (peak) {
      let ex = nb[5] + nb[3] - 2.0 * v;
      let ey = nb[7] + nb[1] - 2.0 * v;
      let ox = select(0.0, clamp(0.5 * (nb[3] - nb[5]) / ex, -0.5, 0.5), ex < 0.0);
      let oy = select(0.0, clamp(0.5 * (nb[1] - nb[7]) / ey, -0.5, 0.5), ey < 0.0);
      let kk = f32(1u << P.level);
      let uu = kk * NOMINAL * exp(sGet(c));
      let at = atomicAdd(&nfound[t], 1u);
      if (at < LOCAL) { foundPut(t * LOCAL + at, vec4f((f32(lx) + ox + 0.5) * kk, (f32(ly) + oy + 0.5) * kk, KAPPA * uu, v)); }
    }
    }
  }
  workgroupBarrier();
  // Each tile's KEEP strongest by rank (ties by position, so ranks are distinct) into its slots; empty slots stay
  // as the clear left them, response 0.
  let t = li / ${WG / RX}u;
  let j = li % ${WG / RX}u;
  let tx = wg.x * ${RX}u + t;
  let n = atomicLoad(&nfound[t]);
  let kept = min(n, LOCAL);
  if (j == 0u && n > 0u) { atomicAdd(&counts[f * 16u + 1u], n); }
  if (j < kept && tx < P.tilesX) {
    let e = foundGet(t * LOCAL + j);
    var rank = 0u;
    for (var q = 0u; q < kept; q++) {
      let o = foundGet(t * LOCAL + q).w;
      if (o > e.w || (o == e.w && q < j)) { rank++; }
    }
    if (rank < KEEP) { raw[f * P.slots + P.slotBase + (wg.y * P.tilesX + tx) * KEEP + rank] = e; }
  }
}
`;
}

// Workgroup memory a source declares, bytes (the comment the source carries says the same).
export function fcnBytes(src) {
  const m = src.match(/(\d+) bytes of workgroup memory/);
  return m ? +m[1] : 0;
}
