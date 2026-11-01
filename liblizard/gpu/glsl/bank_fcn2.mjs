// F2, the trained proposer (version 2), in GLSL: the WGSL bank_fcn2.mjs (archived) fcn2Source statement for statement, the plane
// forms (that file's header says what the kernels compute, how the pools are laid out and why): the X pass (stage
// "x") and the network reading the plane (stage "net"), f32, f16, f32 with halves stored (packed) and int8, at every
// region, X block and walk fcn2Forms gives. The plan, the layout and the literals are that file's.
// ai: Natively these are the proposer's modules (gen's 32 KB device takes a plane form in every precision); the fused
// ai: kernel (stage "all"), the dump build and the timing probes stay the WGSL twin's alone (the browser lab's,
// ai: scripts/gpu/cnn/proposer/check2.mjs) and are refused here.
// ai: Bindings as the WGSL's: x: 0 lev, 1 frames, 2 the plane, 3 P; net: 0 the plane, 1 frames, 2 raw, 3 counts, 4 P.
// ai: The port's rules beyond glsl/common.mjs: no workgroup variable is zeroed (each pool element, cst and nfound is
// ai: written before a barrier that precedes its reads; nfound by its atomicStore at a region's start), where naga
// ai: zeroes every one; round is roundEven (WGSL's round, half to even); pack4xI8 is the low bytes shifted together
// ai: (gen's liz_pack4xI8, WGSL's definition: naga's own form failed on the Adreno 840); dot4I8Packed is
// ai: dotPacked4x8EXT on its words as ints (OpSDot, packed 4x8, naga's instruction); the f16 bitcasts are
// ai: packFloat2x16 and unpackFloat2x16 (OpBitcast between a u32 and two halves); the atomics as naga's (common.mjs
// ai: wgAdd ...); workgroupUniformLoad a barrier, the load, a barrier. GLSL declares before use, so the constants
// ai: and pools come first, then the plane's helpers, the box sums', the network's, then main.
import { VERSION, ATOMICS, F16, DOT, UNROLL, DIMS, PACK4X_I8, wgAdd, wgLoad, wgStore, devAdd } from "./common.mjs";
import { fcn2Plan, fcn2Layout, quantOf, roundHalf, lit9, exactF32, TILE, LOCAL, OVERFLOW, KEEP, WALK, WG, XBLOCKS } from "../plan/bank_fcn2.mjs";
import { KAPPA } from "../cnn/patch.mjs";
import { packI8x4 } from "../cnn/q8.mjs";

// ai: The network's loops' controls stated (SPIR-V Unroll and DontUnroll, common.mjs UNROLL): its small fixed loops
// ai: unrolled, its pixel loops and the walk kept rolled. On the S26 the frame 1.1 to 1.7% less (the half form's
// ai: registers 43 to 17), the iGPU level; the same on the X pass's loops was level on both, so it is left as written.
const vec4Exact = (a) => `vec4(${Array.from(a, exactF32).join(", ")})`;
// ai: a packed weight word as the int OpSDot takes (its bits, folded by glslang)
const intWord = (u) => `int(0x${u.toString(16).padStart(8, "0")}u)`;
// ai: the shared helpers: halves to and from an f16 vector (WGSL's bitcast<vec4<f16>>, bitcast<vec2<u32>>), f32
// ai: through halves (WGSL's pk, upk); pack4xI8 is common.mjs PACK4X_I8
const HALF_FNS = `f16vec4 u2h(uvec2 w) { return f16vec4(unpackFloat2x16(w.x), unpackFloat2x16(w.y)); }
uvec2 h2u(f16vec4 h) { return uvec2(packFloat2x16(h.xy), packFloat2x16(h.zw)); }`;
const PACKED_FNS = `uvec2 pk(vec4 v) { return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw)); }
vec4 upk(uvec2 p) { return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)); }`;

// The GLSL of one stage: "x" the X pass, "net" the network reading the plane (the WGSL's options; dump, probe and
// stage "all" refused).
export function fcn2Source(json, { f16 = false, packed = false, int8 = false, TX = 2, TY = 2, stage = "net", dump = false, probe = "", colrun = 0, XB = XBLOCKS[1], walk = WALK } = {}) {
  if (stage !== "x" && stage !== "net") throw new Error(`fcn2 GLSL: stage ${stage}: the plane forms alone (x, net); the fused kernel is the WGSL twin's`);
  if (dump || probe) throw new Error("fcn2 GLSL: the dump build and the probes are the WGSL twin's (the browser checks)");
  if (f16 && packed) throw new Error("fcn2 bank: f16 already stores halves; packed is for f32");
  if (int8 && (f16 || packed)) throw new Error("fcn2 bank: int8 is a store of its own with the plane form only (stage x, then net)");
  const Q = quantOf(json, int8);
  const P = fcn2Plan(json), halves = f16 || packed, Lz = fcn2Layout(P, { TX, TY, halves, int8, stage, colrun, XB });
  const { HALO, R, nA, nB, bytes } = Lz;
  const T = f16 ? "float16_t" : "float", V4 = f16 ? "f16vec4" : "vec4", K = P.K, KK = K * K;
  const f32 = lit9;
  const num = (v) => { const r = f16 ? roundHalf(v) : v; let s = r.toPrecision(9); if (!/[.e]/.test(s)) s += ".0"; return f16 ? `${s}hf` : s; };
  const vec = (a) => `${V4}(${a.map(num).join(", ")})`;
  const net = stage === "net";

  // ---- The box sums (stage x): the window of v - c, the row pass, the column pass into registers.
  let statsCode = "", statsFns = "";
  if (!net) {
    const { XH, XE, NE, VH, VE, RR } = Lz;
    const NRUN = Math.ceil(XH / RR), NCI = XE * NRUN, NCL = Math.ceil(NCI / WG);
    const NV = VH * VE, NLD = Math.ceil(NV / WG), guard = (r) => ((r + 1) * WG > NV ? `if (q < ${NV}u) ` : "");
    const loads = Array.from({ length: NLD }, (_, r) => `  vec4 v${r} = vec4(0.0);
  { uint q = li + ${r * WG}u; ${guard(r)}{
    uint row = (f * P.rows + uint(clamp(yo0 - ${R} + int(q / VE), 0, h - 1))) * P.stride;
    uvec4 xs = uvec4(clamp(ivec4(xo0 - ${R} + 4 * int(q % VE)) + ivec4(0, 1, 2, 3), ivec4(0), ivec4(w - 1)));
    v${r} = vec4(lev[row + xs.x], lev[row + xs.y], lev[row + xs.z], lev[row + xs.w]);
    s0 += (v${r}.x + v${r}.y) + (v${r}.z + v${r}.w);
  } }`).join("\n");
    const puts = Array.from({ length: NLD }, (_, r) => `  { uint q = li + ${r * WG}u; ${guard(r)}{ poolB[q] = floatBitsToUint(v${r} - vec4(c)); } }`).join("\n");
    const a = Array.from({ length: 4 * NE }, (_, n) => `b${n >> 2}.${"xyzw"[n & 3]}`);
    const sum = (fn) => Array.from({ length: K }, (_, n) => fn(n)).join(" + ");
    const rowPass = `  for (uint it = li; it < ${VH * XE}u; it += ${WG}u) {
    uint j = it / XE;
    uint g = it % XE;
${Array.from({ length: NE }, (_, n) => `    vec4 b${n} = uintBitsToFloat(poolB[j * VE + g + ${n}u]);`).join("\n")}
${Array.from({ length: K + 3 }, (_, n) => `    float a${n} = ${a[n]};\n    float q${n} = a${n} * a${n};`).join("\n")}
    float s10 = ${sum((n) => `a${n}`)};
    float s20 = ${sum((n) => `q${n}`)};
${[1, 2, 3].map((k) => `    float s1${k} = s1${k - 1} + a${K + k - 1} - a${k - 1};\n    float s2${k} = s2${k - 1} + q${K + k - 1} - q${k - 1};`).join("\n")}
    poolA[j * XE + g] = floatBitsToUint(vec4(s10, s11, s12, s13));
    poolA[HOFF + j * XE + g] = floatBitsToUint(vec4(s20, s21, s22, s23));
  }
  barrier();
`;
    const item = (c) => {
      const lines = [`  {
    uint ci = li + ${c * WG}u;
    if (ci < ${NCI}u) {
      uint g = ci % XE;
      uint j0 = (ci / XE) * ${RR}u;
      int x0 = xo0 + int(g) * 4;
      uvec4 cx = uvec4(clamp(ivec4(x0) + ivec4(0, 1, 2, 3), ivec4(0), ivec4(w - 1)) - ivec4(xo0));
      bool inner = x0 >= 0 && x0 + 3 < w;
      vec4 s1 = vec4(0.0);
      vec4 s2 = vec4(0.0);
      uint r0 = uint(clamp(yo0 + int(j0), 0, h - 1) - yo0);
      for (uint r = 0u; r < KBOX; r++) { HS hh = hrow(r0 + r, g, cx, inner); s1 += hh.a; s2 += hh.b; }
      xs${c}_0 = xval(s1, s2, r0 + ${R}u, cx);`];
      for (let k = 1; k < RR; k++) lines.push(`      if (j0 + ${k}u < ${XH}u) {
        uint rn = uint(clamp(yo0 + int(j0 + ${k}u), 0, h - 1) - yo0);
        if (rn != r0) { HS hi = hrow(r0 + KBOX, g, cx, inner); HS lo = hrow(r0, g, cx, inner); s1 += hi.a - lo.a; s2 += hi.b - lo.b; r0 = rn; }
        xs${c}_${k} = xval(s1, s2, r0 + ${R}u, cx);
      }`);
      lines.push("    }\n  }");
      return Array.from({ length: RR }, (_, k) => `  vec4 xs${c}_${k} = vec4(0.0);`).join("\n") + "\n" + lines.join("\n");
    };
    const each = (put) => Array.from({ length: NCL }, (_, c) => `  {
    uint ci = li + ${c * WG}u;
    uint g = ci % XE;
    uint j0 = (ci / XE) * ${RR}u;
${Array.from({ length: RR }, (_, k) => `    if (ci < ${NCI}u && j0 + ${k}u < ${XH}u) { ${put(`j0 + ${k}u`, "g", `xs${c}_${k}`)} }`).join("\n")}
  }`).join("\n");
    statsCode = `  float s0 = 0.0;
${loads}
  // c, the window's mean: partial sums in two rounds of sixteen, every lane adding the last sixteen in one order.
  poolA[li >> 2u][li & 3u] = floatBitsToUint(s0);
  barrier();
  if (li < 16u) {
    float t = 0.0;
    for (uint k = 0u; k < 16u; k++) { t += uintBitsToFloat(poolA[(li * 16u + k) >> 2u][(li * 16u + k) & 3u]); }
    cst[li] = t;
  }
  // ai: WGSL's workgroupUniformLoad(&cst): a barrier, the load, a barrier
  barrier();
  float gs[16] = cst;
  barrier();
  float tc = 0.0;
  for (uint k = 0u; k < 16u; k++) { tc += gs[k]; }
  float c = tc / ${f32(4 * NV)};
${puts}
  barrier();
${rowPass}${Array.from({ length: NCL }, (_, c) => item(c)).join("\n")}
` + each((j, g, v) => `planePut(${j}, ${g}, ${v}, xo0, yo0, w, h, f);`);
    statsFns = `const uint VE = ${VE}u;
const uint HOFF = ${VH * XE}u;
const uint KBOX = ${K}u;
shared float cst[16];
float vGet(uint j, uint i) { return uintBitsToFloat(poolB[j * VE + (i >> 2u)][i & 3u]); }
// The row sums of box row r for X columns 4g .. 4g + 3: one element each where no column is clamped, else a column at
// a time, each column's box being the one centred on its pixel clamped into the level (the spec's step 4).
struct HS { vec4 a; vec4 b; };
HS hrow(uint r, uint g, uvec4 cx, bool inner) {
  if (inner) { return HS(uintBitsToFloat(poolA[r * XE + g]), uintBitsToFloat(poolA[HOFF + r * XE + g])); }
  HS o = HS(vec4(0.0), vec4(0.0));
  for (uint k = 0u; k < 4u; k++) {
    uint e = r * XE + (cx[k] >> 2u);
    o.a[k] = uintBitsToFloat(poolA[e][cx[k] & 3u]);
    o.b[k] = uintBitsToFloat(poolA[HOFF + e][cx[k] & 3u]);
  }
  return o;
}
// X from a box's sums of v - c and (v - c)^2 and its centre's window row vy.
vec4 xval(vec4 s1, vec4 s2, uint vy, uvec4 cx) {
  vec4 mu = s1 / ${f32(KK)};
  vec4 sd = sqrt(max(s2 / ${f32(KK)} - mu * mu, vec4(0.0)));
  vec4 vc = vec4(vGet(vy, cx.x + ${R}u), vGet(vy, cx.y + ${R}u), vGet(vy, cx.z + ${R}u), vGet(vy, cx.w + ${R}u));
  return (vc - mu) / (sd + vec4(EPS));
}
`;
  }

  // The plane: frame f's row y at (f rows + y) SP, SP even so a pair of halves never straddles two rows.
  // ai: int8: SP a multiple of four, so a word of four Xq never straddles two rows.
  const planeFns = int8 ? `uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 3u) & ~3u); }
// ai: the Xq byte of level pixel (x, y), as its bits
uint planeByte(uint f, int x, int y) { uint i = planeRow(f, y) + uint(x); return (xpl[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }` : `uint planeRow(uint f, int y) { return (f * P.rows + uint(y)) * ((P.stride + 1u) & ~1u); }
${halves ? "float planeAt(uint f, int x, int y) { uint i = planeRow(f, y) + uint(x); return unpackHalf2x16(xpl[i >> 1u])[i & 1u]; }" : "float planeAt(uint f, int x, int y) { return xpl[planeRow(f, y) + uint(x)]; }"}`;
  const planePut = net ? "" : int8 ? `// ai: Xq of X columns xo0 + 4g .. + 3 (xo0 a multiple of 4: one word) of block row j into the plane, on the level; past
// ai: its right edge the word repeats the edge's X.
void planePut(uint j, uint g, vec4 xv, int xo0, int yo0, int w, int h, uint f) {
  int x = xo0 + int(g) * 4;
  int y = yo0 + int(j);
  if (x >= w || y >= h) { return; }
  xpl[(planeRow(f, y) + uint(x)) >> 2u] = pack4xI8(ivec4(clamp(roundEven(xv * INV), vec4(-QMAX), vec4(QMAX))));
}` : `// X columns xo0 + 4g .. + 3 (xo0 a multiple of 4, so two whole pairs) of block row j into the plane, on the level.
void planePut(uint j, uint g, vec4 xv, int xo0, int yo0, int w, int h, uint f) {
  int x = xo0 + int(g) * 4;
  int y = yo0 + int(j);
  if (x >= w || y >= h) { return; }
  uint i = planeRow(f, y) + uint(x);
${halves ? `  xpl[i >> 1u] = packHalf2x16(xv.xy);
  if (x + 2 < w) { xpl[(i >> 1u) + 1u] = packHalf2x16(xv.zw); }` : "  for (int k = 0; k < 4; k++) { if (x + k < w) { xpl[i + uint(k)] = xv[k]; } }"}
}`;

  // ---- The network (stage net).
  let netCode = "", netFns = "", netConsts = "", Qn = null;
  if (net) {
    const { RW, RH, XH, XE, G1, RS, NT, MW, convs, epp } = Lz;
    const half2 = (a) => (a ? (f16 ? `h2u(${a})` : `pk(${a})`) : "uvec2(0u)");
    const packPixel = (accs) => {
      const outs = accs.map((a) => `max(${a}, ${V4}(0.0))`);
      return halves ? Array.from({ length: epp(outs.length * 4) }, (_, e) => `uvec4(${half2(outs[2 * e])}, ${half2(outs[2 * e + 1])})`) : outs.map((x) => `floatBitsToUint(${x})`);
    };
    const col = (L, ky, kx, gi, go, j) => vec(Array.from({ length: 4 }, (_, i) => L.w[(((go * 4 + i) * L.inC + gi * 4 + j) * 3 + ky) * 3 + kx]));
    const grp = (es, g) => {
      if (!halves) return `uintBitsToFloat(${es[g]})`;
      const h = `${es[g >> 1]}.${g & 1 ? "zw" : "xy"}`;
      return f16 ? `u2h(${h})` : `upk(${h})`;
    };
    const storePixel = (dst, p, accs) => { const es = packPixel(accs); return es.map((x, e) => `${dst}[${p} * ${es.length}u + ${e}u] = ${x};`).join(" "); };
    // conv1: four output pixels a lane, each tap row read as two elements of X (four pixels an element).
    const conv1 = (L) => {
      const OG = L.outC / 4, N = G1 * L.outH, lines = [];
      for (let j = 0; j < 4; j++) for (let go = 0; go < OG; go++) lines.push(`      ${V4} a${j}_${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`);
      for (let ky = 0; ky < 3; ky++) {
        lines.push(`      {\n        vec4 r0 = uintBitsToFloat(poolB[(y + ${ky * L.dil}u) * ${RS}u + gk]);\n        vec4 r1 = uintBitsToFloat(poolB[(y + ${ky * L.dil}u) * ${RS}u + gk + 1u]);`);
        lines.push(`        ${T} cv[8] = ${T}[8](${T}(r0.x), ${T}(r0.y), ${T}(r0.z), ${T}(r0.w), ${T}(r1.x), ${T}(r1.y), ${T}(r1.z), ${T}(r1.w));`);
        for (let j = 0; j < 4; j++) for (let kx = 0; kx < 3; kx++) for (let go = 0; go < OG; go++)
          lines.push(`        a${j}_${go} = fma(${vec(Array.from({ length: 4 }, (_, i) => L.w[(go * 4 + i) * 9 + ky * 3 + kx]))}, ${V4}(cv[${j + kx * L.dil}]), a${j}_${go});`);
        lines.push("      }");
      }
      const st = Array.from({ length: 4 }, (_, j) => `      if (x0 + ${j}u < ${L.outW}u) { uint p = y * ${L.outW}u + x0 + ${j}u; ${storePixel("poolA", "p", Array.from({ length: OG }, (_, go) => `a${j}_${go}`))} }`);
      return `    [[dont_unroll]] for (uint i = li; i < ${N}u; i += ${WG}u) {
      uint gk = i % ${G1}u;
      uint y = i / ${G1}u;
      uint x0 = gk * 4u;
${lines.join("\n")}
${st.join("\n")}
    }
    barrier();
`;
    };
    const conv = (L, k) => {
      const last = k === convs.length - 1, OG = L.outC / 4, IG = L.inC / 4, N = L.outW * L.outH, Ei = epp(L.inC), src = "poolA";
      const taps = [];
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
        const es = Array.from({ length: Ei }, (_, e) => `e${e}`);
        const reads = es.map((e, n) => `uvec4 ${e} = ${src}[q + ${n}u];`).join(" ");
        const gs = Array.from({ length: IG }, (_, gi) => `${V4} i${gi} = ${grp(es, gi)};`).join(" ");
        const body = Array.from({ length: OG }, (_, go) => Array.from({ length: IG }, (_, gi) => Array.from({ length: 4 }, (_, j) => `        acc${go} = fma(${col(L, ky, kx, gi, go, j)}, ${V4}(i${gi}[${j}]), acc${go});`).join("\n")).join("\n")).join("\n");
        taps.push(`      { uint q = ((y + ${ky * L.dil}u) * ${L.inW}u + x + ${kx * L.dil}u) * ${Ei}u; ${reads} ${gs}\n${body} }`);
      }
      let store;
      if (last) {
        // The head, fused. The map keeps the probability, so the non-maximum test reads it nine times without nine exps.
        const H = P.head, dot = (o) => Array.from({ length: OG }, (_, go) => `dot(${vec(H.w.slice(o * L.outC + go * 4, o * L.outC + go * 4 + 4))}, max(acc${go}, ${V4}(0.0)))`).join(" + ");
        store = `      float zl = ${f32(H.b[0])} + float(${dot(0)});
      float sv = ${f32(H.b[1])} + float(${dot(1)});
      mapPut(i, 1.0 / (1.0 + exp(-zl)), sv);`;
      } else {
        // Staged: the pixel's elements into registers o<r>_<e> for round r, written over the input after a barrier.
        const NR = Math.ceil(N / WG), es = packPixel(Array.from({ length: OG }, (_, go) => `acc${go}`));
        const regs = Array.from({ length: NR }, (_, r) => es.map((_, e) => `o${k}_${r}_${e}`));
        return `${regs.flat().map((o) => `    uvec4 ${o} = uvec4(0u);`).join("\n")}
    [[dont_unroll]] for (uint r = 0u; r < ${NR}u; r++) {
      uint i = li + r * ${WG}u;
      if (i < ${N}u) {
      uint x = i % ${L.outW}u;
      uint y = i / ${L.outW}u;
${Array.from({ length: OG }, (_, go) => `      ${V4} acc${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`).join("\n")}
${taps.join("\n")}
${es.map((x, e) => `      uvec4 st${e} = ${x};`).join("\n")}
${regs.map((rr, r) => `      if (r == ${r}u) { ${rr.map((o, e) => `${o} = st${e};`).join(" ")} }`).join("\n")}
      }
    }
    barrier();
${regs.map((rr, r) => `    if (li + ${r * WG}u < ${N}u) { ${rr.map((o, e) => `poolA[(li + ${r * WG}u) * ${es.length}u + ${e}u] = ${o};`).join(" ")} }`).join("\n")}
    barrier();
`;
      }
      return `    [[dont_unroll]] for (uint i = li; i < ${N}u; i += ${WG}u) {
      uint x = i % ${L.outW}u;
      uint y = i / ${L.outW}u;
${Array.from({ length: OG }, (_, go) => `      ${V4} acc${go} = ${vec(L.b.slice(go * 4, go * 4 + 4))};`).join("\n")}
${taps.join("\n")}
${store}
    }
    barrier();
`;
    };
    Qn = int8 ? int8Net(Q, Lz) : null;
    const layers = Qn ? Qn.layers : conv1(convs[0]) + convs.slice(1).map((L, k) => conv(L, k + 1)).join("");
    // The X window from the plane, four pixels an item, clamped to the level; a group of four wholly in the level is
    // two pairs (the window's origin rx0 - HALO is even), else each pixel is read at its clamped place. A workgroup
    // walks WALK regions along the row and fetches the next region's window into registers (xp<r>) before the
    // current one's convolutions, so the loads' latency hides behind them.
    const NXI = XH * XE, NXL = Math.ceil(NXI / WG);
    const xfetchFn = Qn ? Qn.xfetch : `vec4 xfetch(uint it, int xo0, int yo0, int w, int h, uint f) {
  uint jr = it / XE;
  uint g = it % XE;
  int x0 = xo0 + int(g) * 4;
  int y = clamp(yo0 + int(jr), 0, h - 1);
  vec4 xv = vec4(0.0);
  if (x0 >= 0 && x0 + 3 < w) {
${halves ? `    uint i = (planeRow(f, y) + uint(x0)) >> 1u;
    xv = vec4(unpackHalf2x16(xpl[i]), unpackHalf2x16(xpl[i + 1u]));` : `    uint i = planeRow(f, y) + uint(x0);
    xv = vec4(xpl[i], xpl[i + 1u], xpl[i + 2u], xpl[i + 3u]);`}
  } else {
    [[unroll]] for (int k = 0; k < 4; k++) { xv[k] = planeAt(f, clamp(x0 + k, 0, w - 1), y); }
  }
  return xv;
}
`;
    const xfetch = (xo) => Array.from({ length: NXL }, (_, r) => `      if (li + ${r * WG}u < ${NXI}u) { xp${r} = xfetch(li + ${r * WG}u, ${xo}, yo0, w, h, f); }`).join("\n");
    const xload = `${Array.from({ length: NXL }, (_, r) => `    if (li + ${r * WG}u < ${NXI}u) { xPut((li + ${r * WG}u) / XE, (li + ${r * WG}u) % XE, xp${r}, xo0, yo0, w, h, f); }`).join("\n")}
    barrier();
    if (rr + 1u < WALK && rx0 + RW < w) {
${xfetch("xo0 + RW")}
    }
`;
    const mapFns = Qn
      ? Qn.map
      : `void mapPut(uint i, float z, float s) { poolB[i >> 1u][(i & 1u) * 2u] = floatBitsToUint(z); poolB[i >> 1u][(i & 1u) * 2u + 1u] = floatBitsToUint(s); }
float mapP(uint i) { return uintBitsToFloat(poolB[i >> 1u][(i & 1u) * 2u]); }
float mapS(uint i) { return uintBitsToFloat(poolB[i >> 1u][(i & 1u) * 2u + 1u]); }`;
    // ai: A peak in the found list: a vec4 element of the float pools, two vec2 elements of int8's.
    const peakStore = (i, v) => (int8 ? `peakPut(${i}, ${v})` : `poolA[${i}] = floatBitsToUint(${v})`);
    const peakLoad = (i) => (int8 ? `peakGet(${i})` : `uintBitsToFloat(poolA[${i}])`);
    const peakResp = (i) => (int8 ? `peakW(${i})` : `uintBitsToFloat(poolA[${i}].w)`);
    netConsts = `const int RW = ${RW};
const uint WALK = ${walk}u;
const int RH = ${RH};
const uint XE = ${XE}u;
const uint RS = ${RS}u;
shared uint nfound[${NT}];
`;
    netFns = `${f16 ? HALF_FNS : ""}${packed ? PACKED_FNS : ""}${Qn ? Qn.fns : ""}
${mapFns}
${xfetchFn}${Qn ? Qn.xPut : `// X columns xo0 + 4g .. + 3 of window row jr into the X pool (RS elements a row, the last ones zero).
void xPut(uint jr, uint g, vec4 xv, int xo0, int yo0, int w, int h, uint f) {
  poolB[jr * RS + g] = floatBitsToUint(xv);
  if (g == XE - 1u) { [[unroll]] for (uint e = XE; e < RS; e++) { poolB[jr * RS + e] = uvec4(0u); } }
}
`}// ai: A peak of the probability map at map index cc from its 3 x 3 neighbourhood nb (row major): above the floor and
// ai: its eight neighbours, ties to the earlier one. Placed by a parabola each axis, into tile t's list.
void peakAt(float nb[9], uint cc, uint t, int lx, int ly, int w, int h) {
  float v = nb[4];
  bool peak = v > FLOOR && lx < w && ly < h;
  [[unroll]] for (uint k = 0u; k < 9u; k++) {
    if (k != 4u) { peak = peak && (k < 4u ? v >= nb[k] : v > nb[k]); }
  }
  if (peak) {
    float ex = nb[5] + nb[3] - 2.0 * v;
    float ey = nb[7] + nb[1] - 2.0 * v;
    float ox = ex < 0.0 ? clamp(0.5 * (nb[3] - nb[5]) / ex, -0.5, 0.5) : 0.0;
    float oy = ey < 0.0 ? clamp(0.5 * (nb[1] - nb[7]) / ey, -0.5, 0.5) : 0.0;
    float kk = float(1u << P.level);
    float uu = kk * NOMINAL * exp(mapS(cc));
    uint at = ${wgAdd("nfound[t]", "1u")};
    if (at < LOCAL) { ${peakStore("t * LOCAL + at", "vec4((float(lx) + ox + 0.5) * kk, (float(ly) + oy + 0.5) * kk, KAPPA * uu, v)")}; }
  }
}
`;
    // The non-maximum test: a lane takes a column run of PPL pixels and reads the PPL + 2 map rows round it once,
    // where the region's width divides the workgroup; else a pixel a round.
    const PPL = (RW * RH) / WG, runs = WG % RW === 0 && (WG / RW) * PPL === RH;
    const nms = runs ? `  {
    uint cx = li % ${RW}u;
    uint y0 = (li / ${RW}u) * ${PPL}u;
${Array.from({ length: PPL + 2 }, (_, rr) => `    vec3 m${rr} = vec3(mapP((y0 + ${rr}u) * ${MW}u + cx), mapP((y0 + ${rr}u) * ${MW}u + cx + 1u), mapP((y0 + ${rr}u) * ${MW}u + cx + 2u));`).join("\n")}
${Array.from({ length: PPL }, (_, p) => `    peakAt(float[9](m${p}.x, m${p}.y, m${p}.z, m${p + 1}.x, m${p + 1}.y, m${p + 1}.z, m${p + 2}.x, m${p + 2}.y, m${p + 2}.z), (y0 + ${p + 1}u) * ${MW}u + cx + 1u, ((y0 + ${p}u) / ${TILE}u) * ${TX}u + cx / ${TILE}u, rx0 + int(cx), ry0 + int(y0) + ${p}, w, h);`).join("\n")}
  }` : `  [[unroll]] for (uint r = 0u; r < ${PPL}u; r++) {
    uint q = li + r * ${WG}u;
    uint px = q % ${RW}u + 1u;
    uint py = q / ${RW}u + 1u;
    uint cc = py * ${MW}u + px;
    float nb[9] = float[9](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    [[unroll]] for (uint k = 0u; k < 9u; k++) { nb[k] = mapP(cc + (k / 3u) * ${MW}u + (k % 3u) - ${MW}u - 1u); }
    peakAt(nb, cc, ((py - 1u) / ${TILE}u) * ${TX}u + (px - 1u) / ${TILE}u, rx0 + int(px) - 1, ry0 + int(py) - 1, w, h);
  }`;
    const body = `${xload}${layers}
${nms}
  barrier();
  // Each tile's KEEP strongest by rank (ties by position, so ranks are distinct) into its slots; empty slots stay as
  // the clear left them, response 0. An item is (tile, place in its list).
  [[unroll]] for (uint it = li; it < ${NT * LOCAL}u; it += ${WG}u) {
    uint t = it / LOCAL;
    uint j = it % LOCAL;
    uint tx = rxi * ${TX}u + t % ${TX}u;
    uint ty = wg.y * ${TY}u + t / ${TX}u;
    uint n = ${wgLoad("nfound[t]")};
    uint kept = min(n, LOCAL);
    if (j == 0u && n > 0u) { ${devAdd("counts[f * 16u + 1u]", "n")}; }
    // ai: Peaks past LOCAL were never stored, so which LOCAL a full tile keeps is the lanes' order; count them.
    if (j == 0u && n > LOCAL) { ${devAdd(`counts[f * 16u + ${OVERFLOW}u]`, "n - LOCAL")}; }
    if (j < kept && tx < P.tilesX) {
      vec4 e = ${peakLoad("t * LOCAL + j")};
      uint rank = 0u;
      for (uint q = 0u; q < kept; q++) {
        float o = ${peakResp("t * LOCAL + q")};
        if (o > e.w || (o == e.w && q < j)) { rank++; }
      }
      if (rank < KEEP) { raw[f * P.slots + P.slotBase + (ty * P.tilesX + tx) * KEEP + rank] = e; }
    }
  }
`;
    netCode = `  [[dont_unroll]] for (uint rr = 0u; rr < WALK; rr++) {
    uint rxi = wg.x * WALK + rr;
    int rx0 = int(rxi) * RW;
    int xo0 = rx0 - HALO;
    // The same for every lane (the frame's width and the loop count), so the loop's barriers stay uniform.
    if (rx0 >= w) { break; }
    if (li < ${NT}u) { ${wgStore("nfound[li]", "0u")}; }
${body}    barrier();
  }
`;
  }

  const planeElem = halves || int8 ? "uint" : "float";
  // ai: The pools' element: 16 bytes, or 8 in the int8 network (fcn2Layout).
  const EL = int8 && net ? "uvec2" : "uvec4";
  const bankU = "layout(std140) uniform Bank { uint level; uint stride; uint rows; uint slots; float tau; float eps; uint slotBase; uint tilesX; } P;";
  const bind = !net
    ? `layout(set = 0, binding = 0, std430) readonly buffer Lev { float lev[]; };
layout(set = 0, binding = 1, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 2, std430) buffer Plane { ${planeElem} xpl[]; };
layout(set = 0, binding = 3, ${bankU.slice("layout(".length)}`
    : `layout(set = 0, binding = 0, std430) readonly buffer Plane { ${planeElem} xpl[]; };
layout(set = 0, binding = 1, std430) readonly buffer Frames { Frame frames[]; };
layout(set = 0, binding = 2, std430) buffer Raw { vec4 raw[]; };
layout(set = 0, binding = 3, std430) buffer Counts { uint counts[]; };
layout(set = 0, binding = 4, ${bankU.slice("layout(".length)}`;
  const what = !net ? `the X pass, ${XB[0]} x ${XB[1]} pixels a workgroup${int8 ? ", Xq out" : ""}` : `${P.macs} MACs a map pixel, ${int8 ? `int8 codes of ${Q.bits} bits` : `${f16 ? "f16" : "f32"}${packed ? " with halves stored" : ""}`}, ${TX} x ${TY} tiles a workgroup, ${int8 ? "Xq" : "X"} from the plane`;
  const origin = !net
    ? `  int xo0 = int(wg.x) * ${XB[0]};
  int yo0 = int(wg.y) * ${XB[1]};
  if (F.valid != 1u || xo0 >= w || yo0 >= h) { return; }`
    : `  int ry0 = int(wg.y) * RH;
  int yo0 = ry0 - HALO;
  // A whole workgroup returns together, so the early exit keeps control flow uniform.
  if (F.valid != 1u || int(wg.x * WALK) * RW >= w || ry0 >= h) { return; }
${Array.from({ length: Math.ceil((Lz.XH * Lz.XE) / WG) }, (_, r) => `  ${int8 ? "uint" : "vec4"} xp${r} = ${int8 ? "0u" : "vec4(0.0)"};`).join("\n")}
  {
    int xo0 = int(wg.x * WALK) * RW - HALO;
${Array.from({ length: Math.ceil((Lz.XH * Lz.XE) / WG) }, (_, r) => `    if (li + ${r * WG}u < ${Lz.XH * Lz.XE}u) { xp${r} = xfetch(li + ${r * WG}u, xo0, yo0, w, h, f); }`).join("\n")}
  }`;
  return VERSION + (net ? UNROLL + ATOMICS : "") + (f16 && net ? F16 : "") + (int8 && net ? DOT : "") + DIMS + /* glsl */ `
// slotBase: where this level's tiles start in the frame's slots; tilesX: tiles a row at this level; slots: a frame's
// slots over every level. The bank contract's uniform (the WGSL bank.mjs (archived)); eps and the floor are baked in.
layout(local_size_x = ${WG}, local_size_y = 1, local_size_z = 1) in;
${bind}
const int HALO = ${HALO};
const uint KEEP = ${KEEP}u;
const uint LOCAL = ${LOCAL}u;
const float KAPPA = ${f32(KAPPA)};
const float NOMINAL = ${f32(P.nominal)};
const float EPS = ${f32(P.eps)};
const float FLOOR = ${f32(P.floor)};${int8 ? `
const float QMAX = ${Q.qmax}.0;
const float INV = ${exactF32(Q.inv)};` : ""}
// ${what}, box ${K}: ${bytes} bytes of workgroup memory.
shared ${EL} poolA[${nA}];
shared ${EL} poolB[${nB}];
${!net ? `const uint XE = ${Lz.XE}u;\n` : netConsts}${int8 ? PACK4X_I8 : ""}${planeFns}
${statsFns}${netFns}${planePut}

void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.z;
  Frame F = frames[f];
  int w = int((F.w + (1u << P.level) - 1u) >> P.level);
  int h = int((F.h + (1u << P.level) - 1u) >> P.level);
${origin}
${statsCode}${netCode}}
`;
}

// ai: The int8 network's GLSL for fcn2Source (stage net; Q the file's codes from readQuant, Lz fcn2Layout's int8
// ai: layout), the WGSL bank_fcn2.mjs (archived) int8Net's: { layers (the convolutions and the fused head, inside the walk loop),
// ai: fns (the X words, the peak list, rq), map (one element a pixel), xfetch, xPut (Xq words) }.
function int8Net(Q, Lz) {
  const { HALO, G1, convs } = Lz;
  // ai: G is a conv's geometry (Lz.convs), C its codes (Q.convs). A pixel's eight codes are a uvec2, channels 0 to 3
  // ai: in x (channel 0 the low byte), 4 to 7 in y.
  const acc = (a, lo) => `ivec4(${[0, 1, 2, 3].map((j) => `${a}${lo + j}`).join(", ")})`;
  const mul = (C, lo) => vec4Exact(C.mul.slice(lo, lo + 4));
  // ai: conv1 (one input channel, dilation 1): four output pixels a lane. A tap row's word t<j> is X pixels x0 + j ..
  // ai: x0 + j + 3, cut from the row's words gk and gk + 1; its weight word holds the row's three taps in bytes 0 to 2
  // ai: and 0 in byte 3.
  const conv1Q = (G, C) => {
    const N = G1 * G.outH, lines = [];
    for (let j = 0; j < 4; j++) for (let o = 0; o < C.cout; o++) lines.push(`      int a${j}_${o} = ${C.bq[o]};`);
    for (let ky = 0; ky < 3; ky++) {
      lines.push(`      {
        uint w0 = xr((y + ${ky}u) * RS + gk);
        uint w1 = xr((y + ${ky}u) * RS + gk + 1u);
        uint t0 = w0;
        uint t1 = (w0 >> 8u) | (w1 << 24u);
        uint t2 = (w0 >> 16u) | (w1 << 16u);
        uint t3 = (w0 >> 24u) | (w1 << 8u);`);
      for (let j = 0; j < 4; j++) for (let o = 0; o < C.cout; o++) {
        const W = packI8x4(...[0, 1, 2].map((kx) => C.wq[o * 9 + ky * 3 + kx]), 0);
        if (W) lines.push(`        a${j}_${o} += dotPacked4x8EXT(int(t${j}), ${intWord(W)});`);
      }
      lines.push("      }");
    }
    const st = Array.from({ length: 4 }, (_, j) => `      if (x0 + ${j}u < ${G.outW}u) {
        ivec4 q0 = rq(${acc(`a${j}_`, 0)}, ${mul(C, 0)});
        ivec4 q1 = rq(${acc(`a${j}_`, 4)}, ${mul(C, 4)});
        poolA[y * ${G.outW}u + x0 + ${j}u] = uvec2(pack4xI8(q0), pack4xI8(q1));
      }`);
    return `    [[dont_unroll]] for (uint i = li; i < ${N}u; i += ${WG}u) {
      uint gk = i % ${G1}u;
      uint y = i / ${G1}u;
      uint x0 = gk * 4u;
${lines.join("\n")}
${st.join("\n")}
    }
    barrier();
`;
  };
  // ai: conv2 and conv3: a pixel a lane, one load a tap, each output channel two dot products (input channels 0 to
  // ai: 3, then 4 to 7), each added to the channel's accumulator in a statement of its own. conv2 requantises into
  // ai: registers and is written over conv1's codes after a barrier (staged); conv3 feeds the head.
  const convQ = (G, C, k) => {
    const last = k === convs.length - 1, N = G.outW * G.outH, d = G.dil, taps = [];
    for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const body = [];
      for (let o = 0; o < C.cout; o++) {
        const terms = [0, 1].map((g) => [g, packI8x4(...[0, 1, 2, 3].map((j) => C.wq[((o * C.cin + 4 * g + j) * 3 + ky) * 3 + kx]))]).filter(([, W]) => W).map(([g, W]) => `dotPacked4x8EXT(int(e.${"xy"[g]}), ${intWord(W)})`);
        for (const t of terms) body.push(`        a${o} += ${t};`);
      }
      if (body.length) taps.push(`      {\n        uvec2 e = poolA[(y + ${ky * d}u) * ${G.inW}u + x + ${kx * d}u];\n${body.join("\n")}\n      }`);
    }
    const init = Array.from({ length: C.cout }, (_, o) => `      int a${o} = ${C.bq[o]};`).join("\n");
    if (last) {
      // ai: The head from its bias, channel 0 first, one fma a channel (q8.mjs forwardCodes' order, in f32).
      const H = Q.head, chain = (row) => Array.from({ length: H.cin }, (_, c) => c).reduce((z, c) => `fma(${exactF32(H.w[row * H.cin + c])}, h${c >> 2}.${"xyzw"[c & 3]}, ${z})`, exactF32(H.b[row]));
      return `    [[dont_unroll]] for (uint i = li; i < ${N}u; i += ${WG}u) {
      uint x = i % ${G.outW}u;
      uint y = i / ${G.outW}u;
${init}
${taps.join("\n")}
      vec4 h0 = max(vec4(${acc("a", 0)}) * ${mul(C, 0)}, vec4(0.0));
      vec4 h1 = max(vec4(${acc("a", 4)}) * ${mul(C, 4)}, vec4(0.0));
      float zl = ${chain(0)};
      float sv = ${chain(1)};
      mapPut(i, 1.0 / (1.0 + exp(-zl)), sv);
    }
    barrier();
`;
    }
    const NR = Math.ceil(N / WG);
    return `${Array.from({ length: NR }, (_, r) => `    uvec2 o${k}_${r} = uvec2(0u);`).join("\n")}
    [[dont_unroll]] for (uint r = 0u; r < ${NR}u; r++) {
      uint i = li + r * ${WG}u;
      if (i < ${N}u) {
      uint x = i % ${G.outW}u;
      uint y = i / ${G.outW}u;
${init}
${taps.join("\n")}
      ivec4 q0 = rq(${acc("a", 0)}, ${mul(C, 0)});
      ivec4 q1 = rq(${acc("a", 4)}, ${mul(C, 4)});
      uvec2 st = uvec2(pack4xI8(q0), pack4xI8(q1));
${Array.from({ length: NR }, (_, r) => `      if (r == ${r}u) { o${k}_${r} = st; }`).join("\n")}
      }
    }
    barrier();
${Array.from({ length: NR }, (_, r) => `    if (li + ${r * WG}u < ${N}u) { poolA[li + ${r * WG}u] = o${k}_${r}; }`).join("\n")}
    barrier();
`;
  };
  // ai: xfetch: an item is one word of four Xq. The window starts HALO before a region (a multiple of four), so where
  // ai: HALO is not one either a word inside the level is cut from two plane words.
  const cut = 8 * (((-HALO % 4) + 4) % 4);
  const xfetch = `uint xfetch(uint it, int xo0, int yo0, int w, int h, uint f) {
  uint jr = it / XE;
  uint g = it % XE;
  int x0 = xo0 + int(g) * 4;
  int y = clamp(yo0 + int(jr), 0, h - 1);
  if (x0 >= 0 && x0 + 3 < w) {
    uint i = (planeRow(f, y) + uint(x0)) >> 2u;
    return ${cut ? `(xpl[i] >> ${cut}u) | (xpl[i + 1u] << ${32 - cut}u)` : "xpl[i]"};
  }
  uint v = 0u;
  [[unroll]] for (int k = 0; k < 4; k++) { v |= planeByte(f, clamp(x0 + k, 0, w - 1), y) << (8u * uint(k)); }
  return v;
}
`;
  const fns = `
// ai: The X window's Xq words, four pixels a u32, two an element of B.
void xw(uint i, uint v) { poolB[i >> 1u][i & 1u] = v; }
uint xr(uint i) { return poolB[i >> 1u][i & 1u]; }
// ai: A peak (x, y, sigma, response) is two elements of A.
void peakPut(uint i, vec4 v) { poolA[2u * i] = floatBitsToUint(v.xy); poolA[2u * i + 1u] = floatBitsToUint(v.zw); }
vec4 peakGet(uint i) { return vec4(uintBitsToFloat(poolA[2u * i]), uintBitsToFloat(poolA[2u * i + 1u])); }
float peakW(uint i) { return uintBitsToFloat(poolA[2u * i + 1u].y); }
// ai: The contract's requantisation (cnn/q8.mjs requant) of four channels' accumulators, the bias in: one f32
// ai: multiply, rounded half to even, clamped to the codes 0 .. QMAX.
ivec4 rq(ivec4 a, vec4 m) { return ivec4(clamp(roundEven(vec4(a) * m), vec4(0.0), vec4(QMAX))); }`;
  const xPut = `// ai: X columns xo0 + 4g .. + 3 of window row jr, as Xq, into the X pool (RS words a row, the last ones zero).
void xPut(uint jr, uint g, uint xv, int xo0, int yo0, int w, int h, uint f) {
  xw(jr * RS + g, xv);
  if (g == XE - 1u) { [[unroll]] for (uint e = XE; e < RS; e++) { xw(jr * RS + e, 0u); } }
}
`;
  // ai: The map: one 8-byte element a pixel.
  const map = `void mapPut(uint i, float z, float s) { poolB[i] = uvec2(floatBitsToUint(z), floatBitsToUint(s)); }
float mapP(uint i) { return uintBitsToFloat(poolB[i].x); }
float mapS(uint i) { return uintBitsToFloat(poolB[i].y); }`;
  const layers = conv1Q(convs[0], Q.convs[0]) + convs.slice(1).map((L, k) => convQ(L, Q.convs[k + 1], k + 1)).join("");
  return { layers, fns, map, xfetch, xPut };
}
