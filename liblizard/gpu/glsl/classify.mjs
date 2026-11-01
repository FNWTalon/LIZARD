// F4, the classifier, in GLSL: the WGSL classify_gemm.mjs (archived) classifySource statement for statement (its comments say
// how the convolutions are tiled, what the int8 form computes and why each piece is as it is), its float forms (f16 and
// f32) and its int8 form, with the same plan, weight layout, bindings and dispatch (ceil(cap / P) workgroups a frame).
// ai: The plan is the WGSL module's (classifyPlan): patches a workgroup, tiles, buffers and scratch. WGSL's select is
// ai: GLSL's mix with a bool selector (both operands made, one OpSelect); a bitcast of u32 words to f16 is
// ai: unpackFloat2x16, to f32 uintBitsToFloat; dot4I8Packed is dotPacked4x8EXT (GL_EXT_integer_dot_product, the
// ai: OpSDot naga writes); pack4xI8 is the low bytes shifted together, as gen/gen.mjs rewrites it for naga (the Adreno
// ai: 840 got naga's 8-bit vector form wrong); round is roundEven. The measurement knobs the lab's WebGPU pages use
// ai: (stop, dump) are the WGSL's alone. Every workgroup variable is written before the barrier that precedes its
// ai: reads, so none is zeroed.
import { VERSION, SAMPLERLESS, F16, DOT, SHUFFLE, DIMS, CONVERT, LEVEL_BINDINGS, LEVELS, PACK4X_I8 } from "./common.mjs";
import { KAPPA, PATCH } from "../cnn/patch.mjs";
import { classifyPlan, BUDGET, exactF32 } from "../plan/classify_gemm.mjs";

const WG = 256;
const RED_MIN = 1024;
const TCF = 4;
// ai: the WGSL module's row pitch (pitchOf) and the int8 patch's (PITCH_PATCH_Q8)
const pitchOf = (S, pad) => S / 4 + (S >= (pad >= 2 ? 8 : pad === 1 ? PATCH : Infinity) ? 1 : 0);
const PITCH_PATCH_Q8 = PATCH / 16;
const sel = (f, t, c) => `mix(${f}, ${t}, ${c})`;
// ai: a select of two vectors by one bool (GLSL's mix wants a bool vector there): the conditional, on swizzles alone
const selv = (f, t, c) => `((${c}) ? ${t} : ${f})`;

// ai: A uvec4 `v` holds E weights; f32 expressions for them (the WGSL's elems).
const elems = (v, f16) => f16
  ? { code: `f16vec4 ${v}a = f16vec4(unpackFloat2x16(${v}.x), unpackFloat2x16(${v}.y)); f16vec4 ${v}b = f16vec4(unpackFloat2x16(${v}.z), unpackFloat2x16(${v}.w));`, at: ["xyzw".split("").map((c) => `float(${v}a.${c})`), "xyzw".split("").map((c) => `float(${v}b.${c})`)].flat() }
  : { code: `vec4 ${v}f = uintBitsToFloat(${v});`, at: "xyzw".split("").map((c) => `${v}f.${c}`) };
// ai: A block of TC consecutive weights from element `sub` of `v` (the WGSL's block).
function block(v, TC, f16, sub) {
  const E = f16 ? 8 : 4;
  if (TC === E) return elems(v, f16);
  if (f16) {
    if (TC === 4) return { code: `uvec2 ${v}w = ${selv(`${v}.xy`, `${v}.zw`, `${sub} == 4u`)}; f16vec4 ${v}a = f16vec4(unpackFloat2x16(${v}w.x), unpackFloat2x16(${v}w.y));`, at: "xyzw".split("").map((c) => `float(${v}a.${c})`) };
    const word = sel(sel(`${v}.x`, `${v}.y`, `(${sub} >> 1u) == 1u`), sel(`${v}.z`, `${v}.w`, `(${sub} >> 1u) == 3u`), `${sub} >= 4u`);
    if (TC === 2) return { code: `f16vec2 ${v}a = unpackFloat2x16(${word});`, at: [`float(${v}a.x)`, `float(${v}a.y)`] };
    return { code: `f16vec2 ${v}a = unpackFloat2x16(${word}); float16_t ${v}s = ${sel(`${v}a.x`, `${v}a.y`, `(${sub} & 1u) == 1u`)};`, at: [`float(${v}s)`] };
  }
  if (TC === 2) return { code: `vec2 ${v}a = uintBitsToFloat(${selv(`${v}.xy`, `${v}.zw`, `${sub} == 2u`)});`, at: [`${v}a.x`, `${v}a.y`] };
  return { code: `float ${v}s = uintBitsToFloat(${sel(sel(`${v}.x`, `${v}.y`, `${sub} == 1u`), sel(`${v}.z`, `${v}.w`, `${sub} == 3u`), `${sub} >= 2u`)});`, at: [`${v}s`] };
}
// ai: Element `idx` (a lane value, 0..E-1) of `v`, as f32 (the WGSL's elem).
const elem = (v, idx, f16) => {
  if (!f16) return `uintBitsToFloat(${sel(sel(`${v}.x`, `${v}.y`, `${idx} == 1u`), sel(`${v}.z`, `${v}.w`, `${idx} == 3u`), `${idx} >= 2u`)})`;
  const word = sel(sel(`${v}.x`, `${v}.y`, `(${idx} >> 1u) == 1u`), sel(`${v}.z`, `${v}.w`, `(${idx} >> 1u) == 3u`), `${idx} >= 4u`);
  return `float(${sel(`unpackFloat2x16(${word}).x`, `unpackFloat2x16(${word}).y`, `(${idx} & 1u) == 1u`)})`;
};

// ai: One float conv layer (the WGSL's convGemm).
function convGemm(L, tile, f16, src, SS, dst, DS, pin, pout, lw, subgroups = false, unroll = 1) {
  const { TC, TX, KS, nX, NB } = tile, E = f16 ? 8 : 4, rowG = L.inS / 4, ciPer = L.inC / KS;
  const VT = f16 ? "f16vec4" : "vec4", ST = f16 ? "float16_t" : "float";
  const U = ciPer % unroll === 0 ? unroll : 1;
  const vecsPerBlock = Math.ceil(TC / E);
  const accs = [];
  for (let c = 0; c < TC; c++) for (let t = 0; t < TX; t++) accs.push(`a${c}_${t}`);
  const bias = [];
  for (let vb = 0; vb < vecsPerBlock; vb++) {
    const bl = block(`bv${vb}`, Math.min(TC, E), f16, "wsub");
    bias.push(`uvec4 bv${vb} = W[${lw.b}u + wvec + ${vb}u]; ${bl.code}`);
    for (let c = 0; c < Math.min(TC, E); c++) for (let t = 0; t < TX; t++) bias.push(`a${vb * E + c}_${t} = vec4(${bl.at[c]});`);
  }
  const loads = [], i0 = [], i1 = [], i2 = [];
  const cv = (e) => (f16 ? `vec4(${e} * okm)` : `(${e} * okm)`), cs = (e) => (f16 ? `float(${e} * okm)` : `(${e} * okm)`);
  if (L.stride === 1) {
    for (let t = 0; t < TX; t++) loads.push(`vec4 m${t} = ${cv(`${src}[r + x4 + ${t}u]`)};`);
    loads.push(`float lf = ${sel("0.0", cs(`${src}[r + max(x4, 1u) - 1u].w`), "x4 > 0u")};`);
    loads.push(`float rt = ${sel("0.0", cs(`${src}[r + min(x4 + ${TX}u, ${rowG - 1}u)].x`), `x4 + ${TX}u < ${rowG}u`)};`);
    for (let t = 0; t < TX; t++) {
      i0.push(`vec4(${t ? `m${t - 1}.w` : "lf"}, m${t}.xyz)`); i1.push(`m${t}`); i2.push(`vec4(m${t}.yzw, ${t < TX - 1 ? `m${t + 1}.x` : "rt"})`);
    }
  } else {
    for (let t = 0; t < TX; t++) loads.push(`vec4 m${t}a = ${cv(`${src}[r + 2u * (x4 + ${t}u)]`)}; vec4 m${t}b = ${cv(`${src}[r + 2u * (x4 + ${t}u) + 1u]`)};`);
    loads.push(`float lf = ${sel("0.0", cs(`${src}[r + 2u * max(x4, 1u) - 1u].w`), "x4 > 0u")};`);
    for (let t = 0; t < TX; t++) {
      i0.push(`vec4(${t ? `m${t - 1}b.w` : "lf"}, m${t}a.y, m${t}a.w, m${t}b.y)`); i1.push(`vec4(m${t}a.x, m${t}a.z, m${t}b.x, m${t}b.z)`); i2.push(`vec4(m${t}a.y, m${t}a.w, m${t}b.y, m${t}b.w)`);
    }
  }
  const ins = [i0, i1, i2].map((iv, kx) => iv.map((e, t) => `vec4 i${kx}_${t} = ${e};`).join(" ")).join("\n          ");
  const wcode = [], fmas = [];
  for (let kx = 0; kx < 3; kx++) {
    for (let vb = 0; vb < vecsPerBlock; vb++) {
      const bl = block(`w${kx}_${vb}`, Math.min(TC, E), f16, "wsub");
      wcode.push(`uvec4 w${kx}_${vb} = W[wb + ${kx * lw.kxVecs + vb}u]; ${bl.code}`);
      for (let c = 0; c < Math.min(TC, E); c++) for (let t = 0; t < TX; t++) fmas.push(`a${vb * E + c}_${t} = fma(vec4(${bl.at[c]}), i${kx}_${t}, a${vb * E + c}_${t});`);
    }
  }
  const outAt = (c, t) => `${dst}[dst0 + ((cbE + ${c}u) * ${L.outS}u + y) * ${pout}u + x4 + ${t}u]`;
  const finish = accs.map((a, k) => `${outAt(Math.floor(k / TX), k % TX)} = ${VT}(max(${a}, vec4(0.0)));`).join("\n      ");
  const per = 4 * TC * TX, tilesN = WG / KS;
  const partAt = (s, k) => `red[((${s} - 1u) * ${tilesN}u + ti) * ${per}u + ${k}u]`;
  const reduce = KS === 1 ? finish : subgroups ? /* glsl */ `
    ${[1, 2].filter((m) => m < KS).map((m) => accs.map((a) => `${a} += subgroupShuffleXor(${a}, ${m}u);`).join(" ")).join("\n    ")}
    if (ks == 0u) {
      ${finish}
    }` : /* glsl */ `
    uint ti = li / ${KS}u;
    if (ks > 0u) {
      ${accs.map((a, k) => [0, 1, 2, 3].map((e) => `${partAt("ks", 4 * k + e)} = ${a}[${e}];`).join(" ")).join("\n      ")}
    }
    barrier();
    if (ks == 0u) {
      ${Array.from({ length: KS - 1 }, (_, s) => accs.map((a, k) => `${a} += vec4(${[0, 1, 2, 3].map((e) => partAt(`${s + 1}u`, 4 * k + e)).join(", ")});`).join(" ")).join("\n      ")}
      ${finish}
    }`;
  return /* glsl */ `
  // ${L.name}: ${L.inC} x ${L.inS}^2 -> ${L.outC} x ${L.outS}^2 stride ${L.stride}; tile ${TC} ch x ${4 * TX} px, ${KS} slice${KS > 1 ? "s" : ""}
  {
    uint ks = li % ${KS}u;
    uint r0 = li / ${KS}u;
    uint xg = r0 % ${nX}u;
    uint r1 = r0 / ${nX}u;
    uint y = r1 % ${L.outS}u;
    uint r2 = r1 / ${L.outS}u;
    uint cb = r2 % ${NB}u;
    uint p = r2 / ${NB}u;
    uint cbE = cb * ${TC}u;
    uint wvec = cbE / ${E}u;
    uint wsub = cbE % ${E}u;
    uint x4 = xg * ${TX}u;
    uint src0 = p * ${SS}u;
    uint dst0 = p * ${DS}u;
    ${accs.map((a) => `vec4 ${a} = vec4(0.0);`).join(" ")}
    if (ks == 0u) { ${bias.join(" ")} }
    for (uint ci0 = ks * ${ciPer}u; ci0 < (ks + 1u) * ${ciPer}u; ci0 += ${U}u) {
      ${Array.from({ length: U }, (_, uu) => [0, 1, 2].map((ky) => /* glsl */ `
      {
        uint ci = ci0 + ${uu}u;
        int iy = int(y) * ${L.stride} - 1 + ${ky};
        ${ST} okm = ${sel(`${ST}(0.0)`, `${ST}(1.0)`, `iy >= 0 && iy < ${L.inS}`)};
        uint r = src0 + (ci * ${L.inS}u + uint(clamp(iy, 0, ${L.inS - 1}))) * ${pin}u;
        ${loads.join(" ")}
        ${ins}
        uint wb = ${lw.w}u + ((ci * 3u + ${ky}u) * 3u) * ${lw.kxVecs}u + wvec;
        ${wcode.join("\n        ")}
        ${fmas.join("\n        ")}
      }`).join("")).join("")}
    }
    ${reduce}
  }
  barrier();
`;
}

// ai: The first fc layer as a GEMM over the last conv's buffer (the WGSL's fc1Gemm).
function fc1Gemm(L, plan, src, SS, pitch, lw, S) {
  const { NB, KSf, slice } = plan.fc1, E = plan.layout.E, f16 = plan.f16, steps = slice / E, out = plan.fcOut[L.name];
  const vecAt = (k) => `src0 + ((${k} / ${S * S}u) * ${S}u + (${k} % ${S * S}u) / ${S}u) * ${pitch}u + (${k} % ${S}u) / 4u`;
  const xs = [];
  for (let q = 0; q < E / 4; q++) xs.push(`uint k${q} = k + ${4 * q}u; vec4 x${q} = ${f16 ? "vec4" : ""}(${src}[${vecAt(`k${q}`)}]);`);
  const wl = [], fm = [];
  for (let c = 0; c < TCF; c++) {
    const bl = elems(`w${c}`, f16);
    wl.push(`uvec4 w${c} = W[${lw.w}u + (o0 + ${c}u) * ${lw.rowVecs}u + kv]; ${bl.code}`);
    for (let q = 0; q < E / 4; q++) fm.push(`f${c} += dot(x${q}, vec4(${bl.at.slice(4 * q, 4 * q + 4).join(", ")}));`);
  }
  return /* glsl */ `
  // ${L.name}: ${L.inN} -> ${L.outN}; lane = (patch, ${TCF} outputs, slice of ${slice} inputs), ${KSf} slices
  {
    uint ks = li % ${KSf}u;
    uint r0 = li / ${KSf}u;
    uint ob = r0 % ${NB}u;
    uint p = r0 / ${NB}u;
    uint src0 = p * ${SS}u;
    uint o0 = ob * ${TCF}u;
    ${Array.from({ length: TCF }, (_, c) => `float f${c} = 0.0;`).join(" ")}
    for (uint v = 0u; v < ${steps}u; v++) {
      uint k = ks * ${slice}u + v * ${E}u;
      uint kv = k / ${E}u;
      ${xs.join(" ")}
      ${wl.join("\n      ")}
      ${fm.join(" ")}
    }
    ${Array.from({ length: TCF }, (_, c) => `red[li * ${TCF}u + ${c}u] = f${c};`).join(" ")}
  }
  barrier();
  if (li < ${plan.P * L.outN}u) {
    uint p = li / ${L.outN}u;
    uint o = li % ${L.outN}u;
    uvec4 bv = W[${lw.b}u + o / ${E}u];
    float acc = ${elem("bv", `(o % ${E}u)`, f16)};
    for (uint s = 0u; s < ${KSf}u; s++) { acc += red[((p * ${NB}u + o / ${TCF}u) * ${KSf}u + s) * ${TCF}u + o % ${TCF}u]; }
    red[${out}u + li] = max(acc, 0.0);
  }
  barrier();
`;
}

// ai: A later fc layer, a lane an output of a patch (the WGSL's fcPlain).
function fcPlain(L, plan, inAt, lw, relu) {
  const E = plan.layout.E, f16 = plan.f16, out = plan.fcOut[L.name], steps = Math.ceil(L.inN / E);
  const bl = elems("wv", f16);
  const terms = bl.at.map((w, e) => `if (${e}u < ${L.inN}u - k) { acc += ${w} * red[${inAt}u + p * ${L.inN}u + k + ${e}u]; }`);
  return /* glsl */ `
  // ${L.name}: ${L.inN} -> ${L.outN}
  if (li < ${plan.P * L.outN}u) {
    uint p = li / ${L.outN}u;
    uint o = li % ${L.outN}u;
    uvec4 bv = W[${lw.b}u + o / ${E}u];
    float acc = ${elem("bv", `(o % ${E}u)`, f16)};
    for (uint v = 0u; v < ${steps}u; v++) {
      uint k = v * ${E}u;
      uvec4 wv = W[${lw.w}u + o * ${lw.rowVecs}u + v]; ${bl.code}
      ${terms.join(" ")}
    }
    red[${out}u + li] = ${relu ? "max(acc, 0.0)" : "acc"};
  }
  barrier();
`;
}


// ai: The float stage's parts (the WGSL's partsFloat).
function partsFloat(plan, { lw, subgroups, unroll, G }) {
  const P = plan.P, f16 = plan.f16, pitch32 = pitchOf(PATCH, plan.pad), VT = f16 ? "f16vec4" : "vec4";
  let convs = "";
  plan.convs.forEach((L, k) => {
    const [src, SS, dst, DS] = k % 2 === 0 ? ["bufX", plan.XS, "bufY", plan.YS] : ["bufY", plan.YS, "bufX", plan.XS];
    convs += convGemm(L, plan.tiles[k], f16, src, SS, dst, DS, pitchOf(L.inS, plan.pad), pitchOf(L.outS, plan.pad), lw(L.name), subgroups, unroll);
  });
  const last = plan.convs[plan.convs.length - 1], lastBuf = plan.convs.length % 2 ? ["bufY", plan.YS] : ["bufX", plan.XS];
  let fcs = fc1Gemm(plan.fcs[0], plan, lastBuf[0], lastBuf[1], pitchOf(last.outS, plan.pad), lw(plan.fcs[0].name), last.outS);
  for (let k = 1; k < plan.fcs.length; k++) fcs += fcPlain(plan.fcs[k], plan, plan.fcOut[plan.fcs[k - 1].name], lw(plan.fcs[k].name), k < plan.fcs.length - 1);
  const store = Array.from({ length: P }, (_, k) => `{ uint q = lq + ${G * k}u; bufX[cp * ${plan.XS}u + (q / 8u) * ${pitch32}u + (q % 8u)] = ${VT}((s${k} - vec4(mu)) / (sd + 0.02)); }`).join("\n  ");
  const decls = [
    `// ${plan.name}, ${f16 ? "f16" : "f32"}: ${P} patches a workgroup, ${plan.bytes} bytes of workgroup memory (X ${plan.XS}, Y ${plan.YS} vec4s a patch, ${plan.red} f32 scratch).`,
    `shared ${VT} bufX[${P * plan.XS}];`,
    `shared ${VT} bufY[${P * plan.YS}];`,
    `shared float red[${plan.red}];`,
  ].join("\n");
  return { directives: (f16 ? F16 : "") + (subgroups ? SHUFFLE : ""), decls, store, convs, fcs };
}

// ai: ---- The int8 form ----

// ai: The int8 helpers (the WGSL's Q8_FNS and gen/gen.mjs's pack4xI8 rewrite).
const Q8_FNS = /* glsl */ `
ivec4 d4(uvec4 v, uint w) { return ivec4(dotPacked4x8EXT(int(v.x), int(w)), dotPacked4x8EXT(int(v.y), int(w)), dotPacked4x8EXT(int(v.z), int(w)), dotPacked4x8EXT(int(v.w), int(w))); }
ivec4 d4l(uvec4 v, uint w) { return ivec4(0, dotPacked4x8EXT(int(v.y), int(w)), dotPacked4x8EXT(int(v.z), int(w)), dotPacked4x8EXT(int(v.w), int(w))); }
ivec4 d4r(uvec4 v, uint w) { return ivec4(dotPacked4x8EXT(int(v.x), int(w)), dotPacked4x8EXT(int(v.y), int(w)), dotPacked4x8EXT(int(v.z), int(w)), 0); }
ivec4 rq(ivec4 a, float m) { return f2i(clamp(roundEven(vec4(a) * m), vec4(0.0), vec4(QMAX))); }
${PACK4X_I8}uvec4 packq(ivec4 c0, ivec4 c1, ivec4 c2, ivec4 c3) {
  return uvec4(pack4xI8(ivec4(c0.x, c1.x, c2.x, c3.x)), pack4xI8(ivec4(c0.y, c1.y, c2.y, c3.y)), pack4xI8(ivec4(c0.z, c1.z, c2.z, c3.z)), pack4xI8(ivec4(c0.w, c1.w, c2.w, c3.w)));
}
`;

// ai: One int8 conv (the WGSL's convGemmQ8, the edge's products left out the same way).
function convGemmQ8(L, tile, src, SS, dst, DS, pin, pout, lw, subgroups) {
  const { TC, TX, KS, nX, NB } = tile, VB = TC / 4, X = "xyzw", first = L.inC === 1, rowG = L.inS / 4;
  const accs = [];
  for (let c = 0; c < TC; c++) for (let t = 0; t < TX; t++) accs.push(`a${c}_${t}`);
  const bias = [];
  for (let vb = 0; vb < VB; vb++) {
    bias.push(`ivec4 bv${vb} = ivec4(W[${lw.b}u + wvec + ${vb}u]);`);
    for (let c = 0; c < 4; c++) for (let t = 0; t < TX; t++) bias.push(`a${4 * vb + c}_${t} = ivec4(bv${vb}.${X[c]});`);
  }
  const dots = (tap, wv, dot = () => "d4") => {
    const out = [];
    for (let vb = 0; vb < VB; vb++) for (let c = 0; c < 4; c++) for (let t = 0; t < TX; t++) out.push(`a${4 * vb + c}_${t} += ${dot(t)}(${tap(t)}, ${wv(vb)}.${X[c]});`);
    return out.join("\n        ");
  };
  let rows;
  if (first) {
    const loads = TX === 1 ? `uvec4 v = ${src}[r + x4 / 2u] & uvec4(okm);
        bool hi = (x4 & 1u) == 1u;
        uint A0 = ${sel("v.x", "v.z", "hi")};
        uint B0 = ${sel("v.y", "v.w", "hi")};
        uint L0 = ${sel(sel("0u", `${src}[r + max(x4 / 2u, 1u) - 1u].w & okm`, "x4 > 0u"), "v.y", "hi")};`
      : `uvec4 v = ${src}[r + x4 / 2u] & uvec4(okm);
        uint A0 = v.x;
        uint B0 = v.y;
        uint A1 = v.z;
        uint B1 = v.w;
        uint L0 = ${sel("0u", `${src}[r + max(x4 / 2u, 1u) - 1u].w & okm`, "x4 > 0u")};`;
    const taps = Array.from({ length: TX }, (_, t) => `uvec4 t${t} = uvec4((${t ? `B${t - 1}` : "L0"} >> 24u) | (A${t} << 8u), A${t} >> 8u, (A${t} >> 24u) | (B${t} << 8u), B${t} >> 8u);`).join("\n        ");
    rows = [0, 1, 2].map((ky) => /* glsl */ `
      {
        int iy = int(y) * 2 - 1 + ${ky};
        uint okm = ${sel("0u", "0xffffffffu", `iy >= 0 && iy < ${L.inS}`)};
        uint r = src0 + uint(clamp(iy, 0, ${L.inS - 1})) * ${pin}u;
        ${loads}
        ${taps}
        ${Array.from({ length: VB }, (_, vb) => `uvec4 w${vb} = W[${lw.w + ky * lw.kxVecs}u + wvec + ${vb}u];`).join(" ")}
        ${dots((t) => `t${t}`, (vb) => `w${vb}`)}
      }`).join("");
  } else {
    const loads = [], ins = [[], [], []], edge = nX === 1;
    if (L.stride === 1) {
      for (let t = 0; t < TX; t++) loads.push(`uvec4 m${t} = ${src}[r + x4 + ${t}u] & uvec4(okm);`);
      if (!edge) loads.push(`uint lf = ${sel("0u", `${src}[r + max(x4, 1u) - 1u].w & okm`, "x4 > 0u")};`);
      if (!edge) loads.push(`uint rt = ${sel("0u", `${src}[r + min(x4 + ${TX}u, ${rowG - 1}u)].x & okm`, `x4 + ${TX}u < ${rowG}u`)};`);
      for (let t = 0; t < TX; t++) {
        ins[0].push(`uvec4(${t ? `m${t - 1}.w` : edge ? "0u" : "lf"}, m${t}.xyz)`); ins[1].push(`m${t}`); ins[2].push(`uvec4(m${t}.yzw, ${t < TX - 1 ? `m${t + 1}.x` : edge ? "0u" : "rt"})`);
      }
    } else {
      for (let t = 0; t < TX; t++) loads.push(`uvec4 m${t}a = ${src}[r + 2u * (x4 + ${t}u)] & uvec4(okm); uvec4 m${t}b = ${src}[r + 2u * (x4 + ${t}u) + 1u] & uvec4(okm);`);
      if (!edge) loads.push(`uint lf = ${sel("0u", `${src}[r + 2u * max(x4, 1u) - 1u].w & okm`, "x4 > 0u")};`);
      for (let t = 0; t < TX; t++) {
        ins[0].push(`uvec4(${t ? `m${t - 1}b.w` : edge ? "0u" : "lf"}, m${t}a.y, m${t}a.w, m${t}b.y)`); ins[1].push(`uvec4(m${t}a.x, m${t}a.z, m${t}b.x, m${t}b.z)`); ins[2].push(`uvec4(m${t}a.y, m${t}a.w, m${t}b.y, m${t}b.w)`);
      }
    }
    const dot = (kx) => (t) => (edge && kx === 0 && t === 0 ? "d4l" : edge && L.stride === 1 && kx === 2 && t === TX - 1 ? "d4r" : "d4");
    const taps = ins.map((iv, kx) => iv.map((e, t) => `uvec4 i${kx}_${t} = ${e};`).join(" ")).join("\n        ");
    const wts = [0, 1, 2].map((kx) => Array.from({ length: VB }, (_, vb) => `uvec4 w${kx}_${vb} = W[wb + ${kx * lw.kxVecs + vb}u];`).join(" ")).join("\n        ");
    const gPer = L.inC / 4 / KS;
    rows = /* glsl */ `
    for (uint g = ks * ${gPer}u; g < (ks + 1u) * ${gPer}u; g++) {${[0, 1, 2].map((ky) => /* glsl */ `
      {
        int iy = int(y) * ${L.stride} - 1 + ${ky};
        uint okm = ${sel("0u", "0xffffffffu", `iy >= 0 && iy < ${L.inS}`)};
        uint r = src0 + (g * ${L.inS}u + uint(clamp(iy, 0, ${L.inS - 1}))) * ${pin}u;
        ${loads.join(" ")}
        ${taps}
        uint wb = ${lw.w}u + ((g * 3u + ${ky}u) * 3u) * ${lw.kxVecs}u + wvec;
        ${wts}
        ${[0, 1, 2].map((kx) => dots((t) => `i${kx}_${t}`, (vb) => `w${kx}_${vb}`, dot(kx))).join("\n        ")}
      }`).join("")}
    }`;
  }
  const finish = [];
  for (let vb = 0; vb < VB; vb++) {
    finish.push(`vec4 mv${vb} = uintBitsToFloat(W[${lw.m}u + wvec + ${vb}u]);`);
    for (let t = 0; t < TX; t++) finish.push(`${dst}[dst0 + ((wvec + ${vb}u) * ${L.outS}u + y) * ${pout}u + x4 + ${t}u] = packq(${[0, 1, 2, 3].map((c) => `rq(a${4 * vb + c}_${t}, mv${vb}.${X[c]})`).join(", ")});`);
  }
  const fin = finish.join("\n      ");
  const per = 4 * TC * TX, tilesN = WG / KS;
  const partAt = (s, k) => `redi[((${s} - 1u) * ${tilesN}u + ti) * ${per}u + ${k}u]`;
  const reduce = KS === 1 ? fin : subgroups ? /* glsl */ `
    ${[1, 2].filter((m) => m < KS).map((m) => accs.map((a) => `${a} += subgroupShuffleXor(${a}, ${m}u);`).join(" ")).join("\n    ")}
    if (ks == 0u) {
      ${fin}
    }` : /* glsl */ `
    uint ti = li / ${KS}u;
    if (ks > 0u) {
      ${accs.map((a, k) => [0, 1, 2, 3].map((e) => `${partAt("ks", 4 * k + e)} = ${a}[${e}];`).join(" ")).join("\n      ")}
    }
    barrier();
    if (ks == 0u) {
      ${Array.from({ length: KS - 1 }, (_, s) => accs.map((a, k) => `${a} += ivec4(${[0, 1, 2, 3].map((e) => partAt(`${s + 1}u`, 4 * k + e)).join(", ")});`).join(" ")).join("\n      ")}
      ${fin}
    }`;
  return /* glsl */ `
  // ai: ${L.name} int8: ${L.inC} x ${L.inS}^2 -> ${L.outC} x ${L.outS}^2 stride ${L.stride}; tile ${TC} ch x ${4 * TX} px, ${KS} slice${KS > 1 ? "s" : ""}
  {
    uint ks = li % ${KS}u;
    uint r0 = li / ${KS}u;
    uint xg = r0 % ${nX}u;
    uint r1 = r0 / ${nX}u;
    uint y = r1 % ${L.outS}u;
    uint r2 = r1 / ${L.outS}u;
    uint cb = r2 % ${NB}u;
    uint p = r2 / ${NB}u;
    uint wvec = cb * ${VB}u;
    uint x4 = xg * ${TX}u;
    uint src0 = p * ${SS}u;
    uint dst0 = p * ${DS}u;
    ${accs.map((a) => `ivec4 ${a} = ivec4(0);`).join(" ")}
    if (ks == 0u) { ${bias.join(" ")} }
    ${rows}
    ${reduce}
  }
  barrier();
`;
}

// ai: fc1 int8 over the last conv's words (the WGSL's fc1GemmQ8).
function fc1GemmQ8(L, plan, src, SS, pitch, lw, S) {
  const { NB, KSf, slice } = plan.fc1, out = plan.fcOut[L.name], G4 = S / 4;
  const at = pitch === G4 ? "kv" : `(kv / ${G4}u) * ${pitch}u + kv % ${G4}u`;
  return /* glsl */ `
  // ai: ${L.name} int8: ${L.inN} codes -> ${L.outN} f32; lane = (patch, ${TCF} outputs, ${slice} vecs of 16 codes), ${KSf} slices
  {
    uint ks = li % ${KSf}u;
    uint r0 = li / ${KSf}u;
    uint ob = r0 % ${NB}u;
    uint p = r0 / ${NB}u;
    uint src0 = p * ${SS}u;
    uint o0 = ob * ${TCF}u;
    ${Array.from({ length: TCF }, (_, c) => `int f${c} = 0;`).join(" ")}
    for (uint v = 0u; v < ${slice}u; v++) {
      uint kv = ks * ${slice}u + v;
      uvec4 x = ${src}[src0 + ${at}];
      ${Array.from({ length: TCF }, (_, c) => `uvec4 w${c} = W[${lw.w}u + (o0 + ${c}u) * ${lw.rowVecs}u + kv];`).join("\n      ")}
      ${Array.from({ length: TCF }, (_, c) => `f${c} += dotPacked4x8EXT(int(x.x), int(w${c}.x)) + dotPacked4x8EXT(int(x.y), int(w${c}.y)) + dotPacked4x8EXT(int(x.z), int(w${c}.z)) + dotPacked4x8EXT(int(x.w), int(w${c}.w));`).join("\n      ")}
    }
    ${Array.from({ length: TCF }, (_, c) => `redi[li * ${TCF}u + ${c}u] = f${c};`).join(" ")}
  }
  barrier();
  if (li < ${plan.P * L.outN}u) {
    uint p = li / ${L.outN}u;
    uint o = li % ${L.outN}u;
    int acc = int(W[${lw.b}u + o / 4u][o % 4u]);
    for (uint s = 0u; s < ${KSf}u; s++) { acc += redi[((p * ${NB}u + o / ${TCF}u) * ${KSf}u + s) * ${TCF}u + o % ${TCF}u]; }
    red[${out}u + li] = max(float(acc) * uintBitsToFloat(W[${lw.m}u + o / 4u][o % 4u]), 0.0);
  }
  barrier();
`;
}

// ai: The int8 stage's parts (the WGSL's partsQ8, no dump).
function partsQ8(plan, offsets, arch, { lw, subgroups, G }) {
  const P = plan.P, q = offsets.quant, aq = arch.quant;
  if (!q || q.bits !== aq.bits || q.inv !== aq.input?.inv) throw new Error(`classify int8: weights packed with quant ${JSON.stringify(q)}, the arch says bits ${aq.bits} inv ${aq.input?.inv}`);
  const pp = PITCH_PATCH_Q8;
  let convs = "";
  plan.convs.forEach((L, k) => {
    const [src, SS, dst, DS] = k % 2 === 0 ? ["bufX", plan.XS, "bufY", plan.YS] : ["bufY", plan.YS, "bufX", plan.XS];
    const pin = k === 0 ? pp : L.inS / 4, pout = L.outS / 4;
    convs += convGemmQ8(L, plan.tiles[k], src, SS, dst, DS, pin, pout, lw(L.name), subgroups);
  });
  const last = plan.convs[plan.convs.length - 1], lastBuf = plan.convs.length % 2 ? ["bufY", plan.YS] : ["bufX", plan.XS];
  const [fc1, fc2] = plan.fcs;
  const fcs = fc1GemmQ8(fc1, plan, lastBuf[0], lastBuf[1], last.outS / 4, lw(fc1.name), last.outS) + fcPlain(fc2, plan, plan.fcOut[fc1.name], lw(fc2.name), false);
  const store = Array.from({ length: P }, (_, k) => `{ uint q = lq + ${G * k}u; bufX[cp * ${plan.XS}u + (q / 8u) * ${pp}u + (q % 8u) / 4u][q % 4u] = pack4xI8(f2i(clamp(roundEven(((s${k} - vec4(mu)) / (sd + 0.02)) * INV), vec4(-QMAX), vec4(QMAX)))); }`).join("\n  ");
  const decls = [
    `const float QMAX = ${q.qmax}.0;`,
    `const float INV = ${exactF32(q.inv)};`,
    `// ai: ${plan.name}, int8 codes of ${q.bits} bits: ${P} patches a workgroup, ${plan.bytes} bytes of workgroup memory (X ${plan.XS}, Y ${plan.YS} uvec4s a patch, ${plan.red} f32 and ${plan.redi} i32 scratch).`,
    `shared uvec4 bufX[${P * plan.XS}];`,
    `shared uvec4 bufY[${P * plan.YS}];`,
    `shared float red[${plan.red}];`,
    `shared int redi[${plan.redi}];`,
  ].join("\n") + Q8_FNS;
  return { directives: DOT + (subgroups ? SHUFFLE : ""), decls, store, convs, fcs };
}

// ai: The stage (the WGSL's classifySource; offsets from that module's packWeights, packWeightsF16 or packWeightsInt8).
export const classifySource = (offsets, arch, { f16 = false, int8 = false, budget = BUDGET, pad = 1, patches = 0, partials = RED_MIN, subgroups = false, unroll = 1 } = {}) => {
  const plan = classifyPlan(arch, { f16, int8, budget, pad, patches, partials, subgroups, unroll }), P = plan.P, G = WG / P;
  if (!offsets?.gemm || offsets.key !== plan.layout.key) throw new Error(`classify: weights packed for ${offsets?.key ?? "classify.mjs"}, the shader is ${plan.layout.key}; pack with classify_gemm.mjs packWeights${int8 ? "Int8" : ""}`);
  const lw = (name) => plan.layout.layers[name];
  const zAt = plan.fcOut[plan.fcs[plan.fcs.length - 1].name];
  const { directives, decls, store, convs, fcs } = int8 ? partsQ8(plan, offsets, arch, { lw, subgroups, G }) : partsFloat(plan, { lw, subgroups, unroll, G });
  // ai: the cut, once per pyramid level (the WGSL's cutAt and cut)
  const fetch = (l) => (l === 0 ? (x, y) => `texelFetch(img, ivec3(${x}, ${y}, int(f)), 0).r` : (x, y) => `lv${l}[${y === "iy0" ? "row0" : "row1"} + uint(${x})]`);
  const cutAt = (l) => Array.from({ length: P }, (_, k) => {
    const at = fetch(l);
    return /* glsl */ `
      {
        uint q = lq + ${G * k}u;
        float sy = float(q / 8u);
        float sx = float((q % 8u) * 4u);
        float py = (pk.y + u * (sy - 15.5)) * inv - 0.5;
        float fy = floor(py);
        float ay = py - fy;
        int jy = f2i(fy);
        int iy0 = clamp(jy, 0, lh - 1);
        int iy1 = clamp(jy + 1, 0, lh - 1);
        ${l ? `uint row0 = lbase + uint(iy0) * lsx; uint row1 = lbase + uint(iy1) * lsx;` : ""}
        ${[0, 1, 2, 3].map((t) => /* glsl */ `
        {
          float px = (pk.x + u * (sx + ${t}.0 - 15.5)) * inv - 0.5;
          float fx = floor(px);
          float ax = px - fx;
          int jx = f2i(fx);
          int ix0 = clamp(jx, 0, lw - 1);
          int ix1 = clamp(jx + 1, 0, lw - 1);
          s${k}[${t}] = mix(mix(${at("ix0", "iy0")}, ${at("ix1", "iy0")}, ax), mix(${at("ix0", "iy1")}, ${at("ix1", "iy1")}, ax), ay);
        }`).join("")}
        ssum += s${k}.x + s${k}.y + s${k}.z + s${k}.w;
        ssq += dot(s${k}, s${k});
      }`;
  }).join("");
  const cut = /* glsl */ `
  Frame F = frames[f];
  int lw = int((F.w + (1u << l) - 1u) >> l);
  int lh = int((F.h + (1u << l) - 1u) >> l);
  uvec4 ls = LD.d[l];
  uint lsx = ls.x;
  uint lbase = f * ls.y * lsx;
  float inv = 1.0 / float(1u << l);
  switch (l) {
    ${[0, 1, 2, 3].map((l) => `case ${l}u: {${cutAt(l)}
      break;
    }`).join("\n    ")}
    default: {${cutAt(4)}
    }
  }`;
  return VERSION + SAMPLERLESS + directives + DIMS + CONVERT + LEVEL_BINDINGS + LEVELS + /* glsl */ `
layout(local_size_x = ${WG}, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 7, std430) readonly buffer Peaks { vec4 peaks[]; };
layout(set = 0, binding = 8, std430) readonly buffer Counts { uint counts[]; };
layout(set = 0, binding = 9, std430) buffer Readings { vec4 readings[]; };
layout(set = 0, binding = 10, std140) uniform P4 { uint cap; uint p0; uint p1; uint p2; } P;
layout(set = 0, binding = 11, std430) readonly buffer Weights { uvec4 W[]; };
const float KAPPA = ${KAPPA};
const float PI = 3.14159265;
${decls}
void main() {
  uvec3 wg = gl_WorkGroupID;
  uint li = gl_LocalInvocationIndex;
  uint f = wg.y;
  if (frames[f].valid != 1u) { return; }
  uint n = min(counts[f * 16u + 2u], P.cap);
  uint j0 = wg.x * ${P}u;
  if (j0 >= n) { return; }
  uint cp = li / ${G}u;
  uint lq = li % ${G}u;
  uint jc = j0 + cp;
  uint ic = ${sel("0u", "jc", "jc < n")};
  vec4 pk = peaks[f * P.cap + ic];
  float u = pk.z / KAPPA;
  uint l = levelFor(u);
  ${Array.from({ length: P }, (_, k) => `vec4 s${k} = vec4(0.0);`).join(" ")}
  float ssum = 0.0;
  float ssq = 0.0;
  ${cut}
  red[li] = ssum;
  red[${WG}u + li] = ssq;
  barrier();
  if (li < 32u) {
    float t1 = 0.0;
    float t2 = 0.0;
    for (uint k = 0u; k < 8u; k++) { t1 += red[li * 8u + k]; t2 += red[${WG}u + li * 8u + k]; }
    red[${2 * WG}u + li] = t1;
    red[${2 * WG + 32}u + li] = t2;
  }
  barrier();
  float tsum = 0.0;
  float tsq = 0.0;
  for (uint k = 0u; k < ${G / 8}u; k++) { tsum += red[${2 * WG}u + cp * ${G / 8}u + k]; tsq += red[${2 * WG + 32}u + cp * ${G / 8}u + k]; }
  float mu = tsum / ${PATCH * PATCH}.0;
  float sd = sqrt(max(tsq / ${PATCH * PATCH}.0 - mu * mu, 0.0));
  ${store}
  barrier();
  ${convs}
  ${fcs}
  if (li < ${P}u) {
    uint j = j0 + li;
    if (j < n) {
      uint i = j;
      vec4 pw = peaks[f * P.cap + i];
      float uw = pw.z / KAPPA;
      float z0 = red[${zAt}u + li * 5u];
      float z1 = red[${zAt}u + li * 5u + 1u];
      float z2 = red[${zAt}u + li * 5u + 2u];
      float z3 = red[${zAt}u + li * 5u + 3u];
      float z4 = red[${zAt}u + li * 5u + 4u];
      float t = atan(z2, z1);
      float pr = 1.0 / (1.0 + exp(-z0));
      uint o3 = 3u * (f * P.cap + i);
      readings[o3] = vec4(pw.xy, uw * exp(z3), t - PI / 4.0);
      readings[o3 + 1u] = vec4(pr, 0.0, 0.0, 0.0);
      readings[o3 + 2u] = vec4(${sel("0.0", "1.0", "z4 > 0.0")}, pw.w, float(levelFor(uw)), 1.0);
    }
  }
}
`;
};
