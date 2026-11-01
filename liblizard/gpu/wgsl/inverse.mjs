// ai: The sender's inverse transforms (gpu/encoder.mjs; the native sender through gen/sender.mjs): the transform's
// ai: butterflies (below; the decoder's WGSL twin, back_transform.mjs, runs them forward), conj in and conj out,
// ai: unscaled, a symbol a slot, in f32.
// ai: IROWS, a workgroup a (disc row, symbol), n / 8 threads: the row's coefficients at (u + n) mod n, zeros
// ai:   elsewhere, the inverse along x, the row stored in Y's slot.
// ai: IPIC, a workgroup a (quad of columns, symbol), n / 8 threads: turn_pack's packing takes two columns through one
// ai:   complex inverse along y (Z(0) = 2 Re A(0) + 2i Re B(0), Z(v) = A + iB, Z(n - v) = conj A + i conj B, zero
// ai:   between), the real part column a and the imaginary column b, clipped to 0.5 + clamp(v, -lim, lim) 0.5 / lim
// ai:   (src/focus.c focus_encode, turn_pack, clip_row).
import { stagesOf } from "../back/disc.mjs";

// Complex helpers and the butterflies, on function-scope register arrays.
// ai: BUTTERFLIES, store, firstStage, laterStages and V8 are exported for the decoder's WGSL twin of the transform
// ai: (back_transform.mjs pass 1 and pass 2), which runs the same stages forward; natively the decoder's are GLSL
// ai: (../glsl/back_transform.mjs).
export const BUTTERFLIES = /* wgsl */ `
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn mneg_i(a: vec2f) -> vec2f { return vec2f(a.y, -a.x); }
const C8: f32 = 0.70710678118654752;
const S3: f32 = 0.86602540378443864676;
fn dft2(p: ptr<function, array<vec2f, 8>>, o: u32) {
  let a: vec2f = (*p)[o]; let b: vec2f = (*p)[o + 1u];
  (*p)[o] = a + b; (*p)[o + 1u] = a - b;
}
fn dft4(p: ptr<function, array<vec2f, 8>>, o: u32) {
  let p0: vec2f = (*p)[o]; let p1: vec2f = (*p)[o + 1u]; let p2: vec2f = (*p)[o + 2u]; let p3: vec2f = (*p)[o + 3u];
  let b0: vec2f = p0 + p2; let b2: vec2f = p0 - p2; let b1: vec2f = p1 + p3; let b3: vec2f = mneg_i(p1 - p3);
  (*p)[o] = b0 + b1; (*p)[o + 1u] = b2 + b3; (*p)[o + 2u] = b0 - b1; (*p)[o + 3u] = b2 - b3;
}
fn dft8(p: ptr<function, array<vec2f, 8>>) {
  let a0: vec2f = (*p)[0] + (*p)[4]; let a4: vec2f = (*p)[0] - (*p)[4];
  let a1: vec2f = (*p)[1] + (*p)[5]; let d1: vec2f = (*p)[1] - (*p)[5];
  let a2: vec2f = (*p)[2] + (*p)[6]; let a6: vec2f = mneg_i((*p)[2] - (*p)[6]);
  let a3: vec2f = (*p)[3] + (*p)[7]; let d3: vec2f = (*p)[3] - (*p)[7];
  let a5: vec2f = vec2f(C8 * (d1.x + d1.y), C8 * (d1.y - d1.x));
  let a7: vec2f = vec2f(C8 * (d3.y - d3.x), -C8 * (d3.x + d3.y));
  let b0: vec2f = a0 + a2; let b2: vec2f = a0 - a2; let b1: vec2f = a1 + a3; let b3: vec2f = mneg_i(a1 - a3);
  let c0: vec2f = a4 + a6; let c2: vec2f = a4 - a6; let c1: vec2f = a5 + a7; let c3: vec2f = mneg_i(a5 - a7);
  (*p)[0] = b0 + b1; (*p)[4] = b0 - b1; (*p)[2] = b2 + b3; (*p)[6] = b2 - b3;
  (*p)[1] = c0 + c1; (*p)[5] = c0 - c1; (*p)[3] = c2 + c3; (*p)[7] = c2 - c3;
}
`;

// The exchange buffer's element type and the conversions in and out of it (name: the workgroup array).
export const store = (prec) => ({
  ty: prec === "f16" ? "vec2<f16>" : "vec2f",
  ld: (name, i) => (prec === "f16" ? `vec2f(${name}[${i}])` : `${name}[${i}]`),
  st: (name, i, x) => (prec === "f16" ? `${name}[${i}] = vec2<f16>(${x});` : `${name}[${i}] = ${x};`),
});

// The stages after the first, for one or more FFTs side by side (lanes = [[registers, exchange buffer], ...]; the
// lanes share every barrier and every twiddle). Stage s at Ns reads buf[j + r n / R], twiddles by
// exp(-2 pi i r k / (Ns R)), k = j mod Ns, and writes buf[(j / Ns) Ns R + k + r Ns].
// Twiddles: the base twiddle from the table, the rest by multiplication (reading every one, or computing the base,
// measured slower, 2026-09-24).
export function laterStages(n, prec, lanes = [["v", "buf"]]) {
  const { ld, st } = store(prec);
  const T = n / 8, stages = stagesOf(n);
  let Ns = stages[0], out = "";
  for (let s = 1; s < stages.length; s++) {
    // ai: an odd radix: the last in place (radix3Stage as ever, oddLastStage for a 5), an earlier one through registers
    if (stages[s] === 3 || stages[s] === 5) {
      const R = stages[s];
      out += s < stages.length - 1 ? oddStage(n, R, Ns, prec, lanes) : R === 3 ? radix3Stage(n, prec, lanes) : oddLastStage(n, R, prec, lanes);
      Ns *= R; continue;
    }
    const R = stages[s], per = 8 / R, span = n / R, step = n / (Ns * R);
    out += `  // stage ${s + 1}: radix ${R}, Ns ${Ns}\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      for (const [v, buf] of lanes) for (let r = 0; r < R; r++) out += `  ${v}[${q * R + r}] = ${ld(buf, `${j} + ${r * span}u`)};\n`;
    }
    out += `  workgroupBarrier();\n`;
    for (let q = 0; q < per; q++) {
      const j = per === 1 ? "tid" : `(tid + ${q * T}u)`;
      out += `  {\n    let j: u32 = ${j};\n    let k: u32 = j & ${Ns - 1}u;\n`;
      out += `    let w1: vec2f = tw[k * ${step}u];\n`;
      if (R >= 4) out += `    let w2: vec2f = cmul(w1, w1);\n    let w3: vec2f = cmul(w2, w1);\n`;
      if (R === 8) out += `    let w4: vec2f = cmul(w2, w2);\n    let w5: vec2f = cmul(w4, w1);\n    let w6: vec2f = cmul(w3, w3);\n    let w7: vec2f = cmul(w4, w3);\n`;
      out += `    let d: u32 = (j - k) * ${R}u + k;\n`;
      for (const [v, buf] of lanes) {
        for (let r = 1; r < R; r++) out += `    ${v}[${q * R + r}] = cmul(${v}[${q * R + r}], w${r});\n`;
        out += R === 8 ? `    dft8(&${v});\n` : R === 4 ? `    dft4(&${v}, ${q * R}u);\n` : `    dft2(&${v}, ${q * R}u);\n`;
        for (let r = 0; r < R; r++) out += `    ${st(buf, `d + ${r * Ns}u`, `${v}[${q * R + r}]`)}\n`;
      }
      out += `  }\n`;
    }
    out += `  workgroupBarrier();\n`;
    Ns *= R;
  }
  return out;
}

// ai: The last stage when n has a factor of 3: Ns = n / 3, so it reads and writes the same three points,
// ai: buf[j + r n / 3], r < 3: in place, a thread taking butterflies j = tid, tid + T, ... below n / 3, T = n / 8. Its
// ai: twiddles are tw[r j] (step 1). Forward: out1 = a - t1 / 2 - i (sqrt 3 / 2) t2, out2 the conjugate turn, with
// ai: t1 = b + c and t2 = b - c after the twiddles.
function radix3Stage(n, prec, lanes) {
  const { ld, st } = store(prec);
  const T = n / 8, Ns = n / 3;
  let out = `  // stage: radix 3, Ns ${Ns}, in place\n  for (var j: u32 = tid; j < ${Ns}u; j += ${T}u) {\n`;
  out += `    let w1: vec2f = tw[j];\n    let w2: vec2f = tw[2u * j];\n`;
  for (const [, buf] of lanes) {
    out += `    {\n      let a: vec2f = ${ld(buf, "j")};\n      let b: vec2f = cmul(${ld(buf, `j + ${Ns}u`)}, w1);\n      let c: vec2f = cmul(${ld(buf, `j + ${2 * Ns}u`)}, w2);\n`;
    out += `      let t1: vec2f = b + c;\n      let t2: vec2f = mneg_i(b - c) * S3;\n      let m: vec2f = a - 0.5 * t1;\n`;
    out += `      ${st(buf, "j", "a + t1")}\n      ${st(buf, `j + ${Ns}u`, "m + t2")}\n      ${st(buf, `j + ${2 * Ns}u`, "m - t2")}\n    }\n`;
  }
  return out + `  }\n  workgroupBarrier();\n`;
}

// ai: The odd butterflies on R points p0 .. p(R-1) (twiddled; WGSL expressions), each output handed to put(r, expr):
// ai: radix 3 as radix3Stage's; radix 5 with t1 = p1 + p4, t2 = p2 + p3, t3 = p1 - p4, t4 = p2 - p3, out0 = p0 + t1 +
// ai: t2, m1 = p0 + c1 t1 + c2 t2, m2 = p0 + c2 t1 + c1 t2 (c1 = cos 2 pi / 5, c2 = cos 4 pi / 5), n1 = -i (s1 t3 + s2
// ai: t4), n2 = -i (s2 t3 - s1 t4) (s1 = sin 2 pi / 5, s2 = sin 4 pi / 5): out1 = m1 + n1, out4 = m1 - n1, out2 = m2 +
// ai: n2, out3 = m2 - n2. Forward (exp(-i)); the sender's inverse conjugates in and out (wgsl/inverse.mjs).
function oddButterfly(R, p, put, ind) {
  let out = `${ind}{\n`;
  for (let r = 0; r < R; r++) out += `${ind}  let p${r}: vec2f = ${p(r)};\n`;
  if (R === 3) {
    out += `${ind}  let t1: vec2f = p1 + p2;\n${ind}  let t2: vec2f = mneg_i(p1 - p2) * S3;\n${ind}  let m: vec2f = p0 - 0.5 * t1;\n`;
    out += `${ind}  ${put(0, "p0 + t1")}\n${ind}  ${put(1, "m + t2")}\n${ind}  ${put(2, "m - t2")}\n`;
  } else {
    out += `${ind}  let t1: vec2f = p1 + p4;\n${ind}  let t2: vec2f = p2 + p3;\n${ind}  let t3: vec2f = p1 - p4;\n${ind}  let t4: vec2f = p2 - p3;\n`;
    out += `${ind}  let m1: vec2f = p0 + 0.30901699437494742 * t1 - 0.80901699437494742 * t2;\n${ind}  let m2: vec2f = p0 - 0.80901699437494742 * t1 + 0.30901699437494742 * t2;\n`;
    out += `${ind}  let n1: vec2f = mneg_i(0.95105651629515357 * t3 + 0.58778525229247313 * t4);\n${ind}  let n2: vec2f = mneg_i(0.58778525229247313 * t3 - 0.95105651629515357 * t4);\n`;
    out += `${ind}  ${put(0, "p0 + t1 + t2")}\n${ind}  ${put(1, "m1 + n1")}\n${ind}  ${put(4, "m1 - n1")}\n${ind}  ${put(2, "m2 + n2")}\n${ind}  ${put(3, "m2 - n2")}\n`;
  }
  return out + `${ind}}\n`;
}

// ai: An odd stage before the last (radix R, 3 or 5, at Ns): Stockham as laterStages's stages (read buf[j + r n / R],
// ai: twiddle by exp(-2 pi i r k / (Ns R)), k = j mod Ns, write buf[(j - k) R + k + r Ns]), its n / R butterflies over
// ai: T = n / 8 threads, Q = ceil(8 / R) a thread (j = tid + q T below n / R), their points held in registers across
// ai: the barrier; each twiddle read whole from the table, tw[r k n / (Ns R)], as radix3Stage reads its own. Ns is
// ai: not a power of two after another odd stage (k by a modulo there).
function oddStage(n, R, Ns, prec, lanes) {
  const { ld, st } = store(prec);
  const T = n / 8, M = n / R, Q = Math.ceil(M / T), step = n / (Ns * R), pow2 = (Ns & (Ns - 1)) === 0, zero = Array(Q * R).fill("vec2f(0.0)").join(", ");
  const jOf = (q) => (q === 0 ? "tid" : `(tid + ${q * T}u)`), open = (q) => ((q + 1) * T > M ? `    if (${jOf(q)} < ${M}u) {\n` : `    {\n`);
  let out = `  // stage: radix ${R}, Ns ${Ns}\n  {\n`;
  for (const [v] of lanes) out += `    var ${v}g: array<vec2f, ${Q * R}> = array<vec2f, ${Q * R}>(${zero});\n`;
  for (let q = 0; q < Q; q++) {
    out += open(q);
    for (const [v, buf] of lanes) for (let r = 0; r < R; r++) out += `      ${v}g[${q * R + r}] = ${ld(buf, `${jOf(q)} + ${r * M}u`)};\n`;
    out += `    }\n`;
  }
  out += `    workgroupBarrier();\n`;
  for (let q = 0; q < Q; q++) {
    out += open(q) + `      let j: u32 = ${jOf(q)};\n      let k: u32 = ${pow2 ? `j & ${Ns - 1}u` : `j % ${Ns}u`};\n      let d: u32 = (j - k) * ${R}u + k;\n`;
    for (let r = 1; r < R; r++) out += `      let w${r}: vec2f = tw[k * ${r * step}u];\n`;
    for (const [v, buf] of lanes) out += oddButterfly(R, (r) => (r === 0 ? `${v}g[${q * R}]` : `cmul(${v}g[${q * R + r}], w${r})`), (r, e) => st(buf, `d + ${r * Ns}u`, e), "      ");
    out += `    }\n`;
  }
  return out + `    workgroupBarrier();\n  }\n`;
}

// ai: The last stage when it is odd and not radix 3 (radix3Stage's in-place form for any R): Ns = n / R, so a
// ai: butterfly reads and writes the same R points, buf[j + r n / R]; a thread takes j = tid, tid + T, ... below n / R,
// ai: its twiddles tw[r j] (step 1).
function oddLastStage(n, R, prec, lanes) {
  const { ld, st } = store(prec);
  const T = n / 8, Ns = n / R;
  let out = `  // stage: radix ${R}, Ns ${Ns}, in place\n  for (var j: u32 = tid; j < ${Ns}u; j += ${T}u) {\n`;
  for (let r = 1; r < R; r++) out += `    let w${r}: vec2f = tw[${r === 1 ? "" : `${r}u * `}j];\n`;
  for (const [, buf] of lanes) out += oddButterfly(R, (r) => (r === 0 ? ld(buf, "j") : `cmul(${ld(buf, `j + ${r * Ns}u`)}, w${r})`), (r, e) => st(buf, r === 0 ? "j" : `j + ${r * Ns}u`, e), "    ");
  return out + `  }\n  workgroupBarrier();\n`;
}

// The first stage: Ns = 1, no twiddles, the registers loaded by the caller as v[r] = x[tid + r n / 8].
export function firstStage(prec, lanes = [["v", "buf"]]) {
  const { st } = store(prec);
  let out = `  // stage 1: radix 8, Ns 1\n`;
  for (const [v, buf] of lanes) {
    out += `  dft8(&${v});\n`;
    for (let r = 0; r < 8; r++) out += `  ${st(buf, `tid * 8u + ${r}u`, `${v}[${r}]`)}\n`;
  }
  out += `  workgroupBarrier();\n`;
  return out;
}

export const V8 = "array<vec2f, 8>(vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0), vec2f(0.0))";

// ai: The per size uniform irows and ipic share: a = (n, V, yStride, sStride), b = (PIC a slot, size index, 0, 0),
// ai: c = (lim, 0.5 / lim, 0, 0).
const SIZE_PARAMS = /* wgsl */ `
struct SizeParams { a: vec4u, b: vec4u, c: vec4f }
`;
// ai: Every slot is a symbol of the frame.
const sendGate = /* wgsl */ `
  let r: u32 = wg.z;
`;

// ai: Bindings: S (ro), tw (ro, the size's twiddles), Y (rw), rows (uniform, the size's disc rows as pass 2 binds
// ai: them), P (uniform, SizeParams). Dispatch (V, 1, symbols).
export function irowsSource({ n }) {
  const { ty, ld, st } = store("f32");
  const T = n / 8;
  let load = "";
  for (let k = 0; k < 8; k++) {
    load += `  {\n    let x: u32 = tid + ${k * T}u;\n    var val: vec2f = vec2f(0.0);\n`;
    load += `    if (x < N / 2u) {\n      if (x >= ustart && x < ustart + row.z) { val = ${ld("sb", "sBase + x - ustart")}; }\n    } else {\n`;
    load += `      let un: u32 = N - x;\n      if (un <= row.w) { val = ${ld("sb", "sBase + row.z + row.w - un")}; }\n    }\n`;
    load += `    v[${k}] = vec2f(val.x, -val.y);\n  }\n`;
  }
  return SIZE_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> sb: array<${ty}>;
@group(0) @binding(1) var<storage, read> tw: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> yb: array<${ty}>;
struct Rows { r: array<vec4u, 512> }
@group(0) @binding(3) var<uniform> rows: Rows;
@group(0) @binding(4) var<uniform> P: SizeParams;
` + /* wgsl */ `
const N: u32 = ${n}u;
const T: u32 = ${T}u;
var<workgroup> buf: array<${ty}, N>;
` + BUTTERFLIES + /* wgsl */ `
@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
` + sendGate + /* wgsl */ `
  let vv: u32 = wg.x;
  let row: vec4u = rows.r[vv];
  let ustart: u32 = select(0u, 1u, vv == 0u);
  let sBase: u32 = r * P.a.w + row.x;
  var v: array<vec2f, 8> = ${V8};
  // ai: Register k holds x = tid + k T: the row's entry at u = x (x < n / 2) or u = x - n, conjugated; zero off the row.
` + load + firstStage("f32") + laterStages(n, "f32") + /* wgsl */ `
  let yBase: u32 = r * P.a.z + vv * N;
  for (var x: u32 = tid; x < N; x = x + T) {
    let z: vec2f = ${ld("buf", "x")};
    ${st("yb", "yBase + x", "vec2f(z.x, -z.y)")}
  }
}
`;
}

// ai: Bindings: Y (ro), tw (ro), PIC (rw, vec4f, the picture in f32, n^2 / 4 a slot), P (uniform, SizeParams); the clip
// ai: is P.c. Dispatch (n / 4, 1, symbols). The picture is quad-major: a workgroup's four columns, row by row, are n
// ai: consecutive vec4f (PIC[x / 4][y]), so its stores are whole lines (row-major they were 16 B a row, 2.4 of ipic's
// ai: 4.8 ms at 1536 on the iGPU); the sender turns it row-major after (wgsl/send.mjs TPOSE).
export function ipicSource({ n }) {
  const { ty, ld } = store("f32");
  const T = n / 8, lanes = [["v0", "buf0"], ["v1", "buf1"]];
  let load = "";
  for (let k = 0; k < 8; k++) {
    load += `  {\n    let m: u32 = tid + ${k * T}u;\n    let z0: vec2f = zAt(m, yBase, x0, x0 + 1u, V);\n    let z1: vec2f = zAt(m, yBase, x0 + 2u, x0 + 3u, V);\n`;
    load += `    v0[${k}] = vec2f(z0.x, -z0.y);\n    v1[${k}] = vec2f(z1.x, -z1.y);\n  }\n`;
  }
  return SIZE_PARAMS + /* wgsl */ `
@group(0) @binding(0) var<storage, read> yb: array<${ty}>;
@group(0) @binding(1) var<storage, read> tw: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> PIC: array<vec4f>;
@group(0) @binding(3) var<uniform> P: SizeParams;
` + /* wgsl */ `
const N: u32 = ${n}u;
const T: u32 = ${T}u;
var<workgroup> buf0: array<${ty}, N>;
var<workgroup> buf1: array<${ty}, N>;
` + BUTTERFLIES + /* wgsl */ `
// ai: Z(m) of the packed pair (xa, xb): turn_pack's rows from the inverse rows A = Y[v][xa], B = Y[v][xb].
fn zAt(m: u32, yBase: u32, xa: u32, xb: u32, V: u32) -> vec2f {
  if (m == 0u) {
    let a: vec2f = ${ld("yb", "yBase + xa")};
    let b: vec2f = ${ld("yb", "yBase + xb")};
    return vec2f(2.0 * a.x, 2.0 * b.x);
  }
  if (m < V) {
    let a: vec2f = ${ld("yb", "yBase + m * N + xa")};
    let b: vec2f = ${ld("yb", "yBase + m * N + xb")};
    return vec2f(a.x - b.y, a.y + b.x);
  }
  if (m > N - V) {
    let vq: u32 = N - m;
    let a: vec2f = ${ld("yb", "yBase + vq * N + xa")};
    let b: vec2f = ${ld("yb", "yBase + vq * N + xb")};
    return vec2f(a.x + b.y, b.x - a.y);
  }
  return vec2f(0.0);
}
@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
` + sendGate + /* wgsl */ `
  let x0: u32 = wg.x * 4u;
  let yBase: u32 = r * P.a.z;
  let V: u32 = P.a.y;
  var v0: array<vec2f, 8> = ${V8};
  var v1: array<vec2f, 8> = ${V8};
` + load + firstStage("f32", lanes) + laterStages(n, "f32", lanes) + /* wgsl */ `
  // ai: conj out: column a is the real part, column b the imaginary; clip_row's clamp and map to 0..1.
` + /* wgsl */ `  let lim: f32 = P.c.x;
  let sc: f32 = P.c.y;
  let picBase: u32 = r * P.b.x;
  for (var y: u32 = tid; y < N; y = y + T) {
    let o0: vec2f = ${ld("buf0", "y")};
    let o1: vec2f = ${ld("buf1", "y")};
    PIC[picBase + wg.x * N + y] = 0.5 + clamp(vec4f(o0.x, -o0.y, o1.x, -o1.y), vec4f(-lim), vec4f(lim)) * sc;
  }
}
`;
}
