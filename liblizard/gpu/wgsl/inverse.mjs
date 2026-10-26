// ai: The sender's inverse transforms (gpu/encoder.mjs; the native sender through gen/sender.mjs): the decoder's own
// ai: butterflies (back_transform.mjs), conj in and conj out, unscaled, a symbol a slot, in f32.
// ai: IROWS, a workgroup a (disc row, symbol), n / 8 threads: the row's coefficients at (u + n) mod n, zeros
// ai:   elsewhere, the inverse along x, the row stored in Y's slot.
// ai: IPIC, a workgroup a (quad of columns, symbol), n / 8 threads: turn_pack's packing takes two columns through one
// ai:   complex inverse along y (Z(0) = 2 Re A(0) + 2i Re B(0), Z(v) = A + iB, Z(n - v) = conj A + i conj B, zero
// ai:   between), the real part column a and the imaginary column b, clipped to 0.5 + clamp(v, -lim, lim) 0.5 / lim
// ai:   (src/focus.c focus_encode, turn_pack, clip_row).
import { BUTTERFLIES, store, firstStage, laterStages, V8 } from "./back_transform.mjs";

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
