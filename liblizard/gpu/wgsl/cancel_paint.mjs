// ai: The cancel stage's paint (gpu/back/DESIGN.md 13.1): a reference's known
// ai: picture painted on the device from pass one's verified blocks, the sender's encode with unknown blocks left
// ai: out (their coefficients zero, so 0.5 where nothing is painted). Three shaders, the encoder's arithmetic
// ai: (src/focus.c focus_encode, turn_pack, clip_row; src/ldpc.c ldpc_encode):
// ai: PAINT, a workgroup a (block, paint slot), 128 threads: the block's 473 payload bytes from REC through PLAN's
// ai:   record pointer, its CRC-32 (the LDPC's split CRC), the systematic codeword (data bits, then the 1272 parity
// ai:   bits as a prefix XOR over the checks in check order), the block's 5120 slots through PERMW (the bit map and
// ai:   the whitening: map_out) and the QPSK coefficients (+-a a bit, a = 0.70710678) into the slot's disc in S at
// ai:   the UV entries. A slot not used, a block the reference lacks, or one past the size's count: nothing written.
// ai: IROWS, a workgroup a (disc row, slot), n / 8 threads: the row's coefficients at (u + n) mod n, zeros elsewhere,
// ai:   the inverse along x (the forward kernels, conj in and conj out, unscaled), the row stored in Y's slot.
// ai: IPIC, a workgroup a (quad of columns, slot), n / 8 threads: turn_pack's packing takes two columns through one
// ai:   complex inverse along y (Z(0) = 2 Re A(0) + 2i Re B(0), Z(v) = A + iB, Z(n - v) = conj A + i conj B, zero
// ai:   between), the real part column a and the imaginary column b, clipped to 0.5 + clamp(v, -lim, lim) 0.5 / lim
// ai:   with lim = 2 clip sqrt(0.5 subch 320) at tilt 0, stored in PIC as f16, four adjacent x a vec2u.
// ai: The disc lives in S's frame slot r and the rows in Y's (yStride = V n): scratch pass one has finished with.
// ai: Every workgroup takes its slot's size from PLAN's KEYTAB entry (its tag's size bit): the paint serves every
// ai: size from one pipeline, irows and ipic are built a size and return on a slot of another.
// ai: irows and ipic take mode "send" (2026-09-29, the sender's GPU encoder, gpu/encoder.mjs): one size, a slot a
// ai: symbol, no PLAN; ipic clips at P.c and stores the picture as f32 (vec4f, four adjacent x). Mode "cancel" (the
// ai: default) is the cancel stage's, byte for byte as before. The sender's paint is its own since 2026-10-07
// ai: (wgsl/send.mjs paintSource: every code of the format's rate profile); this one paints the one-rate blocks the
// ai: cancel stage reads.
import { BUTTERFLIES, store, firstStage, laterStages, V8 } from "./back_transform.mjs";
import { CRC_SPLIT, PARAMS_STRUCT } from "./back_ldpc.mjs";
import { planLayout, NONE, KEYTAB_WORDS } from "./cancel_gate.mjs";

export const PAINT_THREADS = 128;   // ai: the split CRC wants 119 lanes; 128 lanes cover the 1272 checks at 10 each
export const A_QPSK = 0.70710678;   // ai: the C's 0.70710678f (focus.c focus_encode), amp 1 at tilt 0

const planConsts = ({ B, blocksMax, refSlots }) => {
  const L = planLayout({ blocksMax, B, refSlots });
  return /* wgsl */ `
const NONE: u32 = ${NONE}u;
const KEYTAB: u32 = ${L.keytab}u;
const PTR: u32 = ${L.ptr}u;
const SLOTS: u32 = ${L.slots}u;
const KVER: u32 = ${L.kver}u;
const KW: u32 = ${KEYTAB_WORDS}u;
const BM: u32 = ${blocksMax}u;
`;
};

// ai: Bindings: REC (ro), PLAN (ro), PERMW (ro, u16 pairs), UV (ro, the soft stage's), S (rw, the disc), P
// ai: (uniform, the LDPC's shape, wgsl/back_ldpc.mjs PARAMS_AT: sizes[s] = (blocks, UV entry offset, PERMW entry
// ai: offset, 0), dims = (blocksMax, sStride, R, 0), lay and pw as the LDPC's uniform holds them). Dispatch
// ai: (blocksMax, 1, R).
export function paintSource({ prec, B, blocksMax, refSlots }) {
  const { ty } = store(prec);
  const head = /* wgsl */ `
${PARAMS_STRUCT}
struct Rec { count: u32, words: array<u32> }
@group(0) @binding(0) var<storage, read> REC: Rec;
@group(0) @binding(1) var<storage, read> PLAN: array<u32>;
@group(0) @binding(2) var<storage, read> PERMW: array<u32>;
@group(0) @binding(3) var<storage, read> UV: array<u32>;
@group(0) @binding(4) var<storage, read_write> S: array<${ty}>;
@group(0) @binding(5) var<uniform> P: Params;
` + planConsts({ B, blocksMax, refSlots });
  return (prec === "f16" ? "enable f16;\n" : "") + head + /* wgsl */ `
const K: u32 = 3816u;
const M: u32 = 1272u;
const Z: u32 = 106u;
const MB: u32 = 12u;
const ND: u32 = 15u;
const CWW: u32 = 159u;
const RECW: u32 = 121u;
const TH: u32 = ${PAINT_THREADS}u;
const A: f32 = ${A_QPSK};
var<workgroup> packw: array<u32, 120>;
var<workgroup> crcw: atomic<u32>;
var<workgroup> cw: array<atomic<u32>, 159>;
var<workgroup> scan: array<u32, ${PAINT_THREADS}>;

// ai: Data bit j of the codeword: the payload bytes MSB first, then the CRC MSB first, as packw holds them (four
// ai: bytes a word in memory order).
fn dataBit(j: u32) -> u32 {
  let t = j & 31u;
  return (packw[j >> 5u] >> (8u * (t >> 3u) + 7u - (t & 7u))) & 1u;
}
// ai: The data bit check (r, i) reads through its slot e (src/ldpc.c ldpc_init: col z + (i - shift) mod z).
fn dataSlot(r: u32, i: u32, e: u32) -> u32 {
  let q = r * ND + e;
  let lc = P.lay[q >> 2u][q & 3u];
  var t: u32 = i + Z - (lc >> 16u);
  if (t >= Z) { t = t - Z; }
  return (lc & 0xffffu) * Z + t;
}
fn cwBit(j: u32) -> u32 { return (atomicLoad(&cw[j >> 5u]) >> (j & 31u)) & 1u; }
${CRC_SPLIT()}
@compute @workgroup_size(${PAINT_THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
` + /* wgsl */ `  let b: u32 = wg.x;
  let r: u32 = wg.z;
  let e: u32 = PLAN[SLOTS + r];
  if (e == NONE) { return; }
  let s: u32 = firstTrailingBit(PLAN[KEYTAB + KW * e + 7u] & 0xffu);
  if (b >= P.sizes[s].x) { return; }
  if (((PLAN[KEYTAB + KW * e + 1u + (b >> 5u)] >> (b & 31u)) & 1u) == 0u) { return; }
  let rec: u32 = PLAN[PTR + e * BM + b];
  if (rec == 0u) { return; }
  // ai: The payload as the LDPC packed it: words 2 to 119 of the record, byte 472 alone in word 120.
  let base: u32 = (rec - 1u) * RECW;
  if (i < 118u) { packw[i] = REC.words[base + 2u + i]; }
  if (i == 118u) { packw[118] = REC.words[base + 120u] & 0xffu; }
  if (i == 119u) { packw[119] = 0u; }
  workgroupBarrier();` + /* wgsl */ `
  let crc: u32 = crcOf(i);
  if (i == 0u) {
    packw[118] = packw[118] | (((crc >> 24u) & 0xffu) << 8u) | (((crc >> 16u) & 0xffu) << 16u) | (((crc >> 8u) & 0xffu) << 24u);
    packw[119] = crc & 0xffu;
  }
  workgroupBarrier();
  // ai: The systematic part of the codeword, a bit an index; then every lane's ten checks in check order, the
  // ai: check's parity as the XOR of its data bits, the prefix within the ten kept a bit each.
  for (var w: u32 = i; w < CWW; w = w + TH) {
    var word: u32 = 0u;
    for (var t: u32 = 0u; t < 32u; t = t + 1u) {
      let j = 32u * w + t;
      if (j < K) { word = word | (dataBit(j) << t); }
    }
    atomicStore(&cw[w], word);
  }
  var pl: u32 = 0u;
  var acc: u32 = 0u;
  for (var k: u32 = 0u; k < 10u; k = k + 1u) {
    let chk = 10u * i + k;
    if (chk < M) {
      let ii = chk / MB;
      let rr = chk % MB;
      var t: u32 = 0u;
      for (var q: u32 = 0u; q < ND; q = q + 1u) { t = t ^ dataBit(dataSlot(rr, ii, q)); }
      acc = acc ^ t;
    }
    pl = pl | (acc << k);
  }
  scan[i] = acc;
  workgroupBarrier();
  // ai: The prefix XOR across lanes (ldpc_encode's running acc), a read, a barrier and a write a step.
  for (var d: u32 = 1u; d < TH; d = d << 1u) {
    var v: u32 = scan[i];
    if (i >= d) { v = v ^ scan[i - d]; }
    workgroupBarrier();
    scan[i] = v;
    workgroupBarrier();
  }
  let ex: u32 = scan[i] ^ acc;
  for (var k: u32 = 0u; k < 10u; k = k + 1u) {
    let chk = 10u * i + k;
    if (chk < M && (((pl >> k) & 1u) ^ ex) != 0u) {
      let j = K + chk;
      atomicOr(&cw[j >> 5u], 1u << (j & 31u));
    }
  }
  workgroupBarrier();
  // ai: Slots 2c and 2c + 1 of coefficient c through PERMW (one u32 holds the pair), the whitening in the entry's
  // ai: top bit, a slot past the codeword (0x7fff) carrying the whitening alone.
  let uvBase: u32 = P.sizes[s].y + 2560u * b;
  let pwBase: u32 = (P.sizes[s].z + 5120u * b) >> 1u;
  let sBase: u32 = r * P.dims.y;
  for (var c: u32 = i; c < 2560u; c = c + TH) {
    let pe = PERMW[pwBase + c];
    let e0 = pe & 0xffffu;
    let e1 = pe >> 16u;
    var b0: u32 = e0 >> 15u;
    var b1: u32 = e1 >> 15u;
    if ((e0 & 0x7fffu) != 0x7fffu) { b0 = b0 ^ cwBit(e0 & 0x7fffu); }
    if ((e1 & 0x7fffu) != 0x7fffu) { b1 = b1 ^ cwBit(e1 & 0x7fffu); }
    let val = vec2f(select(A, -A, b0 != 0u), select(A, -A, b1 != 0u));
    S[sBase + UV[uvBase + c]] = ${ty}(val);
  }
}
`;
}

// ai: The per size uniform irows and ipic share: a = (n, V, yStride, sStride), b = (PIC vec2u a slot, size index,
// ai: 0, 0), c = (lim, 0.5 / lim, 0, 0).
const SIZE_PARAMS = /* wgsl */ `
struct SizeParams { a: vec4u, b: vec4u, c: vec4f }
`;
const slotGate = /* wgsl */ `
  let r: u32 = wg.z;
  let e: u32 = PLAN[SLOTS + r];
  if (e == NONE) { return; }
  if ((PLAN[KEYTAB + KW * e + 7u] & 0xffu) != (1u << P.b.y)) { return; }
`;
// ai: Mode "send": every slot is a symbol of the frame.
const sendGate = /* wgsl */ `
  let r: u32 = wg.z;
`;

// ai: Bindings: PLAN (ro), S (ro), tw (ro, the size's twiddles), Y (rw), rows (uniform, the size's disc rows as
// ai: pass 2 binds them), P (uniform, SizeParams). Dispatch (V, 1, R). Mode "send": no PLAN (S, tw, Y, rows, P at 0
// ai: to 4), every slot run; dispatch (V, 1, symbols).
export function irowsSource({ n, prec, B, blocksMax, refSlots, mode = "cancel" }) {
  const { ty, ld, st } = store(prec);
  const T = n / 8;
  let load = "";
  for (let k = 0; k < 8; k++) {
    load += `  {\n    let x: u32 = tid + ${k * T}u;\n    var val: vec2f = vec2f(0.0);\n`;
    load += `    if (x < N / 2u) {\n      if (x >= ustart && x < ustart + row.z) { val = ${ld("sb", "sBase + x - ustart")}; }\n    } else {\n`;
    load += `      let un: u32 = N - x;\n      if (un <= row.w) { val = ${ld("sb", "sBase + row.z + row.w - un")}; }\n    }\n`;
    load += `    v[${k}] = vec2f(val.x, -val.y);\n  }\n`;
  }
  const send = mode === "send";
  return (prec === "f16" ? "enable f16;\n" : "") + SIZE_PARAMS + (send ? /* wgsl */ `
@group(0) @binding(0) var<storage, read> sb: array<${ty}>;
@group(0) @binding(1) var<storage, read> tw: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> yb: array<${ty}>;
struct Rows { r: array<vec4u, 512> }
@group(0) @binding(3) var<uniform> rows: Rows;
@group(0) @binding(4) var<uniform> P: SizeParams;
` : /* wgsl */ `
@group(0) @binding(0) var<storage, read> PLAN: array<u32>;
@group(0) @binding(1) var<storage, read> sb: array<${ty}>;
@group(0) @binding(2) var<storage, read> tw: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> yb: array<${ty}>;
struct Rows { r: array<vec4u, 512> }
@group(0) @binding(4) var<uniform> rows: Rows;
@group(0) @binding(5) var<uniform> P: SizeParams;
` + planConsts({ B, blocksMax, refSlots })) + /* wgsl */ `
const N: u32 = ${n}u;
const T: u32 = ${T}u;
var<workgroup> buf: array<${ty}, N>;
` + BUTTERFLIES + /* wgsl */ `
@compute @workgroup_size(T)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) tid: u32) {
` + (send ? sendGate : slotGate) + /* wgsl */ `
  let vv: u32 = wg.x;
  let row: vec4u = rows.r[vv];
  let ustart: u32 = select(0u, 1u, vv == 0u);
  let sBase: u32 = r * P.a.w + row.x;
  var v: array<vec2f, 8> = ${V8};
  // ai: Register k holds x = tid + k T: the row's entry at u = x (x < n / 2) or u = x - n, conjugated; zero off the row.
` + load + firstStage(prec) + laterStages(n, prec, "chain") + /* wgsl */ `
  let yBase: u32 = r * P.a.z + vv * N;
  for (var x: u32 = tid; x < N; x = x + T) {
    let z: vec2f = ${ld("buf", "x")};
    ${st("yb", "yBase + x", "vec2f(z.x, -z.y)")}
  }
}
`;
}

// ai: Bindings: PLAN (ro), Y (ro), tw (ro), PIC (rw, vec2u), P (uniform, SizeParams), LIM (uniform, a version's
// ai: (lim, 0.5 / lim, 0, 0), versions 0 to 128: paint.mjs limTable). The clip is the slot's version's (PLAN's KVER
// ai: of its KEYTAB entry), the size's own (P.c) where it has none. Dispatch (n / 4, 1, R). Mode "send": Y, tw, PIC
// ai: (rw, vec4f, the picture in f32, n^2 / 4 a slot), P at 0 to 3; the clip is P.c; dispatch (n / 4, 1, symbols). The
// ai: send picture is quad-major: a workgroup's four columns, row by row, are n consecutive vec4f (PIC[x / 4][y]), so
// ai: its stores are whole lines (row-major they were 16 B a row, 2.4 of ipic's 4.8 ms at 1536 on the iGPU); the
// ai: sender turns it row-major after (wgsl/send.mjs TPOSE).
export function ipicSource({ n, prec, B, blocksMax, refSlots, mode = "cancel" }) {
  const { ty, ld } = store(prec);
  const T = n / 8, lanes = [["v0", "buf0"], ["v1", "buf1"]];
  let load = "";
  for (let k = 0; k < 8; k++) {
    load += `  {\n    let m: u32 = tid + ${k * T}u;\n    let z0: vec2f = zAt(m, yBase, x0, x0 + 1u, V);\n    let z1: vec2f = zAt(m, yBase, x0 + 2u, x0 + 3u, V);\n`;
    load += `    v0[${k}] = vec2f(z0.x, -z0.y);\n    v1[${k}] = vec2f(z1.x, -z1.y);\n  }\n`;
  }
  const send = mode === "send";
  return (prec === "f16" ? "enable f16;\n" : "") + SIZE_PARAMS + (send ? /* wgsl */ `
@group(0) @binding(0) var<storage, read> yb: array<${ty}>;
@group(0) @binding(1) var<storage, read> tw: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> PIC: array<vec4f>;
@group(0) @binding(3) var<uniform> P: SizeParams;
` : /* wgsl */ `
@group(0) @binding(0) var<storage, read> PLAN: array<u32>;
@group(0) @binding(1) var<storage, read> yb: array<${ty}>;
@group(0) @binding(2) var<storage, read> tw: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> PIC: array<vec2u>;
@group(0) @binding(4) var<uniform> P: SizeParams;
struct Lim { v: array<vec4f, 129> }
@group(0) @binding(5) var<uniform> LIM: Lim;
` + planConsts({ B, blocksMax, refSlots })) + /* wgsl */ `
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
` + (send ? sendGate : slotGate) + /* wgsl */ `
  let x0: u32 = wg.x * 4u;
  let yBase: u32 = r * P.a.z;
  let V: u32 = P.a.y;
  var v0: array<vec2f, 8> = ${V8};
  var v1: array<vec2f, 8> = ${V8};
` + load + firstStage(prec, lanes) + laterStages(n, prec, "chain", lanes) + /* wgsl */ `
  // ai: conj out: column a is the real part, column b the imaginary; clip_row's clamp and map to 0..1.
` + (send ? /* wgsl */ `  let lim: f32 = P.c.x;
  let sc: f32 = P.c.y;
  let picBase: u32 = r * P.b.x;
  for (var y: u32 = tid; y < N; y = y + T) {
    let o0: vec2f = ${ld("buf0", "y")};
    let o1: vec2f = ${ld("buf1", "y")};
    PIC[picBase + wg.x * N + y] = 0.5 + clamp(vec4f(o0.x, -o0.y, o1.x, -o1.y), vec4f(-lim), vec4f(lim)) * sc;
  }
}
` : /* wgsl */ `  let kv: u32 = PLAN[KVER + e];
  let lc: vec4f = select(P.c, LIM.v[min(kv, 128u)], kv >= 1u && kv <= 128u);
  let lim: f32 = lc.x;
  let sc: f32 = lc.y;
  let picBase: u32 = r * P.b.x;
  for (var y: u32 = tid; y < N; y = y + T) {
    let o0: vec2f = ${ld("buf0", "y")};
    let o1: vec2f = ${ld("buf1", "y")};
    let p: vec4f = 0.5 + clamp(vec4f(o0.x, -o0.y, o1.x, -o1.y), vec4f(-lim), vec4f(lim)) * sc;
    PIC[picBase + ((y * N + x0) >> 2u)] = vec2u(pack2x16float(p.xy), pack2x16float(p.zw));
  }
}
`);
}
