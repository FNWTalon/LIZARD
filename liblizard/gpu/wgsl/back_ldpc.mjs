// Part C of the GPU back half (DESIGN.md sections 6 and 7): one workgroup a codeword runs the layered normalised
// min-sum of src/ldpc.c ldpc_decode_stall in workgroup memory, the exact syndrome each iteration, the stall rule,
// then packs the systematic bits, checks the CRC-32 and claims a record slot for a verified block. Integer
// arithmetic throughout, the C's caps and norm, so from equal soft values the decisions and the iteration count
// are the C's bit for bit.
//
// Workgroup (x, y, s) is block x of frame LISTS[s][1 + y]; a workgroup past a size's block or frame count
// returns at once, as does a declined block. Lane i < 106 is check position i of every block row: within a row
// no two checks share a bit (a circulant is a permutation, the staircase bits belong to one check each), so a row
// is one barrier. R is kept compressed a check: min1, min2, arg and the 17 message signs in two words, and the
// message on slot e is rebuilt as (e == arg ? min2 : min1) with its sign. Check (0, 0)'s missing staircase slot
// is skipped: the C models it as a certain zero whose minimum can never survive the 127 cap, so the arithmetic
// is the same.
//
// The stop rule (ref_ldpc.mjs stopRule), which can only give up sooner, never admit a codeword; its give-up reports
// as a stall (verdict 3, ITS -3).
// ai: - net (the first read's, "learned", 2026-09-28): after iterations 1 to 29 lane 0 writes its state x = (bad, low,
// ai:   first, prev, t, est) to workgroup memory, lane j < H computes hidden unit j of the first layer, a barrier, of
// ai:   the second and its product with the output weight, a barrier, and lane 0 sums the products and gives up when
// ai:   the sum is under T. No stall test. The weights (stop/rule.mjs netWords) sit in MAP from word net.at, each
// ai:   layer's input-major so the lanes read consecutive words; the order of operations is rule.mjs scoreF32's.
// ai: - otherwise the C's stall rule always, and with a funnel (pass two's funnel2), give up after iteration j once
// ai:   the fewest violated checks seen so far is still at or above the checkpoint's count.
//
// The CRC-32 is split over lanes (ref_ldpc.mjs crcPowers): lane w runs the reflected bitwise CRC over its 4-byte
// chunk from a zero state, multiplies by x^(8 * bytes after it) mod P from a table, and XORs into one word; lane 0
// adds the initial state's term with the final inversion folded in. 119 lanes, 32 steps each, no table in
// workgroup memory.
import { SLOTS } from "./common.mjs";

export const THREADS = 128;
export const WG_BYTES = 4 * 5088 + 8 * 1272 + 4 * 120 + 16;   // what the workgroup declares

// variant: "base" recomputes a slot's index and old message in each of the two passes over a row (no private
// arrays) and tracks the two minima with branches; "hoist" keeps the row's 17 indices and q values in registers
// so pass 2 reads no workgroup memory; "reg" is hoist with the minima, the argument and the old message tracked
// by select (the C's SIMD kernel's form, the same values); "flags" is reg with the exact syndrome gated by the
// rows' own parities (DESIGN section 6): a row's update also returns the parity of its updated signs, and the
// exact check runs only on an iteration whose flags were all clear, or after iterations 1 and 9 for the stall
// counts. The flags are neither necessary nor sufficient, so this variant can exit an iteration later than the
// C and never earlier; its verdicts may then differ on a marginal block.
// "i16" is reg with the posteriors as two i16 a word (workgroup memory 20.8 KB, so six codewords fit a 128 KB
// WGP where four did): the two halves of a word belong to two lanes, so a write is one atomicXor of old ^ new
// on the lane's own half, which is right under any interleaving; "i16flags" adds the flag gate.
export const VARIANTS = ["base", "hoist", "reg", "flags", "i16", "i16flags"];
export const WG_BYTES_I16 = 2 * 5088 + 8 * 1272 + 4 * 120 + 16;

// ai: The uniform P in u32 words, the LDPC's and the cancel stage's paint's (wgsl/cancel_paint.mjs) alike: sizes, a
// ai: vec4u a picture slot; dims; lay, 45 vec4u (a data slot: col | shift << 16, 15 a block row); pw, 30 vec4u (the
// ai: CRC-32 powers, crcPowers(PAYLOAD)). PARAMS_STRUCT is its WGSL.
export const PARAMS_AT = { sizes: 0, dims: 4 * SLOTS, lay: 4 * SLOTS + 4, pw: 4 * SLOTS + 184, words: 4 * SLOTS + 304 };
export const PARAMS_STRUCT = `struct Params { sizes: array<vec4<u32>, ${SLOTS}>, dims: vec4<u32>, lay: array<vec4<u32>, 45>, pw: array<vec4<u32>, 30> }`;

// ai: The split CRC-32 as WGSL, shared with the cancel stage's paint (wgsl/cancel_paint.mjs): crcOf(i) is called by
// ai: every lane of the workgroup with its lane index (it holds two barriers) once the 473 payload bytes sit in the
// ai: workgroup array `packw` (120 u32, four bytes a word in memory order, byte 0 of word 118 the last) and returns
// ai: the CRC to every lane. `crcw` names the workgroup atomic it folds into, `crcStep` the name the step function
// ai: is declared under, `pw` the powers table expression (120 u32: crcPowers(PAYLOAD), as the LDPC's uniform holds
// ai: it). The names are parameters so a shader with its own arrays binds them without a copy of the arithmetic.
export function CRC_SPLIT({ packw = "packw", crcw = "crcw", crcStep = "crcStep", pw = "P.pw" } = {}) {
  return /* wgsl */ `
const CRC_CHUNKS: u32 = 119u;
const CRC_POLY: u32 = 0xedb88320u;
fn ${crcStep}(c: u32) -> u32 { return (c >> 1u) ^ (CRC_POLY & (0u - (c & 1u))); }
fn crcOf(i: u32) -> u32 {
  if (i == 0u) { atomicStore(&${crcw}, 0u); }
  workgroupBarrier();
  if (i < CRC_CHUNKS) {
    let w = ${packw}[i];
    var nb: u32 = 4u;
    if (i == CRC_CHUNKS - 1u) { nb = 1u; }
    var c: u32 = 0u;
    for (var t: u32 = 0u; t < nb; t = t + 1u) {
      c = c ^ ((w >> (8u * t)) & 0xffu);
      for (var q: u32 = 0u; q < 8u; q = q + 1u) { c = ${crcStep}(c); }
    }
    let p = ${pw}[i >> 2u][i & 3u];
    var acc: u32 = 0u;
    var tt = c;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      if (((p >> (31u - j)) & 1u) != 0u) { acc = acc ^ tt; }
      tt = ${crcStep}(tt);
    }
    if (i == 0u) { acc = acc ^ ${pw}[CRC_CHUNKS >> 2u][CRC_CHUNKS & 3u]; }
    atomicXor(&${crcw}, acc);
  }
  workgroupBarrier();
  return atomicLoad(&${crcw});
}
`;
}
// ai: Workgroup bytes the learned rule adds for a net of hidden units: its state x (6), hidden layer and products.
export const netWgBytes = (hidden) => 4 * (6 + 2 * hidden);

// funnel: [[iteration, count], ...] from ref_ldpc.mjs stopRule, empty for the C's rule alone.
// ai: net: { hidden, at } for the learned rule (no stall test, no funnel): its hidden units and its first word in MAP.
export function ldpcSource({ variant = "base", funnel = [], net = null } = {}) {
  if (!VARIANTS.includes(variant)) throw new Error(`ldpc variant ${variant}`);
  for (const [j, c] of funnel) if (!(Number.isInteger(j) && j >= 1 && j <= 30 && Number.isInteger(c) && c > 0)) throw new Error(`funnel checkpoint ${j}: ${c}`);
  if (net && (funnel.length || variant.endsWith("flags"))) throw new Error("the learned stop rule reads the exact count after every iteration: no funnel, no flags variant");
  if (net && !(Number.isInteger(net.hidden) && net.hidden >= 1 && net.hidden <= THREADS && Number.isInteger(net.at) && net.at >= 0)) throw new Error(`stop net ${JSON.stringify(net)}`);
  const give = funnel.map(([j, c]) => `(it == ${j - 1}u && low >= ${c}u)`).join(" || ");
  const counted = funnel.map(([j]) => ` || it == ${j - 1}u`).join("");
  const hoist = variant !== "base", branchless = !["base", "hoist"].includes(variant), flags = variant.endsWith("flags"), i16 = variant.startsWith("i16");
  const H = net?.hidden;
  return /* wgsl */ `
// ai: P (PARAMS_AT): sizes[s] = (blocks, MAP offset in u16 entries, 0, 0), blocks 0 where not served; dims = (B, L
// ai: words a frame, blocksMax, record cap); lay a data slot, col | shift << 16; pw chunk w's x^(8 * bytes after it),
// ai: entry 119 the init term.
${PARAMS_STRUCT}
struct Rec { count: atomic<u32>, words: array<u32> }
@group(0) @binding(0) var<storage, read> Lin: array<u32>;             // int8 soft values, slot order, 4 a word
@group(0) @binding(1) var<storage, read> BLK: array<vec2<u32>>;       // a block: (estimate bits, declined)
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;           // ai: [SLOTS][1 + B]: count, frame indices
@group(0) @binding(3) var<storage, read> MAP: array<u32>;             // u16 a codeword bit: slot | white << 15
@group(0) @binding(4) var<storage, read_write> V: array<u32>;         // verdict a block
@group(0) @binding(5) var<storage, read_write> ITS: array<i32>;       // iterations a block, the C's codes
@group(0) @binding(6) var<storage, read_write> REC: Rec;
@group(0) @binding(7) var<storage, read_write> BCOUNTS: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> P: Params;

const N: u32 = 5088u;
const K: u32 = 3816u;
const Z: u32 = 106u;
const MB: u32 = 12u;
const ND: u32 = 15u;
const NS: u32 = 17u;
const NORM: i32 = 13;
const MAXIT: u32 = 30u;
const STALL_IT: u32 = 9u;
const STALL_RATIO: f32 = 0.95;
const DATA_WORDS: u32 = 120u;
const REC_WORDS: u32 = 121u;

${i16 ? `var<workgroup> Lw: array<atomic<u32>, 2544>;  // posteriors, two i16 a word: data in code order, parity transposed` : `var<workgroup> Lw: array<i32, 5088>;          // posteriors: data in code order, parity transposed (r * 106 + i)`}
var<workgroup> Rw: array<vec2<u32>, 1272>;    // a check: min1 | min2 << 8 | arg << 16, and the slot signs
var<workgroup> badc: atomic<u32>;
var<workgroup> flag: u32;
var<workgroup> packw: array<u32, 120>;
var<workgroup> crcw: atomic<u32>;
${net ? `// ai: The learned stop rule's net (stop/rule.mjs netWords): from MAP word NET_AT, W1 input-major, b1, W2 input-major,
// ai: b2, w3, then T.
const NET_H: u32 = ${H}u;
const NET_AT: u32 = ${net.at}u;
var<workgroup> netX: array<f32, 6>;
var<workgroup> netH: array<f32, ${H}>;
var<workgroup> netP: array<f32, ${H}>;
fn netW(q: u32) -> f32 { return bitcast<f32>(MAP[NET_AT + q]); }
` : ``}
// The posterior at index idx, and its replacement given the value read (the xor touches this lane's half only).
fn ldL(idx: i32) -> i32 {
${i16 ? `  let w = atomicLoad(&Lw[u32(idx) >> 1u]);
  return bitcast<i32>(w << (16u * (1u - (u32(idx) & 1u)))) >> 16u;` : `  return Lw[idx];`}
}
fn stL(idx: i32, old: i32, nl: i32) {
${i16 ? `  atomicXor(&Lw[u32(idx) >> 1u], (u32(old ^ nl) & 0xffffu) << (16u * (u32(idx) & 1u)));` : `  Lw[idx] = nl;`}
}

// Slot e of check (r, i) as an index into Lw; -1 for the absent staircase bit of check (0, 0).
fn slotIdx(r: u32, i: u32, e: u32) -> i32 {
  if (e < ND) {
    let q = r * ND + e;
    let lc = P.lay[q >> 2u][q & 3u];
    var t = i + Z - (lc >> 16u);
    if (t >= Z) { t = t - Z; }
    return i32((lc & 0xffffu) * Z + t);
  }
  if (e == ND) {
    if (r > 0u) { return i32(K + (r - 1u) * Z + i); }
    if (i > 0u) { return i32(K + (MB - 1u) * Z + i - 1u); }
    return -1;
  }
  return i32(K + r * Z + i);
}

// The message check (r, i) last sent on slot e, from its compressed record.
fn oldMsg(rec: vec2<u32>, e: u32) -> i32 {
${branchless ? `  let m = select(i32(rec.x & 0xffu), i32((rec.x >> 8u) & 0xffu), e == ((rec.x >> 16u) & 0x1fu));
  return select(m, -m, ((rec.y >> e) & 1u) != 0u);` : `  var m = i32(rec.x & 0xffu);
  if (e == ((rec.x >> 16u) & 0x1fu)) { m = i32((rec.x >> 8u) & 0xffu); }
  if (((rec.y >> e) & 1u) != 0u) { m = -m; }
  return m;`}
}

// One check's update of block row r. Returns the parity of its updated signs (read by the flags variant only).
fn rowUpdate(r: u32, i: u32) -> u32 {
  let ri = r * Z + i;
  let rec = Rw[ri];
  var min1: i32 = 32767;
  var min2: i32 = 32767;
  var arg: u32 = 0u;
  var sgn: u32 = 0u;
${hoist ? `  var idxs = array<i32, 17>();
  var qs = array<i32, 17>();
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
    let idx = slotIdx(r, i, e);
    idxs[e] = idx;
    var q: i32 = 8191;
    if (idx >= 0) { q = ldL(idx) - oldMsg(rec, e); }
    qs[e] = q;
  }` : ``}
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
${hoist ? `    let idx = idxs[e];
    if (idx < 0) { continue; }
    let q = qs[e];` : `    let idx = slotIdx(r, i, e);
    if (idx < 0) { continue; }
    let q = ldL(idx) - oldMsg(rec, e);`}
    let a = abs(q);
    sgn = sgn ^ u32(q < 0);
${branchless ? `    let lt = a < min1;
    min2 = min(min2, max(min1, a));
    arg = select(arg, e, lt);
    min1 = min(min1, a);` : `    if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) { min2 = a; }`}
  }
  let m1 = min(127, (min1 * NORM) >> 4u);
  let m2 = min(127, (min2 * NORM) >> 4u);
  var signs: u32 = 0u;
  var par: u32 = 0u;
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
${hoist ? `    let idx = idxs[e];
    if (idx < 0) { continue; }
    let q = qs[e];` : `    let idx = slotIdx(r, i, e);
    if (idx < 0) { continue; }
    let q = ldL(idx) - oldMsg(rec, e);`}
${branchless ? `    let mag = select(m1, m2, e == arg);
    let s = sgn ^ u32(q < 0);
    let rn = select(mag, -mag, s != 0u);` : `    var mag = m1;
    if (e == arg) { mag = m2; }
    let s = sgn ^ u32(q < 0);
    var rn = mag;
    if (s != 0u) { rn = -mag; }`}
    signs = signs | (s << e);
    let nl = clamp(q + rn, -8191, 8191);
    stL(idx, q + oldMsg(rec, e), nl);
${flags ? `    par = par ^ u32(nl < 0);` : ``}
  }
  Rw[ri] = vec2<u32>(u32(m1) | (u32(m2) << 8u) | (arg << 16u), signs);
  return par;
}

// Violated checks among this lane's 12, from the signs alone.
fn violated(i: u32) -> u32 {
  var c: u32 = 0u;
  for (var r: u32 = 0u; r < MB; r = r + 1u) {
    var p: u32 = 0u;
    for (var e: u32 = 0u; e < NS; e = e + 1u) {
      let idx = slotIdx(r, i, e);
      if (idx < 0) { continue; }
      p = p ^ u32(ldL(idx) < 0);
    }
    c = c + p;
  }
  return c;
}

${CRC_SPLIT()}
@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let s = wg.z;
  let b = wg.x;
  let B = P.dims.x;
  let lStride = P.dims.y;
  let blocksMax = P.dims.z;
  if (b >= P.sizes[s].x || wg.y >= LISTS[s * (1u + B)]) { return; }
  let f = LISTS[s * (1u + B) + 1u + wg.y];
  // ai: A block past the frame's own count (back_transform.mjs listsBlocks) was not run by the soft stage either.
  if (b >= LISTS[${SLOTS}u * (1u + B) + 2u * B + f]) { return; }
  let slotV = f * blocksMax + b;
  // ai: Pass two (the cancel stage, over LISTS2) leaves a block pass one verified alone; V is zero in pass one.
  // ai: Read through the workgroup flag: to the uniformity analysis a read_write load is not uniform.
  if (i == 0u) { flag = V[slotV]; }
  if (workgroupUniformLoad(&flag) == 1u) { return; }
  if (BLK[slotV].y != 0u) {
    if (i == 0u) { V[slotV] = 2u; ITS[slotV] = -2; }
    return;
  }
  // The soft values through the bit map, whitening off, parity transposed: the C's map_in and its L load.
  let mapBase = P.sizes[s].y + b * N;
  let lBase = f * lStride + b * 1280u;
${i16 ? `  // A lane fills whole words: destination d holds codeword bit d below K, and bit K + 12 i + r at K + r * 106 + i.
  for (var wd: u32 = i; wd < N / 2u; wd = wd + ${THREADS}u) {
    var packed: u32 = 0u;
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let d = 2u * wd + h;
      var j = d;
      if (d >= K) { let t = d - K; j = K + MB * (t % Z) + t / Z; }
      let mi = mapBase + j;
      let me = (MAP[mi >> 1u] >> (16u * (mi & 1u))) & 0xffffu;
      let slot = me & 0x7fffu;
      let w = Lin[lBase + (slot >> 2u)];
      var v = bitcast<i32>(w << (8u * (3u - (slot & 3u)))) >> 24u;
      if ((me & 0x8000u) != 0u) { v = -v; }
      packed = packed | ((u32(v) & 0xffffu) << (16u * h));
    }
    atomicStore(&Lw[wd], packed);
  }` : `  for (var j: u32 = i; j < N; j = j + ${THREADS}u) {
    let mi = mapBase + j;
    let me = (MAP[mi >> 1u] >> (16u * (mi & 1u))) & 0xffffu;
    let slot = me & 0x7fffu;
    let w = Lin[lBase + (slot >> 2u)];
    var v = bitcast<i32>(w << (8u * (3u - (slot & 3u)))) >> 24u;
    if ((me & 0x8000u) != 0u) { v = -v; }
    var dst = j;
    if (j >= K) { let chk = j - K; dst = K + (chk % MB) * Z + chk / MB; }
    Lw[dst] = v;
  }`}
  for (var q: u32 = i; q < MB * Z; q = q + ${THREADS}u) { Rw[q] = vec2<u32>(0u, 0u); }
  if (i == 0u) { atomicStore(&badc, 0u); flag = 0u; }
  workgroupBarrier();

  var first: u32 = 0u;${give || net ? `
  var low: u32 = 0xffffffffu;` : ``}${net ? `
  var prev: u32 = 0u;
  let est = bitcast<f32>(BLK[slotV].x);` : ``}
  var it: u32 = 0u;
  var exitCode: u32 = 0u;
  for (it = 0u; it < MAXIT; it = it + 1u) {
    var rowsBad: u32 = 0u;
    for (var step: u32 = 0u; step < MB; step = step + 1u) {
      var r = step;
      if ((it & 1u) != 0u) { r = MB - 1u - step; }
      if (i < Z) { rowsBad = rowsBad | rowUpdate(r, i); }
      workgroupBarrier();
    }
${flags ? `    // The exact check only where the rows' own parities allow an exit, and where the stall rule needs its count${give ? ` and the
    // funnel its checkpoints (its minimum is then over the iterations counted)` : ``}.
    if (i < Z && rowsBad != 0u) { atomicAdd(&badc, 1u); }
    workgroupBarrier();
    if (i == 0u) {
      let fl = atomicLoad(&badc);
      atomicStore(&badc, 0u);
      flag = select(0u, 1u, fl == 0u || it == 0u || it == STALL_IT - 1u${counted});
    }
    let need = workgroupUniformLoad(&flag);
    if (need == 0u) { continue; }
` : ``}
    if (i < Z) {
      let c = violated(i);
      if (c != 0u) { atomicAdd(&badc, c); }
    }
    workgroupBarrier();
${net ? `    if (i == 0u) {
      let bad = atomicLoad(&badc);
      atomicStore(&badc, 0u);
      var d: u32 = 0u;
      if (bad == 0u) { d = 1u; }
      else {
        if (it == 0u) { first = bad; prev = bad; }
        low = min(low, bad);
        netX[0] = f32(bad); netX[1] = f32(low); netX[2] = f32(first); netX[3] = f32(prev); netX[4] = f32(it + 1u); netX[5] = est;
        prev = bad;
      }
      flag = d;
    }
    let d = workgroupUniformLoad(&flag);
    if (d != 0u) { exitCode = d; break; }
    // ai: the net after iterations 1 to 29 (the cap ends 30): a hidden unit a lane, a unit's sum in input order
    if (it + 1u < MAXIT) {
      if (i < NET_H) {
        var a: f32 = netW(6u * NET_H + i);
        for (var k: u32 = 0u; k < 6u; k = k + 1u) { a = a + netW(k * NET_H + i) * netX[k]; }
        netH[i] = max(a, 0.0);
      }
      workgroupBarrier();
      if (i < NET_H) {
        var a: f32 = netW((7u + NET_H) * NET_H + i);
        for (var k: u32 = 0u; k < NET_H; k = k + 1u) { a = a + netW((7u + k) * NET_H + i) * netH[k]; }
        netP[i] = netW((8u + NET_H) * NET_H + i) * max(a, 0.0);
      }
      workgroupBarrier();
      if (i == 0u) {
        var sum: f32 = netP[0];
        for (var k: u32 = 1u; k < NET_H; k = k + 1u) { sum = sum + netP[k]; }
        flag = select(0u, 2u, sum < netW((9u + NET_H) * NET_H));
      }
      let g = workgroupUniformLoad(&flag);
      if (g != 0u) { exitCode = g; break; }
    }
` : `    if (i == 0u) {
      let bad = atomicLoad(&badc);
      atomicStore(&badc, 0u);
      var d: u32 = 0u;
      if (bad == 0u) { d = 1u; }
      else if (it == 0u) { first = bad; }
      else if (it == STALL_IT - 1u && f32(bad) > STALL_RATIO * f32(first)) { d = 2u; }${give ? `
      if (bad != 0u) {
        low = min(low, bad);
        if (${give}) { d = 2u; }
      }` : ``}
      flag = d;
    }
    let d = workgroupUniformLoad(&flag);
    if (d != 0u) { exitCode = d; break; }
`}  }
  var its: i32 = -1;
  var ran: u32 = MAXIT;
  if (exitCode == 1u) { its = i32(it) + 1; ran = it + 1u; }
  if (exitCode == 2u) { its = -3; ran = it + 1u; }
  if (i == 0u) {
    ITS[slotV] = its;
    atomicAdd(&BCOUNTS[f * 8u + 3u], 1u);
    atomicAdd(&BCOUNTS[f * 8u + 7u], ran);
  }
  if (its < 0) {
    if (i == 0u) { V[slotV] = 3u; }
    return;
  }
  // Systematic bits to bytes, MSB first, four bytes a word in memory order.
  if (i < DATA_WORDS) {
    var w: u32 = 0u;
    for (var t: u32 = 0u; t < 32u; t = t + 1u) {
      let j = 32u * i + t;
      if (j < K && ldL(i32(j)) < 0) { w = w | (1u << (8u * (t >> 3u) + 7u - (t & 7u))); }
    }
    packw[i] = w;
  }
  let crc = crcOf(i);
  if (i == 0u) {
    let w118 = packw[118];
    let w119 = packw[119];
    let expect = (((w118 >> 8u) & 0xffu) << 24u) | (((w118 >> 16u) & 0xffu) << 16u) | (((w118 >> 24u) & 0xffu) << 8u) | (w119 & 0xffu);
    var sl: u32 = 0u;
    if (crc == expect) {
      V[slotV] = 1u;
      atomicAdd(&BCOUNTS[f * 8u + 4u], 1u);
      let n = atomicAdd(&REC.count, 1u);
      if (n < P.dims.w) { sl = n + 1u; atomicAdd(&BCOUNTS[f * 8u + 5u], 1u); }
    } else {
      V[slotV] = 4u;
    }
    flag = sl;
  }
  let sl = workgroupUniformLoad(&flag);
  if (sl == 0u || i >= REC_WORDS) { return; }
  let base = (sl - 1u) * REC_WORDS;
  var word: u32 = 0u;
  if (i == 0u) { word = f + 65536u * b; }
  else if (i == 1u) { word = bitcast<u32>(its); }
  else if (i < REC_WORDS - 1u) { word = packw[i - 2u]; }
  else { word = packw[118] & 0xffu; }
  REC.words[base + i] = word;
}
`;
}
