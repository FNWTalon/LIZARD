// ai: The back half's rate-profile stage: the format's coding since 2026-10-07 (src/focus.h focus_tiers_for). A
// ai: frame of v versions (8 v sub-channels) carries its blocks in up to three tiers from the lowest frequencies out:
// ai: 7/8 (7 sub-channels a block), 3/4 (8), 1/2 (12), every block 473 B; the profile is a function of v alone, so
// ai: these kernels read it from TAB by the frame's version (the gate's chunk count) and decode any version at any
// ai: picture slot. gpu/back/tiers.mjs builds TAB and the stage; it replaces the one-rate soft and LDPC dispatches.
// ai: TAB (u32 words, tabLayout): TT, a row of 16 words a version (blocks, 0, 0, 0, then (first block, first
// ai: sub-channel, blocks, 0) a code in CODES' order); INV, a code's inverse bit map (u16 two a word: the slot that
// ai: carries codeword bit j); WHITE, the whitening a bit a slot over 1024 sub-channels (bitmap.mjs whiten, packed
// ai: little-end first); NET, the learned stop's words (the 3/4 code's kernel reads them, stop/rule.mjs netWords).
// ai:   talign   the pilots' grid shift (back_soft.mjs alignSource's fit) over each block's own tail: 16 coefficients at
// ai:            3/4, 8 at 7/8, none at 1/2 (its codeword fills the block). A workgroup a frame, a dispatch a slot.
// ai:   tsoft    back_soft.mjs softSource's soft values and estimates a sub-channel, exactly; and each sub-channel's
// ai:            pilot reading where a block's tail lies in it. Both to EP (est, pilot) a sub-channel. Over the
// ai:            version's 8-sub-channel chunks, a dispatch a slot.
// ai:   tblocks  a block's estimate (the mean of its sub-channels'), its decline at its code's bar (0.2 k / n, as src/
// ai:            focus.c FOCUS_DECLINE), its pilot (its last sub-channel's). A workgroup a frame, a dispatch a slot.
// ai:   tldpc    a code's blocks: back_ldpc.mjs ldpcSource's i16 min-sum, the code's shape the module's constants, its
// ai:            block rows (unequal at 1/2) from the uniform; codeword bit j's slot and whitening from INV and WHITE
// ai:            (no stored rows); the known zeros past a block's 477 B at 127; the 3/4 code gives up by the learned
// ai:            stop (trained on it), the others by the C's stall rule. A dispatch a code over every slot.
import { SLOTS } from "./common.mjs";
import { CRC_SPLIT } from "./back_ldpc.mjs";
import { ALIGN_CHI } from "./back_soft.mjs";

export const TIER_BLOCKS = 128;   // ai: blocks a frame at most (talign's and tblocks' lanes; the rule's top is 121)
export const VERSIONS = 128;
export const KNOWN = 3816;        // ai: (473 + 4) x 8: the bits a block's payload and CRC fill; past them a code's data bits are zeros
export const WHITE_SLOTS = 1024 * 640;
// ai: the codes TT names, in order: src/focus.c focus_tiers_for's three rates (ldpc.h index, sub-channels a block)
export const CODES = [{ rate: 6, subs: 7, name: "7/8" }, { rate: 4, subs: 8, name: "3/4" }, { rate: 2, subs: 12, name: "1/2" }];
// ai: a NaN made at run time (a constant NaN is a WGSL error, which Tint reports and naga let pass): P is a uniform
const NAN = "bitcast<f32>(0x7fc00000u | (P.n & 0u))";
const TWO_PI = (2 * Math.PI).toFixed(9);
const PARAMS = "struct Params { s: u32, subch: u32, blocks: u32, uvOff: u32, sStride: u32, lStride: u32, subchMax: u32, blocksMax: u32, B: u32, bar: f32, ucOff: u32, n: u32 }";

// ai: TAB's sections' word offsets, from the codes' n and the net's words.
export function tabLayout(codes, netWords) {
  let o = 16 * (VERSIONS + 1);
  const inv = codes.map((c) => { const at = o; o += Math.ceil(c.n / 2); return at; });
  const WHITE = o;
  o += WHITE_SLOTS / 32 + 1;   // ai: one word past the last slot: whiteWord reads the word after a slot's
  const NET = o;
  o += netWords;
  return { TT: 0, inv, WHITE, NET, words: o };
}

// ai: The WGSL every kernel shares: TAB's constants and its readers. codes: [{ n, k, subs, tail, bar }] in CODES' order.
function tables(L, codes) {
  const pick = (f) => `select(select(${f(codes[2])}, ${f(codes[1])}, c == 1u), ${f(codes[0])}, c == 0u)`;
  const barBits = (x) => { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; };
  return /* wgsl */ `
const TT_AT: u32 = ${L.TT}u;
const WHITE_AT: u32 = ${L.WHITE}u;
fn nOf(c: u32) -> u32 { return ${pick((x) => `${x.n}u`)}; }
fn subsOf(c: u32) -> u32 { return ${pick((x) => `${x.subs}u`)}; }
fn tailOf(c: u32) -> u32 { return ${pick((x) => `${x.tail}u`)}; }
fn barOf(c: u32) -> f32 { return bitcast<f32>(${pick((x) => `${barBits(x.bar)}u`)}); }
fn blocksOf(v: u32) -> u32 { return TAB[TT_AT + 16u * v]; }
// ai: (first block, first sub-channel, blocks, 0) of code c in version v
fn ttOf(v: u32, c: u32) -> vec4u { let o = TT_AT + 16u * v + 4u + 4u * c; return vec4u(TAB[o], TAB[o + 1u], TAB[o + 2u], TAB[o + 3u]); }
fn whiteWord(slot: u32) -> u32 {
  let w = WHITE_AT + (slot >> 5u);
  let s = slot & 31u;
  if (s == 0u) { return TAB[w]; }
  return (TAB[w] >> s) | (TAB[w + 1u] << (32u - s));
}
// ai: block b of version v: (its code, its index in its tier); code 3 past the version's blocks
fn blockAt(v: u32, b: u32) -> vec2u {
  for (var c: u32 = 0u; c < 3u; c++) { let t = ttOf(v, c); if (b >= t.x && b < t.x + t.z) { return vec2u(c, b - t.x); } }
  return vec2u(3u, 0u);
}
// ai: sub-channel j of version v: (its code, its block's index in its tier, its place in the block); code 3 past them
fn subAt(v: u32, j: u32) -> vec3u {
  for (var c: u32 = 0u; c < 3u; c++) {
    let t = ttOf(v, c);
    let su = subsOf(c);
    if (j >= t.y && j < t.y + t.z * su) { let d = j - t.y; return vec3u(c, d / su, d % su); }
  }
  return vec3u(3u, 0u, 0u);
}
`;
}
// ai: The frame's version: the gate's chunk count for it (back_transform.mjs listsBlocks).
const versionOf = (lists, B, f) => `${lists}[${SLOTS}u * (${B} + 1u) + 2u * ${B} + ${f}]`;

export const talignSource = ({ f16 = false, layout, codes }) => (f16 ? "enable f16;\n" : "") + /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<storage, read> S: array<${f16 ? "vec2<f16>" : "vec2<f32>"}>;
@group(0) @binding(1) var<storage, read> UV: array<u32>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(3) var<storage, read_write> PILOT: array<f32>;
@group(0) @binding(4) var<storage, read> TAB: array<u32>;
@group(0) @binding(5) var<uniform> P: Params;
${tables(layout, codes)}
var<workgroup> acc: array<f32, ${7 * TIER_BLOCKS}>;
var<workgroup> cur: vec2<f32>;
var<workgroup> last: array<f32, 7>;

@compute @workgroup_size(${TIER_BLOCKS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let count: u32 = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  let f: u32 = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  let v: u32 = ${versionOf("LISTS", "P.B", "f")};
  let blocks: u32 = min(blocksOf(v), ${TIER_BLOCKS}u);
  let kk: f32 = ${TWO_PI} / f32(P.n);
  if (t == 0u) { cur = vec2<f32>(0.0); }
  workgroupBarrier();
  for (var it: u32 = 0u; it < 2u; it++) {
    var a: f32 = 0.0; var b: f32 = 0.0; var c: f32 = 0.0; var p: f32 = 0.0; var q: f32 = 0.0; var yy: f32 = 0.0; var used: f32 = 0.0;
    var cnt: u32 = 0u;
    var at: u32 = 0u;
    var tw: u32 = 0u;
    if (t < blocks) {
      let bc = blockAt(v, t);
      if (bc.x < 3u) {
        cnt = tailOf(bc.x);
        let first = ttOf(v, bc.x).y + bc.y * subsOf(bc.x);
        at = 320u * first + nOf(bc.x) / 2u;
        tw = whiteWord(640u * first + nOf(bc.x));
      }
    }
    if (cnt > 0u) {
      let d: vec2<f32> = cur;
      var z: array<vec2<f32>, 16>;
      var uv: array<vec2<f32>, 16>;
      var sum: f32 = 0.0;
      for (var i: u32 = 0u; i < 16u; i++) {
        if (i < cnt) {
          let y0: vec2<f32> = vec2<f32>(S[f * P.sStride + UV[P.uvOff + at + i]]);
          let w: u32 = UV[P.ucOff + at + i];
          let u: f32 = f32(bitcast<i32>(w) >> 16u);
          let vv: f32 = f32(w & 0xffffu);
          let th: f32 = kk * (u * d.x + vv * d.y);
          let cs: f32 = cos(th);
          let sn: f32 = sin(th);
          let y: vec2<f32> = vec2<f32>(y0.x * cs + y0.y * sn, y0.y * cs - y0.x * sn);
          let xr: f32 = select(1.0, -1.0, ((tw >> (2u * i)) & 1u) != 0u);
          let xi: f32 = select(1.0, -1.0, ((tw >> (2u * i + 1u)) & 1u) != 0u);
          z[i] = vec2<f32>(y.x * xr + y.y * xi, y.y * xr - y.x * xi);
          uv[i] = vec2<f32>(u, vv);
          sum += z[i].x;
        }
      }
      // the block's sign: its pilots' own sum
      let sg: f32 = select(1.0, -1.0, sum < 0.0);
      for (var i: u32 = 0u; i < 16u; i++) {
        if (i < cnt) {
          let zr: f32 = sg * z[i].x;
          let zi: f32 = sg * z[i].y;
          let cu: f32 = kk * zr * uv[i].x;
          let cv: f32 = kk * zr * uv[i].y;
          a += cu * cu; b += cu * cv; c += cv * cv; p += cu * zi; q += cv * zi; yy += zi * zi;
        }
      }
      used = f32(cnt);
    }
    acc[7u * t] = a; acc[7u * t + 1u] = b; acc[7u * t + 2u] = c; acc[7u * t + 3u] = p; acc[7u * t + 4u] = q; acc[7u * t + 5u] = yy; acc[7u * t + 6u] = used;
    workgroupBarrier();
    if (t == 0u) {
      var s: array<f32, 7>;
      for (var k: u32 = 0u; k < 7u; k++) { s[k] = 0.0; }
      for (var i: u32 = 0u; i < ${TIER_BLOCKS}u; i++) { for (var k: u32 = 0u; k < 7u; k++) { s[k] += acc[7u * i + k]; } }
      let det: f32 = s[0] * s[2] - s[1] * s[1];
      if (s[6] >= 8.0 && det > 0.0) { cur = cur + vec2<f32>((s[3] * s[2] - s[4] * s[1]) / det, (s[0] * s[4] - s[1] * s[3]) / det); } else { s[6] = 0.0; }
      for (var k: u32 = 0u; k < 7u; k++) { last[k] = s[k]; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    var out: vec2<f32> = vec2<f32>(0.0);
    let det: f32 = last[0] * last[2] - last[1] * last[1];
    if (last[6] >= 8.0 && det > 0.0) {
      let e: vec2<f32> = vec2<f32>((last[3] * last[2] - last[4] * last[1]) / det, (last[0] * last[4] - last[1] * last[3]) / det);
      let s2: f32 = (last[5] - e.x * last[3] - e.y * last[4]) / (last[6] - 2.0);
      let d: vec2<f32> = cur;
      if (s2 > 0.0 && (d.x * d.x * last[0] + 2.0 * d.x * d.y * last[1] + d.y * d.y * last[2]) > ${ALIGN_CHI} * s2) { out = d; }
    }
    PILOT[P.B * P.blocksMax + 2u * f] = out.x;
    PILOT[P.B * P.blocksMax + 2u * f + 1u] = out.y;
  }
}
`;

// ai: back_soft.mjs softSource's sub-channel arithmetic exactly (32 lanes of 10 coefficients a sub-channel, eight a
// ai: workgroup); a sub-channel's pilot reading where a block's tail lies in it, to EP with its estimate; no BLK.
export const tsoftSource = ({ f16 = false, layout, codes }) => (f16 ? "enable f16;\n" : "") + /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<storage, read> S: array<${f16 ? "vec2<f16>" : "vec2<f32>"}>;
@group(0) @binding(1) var<storage, read> UV: array<u32>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(3) var<storage, read_write> L: array<u32>;
@group(0) @binding(4) var<storage, read_write> EP: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> COUNTS: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> PILOT: array<f32>;
@group(0) @binding(7) var<storage, read> TAB: array<u32>;
@group(0) @binding(8) var<uniform> P: Params;
${tables(layout, codes)}
var<workgroup> red2: array<f32, 256>;
var<workgroup> m2Of: array<f32, 8>;
var<workgroup> red4: array<f32, 256>;
var<workgroup> kOf: array<f32, 8>;
var<workgroup> estOf: array<f32, 8>;

fn jinfo(g: f32) -> f32 {
  if (g <= 0.0) { return 0.0; }
  let base = 1.0 - exp2(-0.3073 * pow(2.0 * sqrt(g), 1.7870));
  if (base <= 0.0) { return 0.0; }
  return pow(base, 1.1064);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let b: u32 = wg.x;
  let count: u32 = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  let f: u32 = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  let v: u32 = ${versionOf("LISTS", "P.B", "f")};
  if (b >= v) { return; }
  let g: u32 = t >> 5u;
  let l: u32 = t & 31u;
  let j: u32 = 8u * b + g;
  let uvBase: u32 = P.uvOff + 320u * j + 10u * l;
  let sBase: u32 = f * P.sStride;
  var y: array<vec2<f32>, 10> = array<vec2<f32>, 10>();
  var m2: f32 = 0.0;
  var m4: f32 = 0.0;
  let sh: vec2<f32> = vec2<f32>(PILOT[P.B * P.blocksMax + 2u * f], PILOT[P.B * P.blocksMax + 2u * f + 1u]);
  let turn: bool = sh.x != 0.0 || sh.y != 0.0;
  let ucBase: u32 = P.ucOff + 320u * j + 10u * l;
  for (var c: u32 = 0u; c < 10u; c++) {
    var vv: vec2<f32> = vec2<f32>(S[sBase + UV[uvBase + c]]);
    if (turn) {
      let w: u32 = UV[ucBase + c];
      let th: f32 = ${TWO_PI} / f32(P.n) * (f32(bitcast<i32>(w) >> 16u) * sh.x + f32(w & 0xffffu) * sh.y);
      let cs: f32 = cos(th);
      let sn: f32 = sin(th);
      vv = vec2<f32>(vv.x * cs + vv.y * sn, vv.y * cs - vv.x * sn);
    }
    y[c] = vv;
    let e: f32 = dot(vv, vv);
    m2 += e;
    m4 += e * e;
  }
  red2[t] = m2;
  red4[t] = m4;
  workgroupBarrier();
  for (var h: u32 = 16u; h > 0u; h = h >> 1u) {
    if (l < h) { red2[t] = red2[t] + red2[t + h]; red4[t] = red4[t] + red4[t + h]; }
    workgroupBarrier();
  }
  if (l == 0u) {
    let mm2: f32 = red2[t] / 320.0;
    let mm4: f32 = red4[t] / 320.0;
    let a2: f32 = sqrt(max(2.0 * mm2 * mm2 - mm4, 0.0));
    let nv: f32 = max(mm2 - a2, 0.02 * mm2);
    var k: f32 = 0.0;
    var est: f32 = 0.0;
    if (nv > 0.0) { k = 0.7 * 2.0 * sqrt(2.0 * a2) / nv; est = jinfo(a2 / nv); }
    kOf[g] = k;
    m2Of[g] = mm2;
    estOf[g] = est;
  }
  workgroupBarrier();
  let k: f32 = kOf[g];
  let lBase: u32 = f * P.lStride + 160u * j + 5u * l;
  for (var w: u32 = 0u; w < 5u; w++) {
    var word: u32 = 0u;
    for (var q: u32 = 0u; q < 2u; q++) {
      let r: vec2<i32> = vec2<i32>(round(clamp(k * y[2u * w + q], vec2<f32>(-10.0), vec2<f32>(10.0)) * 8.0));
      word = word | ((u32(r.x) & 0xffu) << (16u * q)) | ((u32(r.y) & 0xffu) << (16u * q + 8u));
    }
    L[lBase + w] = word;
  }
  // ai: a block's tail in this sub-channel: its last sub-channel's coefficients from tfrom to 320, against the
  // ai: whitening's signs
  let sa = subAt(v, j);
  var tfrom: u32 = 320u;
  var tw: u32 = 0u;
  if (sa.x < 3u && sa.z == subsOf(sa.x) - 1u && tailOf(sa.x) > 0u) {
    let first = ttOf(v, sa.x).y + sa.y * subsOf(sa.x);
    tfrom = nOf(sa.x) / 2u - 320u * (subsOf(sa.x) - 1u);
    tw = whiteWord(640u * first + nOf(sa.x));
  }
  var cs: f32 = 0.0;
  if (tfrom < 320u) {
    for (var c: u32 = 0u; c < 10u; c++) {
      let ci: u32 = 10u * l + c;
      if (ci >= tfrom) {
        let q: u32 = ci - tfrom;
        cs += select(y[c].x, -y[c].x, ((tw >> (2u * q)) & 1u) != 0u) + select(y[c].y, -y[c].y, ((tw >> (2u * q + 1u)) & 1u) != 0u);
      }
    }
  }
  red2[t] = cs;
  workgroupBarrier();
  if (l == 0u) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 32u; i++) { sum += red2[t + i]; }
    let m: f32 = m2Of[g];
    var r: f32 = ${NAN};
    if (tfrom < 320u) { r = select(0.0, sum / f32(2u * (320u - tfrom)) / sqrt(m / 2.0), m > 0.0); }
    EP[f * P.subchMax + j] = vec2<f32>(estOf[g], r);
  }
  if (t == 0u) { atomicAdd(&COUNTS[f * 8u + 6u], 8u); }
}
`;

export const tblocksSource = ({ layout, codes }) => /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<storage, read> LISTS: array<u32>;
@group(0) @binding(1) var<storage, read> EP: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> BLK: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> PILOT: array<f32>;
@group(0) @binding(4) var<storage, read> TAB: array<u32>;
@group(0) @binding(5) var<uniform> P: Params;
${tables(layout, codes)}
@compute @workgroup_size(${TIER_BLOCKS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let count: u32 = LISTS[P.s * (P.B + 1u)];
  if (wg.z >= count) { return; }
  let f: u32 = LISTS[P.s * (P.B + 1u) + 1u + wg.z];
  let v: u32 = ${versionOf("LISTS", "P.B", "f")};
  if (t >= min(blocksOf(v), ${TIER_BLOCKS}u)) { return; }
  let bc = blockAt(v, t);
  if (bc.x >= 3u) { return; }
  let su = subsOf(bc.x);
  let first = ttOf(v, bc.x).y + bc.y * su;
  var sum: f32 = 0.0;
  for (var g: u32 = 0u; g < su; g++) { sum += EP[f * P.subchMax + first + g].x; }
  let be: f32 = sum / f32(su);
  BLK[f * P.blocksMax + t] = vec2<u32>(bitcast<u32>(be), select(0u, 1u, be < barOf(bc.x)));
  var r: f32 = ${NAN};
  if (tailOf(bc.x) > 0u) { r = EP[f * P.subchMax + first + su - 1u].y; }
  PILOT[f * P.blocksMax + t] = r;
}
`;

// ai: A code's module constants: its shape (n, k, z, block rows mb, the most slots a row, staircase included), the
// ai: workgroup's lanes (z rounded up to 32, at least 128) and whether a check's record fits one word.
export function tierShape(code) {
  const { n, k, z, mb, lay } = code;
  let nd = 0;
  for (let r = 0; r < mb; r++) nd = Math.max(nd, lay[r + 1] - lay[r]);
  const ns = nd + 2, slots = lay[mb];
  return { n, k, z, mb, ns, slots, threads: Math.max(128, 32 * Math.ceil(z / 32)), one: ns <= 12, rowv: Math.ceil((mb + 1) / 4), pairv: Math.ceil(slots / 4) };
}
// ai: Workgroup bytes a code's kernel declares (32 KB is the floor every phone we know offers).
export const tierWgBytes = (sh, net) => 2 * sh.n + (sh.one ? 4 : 8) * sh.mb * sh.z + 4 * 120 + 12 + (net ? 4 * (6 + 2 * net.hidden) : 0);
// ai: A code's uniform: dims (B, L words a frame, blocksMax, record cap), the row pointers, the data slots (col |
// ai: shift << 16), the CRC powers.
export const tierUniformStruct = (sh) => `struct TierCode { dims: vec4u, rows: array<vec4u, ${sh.rowv}>, pairs: array<vec4u, ${sh.pairv}>, pw: array<vec4u, 30> }`;
export const tierUniformWords = (sh) => 4 + 4 * sh.rowv + 4 * sh.pairv + 120;

// ai: c: the code's index in CODES; inv: its INV section's offset in TAB; net: { hidden, at } for the learned stop.
export function tldpcSource(sh, { c, inv, layout, codes, net = null }) {
  const { n: N, k: K, z: Z, mb: MB, ns: NS, threads: TH, one } = sh;
  if (N % 2 || K < KNOWN || NS > 32 || (one && NS > 12)) throw new Error(`tier code ${JSON.stringify(sh)}`);
  const R = one ? "u32" : "vec2<u32>", H = net?.hidden;
  return /* wgsl */ `
${tierUniformStruct(sh)}
struct Rec { count: atomic<u32>, words: array<u32> }
@group(0) @binding(0) var<storage, read> Lin: array<u32>;
@group(0) @binding(1) var<storage, read> BLK: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(3) var<storage, read> TAB: array<u32>;
@group(0) @binding(4) var<storage, read_write> V: array<u32>;
@group(0) @binding(5) var<storage, read_write> ITS: array<i32>;
@group(0) @binding(6) var<storage, read_write> REC: Rec;
@group(0) @binding(7) var<storage, read_write> BCOUNTS: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> T: TierCode;
${tables(layout, codes)}
const CODE: u32 = ${c}u;
const INV_AT: u32 = ${inv}u;
const N: u32 = ${N}u;
const K: u32 = ${K}u;
const Z: u32 = ${Z}u;
const MB: u32 = ${MB}u;
const NS: u32 = ${NS}u;
const KNOWN: u32 = ${KNOWN}u;
const NORM: i32 = 13;
const MAXIT: u32 = 30u;
const STALL_IT: u32 = 9u;
const STALL_RATIO: f32 = 0.95;
const DATA_WORDS: u32 = 120u;
const REC_WORDS: u32 = 121u;

var<workgroup> Lw: array<atomic<u32>, ${N / 2}>;
var<workgroup> Rw: array<${R}, ${MB * Z}>;
var<workgroup> badc: atomic<u32>;
var<workgroup> flag: u32;
var<workgroup> packw: array<u32, 120>;
var<workgroup> crcw: atomic<u32>;
${net ? `const NET_H: u32 = ${H}u;
const NET_AT: u32 = ${net.at}u;
var<workgroup> netX: array<f32, 6>;
var<workgroup> netH: array<f32, ${H}>;
var<workgroup> netP: array<f32, ${H}>;
fn netW(q: u32) -> f32 { return bitcast<f32>(TAB[NET_AT + q]); }
` : ``}
fn ldL(idx: i32) -> i32 {
  let w = atomicLoad(&Lw[u32(idx) >> 1u]);
  return bitcast<i32>(w << (16u * (1u - (u32(idx) & 1u)))) >> 16u;
}
fn stL(idx: i32, old: i32, nl: i32) {
  atomicXor(&Lw[u32(idx) >> 1u], (u32(old ^ nl) & 0xffffu) << (16u * (u32(idx) & 1u)));
}
fn rowAt(r: u32) -> u32 { return T.rows[r >> 2u][r & 3u]; }

// Slot e of check (r, i) as an index into Lw: the row's data slots, then the previous check's staircase bit (absent at
// check (0, 0)), then its own; -1 past the row's slots.
fn slotIdx(r: u32, i: u32, e: u32) -> i32 {
  let p0 = rowAt(r);
  let nd = rowAt(r + 1u) - p0;
  if (e < nd) {
    let q = p0 + e;
    let lc = T.pairs[q >> 2u][q & 3u];
    var t = i + Z - (lc >> 16u);
    if (t >= Z) { t = t - Z; }
    return i32((lc & 0xffffu) * Z + t);
  }
  if (e == nd) {
    if (r > 0u) { return i32(K + (r - 1u) * Z + i); }
    if (i > 0u) { return i32(K + (MB - 1u) * Z + i - 1u); }
    return -1;
  }
  if (e == nd + 1u) { return i32(K + r * Z + i); }
  return -1;
}

fn oldMsg(rec: ${R}, e: u32) -> i32 {
${one ? `  let m = select(i32(rec & 0xffu), i32((rec >> 8u) & 0xffu), e == ((rec >> 16u) & 0xfu));
  return select(m, -m, ((rec >> (20u + e)) & 1u) != 0u);` : `  let m = select(i32(rec.x & 0xffu), i32((rec.x >> 8u) & 0xffu), e == ((rec.x >> 16u) & 0x1fu));
  return select(m, -m, ((rec.y >> e) & 1u) != 0u);`}
}

fn rowUpdate(r: u32, i: u32) {
  let ri = r * Z + i;
  let rec = Rw[ri];
  var min1: i32 = 32767;
  var min2: i32 = 32767;
  var arg: u32 = 0u;
  var sgn: u32 = 0u;
  var idxs = array<i32, ${NS}>();
  var qs = array<i32, ${NS}>();
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
    let idx = slotIdx(r, i, e);
    idxs[e] = idx;
    var q: i32 = 8191;
    if (idx >= 0) { q = ldL(idx) - oldMsg(rec, e); }
    qs[e] = q;
  }
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
    let idx = idxs[e];
    if (idx < 0) { continue; }
    let q = qs[e];
    let a = abs(q);
    sgn = sgn ^ u32(q < 0);
    let lt = a < min1;
    min2 = min(min2, max(min1, a));
    arg = select(arg, e, lt);
    min1 = min(min1, a);
  }
  let m1 = min(127, (min1 * NORM) >> 4u);
  let m2 = min(127, (min2 * NORM) >> 4u);
  var signs: u32 = 0u;
  for (var e: u32 = 0u; e < NS; e = e + 1u) {
    let idx = idxs[e];
    if (idx < 0) { continue; }
    let q = qs[e];
    let mag = select(m1, m2, e == arg);
    let s = sgn ^ u32(q < 0);
    let rn = select(mag, -mag, s != 0u);
    signs = signs | (s << e);
    let nl = clamp(q + rn, -8191, 8191);
    stL(idx, q + oldMsg(rec, e), nl);
  }
${one ? `  Rw[ri] = u32(m1) | (u32(m2) << 8u) | (arg << 16u) | (signs << 20u);` : `  Rw[ri] = vec2<u32>(u32(m1) | (u32(m2) << 8u) | (arg << 16u), signs);`}
}

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

${CRC_SPLIT({ pw: "T.pw" })}
@compute @workgroup_size(${TH})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) i: u32) {
  let B = T.dims.x;
  let lStride = T.dims.y;
  let blocksMax = T.dims.z;
  let s = wg.z;
  if (wg.y >= LISTS[s * (1u + B)]) { return; }
  let f = LISTS[s * (1u + B) + 1u + wg.y];
  let v = ${versionOf("LISTS", "B", "f")};
  let tt = ttOf(v, CODE);
  let x = wg.x;
  if (x >= tt.z) { return; }
  let b = tt.x + x;
  let slotV = f * blocksMax + b;
  if (BLK[slotV].y != 0u) {
    if (i == 0u) { V[slotV] = 2u; ITS[slotV] = -2; }
    return;
  }
  // The soft values through the code's bit map (INV: codeword bit j's slot) and the whitening by slot, parity
  // transposed (src/focus.c map_in); the data bits past the block's payload and CRC are known zeros.
  let first = tt.y + x * subsOf(CODE);
  let slot0 = 640u * first;
  let lBase = f * lStride + 160u * first;
  for (var wd: u32 = i; wd < N / 2u; wd = wd + ${TH}u) {
    var packed: u32 = 0u;
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let d = 2u * wd + h;
      var j = d;
      if (d >= K) { let t = d - K; j = K + MB * (t % Z) + t / Z; }
      var vv: i32 = 127;
      if (j < KNOWN || j >= K) {
        let slot = (TAB[INV_AT + (j >> 1u)] >> (16u * (j & 1u))) & 0xffffu;
        let w = Lin[lBase + (slot >> 2u)];
        vv = bitcast<i32>(w << (8u * (3u - (slot & 3u)))) >> 24u;
        let ws = slot0 + slot;
        if (((TAB[WHITE_AT + (ws >> 5u)] >> (ws & 31u)) & 1u) != 0u) { vv = -vv; }
      }
      packed = packed | ((u32(vv) & 0xffffu) << (16u * h));
    }
    atomicStore(&Lw[wd], packed);
  }
  for (var q: u32 = i; q < MB * Z; q = q + ${TH}u) { Rw[q] = ${one ? "0u" : "vec2<u32>(0u, 0u)"}; }
  if (i == 0u) { atomicStore(&badc, 0u); flag = 0u; }
  workgroupBarrier();

  var first0: u32 = 0u;${net ? `
  var low: u32 = 0xffffffffu;
  var prev: u32 = 0u;
  let est = bitcast<f32>(BLK[slotV].x);` : ``}
  var it: u32 = 0u;
  var exitCode: u32 = 0u;
  for (it = 0u; it < MAXIT; it = it + 1u) {
    for (var step: u32 = 0u; step < MB; step = step + 1u) {
      var r = step;
      if ((it & 1u) != 0u) { r = MB - 1u - step; }
      if (i < Z) { rowUpdate(r, i); }
      workgroupBarrier();
    }
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
        if (it == 0u) { first0 = bad; prev = bad; }
        low = min(low, bad);
        netX[0] = f32(bad); netX[1] = f32(low); netX[2] = f32(first0); netX[3] = f32(prev); netX[4] = f32(it + 1u); netX[5] = est;
        prev = bad;
      }
      flag = d;
    }
    let d = workgroupUniformLoad(&flag);
    if (d != 0u) { exitCode = d; break; }
    // ai: the learned stop after iterations 1 to 29 (the cap ends 30): a hidden unit a lane (back_ldpc.mjs ldpcSource)
    if (it + 1u < MAXIT) {
      if (i < NET_H) {
        var a: f32 = netW(6u * NET_H + i);
        for (var k: u32 = 0u; k < 6u; k++) { a = a + netW(k * NET_H + i) * netX[k]; }
        netH[i] = max(a, 0.0);
      }
      workgroupBarrier();
      if (i < NET_H) {
        var a: f32 = netW((7u + NET_H) * NET_H + i);
        for (var k: u32 = 0u; k < NET_H; k++) { a = a + netW((7u + k) * NET_H + i) * netH[k]; }
        netP[i] = netW((8u + NET_H) * NET_H + i) * max(a, 0.0);
      }
      workgroupBarrier();
      if (i == 0u) {
        var sum: f32 = netP[0];
        for (var k: u32 = 1u; k < NET_H; k++) { sum = sum + netP[k]; }
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
      else if (it == 0u) { first0 = bad; }
      else if (it == STALL_IT - 1u && f32(bad) > STALL_RATIO * f32(first0)) { d = 2u; }
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
      if (n < T.dims.w) { sl = n + 1u; atomicAdd(&BCOUNTS[f * 8u + 5u], 1u); }
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
