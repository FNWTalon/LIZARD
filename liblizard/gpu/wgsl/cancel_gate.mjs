// ai: The cancel stage's plan and its bookkeeping (gpu/back/DESIGN.md section 13): three shaders around the paint,
// ai: the bases and the fit. GATE2 (one workgroup, 256 threads, phases with barriers) reads what pass one left
// ai: (REC, BCOUNTS, LISTS) and writes the plan pass two runs by: KEYTAB, the batch's frame keys (id - block) with
// ai: the verified blocks each key holds; PTR, the record a (key, block) sits at; FR, each frame's references; the
// ai: paint slots the used keys get; CANCEL's headers; LISTS2 and ARGS2, the short frames with references by size,
// ai: laid out exactly as the first gate lays LISTS and ARGS (a frame's blocks and rows copied from LISTS, pass 2's
// ai: workgroups the most rows a listed frame of the size needs). ZERO clears the disc of every slot the paint will
// ai: fill (the disc lives in S's frame slots, which pass one wrote). CARRY copies the painted reference with the
// ai: greatest key into the shared CARRY buffer for the next batch's first short frame.
// ai:
// ai: PLAN's layout in words (planLayout): KEYTAB [128] { key, bits[4], slot, count, tag }, PTR [128][blocksMax]
// ai: (record index + 1, 0 none), FR [B] { fb, fa, slotA, slotB, nrefs, short, T, keyOwn }, CARRY_SEL { slot, key,
// ai: count, size }, SLOTS [R + 1] (the KEYTAB entry painted in slot r, NONE unused; SLOTS[R] is 1 when the carry
// ai: slot is a reference this batch), KVER [128] (a KEYTAB entry's version: the fewest blocks a frame holding one of
// ai: its records was listed with, LISTS' block count, NONE none; the paint's clip level). An empty KEYTAB entry has
// ai: key NONE (0xffffffff): the one key value no frame may have. tag's low 8 bits are the sizes whose records
// ai: touched the entry (a bit a size index, so at most 8 picture slots), bit 8 USED.
// ai: A key touched by two sizes is never a reference: a batch mixing formats loses that key, nothing else.
// ai: CANCEL [B] { slotA, slotB, nrefs, ok, coef[18], pad[2] } (24 words): the gate writes the four header words
// ai: (ok 0, slotB = slotA when nrefs is 1), the solve the coefficients and ok.
// ai: The hash is (key / T) & 127 with T the size's blocks, open addressing, every insert and lookup probing the
// ai: same way, so a key inserted under one size's T is found under that T only, which is the size it belongs to.
// ai: A painted frame's ids run over its own blocks (its version, LISTS' block count), so the key after a frame's is
// ai: its key plus that count, and FR's T holds it.
import { ARGS_WORDS, SLOTS, listsHead, listsRows } from "./back_transform.mjs";

export const NONE = 0xffffffff;
export const KEYTAB_ENTRIES = 128;
export const KEYTAB_WORDS = 8;
export const FR_WORDS = 8;
export const CANCEL_WORDS = 24;
export const CARRY_HEADER = 256;   // ai: bytes: { valid, key, count, size } then padding, the slot's bytes after it

// ai: Word offsets of PLAN's parts and its length, from the chain's derived dims.
export function planLayout({ blocksMax, B, refSlots }) {
  const keytab = 0, ptr = KEYTAB_ENTRIES * KEYTAB_WORDS, fr = ptr + KEYTAB_ENTRIES * blocksMax, sel = fr + FR_WORDS * B, slots = sel + 4;
  const nslots = 4 * Math.ceil((refSlots + 1) / 4), kver = slots + nslots;
  return { keytab, ptr, fr, sel, slots, nslots, kver, words: kver + KEYTAB_ENTRIES };
}

export function gate2Source({ B, blocksMax, refSlots, recCap }) {
  if (SLOTS > 8) throw new Error(`${SLOTS} picture slots: KEYTAB's tag holds 8 size bits`);
  const L = planLayout({ blocksMax, B, refSlots });
  return /* wgsl */ `
struct Rec { count: u32, words: array<u32> }
struct Gate { at: array<vec4u, ${SLOTS}> }
struct Carry { valid: u32, key: u32, count: u32, size: u32 }
struct Entry { key: atomic<u32>, bits: array<atomic<u32>, 4>, slot: u32, count: atomic<u32>, tag: atomic<u32> }
struct Fr { fb: u32, fa: u32, slotA: u32, slotB: u32, nrefs: u32, short: u32, T: u32, keyOwn: u32 }
struct Plan { keytab: array<Entry, ${KEYTAB_ENTRIES}>, ptr: array<atomic<u32>, ${KEYTAB_ENTRIES * blocksMax}>, fr: array<Fr, ${B}>, sel: vec4u, slots: array<u32, ${L.nslots}>, kver: array<atomic<u32>, ${KEYTAB_ENTRIES}> }
@group(0) @binding(0) var<storage, read> REC: Rec;
@group(0) @binding(1) var<storage, read> BCOUNTS: array<u32>;
@group(0) @binding(2) var<storage, read> LISTS: array<u32>;
@group(0) @binding(3) var<storage, read_write> PLAN: Plan;
@group(0) @binding(4) var<storage, read_write> LISTS2: array<u32>;
@group(0) @binding(5) var<storage, read_write> ARGS2: array<u32>;
@group(0) @binding(6) var<storage, read_write> BCOUNTS2: array<u32>;
@group(0) @binding(7) var<storage, read_write> CANCEL: array<u32>;
@group(0) @binding(8) var<uniform> G: Gate;
@group(0) @binding(9) var<uniform> CARRY: Carry;
const B: u32 = ${B}u;
const LS: u32 = ${B + 1}u;
const SLOTS: u32 = ${SLOTS}u;
const HEAD: u32 = ${listsHead(B)}u;
const BM: u32 = ${blocksMax}u;
const R: u32 = ${refSlots}u;
const NE: u32 = ${KEYTAB_ENTRIES}u;
const NONE: u32 = 0xffffffffu;
const RECW: u32 = 121u;
const ROWS_AT: u32 = ${listsRows(B)}u;
const CAP: u32 = ${recCap}u;
const CW: u32 = ${CANCEL_WORDS}u;
const USED: u32 = 256u;
const TH: u32 = 256u;
var<workgroup> keyMin: array<atomic<u32>, B>;
var<workgroup> keyMax: array<atomic<u32>, B>;
var<workgroup> sizeOf: array<u32, B>;
var<workgroup> nrefsOf: array<u32, B>;
var<workgroup> usedCount: atomic<u32>;
// ai: the most blocks any frame of each size verified in pass one: what this stream reads when a frame is not straddled
var<workgroup> best: array<atomic<u32>, SLOTS>;
var<workgroup> carryUsed: atomic<u32>;

fn hashOf(key: u32, T: u32) -> u32 { return (key / max(T, 1u)) & (NE - 1u); }
// ai: The entry holding key, or NE: an empty entry ends the probe, as it ended the insert's.
fn findKey(key: u32, T: u32) -> u32 {
  var h: u32 = hashOf(key, T);
  for (var n: u32 = 0u; n < NE; n = n + 1u) {
    let k = atomicLoad(&PLAN.keytab[h].key);
    if (k == key) { return h; }
    if (k == NONE) { return NE; }
    h = (h + 1u) & (NE - 1u);
  }
  return NE;
}
// ai: Insert key (or find it), returning its entry, or NE when the table is full. A weak exchange that fails on an
// ai: empty entry is retried on the same entry.
fn insertKey(key: u32, T: u32) -> u32 {
  var h: u32 = hashOf(key, T);
  for (var n: u32 = 0u; n < 4u * NE; n = n + 1u) {
    let r = atomicCompareExchangeWeak(&PLAN.keytab[h].key, NONE, key);
    if (r.exchanged || r.old_value == key) { return h; }
    if (r.old_value != NONE) { h = (h + 1u) & (NE - 1u); }
  }
  return NE;
}
// ai: Is key a paintable reference for size s: in the table, touched by that size alone.
fn known(key: u32, T: u32, s: u32) -> bool {
  let e = findKey(key, T);
  if (e >= NE) { return false; }
  return (atomicLoad(&PLAN.keytab[e].tag) & 0xffu) == (1u << s);
}
fn carryIs(key: u32, s: u32) -> bool { return CARRY.valid != 0u && CARRY.size == s && CARRY.key == key; }
// ai: The paint slot of a reference key, R for the carry, NONE when it has none.
fn slotOf(key: u32, T: u32, s: u32) -> u32 {
  if (key == NONE) { return NONE; }
  let e = findKey(key, T);
  if (e < NE) {
    if ((atomicLoad(&PLAN.keytab[e].tag) & 0xffu) == (1u << s)) { return PLAN.keytab[e].slot; }
    return NONE;
  }
  if (carryIs(key, s)) { return R; }
  return NONE;
}
fn sync() { storageBarrier(); workgroupBarrier(); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  // ai: Phase 1: everything the plan holds starts empty; LISTS2 carries the frames' (w, h), block counts and row counts
  // ai: as LISTS does.
  if (t < NE) {
    atomicStore(&PLAN.keytab[t].key, NONE);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) { atomicStore(&PLAN.keytab[t].bits[k], 0u); }
    PLAN.keytab[t].slot = NONE;
    atomicStore(&PLAN.keytab[t].count, 0u);
    atomicStore(&PLAN.keytab[t].tag, 0u);
    atomicStore(&PLAN.kver[t], NONE);
  }
  for (var i: u32 = t; i < NE * BM; i += TH) { atomicStore(&PLAN.ptr[i], 0u); }
  if (t < B) {
    PLAN.fr[t] = Fr(NONE, NONE, NONE, NONE, 0u, 0u, 0u, NONE);
    atomicStore(&keyMin[t], NONE);
    atomicStore(&keyMax[t], 0u);
    sizeOf[t] = NONE;
    nrefsOf[t] = 0u;
  }
  if (t == 0u) { PLAN.sel = vec4u(NONE, 0u, 0u, 0u); atomicStore(&usedCount, 0u); atomicStore(&carryUsed, 0u); }
  if (t < ${L.nslots}u) { PLAN.slots[t] = NONE; }
  for (var i: u32 = t; i < 8u * B; i += TH) { BCOUNTS2[i] = 0u; }
  for (var i: u32 = t; i < CW * B; i += TH) { CANCEL[i] = 0u; }
  if (t < SLOTS) { LISTS2[t * LS] = 0u; atomicStore(&best[t], 0u); }
  // ai: The frames' (w, h), blocks, rows and rings, as the first gate wrote them (back_transform.mjs listsWords).
  for (var i: u32 = t; i < 5u * B; i += TH) { LISTS2[HEAD + i] = LISTS[HEAD + i]; }
  sync();
  // ai: Each listed frame's size, from the first gate's lists.
  if (t < SLOTS) {
    let c = LISTS[t * LS];
    for (var k: u32 = 0u; k < c && k < B; k = k + 1u) { sizeOf[LISTS[t * LS + 1u + k]] = t; }
  }
  sync();
  if (t < B && sizeOf[t] != NONE) { atomicMax(&best[sizeOf[t]], BCOUNTS[t * 8u + 4u]); }
  // ai: Phase 2: the keys of every record into the table, then, once every key is in, the bits, the record
  // ai: pointers, the counts and each frame's key range.
  let nrec = min(REC.count, CAP);
  for (var rec: u32 = t; rec < nrec; rec += TH) {
    let w0 = REC.words[rec * RECW];
    let f = w0 & 0xffffu;
    let b = w0 >> 16u;
    if (f >= B || b >= BM) { continue; }
    let s = sizeOf[f];
    if (s == NONE) { continue; }
    let key = REC.words[rec * RECW + 2u] - b;
    if (key == NONE) { continue; }
    let e = insertKey(key, G.at[s].z);
  }
  sync();
  for (var rec: u32 = t; rec < nrec; rec += TH) {
    let w0 = REC.words[rec * RECW];
    let f = w0 & 0xffffu;
    let b = w0 >> 16u;
    if (f >= B || b >= BM) { continue; }
    let s = sizeOf[f];
    if (s == NONE) { continue; }
    let key = REC.words[rec * RECW + 2u] - b;
    if (key == NONE) { continue; }
    let e = findKey(key, G.at[s].z);
    if (e >= NE) { continue; }
    atomicOr(&PLAN.keytab[e].tag, 1u << s);
    atomicOr(&PLAN.keytab[e].bits[b >> 5u], 1u << (b & 31u));
    atomicMax(&PLAN.ptr[e * BM + b], rec + 1u);
    atomicAdd(&PLAN.keytab[e].count, 1u);
    atomicMin(&PLAN.kver[e], LISTS[HEAD + 2u * B + f]);
    atomicMin(&keyMin[f], key);
    atomicMax(&keyMax[f], key);
  }
  sync();
  // ai: Phase 3: a thread a frame decides whether the frame is short and which keys it would cancel against, and
  // ai: marks those keys used (the carry apart).
  var fb: u32 = NONE;
  var fa: u32 = NONE;
  var T: u32 = 0u;
  var s: u32 = NONE;
  if (t < B) {
    s = sizeOf[t];
    var short: u32 = 0u;
    var own: u32 = NONE;
    if (s != NONE) {
      T = LISTS[HEAD + 2u * B + t];
      let verified = BCOUNTS[t * 8u + 4u];
      let hasOwn = atomicLoad(&keyMin[t]) != NONE;
      if (hasOwn) { own = atomicLoad(&keyMin[t]); }
      // ai: Short is under half of what the batch's best frame of its size read, so a straddled frame is caught while a
      // ai: format too dense for the crop is not: there every frame reads about half its blocks (06-25-58, LIZARD-352 at
      // ai: a 1080 crop: 48 of 48 frames cancelled for 6 blocks at 6.6 ms a frame when half meant half of T). A batch that
      // ai: read nothing anywhere keeps its frames short, and without references (the carry at most) costs nothing.
      let bestOf = atomicLoad(&best[s]);
      if (verified * 2u < bestOf || (verified == 0u && bestOf == 0u)) {
        short = 1u;
        if (hasOwn) {
          fb = atomicLoad(&keyMin[t]);
          fa = atomicLoad(&keyMax[t]);
        } else {
          for (var g: i32 = i32(t) - 1; g >= 0; g = g - 1) {
            if (atomicLoad(&keyMin[u32(g)]) != NONE && sizeOf[u32(g)] == s) { fb = atomicLoad(&keyMax[u32(g)]); break; }
          }
          if (fb == NONE && CARRY.valid != 0u && CARRY.size == s) { fb = CARRY.key; }
          for (var g: u32 = t + 1u; g < B; g = g + 1u) {
            if (atomicLoad(&keyMin[g]) != NONE && sizeOf[g] == s) { fa = atomicLoad(&keyMin[g]); break; }
          }
        }
        // ai: One key both ways: the picture after it if the batch holds it, else the one before.
        if (fb != NONE && fa != NONE && fb == fa) {
          if (known(fb + T, G.at[s].z, s)) { fa = fb + T; } else { fb = fa - T; }
        }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
          let key = select(fa, fb, k == 0u);
          if (key == NONE) { continue; }
          let e = findKey(key, G.at[s].z);
          if (e < NE) {
            if ((atomicLoad(&PLAN.keytab[e].tag) & 0xffu) == (1u << s)) { atomicOr(&PLAN.keytab[e].tag, USED); }
          } else if (carryIs(key, s)) { atomicStore(&carryUsed, 1u); }
        }
      }
    }
    PLAN.fr[t] = Fr(fb, fa, NONE, NONE, 0u, short, T, own);
  }
  sync();
  // ai: Phase 4: used keys take paint slots in key order, the first R of them; the greatest slotted key is the
  // ai: carry out. Two entries can hold one key (a batch mixing sizes hashes it under each size's T), so ties break
  // ai: on the entry index: every rank is one entry's, and no two entries paint one slot.
  if (t < NE) {
    let key = atomicLoad(&PLAN.keytab[t].key);
    let used = key != NONE && (atomicLoad(&PLAN.keytab[t].tag) & USED) != 0u;
    if (used) {
      atomicAdd(&usedCount, 1u);
      var rank: u32 = 0u;
      for (var j: u32 = 0u; j < NE; j = j + 1u) {
        let kj = atomicLoad(&PLAN.keytab[j].key);
        if (kj != NONE && (kj < key || (kj == key && j < t)) && (atomicLoad(&PLAN.keytab[j].tag) & USED) != 0u) { rank = rank + 1u; }
      }
      if (rank < R) { PLAN.keytab[t].slot = rank; PLAN.slots[rank] = t; }
    }
  }
  sync();
  if (t == 0u) {
    let n = min(atomicLoad(&usedCount), R);
    if (n > 0u) {
      for (var e: u32 = 0u; e < NE; e = e + 1u) {
        if (PLAN.keytab[e].slot == n - 1u) {
          PLAN.sel = vec4u(n - 1u, atomicLoad(&PLAN.keytab[e].key), atomicLoad(&PLAN.keytab[e].count), firstTrailingBit(atomicLoad(&PLAN.keytab[e].tag) & 0xffu));
        }
      }
    }
    if (atomicLoad(&carryUsed) != 0u) { PLAN.slots[R] = 1u; }
  }
  sync();
  // ai: Phase 5: each frame's references resolve to slots (a key that got none is dropped), then the headers.
  if (t < B) {
    var slotA: u32 = NONE;
    var slotB: u32 = NONE;
    var nrefs: u32 = 0u;
    if (s != NONE) {
      let sA = slotOf(fb, G.at[s].z, s);
      let sB = slotOf(fa, G.at[s].z, s);
      if (sA != NONE) { slotA = sA; nrefs = 1u; }
      if (sB != NONE) {
        if (nrefs == 0u) { slotA = sB; } else { slotB = sB; }
        nrefs = nrefs + 1u;
      }
    }
    if (nrefs == 1u) { slotB = slotA; }
    PLAN.fr[t].slotA = slotA;
    PLAN.fr[t].slotB = slotB;
    PLAN.fr[t].nrefs = nrefs;
    nrefsOf[t] = nrefs;
    CANCEL[t * CW] = slotA;
    CANCEL[t * CW + 1u] = slotB;
    CANCEL[t * CW + 2u] = nrefs;
    CANCEL[t * CW + 3u] = 0u;
  }
  sync();
  // ai: The second lists and args, as the first gate writes them, over the frames with references.
  if (t != 0u) { return; }
  for (var sz: u32 = 0u; sz < SLOTS; sz = sz + 1u) { ARGS2[sz * ${ARGS_WORDS}u + 6u] = 0u; }
  for (var f: u32 = 0u; f < B; f = f + 1u) {
    if (nrefsOf[f] == 0u) { continue; }
    let sz = sizeOf[f];
    let c = LISTS2[sz * LS];
    LISTS2[sz * LS + 1u + c] = f;
    LISTS2[sz * LS] = c + 1u;
    ARGS2[sz * ${ARGS_WORDS}u + 6u] = max(ARGS2[sz * ${ARGS_WORDS}u + 6u], LISTS[ROWS_AT + f]);
  }
  for (var sz: u32 = 0u; sz < SLOTS; sz = sz + 1u) {
    let c = LISTS2[sz * LS];
    let a = sz * ${ARGS_WORDS}u;
    ARGS2[a] = G.at[sz].x; ARGS2[a + 1u] = 1u; ARGS2[a + 2u] = c;
    ARGS2[a + 3u] = 1u; ARGS2[a + 4u] = 1u; ARGS2[a + 5u] = c;
    ARGS2[a + 7u] = 1u; ARGS2[a + 8u] = c;
    ARGS2[a + 9u] = G.at[sz].z; ARGS2[a + 10u] = 1u; ARGS2[a + 11u] = c;
  }
}
`;
}

// ai: ZERO: the disc of every used paint slot r < R (S's frame slot r, sVec4 16-byte units a frame) to zero before
// ai: the paint, which writes only the coefficients of known blocks. Dispatch (ceil(sVec4 / 256), 1, R).
export function zeroSource({ B, blocksMax, refSlots, sVec4 }) {
  const L = planLayout({ blocksMax, B, refSlots });
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> PLAN: array<u32>;
@group(0) @binding(1) var<storage, read_write> S: array<vec4u>;
const NONE: u32 = 0xffffffffu;
const SV: u32 = ${sVec4}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let r = wg.z;
  if (PLAN[${L.slots}u + r] == NONE) { return; }
  let i = wg.x * 256u + t;
  if (i < SV) { S[r * SV + i] = vec4u(0u); }
}
`;
}

// ai: CARRY: the reference slot CARRY_SEL names, bases and means, copied into the shared CARRY buffer behind its
// ai: header { valid, key, count, size }; nothing when no slot was painted. Dispatch (ceil(refVec4 / 4096), 1, 1):
// ai: a workgroup copies 4096 16-byte units, 16 a thread.
export function carrySource({ B, blocksMax, refSlots, refVec4 }) {
  const L = planLayout({ blocksMax, B, refSlots });
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> PLAN: array<u32>;
@group(0) @binding(1) var<storage, read> REF: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> CARRY: array<vec4u>;
const NONE: u32 = 0xffffffffu;
const RV: u32 = ${refVec4}u;
const HV: u32 = ${CARRY_HEADER / 16}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let slot = PLAN[${L.sel}u];
  if (slot == NONE) { return; }
  if (wg.x == 0u && t == 0u) { CARRY[0] = vec4u(1u, PLAN[${L.sel + 1}u], PLAN[${L.sel + 2}u], PLAN[${L.sel + 3}u]); }
  for (var k: u32 = 0u; k < 16u; k = k + 1u) {
    let i = (wg.x * 16u + k) * 256u + t;
    if (i < RV) { CARRY[HV + i] = REF[slot * RV + i]; }
  }
}
`;
}
