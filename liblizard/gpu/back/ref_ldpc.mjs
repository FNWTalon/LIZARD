// Part C's reference: src/ldpc.c ldpc_decode_stall and the block tail of focus.c focus_finish_bits (the decline
// gate, map_in, the LDPC, the packing and the CRC) in JS, integer for integer. It is held to the C through
// Focus.rxLdpc (verdict and iterations a block) and the GPU is held to it: a GPU verdict that differs from the C's
// is first run through this to say which side moved. Also here: the CRC-32 split the shader uses (one lane a
// 4-byte chunk, combined through a table of x^(8 k) mod P), checked against the plain crc32 on random messages.
import { crc32, mapTable } from "./bitmap.mjs";
import { loadStop, gives } from "./stop/rule.mjs";

// The code itself is never restated here: every function takes the wasm's Focus.ldpcCode().code.
export const PAYLOAD = 473;                   // bytes a block, the id first
export const STALL_RATIO = Math.fround(0.95);
export const STALL_IT = 9;

// ai: The stop rules (DESIGN section 6): a rule is { name, stall, funnel, net }, what gpu/wgsl/back_ldpc.mjs builds
// ai: the kernel from and what the decoders here apply after each iteration, a clear syndrome first.
// ai: - "learned", the first read's since 2026-09-28: after iterations 1 to 29, the MLP of stop/ (net, from
// ai:   stop/stop.safetensors) on (bad, low, first, prev, t, est) gives up when its score is under its threshold. It
// ai:   replaced the funnel (the C's stall test and seven checkpoints of the running minimum, 2026-09-24 to 28), which
// ai:   gave up 4 verified blocks of the three dumps where the rule gives up none (scripts/gpu/back/stop/README.md).
// ai: - "c", the C's rule: the stall test (after iteration 9, the count above 0.95 of the count after iteration 1) and
// ai:   the cap of 30. The check that the GPU's min-sum arithmetic is the C's: under it the GPU gives the C's verdict and
// ai:   iterations on every block (scripts/gpu/back/test_ldpc.mjs STOP=c, a page's ?ldpcstop=c).
// ai: - "funnel2", pass two's (the cancel stage's second LDPC dispatch, gpu/back/ldpc.mjs), not a page's to choose:
// ai:   the stall test and checkpoints (iteration j, fraction of m) of the running minimum, set on the codewords pass
// ai:   two tries: 576 frames of the two straddle-rich runs traced on the C's grids (15,494 codewords, 1,822 gained). A
// ai:   cancelled frame's codewords that go on to clear keep a lower running minimum than pass one's population
// ai:   (highest 384, 369, 331, 264, 243, 119 after iterations 1, 2, 4, 12, 15, 20), while the ones that never clear
// ai:   sit near 0.3 m throughout; each count is 0.02 m (26 or 27 checks) above that highest value. The cap stays 30:
// ai:   gained blocks clear as late as iteration 30 (DESIGN 13.6).
export const FIRST_READ = ["learned", "c"];
export const DEFAULT_STOP = "learned";
export const DEFAULT_STOP2 = "funnel2";
const FUNNEL2 = [[1, 0.322], [2, 0.311], [4, 0.281], [12, 0.228], [15, 0.212], [20, 0.114]];
export const C_RULE = Object.freeze({ name: "c", stall: true, funnel: [], net: null });
// ai: The rule a name gives for a code with m checks (a checkpoint's count is ceil(fraction m)); "learned" loads its
// ai: weights file (stop/rule.mjs loadStop, once).
export async function stopRule(name, m) {
  if (name === "c") return C_RULE;
  if (name === "learned") return { name, stall: false, funnel: [], net: await loadStop() };
  if (name === "funnel2") return { name, stall: true, funnel: FUNNEL2.map(([j, f]) => [j, Math.ceil(f * m)]), net: null };
  throw new Error(`ldpc stop rule ${name}: one of ${[...FIRST_READ, DEFAULT_STOP2].join(", ")}`);
}
// ai: The rule's give-up after iteration t (1-based) with the violated counts bad (now), low (the running minimum),
// ai: first (after iteration 1), prev (after iteration t - 1; bad at t = 1), the block's estimate est.
function givesUp(rule, t, bad, low, first, prev, est, maxIter) {
  if (rule.stall && t === STALL_IT && Math.fround(bad) > Math.fround(STALL_RATIO * first)) return true;
  for (const [j, c] of rule.funnel) if (t === j && low >= c) return true;
  return !!rule.net && t < maxIter && gives(rule.net, [bad, low, first, prev, t, est]);
}
// What the decoder reports for a codeword from its trace (makeTracer: the violated checks after each iteration, run
// to a clear syndrome or the cap with no stop rule) under a rule, est its block's estimate: iterations to a clear
// syndrome, -3 given up, -1 the cap.
export function stopAt(trace, used, rule = C_RULE, est = 0, { maxIter = 30 } = {}) {
  let low = Infinity, prev = trace[0];
  for (let it = 0; it < maxIter; it++) {
    if (used === it + 1) return used;
    const bad = trace[it];
    low = Math.min(low, bad);
    if (givesUp(rule, it + 1, bad, low, trace[0], prev, est, maxIter)) return -3;
    prev = bad;
  }
  return -1;
}

// lay: [mb + 1] row pointers then (col, shift) pairs (Focus.ldpcCode().code.lay). Every block row of the
// 3/4 code has exactly nd data slots; the shader is written to that and this checks it.
export function rowTable(code) {
  const { mb, lay } = code, nd = lay[1] - lay[0], rows = [];
  for (let r = 0; r < mb; r++) {
    if (lay[r + 1] - lay[r] !== nd) throw new Error(`block row ${r} has ${lay[r + 1] - lay[r]} data slots, not ${nd}`);
    const row = [];
    for (let e = lay[r]; e < lay[r + 1]; e++) row.push([lay[mb + 1 + 2 * e], lay[mb + 2 + 2 * e]]);
    rows.push(row);
  }
  return { nd, rows };
}

// Slot e of check (r, i) as an index into L laid out as the C keeps it: data bit v at v, parity bit k + chk at
// k + (chk % mb) z + chk / mb. -1 for the absent staircase bit of check 0.
function slotIndex(code, rows, r, i, e) {
  const { k, z, mb } = code, nd = rows[0].length;
  if (e < nd) { const [col, s] = rows[r][e]; return col * z + (i - s + z) % z; }
  if (e === nd) return r > 0 ? k + (r - 1) * z + i : i > 0 ? k + (mb - 1) * z + i - 1 : -1;
  return k + r * z + i;
}

// ldpc_decode_stall on llr in codeword order (Int8Array n) under a stop rule (stopRule; the C's by default), est the
// block's estimate. Returns { used, bits } with bits in codeword order.
export function decodeStall(code, llr, { maxIter = 30, rule = C_RULE, est = 0, trace = null } = {}) {
  const { n, k, m, z, mb, norm } = code, { nd, rows } = rowTable(code), ns = nd + 2;
  const L = new Int32Array(n), R = new Int8Array(mb * z * ns);
  for (let v = 0; v < k; v++) L[v] = llr[v];
  for (let chk = 0; chk < m; chk++) L[k + (chk % mb) * z + Math.floor(chk / mb)] = llr[k + chk];
  const unsat = () => {
    let bad = 0;
    for (let r = 0; r < mb; r++) for (let i = 0; i < z; i++) {
      let p = 0;
      for (let e = 0; e < ns; e++) { const idx = slotIndex(code, rows, r, i, e); if (idx >= 0) p ^= L[idx] < 0 ? 1 : 0; }
      bad += p;
    }
    return bad;
  };
  let used = -1, first = 0, low = Infinity, prev = 0;
  for (let it = 0; it < maxIter; it++) {
    for (let step = 0; step < mb; step++) {
      const r = it & 1 ? mb - 1 - step : step;
      for (let i = 0; i < z; i++) {
        const rb = (r * z + i) * ns;
        let min1 = 32767, min2 = 32767, arg = -1, sign = 0;
        const q = new Int32Array(ns);
        for (let e = 0; e < ns; e++) {
          const idx = slotIndex(code, rows, r, i, e);
          if (idx < 0) { q[e] = 8191; continue; }   // the certain zero: it never wins a minimum that survives the cap
          q[e] = L[idx] - R[rb + e];
          const a = q[e] < 0 ? -q[e] : q[e];
          sign ^= q[e] < 0 ? 1 : 0;
          if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) min2 = a;
        }
        min1 = Math.min(127, (min1 * norm) >> 4); min2 = Math.min(127, (min2 * norm) >> 4);
        for (let e = 0; e < ns; e++) {
          const idx = slotIndex(code, rows, r, i, e);
          if (idx < 0) continue;
          const mag = e === arg ? min2 : min1, rr = (sign ^ (q[e] < 0 ? 1 : 0)) ? -mag : mag;
          R[rb + e] = rr;
          const v = q[e] + rr;
          L[idx] = v > 8191 ? 8191 : v < -8191 ? -8191 : v;
        }
      }
    }
    const bad = unsat();
    if (trace) trace.push(bad);
    if (bad === 0) { used = it + 1; break; }
    if (it === 0) first = prev = bad;
    low = Math.min(low, bad);
    if (givesUp(rule, it + 1, bad, low, first, prev, est, maxIter)) { used = -3; break; }
    prev = bad;
  }
  const bits = new Uint8Array(n);
  for (let v = 0; v < k; v++) bits[v] = L[v] < 0 ? 1 : 0;
  for (let chk = 0; chk < m; chk++) bits[k + chk] = L[k + (chk % mb) * z + Math.floor(chk / mb)] < 0 ? 1 : 0;
  return { used, bits };
}

// The same decoder with its slot indices tabled once, for measuring stop rules (scripts/exp/ldpc_stop.mjs): run to the
// cap with no stall rule, keeping the count of violated checks after every iteration. A rule that reads only that
// count and the block's estimate changes nothing before it fires, so one trace a codeword evaluates any such rule
// exactly. run(llr in codeword order) returns { trace (the count after iteration t at t - 1, up to the exit), used
// (iterations to a clear syndrome, 0 if none), bits (codeword order) }.
export function makeTracer(code, { maxIter = 30 } = {}) {
  const { n, k, m, z, mb, norm } = code, { nd, rows } = rowTable(code), ns = nd + 2, nck = mb * z;
  const IDX = new Int32Array(nck * ns);
  for (let r = 0; r < mb; r++) for (let i = 0; i < z; i++) for (let e = 0; e < ns; e++) IDX[(r * z + i) * ns + e] = slotIndex(code, rows, r, i, e);
  const L = new Int32Array(n), R = new Int32Array(nck * ns), q = new Int32Array(ns), bits = new Uint8Array(n);
  const unsat = () => {
    let bad = 0;
    for (let c = 0; c < nck; c++) {
      let p = 0;
      for (let e = c * ns, end = e + ns; e < end; e++) { const idx = IDX[e]; if (idx >= 0 && L[idx] < 0) p ^= 1; }
      bad += p;
    }
    return bad;
  };
  return (llr) => {
    for (let v = 0; v < k; v++) L[v] = llr[v];
    for (let chk = 0; chk < m; chk++) L[k + (chk % mb) * z + Math.floor(chk / mb)] = llr[k + chk];
    R.fill(0);
    const trace = new Uint16Array(maxIter);
    let used = 0, ran = 0;
    for (let it = 0; it < maxIter; it++) {
      for (let step = 0; step < mb; step++) {
        const r = it & 1 ? mb - 1 - step : step;
        for (let i = 0; i < z; i++) {
          const b0 = (r * z + i) * ns;
          let min1 = 32767, min2 = 32767, arg = -1, sign = 0;
          for (let e = 0; e < ns; e++) {
            const idx = IDX[b0 + e];
            if (idx < 0) continue;
            const qe = L[idx] - R[b0 + e], a = qe < 0 ? -qe : qe;
            q[e] = qe;
            if (qe < 0) sign ^= 1;
            if (a < min1) { min2 = min1; min1 = a; arg = e; } else if (a < min2) min2 = a;
          }
          const m1 = Math.min(127, (min1 * norm) >> 4), m2 = Math.min(127, (min2 * norm) >> 4);
          for (let e = 0; e < ns; e++) {
            const idx = IDX[b0 + e];
            if (idx < 0) continue;
            const qe = q[e], mag = e === arg ? m2 : m1, rr = (sign ^ (qe < 0 ? 1 : 0)) ? -mag : mag, v = qe + rr;
            R[b0 + e] = rr;
            L[idx] = v > 8191 ? 8191 : v < -8191 ? -8191 : v;
          }
        }
      }
      ran = it + 1;
      const bad = unsat();
      trace[it] = bad;
      if (bad === 0) { used = it + 1; break; }
    }
    for (let v = 0; v < k; v++) bits[v] = L[v] < 0 ? 1 : 0;
    for (let chk = 0; chk < m; chk++) bits[k + chk] = L[k + (chk % mb) * z + Math.floor(chk / mb)] < 0 ? 1 : 0;
    return { trace: trace.slice(0, ran), used, bits };
  };
}

// The block tail on one frame's soft values: L int8 slot order (subch x 640), declined a block (part B's gate),
// map the MAP table for the run's bit map mode (bitmap.mjs mapTable). Returns a verdict a block (1 verified,
// 2 declined, 3 stall or cap, 4 CRC), its as the C reports it, and the payload bytes of the verified blocks.
// rule: a stop rule (stopRule; the C's by default), est the blocks' estimates (BLK.x, a Float32Array; the learned rule
// reads them). tracer: a makeTracer(code) to decode with, about 50 times faster than decodeStall and the same
// decisions (the rule applied to its trace by stopAt).
export function finishBlocks(code, L, declined, map, blocks, { trace = null, rule = C_RULE, est = null, tracer = null } = {}) {
  const { n, k } = code, verdict = new Uint8Array(blocks), its = new Int8Array(blocks), bytes = new Uint8Array(blocks * PAYLOAD);
  const cw = new Int8Array(n);
  for (let b = 0; b < blocks; b++) {
    if (declined[b]) { verdict[b] = 2; its[b] = -2; continue; }
    for (let j = 0; j < n; j++) { const e = map[b * n + j], v = L[5120 * b + (e & 0x7fff)]; cw[j] = e & 0x8000 ? -v : v; }
    let used, bits;
    const e = est ? est[b] : 0;
    if (tracer) { const r = tracer(cw); used = stopAt(r.trace, r.used, rule, e); bits = r.bits; if (trace) trace.push({ b, its: used, bad: Array.from(r.trace) }); }
    else { const t = trace ? [] : null; ({ used, bits } = decodeStall(code, cw, { trace: t, rule, est: e })); if (trace) trace.push({ b, its: used, bad: t }); }
    its[b] = used;
    if (used < 0) { verdict[b] = 3; continue; }
    const dst = bytes.subarray(b * PAYLOAD, (b + 1) * PAYLOAD);
    for (let i = 0; i < PAYLOAD * 8; i++) dst[i >> 3] |= bits[i] << (7 - (i & 7));
    let crc = 0;
    for (let i = 0; i < 32; i++) crc = ((crc << 1) | bits[PAYLOAD * 8 + i]) >>> 0;
    verdict[b] = crc === crc32(dst, PAYLOAD) ? 1 : 4;
  }
  return { verdict, its, bytes };
}

// ---- the CRC-32 split (DESIGN section 7, as built) ----
//
// The reflected bitwise step is multiplication by x in GF(2)[x] / P with x^0 at bit 31. Both halves of the
// algorithm are linear in the message, so the CRC of 473 bytes is the XOR over 4-byte chunks of (chunk from a
// zero state) times x^(8 * bytes after the chunk), plus the initial state pushed through the whole message, then
// the final inversion. The shader gives each lane one chunk, one table entry and a 32-step multiply.
const P = 0xedb88320;
export const mulx = (c) => ((c >>> 1) ^ (P & -(c & 1))) >>> 0;
export function xpow(e) { let c = 0x80000000; for (let i = 0; i < e; i++) c = mulx(c); return c >>> 0; }
// c times the polynomial p (both in the reflected representation).
export function mulPoly(c, p) { let r = 0, t = c >>> 0; for (let j = 0; j < 32; j++) { if ((p >>> (31 - j)) & 1) r ^= t; t = mulx(t); } return r >>> 0; }
export function chunkState(bytes, off, len) {
  let c = 0;
  for (let i = 0; i < len; i++) { c ^= bytes[off + i]; for (let q = 0; q < 8; q++) c = mulx(c); }
  return c >>> 0;
}
// pw[w] for chunk w (bytes 4 w .. 4 w + 3, the last one byte 472 alone); pw[119] = the init term with the final
// inversion folded in. What the shader's uniform carries.
export function crcPowers(len = PAYLOAD) {
  const words = Math.ceil(len / 4), pw = new Uint32Array(words + 1);
  for (let w = 0; w < words; w++) { const end = Math.min(4 * w + 4, len); pw[w] = xpow(8 * (len - end)); }
  pw[words] = (mulPoly(0xffffffff, xpow(8 * len)) ^ 0xffffffff) >>> 0;
  return pw;
}
export function crcSplit(bytes, len = PAYLOAD, pw = crcPowers(len)) {
  const words = Math.ceil(len / 4);
  let c = pw[words];
  for (let w = 0; w < words; w++) { const nb = Math.min(4, len - 4 * w); c ^= mulPoly(chunkState(bytes, 4 * w, nb), pw[w]); }
  return c >>> 0;
}

// Self check of the split against the plain CRC, run by the test before anything else.
export function checkCrcSplit(trials = 200) {
  const pw = crcPowers();
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed >>> 8; };
  for (let t = 0; t < trials; t++) {
    const m = new Uint8Array(PAYLOAD);
    for (let i = 0; i < PAYLOAD; i++) m[i] = rnd() & 255;
    if (t === 0) m.fill(0);
    if (t === 1) m.fill(255);
    if (crcSplit(m, PAYLOAD, pw) !== crc32(m, PAYLOAD)) return false;
  }
  for (let t = 0; t < 50; t++) { const c = rnd() >>> 0, e = rnd() % 4000; let d = c; for (let i = 0; i < e; i++) d = mulx(d); if (mulPoly(c, xpow(e)) !== d) return false; }
  return true;
}

export { mapTable };
